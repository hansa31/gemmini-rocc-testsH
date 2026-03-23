/* bert-tiny-sst2-stream.c
 *
 * Streaming evaluation of BERT-Tiny on the SST-2 dataset.
 * Reads pre-embedded int8 sequences from .bin files (produced by
 * setup_BERT/prepare_sst2.py), runs encoder + pooler + classifier,
 * and reports accuracy, F1 score, and confusion matrix.
 *
 * Prerequisites:
 *   1. Generate weights:
 *        conda run -n ImageNet python setup_BERT/generate_bert_params.py
 *   2. Prepare SST-2 data:
 *        conda run -n ImageNet python setup_BERT/prepare_sst2.py
 *      This produces:
 *        sst2_validation_872.bin           — 872 × 128 × 128 int8 embeddings
 *        sst2_validation_872_labels.txt    — 872 labels (0 or 1)
 *        sst2_validation_872_attn_masks.bin — 872 × 128 uint8 attention masks
 *
 * Build (Linux/pk — needs fopen/fread):
 *   make bert-tiny-sst2-stream-pk
 *   # or: make bert-tiny-sst2-stream-linux
 *
 * Run:
 *   spike --extension=gemmini bert-tiny-sst2-stream-pk [ws|os|cpu]
 *   Default mode is ws (Weight Stationary).
 *
 * Metrics reported:
 *   - Accuracy (primary — GLUE standard for SST-2)
 *   - Per-class precision, recall, F1
 *   - Macro-F1
 *   - Confusion matrix (2×2)
 *   - Total cycles
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

/* int8 (elem_t) weights generated from the exported .npy files */
#include "bert_params.h"

/* ── Dataset configuration ──────────────────────────────────────────────────
 * Update these after running prepare_sst2.py to match the output.
 */
#define NUM_EXAMPLES    872
#define EMBEDDINGS_BIN  "sst2_validation_872.bin"
#define LABELS_TXT      "sst2_validation_872_labels.txt"
#define ATTN_MASKS_BIN  "sst2_validation_872_attn_masks.bin"

/* ── Software fallbacks for SOFTMAX / GELU / LAYERNORM ──────────────────────
 * Identical to bert-tiny-sst2.c — these replace the hardware normalization
 * unit when HAS_NORMALIZATIONS is not defined.
 */

static inline elem_t sat_elem(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (elem_t)v;
}

/* Software row-wise softmax on elem_t matrix (in-place).
 * Masked variant: if mask != NULL, positions where mask[j]==0 are set to
 * -128 before computing softmax (prevents padding from affecting attention). */
static void sw_softmax_inplace_masked(elem_t *mat, int rows, int cols,
                                       int stride, const uint8_t *mask) {
    for (int i = 0; i < rows; i++) {
        elem_t *row = mat + i * stride;
        /* Apply attention mask: set padded positions to minimum */
        if (mask) {
            for (int j = 0; j < cols; j++) {
                if (!mask[j]) row[j] = -128;
            }
        }
        float mx = (float)row[0];
        for (int j = 1; j < cols; j++)
            if ((float)row[j] > mx) mx = (float)row[j];
        float sum = 0.f;
        for (int j = 0; j < cols; j++) {
            float e = expf((float)row[j] - mx);
            sum += e;
        }
        for (int j = 0; j < cols; j++) {
            float e = expf((float)row[j] - mx);
            int32_t q = (int32_t)(e / sum * 127.f + 0.5f);
            row[j] = sat_elem(q);
        }
    }
}

/* Non-masked variant for compatibility */
static void sw_softmax_inplace(elem_t *mat, int rows, int cols, int stride) {
    sw_softmax_inplace_masked(mat, rows, cols, stride, NULL);
}

static void sw_gelu_inplace(elem_t *mat, int rows, int cols, int stride,
                           float in_scale, float out_scale) {
    const float sqrt_2_over_pi = 0.7978845608f;
    float inv_out = 1.0f / out_scale;
    for (int i = 0; i < rows; i++) {
        elem_t *row = mat + i * stride;
        for (int j = 0; j < cols; j++) {
            float x = (float)row[j] * in_scale;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
            float g = 0.5f * x * (1.f + tanhf(inner));
            float q = g * inv_out;
            row[j] = sat_elem((int32_t)(q > 0 ? q + 0.5f : q - 0.5f));
        }
    }
}

static void sw_layernorm(const acc_t *in, elem_t *out,
                         int rows, int cols,
                         int in_stride, int out_stride,
                         const float *gamma, const float *beta) {
    for (int i = 0; i < rows; i++) {
        const acc_t *irow = in  + i * in_stride;
        elem_t      *orow = out + i * out_stride;
        double sum = 0.0;
        for (int j = 0; j < cols; j++) sum += (double)irow[j];
        double mean = sum / cols;
        double var = 0.0;
        for (int j = 0; j < cols; j++) {
            double d = (double)irow[j] - mean;
            var += d * d;
        }
        var /= cols;
        double inv_std = 1.0 / sqrt(var + 1e-12);
        for (int j = 0; j < cols; j++) {
            double normed = ((double)irow[j] - mean) * inv_std;
            double scaled = (double)gamma[j] * normed + (double)beta[j];
            orow[j] = sat_elem((int32_t)(scaled > 0 ? scaled + 0.5 : scaled - 0.5));
        }
    }
}

/* BERT post-LN: LayerNorm(acc_output * acc_to_real + int8_residual * res_to_real).
 * Converts both inputs to real values before combining, normalises,
 * and divides by out_scale to properly fill the int8 range. */
static void sw_layernorm_residual(const acc_t *acc_in, const elem_t *res_in,
                                  elem_t *out,
                                  int rows, int cols,
                                  int acc_stride, int res_stride, int out_stride,
                                  const float *gamma, const float *beta,
                                  float acc_to_real, float res_to_real,
                                  float out_scale) {
    double inv_out = 1.0 / (double)out_scale;
    for (int i = 0; i < rows; i++) {
        const acc_t  *arow = acc_in + i * acc_stride;
        const elem_t *rrow = res_in + i * res_stride;
        elem_t       *orow = out    + i * out_stride;
        double sum = 0.0;
        for (int j = 0; j < cols; j++)
            sum += (double)arow[j] * (double)acc_to_real
                 + (double)rrow[j] * (double)res_to_real;
        double mean = sum / cols;
        double var = 0.0;
        for (int j = 0; j < cols; j++) {
            double v = (double)arow[j] * (double)acc_to_real
                     + (double)rrow[j] * (double)res_to_real;
            double d = v - mean;
            var += d * d;
        }
        var /= cols;
        double inv_std = 1.0 / sqrt(var + 1e-12);
        for (int j = 0; j < cols; j++) {
            double val = (double)arow[j] * (double)acc_to_real
                       + (double)rrow[j] * (double)res_to_real;
            double normed = (val - mean) * inv_std;
            double scaled = (double)gamma[j] * normed + (double)beta[j];
            double quantized = scaled * inv_out;
            orow[j] = sat_elem((int32_t)(quantized > 0 ? quantized + 0.5 : quantized - 0.5));
        }
    }
}

/* ── BERT-Tiny architecture constants ───────────────────────────────────────── */
#define HIDDEN_DIM         128
#define EXPANSION_DIM      512
#define NUM_HEADS          2
#define SEQ_LEN            128
#define NUM_LAYERS         2
#define NUM_LABELS         2
#define COMPRESSION_FACTOR 1
#define EXAMPLE_SIZE       (SEQ_LEN * HIDDEN_DIM)

/* ── Attention sub-layer (with optional attention mask) ──────────────────────
 * Same as bert-tiny-sst2.c but softmax can use an attention mask to ignore
 * padding tokens.
 */
static void attention(
        int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,
        enum tiled_matmul_type_t tiled_matmul_type,

        const elem_t *input,  const elem_t *enc_out,
        elem_t       *out,    elem_t       *resadd_out,

        const elem_t *Wq, const elem_t *Wk,
        const elem_t *Wv, const elem_t *Wo,

        const acc_t *Wq_b, const acc_t *Wk_b,
        const acc_t *Wv_b, const acc_t *Wo_b,

        elem_t *Q_buf, elem_t *K_buf, elem_t *V_buf,
        elem_t *attn_buf, elem_t *out_buf, acc_t *out_buf_acc,
        const uint8_t *attn_mask,
        const float *ln_gamma, const float *ln_beta,
        float acc_scale_q, float acc_scale_k, float acc_scale_v,
        float acc_scale_qkt, float acc_scale_ctx,
        float wo_acc_to_real, float res_to_real, float ln_out_scale)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head   = hidden_dim_compressed / num_heads;

    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head   = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    const elem_t *qkv_weights[3] = {Wq, Wk, Wv};
    const elem_t *qkv_ins[3]     = {input, enc_out, enc_out};
    const acc_t  *qkv_bs[3]      = {Wq_b, Wk_b, Wv_b};
    elem_t       *qkv_outs[3]    = {Q_buf, K_buf, V_buf};
    const float   qkv_scales[3]  = {acc_scale_q, acc_scale_k, acc_scale_v};

    for (int i = 0; i < 3; i++) {
        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            qkv_ins[i],    qkv_weights[i],
            qkv_bs[i],     qkv_outs[i],
            hidden_dim, hidden_dim, 0, hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, qkv_scales[i], 0,
            false, false, false, false, false,
            0, tiled_matmul_type);
    }
    gemmini_fence();

    /* Attention scores per head: Q × K^T → masked softmax */
    for (int head = 0; head < num_heads; head++) {
        const elem_t *A = Q_buf    + head * hidden_dim_per_head;
        const elem_t *B = K_buf    + head * hidden_dim_per_head;
        elem_t       *C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            A, B, NULL, C,
            hidden_dim, hidden_dim, 0, seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, acc_scale_qkt, 0,
            false, false, true, false, false,
            0, tiled_matmul_type);

        gemmini_fence();
        /* Apply attention mask to softmax so padding tokens are ignored */
        sw_softmax_inplace_masked(C, seq_len, seq_len, seq_len, attn_mask);
    }
    gemmini_fence();

    /* Context vectors per head */
    for (int head = 0; head < num_heads; head++) {
        const elem_t *A = attn_buf + head * seq_len * seq_len;
        const elem_t *B = V_buf    + head * hidden_dim_per_head;
        elem_t       *C = out_buf  + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            A, B, NULL, C,
            seq_len, hidden_dim, 0, hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, acc_scale_ctx, 0,
            false, false, false, false, false,
            0, tiled_matmul_type);
    }
    gemmini_fence();

    /* Output projection */
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        out_buf, Wo, Wo_b, out_buf_acc,
        hidden_dim, hidden_dim, 0, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false, true, false,
        0, tiled_matmul_type);
    gemmini_fence();

    /* BERT post-LN: LayerNorm(Wo_output + residual) */
    sw_layernorm_residual((acc_t *)out_buf_acc, input, (elem_t *)resadd_out,
                          seq_len, hidden_dim,
                          hidden_dim, hidden_dim, hidden_dim,
                          ln_gamma, ln_beta,
                          wo_acc_to_real, res_to_real, ln_out_scale);
}

/* ── Feed-forward sub-layer ─────────────────────────────────────────────────*/
static void ffn(
        int hidden_dim, int expansion_dim, int seq_len,
        enum tiled_matmul_type_t tiled_matmul_type,
        const elem_t *input, elem_t *out,
        const elem_t *ff1_w, const elem_t *ff2_w,
        const acc_t  *ff1_b, const acc_t  *ff2_b,
        elem_t *out_buf, acc_t *out_buf_acc,
        const float *ln_gamma, const float *ln_beta,
        float acc_scale_ff1,
        float gelu_in_scale, float gelu_out_scale,
        float ff2_acc_to_real, float res_to_real, float ln_out_scale)
{
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, acc_scale_ff1, 0,
        true, false, false, false, false,
        0, tiled_matmul_type);
    gemmini_fence();
    sw_gelu_inplace(out_buf, seq_len, expansion_dim, expansion_dim,
                    gelu_in_scale, gelu_out_scale);

    gemmini_fence();
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim,
        out_buf, ff2_w, ff2_b, out_buf_acc,
        expansion_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, true, false,
        0, tiled_matmul_type);
    gemmini_fence();

    /* BERT post-LN: LayerNorm(ff2_output + residual) */
    sw_layernorm_residual((acc_t *)out_buf_acc, input, (elem_t *)out,
                          seq_len, hidden_dim,
                          hidden_dim, hidden_dim, hidden_dim,
                          ln_gamma, ln_beta,
                          ff2_acc_to_real, res_to_real, ln_out_scale);
}

/* ── Scratch buffers ────────────────────────────────────────────────────────*/
static elem_t layer_ping[SEQ_LEN][HIDDEN_DIM]   row_align(1);
static elem_t layer_pong[SEQ_LEN][HIDDEN_DIM]   row_align(1);

static elem_t attn_ln_out [SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t attn_resadd [SEQ_LEN][HIDDEN_DIM] row_align(1);

static elem_t Q_buf      [SEQ_LEN][HIDDEN_DIM]         row_align(1);
static elem_t K_buf      [SEQ_LEN][HIDDEN_DIM]         row_align(1);
static elem_t V_buf      [SEQ_LEN][HIDDEN_DIM]         row_align(1);
static elem_t attn_buf   [NUM_HEADS][SEQ_LEN][SEQ_LEN] row_align(1);
static elem_t ffn_out_buf[SEQ_LEN][EXPANSION_DIM]      row_align(1);
static acc_t  acc_buf    [SEQ_LEN][HIDDEN_DIM]          row_align_acc(1);

/* Per-example attention mask */
static uint8_t cur_attn_mask[SEQ_LEN];

/* ── Inference ─────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif

    gemmini_flush(0);

    /* ── Select execution mode ─────────────────────────────────────────── */
    enum tiled_matmul_type_t tiled_matmul_type = WS;
    if (argc >= 2) {
        if      (strcmp(argv[1], "cpu") == 0) tiled_matmul_type = CPU;
        else if (strcmp(argv[1], "os")  == 0) tiled_matmul_type = OS;
        else if (strcmp(argv[1], "ws")  == 0) tiled_matmul_type = WS;
        else {
            printf("usage: %s [ws|os|cpu]\n", argv[0]);
            exit(1);
        }
    }

    const char *mode_str = tiled_matmul_type == CPU ? "cpu"
                         : tiled_matmul_type == OS  ? "os" : "ws";

    printf("=== BERT-Tiny SST-2 Streaming Evaluation (mode=%s) ===\n", mode_str);
    printf("hidden=%d  ffn=%d  heads=%d  layers=%d  seq_len=%d\n",
           HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN);
    printf("dataset: %d examples from %s\n\n", NUM_EXAMPLES, EMBEDDINGS_BIN);

    /* ── Load labels ───────────────────────────────────────────────────── */
    int labels[NUM_EXAMPLES];
    FILE *fp_labels = fopen(LABELS_TXT, "r");
    if (!fp_labels) {
        printf("Error: cannot open %s\n", LABELS_TXT);
        exit(1);
    }
    for (int i = 0; i < NUM_EXAMPLES; i++) {
        if (fscanf(fp_labels, "%d", &labels[i]) != 1) {
            printf("Error: could not read label %d from %s\n", i, LABELS_TXT);
            fclose(fp_labels);
            exit(1);
        }
    }
    fclose(fp_labels);

    /* ── Open binary files for streaming ───────────────────────────────── */
    FILE *fp_emb = fopen(EMBEDDINGS_BIN, "rb");
    if (!fp_emb) {
        printf("Error: cannot open %s\n", EMBEDDINGS_BIN);
        exit(1);
    }

    FILE *fp_mask = fopen(ATTN_MASKS_BIN, "rb");
    if (!fp_mask) {
        printf("Error: cannot open %s\n", ATTN_MASKS_BIN);
        exit(1);
    }

    /* ── Weight pointer tables ─────────────────────────────────────────── */
    const elem_t *layer_Wq   [NUM_LAYERS] = { (elem_t*)bert_l0_Wq,    (elem_t*)bert_l1_Wq    };
    const elem_t *layer_Wk   [NUM_LAYERS] = { (elem_t*)bert_l0_Wk,    (elem_t*)bert_l1_Wk    };
    const elem_t *layer_Wv   [NUM_LAYERS] = { (elem_t*)bert_l0_Wv,    (elem_t*)bert_l1_Wv    };
    const elem_t *layer_Wo   [NUM_LAYERS] = { (elem_t*)bert_l0_Wo,    (elem_t*)bert_l1_Wo    };
    const acc_t  *layer_Wq_b [NUM_LAYERS] = { bert_l0_Wq_b,           bert_l1_Wq_b           };
    const acc_t  *layer_Wk_b [NUM_LAYERS] = { bert_l0_Wk_b,           bert_l1_Wk_b           };
    const acc_t  *layer_Wv_b [NUM_LAYERS] = { bert_l0_Wv_b,           bert_l1_Wv_b           };
    const acc_t  *layer_Wo_b [NUM_LAYERS] = { bert_l0_Wo_b,           bert_l1_Wo_b           };
    const elem_t *layer_ff1_w[NUM_LAYERS] = { bert_l0_ff1_w,          bert_l1_ff1_w          };
    const elem_t *layer_ff2_w[NUM_LAYERS] = { bert_l0_ff2_w,          bert_l1_ff2_w          };
    const acc_t  *layer_ff1_b[NUM_LAYERS] = { bert_l0_ff1_b,          bert_l1_ff1_b          };
    const acc_t  *layer_ff2_b[NUM_LAYERS] = { bert_l0_ff2_b,          bert_l1_ff2_b          };
    const float  *layer_attn_ln_gamma[NUM_LAYERS] = { bert_l0_attn_ln_gamma, bert_l1_attn_ln_gamma };
    const float  *layer_attn_ln_beta [NUM_LAYERS] = { bert_l0_attn_ln_beta,  bert_l1_attn_ln_beta  };
    const float  *layer_ffn_ln_gamma [NUM_LAYERS] = { bert_l0_ffn_ln_gamma,  bert_l1_ffn_ln_gamma  };
    const float  *layer_ffn_ln_beta  [NUM_LAYERS] = { bert_l0_ffn_ln_beta,   bert_l1_ffn_ln_beta   };

    /* ── Per-layer quantization scale tables ───────────────────────────── */
    const float layer_acc_scale_q  [NUM_LAYERS] = { bert_l0_acc_scale_q,   bert_l1_acc_scale_q   };
    const float layer_acc_scale_k  [NUM_LAYERS] = { bert_l0_acc_scale_k,   bert_l1_acc_scale_k   };
    const float layer_acc_scale_v  [NUM_LAYERS] = { bert_l0_acc_scale_v,   bert_l1_acc_scale_v   };
    const float layer_acc_scale_qkt[NUM_LAYERS] = { bert_l0_acc_scale_qkt, bert_l1_acc_scale_qkt };
    const float layer_acc_scale_ctx[NUM_LAYERS] = { bert_l0_acc_scale_ctx, bert_l1_acc_scale_ctx };
    const float layer_acc_scale_ff1[NUM_LAYERS] = { bert_l0_acc_scale_ff1, bert_l1_acc_scale_ff1 };
    const float layer_wo_acc_to_real  [NUM_LAYERS] = { bert_l0_wo_acc_to_real,   bert_l1_wo_acc_to_real   };
    const float layer_attn_res_to_real[NUM_LAYERS] = { bert_l0_attn_res_to_real, bert_l1_attn_res_to_real };
    const float layer_attn_ln_out_sc  [NUM_LAYERS] = { bert_l0_attn_ln_out_scale, bert_l1_attn_ln_out_scale };
    const float layer_ff2_acc_to_real [NUM_LAYERS] = { bert_l0_ff2_acc_to_real,  bert_l1_ff2_acc_to_real  };
    const float layer_ff_res_to_real  [NUM_LAYERS] = { bert_l0_ff_res_to_real,   bert_l1_ff_res_to_real   };
    const float layer_ffn_ln_out_sc   [NUM_LAYERS] = { bert_l0_ffn_ln_out_scale, bert_l1_ffn_ln_out_scale };
    const float layer_gelu_in_scale   [NUM_LAYERS] = { bert_l0_gelu_in_scale,    bert_l1_gelu_in_scale    };
    const float layer_gelu_out_scale  [NUM_LAYERS] = { bert_l0_gelu_out_scale,   bert_l1_gelu_out_scale   };
    const float layer_pooler_cls_scale[NUM_LAYERS] = { bert_l0_pooler_cls_scale, bert_l1_pooler_cls_scale };

    /* ── Metric counters ───────────────────────────────────────────────── */
    int correct = 0;
    /* confusion[true_label][pred_label] */
    int confusion[NUM_LABELS][NUM_LABELS];
    memset(confusion, 0, sizeof(confusion));

    setvbuf(stdout, NULL, _IONBF, 0);

    uint64_t total_start = read_cycles();

    /* ── Main evaluation loop ──────────────────────────────────────────── */
    for (int ex = 0; ex < NUM_EXAMPLES; ex++) {
        /* Read one embedded sequence: SEQ_LEN × HIDDEN_DIM int8 */
        size_t items = fread(layer_ping, sizeof(elem_t), EXAMPLE_SIZE, fp_emb);
        if (items != (size_t)EXAMPLE_SIZE) {
            printf("Warning: short read at example %d (got %zu of %d), stopping.\n",
                   ex, items, EXAMPLE_SIZE);
            break;
        }

        /* Read attention mask: SEQ_LEN uint8 */
        size_t mask_items = fread(cur_attn_mask, sizeof(uint8_t), SEQ_LEN, fp_mask);
        if (mask_items != (size_t)SEQ_LEN) {
            printf("Warning: short mask read at example %d, stopping.\n", ex);
            break;
        }

        /* ── Encoder layers ────────────────────────────────────────────── */
        elem_t (*cur)[HIDDEN_DIM] = layer_ping;
        elem_t (*nxt)[HIDDEN_DIM] = layer_pong;

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            attention(
                HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, SEQ_LEN,
                COMPRESSION_FACTOR, tiled_matmul_type,
                (elem_t *)cur, (elem_t *)cur,
                (elem_t *)attn_ln_out, (elem_t *)attn_resadd,
                layer_Wq[layer],  layer_Wk[layer],
                layer_Wv[layer],  layer_Wo[layer],
                layer_Wq_b[layer], layer_Wk_b[layer],
                layer_Wv_b[layer], layer_Wo_b[layer],
                (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf,
                (elem_t *)attn_buf, (elem_t *)ffn_out_buf, (acc_t *)acc_buf,
                cur_attn_mask,
                layer_attn_ln_gamma[layer], layer_attn_ln_beta[layer],
                layer_acc_scale_q[layer], layer_acc_scale_k[layer],
                layer_acc_scale_v[layer], layer_acc_scale_qkt[layer],
                layer_acc_scale_ctx[layer],
                layer_wo_acc_to_real[layer], layer_attn_res_to_real[layer],
                layer_attn_ln_out_sc[layer]);

            ffn(
                HIDDEN_DIM, EXPANSION_DIM, SEQ_LEN,
                tiled_matmul_type,
                (elem_t *)attn_resadd, (elem_t *)nxt,
                layer_ff1_w[layer], layer_ff2_w[layer],
                layer_ff1_b[layer], layer_ff2_b[layer],
                (elem_t *)ffn_out_buf, (acc_t *)acc_buf,
                layer_ffn_ln_gamma[layer], layer_ffn_ln_beta[layer],
                layer_acc_scale_ff1[layer],
                layer_gelu_in_scale[layer], layer_gelu_out_scale[layer],
                layer_ff2_acc_to_real[layer], layer_ff_res_to_real[layer],
                layer_ffn_ln_out_sc[layer]);

            elem_t (*tmp)[HIDDEN_DIM] = cur; cur = nxt; nxt = tmp;
        }
        gemmini_fence();

        /* ── Pooler: CLS token → dense(tanh) ──────────────────────────── */
        /* Dequantize CLS int8 to real values before float pooler */
        float cls_scale = layer_pooler_cls_scale[NUM_LAYERS - 1];
        float cls_pooled[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float acc = bert_pooler_b[i];
            for (int j = 0; j < HIDDEN_DIM; j++)
                acc += (float)cur[0][j] * cls_scale * bert_pooler_w[i][j];
            cls_pooled[i] = tanhf(acc);
        }

        /* ── Classifier: cls_pooled → logits ──────────────────────────── */
        float logits[NUM_LABELS];
        for (int l = 0; l < NUM_LABELS; l++) {
            float acc = bert_clf_b[l];
            for (int j = 0; j < HIDDEN_DIM; j++)
                acc += cls_pooled[j] * bert_clf_w[l][j];
            logits[l] = acc;
        }

        int pred  = (logits[1] > logits[0]) ? 1 : 0;
        int label = labels[ex];

        if (pred == label) correct++;
        confusion[label][pred]++;

        /* Progress every 50 examples */
        if ((ex + 1) % 50 == 0 || (ex + 1) == NUM_EXAMPLES) {
            float acc_pct = 100.0f * correct / (ex + 1);
            printf("  [%d/%d]  accuracy = %.2f%% (%d/%d correct)\n",
                   ex + 1, NUM_EXAMPLES, acc_pct, correct, ex + 1);
        }

        /* Debug: print first 3 examples */
        if (ex < 3) {
            printf("    Example %d: label=%d pred=%d  logits=[%.4f, %.4f]%s\n",
                   ex, label, pred, (double)logits[0], (double)logits[1],
                   pred == label ? "  CORRECT" : "  WRONG");
        }
    }

    uint64_t total_end = read_cycles();

    fclose(fp_emb);
    fclose(fp_mask);

    /* ── Compute and report metrics ────────────────────────────────────── */
    int total = correct; /* count of correct predictions */
    int total_examples = 0;
    for (int i = 0; i < NUM_LABELS; i++)
        for (int j = 0; j < NUM_LABELS; j++)
            total_examples += confusion[i][j];

    float accuracy = 100.0f * correct / total_examples;

    printf("\n========================================\n");
    printf("  BERT-Tiny SST-2 Evaluation Results\n");
    printf("========================================\n");
    printf("Examples evaluated: %d\n", total_examples);
    printf("Mode: %s\n\n", mode_str);

    /* Confusion matrix */
    printf("Confusion Matrix:\n");
    printf("                 Pred NEG   Pred POS\n");
    printf("  True NEG       %5d       %5d\n", confusion[0][0], confusion[0][1]);
    printf("  True POS       %5d       %5d\n", confusion[1][0], confusion[1][1]);
    printf("\n");

    /* Per-class precision, recall, F1 */
    const char *class_names[NUM_LABELS] = {"NEGATIVE", "POSITIVE"};
    float f1_scores[NUM_LABELS];

    for (int c = 0; c < NUM_LABELS; c++) {
        int tp = confusion[c][c];
        int fp = 0, fn = 0;
        for (int i = 0; i < NUM_LABELS; i++) {
            if (i != c) {
                fp += confusion[i][c];  /* other classes predicted as c */
                fn += confusion[c][i];  /* c predicted as other classes */
            }
        }
        float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
        float recall    = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
        float f1 = (precision + recall > 0.0f)
                  ? 2.0f * precision * recall / (precision + recall) : 0.0f;
        f1_scores[c] = f1;

        printf("  %s:  precision=%.4f  recall=%.4f  F1=%.4f\n",
               class_names[c], (double)precision, (double)recall, (double)f1);
    }

    float macro_f1 = 0.0f;
    for (int c = 0; c < NUM_LABELS; c++) macro_f1 += f1_scores[c];
    macro_f1 /= NUM_LABELS;

    printf("\n  Accuracy:  %d / %d = %.2f%%\n", correct, total_examples, (double)accuracy);
    printf("  Macro-F1:  %.4f\n", (double)macro_f1);
    printf("\n  Total cycles: %llu\n",
           (unsigned long long)(total_end - total_start));
    printf("========================================\n");

    exit(0);
}
