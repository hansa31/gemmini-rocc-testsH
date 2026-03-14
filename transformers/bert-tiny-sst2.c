/* bert-tiny-sst2.c
 *
 * End-to-end BERT-Tiny (M-FAC/bert-tiny-finetuned-sst2) inference on Gemmini.
 * Quantised int8 (elem_t) path — compiled against the standard gemmini_params.h
 * where elem_t = int8_t, acc_t = int32_t.
 *
 * Architecture (from setup_BERT/weights/arch_info.json):
 *   hidden_dim = 128 | ffn_dim = 512 | num_heads = 2 | num_layers = 2
 *
 * Prerequisites — generate the weight headers first:
 *   conda run -n ImageNet python setup_BERT/generate_bert_params.py
 * Produces:
 *   bert_params.h   — int8 (elem_t) weights + int32 (acc_t) biases
 *   bert_input.h    — dummy input, analogous to imagenet/images.h
 *
 * Build:
 *   riscv64-unknown-elf-gcc -O2 -DBAREMETAL -Iinclude \
 *       transformers/bert-tiny-sst2.c -o bert-tiny-sst2-baremetal -lm
 *
 * Run:
 *   spike --extension=gemmini bert-tiny-sst2-baremetal [ws|os|cpu]
 *   Default mode is ws (Weight Stationary).  Pass "cpu" to run on the RISC-V
 *   scalar core via tiled_matmul_auto — useful for debugging without hardware.
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

/* Dummy input — deterministic int8 pattern, analogous to imagenet/images.h */
#include "bert_input.h"

/* ── BERT-Tiny architecture constants ───────────────────────────────────────── */
#define HIDDEN_DIM         128
#define EXPANSION_DIM      512
#define NUM_HEADS          2
#define SEQ_LEN            128
#define NUM_LAYERS         2
#define NUM_LABELS         2
#define COMPRESSION_FACTOR 1

/* ── Attention sub-layer ─────────────────────────────────────────────────────
 *
 *   Q  = input × Wq + Wq_b        K = input × Wk + Wk_b
 *   V  = input × Wv + Wv_b
 *   per head: scores = softmax(Q_h × K_h^T)
 *   per head: ctx    = scores × V_h
 *   proj       = concat(ctx heads) × Wo + Wo_b   [accumulator]
 *   norm_out   = LayerNorm(proj)
 *   resadd_out = norm_out + input
 *
 * tiled_matmul_type selects WS / OS / CPU at runtime (same as mobilenet_v1.c).
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
        elem_t *attn_buf, elem_t *out_buf, acc_t *out_buf_acc)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head   = hidden_dim_compressed / num_heads;

    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head   = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    /* Q, K, V projections */
    const elem_t *qkv_weights[3] = {Wq, Wk, Wv};
    const elem_t *qkv_ins[3]     = {input, enc_out, enc_out};
    const acc_t  *qkv_bs[3]      = {Wq_b, Wk_b, Wk_b};
    elem_t       *qkv_outs[3]    = {Q_buf, K_buf, V_buf};

    for (int i = 0; i < 3; i++) {
        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            qkv_ins[i],    qkv_weights[i],
            qkv_bs[i],     qkv_outs[i],
            hidden_dim, hidden_dim, 0, hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            false, false, false, false, false,
            0, tiled_matmul_type);
    }

    gemmini_fence();

    /* Attention scores: softmax(Q × K^T) per head */
    for (int head = 0; head < num_heads; head++) {
        const elem_t *A = Q_buf    + head * hidden_dim_per_head;
        const elem_t *B = K_buf    + head * hidden_dim_per_head;
        elem_t       *C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            A, B, NULL, C,
            hidden_dim, hidden_dim, 0, seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            SOFTMAX, ACC_SCALE_IDENTITY, 0,
            false, false, true, false, false,
            0, tiled_matmul_type);
    }

    gemmini_fence();

    /* Context vectors: attn × V per head */
    for (int head = 0; head < num_heads; head++) {
        const elem_t *A = attn_buf + head * seq_len * seq_len;
        const elem_t *B = V_buf    + head * hidden_dim_per_head;
        elem_t       *C = out_buf  + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            A, B, NULL, C,
            seq_len, hidden_dim, 0, hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
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

    /* LayerNorm + residual add */
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t *)out_buf_acc, (elem_t *)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, tiled_matmul_type);

    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        input, out, resadd_out,
        false, tiled_matmul_type == CPU ? CPU : WS);

    gemmini_fence();
}

/* ── Feed-forward sub-layer ─────────────────────────────────────────────────
 *
 *   ff_hidden  = GELU(input × ff1_w + ff1_b)
 *   ff_out_acc = ff_hidden × ff2_w + ff2_b   [accumulator]
 *   norm_out   = LayerNorm(ff_out_acc)
 *   out        = norm_out + input
 */
static void ffn(
        int hidden_dim, int expansion_dim, int seq_len,
        enum tiled_matmul_type_t tiled_matmul_type,
        const elem_t *input, elem_t *out,
        const elem_t *ff1_w, const elem_t *ff2_w,
        const acc_t  *ff1_b, const acc_t  *ff2_b,
        elem_t *out_buf, acc_t *out_buf_acc)
{
    /* FF1 + GELU → out_buf [seq_len × expansion_dim] */
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        true, false, false, false, false,
        0, tiled_matmul_type);

    gemmini_fence();

    /* FF2 → out_buf_acc [seq_len × hidden_dim] */
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim,
        out_buf, ff2_w, ff2_b, out_buf_acc,
        expansion_dim, hidden_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, true, false,
        0, tiled_matmul_type);

    gemmini_fence();

    /* LayerNorm + residual add */
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t *)out_buf_acc, (elem_t *)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, tiled_matmul_type);

    gemmini_fence();

    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        out, input, out,
        false, tiled_matmul_type == CPU ? CPU : WS);

    gemmini_fence();
}

/* ── Scratch buffers (BSS, zero-initialised by ELF loader) ──────────────────
 * Weights live in bert_params.h (ROM).
 */
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

    /* ── Select execution mode from argv[1] (same convention as mobilenet_v1.c) */
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

    printf("=== BERT-Tiny SST-2 — int8 Gemmini path (mode=%s) ===\n", mode_str);
    printf("hidden=%d  ffn=%d  heads=%d  layers=%d  seq_len=%d\n\n",
           HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN);

    /* ── Load dummy input (bert_input.h) into ping buffer ───────────────────
     * Real inference: token_ids → word+pos+type embeddings → LayerNorm → here
     */
    memcpy(layer_ping, bert_input, sizeof(bert_input));

    /* ── Per-layer weight pointer tables ────────────────────────────────────
     * Maps layer index to the named arrays from bert_params.h.
     */
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

    uint64_t total_start = read_cycles();

    /* ── Encoder layers ─────────────────────────────────────────────────── */
    elem_t (*cur)[HIDDEN_DIM] = layer_ping;
    elem_t (*nxt)[HIDDEN_DIM] = layer_pong;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        uint64_t layer_start = read_cycles();

        attention(
            HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, SEQ_LEN, COMPRESSION_FACTOR,
            tiled_matmul_type,
            (elem_t *)cur, (elem_t *)cur,
            (elem_t *)attn_ln_out, (elem_t *)attn_resadd,
            layer_Wq[layer],  layer_Wk[layer],
            layer_Wv[layer],  layer_Wo[layer],
            layer_Wq_b[layer], layer_Wk_b[layer],
            layer_Wv_b[layer], layer_Wo_b[layer],
            (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf,
            (elem_t *)attn_buf, (elem_t *)ffn_out_buf, (acc_t *)acc_buf);

        ffn(
            HIDDEN_DIM, EXPANSION_DIM, SEQ_LEN,
            tiled_matmul_type,
            (elem_t *)attn_resadd, (elem_t *)nxt,
            layer_ff1_w[layer], layer_ff2_w[layer],
            layer_ff1_b[layer], layer_ff2_b[layer],
            (elem_t *)ffn_out_buf, (acc_t *)acc_buf);

        printf("Encoder layer %d: %llu cycles\n",
               layer, (unsigned long long)(read_cycles() - layer_start));

        elem_t (*tmp)[HIDDEN_DIM] = cur; cur = nxt; nxt = tmp;
    }

    gemmini_fence();

    /* ── Pooler: CLS token → dense(tanh) ────────────────────────────────────
     * tanh is not a Gemmini operation; accumulate in float then cast back.
     * bert_params.h stores pooler/classifier as float (CPU-only weights).
     */
    float cls_pooled[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float acc = bert_pooler_b[i];
        for (int j = 0; j < HIDDEN_DIM; j++)
            acc += (float)cur[0][j] * bert_pooler_w[i][j];
        cls_pooled[i] = tanhf(acc);
    }

    /* ── Classifier: cls_pooled → logits ────────────────────────────────── */
    float logits[NUM_LABELS];
    for (int l = 0; l < NUM_LABELS; l++) {
        float acc = bert_clf_b[l];
        for (int j = 0; j < HIDDEN_DIM; j++)
            acc += cls_pooled[j] * bert_clf_w[l][j];
        logits[l] = acc;
    }

    int        pred = (logits[1] > logits[0]) ? 1 : 0;
    const char *lbl = pred ? "POSITIVE" : "NEGATIVE";

    printf("\nLogits:     NEGATIVE=%.6f   POSITIVE=%.6f\n",
           (double)logits[0], (double)logits[1]);
    printf("Prediction: %s (label %d)\n", lbl, pred);
    printf("Total cycles: %llu\n",
           (unsigned long long)(read_cycles() - total_start));

    exit(0);
}
