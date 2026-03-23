#!/usr/bin/env python3
"""
generate_bert_params.py
=======================
Reads the .npy weight files produced by export_weights.py and emits two C headers:

  ../bert_params.h  — quantized int8 (elem_t) attention/FFN weights + int32 (acc_t)
                      biases for the Gemmini accelerator; pooler and classifier kept
                      as float32 (they run on the CPU scalar core).

  ../bert_input.h   — a static dummy input sequence (elem_t[SEQ_LEN][HIDDEN_DIM])
                      with small deterministic integer values, matching the style of
                      imagenet/images.h.  Replace with real tokenized embeddings later.

Usage (from repo root or from this directory):
    conda run -n ImageNet python setup_BERT/generate_bert_params.py
    # or simply:
    python3 setup_BERT/generate_bert_params.py

The script must be run AFTER export_weights.py has populated setup_BERT/weights/.

Quantisation note
-----------------
PyTorch linear layers store weight as W[out_features, in_features].
Gemmini's tiled_matmul_auto expects matrix B in layout [K][N] = [in][out].
So every weight matrix is transposed before quantisation.

Symmetric per-tensor int8:
    scale   = max(|W_transposed|) / 127           (floor at 1e-6)
    W_q     = round(W_T / scale).clip(-128,127)   → int8 (elem_t)

Bias (acc_t = int32) lives in accumulator space alongside int8×int8 products.
Assuming the dummy input values are small integers treated at scale 1/127:
    b_q = round(b / (w_scale * input_scale))      where input_scale = 1/127
        = round(b * 127 / w_scale)                → int32 (acc_t)

This is a first-pass quantisation suited for hardware flow testing.
Replace scale factors with proper PTQ/QAT values for accuracy runs.
"""

import os
import sys
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR  = os.path.join(SCRIPT_DIR, "weights")
TRANSFORMERS = os.path.join(SCRIPT_DIR, "..")   # ../  (transformers/)
PARAMS_PATH  = os.path.join(TRANSFORMERS, "bert_params.h")
INPUT_PATH   = os.path.join(TRANSFORMERS, "bert_input.h")

# ── Architecture constants (must match bert-tiny-sst2.c) ────────────────────
HIDDEN_DIM    = 128
EXPANSION_DIM = 512
NUM_LAYERS    = 2
NUM_LABELS    = 2
SEQ_LEN       = 128

# INPUT_SCALE is no longer a global constant.  Each sub-block receives a
# per-layer x_scale derived from the actual quantisation of the layer before it.
# See estimate_emb_scale() and estimate_ln_out_scale() below.

# ── Helpers ──────────────────────────────────────────────────────────────────

def load(name):
    """Load a weight .npy file by stem (no extension)."""
    path = os.path.join(WEIGHTS_DIR, name + ".npy")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Weight file not found: {path}\n"
                 "        Run export_weights.py first.")
    return np.load(path).astype(np.float32)


def quantize_weight(W, target_max=127):
    """Symmetric per-tensor int8 quantisation of a 2-D weight array."""
    max_abs  = max(float(np.max(np.abs(W))), 1e-6)
    scale    = max_abs / target_max
    W_q      = np.round(W / scale).clip(-128, 127).astype(np.int8)
    return W_q, scale


def estimate_emb_scale():
    """Compute the per-tensor int8 scale for embedding LN output.

    The embedding pipeline is:
        raw_emb = word_emb[id] + pos_emb[pos] + type_emb[type]
        out = gamma * (raw_emb - mean) / sqrt(var + 1e-12) + beta

    We compute the actual LN output range over all word embeddings
    (at position 0 with type 0) to get a realistic per-tensor scale,
    rather than using a 3-sigma estimate which underestimates outliers.
    """
    word_emb   = load("bert_embeddings_word_embeddings_weight")       # [30522, 128]
    pos_emb    = load("bert_embeddings_position_embeddings_weight")   # [512, 128]
    type_emb   = load("bert_embeddings_token_type_embeddings_weight") # [2, 128]
    emb_ln_gamma = load("bert_embeddings_LayerNorm_weight")          # [128]
    emb_ln_beta  = load("bert_embeddings_LayerNorm_bias")            # [128]

    # Compute LN output for ALL words at position 0, type 0
    emb = word_emb + pos_emb[0:1] + type_emb[0:1]       # [30522, 128]
    mean = emb.mean(axis=-1, keepdims=True)
    var  = emb.var(axis=-1, keepdims=True)
    emb_normed = (emb - mean) / np.sqrt(var + 1e-12)
    emb_out = emb_normed * emb_ln_gamma + emb_ln_beta    # [30522, 128]
    max_abs = max(float(np.max(np.abs(emb_out))), 1e-6)
    return max_abs / 127.0


def estimate_ln_out_scale(ln_gamma, ln_beta):
    """Estimate the int8 output scale after sw_layernorm with gamma/beta.

    After layernorm the channel values are approximately gamma*z + beta where
    z ~ N(0,1).  We use |beta| + 5*|gamma| as a per-channel envelope and take
    the global max as the per-tensor scale.  The 5-sigma bound covers >99.99%
    of typical transformer activations, avoiding the underestimate from 3-sigma.
    """
    max_abs = float(np.max(np.abs(ln_beta) + 5.0 * np.abs(ln_gamma)))
    max_abs = max(max_abs, 1e-6)
    return max_abs / 127.0


def quantize_bias(b, w_scale, in_scale):
    """
    Scale the bias into accumulator space.
        b_q = round( b / (w_scale * in_scale) )
    """
    denom = w_scale * in_scale
    b_q   = np.round(b / denom).clip(-(2**31), 2**31 - 1).astype(np.int64)
    return b_q


def estimate_matmul_out_range(ln_gamma, ln_beta, W_deq, bias, sigma=5.0):
    """Estimate max absolute output of matmul Y = X@W + b
    where X comes from LayerNorm with gamma/beta (X[k] ~ N(beta[k], gamma[k]^2)).
    """
    mean_y = ln_beta.astype(np.float64) @ W_deq.astype(np.float64) + bias.astype(np.float64)
    var_y  = (ln_gamma.astype(np.float64)**2) @ (W_deq.astype(np.float64)**2)
    return max(float(np.max(np.abs(mean_y) + sigma * np.sqrt(np.maximum(var_y, 0)))), 1e-6)


def estimate_score_range(ln_gamma, ln_beta, Wq_deq, Wq_b, Wk_deq, Wk_b,
                         head_dim, num_heads, sigma=5.0):
    """Estimate max absolute attention score  Q·K^T  per head."""
    max_score = 0.0
    for h in range(num_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        mq = ln_beta.astype(np.float64) @ Wq_deq[:, sl].astype(np.float64) + Wq_b[sl].astype(np.float64)
        vq = (ln_gamma.astype(np.float64)**2) @ (Wq_deq[:, sl].astype(np.float64)**2)
        mk = ln_beta.astype(np.float64) @ Wk_deq[:, sl].astype(np.float64) + Wk_b[sl].astype(np.float64)
        vk = (ln_gamma.astype(np.float64)**2) @ (Wk_deq[:, sl].astype(np.float64)**2)
        mean_s = float(np.sum(mq * mk))
        var_s  = float(np.sum(vq * vk + mq**2 * vk + vq * mk**2))
        ms = abs(mean_s) + sigma * np.sqrt(max(var_s, 0))
        if ms > max_score:
            max_score = ms
    return max(max_score, 1e-6)


def flat_c(arr):
    """1-D or N-D array → flat C brace-initialiser e.g. {1,-2,3}"""
    return "{" + ",".join(str(int(v)) for v in arr.flatten()) + "}"


def matrix_c(arr):
    """2-D array → nested C brace-initialiser {{row0},{row1},...}"""
    rows = ["{" + ",".join(str(int(v)) for v in row) + "}" for row in arr]
    return "{" + ",".join(rows) + "}"


def float_row(arr):
    return "{" + ",".join(f"{float(v):.8f}f" for v in arr) + "}"


def float_matrix(arr):
    rows = [float_row(row) for row in arr]
    return "{" + ",".join(rows) + "}"


# ── Generate bert_params.h ───────────────────────────────────────────────────

def gen_params():
    L = []

    L += [
        "/* bert_params.h",
        " * Auto-generated by setup_BERT/generate_bert_params.py — DO NOT EDIT.",
        " *",
        " * Quantised weights for BERT-Tiny (M-FAC/bert-tiny-finetuned-sst2).",
        " * elem_t (int8)  : attention and FFN weights, transposed to gemmini layout.",
        " * acc_t  (int32) : biases in accumulator space.",
        " * float          : LN gamma/beta, pooler and classifier (run on CPU).",
        " */",
        "",
        "#ifndef BERT_PARAMS_H",
        "#define BERT_PARAMS_H",
        "",
        "#include <include/gemmini_params.h>",
        "",
    ]

    # ── Compute per-layer input scales for correct bias quantization ──────────
    # The chain of input scales:
    #   embeddings → embedding LN → [layer 0 attention] → attn LN
    #             → [layer 0 FFN] → ffn LN → [layer 1 attention] → attn LN
    #             → [layer 1 FFN] → ffn LN → pooler (float)
    # After each sw_layernorm, x_scale = estimate_ln_out_scale(gamma, beta)
    # --
    # x_scale for the first attention block = embedding LN output scale
    cur_x_scale = estimate_emb_scale()
    print(f"[INFO] embedding output x_scale = {cur_x_scale:.6f}  "
          f"(max(|beta|+3|gamma|)/127)")

    for layer in range(NUM_LAYERS):
        pfx = f"bert_encoder_layer_{layer}"
        L.append(f"/* ── Encoder layer {layer} {'─'*57}*/")
        L.append("")

        # ── Attention weight matrices (int8): PyTorch [out=H, in=H] → gemmini [in=H, out=H]
        w_scales = {}
        for tag, npy_stem in [
            ("Wq", f"{pfx}_attention_self_query_weight"),
            ("Wk", f"{pfx}_attention_self_key_weight"),
            ("Wv", f"{pfx}_attention_self_value_weight"),
            ("Wo", f"{pfx}_attention_output_dense_weight"),
        ]:
            W_pt    = load(npy_stem)           # [out=128, in=128]
            W_g     = W_pt.T                   # [in=128, out=128]
            W_q, sc = quantize_weight(W_g)
            w_scales[tag] = sc
            L.append(
                f"static const elem_t bert_l{layer}_{tag}"
                f"[{HIDDEN_DIM}][{HIDDEN_DIM}] row_align(1) = {matrix_c(W_q)};"
            )
        L.append("")

        # ── FFN weight matrices (int8)
        # ff1: PyTorch [out=512, in=128] → gemmini [in=128, out=512], flat [128*512]
        ff1_pt     = load(f"{pfx}_intermediate_dense_weight")  # [512,128]
        ff1_g      = ff1_pt.T                                   # [128,512]
        ff1_q, sc1 = quantize_weight(ff1_g)
        w_scales["ff1"] = sc1
        L.append(
            f"static const elem_t bert_l{layer}_ff1_w"
            f"[{HIDDEN_DIM * EXPANSION_DIM}] row_align(1) = {flat_c(ff1_q)};"
        )

        # ff2: PyTorch [out=128, in=512] → gemmini [in=512, out=128], flat [512*128]
        ff2_pt     = load(f"{pfx}_output_dense_weight")         # [128,512]
        ff2_g      = ff2_pt.T                                    # [512,128]
        ff2_q, sc2 = quantize_weight(ff2_g)
        w_scales["ff2"] = sc2
        L.append(
            f"static const elem_t bert_l{layer}_ff2_w"
            f"[{EXPANSION_DIM * HIDDEN_DIM}] row_align(1) = {flat_c(ff2_q)};"
        )
        L.append("")

        # ── Compute per-step output scales (y_scale) for proper ACC_SCALE ────
        HEAD_DIM = HIDDEN_DIM // 2   # NUM_HEADS = 2

        # Dequantised weights for analytical range estimation
        Wq_deq = quantize_weight(load(f"{pfx}_attention_self_query_weight").T)[0].astype(np.float64) * w_scales["Wq"]
        Wk_deq = quantize_weight(load(f"{pfx}_attention_self_key_weight").T)[0].astype(np.float64) * w_scales["Wk"]
        Wv_deq = quantize_weight(load(f"{pfx}_attention_self_value_weight").T)[0].astype(np.float64) * w_scales["Wv"]

        # Use embedding LN gamma/beta for layer 0, previous ffn LN for later layers
        if layer == 0:
            input_ln_gamma = load("bert_embeddings_LayerNorm_weight")
            input_ln_beta  = load("bert_embeddings_LayerNorm_bias")
        else:
            input_ln_gamma = prev_ffn_ln_g
            input_ln_beta  = prev_ffn_ln_b

        b_Wq = load(f"{pfx}_attention_self_query_bias")
        b_Wk = load(f"{pfx}_attention_self_key_bias")
        b_Wv = load(f"{pfx}_attention_self_value_bias")
        b_Wo = load(f"{pfx}_attention_output_dense_bias")

        y_qkv_max = max(
            estimate_matmul_out_range(input_ln_gamma, input_ln_beta, Wq_deq, b_Wq),
            estimate_matmul_out_range(input_ln_gamma, input_ln_beta, Wk_deq, b_Wk),
            estimate_matmul_out_range(input_ln_gamma, input_ln_beta, Wv_deq, b_Wv),
        )
        y_qkv = y_qkv_max / 127.0

        y_scores_max = estimate_score_range(
            input_ln_gamma, input_ln_beta,
            Wq_deq, b_Wq, Wk_deq, b_Wk,
            HEAD_DIM, 2)
        y_scores = y_scores_max / 127.0

        # Context: softmax concentrates on one token → ctx ≈ V → scale ≈ y_qkv
        y_ctx = y_qkv

        # ACC_SCALE for each matmul step
        acc_sc_q   = cur_x_scale * w_scales["Wq"] / y_qkv
        acc_sc_k   = cur_x_scale * w_scales["Wk"] / y_qkv
        acc_sc_v   = cur_x_scale * w_scales["Wv"] / y_qkv
        acc_sc_qkt = y_qkv * y_qkv / y_scores
        acc_sc_ctx = (1.0 / 127.0) * y_qkv / y_ctx

        # Wo input scale = y_ctx (context vector scale)
        wo_x_scale = y_ctx
        wo_acc_to_real = y_ctx * w_scales["Wo"]   # for LN residual

        print(f"[INFO] layer {layer} attn  x_scale = {cur_x_scale:.6f}")
        print(f"[INFO] layer {layer} y_qkv = {y_qkv:.6f}  y_scores = {y_scores:.6f}  y_ctx = {y_ctx:.6f}")
        print(f"[INFO] layer {layer} ACC_SCALE q={acc_sc_q:.6g} k={acc_sc_k:.6g} v={acc_sc_v:.6g}")
        print(f"[INFO] layer {layer} ACC_SCALE qkt={acc_sc_qkt:.6g} ctx={acc_sc_ctx:.6g}")

        # ── Attention biases (acc_t = int32)
        for tag, bias_stem in [
            ("Wq_b", f"{pfx}_attention_self_query_bias"),
            ("Wk_b", f"{pfx}_attention_self_key_bias"),
            ("Wv_b", f"{pfx}_attention_self_value_bias"),
            ("Wo_b", f"{pfx}_attention_output_dense_bias"),
        ]:
            b  = load(bias_stem)
            in_scale = wo_x_scale if tag == "Wo_b" else cur_x_scale
            bq = quantize_bias(b, w_scales[tag[:-2]], in_scale)
            L.append(
                f"static const acc_t bert_l{layer}_{tag}"
                f"[{HIDDEN_DIM}] row_align_acc(1) = {flat_c(bq)};"
            )

        # ── Attention output LayerNorm (float arrays used by sw_layernorm)
        attn_ln_g = load(f"{pfx}_attention_output_LayerNorm_weight")  # [128]
        attn_ln_b = load(f"{pfx}_attention_output_LayerNorm_bias")    # [128]
        L.append(
            f"static const float bert_l{layer}_attn_ln_gamma[{HIDDEN_DIM}] = "
            f"{float_row(attn_ln_g)};"
        )
        L.append(
            f"static const float bert_l{layer}_attn_ln_beta [{HIDDEN_DIM}] = "
            f"{float_row(attn_ln_b)};"
        )

        attn_ln_x_scale = estimate_ln_out_scale(attn_ln_g, attn_ln_b)
        print(f"[INFO] layer {layer} attn_ln_out_scale = {attn_ln_x_scale:.6f}")

        # ── FFN scale estimation ─────────────────────────────────────────
        ff1_deq = ff1_q.astype(np.float64) * w_scales["ff1"]
        b_ff1   = load(f"{pfx}_intermediate_dense_bias")

        y_ff1_max = estimate_matmul_out_range(attn_ln_g, attn_ln_b, ff1_deq, b_ff1)
        y_ff1 = y_ff1_max / 127.0
        # GELU output: gelu(x) ≈ x for x>0, ≈ 0 for x<0 → output range ≈ 60% of input
        y_ff1_gelu = 0.6 * y_ff1
        acc_sc_ff1 = attn_ln_x_scale * w_scales["ff1"] / y_ff1

        # FF2 input scale = y_ff1_gelu
        ff2_x_scale = y_ff1_gelu
        ff2_acc_to_real = y_ff1_gelu * w_scales["ff2"]

        print(f"[INFO] layer {layer} y_ff1 = {y_ff1:.6f}  y_ff1_gelu = {y_ff1_gelu:.6f}")
        print(f"[INFO] layer {layer} ACC_SCALE ff1={acc_sc_ff1:.6g}")

        # ── FFN biases (acc_t = int32)
        bq_ff1 = quantize_bias(b_ff1, w_scales["ff1"], attn_ln_x_scale)
        L.append(
            f"static const acc_t bert_l{layer}_ff1_b"
            f"[{EXPANSION_DIM}] row_align_acc(1) = {flat_c(bq_ff1)};"
        )

        b_ff2  = load(f"{pfx}_output_dense_bias")
        bq_ff2 = quantize_bias(b_ff2, w_scales["ff2"], ff2_x_scale)
        L.append(
            f"static const acc_t bert_l{layer}_ff2_b"
            f"[{HIDDEN_DIM}] row_align_acc(1) = {flat_c(bq_ff2)};"
        )

        # ── FFN output LayerNorm (float arrays used by sw_layernorm)
        ffn_ln_g = load(f"{pfx}_output_LayerNorm_weight")   # [128]
        ffn_ln_b = load(f"{pfx}_output_LayerNorm_bias")     # [128]
        L.append(
            f"static const float bert_l{layer}_ffn_ln_gamma[{HIDDEN_DIM}] = "
            f"{float_row(ffn_ln_g)};"
        )
        L.append(
            f"static const float bert_l{layer}_ffn_ln_beta [{HIDDEN_DIM}] = "
            f"{float_row(ffn_ln_b)};"
        )

        ffn_ln_x_scale = estimate_ln_out_scale(ffn_ln_g, ffn_ln_b)
        print(f"[INFO] layer {layer} ffn_ln_out_scale = {ffn_ln_x_scale:.6f}")

        # ── Per-layer quantization scales ──────────────────────────────────
        L.append("")
        L.append(f"/* Quantization scales for layer {layer} */")
        for name, val in [
            ("acc_scale_q",     acc_sc_q),
            ("acc_scale_k",     acc_sc_k),
            ("acc_scale_v",     acc_sc_v),
            ("acc_scale_qkt",   acc_sc_qkt),
            ("acc_scale_ctx",   acc_sc_ctx),
            ("acc_scale_ff1",   acc_sc_ff1),
            ("attn_ln_out_scale", attn_ln_x_scale),
            ("ffn_ln_out_scale",  ffn_ln_x_scale),
            ("wo_acc_to_real",  wo_acc_to_real),
            ("attn_res_to_real", cur_x_scale),
            ("ff2_acc_to_real", ff2_acc_to_real),
            ("ff_res_to_real",  attn_ln_x_scale),
            ("gelu_in_scale",   y_ff1),
            ("gelu_out_scale",  y_ff1_gelu),
            ("pooler_cls_scale", ffn_ln_x_scale),
        ]:
            L.append(f"static const float bert_l{layer}_{name} = {val:.10g}f;")
        L.append("")

        # ── Update scales for next encoder layer
        cur_x_scale = ffn_ln_x_scale
        prev_ffn_ln_g = ffn_ln_g
        prev_ffn_ln_b = ffn_ln_b
        print(f"[INFO] layer {layer} output x_scale = {cur_x_scale:.6f}  "
              f"(after ffn LN) → x_scale for layer {layer+1}")

    # ── Pooler (CPU float, PyTorch layout [out=128, in=128]) ─────────────────
    L += [
        "/* ── Pooler — float32, runs on the CPU scalar core " + "─"*27 + "*/",
        "/*    C code: out[i] = tanh( sum_j( cls_in[j] * bert_pooler_w[i][j] ) + bert_pooler_b[i] ) */",
        "",
    ]
    pw = load("bert_pooler_dense_weight")   # [128, 128], PyTorch [out, in]
    pb = load("bert_pooler_dense_bias")     # [128]
    L.append(
        f"static const float bert_pooler_w[{HIDDEN_DIM}][{HIDDEN_DIM}]"
        f" = {float_matrix(pw)};"
    )
    L.append(
        f"static const float bert_pooler_b[{HIDDEN_DIM}]"
        f" = {float_row(pb)};"
    )
    L.append("")

    # ── Classifier (CPU float, PyTorch layout [out=2, in=128]) ───────────────
    L += [
        "/* ── Classifier — float32, runs on the CPU scalar core " + "─"*24 + "*/",
        "/*    C code: logit[l] = sum_j( cls_pooled[j] * bert_clf_w[l][j] ) + bert_clf_b[l] */",
        "",
    ]
    cw = load("classifier_weight")   # [2, 128]
    cb = load("classifier_bias")     # [2]
    L.append(
        f"static const float bert_clf_w[{NUM_LABELS}][{HIDDEN_DIM}]"
        f" = {float_matrix(cw)};"
    )
    L.append(
        f"static const float bert_clf_b[{NUM_LABELS}]"
        f" = {float_row(cb)};"
    )
    L.append("")
    L.append("#endif /* BERT_PARAMS_H */")
    L.append("")

    with open(PARAMS_PATH, "w") as f:
        f.write("\n".join(L))

    total_weights = NUM_LAYERS * (4 * HIDDEN_DIM * HIDDEN_DIM
                                  + HIDDEN_DIM * EXPANSION_DIM
                                  + EXPANSION_DIM * HIDDEN_DIM)
    print(f"[OK] {PARAMS_PATH}")
    print(f"     {total_weights:,} int8 weight values across {NUM_LAYERS} encoder layers")


# ── Generate bert_input.h ────────────────────────────────────────────────────

def gen_input():
    """
    Emit a dummy embedded input sequence: elem_t bert_input[SEQ_LEN][HIDDEN_DIM].

    Values follow the pattern  ((t*3 + d*7) % 9) - 4  → range [-4, 4].
    This is small enough to avoid int8 overflow and produces a deterministic,
    non-trivial pattern across both the token (t) and embedding (d) dimensions —
    analogous to the real int8 pixel values in imagenet/images.h.

    In full inference:  token_ids → word_embedding + position_embedding
                        + token_type_embedding → LayerNorm → layer_ping.
    """
    arr   = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=np.int8)
    for t in range(SEQ_LEN):
        for d in range(HIDDEN_DIM):
            arr[t, d] = ((t * 3 + d * 7) % 9) - 4

    rows = ["{" + ",".join(str(int(v)) for v in arr[t]) + "}" for t in range(SEQ_LEN)]

    L = [
        "/* bert_input.h",
        " * Auto-generated by setup_BERT/generate_bert_params.py — DO NOT EDIT.",
        " *",
        " * Dummy tokenised + embedded input for BERT-Tiny hardware testing.",
        " * Values: ((t*3 + d*7) % 9) - 4  →  range [-4, 4] (int8-safe).",
        " * Analogous to imagenet/images.h for the MobileNet benchmark.",
        " *",
        " * Replace with real embedded tokens for an accuracy test:",
        " *   token_ids → word_embed + pos_embed + token_type_embed → LayerNorm",
        " */",
        "",
        "#ifndef BERT_INPUT_H",
        "#define BERT_INPUT_H",
        "",
        "#include <include/gemmini_params.h>",
        "",
        f"static const elem_t bert_input[{SEQ_LEN}][{HIDDEN_DIM}] row_align(1) = {{",
    ]

    # emit each token row on its own line for readability
    for i, row in enumerate(rows):
        comma = "," if i < SEQ_LEN - 1 else ""
        L.append(f"    {row}{comma}")

    L += [
        "};",
        "",
        "#endif /* BERT_INPUT_H */",
        "",
    ]

    with open(INPUT_PATH, "w") as f:
        f.write("\n".join(L))

    print(f"[OK] {INPUT_PATH}")
    print(f"     {SEQ_LEN}×{HIDDEN_DIM} dummy input, values in [-4, 4]")


# ── Generate bert_params_fp.h ─────────────────────────────────────────────────

FP_PARAMS_PATH = os.path.join(TRANSFORMERS, "bert_params_fp.h")


def float_vec(arr):
    """1-D array → C float brace-initialiser with %.7g precision."""
    return "{" + ",".join(f"{float(v):.7g}f" for v in arr.flatten()) + "}"


def float_mat(arr):
    """2-D array → nested C float brace-initialiser."""
    rows = ["{" + ",".join(f"{float(v):.7g}f" for v in row) + "}" for row in arr]
    return "{" + ",".join(rows) + "}"


def gen_params_fp():
    """
    Emit bert_params_fp.h — original float32 weights expressed using elem_t / acc_t
    typedefs from gemmini_params.h.

    Intended for a float-hardware Gemmini configuration where gemmini_params.h
    redefines:
        typedef float   elem_t;
        typedef float   acc_t;

    Weight layout: PyTorch native [out_features, in_features] — NO transpose.
    (bert-tiny-sst2-fp.c accesses W[out_j][in_k], i.e. out = in @ W.T + bias.)

    Array types:
        elem_t  — weight matrices, LN gamma (multiplicative)
        acc_t   — bias vectors, LN beta (additive)
    Both declared with row_align / row_align_acc so the arrays are correctly
    aligned for DMA when used with float Gemmini hardware.

    Intentionally will NOT compile against the standard int8 gemmini_params.h
    (float literals assigned to int8 arrays → compiler error), preventing
    accidental use with the wrong hardware configuration.
    """
    L = []
    L += [
        "/* bert_params_fp.h",
        " * Auto-generated by setup_BERT/generate_bert_params.py — DO NOT EDIT.",
        " *",
        " * Original float32 weights for BERT-Tiny (M-FAC/bert-tiny-finetuned-sst2).",
        " * For use with bert-tiny-sst2-fp.c on FLOAT-hardware Gemmini",
        " * (gemmini_params.h must define  typedef float elem_t / acc_t).",
        " *",
        " * Array types follow gemmini conventions:",
        " *   elem_t (float) : weight matrices, LN gamma — declared with row_align(1)",
        " *   acc_t  (float) : bias vectors, LN beta     — declared with row_align_acc(1)",
        " *",
        " * Weight layout: PyTorch [out_features, in_features] — NO transpose.",
        " * C linear:  out[i][j] = sum_k( in[i][k] * W[j][k] ) + bias[j]",
        " *            i.e. out = in @ W.T + bias  (standard PyTorch linear).",
        " */",
        "",
        "#ifndef BERT_PARAMS_FP_H",
        "#define BERT_PARAMS_FP_H",
        "",
        "#include <include/gemmini_params.h>",
        "",
    ]

    for layer in range(NUM_LAYERS):
        pfx = f"bert_encoder_layer_{layer}"
        L.append(f"/* ── Encoder layer {layer} {'─'*59}*/")
        L.append("")

        # Attention weight matrices: elem_t [out=128][in=128], PyTorch layout
        for tag, npy_stem in [
            ("Wq",  f"{pfx}_attention_self_query_weight"),
            ("Wk",  f"{pfx}_attention_self_key_weight"),
            ("Wv",  f"{pfx}_attention_self_value_weight"),
            ("Wo",  f"{pfx}_attention_output_dense_weight"),
        ]:
            W = load(npy_stem)   # [128, 128]
            L.append(
                f"static const elem_t bert_fp_l{layer}_{tag}"
                f"[{HIDDEN_DIM}][{HIDDEN_DIM}] row_align(1) = {float_mat(W)};"
            )
        L.append("")

        # Attention biases: acc_t [128]
        for tag, stem in [
            ("Wq_b", f"{pfx}_attention_self_query_bias"),
            ("Wk_b", f"{pfx}_attention_self_key_bias"),
            ("Wv_b", f"{pfx}_attention_self_value_bias"),
            ("Wo_b", f"{pfx}_attention_output_dense_bias"),
        ]:
            b = load(stem)
            L.append(
                f"static const acc_t bert_fp_l{layer}_{tag}"
                f"[{HIDDEN_DIM}] row_align_acc(1) = {float_vec(b)};"
            )
        L.append("")

        # Attention output LayerNorm: gamma=elem_t (scale), beta=acc_t (offset)
        attn_ln_g = load(f"{pfx}_attention_output_LayerNorm_weight")
        attn_ln_b = load(f"{pfx}_attention_output_LayerNorm_bias")
        L.append(f"static const elem_t bert_fp_l{layer}_attn_ln_gamma[{HIDDEN_DIM}] row_align(1)     = {float_vec(attn_ln_g)};")
        L.append(f"static const acc_t  bert_fp_l{layer}_attn_ln_beta [{HIDDEN_DIM}] row_align_acc(1) = {float_vec(attn_ln_b)};")
        L.append("")

        # FF1: elem_t weight [out=512][in=128], acc_t bias [512]
        ff1 = load(f"{pfx}_intermediate_dense_weight")
        L.append(
            f"static const elem_t bert_fp_l{layer}_ff1_w"
            f"[{EXPANSION_DIM}][{HIDDEN_DIM}] row_align(1) = {float_mat(ff1)};"
        )
        ff1_b = load(f"{pfx}_intermediate_dense_bias")
        L.append(
            f"static const acc_t bert_fp_l{layer}_ff1_b"
            f"[{EXPANSION_DIM}] row_align_acc(1) = {float_vec(ff1_b)};"
        )
        L.append("")

        # FF2: elem_t weight [out=128][in=512], acc_t bias [128]
        ff2 = load(f"{pfx}_output_dense_weight")
        L.append(
            f"static const elem_t bert_fp_l{layer}_ff2_w"
            f"[{HIDDEN_DIM}][{EXPANSION_DIM}] row_align(1) = {float_mat(ff2)};"
        )
        ff2_b = load(f"{pfx}_output_dense_bias")
        L.append(
            f"static const acc_t bert_fp_l{layer}_ff2_b"
            f"[{HIDDEN_DIM}] row_align_acc(1) = {float_vec(ff2_b)};"
        )
        L.append("")

        # FFN output LayerNorm: gamma=elem_t, beta=acc_t
        ffn_ln_g = load(f"{pfx}_output_LayerNorm_weight")
        ffn_ln_b = load(f"{pfx}_output_LayerNorm_bias")
        L.append(f"static const elem_t bert_fp_l{layer}_ffn_ln_gamma[{HIDDEN_DIM}] row_align(1)     = {float_vec(ffn_ln_g)};")
        L.append(f"static const acc_t  bert_fp_l{layer}_ffn_ln_beta [{HIDDEN_DIM}] row_align_acc(1) = {float_vec(ffn_ln_b)};")
        L.append("")

    # ── Pooler ────────────────────────────────────────────────────────────────
    L += [
        "/* ── Pooler {'─'*61}*/",
        "/*    elem_t weight [out=128][in=128], acc_t bias [128]            */",
        "",
    ]
    L.append(
        f"static const elem_t bert_fp_pooler_w[{HIDDEN_DIM}][{HIDDEN_DIM}] row_align(1) = "
        f"{float_mat(load('bert_pooler_dense_weight'))};"
    )
    L.append(
        f"static const acc_t bert_fp_pooler_b[{HIDDEN_DIM}] row_align_acc(1) = "
        f"{float_vec(load('bert_pooler_dense_bias'))};"
    )
    L.append("")

    # ── Classifier ────────────────────────────────────────────────────────────
    L += [
        "/* ── Classifier {'─'*57}*/",
        "/*    elem_t weight [out=2][in=128], acc_t bias [2]                */",
        "",
    ]
    L.append(
        f"static const elem_t bert_fp_clf_w[{NUM_LABELS}][{HIDDEN_DIM}] row_align(1) = "
        f"{float_mat(load('classifier_weight'))};"
    )
    L.append(
        f"static const acc_t bert_fp_clf_b[{NUM_LABELS}] row_align_acc(1) = "
        f"{float_vec(load('classifier_bias'))};"
    )
    L.append("")
    L.append("#endif /* BERT_PARAMS_FP_H */")
    L.append("")

    with open(FP_PARAMS_PATH, "w") as f:
        f.write("\n".join(L))

    total_elem = NUM_LAYERS * (4 * HIDDEN_DIM * HIDDEN_DIM
                               + EXPANSION_DIM * HIDDEN_DIM
                               + HIDDEN_DIM * EXPANSION_DIM)
    print(f"[OK] {FP_PARAMS_PATH}")
    print(f"     {total_elem:,} elem_t values (float32 originals) — for float Gemmini hardware")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating BERT-Tiny C headers from .npy weights...\n")
    gen_params()
    gen_input()
    gen_params_fp()
    gen_input()
    print("\nDone.  Headers generated:")
    print(f"  {PARAMS_PATH}")
    print(f"  {FP_PARAMS_PATH}")
    print(f"  {INPUT_PATH}")
    print("\nBuild — Gemmini int8 path:")
    print("  riscv64-unknown-elf-gcc -O2 -DBAREMETAL -Iinclude \\")
    print("      transformers/bert-tiny-sst2.c -o bert-tiny-sst2-baremetal -lm")
    print("\nBuild — CPU float32 reference path:")
    print("  riscv64-unknown-elf-gcc -O2 -DBAREMETAL -Iinclude \\")
    print("      transformers/bert-tiny-sst2-fp.c -o bert-tiny-sst2-fp-baremetal -lm")
