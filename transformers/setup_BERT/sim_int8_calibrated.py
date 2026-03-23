#!/usr/bin/env python3
"""
sim_int8_calibrated.py
======================
Full int8 simulation with calibrated ACC_SCALE and scale-aware LayerNorm residual.

Two passes:
  Pass 1 (Calibration): Float forward with int8 quantized weights → output ranges
  Pass 2 (Inference): Int8 forward with calibrated ACC_SCALE → accuracy

Fixes vs the original pipeline:
  1. Proper ACC_SCALE for QKV, Q×K^T, softmax×V, FF1 matmuls
  2. Scale-aware sw_layernorm_residual (convert acc & int8 to real before adding)
  3. Scaled GELU (applied to real values, not raw int8)
  4. Correct BERT post-LN: LayerNorm(projection + residual)
"""
import numpy as np
import os, sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
DATA_DIR    = os.path.join(SCRIPT_DIR, "..")

def load(name):
    return np.load(os.path.join(WEIGHTS_DIR, name + ".npy")).astype(np.float32)

HIDDEN = 128; EXPANSION = 512; HEADS = 2; SEQ = 128; HEAD_DIM = HIDDEN // HEADS

# ── Load weights ──────────────────────────────────────────────────────────────
emb_ln_gamma = load("bert_embeddings_LayerNorm_weight")
emb_ln_beta  = load("bert_embeddings_LayerNorm_bias")

layer_weights = []
for L in range(2):
    pfx = f"bert_encoder_layer_{L}"
    d = {}
    for tag, stem in [("Wq","attention_self_query_weight"),("Wk","attention_self_key_weight"),
                      ("Wv","attention_self_value_weight"),("Wo","attention_output_dense_weight")]:
        d[tag] = load(f"{pfx}_{stem}").T
    for tag, stem in [("Wq_b","attention_self_query_bias"),("Wk_b","attention_self_key_bias"),
                      ("Wv_b","attention_self_value_bias"),("Wo_b","attention_output_dense_bias")]:
        d[tag] = load(f"{pfx}_{stem}")
    d["ff1_w"] = load(f"{pfx}_intermediate_dense_weight").T
    d["ff1_b"] = load(f"{pfx}_intermediate_dense_bias")
    d["ff2_w"] = load(f"{pfx}_output_dense_weight").T
    d["ff2_b"] = load(f"{pfx}_output_dense_bias")
    d["attn_ln_gamma"] = load(f"{pfx}_attention_output_LayerNorm_weight")
    d["attn_ln_beta"]  = load(f"{pfx}_attention_output_LayerNorm_bias")
    d["ffn_ln_gamma"]  = load(f"{pfx}_output_LayerNorm_weight")
    d["ffn_ln_beta"]   = load(f"{pfx}_output_LayerNorm_bias")
    layer_weights.append(d)

pooler_w = load("bert_pooler_dense_weight")
pooler_b = load("bert_pooler_dense_bias")
clf_w    = load("classifier_weight")
clf_b    = load("classifier_bias")

# ── Quantize weights to int8 ─────────────────────────────────────────────────
def quantize_weight(W):
    s = max(float(np.max(np.abs(W))), 1e-6) / 127.0
    return np.round(W / s).clip(-128, 127).astype(np.int8), s

q_layers = []
for L in range(2):
    d = layer_weights[L]
    qd = {}
    for tag in ["Wq", "Wk", "Wv", "Wo"]:
        qd[tag+"_q"], qd[tag+"_s"] = quantize_weight(d[tag])
    qd["ff1_q"], qd["ff1_s"] = quantize_weight(d["ff1_w"])
    qd["ff2_q"], qd["ff2_s"] = quantize_weight(d["ff2_w"])
    q_layers.append(qd)

# ── Load data ─────────────────────────────────────────────────────────────────
emb_data  = np.fromfile(os.path.join(DATA_DIR, "sst2_validation_872.bin"),
                        dtype=np.int8).reshape(-1, SEQ, HIDDEN)
mask_data = np.fromfile(os.path.join(DATA_DIR, "sst2_validation_872_attn_masks.bin"),
                        dtype=np.uint8).reshape(-1, SEQ)
with open(os.path.join(DATA_DIR, "sst2_validation_872_labels.txt")) as f:
    labels = [int(x) for x in f.read().split()]
N = len(labels)

# ── Embedding scale ──────────────────────────────────────────────────────────
def estimate_emb_scale():
    word_emb = load("bert_embeddings_word_embeddings_weight")
    pos_emb  = load("bert_embeddings_position_embeddings_weight")
    type_emb = load("bert_embeddings_token_type_embeddings_weight")
    emb = word_emb + pos_emb[0:1] + type_emb[0:1]
    m = emb.mean(axis=-1, keepdims=True)
    v = emb.var(axis=-1, keepdims=True)
    out = (emb - m) / np.sqrt(v + 1e-12) * emb_ln_gamma + emb_ln_beta
    return max(float(np.max(np.abs(out))), 1e-6) / 127.0

def estimate_ln_out_scale(g, b):
    return max(float(np.max(np.abs(b) + 5.0*np.abs(g))), 1e-6) / 127.0

emb_scale = estimate_emb_scale()

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1: FLOAT CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("=== CALIBRATION PASS (float with int8 weights) ===", flush=True)

calib = {k: [0.0, 0.0] for k in ["qkv", "scores", "ctx", "ff1", "ff1_gelu"]}

for ex in range(N):
    cur_real = emb_data[ex].astype(np.float64) * emb_scale
    mask = mask_data[ex]

    for L in range(2):
        wd = layer_weights[L]
        qd = q_layers[L]

        # Dequantized weights
        Wq_f = qd["Wq_q"].astype(np.float64) * qd["Wq_s"]
        Wk_f = qd["Wk_q"].astype(np.float64) * qd["Wk_s"]
        Wv_f = qd["Wv_q"].astype(np.float64) * qd["Wv_s"]

        Q = cur_real @ Wq_f + wd["Wq_b"]
        K = cur_real @ Wk_f + wd["Wk_b"]
        V = cur_real @ Wv_f + wd["Wv_b"]
        calib["qkv"][L] = max(calib["qkv"][L], abs(Q).max(), abs(K).max(), abs(V).max())

        ctx_full = np.zeros((SEQ, HIDDEN), dtype=np.float64)
        for h in range(HEADS):
            sl = slice(h*HEAD_DIM, (h+1)*HEAD_DIM)
            scores = Q[:, sl] @ K[:, sl].T
            calib["scores"][L] = max(calib["scores"][L], abs(scores).max())

            sc = scores.copy()
            sc[:, mask == 0] = -1e9
            mx = sc.max(axis=1, keepdims=True)
            e = np.exp(sc - mx)
            soft = e / e.sum(axis=1, keepdims=True)

            ctx_h = soft @ V[:, sl]
            calib["ctx"][L] = max(calib["ctx"][L], abs(ctx_h).max())
            ctx_full[:, sl] = ctx_h

        Wo_f = qd["Wo_q"].astype(np.float64) * qd["Wo_s"]
        Wo_out = ctx_full @ Wo_f + wd["Wo_b"]

        combined = Wo_out + cur_real
        m = combined.mean(axis=1, keepdims=True)
        v = ((combined - m)**2).mean(axis=1, keepdims=True)
        attn_out_real = (combined - m) / np.sqrt(v + 1e-12) * wd["attn_ln_gamma"] + wd["attn_ln_beta"]

        ff1_f = qd["ff1_q"].astype(np.float64) * qd["ff1_s"]
        ff1_out = attn_out_real @ ff1_f + wd["ff1_b"]
        calib["ff1"][L] = max(calib["ff1"][L], abs(ff1_out).max())

        inner = 0.7978845608 * (ff1_out + 0.044715 * ff1_out**3)
        ff1_gelu = 0.5 * ff1_out * (1.0 + np.tanh(inner))
        calib["ff1_gelu"][L] = max(calib["ff1_gelu"][L], abs(ff1_gelu).max())

        ff2_f = qd["ff2_q"].astype(np.float64) * qd["ff2_s"]
        ff2_out = ff1_gelu @ ff2_f + wd["ff2_b"]
        combined = ff2_out + attn_out_real
        m = combined.mean(axis=1, keepdims=True)
        v = ((combined - m)**2).mean(axis=1, keepdims=True)
        cur_real = (combined - m) / np.sqrt(v + 1e-12) * wd["ffn_ln_gamma"] + wd["ffn_ln_beta"]

    if (ex+1) % 200 == 0:
        print(f"  Calibrated {ex+1}/{N}", flush=True)

# Compute y_scale for each step
y_scales = {}
for L in range(2):
    for k in calib:
        y_scales[f"{k}_{L}"] = calib[k][L] / 127.0

print("\nCalibrated output scales (y_scale = max_abs / 127):")
for k, v in sorted(y_scales.items()):
    print(f"  y_{k:15s} = {v:.6f}  (max_abs = {v*127:.1f})")
print()

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2: INT8 INFERENCE WITH CALIBRATED ACC_SCALE
# ══════════════════════════════════════════════════════════════════════════════
print("=== INT8 INFERENCE (calibrated ACC_SCALE) ===", flush=True)

def i8mm_with_scale(A_i8, B_i8, bias_float, acc_scale, x_scale, w_scale):
    """Int8 matmul with bias (re-quantized) and ACC_SCALE."""
    bias_i32 = np.round(bias_float / (x_scale * w_scale)).clip(-(2**31), 2**31-1).astype(np.int64)
    acc = A_i8.astype(np.int32) @ B_i8.astype(np.int32) + bias_i32.astype(np.int32)
    out = np.round(acc.astype(np.float64) * acc_scale)
    return np.clip(out, -128, 127).astype(np.int8)

def i8mm_nobias_scale(A_i8, B_i8, acc_scale, transpose_B=False):
    """Int8 matmul without bias, with ACC_SCALE."""
    if transpose_B:
        acc = A_i8.astype(np.int32) @ B_i8.astype(np.int32).T
    else:
        acc = A_i8.astype(np.int32) @ B_i8.astype(np.int32)
    return np.clip(np.round(acc.astype(np.float64) * acc_scale), -128, 127).astype(np.int8)

def i8mm_to_acc(A_i8, B_i8, bias_float, x_scale, w_scale):
    """Int8 matmul → int32 accumulator (full_C=true). No ACC_SCALE."""
    bias_i32 = np.round(bias_float / (x_scale * w_scale)).clip(-(2**31), 2**31-1).astype(np.int64)
    return A_i8.astype(np.int32) @ B_i8.astype(np.int32) + bias_i32.astype(np.int32)

def softmax_masked_i8(m_i8, mask):
    f = m_i8.astype(np.float32)
    if mask is not None:
        f[:, mask == 0] = -128.0
    mx = f.max(axis=1, keepdims=True)
    e = np.exp(f - mx)
    return np.floor(e / e.sum(axis=1, keepdims=True) * 127.0 + 0.5).clip(-128, 127).astype(np.int8)

def gelu_scaled_i8(x_i8, x_real_scale, y_real_scale):
    """GELU applied at correct real scale, then re-quantized to int8."""
    x = x_i8.astype(np.float64) * x_real_scale
    inner = 0.7978845608 * (x + 0.044715 * x**3)
    g = 0.5 * x * (1.0 + np.tanh(inner))
    out = np.round(g / y_real_scale)
    return np.clip(out, -128, 127).astype(np.int8)

def ln_residual_scaled(acc_i32, res_i8, acc_to_real, res_to_real, gamma, beta):
    """Scale-aware post-LN: LayerNorm(acc*acc_scale + res*res_scale)."""
    combined = acc_i32.astype(np.float64) * acc_to_real + res_i8.astype(np.float64) * res_to_real
    m = combined.mean(axis=1, keepdims=True)
    v = ((combined - m)**2).mean(axis=1, keepdims=True)
    normed = (combined - m) / np.sqrt(v + 1e-12)
    scaled = normed * gamma.astype(np.float64) + beta.astype(np.float64)
    adj = np.where(scaled > 0, scaled + 0.5, scaled - 0.5)
    return np.clip(adj.astype(np.int64), -128, 127).astype(np.int8)

correct = 0
for ex in range(N):
    cur = emb_data[ex].copy()
    mask = mask_data[ex]
    cur_x_scale = emb_scale

    for L in range(2):
        wd = layer_weights[L]
        qd = q_layers[L]

        y_qkv     = y_scales[f"qkv_{L}"]
        y_scores  = y_scales[f"scores_{L}"]
        y_ctx     = y_scales[f"ctx_{L}"]
        y_ff1     = y_scales[f"ff1_{L}"]
        y_ff1g    = y_scales[f"ff1_gelu_{L}"]

        # ── QKV projections with ACC_SCALE ────────────────────────────────────
        Q = i8mm_with_scale(cur, qd["Wq_q"], wd["Wq_b"],
                            cur_x_scale * qd["Wq_s"] / y_qkv,
                            cur_x_scale, qd["Wq_s"])
        K = i8mm_with_scale(cur, qd["Wk_q"], wd["Wk_b"],
                            cur_x_scale * qd["Wk_s"] / y_qkv,
                            cur_x_scale, qd["Wk_s"])
        V = i8mm_with_scale(cur, qd["Wv_q"], wd["Wv_b"],
                            cur_x_scale * qd["Wv_s"] / y_qkv,
                            cur_x_scale, qd["Wv_s"])
        # Q, K, V are int8 with scale y_qkv

        # ── Attention scores and context ──────────────────────────────────────
        ctx_full = np.zeros((SEQ, HIDDEN), dtype=np.int8)
        for h in range(HEADS):
            sl = slice(h*HEAD_DIM, (h+1)*HEAD_DIM)
            # Q×K^T: input scales are both y_qkv
            acc_scale_sc = y_qkv * y_qkv / y_scores
            scores = i8mm_nobias_scale(Q[:, sl], K[:, sl], acc_scale_sc, transpose_B=True)
            soft = softmax_masked_i8(scores, mask)
            # softmax output scale = 1/127
            # softmax×V: scales (1/127) × y_qkv
            acc_scale_ctx = (1.0/127.0) * y_qkv / y_ctx
            ctx_full[:, sl] = i8mm_nobias_scale(soft, V[:, sl], acc_scale_ctx)
        # ctx_full int8 with scale y_ctx

        # ── Wo projection (full_C=true) → int32 accumulator ─────────────────
        Wo_acc = i8mm_to_acc(ctx_full, qd["Wo_q"], wd["Wo_b"], y_ctx, qd["Wo_s"])
        # acc_to_real = y_ctx * wo_s;  res_to_real = cur_x_scale
        attn_out = ln_residual_scaled(Wo_acc, cur,
                                      y_ctx * qd["Wo_s"], cur_x_scale,
                                      wd["attn_ln_gamma"], wd["attn_ln_beta"])
        attn_ln_scale = estimate_ln_out_scale(wd["attn_ln_gamma"], wd["attn_ln_beta"])

        # ── FF1 with ACC_SCALE ───────────────────────────────────────────────
        acc_scale_ff1 = attn_ln_scale * qd["ff1_s"] / y_ff1
        ff1_out = i8mm_with_scale(attn_out, qd["ff1_q"], wd["ff1_b"],
                                  acc_scale_ff1, attn_ln_scale, qd["ff1_s"])
        # ff1_out int8 with scale y_ff1

        # ── Scaled GELU ──────────────────────────────────────────────────────
        ff1_gelu = gelu_scaled_i8(ff1_out, y_ff1, y_ff1g)
        # ff1_gelu int8 with scale y_ff1g

        # ── FF2 projection (full_C=true) → int32 accumulator ────────────────
        ff2_acc = i8mm_to_acc(ff1_gelu, qd["ff2_q"], wd["ff2_b"], y_ff1g, qd["ff2_s"])
        cur = ln_residual_scaled(ff2_acc, attn_out,
                                 y_ff1g * qd["ff2_s"], attn_ln_scale,
                                 wd["ffn_ln_gamma"], wd["ffn_ln_beta"])
        cur_x_scale = estimate_ln_out_scale(wd["ffn_ln_gamma"], wd["ffn_ln_beta"])

    # ── Pooler + classifier (float) ──────────────────────────────────────────
    cls = cur[0].astype(np.float32)
    pooled = np.tanh(pooler_w @ cls + pooler_b)
    logits = clf_w @ pooled + clf_b
    pred = 1 if logits[1] > logits[0] else 0
    if pred == labels[ex]:
        correct += 1
    if (ex+1) % 100 == 0 or (ex+1) == N:
        print(f"  [{ex+1}/{N}] acc = {100*correct/(ex+1):.2f}% ({correct}/{ex+1})", flush=True)

print(f"\n=== CALIBRATED INT8 ACCURACY: {100*correct/N:.2f}% ({correct}/{N}) ===", flush=True)
