#!/usr/bin/env python3
"""
sim_int8_v3.py — Fixed int8 simulation with:
  1. LN output divided by output_scale (so int8 uses full range)
  2. Calibrated ACC_SCALE for intermediate matmuls
  3. Scale-aware LN residual
  4. Scaled GELU
  5. Correct BERT post-LN order

The critical bug was sw_layernorm rounding gamma*z+beta directly to int8
(giving values in [-3,3]) instead of dividing by output_scale first
(giving values in [-50,50]).
"""
import numpy as np
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
DATA_DIR    = os.path.join(SCRIPT_DIR, "..")

def load(name):
    return np.load(os.path.join(WEIGHTS_DIR, name + ".npy")).astype(np.float32)

HIDDEN = 128; EXPANSION = 512; HEADS = 2; SEQ = 128; HEAD_DIM = 64

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

emb_data  = np.fromfile(os.path.join(DATA_DIR, "sst2_validation_872.bin"),
                        dtype=np.int8).reshape(-1, SEQ, HIDDEN)
mask_data = np.fromfile(os.path.join(DATA_DIR, "sst2_validation_872_attn_masks.bin"),
                        dtype=np.uint8).reshape(-1, SEQ)
with open(os.path.join(DATA_DIR, "sst2_validation_872_labels.txt")) as f:
    labels = [int(x) for x in f.read().split()]
N = len(labels)

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
# CALIBRATION PASS (float with int8 weights → output ranges)
# ══════════════════════════════════════════════════════════════════════════════
print("=== CALIBRATION PASS ===", flush=True)
calib = {k: [0.0, 0.0] for k in ["qkv", "scores", "ctx", "ff1", "ff1_gelu"]}

for ex in range(N):
    cur_real = emb_data[ex].astype(np.float64) * emb_scale
    mask = mask_data[ex]

    for L in range(2):
        wd = layer_weights[L]; qd = q_layers[L]
        Wq_f = qd["Wq_q"].astype(np.float64)*qd["Wq_s"]
        Wk_f = qd["Wk_q"].astype(np.float64)*qd["Wk_s"]
        Wv_f = qd["Wv_q"].astype(np.float64)*qd["Wv_s"]
        Q = cur_real @ Wq_f + wd["Wq_b"]
        K = cur_real @ Wk_f + wd["Wk_b"]
        V = cur_real @ Wv_f + wd["Wv_b"]
        calib["qkv"][L] = max(calib["qkv"][L], abs(Q).max(), abs(K).max(), abs(V).max())

        ctx_full = np.zeros((SEQ, HIDDEN), np.float64)
        for h in range(HEADS):
            sl = slice(h*HEAD_DIM, (h+1)*HEAD_DIM)
            scores = Q[:,sl] @ K[:,sl].T
            calib["scores"][L] = max(calib["scores"][L], abs(scores).max())
            sc = scores.copy(); sc[:,mask==0] = -1e9
            mx = sc.max(1,keepdims=True); e = np.exp(sc-mx)
            soft = e / e.sum(1,keepdims=True)
            ch = soft @ V[:,sl]
            calib["ctx"][L] = max(calib["ctx"][L], abs(ch).max())
            ctx_full[:,sl] = ch

        Wo_f = qd["Wo_q"].astype(np.float64)*qd["Wo_s"]
        Wo_out = ctx_full @ Wo_f + wd["Wo_b"]
        combined = Wo_out + cur_real
        m = combined.mean(1,keepdims=True); v = ((combined-m)**2).mean(1,keepdims=True)
        attn_out = (combined-m)/np.sqrt(v+1e-12)*wd["attn_ln_gamma"]+wd["attn_ln_beta"]

        ff1_f = qd["ff1_q"].astype(np.float64)*qd["ff1_s"]
        ff1_out = attn_out @ ff1_f + wd["ff1_b"]
        calib["ff1"][L] = max(calib["ff1"][L], abs(ff1_out).max())
        inner = 0.7978845608*(ff1_out + 0.044715*ff1_out**3)
        ff1_gelu = 0.5*ff1_out*(1+np.tanh(inner))
        calib["ff1_gelu"][L] = max(calib["ff1_gelu"][L], abs(ff1_gelu).max())

        ff2_f = qd["ff2_q"].astype(np.float64)*qd["ff2_s"]
        ff2_out = ff1_gelu @ ff2_f + wd["ff2_b"]
        combined = ff2_out + attn_out
        m = combined.mean(1,keepdims=True); v = ((combined-m)**2).mean(1,keepdims=True)
        cur_real = (combined-m)/np.sqrt(v+1e-12)*wd["ffn_ln_gamma"]+wd["ffn_ln_beta"]
    if (ex+1)%200==0: print(f"  Calibrated {ex+1}/{N}", flush=True)

y_sc = {}
for L in range(2):
    for k in calib: y_sc[f"{k}_{L}"] = calib[k][L] / 127.0

print("\nCalibrated y_scales:")
for k,v in sorted(y_sc.items()): print(f"  {k:15s} = {v:.6f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE PASS — INT8 WITH ALL FIXES
# ══════════════════════════════════════════════════════════════════════════════
print("=== INT8 INFERENCE (all fixes) ===", flush=True)

def ln_residual_fixed(acc_i32, res_i8, acc_to_real, res_to_real, gamma, beta, output_scale):
    """Post-LN with:
      1. Scale-aware accumulator + residual addition
      2. LN output divided by output_scale for proper int8 quantization
    """
    combined = acc_i32.astype(np.float64)*acc_to_real + res_i8.astype(np.float64)*res_to_real
    m = combined.mean(1, keepdims=True)
    v = ((combined - m)**2).mean(1, keepdims=True)
    normed = (combined - m) / np.sqrt(v + 1e-12)
    scaled = normed * gamma.astype(np.float64) + beta.astype(np.float64)
    # CRITICAL FIX: divide by output_scale to use full int8 range
    quantized = scaled / output_scale
    adj = np.where(quantized > 0, quantized + 0.5, quantized - 0.5)
    return np.clip(adj.astype(np.int64), -128, 127).astype(np.int8)

correct = 0
for ex in range(N):
    cur = emb_data[ex].copy()  # int8, at scale emb_scale
    mask = mask_data[ex]
    cur_x_scale = emb_scale    # current activation scale

    for L in range(2):
        wd = layer_weights[L]; qd = q_layers[L]
        yq = y_sc[f"qkv_{L}"];   ys = y_sc[f"scores_{L}"]
        yc = y_sc[f"ctx_{L}"];   yf = y_sc[f"ff1_{L}"]
        yfg = y_sc[f"ff1_gelu_{L}"]

        # ── QKV projections with ACC_SCALE ────────────────────────────────
        QKV = []
        for tag in ["Wq", "Wk", "Wv"]:
            ws = qd[tag+"_s"]
            bi = np.round(wd[tag+"_b"]/(cur_x_scale*ws)).clip(-(2**31),2**31-1).astype(np.int64)
            acc = cur.astype(np.int32) @ qd[tag+"_q"].astype(np.int32) + bi.astype(np.int32)
            acc_scale = cur_x_scale * ws / yq
            out = np.clip(np.round(acc.astype(np.float64)*acc_scale), -128, 127).astype(np.int8)
            QKV.append(out)
        Q_i, K_i, V_i = QKV  # all at scale yq

        # ── Attention scores and context per head ─────────────────────────
        ctx_full = np.zeros((SEQ, HIDDEN), np.int8)
        for h in range(HEADS):
            sl = slice(h*HEAD_DIM, (h+1)*HEAD_DIM)
            # Q×K^T
            sc_acc = Q_i[:,sl].astype(np.int32) @ K_i[:,sl].astype(np.int32).T
            sc_i = np.clip(np.round(sc_acc.astype(np.float64)*yq*yq/ys), -128, 127).astype(np.int8)
            # softmax
            sf = sc_i.astype(np.float32); sf[:,mask==0] = -128.0
            mx = sf.max(1,keepdims=True); e = np.exp(sf-mx)
            soft = np.floor(e/e.sum(1,keepdims=True)*127+0.5).clip(-128,127).astype(np.int8)
            # softmax×V  (softmax scale = 1/127, V scale = yq)
            ct_acc = soft.astype(np.int32) @ V_i[:,sl].astype(np.int32)
            ctx_full[:,sl] = np.clip(np.round(ct_acc.astype(np.float64)*(1.0/127.0)*yq/yc), -128, 127).astype(np.int8)
        # ctx_full at scale yc

        # ── Wo projection (full_C=true → int32 accumulator) ──────────────
        wo_bi = np.round(wd["Wo_b"]/(yc*qd["Wo_s"])).clip(-(2**31),2**31-1).astype(np.int64)
        wo_acc = ctx_full.astype(np.int32) @ qd["Wo_q"].astype(np.int32) + wo_bi.astype(np.int32)

        # ── Post-LN with all fixes ───────────────────────────────────────
        attn_ln_scale = estimate_ln_out_scale(wd["attn_ln_gamma"], wd["attn_ln_beta"])
        attn_out = ln_residual_fixed(wo_acc, cur,
                                     yc * qd["Wo_s"],     # acc → real
                                     cur_x_scale,          # res → real
                                     wd["attn_ln_gamma"], wd["attn_ln_beta"],
                                     attn_ln_scale)
        # attn_out at scale attn_ln_scale

        # ── FF1 with ACC_SCALE ───────────────────────────────────────────
        ff1_bi = np.round(wd["ff1_b"]/(attn_ln_scale*qd["ff1_s"])).clip(-(2**31),2**31-1).astype(np.int64)
        ff1_acc = attn_out.astype(np.int32) @ qd["ff1_q"].astype(np.int32) + ff1_bi.astype(np.int32)
        ff1_i = np.clip(np.round(ff1_acc.astype(np.float64)*attn_ln_scale*qd["ff1_s"]/yf), -128, 127).astype(np.int8)
        # ff1_i at scale yf

        # ── Scaled GELU ──────────────────────────────────────────────────
        x_real = ff1_i.astype(np.float64) * yf
        inner = 0.7978845608 * (x_real + 0.044715 * x_real**3)
        g_real = 0.5 * x_real * (1.0 + np.tanh(inner))
        gelu_i = np.clip(np.round(g_real / yfg), -128, 127).astype(np.int8)
        # gelu_i at scale yfg

        # ── FF2 (full_C=true → int32 accumulator) ────────────────────────
        ff2_bi = np.round(wd["ff2_b"]/(yfg*qd["ff2_s"])).clip(-(2**31),2**31-1).astype(np.int64)
        ff2_acc = gelu_i.astype(np.int32) @ qd["ff2_q"].astype(np.int32) + ff2_bi.astype(np.int32)

        # ── Post-LN with all fixes ───────────────────────────────────────
        ffn_ln_scale = estimate_ln_out_scale(wd["ffn_ln_gamma"], wd["ffn_ln_beta"])
        cur = ln_residual_fixed(ff2_acc, attn_out,
                                yfg * qd["ff2_s"],        # acc → real
                                attn_ln_scale,              # res → real
                                wd["ffn_ln_gamma"], wd["ffn_ln_beta"],
                                ffn_ln_scale)
        cur_x_scale = ffn_ln_scale

    # ── Pooler + classifier (float) ──────────────────────────────────────
    # The CLS token int8 is at scale cur_x_scale.
    # The pooler weights expect float input, so dequantize to real values.
    cls_real = cur[0].astype(np.float32) * cur_x_scale
    pooled = np.tanh(pooler_w @ cls_real + pooler_b)
    logits = clf_w @ pooled + clf_b
    pred = 1 if logits[1] > logits[0] else 0
    if pred == labels[ex]: correct += 1
    if (ex+1)%100==0 or (ex+1)==N:
        print(f"  [{ex+1}/{N}] acc = {100*correct/(ex+1):.2f}%", flush=True)

print(f"\n=== FIXED INT8 ACCURACY: {100*correct/N:.2f}% ({correct}/{N}) ===", flush=True)
