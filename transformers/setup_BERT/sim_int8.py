#!/usr/bin/env python3
"""
Full int8 pipeline simulation matching bert-tiny-sst2-stream.c exactly.
This replicates the C code's dataflow to verify quantization correctness.
"""
import numpy as np
import os, sys

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def load(name):
    return np.load(os.path.join(WEIGHTS_DIR, name + ".npy")).astype(np.float32)

HIDDEN = 128; EXPANSION = 512; HEADS = 2; SEQ = 128; HEAD_DIM = HIDDEN // HEADS

# ── Load all original float weights ──────────────────────────────────────────
emb_ln_gamma = load("bert_embeddings_LayerNorm_weight")
emb_ln_beta  = load("bert_embeddings_LayerNorm_bias")

layer_weights = []
for L in range(2):
    pfx = f"bert_encoder_layer_{L}"
    d = {}
    for tag, stem in [("Wq", "attention_self_query_weight"),
                      ("Wk", "attention_self_key_weight"),
                      ("Wv", "attention_self_value_weight"),
                      ("Wo", "attention_output_dense_weight")]:
        W_pt = load(f"{pfx}_{stem}")
        d[tag] = W_pt.T  # PyTorch [out,in] → gemmini [in,out]
    for tag, stem in [("Wq_b", "attention_self_query_bias"),
                      ("Wk_b", "attention_self_key_bias"),
                      ("Wv_b", "attention_self_value_bias"),
                      ("Wo_b", "attention_output_dense_bias")]:
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

pooler_w = load("bert_pooler_dense_weight")  # [out=128, in=128] PyTorch layout
pooler_b = load("bert_pooler_dense_bias")
clf_w    = load("classifier_weight")         # [2, 128]
clf_b    = load("classifier_bias")           # [2]

# ── Quantize weights exactly as generate_bert_params.py does ─────────────────
def quantize_weight(W):
    max_abs = max(float(np.max(np.abs(W))), 1e-6)
    scale = max_abs / 127.0
    W_q = np.round(W / scale).clip(-128, 127).astype(np.int8)
    return W_q, scale

def quantize_bias(b, w_scale, in_scale):
    denom = w_scale * in_scale
    return np.round(b / denom).clip(-(2**31), 2**31-1).astype(np.int64)

def estimate_emb_scale():
    word_emb = load("bert_embeddings_word_embeddings_weight")
    pos_emb  = load("bert_embeddings_position_embeddings_weight")
    type_emb = load("bert_embeddings_token_type_embeddings_weight")
    emb = word_emb + pos_emb[0:1] + type_emb[0:1]
    mean = emb.mean(axis=-1, keepdims=True)
    var  = emb.var(axis=-1, keepdims=True)
    emb_normed = (emb - mean) / np.sqrt(var + 1e-12)
    emb_out = emb_normed * emb_ln_gamma + emb_ln_beta
    return max(float(np.max(np.abs(emb_out))), 1e-6) / 127.0

def estimate_ln_out_scale(gamma, beta):
    return max(float(np.max(np.abs(beta) + 5.0*np.abs(gamma))), 1e-6) / 127.0

# Build quantized weight/bias tables
q_layers = []
cur_x_scale = estimate_emb_scale()
print(f"[INFO] embedding x_scale = {cur_x_scale:.6f}")

for L in range(2):
    d = layer_weights[L]
    qd = {}
    for tag in ["Wq", "Wk", "Wv", "Wo"]:
        qd[tag+"_q"], qd[tag+"_s"] = quantize_weight(d[tag])
    qd["ff1_q"], qd["ff1_s"] = quantize_weight(d["ff1_w"])
    qd["ff2_q"], qd["ff2_s"] = quantize_weight(d["ff2_w"])
    
    wo_x_scale = 1.0/127.0
    for tag in ["Wq_b", "Wk_b", "Wv_b"]:
        w_tag = tag[:-2]
        qd[tag] = quantize_bias(d[tag], qd[w_tag+"_s"], cur_x_scale)
    qd["Wo_b"] = quantize_bias(d["Wo_b"], qd["Wo_s"], wo_x_scale)
    
    attn_ln_x_scale = estimate_ln_out_scale(d["attn_ln_gamma"], d["attn_ln_beta"])
    qd["ff1_b"] = quantize_bias(d["ff1_b"], qd["ff1_s"], attn_ln_x_scale)
    
    ff2_x_scale = 1.0/127.0
    qd["ff2_b"] = quantize_bias(d["ff2_b"], qd["ff2_s"], ff2_x_scale)
    
    print(f"[INFO] layer {L}: attn_x={cur_x_scale:.6f} wo_x={wo_x_scale:.6f} "
          f"ffn_x={attn_ln_x_scale:.6f} ff2_x={ff2_x_scale:.6f}")
    
    cur_x_scale = estimate_ln_out_scale(d["ffn_ln_gamma"], d["ffn_ln_beta"])
    q_layers.append(qd)

# ── Simulate the C code's int8 pipeline ──────────────────────────────────────
def sat8(v):
    return int(max(-128, min(127, round(v))))

def int8_matmul(A_i8, B_i8, bias_i64, out_is_acc=False):
    """Replicate tiled_matmul_auto with ACC_SCALE_IDENTITY=1.0"""
    C = A_i8.astype(np.int32) @ B_i8.astype(np.int32)
    if bias_i64 is not None:
        C = C + bias_i64.astype(np.int32)
    if out_is_acc:
        return C  # keep int32 for layernorm input
    return np.clip(C, -128, 127).astype(np.int8)

def int8_matmul_Bt(A_i8, B_i8):
    """A @ B^T → int8 (for Q*K^T)"""
    C = A_i8.astype(np.int32) @ B_i8.astype(np.int32).T
    return np.clip(C, -128, 127).astype(np.int8)

def sw_softmax_masked(mat_i8, mask):
    rows, cols = mat_i8.shape
    out = np.zeros_like(mat_i8)
    for i in range(rows):
        row = mat_i8[i].astype(np.float32).copy()
        if mask is not None:
            row[mask == 0] = -128.0
        mx = row.max()
        e = np.exp(row - mx)
        s = e.sum()
        for j in range(cols):
            q = int(e[j] / s * 127.0 + 0.5)
            out[i, j] = max(-128, min(127, q))
    return out.astype(np.int8)

def sw_gelu(mat_i8):
    sqrt_2_over_pi = 0.7978845608
    out = np.zeros_like(mat_i8)
    rows, cols = mat_i8.shape
    for i in range(rows):
        for j in range(cols):
            x = float(mat_i8[i, j])
            inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
            g = 0.5 * x * (1.0 + float(np.tanh(inner)))
            out[i, j] = sat8(g + 0.5 if g > 0 else g - 0.5)
    return out.astype(np.int8)

def sw_layernorm(in_i32, gamma, beta):
    rows, cols = in_i32.shape
    out = np.zeros((rows, cols), dtype=np.int8)
    for i in range(rows):
        row = in_i32[i].astype(np.float64)
        mean = row.mean()
        var = ((row - mean)**2).mean()
        inv_std = 1.0 / np.sqrt(var + 1e-12)
        for j in range(cols):
            normed = (float(row[j]) - mean) * inv_std
            scaled = float(gamma[j]) * normed + float(beta[j])
            out[i, j] = sat8(scaled + 0.5 if scaled > 0 else scaled - 0.5)
    return out

def resadd(A_i8, B_i8):
    return np.clip(A_i8.astype(np.int32) + B_i8.astype(np.int32), -128, 127).astype(np.int8)

# ── Load SST-2 data ─────────────────────────────────────────────────────────
emb_data = np.fromfile(os.path.join(DATA_DIR, "sst2_validation_872.bin"),
                       dtype=np.int8).reshape(-1, SEQ, HIDDEN)
mask_data = np.fromfile(os.path.join(DATA_DIR, "sst2_validation_872_attn_masks.bin"),
                        dtype=np.uint8).reshape(-1, SEQ)
with open(os.path.join(DATA_DIR, "sst2_validation_872_labels.txt")) as f:
    labels = [int(x) for x in f.read().split()]

NUM_EXAMPLES = len(labels)
print(f"\nLoaded {NUM_EXAMPLES} examples")

correct = 0
wrong_examples = []

for ex in range(NUM_EXAMPLES):
    cur = emb_data[ex].copy()
    mask = mask_data[ex]
    
    for L in range(2):
        qd = q_layers[L]
        wd = layer_weights[L]
        
        # Attention: Q, K, V projections
        Q = int8_matmul(cur, qd["Wq_q"], qd["Wq_b"])
        K = int8_matmul(cur, qd["Wk_q"], qd["Wk_b"])
        V = int8_matmul(cur, qd["Wv_q"], qd["Wv_b"])
        
        # Per-head attention
        context = np.zeros((SEQ, HIDDEN), dtype=np.int8)
        for h in range(HEADS):
            sl = slice(h*HEAD_DIM, (h+1)*HEAD_DIM)
            Q_h = Q[:, sl]
            K_h = K[:, sl]
            V_h = V[:, sl]
            
            scores_i8 = int8_matmul_Bt(Q_h, K_h)
            scores_soft = sw_softmax_masked(scores_i8, mask)
            ctx_h = int8_matmul(scores_soft, V_h, None)
            context[:, sl] = ctx_h
        
        # Output projection → acc_t (int32)
        proj_acc = int8_matmul(context, qd["Wo_q"], qd["Wo_b"], out_is_acc=True)
        
        # LayerNorm + residual
        attn_ln_out = sw_layernorm(proj_acc, wd["attn_ln_gamma"], wd["attn_ln_beta"])
        attn_resadd = resadd(cur, attn_ln_out)
        
        # FFN
        ff1_out = int8_matmul(attn_resadd, qd["ff1_q"], qd["ff1_b"])
        ff1_gelu = sw_gelu(ff1_out)
        
        ff2_acc = int8_matmul(ff1_gelu, qd["ff2_q"], qd["ff2_b"], out_is_acc=True)
        ffn_ln_out = sw_layernorm(ff2_acc, wd["ffn_ln_gamma"], wd["ffn_ln_beta"])
        
        cur = resadd(ffn_ln_out, attn_resadd)
    
    # Pooler + Classifier (float, same as C code)
    cls_token = cur[0].astype(np.float32)
    pooled = np.tanh(pooler_w @ cls_token + pooler_b)
    logits = clf_w @ pooled + clf_b
    
    pred = 1 if logits[1] > logits[0] else 0
    label = labels[ex]
    if pred == label:
        correct += 1
    else:
        if len(wrong_examples) < 5:
            wrong_examples.append((ex, label, pred, float(logits[0]), float(logits[1])))
    
    if (ex+1) % 100 == 0 or (ex+1) == NUM_EXAMPLES:
        print(f"  [{ex+1}/{NUM_EXAMPLES}] accuracy = {100*correct/(ex+1):.2f}% ({correct}/{ex+1})")

print(f"\n=== FINAL INT8 ACCURACY: {100*correct/NUM_EXAMPLES:.2f}% ({correct}/{NUM_EXAMPLES}) ===")
print(f"\nFirst wrong examples:")
for ex, label, pred, l0, l1 in wrong_examples:
    print(f"  ex={ex}: label={label} pred={pred} logits=[{l0:.4f}, {l1:.4f}]")
