#!/usr/bin/env python3
"""
Vectorized int8 pipeline simulation matching bert-tiny-sst2-stream.c exactly.
"""
import numpy as np
import os, sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
DATA_DIR    = os.path.join(SCRIPT_DIR, "..")

def load(name):
    return np.load(os.path.join(WEIGHTS_DIR, name + ".npy")).astype(np.float32)

HIDDEN = 128; EXPANSION = 512; HEADS = 2; SEQ = 128; HEAD_DIM = HIDDEN // HEADS

# ── Load weights ─────────────────────────────────────────────────────────────
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
        d[tag] = load(f"{pfx}_{stem}").T  # [in, out]
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

pooler_w = load("bert_pooler_dense_weight")
pooler_b = load("bert_pooler_dense_bias")
clf_w    = load("classifier_weight")
clf_b    = load("classifier_bias")

# ── Quantize ─────────────────────────────────────────────────────────────────
def quantize_weight(W):
    s = max(float(np.max(np.abs(W))), 1e-6) / 127.0
    return np.round(W / s).clip(-128, 127).astype(np.int8), s

def quantize_bias(b, w_s, in_s):
    return np.round(b / (w_s * in_s)).clip(-(2**31), 2**31-1).astype(np.int64)

def estimate_emb_scale():
    word_emb = load("bert_embeddings_word_embeddings_weight")
    pos_emb  = load("bert_embeddings_position_embeddings_weight")
    type_emb = load("bert_embeddings_token_type_embeddings_weight")
    emb = word_emb + pos_emb[0:1] + type_emb[0:1]
    mean = emb.mean(axis=-1, keepdims=True)
    var  = emb.var(axis=-1, keepdims=True)
    out = (emb - mean) / np.sqrt(var + 1e-12) * emb_ln_gamma + emb_ln_beta
    return max(float(np.max(np.abs(out))), 1e-6) / 127.0

def estimate_ln_out_scale(g, b):
    return max(float(np.max(np.abs(b) + 5.0*np.abs(g))), 1e-6) / 127.0

q_layers = []
cur_x_scale = estimate_emb_scale()
print(f"[INFO] emb x_scale = {cur_x_scale:.6f}", flush=True)

for L in range(2):
    d = layer_weights[L]
    qd = {}
    for tag in ["Wq", "Wk", "Wv", "Wo"]:
        qd[tag+"_q"], qd[tag+"_s"] = quantize_weight(d[tag])
    qd["ff1_q"], qd["ff1_s"] = quantize_weight(d["ff1_w"])
    qd["ff2_q"], qd["ff2_s"] = quantize_weight(d["ff2_w"])
    
    wo_x = 1.0/127.0
    for tag in ["Wq_b", "Wk_b", "Wv_b"]:
        qd[tag] = quantize_bias(d[tag], qd[tag[:-2]+"_s"], cur_x_scale)
    qd["Wo_b"] = quantize_bias(d["Wo_b"], qd["Wo_s"], wo_x)
    
    aln_x = estimate_ln_out_scale(d["attn_ln_gamma"], d["attn_ln_beta"])
    qd["ff1_b"] = quantize_bias(d["ff1_b"], qd["ff1_s"], aln_x)
    qd["ff2_b"] = quantize_bias(d["ff2_b"], qd["ff2_s"], 1.0/127.0)
    
    cur_x_scale = estimate_ln_out_scale(d["ffn_ln_gamma"], d["ffn_ln_beta"])
    q_layers.append(qd)

# ── Vectorized C-equivalent operations ───────────────────────────────────────
def i8mm(A, B, bias=None, acc=False):
    """int8 matmul → int8 or int32"""
    C = A.astype(np.int32) @ B.astype(np.int32)
    if bias is not None:
        C += bias.astype(np.int32)
    return C if acc else np.clip(C, -128, 127).astype(np.int8)

def i8mm_Bt(A, B):
    """A @ B^T → int8"""
    C = A.astype(np.int32) @ B.astype(np.int32).T
    return np.clip(C, -128, 127).astype(np.int8)

def softmax_masked(m, mask):
    """Vectorized masked softmax matching C code"""
    f = m.astype(np.float32)
    if mask is not None:
        f[:, mask == 0] = -128.0
    mx = f.max(axis=1, keepdims=True)
    e = np.exp(f - mx)
    s = e.sum(axis=1, keepdims=True)
    q = np.floor(e / s * 127.0 + 0.5).clip(-128, 127).astype(np.int8)
    return q

def gelu_i8(m):
    """Vectorized GELU matching C code"""
    x = m.astype(np.float32)
    inner = 0.7978845608 * (x + 0.044715 * x**3)
    g = 0.5 * x * (1.0 + np.tanh(inner))
    # C code: g > 0 ? g+0.5 : g-0.5
    adj = np.where(g > 0, g + 0.5, g - 0.5)
    return np.clip(np.floor(adj + 0.5 - (adj > 0).astype(float) * 0.5 + (adj > 0).astype(float) * 0.5), -128, 127).astype(np.int8)

def gelu_i8_v2(m):
    """More careful: replicate sat_elem((int32_t)(g>0?g+0.5:g-0.5))"""
    x = m.astype(np.float64)
    inner = 0.7978845608 * (x + 0.044715 * x**3)
    g = 0.5 * x * (1.0 + np.tanh(inner))
    # (int32_t) cast truncates toward zero in C, but with +/-0.5 offset it's equivalent to round
    adjusted = np.where(g > 0, g + 0.5, g - 0.5)
    return np.clip(adjusted.astype(np.int32), -128, 127).astype(np.int8)

def layernorm_i32(inp, gamma, beta):
    """LayerNorm on int32 input → int8 output"""
    f = inp.astype(np.float64)
    mean = f.mean(axis=1, keepdims=True)
    var = ((f - mean)**2).mean(axis=1, keepdims=True)
    normed = (f - mean) / np.sqrt(var + 1e-12)
    scaled = normed * gamma.astype(np.float64) + beta.astype(np.float64)
    adjusted = np.where(scaled > 0, scaled + 0.5, scaled - 0.5)
    return np.clip(adjusted.astype(np.int32), -128, 127).astype(np.int8)

def resadd(a, b):
    return np.clip(a.astype(np.int32) + b.astype(np.int32), -128, 127).astype(np.int8)

# ── Load data ────────────────────────────────────────────────────────────────
emb_data = np.fromfile(os.path.join(DATA_DIR, "sst2_validation_872.bin"),
                       dtype=np.int8).reshape(-1, SEQ, HIDDEN)
mask_data = np.fromfile(os.path.join(DATA_DIR, "sst2_validation_872_attn_masks.bin"),
                        dtype=np.uint8).reshape(-1, SEQ)
with open(os.path.join(DATA_DIR, "sst2_validation_872_labels.txt")) as f:
    labels = [int(x) for x in f.read().split()]

N = len(labels)
print(f"Loaded {N} examples", flush=True)

correct = 0
wrongs = []

for ex in range(N):
    cur = emb_data[ex].copy()
    mask = mask_data[ex]
    
    for L in range(2):
        qd = q_layers[L]
        wd = layer_weights[L]
        
        Q = i8mm(cur, qd["Wq_q"], qd["Wq_b"])
        K = i8mm(cur, qd["Wk_q"], qd["Wk_b"])
        V = i8mm(cur, qd["Wv_q"], qd["Wv_b"])
        
        ctx = np.zeros((SEQ, HIDDEN), dtype=np.int8)
        for h in range(HEADS):
            sl = slice(h*HEAD_DIM, (h+1)*HEAD_DIM)
            scores = i8mm_Bt(Q[:, sl], K[:, sl])
            soft = softmax_masked(scores, mask)
            ctx[:, sl] = i8mm(soft, V[:, sl])
        
        proj = i8mm(ctx, qd["Wo_q"], qd["Wo_b"], acc=True)
        aln = layernorm_i32(proj, wd["attn_ln_gamma"], wd["attn_ln_beta"])
        ares = resadd(cur, aln)
        
        f1 = i8mm(ares, qd["ff1_q"], qd["ff1_b"])
        f1g = gelu_i8_v2(f1)
        f2 = i8mm(f1g, qd["ff2_q"], qd["ff2_b"], acc=True)
        fln = layernorm_i32(f2, wd["ffn_ln_gamma"], wd["ffn_ln_beta"])
        cur = resadd(fln, ares)
    
    cls = cur[0].astype(np.float32)
    pooled = np.tanh(pooler_w @ cls + pooler_b)
    logits = clf_w @ pooled + clf_b
    
    pred = 1 if logits[1] > logits[0] else 0
    if pred == labels[ex]:
        correct += 1
    elif len(wrongs) < 5:
        wrongs.append((ex, labels[ex], pred, float(logits[0]), float(logits[1])))
    
    if (ex+1) % 100 == 0 or (ex+1) == N:
        print(f"  [{ex+1}/{N}] acc = {100*correct/(ex+1):.2f}% ({correct}/{ex+1})", flush=True)

print(f"\n=== INT8 ACCURACY: {100*correct/N:.2f}% ({correct}/{N}) ===", flush=True)
for e in wrongs:
    print(f"  ex={e[0]}: label={e[1]} pred={e[2]} logits=[{e[3]:.4f}, {e[4]:.4f}]")
