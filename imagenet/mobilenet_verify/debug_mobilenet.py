#!/usr/bin/env python3
"""Minimal end-to-end diagnostic: conv_1 through FC, with calibration support."""
import os, numpy as np, torch
from transformers import MobileNetV2ForImageClassification

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BN_EPS = 0.001
IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float64)
CALIB_DIR = "/home/hansa/Downloads/Images200"

def fold_bn(w, g, b, m, v, eps=BN_EPS):
    inv = 1.0/np.sqrt(v+eps); s = g*inv
    shape = [w.shape[0]]+[1]*(w.ndim-1)
    return w*s.reshape(shape), b-g*m*inv

def get_bn(sd, pfx):
    return tuple(sd[f"{pfx}.{k}"].float().numpy().astype(np.float64) for k in
                 ["convolution.weight","normalization.weight","normalization.bias",
                  "normalization.running_mean","normalization.running_var"])

print("Loading model...")
model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
nc = sd["classifier.weight"].shape[0]  # 1001

# Layer definitions
ALL_LAYERS = [
    ("conv_1","mobilenet_v2.conv_stem.first_conv",3,2,1,False,True),
    ("conv_dw_2","mobilenet_v2.conv_stem.conv_3x3",3,1,1,True,True),
    ("conv_3","mobilenet_v2.conv_stem.reduce_1x1",1,1,0,False,False),
]
_idx=4
_dw_strides={0:2,2:2,5:2,12:2}
for _hi in range(16):
    _pfx=f"mobilenet_v2.layer.{_hi}"
    _ds=_dw_strides.get(_hi,1)
    ALL_LAYERS.append((f"conv_{_idx}",f"{_pfx}.expand_1x1",1,1,0,False,True)); _idx+=1
    ALL_LAYERS.append((f"conv_dw_{_idx}",f"{_pfx}.conv_3x3",3,_ds,1,True,True)); _idx+=1
    ALL_LAYERS.append((f"conv_{_idx}",f"{_pfx}.reduce_1x1",1,1,0,False,False)); _idx+=1
ALL_LAYERS.append(("conv_52","mobilenet_v2.conv_1x1",1,1,0,False,True))

RESIDUAL_SKIP = {"conv_9":"conv_6","conv_15":"conv_12","conv_18":"conv_15",
    "conv_24":"conv_21","conv_27":"conv_24","conv_30":"conv_27",
    "conv_36":"conv_33","conv_39":"conv_36","conv_45":"conv_42","conv_48":"conv_45"}

# ----- CALIBRATION -----
print("Running calibration on", CALIB_DIR, "...")
import cv2

layer_relu = {gn: relu for gn, pfx, k, s, p, dw, relu in ALL_LAYERS}
all_abs = {}

def make_hook(gn, has_relu):
    def hook(module, inp, out):
        x = out.detach().float()
        if has_relu:
            x = torch.relu(x)  # Just ReLU, NOT ReLU6 — let the actual range be captured
        vals = x.abs().flatten()
        if vals.numel() > 10000:
            idx = torch.randperm(vals.numel())[:10000]
            vals = vals[idx]
        if gn not in all_abs:
            all_abs[gn] = []
        all_abs[gn].append(vals.numpy())
    return hook

hooks = []
for gn, pfx, k, s, p, dw, relu in ALL_LAYERS:
    parts = pfx.split(".")
    mod = model
    for pt in parts:
        mod = getattr(mod, pt)
    bn_mod = mod.normalization
    h = bn_mod.register_forward_hook(make_hook(gn, relu))
    hooks.append(h)

files = sorted([f for f in os.listdir(CALIB_DIR) if f.lower().endswith((".jpeg",".jpg",".png"))])[:100]
with torch.no_grad():
    for i, fname in enumerate(files):
        img_bgr_c = cv2.imread(os.path.join(CALIB_DIR, fname))
        if img_bgr_c is None: continue
        img_c = cv2.resize(img_bgr_c, (224,224))
        img_rgb_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        img_f_c = img_rgb_c.astype(np.float64)/255.0
        for c in range(3): img_f_c[:,:,c] = (img_f_c[:,:,c]-IMAGENET_MEAN[c])/IMAGENET_STD[c]
        x = torch.tensor(img_f_c.transpose(2,0,1)[np.newaxis], dtype=torch.float32)
        model(x)

for h in hooks:
    h.remove()

calibrated = {}
for gn in all_abs:
    combined = np.concatenate(all_abs[gn])
    calibrated[gn] = float(np.percentile(combined, 99.99))

print(f"Calibrated {len(calibrated)} layers\n")

# ----- Float forward -----
import cv2
img_bgr = cv2.imread("/home/hansa/Downloads/Images200/ILSVRC2012_val_00000001.JPEG")
img = cv2.resize(img_bgr, (224,224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_f = img_rgb.astype(np.float64)/255.0
for c in range(3): img_f[:,:,c] = (img_f[:,:,c]-IMAGENET_MEAN[c])/IMAGENET_STD[c]
x_float = torch.tensor(img_f.transpose(2,0,1)[np.newaxis], dtype=torch.float32)

with torch.no_grad():
    out = model(x_float)
    float_logits = out.logits[0].numpy()

if nc == 1001:
    print(f"Float pred (skip bg): {np.argmax(float_logits[1:])}")
    print(f"Float logit range: [{float_logits.min():.2f}, {float_logits.max():.2f}]")

# INT8 forward - pixel-128
img_int8 = (img_rgb.astype(np.int16)-128).clip(-128,127).astype(np.int8)
x_in = img_int8.transpose(2,0,1)[np.newaxis]

def qw(w):
    mx = float(np.max(np.abs(w)))
    if mx < 1e-10: return np.zeros_like(w,dtype=np.int8), 1e-10
    s = mx/127.0
    return np.clip(np.round(w/s),-128,127).astype(np.int8), s

def qb(b, cs):
    if cs < 1e-10: return np.zeros_like(b,dtype=np.int32)
    return np.clip(np.round(b/cs),-(2**31),2**31-1).astype(np.int32)

def conv_op(x, w_int, b_int, os, kernel, stride, pad):
    N,C,H,W = x.shape
    if kernel==1 and stride==1:
        xf = x.reshape(N,C,H*W).transpose(0,2,1).reshape(N*H*W,C)
        oh=ow=H
    else:
        kH=kW=kernel
        OH=(H+2*pad-kH)//stride+1; OW=(W+2*pad-kW)//stride+1
        if pad>0: x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),constant_values=0)
        patches=np.zeros((N,OH,OW,kH*kW*C),dtype=x.dtype)
        for i in range(OH):
            for j in range(OW):
                p=x[:,:,i*stride:i*stride+kH,j*stride:j*stride+kW]
                patches[:,i,j,:]=p.transpose(0,2,3,1).reshape(N,-1)
        xf=patches.reshape(N*OH*OW,kH*kW*C); oh=OH; ow=OW
    acc=xf.astype(np.int32)@w_int.astype(np.int32)+b_int.reshape(1,-1).astype(np.int32)
    yf=np.clip(np.round(acc.astype(np.float64)*os),-128,127).astype(np.int8)
    return yf.reshape(N,oh,ow,w_int.shape[1]).transpose(0,3,1,2)

def dw_conv_op(x, w_int, b_int, os, stride, pad):
    N,C,H,W = x.shape; kH=kW=3
    OH=(H+2*pad-kH)//stride+1; OW=(W+2*pad-kW)//stride+1
    if pad>0: x_pad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),constant_values=0)
    else: x_pad=x
    acc=np.zeros((N,C,OH,OW),dtype=np.int64)
    x32=x_pad.astype(np.int32); w32=w_int.astype(np.int32)
    for ki in range(kH):
        for kj in range(kW):
            rows=np.arange(OH)*stride+ki; cols=np.arange(OW)*stride+kj
            acc+=x32[:,:,rows[:,None],cols[None,:]]*w32[:,ki,kj].reshape(1,C,1,1)
    acc+=b_int.reshape(1,C,1,1).astype(np.int64)
    return np.clip(np.round(acc.astype(np.float64)*os),-128,127).astype(np.int8)

# Run with calibrated y_ranges
print("\n--- INT8 forward with CALIBRATED ranges ---")
out_int = x_in.copy()
stored = {}
x_scale = 128.0/127.0
layer_y_range = {}

for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    cw, gamma, beta, bn_m, bn_v = get_bn(sd, pfx)
    wf, bf = fold_bn(cw, gamma, beta, bn_m, bn_v)

    if gn == "conv_1":
        no = (128.0/255.0-IMAGENET_MEAN)/IMAGENET_STD
        for c in range(3): bf += wf[:,c,:,:].sum(axis=(1,2))*no[c]
        for c in range(3): wf[:,c,:,:] /= (255.0*IMAGENET_STD[c])

    if dw: wg = wf.squeeze(1)
    else: wg = wf.reshape(wf.shape[0],-1).T

    # Use calibrated y_range
    yr = max(calibrated.get(gn, 1.0), 1.0)
    layer_y_range[gn] = yr

    wi, ws = qw(wg)
    bi = qb(bf, ws*x_scale)
    os_val = (ws*x_scale)/(yr/127.0)

    if dw: out_int = dw_conv_op(out_int, wi, bi, os_val, s, pad)
    else: out_int = conv_op(out_int, wi, bi, os_val, k, s, pad)

    if relu: out_int = np.maximum(out_int, np.int8(0))
    stored[gn] = out_int

    if gn in RESIDUAL_SKIP:
        skip_src = RESIDUAL_SKIP[gn]
        skip = stored[skip_src]
        rs = layer_y_range[skip_src]/yr
        out_int = np.clip(np.round(skip.astype(np.float64)*rs + out_int.astype(np.float64)),-128,127).astype(np.int8)
        stored[gn] = out_int

    nz = np.count_nonzero(out_int)
    tot = out_int.size
    pct = nz/tot*100
    print(f"  {gn:12s}: shape={str(out_int.shape):20s} range=[{out_int.min():4d},{out_int.max():4d}]  "
          f"nz={nz:7d}/{tot:7d}({pct:5.1f}%)  os={os_val:.4e}  yr={yr:.2f}  xs={x_scale:.6f}")

    x_scale = yr/127.0

# Global avg pool
avg = out_int.astype(np.float64).mean(axis=(2,3))
avg_int8 = np.clip(np.round(avg),-128,127).astype(np.int8)
print(f"\n  avg_pool: range=[{avg_int8.min()},{avg_int8.max()}]  nz={np.count_nonzero(avg_int8)}/1280")

# FC
fc_w = sd["classifier.weight"].numpy().astype(np.float64)
fc_b = sd["classifier.bias"].numpy().astype(np.float64)
if nc == 1001:
    fc_w = fc_w[1:,:]; fc_b = fc_b[1:]
fc_wi, fc_ws = qw(fc_w)
fc_bi = qb(fc_b, fc_ws*x_scale)

logits = fc_wi.astype(np.int32) @ avg_int8[0].astype(np.int32) + fc_bi.astype(np.int32)
pred = np.argmax(logits)
print(f"\n  FC pred={pred}  logits range=[{logits.min()},{logits.max()}]")
print(f"  Float pred={np.argmax(float_logits[1:]) if nc==1001 else np.argmax(float_logits)}")

# FC diagnostic: compute float logits from avg_int8
avg_float_from_int = avg_int8[0].astype(np.float64) * x_scale  # dequantize
logits_float_ref = fc_w @ avg_float_from_int + fc_b
pred_ref = np.argmax(logits_float_ref)
print(f"\n  Float FC (from dequantized avg): pred={pred_ref}")
print(f"  float logits range=[{logits_float_ref.min():.2f},{logits_float_ref.max():.2f}]")
print(f"  fc_ws={fc_ws:.6f}  x_scale={x_scale:.6f}  combined={fc_ws*x_scale:.8f}")
print(f"  fc_w max={np.max(np.abs(fc_w)):.4f}  fc_b range=[{fc_b.min():.4f},{fc_b.max():.4f}]")
print(f"  avg_int8 nz={np.count_nonzero(avg_int8)}/1280  range=[{avg_int8.min()},{avg_int8.max()}]")

# Check: are int32 logits dominated by bias or weights?
weight_part = fc_wi.astype(np.int32) @ avg_int8[0].astype(np.int32)
print(f"\n  weight_part[65]={weight_part[65]}  bias_part[65]={fc_bi[65]}  total[65]={logits[65]}")
print(f"  weight_part[{pred}]={weight_part[pred]}  bias_part[{pred}]={fc_bi[pred]}  total[{pred}]={logits[pred]}")

# Top-5
top5 = np.argsort(logits)[-5:][::-1]
print(f"\n  INT8 Top-5: {top5.tolist()}")
float_top5 = np.argsort(float_logits[1:] if nc==1001 else float_logits)[-5:][::-1]
print(f"  Float Top-5: {float_top5.tolist()}")
ref_top5 = np.argsort(logits_float_ref)[-5:][::-1]
print(f"  Float-from-int8-avg Top-5: {ref_top5.tolist()}")

# --- Also run the full model to get the true last hidden state ---
# Use hooks to capture the state just before the classifier
last_hidden = {}
def capture_last_hidden(name):
    def hook(module, inp, out):
        last_hidden[name] = out.detach().float().numpy()
    return hook

# Hook on the last conv layer's BN output
parts = "mobilenet_v2.conv_1x1".split(".")
mod2 = model
for pt in parts:
    mod2 = getattr(mod2, pt)
h_last = mod2.normalization.register_forward_hook(capture_last_hidden("conv52_bn"))

x_float_input = torch.tensor(img_f.transpose(2,0,1)[np.newaxis], dtype=torch.float32)
with torch.no_grad():
    model(x_float_input)
h_last.remove()

# conv52_bn output is [1, 1280, 7, 7] — before ReLU6
float_conv52 = last_hidden["conv52_bn"]
float_conv52_relu6 = np.clip(float_conv52, 0, 6.0)  # ReLU6
float_avg = float_conv52_relu6.mean(axis=(2, 3))  # [1, 1280]

# Compare with our int8 avg pool
our_avg_float = avg_int8[0].astype(np.float64) * x_scale  # dequantize
print(f"\n  --- Avg pool comparison ---")
print(f"  Float avg range: [{float_avg.min():.4f}, {float_avg.max():.4f}]")
print(f"  INT8 avg (dequant) range: [{our_avg_float.min():.4f}, {our_avg_float.max():.4f}]")
diff = np.abs(float_avg[0] - our_avg_float)
print(f"  Max abs diff: {diff.max():.4f}  Mean abs diff: {diff.mean():.4f}")
print(f"  Correlation: {np.corrcoef(float_avg[0], our_avg_float)[0,1]:.6f}")

# What does float FC give on float avg?
logits_pure_float = fc_w @ float_avg[0] + fc_b
print(f"  Float FC on float avg: pred={np.argmax(logits_pure_float)} top5={np.argsort(logits_pure_float)[-5:][::-1].tolist()}")
# What does float FC give on dequantized int8 avg?
print(f"  Float FC on dequant int8 avg: pred={pred_ref} top5={ref_top5.tolist()}")
