#!/usr/bin/env python3
"""Trace where float vs INT8 diverge layer-by-layer."""
import os, numpy as np, torch
from transformers import MobileNetV2ForImageClassification
import cv2

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BN_EPS = 0.001
IMAGENET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float64)
IMAGENET_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float64)

print("Loading model...")
model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
nc = sd["classifier.weight"].shape[0]

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

# --- Capture FLOAT activations from the model ---
float_activations = {}
def make_capture_hook(gn, has_relu):
    def hook(module, inp, out):
        x = out.detach().float().numpy().astype(np.float64)
        if has_relu:
            x = np.clip(x, 0, 6.0)  # ReLU6
        float_activations[gn] = x
    return hook

hooks = []
for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    parts = pfx.split(".")
    mod = model
    for pt in parts:
        mod = getattr(mod, pt)
    bn_mod = mod.normalization
    h = bn_mod.register_forward_hook(make_capture_hook(gn, relu))
    hooks.append(h)

# Also need to capture post-resadd activations.
# Those happen OUTSIDE the hook since resadd is applied after the reduce layer.
# We'll capture the reduce layer's post-BN output and manually apply resadd.

img_bgr = cv2.imread("/home/hansa/Downloads/Images200/ILSVRC2012_val_00000001.JPEG")
img = cv2.resize(img_bgr, (224,224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_f = img_rgb.astype(np.float64)/255.0
for c in range(3): img_f[:,:,c] = (img_f[:,:,c]-IMAGENET_MEAN[c])/IMAGENET_STD[c]
x_float = torch.tensor(img_f.transpose(2,0,1)[np.newaxis], dtype=torch.float32)

with torch.no_grad():
    out = model(x_float)
    float_logits = out.logits[0].numpy()

for h in hooks:
    h.remove()

if nc == 1001:
    print(f"Float pred (skip bg): {np.argmax(float_logits[1:])}")

# --- INT8 forward ---
def fold_bn(w, g, b, m, v, eps=BN_EPS):
    inv = 1.0/np.sqrt(v+eps); s = g*inv
    shape = [w.shape[0]]+[1]*(w.ndim-1)
    return w*s.reshape(shape), b-g*m*inv

def get_bn(sd, pfx):
    return tuple(sd[f"{pfx}.{k}"].float().numpy().astype(np.float64) for k in
                 ["convolution.weight","normalization.weight","normalization.bias",
                  "normalization.running_mean","normalization.running_var"])

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

# Run calibration first to get proper ranges
all_abs = {}
def make_calib_hook(gn, has_relu):
    def hook(module, inp, out):
        x = out.detach().float()
        if has_relu:
            x = torch.clamp(torch.relu(x), max=6.0)  # ReLU6 — match actual model behavior
        vals = x.abs().flatten()
        if vals.numel() > 10000:
            idx = torch.randperm(vals.numel())[:10000]
            vals = vals[idx]
        if gn not in all_abs:
            all_abs[gn] = []
        all_abs[gn].append(vals.numpy())
    return hook

ch = []
for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    parts = pfx.split(".")
    mod = model
    for pt in parts:
        mod = getattr(mod, pt)
    bn_mod = mod.normalization
    h = bn_mod.register_forward_hook(make_calib_hook(gn, relu))
    ch.append(h)

calib_dir = "/home/hansa/Downloads/Images200"
files = sorted([f for f in os.listdir(calib_dir) if f.lower().endswith((".jpeg",".jpg",".png"))])[:100]
print("Calibrating on", len(files), "images...")
with torch.no_grad():
    for i, fname in enumerate(files):
        img_c = cv2.imread(os.path.join(calib_dir, fname))
        if img_c is None: continue
        img_c = cv2.resize(img_c, (224,224))
        img_rgb_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        img_f_c = img_rgb_c.astype(np.float64)/255.0
        for c in range(3): img_f_c[:,:,c] = (img_f_c[:,:,c]-IMAGENET_MEAN[c])/IMAGENET_STD[c]
        x_c = torch.tensor(img_f_c.transpose(2,0,1)[np.newaxis], dtype=torch.float32)
        model(x_c)

for h in ch:
    h.remove()

calibrated = {}
for gn in all_abs:
    combined = np.concatenate(all_abs[gn])
    calibrated[gn] = float(np.percentile(combined, 99.99))
print(f"Calibrated {len(calibrated)} layers")

# Run INT8 forward with CALIBRATED y_range for ALL layers (not ReLU6 capped)
img_int8 = (img_rgb.astype(np.int16)-128).clip(-128,127).astype(np.int8)
x_in = img_int8.transpose(2,0,1)[np.newaxis]

out_int = x_in.copy()
stored = {}
x_scale = 128.0/127.0
layer_y_range = {}

print(f"\n{'='*90}")
print(f"{'Layer':12s} {'shape':20s}  {'Float range':20s}  {'INT8 range':20s}  {'Correlation':>12s}")
print(f"{'='*90}")

for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    cw, gamma, beta, bn_m, bn_v = get_bn(sd, pfx)
    wf, bf = fold_bn(cw, gamma, beta, bn_m, bn_v)

    if gn == "conv_1":
        no = (128.0/255.0-IMAGENET_MEAN)/IMAGENET_STD
        for c in range(3): bf += wf[:,c,:,:].sum(axis=(1,2))*no[c]
        for c in range(3): wf[:,c,:,:] /= (255.0*IMAGENET_STD[c])

    if dw: wg = wf.squeeze(1)
    else: wg = wf.reshape(wf.shape[0],-1).T

    # Use calibrated y_range for ALL layers
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

    # Compare with float
    # Dequantize int8: float_value = int8_value * y_scale
    y_scale_val = yr / 127.0
    int8_dequant = out_int.astype(np.float64) * y_scale_val

    # Get float reference (post-BN, post-activation)
    float_ref = float_activations.get(gn)

    if float_ref is not None and float_ref.shape == out_int.shape:
        corr = np.corrcoef(float_ref.flatten(), int8_dequant.flatten())[0,1]
        fr = f"[{float_ref.min():.2f},{float_ref.max():.2f}]"
        ir = f"[{int8_dequant.min():.2f},{int8_dequant.max():.2f}]"
        print(f"  {gn:12s} {str(out_int.shape):20s}  {fr:20s}  {ir:20s}  {corr:12.6f}")
    else:
        print(f"  {gn:12s} {str(out_int.shape):20s}  {'N/A':20s}  "
              f"[{int8_dequant.min():.2f},{int8_dequant.max():.2f}]  {'N/A':>12s}")

    x_scale = yr/127.0

print(f"{'='*90}")
