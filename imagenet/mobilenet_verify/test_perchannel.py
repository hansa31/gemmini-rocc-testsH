#!/usr/bin/env python3
"""Test per-channel DW weight quantization fix."""
import numpy as np, torch
from transformers import MobileNetV2ForImageClassification
import cv2, os

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BN_EPS = 0.001
IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float64)

print("Loading model...")
model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
nc = sd["classifier.weight"].shape[0]

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

def qw_perchannel_dw(w_3d):
    """Per-channel quantization for DW weights [C, kH, kW].
    Returns: w_int8[C, kH, kW], w_scales[C]
    """
    C = w_3d.shape[0]
    w_int = np.zeros_like(w_3d, dtype=np.int8)
    w_scales = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mx = float(np.max(np.abs(w_3d[c])))
        if mx < 1e-10:
            w_scales[c] = 1e-10
        else:
            w_scales[c] = mx / 127.0
            w_int[c] = np.clip(np.round(w_3d[c] / w_scales[c]), -128, 127).astype(np.int8)
    return w_int, w_scales

def dw_conv_op_perchannel(x, w_int, b_int, output_scales, stride, pad):
    """DW conv with per-channel output_scale.
    output_scales: [C] array of per-channel scales.
    """
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
    # Per-channel output scale
    os_arr = output_scales.reshape(1, C, 1, 1)
    return np.clip(np.round(acc.astype(np.float64)*os_arr),-128,127).astype(np.int8)

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

def dw_conv_op_scalar(x, w_int, b_int, os, stride, pad):
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

# Get float activations
float_acts = {}
def capture(name):
    def hook(module, inp, out):
        float_acts[name] = out.detach().float().numpy().astype(np.float64)
    return hook

hooks = []
for name, pfx in [("conv_1", "mobilenet_v2.conv_stem.first_conv"),
                   ("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3"),
                   ("conv_3", "mobilenet_v2.conv_stem.reduce_1x1")]:
    parts = pfx.split(".")
    mod2 = model
    for pt in parts:
        mod2 = getattr(mod2, pt)
    h = mod2.normalization.register_forward_hook(capture(name))
    hooks.append(h)

img_bgr = cv2.imread("/home/hansa/Downloads/Images200/ILSVRC2012_val_00000001.JPEG")
img = cv2.resize(img_bgr, (224,224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_f = img_rgb.astype(np.float64)/255.0
for c in range(3): img_f[:,:,c] = (img_f[:,:,c]-IMAGENET_MEAN[c])/IMAGENET_STD[c]
x_float = torch.tensor(img_f.transpose(2,0,1)[np.newaxis], dtype=torch.float32)

with torch.no_grad():
    model(x_float)

for h in hooks:
    h.remove()

float_conv1_r6 = np.clip(float_acts["conv_1"], 0, 6.0)
float_dw2_r6 = np.clip(float_acts["conv_dw_2"], 0, 6.0)
float_conv3 = float_acts["conv_3"]

# --- conv_1 (same as before) ---
cw1, g1, b1, m1, v1 = get_bn(sd, "mobilenet_v2.conv_stem.first_conv")
wf1, bf1 = fold_bn(cw1, g1, b1, m1, v1)
no = (128.0/255.0-IMAGENET_MEAN)/IMAGENET_STD
for c in range(3): bf1 += wf1[:,c,:,:].sum(axis=(1,2))*no[c]
for c in range(3): wf1[:,c,:,:] /= (255.0*IMAGENET_STD[c])
wg1 = wf1.reshape(wf1.shape[0],-1).T
w1_int, w1_scale = qw(wg1)
x_scale = 128.0/127.0
b1_int = qb(bf1, w1_scale * x_scale)
yr1 = 6.0
os1 = (w1_scale * x_scale) / (yr1/127.0)

img_int8 = (img_rgb.astype(np.int16)-128).clip(-128,127).astype(np.int8)
x_in = img_int8.transpose(2,0,1)[np.newaxis]
out1 = conv_op(x_in, w1_int, b1_int, os1, 3, 2, 1)
out1 = np.maximum(out1, np.int8(0))

# --- conv_dw_2 with per-channel quantization ---
x_scale_dw2 = yr1/127.0  # 0.047244
cw2, g2, b2, m2, v2 = get_bn(sd, "mobilenet_v2.conv_stem.conv_3x3")
wf2, bf2 = fold_bn(cw2, g2, b2, m2, v2)
wg2 = wf2.squeeze(1)  # [32, 3, 3]

# Per-channel quantization
w2_int_pc, w2_scales_pc = qw_perchannel_dw(wg2)
# Per-channel bias quantization
b2_int_pc = np.zeros(32, dtype=np.int32)
for c in range(32):
    b2_int_pc[c] = qb(np.array([bf2[c]]), w2_scales_pc[c] * x_scale_dw2)[0]
# Per-channel output scales
yr2 = 6.0
os2_pc = (w2_scales_pc * x_scale_dw2) / (yr2/127.0)

print("Per-channel DW conv_dw_2:")
print(f"  w_scales range: [{w2_scales_pc.min():.6f}, {w2_scales_pc.max():.6f}]")
print(f"  os range: [{os2_pc.min():.6e}, {os2_pc.max():.6e}]")

# Run with per-channel output_scale
out2_pc = dw_conv_op_perchannel(out1, w2_int_pc, b2_int_pc, os2_pc, 1, 1)
out2_pc = np.maximum(out2_pc, np.int8(0))

# Compare with float
out2_pc_dequant = out2_pc.astype(np.float64) * (yr2/127.0)
corr_pc = np.corrcoef(float_dw2_r6.flatten(), out2_pc_dequant.flatten())[0,1]

# Also run with per-tensor (original)
w2_int_pt, w2_scale_pt = qw(wg2)
b2_int_pt = qb(bf2, w2_scale_pt * x_scale_dw2)
os2_pt = (w2_scale_pt * x_scale_dw2) / (yr2/127.0)
out2_pt = dw_conv_op_scalar(out1, w2_int_pt, b2_int_pt, os2_pt, 1, 1)
out2_pt = np.maximum(out2_pt, np.int8(0))
out2_pt_dequant = out2_pt.astype(np.float64) * (yr2/127.0)
corr_pt = np.corrcoef(float_dw2_r6.flatten(), out2_pt_dequant.flatten())[0,1]

print(f"\nconv_dw_2 correlation:")
print(f"  Per-tensor:  {corr_pt:.6f}")
print(f"  Per-channel: {corr_pc:.6f}")

# Now propagate through conv_3 with per-channel DW output
x_scale_c3 = yr2/127.0
cw3, g3, b3, m3, v3 = get_bn(sd, "mobilenet_v2.conv_stem.reduce_1x1")
wf3, bf3 = fold_bn(cw3, g3, b3, m3, v3)
wg3 = wf3.reshape(wf3.shape[0],-1).T
w3_int, w3_scale = qw(wg3)
b3_int = qb(bf3, w3_scale * x_scale_c3)
yr3 = max(float(np.max(np.abs(b3)+6.0*np.abs(g3))), 1.0)
os3 = (w3_scale * x_scale_c3) / (yr3/127.0)

out3_pc = conv_op(out2_pc, w3_int, b3_int, os3, 1, 1, 0)
out3_pc_dequant = out3_pc.astype(np.float64) * (yr3/127.0)
corr_c3_pc = np.corrcoef(float_conv3.flatten(), out3_pc_dequant.flatten())[0,1]

out3_pt = conv_op(out2_pt, w3_int, b3_int, os3, 1, 1, 0)
out3_pt_dequant = out3_pt.astype(np.float64) * (yr3/127.0)
corr_c3_pt = np.corrcoef(float_conv3.flatten(), out3_pt_dequant.flatten())[0,1]

print(f"\nconv_3 correlation (downstream effect):")
print(f"  Per-tensor DW -> conv_3:  {corr_c3_pt:.6f}")
print(f"  Per-channel DW -> conv_3: {corr_c3_pc:.6f}")
