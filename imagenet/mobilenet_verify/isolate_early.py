#!/usr/bin/env python3
"""Isolate: compare float input vs INT8 input at each early layer."""
import numpy as np, torch
from transformers import MobileNetV2ForImageClassification
import cv2

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BN_EPS = 0.001
IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float64)

print("Loading model...")
model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

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

def conv_op(x, w_int, b_int, os_val, kernel, stride, pad):
    N,C,H,W = x.shape
    kH=kW=kernel
    OH=(H+2*pad-kH)//stride+1; OW=(W+2*pad-kW)//stride+1
    if pad>0: x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),constant_values=0)
    patches=np.zeros((N,OH,OW,kH*kW*C),dtype=x.dtype)
    for i in range(OH):
        for j in range(OW):
            p=x[:,:,i*stride:i*stride+kH,j*stride:j*stride+kW]
            patches[:,i,j,:]=p.transpose(0,2,3,1).reshape(N,-1)
    xf=patches.reshape(N*OH*OW,kH*kW*C)
    acc=xf.astype(np.int32)@w_int.astype(np.int32)+b_int.reshape(1,-1).astype(np.int32)
    yf=np.clip(np.round(acc.astype(np.float64)*os_val),-128,127).astype(np.int8)
    return yf.reshape(N,OH,OW,w_int.shape[1]).transpose(0,3,1,2)

def dw_conv_op(x, w_int, b_int, os_val, stride, pad):
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
    return np.clip(np.round(acc.astype(np.float64)*os_val),-128,127).astype(np.int8)

# Capture float activations
float_acts = {}
def capture(name):
    def hook(module, inp, out):
        float_acts[name] = out.detach().float().numpy().astype(np.float64)
    return hook

hooks = []
for name, pfx in [("conv_1", "mobilenet_v2.conv_stem.first_conv"),
                   ("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3")]:
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

# --- CONV_1 ---
# Get conv_1 weights
cw1, g1, b1, m1, v1 = get_bn(sd, "mobilenet_v2.conv_stem.first_conv")
wf1, bf1 = fold_bn(cw1, g1, b1, m1, v1)
no = (128.0/255.0-IMAGENET_MEAN)/IMAGENET_STD
for c in range(3): bf1 += wf1[:,c,:,:].sum(axis=(1,2))*no[c]
for c in range(3): wf1[:,c,:,:] /= (255.0*IMAGENET_STD[c])
wg1 = wf1.reshape(wf1.shape[0],-1).T  # [27, 32]
w1_int, w1_scale = qw(wg1)
x_scale_0 = 128.0/127.0
b1_int = qb(bf1, w1_scale * x_scale_0)
yr1 = 6.0
os1 = (w1_scale * x_scale_0) / (yr1/127.0)

# Run conv_1 on pixel-128 input
img_int8 = (img_rgb.astype(np.int16)-128).clip(-128,127).astype(np.int8)
x_in = img_int8.transpose(2,0,1)[np.newaxis]
out1 = conv_op(x_in, w1_int, b1_int, os1, 3, 2, 1)
out1_relu = np.maximum(out1, np.int8(0))

# Also run conv_1 on FLOAT input (to separate input quantization error from conv_1 error)
# The float model uses (pixel/255 - mean)/std as input. With pixel-128 mode,
# x_int8 = pixel - 128, and the folded weights account for the normalization.
# So the int8 input range is [-128, 127].
# The float equivalent: x_float_val = x_int8 * x_scale_0 = x_int8 * (128/127)
# And folded conv_1 does: output = conv(x_int8_float_equiv, w_folded) where
# w_folded accounts for /255 and /std.
# So actually our int8 pipeline conv_1 output is directly comparable.

out1_dequant = out1_relu.astype(np.float64) * (yr1/127.0)
corr_c1 = np.corrcoef(float_conv1_r6.flatten(), out1_dequant.flatten())[0,1]
print(f"CONV_1 output correlation (int8 vs float): {corr_c1:.6f}")

# Compare the INT8 values directly
float_c1_quantized = np.clip(np.round(float_conv1_r6 / (yr1/127.0)), -128, 127).astype(np.int8)
diff_c1 = np.abs(out1_relu.astype(np.int16) - float_c1_quantized.astype(np.int16))
print(f"  Max int8 diff: {diff_c1.max()}")
print(f"  Mean int8 diff: {diff_c1.mean():.2f}")
print(f"  Pct exact: {100*np.mean(diff_c1==0):.1f}%")

# --- CONV_DW_2 with float conv_1 input (quantized) ---
# This tests if conv_dw_2 itself is the problem or if the input error propagates
cw2, g2, b2, m2, v2 = get_bn(sd, "mobilenet_v2.conv_stem.conv_3x3")
wf2, bf2 = fold_bn(cw2, g2, b2, m2, v2)
wg2 = wf2.squeeze(1)  # [32, 3, 3]
w2_int, w2_scale = qw(wg2)
yr2 = 6.0
x_scale_after_c1 = yr1/127.0
b2_int = qb(bf2, w2_scale * x_scale_after_c1)
os2 = (w2_scale * x_scale_after_c1) / (yr2/127.0)

# DW with INT8 pipeline input (from our conv_1)
out2_pipeline = dw_conv_op(out1_relu, w2_int, b2_int, os2, 1, 1)
out2_pipeline_relu = np.maximum(out2_pipeline, np.int8(0))

# DW with float conv_1 input (quantized to int8)
out2_fromfloat = dw_conv_op(float_c1_quantized, w2_int, b2_int, os2, 1, 1)
out2_fromfloat_relu = np.maximum(out2_fromfloat, np.int8(0))

out2p_dequant = out2_pipeline_relu.astype(np.float64) * (yr2/127.0)
out2f_dequant = out2_fromfloat_relu.astype(np.float64) * (yr2/127.0)
corr_pipeline = np.corrcoef(float_dw2_r6.flatten(), out2p_dequant.flatten())[0,1]
corr_fromfloat = np.corrcoef(float_dw2_r6.flatten(), out2f_dequant.flatten())[0,1]

print(f"\nCONV_DW_2 from INT8 pipeline: corr={corr_pipeline:.6f}")
print(f"CONV_DW_2 from float input:   corr={corr_fromfloat:.6f}")

# That tells us if the error at conv_dw_2 is from conv_1's error or from conv_dw_2's own quantization
diff_dw2 = np.abs(out2_pipeline_relu.astype(np.int16) - out2_fromfloat_relu.astype(np.int16))
print(f"  Pipeline vs float-input int8 diff: max={diff_dw2.max()} mean={diff_dw2.mean():.2f}")

# Also check: what is the depthwise conv quantization precision?
# DW conv has only 9 weights per channel. The w_scale is shared across all channels.
print(f"\n  DW weight scale: {w2_scale:.6f}")
print(f"  DW weight abs max: {np.max(np.abs(wg2)):.6f}")
print(f"  Per-channel max: min={np.max(np.abs(wg2), axis=(1,2)).min():.6f} max={np.max(np.abs(wg2), axis=(1,2)).max():.6f}")
per_ch_max = np.max(np.abs(wg2), axis=(1,2))
print(f"  Channels where per-ch max < 0.1 * global max: {np.sum(per_ch_max < 0.1 * np.max(per_ch_max))}/32")
