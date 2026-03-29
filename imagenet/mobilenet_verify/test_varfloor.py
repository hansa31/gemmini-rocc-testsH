#!/usr/bin/env python3
"""Test: clamp dead BN channels' weights after folding, then trace correlation."""
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

def fold_bn_clamped(w, g, b, m, v, eps=BN_EPS, var_floor=0.01):
    """BN folding with variance floor to prevent dead channel weight explosion."""
    # If running variance is very small, the channel is effectively dead
    # (contributes near-zero signal in float model). Clamping prevents
    # the folded weight from exploding.
    v_clamped = np.maximum(v, var_floor)
    inv = 1.0/np.sqrt(v_clamped+eps); s = g*inv
    shape = [w.shape[0]]+[1]*(w.ndim-1)
    w_folded = w*s.reshape(shape)
    b_folded = b-g*m*inv
    return w_folded, b_folded

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

# Capture float activations
float_acts = {}
def capture(name):
    def hook(module, inp, out):
        float_acts[name] = out.detach().float().numpy().astype(np.float64)
    return hook

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

hooks = []
for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    parts = pfx.split(".")
    mod2 = model
    for pt in parts:
        mod2 = getattr(mod2, pt)
    h = mod2.normalization.register_forward_hook(capture(gn))
    hooks.append(h)

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

print(f"Float pred: {np.argmax(float_logits[1:])}")

# Run INT8 with clamped BN folding
img_int8 = (img_rgb.astype(np.int16)-128).clip(-128,127).astype(np.int8)
x_in = img_int8.transpose(2,0,1)[np.newaxis]

# Test different var_floor values
for var_floor in [0.0, 0.001, 0.01, 0.1, 1.0]:
    out_int = x_in.copy()
    stored = {}
    x_scale = 128.0/127.0
    layer_y_range = {}

    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        cw, gamma, beta, bn_m, bn_v = get_bn(sd, pfx)

        if var_floor > 0:
            wf, bf = fold_bn_clamped(cw, gamma, beta, bn_m, bn_v, var_floor=var_floor)
        else:
            wf, bf = fold_bn(cw, gamma, beta, bn_m, bn_v)

        if gn == "conv_1":
            no = (128.0/255.0-IMAGENET_MEAN)/IMAGENET_STD
            for c in range(3): bf += wf[:,c,:,:].sum(axis=(1,2))*no[c]
            for c in range(3): wf[:,c,:,:] /= (255.0*IMAGENET_STD[c])

        if dw: wg = wf.squeeze(1)
        else: wg = wf.reshape(wf.shape[0],-1).T

        yr = 6.0 if relu else max(float(np.max(np.abs(beta)+6.0*np.abs(gamma))), 1.0)
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

        x_scale = yr/127.0

    # Global avg pool
    avg = out_int.astype(np.float64).mean(axis=(2,3))
    avg_int8 = np.clip(np.round(avg),-128,127).astype(np.int8)

    # Compare avg pool with float
    float_conv52_r6 = np.clip(float_acts["conv_52"], 0, 6.0)
    float_avg = float_conv52_r6.mean(axis=(2,3))
    avg_dequant = avg_int8[0].astype(np.float64) * (yr/127.0) * 127.0 / 127.0  # last yr = conv_52 yr
    corr_avg = np.corrcoef(float_avg[0], avg_int8[0].astype(np.float64))[0,1]

    # FC
    fc_w = sd["classifier.weight"].numpy().astype(np.float64)
    fc_b = sd["classifier.bias"].numpy().astype(np.float64)
    if nc == 1001: fc_w = fc_w[1:,:]; fc_b = fc_b[1:]
    fc_wi, fc_ws = qw(fc_w)
    fc_bi = qb(fc_b, fc_ws*x_scale)
    logits = fc_wi.astype(np.int32) @ avg_int8[0].astype(np.int32) + fc_bi.astype(np.int32)
    pred = np.argmax(logits)

    # Key layer correlations
    corrs = {}
    for gn in ["conv_1", "conv_dw_2", "conv_3", "conv_6", "conv_12", "conv_33", "conv_52"]:
        float_ref = float_acts.get(gn)
        if float_ref is not None:
            has_relu = any(gn==l[0] and l[6] for l in ALL_LAYERS)
            if has_relu:
                float_ref = np.clip(float_ref, 0, 6.0)
            yr_val = layer_y_range.get(gn, 1.0)
            int8_val = stored.get(gn)
            if int8_val is not None and float_ref.shape == int8_val.shape:
                dequant = int8_val.astype(np.float64) * (yr_val/127.0)
                corrs[gn] = np.corrcoef(float_ref.flatten(), dequant.flatten())[0,1]

    corr_str = "  ".join(f"{k}:{v:.3f}" for k,v in corrs.items())
    print(f"var_floor={var_floor:.3f}  pred={pred:4d}  avg_corr={corr_avg:.4f}  {corr_str}")
