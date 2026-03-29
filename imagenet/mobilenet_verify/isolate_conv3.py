#!/usr/bin/env python3
"""Isolate where quantization error enters at conv_3."""
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

# Capture float activations
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
    mod = model
    for pt in parts:
        mod = getattr(mod, pt)
    h = mod.normalization.register_forward_hook(capture(name))
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

# Get float post-BN outputs (pre-activation)
float_conv1 = np.clip(float_acts["conv_1"], 0, 6.0)  # ReLU6
float_conv_dw2 = np.clip(float_acts["conv_dw_2"], 0, 6.0)  # ReLU6
float_conv3 = float_acts["conv_3"]  # linear (no clip)

print(f"float_conv1 range: [{float_conv1.min():.4f}, {float_conv1.max():.4f}]")
print(f"float_conv_dw2 range: [{float_conv_dw2.min():.4f}, {float_conv_dw2.max():.4f}]")
print(f"float_conv3 range: [{float_conv3.min():.4f}, {float_conv3.max():.4f}]")

# Now test conv_3 with different inputs
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

# Get conv_3 weights
cw3, g3, b3, m3, v3 = get_bn(sd, "mobilenet_v2.conv_stem.reduce_1x1")
wf3, bf3 = fold_bn(cw3, g3, b3, m3, v3)
# Conv_3 is 1x1: [32, 16, 1, 1] -> reshaped to [32, 16].T = [16, 32]... wait
# Actually reshape: [out_ch, in_ch, 1, 1] -> [out_ch, in_ch] -> T = [in_ch, out_ch]
wg3 = wf3.reshape(wf3.shape[0], -1).T  # [32, 16]
w3_int, w3_scale = qw(wg3)

print(f"\nconv_3 weights: w_float max={np.max(np.abs(wg3)):.6f}  w_scale={w3_scale:.6f}")

# Test 1: Float input (from conv_dw_2), float weights -> float conv_3
# This is what the model actually computes:
# compute from float_conv_dw2 using folded weights
# conv_3 is a 1x1 conv: output = input @ weight.T + bias
# input shape: [1, 32, 112, 112] -> flatten spatial: [1*112*112, 32]
N, C, H, W = float_conv_dw2.shape
input_flat = float_conv_dw2.reshape(N, C, H*W).transpose(0, 2, 1).reshape(N*H*W, C)
float_result = input_flat @ wg3 + bf3.reshape(1, -1)
float_result_4d = float_result.reshape(N, H, W, 16).transpose(0, 3, 1, 2)
print(f"\nTest 1 (pure float conv_3): range=[{float_result_4d.min():.2f},{float_result_4d.max():.2f}]")
corr_test1 = np.corrcoef(float_conv3.flatten(), float_result_4d.flatten())[0,1]
print(f"  Correlation with model float output: {corr_test1:.6f}")

# Test 2: Float input quantized then dequantized, quantized weights
# Simulate: what if we quantize conv_dw_2 output to int8, then feed to conv_3 with quantized weights
y_range_dw2 = 6.0  # ReLU6
x_scale_dw2 = y_range_dw2 / 127.0
input_int8 = np.clip(np.round(float_conv_dw2 / x_scale_dw2), -128, 127).astype(np.int8)
# Now do int8 matmul
input_flat_i = input_int8.reshape(N, C, H*W).transpose(0, 2, 1).reshape(N*H*W, C)
b3_int = qb(bf3, w3_scale * x_scale_dw2)
acc = input_flat_i.astype(np.int32) @ w3_int.astype(np.int32) + b3_int.reshape(1, -1).astype(np.int32)

# Output scale
y_range_3 = max(float(np.max(np.abs(b3) + 6.0 * np.abs(g3))), 1.0)
os_val = (w3_scale * x_scale_dw2) / (y_range_3 / 127.0)
result_int8 = np.clip(np.round(acc.astype(np.float64) * os_val), -128, 127).astype(np.int8)
result_dequant = result_int8.astype(np.float64) * (y_range_3 / 127.0)
result_4d = result_dequant.reshape(N, H, W, 16).transpose(0, 3, 1, 2)

corr_test2 = np.corrcoef(float_conv3.flatten(), result_4d.flatten())[0,1]
print(f"\nTest 2 (quantized input from ReLU6 y_range=6.0, quantized weights):")
print(f"  y_range_3={y_range_3:.2f}  os={os_val:.6e}  x_scale={x_scale_dw2:.6f}")
print(f"  Result range: [{result_4d.min():.2f},{result_4d.max():.2f}]")
print(f"  Correlation with model float output: {corr_test2:.6f}")

# Test 3: Same as Test 2 but with calibrated y_range for conv_dw_2
# What is the actual max value of conv_dw_2 output?
actual_max_dw2 = float(np.max(np.abs(float_conv_dw2)))
print(f"\nActual max |conv_dw_2| = {actual_max_dw2:.4f}")
y_range_dw2_calib = actual_max_dw2 * 1.01  # slight margin
x_scale_dw2_calib = y_range_dw2_calib / 127.0
input_int8_calib = np.clip(np.round(float_conv_dw2 / x_scale_dw2_calib), -128, 127).astype(np.int8)
input_flat_ic = input_int8_calib.reshape(N, C, H*W).transpose(0, 2, 1).reshape(N*H*W, C)
b3_int_c = qb(bf3, w3_scale * x_scale_dw2_calib)
acc_c = input_flat_ic.astype(np.int32) @ w3_int.astype(np.int32) + b3_int_c.reshape(1, -1).astype(np.int32)
os_val_c = (w3_scale * x_scale_dw2_calib) / (y_range_3 / 127.0)
result_int8_c = np.clip(np.round(acc_c.astype(np.float64) * os_val_c), -128, 127).astype(np.int8)
result_dequant_c = result_int8_c.astype(np.float64) * (y_range_3 / 127.0)
result_4d_c = result_dequant_c.reshape(N, H, W, 16).transpose(0, 3, 1, 2)

corr_test3 = np.corrcoef(float_conv3.flatten(), result_4d_c.flatten())[0,1]
print(f"\nTest 3 (actual-range quantized input, quantized weights):")
print(f"  y_range_dw2={y_range_dw2_calib:.4f}  x_scale={x_scale_dw2_calib:.6f}")
print(f"  y_range_3={y_range_3:.2f}  os={os_val_c:.6e}")
print(f"  Result range: [{result_4d_c.min():.2f},{result_4d_c.max():.2f}]")
print(f"  Correlation with model float output: {corr_test3:.6f}")

# Test 4: What about with TRUE float input (no quantization at all) and quantized weights?
b3_int_f = qb(bf3, w3_scale * 1.0)  # x_scale = 1.0 (meaningless, just for bias precision)
# Actually, let's just use float input but quantized weights
# acc = float_input @ w_int * w_scale + bias
input_f_flat = float_conv_dw2.reshape(N, C, H*W).transpose(0, 2, 1).reshape(N*H*W, C)
result_qw = input_f_flat @ (w3_int.astype(np.float64) * w3_scale) + bf3.reshape(1, -1)
result_qw_4d = result_qw.reshape(N, H, W, 16).transpose(0, 3, 1, 2)
corr_test4 = np.corrcoef(float_conv3.flatten(), result_qw_4d.flatten())[0,1]
print(f"\nTest 4 (float input, quantized weights dequantized):")
print(f"  Correlation with model float output: {corr_test4:.6f}")

# Test 5: What about with true float input and float weights?
result_ff = input_f_flat @ wg3 + bf3.reshape(1, -1)
result_ff_4d = result_ff.reshape(N, H, W, 16).transpose(0, 3, 1, 2)
corr_test5 = np.corrcoef(float_conv3.flatten(), result_ff_4d.flatten())[0,1]
print(f"\nTest 5 (float input, float weights - manual):")
print(f"  Correlation with model float output: {corr_test5:.6f}")
