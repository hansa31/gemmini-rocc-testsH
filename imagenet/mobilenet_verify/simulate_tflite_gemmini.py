#!/usr/bin/env python3
"""
Gemmini-compatible per-channel INT8 simulation using TFLite MobileNetV2 weights.

Loads mobilenet_v2_quant.tflite, dequantizes all UINT8 weights/INT32 biases
back to float, then re-quantizes with Gemmini per-channel INT8 scheme and
runs inference on ImageNet validation images.

Usage:
    conda run -n ImageNet python3 simulate_tflite_gemmini.py
"""

import os
import sys
import numpy as np

try:
    import tflite
except ImportError:
    print("ERROR: pip install tflite")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH  = os.path.join(SCRIPT_DIR, "mobilenet_v2_quant.tflite")
IMAGE_DIR    = "/home/hansa/Downloads/Images200"
LABELS_FILE  = "/home/hansa/Downloads/imagenet_val_50000_clipped_labels.txt"
NUM_IMAGES   = 200
PRINT_EVERY  = 20

# ---------------------------------------------------------------------------
# TFLite model loading helpers
# ---------------------------------------------------------------------------

def load_tflite_model(path):
    data = open(path, "rb").read()
    model = tflite.Model.GetRootAs(data)
    return model, model.Subgraphs(0)


def get_tensor_data(model, graph, tensor_idx):
    """Extract numpy array from a TFLite tensor buffer."""
    t = graph.Tensors(tensor_idx)
    buf = model.Buffers(t.Buffer())
    raw = buf.DataAsNumpy()
    shape = tuple(t.Shape(j) for j in range(t.ShapeLength()))
    dtype = t.Type()
    if dtype == 3:  # UINT8
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    elif dtype == 2:  # INT32
        arr = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
    elif dtype == 9:  # INT8
        arr = np.frombuffer(raw, dtype=np.int8).reshape(shape)
    elif dtype == 0:  # FLOAT32
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    else:
        raise ValueError(f"Unsupported dtype {dtype} for tensor {tensor_idx}")
    return arr


def get_quant_params(graph, tensor_idx):
    """Return (scales, zero_points) arrays for a tensor."""
    t = graph.Tensors(tensor_idx)
    qp = t.Quantization()
    if qp is None or qp.ScaleLength() == 0:
        return np.array([1.0]), np.array([0])
    scales = np.array([qp.Scale(j) for j in range(qp.ScaleLength())], dtype=np.float64)
    zps = np.array([qp.ZeroPoint(j) for j in range(qp.ZeroPointLength())], dtype=np.int64)
    return scales, zps


# ---------------------------------------------------------------------------
# Operator map: identifies which TFLite ops are CONV2D, DW_CONV, ADD, etc.
# ---------------------------------------------------------------------------

def parse_operators(model, graph):
    """Return list of (op_index, op_type_str, input_tensor_indices, output_tensor_indices)."""
    op_codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        op_codes.append(oc.DeprecatedBuiltinCode())

    OPS = {0: "ADD", 1: "AVG_POOL", 3: "CONV2D", 4: "DW_CONV", 22: "RESHAPE"}
    result = []
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        bc = op_codes[op.OpcodeIndex()]
        name = OPS.get(bc, f"OP{bc}")
        inputs = [op.Inputs(j) for j in range(op.InputsLength())]
        outputs = [op.Outputs(j) for j in range(op.OutputsLength())]
        result.append((i, name, inputs, outputs))
    return result


# ---------------------------------------------------------------------------
# Layer definitions matching the reference simulation
# ---------------------------------------------------------------------------

# Each conv/dw layer: (layer_name, op_index, is_dw, has_relu6)
# We build this from the operator list.

def build_layer_list(ops):
    """
    Build ordered list of compute layers from TFLite operators.
    Returns list of dicts with layer info, plus ADD op mapping.
    """
    layers = []
    add_ops = []
    layer_idx = 0

    for op_idx, op_type, inputs, outputs in ops:
        if op_type == "CONV2D":
            layers.append({
                "name": f"layer_{layer_idx}",
                "op_idx": op_idx,
                "type": "CONV2D",
                "input_tidx": inputs[0],
                "weight_tidx": inputs[1],
                "bias_tidx": inputs[2] if len(inputs) > 2 else None,
                "output_tidx": outputs[0],
            })
            layer_idx += 1
        elif op_type == "DW_CONV":
            layers.append({
                "name": f"layer_{layer_idx}",
                "op_idx": op_idx,
                "type": "DW_CONV",
                "input_tidx": inputs[0],
                "weight_tidx": inputs[1],
                "bias_tidx": inputs[2] if len(inputs) > 2 else None,
                "output_tidx": outputs[0],
            })
            layer_idx += 1
        elif op_type == "ADD":
            add_ops.append({
                "op_idx": op_idx,
                "input_tidxs": inputs,
                "output_tidx": outputs[0],
            })
        elif op_type == "AVG_POOL":
            layers.append({
                "name": f"layer_{layer_idx}",
                "op_idx": op_idx,
                "type": "AVG_POOL",
                "input_tidx": inputs[0],
                "output_tidx": outputs[0],
            })
            layer_idx += 1

    return layers, add_ops


# ---------------------------------------------------------------------------
# Dequantize TFLite weights/biases to float64
# ---------------------------------------------------------------------------

def dequantize_weight(model, graph, weight_tidx):
    """Dequantize UINT8 weight tensor to float64."""
    w_uint8 = get_tensor_data(model, graph, weight_tidx)
    w_sc, w_zp = get_quant_params(graph, weight_tidx)
    # real_value = (uint8_val - zero_point) * scale
    w_float = (w_uint8.astype(np.float64) - float(w_zp[0])) * float(w_sc[0])
    return w_float


def dequantize_bias(model, graph, bias_tidx):
    """Dequantize INT32 bias tensor to float64."""
    b_int32 = get_tensor_data(model, graph, bias_tidx)
    b_sc, b_zp = get_quant_params(graph, bias_tidx)
    b_float = (b_int32.astype(np.float64) - float(b_zp[0])) * float(b_sc[0])
    return b_float


# ---------------------------------------------------------------------------
# Gemmini per-channel quantization
# ---------------------------------------------------------------------------

def qw_perchannel_conv(w_flat):
    """w_flat shape [patch_size, out_ch]. Per-channel over output channels."""
    out_ch = w_flat.shape[1]
    ws = np.zeros(out_ch, dtype=np.float64)
    for j in range(out_ch):
        mx = float(np.max(np.abs(w_flat[:, j])))
        ws[j] = max(mx / 127.0, 1e-10)
    w_int = np.clip(np.round(w_flat / ws[np.newaxis, :]), -128, 127).astype(np.int8)
    return w_int, ws


def qw_perchannel_dw(w_chw):
    """w_chw shape [C, kH, kW]. Per-filter scale."""
    C = w_chw.shape[0]
    ws = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mx = float(np.max(np.abs(w_chw[c])))
        ws[c] = max(mx / 127.0, 1e-10)
    w_int = np.clip(np.round(w_chw / ws.reshape(-1, 1, 1)), -128, 127).astype(np.int8)
    return w_int, ws


def qb_perchannel(b_float, ws_arr, x_scale):
    """Bias quantized with per-channel combined scale."""
    combined = ws_arr * x_scale
    return np.clip(np.round(b_float / combined), -(2**31), 2**31 - 1).astype(np.int32)


def compute_output_scale(ws_arr, x_scale, y_range):
    """Per-channel output scale: os[j] = (ws_j * x_scale) / (y_range / 127)."""
    y_scale = y_range / 127.0
    raw = (ws_arr * x_scale) / y_scale
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 1.0)
    return raw.astype(np.float64)


# ---------------------------------------------------------------------------
# Determine y_range for each layer from TFLite quantization params
# ---------------------------------------------------------------------------

def compute_y_range(graph, output_tidx):
    """
    Compute y_range from TFLite output tensor quantization.
    For ReLU6: output_scale ~= 6/255 = 0.023528, zp=0 => y_range = 6.0
    For linear: y_range = scale * max(zp, 255 - zp)
    """
    sc, zp = get_quant_params(graph, output_tidx)
    scale = float(sc[0])
    zero_point = int(zp[0])

    # ReLU6 outputs have zp=0 and scale ~ 6/255
    if zero_point == 0 and abs(scale - 6.0 / 255.0) < 0.001:
        return 6.0

    # For asymmetric: the UINT8 range is [0, 255], centered at zp
    # Positive range: (255 - zp) * scale
    # Negative range: zp * scale
    y_range = scale * max(zero_point, 255 - zero_point)
    return max(y_range, 0.1)


# ---------------------------------------------------------------------------
# Reshape helpers for weight tensors
# ---------------------------------------------------------------------------

def reshape_conv_weight_tflite(w_ohwi):
    """
    TFLite CONV2D weight: [out_ch, kH, kW, in_ch] (OHWI)
    -> Gemmini im2col: [kH*kW*in_ch, out_ch]
    """
    out_ch = w_ohwi.shape[0]
    # Already OHWI, flatten spatial+input dims
    return w_ohwi.reshape(out_ch, -1).T  # [patch_size, out_ch]


def reshape_dw_weight_tflite(w_1hwc):
    """
    TFLite DW weight: [1, kH, kW, ch]
    -> [ch, kH, kW]
    """
    # [1, kH, kW, ch] -> [ch, kH, kW]
    return w_1hwc.squeeze(0).transpose(2, 0, 1)  # [ch, kH, kW]


# ---------------------------------------------------------------------------
# Determine stride and padding from TFLite output shapes
# ---------------------------------------------------------------------------

def infer_stride_pad(in_shape, out_shape, kernel_size):
    """
    Infer stride and padding from input/output spatial dimensions.
    in_shape: [N, H, W, C], out_shape: [N, oH, oW, C']
    """
    H, W = in_shape[1], in_shape[2]
    oH, oW = out_shape[1], out_shape[2]

    if kernel_size == 1:
        stride = H // oH if oH < H else 1
        return stride, 0

    # For 3x3: try stride=1 with pad=1 or stride=2 with pad=1
    for stride in [1, 2]:
        for pad in [0, 1]:
            calc_oh = (H + 2 * pad - kernel_size) // stride + 1
            if calc_oh == oH:
                return stride, pad

    # Fallback
    stride = H // oH if oH < H else 1
    pad = 1 if kernel_size == 3 else 0
    return stride, pad


# ---------------------------------------------------------------------------
# Build all Gemmini-quantized layers from TFLite model
# ---------------------------------------------------------------------------

def build_gemmini_layers(model, graph):
    ops = parse_operators(model, graph)
    layers_info, add_ops = build_layer_list(ops)

    # Get input tensor quantization
    input_tidx = graph.Inputs(0)
    in_sc, in_zp = get_quant_params(graph, input_tidx)
    print(f"  Input tensor: scale={float(in_sc[0]):.6e}, zp={int(in_zp[0])}")

    # For Gemmini, we use normalized input in [-1, 1] quantized to INT8.
    # x_scale_0 = 1.0/127.0 means range [-1, 1]
    # But TFLite input is uint8 with scale=1/128, zp=128, meaning
    #   real = (uint8 - 128) * (1/128) => range [-1, 127/128]
    # For our simulation with pixel-128 mode: x_int8 = pixel - 128
    #   Then x_scale = 128/127 so that x_int8 * x_scale covers ~[-128/127*128, ...]
    # Actually let's use: input pixels in [0,255] -> normalized to [-1,1]
    #   x_int8 = clip(round(norm_val * 127), -128, 127)
    #   x_scale = 1.0 / 127.0
    # The TFLite first conv expects input scaled by input_scale=1/128.
    # We'll fold the scale difference into the first layer.

    x_scale = 1.0 / 127.0  # Our input scale for [-1, 1] range

    # Pre-compute y_range for each compute layer (conv/dw)
    # and for ADD outputs
    add_output_y_range = {}
    for add_info in add_ops:
        yr = compute_y_range(graph, add_info["output_tidx"])
        add_output_y_range[add_info["op_idx"]] = yr

    # Map: which layers feed into which ADD ops?
    # ADD op has two inputs. The second input is the skip connection.
    # After the linear projection conv, the ADD combines it with an earlier tensor.
    # We need to track: for each ADD, which conv layer produced each input,
    # and what's the y_range of the ADD output.

    # Build a map from output tensor index to layer index
    tidx_to_layer = {}
    for i, linfo in enumerate(layers_info):
        if "output_tidx" in linfo:
            tidx_to_layer[linfo["output_tidx"]] = i

    # For ADD ops, also register the output
    add_by_output_tidx = {}
    for add_info in add_ops:
        add_by_output_tidx[add_info["output_tidx"]] = add_info

    # Build residual connections: for each conv layer that is immediately
    # followed by an ADD, record the ADD info
    conv_to_add = {}  # layer_index -> add_info
    for add_info in add_ops:
        # The first input to ADD is the conv output (main path)
        # The second input is the skip connection
        in0 = add_info["input_tidxs"][0]
        in1 = add_info["input_tidxs"][1]
        if in0 in tidx_to_layer:
            conv_to_add[tidx_to_layer[in0]] = add_info
        elif in1 in tidx_to_layer:
            # Swap: second input is from the just-computed conv
            add_info["input_tidxs"] = [in1, in0]
            conv_to_add[tidx_to_layer[in1]] = add_info

    # Now build quantized layers
    gemmini_layers = []
    # Track the y_range of each layer output (after potential ADD)
    layer_y_range = {}
    # Track ADD output tensor -> y_range for skip connections
    add_tidx_y_range = {}

    print(f"\n  Building {len(layers_info)} compute layers...")

    for li, linfo in enumerate(layers_info):
        if linfo["type"] == "AVG_POOL":
            # Average pooling just passes through, same scale
            gemmini_layers.append({
                "type": "AVG_POOL",
                "name": linfo["name"],
                "op_idx": linfo["op_idx"],
            })
            # y_range same as input
            # Find what feeds this: input_tidx
            in_tidx = linfo["input_tidx"]
            if in_tidx in add_tidx_y_range:
                layer_y_range[li] = add_tidx_y_range[in_tidx]
            else:
                # Look up from previous layer
                for prev_li in range(li - 1, -1, -1):
                    if prev_li in layer_y_range:
                        layer_y_range[li] = layer_y_range[prev_li]
                        break
            continue

        # Conv or DW conv
        is_dw = linfo["type"] == "DW_CONV"
        out_tidx = linfo["output_tidx"]
        w_tidx = linfo["weight_tidx"]
        b_tidx = linfo["bias_tidx"]
        in_tidx = linfo["input_tidx"]

        # Get input/output shapes for stride/pad inference
        in_t = graph.Tensors(in_tidx)
        in_shape = [in_t.Shape(j) for j in range(in_t.ShapeLength())]
        out_t = graph.Tensors(out_tidx)
        out_shape = [out_t.Shape(j) for j in range(out_t.ShapeLength())]

        # Determine kernel size
        w_t = graph.Tensors(w_tidx)
        w_shape = [w_t.Shape(j) for j in range(w_t.ShapeLength())]
        if is_dw:
            kernel = w_shape[1]  # [1, kH, kW, C]
        else:
            kernel = w_shape[1]  # [O, kH, kW, I]

        stride, pad = infer_stride_pad(in_shape, out_shape, kernel)

        # Determine if this layer has ReLU6 activation
        out_sc, out_zp = get_quant_params(graph, out_tidx)
        has_relu6 = (int(out_zp[0]) == 0 and abs(float(out_sc[0]) - 6.0 / 255.0) < 0.001)

        # y_range for this conv/dw output
        yr = compute_y_range(graph, out_tidx)

        # Check if followed by ADD: use ADD output y_range for the effective layer y_range
        effective_yr = yr
        res_scale = 1.0
        has_add = li in conv_to_add
        add_info_for_layer = None

        if has_add:
            add_info_for_layer = conv_to_add[li]
            add_yr = compute_y_range(graph, add_info_for_layer["output_tidx"])
            effective_yr = add_yr
            # res_scale: skip_y_range / add_y_range
            skip_tidx = add_info_for_layer["input_tidxs"][1]
            if skip_tidx in add_tidx_y_range:
                skip_yr = add_tidx_y_range[skip_tidx]
            else:
                skip_yr = compute_y_range(graph, skip_tidx)
            res_scale = skip_yr / add_yr

        # Dequantize weights and biases to float
        w_float = dequantize_weight(model, graph, w_tidx)
        b_float = dequantize_bias(model, graph, b_tidx) if b_tidx is not None else None

        # Reshape weights
        if is_dw:
            w_reshaped = reshape_dw_weight_tflite(w_float)  # [C, kH, kW]
            out_ch = w_reshaped.shape[0]
            if b_float is None:
                b_float = np.zeros(out_ch, dtype=np.float64)
        else:
            w_reshaped = reshape_conv_weight_tflite(w_float)  # [patch, out_ch]
            out_ch = w_reshaped.shape[1]
            if b_float is None:
                b_float = np.zeros(out_ch, dtype=np.float64)

        # Per-channel weight quantization
        if is_dw:
            w_int, ws = qw_perchannel_dw(w_reshaped)
        else:
            w_int, ws = qw_perchannel_conv(w_reshaped)

        # Bias quantization
        b_int = qb_perchannel(b_float, ws, x_scale)

        # Output scale
        output_scale = compute_output_scale(ws, x_scale, yr)

        os_mean = float(np.mean(output_scale))
        kind = "DW" if is_dw else "CV"
        relu_str = "ReLU6" if has_relu6 else "Lin"
        add_str = f" +ADD(rs={res_scale:.4f})" if has_add else ""
        print(f"    [{li:2d}] Op{linfo['op_idx']:3d} {kind:2s} k={kernel} s={stride} p={pad} "
              f"{relu_str:5s} out_ch={out_ch:4d} os_mean={os_mean:.4e} yr={yr:.4f}{add_str}")

        layer_entry = {
            "type": linfo["type"],
            "name": linfo["name"],
            "op_idx": linfo["op_idx"],
            "w_int": w_int,
            "b_int": b_int,
            "output_scale": output_scale,
            "kernel": kernel,
            "stride": stride,
            "padding": pad,
            "is_dw": is_dw,
            "has_relu6": has_relu6,
            "has_add": has_add,
            "res_scale": res_scale,
            "y_range": yr,
        }
        gemmini_layers.append(layer_entry)

        # Update x_scale for next layer
        x_scale = yr / 127.0

        # Store y_range
        layer_y_range[li] = yr

        # If this layer has ADD, store the ADD output y_range and update x_scale
        if has_add:
            add_tidx_y_range[add_info_for_layer["output_tidx"]] = effective_yr
            layer_y_range[li] = effective_yr
            x_scale = effective_yr / 127.0

    return gemmini_layers, x_scale


# ---------------------------------------------------------------------------
# INT8 convolution operations
# ---------------------------------------------------------------------------

def im2col(x_nchw, kernel, stride, pad):
    """im2col for NCHW int8 tensor."""
    N, C, H, W = x_nchw.shape
    kH = kW = kernel
    OH = (H + 2 * pad - kH) // stride + 1
    OW = (W + 2 * pad - kW) // stride + 1
    if pad > 0:
        x_nchw = np.pad(x_nchw, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                        constant_values=0)
    patches = np.zeros((N, OH, OW, kH * kW * C), dtype=x_nchw.dtype)
    for i in range(OH):
        for j in range(OW):
            p = x_nchw[:, :, i * stride:i * stride + kH, j * stride:j * stride + kW]
            patches[:, i, j, :] = p.transpose(0, 2, 3, 1).reshape(N, -1)
    return patches.reshape(N * OH * OW, kH * kW * C), OH, OW


def conv_int8(x_nchw, w_int, b_int, output_scale, kernel, stride, pad):
    """Per-channel scaled INT8 convolution."""
    N, C, H, W = x_nchw.shape
    if kernel == 1 and stride == 1:
        x_flat = x_nchw.reshape(N, C, H * W).transpose(0, 2, 1).reshape(N * H * W, C)
        oh = ow = H
    elif kernel == 1:
        xs = x_nchw[:, :, ::stride, ::stride]
        _N, _C, _H, _W = xs.shape
        x_flat = xs.reshape(_N, _C, _H * _W).transpose(0, 2, 1).reshape(_N * _H * _W, _C)
        oh = ow = _H
    else:
        x_flat, oh, ow = im2col(x_nchw, kernel, stride, pad)

    acc = x_flat.astype(np.int32) @ w_int.astype(np.int32)
    acc += b_int.reshape(1, -1).astype(np.int32)
    y = np.clip(np.round(acc.astype(np.float64) * output_scale[np.newaxis, :]),
                -128, 127).astype(np.int8)
    return y.reshape(N, oh, ow, w_int.shape[1]).transpose(0, 3, 1, 2)


def dw_conv_int8(x_nchw, w_int, b_int, output_scale, stride, pad):
    """Per-channel scaled INT8 depthwise convolution."""
    N, C, H, W = x_nchw.shape
    kH, kW = w_int.shape[1], w_int.shape[2]
    OH = (H + 2 * pad - kH) // stride + 1
    OW = (W + 2 * pad - kW) // stride + 1
    x_pad = (np.pad(x_nchw, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                    constant_values=0) if pad > 0 else x_nchw)
    acc = np.zeros((N, C, OH, OW), dtype=np.int64)
    x32 = x_pad.astype(np.int32)
    w32 = w_int.astype(np.int32)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            acc += x32[:, :, rows[:, None], cols[None, :]] * w32[:, ki, kj].reshape(1, C, 1, 1)
    acc += b_int.reshape(1, C, 1, 1).astype(np.int64)
    out = np.clip(np.round(acc.astype(np.float64) * output_scale.reshape(1, -1, 1, 1)),
                  -128, 127).astype(np.int8)
    return out


def resadd_int8(main_int8, skip_int8, res_scale):
    """Residual add: out = clip(round(skip * res_scale) + main, -128, 127)."""
    return np.clip(
        np.round(skip_int8.astype(np.float64) * res_scale) + main_int8.astype(np.float64),
        -128, 127
    ).astype(np.int8)


def global_avg_pool_int8(x_nchw):
    """Global average pool over spatial dims, returning int8."""
    avg = x_nchw.astype(np.float64).mean(axis=(2, 3))
    return np.clip(np.round(avg), -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Forward pass through all layers
# ---------------------------------------------------------------------------

def forward_pass(x_nchw_int8, gemmini_layers):
    """
    Run the full network forward pass.
    x_nchw_int8: [1, 3, 224, 224] int8 input
    Returns: logits array [num_classes]
    """
    out = x_nchw_int8
    stored = {}  # layer_index -> output tensor (for skip connections)

    # We need to track which tensor IDs map to which stored outputs
    # for residual connections. Since we built the layers in order,
    # and ADD ops combine the current conv output with an earlier layer's
    # (post-ADD) output, we track by layer index.

    # For residual connections: the skip source is the previous ADD output
    # (or a conv output). We identify these by noting that in TFLite,
    # ADD's second input comes from an earlier layer.
    # In our layer list, conv layers that have has_add=True will do the ADD.

    # We need to find which earlier stored output to use as skip.
    # Strategy: store outputs keyed by their TFLite output tensor index.
    tidx_to_output = {}

    for li, layer in enumerate(gemmini_layers):
        if layer["type"] == "AVG_POOL":
            out = global_avg_pool_int8(out)
            # out is now [N, C] shaped
            stored[li] = out
            continue

        if layer["type"] in ("CONV2D", "DW_CONV"):
            if layer["is_dw"]:
                out = dw_conv_int8(out, layer["w_int"], layer["b_int"],
                                   layer["output_scale"], layer["stride"],
                                   layer["padding"])
            else:
                out = conv_int8(out, layer["w_int"], layer["b_int"],
                                layer["output_scale"], layer["kernel"],
                                layer["stride"], layer["padding"])

            if layer["has_relu6"]:
                out = np.maximum(out, np.int8(0))

            stored[li] = out

            if layer["has_add"]:
                # Find skip connection: we need the skip tensor
                # The skip comes from a previous layer that has the same spatial dims
                # In practice, it's the output of a previous ADD or linear conv
                # We search backward for a matching shape
                skip = None
                for prev_li in range(li - 1, -1, -1):
                    if prev_li in stored and isinstance(stored[prev_li], np.ndarray):
                        prev_out = stored[prev_li]
                        if prev_out.ndim == out.ndim and prev_out.shape == out.shape:
                            # Check it's not the immediately preceding DW conv
                            # (which has different channels typically)
                            # The skip should be from a linear conv or ADD
                            if gemmini_layers[prev_li].get("has_add", False) or \
                               (not gemmini_layers[prev_li].get("has_relu6", True) and
                                not gemmini_layers[prev_li].get("is_dw", False)):
                                skip = prev_out
                                break

                if skip is None:
                    # Fallback: find any matching shape
                    for prev_li in range(li - 1, -1, -1):
                        if prev_li in stored and isinstance(stored[prev_li], np.ndarray):
                            if stored[prev_li].shape == out.shape:
                                skip = stored[prev_li]
                                break

                if skip is not None:
                    out = resadd_int8(out, skip, layer["res_scale"])
                    stored[li] = out  # Update with post-ADD output

    return out


# ---------------------------------------------------------------------------
# Full inference: forward pass + FC layer
# ---------------------------------------------------------------------------

def run_inference(x_nchw_int8, gemmini_layers, fc_layer):
    """Run full inference and return logits."""
    out = forward_pass(x_nchw_int8, gemmini_layers)

    # out should be [1, C] after avg pool, or [1, C, 1, 1] before
    if out.ndim == 4:
        out = global_avg_pool_int8(out)

    # FC layer
    fc_w = fc_layer["w_int"]      # [num_classes, in_features]
    fc_b = fc_layer["b_int"]      # [num_classes]
    fc_os = fc_layer["output_scale"]  # [num_classes]

    # Matrix multiply: [num_classes, in_features] @ [in_features] = [num_classes]
    features = out[0].astype(np.int32)  # [in_features]
    acc = fc_w.astype(np.int32) @ features + fc_b.astype(np.int32)

    # Scale and clip
    logits = np.clip(np.round(acc.astype(np.float64) * fc_os), -128, 127)
    return logits


# ---------------------------------------------------------------------------
# Build the network from TFLite model
# ---------------------------------------------------------------------------

def build_network(model, graph):
    """Build all layers including FC."""
    print("\n--- Building Gemmini layers from TFLite model ---")

    ops = parse_operators(model, graph)
    all_layers, add_ops = build_layer_list(ops)

    # Separate FC layer (last CONV2D with 1x1 spatial output = 1001 classes)
    # and AVG_POOL layer
    # The last three ops are: conv_52 (1280), AVG_POOL, FC (1001)

    # Find the FC layer (last CONV2D producing 1001 outputs)
    fc_layer_info = None
    fc_layer_idx = None
    for i in range(len(all_layers) - 1, -1, -1):
        if all_layers[i]["type"] == "CONV2D":
            out_t = graph.Tensors(all_layers[i]["output_tidx"])
            out_shape = [out_t.Shape(j) for j in range(out_t.ShapeLength())]
            if out_shape[-1] == 1001:
                fc_layer_info = all_layers[i]
                fc_layer_idx = i
                break

    # Build main conv layers (everything except FC)
    # We need to re-build using the full pipeline but stop before FC
    gemmini_layers, x_scale_before_fc = build_gemmini_layers(model, graph)

    # Build FC layer separately
    print("\n  Building FC layer...")
    w_float = dequantize_weight(model, graph, fc_layer_info["weight_tidx"])
    b_float = dequantize_bias(model, graph, fc_layer_info["bias_tidx"])

    # FC weight shape: [1001, 1, 1, 1280] -> [1001, 1280]
    fc_w_float = w_float.reshape(w_float.shape[0], -1)  # [1001, 1280]
    num_classes = fc_w_float.shape[0]

    # Per-output-class weight quantization
    ws_fc = np.zeros(num_classes, dtype=np.float64)
    for j in range(num_classes):
        mx = float(np.max(np.abs(fc_w_float[j])))
        ws_fc[j] = max(mx / 127.0, 1e-10)

    fc_w_int = np.clip(np.round(fc_w_float / ws_fc[:, np.newaxis]),
                       -128, 127).astype(np.int8)
    fc_b_int = np.clip(np.round(b_float / (ws_fc * x_scale_before_fc)),
                       -(2**31), 2**31 - 1).astype(np.int32)

    # FC y_range from TFLite output
    fc_yr = compute_y_range(graph, fc_layer_info["output_tidx"])
    fc_os = compute_output_scale(ws_fc, x_scale_before_fc, fc_yr)

    print(f"    FC: {num_classes} classes, os_mean={float(np.mean(fc_os)):.4e}, yr={fc_yr:.4f}")

    fc_layer = {
        "w_int": fc_w_int,
        "b_int": fc_b_int,
        "output_scale": fc_os,
    }

    return gemmini_layers, fc_layer


# ---------------------------------------------------------------------------
# Rebuild build_gemmini_layers to properly separate FC from conv layers
# ---------------------------------------------------------------------------

def build_gemmini_layers(model, graph):
    """Build all conv/dw/avgpool layers EXCEPT the final FC conv."""
    ops = parse_operators(model, graph)

    # Get input tensor quantization
    input_tidx = graph.Inputs(0)
    in_sc, in_zp = get_quant_params(graph, input_tidx)
    print(f"  Input tensor: scale={float(in_sc[0]):.6e}, zp={int(in_zp[0])}")

    x_scale = 1.0 / 127.0  # Our symmetric input scale for [-1, 1]

    # Identify all CONV2D, DW_CONV, ADD, AVG_POOL operators
    conv_ops = []  # (op_idx, op_type, inputs, outputs)
    add_ops = []
    avgpool_op = None

    for op_idx, op_type, inputs, outputs in ops:
        if op_type in ("CONV2D", "DW_CONV"):
            conv_ops.append((op_idx, op_type, inputs, outputs))
        elif op_type == "ADD":
            add_ops.append((op_idx, inputs, outputs))
        elif op_type == "AVG_POOL":
            avgpool_op = (op_idx, inputs, outputs)

    # The last CONV2D is the FC layer (1001 outputs). Exclude it.
    fc_conv = conv_ops[-1]
    conv_ops = conv_ops[:-1]

    # Also exclude the FC-like conv (conv_52, producing 1280 channels) from special handling
    # Actually conv_52 is a normal 1x1 conv, keep it.

    # Build map: which ADD follows which conv (by matching tensor indices)
    conv_out_to_idx = {}
    for i, (op_idx, op_type, inputs, outputs) in enumerate(conv_ops):
        conv_out_to_idx[outputs[0]] = i

    # For each ADD, find which conv's output is its input
    # ADD inputs: [main_path_tidx, skip_tidx]
    add_after_conv = {}  # conv_index -> (add_op_idx, add_inputs, add_outputs)
    for add_op_idx, add_inputs, add_outputs in add_ops:
        for inp_tidx in add_inputs:
            if inp_tidx in conv_out_to_idx:
                ci = conv_out_to_idx[inp_tidx]
                skip_tidx = add_inputs[1] if add_inputs[0] == inp_tidx else add_inputs[0]
                add_after_conv[ci] = (add_op_idx, skip_tidx, add_outputs[0])
                break

    # Track: tensor_idx -> y_range (for skip connections)
    tidx_y_range = {}

    gemmini_layers = []
    print(f"\n  Building {len(conv_ops)} conv/dw layers + avg_pool...")

    for ci, (op_idx, op_type, inputs, outputs) in enumerate(conv_ops):
        is_dw = (op_type == "DW_CONV")
        in_tidx = inputs[0]
        w_tidx = inputs[1]
        b_tidx = inputs[2] if len(inputs) > 2 else None
        out_tidx = outputs[0]

        # Shapes
        in_t = graph.Tensors(in_tidx)
        in_shape = [in_t.Shape(j) for j in range(in_t.ShapeLength())]
        out_t = graph.Tensors(out_tidx)
        out_shape = [out_t.Shape(j) for j in range(out_t.ShapeLength())]
        w_t = graph.Tensors(w_tidx)
        w_shape = [w_t.Shape(j) for j in range(w_t.ShapeLength())]

        kernel = w_shape[1] if not is_dw else w_shape[1]
        stride, pad = infer_stride_pad(in_shape, out_shape, kernel)

        # ReLU6 detection
        out_sc, out_zp = get_quant_params(graph, out_tidx)
        has_relu6 = (int(out_zp[0]) == 0 and abs(float(out_sc[0]) - 6.0 / 255.0) < 0.001)

        # y_range for this conv output
        yr = compute_y_range(graph, out_tidx)

        # Check for ADD
        has_add = ci in add_after_conv
        res_scale = 1.0
        effective_yr = yr

        if has_add:
            add_op_idx, skip_tidx, add_out_tidx = add_after_conv[ci]
            add_yr = compute_y_range(graph, add_out_tidx)
            effective_yr = add_yr

            if skip_tidx in tidx_y_range:
                skip_yr = tidx_y_range[skip_tidx]
            else:
                skip_yr = compute_y_range(graph, skip_tidx)
            res_scale = skip_yr / add_yr

        # Dequantize weights/biases
        w_float = dequantize_weight(model, graph, w_tidx)
        b_float = dequantize_bias(model, graph, b_tidx) if b_tidx is not None else None

        # Reshape
        if is_dw:
            w_reshaped = reshape_dw_weight_tflite(w_float)
            out_ch = w_reshaped.shape[0]
            if b_float is None:
                b_float = np.zeros(out_ch, dtype=np.float64)
        else:
            w_reshaped = reshape_conv_weight_tflite(w_float)
            out_ch = w_reshaped.shape[1]
            if b_float is None:
                b_float = np.zeros(out_ch, dtype=np.float64)

        # Per-channel quantize
        if is_dw:
            w_int, ws = qw_perchannel_dw(w_reshaped)
        else:
            w_int, ws = qw_perchannel_conv(w_reshaped)

        b_int = qb_perchannel(b_float, ws, x_scale)
        output_scale = compute_output_scale(ws, x_scale, yr)

        os_mean = float(np.mean(output_scale))
        kind = "DW" if is_dw else "CV"
        relu_str = "ReLU6" if has_relu6 else "Lin"
        add_str = f" +ADD(rs={res_scale:.4f})" if has_add else ""
        print(f"    [{ci:2d}] Op{op_idx:3d} {kind:2s} k={kernel} s={stride} p={pad} "
              f"{relu_str:5s} ch={out_ch:4d} os={os_mean:.4e} yr={yr:.4f}{add_str}")

        layer_entry = {
            "type": op_type,
            "w_int": w_int,
            "b_int": b_int,
            "output_scale": output_scale,
            "kernel": kernel,
            "stride": stride,
            "padding": pad,
            "is_dw": is_dw,
            "has_relu6": has_relu6,
            "has_add": has_add,
            "res_scale": res_scale,
            "y_range": yr,
            "out_tidx": out_tidx,
        }

        if has_add:
            layer_entry["skip_tidx"] = add_after_conv[ci][1]
            layer_entry["add_out_tidx"] = add_after_conv[ci][2]

        gemmini_layers.append(layer_entry)

        # Update x_scale
        x_scale = yr / 127.0

        # Store y_range for this output tensor
        tidx_y_range[out_tidx] = yr

        if has_add:
            add_out_tidx = add_after_conv[ci][2]
            tidx_y_range[add_out_tidx] = effective_yr
            x_scale = effective_yr / 127.0

    # Add AVG_POOL
    if avgpool_op:
        gemmini_layers.append({"type": "AVG_POOL"})
        print(f"    [AP] Op{avgpool_op[0]:3d} AVG_POOL 7x7")

    return gemmini_layers, x_scale


# ---------------------------------------------------------------------------
# Forward pass (revised, cleaner)
# ---------------------------------------------------------------------------

def forward_pass_v2(x_nchw_int8, gemmini_layers, fc_layer):
    """
    Run the full network.
    x_nchw_int8: [1, 3, 224, 224] int8
    Returns: logits [1001] (or [1001] raw scaled int8 values)
    """
    out = x_nchw_int8

    # For residual connections, we need to store outputs of linear (projection)
    # conv layers and ADD outputs.
    # Strategy: after each layer with has_add, store the post-ADD output
    # indexed by the TFLite output tensor index.
    # When we need a skip, we look up by skip_tidx.
    tidx_to_tensor = {}

    for li, layer in enumerate(gemmini_layers):
        if layer["type"] == "AVG_POOL":
            out = global_avg_pool_int8(out)
            continue

        is_dw = layer["is_dw"]

        if is_dw:
            out = dw_conv_int8(out, layer["w_int"], layer["b_int"],
                               layer["output_scale"], layer["stride"],
                               layer["padding"])
        else:
            out = conv_int8(out, layer["w_int"], layer["b_int"],
                            layer["output_scale"], layer["kernel"],
                            layer["stride"], layer["padding"])

        if layer["has_relu6"]:
            out = np.maximum(out, np.int8(0))

        # Store conv output by tensor index
        tidx_to_tensor[layer["out_tidx"]] = out

        if layer["has_add"]:
            skip_tidx = layer["skip_tidx"]
            add_out_tidx = layer["add_out_tidx"]

            skip = tidx_to_tensor.get(skip_tidx)
            if skip is None:
                raise RuntimeError(f"Skip tensor {skip_tidx} not found for layer {li}")

            out = resadd_int8(out, skip, layer["res_scale"])
            tidx_to_tensor[add_out_tidx] = out

    # out is now [1, C] after avg pool
    if out.ndim == 4:
        out = global_avg_pool_int8(out)

    # FC
    features = out[0].astype(np.int32)  # [1280]
    fc_w = fc_layer["w_int"]            # [1001, 1280]
    fc_b = fc_layer["b_int"]            # [1001]
    fc_os = fc_layer["output_scale"]    # [1001]

    acc = fc_w.astype(np.int32) @ features + fc_b.astype(np.int32)
    logits = np.clip(np.round(acc.astype(np.float64) * fc_os), -128, 127)

    return logits


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(img_path):
    """
    Load and preprocess an image for the network.
    Returns [1, 3, 224, 224] int8 tensor.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [-1, 1] (MobileNetV2 standard preprocessing)
    img_f = img_rgb.astype(np.float64) / 255.0
    img_norm = (img_f - 0.5) / 0.5  # [-1, 1]

    # Quantize to INT8 with scale = 1/127
    x_int8 = np.clip(np.round(img_norm * 127.0), -128, 127).astype(np.int8)

    # NCHW format
    x_nchw = x_int8.transpose(2, 0, 1)[np.newaxis, :, :, :]
    return x_nchw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TFLite MobileNetV2 -> Gemmini Per-Channel INT8 Simulation")
    print("=" * 70)

    # Load TFLite model
    print(f"\nLoading TFLite model: {TFLITE_PATH}")
    model, graph = load_tflite_model(TFLITE_PATH)
    print(f"  Operators: {graph.OperatorsLength()}, Tensors: {graph.TensorsLength()}")

    # Build quantized layers
    gemmini_layers, fc_layer = build_network(model, graph)
    n_conv = sum(1 for l in gemmini_layers if l["type"] in ("CONV2D", "DW_CONV"))
    print(f"\n  Total conv/dw layers: {n_conv}, plus AVG_POOL and FC")

    # Load images and labels
    print(f"\nLoading images from: {IMAGE_DIR}")
    files = sorted([f for f in os.listdir(IMAGE_DIR)
                    if f.lower().endswith((".jpeg", ".jpg", ".png"))])
    print(f"  Found {len(files)} images")

    print(f"Loading labels from: {LABELS_FILE}")
    with open(LABELS_FILE) as f:
        all_labels = [int(line.strip()) for line in f if line.strip().lstrip("-").isdigit()]
    print(f"  Loaded {len(all_labels)} labels")

    N = min(len(files), len(all_labels), NUM_IMAGES)
    print(f"\nRunning inference on {N} images...")
    print("-" * 70)

    top1_correct = 0
    top5_correct = 0

    for i in range(N):
        img_path = os.path.join(IMAGE_DIR, files[i])
        x_int8 = preprocess_image(img_path)
        if x_int8 is None:
            print(f"  [{i+1:4d}] SKIPPED (cannot load {files[i]})")
            continue

        label = all_labels[i]  # 0-999 ImageNet class

        logits = forward_pass_v2(x_int8, gemmini_layers, fc_layer)

        # TFLite has 1001 classes (class 0 = background).
        # Our labels are 0-999. TFLite class index = label + 1.
        # So we skip class 0 when finding the prediction.
        logits_no_bg = logits[1:]  # [1000] classes, index 0 = ImageNet class 0

        pred = int(np.argmax(logits_no_bg))
        top5_preds = set(int(x) for x in np.argsort(logits_no_bg)[-5:])

        if pred == label:
            top1_correct += 1
        if label in top5_preds:
            top5_correct += 1

        if i < 5:
            match = "OK" if pred == label else "MISS"
            print(f"  [{i+1:4d}] label={label:4d}  pred={pred:4d}  "
                  f"logit_range=[{logits.min():.0f},{logits.max():.0f}]  {match}")

        if (i + 1) % PRINT_EVERY == 0:
            t1_pct = 100.0 * top1_correct / (i + 1)
            t5_pct = 100.0 * top5_correct / (i + 1)
            print(f"  [{i+1:4d}/{N}]  Top-1: {top1_correct}/{i+1} ({t1_pct:.1f}%)  "
                  f"Top-5: {top5_correct}/{i+1} ({t5_pct:.1f}%)")

    print("-" * 70)
    print(f"\nFinal Results ({N} images):")
    print(f"  Top-1 Accuracy: {top1_correct}/{N} ({100.0*top1_correct/max(N,1):.1f}%)")
    print(f"  Top-5 Accuracy: {top5_correct}/{N} ({100.0*top5_correct/max(N,1):.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
