#!/usr/bin/env python3
"""
Analyze the HuggingFace MobileNetV2 (google/mobilenet_v2_1.0_224) architecture.
Prints a complete layer-by-layer table including:
  - HuggingFace prefix, layer type, kernel, stride, channels, activation
  - Inverted residual block groupings with residual connection info
  - Global avg pool and FC dimensions
  - Total conv layer count verification (expected: 52 conv + 1 FC = 53)
"""

import torch
from transformers import MobileNetV2ForImageClassification

def get_conv_info(conv_layer_module, prefix_str):
    """Extract info from a MobileNetV2ConvLayer (has .convolution, .normalization, optionally .activation)."""
    c = conv_layer_module.convolution
    has_act = hasattr(conv_layer_module, 'activation') and conv_layer_module.activation is not None
    # Check if activation actually exists as a child
    act_found = False
    for name, child in conv_layer_module.named_children():
        if name == 'activation':
            act_found = True
    activation = "ReLU6" if act_found else "linear"

    is_depthwise = (c.groups == c.in_channels and c.groups > 1)
    if is_depthwise:
        ltype = "depthwise"
    elif c.kernel_size == (1, 1):
        ltype = "pointwise (1x1)"
    else:
        ltype = "regular"

    return {
        "prefix": prefix_str,
        "type": ltype,
        "kernel": tuple(c.kernel_size),
        "stride": tuple(c.stride),
        "in_ch": c.in_channels,
        "out_ch": c.out_channels,
        "groups": c.groups,
        "activation": activation,
    }


def main():
    print("=" * 140)
    print("Loading google/mobilenet_v2_1.0_224 from HuggingFace...")
    print("=" * 140)

    model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
    model.eval()

    conv_layers = []
    block_groups = []

    # =========================================================================
    # STEM: first_conv (3x3 regular) + conv_3x3 (depthwise) + reduce_1x1 (projection)
    # This is the initial conv + the first inverted residual block (t=1, c=16)
    # =========================================================================
    stem = model.mobilenet_v2.conv_stem
    stem_sublayers = ['first_conv', 'conv_3x3', 'reduce_1x1']
    stem_layer_names = []

    for sname in stem_sublayers:
        sl = getattr(stem, sname)
        prefix_str = f"mobilenet_v2.conv_stem.{sname}"
        info = get_conv_info(sl, prefix_str)
        info["block"] = "Stem"
        info["block_idx"] = -1
        conv_layers.append(info)
        stem_layer_names.append(prefix_str)

    # The stem contains the initial conv and the first inverted residual (t=1).
    # The first_conv is a standalone initial conv; conv_3x3+reduce_1x1 form
    # the first inverted residual block (no expansion since t=1).
    block_groups.append({
        "block_name": "Stem (initial conv + first inverted residual, t=1)",
        "layers_in_block": stem_layer_names,
        "has_residual": False,
        "residual_from": None,
        "residual_to": None,
        "note": "first_conv=initial 3x3 conv; conv_3x3+reduce_1x1=inverted residual block 0 (t=1, no expand)",
    })

    # =========================================================================
    # INVERTED RESIDUAL BLOCKS: layer.0 through layer.15
    # Each has: expand_1x1 -> conv_3x3 (depthwise) -> reduce_1x1 (projection)
    # =========================================================================
    for i, layer_module in enumerate(model.mobilenet_v2.layer):
        prefix = f"mobilenet_v2.layer.{i}"
        block_layer_names = []

        for sname in ['expand_1x1', 'conv_3x3', 'reduce_1x1']:
            sl = getattr(layer_module, sname, None)
            if sl is not None:
                prefix_str = f"{prefix}.{sname}"
                info = get_conv_info(sl, prefix_str)
                info["block"] = f"InvRes[{i}]"
                info["block_idx"] = i
                conv_layers.append(info)
                block_layer_names.append(prefix_str)

        has_residual = getattr(layer_module, 'use_residual', False)
        bg = {
            "block_name": f"Inverted Residual Block {i+1} (layer.{i})",
            "layers_in_block": block_layer_names,
            "has_residual": has_residual,
            "residual_from": None,
            "residual_to": None,
            "note": None,
        }
        if has_residual:
            bg["residual_from"] = f"input of {block_layer_names[0]}"
            bg["residual_to"] = f"output of {block_layer_names[-1]}"
        block_groups.append(bg)

    # =========================================================================
    # FINAL 1x1 CONV
    # =========================================================================
    final_conv_module = model.mobilenet_v2.conv_1x1
    info = get_conv_info(final_conv_module, "mobilenet_v2.conv_1x1")
    info["block"] = "Final 1x1"
    info["block_idx"] = -2
    conv_layers.append(info)

    # =========================================================================
    # PRINT LAYER-BY-LAYER TABLE
    # =========================================================================
    total_conv = len(conv_layers)

    print()
    print("=" * 140)
    print("LAYER-BY-LAYER CONV+BN ARCHITECTURE TABLE")
    print("=" * 140)
    print()

    header = (
        f"{'#':>3}  "
        f"{'HuggingFace Prefix':<50} "
        f"{'Block':<15} "
        f"{'Layer Type':<18} "
        f"{'Kernel':>6} "
        f"{'Stride':>6} "
        f"{'In_Ch':>7} "
        f"{'Out_Ch':>7} "
        f"{'Groups':>7} "
        f"{'Activation':<10}"
    )
    print(header)
    print("-" * 140)

    for i, layer in enumerate(conv_layers):
        k = f"{layer['kernel'][0]}x{layer['kernel'][1]}"
        s = f"{layer['stride'][0]}x{layer['stride'][1]}"
        row = (
            f"{i+1:>3}  "
            f"{layer['prefix']:<50} "
            f"{layer['block']:<15} "
            f"{layer['type']:<18} "
            f"{k:>6} "
            f"{s:>6} "
            f"{layer['in_ch']:>7} "
            f"{layer['out_ch']:>7} "
            f"{layer['groups']:>7} "
            f"{layer['activation']:<10}"
        )
        print(row)

    print("-" * 140)
    print(f"Total convolutional layers: {total_conv}")

    # =========================================================================
    # INVERTED RESIDUAL BLOCK SUMMARY
    # =========================================================================
    print()
    print("=" * 140)
    print("INVERTED RESIDUAL BLOCK SUMMARY")
    print("=" * 140)

    for bg in block_groups:
        print(f"\n  {bg['block_name']}:")
        for lname in bg["layers_in_block"]:
            # find matching conv layer info
            matched = [cl for cl in conv_layers if cl["prefix"] == lname][0]
            print(f"    {lname:<50}  {matched['type']:<18}  {matched['in_ch']:>4} -> {matched['out_ch']:<4}  {matched['activation']}")
        print(f"    Residual connection: {'YES' if bg['has_residual'] else 'NO'}")
        if bg["has_residual"]:
            print(f"      Skip from: {bg['residual_from']}")
            print(f"      Skip to:   {bg['residual_to']}  (element-wise add)")
        if bg.get("note"):
            print(f"    Note: {bg['note']}")

    # =========================================================================
    # GLOBAL AVERAGE POOLING
    # =========================================================================
    print()
    print("=" * 140)
    print("GLOBAL AVERAGE POOLING")
    print("=" * 140)

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = model.mobilenet_v2(dummy)
        last_hidden = features.last_hidden_state

    print(f"  Last hidden state shape (before pooling): {list(last_hidden.shape)}")
    print(f"  Channels:                                 {last_hidden.shape[1]}")
    print(f"  Spatial dimensions:                       {last_hidden.shape[2]} x {last_hidden.shape[3]}")
    print(f"  Global Average Pool:                      ({last_hidden.shape[2]} x {last_hidden.shape[3]}) -> (1 x 1)")
    print(f"  Output after pool + flatten:              ({last_hidden.shape[1]},)")

    # =========================================================================
    # FULLY CONNECTED (CLASSIFIER) LAYER
    # =========================================================================
    print()
    print("=" * 140)
    print("FULLY CONNECTED (CLASSIFIER) LAYER")
    print("=" * 140)

    fc = model.classifier
    print(f"  HuggingFace prefix: classifier")
    print(f"  Type:               Linear (Fully Connected)")
    print(f"  in_features:        {fc.in_features}")
    print(f"  out_features:       {fc.out_features}")

    # =========================================================================
    # VERIFICATION: TOTAL LAYER COUNT
    # =========================================================================
    print()
    print("=" * 140)
    print("VERIFICATION: TOTAL LAYER COUNT")
    print("=" * 140)

    total_with_fc = total_conv + 1
    expected = 53

    regular_count = sum(1 for l in conv_layers if l['type'] == 'regular')
    pw_count = sum(1 for l in conv_layers if l['type'] == 'pointwise (1x1)')
    dw_count = sum(1 for l in conv_layers if l['type'] == 'depthwise')

    print(f"  Convolutional layers counted: {total_conv}")
    print(f"    - Regular (3x3, stride):    {regular_count}")
    print(f"    - Pointwise (1x1):          {pw_count}")
    print(f"    - Depthwise (3x3, grouped): {dw_count}")
    print(f"  FC layers counted:            1")
    print(f"  Total (conv + FC):            {total_with_fc}")
    print(f"  Expected:                     {expected}")
    print()

    if total_with_fc == expected:
        print(f"  >>> VERIFIED: {total_with_fc} == {expected}  (52 conv + 1 FC = 53)")
    else:
        print(f"  >>> MISMATCH: got {total_with_fc}, expected {expected}")
        print(f"      Difference: {total_with_fc - expected}")

    print()
    print(f"  Detailed Breakdown:")
    print(f"    Stem first_conv (3x3 regular):          1")
    print(f"    Stem conv_3x3 (depthwise, t=1 block):   1")
    print(f"    Stem reduce_1x1 (projection, t=1 block): 1")
    print(f"    Inv. residual expand_1x1 layers:        {sum(1 for l in conv_layers if 'expand_1x1' in l['prefix'] and 'conv_stem' not in l['prefix'])}")
    print(f"    Inv. residual conv_3x3 (DW) layers:     {sum(1 for l in conv_layers if 'conv_3x3' in l['prefix'] and 'conv_stem' not in l['prefix'])}")
    print(f"    Inv. residual reduce_1x1 layers:        {sum(1 for l in conv_layers if 'reduce_1x1' in l['prefix'] and 'conv_stem' not in l['prefix'])}")
    print(f"    Final conv_1x1:                          1")
    print(f"    FC (classifier):                         1")
    subtotal = (1 + 1 + 1 +
                sum(1 for l in conv_layers if 'expand_1x1' in l['prefix'] and 'conv_stem' not in l['prefix']) +
                sum(1 for l in conv_layers if 'conv_3x3' in l['prefix'] and 'conv_stem' not in l['prefix']) +
                sum(1 for l in conv_layers if 'reduce_1x1' in l['prefix'] and 'conv_stem' not in l['prefix']) +
                1 + 1)
    print(f"    ============================================")
    print(f"    Grand total:                             {subtotal}")

    print()
    print("=" * 140)
    print("DONE")
    print("=" * 140)


if __name__ == "__main__":
    main()
