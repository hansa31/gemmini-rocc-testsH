#!/usr/bin/env python3
"""
apply_pc_mm.py
Modifies mobilenet_v1.c in-place to implement per-channel matmul dispatch (PC_MM).
"""

import re
import sys

FILEPATH = "/home/hansa/Desktop/chipyard/generators/gemmini/software/gemmini-rocc-tests/imagenet/mobilenet_v1.c"

# ---------------------------------------------------------------------------
# 1. Insertion block to add after the batch_images declaration
# ---------------------------------------------------------------------------
PC_MM_BLOCK = r"""
/* Per-channel matmul temp buffers (max pointwise conv dims) */
static elem_t _pc_w_col[960][1]   row_align(1);   /* max K for non-DW conv */
static elem_t _pc_out_col[50176][1] row_align(1);  /* max I = 4*112*112     */

/* PC_MM: per-channel matmul. OS_ is float[J] per-channel output scales.
 * Loops over J output channels; each call uses J=1 with its own output_scale.
 * Activation (RELU/NO_ACTIVATION) is applied manually in the output scatter.
 */
#define PC_MM(I_, J_, K_, A_, W_, B_, C_, ACT_, OS_, TYPE_) do {             \
    elem_t   * const _pc_C   = (elem_t*)(C_);                                \
    const float    * const _pc_os  = (const float*)(OS_);                    \
    const int        _pci = (I_), _pcj = (J_), _pck = (K_);                 \
    const elem_t   * const _pc_W   = (const elem_t*)(W_);                   \
    const acc_t    * const _pc_B   = (const acc_t*)(B_);                     \
    const int        _pc_act = (int)(ACT_);                                   \
    for (int _j = 0; _j < _pcj; _j++) {                                      \
        for (int _k = 0; _k < _pck; _k++)                                    \
            _pc_w_col[_k][0] = _pc_W[_k * _pcj + _j];                       \
        acc_t _pc_bias[1] = {_pc_B[_j]};                                     \
        tiled_matmul_nn_auto(_pci, 1, _pck,                                  \
            (A_), _pc_w_col, _pc_bias, _pc_out_col,                          \
            NO_ACTIVATION, _pc_os[_j], true, (TYPE_), false, "");            \
        for (int _ii = 0; _ii < _pci; _ii++) {                               \
            elem_t _v = _pc_out_col[_ii][0];                                 \
            if (_pc_act == (int)RELU && _v < 0) _v = 0;                      \
            _pc_C[_ii * _pcj + _j] = _v;                                     \
        }                                                                     \
    }                                                                         \
} while(0)"""

# ---------------------------------------------------------------------------
# Target non-DW, non-conv_1 layer names
# ---------------------------------------------------------------------------
TARGET_LAYERS = {
    "conv_3", "conv_4", "conv_6", "conv_7", "conv_9", "conv_10",
    "conv_12", "conv_13", "conv_15", "conv_16", "conv_18", "conv_19",
    "conv_21", "conv_22", "conv_24", "conv_25", "conv_27", "conv_28",
    "conv_30", "conv_31", "conv_33", "conv_34", "conv_36", "conv_37",
    "conv_39", "conv_40", "conv_42", "conv_43", "conv_45", "conv_46",
    "conv_48", "conv_49", "conv_51", "conv_52",
}

# ---------------------------------------------------------------------------
# Read the file
# ---------------------------------------------------------------------------
with open(FILEPATH, "r") as f:
    text = f.read()

original_text = text  # keep for comparison

# ---------------------------------------------------------------------------
# Step 1: Insert PC_MM block after batch_images declaration (line 44)
# ---------------------------------------------------------------------------
ANCHOR = "static elem_t batch_images[BATCH_SIZE * IMAGE_SIZE];"
assert ANCHOR in text, f"Anchor not found: {ANCHOR}"
text = text.replace(ANCHOR, ANCHOR + PC_MM_BLOCK, 1)
print(f"[1] Inserted PC_MM macro block after batch_images declaration.")

# ---------------------------------------------------------------------------
# Step 2a: Replace single-line A/B block calls for target layers
#
# Pattern:
#   tiled_matmul_nn_auto(conv_N_params.I, conv_N_params.J, conv_N_params.K,
#       INPUT, conv_N_w, conv_N_b, conv_N_out, ACT, conv_N_params.output_scale,
#       true, tiled_matmul_type, check, "conv_N");
#
# All on ONE line (the A/B block lines 187-246 area).
# ---------------------------------------------------------------------------

def replace_single_line_call(m):
    layer = m.group("layer")
    if layer not in TARGET_LAYERS:
        return m.group(0)  # leave unchanged
    I  = m.group("I")
    J  = m.group("J")
    K  = m.group("K")
    A  = m.group("A")
    W  = m.group("W")
    B  = m.group("B")
    C  = m.group("C")
    act = m.group("act")
    indent = m.group("indent")
    return (f"{indent}PC_MM({I}, {J}, {K}, {A}, {W}, {B}, {C}, "
            f"{act}, {layer}_os, tiled_matmul_type);")

# Single-line pattern (A/B reference block)
# tiled_matmul_nn_auto(conv_N_params.I, conv_N_params.J, conv_N_params.K,
#     INPUT, conv_N_w, conv_N_b, conv_N_out, ACT, conv_N_params.output_scale,
#     true, tiled_matmul_type, check, "conv_N");
single_line_pat = re.compile(
    r'^(?P<indent>[ \t]*)tiled_matmul_nn_auto\('
    r'(?P<I>\S+\.I),\s*(?P<J>\S+\.J),\s*(?P<K>\S+\.K),\s*'
    r'(?P<A>\S+),\s*(?P<W>\S+),\s*(?P<B>\S+),\s*(?P<C>\S+),\s*'
    r'(?P<act>\w+),\s*\S+\.output_scale,\s*true,\s*tiled_matmul_type,\s*check,\s*"(?P<layer>[^"]+)"\);$',
    re.MULTILINE
)

text, n1 = single_line_pat.subn(replace_single_line_call, text)
print(f"[2a] Replaced {n1} single-line A/B block tiled_matmul_nn_auto calls.")

# ---------------------------------------------------------------------------
# Step 2b: Replace multi-line main loop calls for target layers
#
# Pattern (4 lines):
#   tiled_matmul_nn_auto(conv_N_params.I, conv_N_params.J, conv_N_params.K,
#       INPUT, conv_N_w, conv_N_b, conv_N_out,
#       ACT, conv_N_params.output_scale, true,
#       tiled_matmul_type, check, "conv_N");
# ---------------------------------------------------------------------------

def replace_multi_line_call(m):
    layer = m.group("layer")
    if layer not in TARGET_LAYERS:
        return m.group(0)
    I      = m.group("I")
    J      = m.group("J")
    K      = m.group("K")
    A      = m.group("A")
    W      = m.group("W")
    B      = m.group("B")
    C      = m.group("C")
    act    = m.group("act")
    indent = m.group("indent")
    inner  = m.group("inner")   # whitespace before INPUT on line 2
    return (
        f"{indent}PC_MM({I}, {J}, {K},\n"
        f"{inner}{A}, {W}, {B}, {C},\n"
        f"{inner}{act}, {layer}_os, tiled_matmul_type);"
    )

multi_line_pat = re.compile(
    r'^(?P<indent>[ \t]*)tiled_matmul_nn_auto\('
    r'(?P<I>\S+\.I),\s*(?P<J>\S+\.J),\s*(?P<K>\S+\.K),\n'
    r'(?P<inner>[ \t]*)(?P<A>\S+),\s*(?P<W>\S+),\s*(?P<B>\S+),\s*(?P<C>\S+),\n'
    r'[ \t]*(?P<act>\w+),\s*\S+\.output_scale,\s*true,\n'
    r'[ \t]*tiled_matmul_type,\s*check,\s*"(?P<layer>[^"]+)";',
    re.MULTILINE
)

text, n2 = multi_line_pat.subn(replace_multi_line_call, text)
print(f"[2b] Replaced {n2} multi-line main loop tiled_matmul_nn_auto calls.")

# ---------------------------------------------------------------------------
# Step 3: Replace FC A/B block call
#
#         tiled_matmul_nn_auto(fc_53_params.I, fc_53_params.J, fc_53_params.K,
#             fc_53_w, ref_average, fc_53_b, fc_53_out,
#             NO_ACTIVATION, fc_53_params.output_scale, false,
#             tiled_matmul_type, check, "fc_53");
# ---------------------------------------------------------------------------
fc_ab_old = (
    r'tiled_matmul_nn_auto\(fc_53_params\.I, fc_53_params\.J, fc_53_params\.K,\n'
    r'[ \t]+fc_53_w, ref_average, fc_53_b, fc_53_out,\n'
    r'[ \t]+NO_ACTIVATION, fc_53_params\.output_scale, false,\n'
    r'[ \t]+tiled_matmul_type, check, "fc_53"\);'
)
fc_ab_new = (
    "for (int _fc_j = 0; _fc_j < 1000; _fc_j++) {\n"
    "            tiled_matmul_nn_auto(1, fc_53_params.J, fc_53_params.K,\n"
    "                &fc_53_w[_fc_j][0], ref_average, fc_53_b[_fc_j], &fc_53_out[_fc_j][0],\n"
    "                NO_ACTIVATION, fc_53_os[_fc_j], true,\n"
    "                tiled_matmul_type, false, \"\");\n"
    "        }"
)

text, nfc1 = re.subn(fc_ab_old, fc_ab_new, text)
print(f"[3a] Replaced {nfc1} FC A/B block call(s).")

# ---------------------------------------------------------------------------
# Step 4: Replace FC main loop call
#
#         tiled_matmul_nn_auto(fc_53_params.I, fc_53_params.J, fc_53_params.K,
#             fc_53_w, average, fc_53_b, fc_53_out,
#             NO_ACTIVATION, fc_53_params.output_scale, false,
#             tiled_matmul_type, check, "fc_53");
# ---------------------------------------------------------------------------
fc_main_old = (
    r'tiled_matmul_nn_auto\(fc_53_params\.I, fc_53_params\.J, fc_53_params\.K,\n'
    r'[ \t]+fc_53_w, average, fc_53_b, fc_53_out,\n'
    r'[ \t]+NO_ACTIVATION, fc_53_params\.output_scale, false,\n'
    r'[ \t]+tiled_matmul_type, check, "fc_53"\);'
)
fc_main_new = (
    "for (int _fc_j = 0; _fc_j < 1000; _fc_j++) {\n"
    "            tiled_matmul_nn_auto(1, fc_53_params.J, fc_53_params.K,\n"
    "                &fc_53_w[_fc_j][0], average, fc_53_b[_fc_j], &fc_53_out[_fc_j][0],\n"
    "                NO_ACTIVATION, fc_53_os[_fc_j], true,\n"
    "                tiled_matmul_type, false, \"\");\n"
    "        }"
)

text, nfc2 = re.subn(fc_main_old, fc_main_new, text)
print(f"[3b] Replaced {nfc2} FC main loop call(s).")

# ---------------------------------------------------------------------------
# Write modified file
# ---------------------------------------------------------------------------
with open(FILEPATH, "w") as f:
    f.write(text)

print(f"\nFile written: {FILEPATH}")

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
print("\n=== Verification ===")

pc_mm_count = text.count("PC_MM(")
print(f"  PC_MM( occurrences       : {pc_mm_count}  (expected 68)")

# tiled_matmul_nn_auto remaining (excluding those inside the macro definition)
# Count occurrences excluding the macro body definition lines
all_tmm = [m.start() for m in re.finditer(r'tiled_matmul_nn_auto', text)]
# Filter out those inside the #define block (they are part of the macro body)
# The macro body line contains: tiled_matmul_nn_auto(_pci, 1, _pck,
macro_body_hits = [m.start() for m in re.finditer(r'tiled_matmul_nn_auto\(_pci', text)]
# Also count those in fc_53 loop bodies
fc_body_hits = [m.start() for m in re.finditer(r'tiled_matmul_nn_auto\(1, fc_53_params', text)]

print(f"  Total tiled_matmul_nn_auto: {len(all_tmm)}")
print(f"    - in macro body         : {len(macro_body_hits)}")
print(f"    - in FC loop bodies     : {len(fc_body_hits)}")
remaining = len(all_tmm) - len(macro_body_hits) - len(fc_body_hits)
print(f"    - remaining real calls  : {remaining}  (expected conv_1 x4 = 4)")

fc_os_count = text.count("fc_53_os")
print(f"  fc_53_os occurrences      : {fc_os_count}  (expected 2)")

# Check conv_1 tiled_matmul_nn_auto calls
conv1_hits = [m.start() for m in re.finditer(r'tiled_matmul_nn_auto\(conv_1_params', text)]
print(f"  tiled_matmul_nn_auto(conv_1_params...) : {len(conv1_hits)}  (expected 2)")

print("\nDone.")
