#!/usr/bin/env python3
"""
Transform resnet50_float.c -> resnet50_v1_float.c with mobilenet-style streaming.

Rules:
  A/B test block  (between A/B TEST markers):
    - 4-space `if (!conv) { ... } else { ... }` blocks -> keep only the else (conv) branch
    - 8-space conv_1 block is left untouched (has pool handling)

  Streaming batch loop (after `for (int batch_idx = 0`):
    - 4-space if/else blocks where BOTH branches are identical tiled_matmul_nn_auto
      calls (no im2col inside) -> collapse to a single direct call
    - Everything else -> keep as-is
"""

import sys

def find_4space_if_else_block(lines, start):
    """
    Given lines[start] == '    if (!conv) {', find the matching else and closing brace.
    Returns (else_idx, end_idx) where end_idx points to the '    }' closing line.
    Returns (None, None) if not found.
    """
    else_idx = None
    j = start + 1
    while j < len(lines):
        s = lines[j].rstrip()
        if s == '    } else {':
            else_idx = j
        elif s == '    }' and else_idx is not None:
            return else_idx, j
        j += 1
    return None, None


def normalize(body_lines):
    """Strip blank lines and trailing whitespace for comparison."""
    return [l.rstrip() for l in body_lines if l.strip()]


def transform(content):
    lines = content.split('\n')
    result = []
    i = 0
    in_ab_test = False
    in_streaming_loop = False

    while i < len(lines):
        line = lines[i]

        # ---- section tracking ----
        if '===== A/B TEST:' in line:
            in_ab_test = True
        if '===== END A/B TEST =====' in line:
            in_ab_test = False
        if 'for (int batch_idx = 0' in line:
            in_streaming_loop = True

        # ---- look for transformable blocks ----
        # Only 4-space `if (!conv) {` lines (not the 8-space conv_1 inside the loop)
        if line.rstrip() == '    if (!conv) {':
            else_idx, end_idx = find_4space_if_else_block(lines, i)

            if else_idx is None:
                # Safety: not a standard if/else block, keep as-is
                result.append(line)
                i += 1
                continue

            if_body   = lines[i + 1 : else_idx]      # between 'if (!conv) {' and '} else {'
            else_body = lines[else_idx + 1 : end_idx] # between '} else {' and '}'

            if_norm    = normalize(if_body)
            else_norm  = normalize(else_body)
            if_text    = '\n'.join(if_norm)

            if in_ab_test:
                # A/B test: always use only the else (conv) branch
                for el in else_body:
                    result.append(el)
                i = end_idx + 1

            elif in_streaming_loop:
                # Streaming loop: collapse only identical matmul-only branches
                if (if_norm == else_norm
                        and 'tiled_matmul_nn_auto' in if_text
                        and 'im2col' not in if_text):
                    # Both branches identical and purely matmul -> direct call
                    for el in else_body:
                        result.append(el)
                    i = end_idx + 1
                else:
                    # Keep as-is (3x3 conv, stride-2 skip, or heterogeneous bodies)
                    for j in range(i, end_idx + 1):
                        result.append(lines[j])
                    i = end_idx + 1

            else:
                # Outside both active sections -> keep as-is
                for j in range(i, end_idx + 1):
                    result.append(lines[j])
                i = end_idx + 1

        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


if __name__ == '__main__':
    src = 'resnet50_float.c'
    dst = 'resnet50_v1_float.c'

    with open(src, 'r') as f:
        content = f.read()

    result = transform(content)

    # Fix title
    result = result.replace(
        '"--- ResNet50 Float Streaming Inference ---"',
        '"--- ResNet50 V1 Float Streaming Inference ---"'
    )

    # Fix fc stats print format for float (already done in resnet50_float.c,
    # but double-check the REF print format in case it survived as int)
    result = result.replace(
        '        int fc_min = 127, fc_max = -128;\n',
        '        float fc_min = 1e30f, fc_max = -1e30f;\n'
    )
    result = result.replace(
        '            int v = fc_54_out[0][i];\n',
        '            float v = fc_54_out[0][i];\n'
    )
    result = result.replace(
        '        printf("  [REF] fc_54_out: min=%d, max=%d\\n", fc_min, fc_max);\n',
        '        printf("  [REF] fc_54_out: min=%.4f, max=%.4f\\n", fc_min, fc_max);\n'
    )

    with open(dst, 'w') as f:
        f.write(result)

    # Report stats
    in_lines  = content.split('\n')
    out_lines = result.split('\n')
    print(f"Input  lines: {len(in_lines)}")
    print(f"Output lines: {len(out_lines)}")
    print(f"Removed {len(in_lines) - len(out_lines)} lines (if/else wrappers collapsed)")
