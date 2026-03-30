# gemmini_float_convert.h — Float Conversion Utilities

Software conversion between C `float` (FP32) and the low-precision float bit
patterns stored in `uint16_t` (`elem_t`) for Gemmini's custom float multipliers.

## Location

```
include/gemmini_float_convert.h
```

## Supported Formats

| Format | Chisel Type   | Bit Layout                         | C type     |
|--------|---------------|------------------------------------|------------|
| FP16   | Float(5,11)   | `[1 sign][5 exp, bias=15][10 mant]` | `uint16_t` |
| BF16   | Float(8,8)    | `[1 sign][8 exp, bias=127][7 mant]` | `uint16_t` |
| FP12   | Float(5,7)    | `[1 sign][5 exp, bias=15][6 mant]`  | `uint16_t` (lower 12 bits) |

## Functions

### Format-Specific

```c
// FP16
uint16_t float_to_fp16_bits(float val);
float    fp16_bits_to_float(uint16_t h);

// BF16
uint16_t float_to_bf16_bits(float val);
float    bf16_bits_to_float(uint16_t h);

// FP12
uint16_t float_to_fp12_bits(float val);
float    fp12_bits_to_float(uint16_t h);
```

### Generic (auto-selects based on gemmini_params.h)

When `ELEM_T_IS_FLOAT` is defined (i.e., you have a float Gemmini config),
these wrappers pick the right converter using `ELEM_T_EXP_BITS` and
`ELEM_T_SIG_BITS`:

```c
uint16_t float_to_elem_bits(float val);   // FP32 -> elem_t bit pattern
float    elem_bits_to_float(uint16_t h);  // elem_t bit pattern -> FP32
```

**Requirement:** `gemmini_params.h` must be included before this header
(it is, if you include `gemmini_testutils.h` first).

## Usage Examples

### Basic: encode a known float into elem_t

```c
#include "include/gemmini_testutils.h"
#include "include/gemmini_float_convert.h"

float x = 1.5f;
elem_t encoded = float_to_elem_bits(x);   // e.g. 0x3E00 for FP16
float  back    = elem_bits_to_float(encoded);  // 1.5f
```

### Initialize a matrix with real float values

The standard Gemmini tests do `A[i][j] = rand() % 2`, which stores raw integer
0 or 1 as `uint16_t` — these are subnormal or zero in FP16. For correctness
testing, use proper encoding:

```c
static elem_t A[N][K] row_align(1);

for (int i = 0; i < N; i++)
    for (int j = 0; j < K; j++)
        A[i][j] = float_to_elem_bits(some_float_value);
```

### Run matmul and compare against CPU reference

```c
// Hardware matmul with full_C=true -> output as acc_t (FP32)
tiled_matmul_auto(I, J, K,
    (elem_t*)A, (elem_t*)B, NULL, (acc_t*)hw_C,
    K, J, J, J,
    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
    false, false,
    true, false,   // full_C=true
    0, WS);

// CPU reference: decode, multiply in FP32, accumulate
for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += elem_bits_to_float(A[i][k]) * elem_bits_to_float(B[k][j]);
        cpu_C[i][j] = sum;
    }

// Compare hw_C vs cpu_C with tolerance
```

See `bareMetalC/tiled_matmul_fp_verify.c` for a complete working example.

## Building and Running

### Build the verification test

```bash
cd software/gemmini-rocc-tests/
./build_single.sh bareMetalC tiled_matmul_fp_verify-baremetal
```

### Run on the simulator

```bash
# From sims/verilator/
./simulator-chipyard.lab-GemminiRocketConfig \
    ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/tiled_matmul_fp_verify-baremetal
```

Or with a pre-built simulator:

```bash
./original_simulators/FP16xFP32_4x4_WS/simulator-chipyard.lab-GemminiRocketConfig \
    ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/tiled_matmul_fp_verify-baremetal
```

## Switching Between Float Formats

The generic wrappers (`float_to_elem_bits` / `elem_bits_to_float`) automatically
adapt to whatever format is defined in `gemmini_params.h`. To switch formats:

1. Copy the correct `gemmini_params.h` into `include/`:
   ```bash
   cp ../../sims/verilator/original_simulators/FP16xFP32_4x4_WS/gemmini_params.h include/
   ```
2. Rebuild: `./build_single.sh bareMetalC tiled_matmul_fp_verify-baremetal`
3. Run with the matching simulator

The format-specific functions (`float_to_fp16_bits`, `float_to_bf16_bits`, etc.)
work regardless of the current `gemmini_params.h` — use these when you need to
handle a specific format explicitly.

## Notes

- **Subnormals are flushed to zero** — this matches the hardware behavior in
  `IntVerilogFloatMul.scala` (denormal inputs produce zero output).
- **Truncation, not rounding** — mantissa bits are truncated during conversion.
  This is simpler and matches the hardware's truncation behavior.
- **BF16 conversion is trivial** — `float_to_bf16_bits` just takes the upper
  16 bits of the FP32 representation. `bf16_bits_to_float` pads with zeros.
- **FP12 uses lower 12 bits** — the upper 4 bits of the `uint16_t` are zero.
