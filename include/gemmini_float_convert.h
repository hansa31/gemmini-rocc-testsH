// gemmini_float_convert.h
// Software conversion between C float (FP32) and low-precision float bit
// patterns stored in uint16_t (elem_t).
//
// Supported formats:
//   FP16  — Float(5,11):  [1 sign][5 exp, bias=15 ][10 mant] = 16 bits
//   BF16  — Float(8,8):   [1 sign][8 exp, bias=127][7  mant] = 16 bits
//   FP12  — Float(5,7):   [1 sign][5 exp, bias=15 ][6  mant] = 12 bits (in uint16_t)
//
// Usage:
//   #include "include/gemmini_float_convert.h"
//
//   float x = 1.5f;
//   elem_t fp16_val = float_to_fp16_bits(x);   // 0x3E00
//   float  back     = fp16_bits_to_float(fp16_val);  // 1.5f
//
// These functions are portable C (no FP16 hardware needed).

#ifndef GEMMINI_FLOAT_CONVERT_H
#define GEMMINI_FLOAT_CONVERT_H

#include <stdint.h>
#include <string.h>

// ============================================================================
// Internal helper: read FP32 bit pattern
// ============================================================================
static inline uint32_t _gfc_float_to_bits(float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    return bits;
}

static inline float _gfc_bits_to_float(uint32_t bits) {
    float val;
    memcpy(&val, &bits, sizeof(val));
    return val;
}

// ============================================================================
// FP16 — IEEE 754 half-precision: Float(5,11)
//   [1 sign][5 exp, bias=15][10 mantissa] = 16 bits
//   Range: +-65504, smallest normal: 2^-14 ~ 6.1e-5
// ============================================================================

static inline uint16_t float_to_fp16_bits(float val) {
    uint32_t bits = _gfc_float_to_bits(val);
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127;   // unbias FP32 exponent
    uint32_t mant = bits & 0x7FFFFF;                // 23-bit mantissa

    // FP32 Inf or NaN
    if (exp == 128) {
        if (mant) return (sign << 15) | 0x7C01;    // quiet NaN
        return (sign << 15) | 0x7C00;              // Inf
    }

    // Rebias: FP32 bias=127 -> FP16 bias=15
    int32_t fp16_exp = exp + 15;

    if (fp16_exp >= 31) return (sign << 15) | 0x7C00;   // overflow -> Inf
    if (fp16_exp <= 0)  return (sign << 15);              // underflow -> zero

    // Truncate mantissa: 23 -> 10 bits
    uint16_t fp16_mant = (mant >> 13) & 0x3FF;

    return (sign << 15) | (fp16_exp << 10) | fp16_mant;
}

static inline float fp16_bits_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t fp32_bits;
    if (exp == 0) {
        if (mant == 0) {
            fp32_bits = sign << 31;                  // +-0
        } else {
            // subnormal: flush to zero (matches hardware behavior)
            fp32_bits = sign << 31;
        }
    } else if (exp == 31) {
        if (mant) fp32_bits = (sign << 31) | 0x7FC00000;  // NaN
        else      fp32_bits = (sign << 31) | 0x7F800000;  // Inf
    } else {
        uint32_t fp32_exp = exp - 15 + 127;         // rebias
        fp32_bits = (sign << 31) | (fp32_exp << 23) | (mant << 13);
    }

    return _gfc_bits_to_float(fp32_bits);
}

// ============================================================================
// BF16 — bfloat16: Float(8,8)
//   [1 sign][8 exp, bias=127][7 mantissa] = 16 bits
//   Same exponent range as FP32, reduced mantissa precision.
// ============================================================================

static inline uint16_t float_to_bf16_bits(float val) {
    // BF16 is just the upper 16 bits of FP32
    uint32_t bits = _gfc_float_to_bits(val);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_bits_to_float(uint16_t h) {
    // Pad lower 16 bits with zeros to reconstruct FP32
    uint32_t fp32_bits = (uint32_t)h << 16;
    return _gfc_bits_to_float(fp32_bits);
}

// ============================================================================
// FP12 — custom Float(5,7): stored in lower 12 bits of uint16_t
//   [1 sign][5 exp, bias=15][6 mantissa] = 12 bits
//   Upper 4 bits of the uint16_t are zero.
// ============================================================================

static inline uint16_t float_to_fp12_bits(float val) {
    uint32_t bits = _gfc_float_to_bits(val);
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp == 128) {
        if (mant) return (sign << 11) | ((0x1F) << 6) | 1;  // NaN
        return (sign << 11) | ((0x1F) << 6);                 // Inf
    }

    int32_t fp12_exp = exp + 15;

    if (fp12_exp >= 31) return (sign << 11) | ((0x1F) << 6);  // overflow -> Inf
    if (fp12_exp <= 0)  return (sign << 11);                    // underflow -> zero

    // Truncate mantissa: 23 -> 6 bits
    uint16_t fp12_mant = (mant >> 17) & 0x3F;

    return (sign << 11) | (fp12_exp << 6) | fp12_mant;
}

static inline float fp12_bits_to_float(uint16_t h) {
    // Only lower 12 bits are meaningful
    h &= 0x0FFF;

    uint32_t sign = (h >> 11) & 1;
    uint32_t exp  = (h >> 6) & 0x1F;
    uint32_t mant = h & 0x3F;

    uint32_t fp32_bits;
    if (exp == 0) {
        fp32_bits = sign << 31;                      // flush subnormals to zero
    } else if (exp == 31) {
        if (mant) fp32_bits = (sign << 31) | 0x7FC00000;
        else      fp32_bits = (sign << 31) | 0x7F800000;
    } else {
        uint32_t fp32_exp = exp - 15 + 127;
        fp32_bits = (sign << 31) | (fp32_exp << 23) | (mant << 17);
    }

    return _gfc_bits_to_float(fp32_bits);
}

// ============================================================================
// Generic wrappers — select based on gemmini_params.h defines
// ============================================================================
// These use ELEM_T_EXP_BITS / ELEM_T_SIG_BITS if available to pick the
// right converter automatically.  Falls back to FP16 if not defined.

#if defined(ELEM_T_IS_FLOAT)

static inline uint16_t float_to_elem_bits(float val) {
#if   ELEM_T_EXP_BITS == 5 && ELEM_T_SIG_BITS == 11
    return float_to_fp16_bits(val);
#elif ELEM_T_EXP_BITS == 8 && ELEM_T_SIG_BITS == 8
    return float_to_bf16_bits(val);
#elif ELEM_T_EXP_BITS == 5 && ELEM_T_SIG_BITS == 7
    return float_to_fp12_bits(val);
#else
    #error "float_to_elem_bits: unsupported ELEM_T_EXP_BITS/ELEM_T_SIG_BITS"
#endif
}

static inline float elem_bits_to_float(uint16_t h) {
#if   ELEM_T_EXP_BITS == 5 && ELEM_T_SIG_BITS == 11
    return fp16_bits_to_float(h);
#elif ELEM_T_EXP_BITS == 8 && ELEM_T_SIG_BITS == 8
    return bf16_bits_to_float(h);
#elif ELEM_T_EXP_BITS == 5 && ELEM_T_SIG_BITS == 7
    return fp12_bits_to_float(h);
#else
    #error "elem_bits_to_float: unsupported ELEM_T_EXP_BITS/ELEM_T_SIG_BITS"
#endif
}

#endif // ELEM_T_IS_FLOAT

#endif // GEMMINI_FLOAT_CONVERT_H
