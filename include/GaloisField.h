#ifndef GaloisField_H
#define GaloisField_H

/** \page GF256 GF(256) Math Module

    This module provides efficient implementations of bulk
    GF(2^^8) math operations over memory buffers.

    Addition is done over the base field in GF(2) meaning
    that addition is XOR between memory buffers.

    Multiplication is performed using table lookups via
    SIMD instructions.  This is somewhat slower than XOR,
    but fast enough to not become a major bottleneck when
    used sparingly.
*/

#include <stdint.h> // uint32_t etc
#include <cstring>  // memcpy, memset
#include <memory>

/// Library header version
#define GF256_VERSION 2

//------------------------------------------------------------------------------
// Platform/Architecture

#if defined(__AVX2__) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
    #define GF256_TRY_AVX2 /* 256-bit */
    #include <immintrin.h>
    #define GF256_ALIGN_BYTES 32
#else // __AVX2__
    #define GF256_ALIGN_BYTES 16
#endif // __AVX2__

#if !defined(GF256_TARGET_MOBILE)
    #include <tmmintrin.h> // SSSE3: _mm_shuffle_epi8
    #include <emmintrin.h> // SSE2
#endif                 // GF256_TARGET_MOBILE


// Compiler-specific 128-bit SIMD register keyword
#define GF256_M128 __m128i


// Compiler-specific 256-bit SIMD register keyword
#ifdef GF256_TRY_AVX2
    #define GF256_M256 __m256i
#endif

// Compiler-specific C++11 restrict keyword
#define GF256_RESTRICT __restrict

// Compiler-specific force inline keyword
#ifdef _MSC_VER
    #define GF256_FORCE_INLINE inline __forceinline
#else
    #define GF256_FORCE_INLINE inline __attribute__((always_inline))
#endif

// Compiler-specific alignment keyword
// Note: Alignment only matters for ARM NEON where it should be 16
#ifdef _MSC_VER
    #define GF256_ALIGNED __declspec(align(GF256_ALIGN_BYTES))
#else // _MSC_VER
    #define GF256_ALIGNED __attribute__((aligned(GF256_ALIGN_BYTES)))
#endif // _MSC_VER

//------------------------------------------------------------------------------
// Portability

/// Swap two memory buffers in-place
void gf256_memswap(void *GF256_RESTRICT vx, void *GF256_RESTRICT vy, int bytes);

//------------------------------------------------------------------------------
// GF(256) Context

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4324) // warning C4324: 'gf256_ctx' : structure was padded due to __declspec(align())
#endif                          // _MSC_VER

/// The context object stores tables required to perform library calculations
class gf256_ctx
{
public:
    /// We require memory to be aligned since the SIMD instructions benefit from
    /// or require aligned accesses to the table data.
    struct
    {
        GF256_ALIGNED GF256_M128 TABLE_LO_Y[256];
        GF256_ALIGNED GF256_M128 TABLE_HI_Y[256];
    } MM128;
#ifdef GF256_TRY_AVX2
    struct
    {
        GF256_ALIGNED GF256_M256 TABLE_LO_Y[256];
        GF256_ALIGNED GF256_M256 TABLE_HI_Y[256];
    } MM256;
#endif // GF256_TRY_AVX2

    /// Mul/Div/Inv/Sqr tables
    uint8_t GF256_MUL_TABLE[256 * 256];
    uint8_t GF256_DIV_TABLE[256 * 256];
    uint8_t GF256_INV_TABLE[256];
    uint8_t GF256_SQR_TABLE[256];

    /// Log/Exp tables
    uint16_t GF256_LOG_TABLE[256];
    uint8_t GF256_EXP_TABLE[512 * 2 + 1];

    /// Polynomial used
    unsigned Polynomial;

    static gf256_ctx* getGF256Ctx()
    {
        static gf256_ctx* ctx_ptr = new gf256_ctx();
        return ctx_ptr;
    }

    //------------------------------------------------------------------------------
    // Initialization

    /**
        Initialize a context, filling in the tables.

        Thread-safety / Usage Notes:

        It is perfectly safe and encouraged to use a gf256_ctx object from multiple
        threads.  The gf256_init() is relatively expensive and should only be done
        once, though it will take less than a millisecond.

        The gf256_ctx object must be aligned to 16 byte boundary.
        Simply tag the object with GF256_ALIGNED to achieve this.

        Example:
           static GF256_ALIGNED gf256_ctx TheGF256Context;
           gf256_init(&TheGF256Context, 0);

        Returns 0 on success and other values on failure.
    */
    int gf256_init();
    void gf256_architecture_init();
    void gf256_poly_init(int polynomialIndex);
    void gf256_explog_init();
    void gf256_muldiv_init();
    void gf256_inv_init();
    void gf256_sqr_init();
    void gf256_mul_mem_init();
    bool gf256_self_test();

    //------------------------------------------------------------------------------
    // Bulk Memory Math Operations

    /// Performs "x[] += y[]" bulk memory XOR operation
    void gf256_add_mem(void *GF256_RESTRICT vx,
                       const void *GF256_RESTRICT vy, int bytes);

    /// Performs "z[] += x[] + y[]" bulk memory operation
    void gf256_add2_mem(void *GF256_RESTRICT vz, const void *GF256_RESTRICT vx,
                        const void *GF256_RESTRICT vy, int bytes);

    /// Performs "z[] = x[] + y[]" bulk memory operation
    void gf256_addset_mem(void *GF256_RESTRICT vz, const void *GF256_RESTRICT vx,
                          const void *GF256_RESTRICT vy, int bytes);

    /// Performs "z[] = x[] * y" bulk memory operation
    void gf256_mul_mem(void *GF256_RESTRICT vz,
                       const void *GF256_RESTRICT vx, uint8_t y, int bytes);

    /// Performs "z[] += x[] * y" bulk memory operation
    void gf256_muladd_mem(void *GF256_RESTRICT vz, uint8_t y,
                          const void *GF256_RESTRICT vx, int bytes);

    /// Performs "x[] /= y" bulk memory operation
    GF256_FORCE_INLINE void gf256_div_mem(void *GF256_RESTRICT vz,
                                                 const void *GF256_RESTRICT vx, uint8_t y, int bytes)
    {
        // Multiply by inverse
        gf256_mul_mem(vz, vx, y == 1 ? (uint8_t)1 : GF256_INV_TABLE[y], bytes);
    }

    //------------------------------------------------------------------------------
    // Math Operations

    /// return x + y
    static GF256_FORCE_INLINE uint8_t gf256_add(uint8_t x, uint8_t y)
    {
        return (uint8_t)(x ^ y);
    }

    /// return x * y
    /// For repeated multiplication by a constant, it is faster to put the constant in y.
    GF256_FORCE_INLINE uint8_t gf256_mul(uint8_t x, uint8_t y)
    {
        return GF256_MUL_TABLE[((unsigned)y << 8) + x];
    }

    /// return x / y
    /// Memory-access optimized for constant divisors in y.
    GF256_FORCE_INLINE uint8_t gf256_div(uint8_t x, uint8_t y)
    {
        return GF256_DIV_TABLE[((unsigned)y << 8) + x];
    }

    /// return 1 / x
    GF256_FORCE_INLINE uint8_t gf256_inv(uint8_t x)
    {
        return GF256_INV_TABLE[x];
    }

    /// return x * x
    GF256_FORCE_INLINE uint8_t gf256_sqr(uint8_t x)
    {
        return GF256_SQR_TABLE[x];
    }


private:
    gf256_ctx(){
        gf256_init();
    };
};

#ifdef _MSC_VER
    #pragma warning(pop)
#endif // _MSC_VER

#endif // GaloisField_H
