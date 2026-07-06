/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/optimized/utils/math_utils.h>
#include <executorch/kernels/optimized/utils/unroll.h>

#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

#include <array>
#include <type_traits>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

namespace executorch {
namespace cpublas {

template <typename scalar_t, typename opmath_t>
void scale_(int64_t m, int64_t n, opmath_t alpha, scalar_t* a, int64_t lda) {
  if (alpha == opmath_t(1)) {
    return; // identity
  }

  if (alpha == opmath_t(0)) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < m; ++i) {
        a[j * lda + i] = scalar_t(0);
      }
    }
    return;
  }

  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      a[j * lda + i] *= alpha;
    }
  }
}

template <typename Func>
auto sum(int64_t N, Func f) {
  constexpr int ilp_factor = 4;
  using acc_t = decltype(f(0));

  // Calculate independent partial sums then add together at the end
  std::array<acc_t, ilp_factor> partial_sums{};

  size_t i = 0;
  for (; i + ilp_factor <= N; i += ilp_factor) {
    utils::ForcedUnroll<ilp_factor>{}(
        [&i, &f, &partial_sums](int k) { partial_sums[k] += f(i + k); });
  }
  for (; i < N; ++i) {
    partial_sums[0] += f(i);
  }
  for (int k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }
  return partial_sums[0];
}

template <typename scalar_t, typename opmath_t>
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b)
  for (size_t l = 0; l < k; ++l) {
    for (size_t j = 0; j < n; ++j) {
      opmath_t val = b[l + j * ldb] * alpha;
      int64_t i_m = m / 4;
      for (int64_t i_i = 0; i_i < i_m; ++i_i) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      int64_t i = i_m * 4;
      for (; i < m; i++) {
        c[j * ldc + i] += a[i + l * lda] * val;
      }
    }
  }
}

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
template <typename scalar_t, typename opmath_t>
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c += alpha * (a @ b)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[l * lda + i]) *
            static_cast<opmath_t>(b[j * ldb + l]);
      });
      if (beta == opmath_t(0)) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}

#ifdef __riscv_vector
namespace rvv_detail {
inline float rvv_bf16_dot_f32(
    const torch::executor::BFloat16* a,
    const torch::executor::BFloat16* b,
    int64_t len)
{
    const uint16_t* pa = reinterpret_cast<const uint16_t*>(a);
    const uint16_t* pb = reinterpret_cast<const uint16_t*>(b);

    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vacc = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);

    int64_t n = len;
    
    while (n >= (int64_t)vlmax) {
        size_t vl = vlmax;
     	
     	//BFloat16 to float32, for better precision
     	vuint16m1_t va16 = __riscv_vle16_v_u16m1(pa, vl);
        vuint32m2_t va32 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(va16, vl), 16, vl);
        vfloat32m2_t vaf = __riscv_vreinterpret_v_u32m2_f32m2(va32);

        vuint16m1_t vb16 = __riscv_vle16_v_u16m1(pb, vl);
        vuint32m2_t vb32 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16, vl), 16, vl);
        vfloat32m2_t vbf = __riscv_vreinterpret_v_u32m2_f32m2(vb32);

        vacc = __riscv_vfmacc_vv_f32m2(vacc, vaf, vbf, vl);

        pa += vl;
        pb += vl;
        n -= (int64_t)vl;
       
    }
    
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)n);
	
        vuint16m1_t va16 = __riscv_vle16_v_u16m1(pa, vl);
        vuint32m2_t va32 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(va16, vl), 16, vl);
        vfloat32m2_t vaf = __riscv_vreinterpret_v_u32m2_f32m2(va32);

        vuint16m1_t vb16 = __riscv_vle16_v_u16m1(pb, vl);
        vuint32m2_t vb32 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16, vl), 16, vl);
        vfloat32m2_t vbf = __riscv_vreinterpret_v_u32m2_f32m2(vb32);

        vacc = __riscv_vfmacc_vv_f32m2(vacc, vaf, vbf, vl);

        pa += vl;
        pb += vl;
        n -= (int64_t)vl;
    }
    
    //vector vacc to one float number
    size_t vl_red = __riscv_vsetvl_e32m2((size_t)(len < (int64_t)vlmax ? len : (int64_t)vlmax));
    vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m2_f32m1(vacc, vzero, vl_red);
    return __riscv_vfmv_f_s_f32m1_f32(vsum);
}

inline void rvv_bf16_dot_f32_tile(
    const torch::executor::BFloat16* a_col,        // one A col with length k
    const torch::executor::BFloat16* const* b_cols, // TILE_N B cols each has length k
    int64_t k,
    float* out_dots)                                 // out: TILE_N float
{
    const uint16_t* pa = reinterpret_cast<const uint16_t*>(a_col);
    uint16_t const* pb0 = reinterpret_cast<const uint16_t*>(b_cols[0]);
    uint16_t const* pb1 = reinterpret_cast<const uint16_t*>(b_cols[1]);
    uint16_t const* pb2 = reinterpret_cast<const uint16_t*>(b_cols[2]);
    uint16_t const* pb3 = reinterpret_cast<const uint16_t*>(b_cols[3]);
 
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    
    vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
 
    int64_t n = k;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)n);
 
        // Load A once
        vuint16m1_t va16 = __riscv_vle16_v_u16m1(pa, vl);
        vuint32m2_t va32 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(va16, vl), 16, vl);
        vfloat32m2_t vaf = __riscv_vreinterpret_v_u32m2_f32m2(va32);
 
        // unrolled loads and multiply-accumulates for each B column
        vuint16m1_t vb16_0 = __riscv_vle16_v_u16m1(pb0, vl);
        vuint32m2_t vb32_0 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16_0, vl), 16, vl);
        vfloat32m2_t vbf_0 = __riscv_vreinterpret_v_u32m2_f32m2(vb32_0);
        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, vaf, vbf_0, vl);
        pb0 += vl;

        vuint16m1_t vb16_1 = __riscv_vle16_v_u16m1(pb1, vl);
        vuint32m2_t vb32_1 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16_1, vl), 16, vl);
        vfloat32m2_t vbf_1 = __riscv_vreinterpret_v_u32m2_f32m2(vb32_1);
        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, vaf, vbf_1, vl);
        pb1 += vl;

        vuint16m1_t vb16_2 = __riscv_vle16_v_u16m1(pb2, vl);
        vuint32m2_t vb32_2 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16_2, vl), 16, vl);
        vfloat32m2_t vbf_2 = __riscv_vreinterpret_v_u32m2_f32m2(vb32_2);
        vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, vaf, vbf_2, vl);
        pb2 += vl;

        vuint16m1_t vb16_3 = __riscv_vle16_v_u16m1(pb3, vl);
        vuint32m2_t vb32_3 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16_3, vl), 16, vl);
        vfloat32m2_t vbf_3 = __riscv_vreinterpret_v_u32m2_f32m2(vb32_3);
        vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, vaf, vbf_3, vl);
        pb3 += vl;
 
        pa += vl;
        n -= (int64_t)vl;
    }
 
    // vector reduction out of the loop
    size_t vl_red = __riscv_vsetvl_e32m2((size_t)(k < (int64_t)vlmax ? k : (int64_t)vlmax));
    
    vfloat32m1_t vsum0 = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, vzero, vl_red);
    out_dots[0] = __riscv_vfmv_f_s_f32m1_f32(vsum0);

    vfloat32m1_t vsum1 = __riscv_vfredusum_vs_f32m2_f32m1(vacc1, vzero, vl_red);
    out_dots[1] = __riscv_vfmv_f_s_f32m1_f32(vsum1);

    vfloat32m1_t vsum2 = __riscv_vfredusum_vs_f32m2_f32m1(vacc2, vzero, vl_red);
    out_dots[2] = __riscv_vfmv_f_s_f32m1_f32(vsum2);

    vfloat32m1_t vsum3 = __riscv_vfredusum_vs_f32m2_f32m1(vacc3, vzero, vl_red);
    out_dots[3] = __riscv_vfmv_f_s_f32m1_f32(vsum3);
}

inline void rvv_bf16_gemv_2a_1b(
    const torch::executor::BFloat16* pa0,
    const torch::executor::BFloat16* pa1,
    const torch::executor::BFloat16* b_col,
    int64_t k,
    float* out_dots)
{
    const uint16_t* p_a0 = reinterpret_cast<const uint16_t*>(pa0);
    const uint16_t* p_a1 = reinterpret_cast<const uint16_t*>(pa1);
    const uint16_t* pb   = reinterpret_cast<const uint16_t*>(b_col);
 
    size_t vlmax = __riscv_vsetvlmax_e32m2();
 
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
 
    vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
 
    int64_t n = k;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)n);
 
        vuint16m1_t vb16 = __riscv_vle16_v_u16m1(pb, vl);
        vuint32m2_t vb32 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16, vl), 16, vl);
        vfloat32m2_t vbf = __riscv_vreinterpret_v_u32m2_f32m2(vb32);
 
 	asm volatile("" : : : "memory");
 	
        vuint16m1_t va16_0 = __riscv_vle16_v_u16m1(p_a0, vl);
        vuint32m2_t va32_0 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(va16_0, vl), 16, vl);
        vfloat32m2_t vaf_0 = __riscv_vreinterpret_v_u32m2_f32m2(va32_0);
        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, vaf_0, vbf, vl);
        p_a0 += vl;
        
        asm volatile("" : : : "memory");
 
        vuint16m1_t va16_1 = __riscv_vle16_v_u16m1(p_a1, vl);
        vuint32m2_t va32_1 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(va16_1, vl), 16, vl);
        vfloat32m2_t vaf_1 = __riscv_vreinterpret_v_u32m2_f32m2(va32_1);
        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, vaf_1, vbf, vl);
        p_a1 += vl;
 
        pb += vl;
        n -= (int64_t)vl;
    }
 
    size_t vl_red = __riscv_vsetvl_e32m2((size_t)(k < (int64_t)vlmax ? k : (int64_t)vlmax));
 
    vfloat32m1_t vsum0 = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, vzero, vl_red);
    out_dots[0] = __riscv_vfmv_f_s_f32m1_f32(vsum0);
 
    vfloat32m1_t vsum1 = __riscv_vfredusum_vs_f32m2_f32m1(vacc1, vzero, vl_red);
    out_dots[1] = __riscv_vfmv_f_s_f32m1_f32(vsum1);

}

inline void rvv_bf16_dot_f32_tile_2b(
    const torch::executor::BFloat16* a_col,
    const torch::executor::BFloat16* pb0,
    const torch::executor::BFloat16* pb1,
    int64_t k,
    float* out_dots)
{
    const uint16_t* pa   = reinterpret_cast<const uint16_t*>(a_col);
    const uint16_t* p_b0 = reinterpret_cast<const uint16_t*>(pb0);
    const uint16_t* p_b1 = reinterpret_cast<const uint16_t*>(pb1);
 
    size_t vlmax = __riscv_vsetvlmax_e32m2();
 
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
 
    vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
 
    int64_t n = k;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)n);
 
        vuint16m1_t va16 = __riscv_vle16_v_u16m1(pa, vl);
        vuint32m2_t va32 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(va16, vl), 16, vl);
        vfloat32m2_t vaf = __riscv_vreinterpret_v_u32m2_f32m2(va32);
        
        asm volatile("" : : : "memory");
 
        vuint16m1_t vb16_0 = __riscv_vle16_v_u16m1(p_b0, vl);
        vuint32m2_t vb32_0 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16_0, vl), 16, vl);
        vfloat32m2_t vbf_0 = __riscv_vreinterpret_v_u32m2_f32m2(vb32_0);
        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, vaf, vbf_0, vl);
        p_b0 += vl;
        
        asm volatile("" : : : "memory");
 
        vuint16m1_t vb16_1 = __riscv_vle16_v_u16m1(p_b1, vl);
        vuint32m2_t vb32_1 = __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(vb16_1, vl), 16, vl);
        vfloat32m2_t vbf_1 = __riscv_vreinterpret_v_u32m2_f32m2(vb32_1);
        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, vaf, vbf_1, vl);
        p_b1 += vl;
 
 
        pa += vl;
        n -= (int64_t)vl;
    }
 
    size_t vl_red = __riscv_vsetvl_e32m2((size_t)(k < (int64_t)vlmax ? k : (int64_t)vlmax));
 
    vfloat32m1_t vsum0 = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, vzero, vl_red);
    out_dots[0] = __riscv_vfmv_f_s_f32m1_f32(vsum0);
 
    vfloat32m1_t vsum1 = __riscv_vfredusum_vs_f32m2_f32m1(vacc1, vzero, vl_red);
    out_dots[1] = __riscv_vfmv_f_s_f32m1_f32(vsum1);

}

} // namespace rvv_detail
#endif // __riscv_vector

// clang-format off
template <typename scalar_t, typename opmath_t>
void gemm_transa_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
/* ---1st and the best optimized version---
  const scalar_t *a_ = a;
  for (size_t i = 0; i < m; ++i) {
    const scalar_t *b_ = b;
    for (size_t j = 0; j < n; ++j) {
      opmath_t dot;
#ifdef __riscv_vector
      if constexpr (std::is_same_v<scalar_t, torch::executor::BFloat16>) {
        // scalar_t=BFloat16, opmath_t=float
        dot = static_cast<opmath_t>(rvv_detail::rvv_bf16_dot_f32(a_, b_, k));
      } else {
        dot = sum(k, [&](int64_t l) -> opmath_t {
          return static_cast<opmath_t>(a_[l]) * static_cast<opmath_t>(b_[l]);
        });
      }
#else
      dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a_[l]) * static_cast<opmath_t>(b_[l]);
      });
#endif
      b_ += ldb;
      if (beta == opmath_t(0)) {
        c[j*ldc+i] = alpha*dot;
      } else {
        c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
      }
    }
    a_ += lda;
  }
*/
/* ---2nd optimization version using pointers to eliminate the need for rvv intrinsics strided load---
#ifdef __riscv_vector
      if constexpr (std::is_same_v<scalar_t, torch::executor::BFloat16>) {
	    using BF16 = torch::executor::BFloat16;
	    constexpr int TILE_N = 4;
	 
	    const scalar_t *a_ = a;
	    for (size_t i = 0; i < m; ++i) {
	 
	      int64_t j = 0;
	 
	      // TILE_N B cols at the same time
	      for (; j + TILE_N <= (int64_t)n; j += TILE_N) {
		const BF16* b_cols[TILE_N];
		for (int t = 0; t < TILE_N; ++t) {
		  b_cols[t] = b + (j + t) * ldb;
		}
	 
		float dots[TILE_N];
		rvv_detail::rvv_bf16_dot_f32_tile(a_, b_cols, k, dots);
	 
		for (int t = 0; t < TILE_N; ++t) {
		  int64_t jj = j + t;
		  if (beta == opmath_t(0)) {
		    c[jj*ldc+i] = static_cast<scalar_t>(static_cast<float>(alpha) * dots[t]);
		  } else {
		    c[jj*ldc+i] = static_cast<scalar_t>(
		        static_cast<float>(beta) * static_cast<float>(c[jj*ldc+i]) +
		        static_cast<float>(alpha) * dots[t]);
		  }
		}
	      }
	 
	      // the rest of the cols (n % TILE_N), one by one
	      for (; j < (int64_t)n; ++j) {
		const BF16* b_col = b + j * ldb;
		float dot = rvv_detail::rvv_bf16_dot_f32(a_, b_col, k);
		if (beta == opmath_t(0)) {
		  c[j*ldc+i] = static_cast<scalar_t>(static_cast<float>(alpha) * dot);
		} else {
		  c[j*ldc+i] = static_cast<scalar_t>(
		      static_cast<float>(beta) * static_cast<float>(c[j*ldc+i]) +
		      static_cast<float>(alpha) * dot);
		}
	      }
	 
	      a_ += lda;
	    }
	    return;
        
        
        }
#endif
  const scalar_t *a_ = a;
  for (size_t i = 0; i < m; ++i) {
    const scalar_t *b_ = b;
    for (size_t j = 0; j < n; ++j) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a_[l]) * static_cast<opmath_t>(b_[l]);
      });
      b_ += ldb;
      if (beta == opmath_t(0)) {
        c[j*ldc+i] = alpha*dot;
      } else {
        c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
      }
    }
    a_ += lda;
  }
*/

// ---3rd optimization version that calculates multiple A cols while using one B col---
#ifdef __riscv_vector
    if constexpr (std::is_same_v<scalar_t, torch::executor::BFloat16>) {
        using BF16 = torch::executor::BFloat16;

        // Ultra-optimization for N = 1 ( Matrix-Vector)
        if (n == 1) {
            int64_t i = 0;
            const BF16* b_col = b; // just one B col exists
 
            for (; i + 2 <= m; i += 2) {
                const BF16* pa0 = a + (i + 0) * lda;
                const BF16* pa1 = a + (i + 1) * lda;
 
                float dots[2];
                rvv_detail::rvv_bf16_gemv_2a_1b(pa0, pa1, b_col, k, dots);
 
                for (int t = 0; t < 2; ++t) {
                    int64_t c_idx = i + t; // j is 0
                    if (beta == opmath_t(0)) {
                        c[c_idx] = static_cast<scalar_t>(static_cast<float>(alpha) * dots[t]);
                    } else {
                        c[c_idx] = static_cast<scalar_t>(
                            static_cast<float>(beta) * static_cast<float>(c[c_idx]) +
                            static_cast<float>(alpha) * dots[t]);
                    }
                }
            }
            // tail for m
            for (; i < m; ++i) {
                const BF16* a_ = a + i * lda;
                float dot = rvv_detail::rvv_bf16_dot_f32(a_, b_col, k);
                if (beta == opmath_t(0)) {
                    c[i] = static_cast<scalar_t>(static_cast<float>(alpha) * dot);
                } else {
                    c[i] = static_cast<scalar_t>(static_cast<float>(beta) * static_cast<float>(c[i]) + static_cast<float>(alpha) * dot);
                }
            }
            return;
        }
 
        // 4A for 1B Tiling (for little N, n <= 8)
        if (n <= 8) {
            int64_t i = 0;
            for (; i + 2 <= m; i += 2) {
                const BF16* pa0 = a + (i + 0) * lda;
                const BF16* pa1 = a + (i + 1) * lda;

 
                for (int64_t j = 0; j < n; ++j) {
                    const BF16* b_col = b + j * ldb;
                    float dots[2];
                    rvv_detail::rvv_bf16_gemv_2a_1b(pa0, pa1, b_col, k, dots);
 
                    for (int t = 0; t < 2; ++t) {
                        int64_t c_idx = j * ldc + (i + t);
                        if (beta == opmath_t(0)) {
                            c[c_idx] = static_cast<scalar_t>(static_cast<float>(alpha) * dots[t]);
                        } else {
                            c[c_idx] = static_cast<scalar_t>(
                                static_cast<float>(beta) * static_cast<float>(c[c_idx]) +
                                static_cast<float>(alpha) * dots[t]);
                        }
                    }
                }
            }
            // tail for m
            for (; i < m; ++i) {
                const BF16* a_ = a + i * lda;
                for (int64_t j = 0; j < n; ++j) {
                    const BF16* b_col = b + j * ldb;
                    float dot = rvv_detail::rvv_bf16_dot_f32(a_, b_col, k);
                    int64_t c_idx = j * ldc + i;
                    if (beta == opmath_t(0)) {
                        c[c_idx] = static_cast<scalar_t>(static_cast<float>(alpha) * dot);
                    } else {
                        c[c_idx] = static_cast<scalar_t>(static_cast<float>(beta) * static_cast<float>(c[c_idx]) + static_cast<float>(alpha) * dot);
                    }
                }
            }
            return;
        }
 
        // 1A for 4B Tiling (for big N, n > 8)
        else {
            const BF16* a_ = a;
            for (size_t i = 0; i < m; ++i) {
                int64_t j = 0;
                for (; j + 2 <= n; j += 2) {
                    const BF16* pb0 = b + (j + 0) * ldb;
                    const BF16* pb1 = b + (j + 1) * ldb;
 
                    float dots[2];
                    rvv_detail::rvv_bf16_dot_f32_tile_2b(a_, pb0, pb1, k, dots);
 
                    for (int t = 0; t < 2; ++t) {
                        int64_t jj = j + t;
                        int64_t c_idx = jj * ldc + i;
                        if (beta == opmath_t(0)) {
                            c[c_idx] = static_cast<scalar_t>(static_cast<float>(alpha) * dots[t]);
                        } else {
                            c[c_idx] = static_cast<scalar_t>(
                                static_cast<float>(beta) * static_cast<float>(c[c_idx]) +
                                static_cast<float>(alpha) * dots[t]);
                        }
                    }
                }
                // tail for n
                for (; j < n; ++j) {
                    const BF16* b_col = b + j * ldb;
                    float dot = rvv_detail::rvv_bf16_dot_f32(a_, b_col, k);
                    int64_t c_idx = j * ldc + i;
                    if (beta == opmath_t(0)) {
                        c[c_idx] = static_cast<scalar_t>(static_cast<float>(alpha) * dot);
                    } else {
                        c[c_idx] = static_cast<scalar_t>(
                            static_cast<float>(beta) * static_cast<float>(c[c_idx]) +
                            static_cast<float>(alpha) * dot);
                    }
                }
                a_ += lda;
            }
            return;
        }
    }
#endif
 
    // fallback
    const scalar_t *a_ = a;
    for (size_t i = 0; i < m; ++i) {
        const scalar_t *b_ = b;
        for (size_t j = 0; j < n; ++j) {
            const auto dot = sum(k, [&](int64_t l) -> opmath_t {
                return static_cast<opmath_t>(a_[l]) * static_cast<opmath_t>(b_[l]);
            });
            b_ += ldb;
            if (beta == opmath_t(0)) {
                c[j*ldc+i] = alpha*dot;
            } else {
                c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
            }
        }
        a_ += lda;
    }

}

namespace internal {
float bf16_dot_with_fp32_arith(const torch::executor::BFloat16* vec1, const torch::executor::BFloat16* vec2, int64_t len);
} // namespace internal

template <>
inline void gemm_transa_<torch::executor::BFloat16, torch::executor::BFloat16>(
    int64_t m, int64_t n, int64_t k,
    torch::executor::BFloat16 alpha,
    const torch::executor::BFloat16 *a, int64_t lda,
    const torch::executor::BFloat16 *b, int64_t ldb,
    torch::executor::BFloat16 beta,
    torch::executor::BFloat16 *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  if (alpha == 1 && beta == 0) {
    executorch::extension::parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
      const auto *a_ = a + begin * lda;
      for (int i = begin; i < end; ++i) {
        const auto *b_ = b;
        for (int j = 0; j < n; ++j) {
          const auto dot = internal::bf16_dot_with_fp32_arith(a_, b_, k);
          b_ += ldb;
          c[j*ldc+i] = dot;
        }
        a_ += lda;
      }
    });
    return;
  }
  executorch::extension::parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    const auto *a_ = a + begin * lda;
    for (int i = begin; i < end; ++i) {
      const auto *b_ = b;
      for (int j = 0; j < n; ++j) {
        const auto dot = internal::bf16_dot_with_fp32_arith(a_, b_, k);
        b_ += ldb;
        if (beta == 0) {
          c[j*ldc+i] = alpha*dot;
        } else {
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
        }
      }
      a_ += lda;
    }
  });
}

// clang-format on

template <typename scalar_t, typename opmath_t>
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_transb_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b.T)
  for (size_t l = 0; l < k; ++l) {
    for (size_t j = 0; j < n; ++j) {
      opmath_t val = b[j + l * ldb] * alpha;
      int64_t i_m = m / 4;
      for (int64_t i_i = 0; i_i < i_m; ++i_i) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      int64_t i = i_m * 4;
      for (; i < m; i++) {
        c[j * ldc + i] += a[i + l * lda] * val;
      }
    }
  }
}

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
template <typename scalar_t, typename opmath_t>
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_transb_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c += alpha * (a @ b.T)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[l * lda + i]) *
            static_cast<opmath_t>(b[l * ldb + j]);
      });
      if (beta == opmath_t(0)) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}

// clang-format off
template <typename scalar_t, typename opmath_t>
void gemm_transab_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  // c = beta * c + alpha * (a.T @ b.T)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[i * lda + l]) *
            static_cast<opmath_t>(b[l * ldb + j]);
      });

      if (beta == opmath_t(0)) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}
// clang-format on

} // namespace cpublas
} // namespace executorch
