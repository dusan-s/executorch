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
