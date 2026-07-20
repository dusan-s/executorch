/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
 
#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
 
#include <type_traits>
 
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

#define PERMUTE_RVV_TILE_ROWS 32
 
namespace torch {
namespace executor {
namespace native {
 
using SizesType = executorch::aten::SizesType;
using Tensor = executorch::aten::Tensor;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;
 
namespace {

void increment_coordinate_permuted(
    const Tensor& tensor,
    size_t* const coordinate,
    IntArrayRef dims) {
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; i--) {
    size_t d = dims[i] >= 0 ? dims[i] : dims[i] + tensor.dim();
    coordinate[d]++;
    if (static_cast<ssize_t>(coordinate[d]) == tensor.size(d)) {
      coordinate[d] = 0;
    } else {
      return;
    }
  }
}
 
// skips last/fastest out dimension
// the vectorized inner loop keeps that dimension at 0 and processes it in the entire row at once
void increment_coordinate_permuted_outer(
    const Tensor& tensor,
    size_t* const coordinate,
    IntArrayRef dims) {
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; i--) {
    size_t d = dims[i] >= 0 ? dims[i] : dims[i] + tensor.dim();
    coordinate[d]++;
    if (static_cast<ssize_t>(coordinate[d]) == tensor.size(d)) {
      coordinate[d] = 0;
    } else {
      return;
    }
  }
}

size_t increment_coordinate_permuted_outer_incremental(
    const Tensor& tensor,
    size_t* const coordinate,
    IntArrayRef dims,
    const size_t* const trailing_dims_memo,
    size_t base_index) {
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; i--) {
    size_t d = dims[i] >= 0 ? dims[i] : dims[i] + tensor.dim();
    coordinate[d]++;
    base_index += trailing_dims_memo[d];
    if (static_cast<ssize_t>(coordinate[d]) == tensor.size(d)) {
      base_index -=
          static_cast<size_t>(tensor.size(d)) * trailing_dims_memo[d];
      coordinate[d] = 0;
    } else {
      return base_index;
    }
  }
  return base_index;
}
 
#ifdef __riscv_vector
 
template <size_t N>
struct UIntOfSize;
template <>
struct UIntOfSize<1> {
  using type = uint8_t;
};
template <>
struct UIntOfSize<2> {
  using type = uint16_t;
};
template <>
struct UIntOfSize<4> {
  using type = uint32_t;
};
template <>
struct UIntOfSize<8> {
  using type = uint64_t;
};
 
template <size_t N>
constexpr bool kHasRvvPath =
    (N == 1) || (N == 2) || (N == 4) || (N == 8);
 
// strided gather (stride_elems elements) then continual store
template <typename UIntT>
void gather_row_rvv(
    const UIntT* const in_base,
    UIntT* const out_row,
    size_t stride_elems,
    size_t length) {
    
  const ptrdiff_t stride_bytes = static_cast<ptrdiff_t>(stride_elems) * static_cast<ptrdiff_t>(sizeof(UIntT));
  const UIntT* src = in_base;
  UIntT* dst = out_row;
  size_t n = length;
 
  // stride == 1 - row  is continuous
  // (the permutation leaves the last dimension intact at that location)
  if (stride_elems == 1) {
    while (n > 0) {
      size_t vl;
      if constexpr (sizeof(UIntT) == 1) {
        vl = __riscv_vsetvl_e8m8(n);
        auto v = __riscv_vle8_v_u8m8(reinterpret_cast<const uint8_t*>(src), vl);
        __riscv_vse8_v_u8m8(reinterpret_cast<uint8_t*>(dst), v, vl);
        
      } else if constexpr (sizeof(UIntT) == 2) {
        vl = __riscv_vsetvl_e16m8(n);
        auto v = __riscv_vle16_v_u16m8(reinterpret_cast<const uint16_t*>(src), vl);
        __riscv_vse16_v_u16m8(reinterpret_cast<uint16_t*>(dst), v, vl);
        
      } else if constexpr (sizeof(UIntT) == 4) {
        vl = __riscv_vsetvl_e32m8(n);
        auto v = __riscv_vle32_v_u32m8(reinterpret_cast<const uint32_t*>(src), vl);
        __riscv_vse32_v_u32m8(reinterpret_cast<uint32_t*>(dst), v, vl);
        
      } else {
        static_assert(sizeof(UIntT) == 8, "unsupported element width");
        vl = __riscv_vsetvl_e64m8(n);
        auto v = __riscv_vle64_v_u64m8(reinterpret_cast<const uint64_t*>(src), vl);
        __riscv_vse64_v_u64m8(reinterpret_cast<uint64_t*>(dst), v, vl);
        
      }
      src += vl;
      dst += vl;
      n -= vl;
    }
    return;
  }
 
  while (n > 0) {
    size_t vl;
    if constexpr (sizeof(UIntT) == 1) {
      vl = __riscv_vsetvl_e8m8(n);
      auto v = __riscv_vlse8_v_u8m8(reinterpret_cast<const uint8_t*>(src), stride_bytes, vl);
      __riscv_vse8_v_u8m8(reinterpret_cast<uint8_t*>(dst), v, vl);
      
    } else if constexpr (sizeof(UIntT) == 2) {
      vl = __riscv_vsetvl_e16m8(n);
      auto v = __riscv_vlse16_v_u16m8(reinterpret_cast<const uint16_t*>(src), stride_bytes, vl);
      __riscv_vse16_v_u16m8(reinterpret_cast<uint16_t*>(dst), v, vl);
      
    } else if constexpr (sizeof(UIntT) == 4) {
      vl = __riscv_vsetvl_e32m8(n);
      auto v = __riscv_vlse32_v_u32m8(reinterpret_cast<const uint32_t*>(src), stride_bytes, vl);
      __riscv_vse32_v_u32m8(reinterpret_cast<uint32_t*>(dst), v, vl);
      
    } else {
      static_assert(sizeof(UIntT) == 8, "unsupported element width");
      vl = __riscv_vsetvl_e64m8(n);
      auto v = __riscv_vlse64_v_u64m8(reinterpret_cast<const uint64_t*>(src), stride_bytes, vl);
      __riscv_vse64_v_u64m8(reinterpret_cast<uint64_t*>(dst), v, vl);
      
    }
    src += vl * stride_elems;
    dst += vl;
    n -= vl;
  }
}

bool is_simple_last_two_dims_swap(
    const Tensor& in,
    IntArrayRef dims,
    size_t* const R,
    size_t* const C) {
    
  const int64_t n = in.dim();
  if (n < 2) {
    return false;
  }
  
  // if only the last two dimensions are swapped, all other axes must remain the same
  for (int64_t i = 0; i < n - 2; i++) {
    int64_t d = dims[i] >= 0 ? dims[i] : dims[i] + n;
    
    if (d != i) {
      return false;
    }
    
  }
  
  int64_t d_second_last = dims[n - 2] >= 0 ? dims[n - 2] : dims[n - 2] + n;
  int64_t d_last = dims[n - 1] >= 0 ? dims[n - 1] : dims[n - 1] + n;
  
  // the second last must be equal to last, and last must be equal to second last
  if (d_second_last != n - 1 || d_last != n - 2) {
    return false;    
  }
  
  *R = static_cast<size_t>(in.size(n - 2));
  *C = static_cast<size_t>(in.size(n - 1));
  
  return true;
}

// in_base: R x C, row-major (element (r,c) on r*C + c)
// out_base: C x R, row-major (element (c,r) on c*R + r)
template <typename UIntT>
void transpose_block_rvv(
    const UIntT* const in_base,
    UIntT* const out_base,
    size_t R,
    size_t C) {
    
  constexpr size_t kTileRows = PERMUTE_RVV_TILE_ROWS; // the height of the block that is being processed

  for (size_t bi = 0; bi < R; bi += kTileRows) {
    const size_t i_end = std::min(bi + kTileRows, R); // end of current block
    const size_t block_len = i_end - bi; // current block height (usually PERMUTE_RVV_TILE_ROWS height)

    for (size_t j = 0; j < C; j++) {
      gather_row_rvv<UIntT>(
          in_base + bi * C + j, // first element of the block in column j
          out_base + j * R + bi, // appropriate place in the output queue j
          C, // strides between consecutive rows within a block
          block_len);
    }
  }
}
 
template <typename CTYPE>
void permute_copy_row_loop(
    const Tensor& in,
    const Tensor& out,
    IntArrayRef dims,
    size_t* const in_coord,
    const size_t* const trailing_dims_memo,
    const CTYPE* const in_data,
    CTYPE* const out_data) {
  const size_t out_dim = out.dim();
 
  if constexpr (kHasRvvPath<sizeof(CTYPE)>) {
    using UIntT = typename UIntOfSize<sizeof(CTYPE)>::type;
 
    if (out_dim == 0) {
      // 0-d tensor just 1 element
      out_data[0] = in_data[0];
      return;
    }

    // swap of the last two axes
    size_t R = 0, C = 0;
    if (is_simple_last_two_dims_swap(in, dims, &R, &C)) {
      const size_t batch_elems = R * C; // number of elements in one matrix
      const size_t num_batches = (R * C == 0) ? 0 : in.numel() / batch_elems; // number of matrices
      
      for (size_t b = 0; b < num_batches; b++) {
        transpose_block_rvv<UIntT>(reinterpret_cast<const UIntT*>(in_data) + b * batch_elems, reinterpret_cast<UIntT*>(out_data) + b * batch_elems, R, C);
            
      }
      return;
      
    }

    const int64_t last_dim_signed = dims[out_dim - 1];
    const size_t d_last = last_dim_signed >= 0 ? static_cast<size_t>(last_dim_signed) : static_cast<size_t>(last_dim_signed + in.dim());
 
    const size_t row_len = out.size(out_dim - 1);
    const size_t stride_elems = trailing_dims_memo[d_last];
    const size_t num_rows = out.numel() / row_len;
    
    
    size_t out_offset = 0;
    for (size_t r = 0; r < num_rows; r++) {
      const size_t base_index = executorch::runtime::coordinateToIndexWithTrailingDimsMemo(in, in_coord, trailing_dims_memo);
 
      gather_row_rvv<UIntT>(reinterpret_cast<const UIntT*>(in_data + base_index), reinterpret_cast<UIntT*>(out_data + out_offset), stride_elems, row_len);
 
      out_offset += row_len;
      increment_coordinate_permuted_outer(in, in_coord, dims);
      
    }
    
    /*
    size_t base_index = 0;
    size_t out_offset = 0;
    for (size_t r = 0; r < num_rows; r++) {
      gather_row_rvv<UIntT>(reinterpret_cast<const UIntT*>(in_data + base_index), reinterpret_cast<UIntT*>(out_data + out_offset), stride_elems, row_len);

      out_offset += row_len;
      base_index = increment_coordinate_permuted_outer_incremental(in, in_coord, dims, trailing_dims_memo, base_index);
      
    }
    */
  } else {
    // types without a direct RVV path have a scalar fallback
    for (const auto i : c10::irange(out.numel())) {
      out_data[i] =
          in_data[executorch::runtime::coordinateToIndexWithTrailingDimsMemo(in, in_coord, trailing_dims_memo)];
      increment_coordinate_permuted(in, in_coord, dims);
      
    }
  }
}
 
#endif // __riscv_vector
 
} // namespace
 
Tensor& opt_permute_copyRVV_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dims,
    Tensor& out) {
  (void)ctx;
 
  ET_KERNEL_CHECK(
      ctx, check_permute_copy_args(in, dims, out), InvalidArgument, out);
 
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);
 
  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_permute_copy_out_target_size(
      in, dims, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);
 
  const auto in_type = out.scalar_type();
 
  size_t in_coord[kTensorDimensionLimit] = {0};
  size_t trailing_dims_memo[kTensorDimensionLimit];
  executorch::runtime::memoizeTrailingDims(in, trailing_dims_memo); // calculates how much the linear index should jump when the coordinate of a dimension increases by 1
 
  // in and out must be the same dtype
  ET_SWITCH_ALL_TYPES(in_type, ctx, "permute_copy.out", CTYPE, [&] {
    const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();
 
#ifdef __riscv_vector
    permute_copy_row_loop<CTYPE>(
        in, out, dims, in_coord, trailing_dims_memo, in_data, out_data);
#else
    // fallback
    // coordinateToIndexWithTrailingDimsMemo converts a multidimensional coordinate of a tensor into a linear memory index using stride values ​​from trailing_dims_memo
    for (const auto i : c10::irange(out.numel())) {
      out_data[i] =
          in_data[executorch::runtime::coordinateToIndexWithTrailingDimsMemo(in, in_coord, trailing_dims_memo)];
      increment_coordinate_permuted(in, in_coord, dims);
    }
#endif
  });
 
  return out;
}
 
} // namespace native
} // namespace executor
} // namespace torch
