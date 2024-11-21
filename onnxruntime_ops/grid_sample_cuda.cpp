// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/GridSampler.cpp
#include "pch.h"
#include "grid_sample_cuda.h"
#include "trt_grid_sampler_kernel.hpp"

#include <cmath>
#include <iostream>

#include "ort_utils.h"

namespace mmdeploy {
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit - 1), MAX(in, 0))

GridSampleKernel_cuda::GridSampleKernel_cuda(const OrtApi &api, const OrtKernelInfo *info)
    : ort_(api), info_(info) {
  align_corners_ = ort_.KernelInfoGetAttribute<int64_t>(info, "align_corners");
  interpolation_mode_ = ort_.KernelInfoGetAttribute<int64_t>(info, "interpolation_mode");
  padding_mode_ = ort_.KernelInfoGetAttribute<int64_t>(info, "padding_mode");

  allocator_ = Ort::AllocatorWithDefaultOptions();
}


void GridSampleKernel_cuda::Compute(OrtKernelContext *context) {
  const bool align_corners = align_corners_;
  
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ort_.KernelContext_GetGPUComputeStream(context));

  GridSamplerInterpolation interp_mode = GridSamplerInterpolation::Bilinear;
  switch (interpolation_mode_) {
  case 0:
      interp_mode = GridSamplerInterpolation::Bilinear;
      break;
  case 1:
      interp_mode = GridSamplerInterpolation::Nearest;
      break;
  default:
      break;
  }
  GridSamplerPadding padding_mode = GridSamplerPadding::Zeros;
  switch (padding_mode_) {
  case 0:
      padding_mode = GridSamplerPadding::Zeros;
      break;


  case 1:
      padding_mode = GridSamplerPadding::Border;
      break;


  case 2:
      padding_mode = GridSamplerPadding::Reflection;
      break;
  default:
      break;
  }



  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  const float *input_data = reinterpret_cast<const float *>(ort_.GetTensorData<float>(input));

  const OrtValue *grid = ort_.KernelContext_GetInput(context, 1);
  const float *grid_data = reinterpret_cast<const float *>(ort_.GetTensorData<float>(grid));
  

  OrtTensorDimensions input_dims(ort_, input);
  OrtTensorDimensions grid_dims(ort_, grid);
  int64_t N = input_dims[0];
  int64_t C = input_dims[1];
  int64_t out_H = grid_dims[1];
  int64_t out_W = grid_dims[2];

  int nb_dims = input_dims.size();

  std::vector<int64_t> output_dims = {N, C, out_H, out_W};
  OrtValue *output =
      ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
  float *out_ptr = ort_.GetTensorMutableData<float>(output);

  int64_t* input_dims_ = input_dims.data();
  int64_t* grid_dims_ = grid_dims.data();

  grid_sample<float>(out_ptr, input_data, grid_data, output_dims.data(), input_dims_, grid_dims_,
      nb_dims, interp_mode, padding_mode, align_corners, stream);


}

REGISTER_ONNXRUNTIME_OPS(mmdeploy, GridSampleOp);
}  // namespace mmdeploy
