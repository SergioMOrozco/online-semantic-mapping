#pragma once

#include <stdint.h>
#include <torch/torch.h>


void composite_rays_train_forward_semantic(const at::Tensor sigmas,const at::Tensor semantics, const uint32_t semantic_class_length,  const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor depth,at::Tensor semantic_image);
void composite_rays_train_backward_semantic(const at::Tensor grad_weights_sum,const at::Tensor grad_semantic_image, const at::Tensor sigmas,const at::Tensor semantics, const uint32_t semantic_class_length, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum,const at::Tensor semantic_image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas,at::Tensor grad_semantics);

void composite_rays_semantic(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas,at::Tensor semantics, const uint32_t semantic_class_length, at::Tensor deltas, at::Tensor weights_sum, at::Tensor depth, at::Tensor semantic_image);

