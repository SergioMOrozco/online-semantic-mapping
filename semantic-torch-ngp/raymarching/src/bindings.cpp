#include <torch/extension.h>

#include "raymarching_utils.h"
#include "raymarching_rgb.h"
#include "raymarching_semantic.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // utils
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("near_far_from_aabb", &near_far_from_aabb, "near_far_from_aabb(CUDA)");
    m.def("sph_from_ray", &sph_from_ray, "sph_from_ray(CUDA)");
    m.def("morton3D", &morton3D, "morton3D(CUDA)");
    m.def("morton3D_invert", &morton3D_invert, "morton3D_invert(CUDA)");
    m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.def("march_rays", &march_rays, "march rays (CUDA)");
    // train
    m.def("composite_rays_train_forward_rgb", &composite_rays_train_forward_rgb, "composite_rays_train_forward_rgb (CUDA)");
    m.def("composite_rays_train_backward_rgb", &composite_rays_train_backward_rgb, "composite_rays_train_backward_rgb (CUDA)");
    m.def("composite_rays_train_forward_semantic", &composite_rays_train_forward_semantic, "composite_rays_train_forward_semantic (CUDA)");
    m.def("composite_rays_train_backward_semantic", &composite_rays_train_backward_semantic, "composite_rays_train_backward_semantic (CUDA)");
    // infer
    m.def("composite_rays_rgb", &composite_rays_rgb, "composite rays_rgb (CUDA)");
    m.def("composite_rays_semantic", &composite_rays_semantic, "composite rays_semantic (CUDA)");
}
