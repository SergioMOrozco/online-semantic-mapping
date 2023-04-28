#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }

template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

inline __host__ __device__ void swapf(float& a, float& b) {
    float c = a; a = b; b = c;
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent;
    frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
    const float mx = dt * H * 0.5;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = __expand_bits(x);
	uint32_t yy = __expand_bits(y);
	uint32_t zz = __expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

// sigmas: [M]
// semantics: [M, CLASS_NUM]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N], final pixel alpha
// depth: [N,]
// image: [N, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ semantics,  
    const uint32_t semantic_class_length,
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const uint32_t M, const uint32_t N, const float T_thresh, 
    scalar_t * weights_sum,
    scalar_t * depth,
    scalar_t * semantic_image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps > M) {
        weights_sum[index] = 0;
        depth[index] = 0;

        for (uint8_t i = 0; i < semantic_class_length; i++) 
        {
          semantic_image[index * 3 + i] = 0;
        }
        return;
    }

    sigmas += offset;
    semantics += offset * semantic_class_length;
    deltas += offset * 2;

    // accumulate 
    uint32_t step = 0;

    scalar_t T = 1.0f;
    scalar_t ws = 0, t = 0, d = 0;

    // lets just use 1000 and hope nobody uses more than 1000 classes
    scalar_t semantic_ids[1000] = {0};

    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;


        //TODO: make semantic class length a parameter
        for (uint8_t i = 0; i < 3; i++) 
        {
          semantic_ids[i] += weight * semantics[i];
        }

        t += deltas[1]; // real delta
        d += weight * t;
        
        ws += weight;
        
        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas++;
        semantics += semantic_class_length;
        deltas += 2;

        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d;

    for (uint8_t i = 0; i < 3; i++) 
    {
      semantic_image[index * 3 + i] = semantic_ids[i];
    }
}


void composite_rays_train_forward_semantic(const at::Tensor sigmas, const at::Tensor semantics, const uint32_t semantic_class_length, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor depth, at::Tensor semantic_image ) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_forward", ([&] {
        kernel_composite_rays_train_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas.data_ptr<scalar_t>(), semantics.data_ptr<scalar_t>(), semantic_class_length, deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), M, N, T_thresh, weights_sum.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), semantic_image.data_ptr<scalar_t>());
    }));
}


// grad_weights_sum: [N,]
// grad: [N, 3]
// sigmas: [M]
// semantics: [M, NUM_CLASS]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here 
// semantic_image: [N, 100]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad_semantic_image,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ semantics, 
    const uint32_t semantic_class_length,
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ semantic_image,
    const uint32_t M, const uint32_t N, const float T_thresh,
    scalar_t * grad_sigmas,
    scalar_t * grad_semantics
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps > M) return;

    grad_weights_sum += index;
    grad_semantic_image += index * semantic_class_length;
    weights_sum += index;
    semantic_image += index * semantic_class_length;
    sigmas += offset;
    semantics += offset * semantic_class_length;
    deltas += offset * 2;
    grad_sigmas += offset;
    grad_semantics += offset * semantic_class_length;

    // accumulate 
    uint32_t step = 0;
    
    scalar_t T = 1.0f;
    const scalar_t ws_final = weights_sum[0];

    scalar_t semantic_ids_final[1000] ={0};

    for (uint8_t i = 0; i < 3; i++) 
    {
      semantic_ids_final[i] = semantic_image[i];
    }

    scalar_t ws = 0;

    scalar_t semantic_ids[1000] ={0};

    while (step < num_steps) {
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        ws += weight;

        for (uint8_t i = 0; i < semantic_class_length; i++) 
        {
          semantic_ids[i] += weight * semantics[i];
        }

        T *= 1.0f - alpha;
        
        // check https://note.kiui.moe/others/nerf_gradient/ for the gradient calculation.

        for (uint8_t i = 0; i < semantic_class_length; i++) 
        {
          grad_semantics[i] = grad_semantic_image[i] * weight;
        }


        scalar_t sum = 0; 

        for (uint8_t i = 0; i < semantic_class_length; i++) 
        {
          sum += grad_semantic_image[i] * (T * semantics[i] - (semantic_ids_final[i] - semantic_ids[i]));
        }

        sum += grad_weights_sum[0] * (1 - ws_final);

        grad_sigmas[0] = deltas[0] * sum;

        // write grad_sigmas
        //grad_sigmas[0] = deltas[0] * (
        //    grad_image[0] * (T * rgbs[0] - (r_final - r)) + 
        //    grad_image[1] * (T * rgbs[1] - (g_final - g)) + 
        //    grad_image[2] * (T * rgbs[2] - (b_final - b)) +
        //    grad_weights_sum[0] * (1 - ws_final)
        //);

        //printf("[n=%d] num_steps=%d, T=%f, grad_sigmas=%f, r_final=%f, r=%f\n", n, step, T, grad_sigmas[0], r_final, r);
        // minimal remained transmittence
        if (T < T_thresh) break;
        
        // locate
        sigmas++;
        semantics += semantic_class_length;
        deltas += 2;
        grad_sigmas++;
        grad_semantics += semantic_class_length;

        step++;
    }
}


void composite_rays_train_backward_semantic(const at::Tensor grad_weights_sum,const at::Tensor grad_semantic_image, const at::Tensor sigmas,const at::Tensor semantics, const uint32_t semantic_class_length,  const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum,const at::Tensor semantic_image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_semantics) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_semantic_image.scalar_type(), "composite_rays_train_backward", ([&] {
        kernel_composite_rays_train_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad_weights_sum.data_ptr<scalar_t>(),grad_semantic_image.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(),semantics.data_ptr<scalar_t>(), semantic_class_length, deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), weights_sum.data_ptr<scalar_t>(),semantic_image.data_ptr<scalar_t>(), M, N, T_thresh, grad_sigmas.data_ptr<scalar_t>(),grad_semantics.data_ptr<scalar_t>());
    }));
}


////////////////////////////////////////////////////
/////////////          infernce        /////////////
////////////////////////////////////////////////////



template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const float T_thresh,
    int* rays_alive, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ sigmas, 
    const scalar_t* __restrict__ semantics, 
    const uint32_t semantic_class_length,
    const scalar_t* __restrict__ deltas, 
    scalar_t* weights_sum, scalar_t* depth, scalar_t* semantic_image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    
    // locate 
    sigmas += n * n_step;
    semantics += n * n_step * semantic_class_length;
    deltas += n * n_step * 2;
    
    rays_t += index;
    weights_sum += index;
    depth += index;
    semantic_image += index * semantic_class_length;

    scalar_t t = rays_t[0]; // current ray's t
    
    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];

    scalar_t semantic_ids[1000] = {0};

    for (uint8_t i = 0; i < semantic_class_length; i++) 
    {
      semantic_ids[i] = semantic_image[i];
    }



    // accumulate 
    uint32_t step = 0;
    while (step < n_step) {
        
        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        /* 
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        --> 
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;

        for (uint8_t i = 0; i < 3; i++) 
        {
          semantic_ids[i] += weight * semantics[i];
        }

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        semantics += semantic_class_length;
        deltas += 2;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t[0] = t;
    }

    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;

    for (uint8_t i = 0; i < semantic_class_length; i++) 
    {
      semantic_image[i] = semantic_ids[i];
    }
}


void composite_rays_semantic(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor semantics, const uint32_t semantic_class_length, at::Tensor deltas, at::Tensor weights, at::Tensor depth,at::Tensor semantic_image) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    semantic_image.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, T_thresh, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(),semantics.data_ptr<scalar_t>(), semantic_class_length, deltas.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(),semantic_image.data_ptr<scalar_t>());
    }));
}
