#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>

__device__ unsigned char clamp_pixel(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (unsigned char)value;
}


__global__ void blur_kernel(const unsigned char* input, unsigned char* output, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (x < width && y < height) {
        int sum = 0;
        int count = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                int ny = y + ky;

                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }
       
        output[y * width + x] = (unsigned char)(sum / count);
    }
}


__global__ void sharpen_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        
        int filter[3][3] = {
            { 0, -1,  0},
            {-1,  5, -1},
            { 0, -1,  0}
        };

        int sum = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                int ny = y + ky;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    
                    sum += input[ny * width + nx] * filter[ky+1][kx+1];
                }
            }
        }
        
        output[y * width + x] = clamp_pixel(sum);
    }
}


void launch_blur_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    blur_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
}

void launch_sharpen_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    dim3 blockSize(16, 16); 
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    sharpen_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
}

__global__ void rotate_kernel(const unsigned char* input, unsigned char* output, int width, int height, float angle_rad) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Find the center of the image
        float cx = width / 2.0f;
        float cy = height / 2.0f;

        float cos_theta = cosf(angle_rad);
        float sin_theta = sinf(angle_rad);

        // Calculate the corresponding source pixel coordinate (Inverse Rotation)
        int src_x = (int)((x - cx) * cos_theta + (y - cy) * sin_theta + cx);
        int src_y = (int)(-(x - cx) * sin_theta + (y - cy) * cos_theta + cy);

        // If the source pixel is within the original image bounds, copy it. Otherwise, set to black (0).
        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            output[y * width + x] = input[src_y * width + src_x];
        } else {
            output[y * width + x] = 0; 
        }
    }
}

// --- 4. Scale/Resize Kernel (Nearest Neighbor) ---
__global__ void scale_kernel(const unsigned char* input, unsigned char* output, int in_width, int in_height, int out_width, int out_height) {
    // x and y correspond to the NEW output dimensions
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    if (x < out_width && y < out_height) {
        // Calculate the ratio between original and new image
        float x_ratio = (float)in_width / out_width;
        float y_ratio = (float)in_height / out_height;

        // Find the nearest pixel in the original image
        int src_x = (int)(x * x_ratio);
        int src_y = (int)(y * y_ratio);

        output[y * out_width + x] = input[src_y * in_width + src_x];
    }
}


// --- New Host Launchers ---

void launch_rotate_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height, float angle_degrees) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Convert degrees to radians for the math functions inside the kernel
    float angle_rad = angle_degrees * (3.14159265f / 180.0f);

    rotate_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, angle_rad);
}

void launch_scale_kernel(const unsigned char* d_input, unsigned char* d_output, int in_width, int in_height, int out_width, int out_height) {
    dim3 blockSize(16, 16);
    // Grid size is based on the OUTPUT dimensions, not the input dimensions
    dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x, (out_height + blockSize.y - 1) / blockSize.y);
    
    scale_kernel<<<gridSize, blockSize>>>(d_input, d_output, in_width, in_height, out_width, out_height);
}