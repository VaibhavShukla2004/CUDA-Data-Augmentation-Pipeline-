#ifndef KERNELS_CUH
#define KERNELS_CUH

// Blur and Sharpen
void launch_blur_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height);
void launch_sharpen_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height);

// Rotation and Scaling
void launch_rotate_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height, float angle_degrees);
void launch_scale_kernel(const unsigned char* d_input, unsigned char* d_output, int in_width, int in_height, int out_width, int out_height);

#endif