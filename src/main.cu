#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "kernels.cuh"

// STB Image definitions (These tell the compiler to build the STB source code here)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Macro for checking CUDA errors
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

int main(int argc, char** argv) {
    // 1. Define file paths (Update these to match your actual testing files)
    const char* input_file = "data/input/sample.jpg";
    const char* output_blur_file = "data/output/sample_blurred.jpg";
    const char* output_sharpen_file = "data/output/sample_sharpened.jpg";

    int width, height, channels;

    // 2. Load the image using STB (Force loading as 1 channel / Grayscale)
    std::cout << "Loading image: " << input_file << std::endl;
    unsigned char* h_input = stbi_load(input_file, &width, &height, &channels, 1);
    
    if (h_input == NULL) {
        std::cerr << "Error: Could not load image. Check the file path!" << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully. Dimensions: " << width << "x" << height << std::endl;

    // Calculate the total size of the image in bytes
    size_t img_size = width * height * sizeof(unsigned char);

    // 3. Allocate Host (CPU) memory for the outputs
    unsigned char* h_output_blur = (unsigned char*)malloc(img_size);
    unsigned char* h_output_sharpen = (unsigned char*)malloc(img_size);

    // 4. Allocate Device (GPU) memory
    unsigned char *d_input, *d_output;
    cudaCheckError(cudaMalloc((void**)&d_input, img_size));
    cudaCheckError(cudaMalloc((void**)&d_output, img_size));

    // 5. Copy the input image from Host to Device
    std::cout << "Transferring data to GPU..." << std::endl;
    cudaCheckError(cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice));

    // ==========================================
    // Run Blur Kernel
    // ==========================================
    std::cout << "Running Blur Kernel..." << std::endl;
    launch_blur_kernel(d_input, d_output, width, height);
    cudaCheckError(cudaPeekAtLastError()); // Check if kernel launch failed
    cudaCheckError(cudaDeviceSynchronize()); // Wait for GPU to finish

    // Copy result back to Host
    cudaCheckError(cudaMemcpy(h_output_blur, d_output, img_size, cudaMemcpyDeviceToHost));
    
    // Save Blurred Image
    stbi_write_jpg(output_blur_file, width, height, 1, h_output_blur, 100);
    std::cout << "Saved blurred image to: " << output_blur_file << std::endl;

    // ==========================================
    // Run Sharpen Kernel
    // ==========================================
    std::cout << "Running Sharpen Kernel..." << std::endl;
    launch_sharpen_kernel(d_input, d_output, width, height);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to Host
    cudaCheckError(cudaMemcpy(h_output_sharpen, d_output, img_size, cudaMemcpyDeviceToHost));

    // Save Sharpened Image
    stbi_write_jpg(output_sharpen_file, width, height, 1, h_output_sharpen, 100);
    std::cout << "Saved sharpened image to: " << output_sharpen_file << std::endl;

    // 6. Clean up memory to prevent leaks
    std::cout << "Cleaning up memory..." << std::endl;
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output_blur);
    free(h_output_sharpen);
    stbi_image_free(h_input);

    std::cout << "Pipeline execution completed successfully!" << std::endl;
    
    return 0;
}