#include <iostream>
#include <cuda_runtime.h>

using std::cout;
using std::endl;

void printGPUInfo() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << " -> " << cudaGetErrorString(error_id) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "There are no available CUDA devices." << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)." << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  CUDA Capability Major/Minor version number: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total amount of global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Total amount of constant memory: " << deviceProp.totalConstMem << " bytes" << std::endl;
        std::cout << "  Total amount of shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
    }
}
int
main()
{
    cout << "Transformer Model C++ Basic Implementation" << endl;
    printGPUInfo();

    return 0;
}
