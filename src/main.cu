#include "cuda.h"
#include "cuda_runtime.h"
#include "nvrtc.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void deviceQuery();

const char* logFile("log.txt");

#define LOG(exp)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        std::ofstream ostr(logFile, std::ios::app);                                                \
        ostr << exp;                                                                               \
        std::cout << exp;                                                                          \
    } while(false)

#define INSPECT(exp) LOG(#exp << ": " << (exp) << "\n")
#define WHERE LOG(__FILE__ << ", " << __LINE__ << "\n")

namespace
{
std::string format(size_t i)
{
    std::ostringstream ostr;
    ostr << i;

    std::string s(ostr.str().c_str());
    std::string result;

    for(int i = 0; i < int(s.size()); ++i)
    {
        int const j(int(s.size()) - i - 1);

        if(i != 0 && i % 3 == 0)
        {
            result += ",";
        }

        result += s[j];
    }

    std::reverse(result.begin(), result.end());

    return result;
}

void writeResult(cudaError_t result, std::string const& description)
{
    std::cout << std::setw(25) << std::left << description << " : ";
    std::cout << result << " ";
    std::cout << cudaGetErrorName(result) << " ";
    std::cout << cudaGetErrorString(result) << "\n";

    if(result != 0)
    {
        throw 1;
    }
}
}

__global__ void testKernel(int input, bool flip, int* pOutput)
{
    int const i(blockIdx.x * blockDim.x + threadIdx.x);
    printf("a");

    if(i == 0)
    {
        printf("b");
        *pOutput = flip ? 1 - input : input;
    }
}

int main(int argc, char* argv[])
{
    std::cout << "cuda-test0\n\n";

    std::chrono::high_resolution_clock::time_point startTime(
        std::chrono::high_resolution_clock::now());

    try
    {
        std::ofstream ostr(logFile, std::ios::trunc);
        ostr.close();

        size_t nBytes(1000000000);

        if(argc > 1)
        {
            nBytes = atol(argv[1]);
        }

        std::cout << format(nBytes) << " bytes\n\n";

        LOG(format(nBytes) << " bytes\n\n");

        unsigned char* src;
        unsigned char* dest;

        cudaError_t rc(cudaSuccess);

        // Allocate Unified Memory � accessible from CPU or GPU
        rc = cudaMallocManaged(&src, nBytes);
        writeResult(rc, "cudaMallocManaged");

        rc = cudaMallocManaged(&dest, nBytes);
        writeResult(rc, "cudaMallocManaged");
        /*
        // initialize x and y arrays on the host
        for(size_t i(0); i < nBytes; ++i)
        {
            src[i] = i % 256;
            dest[i] = 0;
        }
*/
        int const numBlocks(1);
        int const numThreadsPerThreadBlock(10);

        int* pInput;
        cudaMallocManaged(&pInput, sizeof(int));

        int* pOutput;
        cudaMallocManaged(&pOutput, sizeof(int));

        bool* pFlip;
        cudaMallocManaged(&pFlip, sizeof(bool));

        *pInput = 17;
        *pOutput = -1000000;

        *pFlip = false;
        testKernel<<<numBlocks, numThreadsPerThreadBlock>>>(*pInput, *pFlip, pOutput);
        LOG("Start synchronize\n");
        cudaDeviceSynchronize();
        LOG("Finish synchronize\n");

        INSPECT(*pInput);
        INSPECT(*pOutput);

        *pFlip = true;
        testKernel<<<numBlocks, numThreadsPerThreadBlock>>>(*pInput, *pFlip, pOutput);
        LOG("Start synchronize\n");
        cudaDeviceSynchronize();
        LOG("Finish synchronize\n");

        INSPECT(*pInput);
        INSPECT(*pOutput);
    }
    catch(...)
    {
        std::cout << "Caught exception\n";
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::high_resolution_clock::now() - startTime)
                     .count();

    return 0;
}
