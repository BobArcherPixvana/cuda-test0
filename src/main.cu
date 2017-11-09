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

class Float2
{
public:
    float x;
    float y;
};

__global__ void testKernel(Float2 input, bool flip, Float2* pOutput)
{
    int const i(blockIdx.x * blockDim.x + threadIdx.x);
    // printf("a");

    pOutput->x = input.x;

    if(i == 0)
    {
        // printf("b");
        pOutput->y = flip ? 1.0 - input.y : input.y;
    }
}

int main(int argc, char* argv[])
{
    std::cout << "cuda-test0\n\n";

    std::chrono::high_resolution_clock::time_point startTime(
        std::chrono::high_resolution_clock::now());

    try
    {
        cudaError_t rc(cudaSuccess);

        int const numBlocks(1);
        int const numThreadsPerThreadBlock(10);

        Float2* pInput;
        cudaMallocManaged(&pInput, sizeof(Float2));

        Float2* pOutput;
        cudaMallocManaged(&pOutput, sizeof(Float2));

        bool* pFlip;
        cudaMallocManaged(&pFlip, sizeof(bool));

        pInput->x = 0.1;
        pInput->y = 0.2;

        pOutput->x = -10.0;
        pOutput->x = -20.0;
        {
            *pFlip = false;
            LOG("Flip = " << *pFlip << "\n");
            testKernel<<<numBlocks, numThreadsPerThreadBlock>>>(*pInput, *pFlip, pOutput);
            cudaDeviceSynchronize();

            INSPECT(pInput->x);
            INSPECT(pInput->y);
            INSPECT(pOutput->x);
            INSPECT(pOutput->y);
        }

        {
            *pFlip = true;
            LOG("Flip = " << *pFlip << "\n");
            testKernel<<<numBlocks, numThreadsPerThreadBlock>>>(*pInput, *pFlip, pOutput);
            cudaDeviceSynchronize();

            INSPECT(pInput->x);
            INSPECT(pInput->y);
            INSPECT(pOutput->x);
            INSPECT(pOutput->y);
        }
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
