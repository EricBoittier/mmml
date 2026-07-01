#include<iostream>
#include<OpenMM.h>
#include <openmm/internal/ThreadPool.h>

// vectorize headers will not compile on Apple M1 platform
#if defined(__x86_64__) /* 64 bit detected */
#include <openmm/internal/vectorize.h>
#endif

int main() {
  OpenMM::Platform::getPlatformByName("CUDA");
  std::cout << "Hi there!" << std::endl;
}


