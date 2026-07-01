#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <cstring>

#include "ocl_util.h"

#ifdef HAS_OPENCL

std::string ocl_get_error(cl_int err) {
  switch(err){
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";

  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  default: return "Unknown OpenCL error";
  }
}

void ocl_handle_error(cl_int err, std::string msg) {
  if (err != CL_SUCCESS) {
    std::cout << ocl_get_error(err)
              << msg
              << std::endl;
    // TODO: throw error and exit or something
  }
}

OclDevice::OclDevice(size_t newId, cl_device_id newDevId)
  : id(newId), devId(newDevId) {
  char devName[80];
  size_t devNameLength = 0;
  cl_int stat = clGetDeviceInfo(this->devId, CL_DEVICE_NAME,
                                80 * sizeof(char), devName, &devNameLength);
  ocl_check_status(stat);

  std::string newName(devName);
  this->name = newName;

  cl_ulong globalMemSize = 0;
  stat = clGetDeviceInfo(this->devId, CL_DEVICE_GLOBAL_MEM_SIZE,
                         sizeof(cl_ulong), &globalMemSize, NULL);
  ocl_check_status(stat);

  this->mem = globalMemSize;

}

size_t OclDevice::getId() {
  return this->id;
}

cl_device_id OclDevice::getDevId() {
  return this->devId;
}

size_t OclDevice::getMemBytes() {
  return this->mem;
}

std::string OclDevice::getName() {
  return std::string(this->name);
}

std::string OclDevice::toString() {
  double memGB = this->mem / 1024.0 / 1024.0 / 1024.0;

  std::ostringstream idStream;
  idStream << this->id;

  std::ostringstream memStream;
  memStream << memGB;

  return " " + idStream.str() + ". "
    + this->name
    + " with "
    + memStream.str()
    + " GB of memory.";

}

void ocl_print_build_log(cl_program prog, cl_device_id dev_id) {
  size_t len = 0;
  cl_int ret = CL_SUCCESS;
  ret = clGetProgramBuildInfo(prog, dev_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  char * buffer = new char[len];
  ret = clGetProgramBuildInfo(prog, dev_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
  std::cout << buffer << std::endl;
  delete [] buffer;
}

cl_int ocl_compile_kernel(std::string kernelSrc,
                          std::string kernelName,
                          cl_context context, cl_device_id devId,
                          cl_kernel &outKernel) {

  const char * kSrc = kernelSrc.c_str();
  size_t kLen = kernelSrc.length();

  cl_int status = CL_SUCCESS;
  cl_program prog = clCreateProgramWithSource(context, 1,
                  &kSrc, &kLen, &status);
  ocl_check_status(status);

  status = clBuildProgram(prog, 1, &devId, NULL, NULL, NULL);
  ocl_check_status(status);
  if (status != CL_SUCCESS) {
    ocl_print_build_log(prog, devId);
    return status;
  }

  status = CL_SUCCESS;
  outKernel = clCreateKernel(prog, kernelName.c_str(), &status);
  ocl_check_status(status);

  status = clReleaseProgram(prog);
  ocl_check_status(status);

  return status;
}

cl_int ocl_clean_up(cl_kernel &kern,
                    cl_command_queue &queue, cl_context &context) {
  cl_int status = clReleaseKernel(kern);
  ocl_check_status(status);

  status = clFlush(queue);
  ocl_check_status(status);

  status = clFinish(queue);
  ocl_check_status(status);

  status = clReleaseCommandQueue(queue);
  ocl_check_status(status);

  status = clReleaseContext(context);
  ocl_check_status(status);

  return status;
}
#endif /* HAS_OPENCL */

int ocl_device_init(void ** fortran_devices) {
#ifdef HAS_OPENCL
  std::vector<OclDevice *> * ocl_devices = new std::vector<OclDevice *>();

  cl_int stat = CL_SUCCESS;

  // query the number of platforms
  cl_uint numPlatforms = 0;
  stat = clGetPlatformIDs(0, NULL, &numPlatforms);
  ocl_check_status(stat);

  // now get all the platform IDs
  cl_platform_id platforms[numPlatforms];
  stat = clGetPlatformIDs(numPlatforms, platforms, NULL);
  ocl_check_status(stat);

  size_t totalDevices = 0;
  for(size_t platform_i = 0; platform_i < numPlatforms; ++platform_i) {
  // query the number of devices
    cl_uint numPlatDevs = 0;
    stat = clGetDeviceIDs(platforms[platform_i], CL_DEVICE_TYPE_ALL, 0, NULL,
                          &numPlatDevs);
    ocl_check_status(stat);

    // now get all the device IDs
    cl_device_id devices[numPlatDevs];
    stat = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numPlatDevs, devices,
                          NULL);
    ocl_check_status(stat);

    for(size_t device_i = 0; device_i < numPlatDevs; ++device_i) {
      cl_device_id devId = devices[device_i];
      ++totalDevices;
      ocl_devices->push_back(new OclDevice(totalDevices, devId));
    }  // loop over devices
  }  // loop over platforms

  *fortran_devices = ocl_devices;
  return (int) stat;
#else
  std::cout << " WARNING: OpenCL not found at compile time"
            << std::endl;
  return -1;
#endif /* HAS_OPENCL */
}

void ocl_device_print(void * fortran_devices) {
#ifdef HAS_OPENCL
  std::vector<OclDevice *> * ocl_devices =
    static_cast<std::vector<OclDevice *> *>(fortran_devices);

  std::cout << " OpenCL devices: (id #) (name) (bytes of memory)"
            << std::endl;

  for (std::vector<OclDevice *>::iterator dev = ocl_devices->begin();
       dev != ocl_devices->end(); ++dev) {
    std::cout << (*dev)->toString() << std::endl;
  }
#else
  std::cout << " WARNING: OpenCL not found at compile time"
            << std::endl;
#endif /* HAS_OPENCL */
}

void ocl_device_print_one(void * dev) {
#ifdef HAS_OPENCL
  OclDevice * ocl_dev = static_cast<OclDevice *>(dev);

  std::cout << " OpenCL device: (id #) (name) (bytes of memory)"
            << std::endl;
  std::cout << ocl_dev->toString() << std::endl;
#else
  std::cout << " WARNING: OpenCL not found at compile time"
            << std::endl;
#endif /* HAS_OPENCL */
}

void ocl_device_string(void * dev, char * c_string, int max_size) {
#ifdef HAS_OPENCL
  OclDevice * ocl_dev = static_cast<OclDevice *>(dev);
  strncpy(c_string, ocl_dev->toString().c_str(), max_size);
#else
  c_string = NULL;
  std::cout << " WARNING: OpenCL not found at compile time"
            << std::endl;
#endif /* HAS_OPENCL */
}

int ocl_device_get(void * fortran_devices, int dev_id,
                   void ** out_device) {
#ifdef HAS_OPENCL
  std::vector<OclDevice *> * ocl_devices =
    static_cast<std::vector<OclDevice *> *>(fortran_devices);

  for (std::vector<OclDevice *>::iterator dev = ocl_devices->begin();
       dev != ocl_devices->end(); ++dev) {
    if ((*dev)->getId() == dev_id) {
      *out_device = *dev;
      break;
    }
  }
#else
  std::cout << " WARNING: OpenCL not found at compile time"
            << std::endl;
  return -1;
#endif /* HAS_OPENCL */
  return 1;
}

int ocl_device_max_mem_get(void * fortran_devices, void ** out_device) {
#ifdef HAS_OPENCL
  std::vector<OclDevice *> * ocl_devices =
    static_cast<std::vector<OclDevice *> *>(fortran_devices);

  size_t maxMem = 0;
  OclDevice * maxMemDev = NULL;
  for (std::vector<OclDevice *>::iterator dev = ocl_devices->begin();
       dev != ocl_devices->end(); ++dev) {
    if ((*dev)->getMemBytes() > maxMem) {
      maxMemDev = *dev;
    }
  }
  *out_device = maxMemDev;
#else
  std::cout << " WARNING: OpenCL not found at compile time"
            << std::endl;
  return -1;
#endif /* HAS_OPENCL */
  return 1;
}

int ocl_begin_session(void * in_dev,
                      void ** out_ctx,
                      void ** out_q) {
#ifdef HAS_OPENCL
  OclDevice * selectedDev = static_cast<OclDevice *>(in_dev);
  cl_device_id dev_id = selectedDev->getDevId();

  cl_int status = CL_SUCCESS;
  cl_context * context = new cl_context();
  *context = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &status);
  ocl_check_status(status);
  if (status == CL_SUCCESS) {
    *out_ctx = static_cast<void *>(context);
  }

  status = CL_SUCCESS;
  cl_command_queue * queue = new cl_command_queue();
  *queue = clCreateCommandQueue(*context, dev_id, 0, &status);
  ocl_check_status(status);
  if (status == CL_SUCCESS) {
    *out_q = static_cast<void *>(queue);
  }

  return (int) status;
#else
  std::cout << " WARNING: OpenCL not found at compile time"
            << std::endl;
  return -1;
#endif /* HAS_OPENCL */
}

int ocl_end_session(void ** ctx, void ** q) {
#ifdef HAS_OPENCL
  cl_context * context_ptr = static_cast<cl_context *>(*ctx);
  cl_context context = *context_ptr;
  
  cl_command_queue * queue_ptr = static_cast<cl_command_queue *>(*q);
  cl_command_queue queue = *queue_ptr;

  cl_int status;
  status = clFlush(queue);
  ocl_check_status(status);

  status = clFinish(queue);
  ocl_check_status(status);

  status = clReleaseCommandQueue(queue);
  ocl_check_status(status);

  status = clReleaseContext(context);
  ocl_check_status(status);

  delete context_ptr;
  *ctx = NULL;
  
  delete queue_ptr;
  *q = NULL;

  return status;
#else
  std::cout << " WARNING: OpenCL not found at compile time"
            << std::endl;
  return -1;
#endif /* HAS_OPENCL */
}
