#ifdef HAS_OPENCL

#include <iostream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif /* APPLE */

#define ocl_check_status(code) if (code != CL_SUCCESS) std::cout << ocl_get_error(code) << " " << __FILE__ << " : " << __LINE__ << std::endl;

class OclDevice {
 public:
  OclDevice(size_t newId, cl_device_id newDevId);
  size_t getId();
  cl_device_id getDevId();
  size_t getMemBytes();
  std::string getName();
  std::string toString();
 private:
  size_t id;
  cl_device_id devId;
  cl_ulong mem;  // in bytes
  std::string name;
};

std::string ocl_get_error(cl_int err);

cl_int ocl_compile_kernel(std::string kernelSrc,
                          std::string kernelName,
                          cl_context context, cl_device_id devId,
                          cl_kernel &outKernel);

cl_int ocl_clean_up(cl_kernel &kern,
                    cl_command_queue &queue, cl_context &context);
#endif /* HAS_OPENCL */

extern "C" {
  int ocl_device_init(void ** fortran_devices);
  void ocl_device_print(void * fortran_devices);
  void ocl_device_print_one(void * dev);
  void ocl_device_string(void * dev, char * c_string, int max_size);
  int ocl_device_get(void * fortran_devices, int dev_id, void ** out_device);
  int ocl_device_max_mem_get(void * fortran_devices, void ** out_device);
  int ocl_begin_session(void * in_dev, void ** out_ctx, void ** out_q);
  int ocl_end_session(void ** ctx, void ** q);
}
