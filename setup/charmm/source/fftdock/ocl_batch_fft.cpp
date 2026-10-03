#if KEY_FFTDOCK == 1
#if HAS_OPENCL == 1

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <kernels.h>
#include <ocl_util.h>

#ifdef VKFFT_BACKEND
/* ================================================================== */
/*  VkFFT implementation                                               */
/* ================================================================== */

#include <vkFFT.h>

/*
 * VkFFT plan wrapper.
 *
 * Stores the VkFFT application object and the OpenCL handles that must
 * remain valid for the lifetime of the plan.  The bufferSize /
 * inputBufferSize members are pointed to by the VkFFT configuration,
 * so they must also survive.
 */
struct VkFFTPlanWrapper {
  VkFFTApplication app;
  uint64_t bufferSize;
  uint64_t inputBufferSize;
  cl_context context;
  cl_device_id device;
};

extern "C" void init_fft() {}
extern "C" void tear_down_fft() {}

/*
 * Helper: create a VkFFT R2C plan.
 *
 * Grid data layout (set by the grid-generation kernels):
 *   linear index = ix * ydim * zdim + iy * zdim + iz
 * i.e. Z is the fastest-varying (contiguous) dimension.
 *
 * VkFFT size[0] = fastest dimension, so:
 *   size[0] = zdim,  size[1] = ydim,  size[2] = xdim
 *
 * For R2C the last *logical* dimension (size[0] = zdim) is halved in
 * the complex output: odist = xdim * ydim * (zdim/2 + 1).
 *
 * inverseReturn – if true, sets inverseReturnToInputBuffer so that an
 * inverse (C2R) transform writes to the real (inputBuffer) side.
 */
static VkFFTPlanWrapper * create_plan(cl_context ctx, cl_command_queue queue,
                                      int x, int y, int z,
                                      int batch, int inverseReturn) {
  VkFFTPlanWrapper * w = new VkFFTPlanWrapper();
  memset(w, 0, sizeof(*w));

  w->context = ctx;
  clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
                        sizeof(cl_device_id), &w->device, NULL);

  uint64_t idist = (uint64_t)x * y * z;
  uint64_t odist = (uint64_t)x * y * (z / 2 + 1);
  w->inputBufferSize = sizeof(float) * batch * idist;
  w->bufferSize      = 2 * sizeof(float) * batch * odist;

  VkFFTConfiguration cfg = {};
  cfg.FFTdim = 3;
  cfg.size[0] = z;   /* fastest — R2C halves this dimension */
  cfg.size[1] = y;
  cfg.size[2] = x;   /* slowest */
  cfg.coordinateFeatures = batch;
  cfg.performR2C = 1;
  if (inverseReturn)
    cfg.inverseReturnToInputBuffer = 1;

  cfg.device  = &w->device;
  cfg.context = &w->context;

  cfg.bufferNum  = 1;
  cfg.bufferSize = &w->bufferSize;

  cfg.isInputFormatted = 1;
  cfg.inputBufferNum   = 1;
  cfg.inputBufferSize  = &w->inputBufferSize;

  VkFFTResult res = initializeVkFFT(&w->app, cfg);
  if (res != VKFFT_SUCCESS) {
    fprintf(stderr, "VkFFT plan creation failed (error %d), "
            "dims %dx%dx%d batch %d\n", (int)res, x, y, z, batch);
    delete w;
    return NULL;
  }
  return w;
}

extern "C"
void make_fft_r2c_plan(void * ocl_context, void * ocl_queue,
                        int x, int y, int z, int batch_size,
                        void ** out_plan) {
  cl_context ctx   = *static_cast<cl_context *>(ocl_context);
  cl_command_queue q = *static_cast<cl_command_queue *>(ocl_queue);
  *out_plan = static_cast<void *>(create_plan(ctx, q, x, y, z, batch_size, 0));
}

extern "C"
void make_fft_c2r_plan(void * ocl_context, void * ocl_queue,
                        int x, int y, int z, int batch_size,
                        void ** out_plan) {
  cl_context ctx   = *static_cast<cl_context *>(ocl_context);
  cl_command_queue q = *static_cast<cl_command_queue *>(ocl_queue);
  *out_plan = static_cast<void *>(create_plan(ctx, q, x, y, z, batch_size, 1));
}

extern "C"
void destroy_fft_plan(void ** plan_ptr) {
  VkFFTPlanWrapper * w = static_cast<VkFFTPlanWrapper *>(*plan_ptr);
  if (w) {
    deleteVkFFT(&w->app);
    delete w;
  }
  *plan_ptr = NULL;
}

static void fft_forward_r2c(void * plan, cl_command_queue * queue,
                             cl_mem * real_buf, cl_mem * complex_buf) {
  VkFFTPlanWrapper * w = static_cast<VkFFTPlanWrapper *>(plan);
  VkFFTLaunchParams lp = {};
  lp.commandQueue = queue;
  lp.inputBuffer  = real_buf;
  lp.buffer       = complex_buf;
  VkFFTResult res = VkFFTAppend(&w->app, -1, &lp);
  if (res != VKFFT_SUCCESS)
    fprintf(stderr, "VkFFT R2C error %d\n", (int)res);
}

static void fft_inverse_c2r(void * plan, cl_command_queue * queue,
                              cl_mem * complex_buf, cl_mem * real_buf) {
  VkFFTPlanWrapper * w = static_cast<VkFFTPlanWrapper *>(plan);
  VkFFTLaunchParams lp = {};
  lp.commandQueue = queue;
  lp.buffer      = complex_buf;
  lp.inputBuffer = real_buf;
  VkFFTResult res = VkFFTAppend(&w->app, 1, &lp);
  if (res != VKFFT_SUCCESS)
    fprintf(stderr, "VkFFT C2R error %d\n", (int)res);
}

#else /* !VKFFT_BACKEND — use clFFT */
/* ================================================================== */
/*  clFFT implementation                                               */
/* ================================================================== */

#include <clFFT.h>

#define clfft_check_status ocl_check_status

extern "C"
void init_fft() {
  clfftSetupData fftSetup;
  cl_int err = clfftInitSetupData(&fftSetup);
  clfft_check_status(err);
  err = clfftSetup(&fftSetup);
  clfft_check_status(err);
}

extern "C"
void tear_down_fft() {
  cl_int err = clfftTeardown();
  clfft_check_status(err);
}

/*
 * Create a clFFT R2C or C2R plan.
 *
 * clFFT uses column-major convention: dims[0] = fastest dimension.
 * Our grid layout has Z fastest, so we pass dims = {z, y, x}.
 * Default strides are then {1, z, z*y} which matches our data.
 * R2C halves dim[0] in the output: odist = (z/2+1) * y * x.
 */
static clfftPlanHandle * create_clfft_plan(cl_context ctx,
                                            cl_command_queue * q_ptr,
                                            int x, int y, int z,
                                            int batch_size,
                                            int is_c2r) {
  /* dims[0] = fastest varying = z */
  size_t dims[3] = {(size_t)z, (size_t)y, (size_t)x};
  size_t idist = (size_t)x * y * z;
  size_t odist = (size_t)x * y * (z / 2 + 1);

  clfftPlanHandle * plan = new clfftPlanHandle();
  clfftStatus res;

  res = clfftCreateDefaultPlan(plan, ctx, CLFFT_3D, dims);
  clfft_check_status(res);

  if (is_c2r) {
    res = clfftSetLayout(*plan, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
    clfft_check_status(res);
    res = clfftSetPlanDistance(*plan, odist, idist);
  } else {
    res = clfftSetLayout(*plan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    clfft_check_status(res);
    res = clfftSetPlanDistance(*plan, idist, odist);
  }
  clfft_check_status(res);

  res = clfftSetPlanBatchSize(*plan, batch_size);
  clfft_check_status(res);

  /* Use default column-major strides: {1, z, z*y} */

  res = clfftSetPlanPrecision(*plan, CLFFT_SINGLE);
  clfft_check_status(res);
  res = clfftSetResultLocation(*plan, CLFFT_OUTOFPLACE);
  clfft_check_status(res);

  res = clfftBakePlan(*plan, 1, q_ptr, NULL, NULL);
  clfft_check_status(res);

  return plan;
}

extern "C"
void make_fft_r2c_plan(void * ocl_context, void * ocl_queue,
                        int x, int y, int z, int batch_size,
                        void ** out_plan) {
  cl_context ctx = *static_cast<cl_context *>(ocl_context);
  cl_command_queue * q_ptr = static_cast<cl_command_queue *>(ocl_queue);
  *out_plan = static_cast<void *>(create_clfft_plan(ctx, q_ptr,
                                                     x, y, z, batch_size, 0));
}

extern "C"
void make_fft_c2r_plan(void * ocl_context, void * ocl_queue,
                        int x, int y, int z, int batch_size,
                        void ** out_plan) {
  cl_context ctx = *static_cast<cl_context *>(ocl_context);
  cl_command_queue * q_ptr = static_cast<cl_command_queue *>(ocl_queue);
  *out_plan = static_cast<void *>(create_clfft_plan(ctx, q_ptr,
                                                     x, y, z, batch_size, 1));
}

extern "C"
void destroy_fft_plan(void ** plan_ptr) {
  clfftPlanHandle * plan = static_cast<clfftPlanHandle *>(*plan_ptr);
  if (plan) {
    clfftDestroyPlan(plan);
    delete plan;
  }
  *plan_ptr = NULL;
}

static void fft_forward_r2c(void * plan, cl_command_queue * queue,
                             cl_mem * real_buf, cl_mem * complex_buf) {
  clfftPlanHandle * p = static_cast<clfftPlanHandle *>(plan);
  clfftStatus res = clfftEnqueueTransform(*p, CLFFT_FORWARD, 1, queue,
                                          0, NULL, NULL,
                                          real_buf, complex_buf, NULL);
  clfft_check_status(res);
}

static void fft_inverse_c2r(void * plan, cl_command_queue * queue,
                              cl_mem * complex_buf, cl_mem * real_buf) {
  clfftPlanHandle * p = static_cast<clfftPlanHandle *>(plan);
  clfftStatus res = clfftEnqueueTransform(*p, CLFFT_BACKWARD, 1, queue,
                                          0, NULL, NULL,
                                          complex_buf, real_buf, NULL);
  clfft_check_status(res);
}

#endif /* VKFFT_BACKEND */

/* ================================================================== */
/*  rigid_fft_dock – main FFT docking pipeline (shared)                */
/* ================================================================== */

extern "C"
void rigid_fft_dock(void * ocl_device, void * ocl_context, void * ocl_queue,
                    int xdim, int ydim, int zdim,
                    int batch_size, int idx_batch,
                    int num_quaternions, int num_grid,
                    void * potential_r2c_plan,
                    void * lig_r2c_plan,
                    void * c2r_plan,
                    const float * grid_potential,
                    float * EnergyGrid,
                    void ** d_LigGrid_Fort, void ** d_LigGrid_FFT_Fort,
                    void ** d_GridPot_Fort, void ** d_GridPot_FFT_Fort,
                    void ** d_LigSum_Fort, void ** d_LigSum_FFT_Fort) {

  OclDevice * selectedDev = static_cast<OclDevice *>(ocl_device);
  cl_device_id dev_id = selectedDev->getDevId();

  cl_context ctx   = *static_cast<cl_context *>(ocl_context);
  cl_command_queue queue = *static_cast<cl_command_queue *>(ocl_queue);
  cl_command_queue * q_ptr = static_cast<cl_command_queue *>(ocl_queue);

  int idist = xdim * ydim * zdim;
  int odist = xdim * ydim * (zdim / 2 + 1);

  cl_mem * d_potential_real_ptr    = NULL;
  cl_mem * d_potential_complex_ptr = NULL;
  cl_mem * d_lig_complex_ptr       = NULL;
  cl_mem * d_lig_sum_complex_ptr   = NULL;
  cl_mem * d_lig_sum_real_ptr      = NULL;

  cl_int status;

  /* ------ allocate GPU buffers (first batch only) ------ */

  if (idx_batch == 1) {
    status = CL_SUCCESS;
    d_potential_real_ptr = new cl_mem();
    *d_potential_real_ptr = clCreateBuffer(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * num_grid * idist,
        (void *)grid_potential, &status);
    ocl_check_status(status);
    *d_GridPot_Fort = static_cast<void *>(d_potential_real_ptr);

    status = CL_SUCCESS;
    d_potential_complex_ptr = new cl_mem();
    *d_potential_complex_ptr = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        2 * sizeof(float) * num_grid * odist, NULL, &status);
    ocl_check_status(status);
    *d_GridPot_FFT_Fort = static_cast<void *>(d_potential_complex_ptr);

    status = CL_SUCCESS;
    d_lig_complex_ptr = new cl_mem();
    *d_lig_complex_ptr = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        2 * sizeof(float) * num_grid * batch_size * odist, NULL, &status);
    ocl_check_status(status);
    *d_LigGrid_FFT_Fort = static_cast<void *>(d_lig_complex_ptr);

    status = CL_SUCCESS;
    d_lig_sum_complex_ptr = new cl_mem();
    *d_lig_sum_complex_ptr = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        2 * sizeof(float) * batch_size * odist, NULL, &status);
    ocl_check_status(status);
    *d_LigSum_FFT_Fort = static_cast<void *>(d_lig_sum_complex_ptr);

    status = CL_SUCCESS;
    d_lig_sum_real_ptr = new cl_mem();
    *d_lig_sum_real_ptr = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(float) * batch_size * idist, NULL, &status);
    ocl_check_status(status);
    *d_LigSum_Fort = static_cast<void *>(d_lig_sum_real_ptr);
  } else {
    d_potential_real_ptr    = static_cast<cl_mem *>(*d_GridPot_Fort);
    d_potential_complex_ptr = static_cast<cl_mem *>(*d_GridPot_FFT_Fort);
    d_lig_complex_ptr       = static_cast<cl_mem *>(*d_LigGrid_FFT_Fort);
    d_lig_sum_complex_ptr   = static_cast<cl_mem *>(*d_LigSum_FFT_Fort);
    d_lig_sum_real_ptr      = static_cast<cl_mem *>(*d_LigSum_Fort);
  }

  /* ------ 1. Forward R2C: potential grids ------ */

  fft_forward_r2c(potential_r2c_plan, q_ptr,
                   d_potential_real_ptr, d_potential_complex_ptr);

  /* ------ 2. Forward R2C: ligand grids ------ */

  cl_mem * d_lig_real_ptr = static_cast<cl_mem *>(*d_LigGrid_Fort);
  fft_forward_r2c(lig_r2c_plan, q_ptr,
                   d_lig_real_ptr, d_lig_complex_ptr);

  status = clFinish(queue);
  ocl_check_status(status);

  /* ------ 3. Conjugate multiplication ------ */

  cl_kernel conj_mult_kernel;
  status = ocl_compile_kernel(Kernels::gpu_compat + "\n" + Kernels::conjMult,
                              "conjMult", ctx, dev_id, conj_mult_kernel);
  if (status != CL_SUCCESS) return;

  int conj_N = batch_size * num_grid * odist;   /* complex-element count */
  status = clSetKernelArg(conj_mult_kernel, 0, sizeof(int),    &conj_N);
  ocl_check_status(status);
  status = clSetKernelArg(conj_mult_kernel, 1, sizeof(cl_mem), d_potential_complex_ptr);
  ocl_check_status(status);
  status = clSetKernelArg(conj_mult_kernel, 2, sizeof(cl_mem), d_lig_complex_ptr);
  ocl_check_status(status);
  status = clSetKernelArg(conj_mult_kernel, 3, sizeof(int),    &odist);
  ocl_check_status(status);
  status = clSetKernelArg(conj_mult_kernel, 4, sizeof(int),    &num_grid);
  ocl_check_status(status);

  size_t localSize  = 256;
  size_t globalSize = 1024 * localSize;

  status = clEnqueueNDRangeKernel(queue, conj_mult_kernel, 1, NULL,
                                  &globalSize, &localSize, 0, NULL, NULL);
  ocl_check_status(status);

  /* ------ 4. Sum grids across grid types ------ */

  cl_kernel sum_grids_kernel;
  status = ocl_compile_kernel(Kernels::gpu_compat + "\n" + Kernels::sumGrids,
                              "sumGrids", ctx, dev_id, sum_grids_kernel);
  if (status != CL_SUCCESS) return;

  int sum_N = batch_size * odist;               /* complex-element count */
  status = clSetKernelArg(sum_grids_kernel, 0, sizeof(int),    &sum_N);
  ocl_check_status(status);
  status = clSetKernelArg(sum_grids_kernel, 1, sizeof(cl_mem), d_lig_complex_ptr);
  ocl_check_status(status);
  status = clSetKernelArg(sum_grids_kernel, 2, sizeof(cl_mem), d_lig_sum_complex_ptr);
  ocl_check_status(status);
  status = clSetKernelArg(sum_grids_kernel, 3, sizeof(int),    &num_grid);
  ocl_check_status(status);
  status = clSetKernelArg(sum_grids_kernel, 4, sizeof(int),    &odist);
  ocl_check_status(status);
  status = clSetKernelArg(sum_grids_kernel, 5, sizeof(int),    &idist);
  ocl_check_status(status);

  status = clEnqueueNDRangeKernel(queue, sum_grids_kernel, 1, NULL,
                                  &globalSize, &localSize, 0, NULL, NULL);
  ocl_check_status(status);

  /* ------ 5. Inverse C2R: energy grids ------ */

  fft_inverse_c2r(c2r_plan, q_ptr,
                   d_lig_sum_complex_ptr, d_lig_sum_real_ptr);

  status = clFinish(queue);
  ocl_check_status(status);

  /* ------ 6. Correct energy (divide by idist) ------ */

  cl_kernel correct_kernel;
  status = ocl_compile_kernel(Kernels::gpu_compat + "\n" + Kernels::correctEnergy,
                              "correctEnergy", ctx, dev_id, correct_kernel);
  if (status != CL_SUCCESS) return;

  int ener_N = batch_size * idist;
  status = clSetKernelArg(correct_kernel, 0, sizeof(int),    &ener_N);
  ocl_check_status(status);
  status = clSetKernelArg(correct_kernel, 1, sizeof(int),    &idist);
  ocl_check_status(status);
  status = clSetKernelArg(correct_kernel, 2, sizeof(cl_mem), d_lig_sum_real_ptr);
  ocl_check_status(status);

  status = clEnqueueNDRangeKernel(queue, correct_kernel, 1, NULL,
                                  &globalSize, &localSize, 0, NULL, NULL);
  ocl_check_status(status);

  status = clFinish(queue);
  ocl_check_status(status);

  /* ------ 7. Copy results back to host ------ */

  status = clEnqueueReadBuffer(queue, *d_lig_sum_real_ptr, CL_TRUE, 0,
                               sizeof(float) * ener_N,
                               EnergyGrid, 0, NULL, NULL);
  ocl_check_status(status);

  /* release compiled kernels */
  clReleaseKernel(conj_mult_kernel);
  clReleaseKernel(sum_grids_kernel);
  clReleaseKernel(correct_kernel);
}

/* ================================================================== */
/*  GPU memory cleanup (shared by both implementations)                */
/* ================================================================== */

extern "C"
void clean_fftdock_gpu(void ** d_LigGrid_Fort, void ** d_LigGrid_FFT_Fort,
                       void ** d_GridPot_Fort, void ** d_GridPot_FFT_Fort,
                       void ** d_LigSum_Fort,  void ** d_LigSum_FFT_Fort) {
  cl_mem * buf;
  cl_int status;

  buf = static_cast<cl_mem *>(*d_LigGrid_Fort);
  status = clReleaseMemObject(*buf); ocl_check_status(status);
  delete buf; *d_LigGrid_Fort = NULL;

  buf = static_cast<cl_mem *>(*d_LigGrid_FFT_Fort);
  status = clReleaseMemObject(*buf); ocl_check_status(status);
  delete buf; *d_LigGrid_FFT_Fort = NULL;

  buf = static_cast<cl_mem *>(*d_GridPot_Fort);
  status = clReleaseMemObject(*buf); ocl_check_status(status);
  delete buf; *d_GridPot_Fort = NULL;

  buf = static_cast<cl_mem *>(*d_GridPot_FFT_Fort);
  status = clReleaseMemObject(*buf); ocl_check_status(status);
  delete buf; *d_GridPot_FFT_Fort = NULL;

  buf = static_cast<cl_mem *>(*d_LigSum_Fort);
  status = clReleaseMemObject(*buf); ocl_check_status(status);
  delete buf; *d_LigSum_Fort = NULL;

  buf = static_cast<cl_mem *>(*d_LigSum_FFT_Fort);
  status = clReleaseMemObject(*buf); ocl_check_status(status);
  delete buf; *d_LigSum_FFT_Fort = NULL;
}

#endif /* HAS_OPENCL */
#endif /* KEY_FFTDOCK == 1 */
