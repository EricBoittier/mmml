#if KEY_FFTDOCK == 1
#if HAS_OPENCL == 1

// target OpenCL 1.2 devices
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <clFFT.h>
#include <stdio.h>

#include <kernels.h>
#include <ocl_util.h>

#define clfft_check_status ocl_check_status

extern "C"
void init_fft() {
  clfftSetupData fftSetup;
  cl_int err;
  err = clfftInitSetupData(&fftSetup);
  clfft_check_status(err);

  err = clfftSetup(&fftSetup);
  clfft_check_status(err);
}

extern "C"
void tear_down_fft() {
  cl_int err = clfftTeardown();
  clfft_check_status(err);
}

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

  cl_context * context_ptr = static_cast<cl_context *>(ocl_context);
  cl_context ctx = *context_ptr;

  cl_command_queue * q_ptr = static_cast<cl_command_queue *>(ocl_queue);
  cl_command_queue queue = *q_ptr;

  int inembed[3];
  inembed[0] = xdim;
  inembed[1] = ydim;
  inembed[2] = zdim;
  int idist = inembed[0] * inembed[1] * inembed[2];

  int onembed[3];
  onembed[0] = xdim;
  onembed[1] = ydim;
  onembed[2] = (zdim / 2) + 1;
  int odist = onembed[0] * onembed[1] * onembed[2];

  cl_mem
    * d_lig_complex_ptr = NULL,
    * d_potential_real_ptr = NULL,
    * d_potential_complex_ptr = NULL,
    * d_lig_sum_real_ptr = NULL,
    * d_lig_sum_complex_ptr = NULL;

  cl_int status = CL_SUCCESS;

  // FFT transformation for potential grids
  if (idx_batch == 1) {
    d_potential_real_ptr = new cl_mem();
    status = CL_SUCCESS;
    *d_potential_real_ptr = clCreateBuffer(ctx,
                                        CL_MEM_READ_ONLY |
                                        CL_MEM_COPY_HOST_PTR,
                                        sizeof(float) * num_grid * idist,
                                        (void *) grid_potential,
                                        &status);
    ocl_check_status(status);
    *d_GridPot_Fort = static_cast<void *>(d_potential_real_ptr);

    d_potential_complex_ptr = new cl_mem();
    status = CL_SUCCESS;
    *d_potential_complex_ptr = clCreateBuffer(ctx,
                                              CL_MEM_READ_WRITE,
                                              2 * sizeof(float)
                                                * num_grid * odist,
                                              NULL, &status);
    ocl_check_status(status);
    *d_GridPot_FFT_Fort = static_cast<void *>(d_potential_complex_ptr);

    status = CL_SUCCESS;
    int d_lig_complex_bytes = 2 * sizeof(float) * num_grid * batch_size * odist;
    d_lig_complex_ptr = new cl_mem();
    *d_lig_complex_ptr = clCreateBuffer(ctx,
                                        CL_MEM_READ_WRITE,
                                        d_lig_complex_bytes,
                                        NULL, &status);
    ocl_check_status(status);
    *d_LigGrid_FFT_Fort = static_cast<void *>(d_lig_complex_ptr);

    d_lig_sum_complex_ptr = new cl_mem();
    status = CL_SUCCESS;
    int d_lig_sum_complex_bytes = 2 * sizeof(float) * batch_size * odist;
    *d_lig_sum_complex_ptr = clCreateBuffer(ctx,
                                            CL_MEM_READ_WRITE,
                                            d_lig_sum_complex_bytes,
                                            NULL, &status);
    ocl_check_status(status);
    *d_LigSum_FFT_Fort = static_cast<void *>(d_lig_sum_complex_ptr);

    d_lig_sum_real_ptr = new cl_mem();
    status = CL_SUCCESS;
    int d_lig_sum_real_bytes = sizeof(float) * batch_size * idist;
    *d_lig_sum_real_ptr = clCreateBuffer(ctx,
                                         CL_MEM_READ_WRITE,
                                         d_lig_sum_real_bytes,
                                         NULL, &status);
    ocl_check_status(status);
    *d_LigSum_Fort = static_cast<void *>(d_lig_sum_real_ptr);
  } else {
    d_potential_real_ptr = static_cast<cl_mem *>(*d_GridPot_Fort);
    d_potential_complex_ptr = static_cast<cl_mem *>(*d_GridPot_FFT_Fort);
    d_lig_complex_ptr = static_cast<cl_mem *>(*d_LigGrid_FFT_Fort);
    d_lig_sum_real_ptr = static_cast<cl_mem *>(*d_LigSum_Fort);
    d_lig_sum_complex_ptr = static_cast<cl_mem *>(*d_LigSum_FFT_Fort);
  }

  clfftPlanHandle * pot_plan = static_cast<clfftPlanHandle *>(potential_r2c_plan);
  clfftStatus clfftResult = clfftEnqueueTransform(*pot_plan,
                                                  CLFFT_FORWARD, 1,
                                                  q_ptr, 0, NULL, NULL,
                                                  d_potential_real_ptr,
                                                  d_potential_complex_ptr,
                                                  NULL);
  clfft_check_status(clfftResult);

  // FFT transform for ligand grids
  //printf("FFT memory=%d\n",d_LigGrid_Fort);

  //cudaMalloc((void **)&d_lig_f, sizeof(cufftReal)*num_grid*batch_size*idist);
  //CUDA_CHECK();
  //cudaMemcpy(d_lig_f, LigGrid,
  // 	     sizeof(cufftReal)*num_grid*batch_size*idist,
  //     cudaMemcpyHostToDevice);

  clfftPlanHandle * lig_plan = static_cast<clfftPlanHandle *>(lig_r2c_plan);
  cl_mem * d_lig_real_ptr = static_cast<cl_mem *>(*d_LigGrid_Fort);
  clfftResult = clfftEnqueueTransform(*lig_plan,
                                      CLFFT_FORWARD, 1,
                                      q_ptr, 0, NULL, NULL,
                                      d_lig_real_ptr, d_lig_complex_ptr,
                                      NULL);
  clfft_check_status(clfftResult);

  status = clFinish(queue);
  ocl_check_status(status);

  cl_kernel conj_mult_kernel;
  status = ocl_compile_kernel(Kernels::conjMult, "conjMult",
                              ctx, dev_id,
                              conj_mult_kernel);
  if (status != CL_SUCCESS) {
    return;
  }

  int conj_mult_size = batch_size * num_grid * odist;
  status = clSetKernelArg(conj_mult_kernel, 0, sizeof(int), (void *) &conj_mult_size);
  ocl_check_status(status);

  status = clSetKernelArg(conj_mult_kernel, 1, sizeof(cl_mem), (void *) d_potential_complex_ptr);
  ocl_check_status(status);

  status = clSetKernelArg(conj_mult_kernel, 2, sizeof(cl_mem), (void *) d_lig_complex_ptr);
  ocl_check_status(status);

  status = clSetKernelArg(conj_mult_kernel, 3, sizeof(int), (void *) &odist);
  ocl_check_status(status);

  status = clSetKernelArg(conj_mult_kernel, 4, sizeof(int), (void *) &num_grid);
  ocl_check_status(status);

  size_t
    localSize = 256,
    globalSize = 1024 * localSize;

  // Inverse FFT transform to calcualte energy grids
  status = clEnqueueNDRangeKernel(queue, conj_mult_kernel, 1, NULL,
                                  &globalSize, &localSize,
                                  0, NULL, NULL);
  ocl_check_status(status);

  cl_kernel sum_grids_kernel;
  status = ocl_compile_kernel(Kernels::sumGrids, "sumGrids",
                              ctx, dev_id,
                              sum_grids_kernel);
  if (status != CL_SUCCESS) {
    return;
  }

  int sum_grids_size = batch_size * odist;
  status = clSetKernelArg(sum_grids_kernel, 0, sizeof(int), (void *) &sum_grids_size);
  ocl_check_status(status);

  status = clSetKernelArg(sum_grids_kernel, 1, sizeof(cl_mem), (void *) d_lig_complex_ptr);
  ocl_check_status(status);

  status = clSetKernelArg(sum_grids_kernel, 2, sizeof(cl_mem), (void *) d_lig_sum_complex_ptr);
  ocl_check_status(status);

  status = clSetKernelArg(sum_grids_kernel, 3, sizeof(int), (void *) &num_grid);
  ocl_check_status(status);

  status = clSetKernelArg(sum_grids_kernel, 4, sizeof(int), (void *) &odist);
  ocl_check_status(status);

  status = clSetKernelArg(sum_grids_kernel, 5, sizeof(int), (void *) &idist);
  ocl_check_status(status);

  status = clEnqueueNDRangeKernel(queue, sum_grids_kernel, 1, NULL,
                                  &globalSize, &localSize,
                                  0, NULL, NULL);
  ocl_check_status(status);

  clfftPlanHandle * back_plan = static_cast<clfftPlanHandle *>(c2r_plan);
  clfftResult = clfftEnqueueTransform(*back_plan, CLFFT_BACKWARD, 1, q_ptr,
                                      0, NULL, NULL,
                                      d_lig_sum_complex_ptr, d_lig_sum_real_ptr, NULL);
  clfft_check_status(clfftResult);

  status = clFinish(queue);
  ocl_check_status(status);

  // copy energy grid from GPU to CPU
  //printf("batch_size: %d \n", batch_size);
  //printf("idist: %d \n", idist);

  cl_kernel correct_ener_kernel;
  status = ocl_compile_kernel(Kernels::correctEnergy, "correctEnergy",
                              ctx, dev_id,
                              correct_ener_kernel);
  if (status != CL_SUCCESS) {
    return;
  }

  int correct_ener_size = batch_size * idist;
  status = clSetKernelArg(correct_ener_kernel, 0, sizeof(int), (void *) &correct_ener_size);
  ocl_check_status(status);

  status = clSetKernelArg(correct_ener_kernel, 1, sizeof(int), (void *) &idist);
  ocl_check_status(status);

  status = clSetKernelArg(correct_ener_kernel, 2, sizeof(cl_mem), (void *) d_lig_sum_real_ptr);
  ocl_check_status(status);

  status = clEnqueueNDRangeKernel(queue, correct_ener_kernel, 1, NULL,
                                  &globalSize, &localSize,
                                  0, NULL, NULL);
  ocl_check_status(status);

  status = clFinish(queue);
  ocl_check_status(status);

  // copy energy grid from GPU to CPU
  status = clEnqueueReadBuffer(queue, *d_lig_sum_real_ptr, CL_TRUE, 0,
                               sizeof(float) * correct_ener_size,
                               EnergyGrid, 0, NULL, NULL);
  ocl_check_status(status);

  //removed for efficiency
  //for (int i = 0; i < batch_size*idist; i++)
  //{
  //  EnergyGrid[i] = EnergyGrid[i] / sqrt(idist);
  //}

  //GPU memory deallocation was moved to clean_FFTDock_GPU
  //cudaFree(d_potential_f);
  //cudaFree(d_potential_F);
  //cudaFree(d_lig_f);
  //cudaFree(d_lig_F);
  //cudaFree(d_lig_sum_F);
  //cudaFree(d_lig_sum_f);
}

extern "C"
void destroy_fft_plan(void ** clfft_plan) {
  clfftPlanHandle * plan = static_cast<clfftPlanHandle *>(*clfft_plan);
  clfftStatus clfftResult = clfftDestroyPlan(plan);
  clfft_check_status(clfftResult);
  delete plan;
  *clfft_plan = NULL;
}

extern "C"
void make_fft_r2c_plan(void * ocl_context, void * ocl_queue,
                       size_t x, size_t y, size_t z,
                       size_t batch_size,
                       void ** out_plan) {
  cl_context * ctx_ptr = static_cast<cl_context *>(ocl_context);
  cl_context ctx = *ctx_ptr;

  cl_command_queue * q_ptr = static_cast<cl_command_queue *>(ocl_queue);

  size_t
    inembed[3] = {x, y, z},
    onembed[3] = {x, y, 1 + z / 2},
    idist = x * y * z,
    odist = x * y * (1 + z / 2),
    stride[3] = {1, x, x * y};

  clfftStatus clfftResult;

  clfftPlanHandle * plan = new clfftPlanHandle();
  clfftResult = clfftCreateDefaultPlan(plan, ctx, CLFFT_3D, inembed);
  clfft_check_status(clfftResult);
  if (clfftResult == CLFFT_SUCCESS) {
    *out_plan = static_cast<void *>(plan);
  }

  clfftResult = clfftSetLayout(*plan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanBatchSize(*plan, batch_size);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanDistance(*plan, idist, odist);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanInStride(*plan, CLFFT_3D, stride);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanOutStride(*plan, CLFFT_3D, stride);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanPrecision(*plan, CLFFT_SINGLE);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetResultLocation(*plan, CLFFT_OUTOFPLACE);
  clfft_check_status(clfftResult);

  clfftResult = clfftBakePlan(*plan, 1, q_ptr, NULL, NULL);
  clfft_check_status(clfftResult);
};

extern "C"
void make_fft_c2r_plan(void * ocl_context, void * ocl_queue,
                       size_t x, size_t y, size_t z,
                       size_t batch_size,
                       void ** out_plan) {
  cl_context * ctx_ptr = static_cast<cl_context *>(ocl_context);
  cl_context ctx = *ctx_ptr;

  cl_command_queue * q_ptr = static_cast<cl_command_queue *>(ocl_queue);

  size_t
    inembed[3] = {x, y, z},
    onembed[3] = {x, y, 1 + z / 2},
    idist = x * y * z,
    odist = x * y * (1 + z / 2),
    stride[3] = {1, x, x * y};

  clfftStatus clfftResult;

  clfftPlanHandle * plan = new clfftPlanHandle();
  clfftResult = clfftCreateDefaultPlan(plan, ctx, CLFFT_3D, inembed);
  clfft_check_status(clfftResult);
  if (clfftResult == CLFFT_SUCCESS) {
    *out_plan = static_cast<void *>(plan);
  }

  clfftResult = clfftCreateDefaultPlan(plan, ctx, CLFFT_3D, onembed);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetLayout(*plan, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanBatchSize(*plan, batch_size);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanDistance(*plan, odist, idist);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanInStride(*plan, CLFFT_3D, stride);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanOutStride(*plan, CLFFT_3D, stride);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetPlanPrecision(*plan, CLFFT_SINGLE);
  clfft_check_status(clfftResult);

  clfftResult = clfftSetResultLocation(*plan, CLFFT_OUTOFPLACE);
  clfft_check_status(clfftResult);

  clfftResult = clfftBakePlan(*plan, 1, q_ptr, NULL, NULL);
  clfft_check_status(clfftResult);
};

extern "C"
void batchFFT(cl_context * ctx,
              cl_command_queue * q,
              clfftPlanHandle plan,
              size_t x, size_t y, size_t z,
              size_t batch_size,
              float * grid_potential)
{
  size_t
    inembed[3] = {x, y, z},
    onembed[3] = {x, y, 1 + z / 2},
    idist = x * y * z,
    odist = x * y * (1 + z / 2),
    stride[3] = {1, x, x * y};

  clfftStatus clfftResult;

  cl_int status = CL_SUCCESS;
  cl_mem d_potential_f = clCreateBuffer(*ctx,
                                        CL_MEM_READ_ONLY |
                                        CL_MEM_COPY_HOST_PTR,
                                        sizeof(float) * batch_size * idist,
                                        (void *) grid_potential,
                                        &status);
  ocl_check_status(status);

  status = CL_SUCCESS;
  cl_mem d_potential_F = clCreateBuffer(*ctx, CL_MEM_READ_WRITE,
                                        2 * sizeof(float) * batch_size * odist,
                                        NULL, &status);
  ocl_check_status(status);

  clfftStatus fft_status = clfftEnqueueTransform(plan, CLFFT_FORWARD, 1,
                                                 q, 0, NULL, NULL,
                                                 &d_potential_f, &d_potential_F,
                                                 NULL);
  clfft_check_status(fft_status);

  status = clFinish(*q);
  ocl_check_status(status);

  status = clReleaseMemObject(d_potential_f);
  ocl_check_status(status);

  clReleaseMemObject(d_potential_f);
  ocl_check_status(status);

  clReleaseMemObject(d_potential_F);
  ocl_check_status(status);
}

extern "C"
void clean_fftdock_gpu(void ** d_LigGrid_Fort,
                       void ** d_LigGrid_FFT_Fort,
                       void ** d_GridPot_Fort,
                       void ** d_GridPot_FFT_Fort,
                       void ** d_LigSum_Fort,
                       void ** d_LigSum_FFT_Fort) {
  cl_mem * lig_grid_real = static_cast<cl_mem *>(*d_LigGrid_Fort);
  cl_int status = clReleaseMemObject(*lig_grid_real);
  ocl_check_status(status);
  delete lig_grid_real;
  *d_LigGrid_Fort = NULL;

  cl_mem * lig_grid_complex = static_cast<cl_mem *>(*d_LigGrid_FFT_Fort);
  status = clReleaseMemObject(*lig_grid_complex);
  ocl_check_status(status);
  delete lig_grid_complex;
  *d_LigGrid_FFT_Fort = NULL;

  cl_mem * grid_pot_real = static_cast<cl_mem *>(*d_GridPot_Fort);
  status = clReleaseMemObject(*grid_pot_real);
  ocl_check_status(status);
  delete grid_pot_real;
  *d_GridPot_Fort = NULL;

  cl_mem * grid_pot_complex = static_cast<cl_mem *>(*d_GridPot_FFT_Fort);
  status = clReleaseMemObject(*grid_pot_complex);
  ocl_check_status(status);
  delete grid_pot_complex;
  *d_GridPot_FFT_Fort = NULL;

  cl_mem * lig_sum_real = static_cast<cl_mem *>(*d_LigSum_Fort);
  status = clReleaseMemObject(*lig_sum_real);
  ocl_check_status(status);
  delete lig_sum_real;
  *d_LigSum_Fort = NULL;

  cl_mem * lig_sum_complex = static_cast<cl_mem *>(*d_LigSum_FFT_Fort);
  status = clReleaseMemObject(*lig_sum_complex);
  ocl_check_status(status);
  delete lig_sum_complex;
  *d_LigSum_FFT_Fort = NULL;
}

#endif /* HAS_OPENCL */
#endif /* KEY_FFTDOCK == 1 */
