#if KEY_FFTDOCK == 1
#if HAS_OPENCL == 1
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include <kernels.h>
#include <ocl_util.h>

extern "C"
void calcLigGrid(void * ocl_selected_device,
                 void * ocl_context, void * ocl_queue,
                 int BatchIdx, int BatchSize,
                 int NumRotamers, int NumAtoms, int NumVdwGridUsed,
                 float DGrid,
                 int XGridNum, int YGridNum, int ZGridNum,
                 const float * SelectAtomsParameters,
                 const float * LigRotamerCoors,
                 const float * LigRotamerMinCoors,
                 void ** d_lig_grid_real) {
  OclDevice * selectedDev = static_cast<OclDevice *>(ocl_selected_device);
  cl_device_id dev_id = selectedDev->getDevId();

  cl_context * context_ptr = static_cast<cl_context *>(ocl_context);
  cl_context ctx = *context_ptr;

  cl_command_queue * queue_ptr = static_cast<cl_command_queue *>(ocl_queue);
  cl_command_queue q = *queue_ptr;

  // BatchIdx starting from 1
  int NumGrids = NumVdwGridUsed + 1;
  int RealBatchSize = BatchSize;
  if ((BatchIdx * BatchSize) > NumRotamers) {
    RealBatchSize = NumRotamers % BatchSize;
  }
  // printf("BatchIdx = %d RealBatchSize = %d\n",BatchIdx,RealBatchSize);

  // generate ligand grid properties
  int NumGridPoints = 1;
  int GridNum[3];
  GridNum[0] = XGridNum;
  GridNum[1] = YGridNum;
  GridNum[2] = ZGridNum;
  NumGridPoints = XGridNum * YGridNum * ZGridNum;
  // printf("GridNum x=%d y=%d z=%d\n",GridNum[0],GridNum[1],GridNum[2]);
  // printf("NumGridPoints=%d\n",NumGridPoints);

  cl_int status = CL_SUCCESS;
  cl_mem d_rotamersCoor = clCreateBuffer(ctx,
                                         CL_MEM_READ_ONLY |
                                         CL_MEM_COPY_HOST_PTR,
                                         3 * sizeof(float) * BatchSize * NumAtoms,
                                         (void *) LigRotamerCoors,
                                         &status);
  ocl_check_status(status);

  status = CL_SUCCESS;
  cl_mem d_GridMinCoor = clCreateBuffer(ctx,
                                        CL_MEM_READ_ONLY |
                                        CL_MEM_COPY_HOST_PTR,
                                        3 * sizeof(float) * BatchSize,
                                        (void *) LigRotamerMinCoors,
                                        &status);
  ocl_check_status(status);

  // printf("GridDim=%d BlockDim=%d\n",GridDim,BlockDim);
  // printf("Finished generating rotamers\n");
  // generate LigGrid
  cl_mem * d_LigGrid = new cl_mem();
  int LigGridSize = BatchSize * NumGrids * NumGridPoints * sizeof(float);
  // printf("Old Address=%p\n",d_LigGrid_F);
  if (BatchIdx == 1) {
    // printf("Allocate memory\n");
    status = CL_SUCCESS;
    *d_LigGrid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, LigGridSize,
                                NULL, &status);
    ocl_check_status(status);

    *d_lig_grid_real = static_cast<void *>(d_LigGrid);
    // printf("New Address1=%p\n",d_LigGrid);
    // printf("New Address2=%p\n",d_LigGrid_F);
  } else {
    d_LigGrid = static_cast<cl_mem *>(*d_lig_grid_real);
    // printf("Old Address=%p\n",d_LigGrid_F);
  }

  cl_float zero = 0.0;
  status = clEnqueueFillBuffer(q, *d_LigGrid, &zero, sizeof(cl_float), 0,
                               LigGridSize, 0, NULL, NULL);
  ocl_check_status(status);

  // atoms parameters
  status = CL_SUCCESS;
  cl_mem d_par = clCreateBuffer(ctx,
                                CL_MEM_READ_ONLY |
                                CL_MEM_COPY_HOST_PTR,
                                NumAtoms * 4 * sizeof(float),
                                (void *) SelectAtomsParameters,
                                &status);
  ocl_check_status(status);

  // grid dimension
  status = CL_SUCCESS;
  cl_mem d_GridNum = clCreateBuffer(ctx,
                                    CL_MEM_READ_ONLY |
                                    CL_MEM_COPY_HOST_PTR,
                                    3 * sizeof(int),
                                    (void *) GridNum,
                                    &status);
  ocl_check_status(status);

  cl_kernel lig_kernel;
  status = ocl_compile_kernel(Kernels::generateLigGrid, "generateLigGrid",
                              ctx, dev_id,
                              lig_kernel);
  if (status != CL_SUCCESS) {
    return;
  }

  status = clSetKernelArg(lig_kernel, 0, sizeof(int), (void *) &RealBatchSize);
  ocl_check_status(status);

  status = clSetKernelArg(lig_kernel, 1, sizeof(int), (void *) &NumAtoms);
  ocl_check_status(status);

  status = clSetKernelArg(lig_kernel, 2, sizeof(int), (void *) &NumGrids);
  ocl_check_status(status);

  status = clSetKernelArg(lig_kernel, 3, sizeof(cl_mem), (void *) &d_GridNum);
  ocl_check_status(status);

  status = clSetKernelArg(lig_kernel, 4, sizeof(float), (void *) &DGrid);
  ocl_check_status(status);

  status = clSetKernelArg(lig_kernel, 5, sizeof(cl_mem), (void *) &d_rotamersCoor);
  ocl_check_status(status);

  status = clSetKernelArg(lig_kernel, 6, sizeof(cl_mem), (void *) &d_par);
  ocl_check_status(status);

  status = clSetKernelArg(lig_kernel, 7, sizeof(cl_mem), (void *) &d_GridMinCoor);
  ocl_check_status(status);

  status = clSetKernelArg(lig_kernel, 8, sizeof(cl_mem), (void *) d_LigGrid);
  ocl_check_status(status);

  size_t
    localSize = 128,
    globalSize = ceil((float) BatchSize / (float) localSize) * localSize;

  status = clEnqueueNDRangeKernel(q, lig_kernel, 1, NULL,
                                  &globalSize, &localSize,
                                  0, NULL, NULL);
  ocl_check_status(status);

  // free GPU memory
  status = clReleaseMemObject(d_par);
  ocl_check_status(status);

  status = clReleaseMemObject(d_rotamersCoor);
  ocl_check_status(status);

  status = clReleaseMemObject(d_GridMinCoor);
  ocl_check_status(status);

  status = clReleaseMemObject(d_GridNum);
  ocl_check_status(status);
}
#endif /* HAS_OPENCL */
#endif /* KEY_FFTDOCK == 1 */
