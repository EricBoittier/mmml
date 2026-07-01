#if KEY_FFTDOCK == 1
#if HAS_OPENCL == 1
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include<math.h>

#include <kernels.h>
#include <ocl_util.h>

extern "C"
int calcPotGrid(void * ocl_selected_device,
                cl_context * ocl_context,
                cl_command_queue * ocl_queue,
                int NumGrids, int NumAtoms,
                float DGrid,
                int XGridLen, int YGridLen, int ZGridLen,
                float XMin, float YMin, float ZMin,
                float Fa , float Fb, float Gmax,
                float VdwEmax, float ElecAttrEmax,
                float ElecReplEmax, float CCELEC,
                int ElecMode, float Dielec,
                const float * selectAtomsParameters,
                float * GridPot,
                const float * GridRadii) {
  /*
    FILE * fp;
    fp = fopen("param.dat", "w");
    for(int ii = 0; ii < NumAtoms; ++ii){
    float x, y, z, eps, vdwr, cg;
    x = SelectAtomsParameters[6 * ii + 0];
    y = SelectAtomsParameters[6 * ii + 1];
    z = SelectAtomsParameters[6 * ii + 2];
    eps = SelectAtomsParameters[6 * ii + 3];
    vdwr = SelectAtomsParameters[6 * ii + 4];
    cg = SelectAtomsParameters[6 * ii + 5];
    fprintf(fp, "%f %f %f %f %f %f\n", x, y, z, eps, vdwr, cg);
    }
    fclose(fp);
  */
  
  const int NUM_PARAMS_PER_ATOM = 8;

  // the coordinates of the grid center

  // generate grid properties
  // number of grid points in three dimensions
  int gridNum[3];
  int NumGridPoints = 1;  // total number of grid points
  float GridMinCoor[3];  // the coordinate of the origin of the grid

  gridNum[0] = XGridLen;
  gridNum[1] = YGridLen;
  gridNum[2] = ZGridLen;
  GridMinCoor[0] = XMin;
  GridMinCoor[1] = YMin;
  GridMinCoor[2] = ZMin;
  NumGridPoints = gridNum[0] * gridNum[1] * gridNum[2];

  // allocate GPU memory
  int paramMemSize = NumAtoms * NUM_PARAMS_PER_ATOM * sizeof(float);
  int gridMemSize = NumGridPoints * NumGrids * sizeof(float);
  int probesMemSize = (NumGrids - 3) * sizeof(float);

  OclDevice * selectedDev = static_cast<OclDevice *>(ocl_selected_device);
  cl_device_id dev_id = selectedDev->getDevId();

  cl_context * context_ptr = static_cast<cl_context *>(ocl_context);
  cl_context context = *context_ptr;
  
  cl_command_queue * queue_ptr = static_cast<cl_command_queue *>(ocl_queue);
  cl_command_queue queue = *queue_ptr;
 
  cl_int status = CL_SUCCESS;
  cl_mem d_parameter = clCreateBuffer(context,
                                      CL_MEM_READ_ONLY |
                                      CL_MEM_COPY_HOST_PTR,
                                      paramMemSize, (void *) selectAtomsParameters,
                                      &status);
  ocl_check_status(status);

  status = CL_SUCCESS;
  cl_mem d_GridNum = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY |
                                    CL_MEM_COPY_HOST_PTR,
                                    3 * sizeof(int), gridNum,
                                    &status);
  ocl_check_status(status);

  status = CL_SUCCESS;
  cl_mem d_GridMinCoor = clCreateBuffer(context,
                                        CL_MEM_READ_ONLY |
                                        CL_MEM_COPY_HOST_PTR,
                                        3 * sizeof(float), GridMinCoor,
                                        &status);
  ocl_check_status(status);

  status = CL_SUCCESS;
  cl_mem d_probes = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY |
                                   CL_MEM_COPY_HOST_PTR,
                                   probesMemSize, (void *) GridRadii,
                                   &status);
  ocl_check_status(status);

  status = CL_SUCCESS;
  cl_mem d_GridPot = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    gridMemSize, NULL, &status);
  ocl_check_status(status);

  cl_float zero = 0.0;
  status = clEnqueueFillBuffer(queue, d_GridPot, &zero, sizeof(cl_float), 0,
                               gridMemSize, 0, NULL, NULL);
  ocl_check_status(status);

  status = clEnqueueWriteBuffer(queue, d_GridPot, CL_TRUE, 0,   
                                gridMemSize, GridPot, 0, NULL, NULL);
  ocl_check_status(status);

  cl_kernel pot_kernel;
  status = ocl_compile_kernel(Kernels::generateProtGrid, "generateProtGrid",
                              context, dev_id,
                              pot_kernel);
  if (status != CL_SUCCESS) {
    return status;
  }

  size_t
    localSize = 256,
    globalSize = ceil((float) NumGridPoints / (float) localSize) * localSize;

  status = clSetKernelArg(pot_kernel, 0, sizeof(cl_mem), (void *) &d_probes);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 1, sizeof(cl_mem), (void *) &d_parameter);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 2, sizeof(cl_mem), (void *) &d_GridPot);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 3, sizeof(int), (void *) &NumGrids);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 4, sizeof(int), (void *) &NumAtoms);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 5, sizeof(cl_mem), (void *) &d_GridNum);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 6, sizeof(cl_mem), (void *) &d_GridMinCoor);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 7, sizeof(float), (void *) &Fa);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 8, sizeof(float), (void *) &Fb);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 9, sizeof(float), (void *) &Gmax);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 10, sizeof(float), (void *) &DGrid);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 11, sizeof(float), (void *) &VdwEmax);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 12, sizeof(float), (void *) &ElecReplEmax);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 13, sizeof(float), (void *) &ElecAttrEmax);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 14, sizeof(int), (void *) &CCELEC);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 15, sizeof(int), (void *) &ElecMode);
  ocl_check_status(status);

  status = clSetKernelArg(pot_kernel, 16, sizeof(float), (void *) &Dielec);
  ocl_check_status(status);

  status = clEnqueueNDRangeKernel(queue, pot_kernel, 1, NULL,
                                  &globalSize, &localSize,
                                  0, NULL, NULL);
  ocl_check_status(status);

  status = clEnqueueReadBuffer(queue, d_GridPot, CL_TRUE, 0,
                               gridMemSize, GridPot, 0, NULL, NULL);
  ocl_check_status(status);

  // free GPU memory
  // the d_GridPot could also be reused to speedup the program
  status = clReleaseMemObject(d_parameter);
  ocl_check_status(status);

  status = clReleaseMemObject(d_GridNum);
  ocl_check_status(status);

  status = clReleaseMemObject(d_GridMinCoor);
  ocl_check_status(status);

  status = clReleaseMemObject(d_GridPot);
  ocl_check_status(status);

  status = clReleaseMemObject(d_probes);
  ocl_check_status(status);

  status = clReleaseKernel(pot_kernel);
  ocl_check_status(status);

  return status;
}
#endif /* HAS_OPENCL */
#endif /* KEY_FFTDOCK == 1 */
