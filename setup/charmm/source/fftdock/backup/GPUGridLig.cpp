#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include<math.h>

#include <kernels.h>
#include "cl_util.h"

extern "C"
void calcLigGrid(const int BatchIdx, const int BatchSize,
                 const int NumRotamers, const int NumAtoms,
                 const int NumVdwGridUsed,
                 const float DGrid,
                 const int XGridNum, const int YGridNum, const int ZGridNum,
                 float * SelectAtomsParameters,
                 float * LigGrid, float * LigRotamerCoors,
                 float * LigRotamerMinCoors, void * &d_LigGrid_F) {
  //BatchIdx starting from 1
  const int BlockDim[1];
  BlockDim[0] = 128;

  const int NumGrids = NumVdwGridUsed + 1;
  int RealBatchSize = BatchSize;
  if ((BatchIdx * BatchSize) > NumRotamers) {
    RealBatchSize = NumRotamers % BatchSize;
  }
  //printf("BatchIdx = %d RealBatchSize = %d\n",BatchIdx,RealBatchSize);
  int GridDim[1];
  GridDim[0] = ceil(float(BatchSize) / float(BlockDim));

  //generate ligand grid properties
  int GridNum[3];
  GridNum[0] = XGridNum;
  GridNum[1] = YGridNum;
  GridNum[2] = ZGridNum;

  int NumGridPoints = XGridNum * YGridNum * ZGridNum;

  cl_context context;
  cl_command_queue queue;
  cl_gpu_init(context, queue);

  //printf("GridNum x=%d y=%d z=%d\n",GridNum[0],GridNum[1],GridNum[2]);
  //printf("NumGridPoints=%d\n",NumGridPoints);

  cl_mem d_rotamersCoor = clCreateBuffer(context,
                                         CL_MEM_READ_ONLY |
                                         CL_MEM_COPY_HOST_PTR,
                                         BatchSize * NumAtoms * 3 * sizeof(float),
                                         NULL, NULL);
  cl_mem d_GridMinCoor = clCreateBuffer(context,
                                        CL_MEM_READ_ONLY |
                                        CL_MEM_COPY_HOST_PTR,
                                        BatchSize * 3 * sizeof(float),
                                        NULL, NULL);

  clEnqueueWriteBuffer(queue, d_rotamersCoor, CL_TRUE, 0,
                       BatchSize * NumAtoms * 3 * sizeof(float),
                       LigRotamerCoors, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, d_GridMinCoor, CL_TRUE, 0,
                       BatchSize * 3 * sizeof(float),
                       LigRotamerMinCoors, 0, NULL, NULL);

  //printf("GridDim=%d BlockDim=%d\n",GridDim,BlockDim);
  //printf("Finished generating rotamers\n");

  //generate LigGrid
  cl_mem d_LigGrid; =
  int LigGridSize = BatchSize * NumGrids * NumGridPoints * sizeof(float);

  //printf("Old Address=%p\n",d_LigGrid_F);

  if (BatchIdx == 1) {
    //printf("Allocate memory\n");

    d_LigGrid = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               LigGridSize, NULL, NULL);

    d_LigGrid_F = (void *) d_LigGrid;

    //printf("New Address1=%p\n",d_LigGrid);
    //printf("New Address2=%p\n",d_LigGrid_F);
  } else {
    d_LigGrid = (cl_mem) d_LigGrid_F;

    //printf("Old Address=%p\n",d_LigGrid_F);
  }

  cl_float zero = 0.0;
  clEnqueueFillBuffer(queue, d_LigGrid, &zero, sizeof(cl_float), 0,
                      LigGridSize, 0, NULL, NULL);


  // atom parameters
  cl_mem d_par = clCreateBuffer(context,
                                CL_MEM_READ_ONLY |
                                CL_MEM_COPY_HOST_PTR,
                                NumAtoms * 4 * sizeof(float), NULL, NULL);
  // grid dimensions
  cl_mem d_GridNum = clCreateBuffer(context,
                                CL_MEM_READ_ONLY |
                                CL_MEM_COPY_HOST_PTR,
                                3 * sizeof(int), NULL, NULL);

  clEnqueueWriteBuffer(queue, d_par, CL_TRUE, 0,
                       NumAtoms * 4 * sizeof(float),
                       SelectAtomsParameters, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, d_GridNum, CL_TRUE, 0,
                       3 * sizeof(int),
                       GridNum, 0, NULL, NULL);

  cl_program lig_program;
  cl_kernel lig_kernel;
  cl_compile_kernel(Kernels::generateLigGrid, "generateLigGrid",
                    context, lig_program, lig_kernel);

  clSetKernelArg(lig_kernel, 0, sizeof(int), (void *) &RealBatchSize);
  clSetKernelArg(lig_kernel, 1, sizeof(int), (void *) &NumAtoms);
  clSetKernelArg(lig_kernel, 2, sizeof(int), (void *) &NumGrids);
  clSetKernelArg(lig_kernel, 3, sizeof(cl_mem), (void *) &d_GridNum);
  clSetKernelArg(lig_kernel, 4, sizeof(float), (void *) &DGrid);
  clSetKernelArg(lig_kernel, 5, sizeof(cl_mem), (void *) &d_rotamersCoor);
  clSetKernelArg(lig_kernel, 6, sizeof(cl_mem), (void *) &d_par);
  clSetKernelArg(lig_kernel, 7, sizeof(cl_mem), (void *) &d_GridMinCoor);
  clSetKernelArg(lig_kernel, 8, sizeof(cl_mem), (void *) &d_LigGrid);

  //call cuda kernel to generate LigGrid
  clEnqueueNDRangeKernel(queue, lig_kernel, 1, NULL, GridDim, BlockDim,
                         0, NULL, NULL);

  // do not copy ligand grid back to cpu
  //cudaMemcpy(LigGrid,d_LigGrid,BatchSize*NumGrids*NumGridPoints*sizeof(float),
  //        cudaMemcpyDeviceToHost);

  clReleaseMemObject(d_par);
  clReleaseMemObject(d_rotamersCoor);
  clReleaseMemObject(d_GridMinCoor);
  clReleaseMemObject(d_GridNum);
}
