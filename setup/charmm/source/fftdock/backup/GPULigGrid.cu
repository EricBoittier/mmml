#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n in generating ligand grids using GPU\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 

//generate ligand grid
__global__ void d_generateLigGrid(int numRotamers,int NAtoms, int numGrids, int *d_GridNum, float DGrid, float *d_rotamersCoor,float *d_par,float *d_GridMinCoor, float *d_LigGrid){
    int globalId=blockIdx.x*blockDim.x+threadIdx.x;
    if(globalId<numRotamers){
        //printf("GlobalId=%d NumRotamers=%d\n",globalId,numRotamers);
        float dx,dy,dz;
        int i,j,grid_idx;
        int idx_x,idx_y,idx_z, vdw_grid_idx;
        int xlen,ylen,zlen;
        float xRatio,yRatio,zRatio;
        float eps,vdwr,charge;
        float energyFactor;
        unsigned int NumGridPoints = d_GridNum[0]*d_GridNum[1]*d_GridNum[2];
        //offset used to locate the right position to write in the LigGrid for each type of grid
        unsigned int rotamerOffset,gridTypeOffset;
        //printf("grid x=%d y=%d z=%d\n",d_GridNum[0],d_GridNum[1],d_GridNum[2]);
        xlen = d_GridNum[0];
        ylen = d_GridNum[1];
        zlen = d_GridNum[2];
        rotamerOffset=globalId*numGrids*NumGridPoints;
        for(i=0;i<NAtoms;++i){
            dx = d_rotamersCoor[globalId*3*NAtoms + 3*i + 0]-
                d_GridMinCoor[globalId*3 + 0];
            dy = d_rotamersCoor[globalId*3*NAtoms + 3*i + 1]-
                d_GridMinCoor[globalId*3 + 1];
            dz = d_rotamersCoor[globalId*3*NAtoms + 3*i + 2]-
                d_GridMinCoor[globalId*3 + 2];

            //calculate the atomi 3-d index
            idx_x = __float2int_rd(dx/DGrid);
            idx_y = __float2int_rd(dy/DGrid);
            idx_z = __float2int_rd(dz/DGrid);

            xRatio = (dx-(idx_x)*DGrid)/DGrid;
            yRatio = (dy-(idx_y)*DGrid)/DGrid;
            zRatio = (dz-(idx_z)*DGrid)/DGrid;

            charge = d_par[i*4+0];
            eps    = d_par[i*4+1];
            vdwr   = d_par[i*4+2];
            vdw_grid_idx = d_par[i*4+3];

            /*
            if(globalId == 40 or globalId==0){
                printf("idx=%d atom%d dx=%f dy=%f dz=%f idx_x=%d idx_y=%d idx_z=%d\n",globalId,
                        i,d_rotamersCoor[globalId*3*NAtoms+3*i+0]
                        ,d_rotamersCoor[globalId*3*NAtoms+3*i+1],
                        d_rotamersCoor[globalId*3*NAtoms+3*i+2],idx_x,idx_y,idx_z);
                //printf("charge=%f eps=%f vdwr=%f vdw_grid_idx=%d\n",charge,eps,
                //        vdwr,vdw_grid_idx);
            }*/
            //vdw

            //loop through elec,vdwrepl,vdwattr grids
            energyFactor = 0.0;
            for(j=0; j<numGrids; ++j){
                if(j == vdw_grid_idx){
                    energyFactor = sqrtf(fabs(eps));
                }
                else if(j == numGrids - 1){
                    energyFactor = charge;
                }
                else{
                    energyFactor = 0.0;
                }
                /*
                if(globalId == 38){
                    printf("atom%d energyFactor[%d]==%f vdw_grid_idx=%d\n",i,j,energyFactor,
                            vdw_grid_idx);
                    printf("offset=%d\n",rotamerOffset+gridTypeOffset);
                }*/
                gridTypeOffset=j*NumGridPoints;
                //if (globalId==1){
                    //printf("grid%d offset=%d numGridPoints=%d\n",j,rotamerOffset+gridTypeOffset,NumGridPoints);
                //}
                //(0,0,0)
                d_LigGrid[rotamerOffset+gridTypeOffset+
                    (idx_x*ylen+idx_y)*zlen+idx_z]+=
                    (1-xRatio)*(1-yRatio)*(1-zRatio)*energyFactor;
                //(0,0,1)
                d_LigGrid[rotamerOffset+gridTypeOffset+
                    (idx_x*ylen+idx_y)*zlen+idx_z+1]+=
                    (1-xRatio)*(1-yRatio)*(zRatio)*energyFactor;
                //(0,1,0)
                d_LigGrid[rotamerOffset+gridTypeOffset+
                    (idx_x*ylen+idx_y+1)*zlen+idx_z]+=
                    (1-xRatio)*yRatio*(1-zRatio)*energyFactor;
                //(0,1,1)
                d_LigGrid[rotamerOffset+gridTypeOffset+
                    (idx_x*ylen+idx_y+1)*zlen+idx_z+1]+=
                    (1-xRatio)*yRatio*zRatio*energyFactor;
                //(1,0,0)
                d_LigGrid[rotamerOffset+gridTypeOffset+
                    ((idx_x+1)*ylen+idx_y)*zlen+idx_z]+=
                    xRatio*(1-yRatio)*(1-zRatio)*energyFactor;
                //(1,0,1)
                d_LigGrid[rotamerOffset+gridTypeOffset+
                    ((idx_x+1)*ylen+idx_y)*zlen+idx_z+1]+=
                    xRatio*(1-yRatio)*zRatio*energyFactor;
                //(1,1,0)
                d_LigGrid[rotamerOffset+gridTypeOffset+
                    ((idx_x+1)*ylen+idx_y+1)*zlen+idx_z]+=
                    xRatio*yRatio*(1-zRatio)*energyFactor;
                //(1,1,1)
                d_LigGrid[rotamerOffset+gridTypeOffset+
                    ((idx_x+1)*ylen+idx_y+1)*zlen+idx_z+1]+=
                    xRatio*yRatio*zRatio*energyFactor;
            }
        }
    }
}

//SUBROUTINE calcLigGrid(BatchIdx, BatchSize, NumQuaterions, NumAtoms, NumVdwGridUsed, DGrid, XGridNum, YGridNum, &
//        ZGridNum,SelectAtomsParameters, LigGrid, LigRotamerCoors, &
//        LigRotamerMinCoors, LigRotamerMaxCoors) bind(c,name='calcLigGrid')
extern "C"
void calcLigGrid(const int BatchIdx, const int BatchSize, const int NumRotamers, const int NumAtoms,
        const int NumVdwGridUsed,
        const float DGrid, const int XGridNum, const int YGridNum, const int ZGridNum,
        float *SelectAtomsParameters, float *LigGrid, float *LigRotamerCoors,
        float *LigRotamerMinCoors, void* &d_LigGrid_F)
{
    //BatchIdx starting from 1
    const int BlockDim=128;
    const int NumGrids = NumVdwGridUsed + 1;
    int RealBatchSize = BatchSize;
    if((BatchIdx*BatchSize > NumRotamers)){
        RealBatchSize = NumRotamers%BatchSize;
    }
    //printf("BatchIdx = %d RealBatchSize = %d\n",BatchIdx,RealBatchSize);
    int GridDim=ceil(float(BatchSize)/float(BlockDim));

    //generate ligand grid properties
    int NumGridPoints = 1;
    int GridNum[3];
    GridNum[0] = XGridNum;
    GridNum[1] = YGridNum;
    GridNum[2] = ZGridNum;
    NumGridPoints = XGridNum*YGridNum*ZGridNum;
    //printf("GridNum x=%d y=%d z=%d\n",GridNum[0],GridNum[1],GridNum[2]);
    //printf("NumGridPoints=%d\n",NumGridPoints);

    float *d_rotamersCoor;
    float *d_GridMinCoor;
    cudaMalloc(&d_rotamersCoor,BatchSize*NumAtoms*3*sizeof(float));
    cudaMalloc(&d_GridMinCoor,BatchSize*3*sizeof(float));
    CUDA_CHECK();
    cudaMemcpy(d_rotamersCoor,LigRotamerCoors,BatchSize*NumAtoms*3*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_GridMinCoor,LigRotamerMinCoors,BatchSize*3*sizeof(float),cudaMemcpyHostToDevice);
    CUDA_CHECK();
    //printf("GridDim=%d BlockDim=%d\n",GridDim,BlockDim);
    //printf("Finished generating rotamers\n");
    //generate LigGrid
    float *d_LigGrid;
    int LigGridSize = BatchSize*NumGrids*NumGridPoints*sizeof(float);
    //printf("Old Address=%p\n",d_LigGrid_F);
    if(BatchIdx==1){
        //printf("Allocate memory\n");
        cudaMalloc(&d_LigGrid, LigGridSize);
        d_LigGrid_F = (void*)d_LigGrid;
        //printf("New Address1=%p\n",d_LigGrid);
        //printf("New Address2=%p\n",d_LigGrid_F);
        CUDA_CHECK();
    }else{
        d_LigGrid = (float*)(d_LigGrid_F);
        //printf("Old Address=%p\n",d_LigGrid_F);
    }

    cudaMemset(d_LigGrid, 0.0, LigGridSize);
    CUDA_CHECK();

    //atoms parameters
    float *d_par;
    cudaMalloc(&d_par,NumAtoms*4*sizeof(float));
    cudaMemcpy(d_par,SelectAtomsParameters,NumAtoms*4*sizeof(float),cudaMemcpyHostToDevice);
    CUDA_CHECK();

    //grid dimension
    int *d_GridNum;
    cudaMalloc(&d_GridNum,3*sizeof(int));
    cudaMemcpy(d_GridNum,GridNum,3*sizeof(int),cudaMemcpyHostToDevice);
    CUDA_CHECK();

    //call cuda kernel to generate LigGrid
    d_generateLigGrid<<<GridDim,BlockDim>>>(RealBatchSize,NumAtoms, NumGrids, d_GridNum,
            DGrid,d_rotamersCoor,d_par,d_GridMinCoor,d_LigGrid);
    CUDA_CHECK();
    // do not copy ligand grid back to cpu
    //cudaMemcpy(LigGrid,d_LigGrid,BatchSize*NumGrids*NumGridPoints*sizeof(float),
    //        cudaMemcpyDeviceToHost);
    cudaFree(d_par);
    cudaFree(d_rotamersCoor);
    cudaFree(d_GridMinCoor);
    cudaFree(d_GridNum);
    CUDA_CHECK();
}
