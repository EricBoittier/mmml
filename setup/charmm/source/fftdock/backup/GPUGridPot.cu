#include "stdio.h"
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()),\
   __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n in generating protein grid using GPU.\n",\
          cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 

__global__ void generateProtGrid(float *d_probes, float *d_parameter,
        float *d_GridPot, const int NGrids, const int NAtoms, int *GridNum,
        float *GridMinCoor, const float Fa ,const float Fb, const float Gmax, 
        const float DGrid, const float VdwEmax, 
        const float ElecReplEmax,const float ElecAttrEmax,
        const int CCELEC_CHARMM, const int ElecMode, const float Dielec){
    /*
       Kernel of generating protein grid
       Each thread handle a grid point
     */
    int GridGlobalId=blockIdx.x*blockDim.x+threadIdx.x;
    int NumGridPoints=GridNum[0]*GridNum[1]*GridNum[2];
    /*
    if(GridGlobalId==0){
        printf("I am thread %d\n",GridGlobalId);
        printf("NumGridPoints=%d\n",NumGridPoints);
        printf("NAtoms=%d\n",NAtoms);
    }*/
    if(GridGlobalId<NumGridPoints){
        int Gridx,Gridy,Gridz;
        int gridIdx;
        //calculate the 3-d grid x,y,z index from the 1-d global index
        //GridPotId=(Gridx*GridNum[1]+Gridy)*GridNum[2]+Gridz
        Gridx = GridGlobalId/(GridNum[1]*GridNum[2]);
        Gridy = (GridGlobalId%(GridNum[1]*GridNum[2]))/GridNum[0];
        Gridz = (GridGlobalId%(GridNum[1]*GridNum[2]))%GridNum[0];

        //calculate the grid point coordinate
        //grid point coordinate build from GridPotId
        float x,y,z;
        x = GridMinCoor[0]+Gridx*DGrid;
        y = GridMinCoor[1]+Gridy*DGrid;
        z = GridMinCoor[2]+Gridz*DGrid;
        /*
        if(GridGlobalId==0){
            printf("Grid x=%d y=%d z=%d\n",Gridx,Gridy,Gridz);
            printf("x=%f y=%f z=%f\n",x,y,z);
        }*/
        int n;
        float atomx,atomy,atomz;
        float eps,vdwr,cg;
        float eps_sqrt;
        float r_min;
        float eleconst, vdwconst;
        float r,rc,rh;
        float alpha,beta;
        float dener, aener; 
        int hd, ha;
        int NumFeaturePerAtom=8;

        for(n=0; n<NAtoms; ++n){
            //read atom parameters from global memory
            atomx=d_parameter[NumFeaturePerAtom*n+0];
            atomy=d_parameter[NumFeaturePerAtom*n+1];
            atomz=d_parameter[NumFeaturePerAtom*n+2];
            eps=  d_parameter[NumFeaturePerAtom*n+3];
            vdwr= d_parameter[NumFeaturePerAtom*n+4];
            cg=   d_parameter[NumFeaturePerAtom*n+5];
            hd=   d_parameter[NumFeaturePerAtom*n+6];
            ha=   d_parameter[NumFeaturePerAtom*n+7];
            eps_sqrt = sqrtf(fabs(eps));
            //calculate distance between grid point and atom n
            r = sqrtf((atomx-x)*(atomx-x)+(atomy-y)*(atomy-y)+(atomz-z)*(atomz-z));
            rh = r - Fb;
            //calculate electrostatic energy
            /*
            if(GridGlobalId==0){
                printf("%d x=%f y=%f z=%f eps=%f vdwr=%f cg=%f r=%f\n",n,
                atomx,atomy,atomz,eps,vdwr,cg,r);
            }*/

            /*Electrostatics*/
            //ElecMode == 0 cdie
            //ElecMode == 1 rdie
            eleconst=CCELEC_CHARMM*cg/Dielec;
            //cdie
            if(ElecMode == 0){
                if( cg > 0) { //repulsive
                    rc = 2.0*eleconst/fabs(ElecReplEmax);
                    if(r > rc){
                        d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] += 
                            eleconst/(r);
                    }else{
                        alpha = eleconst/(rc*rc);
                        d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] += 
                            (ElecReplEmax-alpha*r);
                    }
                }
                else if (cg < 0){ //attractive
                    rc = -2.0*eleconst/fabs(ElecAttrEmax);
                    if(r > rc){
                        d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId]+= 
                            eleconst/(r);
                    }else{
                        alpha = eleconst/(rc*rc);
                        d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] += 
                            (ElecAttrEmax-alpha*r);
                    }
                }
            }
            //rdie
            else if(ElecMode == 1){
                if( cg > 0) { //repulsive
                    rc = sqrtf(2.0*fabs(eleconst/ElecReplEmax));
                    if(r > rc){
                        d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] += 
                            eleconst/(r*r);
                    }else{
                        alpha = fabs(ElecReplEmax / (2.0*rc*rc));
                        d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] += 
                            (ElecReplEmax - alpha*(r*r));
                    }
                }
                else if (cg < 0){ //attractive
                    rc = sqrtf(2.0*fabs(eleconst/ElecAttrEmax));
                    if(r > rc){
                        d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId]+= 
                            eleconst/(r*r);
                    }else{
                        alpha = fabs(ElecAttrEmax / (2.0*rc*rc));
                        d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] += 
                            (ElecAttrEmax + alpha*(r*r));
                    }
                }
            }

            /*Hydrogen donor grid*/
            gridIdx = NGrids - 3; 
            dener = ((rh * rh) * Fa + Gmax) * hd; 
            if (dener < 0) {
                d_GridPot[NumGridPoints*gridIdx+GridGlobalId] += 
                    dener; 
            } else{
                d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
                    0;
            }

            /*Hydrogen acceptor grid*/
            gridIdx = NGrids - 2;
            aener = ((rh * rh) * Fa + Gmax) * ha; 
            if (aener < 0) {
                d_GridPot[NumGridPoints*gridIdx+GridGlobalId] += 
                    aener; 
            } else{
                d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
                    0;
            }

            /*Van der waals*/
            for(gridIdx=0; gridIdx<NGrids-3; ++gridIdx){
                float radii = d_probes[gridIdx];
                r_min = vdwr + radii;
                vdwconst = 1.0 + sqrtf(1.0 + 0.5*fabs(VdwEmax)/eps_sqrt);
                rc = r_min * powf(vdwconst,-1.0/6.0);
                beta = 24.0 * eps_sqrt/VdwEmax * (vdwconst*vdwconst-2*vdwconst);
                alpha = 0.5*VdwEmax*powf(rc,-1.0*beta);
                if (r > rc) {
                    d_GridPot[NumGridPoints*gridIdx+GridGlobalId] += 
                        eps_sqrt*(powf(r_min/r,12.0) - 2.0*powf( r_min/r, 6.0));
                } else{
                    d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
                        VdwEmax - alpha*powf(r, beta);
                }
            }

        } //end looping all atoms
    }//enf if (GlobalId <  NumGridPoints)
}
    
extern "C"
void calcPotGrid(const int NumGrids, const int NumAtoms, 
        const float DGrid, const int XGridLen, const int YGridLen, 
        const int ZGridLen, const float XMin, const float YMin, const float ZMin,
        const float Fa ,const float Fb, const float Gmax, 
        const float VdwEmax, const float ElecAttrEmax,
        const float ElecReplEmax, const float CCELEC, const int ElecMode, 
        const float Dielec, float *SelectAtomsParameters, float *GridPot, 
        float *GridRadii)
{
    /*
    FILE *fp;
    fp = fopen("param.dat","w");
    for(int ii=0; ii<NumAtoms; ++ii){
        float x,y,z,eps,vdwr,cg;
        x=SelectAtomsParameters[6*ii+0];
        y=SelectAtomsParameters[6*ii+1];
        z=SelectAtomsParameters[6*ii+2];
        eps=SelectAtomsParameters[6*ii+3];
        vdwr=SelectAtomsParameters[6*ii+4];
        cg=SelectAtomsParameters[6*ii+5];
        fprintf(fp,"%f %f %f %f %f %f\n",x,y,z,eps,vdwr,cg);
    }
    fclose(fp);
    */

    const int NUM_PARAMS_PER_ATOM = 8;

    //the coordinates of the grid center

    //generate grid properties
    //number of grid points in three dimensions
    int GridNum[3]; 
    int NumGridPoints = 1; //total number of grid points
    float GridMinCoor[3]; //the coordinate of the origin of the grid
    GridNum[0] = XGridLen;
    GridNum[1] = YGridLen;
    GridNum[2] = ZGridLen;
    GridMinCoor[0] = XMin; 
    GridMinCoor[1] = YMin;
    GridMinCoor[2] = ZMin;
    NumGridPoints = GridNum[0]*GridNum[1]*GridNum[2];

    //allocate GPU memory
    int paramMemSize = NumAtoms*NUM_PARAMS_PER_ATOM*sizeof(float);
    int gridMemSize = NumGridPoints*NumGrids*sizeof(float);
    int probesMemSize = (NumGrids-3)*sizeof(float);
    float* d_parameter; //gpu parameters
    float* d_GridPot; //gpu grid
    float* d_probes; //probes radii
    int* d_GridNum; //gpu GridNum
    float* d_GridMinCoor; //gpu GridMinCoor
    cudaMalloc(&d_parameter,paramMemSize);
    cudaMalloc(&d_GridNum,3*sizeof(int));
    cudaMalloc(&d_GridMinCoor,3*sizeof(float));
    cudaMalloc(&d_GridPot,gridMemSize);
    cudaMalloc(&d_probes,probesMemSize);
    CUDA_CHECK()

    //copy datato GPU memory
    cudaMemcpy(d_parameter,SelectAtomsParameters,paramMemSize,
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_probes,GridRadii,probesMemSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_GridNum,GridNum,3*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_GridMinCoor,GridMinCoor,3*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_GridPot, 0, gridMemSize);
    CUDA_CHECK()

    //calculate grid and block size
    int BlockDim=512;
    int GridDim=ceil(float(NumGridPoints)/float(BlockDim));
    //printf("GridDim=%d\n",GridDim);

    //do calculations on GPU
    generateProtGrid<<<GridDim,BlockDim>>>(d_probes, d_parameter, d_GridPot,
            NumGrids, NumAtoms,d_GridNum,d_GridMinCoor,Fa,Fb,Gmax,DGrid,VdwEmax,ElecReplEmax,
            ElecAttrEmax,CCELEC,ElecMode,Dielec);
    CUDA_CHECK()
    cudaMemcpy(GridPot,d_GridPot,gridMemSize,cudaMemcpyDeviceToHost);
    CUDA_CHECK()
    //free GPU memory
    //the d_GridPot could also be reused to speedup the program
    cudaFree(d_parameter);
    cudaFree(d_GridNum);
    cudaFree(d_GridMinCoor);
    cudaFree(d_GridPot);
    cudaFree(d_probes);
    CUDA_CHECK()
    //printf("%f\n",GridPot[(NumGrids-3)*NumGridPoints]);
}
