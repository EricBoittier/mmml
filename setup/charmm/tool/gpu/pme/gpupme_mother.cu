#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <cufft.h>
/* #include "timer.h" */

const int MAX_BSIZE = 65535;

#define FLG_TIMER (0) // when FLG_TIMER=1, measure profile. Added by Ryuji on Nov. 14, 2011.
#define FLG_STRESS_SAKAMAKI (1) // compute stress tensor. Added by Ryuji on Oct. 29, 2010.

/*
  VG_DEVICEID ---------- Device ID of GPU
  VG_DEVICEID_WAVE ----- Device ID of GPU. If VG_DEVICEID is defined, VG_DEVICEID_WAVE is used.
  VG_NOCELLINDEX_WAVE -- Cellindex is not used for wavepart. 
  VG_SNAPSHOT ---------- Save snapshot
  VG_SNAPSHOT_COMPARE -- Compare the result with snapshot file
                         This is stronger than VG_SNAPSHOT
 */

//#define PRINT_WARN
#define SNAPSHOT         // save snapshot to file when VG_SNAPSHOT is defined
#define FIXEDPOINT 1000000.f // if this is not defined, float is used for atomic add
#define NSEG 256 // number of segmentation for interpolation
#define CELLINDEX // use cell-index for fill charge grid part. 
//#define MULTIPLE_IBLOCK // one block treats all the particles in a cell
                        // instread of BSIZEPME1 particles.
                        // not work
#ifdef CELLINDEX
#define MULTIPLE_IBLOCK_TEST // one block treats all the particles in a cell
#define MULTIPLE_IBLOCK_TEST2 // reduce the number of threads. MULTIPLE_IBLOCK_TEST must be defined.
                              // this is practically not needed because there is little difference
                              // between ..IBLOCK_TEST.
#endif

#include <gpuheader.h>
#include <gpupme_kernel.cu>

#if 0 // error check
#define CHECK_ERROR(x) cudaThreadSynchronize();CudaError(x,1);
#else
#define CHECK_ERROR(x) 
#endif

#ifdef CELLINDEX
typedef struct {
  int base;
  int size;
} VG_CELL;
#ifdef MR3_MALLOC
#define MR3_malloc_pointer(x,y) MR3_my_malloc2(x,y)
#define MR3_free_pointer(x,y)   MR3_my_free2((void **)(&(x)),y)
#else
#define MR3_malloc_pointer(x,y) malloc(x)
#define MR3_free_pointer(x,y)   free(x)
#endif

#define MAX(a,b) ((a)>(b)? (a):(b))

static void assign_cell(int n, double x_org[],
                        int size_xyz[3],
			int ldimtotal[3], float recipf[9],
			int blocksize, int order,
			int (*cip)[4], VG_CELL *ci,
			int *cinthp)
{
  /* 
    input
     blocksize ---  1 means no dummy particle
                   >1 means some dummy particles in each cell
    output
     cip[i][0,1,2] -- index of a cell (cx,cy,cz) in which a paricle-i is.
     cip[i][3] ------ cellindex in which a particle-i is.
     ci[ci].base ---- index of the first particle in a cell-ci.
                      this becomes a multiple of a blocksize.
     ci[ci].size ---- number of particles in a cell-ci.
                      this does not become a multiple of a blocksize.
     cinthp[i] ------ offset of particles in the cell which a particle-i is.
                      particle-i is the cinthp[i]-th particle in a cell.
  */

  int i,j;
  int cid[3],cit,ncell;
  float *x;
  int gridperblockxyz[3]={size_xyz[0]/ldimtotal[0],
                          size_xyz[1]/ldimtotal[1],
                          size_xyz[2]/ldimtotal[2]};

  ncell=ldimtotal[0]*ldimtotal[1]*ldimtotal[2];

  /* convert particle position to [0,volume) coordinate */
  if((x=(float *)MR3_malloc_pointer(sizeof(float)*n*3,"x in assign_cell"))==NULL){
    fprintf(stderr,"** error : can't malloc x in assign_cell **\n");
    exit(1);
  }
  for(i=0;i<n;i++){
    float xf[3];
    for(j=0;j<3;j++) xf[j]=x_org[i*3+j];
    for(j=0;j<3;j++){
      x[i*3+j] = xf[0]*recipf[j*3]+xf[1]*recipf[j*3+1]+xf[2]*recipf[j*3+2];
    }
  }

  /* assign a particle into a cell */
  for(i=0;i<ncell;i++) ci[i].size=0;
  for(i=0;i<n;i++){
    float ui[3];
    for(j=0;j<3;j++){
      ui[j]=x[i*3+j];
#if 1
      while(ui[j]<0.0f)  ui[j]+=1.0f;
      while(ui[j]>=1.0f) ui[j]-=1.0f;
#endif                             
      ui[j] *= (float)size_xyz[j];
      cid[j] = (int)ui[j];
      ui[j] = - ui[j] + (float)cid[j];
      cid[j] += size_xyz[j] - order;
      cid[j] += 1;
      while(cid[j]<0){
	cid[j]+=size_xyz[j];
      }
      while(cid[j]>=size_xyz[j]){
	cid[j]-=size_xyz[j];
      }
      cid[j]/=gridperblockxyz[j];
    }
    cit=(cid[2]*ldimtotal[1]+cid[1])*ldimtotal[0]+cid[0];
    cip[i][3]=cit;
    for(j=0;j<3;j++) cip[i][j]=cid[j];
    cinthp[i]=ci[cit].size;
    ci[cit].size++;
    //    if((i % blocksize)==0) printf("cip[%d]=(%d,%d,%d) cinthp=%d\n",i,cip[i][0],cip[i][1],cip[i][2],cinthp[i]);
  }

  /* make cell index list */
  ci[0].base=0;
  for(i=1;i<ncell;i++){
    //    ci[i].base=ci[i-1].base+ci[i-1].size;
    ci[i].base=ci[i-1].base+(ci[i-1].size+blocksize-1)/blocksize*blocksize;
  }

  /* free */
  MR3_free_pointer(x,"x in assign_cell");

#if 0
  for(i=0;i<ncell;i++){
    printf("ci[%d].base=%d size=%d\n",i,ci[i].base,ci[i].size);
  }
#endif  
}


static void make_contiguous_xq(int n, double *x, double *q,
			       int (*cip)[4], VG_CELL *ci, int *cinthp,
			       float4 *xq)
{
  int i,cellindex,xcellpointer;

  for(i=0;i<n;i++){
    cellindex=cip[i][3];
    xcellpointer=ci[cellindex].base+cinthp[i];
    xq[xcellpointer].x=x[i*3];
    xq[xcellpointer].y=x[i*3+1];
    xq[xcellpointer].z=x[i*3+2];
    xq[xcellpointer].w=q[i];
  }
}


static void make_cellindex(int natm, double *xi, double *qi,
			   int size_x, int size_y, int size_z, int order, 
			   float recipf[9], int blocksize,
			   int *ni2_ret, float4 **xq, int (**cellpositionofblock)[4],
			   int ldim[3], double flag)
{
  /*
    flag 0.0 : malloc and do cellindex operation
         1   : natm==0 : free
               natm!=0 : copy back forces from xq to xi and qi
   */
  static int (*cip)[4]=NULL,*cinthp=NULL;
  static VG_CELL *ci=NULL;
  int ncell;
  int ni=natm;
  int i,j,k,ni2,ib,ip,size_xyz[3]={size_x,size_y,size_z};
#ifdef MULTIPLE_IBLOCK
  int ibm;
#endif

  if(flag==0.0){
    //    if(size_x==64 && size_y==64 && size_z==64){
    for(i=0;i<3;i++){
      ldim[i]=0;
      // this should match QS_SIZEX,Y,Z-order
      for(j=order+2;j<=8;j++){// this work 
      //      for(j=QS_SIZEX-order;j>=6;j--){// 
      //      for(j=order;j<=QS_SIZEX-order;j++){// 
	//      for(j=order+1;j<=9;j++){// not work for j=9, nfft=108
	if((size_xyz[i] % (j*2))==0){
	  ldim[i]=size_xyz[i]/j;
	  //	  printf("** info : j=%d is used for xyz=%d\n",j,i);
	  break;
	}
      }
      if(ldim[i]==0){
	fprintf(stderr,"** error : not supported grid size_xyz[%d]=%d **\n",
		i,size_xyz[i]);
	exit(1);
      }
    }
    //    if((size_x % 8)==0 && (size_y % 8)==0 && (size_z % 8)==0){
    //      ldim[0]=size_x/8;
    //      ldim[1]=size_y/8;
    //      ldim[2]=size_z/8;
    //    }
    //    else if((size_x % 6)==0 && (size_y % 6)==0 && (size_z % 6)==0){
    //      ldim[0]=size_x/6;
    //      ldim[1]=size_y/6;
    //      ldim[2]=size_z/6;
    //    }
    //    else{
    //      fprintf(stderr,"** error : not supported size=(%d,%d,%d) **\n",
    //	      size_x,size_y,size_z);
    //      exit(1);
    //    }
    
    // assign cell
    ncell=ldim[0]*ldim[1]*ldim[2];
    if((cip=(int (*)[4])MR3_malloc_pointer(sizeof(int)*4*ni,"MR3calccoulomb_ij_ci"))==NULL){
      fprintf(stderr,"** error : can't malloc cip in MR3calccoulomb_ci **\n");
      exit(1);
    }
    if((ci=(VG_CELL *)MR3_malloc_pointer(sizeof(VG_CELL)*ncell,"MR3calccoulomb_ij_ci"))==NULL){
      fprintf(stderr,"** error : can't malloc ci in MR3calccoulomb_ci **\n");
      exit(1);
    }
    if((cinthp=(int *)MR3_malloc_pointer(sizeof(int)*ni,"MR3calccoulomb_ij_ci"))==NULL){
      fprintf(stderr,"** error : can't malloc cinthp in MR3calccoulomb_ci **\n");
      exit(1);
    }
    assign_cell(ni,xi,size_xyz,ldim,recipf,blocksize,order,cip,ci,cinthp);
#if 0
    // duplicate call for timing
    assign_cell(ni,xi,size_xyz,ldim,recipf,blocksize,order,cip,ci,cinthp);
#endif
    
    // copy to contiguous array
    for(i=ni2=0;i<ncell;i++) ni2+=(ci[i].size+blocksize-1)/blocksize*blocksize;
#ifdef PRINT_WARN
    {
      static int ini=0;
      if(ini==0){
	printf("natm=%d ni2=%d in make_cellindex\n",natm,ni2);
	ini=1;
      }
    }
#endif
    if((*xq=(float4 *)MR3_malloc_pointer(sizeof(float4)*ni2,"MR3calccoulomb_ij_ci"))==NULL){
      fprintf(stderr,"** error : can't malloc *xq in MR3calccoulomb_ci **\n");
      exit(1);
    }
    bzero(*xq,sizeof(float4)*ni2);
    make_contiguous_xq(ni,xi,qi,cip,ci,cinthp,*xq);

    if((*cellpositionofblock=(int (*)[4])MR3_malloc_pointer(sizeof(int)*4*ni2/blocksize,"MR3calccoulomb_ij_ci"))==NULL){
      fprintf(stderr,"** error : can't malloc *cellpositionofblock in MR3calccoulomb_ci **\n");
      exit(1);
    }
#if defined(MULTIPLE_IBLOCK_TEST2) && 0
    ib=0;
    for(i=0;i<ldim[0]*ldim[1]*ldim[2];i++){
      int ncc=(ci[i].size+blocksize-1)/blocksize;
      (*cellpositionofblock)[i][0]=ib<<8;
      (*cellpositionofblock)[i][1]=ncc<<8;
      if(i==501) printf("i=%d size=%d ib=%d ncc=%d\n",i,ci[i].size,ib,ncc);
      ib+=ncc;
    }
#endif
    ib=0;
#ifdef MULTIPLE_IBLOCK
    ibm=0;
#endif
    for(int iz=0;iz<ldim[2];iz++){
      for(int iy=0;iy<ldim[1];iy++){
	for(int ix=0;ix<ldim[0];ix++){
	  i=ix+ldim[0]*(iy+ldim[1]*iz);
	  //	  printf("ix,y,z=%d %d %d, i=%d ci[%d].size=%d\n",ix,iy,iz,i,i,ci[i].size);
#ifdef MULTIPLE_IBLOCK
	  (*cellpositionofblock)[ibm][0]=(ix & 0xff) | ib<<8;
	  for(j=0;j<ci[i].size;j+=blocksize) ib++;
	  (*cellpositionofblock)[ibm][1]=(iy & 0xff) | ib<<8;
	  (*cellpositionofblock)[ibm][2]=iz & 0xff;
	  (*cellpositionofblock)[ibm][3]=(ix % 2)+(iy % 2)*2+(iz % 2)*4;
	  ibm++;
#else
	  for(j=0;j<ci[i].size;j+=blocksize){
#if defined(MULTIPLE_IBLOCK_TEST2) && 0
	    if(ib<ldim[0]*ldim[1]*ldim[2]){
	      (*cellpositionofblock)[ib][0]|=ix;
	      (*cellpositionofblock)[ib][1]|=iy;
	      if(ib==501) printf("a ib=%d ix=%d iy=%d\n",ib,ix,iy);
	    }
	    else{
	      (*cellpositionofblock)[ib][0]=ix;
	      (*cellpositionofblock)[ib][1]=iy;
	      if(ib==501) printf("b ib=%d ix=%d iy=%d\n",ib,ix,iy);
	    }
#else
	    (*cellpositionofblock)[ib][0]=ix;
	    (*cellpositionofblock)[ib][1]=iy;
#endif
	    (*cellpositionofblock)[ib][2]=iz;
	    if(j==0){
	    //	    if(j+blocksize<=ci[i].size){
	    //	    if(j==0 && ib==404){  // ib<405 causes error
	      //	  if(ib==7){
	    //	    if(0){
	      (*cellpositionofblock)[ib][3]=(ix % 2)+(iy % 2)*2+(iz % 2)*4;
#ifdef MULTIPLE_IBLOCK_TEST
	      (*cellpositionofblock)[ib][3]|=(ci[i].size+blocksize-1)/blocksize*16;
#endif
#if defined(MULTIPLE_IBLOCK_TEST2) && 0
	      (*cellpositionofblock)[i][0]|=ib<<8;
#endif
	    }
	    else{
	      (*cellpositionofblock)[ib][3]=8;
	    }
	    //	    printf("  ix,y,z=%d %d %d, i=%d ib=%d\n",ix,iy,iz,i,ib);
	    ib++;
	  }
#endif
	}
      }
    }
#if defined(MULTIPLE_IBLOCK_TEST2) && 0
    ib=0;
    for(i=0;i<ldim[0]*ldim[1]*ldim[2];i++){
      int ncc=(ci[i].size+blocksize-1)/blocksize;
      int ncc2=((*cellpositionofblock)[i][1]>>8) & 0xff;
      printf("i=%d size=%d ib=%d ncc=%d ncc2=%d\n",i,ci[i].size,ib,ncc,ncc2);
      ib+=ncc;
    }
#endif
    *ni2_ret=ni2;
  }
  else{
    if(natm==0){
      free(cip);cip=NULL;
      free(ci);ci=NULL;
      free(cinthp);cinthp=NULL;
    }
    else{
      for (int i = 0; i < natm; i++){
#if 1 // cellindex conversion
	int cellindex,ioffset;
	cellindex=cip[i][3];
	ioffset=ci[cellindex].base+cinthp[i];
	xi[i*3+0] += (double)(*xq)[ioffset].x * flag;
	xi[i*3+1] += (double)(*xq)[ioffset].y * flag;
	xi[i*3+2] += (double)(*xq)[ioffset].z * flag;
	qi[i]     += (double)(*xq)[ioffset].w * flag;
#else
	xi[i*3+0] += (double)(*xq)[i].x * flag;
	xi[i*3+1] += (double)(*xq)[i].y * flag;
	xi[i*3+2] += (double)(*xq)[i].z * flag;
	qi[i]     += (double)(*xq)[i].w * flag;
#endif
      }
    }
  }
}
#endif


static int CudaError(char *message, int flag)
{
  /*
    return 0 -- no error
           1 -- error
    flag   0 -- do not display message
           1 -- display message
   */
  int ret=0;
  cudaError_t err;
  
  err=cudaGetLastError();
  if(cudaSuccess!=err){
    if(flag) fprintf(stderr,"** CUDA error occurred : %s **\n",message);
    ret=1;
  }
  else{
    if(flag) fprintf(stderr,"%s\n",message);
  }
  return ret;
}

#if FLG_STRESS_SAKAMAKI
extern "C"
void gpupme   (int, double*, double*, double*, double*, int, int, int, double, double*, double*, double*, double, double*, double, int, int, double*);
extern "C"
void gpupme_  (int, double*, double*, double*, double*, int, int, int, double, double*, double*, double*, double, double*, double, int, int, double*);
void gpupme__ (int, double*, double*, double*, double*, int, int, int, double, double*, double*, double*, double, double*, double, int, int, double*);
#else // FLG_STRESS_SAKAMAKI
extern "C"
void gpupme   (int, double*, double*, double*, double*, int, int, int, double, double*, double*, double*, double, double*, double, int, int);
extern "C"
void gpupme_  (int, double*, double*, double*, double*, int, int, int, double, double*, double*, double*, double, double*, double, int, int);
void gpupme__ (int, double*, double*, double*, double*, int, int, int, double, double*, double*, double*, double, double*, double, int, int);
#endif // FLG_STRESS_SAKAMAKI
////////////////////////////////////////////////////////////////////////////////
// PME FOR GPU
////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpupme
#if FLG_STRESS_SAKAMAKI
(int natm, double* x, double* q, double* f, double* p,
 int size_x, int size_y, int size_z, double alpha,
 double* bsp_mod1, double* bsp_mod2, double* bsp_mod3,
 double volume, double* recip, double coef, int order, int idevice, double* stress9)
#else // FLG_STRESS_SAKAMAKI
(int natm, double* x, double* q, double* f, double* p,
 int size_x, int size_y, int size_z, double alpha,
 double* bsp_mod1, double* bsp_mod2, double* bsp_mod3,
 double volume, double* recip, double coef, int order, int idevice)
#endif // FLG_STRESS_SAKAMAKI
{
//  Timer timer[10];
#if FLG_TIMER
  timer[0].start("all");
  timer[1].start("preprocessing");
#endif // FLG_TIMER
#ifdef CELLINDEX
  static int iniwave=0;
  int newroutineflag=0;
  if(iniwave==0){
    char *s=getenv("VG_NOCELLINDEX_WAVE");
    if(s!=NULL){
      printf("** VG_NOCELLINDEX_WAVE is set, cellindex is not used for wavepart **\n");
      iniwave=-1;// cellindex is not used even when it is possible
    }
    else{
      iniwave=1;
    }
  }
  if(iniwave==1 && 
     order<=6 &&
     //     order<=6 && order>=6 &&
     //     order<=8 && 
     (((size_x % 16)==0 && (size_y % 16)==0 && (size_z % 16)==0)
     //     || ((size_x % 12)==0 && (size_y % 12)==0 && (size_z % 12)==0)
#if 1 // modified by Ryuji in Nov. 2011
      && ((double)size_x*size_y*size_z <= (double)128*128*128)
#endif
				)){
    newroutineflag=1;
    //    newroutineflag=0;fprintf(stderr,"** newroutine flag is set 0 **\n");
  }
  else{
    static int ini=0;
    if(ini==0){
      if ((double)size_x*size_y*size_z > (double)384*384*384){
	fprintf(stderr,"** warning : GPU may run out of memory because grid size is too large \n");
      }

      char message[]="** warning : optimized routine for GPU wavepart is not used because number of FFT grids is not multiple of 16 or order is larger than 6 **\n";
      fprintf(stderr,"%s",message);
      printf("%s",message);
      ini=1;
    }
  }
#if 0
  //  if(order!=6 || size_x!=64 || size_y!=64 || size_z!=64){
  if(order>8){
    //    fprintf(stderr,"** error : order>8, fftx,y,z=64 **\n");
    fprintf(stderr,"** error : order>8 **\n");
    // fillchgrid 14*14*14 array
    exit(1);
  }
#endif
#endif // end of CELLINDEX
#if 1
  static int ini=-1;
  if(ini==-1){
    if(idevice<0){
      char *s,*sw;
      if((sw=getenv("VG_DEVICEID_WAVE"))!=NULL){
	sscanf(sw,"%d",&idevice);
	ini=idevice;
	printf("VG_DEVICEID_WAVE is set %d in gpupme_\n",ini);
      }
      else if((s=getenv("VG_DEVICEID"))!=NULL){
	sscanf(s,"%d",&idevice);
	ini=idevice;
	printf("VG_DEVICEID is set %d in gpupme_\n",ini);
      }
      else{
	ini=0;
	printf("VG_DEVICEID or VG_DEVICEID_WAVE is not set, device=%d is used in gpupme_\n",ini);
      }
    }
    else{
      printf("device is %d in gpupme_\n",idevice);
      ini=0;
    }
  }
  if(idevice<0){
    //    printf("cudaSetDevice=%d in gpupme_mother\n",ini);
    /* CUDA_SAFE_CALL(cudaSetDevice(ini)); */
    CHECK_ERROR("after cudaSetDevice");  
  }
  else{
    //    printf("cudaSetDevice=%d in gpupme_mother\n",idevice);
    /* CUDA_SAFE_CALL(cudaSetDevice(idevice)); */
    CHECK_ERROR("after cudaSetDevice");  
  }
  //  CudaError("after cudaSetDevice",0);
#endif

////////////////////////////////////////////////////////////////////////////////
// Initialize arrays
////////////////////////////////////////////////////////////////////////////////
//  int n = ((int)((float)(natm-1)/(float)BSIZEPME2)+1)*BSIZEPME2;
  int n = (natm+BSIZEPME2-1)/BSIZEPME2*BSIZEPME2;
  float factor = (float)(SQPI/(alpha*alpha));
  float density = (float)((double)RPI/(double)volume);
  float rorder = 1.f/(float)order;
  INIT_AND_ALLOCATE(float4,xq    ,memsize_xq    ,n)
  CHECK_ERROR("after INIT_AND_ALLOCATE a2");  
  INIT_AND_ALLOCATE(float4,fp    ,memsize_fp    ,n)
  CHECK_ERROR("after INIT_AND_ALLOCATE a3");  
  INIT_AND_ALLOCATE(float ,bx    ,memsize_bx    ,size_x)
  CHECK_ERROR("after INIT_AND_ALLOCATE a4");  
  INIT_AND_ALLOCATE(float ,by    ,memsize_by    ,size_y)
  CHECK_ERROR("after INIT_AND_ALLOCATE a5");  
  INIT_AND_ALLOCATE(float ,bz    ,memsize_bz    ,size_z)
  CHECK_ERROR("after INIT_AND_ALLOCATE a6");  
  INIT_AND_ALLOCATE(float ,recipf,memsize_recipf,9)
  CHECK_ERROR("after INIT_AND_ALLOCATE a7");  
  for (int i = 0; i < size_x; i++) bx[i] = (float)bsp_mod1[i];
  for (int i = 0; i < size_y; i++) by[i] = (float)bsp_mod2[i];
  for (int i = 0; i < size_z; i++) bz[i] = (float)bsp_mod3[i];
  for (int i = 0; i < 9; i++) recipf[i] = (float)recip[i];
  //  CHECK_ERROR("before cudaMalloc fp_dev");  
  float4* fp_dev;CUDA_SAFE_CALL(cudaMalloc((void**)&fp_dev, memsize_fp));
  //  INIT_AND_ALLOCATE(float4,fp_dev,memsize_fp_dev,n)
  CHECK_ERROR("cudaMalloc fp_dev");  
  INIT_AND_ALLOCATE(INT_OR_FLOAT   ,qm    ,memsize_qm    ,size_x*size_y*size_z*2)
  CHECK_ERROR("after INIT_AND_ALLOCATE a1");  
#if FLG_STRESS_SAKAMAKI
  INIT_AND_ALLOCATE(float ,stress,memsize_stress,(size_x*size_y*size_z+BSIZEPME1-1)/BSIZEPME1*6);
  float* stress_dev; CUDA_SAFE_CALL(cudaMalloc((void**)&stress_dev, memsize_stress));
#endif // FLG_STRESS_SAKAMAKI
#ifdef CELLINDEX
  float4 *xq2;
  int ni2,(*cpb)[4],ldim[3];
  INT_OR_FLOAT *qm_dev;
  INT_OR_FLOAT *qm2_dev;
  float4 *xq_dev,*xq2_dev;
  int *cpb_dev;
  float4 *fp2;
  unsigned int memsize_fp2;
  float4* fp2_dev;
  INT_OR_FLOAT *qm2;
  unsigned int memsize_qm2=sizeof(INT_OR_FLOAT)*size_x*size_y*size_z*2*8;
  if(newroutineflag!=0){
    CUDA_SAFE_CALL(cudaMalloc((void**)&qm_dev, memsize_qm));  
    CHECK_ERROR("cudaMalloc qm_dev");  
    //INIT_AND_ALLOCATE(INT_OR_FLOAT   ,qm2    ,memsize_qm2    ,size_x*size_y*size_z*2*8);
    //CHECK_ERROR("after INIT_AND_ALLOCATE qm2");  
    //    unsigned int memsize_qm2=sizeof(INT_OR_FLOAT)*size_x*size_y*size_z*2*8*2;
    qm2=(INT_OR_FLOAT *)malloc(memsize_qm2);
#if 0 // initialize qm2 is not required
    //    for (int i = 0; i < size_x*size_y*size_z*2*8; i++) qm2[i] = 0;
    bzero(qm2,memsize_qm2);
    GPU_ALLOCATE_AND_COPY(INT_OR_FLOAT,qm2,qm2_dev,memsize_qm2);
    CHECK_ERROR("GPU_ALLOCATE_AND_COPY qm2");  
#else
    CUDA_SAFE_CALL(cudaMalloc((void**)&qm2_dev, memsize_qm2));  
    CHECK_ERROR("cudaMalloc qm2");  
#endif

#if 1 // call GPU to initialize qm and qm2    
    dim3 threadsq(BSIZEPME1);
    //  dim3 gridq((int)((float)(size_x*size_y*size_z-1)/(float)(BSIZEPME1))+1);
    dim3 gridq((size_x*size_y*size_z+BSIZEPME1-1)/BSIZEPME1);
#if 1 // modified by Ryuji, Nov. 2011
    if (gridq.x > MAX_BSIZE){
      gridq.y=(gridq.x-1)/MAX_BSIZE+1;
      gridq.x=(gridq.x-1)/gridq.y+1;
    }
#endif
    gpupme1initializeq_kernel<<< gridq, threadsq >>>
      (qm_dev,qm2_dev,size_x,size_y,size_z);
    CHECK_ERROR("gpupme1initializeq_kernel");  
#else
    printf("*** gpupme1initializeq_kernel is not called **\n");
    bzero(qm2,memsize_qm2);
    bzero(qm,memsize_qm);
    cudaMemcpy(qm_dev,qm,memsize_qm,cudaMemcpyHostToDevice);
    cudaMemcpy(qm2_dev,qm2,memsize_qm2,cudaMemcpyHostToDevice);
#endif
    
    make_cellindex(natm,x,q,size_x,size_y,size_z,order,recipf,BSIZEPME2,&ni2,&xq2,&cpb,ldim,0.0);
    //  ni2=221184;fprintf(stderr,"** ni2 is modified to %d **\n",ni2);
#if 0 // duplicate make_cellindex for timing
    free(xq2);free(cpb);
    make_cellindex(natm,x,q,size_x,size_y,size_z,order,recipf,BSIZEPME2,&ni2,&xq2,&cpb,ldim,0.0);
#endif
    int memsize_xq2=sizeof(float4)*ni2;
    //GPU_ALLOCATE_AND_COPY(float4      ,xq2    ,xq2_dev    ,memsize_xq2);
    //CHECK_ERROR("GPU_ALLOCATE_AND_COPY xq2");  
    CUDA_SAFE_CALL(cudaMalloc((void**)&xq2_dev,memsize_xq2));
    CHECK_ERROR("cudaMalloc xq2_dev");
    CUDA_SAFE_CALL(cudaMemcpy(xq2_dev,xq2,memsize_xq2,cudaMemcpyHostToDevice));
    CHECK_ERROR("cudaMemcpy xq2->xq2_dev");

    //INIT_AND_ALLOCATE(float4,fp2    ,memsize_fp2    ,ni2);
    //CHECK_ERROR("after INIT_AND_ALLOCATE fp2");  
    memsize_fp2=sizeof(float4)*ni2;
    fp2=(float4 *)malloc(memsize_fp2);

    CUDA_SAFE_CALL(cudaMalloc((void**)&fp2_dev, memsize_fp2));
    CHECK_ERROR("cudaMalloc fp2_dev");  
    
    //    int memsize_cpb=sizeof(int)*4*(ni2+BSIZEPME2-1)/BSIZEPME2;
    int memsize_cpb=sizeof(int)*4*ni2/BSIZEPME2;
    //    printf("ni2=%d memsize_cpb=%d  int*4*ni2/128=%d int*4*ldim012/128=%d\n",ni2,memsize_cpb,sizeof(int)*4*((ni2+BSIZEPME2-1)/BSIZEPME2),sizeof(int)*4*((ldim[0]*ldim[1]*ldim[2]+BSIZEPME2-1)/BSIZEPME2));
    //GPU_ALLOCATE_AND_COPY(int      ,cpb    ,cpb_dev    ,memsize_cpb);
    //CHECK_ERROR("GPU_ALLOCATE_AND_COPY cpb");  
    CUDA_SAFE_CALL(cudaMalloc((void**)&cpb_dev,memsize_cpb));
    CHECK_ERROR("cudaMalloc cpb_dev");
    CUDA_SAFE_CALL(cudaMemcpy(cpb_dev,cpb,memsize_cpb,cudaMemcpyHostToDevice));
    CHECK_ERROR("cudaMemcpy cpb->cpb_dev");
    //  for(int i=0;i<ni2/BSIZEPME2;i++) printf("cpb[%d]=(%d,%d,%d) %d\n",i,cpb[i][0],cpb[i][1],cpb[i][2],cpb[i][3]);
#if 0 // use gradsum_kernel_simple
    for (int i = 0; i < natm; i++){ COPY_DOUBLE_TO_COULOMB(xq,x,q,i,i) }
    for (int i = natm; i < n; i++){ SET_FLOAT4_ZERO(xq[i])             }
    //GPU_ALLOCATE_AND_COPY(float4      ,xq    ,xq_dev    ,memsize_xq);
    //CHECK_ERROR("GPU_ALLOCATE_AND_COPY xq");  
    CUDA_SAFE_CALL(cudaMalloc((void**)&xq_dev, memsize_xq));  
    CHECK_ERROR("cudaMalloc xq_dev");  
    CUDA_SAFE_CALL(cudaMemcpy(xq_dev,xq,memsize_xq,cudaMemcpyHostToDevice));
    CHECK_ERROR("cudaMemcpy xq->xq_dev");
#endif
  }
  else{
    for (int i = 0; i < size_x*size_y*size_z*2; i++) qm[i] = 0;
    //GPU_ALLOCATE_AND_COPY(INT_OR_FLOAT         ,qm    ,qm_dev    ,memsize_qm);
    //CHECK_ERROR("GPU_ALLOCATE_AND_COPY qm");  
    CUDA_SAFE_CALL(cudaMalloc((void**)&qm_dev, memsize_qm));  
    CHECK_ERROR("cudaMalloc qm_dev");  
    CUDA_SAFE_CALL(cudaMemcpy(qm_dev,qm,memsize_qm,cudaMemcpyHostToDevice));
    CHECK_ERROR("cudaMemcpy qm->qm_dev");

    for (int i = 0; i < natm; i++){ COPY_DOUBLE_TO_COULOMB(xq,x,q,i,i) }
    for (int i = natm; i < n; i++){ SET_FLOAT4_ZERO(xq[i])             }
    //GPU_ALLOCATE_AND_COPY(float4      ,xq    ,xq_dev    ,memsize_xq);
    //CHECK_ERROR("GPU_ALLOCATE_AND_COPY xq");  
    CUDA_SAFE_CALL(cudaMalloc((void**)&xq_dev, memsize_xq));  
    CHECK_ERROR("cudaMalloc xq_dev");  
    CUDA_SAFE_CALL(cudaMemcpy(xq_dev,xq,memsize_xq,cudaMemcpyHostToDevice));
    CHECK_ERROR("cudaMemcpy xq->xq_dev");
  }
#else // else of CELLINDEX
  for (int i = 0; i < size_x*size_y*size_z*2; i++) qm[i] = 0;
  GPU_ALLOCATE_AND_COPY(INT_OR_FLOAT,qm    ,qm_dev    ,memsize_qm);
  CHECK_ERROR("GPU_ALLOCATE_AND_COPY qm");  
  for (int i = 0; i < natm; i++){ COPY_DOUBLE_TO_COULOMB(xq,x,q,i,i) }
  for (int i = natm; i < n; i++){ SET_FLOAT4_ZERO(xq[i])             }
  GPU_ALLOCATE_AND_COPY(float4      ,xq    ,xq_dev    ,memsize_xq);
  CHECK_ERROR("GPU_ALLOCATE_AND_COPY xq");  
#endif // end of CELLINDEX

  GPU_ALLOCATE_AND_COPY(float       ,bx    ,bx_dev    ,memsize_bx)
  CHECK_ERROR("GPU_ALLOCATE_AND_COPY bx");  
  GPU_ALLOCATE_AND_COPY(float       ,by    ,by_dev    ,memsize_by)
  CHECK_ERROR("GPU_ALLOCATE_AND_COPY by");  
  GPU_ALLOCATE_AND_COPY(float       ,bz    ,bz_dev    ,memsize_bz)
  CHECK_ERROR("GPU_ALLOCATE_AND_COPY bz");  
  GPU_ALLOCATE_AND_COPY(float       ,recipf,recipf_dev,memsize_recipf)
  CHECK_ERROR("GPU_ALLOCATE_AND_COPY recipf");  

////////////////////////////////////////////////////////////////////////////////
// Initialize texture unit (calculate bspline curve)
////////////////////////////////////////////////////////////////////////////////
  //  int tex_l = 1020;
  int tex_l = NSEG * order;
  INIT_AND_ALLOCATE(float2,tex_data,memsize_tex,tex_l)
  INIT_AND_ALLOCATE(double,bspline,memsize_bspline,order*2)
  CHECK_ERROR("after INIT_AND_ALLOCATE b");  
  double tmp, tmp1;
  for (int i = 0; i < NSEG; i++){
    tmp = ((double)i+0.0e0)/((double)(NSEG-1));
    bspline[0*2+0] = 1.e0 - tmp;
    bspline[1*2+0] = tmp;
    bspline[(order-1)*2+0] = 0.e0;
    for(int k = 3; k < order; k++){
      tmp1 = 1.e0 / ((double)(k-1));
      bspline[(k-1)*2+0] = tmp1*tmp*bspline[(k-2)*2+0];
      for(int j = 1; j < k-1; j++)
	bspline[(k-j-1)*2+0] = tmp1*(((double)j+tmp)*bspline[(k-j-2)*2+0]
		                  +((double)(k-j)-tmp)*bspline[(k-j-1)*2+0]);
      bspline[0*2+0] = tmp1*(1.e0-tmp)*bspline[0*2+0];
    }
    bspline[0*2+1] = -bspline[0*2+0];
    for (int j = 2; j < order+1; j++)
      bspline[(j-1)*2+1] = bspline[(j-2)*2+0] - bspline[(j-1)*2+0];
    tmp1 = 1.e0 / ((double)order-1);
    bspline[(order-1)*2+0] = tmp1*tmp*bspline[(order-2)*2+0];
    for (int j = 1; j < order-1; j++)
      bspline[(order-j-1)*2+0] = tmp1*(((double)j+tmp)*bspline[(order-j-2)*2+0]
	                         +((double)(order-j)-tmp)*bspline[(order-j-1)*2+0]);
    bspline[0*2+0] = tmp1*(1.e0-tmp)*bspline[0*2+0];
    for (int j = 0; j < order; j++){
      tex_data[(NSEG)*j+(NSEG-i-1)].x = (float)bspline[j*2+0];
      tex_data[(NSEG)*j+(NSEG-i-1)].y = (float)bspline[j*2+1];
    }
  }
  free(bspline);
  //  SET_FLOAT2_ZERO(tex_data[tex_l-1])
  //  SET_FLOAT2_ZERO(tex_data[0])
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();
  CHECK_ERROR("cudaChannelFormatDesc");  
  cudaArray* cu_array;
  cudaMallocArray(&cu_array, &channelDesc, tex_l);
  cudaMemcpyToArray(cu_array, 0, 0, tex_data, memsize_tex, cudaMemcpyHostToDevice);
  CHECK_ERROR("cudaMemcpyToArray of tex_data");  
  tex.normalized = true;
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModeLinear;
  CUDA_SAFE_CALL(cudaBindTextureToArray(tex, cu_array, channelDesc));
  CHECK_ERROR("cudaBindTextureToArray");  
#if FLG_TIMER
  timer[1].stopgpu();
#endif // FLG_TIMER
////////////////////////////////////////////////////////////////////////////////
// Fill charge to grid
////////////////////////////////////////////////////////////////////////////////
#if FLG_TIMER
  timer[2].start("p2m");
#endif // FLG_TIMER
#if 1 // skip fill charge grid
  dim3 threads0(BSIZEPME2);
#ifdef CELLINDEX
  if(newroutineflag!=0 && 1){
//#if defined(MULTIPLE_IBLOCK) || defined(MULTIPLE_IBLOCK_TEST2)
#if defined(MULTIPLE_IBLOCK) 
    dim3 grid0(ldim[0]*ldim[1]*ldim[2]);
    //    printf("** ldim=%d %d %d, ni2=%d BSIZEPME2=%d\n",ldim[0],ldim[1],ldim[2],ni2,BSIZEPME2);
#else
    dim3 grid0(ni2/BSIZEPME2);
    //    printf("ni2=%d BSIZEPME2=%d grid size=%d\n",ni2,BSIZEPME2,ni2/BSIZEPME2);
#endif
    if (grid0.x > MAX_BSIZE){
      fprintf(stderr,"** error : x dimention of grid block should be less than %d (2)**\n",MAX_BSIZE);
      exit(1);
    }
    gpupme1fillch_kernel<<< grid0, threads0 >>>
      (xq2_dev, qm_dev, qm2_dev, size_x, size_y, size_z, recipf_dev, order, rorder,
       (int (*)[4])cpb_dev, size_x/ldim[0], size_y/ldim[1], size_z/ldim[2]);
    //    (xq2_dev, qm_dev, size_x, size_y, size_z, recipf_dev, order, rorder);
    CHECK_ERROR("gpupme1fillch_kernel");  
    CUT_CHECK_ERROR("Kernel execution failed");
#if 0 // debug for qm and qm2
    fprintf(stderr,"** printing debugging information **\n");
    CUDA_SAFE_CALL(cudaMemcpy(qm, qm_dev, memsize_qm, cudaMemcpyDeviceToHost));
    CHECK_ERROR("memcpy qm_dev");  
    CUDA_SAFE_CALL(cudaMemcpy(qm2, qm2_dev, memsize_qm2, cudaMemcpyDeviceToHost));
    CHECK_ERROR("memcpy qm2_dev");  
    printf("qm2\n");for(int i=0;i<size_x*size_y*size_z*8;i++) if(i<10 && qm2[i*2]!=0) printf("qm2[%d+%d(%d,%d,%d)]=%d %e\n",i/size_x/size_y/size_z,i % (size_x*size_y*size_z),i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,qm2[i*2],qm2[i*2]/FIXEDPOINT);
    //  for(int i=0;i<ni2;i++) printf("cptoffset i=%d num=%d cptoffset=%d %d blockindex=%d\n",i,qm2[i*4],qm2[i*4+1],qm2[i*4+2],qm2[i*4+3]);
    bzero(qm,memsize_qm);
    for(int i=0;i<8;i++){
      for(int j=0;j<size_x*size_y*size_z;j++){
	if(j==47) printf(" i=%d qm2[%d(%d,%d,%d)]=%d %e\n",
			 i,j,j % size_x,(j/size_x) % size_y,(j/size_x/size_y) % size_z,qm2[j*2+i*size_x*size_y*size_z*2],qm2[j*2+i*size_x*size_y*size_z*2]/FIXEDPOINT);
	qm[j*2]+=qm2[j*2+i*size_x*size_y*size_z*2];
      }
    }
    printf("qm\n");for(int i=0;i<size_x*size_y*size_z;i++) if(i==47 && qm[i*2]!=0) printf("qm[%d(%d,%d,%d)]=%d %e\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,qm[i*2],qm[i*2]/FIXEDPOINT);
#endif // end of debug
  }
  else{
    dim3 grid0(n/BSIZEPME2);
    if (grid0.x > MAX_BSIZE){
      fprintf(stderr,"** error : x dimention of grid block should be less than %d (3)**\n",MAX_BSIZE);
      exit(1);
    }
    gpupme1fillch_kernel_org<<< grid0, threads0 >>>
      (xq_dev, qm_dev, size_x, size_y, size_z, recipf_dev, order, rorder);
    CHECK_ERROR("gpupme1fillch_kernel_org");  
    CUT_CHECK_ERROR("Kernel execution failed");
#if 0 // debug for qm
    CUDA_SAFE_CALL(cudaMemcpy(qm, qm_dev, memsize_qm, cudaMemcpyDeviceToHost));
    CHECK_ERROR("memcpy qm_dev");  
#ifdef FIXEDPOINT
    printf("qm\n");for(int i=0;i<size_x*size_y*size_z;i++) if(i==47 && qm[i*2]!=0) printf("qm[%d(%d,%d,%d)]=%d %e\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,qm[i*2],qm[i*2]/FIXEDPOINT);
#else
    printf("qm\n");for(int i=0;i<size_x*size_y*size_z;i++) if(i<10 && qm[i*2]!=0) printf("qm[%d(%d,%d,%d)]=%e\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,qm[i*2]);
#endif
#endif // end of debug
  }
#else // else of CELLINDEX
  dim3 grid0(n/BSIZEPME2);
  if (grid0.x > MAX_BSIZE){
    fprintf(stderr,"** error : x dimention of grid block should be less than %d (4)**\n",MAX_BSIZE);
    exit(1);
  }
  gpupme1fillch_kernel_org<<< grid0, threads0 >>>
    (xq_dev, qm_dev, size_x, size_y, size_z, recipf_dev, order, rorder);
  CHECK_ERROR("gpupme1fillch_kernel_org");  
  CUT_CHECK_ERROR("Kernel execution failed");
#if 0 // debug for qm
  CUDA_SAFE_CALL(cudaMemcpy(qm, qm_dev, memsize_qm, cudaMemcpyDeviceToHost));
  CHECK_ERROR("memcpy qm_dev");  
  printf("qm\n");for(int i=0;i<size_x*size_y*size_z;i++) if(qm[i*2]!=0) printf("qm[%d(%d,%d,%d)]=%d %e\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,qm[i*2],qm[i*2]/FIXEDPOINT);
#endif // end of debug
#endif // end of CELLINDEX
#endif // end of skip
#if FLG_TIMER
  timer[2].stopgpu();
#endif // FLG_TIMER
////////////////////////////////////////////////////////////////////////////////
// Exchange integer to float on mesh
////////////////////////////////////////////////////////////////////////////////
#if FLG_TIMER
  timer[3].start("convert");
#endif // FLG_TIMER
  float* q_dev = (float*)qm_dev;
  dim3 threads1(BSIZEPME1);
  //  dim3 grid1((int)((float)(size_x*size_y*size_z-1)/(float)(BSIZEPME1))+1);
  dim3 grid1((size_x*size_y*size_z+BSIZEPME1-1)/BSIZEPME1);
#if 1 // modified by Ryuji, Nov. 2011
  if (grid1.x > MAX_BSIZE){
    grid1.y=(grid1.x-1)/MAX_BSIZE+1;
    grid1.x=(grid1.x-1)/grid1.y+1;
  }
#endif
#if 1 // skip exchange
#ifdef CELLINDEX
  if(newroutineflag!=0){
    gpupme1exchange_kernel<<< grid1, threads1 >>>
      (qm_dev, qm2_dev, q_dev, size_x, size_y, size_z);
    //    (qm_dev, q_dev, size_x, size_y, size_z);
  }
  else{
    gpupme1exchange_kernel_org<<< grid1, threads1 >>>
      (qm_dev, q_dev, size_x, size_y, size_z);
  }
#else // else of CELLINDEX
  gpupme1exchange_kernel_org<<< grid1, threads1 >>>
    (qm_dev, q_dev, size_x, size_y, size_z);
#endif // end of CELLINDEX
  CHECK_ERROR("gpupme1exchange_kernel");  
  CUT_CHECK_ERROR("Kernel execution failed");
#endif
#if 0 // printout result of exchange_kernel
  cudaMemcpy(qm,q_dev,memsize_qm,cudaMemcpyDeviceToHost);
  CHECK_ERROR("memcpy qm_dev");  
  for(int i=0;i<4;i++) printf("qm[%d]=%e\n",i,qm[i]);
#endif
#if 0 // debug for q
  cudaThreadSynchronize();
  cudaMemcpy(qm, q_dev, memsize_qm, cudaMemcpyDeviceToHost);
  for(int i=0;i<10;i++) if(qm[i*2]!=0) 
#if defined(FIXEDPOINT) && 0
			  printf("before forwardFFT a qm[%d(%d,%d,%d)]=%d %016llx\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,qm[i*2],*(unsigned long long *)(qm+i*2));
#else
			  printf("before forwardFFT a qm[%d(%d,%d,%d)]=%e %016llx\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,((float *)qm)[i*2],*(unsigned long long *)(qm+i*2));
#endif
  cudaMemcpy(qm, qm_dev, memsize_qm, cudaMemcpyDeviceToHost);
  for(int i=0;i<10;i++) if(qm[i*2]!=0) 
#if defined(FIXEDPOINT) && 0
			  printf("before forwardFFT b qm[%d(%d,%d,%d)]=%d %016llx\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,qm[i*2],*(unsigned long long *)(qm+i*2));
#else
			  printf("before forwardFFT b qm[%d(%d,%d,%d)]=%e %016llx\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,((float *)qm)[i*2],*(unsigned long long *)(qm+i*2));
#endif
  cudaMemcpy(qm, qm2_dev, memsize_qm, cudaMemcpyDeviceToHost);
  for(int i=0;i<10;i++) if(qm[i*2]!=0) 
#ifdef FIXEDPOINT
			  printf("before forwardFFT c qm[%d(%d,%d,%d)]=%d %016llx\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,qm[i*2],*(unsigned long long *)(qm+i*2));
#else
			  printf("before forwardFFT c qm[%d(%d,%d,%d)]=%e %016llx\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,((float *)qm)[i*2],*(unsigned long long *)(qm+i*2));
#endif
#endif // end of debug
#if FLG_TIMER
  timer[3].stopgpu();
#endif // FLG_TIMER
////////////////////////////////////////////////////////////////////////////////
// Forward FFT
////////////////////////////////////////////////////////////////////////////////
#if FLG_TIMER
  timer[4].start("FFT");
#endif // FLG_TIMER
  cufftComplex* idata = (cufftComplex*)qm_dev;
  cufftHandle plan1;
#if 1 // skip FFT
  cufftPlan3d(&plan1, size_z, size_y, size_x, CUFFT_C2C);
  CHECK_ERROR("cuttfPlan3d");  
  cufftExecC2C(plan1, idata, idata, CUFFT_FORWARD);
  CHECK_ERROR("cufftExecC2C");  
  cufftDestroy(plan1);
  CHECK_ERROR("cufftDestroy");  
#endif
#if 0 // debug for q
  cudaThreadSynchronize();
  cudaMemcpy(qm, q_dev, memsize_qm, cudaMemcpyDeviceToHost);
  for(int i=0;i<4;i++) if(qm[i*2]!=0) printf("after forwardFFT qm[%d(%d,%d,%d)]=%e %016llx\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,((float *)qm)[i*2],*(unsigned long long *)(qm+i*2));
#endif // end of debug
#if FLG_TIMER
  timer[4].stopgpu();
#endif // FLG_TIMER
////////////////////////////////////////////////////////////////////////////////
// Convolution in reciprocal space
////////////////////////////////////////////////////////////////////////////////
#if FLG_TIMER
  timer[5].start("Convolution");
#endif // FLG_TIMER
  dim3 threads2(BSIZEPME1);
  //  dim3 grid2((int)((float)(size_x*size_y*size_z-1)/(float)(BSIZEPME1))+1);
  dim3 grid2((size_x*size_y*size_z+BSIZEPME1-1)/BSIZEPME1);
  //dim3 threads2(BSIZEPME1,BSIZEPME1,BSIZEPME1);
  //dim3 grid2(size_x/BSIZEPME1,size_y/BSIZEPME1,size_z/BSIZEPME1);
#if 1 // modified by Ryuji, Nov. 2011
  if (grid2.x > MAX_BSIZE){
    grid2.y=(grid2.x-1)/MAX_BSIZE+1;
    grid2.x=(grid2.x-1)/grid2.y+1;
  }
#endif
#if 1 // skip convolution
#if FLG_STRESS_SAKAMAKI && 1
  gpupme1_kernel_stress<<< grid2, threads2 >>>
    (q_dev, bx_dev, by_dev, bz_dev, size_x, size_y, size_z, recipf_dev,
    factor, density, stress_dev);
  //gpupme1_kernel<<< grid2, threads2 >>>
    //(q_dev, bx_dev, by_dev, bz_dev, size_x, size_y, size_z, recipf_dev,
    //factor, density);
#else // FLG_STRESS_SAKAMAKI
  //  printf("** gpupme1_kernel is used instead of gpupme1_kernel_stress **\n");
  gpupme1_kernel<<< grid2, threads2 >>>
    (q_dev, bx_dev, by_dev, bz_dev, size_x, size_y, size_z, recipf_dev,
    factor, density);
#endif // FLG_STRESS_SAKAMAKI
  CHECK_ERROR("gpupme1_kernel");  
  CUT_CHECK_ERROR("Kernel execution failed");
#endif
#if FLG_TIMER
  timer[5].stopgpu();
#endif // FLG_TIMER
////////////////////////////////////////////////////////////////////////////////
// Inverse FFT
////////////////////////////////////////////////////////////////////////////////
#if FLG_TIMER
  timer[6].start("IFFT");
#endif // FLG_TIMER
  cufftHandle plan2;
#if 1 // skip inverse FFT
  cufftPlan3d(&plan2, size_z, size_y, size_x, CUFFT_C2C);
  CHECK_ERROR("cufftPlan3d");  
  cufftExecC2C(plan2, idata, idata, CUFFT_INVERSE);
  CHECK_ERROR("cufftExecC2C");  
  cufftDestroy(plan2);
  CHECK_ERROR("cufftDestroy");  
#endif
#if 0 // debug for q
  cudaThreadSynchronize();
  cudaMemcpy(qm, q_dev, memsize_qm, cudaMemcpyDeviceToHost);
  for(int i=0;i<4;i++) if(qm[i*2]!=0) printf("after inverseFFT qm[%d(%d,%d,%d)]=%e %016llx\n",i,i % size_x,(i/size_x) % size_y,(i/size_x/size_y) % size_z,((float *)qm)[i*2],*(unsigned long long *)(qm+i*2));
#endif // end of debug
#if FLG_TIMER
  timer[6].stopgpu();
#endif // FLG_TIMER
////////////////////////////////////////////////////////////////////////////////
// potential assignment mesh to point
////////////////////////////////////////////////////////////////////////////////
#if FLG_TIMER
  timer[7].start("M2P");
#endif // FLG_TIMER
#if 1 // skip gradsum
#if defined(CELLINDEX) && 1
  if(newroutineflag!=0 && 1){
#if 0 // use xq instead of xq2
    dim3 threads3(BSIZEPME2);
    dim3 grid3(n/BSIZEPME2);
    if (grid3.x > MAX_BSIZE){
      fprintf(stderr,"** error : x dimention of grid block should be less than %d (7)**\n",MAX_BSIZE);
      exit(1);
    }
    gpupme1gradsum_kernel_simple<<< grid3, threads3 >>>
      (xq_dev, fp_dev, q_dev, size_x, size_y, size_z, recipf_dev, order, rorder);
    CHECK_ERROR("gpupme1gradsum_kernel_simple");  
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL(cudaMemcpy(fp, fp_dev, memsize_xq, cudaMemcpyDeviceToHost));
    CHECK_ERROR("memcpy fp_dev");  
    for (int i = 0; i < natm; i++){
      f[i*3+0] += (double)fp[i].x * coef;
      f[i*3+1] += (double)fp[i].y * coef;
      f[i*3+2] += (double)fp[i].z * coef;
      p[i]     += (double)fp[i].w * coef;
    }
#else // else of use xq instread of xq2
    dim3 threads3(BSIZEPME2);
#ifdef MULTIPLE_IBLOCK
    dim3 grid3(ldim[0]*ldim[1]*ldim[2]);
#else
    dim3 grid3(ni2/BSIZEPME2);
#endif
    if (grid3.x > MAX_BSIZE){
      fprintf(stderr,"** error : x dimention of grid block should be less than %d (8)**\n",MAX_BSIZE);
      exit(1);
    }
#if 1
    gpupme1gradsum_kernel<<< grid3, threads3 >>>
      (xq2_dev, fp2_dev, q_dev, size_x, size_y, size_z, recipf_dev, order, rorder,
       (int (*)[4])cpb_dev, size_x/ldim[0], size_y/ldim[1], size_z/ldim[2]);
#else
    gpupme1gradsum_kernel_org<<< grid3, threads3 >>>
      (xq2_dev, fp2_dev, q_dev, size_x, size_y, size_z, recipf_dev, order, rorder);
#endif
    CHECK_ERROR("gpupme1gradsum_kernel");  
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL(cudaMemcpy(fp2, fp2_dev, memsize_fp2, cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL(cudaMemcpy(fp, fp_dev, memsize_fp, cudaMemcpyDeviceToHost));
    CHECK_ERROR("memcpy fp2_dev");  
    make_cellindex(natm,f,p,0,0,0,0,NULL,0,NULL,&fp2,NULL,NULL,coef);
    //  make_cellindex(natm,f,p,0,0,0,0,NULL,0,NULL,&fp,NULL,NULL,coef);
#endif // end of use xq instread of xq2
  }
  else{
    dim3 threads3(BSIZEPME2);
    dim3 grid3(n/BSIZEPME2);
    if (grid3.x > MAX_BSIZE){
      fprintf(stderr,"** error : x dimention of grid block should be less than %d (9)**\n",MAX_BSIZE);
      exit(1);
    }
    gpupme1gradsum_kernel_org<<< grid3, threads3 >>>
      (xq_dev, fp_dev, q_dev, size_x, size_y, size_z, recipf_dev, order, rorder);
    CHECK_ERROR("gpupme1gradsum_kernel");  
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL(cudaMemcpy(fp, fp_dev, memsize_xq, cudaMemcpyDeviceToHost));
    CHECK_ERROR("memcpy fp_dev");  
    for (int i = 0; i < natm; i++){
      f[i*3+0] += (double)fp[i].x * coef;
      f[i*3+1] += (double)fp[i].y * coef;
      f[i*3+2] += (double)fp[i].z * coef;
      p[i]     += (double)fp[i].w * coef;
    }
  }
#else // else of CELLINDEX 
  dim3 threads3(BSIZEPME2);
  dim3 grid3(n/BSIZEPME2);
  if (grid3.x > MAX_BSIZE){
    fprintf(stderr,"** error : x dimention of grid block should be less than %d (10)**\n",MAX_BSIZE);
    exit(1);
  }
  gpupme1gradsum_kernel_org<<< grid3, threads3 >>>
    (xq_dev, fp_dev, q_dev, size_x, size_y, size_z, recipf_dev, order, rorder);
  CHECK_ERROR("gpupme1gradsum_kernel");  
  CUT_CHECK_ERROR("Kernel execution failed");
  CUDA_SAFE_CALL(cudaMemcpy(fp, fp_dev, memsize_xq, cudaMemcpyDeviceToHost));
  CHECK_ERROR("memcpy fp_dev");  
  for (int i = 0; i < natm; i++){
    f[i*3+0] += (double)fp[i].x * coef;
    f[i*3+1] += (double)fp[i].y * coef;
    f[i*3+2] += (double)fp[i].z * coef;
    p[i]     += (double)fp[i].w * coef;
  }
#endif // end of CELLINDEX
#endif
#if FLG_STRESS_SAKAMAKI
  CUDA_SAFE_CALL(cudaMemcpy(stress, stress_dev, memsize_stress, cudaMemcpyDeviceToHost));
  for (int i = 0; i < (size_x*size_y*size_z+BSIZEPME1-1)/BSIZEPME1; i++){
    stress9[0] += (double)stress[i*6+0] * 0.5e0 * recip[0] * recip[4] * recip [8];
    stress9[1] += (double)stress[i*6+1] * 0.5e0 * recip[0] * recip[4] * recip [8];
    stress9[2] += (double)stress[i*6+2] * 0.5e0 * recip[0] * recip[4] * recip [8];
    stress9[3] += (double)stress[i*6+1] * 0.5e0 * recip[0] * recip[4] * recip [8];
    stress9[4] += (double)stress[i*6+3] * 0.5e0 * recip[0] * recip[4] * recip [8];
    stress9[5] += (double)stress[i*6+4] * 0.5e0 * recip[0] * recip[4] * recip [8];
    stress9[6] += (double)stress[i*6+2] * 0.5e0 * recip[0] * recip[4] * recip [8];
    stress9[7] += (double)stress[i*6+4] * 0.5e0 * recip[0] * recip[4] * recip [8];
    stress9[8] += (double)stress[i*6+5] * 0.5e0 * recip[0] * recip[4] * recip [8];
  }
  //printf("%e %e %e %e %e %e\n",stress9[0],stress9[1],stress9[2],stress9[3],stress9[4],stress9[5]);
#endif // FLG_STRESS_SAKAMAKI
#if FLG_TIMER
  timer[7].stopgpu();
#endif // FLG_TIMER
////////////////////////////////////////////////////////////////////////////////
//  finalize
////////////////////////////////////////////////////////////////////////////////
#if FLG_TIMER
  timer[8].start("Postprocess");
#endif // FLG_TIMER
  CUDA_SAFE_CALL(cudaFree(qm_dev));
  CUDA_SAFE_CALL(cudaFree(bx_dev));
  CUDA_SAFE_CALL(cudaFree(by_dev));
  CUDA_SAFE_CALL(cudaFree(bz_dev));
  CUDA_SAFE_CALL(cudaFree(recipf_dev));
  //  CUDA_SAFE_CALL(cudaFree(xq_dev));
  CUDA_SAFE_CALL(cudaFree(fp_dev));
#ifdef CELLINDEX
  if(newroutineflag!=0){
    make_cellindex(0,NULL,NULL,0,0,0,0,NULL,0,NULL,NULL,NULL,NULL,1.0);
    CUDA_SAFE_CALL(cudaFree(xq2_dev));
    free(xq2);
    CUDA_SAFE_CALL(cudaFree(qm2_dev));
    free(qm2);
    CUDA_SAFE_CALL(cudaFree(cpb_dev));
    free(cpb);
    CUDA_SAFE_CALL(cudaFree(fp2_dev));
    free(fp2);
#if 0 // use gradsum_kernel_simple
    CUDA_SAFE_CALL(cudaFree(xq_dev));
#endif
  }
  else{
    CUDA_SAFE_CALL(cudaFree(xq_dev));
  }
#else
  CUDA_SAFE_CALL(cudaFree(xq_dev));
#endif
  CUDA_SAFE_CALL(cudaFreeArray(cu_array));
  CHECK_ERROR("free");  
  free(qm);
  free(xq);
  free(fp);
  free(bx);
  free(by);
  free(bz);
  free(recipf);
  free(tex_data);
#if FLG_STRESS_SAKAMAKI
  CUDA_SAFE_CALL(cudaFree(stress_dev));
  free(stress);
#endif // FLG_STRESS_SAKAMAKI
#if FLG_TIMER
  timer[8].stop();
  timer[0].stop();
  //for (int i = 0; i < 9; i++) printf("(%d) %s ",i,timer[i].label.c_str());
  for (int i = 0; i < 9; i++) printf(" %s, ",timer[i].label.c_str());
  printf("\n");
  //for (int i = 0; i < 9; i++) printf("(%d) %f ",i,timer[i].gettime());
  for (int i = 0; i < 9; i++) printf("%f ",i,timer[i].gettime());
  printf("\n");
#endif // FLG_TIMER
}

extern "C"
void gpupme_ 
#if FLG_STRESS_SAKAMAKI
(int natm, double* x, double* q, double* f, double* p,
 int size_x, int size_y, int size_z, double alpha,
 double* bsp_mod1, double* bsp_mod2, double* bsp_mod3,
 double volume, double* recip, double coef, int order, int idevice, double* stress9)
#else // FLG_STRESS_SAKAMAKI
(int natm, double* x, double* q, double* f, double* p,
 int size_x, int size_y, int size_z, double alpha,
 double* bsp_mod1, double* bsp_mod2, double* bsp_mod3,
 double volume, double* recip, double coef, int order, int idevice)
#endif // FLG_STRESS_SAKAMAKI
{
#ifdef SNAPSHOT
  static int snapcount=0,snapshotflag=0;
  FILE *file;
  char fname[1024];
  char *s;
  if(snapshotflag==0){
    if((s=getenv("VG_SNAPSHOT_COMPARE"))!=NULL){
      snapcount++;
      snapshotflag=3;// compare snapshot
      printf("VG_SNAPSHOT_COMPARE is defined\n");
    }
    else if((s=getenv("VG_SNAPSHOT"))!=NULL){
      snapcount++;
      snapshotflag=2;// write snapshot
      printf("VG_SNAPSHOT is defined\n");
    }
    else{
      snapshotflag=1;// nothing
    }
  }
  if(snapshotflag==3){// compare
    char line[1024];
    int lcount=0,bcount_x=0,bcount_y=0,bcount_z=0;
    int natom_new=0,size_x_new,size_y_new,size_z_new,order_new;
    double alpha_new=0.0,volume_new,coef_new,recip_new[9];
    double *bsp_mod1_new,*bsp_mod2_new,*bsp_mod3_new;
    double *x_new,*q_new,*f_new,*f_new2,*p_new,*p_new2;
    double stress9_new[9];
    sprintf(fname,"vggpu.dat%d",snapcount);
    if((file=fopen(fname,"r"))==NULL){
      fprintf(stderr,"** error : can't open %s **\n",fname);
      exit(1);
    }
    while((fgets(line,1024,file))!=NULL){
      if(line[0]!='#'){
	unsigned long long ull[20];
	int idx;
	if(natom_new==0){
	  sscanf(line,"%d%d%d%d%d",&natom_new,&size_x_new,&size_y_new,&size_z_new,&order_new);
	}
	else if(alpha_new==0.0){
	  sscanf(line,"%llx%llx%llx%llx%llx%llx%llx%llx%llx%llx%llx%llx",
		 ull+0,ull+1,ull+2,ull+3,ull+4,ull+5,ull+6,ull+7,ull+8,ull+9,ull+10,ull+11);
	  alpha_new=*((double *)(ull+0));
	  volume_new=*((double *)(ull+1));
	  coef_new=*((double *)(ull+2));
	  for(int j=0;j<9;j++) recip_new[j]=*((double *)(ull+3+j));
	}
	else if(bcount_x<size_x_new){
	  if(bcount_x==0){
	    bsp_mod1_new=(double *)malloc(sizeof(double)*size_x_new);
	  }
	  sscanf(line,"%d%llx",&idx,ull+0);
	  bsp_mod1_new[idx]=*((double *)(ull+0));
	  bcount_x++;
	}
	else if(bcount_y<size_y_new){
	  if(bcount_y==0){
	    bsp_mod2_new=(double *)malloc(sizeof(double)*size_y_new);
	  }
	  sscanf(line,"%d%llx",&idx,ull+0);
	  bsp_mod2_new[idx]=*((double *)(ull+0));
	  bcount_y++;
	}
	else if(bcount_z<size_z_new){
	  if(bcount_z==0){
	    bsp_mod3_new=(double *)malloc(sizeof(double)*size_z_new);
	  }
	  sscanf(line,"%d%llx",&idx,ull+0);
	  bsp_mod3_new[idx]=*((double *)(ull+0));
	  bcount_z++;
	}
	else{
	  if(lcount==0){
	    x_new=(double *)malloc(sizeof(double)*natom_new*3);
	    q_new=(double *)malloc(sizeof(double)*natom_new);
	    f_new=(double *)malloc(sizeof(double)*natom_new*3);
	    f_new2=(double *)malloc(sizeof(double)*natom_new*3);bzero(f_new2,sizeof(double)*natom_new*3);
	    p_new=(double *)malloc(sizeof(double)*natom_new);
	    p_new2=(double *)malloc(sizeof(double)*natom_new);bzero(p_new2,sizeof(double)*natom_new);
	    //	    printf("allocated %d x_new etc.\n",natom_new);
	  }
	  sscanf(line,"%d%llx%llx%llx%llx%llx%llx%llx%llx",&idx,ull+0,ull+1,ull+2,ull+3,ull+4,ull+5,ull+6,ull+7);
	  //	  if(lcount<5) printf("read-%d : %s\n",lcount,line);
	  for(int j=0;j<3;j++) x_new[idx*3+j]=*((double *)(ull+j));
	  q_new[idx]=*((double *)(ull+3));
	  for(int j=0;j<3;j++) f_new[idx*3+j]=*((double *)(ull+j+4));
	  p_new[idx]=*((double *)(ull+7));
	  lcount++;
	}
      }
    }
    fclose(file);

    /*
    printf("natom=%d size=%d %d %d order=%d\n",natom_new,size_x_new,size_y_new,size_x_new,order_new);
    printf("alpha=%e volume=%e coef=%e ",alpha_new,volume_new,coef_new);
    for(int j=0;j<9;j++) printf("%e ",recip_new[j]);printf("\n");
    for(int i=0;i<3;i++){
      printf("%d pos=%e %e %e q=%e f=%e %e %e p=%e\n",i,x_new[i*3+0],x_new[i*3+1],x_new[i*3+2],q_new[i],f_new[i*3+0],f_new[i*3+1],f_new[i*3+2],p_new[i]);
      }*/

    gpupme(natom_new,x_new,q_new,f_new2,p_new2,size_x_new,size_y_new,size_z_new,alpha_new,
	   bsp_mod1_new,bsp_mod2_new,bsp_mod3_new,volume_new,recip_new,coef_new,order_new,idevice
#if FLG_STRESS_SAKAMAKI
	   ,stress9_new
#endif
	   );

    {
      int ec=0;
      double sp_new=0.0,sp_new2=0.0;
      for(int i=0;i<natom_new;i++){
	sp_new+=p_new[i];
	sp_new2+=p_new2[i];
	if(f_new[i*3+0]!=f_new2[i*3+0] || f_new[i*3+1]!=f_new2[i*3+1] || f_new[i*3+2]!=f_new2[i*3+2]){
	  if(i<3) printf("  error : f_new[%d]=%e %e %e f_new2=%e %e %e\n",i,f_new[i*3+0],f_new[i*3+1],f_new[i*3+2],f_new2[i*3+0],f_new2[i*3+1],f_new2[i*3+2]);
	  ec++;
	}
      }
      printf("Sum of half of pot_new=%e pot_new2=%e. Found %d errors when comparing with %s\n",sp_new*0.5,sp_new2*0.5,ec,fname);
    }

    free(x_new);free(q_new);free(f_new);free(p_new);free(f_new2);free(p_new2);
    free(bsp_mod1_new);free(bsp_mod2_new);free(bsp_mod3_new);
    snapcount++;
  }
  else{
    if(snapshotflag==2){// write snapshot : check f or p is zero
      int c=0;
      for(int i=0;i<natm;i++){
	if(c<10 && (f[i*3]!=0.0 || f[i*3+1]!=0.0 || f[i*3+2]!=0.0 || p[i]!=0.0)){
	  printf("** warning : f or p is not zero : f[%d]=%e %e %e p=%e\n",i,f[i*3],f[i*3+1],f[i*3+2],p[i]);
	}
	c++;
      }
    }
    gpupme(natm,x,q,f,p,size_x,size_y,size_z,alpha,
	   bsp_mod1,bsp_mod2,bsp_mod3,volume,recip,coef,order,idevice
#if FLG_STRESS_SAKAMAKI
	   ,stress9
#endif
	   );
  }
#else // else of SNAPSHOT
  gpupme(natm,x,q,f,p,size_x,size_y,size_z,alpha,
	 bsp_mod1,bsp_mod2,bsp_mod3,volume,recip,coef,order,idevice
#if FLG_STRESS_SAKAMAKI
	 ,stress9
#endif
	 );
#endif // end of SNAPSHOT
  
#ifdef SNAPSHOT
  if(snapcount>0){
    sprintf(fname,"vggpu.dat%d",snapcount);
    if(snapshotflag==2){      // output snapshot
      if((file=fopen(fname,"w"))==NULL){
	fprintf(stderr,"** error : can't open %s **\n",fname);
	exit(1);
      }
      fprintf(file,"# number_of_atoms nfft_x nfft_y nfft_z order\n");
      fprintf(file,"%d %d %d %d %d\n",natm,size_x,size_y,size_z,order);
      fprintf(file,"# alpha volume coef recip(9)\n");
      fprintf(file,"# %e %e %e ",alpha,volume,coef);
      for(int i=0;i<9;i++) fprintf(file,"%e ",recip[i]);
      fprintf(file,"\n");
      fprintf(file,"%016llx  %016llx  %016llx  ",*((unsigned long long *)&alpha),*((unsigned long long *)&volume),*((unsigned long long *)&coef));
      for(int i=0;i<9;i++) fprintf(file,"%016llx  ",*((unsigned long long *)&recip[i]));
      fprintf(file,"\n");
      for(int i=0;i<size_x;i++){
	fprintf(file,"# bsp_mod1 %d %e\n",i,bsp_mod1[i]);
	fprintf(file,"%d %016llx\n",i,((unsigned long long *)bsp_mod1)[i]);
      }
      for(int i=0;i<size_y;i++){
	fprintf(file,"# bsp_mod2 %d %e\n",i,bsp_mod2[i]);
	fprintf(file,"%d %016llx\n",i,((unsigned long long *)bsp_mod2)[i]);
      }
      for(int i=0;i<size_z;i++){
	fprintf(file,"# bsp_mod3 %d %e\n",i,bsp_mod3[i]);
	fprintf(file,"%d %016llx\n",i,((unsigned long long *)bsp_mod3)[i]);
      }
      fprintf(file,"# index pos_x pos_y pos_z charge f_x f_y f_z p\n");
      for(int i=0;i<natm;i++){
	fprintf(file,"# %d ",i);
	for(int j=0;j<3;j++) fprintf(file,"%e ",x[i*3+j]);
	fprintf(file,"%e ",q[i]);
	for(int j=0;j<3;j++) fprintf(file,"%e ",f[i*3+j]);
	fprintf(file,"%e\n",p[i]);
	fprintf(file,"%d ",i);
	for(int j=0;j<3;j++) fprintf(file,"%016llx ",((unsigned long long *)x)[i*3+j]);
	fprintf(file,"%016llx ",((unsigned long long *)q)[i]);
	for(int j=0;j<3;j++) fprintf(file,"%016llx ",((unsigned long long *)f)[i*3+j]);
	fprintf(file,"%016llx\n",((unsigned long long *)p)[i]);
      }
      fclose(file);
      printf("wrote snapshot file %s\n",fname);
      snapcount++;
    }
  }
#endif
}


#if FLG_STRESS_SAKAMAKI
extern "C"
void gpupme__
(int* natm, double* x, double* q, double* f, double* p,
 int* size_x, int* size_y, int* size_z, double* alpha,
 double* bsp_mod1, double* bsp_mod2, double* bsp_mod3,
 double* volume, double* recip, double* coef, int* order, int* idevice, double* stress9)
{
  gpupme_
    (*natm, x, q, f, p, *size_x, *size_y, *size_z, *alpha,
     bsp_mod1, bsp_mod2, bsp_mod3, *volume, recip, *coef, *order, *idevice, stress9);
}
#else // FLG_STRESS_SAKAMAKI
extern "C"
void gpupme__
(int* natm, double* x, double* q, double* f, double* p,
 int* size_x, int* size_y, int* size_z, double* alpha,
 double* bsp_mod1, double* bsp_mod2, double* bsp_mod3,
 double* volume, double* recip, double* coef, int* order, int* idevice)
{
  gpupme_
    (*natm, x, q, f, p, *size_x, *size_y, *size_z, *alpha,
     bsp_mod1, bsp_mod2, bsp_mod3, *volume, recip, *coef, *order, *idevice);
}
#endif // FLG_STRESS_SAKAMAKI
