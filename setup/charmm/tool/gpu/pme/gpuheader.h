#if 0 // use cuda util in SDK
#include <cutil.h> 
#else             
#define CUDA_SAFE_CALL(x) x
#define CUT_CHECK_ERROR(x) 
#endif                  

//#define CELLINDEX // use cell-index for fill charge grid part

// this is defined in gpupme_mother.cu
//#define FIXEDPOINT 10000.f
//#define FIXEDPOINT 100000.f
//#define FIXEDPOINT 300000.f
//#define FIXEDPOINT 1000000.f
//#define FIXEDPOINT 3000000.f
//#define FIXEDPOINT 10000000.f
//#define FIXEDPOINT 30000000.f

//#define BLOCKSIZE 64
//#define BLOCKSIZE 96 //not work
#define BLOCKSIZE 128
//#define BLOCKSIZE 256

#define BSIZEPME1 BLOCKSIZE
#define BSIZEPME2 BLOCKSIZE
#define BVDWDIRC BLOCKSIZE
#define BVDWCELL BLOCKSIZE
#define BEWRDIRC BLOCKSIZE
#define BEWRCELL BLOCKSIZE

#if 0 // original
#define BSIZEPME1 256
#define BSIZEPME2 256
#define BVDWDIRC 256
#define BVDWCELL 256
#define BEWRDIRC 256
#define BEWRCELL 256
#endif

#ifdef CELLINDEX
#define GPU_ALLOCATE_AND_COPY8(TYPE,HOST,DEV,MEM)			\
  TYPE* DEV;								\
  CUDA_SAFE_CALL(cudaMalloc((void**)&DEV, (MEM)*8));			\
  for(int i=0;i<8;i++) CUDA_SAFE_CALL(cudaMemcpy(DEV+(MEM)*i, HOST, MEM, cudaMemcpyHostToDevice));
#endif
#define GPU_ALLOCATE_AND_COPY(TYPE,HOST,DEV,MEM)			\
  TYPE* DEV;								\
  /*usleep(100000);*/\
  CUDA_SAFE_CALL(cudaMalloc((void**)&DEV, MEM));	\
  CHECK_ERROR("  cudaMalloc pointer");\
  /*fprintf(stderr,"    DEV=%x MEM=%d\n",DEV,MEM);*/			\
  CUDA_SAFE_CALL(cudaMemcpy(DEV, HOST, MEM, cudaMemcpyHostToDevice));\
  CHECK_ERROR("  cudaMemcpy host->device");
#define INIT_AND_ALLOCATE(TYPE,NAME,MEM,SIZE)		\
  unsigned int MEM = sizeof(TYPE) * SIZE;	\
  TYPE* NAME = (TYPE*) malloc(MEM);
#define SET_FLOAT4_ZERO(f4)				\
  f4.x = 0.f; f4.y = 0.f; f4.z = 0.f; f4.w = 0.f;
#define SET_FLOAT2_ZERO(f2)				\
  f2.x = 0.f; f2.y = 0.f;
#define SET_DUMMY_VDWATM(va,iatm)			\
  va.x = 0.f; va.y = 0.f; va.z = 0.f; va.a = iatm;
#define SET_DUMMY_VDWATMEX(va,iatm)					\
  va.x = 0.f; va.y = 0.f; va.z = 0.f; va.a = (short)iatm; va.m = 0;
#define COPY_DOUBLE_TO_VDWATM(vi,xi,ai,iatm,jatm)	\
  vi[iatm].x = (float)xi[jatm*3+0];			\
  vi[iatm].y = (float)xi[jatm*3+1];			\
  vi[iatm].z = (float)xi[jatm*3+2];			\
  vi[iatm].a = ai[jatm]-1;			
#define COPY_DOUBLE_TO_COULOMB(vi,xi,ai,iatm,jatm)	\
  vi[iatm].x = (float)xi[jatm*3+0];			\
  vi[iatm].y = (float)xi[jatm*3+1];			\
  vi[iatm].z = (float)xi[jatm*3+2];			\
  vi[iatm].w = (float)ai[jatm];			
#define COPY_DOUBLE_TO_VDWATMEX(vi,xi,ai,mi,iatm,jatm)	\
  vi[iatm].x = (float)xi[jatm*3+0];			\
  vi[iatm].y = (float)xi[jatm*3+1];			\
  vi[iatm].z = (float)xi[jatm*3+2];			\
  vi[iatm].a = (short)ai[jatm]-1;			\
  vi[iatm].m = (unsigned short)mi[jatm]-1;			
#define COPY_SINGLE_TO_FORCEPOT(vi,xi,ai,iatm,jatm)			\
  xi[iatm*3+0] = (double)vi[jatm].x;					\
  xi[iatm*3+1] = (double)vi[jatm].y;					\
  xi[iatm*3+2] = (double)vi[jatm].z;					\
  ai[iatm]     = (double)vi[jatm].w;			
#define COPY_SINGLE_TO_FORCEPOT_DOUBLE(vi,xi,ai,iatm,jatm)		\
  xi[iatm*3+0] = (double)vi[jatm*2+0].x + (double)vi[jatm*2+0].y;	\
  xi[iatm*3+1] = (double)vi[jatm*2+0].z + (double)vi[jatm*2+0].w;	\
  xi[iatm*3+2] = (double)vi[jatm*2+1].x + (double)vi[jatm*2+1].y;	\
  ai[iatm]     = (double)vi[jatm*2+1].z + (double)vi[jatm*2+1].w;	
#define COPY_SINGLE_TO_STRESS(vi,xi,iatm,jatm)				\
  xi[iatm*3+0] = (double)vi[jatm].x;					\
  xi[iatm*3+1] = (double)vi[jatm].y;					\
  xi[iatm*3+2] = (double)vi[jatm].z;					
#define COPY_DOUBLE_TO_VDWCOEF(gr_float,epsilon4,sigma2,itmp,itmp1)	\
  gr_float[itmp].x = \
    (float)(epsilon4[itmp1]/					\
	    (sigma2[itmp1]*sigma2[itmp1]*sigma2[itmp1]		\
	     *sigma2[itmp1]*sigma2[itmp1]*sigma2[itmp1]));	\
  gr_float[itmp].y =							\
    (float)(epsilon4[itmp1]/(sigma2[itmp1]*sigma2[itmp1]*sigma2[itmp1])); \
  gr_float[itmp].z = gr_float[itmp].x*12.f;				\
  gr_float[itmp].w = gr_float[itmp].y*6.f;
#define SET_CELL_IMAGE(dim,itmp31,num_div_cell,itmp32,num_cell_real,ftmp3,num_cell_psd) \
  if (itmp31[dim] < num_div_cell){					\
    itmp32[dim] = itmp31[dim] + num_cell_real[dim] - num_div_cell;	\
    ftmp3[dim] = 1.f;							\
  }else if (itmp31[dim] > num_cell_psd[dim] - num_div_cell - 1){	\
    itmp32[dim] = itmp31[dim] - num_cell_real[dim] - num_div_cell;	\
    ftmp3[dim] = -1.f;							\
  }else{								\
    itmp32[dim] = itmp31[dim] - num_div_cell;				\
    ftmp3[dim] = 0.f;							\
  }
#define GET_CELL_DISTANCE(dim,itmp3,dtmp3,cell_width)			\
  if (itmp3[dim] < 0){							\
    dtmp3[dim] = ((double)(itmp3[dim]+1))*cell_width[dim];		\
  }else if (itmp3[dim] > 0){						\
    dtmp3[dim] = ((double)(itmp3[dim]-1))*cell_width[dim];		\
  }else{								\
    dtmp3[dim] = 0.e0;							\
  }
#define GET_ROUND_CELL(itmp1,itmp3,num_div_cell,dtmp3,cell_width,tmp2,radius_cut,tmp_round) \
  itmp1 = 1;								\
  tmp2 = radius_cut*radius_cut;						\
  tmp_round[0] = 0;							\
  tmp_round[1] = 0;							\
  tmp_round[2] = 0;							\
  for (itmp3[0] = -num_div_cell; itmp3[0] < num_div_cell+1; itmp3[0]++){ \
    GET_CELL_DISTANCE(0,itmp3,dtmp3,cell_width)				\
      dtmp3[0] *= dtmp3[0];						\
    for (itmp3[1] = -num_div_cell; itmp3[1] < num_div_cell+1; itmp3[1]++){ \
      GET_CELL_DISTANCE(1,itmp3,dtmp3,cell_width)			\
	dtmp3[1] *= dtmp3[1];						\
      for (itmp3[2] = -num_div_cell; itmp3[2] < num_div_cell+1; itmp3[2]++){ \
	GET_CELL_DISTANCE(2,itmp3,dtmp3,cell_width)			\
	  dtmp3[2] *= dtmp3[2];						\
	if (dtmp3[0] + dtmp3[1] + dtmp3[2] < tmp2){			\
	  if (!(itmp3[0] == 0 && itmp3[1] == 0 && itmp3[2] == 0)){	\
	    tmp_round[itmp1*3+0] = itmp3[0];				\
	    tmp_round[itmp1*3+1] = itmp3[1];				\
	    tmp_round[itmp1*3+2] = itmp3[2];				\
	    itmp1 += 1;							\
	  }								\
	}								\
      }									\
    }									\
  }
#define GET_3DIM_TO_1DIM(itmp1,itmp3,num_cell)				\
    itmp1 = (itmp3[2] * num_cell[1] + itmp3[1]) * num_cell[0] + itmp3[0];
#define SET_POS_TO_REGION(dim,iatm,jatm,xmax,xout,xin)			\
  if (xin[jatm*3+dim] > xmax[dim]){					\
    xout[iatm*3+dim] = (float)(xin[jatm*3+dim]-xmax[dim]);		\
  }else if (xin[jatm*3+dim] < 0.e0){					\
    xout[iatm*3+dim] = (float)(xin[jatm*3+dim]+xmax[dim]);		\
  }else{								\
    xout[iatm*3+dim] = (float)(xin[jatm*3+dim]);			\
  }
#define ADD_DOUBLEFLOAT(tmp0,tmp1,tmp2,tmp3,tmph,tmpl)		\
	tmp1 = tmph + tmp0;\
	tmp2 = tmp1 - tmph;\
	tmp3 = tmp1 - tmp2;\
	tmp3 = tmph - tmp3;\
	tmp2 = tmp0 - tmp2;\
	tmp3 = tmp2 + tmp3;\
	tmp3 = tmp3 + tmpl;\
	tmph = tmp1 + tmp3;\
	tmpl = tmph - tmp1;\
	tmpl = tmp3 - tmpl;
#define ROLLING_FORWARD(itmp1,itmp2,itmp3)			\
  itmp1  = ((int)((float)(itmp2-1)/(float)itmp3)+1)*itmp3;
#define RPI 0.31830988618e0   // 1 / pi
#define SQPI 9.86960440109e0   // pi^2
#define RSQRTPI2   1.12837916709e0   // 2 / sqrt(pi)
#define SET_XMAX_FLOAT(xmax_float0,xmax_float1,xmax_float2,xdouble)	\
  xmax_float0 = (float)xdouble[0];					\
  xmax_float1 = (float)xdouble[1];					\
  xmax_float2 = (float)xdouble[2];					
#define SET_RCP_XMAX(rcp_xmax0,rcp_xmax1,rcp_xmax2,xdouble)	\
  rcp_xmax0 = (float)(1.e0/xdouble[0]);				\
  rcp_xmax1 = (float)(1.e0/xdouble[1]);				\
  rcp_xmax2 = (float)(1.e0/xdouble[2]);
#define GET_SQUARE_AND_R2(dx)			\
  dx[0] *= dx[0];				\
  dx[1] *= dx[1];				\
  dx[2] *= dx[2];				\
  dx[3] = dx[0] + dx[1] + dx[2];

struct __align__(16) vdwatm {float x;  float y;  float z;  int a;};
struct __align__(16) vdwatmex{float x;  float y;  float z; short a; unsigned short m;};
