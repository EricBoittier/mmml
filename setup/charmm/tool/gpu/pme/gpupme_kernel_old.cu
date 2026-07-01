#if 0 // not work, too much shared memory
#define QS_SIZEX 16
#define QS_SIZEY 16
#define QS_SIZEZ 16
#elif 0 // supports 9 grids per block and order=6, work
#define QS_SIZEX 15
#define QS_SIZEY 15
#define QS_SIZEZ 15
#else // supports 8 grids per block and order=6, work
#define QS_SIZEX 14
#define QS_SIZEY 14
#define QS_SIZEZ 14
#endif
#define QS_SIZE  QS_SIZEX*QS_SIZEY*QS_SIZEZ

texture<float2,1,cudaReadModeElementType> tex;
extern "C"
__global__ void gpupme1_kernel
(float* q_dev, float* bx_dev, float* by_dev, float* bz_dev, int sizex,
 int sizey, int sizez, float* recipf_dev, float factor, float density)
{
  float m2, tmp, eterm;
  float mx, my, mz, q[2];
  //int nx = blockIdx.x * BSIZEPME1 + threadIdx.x;
  //int ny = blockIdx.y * BSIZEPME1 + threadIdx.y;
  //int nz = blockIdx.z * BSIZEPME1 + threadIdx.z;
  //int num = nx + ny * sizex + nz * sizex * sizey;
  int num = blockIdx.x * BSIZEPME1 + threadIdx.x;
  int nz = (int)((float)(num)/(float)(sizex*sizey));
  int ny = (int)((float)(num-nz*sizex*sizey)/(float)(sizex));
  int nx = num-nz*sizex*sizey-ny*sizex;
  if (2*nx > sizex){ mx = (float)(nx-sizex);
  }else{             mx = (float)(nx);      }
  if (2*ny > sizey){ my = (float)(ny-sizey);
  }else{             my = (float)(ny);      }
  if (2*nz > sizez){ mz = (float)(nz-sizez);
  }else{             mz = (float)(nz);      }
  q[0] = q_dev[num*2+0];
  q[1] = q_dev[num*2+1];
  tmp = mx*recipf_dev[0]+my*recipf_dev[3]+mz*recipf_dev[6];
  m2 = tmp*tmp;
  tmp = mx*recipf_dev[1]+my*recipf_dev[4]+mz*recipf_dev[7];
  m2 += tmp*tmp;
  tmp = mx*recipf_dev[2]+my*recipf_dev[5]+mz*recipf_dev[8];
  m2 += tmp*tmp;
  if (num > 0 && num < sizex*sizey*sizez){
    tmp = 1.f/m2;
  }else{
    tmp = 0.f;    
  }
  tmp *= density;
  tmp *= bx_dev[nx];
  tmp *= by_dev[ny];
  tmp *= bz_dev[nz];
  eterm = exp(-factor*m2)*tmp; 
  q_dev[num*2+0] = q[0] * eterm;
  q_dev[num*2+1] = q[1] * eterm;
}

extern "C"
__global__ void gpupme1gradsum_kernel
(float4* x, float4* f, float* q, int sizex, int sizey, int sizez,
 float* recipf_dev, int order, float rorder,
 int (*cpb)[4], int gridperblockx, int gridperblocky, int gridperblockz)
{
  float ui[3], qtmp, utmp1, utmp0;
  float4 ftmp, xi;
  float2 ttmp0, ttmp1, ttmp2;
  int mi[3], mtmp1, mtmp0, mtmp2;
  int cpt=cpb[blockIdx.x][3] & 0xf;
  int cpbone[3]={(cpb[blockIdx.x][0] & 0xff)*gridperblockx,
		 (cpb[blockIdx.x][1] & 0xff)*gridperblocky,
		 (cpb[blockIdx.x][2] & 0xff)*gridperblockz};
  int i,cptoffset=cpt*sizex*sizey*sizez*2;
  int miorg[3],mioffset[3];
#ifdef MULTIPLE_IBLOCK
  int fb=cpb[blockIdx.x][0]>>8,nb=(cpb[blockIdx.x][1]>>8)-fb;
  int num = fb * BSIZEPME2 + threadIdx.x;
#else
  int num = blockIdx.x * BSIZEPME2 + threadIdx.x;
#endif
#define USE_SHARED_FOR_GRADSUM

  for(int k=0;k<3;k++) miorg[k]=cpbone[k];

#ifdef USE_SHARED_FOR_GRADSUM
  __shared__ float qs[QS_SIZE];
#if 0 // try to optimize, not work
  unsigned int ti;
  ti=threadIdx.x & 0xf;
  for(i=(threadIdx.x>>4)<<4;i<QS_SIZE;i+=BSIZEPME2){
    if(i<QS_SIZE){
      mtmp1=(i/QS_SIZEX) % QS_SIZEY;
      mtmp2=(i/QS_SIZEX/QS_SIZEY) % QS_SIZEZ;
      if(ti<QS_SIZEX){
	mtmp0=ti;
	mtmp0+=miorg[0];if(mtmp0>=sizex) mtmp0-=sizex;
	mtmp1+=miorg[1];if(mtmp1>=sizey) mtmp1-=sizey;
	mtmp2+=miorg[2];if(mtmp2>=sizez) mtmp2-=sizez;
	qs[i+ti]=q[2*(mtmp0+sizex*(mtmp1+sizey*mtmp2))];
      }
      __syncthreads();
    }
  }
  __syncthreads();
#else // work
  for(i=threadIdx.x;i<QS_SIZE;i+=BSIZEPME2){
    mtmp0=i % QS_SIZEX;
    mtmp1=(i/QS_SIZEX) % QS_SIZEY;
    mtmp2=(i/QS_SIZEX/QS_SIZEY) % QS_SIZEZ;
    mtmp0+=miorg[0];if(mtmp0>=sizex) mtmp0-=sizex;
    mtmp1+=miorg[1];if(mtmp1>=sizey) mtmp1-=sizey;
    mtmp2+=miorg[2];if(mtmp2>=sizez) mtmp2-=sizez;
    qs[i]=q[2*(mtmp0+sizex*(mtmp1+sizey*mtmp2))];
  }
  __syncthreads();
#endif
#endif

#ifdef MULTIPLE_IBLOCK
  for(int b=0;b<nb;b++,num+=BSIZEPME2){
#endif
  SET_FLOAT4_ZERO(ftmp);
  xi = x[num];
  if(1 && xi.w!=0.0f){
    //  if(1){
#if defined(CELLINDEX) && 1
    ui[0] = __fmul_rn(xi.x,recipf_dev[0])+__fmul_rn(xi.y,recipf_dev[1])+__fmul_rn(xi.z,recipf_dev[2]);
    ui[1] = __fmul_rn(xi.x,recipf_dev[3])+__fmul_rn(xi.y,recipf_dev[4])+__fmul_rn(xi.z,recipf_dev[5]);
    ui[2] = __fmul_rn(xi.x,recipf_dev[6])+__fmul_rn(xi.y,recipf_dev[7])+__fmul_rn(xi.z,recipf_dev[8]);
#else
    ui[0] = xi.x*recipf_dev[0]+xi.y*recipf_dev[1]+xi.z*recipf_dev[2];
    ui[1] = xi.x*recipf_dev[3]+xi.y*recipf_dev[4]+xi.z*recipf_dev[5];
    ui[2] = xi.x*recipf_dev[6]+xi.y*recipf_dev[7]+xi.z*recipf_dev[8];
#endif
    ui[0] *= (float)sizex;
    ui[1] *= (float)sizey;
    ui[2] *= (float)sizez;
    mi[0] = (int)ui[0];
    mi[1] = (int)ui[1];
    mi[2] = (int)ui[2];
    ui[0] = - ui[0] + (float)mi[0];
    ui[1] = - ui[1] + (float)mi[1];
    ui[2] = - ui[2] + (float)mi[2];
    mi[0] += sizex - order;
    mi[1] += sizey - order;
    mi[2] += sizez - order;
    //    if(xi.w==0.0f) for(int k=0;k<3;k++) mi[k]=miorg[k]-1;
#ifdef USE_SHARED_FOR_GRADSUM // minimum image conversion
    while(mi[0]-miorg[0]>=sizex/2) mi[0]-=sizex;
    while(mi[1]-miorg[1]>=sizey/2) mi[1]-=sizey;
    while(mi[2]-miorg[2]>=sizez/2) mi[2]-=sizez;
    while(mi[0]-miorg[0]<-sizex/2) mi[0]+=sizex;
    while(mi[1]-miorg[1]<-sizey/2) mi[1]+=sizey;
    while(mi[2]-miorg[2]<-sizez/2) mi[2]+=sizez;
#else // added to ensure overflow of grids
    if(mi[0]+1>=sizex) mi[0]-=sizex;
    if(mi[1]+1>=sizey) mi[1]-=sizey;
    if(mi[2]+1>=sizez) mi[2]-=sizez;
#endif
    for (int k = 0; k < order; k++){
      ui[2] += 1.f;
      ttmp2 = tex1D(tex,ui[2]*rorder);
      mi[2] += 1;
#ifndef USE_SHARED_FOR_GRADSUM
      if (mi[2] >= sizez) mi[2] -= sizez;
#endif
      mioffset[2]=mi[2]-miorg[2];//if(mioffset[2]<0) mioffset[2]+=sizez;if(mioffset[2]>=sizez) mioffset[2]-=sizez;
      utmp1 = ui[1];
      mtmp1 = mi[1];
      for (int j = 0; j < order; j++){
	utmp1 += 1.f;
	ttmp1 = tex1D(tex,utmp1*rorder);
	mtmp1 += 1;
#ifndef USE_SHARED_FOR_GRADSUM
	if (mtmp1 >= sizey) mtmp1 -= sizey;
#endif
	mioffset[1]=mtmp1-miorg[1];//if(mioffset[1]<0) mioffset[1]+=sizey;if(mioffset[1]>=sizey) mioffset[1]-=sizey;
	utmp0 = ui[0];
	mtmp0 = mi[0];
	for (int i = 0; i < order; i++){
	  utmp0 += 1.f;
	  ttmp0 = tex1D(tex,utmp0*rorder);
	  mtmp0 += 1;
#ifndef USE_SHARED_FOR_GRADSUM
	  if (mtmp0 >= sizex) mtmp0 -= sizex;
#endif
	  mioffset[0]=mtmp0-miorg[0];//if(mioffset[0]<0) mioffset[0]+=sizex;if(mioffset[0]>=sizex) mioffset[0]-=sizex;
#ifdef USE_SHARED_FOR_GRADSUM
	  qtmp=qs[mioffset[0]+QS_SIZEX*(mioffset[1]+QS_SIZEY*mioffset[2])];
#else
	  qtmp = q[2*(mtmp0+sizex*(mtmp1+sizey*mi[2]))];
#endif
	  ftmp.x += qtmp*ttmp0.y*ttmp1.x*ttmp2.x;
	  ftmp.y += qtmp*ttmp0.x*ttmp1.y*ttmp2.x;
	  ftmp.z += qtmp*ttmp0.x*ttmp1.x*ttmp2.y;
	  ftmp.w += qtmp*ttmp0.x*ttmp1.x*ttmp2.x;
	}
      }
    }
    ftmp.x *= (float)sizex;
    ftmp.y *= (float)sizey;
    ftmp.z *= (float)sizez;
  }
  f[num].x = xi.w*(ftmp.x*recipf_dev[0]+ftmp.y*recipf_dev[3]+ftmp.z*recipf_dev[6]);
  f[num].y = xi.w*(ftmp.x*recipf_dev[1]+ftmp.y*recipf_dev[4]+ftmp.z*recipf_dev[7]);
  f[num].z = xi.w*(ftmp.x*recipf_dev[2]+ftmp.y*recipf_dev[5]+ftmp.z*recipf_dev[8]);
  f[num].w = xi.w*ftmp.w;
#ifdef MULTIPLE_IBLOCK
  }
#endif
}


extern "C"
__global__ void gpupme1gradsum_kernel_org
(float4* x, float4* f, float* q, int sizex, int sizey, int sizez,
 float* recipf_dev, int order, float rorder)
{
  int num = blockIdx.x * BSIZEPME2 + threadIdx.x;
  float ui[3], qtmp, utmp1, utmp0;
  float4 ftmp, xi;
  float2 ttmp0, ttmp1, ttmp2;
  int mi[3], mtmp1, mtmp0;
    
  SET_FLOAT4_ZERO(ftmp);
  xi = x[num];
  ui[0] = xi.x*recipf_dev[0]+xi.y*recipf_dev[1]+xi.z*recipf_dev[2];
  ui[1] = xi.x*recipf_dev[3]+xi.y*recipf_dev[4]+xi.z*recipf_dev[5];
  ui[2] = xi.x*recipf_dev[6]+xi.y*recipf_dev[7]+xi.z*recipf_dev[8];
  ui[0] *= (float)sizex;
  ui[1] *= (float)sizey;
  ui[2] *= (float)sizez;
  mi[0] = (int)ui[0];
  mi[1] = (int)ui[1];
  mi[2] = (int)ui[2];
  ui[0] = - ui[0] + (float)mi[0];
  ui[1] = - ui[1] + (float)mi[1];
  ui[2] = - ui[2] + (float)mi[2];
  mi[0] += sizex - order;
  mi[1] += sizey - order;
  mi[2] += sizez - order;
#if 1 // added to ensure overflow of grids
  if(mi[0]+1>=sizex) mi[0]-=sizex;
  if(mi[1]+1>=sizey) mi[1]-=sizey;
  if(mi[2]+1>=sizez) mi[2]-=sizez;
#endif
  for (int k = 0; k < order; k++){
    ui[2] += 1.f;
    ttmp2 = tex1D(tex,ui[2]*rorder);
    mi[2] += 1;
    if (mi[2] >= sizez) mi[2] -= sizez;
    utmp1 = ui[1];
    mtmp1 = mi[1];
    for (int j = 0; j < order; j++){
      utmp1 += 1.f;
      ttmp1 = tex1D(tex,utmp1*rorder);
      mtmp1 += 1;
      if (mtmp1 >= sizey) mtmp1 -= sizey;
      utmp0 = ui[0];
      mtmp0 = mi[0];
      for (int i = 0; i < order; i++){
	utmp0 += 1.f;
	ttmp0 = tex1D(tex,utmp0*rorder);
	mtmp0 += 1;
	if (mtmp0 >= sizex) mtmp0 -= sizex;
	qtmp = q[2*(mtmp0+sizex*(mtmp1+sizey*mi[2]))];
	ftmp.x += qtmp*ttmp0.y*ttmp1.x*ttmp2.x;
	ftmp.y += qtmp*ttmp0.x*ttmp1.y*ttmp2.x;
	ftmp.z += qtmp*ttmp0.x*ttmp1.x*ttmp2.y;
	ftmp.w += qtmp*ttmp0.x*ttmp1.x*ttmp2.x;
      }
    }
  }
  ftmp.x *= (float)sizex;
  ftmp.y *= (float)sizey;
  ftmp.z *= (float)sizez;
  f[num].x = xi.w*(ftmp.x*recipf_dev[0]+ftmp.y*recipf_dev[3]+ftmp.z*recipf_dev[6]);
  f[num].y = xi.w*(ftmp.x*recipf_dev[1]+ftmp.y*recipf_dev[4]+ftmp.z*recipf_dev[7]);
  f[num].z = xi.w*(ftmp.x*recipf_dev[2]+ftmp.y*recipf_dev[5]+ftmp.z*recipf_dev[8]);
  f[num].w = xi.w*ftmp.w;
}


extern "C"
__global__ void gpupme1gradsum_kernel_simple
(float4* x, float4* f, float* q, int sizex, int sizey, int sizez,
 float* recipf_dev, int order, float rorder)
{
  float ui[3], qtmp, utmp1, utmp0;
  float4 ftmp, xi;
  float2 ttmp0, ttmp1, ttmp2;
  int mi[3], mtmp1, mtmp0, mtmp2;
  int i;
  int num = blockIdx.x * BSIZEPME2 + threadIdx.x;

  SET_FLOAT4_ZERO(ftmp);
  xi = x[num];
  if(1 && xi.w!=0.0f){
    //  if(1){
#if defined(CELLINDEX) && 1
    ui[0] = __fmul_rn(xi.x,recipf_dev[0])+__fmul_rn(xi.y,recipf_dev[1])+__fmul_rn(xi.z,recipf_dev[2]);
    ui[1] = __fmul_rn(xi.x,recipf_dev[3])+__fmul_rn(xi.y,recipf_dev[4])+__fmul_rn(xi.z,recipf_dev[5]);
    ui[2] = __fmul_rn(xi.x,recipf_dev[6])+__fmul_rn(xi.y,recipf_dev[7])+__fmul_rn(xi.z,recipf_dev[8]);
#else
    ui[0] = xi.x*recipf_dev[0]+xi.y*recipf_dev[1]+xi.z*recipf_dev[2];
    ui[1] = xi.x*recipf_dev[3]+xi.y*recipf_dev[4]+xi.z*recipf_dev[5];
    ui[2] = xi.x*recipf_dev[6]+xi.y*recipf_dev[7]+xi.z*recipf_dev[8];
#endif
    ui[0] *= (float)sizex;
    ui[1] *= (float)sizey;
    ui[2] *= (float)sizez;
    mi[0] = (int)ui[0];
    mi[1] = (int)ui[1];
    mi[2] = (int)ui[2];
    ui[0] = - ui[0] + (float)mi[0];
    ui[1] = - ui[1] + (float)mi[1];
    ui[2] = - ui[2] + (float)mi[2];
    mi[0] += sizex - order;
    mi[1] += sizey - order;
    mi[2] += sizez - order;
    if(mi[0]+1>=sizex) mi[0]-=sizex;
    if(mi[1]+1>=sizey) mi[1]-=sizey;
    if(mi[2]+1>=sizez) mi[2]-=sizez;
    for (int k = 0; k < order; k++){
      ui[2] += 1.f;
      ttmp2 = tex1D(tex,ui[2]*rorder);
      mi[2] += 1;
      //      if (mi[2] >= sizez) mi[2] -= sizez;
      mi[2]=mi[2] % sizez;
      utmp1 = ui[1];
      mtmp1 = mi[1];
      for (int j = 0; j < order; j++){
	utmp1 += 1.f;
	ttmp1 = tex1D(tex,utmp1*rorder);
	mtmp1 += 1;
	//	if (mtmp1 >= sizey) mtmp1 -= sizey;
	mtmp1=mtmp1 % sizey;
	utmp0 = ui[0];
	mtmp0 = mi[0];
	for (int i = 0; i < order; i++){
	  utmp0 += 1.f;
	  ttmp0 = tex1D(tex,utmp0*rorder);
	  mtmp0 += 1;
	  //	  if (mtmp0 >= sizex) mtmp0 -= sizex;
	  mtmp0=mtmp0 % sizex;
	  qtmp = q[2*(mtmp0+sizex*(mtmp1+sizey*mi[2]))];
	  ftmp.x += qtmp*ttmp0.y*ttmp1.x*ttmp2.x;
	  ftmp.y += qtmp*ttmp0.x*ttmp1.y*ttmp2.x;
	  ftmp.z += qtmp*ttmp0.x*ttmp1.x*ttmp2.y;
	  ftmp.w += qtmp*ttmp0.x*ttmp1.x*ttmp2.x;
	}
      }
    }
    ftmp.x *= (float)sizex;
    ftmp.y *= (float)sizey;
    ftmp.z *= (float)sizez;
  }
  f[num].x = xi.w*(ftmp.x*recipf_dev[0]+ftmp.y*recipf_dev[3]+ftmp.z*recipf_dev[6]);
  f[num].y = xi.w*(ftmp.x*recipf_dev[1]+ftmp.y*recipf_dev[4]+ftmp.z*recipf_dev[7]);
  f[num].z = xi.w*(ftmp.x*recipf_dev[2]+ftmp.y*recipf_dev[5]+ftmp.z*recipf_dev[8]);
  f[num].w = xi.w*ftmp.w;
}


__device__ void gpupme1fillch_kernel_sub
(float4* x, int* q, int sizex, int sizey, int sizez,
 float* recipf_dev, int order, float rorder)
{
  int num = blockIdx.x * BSIZEPME2 + threadIdx.x;
  float ui[3], utmp1, utmp0;
  float4 xi;
  float2 ttmp0, ttmp1, ttmp2;
  int mi[3], mtmp1, mtmp0;

  //  if(num>=BSIZEPME2) return;
  //  if(num>0) return;
  //  if(!(num>=BSIZEPME2*7 && num<BSIZEPME2*8)) return;
  //  if(!(num>=BSIZEPME2*0 && num<BSIZEPME2*1)) return;
  xi = x[num];
  //  if(threadIdx.x!=14) return;
  ui[0] = xi.x*recipf_dev[0]+xi.y*recipf_dev[1]+xi.z*recipf_dev[2];
  ui[1] = xi.x*recipf_dev[3]+xi.y*recipf_dev[4]+xi.z*recipf_dev[5];
  ui[2] = xi.x*recipf_dev[6]+xi.y*recipf_dev[7]+xi.z*recipf_dev[8];
  ui[0] *= (float)sizex;
  ui[1] *= (float)sizey;
  ui[2] *= (float)sizez;
  mi[0] = (int)ui[0];
  mi[1] = (int)ui[1];
  mi[2] = (int)ui[2];
  ui[0] = - ui[0] + (float)mi[0];
  ui[1] = - ui[1] + (float)mi[1];
  ui[2] = - ui[2] + (float)mi[2];
  mi[0] += sizex - order;
  mi[1] += sizey - order;
  mi[2] += sizez - order;
#if 1 // added to ensure overflow of grids
  if(mi[0]+1>=sizex) mi[0]-=sizex;
  if(mi[1]+1>=sizey) mi[1]-=sizey;
  if(mi[2]+1>=sizez) mi[2]-=sizez;
#endif
  for (int k = 0; k < order; k++){
    //  for (int k = 0; k < 1; k++){
    ui[2] += 1.f;
    ttmp2 = tex1D(tex,ui[2]*rorder);
    mi[2] += 1;
    if (mi[2] >= sizez) mi[2] -= sizez;
    utmp1 = ui[1];
    mtmp1 = mi[1];
    for (int j = 0; j < order; j++){
      //    for (int j = 0; j < 2; j++){
      utmp1 += 1.f;
      ttmp1 = tex1D(tex,utmp1*rorder);
      mtmp1 += 1;
      if (mtmp1 >= sizey) mtmp1 -= sizey;
      utmp0 = ui[0];
      mtmp0 = mi[0];
      for (int i = 0; i < order; i++){
	//      for (int i = 0; i < 2; i++){
	utmp0 += 1.f;
	ttmp0 = tex1D(tex,utmp0*rorder);
	mtmp0 += 1;
	if (mtmp0 >= sizex) mtmp0 -= sizex;
	//	q[(0+i+j*2)*2]=mtmp0;q[(4+i+j*2)*2]=mtmp1;q[(8+i+j*2)*2]=mi[2];q[12*2]=sizex;q[13*2]=sizey;q[14*2]=sizez;
	//        printf("threadIdx.x=%d mtmp0=%d mtmp1=%d mi[2]=%d\n",threadIdx.x,mtmp0,mtmp1,mi[2]);
	atomicAdd(&q[2*(mtmp0+sizex*(mtmp1+sizey*mi[2]))],
		  (int)(xi.w*ttmp0.x*ttmp1.x*ttmp2.x*FIXEDPOINT));
      }
    }
  }
}

extern "C"
__global__ void gpupme1fillch_kernel
(float4* x, int* q, int* q2, int sizex, int sizey, int sizez,
 float* recipf_dev, int order, float rorder,
 int (*cpb)[4], int gridperblockx, int gridperblocky, int gridperblockz)
{
  float ui[3], utmp1, utmp0;
  float4 xi;
  float2 ttmp0, ttmp1, ttmp2;
  int mi[3], mtmp1, mtmp0, mtmp2;
  __shared__ int qs[QS_SIZE];
#ifdef MULTIPLE_IBLOCK_TEST2
  int bidx2=cpb[blockIdx.x][0]>>8;
  int cpt=cpb[bidx2][3] & 0xf;
  int cpbone[3]={(cpb[bidx2][0] & 0xff)*gridperblockx,
		 (cpb[bidx2][1] & 0xff)*gridperblocky,
		 (cpb[bidx2][2] & 0xff)*gridperblockz};
#else
  int cpt=cpb[blockIdx.x][3] & 0xf;
  int cpbone[3]={(cpb[blockIdx.x][0] & 0xff)*gridperblockx,
		 (cpb[blockIdx.x][1] & 0xff)*gridperblocky,
		 (cpb[blockIdx.x][2] & 0xff)*gridperblockz};
#endif
  int i,cptoffset=cpt*sizex*sizey*sizez*2;
  int miorg[3],mioffset[3];
#ifdef MULTIPLE_IBLOCK
  int fb=cpb[blockIdx.x][0]>>8,nb=(cpb[blockIdx.x][1]>>8)-fb;
  int num = fb * BSIZEPME2 + threadIdx.x;
#elif defined(MULTIPLE_IBLOCK_TEST2)
  int num = bidx2 * BSIZEPME2 + threadIdx.x;
#else
  int num = blockIdx.x * BSIZEPME2 + threadIdx.x;
#endif

#if 0
  if(num>=4*BSIZEPME2) return;
#endif
#if 0
  if(blockIdx.x!=404) return;
#endif
#ifndef MULTIPLE_IBLOCK_TEST2
  if(cpt>=8){ // normal calculation with global atomicadd
#ifndef MULTIPLE_IBLOCK_TEST
    gpupme1fillch_kernel_sub(x,q,sizex,sizey,sizez,
                             recipf_dev,order,rorder);
#endif
    return;
  }
#endif // end of not MULTIPLE_IBLOCK_TEST2
  
  for(i=threadIdx.x;i<QS_SIZE;i+=BSIZEPME2) qs[i]=0;
  __syncthreads();

  for(int k=0;k<3;k++) miorg[k]=cpbone[k];
#ifdef MULTIPLE_IBLOCK
  for(int b=0;b<nb;b++,num+=BSIZEPME2){
#elif defined(MULTIPLE_IBLOCK_TEST)
#ifdef MULTIPLE_IBLOCK_TEST2
  for(int b=0;b<(cpb[bidx2][3]>>4);b++,num+=BSIZEPME2){
#else
  for(int b=0;b<(cpb[blockIdx.x][3]>>4);b++,num+=BSIZEPME2){
#endif
#endif
  xi = x[num];
  //  if(threadIdx.x!=14) goto skip_loop;
  if(1 && xi.w!=0.0f){
#if defined(CELLINDEX) && 1
    ui[0] = __fmul_rn(xi.x,recipf_dev[0])+__fmul_rn(xi.y,recipf_dev[1])+__fmul_rn(xi.z,recipf_dev[2]);
    ui[1] = __fmul_rn(xi.x,recipf_dev[3])+__fmul_rn(xi.y,recipf_dev[4])+__fmul_rn(xi.z,recipf_dev[5]);
    ui[2] = __fmul_rn(xi.x,recipf_dev[6])+__fmul_rn(xi.y,recipf_dev[7])+__fmul_rn(xi.z,recipf_dev[8]);
#else
    ui[0] = xi.x*recipf_dev[0]+xi.y*recipf_dev[1]+xi.z*recipf_dev[2];
    ui[1] = xi.x*recipf_dev[3]+xi.y*recipf_dev[4]+xi.z*recipf_dev[5];
    ui[2] = xi.x*recipf_dev[6]+xi.y*recipf_dev[7]+xi.z*recipf_dev[8];
#endif
    ui[0] *= (float)sizex;
    ui[1] *= (float)sizey;
    ui[2] *= (float)sizez;
    mi[0] = (int)ui[0];
    mi[1] = (int)ui[1];
    mi[2] = (int)ui[2];
    ui[0] = - ui[0] + (float)mi[0];
    ui[1] = - ui[1] + (float)mi[1];
    ui[2] = - ui[2] + (float)mi[2];
    mi[0] += sizex - order;
    mi[1] += sizey - order;
    mi[2] += sizez - order;
#if defined(CELLINDEX) && 1
    int qoffset;
    qoffset=miorg[0]+sizex*(miorg[1]+sizey*miorg[2]);
#endif

#if 1 // minimum image conversion
    while(mi[0]-miorg[0]>=sizex/2) mi[0]-=sizex;
    while(mi[1]-miorg[1]>=sizey/2) mi[1]-=sizey;
    while(mi[2]-miorg[2]>=sizez/2) mi[2]-=sizez;
    while(mi[0]-miorg[0]<-sizex/2) mi[0]+=sizex;
    while(mi[1]-miorg[1]<-sizey/2) mi[1]+=sizey;
    while(mi[2]-miorg[2]<-sizez/2) mi[2]+=sizez;
#else // not complete    
    if(mi[0]+1>=sizex) mi[0]-=sizex;
    if(mi[1]+1>=sizey) mi[1]-=sizey;
    if(mi[2]+1>=sizez) mi[2]-=sizez;
#endif
    if(num==BSIZEPME2*8-1 && 0){
      q2[6]=miorg[0];
      q2[8]=miorg[1];
      q2[10]=miorg[2];
      q2[12]=mi[0];
      q2[14]=mi[1];
      q2[16]=mi[2];
    }
    
    for (int k = 0; k < order; k++){
      //    for (int k = 0; k < 1; k++){
    //for (int k = 0; k < order && num==BSIZEPME2*8-1; k++){
    //  for (int k = 0; k < order && num>=BSIZEPME2*7 && num<BSIZEPME2*8; k++){
    //  for (int k = 0; k < order && num>=BSIZEPME2*0 && num<BSIZEPME2*1; k++){
    //    for (int k = 0; k < 1 && num==0; k++){
      ui[2] += 1.f;
      ttmp2 = tex1D(tex,ui[2]*rorder);
      mi[2] += 1;
      //    if (mi[2] >= sizez) mi[2] -= sizez;
      mioffset[2]=mi[2]-miorg[2];//if(mioffset[2]<0) mioffset[2]+=sizez;if(mioffset[2]>=sizez) mioffset[2]-=sizez;
      utmp1 = ui[1];
      mtmp1 = mi[1];
      for (int j = 0; j < order; j++){
	//for (int j = 0; j < 2; j++){
	utmp1 += 1.f;
	ttmp1 = tex1D(tex,utmp1*rorder);
	mtmp1 += 1;
	//	if (mtmp1 >= sizey) mtmp1 -= sizey;
	mioffset[1]=mtmp1-miorg[1];//if(mioffset[1]<0) mioffset[1]+=sizey;if(mioffset[1]>=sizey) mioffset[1]-=sizey;
	utmp0 = ui[0];
	mtmp0 = mi[0];
	for (int i = 0; i < order; i++){
	  //	for (int i = 0; i < 2; i++){
	  utmp0 += 1.f;
	  ttmp0 = tex1D(tex,utmp0*rorder);
	  mtmp0 += 1;
	  //	if (mtmp0 >= sizex) mtmp0 -= sizex;
	  mioffset[0]=mtmp0-miorg[0];//if(mioffset[0]<0) mioffset[0]+=sizex;if(mioffset[0]>=sizex) mioffset[0]-=sizex;
	  //	mtmp0 = mtmp0 % sizex;
	  
	  //	  atomicAdd(&q[2*(mtmp0+sizex*(mtmp1+sizey*mi[2]))],(int)(xi.w*ttmp0.x*ttmp1.x*ttmp2.x*FIXEDPOINT));
	  //	atomicAdd(&q2[cptoffset+2*(mtmp0+sizex*(mtmp1+sizey*mi[2]))],(int)(xi.w*ttmp0.x*ttmp1.x*ttmp2.x*FIXEDPOINT));
	  //	printf("cptoffset=%d %d\n",cptoffset,cptoffset/(sizex*sizey*sizez*2));
	  
	  //	atomicAdd(&q2[cptoffset+2*(qoffset+mioffset[0]+sizex*(mioffset[1]+sizey*mioffset[2]))],(int)(xi.w*ttmp0.x*ttmp1.x*ttmp2.x*FIXEDPOINT));
	  //	  q[(0+i+j*2)*2]=mtmp0;q[(4+i+j*2)*2]=mtmp1;q[(8+i+j*2)*2]=mi[2];q[12*2]=miorg[0];q[13*2]=miorg[1];q[14*2]=miorg[2];
	  //	  q[(0+i+j*2)*2]=mtmp0;q[(4+i+j*2)*2]=mtmp1;q[(8+i+j*2)*2]=mi[2];
	  //	  q[(0+i+j*2)*2]=mioffset[0];q[(4+i+j*2)*2]=mioffset[1];q[(8+i+j*2)*2]=mioffset[2];q[12*2]=miorg[0];q[13*2]=miorg[1];q[14*2]=miorg[2];q[15*2]=mi[0];q[16*2]=mi[1];q[17*2]=mi[2];
	  //	  printf("threadIdx.x=%d mtmp0=%d mtmp1=%d mi[2]=%d\n",threadIdx.x,mtmp0,mtmp1,mi[2]);
	  //	  printf("threadIdx.x=%d mioffset[0]=%d mioffset[1]=%d mioffset[2]=%d miorg=%d %d %d\n",threadIdx.x,mioffset[0],mioffset[1],mioffset[2],miorg[0],miorg[1],miorg[2]);
	  atomicAdd(&qs[mioffset[0]+QS_SIZEX*(mioffset[1]+QS_SIZEY*mioffset[2])],(int)(xi.w*ttmp0.x*ttmp1.x*ttmp2.x*FIXEDPOINT)); // work
	  if(0 && 
	     (mioffset[0]+QS_SIZEX*(mioffset[1]+QS_SIZEY*mioffset[2])>=QS_SIZE 
	      || mioffset[0]>=QS_SIZEX || mioffset[1]>=QS_SIZEY 
	      || mioffset[2]>=QS_SIZEZ
	      || mioffset[0]<0 || mioffset[1]<0 || mioffset[2]<0)){
	    q[0]=-999999;
	    q[2]=mioffset[0];
	    q[4]=mioffset[1];
	    q[6]=mioffset[2];
	    q[8]=num;
	  }
	  if(0 && num>=BSIZEPME2*7 && num<BSIZEPME2*8){
	    q[2*order*order*order*3+2*(3*(i+order*(j+order*k)))]=mioffset[0];
	    q[2*order*order*order*3+2*(3*(i+order*(j+order*k))+1)]=mioffset[1];
	    q[2*order*order*order*3+2*(3*(i+order*(j+order*k))+2)]=mioffset[2];
	  }
	}
      }
    }
  }
#if defined(MULTIPLE_IBLOCK) || defined(MULTIPLE_IBLOCK_TEST)
  }
#endif

 skip_loop:;    
  __syncthreads();
#if 0
  for(i=threadIdx.x;i<sizex*sizey*sizez;i+=BSIZEPME2){
    mtmp0=i % sizex;
    mtmp1=(i/sizex) % sizey;
    mtmp2=(i/sizex/sizey) % sizez;
    mtmp0+=miorg[0];if(mtmp0>=sizex) mtmp0-=sizex;
    mtmp1+=miorg[1];if(mtmp1>=sizey) mtmp1-=sizey;
    mtmp2+=miorg[2];if(mtmp2>=sizez) mtmp2-=sizez;
    q2[cptoffset+2*(mtmp0+sizex*(mtmp1+sizey*mtmp2))]=qs[i];
  }
#else // work
  for(i=threadIdx.x;i<QS_SIZE;i+=BSIZEPME2){
    mtmp0=i % QS_SIZEX;
    mtmp1=(i/QS_SIZEX) % QS_SIZEY;
    mtmp2=(i/QS_SIZEX/QS_SIZEY) % QS_SIZEZ;
    mtmp0+=miorg[0];if(mtmp0>=sizex) mtmp0-=sizex;
    mtmp1+=miorg[1];if(mtmp1>=sizey) mtmp1-=sizey;
    mtmp2+=miorg[2];if(mtmp2>=sizez) mtmp2-=sizez;
    q2[cptoffset+2*(mtmp0+sizex*(mtmp1+sizey*mtmp2))]=qs[i];
  }
#endif
  __syncthreads();
}

extern "C"
__global__ void gpupme1fillch_kernel_org
(float4* x, int* q, int sizex, int sizey, int sizez,
 float* recipf_dev, int order, float rorder)
{
  int num = blockIdx.x * BSIZEPME2 + threadIdx.x;
  float ui[3], utmp1, utmp0;
  float4 xi;
  float2 ttmp0, ttmp1, ttmp2;
  int mi[3], mtmp1, mtmp0;
    
  xi = x[num];
  ui[0] = xi.x*recipf_dev[0]+xi.y*recipf_dev[1]+xi.z*recipf_dev[2];
  ui[1] = xi.x*recipf_dev[3]+xi.y*recipf_dev[4]+xi.z*recipf_dev[5];
  ui[2] = xi.x*recipf_dev[6]+xi.y*recipf_dev[7]+xi.z*recipf_dev[8];
  ui[0] *= (float)sizex;
  ui[1] *= (float)sizey;
  ui[2] *= (float)sizez;
  mi[0] = (int)ui[0];
  mi[1] = (int)ui[1];
  mi[2] = (int)ui[2];
  ui[0] = - ui[0] + (float)mi[0];
  ui[1] = - ui[1] + (float)mi[1];
  ui[2] = - ui[2] + (float)mi[2];
  mi[0] += sizex - order;
  mi[1] += sizey - order;
  mi[2] += sizez - order;
#if 1 // added to ensure overflow of grids
  if(mi[0]+1>=sizex) mi[0]-=sizex;
  if(mi[1]+1>=sizey) mi[1]-=sizey;
  if(mi[2]+1>=sizez) mi[2]-=sizez;
#endif
  for (int k = 0; k < order; k++){
    //  for (int k = 0; k < 1; k++){
    ui[2] += 1.f;
    ttmp2 = tex1D(tex,ui[2]*rorder);
    mi[2] += 1;
    if (mi[2] >= sizez) mi[2] -= sizez;
    utmp1 = ui[1];
    mtmp1 = mi[1];
    for (int j = 0; j < order; j++){
      //    for (int j = 0; j < 2; j++){
      utmp1 += 1.f;
      ttmp1 = tex1D(tex,utmp1*rorder);
      mtmp1 += 1;
      if (mtmp1 >= sizey) mtmp1 -= sizey;
      utmp0 = ui[0];
      mtmp0 = mi[0];
      for (int i = 0; i < order; i++){
	//      for (int i = 0; i < 2; i++){
	utmp0 += 1.f;
	ttmp0 = tex1D(tex,utmp0*rorder);
	mtmp0 += 1;
	if (mtmp0 >= sizex) mtmp0 -= sizex;
	atomicAdd(&q[2*(mtmp0+sizex*(mtmp1+sizey*mi[2]))],
		  (int)(xi.w*ttmp0.x*ttmp1.x*ttmp2.x*FIXEDPOINT));
      }
    }
  }
}

extern "C"
__global__ void gpupme1exchange_kernel
(int*qm, int* qm2, float* q_dev, int sizex, int sizey, int sizez)
{
  //int nx = blockIdx.x * BSIZEPME1 + threadIdx.x;
  //int ny = blockIdx.y * BSIZEPME1 + threadIdx.y;
  //int nz = blockIdx.z * BSIZEPME1 + threadIdx.z;
  //int num = nx + ny * sizex + nz * sizex * sizey;
  int num = blockIdx.x * BSIZEPME1 + threadIdx.x;
#if defined(CELLINDEX) && 1
#if 1 // sum up by integer
  int sum=qm[num*2+0];
  for(int i=0;i<8;i++) sum += qm2[i*sizex*sizey*sizez*2+num*2+0];
  q_dev[num*2+0] = sum/FIXEDPOINT;
  q_dev[num*2+1] = 0.f;
#else // sum up by float
  q_dev[num*2+0] = qm2[num*2+0]/FIXEDPOINT;
  q_dev[num*2+1] = 0.f;
  for(int i=1;i<8;i++){
    q_dev[num*2+0] += qm2[i*sizex*sizey*sizez*2+num*2+0]/FIXEDPOINT;
  }
#endif
#else
  q_dev[num*2+0] = qm[num*2+0]/FIXEDPOINT;
  q_dev[num*2+1] = 0.f;
#endif
}

extern "C"
__global__ void gpupme1initializeq_kernel
(int* q_dev, int *q2_dev, int sizex, int sizey, int sizez)
{
  int num = blockIdx.x * BSIZEPME1 + threadIdx.x;
  int offset=sizex*sizey*sizez*2;
  q_dev[num*2+0] = 0;
  q_dev[num*2+1] = 0;
  for(int i=0;i<8;i++){
    __syncthreads();
    q2_dev[i*offset+num*2+0]=q2_dev[i*offset+num*2+1]=0;
  }
}

extern "C"
__global__ void gpupme1exchange_kernel_org
(int* qm, float* q_dev, int sizex, int sizey, int sizez)
{
  //int nx = blockIdx.x * BSIZEPME1 + threadIdx.x;
  //int ny = blockIdx.y * BSIZEPME1 + threadIdx.y;
  //int nz = blockIdx.z * BSIZEPME1 + threadIdx.z;
  //int num = nx + ny * sizex + nz * sizex * sizey;
  int num = blockIdx.x * BSIZEPME1 + threadIdx.x;
  q_dev[num*2+0] = qm[num*2+0]/FIXEDPOINT;
  q_dev[num*2+1] = 0.f;
}

