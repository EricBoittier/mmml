#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gpuheader.h>
#define BOLTZMANN  0.00831447f
#define DIELECTRIC 1389.354f // 1 / (4*pi*eps0)
#define RSQRTPI2   1.128379f   // 2 / sqrt(pi)
#define RSQRTPI    0.5641895f  // 1 / sqrt(pi)
#define PI         3.141593f
#define PI2        6.283186f    // 2 * pi
#define RCP_PI2    0.1591549f   // 1 / (2*pi)
#define NUM_THREAD 256
#define NUM_BLOCK  256
#define NUM_N1_BLOCK  256
#define NUM_N1_THREAD 256
#define NUM_SITE   3

extern "C"
void gpumd_();

struct mole{
  float x[3]; float v[3]; float f[3];
  float q[4]; float p[4]; float n[4];
};//old v hozon sureba kinetic energy motomaru

struct atm{
  float x[3];
  int   atype;
  int   id; // chain topology
  float epsilon; // sqrt of L-J epsilon
  float sigma;   // half of L-J sigma
  float charge;  // charge
  float pot;     // potential
  float f[3];
  float pressure;  // pressure
};

struct recip{
  float vec[3]; float coef; float pot; float sumcos; float sumsin; float pressure;
};

extern "C"
void get_cputime(double *laptime, double *sprittime)
{
  struct timeval tv;
  struct timezone tz;
  double sec,microsec;

  gettimeofday(&tv, &tz);
  sec=tv.tv_sec;
  microsec=tv.tv_usec;

  *sprittime = sec + microsec * 1e-6 - *laptime;
  *laptime = sec + microsec * 1e-6;
}


void setParamWater
(struct atm* atms, int num_mole, float* pos_site, float* mass_site,
 float* rcp_mass_mole, float* mass_mole, float* inertia_moment, int num_atype,
 float* sigma2, float* epsilon4, float* epsilon24)
{
  int itmp;
  for (int i = 0; i < num_mole; i++){
    itmp = i*3;
    atms[itmp].id        = (i+1);
    atms[itmp].atype     = 0;
    atms[itmp].epsilon   = 0.8062f;
    atms[itmp].sigma     = 1.583f;
    atms[itmp].charge    = -0.8476f;
    atms[itmp+1].id      = (i+1);
    atms[itmp+1].atype   = 1;
    atms[itmp+1].epsilon = 0.f;
    atms[itmp+1].sigma   = 1.f;
    atms[itmp+1].charge  = 0.4238f;
    atms[itmp+2].id      = (i+1);
    atms[itmp+2].atype   = 1;
    atms[itmp+2].epsilon = 0.f;
    atms[itmp+2].sigma   = 1.f;
    atms[itmp+2].charge  = 0.4238f;
  }
  itmp = 0*num_atype+0;
  sigma2   [itmp] = 1.f/((1.583f+1.583f)*(1.583f+1.583f));
  epsilon4 [itmp] = 4.f * (0.8062f*0.8062f);
  itmp = 1*num_atype+0;
  sigma2   [itmp] = 1.f;
  epsilon4 [itmp] = 0.;
  itmp = 1*num_atype+1;
  sigma2   [itmp] = 1.f;
  epsilon4 [itmp] = 0.f;
  for (int i = 0; i < num_atype; i++)
    for (int j = 0; j < num_atype; j++){
      if (i < j){
	sigma2[i*num_atype+j] = sigma2[j*num_atype+i];
	epsilon4[i*num_atype+j] = epsilon4[j*num_atype+i];
      }
      epsilon24[i*num_atype+j] = epsilon4[i*num_atype+j]*6.f*sigma2[i*num_atype+j];
    }
  pos_site[0*3]   = 0.f;
  pos_site[0*3+1] = -0.0646054f;
  pos_site[0*3+2] = 0.f;
  pos_site[1*3]   = 0.816490f;
  pos_site[1*3+1] = 0.512753f;
  pos_site[1*3+2] = 0.f;
  pos_site[2*3]   = -0.816490f;
  pos_site[2*3+1] = 0.512753f;
  pos_site[2*3+2] = 0.f;
  mass_site[0] = 15.9994f;
  mass_site[1] = 1.0079f;
  mass_site[2] = 1.0079f;
  *mass_mole = 0.f;
  for (int i = 0; i < 3; i++)
    *mass_mole += mass_site[i];
  *rcp_mass_mole = 1.f / *mass_mole;
  return;
}

void setParamMethane
(struct atm* atms, int num_mole, float* pos_site, float* mass_site, 
 float* rcp_mass_mole, float* mass_mole, float* inertia_moment)
{
  int itmp;
  for (int i = 0; i < num_mole; i++){
    itmp = i;
    atms[itmp].epsilon = 1.10929f;
    atms[itmp].sigma   = 1.865f;
    atms[itmp].charge  = 0.f;
    atms[itmp].id      = (i+1);
  }
  pos_site[0] = 0.f;
  pos_site[1] = 0.f;
  pos_site[2] = 0.f;
  mass_site[0] = 16.0424f;
  *mass_mole = mass_site[0];
  *rcp_mass_mole = 1.f / *mass_mole;
  return;
}

void getInertiaMoment
(int num_site, float* inertia_moment, float* mass_site, float* pos_site)
{
  for (int i = 0; i < 3; i++)
    inertia_moment[i] = 0.f;
  for (int i = 0; i < num_site; i++){
    inertia_moment[0] += mass_site[i] *
      (pos_site[i*3+1]*pos_site[i*3+1] + pos_site[i*3+2]*pos_site[i*3+2]);
    inertia_moment[1] += mass_site[i] *
      (pos_site[i*3+0]*pos_site[i*3+0] + pos_site[i*3+2]*pos_site[i*3+2]);
    inertia_moment[2] += mass_site[i] *
      (pos_site[i*3+0]*pos_site[i*3+0] + pos_site[i*3+1]*pos_site[i*3+1]);
  }
  for (int i = 0; i < 3; i++)
    if (inertia_moment[i] == 0.f)
      inertia_moment[i] = 1.f;
  return;
}

void setFCCPosition(struct mole* moles, int num_mole, float* dom_width)
{
  int num_lattice = (int)pow(float((num_mole-1)/4),1.f/3.f)+1;
  float lattice_width[3];
  for (int i0 = 0; i0 < 3; i0++)
    lattice_width[i0] = 0.5f * dom_width[i0] / (float)num_lattice;
  int itmp = -1;
  for (int i = 0; i < num_lattice; i++)
    for (int j = 0; j < num_lattice; j++)
      for (int k = 0; k < num_lattice; k++){
	if (itmp*4+3 <= num_mole){
	  itmp = itmp + 1;
	  moles[itmp*4  ].x[0] = ((float)i)*lattice_width[0]*2.f;
	  moles[itmp*4  ].x[1] = ((float)j)*lattice_width[1]*2.f;
	  moles[itmp*4  ].x[2] = ((float)k)*lattice_width[2]*2.f;
	  moles[itmp*4+1].x[0] = ((float)i)*lattice_width[0]*2.f;
	  moles[itmp*4+1].x[1] = ((float)j)*lattice_width[1]*2.f + lattice_width[1];
	  moles[itmp*4+1].x[2] = ((float)k)*lattice_width[2]*2.f + lattice_width[2];
	  moles[itmp*4+2].x[0] = ((float)i)*lattice_width[0]*2.f + lattice_width[0];
	  moles[itmp*4+2].x[1] = ((float)j)*lattice_width[1]*2.f;
	  moles[itmp*4+2].x[2] = ((float)k)*lattice_width[2]*2.f + lattice_width[2];
	  moles[itmp*4+3].x[0] = ((float)i)*lattice_width[0]*2.f + lattice_width[0];
	  moles[itmp*4+3].x[1] = ((float)j)*lattice_width[1]*2.f + lattice_width[1];
	  moles[itmp*4+3].x[2] = ((float)k)*lattice_width[2]*2.f;
	  for (int i1 = 0; i1 < 4; i1++){
	    for (int i0 = 0; i0 < 4; i0++)
	      moles[itmp*4+i1].p[i0] = 0.f;
	    for (int i0 = 0; i0 < 4; i0++)
	      moles[itmp*4+i1].q[i0] = 0.f;
	    moles[itmp*4+i1].q[3] = 1.f;
	  }
	}
      }
  return;
}

void setVelocityRandom(struct mole* moles, int num_mole, float mass_mole, float temp_fix)
{
  float sum[4];
  for (int i = 0; i < 3; i++)
    sum[i] = 0.f;
  for (int i = 0; i < num_mole; i++)
    for (int i0 = 0; i0 < 3; i0++){
      moles[i].v[i0] = (float)rand()/(float)RAND_MAX;
      sum[i0] += moles[i].v[i0];
    }
  sum[3] = 0.f;
  for (int i = 0; i < num_mole; i++){
    for (int i0 = 0; i0 < 3; i0++)
      moles[i].v[i0] -= sum[i0] / (float)num_mole;
    sum[3] += moles[i].v[0]*moles[i].v[0] + 
              moles[i].v[1]*moles[i].v[1] + 
              moles[i].v[2]*moles[i].v[2];
  }
  sum[3] *= mass_mole;
  sum[3] = sqrt(temp_fix * BOLTZMANN * (int)(6*num_mole) / sum[3]);
  for (int i = 0; i < num_mole; i++)
    for (int i0 = 0; i0 < 3; i0++)
      moles[i].v[i0] *= sum[3];
  for (int i = 0; i < num_mole; i++)
    for (int i0 = 0; i0 < 4; i0++)
      moles[i].p[i0] *= 0.f;
  return;
}

void setDomainInfo
(int num_mole, float mass_mole, float* volume, float density,
 float* dom_width, float cutoff_wave, int* num_wave)
{
  int   icut = (int)(cutoff_wave)+1;
  float cutoff2 = cutoff_wave * cutoff_wave;
  *volume = (mass_mole*(float)num_mole)/density;
  for (int i = 0; i < 3; i++)
    dom_width[i] = pow(*volume,1.f/3.f);
  *num_wave = 0;
  for (int i = -icut; i <= icut; i++)
    for (int j = -icut; j <= icut; j++)
      for (int k = -icut; k <= icut; k++)
	if ((float)(i*i+j*j+k*k) < cutoff2)
	  *num_wave = *num_wave + 1;
  *num_wave -= 1;
  return;
}

void updateMomentum(struct mole* moles, int num_mole, float delta_t, float mass_mole){
  float tmp = 0.5f * delta_t / mass_mole;
  for (int i = 0; i < num_mole; i++){
    for (int j = 0; j < 3; j++)
      moles[i].v[j] += tmp * moles[i].f[j];
    moles[i].p[0] += delta_t * 
      (+moles[i].q[0]*moles[i].n[0]-moles[i].q[1]*moles[i].n[1]
       -moles[i].q[2]*moles[i].n[2]-moles[i].q[3]*moles[i].n[3]);
    moles[i].p[1] += delta_t * 
      (+moles[i].q[1]*moles[i].n[0]+moles[i].q[0]*moles[i].n[1]
       -moles[i].q[3]*moles[i].n[2]+moles[i].q[2]*moles[i].n[3]);
    moles[i].p[2] += delta_t * 
      (+moles[i].q[2]*moles[i].n[0]+moles[i].q[3]*moles[i].n[1]
       +moles[i].q[0]*moles[i].n[2]-moles[i].q[1]*moles[i].n[3]);
    moles[i].p[3] += delta_t * 
      (+moles[i].q[3]*moles[i].n[0]-moles[i].q[2]*moles[i].n[1]
       +moles[i].q[1]*moles[i].n[2]+moles[i].q[0]*moles[i].n[3]);
  }
  return;
}

void updatePosition(struct mole* moles, int num_mole, float delta_t, float* dom_width){
  for (int i = 0; i < num_mole; i++)
    for (int j = 0; j < 3; j++){
      moles[i].x[j] += delta_t * moles[i].v[j];
      if (moles[i].x[j] >= dom_width[j])
	moles[i].x[j] -= dom_width[j];
      else if (moles[i].x[j] < 0.f)
	moles[i].x[j] += dom_width[j];
    }
  return;
}

void updateQuaternion1
(struct mole* moles, int num_mole, float delta_t, float* inertia_moment){
  float zeta, coszeta, sinzeta, tmp[4];
  for (int i = 0; i < num_mole; i++){
    tmp[0] = -moles[i].q[3];
    tmp[1] =  moles[i].q[2];
    tmp[2] = -moles[i].q[1];
    tmp[3] =  moles[i].q[0];
    zeta = 0.25f / inertia_moment[2] * delta_t * 0.5f *
      (moles[i].p[0]*tmp[0]+moles[i].p[1]*tmp[1]+
       moles[i].p[2]*tmp[2]+moles[i].p[3]*tmp[3]);
    sinzeta = sin(zeta);
    coszeta = cos(zeta);
    for (int j = 0; j < 4; j++)
      moles[i].q[j] = coszeta*moles[i].q[j] + sinzeta*tmp[j];
    tmp[0] = -moles[i].p[3];
    tmp[1] =  moles[i].p[2];
    tmp[2] = -moles[i].p[1];
    tmp[3] =  moles[i].p[0];
    for (int j = 0; j < 4; j++)
      moles[i].p[j] = coszeta*moles[i].p[j] + sinzeta*tmp[j];
  }
  return;
}

void updateQuaternion2
(struct mole* moles, int num_mole, float delta_t, float* inertia_moment){
  float zeta, coszeta, sinzeta, tmp[4];
  for (int i = 0; i < num_mole; i++){
    tmp[0] = -moles[i].q[2];
    tmp[1] = -moles[i].q[3];
    tmp[2] =  moles[i].q[0];
    tmp[3] =  moles[i].q[1];
    zeta = 0.25f / inertia_moment[1] * delta_t * 0.5f *
      (moles[i].p[0]*tmp[0]+moles[i].p[1]*tmp[1]+
       moles[i].p[2]*tmp[2]+moles[i].p[3]*tmp[3]);
    sinzeta = sin(zeta);
    coszeta = cos(zeta);
    for (int j = 0; j < 4; j++)
      moles[i].q[j] = coszeta*moles[i].q[j] + sinzeta*tmp[j];
    tmp[0] = -moles[i].p[2];
    tmp[1] = -moles[i].p[3];
    tmp[2] =  moles[i].p[0];
    tmp[3] =  moles[i].p[1];
    for (int j = 0; j < 4; j++)
      moles[i].p[j] = coszeta*moles[i].p[j] + sinzeta*tmp[j];
  }
  return;
}

void updateQuaternion3
(struct mole* moles, int num_mole, float delta_t, float* inertia_moment){
  float zeta, coszeta, sinzeta, tmp[4];
  for (int i = 0; i < num_mole; i++){
    tmp[0] = -moles[i].q[1];
    tmp[1] =  moles[i].q[0];
    tmp[2] =  moles[i].q[3];
    tmp[3] = -moles[i].q[2];
    zeta = 0.25f / inertia_moment[0] * delta_t * 1.f *
      (moles[i].p[0]*tmp[0]+moles[i].p[1]*tmp[1]+
       moles[i].p[2]*tmp[2]+moles[i].p[3]*tmp[3]);
    sinzeta = sin(zeta);
    coszeta = cos(zeta);
    for (int j = 0; j < 4; j++)
      moles[i].q[j] = coszeta*moles[i].q[j] + sinzeta*tmp[j];
    tmp[0] = -moles[i].p[1];
    tmp[1] =  moles[i].p[0];
    tmp[2] =  moles[i].p[3];
    tmp[3] = -moles[i].p[2];
    for (int j = 0; j < 4; j++)
      moles[i].p[j] = coszeta*moles[i].p[j] + sinzeta*tmp[j];
  }
  return;
}

void setSitePosition
(struct atm* atms, struct mole* moles, int num_mole, int num_site, float* pos_site)
{
  for (int i = 0; i < num_mole; i++)
    for (int k = 0; k < num_site; k++){
      atms[i*num_site+k].x[0] = moles[i].x[0]+
	(+moles[i].q[0]*moles[i].q[0]+moles[i].q[1]*moles[i].q[1]
	 -moles[i].q[2]*moles[i].q[2]-moles[i].q[3]*moles[i].q[3])*pos_site[k*3]+
	2.f*(moles[i].q[1]*moles[i].q[2]-moles[i].q[0]*moles[i].q[3])*pos_site[k*3+1]+
	2.f*(moles[i].q[1]*moles[i].q[3]+moles[i].q[0]*moles[i].q[2])*pos_site[k*3+2];
      atms[i*num_site+k].x[1] = moles[i].x[1]+
	2.f*(moles[i].q[1]*moles[i].q[2]+moles[i].q[0]*moles[i].q[3])*pos_site[k*3]+
	(+moles[i].q[0]*moles[i].q[0]-moles[i].q[1]*moles[i].q[1]
	 +moles[i].q[2]*moles[i].q[2]-moles[i].q[3]*moles[i].q[3])*pos_site[k*3+1]+
	2.f*(moles[i].q[2]*moles[i].q[3]-moles[i].q[0]*moles[i].q[1])*pos_site[k*3+2];
      atms[i*num_site+k].x[2] = moles[i].x[2]+
	2.f*(moles[i].q[1]*moles[i].q[3]-moles[i].q[0]*moles[i].q[2])*pos_site[k*3]+
	2.f*(moles[i].q[2]*moles[i].q[3]+moles[i].q[0]*moles[i].q[1])*pos_site[k*3+1]+
	(+moles[i].q[0]*moles[i].q[0]-moles[i].q[1]*moles[i].q[1]
	 -moles[i].q[2]*moles[i].q[2]+moles[i].q[3]*moles[i].q[3])*pos_site[k*3+2];
    }
  return;
}

void setMoleTorque
(struct atm* atms, struct mole* moles, int num_mole, int num_site, float* pos_site)
{
  float force[3];
  int itmp;
  for (int i = 0; i < num_mole; i++){
    for (int j = 0; j < 4; j++)
      moles[i].n[j] = 0.f;
    for (int k = 0; k < num_site; k++){
      itmp = i*num_site+k;
      force[0] = 
	(+moles[i].q[0]*moles[i].q[0]+moles[i].q[1]*moles[i].q[1]
	 -moles[i].q[2]*moles[i].q[2]-moles[i].q[3]*moles[i].q[3])    *atms[itmp].f[0]+
	2.f*(+moles[i].q[1]*moles[i].q[2]+moles[i].q[0]*moles[i].q[3])*atms[itmp].f[1]+
	2.f*(+moles[i].q[1]*moles[i].q[3]-moles[i].q[0]*moles[i].q[2])*atms[itmp].f[2];
      force[1] = 
	2.f*(+moles[i].q[1]*moles[i].q[2]-moles[i].q[0]*moles[i].q[3])*atms[itmp].f[0]+
	(+moles[i].q[0]*moles[i].q[0]-moles[i].q[1]*moles[i].q[1]
	 +moles[i].q[2]*moles[i].q[2]-moles[i].q[3]*moles[i].q[3])    *atms[itmp].f[1]+
	2.f*(+moles[i].q[2]*moles[i].q[3]+moles[i].q[0]*moles[i].q[1])*atms[itmp].f[2];
      force[2] = 
	2.f*(+moles[i].q[1]*moles[i].q[3]+moles[i].q[0]*moles[i].q[2])*atms[itmp].f[0]+
	2.f*(+moles[i].q[2]*moles[i].q[3]-moles[i].q[0]*moles[i].q[1])*atms[itmp].f[1]+
	(+moles[i].q[0]*moles[i].q[0]-moles[i].q[1]*moles[i].q[1]
	 -moles[i].q[2]*moles[i].q[2]+moles[i].q[3]*moles[i].q[3])    *atms[itmp].f[2];
      moles[i].n[0] += 
	pos_site[k*3]*force[0] + pos_site[k*3+1]*force[1] + pos_site[k*3+2]*force[2];
      moles[i].n[1] += pos_site[k*3+1]*force[2] - pos_site[k*3+2]*force[1];
      moles[i].n[2] += pos_site[k*3+2]*force[0] - pos_site[k*3  ]*force[2];
      moles[i].n[3] += pos_site[k*3  ]*force[1] - pos_site[k*3+1]*force[0];
    }
  }
  return;
}

void setMoleForce(struct atm* atms, struct mole* moles, int num_mole, int num_site){
  for (int i = 0; i < num_mole; i++){
    for (int j = 0; j < 3; j++)
      moles[i].f[j] = 0.f;
    for (int k = 0; k < num_site; k++){
      for (int j = 0; j < 3; j++)
	moles[i].f[j] += atms[i*num_site+k].f[j];
      for (int j = 0; j < 3; j++)
	atms[i*num_site+k].pressure -= 
	  (atms[i*num_site+k].x[j]-moles[i].x[j]) * atms[i*num_site+k].f[j];
    }
  }
  return;
}

void setVelocityScaling
(struct mole* moles, int num_mole, float mass_mole, float temp_fix, float* inertia_moment)
{
  float sum, tmp3[3], rcp_inertia[3];
  sum = 0.f;
  for (int i = 0; i < num_mole; i++)
    sum += 0.5f * mass_mole * (moles[i].v[0]*moles[i].v[0] + 
			       moles[i].v[1]*moles[i].v[1] + 
			       moles[i].v[2]*moles[i].v[2]);
  for (int i = 0; i < 3; i++)
    rcp_inertia[i] = 0.5f / inertia_moment[i];
  for (int i = 0; i < num_mole; i++){
    tmp3[0] = rcp_inertia[0] * (-moles[i].q[1] * moles[i].p[0]
				+moles[i].q[0] * moles[i].p[1] 
				+moles[i].q[3] * moles[i].p[2] 
				-moles[i].q[2] * moles[i].p[3]);
    tmp3[1] = rcp_inertia[1] * (-moles[i].q[2] * moles[i].p[0]
				-moles[i].q[3] * moles[i].p[1] 
				+moles[i].q[0] * moles[i].p[2] 
				+moles[i].q[1] * moles[i].p[3]);
    tmp3[2] = rcp_inertia[2] * (-moles[i].q[3] * moles[i].p[0]
				+moles[i].q[2] * moles[i].p[1] 
				-moles[i].q[1] * moles[i].p[2] 
				+moles[i].q[0] * moles[i].p[3]);
    sum += 0.5f * (inertia_moment[0]*tmp3[0]*tmp3[0]+
		   inertia_moment[1]*tmp3[1]*tmp3[1]+
		   inertia_moment[2]*tmp3[2]*tmp3[2]);
  }
  sum = sqrt(temp_fix / (sum * 2.f / BOLTZMANN / (float)(6*num_mole)));
  for (int i = 0; i < num_mole; i++)
    for (int i0 = 0; i0 < 3; i0++)
      moles[i].v[i0] *= sum;
  for (int i = 0; i < num_mole; i++)
    for (int i0 = 0; i0 < 4; i0++)
      moles[i].p[i0] *= sum;
  return;
}

extern "C"
//__constant__ float pos_site_const[NUM_SITE*3];
__global__ void updateCoordinateAndMomentum_gpukernel
(struct mole* moles_dev, struct atm* atms_dev, float delta_t, float rcp_mass_mole,
 float l0, float l1, float l2)
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int mole_id = NUM_N1_BLOCK*bx+tx;
  struct mole mole_reg;
  float dr[3];
  float tmp[4];

  mole_reg.x[0] = moles_dev[mole_id].x[0];
  mole_reg.x[1] = moles_dev[mole_id].x[1];
  mole_reg.x[2] = moles_dev[mole_id].x[2];
  mole_reg.v[0] = moles_dev[mole_id].v[0];
  mole_reg.v[1] = moles_dev[mole_id].v[1];
  mole_reg.v[2] = moles_dev[mole_id].v[2];
  mole_reg.f[0] = 0.f;
  mole_reg.f[1] = 0.f;
  mole_reg.f[2] = 0.f;
  mole_reg.q[0] = moles_dev[mole_id].q[0];
  mole_reg.q[1] = moles_dev[mole_id].q[1];
  mole_reg.q[2] = moles_dev[mole_id].q[2];
  mole_reg.q[3] = moles_dev[mole_id].q[3];
  mole_reg.p[0] = moles_dev[mole_id].p[0];
  mole_reg.p[1] = moles_dev[mole_id].p[1];
  mole_reg.p[2] = moles_dev[mole_id].p[2];
  mole_reg.p[3] = moles_dev[mole_id].p[3];
  mole_reg.n[0] = 0.f;
  mole_reg.n[1] = 0.f;
  mole_reg.n[2] = 0.f;
  mole_reg.n[3] = 0.f;
  for (int i = 0; i < NUM_SITE; i++){
    tmp[0] = atms_dev[mole_id*NUM_SITE+i].f[0];
    tmp[1] = atms_dev[mole_id*NUM_SITE+i].f[1];
    tmp[2] = atms_dev[mole_id*NUM_SITE+i].f[2];
    mole_reg.f[0] += tmp[0];
    mole_reg.f[1] += tmp[1];
    mole_reg.f[2] += tmp[2];
  }
  mole_reg.v[0] += delta_t * rcp_mass_mole * mole_reg.f[0];
  mole_reg.v[1] += delta_t * rcp_mass_mole * mole_reg.f[1];
  mole_reg.v[2] += delta_t * rcp_mass_mole * mole_reg.f[2];
  mole_reg.x[0] += delta_t * mole_reg.v[0];
  mole_reg.x[1] += delta_t * mole_reg.v[1];
  mole_reg.x[2] += delta_t * mole_reg.v[2];
  if (mole_reg.x[0] >= l0)
    mole_reg.x[0] -= l0;
  else if (mole_reg.x[0] < 0.f)
    mole_reg.x[0] += l0;
  if (mole_reg.x[1] >= l1)
    mole_reg.x[1] -= l1;
  else if (mole_reg.x[1] < 0.f)
    mole_reg.x[1] += l1;
  if (mole_reg.x[2] >= l2)
    mole_reg.x[2] -= l2;
  else if (mole_reg.x[2] < 0.f)
    mole_reg.x[2] += l2;
  for (int i = 0; i < NUM_SITE; i++){
    dr[0] = 0.f;
    dr[1] = 0.f;
    dr[2] = 0.f;
    //dr[0] = pos_site_const[i*3];
    //dr[1] = pos_site_const[i*3+1];
    //dr[2] = pos_site_const[i*3+2];
    tmp[0] = mole_reg.x[0] + dr[0];
    tmp[1] = mole_reg.x[1] + dr[1];
    tmp[2] = mole_reg.x[2] + dr[2];
    atms_dev[mole_id*NUM_SITE+i].x[0] = tmp[0];
    atms_dev[mole_id*NUM_SITE+i].x[1] = tmp[1];
    atms_dev[mole_id*NUM_SITE+i].x[2] = tmp[2];
  }
  atms_dev[mole_id].x[0] = mole_reg.x[0];
  atms_dev[mole_id].x[1] = mole_reg.x[1];
  atms_dev[mole_id].x[2] = mole_reg.x[2];
  moles_dev[mole_id].x[0] = mole_reg.x[0];
  moles_dev[mole_id].x[1] = mole_reg.x[1];
  moles_dev[mole_id].x[2] = mole_reg.x[2];
  moles_dev[mole_id].v[0] = mole_reg.v[0];
  moles_dev[mole_id].v[1] = mole_reg.v[1];
  moles_dev[mole_id].v[2] = mole_reg.v[2];
  //moles_dev[mole_id].f[0] = mole_reg.f[0];
  //moles_dev[mole_id].f[1] = mole_reg.f[1];
  //moles_dev[mole_id].f[2] = mole_reg.f[2];
  moles_dev[mole_id].q[0] = mole_reg.q[0];
  moles_dev[mole_id].q[1] = mole_reg.q[1];
  moles_dev[mole_id].q[2] = mole_reg.q[2];
  moles_dev[mole_id].q[3] = mole_reg.q[3];
  moles_dev[mole_id].p[0] = mole_reg.p[0];
  moles_dev[mole_id].p[1] = mole_reg.p[1];
  moles_dev[mole_id].p[2] = mole_reg.p[2];
  moles_dev[mole_id].p[3] = mole_reg.p[3];
  //moles_dev[mole_id].n[0] = mole_reg.n[0];
  //moles_dev[mole_id].n[1] = mole_reg.n[1];
  //moles_dev[mole_id].n[2] = mole_reg.n[2];
  //moles_dev[mole_id].n[3] = mole_reg.n[3];
  return;
}

void getEnergy
(struct atm* atms, struct mole* moles, int num_mole,  int num_atm,
 float* energy, float mass_mole, float alpha_ewald, int num_wave,
 struct recip* recips, float* inertia_moment, float volume, float* pressure)
{
  float tmp3[3], rcp_inertia[3], tmp, p;
  energy[1] = 0.f;
  p = 0.f;
  tmp = 0.f;
  for (int i = 0; i < num_atm; i++){
    energy[1] += atms[i].pot;
    p += atms[i].pressure;
  }
  energy[1] *= 0.5f;
  for (int i = 0; i < num_wave; i++){
    energy[1] += recips[i].pot;
    tmp += recips[i].pressure;
  }
  printf("recip pressure %f \n",tmp/(3.f*volume)*1660.538f);
  p += tmp;
  energy[2] = 0.f;
  for (int i = 0; i < num_mole; i++)
    for (int j = 0; j < 3; j++)
      energy[2] += moles[i].v[j]*moles[i].v[j];
  energy[2] *= 0.5f * mass_mole;
  p += energy[2] * 2.f;
  energy[3] = 0.f;
  for (int i = 0; i < 3; i++)
    rcp_inertia[i] = 0.5f / inertia_moment[i];
  for (int i = 0; i < num_mole; i++){
    tmp3[0] = rcp_inertia[0] * (-moles[i].q[1] * moles[i].p[0]
				+moles[i].q[0] * moles[i].p[1] 
				+moles[i].q[3] * moles[i].p[2] 
				-moles[i].q[2] * moles[i].p[3]);
    tmp3[1] = rcp_inertia[1] * (-moles[i].q[2] * moles[i].p[0]
				-moles[i].q[3] * moles[i].p[1] 
				+moles[i].q[0] * moles[i].p[2] 
				+moles[i].q[1] * moles[i].p[3]);
    tmp3[2] = rcp_inertia[2] * (-moles[i].q[3] * moles[i].p[0]
				+moles[i].q[2] * moles[i].p[1] 
				-moles[i].q[1] * moles[i].p[2] 
				+moles[i].q[0] * moles[i].p[3]);
    for (int j = 0; j < 3; j++)
      energy[3] += inertia_moment[j]*tmp3[j]*tmp3[j];
  }
  energy[3] *= 0.5f;
  for (int i = 0; i < 4; i++)
    energy[i] /= (float)num_mole;
  energy[0] = energy[1] + energy[2] + energy[3] + energy[4];
  p /= 3.f * volume;
  p *= 1660.538f;
  printf("total %f pot %f trs %f rot %f pressure %f\n"
	 ,energy[0],energy[1],energy[2],energy[3],p);
  *pressure = p;
  return;
}

void getForce
(struct atm* atms, int num_atm, float cutoff, float* dom_width, float alpha_ewald,
 float* sigma2, float* epsilon4, float* epsilon24, int num_atype)
{
  int itmp;
  float dr[9], rcp_width[3], sig, eps;
  float alpha1 = DIELECTRIC * alpha_ewald;
  float alpha2 = alpha_ewald * alpha_ewald;
  float alpha3 = DIELECTRIC * alpha_ewald * alpha_ewald * alpha_ewald;
  float cut2   = cutoff * cutoff;
  for (int i0 = 0; i0 < 3; i0++)
    rcp_width[i0] = 1.f / dom_width[i0];
  for (int i = 0; i < num_atm; i++){
    for (int i0 = 0; i0 < 3; i0++)
      atms[i].f[i0] = 0.f;
    atms[i].pot = 0.f;
    atms[i].pressure = 0.f;
  }
  for (int i = 0; i < num_atm-1; i++){
    for (int j = i+1; j < num_atm; j++)
      if (atms[i].id != atms[j].id){
	for (int i0 = 0; i0 < 3; i0++){
	  dr[i0] = atms[i].x[i0] - atms[j].x[i0];
	  dr[i0] -= rint(dr[i0]*rcp_width[i0])*dom_width[i0];
	}
	dr[3] = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
	if (dr[3] < cut2){
	  itmp = num_atype*atms[i].atype + atms[j].atype;
	  dr[4] = 1.f / (dr[3]*sigma2[itmp]);
	  dr[5] = dr[4] * dr[4] * dr[4];
	  dr[4] *= epsilon24[itmp] * dr[5] * (2.f * dr[5] - 1.f);
	  dr[6] = dr[3] * alpha2; // (alpha*r)^2
	  dr[7] = sqrt(dr[6]); // (alpha*r)
	  dr[8] = 1.f / dr[7]; //  1 / (alpha*r)
	  dr[7] = erfc(dr[7]) * dr[8]; // erfc(alpha*r) / (alpha*r)
	  dr[4] += alpha3 * atms[i].charge * atms[j].charge *
	    (RSQRTPI2 * exp(-dr[6]) + dr[7]) * dr[8] * dr[8];
	  for (int i0 = 0; i0 < 3; i0++){
	    atms[i].f[i0] += dr[4] * dr[i0];
	    atms[j].f[i0] -= dr[4] * dr[i0];
	  }
	  atms[i].pressure += dr[4] * dr[3];
	  atms[i].pot += epsilon4[itmp] * dr[5] * (dr[5] - 1.f); 
	  atms[i].pot += alpha1 * atms[i].charge * atms[j].charge * dr[7];
	}
      }
  }
  return;
}

void getPotentialConst
(struct atm* atms, int num_site, float alpha_ewald, float* energy, float* pos_site)
{
  float pot_self, pot_intra;
  float dr[4];

  pot_self = 0.f;
  pot_intra = 0.f;
  for (int i = 0; i < num_site; i++)
    pot_self -= DIELECTRIC * alpha_ewald * RSQRTPI * atms[i].charge * atms[i].charge;
  for (int i = 0; i < num_site-1; i++)
    for (int j = i+1; j < num_site; j++){
      for (int i0 = 0; i0 < 3; i0++)
	dr[i0] = pos_site[i*3+i0] - pos_site[j*3+i0];
      dr[3] = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
      dr[3] = sqrt(dr[3]);
      pot_intra -= DIELECTRIC * atms[i].charge * atms[j].charge *
	erf(alpha_ewald * dr[3]) / dr[3];
    }
  energy[4] = pot_self + pot_intra;
  printf("self %f intra %f \n",pot_self,pot_intra);
  return;
}

extern "C"
__global__ void getForce_gpukernel
(float l0, float l1, float l2, int num_atm256, struct atm* atms_dev)
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int atm_id = NUM_THREAD*bx+tx;
  float dr[4], dn6, coef[2], rcp_l[3];
  rcp_l[0] = 1.f / l0;
  rcp_l[1] = 1.f / l1;
  rcp_l[2] = 1.f / l2;
  
  struct atm atm_reg;
  atm_reg.x[0]     = atms_dev[atm_id].x[0];
  atm_reg.x[1]     = atms_dev[atm_id].x[1];
  atm_reg.x[2]     = atms_dev[atm_id].x[2];
  atm_reg.id       = atms_dev[atm_id].id;
  atm_reg.epsilon  = atms_dev[atm_id].epsilon;
  atm_reg.sigma    = atms_dev[atm_id].sigma;
  atm_reg.charge   = atms_dev[atm_id].charge;
  atm_reg.pot      = 0.f;
  atm_reg.f[0]     = 0.f;
  atm_reg.f[1]     = 0.f;
  atm_reg.f[2]     = 0.f;
  atm_reg.pressure = 0.f;
  for (int j = 0; j < num_atm256/NUM_THREAD; j++){
    __shared__ struct atm atm_shd[NUM_THREAD];
    atm_shd[tx].x[0]    = atms_dev[j*NUM_THREAD+tx].x[0];
    atm_shd[tx].x[1]    = atms_dev[j*NUM_THREAD+tx].x[1];
    atm_shd[tx].x[2]    = atms_dev[j*NUM_THREAD+tx].x[2];
    atm_shd[tx].id      = atms_dev[j*NUM_THREAD+tx].id;
    atm_shd[tx].epsilon = atms_dev[j*NUM_THREAD+tx].epsilon;
    atm_shd[tx].sigma   = atms_dev[j*NUM_THREAD+tx].sigma;
    atm_shd[tx].charge  = atms_dev[j*NUM_THREAD+tx].charge;
    __syncthreads();
    for (int i = 0; i < NUM_THREAD; i++){
      dr[0] = atm_reg.x[0] - atm_shd[i].x[0];
      dr[1] = atm_reg.x[1] - atm_shd[i].x[1];
      dr[2] = atm_reg.x[2] - atm_shd[i].x[2];
      dr[0] -= rintf(dr[0]*rcp_l[0])*l0;
      dr[1] -= rintf(dr[1]*rcp_l[1])*l1;
      dr[2] -= rintf(dr[2]*rcp_l[2])*l2;
      dr[3] = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
      if (dr[3] < 81.f && dr[3] > 0.0001f){
	coef[0] = atm_reg.sigma + atm_shd[i].sigma;
	coef[0] *= coef[0];
	dr[3] = coef[0] / dr[3];
	dn6 = dr[3] * dr[3] * dr[3];
	coef[1] = atm_reg.epsilon * atm_shd[i].epsilon;
	coef[1] *= dn6;
	dr[3] *= 24.f * coef[1] / coef[0] * (2.f * dn6 - 1.f);
	atm_reg.pot  += 4.f * coef[1] * (dn6 - 1.f);
	atm_reg.f[0] += dr[3] * dr[0];
	atm_reg.f[1] += dr[3] * dr[1];
	atm_reg.f[2] += dr[3] * dr[2];
	atm_reg.pressure += dr[3] * (dr[2]*dr[2]+dr[0]*dr[0]+dr[1]*dr[1]);
	//atm_reg.pressure += dr[3] * (dr[2]*dr[2]-0.5f*(dr[0]*dr[0]+dr[1]*dr[1]));
      }
    }
    __syncthreads();
  }
  atms_dev[atm_id].pot  = atm_reg.pot;
  atms_dev[atm_id].f[0] = atm_reg.f[0];
  atms_dev[atm_id].f[1] = atm_reg.f[1];
  atms_dev[atm_id].f[2] = atm_reg.f[2];
  atms_dev[atm_id].pressure = atm_reg.pressure;
  return;
}

extern "C"
void getForce_gpumother(struct atm* atms, int num_atm, float cutoff, float* dom_width)
{
  //CUT_CHECK_DEVICE();
  int num_atm256 = num_atm;
  float l0 = dom_width[0];
  float l1 = dom_width[1];
  float l2 = dom_width[2];
  struct atm* atms_dev;
  int memsize_atms_dev = num_atm256*sizeof(struct atm);
  CUDA_SAFE_CALL(cudaMalloc((void**) &atms_dev, memsize_atms_dev));
  CUDA_SAFE_CALL(cudaMemcpy(atms_dev, atms, memsize_atms_dev, cudaMemcpyHostToDevice));
  dim3 threads(NUM_THREAD);
  dim3 grid(num_atm256 / NUM_THREAD);
  getForce_gpukernel<<< grid , threads >>>(l0,l1,l2,num_atm256,atms_dev);
  CUT_CHECK_ERROR("Kernel execution failed");
  CUDA_SAFE_CALL(cudaMemcpy(atms, atms_dev, memsize_atms_dev, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(atms_dev));
  return;
}

void setReciprocalVectors
(float cutoff_wave, struct recip* recips, float* dom_width, float volume, float alpha_ewald)
{
  float tmp;
  int   icut    = (int)(cutoff_wave)+1;
  float cutoff2 = cutoff_wave * cutoff_wave;
  int   iwave   = 0;
  float rcp_alpha2 = (PI*PI) / (alpha_ewald*alpha_ewald);
  float tmp_wave = DIELECTRIC * 2.f / volume;
  for (int i = -icut; i <= icut; i++)
    for (int j = -icut; j <= icut; j++)
      for (int k = -icut; k <= icut; k++)
	if ((float)(i*i+j*j+k*k) < cutoff2 && !(i == 0 && j == 0 && k == 0)){
	  recips[iwave].vec[0] = ((float)i) / dom_width[0];
	  recips[iwave].vec[1] = ((float)j) / dom_width[1];
	  recips[iwave].vec[2] = ((float)k) / dom_width[2];
	  tmp = recips[iwave].vec[0]*recips[iwave].vec[0]
	    +recips[iwave].vec[1]*recips[iwave].vec[1]
	    +recips[iwave].vec[2]*recips[iwave].vec[2];
	  recips[iwave].coef = tmp_wave * exp(-rcp_alpha2*tmp) / tmp;
	  iwave += 1;
	}
  return;
}

void getForceWave
(int num_atm, int num_wave, float alpha_ewald, float volume, 
 struct atm* atms, struct recip* recips)
{
  float tmp;
  float rcp_alpha2 = (PI*PI) / (alpha_ewald*alpha_ewald);
  for (int i = 0; i < num_wave; i++){
    recips[i].sumcos = 0.f;
    recips[i].sumsin = 0.f;
    for (int j = 0; j < num_atm; j ++){
      tmp = PI2 * (recips[i].vec[0]*atms[j].x[0]+
		   recips[i].vec[1]*atms[j].x[1]+
		   recips[i].vec[2]*atms[j].x[2]);
      recips[i].sumcos += atms[j].charge * cos(tmp);
      recips[i].sumsin += atms[j].charge * sin(tmp);
    }
    tmp = recips[i].vec[0]*recips[i].vec[0]
      +recips[i].vec[1]*recips[i].vec[1]
      +recips[i].vec[2]*recips[i].vec[2];
    recips[i].pot  = 0.5f * RCP_PI2 * recips[i].coef * 
      (recips[i].sumcos * recips[i].sumcos + recips[i].sumsin * recips[i].sumsin);
    recips[i].pressure = recips[i].pot * (3.f-2.f*(1.f+tmp*rcp_alpha2));
  }
  for (int j = 0; j < num_atm; j ++){
    for (int i = 0; i < num_wave; i++){
      tmp = PI2 * (recips[i].vec[0]*atms[j].x[0]+
		   recips[i].vec[1]*atms[j].x[1]+
		   recips[i].vec[2]*atms[j].x[2]);
      tmp = atms[j].charge * recips[i].coef * 
	(sin(tmp) * recips[i].sumcos - cos(tmp) * recips[i].sumsin);
      for (int k = 0; k < 3; k++)
	atms[j].f[k] += tmp * recips[i].vec[k];
    }
  }
  return;
}

void saveCvfile
(struct atm* atms,float* dom_width, int num_atm, int num_mole, int num_site)
{
  FILE *fp_cv;
  fp_cv = fopen("000000.d","w");
  fprintf(fp_cv,"#box_sx=0 box_sy=0 box_sz=0 box_ex=%d box_ey=%d box_ez=%d bond_file=bond.d r1=0.1\n"
	  ,(int)dom_width[0],(int)dom_width[1],(int)dom_width[2]);
  for (int i = 0; i < num_atm; i++)
    fprintf(fp_cv,"%d 1 %f %f %f\n",i+1,atms[i].x[0],atms[i].x[1],atms[i].x[2]);
  fclose(fp_cv);
  FILE *fp_bond;
  fp_bond = fopen("bond.d","w");
  for (int i = 0; i < num_mole; i++)
    for (int j = 0; j < num_site-1; j++)
      for (int k = j+1; k < num_site; k++)
	fprintf(fp_bond,"%d %d\n",num_site*i+j+1,num_site*i+k+1);
  fclose(fp_bond);
  return;
}

void gpumd_()
{
  int num_mole      = 256;
  int max_step      = 10000;
  int interval_save = 10;
  int num_site      = NUM_SITE;
  int num_atm       = num_mole * num_site;
  int num_atype     = 2;
  float density     = 1.f * 0.6022142f; // g/cm^3
  float temp_fix    = 273.f;
  float alpha_ewald = 0.35f;
  int num_wave;
  double stime,ltime;
  float volume;
  float pressure;
  float mass_mole;
  float rcp_mass_mole;
  float delta_t = 0.04f;
  float cutoff = 9.f;
  float cutoff_wave = 6.f;
  float dom_width[3];
  float energy[5];
  float inertia_moment[3];

  unsigned int memsize_atms = num_atm * sizeof(struct atm);
  struct atm* atms      = (struct atm*) malloc(memsize_atms);
  unsigned int memsize_moles = num_mole * sizeof(struct mole);
  struct mole* moles    = (struct mole*) malloc(memsize_moles);
  unsigned int memsize_pos_site = num_site * 3 * sizeof(float);
  float* pos_site       = (float*) malloc(memsize_pos_site);
  unsigned int memsize_mass_site = num_site * sizeof(float);
  float* mass_site      = (float*) malloc(memsize_mass_site);
  unsigned int memsize_atypes = num_atype * num_atype * sizeof(float);
  float* sigma2    = (float*) malloc(memsize_atypes);
  float* epsilon4  = (float*) malloc(memsize_atypes);
  float* epsilon24 = (float*) malloc(memsize_atypes);
  FILE *fp;
  fp = fopen("data.d","w");

  setParamWater
    //setParamMethane
    (atms,num_mole,pos_site,mass_site,&rcp_mass_mole,&mass_mole,inertia_moment,
     num_atype,sigma2,epsilon4,epsilon24);
  getInertiaMoment(num_site,inertia_moment,mass_site,pos_site);
  printf("inr1 %f 2 %f 3 %f\n",inertia_moment[0],inertia_moment[1],inertia_moment[2]);
  setDomainInfo(num_mole,mass_mole,&volume,density,dom_width,cutoff_wave,&num_wave);
  unsigned int memsize_recips = num_wave * sizeof(struct recip);
  struct recip* recips = (struct recip*) malloc(memsize_recips);
  printf("dom1 %f 2 %f 3 %f\n",dom_width[0],dom_width[1],dom_width[2]);
  printf("number(atm): %d      number(wave): %d\n",num_atm,num_wave);
  setReciprocalVectors(cutoff_wave,recips,dom_width,volume,alpha_ewald);
  setFCCPosition(moles,num_mole,dom_width);
  setVelocityRandom(moles,num_mole,mass_mole,temp_fix);
  setSitePosition(atms,moles,num_mole,num_site,pos_site);
  saveCvfile(atms,dom_width,num_atm,num_mole,num_site);
  getPotentialConst(atms,num_site,alpha_ewald,energy,pos_site);
  getEnergy(atms,moles,num_mole,num_atm,energy,mass_mole,alpha_ewald,num_wave,
	    recips,inertia_moment,volume,&pressure);
#ifndef GPU
  getForce(atms,num_atm,cutoff,dom_width,alpha_ewald,sigma2,epsilon4,epsilon24,num_atype);
  getForceWave(num_atm,num_wave,alpha_ewald,volume,atms,recips);
  setMoleForce(atms,moles,num_mole,num_site);
  setMoleTorque(atms,moles,num_mole,num_site,pos_site);
  getEnergy(atms,moles,num_mole,num_atm,energy,mass_mole,alpha_ewald,num_wave,
	    recips,inertia_moment,volume,&pressure);
  for (int step = 0; step < max_step; step++){
    //get_cputime(&ltime,&stime);
    updateMomentum(moles,num_mole,delta_t,mass_mole);
    updateQuaternion1(moles,num_mole,delta_t,inertia_moment);
    updateQuaternion2(moles,num_mole,delta_t,inertia_moment);
    updateQuaternion3(moles,num_mole,delta_t,inertia_moment);
    updatePosition(moles,num_mole,delta_t,dom_width);
    updateQuaternion2(moles,num_mole,delta_t,inertia_moment);
    updateQuaternion1(moles,num_mole,delta_t,inertia_moment);
    setSitePosition(atms,moles,num_mole,num_site,pos_site);
    //get_cputime(&ltime,&stime);
    //printf("Processing time 1 : %10.3f (sec)\n", stime);
    getForce(atms,num_atm,cutoff,dom_width,alpha_ewald,sigma2,epsilon4,epsilon24,num_atype);
    getForceWave(num_atm,num_wave,alpha_ewald,volume,atms,recips);
    //getForce_gpumother(atms,num_atm,cutoff,dom_width);
    //get_cputime(&ltime,&stime);
    //printf("Processing time 2 : %10.3f (sec)\n", stime);
    setMoleForce(atms,moles,num_mole,num_site);
    setMoleTorque(atms,moles,num_mole,num_site,pos_site);
    updateMomentum(moles,num_mole,delta_t,mass_mole);
    //get_cputime(&ltime,&stime);
    //printf("Processing time 3 : %10.3f (sec)\n", stime);
    if (step % interval_save == 0){
      setVelocityScaling(moles,num_mole,mass_mole,temp_fix,inertia_moment);
      getEnergy(atms,moles,num_mole,num_atm,energy,mass_mole,alpha_ewald,
		num_wave,recips,inertia_moment,volume,&pressure);
      fprintf(fp,"%f %f %f %f %f %f\n",
	      energy[0],energy[1]+energy[4],energy[2]+energy[3],energy[2],energy[3],pressure);
      fflush(fp);
      saveCvfile(atms,dom_width,num_atm,num_mole,num_site);
    }
  }
#else
  int num_atm256 = num_atm;
  int num_mole256 = num_mole;
  float l0 = dom_width[0];
  float l1 = dom_width[1];
  float l2 = dom_width[2];
  struct atm*  atms_dev;
  struct mole* moles_dev;
  int memsize_atms_dev  = num_atm256  * sizeof(struct atm);
  int memsize_moles_dev = num_mole256 * sizeof(struct mole);
  CUDA_SAFE_CALL(cudaMalloc((void**)&atms_dev , memsize_atms_dev));
  CUDA_SAFE_CALL(cudaMalloc((void**)&moles_dev, memsize_moles_dev));
  CUDA_SAFE_CALL(cudaMemcpy(atms_dev, atms, memsize_atms_dev, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(moles_dev,moles,memsize_moles_dev,cudaMemcpyHostToDevice));
  //CUDA_SAFE_CALL(cudaMemcpyToSymbol(pos_site_const , pos_site , NUM_SITE*3) );
  dim3 threads_N1(NUM_N1_THREAD);
  dim3 grid_N1(num_mole256 / NUM_N1_THREAD);
  dim3 threads_N2(NUM_THREAD);
  dim3 grid_N2(num_atm256 / NUM_THREAD);
  getForce_gpukernel<<< grid_N2 , threads_N2 >>>(l0,l1,l2,num_atm256,atms_dev);
  for (int step = 0; step < max_step; step++){
    get_cputime(&ltime,&stime);
    updateCoordinateAndMomentum_gpukernel<<< grid_N1 , threads_N1 >>>
      (moles_dev,atms_dev,delta_t,rcp_mass_mole,l0,l1,l2);
    get_cputime(&ltime,&stime);
    printf("Processing time 1 : %10.3f (sec)\n", stime);
    getForce_gpukernel<<< grid_N2 , threads_N2 >>>(l0,l1,l2,num_atm256,atms_dev);
    get_cputime(&ltime,&stime);
    printf("Processing time 2 : %10.3f (sec)\n", stime);
    //if (step % interval_save == 0){
    //  CUDA_SAFE_CALL(cudaMemcpy(atms, atms_dev, memsize_atms_dev, cudaMemcpyDeviceToHost));
    //  CUDA_SAFE_CALL(cudaMemcpy(moles,moles_dev,memsize_moles_dev,cudaMemcpyDeviceToHost));
    //  getEnergy(atms,moles,num_mole,num_atm,energy,mass_mole,alpha_ewald,num_wave,recips,inertia_moment,volume,&pressure);
    //  fprintf(fp,"%f %f %f\n",energy[0],energy[1],energy[2]);
    //}
  }  
  CUDA_SAFE_CALL(cudaFree(atms_dev));
  CUDA_SAFE_CALL(cudaFree(moles_dev));
#endif
  fclose(fp);
  free(atms);
  free(moles);
  free(pos_site);
  free(mass_site);
  free(recips);
  free(sigma2);
  free(epsilon4);
  free(epsilon24);
}
