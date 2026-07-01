/*

Packing routines to replace mpi_pack, which is really slow

*/

#include <stdio.h>
#include <string.h>

//################################################################################
// pack_int

extern "C" void pack_int_double(int *in, int *inlen, double *out, int *outpos) {

  char *p = (char *)out;

  memcpy((void *)&p[*outpos], (void *)in, (*inlen)*sizeof(int));

  *outpos += (*inlen)*sizeof(int);

  return;
}

extern "C" void pack_int_byte(int *in, int *inlen, char *out, int *outpos) {

  char *p = (char *)out;

  memcpy((void *)&p[*outpos], (void *)in, (*inlen)*sizeof(int));

  *outpos += (*inlen)*sizeof(int);

  return;
}

//################################################################################
// pack_double

extern "C" void pack_double_double(double *in, int *inlen, double *out, int *outpos) {

  char *p = (char *)out;

  memcpy((void *)&p[*outpos], (void *)in, (*inlen)*sizeof(double));

  *outpos += (*inlen)*sizeof(double);

  return;
}

extern "C" void pack_double_byte(double *in, int *inlen, char *out, int *outpos) {

  char *p = (char *)out;

  memcpy((void *)&p[*outpos], (void *)in, (*inlen)*sizeof(double));

  *outpos += (*inlen)*sizeof(double);

  return;
}

//################################################################################
// pack_double_zeros

extern "C" void pack_double_zeros(int *inlen, double *out, int *outpos) {

  char *p = (char *)out;

  memset((void *)&p[*outpos], 0, (*inlen)*sizeof(double));

  *outpos += (*inlen)*sizeof(double);

  return;
}

//################################################################################
// unpack_int

extern "C" void unpack_int_double(double *in, int *inpos, int *out, int *outlen) {

  char *p = (char *)in;

  memcpy((void *)out , (void *)&p[*inpos], (*outlen)*sizeof(int));

  *inpos += (*outlen)*sizeof(int);

  return;

}

extern "C" void unpack_int_byte(char *in, int *inpos, int *out, int *outlen) {

  char *p = (char *)in;

  memcpy((void *)out , (void *)&p[*inpos], (*outlen)*sizeof(int));

  *inpos += (*outlen)*sizeof(int);

  return;

}

//################################################################################
// unpack_double

extern "C" void unpack_double_double(double *in, int *inpos, double *out, int *outlen) {

  char *p = (char *)in;

  memcpy((void *)out , (void *)&p[*inpos], (*outlen)*sizeof(double));

  *inpos += (*outlen)*sizeof(double);

  return;

}

extern "C" void unpack_double_byte(char *in, int *inpos, double *out, int *outlen) {

  char *p = (char *)in;

  memcpy((void *)out , (void *)&p[*inpos], (*outlen)*sizeof(double));

  *inpos += (*outlen)*sizeof(double);

  return;

}

//################################################################################
// pack_xyz

extern "C" void pack_xyz_double(int *ngroupl, int *groupl, int *group, double *buffer, int *k,
				double *x, double *y, double *z) {

  int k_loc = *k;
  int ngroupl_loc = *ngroupl;
  char *tmp = (char *)buffer;
  double *p = (double *)&tmp[k_loc];

  int ig;
  for (ig=0;ig < ngroupl_loc;ig++) {
    int i = groupl[ig];
    int groupi = group[i-1];
    int is = groupi & 0xfffffff;
    int iq = is + ((groupi >> 28) & 7);
    int ii;
    for (ii=is;ii <= iq;ii++) {
      p[0] = x[ii-1];
      p[1] = y[ii-1];
      p[2] = z[ii-1];
      p += 3;
    }
    k_loc += 3*sizeof(double)*(iq-is+1);
  }

  *k = k_loc;
}

extern "C" void pack_xyz_byte(int *ngroupl, int *groupl, int *group, char *buffer, int *k,
			      double *x, double *y, double *z) {
  
  int k_loc = *k;
  int ngroupl_loc = *ngroupl;
  char *tmp = (char *)buffer;
  double *p = (double *)&tmp[k_loc];

  int ig;
  for (ig=0;ig < ngroupl_loc;ig++) {
    int i = groupl[ig];
    int groupi = group[i-1];
    int is = groupi & 0xfffffff;
    int iq = is + ((groupi >> 28) & 7);
    int ii;
    for (ii=is;ii <= iq;ii++) {
      p[0] = x[ii-1];
      p[1] = y[ii-1];
      p[2] = z[ii-1];
      p += 3;
    }
    k_loc += 3*sizeof(double)*(iq-is+1);
  }

  *k = k_loc;
}

extern "C" void pack_xyz_atom_byte(int *natoml, int *atoml, char *buffer, int *k,
				   double *x, double *y, double *z) {

  char *tmp = (char *)buffer;
  double *p = (double *)&tmp[*k];
  int natoml_loc = *natoml;

  int i, j;
#pragma omp for schedule(static) private(j, i)
  for (j=0;j < natoml_loc;j++) {
    i = atoml[j] - 1;
    p[j*3]   = x[i];
    p[j*3+1] = y[i];
    p[j*3+2] = z[i];
  }

  *k += 3*sizeof(double)*natoml_loc;
}

//################################################################################
// unpack_xyz

extern "C" void unpack_xyz_group(int *ngroup, int *groupl, int *group, char *buffer, int *k,
				 double *x, double *y, double *z, int *natom, int *atomlist) {

  int natom_loc = *natom;
  int k_loc = *k;
  int ngroup_loc = *ngroup;
  char *tmp = (char *)buffer;
  double *p = (double *)&tmp[k_loc];

  int ig;
  for (ig=0;ig < ngroup_loc;ig++) {
    int j = groupl[ig];
    int groupj = group[j-1];
    int is = groupj & 0xfffffff;
    int iq = is + ((groupj >> 28) & 7);
    int ii;
    for (ii=is;ii <= iq;ii++) {
      x[ii-1] = p[0];
      y[ii-1] = p[1];
      z[ii-1] = p[2];
      p += 3;
      natom_loc++;
      atomlist[natom_loc - 1] = ii;
    }
    k_loc += 3*sizeof(double)*(iq-is+1);
  }

  *natom = natom_loc;
  *k = k_loc;
}

extern "C" void unpack_xyz_atom(int *natoml, int *atoml, char *buffer, int *k,
				double *x, double *y, double *z, int *atomlist) {

  int natoml_loc = *natoml;
  char *tmp = (char *)buffer;
  double *p = (double *)&tmp[*k];

  int j, i;
#pragma omp for schedule(static) private(j, i)
  for (j=0;j < natoml_loc;j++) {
    i = atoml[j] - 1;
    x[i] = p[j*3];
    y[i] = p[j*3+1];
    z[i] = p[j*3+2];
    atomlist[j] = i + 1;
  }

  *k += 3*sizeof(double)*natoml_loc;
}

//################################################################################
// pack_coord

extern "C" void pack_coord(int *ngroupl, int *groupl, int *grouppos, int *group, double *buffer, int *k,
			   double *x, double *y, double *z) {

  int k0 = *k;
  int ngroupl_loc = *ngroupl;
  char *tmp = (char *)buffer;

  int m = 0;

  int ig;
#pragma omp for schedule(static) private(ig)
  for (ig=0;ig < ngroupl_loc;ig++) {
    int i = groupl[ig];
    int groupi = group[i-1];
    int is = groupi & 0xfffffff;
    int iq = is + ((groupi >> 28) & 7);
    int ii;
    double *p = (double *)&tmp[k0 + 3*sizeof(double)*grouppos[ig]];
    for (ii=is;ii <= iq;ii++) {
      *p = x[ii-1];
      p++;
    }
    for (ii=is;ii <= iq;ii++) {
      *p = y[ii-1];
      p++;
    }
    for (ii=is;ii <= iq;ii++) {
      *p = z[ii-1];
      p++;
    }
    m += iq-is+1;
  }

  if (ngroupl_loc > 0) {
    int i = groupl[ngroupl_loc-1];
    int groupi = group[i-1];
    int is = groupi & 0xfffffff;
    int iq = is + ((groupi >> 28) & 7);
    *k += 3*sizeof(double)*(grouppos[ngroupl_loc-1] + (iq-is+1) );
  }
}

//################################################################################
// unpack_coord

extern "C" void unpack_coord(int *ngroup, int *groupl, int *group, double *buffer, int *k,
			     double *x, double *y, double *z) {

  int k_loc = *k;
  int ngroup_loc = *ngroup;
  char *tmp = (char *)buffer;
  double *p = (double *)&tmp[k_loc];

  int ig;
  for (ig=0;ig < ngroup_loc;ig++) {
    int j = groupl[ig];
    int groupj = group[j-1];
    int is = groupj & 0xfffffff;
    int iq = is + ((groupj >> 28) & 7);
    int ii;
    for (ii=is;ii <= iq;ii++) {
      x[ii-1] = *p;
      p++;
    }
    for (ii=is;ii <= iq;ii++) {
      y[ii-1] = *p;
      p++;
    }
    for (ii=is;ii <= iq;ii++) {
      z[ii-1] = *p;
      p++;
    }
    k_loc += 3*sizeof(double)*(iq-is+1);
  }

  *k = k_loc;
}

//################################################################################
// unpack_force

extern "C" void unpack_force(int *natoml, int *atoml, int *k, char *buffer,
			     double *forcex, double *forcey, double *forcez) {

  int natoml_loc = *natoml;
  char *tmp = (char *)buffer;
  double *p = (double *)&tmp[*k];

  int i;
  for (i=0;i < natoml_loc;i++) {
    int j = atoml[i] - 1;
    forcex[j] += p[0];
    forcey[j] += p[1];
    forcez[j] += p[2];
    p += 3;
  }

  *k += 3*sizeof(double)*natoml_loc;
}
