
#ifdef __SSE2__ /* __SSE2__ */
static inline __attribute__((always_inline)) void FIX_KERNEL_NAME(
			const int *i, const int *j, const int *is,
			const xyzq_sp_t *xyzq, const float *scoordtab,
			double *fijx, double *fijy, double *fijz,
#if (VDWTYPE == VSH)
			const __m128 roff6, const __m128 roff12, const __m128 roff18,
#endif
#if (VDWTYPE == VSW)
			const __m128 roff2, const __m128 ron2, const __m128 inv_roff2_ron2,
#endif
#if (VDWTYPE == VFSW)
			const __m128 k6, const __m128 k12, const __m128 dv6, const __m128 dv12,
			const __m128 ron2, const __m128 roffinv3, const __m128 roffinv6,
#endif
			const __m128 kappa, const __m128 neg_kappasq, const __m128 neg_kappa_fac,
			const __m128 qq_scale,
#if (INTTYPE == IN14)
			const int *vdwtype, const float *vdwparam, 
			const __m128 e14fac_minone_sp,
			double *dvdwpot,
#endif
			double *dcoulpot) {

  const __m128 one    = _mm_set1_ps(1.0f);
  const __m128 two    = _mm_set1_ps(2.0f);
  const __m128 three  = _mm_set1_ps(3.0f);
  const __m128 six    = _mm_set1_ps(6.0f);
  const __m128 twelve = _mm_set1_ps(12.0f);

  // Load i atom coordinates
  __m128 xi = _mm_load_ps((float *)&xyzq[i[0]]);
  __m128 yi = _mm_load_ps((float *)&xyzq[i[1]]);
  __m128 zi = _mm_load_ps((float *)&xyzq[i[2]]);
  __m128 qi = _mm_load_ps((float *)&xyzq[i[3]]);
  transpose_4x4_m128(xi, yi, zi, qi);
  
  // Get shift coordinate
  __m128 sx = load_4sp(&scoordtab[is[0]],   &scoordtab[is[1]],
		       &scoordtab[is[2]],   &scoordtab[is[3]]);
  __m128 sy = load_4sp(&scoordtab[is[0]+1], &scoordtab[is[1]+1],
		       &scoordtab[is[2]+1], &scoordtab[is[3]+1]);
  __m128 sz = load_4sp(&scoordtab[is[0]+2], &scoordtab[is[1]+2],
		       &scoordtab[is[2]+2], &scoordtab[is[3]+2]);
  // Load j atom coordinates
  __m128 xj = _mm_load_ps((float *)&xyzq[j[0]]);
  __m128 yj = _mm_load_ps((float *)&xyzq[j[1]]);
  __m128 zj = _mm_load_ps((float *)&xyzq[j[2]]);
  __m128 qj = _mm_load_ps((float *)&xyzq[j[3]]);
  transpose_4x4_m128(xj, yj, zj, qj);

  // Calculate distance
  __m128 dx = _mm_add_ps(_mm_sub_ps(xi, xj), sx);
  __m128 dy = _mm_add_ps(_mm_sub_ps(yi, yj), sy);
  __m128 dz = _mm_add_ps(_mm_sub_ps(zi, zj), sz);
  __m128 rsq = square_ps(dx, dy, dz);
  __m128 qq = _mm_mul_ps(qq_scale, _mm_mul_ps(qi, qj));
  // Calculate the interaction
  __m128 rinv = invsqrt_ps(rsq);
  __m128 r = _mm_mul_ps(rsq, rinv);

  float kappa_r_float[4], neg_kappasq_rsq_float[4], erfc_val_float[4], exp_val_float[4];

  __m128 kappa_r = _mm_mul_ps(kappa, r);
  _mm_storeu_ps(kappa_r_float, kappa_r);

  __m128 neg_kappasq_rsq = _mm_mul_ps(neg_kappasq, rsq);
  _mm_storeu_ps(neg_kappasq_rsq_float, neg_kappasq_rsq);

  int k;
#pragma simd assert
  for (k=0;k < 4;k++) erfc_val_float[k] = erfcf(kappa_r_float[k]);
  __m128 erfc_val = _mm_loadu_ps(erfc_val_float);

#pragma simd assert
  for (k=0;k < 4;k++) exp_val_float[k] = expf(neg_kappasq_rsq_float[k]);
  __m128 exp_val = _mm_loadu_ps(exp_val_float);

#if (INTTYPE == IN14) // opt14
  // 1-4 interaction

  int ia[4];
#pragma simd assert
  for (k=0;k < 4;k++) ia[k] = vdwtype[i[k]];

  int ja[4];
#pragma simd assert
  for (k=0;k < 4;k++) ja[k] = vdwtype[j[k]];

  int aa[4];
#pragma simd assert
  for (k=0;k < 4;k++) aa[k] = (ja[k] > ia[k]) ? ja[k] : ia[k];

  int ivdw[4];
#pragma simd assert
  for (k=0;k < 4;k++) ivdw[k] = aa[k]*(aa[k]-3) + 2*(ja[k] + ia[k]) - 2;

  __m128 c6 = load_4sp(&vdwparam[ivdw[0]], &vdwparam[ivdw[1]],
		       &vdwparam[ivdw[2]], &vdwparam[ivdw[3]]);

  __m128 c12 = load_4sp(&vdwparam[ivdw[0]+1], &vdwparam[ivdw[1]+1],
			&vdwparam[ivdw[2]+1], &vdwparam[ivdw[3]+1]);

  __m128 rinv2 = _mm_mul_ps(rinv, rinv);
#if (VDWTYPE == VSH)
  __m128 r6 = _mm_mul_ps(rsq, _mm_mul_ps(rsq, rsq));
  __m128 rinv6 = _mm_mul_ps(rinv2, _mm_mul_ps(rinv2, rinv2));
  __m128 rinv12 = _mm_mul_ps(rinv6, rinv6);

  __m128 dpot1 = _mm_mul_ps(c12,_mm_add_ps(rinv12,
					   _mm_sub_ps(_mm_mul_ps(r6,_mm_mul_ps(two,roff18)),
						      _mm_mul_ps(three,roff12))));
  __m128 dpot2 = _mm_mul_ps(c6,_mm_add_ps(rinv6,
					  _mm_sub_ps(_mm_mul_ps(r6,roff12), _mm_mul_ps(two,roff6))));
  store_4sp_4dp(_mm_sub_ps(dpot1,dpot2), dvdwpot);

  __m128 fij_vdw1 = _mm_mul_ps(six,_mm_mul_ps(c6,_mm_sub_ps(rinv6, _mm_mul_ps(r6,roff12))));
  __m128 fij_vdw2 = _mm_mul_ps(twelve,_mm_mul_ps(c12,_mm_add_ps(rinv12,_mm_mul_ps(r6,roff18))));
  __m128 fij_vdw = _mm_sub_ps(fij_vdw1, fij_vdw2);
#elif (VDWTYPE == VSW)

  __m128 roff2_r2_sq = _mm_sub_ps(roff2, rsq);
  roff2_r2_sq = _mm_mul_ps(roff2_r2_sq, roff2_r2_sq);
 
  // _mm_cmpgt_ps returns: if (rsq > ron2)  0xffffffff 
  //                       if (rsq <= ron2) 0x0
  __m128 mask = _mm_cmpgt_ps(rsq, ron2);

  __m128 sw = _mm_mul_ps(_mm_mul_ps(roff2_r2_sq,
				    _mm_sub_ps(_mm_add_ps(roff2, 
							  _mm_mul_ps(two,rsq)), 
					       _mm_mul_ps(three,ron2))), inv_roff2_ron2);

  __m128 dsw_6 = _mm_mul_ps(_mm_sub_ps(roff2,rsq),_mm_mul_ps(_mm_sub_ps(ron2, rsq), inv_roff2_ron2));
  dsw_6 = _mm_and_ps(mask, dsw_6);

  sw = _mm_and_ps(mask, sw);
  mask = _mm_cmple_ps(rsq, ron2); // make inverse mask:
  // _mm_cmple_ps returns: if (rsq <= ron2) 0xffffffff 
  //                       if (rsq > ron2)  0x0
  sw = _mm_add_ps(sw, _mm_and_ps(one, mask));

  //float sw = (rsq <= ron2) ? 1.0f : 
  //  roff2_r2_sq*(roff2 + two*rsq - three*ron2)*inv_roff2_ron2;

  // dsw_6 = dsw/6.0

  //  float dsw_6 = (r2 <= ron2) ? 0.0f : 
  //  (roff2-r2)*(ron2-r2)*inv_roff2_ron2;

  __m128 rinv4 = _mm_mul_ps(rinv2, rinv2);
  __m128 rinv6 = _mm_mul_ps(rinv4, rinv2);

  __m128 c12_rinv6 = _mm_mul_ps(c12, rinv6);

  __m128 sw_rinv2 = _mm_mul_ps(sw, rinv2);

  __m128 tmp12 = _mm_mul_ps(twelve,_mm_mul_ps(c12, _mm_mul_ps(rinv6, _mm_sub_ps(dsw_6, sw_rinv2))));
  __m128 tmp6  = _mm_mul_ps(six, _mm_mul_ps(c6, _mm_sub_ps(_mm_mul_ps(two, dsw_6), sw_rinv2)));

  __m128 fij_vdw = _mm_mul_ps(rinv4, _mm_sub_ps(tmp12, tmp6));

  __m128 dpot = _mm_mul_ps(sw, _mm_mul_ps(rinv6, _mm_sub_ps(c12_rinv6, c6)));

  store_4sp_4dp(dpot, dvdwpot);

#elif (VDWTYPE == VFSW)

  __m128 rinv3 = _mm_mul_ps(rinv, rinv2);
  __m128 rinv6 = _mm_mul_ps(rinv3, rinv3);

  // Mask
  // _mm_cmpgt_ps returns: if (rsq > ron2)  0xffffffff 
  //                       if (rsq <= ron2) 0x0
  __m128 mask =  _mm_cmpgt_ps(rsq, ron2);

  // Inverse mask
  // _mm_cmple_ps returns: if (rsq <= ron2) 0xffffffff 
  //                       if (rsq > ron2)  0x0
  __m128 imask = _mm_cmple_ps(rsq, ron2);
  // one_mask has 1.0 if (rsq <= ron), 0.0 otherwise
  __m128 one_mask = _mm_and_ps(one, imask);

  __m128 A6  = _mm_add_ps(_mm_and_ps(k6,  mask), one_mask);
  __m128 A12 = _mm_add_ps(_mm_and_ps(k12, mask), one_mask);

  __m128 B6  = _mm_and_ps(roffinv3, mask);
  __m128 B12 = _mm_and_ps(roffinv6, mask);

  //float A6 = (r2 > d_setup.ron2) ? d_setup.k6 : 1.0f;
  //float B6 = (r2 > d_setup.ron2) ? d_setup.roffinv3  : 0.0f;
  //float A12 = (r2 > d_setup.ron2) ? d_setup.k12 : 1.0f;
  //float B12 = (r2 > d_setup.ron2) ? d_setup.roffinv6 : 0.0f;

  __m128 rinv3_B6  = _mm_sub_ps(rinv3, B6);
  __m128 rinv6_B12 = _mm_sub_ps(rinv6, B12);

  __m128 fij_vdw = _mm_sub_ps(_mm_mul_ps(six,_mm_mul_ps(c6,_mm_mul_ps(A6,_mm_mul_ps(rinv3_B6,rinv3)))), 
			      _mm_mul_ps(twelve,_mm_mul_ps(c12,_mm_mul_ps(A12,_mm_mul_ps(rinv6_B12,rinv6)))));

  __m128 C6  = _mm_and_ps(dv6,  imask);
  __m128 C12 = _mm_and_ps(dv12, imask);

  //float C6  = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv6;
  //float C12 = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv12;

  __m128 rinv3_B6_sq = _mm_mul_ps(rinv3_B6, rinv3_B6);
  __m128 rinv6_B12_sq = _mm_mul_ps(rinv6_B12, rinv6_B12);

  __m128 dpot = _mm_sub_ps(_mm_mul_ps(c12,_mm_add_ps(_mm_mul_ps(A12,rinv6_B12_sq), C12)),
			   _mm_mul_ps(c6,_mm_add_ps(_mm_mul_ps(A6,rinv3_B6_sq), C6)));

  store_4sp_4dp(dpot, dvdwpot);

#elif (VDWTYPE == NONE)
  __m128 fij_vdw = _mm_setzero_ps();
  dvdwpot[0] = 0.0;
  dvdwpot[1] = 0.0;
  dvdwpot[2] = 0.0;
  dvdwpot[3] = 0.0;
#endif

#if (ELECTYPE == EWALD)
  __m128 qq_efac_rinv = _mm_mul_ps(qq,_mm_mul_ps(_mm_add_ps(erfc_val,e14fac_minone_sp),rinv));
  store_4sp_4dp(qq_efac_rinv, dcoulpot);
    
  __m128 fij_elec = _mm_sub_ps(_mm_mul_ps(qq,_mm_mul_ps(neg_kappa_fac,exp_val)), qq_efac_rinv);
#elif (ELECTYPE == NONE)
  __m128 fij_elec = _mm_setzero_ps();
  dcoulpot[0] = 0.0;
  dcoulpot[1] = 0.0;
  dcoulpot[2] = 0.0;
  dcoulpot[3] = 0.0;
#endif
    
  __m128 fij = _mm_mul_ps(_mm_add_ps(fij_vdw,fij_elec),rinv2);
    
#else // opt14

  // 1-4 exclusion
  __m128 rinv2 = _mm_mul_ps(rinv, rinv);
  __m128 qq_efac_rinv = _mm_mul_ps(qq,_mm_mul_ps(rinv,_mm_sub_ps(erfc_val, one)));
  store_4sp_4dp(qq_efac_rinv, dcoulpot);

  __m128 fij = _mm_mul_ps(rinv2,_mm_sub_ps(_mm_mul_ps(qq,
						      _mm_mul_ps(neg_kappa_fac,exp_val)),
					   qq_efac_rinv));
#endif // opt14

  // Calculate force components
  __m128 fij_dx = _mm_mul_ps(fij, dx);
  __m128 fij_dy = _mm_mul_ps(fij, dy);
  __m128 fij_dz = _mm_mul_ps(fij, dz);

  store_4sp_4dp(fij_dx, fijx);
  store_4sp_4dp(fij_dy, fijy);
  store_4sp_4dp(fij_dz, fijz);

}
#endif

void KERNEL_NAME(const int *ii_start_in, const int *ii_end_in,
		 const list14_t *xx14list,
#if (INTTYPE == IN14)
		 const int *vdwtype, const float *vdwparam, const float *ron_in,
		 const float *roff_in, const float *e14fac_in,
#endif
		 const float *kappa_in, const float *qq_scale_in,
		 const xyzq_sp_t *xyzq, const float *scoordtab,
		 double *force, double *sforce,
#if (INTTYPE == IN14)
		 double *vdwpot,
#endif
		 double *coulpot) {

#ifndef __SSE2__ /* __SSE2__ */

  printf("enbfix14_sse.h: calling #KERNEL_NAME when it is not compiled\n");
  exit(1);

#else /* __SSE2__ */
#if (VDWTYPE == VSH)
  float roff2_float = (*roff_in)*(*roff_in);
  float roff6_float = 1.0f/(roff2_float*roff2_float*roff2_float);
  float roff12_float = roff6_float*roff6_float;
  float roff18_float = roff12_float*roff6_float;
  __m128 roff6 = _mm_set1_ps(roff6_float);
  __m128 roff12 = _mm_set1_ps(roff12_float);
  __m128 roff18 = _mm_set1_ps(roff18_float);
#endif

#if (VDWTYPE == VSW)
  float roff2_float = (*roff_in)*(*roff_in);
  float ron2_float = (*ron_in)*(*ron_in);
  float roff2_min_ron2 = roff2_float - ron2_float;
  float inv_roff2_ron2_float = 1.0f/(roff2_min_ron2*roff2_min_ron2*roff2_min_ron2);
  __m128 roff2 = _mm_set1_ps(roff2_float);
  __m128 ron2 = _mm_set1_ps(ron2_float);
  __m128 inv_roff2_ron2 = _mm_set1_ps(inv_roff2_ron2_float);
#endif

#if (VDWTYPE == VFSW)
  float roff3_float = (*roff_in)*(*roff_in)*(*roff_in);
  float ron3_float = (*ron_in)*(*ron_in)*(*ron_in);
  float roff6_float = roff3_float*roff3_float;
  float ron6_float = ron3_float*ron3_float;
  float k6_float, k12_float, dv6_float, dv12_float;
  if (*ron_in < *roff_in) {
    k6_float = roff3_float/(roff3_float - ron3_float);
    k12_float = roff6_float/(roff6_float - ron6_float);
    dv6_float = -1.0f/(ron3_float*roff3_float);
    dv12_float = -1.0f/(ron6_float*roff6_float);
  } else {
    k6_float = 1.0f;
    k12_float = 1.0f;
    dv6_float = -1.0f/(roff6_float);
    dv12_float = -1.0f/(roff6_float*roff6_float);
  }
  float ron2_float = (*ron_in)*(*ron_in);
  float roffinv3_float =  1.0f/roff3_float;
  float roffinv6_float =  1.0f/roff6_float;
  __m128 k6 = _mm_set1_ps(k6_float);
  __m128 k12 = _mm_set1_ps(k12_float);
  __m128 dv6 = _mm_set1_ps(dv6_float);
  __m128 dv12 = _mm_set1_ps(dv12_float);
  __m128 ron2 = _mm_set1_ps(ron2_float);
  __m128 roffinv3 = _mm_set1_ps(roffinv3_float);
  __m128 roffinv6 = _mm_set1_ps(roffinv6_float);
#endif

  __m128 kappa = _mm_set1_ps(*kappa_in);
  __m128 qq_scale = _mm_set1_ps(*qq_scale_in);
  __m128 neg_kappasq = _mm_set1_ps(-(*kappa_in)*(*kappa_in));
  __m128 neg_kappa_fac = _mm_set1_ps(-1.12837916709551f*(*kappa_in));
#if (INTTYPE == IN14)
  __m128 e14fac_minone_sp = _mm_set1_ps(*e14fac_in - 1.0f);
#endif

  int ii_start = *ii_start_in - 1;
  int ii_end = *ii_end_in - 1;
  int n_round4 = ((ii_end - ii_start + 1)/4)*4;
  int ii_end_round4 = ii_start + n_round4 - 1;

#pragma omp barrier
  int ii;
  for (ii=ii_start;ii <= ii_end_round4;ii+=4) {

    double fijx[4], fijy[4], fijz[4];

    int i[4], j[4], is[4];

    i[0] = xx14list[ii].i;
    j[0] = xx14list[ii].j;
    is[0] = xx14list[ii].ishift*3;
    i[1] = xx14list[ii+1].i;
    j[1] = xx14list[ii+1].j;
    is[1] = xx14list[ii+1].ishift*3;
    i[2] = xx14list[ii+2].i;
    j[2] = xx14list[ii+2].j;
    is[2] = xx14list[ii+2].ishift*3;
    i[3] = xx14list[ii+3].i;
    j[3] = xx14list[ii+3].j;
    is[3] = xx14list[ii+3].ishift*3;

    double dcoulpot[4];
#if (INTTYPE == IN14)
    double dvdwpot[4];
#endif

    FIX_KERNEL_NAME(i, j, is, xyzq, scoordtab, 
		    fijx, fijy, fijz, 
#if (VDWTYPE == VSH)
		    roff6, roff12, roff18,
#endif
#if (VDWTYPE == VSW)
		    roff2, ron2, inv_roff2_ron2,
#endif
#if (VDWTYPE == VFSW)
		    k6, k12, dv6, dv12, ron2, roffinv3, roffinv6,
#endif
		    kappa, neg_kappasq, neg_kappa_fac, qq_scale,
#if (INTTYPE == IN14)
		    vdwtype, vdwparam,
		    e14fac_minone_sp,
		    dvdwpot,
#endif
		    dcoulpot);
 
    *coulpot += dcoulpot[0] + dcoulpot[1] + dcoulpot[2] + dcoulpot[3];
#if (INTTYPE == IN14)
    *vdwpot += dvdwpot[0] + dvdwpot[1] + dvdwpot[2] + dvdwpot[3];
#endif

    // Store forces
    j[0] *= 3;
    j[1] *= 3;
    j[2] *= 3;
    j[3] *= 3;
    force[j[0]]   -= fijx[0];
    force[j[0]+1] -= fijy[0];
    force[j[0]+2] -= fijz[0];
    force[j[1]]   -= fijx[1];
    force[j[1]+1] -= fijy[1];
    force[j[1]+2] -= fijz[1];
    force[j[2]]   -= fijx[2];
    force[j[2]+1] -= fijy[2];
    force[j[2]+2] -= fijz[2];
    force[j[3]]   -= fijx[3];
    force[j[3]+1] -= fijy[3];
    force[j[3]+2] -= fijz[3];

    i[0] *= 3;
    i[1] *= 3;
    i[2] *= 3;
    i[3] *= 3;
    force[i[0]]   += fijx[0];
    force[i[0]+1] += fijy[0];
    force[i[0]+2] += fijz[0];
    force[i[1]]   += fijx[1];
    force[i[1]+1] += fijy[1];
    force[i[1]+2] += fijz[1];
    force[i[2]]   += fijx[2];
    force[i[2]+1] += fijy[2];
    force[i[2]+2] += fijz[2];
    force[i[3]]   += fijx[3];
    force[i[3]+1] += fijy[3];
    force[i[3]+2] += fijz[3];

    // Store shifted forces
    sforce[is[0]]   += fijx[0];
    sforce[is[0]+1] += fijy[0];
    sforce[is[0]+2] += fijz[0];
    sforce[is[1]]   += fijx[1];
    sforce[is[1]+1] += fijy[1];
    sforce[is[1]+2] += fijz[1];
    sforce[is[2]]   += fijx[2];
    sforce[is[2]+1] += fijy[2];
    sforce[is[2]+2] += fijz[2];
    sforce[is[3]]   += fijx[3];
    sforce[is[3]+1] += fijy[3];
    sforce[is[3]+2] += fijz[3];

  }

  int klen = ii_end - ii + 1;
  if (klen > 0) {

    double fijx[4], fijy[4], fijz[4];

    int i[4], j[4], is[4];

    int k;

    for (k=0;k < klen;k++) {
      i[k]  = xx14list[ii+k].i;
      j[k]  = xx14list[ii+k].j;
      is[k] = xx14list[ii+k].ishift*3;
    }
    for (;k < 4;k++) {
      i[k] = i[0];
      j[k] = j[0];
      is[k] = is[0];
    }

    double dcoulpot[4];
#if (INTTYPE == IN14)
    double dvdwpot[4];
#endif

    FIX_KERNEL_NAME(i, j, is, xyzq, scoordtab, 
		    fijx, fijy, fijz, 
#if (VDWTYPE == VSH)
		    roff6, roff12, roff18,
#endif
#if (VDWTYPE == VSW)
		    roff2, ron2, inv_roff2_ron2,
#endif
#if (VDWTYPE == VFSW)
		    k6, k12, dv6, dv12, ron2, roffinv3, roffinv6,
#endif
		    kappa, neg_kappasq, neg_kappa_fac, qq_scale,
#if (INTTYPE == IN14)
		    vdwtype, vdwparam,
		    e14fac_minone_sp,
		    dvdwpot,
#endif
		    dcoulpot);

    for (k=0;k < klen;k++) *coulpot += dcoulpot[k];
#if (INTTYPE == IN14)
    for (k=0;k < klen;k++) *vdwpot += dvdwpot[k];
#endif

    // Store forces
    for (k=0;k < klen;k++) {
      force[j[k]*3]   -= fijx[k];
      force[j[k]*3+1] -= fijy[k];
      force[j[k]*3+2] -= fijz[k];
    }

    for (k=0;k < klen;k++) {
      force[i[k]*3]   += fijx[k];
      force[i[k]*3+1] += fijy[k];
      force[i[k]*3+2] += fijz[k];
    }

    // Store shifted forces
    for (k=0;k < klen;k++) {
      sforce[is[k]]   += fijx[k];
      sforce[is[k]+1] += fijy[k];
      sforce[is[k]+2] += fijz[k];
    }

  }

  return;
#endif /* __SSE2__ */
}
