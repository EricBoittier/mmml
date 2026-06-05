#define PASTE(a) a
#define ADD_PREC(name) PASTE(name)PASTE(_)COLFFT_PREC
#define ADD_ORTHO(name) PASTE(name)PASTE(_)ISORTHO

! this code relies on the C Preprocessor
! # define COLFFT_PREC to be either
! PS => SINGLE and set SINGLEP <= 1
! PD => DOUBLE and set DOUBLEP <= 1

! # define ISORTHO to be either
! O0 => nonorthogonal and set ORTHOG <=1
! O1 => orthogonal and set ORTHOG <= 0

! ##EXP O0 non_orthog orthog .when. EXP (exp_orthog)
! ##PASS1 O0 non_orthog
! ##PASS2 O1 orthog
! ##EXEND

  subroutine ADD_PREC(ADD_ORTHO(scalar_sumrc_kernel))(nfft1, nfft2, nfft3, nf1, nf2, nf3, &
       xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax, &
       zgridsize, xzgridsize, &
       fac, piv_inv, prefac1, prefac2, prefac3, recip, &
#if ORTHOG==1
       m1_tbl, m2_tbl, m3_tbl, &  
#endif
       qlen, q, energy, virial, q_calc_energy_virial)
#if KEY_PARALLEL==1
    use parallel,only:mynod  
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: nfft1, nfft2, nfft3
    integer, intent(in) :: nf1, nf2, nf3
    integer, intent(in) :: xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax
    integer, intent(in) :: zgridsize, xzgridsize
    integer, intent(in) :: qlen
#if DOUBLEP==1
    real(chm_real), intent(in) :: fac, piv_inv
    real(chm_real), intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
    real(chm_real), intent(in) :: recip(3,3)
#if ORTHOG==1
    real(chm_real), intent(in) :: m1_tbl(-(nfft1/2+1):(nfft1/2+1))
    real(chm_real), intent(in) :: m2_tbl(-(nfft2/2+1):(nfft2/2+1))
    real(chm_real), intent(in) :: m3_tbl(-(nfft3/2+1):(nfft3/2+1))
#endif 
    real(chm_real), intent(inout) :: q(0:qlen-1)
#else /**/
    real(chm_real4), intent(in) :: fac, piv_inv
    real(chm_real4), intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
    real(chm_real4), intent(in) :: recip(3,3)
#if ORTHOG==1
    real(chm_real4), intent(in) :: m1_tbl(-(nfft1/2+1):(nfft1/2+1))
    real(chm_real4), intent(in) :: m2_tbl(-(nfft2/2+1):(nfft2/2+1))
    real(chm_real4), intent(in) :: m3_tbl(-(nfft3/2+1):(nfft3/2+1))
#endif 
    real(chm_real4), intent(inout) :: q(0:qlen-1)
#endif 
    real(chm_real), intent(inout) :: energy, virial(6)
    logical, intent(in) :: q_calc_energy_virial
    ! Parameters
#if DOUBLEP==1
    real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real, two = 2.0_chm_real
#elif SINGLEP==1
    real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4, two = 2.0_chm_real4
#endif 
    real(chm_real), parameter :: zero_dp = 0.0_chm_real
    ! Variables
#if DOUBLEP==1
    real(chm_real) eterm,vterm
    real(chm_real) mhat1,mhat2,mhat3,msq,struc2,msq_inv,msqs_inv
    real(chm_real) mhat1s,mhat2s,mhat3s,msqs
    real(chm_real) struc2s,eterms,vterms
    real(chm_real) tmp1,tmp2,tmp1s,tmp2s,m2_m1_tbl,m2_m1_tbls
#if ORTHOG==1
    real(chm_real) recip11,recip22,recip33  
#endif
    real(chm_real) mhat11, mhat22, mhat33, mhat12, mhat13, mhat23
    real(chm_real) mhat11s, mhat22s, mhat12s, mhat13s, mhat23s
    real(chm_real) prefac12_piv_inv, prefac12s_piv_inv
#elif SINGLEP==1
    real(chm_real4) eterm,vterm
    real(chm_real4) mhat1,mhat2,mhat3,msq,struc2,msq_inv,msqs_inv
    real(chm_real4) mhat1s,mhat2s,mhat3s,msqs
    real(chm_real4) struc2s,eterms,vterms
    real(chm_real4) tmp1,tmp2,tmp1s,tmp2s,m2_m1_tbl,m2_m1_tbls
#if ORTHOG==1
    real(chm_real4) recip11,recip22,recip33  
#endif
    real(chm_real4) mhat11, mhat22, mhat33, mhat12, mhat13, mhat23
    real(chm_real4) mhat11s, mhat22s, mhat12s, mhat13s, mhat23s
    real(chm_real4) prefac12_piv_inv, prefac12s_piv_inv
!    real(chm_real4) dvirial1, dvirial2, dvirial3, dvirial4, dvirial5, dvirial6
!    real(chm_real4) denergy
#endif 
    real(chm_real) denergy
    real(chm_real) dvirial1, dvirial2, dvirial3, dvirial4, dvirial5, dvirial6
    integer k3_start,k3_stop,k1s,k2s,k3s,m1s,m2s,m3s
    integer k2q,k1, k2, k3, m1, m2, m3, nff
    integer index, index2
#if KEY_PARALLEL==1
    logical master  
#endif

#if KEY_PARALLEL==1
    master = (mynod == 0)  
#endif

#if ORTHOG==1 /*orthog*/
    ! ######### orthog ########
    recip11 = recip(1,1)
    recip22 = recip(2,2)
    recip33 = recip(3,3)

!$omp do schedule(static) reduction(+:energy, virial)
    do k2q = 1, ygridmax-ygridmin+1
       k2 = k2q + ygridmin
       k2s = mod(nfft2-k2+1,nfft2) + 1

       m2 = k2 - 1
       if (k2 > nf2) m2 = m2 - nfft2
       mhat2 = recip22*m2
       m2s = k2s - 1
       if (k2s > nf2) m2s = m2s - nfft2
       mhat2s = recip22*m2s

       mhat22 = mhat2*mhat2
       mhat22s = mhat2s*mhat2s

       index2 = (k2q-1)*xzgridsize

       ! Loop over x
       do k1 = xgridmin+1, xgridmax+1
          k1s = nfft1 - k1 + 2
          m1 = k1 - 1
          if ( k1 > nf1 ) m1 = m1 - nfft1
          mhat1 = recip11*m1
          m1s = k1s - 1
          if ( k1s > nf1 ) m1s = m1s - nfft1
          mhat1s = recip11*m1s

          mhat11 = mhat1*mhat1
          mhat11s = mhat1s*mhat1s
          mhat12 = mhat1*mhat2
          mhat12s = mhat1s*mhat2s

          m2_m1_tbl  = piv_inv*m2_tbl(m2)*m1_tbl(m1)*prefac2(k2)*prefac1(k1)

          k3_start = zgridmin + 1

          ! This is used to skip the k = 0 term
#if KEY_PARALLEL==1
          if(master .and. (k1+k2 == 2) ) k3_start = 2  
#else
          if(k1+k2 == 2) k3_start = 2  
#endif

          index = 2*(k3_start-1 + (k1-xgridmin-1)*zgridsize + index2)

          ! Loop over z
          if (k1 <= 1 .or. k1 > nfft1) then
             ! k3 <= nf3
             dvirial1 = zero_dp
             dvirial2 = zero_dp
             dvirial3 = zero_dp
             dvirial4 = zero_dp
             dvirial5 = zero_dp
             dvirial6 = zero_dp
             denergy = zero_dp
!dec$ simd assert reduction(+:denergy, dvirial1, dvirial2, dvirial3, dvirial4, dvirial5, dvirial6)
             do k3 = k3_start,nf3
                mhat3 = recip33*(k3-1)
                mhat33 = mhat3*mhat3

                msq = mhat11 + mhat22 + mhat33
                msq_inv = one/msq

                eterm  = m3_tbl(k3 - 1)*prefac3(k3)*m2_m1_tbl *msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index) = eterm*q(index)
                q(index+1) = eterm*q(index+1)
                index = index + 2

                if (q_calc_energy_virial) then
                   tmp1  = eterm*struc2
                   vterm  = two*(fac + msq_inv)   !two*(fac*msq  + one)*msq_inv
                   tmp2   = tmp1*vterm
                   denergy = denergy + tmp1

                   mhat13 = mhat1*mhat3
                   mhat23 = mhat2*mhat3

                   dvirial1 = dvirial1 + tmp1*(vterm*mhat11 - one)
                   dvirial2 = dvirial2 + tmp2*mhat12
                   dvirial3 = dvirial3 + tmp2*mhat13
                   dvirial4 = dvirial4 + tmp1*(vterm*mhat22 - one)
                   dvirial5 = dvirial5 + tmp2*mhat23
                   dvirial6 = dvirial6 + tmp1*(vterm*mhat33 - one)
                endif
             enddo
             virial(1) = virial(1) + dvirial1
             virial(2) = virial(2) + dvirial2
             virial(3) = virial(3) + dvirial3
             virial(4) = virial(4) + dvirial4
             virial(5) = virial(5) + dvirial5
             virial(6) = virial(6) + dvirial6
             energy = energy + denergy

             ! k3 > nf3
             dvirial1 = zero_dp
             dvirial2 = zero_dp
             dvirial3 = zero_dp
             dvirial4 = zero_dp
             dvirial5 = zero_dp
             dvirial6 = zero_dp
             denergy = zero_dp
!dec$ simd assert reduction(+:denergy, dvirial1, dvirial2, dvirial3, dvirial4, dvirial5, dvirial6)
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3
                mhat3 = recip33*m3
                mhat33 = mhat3*mhat3

                msq  = mhat11   + mhat22   + mhat33
                msq_inv = one/msq

                eterm  = m3_tbl(m3) * prefac3(k3) * m2_m1_tbl  * msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index) = eterm*q(index)
                q(index+1) = eterm*q(index+1)
                index = index + 2

                if (q_calc_energy_virial) then
                   tmp1   = eterm*struc2
                   vterm  = two*(fac + msq_inv)   !two*(fac*msq + one)*msq_inv
                   tmp2   = tmp1*vterm
                   denergy = denergy + tmp1

                   mhat13 = mhat1*mhat3
                   mhat23 = mhat2*mhat3

                   dvirial1 = dvirial1 + tmp1*(vterm*mhat11 - one)
                   dvirial2 = dvirial2 + tmp2*mhat12
                   dvirial3 = dvirial3 + tmp2*mhat13
                   dvirial4 = dvirial4 + tmp1*(vterm*mhat22 - one)
                   dvirial5 = dvirial5 + tmp2*mhat23
                   dvirial6 = dvirial6 + tmp1*(vterm*mhat33 - one)
                endif
             enddo
             virial(1) = virial(1) + dvirial1
             virial(2) = virial(2) + dvirial2
             virial(3) = virial(3) + dvirial3
             virial(4) = virial(4) + dvirial4
             virial(5) = virial(5) + dvirial5
             virial(6) = virial(6) + dvirial6
             energy = energy + denergy
          else
             m2_m1_tbls = piv_inv*m2_tbl(m2s)*m1_tbl(m1s)*prefac2(k2s)*prefac1(k1s)
             ! k3 <= nf3
             dvirial1 = zero_dp
             dvirial2 = zero_dp
             dvirial3 = zero_dp
             dvirial4 = zero_dp
             dvirial5 = zero_dp
             dvirial6 = zero_dp
             denergy = zero_dp
!dec$ simd assert reduction(+:denergy, dvirial1, dvirial2, dvirial3, dvirial4, dvirial5, dvirial6)
             do k3 = k3_start,nf3
                m3 = k3 - 1
                mhat3 = recip33*m3
                mhat33 = mhat3*mhat3

                msqs = mhat11s + mhat22s + mhat33
                msq = mhat11 + mhat22 + mhat33
                msq_inv = one/msq
                msqs_inv = one/msqs

                eterms = m3_tbl(m3)*prefac3(k3)
                eterm  = eterms
                eterms = eterms*m2_m1_tbls*msqs_inv
                eterm  = eterm *m2_m1_tbl *msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index) = eterm*q(index)
                q(index+1) = eterm*q(index+1)
                index = index + 2

                if (q_calc_energy_virial) then
                   tmp1s = eterms*struc2
                   tmp1  = eterm*struc2
                   vterms = two*(fac + msqs_inv)   !two*(fac*msqs + one)*msqs_inv
                   vterm  = two*(fac + msq_inv)    !two*(fac*msq  + one)*msq_inv
                   tmp2s  = tmp1s*vterms
                   tmp2   = tmp1*vterm
                   denergy = denergy + tmp1s + tmp1

                   mhat13 = mhat1*mhat3
                   mhat13s = mhat1s*(-mhat3)
                   mhat23 = mhat2*mhat3
                   mhat23s = mhat2s*(-mhat3)

                   dvirial1 = dvirial1 + tmp1s*(vterms*mhat11s - one) + tmp1*(vterm*mhat11 - one)
                   dvirial2 = dvirial2 + tmp2s*mhat12s + tmp2*mhat12
                   dvirial3 = dvirial3 + tmp2s*mhat13s + tmp2*mhat13
                   dvirial4 = dvirial4 + tmp1s*(vterms*mhat22s - one) + tmp1*(vterm*mhat22 - one)
                   dvirial5 = dvirial5 + tmp2s*mhat23s + tmp2*mhat23
                   dvirial6 = dvirial6 + tmp1s*(vterms*mhat33 - one) + tmp1*(vterm*mhat33 - one)
                endif
             enddo
             virial(1) = virial(1) + dvirial1
             virial(2) = virial(2) + dvirial2
             virial(3) = virial(3) + dvirial3
             virial(4) = virial(4) + dvirial4
             virial(5) = virial(5) + dvirial5
             virial(6) = virial(6) + dvirial6
             energy = energy + denergy
             ! k3 > nf3
             dvirial1 = zero_dp
             dvirial2 = zero_dp
             dvirial3 = zero_dp
             dvirial4 = zero_dp
             dvirial5 = zero_dp
             dvirial6 = zero_dp
             denergy = zero_dp
!dec$ simd assert reduction(+:denergy, dvirial1, dvirial2, dvirial3, dvirial4, dvirial5, dvirial6)
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3
                mhat3 = recip33*m3
                mhat33 = mhat3*mhat3

                msqs = mhat11s + mhat22s + mhat33
                msq  = mhat11   + mhat22   + mhat33
                msqs_inv = one/msqs
                msq_inv = one/msq

                eterms = m3_tbl(m3)*prefac3(k3)
                eterm = eterms
                eterms = eterms * m2_m1_tbls * msqs_inv
                eterm  = eterm  * m2_m1_tbl  * msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index) = eterm*q(index)
                q(index+1) = eterm*q(index+1)
                index = index + 2

                if (q_calc_energy_virial) then
                   tmp1s  = eterms*struc2
                   tmp1   = eterm*struc2
                   vterms = two*(fac*msqs + one)*msqs_inv
                   vterm  = two*(fac*msq + one)*msq_inv
                   tmp2s  = tmp1s*vterms
                   tmp2   = tmp1*vterm
                   denergy = denergy + tmp1s + tmp1
                   
                   mhat13 = mhat1*mhat3
                   mhat13s = mhat1s*(-mhat3)
                   mhat23 = mhat2*mhat3
                   mhat23s = mhat2s*(-mhat3)
                   
                   dvirial1 = dvirial1 + tmp1s*(vterms*mhat11s - one) + tmp1*(vterm*mhat11 - one)
                   dvirial2 = dvirial2 + tmp2s*mhat12s + tmp2*mhat12
                   dvirial3 = dvirial3 + tmp2s*mhat13s + tmp2*mhat13
                   dvirial4 = dvirial4 + tmp1s*(vterms*mhat22s - one) + tmp1*(vterm*mhat22 - one)
                   dvirial5 = dvirial5 + tmp2s*mhat23s + tmp2*mhat23
                   dvirial6 = dvirial6 + tmp1s*(vterms*mhat33 - one) + tmp1*(vterm*mhat33 - one)
                endif
             enddo
             virial(1) = virial(1) + dvirial1
             virial(2) = virial(2) + dvirial2
             virial(3) = virial(3) + dvirial3
             virial(4) = virial(4) + dvirial4
             virial(5) = virial(5) + dvirial5
             virial(6) = virial(6) + dvirial6
             energy = energy + denergy
          endif
       enddo
    enddo
!$omp end do

#else /* (orthog)*/
    ! ######### non-orthog ########

!$omp do schedule(static) reduction(+:energy, virial)
    do k2q = 1, ygridmax-ygridmin+1
#if KEY_PARALLEL==1
       if(master)then  
#endif
          k2 = k2q + ygridmin
          k2s = mod(nfft2-k2+1,nfft2) + 1
#if KEY_PARALLEL==1
       else
          k2 = k2q + ygridmin
          k2s = mod(nfft2-k2+1,nfft2) + 1
       end if
#endif 
       m2 = k2 - 1
       if (k2 > nf2) m2 = m2 - nfft2
       m2s = k2s - 1
       if (k2s > nf2) m2s = m2s - nfft2

       index2 = (k2q-1)*xzgridsize

       ! Loop over x
       do k1 = xgridmin+1, xgridmax+1
          k1s = nfft1 - k1 + 2
          m1 = k1 - 1
          if ( k1 > nf1 ) m1 = m1 - nfft1
          m1s = k1s - 1
          if ( k1s > nf1 ) m1s = m1s - nfft1

          prefac12_piv_inv = piv_inv*prefac2(k2)*prefac1(k1)

          k3_start = zgridmin+1

          ! This is used to skip the k = 0 term
#if KEY_PARALLEL==1
          if(master .and. (k1+k2 == 2) ) k3_start = 2  
#endif
#if KEY_PARALLEL==0
          if(k1+k2 == 2) k3_start = 2  
#endif

          index = 2*(k3_start-1 + (k1-xgridmin-1)*zgridsize + index2)

          ! Loop over z
          if(k1 <= 1 .or. k1 > nfft1)then
             ! k3 <= nf3
             do k3 = k3_start,nf3
                m3 = k3 - 1

                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                msq = mhat1*mhat1 + mhat2*mhat2 + mhat3*mhat3
                msq_inv = one/msq

                eterm  = exp(-fac*msq)*prefac12_piv_inv*prefac3(k3)*msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1  = eterm*struc2
                vterm  = two*(fac + msq_inv)   !two*(fac*msq  + one)*msq_inv
                tmp2   = tmp1*vterm
                energy = energy + tmp1

                virial(1) = virial(1) + tmp1*(vterm*mhat1*mhat1 - one)
                virial(2) = virial(2) + tmp2*mhat1*mhat2
                virial(3) = virial(3) + tmp2*mhat1*mhat3
                virial(4) = virial(4) + tmp1*(vterm*mhat2*mhat2 - one)
                virial(5) = virial(5) + tmp2*mhat2*mhat3
                virial(6) = virial(6) + tmp1*(vterm*mhat3*mhat3 - one)
             enddo
             ! k3 > nf3
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3

                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                msq = mhat1*mhat1 + mhat2*mhat2 + mhat3*mhat3
                msq_inv = one/msq

                eterm  = exp(-fac*msq)*prefac12_piv_inv*prefac3(k3)*msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)                
                index = index + 2

                tmp1   = eterm*struc2
                vterm  = two*(fac + msq_inv)       !two*(fac*msq + one)*msq_inv
                tmp2   = tmp1*vterm
                energy = energy + tmp1

                virial(1) = virial(1) + tmp1*(vterm*mhat1*mhat1 - one)
                virial(2) = virial(2) + tmp2*mhat1*mhat2
                virial(3) = virial(3) + tmp2*mhat1*mhat3
                virial(4) = virial(4) + tmp1*(vterm*mhat2*mhat2 - one)
                virial(5) = virial(5) + tmp2*mhat2*mhat3
                virial(6) = virial(6) + tmp1*(vterm*mhat3*mhat3 - one)
             enddo
          else
             prefac12s_piv_inv = piv_inv*prefac2(k2s)*prefac1(k1s)
             ! k3 <= nf3
             do k3 = k3_start,nf3
                m3 = k3 - 1
                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                k3s = mod(nfft3-k3+1,nfft3)+1
                m3s = k3s - 1
                if (k3s > nf3) m3s = k3s - 1 - nfft3

                mhat1s = recip(1,1)*m1s+recip(1,2)*m2s+recip(1,3)*m3s
                mhat2s = recip(2,1)*m1s+recip(2,2)*m2s+recip(2,3)*m3s
                mhat3s = recip(3,1)*m1s+recip(3,2)*m2s+recip(3,3)*m3s

                msqs = mhat1s*mhat1s + mhat2s*mhat2s + mhat3s*mhat3s
                msq  = mhat1*mhat1   + mhat2*mhat2   + mhat3*mhat3
                msq_inv = one/msq
                msqs_inv = one/msqs

                eterms = exp(-fac*msqs)*prefac12s_piv_inv*prefac3(k3s)*msqs_inv
                eterm  = exp(-fac*msq)*prefac12_piv_inv*prefac3(k3)*msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1s = eterms*struc2
                tmp1  = eterm*struc2
                vterms = two*(fac + msqs_inv) !two*(fac*msqs + one)*msqs_inv
                vterm  = two*(fac + msq_inv)  !two*(fac*msq  + one)*msq_inv
                tmp2s  = tmp1s*vterms
                tmp2   = tmp1*vterm
                energy = energy + tmp1s + tmp1

                virial(1) = virial(1) + tmp1s*(vterms*mhat1s*mhat1s - one) + &
                     tmp1*(vterm*mhat1*mhat1 - one)
                virial(2) = virial(2) + tmp2s*mhat1s*mhat2s + tmp2*mhat1*mhat2
                virial(3) = virial(3) + tmp2s*mhat1s*mhat3s + tmp2*mhat1*mhat3
                virial(4) = virial(4) + tmp1s*(vterms*mhat2s*mhat2s - one) + &
                     tmp1*(vterm*mhat2*mhat2 - one)
                virial(5) = virial(5) + tmp2s*mhat2s*mhat3s + tmp2*mhat2*mhat3
                virial(6) = virial(6) + tmp1s*(vterms*mhat3s*mhat3s - one) + &
                     tmp1*(vterm*mhat3*mhat3 - one)
             enddo
             ! k3 > nf3
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3
                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                k3s = mod(nfft3-k3+1,nfft3)+1
                m3s = k3s - 1
                if (k3s > nf3) m3s = k3s - 1 - nfft3

                mhat1s = recip(1,1)*m1s+recip(1,2)*m2s+recip(1,3)*m3s
                mhat2s = recip(2,1)*m1s+recip(2,2)*m2s+recip(2,3)*m3s
                mhat3s = recip(3,1)*m1s+recip(3,2)*m2s+recip(3,3)*m3s

                msqs = mhat1s*mhat1s + mhat2s*mhat2s + mhat3s*mhat3s
                msq  = mhat1*mhat1   + mhat2*mhat2   + mhat3*mhat3
                msq_inv = one/msq
                msqs_inv = one/msqs

                eterms = exp(-fac*msqs)*prefac12s_piv_inv*prefac3(k3s)*msqs_inv
                eterm  = exp(-fac*msq)*prefac12_piv_inv*prefac3(k3)*msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1s  = eterms*struc2
                tmp1   = eterm*struc2
                vterms = two*(fac + msqs_inv)  !two*(fac*msqs + one)*msqs_inv
                vterm  = two*(fac + msq_inv)   !two*(fac*msq + one)*msq_inv
                tmp2s  = tmp1s*vterms
                tmp2   = tmp1*vterm
                energy = energy + tmp1s + tmp1

                virial(1) = virial(1) + tmp1s*(vterms*mhat1s*mhat1s - one) + &
                     tmp1*(vterm*mhat1*mhat1 - one)
                virial(2) = virial(2) + tmp2s*mhat1s*mhat2s + tmp2*mhat1*mhat2
                virial(3) = virial(3) + tmp2s*mhat1s*mhat3s + tmp2*mhat1*mhat3
                virial(4) = virial(4) + tmp1s*(vterms*mhat2s*mhat2s - one) + &
                     tmp1*(vterm*mhat2*mhat2 - one)
                virial(5) = virial(5) + tmp2s*mhat2s*mhat3s + tmp2*mhat2*mhat3
                virial(6) = virial(6) + tmp1s*(vterms*mhat3s*mhat3s - one) + &
                     tmp1*(vterm*mhat3*mhat3 - one)
             enddo
          endif
       enddo
    enddo
!$omp end do
#endif /* (orthog)*/

    return
  end subroutine ADD_PREC(ADD_ORTHO(scalar_sumrc_kernel))

  ! Original scalar_sumrc -kernel. Used only as a reference. NOT threaded using OpenMP!
  subroutine ADD_PREC(ADD_ORTHO(scalar_sumrc_kernel_orig))(nfft1, nfft2, nfft3, nf1, nf2, nf3, &
       xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax, &
       zgridsize, xzgridsize, &
       fac, piv_inv, prefac1, prefac2, prefac3, recip, &
#if ORTHOG==1
       m1_tbl, m2_tbl, m3_tbl, &  
#endif
       q, energy, virial)
#if KEY_PARALLEL==1
    use parallel,only:mynod  
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: nfft1, nfft2, nfft3
    integer, intent(in) :: nf1, nf2, nf3
    integer, intent(in) :: xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax
    integer, intent(in) :: zgridsize, xzgridsize
#if DOUBLEP==1
    real(chm_real), intent(in) :: fac, piv_inv
    real(chm_real), intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
    real(chm_real), intent(in) :: recip(3,3)
#if ORTHOG==1
    real(chm_real), intent(in) :: m1_tbl(-(nfft1/2+1):(nfft1/2+1))
    real(chm_real), intent(in) :: m2_tbl(-(nfft2/2+1):(nfft2/2+1))
    real(chm_real), intent(in) :: m3_tbl(-(nfft3/2+1):(nfft3/2+1))
#endif 
    real(chm_real), intent(inout) :: q(0:*)
#else /**/
    real(chm_real4), intent(in) :: fac, piv_inv
    real(chm_real4), intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
    real(chm_real4), intent(in) :: recip(3,3)
#if ORTHOG==1
    real(chm_real4), intent(in) :: m1_tbl(-(nfft1/2+1):(nfft1/2+1))
    real(chm_real4), intent(in) :: m2_tbl(-(nfft2/2+1):(nfft2/2+1))
    real(chm_real4), intent(in) :: m3_tbl(-(nfft3/2+1):(nfft3/2+1))
#endif 
    real(chm_real4), intent(inout) :: q(0:*)
#endif 
    real(chm_real), intent(inout) :: energy, virial(6)
    ! Parameters
#if DOUBLEP==1
    real(chm_real), parameter :: one = 1.0_chm_real, two = 2.0_chm_real
#elif SINGLEP==1
    real(chm_real4), parameter :: one = 1.0_chm_real4, two = 2.0_chm_real4
#endif 
    ! Variables
#if DOUBLEP==1
    real(chm_real) eterm,vterm
    real(chm_real) mhat1,mhat2,mhat3,msq,struc2,msq_inv,msqs_inv
    real(chm_real) mhat1s,mhat2s,mhat3s,msqs
    real(chm_real) struc2s,eterms,vterms
    real(chm_real) tmp1,tmp2,tmp1s,tmp2s,m2_m1_tbl,m2_m1_tbls
#if ORTHOG==1
    real(chm_real) recip11,recip22,recip33  
#endif
    real(chm_real) mhat11, mhat22, mhat33, mhat12, mhat13, mhat23
    real(chm_real) mhat11s, mhat22s, mhat12s, mhat13s, mhat23s
    real(chm_real) prefac12_piv_inv, prefac12s_piv_inv
#elif SINGLEP==1
    real(chm_real4) eterm,vterm
    real(chm_real4) mhat1,mhat2,mhat3,msq,struc2,msq_inv,msqs_inv
    real(chm_real4) mhat1s,mhat2s,mhat3s,msqs
    real(chm_real4) struc2s,eterms,vterms
    real(chm_real4) tmp1,tmp2,tmp1s,tmp2s,m2_m1_tbl,m2_m1_tbls
#if ORTHOG==1
    real(chm_real4) recip11,recip22,recip33  
#endif
    real(chm_real4) mhat11, mhat22, mhat33, mhat12, mhat13, mhat23
    real(chm_real4) mhat11s, mhat22s, mhat12s, mhat13s, mhat23s
    real(chm_real4) prefac12_piv_inv, prefac12s_piv_inv
#endif 
    integer k3_start,k3_stop,k1s,k2s,k3s,m1s,m2s,m3s
    integer k2q,k1, k2, k3, m1, m2, m3, nff
    integer index, index2
#if KEY_PARALLEL==1
    logical master  
#endif

#if KEY_PARALLEL==1
    master = (mynod == 0)  
#endif

#if ORTHOG==1 /*orthog*/
    ! ######### orthog ########
    recip11 = recip(1,1)
    recip22 = recip(2,2)
    recip33 = recip(3,3)

!CC$omp do schedule(static) reduction(+:energy, virial)
    do k2q = 1, ygridmax-ygridmin+1
#if KEY_PARALLEL==1
       if(master)then  
#endif
          k2 = k2q + ygridmin
          k2s = mod(nfft2-k2+1,nfft2) + 1
#if KEY_PARALLEL==1
       else
          k2 = k2q + ygridmin
          k2s = mod(nfft2-k2+1,nfft2) + 1
       end if
#endif 
       m2 = k2 - 1
       if (k2 > nf2) m2 = m2 - nfft2
       mhat2 = recip22*m2
       m2s = k2s - 1
       if (k2s > nf2) m2s = m2s - nfft2
       mhat2s = recip22*m2s

       mhat22 = mhat2*mhat2
       mhat22s = mhat2s*mhat2s

       index2 = (k2q-1)*xzgridsize

       ! Loop over x
       do k1 = xgridmin+1, xgridmax+1
          k1s = nfft1 - k1 + 2
          m1 = k1 - 1
          if ( k1 > nf1 ) m1 = m1 - nfft1
          mhat1 = recip11*m1
          m1s = k1s - 1
          if ( k1s > nf1 ) m1s = m1s - nfft1
          mhat1s = recip11*m1s

          mhat11 = mhat1*mhat1
          mhat11s = mhat1s*mhat1s
          mhat12 = mhat1*mhat2
          mhat12s = mhat1s*mhat2s

          m2_m1_tbl  = piv_inv*m2_tbl(m2)*m1_tbl(m1)*prefac2(k2)*prefac1(k1)

          k3_start = zgridmin + 1

          ! This is used to skip the k = 0 term
#if KEY_PARALLEL==1
          if(master .and. (k1+k2 == 2) ) k3_start = 2  
#endif
#if KEY_PARALLEL==0
          if(k1+k2 == 2) k3_start = 2  
#endif

          index = 2*(k3_start-1 + (k1-xgridmin-1)*zgridsize + index2)

          ! Loop over z
          if (k1 <= 1 .or. k1 > nfft1) then
             ! k3 <= nf3
             do k3 = k3_start,nf3
                m3 = k3 - 1
                mhat3 = recip33*m3
                mhat33 = mhat3*mhat3

                msq = mhat11 + mhat22 + mhat33
                msq_inv = one/msq

                eterm  = m3_tbl(m3)*prefac3(k3)*m2_m1_tbl *msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1  = eterm*struc2
                vterm  = two*(fac*msq  + one)*msq_inv
                tmp2   = tmp1*vterm
                energy = energy + tmp1

                mhat13 = mhat1*mhat3
                mhat23 = mhat2*mhat3

                virial(1) = virial(1) + tmp1*(vterm*mhat11 - one)
                virial(2) = virial(2) + tmp2*mhat12
                virial(3) = virial(3) + tmp2*mhat13
                virial(4) = virial(4) + tmp1*(vterm*mhat22 - one)
                virial(5) = virial(5) + tmp2*mhat23
                virial(6) = virial(6) + tmp1*(vterm*mhat33 - one)
             enddo
             ! k3 > nf3
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3
                mhat3 = recip33*m3
                mhat33 = mhat3*mhat3

                msq  = mhat11   + mhat22   + mhat33
                msq_inv = one/msq

                eterm  = m3_tbl(m3) * prefac3(k3) * m2_m1_tbl  * msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)                
                index = index + 2

                tmp1   = eterm*struc2
                vterm  = two*(fac*msq + one)*msq_inv
                tmp2   = tmp1*vterm
                energy = energy + tmp1

                mhat13 = mhat1*mhat3
                mhat23 = mhat2*mhat3

                virial(1) = virial(1) + tmp1*(vterm*mhat11 - one)
                virial(2) = virial(2) + tmp2*mhat12
                virial(3) = virial(3) + tmp2*mhat13
                virial(4) = virial(4) + tmp1*(vterm*mhat22 - one)
                virial(5) = virial(5) + tmp2*mhat23
                virial(6) = virial(6) + tmp1*(vterm*mhat33 - one)
             enddo
          else
             m2_m1_tbls = piv_inv*m2_tbl(m2s)*m1_tbl(m1s)*prefac2(k2s)*prefac1(k1s)
             ! k3 <= nf3
             do k3 = k3_start,nf3
                m3 = k3 - 1
                mhat3 = recip33*m3
                mhat33 = mhat3*mhat3

                msqs = mhat11s + mhat22s + mhat33
                msq = mhat11 + mhat22 + mhat33
                msq_inv = one/msq
                msqs_inv = one/msqs

                eterms = m3_tbl(m3)*prefac3(k3)
                eterm  = eterms
                eterms = eterms*m2_m1_tbls*msqs_inv
                eterm  = eterm *m2_m1_tbl *msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1s = eterms*struc2
                tmp1  = eterm*struc2
                vterms = two*(fac*msqs + one)*msqs_inv
                vterm  = two*(fac*msq  + one)*msq_inv
                tmp2s  = tmp1s*vterms
                tmp2   = tmp1*vterm
                energy = energy + tmp1s + tmp1

                mhat13 = mhat1*mhat3
                mhat13s = mhat1s*(-mhat3)
                mhat23 = mhat2*mhat3
                mhat23s = mhat2s*(-mhat3)

                virial(1) = virial(1) + tmp1s*(vterms*mhat11s - one) + tmp1*(vterm*mhat11 - one)
                virial(2) = virial(2) + tmp2s*mhat12s + tmp2*mhat12
                virial(3) = virial(3) + tmp2s*mhat13s + tmp2*mhat13
                virial(4) = virial(4) + tmp1s*(vterms*mhat22s - one) + tmp1*(vterm*mhat22 - one)
                virial(5) = virial(5) + tmp2s*mhat23s + tmp2*mhat23
                virial(6) = virial(6) + tmp1s*(vterms*mhat33 - one) + tmp1*(vterm*mhat33 - one)
             enddo
             ! k3 > nf3
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3
                mhat3 = recip33*m3
                mhat33 = mhat3*mhat3

                msqs = mhat11s + mhat22s + mhat33
                msq  = mhat11   + mhat22   + mhat33
                msqs_inv = one/msqs
                msq_inv = one/msq

                eterms = m3_tbl(m3)*prefac3(k3)
                eterm = eterms
                eterms = eterms * m2_m1_tbls * msqs_inv
                eterm  = eterm  * m2_m1_tbl  * msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1s  = eterms*struc2
                tmp1   = eterm*struc2
                vterms = two*(fac*msqs + one)*msqs_inv
                vterm  = two*(fac*msq + one)*msq_inv
                tmp2s  = tmp1s*vterms
                tmp2   = tmp1*vterm
                energy = energy + tmp1s + tmp1

                mhat13 = mhat1*mhat3
                mhat13s = mhat1s*(-mhat3)
                mhat23 = mhat2*mhat3
                mhat23s = mhat2s*(-mhat3)

                virial(1) = virial(1) + tmp1s*(vterms*mhat11s - one) + tmp1*(vterm*mhat11 - one)
                virial(2) = virial(2) + tmp2s*mhat12s + tmp2*mhat12
                virial(3) = virial(3) + tmp2s*mhat13s + tmp2*mhat13
                virial(4) = virial(4) + tmp1s*(vterms*mhat22s - one) + tmp1*(vterm*mhat22 - one)
                virial(5) = virial(5) + tmp2s*mhat23s + tmp2*mhat23
                virial(6) = virial(6) + tmp1s*(vterms*mhat33 - one) + tmp1*(vterm*mhat33 - one)
             enddo
          endif
       enddo
    enddo
!CC$omp end do

#else /* (orthog)*/
    ! ######### non-orthog ########

!CC$omp do schedule(static) reduction(+:energy, virial)
    do k2q = 1, ygridmax-ygridmin+1
#if KEY_PARALLEL==1
       if(master)then  
#endif
          k2 = k2q + ygridmin
          k2s = mod(nfft2-k2+1,nfft2) + 1
#if KEY_PARALLEL==1
       else
          k2 = k2q + ygridmin
          k2s = mod(nfft2-k2+1,nfft2) + 1
       end if
#endif 
       m2 = k2 - 1
       if (k2 > nf2) m2 = m2 - nfft2
       m2s = k2s - 1
       if (k2s > nf2) m2s = m2s - nfft2

       index2 = (k2q-1)*xzgridsize

       ! Loop over x
       do k1 = xgridmin+1, xgridmax+1
          k1s = nfft1 - k1 + 2
          m1 = k1 - 1
          if ( k1 > nf1 ) m1 = m1 - nfft1
          m1s = k1s - 1
          if ( k1s > nf1 ) m1s = m1s - nfft1

          prefac12_piv_inv = piv_inv*prefac2(k2)*prefac1(k1)

          k3_start = zgridmin+1

          ! This is used to skip the k = 0 term
#if KEY_PARALLEL==1
          if(master .and. (k1+k2 == 2) ) k3_start = 2  
#endif
#if KEY_PARALLEL==0
          if(k1+k2 == 2) k3_start = 2  
#endif

          index = 2*(k3_start-1 + (k1-xgridmin-1)*zgridsize + index2)

          ! Loop over z
          if(k1 <= 1 .or. k1 > nfft1)then
             ! k3 <= nf3
             do k3 = k3_start,nf3
                m3 = k3 - 1

                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                msq = mhat1*mhat1 + mhat2*mhat2 + mhat3*mhat3
                msq_inv = one/msq

                eterm  = exp(-fac*msq)*prefac12_piv_inv*prefac3(k3)*msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1  = eterm*struc2
                vterm  = two*(fac*msq  + one)*msq_inv
                tmp2   = tmp1*vterm
                energy = energy + tmp1

                virial(1) = virial(1) + tmp1*(vterm*mhat1*mhat1 - one)
                virial(2) = virial(2) + tmp2*mhat1*mhat2
                virial(3) = virial(3) + tmp2*mhat1*mhat3
                virial(4) = virial(4) + tmp1*(vterm*mhat2*mhat2 - one)
                virial(5) = virial(5) + tmp2*mhat2*mhat3
                virial(6) = virial(6) + tmp1*(vterm*mhat3*mhat3 - one)
             enddo
             ! k3 > nf3
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3

                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                msq = mhat1*mhat1 + mhat2*mhat2 + mhat3*mhat3
                msq_inv = one/msq

                eterm  = exp(-fac*msq)*prefac12_piv_inv*prefac3(k3)*msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)                
                index = index + 2

                tmp1   = eterm*struc2
                vterm  = two*(fac*msq + one)*msq_inv
                tmp2   = tmp1*vterm
                energy = energy + tmp1

                virial(1) = virial(1) + tmp1*(vterm*mhat1*mhat1 - one)
                virial(2) = virial(2) + tmp2*mhat1*mhat2
                virial(3) = virial(3) + tmp2*mhat1*mhat3
                virial(4) = virial(4) + tmp1*(vterm*mhat2*mhat2 - one)
                virial(5) = virial(5) + tmp2*mhat2*mhat3
                virial(6) = virial(6) + tmp1*(vterm*mhat3*mhat3 - one)
             enddo
          else
             prefac12s_piv_inv = piv_inv*prefac2(k2s)*prefac1(k1s)
             ! k3 <= nf3
             do k3 = k3_start,nf3
                m3 = k3 - 1
                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                k3s = mod(nfft3-k3+1,nfft3)+1
                m3s = k3s - 1
                if (k3s > nf3) m3s = k3s - 1 - nfft3

                mhat1s = recip(1,1)*m1s+recip(1,2)*m2s+recip(1,3)*m3s
                mhat2s = recip(2,1)*m1s+recip(2,2)*m2s+recip(2,3)*m3s
                mhat3s = recip(3,1)*m1s+recip(3,2)*m2s+recip(3,3)*m3s

                msqs = mhat1s*mhat1s + mhat2s*mhat2s + mhat3s*mhat3s
                msq  = mhat1*mhat1   + mhat2*mhat2   + mhat3*mhat3
                msq_inv = one/msq
                msqs_inv = one/msqs

                eterms = exp(-fac*msqs)*prefac12s_piv_inv*prefac3(k3s)*msqs_inv
                eterm  = exp(-fac*msq)*prefac12_piv_inv*prefac3(k3)*msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1s = eterms*struc2
                tmp1  = eterm*struc2
                vterms = two*(fac*msqs + one)*msqs_inv
                vterm  = two*(fac*msq  + one)*msq_inv
                tmp2s  = tmp1s*vterms
                tmp2   = tmp1*vterm
                energy = energy + tmp1s + tmp1

                virial(1) = virial(1) + tmp1s*(vterms*mhat1s*mhat1s - one) + &
                     tmp1*(vterm*mhat1*mhat1 - one)
                virial(2) = virial(2) + tmp2s*mhat1s*mhat2s + tmp2*mhat1*mhat2
                virial(3) = virial(3) + tmp2s*mhat1s*mhat3s + tmp2*mhat1*mhat3
                virial(4) = virial(4) + tmp1s*(vterms*mhat2s*mhat2s - one) + &
                     tmp1*(vterm*mhat2*mhat2 - one)
                virial(5) = virial(5) + tmp2s*mhat2s*mhat3s + tmp2*mhat2*mhat3
                virial(6) = virial(6) + tmp1s*(vterms*mhat3s*mhat3s - one) + &
                     tmp1*(vterm*mhat3*mhat3 - one)
             enddo
             ! k3 > nf3
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3
                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                k3s = mod(nfft3-k3+1,nfft3)+1
                m3s = k3s - 1
                if (k3s > nf3) m3s = k3s - 1 - nfft3

                mhat1s = recip(1,1)*m1s+recip(1,2)*m2s+recip(1,3)*m3s
                mhat2s = recip(2,1)*m1s+recip(2,2)*m2s+recip(2,3)*m3s
                mhat3s = recip(3,1)*m1s+recip(3,2)*m2s+recip(3,3)*m3s

                msqs = mhat1s*mhat1s + mhat2s*mhat2s + mhat3s*mhat3s
                msq  = mhat1*mhat1   + mhat2*mhat2   + mhat3*mhat3
                msq_inv = one/msq
                msqs_inv = one/msqs

                eterms = exp(-fac*msqs)*prefac12s_piv_inv*prefac3(k3s)*msqs_inv
                eterm  = exp(-fac*msq)*prefac12_piv_inv*prefac3(k3)*msq_inv
                struc2 = q(index)**2 + q(index+1)**2
                q(index:index+1) = eterm*q(index:index+1)
                index = index + 2

                tmp1s  = eterms*struc2
                tmp1   = eterm*struc2
                vterms = two*(fac*msqs + one)*msqs_inv
                vterm  = two*(fac*msq + one)*msq_inv
                tmp2s  = tmp1s*vterms
                tmp2   = tmp1*vterm
                energy = energy + tmp1s + tmp1

                virial(1) = virial(1) + tmp1s*(vterms*mhat1s*mhat1s - one) + &
                     tmp1*(vterm*mhat1*mhat1 - one)
                virial(2) = virial(2) + tmp2s*mhat1s*mhat2s + tmp2*mhat1*mhat2
                virial(3) = virial(3) + tmp2s*mhat1s*mhat3s + tmp2*mhat1*mhat3
                virial(4) = virial(4) + tmp1s*(vterms*mhat2s*mhat2s - one) + &
                     tmp1*(vterm*mhat2*mhat2 - one)
                virial(5) = virial(5) + tmp2s*mhat2s*mhat3s + tmp2*mhat2*mhat3
                virial(6) = virial(6) + tmp1s*(vterms*mhat3s*mhat3s - one) + &
                     tmp1*(vterm*mhat3*mhat3 - one)
             enddo
          endif
       enddo
    enddo
!CC$omp end do
#endif /* (orthog)*/

    return
  end subroutine ADD_PREC(ADD_ORTHO(scalar_sumrc_kernel_orig))
! ##ENDEX (exp_orthog)
