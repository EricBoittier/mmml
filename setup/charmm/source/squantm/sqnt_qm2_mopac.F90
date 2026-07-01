#if KEY_SQUANTM==1 /*mainsquatn*/
!************************************************************************
!                              AMBER                                   **
!                                                                      **
!                        Copyright (c) 2003                            **
!                Regents of the University of California               **
!                       All Rights Reserved.                           **
!                                                                      **
!  This software provided pursuant to a license agreement containing   **
!  restrictions on its disclosure, duplication, and use. This software **
!  contains confidential and proprietary information, and may not be   **
!  extracted or distributed, in whole or in part, for any purpose      **
!  whatsoever, without the express written permission of the authors.  **
!  This notice, and the associated author list, must be attached to    **
!  all copies, or extracts, of this software. Any additional           **
!  restrictions set forth in the license agreement also apply to this  **
!  software.                                                           **
!************************************************************************

      SUBROUTINE qm2_hcore_qmqm(COORD,H,W,ENUCLR)
!
! Current code, optimisation and inlining 
!  by: Ross Walker (TSRI, 2005)
!
! This routine is responsible for generating the one-electron 
! matrix and the two electron integrals via calls to qm2_h1elec
! and qm2_rotate_qmqm.
! qm2_h1elec has been inlined in this code for speed.
!
! Current Version: Ross Walker (TSRI, 2005)
!
! IN -
!     COORD  : QM coordinates
!
! OUT-
!     H      : One electron matix
!     W      : Two electron integrals
!     ENUCLR : Nuclear energy 
!

  use qmmm_module, only : qmmm_nml, qmmm_struct, qm2_struct,  &
                              qm2_params, qm2_rij_eqns
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!Passed in
      real(chm_real), intent(in)  :: COORD(3,qmmm_struct%nquant_nlink)
      real(chm_real), intent(out) :: W(qm2_struct%n2el)
      real(chm_real), intent(out) :: ENUCLR
      real(chm_real), intent(out) :: H(qm2_struct%matsize)

!Local
      real(chm_real) :: E1B(10),E2A(10),SHMAT(4,4)
      integer  :: i, first_si, last_pi, first_pi, ni, i1, i2, j
      integer  :: kr, j1, first_sj, last_pj, ii, j2, jj, k, l
      integer  :: loop_count, iminus
      integer  :: n_atomic_orbi, n_atomic_orbj, qmitype, qmjtype
      real(chm_real) :: enuc, elec_ke_s, elec_ke_p
      real(chm_real) :: vec_qm_qm1, vec_qm_qm2, vec_qm_qm3, vec_qm_qm(3)
      real(chm_real) :: half_num, R2, BIJSP, BIJPS, BIJPP, ADBR2, SH, TOMB
      integer  :: mmynod,mnumnod,KI
      logical  :: SI,SJ


! FILL THE DIAGONALS, AND OFF-DIAGONALS ON ATOM1 
! as we don't do it in the loop below. 
! Zero the one electron matrix before we start.
 
! Initizalize Hcore matrix of qm2_struct%matsize
      H                        = zero
      W                        = zero  ! 2e-integral 
      qm2_struct%qm_qm_e_repul = zero

#if KEY_PARALLEL==1
      if(mynod.eq.0) then
#endif 
      first_si = qm2_params%orb_loc(1,1)
      first_pi = first_si+1
      last_pi  = qm2_params%orb_loc(2,1)

      H(1)     = qm2_params%orb_elec_ke(1,1)
      elec_ke_s= qm2_params%orb_elec_ke(1,1)
      elec_ke_p= qm2_params%orb_elec_ke(2,1)

      do I1=first_pi,last_pi
        I2     = qm2_params%pascal_tri2(I1)
        H(I2)  = elec_ke_p
      end do
#if KEY_PARALLEL==1
      end if

      mmynod  = mynod
      mnumnod = numnod
#endif 

! Now do all the pairs.
      KR         = 1
      loop_count = 0
      do I=2,qmmm_struct%nquant_nlink

#if KEY_PARALLEL==1
         if(mmynod .eq. mod(i-2,mnumnod)) then
#endif 

         first_si      = qm2_params%orb_loc(1,I)
         first_pi      = first_si+1
         last_pi       = qm2_params%orb_loc(2,I)

         n_atomic_orbi = qm2_params%natomic_orbs(i)
         qmitype       = qmmm_struct%qm_atom_type(i) 
         NI            = qmmm_struct%iqm_atomic_numbers(I)
         elec_ke_s     = qm2_params%orb_elec_ke(1,i)
         elec_ke_p     = qm2_params%orb_elec_ke(2,i)

! FIRST WE FILL THE DIAGONALS, 
! AND OFF-DIAGONALS ON THE SAME ATOM
         I2            = qm2_params%pascal_tri2(first_si)
         H(I2)         = elec_ke_s
         do I1=first_pi,last_pi
           I2          = qm2_params%pascal_tri2(I1)
           H(I2)       = elec_ke_p
         end do

! FILL THE ATOM-OTHER ATOM ONE-ELECTRON MATRIX
! <PSI(LAMBDA)|PSI(SIGMA)> 
         iminus = i-1 
         do J=1,iminus
            loop_count = loop_count+1
            first_sj   = qm2_params%orb_loc(1,J)
            last_pj    = qm2_params%orb_loc(2,J) 
            qmjtype    = qmmm_struct%qm_atom_type(j)

!Calculate Overlap Integrals using a Gaussian Expansion
!STO-6G BY R.F. STEWART, J. CHEM. PHYS., 52 431-438, 1970
!Fill SHMAT with a 4x4 array of overlaps, 
!     in order S,PX,PY,PZ
!     R2   =  INTERATOMIC DISTANCE^2 IN BOHRS2
            if (qmmm_nml%qmqmrij_incore) then
              R2         = qm2_rij_eqns%qmqmrijdata(QMQMRIJBOHRS2,  &
                           loop_count)
            else
              vec_qm_qm1 = (coord(1,i)-coord(1,j))
              vec_qm_qm2 = (coord(2,i)-coord(2,j))
              vec_qm_qm3 = (coord(3,i)-coord(3,j))
              R2         = ( vec_qm_qm1*vec_qm_qm1 +   &
                             vec_qm_qm2*vec_qm_qm2 +   &
                             vec_qm_qm3*vec_qm_qm3 )*A2_TO_BOHRS2
            end if
            if (R2 .LT. OVERLAP_CUTOFF) then
              n_atomic_orbj = qm2_params%natomic_orbs(j)
              SHMAT         = zero

              CALL qm2_h1elec( loop_count,R2,COORD(1,I),COORD(1,J),   &
                               n_atomic_orbi,n_atomic_orbj,SHMAT,   &
              qm2_params%atom_orb_zz_sxs_over_sas(1,1,qmitype,qmjtype),  &
              qm2_params%atom_orb_ss_eqn(1,1,qmitype,qmjtype),   &
              qm2_params%atom_orb_zz_sxp_over_sap(1,1,qmitype,qmjtype),  &
              qm2_params%atom_orb_zz_sxp_over_sap(1,1,qmjtype,qmitype),  &
              qm2_params%atom_orb_sp_ovlp(1,1,qmitype,qmjtype),   &
              qm2_params%atom_orb_sp_ovlp(1,1,qmjtype,qmitype),   &
              qm2_params%atom_orb_zz_pxp_over_pap(1,1,qmitype,qmjtype),  &
              qm2_params%atom_orb_pp_ovlp_ieqj1(1,1,qmitype,qmjtype),   &
              qm2_params%atom_orb_pp_ovlp_ieqj2(1,1,qmitype,qmjtype),   &
              qm2_params%atom_orb_pp_ovlp_inj(1,1,qmitype,qmjtype),   &
              qm2_params%betasas(qmitype,qmjtype),   &
              qm2_params%betasap(qmitype,qmjtype),   &
              qm2_params%betasap(qmjtype,qmitype),   &
              qm2_params%betapap(qmitype,qmjtype) )

              I2=0
              do I1=first_si,last_pi
                 II=qm2_params%pascal_tri1(i1)+first_sj-1
                 I2=I2+1
                 J2=0
                 JJ=MIN(I1,last_pj)
                 do J1=first_sj,JJ
                    II   =II+1
                    J2   =J2+1
                    H(II)=H(II)+SHMAT(I2,J2) 
                 end do
              end do
            end if !(R2 < OVERLAP_CUTOFF)

! CALCULATE THE TWO-ELECTRON INTEGRALS, W; 
!           THE ELECTRON NUCLEAR TERMS, E1B AND E2A;
!           THE NUCLEAR-NUCLEAR TERM ENUC.
            CALL qm2_rotate_qmqm(loop_count,i,j,NI,   &
                                 qmmm_struct%iqm_atomic_numbers(J),   &
                                 COORD(1,I),COORD(1,J),W(KR),   &
                                 KR,E1B,E2A,ENUC,qmitype,qmjtype)

            ENUCLR = ENUCLR + ENUC

! ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM I.
            if(I .EQ. J) then
               half_num = half
            else
               half_num = one
            end if

            I2=0
            do I1=first_si,last_pi
               II=qm2_params%pascal_tri1(i1)+first_si-1
               do J1=first_si,I1
                  II   =II+1
                  I2   =I2+1
                  H(II)=H(II)+E1B(I2)*half_num
               end do
            end do

! ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM J.
            I2=0
            do I1=first_sj,last_pj
               II=qm2_params%pascal_tri1(i1)+first_sj-1
               do J1=first_sj,I1
                  II   =II+1
                  I2   =I2+1
                  H(II)=H(II)+E2A(I2)*half_num
               end do
            end do
         end do                     ! J=1,iminus
#if KEY_PARALLEL==1
         else
            iminus = i-1
            loop_count  = loop_count + iminus

            SI    = (qm2_params%natomic_orbs(i) .GT. 1)
            do J=1,iminus 
               SJ = (qm2_params%natomic_orbs(j) .GT. 1)
               KI = 1
               if(SI.or.SJ)  KI=10
               if(SI.and.SJ) KI=100

!              IF(SJ) KI=10
!              IF(SI) THEN
!                 IF(SJ) THEN
!                    KI=100
!                 ELSE
!                    KI=10
!                 ENDIF
!              ENDIF

               KR = KR+ KI
            end do
         end if
#endif 
      end do                        ! I=1,qmmm_struct%nquant_nlink

! This will be done after call "qm2_hcore_qmmm".
#if KEY_PARALLEL==1
      if(numnod.gt.1) then
!         call GCOMB(H,qm2_struct%matsize)
!         call GCOMB(W,qm2_struct%n2el)
!         call GCOMB(ENUCLR,1)
          J = qmmm_struct%nquant_nlink*(qmmm_struct%nquant_nlink-1)/2
          call GCOMBR(qm2_struct%qm_qm_e_repul,22*J)   ! used also in
                                                       ! gradient (qm2_get_qm_forces)
      end if
#endif 

      return
      END SUBROUTINE qm2_hcore_qmqm


      SUBROUTINE  qm2_hcore_qmmm(qm_COORD,H,ENUCLR,crdsmm)
!
! This routine is repsonsible for calculating the 
! QM-MM interactions between QM-MM pairs that are
! in the QMMM pair list. This routine calls qm2_rotate_qmmm.
! It is responsible for updating the one-electron matix H 
! with the QM electron MM core interaction. It also
! calculates the QM-MM core-core interactions that
! are returned as an addition to ENUCLR.
!
! QM core charge interacts with MM charge, which is
! stored as the 4th index of the qm_COORD array.
! Hence, the qm_coord array: x1, y1, z1, q1, x2, y2, z2, q2... 
!
! Current code: Ross Walker (TSRI, 2005)

  use qmmm_module, only : qmmm_struct, qm2_struct, qm2_params,  &
                              qmmm_switch
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!Passed in
      real(chm_real), intent(in)    :: qm_coord(3,*), crdsmm(*)
      real(chm_real), intent(inout) :: H(qm2_struct%matsize)
      real(chm_real), intent(out)   :: enuclr

!Local variables
! Keeps track of the nquant * ni_mm loop iterations
      integer  :: loop_count, i, j, ia, ib, i2, i1, j1, ii
      integer  :: ncount,inner_loop_count
      logical  :: heavy_atom
      real(chm_real) :: enucij
      real(chm_real) :: E1B(10), E1B_light
      integer  :: mstart,mstop,mstart2,mstop2
      integer  :: ISTRT_CHECK                 ! for external function

! do some initialization
      mstart = 1
      mstop  = qmmm_struct%qm_mm_pairs
      mstart2= 1
      mstop2 = 4*qmmm_struct%qm_mm_pairs
#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         mstart = ISTRT_CHECK(mstop,qmmm_struct%qm_mm_pairs)
         if(mstart.gt.0) mstart2= 4*(mstart-1)+1
         mstop2 = 4*mstop

         if(associated(qmmm_switch%dxqmmm_elec)) qmmm_switch%dxqmmm_elec = zero
      end if
#endif 

! Loop over REAL QM atoms and determine interactions with MM atoms.
      loop_count=0
      do i=1,qmmm_struct%nquant_nlink
!
! Definitions:
!
! orb_loc(1,i)          : first atomic orbital on atom i
! orb_loc(2,i)          : last atomic orbital on atom i
! iqm_atomic_numbers(i) : atomic number for atom i

         heavy_atom =(qm2_params%natomic_orbs(i) .GT. 1)
         ia         = qm2_params%orb_loc(1,i)

! Loop over MM atoms that interact with QM atom i, update the
! one-electron matrix H, and accumulate core-core repulsions
! ENUCLR.
!
! Loop in steps of since crdsmm array is x,y,z,chg,x,y,z,chg...
! Duplicated code here but it avoids the if(heavy_atom) inside 
! the loop
         if (heavy_atom) then
            ib = qm2_params%orb_loc(2,i)     
! Will be same as ia for light atom

#if KEY_PARALLEL==1
! befor loop (for loop_count)                         
#endif
#if KEY_PARALLEL==1
            loop_count = loop_count + (mstart-1)      
#endif

            ncount           = 0
            inner_loop_count = 0
#if KEY_PARALLEL==1
            if(qmmm_switch%qswitch) then
               do j=1,mstart-1
                  inner_loop_count=inner_loop_count+1
                  if( qmmm_switch%scmask( &
                      qmmm_struct%qm_mm_pair_list(inner_loop_count) ) &
                     ) ncount = ncount + 1
               end do
            end if
#endif 
            do j=mstart2,mstop2,4           ! 1,4*qmmm_struct%qm_mm_pairs,4

! Get the QM-MM interactions:
!     QM electrons - MM core ---> E1B
!     QM core - MM core ----> enucij
               loop_count = loop_count+1
               call qm2_rotate_qmmm_heavy(loop_count,i,qm_COORD(1,i),   &
                                          crdsmm(j),E1B,enucij)

! Add E1B to the corresponding diagonal elements of 
! the one-electron matrix H.
               i2 = 0
               do i1=ia,ib
                  ii=qm2_params%pascal_tri1(i1) + ia - 1
                  do j1=ia,i1
                     ii    = ii + 1
                     i2    = i2 + 1
                     H(ii) = H(ii) + E1B(i2)
                  end do 
               end do 

! Add on QM core - MM core interactions.
               ENUCLR = ENUCLR + enucij

! Electrostatic switching function
               inner_loop_count=inner_loop_count+1
!  if(qmmm_switch%qswitch.and.  
! *   qmmm_switch%scmask(inner_loop_count)) then
               if(qmmm_switch%qswitch) then
                  if( qmmm_switch%scmask(   &
                      qmmm_struct%qm_mm_pair_list(inner_loop_count) )   &
                     ) then
                  ncount = ncount + 1
                  qmmm_switch%dxqmmm_elec(1:10,ncount,i) = E1B(1:10)
                  qmmm_switch%dxqmmm_elec(11,ncount,i)   = enucij
                  end if
               end if
            end do

#if KEY_PARALLEL==1
! after loop (for loop_count)                                         
#endif
#if KEY_PARALLEL==1
            loop_count = loop_count + (qmmm_struct%qm_mm_pairs-mstop) 
#endif
         else                                       ! if (heavy_atom)
! Light atom (Hydrogen) - same notes as above for heavy atom

#if KEY_PARALLEL==1
! befor loop (for loop_count)                     
#endif
#if KEY_PARALLEL==1
            loop_count = loop_count + (mstart-1)  
#endif

            ncount           = 0
            inner_loop_count = 0
#if KEY_PARALLEL==1
            if(qmmm_switch%qswitch) then
               do j=1,mstart-1
                  inner_loop_count=inner_loop_count+1
                  if( qmmm_switch%scmask( &
                      qmmm_struct%qm_mm_pair_list(inner_loop_count) ) &
                     ) ncount = ncount + 1
               end do
            end if
#endif 
            do j=mstart2,mstop2,4           ! 1,4*qmmm_struct%qm_mm_pairs,4

               loop_count = loop_count+1
               call qm2_rotate_qmmm_light(loop_count,i,qm_COORD(1,i),   &
                                          crdsmm(j),E1B_light,enucij)
               ii     = qm2_params%pascal_tri1(ia) + ia 
               H(ii)  = H(ii) + E1B_light

               ENUCLR = ENUCLR + enucij

! Electrostatic switching function
               inner_loop_count = inner_loop_count+1
!  if(qmmm_switch%qswitch.and.  
! *   qmmm_switch%scmask(inner_loop_count)) then
               if(qmmm_switch%qswitch) then
                  if( qmmm_switch%scmask(   &
                      qmmm_struct%qm_mm_pair_list(inner_loop_count) )   &
                     ) then
                  ncount = ncount + 1
                  qmmm_switch%dxqmmm_elec(1,ncount,i)    = E1B_light
                  qmmm_switch%dxqmmm_elec(2:10,ncount,i) = zero
                  qmmm_switch%dxqmmm_elec(11,ncount,i)   = enucij 
                  end if
               end if
            end do

#if KEY_PARALLEL==1
! after loop (for loop_count)                                         
#endif
#if KEY_PARALLEL==1
            loop_count = loop_count + (qmmm_struct%qm_mm_pairs-mstop) 
#endif
         end if                                     ! if (heavy_atom)
      end do

#if KEY_PARALLEL==1
      if(numnod .gt. 1 .and. qmmm_switch%qswitch) then
          call GCOMBR(qmmm_switch%dxqmmm_elec, &
                      11 * qmmm_switch%nswitch_atom_old &
                         * qmmm_struct%nquant_nlink)
      end if
#endif 

      return
      END SUBROUTINE qm2_hcore_qmmm


      SUBROUTINE qm2_rotate_qmmm_light(loop_count,IQM,xyz_qm,xyz_mm, &
                                       E1B,ENUC)
!
! For light atoms
! See heavy routine below for notes.
!
! Written by Ross Walker (TSRI, 2005)
!

  use qmmm_module, only : qmmm_nml,qm2_struct,qm2_params,  &
                          qm2_rij_eqns,qmmm_struct
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations
      implicit none
!Passed in
      integer, intent(in)  :: loop_count, iqm
      real(chm_real),intent(in)  :: xyz_qm(3), xyz_mm(4)
      real(chm_real),intent(out) :: e1b, enuc

!Local variables
      real(chm_real) :: RI
      real(chm_real) :: oneBDDi1
      real(chm_real) :: X1, X2, X3
      real(chm_real) :: chrgmm, RR2, RIJ1, RIJ, RIJ2
      real(chm_real) :: scale
      integer  :: i,iqmtype
      real(chm_real) :: temp_real,temp_real2,c1,anam1

      if (qmmm_nml%qmmmrij_incore) then
         RIJ1 = qm2_rij_eqns%qmmmrijdata(QMMMONERIJ,loop_count)
         RIJ  = qm2_rij_eqns%qmmmrijdata(QMMMRIJ,loop_count)
         scale= qm2_rij_eqns%qmmmrijdata(QMMMEXP1,loop_count)+   &
                qm2_rij_eqns%qmmmrijdata(QMMMEXP2,loop_count)
         RI   = AU_TO_EV*qm2_rij_eqns%qmmmrijdata(QMMMSQRTAEE,   &
                                                  loop_count)
      else
         X1   = xyz_qm(1) - xyz_mm(1)
         X2   = xyz_qm(2) - xyz_mm(2)
         X3   = xyz_qm(3) - xyz_mm(3)
         RIJ2 = X1*X1+X2*X2+X3*X3
         RR2  = RIJ2*A2_TO_BOHRS2
         RIJ1 = one/sqrt(RIJ2)
         RIJ  = one/RIJ1 
         scale= EXP(-qm2_params%cc_exp_params(iqm)*RIJ) +   &
                EXP(-ALPH_MM*RIJ)

         oneBDDi1 = qm2_params%multip_2c_elec_params(6,iqm)
         RI       = AU_TO_EV/sqrt(RR2+oneBDDi1)
      end if

      if (qmmm_nml%qmmm_erep_incore) then
        qm2_struct%qm_mm_e_repul(1,loop_count) = RI
      end if

      chrgmm = xyz_mm(4)
      E1B    =-chrgmm*RI
      ENUC   =-qm2_params%core_chg(IQM)*E1B
      ENUC   = ENUC+ABS(scale*ENUC)

! for AM1/PM3/PM3CARB1: Gaussian core-core terms.
!     PDDGPM3 should be done differently.
      If(qmmm_nml%QMMM_Gauss) then
        iqmtype= qmmm_struct%qm_atom_type(iqm)
        C1     = qm2_params%core_chg(IQM)*chrgmm
        anam1  = zero
        do i=1,qm2_params%num_fn(iqmtype)
            temp_real = RIJ-qm2_params%FN3(i,iqmtype)
            temp_real2= qm2_params%FN2(i,iqmtype)*temp_real*temp_real

            ! Skip doing the exponential if it is essentially zer
            if (temp_real2 .LT. EXPONENTIAL_CUTOFF) then
              anam1 = anam1+qm2_params%FN1(i,iqmtype)*EXP(-temp_real2)
            end if
        end do
        anam1 = anam1*c1*RIJ1
        ENUC = ENUC + anam1
      End if

      return
      END SUBROUTINE qm2_rotate_qmmm_light

      SUBROUTINE qm2_rotate_qmmm_heavy(loop_count,IQM,xyz_qm,xyz_mm, &
                                       E1B,ENUC)
! 
! For heavy atoms
!
! This routine calculates the QMelectron - MMcore interactions 
! and the QMcore-MMcore interaction for a give QM-MM pair.
! Initially calculated in a local frame by qm2_repp_qmmm they are 
! subsequently rotated into the molecular frame by this routine.
! The core charge for the MM atom is the MM charge in eletcron 
! units.
!
! On Input:
!   loop_count : Offset into certain arrays
!   iqm        : Current qm atom in our loop from 1 to nquant
!   xyz_qm     : Cartesian coordinates of the QM atom.
!   xyz_mm     : Cartesian coordinates and charge (4) of MM atom.
!   E1B        : QM-MM electron core interaction in KCal/mol 
!                for each of the 10 multipoles of the QM atoms.
!                (s,s), (s,px), (s,py), (s,pz), (px,px), (px,py),
!                (px,pz), (py,py), (py,pz), (pz,pz)
!   ENUC       : QM-MM core core interaction in KCal/mol.
!
! Written by Ross Walker (TSRI, 2005)
!

  use qmmm_module, only : qmmm_nml,qm2_struct,qm2_params,  &
                              qm2_rij_eqns,qmmm_struct
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations
      implicit none
!Passed in
      integer, intent(in)    :: loop_count, iqm
      real(chm_real),  intent(in)  :: xyz_qm(3), xyz_mm(4)
      real(chm_real),  intent(out) :: e1b(10), enuc

!Local variables
      real(chm_real) :: RI(4)
      real(chm_real) :: oneBDDi1, oneBDDi2, oneBDDi3
      real(chm_real) :: sqrtrr2aqe, qmi_DD, qmi_QQ
      real(chm_real) :: X1, X2, X3,Y1, Y2, Y3,Z1, Z2,Z3, oneZ3, X3_2
      real(chm_real) :: chrgmm, RR2, RIJ1, RIJ, RR, RIJ2 
      real(chm_real) :: CHGMM_RI2, CHGMM_RI3, CHGMM_RI4
      real(chm_real) :: scale
      integer  :: i,iqmtype
      real(chm_real) :: temp_real,temp_real2,c1,anam1

      X1 = xyz_qm(1) - xyz_mm(1)
      X2 = xyz_qm(2) - xyz_mm(2)
      X3 = xyz_qm(3) - xyz_mm(3)
      chrgmm = xyz_mm(4)

!---RIJ and related equation storage specifics---
      if (qmmm_nml%qmmmrij_incore) then
         RIJ   = qm2_rij_eqns%qmmmrijdata(QMMMRIJ,loop_count)
         scale = qm2_rij_eqns%qmmmrijdata(QMMMEXP1,loop_count)+   &
                 qm2_rij_eqns%qmmmrijdata(QMMMEXP2,loop_count)
         RI(1) = AU_TO_EV*qm2_rij_eqns%qmmmrijdata(QMMMSQRTAEE,   &
                                                   loop_count)
         RIJ1  = qm2_rij_eqns%qmmmrijdata(QMMMONERIJ,loop_count)
         RI(2) = HALF_AU_TO_EV*   &
                 (qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRADDADE,   &
                                           loop_count)         &
                 -qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRMDDADE,   &
                                           loop_count))
         RI(3) = RI(1) + FOURTH_AU_TO_EV*   &
                 (qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRAQQAQE,   &
                                           loop_count)         &
                 +qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRMQQAQE,   &
                                           loop_count))   &
                       - HALF_AU_TO_EV*   &
                         qm2_rij_eqns%qmmmrijdata(QMMMSQRTRR2AQE,   &
                                                  loop_count)
         RI(4) = RI(1) + HALF_AU_TO_EV*   &
                        (qm2_rij_eqns%qmmmrijdata(   &
                                      QMMMSQRTRRAQQ2AQE,loop_count)   &
                        -qm2_rij_eqns%qmmmrijdata(   &
                                      QMMMSQRTRR2AQE,loop_count))
      else
         RIJ2    = X1*X1+X2*X2+X3*X3
         RR2     = RIJ2*A2_TO_BOHRS2
         RIJ1    = one/sqrt(RIJ2)
         RIJ     = one/RIJ1 
         RR      = RIJ*A_TO_BOHRS
         scale   = EXP(-qm2_params%cc_exp_params(iqm)*RIJ) +  &
                   EXP(-ALPH_MM*RIJ)
         oneBDDi1= qm2_params%multip_2c_elec_params(6,iqm)
         RI(1)   = AU_TO_EV/sqrt(RR2+oneBDDi1)
         qmi_DD  = qm2_params%multip_2c_elec_params(1,iqm)
         qmi_QQ  = two*qm2_params%multip_2c_elec_params(2,iqm)
         oneBDDi2= qm2_params%multip_2c_elec_params(7,iqm)
         oneBDDi3= qm2_params%multip_2c_elec_params(8,iqm)
         RI(2)   = HALF_AU_TO_EV*(one/SQRT((RR+qmi_DD)**2+oneBDDi2) -  &
                                  one/SQRT((RR-qmi_DD)**2+oneBDDi2))
         SQRTRR2AQE = one/SQRT(RR2+oneBDDi3)
         RI(3)   = RI(1) + FOURTH_AU_TO_EV*(one/SQRT((RR+qmi_QQ)**2 +   &
                                                     oneBDDi3) +   &
                                            one/SQRT((RR-qmi_QQ)**2 +   &
                                                     oneBDDi3))   &
                         - HALF_AU_TO_EV*SQRTRR2AQE
         RI(4)   = RI(1) + HALF_AU_TO_EV*(one/SQRT(RR2 +   &
                                                   (qmi_QQ*qmi_QQ) +   &
                                                   oneBDDi3) -   &
                                          SQRTRR2AQE)
      end if

!1) Step 1
!     Calculated the interaction between the QM electrons 
!     and the MM atomic charge for this QM-MM pair in the
!     local frame. MM charge it the MM charge, which is stored
!     in the 4th index of the MM cartesian coordinate array.
!     The reason the charge is packed into the 4th index of
!     the coordinate array is so that we move linearly 
!     through memory
!     : more cache hits. (RCW)

      if (qmmm_nml%qmmm_erep_incore) then
! Ross Walker - We will need these repulsion integrals later to 
!               calculate QM-MM derivatives so store them in.
         qm2_struct%qm_mm_e_repul(1,loop_count) = RI(1)
         qm2_struct%qm_mm_e_repul(2,loop_count) = RI(2)
         qm2_struct%qm_mm_e_repul(3,loop_count) = RI(3)
         qm2_struct%qm_mm_e_repul(4,loop_count) = RI(4)
      end if
! Otherwise we will calculate the repulsion integrals again 
! when we do the derivatives.

! S orbitals
      E1B(1)=-chrgmm*RI(1)

! Calculate QM-QM nuclear term
!    This is calculates as the QM core charge * chrgmm * RI(1)
!    Reuse E1B term to save doing the multiplication of 
!    chrgmm*RI(1) again.
      ENUC =-qm2_params%core_chg(IQM)*E1B(1)
      ENUC = ENUC+ABS(scale*ENUC)

! for AM1/PM3/PM3CARB1: Gaussian core-core terms.
!     PDDGPM3 should be done differently.
      If(qmmm_nml%QMMM_Gauss) then
        iqmtype= qmmm_struct%qm_atom_type(iqm)
        C1     = qm2_params%core_chg(IQM)*chrgmm
        anam1  = zero
        do i=1,qm2_params%num_fn(iqmtype)
            temp_real = RIJ-qm2_params%FN3(i,iqmtype)
            temp_real2= qm2_params%FN2(i,iqmtype)*temp_real*temp_real

            ! Skip doing the exponential if it is essentially zer
            if (temp_real2 .LT. EXPONENTIAL_CUTOFF) then
              anam1 = anam1+qm2_params%FN1(i,iqmtype)*EXP(-temp_real2)
            end if
         end do
         anam1 = anam1*c1*RIJ1
         ENUC = ENUC + anam1
      End if

      X1 = X1*RIJ1
      X2 = X2*RIJ1
      X3 = X3*RIJ1
      X3_2 = X3*X3

      if (abs(X3) .GT. (one-AXIS_TOL)) then
          X3 = SIGN(one,X3)
          Y1 = zero
          Y2 = one
          Y3 = zero
          Z1 = one
          Z2 = zero
          Z3 = zero
      else
          Z3   = SQRT(one-X3_2)
          oneZ3= one/Z3
          Y1   =-oneZ3*X2*SIGN(1.D0,X1)
          Y2   = ABS(oneZ3*X1)
          Y3   = zero
          Z1   =-oneZ3*X1*X3
          Z2   =-oneZ3*X2*X3
      end if

      CHGMM_RI2=-chrgmm*RI(2)
      CHGMM_RI3=-chrgmm*RI(3)
      CHGMM_RI4=-chrgmm*RI(4)

      E1B(2) = CHGMM_RI2*X1
      E1B(3) = CHGMM_RI3*X1*X1+CHGMM_RI4*((Y1*Y1)+(Z1*Z1))
      E1B(4) = CHGMM_RI2*X2
      E1B(5) = CHGMM_RI3*X2*X1+CHGMM_RI4*((Y2*Y1)+(Z2*Z1))
      E1B(6) = CHGMM_RI3*X2*X2+CHGMM_RI4*((Y2*Y2)+(Z2*Z2))
      E1B(7) = CHGMM_RI2*X3
      E1B(8) = CHGMM_RI3*X3*X1+CHGMM_RI4*Z3*Z1
      E1B(9) = CHGMM_RI3*X3*X2+CHGMM_RI4*Z3*Z2
      E1B(10)= CHGMM_RI3*X3_2+CHGMM_RI4*Z3*Z3

      return
      END SUBROUTINE qm2_rotate_qmmm_heavy


      SUBROUTINE qm2_h1elec(loop_count,R2,XI,XJ,n_atomic_orbi,   &
                            n_atomic_orbj,SHMAT,sxs_over_sas,ss_eqn,   &
                            sxp_over_sap,pxs_over_pas,sp_ovlp,ps_ovlp,   &
                            pxp_over_pap,pp_ovlp_ieqj1,pp_ovlp_ieqj2,   &
                            pp_ovlp_inj,betasas,betasap,betapas,betapap)
!
! qm2_h1elec forms the one-electron matrix between two atoms and
! calculates the overlaps.
!*****************************************************************
!   CALCULATE THE OVERLAP INTEGRALS USING A GAUSSIAN EXPANSION   *
!    STO-6G BY R.F. STEWART, J. CHEM. PHYS., 52 431-438, 1970    *
!                                                                *
!    FILL SHMAT=  4X4 ARRAY OF OVERLAPS, IN ORDER S,PX,PY,PZ     *
!******************************************************************
!         Current code optimised by Ross Walker (TSRI, 2005)
!
! ON INPUT
!   XI              : COORDINATES OF FIRST ATOM.
!   XJ              : COORDINATES OF SECOND ATOM.
!   n_atomic_orbi,j : number of atomic orbitals on i and j.
!                                                 
! ON OUTPUT
!   SHMAT           : MATRIX OF ONE-ELECTRON INTERACTIONS.
!                                                           
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
      implicit none
!Passed In
      real(chm_real), intent(in)  :: R2,XI(3),XJ(3)
      integer, intent(in) :: loop_count
      integer, intent(in) :: n_atomic_orbi, n_atomic_orbj
      real(chm_real), intent(out) :: SHMAT(4,4)
      real(chm_real), intent(in)  :: betasas,betasap,betapas,betapap

! These are passed in here rather than used from the module to 
! avoid having to do a look up on a 4 dimensional array within
! a loop.
      real(chm_real), intent(in):: sxs_over_sas(6,6), ss_eqn(6,6),   &
                             sxp_over_sap(6,6)
      real(chm_real), intent(in):: pxs_over_pas(6,6), sp_ovlp(6,6),   &
                             ps_ovlp(6,6)
      real(chm_real), intent(in):: pxp_over_pap(6,6), pp_ovlp_ieqj1(6,6),   &
                             pp_ovlp_ieqj2(6,6)
      real(chm_real), intent(in):: pp_ovlp_inj(6,6)
      
!Local variables
      real(chm_real)  :: BIJSP, BIJPS, BIJPP
      real(chm_real)  :: SH, vec_qm_qm(3)
      real(chm_real)  :: ADBR2, TOMB
      integer :: i,j,k,l, ii,jj

! R2 : Interatomic distance^2 in Bohrs2
      if (n_atomic_orbi.GT.1 .OR. n_atomic_orbj.GT.1) then
         vec_qm_qm(1) = (XI(1)-XJ(1))*A_TO_BOHRS
         vec_qm_qm(2) = (XI(2)-XJ(2))*A_TO_BOHRS
         vec_qm_qm(3) = (XI(3)-XJ(3))*A_TO_BOHRS
      end if

! S-S
      do K=1,6 !1 to NGAUSS
         do L=1,6
           ADBR2 = sxs_over_sas(k,l)*R2

! Check of overlap is non-zero before taking exponential
           IF (ADBR2 .LT. EXPONENTIAL_CUTOFF) THEN
              SHMAT(1,1)= SHMAT(1,1)+ss_eqn(k,l)*EXP(-ADBR2)
           ENDIF
         end do
      end do

! Multiply by S-S beta factor
      SHMAT(1,1)= SHMAT(1,1)*half*betasas

! S-P : atom J has P orbitals
      if (n_atomic_orbj .GT. 1) then
         do K=1,6                             ! 1 to NGAUSS
            do L=1,6
              ADBR2 = sxp_over_sap(k,l)*R2

! Check of overlap is non-zero before taking exponential
              IF (ADBR2 .LT. EXPONENTIAL_CUTOFF) THEN
                 SH        = sp_ovlp(k,l)*EXP(-ADBR2)
                 SHMAT(1,2)= SHMAT(1,2)+SH*vec_qm_qm(1)
                 SHMAT(1,3)= SHMAT(1,3)+SH*vec_qm_qm(2)
                 SHMAT(1,4)= SHMAT(1,4)+SH*vec_qm_qm(3)
              ENDIF
            end do
         end do

! Multiply by S-P beta factor
         BIJSP     = half*betasap
         SHMAT(1,2)= SHMAT(1,2)*BIJSP
         SHMAT(1,3)= SHMAT(1,3)*BIJSP
         SHMAT(1,4)= SHMAT(1,4)*BIJSP
      end if

! P-S : atom I has P orbitals
      if (n_atomic_orbi .GT. 1) then
         do K=1,6                             ! 1 to NGAUSS
            do L=1,6
              ADBR2 = pxs_over_pas(l,k)*R2

! Check of overlap is non-zero before taking exponential
              IF (ADBR2 .LT. EXPONENTIAL_CUTOFF) THEN
                 SH        =-ps_ovlp(l,k)*EXP(-ADBR2)
                 SHMAT(2,1)= SHMAT(2,1)+SH*vec_qm_qm(1)
                 SHMAT(3,1)= SHMAT(3,1)+SH*vec_qm_qm(2)
                 SHMAT(4,1)= SHMAT(4,1)+SH*vec_qm_qm(3)
              ENDIF
            end do
         end do

! Multiply by P-S beta factor
         BIJPS     = half*betapas
         SHMAT(2,1)= SHMAT(2,1)*BIJPS
         SHMAT(3,1)= SHMAT(3,1)*BIJPS
         SHMAT(4,1)= SHMAT(4,1)*BIJPS
      end if

! P-P : both atoms have P orbitals
      if (n_atomic_orbi .GT. 1 .AND. n_atomic_orbj .GT. 1) then
         BIJPP = half*betapap
         do I=1,n_atomic_orbi-1
            ii=i+1
            do J=1,n_atomic_orbj-1
               jj=j+1

! P-P
               TOMB = vec_qm_qm(i)*vec_qm_qm(j)
               do K=1,6                          ! 1 to NGAUSS
                  do L=1,6
                     ADBR2=pxp_over_pap(k,l)*R2

! Check of overlap is non-zero before taking exponential
                     IF (ADBR2 .LT. EXPONENTIAL_CUTOFF) THEN
                        if (ii.EQ.jj) then
                           SH = EXP(-ADBR2)*(pp_ovlp_ieqj1(k,l)*TOMB +   &
                                             pp_ovlp_ieqj2(k,l))
                        else
                           SH = EXP(-ADBR2)*TOMB*pp_ovlp_inj(k,l)
                        end if

                        SHMAT(ii,jj)= SHMAT(ii,jj)+SH
                     END IF
                  end do 
               end do 

! Multiply by P-P beta factor
               SHMAT(ii,jj) = SHMAT(ii,jj)*BIJPP

            end do                            ! j=1,n_atomic_orbj-1
         end do                               ! i=1,n_atomic_orbi-1
      end if

      return
      END SUBROUTINE qm2_h1elec


      SUBROUTINE qm2_rotate_qmqm(loop_count,IQM,JQM,NI,NJ,XI,XJ,  &
                                 W,KR,E1B,E2A,ENUC,qmitype,qmjtype)
!
!   ROTATE CALCULATES THE TWO-PARTICLE INTERACTIONS.
!
!   ON INPUT
!             IQM    = qm atom I number (in 1 to nquant loop)
!             JQM    = qm atom J number in inner loop
!             NI     = ATOMIC NUMBER OF FIRST ATOM.
!             NJ     = ATOMIC NUMBER OF SECOND ATOM.
!             XI     = COORDINATE OF FIRST ATOM.
!             XJ     = COORDINATE OF SECOND ATOM.
!
! ON OUTPUT W      = ARRAY OF TWO-ELECTRON REPULSION INTEGRALS.
!           E1B,E2A= ARRAY OF ELECTRON-NUCLEAR ATTRACTION 
!                    INTEGRALS,
!                    E1B = ELECTRON ON ATOM NI ATTRACTING 
!                    NUCLEUS OF NJ.
!           ENUC   = NUCLEAR-NUCLEAR REPULSION TERM.
!
! *** THIS ROUTINE COMPUTES THE REPULSION AND NUCLEAR ATTRACTION
!     INTEGRALS OVER MOLECULAR-FRAME COORDINATES.  THE INTEGRALS 
!     OVER LOCAL FRAME COORDINATES ARE EVALUATED BY SUBROUTINE
!     REPP AND STORED AS FOLLOWS
!     (WHERE P-SIGMA = O,   AND P-PI = P AND P* ) IN RI
!     (SS/SS)=1,  (SO/SS)=2,  (OO/SS)=3,  (PP/SS)=4,  (SS/OS)=5,
!     (SO/SO)=6,  (SP/SP)=7,  (OO/SO)=8,  (PP/SO)=9,  (PO/SP)=10,
!     (SS/OO)=11, (SS/PP)=12, (SO/OO)=13, (SO/PP)=14, (SP/OP)=15,
!     (OO/OO)=16, (PP/OO)=17, (OO/PP)=18, (PP/PP)=19, (PO/PO)=20,
!     (PP/P*P*)=21,  (P*P/P*P)=22.
!
! Inlining and Optimising by: Ross Walker (TSRI, 2005)
!

  use qmmm_module, only : qmmm_nml,qmmm_struct,qm2_struct, &
                              qm2_params,qm2_rij_eqns
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations
      implicit none
!Passed in
      integer, intent(in)    :: loop_count, iqm, jqm, ni, nj,   &
                                qmitype,qmjtype
      integer, intent(inout) :: kr
      real(chm_real), intent(in)     :: xi(3), xj(3)
      real(chm_real), intent(out)    :: W(100), e1b(10), E2A(10), enuc

!Local
      real(chm_real)  :: X(3),Y(3),Z(3), RI(22)
      real(chm_real)  :: temp_real, temp_real2, anam1, C1, oneRIJ, RIJ
      real(chm_real)  :: rr,rr2, exp1i, exp1j, sqrtaee, BDD1i, BDD1j, bdd1ij
      real(chm_real)  :: a, xx11, xx21, xx22, xx31, xx32, xx33,  &
                   yy11, yy21, yy22
      real(chm_real)  :: zz11, zz21, zz22, zz31, zz32, zz33,  &
                   yyzz11, yyzz21, yyzz22
      real(chm_real)  :: xy11, xy21, xy22, xy31, xy32,  &
                   xz11, xz21, xz22, xz31, xz32, xz33
      real(chm_real)  :: YZ11, yz21, yz22, yz31, yz32
      real(chm_real)  :: css1, css2, csp1, cpps1, cppp1, csp2, cpps2, cppp2,  &
                   scale
      real(chm_real)  :: PDDG_EXP1, PDDG_EXP2, PDDG_EXP3, PDDG_EXP4, PDDG_CORR
      integer :: i, ki
      logical :: SI,SJ
      LOGICAL :: AM1_OR_PM3, PDDG_IN_USE, first_call

      data first_call /.true./
      save first_call
      save AM1_OR_PM3, PDDG_IN_USE

      IF (first_call) THEN
         first_call  = .false.
         AM1_OR_PM3  = (qmmm_nml%qmtheory .EQ. AM1 .OR.   &
                        qmmm_nml%qmtheory .EQ. PM3 .OR.   &
                        qmmm_nml%qmtheory .EQ. PDDGPM3 .OR.   &
                        qmmm_nml%qmtheory .EQ. PM3CARB1 ) 
         PDDG_IN_USE = (qmmm_nml%qmtheory .EQ. PDDGPM3 .OR.   &
                        qmmm_nml%qmtheory .EQ. PDDGMNDO )
      ENDIF

      X(1)=XI(1)-XJ(1)
      X(2)=XI(2)-XJ(2)
      X(3)=XI(3)-XJ(3)

      if (qmmm_nml%qmqmrij_incore) then

! We already have RIJ info in memory
         rr2    = qm2_rij_eqns%qmqmrijdata(QMQMRIJBOHRS2,loop_count)
         oneRIJ = qm2_rij_eqns%qmqmrijdata(QMQMONERIJ,loop_count)
         RIJ    = qm2_rij_eqns%qmqmrijdata(QMQMRIJ,loop_count)
         rr     = qm2_rij_eqns%qmqmrijdata(QMQMRIJBOHRS,loop_count)
         exp1i  = qm2_rij_eqns%qmqmrijdata(QMQMEXP1I,loop_count)
         exp1j  = qm2_rij_eqns%qmqmrijdata(QMQMEXP1J,loop_count)
         sqrtaee= qm2_rij_eqns%qmqmrijdata(QMQMSQRTAEE,loop_count)

         if (PDDG_IN_USE) then
            PDDG_EXP1 = qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP1,   &
                                                 loop_count)
            PDDG_EXP2 = qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP2,   &
                                                 loop_count)
            PDDG_EXP3 = qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP3,   &
                                                 loop_count)
            PDDG_EXP4 = qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP4,   &
                                                 loop_count)
         end if
      else
         RIJ     = X(1)*X(1)+X(2)*X(2)+X(3)*X(3) 
         rr2     = RIJ*A2_TO_BOHRS2
         oneRIJ  = one/SQRT(RIJ)
         RIJ     = RIJ*oneRIJ                  ! faster than one/oneRIJ
         rr      = RIJ*A_TO_BOHRS
         exp1i   = EXP(-qm2_params%cc_exp_params(iqm)*RIJ)
         exp1j   = EXP(-qm2_params%cc_exp_params(jqm)*RIJ)
         BDD1i   = qm2_params%multip_2c_elec_params(3,iqm)
         BDD1j   = qm2_params%multip_2c_elec_params(3,jqm)
         BDD1ij  = bdd1i+bdd1j
         BDD1ij  = bdd1ij*bdd1ij
         SQRTAEE = one/sqrt(RR2+bdd1ij)

         if (PDDG_IN_USE) then
! PDDG Specific terms
            PDDG_EXP1 = EXP(-ten*(RIJ-qm2_params%pddge1(iqm) -   &
                                  qm2_params%pddge1(jqm))**2)
            PDDG_EXP2 = EXP(-ten*(RIJ-qm2_params%pddge1(iqm) -   &
                                  qm2_params%pddge2(jqm))**2)
            PDDG_EXP3 = EXP(-ten*(RIJ-qm2_params%pddge2(iqm) -   &
                                  qm2_params%pddge1(jqm))**2)
            PDDG_EXP4 = EXP(-ten*(RIJ-qm2_params%pddge2(iqm) -   &
                                  qm2_params%pddge2(jqm))**2)
         end if
      end if
      X(1) = X(1)*oneRIJ
      X(2) = X(2)*oneRIJ
      X(3) = X(3)*oneRIJ

      CALL qm2_repp(iqm,jqm,rr,rr2,RI,SQRTAEE)
      IF(qmmm_nml%qmqm_erep_incore) then
! Ross Walker - We will need these repulsion integrals later to
!               calculate qm-qm analytical derivatives so store
!               them in our array.
         do i=1,22
            qm2_struct%qm_qm_e_repul(i,loop_count) = RI(I)
         end do
      end if

      IF (ABS(X(3)).GT.0.99999999D0) THEN
         X(3) = SIGN(one,X(3))
         Y(1) = zero
         Y(2) = one
         Y(3) = zero
         Z(1) = one
         Z(2) = zero
         Z(3) = zero
      ELSE
         Z(3) = SQRT(one-X(3)*X(3))
         A    = one/Z(3)
         Y(1) =-A*X(2)*SIGN(one,X(1))
         Y(2) = ABS(A*X(1))
         Y(3) = zero
         Z(1) =-A*X(1)*X(3)
         Z(2) =-A*X(2)*X(3)
      ENDIF
      SI = (qm2_params%natomic_orbs(iqm) .GT. 1)
      SJ = (qm2_params%natomic_orbs(jqm) .GT. 1)
      IF ( SI .OR. SJ) THEN
         XX11   = X(1)*X(1)
         XX21   = X(2)*X(1)
         XX22   = X(2)*X(2)
         XX31   = X(3)*X(1)
         XX32   = X(3)*X(2)
         XX33   = X(3)*X(3)
         YY11   = Y(1)*Y(1)
         YY21   = Y(2)*Y(1)
         YY22   = Y(2)*Y(2)
         ZZ11   = Z(1)*Z(1)
         ZZ21   = Z(2)*Z(1)
         ZZ22   = Z(2)*Z(2)
         ZZ31   = Z(3)*Z(1)
         ZZ32   = Z(3)*Z(2)
         ZZ33   = Z(3)*Z(3)
         YYZZ11 = YY11+ZZ11
         YYZZ21 = YY21+ZZ21
         YYZZ22 = YY22+ZZ22
         XY11   = two*X(1)*Y(1)
         XY21   =     X(1)*Y(2)+X(2)*Y(1)
         XY22   = two*X(2)*Y(2)
         XY31   =     X(3)*Y(1)
         XY32   =     X(3)*Y(2)
         XZ11   = two*X(1)*Z(1)
         XZ21   =     X(1)*Z(2)+X(2)*Z(1)
         XZ22   = two*X(2)*Z(2)
         XZ31   =     X(1)*Z(3)+X(3)*Z(1)
         XZ32   =     X(2)*Z(3)+X(3)*Z(2)
         XZ33   = two*X(3)*Z(3)
         YZ11   = two*Y(1)*Z(1)
         YZ21   =     Y(1)*Z(2)+Y(2)*Z(1)
         YZ22   = two*Y(2)*Z(2)
         YZ31   =     Y(1)*Z(3)
         YZ32   =     Y(2)*Z(3)
      ENDIF

      W(1)=RI(1)                              ! (S S/S S)
      KI = 1
      IF (SJ) THEN
         W(2) =RI(5)*X(1)                     ! (S S/PX S)
         W(3) =RI(11)*XX11+RI(12)*YYZZ11      ! (S S/PX PX)
         W(4) =RI(5)*X(2)                     ! (S S/PY S)
         W(5) =RI(11)*XX21+RI(12)*YYZZ21      ! (S S/PY PX)
         W(6) =RI(11)*XX22+RI(12)*YYZZ22      ! (S S/PY PY)
         W(7) =RI(5)*X(3)                     ! (S S/PZ S)
         W(8) =RI(11)*XX31+RI(12)*ZZ31        ! (S S/PZ PX)
         W(9) =RI(11)*XX32+RI(12)*ZZ32        ! (S S/PZ PY)
         W(10)=RI(11)*XX33+RI(12)*ZZ33        ! (S S/PZ PZ)
         KI   =10
      ENDIF
      IF (SI) THEN
         IF (SJ) THEN
            W(11)=RI(2)*X(1)                                         ! (PX S/S S)
            W(12)=RI(6)*XX11+RI(7)*YYZZ11                            ! (PX S/PX S)
            W(13)=X(1)*(RI(13)*XX11+RI(14)*YYZZ11)+   &
                  RI(15)*(Y(1)*XY11+Z(1)*XZ11)                       ! (PX S/PX PX)
            W(14)=RI(6)*XX21+RI(7)*YYZZ21                            ! PX S/PY S)
            W(15)=X(1)*(RI(13)*XX21+RI(14)*YYZZ21)+   &
                  RI(15)*(Y(1)*XY21+Z(1)*XZ21)                       ! (PX S/PY PX)
            W(16)=X(1)*(RI(13)*XX22+RI(14)*YYZZ22)+   &
                  RI(15)*(Y(1)*XY22+Z(1)*XZ22)                       ! (PX S/PY PY)
            W(17)=RI(6)*XX31+RI(7)*ZZ31                              ! (PX S/PZ S)
            W(18)=X(1)*(RI(13)*XX31+RI(14)*ZZ31)+  &
                  RI(15)*(Y(1)*XY31+Z(1)*XZ31)                       ! (PX S/PZ PX)
            W(19)=X(1)*(RI(13)*XX32+RI(14)*ZZ32)+  &
                  RI(15)*(Y(1)*XY32+Z(1)*XZ32)                       ! (PX S/PZ PY)
            W(20)=X(1)*(RI(13)*XX33+RI(14)*ZZ33)+  &
                  RI(15)*(Z(1)*XZ33)                                 ! (PX S/PZ PZ)
            W(21)=RI(3)*XX11+RI(4)*YYZZ11                            ! (PX PX/S S)
            W(22)=X(1)*(RI(8)*XX11+RI(9)*YYZZ11)+   &
                  RI(10)*(Y(1)*XY11+Z(1)*XZ11)                       ! (PX PX/PX S)
            W(23)=(RI(16)*XX11+RI(17)*YYZZ11)*XX11   &
                  +RI(18)*XX11*YYZZ11+RI(19)*(YY11*YY11+ZZ11*ZZ11)   &
                  +RI(20)*(XY11*XY11+XZ11*XZ11)+RI(21)*(YY11*ZZ11+   &
                                                        ZZ11*YY11)   &
                  +RI(22)*YZ11*YZ11                                  ! (PX PX/PX PX)
            W(24)=X(2)*(RI(8)*XX11+RI(9)*YYZZ11)+   &
                  RI(10)*(Y(2)*XY11+Z(2)*XZ11)                       ! (PX PX/PY S)
            W(25)=(RI(16)*XX11+RI(17)*YYZZ11)*XX21   &
                  +RI(18)*XX11*YYZZ21+RI(19)*(YY11*YY21+ZZ11*ZZ21)   &
                  +RI(20)*(XY11*XY21+XZ11*XZ21)+RI(21)*(YY11*ZZ21+   &
                                                        ZZ11*YY21)   &
                  +RI(22)*YZ11*YZ21                                  ! (PX PX/PY PX)
            W(26)=(RI(16)*XX11+RI(17)*YYZZ11)*XX22   &
                  +RI(18)*XX11*YYZZ22+RI(19)*(YY11*YY22+ZZ11*ZZ22)   &
                  +RI(20)*(XY11*XY22+XZ11*XZ22)+RI(21)*(YY11*ZZ22+   &
                                                        ZZ11*YY22)   &
                  +RI(22)*YZ11*YZ22                                  ! (PX PX/PY PY)
            W(27)=X(3)*(RI(8)*XX11+RI(9)*YYZZ11)+   &
                  RI(10)*(         +Z(3)*XZ11)                       ! (PX PX/PZ S)    ????
            W(28)=(RI(16)*XX11+RI(17)*YYZZ11)*XX31   &
                  +(RI(18)*XX11+RI(19)*ZZ11+RI(21)*YY11)*ZZ31   &
                  +RI(20)*(XY11*XY31+XZ11*XZ31)+RI(22)*YZ11*YZ31     ! (PX PX/PZ PX)
            W(29)=(RI(16)*XX11+RI(17)*YYZZ11)*XX32   &
                  +(RI(18)*XX11+RI(19)*ZZ11+RI(21)*YY11)*ZZ32   &
                  +RI(20)*(XY11*XY32+XZ11*XZ32)+RI(22)*YZ11*YZ32     ! (PX PX/PZ PY)
            W(30)=(RI(16)*XX11+RI(17)*YYZZ11)*XX33   &
                  +(RI(18)*XX11+RI(19)*ZZ11+RI(21)*YY11)*ZZ33   &
                  +RI(20)*XZ11*XZ33                                  ! (PX PX/PZ PZ)
            W(31)=RI(2)*X(2)                                         ! (PY S/S S)
            W(32)=RI(6)*XX21+RI(7)*YYZZ21                            ! (PY S/PX S)
            W(33)=X(2)*(RI(13)*XX11+RI(14)*YYZZ11)+   &
                  RI(15)*(Y(2)*XY11+Z(2)*XZ11)                       ! (PY S/PX PX)
            W(34)=RI(6)*XX22+RI(7)*YYZZ22                            ! (PY S/PY S)
            W(35)=X(2)*(RI(13)*XX21+RI(14)*YYZZ21)+   &
                  RI(15)*(Y(2)*XY21+Z(2)*XZ21)                       ! (PY S/PY PX)
            W(36)=X(2)*(RI(13)*XX22+RI(14)*YYZZ22)+   &
                  RI(15)*(Y(2)*XY22+Z(2)*XZ22)                       ! (PY S/PY PY)
            W(37)=RI(6)*XX32+RI(7)*ZZ32                              ! (PY S/PZ S)
            W(38)=X(2)*(RI(13)*XX31+RI(14)*ZZ31)+   &
                  RI(15)*(Y(2)*XY31+Z(2)*XZ31)                       ! (PY S/PZ PX)
            W(39)=X(2)*(RI(13)*XX32+RI(14)*ZZ32)+   &
                  RI(15)*(Y(2)*XY32+Z(2)*XZ32)                       ! (PY S/PZ PY)
            W(40)=X(2)*(RI(13)*XX33+RI(14)*ZZ33)+   &
                  RI(15)*(         +Z(2)*XZ33)                       ! (PY S/PZ PZ)   ????
            W(41)=RI(3)*XX21+RI(4)*YYZZ21                            ! (PY PX/S S)
            W(42)=X(1)*(RI(8)*XX21+RI(9)*YYZZ21)+   &
                  RI(10)*(Y(1)*XY21+Z(1)*XZ21)                       ! (PY PX/PX S)
            W(43)=(RI(16)*XX21+RI(17)*YYZZ21)*XX11   &
                  +RI(18)*XX21*YYZZ11+RI(19)*(YY21*YY11+ZZ21*ZZ11)   &
                  +RI(20)*(XY21*XY11+XZ21*XZ11)+RI(21)*(YY21*ZZ11+   &
                                                        ZZ21*YY11)   &
                  +RI(22)*YZ21*YZ11                                  ! (PY PX/PX PX)
            W(44)=X(2)*(RI(8)*XX21+RI(9)*YYZZ21)+   &
                  RI(10)*(Y(2)*XY21+Z(2)*XZ21)                       ! (PY PX/PY S)
            W(45)=(RI(16)*XX21+RI(17)*YYZZ21)*XX21   &
                  +RI(18)*XX21*YYZZ21+RI(19)*(YY21*YY21+ZZ21*ZZ21)   &
                  +RI(20)*(XY21*XY21+XZ21*XZ21)+RI(21)*(YY21*ZZ21+   &
                                                        ZZ21*YY21)   &
                  +RI(22)*YZ21*YZ21                                  ! (PY PX/PY PX)
            W(46)=(RI(16)*XX21+RI(17)*YYZZ21)*XX22   &
                  +RI(18)*XX21*YYZZ22+RI(19)*(YY21*YY22+ZZ21*ZZ22)   &
                  +RI(20)*(XY21*XY22+XZ21*XZ22)+RI(21)*(YY21*ZZ22+   &
                                                        ZZ21*YY22)   &
                  +RI(22)*YZ21*YZ22                                  ! (PY PX/PY PY)
            W(47)=X(3)*(RI(8)*XX21+RI(9)*YYZZ21)+   &
                  RI(10)*(         +Z(3)*XZ21)                       ! (PY PX/PZ S)    ????
            W(48)=(RI(16)*XX21+RI(17)*YYZZ21)*XX31   &
                  +(RI(18)*XX21+RI(19)*ZZ21+RI(21)*YY21)*ZZ31   &
                  +RI(20)*(XY21*XY31+XZ21*XZ31)+RI(22)*YZ21*YZ31     ! (PY PX/PZ PX)
            W(49)=(RI(16)*XX21+RI(17)*YYZZ21)*XX32   &
                  +(RI(18)*XX21+RI(19)*ZZ21+RI(21)*YY21)*ZZ32   &
                  +RI(20)*(XY21*XY32+XZ21*XZ32)+RI(22)*YZ21*YZ32     ! (PY PX/PZ PY)
            W(50)=(RI(16)*XX21+RI(17)*YYZZ21)*XX33   &
                  +(RI(18)*XX21+RI(19)*ZZ21+RI(21)*YY21)*ZZ33   &
                  +RI(20)*XZ21*XZ33                                  ! (PY PX/PZ PZ)
            W(51)=RI(3)*XX22+RI(4)*YYZZ22                            ! (PY PY/S S)
            W(52)=X(1)*(RI(8)*XX22+RI(9)*YYZZ22)+   &
                  RI(10)*(Y(1)*XY22+Z(1)*XZ22)                       ! (PY PY/PX S)
            W(53)=(RI(16)*XX22+RI(17)*YYZZ22)*XX11   &
                  +RI(18)*XX22*YYZZ11+RI(19)*(YY22*YY11+ZZ22*ZZ11)   &
                  +RI(20)*(XY22*XY11+XZ22*XZ11)+RI(21)*(YY22*ZZ11+   &
                                                        ZZ22*YY11)   &
                  +RI(22)*YZ22*YZ11                                  ! (PY PY/PX PX)
            W(54)=X(2)*(RI(8)*XX22+RI(9)*YYZZ22)+   &
                  RI(10)*(Y(2)*XY22+Z(2)*XZ22)                       ! (PY PY/PY S)
            W(55)=(RI(16)*XX22+RI(17)*YYZZ22)*XX21   &
                  +RI(18)*XX22*YYZZ21+RI(19)*(YY22*YY21+ZZ22*ZZ21)   &
                  +RI(20)*(XY22*XY21+XZ22*XZ21)+RI(21)*(YY22*ZZ21+   &
                                                        ZZ22*YY21)   &
                  +RI(22)*YZ22*YZ21                                  ! (PY PY/PY PX)
            W(56)=(RI(16)*XX22+RI(17)*YYZZ22)*XX22   &
                  +RI(18)*XX22*YYZZ22+RI(19)*(YY22*YY22+ZZ22*ZZ22)   &
                  +RI(20)*(XY22*XY22+XZ22*XZ22)+RI(21)*(YY22*ZZ22+   &
                                                        ZZ22*YY22)   &
                  +RI(22)*YZ22*YZ22                                  ! (PY PY/PY PY)
            W(57)=X(3)*(RI(8)*XX22+RI(9)*YYZZ22)+   &
                  RI(10)*(         +Z(3)*XZ22)                       ! (PY PY/PZ S)
            W(58)=(RI(16)*XX22+RI(17)*YYZZ22)*XX31   &
                  +(RI(18)*XX22+RI(19)*ZZ22+RI(21)*YY22)*ZZ31   &
                  +RI(20)*(XY22*XY31+XZ22*XZ31)+RI(22)*YZ22*YZ31     ! (PY PY/PZ PX)
            W(59)=(RI(16)*XX22+RI(17)*YYZZ22)*XX32   &
                  +(RI(18)*XX22+RI(19)*ZZ22+RI(21)*YY22)*ZZ32   &
                  +RI(20)*(XY22*XY32+XZ22*XZ32)+RI(22)*YZ22*YZ32     ! (PY PY/PZ PY)
            W(60)=(RI(16)*XX22+RI(17)*YYZZ22)*XX33   &
                  +(RI(18)*XX22+RI(19)*ZZ22+RI(21)*YY22)*ZZ33   &
                  +RI(20)*XZ22*XZ33                                  ! (PY PY/PZ PZ)
            W(61)=RI(2)*X(3)                                         ! (PZ S/SS)
            W(62)=RI(6)*XX31+RI(7)*ZZ31                              ! (PZ S/PX S)
            W(63)=X(3)*(RI(13)*XX11+RI(14)*YYZZ11)+   &
                  RI(15)*(         +Z(3)*XZ11)                       ! (PZ S/PX PX)
            W(64)=RI(6)*XX32+RI(7)*ZZ32                              ! (PZ S/PY S)
            W(65)=X(3)*(RI(13)*XX21+RI(14)*YYZZ21)+   &
                  RI(15)*(         +Z(3)*XZ21)                       ! (PZ S/PY PX)
            W(66)=X(3)*(RI(13)*XX22+RI(14)*YYZZ22)+   &
                  RI(15)*(         +Z(3)*XZ22)                       ! (PZ S/PY PY)
            W(67)=RI(6)*XX33+RI(7)*ZZ33                              ! (PZ S/PZ S)
            W(68)=X(3)*(RI(13)*XX31+RI(14)*ZZ31)+   &
                  RI(15)*(         +Z(3)*XZ31)                       ! (PZ S/PZ PX)
            W(69)=X(3)*(RI(13)*XX32+RI(14)*ZZ32)+   &
                  RI(15)*(         +Z(3)*XZ32)                       ! (PZ S/PZ PY)
            W(70)=X(3)*(RI(13)*XX33+RI(14)*ZZ33)+   &
                  RI(15)*(         +Z(3)*XZ33)                       ! (PZ S/PZ PZ)
            W(71)=RI(3)*XX31+RI(4)*ZZ31                              ! (PZ PX/S S)
            W(72)=X(1)*(RI(8)*XX31+RI(9)*ZZ31)+   &
                  RI(10)*(Y(1)*XY31+Z(1)*XZ31)                       ! (PZ PX/PX S)
            W(73)=(RI(16)*XX31+RI(17)*ZZ31)*XX11+RI(18)*XX31*YYZZ11   &
                  +RI(19)*ZZ31*ZZ11+RI(20)*(XY31*XY11+XZ31*XZ11)   &
                  +RI(21)*ZZ31*YY11+RI(22)*YZ31*YZ11                 ! (PZ PX/PX PX)
            W(74)=X(2)*(RI(8)*XX31+RI(9)*ZZ31)+   &
                  RI(10)*(Y(2)*XY31+Z(2)*XZ31)                       ! (PZ PX/PY S)
            W(75)=(RI(16)*XX31+RI(17)*ZZ31)*XX21+RI(18)*XX31*YYZZ21   &
                  +RI(19)*ZZ31*ZZ21+RI(20)*(XY31*XY21+XZ31*XZ21)   &
                  +RI(21)*ZZ31*YY21+RI(22)*YZ31*YZ21                 ! (PZ PX/PY PX)
            W(76)=(RI(16)*XX31+RI(17)*ZZ31)*XX22+RI(18)*XX31*YYZZ22   &
                  +RI(19)*ZZ31*ZZ22+RI(20)*(XY31*XY22+XZ31*XZ22)   &
                  +RI(21)*ZZ31*YY22+RI(22)*YZ31*YZ22                 ! (PZ PX/PY PY)
            W(77)=X(3)*(RI(8)*XX31+RI(9)*ZZ31)+   &
                  RI(10)*(         +Z(3)*XZ31)                       ! (PZ PX/PZ S)
            W(78)=(RI(16)*XX31+RI(17)*ZZ31)*XX31   &
                  +(RI(18)*XX31+RI(19)*ZZ31)*ZZ31   &
                  +RI(20)*(XY31*XY31+XZ31*XZ31)   &
                  +RI(22)*YZ31*YZ31                                  ! (PZ PX/PZ PX)
            W(79)=(RI(16)*XX31+RI(17)*ZZ31)*XX32   &
                  +(RI(18)*XX31+RI(19)*ZZ31)*ZZ32   &
                  +RI(20)*(XY31*XY32+XZ31*XZ32)   &
                  +RI(22)*YZ31*YZ32                                  ! (PZ PX/PZ PY)
            W(80)=(RI(16)*XX31+RI(17)*ZZ31)*XX33   &
                  +(RI(18)*XX31+RI(19)*ZZ31)*ZZ33+RI(20)*XZ31*XZ33   ! (PZ PX/PZ PZ)
            W(81)=RI(3)*XX32+RI(4)*ZZ32                              ! (PZ PY/S S)
            W(82)=X(1)*(RI(8)*XX32+RI(9)*ZZ32)+   &
                  RI(10)*(Y(1)*XY32+Z(1)*XZ32)                       ! (PZ PY/PX S)
            W(83)=(RI(16)*XX32+RI(17)*ZZ32)*XX11+RI(18)*XX32*YYZZ11   &
                  +RI(19)*ZZ32*ZZ11+RI(20)*(XY32*XY11+XZ32*XZ11)   &
                  +RI(21)*ZZ32*YY11+RI(22)*YZ32*YZ11                 ! (PZ PY/PX PX)
            W(84)=X(2)*(RI(8)*XX32+RI(9)*ZZ32)+   &
                  RI(10)*(Y(2)*XY32+Z(2)*XZ32)                       ! (PZ PY/PY S)
            W(85)=(RI(16)*XX32+RI(17)*ZZ32)*XX21+RI(18)*XX32*YYZZ21   &
                  +RI(19)*ZZ32*ZZ21+RI(20)*(XY32*XY21+XZ32*XZ21)   &
                  +RI(21)*ZZ32*YY21+RI(22)*YZ32*YZ21                 ! (PZ PY/PY PX)
            W(86)=(RI(16)*XX32+RI(17)*ZZ32)*XX22+RI(18)*XX32*YYZZ22   &
                  +RI(19)*ZZ32*ZZ22+RI(20)*(XY32*XY22+XZ32*XZ22)   &
                  +RI(21)*ZZ32*YY22+RI(22)*YZ32*YZ22                 ! (PZ PY/PY PY)
            W(87)=X(3)*(RI(8)*XX32+RI(9)*ZZ32)+   &
                  RI(10)*(         +Z(3)*XZ32)                       ! (PZ PY/PZ S)
            W(88)=(RI(16)*XX32+RI(17)*ZZ32)*XX31   &
                  +(RI(18)*XX32+RI(19)*ZZ32)*ZZ31+RI(20)*(XY32*XY31+   &
                                                          XZ32*XZ31)   &
                  +RI(22)*YZ32*YZ31                                  ! (PZ PY/PZ PX)
            W(89)=(RI(16)*XX32+RI(17)*ZZ32)*XX32   &
                  +(RI(18)*XX32+RI(19)*ZZ32)*ZZ32+RI(20)*(XY32*XY32+   &
                                                          XZ32*XZ32)   &
                  +RI(22)*YZ32*YZ32                                  ! (PZ PY/PZ PY)
            W(90)=(RI(16)*XX32+RI(17)*ZZ32)*XX33   &
                  +(RI(18)*XX32+RI(19)*ZZ32)*ZZ33+RI(20)*XZ32*XZ33   ! (PZ PY/PZ PZ)
            W(91)=RI(3)*XX33+RI(4)*ZZ33                              ! (PZ PZ/S S)
            W(92)=X(1)*(RI(8)*XX33+RI(9)*ZZ33)+   &
                  RI(10)*(          Z(1)*XZ33)                       ! (PZ PZ/PX S)
            W(93)=(RI(16)*XX33+RI(17)*ZZ33)*XX11+RI(18)*XX33*YYZZ11   &
                  +RI(19)*ZZ33*ZZ11+RI(20)*XZ33*XZ11   &
                  +RI(21)*ZZ33*YY11                                  ! (PZ PZ/PX PX)
            W(94)=X(2)*(RI(8)*XX33+RI(9)*ZZ33)+   &
                  RI(10)*(         +Z(2)*XZ33)                       ! (PZ PZ/PY S)
            W(95)=(RI(16)*XX33+RI(17)*ZZ33)*XX21+RI(18)*XX33*YYZZ21   &
                  +RI(19)*ZZ33*ZZ21+RI(20)*XZ33*XZ21   &
                  +RI(21)*ZZ33*YY21                                  ! (PZ PZ/PY PX)
            W(96)=(RI(16)*XX33+RI(17)*ZZ33)*XX22+RI(18)*XX33*YYZZ22   &
                  +RI(19)*ZZ33*ZZ22+RI(20)*XZ33*XZ22   &
                  +RI(21)*ZZ33*YY22                                  ! (PZ PZ/PY PY)
            W(97)=X(3)*(RI(8)*XX33+RI(9)*ZZ33)+   &
                  RI(10)*(         +Z(3)*XZ33)                       ! PZ PZ/PZ S)
            W(98)=(RI(16)*XX33+RI(17)*ZZ33)*XX31   &
                  +(RI(18)*XX33+RI(19)*ZZ33)*ZZ31   &
                  +RI(20)*XZ33*XZ31                                  ! (PZ PZ/PZ PX)
            W(99)=(RI(16)*XX33+RI(17)*ZZ33)*XX32   &
                  +(RI(18)*XX33+RI(19)*ZZ33)*ZZ32   &
                  +RI(20)*XZ33*XZ32                                  ! (PZ PZ/PZ PY)
            W(100)=(RI(16)*XX33+RI(17)*ZZ33)*XX33   &
                   +(RI(18)*XX33+RI(19)*ZZ33)*ZZ33   &
                   +RI(20)*XZ33*XZ33                                 ! (PZ PZ/PZ PZ)
            KI = 100                                                         
         ELSE
            W(2)=RI(2)*X(1)                   ! (PX S/S S)
            W(3)=RI(3)*XX11+RI(4)*YYZZ11      ! (PX PX/S S)
            W(4)=RI(2)*X(2)                   ! (PY S/S S)
            W(5)=RI(3)*XX21+RI(4)*YYZZ21      ! (PY PX/S S)
            W(6)=RI(3)*XX22+RI(4)*YYZZ22      ! (PY PY/S S)
            W(7)=RI(2)*X(3)                   ! (PZ S/SS)
            W(8)=RI(3)*XX31+RI(4)*ZZ31        ! (PZ PX/S S)
            W(9)=RI(3)*XX32+RI(4)*ZZ32        ! (PZ PY/S S)
            W(10)=RI(3)*XX33+RI(4)*ZZ33       ! (PZ PZ/S S)
            KI = 10
         END IF
      END IF

! *** NOW ROTATE THE NUCLEAR ATTRACTION INTEGRALS.
! *** THE STORAGE OF THE NUCLEAR ATTRACTION INTEGRALS
      CSS1   = qm2_params%core_chg(jqm)*RI(1)
      CSS2   = qm2_params%core_chg(iqm)*RI(1) 
      E1B(1) =-CSS1
      E2A(1) =-CSS2
      IF(qm2_params%natomic_orbs(iqm) .EQ. 4) THEN
         CSP1   = qm2_params%core_chg(jqm)*RI(2)
         CPPS1  = qm2_params%core_chg(jqm)*RI(3)
         CPPP1  = qm2_params%core_chg(jqm)*RI(4)
         E1B(2) =-CSP1 *X(1)
         E1B(3) =-CPPS1*XX11-CPPP1*YYZZ11
         E1B(4) =-CSP1 *X(2)
         E1B(5) =-CPPS1*XX21-CPPP1*YYZZ21
         E1B(6) =-CPPS1*XX22-CPPP1*YYZZ22
         E1B(7) =-CSP1 *X(3)
         E1B(8) =-CPPS1*XX31-CPPP1*ZZ31
         E1B(9) =-CPPS1*XX32-CPPP1*ZZ32
         E1B(10)=-CPPS1*XX33-CPPP1*ZZ33
      END IF
      IF(qm2_params%natomic_orbs(jqm) .EQ. 4) THEN
         CSP2   = qm2_params%core_chg(iqm)*RI(5)
         CPPS2  = qm2_params%core_chg(iqm)*RI(11)
         CPPP2  = qm2_params%core_chg(iqm)*RI(12)
         E2A(2) =-CSP2 *X(1)
         E2A(3) =-CPPS2*XX11-CPPP2*YYZZ11
         E2A(4) =-CSP2 *X(2)
         E2A(5) =-CPPS2*XX21-CPPP2*YYZZ21
         E2A(6) =-CPPS2*XX22-CPPP2*YYZZ22
         E2A(7) =-CSP2 *X(3)
         E2A(8) =-CPPS2*XX31-CPPP2*ZZ31
         E2A(9) =-CPPS2*XX32-CPPP2*ZZ32
         E2A(10)=-CPPS2*XX33-CPPP2*ZZ33
      END IF

! SCALE = EXP(-ALP(NI)*RIJ)+EXP(-ALP(NJ)*RIJ)
      SCALE = EXP1i+EXP1j

      if (ni .EQ. 1 .AND. (nj .EQ. 7 .OR. nj .EQ. 8)) then
         SCALE = SCALE+(RIJ-one)*EXP1j
      else if ((ni .EQ. 7 .OR. ni .EQ. 8) .AND. nj .EQ. 1) then
         SCALE = SCALE+(RIJ-one)*EXP1i
      end if
      C1   = qm2_params%core_chg(IQM)*qm2_params%core_chg(JQM)      
      ENUC = C1*RI(1)
      SCALE= ABS(SCALE*ENUC)

      IF (AM1_OR_PM3) THEN
! Add gaussians.
         anam1 = zero
         do I=1,qm2_params%num_fn(qmitype)                                                      
            temp_real = RIJ-qm2_params%FN3(i,qmitype)
            temp_real2= qm2_params%FN2(i,qmitype)*temp_real*temp_real

! Skip doing the exponential if it is essentially zer
            if (temp_real2 .LT. EXPONENTIAL_CUTOFF) then
              anam1 = anam1+qm2_params%FN1(i,qmitype)*EXP(-temp_real2)
            end if
         end do
         do i=1,qm2_params%num_fn(qmjtype)
            temp_real = RIJ-qm2_params%FN3(i,qmjtype)
            temp_real2= qm2_params%FN2(i,qmjtype)*temp_real*temp_real

! Skip doing the exponential if it is essentially zer
            if (temp_real2 .LT. EXPONENTIAL_CUTOFF) then
              anam1 = anam1+qm2_params%FN1(i,qmjtype)*EXP(-temp_real2)
            end if
         end do
         anam1 = anam1*c1*oneRIJ
         scale = scale + anam1 
      ENDIF

! PDDG Specific terms
      if (PDDG_IN_USE) then
         PDDG_CORR = qm2_params%PDDG_TERM1(qmitype,qmjtype)*PDDG_EXP1   &
                    +qm2_params%PDDG_TERM2(qmitype,qmjtype)*PDDG_EXP2   &
                    +qm2_params%PDDG_TERM3(qmitype,qmjtype)*PDDG_EXP3   &
                    +qm2_params%PDDG_TERM4(qmitype,qmjtype)*PDDG_EXP4
         SCALE = SCALE + PDDG_CORR
      end if

      ENUC = ENUC+SCALE
      KR   = KR  +KI

      return
      END SUBROUTINE qm2_rotate_qmqm


      SUBROUTINE qm2_dihed(XYZ,I,J,K,L,ANGLE)
! 
! DIHED calculates the dihedral angle between atoms i, j, k,
!       and l. The cartesian coordinates of these atopms are
!       in array xyz.
! DIHED is a modified version of a surboutine of the same
!       name which was writted by DR. W. Thiel in 1973.
!
  use chm_kinds
  use qm2_double
  use qm2_constants
      implicit none
!Passed in
      real(chm_real),  intent(in)  :: XYZ(3,*)
      integer, intent(in)  :: i,j,k,l
      real(chm_real),  intent(out) :: angle

!Local variables
      real(chm_real)  :: xi1, xj1, xl1, yi1, yj1, yl1, zi1, zj1, zl1
      real(chm_real)  :: xi2, xl2, yi2, yl2, costh, sinth, cosph, sinph
      real(chm_real)  :: yj2, yi3, yl3
      real(chm_real)  :: dist, cosa, ddd, YXDIST

      XI1=XYZ(1,I)-XYZ(1,K)
      XJ1=XYZ(1,J)-XYZ(1,K)
      XL1=XYZ(1,L)-XYZ(1,K)
      YI1=XYZ(2,I)-XYZ(2,K)
      YJ1=XYZ(2,J)-XYZ(2,K)
      YL1=XYZ(2,L)-XYZ(2,K)
      ZI1=XYZ(3,I)-XYZ(3,K)
      ZJ1=XYZ(3,J)-XYZ(3,K)
      ZL1=XYZ(3,L)-XYZ(3,K)

! Rotate around Z axis to put Kj along Y axis
      DIST  = one/SQRT(XJ1**2+YJ1**2+ZJ1**2)
      COSA  = ZJ1*DIST
      IF (COSA.GT.one)  COSA = one
      IF (COSA.LT.-one) COSA =-one

      DDD   = one-COSA**2
      IF (DDD.LE.zero) GO TO 10

      YXDIST= DIST* (one/SQRT(DDD))
      IF (YXDIST.LT.1.0D6) GO TO 20

   10 CONTINUE
      XI2   = XI1
      XL2   = XL1
      YI2   = YI1
      YL2   = YL1
      COSTH = COSA
      SINTH = zero
      GO TO 30

   20 COSPH = YJ1*YXDIST
      SINPH = XJ1*YXDIST
      XI2   = XI1*COSPH-YI1*SINPH
      XL2   = XL1*COSPH-YL1*SINPH
      YI2   = XI1*SINPH+YI1*COSPH
      YJ2   = XJ1*SINPH+YJ1*COSPH
      YL2   = XL1*SINPH+YL1*COSPH

! Rotate Kj around the X axis so Kj lies along the Z axis
      COSTH = COSA
      SINTH = YJ2*DIST
   30 CONTINUE

      YI3   = YI2*COSTH-ZI1*SINTH
      YL3   = YL2*COSTH-ZL1*SINTH

      CALL qm2_dang(XL2,YL3,XI2,YI3,ANGLE)

      IF (ANGLE .LT. zero)         ANGLE=four * ASIN(one) + ANGLE
      IF (ANGLE .GE. 6.2831853D0 ) ANGLE=zero

      return
      END SUBROUTINE qm2_dihed


      SUBROUTINE qm2_dang(A1,A2,B1,B2,RCOS)
!
! DANG determines the angle between the points (A1,A2),
!      (0,0), and (B1,B2). The results is put in RCOS.
!
  use chm_kinds
  use qm2_double
  use qm2_constants
      implicit none
!Passed in
      real(chm_real), intent(inout) :: a1,a2,b1,b2
      real(chm_real), intent(out)   :: rcos

!Local variables
      real(chm_real) :: rsmall_local, anorm, bnorm, sinth, costh
 
      rsmall_local = TEN_TO_MINUS6                     ! 1.0D-6

      IF( ABS(A1).LT.rsmall_local .AND. ABS(A2).LT.rsmall_local) GO TO 10
      IF( ABS(B1).LT.rsmall_local .AND. ABS(B2).LT.rsmall_local) GO TO 10

      ANORM= one / SQRT(A1**2+A2**2)
      BNORM= one / SQRT(B1**2+B2**2)
      A1   = A1*ANORM
      A2   = A2*ANORM
      B1   = B1*BNORM
      B2   = B2*BNORM
      SINTH= (A1*B2)-(A2*B1)
      COSTH= A1*B1+A2*B2
      IF (COSTH.GT. one) COSTH= one
      IF (COSTH.LT.-one) COSTH=-one

      RCOS = ACOS(COSTH)
      IF (ABS(RCOS).LT.4.0D-4) GO TO 10
      IF (SINTH.GT.zero)       RCOS=four * ASIN(one) - RCOS
      RCOS =-RCOS
      return

   10 RCOS = zero

      return
      END SUBROUTINE qm2_dang


      SUBROUTINE qm2_fock1(F, PTOT, indx_for_qm)
!
! Compute the remaining contributions to the one-center elements
!
! Current routine streamlined and optimised by 
! Ross Walker (TSRI, 2005)
!

  use qmmm_module, only : qmmm_struct, qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!Passed in
      real(chm_real), intent(inout) :: F(*)
      real(chm_real), intent(in)    :: PTOT(*)
      integer,  intent(in)    :: indx_for_qm
     
!Local variables
      integer :: ii,ia,ib,ka,l,m,j,iminus,iplus,icc
      real(chm_real):: ptpop, GSSII, GSPII, GPPII, GP2II, HSPII 
      real(chm_real):: GSPIIHSPII, GSPIIHSPIIPTK, HSPIIGSPII, GP2IIGPPII
      real(chm_real):: PTOTKA, PTOTM, PTOTL, PTPOPTL, GPPIIGP2II

      integer :: mstart,mstop
      integer :: mstart_keep(2),mstop_keep(2)
      integer :: ISTRT_CHECK                 ! for external function
      integer :: old_N(2) = 0
      save old_N,mstart_keep,mstop_keep

! for parallelization
      if(old_N(indx_for_qm).ne.qmmm_struct%nquant_nlink) then
         old_N(indx_for_qm) = qmmm_struct%nquant_nlink
         mstart_keep(indx_for_qm) = 1
         mstop_keep(indx_for_qm)  = qmmm_struct%nquant_nlink
#if KEY_PARALLEL==1
         if(numnod.gt.1)  &
           mstart_keep(indx_for_qm) =  &
           ISTRT_CHECK(mstop_keep(indx_for_qm),qmmm_struct%nquant_nlink)
      else
         if(QMPI) then
            mstart_keep(indx_for_qm) = 1
            mstop_keep(indx_for_qm)  = qmmm_struct%nquant_nlink
         end if
#endif 
      end if
      mstart = mstart_keep(indx_for_qm)
      mstop  = mstop_keep(indx_for_qm)
      
      do II=mstart,mstop                     ! 1,qmmm_struct%nquant_nlink
         GSSII  = qm2_params%onec2elec_params(1,II)
         IA     = qm2_params%orb_loc(1,II)
         KA     = qm2_params%pascal_tri2(IA)
         PTOTKA = PTOT(KA)

         if (qm2_params%natomic_orbs(ii) .EQ. 1) then
            F(KA) = F(KA)+PTOTKA*GSSII
         else
! P ortbitals
            GSPII         = qm2_params%onec2elec_params(2,II)
            GPPII         = qm2_params%onec2elec_params(3,II)
            GP2II         = qm2_params%onec2elec_params(4,II)
            HSPII         = qm2_params%onec2elec_params(5,II)
            GSPIIHSPII    = GSPII-HSPII
            GSPIIHSPIIPTK = GSPIIHSPII*PTOTKA
            HSPIIGSPII    = six*HSPII-GSPII
            GP2IIGPPII    = GP2II-half*GPPII
            GPPIIGP2II    = 1.5d0*GPPII-GP2II

            IB            = qm2_params%orb_loc(2,II)
            PTPOP         = PTOT(qm2_params%pascal_tri2(IB))+   &
                            PTOT(qm2_params%pascal_tri2(IB-1))+   &
                            PTOT(qm2_params%pascal_tri2(IB-2))
            F(KA)         = F(KA)+PTOTKA*GSSII+PTPOP*GSPIIHSPII

! F(S,S)
            IPLUS=IA+1
            L    =KA
            do J=IPLUS,IB
               M      = L+IA
               L      = L+J
               PTOTL  = PTOT(L)
               PTPOPTL= PTPOP-PTOTL
 
! F(P,P)
               F(L)   = F(L)+GSPIIHSPIIPTK + PTOTL*GPPII +   &
                        PTPOPTL*GP2IIGPPII

! F(S,P)
               PTOTM  = half*PTOT(M)
               F(M)   = F(M)+PTOTM*HSPIIGSPII
            end do

! F(P,P*)
            IMINUS    = IB-1
            do J=IPLUS,IMINUS
               ICC=J+1
               do L=ICC,IB
                  M    = qm2_params%pascal_tri1(L)+J
                  PTOTM= PTOT(M)
                  F(M) = F(M)+PTOTM*GPPIIGP2II
               end do
            end do                              
         end if
      end do

      return
      END SUBROUTINE qm2_fock1


      SUBROUTINE qm2_fock2(F, PTOT, W, NUMAT, orb_loc, indx_for_qm)
!
! FOCK2 FORMS THE TWO-ELECTRON TWO-CENTER REPULSION PART OF
! THE FOCK MATRIX
!
! On input
!    PTOT  : Total density matrix
!    W     : Two-electron integral matrix
!
! On output
!    F     : Partial fock matrix
!

  use qmmm_module, only : qmmm_struct, qm2_struct, qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!Passed in
      real(chm_real),  intent(inout) :: F(*)
      real(chm_real),  intent(in)    :: ptot(*)
      real(chm_real),  intent(in)    :: W(qm2_struct%n2el)
      integer, intent(in)      :: numat
      integer, intent(in)      :: orb_loc(2,numat)
      integer, intent(in)      :: indx_for_qm

!Local variables
      integer  :: JINDEX(256),JINDEX_keep(256,2)
      real(chm_real) :: PK(16),PJA(16),PJB(16)
      integer  :: m,i,j, ij, ji, k, l, kl, lk, kk, ii, ia, ib, jk, kj
      integer  :: jj, ja, jb, i1, ll, j1
      real(chm_real) :: sumdia, sumoff, sum, wkk

      integer  :: ncount

      logical  :: first_call(2)

!#ifdef MOPAC_VECTOR
!      integer  :: KINDEX(256),JJNDEX(256),LLPERM(10)
!      integer  :: mmperm(10),ijperm(10)
!      integer  :: ik,ki,jl,lj
!      save kindex,jjndex,llperm,mmperm,ijperm
!#endif
      data first_call /.true.,.true./
      save first_call, jindex_keep

      IF (first_call(indx_for_qm)) THEN
         first_call(indx_for_qm) = .false.

! Setup gather-scatter type arrays for use with two-electron
! integrals. 
! Jindex are the indices of the j-integrals for atom i integrals
! Jjndex are the indices of the j-integrals for atom j integrals
! Kindex are the indices of the k-integrals
         M = 0
         do I=1,4
            do J=1,4
               IJ = MIN(I,J)
               JI = I+J-IJ
               do K=1,4
!#ifdef MOPAC_VECTOR
!                  IK = MIN(I,K)
!                  KI = I+K-IK
!#endif
                  do L=1,4
                     M  = M+1
                     KL = MIN(K,L)
                     LK = K+L-KL
!#ifdef MOPAC_VECTOR                                                                               
!                     JL = MIN(J,L)
!                     LJ = J+L-JL
!                     KINDEX(M)= qm2_params%pascal_tri1(LJ) +JL +  
!    *                           10*(qm2_params%pascal_tri1(KI)+IK) - 10
!#endif
                     JINDEX_keep(M,indx_for_qm) &
                              =(qm2_params%pascal_tri1(JI)+IJ)*10 +   &
                                qm2_params%pascal_tri1(LK)+KL - 10
                  end do
               end do
            end do
         end do

!#ifdef MOPAC_VECTOR
!         L = 0
!         do I=1,4
!            I1 = (I-1)*4
!            do J=1,I
!               I1 = I1+1
!               L  = L+1
!               IJPERM(L)=I1
!               MMPERM(L)=IJPERM(L)-16
!               LLPERM(L)=(I1-1)*16
!            end do
!         end do
!
!         L = 0
!         do I=1,10
!            M=MMPERM(I)
!            L=LLPERM(I)
!            do K=1,16
!               L=L+1
!               M=M+16
!               JJNDEX(L)=JINDEX(M)
!            end do
!         end do
!#endif
      END IF

      JINDEX(1:256) = JINDEX_keep(1:256,indx_for_qm)

! moved from below.
      do II=1,NUMAT
         IA=orb_loc(1,II)
         IB=orb_loc(2,II)
         M =0
         do J=IA,IB
            do K=IA,IB
               M =M+1
               JK=MIN(J,K)
               KJ=K+J-JK
               JK=JK+qm2_params%pascal_tri1(KJ)
               qm2_struct%fock2_PTOT2(II,M)=PTOT(JK)
            end do
         end do
      end do

      KK    =0
      ncount=0
      do II=1,NUMAT
         IA=orb_loc(1,II)
         IB=orb_loc(2,II)

! moved to above.
!        M =0
!        do J=IA,IB
!           do K=IA,IB
!              M =M+1
!              JK=MIN(J,K)
!              KJ=K+J-JK
!              JK=JK+qm2_params%pascal_tri1(KJ)
!              qm2_struct%fock2_PTOT2(II,M)=PTOT(JK)
!           end do
!        end do

         do JJ=1,(II-1)
            JA    = orb_loc(1,JJ)
            JB    = orb_loc(2,JJ)

            IF (IB .NE. IA .AND. JA .NE. JB) THEN
#if KEY_PARALLEL==1
               ! note: ncount is updated at the end of loop.     
#endif
#if KEY_PARALLEL==1
               if(mynod .eq. MOD(ncount,numnod)) then            
#endif

! Heavy atom - Heavy atom
! Extract coulomb terms
               do I=1,16
                  PJA(I)=qm2_struct%fock2_PTOT2(II,I)
                  PJB(I)=qm2_struct%fock2_PTOT2(JJ,I)
               end do

! Coulomb terms:

!#ifdef MOPAC_VECTOR
!               CALL qm2_jab(IA,JA,LLPERM,JINDEX,JJNDEX,PJA,PJB,  
!    *                       W(KK+1),F)
!#else
               CALL qm2_jab(IA,JA,PJA,PJB,W(KK+1),F)
!#endif

! Exchange terms: extract intersection of atoms II and JJ
!                 in the spin density matrix
               L = 0
               do I=IA,IB
                  I1=qm2_params%pascal_tri1(I)+JA
                  do J=I1,I1+3
                     L    =L+1
                     PK(L)=half*PTOT(J)
                  end do
               end do

!#ifdef MOPAC_VECTOR
!               CALL qm2_kab(IA,JA, PK, W(KK+1), KINDEX, F)
!#else
               CALL qm2_kab(IA,JA, PK, W(KK+1), F)
!#endif

#if KEY_PARALLEL==1
               end if           
#endif

               KK=KK+100

            ELSE IF (IA .NE. IB) THEN
#if KEY_PARALLEL==1
               ! note: ncount is updated at the end of loop.      
#endif
#if KEY_PARALLEL==1
               if(mynod .eq. MOD(ncount,numnod)) then             
#endif
! Light atom - Heavy atom
! Coulomb terms
               SUMDIA=zero
               SUMOFF=zero
               LL    =qm2_params%pascal_tri2(JA)
               K     =0
               do I=0,3
                  J1=qm2_params%pascal_tri1(IA+I)+IA-1
                  do J=0,I-1
                     K     =K+1
                     J1    =J1+1
                     WKK   =W(KK+K)
                     F(J1) =F(J1)+PTOT(LL)*WKK
                     SUMOFF=SUMOFF+PTOT(J1)*WKK
                  end do

                  J1    =J1+1
                  K     =K+1
                  WKK   =W(KK+K)
                  F(J1) =F(J1)+PTOT(LL)*WKK
                  SUMDIA=SUMDIA+PTOT(J1)*WKK
               end do

               F(LL)=F(LL)+two*SUMOFF+SUMDIA

! Exchange terms: extract intersetion of ATOMS II and JJ
!                 in the spin density matrix
               K = 0
               do I=IA,IB
                  I1 =qm2_params%pascal_tri1(I)+JA
                  SUM=zero
                  do J=IA,IB
                     K  =K+1
                     J1 =qm2_params%pascal_tri1(J)+JA
                     SUM=SUM+half*PTOT(J1)*W(KK+JINDEX(K))
                  end do
                  F(I1)=F(I1)-SUM
               end do

#if KEY_PARALLEL==1
               end if         
#endif

               KK=KK+10

            ELSE IF (JA .NE. JB) THEN
#if KEY_PARALLEL==1
               ! note: ncount is updated at the end of loop.     
#endif
#if KEY_PARALLEL==1
               if(mynod .eq. MOD(ncount,numnod)) then            
#endif
! Heavy atom - Light atom
! Coulomb terms
               SUMDIA=zero
               SUMOFF=zero
               LL    =qm2_params%pascal_tri2(IA)
               K     =0
               do I=0,3
                  J1=qm2_params%pascal_tri1(JA+I)+JA-1
                  do J=0,I-1
                     K     =K+1
                     J1    =J1+1
                     WKK   =W(KK+K)
                     F(J1) =F(J1)+PTOT(LL)*WKK
                     SUMOFF=SUMOFF+PTOT(J1)*WKK
                  end do

                  J1    =J1+1
                  K     =K+1
                  WKK   =W(KK+K)
                  F(J1) =F(J1)+PTOT(LL)*WKK
                  SUMDIA=SUMDIA+PTOT(J1)*WKK
               end do

               F(LL)=F(LL)+two*SUMOFF+SUMDIA

! Exchange terms: extract intersection of atoms II and JJ
!                 in the spin density matrix
               K=qm2_params%pascal_tri1(IA)+JA
               J=0
               do I=K,K+3
                  SUM=zero
                  do L=K,K+3
                     J  =J+1
                     SUM=SUM+half*PTOT(L)*W(KK+JINDEX(J))
                  end do
                  F(I)=F(I)-SUM
               end do

#if KEY_PARALLEL==1
               end if           
#endif

               KK=KK+10

            ELSE
#if KEY_PARALLEL==1
               ! note: ncount is updated at the end of loop.  
#endif
#if KEY_PARALLEL==1
               if(mynod .eq. MOD(ncount,numnod)) then         
#endif
! Light atom - Light atom
               I1   =qm2_params%pascal_tri2(IA)
               J1   =qm2_params%pascal_tri2(JA)
               IJ   =I1+JA-IA
               WKK  =W(KK+1)
               F(I1)=F(I1)+PTOT(J1)*WKK
               F(J1)=F(J1)+PTOT(I1)*WKK
               F(IJ)=F(IJ)-half*PTOT(IJ)*WKK

#if KEY_PARALLEL==1
               end if             
#endif

               KK   =KK+1
            END IF

#if KEY_PARALLEL==1
            ! update ncount               
#endif
#if KEY_PARALLEL==1
            ncount= ncount + 1            
#endif
         end do                                  ! JJ=1,(II-1)
      end do                                     ! II=1,NUMAT

      return
      END SUBROUTINE qm2_fock2


!#ifdef MOPAC_VECTOR
!     SUBROUTINE qm2_jab(IA,JA,LLPERM,JINDEX,JJNDEX,PJA,PJB,W, F)
!#else
      SUBROUTINE qm2_jab(IA,JA,PJA,PJB,W, F)
!#endif

  use qmmm_module, only : qm2_params
  use chm_kinds
  use qm2_double
      implicit none
!Passed in
      real(chm_real), intent(in)    :: PJA(16), PJB(16), W(100)
      real(chm_real), intent(inout) :: F(*)

!Local
      integer :: i,i5,i6,iia,ija,ia,ja,ioff,joff

!#ifdef MOPAC_VECTOR
!      integer, intent(in) :: LLPERM(10),JINDEX(256), JJNDEX(256)
!      real(chm_real)              :: suma,sumb
!      integer             :: l,k
!#else
      real(chm_real)              :: suma(10), sumb(10)
!#endif

!#ifdef MOPAC_VECTOR
!      ! May be faster on vector machines.
!      I = 0
!      DO I5=1,4
!         IIA =IA+I5-1
!         IJA =JA+I5-1
!         IOFF=qm2_params%pascal_tri1(iia)+IA-1
!         JOFF=qm2_params%pascal_tri1(ija)+JA-1
!         DO I6=1,I5
!            IOFF=IOFF+1
!            JOFF=JOFF+1
!            I   =I+1
!            L   =LLPERM(I)
!            SUMA=0
!            SUMB=0
!            DO K=1,16
!               L   =L+1
!               SUMB=SUMB+PJA(K)*W(JJNDEX(L))
!               SUMA=SUMA+PJB(K)*W(JINDEX(L))
!            END DO
!            F(IOFF)=F(IOFF)+SUMA
!            F(JOFF)=F(JOFF)+SUMB
!         END DO
!      END DO
!#else
      SUMA( 1)=                                                      &
       +PJA( 1)*W(  1)+PJA( 2)*W( 11)+PJA( 3)*W( 31)+PJA( 4)*W( 61)   &
       +PJA( 5)*W( 11)+PJA( 6)*W( 21)+PJA( 7)*W( 41)+PJA( 8)*W( 71)   &
       +PJA( 9)*W( 31)+PJA(10)*W( 41)+PJA(11)*W( 51)+PJA(12)*W( 81)   &
       +PJA(13)*W( 61)+PJA(14)*W( 71)+PJA(15)*W( 81)+PJA(16)*W( 91)
      SUMA( 2)=                                                      &
       +PJA( 1)*W(  2)+PJA( 2)*W( 12)+PJA( 3)*W( 32)+PJA( 4)*W( 62)   &
       +PJA( 5)*W( 12)+PJA( 6)*W( 22)+PJA( 7)*W( 42)+PJA( 8)*W( 72)   &
       +PJA( 9)*W( 32)+PJA(10)*W( 42)+PJA(11)*W( 52)+PJA(12)*W( 82)   &
       +PJA(13)*W( 62)+PJA(14)*W( 72)+PJA(15)*W( 82)+PJA(16)*W( 92) 
      SUMA( 3)=                                                      &
       +PJA( 1)*W(  3)+PJA( 2)*W( 13)+PJA( 3)*W( 33)+PJA( 4)*W( 63)   &
       +PJA( 5)*W( 13)+PJA( 6)*W( 23)+PJA( 7)*W( 43)+PJA( 8)*W( 73)   &
       +PJA( 9)*W( 33)+PJA(10)*W( 43)+PJA(11)*W( 53)+PJA(12)*W( 83)   &
       +PJA(13)*W( 63)+PJA(14)*W( 73)+PJA(15)*W( 83)+PJA(16)*W( 93)
      SUMA( 4)=                                                      &
       +PJA( 1)*W(  4)+PJA( 2)*W( 14)+PJA( 3)*W( 34)+PJA( 4)*W( 64)   &
       +PJA( 5)*W( 14)+PJA( 6)*W( 24)+PJA( 7)*W( 44)+PJA( 8)*W( 74)   &
       +PJA( 9)*W( 34)+PJA(10)*W( 44)+PJA(11)*W( 54)+PJA(12)*W( 84)   &
       +PJA(13)*W( 64)+PJA(14)*W( 74)+PJA(15)*W( 84)+PJA(16)*W( 94)
      SUMA( 5)=                                                      &
       +PJA( 1)*W(  5)+PJA( 2)*W( 15)+PJA( 3)*W( 35)+PJA( 4)*W( 65)   &
       +PJA( 5)*W( 15)+PJA( 6)*W( 25)+PJA( 7)*W( 45)+PJA( 8)*W( 75)    &
       +PJA( 9)*W( 35)+PJA(10)*W( 45)+PJA(11)*W( 55)+PJA(12)*W( 85)   &
       +PJA(13)*W( 65)+PJA(14)*W( 75)+PJA(15)*W( 85)+PJA(16)*W( 95)
      SUMA( 6)=                                                      &
       +PJA( 1)*W(  6)+PJA( 2)*W( 16)+PJA( 3)*W( 36)+PJA( 4)*W( 66)   &
       +PJA( 5)*W( 16)+PJA( 6)*W( 26)+PJA( 7)*W( 46)+PJA( 8)*W( 76)   &
       +PJA( 9)*W( 36)+PJA(10)*W( 46)+PJA(11)*W( 56)+PJA(12)*W( 86)   &
       +PJA(13)*W( 66)+PJA(14)*W( 76)+PJA(15)*W( 86)+PJA(16)*W( 96)
      SUMA( 7)=                                                      &
       +PJA( 1)*W(  7)+PJA( 2)*W( 17)+PJA( 3)*W( 37)+PJA( 4)*W( 67)   &
       +PJA( 5)*W( 17)+PJA( 6)*W( 27)+PJA( 7)*W( 47)+PJA( 8)*W( 77)   &
       +PJA( 9)*W( 37)+PJA(10)*W( 47)+PJA(11)*W( 57)+PJA(12)*W( 87)   &
       +PJA(13)*W( 67)+PJA(14)*W( 77)+PJA(15)*W( 87)+PJA(16)*W( 97)
      SUMA( 8)=                                                      &
       +PJA( 1)*W(  8)+PJA( 2)*W( 18)+PJA( 3)*W( 38)+PJA( 4)*W( 68)   &
       +PJA( 5)*W( 18)+PJA( 6)*W( 28)+PJA( 7)*W( 48)+PJA( 8)*W( 78)   &
       +PJA( 9)*W( 38)+PJA(10)*W( 48)+PJA(11)*W( 58)+PJA(12)*W( 88)   &
       +PJA(13)*W( 68)+PJA(14)*W( 78)+PJA(15)*W( 88)+PJA(16)*W( 98)
      SUMA( 9)=                                                      &
       +PJA( 1)*W(  9)+PJA( 2)*W( 19)+PJA( 3)*W( 39)+PJA( 4)*W( 69)   &
       +PJA( 5)*W( 19)+PJA( 6)*W( 29)+PJA( 7)*W( 49)+PJA( 8)*W( 79)   &
       +PJA( 9)*W( 39)+PJA(10)*W( 49)+PJA(11)*W( 59)+PJA(12)*W( 89)   &
       +PJA(13)*W( 69)+PJA(14)*W( 79)+PJA(15)*W( 89)+PJA(16)*W( 99)
      SUMA(10)=                                                      &
       +PJA( 1)*W( 10)+PJA( 2)*W( 20)+PJA( 3)*W( 40)+PJA( 4)*W( 70)   &
       +PJA( 5)*W( 20)+PJA( 6)*W( 30)+PJA( 7)*W( 50)+PJA( 8)*W( 80)   &
       +PJA( 9)*W( 40)+PJA(10)*W( 50)+PJA(11)*W( 60)+PJA(12)*W( 90)   &
       +PJA(13)*W( 70)+PJA(14)*W( 80)+PJA(15)*W( 90)+PJA(16)*W(100)
      SUMB( 1)=                                                      &
       +PJB( 1)*W(  1)+PJB( 2)*W(  2)+PJB( 3)*W(  4)+PJB( 4)*W(  7)   &
       +PJB( 5)*W(  2)+PJB( 6)*W(  3)+PJB( 7)*W(  5)+PJB( 8)*W(  8)   &
       +PJB( 9)*W(  4)+PJB(10)*W(  5)+PJB(11)*W(  6)+PJB(12)*W(  9)   &
       +PJB(13)*W(  7)+PJB(14)*W(  8)+PJB(15)*W(  9)+PJB(16)*W( 10)
      SUMB( 2)=                                                      &
       +PJB( 1)*W( 11)+PJB( 2)*W( 12)+PJB( 3)*W( 14)+PJB( 4)*W( 17)   &
       +PJB( 5)*W( 12)+PJB( 6)*W( 13)+PJB( 7)*W( 15)+PJB( 8)*W( 18)   &
       +PJB( 9)*W( 14)+PJB(10)*W( 15)+PJB(11)*W( 16)+PJB(12)*W( 19)   &
       +PJB(13)*W( 17)+PJB(14)*W( 18)+PJB(15)*W( 19)+PJB(16)*W( 20)
      SUMB( 3)=                                                      &
       +PJB( 1)*W( 21)+PJB( 2)*W( 22)+PJB( 3)*W( 24)+PJB( 4)*W( 27)   &
       +PJB( 5)*W( 22)+PJB( 6)*W( 23)+PJB( 7)*W( 25)+PJB( 8)*W( 28)   &
       +PJB( 9)*W( 24)+PJB(10)*W( 25)+PJB(11)*W( 26)+PJB(12)*W( 29)   &
       +PJB(13)*W( 27)+PJB(14)*W( 28)+PJB(15)*W( 29)+PJB(16)*W( 30)
      SUMB( 4)=                                                      &
       +PJB( 1)*W( 31)+PJB( 2)*W( 32)+PJB( 3)*W( 34)+PJB( 4)*W( 37)   &
       +PJB( 5)*W( 32)+PJB( 6)*W( 33)+PJB( 7)*W( 35)+PJB( 8)*W( 38)   &
       +PJB( 9)*W( 34)+PJB(10)*W( 35)+PJB(11)*W( 36)+PJB(12)*W( 39)   &
       +PJB(13)*W( 37)+PJB(14)*W( 38)+PJB(15)*W( 39)+PJB(16)*W( 40) 
      SUMB( 5)=                                                      &
       +PJB( 1)*W( 41)+PJB( 2)*W( 42)+PJB( 3)*W( 44)+PJB( 4)*W( 47)    &
       +PJB( 5)*W( 42)+PJB( 6)*W( 43)+PJB( 7)*W( 45)+PJB( 8)*W( 48)   &
       +PJB( 9)*W( 44)+PJB(10)*W( 45)+PJB(11)*W( 46)+PJB(12)*W( 49)   &
       +PJB(13)*W( 47)+PJB(14)*W( 48)+PJB(15)*W( 49)+PJB(16)*W( 50) 
      SUMB( 6)=                                                      &
       +PJB( 1)*W( 51)+PJB( 2)*W( 52)+PJB( 3)*W( 54)+PJB( 4)*W( 57)   &
       +PJB( 5)*W( 52)+PJB( 6)*W( 53)+PJB( 7)*W( 55)+PJB( 8)*W( 58)   &
       +PJB( 9)*W( 54)+PJB(10)*W( 55)+PJB(11)*W( 56)+PJB(12)*W( 59)   &
       +PJB(13)*W( 57)+PJB(14)*W( 58)+PJB(15)*W( 59)+PJB(16)*W( 60)
      SUMB( 7)=                                                      &
       +PJB( 1)*W( 61)+PJB( 2)*W( 62)+PJB( 3)*W( 64)+PJB( 4)*W( 67)   &
       +PJB( 5)*W( 62)+PJB( 6)*W( 63)+PJB( 7)*W( 65)+PJB( 8)*W( 68)   &
       +PJB( 9)*W( 64)+PJB(10)*W( 65)+PJB(11)*W( 66)+PJB(12)*W( 69)   &
       +PJB(13)*W( 67)+PJB(14)*W( 68)+PJB(15)*W( 69)+PJB(16)*W( 70)
      SUMB( 8)=                                                      &
       +PJB( 1)*W( 71)+PJB( 2)*W( 72)+PJB( 3)*W( 74)+PJB( 4)*W( 77)    &
       +PJB( 5)*W( 72)+PJB( 6)*W( 73)+PJB( 7)*W( 75)+PJB( 8)*W( 78)   &
       +PJB( 9)*W( 74)+PJB(10)*W( 75)+PJB(11)*W( 76)+PJB(12)*W( 79)   &
       +PJB(13)*W( 77)+PJB(14)*W( 78)+PJB(15)*W( 79)+PJB(16)*W( 80)
      SUMB( 9)=                                                      &
       +PJB( 1)*W( 81)+PJB( 2)*W( 82)+PJB( 3)*W( 84)+PJB( 4)*W( 87)   &
       +PJB( 5)*W( 82)+PJB( 6)*W( 83)+PJB( 7)*W( 85)+PJB( 8)*W( 88)   &
       +PJB( 9)*W( 84)+PJB(10)*W( 85)+PJB(11)*W( 86)+PJB(12)*W( 89)   &
       +PJB(13)*W( 87)+PJB(14)*W( 88)+PJB(15)*W( 89)+PJB(16)*W( 90)
      SUMB(10)=                                                      &
       +PJB( 1)*W( 91)+PJB( 2)*W( 92)+PJB( 3)*W( 94)+PJB( 4)*W( 97)   &
       +PJB( 5)*W( 92)+PJB( 6)*W( 93)+PJB( 7)*W( 95)+PJB( 8)*W( 98)   &
       +PJB( 9)*W( 94)+PJB(10)*W( 95)+PJB(11)*W( 96)+PJB(12)*W( 99)   &
       +PJB(13)*W( 97)+PJB(14)*W( 98)+PJB(15)*W( 99)+PJB(16)*W(100)
      I=0
      DO I5=1,4
         IIA =IA+I5-1
         IJA =JA+I5-1
         IOFF=qm2_params%pascal_tri1(IIA)+IA-1
         JOFF=qm2_params%pascal_tri1(IJA)+JA-1
         DO I6=1,I5
            IOFF   =IOFF+1
            JOFF   =JOFF+1
            I      =I+1
            F(IOFF)=F(IOFF)+SUMB(I)
            F(JOFF)=F(JOFF)+SUMA(I)
         end do
      end do
!#endif
      return
      END SUBROUTINE qm2_jab
                  
      
      SUBROUTINE qm2_kab(IA,JA, PK, W, F)

  use qmmm_module, only : qm2_params
  use chm_kinds
  use qm2_double
      implicit none
!Passed in
      real(chm_real), intent(in)    :: PK(*), W(100)
      real(chm_real), intent(inout) :: F(*)
     
!Local variables
      integer  :: m,ia,j,ja,j1,j2,j3

!#ifdef MOPAC_VECTOR
!      integer, intent(in) :: KINDEX(256)
!      real(chm_real)              :: sum
!      integer             :: l,i
!#else
      real(chm_real)              :: SUM(16)
!#endif

!#ifdef MOPAC_VECTOR
!      ! may be faster on vector machines.
!      L=0
!      M=0
!      DO J1=IA,IA+3
!         J=qm2_params%pascal_tri1(j1)
!         DO J2=JA,JA+3
!            M=M+1
!            IF(IA.GT.JA)THEN
!               J3=J+J2
!            ELSE
!               J3=J1+qm2_params%pascal_tri1(j2)
!            ENDIF
!            SUM=0
!            DO I=1,16
!               L=L+1
!               SUM=SUM+PK(I)*W(KINDEX(L))
!            END DO
!            F(J3)=F(J3)-SUM
!         END DO
!      END DO
!#else
      SUM( 1)=                                                   &
       +PK( 1)*W(  1)+PK( 2)*W(  2)+PK( 3)*W(  4)+PK( 4)*W(  7)   &
       +PK( 5)*W( 11)+PK( 6)*W( 12)+PK( 7)*W( 14)+PK( 8)*W( 17)   &
       +PK( 9)*W( 31)+PK(10)*W( 32)+PK(11)*W( 34)+PK(12)*W( 37)   &
       +PK(13)*W( 61)+PK(14)*W( 62)+PK(15)*W( 64)+PK(16)*W( 67)
      SUM( 2)=                                                   &
       +PK( 1)*W(  2)+PK( 2)*W(  3)+PK( 3)*W(  5)+PK( 4)*W(  8)   &
       +PK( 5)*W( 12)+PK( 6)*W( 13)+PK( 7)*W( 15)+PK( 8)*W( 18)   &
       +PK( 9)*W( 32)+PK(10)*W( 33)+PK(11)*W( 35)+PK(12)*W( 38)   &
       +PK(13)*W( 62)+PK(14)*W( 63)+PK(15)*W( 65)+PK(16)*W( 68) 
      SUM( 3)=                                                   &
       +PK( 1)*W(  4)+PK( 2)*W(  5)+PK( 3)*W(  6)+PK( 4)*W(  9)   &
       +PK( 5)*W( 14)+PK( 6)*W( 15)+PK( 7)*W( 16)+PK( 8)*W( 19)   &
       +PK( 9)*W( 34)+PK(10)*W( 35)+PK(11)*W( 36)+PK(12)*W( 39)    &
       +PK(13)*W( 64)+PK(14)*W( 65)+PK(15)*W( 66)+PK(16)*W( 69)
      SUM( 4)=                                                   &
       +PK( 1)*W(  7)+PK( 2)*W(  8)+PK( 3)*W(  9)+PK( 4)*W( 10)   &
       +PK( 5)*W( 17)+PK( 6)*W( 18)+PK( 7)*W( 19)+PK( 8)*W( 20)   &
       +PK( 9)*W( 37)+PK(10)*W( 38)+PK(11)*W( 39)+PK(12)*W( 40)   &
       +PK(13)*W( 67)+PK(14)*W( 68)+PK(15)*W( 69)+PK(16)*W( 70)
      SUM( 5)=                                                   &
       +PK( 1)*W( 11)+PK( 2)*W( 12)+PK( 3)*W( 14)+PK( 4)*W( 17)   &
       +PK( 5)*W( 21)+PK( 6)*W( 22)+PK( 7)*W( 24)+PK( 8)*W( 27)   &
       +PK( 9)*W( 41)+PK(10)*W( 42)+PK(11)*W( 44)+PK(12)*W( 47)   &
       +PK(13)*W( 71)+PK(14)*W( 72)+PK(15)*W( 74)+PK(16)*W( 77)
      SUM( 6)=                                                   &
       +PK( 1)*W( 12)+PK( 2)*W( 13)+PK( 3)*W( 15)+PK( 4)*W( 18)   &
       +PK( 5)*W( 22)+PK( 6)*W( 23)+PK( 7)*W( 25)+PK( 8)*W( 28)   &
       +PK( 9)*W( 42)+PK(10)*W( 43)+PK(11)*W( 45)+PK(12)*W( 48)   &
       +PK(13)*W( 72)+PK(14)*W( 73)+PK(15)*W( 75)+PK(16)*W( 78)
      SUM( 7)=                                                   &
       +PK( 1)*W( 14)+PK( 2)*W( 15)+PK( 3)*W( 16)+PK( 4)*W( 19)   &
       +PK( 5)*W( 24)+PK( 6)*W( 25)+PK( 7)*W( 26)+PK( 8)*W( 29)   &
       +PK( 9)*W( 44)+PK(10)*W( 45)+PK(11)*W( 46)+PK(12)*W( 49)   &
       +PK(13)*W( 74)+PK(14)*W( 75)+PK(15)*W( 76)+PK(16)*W( 79)
      SUM( 8)=                                                   &
       +PK( 1)*W( 17)+PK( 2)*W( 18)+PK( 3)*W( 19)+PK( 4)*W( 20)   &
       +PK( 5)*W( 27)+PK( 6)*W( 28)+PK( 7)*W( 29)+PK( 8)*W( 30)   &
       +PK( 9)*W( 47)+PK(10)*W( 48)+PK(11)*W( 49)+PK(12)*W( 50)   &
       +PK(13)*W( 77)+PK(14)*W( 78)+PK(15)*W( 79)+PK(16)*W( 80)
      SUM( 9)=                                                   &
       +PK( 1)*W( 31)+PK( 2)*W( 32)+PK( 3)*W( 34)+PK( 4)*W( 37)   &
       +PK( 5)*W( 41)+PK( 6)*W( 42)+PK( 7)*W( 44)+PK( 8)*W( 47)   &
       +PK( 9)*W( 51)+PK(10)*W( 52)+PK(11)*W( 54)+PK(12)*W( 57)   &
       +PK(13)*W( 81)+PK(14)*W( 82)+PK(15)*W( 84)+PK(16)*W( 87)
      SUM(10)=                                                   &
       +PK( 1)*W( 32)+PK( 2)*W( 33)+PK( 3)*W( 35)+PK( 4)*W( 38)   &
       +PK( 5)*W( 42)+PK( 6)*W( 43)+PK( 7)*W( 45)+PK( 8)*W( 48)   &
       +PK( 9)*W( 52)+PK(10)*W( 53)+PK(11)*W( 55)+PK(12)*W( 58)   &
       +PK(13)*W( 82)+PK(14)*W( 83)+PK(15)*W( 85)+PK(16)*W( 88)
      SUM(11)=                                                   &
       +PK( 1)*W( 34)+PK( 2)*W( 35)+PK( 3)*W( 36)+PK( 4)*W( 39)   &
       +PK( 5)*W( 44)+PK( 6)*W( 45)+PK( 7)*W( 46)+PK( 8)*W( 49)   &
       +PK( 9)*W( 54)+PK(10)*W( 55)+PK(11)*W( 56)+PK(12)*W( 59)   &
       +PK(13)*W( 84)+PK(14)*W( 85)+PK(15)*W( 86)+PK(16)*W( 89)
      SUM(12)=                                                   &
       +PK( 1)*W( 37)+PK( 2)*W( 38)+PK( 3)*W( 39)+PK( 4)*W( 40)   &
       +PK( 5)*W( 47)+PK( 6)*W( 48)+PK( 7)*W( 49)+PK( 8)*W( 50)   &
       +PK( 9)*W( 57)+PK(10)*W( 58)+PK(11)*W( 59)+PK(12)*W( 60)   &
       +PK(13)*W( 87)+PK(14)*W( 88)+PK(15)*W( 89)+PK(16)*W( 90)
      SUM(13)=                                                   &
       +PK( 1)*W( 61)+PK( 2)*W( 62)+PK( 3)*W( 64)+PK( 4)*W( 67)   &
       +PK( 5)*W( 71)+PK( 6)*W( 72)+PK( 7)*W( 74)+PK( 8)*W( 77)   &
       +PK( 9)*W( 81)+PK(10)*W( 82)+PK(11)*W( 84)+PK(12)*W( 87)   &
       +PK(13)*W( 91)+PK(14)*W( 92)+PK(15)*W( 94)+PK(16)*W( 97)
      SUM(14)=                                                   &
       +PK( 1)*W( 62)+PK( 2)*W( 63)+PK( 3)*W( 65)+PK( 4)*W( 68)   &
       +PK( 5)*W( 72)+PK( 6)*W( 73)+PK( 7)*W( 75)+PK( 8)*W( 78)   &
       +PK( 9)*W( 82)+PK(10)*W( 83)+PK(11)*W( 85)+PK(12)*W( 88)   &
       +PK(13)*W( 92)+PK(14)*W( 93)+PK(15)*W( 95)+PK(16)*W( 98)
      SUM(15)=                                                   &
       +PK( 1)*W( 64)+PK( 2)*W( 65)+PK( 3)*W( 66)+PK( 4)*W( 69)   &
       +PK( 5)*W( 74)+PK( 6)*W( 75)+PK( 7)*W( 76)+PK( 8)*W( 79)   &
       +PK( 9)*W( 84)+PK(10)*W( 85)+PK(11)*W( 86)+PK(12)*W( 89)   &
       +PK(13)*W( 94)+PK(14)*W( 95)+PK(15)*W( 96)+PK(16)*W( 99)
      SUM(16)=                                                   &
       +PK( 1)*W( 67)+PK( 2)*W( 68)+PK( 3)*W( 69)+PK( 4)*W( 70)   &
       +PK( 5)*W( 77)+PK( 6)*W( 78)+PK( 7)*W( 79)+PK( 8)*W( 80)   &
       +PK( 9)*W( 87)+PK(10)*W( 88)+PK(11)*W( 89)+PK(12)*W( 90)   &
       +PK(13)*W( 97)+PK(14)*W( 98)+PK(15)*W( 99)+PK(16)*W(100)
      IF (IA.GT.JA)THEN
         M=0
         do J1=IA,IA+3
            J=qm2_params%pascal_tri1(j1)
            do J2=JA,JA+3
               M    =M+1
               J3   =J+J2
               F(J3)=F(J3)-SUM(M)
            end do
         end do
      ELSE
! IA is less than JA, therefore use other half of triangle
         M=0
         do J1=IA,IA+3
            do J2=JA,JA+3
               M    =M+1
               J3   =qm2_params%pascal_tri1(j2)+j1
               F(J3)=F(J3)-SUM(M)
            end do
         end do
      END IF
!#endif
      return
      END SUBROUTINE qm2_kab


      SUBROUTINE qm2_identify_peptide_links(n_peptide_links,coord)
!
! Identigy peptide linkages based on distance, and allocates
! necessary memory for storing the identities of the peptide
! linkages.
!
! Note: At some point, it would be better to use sander's bond
!       formation for this but for the moment this will suffice.
!
! Ross Walker (TSRI, 2005)
!

  use qmmm_module, only : qmmm_struct, qm2_struct
  use chm_kinds
  use qm2_double
      implicit none
!Passed in
      integer, intent(out) :: n_peptide_links
      real(chm_real),  intent(in) :: coord(3,qmmm_struct%nquant)

!Local variables
      real(chm_real),  pointer :: RXYZ(:)      ! Distance matrix used in finding
!                                        ! peptide linkages
      integer          :: l,i,j,k,jk,kj,ij,ji,kl,lk,m,mk,km
      integer          :: ier=0

! Distance matrix is currently only needed for calculating peptide
! linkages.
!
! Note: Ross Walker originally had a sqrt in this but I removed 
!       this for speed. Also, untimately it might be better to
!       use AMBERs bonding info for this.
      allocate (RXYZ(ishft(qmmm_struct%nquant_nlink*(   &
                           qmmm_struct%nquant_nlink+1),-1)),   &
                stat=ier)
      if(ier.ne.0) call Aass(1,'qm2_identify_peptide_links','RXYZ')

      L = 0
      do I=1,qmmm_struct%nquant_nlink
         do J=1,I
            L=L+1
            RXYZ(L)=((COORD(1,I)-COORD(1,J))**2 +   &
                     (COORD(2,I)-COORD(2,J))**2 +   &
                     (COORD(3,I)-COORD(3,J))**2)
         end do
      end do

! idensity how many O=C=N=H systems via the interatomic distances
! matrix
      do i=1,qmmm_struct%nquant_nlink
       if (qmmm_struct%iqm_atomic_numbers(i) .EQ. 8) then
        do j=1,qmmm_struct%nquant_nlink
         if (qmmm_struct%iqm_atomic_numbers(j) .EQ. 6) then
          IJ=MAX(I,J)
          JI=I+J-IJ
! RCW  1.69=1.3^2
          if (RXYZ((IJ*(IJ-1))/2+JI) .LE. 1.69D0) then
           do k=1,qmmm_struct%nquant_nlink
            if (qmmm_struct%iqm_atomic_numbers(k).EQ.7) then
             JK=MAX(J,K)
             KJ=J+K-JK
! 2.56=1.6^2
             if (RXYZ((JK*(JK-1))/2+KJ) .LE. 2.56D0) then
              do l=1,qmmm_struct%nquant_nlink
               if (qmmm_struct%iqm_atomic_numbers(L) .EQ. 1) then
                KL=MAX(K,L)
                LK=K+L-KL
                if (RXYZ((KL*(KL-1))/2+LK) .LE. 1.69) then

! We have a H-N-C=O system. The atom numbers are L-K-J-I.
! Now search out atom attachged to N, this specifies the system X-N-C=O.
                 do M=1,qmmm_struct%nquant_nlink
                    if ((M .NE. K).AND.(M .NE. L).AND.(M .NE. J)) then
                        MK=MAX(M,K)
                        KM=M+K-MK
! 2.89=1.7^2
                        if (RXYZ((MK*(MK-1))/2+KM).LE. 2.89D0) then
                             n_peptide_links = n_peptide_links+2
                        end if
                    end if
                 end do
                end if
               end if
              end do
             end if
            end if
           end do
          end if
         end if
        end do
       end if
      end do

! State 2
!       Allocate the identity array and fill it.
      allocate (qm2_struct%peptide_links(4,n_peptide_links),  &
                stat=ier)
      if(ier.ne.0) call Aass(1,'qm2_identify_peptide_links',  &
                            'peptide_links')

      n_peptide_links=0
      do i=1,qmmm_struct%nquant_nlink
         if (qmmm_struct%iqm_atomic_numbers(i) .EQ. 8) then
           do j=1,qmmm_struct%nquant_nlink
            if (qmmm_struct%iqm_atomic_numbers(j) .EQ. 6) then
             IJ=MAX(I,J)
             JI=I+J-IJ
! RCW  1.69=1.3^2
             if (RXYZ((IJ*(IJ-1))/2+JI) .LE. 1.69D0) then
              do k=1,qmmm_struct%nquant_nlink
               if (qmmm_struct%iqm_atomic_numbers(k) .EQ. 7) then
                JK=MAX(J,K)
                KJ=J+K-JK
! 2.56=1.6^2
                if (RXYZ((JK*(JK-1))/2+KJ) .LE. 2.56D0) then
                 do l=1,qmmm_struct%nquant_nlink
                  if (qmmm_struct%iqm_atomic_numbers(L) .EQ. 1) then
                   KL=MAX(K,L)
                   LK=K+L-KL

                   if (RXYZ((KL*(KL-1))/2+LK) .LE. 1.69D0) then

! We have a H-N-C=O system. The atom numbers are L-K-J-I.
! Now search out atom attached to N, this specifies the system X-N-C=O.
                    do M=1,qmmm_struct%nquant_nlink
                     if ((M .NE. K).AND.(M .NE. L).AND.(M .NE. J)) then
                         MK=MAX(M,K)
                         KM=M+K-MK
! 2.89=1.7^2
                         if (RXYZ((MK*(MK-1))/2+KM) .LE. 2.89D0) then
                           n_peptide_links=n_peptide_links+1
                           qm2_struct%peptide_links(1,n_peptide_links)=I
                           qm2_struct%peptide_links(2,n_peptide_links)=J
                           qm2_struct%peptide_links(3,n_peptide_links)=K
                           qm2_struct%peptide_links(4,n_peptide_links)=M
                           n_peptide_links=n_peptide_links+1
                           qm2_struct%peptide_links(1,n_peptide_links)=I
                           qm2_struct%peptide_links(2,n_peptide_links)=J
                           qm2_struct%peptide_links(3,n_peptide_links)=K
                           qm2_struct%peptide_links(4,n_peptide_links)=L
                         end if
                     end if
                    end do
                   end if
                  end if
                 end do
                end if
               end if
              end do
             end if
            end if
           end do
         end if
      end do

      deallocate ( RXYZ, stat=ier )
      if(ier.ne.0) call Aass(0,'qm2_identify_peptide_links','RXYZ')

      return
      END SUBROUTINE qm2_identify_peptide_links


      SUBROUTINE qm2_repp(IQM, JQM,R,RR2,RI,SQRTAEE)
!
!  REPP CALCULATES THE TWO-ELECTRON REPULSION INTEGRALS
!
!  ON INPUT 
!    R,RR2    = INTERATOMIC DISTANCE, ^2
!    IQM, JQM = number of the QM pair we are dealing with from
!               a 1 to nquant loop.
!
!  ON OUTPUT 
!    RI       = ARRAY OF TWO-ELECTRON REPULSION INTEGRALS
!
! Optimization by Ross Walker (TSRI,2005)
!

  use qmmm_module, only : qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
      implicit none
! Passed in
      integer, intent(in) :: iqm, jqm
      real(chm_real), intent(in)  :: R, RR2, sqrtaee
      real(chm_real), intent(out) :: RI(22)

!Local
      LOGICAL :: SI,SJ
      real(chm_real)  :: ARG(71),SQR(71)
      real(chm_real)  :: DA,QA,ADE,AQE,XXX,DB,QB,AED,AEQ,AXX,ADQ,AQD,AQQ
      real(chm_real)  :: YYY,ZZZ,WWW,DZE,QZZE,QXXE
      real(chm_real)  :: EDZ,EQZZ,EQXX,DXDX,DZDZ,DZQXX,QXXDZ
      real(chm_real)  :: DZQZZ,QZZDZ,QXXQXX,QXXQYY,QXXQZZ
      real(chm_real)  :: QZZQXX,QZZQZZ,DXQXZ,QXZDX,QXZQXZ


! Atomic units are used in the calculation, and 
! final results are converted to eV. 
      SI = (qm2_params%natomic_orbs(iqm) .GT. 1)
      SJ = (qm2_params%natomic_orbs(jqm) .GT. 1)

!S orbital : (SS/SS)
      RI(1) = AU_TO_EV*SQRTAEE

      IF (SI .AND. (.NOT.SJ)) THEN
! Heavy atom - Hydrogen
         DA  = qm2_params%multip_2c_elec_params(1,iqm)
         QA  = qm2_params%multip_2c_elec_params(2,iqm) * two
         ADE = qm2_params%multip_2c_elec_params(4,iqm) +  &
                qm2_params%multip_2c_elec_params(3,jqm)
         ADE = ADE * ADE
         AQE = qm2_params%multip_2c_elec_params(5,iqm) +   &
               qm2_params%multip_2c_elec_params(3,jqm)
         AQE = AQE * AQE
         XXX = R+DA
         ARG(1) = XXX*XXX + ADE
         XXX = R-DA
         ARG(2) = XXX*XXX + ADE
         XXX = R+QA
         ARG(3) = XXX*XXX + AQE
         XXX = R-QA
         ARG(4) = XXX*XXX + AQE
         ARG(5) = rr2 + AQE
         ARG(6) = ARG(5) + QA*QA

! RW: Inverted this for speed
!!do I = 1,6
!!   SQR(I) = 1.0D0/SQRT(ARG(I))
!!end do
         call vdinvsqrt( 6, arg, sqr )
!

         RI(2) = HALF_AU_TO_EV*SQR(1) - HALF_AU_TO_EV*SQR(2)
         RI(3) = RI(1) + FOURTH_AU_TO_EV*SQR(3) +   &
                 FOURTH_AU_TO_EV*SQR(4) - HALF_AU_TO_EV*SQR(5)
         RI(4) = RI(1) + HALF_AU_TO_EV*SQR(6) - HALF_AU_TO_EV*SQR(5)

      ELSE IF ((.NOT.SI).AND.SJ) THEN                                           
   ! Hydrogen - Heavy atom
         DB  = qm2_params%multip_2c_elec_params(1,jqm)
         QB  = qm2_params%multip_2c_elec_params(2,jqm) * two
         AED = qm2_params%multip_2c_elec_params(3,iqm) +   &
               qm2_params%multip_2c_elec_params(4,jqm)
         AED = AED * AED
         AEQ = qm2_params%multip_2c_elec_params(3,iqm) +   &
               qm2_params%multip_2c_elec_params(5,jqm)
         AEQ = AEQ * AEQ
         XXX = R-DB
         ARG(1) = XXX*XXX + AED
         XXX = R+DB
         ARG(2) = XXX*XXX + AED
         XXX = R-QB
         ARG(3) = XXX*XXX + AEQ 
         XXX = R+QB
         ARG(4) = XXX*XXX + AEQ
         ARG(5) = rr2 + AEQ
         ARG(6) = ARG(5) + QB*QB

! RW: Inverted this for speed 
!!do I = 1,6
!!   SQR(I) = 1.0D0/SQRT(ARG(I))
!!end do
         call vdinvsqrt( 6, arg, sqr )
!

         RI(5) = HALF_AU_TO_EV*SQR(1) - HALF_AU_TO_EV*SQR(2)
         RI(11) = RI(1) + FOURTH_AU_TO_EV*SQR(3) +   &
                  FOURTH_AU_TO_EV*SQR(4) - HALF_AU_TO_EV*SQR(5)
         RI(12) = RI(1) + HALF_AU_TO_EV*SQR(6) - HALF_AU_TO_EV*SQR(5)

      ELSE IF (SI .AND. SJ) then
! Heavy atom - Heavy atom
! Define charge separations
         DA  = qm2_params%multip_2c_elec_params(1,iqm)
         DB  = qm2_params%multip_2c_elec_params(1,jqm)
         QA  = qm2_params%multip_2c_elec_params(2,iqm) * two
         QB  = qm2_params%multip_2c_elec_params(2,jqm) * two
         ADE = qm2_params%multip_2c_elec_params(4,iqm) +   &
               qm2_params%multip_2c_elec_params(3,jqm)
         ADE = ADE * ADE
         AQE = qm2_params%multip_2c_elec_params(5,iqm) +   &
               qm2_params%multip_2c_elec_params(3,jqm)
         AQE = AQE * AQE
         AED = qm2_params%multip_2c_elec_params(3,iqm) +   &
               qm2_params%multip_2c_elec_params(4,jqm)
         AED = AED * AED
         AEQ = qm2_params%multip_2c_elec_params(3,iqm) +   &
               qm2_params%multip_2c_elec_params(5,jqm)
         AEQ = AEQ * AEQ
         AXX = qm2_params%multip_2c_elec_params(4,iqm) +   &
               qm2_params%multip_2c_elec_params(4,jqm)
         AXX = AXX * AXX
         ADQ = qm2_params%multip_2c_elec_params(4,iqm) +   &
               qm2_params%multip_2c_elec_params(5,jqm)
         ADQ = ADQ * ADQ
         AQD = qm2_params%multip_2c_elec_params(5,iqm) +   &
               qm2_params%multip_2c_elec_params(4,jqm)
         AQD = AQD * AQD
         AQQ = qm2_params%multip_2c_elec_params(5,iqm) +   &
               qm2_params%multip_2c_elec_params(5,jqm)
         AQQ = AQQ * AQQ
         XXX = R + DA
         ARG(1) = XXX * XXX + ADE
         XXX = R - DA
         ARG(2) = XXX*XXX + ADE
         XXX = R - QA
         ARG(3) = XXX*XXX + AQE
         XXX = R + QA
         ARG(4) = XXX*XXX + AQE
         ARG(5) = rr2 + AQE
         ARG(6) = ARG(5) + QA*QA
         XXX = R-DB
         ARG(7) = XXX*XXX + AED
         XXX = R+DB
         ARG(8) = XXX*XXX + AED
         XXX = R - QB
         ARG(9) = XXX*XXX + AEQ
         XXX = R + QB
         ARG(10) = XXX*XXX + AEQ
         ARG(11) = rr2 + AEQ
         ARG(12) = ARG(11) + QB*QB
         XXX = DA-DB
         ARG(13) = rr2 + AXX + XXX*XXX
         XXX = DA+DB
         ARG(14) = rr2 + AXX + XXX*XXX
         XXX = R + DA - DB
         ARG(15) = XXX*XXX + AXX
         XXX = R - DA + DB
         ARG(16) = XXX*XXX + AXX
         XXX = R - DA - DB
         ARG(17) = XXX*XXX + AXX
         XXX = R + DA + DB
         ARG(18) = XXX*XXX + AXX
         XXX = R + DA
         ARG(19) = XXX*XXX + ADQ
         ARG(20) = ARG(19) + QB*QB
         XXX = R - DA
         ARG(21) = XXX*XXX + ADQ
         ARG(22) = ARG(21) + QB*QB
         XXX = R - DB
         ARG(23) = XXX*XXX + AQD
         ARG(24) = ARG(23) + QA*QA
         XXX = R + DB
         ARG(25) = XXX*XXX + AQD
         ARG(26) = ARG(25) + QA*QA
         XXX = R + DA - QB
         ARG(27) = XXX*XXX + ADQ
         XXX = R - DA - QB
         ARG(28) = XXX*XXX + ADQ
         XXX = R + DA + QB
         ARG(29) = XXX*XXX + ADQ
         XXX = R - DA + QB
         ARG(30) = XXX*XXX + ADQ
         XXX = R + QA - DB
         ARG(31) = XXX*XXX + AQD
         XXX = R + QA + DB
         ARG(32) = XXX*XXX + AQD
         XXX = R - QA - DB
         ARG(33) = XXX*XXX + AQD
         XXX = R - QA + DB
         ARG(34) = XXX*XXX + AQD
         ARG(35) = rr2 + AQQ
         XXX = QA - QB
         ARG(36) = ARG(35) + XXX*XXX
         XXX = QA + QB
         ARG(37) = ARG(35) + XXX*XXX
         ARG(38) = ARG(35) + QA*QA
         ARG(39) = ARG(35) + QB*QB
         ARG(40) = ARG(38) + QB*QB
         XXX = R - QB
         ARG(41) = XXX*XXX + AQQ
         ARG(42) = ARG(41) + QA*QA
         XXX = R + QB
         ARG(43) = XXX*XXX + AQQ
         ARG(44) = ARG(43) + QA*QA
         XXX = R + QA
         ARG(45) = XXX*XXX + AQQ
         ARG(46) = ARG(45) + QB*QB
         XXX = R - QA
         ARG(47) = XXX*XXX + AQQ
         ARG(48) = ARG(47) + QB*QB
         XXX = R + QA - QB
         ARG(49) = XXX*XXX + AQQ
         XXX = R + QA + QB
         ARG(50) = XXX*XXX + AQQ
         XXX = R - QA - QB
         ARG(51) = XXX*XXX + AQQ
         XXX = R - QA + QB
         ARG(52) = XXX*XXX + AQQ

         QA  = qm2_params%multip_2c_elec_params(2,iqm)
         QB  = qm2_params%multip_2c_elec_params(2,jqm)
         XXX = DA - QB
         XXX = XXX*XXX
         YYY = R - QB
         YYY = YYY*YYY
         ZZZ = DA + QB
         ZZZ = ZZZ*ZZZ
         WWW = R + QB
         WWW = WWW*WWW
         ARG(53) = XXX + YYY + ADQ
         ARG(54) = XXX + WWW + ADQ
         ARG(55) = ZZZ + YYY + ADQ
         ARG(56) = ZZZ + WWW + ADQ
         XXX = QA - DB
         XXX = XXX*XXX
         YYY = QA + DB
         YYY = YYY*YYY
         ZZZ = R + QA
         ZZZ = ZZZ*ZZZ
         WWW = R - QA
         WWW = WWW*WWW
         ARG(57) = ZZZ + XXX + AQD
         ARG(58) = WWW + XXX + AQD
         ARG(59) = ZZZ + YYY + AQD
         ARG(60) = WWW + YYY + AQD
         XXX = QA - QB
         XXX = XXX*XXX
         ARG(61) = ARG(35) + two*XXX
         YYY = QA + QB
         YYY = YYY*YYY
         ARG(62) = ARG(35) + two*YYY
         ARG(63) = ARG(35) + two*(QA*QA+QB*QB)
         ZZZ = R + QA - QB
         ZZZ = ZZZ*ZZZ
         ARG(64) = ZZZ + XXX + AQQ
         ARG(65) = ZZZ + YYY + AQQ
         ZZZ = R + QA + QB
         ZZZ = ZZZ*ZZZ
         ARG(66) = ZZZ + XXX + AQQ
         ARG(67) = ZZZ + YYY + AQQ
         ZZZ = R - QA - QB
         ZZZ = ZZZ*ZZZ
         ARG(68) = ZZZ + XXX + AQQ
         ARG(69) = ZZZ + YYY + AQQ
         ZZZ = R - QA + QB
         ZZZ = ZZZ*ZZZ
         ARG(70) = ZZZ + XXX + AQQ
         ARG(71) = ZZZ + YYY + AQQ

! RW: Inverted this for speed
!!do I = 1,71
!!   SQR(I) = 1.0d0/SQRT(ARG(I))
!!end do
         call vdinvsqrt( 71, arg, sqr )
!

         DZE    = -HALF_AU_TO_EV*SQR(1) + HALF_AU_TO_EV*SQR(2)
         QZZE   = FOURTH_AU_TO_EV*SQR(3) + FOURTH_AU_TO_EV*SQR(4)   &
                  - HALF_AU_TO_EV*SQR(5)
         QXXE   = HALF_AU_TO_EV*SQR(6) - HALF_AU_TO_EV*SQR(5)
         EDZ    = - HALF_AU_TO_EV*SQR(7) + HALF_AU_TO_EV*SQR(8)
         EQZZ   = FOURTH_AU_TO_EV*SQR(9) + FOURTH_AU_TO_EV*SQR(10)   &
                  - HALF_AU_TO_EV*SQR(11)
         EQXX   = HALF_AU_TO_EV*SQR(12) - HALF_AU_TO_EV*SQR(11)
         DXDX   = HALF_AU_TO_EV*SQR(13) - HALF_AU_TO_EV*SQR(14)
         DZDZ   = FOURTH_AU_TO_EV*SQR(15) + FOURTH_AU_TO_EV*SQR(16)   &
                  - FOURTH_AU_TO_EV*SQR(17) - FOURTH_AU_TO_EV*SQR(18)
         DZQXX  = FOURTH_AU_TO_EV*SQR(19) - FOURTH_AU_TO_EV*SQR(20)   &
                  - FOURTH_AU_TO_EV*SQR(21) + FOURTH_AU_TO_EV*SQR(22)
         QXXDZ  = FOURTH_AU_TO_EV*SQR(23) - FOURTH_AU_TO_EV*SQR(24)   &
                  - FOURTH_AU_TO_EV*SQR(25) + FOURTH_AU_TO_EV*SQR(26)
         DZQZZ  = -EIGHTH_AU_TO_EV*SQR(27) + EIGHTH_AU_TO_EV*SQR(28)   &
                  - EIGHTH_AU_TO_EV*SQR(29) + EIGHTH_AU_TO_EV*SQR(30)   &
                  - FOURTH_AU_TO_EV*SQR(21) + FOURTH_AU_TO_EV*SQR(19)
         QZZDZ  = -EIGHTH_AU_TO_EV*SQR(31) + EIGHTH_AU_TO_EV*SQR(32)   &
                  - EIGHTH_AU_TO_EV*SQR(33) + EIGHTH_AU_TO_EV*SQR(34)   &
                  + FOURTH_AU_TO_EV*SQR(23) - FOURTH_AU_TO_EV*SQR(25)
         QXXQXX = EIGHTH_AU_TO_EV*SQR(36) + EIGHTH_AU_TO_EV*SQR(37)   &
                  - FOURTH_AU_TO_EV*SQR(38) - FOURTH_AU_TO_EV*SQR(39)   &
                  + FOURTH_AU_TO_EV*SQR(35)
         QXXQYY = FOURTH_AU_TO_EV*SQR(40) - FOURTH_AU_TO_EV*SQR(38)   &
                  - FOURTH_AU_TO_EV*SQR(39) + FOURTH_AU_TO_EV*SQR(35)
         QXXQZZ = EIGHTH_AU_TO_EV*SQR(42) + EIGHTH_AU_TO_EV*SQR(44)   &
                  - EIGHTH_AU_TO_EV*SQR(41) - EIGHTH_AU_TO_EV*SQR(43)   &
                  - FOURTH_AU_TO_EV*SQR(38) + FOURTH_AU_TO_EV*SQR(35)
         QZZQXX = EIGHTH_AU_TO_EV*SQR(46) + EIGHTH_AU_TO_EV*SQR(48)   &
                  - EIGHTH_AU_TO_EV*SQR(45) - EIGHTH_AU_TO_EV*SQR(47)   &
                  - FOURTH_AU_TO_EV*SQR(39) + FOURTH_AU_TO_EV*SQR(35)
         QZZQZZ = SXNTH_AU_TO_EV*SQR(49) + SXNTH_AU_TO_EV*SQR(50)    &
                  + SXNTH_AU_TO_EV*SQR(51) + SXNTH_AU_TO_EV*SQR(52)   &
                  - EIGHTH_AU_TO_EV*SQR(47) - EIGHTH_AU_TO_EV*SQR(45)   &
                  - EIGHTH_AU_TO_EV*SQR(41) - EIGHTH_AU_TO_EV*SQR(43)   &
                  + FOURTH_AU_TO_EV*SQR(35)
         DXQXZ  = -FOURTH_AU_TO_EV*SQR(53) + FOURTH_AU_TO_EV*SQR(54)   &
                  + FOURTH_AU_TO_EV*SQR(55) - FOURTH_AU_TO_EV*SQR(56)
         QXZDX  = -FOURTH_AU_TO_EV*SQR(57) + FOURTH_AU_TO_EV*SQR(58)   &
                  + FOURTH_AU_TO_EV*SQR(59) - FOURTH_AU_TO_EV*SQR(60)
         QXZQXZ = EIGHTH_AU_TO_EV*SQR(64) - EIGHTH_AU_TO_EV*SQR(66)   &
                  - EIGHTH_AU_TO_EV*SQR(68) + EIGHTH_AU_TO_EV*SQR(70)   &
                  - EIGHTH_AU_TO_EV*SQR(65) + EIGHTH_AU_TO_EV*SQR(67)   &
                  + EIGHTH_AU_TO_EV*SQR(69) - EIGHTH_AU_TO_EV*SQR(71)
         RI(2)  = -DZE
         RI(3)  = RI(1) + QZZE
         RI(4)  = RI(1) + QXXE
         RI(5)  = -EDZ
         RI(6)  = DZDZ
         RI(7)  = DXDX
         RI(8)  = -EDZ -QZZDZ
         RI(9)  = -EDZ -QXXDZ
         RI(10) = -QXZDX
         RI(11) = RI(1) + EQZZ
         RI(12) = RI(1) + EQXX
         RI(13) = -DZE -DZQZZ
         RI(14) = -DZE -DZQXX
         RI(15) = -DXQXZ
         RI(16) = RI(1) +EQZZ +QZZE +QZZQZZ
         RI(17) = RI(1) +EQZZ +QXXE +QXXQZZ
         RI(18) = RI(1) +EQXX +QZZE +QZZQXX
         RI(19) = RI(1) +EQXX +QXXE +QXXQXX
         RI(20) = QXZQXZ
         RI(21) = RI(1) +EQXX +QXXE +QXXQYY
         RI(22) = half * (QXXQXX -QXXQYY)
      END IF

      return
      END SUBROUTINE qm2_repp


      SUBROUTINE qm2_smallest_number(SMALL_local,SMALLSUM)
!
! Calculates the smallest representable number SMALL_local and the
! smallest number SMALLSUM for which 1+SMALLSUM != 1
!
  use chm_kinds
  use qm2_double
  use qm2_constants
      implicit none
      real(chm_real), intent(out) :: SMALL_local, SMALLSUM


      SMALL_local = one
      do while ((SMALL_local*half) .NE. zero)
        SMALL_local = SMALL_local * half
      end do

      SMALLSUM=one
      do while ((one+(SMALLSUM*half)) .NE. one)
        SMALLSUM=SMALLSUM*half
      end do

      return
      END SUBROUTINE qm2_smallest_number


      FUNCTION qm2_HELECT(N,indx_for_qm,P,H,F)
!
! Function calculates the electronic energy of the system in eV
!
! On input
!    N          : Number of atomic orbitals
!    P          : Density matrix, packed, lower triangle
!    H          : One-electron Fock matrix, packed, lower triangle
!    F          : Two-electron Fock matrix, packed, lower triangle 
!
! On output
!    qm2_HELECT : Electronic energy
!
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!Passed in
      real(chm_real),intent(in) :: P(*), H(*), F(*)
      integer, intent(in) :: n, indx_for_qm

!Local variables
      real(chm_real):: ed, qm2_helect, escf
      integer :: k,i,j
      integer :: mstart,mstop,mstart2,mstop2
      integer :: mstart_keep(2),mstop_keep(2), &
                 mstart2_keep(2),mstop2_keep(2)
      integer :: ISTRT_CHECK                 ! for external function

      integer :: old_N(2) = -1000, ier=0
      integer, pointer :: indx_i(:,:) => NULL()
      save old_N,mstart_keep,mstop_keep,mstart2_keep,mstop2_keep
      save indx_i

      if(old_N(indx_for_qm).ne.N) then
         if(indx_for_qm.eq.1) then
            if(associated(indx_i)) then
               deallocate(indx_i,stat=ier)
               if(ier.ne.0) call Aass(0,'qm2_HELECT','indx_i')
            end if
            allocate(indx_i(N,2),stat=ier)
            if(ier.ne.0) call Aass(1,'qm2_HELECT','indx_i')
         end if

         old_N(indx_for_qm) = N
         do i=1,N
            indx_i(i,indx_for_qm)=i*(i+1)/2
         end do

         mstart_keep(indx_for_qm)  = 1
         mstop_keep(indx_for_qm)   = N
         mstart2_keep(indx_for_qm) = 1
         mstop2_keep(indx_for_qm)  = indx_i(N,indx_for_qm)
#if KEY_PARALLEL==1
         if(numnod.gt.1) then
            mstart_keep(indx_for_qm) =  &
             ISTRT_CHECK(mstop_keep(indx_for_qm),N)
            mstart2_keep(indx_for_qm)=  &
             ISTRT_CHECK(mstop2_keep(indx_for_qm),indx_i(N,indx_for_qm))
         end if
      else
         if(QMPI) then
            mstart_keep(indx_for_qm)  = 1
            mstop_keep(indx_for_qm)   = N
            mstart2_keep(indx_for_qm) = 1
            mstop2_keep(indx_for_qm)  = indx_i(N,indx_for_qm)
         end if
#endif 
      end if

      mstart  = mstart_keep(indx_for_qm)
      mstop   = mstop_keep(indx_for_qm)
      mstart2 = mstart2_keep(indx_for_qm)
      mstop2  = mstop2_keep(indx_for_qm)

      ED         = zero
      qm2_helect = zero
      escf       = zero

!! new way
      do I=mstart,mstop              ! 1,N
         K = indx_i(i,indx_for_qm)
         ED= ED + P(K)*(H(K)+F(K))
      end do
      do K=mstart2,mstop2            ! 1:N*(N+1)/2
         escf =escf + P(K)*(H(K)+F(K)) 
      end do
      escf =half*(escf - half*ED)

!! old way
!!    K          = 0
!!    do I=1,N-1
!!       K = K+1
!!       ED= ED+half*P(K)*(H(K)+F(K))
!!       do J=1,I
!!          K    = K+1
!!          escf = escf + half*P(K)*(H(K)+F(K))   
!!
!!! qm2_helect = qm2_helect + half*P(K)*(H(K)+F(K))
!!
!!       end do
!!    end do
!!
!!    K          = K+1
!!    ED         = ED   + half*P(K)*(H(K)+F(K))
!!    escf       = escf + half*ED

! on return:
! in case of parallel run, the GCOMB has called in parent routine.(Not here!)
      qm2_helect = escf

      return
      END FUNCTION qm2_helect


      SUBROUTINE qm2_calc_mulliken(nquant,mul_chg)
!
! Calculates the Mulliken charges on QM atoms
! from QM/MM calculations 
!
! Variables:
!   iqm     : a number from 1 to nquant_nlink
!   mul_chg : Mulliken charges returned, electron units
!
! Codes written by Ross Walker (TSRI, 2005)

  use qmmm_module, only : qm2_params, qm2_struct, qmmm_ewald
  use chm_kinds
  use qm2_double
  use qm2_constants
      implicit none
!Passed in
      integer, intent(in)  :: nquant
      real(chm_real),intent(out) :: mul_chg(nquant)

!Local variables
      integer :: iqm
      integer :: loop_count, orb_beg, orb_end, tri
      real(chm_real):: density_sum

      Do iqm = 1, nquant
! find the beginning and ending orbital locations
         orb_beg = qm2_params%orb_loc(1,iqm)
         orb_end = qm2_params%orb_loc(2,iqm)

! Loop over the density matrix
         density_sum = zero
         do loop_count=orb_beg,orb_end
            tri         = qm2_params%pascal_tri2(loop_count)
            density_sum = density_sum+qm2_struct%den_matrix(tri)
         end do

         mul_chg(iqm) = qm2_params%core_chg(iqm)-density_sum
      End do
      qmmm_ewald%scf_mchg_2(1:nquant)= mul_chg(1:nquant)

! in case of pKa FEP-QM/MM
!     If (qmmm_ewald%qpka_calc_on) then
!        mul_chg(1:nquant) = qmmm_ewald%mm_chg_fix_qm(1:nquant)
!        mul_chg(1:nquant) = zero
!     End if

      return
      END SUBROUTINE qm2_calc_mulliken

#endif /* (mainsquatn)*/

SUBROUTINE qm2_hcore_qmmm_BLANK
  !
  ! dummy routine for compilation
  !
  RETURN
END SUBROUTINE qm2_hcore_qmmm_BLANK

