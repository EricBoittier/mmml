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


      SUBROUTINE qm2_dhc(P,iqm, jqm,qmitype,qmjtype,xyz_qmi,xyz_qmj,  &
                         natqmi,natqmj,IF,IL,JF,JL,indx_for_qm,DENER)
!
! DHC calculates the energy contributions from those pairs
!     of atoms that have been moved by SUBROUTINE DERIV.
!
      
  use qmmm_module, only : qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
      implicit none
!Passed in
      real(chm_real)               :: P(*)
      real(chm_real),  intent(in)  :: xyz_qmi(3),xyz_qmj(3)
      integer, intent(in)  :: iqm, jqm, natqmi, natqmj, qmitype, qmjtype
      integer, intent(in)  :: if,il,jf,jl,indx_for_qm
      real(chm_real),  intent(out) :: DENER

!Local variables
      integer  :: n_atomic_orbi, n_atomic_orbj, ja, jb, ia,ib,i,j
      integer  :: j1,jj,i1, linear, i2, ii
      real(chm_real)   :: H(36), F(36)          ! 36=max of 8*(8+1)/2 
                                          !    4 orbs with 4 orbs (S,3P)
      real(chm_real)   :: SHMAT(4,4),E1B(10), E2A(10), W(100)
      real(chm_real)   :: enuclr, ee, vec_qm_qm1, vec_qm_qm2, vec_qm_qm3, R2
      integer  :: orb_loc(2,2),KR

      real(chm_real)   :: qm2_helect            ! qm2_Helect is a function

      orb_loc(1,1)  = 1
      orb_loc(2,1)  = IL-IF+1
      orb_loc(1,2)  = orb_loc(2,1)+1
      orb_loc(2,2)  = orb_loc(1,2)+JL-JF
      n_atomic_orbi = orb_loc(2,2)-orb_loc(1,2)+1
      n_atomic_orbj = orb_loc(2,1)-orb_loc(1,1)+1
      LINEAR        = qm2_params%pascal_tri2(orb_loc(2,2))

! replace
      F(1:linear) = zero
      H(1:linear) = zero
!do I=1,LINEAR
!   F(I)=0.0D0 
!   H(I)=0.0D0
!end do

      IA = orb_loc(1,1)
      IB = orb_loc(2,1)
      JA = orb_loc(1,2)
      JB = orb_loc(2,2)
      J  = 2
      I  = 1

! RCW: Caution, i and j reversed here      

      vec_qm_qm1 = (xyz_qmj(1)-xyz_qmi(1))
      vec_qm_qm2 = (xyz_qmj(2)-xyz_qmi(2))
      vec_qm_qm3 = (xyz_qmj(3)-xyz_qmi(3))
      R2 = (vec_qm_qm1*vec_qm_qm1 + vec_qm_qm2*vec_qm_qm2 +   &
            vec_qm_qm3*vec_qm_qm3)*A2_TO_BOHRS2

      if (R2 .LT. OVERLAP_CUTOFF) then
         SHMAT = zero

         CALL qm2_h1elec(-1,R2,xyz_qmj(1),xyz_qmi(1),   &
                         n_atomic_orbj,n_atomic_orbi,SHMAT,   &
             qm2_params%atom_orb_zz_sxs_over_sas(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_zz_sxs_over_sas(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_ss_eqn(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_zz_sxp_over_sap(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_zz_sxp_over_sap(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_sp_ovlp(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_sp_ovlp(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_zz_pxp_over_pap(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_pp_ovlp_ieqj1(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_pp_ovlp_ieqj2(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_pp_ovlp_inj(1,1,qmjtype,qmitype),   &
             qm2_params%betasas(qmjtype,qmitype),   &
             qm2_params%betasap(qmjtype,qmitype),   &
             qm2_params%betasap(qmitype,qmjtype),   &
             qm2_params%betapap(qmjtype,qmitype))          
         J1=0
         do J=JA,JB
            JJ=qm2_params%pascal_tri1(j)
            J1=J1+1
            I1=0
            do I=IA,IB
               JJ=JJ+1
               I1=I1+1
               H(JJ)=SHMAT(I1,J1)
               F(JJ)=SHMAT(I1,J1)
            end do
         end do
      end if                                     !(R2 < OVERLAP_CUTOFF)

      KR=1
      CALL qm2_rotate_qmqm(-1,iqm,jqm,natqmi,natqmj,xyz_qmi,xyz_qmj,   &
                           W(KR),KR,E2A,E1B,ENUCLR,qmitype,qmjtype)

! Nuclear is summed over core-core repulsion integrals
      I2=0
      do I1=IA,IB
         II=qm2_params%pascal_tri1(i1)+IA-1
         do J1=IA,I1
            II=II+1
            I2=I2+1
            H(II)=H(II)+E1B(I2)
            F(II)=F(II)+E1B(I2)
         end do
      end do

      I2=0
      do I1=JA,JB
         II=qm2_params%pascal_tri1(i1)+JA-1
         do J1=JA,I1
            II=II+1
            I2=I2+1
            H(II)=H(II)+E2A(I2)
            F(II)=F(II)+E2A(I2)
         end do
      end do

      CALL qm2_fock2(F,P,W,2,orb_loc,indx_for_qm)

      EE   = two*qm2_HELECT(orb_loc(2,2),P,H,F)
      DENER= EE+ENUCLR

      return
      END SUBROUTINE qm2_dhc


      SUBROUTINE qm2_get_qm_forces(indx_for_qm,dxyzqm)
!
! This routine calculates the derivatives of the energy
! for QM-QM interactions.
!
!    qmmm_struct%qm_coords : QM Atom Coordinates
!    dxyzqm                : Gradients for QM atoms
!
! Code written by Ross Walker (TSRI 2004)
!

  use qmmm_module, only : qmmm_nml,qmmm_struct, qm2_struct,  &
                              qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!     CHNGE, HALFCHNGE, oneCHNGE is defined in qm2_constants.f90

!Passed in
      integer,  intent(in)  :: indx_for_qm
      real(chm_real), intent(out) :: dxyzqm(3,qmmm_struct%nquant_nlink) 

!Local
      real(chm_real):: e_repul(22)        ! Used when qmqm_erep_incore = false
      real(chm_real):: pair_force(3)
      integer :: loop_count         ! Keeps track of number of times 
                                    ! through nquant * (nqaunt-1)/2 loop
      real(chm_real):: psum(36)           ! 36 = max combinations with 
                                    !      heavy and heavy 
                                    !      = 4 orbs * 4 orbs 
                                    ! Note: no d orb support
      real(chm_real):: xyz_qmi(3), xyz_qmj(3)
      real(chm_real):: vec_qm_qm1, vec_qm_qm2, vec_qm_qm3
      integer :: natqmi, natqmj, qmitype, qmjtype
      integer :: ii, iif, iil, jj, jjf, jjl, ij
      integer :: i, j, k, l
      integer :: n_atomic_orbi, n_atomic_orbj
      real(chm_real):: aa,ee,deriv,del,angle,refh,heat,sum
      real(chm_real):: corei, corej
      real(chm_real):: alphai, alphaj 
      real(chm_real):: betasas, betasap, betapas, betapap
      real(chm_real):: bdd1i, bdd2i, bdd3i, bdd1j, bdd2j, bdd3j
      real(chm_real):: qqi, qqi2, qqj, qqj2, ddi,ddj
      real(chm_real):: htype

! this part has been moved into "qm2_constants.f90"
!     #define CHNGE 1.D-4
!     #define HALFCHNGE 5.D-5
!     ! one/CHNGE = 10000
!     #define oneCHNGE 10000

! RCW: Note there is a lot of repeated code in the two options 
!      below but it is quicker to have the if statement outside
!      of the loop. Even if it is uglier.
! ************** ANALYTICAL DERIVATIVES **************
      if (qmmm_nml%qmqm_analyt) then
         loop_count = 0
         do II=2,qmmm_struct%nquant_nlink

! Loop over all pairs of quantum atoms
     
! First atom info
            iif = qm2_params%orb_loc(1,II)
            iil = qm2_params%orb_loc(2,II)

! n_atomic_orbi = iil - iif + 1
            n_atomic_orbi = qm2_params%natomic_orbs(ii)
            natqmi        = qmmm_struct%iqm_atomic_numbers(II)
            corei         = qm2_params%core_chg(ii)
            alphai        = qm2_params%cc_exp_params(ii)
            ddi           = qm2_params%multip_2c_elec_params(1,ii)
            qqi           = qm2_params%multip_2c_elec_params(2,ii)
            qqi2          = qqi*qqi
            bdd1i         = qm2_params%multip_2c_elec_params(3,ii)
            bdd2i         = qm2_params%multip_2c_elec_params(4,ii)
            bdd3i         = qm2_params%multip_2c_elec_params(5,ii)
            xyz_qmi(1)    = qmmm_struct%qm_coords(1,II)
            xyz_qmi(2)    = qmmm_struct%qm_coords(2,II)
            xyz_qmi(3)    = qmmm_struct%qm_coords(3,II)
            qmitype       = qmmm_struct%qm_atom_type(ii)

            do JJ=1,II-1
               loop_count= loop_count + 1

#if KEY_PARALLEL==1
               if ( mynod .eq. MOD(loop_count-1,numnod) ) then
#endif 

! Secon atom info
               jjf = qm2_params%orb_loc(1,JJ)
               jjl = qm2_params%orb_loc(2,JJ)

! n_atomic_orbj = jjl - jjf + 1
               n_atomic_orbj = qm2_params%natomic_orbs(jj)
               natqmj        = qmmm_struct%iqm_atomic_numbers(JJ)
               corej         = qm2_params%core_chg(jj)
               alphaj        = qm2_params%cc_exp_params(jj)
               ddj           = qm2_params%multip_2c_elec_params(1,jj)
               qqj           = qm2_params%multip_2c_elec_params(2,jj)
               qqj2          = qqj*qqj
               bdd1j         = qm2_params%multip_2c_elec_params(3,jj)
               bdd2j         = qm2_params%multip_2c_elec_params(4,jj)
               bdd3j         = qm2_params%multip_2c_elec_params(5,jj)
               xyz_qmj(1)    = qmmm_struct%qm_coords(1,JJ)
               xyz_qmj(2)    = qmmm_struct%qm_coords(2,JJ)
               xyz_qmj(3)    = qmmm_struct%qm_coords(3,JJ)
               vec_qm_qm1    = xyz_qmi(1) - xyz_qmj(1)
               vec_qm_qm2    = xyz_qmi(2) - xyz_qmj(2)
               vec_qm_qm3    = xyz_qmi(3) - xyz_qmj(3)
               qmjtype       = qmmm_struct%qm_atom_type(jj)
               betasas       = qm2_params%betasas(qmitype,qmjtype)
               betasap       = qm2_params%betasap(qmitype,qmjtype)
               betapas       = qm2_params%betasap(qmjtype,qmitype)
               betapap       = qm2_params%betapap(qmitype,qmjtype)

               IJ = 0
! Form diatomic matrices
               do I = jjf, jjl
                  K = qm2_params%pascal_tri1(i)+jjf-1
                  do J= jjf, I
                     IJ      = IJ+1
                     K       = K+1
                     psum(IJ)= qm2_struct%den_matrix(K)
                  end do
               end do

! Get second atom first atom intersection
               do I= iif, iil
                  L= qm2_params%pascal_tri1(i)
                  K= L + jjf - 1
                  do J= jjf, jjl
                     IJ      = IJ+1
                     K       = K+1
                     psum(IJ)= qm2_struct%den_matrix(K)
                  end do

                  K= L + iif - 1
                  do L= iif, I
                     K       = K+1
                     IJ      = IJ+1
                     psum(IJ)= qm2_struct%den_matrix(K)
                  end do
               end do

!! loop_count update moved above.
!!             loop_count= loop_count + 1
               if (qmmm_nml%qmqm_erep_incore) then
                  CALL qm2_deriv_qm_analyt(ii,jj,loop_count,   &
                       qm2_struct%qm_qm_e_repul(1,loop_count),   &
                       psum,natqmi,natqmj,n_atomic_orbj,n_atomic_orbi,   &
                       corei,corej,betasas,betasap,betapas,betapap,   &
                       vec_qm_qm1,vec_qm_qm2,vec_qm_qm3,alphai,alphaj,   &
                       pair_force, qqi, qqi2, qqj, qqj2, ddi, ddj,   &
                       bdd1i, bdd2i, bdd3i, bdd1j, bdd2j, bdd3j,   &
             qm2_params%atom_orb_zz_sxs_over_sas(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_ss_eqn_adb(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_zz_sxp_over_sap(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_zz_sxp_over_sap(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_sp_eqn_xx1(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_sp_eqn_xx2(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_sp_eqn_xx1(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_sp_eqn_xx2(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_sp_eqn_xy(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_sp_eqn_xy(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_zz_pxp_over_pap(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_pp_eqn_xxy1(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_pp_eqn_xxy2(1,1,qmitype,qmjtype),   &
                  qmitype,qmjtype)
               else
! Same call as above, 
! just qm2_struct%qm_qm_e_repul(1,loop_count) 
! replaced with local e_repul
                  CALL qm2_deriv_qm_analyt(ii,jj,loop_count,e_repul,   &
                       psum,natqmi,natqmj,n_atomic_orbj,n_atomic_orbi,   &
                       corei,corej,betasas,betasap,betapas,betapap,   &
                       vec_qm_qm1,vec_qm_qm2,vec_qm_qm3,alphai,alphaj,   &
                       pair_force, qqi, qqi2, qqj, qqj2, ddi, ddj,   &
                       bdd1i, bdd2i, bdd3i, bdd1j, bdd2j, bdd3j,   &
             qm2_params%atom_orb_zz_sxs_over_sas(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_ss_eqn_adb(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_zz_sxp_over_sap(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_zz_sxp_over_sap(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_sp_eqn_xx1(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_sp_eqn_xx2(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_sp_eqn_xx1(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_sp_eqn_xx2(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_sp_eqn_xy(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_sp_eqn_xy(1,1,qmjtype,qmitype),   &
             qm2_params%atom_orb_zz_pxp_over_pap(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_pp_eqn_xxy1(1,1,qmitype,qmjtype),   &
             qm2_params%atom_orb_pp_eqn_xxy2(1,1,qmitype,qmjtype),   &
                       qmitype,qmjtype)
               end if

               dxyzqm(1,II) = dxyzqm(1,II) - pair_force(1)
               dxyzqm(2,II) = dxyzqm(2,II) - pair_force(2)
               dxyzqm(3,II) = dxyzqm(3,II) - pair_force(3)
               dxyzqm(1,JJ) = dxyzqm(1,JJ) + pair_force(1)
               dxyzqm(2,JJ) = dxyzqm(2,JJ) + pair_force(2)
               dxyzqm(3,JJ) = dxyzqm(3,JJ) + pair_force(3)

#if KEY_PARALLEL==1
               end if                    ! mod(loop_count,numnod)
#endif 
            end do                       ! JJ=1,II-1
         end do                          ! II=2,qmmm_struct%nquant_nlink

!************** NUMERICAL DERIVATIVES **************
      else  

#if KEY_PARALLEL==1
        loop_count = 0
#endif 
        do II=2,qmmm_struct%nquant_nlink

! Loop over all pairs of quantum atoms
           iif = qm2_params%orb_loc(1,II)
           iil = qm2_params%orb_loc(2,II)

           qmitype   = qmmm_struct%qm_atom_type(ii)
           natqmi    = qmmm_struct%iqm_atomic_numbers(II)
           xyz_qmi(1)= qmmm_struct%qm_coords(1,II)
           xyz_qmi(2)= qmmm_struct%qm_coords(2,II)
           xyz_qmi(3)= qmmm_struct%qm_coords(3,II)

           do JJ=1,II-1
#if KEY_PARALLEL==1
              loop_count= loop_count + 1
              if ( mynod .eq. MOD(loop_count-1,numnod) ) then
#endif 
! Form diatomic matrices

! First atom
              jjf = qm2_params%orb_loc(1,JJ)
              jjl = qm2_params%orb_loc(2,JJ)

              qmjtype   = qmmm_struct%qm_atom_type(jj)
              natqmj    = qmmm_struct%iqm_atomic_numbers(JJ)
              xyz_qmj(1)= qmmm_struct%qm_coords(1,JJ)
              xyz_qmj(2)= qmmm_struct%qm_coords(2,JJ)
              xyz_qmj(3)= qmmm_struct%qm_coords(3,JJ)

              IJ = 0
              do I=jjf, jjl
                 K= qm2_params%pascal_tri1(i) + jjf - 1
                 do J=jjf, I
                    IJ      = IJ+1
                    K       = K+1
                    psum(IJ)= qm2_struct%den_matrix(K)
                 end do
              end do

! Get second atom first atom intersection
              do I=iif, iil
                 L= qm2_params%pascal_tri1(i)
                 K= L+jjf-1
                 do J=jjf, jjl
                    IJ      = IJ+1
                    K       = K+1
                    psum(IJ)= qm2_struct%den_matrix(K)
                 end do

                 K = L + iif - 1
                 do L=iif, I
                     K       = K+1
                     IJ      = IJ+1
                     psum(IJ)= qm2_struct%den_matrix(K)
                 end do
              end do

              do K=1,3
                 xyz_qmi(K) = xyz_qmi(K)-HALFCHNGE
                 CALL qm2_dhc(psum,ii,jj,qmitype,qmjtype,   &
                              xyz_qmi,xyz_qmj,   &
                              natqmi,natqmj,jjf,jjl,iif,iil, &
                              indx_for_qm,AA)

                 xyz_qmi(K) = xyz_qmi(K)+CHNGE
                 CALL qm2_dhc(psum,ii,jj,qmitype,qmjtype,   &
                              xyz_qmi,xyz_qmj,   &
                              natqmi,natqmj,jjf,jjl,iif,iil, &
                              indx_for_qm,EE)

                 xyz_qmi(K) = xyz_qmi(K)-HALFCHNGE
                 DERIV      = (AA-EE)*EV_TO_KCAL*oneCHNGE

                 dxyzqm(K,II) = dxyzqm(K,II) - DERIV
                 dxyzqm(K,JJ) = dxyzqm(K,JJ) + DERIV
              end do

#if KEY_PARALLEL==1
              end if                     ! mod(loop_count,numnod)
#endif 
           end do                        ! JJ=1,II-1
        end do                           ! II=2,qmmm_struct%nquant_nlink
      end if                             ! (qmmm_nml%qmqm_analyt)


! Now add in molecular mechanical correction to H-N-C=O torsion
      IF (qmmm_nml%peptide_corr) THEN
#if KEY_PARALLEL==1
        if(mynod.eq.0) then
#endif 
        if (qmmm_nml%qmtheory .EQ. PM3 .OR.   &
            qmmm_nml%qmtheory .EQ. PDDGPM3 .OR.   &
            qmmm_nml%qmtheory .EQ. PM3CARB1 ) then
           htype = 7.1853D0                      ! PM3 related model
        else if (qmmm_nml%qmtheory==AM1) then
           htype = 3.3191D0                      ! AM1 model
        else
           htype = 6.1737D0                      ! Assume MNDO
        end if

        DEL=1.D-8
        do I=1,qm2_struct%n_peptide_links
           do J=1,4
              do K=1,3
               qmmm_struct%qm_coords(K,qm2_struct%peptide_links(J,I))   &
                        =  qmmm_struct%qm_coords(K,   &
                                       qm2_struct%peptide_links(J,I))   &
                         - DEL
               CALL qm2_dihed(qmmm_struct%qm_coords,   &
                              qm2_struct%peptide_links(1,I),   &
                              qm2_struct%peptide_links(2,I),   &
                              qm2_struct%peptide_links(3,I),   &
                              qm2_struct%peptide_links(4,I),ANGLE)
               REFH=HTYPE*SIN(ANGLE)**2

               qmmm_struct%qm_coords(K,qm2_struct%peptide_links(J,I))   &
                        =  qmmm_struct%qm_coords(K,   &
                                       qm2_struct%peptide_links(J,I))   &
                         + two*DEL
               CALL qm2_dihed(qmmm_struct%qm_coords,   &
                              qm2_struct%peptide_links(1,I),   &
                              qm2_struct%peptide_links(2,I),   &
                              qm2_struct%peptide_links(3,I),   &
                              qm2_struct%peptide_links(4,I),ANGLE)

               qmmm_struct%qm_coords(K,qm2_struct%peptide_links(J,I))   &
                        =  qmmm_struct%qm_coords(K,   &
                                       qm2_struct%peptide_links(J,I))   &
                         - DEL

               HEAT = HTYPE*SIN(ANGLE)**2
               SUM  = (REFH-HEAT)/(two*DEL)

               dxyzqm(K,qm2_struct%peptide_links(J,I)) =   &
                           dxyzqm(K,qm2_struct%peptide_links(J,I))-SUM 
              end do
           end do
        end do
#if KEY_PARALLEL==1
        end if
        call psync()
#endif 
      end if

      return
      END SUBROUTINE qm2_get_qm_forces


      SUBROUTINE qm2_deriv_qm_analyt(iqm,jqm,loop_count,qm_qm_e_repul,   &
                       PSUM,natqmi,natqmj,n_atomic_orbj,n_atomic_orbi,   &
                       corei,corej,betasas,betasap,betapas,betapap,   &
                       vec_qm_qm1,vec_qm_qm2,vec_qm_qm3,alphai,alphaj,   &
                       pair_force,qqi,qqi2,qqj,qqj2,ddi,ddj,   &
                       bdd1i,bdd2i,bdd3i,bdd1j,bdd2j,bdd3j,   &
                       zz_sxs_over_sas,ss_eqn_adb,zz_sxp_over_sapij,   &
                       zz_sxp_over_sapji,sp_eqn_xx1ij,sp_eqn_xx2ij,   &
                       sp_eqn_xx1ji,sp_eqn_xx2ji,   &
                       sp_eqn_xyij,sp_eqn_xyji,   &
                       zz_pxp_over_pap,pp_eqn_xxy1,pp_eqn_xxy2,   &
                       qmitype,qmjtype)
!
! Calculation of analytical derivatives
!
! Variable Definitions:
!    qm_qm_e_repul : QM-QM electron repulsion integrals for 
!                    this QM-QM pair
!    psum          : Density matrix elements for orbitals 
!                    centered on this
!                    QM-QM pair. - For coulomb term
!    natqmi        : atomic number of QM atom i
!    natqmj        : atomic number of QM atom j
!    n_atomic_orbj : number of atomic orbitals on atom j
!    n_atomic_orbi : number of atomic orbitals on atom i
!    vec_qm_qmx    : xyz_qmi(x) - xyz_qmj(x)
!    alphai        :
!    alphaj        :
!    pair_force    : Returned with cartesian force on this 
!                    QM-QM pair
!
! Current routine, inlining, and optimization by Ross Walker 
! (TSRI, 2004)
!
! Note: this routine is tooooooo long, it should separated into
!       several subroutines. (namkh)
!

  use qmmm_module, only : qmmm_nml,qmmm_struct,qm2_rij_eqns,   &
                              qm2_struct,qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations
      implicit none
! Passed in
      integer, intent(in)    :: iqm,jqm,loop_count,qmitype,qmjtype
      real(chm_real),  intent(inout) :: qm_qm_e_repul(22)    
!                                           !  for qmqm_erep_incore=true 
!                                           !  it is in, 
!                                           !  for =false it is out.
      real(chm_real),  intent(in)  :: psum(36)
      integer, intent(in)    :: natqmi,natqmj
      integer, intent(in)    :: n_atomic_orbj,n_atomic_orbi
      real(chm_real),  intent(in)  :: corei,corej,   &
                                  betasas,betasap,betapas,betapap
      real(chm_real),  intent(in)  :: vec_qm_qm1,vec_qm_qm2,vec_qm_qm3
      real(chm_real),  intent(in)  :: alphai,alphaj,qqi,qqi2,qqj,qqj2,ddi,ddj
      real(chm_real),  intent(in)  :: bdd1i,bdd2i,bdd3i,bdd1j,bdd2j,bdd3j
      real(chm_real),  intent(out) :: pair_force(3)
      real(chm_real),  intent(in)  :: zz_sxs_over_sas(6,6),ss_eqn_adb(6,6)
      real(chm_real),  intent(in)  :: zz_sxp_over_sapij(6,6),   &
                                zz_sxp_over_sapji(6,6)
      real(chm_real),  intent(in)  :: sp_eqn_xx1ij(6,6),sp_eqn_xx2ij(6,6)
      real(chm_real),  intent(in)  :: sp_eqn_xx1ji(6,6),sp_eqn_xx2ji(6,6)
      real(chm_real),  intent(in)  :: sp_eqn_xyij(6,6),sp_eqn_xyji(6,6)
      real(chm_real),  intent(in)  :: zz_pxp_over_pap(6,6)
      real(chm_real),  intent(in)  :: pp_eqn_xxy1(6,6),pp_eqn_xxy2(6,6)

! Local variables
      real(chm_real) :: FAAX, FAAY, FAAZ,FABX, FABY, FABZ,FNUCX, FNUCY, FNUCZ
      real(chm_real) :: dsx,dsy,dsz,DSPX(3), DSPY(3), DSPZ3
      real(chm_real) :: DPXPX(3), DPYPY(3), DPZPZ(3), DPCROSS
      real(chm_real) :: dgx(22), dgy(22), dgz(22)
      real(chm_real) :: drx(100), dry(100), drz(100)
      real(chm_real) :: r2, oner2, rij, onerij, rr, rr2
      real(chm_real) :: c1, f3, ddx, ddy, ddz
      real(chm_real) :: part1x, part1y, part1z
      real(chm_real) :: part2x, part2y, part2z
      real(chm_real) :: part3x, part3y, part3z
      real(chm_real) :: anam1
      real(chm_real) :: bb, aa
      integer :: total_atomic_orb, qm_atomi_orb_start, orb_offset
      integer :: isp, k, l
      integer :: m, n, mn, kk, ll, kl, mk, nk, ml, nl

      real(chm_real) :: vec_qm_qm_onerij1,vec_qm_qm_onerij2,vec_qm_qm_onerij3
      real(chm_real) :: vec2_qm_qm1, vec2_qm_qm2, vec2_qm_qm3
      real(chm_real) :: vec_qm_qm123, vec_qm_qm12, vec_qm_qm23, vec_qm_qm13
      integer :: i, j
      real(chm_real) :: ABN, ADBR2, SS
      real(chm_real) :: temp_real, temp_real2, temp_real3, temp_real4
      real(chm_real) :: EXP1i, EXP1j, SQRTAEE, bdd1ij

! Specific for PDDG
      real(chm_real) :: PDDG_EXP1, PDDG_EXP2, PDDG_EXP3, PDDG_EXP4, PDDG_CORR

! Originally in delri
      real(chm_real) :: termx, termy, termz, ee, ade, aqe, dze, qzze, qxxe
      real(chm_real) :: aed, aeq, edz, eqzz, eqxx
      real(chm_real) :: add, adq, aqd, aqq, dxdx, dzdz,dzqxx,qxxdz,dzqzz,qzzdz
      real(chm_real) :: qxxqxx, qxxqyy, qxxqzz, qzzqxx, qzzqzz, dxqxz,  &
                  qxzdx, qxzqxz

! Originally in delmol
      integer :: mm, nn
      real(chm_real) :: xTDX(3),xTDY(3),xTDZ(3)
      real(chm_real) :: yTDX(3),yTDY(3),yTDZ(3)
      real(chm_real) :: zTDX(3),zTDY(3),zTDZ(3)
      real(chm_real) :: TX(3),TY(3),TZ(3), TZ3i, TZ3i2
      real(chm_real) :: xtemp1, ytemp1, ztemp1, xtemp2, ytemp2, ztemp2

! Originally for rotat
      real(chm_real) :: RXY2, RYZ2, RZX2, oneRXY
      logical :: LRXY2, LRYZ2, LRZX2

      LOGICAL :: AM1_OR_PM3, PDDG_IN_USE
      logical :: first_call

      data first_call /.true./

      save first_call
      SAVE AM1_OR_PM3, PDDG_IN_USE

! Only first call
      if (first_call) then
         first_call  = .false.
         AM1_OR_PM3  = (qmmm_nml%qmtheory .EQ. AM1 .OR.   &
                        qmmm_nml%qmtheory .EQ. PM3 .OR.   &
                        qmmm_nml%qmtheory .EQ. PDDGPM3 .OR.   &
                        qmmm_nml%qmtheory .EQ. PM3CARB1 ) 
         PDDG_IN_USE = (qmmm_nml%qmtheory .EQ. PDDGPM3 .OR.   &
                        qmmm_nml%qmtheory .EQ. PDDGMNDO)
      end if
 
      FAAX=zero
      FAAY=zero
      FAAZ=zero
      FABX=zero
      FABY=zero
      FABZ=zero

! heavy atoms
      if (n_atomic_orbi .GT. 1 .OR. n_atomic_orbj .GT.1) then
        vec_qm_qm123 = vec_qm_qm1*vec_qm_qm2*vec_qm_qm3
        vec_qm_qm12  = vec_qm_qm1*vec_qm_qm2
        vec_qm_qm23  = vec_qm_qm2*vec_qm_qm3
        vec_qm_qm13  = vec_qm_qm1*vec_qm_qm3
      end if

      vec2_qm_qm1   = vec_qm_qm1*vec_qm_qm1
      vec2_qm_qm2   = vec_qm_qm2*vec_qm_qm2
      vec2_qm_qm3   = vec_qm_qm3*vec_qm_qm3

!***************************************************************
! Step 1: get the distance related terms

      if (qmmm_nml%qmqmrij_incore) then
         rr2    = qm2_rij_eqns%qmqmrijdata(QMQMRIJBOHRS2,loop_count)
         oneRIJ = qm2_rij_eqns%qmqmrijdata(QMQMONERIJ,loop_count) 
         RIJ    = qm2_rij_eqns%qmqmrijdata(QMQMRIJ,loop_count)
         RR     = qm2_rij_eqns%qmqmrijdata(QMQMRIJBOHRS,loop_count)
         EXP1i  = qm2_rij_eqns%qmqmrijdata(QMQMEXP1I,loop_count)
         EXP1j  = qm2_rij_eqns%qmqmrijdata(QMQMEXP1J,loop_count)
         SQRTAEE= qm2_rij_eqns%qmqmrijdata(QMQMSQRTAEE,loop_count)

! PDDG specific terms
         if (PDDG_IN_USE) then
            PDDG_EXP1 =qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP1,   &
                                                loop_count)
            PDDG_EXP2 =qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP2,   &
                                                loop_count)
            PDDG_EXP3 =qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP3,   &
                                                loop_count)
            PDDG_EXP4 =qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP4,   &
                                                loop_count)
         end if
      else
         r2     = vec2_qm_qm1+vec2_qm_qm2+vec2_qm_qm3
         rr2    = r2 * A2_TO_BOHRS2
         oneRIJ = one/sqrt(r2)                 ! 1/sqrt is faster than 
!!                                             ! doing sqrt.
         RIJ    = r2*oneRIJ                    ! one/oneRIJ
         RR     = RIJ*A_TO_BOHRS
         EXP1i  = EXP(-alphai*RIJ)
         EXP1j  = EXP(-alphaj*RIJ)
         bdd1ij = bdd1i+bdd1j
         bdd1ij = bdd1ij*bdd1ij
         SQRTAEE= one/sqrt(RR2+bdd1ij)

! PDDG specific terms
         if (PDDG_IN_USE) then
            PDDG_EXP1 =EXP(-ten * ( RIJ-qm2_params%pddge1(iqm)-   &
                                        qm2_params%pddge1(jqm))**2 )
            PDDG_EXP2 =EXP(-ten * ( RIJ-qm2_params%pddge1(iqm)-   &
                                        qm2_params%pddge2(jqm))**2 )
            PDDG_EXP3 =EXP(-ten * ( RIJ-qm2_params%pddge2(iqm)-   &
                                        qm2_params%pddge1(jqm))**2 )
            PDDG_EXP4 =EXP(-ten * ( RIJ-qm2_params%pddge2(iqm)-   &
                                        qm2_params%pddge2(jqm))**2 )
         end if
      end if

      vec_qm_qm_onerij1 = vec_qm_qm1 * oneRIJ
      vec_qm_qm_onerij2 = vec_qm_qm2 * oneRIJ
      vec_qm_qm_onerij3 = vec_qm_qm3 * oneRIJ

      total_atomic_orb=n_atomic_orbi+n_atomic_orbj

      qm_atomi_orb_start=n_atomic_orbj+1


!***************************************************************
! Step2 : get the 1-e repulsion integrals realted terms

! If we don't have the 1-e repul integrals for this
! QM-QM pairs in memory we need to calculate them now.
      if (.NOT. qmmm_nml%qmqm_erep_incore) then
         call qm2_repp(iqm,jqm,rr,rr2,qm_qm_e_repul,SQRTAEE)
      end if

      C1=corei*corej


!***************************************************************
! Step3 : get AM1/PM3/PDDG related terms

! Start of the AM1 and PM3 specific derivative code
! Analytical derivatives: -A*(1/(R*R)+2.D0*B*(R-C)/R) &
!                          *EXP(-B*(R-C)**2)
      IF ( AM1_OR_PM3 )THEN
         ANAM1=zero
         oner2=oneRIJ*oneRIJ

         do I=1,qm2_params%num_fn(qmitype)
            temp_real = RIJ-qm2_params%FN3(i,qmitype)
            temp_real2= qm2_params%FN2(i,qmitype)*temp_real*temp_real

! Skip when the exponential is close enough to be zero
            if (temp_real2 .LT. EXPONENTIAL_CUTOFF) then 
               ANAM1 = ANAM1 + qm2_params%FN1(i,qmitype)*   &
                               EXP(-temp_real2)*   &
                              (oner2 + two*qm2_params%FN2(i,qmitype)*   &
                                           temp_real*oneRIJ)
            end if
         end do

         do i=1,qm2_params%num_fn(qmjtype)
            temp_real=RIJ-qm2_params%FN3(i,qmjtype)
            temp_real2=qm2_params%FN2(i,qmjtype)*temp_real*temp_real

! Skip when the exponential is close enough to be zero
            if (temp_real2 .LT. EXPONENTIAL_CUTOFF) then
               ANAM1 = ANAM1 + qm2_params%FN1(i,qmjtype)*   &
                               EXP(-temp_real2)*   &
                              (oner2+two*qm2_params%FN2(i,qmjtype)*   &
                                         temp_real*oneRIJ)
           end if
         end do

         ANAM1 =-ANAM1*c1
         FNUCX = ANAM1*vec_qm_qm_oneRIJ1
         FNUCY = ANAM1*vec_qm_qm_oneRIJ2
         FNUCZ = ANAM1*vec_qm_qm_oneRIJ3
      else
         FNUCX = zero
         FNUCY = zero
         FNUCZ = zero
      endif

! PDDG specific terms
      if (PDDG_IN_USE) then
            PDDG_EXP1 = -20.0D0 * (RIJ - qm2_params%pddge1(iqm) -    &
                                         qm2_params%pddge1(jqm))*   &
                                         PDDG_EXP1
            PDDG_EXP2 = -20.0D0 * (RIJ - qm2_params%pddge1(iqm) -   &
                                         qm2_params%pddge2(jqm))*   &
                                         PDDG_EXP2
            PDDG_EXP3 = -20.0D0 * (RIJ - qm2_params%pddge2(iqm) -   &
                                         qm2_params%pddge1(jqm))*   &
                                         PDDG_EXP3
            PDDG_EXP4 = -20.0D0 * (RIJ - qm2_params%pddge2(iqm) -   &
                                         qm2_params%pddge2(jqm))*   &
                                         PDDG_EXP4
            PDDG_CORR = qm2_params%PDDG_TERM1(qmitype,qmjtype)*   &
                                   PDDG_EXP1 +   &
                        qm2_params%PDDG_TERM2(qmitype,qmjtype)*   &
                                   PDDG_EXP2 +   &
                        qm2_params%PDDG_TERM3(qmitype,qmjtype)*   &
                                   PDDG_EXP3 +   &
                        qm2_params%PDDG_TERM4(qmitype,qmjtype)*   &
                                   PDDG_EXP4

            FNUCX     = FNUCX + PDDG_CORR*vec_qm_qm_oneRIJ1
            FNUCY     = FNUCY + PDDG_CORR*vec_qm_qm_oneRIJ2
            FNUCZ     = FNUCZ + PDDG_CORR*vec_qm_qm_oneRIJ3
      end if


!***************************************************************
! Step4 : Resonance integrals and Orbital Overlap related terms

! Get the first derivatives of overlap integrals 
! and store in DS

! Loop over atomic orbitals for QM atom i for PX,PY,PZ
! So K=0 = S orbital, K=1 = PX, 2 = PY, 3 = PZ

! 10A cutoff on overlaps
      if (RR2 .LT. OVERLAP_CUTOFF) then 

!a) S-S orbitals: (S/S) terms
         dsx = zero
         dsy = zero
         dsz = zero
         do I=1,6 !1 to NGAUSS
            do J=1,6
               ADBR2=zz_sxs_over_sas(i,j)*RR2

! Check overlap is non-zero
               if (ADBR2 .LT. EXPONENTIAL_CUTOFF) then
                  SS=ss_eqn_adb(i,j)*EXP(-ADBR2)
                  DSx=DSx+vec_qm_qm1*SS
                  DSy=DSy+vec_qm_qm2*SS
                  DSz=DSz+vec_qm_qm3*SS
               end if
            end do
         end do

! Combined together the overlap derivative parts
         orb_offset = 1+qm2_params%pascal_tri1(qm_atomi_orb_start)
         temp_real  = betasas*PSUM(orb_offset)

         FABx       = FABx+temp_real*DSx
         FABy       = FABy+temp_real*DSy
         FABz       = FABz+temp_real*DSz


!b) S-P orbitals if necessary (K=0)
         if (n_atomic_orbj .GT. 1) then
            DSPX  = zero                      ! Zeros entire arrays of 3
            DSPY  = zero
            DSPZ3 = zero

            do j=1,6
               do i=1,6
                  ADBR2=zz_sxp_over_sapij(i,j)*RR2

! Check overlap is non-zero
                  if (ADBR2 .LT. EXPONENTIAL_CUTOFF) then
                     ADBR2  = EXP(-ADBR2)
                     ABN    = sp_eqn_xx1ij(i,j) - (sp_eqn_xx2ij(i,j)*   &
                                                   vec2_qm_qm1)
                     DSPX(1)= DSPX(1)+ABN*ADBR2          ! (S/PX-x) TERM

                     ABN    = sp_eqn_xx1ij(i,j) - (sp_eqn_xx2ij(i,j)*   &
                                                   vec2_qm_qm2)
                     DSPY(2)= DSPY(2)+ABN*ADBR2          ! (S/PY-y) TERM

                     ABN    = sp_eqn_xx1ij(i,j) - (sp_eqn_xx2ij(i,j)*   &
                                                   vec2_qm_qm3)
                     DSPZ3  = DSPZ3+ABN*ADBR2            ! (S/PZ-z) TERM

                     ABN    = ADBR2*sp_eqn_xyij(i,j)
                     DSPX(2)= DSPX(2)+ABN*vec_qm_qm12    ! (PX-y/S) TERM 
                     DSPX(3)= DSPX(3)+ABN*vec_qm_qm13    ! (PX-z/S) TERM
                     DSPY(3)= DSPY(3)+ABN*vec_qm_qm23    ! (PY-z/S) TERM

! DSPY(1)=DSPY(1)-ABN*vec_qm_qm12 DSPY(1)=DSPX(2) 
! (PY-x/S) TERM
! DSPZ(1)=DSPZ(1)-ABN*vec_qm_qm13 DSPZ(1)=DSPX(3) 
! (PZ-x/S) TERM
! DSPZ(2)=DSPZ(2)-ABN*vec_qm_qm23 DSPZ(2)=DSPY(3) 
! (PZ-y/S) TERM
                  end if                    ! (ADBR2<EXPONENTIAL_CUTOFF)
               end do
            end do

! Combined together the overlap derivative parts
            temp_real = betasap*PSUM(orb_offset+1)
            FABx=FABx+temp_real*DSPX(1)
            FABy=FABy+temp_real*DSPX(2)
            FABz=FABz+temp_real*DSPX(3)

            temp_real = betasap*PSUM(orb_offset+2)
            FABx=FABx+temp_real*DSPX(2)          ! DSPY(1)=DSPX(2)
            FABy=FABy+temp_real*DSPY(2)
            FABz=FABz+temp_real*DSPY(3)

            temp_real = betasap*PSUM(orb_offset+3)
            FABx=FABx+temp_real*DSPX(3)          ! DSPZ(1)=DSPX(3)
            FABy=FABy+temp_real*DSPY(3)          ! DSPZ(2)=DSPY(3)
            FABz=FABz+temp_real*DSPZ3
         end if                                  ! if (n_atomic_orbj>1)

!c) P-S orbitals (K>0 and L=0)
         if (n_atomic_orbi .GT. 1) then
            DSPX  = zero                      ! Zeros entire arrays of 3
            DSPY  = zero
            DSPZ3 = zero

            do I=1,6
               do J=1,6
                  ADBR2=zz_sxp_over_sapji(j,i)*RR2

! Check overlap is non-zero
                  if (ADBR2 .LT. EXPONENTIAL_CUTOFF) then 
                     ADBR2  = EXP(-ADBR2)
                     ABN    =-sp_eqn_xx1ji(j,i) + (sp_eqn_xx2ji(j,i)*   &
                                                   vec2_qm_qm1)
                     DSPX(1)= DSPX(1)+ABN*ADBR2          ! (PX-x/S) TERM

                     ABN    =-sp_eqn_xx1ji(j,i) + (sp_eqn_xx2ji(j,i)*   &
                                                   vec2_qm_qm2)
                     DSPY(2)= DSPY(2)+ABN*ADBR2          ! (PY-y/S) TERM

                     ABN    =-sp_eqn_xx1ji(j,i) + (sp_eqn_xx2ji(j,i)*   &
                                                   vec2_qm_qm3)
                     DSPZ3  = DSPZ3+ABN*ADBR2            ! (PZ-z/S) TERM

                     ABN    = ADBR2*sp_eqn_xyji(j,i)
                     DSPX(2)= DSPX(2)-ABN*vec_qm_qm12    ! (PX-y/S) TERM
                     DSPX(3)= DSPX(3)-ABN*vec_qm_qm13    ! (PX-z/S) TERM
                     DSPY(3)= DSPY(3)-ABN*vec_qm_qm23    ! (PY-z/S) TERM
! DSPY(1)=DSPY(1)-ABN*vec_qm_qm12 DSPY(1)=DSPX(2) 
! (PY-x/S) TERM
! DSPZ(1)=DSPZ(1)-ABN*vec_qm_qm13 DSPZ(1)=DSPX(3) 
! (PZ-x/S) TERM
! DSPZ(2)=DSPZ(2)-ABN*vec_qm_qm23 DSPZ(2)=DSPY(3) 
! (PZ-y/S) TERM
                  end if                    ! (ADBR2<EXPONENTIAL_CUTOFF)
               end do
            end do

! Combined together the overlap derivative parts
            temp_real = betapas*PSUM(1 +   &
                        qm2_params%pascal_tri1(qm_atomi_orb_start+1))
            FABx=FABx+temp_real*DSPX(1)
            FABy=FABy+temp_real*DSPX(2)
            FABz=FABz+temp_real*DSPX(3)

            temp_real = betapas*PSUM(1 +   &
                        qm2_params%pascal_tri1(qm_atomi_orb_start+2))
            FABx=FABx+temp_real*DSPX(2)          ! DSPY(1)=DSPX(2)
            FABy=FABy+temp_real*DSPY(2)
            FABz=FABz+temp_real*DSPY(3)

            temp_real = betapas*PSUM(1 +   &
                        qm2_params%pascal_tri1(qm_atomi_orb_start+3))
            FABx=FABx+temp_real*DSPX(3)          ! DSPZ(1)=DSPX(3)
            FABy=FABy+temp_real*DSPY(3)          ! DSPZ(2)=DSPY(3)
            FABz=FABz+temp_real*DSPZ3
         end if                                  ! if (n_atomic_orbi>1)

!d) P-P orbitals
! Ross Walker - PP combinations that are the same have been 
!               condensed to one variable for speed and to
!               save memory.
         if (n_atomic_orbi .GT. 1 .and. n_atomic_orbj .GT. 1) then
            DPXPX  = zero                     ! Zeros entire arrays of 3
            DPYPY  = zero
            DPZPZ  = zero
            DPCROSS= zero

            do I=1,6
               do J=1,6
                  ADBR2=zz_pxp_over_pap(i,j)*RR2

! Check overlap is non-zero
                  if (ADBR2 .LT. EXPONENTIAL_CUTOFF) then 
                     ADBR2   = EXP(-ADBR2)
                     ABN     = ( three*pp_eqn_xxy1(i,j) +   &
                                 pp_eqn_xxy2(i,j) * vec2_qm_qm1 )*ADBR2
                     DPXPX(1)= DPXPX(1)+ABN*vec_qm_qm1   ! (PX / PX) - x

                     ABN     = ( three*pp_eqn_xxy1(i,j) +   &
                                 pp_eqn_xxy2(i,j) * vec2_qm_qm2 )*ADBR2
                     DPYPY(2)= DPYPY(2)+ABN*vec_qm_qm2   ! (PY / PY) - y

                     ABN     = ( three*pp_eqn_xxy1(i,j) +   &
                                 pp_eqn_xxy2(i,j) * vec2_qm_qm3 )*ADBR2
                     DPZPZ(3)= DPZPZ(3)+ABN*vec_qm_qm3   ! (PZ / PZ) - z

                     ABN     = ( pp_eqn_xxy1(i,j) +   &
                                 pp_eqn_xxy2(i,j) * vec2_qm_qm1 )*ADBR2
                     DPXPX(2)= DPXPX(2)+ABN*vec_qm_qm2   ! (PX / PX) - y
                     DPXPX(3)= DPXPX(3)+ABN*vec_qm_qm3   ! (PX / PX) - z
! DPXPY(1)=DPXPY(1)+ABN*vec_qm_qm2 PXPY(1)=PXPX(2)
! (PX / PY) - x
! DPXPZ(1)=DPXPZ(1)+ABN*vec_qm_qm3 PXPZ(1)=PXPX(3)
! (PX / PZ) - x
! DPYPX(1)=DPYPX(1)+ABN*vec_qm_qm2 PXPY=PYPX 
! (PY / PX) - x
! DPZPX(1)=DPZPX(1)+ABN*vec_qm_qm3 PXPZ=PZPX 
! (PZ / PX) - x

                     ABN     = ( pp_eqn_xxy1(i,j) +   &
                                 pp_eqn_xxy2(i,j) * vec2_qm_qm2 )*ADBR2
                     DPYPY(1)= DPYPY(1)+ABN*vec_qm_qm1   ! (PY / PY) - x
                     DPYPY(3)= DPYPY(3)+ABN*vec_qm_qm3   ! (PY / PY) - z
! DPXPY(2)=DPXPY(2)+ABN*vec_qm_qm1 
! PXPY(2)=PYPX(2)=PYPY(1)  ! (PX / PY) - y
! DPYPZ(2)=DPYPZ(2)+ABN*vec_qm_qm3 
! PYPZ(2)=PZPY(2)=PYPY(3)  ! (PY / PZ) - y
! DPYPX(2)=DPYPX(2)+ABN*vec_qm_qm1 
! PXPY=PYPX                ! (PY / PX) - y
! DPZPY(2)=DPZPY(2)+ABN*vec_qm_qm3 
! PYPZ=PZPY                ! (PZ / PY) - y

                     ABN     = ( pp_eqn_xxy1(i,j) +   &
                                 pp_eqn_xxy2(i,j) * vec2_qm_qm3 )*ADBR2
                     DPZPZ(1)= DPZPZ(1)+ABN*vec_qm_qm1   ! (PZ / PZ) - x
                     DPZPZ(2)= DPZPZ(2)+ABN*vec_qm_qm2   ! (PZ / PZ) - y
! DPXPZ(3)=DPXPZ(3)+ABN*vec_qm_qm1 
! PXPZ(3)=PZPX(3)=PZPZ(1)  ! (PX / PZ) - z
! DPYPZ(3)=DPYPZ(3)+ABN*vec_qm_qm2 
! PYPZ(3)=PZPY(3)=PZPZ(2)  ! (PY / PZ) - z
! DPZPY(3)=DPZPY(3)+ABN*vec_qm_qm2 
! PYPZ=PZPY                ! (PZ / PY) - z
! DPZPX(3)=DPZPX(3)+ABN*vec_qm_qm1 
! PXPZ=PZPX                ! (PZ / PX) - z

                     ABN     = pp_eqn_xxy2(i,j) * ADBR2*vec_qm_qm123
                     DPCROSS = DPCROSS+ABN
! DPXPY(3)=DPXPY(3)+ABN 
! PXPY(3)=PYPX(3)=DPCROSS    ! (PX / PY) - z
! DPXPZ(2)=DPXPZ(2)+ABN 
! PXPZ(2)=PZPX(2)=DPCROSS    ! (PX / PZ) - y
! DPYPZ(1)=DPYPZ(1)+ABN 
! PYPZ(1)=PZPY(1)=DPCROSS    ! (PY / PZ) - x
! DPZPY(1)=DPZPY(1)+ABN 
! PYPZ=PZPY                  ! (PZ / PY) - x
! DPYPX(3)=DPYPX(3)+ABN 
! PXPY=PYPX                  ! (PY / PX) - z
! DPZPX(2)=DPZPX(2)+ABN 
! PXPZ=PZPX                  ! (PZ / PX) - y
                  end if                      ! ADBR2>EXPONENTIAL_CUTOFF
               end do
            end do

! Combined together the overlap derivative parts
            orb_offset= 2+qm2_params%pascal_tri1(qm_atomi_orb_start+1)
            temp_real = betapap*PSUM(orb_offset)
            FABx      = FABx+temp_real*DPXPX(1)
            FABy      = FABy+temp_real*DPXPX(2)
            FABz      = FABz+temp_real*DPXPX(3)

            temp_real = betapap*PSUM(orb_offset+1)
            FABx      = FABx+temp_real*DPXPX(2) ! PYPX(1)=PXPY(1)=PXPX(2)
            FABy      = FABy+temp_real*DPYPY(1) ! PXPY(2)=PYPX(2)=PYPY(1)
            FABz      = FABz+temp_real*DPCROSS  ! PXPY(3)=PYPX(3)=DPCROSS

            temp_real = betapap*PSUM(orb_offset+2)
            FABx      = FABx+temp_real*DPXPX(3) ! PXPZ(1)=PZPX(1)=PXPX(3)
            FABy      = FABy+temp_real*DPCROSS  ! PXPZ(2)=PZPX(2)=DPCROSS
            FABz      = FABz+temp_real*DPZPZ(1) ! PXPZ(3)=PZPX(3)=PZPZ(1)

            orb_offset= 2+qm2_params%pascal_tri1(qm_atomi_orb_start+2)
            temp_real = betapap*PSUM(orb_offset)
            FABx      = FABx+temp_real*DPXPX(2) ! PYPX(1)=PXPY(1)=PXPX(2)
            FABy      = FABy+temp_real*DPYPY(1) ! PXPY(2)=PYPX(2)=PYPY(1)
            FABz      = FABz+temp_real*DPCROSS  ! PXPY(3)=PYPX(3)=DPCROSS 

            temp_real = betapap*PSUM(orb_offset+1)
            FABx      = FABx+temp_real*DPYPY(1)
            FABy      = FABy+temp_real*DPYPY(2)
            FABz      = FABz+temp_real*DPYPY(3)

            temp_real = betapap*PSUM(orb_offset+2)
            FABx      = FABx+temp_real*DPCROSS  ! PYPZ(1)=PZPY(1)=DPCROSS
            FABy      = FABy+temp_real*DPYPY(3) ! PYPZ(2)=PZPY(2)=PYPY(3)
            FABz      = FABz+temp_real*DPZPZ(2) ! PYPZ(3)=PZPY(3)=PZPZ(2)

            orb_offset= 2+qm2_params%pascal_tri1(qm_atomi_orb_start+3)
            temp_real = betapap*PSUM(orb_offset)
            FABx      = FABx+temp_real*DPXPX(3) ! PXPZ(1)=PZPX(1)=PXPX(3)
            FABy      = FABy+temp_real*DPCROSS  ! PXPZ(2)=PZPX(2)=DPCROSS
            FABz      = FABz+temp_real*DPZPZ(1) ! PXPZ(3)=PZPX(3)=PZPZ(1)

            temp_real = betapap*PSUM(orb_offset+1)
            FABx      = FABx+temp_real*DPCROSS  ! PYPZ(1)=PZPY(1)=DPCROSS
            FABy      = FABy+temp_real*DPYPY(3) ! PYPZ(2)=PZPY(2)=PYPY(3)
            FABz      = FABz+temp_real*DPZPZ(2) ! PYPZ(3)=PZPY(3)=PZPZ(2)

            temp_real = betapap*PSUM(orb_offset+2)
            FABx      = FABx+temp_real*DPZPZ(1)
            FABy      = FABy+temp_real*DPZPZ(2)
            FABz      = FABz+temp_real*DPZPZ(3)
         end if                ! n_atomic_orbi >1 .and. n_atomic_orbj > 1

      end if                   ! 10A cutoff on overlaps
! End of 10A cutoff on overlaps


! Code is linear from this point on for a given pair.

!***************************************************************
! Step5: Still Resonance integral related terms????

! NOTE: Ross Walker - In the equations below the only thing that
!                     varies for this pair during a simulations
!                     is the values of RR, so we may want to 
!                     pre-computing lots of portions of this.
!                     E.g. For a given pair AEE is constant.

!a) S-S orbital interactions
         TERMX= vec_qm_qm_onerij1*AU_TO_EV*A_TO_BOHRS
         TERMY= vec_qm_qm_onerij2*AU_TO_EV*A_TO_BOHRS
         TERMZ= vec_qm_qm_onerij3*AU_TO_EV*A_TO_BOHRS
         EE    =-RR*(SQRTAEE)**3
         DGX(1)= TERMX*EE
         DGY(1)= TERMY*EE
         DGZ(1)= TERMZ*EE

!b) Heavy atom - (Heavy/light) atom pairs
         if (n_atomic_orbi .GT. 2) then
            ADE   = (bdd2i+bdd1j)**2
            AQE   = (bdd3i+bdd1j)**2

            DZE   = (RR+ddi)/(SQRT((RR+ddi)**2+ADE))**3 -   &
                    (RR-ddi)/(SQRT((RR-ddi)**2+ADE))**3

            temp_real = (two*RR)/(SQRT(RR2+AQE))**3
            QZZE  =-(RR+two*qqi)/(SQRT((RR+two*qqi)**2+AQE))**3 -   &
                    (RR-two*qqi)/(SQRT((RR-two*qqi)**2+AQE))**3 +   &
                     temp_real
            QXXE  =-(two*RR)/(SQRT(RR2+four*qqi2+AQE))**3 +   &
                     temp_real

            temp_real =-half*DZE
            DGX(2)    = TERMX*temp_real
            DGY(2)    = TERMY*temp_real
            DGZ(2)    = TERMZ*temp_real

            temp_real = EE+fourth*QZZE
            DGX(3)    = TERMX*temp_real
            DGY(3)    = TERMY*temp_real
            DGZ(3)    = TERMZ*temp_real

            temp_real = EE+fourth*QXXE
            DGX(4)    = TERMX*temp_real
            DGY(4)    = TERMY*temp_real
            DGZ(4)    = TERMZ*temp_real
         end if

!c) (Heavy/light) atom - Heavy atom pairs
         if (n_atomic_orbj .GT. 2) then
            AED   = (bdd1i+bdd2j)**2
            AEQ   = (bdd1i+bdd3j)**2

            EDZ   = (RR-ddj)/(SQRT((RR-ddj)**2+AED))**3 -   &
                    (RR+ddj)/(SQRT((RR+ddj)**2+AED))**3

            temp_real = (two*RR)/(SQRT(RR2+AEQ))**3
            EQZZ  =-(RR-two*qqj)/(SQRT((RR-two*qqj)**2+AEQ))**3 -   &
                    (RR+two*qqj)/(SQRT((RR+two*qqj)**2+AEQ))**3 +   &
                     temp_real
            EQXX  =-(two*RR)/(SQRT(RR2+four*qqj2+AEQ))**3 +   &
                     temp_real

            temp_real =-half*EDZ
            DGX(5)    = TERMX*temp_real
            DGY(5)    = TERMY*temp_real
            DGZ(5)    = TERMZ*temp_real

            temp_real = EE+fourth*EQZZ
            DGX(11)   = TERMX*temp_real
            DGY(11)   = TERMY*temp_real
            DGZ(11)   = TERMZ*temp_real

            temp_real = EE+fourth*EQXX
            DGX(12)   = TERMX*temp_real
            DGY(12)   = TERMY*temp_real
            DGZ(12)   = TERMZ*temp_real
         end if

!d) Heavy atom - Heavy atom pairs
         if (n_atomic_orbi .GT. 2 .and. n_atomic_orbj .GT. 2) then
           ADD=(bdd2i+bdd2j)**2
           ADQ=(bdd2i+bdd3j)**2
           AQD=(bdd3i+bdd2j)**2
           AQQ=(bdd3i+bdd3j)**2

           DXDX  =-(two*RR)/(SQRT(RR2+(ddi-ddj)**2+ADD))**3   &
                  +(two*RR)/(SQRT(RR2+(ddi+ddj)**2+ADD))**3

           DZDZ  =-(RR+ddi-ddj)/(SQRT((RR+ddi-ddj)**2+ADD))**3   &
                  -(RR-ddi+ddj)/(SQRT((RR-ddi+ddj)**2+ADD))**3   &
                  +(RR-ddi-ddj)/(SQRT((RR-ddi-ddj)**2+ADD))**3   &
                  +(RR+ddi+ddj)/(SQRT((RR+ddi+ddj)**2+ADD))**3

           DZQXX = two*(RR+ddi)/(SQRT((RR+ddi)**2+four*qqj2+ADQ))**3   &
                  -two*(RR-ddi)/(SQRT((RR-ddi)**2+four*qqj2+ADQ))**3   &
                  -two*(RR+ddi)/(SQRT((RR+ddi)**2+ADQ))**3   &
                  +two*(RR-ddi)/(SQRT((RR-ddi)**2+ADQ))**3

           QXXDZ = two*(RR-ddj)/(SQRT((RR-ddj)**2+four*qqi2+AQD))**3   &
                   -two*(RR+ddj)/(SQRT((RR+ddj)**2+four*qqi2+AQD))**3    &
                  -two*(RR-ddj)/(SQRT((RR-ddj)**2+AQD))**3   &
                  +two*(RR+ddj)/(SQRT((RR+ddj)**2+AQD))**3

           DZQZZ = (RR+ddi-two*qqj)/(SQRT((RR+ddi-two*qqj)**2+ADQ))**3   &
                  -(RR-ddi-two*qqj)/(SQRT((RR-ddi-two*qqj)**2+ADQ))**3   &
                  +(RR+ddi+two*qqj)/(SQRT((RR+ddi+two*qqj)**2+ADQ))**3   &
                  -(RR-ddi+two*qqj)/(SQRT((RR-ddi+two*qqj)**2+ADQ))**3   &
                  +two*(RR-ddi)/(SQRT((RR-ddi)**2+ADQ))**3   &
                  -two*(RR+ddi)/(SQRT((RR+ddi)**2+ADQ))**3

           QZZDZ = (RR+two*qqi-ddj)/(SQRT((RR+two*qqi-ddj)**2+AQD))**3   &
                  -(RR+two*qqi+ddj)/(SQRT((RR+two*qqi+ddj)**2+AQD))**3   &
                  +(RR-two*qqi-ddj)/(SQRT((RR-two*qqi-ddj)**2+AQD))**3   &
                  -(RR-two*qqi+ddj)/(SQRT((RR-two*qqi+ddj)**2+AQD))**3   &
                  -two*(RR-ddj)/(SQRT((RR-ddj)**2+AQD))**3   &
                  +two*(RR+ddj)/(SQRT((RR+ddj)**2+AQD))**3

           QXXQXX=-(two*RR)/(SQRT(RR2+four*(qqi-qqj)**2+AQQ))**3   &
                  -(two*RR)/(SQRT(RR2+four*(qqi+qqj)**2+AQQ))**3   &
                  +(four*RR)/(SQRT(RR2+four*qqi2+AQQ))**3   &
                  +(four*RR)/(SQRT(RR2+four*qqj2+AQQ))**3   &
                  -(four*RR)/(SQRT(RR2+AQQ))**3

           QXXQYY=-(four*RR)/(SQRT(RR2+four*qqi2+four*qqj2+AQQ))**3   &
                  +(four*RR)/(SQRT(RR2+four*qqi2+AQQ))**3   &
                  +(four*RR)/(SQRT(RR2+four*qqj2+AQQ))**3   &
                  -(four*RR)/(SQRT(RR2+AQQ))**3

           QXXQZZ=   &
                  -two*(RR-two*qqj)/(SQRT((RR-two*qqj)**2 +   &
                                           four*qqi2+AQQ))**3   &
                  -two*(RR+two*qqj)/(SQRT((RR+two*qqj)**2 +   &
                                           four*qqi2+AQQ))**3   &
                  +two*(RR-two*qqj)/(SQRT((RR-two*qqj)**2+AQQ))**3   &
                  +two*(RR+two*qqj)/(SQRT((RR+two*qqj)**2+AQQ))**3   &
                  +(four*RR)/(SQRT(RR2+four*qqi2+AQQ))**3   &
                  -(four*RR)/(SQRT(RR2+AQQ))**3

           QZZQXX=   &
                  -two*(RR+two*qqi)/(SQRT((RR+two*qqi)**2 +   &
                                           four*qqj2+AQQ))**3   &
                  -two*(RR-two*qqi)/(SQRT((RR-two*qqi)**2 +   &
                                           four*qqj2+AQQ))**3   &
                  +two*(RR+two*qqi)/(SQRT((RR+two*qqi)**2+AQQ))**3   &
                  +two*(RR-two*qqi)/(SQRT((RR-two*qqi)**2+AQQ))**3   &
                  +(four*RR)/(SQRT(RR2+four*qqj2+AQQ))**3   &
                  -(four*RR)/(SQRT(RR2+AQQ))**3

           QZZQZZ=   &
                -(RR+two*qqi-two*qqj)/(SQRT((RR+two*qqi-two*qqj)**2+   &
                                             AQQ))**3   &
                -(RR+two*qqi+two*qqj)/(SQRT((RR+two*qqi+two*qqj)**2+   &
                                             AQQ))**3   &
                -(RR-two*qqi-two*qqj)/(SQRT((RR-two*qqi-two*qqj)**2+   &
                                             AQQ))**3   &
                -(RR-two*qqi+two*qqj)/(SQRT((RR-two*qqi+two*qqj)**2+   &
                                             AQQ))**3   &
                  +two*(RR-two*qqi)/(SQRT((RR-two*qqi)**2+AQQ))**3   &
                  +two*(RR+two*qqi)/(SQRT((RR+two*qqi)**2+AQQ))**3   &
                  +two*(RR-2.D0*qqj)/(SQRT((RR-2.D0*qqj)**2+AQQ))**3   &
                  +two*(RR+2.D0*qqj)/(SQRT((RR+2.D0*qqj)**2+AQQ))**3   &
                  -(four*RR)/(SQRT(RR2+AQQ))**3

           DXQXZ = two*(RR-qqj)/(SQRT((RR-qqj)**2+(ddi-qqj)**2+   &
                                       ADQ))**3   &
                  -two*(RR+qqj)/(SQRT((RR+qqj)**2+(ddi-qqj)**2+   &
                                       ADQ))**3   &
                  -two*(RR-qqj)/(SQRT((RR-qqj)**2+(ddi+qqj)**2+   &
                                       ADQ))**3   &
                  +two*(RR+qqj)/(SQRT((RR+qqj)**2+(ddi+qqj)**2+   &
                                       ADQ))**3

           QXZDX = two*(RR+qqi)/(SQRT((RR+qqi)**2+(qqi-ddj)**2+   &
                                       AQD))**3   &
                  -two*(RR-qqi)/(SQRT((RR-qqi)**2+(qqi-ddj)**2+   &
                                       AQD))**3   &
                  -two*(RR+qqi)/(SQRT((RR+qqi)**2+(qqi+ddj)**2+   &
                                       AQD))**3   &
                  +two*(RR-qqi)/(SQRT((RR-qqi)**2+(qqi+ddj)**2+   &
                                       AQD))**3

           QXZQXZ=-two*(RR+qqi-qqj)/(SQRT((RR+qqi-qqj)**2+   &
                                          (qqi-qqj)**2+AQQ))**3   &
                  +two*(RR+qqi+qqj)/(SQRT((RR+qqi+qqj)**2+   &
                                          (qqi-qqj)**2+AQQ))**3   &
                  +two*(RR-qqi-qqj)/(SQRT((RR-qqi-qqj)**2+   &
                                          (qqi-qqj)**2+AQQ))**3   &
                  -two*(RR-qqi+qqj)/(SQRT((RR-qqi+qqj)**2+   &
                                          (qqi-qqj)**2+AQQ))**3   &
                  +two*(RR+qqi-qqj)/(SQRT((RR+qqi-qqj)**2+   &
                                          (qqi+qqj)**2+AQQ))**3   &
                  -two*(RR+qqi+qqj)/(SQRT((RR+qqi+qqj)**2+   &
                                          (qqi+qqj)**2+AQQ))**3   &
                  -two*(RR-qqi-qqj)/(SQRT((RR-qqi-qqj)**2+   &
                                          (qqi+qqj)**2+AQQ))**3   &
                  +two*(RR-qqi+qqj)/(SQRT((RR-qqi+qqj)**2+   &
                                          (qqi+qqj)**2+AQQ))**3

           temp_real = DZDZ*fourth
           DGX(6)    = TERMX*temp_real
           DGY(6)    = TERMY*temp_real
           DGZ(6)    = TERMZ*temp_real

           temp_real = DXDX*fourth
           DGX(7)    = TERMX*temp_real
           DGY(7)    = TERMY*temp_real
           DGZ(7)    = TERMZ*temp_real

           temp_real =-(EDZ*half+QZZDZ*eighth)
           DGX(8)    = TERMX*temp_real
           DGY(8)    = TERMY*temp_real
           DGZ(8)    = TERMZ*temp_real

           temp_real =-(EDZ*half+QXXDZ*eighth)
           DGX(9)    = TERMX*temp_real
           DGY(9)    = TERMY*temp_real
           DGZ(9)    = TERMZ*temp_real

           temp_real =-QXZDX*eighth
           DGX(10)   = TERMX*temp_real
           DGY(10)   = TERMY*temp_real
           DGZ(10)   = TERMZ*temp_real

           temp_real =-(DZE*half+DZQZZ*eighth) 
           DGX(13)   = TERMX*temp_real
           DGY(13)   = TERMY*temp_real
           DGZ(13)   = TERMZ*temp_real

           temp_real =-(DZE*half+DZQXX*eighth)
           DGX(14)   = TERMX*temp_real
           DGY(14)   = TERMY*temp_real
           DGZ(14)   = TERMZ*temp_real

           temp_real =-DXQXZ*eighth
           DGX(15)   = TERMX*temp_real
           DGY(15)   = TERMY*temp_real
           DGZ(15)   = TERMZ*temp_real

           temp_real2= EE+EQZZ*fourth
           temp_real = temp_real2+QZZE*fourth+QZZQZZ*sixteenth
           DGX(16)   = TERMX*temp_real
           DGY(16)   = TERMY*temp_real
           DGZ(16)   = TERMZ*temp_real

           temp_real = temp_real2+QXXE*fourth+QXXQZZ*sixteenth
           DGX(17)   = TERMX*temp_real
           DGY(17)   = TERMY*temp_real
           DGZ(17)   = TERMZ*temp_real

           temp_real2= EE+EQXX*fourth
           temp_real = temp_real2+QZZE*fourth+QZZQXX*sixteenth
           DGX(18)   = TERMX*temp_real
           DGY(18)   = TERMY*temp_real
           DGZ(18)   = TERMZ*temp_real

           temp_real = temp_real2+QXXE*fourth+QXXQXX*sixteenth
           DGX(19)   = TERMX*temp_real
           DGY(19)   = TERMY*temp_real
           DGZ(19)   = TERMZ*temp_real

           temp_real = QXZQXZ*sixteenth
           DGX(20)   = TERMX*temp_real
           DGY(20)   = TERMY*temp_real
           DGZ(20)   = TERMZ*temp_real

           temp_real = temp_real2+QXXE*fourth+QXXQYY*sixteenth
           DGX(21)   = TERMX*temp_real
           DGY(21)   = TERMY*temp_real
           DGZ(21)   = TERMZ*temp_real

           temp_real = (QXXQXX-QXXQYY)*thirtysecond
           DGX(22)   = TERMX*temp_real
           DGY(22)   = TERMY*temp_real
           DGZ(22)   = TERMZ*temp_real
         end if                            ! (n_atomic_orbi .GT. 2 .and. 
!!                                         !  n_atomic_orbj .GT. 2)


!***************************************************************
! Step6: Two-e integrals related terms 
 
      if (n_atomic_orbi .GT. 1 .OR. n_atomic_orbj .GT.1) then

! Heavy atom

         RXY2  = vec2_qm_qm1 + vec2_qm_qm2
         RYZ2  = vec2_qm_qm2 + vec2_qm_qm3
         RZX2  = vec2_qm_qm3 + vec2_qm_qm1

         LRXY2 = (RXY2 .LT. AXIS_TOL)
         LRYZ2 = (RYZ2 .LT. AXIS_TOL)
         LRZX2 = (RZX2 .LT. AXIS_TOL)

         XTDX  = zero
         YTDX  = zero
         ZTDX  = zero

         XTDY  = zero
         YTDY  = zero
         ZTDY  = zero

         XTDZ  = zero
         YTDZ  = zero
         ZTDZ  = zero

! Depends on molecular orientation wrt diatomic Z axis
!         : molecular diatomic rotations
         if (.NOT.(LRXY2 .OR. LRYZ2 .OR. LRZX2)) then

! Molecular X/Y/Z axis is not parallel to diatomic Z axis
            TERMX = vec_qm_qm_onerij1 * oneRIJ    ! vec_qm_qm1/(RIJ*RIJ)
            TERMY = vec_qm_qm_onerij2 * oneRIJ
            TERMZ = vec_qm_qm_onerij3 * oneRIJ
            oneRXY= one/sqrt(RXY2)

!Ross Walker - rearranged order here slightly for speed.
            TZ(3) = oneRIJ/oneRXY
            TZ3i  = RIJ * oneRXY                 ! Inverse of TZ(3) to 
!                                                ! avoid other divisions.
            TZ3i2 = TZ3i*TZ3i                    ! Square of 1/TZ(3)

            TX(1) = vec_qm_qm_onerij1
            TX(2) = vec_qm_qm_onerij2
            TX(3) = vec_qm_qm_onerij3

            TY(1) =-TX(2)*SIGN(one,TX(1)) * TZ3i
            TY(2) = ABS(TX(1) * TZ3i)
            TY(3) = zero

            TZ(1) =-TX(1)*TX(3)*TZ3i
            TZ(2) =-TX(2)*TX(3)*TZ3i
! TZ(3) = see above

            XTDX(1)= oneRIJ-TX(1)*TERMX
            YTDX(1)=-TX(1)*TERMY
            ZTDX(1)=-TX(1)*TERMZ

            XTDX(2)=-TX(2)*TERMX
            YTDX(2)= oneRIJ-TX(2)*TERMY
            ZTDX(2)=-TX(2)*TERMZ

            XTDX(3)=-TX(3)*TERMX
            YTDX(3)=-TX(3)*TERMY
            ZTDX(3)= oneRIJ-TX(3)*TERMZ

            XTDZ(3)= TX(1)*oneRXY-TZ(3)*TERMX
            YTDZ(3)= TX(2)*oneRXY-TZ(3)*TERMY
            ZTDZ(3)=-TZ(3)*TERMZ

            XTDY(1)=-XTDX(2)*TZ3i+TX(2)*XTDZ(3)*TZ3i2
            YTDY(1)=-YTDX(2)*TZ3i+TX(2)*YTDZ(3)*TZ3i2
            ZTDY(1)=-ZTDX(2)*TZ3i+TX(2)*ZTDZ(3)*TZ3i2

            XTDY(1)= XTDY(1)*sign(one,TX(1))
            YTDY(1)= YTDY(1)*sign(one,TX(1))
            ZTDY(1)= ZTDY(1)*sign(one,TX(1))

            XTDY(2)= XTDX(1)*TZ3I-TX(1)*XTDZ(3)*TZ3I2
            YTDY(2)= YTDX(1)*TZ3I-TX(1)*YTDZ(3)*TZ3I2
            ZTDY(2)= ZTDX(1)*TZ3I-TX(1)*ZTDZ(3)*TZ3I2

            XTDY(2)= XTDY(2)*sign(one,TX(1))
            YTDY(2)= YTDY(2)*sign(one,TX(1))
            ZTDY(2)= ZTDY(2)*sign(one,TX(1))

! Don't need to zero these again as they were zeroed above
!       XTDY(3)=0.0D0
!       YTDY(3)=0.0D0
!       ZTDY(3)=0.0D0

! Note: Ross Walker and Mike Crowley, we could factor out 
!       TZ3I here or we could pre-compute -TX(3)*TZ3I etc.
!       But, for the moment we will leave it as is since
!       this is really just doing the compiler's work.
            XTDZ(1)=-TX(3)*XTDX(1)*TZ3I-TX(1)*XTDX(3)*TZ3I   &
                    +TX(1)*TX(3)*XTDZ(3)*TZ3I2
            YTDZ(1)=-TX(3)*YTDX(1)*TZ3I-TX(1)*YTDX(3)*TZ3I   &
                    +TX(1)*TX(3)*YTDZ(3)*TZ3I2
            ZTDZ(1)=-TX(3)*ZTDX(1)*TZ3I-TX(1)*ZTDX(3)*TZ3I   &
                    +TX(1)*TX(3)*ZTDZ(3)*TZ3I2

            XTDZ(2)=-TX(3)*XTDX(2)*TZ3I-TX(2)*XTDX(3)*TZ3I   &
                    +TX(2)*TX(3)*XTDZ(3)*TZ3I2
            YTDZ(2)=-TX(3)*YTDX(2)*TZ3I-TX(2)*YTDX(3)*TZ3I   &
                    +TX(2)*TX(3)*YTDZ(3)*TZ3I2
            ZTDZ(2)=-TX(3)*ZTDX(2)*TZ3I-TX(2)*ZTDX(3)*TZ3I   &
                    +TX(2)*TX(3)*ZTDZ(3)*TZ3I2

         else if (LRXY2) then
! Molecular Z axis is parallel to diatomix Z aixs
            TX(1) = zero
            TX(2) = zero
            TX(3) = sign(one,vec_qm_qm3)
            TY(1) = zero
            TY(2) = one
            TY(3) = zero
            TZ(1) = TX(3)
            TZ(2) = zero
            TZ(3) = zero

            XTDX(1) = oneRIJ                  ! X axis
            XTDZ(3) =-oneRIJ
            YTDX(2) = oneRIJ                  ! Y axis
            YTDY(3) =-TX(3)*oneRIJ
!                                             ! Z axis
         else if (LRYZ2) then
! Molecular X axis is parallel to diatomic Z axis
            TX(1) = sign(one,vec_qm_qm1)
            TX(2) = zero
            TX(3) = zero
            TY(1) = zero
            TY(2) = TX(1)
            TY(3) = zero
            TZ(1) = zero
            TZ(2) = zero
            TZ(3) = one
!                                             ! X axis
            YTDX(2) = oneRIJ                  ! Y axis
            YTDY(1) =-oneRIJ
            ZTDX(3) = oneRIJ                  ! Z axis
            ZTDZ(1) =-TX(1)*oneRIJ
         else
! Molecular Y axis is parallel to diatomic Z axis
            TX(1) = zero
            TX(2) = sign(one,vec_qm_qm2)
            TX(3) = zero
            TY(1) =-TX(2)
            TY(2) = zero
            TY(3) = zero
            TZ(1) = zero
            TZ(2) = zero
            TZ(3) = one
            XTDX(1) = oneRIJ                  ! X axis
            XTDY(2) = oneRIJ
!                                             ! Y axis
            ZTDX(3) = oneRIJ                  ! Z axis
            ZTDZ(2) =-TX(2)*oneRIJ
         end if                               ! if (LRZX2) then

! Gradients from 2-e integrals:
!     At least one atom has more than one atomic orbital so 
!     need to consider S and P interactions.
!
         isp = 0
         do K=qm_atomi_orb_start,total_atomic_orb
            KK=K-qm_atomi_orb_start
            do L=K,total_atomic_orb
               LL=L-qm_atomi_orb_start
               do M=1,n_atomic_orbj
                  MM=M-1
                  do N=M,n_atomic_orbj
                     NN=N-1
                     ISP=ISP+1

                     IF (NN .EQ. 0)THEN
                        IF (LL .EQ. 0) THEN
! (SS/SS)
                        DRX(ISP)=DGX(1)
                        DRY(ISP)=DGY(1)
                        DRZ(ISP)=DGZ(1)
!                           
                        ELSE IF(KK .EQ. 0) THEN
! (SP/SS)
                        DRX(ISP)=DGX(2)*TX(LL)+qm_qm_e_repul(2)*xTDX(LL)
                        DRY(ISP)=DGY(2)*TX(LL)+qm_qm_e_repul(2)*yTDX(LL)
                        DRZ(ISP)=DGZ(2)*TX(LL)+qm_qm_e_repul(2)*zTDX(LL)
!                           
                        ELSE
! (PP/SS)
                        DRX(ISP)=DGX(3)*TX(KK)*TX(LL)+qm_qm_e_repul(3)   &
                               *(xTDX(KK)*TX(LL)+TX(KK)*xTDX(LL))   &
                                +DGX(4)*(TY(KK)*TY(LL)+TZ(KK)*TZ(LL))   &
                                +qm_qm_e_repul(4)*(xTDY(KK)*TY(LL)   &
                                +TY(KK)*xTDY(LL)+xTDZ(KK)*TZ(LL)   &
                                +TZ(KK)*xTDZ(LL))
                        DRY(ISP)=DGY(3)*TX(KK)*TX(LL)+qm_qm_e_repul(3)   &
                                *(yTDX(KK)*TX(LL)+TX(KK)*yTDX(LL))   &
                                 +DGY(4)*(TY(KK)*TY(LL)+TZ(KK)*TZ(LL))   &
                                 +qm_qm_e_repul(4)*(yTDY(KK)*TY(LL)   &
                                 +TY(KK)*yTDY(LL)+yTDZ(KK)*TZ(LL)   &
                                 +TZ(KK)*yTDZ(LL))
                        DRZ(ISP)=DGZ(3)*TX(KK)*TX(LL)+qm_qm_e_repul(3)   &
                               *(zTDX(KK)*TX(LL)+TX(KK)*zTDX(LL))   &
                                +DGZ(4)*(TY(KK)*TY(LL)+TZ(KK)*TZ(LL))   &
                                +qm_qm_e_repul(4)*(zTDY(KK)*TY(LL)   &
                                +TY(KK)*zTDY(LL)+zTDZ(KK)*TZ(LL)   &
                                +TZ(KK)*zTDZ(LL))
!                           
                        END IF

                     ELSE IF (MM .EQ. 0) THEN
                        IF (LL .EQ. 0) THEN
! (SS/SP)
                        DRX(ISP)=DGX(5)*TX(NN)+qm_qm_e_repul(5)*xTDX(NN)
                        DRY(ISP)=DGY(5)*TX(NN)+qm_qm_e_repul(5)*yTDX(NN)
                        DRZ(ISP)=DGZ(5)*TX(NN)+qm_qm_e_repul(5)*zTDX(NN)
!                           
                        ELSE IF(KK .EQ. 0) THEN
! (SP/SP)
                        DRX(ISP)=DGX(6)*TX(LL)*TX(NN)+qm_qm_e_repul(6)   &
                                *(xTDX(LL)*TX(NN)+TX(LL)*xTDX(NN))   &
                                 +DGX(7)*(TY(LL)*TY(NN)+TZ(LL)*TZ(NN))   &
                                 +qm_qm_e_repul(7)*(xTDY(LL)*TY(NN)   &
                                 +TY(LL)*xTDY(NN)   &
                                 +xTDZ(LL)*TZ(NN)+TZ(LL)*xTDZ(NN))
                        DRY(ISP)=DGY(6)*TX(LL)*TX(NN)+qm_qm_e_repul(6)   &
                               *(yTDX(LL)*TX(NN)+TX(LL)*yTDX(NN))   &
                                +DGY(7)*(TY(LL)*TY(NN)+TZ(LL)*TZ(NN))   &
                                +qm_qm_e_repul(7)*(yTDY(LL)*TY(NN)   &
                                +TY(LL)*yTDY(NN)   &
                                +yTDZ(LL)*TZ(NN)+TZ(LL)*yTDZ(NN))
                        DRZ(ISP)=DGZ(6)*TX(LL)*TX(NN)+qm_qm_e_repul(6)   &
                               *(zTDX(LL)*TX(NN)+TX(LL)*zTDX(NN))   &
                                +DGZ(7)*(TY(LL)*TY(NN)+TZ(LL)*TZ(NN))   &
                                +qm_qm_e_repul(7)*(zTDY(LL)*TY(NN)   &
                                +TY(LL)*zTDY(NN)   &
                                +zTDZ(LL)*TZ(NN)+TZ(LL)*zTDZ(NN))
!                           
                        ELSE
! (PP/SP)
      DRX(ISP)=DGX(8)*TX(KK)*TX(LL)*TX(NN)+qm_qm_e_repul(8)*(xTDX(KK)   &
              *TX(LL)*TX(NN)+TX(KK)*xTDX(LL)*TX(NN)+TX(KK)*TX(LL)   &
              *xTDX(NN))+DGX(9)*(TY(KK)*TY(LL)+TZ(KK)*TZ(LL))*TX(NN)   &
              +qm_qm_e_repul(9)*((xTDY(KK)*TY(LL)+TY(KK)*xTDY(LL)   &
              +xTDZ(KK)*TZ(LL)+TZ(KK)*xTDZ(LL))*TX(NN)+(TY(KK)*TY(LL)   &
              +TZ(KK)*TZ(LL))*xTDX(NN))+DGX(10)*(TX(KK)*(TY(LL)*TY(NN)   &
              +TZ(LL)*TZ(NN))+TX(LL)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN)))   &
              +qm_qm_e_repul(10)*(xTDX(KK)*(TY(LL)*TY(NN)+TZ(LL)   &
              *TZ(NN))+xTDX(LL)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN))+TX(KK)   &
              *(xTDY(LL)*TY(NN)+TY(LL)*xTDY(NN)+xTDZ(LL)*TZ(NN)+TZ(LL)   &
              *xTDZ(NN))+TX(LL)*(xTDY(KK)*TY(NN)+TY(KK)*xTDY(NN)    &
              +xTDZ(KK)*TZ(NN)+TZ(KK)*xTDZ(NN)))
      DRY(ISP)=DGY(8)*TX(KK)*TX(LL)*TX(NN)+qm_qm_e_repul(8)*(yTDX(KK)   &
              *TX(LL)*TX(NN)+TX(KK)*yTDX(LL)*TX(NN)+TX(KK)*TX(LL)   &
              *yTDX(NN))+DGY(9)*(TY(KK)*TY(LL)+TZ(KK)*TZ(LL))*TX(NN)   &
              +qm_qm_e_repul(9)*((yTDY(KK)*TY(LL)+TY(KK)*yTDY(LL)   &
              +yTDZ(KK)*TZ(LL)+TZ(KK)*yTDZ(LL))*TX(NN)+(TY(KK)*TY(LL)   &
              +TZ(KK)*TZ(LL))*yTDX(NN))+DGY(10)*(TX(KK)*(TY(LL)*TY(NN)   &
              +TZ(LL)*TZ(NN))+TX(LL)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN)))   &
              +qm_qm_e_repul(10)*(yTDX(KK)*(TY(LL)*TY(NN)+TZ(LL)   &
              *TZ(NN))+yTDX(LL)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN))+TX(KK)   &
              *(yTDY(LL)*TY(NN)+TY(LL)*yTDY(NN)+yTDZ(LL)*TZ(NN)+TZ(LL)   &
              *yTDZ(NN))+TX(LL)*(yTDY(KK)*TY(NN)+TY(KK)*yTDY(NN)   &
              +yTDZ(KK)*TZ(NN)+TZ(KK)*yTDZ(NN)))
      DRZ(ISP)=DGZ(8)*TX(KK)*TX(LL)*TX(NN)+qm_qm_e_repul(8)*(zTDX(KK)   &
              *TX(LL)*TX(NN)+TX(KK)*zTDX(LL)*TX(NN)+TX(KK)*TX(LL)   &
              *zTDX(NN))+DGZ(9)*(TY(KK)*TY(LL)+TZ(KK)*TZ(LL))*TX(NN)   &
              +qm_qm_e_repul(9)*((zTDY(KK)*TY(LL)+TY(KK)*zTDY(LL)   &
              +zTDZ(KK)*TZ(LL)+TZ(KK)*zTDZ(LL))*TX(NN)+(TY(KK)*TY(LL)   &
              +TZ(KK)*TZ(LL))*zTDX(NN))+DGZ(10)*(TX(KK)*(TY(LL)*TY(NN)   &
              +TZ(LL)*TZ(NN))+TX(LL)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN)))   &
              +qm_qm_e_repul(10)*(zTDX(KK)*(TY(LL)*TY(NN)+TZ(LL)   &
              *TZ(NN))+zTDX(LL)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN))+TX(KK)   &
              *(zTDY(LL)*TY(NN)+TY(LL)*zTDY(NN)+zTDZ(LL)*TZ(NN)+TZ(LL)   &
              *zTDZ(NN))+TX(LL)*(zTDY(KK)*TY(NN)+TY(KK)*zTDY(NN)   &
              +zTDZ(KK)*TZ(NN)+TZ(KK)*zTDZ(NN)))
!                           
                        END IF

                     ELSE IF (LL .EQ. 0) THEN
! (SS/PP)
                     DRX(ISP)=DGX(11)*TX(MM)*TX(NN)+qm_qm_e_repul(11)   &
                             *(xTDX(MM)*TX(NN)+TX(MM)*xTDX(NN))   &
                             +DGX(12)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))   &
                             +qm_qm_e_repul(12)*(xTDY(MM)*TY(NN)   &
                             +TY(MM)*xTDY(NN)+xTDZ(MM)*TZ(NN)   &
                             +TZ(MM)*xTDZ(NN))
                     DRY(ISP)=DGY(11)*TX(MM)*TX(NN)+qm_qm_e_repul(11)   &
                             *(yTDX(MM)*TX(NN)+TX(MM)*yTDX(NN))   &
                             +DGY(12)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))   &
                             +qm_qm_e_repul(12)*(yTDY(MM)*TY(NN)   &
                             +TY(MM)*yTDY(NN)+yTDZ(MM)*TZ(NN)   &
                             +TZ(MM)*yTDZ(NN))
                     DRZ(ISP)=DGZ(11)*TX(MM)*TX(NN)+qm_qm_e_repul(11)   &
                             *(zTDX(MM)*TX(NN)+TX(MM)*zTDX(NN))   &
                             +DGZ(12)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))   &
                             +qm_qm_e_repul(12)*(zTDY(MM)*TY(NN)   &
                             +TY(MM)*zTDY(NN)+zTDZ(MM)*TZ(NN)   &
                             +TZ(MM)*zTDZ(NN))
!                        
                     ELSE IF (KK .EQ. 0) THEN
! (SP/PP)
      DRX(ISP)=DGX(13)*TX(LL)*TX(MM)*TX(NN)+   &
               qm_qm_e_repul(13)*(xTDX(LL)*TX(MM)   &
              *TX(NN)+TX(LL)*xTDX(MM)*TX(NN)+   &
               TX(LL)*TX(MM)*xTDX(NN))+DGX(14)     &
              *TX(LL)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))+   &
               qm_qm_e_repul(14)         &
              *(xTDX(LL)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))+   &
               TX(LL)*(xTDY(MM)*TY(NN)  &
              +TY(MM)*xTDY(NN)+xTDZ(MM)*TZ(NN)+   &
               TZ(MM)*xTDZ(NN)))+DGX(15)*(TY(LL)  &
              *(TY(MM)*TX(NN)+TY(NN)*TX(MM))+   &
               TZ(LL)*(TZ(MM)*TX(NN)+TZ(NN)     &
              *TX(MM)))+qm_qm_e_repul(15)   &
              *(xTDY(LL)*(TY(MM)*TX(NN)+TY(NN)      &
              *TX(MM))+xTDZ(LL)*(TZ(MM)*TX(NN)+   &
               TZ(NN)*TX(MM))+TY(LL)*(xTDY(MM)  &
              *TX(NN)+TY(MM)*xTDX(NN)+xTDY(NN)*TX(MM)+   &
               TY(NN)*xTDX(MM))+TZ(LL)    &
              *(xTDZ(MM)*TX(NN)+TZ(MM)*xTDX(NN)+   &
               xTDZ(NN)*TX(MM)+TZ(NN)*xTDX(MM)))
      DRY(ISP)=DGY(13)*TX(LL)*TX(MM)*TX(NN)+   &
               qm_qm_e_repul(13)*(yTDX(LL)*TX(MM)   &
              *TX(NN)+TX(LL)*yTDX(MM)*TX(NN)+   &
               TX(LL)*TX(MM)*yTDX(NN))+DGY(14)     &
              *TX(LL)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))+   &
               qm_qm_e_repul(14)         &
              *(yTDX(LL)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))+   &
               TX(LL)*(yTDY(MM)*TY(NN)  &
              +TY(MM)*yTDY(NN)+yTDZ(MM)*TZ(NN)+   &
               TZ(MM)*yTDZ(NN)))+DGY(15)*(TY(LL)  &
              *(TY(MM)*TX(NN)+TY(NN)*TX(MM))+   &
               TZ(LL)*(TZ(MM)*TX(NN)+TZ(NN)     &
              *TX(MM)))+qm_qm_e_repul(15)*(yTDY(LL)   &
              *(TY(MM)*TX(NN)+TY(NN)      &
              *TX(MM))+yTDZ(LL)*(TZ(MM)*TX(NN)+   &
               TZ(NN)*TX(MM))+TY(LL)*(yTDY(MM)  &
              *TX(NN)+TY(MM)*yTDX(NN)+yTDY(NN)*TX(MM)+   &
               TY(NN)*yTDX(MM))+TZ(LL)    &
              *(yTDZ(MM)*TX(NN)+TZ(MM)*yTDX(NN)+   &
               yTDZ(NN)*TX(MM)+TZ(NN)*yTDX(MM)))
      DRZ(ISP)=DGZ(13)*TX(LL)*TX(MM)*TX(NN)+   &
               qm_qm_e_repul(13)*(zTDX(LL)*TX(MM)   &
              *TX(NN)+TX(LL)*zTDX(MM)*TX(NN)+   &
               TX(LL)*TX(MM)*zTDX(NN))+DGZ(14)     &
              *TX(LL)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))+   &
               qm_qm_e_repul(14)         &
              *(zTDX(LL)*(TY(MM)*TY(NN)+TZ(MM)*TZ(NN))+   &
                TX(LL)*(zTDY(MM)*TY(NN)  &
              +TY(MM)*zTDY(NN)+zTDZ(MM)*TZ(NN)+   &
               TZ(MM)*zTDZ(NN)))+DGZ(15)*(TY(LL)  &
              *(TY(MM)*TX(NN)+TY(NN)*TX(MM))+   &
               TZ(LL)*(TZ(MM)*TX(NN)+TZ(NN)     &
              *TX(MM)))+qm_qm_e_repul(15)*(zTDY(LL)   &
              *(TY(MM)*TX(NN)+TY(NN)      &
              *TX(MM))+zTDZ(LL)*(TZ(MM)*TX(NN)+   &
               TZ(NN)*TX(MM))+TY(LL)*(zTDY(MM)  &
              *TX(NN)+TY(MM)*zTDX(NN)+zTDY(NN)*TX(MM)+   &
               TY(NN)*zTDX(MM))+TZ(LL)    &
              *(zTDZ(MM)*TX(NN)+TZ(MM)*zTDX(NN)+   &
                zTDZ(NN)*TX(MM)+TZ(NN)*zTDX(MM)))
!
                     ELSE
! (PP/PP)
      DRX(ISP)=DGX(16)*TX(KK)*TX(LL)*TX(MM)*TX(NN)+   &
               qm_qm_e_repul(16)*(xTDX(KK)*TX(LL)*TX(MM)       &
              *TX(NN)+TX(KK)*xTDX(LL)*TX(MM)*TX(NN)+   &
               TX(KK)*TX(LL)*xTDX(MM)*TX(NN)+TX(KK)*TX(LL)   &
              *TX(MM)*xTDX(NN))+DGX(17)*(TY(KK)*TY(LL)+   &
               TZ(KK)*TZ(LL))*TX(MM)*TX(NN)               &
              +qm_qm_e_repul(17)*((xTDY(KK)*TY(LL)+   &
               TY(KK)*xTDY(LL)+xTDZ(KK)*TZ(LL)+TZ(KK)          &
              *xTDZ(LL))*TX(MM)*TX(NN)+(TY(KK)*TY(LL)+   &
               TZ(KK)*TZ(LL))*(xTDX(MM)*TX(NN)+TX(MM)      &
              *xTDX(NN)))+DGX(18)*TX(KK)*TX(LL)*(TY(MM)*TY(NN)+   &
               TZ(MM)*TZ(NN))+qm_qm_e_repul(18)   &
              *((xTDX(KK)*TX(LL)+TX(KK)*xTDX(LL))*(TY(MM)*TY(NN)+   &
               TZ(MM)*TZ(NN))+TX(KK)*TX(LL)     &
              *(xTDY(MM)*TY(NN)+TY(MM)*xTDY(NN)+   &
               xTDZ(MM)*TZ(NN)+TZ(MM)*xTDZ(NN)))
      DRY(ISP)=DGY(16)*TX(KK)*TX(LL)*TX(MM)*TX(NN)+   &
               qm_qm_e_repul(16)*(yTDX(KK)*TX(LL)*TX(MM)       &
              *TX(NN)+TX(KK)*yTDX(LL)*TX(MM)*TX(NN)+   &
               TX(KK)*TX(LL)*yTDX(MM)*TX(NN)+TX(KK)*TX(LL)   &
              *TX(MM)*yTDX(NN))+DGY(17)*(TY(KK)*TY(LL)+   &
               TZ(KK)*TZ(LL))*TX(MM)*TX(NN)               &
              +qm_qm_e_repul(17)*((yTDY(KK)*TY(LL)+   &
               TY(KK)*yTDY(LL)+yTDZ(KK)*TZ(LL)+TZ(KK)          &
              *yTDZ(LL))*TX(MM)*TX(NN)+(TY(KK)*TY(LL)+   &
               TZ(KK)*TZ(LL))*(yTDX(MM)*TX(NN)+TX(MM)      &
              *yTDX(NN)))+DGY(18)*TX(KK)*TX(LL)*(TY(MM)*TY(NN)+   &
               TZ(MM)*TZ(NN))+qm_qm_e_repul(18)   &
              *((yTDX(KK)*TX(LL)+TX(KK)*yTDX(LL))*(TY(MM)*TY(NN)+   &
               TZ(MM)*TZ(NN))+TX(KK)*TX(LL)     &
              *(yTDY(MM)*TY(NN)+TY(MM)*yTDY(NN)+   &
                yTDZ(MM)*TZ(NN)+TZ(MM)*yTDZ(NN)))
      DRZ(ISP)=DGZ(16)*TX(KK)*TX(LL)*TX(MM)*TX(NN)+   &
               qm_qm_e_repul(16)*(zTDX(KK)*TX(LL)*TX(MM)       &
              *TX(NN)+TX(KK)*zTDX(LL)*TX(MM)*TX(NN)+   &
               TX(KK)*TX(LL)*zTDX(MM)*TX(NN)+TX(KK)*TX(LL)   &
              *TX(MM)*zTDX(NN))+DGZ(17)*(TY(KK)*TY(LL)+   &
               TZ(KK)*TZ(LL))*TX(MM)*TX(NN)               &
              +qm_qm_e_repul(17)*((zTDY(KK)*TY(LL)+   &
               TY(KK)*zTDY(LL)+zTDZ(KK)*TZ(LL)+TZ(KK)          &
              *zTDZ(LL))*TX(MM)*TX(NN)+(TY(KK)*TY(LL)+   &
               TZ(KK)*TZ(LL))*(zTDX(MM)*TX(NN)+TX(MM)      &
              *zTDX(NN)))+DGZ(18)*TX(KK)*TX(LL)*(TY(MM)*TY(NN)+   &
               TZ(MM)*TZ(NN))+qm_qm_e_repul(18)   &
              *((zTDX(KK)*TX(LL)+TX(KK)*zTDX(LL))*(TY(MM)*TY(NN)+   &
               TZ(MM)*TZ(NN))+TX(KK)*TX(LL)     &
              *(zTDY(MM)*TY(NN)+TY(MM)*zTDY(NN)+   &
                zTDZ(MM)*TZ(NN)+TZ(MM)*zTDZ(NN)))
      DRX(ISP)=DRX(ISP)+DGX(19)*(TY(KK)*TY(LL)*TY(MM)*TY(NN)+   &
               TZ(KK)*TZ(LL)*TZ(MM)*TZ(NN))          &
              +qm_qm_e_repul(19)*(xTDY(KK)*TY(LL)*TY(MM)*TY(NN)+   &
               TY(KK)*xTDY(LL)*TY(MM)*TY(NN)     &
              +TY(KK)*TY(LL)*xTDY(MM)*TY(NN)+   &
               TY(KK)*TY(LL)*TY(MM)*xTDY(NN)+xTDZ(KK)*TZ(LL)*TZ(MM)  &
              *TZ(NN)+TZ(KK)*xTDZ(LL)*TZ(MM)*TZ(NN)+   &
               TZ(KK)*TZ(LL)*xTDZ(MM)*TZ(NN)+TZ(KK)*TZ(LL)   &
              *TZ(MM)*xTDZ(NN))+DGX(20)*(TX(KK)*(TX(MM)*(TY(LL)*TY(NN)   &
              +TZ(LL)*TZ(NN))+TX(NN)      &
              *(TY(LL)*TY(MM)+TZ(LL)*TZ(MM)))+   &
               TX(LL)*(TX(MM)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN))      &
              +TX(NN)*(TY(KK)*TY(MM)+TZ(KK)*TZ(MM))))
      DRY(ISP)=DRY(ISP)+DGY(19)*(TY(KK)*TY(LL)*TY(MM)*TY(NN)+   &
               TZ(KK)*TZ(LL)*TZ(MM)*TZ(NN))          &
              +qm_qm_e_repul(19)*(yTDY(KK)*TY(LL)*TY(MM)*TY(NN)+   &
               TY(KK)*yTDY(LL)*TY(MM)*TY(NN)     &
              +TY(KK)*TY(LL)*yTDY(MM)*TY(NN)+   &
               TY(KK)*TY(LL)*TY(MM)*yTDY(NN)+yTDZ(KK)*TZ(LL)*TZ(MM)  &
              *TZ(NN)+TZ(KK)*yTDZ(LL)*TZ(MM)*TZ(NN)+   &
               TZ(KK)*TZ(LL)*yTDZ(MM)*TZ(NN)+TZ(KK)*TZ(LL)   &
              *TZ(MM)*yTDZ(NN))+DGY(20)*(TX(KK)*(TX(MM)*(TY(LL)*TY(NN)   &
              +TZ(LL)*TZ(NN))+TX(NN)      &
              *(TY(LL)*TY(MM)+TZ(LL)*TZ(MM)))+   &
               TX(LL)*(TX(MM)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN))      &
              +TX(NN)*(TY(KK)*TY(MM)+TZ(KK)*TZ(MM))))
      DRZ(ISP)=DRZ(ISP)+DGZ(19)*(TY(KK)*TY(LL)*TY(MM)*TY(NN)+   &
               TZ(KK)*TZ(LL)*TZ(MM)*TZ(NN))          &
              +qm_qm_e_repul(19)*(zTDY(KK)*TY(LL)*TY(MM)*TY(NN)+   &
               TY(KK)*zTDY(LL)*TY(MM)*TY(NN)     &
              +TY(KK)*TY(LL)*zTDY(MM)*TY(NN)+   &
               TY(KK)*TY(LL)*TY(MM)*zTDY(NN)+zTDZ(KK)*TZ(LL)*TZ(MM)  &
              *TZ(NN)+TZ(KK)*zTDZ(LL)*TZ(MM)*TZ(NN)+   &
               TZ(KK)*TZ(LL)*zTDZ(MM)*TZ(NN)+TZ(KK)*TZ(LL)   &
              *TZ(MM)*zTDZ(NN))+DGZ(20)*(TX(KK)*(TX(MM)*(TY(LL)*TY(NN)   &
              +TZ(LL)*TZ(NN))+TX(NN)      &
              *(TY(LL)*TY(MM)+TZ(LL)*TZ(MM)))+   &
               TX(LL)*(TX(MM)*(TY(KK)*TY(NN)+TZ(KK)*TZ(NN))      &
              +TX(NN)*(TY(KK)*TY(MM)+TZ(KK)*TZ(MM))))

! to avoid compiler difficulties this is divided
      xTEMP1=  xTDX(KK)*(TX(MM)*(TY(LL)*TY(NN)+TZ(LL)*TZ(NN))+   &
               TX(NN)*(TY(LL)*TY(MM)+TZ(LL)        &
              *TZ(MM)))+xTDX(LL)*(TX(MM)*(TY(KK)*TY(NN)+   &
               TZ(KK)*TZ(NN))+TX(NN)*(TY(KK)*TY(MM)     &
              +TZ(KK)*TZ(MM)))+TX(KK)*(xTDX(MM)*(TY(LL)*TY(NN)+   &
               TZ(LL)*TZ(NN))+xTDX(NN)*(TY(LL)    &
              *TY(MM)+TZ(LL)*TZ(MM)))+TX(LL)*(xTDX(MM)*(TY(KK)*TY(NN)   &
              +TZ(KK)*TZ(NN))+xTDX(NN)     &
              *(TY(KK)*TY(MM)+TZ(KK)*TZ(MM)))
      yTEMP1=  yTDX(KK)*(TX(MM)*(TY(LL)*TY(NN)+   &
               TZ(LL)*TZ(NN))+TX(NN)*(TY(LL)*TY(MM)+TZ(LL)        &
              *TZ(MM)))+yTDX(LL)*(TX(MM)*(TY(KK)*TY(NN)+   &
               TZ(KK)*TZ(NN))+TX(NN)*(TY(KK)*TY(MM)     &
              +TZ(KK)*TZ(MM)))+TX(KK)*(yTDX(MM)*(TY(LL)*TY(NN)+   &
               TZ(LL)*TZ(NN))+yTDX(NN)*(TY(LL)    &
              *TY(MM)+TZ(LL)*TZ(MM)))+TX(LL)*(yTDX(MM)*(TY(KK)*TY(NN)   &
              +TZ(KK)*TZ(NN))+yTDX(NN)     &
              *(TY(KK)*TY(MM)+TZ(KK)*TZ(MM)))
      zTEMP1=  zTDX(KK)*(TX(MM)*(TY(LL)*TY(NN)+TZ(LL)*TZ(NN))+   &
               TX(NN)*(TY(LL)*TY(MM)+TZ(LL)        &
              *TZ(MM)))+zTDX(LL)*(TX(MM)*(TY(KK)*TY(NN)+   &
               TZ(KK)*TZ(NN))+TX(NN)*(TY(KK)*TY(MM)     &
              +TZ(KK)*TZ(MM)))+TX(KK)*(zTDX(MM)*(TY(LL)*TY(NN)+   &
               TZ(LL)*TZ(NN))+zTDX(NN)*(TY(LL)    &
              *TY(MM)+TZ(LL)*TZ(MM)))+TX(LL)*(zTDX(MM)*(TY(KK)*TY(NN)   &
              +TZ(KK)*TZ(NN))+zTDX(NN)     &
              *(TY(KK)*TY(MM)+TZ(KK)*TZ(MM)))

      xTEMP2=  TX(KK)*(TX(MM)*(xTDY(LL)*TY(NN)+TY(LL)*xTDY(NN)+   &
               xTDZ(LL)*TZ(NN)+TZ(LL)*xTDZ(NN))      &
              +TX(NN)*(xTDY(LL)*TY(MM)+TY(LL)*xTDY(MM)+   &
               xTDZ(LL)*TZ(MM)+TZ(LL)*xTDZ(MM)))+TX(LL)     &
              *(TX(MM)*(xTDY(KK)*TY(NN)+TY(KK)*xTDY(NN)+   &
                xTDZ(KK)*TZ(NN)+TZ(KK)*xTDZ(NN))+TX(NN)     &
              *(xTDY(KK)*TY(MM)+TY(KK)*xTDY(MM)+   &
                xTDZ(KK)*TZ(MM)+TZ(KK)*xTDZ(MM)))
      yTEMP2=  TX(KK)*(TX(MM)*(yTDY(LL)*TY(NN)+TY(LL)*yTDY(NN)+   &
               yTDZ(LL)*TZ(NN)+TZ(LL)*yTDZ(NN))      &
              +TX(NN)*(yTDY(LL)*TY(MM)+TY(LL)*yTDY(MM)+   &
               yTDZ(LL)*TZ(MM)+TZ(LL)*yTDZ(MM)))+TX(LL)     &
              *(TX(MM)*(yTDY(KK)*TY(NN)+TY(KK)*yTDY(NN)+   &
                yTDZ(KK)*TZ(NN)+TZ(KK)*yTDZ(NN))+TX(NN)     &
              *(yTDY(KK)*TY(MM)+TY(KK)*yTDY(MM)+   &
                yTDZ(KK)*TZ(MM)+TZ(KK)*yTDZ(MM)))
      zTEMP2=  TX(KK)*(TX(MM)*(zTDY(LL)*TY(NN)+TY(LL)*zTDY(NN)+   &
               zTDZ(LL)*TZ(NN)+TZ(LL)*zTDZ(NN))      &
              +TX(NN)*(zTDY(LL)*TY(MM)+TY(LL)*zTDY(MM)+   &
               zTDZ(LL)*TZ(MM)+TZ(LL)*zTDZ(MM)))+TX(LL)     &
              *(TX(MM)*(zTDY(KK)*TY(NN)+TY(KK)*zTDY(NN)+   &
               zTDZ(KK)*TZ(NN)+TZ(KK)*zTDZ(NN))+TX(NN)     &
              *(zTDY(KK)*TY(MM)+TY(KK)*zTDY(MM)+   &
               zTDZ(KK)*TZ(MM)+TZ(KK)*zTDZ(MM)))

      DRX(ISP)=DRX(ISP)+qm_qm_e_repul(20)*(xTEMP1+xTEMP2)
      DRY(ISP)=DRY(ISP)+qm_qm_e_repul(20)*(yTEMP1+yTEMP2)
      DRZ(ISP)=DRZ(ISP)+qm_qm_e_repul(20)*(zTEMP1+zTEMP2)
      DRX(ISP)=DRX(ISP)+DGX(21)*( TY(KK)*TY(LL)*TZ(MM)*TZ(NN)   &
                                 +TZ(KK)*TZ(LL)*TY(MM)*TY(NN) )   &
             +qm_qm_e_repul(21)*( xTDY(KK)*TY(LL)*TZ(MM)*TZ(NN)   &
                                 +TY(KK)*xTDY(LL)*TZ(MM)*TZ(NN)   &
                                 +TY(KK)*TY(LL)*xTDZ(MM)*TZ(NN)   &
                                 +TY(KK)*TY(LL)*TZ(MM)*xTDZ(NN)   &
                                 +xTDZ(KK)*TZ(LL)*TY(MM)*TY(NN)   &
                                 +TZ(KK)*xTDZ(LL)*TY(MM)*TY(NN)   &
                                 +TZ(KK)*TZ(LL)*xTDY(MM)*TY(NN)   &
                                 +TZ(KK)*TZ(LL)*TY(MM)*xTDY(NN) )
      DRY(ISP)=DRY(ISP)+DGY(21)*( TY(KK)*TY(LL)*TZ(MM)*TZ(NN)   &
                                 +TZ(KK)*TZ(LL)*TY(MM)*TY(NN) )   &
             +qm_qm_e_repul(21)*( yTDY(KK)*TY(LL)*TZ(MM)*TZ(NN)   &
                                 +TY(KK)*yTDY(LL)*TZ(MM)*TZ(NN)   &
                                 +TY(KK)*TY(LL)*yTDZ(MM)*TZ(NN)   &
                                 +TY(KK)*TY(LL)*TZ(MM)*yTDZ(NN)   &
                                 +yTDZ(KK)*TZ(LL)*TY(MM)*TY(NN)   &
                                 +TZ(KK)*yTDZ(LL)*TY(MM)*TY(NN)   &
                                 +TZ(KK)*TZ(LL)*yTDY(MM)*TY(NN)   &
                                 +TZ(KK)*TZ(LL)*TY(MM)*yTDY(NN) )
      DRZ(ISP)=DRZ(ISP)+DGZ(21)*( TY(KK)*TY(LL)*TZ(MM)*TZ(NN)   &
                                 +TZ(KK)*TZ(LL)*TY(MM)*TY(NN) )   &
             +qm_qm_e_repul(21)*( zTDY(KK)*TY(LL)*TZ(MM)*TZ(NN)   &
                                 +TY(KK)*zTDY(LL)*TZ(MM)*TZ(NN)   &
                                 +TY(KK)*TY(LL)*zTDZ(MM)*TZ(NN)   &
                                 +TY(KK)*TY(LL)*TZ(MM)*zTDZ(NN)   &
                                 +zTDZ(KK)*TZ(LL)*TY(MM)*TY(NN)   &
                                 +TZ(KK)*zTDZ(LL)*TY(MM)*TY(NN)   &
                                 +TZ(KK)*TZ(LL)*zTDY(MM)*TY(NN)   &
                                 +TZ(KK)*TZ(LL)*TY(MM)*zTDY(NN) )

      DRX(ISP)=DRX(ISP)+DGX(22)*( TY(KK)*TZ(LL)+TZ(KK)*TY(LL) )    &
                               *( TY(MM)*TZ(NN)+TZ(MM)*TY(NN) )   &
                       +qm_qm_e_repul(22)*( ( xTDY(KK)*TZ(LL)   &
                                             +TY(KK)*xTDZ(LL)   &
                                             +xTDZ(KK)*TY(LL)   &
                                             +TZ(KK)*xTDY(LL) )   &
                               *( TY(MM)*TZ(NN)+TZ(MM)*TY(NN) )   &
                               +( TY(KK)*TZ(LL)+TZ(KK)*TY(LL) )   &
                               *( xTDY(MM)*TZ(NN)+TY(MM)*xTDZ(NN)   &
                                 +xTDZ(MM)*TY(NN)+TZ(MM)*xTDY(NN) ) )
      DRY(ISP)=DRY(ISP)+DGY(22)*( TY(KK)*TZ(LL)+TZ(KK)*TY(LL) )   &
                               *( TY(MM)*TZ(NN)+TZ(MM)*TY(NN) )   &
                       +qm_qm_e_repul(22)*( ( yTDY(KK)*TZ(LL)   &
                                             +TY(KK)*yTDZ(LL)   &
                                             +yTDZ(KK)*TY(LL)   &
                                             +TZ(KK)*yTDY(LL) )   &
                               *( TY(MM)*TZ(NN)+TZ(MM)*TY(NN) )   &
                               +( TY(KK)*TZ(LL)+TZ(KK)*TY(LL) )   &
                               *( yTDY(MM)*TZ(NN)+TY(MM)*yTDZ(NN)   &
                                 +yTDZ(MM)*TY(NN)+TZ(MM)*yTDY(NN) ) )
      DRZ(ISP)=DRZ(ISP)+DGZ(22)*( TY(KK)*TZ(LL)+TZ(KK)*TY(LL) )   &
                               *( TY(MM)*TZ(NN)+TZ(MM)*TY(NN) )   &
                       +qm_qm_e_repul(22)*( ( zTDY(KK)*TZ(LL)    &
                                             +TY(KK)*zTDZ(LL)   &
                                             +zTDZ(KK)*TY(LL)   &
                                             +TZ(KK)*zTDY(LL) )   &
                               *( TY(MM)*TZ(NN)+TZ(MM)*TY(NN) )   &
                               +( TY(KK)*TZ(LL)+TZ(KK)*TY(LL) )   &
                               *( zTDY(MM)*TZ(NN)+TY(MM)*zTDZ(NN)   &
                                 +zTDZ(MM)*TY(NN)+TZ(MM)*zTDY(NN) ) )
!                        
                     END IF

                  end do                ! N=M,n_atomic_orbj
               end do                   ! M=1,n_atomic_orbj
            end do                      ! L=K,total_atomic_orb 
         end do                         ! K=qm_atomi_orb_start,
!                                       !   total_atomic_orb

      else
! Light atom - Light atom pair

! Just have 1 atomic orbital interacting with 1 atomic orbital
! so only need to do (SS/SS)

         DRX(1) = DGX(1)
         DRY(1) = DGY(1)
         DRZ(1) = DGZ(1)
      end if                       ! (n_atomic_orbi>1.OR.n_atomic_orbj>1)
! end of Step 6
!***************************************************************

!***************************************************************
! Step7: Nuclear-nuclear and Nuclear-electron interactions

!a) Core-core terms: MNDO, AM1 and PM3
!1) Specialy treatment for N-H and O-H terms
         if (natqmi.EQ.1 .AND. (natqmj.EQ.7 .OR. natqmj.EQ.8)) then
            F3         = one+EXP1i+RIJ*EXP1j
            temp_real3 = alphai*EXP1i+(alphaj*RIJ-one)*EXP1j
            temp_real3 = temp_real3 * qm_qm_e_repul(1)
            DDX        =(dgx(1)*F3-vec_qm_qm_oneRIJ1*temp_real3)*C1
            DDY        =(dgy(1)*F3-vec_qm_qm_oneRIJ2*temp_real3)*C1
            DDZ        =(dgz(1)*F3-vec_qm_qm_oneRIJ3*temp_real3)*C1
         else if ((natqmi.EQ.7 .OR. natqmi.EQ.8).AND. natqmj.EQ.1) then
            F3         = one+EXP1j+RIJ*EXP1i
            temp_real3 = alphaj*EXP1j+(alphai*RIJ-one)*EXP1i
            temp_real3 = temp_real3 * qm_qm_e_repul(1)
            DDX        =(dgx(1)*F3-vec_qm_qm_oneRIJ1*temp_real3)*C1
            DDY        =(dgy(1)*F3-vec_qm_qm_oneRIJ2*temp_real3)*C1
            DDZ        =(dgz(1)*F3-vec_qm_qm_oneRIJ3*temp_real3)*C1

         else
!2) Other cases: Core-core repulsion
            PART1x     = dgx(1)*C1
            PART1y     = dgy(1)*C1
            PART1z     = dgz(1)*C1
            temp_real3 =(EXP1i+EXP1j)*ABS(C1)
            PART3x     = dgx(1)*temp_real3
            PART3y     = dgy(1)*temp_real3
            PART3z     = dgz(1)*temp_real3

            temp_real4 =-qm_qm_e_repul(1)*(alphai*EXP1i+alphaj*EXP1j)   &
                        *ABS(C1)
            PART2x     = temp_real4*vec_qm_qm_oneRIJ1
            PART2y     = temp_real4*vec_qm_qm_oneRIJ2
            PART2z     = temp_real4*vec_qm_qm_oneRIJ3

            DDx        = PART1x+PART2x+PART3x
            DDy        = PART1y+PART2y+PART3y
            DDz        = PART1z+PART2z+PART3z
         end if
 
         FNUCX = DDX+FNUCX
         FNUCY = DDY+FNUCY
         FNUCZ = DDZ+FNUCZ

!b) Core-electron attaraction derivatives (MNDO, AM1, and PM3)
!1) Atom core I affecting A.O.S on J
         ISP=0
         do M=1,n_atomic_orbj
            BB=one
            do N=M,n_atomic_orbj
               MN        = M+qm2_params%pascal_tri1(N)
               ISP       = ISP+1
               temp_real = BB*corei*PSUM(MN)
               FABx      = FABx-temp_real*DRX(ISP)
               FABy      = FABy-temp_real*DRY(ISP)
               FABz      = FABz-temp_real*DRZ(ISP)

               BB        = two
            end do
         end do

!2) Atom core J affecting A.O.S on I
         K=n_atomic_orbj
         K=qm2_params%pascal_tri2(n_atomic_orbj)
         ISP=-K+1
         do M=qm_atomi_orb_start,total_atomic_orb
            BB=one
            do N=M,total_atomic_orb
               MN        = M+qm2_params%pascal_tri1(N)
               ISP       = ISP+K
               temp_real = BB*corej*PSUM(MN)
               FABx      = FABx-temp_real*DRX(ISP)
               FABy      = FABy-temp_real*DRY(ISP)
               FABz      = FABz-temp_real*DRZ(ISP)

               BB        = two
            end do
         end do

!c) Coulomb and Exchange terms (MNDO, AM1, and PM3)
         ISP=0
         do K=qm_atomi_orb_start,total_atomic_orb
            AA=one
            KK=qm2_params%pascal_tri1(K)
            do L=K,total_atomic_orb
               LL=qm2_params%pascal_tri1(L)
               KL=K+LL
               do M=1,n_atomic_orbj
                  BB=one
                  MK=M+KK
                  ML=M+LL
                  do N=M,n_atomic_orbj
                     ISP= ISP+1
                     MN = M+qm2_params%pascal_tri1(N)

! Comlomb term
                     temp_real= AA*BB*PSUM(KL)*PSUM(MN)
                     FAAx     = FAAx+temp_real*DRX(ISP)
                     FAAy     = FAAy+temp_real*DRY(ISP)
                     FAAz     = FAAz+temp_real*DRZ(ISP)

! Exchange term
                     NK=N+KK
                     NL=N+LL
                     temp_real = AA*BB*fourth*( PSUM(MK)*PSUM(NL)+   &
                                                PSUM(NK)*PSUM(ML) )
                     FAAx      = FAAx-temp_real*DRX(ISP)
                     FAAy      = FAAy-temp_real*DRY(ISP)
                     FAAz      = FAAz-temp_real*DRZ(ISP)

                     BB        = two
                  end do                      ! N=M,n_atomic_orbj
               end do                         ! M=1,n_atomic_orbj

               AA = two
            end do               ! L=K,total_atomic_orb
         end do                  ! K=qm_atomi_orb_start,total_atomic_orb


!***************************************************************
! Step8: Combine all  
      pair_force(1) = FAAx+FABx+FNUCX
      pair_force(1) =-pair_force(1)*EV_TO_KCAL

      pair_force(2) = FAAy+FABy+FNUCY
      pair_force(2) =-pair_force(2)*EV_TO_KCAL

      pair_force(3) = FAAz+FABz+FNUCZ
      pair_force(3) =-pair_force(3)*EV_TO_KCAL

      return
      END SUBROUTINE qm2_deriv_qm_analyt

                                                                       
      SUBROUTINE qm2_get_qmmm_forces(dxyzqm,qm_xcrd,dxyzmm)
!
! Routine calculates gradient on the MM atoms due to the 
!         QM atoms and vice versa. The forces are added 
!         to dxyzmm and dxyzqm respectively.
!
! Note:   Currently, only analytical derivatives are
!         available.
!         This routine does not need igmatoms since it 
!         gets mm atoms from the pair list and puts QM
!         forces in it's own array which gets moved to
!         the main gradient array later.
!
! Variable definitions:
!    qm_coords : Cartesian coordinates of QM atoms
!    dxyzqm    : Coupled potential energy derivatives with
!                respect to movement of QM atoms
!    qm_xcrd   : Cartesian coordinates of QM and MM atoms
!                In same order as cutoff list, and also
!                contains MM charges in 4th elements
!    dxyzmm    : Coupled potential energy derivatives with
!                respect to movement of MM atoms
!
! Current Code and optimizations by Ross Walker and
! Mike Crowley (TSRI, 2004)
!

  use qmmm_module, only : qmmm_nml,qmmm_struct,qm2_struct,   &
                              qm2_params,qmmm_switch
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
#if KEY_PARALLEL==1
  use parallel  
#endif
!
      implicit none
!Passed in
      real(chm_real), intent(out) :: dxyzqm(3,qmmm_struct%nquant_nlink)
      real(chm_real), intent(in)  :: qm_xcrd(*)
      real(chm_real), intent(out) :: dxyzmm(*)

!Local variables
      real(chm_real):: e_repul(4), e_repul_light    ! Used when 
!                                               ! qmmm_erep_incore = false.
      real(chm_real):: psum(36), psum_light,  &
                 pden_sum(10,qmmm_struct%nquant_nlink)
      real(chm_real):: pair_force(3)
      real(chm_real):: qm_atom_coord(3)
      real(chm_real):: qm_atom_core, qm_atom_alpa
      real(chm_real):: sum_elec, qmmm_sw_force(6), denfac(36)
      integer :: jj, jf, jl, ij, i, k, ii, j
      integer :: n_atomic_orb                 ! Number of atomic orbitals 
!                                             ! on qm atom
      integer :: loop_count           ! Keeps track of number of 
!                                     ! times through nquant * ni_mm loop
      integer :: inner_loop_count     ! xyz offset for loop over pairs.
      integer :: ncount, im,imr, iqr,iqr2, ncnts,ncntf,jsq(2)
      integer :: local_qm_indx(4,qmmm_switch%nswitch_atom)
      logical :: heavy_atom

      integer :: mstart,mstop,mstart2,mstop2,ncnt2
      integer :: ISTRT_CHECK                 ! for external function

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
      end if
#endif 

! Fill the array of density factors (off-diagonal elements * 2) 
! -used when qmmm_switch%qswitch =.true.
      If(qmmm_switch%qswitch) then
         denfac(1:36)= two
         do i=1,8
            ii=i*(i+1)/2
            denfac(ii) = one  ! diagonal
         end do
!  pden_sum(1:10,1:qmmm_struct%nquant_nlink) = zero
      End if

! Loop over QM atoms
      loop_count=0
      do jj=1,qmmm_struct%nquant_nlink
         jf               = qm2_params%orb_loc(1,jj)
         jl               = qm2_params%orb_loc(2,jj)
         qm_atom_coord(1) = qmmm_struct%qm_coords(1,jj)
         qm_atom_coord(2) = qmmm_struct%qm_coords(2,jj)
         qm_atom_coord(3) = qmmm_struct%qm_coords(3,jj)
         qm_atom_core     = qm2_params%core_chg(jj)
         qm_atom_alpa     = qm2_params%cc_exp_params(jj)
         n_atomic_orb     = qm2_params%natomic_orbs(jj)
         heavy_atom       = (n_atomic_orb .GT. 1)

! Split into heavy and light atoms
!   it has code duplication but is good for speed.

! for heavy atom
         if (heavy_atom) then
! get density matrix elements involving only orbitals
! centered on current QM atom
            ij = 0
            do i=jf,jl
               k = qm2_params%pascal_tri1(i)+jf-1
               do j=jf,i
                  ij       = ij + 1
                  k        = k + 1
                  psum(ij) = qm2_struct%den_matrix(k)
               end do
            end do
            if(qmmm_switch%qswitch)  &
               pden_sum(1:10,jj) = psum(1:10)*denfac(1:10)

! RCW : We could maybe move this outside of the loop 
!       optimization purposes. Loop over MM atoms that are
!       in the interaction list for current QM atom.
!       We have the loop repeated twice here the qm-mm 1-e
!       repulsion integrals in memory and once for when
!       we need to calculate them on the fly.
!
!       This is excess code but it avoids the if(...) being 
!       inside the inner loop.

            if (qmmm_nml%qmmm_erep_incore) then
! Loop in steps of 4 since mm_xcrd is laid out as
! x,y,z,chg,...x,y,z,chg...
               inner_loop_count=1
#if KEY_PARALLEL==1
! befor loop (for loop_count)
               loop_count = loop_count + (mstart-1)
               inner_loop_count = inner_loop_count+3*(mstart-1)
#endif 
               do ii=mstart2,mstop2,4        ! 1,4*qmmm_struct%qm_mm_pairs,4
                 loop_count=loop_count+1

! get analytical derivatives wrt x, y, and z 
! directions for the interaction of the current
! QM-MM pair
                 call qm2_deriv_qmmm_heavy(jj,loop_count,   &
                              qm2_struct%qm_mm_e_repul(1,loop_count),   &
                              psum,qm_atom_coord,qm_xcrd(ii),   &
                              n_atomic_orb,pair_force,   &
                              qm_atom_core,qm_atom_alpa)

! save into MM array
                 dxyzmm(inner_loop_count)  =dxyzmm(inner_loop_count)   &
                                           -pair_force(1)
                 dxyzmm(inner_loop_count+1)=dxyzmm(inner_loop_count+1)   &
                                           -pair_force(2)
                 dxyzmm(inner_loop_count+2)=dxyzmm(inner_loop_count+2) &
                                           -pair_force(3)
! save into QM array
                 dxyzqm(1,jj) = dxyzqm(1,jj) + pair_force(1)
                 dxyzqm(2,jj) = dxyzqm(2,jj) + pair_force(2)
                 dxyzqm(3,jj) = dxyzqm(3,jj) + pair_force(3)

                 inner_loop_count = inner_loop_count+3
               end do
#if KEY_PARALLEL==1
! after loop (for loop_count)
               loop_count = loop_count+(qmmm_struct%qm_mm_pairs-mstop)
#endif 

! not saved in memory, so calc on the fly
            else
               inner_loop_count=1
#if KEY_PARALLEL==1
! befor loop (for loop_count)
               loop_count = loop_count + (mstart-1)
               inner_loop_count = inner_loop_count+3*(mstart-1)
#endif 
               do ii=mstart2,mstop2,4        ! 1,4*qmmm_struct%qm_mm_pairs,4
                 loop_count=loop_count+1

! get analytical derivatives wrt x, y, and z 
! directions for the interaction of the current
! QM-MM pair
                 call qm2_deriv_qmmm_heavy(jj,loop_count,   &
                                           e_repul,   &
                                           psum,qm_atom_coord,   &
                                           qm_xcrd(ii),   &
                                           n_atomic_orb,pair_force,   &
                                           qm_atom_core,qm_atom_alpa)
  
! save into MM array
                 dxyzmm(inner_loop_count)  =dxyzmm(inner_loop_count)     &
                                           -pair_force(1)
                 dxyzmm(inner_loop_count+1)=dxyzmm(inner_loop_count+1)   &
                                           -pair_force(2)
                 dxyzmm(inner_loop_count+2)=dxyzmm(inner_loop_count+2)   &
                                           -pair_force(3)
! save into QM array
                 dxyzqm(1,jj) = dxyzqm(1,jj) + pair_force(1)
                 dxyzqm(2,jj) = dxyzqm(2,jj) + pair_force(2)
                 dxyzqm(3,jj) = dxyzqm(3,jj) + pair_force(3)

                 inner_loop_count = inner_loop_count+3
               end do
#if KEY_PARALLEL==1
! after loop (for loop_count)
               loop_count = loop_count+(qmmm_struct%qm_mm_pairs-mstop)
#endif 

            end if                           ! qmmm_nml%qmmm_erep_incore

! hydrogen atoms : see above for comments
         else 
            k              = qm2_params%pascal_tri1(jf) + jf 
            psum_light     = qm2_struct%den_matrix(k)
            if(qmmm_switch%qswitch) then
               pden_sum(1,jj)    = psum_light*denfac(1)
               pden_sum(2:10,jj) = zero
            end if

            if (qmmm_nml%qmmm_erep_incore) then
               inner_loop_count=1
#if KEY_PARALLEL==1
! befor loop (for loop_count)
               loop_count = loop_count + (mstart-1)
               inner_loop_count = inner_loop_count+3*(mstart-1)
#endif 
               do ii=mstart2,mstop2,4        ! 1,4*qmmm_struct%qm_mm_pairs,4
                 loop_count=loop_count+1
                 call qm2_deriv_qmmm_light(jj,loop_count,   &
                              qm2_struct%qm_mm_e_repul(1,loop_count),   &
                              psum_light,qm_atom_coord,qm_xcrd(ii),   &
                              pair_force,qm_atom_core,qm_atom_alpa)

! save into MM array
                 dxyzmm(inner_loop_count)  =dxyzmm(inner_loop_count)    &
                                           -pair_force(1)
                 dxyzmm(inner_loop_count+1)=dxyzmm(inner_loop_count+1)  &
                                           -pair_force(2)
                 dxyzmm(inner_loop_count+2)=dxyzmm(inner_loop_count+2)  &
                                           -pair_force(3)
! save into QM array
                 dxyzqm(1,jj) = dxyzqm(1,jj) + pair_force(1)
                 dxyzqm(2,jj) = dxyzqm(2,jj) + pair_force(2)
                 dxyzqm(3,jj) = dxyzqm(3,jj) + pair_force(3)

                 inner_loop_count = inner_loop_count+3
               end do
#if KEY_PARALLEL==1
! after loop (for loop_count)
               loop_count = loop_count+(qmmm_struct%qm_mm_pairs-mstop)
#endif 

! do on the fly
            else
               inner_loop_count=1
#if KEY_PARALLEL==1
! befor loop (for loop_count)
               loop_count = loop_count + (mstart-1)
               inner_loop_count = inner_loop_count+3*(mstart-1)
#endif 
               do ii=mstart2,mstop2,4        ! 1,4*qmmm_struct%qm_mm_pairs,4
                 loop_count=loop_count+1
                 call qm2_deriv_qmmm_light(jj,loop_count,   &
                                           e_repul_light,   &
                                           psum_light,qm_atom_coord,   &
                                           qm_xcrd(ii),   &
                                           pair_force,qm_atom_core,   &
                                           qm_atom_alpa)

! save into MM array
                 dxyzmm(inner_loop_count)  =dxyzmm(inner_loop_count)    &
                                           -pair_force(1)
                 dxyzmm(inner_loop_count+1)=dxyzmm(inner_loop_count+1)  &
                                           -pair_force(2)
                 dxyzmm(inner_loop_count+2)=dxyzmm(inner_loop_count+2)  &
                                           -pair_force(3)
! save into QM array
                 dxyzqm(1,jj) = dxyzqm(1,jj) + pair_force(1)
                 dxyzqm(2,jj) = dxyzqm(2,jj) + pair_force(2)
                 dxyzqm(3,jj) = dxyzqm(3,jj) + pair_force(3)

                 inner_loop_count = inner_loop_count+3
               end do
#if KEY_PARALLEL==1
! after loop (for loop_count)
               loop_count = loop_count+(qmmm_struct%qm_mm_pairs-mstop)
#endif 
            end if                           ! qmmm_nml%qmmm_erep_incore
         end if                              ! if (heavy_atom)
      end do                                 ! jj=1,qmmm_struct%nquant

! Electrostatic switching function for cutoff
      If(qmmm_switch%qswitch) then
! initialize temporay gradient array
         qmmm_switch%dxmm_wrk(1:3,1:qmmm_switch%nswitch_atom) = zero
         qmmm_switch%dxyz_qm(1:3,1:qmmm_switch%nqmgrp) = zero

         ncount = 0
#if KEY_PARALLEL==1
         do ii=1,mstart-1
            if(qmmm_switch%scmask(qmmm_struct%qm_mm_pair_list(ii))) then
               ncount = ncount + 1
            end if
         end do
#endif 
         ncnt2=ncount                       ! only used in parallel run.
         do ii=mstart,mstop                    ! 1,qmmm_struct%qm_mm_pairs
            im =qmmm_struct%qm_mm_pair_list(ii)
            if(qmmm_switch%scmask(im)) then
               ncount = ncount + 1
               imr      = qmmm_switch%immatm(im)
               iqr      = qmmm_switch%immgrp(1,imr)
               jsq(1:2) = qmmm_switch%immgrp(2:3,imr)
               if(iqr.lt.0) then
                  write(6,*)'Error in immgrp or immatm array:',imr,im
               end if

               local_qm_indx(1,ncount)   = iqr
               local_qm_indx(2,ncount)   = imr
               local_qm_indx(3:4,ncount) = jsq(1:2)

! check
               if(im.lt.jsq(1) .or. im.gt.jsq(2)) then
                  write(6,*)'Error in im',im,'/js',jsq(1),'/jq',jsq(2)
               end if
            end if
         end do

         do jj=1,qmmm_struct%nquant
            iqr    = qmmm_switch%iqmatm(jj)
            ncount = 0
#if KEY_PARALLEL==1
            ncount = ncnt2                     ! update loop count
#endif 
            do ii=mstart,mstop                 ! 1,qmmm_struct%qm_mm_pairs
               im  = qmmm_struct%qm_mm_pair_list(ii)
!
! loop over masked mm atom
               if(qmmm_switch%scmask(im)) then
                  ncount   = ncount + 1
                  iqr2     = local_qm_indx(1,ncount)
                  imr      = local_qm_indx(2,ncount)
                  jsq(1:2) = local_qm_indx(3:4,ncount)

! it is all same for heavy/H-atom, since pden_sum 
! has initialized...
                  sum_elec = zero
                  do i=1,10
                     sum_elec = sum_elec +    &
                                qmmm_switch%dxqmmm_elec(i,ncount,jj)   &
                               *pden_sum(i,jj)
                  end do
                  sum_elec = sum_elec + qmmm_switch%dxqmmm_elec(11,   &
                                                            ncount,jj)
                  sum_elec = sum_elec*EV_TO_KCAL

! multiply the gradient of switching function contribution
                  qmmm_sw_force(1:6)=qmmm_switch%dxqmmm(1:6,im)*sum_elec

! save into QM array
                  qmmm_switch%dxyz_qm(1:3,iqr2) =   &
                                   qmmm_switch%dxyz_qm(1:3,iqr2) +   &
                                   qmmm_sw_force(4:6)

! save into MM array
                  if(im.ge.jsq(1).or.im.le.jsq(2)) then
                     ncnts = ncount + (jsq(1)-im)
                     ncntf = ncount + (jsq(2)-im)
                     do j= ncnts,ncntf
                        qmmm_switch%dxmm_wrk(1:3,j) =   &
                                   qmmm_switch%dxmm_wrk(1:3,j) -   &
                                   qmmm_sw_force(1:3)
                     end do
                  end if
               end if
            end do
         end do

! copying QM gradient
         do jj=1,qmmm_struct%nquant
            iqr            = qmmm_switch%iqmatm(jj)
            dxyzqm(1:3,jj) = dxyzqm(1:3,jj) +   &
                             qmmm_switch%dxyz_qm(1:3,iqr)
         end do

! copying MM gradients
! Note: in parallel run, 
! since there is possibility mstart/mstop cut group atoms,
! the loop should be over all atoms, not mstart:mstop
         ncount           = 0
         inner_loop_count = 1
         do ii=1,qmmm_struct%qm_mm_pairs
           if(qmmm_switch%scmask(qmmm_struct%qm_mm_pair_list(ii))) then
              ncount   = ncount + 1
              dxyzmm(inner_loop_count:inner_loop_count+2)    &
                       = dxyzmm(inner_loop_count:inner_loop_count+2) +   &
                         qmmm_switch%dxmm_wrk(1:3,ncount)
           end if
           inner_loop_count = inner_loop_count+3
         end do

      End if

      return
      END SUBROUTINE qm2_get_qmmm_forces


      SUBROUTINE qm2_deriv_qmmm_light(iqm,loop_count,   &
                                   qm_mm_e_repul,   &
                                   psum_light,xyz_qm,xyz_mm,   &
                                   pair_force,qm_atom_core,alpa)
!
! For Light atoms
! See heavy version of routine for comments. On return,
! pari_force holds analytical derivatives
!
! Current Version: Ross Walker (TSRI, 2005)
!

  use qmmm_module, only :  qmmm_nml,qm2_params, qm2_struct,  &
                               qm2_rij_eqns,qmmm_struct
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations
      implicit none
!Passed in
      real(chm_real),intent(inout) :: qm_mm_e_repul    ! intent is in for 
!                                                ! qmmm_erep_incore = true. 
!                                                ! inout for = false.
      real(chm_real),intent(in)    :: xyz_qm(3),xyz_mm(4),psum_light
      real(chm_real),intent(out)   :: pair_force(3)
      integer, intent(in)    :: iqm, loop_count
      real(chm_real),intent(in)    :: qm_atom_core, alpa

!Local variables
      real(chm_real) :: FABX, FABY, FABZ ,FNUCX, FNUCY, FNUCZ
      real(chm_real) :: r2, rij, onerij, rr2
      real(chm_real) :: vec_qm_mm1, vec_qm_mm2, vec_qm_mm3
      real(chm_real) :: c1, mm_charge
      real(chm_real) :: ee, sqrtaee
      real(chm_real) :: DGX, DGY, DGZ
      real(chm_real) :: EXP1, EXP2, EXP3, EXP4
      integer  :: i,qmitype
      real(chm_real) :: temp_real,temp_real2,anam1,oner2

      vec_qm_mm1 = xyz_qm(1)-xyz_mm(1)
      vec_qm_mm2 = xyz_qm(2)-xyz_mm(2)
      vec_qm_mm3 = xyz_qm(3)-xyz_mm(3)
      mm_charge  = xyz_mm(4)

      if (qmmm_nml%qmmmrij_incore) then
         RIJ    = qm2_rij_eqns%qmmmrijdata(QMMMRIJ,loop_count)
         oneRIJ = qm2_rij_eqns%qmmmrijdata(QMMMONERIJ,loop_count)
         EXP1   = qm2_rij_eqns%qmmmrijdata(QMMMEXP1,loop_count)
         EXP2   = qm2_rij_eqns%qmmmrijdata(QMMMEXP2,loop_count)
         SQRTAEE= qm2_rij_eqns%qmmmrijdata(QMMMSQRTAEE,loop_count)
      else
         r2      = vec_qm_mm1*vec_qm_mm1 +   &
                   vec_qm_mm2*vec_qm_mm2 +   &
                   vec_qm_mm3*vec_qm_mm3
         RR2     = r2*A2_TO_BOHRS2           ! Conversion to bohrs^2
         oneRIJ  = one/sqrt(r2)              ! 1/sqrt is faster than doing sqrt.
         RIJ     = r2*oneRIJ                 ! faster than one/oneRIJ
         EXP1    = exp(-alpa*rij)
         EXP2    = exp(-ALPH_MM*rij)
         SQRTAEE = one/sqrt(RR2+qm2_params%multip_2c_elec_params(6,iqm))
      end if

      if (.NOT.qmmm_nml%qmmm_erep_incore) then
         qm_mm_e_repul = AU_TO_EV*SQRTAEE
      end if

      EE =-A2_TO_BOHRS2xAU_TO_EV*SQRTAEE*SQRTAEE*SQRTAEE

! S-orbital of QM atom
      DGX= vec_qm_mm1*EE
      DGY= vec_qm_mm2*EE
      DGZ= vec_qm_mm3*EE

! first derivative of nuclear repulsion term
      C1    = qm_atom_core*mm_charge
      EXP3  = (EXP1+EXP2)*abs(c1)
      EXP4  = qm_mm_e_repul*onerij*(alpa*EXP1 + ALPH_MM*EXP2)*abs(c1)
      FNUCX = dgx*c1-vec_qm_mm1*EXP4+dgX*EXP3
      FNUCY = dgy*c1-vec_qm_mm2*EXP4+dgy*EXP3
      FNUCZ = dgz*c1-vec_qm_mm3*EXP4+dgz*EXP3

! mm core affecting AO's on QM atom
      mm_charge =-mm_charge*psum_light
      FABX      = mm_charge*DGX
      FABY      = mm_charge*DGY
      FABZ      = mm_charge*DGZ                    

! for AM1/PM3/PM3CARB1: Gaussian core-core terms.
!     PDDGPM3 should be done differently.
      If(qmmm_nml%QMMM_Gauss) then
         qmitype= qmmm_struct%qm_atom_type(iqm)
         oner2  = oneRIJ*oneRIJ
         anam1  = zero
         do i=1,qm2_params%num_fn(qmitype)
            temp_real = RIJ-qm2_params%FN3(i,qmitype)
            temp_real2= qm2_params%FN2(i,qmitype)*temp_real*temp_real

! Skip when the exponential is close enough to be zero
            if (temp_real2 .LT. EXPONENTIAL_CUTOFF) then
               anam1 = anam1 + qm2_params%FN1(i,qmitype)* &
                               EXP(-temp_real2)* &
                              (oner2 + two*qm2_params%FN2(i,qmitype)* &
                                           temp_real*oneRIJ)
            end if
         end do

         anam1 = -anam1*c1
         FNUCX = FNUCX+anam1*vec_qm_mm1*oneRIJ
         FNUCY = FNUCY+anam1*vec_qm_mm2*oneRIJ
         FNUCZ = FNUCZ+anam1*vec_qm_mm3*oneRIJ
      End if

! copying into main gradient array
      pair_force(1) = (FABX+FNUCX)*EV_TO_KCAL
      pair_force(2) = (FABY+FNUCY)*EV_TO_KCAL
      pair_force(3) = (FABZ+FNUCZ)*EV_TO_KCAL

      return
      END SUBROUTINE qm2_deriv_qmmm_light


      SUBROUTINE qm2_deriv_qmmm_heavy(iqm,loop_count,   &
                                   qm_mm_e_repul,   &
                                   psum,xyz_qm,xyz_mm,   &
                                   n_atomic_orb,pair_force,   &
                                   qm_atom_core,alpa)
!
! Calculation of analytical derivatives
!   This routine computed the analytical energy derivatives
!   for the QM-MM interaction energy arising from a single
!   QM-MM pair. The contributions to the derivatives come
!   from the electron-core and core-core interactions.
!
!   On return, pair_forces holds analytical derivatives
!
! Variable definitions:
!   qm_mm_e_repul : QM-MM electron repulsion integrals for 
!                   this QM-MM pair
!   psum          : Density matrix elements for orbitals 
!                   centered on QM atom
!   xyz_qm        : Cartesian coordinates of QM atom
!   xyz_mm        : Cartesian coordinates of MM atom, and
!                   charge 
!   n_atomic_orb  : Number of atomic orbitals
!   pair_force    : Energy derivatives in the x, y, and z
!                   directions for the interaction of the
!                   QM-MM atom pair. The algebraic sign of
!                   pair-force corresponds to dE/dr for 
!                   the MM atoma and -dE/dr for QM atom
!
! Current version and optimization by
! Ross Walker and Mike Crowley (TSRI 2004)
!

  use qmmm_module, only :  qmmm_nml,qm2_params, qm2_struct,  &
                               qm2_rij_eqns,qmmm_struct
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations
      implicit none
!Passed in
      real(chm_real),intent(inout) :: qm_mm_e_repul(4)   ! intent is in for 
!                                                  ! qmmm_erep_incore = true. 
!                                                  ! inout for = false.
      real(chm_real),intent(in)    :: xyz_qm(3),xyz_mm(4),psum(36)
      real(chm_real),intent(out)   :: pair_force(3)
      integer, intent(in)    :: iqm, loop_count, n_atomic_orb
      real(chm_real),intent(in)    :: qm_atom_core, alpa

!Local variables
      real(chm_real) :: FABX, FABY, FABZ ,FNUCX, FNUCY, FNUCZ
      real(chm_real) :: r2, rij, onerij, rr2, rr, one_rija0ev
      real(chm_real) :: vec_qm_mm1, vec_qm_mm2, vec_qm_mm3
      real(chm_real) :: c1, bb, mm_charge
      real(chm_real) :: sqrtaee, dze, qzze, qxxe
      real(chm_real) :: SQRTRRMDDADE, SQRTRRADDADE, SQRTRR2AQE
      real(chm_real) :: SQRTRRAQQAQE, SQRTRRMQQAQE, SQRTRRAQQ2AQE
      real(chm_real) :: Xtdx(3), Xtdy(3), Xtdz(3)
      real(chm_real) :: Ytdx(3), Ytdy(3), Ytdz(3)
      real(chm_real) :: Ztdx(3), Ztdy(3), Ztdz(3)
      real(chm_real) :: TX(3),TY(3),TZ(3), TZ3i, TZ3i2
      real(chm_real) :: RXY2, RYZ2, RZX2, oneRXY
      real(chm_real) :: TERMX, TERMY, TERMZ
      real(chm_real) :: DGX(4), DGY(4), DGZ(4)
      real(chm_real) :: DRX(MAX_VALENCE_DIMENSION)
      real(chm_real) :: DRY(MAX_VALENCE_DIMENSION)
      real(chm_real) :: DRZ(MAX_VALENCE_DIMENSION)
      real(chm_real) :: EXP1, EXP2, EXP3, EXP4
      real(chm_real) :: temp_real1, DD, QQ

      integer  :: isp, m, n, mn
      integer  :: k, kk, l, ll
      logical  :: LRXY2, LRYZ2, LRZX2
      integer  :: i,qmitype
      real(chm_real) :: temp_real,temp_real2,anam1,oner2



      vec_qm_mm1 = xyz_qm(1)-xyz_mm(1)
      vec_qm_mm2 = xyz_qm(2)-xyz_mm(2)
      vec_qm_mm3 = xyz_qm(3)-xyz_mm(3)
      mm_charge  = xyz_mm(4)

      if (qmmm_nml%qmmmrij_incore) then
         RIJ        = qm2_rij_eqns%qmmmrijdata(QMMMRIJ,loop_count)
         oneRIJ     = qm2_rij_eqns%qmmmrijdata(QMMMONERIJ,loop_count)
         one_rija0ev= oneRIJ*A_TO_BOHRS*AU_TO_EV
         EXP1       = qm2_rij_eqns%qmmmrijdata(QMMMEXP1,loop_count)
         EXP2       = qm2_rij_eqns%qmmmrijdata(QMMMEXP2,loop_count)
         SQRTAEE    = qm2_rij_eqns%qmmmrijdata(QMMMSQRTAEE,loop_count)

! Heavy Atom Specific
         SQRTRRADDADE = qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRADDADE,   &
                                                 loop_count)
         SQRTRRMDDADE = qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRMDDADE,   &
                                                 loop_count)
         SQRTRR2AQE   = qm2_rij_eqns%qmmmrijdata(QMMMSQRTRR2AQE,   &
                                                 loop_count)
         SQRTRRAQQAQE = qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRAQQAQE,   &
                                                 loop_count)
         SQRTRRMQQAQE = qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRMQQAQE,   &
                                                 loop_count)
         SQRTRRAQQ2AQE= qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRAQQ2AQE,   &
                                                 loop_count)

      else
         r2         = vec_qm_mm1*vec_qm_mm1+   &
                      vec_qm_mm2*vec_qm_mm2+   &
                      vec_qm_mm3*vec_qm_mm3
         RR2        = r2*A2_TO_BOHRS2           ! Conversion to bohrs^2
         oneRIJ     = one/sqrt(r2)              ! 1/sqrt is faster 
!                                               ! than doing sqrt.
         RIJ        = r2*oneRIJ                 ! faster than one/oneRIJ
         RR         = RIJ*A_TO_BOHRS
         one_rija0ev= A_TO_BOHRS * oneRIJ*AU_TO_EV
         EXP1       = exp(-alpa*rij)
         EXP2       = exp(-ALPH_MM*rij)
         SQRTAEE    = one/sqrt(RR2+   &
                      qm2_params%multip_2c_elec_params(6,iqm))

! Heavy atom specific
         DD           = qm2_params%multip_2c_elec_params(1,iqm)
         QQ           = qm2_params%multip_2c_elec_params(2,iqm)
         SQRTRRADDADE = one/SQRT( (RR+DD)**2 +   &
                              qm2_params%multip_2c_elec_params(7,iqm) )
         SQRTRRMDDADE = one/SQRT( (RR-DD)**2 +   &
                              qm2_params%multip_2c_elec_params(7,iqm) )
         SQRTRR2AQE   = one/SQRT( RR2 +   &
                              qm2_params%multip_2c_elec_params(8,iqm) )
         SQRTRRAQQAQE = one/SQRT( (RR+two*QQ)**2 +   &
                              qm2_params%multip_2c_elec_params(8,iqm) )
         SQRTRRMQQAQE = one/SQRT( (RR-two*QQ)**2 +   &
                              qm2_params%multip_2c_elec_params(8,iqm) )
         SQRTRRAQQ2AQE= one/SQRT( RR2+(four*(QQ**2) +   &
                              qm2_params%multip_2c_elec_params(8,iqm)))

      end if

! In case 1-e repulsion integrals for QM-MM pair in memory,
! calculate here.
      if (.NOT.qmmm_nml%qmmm_erep_incore) then
! 1-e repulsion integrals
         qm_mm_e_repul(1) = AU_TO_EV*SQRTAEE
         qm_mm_e_repul(2) = HALF_AU_TO_EV*(SQRTRRADDADE-SQRTRRMDDADE)
         qm_mm_e_repul(3) = qm_mm_e_repul(1) +   &
                            FOURTH_AU_TO_EV*(SQRTRRAQQAQE+   &
                                             SQRTRRMQQAQE)-   &
                            HALF_AU_TO_EV*SQRTRR2AQE
         qm_mm_e_repul(4) = qm_mm_e_repul(1) +   &
                            HALF_AU_TO_EV*(SQRTRRAQQ2AQE-   &
                                           SQRTRR2AQE)
      end if

! returns the derivatives of the electron-core interaction 
! energies in a local diatomic frame. At most only four terms
! are computed because there are at most four unique electron
! -core interactions for the QM-MM pair:
!   (ss| ), (so| ), (oo| ), (pp| ),.
!
! note that it is not necessary to specify whether an x, y, or
! z derivatives is being evaluated because the fourmula is the
! same for all three directions.
! one_rija0 = one_rija0
  
      SQRTAEE = -SQRTAEE*SQRTAEE*SQRTAEE*A2_TO_BOHRS2xAU_TO_EV

! S-orbital of QM atom:
      DGX(1)  = vec_qm_mm1*SQRTAEE
      DGY(1)  = vec_qm_mm2*SQRTAEE
      DGZ(1)  = vec_qm_mm3*SQRTAEE

      DD          = qm2_params%multip_2c_elec_params(1,iqm)*   &
                    one_rija0ev
      SQRTRRADDADE= SQRTRRADDADE*SQRTRRADDADE*SQRTRRADDADE
      SQRTRRMDDADE= SQRTRRMDDADE*SQRTRRMDDADE*SQRTRRMDDADE

      DZE = half*( A2_TO_BOHRS2xAU_TO_EV*   &
                   (SQRTRRMDDADE-SQRTRRADDADE) -   &
                  DD*(SQRTRRMDDADE+SQRTRRADDADE) )

      SQRTRR2AQE   = SQRTRR2AQE*SQRTRR2AQE*SQRTRR2AQE
      SQRTRRAQQ2AQE= SQRTRRAQQ2AQE*SQRTRRAQQ2AQE*SQRTRRAQQ2AQE
      QXXE         = SQRTAEE+half*A2_TO_BOHRS2xAU_TO_EV*   &
                                  (SQRTRR2AQE-SQRTRRAQQ2AQE)

      SQRTRRAQQAQE = SQRTRRAQQAQE*SQRTRRAQQAQE*SQRTRRAQQAQE
      SQRTRRMQQAQE = SQRTRRMQQAQE*SQRTRRMQQAQE*SQRTRRMQQAQE

      QZZE         = SQRTAEE +   &
                     half*( one_rija0ev*   &
                            qm2_params%multip_2c_elec_params(2,iqm)*   &
                            (SQRTRRMQQAQE-SQRTRRAQQAQE)   &
                           - half*A2_TO_BOHRS2xAU_TO_EV*   &
                            (SQRTRRMQQAQE+SQRTRRAQQAQE-two*SQRTRR2AQE) )

      DGX(2) = vec_qm_mm1*DZE
      DGX(3) = vec_qm_mm1*QZZE
      DGX(4) = vec_qm_mm1*QXXE

      DGY(2) = vec_qm_mm2*DZE
      DGY(3) = vec_qm_mm2*QZZE
      DGY(4) = vec_qm_mm2*QXXE

      DGZ(2) = vec_qm_mm3*DZE
      DGZ(3) = vec_qm_mm3*QZZE
      DGZ(4) = vec_qm_mm3*QXXE

! This routine takes the electron-core integral derivatives
! DG which have been computed for a local frame and rotates
! them to molecular frame. The rotated derivatives are returned
! in DR 

! determines the transformation (Tx,Ty,Tz) and derivatives
! of the transformation (TDx,TDy,TDz) involved in rotating
! from a local frame to the molecular frame for calculation
! of electron-core interactions.
        RXY2  = vec_qm_mm1*vec_qm_mm1+vec_qm_mm2*vec_qm_mm2   
        RYZ2  = vec_qm_mm2*vec_qm_mm2+vec_qm_mm3*vec_qm_mm3
        RZX2  = vec_qm_mm3*vec_qm_mm3+vec_qm_mm1*vec_qm_mm1
        LRXY2 = (RXY2 .LT. AXIS_TOL)
        LRYZ2 = (RYZ2 .LT. AXIS_TOL)
        LRZX2 = (RZX2 .LT. AXIS_TOL)

! Zeros entire array of 3
        XTDX = zero
        YTDX = zero
        ZTDX = zero
        XTDY = zero
        YTDY = zero
        ZTDY = zero
        XTDZ = zero
        YTDZ = zero
        ZTDZ = zero

        if (.NOT.(LRXY2 .OR. LRYZ2 .OR. LRZX2)) then
           oneRXY = one/sqrt(RXY2)

! RW and MC: rearranged order here slightly for speed
           TZ(3) = oneRIJ/oneRXY
           TZ3i  = one/TZ(3)           ! Inverse of TZ(3) 
!                                      ! to avoid other divisions.
           TZ3i2 = TZ3i*TZ3i           ! Square of 1/TZ(3)

           TX(1) = vec_qm_mm1*oneRIJ
           TX(2) = vec_qm_mm2*oneRIJ
           TX(3) = vec_qm_mm3*oneRIJ

           TY(1) =-TX(2)*SIGN(one,TX(1))*TZ3i
           TY(2) = ABS(TX(1)*TZ3i)
           TY(3) = zero

           TZ(1) =-TX(1)*TX(3)*TZ3i
           TZ(2) =-TX(2)*TX(3)*TZ3i

           TERMX = TX(1)*oneRIJ
           TERMY = TX(2)*oneRIJ
           TERMZ = TX(3)*oneRIJ

           XTDX(1)= oneRIJ-TX(1)*TERMX
           YTDX(1)=-TX(1)*TERMY
           ZTDX(1)=-TX(1)*TERMZ

           XTDX(2)=-TX(2)*TERMX
           YTDX(2)= oneRIJ-TX(2)*TERMY
           ZTDX(2)=-TX(2)*TERMZ

           XTDX(3)=-TX(3)*TERMX
           YTDX(3)=-TX(3)*TERMY
           ZTDX(3)= oneRIJ-TX(3)*TERMZ

           XTDZ(3)= TX(1)*oneRXY-TZ(3)*TERMX
           YTDZ(3)= TX(2)*oneRXY-TZ(3)*TERMY
           ZTDZ(3)=-TZ(3)*TERMZ

           XTDY(1)=-XTDX(2)*TZ3i+TX(2)*XTDZ(3)*TZ3i2
           YTDY(1)=-YTDX(2)*TZ3i+TX(2)*YTDZ(3)*TZ3i2
           ZTDY(1)=-ZTDX(2)*TZ3i+TX(2)*ZTDZ(3)*TZ3i2

           XTDY(1)= XTDY(1)*sign(one,TX(1))
           YTDY(1)= YTDY(1)*sign(one,TX(1))
           ZTDY(1)= ZTDY(1)*sign(one,TX(1))

           XTDY(2)= XTDX(1)*TZ3I-TX(1)*XTDZ(3)*TZ3I2
           YTDY(2)= YTDX(1)*TZ3I-TX(1)*YTDZ(3)*TZ3I2
           ZTDY(2)= ZTDX(1)*TZ3I-TX(1)*ZTDZ(3)*TZ3I2

           XTDY(2)= XTDY(2)*sign(one,TX(1))
           YTDY(2)= YTDY(2)*sign(one,TX(1))
           ZTDY(2)= ZTDY(2)*sign(one,TX(1))

! Don't need to zero again they have been zeroed above.
! XTDY(3)=0.0D0
! YTDY(3)=0.0D0
! ZTDY(3)=0.0D0

! Note: Ross Walker and Mike Crowley
!      we could factor out TZ3I here or we could pre-compute
!      -TX(3)*TZ3I etc etc. But for the momemnt we will leave
!      it as is since this is really just doing the compiler's
!      work.
           XTDZ(1)=-TX(3)*XTDX(1)*TZ3I-TX(1)*XTDX(3)*TZ3I   &
                   +TX(1)*TX(3)*XTDZ(3)*TZ3I2
           YTDZ(1)=-TX(3)*YTDX(1)*TZ3I-TX(1)*YTDX(3)*TZ3I   &
                   +TX(1)*TX(3)*YTDZ(3)*TZ3I2
           ZTDZ(1)=-TX(3)*ZTDX(1)*TZ3I-TX(1)*ZTDX(3)*TZ3I   &
                   +TX(1)*TX(3)*ZTDZ(3)*TZ3I2

           XTDZ(2)=-TX(3)*XTDX(2)*TZ3I-TX(2)*XTDX(3)*TZ3I   &
                   +TX(2)*TX(3)*XTDZ(3)*TZ3I2
           YTDZ(2)=-TX(3)*YTDX(2)*TZ3I-TX(2)*YTDX(3)*TZ3I   &
                   +TX(2)*TX(3)*YTDZ(3)*TZ3I2
           ZTDZ(2)=-TX(3)*ZTDX(2)*TZ3I-TX(2)*ZTDX(3)*TZ3I   &
                   +TX(2)*TX(3)*ZTDZ(3)*TZ3I2

        else if (LRXY2) THEN
! molecular z axis is parallel to datomic z axis
           TX(1) = zero
           TX(2) = zero
           TX(3) = sign(one,vec_qm_mm3)
           TY(1) = zero
           TY(2) = one
           TY(3) = zero
           TZ(1) = TX(3)
           TZ(2) = zero
           TZ(3) = zero
           XTDX(1)= oneRIJ                       ! X axis
           XTDZ(3)=-oneRIJ
           YTDX(2)= oneRIJ                       ! Y axis
           YTDY(3)=-TX(3)*oneRIJ
!                                                ! Z axis
        else if (LRYZ2) THEN
! molecular x azis is parallel to diatomic z axis
           TX(1) = sign(one,vec_qm_mm1)
           TX(2) = zero
           TX(3) = zero
           TY(1) = zero
           TY(2) = TX(1)
           TY(3) = zero
           TZ(1) = zero
           TZ(2) = zero
           TZ(3) = one
!                                                ! X axis
           YTDX(2)= oneRIJ                       ! Y axis
           YTDY(1)=-oneRIJ
           ZTDX(3)= oneRIJ                       ! Z axis
           ZTDZ(1)=-TX(1)*oneRIJ

        else
! molecular y axis is paralle to diatomic z axis
           TX(1) = zero
           TX(2) = sign(one,vec_qm_mm2)
           TX(3) = zero
           TY(1) =-TX(2)
           TY(2) = zero
           TY(3) = zero
           TZ(1) = zero
           TZ(2) = zero
           TZ(3) = one
           XTDX(1)= oneRIJ                       ! X axis
           XTDY(2)= oneRIJ
!                                                ! Y axis
           ZTDX(3)= oneRIJ                       ! Z axis
           ZTDZ(2)=-TX(2)*oneRIJ

        end if                 ! if (.NOT.(LRXY2 .OR. LRYZ2 .OR. LRZX2))


      ISP=0
      do K=1,n_atomic_orb
         KK=K-1
         do L=K,n_atomic_orb
            LL=L-1
            ISP=ISP+1
            IF (LL .EQ. 0) THEN
               DRX(ISP)=DGX(1)                                   !(SS/SS)
               DRY(ISP)=DGY(1)                                   !(SS/SS)
               DRZ(ISP)=DGZ(1)                                   !(SS/SS)
            ELSE IF(KK .EQ. 0) THEN
               DRX(ISP)=DGX(2)*TX(LL)+qm_mm_e_repul(2)*XTDX(LL)  !(SP/SS)
               DRY(ISP)=DGY(2)*TX(LL)+qm_mm_e_repul(2)*YTDX(LL)  !(SP/SS)
               DRZ(ISP)=DGZ(2)*TX(LL)+qm_mm_e_repul(2)*ZTDX(LL)  !(SP/SS)
            ELSE
               DRX(ISP)=DGX(3)*TX(KK)*TX(LL) +   &
                        qm_mm_e_repul(3)*(XTDX(KK)*TX(LL) +   &
                        TX(KK)*XTDX(LL)) +   &
                        DGX(4)*(TY(KK)*TY(LL) +   &
                        TZ(KK)*TZ(LL)) +   &
                        qm_mm_e_repul(4)*(XTDY(KK)*TY(LL) +   &
                        TY(KK)*XTDY(LL) +   &
                        XTDZ(KK)*TZ(LL)+TZ(KK)*XTDZ(LL))         !(PP/SS)

               DRY(ISP)=DGY(3)*TX(KK)*TX(LL) +   &
                        qm_mm_e_repul(3)*(YTDX(KK)*TX(LL) +   &
                        TX(KK)*YTDX(LL)) +   &
                        DGY(4)*(TY(KK)*TY(LL) +   &
                        TZ(KK)*TZ(LL)) +   &
                        qm_mm_e_repul(4)*(YTDY(KK)*TY(LL) +   &
                        TY(KK)*YTDY(LL) +   &
                        YTDZ(KK)*TZ(LL)+TZ(KK)*YTDZ(LL))         !(PP/SS) 

               DRZ(ISP)=DGZ(3)*TX(KK)*TX(LL) +    &
                        qm_mm_e_repul(3)*(ZTDX(KK)*TX(LL) +   &
                        TX(KK)*ZTDX(LL)) +   &
                        DGZ(4)*(TY(KK)*TY(LL) +   &
                        TZ(KK)*TZ(LL)) +   &
                        qm_mm_e_repul(4)*(ZTDY(KK)*TY(LL) +   &
                        TY(KK)*ZTDY(LL) +   &
                        ZTDZ(KK)*TZ(LL)+TZ(KK)*ZTDZ(LL))         !(PP/SS)
            END IF
         end do
      end do

! The first derivative of nuclear repulsion term
      C1    = qm_atom_core*mm_charge
      FNUCX = dgX(1)*c1
      FNUCY = dgy(1)*c1
      FNUCZ = dgz(1)*c1
      EXP3  = (EXP1+EXP2)*abs(c1)
      EXP4  = qm_mm_e_repul(1)*onerij*(alpa*EXP1+ALPH_MM*EXP2)*abs(c1)
      FNUCX = FNUCX-vec_qm_mm1*EXP4
      FNUCY = FNUCY-vec_qm_mm2*EXP4
      FNUCZ = FNUCZ-vec_qm_mm3*EXP4

      FNUCX = FNUCX+dgX(1)*EXP3
      FNUCY = FNUCY+dgy(1)*EXP3
      FNUCZ = FNUCZ+dgz(1)*EXP3

      FABX=zero
      FABY=zero
      FABZ=zero

! MM core affecting AO's on QM atom
      ISP=0
      do M=1,n_atomic_orb
         BB=one
         do N=M,n_atomic_orb
            MN  =M+qm2_params%pascal_tri1(N)
            ISP =ISP+1
            FABX=FABX-BB*mm_charge*PSUM(MN)*DRX(ISP)
            FABY=FABY-BB*mm_charge*PSUM(MN)*DRY(ISP)
            FABZ=FABZ-BB*mm_charge*PSUM(MN)*DRZ(ISP)                    
            BB  = two
         end do
      end do

! for AM1/PM3/PM3CARB1: Gaussian core-core terms.
!     PDDGPM3 should be done differently.
      If(qmmm_nml%QMMM_Gauss) then
         qmitype= qmmm_struct%qm_atom_type(iqm)
         oner2  = oneRIJ*oneRIJ
         anam1  = zero
         do i=1,qm2_params%num_fn(qmitype)
            temp_real = RIJ-qm2_params%FN3(i,qmitype)
            temp_real2= qm2_params%FN2(i,qmitype)*temp_real*temp_real

! Skip when the exponential is close enough to be zero
            if (temp_real2 .LT. EXPONENTIAL_CUTOFF) then
               anam1 = anam1 + qm2_params%FN1(i,qmitype)* &
                               EXP(-temp_real2)* &
                              (oner2 + two*qm2_params%FN2(i,qmitype)* &
                                           temp_real*oneRIJ)
            end if
         end do

         anam1 = -anam1*c1
         FNUCX = FNUCX+anam1*vec_qm_mm1*oneRIJ
         FNUCY = FNUCY+anam1*vec_qm_mm2*oneRIJ
         FNUCZ = FNUCZ+anam1*vec_qm_mm3*oneRIJ
      End if

! copy to main gradient array
      pair_force(1) = (FABX+FNUCX)*EV_TO_KCAL 
      pair_force(2) = (FABY+FNUCY)*EV_TO_KCAL 
      pair_force(3) = (FABZ+FNUCZ)*EV_TO_KCAL 

      return
      END SUBROUTINE qm2_deriv_qmmm_heavy

#endif /* (mainsquatn)*/

SUBROUTINE qm2_dhc_BLANK
  !
  ! dummy routine for compilation
  !
  RETURN
END SUBROUTINE qm2_dhc_BLANK

