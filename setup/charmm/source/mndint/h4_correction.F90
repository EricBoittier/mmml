! H4 correction terms
module H4_mndo
   use chm_kinds
   use number
#if KEY_MNDO97==1 /*mndo97*/
   use mndo97  , only    : NUMAT
   use qm1_info, only    : qm_main_r
   use qm1_constant, only: HALFPI
#endif /* (mndo97)*/

#if KEY_MNDO97==1 /*mndo97*/
   implicit none
   ! ======================================
   ! Parameters
   ! H4 correction
   real(chm_real) :: para_oh_o 
   real(chm_real) :: para_oh_n 
   real(chm_real) :: para_nh_o 
   real(chm_real) :: para_nh_n 
   real(chm_real) :: multiplier_wh_o
   real(chm_real) :: multiplier_nh4
   real(chm_real) :: multiplier_coo
   !
   ! H repulsion
   real(chm_real) :: hh_rep_k
   real(chm_real) :: hh_rep_e 
   real(chm_real) :: hh_rep_r0

   ! ======================================
   ! Default parameters
   !  H4 correction                         !     MNDO   AM1    PM3    AM1/d  MNDO/d
   !                                             (AM1)                 (AM1)  (AM1)
   real(chm_real),parameter :: para_oh_o_def(5)=(/4.89d0,4.89d0,2.71d0,4.89d0,4.89d0/)
   real(chm_real),parameter :: para_oh_n_def(5)=(/6.23d0,6.23d0,4.37d0,6.23d0,6.23d0/)
   real(chm_real),parameter :: para_nh_o_def(5)=(/2.54d0,2.54d0,2.29d0,2.54d0,2.54d0/)
   real(chm_real),parameter :: para_nh_n_def(5)=(/4.56d0,4.56d0,3.86d0,4.56d0,4.56d0/)
   real(chm_real),parameter :: mtpl_wh_o_def(5)=(/0.49d0,0.49d0,0.91d0,0.49d0,0.49d0/)
   real(chm_real),parameter :: mtpl_nh4_def(5) =(/2.78d0,2.78d0,2.54d0,2.78d0,2.78d0/)
   real(chm_real),parameter :: mtpl_coo_def(5) =(/1.08d0,1.08d0,0.89d0,1.08d0,1.08d0/)
   real(chm_real),parameter :: hh_rep_k_def(5) =(/0.90d0,0.90d0,0.90d0,0.90d0,0.90d0/)
   real(chm_real),parameter :: hh_rep_e_def(5) =(/4.46d0,4.46d0,6.86d0,4.46d0,4.46d0/)
   real(chm_real),parameter :: hh_rep_r0_def(5)=(/2.11d0,2.11d0,2.23d0,2.11d0,2.11d0/)

   !
   ! define an element and radii array
   !character(len=2), parameter :: element(94)= (/                                           &
   !                 'H ', 'He',                                                             &
   !                 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne',                         &
   !                 'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar',                         &
   !                 'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',       &
   !                 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',                               &
   !                 'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',       &
   !                 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe',                               &
   !                 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', &
   !                 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', &
   !                 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',                         &
   !                 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U ', 'Np', 'Pu'/)
   real(chm_real),parameter :: radius(94)=(/                                                            &
                    0.37d0,0.32d0,                                                                      &
                    1.34d0,0.90d0,0.82d0,0.77d0,0.75d0,0.73d0,0.71d0,0.69d0,                            &
                    1.54d0,1.30d0,1.18d0,1.11d0,1.06d0,1.02d0,0.99d0,0.97d0,                            &
                    1.96d0,1.74d0,1.44d0,1.36d0,1.25d0,1.27d0,1.39d0,1.25d0,1.26d0,1.21d0,1.38d0,       &
                    1.31d0,1.26d0,1.22d0,1.19d0,1.16d0,1.14d0,1.10d0,                                   &
                    2.11d0,1.92d0,1.62d0,1.48d0,1.37d0,1.45d0,1.56d0,1.26d0,1.35d0,1.31d0,1.53d0,       &
                    1.48d0,1.44d0,1.41d0,1.38d0,1.35d0,1.33d0,1.30d0,                                   &
                    2.25d0,1.98d0,1.69d0,0.00d0,0.00d0,0.00d0,0.00d0,0.00d0,0.00d0,0.00d0,0.00d0,0.00d0,&
                    0.00d0,0.00d0,0.00d0,0.00d0,1.60d0,1.50d0,1.38d0,1.46d0,1.59d0,1.28d0,1.37d0,1.28d0,&
                    1.44d0,1.49d0,0.00d0,0.00d0,1.46d0,0.00d0,0.00d0,1.45d0,                            &
                    0.00d0,0.00d0,0.00d0,0.00d0,0.00d0,0.00d0,0.00d0,0.00d0 /)

   ! cut-offs for H4 correction
   real(chm_real), parameter :: hb_r_cutoff = 5.5d0    ! more distant donor-acceptor pairs do not contribute
   real(chm_real), parameter :: hb_r_0 = 1.5d0         ! closer donor-acceptor pairs do not contribute
   real(chm_real), parameter :: max_xh_bond = 1.15d0   ! maximal X-H covalent bond distance

   ! miscellanea
   real(chm_real),parameter :: r_HALFPI = one/HALFPI
   logical :: q_h4corr   =.false.                    ! logical flag to use H4 term.
   logical :: q_gradient = .true.                    ! calculate/not calculate the gradient (T/F)
   integer, save             :: ij_indx_size = 0, &  ! size for ij_indx index
                                hh_indx_size = 0     ! size for hh_indx index
   integer, allocatable,save :: ij_indx(:,:)  ! index for the heavy atom i,j loop in the main routine
   integer, allocatable,save :: hh_indx(:,:)  ! index for the H     atom i,j loop in the main routine
   real(chm_real),allocatable,save :: dist_ij_d(:), &! distance for donor to all other atoms. 
                                      dist_ij_a(:), &! distance for acceptor to all other atoms.
                                      dist_ij_c(:), &
                                      dist_ij_o(:)

   !----- gradients array ----------------
   type deriv_array                                ! template array to store derivatives
      real(chm_real), allocatable    :: d_radial_d(:,:),  &
                                        d_radial_a(:,:),  &
                                        d_angular_d(:,:), &
                                        d_angular_a(:,:), &
                                        d_angular_h(:,:), &
                                        d_bs_d(:,:),      &
                                        d_bs_a(:,:),      &
                                        d_bs_h(:,:) 
   endtype deriv_array
   type (deriv_array),save :: grad

   !
   contains

   !
   subroutine h4_correction_setup(COMLYN,COMLEN)
      use chm_kinds
      use dimens_fcm
      use number
      use string
      use stream
      use qm1_info, only : qm_control_r,qm_main_r

      implicit none
      CHARACTER(len=*):: COMLYN
      INTEGER         :: COMLEN

      logical         :: q_read
      integer         :: natom_qm, ndim2

      !
      q_read = (indxa(COMLYN, COMLEN, 'PRED') > 0)
      call get_h4_param(q_read)

      !
      q_gradient = .true. ! default
      natom_qm   = qm_main_r%numat
      ndim2      = natom_qm*(natom_qm-1)/2

      ! allocate memory
      allocate(ij_indx(2,ndim2))
      allocate(hh_indx(2,ndim2))
      allocate(dist_ij_d(natom_qm))
      allocate(dist_ij_a(natom_qm))
      allocate(dist_ij_c(natom_qm))
      allocate(dist_ij_o(natom_qm))

      allocate(grad%d_radial_d(3,natom_qm))
      allocate(grad%d_radial_a(3,natom_qm))
      allocate(grad%d_angular_d(3,natom_qm))
      allocate(grad%d_angular_a(3,natom_qm))
      allocate(grad%d_angular_h(3,natom_qm))
      allocate(grad%d_bs_d(3,natom_qm))
      allocate(grad%d_bs_a(3,natom_qm))
      allocate(grad%d_bs_h(3,natom_qm))

      if(prnlev >= 2) then
         write(outu,'(10x,'' H4 parameters'')')
         write(outu,'('' POHO/para_oh_o       :'',f10.4)' ) para_oh_o
         write(outu,'('' POHN/para_oh_n       :'',f10.4)' ) para_oh_n
         write(outu,'('' PNHO/para_nh_o       :'',f10.4)' ) para_nh_o
         write(outu,'('' PNHN/para_nh_n       :'',f10.4)' ) para_nh_n
         write(outu,'('' MWHO/multiplier_wh_o :'',f10.4)' ) multiplier_wh_o
         write(outu,'('' MNH4/multiplier_nh4  :'',f10.4)' ) multiplier_nh4
         write(outu,'('' MCOO/multiplier_coo  :'',f10.4)' ) multiplier_coo
         write(outu,'('' REPK/hh_rep_k        :'',f10.4)' ) hh_rep_k
         write(outu,'('' REPE/hh_rep_e        :'',f10.4)' ) hh_rep_e
         write(outu,'('' REPR/hh_rep_r0       :'',f10.4)' ) hh_rep_r0
      end if

      return
      !
      contains
      subroutine get_h4_param(read_local)
         implicit none
         logical :: read_local
         integer :: nqmtheory

         !
         nqmtheory = qm_control_r%iqm_mode
         ! load default parameters
         ! MNDO  : nqmtheory == 1
         ! AM1   : nqmtheory == 2
         ! PM3   : nqmtheory == 3
         ! AM1/d : nqmtheory == 4
         ! MNDO/d: nqmtheory == 5
         para_oh_o      = para_oh_o_def(nqmtheory)
         para_oh_n      = para_oh_n_def(nqmtheory)
         para_nh_o      = para_nh_o_def(nqmtheory)
         para_nh_n      = para_nh_n_def(nqmtheory)
         multiplier_wh_o= mtpl_wh_o_def(nqmtheory)
         multiplier_nh4 = mtpl_nh4_def(nqmtheory)
         multiplier_coo = mtpl_coo_def(nqmtheory)
         hh_rep_k       = hh_rep_k_def(nqmtheory)
         hh_rep_e       = hh_rep_e_def(nqmtheory)
         hh_rep_r0      = hh_rep_r0_def(nqmtheory)
         !
         if(read_local) then
            ! read in parameters from the input
            para_oh_o      = gtrmf(COMLYN,COMLEN,'POHO',para_oh_o)
            para_oh_n      = gtrmf(COMLYN,COMLEN,'POHN',para_oh_n)
            para_nh_o      = gtrmf(COMLYN,COMLEN,'PNHO',para_nh_o)
            para_nh_n      = gtrmf(COMLYN,COMLEN,'PNHN',para_nh_n)
            multiplier_wh_o= gtrmf(COMLYN,COMLEN,'MWHO',multiplier_wh_o)
            multiplier_nh4 = gtrmf(COMLYN,COMLEN,'MNH4',multiplier_nh4)
            multiplier_coo = gtrmf(COMLYN,COMLEN,'MCOO',multiplier_coo)
            hh_rep_k       = gtrmf(COMLYN,COMLEN,'REPK',hh_rep_k)
            hh_rep_e       = gtrmf(COMLYN,COMLEN,'REPE',hh_rep_e)
            hh_rep_r0      = gtrmf(COMLYN,COMLEN,'REPR',hh_rep_r0)
         end if
         return
      end subroutine get_h4_param
   end subroutine h4_correction_setup

   subroutine h4_correction(EH4_corr,natom,dx,dy,dz)
      use chm_kinds
      use dimens_fcm
      use number
      use string
      use stream
      use qm1_info, only : qm_control_r,qm_main_r
#if KEY_PARALLEL==1
      use parallel,only: mynod, numnod, gcomb
#endif

      implicit none
      real(chm_real):: EH4_corr
      integer       :: natom
      real(chm_real):: dx(natom),dy(natom),dz(natom)

      ! local variables
      integer       :: i,j,k,kk,icnt,n
      integer       :: d_i, a_i, h_i                         ! donor/acceptor/H indices
      real(chm_real):: rda, rih, rjh, r_rda                  ! donor-acceptor distance; donor/acceptor-H distance
      real(chm_real):: rdh, rah                              ! donor/acceptor-H distance
      real(chm_real):: r                                     ! H-H distance
      real(chm_real):: angle
      real(chm_real):: rdhs, ravgs                           ! bond switching
      real(chm_real):: xyz_i(3),xyz_j(3),xyz_h(3),xyz_d(3),xyz_a(3),xyz_dh(3),xyz_ah(3), &
                       xyz_c(3),xyz_k(3),xyz_o(3)
      real(chm_real):: dxyz_i(3),dxyz_local(3)
      real(chm_real):: a,cdist,odist,dd,x,xd,xd2,v,fv,fv2,slope,r_tmp1,r_tmp2,r_tmp3
      !!real(chm_real),parameter :: r_HALFPI = one/HALFPI

      ! energy terms
      real(chm_real):: e_para                                ! semiempiral method-specific parameters
      real(chm_real):: e_bond_switch, e_radial, e_angular
      real(chm_real):: e_scale_w, e_scale_chd, e_scale_cha
      real(chm_real):: e_corr                                ! total energy correction for one donor-acceptor pair
      real(chm_real):: e_corr_sum                            ! cumulative energy correction for the dimer
      real(chm_real):: e_corr_sum_hh                         ! cumulative energy correction for H-H repulsion

      ! derivatives and Cartesian gradients
      real(chm_real):: d_angular, d_bs, d_radial, d_rad
      real(chm_real):: mult_radial, mult_angular, mult_bs    ! gradient coords scaling factors
      real(chm_real):: mult_wh_o, mult_nh4, mult_coo         ! scaled groups gradient coords scaling factors

      ! Scaling variables
      real(chm_real):: sign_wat, hydro, others
      real(chm_real):: cv_o1, cv_o2, cv_cc
      real(chm_real):: f_o1, f_o2, f_cc
      integer       :: o1, o2, cc
#if KEY_PARALLEL==1
      integer       :: m_cnt
#endif

      !
      logical, save :: q_first=.true.  

      ! do some checking and filling in arrays.
      if(q_first) then
         ! 1) loop for heavy O and N atoms
         icnt = 0
#if KEY_PARALLEL==1
         m_cnt = 0
#endif
         do i = 1, qm_main_r%numat
            if(qm_main_r%nat(i) == 7 .or. qm_main_r%nat(i) == 8) then
               do j = 1, i-1
                  if (qm_main_r%nat(j) == 7 .or. qm_main_r%nat(j) == 8) then
#if KEY_PARALLEL==1
                     m_cnt = m_cnt + 1
                     if(mynod /= mod(m_cnt,numnod)) cycle 
#endif
                     icnt = icnt + 1
                     ij_indx(1,icnt) = i
                     ij_indx(2,icnt) = j
                  end if      ! j atom N or O
               end do         ! loop over j atoms
            end if            ! i atom N or O
         end do               ! loop over i atoms
         ij_indx_size = icnt  ! save the total number of terms.

         ! 2) loop for H atoms
         icnt = 0
#if KEY_PARALLEL==1
         m_cnt = 0
#endif
         do i = 1, qm_main_r%numat
            if(qm_main_r%nat(i) == 1) then
               do j = 1, i-1
                  if (qm_main_r%nat(j) == 1) then
#if KEY_PARALLEL==1
                     m_cnt = m_cnt + 1
                     if(mynod /= mod(m_cnt,numnod)) cycle  
#endif
                     icnt = icnt + 1
                     hh_indx(1,icnt) = i
                     hh_indx(2,icnt) = j
                  end if
               end do
            end if
         end do
         hh_indx_size = icnt  ! save the total number of terms

         q_first =.false.
      end if

      ! initializations
      EH4_corr      = zero
      e_corr_sum    = zero
      e_corr_sum_hh = zero
      do i= 1, qm_main_r%numat
         do k = 1, 3
            qm_main_r%qm_grads(k,i) = zero
            grad%d_radial_d(k,i)    = zero
            grad%d_radial_a(k,i)    = zero
            grad%d_angular_d(k,i)   = zero
            grad%d_angular_a(k,i)   = zero
            grad%d_angular_h(k,i)   = zero
            grad%d_bs_d(k,i)        = zero
            grad%d_bs_a(k,i)        = zero
            grad%d_bs_h(k,i)        = zero
         end do
      end do

      ! Iterate over donor-acceptor pairs; no duplicates
      loopij: do icnt = 1, ij_indx_size
         i = ij_indx(1,icnt) 
         j = ij_indx(2,icnt) 
         xyz_i(1:3) = qm_main_r%qm_coord(1:3,i)
         xyz_j(1:3) = qm_main_r%qm_coord(1:3,j)

         ! donor-acceptor distance
         rda = distance(xyz_i,xyz_j)

         ! check whether r donor-acceptor is within the cutoff range, and
         ! continue only when in range where the correction acts.
         if ( rda <= hb_r_0 .or. rda >= hb_r_cutoff) cycle loopij

         ! Iterate over hydrogens
         loophi: do h_i = 1, qm_main_r%numat
            if (qm_main_r%nat(h_i) /= 1) cycle loophi  
          
            xyz_h(1:3) = qm_main_r%qm_coord(1:3,h_i)

            ! Calculate donor/acceptor distances to hydrogen
            rih = distance(xyz_i,xyz_h)
            rjh = distance(xyz_j,xyz_h)

            ! check donor/acceptor-H-acceptor/donor angle
            angle = 2*HALFPI - atomangle(xyz_i,xyz_h,xyz_j)

            ! Check for H-bond directionality (here: i-h_i-j angle must be larger that 90deg)
            if (angle >= HALFPI) cycle loophi

            ! Identify donor/acceptor atoms; donor is the closer one to H.
            if (rih <= rjh) then
                d_i = i
                a_i = j
                rdh = rih
                rah = rjh
                xyz_d(1:3) = xyz_i(1:3)
                xyz_a(1:3) = xyz_j(1:3)
            else
                d_i = j
                a_i = i
                rdh = rjh
                rah = rih
                xyz_d(1:3) = xyz_j(1:3)
                xyz_a(1:3) = xyz_i(1:3)
            end if  ! identify donor/acceptor atoms

            ! compute distance from donor to all other qm atoms
            do k=1,qm_main_r%numat
               dist_ij_d(k) = zero
               dist_ij_a(k) = zero
            end do
            if( (qm_main_r%nat(d_i) == 8 .and. qm_main_r%nat(a_i) == 8) .or.  &  ! water scaling
                (qm_main_r%nat(d_i) == 7)                               .or.  &  ! NR4+  scaling
                (qm_main_r%nat(a_i) == 8) ) then                                 ! COO-  scaling
                if(qm_main_r%nat(a_i) == 8) then
                   do k=1,qm_main_r%numat
                      dist_ij_d(k) = sqrt( (xyz_d(1)-qm_main_r%qm_coord(1,k))**2 + &
                                           (xyz_d(2)-qm_main_r%qm_coord(2,k))**2 + &
                                           (xyz_d(3)-qm_main_r%qm_coord(3,k))**2 ) 
                      dist_ij_a(k) = sqrt( (xyz_a(1)-qm_main_r%qm_coord(1,k))**2 + &
                                           (xyz_a(2)-qm_main_r%qm_coord(2,k))**2 + &
                                           (xyz_a(3)-qm_main_r%qm_coord(3,k))**2 )
                   end do
                else
                   do k=1,qm_main_r%numat
                      dist_ij_d(k) = sqrt( (xyz_d(1)-qm_main_r%qm_coord(1,k))**2 + &
                                           (xyz_d(2)-qm_main_r%qm_coord(2,k))**2 + &
                                           (xyz_d(3)-qm_main_r%qm_coord(3,k))**2 )
                   end do
                end if
            end if

            ! (1) Radial term
            !     get radial energy and gradient
            e_radial = E_Radial_FN(rda,d_radial,q_gradient)

            if (q_gradient) then
                ! Cartesian gradients on donor/acceptor atoms
                r_rda = one/rda
                do kk=1,3
                   grad%d_radial_d(kk,d_i) = (xyz_d(kk) - xyz_a(kk))*r_rda*d_radial
                   grad%d_radial_a(kk,a_i) =-grad%d_radial_d(kk,d_i)
                end do
            end if  ! radial gradiant (if requested)

            ! (2) Angular term
            a = angle * r_HALFPI
            e_angular = E_Angular_FN(a,d_angular,q_gradient)

            if (q_gradient) then
                xyz_dh(1:3) = xyz_d(1:3)-xyz_h(1:3)
                xyz_ah(1:3) = xyz_a(1:3)-xyz_h(1:3)

                ! dot product of donor-H/acceptor-H bond vectors
                dd = DOT_PRODUCT(xyz_dh(1:3),xyz_ah(1:3)) 

                ! what is that? :
                r_tmp1 = one/(rdh*rah)
                r_tmp2 = dd/((rdh**3)*rah)
                r_tmp3 = dd/((rah**3)*rdh)
                x      =-d_angular/sqrt(one-(dd*r_tmp1)**2)

                ! Cartesian gradients on:
                do k=1,3
                   grad%d_angular_d(k,d_i) =-x*( xyz_ah(k)*r_tmp1  - xyz_dh(k)*r_tmp2 )       ! donors
                   grad%d_angular_a(k,a_i) =-x*( xyz_dh(k)*r_tmp1  - xyz_ah(k)*r_tmp3 )       ! acceptors
                   grad%d_angular_h(k,h_i) =-grad%d_angular_d(k,d_i)- grad%d_angular_a(k,a_i) ! hydrogen
                end do
            end if  ! angular gradient (if requested)

            ! Energy coefficients
            if(qm_main_r%nat(d_i) == 8 .and. qm_main_r%nat(a_i) == 8) then
               ! O-H---O pair
               e_para = para_oh_o
            else if(qm_main_r%nat(d_i) == 8 .and. qm_main_r%nat(a_i) == 7) then
               ! O-H---N pair
               e_para = para_oh_n
            else if(qm_main_r%nat(d_i) == 7 .and. qm_main_r%nat(a_i) == 8) then
               ! N-H---O pair
               e_para = para_nh_o
            else if(qm_main_r%nat(d_i) == 7 .and. qm_main_r%nat(a_i) == 7) then
               ! N-H---N pair
               e_para = para_nh_n
            end if

            ! (3) Bond switching
            if (rdh > max_xh_bond) then
               rdhs  = rdh - max_xh_bond
               ravgs = half*(rdh + rah) - max_xh_bond
               x     = rdhs / ravgs
               e_bond_switch = BOND_switch_FN(x,d_bs,q_gradient)
               if (q_gradient) then
                    xd  = d_bs/ravgs       ! d_bs from BOND_switch_FN
                    xd2 =-half*d_bs*x/ravgs
                    r_tmp1 = one/rdh
                    r_tmp2 = one/rah
                    do k=1,3
                       xyz_dh(k)          = xyz_d(k)-xyz_h(k)
                       xyz_ah(k)          = xyz_a(k)-xyz_h(k)
                       grad%d_bs_d(k,d_i) = xyz_dh(k)*r_tmp1*xd + xyz_dh(k)*r_tmp1*xd2
                       grad%d_bs_a(k,a_i) = xyz_ah(k)*r_tmp2*xd2
                       grad%d_bs_h(k,h_i) =-grad%d_bs_d(k,d_i) - grad%d_bs_a(k,a_i)
                    end do
               end if ! bond switching gradient
            else ! no switching, no gradient:
               e_bond_switch = one
               if (q_gradient) then
                    do k=1,3
                       grad%d_bs_d(k,d_i) = zero
                       grad%d_bs_a(k,a_i) = zero
                       grad%d_bs_h(k,h_i) = zero
                    end do
               end if
            end if  ! bond switching

            ! (4) Water scaling: O-H ---- O pair?
            e_scale_w = one
            if(qm_main_r%nat(d_i) == 8 .and. qm_main_r%nat(a_i) == 8) then
               ! count hydrogens and other atoms in the vicinity
               hydro  = zero
               others = zero

               ! look for hydrogen atoms around O ... O
               do k = 1, qm_main_r%numat
                  if(qm_main_r%nat(k) == 1) then
                     hydro  = hydro  + cvalence_contribution(d_i,k,dist_ij_d(k))
                  else
                     others = others + cvalence_contribution(d_i,k,dist_ij_d(k))
                  end if
               end do

               ! if it is water
               if (hydro >= one ) then
                  sign_wat = one
                  slope    = multiplier_wh_o - one
                  v        = hydro
                  fv       = zero

                  if (v > one .and. v <= two) then
                      fv       = v - one
                      sign_wat = one
                  else if (v > two .and. v < three) then
                      fv       = three - v
                      sign_wat = -one
                  end if

                  fv2 = one - others
                  if (fv2 < zero) fv2 = zero

                  e_scale_w = one + slope * fv * fv2
               end if
            end if

            ! (5) Charged groups
            e_scale_chd = one
            e_scale_cha = one

            ! (5-1) Scaled groups: NR4+
            if (qm_main_r%nat(d_i) == 7) then
               slope = multiplier_nh4 - one
               v     = zero
               do k = 1, qm_main_r%numat
                  v  = v + cvalence_contribution(d_i,k,dist_ij_d(k))
               end do

               if (v > three) then
                  v = v - three
               else
                  v = zero
               end if

               e_scale_chd = one + slope*v
            end if

            ! (5-2) Scaled groups: COO-
            f_o1 = zero
            f_o2 = zero
            f_cc = zero
            o1   = a_i
            o2   =-1
            cc   =-1
            if (qm_main_r%nat(a_i) == 8) then
               slope = multiplier_coo - one

               ! Search for the closest C atom
               cdist = 9.9d9
               cv_o1 = zero  ! 01 valence

               do k = 1, qm_main_r%numat
                  dd         = dist_ij_a(k)             ! distance(xyz_a,qm_main_r%qm_coord(1:3,k))
                  v          = cvalence_contribution(o1,k,dist_ij_a(k))
                  cv_o1      = cv_o1 + v  ! sum 01 Valence
                  if (v > zero .and. qm_main_r%nat(k) == 6 .and. dd < cdist) then
                     cdist = dd
                     cc = k
                  end if
               end do

               ! If C atom found, look for the second O
               if (cc /= -1) then
                  odist = 9.9d9
                  cv_cc = zero

                  xyz_c(1:3) = qm_main_r%qm_coord(1:3,cc)
                  do k = 1, qm_main_r%numat
                     xyz_k(1:3)   = qm_main_r%qm_coord(1:3,k)
                     dd           = distance(xyz_c,xyz_k)
                     v            = cvalence_contribution(cc,k,dd)
                     cv_cc        = cv_cc + v
                     dist_ij_c(k) = dd          ! save for later
                     if (v > zero .and. k /= o1 .and. qm_main_r%nat(k) == 8 .and. dd < odist) then
                        odist = dd 
                        o2 = k
                     end if
                  end do
               end if

               ! O1-C-O2 triad
               if (o2 /= -1) then
                  xyz_o(1:3) = qm_main_r%qm_coord(1:3,o2)
                  cv_o2 = zero    ! 02 valence
                                     
                  do k = 1, qm_main_r%numat
                     xyz_k(1:3)   = qm_main_r%qm_coord(1:3,k)
                     dd           = distance(xyz_o,xyz_k)
                     cv_o2        = cv_o2 + cvalence_contribution(o2,k,dd)
                     dist_ij_o(k) = dd          ! save for later
                  end do

                  f_o1 = one - abs(one - cv_o1)
                  if (f_o1 < zero) f_o1 = zero

                  f_o2 = one - abs(one - cv_o2)
                  if (f_o2 < zero) f_o2 = zero

                  f_cc = one - abs(three - cv_cc)
                  if (f_cc < zero) f_cc = zero

                  e_scale_cha = one + slope*f_o1*f_o2*f_cc
               end if
            end if ! scaled COO-

            ! Final energy calculation
            e_corr     = e_para*e_radial*e_angular*e_bond_switch*e_scale_w*e_scale_chd*e_scale_cha
            e_corr_sum = e_corr_sum + e_corr

            !=========================================================================================
            ! Total gradient
            ! Scaling factors
            mult_radial  = e_para *            e_angular * e_bond_switch * e_scale_w * e_scale_chd * e_scale_cha
            mult_angular = e_para * e_radial *             e_bond_switch * e_scale_w * e_scale_chd * e_scale_cha
            mult_bs      = e_para * e_radial * e_angular *                 e_scale_w * e_scale_chd * e_scale_cha
            mult_wh_o    = e_para * e_radial * e_angular * e_bond_switch *             e_scale_chd * e_scale_cha
            mult_nh4     = e_para * e_radial * e_angular * e_bond_switch * e_scale_w *               e_scale_cha
            mult_coo     = e_para * e_radial * e_angular * e_bond_switch * e_scale_w * e_scale_chd      

            ! radial, angular, bond switch
            qm_main_r%qm_grads(1:3,d_i) = qm_main_r%qm_grads(1:3,d_i)            &
                                         +grad%d_radial_d(1:3,d_i)*mult_radial   &
                                         +grad%d_angular_d(1:3,d_i)*mult_angular &
                                         +grad%d_bs_d(1:3,d_i)*mult_bs
            qm_main_r%qm_grads(1:3,a_i) = qm_main_r%qm_grads(1:3,a_i)            &
                                         +grad%d_radial_a(1:3,a_i)*mult_radial   &
                                         +grad%d_angular_a(1:3,a_i)*mult_angular &
                                         +grad%d_bs_a(1:3,a_i)*mult_bs
            qm_main_r%qm_grads(1:3,h_i) = qm_main_r%qm_grads(1:3,h_i)            &
                                         +grad%d_angular_h(1:3,h_i)*mult_angular &
                                         +grad%d_bs_h(1:3,h_i)*mult_bs

            ! water scaling
            if (q_gradient .and. e_scale_w /= one) then
               slope       = multiplier_wh_o - one 
               dxyz_i(1:3) = zero
               xyz_d(1:3)  = qm_main_r%qm_coord(1:3,d_i)
               do k = 1, qm_main_r%numat
                  if (k == d_i) cycle

                  xyz_k(1:3) = qm_main_r%qm_coord(1:3,k)
                  x          = dist_ij_d(k)             ! distance(xyz_d,xyz_k)
                  xd         = cvalence_contribution_d(d_i,k,x)
                  if (qm_main_r%nat(k) == 1) then
                     r_tmp1 = -xd*sign_wat/x
                     do kk=1,3
                        dxyz_local(kk) = (xyz_d(kk) - xyz_k(kk))*r_tmp1*slope*mult_wh_o
                        dxyz_i(kk)     = dxyz_i(kk) - dxyz_local(kk)
                        qm_main_r%qm_grads(kk,k) = qm_main_r%qm_grads(kk,k) + dxyz_local(kk)
                     end do
                  else
                     r_tmp1 = xd/x
                     do kk=1,3
                        dxyz_local(kk) = (xyz_d(kk) - xyz_k(kk))*r_tmp1*slope*mult_wh_o
                        dxyz_i(kk)     = dxyz_i(kk) - dxyz_local(kk)
                        qm_main_r%qm_grads(kk,k) = qm_main_r%qm_grads(kk,k) + dxyz_local(kk)
                     end do
                  end if
               end do
               qm_main_r%qm_grads(1:3,d_i) = qm_main_r%qm_grads(1:3,d_i) + dxyz_i(1:3)
            end if

            ! scaled groups: NR4+
            if (q_gradient .and. e_scale_chd /= one) then
               slope       = multiplier_nh4 - one
               dxyz_i(1:3) = zero
               xyz_d(1:3)  = qm_main_r%qm_coord(1:3,d_i)
               do k = 1, qm_main_r%numat
                  if (k == d_i) cycle

                  xyz_k(1:3) = qm_main_r%qm_coord(1:3,k)
                  x = dist_ij_d(k)            ! distance(xyz_d,xyz_k)
                  xd = cvalence_contribution_d(d_i,k,x)
                  r_tmp1 = -xd/x
                  do kk=1,3
                     dxyz_local(kk) = (xyz_d(kk) - xyz_k(kk))*r_tmp1*slope*mult_nh4
                     dxyz_i(kk)     = dxyz_i(kk) - dxyz_local(kk)
                     qm_main_r%qm_grads(kk,k) = qm_main_r%qm_grads(kk,k) + dxyz_local(kk)
                  end do
               end do
               qm_main_r%qm_grads(1:3,d_i) = qm_main_r%qm_grads(1:3,d_i) + dxyz_i(1:3)
            end if

            ! scaled groups: COO-
            if (q_gradient .and. (f_o1*f_o2*f_cc) /= zero) then
               slope = multiplier_coo - one
               dxyz_i(1:3) = zero
               ! atoms around 01
               do k = 1, qm_main_r%numat
                  if (k == o1) cycle
                  xyz_k(1:3) = qm_main_r%qm_coord(1:3,k)
                  x          = dist_ij_a(k)              ! distance(xyz_a,xyz_k)
                  xd         = cvalence_contribution_d(o1,k,x)  ! o1 = a_i
                  if (xd /= zero) then
                     if (cv_o1 > one) xd = -one*xd
                     xd = f_o2 * f_cc * xd
                     r_tmp1 = -xd/x
                     do kk=1,3
                        dxyz_local(kk) = (xyz_a(kk) - xyz_k(kk))*r_tmp1*slope*mult_coo
                        dxyz_i(kk)     = dxyz_i(kk)     - dxyz_local(kk)
                        qm_main_r%qm_grads(kk,k) = qm_main_r%qm_grads(kk,k) + dxyz_local(kk)
                     end do
                  end if
               end do
               qm_main_r%qm_grads(1:3,o1) = qm_main_r%qm_grads(1:3,o1) + dxyz_i(1:3)

               dxyz_i(1:3) = zero
               xyz_o(1:3)  = qm_main_r%qm_coord(1:3,o2)
               ! atoms around 02
               do k = 1, qm_main_r%numat
                  if (k == o2) cycle
                  xyz_k(1:3) = qm_main_r%qm_coord(1:3,k)
                  x          = distance(xyz_o,xyz_k)   ! dist_ij_o(k)    ! distance(xyz_o,xyz_k)
                  xd         = cvalence_contribution_d(o2,k,x)
                  if (xd /= zero) then
                     if (cv_o2 > one) then
                        xd = -one * xd
                        xd = f_o1 * f_cc * xd
                        r_tmp1 = -xd/x
                        do kk=1,3
                           dxyz_local(kk) = (xyz_o(kk) - xyz_k(kk))*r_tmp1*slope*mult_coo
                           dxyz_i(kk)     = dxyz_i(kk)     - dxyz_local(kk)
                           qm_main_r%qm_grads(kk,k) = qm_main_r%qm_grads(kk,k) + dxyz_local(kk)
                        end do
                     end if
                  end if
               end do
               qm_main_r%qm_grads(1:3,o2) = qm_main_r%qm_grads(1:3,o2) + dxyz_i(1:3)

               dxyz_i(1:3) = zero
               xyz_c(1:3) = qm_main_r%qm_coord(1:3,cc)
               ! Think of a label
               do k = 1, qm_main_r%numat
                  if (k == cc) cycle
                  xyz_k(1:3) = qm_main_r%qm_coord(1:3,k)
                  x          = distance(xyz_c,xyz_k)  ! dist_ij_c(k)  ! distance(xyz_c,xyz_k)
                  xd         = cvalence_contribution_d(cc,k,x)
                  if (xd /= zero) then
                     if (cv_cc > three) then
                        xd = -one * xd
                        xd = f_o1 * f_o2 * xd
                        r_tmp1 = -xd/x
                        do kk=1,3
                           dxyz_local(kk) = (xyz_c(kk) - xyz_k(kk))*r_tmp1*slope*mult_coo
                           dxyz_i(kk)     = dxyz_i(kk)     - dxyz_local(kk)
                           qm_main_r%qm_grads(kk,k) = qm_main_r%qm_grads(kk,k) + dxyz_local(kk)
                        end do
                     end if
                  end if
               end do
               qm_main_r%qm_grads(1:3,cc) = qm_main_r%qm_grads(1:3,cc) + dxyz_i(1:3)
            end if  ! COO- scaling

         end do loophi       ! iterate over hydrogens within the cutoff
      end do loopij

      ! ---------------------------------------------! H-H repulsion calculation
      ! Iterate over H atoms twice
      r_tmp1 = one/hh_rep_r0
      loophh: do icnt = 1, hh_indx_size
         i          = hh_indx(1,icnt)
         j          = hh_indx(2,icnt)
         xyz_i(1:3) = qm_main_r%qm_coord(1:3,i)
         xyz_j(1:3) = qm_main_r%qm_coord(1:3,j)
         r          = distance(xyz_i,xyz_j)
         r_tmp2     = exp(-hh_rep_e*(r*r_tmp1-one))   ! one/hh_rep_r0
         e_corr_sum_hh = e_corr_sum_hh + hh_rep_k*(one-one/(one+r_tmp2))

         if (q_gradient) then
            ! gradient in the internal coordinates
            d_rad = (hh_rep_e*r_tmp1*hh_rep_k*r_tmp2/((one + r_tmp2)**2))/r

            ! Cartesian components of the gradient
            dxyz_local(1:3) = (xyz_i(1:3)-xyz_j(1:3))*d_rad

            ! Add pair contribution to the global gradient
            qm_main_r%qm_grads(1:3,i) = qm_main_r%qm_grads(1:3,i) - dxyz_local(1:3)
            qm_main_r%qm_grads(1:3,j) = qm_main_r%qm_grads(1:3,j) + dxyz_local(1:3)
         end if ! gradient calaculation if requested
      end do loophh

      ! return energy
      EH4_corr = e_corr_sum + e_corr_sum_hh
#if KEY_PARALLEL==1
     if(numnod>1) call gcomb(EH4_corr,1)
#endif

      ! gradients
      do i= 1, qm_main_r%numat
         n=qm_control_r%qminb(i)
         dx(n)=dx(n)+qm_main_r%qm_grads(1,i)
         dy(n)=dy(n)+qm_main_r%qm_grads(2,i)
         dz(n)=dz(n)+qm_main_r%qm_grads(3,i)
         ! debug
         !write(6,'(I4,3F15.8)') i,qm_main_r%qm_grads(1,i),qm_main_r%qm_grads(2,i),qm_main_r%qm_grads(3,i)
      end do

      return
   end subroutine h4_correction

   real(chm_real) function E_Radial_FN(rx,grad_radi,q_grad)
      ! compute radial energy and gradient
      use number, only : zero
      implicit none
      logical        :: q_grad
      real(chm_real) :: rx,grad_radi
      real(chm_real) :: e_local,grad_local,rx2,rx3,rx6

      rx2 = rx *rx
      rx3 = rx *rx2
      rx6 = rx3*rx3
      e_local = -0.00303407407407313510d0 * rx  * rx6 &  ! rda ** 7
               + 0.07357629629627092382d0 * rx6       &  ! rda ** 6
               - 0.70087111111082800452d0 * rx2 * rx3 &  ! rda ** 5
               + 3.25309629629461749545d0 * rx2 * rx2 &  ! rda ** 4
               - 7.20687407406838786983d0 * rx3       &  ! rda ** 3
               + 5.31754666665572184314d0 * rx2       &  ! rda ** 2 
               + 3.40736000001102778967d0 * rx        &  ! rda
               - 4.68512000000450434811d0

      if(q_grad) then
         grad_radi = -0.02123851851851194655d0 * rx6        & ! rda ** 6
                    + 0.44145777777762551519d0 * rx2 * rx3  & ! rda ** 5
                    - 3.50435555555413991158d0 * rx2 * rx2  & ! rda ** 4
                    +13.01238518517846998179d0 * rx3        & ! rda ** 3
                    -21.62062222220516360949d0 * rx2        & ! rda ** 2
                    +10.63509333331144368628d0 * rx         & ! rda
                    + 3.40736000001102778967d0
      else
         grad_radi = zero
      end if

      E_Radial_FN = e_local
      return
   end function E_Radial_FN

   real(chm_real) function E_Angular_FN(rx,grad_angl,q_grad)
      ! polynomial switching function for angle
      use number, only : zero,one,two
      implicit none
      logical :: q_grad
      real(chm_real) :: rx,grad_angl
      real(chm_real) :: rx2,rx3,x,xd

      rx2 = rx *rx
      rx3 = rx *rx2
      x   =(-20.0d0*rx3 + 70.0d0*rx2 - 84.0d0*rx + 35.0d0)*rx2*rx2

      if(q_grad) then
         xd        =(-140.0d0*rx3 + 420.0d0*rx2 - 420.0d0*rx + 140.0d0)*rx3*r_HALFPI
         grad_angl =- two * xd * x           
      else
         grad_angl = zero
      end if

      E_Angular_FN = one - x*x
      return
   end function E_Angular_FN

   real(chm_real) function BOND_switch_FN(rx,grad_radi,q_grad)
      ! bond switching energy and grad
      use number, only : zero
      implicit none
      logical        :: q_grad
      real(chm_real) :: rx,grad_radi
      real(chm_real) :: e_local,rx2,rx3,rx4,rx6

      rx2 = rx *rx
      rx3 = rx *rx2
      e_local = 1.0d0 + (20.0d0*rx3 - 70.0d0*rx2 + 84.0d0*rx - 35.0d0)*rx2*rx2

      if(q_grad) then
         grad_radi = (140.0d0*rx3 - 420.0d0*rx2 + 420.0d0*rx - 140.0d0)*rx3
      else
         grad_radi = zero
      end if

      BOND_switch_FN = e_local
      return
   end function BOND_switch_FN

   ! --------------------------------------
   ! Continuous valence contribution of a pair of atoms
   real(chm_real) function cvalence_contribution(c,d,r)
      use qm1_info, only : qm_main_r
      use number,   only : one,zero
      implicit none
      integer, intent(in) :: c, d  ! c=donor(d_i), d=hydrogen/other(k)
      integer             :: i
      real(chm_real)      :: r, ri, rj, r0, r1, x,x2,x3

      ! identify the donor and hydrogen and fetch the radius
      if(qm_main_r%nat(c) == 1 .or. qm_main_r%nat(c) == 6 .or. qm_main_r%nat(c) == 7 .or. qm_main_r%nat(c) == 8) then
         ri = radius(qm_main_r%nat(c)) ! donor
      else
         ri = zero
      end if
      if(qm_main_r%nat(d) == 1 .or. qm_main_r%nat(d) == 6 .or. qm_main_r%nat(d) == 7 .or. qm_main_r%nat(d) == 8) then
         rj = radius(qm_main_r%nat(d)) ! h or other atoms.
      else
         rj = zero
      end if

      r0 = ri + rj
      r1 = r0 * 1.6d0
      if (r > r0 .and. r < r1) then
          x = (r - r0) / (r1 - r0)
          x2= x*x
          x3= x2*x
          cvalence_contribution = one + (20.0d0*x3 - 70.0d0*x2 + 84.0d0*x - 35.0d0)*x2*x2
      else if(r <= r0 .and. r > zero) then
          cvalence_contribution = one
      else
          cvalence_contribution = zero
      end if
      return
   end function cvalence_contribution

   ! --------------------------------------
   ! Continuous valence contribution of a pair of atoms - derivative
   ! in the internal coordinates
   real(chm_real) function cvalence_contribution_d(c,d,r)
      use qm1_info, only : qm_main_r
      use number,  only : one,zero
      implicit none
      integer, intent(in) :: c, d  ! c=donor(d_i), d=hydrogen/other(k)
      integer             :: i
      real(chm_real)      :: r, ri, rj, r0, r1, x,x2,x3,r_tmpx

      ! identify the donor and hydrogen and fetch the radius
      if(qm_main_r%nat(c) == 1 .or. qm_main_r%nat(c) == 6 .or. qm_main_r%nat(c) == 7 .or. qm_main_r%nat(c) == 8) then
         ri = radius(qm_main_r%nat(c)) ! donor
      else
         ri = zero
      end if
      if(qm_main_r%nat(d) == 1 .or. qm_main_r%nat(d) == 6 .or. qm_main_r%nat(d) == 7 .or. qm_main_r%nat(d) == 8) then
         rj = radius(qm_main_r%nat(d)) ! h or other atoms.
      else
         rj = zero
      end if

      r0 = ri + rj
      r1 = r0 * 1.6d0
      if (r > r0 .and. r < r1) then
         r_tmpx = one/(r1 - r0)
         x = (r - r0)/(r1 - r0)
         x2= x*x
         x3= x2*x
         cvalence_contribution_d = (140.0d0*x3 - 420.0d0*x2 + 420.0d0*x - 140.0d0)*x3*r_tmpx
      else
         cvalence_contribution_d = zero
      end if
      return
   end function cvalence_contribution_d

   ! --------------------------------------
   ! Distance between two atoms
   real(chm_real) function distance(xyz_1,xyz_2)
      implicit none
      real(chm_real) :: xyz_1(3),xyz_2(3)

      distance = sqrt( (xyz_1(1)-xyz_2(1))**2 + (xyz_1(2)-xyz_2(2))**2 + (xyz_1(3)-xyz_2(3))**2 )
      return
   end function distance

   ! --------------------------------------
   ! Angle between three atoms a, b, c ???? b has to be the hydrogen ????
   real(chm_real) function atomangle(xyz_1,xyz_h1,xyz_2)
      use number, only : one
      implicit none
      real(chm_real) :: xyz_1(3),xyz_2(3),xyz_h1(3)
      real(chm_real) :: abs1, abs2, dot, cs ! atomangle

      ! two vectors
      real(chm_real) :: ux, uy, uz, vx, vy, vz

      ux = xyz_1(1) - xyz_h1(1) ! geom%xyz(1,a) - geom%xyz(1,b)
      uy = xyz_1(2) - xyz_h1(2) ! geom%xyz(2,a) - geom%xyz(2,b)
      uz = xyz_1(3) - xyz_h1(3) ! geom%xyz(3,a) - geom%xyz(3,b)
      vx = xyz_2(1) - xyz_h1(1) ! geom%xyz(1,c) - geom%xyz(1,b)
      vy = xyz_2(2) - xyz_h1(2) ! geom%xyz(2,c) - geom%xyz(2,b)
      vz = xyz_2(3) - xyz_h1(3) ! geom%xyz(3,c) - geom%xyz(3,b)

      abs1 = sqrt(ux*ux + uy*uy + uz*uz)  ! vector u is the donor/acceptor-H distance
      abs2 = sqrt(vx*vx + vy*vy + vz*vz)  ! vector v is the acceptor/donor-H distance

      dot = ux*vx + uy*vy + uz*vz
      cs = dot/(abs1*abs2)

      ! Numerical issue: can be close to 1 but out of the valid interval for the Gauchy-Schwarz inequality
      if (cs < -one) then
          cs = -one
      end if
      if (cs > one) then
          cs = one
      end if

      atomangle = acos(cs)
      !
      return
   end function atomangle
!
#endif /* (mndo97)*/
end module H4_mndo
