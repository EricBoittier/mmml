! nbond setup routines
module mndnbnd_module
  use chm_kinds
  use number

  contains 
#if KEY_MNDO97==1 /*mndo97*/
  SUBROUTINE CH2MND(NUMQM,IGMSEL,X,Y,Z,LFIRST)
  !-----------------------------------------------------------------------
  !     Setup QM/MM-non-bonded part for MNDO97
  !
  use chm_kinds
  use number
  use dimens_fcm
  use exfunc
  use memory

  use bases_fcm
  use datstr
  use contrl
  use inbnd
  use nbndqm_mod
  use psf
  !
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r
  !
  !     Adjust nonbonded group list for IMAGES.
  use image
  !     Adjust nonbonded group list for simple pbc.
#if KEY_PBOUND==1
  use pbound
#endif

  implicit none

  INTEGER :: NUMQM,IGMSEL(*)
  real(chm_real)::  X(*),Y(*),Z(*)
  LOGICAL :: LFIRST,Qimage

  integer :: i,j,is,iq,nlat,ngfnd,num_qm_grp
  logical,save :: q_first=.true.

  !-----------------------------------------------------------------------
  Qimage =.false.
  if(.not.useddt_nbond(bnbnd)) call wrndie(-3,'<CH2MND>','Nonbond data structure is not defined.')
#if KEY_PBOUND==1
  if(.not.qBoun) then 
#endif
     if(ntrans.gt.0) then
        !if(lgroup) then
           Qimage =.true.
        !   if(.not.useddt_image(bimag)) call wrndie(-3,'<CH2MND>','Image nonbond data structure is not defined.')
        !else
        !   call wrndie(-2,'<CH2MND>','QM/MM do not interact with Images under Atom Based Cutoff.')
        !end if
     end if
#if KEY_PBOUND==1
  end if
#endif

  ! Prepare the list for QM/MM-one-electron interaction list
  !     note xyzg(3,ngrp), this is different for original mndo97/quantum
  if(mm_main_r%q_cut_by_group .or. mm_main_r%q_switch) then
     if(mm_main_r%q_cut_by_group) then
        if(associated(mm_main_r%q_mmgrp_qmgrp_cut))   deallocate(mm_main_r%q_mmgrp_qmgrp_cut)
        if(associated(mm_main_r%r2_mmgrp_qmgrp_dist)) deallocate(mm_main_r%r2_mmgrp_qmgrp_dist)
        allocate(mm_main_r%q_mmgrp_qmgrp_cut(num_mm_group,num_qm_group))
        allocate(mm_main_r%r2_mmgrp_qmgrp_dist(num_mm_group,num_qm_group))
     end if
     !
     if(associated(mm_main_r%r_num_atom_qm_grp))   deallocate(mm_main_r%r_num_atom_qm_grp)
     if(associated(mm_main_r%r_num_atom_mm_grp))   deallocate(mm_main_r%r_num_atom_mm_grp)
     allocate(mm_main_r%r_num_atom_qm_grp(num_qm_group))
     allocate(mm_main_r%r_num_atom_mm_grp(num_mm_group))

     !
     if(mm_main_r%q_switch) then
        if(associated(mm_main_r%q_mmgrp_qmgrp_swt)) deallocate(mm_main_r%q_mmgrp_qmgrp_swt)
        if(associated(mm_main_r%q_mmgrp_point_swt)) deallocate(mm_main_r%q_mmgrp_point_swt)
        allocate(mm_main_r%q_mmgrp_qmgrp_swt(num_mm_group,num_qm_group))
        allocate(mm_main_r%q_mmgrp_point_swt(num_mm_group))
     end if
  endif
  if(ngrp_old.ne.ngrp) then
     if(allocated(xyzg_sq)) call chmdealloc('mndo97_nbnd.src','CH2MND','xyzg_sq',3,size(xyzg_sq,2),crl=xyzg_sq)
     if(allocated(igrpg))   call chmdealloc('mndo97_nbnd.src','CH2MND','igrpg',size(igrpg),intg=igrpg)
     if(allocated(igrpg2))  call chmdealloc('mndo97_nbnd.src','CH2MND','igrpg2',size(igrpg2),intg=igrpg2)
     if(allocated(qgrpmm))  call chmdealloc('mndo97_nbnd.src','CH2MND','qgrpmm',size(qgrpmm),log=qgrpmm)
     ngrp_old=ngrp
  end if
  if(.not.allocated(xyzg_sq)) call chmalloc('mndo97_nbnd.src','CH2MND','xyzg_sq',3,ngrp,crl=xyzg_sq)
  if(.not.allocated(igrpg))   call chmalloc('mndo97_nbnd.src','CH2MND','igrpg',ngrp,intg=igrpg)
  if(.not.allocated(igrpg2))  call chmalloc('mndo97_nbnd.src','CH2MND','igrpg2',ngrp,intg=igrpg2)
  if(.not.allocated(qgrpmm))  call chmalloc('mndo97_nbnd.src','CH2MND','qgrpmm',ngrp,log=qgrpmm)

  if(q_first) then
     ! find number of qm groups and allocate memory.
     ngfnd  = 1
     do i = 1, ngrp
        is = igpbs(i) + 1
        iq = igpbs(i+1)
        nlat = 0
        do j = is, iq
           if(igmsel(j).eq.1.or.igmsel(j).eq.2) nlat = nlat + 1
        end do
        if(nlat.gt.0) ngfnd = ngfnd + 1
     end do
     qm_main_r%nmaxgrp = ngfnd-1  ! number of qm groups
     if(associated(qm_main_r%nqmgrp)) deallocate(qm_main_r%nqmgrp)
     allocate(qm_main_r%nqmgrp(qm_main_r%nmaxgrp+1))

     q_first=.false.
  end if

  ! Calculate the group centres of geometry.
  call grpcen_g(ngrp,igpbs,x,y,z,xyzg_sq)
  
  ! Fill the QM group array.
  call grpmm(ngrp,natom,igpbs,numqm,num_qm_grp,igmsel,qgrpmm)

  ! Fill MMINB array and checkup for non-list list
  if(mm_main_r%q_cut_by_group) then
     call ch2mnd2_by_group(x,y,z,xyzg_sq,igmsel,iqmgpe,jqmgpe,igrpg,qgrpmm,LFIRST,             &
                          qm_main_r%numat,qm_control_r%qminb,qm_control_r%mminb1,qm_control_r%mminb2, &
                          qm_main_r%nmaxgrp,qm_main_r%nqmgrp,mm_main_r%LQMEWD,mm_main_r%EWMODE,       &
                          mm_main_r%NUMATM)
  else
     call ch2mnd2(x,y,z,xyzg_sq,igmsel,iqmgpe,jqmgpe,igrpg,igrpg2,qgrpmm,LFIRST,              &
                  qm_main_r%numat,qm_control_r%qminb,qm_control_r%mminb1,qm_control_r%mminb2, &
                  qm_main_r%nmaxgrp,qm_main_r%nqmgrp,mm_main_r%LQMEWD,mm_main_r%EWMODE,       &
                  mm_main_r%NUMATM)
  end if

  !-----------------------------------------------------------------------
  return
  END subroutine CH2MND

  SUBROUTINE CH2MND2(X,Y,Z,XYZG,IGMSEL,INBX,JNBX,IGRPG,IGRPG2,QGRPMM,LFIRST, &
                     numat,qminb,mminb1,mminb2,nmaxgrp,nqmgrp,               &
                     LQMEWD,EWMODE,NUMATM)
  !-----------------------------------------------------------------------
  !     Do the actual work of CH2MND:
  !        Define CHARMM atoms as point charges and copy for MNDO97
  !        interface. Currently, only support Group based non-bond
  !        cutoff
  !
  !     Author: Kwangho Nam (2005)
  !
  !     For atom-based cutoffs:
  !        All atoms which are not quantum contribute to electrostatic
  !        interaction in the QM part. See MJF reference:
  !        J. Comp. Chem., Vol. 11, No. 6, 700-733 (1990)
  !
  !     For group-based cutoffs:
  !        Use different scheme, See Nam et al..
  !        J. Chem. Theor. Comput., Vol. 1, No. 1, 2-13 (2005)
  !
  !        1) Changes to support Group based Non-bond cutoff methods
  !           with multiple QM groups.
  !        2) Image atoms are included into QM/MM interaction when
  !           group based non-bond cutoff methods are used.
  !
  !
  use chm_kinds
  use number
  use dimens_fcm
  use exfunc
  use consta
  !
  use inbnd
  !use mndo97
  use psf
  use stream

#if KEY_PBOUND==1
  use pbound
#endif
#if KEY_PARALLEL==1
  use parallel
#endif

  use qm1_info, only : mm_main_r !, qm_control_r,qm_main_r
  use nbndqm_mod, only: map_qmatom_to_group,map_mmatom_to_group,map_mmgrp_to_group,map_allatm_to_group, &
                        num_mm_group,num_mmatm_in_list,num_qm_group

  implicit none

  real(chm_real) :: X(*),Y(*),Z(*),XYZG(3,NGRP)
  INTEGER        :: IGMSEL(*),INBX(*),JNBX(*),IGRPG(*),IGRPG2(*)
  LOGICAL        :: QGRPMM(*),LFIRST,LQMEWD
  integer        :: numat,qminb(*),mminb1(*),mminb2(*),nmaxgrp,nqmgrp(*), &
                    EWMODE,NUMATM
  !
  real(chm_real) :: C2OFNB,C2ONNB,DELR(3),SCENT,xyz_qm(3),xyz_mm(3),scent_min
  real(chm_real) :: rij2_pair,rul3,rul12,rijl,riju
  INTEGER :: I,J,IS,IQ,JS,JQ,IRS,JRS,NB,ITEMP,NLAT,iirs,irs_qm
  INTEGER :: NFLAGS,MM,NGFND,IGRP,JRSPR,NPR,icnt_atm,icnt_grp,irs_old,icnt_swt,iqm,imm,icnt_swt_grp
  LOGICAL :: QPBCHECK,QCHECK,qmswitch_local
  integer :: mstart,mstop
  integer :: ISTRT_CHECK                 ! for external function
  real(chm_real),pointer :: dxyz_pair(:,:)=>Null(),swij_pair(:)=>Null(),dswij_pair(:)=>Null()
  !
  !
  !
  C2OFNB=CTOFNB*CTOFNB
  C2ONNB=CTONNB*CTONNB
  !
  qpbcheck=.false.
#if KEY_PBOUND==1
  if (qBoun) qpbcheck=.true.
#endif

  if(LFIRST) then
     ! first for QM atoms
     do i = 1,numat
        mm        = iabs(qminb(i)) 
        mminb1(i) = mm
        mminb2(MM)= i
     end do

     nqmgrp(1:nmaxgrp)     = 0
     !
     ! Initialize the MMINB flag for the selection of MM atoms
     nflags = -1

     itemp = 0
     do i = 1, natom
        if(igmsel(i)==5 .or. igmsel(i)==0) then
           itemp     = itemp + 1
           mm        = numat + itemp
           mminb1(mm)= i * nflags
           mminb2(i) = mm
        end if
     end do

     ! Let's find QM group number
     ngfnd  = 1
     do i = 1, ngrp
        is = igpbs(i) + 1
        iq = igpbs(i+1)
        nlat = 0
        do j = is, iq
           if(igmsel(j)==1 .or. igmsel(j)==2) nlat = nlat + 1
        end do
        if(nlat > 0) then
           if(ngfnd > nmaxgrp) call wrndie(-5,'<CH2MND2>','Too Many QM groups. Code error. Fix it.')
           ngfnd = ngfnd + 1
           nqmgrp(ngfnd) = i
        end if
     end do
     nqmgrp(1) = ngfnd - 1
  end if
  !
  ! Prepare to pass the charges and coordinates for MM atoms
  ! Reinitialize MMINB array for rechecking
  nflags = -1
  mminb1(numat+1:natom)= IABS(mminb1(numat+1:NATOM))*nflags

  ! QGRPMM array,    if QM group: QGRPMM(I)=.FALSE.
  !                     MM group: QGRPMM(I)=.TRUE. (This includes mixed qm/mm group.)
  !
  ! It only uses Group based cutoff for QM-MM interactions.
  ! First check of MM atoms in the same group with QM atoms
  nlat  = 0
  if(mm_main_r%q_switch) then
     ! mapping between atom list and group list.
     map_qmatom_to_group(1:numat)             = -1  ! each qm atom to each qm group in NQMGRP array.
     map_mmatom_to_group(1:num_mmatm_in_list) = -1  ! see below.. where the array is filled in.

     ! It only uses Group based cutoff for QM-MM interactions.
     ! First check of MM atoms in the same group with QM atoms
     do irs = 1, NQMGRP(1)
        igrp = nqmgrp(irs+1)
        is   = igpbs(igrp)+1
        iq   = igpbs(igrp+1)
        mm_main_r%r_num_atom_qm_grp(irs) = one/float(iq - is + 1)
        do i = is, iq
           ! On the MMINB array when any MM atoms that need to be included in QM/MM
           if(igmsel(i) == 0) then
              ! mm atom in the mixed qm/mm group
              imm = mminb2(i)
              mminb1(imm)= iabs(mminb1(imm))
              nlat       = nlat + 1
           else if(igmsel(i)==1 .or. igmsel(i)==2) then
              ! qm atom
              iqm = mminb2(i)  ! iqm within 1-numat
              map_qmatom_to_group(iqm) = irs     ! iqm atom belongs irs-th qm group.
           end if
        end do
     end do

     ! check and complete necessary arrays.
     !
     ! map_qmatom_to_group(1:numat)             -> 1,num_qm_group
     do i=1,numat
        if(map_qmatom_to_group(i) < 0) &
        call wrndie(-5,'<CH2MND2>','Ill-defined map_qmatom_to_group array for qm atoms.')
     end do
  else
     do irs = 1, NQMGRP(1)
        igrp = nqmgrp(irs+1)
        is   = igpbs(igrp)+1
        iq   = igpbs(igrp+1)
        do i = is, iq
           ! On the MMINB array when any MM atoms that need to be included in QM/MM
           if(igmsel(i) == 0) then
              mminb1(mminb2(i))= iabs(mminb1(mminb2(i)))
              nlat      = nlat + 1
           end if
        end do
     end do
  end if

  ! Initialize IGRPG array
  igrpg(1:ngrp)  = 0
  igrpg2(1:ngrp) = 0

  ! Always IRS is for QM group and JRS is for MM group
  nb    = 0
  itemp = 0
  icnt_swt_grp = 0
#if KEY_PBOUND==1
  if(qpbcheck) then
     do irs=1,ngrp
        npr  = inbx(irs) - itemp
        itemp= inbx(irs)

        ! Loop over the MM groups: see Note below.
        if(npr > 0) then
           ! down-graded: not separate between nodes.
           !              mstart=ISTRT_CHECK(mstop,NPR)       ! get nstart abd mstop
           !!xyz_qm(1:3)= xyzg(1:3,irs)
           !!do jrspr=1,npr
           !!   jrs       = iabs(jnbx(nb+jrspr))
           !!   delr(1:3) = xyz_qm(1:3)-xyzg(1:3,jrs)
           !!   ! swapxyz_image is only done with image setup. So, should check this here.
           !!   call pbcheck(delr)
           !!
           !!   scent=delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
           !!   if(scent <= c2ofnb) then
           !!      igrpg(jrs) = igrpg(jrs) + 1
           !!      if(scent > c2onnb) then
           !!         igrpg2(jrs)  = irs
           !!         icnt_swt_grp = icnt_swt_grp + 1
           !!      end if
           !!   end if
           !!end do

           !
           do jrspr=1,npr
#if KEY_PARALLEL==1
              if(mynod /= mod(jrspr,numnod)) cycle
#endif
              jrs        = iabs(jnbx(nb+jrspr))
              xyz_mm(1:3)= xyzg(1:3,jrs)
              scent_min = 9999.0d0  !
              ! find minimum distance to jrs mm group.
              do i=1,NQMGRP(1)
                 iirs=nqmgrp(i+1)   ! qm group index.
                 delr(1:3) = xyzg(1:3,iirs) - xyz_mm(1:3)
                 ! swapxyz_image is only done with image setup. So, should check this here.
                 call pbcheck(delr)

                 scent     = delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
                 if(scent < scent_min) then
                    scent_min = scent
                    irs_qm    = iirs
                 end if
              end do

              ! now, the shortest distance is scent_min
              if(scent_min <= c2ofnb) then
                 igrpg(jrs) = igrpg(jrs) + 1
                 if(scent_min > c2onnb) then
                    igrpg2(jrs) = irs_qm
                    icnt_swt_grp = icnt_swt_grp + 1
                 end if
              end if
           end do
           nb = nb + npr
        end if
     end do
  else
#endif
     do irs=1,ngrp
        npr  = inbx(irs) - itemp
        itemp= inbx(irs)

        ! Loop over the MM groups:
        !
        ! Note about the list: 
        !      mm groups that are in the inbx list is closest to irs qm group in the non-bond list update.
        !      However, since it is possible that some mm group gets closer to other qm group between list
        !      update (i.e., during md), we should check over all qm groups to find if that mm group is
        !      in the cutoff.
        !
        if(npr > 0) then
           ! down-graded: not separate between nodes.
           !              mstart=ISTRT_CHECK(mstop,NPR)       ! get nstart abd mstop
           !xyz_qm(1:3)= xyzg(1:3,irs)
           !do jrspr=1,npr
           !   jrs       = iabs(jnbx(nb+jrspr))
           !   delr(1:3) = xyz_qm(1:3)-xyzg(1:3,jrs)
           !   scent     = delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
           !   if(scent <= c2ofnb) then
           !      igrpg(jrs) = igrpg(jrs) + 1
           !      if(scent > c2onnb) then
           !         igrpg2(jrs) = irs
           !         icnt_swt_grp = icnt_swt_grp + 1
           !      end if
           !   end if
           !end do

           !
           do jrspr=1,npr
#if KEY_PARALLEL==1
              if(mynod /= mod(jrspr,numnod)) cycle
#endif
              jrs        = iabs(jnbx(nb+jrspr))
              xyz_mm(1:3)= xyzg(1:3,jrs)
              scent_min = 9999.0d0  !
              ! find minimum distance to jrs mm group.
              do i=1,NQMGRP(1)
                 iirs=nqmgrp(i+1)   ! qm group index.
                 delr(1:3) = xyzg(1:3,iirs) - xyz_mm(1:3)
                 scent     = delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
                 if(scent < scent_min) then
                    scent_min = scent
                    irs_qm    = iirs
                 end if
              end do

              ! now, the shortest distance is scent_min, irs_qm qm group.
              if(scent_min <= c2ofnb) then
                 igrpg(jrs) = igrpg(jrs) + 1
                 if(scent_min > c2onnb) then
                    igrpg2(jrs)  = irs_qm
                    icnt_swt_grp = icnt_swt_grp + 1
                 end if
              end if
           end do
           nb = nb + npr
        end if
     end do
#if KEY_PBOUND==1
  end if
#endif

#if KEY_PARALLEL==1
  if(numnod>1) then
     call igcomb(igrpg,ngrp)
     call igcomb(igrpg2,ngrp)
     call igcomb(icnt_swt_grp,1)
  end if
#endif
  !
  nflags=1
  if(LQMEWD .and. EWMODE == 2) nflags=-1  ! this may not be used at all in the present implementation.
  do irs=1,ngrp
     if(igrpg(irs) >= 1) then
        do i = igpbs(irs)+1,igpbs(irs+1)
           if(igmsel(i) == 0) then
              mm   = mminb2(i) 
              if(mminb1(mm) < 0) then
                 mminb1(mm)=nflags*iabs(mminb1(mm)) 
                 nlat      = nlat + 1
              end if
           end if
        end do
     end if
  end do
  !
  NUMATM = nlat

  if(mm_main_r%q_switch) then
     ! check and complete necessary arrays.
     ! after this:
     !
     ! map_qmatom_to_group(1:numat)             -> 1,num_qm_group
     ! map_mmatom_to_group(1:num_mmatm_in_list) -> 1,num_mm_group
     ! map_mmgrp_to_group(1:num_mm_group)       -> 1,ngrp
     !
     icnt_atm = 0
     icnt_grp = 0
     irs_old  = -1
     do i = numat+1,natom
        imm = mminb1(i)             ! position in the main psf array.
        if(imm > 0) then
           ! included in the qm-mm interactions.
           icnt_atm = icnt_atm + 1
           irs = map_allatm_to_group(imm)   ! the group index in the main group array.
           if(irs /= irs_old) then
              ! new group
              icnt_grp = icnt_grp + 1
              irs_old  = irs
              map_mmgrp_to_group(icnt_grp) = irs
              mm_main_r%r_num_atom_mm_grp(icnt_grp) = one/float(igpbs(irs+1)-igpbs(irs))
           end if
           map_mmatom_to_group(icnt_atm) = icnt_grp ! group index in the qm-mm int. list.
        end if
     end do

     ! icnt_swt_grp is the number of mm groups in the switching region.

     rul3 = one/((C2OFNB - C2ONNB)**3)
     rul12= twelve * rul3

     mm_main_r%inum_mm_grp = icnt_grp  ! total number of mm groups within the cutoff dist.
     if(associated(dxyz_pair))  deallocate(dxyz_pair)
     if(associated(swij_pair))  deallocate(swij_pair)
     if(associated(dswij_pair)) deallocate(dswij_pair)
     allocate(dxyz_pair(3,ngrp))
     allocate(swij_pair(ngrp))
     allocate(dswij_pair(ngrp))
     icnt_swt = 0
     do j=1,icnt_grp
        jrs= map_mmgrp_to_group(j)  ! mm group
        if(igrpg2(jrs) > 0) then
           ! jrs mm group is within the switching region for qm group igrpg2(jrs).
           irs       = igrpg2(jrs)  ! qm group
           icnt_swt  = icnt_swt + 1
           delr(1:3) = xyzg(1:3,irs) - xyzg(1:3,jrs)  ! qm - mm dist.
           scent     = delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
           !
           ! if this mm group is in the switching region, it applies to all qm group based on
           ! the cutoff information of irs qm group.
           ! one qm-mm is in the switching region, all qm groups are in the switching region.

           if(qgrpmm(irs)) then
              ! mixed qm/mm group..
              ! find qm atom in the irs group (could not be the case as some of them are mixed qm/mm groups.
              do i=igpbs(irs) + 1,igpbs(irs+1)
                 if(mminb2(i) <= numat) then
                    is = i
                    exit
                 end if
              end do
           else
              is        = igpbs(irs) + 1
           end if

           irs       = map_qmatom_to_group(mminb2(is)) ! back to qm group in NQMGRP(1)
           mm_main_r%q_mmgrp_point_swt(j) = irs                  ! point qm group in NQMGRP(1)
           do i=1,NQMGRP(1)
              ! point array of the computed info.
              mm_main_r%q_mmgrp_qmgrp_swt(j,i) = icnt_swt + icnt_swt_grp*(i-1)
           end do
           dxyz_pair(1:3,j) = delr(1:3)
           rij2_pair        = scent   ! rij^2 value.

           ! now, determine switching function value.
           rijl = c2onnb - rij2_pair
           riju = c2ofnb - rij2_pair
           swij_pair(j) = riju*riju*(riju-three*rijl)*rul3   ! switching function value.
           dswij_pair(j)= rijl*riju*rul12                    ! dSw/drij value.
        else
           mm_main_r%q_mmgrp_qmgrp_swt(j,1:NQMGRP(1)) =-1
        end if
     end do

     ! now, setup switching function part.
     icnt_swt_grp              = icnt_swt 
     icnt_swt                  = icnt_swt*NQMGRP(1)
     mm_main_r%isize_swt_array = icnt_swt
     if(associated(mm_main_r%sw_val)) then
        if(size(mm_main_r%sw_val) < icnt_swt) then
           deallocate(mm_main_r%sw_val)
           deallocate(mm_main_r%dxyz_sw)
           deallocate(mm_main_r%dxyz_sw2)
           deallocate(mm_main_r%dsw_val)
           deallocate(mm_main_r%q_backmap_dxyz_sw)
        end if
     end if
     if(.not.associated(mm_main_r%sw_val))  allocate(mm_main_r%sw_val(icnt_swt))
     if(.not.associated(mm_main_r%dxyz_sw)) allocate(mm_main_r%dxyz_sw(6,icnt_swt))
     if(.not.associated(mm_main_r%dxyz_sw2)) allocate(mm_main_r%dxyz_sw2(6,icnt_swt))
     if(.not.associated(mm_main_r%dsw_val)) allocate(mm_main_r%dsw_val(3,icnt_swt))
     if(.not.associated(mm_main_r%q_backmap_dxyz_sw)) allocate(mm_main_r%q_backmap_dxyz_sw(icnt_swt))
     call qmmm_nbnd_switch_all_qm(NQMGRP(1),mm_main_r%inum_mm_grp,icnt_swt_grp)

     deallocate(dxyz_pair)
     deallocate(swij_pair)
     deallocate(dswij_pair)
  end if

  return
    !
    contains

    subroutine qmmm_nbnd_switch_all_qm(icnt_qmg,icnt_mmg,icnt_swt_local)
    !
    ! This routine computes, Sw(rij) and gradient of Sw(rij) values.
    !
    ! r_ij = r_j - r_i, where r_j=sum_k (r_j,k) (k=1,Natom_i, no. atoms in group i), and
    !                         r_i=sum_k (r_i,k) (k=1,Natom_j, no. atoms in group j).
    !
    ! Sw(r_ij) = (r_off^2 - r_ij^2)^2 {(r_off^2 - r_ij^2) - 3*(r_on^2 - r_ij^2)}/{(r_off^2 - r_on^2)^3}.
    ! dSw(r_ij)/dr_ij = (12/{r_off^2-r_on^2)^3})*{(r_off^2-r_ij^2)*(r_on^2-r_ij^2)}
    !
    ! dSw(r_ij)/dx_i,k =-dSw(r_ij)/dr_ij * (x_j - x_i) / N_atom_in_group_i, x gradient of k-th atom in group i.
    ! dSw(r_ij)/dx_j,k = dSw(r_ij)/dr_ij * (x_j - x_i) / N_atom_in_group_j, x gradient of k-th atom in group j.
    !
    use qm1_info, only : qm_control_r
    implicit none
    integer :: icnt_qmg,icnt_mmg,icnt_swt_local,icnt_local,jcnt

    do i=1,icnt_qmg
       jcnt = 0
       do j=1,icnt_mmg
          if(mm_main_r%q_mmgrp_qmgrp_swt(j,i) > 0) then
             ! this is the mm group in the switching region.
             icnt_local = mm_main_r%q_mmgrp_qmgrp_swt(j,i)                      ! i.e., icnt_swt 
             mm_main_r%sw_val(icnt_local)     = swij_pair(j)                    ! switching function value.
             mm_main_r%dsw_val(1:3,icnt_local)= dswij_pair(j)*dxyz_pair(1:3,j)  ! dSw/drij*{xyz(1:3,qm)-xyz(1:3,mm)}

             ! for back-mapping
             jcnt = jcnt + 1
             mm_main_r%q_backmap_dxyz_sw(icnt_local)=jcnt + icnt_swt_local*(mm_main_r%q_mmgrp_point_swt(j)-1)
          end if
       end do
    end do
    return
    end subroutine qmmm_nbnd_switch_all_qm
    !
  END subroutine CH2MND2


  SUBROUTINE CH2MND2_by_group(X,Y,Z,XYZG,IGMSEL,INBX,JNBX,IGRPG,QGRPMM,LFIRST, &
                              numat,qminb,mminb1,mminb2,nmaxgrp,nqmgrp,               &
                              LQMEWD,EWMODE,NUMATM)
  !-----------------------------------------------------------------------
  !     Do the actual work of CH2MND:
  !        Define CHARMM atoms as point charges and copy for MNDO97 interface.
  !        This part is to setup the group-by-group based pair list.
  !
  !     Author: Kwangho Nam (2015)
  !
  ! Note:
  ! Cutoff based on group-based cutoff.
  ! 1. In the non-bond list, differently from that of the default (default) group-based list,
  !    in which any MM group that is within the cutoff distance from any QM group
  !    is included, the list is splitted such that each QM group has different MM list, which
  !    is within the cutoff update distance. (see routine GUQME3_by_group in qmnbnd.src.)
  ! 2. When evaluating qm-mm interactions, for each QM group, any MM group in the list
  !    is exlcuded in the interaction evaluation if that MM group is away more than
  !    the cutoff distance.
  ! 3. If the following switch is on, the interaction between the dist_on and dist_off
  !    is switched off at the dist_off distance. (This is the default with q_cut_by_group=.true.
  ! 4. When LQMEWD is true, the interaction energy between the i (QM) and j (MM) atom pair is
  !
  !    E_ij (r_ij) = Sw(r_ij)*E_qm-mm-model(r_ij) + (1-Sw(r_ij))*E_qm-mm-long-distance-model(r_ij)
  !
  !    where E_qm-mm-model is the regular QM-MM interaction model, and
  !          E_qm-mm-long-distance-model is the QM-Mulliken-charge and MM charge interaction,
  !          which is used in the QM/MM-Ewald and QM/MM-PME long interaction model.
  !
  use chm_kinds
  use number
  use dimens_fcm
  use exfunc
  use consta
  !
  use inbnd
  use psf
  use stream
  use nbndqm_mod, only: map_qmatom_to_group,map_mmatom_to_group,map_mmgrp_to_group,map_allatm_to_group, &
                        num_mm_group,num_mmatm_in_list,num_qm_group
  use qm1_info,only : mm_main_r

#if KEY_PBOUND==1
  use pbound
#endif
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none

  real(chm_real) :: X(*),Y(*),Z(*),XYZG(3,NGRP)
  INTEGER        :: IGMSEL(*),INBX(*),JNBX(*),igrpg(*)
  LOGICAL        :: QGRPMM(*),LFIRST,LQMEWD
  integer        :: numat,qminb(*),mminb1(*),mminb2(*),nmaxgrp,nqmgrp(*), &
                    EWMODE,NUMATM
  !
  real(chm_real) :: C2OFNB,C2ONNB,DELR(3),SCENT,xyz_qm(3),xyz_mm(3),scent_min
  INTEGER :: I,J,IS,IQ,JS,JQ,IRS,JRS,NB,ITEMP,NLAT,iirs,irs_qm
  INTEGER :: NFLAGS,iqm,imm,NGFND,IGRP,JRSPR,NPR,icnt_atm,icnt_grp,irs_old,icnt_swt
  LOGICAL :: qcheck,q_mm_found
  real(chm_real),pointer :: dxyz_pair(:,:,:)=>Null()
  !
  !
  !
  C2OFNB=CTOFNB*CTOFNB
  C2ONNB=CTONNB*CTONNB
  !
  ! used in the actual qm-mm interaction evaluation (cutoff and switching).
  mm_main_r%r_cut_group_dist_off = C2OFNB
  mm_main_r%r_cut_group_dist_on  = C2ONNB

  !
#if KEY_PBOUND==1
  if (qBoun) then
     call wrndie(-5,'<CH2MND2_by_group>','BYGRoup option in qm/mm is not compatible with PBOUND option.')
     return
  end if
#endif

  if(LFIRST) then
     ! first for QM atoms
     do i = 1,numat
        imm        = iabs(qminb(i))
        mminb1(i)  = imm             ! map i=1,numat to i=1,natom
        mminb2(imm)= i               ! map i=1,natom to i=1,numat
     end do

     nqmgrp(1:nmaxgrp)     = 0
     !
     ! Initialize the MMINB flag for the selection of MM atoms
     nflags = -1

     itemp = 0
     do i = 1, natom
        if(igmsel(i)==5 .or. igmsel(i)==0) then
           itemp      = itemp + 1
           imm        = numat + itemp
           mminb1(imm)= i * nflags
           mminb2(i)  = imm         ! map i=1,natom to i=numat+1,natom
        end if
     end do

     ! Let's find QM group number
     ngfnd  = 1
     do i = 1, ngrp
        is = igpbs(i) + 1
        iq = igpbs(i+1)
        nlat = 0
        do j = is, iq
           if(igmsel(j)==1 .or. igmsel(j)==2) nlat = nlat + 1
        end do
        if(nlat > 0) then
           if(ngfnd > nmaxgrp) call wrndie(-5,'<CH2MND2_by_group>','Too Many QM groups. Code error. Fix it.')
           ngfnd = ngfnd + 1
           nqmgrp(ngfnd) = i
        end if
     end do
     nqmgrp(1) = ngfnd - 1
  end if
  !
  ! Prepare to pass the charges and coordinates for MM atoms
  ! Reinitialize MMINB array for rechecking
  nflags = -1
  mminb1(numat+1:natom)= IABS(mminb1(numat+1:NATOM))*nflags

  ! mapping between atom list and group list.
  map_qmatom_to_group(1:numat)             = -1
  map_mmatom_to_group(1:num_mmatm_in_list) = -1

  ! It only uses Group based cutoff for QM-MM interactions.
  ! First check of MM atoms in the same group with QM atoms
  nlat  = 0
  do irs = 1, NQMGRP(1)
     igrp = nqmgrp(irs+1)
     is   = igpbs(igrp)+1
     iq   = igpbs(igrp+1)
     do i = is, iq
        ! On the MMINB array when any MM atoms that need to be included in QM/MM
        if(igmsel(i) == 0) then
           ! mm atom in the mixed qm/mm group
           imm = mminb2(i)
           mminb1(imm)= iabs(mminb1(imm))
           nlat      = nlat + 1
        else if(igmsel(i)==1 .or. igmsel(i)==2) then
           ! qm atom
           iqm = mminb2(i)  ! iqm within 1-numat
           map_qmatom_to_group(iqm) = irs     ! iqm atom belongs irs-th qm group.
        end if
     end do
  end do
  !
  ! Initialize IGRPG array
  igrpg(1:ngrp)  = 0
  !
  !
  ! Always IRS is for QM group and JRS is for MM group
  nb    = 0
  itemp = 0
  do irs=1,ngrp
     npr  = inbx(irs) - itemp
     itemp= inbx(irs)

     ! Loop over the MM groups
     !
     ! Note about the list: 
     !      mm groups that are in the inbx list is closest to irs qm group in the non-bond list update.
     !      However, since it is possible that some mm group gets closer to other qm group between list
     !      update (i.e., during md), we should check over all qm groups to find if that mm group is
     !      in the cutoff.
     !
     if(npr > 0) then
        ! down-graded: not separate between nodes.
        !              mstart=ISTRT_CHECK(mstop,NPR)       ! get nstart abd mstop
        !xyz_qm(1:3)= xyzg(1:3,irs)
        !do jrspr=1,npr
        !   jrs = iabs(jnbx(nb+jrspr))
        !   delr(1:3) = xyz_qm(1:3)-xyzg(1:3,jrs)
        !   scent=delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
        !   if(scent <= c2ofnb) then
        !      igrpg(jrs) = igrpg(jrs) + 1
        !   end if
        !end do

        !
        do jrspr=1,npr
#if KEY_PARALLEL==1
           if(mynod /= mod(jrspr,numnod)) cycle
#endif
           jrs = iabs(jnbx(nb+jrspr))
           xyz_mm(1:3)= xyzg(1:3,jrs)
           scent_min = 9999.0d0  !
           ! if any qm group is within the cutoff distance.
           do i=1,NQMGRP(1)
              iirs=nqmgrp(i+1)         ! qm group index.
              delr(1:3) = xyzg(1:3,iirs) - xyz_mm(1:3)
              scent     = delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
              if(scent <= c2ofnb) then
                 igrpg(jrs) = igrpg(jrs) + 1
                 exit
              end if
           end do
        end do

        nb = nb + npr
     end if
  end do

#if KEY_PARALLEL==1
  if(numnod>1) call igcomb(igrpg,ngrp)
#endif

  !
  nflags=1
  if(LQMEWD .and. EWMODE == 2) nflags=-1  ! this may not be used at all in the present implementation.
  do irs=1,ngrp
     if(igrpg(irs) >= 1) then
        do i = igpbs(irs)+1,igpbs(irs+1)
           if(igmsel(i) == 0) then
              imm   = mminb2(i)
              if(mminb1(imm) < 0) then
                 mminb1(imm)=nflags*iabs(mminb1(imm))
                 nlat      = nlat + 1
              end if
           end if
        end do
     end if
  end do
  !
  NUMATM = nlat

  ! check and complete necessary arrays.
  ! after this:
  !
  ! map_qmatom_to_group(1:numat)             -> 1,num_qm_group
  ! map_mmatom_to_group(1:num_mmatm_in_list) -> 1,num_mm_group
  ! map_mmgrp_to_group(1:num_mm_group)       -> 1,ngrp
  !
  !
  !
  do i=1,numat
     if(map_qmatom_to_group(i) < 0) &
        call wrndie(-5,'<CH2MND2_by_group>','Ill-defined map_qmatom_to_group array for qm atoms.')
  end do
  !
  icnt_atm = 0
  icnt_grp = 0
  irs_old  = -1
  do i = numat+1,natom
     imm = mminb1(i)             ! position in the main psf array.
     if(imm > 0) then
        ! included in the qm-mm interactions.
        icnt_atm = icnt_atm + 1
        irs = map_allatm_to_group(imm)   ! the group index in the main group array.
        if(irs /= irs_old) then
           ! new group
           icnt_grp = icnt_grp + 1
           irs_old  = irs
           map_mmgrp_to_group(icnt_grp) = irs
           mm_main_r%r_num_atom_mm_grp(icnt_grp) = one/float(igpbs(irs+1)-igpbs(irs))
        end if
        map_mmatom_to_group(icnt_atm) = icnt_grp ! group index in the qm-mm int. list.
     end if
  end do

  mm_main_r%inum_mm_grp = icnt_grp  ! total number of mm groups within the cutoff dist.
  if(mm_main_r%q_switch) then
     if(associated(dxyz_pair)) deallocate(dxyz_pair)
     allocate(dxyz_pair(3,icnt_grp,NQMGRP(1)))
     !do i=1,NQMGRP(1)
     !   do j=1,icnt_grp
     !      mm_main_r%r2_mmgrp_qmgrp_dist(j,i) =-one
     !      mm_main_r%q_mmgrp_qmgrp_swt(j,i)   =-1
     !      mm_main_r%q_mmgrp_qmgrp_cut(j,i)   =.false.
     !   end do
     !end do
     icnt_swt = 0
     do i=1,NQMGRP(1)
        irs = nqmgrp(i+1)
        mm_main_r%r_num_atom_qm_grp(i)=one/float(igpbs(irs+1)-igpbs(irs))
        do j=1,icnt_grp
           jrs= map_mmgrp_to_group(j)
           delr(1:3) = xyzg(1:3,irs) - xyzg(1:3,jrs)  ! qm - mm dist.
           scent     = delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
           !mm_main_r%r2_mmgrp_qmgrp_dist(j,i) = scent
           !
           if(scent <= c2ofnb) then
              mm_main_r%r2_mmgrp_qmgrp_dist(j,i) = scent
              mm_main_r%q_mmgrp_qmgrp_cut(j,i)   = .true.
              if(scent >= c2onnb) then
                 icnt_swt = icnt_swt + 1
                 mm_main_r%q_mmgrp_qmgrp_swt(j,i) = icnt_swt  ! point array in the computed info.
                 dxyz_pair(1:3,j,i) = delr(1:3)
              else
                 mm_main_r%q_mmgrp_qmgrp_swt(j,i) =-1
              end if
           else
              mm_main_r%r2_mmgrp_qmgrp_dist(j,i) =-one
              mm_main_r%q_mmgrp_qmgrp_swt(j,i)   =-1
              mm_main_r%q_mmgrp_qmgrp_cut(j,i)   =.false.
           end if
        end do
     end do

     ! now, setup switching function part.
     mm_main_r%isize_swt_array = icnt_swt
     if(associated(mm_main_r%sw_val)) then
        if(size(mm_main_r%sw_val) < icnt_swt) then
           deallocate(mm_main_r%sw_val)
           deallocate(mm_main_r%dxyz_sw)
           deallocate(mm_main_r%dxyz_sw2)
           deallocate(mm_main_r%dsw_val)
        end if
     end if
     if(.not.associated(mm_main_r%sw_val))  allocate(mm_main_r%sw_val(icnt_swt))
     if(.not.associated(mm_main_r%dxyz_sw)) allocate(mm_main_r%dxyz_sw(6,icnt_swt))
     if(.not.associated(mm_main_r%dxyz_sw2)) allocate(mm_main_r%dxyz_sw2(6,icnt_swt))
     if(.not.associated(mm_main_r%dsw_val)) allocate(mm_main_r%dsw_val(3,icnt_swt))
     call qmmm_nbnd_switch(NQMGRP(1),mm_main_r%inum_mm_grp,icnt_swt)
     !
     ! initialize gradient components.
     !mm_main_r%dxyz_sw(1:6,1:icnt_swt) =zero

     deallocate(dxyz_pair)
  else
     !do i=1,NQMGRP(1)
     !   do j=1,icnt_grp
     !      mm_main_r%r2_mmgrp_qmgrp_dist(j,i) =-one
     !      mm_main_r%q_mmgrp_qmgrp_cut(j,i)   =.false.
     !   end do
     !end do
     do i=1,NQMGRP(1)
        irs = nqmgrp(i+1)
        mm_main_r%r_num_atom_qm_grp(i)=one/float(igpbs(irs+1)-igpbs(irs))
        do j=1,icnt_grp
           jrs= map_mmgrp_to_group(j)
           delr(1:3) = xyzg(1:3,irs) - xyzg(1:3,jrs)  ! qm - mm dist.
           scent     = delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
           !mm_main_r%r2_mmgrp_qmgrp_dist(j,i) = scent
           !
           if(scent <= c2ofnb) then
              mm_main_r%r2_mmgrp_qmgrp_dist(j,i) = scent
              mm_main_r%q_mmgrp_qmgrp_cut(j,i) = .true.
           else
              mm_main_r%r2_mmgrp_qmgrp_dist(j,i) =-one
              mm_main_r%q_mmgrp_qmgrp_cut(j,i)   =.false.
           end if
        end do
     end do
  end if
  !
  return
    !
    contains 
    !
    subroutine qmmm_nbnd_switch(icnt_qmg,icnt_mmg,icnt_swt_local)
    !
    ! This routine computes, Sw(rij) and gradient of Sw(rij) values.
    !
    ! r_ij = r_j - r_i, where r_j=sum_k (r_j,k) (k=1,Natom_i, no. atoms in group i), and
    !                         r_i=sum_k (r_i,k) (k=1,Natom_j, no. atoms in group j).
    !
    ! Sw(r_ij) = (r_off^2 - r_ij^2)^2 {(r_off^2 - r_ij^2) - 3*(r_on^2 - r_ij^2)}/{(r_off^2 - r_on^2)^3}.
    ! dSw(r_ij)/dr_ij = (12/{r_off^2-r_on^2)^3})*{(r_off^2-r_ij^2)*(r_on^2-r_ij^2)}
    !
    ! dSw(r_ij)/dx_i,k =-dSw(r_ij)/dr_ij * (x_j - x_i) / N_atom_in_group_i, x gradient of k-th atom in group i.
    ! dSw(r_ij)/dx_j,k = dSw(r_ij)/dr_ij * (x_j - x_i) / N_atom_in_group_j, x gradient of k-th atom in group j.
    !
    use qm1_info, only : qm_control_r
    implicit none
    integer :: icnt_qmg,icnt_mmg,icnt_swt_local,icnt_local,k
    real(chm_real):: rul3,rul12,rijl,riju,funct,dfn,rij2_local,rij2_local_1,rij2_local_2,dtmp

    rul3 = one/((C2OFNB - C2ONNB)**3)
    rul12= twelve * rul3

    do i=1,icnt_qmg
       !
       dtmp = qm_control_r%DEL*mm_main_r%r_num_atom_qm_grp(i)
       do j=1,icnt_mmg
          if(mm_main_r%q_mmgrp_qmgrp_swt(j,i) > 0) then
             ! this is the array in the switching region.
             icnt_local = mm_main_r%q_mmgrp_qmgrp_swt(j,i)
             rij2_local = mm_main_r%r2_mmgrp_qmgrp_dist(j,i)

             rijl  = c2onnb - rij2_local
             riju  = c2ofnb - rij2_local
             mm_main_r%sw_val(icnt_local)     = riju*riju*(riju-three*rijl)*rul3   ! switching function value.
             mm_main_r%dsw_val(1:3,icnt_local)=(rijl*riju*rul12)*dxyz_pair(1:3,j,i)! dSw/drij*{xyz(1:3,qm)-xyz(1:3,mm)}
          end if
       end do
    end do
    return
    end subroutine qmmm_nbnd_switch
    !
  END subroutine CH2MND2_by_group

#endif /*mndo97*/

end module mndnbnd_module

