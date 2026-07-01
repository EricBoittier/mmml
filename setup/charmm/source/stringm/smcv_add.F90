#if (KEY_STRINGM==1) /*  automatically protect all code */
! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! THIS FILE CONTAINS ROUTINES FOR ADDING NEW CV
! THERE IS A SEPARATE ROUTINE FOR EACH CV TYPE
!
#if (KEY_STRINGM==1)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_add(COMLYN,COMLEN)
      use cv_types
      use cv_common
      use cv_quaternion, only: quat_initialized, quat, quat_init
      use sm_config, only: qt_send_displ, qt_send_count, &
     & imap_displ, imap_count
!
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      implicit none
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=*) :: COMLYN
      integer :: COMLEN
! local
      integer i, j, k
      character(len=20) :: keyword
      character(len=len("SMCV_ADD>") ),parameter::whoami="SMCV_ADD>";!macro
!
      keyword=next20(comlyn,comlen) ! directive
! COM positions
      if (( keyword(1:10).eq.'POSI_COM_X'(1:10) )) then
       call smcv_posi_com_add(comlyn, comlen, posi_com_x) ! note: the same routine for all position coordinates
      else if (( keyword(1:10).eq.'POSI_COM_Y'(1:10) )) then
       call smcv_posi_com_add(comlyn, comlen, posi_com_y)
      else if (( keyword(1:10).eq.'POSI_COM_Z'(1:10) )) then
       call smcv_posi_com_add(comlyn, comlen, posi_com_z)
! dihedral_COM
      else if (( keyword(1:8).eq.'DIHE_COM'(1:8) )) then
       call smcv_dihe_com_add(comlyn, comlen)
! angle_COM
      else if (( keyword(1:9).eq.'ANGLE_COM'(1:9) )) then
       call smcv_angle_com_add(comlyn, comlen)
! anglvec
      else if (( keyword(1:7).eq.'ANGLVEC'(1:7) )) then
       call smcv_anglvec_add(comlyn, comlen)
! distance_COM
      else if (( keyword(1:8).eq.'DIST_COM'(1:8) )) then
       call smcv_dist_com_add(comlyn, comlen)
! reference frame (not really a CV, but processed here)
      else if (( keyword(1:5).eq.'FRAME'(1:5) )) then
       call smcv_frame_add(comlyn, comlen)
! orientation quaternion
      else if (( keyword(1:10).eq.'QUATERNION'(1:10) )) then
       call smcv_quaternion_add(comlyn, comlen)
!
! (re)compute quaternion index limits (for parallelization) after each addition
!
       if (SIZE_LOCAL.gt.0) then
        if (.not.quat_initialized) call quat_init() ! make sure frames%num_frames is defined
        j=ceiling(1.0d0*quat%num_quat/SIZE_LOCAL) ! max. number of frames assigned to slave node
        k=ceiling(1.0d0*cv%amap%last/SIZE_LOCAL) ! max. number of amap indices assigned to slave node
!
        do i=1,SIZE_LOCAL
         qt_send_displ(i)=min((i-1)*j,quat%num_quat-1) ! cannot exceed num_cv
         qt_send_count(i)=max(0,min(j,quat%num_quat-j*(i-1))) ! how many CV I will send to CPU i
! atom map partitioning (for parallel computation of M
!
         imap_displ(i)=min((i-1)*k,cv%amap%last-1)
         imap_count(i)=max(0,min(k,cv%amap%last-k*(i-1)))
        enddo
       endif ! SIZE
! rmsd from a target structure
      else if (( keyword(1:4).eq.'RMSD'(1:4) )) then
       call smcv_rmsd_add(comlyn, comlen, rmsd)
! difference in the rmsd from two target structure (same routine!)
      else if (( keyword(1:5).eq.'DRMSD'(1:5) )) then
       call smcv_rmsd_add(comlyn, comlen, drmsd)
! normalized projection onto the vector connecting two structures aligned with the simulation structure
      else if (( keyword(1:4).eq.'PROJ'(1:4) )) then
       call smcv_rmsd_add(comlyn, comlen, proj)
! rms sum of CVs
      else if (( keyword(1:5).eq.'CVRMS'(1:5) )) then
       call smcv_cvrms_add(comlyn, comlen) ! rtmd-like variable (compatibility limited to steered dynamics as of 7.2010):
! z=sqrt( 1/N sum^N_i (z_i - z^0_i)^2 )
      else
        write(info(1),*)'UNRECOGNIZED CV TYPE: ',keyword;call wrndie(0,whoami,trim(info(1)))
      endif
!
      end subroutine smcv_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccc CV-SPECIFIC CODE ccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_posi_com_add(comlyn, comlen, cvtype)
      use cv_posi_com
      use cv_types
      use ivector
      use stream
      use dimens_fcm
      use coord; use coordc
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
!
      implicit none
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=*) comlyn
      integer :: comlen
      integer :: cvtype
! locals
      character(len=11) :: cv_name
!
      integer :: imode, iselct(natom)


!
      integer :: i, j, nslct
      logical :: ok
      real(chm_real) :: k, gam, w
      integer :: frame
      type (int_vector) :: posi_com_list ! for storing atom indices
!
      character(len=len("SMCV_POSI_COM_ADD>") ),parameter::whoami="SMCV_POSI_COM_ADD>";!macro
      data cv_name/'POSI_COM'/
!
      select case (cvtype)
       case (posi_com_x); cv_name=' POSI_COM_X'
       case (posi_com_y); cv_name=' POSI_COM_Y'
       case (posi_com_z); cv_name=' POSI_COM_Z'
       case default;
        call wrndie(0,whoami,trim('UNKNOWN CV TYPE. NOTHING DONE.'))
        return
      end select
! first check for CV options
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      k=gtrmf(comlyn, comlen, 'FORC', 0.0d0) ! can specify force constant manually
      w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0) ! can specify weight manually
      gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0) ! friction coefficient
      frame=gtrmi(comlyn, comlen, 'FRAM', 0) ! coordinate frame index for this position variable
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now process atom selections;
! expecting 1 atom group

      imode=0
      CALL SELRPN(COMLYN,COMLEN,ISELCT,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
      IF(IMODE.NE.0) THEN
       CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
       RETURN
      ENDIF
      NSLCT=NSELCT(NATOM,ISELCT)
!




!
      if(nslct.eq.0) then
       call wrndie(0,whoami,trim('ZERO ATOMS SELECTED. ABORTING.'))



       return
      endif
!
      call int_vector_init(posi_com_list)

      do i=1,natom ! loop over all atoms
       if (iselct(i).eq.1) then
        j=int_vector_add(posi_com_list,i)
        if (j.le.0) then ; call wrndie(0,whoami,trim('COULD NOT ADD ATOM INDEX TO COM LIST')) ; endif
       endif
      enddo







!
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       if (w.lt.0d0) then
        write(info, 664) whoami,cv_name,k,gam
       else
        write(info, 665) whoami,cv_name,k,w,gam
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
       if (frame.ge.1) then
        write(info,'(A,I3)') whoami//' RELATIVE TO LOCAL FRAME ',frame
       else
        write(info,'(A)') whoami//' RELATIVE TO THE ABSOLUTE FRAME'
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
      endif
!
  664 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' AND GAMMA =',F7.3,'.')
  665 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' WEIGHT =',F7.3,' AND GAMMA =',F7.3)
!
! now attempt to add CV
      ok=cv_posi_com_add(cvtype,posi_com_list,k,gam,w,max(frame,0)) ! no mass weighting; disallow negative frame indices
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD CV'))
      endif
! deallocate lists
      call int_vector_done(posi_com_list)
! done!
      end subroutine smcv_posi_com_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_dihe_com_add(comlyn, comlen)
      use cv_dihe_com
      use ivector
!
      use stream
      use coord; use coordc
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
!
      implicit none
!
      character(len=*) :: comlyn
      integer :: comlen
! locals
      character(len=len("DIHE_COM") ),parameter::cv_name="DIHE_COM";!macro
!

      integer :: imode, iselct(natom)



 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      integer :: i, j, atom_group, nslct
      logical :: ok
      real(chm_real) :: k, gam, w
      type (int_vector), dimension(4) :: dihe_com_list ! for storing atom indices
!
      character(len=len("SMCV_DIHE_COM_ADD>") ),parameter::whoami="SMCV_DIHE_COM_ADD>";!macro
!
! first check for CV options
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      k=gtrmf(comlyn, comlen, 'FORC', 0.0d0) ! can specify force constant manually
      w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0) ! can specify weight manually
      gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0) ! friction coeff.
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now process atom selections;
! expecting 4 atom selections that specify each atom group in succession;
! process each selection sequentially
      do atom_group=1,4
!

!
       imode=0
       CALL SELRPN(COMLYN,COMLEN,ISELCT,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0) THEN
        CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
        RETURN
       ENDIF
!
       NSLCT=NSELCT(NATOM,ISELCT)
!




!
       IF(NSLCT.EQ.0) THEN
       call wrndie(0,whoami,trim('ZERO ATOMS SELECTED. ABORTING.'))



       return
       endif
!
       call int_vector_init(dihe_com_list(atom_group))

       do i=1,natom ! loop over all atoms
        if (iselct(i).eq.1) then
         j=int_vector_add(dihe_com_list(atom_group),i)
         if (j.le.0) then ; call wrndie(0,whoami,trim('COULD NOT ADD ATOM INDEX TO COM LIST')) ; endif
        endif
       enddo
!







      enddo ! loop over dihe_com selections
!
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       if (w.lt.0d0) then
        write(info, 664) whoami,cv_name,k,gam
       else
        write(info, 665) whoami,cv_name,k,w,gam
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
!
  664 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' AND GAMMA =',F7.3,'.')
  665 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' WEIGHT =',F7.3,' AND GAMMA =',F7.3)
!
! now attempt to add CV
      ok=cv_dihe_com_add(dihe_com_list,k,gam,w) ! no mass weighting
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD CV'))
      endif
! deallocate lists
      do i=1,4; call int_vector_done(dihe_com_list(i)); enddo
! done!
      end subroutine smcv_dihe_com_add
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_angle_com_add(comlyn, comlen)
      use cv_angle_com
      use ivector
      use stream
      use dimens_fcm
      use coord; use coordc
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
      implicit none
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=*) :: comlyn
      integer :: comlen
! locals
      character(len=len("ANGLE_COM") ),parameter::cv_name="ANGLE_COM";!macro
!

      integer :: imode, iselct(natom)



!
      integer :: i, j, atom_group, nslct
      logical :: ok
      real(chm_real) :: k, gam, w
      type (int_vector), dimension(3) :: angle_com_list ! for storing atom indices
!
      character(len=len("SMCV_ANGLE_COM_ADD>") ),parameter::whoami="SMCV_ANGLE_COM_ADD>";!macro
!
! first check for CV options
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      k=gtrmf(comlyn, comlen, 'FORC', 0.0d0) ! can specify force constant manually
      w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0) ! can specify weight manually
      gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0) ! friction
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now process atom selections;
! expecting 3 atom selections that specify each atom group in succession;
! process each selection sequentially
      do atom_group=1,3

       imode=0
       CALL SELRPN(COMLYN,COMLEN,ISELCT,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0) THEN
        CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
        RETURN
       ENDIF
!
       NSLCT=NSELCT(NATOM,ISELCT)




!
       if (nslct.eq.0) then
        call wrndie(0,whoami,trim('ZERO ATOMS SELECTED. ABORTING.'))



        return
       endif
!
       call int_vector_init(angle_com_list(atom_group))

       do i=1,natom ! loop over all atoms
        if (iselct(i).eq.1) then
         j=int_vector_add(angle_com_list(atom_group),i)
         if (j.le.0) then ; call wrndie(0,whoami,trim('COULD NOT ADD ATOM INDEX TO COM LIST')) ; endif
        endif
       enddo







      enddo ! loop over angle_com selections
!
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       if (w.lt.0d0) then
        write(info, 664) whoami,cv_name,k,gam
       else
        write(info, 665) whoami,cv_name,k,w,gam
       endif
      endif
      write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
  664 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' AND GAMMA =',F7.3,'.')
  665 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' WEIGHT =',F7.3,' AND GAMMA =',F7.3)
!
! now attempt to add CV
      ok=cv_angle_com_add(angle_com_list,k,gam,w) ! no mass weighting
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD CV'))
      endif
! deallocate lists
      do i=1,3; call int_vector_done(angle_com_list(i)); enddo
! done!
      end subroutine smcv_angle_com_add
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_dist_com_add(comlyn, comlen)
      use cv_dist_com
      use ivector
      use stream
      use dimens_fcm
      use coord; use coordc
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
!
      implicit none
!
      character(len=*) comlyn
      integer :: comlen
! locals
      character(len=len("DIST_COM") ),parameter::cv_name="DIST_COM";!macro
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer

      integer :: imode, iselct(natom)



!
      integer :: i, j, atom_group, nslct
      logical :: ok
      real(chm_real) :: k, gam, w
      type (int_vector), dimension(2) :: dist_com_list ! for storing atom indices
!
      character(len=len("SMCV_DIST_COM_ADD>") ),parameter::whoami="SMCV_DIST_COM_ADD>";!macro
!
! first check for CV options
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      k=gtrmf(comlyn, comlen, 'FORC', 0.0d0) ! can specify force constant manually
      w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0) ! can specify weight manually
      gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0) ! friction coeff.
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now process atom selections;
! expecting 2 atom selections that specify each atom group in succession;
! process each selection sequentially
      do atom_group=1,2
!

!
       imode=0
       CALL SELRPN(COMLYN,COMLEN,ISELCT,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0) THEN
        CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
        RETURN
       ENDIF
!
       NSLCT=NSELCT(NATOM,ISELCT)
!




!
       if (nslct.eq.0) then
        call wrndie(0,whoami,trim('ZERO ATOMS SELECTED. ABORTING.'))



        return
       endif
!
       call int_vector_init(dist_com_list(atom_group))

       do i=1,natom ! loop over all atoms
        if (iselct(i).eq.1) then
         j=int_vector_add(dist_com_list(atom_group),i)
         if (j.le.0) then ; call wrndie(0,whoami,trim('COULD NOT ADD ATOM INDEX TO COM LIST')) ; endif
        endif
       enddo







      enddo ! loop over dist_com selections
!
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       if (w.lt.0d0) then
        write(info, 664) whoami,cv_name,k,gam
       else
        write(info, 665) whoami,cv_name,k,w,gam
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
!
  664 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' AND GAMMA =',F7.3,'.')
  665 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' WEIGHT =',F7.3,' AND GAMMA =',F7.3)
!
! now attempt to add CV
      ok=cv_dist_com_add(dist_com_list,k,gam,w) ! no mass weighting
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD CV'))
      endif
! deallocate lists
      do i=1,2; call int_vector_done(dist_com_list(i)); enddo
! done!
      end subroutine smcv_dist_com_add
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_anglvec_add(comlyn, comlen)
      use cv_anglvec
      use ivector
      use stream
      use dimens_fcm
      use coord; use coordc
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
!
      implicit none
!
      character(len=*) :: comlyn
      integer :: comlen
! locals
      character(len=len("ANGLVEC") ),parameter::cv_name="ANGLVEC";!macro
      character(len=8) :: key
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!

      integer :: imode, iselct(natom)



!
      integer :: i, ipt, j, l, nslct
      logical :: ok
      real(chm_real) :: k, gam, w
      type (int_vector), dimension(4) :: atom_list ! for storing atom indices
      integer :: f1, f2
      real(chm_real) :: p(4,3) ! for point definition
      logical :: qp1=.false., qp2=.false., qp3=.false., qp4=.false.
!
      character(len=len("SMCV_ANGLVEC_ADD>") ),parameter::whoami="SMCV_ANGLVEC_ADD>";!macro
!
! first check for CV options
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      k=gtrmf(comlyn, comlen, 'FORC', 0.0d0) ! can specify force constant manually
      w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0) ! can specify weight manually
      gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0) ! friction
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check for frame specification, so that comlyn is cleaned up
      f1=gtrmi(comlyn, comlen, 'FR1', 0)
      f2=gtrmi(comlyn, comlen, 'FR2', 0)
!
! initialize atom arrays
      do i=1,4; call int_vector_init(atom_list(i)); enddo
!
      p=0d0 ! initialize points
      do l=1,4
! now process vector specifications (expecting four points)
       key=nexta8(comlyn,comlen)
       if (( key(1:2).eq.'P1'(1:2) )) then
        ipt=1 ; qp1=.true.
       elseif (( key(1:2).eq.'P2'(1:2) )) then
        ipt=2 ; qp2=.true.
       elseif (( key(1:2).eq.'P3'(1:2) )) then
        ipt=3 ; qp3=.true.
       elseif (( key(1:2).eq.'P4'(1:2) )) then
        ipt=4 ; qp4=.true.
       else
        write(info(1),*)'VECTOR DEFINITION ERROR. EXPECTED "P[1-4]", GOT "',key,'"';call wrndie(0,whoami,trim(info(1)))
       endif
!
       call trima(comlyn, comlen)
       if (indx(comlyn, comlen, 'SELE', 4).eq.1) then ! next word is select
!ccccccccccccc process atom selection

!
        imode=0
        CALL SELRPN(COMLYN,COMLEN,ISELCT,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
        IF(IMODE.NE.0) THEN
         CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
         RETURN
        ENDIF
!
        NSLCT=NSELCT(NATOM,ISELCT)
!




!
        IF(NSLCT.EQ.0) THEN
         call wrndie(0,whoami,trim('ZERO ATOMS SELECTED. ABORTING.'))



         return
        endif
!

        do i=1,natom ! loop over all atoms
         if (iselct(i).eq.1) then
          j=int_vector_add(atom_list(ipt),i)
          if (j.le.0) then ; call wrndie(0,whoami,trim('COULD NOT ADD ATOM INDEX TO COM LIST')) ; endif
         endif
        enddo







!
       else ! specify point manually
        do i=1,3; p(ipt,i)=nextf(comlyn, comlen); enddo
       endif
      enddo
! check that all four points have been added
      if (.not.(qp1.and.qp2.and.qp3.and.qp4)) then
       call wrndie(0,whoami,trim('SOME POINTS WERE NOT DEFINED. NOTHING DONE'))
       return
      endif
!
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       if (w.lt.0d0) then
        write(info, 664) whoami,cv_name,k,gam
       else
        write(info, 665) whoami,cv_name,k,w,gam
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
!
  664 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' AND GAMMA =',F7.3,'.')
  665 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' WEIGHT =',F7.3,' AND GAMMA =',F7.3)
!
! now attempt to add CV
      ok=cv_anglvec_add(atom_list,p,f1,f2,k,gam,w) ! no mass weighting
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD CV'))
      endif
! deallocate lists
      do i=1,4; call int_vector_done(atom_list(i)); enddo
! done!
      end subroutine smcv_anglvec_add
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
! 'frame' is not really a CV but processed here for convenience
      subroutine smcv_frame_add(comlyn, comlen)
      use cv_frames
      use ivector
      use stream
      use dimens_fcm
      use coord; use coordc
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
!
      implicit none
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=*) :: comlyn
      integer :: comlen
! locals

      integer :: imode, iselct(natom)



      integer :: i, j, nslct
      logical :: ok
      type (int_vector) :: frame_list ! for storing atom indices
!
      character(len=len("SMCV_FRAME_ADD>") ),parameter::whoami="SMCV_FRAME_ADD>";!macro
!
! process atom selections;
! specify atom group

!
      imode=0
      CALL SELRPN(COMLYN,COMLEN,ISELCT,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
      IF(IMODE.NE.0) THEN
       CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
       RETURN
      ENDIF
!
      NSLCT=NSELCT(NATOM,ISELCT)
!




!
      if (nslct.lt.4) then ! require at least four atoms, otherwise, can never define frame uniquely
       call wrndie(0,whoami,trim(' FEWER THAN FOUR ATOMS SELECTED. ABORTING.'))



       return
      endif
!
      call int_vector_init(frame_list)

      do i=1,natom ! loop over all atoms
       if (iselct(i).eq.1) then
         j=int_vector_add(frame_list,i)
         if (j.le.0) then
          call wrndie(0,whoami,trim('COULD NOT ADD ATOM INDEX TO FRAME LIST.'))
         endif
       endif
      enddo
!
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       write(info, 665) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
!
  665 format(/A,' WILL ADD NEW REFERENCE FRAME')
! now attempt to add frame
      ok=(frames_add(frame_list).gt.0) !
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD FRAME.'))
      endif
! deallocate lists
      call int_vector_done(frame_list)
! done!
      end subroutine smcv_frame_add
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_quaternion_add(comlyn, comlen)
      use cv_qcomp
      use cv_types
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
      implicit none
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=*) :: comlyn
      integer :: comlen
! locals
      character(len=len("QUATERNION") ),parameter::cv_name="QUATERNION";!macro
      logical :: ok
      real(chm_real) :: k, gam, w
      integer :: f1, f2
!
      character(len=len("SMCV_QUATERNION_ADD>") ),parameter::whoami="SMCV_QUATERNION_ADD>";!macro
!
! first check for CV options
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      k=gtrmf(comlyn, comlen, 'FORC', 0.0d0) ! can specify force constant manually
      w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0) ! can specify weight manually
      gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0) ! friction
      f1=gtrmi(comlyn, comlen, 'FRA1', 0) ! coordinate frame index 1
      f2=gtrmi(comlyn, comlen, 'FRA2', 0) ! coordinate frame index 2
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
!
       if (w.lt.0d0) then
        write(info, 664) whoami,cv_name,k,gam
       else
        write(info, 665) whoami,cv_name,k,w,gam
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
       if (f1.ge.1) then
        write(info,'(A,I3)') whoami//' FRAME1: LOCAL FRAME #',f1
       else
        write(info,'(A)') whoami//' FRAME1: ABSOLUTE FRAME'
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
       if (f2.ge.1) then
        write(info,'(A,I3)') whoami//' FRAME2: LOCAL FRAME #',f2
       else
        write(info,'(A)') whoami//' FRAME2: ABSOLUTE FRAME'
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
      endif
!
  664 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' AND GAMMA =',F7.3,'.')
  665 format(/A,' WILL ADD ',A,' CV WITH K =',F7.3, &
     & ' WEIGHT =',F7.3,' AND GAMMA =',F7.3)
!
! now attempt to add quaternion components, one by one:
      ok= cv_qcomp_add(qcomp_1, f1, f2, k, gam, w) &
     & .and.cv_qcomp_add(qcomp_2, f1, f2, k, gam, w) &
     & .and.cv_qcomp_add(qcomp_3, f1, f2, k, gam, w) &
     & .and.cv_qcomp_add(qcomp_4, f1, f2, k, gam, w)
!
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD QUATERNION CV'))
      endif
! done!
      end subroutine smcv_quaternion_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_cvrms_add(comlyn, comlen)
! this CV is an RMS combination of existing CV; experimental and not fully implemented (intented for steered dynamics)
      use cv_cvrms
      use parselist
      use ivector
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      implicit none
      character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=*) :: comlyn
      integer :: comlen
! locals
      character(len=len("CVRMS") ),parameter::cv_name="CVRMS";!macro
      logical :: ok
      real(chm_real) :: k, gam, w
      type (int_vector) :: cv_list ! for storing cv indices used to calculate RMS
!
      character(len=len("SMCV_CVRMS_ADD>") ),parameter::whoami="SMCV_CVRMS_ADD>";!macro
!
! first check for CV options
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      k=gtrmf(comlyn, comlen, 'FORC', 0.0d0) ! can specify force constant manually
      w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0) ! can specify weight manually
      gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0) ! friction coeff.
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! specify cv to use in the RMS definition
      call ilist_parse(cv_list, comlyn) ! will return allocated cv_list with the indices
!
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       if (w.lt.0d0) then
        write(info, 664) whoami,cv_name,k,gam
       else
        write(info, 665) whoami,cv_name,k,w,gam
       endif
      endif
      write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
  664 format(/A,' WILL ADD ',A,' CV WITH K =',F8.3, &
     & ' AND GAMMA =',F8.3,'.')
  665 format(/A,' WILL ADD ',A,' CV WITH K =',F8.3, &
     & ' WEIGHT =',F8.3,' AND GAMMA =',F8.3)
!
! now attempt to add CV
      ok=cv_cvrms_add(cv_list,k,gam,w) ! no mass weighting
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD CV'))
      endif
! deallocate list
      call int_vector_done(cv_list)
! done!
      end subroutine smcv_cvrms_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_rmsd_add(comlyn, comlen, cvtype)
! much of the code taken from RTMD sources in ../misc
      use cv_rmsd
      use cv_drmsd
      use cv_proj, only : cv_proj_add
      use cv_types
      use stream
      use dimens_fcm
      use coord; use coordc
      use string
      use number
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
!
      implicit none
      character(len=*) :: comlyn
      integer :: comlen
      integer :: cvtype
! locals
      character(len=5) :: cv_name
      logical :: ok, oset
      real(chm_real) :: k, gam, w, rtarget_com(3)
!
      real(chm_real), pointer :: rtarget_o(:,:), rtarget_f(:,:), &
     & rtarget1_o(:,:), rtarget1_f(:,:), &
     & orientWeights(:), forcedWeights(:)
      integer, pointer :: iatom_o(:), iatom_f(:)
      integer :: norient, nforced
!
      integer :: iselct(natom), jselct(natom)
      integer :: imode
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      integer :: i, j, n
      real(chm_real) :: a, b
!
      logical :: use_main, use_comp, qroot, qmass, qtwo ! qtwo: true if using two target structures
!
      character(len=len("SMCV_RMSD_ADD>") ),parameter::whoami="SMCV_RMSD_ADD>";!macro
!
      select case (cvtype)
       case(rmsd ); cv_name='RMSD '; qtwo=.false.
       case(drmsd); cv_name='DRMSD'; qtwo=.true.
       case(proj ); cv_name='PROJ '; qtwo=.true.
       case default
        call wrndie(0,whoami,trim('UNKNOWN CV REQUESTED. NOTHING DONE.'));
        return
      end select
!
      qroot=(MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0)
!
! first check for CV options
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      k=gtrmf(comlyn, comlen, 'FORC', 0.0d0) ! can specify force constant manually
      w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0) ! can specify weight manually
      gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0) ! friction coeff.
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! process atom selections
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      iselct=0
      jselct=0
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! look for orientation atom selection
      oset=.false. ! is there a separate orientation specification?
      i=indxa(comlyn, comlen, 'ORIE') ! find the position of `ORIE`
      oset=(i.gt.0)
! first determine whether a selection keyword follows orie
      if (oset) then
       n=comlen-i+1
       j=indx(comlyn(i:comlen), n, 'SELE', 4)
       if (j.le.0) then ! only if the ORIE directive exists
         call wrndie(0,whoami,trim('ATOM SELECTION MUST BE SPECIFIED AFTER ORIE.'))
         return
        endif
      endif
!*************************************************************************************
! look for the first selection
!*************************************************************************************
      j=indx(comlyn, comlen, 'SELE', 4)
      if (j.eq.0) then ! sele keyword does not exist
       call wrndie(0,whoami,trim('RMSD ATOMS NOT SPECIFIED. NOTHING DONE.'));
       return
      elseif (j.le.i.or..not.oset) then ! no 'orie', or first occurrence of sele before orie (deleted by __INDXa above)
! assume that this is the forcing set selection
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,jselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0) THEN
          CALL WRNDIE(0,whoami,'RMSD ATOMS SELECTION ERROR')
          RETURN
       ENDIF
! orientation selection
       if (oset) then
        IMODE=0
        CALL SELRPN(COMLYN,COMLEN,iselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
        IF(IMODE.NE.0) THEN
         CALL WRNDIE(0,whoami,'ORIENTATION ATOMS SELECTION ERROR')
         RETURN
        ENDIF
       endif
!
      else ! first selection after orie
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,iselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0) THEN
        CALL WRNDIE(0,whoami,'ORIENTATION ATOMS SELECTION ERROR')
        RETURN
       ENDIF
! forcing set selection
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,jselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0) THEN
          CALL WRNDIE(0,whoami,'RMSD ATOMS SELECTION ERROR')
          RETURN
       ENDIF
! number of atoms
       norient=count( iselct(1:natom).gt.0 )
       nforced=count( jselct(1:natom).gt.0 )
!
!
      endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if (nforced.eq.0) then
       call wrndie(0,whoami,trim('RMSD ATOMS NOT SPECIFIED. NOTHING DONE.'))
       return
      endif
!
      if (.not.oset) then ! use forced atoms for orientation too
       norient=nforced
       iselct=jselct
      endif
!
! currently we require at least three atoms for orientation
!
      if (norient.lt.3) then
        call wrndie(0,whoami,trim(' FEWER THAN THREE ATOMS SELECTED FOR ORIENTATION. ABORT.'))
        return
      endif
!
      qmass=(indxa(comlyn, comlen, 'MASS').gt.0) ! mass-weighting flag
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! done with selections
      allocate(iatom_o(norient), &
     & iatom_f(nforced), &
     & rtarget_o(norient,3), &
     & rtarget_f(nforced,3), &
     & orientWeights(norient), &
     & forcedWeights(nforced))
      if (qtwo) allocate(rtarget1_o(norient,3),rtarget1_f(nforced,3))
!
! initialize arrays
      iatom_o=0; iatom_f=0;
      rtarget_o=0.0d0; rtarget_f=0.0d0;
      if (qtwo) then ; rtarget1_o=0.0d0; rtarget1_f=0.0d0; endif ! second structure
      orientWeights=1d0; forcedWeights=1d0
!
! build index arrays
      norient=0
      nforced=0
!
      do i=1,natom
        if (iselct(i).gt.0) then
         norient=norient+1
         iatom_o(norient)=i
        endif
!
        if (jselct(i).gt.0) then
         nforced=nforced+1
         iatom_f(nforced)=i
        endif
!
      enddo
!!!!!!!!! mass-weighting
      if (qmass) then
        do i=1,norient
         orientWeights(i)= &
     & amass(iatom_o(i))*orientWeights(i)
        enddo
!
        do i=1, nforced
         forcedWeights(i)= &
     & amass(iatom_f(i))*forcedWeights(i)
        enddo
!
      endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! load and save coordinates of the target structure
      use_main=((indxa(comlyn, comlen, 'MAIN')).gt.0)
      use_comp=((indxa(comlyn, comlen, 'COMP')).gt.0)
      if (use_comp) then
        if (use_main) then
         call wrndie(0,whoami,trim('MAIN AND COMP CANNOT BOTH BE SPECIFIED. USING MAIN.'))
         use_comp=.false.
        endif
      else
        use_main=.true. ! default
      endif
!
      if (use_comp) then
! orient
        do i=1,norient
         rtarget_o(i,1)=xcomp(iatom_o(i))
         rtarget_o(i,2)=ycomp(iatom_o(i))
         rtarget_o(i,3)=zcomp(iatom_o(i))
        enddo
! forced
        do i=1,nforced
         rtarget_f(i,1)=xcomp(iatom_f(i))
         rtarget_f(i,2)=ycomp(iatom_f(i))
         rtarget_f(i,3)=zcomp(iatom_f(i))
        enddo
! second reference structure:
        if (qtwo) then
         do i=1,norient
          rtarget1_o(i,1)=x(iatom_o(i))
          rtarget1_o(i,2)=y(iatom_o(i))
          rtarget1_o(i,3)=z(iatom_o(i))
         enddo
!
         do i=1,nforced
          rtarget1_f(i,1)=x(iatom_f(i))
          rtarget1_f(i,2)=y(iatom_f(i))
          rtarget1_f(i,3)=z(iatom_f(i))
         enddo
        endif ! qtwo
!
      else ! use main coordinates
! orient
        do i=1,norient
         rtarget_o(i,1)=x(iatom_o(i))
         rtarget_o(i,2)=y(iatom_o(i))
         rtarget_o(i,3)=z(iatom_o(i))
        enddo
! forced
        do i=1,nforced
         rtarget_f(i,1)=x(iatom_f(i))
         rtarget_f(i,2)=y(iatom_f(i))
         rtarget_f(i,3)=z(iatom_f(i))
        enddo
! second reference structure:
        if (qtwo) then
         do i=1,norient
          rtarget1_o(i,1)=xcomp(iatom_o(i))
          rtarget1_o(i,2)=ycomp(iatom_o(i))
          rtarget1_o(i,3)=zcomp(iatom_o(i))
         enddo
!
         do i=1,nforced
          rtarget1_f(i,1)=xcomp(iatom_f(i))
          rtarget1_f(i,2)=ycomp(iatom_f(i))
          rtarget1_f(i,3)=zcomp(iatom_f(i))
         enddo
        endif ! qtwo
!
      endif ! use_comp
!
! check for undefined values
      if (any(rtarget_o.eq.anum).or.any(rtarget_f.eq.anum)) then
        call wrndie(0,whoami,trim('FIRST TARGET STRUCTURE HAS UNDEFINED COORDINATES. QUITTING.'))
        return
      elseif (qtwo) then
       if (any(rtarget1_o.eq.anum).or.any(rtarget1_f.eq.anum)) then
        call wrndie(0,whoami,trim('SECOND TARGET STRUCTURE HAS UNDEFINED COORDINATES. QUITTING.'))
        return
       endif
      endif
!
! normalize weighting coefficients
! note: routines in rtmd_aux do _not_ perform normalization; result affects FP precision, analytically, there is no difference
!
      a=sum(orientWeights)
      b=sum(forcedWeights)
      if (abs(a).gt.RSMALL) then
        a=1d0/a
        orientWeights=a*orientWeights
      endif
!
      if (abs(b).gt.RSMALL) then
        b=1d0/b
        forcedWeights=b*forcedWeights
      endif
!
! translate target structure so that its centroid is at the origin
      rtarget_com=matmul(transpose(rtarget_o),orientWeights)
!
      rtarget_o(:,1)=rtarget_o(:,1)-rtarget_com(1)
      rtarget_o(:,2)=rtarget_o(:,2)-rtarget_com(2)
      rtarget_o(:,3)=rtarget_o(:,3)-rtarget_com(3)
!
      rtarget_f(:,1)=rtarget_f(:,1)-rtarget_com(1)
      rtarget_f(:,2)=rtarget_f(:,2)-rtarget_com(2)
      rtarget_f(:,3)=rtarget_f(:,3)-rtarget_com(3)
!
      if (qtwo) then ! repeat for second structure
       rtarget_com=matmul(transpose(rtarget1_o),orientWeights)
!
       rtarget1_o(:,1)=rtarget1_o(:,1)-rtarget_com(1)
       rtarget1_o(:,2)=rtarget1_o(:,2)-rtarget_com(2)
       rtarget1_o(:,3)=rtarget1_o(:,3)-rtarget_com(3)
!
       rtarget1_f(:,1)=rtarget1_f(:,1)-rtarget_com(1)
       rtarget1_f(:,2)=rtarget1_f(:,2)-rtarget_com(2)
       rtarget1_f(:,3)=rtarget1_f(:,3)-rtarget_com(3)
      endif
!
! print summary
      if (qroot) then
!
       if (w.lt.0d0) then
        write(info, 664) whoami,cv_name,k,gam
       else
        write(info, 665) whoami,cv_name,k,w,gam
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
       write(info,103) whoami, nforced ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
       write(info,100) whoami, norient ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
       if (qmass) then ; write(info,102) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
       if (qtwo) then
        if (use_comp) then
         write(info,105) whoami, 'COMP'
        else
         write(info,105) whoami, 'MAIN'
        endif
       else
        if (use_comp) then
         write(info,104) whoami, 'COMP'
        else
         write(info,104) whoami, 'MAIN'
        endif
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';!
  664 format(/A,' WILL ADD ',A,' CV WITH K =',F8.3, &
     & ' AND GAMMA =',F8.3,'.')
  665 format(/A,' WILL ADD ',A,' CV WITH K =',F8.3, &
     & ' WEIGHT =',F8.3,' AND GAMMA =',F8.3)
!
 100 format(A,' WILL ORIENT TARGET STRUCTURE(S) BASED ON ',i5, &
     & ' ATOMS')
 102 format(A,' WILL USE MASS-WEIGHTING.')
 103 format(A,' ',i5,' FORCED ATOMS FOUND.')
 104 format(A,' TARGET STRUCTURE TAKEN FROM ',A,' SET.')
 105 format(A,' FIRST TARGET STRUCTURE TAKEN FROM ',A,' SET.')
!
      endif ! qroot
! now attempt to add CV
      ok=.false.
      select case(cvtype)
       case(rmsd);
        ok=cv_rmsd_add(iatom_o, iatom_f, rtarget_o, rtarget_f, &
     & orientWeights, forcedWeights, &
     & k, gam, w)
       case(drmsd);
        ok=cv_drmsd_add(iatom_o, iatom_f, rtarget_o, rtarget_f, &
     & rtarget1_o, rtarget1_f, &
     & orientWeights, forcedWeights, &
     & k, gam, w)
       case(proj);
        ok=cv_proj_add(iatom_o, iatom_f, rtarget_o, rtarget_f, &
     & rtarget1_o, rtarget1_f, &
     & orientWeights, forcedWeights, &
     & k, gam, w)
      end select
!
      if (.not.ok) then
       call wrndie(0,whoami,trim('COULD NOT ADD CV'))
      endif
! deallocate variables
      deallocate(iatom_o, iatom_f, rtarget_o, rtarget_f, &
     & orientWeights, forcedWeights)
      if (qtwo) deallocate(rtarget1_o, rtarget1_f)
!
      end subroutine smcv_rmsd_add
!
#endif
#endif /* automatically protect all code */
