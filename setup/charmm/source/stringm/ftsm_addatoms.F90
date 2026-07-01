#if (KEY_STRINGM==1) /*  automatically protect all code */
! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! routines for adding atoms or COMs of atoms to FTSM forcing/orientation sets
! replaces the built-in selection process inside ftsm.ftn (string ftsm set orie/rmsd)
!
#if (KEY_STRINGM==1)
!=====================================================================================
      subroutine ftsm_add_atomic_set(comlyn, comlen)
! implements original functionality : one ftsm bead per atom
      use ftsm_var
      use ftsm_util
      use string
      use dimens_fcm
      use select, only : selcta, selrpn, nselct; use psf
      use consta
      use multicom_aux;
      use stream
      use coord; use coordc
!
      implicit none
!
      character(len=*) :: comlyn
      integer :: comlen
      integer :: iorie, irmsd
      integer :: i, j, a
      logical :: qroot, qslave, qprint
!
      character(len=len("FTSM_ADD_ATOMIC_SET>") ),parameter::whoami="FTSM_ADD_ATOMIC_SET>";!macro
!
      integer :: imode, isele
      integer :: iselct(natom)


 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qslave=((MPI_COMM_LOCAL.ne.MPI_COMM_NULL).and.SIZE_LOCAL.gt.1)
      qprint=qroot.and.ME_STRNG.eq.0
!
      iorie=indxa(comlyn, comlen, 'ORIE')
      irmsd=indxa(comlyn, comlen, 'RMSD')
!
      if (iorie.gt.0) then
! process orientation atom selection
! determine whether a selection keyword follows orie
         isele=indx(comlyn, comlen, 'SELE', 4)
         if (isele.ge.iorie) then
!

          iselct=0
! process selection
          IMODE=0
          CALL SELRPN(COMLYN,COMLEN,iselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
          IF(IMODE.NE.0) THEN
           call wrndie(0,whoami,trim('ORIENTATION ATOMS SELECTION ERROR'))
           RETURN
          ENDIF
          norient=count( iselct.gt.0 )




! deallocate existing orientation arrays
!
          if (ftsm_com_on) then
           call int_vlist_done(iatoms_o)
          else
           if(associated(iatom_o))deallocate(iatom_o)
          endif
          if(associated(orientWeights))deallocate(orientWeights)
!
! currently we require at least three atoms for orientation
!
          if (norient.lt.3) then
           call wrndie(0,whoami,trim(' FEWER THAN THREE ATOMS SELECTED FOR ORIENTATION. WILL NOT ORIENT.'))
           norient=0
           qorient=.false.
           return
          else
           qorient=.true.
          endif
!
          if (.not. ftsm_com_on) then
           allocate(iatom_o(norient))
           allocate(orientWeights(norient));
           orientWeights=one/norient ! default behavior
          endif
! build index array
!

          norient=0
          do i=1,natom
           if (iselct(i).gt.0) then
            norient=norient+(1);
            if (ftsm_com_on) then
             j=int_vlist_uadd(iatoms_o,norient,i) ! create a new list with label norient ; add to it the index i
            else
             iatom_o(norient)=i
            endif ! ftsm_com_on
!
           endif ! iselct
          enddo
!
! determine whether the new orientation set is the same as the existing forcing set
!
          call ftsm_qdiffrot() ! compute qdiffrot
!
          if(associated(r_o,target=r_f)) then; nullify(r_o);else; if(associated(r_o))deallocate(r_o); endif
          if (.not.ftsm_com_on) then ! with ftsm_com_on, done in ftsm_alloc_coor_wgt
           if (.not.qdiffrot) then
            r_o=>r_f;
           else
            allocate(r_o(norient,3,num_sets)); r_o=anum;
            if (nforced.gt.0) call ftsm_compute_overlap_ind() ! compute overlap indices in iatom_both (works for ftsm_com_on)
           endif
           if (.not. associated(rcom)) allocate(rcom(3,num_sets))
           rcom=zero ! must be zero initially
          endif
!
!
! print summary
          if (qprint) then
            write(info,100) whoami, norient ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 100 format(A,' WILL ORIENT STRUCTURES BASED ON ',i5,' ATOMS')
            write(info,101) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 101 format(A,' ORIENTATION WEIGHTS UNIFORM.')
            if (qdiffrot) then
             write(info,102) whoami
            else
             write(info,103) whoami
            endif
            write(OUTU,'(A)') pack(info,info.ne.'');info='';
 102 format (A, ' ORIENTATION AND FORCING ATOMS ARE DIFFERENT')
 103 format (A, ' ORIENTATION AND FORCING ATOMS ARE IDENTICAL')
          endif ! qprint
         else
          call wrndie(0,whoami,trim(' ATOM SELECTION MUST BE SPECIFIED AFTER ORIE.'))
          return
         endif
         call ftsm_define_rtmd_type()
!=======================================================================
      elseif (irmsd.gt.0) then
! process forcing atom selection
! determine whether a selection keyword follows 'rmsd'
         isele=indx(comlyn, comlen, 'SELE', 4)
         if (isele.gt.irmsd) then
!
          iselct=0
          IMODE=0
          CALL SELRPN(COMLYN,COMLEN,iselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
          IF(IMODE.NE.0) THEN
           call wrndie(0,whoami,trim('FORCING ATOMS SELECTION ERROR'))
           RETURN
          ENDIF
          nforced=count( iselct.gt.0 )
!
          if (nforced.le.0) then
           call wrndie(0,whoami,trim('NO FORCING ATOMS SELECTED. ABORT.'))
           return
          endif
!
          if(associated(forcedWeights))deallocate(forcedWeights)
          if(associated(Mtensor))deallocate(Mtensor)
!
          if (ftsm_com_on) then
           call int_vlist_done(iatoms_f)
          else
            if(associated(iatom_f))deallocate(iatom_f)
           allocate(iatom_f(nforced)); iatom_f=0
           allocate(forcedWeights(nforced));
           forcedWeights=one/nforced ! default behavior
           allocate(Mtensor(3,3,nforced,nforced,2)); Mtensor=zero ; do a=1,3; do i=1, nforced ; Mtensor(a,a,i,i,:)=one ; enddo ; enddo! allocate & set to I
          endif
!
!
! build index array
          nforced=0
          do i=1,natom
           if (iselct(i).gt.0) then
            nforced=nforced+(1);
            if (ftsm_com_on) then
             j=int_vlist_uadd(iatoms_f,nforced,i) ! create a new list with label norient ; add to it the index i
            else
             iatom_f(nforced)=i
            endif ! ftsm_com_on
!
           endif ! iselct
          enddo
!
! determine whether the new orientation set is the same as the existing forcing set
!
          call ftsm_qdiffrot() ! computes qdiffrot
!
          if(associated(r_f,target=r_o)) then; nullify(r_f);else; if(associated(r_f))deallocate(r_f); endif ! free/nullify r_f
          if (.not. ftsm_com_on) then ! with ftsm_com_on, done in ftsm_alloc_coor_wgt
           if (.not.qdiffrot) then
            r_f=>r_o;
           else
            allocate(r_f(nforced,3,num_sets)); r_f=anum;
            if (norient.gt.0) call ftsm_compute_overlap_ind() ! compute overlap indices in iatom_both
           endif
          endif
! print summary
          if (qprint) then
            write(info,104) whoami, nforced ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 104 format(A,' WILL APPLY FORCES TO ',i5,' ATOMS')
            write(info,105) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 105 format(A,' FORCING WEIGHTS UNIFORM.')
            if (qdiffrot) then
             write(info,102) whoami
             if (.not.qorient) then
              write(info(2),106) whoami
 106 format(A,' BEST-FITTING IS DISABLED (PERHAPS ORIENTATION ATOMS ARE NOT DEFINED).')
             endif
            else
             write(info,103) whoami
            endif
            write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif ! qprint
         else
          call wrndie(0,whoami,trim('ATOM SELECTION MUST BE SPECIFIED AFTER "RMSD".'))
          return
         endif
!=====================================================================================
      endif ! iorie or irmsd
!
      end subroutine ftsm_add_atomic_set
!=====================================================================================
      subroutine ftsm_add_com(comlyn, comlen)
! COM functionality : each ftsm bead centered on the COM of a group of one or more atoms
      use ftsm_var
      use ivector; use ivector_list; use rvector; use rvector_list
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
      character(len=len("FTSM_ADD_COM>") ),parameter::whoami="FTSM_ADD_COM>";!macro
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=*) comlyn
      integer :: comlen
!
      integer :: imode, iselct(natom)
!
      integer :: i, j, nslct, ind, ior, ifc
      type (int_vector) :: posi_com_list ! for storing atom indices
      type (int_vlist), pointer :: iatoms
      logical :: qdupl ! check for duplicates
!
      character(len=11) :: keyword
!
      if (.not.ftsm_initialized) then; call wrndie(0,whoami,trim('FTSM NOT INITIALIZED. NOTHING DONE.')); return; endif;
!
      keyword=nexta8(comlyn,comlen) ! directive
      if (( keyword(1:4).eq.'ORIE'(1:4) )) then
       keyword='ORIENTATION'
       iatoms=>iatoms_o
       ior=1 ; ifc=0
       if(associated(r_o,target=r_f)) then; nullify(r_o);else; if(associated(r_o))deallocate(r_o); endif ! ensure that arrays that depend on the COM group definitions get reallocated
      elseif ( (( keyword(1:4).eq.'RMSD'(1:4) )) .or. (( keyword(1:4).eq.'FORC'(1:4) ) ) ) then
       keyword='FORCING'
       iatoms=>iatoms_f
       ior=0 ; ifc=1
       if(associated(r_f,target=r_o)) then; nullify(r_f);else; if(associated(r_f))deallocate(r_f); endif
      else
       call wrndie(0,whoami,trim('"ADD" MUST BE FOLLOWED BY "RMSD" OR "ORIE". NOTHING DONE.'))
       return
      endif
!
! process atom selections;
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
        j=int_vector_uadd(posi_com_list,i)
        if (j.le.0) then ; call wrndie(0,whoami,trim('COULD NOT ADD ATOM INDEX TO COM LIST')) ; endif
       endif
      enddo
!
! print short summary
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
        write(info, 664) whoami,trim(keyword)
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
  664 format(/A,' WILL ADD COM POSITION GROUP TO ',A,' SET.')
      endif
! now attempt to add group
! 1) sort atoms in increasing order just in case (not needed for CHARMM)
      call int_vector_sort(posi_com_list,'i')
!
! search for an _exact_ duplicate:
      qdupl=.false.
      ind=iatoms%last
      do i=1, ind
       qdupl=int_vector_eq_ordered(iatoms%v(i), posi_com_list)
       if (qdupl) then
        write(info(1),*)'SPECIFIED COM GROUP IS ALREADY PRESENT IN THE ',keyword,' SET. ABORT.';call wrndie(0,whoami,trim(info(1)));
        exit
       endif
      enddo
!
      if (.not.qdupl) then
       ind=ind+(1);
       j=int_vlist_uadd(iatoms, ind) ! add list with label "ind"
       call int_vector_set(iatoms%v(j), posi_com_list)
       norient=norient+(ior);
       nforced=nforced+(ifc);
       ftsm_compute_qdiffrot=.true.
      endif
! deallocate list
      call int_vector_done(posi_com_list)
!
      end subroutine ftsm_add_com
!=====================================================================================
      subroutine ftsm_clear_com(comlyn, comlen)
      use ftsm_var
      use stream
      use ivector; use ivector_list; use rvector; use rvector_list
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      implicit none
      character(len=*) :: comlyn
      integer :: comlen
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
      character(len=11) :: keyword
!
      character(len=len("FTSM_CLEAR_COM>") ),parameter::whoami="FTSM_CLEAR_COM>";!macro
      if (.not.ftsm_initialized) then; call wrndie(0,whoami,trim('FTSM NOT INITIALIZED. NOTHING DONE.')); return; endif;
!
      keyword=nexta8(comlyn,comlen) ! directive
      if (( keyword(1:4).eq.'ORIE'(1:4) )) then
       keyword='ORIENTATION'
       norient=0
       qorient=.false.
       call int_vlist_done(iatoms_o)
!
       if (associated(wgts_o, target=wgts_f)) then ! take care not to destroy wgts_f
        allocate(wgts_o)
        call real_vlist_init(wgts_o)
       else
        call real_vlist_done(wgts_o)
       endif
!
       if(associated(r_o,target=r_f)) then; nullify(r_o);else; if(associated(r_o))deallocate(r_o); endif
       if(associated(orientWeights))deallocate(orientWeights)
      elseif ( (( keyword(1:4).eq.'RMSD'(1:4) )) .or. (( keyword(1:4).eq.'FORC'(1:4) ) ) ) then
       keyword='FORCING'
       nforced=0
       call int_vlist_done(iatoms_f)
!
       if (associated(wgts_f, target=wgts_o)) then
        allocate(wgts_f)
        call real_vlist_init(wgts_f)
       else
        call real_vlist_done(wgts_f)
       endif
!
       if(associated(r_f,target=r_o)) then; nullify(r_f);else; if(associated(r_f))deallocate(r_f); endif
       if(associated(forcedWeights))deallocate(forcedWeights)
      else
       call wrndie(0,whoami,trim('"CLEAR" MUST BE FOLLOWED BY "RMSD" OR "ORIE". NOTHING DONE.'))
       return
      endif
!
      ftsm_compute_qdiffrot=.true.
! print
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
        write(info, 665) whoami,trim(keyword)
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
  665 format(/A,' ',A,' COM POSITION SET CLEARED.')
      endif
!
      end subroutine ftsm_clear_com
!
#endif
#endif /* automatically protect all code */
