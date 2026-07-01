! Because these routines are called from many places in the
! original GAMESS we cannot make this a module
#if KEY_GAMESS==1 /*gamess*/
#if 1 /* 0 for original-ddi*/
#if KEY_PARALLEL==1
!
!     Distributed Data Interface (DDI)
!     ==========================
!
!     The DDI library is used by parallel GAMESS.
!     The implementation in GAMESS uses 2 threads per CPU,
!     and client-server model which CANNOT be made compatible 
!     with CHARMM parallel library. The new library was needed.
!     This is the implementation of DDI API with the CHARMM
!     parallel communication routines.
!
!     There is also efficiency argument: Original DDI uses direct
!     MPI global communication routines (MPI_ALLREDUCE,...) while
!     CHARMM implementation of these (##CMPI) is usually faster.
!
!     Written in June 2000 by Milan Hodoscek
!
!     Status: All the routines needed for standard GAMESS 
!             (functionality of versions before year 2000) are
!             implemented. The new mp2ddi is not supported yet.
!
!
!     Subroutines in this file:
!     =========================
!     DDI_PBEG(NWDVAR)                      INITIALIZE DDI PARAMETERS
!     DDI_PEND(ISTAT)                       MAKE TIDY EXIT
!     DDI_NPROC(DDI_NP,DDI_ME)              NUMBER OF NODES AND NODE ID
! [*] DDI_MEMORY(MEMREP,MEMDDI,EXETYP)      ALLOCATE SHARED MEMORY REGION
! [*] DDI_CREATE(IDIM,JDIM,HANDLE)          CREATE DISTRIBUTED MATRIX
! [*] DDI_DESTROY(HANDLE)                   DESTROY DISTRIBUTED MATRIX
! [*] DDI_DISTRIB(HANDLE,NODE,ILOC,IHIC,JLOC,JHIC)  QUERY DM DISTRIBUTION
! [*] DDI_DLBRESET                          RESET DLB TASK COUNTER
!     DDI_SYNC(SNCTAG)                      BARRIER SYNCHRONIZATION
!     DDI_GSUMF(MSGTAG,BUFF,MSGLEN)         FLOATING POINT GLOBAL SUM
!     DDI_GSUMI(MSGTAG,BUFF,MSGLEN)         INTEGER GLOBAL SUM
!     DDI_OUTPUT(PFLAG)                     SET PRINT FLAG OFF/ON
!     DDI_BCAST(MSGTAG,TYPE,BUFF,LEN,FROM)  BROADCAST DATA TO ALL NODES
!    
!                         POINT-TO-POINT TASKS
!     DDI_SEND(SNDBUF,LEN,TO)               SYNCHRONOUS SEND
!     DDI_RECV(RCVBUF,LEN,FROM)             SYNCHRONOUS RECEIVE
! [*] DDI_RCVANY(RCVBUF,LEN,FROM)           SYNCHRONOUS RECEIVE FROM ANY
! [*] DDI_DLBNEXT(DLB_COUNTER)              GET NEXT DLB TASK INDEX
! [*] DDI_GET(HANDLE,ILO,IHI,JLO,JHI,BUFF)  GET BLOCK OF MATRIX
! [*] DDI_PUT(HANDLE,ILO,IHI,JLO,JHI,BUFF)  PUT BLOCK OF MATRIX
! [*] DDI_ACC(HANDLE,ILO,IHI,JLO,JHI,BUFF)  ACCUMULATE DATA INTO BLOCK 
!
!     As of June 2000 the routines noted by [*] are empty.
!
SUBROUTINE DDI_PBEG(NWDVAR)
  !-----------------------------------------------------------------------
  !
  !     This is empty routine since CHARMM already did the job
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) NWDVAR
  !
  RETURN
END SUBROUTINE DDI_PBEG
!
SUBROUTINE DDI_PEND(ISTAT)
  !-----------------------------------------------------------------------
  !
  !     This is empty routine since CHARMM will take care of it
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) ISTAT
  !
  RETURN
END SUBROUTINE DDI_PEND
!
SUBROUTINE DDI_NPROC(DDI_NP,DDI_ME)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  use parallel
  implicit none
  !
  INTEGER(gms_int) DDI_NP, DDI_ME
  !
  DDI_ME = MYNOD
  DDI_NP = NUMNOD
  !
  RETURN
END SUBROUTINE DDI_NPROC
!
SUBROUTINE DDI_NPROC_comm(comm,DDI_NP,DDI_ME)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  use parallel
  implicit none
  !
  INTEGER(gms_int) DDI_NP, DDI_ME, comm(*)
  !
  DDI_ME = MYNOD
  DDI_NP = NUMNOD
  !
  RETURN
END SUBROUTINE DDI_NPROC_COMM
!
SUBROUTINE DDI_NNODE(DDI_NP,DDI_ME)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  use parallel
  implicit none
  !
  INTEGER(gms_int) DDI_NP, DDI_ME
  !
  DDI_ME = MYNOD
  DDI_NP = NUMNOD
  !
  !     GDDI is not supported in CHARMM so this returns the same
  !     data as DDI_NPROC, because it is always called
  !      WRITE(OUTU,*)'**** $GDDI not supported with CHARMM ****'
  !
  RETURN
END SUBROUTINE DDI_NNODE
!
SUBROUTINE DDI_MEMORY(MEMREP,MEMDDI,EXETYP)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) MEMREP, MEMDDI
  real(chm_real) EXETYP
  !
  !     We don't want these print messages
  !
  !CC      IF(PRNLEV.GE.2)
  !CC     $     WRITE(OUTU,'(A)')'%%%-DDI: DDI_MEMORY not implemented.'
  !
  RETURN
END SUBROUTINE DDI_MEMORY
!
SUBROUTINE DDI_CREATE(IDIM,JDIM,HANDLE)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) IDIM,JDIM,HANDLE
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_CREATE not implemented.'
  !
  RETURN
END SUBROUTINE DDI_CREATE
!
SUBROUTINE DDI_DESTROY(HANDLE)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) HANDLE
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_DESTROY not implemented.'
  !
  RETURN
END SUBROUTINE DDI_DESTROY
!
SUBROUTINE DDI_ZERO(HANDLE)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) HANDLE
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_ZERO not implemented.'
  !
  RETURN
END SUBROUTINE DDI_ZERO
!
SUBROUTINE DDI_DISTRIB(HANDLE,NODE,ILOC,IHIC,JLOC,JHIC)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) HANDLE,NODE, ILOC, IHIC, JLOC, JHIC
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_DISTRIB not implemented.'
  !
  RETURN
END SUBROUTINE DDI_DISTRIB
!
SUBROUTINE DDI_DLBRESET
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  ! this routine can be empty in case of CHARMM/MPI interface
  !!! WRITE(OUTU,'(A)')'%%%-DDI: DDI_DLBRESET not implemented.'
  !
  RETURN
END SUBROUTINE DDI_DLBRESET
!
SUBROUTINE DDI_SYNC(SNCTAG)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) SNCTAG
  !
  CALL PSYNC()
  !
  RETURN
END SUBROUTINE DDI_SYNC
!
SUBROUTINE DDI_GSUMF(MSGTAG,BUFF,MSGLEN)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  use parallel, only: gcomb
  implicit none
  !
  !
  INTEGER(gms_int) MSGTAG,MSGLEN
  integer lmsglen
  real(chm_real) BUFF(*)
  !
  lmsglen=msglen
  CALL GCOMB(BUFF,LMSGLEN)
  !
  RETURN
END SUBROUTINE DDI_GSUMF
!
SUBROUTINE DDI_GSUMI(MSGTAG,BUFF,MSGLEN)
  !-----------------------------------------------------------------------
  !
  !     This routine has no equivalent in the CHARMM.
  !     It is now implemented only for MPI. Any non MPI
  !     based parallel computer will not be able to compile
  !     this module.
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(GMS_INT) MSGTAG,BUFF(*),MSGLEN
  INTEGER,ALLOCATABLE,DIMENSION(:) :: LBUFF
  INTEGER LMSGLEN,ICOPY
  !
  LMSGLEN=MSGLEN
  ALLOCATE(LBUFF(LMSGLEN))
  DO ICOPY=1,LMSGLEN
     LBUFF(ICOPY)=BUFF(ICOPY)
  ENDDO
  !
  CALL IGCOMB(LBUFF,LMSGLEN)
  DO ICOPY=1,LMSGLEN
     BUFF(ICOPY)=LBUFF(ICOPY)
  ENDDO
  !
  DEALLOCATE(LBUFF)
  !
  RETURN
END SUBROUTINE DDI_GSUMI
!
SUBROUTINE DDI_BCAST(MSGTAG,ELTYPE,BUFF,LEN,FROM)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) MSGTAG, BUFF(*), LEN, FROM
  CHARACTER(len=1) ELTYPE
  !
  IF(FROM.NE.0) THEN
     WRITE(OUTU,'(A)') &
          '%%%-DDI::DDI_BCAST>broadcast from non 0 not implemented.'
     CALL WRNDIE(-5,'<DDI_BCAST>','Not implemented yet.')
  ENDIF
  IF (ELTYPE == 'F') THEN
     CALL PSND8i(BUFF,LEN)
  ELSE IF (ELTYPE == 'I') THEN
!     CALL PSND4(BUFF,LEN)
     CALL PSND8i(BUFF,LEN)     ! Now, we must use this one here...
  ELSE
     CALL WRNDIE(-5,'<DDI_BCAST>','Type not supported.')
  ENDIF
  !
  RETURN
END SUBROUTINE DDI_BCAST
!                
SUBROUTINE DDI_SEND(SNDBUF,LEN,TO)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  use parallel
  implicit none
  !
  INTEGER(gms_int) LEN,TO
  real(chm_real) SNDBUF(*)
  integer llen,lto,typ
  !
  lto=ippmap(to)
  typ=1
  llen=len
  CALL GSEN(LTO,typ,SNDBUF,LEN)
  !
  RETURN
END SUBROUTINE DDI_SEND
!
SUBROUTINE DDI_RECV(RCVBUF,LEN,FROM)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  use parallel
  implicit none
  !
  INTEGER(gms_int) LEN,FROM
  real(chm_real) RCVBUF(*)
  integer llen,lfrom,ltyp
  !
  ltyp=1
  llen=len
  lfrom=ippmap(from)
  !
  CALL GREC(LFROM,LTYP,RCVBUF,LLEN)
  !
  RETURN
END SUBROUTINE DDI_RECV
!
!*MODULE DDI      *DECK DDI_OUTPUT
SUBROUTINE DDI_OUTPUT(DDI_ON_OFF)
  !
  !  -------------------------------------------------------------------
  !  TOGGLE MESSAGES FROM DDI_CREATE,DDI_DESTROY
  !  -------------------------------------------------------------------
  !
  IMPLICIT NONE
  LOGICAL DDI_ON_OFF, DDI_PRT
  COMMON /DDIFLG/ DDI_PRT
  !
  DDI_PRT = DDI_ON_OFF
  RETURN
END SUBROUTINE DDI_OUTPUT
!
SUBROUTINE DDI_RCVANY(RCVBUF,LEN,FROM)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) LEN,FROM,RCVBUF(*)
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_RCVANY not implemented.'
  !
  RETURN
END SUBROUTINE DDI_RCVANY
!
SUBROUTINE DDI_DLBNEXT(DLB_COUNTER)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) DLB_COUNTER
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_DLBNEXT not implemented.'
  !
  return
end SUBROUTINE DDI_DLBNEXT
!
SUBROUTINE DDI_GET(HANDLE,ILO,IHI,JLO,JHI,BUFF)
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) HANDLE,ILO,IHI,JLO,JHI
  real(chm_real) BUFF(*)
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_GET not implemented.'
  !
  RETURN
END SUBROUTINE DDI_GET
!
SUBROUTINE DDI_PUT(HANDLE,ILO,IHI,JLO,JHI,BUFF)
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) HANDLE,ILO,IHI,JLO,JHI
  real(chm_real) BUFF(*)
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_PUT not implemented.'
  !
  RETURN
END SUBROUTINE DDI_PUT
!
SUBROUTINE DDI_GET_COMM(HANDLE,ILO,IHI,JLO,JHI,BUFF,COMMID)
  use chm_kinds
  implicit none
  INTEGER(gms_int) HANDLE,COMMID,ILO,IHI,JLO,JHI
  REAL(CHM_REAL) BUFF(*)
  !     ignore the communicator
  CALL DDI_GET(HANDLE, ILO, IHI, JLO, JHI, BUFF)
  RETURN
END SUBROUTINE DDI_GET_COMM
!
SUBROUTINE DDI_PUT_COMM(HANDLE,ILO,IHI,JLO,JHI,BUFF,COMMID)
  use chm_kinds
  implicit none
  INTEGER(gms_int) HANDLE,COMMID,ILO,IHI,JLO,JHI
  REAL(CHM_REAL) BUFF(*)
  !     ignore the communicator
  CALL DDI_PUT(HANDLE, ILO, IHI, JLO, JHI, BUFF)
  RETURN
END SUBROUTINE DDI_PUT_COMM
!
SUBROUTINE DDI_ACC(HANDLE,ILO,IHI,JLO,JHI,BUFF)
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  INTEGER(gms_int) HANDLE,ILO,IHI,JLO,JHI
  real(chm_real) BUFF(*)
  !
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)')'%%%-DDI: DDI_ACC not implemented.'
  !
  RETURN
END SUBROUTINE DDI_ACC
!
SUBROUTINE DDI_LEVEL(L)
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  LOGICAL L
  L=.TRUE.
  RETURN
END SUBROUTINE DDI_LEVEL
!
!     DDI version 2 additions
!
SUBROUTINE ddi_irecv()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  return
end SUBROUTINE ddi_irecv
SUBROUTINE ddi_isend()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_isend not implemented.'
  return
end SUBROUTINE ddi_isend
SUBROUTINE ddi_wait()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_WAIT not implemented.'
  return
end SUBROUTINE ddi_wait
SUBROUTINE ddi_procdlb_destroy()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_PROCDLB_DESTROY not implemented.'
  return
end SUBROUTINE ddi_procdlb_destroy
SUBROUTINE ddi_procdlb_create()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)')'%%%-DDI2: DDI_PROCDLB_CREATE not implemented.'
  return
end SUBROUTINE ddi_procdlb_create
SUBROUTINE ddi_procdlb_reset()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)')'%%%-DDI2: DDI_PROCDLB_RESET not implemented.'
  return
end SUBROUTINE ddi_procdlb_reset
subroutine ddi_procdlb_next()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)')'%%%-DDI2: DDI_PROCDLB_NEXT not implemented.'
  return
end subroutine ddi_procdlb_next
SUBROUTINE ddi_group_create_custom()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_GROUP_CREATE_CUSTOM not implemented.'
  return
end SUBROUTINE ddi_group_create_custom
SUBROUTINE ddi_group_create()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)')'%%%-DDI2: DDI_GROUP_CREATE not implemented.'
  return
end SUBROUTINE ddi_group_create
SUBROUTINE ddi_gdlbnext()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)')'%%%-DDI2: DDI_GDLBNEXT not implemented.'
  return
end SUBROUTINE ddi_gdlbnext
SUBROUTINE ddi_gdlbreset()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)')'%%%-DDI2: DDI_GDLBRESET not implemented.'
  return
end SUBROUTINE ddi_gdlbreset
SUBROUTINE ddi_scope()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)')'%%%-DDI2: DDI_SCOPE not implemented.'
  return
end SUBROUTINE ddi_scope
SUBROUTINE ddi_ngroup()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)')'%%%-DDI2: DDI_NGROUP not implemented.'
  !
  return
end SUBROUTINE ddi_ngroup
!
!     DDI version 3 additions
!
SUBROUTINE ddi_masters_bcast()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_masters_bcast
!
SUBROUTINE ddi_masters_gsumf()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_masters_gsumf
!
!
SUBROUTINE ddi_ndistrib()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_ndistrib
!
SUBROUTINE ddi_timer_reset()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_timer_reset
!
SUBROUTINE ddi_smp_create()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_smp_create
!
SUBROUTINE ddi_smp_destroy()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_smp_destroy
!
SUBROUTINE ddi_smp_offset()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_smp_offset
!
SUBROUTINE ddi_smp_bcast()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_smp_bcast
!
SUBROUTINE ddi_smp_gsumf()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_smp_gsumf
!
SUBROUTINE ddi_smp_sync()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  !
  return
end SUBROUTINE ddi_smp_sync
!
SUBROUTINE ddi_smp_nproc()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_IRECV not implemented.'
  return
end SUBROUTINE ddi_smp_nproc
!
SUBROUTINE ddi_ascope()
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  WRITE(OUTU,'(A)') &
       '%%%-DDI2: DDI_ASCOPE not implemented.'
  return
end SUBROUTINE ddi_ascope
!
subroutine ddi_scatter_acc( handle, nbuff, ibuff, buff)
  use stream
  implicit none    
  integer handle, nbuff, ibuff(*)
  double precision buff(*)
  write(outu,*)'DDI to GA: DDI_SCATTER_ACC is not implemented '
  return            
end subroutine ddi_scatter_acc
!
subroutine ddi_comm_destroy( comm )
  use stream
  implicit none    
  integer comm
  write(outu,*)'DDI to GA: DDI_COMM_DESTROY is not implemented '
  return            
end subroutine ddi_comm_destroy

!
! new from 30 JUN 2020 (R1) version:

SUBROUTINE ddi_get_dsid(i)
  integer i
  return
end SUBROUTINE ddi_get_dsid
SUBROUTINE ddi_gdlbnext_inner(i)
  integer i
  return
end SUBROUTINE ddi_gdlbnext_inner

SUBROUTINE ddi_gdlbreset_inner(i)
  integer i
  return
end SUBROUTINE ddi_gdlbreset_inner


#else /**/
SUBROUTINE DDI_PBEG(NWDVAR)
  RETURN
END SUBROUTINE DDI_PBEG
SUBROUTINE DDI_PEND(ISTAT)
  RETURN
END SUBROUTINE DDI_PEND
SUBROUTINE DDI_NPROC(DDI_NP,DDI_ME)
  use chm_kinds
  integer(gms_int)  ddi_np,ddi_me
  ddi_np=1
  ddi_me=0
  RETURN
END SUBROUTINE DDI_NPROC
SUBROUTINE DDI_MEMORY(MEMREP,MEMDDI,EXETYP)
  RETURN
END SUBROUTINE DDI_MEMORY
SUBROUTINE DDI_CREATE(IDIM,JDIM,HANDLE)
  RETURN
END SUBROUTINE DDI_CREATE
SUBROUTINE DDI_DESTROY(HANDLE)
  RETURN
END SUBROUTINE DDI_DESTROY
SUBROUTINE DDI_DISTRIB(HANDLE,NODE,ILOC,IHIC,JLOC,JHIC)
  RETURN
END SUBROUTINE DDI_DISTRIB
SUBROUTINE DDI_DLBRESET
  RETURN
END SUBROUTINE DDI_DLBRESET
SUBROUTINE DDI_SYNC(SNCTAG)
  RETURN
END SUBROUTINE DDI_SYNC
SUBROUTINE DDI_GSUMF(MSGTAG,BUFF,MSGLEN)
  RETURN
END SUBROUTINE DDI_GSUMF
SUBROUTINE DDI_GSUMI(MSGTAG,BUFF,MSGLEN)
  RETURN
END SUBROUTINE DDI_GSUMI
SUBROUTINE DDI_BCAST(MSGTAG,TYPE,BUFF,LEN,FROM)
  RETURN
END SUBROUTINE DDI_BCAST
!
SUBROUTINE DDI_SEND(SNDBUF,LEN,TO)
  RETURN
END SUBROUTINE DDI_SEND
SUBROUTINE DDI_RECV(RCVBUF,LEN,TO)
  RETURN
END SUBROUTINE DDI_RECV
SUBROUTINE DDI_RCVANY(RCVBUF,LEN,TO)
  RETURN
END SUBROUTINE DDI_RCVANY
SUBROUTINE DDI_DLBNEXT(DLB_COUNTER)
  RETURN
END SUBROUTINE DDI_DLBNEXT
SUBROUTINE DDI_GET(HANDLE,ILO,IHI,JLO,JHI,BUFF)
  RETURN
END SUBROUTINE DDI_GET
SUBROUTINE DDI_PUT(HANDLE,ILO,IHI,JLO,JHI,BUFF)
  RETURN
END SUBROUTINE DDI_PUT
SUBROUTINE DDI_ACC(HANDLE,ILO,IHI,JLO,JHI,BUFF)
  RETURN
END SUBROUTINE DDI_ACC
SUBROUTINE DDI_OUTPUT(DDI_ON_OFF)
  RETURN
END SUBROUTINE DDI_OUTPUT
SUBROUTINE DDI_LEVEL(L)
  RETURN
END SUBROUTINE DDI_LEVEL
!
!     DDI version 2 additions
!
SUBROUTINE ddi_irecv()
  return
end SUBROUTINE ddi_irecv
SUBROUTINE ddi_isend()
  return
end SUBROUTINE ddi_isend
SUBROUTINE ddi_wait()
  return
end SUBROUTINE ddi_wait
SUBROUTINE ddi_procdlb_destroy()
  return
end SUBROUTINE ddi_procdlb_destroy
SUBROUTINE ddi_procdlb_create()
  return
end SUBROUTINE ddi_procdlb_create
SUBROUTINE ddi_procdlb_reset()
  return
end SUBROUTINE ddi_procdlb_reset
subroutine ddi_procdlb_next()
  return
end subroutine ddi_procdlb_next
SUBROUTINE ddi_group_create_custom()
  return
end SUBROUTINE ddi_group_create_custom
SUBROUTINE ddi_group_create()
  return
end SUBROUTINE ddi_group_create
SUBROUTINE ddi_gdlbnext()
  return
end SUBROUTINE ddi_gdlbnext
SUBROUTINE ddi_gdlbreset()
  return
end SUBROUTINE ddi_gdlbreset
SUBROUTINE ddi_scope()
  return
end SUBROUTINE ddi_scope
SUBROUTINE ddi_nnode(n,m)
  use chm_kinds
  integer(gms_int) n,m
  n=1
  m=0
  return
end SUBROUTINE ddi_nnode
SUBROUTINE ddi_ngroup()
  return
end SUBROUTINE ddi_ngroup
#endif 
#endif /* 0 for original-ddi */
#else /*  (gamess)*/
SUBROUTINE NULLDDI
  RETURN
END SUBROUTINE NULLDDI
!
#endif /* (gamess)*/
