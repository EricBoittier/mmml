#if KEY_PARALLEL==1 /*parallel*/
SUBROUTINE APSYNC()
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  !
  !     Actual Global Sync routine (sending one byte around)
  !
  use dimens_fcm
  use parallel
  !
  implicit none
  INTEGER DUMMY,IDIM,NEXT,ISYNC,IBIT,TAG,ME
  INTEGER IEOR,IAND
  !
  !     Has to send at least one byte to really block the program
  !
  TAG=1
  ME=MYNOD
  DO IDIM = 1, NODDIM
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     ISYNC=IAND(ME,IBIT)
     IF(ISYNC.EQ.IBIT) THEN
        CALL GREC(NEXT,TAG,DUMMY,1)
        CALL GSEN(NEXT,TAG,DUMMY,1)
     ELSE
        CALL GSEN(NEXT,TAG,DUMMY,1)
        CALL GREC(NEXT,TAG,DUMMY,1)
     ENDIF
     TAG=TAG+1
  ENDDO
  !
  RETURN
END SUBROUTINE APSYNC
!
SUBROUTINE AGGSYNC()
  !-----------------------------------------------------------------------
  use chm_kinds
  !
  !     Actual General Global SYNC routine (sending one byte around)
  !
  use dimens_fcm
  use parallel
  !
  implicit none
  INTEGER ITYPE,DUMMY,ISTEP,ME
  INTEGER ISYNC
  !
  ITYPE=1
  ME=MYNOD
  ISYNC=MOD(ME,2)
  DO ISTEP = 1, NUMNOD-1
     IF(ISYNC.EQ.1) THEN
        CALL GRECL(IPPMAP,ITYPE,DUMMY,1)
        CALL GSENR(IPPMAP,ITYPE,DUMMY,1)
     ELSE          
        CALL GSENR(IPPMAP,ITYPE,DUMMY,1)
        CALL GRECL(IPPMAP,ITYPE,DUMMY,1)
     ENDIF
     ITYPE=ITYPE+1
  ENDDO
  !
  RETURN
END SUBROUTINE AGGSYNC
!
SUBROUTINE GREC(FROM,TAG,BUF,LEN)
  !-----------------------------------------------------------------------
  !
  !     General RECeive routine. Its purpose is to ease replacement
  !     for various platforms (Intel, CM-5, SP1, TCP/IP, ...)
  !
  use chm_kinds
#if KEY_MPI==1
  use parallel,only:COMM_CHARMM    
#endif
#if KEY_MPI==1
  use mpi                          
#endif

  implicit none
  !
  INTEGER FROM, TAG, LEN
#if KEY_GNU==0
  integer(int_byte) buf(*)
#else /**/
  INTEGER BUF(*)
#endif 
  !
#if KEY_MPI==1
  INTEGER M_STAT(MPI_STATUS_SIZE),IERR,REQ,COUNT,size
  IF(LEN.GT.0) THEN
     CALL MPI_IRECV(BUF,LEN,MPI_BYTE,FROM,TAG,COMM_CHARMM,REQ, &
          IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GREC>','MPI IRECV ERROR IN GREC')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GREC>','MPI WAIT ERROR IN GREC')
  ENDIF
  !      CALL DDI_RECV(BUF,(LEN+7)/8,FROM)
! for DDI:  CALL DDI_RECV(BUF,LEN/8,FROM)
  !
#endif 
  !
  RETURN
END SUBROUTINE GREC
!
SUBROUTINE GRECMAP(FROM,TAG,BUF,LEN)
  !-----------------------------------------------------------------------
  !
  !     General RECeive routine. Its purpose is to ease replacement
  !     for various platforms (Intel, CM-5, SP1, TCP/IP, ...)
  !     This one maps the processor ID according to IPPMAP array
  !
  use chm_kinds
  use parallel
#if KEY_MPI==1
  use mpi                          
#endif
  !
  implicit none
  INTEGER FROM, TAG, LEN
#if KEY_GNU==0
  BYTE BUF(*)
#else /**/
  INTEGER BUF(*)
#endif 
  !
#if KEY_MPI==1
  INTEGER M_STAT(MPI_STATUS_SIZE),IERR,REQ,COUNT,size
  IF(LEN.GT.0) THEN
     CALL MPI_IRECV(BUF,LEN,MPI_BYTE,IPPMAP(FROM),TAG, &
          COMM_CHARMM,REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GREC>','MPI IRECV ERROR IN GREC')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GREC>','MPI WAIT ERROR IN GREC')
  ENDIF
  !      CALL DDI_RECV(BUF,(LEN+7)/8,IPPMAP(FROM))
! for DDI:  CALL DDI_RECV(BUF,LEN/8,IPPMAP(FROM))
  !
#endif 
  !
  RETURN
END SUBROUTINE GRECMAP
!
SUBROUTINE GRECL(IPPMAP,TAG,BUF,LEN)
  !-----------------------------------------------------------------------
  !
  !     RECeive from Left routine. Used in general communication routines
  !
  use chm_kinds
  use exfunc
  use parallel,only:lnnod,lmnod
  !
  implicit none
  INTEGER TAG, LEN, IPPMAP(0:*)
#if KEY_GNU==0
  BYTE BUF(*)
#else /**/
  INTEGER BUF(*)
#endif 
  INTEGER ME,NUMNOD
  ME=LMNOD()
  NUMNOD=LNNOD()
  CALL GREC(IPPMAP(MOD(ME+NUMNOD-1,NUMNOD)),TAG,BUF,LEN)
  RETURN
END SUBROUTINE GRECL
!
SUBROUTINE GRECR(IPPMAP,TAG,BUF,LEN)
  !-----------------------------------------------------------------------
  !
  !     RECeive from Right routine. Used in general communication routines
  !
  use chm_kinds
  use exfunc
  use parallel,only:lnnod,lmnod
  !
  implicit none
  INTEGER TAG, LEN, IPPMAP(0:*)
#if KEY_GNU==0
  BYTE BUF(*)
#else /**/
  INTEGER BUF(*)
#endif 
  INTEGER NUMNOD,ME
  ME=LMNOD()
  NUMNOD=LNNOD()
  CALL GREC(IPPMAP(MOD(ME+1,NUMNOD)),TAG,BUF,LEN)
  RETURN
END SUBROUTINE GRECR
!
SUBROUTINE GSEN(TO,TAG,BUF,LEN)
  !-----------------------------------------------------------------------
  !
  !     General SENd routine. Its purpose is to ease replacement
  !     for various platforms (Intel, CM-5, SP1, TCP/IP, ...)
  !
  use chm_kinds
#if KEY_MPI==1
  use parallel,only:COMM_CHARMM    
#endif
#if KEY_MPI==1
  use mpi                          
#endif
  implicit none
  !
  INTEGER TO, TAG, LEN
#if KEY_GNU==0
  BYTE BUF(*)
#else /**/
  INTEGER BUF(*)
#endif 
  !
#if KEY_MPI==1
  INTEGER IERR,REQ,M_STAT(MPI_STATUS_SIZE)
  IF(LEN.GT.0) THEN
     CALL MPI_ISEND(BUF,LEN,MPI_BYTE,TO,TAG,COMM_CHARMM,REQ, &
          IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GSEN>','MPI ISEND ERROR IN GSEN')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GSEN>','MPI WAIT ERROR IN GSEN')
  ENDIF
  !
  !      CALL DDI_SEND(BUF,(LEN+7)/8,TO)
! for DDI:  CALL DDI_SEND(BUF,LEN/8,TO)
  !
#endif 
  !
  RETURN
END SUBROUTINE GSEN
!
SUBROUTINE GSENMAP(TO,TAG,BUF,LEN)
  !-----------------------------------------------------------------------
  !
  !     General SENd routine. Its purpose is to ease replacement
  !     for various platforms (Intel, CM-5, SP1, TCP/IP, ...)
  !     This one maps the processor ID according to IPPMAP array
  !
  use chm_kinds
  use parallel
#if KEY_MPI==1
  use mpi                          
#endif
  !
  implicit none
  INTEGER TO, TAG, LEN
#if KEY_GNU==0
  BYTE BUF(*)
#else /**/
  INTEGER BUF(*)
#endif 
  !
#if KEY_MPI==1
  INTEGER IERR,REQ,M_STAT(MPI_STATUS_SIZE)
  IF(LEN.GT.0) THEN
     CALL MPI_ISEND(BUF,LEN,MPI_BYTE,IPPMAP(TO),TAG, &
          COMM_CHARMM,REQ,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GSEN>','MPI ISEND ERROR IN GSEN')
     CALL MPI_WAIT(REQ,M_STAT,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GSEN>','MPI WAIT ERROR IN GSEN')
  ENDIF
  !
  !      CALL DDI_SEND(BUF,(LEN+7)/8,IPPMAP(TO))
! for DDI:  CALL DDI_SEND(BUF,LEN/8,IPPMAP(TO))
  !
#endif 
  !
  RETURN
END SUBROUTINE GSENMAP
!
SUBROUTINE GSENL(IPPMAP,TAG,BUF,LEN)
  !-----------------------------------------------------------------------
  !
  !     SENd to Left routine. Used in general communication routines
  !
  use chm_kinds
  use exfunc
  use parallel,only:lnnod,lmnod
  !
  implicit none
  INTEGER TAG, LEN, IPPMAP(0:*)
#if KEY_GNU==0
  BYTE BUF(*)
#else /**/
  INTEGER BUF(*)
#endif 
  INTEGER ME,NUMNOD
  ME=LMNOD()
  NUMNOD=LNNOD()
  CALL GSEN(IPPMAP(MOD(ME+NUMNOD-1,NUMNOD)),TAG,BUF,LEN)
  RETURN
END SUBROUTINE GSENL
!
SUBROUTINE GSENR(IPPMAP,TAG,BUF,LEN)
  !-----------------------------------------------------------------------
  !
  !     SENd to Right routine. Used in general communication routines
  !
  use chm_kinds
  use exfunc
  use parallel,only:lnnod,lmnod
  !
  implicit none
  INTEGER TAG, LEN, IPPMAP(0:*)
#if KEY_GNU==0
  BYTE BUF(*)
#else /**/
  INTEGER BUF(*)
#endif 
  INTEGER NUMNOD,ME
  ME=LMNOD()
  NUMNOD=LNNOD()
  CALL GSEN(IPPMAP(MOD(ME+1,NUMNOD)),TAG,BUF,LEN)
  RETURN
END SUBROUTINE GSENR
!
!
SUBROUTINE COPYARR4TO8(LEN,BUF4,BUF)
  use chm_kinds
  implicit none
  INTEGER LEN,I
  real(chm_real) BUF(*)
  REAL(chm_real4) BUF4(*)
  DO I=1,LEN
     BUF(I)=BUF4(I)
  ENDDO
  RETURN
END SUBROUTINE COPYARR4TO8
!
SUBROUTINE COPYARR8TO4(LEN,BUF,BUF4)
  use chm_kinds
  implicit none
  INTEGER LEN,I
  real(chm_real) BUF(*)
  REAL(chm_real4) BUF4(*)
  DO I=1,LEN
     BUF4(I)=BUF(I)
  ENDDO
  RETURN
END SUBROUTINE COPYARR8TO4
!
SUBROUTINE GRECSEN(TO,TAG,RBUF,RLEN,SBUF,SLEN)
  !-----------------------------------------------------------------------
  !
  !     General RECeive and SENd routine. Its purpose is to ease replacement
  !     for various platforms (Intel, CM-5, SP1, TCP/IP, ...)
  !
  use chm_kinds
  use exfunc
  use memory
#if KEY_MPI==1
  use parallel,only:COMM_CHARMM    
  use mpi                          
#endif

  implicit none
  !
  real(chm_real4),allocatable,dimension(:) :: R4BUF
  real(chm_real4),allocatable,dimension(:) :: S4BUF
  INTEGER TO, TAG, RLEN, SLEN
  real(chm_real) RBUF(*),SBUF(*)
  !
#if KEY_MPI==1
  INTEGER I
  INTEGER REQ(2),STATUS_ARRAY(MPI_STATUS_SIZE,2),IERR, &
       MTO,MTYPE,MRLEN,MSLEN,WTALL,MDP,MCOMM
  !
  MTO=TO
  MTYPE=ABS(TAG)
  MRLEN=RLEN
  MSLEN=SLEN
  WTALL=2
  MDP = MPI_DOUBLE_PRECISION
  MCOMM = COMM_CHARMM
  !     START: try single precision communication
  IF(TAG.LT.0)THEN
     MDP = MPI_REAL
     call chmalloc('paral3.src','GRECSEN','S4BUF',SLEN,cr4=S4BUF)
     call chmalloc('paral3.src','GRECSEN','R4BUF',RLEN,cr4=R4BUF)
     !C         write(*,*)
     !C     $        'GRECSEN>Using single prec comm:slen,rlen=',mslen,mrlen
     CALL COPYARR8TO4(SLEN,SBUF,S4BUF)
     CALL COPYARR8TO4(RLEN,RBUF,R4BUF)
     CALL MPI_ISEND(S4BUF,MSLEN,MDP,MTO,MTYPE,MCOMM,REQ(1),IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GRECSEN>','MPI ISEND ERROR')
     CALL MPI_IRECV(R4BUF,MRLEN,MDP,MTO,MTYPE,MCOMM,REQ(2),IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GRECSEN>','MPI IRECV ERROR')
     CALL MPI_WAITALL(WTALL,REQ,STATUS_ARRAY,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GRECSEN>','MPI WAITALL ERROR')
     CALL COPYARR4TO8(SLEN,S4BUF,SBUF)
     CALL COPYARR4TO8(RLEN,R4BUF,RBUF)
     call chmdealloc('paral3.src','GRECSEN','S4BUF',SLEN,cr4=S4BUF)
     call chmdealloc('paral3.src','GRECSEN','R4BUF',RLEN,cr4=R4BUF)
  ELSE
     !     END: try  single precision communication
     CALL MPI_ISEND(SBUF,MSLEN,MDP,MTO,MTYPE,MCOMM,REQ(1),IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GRECSEN>','MPI ISEND ERROR')
     CALL MPI_IRECV(RBUF,MRLEN,MDP,MTO,MTYPE,MCOMM,REQ(2),IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GRECSEN>','MPI IRECV ERROR')
     CALL MPI_WAITALL(WTALL,REQ,STATUS_ARRAY,IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<GRECSEN>','MPI WAITALL ERROR')
  ENDIF
  !
  !
! for DDI:  CALL WRNDIE(-5,'<GRECSEN>','Asynch. sockets not supported yet.')
#endif 
  !
  RETURN
END SUBROUTINE GRECSEN
!
SUBROUTINE GRECSENR(ME,IPPMAP,NUMNOD,ITYPE,W,JLEN,X,ILEN)
  !-----------------------------------------------------------------------
  !
  !     General RECeive and SENd routine. Its purpose is to ease replacement
  !     for various platforms (Intel, CM-5, SP1, TCP/IP, ...)
  !
  use chm_kinds
#if KEY_MPI==1
  use parallel,only:COMM_CHARMM    
#endif
#if KEY_MPI==1
  use mpi                          
#endif

  implicit none
  !
  INTEGER ME,IPPMAP(0:*),NUMNOD,ITYPE,ILEN,JLEN
  real(chm_real) W(*),X(*)
  !
  !      
  INTEGER LEFT,RIGHT
  !
#if KEY_MPI==1
  !
  INTEGER ITMP,REQ(2),STATUS_ARRAY(MPI_STATUS_SIZE,2),IERR, &
       MILEN,MITYPE,MJLEN,WTALL,MDP,MCOMM
  !

  MDP = MPI_DOUBLE_PRECISION
  MCOMM = COMM_CHARMM
  MILEN=ILEN
  MJLEN=JLEN
  MITYPE=ITYPE

  LEFT=IPPMAP(MOD(ME+NUMNOD-1,NUMNOD))
  RIGHT=IPPMAP(MOD(ME+1,NUMNOD))
  !
  CALL MPI_IRECV(W,MJLEN,MDP,LEFT,MITYPE, &
       MCOMM,REQ(1),IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<GRECSENR>','MPI IRECV ERROR')
  CALL MPI_ISEND(X,MILEN,MDP,RIGHT,MITYPE, &
       MCOMM,REQ(2),IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<GRECSENR>','MPI ISEND ERROR')
  WTALL=2
  CALL MPI_WAITALL(WTALL,REQ,STATUS_ARRAY,IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<GRECSENR>','MPI WAITALL ERROR')
  !
#else /**/
  CALL WRNDIE(-5,'<GRECSENR>','Async. ring not supported yet.')
#endif 

  RETURN
END SUBROUTINE GRECSENR
!
SUBROUTINE SWAP1D(N,XL,XR,PL,PR)
  !-----------------------------------------------------------------------
  !
  !     One Direction receive & send routine:
  !     receives XL, from PL & sends XR to PR
  !     NOTE: Size of XL,XR is the same!
  !           If PL<0 ignore receive
  !           If PR<0 ignore send
  !
  use chm_kinds
  !
  use stream
  !
#if KEY_MPI==1 && KEY_CMPI==1
  use parallel,only:COMM_CHARMM    
#endif
#if KEY_MPI==1 && KEY_CMPI==1
  use mpi                          
#endif
  implicit none
  INTEGER N,PL,PR
  real(chm_real) XL(*),XR(*)
  INTEGER TAG
  !
#if KEY_CMPI==1
#if KEY_MPI==1
  INTEGER REQ(2),STATUS_ARRAY(MPI_STATUS_SIZE,2),IERR,WTALL, &
       MN,MPL,MPR,MDP,MCOMM
  !
  IF ( (PL.LT.0) .AND. (PR.LT.0) ) RETURN
  MN=N
  MPL=PL
  MPR=PR
  TAG=1
  WTALL=0
  MCOMM=COMM_CHARMM
  MDP=MPI_DOUBLE_PRECISION
  IF(PR.GE.0)THEN
     WTALL=WTALL+1
     CALL MPI_ISEND(XR,MN,MDP,MPR,TAG,MCOMM,REQ(WTALL),IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SWAP1D>','MPI ISEND ERROR')
  ENDIF
  IF(PL.GE.0)THEN
     WTALL=WTALL+1
     CALL MPI_IRECV(XL,MN,MDP,MPL,TAG,MCOMM,REQ(WTALL),IERR)
     IF(IERR.NE.MPI_SUCCESS)  &
          CALL WRNDIE(-4,'<SWAP1D>','MPI IRECV ERROR')
  ENDIF
  CALL MPI_WAITALL(WTALL,REQ,STATUS_ARRAY,IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<SWAP1D>','MPI WAITALL ERROR')
  !
#else /**/
  CALL WRNDIE(-5,'<SWAP1D>','Not compiled with MPI.')
#endif 
#else /**/
  CALL WRNDIE(-5,'<SWAP1D>','Not compiled with CMPI.')
#endif 
  !
  RETURN
END SUBROUTINE SWAP1D
!
SUBROUTINE SWAPD3(N,X,XL,XR,PL,PR)
  !-----------------------------------------------------------------------
  !
  !     Double receive & send routine. Works in 2 direction concurrently
  !
  use chm_kinds
  use stream
#if KEY_MPI==1 && KEY_CMPI==1
  use parallel,only:COMM_CHARMM   
#endif
#if KEY_MPI==1 && KEY_CMPI==1
  use mpi                         
#endif

  implicit none
  INTEGER N,PL,PR
  real(chm_real) X(*),XL(*),XR(*)
  INTEGER TAG
  !
#if KEY_CMPI==1
#if KEY_MPI==1
  INTEGER REQ(4),STATUS_ARRAY(MPI_STATUS_SIZE,4),IERR,WTALL, &
       MN,MPL,MPR,MDP,MCOMM
  !
  MN=N
  MPL=PL
  MPR=PR
  TAG=1
  MCOMM=COMM_CHARMM
  MDP=MPI_DOUBLE_PRECISION
  CALL MPI_ISEND(X,MN,MDP,MPL,TAG, &
       MCOMM,REQ(1),IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<GRECSEN>','MPI ISEND ERROR')
  CALL MPI_IRECV(XL,MN,MDP,MPL,TAG, &
       MCOMM,REQ(2),IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<GRECSEN>','MPI IRECV ERROR')
  CALL MPI_ISEND(X,MN,MDP,MPR,TAG, &
       MCOMM,REQ(3),IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<GRECSEN>','MPI ISEND ERROR')
  CALL MPI_IRECV(XR,MN,MDP,MPR,TAG, &
       MCOMM,REQ(4),IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<GRECSEN>','MPI IRECV ERROR')
  WTALL=4
  CALL MPI_WAITALL(WTALL,REQ,STATUS_ARRAY,IERR)
  IF(IERR.NE.MPI_SUCCESS)  &
       CALL WRNDIE(-4,'<GRECSEN>','MPI WAITALL ERROR')
  !
#else /**/
  CALL WRNDIE(-5,'<SWAPD3>','Not compiled with MPI.')
#endif 
#else /**/
  CALL WRNDIE(-5,'<SWAPD3>','Not compiled with CMPI.')
#endif 
  !
  RETURN
END SUBROUTINE SWAPD3
!
SUBROUTINE NXYMSH(NX,NY)
  !-----------------------------------------------------------------------
  use chm_kinds
  use parallel,only:nptwo
  implicit none
  !
  !     Returns number of rows and columns in mesh. Valid for 2**N
  !
  INTEGER NX, NY, IDIM

  IDIM=NPTWO()
  NX=2**(IDIM/2)
  NY=2**(IDIM-IDIM/2)
  RETURN
END SUBROUTINE NXYMSH
!
SUBROUTINE MESH(MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use parallel,only:nnod,mnod
  use machutil,only:die
  implicit none
  !
  !     Remaps cube network node numbers to mesh network
  !
  INTEGER  MYNOD,IPPMAP(0:*)
  INTEGER I,NODDIM,NUMNOD
  !
  INTEGER NX,NY
  INTEGER DIMX,DIMY,IBASE,STOD,STEV
  INTEGER INOD,J,K,IBIT,BIT(0:15),BWT(0:15)
  !
  CALL NXYMSH(NX,NY)
  NUMNOD=NX*NY
  IF(NUMNOD.NE.NNOD()) CALL DIE
  !
  DIMX=-1
  DIMY=-1
  DO I = 0,16
     IF(2**I.EQ.NX) DIMX=I
     IF(2**I.EQ.NY) DIMY=I
  ENDDO
  IF(DIMX.LT.0) CALL DIE
  IF(DIMY.LT.0) CALL DIE
  NODDIM=DIMX+DIMY
  !
  STEV=0
  STOD=-DIMX
  !
  DO I = DIMX+DIMY,15
     BWT(I)=0
  ENDDO
  !
  DO I = DIMX+DIMY-1,0,-1
     IF(STOD.LT.0) THEN
        BWT(I)=STEV
        STEV=STEV+1
        IF(STOD.GT.-DIMX-DIMY) STOD=-STOD
     ELSE
        BWT(I)=STOD
        STOD=STOD+1
        IF(STEV.LT.DIMX) STOD=-STOD
     ENDIF
  ENDDO
  !
  !C      WRITE(6,47) (BWT(I),I=13,0,-1)
  !C  47  FORMAT(' BTW:  ',16I5)
  !
  DO I = 0,DIMX+DIMY-1
     BWT(I)=2**BWT(I)
  ENDDO
  !
  !C      WRITE(6,47) (BWT(I),I=13,0,-1)
  !
  MYNOD=-1
  DO INOD = 0,NUMNOD-1
     DO IBIT = 0,15
        IBASE=2**IBIT
        BIT(IBIT)=(MOD(INOD,IBASE*2)-MOD(INOD,IBASE))/IBASE
     ENDDO
     !C         WRITE(6,48) INOD,(BIT(J),J=15,0,-1)
     !C  48     FORMAT(' INOD,BIT: ',I5,5X,16I2)
     J=0
     DO K = 0,15
        J=J+BWT(K)*BIT(K)
     ENDDO
     !C         WRITE(6,49) INOD,J
     !C  49     FORMAT(' INOD,J:  ',2I8)
     IPPMAP(INOD)=J
     IF(J.EQ.MNOD()) MYNOD=INOD
  ENDDO
  IF(MYNOD.LT.0) CALL DIE
  !
  RETURN
END SUBROUTINE MESH
!
SUBROUTINE AGVDGB(X,IPARPT,NUMNOD,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     This routine performs Actual General Vector Distributed Global Broadcast
  !     using communication between neighbours only
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and like.
  !
  !     X      - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NUMNOD - number of nodes [input]
  !     MYNOD  - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !
  !     It is based on ideas from Robert van de Geijn's
  !     papers and is optimal in the amount of communication to be performed
  !     but needs NUMNOD-1 startups (latency!)
  !
  real(chm_real) X(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NUMNOD
  INTEGER ITYPE,IBIT,ISTEP,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,ISYNC
  !
  ITYPE=1
  ME=MYNOD
  ISYNC=MOD(ME,2)
  DO ISTEP = 1, NUMNOD-1
     IBIT=NUMNOD-ISTEP+ME
     IOFF=MOD(IBIT+1,NUMNOD)
     JOFF=MOD(IBIT,NUMNOD)
     ILEN=IPARPT(IOFF+1)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+1)-IPARPT(JOFF)
     IF(ISYNC.EQ.1) THEN
        CALL GRECL(IPPMAP,ITYPE,X(IPARPT(JOFF)+1),8*JLEN)
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),8*ILEN)
     ELSE          
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),8*ILEN)
        CALL GRECL(IPPMAP,ITYPE,X(IPARPT(JOFF)+1),8*JLEN)
     ENDIF
     ITYPE=ITYPE+1
  ENDDO
  !
  RETURN
END SUBROUTINE AGVDGB
!
SUBROUTINE AGVDGBA(X,IPARPT,NUMNOD,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     This routine performs:
  !     Actual General Vector Distributed Global Broadcast Asynchronous
  !     using communication between neighbours only
  !
  !     This is asynchronized version
  !
  !     X      - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NUMNOD - number of nodes [input]
  !     MYNOD  - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !
  !     It is based on ideas from Robert van de Geijn's
  !     papers and is optimal in the amount of communication to be performed
  !     but needs NUMNOD-1 startups (latency!)
  !
  real(chm_real) X(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NUMNOD
  INTEGER ITYPE,IBIT,ISTEP,ME,IOFF,JOFF
  INTEGER ILEN,JLEN
  !
  ITYPE=1
  ME=MYNOD
  DO ISTEP = 1, NUMNOD-1
     IBIT=NUMNOD-ISTEP+ME
     IOFF=MOD(IBIT+1,NUMNOD)
     JOFF=MOD(IBIT,NUMNOD)
     ILEN=IPARPT(IOFF+1)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+1)-IPARPT(JOFF)
     CALL GRECSENR(ME,IPPMAP,NUMNOD,ITYPE,X(IPARPT(JOFF)+1),JLEN, &
          X(IPARPT(IOFF)+1),ILEN)
     ITYPE=ITYPE+1
  ENDDO
  !
  RETURN
END SUBROUTINE AGVDGBA
!
SUBROUTINE IGVDGB(X,IPARPT,NUMNOD,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     This routine performs Integer General Vector Distributed Global Broadcast
  !     using communication between neighbours only. 
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and like.
  !
  !     X      - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NUMNOD - number of nodes [input]
  !     MYNOD  - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !
  !     It is based on ideas from Robert van de Geijn's
  !     papers and is optimal in the amount of communication to be performed
  !     but needs NUMNOD-1 startups (latency!)
  !
  INTEGER X(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NUMNOD
  INTEGER ITYPE,IBIT,ISTEP,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,ISYNC
  !
  ITYPE=1
  ME=MYNOD
  ISYNC=MOD(ME,2)
  DO ISTEP = 1, NUMNOD-1
     IBIT=NUMNOD-ISTEP+ME
     IOFF=MOD(IBIT+1,NUMNOD)
     JOFF=MOD(IBIT,NUMNOD)
     ILEN=IPARPT(IOFF+1)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+1)-IPARPT(JOFF)
     IF(ISYNC.EQ.1) THEN
        CALL GRECL(IPPMAP,ITYPE,X(IPARPT(JOFF)+1),4*JLEN)
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
     ELSE          
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
        CALL GRECL(IPPMAP,ITYPE,X(IPARPT(JOFF)+1),4*JLEN)
     ENDIF
     ITYPE=ITYPE+1
  ENDDO
  !
  RETURN
END SUBROUTINE IGVDGB
!
SUBROUTINE AGVDGS(X,IPARPT,W,NUMNOD,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     This routine performs General Actual Vector Distributed Global Sum
  !     using communication between neighbours only
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and others.
  !
  !     X      - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NUMNOD - number of nodes [input]
  !     MYNOD  - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from Robert van de Geijn's
  !     papers and is optimal in the amount of communication to be performed
  !     but needs NUMNOD-1 startups (latency!)
  !
  real(chm_real) :: X(*), W(*), ONE=1.0D0
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NUMNOD
  INTEGER ITYPE,IBIT,ISTEP,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,ISYNC
  !
  ITYPE=1
  ME=MYNOD
  ISYNC=MOD(ME,2)
  DO ISTEP = 1, NUMNOD-1
     IBIT=NUMNOD-ISTEP+ME
     IOFF=MOD(IBIT,NUMNOD)
     JOFF=MOD(IBIT-1,NUMNOD)
     ILEN=IPARPT(IOFF+1)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+1)-IPARPT(JOFF)
     IF(ISYNC.EQ.1) THEN
        CALL GRECL(IPPMAP,ITYPE,W(IPARPT(JOFF)+1),8*JLEN)
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),8*ILEN)
     ELSE
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),8*ILEN)
        CALL GRECL(IPPMAP,ITYPE,W(IPARPT(JOFF)+1),8*JLEN)
     ENDIF
     ITYPE=ITYPE+1
     CALL DAXPY_s(JLEN,ONE,W(IPARPT(JOFF)+1),1,X(IPARPT(JOFF)+1),1)
  ENDDO
  RETURN
END SUBROUTINE AGVDGS
!
SUBROUTINE AGVDGSA(X,IPARPT,W,NUMNOD,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     This routine performs:
  !     Actual General Vector Distributed Global Sum Asynchronous
  !     using communication between neighbours only
  !
  !     This is asynchronous version i.e. it can communicate in both
  !     directions at once.
  !
  !     X      - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NUMNOD - number of nodes [input]
  !     MYNOD  - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from Robert van de Geijn's
  !     papers and is optimal in the amount of communication to be performed
  !     but needs NUMNOD-1 startups (latency!)
  !
  real(chm_real) :: X(*), W(*), ONE=1.0D0
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NUMNOD
  INTEGER ITYPE,IBIT,ISTEP,ME,IOFF,JOFF
  INTEGER ILEN,JLEN
  !
  ITYPE=1
  ME=MYNOD
  DO ISTEP = 1, NUMNOD-1
     IBIT=NUMNOD-ISTEP+ME
     IOFF=MOD(IBIT,NUMNOD)
     JOFF=MOD(IBIT-1,NUMNOD)
     ILEN=IPARPT(IOFF+1)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+1)-IPARPT(JOFF)
     CALL GRECSENR(ME,IPPMAP,NUMNOD,ITYPE,W(IPARPT(JOFF)+1),JLEN, &
          X(IPARPT(IOFF)+1),ILEN)
     ITYPE=ITYPE+1
     CALL DAXPY_s(JLEN,ONE,W(IPARPT(JOFF)+1),1,X(IPARPT(JOFF)+1),1)
  ENDDO
  RETURN
END SUBROUTINE AGVDGSA
!
SUBROUTINE IGVDGO(X,IPARPT,W,NUMNOD,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     This routine performs Integer General Actual Vector Distributed Global Or
  !     using communication between neighbours only
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and others.
  !
  !     X      - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NUMNOD - number of nodes [input]
  !     MYNOD  - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from Robert van de Geijn's
  !     papers and is optimal in the amount of communication to be performed
  !     but needs NUMNOD-1 startups (latency!)
  !
  INTEGER X(*), W(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NUMNOD
  INTEGER ITYPE,IBIT,ISTEP,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,ISYNC
  !
  ITYPE=1
  ME=MYNOD
  ISYNC=MOD(ME,2)
  DO ISTEP = 1, NUMNOD-1
     IBIT=NUMNOD-ISTEP+ME
     IOFF=MOD(IBIT,NUMNOD)
     JOFF=MOD(IBIT-1,NUMNOD)
     ILEN=IPARPT(IOFF+1)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+1)-IPARPT(JOFF)
     IF(ISYNC.EQ.1) THEN
        CALL GRECL(IPPMAP,ITYPE,W(IPARPT(JOFF)+1),4*JLEN)
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
     ELSE          
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
        CALL GRECL(IPPMAP,ITYPE,W(IPARPT(JOFF)+1),4*JLEN)
     ENDIF
     ITYPE=ITYPE+1
     CALL IARROR(JLEN,W(IPARPT(JOFF)+1),X(IPARPT(JOFF)+1))
  ENDDO
  RETURN
END SUBROUTINE IGVDGO
!
SUBROUTINE IARROR(N,W,X)
  !-----------------------------------------------------------------------
  !
  !     Integer ARRay OR routine
  !
  INTEGER I, N
  INTEGER W(*), X(*)
  DO I = 1, N
     X(I) = IOR(X(I), W(I))
  ENDDO
  RETURN
END SUBROUTINE IARROR
!
SUBROUTINE IGVDGS(X,IPARPT,W,NUMNOD,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     This routine performs Integer General Actual Vector Distributed Global Sum
  !     using communication between neighbours only
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and others.
  !
  !     X      - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NUMNOD - number of nodes [input]
  !     MYNOD  - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from Robert van de Geijn's
  !     papers and is optimal in the amount of communication to be performed
  !     but needs NUMNOD-1 startups (latency!)
  !
  INTEGER X(*), W(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NUMNOD
  INTEGER ITYPE,IBIT,ISTEP,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,ISYNC
  !
  ITYPE=1
  ME=MYNOD
  ISYNC=MOD(ME,2)
  DO ISTEP = 1, NUMNOD-1
     IBIT=NUMNOD-ISTEP+ME
     IOFF=MOD(IBIT,NUMNOD)
     JOFF=MOD(IBIT-1,NUMNOD)
     ILEN=IPARPT(IOFF+1)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+1)-IPARPT(JOFF)
     IF(ISYNC.EQ.1) THEN
        CALL GRECL(IPPMAP,ITYPE,W(IPARPT(JOFF)+1),4*JLEN)
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
     ELSE          
        CALL GSENR(IPPMAP,ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
        CALL GRECL(IPPMAP,ITYPE,W(IPARPT(JOFF)+1),4*JLEN)
     ENDIF
     ITYPE=ITYPE+1
     CALL IARRSUM(JLEN,W(IPARPT(JOFF)+1),X(IPARPT(JOFF)+1))
  ENDDO
  RETURN
END SUBROUTINE IGVDGS
!
SUBROUTINE IARRSUM(N,W,X)
  !-----------------------------------------------------------------------
  !
  !     Integer ARRay SUM routine
  !
  INTEGER I, N
  INTEGER W(*), X(*)
  DO I = 1, N
     X(I) = X(I)+W(I)
  ENDDO
  RETURN
END SUBROUTINE IARRSUM
!
SUBROUTINE AVDGBR(X,IPARPT,NDIM,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global BRoadcast
  !     using recursive doubling scheme.
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and like.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  real(chm_real) X(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT,ISYNC
  INTEGER IEOR,IAND
  !
  ITYPE=1
  ME=MYNOD
  DO IDIM = 1, NDIM
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     ISYNC=IAND(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     IF(ISYNC.EQ.IBIT) THEN
        CALL GREC(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),8*ILEN)
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(JOFF)+1),8*JLEN)
     ELSE
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(JOFF)+1),8*JLEN)
        CALL GREC(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),8*ILEN)
     ENDIF
     ITYPE=ITYPE+1
  ENDDO
  !
  ! Problems with Intel communication
  !
  RETURN
END SUBROUTINE AVDGBR
!
SUBROUTINE IVDGBR(X,IPARPT,NDIM,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none

  !
  !     This routine performs Integer Vector Distributed Global BRoadcast
  !     using recursive doubling scheme.
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and like.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  INTEGER X(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT,ISYNC
  INTEGER IEOR,IAND
  !
  ITYPE=1
  ME=MYNOD
  DO IDIM = 1, NDIM
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     ISYNC=IAND(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     IF(ISYNC.EQ.IBIT) THEN
        CALL GREC(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(JOFF)+1),4*JLEN)
     ELSE
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(JOFF)+1),4*JLEN)
        CALL GREC(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
     ENDIF
     ITYPE=ITYPE+1
  ENDDO
  !
  ! Problems with Intel communication
  !
  RETURN
END SUBROUTINE IVDGBR
!
!
SUBROUTINE AVDGBRA(X,IPARPT,NDIM,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global BRoadcast
  !     using recursive doubling scheme.
  !
  !     This is asynchronous version i.e. it can overlap send and receive
  !     operations for maximum speed.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  real(chm_real) X(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT
  INTEGER IEOR,IAND
  !
  ITYPE=1
  ME=MYNOD
  DO IDIM = 1, NDIM
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     CALL GRECSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),ILEN, &
          X(IPARPT(JOFF)+1),JLEN)
     ITYPE=ITYPE+1
  ENDDO
  !
  ! Problems with Intel communication
  !
  RETURN
END SUBROUTINE AVDGBRA
!
SUBROUTINE AVDGSUM(X,IPARPT,W,NDIM,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global SUM
  !     using recursive halving scheme.
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and others.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  real(chm_real) :: X(*), W(*), ONE=1.0D0
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT,ISYNC
  INTEGER IEOR,IAND
  !
  ITYPE=1
  ME=MYNOD
  DO IDIM = NDIM, 1, -1
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     ISYNC=IAND(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     IF(ISYNC.EQ.IBIT) THEN
        CALL GREC(IPPMAP(NEXT),ITYPE,W(IPARPT(JOFF)+1),8*JLEN)
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),8*ILEN)
     ELSE
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),8*ILEN)
        CALL GREC(IPPMAP(NEXT),ITYPE,W(IPARPT(JOFF)+1),8*JLEN)
     ENDIF
     ITYPE=ITYPE+1
     CALL DAXPY_s(JLEN,ONE,W(IPARPT(JOFF)+1),1,X(IPARPT(JOFF)+1),1)
  ENDDO
  !
  ! Problems with Intel communication
  !
  RETURN
END SUBROUTINE AVDGSUM
!
SUBROUTINE IVDGOR(X,IPARPT,W,NDIM,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Integer Vector Distributed Global OR
  !     using recursive halving scheme.
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and others.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  INTEGER X(*), W(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT,ISYNC
  INTEGER IEOR,IAND
  !
  !
  ITYPE=1
  ME=MYNOD
  DO IDIM = NDIM, 1, -1
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     ISYNC=IAND(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     IF(ISYNC.EQ.IBIT) THEN
        CALL GREC(IPPMAP(NEXT),ITYPE,W(IPARPT(JOFF)+1),4*JLEN)
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
     ELSE
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
        CALL GREC(IPPMAP(NEXT),ITYPE,W(IPARPT(JOFF)+1),4*JLEN)
     ENDIF
     ITYPE=ITYPE+1
     CALL IARROR(JLEN,W(IPARPT(JOFF)+1),X(IPARPT(JOFF)+1))
  ENDDO
  !
  ! Problems with Intel communication
  !
  RETURN
END SUBROUTINE IVDGOR
!
SUBROUTINE IVDGSUM(X,IPARPT,W,NDIM,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Integer Vector Distributed Global SUM
  !     using recursive halving scheme.
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and others.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  INTEGER X(*), W(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT,ISYNC
  INTEGER IEOR,IAND
  !
  !
  ITYPE=1
  ME=MYNOD
  DO IDIM = NDIM, 1, -1
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     ISYNC=IAND(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     IF(ISYNC.EQ.IBIT) THEN
        CALL GREC(IPPMAP(NEXT),ITYPE,W(IPARPT(JOFF)+1),4*JLEN)
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
     ELSE
        CALL GSEN(IPPMAP(NEXT),ITYPE,X(IPARPT(IOFF)+1),4*ILEN)
        CALL GREC(IPPMAP(NEXT),ITYPE,W(IPARPT(JOFF)+1),4*JLEN)
     ENDIF
     ITYPE=ITYPE+1
     CALL IARRSUM(JLEN,W(IPARPT(JOFF)+1),X(IPARPT(JOFF)+1))
  ENDDO
  !
  RETURN
END SUBROUTINE IVDGSUM
!
SUBROUTINE AVDGSA(X,IPARPT,W,NDIM,MYNOD,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global SUM
  !     using recursive halving scheme.
  !
  !     This is asynchronous version for concurent send/recieve
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  real(chm_real) :: X(*), W(*), ONE=1.0D0
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT
  INTEGER IEOR,IAND
  !
  ITYPE=1
  ME=MYNOD
  DO IDIM = NDIM, 1, -1
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     CALL GRECSEN(IPPMAP(NEXT),ITYPE,W(IPARPT(JOFF)+1),JLEN, &
          X(IPARPT(IOFF)+1),ILEN)
     ITYPE=ITYPE+1
     CALL DAXPY_s(JLEN,ONE,W(IPARPT(JOFF)+1),1,X(IPARPT(JOFF)+1),1)
  ENDDO
  !
  ! Problems with Intel communication
  !
  RETURN
END SUBROUTINE AVDGSA
!
SUBROUTINE DAXPY_s(N,ONE,W,IONE,X,JONE)
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     Simplified BLAS routine with the same name
  !
  INTEGER I, N, IONE, JONE
  real(chm_real) ONE, W(*), X(*)
  DO I = 1, N
     X(I) = X(I) + W(I)
  ENDDO
  RETURN
END SUBROUTINE DAXPY_s
!
!=======================
!     Block communication for force decomposition....
!     Maybe we don't put PARASCAL keyword here because this
!     routines are used also in small communication only test
!     programs without any changes
!
!=======================
#if KEY_PARASCAL==1 /*force_decomp*/
SUBROUTINE BLOCBRA(IB,X,IPARPT,NDIM,MYNOD,IPPMAP,IPMAT,JPMAT)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global BRoadcast
  !     using recursive doubling scheme.
  !
  !     This is asynchronous version i.e. it can overlap send and receive
  !     operations for maximum speed.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !     IPMAT - block organization
  !     JPMAT - mapping of MYNOD to ME which is valid within one block.
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  real(chm_real) X(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER IB, IPMAT(4,*),JPMAT(4,0:*)
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT
  INTEGER IEOR,IAND
  !
  ITYPE=IB*NDIM
  ME=JPMAT(IB,MYNOD)
  !     Am I in this block?
  IF(ME.EQ.-1) RETURN
  DO IDIM = 1, NDIM
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     CALL GRECSEN(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE,X(IPARPT(IOFF)+1), &
          ILEN,X(IPARPT(JOFF)+1),JLEN)
     ITYPE=ITYPE+1
  ENDDO
  !
  RETURN
END SUBROUTINE BLOCBRA
!
SUBROUTINE BLOCKBRO(IB,X,IPARPT,NDIM,MYNOD,IPPMAP,IPMAT,JPMAT)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global BRoadcast
  !     using recursive doubling scheme.
  !
  !     This is version for force decomposition, where vectors are organized in
  !     blocks and each block is further divided on parts. This is very similar
  !     to AVDGBR with some additional mapping.
  !
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and like.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !     IPMAT - block organization
  !     JPMAT - mapping of MYNOD to ME which is valid within one block.
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  real(chm_real) X(*)
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER IB, IPMAT(4,*),JPMAT(4,0:*)
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT,ISYNC
  INTEGER IEOR,IAND
  !
  ITYPE=IB*NDIM
  ME=JPMAT(IB,MYNOD)
  !     Am I in this block?
  IF(ME.EQ.-1) RETURN
  DO IDIM = 1, NDIM
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     ISYNC=IAND(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     IF(ISYNC.EQ.IBIT) THEN
        CALL GREC(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE, &
             X(IPARPT(IOFF)+1),8*ILEN)
        CALL GSEN(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE, &
             X(IPARPT(JOFF)+1),8*JLEN)
     ELSE
        CALL GSEN(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE, &
             X(IPARPT(JOFF)+1),8*JLEN)
        CALL GREC(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE, &
             X(IPARPT(IOFF)+1),8*ILEN)
     ENDIF
     ITYPE=ITYPE+1
  ENDDO
  !
  RETURN
END SUBROUTINE BLOCKBRO
!
SUBROUTINE BLKBRA(X,IPARPT,NPART,MNSTEP,ISTEP,IDIM,MYNOD, &
     ISCHED,JSCHED,IPMAT,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global SUM
  !     using recursive halving scheme.
  !
  !     This is asynchronous version for concurent send/receive
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !     NPART - How many blocks
  !     ISTEP - communication step
  !     IPMAT - block organization
  !     ISCHED - mapping of MYNOD to ME which is valid within one block
  !              on a given step.
  !     JSCHED - To which block MYNOD processor belong on a given step.
  !
  !
  real(chm_real) X(*)
  INTEGER IPARPT(0:*),IPPMAP(0:*),MYNOD,IDIM,ISTEP,NPART,MNSTEP
  INTEGER IB,IPMAT(8,*),ISCHED(MNSTEP,0:*),JSCHED(MNSTEP,0:*)
  INTEGER ITYPE,IBIT,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT
  INTEGER IEOR,IAND
  !
  ME=ISCHED(ISTEP,MYNOD)
  !     Am I in this step?
  IF(ME.EQ.-1) RETURN
  IB=JSCHED(ISTEP,MYNOD)
  !
  ITYPE=IB
  !
  IBIT=2**(IDIM-1)
  NEXT=IEOR(ME,IBIT)
  IOFF=IAND(NEXT,-IBIT)+(IB-1)*NPART
  JOFF=IAND(ME,-IBIT)+(IB-1)*NPART
  ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
  JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
  CALL GRECSEN(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE,X(IPARPT(IOFF)+1), &
       ILEN,X(IPARPT(JOFF)+1),JLEN)
  !
  RETURN
END SUBROUTINE BLKBRA
!
SUBROUTINE BLKSA(X,IPARPT,W,NPART,MNSTEP,ISTEP,IDIM,MYNOD, &
     ISCHED,JSCHED,IPMAT,IPPMAP)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global SUM
  !     using recursive halving scheme.
  !
  !     This is asynchronous version for concurent send/receive
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !     NPART - How many blocks
  !     ISTEP - communication step
  !     IPMAT - block organization
  !     ISCHED - mapping of MYNOD to ME which is valid within one block
  !              on a given step.
  !     JSCHED - To which block MYNOD processor belong on a given step.
  !
  !
  real(chm_real) X(*), W(*), ONE/1.0D0/
  INTEGER IPARPT(0:*),IPPMAP(0:*),MYNOD,IDIM,ISTEP,NPART,MNSTEP
  INTEGER IB,IPMAT(8,*),ISCHED(MNSTEP,0:*),JSCHED(MNSTEP,0:*)
  INTEGER ITYPE,IBIT,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT
  INTEGER IEOR,IAND
  !
  ME=ISCHED(ISTEP,MYNOD)
  !     Am I in this step?
  IF(ME.EQ.-1) RETURN
  IB=JSCHED(ISTEP,MYNOD)
  !
  ITYPE=IB
  !
  IBIT=2**(IDIM-1)
  NEXT=IEOR(ME,IBIT)
  IOFF=IAND(NEXT,-IBIT)+(IB-1)*NPART
  JOFF=IAND(ME,-IBIT)+(IB-1)*NPART
  ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
  JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
  CALL GRECSEN(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE,W(IPARPT(JOFF)+1), &
       JLEN,X(IPARPT(IOFF)+1),ILEN)
  CALL DAXPY_s(JLEN,ONE,W(IPARPT(JOFF)+1),1,X(IPARPT(JOFF)+1),1)
  !
  RETURN
END SUBROUTINE BLKSA
!
!
SUBROUTINE BLOCKSA(IB,X,IPARPT,W,NDIM,MYNOD,IPPMAP,IPMAT,JPMAT)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global SUM
  !     using recursive halving scheme.
  !
  !     This is asynchronous version for concurent send/recieve
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !     IPMAT - block organization
  !     JPMAT - mapping of MYNOD to ME which is valid within one block.
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance. 
  !
  real(chm_real) X(*), W(*), ONE/1.0D0/
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER IB, IPMAT(4,4),JPMAT(4,0:9)
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT
  INTEGER IEOR,IAND
  !
  ITYPE=IB*NDIM
  ME=JPMAT(IB,MYNOD)
  !     Am I in this block?
  IF(ME.EQ.-1) RETURN
  DO IDIM = NDIM, 1, -1
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     CALL GRECSEN(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE,W(IPARPT(JOFF)+1), &
          JLEN,X(IPARPT(IOFF)+1),ILEN)
     ITYPE=ITYPE+1
     CALL DAXPY_s(JLEN,ONE,W(IPARPT(JOFF)+1),1,X(IPARPT(JOFF)+1),1)
  ENDDO
  RETURN
END SUBROUTINE BLOCKSA
!
SUBROUTINE BLOCKSUM(IB,X,IPARPT,W,NDIM,MYNOD,IPPMAP,IPMAT,JPMAT)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  implicit none
  !
  !     This routine performs Actual Vector Distributed Global SUM
  !     using recursive halving scheme. 
  !
  !     This is version for force decomposition, where vectors are organized in
  !     blocks and each block is further divided on parts. This is very similar
  !     to AVDGSUM with some additional mapping.
  !
  !     This is synchronized version i.e. it can use synchronized communication.
  !     Useful for TCP/IP and others.
  !
  !     X - vector to be broadcasted [input&output]
  !     IPARPT - vector of length NUMNODES+1 for offsets on each node [input]
  !     NDIM - dimension of hypercube [input]
  !     MYNOD - who am I [input]
  !     IPPMAP - mapping numbers of nodes (cube -> mesh) [1 to 1 on cube] [input]
  !     IPMAT - block organization
  !     JPMAT - mapping of MYNOD to ME which is valid within one block.
  !
  !     It is based on ideas from book Fox, et al, and Robert van de Geijn's
  !     papers & private communications and gives theoretically 
  !     best possible performance on hypercube architecture.
  !
  real(chm_real) X(*), W(*), ONE/1.0D0/
  INTEGER IPARPT(0:*), IPPMAP(0:*), MYNOD, NDIM
  INTEGER IB, IPMAT(4,4),JPMAT(4,0:9)
  INTEGER ITYPE,IBIT,IDIM,ME,IOFF,JOFF
  INTEGER ILEN,JLEN,NEXT,ISYNC
  INTEGER IEOR,IAND
  !
  ITYPE=IB*NDIM
  ME=JPMAT(IB,MYNOD)
  !     Am I in this block?
  IF(ME.EQ.-1) RETURN

  DO IDIM = NDIM, 1, -1
     IBIT=2**(IDIM-1)
     NEXT=IEOR(ME,IBIT)
     ISYNC=IAND(ME,IBIT)
     IOFF=IAND(NEXT,-IBIT)
     JOFF=IAND(ME,-IBIT)
     ILEN=IPARPT(IOFF+IBIT)-IPARPT(IOFF)
     JLEN=IPARPT(JOFF+IBIT)-IPARPT(JOFF)
     IF(ISYNC.EQ.IBIT) THEN
        CALL GREC(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE, &
             W(IPARPT(JOFF)+1),8*JLEN)
        CALL GSEN(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE, &
             X(IPARPT(IOFF)+1),8*ILEN)
     ELSE
        CALL GSEN(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE, &
             X(IPARPT(IOFF)+1),8*ILEN)
        CALL GREC(IPPMAP(IPMAT(IB,NEXT+1)),ITYPE, &
             W(IPARPT(JOFF)+1),8*JLEN)
     ENDIF
     ITYPE=ITYPE+1
     CALL DAXPY_s(JLEN,ONE,W(IPARPT(JOFF)+1),1,X(IPARPT(JOFF)+1),1)
  ENDDO
  RETURN
END SUBROUTINE BLOCKSUM
!
#endif /* (force_decomp)*/
!
SUBROUTINE APSYNG()
  !-----------------------------------------------------------------------
  use chm_kinds
  !
  use number
  implicit none
  !
  !     Actual Global Sync routine General 
  !     Just do Global sum of one number
  !
  real(chm_real) DUMMY,WDUMMY
  !
  DUMMY=ONE
  CALL GGSUM0(1,DUMMY,WDUMMY)
  !
  RETURN
END SUBROUTINE APSYNG
!
SUBROUTINE APSYNH()
  !-----------------------------------------------------------------------
  use chm_kinds
  !
  !     Actual Global Sync routine General (sending one byte around)
  !
  use dimens_fcm
  use parallel
  implicit none
  !
  INTEGER DUMMY,NEXT,IBIT,TAG,ME
  !
  !     Has to send at least something to really block the program
  !
  TAG=100
  ME=MYNOD
  IBIT=1
10 CONTINUE
  IF (ME.LT.IBIT) THEN
     NEXT=ME+IBIT
     IF (NEXT.LT.NUMNOD) THEN 
        CALL GSEN(IPPMAP(NEXT),TAG,DUMMY,1)
     ENDIF
  ELSE IF (ME.LT.2*IBIT) THEN
     NEXT=ME-IBIT
     CALL GREC(IPPMAP(NEXT),TAG,DUMMY,1)
  ENDIF
  IBIT=IBIT*2
  TAG=TAG+1
  IF (IBIT.LT.NUMNOD) GOTO 10
  !
  RETURN
END SUBROUTINE APSYNH
!
SUBROUTINE GGSUM0(N,X,W)
  !-----------------------------------------------------------------------
  use chm_kinds
  !
  !     General Global SUM (valid for any number of nodes)
  !
  !     Inefficient and simple, but it is true global sum
  !
  use dimens_fcm
  use parallel
  implicit none
  INTEGER N
  real(chm_real) X(*),W(*)
  !
  INTEGER R1, R2, S, TAG
  real(chm_real) :: ONE=1.0D0
  INTEGER :: IONE=1, JONE=1
  !
  TAG=1
  R1=2*MYNOD+1
  R2=R1+1
  S=(MYNOD-1)/2
  !
  IF(R1.LT.NUMNOD) THEN
     CALL GREC(IPPMAP(R1),TAG,W,8*N)
     CALL DAXPY_S(N,ONE,W,IONE,X,JONE)
  ENDIF
  !
  IF(R2.LT.NUMNOD) THEN
     CALL GREC(IPPMAP(R2),TAG,W,8*N)
     CALL DAXPY_S(N,ONE,W,IONE,X,JONE)
  ENDIF
  !
  IF(MYNOD.NE.0) THEN
     CALL GSEN(IPPMAP(S),TAG,X,8*N)
  ENDIF
  !
  CALL GBRDCS(N,X)
  !
  RETURN
END SUBROUTINE GGSUM0
!
SUBROUTINE GBRDCS(N,X)
  !-----------------------------------------------------------------------
  use chm_kinds
  !
  !     General BRoaDCaSt routine from node 0 
  !      (valid for any number of nodes)
  !
  use dimens_fcm
  use parallel
  implicit none
  INTEGER N
  real(chm_real) X(*)
  !
  INTEGER NEXT,IBIT,TAG,ME,I,IMAX
  !
  TAG=1
  ME=MYNOD
  IBIT=1
  IMAX=0
  IF(NUMNOD.GT.2**NODDIM)IMAX=1
  !CC      DO I = 1, NUMNOD/2
  DO I = 1,NODDIM+IMAX
     IF (ME.LT.IBIT) THEN
        NEXT=ME+IBIT
        !CC            IF (NEXT.LT.NUMNOD) CALL GSEN(IPPMAP(NEXT),TAG,X,8*N)
        IF (NEXT.LT.NUMNOD) CALL GSEN(IPPMAP(NEXT),TAG,X,N)
     ELSE IF (ME.LT.2*IBIT) THEN
        NEXT=ME-IBIT
        !CC            CALL GREC(IPPMAP(NEXT),TAG,X,8*N)
        CALL GREC(IPPMAP(NEXT),TAG,X,N)
     ENDIF
     IBIT=IBIT*2
  ENDDO
  !
  RETURN
  !
  !=========================================
  ! End of BLOCK stuff
  !
  !===================================
END SUBROUTINE GBRDCS
#else /* (parallel)*/
SUBROUTINE NULL_PL3
  RETURN
end SUBROUTINE NULL_PL3
#endif /* (parallel)*/

