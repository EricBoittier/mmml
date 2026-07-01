#if KEY_PARALLEL==1 /*pll_main*/
#if KEY_CMPI==1 /*cmpi*/
SUBROUTINE CMPI_INIT(STATUS &
#if KEY_MPI==1 && KEY_MULTICOM==1 /*  VO stringm */
&  ,initialized_mpi &           
#endif
&                          )
  !-----------------------------------------------------------------------
  ! Startup for parallel code.
  !-----------------------------------------------------------------------
  !  This code must be run when CHARMM is started to properly setup all
  !  data communication.
  !
  use chm_kinds
  !
  use dimens_fcm
  use parallel
  use stream
  use number
#if KEY_MPI==1
  use mpi                     
#endif
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux            
#endif
   !
  implicit none
  INTEGER STATUS,I
#if KEY_MULTICOM==1 /*  VO stringm */
  logical :: initialized_mpi              
#endif
  !
#if KEY_GAMESS==1
  character(len=40) versn                    
#endif
  !
#if KEY_PARASCAL==1
  NPART=8
#endif 
  ! VO stringm
#if KEY_MULTICOM==1
  initialized_mpi=.false.
#if KEY_MPI==1
  call MPI_INITIALIZED(initialized_mpi, status)
#endif
#endif
  !
! for DDI:   I=2
! for DDI:   CALL DDI_PBEG(I)
! for DDI:   !     Initialize for DDI /comment out this call in gamess.src
! for DDI:   CALL BEGING(VERSN)
#if KEY_MPI==1 /*tp*/
#if KEY_GAMESSUK==1 /*tp2*/
  CALL PARALLEL_INIT(STATUS)
#else /*         (tp2)*/
#if KEY_MULTICOM==1 /*  VO stringm v */
  if (initialized_mpi) then
   COMM_CHARMM=MPI_COMM_LOCAL
  else
#endif /* VO stringm ^ */
   CALL MPI_INIT(STATUS)
   COMM_CHARMM=MPI_COMM_WORLD
#if KEY_MULTICOM==1 /*  VO stringm */
  endif                                         
#endif
  CALL MPI_BARRIER(COMM_CHARMM,STATUS)
#endif /*         (tp2)*/
#endif /*   (tp)*/
  !
  !
  !     MYNODE - who am I
  !     NUMNOD - how many we are all together
  !
  !     This is for all parallel systems
  !
#if KEY_MULTICOM==1 /*  VO stringm v */
  if (initialized_mpi) then
    MYNOD=ME_LOCAL
    NUMNOD=SIZE_LOCAL
  else
#endif /* VO */
    MYNOD = MNOD() 
    NUMNOD = NNOD()
#if KEY_MULTICOM==1 /*  VO */
  endif                                 
#endif
  NODDIM = NPTWO()
  !C I don't think we need this??!! MH05:      write (outu,*) mynod," process started"
  !
  !     NUMNOD is 0 if using csh. Put it back to 1!
  if(numnod.gt.maxnode) call wrndie(-5,'<CMPI_INIT>','Too many processors acquired')
  IF(NUMNOD.EQ.0) NUMNOD = 1
#if KEY_MPI==1
  call cube(mynod,numnod,ippmap)
#else /**/
  CALL WRNDIE(-4,'<PARSTRT>','Unrecognized parallel machine type')
#endif 
  !
  RETURN
END SUBROUTINE CMPI_INIT

SUBROUTINE CMPI_BARRIER(COMM, STATUS)
  !-----------------------------------------------------------------------
  use chm_kinds
  !
  use dimens_fcm
  use stream
  use parallel
  implicit none
  !
  INTEGER COMM, STATUS

#if KEY_MULTICOM==0
  IF (COMM == CMPI_COMM_WORLD) THEN                    
#endif
     !
     !     This is a wrapper routine for global sync.
     !
     !     Use own gsync algorithm
#if KEY_GENCOMM==1
     !     The following is IPPMAP safe routine!
     CALL AGGSYNC()
#elif KEY_PARAFULL==1
     CALL APSYNC()
#elif KEY_PARASCAL==1
     CALL AGGSYNC()
#endif /*  TE*/
     !
#if KEY_MULTICOM==0 /*  VO stringm v */
  ELSE
     write(outu,*) &
          'CMPI-Error>Other than CMPI_COMM_WORLD not supported.'
  ENDIF
#endif /* VO stringm ^ */
  RETURN
END SUBROUTINE CMPI_BARRIER
!
!
SUBROUTINE CMPI_BCAST(ARRAY,LENGTH,SIZE,ROOT,WORLD,STATUS)
  !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  use parallel
#if KEY_MPI==1
  use mpi   
#endif

  implicit none
#if KEY_GNU==0 && KEY_OSX==0
  integer(int_byte) array(*)
#else /**/
  INTEGER ARRAY(*)
#endif 
  INTEGER LENGTH,SIZE,ROOT,WORLD,STATUS
  !
  INTEGER TAG,I
  !
#if KEY_MULTICOM==0 /*  VO stringm v */
  IF (WORLD .NE. CMPI_COMM_WORLD) THEN
     STATUS=-1
     write(outu,*) &
          'CMPI-Error>Other than CMPI_COMM_WORLD not supported.'
     RETURN
  ENDIF
  IF (SIZE .NE. CMPI_BYTE) THEN
     STATUS=-1
     write(outu,*) 'CMPI-Error>Other than CMPI_BYTE not supported.'
     RETURN
  ENDIF
#endif /* VO stringm ^ */
  !
#if KEY_MPI==1 /*comlib*/
#if KEY_REPLICA==1 || KEY_REPDSTR==1 /*pllrep*/
  !
  !     This code is IPPMAP safe, so it can be used in
  !     "group" based parallel methods (CONCURR,QM REPLICA/PATH)
  TAG=1
  IF (MYNOD.EQ.0) THEN
     CALL GSEN(IPPMAP(MOD(MYNOD+1,NUMNOD)),TAG,ARRAY,LENGTH)
  ELSE
     CALL GREC(IPPMAP(MOD(MYNOD-1+NUMNOD,NUMNOD)),TAG,ARRAY,LENGTH)
     IF(MYNODP.LT.NUMNOD) &
          CALL GSEN(IPPMAP(MOD(MYNOD+1,NUMNOD)),TAG,ARRAY,LENGTH)
  ENDIF
#else /* (pllrep)*/
  !
  !      write(outu,*) mynod," CALLING MPI_BCAST",length,root
  !      if(mynod .eq.0) write(outu,*) "+",array(1:length)
  CALL MPI_BCAST(ARRAY,LENGTH,MPI_BYTE,ROOT,COMM_CHARMM,STATUS)
#endif /* (pllrep)*/
! for DDI:  CALL DDI_BCAST(3333,'F',ARRAY,(LENGTH+7)/8,ROOT)
#else /* (comlib)*/
  !
  !     The code below uses SEND/REC from paral3.src
  !
  !      write(*,*)'CMPI_BCAST-start>me=',mynod
  TAG=1
  IF (MYNOD.EQ.0) THEN
#if KEY_GENCOMM==1
     CALL GSENR(IPPMAP, TAG, ARRAY, LENGTH)
#else /**/
     DO I=1,NUMNOD-1
        CALL GSEN(I, TAG, ARRAY, LENGTH)
     ENDDO
#endif /*  GENCOMM*/
  ELSE
#if KEY_GENCOMM==1
     CALL GRECL(IPPMAP, TAG, ARRAY, LENGTH)
     IF(MYNODP.LT.NUMNOD)CALL GSENR(IPPMAP,TAG,ARRAY,LENGTH)
#else /**/
     CALL GREC(0, TAG, ARRAY, LENGTH)
#endif /*  GENCOMM*/
  ENDIF
  !
  CALL PSYNC()
  !
  !      write(*,*)'CMPI_BCAST-end>me=',mynod
#endif /* (comlib)*/
  !
  RETURN
END SUBROUTINE CMPI_BCAST
!
SUBROUTINE CMPI_RED_SCAT(X,W,KPARPT,SIZE,OP,WORLD,STATUS)
  !-----------------------------------------------------------------------
  use chm_kinds
  use parallel
  implicit none
  !
  real(chm_real) X(*),W(*)
  INTEGER KPARPT(0:*),SIZE,OP,WORLD,STATUS
  INTEGER I,LPARPT(0:MAXNODE)
  !
  IF (SIZE .NE. CMPI_DOUBLE_PRECISION) THEN
     STATUS=-1
  ENDIF
  IF (OP .NE. CMPI_SUM) THEN
     STATUS=-2
  ENDIF
  IF (WORLD .NE. CMPI_COMM_WORLD) THEN
     STATUS=-3
  ENDIF
  LPARPT(0)=0
  DO I=0,NUMNOD-1
     LPARPT(I+1)=LPARPT(I)+KPARPT(I)
  ENDDO
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
  CALL AGVDGS(X,LPARPT,W,NUMNOD,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
  CALL AGVDGSA(X,LPARPT,W,NUMNOD,MYNOD,IPPMAP) 
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==1
  CALL AVDGSUM(X,LPARPT,W,NODDIM,MYNOD,IPPMAP) 
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==0
  CALL AVDGSA(X,LPARPT,W,NODDIM,MYNOD,IPPMAP)  
#endif
  !
  w(1:kparpt(mynod)) = x(lparpt(mynod)+1:lparpt(mynod)+kparpt(mynod))
  !
  RETURN
END SUBROUTINE CMPI_RED_SCAT
!
SUBROUTINE CMPI_GATHERV(X,KPARPT,XSIZE,W,LPARPT,MPARPT,WSIZE, &
     WORLD,STATUS)
  !-----------------------------------------------------------------------
  use chm_kinds
  use parallel
  implicit none
  !
  real(chm_real) X(*),W(*)
  INTEGER KPARPT,LPARPT(0:*),MPARPT(0:*)
  INTEGER XSIZE,WSIZE,WORLD,STATUS
  !
  STATUS=0
  IF ((XSIZE .NE. CMPI_DOUBLE_PRECISION) &
       .OR. (WSIZE .NE. CMPI_DOUBLE_PRECISION)) THEN
     STATUS=-1
  ENDIF
  IF (WORLD .NE. CMPI_COMM_WORLD) THEN
     STATUS=-3
  ENDIF
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
  CALL AGVDGB(X,MPARPT,NUMNOD,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
  CALL AGVDGBA(X,MPARPT,NUMNOD,MYNOD,IPPMAP) 
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==1
  CALL AVDGBR(X,MPARPT,NODDIM,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==0
  CALL AVDGBRA(X,MPARPT,NODDIM,MYNOD,IPPMAP) 
#endif
  !
  w(1:mparpt(numnod)) = x(1:mparpt(numnod))
  !
  RETURN
END SUBROUTINE CMPI_GATHERV
!
SUBROUTINE CIMPI_GATHERV(X,XSIZE,MPARPT,WORLD,STATUS)
  !-----------------------------------------------------------------------
  use chm_kinds
  use parallel
  implicit none
  !
  real(chm_real) X(*)
  INTEGER MPARPT(0:*)
  INTEGER XSIZE,WORLD,STATUS
  !
  STATUS=0
  IF (XSIZE .NE. CMPI_DOUBLE_PRECISION) THEN
     STATUS=-1
  ENDIF
  IF (WORLD .NE. CMPI_COMM_WORLD) THEN
     STATUS=-3
  ENDIF
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
  CALL AGVDGB(X,MPARPT,NUMNOD,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
  CALL AGVDGBA(X,MPARPT,NUMNOD,MYNOD,IPPMAP) 
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==1
  CALL AVDGBR(X,MPARPT,NODDIM,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==0
  CALL AVDGBRA(X,MPARPT,NODDIM,MYNOD,IPPMAP) 
#endif
  !
  RETURN
END SUBROUTINE CIMPI_GATHERV
!
SUBROUTINE CIMPI_ALLRED(X,W,N,SIZE,OP,WORLD,STATUS)
  !-----------------------------------------------------------------------
  use chm_kinds
  !
  use parallel
  use exfunc
  implicit none
  real(chm_real) X(*),W(*)
  INTEGER SIZE,OP,WORLD,STATUS
  !
  INTEGER I,N
  !
  STATUS=0
  JPARPT(0)=0
  DO I = 1, NUMNOD
     JPARPT(I)=N*I/NUMNOD
  ENDDO
  !
  IF ((SIZE .EQ. CMPI_DOUBLE_PRECISION) .AND. (OP .EQ. CMPI_SUM)) THEN
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
     CALL AGVDGS(X,JPARPT,W,NUMNOD,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
     CALL AGVDGB(X,JPARPT,NUMNOD,MYNOD,IPPMAP)    
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
     CALL AGVDGSA(X,JPARPT,W,NUMNOD,MYNOD,IPPMAP) 
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
     CALL AGVDGBA(X,JPARPT,NUMNOD,MYNOD,IPPMAP)   
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==1
     CALL AVDGSUM(X,JPARPT,W,NODDIM,MYNOD,IPPMAP) 
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==1
     CALL AVDGBR(X,JPARPT,NODDIM,MYNOD,IPPMAP)    
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==0
     CALL AVDGSA(X,JPARPT,W,NODDIM,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==0 && KEY_SYNCHRON==0
     CALL AVDGBRA(X,JPARPT,NODDIM,MYNOD,IPPMAP)   
#endif
  ELSEIF ((SIZE .EQ. CMPI_INTEGER) .AND. (OP .EQ. CMPI_SUM)) THEN
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
     CALL IGVDGS(X,JPARPT,W,NUMNOD,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
     CALL IGVDGB(X,JPARPT,NUMNOD,MYNOD,IPPMAP)    
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
     CALL IGVDGS(X,JPARPT,W,NUMNOD,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
     CALL IGVDGB(X,JPARPT,NUMNOD,MYNOD,IPPMAP)    
#endif
#if KEY_GENCOMM==0
     CALL IVDGSUM(X,JPARPT,W,NODDIM,MYNOD,IPPMAP) 
#endif
#if KEY_GENCOMM==0
     CALL IVDGBR(X,JPARPT,NODDIM,MYNOD,IPPMAP)    
#endif
  ELSEIF ((SIZE .EQ. CMPI_INTEGER) .AND. (OP .EQ. CMPI_BOR)) THEN
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
     CALL IGVDGO(X,JPARPT,W,NUMNOD,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==1
     CALL IGVDGB(X,JPARPT,NUMNOD,MYNOD,IPPMAP)    
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
     CALL IGVDGO(X,JPARPT,W,NUMNOD,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==1 && KEY_SYNCHRON==0
     CALL IGVDGB(X,JPARPT,NUMNOD,MYNOD,IPPMAP)    
#endif
#if KEY_GENCOMM==0
     CALL IVDGOR(X,JPARPT,W,NODDIM,MYNOD,IPPMAP)  
#endif
#if KEY_GENCOMM==0
     CALL IVDGBR(X,JPARPT,NODDIM,MYNOD,IPPMAP)    
#endif
  ELSE
     STATUS=-1
  ENDIF
  !
  RETURN
END SUBROUTINE CIMPI_ALLRED
#endif /* (cmpi)*/

SUBROUTINE SWAPD1(N,X,XPL,XPR,LPEER,RPEER)
  !-----------------------------------------------------------------------
  ! This routine performs a symple swap of data between X and XP.
  ! X is on MYNOD and XP is on PEER
  !------------------------------------
  ! MODE =  0   swap between MYNOD and PEER
  ! MODE = -1   only send to PEER
  ! MODE =  1   only receive from PEER
  !
  use chm_kinds
  implicit none
  !
  INTEGER N,LPEER,RPEER
  real(chm_real) X(*),XPL(*),XPR(*)
  !
  IF((LPEER.GE.0).AND.(RPEER.GE.0)) THEN
     CALL SWAPD3(N,X,XPL,XPR,LPEER,RPEER)
  ELSE IF(LPEER.LT.0) THEN
     CALL GRECSEN(RPEER,1,XPR,N,X,N)
  ELSE
     CALL GRECSEN(LPEER,1,XPL,N,X,N)
  ENDIF
  !
  RETURN
END SUBROUTINE SWAPD1
!
#else /*     (pll_main)    PARALLEL*/
SUBROUTINE NULL_PL2
  RETURN
END SUBROUTINE NULL_PL2
#endif /*   (pll_main)*/

