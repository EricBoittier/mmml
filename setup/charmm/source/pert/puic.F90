#if KEY_TSM==1
SUBROUTINE PUICA(COMLY1,COMLE1,MXSIZE,IBX,IBY,IBZ,XTEM,YTEM,ZTEM, &
     RMSV,AMASST)
  !-----------------------------------------------------------------------
  !     Basic function:
  !     Search for the conformational path under some internal
  !     coordinate constrains. The simplest method is used:
  !
  !     1. A ref. frame is obtained by step forward along the straight
  !        line between the previous frame and the final frame in Cartesian
  !        coordinate space.
  !     2. Subroutine ICFCNS is use to optimize the ref. frame under the
  !        first set of internal coordinate constraint.
  !     3. Minimization of the optimized structure with the second set of
  !        internal coordinate constraints
  !     4. If the distance between the new frame and the final frame is too
  !        big, goto step1, else stop.
  !
  !     Usage: PUIC NMAX int STEPSIZE real USTART int UEND int UOUT int
  !        NMAX:     Maximum number of the intermediate grids
  !        STEPSIZE: RMSD between adjacent grids,default (SDEF=0.02D0,)
  !        USTART:   Unit connected to the standard CHARMM CRD file of the
  !                   reactant conformation
  !        UEND:     Unit connected to the standard CHARMM CRD file of the
  !                   product conformation
  !         UOUT:     Unit connected to the unformated CHARMM trajectory file
  !                   where the path will be written.
  !        CSTP:     Constrained stepsize for IC constraint set 2.
  !                   default (CSPSEF=15.0D0, for dihedral constraint)
  !
  !
  !     Y. Wang, Lawrence, KS 08/23/96
  !
  !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg                     
#endif

  use chm_kinds
  use intcor_module
  use intcor2,only:writic,fillic
  use dimens_fcm
  use exfunc
  use number
  use memory
  use stream
  !
  use bases_fcm
  use comand
  use consta
  use coord
  use coordc
  use corsubs,only:orintc
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use icpert
  use icfix
  use string

  use minmiz_module, only: minmiz
  
  implicit none
  integer,allocatable,dimension(:) :: islct, freeat, iscr
  CHARACTER(LEN=*) COMLY1
  CHARACTER(LEN=50) COMLY2
  INTEGER COMLE1,COMLE2
  !
  INTEGER NMAX, NUI, NUF, NUO, MXSIZE,MMSIZE,I,J,K,NSTEP
  real(chm_real) STEP
  PARAMETER (MMSIZE=20)

  type(chm_ptr) ibx(*), iby(*), ibz(*)

  real(chm_real) TEMP1, STEPF
  !
  INTEGER IMODE
  !
  ! RMS parameters
  real(chm_real) AMASST(*),RMSV(*)
  !
  ! parameters for stepsize constraint in IC set 2
  real(chm_real) CSTP,MRATIO,MXD,STEPFO
  real(chm_real) XTEM(*),YTEM(*),ZTEM(*)
  INTEGER NCOUNT
  !
  ! parameters used in WRITCV
  INTEGER      NFREAT,NDEGF
  !
  ! local parameters used for internal coordinate constraint
  !   For the first set:
  INTEGER NICF1,IICF1,ICFTY1(MAXICF),ICFAT1(4,MAXICF),MAXI1
  real(chm_real) SVAL1(MAXICF),TOLI1(MAXICF),SMF1(3,MXLENS), &
       GCOEF1(MAXICF),DS1(MAXICF)
  LOGICAL ICFAD1(MAXICF),ANYAD1
  !   For the second set:
  INTEGER NICF2,IICF2,ICFTY2(MAXICF),ICFAT2(4,MAXICF),MAXI2
  real(chm_real) SVAL2(MAXICF),TOLI2(MAXICF),SMF2(3,MXLENS), &
       GCOEF2(MAXICF),DS2(MAXICF)
  LOGICAL ICFAD2(MAXICF),ANYAD2
  !  local parameter for command line process
  LOGICAL EOF,GO,OK,LPRICF
  CHARACTER(LEN=4) WRD
  !
  ! Finish processing the first command line
  !

  call chmalloc('puic.src','PUICA','islct',natom,intg=islct)

  CALL PROCES(COMLY1,COMLE1,NMAX,STEP,NUI,NUF,NUO,MXSIZE, &
       MMSIZE,AMASST,RMSV,IMODE,ISLCT,CSTP)
  COMLE2=COMLE1
  COMLY2 = COMLY1(1:COMLE2)
  !
  !  save the first constrained I.C. set, SVAL1 as in the final config.
  !
  CALL ICVUPD(XCOMP,YCOMP,ZCOMP)
  IF (NICF  >  0) THEN
     CALL COPYIC(NICF,IICF,ICFTYP,ICFATN,MAXI,SVAL,TOLI,SMF,GCOEF, &
          ICFADJ,ANYADJ,DS,NICF1,IICF1,ICFTY1,ICFAT1,MAXI1,SVAL1,TOLI1, &
          SMF1,GCOEF1,ICFAD1,ANYAD1,DS1,MAXICF,MXLENS)
     NICF=0
     IICF=0
  ELSE
     CALL WRNDIE(-3,'<PUIC>','Error in the ICFIX')
     RETURN
  ENDIF
  !
  ! Read and preocess the rest command lines
  !
  LPRICF = .FALSE.
  GO = .TRUE.
  EOF = .FALSE.
  loop50: do while( GO )
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
          .TRUE.,'PUIC> ')
     IF (EOF) THEN
        IF (NSTRM  ==  1) THEN
           GO = .FALSE.
           exit loop50
        ELSE
           CALL PPSTRM(OK)
           EOF=.FALSE.
        END IF
     END IF
     WRD=NEXTA4(COMLYN,COMLEN)
     IF (WRD  ==  '    ') THEN
        CONTINUE
     ELSE IF (WRD == 'FIX') THEN
        CALL ICFSET
        IF (NICF > 0) LPRICF = .TRUE.
     ELSE IF (WRD == 'MAXI') THEN
        MAXI = NEXTI(COMLYN,COMLEN)
        IF(MAXI <= 0) CALL WRNDIE(0,'<PUIC>', &
             'error parsing number of iterations')
     ELSE IF (WRD == 'END') THEN
        ! PUIC exit
        GO = .FALSE.
        IF(LPRICF) THEN
           CALL PRNICF
           IICF = 1
           LPRICF = .FALSE.
        END IF
     ELSE
        CALL WRNDIE(0,'<TSMS>','Unknown command.  Try again.')
     END IF
     ! end of the loop for PUIC command line parsing
  enddo loop50

  IF(EOF) THEN
     ! program termination
     CALL WRNDIE(-4,'<PUIC>', &
          'Eof found during input for PUIC set up.')
     RETURN
  ENDIF
  !
  ! end of the PUIC command line process
  !
  ! save the second IC constraint immediately
  !
  !
  IF (NICF  >  0) THEN
     CALL COPYIC(NICF,IICF,ICFTYP,ICFATN,MAXI,SVAL,TOLI,SMF,GCOEF, &
          ICFADJ,ANYADJ,DS,NICF2,IICF2,ICFTY2,ICFAT2,MAXI2,SVAL2,TOLI2, &
          SMF2,GCOEF2,ICFAD2,ANYAD2,DS2,MAXICF,MXLENS)
  ELSE
     CALL WRNDIE(-3,'<PUIC>','Error in the ICFIX')
     RETURN
  END IF
  !
  !
  ! Main loop.
  ! a.    walk forward a step
  ! b.        set IC using IC set 1 and check it using IC2 in ICC2 loop
  ! c.    energy minimization using under IC constraint 2
  !
  NDEGF=3*NATOM-6
  NFREAT = NATOM
  call chmalloc('puic.src','PUICA','freeat',nfreat,intg=freeat)
  NSTEP = 1
  mainloop: DO I=1,NMAX-1
     call chmalloc('puic.src','PUICA','IBX(I)',NATOM,crlp=IBX(I)%a)
     call chmalloc('puic.src','PUICA','IBY(I)',NATOM,crlp=IBY(I)%a)
     call chmalloc('puic.src','PUICA','IBZ(I)',NATOM,crlp=IBZ(I)%a)
     CALL COPYD (X,Y,Z,IBX(I)%a,IBY(I)%a,IBZ(I)%a,NATOM)
     !
     IF (RMSV(I)  <  STEP) exit mainloop
     !
     !
     NSTEP = NSTEP + 1
     STEPF = STEP / RMSV(I)
     NCOUNT = 0
     STEPFO = 0.0D0
     MRATIO = 1.0D0
     DO J=1,NATOM
        XTEM(J) = X(J)
        YTEM(J) = Y(J)
        ZTEM(J) = Z(J)
     ENDDO
     DO J=1,NICF
        SVAL2(J) = SVAL(J)
     ENDDO
     !
     !    ICC2 loop to ensure small stepsize in IC space defined by IC constraint
     !           set 2
     !
     !       try a step forward
1234 STEPF  = STEPF / MRATIO
     NCOUNT = NCOUNT + 1
     CALL COPYIC(NICF1,IICF1,ICFTY1,ICFAT1,MAXI1,SVAL1,TOLI1, &
          SMF1,GCOEF1,ICFAD1,ANYAD1,DS1,NICF,IICF,ICFTYP,ICFATN, &
          MAXI,SVAL,TOLI,SMF,GCOEF,ICFADJ,ANYADJ,DS,MAXICF,MXLENS)
     DO J=1,NATOM
        X(J) = XTEM(J) + (XCOMP(J)-XTEM(J)) * STEPF
        Y(J) = YTEM(J) + (YCOMP(J)-YTEM(J)) * STEPF
        Z(J) = ZTEM(J) + (ZCOMP(J)-ZTEM(J)) * STEPF
     enddo
     !
     !       set SVAL in IC 1
     CALL ICFCNS(XCOMP,YCOMP,ZCOMP,X,Y,Z,AMASS,NATOM)
     !
     !       update SVAL in IC 2
     CALL COPYIC(NICF2,IICF2,ICFTY2,ICFAT2,MAXI2,SVAL2,TOLI2, &
          SMF2,GCOEF2,ICFAD2,ANYAD2,DS2,NICF,IICF,ICFTYP,ICFATN, &
          MAXI,SVAL,TOLI,SMF,GCOEF,ICFADJ,ANYADJ,DS,MAXICF,MXLENS)
     CALL ICVUPD(X,Y,Z)
     !
     !       find the maxdif between SVAL and SVAL2
     MXD = 0.0D0
     DO K = 1, NICF
        TEMP1 = SVAL(K) - SVAL2(K)
        IF (TEMP1  >=  180.0D0) TEMP1 = TEMP1 - 360.0D0
        IF (TEMP1  <  -180.0D0) TEMP1 = TEMP1 + 360.0D0
        TEMP1 = DABS(TEMP1)
        IF (TEMP1  >  MXD) MXD = TEMP1
     END DO
     MRATIO = MXD / CSTP
     IF(PRNLEV > 3) write(OUTU,*) 'STEPF,MXD', STEPF,MXD
     IF (((MRATIO > 1.0D0).OR.(MRATIO < 0.5)).AND.(NCOUNT < 100)) &
          GOTO 1234
     !
     !  end of the ICC2 loop
     !
     !    Energy minimization
     COMLE1=COMLE2
     COMLY1 = COMLY2(1:COMLE2)
     CALL MINMIZ(COMLY1,COMLE1)
     !...test_start
     IF(PRNLEV > 3) WRITE(OUTU,*) ' ...INTCOR after MINMIZ step'
     CALL FILLIC(icr_struct%LENIC,.FALSE.,.FALSE.,X,Y,Z, &
          icr_struct%B1ic,icr_struct%B2ic, &
          icr_struct%T1ic,icr_struct%T2ic, &
          icr_struct%PIC, icr_struct%IAR, &
          icr_struct%JAR, icr_struct%KAR, &
          icr_struct%LAR, icr_struct%TAR)
     CALL WRITIC(1,icr_struct%LENIC,1,1,OUTU, &
          icr_struct%B1ic,icr_struct%B2ic, &
          icr_struct%T1ic,icr_struct%T2ic, &
          icr_struct%PIC, icr_struct%IAR, &
          icr_struct%JAR, icr_struct%KAR, &
          icr_struct%LAR, icr_struct%TAR)
     !...test_stop
     !
     call chmalloc('puic.src','PUICA','iscr',2*natom,intg=iscr)
     CALL ORINTC(NATOM,X,Y,Z,XCOMP,YCOMP,ZCOMP,AMASS,.FALSE.,.TRUE., &
          ISCR,ISLCT,.FALSE.,WMAIN,.FALSE.,.TRUE.)
     ! APH: Note: FRESTK below was commented in the original code = memory leak
     call chmdealloc('puic.src','PUICA','iscr',2*natom,intg=iscr)
     !
     CALL RMSDIF(X,Y,Z,XCOMP,YCOMP,ZCOMP,NATOM,ISLCT, &
          AMASST(NSTEP),RMSV(NSTEP))
     IF(PRNLEV > 3) THEN
        WRITE(OUTU,*) '<PUIC> There are ',NCOUNT,'IC modifications.'
        WRITE(OUTU,*) '       RMSDIF = ',RMSV(NSTEP),'in step ',NSTEP
     ENDIF
     !
  END DO mainloop

  !  End main loop

  call chmdealloc('puic.src','PUICA','iscr',2*natom,intg=iscr)
  call chmalloc('puic.src','PUICA','islct',natom,intg=islct)
  !
  IF (NSTEP  <  NMAX) THEN
     NSTEP = NSTEP + 1
     call chmalloc('puic.src','PUICA','IBX(NSTEP)',NATOM,crlp=IBX(NSTEP)%a)
     call chmalloc('puic.src','PUICA','IBY(NSTEP)',NATOM,crlp=IBY(NSTEP)%a)
     call chmalloc('puic.src','PUICA','IBZ(NSTEP)',NATOM,crlp=IBZ(NSTEP)%a)
     CALL COPYD (XCOMP,YCOMP,ZCOMP,IBX(NSTEP)%a, &
          IBY(NSTEP)%a,IBZ(NSTEP)%a,NATOM)
     RMSV(NSTEP)=0.0D0
     IF(PRNLEV > 3) WRITE(OUTU,*) &
          ' PUIC> Path found. There are total ', NSTEP, &
          'points along the path.'
  ELSE
     IF(PRNLEV > 3) THEN
        WRITE(OUTU,*) ' PUIC> Path not found at cycle ', NSTEP
        WRITE(OUTU,*) ' The RMS between the last two points is ', &
             RMSV(NSTEP-1)
     ENDIF
  ENDIF
  !
  !  print the RMSD between ith step and the final step
  IF(PRNLEV > 3) THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) 'iTH POINT                 RMSD '
  ENDIF
  DO I=1,NSTEP
     IF(PRNLEV > 3) WRITE(OUTU,11) I, RMSV(I)
     CALL WRITCV(IBX(I)%a,IBY(I)%a,IBZ(I)%a, &
#if KEY_CHEQ==1
          CG,QCG,                                  & 
#endif
          NATOM, &
          FREEAT,NFREAT,1,I,NDEGF,ZERO,1,NSTEP,TITLEA, &
          NTITLA,NUO,.FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
     call chmdealloc('puic.src','PUICA','IBX(I)',NATOM,crlp=IBX(I)%a)
     call chmdealloc('puic.src','PUICA','IBY(I)',NATOM,crlp=IBY(I)%a)
     call chmdealloc('puic.src','PUICA','IBZ(I)',NATOM,crlp=IBZ(I)%a)
  enddo
  call chmdealloc('puic.src','PUICA','freeat',nfreat,intg=freeat)
11 FORMAT(1X,I5,1X,F16.4)
  !
  RETURN
END SUBROUTINE PUICA

SUBROUTINE COPYIC(NICF2,IICF2,ICFTY2,ICFAT2,MAXI2,SVAL2,TOLI2, &
     SMF2,GCOEF2,ICFAD2,ANYAD2,DS2,NICF1,IICF1,ICFTY1,ICFAT1, &
     MAXI1,SVAL1,TOLI1,SMF1,GCOEF1,ICFAD1,ANYAD1,DS1,MAXICF,MXLENS)
  !
  ! Copy IC constraint set 2 to set 1
  !
  use chm_kinds
  implicit none
  INTEGER MAXICF,I,J,MXLENS
  !   For set 1:
  INTEGER NICF1,IICF1,ICFTY1(MAXICF),ICFAT1(4,MAXICF),MAXI1
  real(chm_real) SVAL1(MAXICF),TOLI1(MAXICF),SMF1(3,MXLENS), &
       GCOEF1(MAXICF),DS1(MAXICF)
  LOGICAL ICFAD1(MAXICF),ANYAD1
  !   For set 2:
  INTEGER NICF2,IICF2,ICFTY2(MAXICF),ICFAT2(4,MAXICF),MAXI2
  real(chm_real) SVAL2(MAXICF),TOLI2(MAXICF),SMF2(3,MXLENS), &
       GCOEF2(MAXICF),DS2(MAXICF)
  LOGICAL ICFAD2(MAXICF),ANYAD2
  !
  NICF1 = NICF2
  IICF1 = IICF2
  MAXI1 = MAXI2
  ANYAD1=ANYAD2
  DO I = 1, NICF1
     ICFTY1(I)= ICFTY2(I)
     SVAL1(I) = SVAL2(I)
     TOLI1(I) = TOLI2(I)
     GCOEF1(I)= GCOEF2(I)
     DS1(I)   = DS2(I)
     ICFAD1(I)= ICFAD2(I)
     DO J = 1, 4
        ICFAT1(J,I)=ICFAT2(J,I)
     END DO
  END DO
  DO I = 1, MXLENS
     DO J = 1, 3
        SMF1(J,I) = SMF2(J,I)
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE COPYIC

SUBROUTINE PROCES(COMLYN,COMLEN,NMAX,STEP,NUI,NUF,NUO,MXSIZE, &
     MMSIZE,AMASST,RMSV,IMODE,ISLCT,CSTP)
  !
  ! Finish processing the first command line of PUICA
  !
  use chm_kinds
  use dimens_fcm
  use number
  use memory
  use select
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use coordc
  use ctitla
  use deriv
  use psf
  use icpert
  use corsubs,only:orintc
  use coorio_mod,only:coorio

  implicit none
  integer,allocatable,dimension(:) :: iscr
  CHARACTER(LEN=*) COMLYN
  CHARACTER(LEN=8) COMLY1
  INTEGER COMLEN,COMLE1
  !
  INTEGER NMAX, NUI, NUF, NUO, I,J,MXSIZE,MMSIZE,NSTEP
  real(chm_real) STEP, SDE ,CSTP
  real(chm_real),PARAMETER :: SDEF=0.02D0,CSPDEF=15.0D0
  INTEGER ICNTRL(20)
  !
  real(chm_real) TEMP1
  !
  INTEGER IMODE, ISLCT(*)
  !
  ! RMS parameters
  real(chm_real) AMASST(MXSIZE),RMSV(MXSIZE)
  !
  NMAX  =GTRMI(COMLYN, COMLEN, 'NMAX',  MXSIZE)
  STEP  =GTRMF(COMLYN, COMLEN, 'STEP',   SDEF)
  NUI   =GTRMI(COMLYN, COMLEN, 'USTA',      1)
  NUF   =GTRMI(COMLYN, COMLEN, 'UEND',      2)
  NUO   =GTRMI(COMLYN, COMLEN, 'UOUT',      3)
  CSTP  =GTRMF(COMLYN, COMLEN, 'CSTP', CSPDEF)
  !
  IF(PRNLEV > 3) THEN
     WRITE(OUTU,*) ' PUIC> Maximum grid number  = ', NMAX
     WRITE(OUTU,*) ' PUIC> Initial stepsize     = ', STEP
     WRITE(OUTU,*) ' PUIC> Unit of the reactant = ', NUI
     WRITE(OUTU,*) ' PUIC> Unit of the product  = ', NUF
     WRITE(OUTU,*) ' PUIC> Unit of path         = ', NUO
     WRITE(OUTU,*) ' PUIC> Maximum step for IC2 = ', CSTP
  ENDIF
  !
  !   SELECT-ATOMS
  IMODE=0
  CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
       .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
       .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
  IF(IMODE /= 0) THEN
     CALL WRNDIE(0,'<PUIC>','ATOM SELECTION ERROR')
  ENDIF
  !
  !   Calculate RMSD between the end-points
  COMLY1 = 'CARD'
  COMLE1 = LEN(COMLY1)
  CALL COORIO(-1,NUI,COMLY1,COMLE1,TITLEB,NTITLB, &
       ICNTRL,NATOM,X,Y,Z,WMAIN,ATYPE, &
       RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)
  COMLY1 = 'CARD'
  COMLE1 = LEN(COMLY1)
  CALL COORIO(-1,NUF,COMLY1,COMLE1,TITLEB,NTITLB, &
       ICNTRL,NATOM,XCOMP,YCOMP,ZCOMP,WMAIN,ATYPE, &
       RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)

  call chmalloc('puic.src','PROCES','iscr',2*natom,intg=iscr)

  CALL ORINTC(NATOM,X,Y,Z,XCOMP,YCOMP,ZCOMP,AMASS,.FALSE.,.TRUE., &
       ISCR,ISLCT,.FALSE.,WMAIN,.FALSE.,.TRUE.)
  ! APH: Note: The FRESTK call was commented in the original = memory leak
  call chmdealloc('puic.src','PROCES','iscr',2*natom,intg=iscr)
  !
  CALL RMSDIF(X,Y,Z,XCOMP,YCOMP,ZCOMP,NATOM,ISLCT, &
       AMASST(1),RMSV(1))
  !
  !     CHECK NMAX and STEP
  IF (NMAX  >  MXSIZE) THEN
     NMAX = MXSIZE
     IF(WRNLEV > 3) WRITE(OUTU,*) &
          ' PUIC> Warning: Too many grids: NMAX =',NMAX, &
          ' #grids reset to default value ', MXSIZE
  ELSE IF (NMAX  <  MMSIZE .AND. NMAX  >  2) THEN
     IF(WRNLEV > 3) WRITE(OUTU,*) &
          ' PUIC> **Warning: Too few grids: NMAX =',NMAX
  ELSE IF (NMAX  <  2) THEN
     NMAX = MXSIZE
     IF(WRNLEV > 3) WRITE(OUTU,*) ' PUIC> NMAX is reset to ',NMAX
  END IF
  !      TEMP1 = RMSV(1) / (NMAX-1) * 2
  !      IF (TEMP1*0.75  >  STEP .OR. STEP  >  TEMP1*1.5) THEN
  !        STEP = TEMP1
  !        WRITE(OUTU,*) ' PUIC> STEP is reset to ', STEP
  !      ENDIF
  !
  RETURN
END SUBROUTINE PROCES

SUBROUTINE RMSDIF(X,Y,Z,XCOMP,YCOMP,ZCOMP,NATOM,ISLCT, &
     AMASST,RMSV)
  !
  !  Simply compute the RMS difference between the two coordinate set
  !  for the selected atoms, and calculate the unit vector between the
  !  two configurations
  !
  !  RMSV               :  The actual RMSD
  !  AMASST      :  Number of selected atoms
  !
  use chm_kinds
  use number
  use stream
  use machutil,only:die
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),XCOMP(*),YCOMP(*),ZCOMP(*)
  real(chm_real) RMST,AMASST,RMSV
  INTEGER NATOM, ISLCT(*), I, NMISS

  NMISS = 0
  AMASST=0.0D0
  RMST  =0.0D0
  DO I=1,NATOM
     IF (ISLCT(I) == 1) THEN
        IF (X(I) == ANUM) THEN
           NMISS=NMISS+1
        ELSEIF (XCOMP(I) == ANUM) THEN
           NMISS=NMISS+1
        ELSE
           AMASST = AMASST+1.0D0
           RMST=RMST+(X(I)-XCOMP(I))**2
           RMST=RMST+(Y(I)-YCOMP(I))**2
           RMST=RMST+(Z(I)-ZCOMP(I))**2
        ENDIF
     ENDIF
  enddo
  !
  IF (NMISS > 0) THEN
     IF(WRNLEV > 3) WRITE(OUTU,22) NMISS
     CALL DIE
22   FORMAT(/' **** ERROR **** THERE WERE',I5, &
          ' MISSING COORDINATES, TERMINATING'/)
     !
  ELSEIF (AMASST >  0.0D0) THEN
     RMSV=SQRT(RMST/AMASST)
     IF(PRNLEV >= 3) WRITE(OUTU,33) RMST,AMASST,RMSV
33   FORMAT(' TOTAL SQUARE DIFF IS',F12.4,'  DENOMINATOR IS',F12.4,/ &
          '       THUS RMS DIFF IS',F12.6)
  ELSE
     RMSV=0.0D0
     IF(WRNLEV > 3) WRITE(OUTU,*) &
          ' *** WARNING *** No atom is selsected'
  ENDIF
  !
  RETURN
END SUBROUTINE RMSDIF

SUBROUTINE ICVUPD(X,Y,Z)
  !
  !     This routine updates the values of coordinates constrained
  !     using the TSM "FIX" commands
  !
  !     Author: K. Kuczera and Y. Wang
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use icfix
  use intcor2,only:geticv
  implicit none
  !
  real(chm_real) X(1),Y(1),Z(1)
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  INTEGER IIC,ICTYPE,I,J,K,L
  !
  IF(NICF == 0) RETURN
  !
  !     Loop over constraints
  !
  DO IIC=1,NICF
     ICTYPE=ICFTYP(IIC)
     I=ICFATN(1,IIC)
     J=ICFATN(2,IIC)
     K=ICFATN(3,IIC)
     L=ICFATN(4,IIC)
     ICTYPE=ICFTYP(IIC)
     CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
     IF(ICTYPE == 1) SVAL(IIC)=RIJ
     IF(ICTYPE == 2) SVAL(IIC)=TIJK
     IF(ICTYPE == 3) SVAL(IIC)=PIJKL
     IF(ICTYPE < 1 .OR. ICTYPE > 3) THEN
        IF(WRNLEV > 3) WRITE(OUTU,*) &
             ' ****Error in ICVUPD: illegal ICTYPE'
     END IF
  END DO
  return
end SUBROUTINE ICVUPD

#endif 
SUBROUTINE NULL_PUI
  RETURN
END SUBROUTINE NULL_PUI

