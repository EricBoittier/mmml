module rxpath
  !     Files in the PATH module are concatenated into PATH.SRC:
  !     ORTHO.SRC(4), PAT0.SRC(4), PAT11.SRC(3), PAT12.SRC(3),
  !     PAT14.SRC(4), PAT2.SRC(6), PROJECT.SRC(4) and ZXPTH.SRC(3).
  !     PATH.SRC is belongs to RXNCOR group.
  !
contains

  SUBROUTINE PATHC(X,Y,Z,XCOMP,YCOMP,ZCOMP,WMAIN,COMLYN,COMLEN)
    !
    ! *** CALCULATION OF REACTION PATH EMPLOYING LINEAR CONSTRAINTS
    ! *** ON CROSS SECTIONS BETWEEN REACTANTS AND PRODUCTS.
    ! *** DEVELOPED BY RYSZARD CZERMINSKI AND RON ELBER.
    ! *** ACKNOWLEDGEMNT APPRECIATED (CALL 312-996-4732) TO FIND
    ! *** REFERENCES (IF ANY)
    !
    ! *** THE SUBROUTINE CAN BE COPIED AND USED BY ANYONE.
    ! *** HOWEVER IF YOU FOUND A NEW WAY OF PRODUCING GOLD
    ! *** WHILE USING THIS SUBROUTINE WE SHALL GLADLY SHARE IT WITH YOU.
    !
    ! THIS SUBROUTINE OPTIMIZED AN INITIAL GUESS FOR THE MINIMIUM ENERGY
    ! PATH TO SATISFY A MINIMUM ENERGY CRITERION. GIVEN AN
    ! INTERMEDIATE POINT BETWEEN TWO STATES OF INTEREST
    ! THIS ROUTINE WILL FIND A MINIMUM ENERGY
    ! SUBJECTED TO THE CONSTRAINTS ON AN INTERMDIATE CROSS_SECTION
    ! AND ON CENTER OF
    ! MASS TRANSLATION AND OVERALL ROTATION.
    !
    ! mode = 1 copy path coordinates to the main set
    ! mode = 2 search for minimum energy path using constrained minimization
    ! mode = 3 simulated annealing search for minimum energy path
    ! mode = 4 stepest descent search for minimum energy path
    !          from transition state to the minimum
    !
    use chm_kinds
    use exfunc
    use dimens_fcm
    use number
    !
    use memory
    use psf
    use bases_fcm
    use stream
    use string
    implicit none
    !
    integer,allocatable,dimension(:) :: ATOMPR
    real(chm_real),allocatable,dimension(:) :: RW
    real(chm_real),allocatable,dimension(:) :: RW0
    real(chm_real),allocatable,dimension(:) :: AUX
    integer,allocatable,dimension(:) :: FLAGS
    real(chm_real),allocatable,dimension(:) :: DC
    real(chm_real),allocatable,dimension(:) :: DS
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER NSTEP,NPATH
    INTEGER CCRD,UNIT,SAVP,SUNIT,SAV,SOLD
    INTEGER IFPEN,IPR
    real(chm_real) TOLC,TOLG,DFPRED,SCALE
    real(chm_real) FCNC,FCNI,FCNS,DFAC
    real(chm_real) :: X(:), Y(:), Z(:), WMAIN(*)
    real(chm_real) :: XCOMP(:), YCOMP(:), ZCOMP(:)
    INTEGER NGRID,IPATH,MODE,NDIM,NSTR
    !
#if KEY_RXNCOR==0
    CALL WRNDIE(-1,'<PATH>','PATH code is not compiled.')
    return
  end SUBROUTINE PATHC
#else /**/
    !
    IF(PRNLEV >= 3) WRITE(OUTU,*)
    NDIM   = 3*NATOM
    TOLC   = GTRMF(COMLYN,COMLEN,'TOLC',PT001)
    TOLG   = GTRMF(COMLYN,COMLEN,'TOLG',PT001)**2 *NDIM
    DFAC   = GTRMF(COMLYN,COMLEN,'DFAC',HALF)
    DFPRED = GTRMF(COMLYN,COMLEN,'DFPR',PTONE)
    !      DFPRED = GTRMF(COMLYN,COMLEN,'STEP',PT01)*NDIM
    FCNC   = GTRMF(COMLYN,COMLEN,'FCNC',HUNDRD)
    FCNI   = GTRMF(COMLYN,COMLEN,'FCNI',HUNDRD)
    FCNS   = GTRMF(COMLYN,COMLEN,'FCNS',HUNDRD)
    SCALE  = GTRMF(COMLYN,COMLEN,'SVEL',ONE)
    !C       TMPR   = GTRMF(COMLYN,COMLEN,'TEMP',THRHUN)
    !
    CCRD   = GTRMI(COMLYN,COMLEN,'CCRD',0)
    IFPEN  = GTRMI(COMLYN,COMLEN,'IFPE',2)
    IPATH  = GTRMI(COMLYN,COMLEN,'IPATH',0)
    IPR    = GTRMI(COMLYN,COMLEN,'IPRI',0)
    IF(PRNLEV < 2) IPR=-1
    NSTEP  = GTRMI(COMLYN,COMLEN,'NSTE',500)
    MODE   = GTRMI(COMLYN,COMLEN,'MODE',1)
    NPATH  = GTRMI(COMLYN,COMLEN,'NPRI',1)
    IF(NPATH <= 0) NPATH=1
    NSTR   = GTRMI(COMLYN,COMLEN,'NSTR',1)
    SAV    = GTRMI(COMLYN,COMLEN,'REST',0)
    SAVP   = GTRMI(COMLYN,COMLEN,'SAVP',0)
    SUNIT  = GTRMI(COMLYN,COMLEN,'SUNI',31)
    SOLD   = GTRMI(COMLYN,COMLEN,'SOLD',32)
    UNIT   = GTRMI(COMLYN,COMLEN,'UNIT',30)
    IF(PRNLEV >= 3) WRITE(OUTU,1) NSTEP,DFPRED/NDIM,TOLG,NPATH, &
         FCNS,UNIT,CCRD,SUNIT,SOLD,SAV,SAVP,MODE,TOLC
1   FORMAT(' PATH : ' &
         /' NSTEP =',I10,  '  STEP=',E10.2,'  TOLG=',E10.2,'  NPRINT=',I10 &
         /' FCNS  =',E10.2,'  UNIT=',I10,  '  CCRD=',I10,  '  SUNIT =',I10 &
         /' SOLD  =',I10,  '  REST=',I10,  ' SAVPR=',I10,  '   MODE =',I10 &
         /' TOLC  =',E10.2)
    !
    ! DS - THE DERIVATIVES OF THE TARGET FUNCTION
    !
    call chmalloc('path.src','PATHC','ATOMPR',2*NATOM,intg=ATOMPR)
    call chmalloc('path.src','PATHC','RW',NDIM,crl=RW)
    call chmalloc('path.src','PATHC','RW0',NDIM,crl=RW0)
    call chmalloc('path.src','PATHC','AUX',NDIM,crl=AUX)
    call chmalloc('path.src','PATHC','FLAGS',NATOM,intg=FLAGS)
    call chmalloc('path.src','PATHC','DC',7*NDIM,crl=DC)
    call chmalloc('path.src','PATHC','DS',NDIM,crl=DS)
    !
    ! FLAGS(I) [VECTOR OF INTEGER*2] IS SET TO 1 IF THE ATOM WAS
    ! SELECTED AND TO 0 IF IT WAS NOT. IF NO 'SELE' IS SPECIFIED
    ! ALL FLAGS ARE SET TO 1
    !

    CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
         .TRUE., .TRUE.,.TRUE.,.TRUE., &
         0,[zero],[zero],[zero],[zero],[zero],[zero])
    if(mode < 1 .or. mode > 4) then
       IF(WRNLEV >= 3) write(outu,'(A,I5)') &
            ' pat0> mode out off range(1-4); mode = ',mode
       CALL WRNDIE(-4,'<PAT0>','error')
    endif
    goto(100,200,300,400), mode
100 continue
    IF(PRNLEV >= 3) write(outu,*) &
         ' path> copy coordinates to the main set'

    !.....copy path coordinates to the main set

    call PAT11(X,Y,Z,RW,NDIM,UNIT,NSTR)
    goto 900
    !
    !.....search for minimum energy path with direct minimization
    !
200 continue
    IF(PRNLEV >= 3) write(outu,'(a,i5)') &
         ' path> direct minimization; ifpen=',ifpen
    call PAT12(COMLYN,COMLEN,X,Y,Z,XCOMP,YCOMP,ZCOMP,DS &
         ,ATOMPR,RW,RW0,DC,AUX &
         ,FCNC,FCNI,FCNS,NSTEP &
         ,DFPRED,NGRID,TOLG,TOLC,NDIM,NPATH,UNIT,SOLD,SAV,SAVP &
         ,FLAGS,SCALE,WMAIN,IFPEN,IPR,DFAC,NSTR,IPATH)
    goto 900
    !
    !.....simulated annealing search for minimum energy path
    !
300 continue
    IF(WRNLEV >= 3) write(outu,*) ' path> simulated annealing'
    IF(WRNLEV >= 3) write(outu,*) ' not implemented'
    return
    ! added x in memory allocation names so that they dont
    ! show up as real names in conversion process statistics /MH09/
    !C      WORK = ALLxHP(IREAL8(NDIM))
    !C      DS1  = ALLxHP(IREAL8(NDIM))
    !C      MNEW = ALLxHP(IREAL8(NDIM))
    !C      CALL ANNEAL(
    !C     &  COMLYN,COMLEN,PAT2,NSIZ,TOLG,MAXFN,DFPRED,HExAP(RW),STxACK(DS),
    !C     &  HExAP(DS1),S,HExAP(WORK),IER,NPATH,X,Y,Z,XCOMP,YCOMP,ZCOMP,
    !C     &  FCNS,NDIM,NGRID,UNIT,SUNIT,SAV,SAVP,HExAP(RW0),HExAP(FLAGS),
    !C     &  DT,SCALE,TMPR,HExAP(MNEW),IFPEN,IPR,MCON,HExAP(DC),
    !C     &  FCNC,FCNI,FCNS,HExAP(AUX),DFAC)
    !C      CALL FRxEHP(WORK,IREAL8(NDIM))
    !C      CALL FRxEHP(DS1 ,IREAL8(NDIM))
    !C      CALL FRxEHP(MNEW,IREAL8(NDIM))
    !C      goto 900
    !
    !.....stepest descent search for minimum energy path
    !     from transition state to the minimum
    !
400 continue
    IF(PRNLEV >= 3) write(outu,*) ' path> stepest descent search'

    call PAT14(X,Y,Z,XCOMP,YCOMP,ZCOMP,RW, &
         WMAIN,TOLG,DFAC,DFPRED,ATOMPR,UNIT,IPR,NSTR,IPATH)
    !
    !.....FREE THE ALLOCATED SPACE
    !
900 continue
    call chmdealloc('path.src','PATHC','ATOMPR',NATOM,intg=ATOMPR)
    call chmdealloc('path.src','PATHC','RW',NDIM,crl=RW)
    call chmdealloc('path.src','PATHC','RW0',NDIM,crl=RW0)
    call chmdealloc('path.src','PATHC','AUX',NDIM,crl=AUX)
    call chmdealloc('path.src','PATHC','DC',7*NDIM,crl=DC)
    call chmdealloc('path.src','PATHC','FLAGS',NATOM,intg=FLAGS)
    call chmdealloc('path.src','PATHC','DS',NDIM,crl=DS)
    !
    ! CLOSE FILES OPEN IN PATH : UNIT SUNIT SOLD
    !
    IF(PRNLEV >= 3) WRITE(OUTU,*)' CLOSING FILES OPENED IN PATH :'
    IF(PRNLEV >= 3) WRITE(OUTU,'(3(A,I5))') ' UNIT = ',UNIT, &
         '  SUNIT = ',SUNIT,'  SOLD = ',SOLD
    IF(IOLEV > 0) THEN
       CLOSE (UNIT)
       CLOSE (SUNIT)
       CLOSE (SOLD)
    ENDIF
    RETURN
  END subroutine pathc

  ! GETTING COORDINATES FROM A SET OF CHARMM FILES
  !
  SUBROUTINE RCRD(RW,NAT,ERROR)
    !
    use chm_kinds
    use machio,only:vopen
    use stream
    use parallel,only: psnd4,psnd8
    implicit none
    !
    real(chm_real) RW(*)
    real(chm_real) X,Y,Z
    INTEGER NAT,UN,NT1,ICOUNT,I
    LOGICAL ERROR
    CHARACTER(len=80) FILE
    CHARACTER(len=4) ACCESS
    CHARACTER(len=9) FORM
#if KEY_UNUSED==1 /*ltl_unused*/
    !      INTEGER LTL
    !      EXTERNAL LTL
#endif /* (ltl_unused)*/
    !
    IF(IOLEV > 0) THEN
       !
       ACCESS='READ'
       FORM='FORMATTED'
       UN=40
       IF(PRNLEV >= 3) THEN
          WRITE(OUTU,*)
          WRITE(OUTU,*)' *** READING INITIAL PATH USING CHARMM'
          WRITE(OUTU,*)'        COORDINATE FILES ***'
          WRITE(OUTU,*)' YOU MUST ENTER THE NAME OF THE FILE'
          WRITE(OUTU,*)' ON A SEPARATE LINE. END '
          WRITE(OUTU,*)' BY END (THUS FILE NAME END IS ILLEGAL)'
          WRITE(OUTU,*)' *** NOTE THAT UNIT 40 IS USED *** '
          WRITE(OUTU,*)
       ENDIF
       ICOUNT = 0
1      CONTINUE
       ICOUNT = ICOUNT + 1
       !      WRITE(OUTU,*)' READING STRUCTURE NUMBER(1),ICOUNT= ',ICOUNT
       READ(ISTRM,2) FILE
       !      WRITE(OUTU,*)' READING STRUCTURE NUMBER(2),ICOUNT= ',ICOUNT
2      FORMAT(80A)
       IF(PRNLEV >= 3) WRITE(OUTU,*)'PATH RCRD> ',FILE(1:50)
       IF (FILE(1:3) == 'END') GO TO 5
       CALL VOPEN(UN,FILE,FORM,ACCESS,ERROR,0)
       IF(ERROR) THEN
          IF(WRNLEV >= 3) WRITE(OUTU,'(A,I5)') &
               ' PROBLEMS IN OPENING FILE ',FILE
          GOTO 950
       ENDIF
       !
       ! LTL : READ THE TITLE LINES
       !
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,I5)') &
            ' READING STRUCTURE NUMBER ',ICOUNT
       READ(UN,*)NT1
       IF(NT1 /= NAT) THEN
          IF(WRNLEV >= 3) WRITE(OUTU,*)' NUMBER OF ATOMS DO NOT MATCH'
          IF(WRNLEV >= 3) WRITE(OUTU,'(2(A,I5))') &
               ' DEFINED = ',NAT,' READ = ',NT1
          IF(NT1 /= 0) THEN
             ERROR=.TRUE.
             GOTO 950
          ENDIF
       ENDIF
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,I5)')' THE NUMBER OF ATOMS IS ',NAT
       DO I = 1,NAT
          READ(UN,3) X,Y,Z
          ! READ(UN,3)FILE(1:50)
          ! READ(FILE(21:30),'(F10.5)')X
          ! READ(FILE(31:40),'(F10.5)')Y
          ! READ(FILE(41:50),'(F10.5)')Z
          RW(I)=X
          RW(NAT+I)=Y
          RW(2*NAT+I)=Z
          !3        FORMAT(50A)
3         FORMAT(20X,3F10.5)
       ENDDO
       CALL VCLOSE(UN,FILE,ERROR)
       GO TO 1
5      CONTINUE
       !
    ENDIF
    !
950 CONTINUE
#if KEY_PARALLEL==1
    CALL PSND4(ERROR,1)
    CALL PSND8(RW,NAT*3)
#endif 
    !
    RETURN
  END subroutine rcrd

  !..... READING THE TITLE LINES
#if KEY_UNUSED==1 /*ltl_unused*/
  !
  INTEGER FUNCTION LTL(UN)
    !
    use chm_kinds
    use stream
    implicit none
    !
    CHARACTER(len=80) C
    INTEGER UN,NTITLE
    !
    NTITLE=0
10  CONTINUE
    READ(UN,1)C
1   FORMAT(A80)
    IF (C(1:1) == '*') THEN
       NTITLE=NTITLE+1
       IF(PRNLEV >= 3) WRITE(OUTU,*) C(1:79)
       GO TO 10
    END  IF
    BACKSPACE UN
    !     REWIND (UN)
    !     DO 2 I=1,NTITLE
    !       READ(UN,1)C
    !2     CONTINUE
    LTL = NTITLE
    RETURN
  END function ltl
#endif /* (ltl_unused)*/

  SUBROUTINE PAT2(NSIZ, RW, TARGET, DS, NPATH, X, Y, Z, &
       XCOMP, YCOMP, ZCOMP, NDIM, NPATH1, FLAGS, &
       IFPEN, IPR, MCON, DC, FC1, FC2, FC3, AUX, &
       DFAC, R1, R2, DST, DSE, CONSTR, RCN)
    !
    use chm_kinds
    use chm_types

    use dimens_fcm
    use number
    !
    use bases_fcm
    use deriv
    use energym
    use psf
    use stream
    implicit none
    !
    real(chm_real) :: DS(*), X(:), Y(:), Z(:), XCOMP(:), YCOMP(:), ZCOMP(:)
    real(chm_real) RW(*),AUX(*),TARGET,RCN
    INTEGER MCON,NDIM,NSIZ
    real(chm_real) DC(MCON,NDIM),FC1,FC2,FC3,DFAC
    real(chm_real) TX,TY,TZ,TXY,TXZ,TYZ,PENI,PENC
    real(chm_real) A,AX,AY,AZ,DSX,DSY,DSZ,R,R1,R2,S,CONSTR
    real(chm_real) DSC,DSE,DSI,DSL,DSM,DST,EPS
    real(chm_real)  XL,YL,ZL,XYL,XZL,YZL,CL
    INTEGER IFPEN
    INTEGER I,IPR,NPATH,NPATH1
    INTEGER IY,IZ
    INTEGER FLAGS(*)

    !
    DATA  EPS,PENC,PENI/1.D-30,999.999D0,999.999D0/
    !
    IF(MCON /= 7) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,I5)') ' PAT2> ERROR; MCON=',MCON
       CALL WRNDIE(-4,'<PAT2>','error')
    ENDIF
    if(dfac < -PT0001 .or. (dfac-ONE) > PT0001)then
       !      if(dfac < ZERO .or. dfac > ONE) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(a,f10.5)') &
            ' pat2> error1 ; dfac = ',dfac
       CALL WRNDIE(-4,'<PAT2>','error')
    endif
    !
    ! COPY RW(IX),RW(IY),RW(IZ) TO X,Y,Z TO ALLOW THE USE OF
    ! FAST ENERGY ROUTINE
    !
    DO I=1,NATOM
       IY = I  + NATOM
       IZ = IY + NATOM
       X(I)    = RW(I )
       Y(I)    = RW(IY)
       Z(I)    = RW(IZ)
    ENDDO
    !
    ! ENERGY CALL FOR THE INTERMEDIATE POINT
    !
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
#if KEY_PARALLEL==1
    CALL VDGBR(DX,DY,DZ,1)
#endif 

    TARGET = EPROP(EPOT)

    IF (((NPATH1 == 1).OR.(NPATH1/NPATH*NPATH == NPATH1)) .AND. &
         IPR > 0) THEN
       WRITE(OUTU,*)'-------------------------------------'
       WRITE(OUTU,*)' INTERMEDIATE STRUCTURE NUMBER : '
       CALL PRINTE(OUTU, EPROP, ETERM, 'PATH', 'MIN', .TRUE., &
            NPATH1, ZERO, ZERO, .TRUE.)
       WRITE(OUTU,*)
    ENDIF
    !
    ! ENERGY CONTRIBUTION TO THE GRADIENT
    !
    DSE = ZERO
    DO I =1,NATOM
       IY = I  + NATOM
       IZ = IY + NATOM
       DS(I ) = DX(I)
       DS(IY) = DY(I)
       DS(IZ) = DZ(I)
       DSE    = DSE + DX(I)*DX(I) + DY(I)*DY(I) + DZ(I)*DZ(I)
    ENDDO

    !
    ! CALCULATE THE DISTANCE OF THE INTERMEDIATE POINT
    ! FROM THE INITIAL MAIN (IN ARRAY AUX) & COMPARISON COORDINATE SETS
    !
    R1 = ZERO
    R2 = ZERO
    S  = ZERO

    if(ipr > 5) WRITE(OUTU,*) ' pat2> 139> i,rw,aux,comp'
    DO I=1,NATOM
       IF (FLAGS(I) == 0) GOTO 140
       IY = I  + NATOM
       IZ = IY + NATOM
       R1 = R1 + (RW(I )-AUX(I ))**2 + (RW(IY)-AUX(IY))**2 &
            + (RW(IZ)-AUX(IZ))**2
       R2 = R2 + (RW(I )-XCOMP(I))**2 + (RW(IY)-YCOMP(I))**2 &
            + (RW(IZ)-ZCOMP(I))**2
       S  = S  + &
            (AUX(I )-XCOMP(I))*(RW(I ) - AUX(I ) - DFAC*(XCOMP(I) - AUX(I )))+ &
            (AUX(IY)-YCOMP(I))*(RW(IY) - AUX(IY) - DFAC*(YCOMP(I) - AUX(IY)))+ &
            (AUX(IZ)-ZCOMP(I))*(RW(IZ) - AUX(IZ) - DFAC*(ZCOMP(I) - AUX(IZ)))
       !@
       IF(IPR > 5 .AND. I <= 15) WRITE(OUTU,139) I,RW(I),RW(IY),RW(IZ), &
            aux(i),aux(iy),aux(iz),xcomp(i),ycomp(i),zcomp(i)
139    format(i3,9(1x,f6.3))
       !@
140    CONTINUE
    ENDDO
    CONSTR = S
    R1 = SQRT(R1)
    R2 = SQRT(R2)
    !@
    IF(IPR > 0) WRITE(OUTU,141) R1,R2,FC1,FC2,FC3,PENC,PENI
141 format(' pat2> r1,r2,fc1,fc2,fc3,penc,peni = '/7(1x,f8.2))
    !

    IF(IFPEN <= 0) GOTO 1400
    !
    !  DISTANCE CONSTRAINT IS EQUAL :  (0<DFAC<1)
    !
    !     IFPEN=0  projection on linear subspaces perpendicular to the
    !              vector = PRODUCTS - REACTANTS
    !
    !     IFPEN = 1 or 11  CON(7) = (R-Rp)*(AUX-COMP) (PLANES)
    !
    !     IFPEN = 2 or 12  CON(7) = (1-DFAC)*R1 - DFAC*R2
    !                             (i.e. R1/(R1+R2) = DFAC )
    !
    !     IFPEN = 3 or 13  CON(7) = ((1-DFAC)*R1)**2 - (DFAC*R2)**2
    !
    !     IFPEN = 2 or 3 ; SPHERICAL CONSTRAINTS ( SORT OF)
    !
    ! CALCULATE THE CHAIN CONTRIBUTION TO THE TARGET
    ! FUNCTION AND ITS DERIVATIVES
    !
    IF(IFPEN == 1.OR.IFPEN == 11.OR.IFPEN == 2.OR.IFPEN == 12) &
         S = (1-DFAC)*R1 - DFAC*R2
    !
    IF(IFPEN == 3 .OR. IFPEN == 13) S=((1-DFAC)*R1)**2-(DFAC*R2)**2
    !
    CONSTR = S
    PENC   = FC3*S*S
    TARGET = TARGET + PENC
    S = 2.D0 * FC3 * S
    IF(IFPEN == 2 .OR. IFPEN == 12) THEN
       A = (1.D0-DFAC)*S/(R1+EPS)
       R =    DFAC    *S/(R2+EPS)
    ELSEIF(IFPEN == 3 .OR. IFPEN == 13) THEN
       S = 2*S
       A = (1-DFAC)**2*S
       R =   DFAC  **2*S
    ENDIF
    DSC=0.
    DO I=1,NATOM
       IF (FLAGS(I) == 0) GOTO 170
       IY = I  + NATOM
       IZ = IY + NATOM
       IF(IFPEN == 1 .OR. IFPEN == 11) THEN
          AX = S*(AUX(I )-XCOMP(I))
          AY = S*(AUX(IY)-YCOMP(I))
          AZ = S*(AUX(IZ)-ZCOMP(I))
       ELSEIF(IFPEN == 2.OR.IFPEN == 12.OR.IFPEN == 3.OR.IFPEN == 13) &
            THEN
          AX = A*(RW(I ) - AUX(I )) - R*(RW(I ) - XCOMP(I))
          AY = A*(RW(IY) - AUX(IY)) - R*(RW(IY) - YCOMP(I))
          AZ = A*(RW(IZ) - AUX(IZ)) - R*(RW(IZ) - ZCOMP(I))
       ELSE
          IF(WRNLEV >= 2) WRITE(OUTU,'(A,I5)') &
               ' pat2 > error2 ; improper ifpen = ',ifpen
          CALL WRNDIE(-4,'<PAT2>','error')
       ENDIF
       DS(I ) = DS(I ) + AX
       DS(IY) = DS(IY) + AY
       DS(IZ) = DS(IZ) + AZ
       IF (IPR > 5) WRITE(OUTU,*)' IX IY IA DX DY DZ' &
            ,I,IY,IZ,DS(I),DS(IY),DS(IZ)
       DSC = DSC + AX*AX + AY*AY + AZ*AZ
170    CONTINUE
    ENDDO

    IF(IFPEN < 10) GOTO 1400
    !
    ! CALCULATING CENTER OF MASS FOR FLAGS /= 0
    !
    ! MOMENT OF MOMENTUM COMPONENTS  FOR FLAGS /= 0
    !
    !
    if(ipr > 0) write(outu,'(a,2i5)') &
         ' pat2> npath1,ifpen = ',npath1,ifpen
    if(ipr > 2) write(outu,*) 'i,iy,iz,x,y,z,m = '

    !     TM = 0.
    TX = ZERO
    TY = ZERO
    TZ = ZERO
    TXY= ZERO
    TXZ= ZERO
    TYZ= ZERO
    DO I=1,NATOM
       IY = I  + NATOM
       IZ = IY + NATOM

       IF(IPR > 2 .AND. I <= 5) THEN
          WRITE(OUTU,190) I,IY,IZ,RW(I),RW(IY),RW(IZ),AMASS(I)
       ENDIF
190    FORMAT(1x,3i3,4(1x,f9.3))

       IF (FLAGS(I) == 0) GOTO 200
       A  = AMASS(I)
       !     TM = TM + A
       TX  = TX  + A*(RW(I ) - AUX(I ))
       TY  = TY  + A*(RW(IY) - AUX(IY))
       TZ  = TZ  + A*(RW(IZ) - AUX(IZ))
       TYZ = TYZ + A*(AUX(IY)*RW(IZ) - AUX(IZ)*RW(IY))
       TXZ = TXZ + A*(AUX(IZ)*RW(I ) - AUX(I )*RW(IZ))
       TXY = TXY + A*(AUX(I )*RW(IY) - AUX(IY)*RW(I ))
200    CONTINUE
    ENDDO

    IF(IPR > 2) WRITE(OUTU,210) TX,TY,TZ
    IF(IPR > 2) WRITE(OUTU,220) TYZ,TXZ,TXY
210 format(' pat2> tx,ty,tz   : ',3e10.2)
220 format(' pat2> tyz,txz,txy: ',3e10.2)

    ! RIGID BODY CONSTRAINTS ARE EQUAL : TX,TY,TZ,TXY,TXZ,TYZ

    ! RIGID BODY PENALTY FUNCTION IS EQUAL

    PENI = FC1*(TX*TX+TY*TY+TZ*TZ) + FC2*(TXY*TXY+TXZ*TXZ+TYZ*TYZ)

    TARGET = TARGET + PENI

    if(mcon /= 7) then
       IF(WRNLEV >= 2) WRITE(OUTU,'(A,I5)') &
            ' pat2> error3;  mcon /= 7; mcon = ',mcon
       CALL WRNDIE(-4,'<PAT2>','error')
    endif
    !
    ! CALCULATING CENTER OF MASS AND RIGID ROTATION
    ! CONSTRAINTS COMPONENT TO THE GRADIENT
    !
    A   = FC1 + FC1
    AX  = TX *A
    AY  = TY *A
    AZ  = TZ *A
    A   = FC2 + FC2
    TXY = TXY*A
    TXZ = TXZ*A
    TYZ = TYZ*A
    DSI = 0.
    DSM = 0.
    DSL = 0.
    DO I=1,NATOM
       IY  = I  + NATOM
       IZ  = IY + NATOM
       IF(FLAGS(I) == 0) GOTO 1100
       A   = AMASS(I)
       DSM = DSM + A*A
       DSX = TXZ*AUX(IZ) - TXY*AUX(IY)
       DSY = TXY*AUX(I ) - TYZ*AUX(IZ)
       DSZ = TYZ*AUX(IY) - TXZ*AUX(I )
       DSL = DSL + A*A*(DSX*DSX + DSY*DSY + DSZ*DSZ)
       DSX = A*(AX + DSX)
       DSY = A*(AY + DSY)
       DSZ = A*(AZ + DSZ)
       DS(I ) = DS(I ) + DSX
       DS(IY) = DS(IY) + DSY
       DS(IZ) = DS(IZ) + DSZ
       DSI = DSI + DSX*DSX + DSY*DSY + DSZ*DSZ
1100   CONTINUE
    ENDDO
    DSM = DSM*(AX*AX + AY*AY +AZ*AZ)

1400 CONTINUE
    IF(IPR > 2) WRITE(OUTU,1410) TX,TY,TZ
    IF(IPR > 2) WRITE(OUTU,1420) TYZ,TXZ,TXY
    IF(IPR > 0) WRITE(OUTU,1430) FC1,FC2,FC3,PENC,PENI
1410 FORMAT(' pat2 1410> tx,ty,tz   : ',3e10.2)
1420 FORMAT(' pat2 1420> tyz,txz,txy: ',3e10.2)
1430 FORMAT(' pat2 1430> fc1.3,penci: ',5e10.2)
    !
    ! GRADIENT PROJECTION INTO THE LINER SUBSPACE ALLOWED BY THE CONSTRAINTS
    !
    IF(IFPEN <= 10) THEN
       IF(IFPEN <= 0) I = MCON
       IF(IFPEN > 0) I = MCON - 1
       !@DEBUG
       !@      IF(IPR > 2) WRITE(OUTU,1430) MCON,IFPEN,I
       !@ 1430 format(' pat2 1430> before project; mcon,ifpen,i = ',3i5)
       CALL PROJECT(DS,DC,MCON,I,NDIM)
       !@      IF(IPR > 2) WRITE(OUTU,1440) MCON,IFPEN,I
       !@ 1440 format(' pat2 1430> after  project; mcon,ifpen,i = ',3i5)
       !@
    ENDIF
    !
    ! THE GRADIENT OF THE TARGET FUNCTION (HOPEFULLY SMALL)
    !
    DST=0.
    DO I=1,NDIM
       DST=DST+DS(I)*DS(I)
    ENDDO
    !
    IF (NPATH1/NPATH*NPATH == NPATH1.OR.NPATH1 == 1) THEN
       R = (R1*R1+RCN-R2*R2)/(2*RCN)
       S   = 1./NDIM
       DST = 0.5*LOG10(DST*S+EPS)
       IF(PRNLEV >= 2) WRITE(OUTU,2100) &
            NPATH1,EPROP(EPOT),TARGET,DST,R1,R2,SQRT(RCN),R,DFAC
       IF(IPR > 0) THEN
          DSE = 0.5*LOG10(DSE*S+EPS)
          DSC = 0.5*LOG10(DSC*S+EPS)
          DSI = 0.5*LOG10(DSI*S+EPS)
          WRITE(OUTU,2200) TARGET,EPROP(EPOT),PENC,PENI
          WRITE(OUTU,2300) DST,DSE,DSC,DSI
          WRITE(OUTU,2400) R1,R2,R1-R2,DSC
          WRITE(OUTU,2600) TX ,TY ,TZ ,0.5*LOG10(DSM*S+EPS)
          WRITE(OUTU,2700) TXY,TXZ,TYZ,0.5*LOG10(DSL*S+EPS)
          WRITE(OUTU,*)'***********************************'
          WRITE(OUTU,*)
          WRITE(OUTU,*)
       ENDIF
    ENDIF
    !
    ! CHANGE OF FORMAT TO GET SOMETHING FROM PROTEINS
    ! (f7.2 -> f8.1) RE.
    !
2100 FORMAT(' pat2>',i5,8(1x,f8.1))
2200 FORMAT(/' TARGET,E,PENC,PENI =',4(1X,F9.3))
2300 FORMAT( ' LOG(G)             =',4(1X,F9.3))
2400 FORMAT( ' DISTANCES & LOG(G) =',4(1X,F9.3))
2600 FORMAT( ' C.O.M.    & LOG(G) =',4(1X,F9.3))
2700 FORMAT( ' LZ,LY,LX  & LOG(G) =',4(1X,F9.3))
    IF(IPR == 0 .OR. IFPEN <= 10) RETURN
    XL  = LOG10(ABS(TX)+EPS)
    YL  = LOG10(ABS(TY)+EPS)
    ZL  = LOG10(ABS(TZ)+EPS)
    XYL = LOG10(ABS(TXY)+EPS)
    XZL = LOG10(ABS(TXZ)+EPS)
    YZL = LOG10(ABS(TYZ)+EPS)
    CL  = LOG10(ABS(CONSTR)+EPS)
    A = MAX(XL,YL)
    A = MAX(A,ZL)
    A = MAX(A,XYL)
    A = MAX(A,XZL)
    A = MAX(A,YZL)
    A = MAX(A,CL)
    IF(PRNLEV >= 2) WRITE(OUTU,2800) XL,YL,ZL,XYL,XZL,YZL,CL
2800 FORMAT( ' LOG(CONSTR)= ',7(1x,f6.2))
2900 FORMAT( ' ?CONSTR= ',7(1x,f9.6))
    IF(A < -9.) RETURN
    IF(PRNLEV >= 2) WRITE(OUTU,2800) &
         LOG10(ABS(TX)),LOG10(ABS(TY)),LOG10(ABS(TZ)),LOG10(ABS(TXY)), &
         LOG10(ABS(TXZ)),LOG10(ABS(TYZ)),LOG10(ABS(CONSTR))
    IF(PRNLEV >= 2) WRITE(OUTU,2900) TX ,TY ,TZ ,TXY,TXZ,TYZ,CONSTR
    RETURN
    !.....PAT2
  END subroutine pat2

  SUBROUTINE PAT11(X,Y,Z,RW,NDIM,UNIT,NSTR)
    !
    use chm_kinds
    use dimens_fcm
    !
    use psf
    use stream
    use parallel,only: psnd8
    implicit none
    !
    real(chm_real)  X(*),Y(*),Z(*),RW(*)
    INTEGER NDIM,NSTR,UNIT
    INTEGER I,IY,IZ,J
    !
    IF(IOLEV > 0) THEN
       REWIND (UNIT)

       IF(PRNLEV >= 2) WRITE(OUTU,400) NSTR,UNIT,NDIM
400    FORMAT(/' COPYING PATH COORDINATES TO MAIN; NSTR,UNIT,NDIM= ',3I5)
       DO I=1,NSTR
          READ(UNIT,END=900)(RW(J),J=1,NDIM)
       ENDDO
       NATOM = NDIM/3
       DO I=1,NATOM
          IY = I  + NATOM
          IZ = IY + NATOM
          X(I)=RW(I)
          Y(I)=RW(IY)
          Z(I)=RW(IZ)
       ENDDO
       GOTO 950
900    CONTINUE
       IF(WRNLEV >= 2) WRITE(OUTU,*)' NSIZ TOO LARGE, CHECK YOUR NGRID'
       IF(WRNLEV >= 2) WRITE(OUTU,*)' GETTING OUT .. NOTHING DONE'
    ENDIF
    !
950 CONTINUE
#if KEY_PARALLEL==1
    CALL PSND8(X,NATOM)
    CALL PSND8(Y,NATOM)
    CALL PSND8(Z,NATOM)
#endif 
    !
    RETURN
  END subroutine pat11

  SUBROUTINE PAT12(COMLYN,COMLEN,X,Y,Z,XCOMP,YCOMP,ZCOMP,DS &
       ,ATOMPR,RW,RW0,DC,AUX,FCNC,FCNI,FCNS,MAXFN &
       ,DFPRED,NGRID,TOLG,TOLC,NDIM,NPATH,UNIT,SOLD,SAV,SAVP &
       ,FLAGS,SCALE,WMAIN,IFPEN,IPR,DFAC,NSTR,IPATH)
    !
    !.....search for minimum energy path with direct minimization
    !
    use chm_kinds
    use dimens_fcm
    use memory
    use number
    use energym
    use psf
    use select
    use stream
    use corsubs,only:rotls1
    use parallel,only: psnd8
    implicit none
    !
    real(chm_real) :: XCOMP(:), YCOMP(:), ZCOMP(:), RW0(*),AUX(*),RCN
    real(chm_real) :: X(:), Y(:), Z(:), RW(*),DS(*),DFPRED
    INTEGER NDIM
    real(chm_real) DC(7,NDIM)
    real(chm_real) WMAIN(*)
    real(chm_real) TOLG,ACCT,S,SCALE
    real(chm_real) FCNC,FCNI,FCNS,DFAC,FCNST
    real(chm_real) TARGET,R1,R2,DSE,DST,CONSTR,EPS,TOLC
    real(chm_real) PENS,PSTEP,PSTEP0,DIST0
    real(chm_real), allocatable, dimension(:) :: work
    INTEGER NPATH,NPATH1,IREP,ISTR
    INTEGER MAXFN,IER,NGRID,NSIZ,I,J,NSTR
    INTEGER UNIT,SOLD,SAV,SAVP,IPATH
    INTEGER COMLEN
    INTEGER IFPEN,IPR,ATOMPR(2,*)
    INTEGER FLAGS(*)
    !C      real(chm_real)    TIME(2),TIM
    LOGICAL ERROR
    CHARACTER(len=*) COMLYN
    INTEGER MCON,IY,IZ
    DATA EPS/1.D-30/
    DATA MCON/7/
    NPATH1=0
    NSIZ=NDIM
    PENS=1.D0
    !
    !  NSTR  >=  3   -  path is constructed
    !  NSTR  =   2   -  one intermediate point is calculated
    !  NSTR   <= 0 .OR. =1  is illegal !!!

    call chmalloc('path.src','PAT12','WORK',NSIZ*6,crl=work)
    IF(NSTR <= 1) NSTR=2
    IF(NSTR >= 3) PSTEP0 = ONE/(NSTR-1)
    IF(NSTR == 2) PSTEP0 = HALF

    IF(IOLEV > 0) REWIND (UNIT)
    !
    DO I=1,NATOM
       FLAGS(I)=1
       ATOMPR(1,I) = I
       ATOMPR(2,I) = I
       WORK(I) = AMASS(I)
    ENDDO
    CALL SELCTA(COMLYN,COMLEN,FLAGS,X,Y,Z,WMAIN,.TRUE.)
    !
    ! LPRINT = IPR > 0 &  LNOROT = .FALSE.
    !
    CALL ROTLS1(XCOMP,YCOMP,ZCOMP,X,Y,Z,NATOM,ATOMPR,NATOM, &
         WORK,(IPR > 0),.FALSE.)
    RCN = 0
    DO I=1,NATOM
       IY = I  + NATOM
       IZ = IY + NATOM
       AUX(I ) = X(I)
       AUX(IY) = Y(I)
       AUX(IZ) = Z(I)
       R1 = XCOMP(I) - X(I)
       R2 = YCOMP(I) - Y(I)
       S  = ZCOMP(I) - Z(I)
       RW0(I ) = PSTEP0*R1
       RW0(IY) = PSTEP0*R2
       RW0(IZ) = PSTEP0*S
       RCN     = RCN + R1*R1 + R2*R2 + S*S
    ENDDO
    ! RCN - squared distance between PRODUCTS and REACTANTS
    IF(RCN < 0.01) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,'(A,E10.2)') &
            ' pat12> distance too small; dist = ',sqrt(rcn)
       CALL WRNDIE(-4,'<PAT12>','error')
    ENDIF
    DIST0 = SQRT(RCN)

    ! GENERATION AND ORTHOGONALIZATION OF CONSTRAINTS DERIV. - DC(1..MCON)

    CALL GENDC(DC,MCON,1,NATOM,AMASS,X,Y,Z,XCOMP,YCOMP,ZCOMP,WORK)
    CALL ORTHO1P(DC,MCON,1,NSIZ,WORK,IPR)

    ! Projection of the reaction coordinate step into constraints
    ! subspace (only 1-6 !!!)
    !     CALL PROJECT(RW0,DC,MCON,6,NDIM)

    IF (SAV > 0) THEN
       !
       ! RESTART OPTION : READING PATH FROM A BINARY SAVE FILE
       !                  ERRORS ARE NOT CHECKED !!!
       !
       IF(PRNLEV >= 2) WRITE(OUTU,*)' READING RW .... '
       IF(IOLEV > 0) THEN
          REWIND (SOLD)
          READ(SOLD) (RW(I),I=1,NSIZ)
       ENDIF
#if KEY_PARALLEL==1
       CALL PSND8(RW,NSIZ)
#endif 
    ELSE  IF (SAV == 0) THEN
       !
       ! USING STRAIGHT LINE INTERPOLATION FOR INITIAL GUESS
       !
       ! RW0 = PSTEP0*(PRODUCTS - REACTANTS) ; zero-order reaction coordinate

       DO I=1,NATOM
          IY = I  + NATOM
          IZ = IY + NATOM
          RW(I ) = X(I)
          RW(IY) = Y(I)
          RW(IZ) = Z(I)
       ENDDO
    ELSE  IF (SAV < 0) THEN
       !
       ! READING INITIAL GUESS FROM SEVERAL CHARMM COORDINATE FILES
       !
       ERROR=.FALSE.
       CALL RCRD(RW,NATOM,ERROR)
       IF (ERROR .AND. WRNLEV >= 2) THEN
          WRITE(OUTU,*)' *** ERROR WHILE READING CHARMM'
          WRITE(OUTU,*)' *** COORDINATE FILE ... RETURN'
          WRITE(OUTU,*)' *** NOTHING DONE'
          RETURN
       ENDIF

       !...ORIENTING EXTERNAL INITIAL GUESS

       DO I=1,NATOM
          WORK(I) = AMASS(I)
       ENDDO
       CALL ROTLS1(XCOMP,YCOMP,ZCOMP,RW(1),RW(1+NATOM),RW(1+2*NATOM), &
            NATOM,ATOMPR,NATOM,WORK,(IPR > 0),.FALSE.)
       DO I=1,NATOM
          IY = I  + NATOM
          IZ = IY + NATOM
          WORK(I ) = RW(I ) - (X(I)+RW0(I))
          WORK(IY) = RW(IY) - (Y(I)+RW0(IY))
          WORK(IZ) = RW(IZ) - (Z(I)+RW0(IZ))
       ENDDO

       !.....PROJECTING OUT UNDESIRED COMPONENTS FROM INITIAL GUESS

       CALL PROJECT(WORK,DC,MCON,7,NDIM)
       DO I=1,NATOM
          IY = I  + NATOM
          IZ = IY + NATOM
          RW(I ) = X(I)+ WORK(I )
          RW(IY) = Y(I)+ WORK(IY)
          RW(IZ) = Z(I)+ WORK(IZ)
       ENDDO
    ENDIF

    IF(IOLEV > 0) WRITE(IPATH,50)
50  FORMAT(' DFAC,EN,TARGET,R1,R2,DST,DSE,LOG(DFAC-Q)')
    !
    ! CALL UPDATE WITH X,Y,Z (only once !!!)
    !
    CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
         .TRUE.,.TRUE.,.TRUE.,.TRUE., &
         0,[zero],[zero],[zero],[zero],[zero],[zero])
    !
    ! ENERGY CALL FOR THE FIRST POINT (DFAC=0)
    !
    DFAC=0
    CALL PAT2 &
         (NSIZ,RW,TARGET,DS,NPATH,X,Y,Z,XCOMP,YCOMP,ZCOMP,NDIM,NPATH1, &
         FLAGS,IFPEN,IPR,MCON,DC,FCNC,FCNI,FCNS,AUX,DFAC,R1,R2,DST,DSE, &
         CONSTR,RCN)
    !
    IF(IOLEV > 0) WRITE(IPATH,200) &
         DFAC,EPROP(EPOT),TARGET,R1,R2,DST,DSE,LOG10(EPS)
    IF(IOLEV > 0) WRITE(UNIT)(RW(J),J=1,NSIZ)
    !C      s = etime(time)
    !C      tim = time(1)
    !C      NCALL = 0
    !
    !.....PATH LOOP ....
    !
    DO ISTR=2,NSTR
       !
       DFAC = PSTEP0*(ISTR-1)
       IF(IFPEN < 0) THEN
          IF(MOD(ISTR,IFPEN) == 0) THEN
             DO I=1,NATOM
                WORK(I)=AMASS(I)
             ENDDO
             CALL ROTLS1(XCOMP,YCOMP,ZCOMP,X,Y,Z,NATOM,ATOMPR,NATOM, &
                  WORK,(IPR > 0),.FALSE.)
             S = 0
             DO I=1,NATOM
                S = S +(X(I)-XCOMP(I))**2+(Y(I)-YCOMP(I))**2 &
                     +(Z(I)-ZCOMP(I))**2
             ENDDO
             IF(S < EPS) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,66) S
                GOTO 5010
             ENDIF
66           format(/' pat12>66> distance too small ; dist =',e10.2)
             pstep = pstep0*dist0/sqrt(s)
             DO I=1,NATOM
                IY = I  + NATOM
                IZ = IY + NATOM
                RW0(I ) = PSTEP*(XCOMP(I) - X(I))
                RW0(IY) = PSTEP*(YCOMP(I) - Y(I))
                RW0(IZ) = PSTEP*(ZCOMP(I) - Z(I))
             ENDDO

             ! GENERATION AND ORTHOGONALIZATION OF CONSTRAINTS DERIV. - DC(4..MCON)

             CALL GENDC(DC,MCON,4,NATOM,AMASS,X,Y,Z,XCOMP,YCOMP,ZCOMP,WORK)
             CALL ORTHO1P(DC,MCON,4,NSIZ,WORK,IPR)
             ! Projection of the reaction coordinate step into constraints
             ! subspace
             CALL PROJECT(RW0,DC,MCON,6,NDIM)
          ENDIF
       ENDIF

       DO I=1,NATOM
          IY = I  + NATOM
          IZ = IY + NATOM
          RW(I ) = RW(I ) + RW0(I)
          RW(IY) = RW(IY) + RW0(IY)
          RW(IZ) = RW(IZ) + RW0(IZ)
       ENDDO
       !
       ! DO IT !!!
       !
       ! MINIMIZE THE PATH DIRECTLY, ZXPTH WILL BE MODIFIED FIRST FOR
       ! THE CHANGES.
       !
       if(ipr > 0) write(outu,*) ' before zxpth> dfac =',dfac

       FCNST = FCNS/PENS
       ACCT  = 1000*TOLG
       NPATH1=0
       IREP  = 0
90     CONTINUE
       IREP = IREP + 1
       IF(PRNLEV >= 2) WRITE(OUTU,100)
100    FORMAT(///' ### PATH ###' &
            /' NPATH1,ENERGY,TARGET,DST,R1,R2,RCN,Q,DFAC')
       CALL ZXPTH(NSIZ,ACCT,MAXFN,DFPRED,RW,DS,TARGET,WORK,IER, &
            NPATH,NPATH1,X,Y,Z,XCOMP,YCOMP,ZCOMP,NDIM,FLAGS,IFPEN,IPR, &
            MCON,DC,FCNC,FCNI,FCNST,AUX,DFAC,R1,R2,DST,DSE,CONSTR,RCN)
       IF(IFPEN > 0 .AND. &
            (IER == 0.OR.IER == 131) .AND. ABS(CONSTR) > TOLC) THEN
          IF(IREP < 6) THEN
             FCNST = 10*FCNST
             IF(ACCT > TOLG) ACCT = 0.1*ACCT
             IF(WRNLEV >= 2) WRITE(OUTU,110) CONSTR,TOLC,FCNST
             IF(IFPEN <= 0) then
                IF(WRNLEV >= 2) WRITE(OUTU,'(A,I5,1X,F10.5)') &
                     ' PAT12 ERROR> IFPEN,CONSTR=',IFPEN,CONSTR
                CALL WRNDIE(-4,'<PAT12>','error')
             ELSE
                S  = R1/(R1+R2)
             ENDIF
             IF(PRNLEV >= 2) WRITE(OUTU,120) DFAC,S,R1,R2
             GOTO 90
          ENDIF
       ENDIF
       !C      NCALL = NCALL + NPATH1
       PENS  = 1.D-06*(R1*R1 + R2*R2)**2
       !.....CALL PAT2 WITH IFPEN=11 TO CHECK QUALITY OF THE PROJECTION
       IF(IFPEN <= 0 .AND. IPR > 0) CALL PAT2 &
            (NSIZ,RW,TARGET,DS,NPATH,X,Y,Z,XCOMP,YCOMP,ZCOMP,NDIM,NPATH1, &
            FLAGS,11,IPR,MCON,DC,FCNC,FCNI,FCNS,AUX,DFAC,R1,R2,DST,DSE, &
            CONSTR,RCN)
       IF(ABS(CONSTR) > TOLC .AND. WRNLEV >= 2) WRITE(OUTU,130) &
            IFPEN,CONSTR,TOLC
       IF(ABS(CONSTR) > TOLC .AND. IFPEN <= 0) THEN
          CALL WRNDIE(-4,'<PAT12>','error')
       ENDIF
       IF(IPR > 0) WRITE(OUTU,131) (dfac-r1/(r1+r2)),constr,tolc
110    FORMAT( &
            /' pat12>warning; constr,tolc =',2(1x,e10.2) &
            /'       new trial with larger force constant FCNST = ',e10.2)
120    FORMAT( ' pat12> dfac,r1/(r1+r2),r1,r2 = ',4f10.4)
130    FORMAT(/' pat12> warning; ifpen,constr,tolc = ',i2,2e10.2)
131    FORMAT(/' pat12> dfac-r1/(r1+r2),constr,tolc=',3e10.2)

       if(ipr > 0) write(outu,*) ' after zxpth> ier=',ier
       !
       ! SUMMARIZED RESULTS (ERROR CHECKES AND GOOD BYES)
       !
       IF(DST > 0.) DST=0.5*LOG10(DST+EPS)
       IF(DSE > 0.) DSE=0.5*LOG10(DSE+EPS)
       IF(IFPEN <= 0) then
          S = (R1*R1 + RCN - R2*R2)/(2*RCN)
       else
          S  = R1/(R1+R2)
       endif
       IF(IOLEV > 0) WRITE(IPATH,200) &
            DFAC,EPROP(EPOT),TARGET,R1,R2,DST,DSE,LOG10(ABS(DFAC-S)+EPS)
200    FORMAT(' pat1> ',f5.3,4(1x,f6.2),1x,3(1x,f6.1))
       IF (IER == 0) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,300)
       ELSEIF (IER == 129) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,310)
       ELSEIF (IER == 130) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,320)
       ELSEIF (IER == 131) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,330)
       ELSEIF (IER == 132) THEN
          CALL WRNDIE(1,'PATH : ','MORE PROBLEMS ...')
       ELSE
          CALL WRNDIE(1,'PATH : ',' UNKNOWN IER VALUE')
       ENDIF
       !
300    FORMAT(1X,'PATH - GRADIENT CONVERGE')
310    FORMAT(1X,'PATH : PROBLEM IN GRADIENT OR STEP')
320    FORMAT(1X,'PATH : CANNOT MAKE FURTHER REDUCTION')
330    FORMAT(1X,'PATH : NUMBER OF STEP LIMIT')
       !
       IF (IER == 0 .OR. IER == 129 .OR. IER == 131) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,4900) ISTR,UNIT,NSIZ
          IF(IOLEV > 0) WRITE(UNIT)(RW(J),J=1,NSIZ)
       ENDIF
4900   FORMAT(/' WRITING COORDINATES : ISTR,UNIT,NSIZ =',3I5)
       !
       !
    ENDDO
    !C      s = etime(time)
    !
    ! END OF PATH LOOP...
    !
    ! ENERGY CALL FOR THE LAST POINT (DFAC=1)
    !
5010 CONTINUE
    DO I=1,NATOM
       IY = I  + NATOM
       IZ = IY + NATOM
       RW(I ) = XCOMP(I)
       RW(IY) = YCOMP(I)
       RW(IZ) = ZCOMP(I)
    ENDDO
    DFAC=1
    CALL PAT2 &
         (NSIZ,RW,TARGET,DS,NPATH,X,Y,Z,XCOMP,YCOMP,ZCOMP,NDIM,NPATH1, &
         FLAGS,IFPEN,IPR,MCON,DC,FCNC,FCNI,FCNS,AUX,DFAC,R1,R2,DST,DSE, &
         CONSTR,RCN)
    IF(IOLEV > 0) WRITE(IPATH,200) &
         DFAC,TARGET,TARGET,R1,R2,DST,DSE,LOG10(EPS)
    IF(IOLEV > 0) WRITE(UNIT)(RW(J),J=1,NSIZ)
    !C      tim=time(1)-tim
    !C      write(outu,5200) ncall,tim,tim/ncall
    !C 5200 format(/' pat12> ncall,time,time/ncall=',i10,1x,2e10.2)
    call chmdealloc('path.src','PAT12','WORK',NSIZ*6,crl=work)
    RETURN
    !.....PAT12 (search for minimum energy path with direct minimization)
  END subroutine pat12

  SUBROUTINE PAT14(X,Y,Z,XCOMP,YCOMP,ZCOMP,RW, &
       WMAIN,TOLG,DFAC,DFPRED,ATOMPR,UNIT,IPR,NSTR,IPATH)
    !
    !.....stepest descent search for minimum energy path
    !     from transition state to the minimum
    !
    use chm_kinds
    use chm_types

    use dimens_fcm
    use number
    use bases_fcm
    use deriv
    use energym
    use stream
    use psf
    use corsubs,only:rotls1
    use heurist,only:updeci
    implicit none
    !
    real(chm_real) :: XCOMP(:), YCOMP(:), ZCOMP(:)
    real(chm_real) :: X(:), Y(:), Z(:), RW(*)
    real(chm_real) WMAIN(*)
    real(chm_real) TOLG,DFAC,DFPRED,GG
    real(chm_real) R1,R2,DSE,DST,EPS
    real(chm_real) STEP,SOLD,EVERY
    INTEGER UNIT,IPATH,NSTR,IPR,ATOMPR(2,*)
    INTEGER I,J,IREP,ISTR,IY,JY,IZ,JZ
    DATA EPS/1.D-8/
    !
    STEP = DFAC
    IF(IOLEV > 0) THEN
       REWIND (IPATH)
       REWIND (UNIT)
    ENDIF
    !
    DO I=1,NATOM
       ATOMPR(1,I) = I
       ATOMPR(2,I) = I
       RW(I) = AMASS(I)
    ENDDO
    !
    ! LPRINT = IPR > 0 &  LNOROT = .FALSE.
    !
    CALL ROTLS1( &
         X,Y,Z,XCOMP,YCOMP,ZCOMP,NATOM,ATOMPR,NATOM,RW,(IPR > 0),.FALSE.)
    !
    ! Do list updates if appropriate
    CALL UPDECI(0,X,Y,Z,WMAIN,0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
    !
    ! ENERGY CALL FOR THE FIRST POINT (DFAC=0)
    !
    DFAC=0
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
#if KEY_PARALLEL==1
    CALL VDGBR(DX,DY,DZ,1)
#endif 
    !
    R1=0
    R2=0
    DST=0
    DSE=0
    IF(IOLEV > 0) WRITE(IPATH,20)
20  FORMAT(' DFAC,EN,TARGET,R1,R2,DST,DSE,LOG(DFAC-Q)')
    IF(IOLEV > 0) WRITE(IPATH,200) &
         DFAC,EPROP(EPOT),EPROP(EPOT),R1,R2,DST,DSE,EPS
    !
    IF(PRNLEV > 2) WRITE(OUTU,*) ' write unit '
    IF(IOLEV > 0) WRITE(UNIT) &
         (X(I),I=1,NATOM),(Y(I),I=1,NATOM),(Z(I),I=1,NATOM)
    !      IF(IOLEV > 0) WRITE(UNIT)
    !     &               (X(I),I=1,NATOM),(Y(J),J=1,NATOM),(Z(K),K=1,NATOM)
    !
    ! USING STRAIGHT LINE INTERPOLATION FOR INITIAL GUESS
    !
    ! RW = STEP*(PRODUCTS - REACTANTS) ; zero order reaction coordinate

    !
    IF(PRNLEV > 3) write(OUTU,*) ' do 20'
    gg = 0
    DO I=1,NATOM
       iy = i  + natom
       iz = iy + natom
       rw(i ) = xcomp(i)-x(i)
       rw(iy) = ycomp(i)-y(i)
       rw(iz) = zcomp(i)-z(i)
       gg = gg + rw(i)**2 + rw(iy)**2 + rw(iz)**2
    ENDDO
    if(gg < tolg) then
       IF(WRNLEV > 2) WRITE(OUTU,31) TOLG,GG
       goto 5100
    endif
31  format(' pat14> distance too small; tolg,gg = ',2e10.2)
    !
    IF(PRNLEV > 3) write(OUTU,*) ' do 40'
    gg=step/sqrt(gg)
    DO I=1,NATOM
       iy = i  + natom
       iz = iy + natom
       x(i) = x(i) +  rw(i )*gg
       y(i) = y(i) +  rw(iy)*gg
       z(i) = z(i) +  rw(iz)*gg
    ENDDO

    !
    !  PATH LOOP ....
    !
    IF(PRNLEV > 3) WRITE(OUTU,*) ' DO 5000'
    IF(NSTR > 10000) THEN
       IF(WRNLEV > 2) write(OUTU,'(A,I5)') ' pat14> ??? nstr = ',nstr
       CALL WRNDIE(-4,'<PAT14>','error')
    ENDIF
    !
    ! Do list updates if appropriate
    CALL UPDECI(1,X,Y,Z,WMAIN,0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
    !
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
#if KEY_PARALLEL==1
    CALL VDGBR(DX,DY,DZ,1)
#endif 
    EVERY=EPROP(EPOT)
    SOLD=STEP
    DO ISTR=1,NSTR
       !
       DFAC = DFAC + STEP

       EOLD=EPROP(EPOT)
       GG=0
       DO J=1,NATOM
          XCOMP(J)=X(J)
          YCOMP(J)=Y(J)
          ZCOMP(J)=Z(J)
          JY = J  + NATOM
          JZ = JY + NATOM
          RW(J ) = DX(J)
          RW(JY) = DY(J)
          RW(JZ) = DZ(J)
          GG = GG + DX(J)*DX(J) + DY(J)*DY(J) + DZ(J)*DZ(J)
       ENDDO
       IF(GG < TOLG) THEN
          IF(WRNLEV >= 2) WRITE(OUTU,80) TOLG,GG
          GOTO 5100
       ENDIF
80     FORMAT(/' PAT14> gradient too small; tolg,gg = ',2E10.2)

       !.....GO ALONG GRADIENT WITH GIVEN STEP

       GG   = SQRT(GG)
       STEP = SOLD/GG
       IREP=0
100    CONTINUE
       DO I=1,NATOM
          IY = I  + NATOM
          IZ = IY + NATOM
          X(I) = XCOMP(I) - RW(I)*step
          Y(I) = YCOMP(I) - RW(IY)*step
          Z(I) = ZCOMP(I) - RW(IZ)*step
       ENDDO
       ! Do list updates if appropriate
       CALL UPDECI(ISTR,X,Y,Z,WMAIN,0,&
            (/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
       !
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
#if KEY_PARALLEL==1
       CALL VDGBR(DX,DY,DZ,1)
#endif 
       IREP=IREP+1
       IF(IPR > 1) THEN
          WRITE(OUTU,140)ISTR,IREP,GG,EPROP(EPOT),EPROP(EPOT)-EOLD
       ENDIF
140    FORMAT(' PAT14> ISTR,IREP,GG,E,E-EOLD = ',2I5,1X,3E12.4)
       IF(EPROP(EPOT) > EOLD) THEN
          STEP=0.5*STEP
          IF(STEP < TOLG) THEN
             IF(WRNLEV >= 2) WRITE(OUTU,180) STEP,TOLG
             GOTO 5100
          ENDIF
          GOTO 100
       ENDIF
       !
       IF(EVERY-EPROP(EPOT) > DFPRED) THEN
          EVERY=EPROP(EPOT)
          SOLD=1.1*STEP*GG
          IF(IPR > 0) WRITE(OUTU,160) ISTR,SOLD,GG,EPROP(EPOT)
          IF(IOLEV > 0) WRITE(IPATH,200) &
               DFAC,EPROP(EPOT),EPROP(EPOT),R1,R2,DST,DSE,EPS
          IF(IOLEV > 0) WRITE(UNIT) (X(I),I=1,NATOM), &
               (Y(I),I=1,NATOM),(Z(I),I=1,NATOM)
       ENDIF
       !
160    FORMAT(' PAT14> I,S,G,E = ',I5,1X,3(1X,F10.5))
180    FORMAT(' PAT14> STEP TOO SMALL; STEP,TOLG=',2E10.2)
       !
       !
200    FORMAT(' pat14>',f5.3,4(1x,f6.2),1x,3(1x,f6.1))
       !
    ENDDO
    !
5100 CONTINUE
    IF(IPR > 0) WRITE(OUTU,160) ISTR,SOLD,GG,EPROP(EPOT)
    IF(IOLEV > 0) WRITE(IPATH,200) &
         DFAC,EOLD,EPROP(EPOT),R1,R2,DST,DSE,EPS
    IF(IOLEV > 0) WRITE(UNIT) &
         (XCOMP(I),I=1,NATOM),(YCOMP(I),I=1,NATOM),(ZCOMP(I),I=1,NATOM)
    !
    IF(PRNLEV >= 2) WRITE(OUTU,*) &
         ' PAT14> return after 5000 continue'
    RETURN
    !.....PAT14 (stepest descent search for minimum energy path)
  END subroutine pat14

  SUBROUTINE GENDC(DC,NC,N1,NATOM,AMASS,X1,Y1,Z1,X2,Y2,Z2,DOT)
    !
    use chm_kinds
    use number
    use stream
    use machutil,only:die
    implicit none
    !
    INTEGER NATOM,NC,N1
    integer, parameter :: n2=7
    real(chm_real) amass(natom)
    real(chm_real) x1(*),y1(*),z1(*),x2(*),y2(*),z2(*), &
         dc(nc,*),dot(nc)
    real(chm_real) a,xm,ym,zm
    INTEGER I,J,IY,IZ
    !
    if(n1 /= 1 .and. n1 /= 4 .or. nc /= 7) then
       IF(WRNLEV >= 2) write(OUTU,'(A,2I5)') ' gendc> n1,nc =',n1,nc
       CALL DIE
    endif
    do i=n1,n2
       dot(i) = 0
    enddo
    if(n1 > 1) goto 30

    !...... dc(1..3) - center of mass constraints

    do i=1,natom
       iy = i  + natom
       iz = iy + natom
       a  = amass(i)
       dc(1,i ) = a
       dc(1,iy) = ZERO
       dc(1,iz) = ZERO
       dot(1) = dot(1) + a*a
       dc(2,i ) = ZERO
       dc(2,iy) = a
       dc(2,iz) = ZERO
       !      dot(2) = dot(2) + a*a
       dc(3,i ) = ZERO
       dc(3,iy) = ZERO
       dc(3,iz) = a
       !      dot(3) = dot(3) + a*a
       !      if(n2 == 3) goto 20
    enddo
    dot(2)=dot(1)
    dot(3)=dot(1)
30  continue
    do i=1,natom
       iy = i  + natom
       iz = iy + natom
       a  = amass(i)
       xm = a*x1(i)
       ym = a*y1(i)
       zm = a*z1(i)

       !...... dc(4..6) - rigid body rotation constraints

       dc(4,i ) = ZERO
       dc(4,iy) = -zm
       dc(4,iz) =  ym
       dot(4) = dot(4) + zm*zm + ym*ym
       dc(5,i ) =  zm
       dc(5,iy) =  ZERO
       dc(5,iz) = -xm
       dot(5) = dot(5) + zm*zm + xm*xm
       dc(6,i ) = -ym
       dc(6,iy) =  xm
       dc(6,iz) =  ZERO
       dot(6) = dot(6) + ym*ym + xm*xm

       !...... dc(7) - zero-order path constraints. Zero-order path is just
       !               linear guess

       dc(7,i ) = x1(i)-x2(i)
       dc(7,iy) = y1(i)-y2(i)
       dc(7,iz) = z1(i)-z2(i)
       dot(7) = dot(7) + dc(7,i)**2 + dc(7,iy)**2 + dc(7,iz)**2
    enddo

    !.......check length of vectors defining constraints

    do i=n1,n2
       if(dot(i) < 1.d-9) then
          IF(WRNLEV >= 2) write(OUTU,80) i,dot(i)
          goto 900
       endif
80     format(/' gendc> to small ??? i,dot(i) = ',i5,1x,e15.4)
    enddo
    return
900 IF(PRNLEV >= 2) write(OUTU,910) &
         (x1(i),y1(i),z1(i),x2(i),y2(i),z2(i),i=1,min0(15,natom))
910 format(/' gendc> x1,...,z2 = '/15(/1x,6(1x,f9.4)))
    IF(PRNLEV >= 2) write(OUTU,920) &
         (j,(dc(i,j),i=1,7),j=1,min0(45,3*natom))
920 format(/' gendc> dc = '/45(/1x,i5,7(1x,f8.3)))
    !.......gendc
  end subroutine gendc

  subroutine ortho1p(dc,nc,n1,n,dot,ipr)
    !
    use chm_kinds
    use stream
    use machutil,only:die
    implicit none
    !
    INTEGER N1,NC,N
    real(chm_real) dc(nc,n),dot(nc)
    real(chm_real) a,cn,s
    integer, parameter :: n2=7
    real(chm_real), parameter :: eps=1.d-9
    INTEGER IER,IPR,I,J,K,I1
    !
    if(n1 /= 1 .and. n1 /= 4 .or. nc /= 7) then
       IF(WRNLEV >= 2) write(OUTU,'(a,2i5)') &
            ' ortho1p> n1,nc = ',n1,nc
       CALL DIE
    endif

    !....... this subroutine orthonormalize vectors in dc array

    ier = 0
    if(n1 > 1) goto 44
    s   = 0
    do k=1,n
       s = s + dc(1,k)**2
    enddo
    if(s <= eps) then
       IF(WRNLEV >= 2) write(OUTU,32) s,eps
       CALL WRNDIE(-4,'<ortho1p>','error in first vector')
    endif
32  format(/' ortho1p> s,eps = ',2(1x,e15.4))
    s = 1.d0/sqrt(s)
    do k=1,n
       dc(1,k) = s*dc(1,k)
    enddo

44  continue
    if(n1 == 1) i1 = 2
    if(n1 > 1) i1 = n1

    do i=i1,n2

       do j=n1,i-1
          s = 0
          do k=1,n
             s = s + dc(i,k)*dc(j,k)
          enddo
          dot(j)=s
       enddo

       cn = 0
       do k=1,n
          s = 0
          do j=n1,i-1
             s = s + dot(j)*dc(j,k)
          enddo
          a = dc(i,k) - s
          dc(i,k) = a
          cn = cn + a*a
       enddo
       if(cn < eps) then
          IF(WRNLEV >= 2) write(OUTU,'(A,2E10.2)') &
               ' ortho1p error> cn to small; cn,eps=',cn,eps
          CALL WRNDIE(-4,'<ortho1p>','error')
       endif
       cn = 1.d0/sqrt(cn)
       do k=1,n
          dc(i,k) = cn*dc(i,k)
       enddo

    enddo

    !      if(n2 /= nc) return
    if(ipr == 0) return

    !.......check

    do i=1,nc
       do j=1,i
          s = 0
          do k=1,n
             s = s + dc(i,k)*dc(j,k)
          enddo
          if(i == j) s = 1.d0 - s
          if(abs(s) > eps) then
             IF(WRNLEV >= 2) write(OUTU,400) i,j,s,eps
             ier = 1
          endif
       enddo
    enddo
400 format(' ortho1p error: i,j,s,eps = ',2(1x,i5),1x,2e10.2)
    if(ier == 0) return
    CALL WRNDIE(-4,'<ortho1p>','dc not orthonormal')
    !.......ortho
  end subroutine ortho1p

  SUBROUTINE PROJECT(G,DC,NCDIM,NC,N)
    !
    use chm_kinds
    use stream
    implicit none
    !
    INTEGER NCDIM,NC,N
    real(chm_real) DC(NCDIM,N),G(N)
    !
    real(chm_real) DOT(7),S,GN,GOLD
    real(chm_real) EPS
    INTEGER IER,I,J,K
    DATA EPS/1.D-9/
    !
    gold = 0
    do k=1,n
       gold = gold + g(k)*g(k)
    enddo

    do j=1,nc
       s=0
       do k=1,n
          s = s + g(k)*dc(j,k)
       enddo
       dot(j)=s
    enddo

    gn = 0
    do k=1,n
       s = 0
       do j=1,nc
          s = s + dot(j)*dc(j,k)
       enddo
       g(k) = g(k) - s
       gn = gn + g(k)**2
    enddo

    !.......check if constraints are orthonormal
    !             and the lenght of the gradient

    ier = 0
    do i=1,nc
       do j=1,i
          s = 0
          do k=1,n
             s = s + dc(i,k)*dc(j,k)
          enddo
          if(i == j) s = 1.d0 - s
          if(abs(s) > eps) then
             IF(WRNLEV >= 2) write(OUTU,400) i,j,s,eps
             ier = ier + 1
          endif
       enddo
    enddo
400 format(' project error: i,j,s,eps = ',2(1x,i5),1x,2e10.2)
    if(gn < eps .AND. WRNLEV >= 2) write(OUTU,410) gn,gold
410 format(' project> warning;  gn,gold=',2e10.2)
    if(ier /= 0 .AND. WRNLEV >= 2) write(OUTU,*) &
         ' project> error; dc not orthonormal '
    if(ier == 0) return
    CALL WRNDIE(-4,'<PROJECT>','error')
    stop
    !.......projection
  end subroutine project

  SUBROUTINE ZXPTH(N,ACC,MAXFN,DFPRED,X,G,F,W,IER, &
       NPATH,NPATH1,XX,YY,ZZ,XCOMP,YCOMP,ZCOMP,NDIM,FLAGS,IFPEN,IPR, &
       MCON,DC,FC1,FC2,FC3,AUX,DFAC,R1,R2,DST,DSE,CONSTR,RCN)
    !
    ! THIS IS THE IMSL ROUTINE ZCGR WITH SLIGHT MODIFICATION OF INPUT
    ! AND OUTPUT
    !
    use chm_kinds
    use stream
    implicit none
    !
    INTEGER   N,MAXFN,IER,NPATH,NDIM
    INTEGER   IFPEN,IPR,NPATH1,MCON
    !C     INTEGER   UNIT,SUNIT,SAV,SAVP
    INTEGER   FLAGS(*)
    real(chm_real)    ACC,DFPRED,X(N),G(N),F,W(N*6),RCN
    real(chm_real)    AUX(*)
    real(chm_real)    XX(:),YY(:),ZZ(:),XCOMP(:),YCOMP(:),ZCOMP(:)
    real(chm_real)    DC(MCON,N),FC1,FC2,FC3,DFAC,CONSTR
    real(chm_real)    R1,R2,DST,DSE
    INTEGER   MAXLIN,MXFCON,I,IGINIT,IGOPT,IRETRY,IRSDG, &
         IRSDX,ITERC,ITERFM,ITERRS,IXOPT,NCALLS,NFBEG, &
         NFOPT
    real(chm_real)    BETA,DDSPLN,DFPR,FCH,FINIT,FMIN,GAMDEN,GAMA, &
         GINIT,GMIN,GNEW,GSPLN,GSQRD,SBOUND,STEP,STEPCH, &
         STMIN,SUM,WORK
    DATA      MAXLIN/5/,MXFCON/2/
    !
    IF(MCON /= 7) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,'(A,I5)') &
            'ZXPTH> ERROR; mcon /= 7; mcon=',mcon
       CALL WRNDIE(-4,'<ZXPTH>','error')
    ENDIF
    !
    !CC     NPATH1=0
    IER = 0
    IRSDX = N
    IRSDG = IRSDX+N
    IGINIT = IRSDG+N
    IXOPT = IGINIT+N
    IGOPT = IXOPT+N
    ITERC = 0
    ITERFM = ITERC
    NCALLS = 0
5   NCALLS = NCALLS+1
    NPATH1 = NPATH1+1
    CALL PAT2(N,X,F,G,NPATH,XX,YY,ZZ,XCOMP,YCOMP,ZCOMP, &
         NDIM,NPATH1,FLAGS,IFPEN,IPR,MCON,DC, &
         FC1,FC2,FC3,AUX,DFAC,R1,R2,DST,DSE,CONSTR,RCN)
    IF (NCALLS >= 2) GO TO 20
10  DO I=1,N
       W(I)=-G(I)
    ENDDO
    ITERRS = 0
    IF (ITERC > 0) GO TO 80
20  GNEW = 0.0D0
    SUM  = 0.0D0
    DO I=1,N
       GNEW = GNEW+W(I)*G(I)
       SUM  = SUM+G(I)**2
    ENDDO
    IF (NCALLS == 1) GO TO 35
    FCH = F-FMIN
    IF(FCH > 0.0) GOTO 50
    IF(FCH < 0.0) GOTO 35
    IF (GNEW/GMIN < -1.0D0) GO TO 45
35  FMIN  = F
    GSQRD = SUM
    NFOPT = NCALLS
    DO I=1,N
       W(IXOPT+I) = X(I)
       W(IGOPT+I) = G(I)
    ENDDO
45  IF (SUM <= ACC) GO TO 9005
50  IF (NCALLS /= MAXFN) GO TO 55
    IER = 131
    GO TO 9000
55  IF (NCALLS > 1) GO TO 100
    DFPR  = DFPRED
    STMIN = DFPRED/GSQRD
80  ITERC = ITERC+1
    FINIT = F
    GINIT = 0.0D0
    DO I=1,N
       W(IGINIT+I) = G(I)
       GINIT = GINIT+W(I)*G(I)
    ENDDO
    IF (GINIT >= 0.0D0) GO TO 165
    GMIN = GINIT
    SBOUND = -1.0D0
    NFBEG = NCALLS
    IRETRY = -1
    STEPCH = MIN(STMIN,ABS(DFPR/GINIT))
    STMIN = 0.0D0
90  STEP = STMIN+STEPCH
    WORK = 0.0D0
    DO I=1,N
       X(I) = W(IXOPT+I)+STEPCH*W(I)
       WORK = MAX(WORK,ABS(X(I)-W(IXOPT+I)))
    ENDDO
    IF (WORK > 0.0D0) GO TO 5
    IF (NCALLS > NFBEG+1) GO TO 115
    IF (ABS(GMIN/GINIT) > 0.2D0) GOTO 115
    GOTO 170
    !
100 WORK = (FCH+FCH)/STEPCH-GNEW-GMIN
    DDSPLN = (GNEW-GMIN)/STEPCH
    IF (NCALLS > NFOPT) SBOUND = STEP
    IF (NCALLS > NFOPT) GO TO 105
    IF (GMIN*GNEW <= 0.0D0) SBOUND = STMIN
    STMIN = STEP
    GMIN = GNEW
    STEPCH = -STEPCH
105 IF (FCH /= 0.0D0) DDSPLN = DDSPLN+(WORK+WORK)/STEPCH
    IF (GMIN == 0.0D0) GO TO 170
    IF (NCALLS <= NFBEG+1) GO TO 120
    IF (ABS(GMIN/GINIT) <= 0.2D0) GO TO 170
110 IF (NCALLS < NFOPT+MAXLIN) GO TO 120
115 IER = 129
    GO TO 170
120 STEPCH = 0.5D0*(SBOUND-STMIN)
    IF (SBOUND < -0.5D0) STEPCH = 9.0D0*STMIN
    GSPLN = GMIN+STEPCH*DDSPLN
    IF (GMIN*GSPLN < 0.0D0) STEPCH = STEPCH*GMIN/(GMIN-GSPLN)
    GO TO 90
125 SUM = 0.0D0
    DO I=1,N
       SUM = SUM+G(I)*W(IGINIT+I)
    ENDDO
    BETA = (GSQRD-SUM)/(GMIN-GINIT)
    IF (ABS(BETA*GMIN) <= 0.2D0*GSQRD) GO TO 135
    IRETRY = IRETRY+1
    IF (IRETRY <= 0) GO TO 110
135 IF (F < FINIT) ITERFM = ITERC
    IF (ITERC < ITERFM+MXFCON) GO TO 140
    IER = 132
    GO TO 9000
140 DFPR = STMIN*GINIT
    IF (IRETRY > 0) GO TO 10
    IF (ITERRS == 0) GO TO 155
    IF (ITERC-ITERRS >= N-6) GO TO 155
    IF (ABS(SUM) >= 0.2D0*GSQRD) GO TO 155
    GAMA = 0.0D0
    SUM = 0.0D0
    DO I=1,N
       GAMA = GAMA+G(I)*W(IRSDG+I)
       SUM = SUM+G(I)*W(IRSDX+I)
    ENDDO
    GAMA = GAMA/GAMDEN
    IF (ABS(BETA*GMIN+GAMA*SUM) >= 0.2D0*GSQRD) GO TO 155
    DO I=1,N
       W(I) = -G(I)+BETA*W(I)+GAMA*W(IRSDX+I)
    ENDDO
    GO TO 80
155 GAMDEN = GMIN-GINIT
    DO I=1,N
       W(IRSDX+I) = W(I)
       W(IRSDG+I) = G(I)-W(IGINIT+I)
       W(I) = -G(I)+BETA*W(I)
    ENDDO
    ITERRS = ITERC
    GO TO 80
165 IER = 130
170 IF (NCALLS == NFOPT) GO TO 180
    F = FMIN
    DO I=1,N
       X(I) = W(IXOPT+I)
       G(I) = W(IGOPT+I)
    ENDDO
180 IF (IER == 0) GO TO 125
9000 CONTINUE
    RETURN
9005 CONTINUE
    RETURN
  end SUBROUTINE zxpth

#endif 

end module rxpath

