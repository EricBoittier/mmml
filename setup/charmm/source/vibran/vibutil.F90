SUBROUTINE GETMOD(X,Y,Z,XNEW,YNEW,ZNEW,XCOMP,YCOMP,ZCOMP, &
     DDSCR,DDM,LNOMA,LNOTR,NNMDS,NFREQ,DDEV,NFJAC,FJAC,ANGA,ERR)
  !
  ! This routine gets a mode as requested for the EDIT INCLude command
  ! or the PROJect command.
  !
  ! Bernard R. Brooks     24-Nov-1984
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use deriv
  use memory
  use comand
  use select
  use string
  use usermod,only: usernm
  use coord,only: wmain
  use vibsub
  use number
  implicit none

  real(chm_real) X(*),Y(*),Z(*),XNEW(*),YNEW(*),ZNEW(*)
  real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),DDSCR(*),DDM(*)
  LOGICAL LNOMA,LNOTR,ERR
  INTEGER NNMDS,NFREQ ! Number of max normal modes, actual number of normal modes so far
  INTEGER NFJAC ! Number of entries in FJAC  (OPTI command)
  real(chm_real) FJAC(3,NNMDS) ! Array for Jacobian factors in OPTI
  INTEGER ANGA(3,NNMDS) ! Array for indices of atoms in angle CANG
  real(chm_real) DDEV(NNMDS)

  INTEGER I,NAT3
  real(chm_real),allocatable,dimension(:) :: ITROT
  integer,allocatable,dimension(:) :: ICON,ISLCT
  INTEGER NATIC,IATIC(4)
  INTEGER IXE,IYE,IZE,IRE
  real(chm_real) VAL
  CHARACTER(len=4) :: WRD
  LOGICAL OK

  NAT3=NATOM*3
  ERR=.FALSE.

  WRD=NEXTA4(COMLYN,COMLEN)
  NATIC=0

  call chmalloc('vibutil.src','GETMOD','ISLCT',NATOM,intg=ISLCT)
  xnew(1:natom)=zero
  ynew(1:natom)=zero
  znew(1:natom)=zero

  ! translation option
  IF(WRD.EQ.'TRAN') THEN
     WRD=NEXTA4(COMLYN,COMLEN)
     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
        XNEW(I)=0.0
        YNEW(I)=0.0
        ZNEW(I)=0.0
        IF(WRD.EQ.'X   ') THEN
           XNEW(I)=1.0
        ELSE IF(WRD.EQ.'Y   ') THEN
           YNEW(I)=1.0
        ELSE IF(WRD.EQ.'Z   ') THEN
           ZNEW(I)=1.0
        ELSE
           CALL WRNDIE(0,'<GETMOD>','ERROR IN MODE SPECIFICATION')
           RETURN
        ENDIF
        ENDIF
     ENDDO

     ! rotation option
  ELSE IF(WRD.EQ.'ROTA') THEN
     WRD=NEXTA4(COMLYN,COMLEN)
     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     IF(WRD.EQ.'X    ') THEN
        DO I=1,NATOM
           IF(ISLCT(I).EQ.1) THEN
           XNEW(I)=0.0
           YNEW(I)=-Z(I)
           ZNEW(I)=Y(I)
           ENDIF
        ENDDO
     ELSE IF(WRD.EQ.'Y    ') THEN
        DO I=1,NATOM
           IF(ISLCT(I).EQ.1) THEN
           XNEW(I)=Z(I)
           YNEW(I)=0.0
           ZNEW(I)=-X(I)
           ENDIF
        ENDDO
     ELSE IF(WRD.EQ.'Z    ') THEN
        DO I=1,NATOM
           IF(ISLCT(I).EQ.1) THEN
           XNEW(I)=-Y(I)
           YNEW(I)=X(I)
           ZNEW(I)=0.0
           ENDIF
        ENDDO
     ELSE
        CALL WRNDIE(0,'<GETMOD>','ERROR IN MODE SPECIFICATION')
        RETURN
     ENDIF

     ! get specified spherical harmonic motion
  ELSE IF(WRD.EQ.'SPHE') THEN
     WRD=NEXTA4(COMLYN,COMLEN)
     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     IXE=GTRMI(COMLYN,COMLEN,'IX',0)
     IYE=GTRMI(COMLYN,COMLEN,'IY',0)
     IZE=GTRMI(COMLYN,COMLEN,'IZ',0)
     IRE=GTRMI(COMLYN,COMLEN,'IR',0)
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
        XNEW(I)=0.0
        YNEW(I)=0.0
        ZNEW(I)=0.0
        VAL=1.0
        IF(IXE.GT.0) VAL=X(I)**IXE
        IF(IYE.GT.0) VAL=VAL*Y(I)**IYE
        IF(IZE.GT.0) VAL=VAL*Z(I)**IZE
        IF(IRE.GT.0) VAL=VAL*(SQRT(X(I)**2+Y(I)**2+Z(I)**2))**IRE
        IF(WRD.EQ.'X') THEN
           XNEW(I)=VAL
        ELSE IF(WRD.EQ.'Y') THEN
           YNEW(I)=VAL
        ELSE IF(WRD.EQ.'Z') THEN
           ZNEW(I)=VAL
        ELSE IF(WRD.EQ.'R') THEN
           XNEW(I)=VAL*X(I)
           YNEW(I)=VAL*Y(I)
           ZNEW(I)=VAL*Z(I)
        ELSE IF(WRD.EQ.'TX') THEN
           YNEW(I)=VAL*Z(I)
           ZNEW(I)=-VAL*Y(I)
        ELSE IF(WRD.EQ.'TY') THEN
           ZNEW(I)=VAL*X(I)
           XNEW(I)=-VAL*Z(I)
        ELSE IF(WRD.EQ.'TZ') THEN
           XNEW(I)=VAL*Y(I)
           YNEW(I)=-VAL*X(I)
        ELSE
           CALL WRNDIE(0,'<GETMOD>','ERROR IN MODE SPECIFICATION')
           RETURN
        ENDIF
        ENDIF
     ENDDO

     ! use comparison coordinate vector as is
  ELSE IF(WRD.EQ.'COMP') THEN
     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
        XNEW(I)=XCOMP(I)
        YNEW(I)=YCOMP(I)
        ZNEW(I)=ZCOMP(I)
        ENDIF
     ENDDO

     ! use difference betwenn main and comparison coordinates
  ELSE IF(WRD.EQ.'DIFF') THEN
     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
        XNEW(I)=XCOMP(I)-X(I)
        YNEW(I)=YCOMP(I)-Y(I)
        ZNEW(I)=ZCOMP(I)-Z(I)
        ENDIF
     ENDDO

     ! use the force array (with mass weighting)
  ELSE IF(WRD.EQ.'FORC') THEN
     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
        XNEW(I)=DX(I)/AMASS(I)
        YNEW(I)=DY(I)/AMASS(I)
        ZNEW(I)=DZ(I)/AMASS(I)
        ENDIF
     ENDDO

     ! get a user vector
  ELSE IF(WRD.EQ.'USER') THEN
     I=NEXTI(COMLYN,COMLEN)
     CALL USERNM(I,X,Y,Z,XNEW,YNEW,ZNEW,NATOM)

     ! get internal mode if specified
  ELSE IF(WRD.EQ.'BOND') THEN
     NATIC=2
  ELSE IF(WRD.EQ.'ANGL') THEN
     NATIC=3
  ELSE IF(WRD.EQ.'DIHE') THEN
     NATIC=4
  ELSE IF(WRD.EQ.'CBND') THEN  ! For constrained bonds GK14 
     NATIC=2
  ELSE IF(WRD.EQ.'CANG') THEN  ! For constrained angles GK14
     NATIC=3
  ELSE
     CALL WRNDIE(0,'<GETMOD>','ERROR IN MODE SPECIFICATION')
     RETURN
  ENDIF

  IF(ABS(NATIC).GT.0) THEN
     CALL NXTATM(IATIC,I,NATIC,COMLYN,COMLEN,ISLCT, &
          SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
     IF(NATIC.NE.I) THEN
        CALL WRNDIE(0,'<VIBRAN>','WRONG NUMBER OF ATOMS FOR IC.')
        RETURN
     ENDIF
     DO I=1,NATIC
        IF(IATIC(I).LE.0) THEN
           CALL WRNDIE(0,'<VIBRAN>','ATOM OF IC DOES NOT EXIST.')
           RETURN
        ENDIF
     ENDDO

     ! Use a special basis for constraint-corrections, otherwise use standard basis
     IF (WRD.EQ.'CBND' .OR. WRD.EQ.'CANG') THEN  
        CALL CONSINTEMD(NATIC,IATIC,X,Y,Z,XNEW,YNEW,ZNEW, &
             NAT3,DDSCR,DDM,NNMDS,NFREQ,DDEV,NFJAC,FJAC,ANGA,OK)
     ELSE 
        ! Get the internal motion mode
        CALL INTEMD(NATIC,IATIC,X,Y,Z,XNEW,YNEW,ZNEW, &
             NAT3,DDSCR,DDM,LNOMA,OK)
     ENDIF 
     IF(.NOT.OK) THEN
        CALL WRNDIE(0,'<GETMOD>','ERROR IN MODE SPECIFICATION')
        RETURN
     ENDIF
  ENDIF
  call chmdealloc('vibutil.src','GETMOD','ISLCT',NATOM,intg=ISLCT)

  ! Now remove any net translation/rotation if LNOTR flag is set
  IF(LNOTR) THEN
     call chmalloc('vibutil.src','GETMOD','ICON',NATOM,intg=ICON)
     ICON(1:NATOM)=1
     CALL REMVTR(X,Y,Z,XNEW,YNEW,ZNEW,ICON,NAT3,DDSCR,DDM, &
          LNOMA,AMASS)
     call chmdealloc('vibutil.src','GETMOD','ICON',NATOM,intg=ICON)
  ENDIF

  RETURN
END SUBROUTINE GETMOD

SUBROUTINE COMDOT(ISTRT,ISTOP,NAT3,XNEW,YNEW,ZNEW, &
     DDV,DDM,DDSCR,DDF,LNONO,LTOTAL)
  !
  ! THIS ROUTINE COMPUTES THE DOT PRODUCT OF THE COORDINATE
  ! DIFFERENCES ONTO THE SPECIFIED NORMAL MODE SPACE
  !
  ! By Bernard R. Brooks   1982
  !
  use chm_kinds
  use vector
  use stream
  implicit none
  INTEGER ISTRT,ISTOP,NAT3
  real(chm_real) DDV(NAT3,*),DDM(*),DDSCR(*),DDF(*)
  real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
  LOGICAL LNONO,LTOTAL

  ! LOCAL STORAGE
  INTEGER NATOM,IUNIT,I,J,IPT
  real(chm_real) PDR,DE,STOT

  NATOM=NAT3/3
  IUNIT=OUTU

  DO J=1,NATOM
     IPT=J*3-3
     DDSCR(IPT+1)=XNEW(J)/DDM(J)
     DDSCR(IPT+2)=YNEW(J)/DDM(J)
     DDSCR(IPT+3)=ZNEW(J)/DDM(J)
  ENDDO
  IF(LNONO) THEN
     CALL DOTPR(DDSCR,DDSCR,NAT3,DE)
     DE=SQRT(DE)
  ELSE
     CALL NORMALL(DDSCR,NAT3)
     DE=1.0
  ENDIF
  IF(.NOT.LTOTAL .AND. PRNLEV.GE.2) WRITE(OUTU,85) DE
85 FORMAT(' THE NORM OF THE PROJECTION VECTOR IS',F12.6/)

  STOT=0.0
  DO I=ISTRT,ISTOP
     CALL DOTPR(DDSCR,DDV(1,I),NAT3,PDR)
     STOT=STOT+PDR*PDR
     IF(.NOT.LTOTAL .AND. PRNLEV.GE.2) WRITE(IUNIT,629) I,DDF(I), &
          PDR,100.0*PDR*PDR
629  FORMAT(' MODE',I4,'  FREQ=',F10.2,'  DOTPR=',F10.6, &
          '  %=',F10.4)
  ENDDO
  STOT=STOT*100.0/DE
  IF(PRNLEV.GE.2) WRITE(IUNIT,633) STOT
633 FORMAT(' TOTAL PERCENT ACCOUNTED FOR IS',F12.4)

  RETURN
END SUBROUTINE COMDOT


SUBROUTINE MODEOPTI(ISTRT,ISTOP,IUNO,NAT3,NATOM,NNMDS,TEMP,XNEW,YNEW,ZNEW, &
     X,Y,Z,DDV,DDM,DDSCR,DDEV,DDF,NFJAC,FJAC,ANGA,LENER,LPARA)
  !
  ! This routine optimizes the energy of the specified  modes.
  ! First, the dot product of the gradient with the mode is calculated
  ! and a Newton-Raphson step is performed in the direction of the mode
  ! according to the dotproduct and the eigenvalue (force constant)
  ! 
  ! If the "ENER" keyword is specified, it calculates the energy difference to the 
  ! local energy minimum and the vibrational entropy and Jacobian entropy terms.
  ! 
  ! Gerhard Koenig and Bernard R. Brooks  2014 
  !
  use chm_kinds
  use vector
  use stream
  implicit none
  INTEGER ISTRT,ISTOP,NAT3,NATOM    ! start index, stop index, dimension, number of atoms
  INTEGER NNMDS                                   ! Number of normal modes
  INTEGER IUNO                                      ! Output unit for total energy
  real(chm_real) DDV(NAT3,NNMDS)                ! eigenvectors
  real(chm_real) DDM(NATOM)               ! reciprocal root mass array  (1/sqrt{m})
  real(chm_real) DDSCR(NAT3)              ! scratch 
  real(chm_real) DDEV(NNMDS)                        ! eigenvalues of modes
  real(chm_real) DDF(NNMDS)                          ! frequencies of modes in cm-1 (sqrt(K/m))  
  real(chm_real) TEMP                            ! Temperature
  real(chm_real) XNEW(NATOM),YNEW(NATOM),ZNEW(NATOM) ! gradients 
  real(chm_real) X(NATOM),Y(NATOM),Z(NATOM)                        ! coordinates
  INTEGER NFJAC  ! Number of entries in FJAC and ANGA
  real(chm_real) FJAC(3,NNMDS) ! Array for Jacobian factors, 
  ! angle radii for cuvature of CANG modes, and parameter-based-eigenvalues 
  INTEGER ANGA(3,NNMDS) ! Array for indices of atoms in angle CANG
  LOGICAL LENER  ! Flag, Calculate energy difference to minimum with one step Newton-Raphson
                             ! plus entropy contributions from vibration and the Jacobian
  LOGICAL LPARA  ! Flag, Calculate energy difference to minimum with one step Newton-Raphson
                             ! plus entropy contributions from vibration and the Jacobian based on 
                             ! force constant in parameter file (saved in FJAC). 

  ! LOCAL STORAGE
  INTEGER IUNIT,I,J,IPT
  real(chm_real) PDR     !Cross product of the mass weighted projection vector with the mode DDV
  real(chm_real) DE      !Magnitude of mass weighted projection vector
  real(chm_real) STOT    !Proportion of vector that can be projected on modes
  real(chm_real) EVTOL   !Tolerance for including a basis set in ENER computation
  real(chm_real) ETOT    !Total energy drop in ENER
  real(chm_real) EVAL    !Energy drop of mode 
  real(chm_real) SVAL    !Displacement along mode 
  real(chm_real) JACS    !Contribution of mode to Jacobian factor
  real(chm_real) JACSTOT !Sum of Jacobian contributions
  real(chm_real) VIBS    !Contribution of mode to vibrational entropy
  real(chm_real) VIBSTOT !Sum of vibrational entropy contributions
  real(chm_real) TMPARR(9)  ! Temporary array used to calculate bonds, angles + rotate angles
  real(chm_real) ROTR    !Radius of angle rotation
  real(chm_real) DTHETA     ! Change of bond angle (in rad)
  real(chm_real) JTHETA     ! New Jacobian of bond angle 
  real(chm_real) TMP    
  real(chm_real) KT
  real(chm_real) PI  
  IUNIT=OUTU
  EVTOL = 0.01 ! rayleigh tolerance on including a basis set in energy computation

  KT=1.9872041E-3 * TEMP
  PI = 3.141592653589793D0 

  DO J=1,NATOM
        IPT=J*3-3
        DDSCR(IPT+1)=XNEW(J)/DDM(J)   
        DDSCR(IPT+2)=YNEW(J)/DDM(J)
        DDSCR(IPT+3)=ZNEW(J)/DDM(J)
  ENDDO

! If PARA is specified, we replace the eigenvalues by data from the parameter file
! (later on, we are going to substitute them back)
  IF(LPARA) THEN
      DO I=1,NFJAC
            TMP = DDEV(I)
            DDEV(I) =  FJAC(3,I)
            FJAC(3,I) = TMP
      ENDDO
  ENDIF

  CALL DOTPR(DDSCR,DDSCR,NAT3,DE) ! Calculate magnitude of the projection vector for ENER (DE)
  DE=SQRT(DE)
  IF(PRNLEV.GE.2) WRITE(OUTU,85) DE ! 
85 FORMAT(' THE NORM OF THE PROJECTION VECTOR IS',F12.6/)

  STOT=0.0
  ETOT=0.0 ! 
  JACSTOT=0.0E+0
  VIBSTOT=0.0E+0

  ! Loop over selected modes  
  DO I=ISTRT,ISTOP
      JACS = 0.0E+0

     CALL DOTPR(DDSCR,DDV(1,I),NAT3,PDR)     ! Project vector into mode
     STOT=STOT+PDR*PDR !  ratio of projection vector accounted by mode

     
     IF(ABS(DDEV(I)) .LT. EVTOL ) THEN  ! Ignore modes with too low eigenvalues
        EVAL=0.0  
        SVAL=0.0E+0
        IF(PRNLEV.GE.2) WRITE(IUNIT,622) I,DDEV(I)  
622  FORMAT(' MODE',I4,' Eigenvalue too low',F14.4,'  - Ignored.')  
     ELSE  
        ! Calculate energy drop (Newton Raphson step) 
        EVAL=0.5*PDR*PDR/DDEV(I)                    ! drop = 0.5 * gradient^2 / curvature
        SVAL=-PDR/DDEV(I)                           ! Displacement along mode
        ETOT=ETOT+EVAL                              ! Add to total energy drop for all modes

        ! Calculate vibrational entropy
        VIBS = -KT*LOG(2.0E+0*PI*KT/DDEV(I))/2.0E+0


        ! Change the coordinates according to energy drop

        ! First, we treat normal modes
        IF(ANGA(2,I).EQ.0.AND.ANGA(3,I).EQ.0) THEN 
             DO J=1,NATOM
                 IPT=J*3-3
  
                 ! Calculate displacement
                 TMPARR(1:3)=  SVAL*DDV(IPT+1:IPT+3,I)*DDM(J)

                 ! Add displacement
                 X(J) = X(J) + TMPARR(1)
                 Y(J) = Y(J) + TMPARR(2)  
                 Z(J) = Z(J) + TMPARR(3) 
                
             ENDDO

        ! For CBND modes 
        ELSEIF(ANGA(3,I).EQ.0) THEN     
             DO J=ANGA(2,I),ANGA(2,I) ! We already know which atoms will be changed based on ANGA
                 IPT=J*3-3

                 ! Check whether mode really matches the bond vector 
                 ! (if it does not match, there is probably something going wrong)
                 TMPARR(1) =   X(ANGA(2,I))-X(ANGA(1,I))
                 TMPARR(2) =   Y(ANGA(2,I))-Y(ANGA(1,I))
                 TMPARR(3) =   Z(ANGA(2,I))-Z(ANGA(1,I))
                 CALL DOTPR(TMPARR(1:3),DDV(IPT+1:IPT+3,I),3,TMP)
                 ! This should match the data in FJAC 
                 IF(ABS(TMP**2-FJAC(1,I)).GT.0.00001) CALL WRNDIE(-1,'<MODEOPTI>', &
                 'BOND HAS BEEN MODIFIED SINCE CREATION OF BASIS SET')

                 ! Calculate displacement
                 TMPARR(1:3)=  SVAL*DDV(IPT+1:IPT+3,I)*DDM(J)

                 ! Add displacement
                 X(J) = X(J) + TMPARR(1)
                 Y(J) = Y(J) + TMPARR(2)  
                 Z(J) = Z(J) + TMPARR(3) 
                
                 IF(LENER) THEN
                    ! Calculate change of Jacobian factor 
                    CALL DOTPR(TMPARR(1:3),DDV(IPT+1:IPT+3,I),3,TMP)
                    JACS =  JACS-KT*LOG((SQRT(FJAC(1,I))+TMP)**2/FJAC(1,I))
                 ENDIF
             ENDDO

        ! For CANG modes
        ELSE

             ! For angles, we have to consider the curvature. 
             ! The force has been projected into the tangent vector. However, the projection 
             ! does not preserve the bond length. We, therefore, project into a angle rotation by 
             ! distributing the displacement on the tangent (in DDV) and the curvature vectors 
             ! (calculated based on atoms in ANGA)
             ! Moreover, the displacement vector is the change of the angle in radians times the 
             ! radius of the bond rotation at the time that RAYLEIGH or REDUCE were invoked. 
             ! The radius of the bond rotation has been saved to FJAC(2,I) when the CANG mode
             ! was created.

             DO J=ANGA(3,I),ANGA(3,I) ! We already know which atoms will be changed based on ANGA

                 IPT=J*3-3

                 ! Calculate change of bond angle (divide by bond length used for force constant)
                 DTHETA = SVAL/FJAC(2,I)*DDM(J)

                 ! Calculate Jacobian term before changes and check whether bond angle is linear
                 ! Vector for bond between atoms 1 and 2 of angle 
                 TMPARR(1) =   X(ANGA(2,I))-X(ANGA(1,I))
                 TMPARR(2) =   Y(ANGA(2,I))-Y(ANGA(1,I))
                 TMPARR(3) =   Z(ANGA(2,I))-Z(ANGA(1,I))

                 ! Vector for bond between atoms 2 and 3 of angle = curvature vector 
                 TMPARR(4) =   X(ANGA(2,I))-X(ANGA(3,I))
                 TMPARR(5) =   Y(ANGA(2,I))-Y(ANGA(3,I))
                 TMPARR(6) =   Z(ANGA(2,I))-Z(ANGA(3,I))
                 CALL CROSS3(TMPARR(1:3),TMPARR(4:6),TMPARR(7:9))
                 
                 JTHETA= SQRT(SUM(TMPARR(7:9)**2) / SUM(TMPARR(1:3)**2) / SUM(TMPARR(4:6)**2))

                 ! Check whether angle is linear 
                 IF (JTHETA.LT.0.001) CALL WRNDIE(-1,'<MODEOPTI>', &
                 'STARTING ANGLE OF MODE IS ALMOST LINEAR (< 0.001 RAD)')

                 ! Check whether Jacobian matches the Jacobian used for the force constant
                 ! (if it does not match, our force constant will probably be wrong)
                 IF ((JTHETA-FJAC(1,I)).GT.0.001) CALL WRNDIE(-1,'<MODEOPTI>', &
                 'ANGLE HAS BEEN MODIFIED SINCE CREATION OF BASIS SET')

                 ! Calculate radius of rotation - the 2-3 bond lenght might have changed 
                 ! due to changes from other modes
                 ROTR=SQRT(SUM(TMPARR(4:6)**2)) 

                ! Displacement along tangential vector
                 TMP = SIN(DTHETA)
                 X(J)=X(J)+TMP*DDV(IPT+1,I)*ROTR
                 Y(J)=Y(J)+TMP*DDV(IPT+2,I)*ROTR  
                 Z(J)=Z(J)+TMP*DDV(IPT+3,I)*ROTR

                 ! Displacement along curvature vector (bond from central atom) 
                 TMP = COS(DTHETA)
                 X(J)=X(J)+TMPARR(4)-TMP*TMPARR(4)
                 Y(J)=Y(J)+TMPARR(5)-TMP*TMPARR(5)
                 Z(J)=Z(J)+TMPARR(6)-TMP*TMPARR(6)

                 ! Calculate new Jacobian term
                 TMPARR(1) =   X(ANGA(2,I))-X(ANGA(1,I))
                 TMPARR(2) =   Y(ANGA(2,I))-Y(ANGA(1,I))
                 TMPARR(3) =   Z(ANGA(2,I))-Z(ANGA(1,I))
                 ! Vector for second bond between atom 2 and 3
                 TMPARR(4) =   X(ANGA(2,I))-X(ANGA(3,I))
                 TMPARR(5) =   Y(ANGA(2,I))-Y(ANGA(3,I))
                 TMPARR(6) =   Z(ANGA(2,I))-Z(ANGA(3,I))
                 CALL CROSS3(TMPARR(1:3),TMPARR(4:6),TMPARR(7:9))
                 
                 JTHETA= SQRT(SUM(TMPARR(7:9)**2) / SUM(TMPARR(1:3)**2) / SUM(TMPARR(4:6)**2))
              
                 ! IF ENER has been called, calculate Jacobian
                 IF(LENER) THEN
                    ! Calculate change of Jacobian factor 

                    ! Check whether molecule was or is linear (no Jacobian contribution)
                    IF(FJAC(1,I).LT.0.001) THEN
                       JACS =  JACS-KT*LOG(JTHETA)
                    ELSEIF(JTHETA.LT.0.001) THEN
                       JACS = JACS-KT*LOG(1/FJAC(1,I))
                    ELSE
                       JACS =  JACS-KT*LOG(JTHETA/FJAC(1,I))
                    ENDIF
                 ENDIF                 
             ENDDO
        ENDIF        

     ENDIF  
     

     IF(PRNLEV.GE.4) THEN

        IF(LENER) THEN
           IF(FJAC(1,I).GT.EVTOL) THEN
              WRITE(IUNIT,631) I,EVAL,VIBS,JACS
631  FORMAT(' MODE',I4,'  ENERGY DROP EST.= ',F14.4,' VIBRATIONAL ENTROPY= ',F14.4, &
               ' JACOBIAN CHANGE= ',F14.4)
           ELSE
              WRITE(IUNIT,632) I,DDEV(I),PDR,SVAL,EVAL
632  FORMAT(' MODE',I4,'  EIGENVALUE= ',F14.4,'  DOTPR= ',F10.6, &
              ' SCALE FACTOR= ',F14.4,  '  ENERGY DROP EST.= ',F14.4)
           ENDIF
        ENDIF
     ENDIF
     JACSTOT=JACSTOT+JACS
     VIBSTOT=VIBSTOT+VIBS
  ENDDO

  STOT=STOT*100.0/DE         ! Percentage of total gradient that has been projected into the modes
  IF (PRNLEV.GE.2) THEN
     IF(LENER) THEN 
        WRITE(IUNO,636) (-ETOT+ VIBSTOT+ JACSTOT), -ETOT, VIBSTOT, JACSTOT
636 FORMAT(' -DG_CONS ',F16.4,' ENERGY CHANGE ',F16.4,' VIBR. ENTROPY ', F16.4, ' JACOBIAN ', F16.4) 
     ELSE
        WRITE(IUNO,633) ETOT
633 FORMAT(' TOTAL ENERGY DROP ESTIMATE IS',F16.4) 
  ENDIF
  ENDIF

! Putting the eigenvalues back where they belong to if PARA was specified
  IF(LPARA) THEN
      DO I=1,NFJAC
            TMP = DDEV(I)
            DDEV(I) = FJAC(3,I)
            FJAC(3,I) = TMP
      ENDDO
  ENDIF

  RETURN
END SUBROUTINE MODEOPTI



SUBROUTINE TRJDOT(ISTRT,ISTOP,NAT3, &
     XCOMP,YCOMP,ZCOMP,XNEW,YNEW,ZNEW, &
     DDV,DDM,DDSCR,DDF,LNONO,LTOTAL,LNOMAS, &
     NUNIT,FIRSTU,NBEGN,NSKIP,NSTOP, &
     LWRIT,COMLYN,COMLEN,LSPIR)
  !
  ! THIS ROUTINE COMPUTES THE DOT PRODUCT OF THE COORDINATE
  ! DIFFERENCES ONTO THE GIVEN TRAJECTORY
  !
  ! By Nathan Desdouits and Arnaud Blondel, adapted from COMDOT, 4/2012
  use chm_kinds
  use ctitla
  use cvio
  use coorio_mod
  use ctitla
  use stream
  use string
  use memory
  use number
  use vector
  implicit none
  INTEGER ISTRT,ISTOP,NAT3
  INTEGER NUNIT,FIRSTU,NSKIP,NBEGN,NBEGN2,NSTOP,NSTOP2
  real(chm_real) DDV(NAT3,*),DDM(*),DDSCR(*),DDF(*)
  real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*)
  real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
  LOGICAL LNONO,LTOTAL,LNOMAS,LWRIT,NOPROJ,OFFICIAL,LSPIR
  real(chm_real) DELTA,T
  INTEGER NATOM,IUNIT,I,J,IPT,WUNIT
  INTEGER ICOORD,NCOORD,ISTATS
  INTEGER MODECW,MODEL,MODFL,NICTOTP(2)
  real(chm_real) PDR,DE,STOT
  INTEGER NFREAT,IFILE,ISTEP,NDEGF,NSAVV
  integer,allocatable,dimension(:) :: IFREAT
  real(chm_real4),allocatable,dimension(:) :: ITEMP
  real(chm_real),allocatable,dimension(:) :: XP,YP,ZP,WP
  real(chm_real),allocatable,dimension(:) :: XP2,YP2,ZP2
  integer,allocatable,dimension(:) :: ISLCTP,IBASEP
  CHARACTER(len=8),allocatable,dimension(:) :: SEGIDP,RESP
  CHARACTER(len=8),allocatable,dimension(:) :: RESIDP,ATYPEP
  CHARACTER(len=8) :: DUMCHA
  CHARACTER(len=4) :: HDR1,HDR2,HDR
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN,ICNTRL(20)
  DATA HDR1,HDR2/'COOR','CORD'/
  NATOM=NAT3/3
  ICOORD=0
  call chmalloc('vibutil.src','TRJDOT','IFREAT',NATOM,intg=IFREAT)
  call chmalloc('vibutil.src','TRJDOT','ITEMP',NATOM,cr4=ITEMP)
  IUNIT=FIRSTU
  ISTATS=1

  IF(LWRIT)THEN
     IF(NBEGN.EQ.0) THEN
        NBEGN2=1
     ENDIF
     IF(NSTOP.EQ.0) THEN
        NSTOP2=0
        DO I=FIRSTU,FIRSTU+NUNIT-1
           CALL GTICNT(I,HDR,ICNTRL,.FALSE.,.TRUE.,.FALSE.)
           NSTOP2=NSTOP2+ICNTRL(1)
        ENDDO
        NCOORD=(NSTOP2-NBEGN)/NSKIP
     ELSE
        NCOORD=(NSTOP-NBEGN)/NSKIP
     ENDIF

     call chmalloc('vibutil.src','TRJDOT','XP',NCOORD,crl=XP)
     call chmalloc('vibutil.src','TRJDOT','YP',NCOORD,crl=YP)
     call chmalloc('vibutil.src','TRJDOT','ZP',NCOORD,crl=ZP)
     call chmalloc('vibutil.src','TRJDOT','WP',NCOORD,crl=WP)
     call chmalloc('vibutil.src','TRJDOT','ISLCTP',NCOORD,intg=ISLCTP)
     call chmalloc('vibutil.src','TRJDOT','IBASEP',NCOORD+1,intg=IBASEP)
     call chmalloc('vibutil.src','TRJDOT','ATYPEP',NCOORD,ch8=ATYPEP)
     call chmalloc('vibutil.src','TRJDOT','RESIDP',NCOORD,ch8=RESIDP)
     call chmalloc('vibutil.src','TRJDOT','RESP',NCOORD,ch8=RESP)
     call chmalloc('vibutil.src','TRJDOT','SEGIDP',NCOORD,ch8=SEGIDP)
     IBASEP(1)=0
     NICTOTP(1)=0
     NICTOTP(2)=NCOORD
     ISLCTP(1:NCOORD)=1
     ATYPEP(1:NCOORD)='DUM     '
     DO I=1,NCOORD
        WRITE(DUMCHA,'(I8)')I
        RESIDP(I)=ADJUSTL(DUMCHA)
        IBASEP(I+1)=I
     ENDDO
     RESP(1:NCOORD)='DUM     '
     SEGIDP(1:NCOORD)='        '
     NOPROJ=.FALSE.
     XP(1:NCOORD)=0.0
     YP(1:NCOORD)=0.0
     ZP(1:NCOORD)=0.0
     WP(1:NCOORD)=0.0
  ELSE
     call chmalloc('vibutil.src','TRJDOT','XP',1,crl=XP)
     call chmalloc('vibutil.src','TRJDOT','YP',1,crl=YP)
     call chmalloc('vibutil.src','TRJDOT','ZP',1,crl=ZP)
     call chmalloc('vibutil.src','TRJDOT','WP',1,crl=WP)
     call chmalloc('vibutil.src','TRJDOT','XP2',1,crl=XP2)
     call chmalloc('vibutil.src','TRJDOT','YP2',1,crl=YP2)
     call chmalloc('vibutil.src','TRJDOT','ZP2',1,crl=ZP2)
     call chmalloc('vibutil.src','TRJDOT','ISLCTP',1,intg=ISLCTP)
     call chmalloc('vibutil.src','TRJDOT','IBASEP',1,intg=IBASEP)
     call chmalloc('vibutil.src','TRJDOT','ATYPEP',1,ch8=ATYPEP)
     call chmalloc('vibutil.src','TRJDOT','RESIDP',1,ch8=RESIDP)
     call chmalloc('vibutil.src','TRJDOT','RESP',1,ch8=RESP)
     call chmalloc('vibutil.src','TRJDOT','SEGIDP',1,ch8=SEGIDP)
     NOPROJ=.TRUE.
  ENDIF

  DO WHILE (ISTATS.GE.0)
     CALL READCV(XNEW,YNEW,ZNEW, &
#if KEY_CHEQ==1
          (/ ZERO /), .FALSE., &  
#endif
          ITEMP,NATOM,IFREAT,NFREAT, &
          FIRSTU,NUNIT,IUNIT,IFILE, &
          ISTEP,ISTATS,NDEGF,DELTA, &
          NBEGN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
          TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
     ICOORD=ICOORD+1
     IF(LWRIT.AND.ICOORD.GT.NCOORD)THEN
        CALL WRNDIE(-1,'<TRJDOT>', &
      'MORE FRAMES THAN EXPECTED. NO MORE PROJECTION CAN BE MADE.')
        NOPROJ=.TRUE.
     ENDIF
     IPT=0
     IF(LNOMAS) THEN
        DO J=1,NATOM
           IPT=J*3-3
           DDSCR(IPT+1)=XNEW(J)-XCOMP(J)
           DDSCR(IPT+2)=YNEW(J)-YCOMP(J)
           DDSCR(IPT+3)=ZNEW(J)-ZCOMP(J)
        ENDDO
     ELSE
        DO J=1,NATOM
           IPT=J*3-3
           DDSCR(IPT+1)=(XNEW(J)-XCOMP(J))/DDM(J)
           DDSCR(IPT+2)=(YNEW(J)-YCOMP(J))/DDM(J)
           DDSCR(IPT+3)=(ZNEW(J)-ZCOMP(J))/DDM(J)
        ENDDO
     ENDIF
     IF(LNONO) THEN
        CALL DOTPR(DDSCR,DDSCR,NAT3,DE)
        DE=SQRT(DE)
     ELSE
        CALL NORMALL(DDSCR,NAT3)
        DE=1.0
     ENDIF
     IF(.NOT.LTOTAL .AND. PRNLEV.GE.2) WRITE(OUTU,85) DE
85 FORMAT(' THE NORM OF THE PROJECTION VECTOR IS',F12.6/)

     STOT=0.0
     DO I=ISTRT,ISTOP
        CALL DOTPR(DDSCR,DDV(1,I),NAT3,PDR)
        STOT=STOT+PDR*PDR
        IF(.NOT.LTOTAL .AND. PRNLEV.GE.2) WRITE(OUTU,629) ICOORD,I,DDF(I), &
             PDR,100.0*PDR*PDR/DE/DE
629  FORMAT(' FRAME',I8,' MODE',I4,'  FREQ=',F10.2, &
             '  DOTPR=',F12.8,'  %=',F10.4)
        J=ISTOP-I
        IF(LWRIT.AND.((.NOT.NOPROJ).AND.J.LT.4))THEN
           IF(J.EQ.0)THEN
              XP(ICOORD)=PDR
           ELSE IF(J.EQ.1)THEN
              YP(ICOORD)=PDR
           ELSE IF(J.EQ.2)THEN
              ZP(ICOORD)=PDR
           ELSE IF(J.EQ.3)THEN
              WP(ICOORD)=PDR
           ENDIF
        ENDIF
     ENDDO
     STOT=STOT*100.0/DE
     IF(PRNLEV.GE.2) WRITE(OUTU,633) STOT
633 FORMAT(' TOTAL PERCENT ACCOUNTED FOR IS',F12.4)
  ENDDO
  IF(LWRIT)THEN
     WUNIT=GTRMI(COMLYN,COMLEN,'WUNI',-1)
     MODECW=4
     IF(INDXA(COMLYN,COMLEN,'FILE').NE.0) MODECW=1
     IF(INDXA(COMLYN,COMLEN,'CARD').NE.0) MODECW=2
     IF(INDXA(COMLYN,COMLEN,'PDB').NE.0) THEN
        MODECW=4
        OFFICIAL = INDXA(COMLYN,COMLEN,'OFFI').GT.0
        IF(OFFICIAL)THEN
           WRITE(OUTU,*) ' Write official pdb format.  '// &
                'Note that the segid (chain id) will be '// &
                'truncated to only one character.'
        ELSE
           WRITE(OUTU,*) ' Write CHARMM-pdb format'
        ENDIF
     ENDIF
     IF(LSPIR)THEN
        call chmalloc('vibutil.src','TRJDOT','XP2',NCOORD,crl=XP2)
        call chmalloc('vibutil.src','TRJDOT','YP2',NCOORD,crl=YP2)
        call chmalloc('vibutil.src','TRJDOT','ZP2',NCOORD,crl=ZP2)
        T=1.0
        DO I=1,NCOORD
           T=T+1.5/T/3.141592
           XP2(I)=T*COS(T*3.141592)
           YP2(I)=T*SIN(T*3.141592)
        ENDDO
        ZP2(1:NCOORD)=0.0
        CALL CWRITE2(WUNIT,TITLEA,NTITLA,ICNTRL,XP2,YP2,ZP2,WP, &
             SEGIDP,RESP,RESIDP,ATYPEP,NICTOTP,IBASEP,1,NCOORD,NCOORD, &
             ISLCTP,MODECW,0,0,OFFICIAL)
        call chmdealloc('vibutil.src','TRJDOT','XP2',NCOORD,crl=XP2)
        call chmdealloc('vibutil.src','TRJDOT','YP2',NCOORD,crl=YP2)
        call chmdealloc('vibutil.src','TRJDOT','ZP2',NCOORD,crl=ZP2)
     ENDIF
     CALL CWRITE2(WUNIT,TITLEA,NTITLA,ICNTRL,XP,YP,ZP,WP, &
          SEGIDP,RESP,RESIDP,ATYPEP,NICTOTP,IBASEP,1,NCOORD,NCOORD, &
          ISLCTP,MODECW,0,0,OFFICIAL)
     call chmdealloc('vibutil.src','TRJDOT','XP',NCOORD,crl=XP)
     call chmdealloc('vibutil.src','TRJDOT','YP',NCOORD,crl=YP)
     call chmdealloc('vibutil.src','TRJDOT','ZP',NCOORD,crl=ZP)
     call chmdealloc('vibutil.src','TRJDOT','WP',NCOORD,crl=WP)
     call chmdealloc('vibutil.src','TRJDOT','ISLCTP',NCOORD,intg=ISLCTP)
     call chmdealloc('vibutil.src','TRJDOT','IBASEP',NCOORD+1,intg=IBASEP)
     call chmdealloc('vibutil.src','TRJDOT','ATYPEP',NCOORD,ch8=ATYPEP)
     call chmdealloc('vibutil.src','TRJDOT','RESIDP',NCOORD,ch8=RESIDP)
     call chmdealloc('vibutil.src','TRJDOT','RESP',NCOORD,ch8=RESP)
     call chmdealloc('vibutil.src','TRJDOT','SEGIDP',NCOORD,ch8=SEGIDP)
  ELSE
     call chmdealloc('vibutil.src','TRJDOT','XP',1,crl=XP)
     call chmdealloc('vibutil.src','TRJDOT','YP',1,crl=YP)
     call chmdealloc('vibutil.src','TRJDOT','ZP',1,crl=ZP)
     call chmdealloc('vibutil.src','TRJDOT','WP',1,crl=WP)
     call chmdealloc('vibutil.src','TRJDOT','ISLCTP',1,intg=ISLCTP)
     call chmdealloc('vibutil.src','TRJDOT','IBASEP',1,intg=IBASEP)
     call chmdealloc('vibutil.src','TRJDOT','ATYPEP',1,ch8=ATYPEP)
     call chmdealloc('vibutil.src','TRJDOT','RESIDP',1,ch8=RESIDP)
     call chmdealloc('vibutil.src','TRJDOT','RESP',1,ch8=RESP)
     call chmdealloc('vibutil.src','TRJDOT','SEGIDP',1,ch8=SEGIDP)
  ENDIF 

  RETURN
END SUBROUTINE TRJDOT

SUBROUTINE SETCMP(ISTRT,ISTOP,NAT3, &
     XNEW,YNEW,ZNEW,DDV,DDM,DDSCR,DDEV, &
     ITYPE,RTYPE,LNOMA,LNONO,TFREQ,AFACT)
  !
  ! THIS ROUTINE GETS A NORMAL MODE WITH A PARTICULAR
  ! LENGTH FACTOR.
  !
  ! By Bernard R. Brooks   1982
  !
  use chm_kinds
  use stream
  implicit none
  INTEGER ISTRT,ISTOP,NAT3,ITYPE
  real(chm_real) DDV(NAT3,*),DDM(*),DDSCR(*),DDEV(*)
  real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
  real(chm_real) RTYPE,TFREQ,AFACT
  LOGICAL LNOMA,LNONO,LTFREQ

  CALL MAGFAC(NAT3,XNEW,YNEW,ZNEW,DDV(1,ISTRT),DDM,DDSCR, &
       DDEV(ISTRT),.TRUE.,ITYPE,RTYPE,LNOMA,LNONO,TFREQ,AFACT,LTFREQ)

  IF(ISTRT.NE.ISTOP) CALL WRNDIE(-2,'<SETCMP>', &
       'MORE THAN ONE MODE REQUESTED')

  RETURN
END SUBROUTINE SETCMP

SUBROUTINE APPENM(IFREQ,NFREQ,NAT3,XNEW,YNEW,ZNEW,DDV,DDM,DDF, &
     DDSCR,DDEV,ITYPE,RTYPE,LNOMA,LNONO,LORTH,TFREQ)
  !
  ! THIS ROUTINE APPENDS A SPECIFIED SET OF NORMAL MODES TO
  ! THE CURRENT SET OF NORMAL MODES
  !
  ! By Bernard R. Brooks   1982
  !
  use chm_kinds
  use number,only:zero
  use vector
  implicit none
  INTEGER IFREQ,NFREQ,NAT3,ITYPE
  real(chm_real) DDV(NAT3,*),DDM(*),DDSCR(*),DDEV(*),DDF(*)
  real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
  real(chm_real) RTYPE,TFREQ,AFACT
  LOGICAL LNOMA,LNONO,LTFREQ,LORTH
  INTEGER I,ISTOP
  real(chm_real) Q,TOL
  DATA TOL/1.0D-10/

  CALL MAGFAC(NAT3,XNEW,YNEW,ZNEW,DDV(1,IFREQ),DDM,DDSCR, &
       DDEV(IFREQ),.FALSE.,ITYPE,RTYPE,LNOMA,LNONO,TFREQ,AFACT,LTFREQ)

  DDEV(IFREQ)=0.D0
  DDF(IFREQ)=0.D0
  IF (LORTH) THEN
     ISTOP=IFREQ-1
     DO I=1,ISTOP
        CALL ORTHOG(DDV(1,IFREQ),DDV(1,I),NAT3)
     ENDDO
  ENDIF

  CALL DOTPR(DDV(1,IFREQ),DDV(1,IFREQ),NAT3,Q)
  IF(Q.GT.TOL) THEN
     IF(LORTH) CALL NORMALL(DDV(1,IFREQ),NAT3)
  ELSE
     IF(NFREQ.EQ.IFREQ) THEN
        CALL WRNDIE(1,'<APPENM>','Vector with zero norm rejected.')
        NFREQ=NFREQ-1
     ELSE
        CALL WRNDIE(0,'<APPENM>','Vector with zero norm inserted.')
        DDV(1:nat3,IFREQ)=zero
     ENDIF
  ENDIF

  RETURN
END SUBROUTINE APPENM

SUBROUTINE REMONM(ISTRT,ISTOP,NAT3,XNEW,YNEW,ZNEW,DDM, &
     DDSCR,DDV,LNONO)
  !
  ! THIS ROUTINE REMOVES A PARTICULAR MOTION FROM A SPECIFIED SET
  ! OF NORMAL MODES
  !
  ! By Bernard R. Brooks   1982
  !
  use chm_kinds
  use vector
  implicit none
  INTEGER ISTRT,ISTOP,NAT3
  real(chm_real) DDV(NAT3,*),DDM(*),DDSCR(*)
  real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
  LOGICAL LNONO
  INTEGER NATOM,IPT,I

  NATOM=NAT3/3
  IPT=0
  DO I=1,NATOM
     DDSCR(IPT+1)=XNEW(I)/DDM(I)
     DDSCR(IPT+2)=YNEW(I)/DDM(I)
     DDSCR(IPT+3)=ZNEW(I)/DDM(I)
     IPT=IPT+3
  ENDDO

  CALL NORMALL(DDSCR,NAT3)
  DO I=ISTRT,ISTOP
     CALL ORTHOG(DDV(1,I),DDSCR,NAT3)
     IF(.NOT.LNONO) CALL NORMALL(DDV(1,I),NAT3)
  ENDDO

  RETURN
END SUBROUTINE REMONM

SUBROUTINE SHAKNM(ISTRT,ISTOP,NAT3,XNEW,YNEW,ZNEW,X,Y,Z,DDM, &
     DDV,LNONO,AMASS,LMASS,IMOVE)
  !
  ! THIS ROUTINE REMOVES SHAKED BOND STRETCHES FROM THE SELECTED SET
  ! OF NORMAL MODES
  !
  ! By Bernard R. Brooks   24-Nov-1984
  !
  use chm_kinds
  use vector
  use dimens_fcm
  use shake
  use holonom,only:holonomf
  implicit none

  INTEGER ISTRT,ISTOP,NAT3
  real(chm_real) DDV(NAT3,*),DDM(*)
  real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL LNONO
  real(chm_real) AMASS(*)
  LOGICAL LMASS
  INTEGER IMOVE(*)

  INTEGER NATOM,IPT,I,IFREQ
  LOGICAL QOK

  NATOM=NAT3/3

  DO IFREQ=ISTRT,ISTOP
     IPT=0
     DO I=1,NATOM
        XNEW(I)=DDV(IPT+1,IFREQ)/DDM(I)
        YNEW(I)=DDV(IPT+2,IFREQ)/DDM(I)
        ZNEW(I)=DDV(IPT+3,IFREQ)/DDM(I)
        IPT=IPT+3
     ENDDO

     IF(QHOLO) THEN
        CALL HOLONOMF(XNEW,YNEW,ZNEW,X,Y,Z,LMASS,.FALSE.,QOK)
     ENDIF

     IPT=0
     DO I=1,NATOM
        DDV(IPT+1,IFREQ)=XNEW(I)*DDM(I)
        DDV(IPT+2,IFREQ)=YNEW(I)*DDM(I)
        DDV(IPT+3,IFREQ)=ZNEW(I)*DDM(I)
        IPT=IPT+3
     ENDDO

     IF(.NOT.LNONO) CALL NORMALL(DDV(1,IFREQ),NAT3)
  ENDDO
  RETURN
END SUBROUTINE SHAKNM

SUBROUTINE DELNRM(ISTRT,ISTOP,NFREQ,DDV,DDEV,DDF,NAT3)
  !
  ! THIS ROUTINE DELETES SPECIFIED NORMAL MODES FROM THE DATA FILE
  !
  ! By Bernard R. Brooks   1982
  !
  use chm_kinds
  implicit none
  INTEGER ISTRT,ISTOP,NFREQ,NAT3
  real(chm_real) DDV(NAT3,*),DDF(*),DDEV(*)
  INTEGER I,J,ILOW

  ILOW=ISTOP+1
  J=ISTRT
  DO I=ILOW,NFREQ
     DDF(J)=DDF(I)
     DDEV(J)=DDEV(I)
     DDV(1:nat3,j) = DDV(1:nat3,i)
     J=J+1
  ENDDO
  NFREQ=J-1

  RETURN
END SUBROUTINE DELNRM

SUBROUTINE ORTHNM(ISTRT,ISTOP,NFREQ,DDV,NAT3,LPURG,TOL)
  !
  ! THIS ROUTINE ORTHOGONALIZES AND NORMALIZES A SET OF NORMAL MODES
  !
  ! By Bernard R. Brooks   1982
  !
  use chm_kinds
  use number,only:zero
  use stream
  use vector
  implicit none
  INTEGER ISTRT,ISTOP,NFREQ,NAT3
  real(chm_real) DDV(NAT3,*)
  LOGICAL LPURG
  real(chm_real) PDR
  INTEGER I,J,IP,IFREQ
  real(chm_real) TOL

  IF(PRNLEV.GE.2) WRITE(OUTU,21) ISTRT,ISTOP,TOL
21 FORMAT(/' ORTHNM: Modes',I4,' to',I4,' will be orthonormalized', &
       ' using a tolerance of',E12.4)
  IFREQ=ISTRT-1
  DO I=ISTRT,ISTOP
     CALL DOTPR(DDV(1:nat3,I),DDV(1:nat3,I),NAT3,PDR)
     IF(PRNLEV.GE.2) WRITE(OUTU,22) I,PDR
22   FORMAT(' Mode',I4,' has a norm of',E14.6, &
          ' after orthogonalization')
     IF(PDR.GT.TOL) THEN
        IFREQ=IFREQ+1
        CALL NORMALL(DDV(1:nat3,I),NAT3)
        IP=I+1
        DO J=IP,ISTOP
           CALL ORTHOG(DDV(1:nat3,J),DDV(1:nat3,I),NAT3)
        ENDDO
        IF(IFREQ.NE.I) ddv(1:nat3,ifreq) = DDV(1:nat3,I)
     ELSE
        IF(LPURG) THEN
           IF(PRNLEV.GE.2) WRITE(OUTU,44) 'It will be removed.'
44         FORMAT(8X,A)
           !              remove this mode
        ELSE
           IF(PRNLEV.GE.2) WRITE(OUTU,44) 'It will be zeroed.'
           IFREQ=IFREQ+1
           DDV(1:nat3,IFREQ)=zero
        ENDIF
     ENDIF
  ENDDO

  IF(IFREQ.LT.ISTOP) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,44) &
          'Some modes deleted. Remapping remaining modes.'
     DO I=ISTOP+1,NFREQ
        IFREQ=IFREQ+1
        ddv(1:nat3,ifreq) = DDV(1:nat3,I)
     ENDDO
     NFREQ=IFREQ
     IF(PRNLEV.GE.2) WRITE(OUTU,45) NFREQ
45   FORMAT(' ORTHNM: There are',I4,' modes remaining.')
  ENDIF
  RETURN
END SUBROUTINE ORTHNM

SUBROUTINE MAGFAC(NAT3,XNORM,YNORM,ZNORM,DDV,DDM,DDSCR, &
     DDEV,LIN,ITYPE,RTYPE,LNOMA,LNONO,TFREQX,AFACTX,LTFREQ)
  !
  ! THIS ROUTINE TRANSFORMS THE NORMAL MODE (DDV) INTO A COORDINATE
  ! DISPLACEMENT ARRAYS (XNORM,YNORM,ZNORM)
  ! OR IT DOES THE REVERSE PROCESS (LIN=.FALSE.)
  ! THE MAGNITUDE IS SPECIFIED BY:
  !     ITYPE - 1=TEMP / 2=KCAL / 3=RMS / 4=FACT / 5=MRMS
  !     RTYPE - MULTIPLYING FACTOR
  !     LNOMA - NO MASS WEIGHTING FLAG
  !     LNONO - NO NORMALIZATION FLAG
  !     TFREQX - LOW FREQUENCY CUTOFF FOR TEMP OR KCAL SPECS
  ! RETURNED IS: AFACTX - FINAL MULT FACTOR, AND LTFREQ - LOW FREQ FLAG
  !
  ! By Bernard R. Brooks   1982
  !
  use chm_kinds
  use consta
  use vector
  implicit none
  real(chm_real) DDV(*),DDM(*),DDSCR(*),DDEV
  real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
  LOGICAL LIN,LNOMA,LNONO,LTFREQ
  real(chm_real) RTYPE,TFREQX,TFREQ,AFACTX
  INTEGER ITYPE,NAT3

  ! LOCAL STORAGE
  INTEGER NATOM,I,J,IPT
  INTEGER I800

  real(chm_real) AFACT
  real(chm_real) PDR,PDX,PDEV

  LTFREQ=.FALSE.
  TFREQ=(TFREQX/CNVFRQ)**2
  NATOM=NAT3/3

  IF(LIN) THEN
     ! copy into coordinate disp
     DDSCR(1:NAT3)=DDV(1:NAT3)
     IF(.NOT.LNONO) CALL NORMALL(DDSCR,NAT3)
     CALL DOTPR(DDSCR,DDSCR,NAT3,PDR)
     IF(.NOT.LNOMA) THEN
        IPT=0
        DO I=1,NATOM
           DO J=1,3
              IPT=IPT+1
              DDSCR(IPT)=DDSCR(IPT)*DDM(I)
           ENDDO
        ENDDO
     ENDIF

     call vib_get_scale_factor

     IPT=1
     DO I=1,NATOM
        XNORM(I)=DDSCR(IPT)*AFACT
        YNORM(I)=DDSCR(IPT+1)*AFACT
        ZNORM(I)=DDSCR(IPT+2)*AFACT
        IPT=IPT+3
     ENDDO

  ELSE
     ! copy into normal mode array
     IPT=1
     DO I=1,NATOM
        DDSCR(IPT)=XNORM(I)
        DDSCR(IPT+1)=YNORM(I)
        DDSCR(IPT+2)=ZNORM(I)
        IPT=IPT+3
     ENDDO
     CALL DOTPR(DDSCR,DDSCR,NAT3,PDX)

     IF(.NOT.LNOMA) THEN
        IPT=0
        DO I=1,NATOM
           DO J=1,3
              IPT=IPT+1
              DDSCR(IPT)=DDSCR(IPT)/DDM(I)
           ENDDO
        ENDDO
     ENDIF

     IF(LNONO) THEN
        DDV(1:NAT3)=DDSCR(1:NAT3)
        CALL NORMALL(DDSCR,NAT3)
     ELSE
        CALL NORMALL(DDSCR,NAT3)
        DDV(1:NAT3)=DDSCR(1:NAT3)
     ENDIF

     CALL DOTPR(DDSCR,DDSCR,NAT3,PDR)
     IF(.NOT.LNOMA) THEN
        IPT=0
        DO I=1,NATOM
           DO J=1,3
              IPT=IPT+1
              DDSCR(IPT)=DDSCR(IPT)*DDM(I)
           ENDDO
        ENDDO
     ENDIF

     call vib_get_scale_factor

     AFACT=SQRT(PDX/PDR)/AFACT
  ENDIF

  AFACTX=AFACT
  RETURN

  !---------- Containted recursive subroutines ------------------------------
contains

  ! to get-scale-factor
  subroutine vib_get_scale_factor()
    use machutil,only:die
    IF(ITYPE.EQ.1) THEN
       ! temperature factor
       LTFREQ=(ABS(DDEV).LT.TFREQ)
       PDEV=ABS(DDEV)
       IF(ABS(DDEV).LT.TFREQ) PDEV=TFREQ
       AFACT=SQRT((KBOLTZ)*RTYPE*2.0/(PDEV))
    ELSE IF(ITYPE.EQ.2) THEN
       ! kcals factor
       LTFREQ=(ABS(DDEV).LT.TFREQ)
       PDEV=ABS(DDEV)
       IF(ABS(DDEV).LT.TFREQ) PDEV=TFREQ
       AFACT=SQRT(RTYPE*2.0/(PDEV))
    ELSE IF(ITYPE.EQ.3) THEN
       ! rms factor
       CALL DOTPR(DDSCR,DDSCR,NAT3,PDX)
       AFACT=RTYPE*SQRT(NATOM*PDR/PDX)
    ELSE IF(ITYPE.EQ.4) THEN
       ! ordinary factor
       AFACT=RTYPE
    ELSE IF(ITYPE.EQ.5) THEN
       ! mass weighted rms factor
       PDR=0.0
       DO I=1,NATOM
          PDR=PDR+1.0/(DDM(I)*DDM(I))
       ENDDO
       AFACT=RTYPE*SQRT(PDR)
    ELSE
       CALL DIE
    ENDIF
    return
  end subroutine vib_get_scale_factor

END SUBROUTINE MAGFAC

SUBROUTINE MANMOD(NAT3,DDV,ISLCT, &
     IDEST,ISOURC,FACD,FACS,KEYWRD,DDEV,DDF)
  !
  ! SIMPLE MODE MANIPULATIONS: KEWRD DETERMINES ACTION
  !    'ADD '  : X/Y/Z(IDEST)=FACD*DDV(IDEST)+FACS*DDV(ISOURC)
  !    'MULT'  : DDV(IDEST)=DDV(IDEST)*FACS
  !    'SET '  : X/Y/Z(IDEST)=FACS
  ! ONLY ATOMS FOR WHICH ISLCT=1 ARE TOUCHED
  !
  use chm_kinds
  use vector
  implicit none
  INTEGER NAT3,IDEST,ISOURC
  real(chm_real) DDV(NAT3,*),DDEV(*),DDF(*)
  real(chm_real) FACD,FACS
  INTEGER ISLCT(*)
  character(len=*) KEYWRD
  INTEGER NATOM,I,IX,IY,IZ

  NATOM=NAT3/3
  IF(KEYWRD.EQ.'ADD ') THEN
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
           IX=3*(I-1)+1
           IY=IX+1
           IZ=IY+1
           DDV(IX,IDEST)=FACD*DDV(IX,IDEST)+FACS*DDV(IX,ISOURC)
           DDV(IY,IDEST)=FACD*DDV(IY,IDEST)+FACS*DDV(IY,ISOURC)
           DDV(IZ,IDEST)=FACD*DDV(IZ,IDEST)+FACS*DDV(IZ,ISOURC)
        ENDIF
     ENDDO
  ELSE IF(KEYWRD.EQ.'SET ') THEN
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
           IX=3*(I-1)+1
           IY=IX+1
           IZ=IY+1
           DDV(IX,IDEST)=FACS
           DDV(IY,IDEST)=FACS
           DDV(IZ,IDEST)=FACS
        ENDIF
     ENDDO
  ELSE IF(KEYWRD.EQ.'NORM') THEN
     CALL NORMALL(DDV(1,IDEST),NAT3)
  ELSE IF(KEYWRD.EQ.'COPY') THEN
     DDV(1:NAT3,IDEST)=DDV(1:NAT3,ISOURC)
     DDEV(IDEST)=DDEV(ISOURC)
     DDF(IDEST)=DDF(ISOURC)
  ELSE IF(KEYWRD.EQ.'ZERO') THEN
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
           IX=3*(I-1)+1
           IY=IX+1
           IZ=IY+1
           DDV(IX,IDEST)=0.0
           DDV(IY,IDEST)=0.0
           DDV(IZ,IDEST)=0.0
        ENDIF
     ENDDO
     DDEV(IDEST)=0.0
     DDF(IDEST)=0.0
  ENDIF
  RETURN
END SUBROUTINE MANMOD


SUBROUTINE FREQPR(DDF,NFREQ,OUTU)
  ! THIS SUBROUTINE PRINTS FREQUENCIES (DDF) TO UNIT (OUTU)
  ! Victor Anisimov, 2004

  use chm_kinds
  use exfunc
  use string
  implicit none
  INTEGER NFREQ,OUTU
  real(chm_real) DDF(NFREQ)
  ! Local variables
  INTEGER I,L
  CHARACTER(len=32) :: STRFMT
  !
  ! Convert number to character string
  !
  STRFMT=" "
  WRITE(STRFMT,*) NFREQ

  ! Determine number of non-blank characters
  L=STRLNG(STRFMT)
  ! Subtract number of leading blanks
  DO I=1,STRLNG(STRFMT)
     IF(INDEX(STRFMT(I:I),' ').GT.0) THEN
        L=L-1
     ELSE
        EXIT
     ENDIF
  ENDDO
  !
  ! Compose format string, e.g. '(5(1X,I3,F12.6))'
  !
  L=MAX(L,3)
  L=MIN(L,9)
  WRITE(STRFMT,'(A7,I1,A8)') '(5(1X,I', L, ',F12.6))'
  WRITE(OUTU,STRFMT) (I,DDF(I),I=1,NFREQ)

  RETURN
END SUBROUTINE FREQPR

