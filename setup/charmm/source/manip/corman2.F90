module corman2
  use chm_kinds
  implicit none

contains

  SUBROUTINE CONCOR(FROM,TO,X,Y,Z,NATOM,ISLCT,XUCELL)
    !
    !     Converts coordinates between FRACtional and
    !     cartesian in ALIGned and SYMMetric unit cell vectors.
    !
    !     ALLOWED OPTIONS (for FROM and TO) ARE: FRAC,ALIG,SYMM
    !
    !     Authors: Barry Olafson
    !              Bernie Brooks
    !              Overhauled by Ryszard Czerminski, February 1993
    !
    !     Computation tested by REB 30-Jan-1982  20:08:44
    !
  use number
  use consta
  use exfunc
  use stream
  use chutil,only:initia
    !
    character(len=*) FROM,TO
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER NATOM,ISLCT(*)
    real(chm_real) XUCELL(6)
    !
    INTEGER I
    LOGICAL OK, STATUS
    real(chm_real) HA(3,3),HI(3,3),HS(3,3),U,V,W,SYMM(6),R(3,3)
    !
    !.....test options
    !
    ok=(from == 'SYMM'.or.from == 'ALIG'.or.from == 'FRAC') .and. &
         (to   == 'SYMM'.or.  to == 'ALIG'.or.  to == 'FRAC')
    if(.not.ok) then
       write(outu,'(a,2(1x,a))') ' CONCOR> FROM,TO = ',from,to
       CALL WRNDIE(-5,'<CONCOR>','INVALID OPTIONS')
    endif
    if(from == to) return
    !
    !.....conversion from/to fractional coordinates to/from
    !     aligned or symmetric cartesian coordinates
    !
#if KEY_DEBUG==1
    IF(PRNLEV > 5) THEN
       WRITE(OUTU,'(A,2(1X,A))')' CONCOR> FROM,TO = ',FROM,TO
       WRITE(OUTU,'(A)')        ' CONCOR> XUCELL(1..6)='
       WRITE(OUTU,'(6F10.5)') (XUCELL(I),I=1,6)
    ENDIF
#endif /*  DEBUG*/
    IF(FROM == 'ALIG' .OR. TO == 'ALIG') THEN
       !
       ! XTLAXA calculates aligned unit cell vectors
       ! a: in xy-plane, b: along y-axis
       !
       CALL XTLAXA(HA,XUCELL,STATUS)
       IF(.NOT.STATUS) return
#if KEY_DEBUG==1
       IF(PRNLEV > 5) THEN
          WRITE(OUTU,'(A)') ' CONCOR> HA ='
          !         CALL MATOUT2(HA,XUCELL(1),3,3,3,OUTU)
       ENDIF
#endif /*  DEBUG*/
       if(to == 'FRAC') then
          CALL INVT33(R,HA,OK)
          IF(.not.OK) &
               CALL WRNDIE(-5,'<CONCOR>','INVT33(R,HA,OK) failed')
       else
#if KEY_GAMESS==1
          CALL DCOPY(int8(9),HA,int8(1),R,int8(1))
#else
          CALL DCOPY(9,HA,1,R,1)
#endif
       endif
    ENDIF
    IF(FROM == 'SYMM' .OR. TO == 'SYMM') THEN
       !
       ! XTLAXS calculates symmetric unit cell vectors
       !
       CALL XTLAXS(SYMM,XUCELL)
       !... ltof copies symmetric matrix stored in 1D array to 2D array
       call ltof(hs,symm,3,3)
#if KEY_DEBUG==1
       IF(PRNLEV > 5) THEN
          WRITE(OUTU,'(A)') ' CONCOR> HS ='
          !         CALL MATOUT2(HS,XUCELL(1),3,3,3,OUTU)
       ENDIF
#endif /*  DEBUG*/
       if(to == 'FRAC') then
          CALL INVT33(R,HS,OK)
          IF(.not.OK) &
               CALL WRNDIE(-5,'<CONCOR>','INVT33(R,HS,OK) failed')
       else
#if KEY_GAMESS==1
          CALL DCOPY(int8(9),HS,int8(1),R,int8(1))
#else
          CALL DCOPY(9,HS,1,R,1)
#endif
       endif
    ENDIF
    !
    !.....conversion between aligned and symmetric coordinates
    !
    IF(FROM == 'SYMM' .and. TO == 'ALIG')   THEN
       !
       !.....conversion from SYMMETRIC to ALIGNED coordinates
       !
       CALL INVT33(HI,HS,OK)
       IF(OK) THEN
          CALL MULMXN(R,HA,3,3,HI,3,3)
       ELSE
          CALL WRNDIE(-5,'<CONCOR>','INVT33(HI,HS,OK) failed')
       ENDIF
    ELSEIF(FROM == 'ALIG' .AND. TO == 'SYMM') THEN
       !
       !.....conversion from ALIGNED to SYMMETRIC coordinates
       !
       CALL INVT33(HI,HA,OK)
       IF(OK) THEN
          CALL MULMXN(R,HS,3,3,HI,3,3)
       ELSE
          CALL WRNDIE(-5,'<CONCOR>','INVT33(HI,HA,OK) failed')
       ENDIF
    ENDIF
    !
#if KEY_DEBUG==1
    IF(PRNLEV > 5) THEN
       WRITE(OUTU,'(/A)') ' CONCOR> R ='
       !       CALL MATOUT2(R,XUCELL(4),3,3,3,OUTU)
    ENDIF
#endif /*  DEBUG*/
    DO I=1,NATOM
       IF(ISLCT(I) == 1) THEN
          IF(INITIA(I,X,Y,Z)) THEN
             U=X(I)
             V=Y(I)
             W=Z(I)
             X(I)=R(1,1)*U+R(1,2)*V+R(1,3)*W
             Y(I)=R(2,1)*U+R(2,2)*V+R(2,3)*W
             Z(I)=R(3,1)*U+R(3,2)*V+R(3,3)*W
          ENDIF
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE CONCOR

  SUBROUTINE XTLAXA(HA,XUCELL,STATUS)
    !
    !     Calculates aligned shape matrix in the form
    !
    !     AX  0  CX
    !     AY  B  CY
    !      0  0  CZ
    !
    !     i.e. b-vector is along y-axis, a-vector in in xy-plane
    !
    !     Authors: Barry Olafson
    !              Bernie Brooks
    !              Overhauled by Ryszard Czerminski, February 1993
    !
  use consta
  use number
    real(chm_real) HA(3,3),XUCELL(6)
    LOGICAL STATUS
    real(chm_real) COSA,COSB,COSG,SING,FACT1,FACT2
    !
    STATUS = .FALSE.
    COSA=COS(DEGRAD*XUCELL(4))
    COSB=COS(DEGRAD*XUCELL(5))
    COSG=COS(DEGRAD*XUCELL(6))
    SING=SIN(DEGRAD*XUCELL(6))
    FACT1=(COSB-COSA*COSG)/SING
    if(abs(SING) > RSMALL) then
       FACT2=(1.D0-COSA*COSA-COSB*COSB-COSG*COSG+ &
            2.D0*COSA*COSB*COSG)/(SING*SING)
       if(FACT2 > ZERO) then
          STATUS = .TRUE.
          FACT2 = SQRT(FACT2)
          HA(1,1)=XUCELL(1)*SING
          HA(1,2)=0
          HA(1,3)=XUCELL(3)*FACT1
          HA(2,1)=XUCELL(1)*COSG
          HA(2,2)=XUCELL(2)
          HA(2,3)=XUCELL(3)*COSA
          HA(3,1)=0
          HA(3,2)=0
          HA(3,3)=XUCELL(3)*FACT2
       else
          call wrndie(0,'<XTLAXA>','FACT2 < 0')
       endif
    else
       call wrndie(0,'<XTLAXA>','abs(SING) < RSMALL')
    endif
    RETURN
  END SUBROUTINE XTLAXA

  SUBROUTINE VOLUME(NATOM,X,Y,Z,WMAIN,ISLCT,RMIN,RMAX,SPACE,LHOLES)
    !-----------------------------------------------------------------------
    !     THIS ROUTINES SET UP THE COMPUTATION OF THE ACCESIBLE VOLUME
    !
  use number
  use stream
  use memory
    !
    integer,allocatable,dimension(:) :: jMAT
    INTEGER NATOM
    real(chm_real)  X(*),Y(*),Z(*),WMAIN(*)
    INTEGER ISLCT(*)
    real(chm_real)  RMIN(*),RMAX(*)
    INTEGER SPACE
    LOGICAL LHOLES

    real(chm_real),allocatable,dimension(:),target :: space_mat
    real(chm_real),pointer,dimension(:) :: xmat,ymat,zmat
    
    real(chm_real) A,DELX,DELY,DELZ,XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX
    INTEGER I,ISPACE,IXGRID,IYGRID,IZGRID

    XMIN=ANUM
    XMAX=-ANUM
    YMIN=ANUM
    YMAX=-ANUM
    ZMIN=ANUM
    ZMAX=-ANUM

    DO I=1,NATOM
       IF(ISLCT(I) == 1) THEN
          IF(X(I)+RMIN(I) > XMAX) XMAX=X(I)+RMIN(I)
          IF(X(I)-RMIN(I) < XMIN) XMIN=X(I)-RMIN(I)
          IF(Y(I)+RMIN(I) > YMAX) YMAX=Y(I)+RMIN(I)
          IF(Y(I)-RMIN(I) < YMIN) YMIN=Y(I)-RMIN(I)
          IF(Z(I)+RMIN(I) > ZMAX) ZMAX=Z(I)+RMIN(I)
          IF(Z(I)-RMIN(I) < ZMIN) ZMIN=Z(I)-RMIN(I)
       ENDIF
    ENDDO
    !
    DELX=XMAX-XMIN
    DELY=YMAX-YMIN
    DELZ=ZMAX-ZMIN
    A=(SPACE/(DELX*DELY*DELZ))**(ONE/THREE)
    IXGRID=A*DELX
    IYGRID=A*DELY
    IZGRID=A*DELZ
    IF(IXGRID <= 1) IXGRID=2
    IF(IYGRID <= 1) IYGRID=2
    IF(IZGRID <= 1) IZGRID=2
    !
    if (prnlev >= 2) WRITE(OUTU,22) XMIN,XMAX,IXGRID,YMIN,YMAX,IYGRID,ZMIN,ZMAX,IZGRID
22  FORMAT(' COMPUTING VOLUME PER ATOM WITH:'/, &
         '   XMIN=',F12.4,' XMAX=',F12.4,' NX=',I4/, &
         '   YMIN=',F12.4,' YMAX=',F12.4,' NY=',I4/, &
         '   ZMIN=',F12.4,' ZMAX=',F12.4,' NZ=',I4/)
    !
    ISPACE=IXGRID*IYGRID*IZGRID
    IF(ISPACE > SPACE) THEN
       CALL WRNDIE(0,'<VOLUME>','SPACE DECLARATION TOO SMALL')
       RETURN
    ENDIF
    call chmalloc('corman2.src','VOLUME','jMAT',ISPACE,intg=jMAT)

    call chmalloc('corman2.src','volume','space_MAT',IXGrid+iygrid+izgrid,crl=space_MAT)
    xmat => space_mat(1:ixgrid)
    ymat => space_mat(ixgrid+1:ixgrid+iygrid)
    zmat => space_mat(ixgrid+iygrid+1:ixgrid+iygrid+izgrid)


    CALL VSEARC(NATOM,X,Y,Z,WMAIN,IXGRID,IYGRID,IZGRID, &
         xmat,ymat,zmat,XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,ISLCT,.FALSE.,OUTU, &
         jMAT,jmat,0,.FALSE.,LHOLES,.TRUE., &
         0,RMIN,RMAX)
    call chmdealloc('corman2.src','VOLUME','jMAT',ISPACE,intg=jMAT)
    call chmdealloc('corman2.src','vsearc','space_MAT',IXGrid+iygrid+izgrid,crl=space_MAT)

    RETURN
  END SUBROUTINE VOLUME

  SUBROUTINE VSEARC(NATOM,X,Y,Z,W,IXG,IYG,IZG, &
       xmat,ymat,zmat,XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,ISLCT,LPRINT,IUNIT, &
       JMAT,KMAT,PSOPER,LVACUM,LHOLES,LVOLUM,NSAVE,RMIN,RMAX)
    !-----------------------------------------------------------------------
    !
    !      THIS ROUTINE SEARCHES IN THE SPECIFIED VOLUME FOR HOLES.
    !      THE WEIGHTING ARRAY IS USED AS THE HARD SPHERE RADIUS FOR EACH
    !      ATOM.
    !
    !       BERNARD R. BROOKS   5-DEC-1983
    !
    !      NATOM          - Number of atoms for the analysis
    !      X,Y,Z(NATOM)   - Coordinates of atoms
    !      W(NATOM)       - Weighting array
    !      IXG,IYG,IZG    - Number of grid points in each direction
    !      XMAT,YMAT,ZMAT - coordinates of grid points, locally allocated and deallocated
    !      XMIN,XMAX,...  - ranges of the grid space
    !      ISLCT(NATOM)   - Atom selection array
    !      LPRINT         - Flag to print of write resulting grid points
    !      IUNIT          - Unit number for writing grid points
    !      no longer used: IMAT(IZG,IYG,IXG) - Matrix indicating free volumes
    !      JMAT(NPTS)        - Same matrix, but one dimensional
    !      KMAT(NPTS)        - Matrix from previous search command.
    !      PSOPER         - 0=ignore previous search command.
    !                       1=AND with previous results
    !                       2=OR  with previous results
    !                       3=XOR with previous results
    !      LVACUM  .true. - print or plot vacume points
    !              .false.- print of display filled points
    !      LHOLES         - Just find inacessible holes
    !      LVOLUME        - Find fractional free volume for each atom
    !      NSAVE          - Number of previous calculations(for fraction)
    !      RMIN(NATOM)    - Inner radii for each atom
    !      RMAX(NATOM)    - Outer radii for each atom
    !      IUTIL(NATOM)   - Utility array for calculations
    !
    !
  use consta
  use memory
  use number
  use param_store, only: set_param
  use polymer
  use stream
  use string

  implicit none

    integer natom
    real(chm_real) x(*),y(*),z(*),w(*)
    integer ixg,iyg,izg
    real(chm_real) xmin,xmax,ymin,ymax,zmin,zmax
    integer islct(*)
    logical lprint
    integer iunit
    integer jmat(*),kmat(*)  
    integer psoper
    logical lvacum,lholes,lvolum
    integer nsave
    real(chm_real) rmin(*),rmax(*)
    integer,allocatable,dimension(:) :: iutil
    !
    !
    integer :: ixbas,iybas,izbas
    INTEGER NPTS,I,IAT,NVAC,IPT,IX,IY,IZ,NPOSS,NFOUND
    INTEGER IXMIN,IYMIN,IZMIN,IXMAX,IYMAX,IZMAX
    INTEGER NOCC,NOUT,DUMIDL
    INTEGER IPASS,IADD,NOCH,NNOCH,IYGIZG,IXX,JXX,NEXT
    integer,PARAMETER :: DUMIDM=6
    character(len=DUMIDM) DUMID
    real(chm_real) XSTEP,YSTEP,ZSTEP,VOL,W2,XN,YN,PI43,XVAL,YVAL,ZVAL
    real(chm_real) SELVOL
    LOGICAL OK,XOK,YOK,ZOK,LPRNT
    real(chm_real),dimension(*) :: xmat,ymat,zmat

    call chmalloc('corman2.src','VOLUME','IUTIL',NATOM,intg=IUTIL)

    LPRNT = (LPRINT .AND. PRNLEV >= 2)
    !
    NPTS=IXG*IYG*IZG
    ! Clear the grid
    DO I=1,NPTS
       JMAT(I)=0
    ENDDO
    XSTEP=(XMAX-XMIN)/IXG
    YSTEP=(YMAX-YMIN)/IYG
    ZSTEP=(ZMAX-ZMIN)/IZG
    VOL=XSTEP*YSTEP*ZSTEP
    PI43=(FOUR*PI)/(THREE*VOL)
    !
    DO I=1,IXG
       XMAT(I)=XSTEP*(I-HALF)+XMIN
    ENDDO
    DO I=1,IYG
       YMAT(I)=YSTEP*(I-HALF)+YMIN
    ENDDO
    DO I=1,IZG
       ZMAT(I)=ZSTEP*(I-HALF)+ZMIN
    ENDDO
    !
    IF(LPRNT) THEN
       WRITE(OUTU,122) (I,XMAT(I),I=1,IXG)
       WRITE(OUTU,123) (I,YMAT(I),I=1,IYG)
       WRITE(OUTU,124) (I,ZMAT(I),I=1,IZG)
122    FORMAT(' XGRID POINTS:'/5(I5,F10.4))
123    FORMAT(' YGRID POINTS:'/5(I5,F10.4))
124    FORMAT(' ZGRID POINTS:'/5(I5,F10.4))
       !
       IF(OUTU /= IUNIT .AND. IOLEV > 0) THEN
          WRITE(IUNIT,126) '* Vacuum search points'
          WRITE(IUNIT,126) '*'
          WRITE(IUNIT,126) '    0'
126       FORMAT(A)
       ENDIF
    ENDIF
    !
    ! Fill the grid points around each atom
    !
    DO IAT=1,NATOM
       IF(ISLCT(IAT) == 1) THEN
          IUTIL(IAT)=0
          IXMIN=(X(IAT)-RMIN(IAT)-XMIN)/XSTEP+ONE
          IYMIN=(Y(IAT)-RMIN(IAT)-YMIN)/YSTEP+ONE
          IZMIN=(Z(IAT)-RMIN(IAT)-ZMIN)/ZSTEP+ONE
          IXMAX=(X(IAT)+RMIN(IAT)-XMIN)/XSTEP+ONE
          IYMAX=(Y(IAT)+RMIN(IAT)-YMIN)/YSTEP+ONE
          IZMAX=(Z(IAT)+RMIN(IAT)-ZMIN)/ZSTEP+ONE
          W2=RMIN(IAT)*RMIN(IAT)
          OK=.TRUE.
          IF(IXMIN > IXG) OK=.FALSE.
          IF(IYMIN > IYG) OK=.FALSE.
          IF(IZMIN > IZG) OK=.FALSE.
          IF(IXMAX < 1) OK=.FALSE.
          IF(IYMAX < 1) OK=.FALSE.
          IF(IZMAX < 1) OK=.FALSE.
          IF (OK) THEN
             IXMIN=MAX(IXMIN,1)
             IYMIN=MAX(IYMIN,1)
             IZMIN=MAX(IZMIN,1)
             IXMAX=MIN(IXMAX,IXG)
             IYMAX=MIN(IYMAX,IYG)
             IZMAX=MIN(IZMAX,IZG)
             !
             XVAL=XSTEP*(IXMIN-1.5)+XMIN-X(IAT)
             DO IX=IXMIN,IXMAX
                ixbas=(ix-1)*iyg*izg
                XVAL=XVAL+XSTEP
                XN=XVAL*XVAL
                YVAL=YSTEP*(IYMIN-1.5)+YMIN-Y(IAT)
                DO IY=IYMIN,IYMAX
                   iybas=ixbas + (iy-1)*izg
                   YVAL=YVAL+YSTEP
                   YN=YVAL*YVAL+XN
                   IF(W2 > YN) THEN
                      ZVAL=ZSTEP*(IZMIN-1.5)+ZMIN-Z(IAT)
                      DO IZ=IZMIN,IZMAX
                         izbas = iybas + iz
                         ZVAL=ZVAL+ZSTEP
                         IF(W2 > ZVAL*ZVAL+YN) THEN
                            jMAT(IZbas)=1
                            IUTIL(IAT)=IUTIL(IAT)+1
                         ENDIF
                      ENDDO
                   ENDIF
                ENDDO
             ENDDO
             !
             !      IF(LPRNT) WRITE(33,244) IAT,IUTIL(IAT),W2
             ! 244  FORMAT(/' IAT,IUTIL(IAT),W2'/,2I4,F15.4)
             !      IF(LPRNT) WRITE(33,245) IXMIN,IXMAX,IYMIN,IYMAX,IZMIN,IZMAX
             ! 245  FORMAT(' IXMIN,IXMAX,IYMIN,IYMAX,IZMIN,IZMAX'/,6I4)
             !      IF(LPRNT) WRITE(33,246) X(IAT),Y(IAT),Z(IAT),XVAL,YVAL,ZVAL
             ! 246  FORMAT(' X(IAT),Y(IAT),Z(IAT),XVAL,YVAL,ZVAL'/,6F12.4)
             !
          ENDIF
          IF(RMIN(IAT) > ZERO) THEN
             W(IAT)=IUTIL(IAT)/(PI43*RMIN(IAT)*W2)
          ELSE
             W(IAT)=ZERO
          ENDIF
          !            IF(W(IAT) < 0.9) ...
       ENDIF
    ENDDO
    !
    ! Now find holes if requested by reseting all others
    !
    IF(LHOLES) THEN
       IADD=0
       DO IX=1,IXG
          DO IY=1,IYG
             IF(JMAT(IADD+1) == 0) JMAT(IADD+1)=-1
             IADD=IADD+IZG
             IF(JMAT(IADD) == 0) JMAT(IADD)=-1
          ENDDO
       ENDDO
       !
       IXX=0
       IYGIZG=IYG*IZG
       JXX=IYGIZG-IZG
       DO IX=1,IXG
          IADD=IXX
          DO IZ=1,IZG
             IADD=IADD+1
             IF(JMAT(IADD) == 0) JMAT(IADD)=-1
             IF(JMAT(IADD+JXX) == 0) JMAT(IADD+JXX)=-1
          ENDDO
          IXX=IXX+IYGIZG
       ENDDO
       !
       IADD=0
       IXX=NPTS-IYGIZG
       DO IY=1,IYG
          DO IZ=1,IZG
             IADD=IADD+1
             IF(JMAT(IADD) == 0) JMAT(IADD)=-1
             IF(JMAT(IADD+IXX) == 0) JMAT(IADD+IXX)=-1
          ENDDO
       ENDDO
       !
       !
       IPASS=0
       NNOCH=0
       DO WHILE(NNOCH < 7)
          NOCH=0
          IPASS=IPASS+1
          IF(IPASS > 6) IPASS=1
          IF(IPASS == 1) THEN
             ! +X PASS
             IADD=IYGIZG
             DO IX=2,IXG
                DO IY=1,IYGIZG
                   IADD=IADD+1
                   IF(JMAT(IADD) == 0) THEN
                      IF(JMAT(IADD-IYGIZG) == -1) THEN
                         JMAT(IADD)=-1
                         NOCH=NOCH+1
                      ENDIF
                   ENDIF
                ENDDO
             ENDDO
          ELSE IF(IPASS == 2) THEN
             ! +Y PASS
             IADD=0
             DO IX=1,IXG
                IADD=IADD+IZG
                DO IY=2,IYG
                   DO IZ=1,IZG
                      IADD=IADD+1
                      IF(JMAT(IADD) == 0) THEN
                         IF(JMAT(IADD-IZG) == -1) THEN
                            JMAT(IADD)=-1
                            NOCH=NOCH+1
                         ENDIF
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
          ELSE IF(IPASS == 3) THEN
             ! +Z PASS
             IADD=0
             DO IX=1,IXG
                DO IY=1,IYG
                   IADD=IADD+1
                   DO IZ=2,IZG
                      IADD=IADD+1
                      IF(JMAT(IADD) == 0) THEN
                         IF(JMAT(IADD-1) == -1) THEN
                            JMAT(IADD)=-1
                            NOCH=NOCH+1
                         ENDIF
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
          ELSE IF(IPASS == 4) THEN
             ! -X PASS
             IADD=NPTS-IYGIZG-IYGIZG
             DO IX=2,IXG
                DO IY=1,IYGIZG
                   IADD=IADD+1
                   IF(JMAT(IADD) == 0) THEN
                      IF(JMAT(IADD+IYGIZG) == -1) THEN
                         JMAT(IADD)=-1
                         NOCH=NOCH+1
                      ENDIF
                   ENDIF
                ENDDO
                IADD=IADD-IYGIZG-IYGIZG
             ENDDO
          ELSE IF(IPASS == 5) THEN
             ! -Y PASS
             IADD=IYGIZG-IZG-IZG
             DO IX=1,IXG
                DO IY=2,IYG
                   DO IZ=1,IZG
                      IADD=IADD+1
                      IF(JMAT(IADD) == 0) THEN
                         IF(JMAT(IADD+IZG) == -1) THEN
                            JMAT(IADD)=-1
                            NOCH=NOCH+1
                         ENDIF
                      ENDIF
                   ENDDO
                   IADD=IADD-IZG-IZG
                ENDDO
                IADD=IADD+IYGIZG+IYGIZG-IZG
             ENDDO
          ELSE IF(IPASS == 6) THEN
             ! -Z PASS (Bugfix B960605.bm, +Z -> -Z)
             IADD=IZG
             DO IX=1,IXG
                DO IY=1,IYG
                   DO IZ=2,IZG
                      IADD=IADD-1
                      IF(JMAT(IADD) == 0) THEN
                         IF(JMAT(IADD+1) == -1) THEN
                            JMAT(IADD)=-1
                            NOCH=NOCH+1
                         ENDIF
                      ENDIF
                   ENDDO
                   ! Bugfix B960605.bm, IADD=IADD+IZG+IZG to
                   IADD=IADD-1+IZG+IZG
                ENDDO
             ENDDO
          ENDIF
          !
          IF(NOCH == 0) THEN
             NNOCH=NNOCH+1
          ELSE
             NNOCH=0
          ENDIF
          IF(PRNLEV > 6) WRITE(OUTU,26) IPASS,NOCH,NNOCH
26        FORMAT('      IPASS=',I3,'  NOCH=',I8,'  NNOCH=',I3)
       ENDDO
       !
    ENDIF
    !
    ! Compute accesible volume for each atom
    !
    IF(LVOLUM) THEN
       !
       DO IAT=1,NATOM
          IF(ISLCT(IAT) == 1) THEN
             W2=RMAX(IAT)*RMAX(IAT)
             NPOSS=0
             NFOUND=0
             IXMIN=(X(IAT)-RMAX(IAT)-XMIN)/XSTEP+ONE
             IYMIN=(Y(IAT)-RMAX(IAT)-YMIN)/YSTEP+ONE
             IZMIN=(Z(IAT)-RMAX(IAT)-ZMIN)/ZSTEP+ONE
             IXMAX=(X(IAT)+RMAX(IAT)-XMIN)/XSTEP+ONE
             IYMAX=(Y(IAT)+RMAX(IAT)-YMIN)/YSTEP+ONE
             IZMAX=(Z(IAT)+RMAX(IAT)-ZMIN)/ZSTEP+ONE
             !
             XVAL=XSTEP*(IXMIN-1.5)+XMIN-X(IAT)
             DO IX=IXMIN,IXMAX
                ixbas=(ix-1)*iyg*izg
                XVAL=XVAL+XSTEP
                XN=XVAL*XVAL
                XOK=(IX > 0.AND.IX <= IXG)
                YVAL=YSTEP*(IYMIN-1.5)+YMIN-Y(IAT)
                DO IY=IYMIN,IYMAX
                   iybas=ixbas + (iy-1)*izg
                   YVAL=YVAL+YSTEP
                   YN=YVAL*YVAL+XN
                   IF(W2 > YN) THEN
                      YOK=(IY > 0.AND.IY <= IYG)
                      ZVAL=ZSTEP*(IZMIN-1.5)+ZMIN-Z(IAT)
                      DO IZ=IZMIN,IZMAX
                         izbas=iybas + iz
                         ZVAL=ZVAL+ZSTEP
                         IF(W2 > ZVAL*ZVAL+YN) THEN
                            NPOSS=NPOSS+1
                            ZOK=(IZ > 0.AND.IZ <= IZG)
                            IF(XOK.AND.YOK.AND.ZOK) THEN
                               IF(jMAT(IZbas) /= 0) NFOUND=NFOUND+1
                            ENDIF
                         ENDIF
                      ENDDO
                   ENDIF
                ENDDO
             ENDDO
             !
             W(IAT)=NPOSS-NFOUND
             IF(NPOSS-IUTIL(IAT) > 0) THEN
                W(IAT)=W(IAT)/(NPOSS-IUTIL(IAT))
             ELSE
                W(IAT)=ZERO
             ENDIF
             !               IF(PRNLEV >= 2) WRITE(OUTU,322)
             !    &                     IAT,IUTIL(IAT),(NFOUND-IUTIL(IAT)),NPOSS
             ! 322           FORMAT(I5,' IN=',I10,' OCC=',I10,' POSS=',I10)
          ENDIF
       ENDDO
    ENDIF
    !
    ! Count each pixel type
    NEXT=0
    NVAC=0
    NOCC=0
    DO IPT=1,NPTS
       IF(JMAT(IPT) == -1) THEN
          NEXT=NEXT+1
          JMAT(IPT)=1
       ELSE IF(JMAT(IPT) == 0) THEN
          NVAC=NVAC+1
       ELSE IF(JMAT(IPT) == 1) THEN
          NOCC=NOCC+1
       ENDIF
    ENDDO
    !
    IF(LVACUM) THEN
       DO IPT=1,NPTS
          JMAT(IPT)=1-JMAT(IPT)
       ENDDO
    ENDIF
    !
    ! Merge current pixel list with previous one.
    !
    IF(PSOPER > 0) THEN
       IF(PSOPER == 1) THEN
          ! AND function
          DO IPT=1,NPTS
             JMAT(IPT)=JMAT(IPT)*KMAT(IPT)
          ENDDO
          ! OR function
       ELSE IF(PSOPER == 2) THEN
          DO IPT=1,NPTS
             JMAT(IPT)=JMAT(IPT)+KMAT(IPT)
             IF(JMAT(IPT) > 1) JMAT(IPT)=1
          ENDDO
          ! XOR function
       ELSE IF(PSOPER == 3) THEN
          DO IPT=1,NPTS
             IF(KMAT(IPT) > 1) KMAT(IPT)=1
             JMAT(IPT)=JMAT(IPT)+KMAT(IPT)-2*JMAT(IPT)*KMAT(IPT)
          ENDDO
          ! Add function
       ELSE IF(PSOPER == 4) THEN
          DO IPT=1,NPTS
             JMAT(IPT)=JMAT(IPT)+KMAT(IPT)
          ENDDO
       ENDIF
    ENDIF
    !
    ! Now write out the pixels which are still nonzero.
    !
    NOUT=0
    IPT=0
    W2=0.0
    DO IX=1,IXG
       DO IY=1,IYG
          DO IZ=1,IZG
             IPT=IPT+1
             IF(JMAT(IPT) >= 1) THEN
                NOUT=NOUT+1
                IF(NSAVE > 0) W2=JMAT(IPT)/(NSAVE+ONE)
                IF(LPRNT .AND. NOUT <= 9999) THEN
                   CALL ENCODI(NOUT,DUMID,DUMIDM,DUMIDL)
                   IF(IOLEV > 0) WRITE(IUNIT,132) NOUT,NOUT,'DUM ', &
                        'DUM ',XMAT(IX),YMAT(IY),ZMAT(IZ), &
                        'VAC ',DUMID,W2,IX,IY,IZ
132                FORMAT(2I5,1X,A4,1X,A4,3F10.5,1X,A4,1X,A4,F10.5,3I5)
                ENDIF
             ENDIF
             !
          ENDDO
       ENDDO
    ENDDO
    !
    VOLUME=VOL*NOCC
    FREVOL=VOL*NVAC
    FFVOL = FREVOL/(VOLUME+FREVOL)*100.0
    SELVOL=VOL*NOUT
    IF(PRNLEV >= 2) WRITE(OUTU,25) &
         NVAC,NOCC,NEXT,NOUT,VOLUME,SELVOL,FREVOL,FFVOL
25  FORMAT(/' A TOTAL OF',I9,'   VACUUM POINTS WERE FOUND'/ &
         ' A TOTAL OF',I9,' OCCUPIED POINTS WERE FOUND'/ &
         ' A TOTAL OF',I9,' EXTERNAL POINTS WERE FOUND'/ &
         ' A TOTAL OF',I9,' SELECTED POINTS WERE FOUND'/ &
         ' TOTAL OCCUPIED  VOLUME =',F15.6/ &
         ' TOTAL SELECTED  VOLUME =',F15.6/ &
         ' TOTAL FREE      VOLUME =',F15.6/ &
         ' FRACTIONAL FREE VOLUME =',F15.6)
    CALL set_param('NVAC',NVAC)
    CALL set_param('NOCC',NOCC)
    CALL set_param('NSEL',NOUT)
    call set_param('VOLUME',VOLUME)
    call set_param('VOLU',VOLUME)
    call set_param('FREEVOL',FREVOL)
    !
    call chmdealloc('corman2.src','VOLUME','IUTIL',NATOM,intg=IUTIL)
    RETURN
  END SUBROUTINE VSEARC

  SUBROUTINE COVARI()
    !-----------------------------------------------------------------------
    ! computes covariances of the spatial atom displacements of
    ! a dynamics trajectory for selected pairs of atoms.
    !
    ! mu  = E[ (R   - E[R ])  (R  -  E[R ] )
    !   JK         J       J      K       K
    !
    !     = E[R R ]    -   E[R ]  E[R ]
    !            J K            J      K
    !
    ! and the normalized covariance matrix is given by
    !
    ! CO  = mu  / SQRT(mu   mu  )
    !   JK      JK         JJ   KK
    !
    ! SYNTAX:
    !   CHARMM> COORdinates COVAriance -
    !           FIRSt_unit <int> NUNIt <int> BEGIn <int> SKIP <int> -
    !           STOP <int> 2x<atom selection (SET1, SET2)> -
    !           UNIT_for_output <int>  RESIdue_average <nsets>
    !
    !**********************************************************************C
    !                                                                      C
    !                   Written by C. L. Brooks III                        C
    !                   Spring, 1983                                       C
    !                                                                      C
    !**********************************************************************C
    !
  use consta
  use dimens_fcm
    ! input/output
  use comand
  use select
  use stream
  use string
  use psf
  use coord
  use coordc
  use memory
  use cvio,only:trjspc
  use chutil,only:atomid

    integer siz_rl1,codim1,codim2,cordim2,xsdim,wkdim,xxdim
    real(chm_real),allocatable,dimension(:,:) :: sub,co,cored
    real(chm_real),allocatable,dimension(:),target :: space_rl1,space_rl2
    real(chm_real),pointer,dimension(:) :: wk,xx,xsum,ysum,zsum,x2sum,y2sum,z2sum, &
         wsum,xl,yl,zl

    INTEGER NSET1, NSET2, UNIT, NRES2, nres22
    INTEGER BEGIN, SKIP, STOP, NUNIT, FIRSTU

    integer,allocatable,dimension(:),target :: space_int1
    integer,allocatable,dimension(:),target :: space_int2
    integer,pointer,dimension(:) :: set1,set2,freeat,ind1,ind2


    LOGICAL ERROR, LRESI, LMAT, LENTR, LDIAG, LSPRIM, LCHOL, LENTRES, LDIID

    ! Variables for DICC --> Amitava Roy 12/08/2016
    logical ldicc,ldcor,ldcov
    ! <-- Amitava Roy 12/08/2016

    character(len=8) RIDO
    character(len=8) SID,RID,REN,AC
    INTEGER I,NDIM,N,NMAX
    real(chm_real) TMPR
    !
    call chmalloc('corman2.src','covari','space_int1',2*natom,intg=space_int1)
    set1 => space_int1(1:natom)
    set2 => space_int1(natom+1:2*natom)

    ERROR=.FALSE.
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,BEGIN,SKIP,STOP)
    UNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
    IF(PRNLEV >= 3) WRITE(OUTU,'(A,I4)') &
         ' COVARI: Covariance matrix output to unit',UNIT
    CALL SELCTD(COMLYN,COMLEN,SET1,SET2,X,Y,Z,WMAIN,.TRUE.,ERROR)
    NSET1=NSELCT(NATOM,SET1)
    NSET2=NSELCT(NATOM,SET2)
    IF(PRNLEV >= 3) WRITE(OUTU,'(A,I8)') &
         ' COVARI: Number of atoms selected for SET1',NSET1
    IF(PRNLEV >= 3) WRITE(OUTU,'(A,I8)') &
         ' COVARI: Number of atoms selected for SET2',NSET2
    !
    LMAT = (INDXA(COMLYN,COMLEN,'MATR') > 0)
    LENTR = (INDXA(COMLYN,COMLEN,'ENTR') > 0)
    LDIID = (INDXA(COMLYN,COMLEN,'DIID') > 0)

    ! --> Amitava Roy 12/08/2016
    ldcor = (indxa(comlyn,comlen,'DCOR') > 0)
    ldcov = (indxa(comlyn,comlen,'DCOV') > 0)
    ldicc = .false.

    if (ldiid .and. (ldcor .or. ldcov)) then
      call wrndie(-2,'<COVARI>', &
        'DCOR IS NOT COMPATIBLE WITH DIID; DISABLING DCOR')
      ldcor = .false.
      ldcov = .false.
    end if

    if (ldcor.or.ldcov) ldicc=.true.
    if (ldcor.and.ldcov) ldcor=.false.
    if (ldicc) lentr=.false.
    ! <-- Amitava Roy 12/08/2016   

    IF(LENTR)THEN
       !
       ! Entropy
       IF(NSET1 /= NSET2)THEN
          CALL WRNDIE(-2,'<COVARI>', &
               'UNEQUAL SETS NOT ALLOWED FOR ENTROPY CALC')
       ENDIF
       LRESI=(INDXA(COMLYN,COMLEN,'RESI') > 0)
       IF(LRESI.AND.PRNLEV >= 2) WRITE(OUTU,'(/5X,A)') &
            'Total entropy/residue will be analyzed'
       LDIAG=(INDXA(COMLYN,COMLEN,'DIAG')  >  0)
       ! S'' of Andricioaei & Karplus is default, SCHLitter specifies to
       ! do Schlitter's S'  approximation
       LSPRIM=(INDXA(COMLYN,COMLEN,'SCHL')  >  0)
       ! LU-decomp is default, but we parse LUDC also
       LCHOL=(INDXA(COMLYN,COMLEN,'CHOL')  >  0)
       IF(INDXA(COMLYN,COMLEN,'LUDC')  >  0) LCHOL=.FALSE.
       TMPR=GTRMF(COMLYN,COMLEN,'TEMP',ROOMT)
       IF(PRNLEV >= 2) WRITE(OUTU,'(/A,F7.1)') &
            'Estimating entropy at temperature [K]=',TMPR
       NRES2=0
    ELSE
       NRES2=GTRMI(COMLYN,COMLEN,'RESI',0)
       LRESI = .FALSE.
    ENDIF
    IF(LDIID) THEN
       IF(PRNLEV >= 2) THEN
          WRITE(OUTU,*) 'Write out matrix for Direct-ID analysis'
       ENDIF
       LENTR=.FALSE.
       LMAT=.TRUE.
       LRESI=.FALSE.
    ENDIF
    !
    IF (NRES2 > 0) THEN
       IF (NRES2 < 2) NRES2 = 0
       LRESI = .TRUE.
       IF(LMAT)THEN
          LMAT=.FALSE.
          WRITE(OUTU,'(A)') 'Cannot do MATRIX output w/ RESI ave'
       ENDIF
       IF(PRNLEV >= 3) WRITE(OUTU,'(A)') &
            ' COVARI: Residue average will be taken over SET2'
       IF (NRES2 > 1) THEN
          NRES2 = 0
          RIDO='    '
          DO I=1,NATOM
             IF(SET2(I) == 1) THEN
                CALL ATOMID(I,SID,RID,REN,AC)
                IF (RID /= RIDO) THEN
                   RIDO=RID
                   NRES2 = NRES2 + 1
                ENDIF
             ENDIF
          ENDDO
          IF(PRNLEV >= 3) WRITE(OUTU,'(A,2X,I5)') &
               ' COVARI: Number of residues in SET2 is', NRES2
          IF(PRNLEV >= 3) WRITE(OUTU,'(A)') &
               ' COVARI: Residue average will be taken over SET1 as well'
       ENDIF
       !
    ENDIF
    IF (.NOT.ERROR.AND.NSET1 > 0.AND.NSET2 > 0) THEN
       !
       ! dynamic

       LENTRES=.FALSE.
       IF(LENTR.OR.LDIID)THEN
          IF(LRESI)THEN
             LENTRES=.TRUE.
             !
             !               IF(LCHOL)THEN
             ! find maximum number of selected atoms in a residue
             !                  RIDO='    '
             !                  NMAX=0
             !                  N=0
             !                  DO I=1,NATOM
             !                     IF(SET1(I) > 0)THEN
             !                        CALL ATOMID(I,SID,RID,REN,AC)
             !                        IF(RID /= RIDO)THEN
             !                           RIDO=RID
             !                           NMAX=MAX(N,NMAX)
             !                           N=1
             !                        ELSE
             !                           N=N+1
             !                        ENDIF
             !                     ENDIF
             !                  ENDDO
             !                  NMAX=MAX(N,NMAX)
             !               ELSE
             ! for LU-decomp we need the full dimension
             !                  NMAX=3*NSET1
             !               ENDIF
             NMAX=3*NSET1
             call chmalloc('corman2.src','covari','sub',nmax,nmax,crl=sub)
          ELSE
             call chmalloc('corman2.src','covari','sub',1,1,crl=sub)
             NMAX=1
          ENDIF
          NDIM=3*NSET1
          siz_rl1 = 4*ndim
          call chmalloc('corman2.src','covari','space_rl1',siz_rl1,crl=space_rl1)
          wk  => space_rl1(        1 : 2*ndim)
          xx  => space_rl1( 2*ndim+1 : 3*ndim)
          xsum => space_rl1(3*ndim+1 : 4*ndim)
          wkdim=2*ndim
          xxdim=ndim
          xsdim=ndim
          codim1 = ndim
          codim2 = ndim
       ELSE
          siz_rl1 = natom+2
          call chmalloc('corman2.src','covari','space_rl1',siz_rl1,crl=space_rl1)
          wk  => space_rl1( 1 : 1)
          xx  => space_rl1( 2 : 2)
          xsum => space_rl1(3 : natom+2)
          wkdim=1
          xxdim=1
          xsdim=natom
          call chmalloc('corman2.src','covari','sub',1,1,crl=sub)
          codim1 = nset1
          codim2 = nset2
          NDIM=NSET1
       ENDIF
       call chmalloc('corman2.src','covari','co',codim1,codim2,crl=CO)
       call chmalloc('corman2.src','covari','space_rl2',9*natom,crl=space_rl2)
       ysum  => space_rl2(        1 :   natom)
       zsum  => space_rl2(  natom+1 : 2*natom)
       x2sum => space_rl2(2*natom+1 : 3*natom)
       y2sum => space_rl2(3*natom+1 : 4*natom)
       z2sum => space_rl2(4*natom+1 : 5*natom)
       wsum  => space_rl2(5*natom+1 : 6*natom)
       xl    => space_rl2(6*natom+1 : 7*natom)
       yl    => space_rl2(7*natom+1 : 8*natom)
       zl    => space_rl2(8*natom+1 : 9*natom)

       call chmalloc('corman2.src','covari','space_int2',natom+nset1+nset2,intg=space_int2)
       freeat => space_int2(1:natom)
       ind1   => space_int2(natom+1:natom+nset1)
       ind2   => space_int2(natom+nset1+1:natom+nset1+nset2)

       IF(NRES2 >= 1) then
          cordim2=nres2
       else
          NRES22 = 0
          RIDO='    '
          DO I=1,NATOM
             IF(SET2(I) == 1) THEN
                CALL ATOMID(I,SID,RID,REN,AC)
                IF (RID /= RIDO) THEN
                   RIDO=RID
                   NRES22 = NRES22 + 1
                ENDIF
             ENDIF
          ENDDO
          cordim2=nres22
       endif
       call chmalloc('corman2.src','COVARI','CORED',NSET1, cordim2,crl=CORED)

       CALL COVAR2(UNIT,NATOM,XSUM,xsdim,YSUM,ZSUM, &
            X2SUM,Y2SUM,Z2SUM,WSUM, &
            FREEAT, &
            BEGIN,SKIP,STOP,NUNIT,FIRSTU, &
            NSET1,SET1,NSET2,SET2,NDIM, &
            XL,YL,ZL, &
            IND1,IND2, &
            CO,codim1,codim2,LRESI,NRES2,CORED,cordim2,LMAT, &
            LDIID, &
            LENTR,XX,xxdim,TMPR,WK,wkdim,AMASS,LDIAG,LSPRIM, &
            LCHOL,SUB,NMAX, &
            ldicc,ldcov,ldcor)  !dcor --> Amitava Roy 12/12/2016
       !
       IF(LENTRES) then 
          call chmdealloc('corman2.src','covari','sub',nmax,nmax,crl=sub)
       else
          call chmdealloc('corman2.src','covari','sub',1,1,crl=sub)
       endif
       call chmdealloc('corman2.src','covari','co',codim1,codim2,crl=CO)
       call chmdealloc('corman2.src','COVARI','CORED',NSET1,cordim2,crl=CORED)
       call chmdealloc('corman2.src','covari','space_rl1',siz_rl1,crl=space_rl1)
       nullify(wk,xx,xsum)
       call chmdealloc('corman2.src','covari','space_rl2',9*natom,crl=space_rl2)
       nullify(ysum,zsum,x2sum,y2sum,z2sum,wsum,xl,yl,zl)
    ENDIF
    call chmdealloc('corman2.src','covari','space_int1',2*natom,intg=space_int1)
    call chmdealloc('corman2.src','covari','space_int2',natom+nset1+nset2,intg=space_int2)
    RETURN
  END SUBROUTINE COVARI

  SUBROUTINE COVAR2(UNIT,NATOM,XSUM,xsdim,YSUM,ZSUM,X2SUM,Y2SUM,Z2SUM, &
       WSUM,FREEAT, &
       BEGIN,SKIP,STOP,NUNIT,FIRSTU, &
       NSET1,SET1,NSET2,SET2,NDIM, &
       XL,YL,ZL,IND1,IND2, &
       CO,codim1,codim2,LRESI,NRES2,CORED,cordim2,LMAT,LDIID, &
       LENTR,XX,xxdim,TMPR,WK,wkdim,AMASS,LDIAG,LSPRIM,LCHOL, &
       COSAV,NMAX, &
       ldicc,ldcov,ldcor) !dcor --> Amitava Roy 12/12/2016
    !
    !
  use chutil, only: atomid
  use consta
  use ctitla
  use cvio
  use memory
  use coordc
  use machutil, only: wrttim
  use number
  use param_store, only: set_param
  use stream
  use timerm

! For DICC --> Amitava Roy 12/08/2016
  use dcor_network
  use rmsdyn_mod,only:rdtrj
#if KEY_PARALLEL==1
  use parallel
#endif
! <-- Amitava Roy 12/08/2016

  implicit none

    INTEGER UNIT
    INTEGER NATOM,codim1,codim2,cordim2,xsdim,xxdim,wkdim
    real(chm_real) XSUM(xsdim),YSUM(natom),ZSUM(natom)
    real(chm_real) X2SUM(natom),Y2SUM(natom),Z2SUM(natom)
    real(chm_real) WSUM(natom)
    INTEGER FREEAT(natom)
    INTEGER BEGIN, SKIP, STOP, NUNIT, FIRSTU
    INTEGER NSET1, NSET2, NRES2, NDIM, NMAX
    INTEGER SET1(natom),SET2(Natom)
    real(chm_real) XL(natom), YL(natom), ZL(natom),TMPR,WK(wkdim),AMASS(*),XX(xxdim)
    INTEGER IND1(NSET1), IND2(NSET2)
    real(chm_real) CO(codim1,codim2), CORED(NSET1,cordim2),COSAV(NMAX,NMAX)
    LOGICAL LRESI,LMAT,LENTR,LDIAG,LSPRIM,LCHOL,LDIID
    ! local
    INTEGER NCOORD, I, I1, I2, ISTATS, NFREAT
    INTEGER ISTEP, IUNIT, NDEGF, NFILE, NSAV
    INTEGER IRES2, NRES1
    INTEGER NUMAT,IER,IDUM,N,IFIRST,NIGNORE
    LOGICAL LENTRES
    real(chm_real) COCUM,S,D1,D2,COEFF,COF1,COF2,SUM,SFULL
    real(chm_real) DELTA
    character(len=20) MATFMT
    character(len=8) SID,RID,REN,AC
    character(len=8) RIDO,RIDO2,SIDO,RENO
    character(len=4),parameter :: HDR='CORD'
    REAL(chm_real4),dimension(:),allocatable :: TEMP
    ! Variables for DICC --> Amitava Roy 12/08/2016
    integer,allocatable,dimension(:)                    :: aislc,ajslc
    real(chm_real4),allocatable,dimension(:,:,:),target :: xyz   
    real(chm_real),allocatable,dimension(:,:)           :: t1,t2
    real(chm_real)                                      :: dvar1,dvar2,dcov,dcor
    logical ldicc,ldcor,ldcov
    ! <-- Amitava Roy 12/08/2016
    ! begin
    !

    call chmalloc('corman2.src','covar2','temp',natom,cr4=temp)

    IF(LRESI.AND.LENTR)THEN
       LRESI=.FALSE.
       LENTRES=.TRUE.
    ELSE
       LENTRES=.FALSE.
    ENDIF
    IUNIT=FIRSTU
    NCOORD=0
    IF(TIMER > 0) CALL WRTTIM('TIME INTO COVAR')
    !
    ! generate index lists
    I1=0
    I2=0
    NRES1 = 0
    RIDO='    '
    IRES2 = 0
    RIDO2='    '
    DO I=1,NATOM
       IF (SET1(I) == 1) THEN
          I1=I1+1
          IND1(I1)=I
          CALL ATOMID(I,SID,RID,REN,AC)
          IF (RID /= RIDO) THEN
             RIDO=RID
             NRES1 = NRES1 + 1
          ENDIF
       ENDIF
       IF (SET2(I) == 1) THEN
          I2=I2+1
          IND2(I2)=I
          CALL ATOMID(I,SID,RID,REN,AC)
          IF (RID /= RIDO2) THEN
             RIDO2=RID
             IRES2 = IRES2 + 1
          ENDIF
       ENDIF
    ENDDO
    !
    !
    IF(LENTR)THEN
       ! NB! Important to do first index in inner loop!
       DO I2=1,NDIM
          XSUM(I2)=ZERO
          DO I1=1,NDIM
             CO(I1,I2)=ZERO
          ENDDO
       ENDDO
    ELSEIF(LDIID)THEN
       DO I2=1,NDIM
          DO I1=1,NDIM
             CO(I1,I2)=ZERO
          ENDDO
       ENDDO
    ELSE
       if (.not. ldicc) then  !Amitava Roy 12/14/16
         DO I=1,NATOM
            XSUM(I)=ZERO
            YSUM(I)=ZERO
            ZSUM(I)=ZERO
            X2SUM(I)=ZERO
            Y2SUM(I)=ZERO
            Z2SUM(I)=ZERO
         ENDDO
       endif  !Amitava Roy 12/14/16
       !
       ! initialize covariance matrix
       DO I1=1,NSET1
          DO I2=1,NSET2
             CO(I1,I2)=ZERO
          ENDDO
       ENDDO
    ENDIF
    !

    ISTATS=1
    DO WHILE(ISTATS  >=  0)
       CALL READCV(XL,YL,ZL, &
#if KEY_CHEQ==1
            (/ ZERO /), .FALSE., &  
#endif
            TEMP,NATOM,FREEAT,NFREAT, &
            FIRSTU,NUNIT,IUNIT,NFILE,ISTEP,ISTATS, &
            NDEGF,DELTA,BEGIN,STOP,SKIP, &
            NSAV,HDR,HDR,TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       NCOORD=NCOORD+1
       !
       !
       IF(LENTR)THEN
          I2=1
          DO I=1,NSET1
             I1=IND1(I)
             XX(I2)=XL(I1)
             XX(I2+1)=YL(I1)
             XX(I2+2)=ZL(I1)
             I2=I2+3
          ENDDO
          ! NB! Important to do first index in inner loop!
          DO I2=1,NDIM
             XSUM(I2)=XSUM(I2)+XX(I2)
             DO I1=I2,NDIM
                CO(I1,I2)=CO(I1,I2) + XX(I1)*XX(I2)
             ENDDO
          ENDDO
       ELSEIF(LDIID)THEN
          I2=1
          DO I=1,NSET1
             I1=IND1(I)
             XX(I2)=XL(I1)-XCOMP(I1)
             XX(I2+1)=YL(I1)-YCOMP(I1)
             XX(I2+2)=ZL(I1)-ZCOMP(I1)
             I2=I2+3
          ENDDO
          ! NB! Important to do first index in inner loop!
          DO I2=1,NDIM
             DO I1=I2,NDIM
                CO(I1,I2)=CO(I1,I2) + XX(I1)*XX(I2)
             ENDDO
          ENDDO
       ELSEif (.not. ldicc) then  !if DICC, then we only need NCOORD --> Amitava Roy 12/14/16
          DO I=1,NATOM
             XSUM(I)=XL(I)+XSUM(I)
             YSUM(I)=YL(I)+YSUM(I)
             ZSUM(I)=ZL(I)+ZSUM(I)
             X2SUM(I)=XL(I)*XL(I)+X2SUM(I)
             Y2SUM(I)=YL(I)*YL(I)+Y2SUM(I)
             Z2SUM(I)=ZL(I)*ZL(I)+Z2SUM(I)
          ENDDO
          !
          ! now add in the contribution for this coordinate set
          ! NB! Important to do first index in inner loop!
          DO I2=1,NSET2
             DO I1=1,NSET1
                CO(I1,I2)=CO(I1,I2) &
                     +XL(IND1(I1))*XL(IND2(I2)) &
                     +YL(IND1(I1))*YL(IND2(I2)) &
                     +ZL(IND1(I1))*ZL(IND2(I2))
             ENDDO
          ENDDO
       ENDIF
    ENDDO
    !

    ! get distance covariance/correlation
    if (ldicc) then ! For DCOR --> Amitava Roy 12/09/2016
    ! DICC loops over selected atom and read the whole trajectory at a time.
    ! COVAR loops over time and reads all coordinates from snap shot at a time.
    ! If DICC has been rquested read coordinates using the subroutine rdtrj subroutine
    ! in rmsdyn.src. Otherwise continue using the subroutine READCV

      call chmalloc('corman2.src','dcor','aislc',natom,intg=aislc)
      call chmalloc('corman2.src','dcor','ajslc',natom,intg=ajslc)
      aislc=0
      do i=1,nset1
       aislc(ind1(i))=1
      enddo
      ajslc=0
      do i=1,nset2
       ajslc(ind2(i))=1
      enddo
      call chmalloc('corman2.src','dcor','xyz',3,nset1+nset2,ncoord,cr4=xyz)
      call chmalloc('corman2.src','dcor','t1',ncoord,3,crl=t1)
      call chmalloc('corman2.src','dcor','t2',ncoord,3,crl=t2)
      xyz=0
      call rdtrj(natom,aislc,ncoord,nset1,ajslc,nset2,xyz,firstu,nunit,begin,skip,stop)

      do i1=1,nset1
        t1(:,1)=xyz(1,i1,:)
        t1(:,2)=xyz(2,i1,:)
        t1(:,3)=xyz(3,i1,:)
        i2=1
#if KEY_PARALLEL==1
        do i2=1,nset2,numnod
          i=i2+mynod
#else
        do i2=1,nset2
          i=i2
#endif
#if KEY_PARALLEL==1
          if(i.le.nset2) then
#endif
            t2(:,1)=xyz(1,nset1+i,:)
            t2(:,2)=xyz(2,nset1+i,:)
            t2(:,3)=xyz(3,nset1+i,:)
            call distcor(t1,3,t2,3,ncoord,dvar1,dvar2,dcov,dcor)
            if (ldcor) then
              co(i1,i)=dcor
            else
             co(i1,i)=dcov
            endif
#if KEY_PARALLEL==1
          endif
#endif
        enddo
      enddo
#if KEY_PARALLEL==1
      call gcombr(co,nset1*nset2)  ! MH2023: extend the gcomb interface for fix
#endif

      call chmdealloc('corman2.src','dcor','aislc',natom,intg=aislc)
      call chmdealloc('corman2.src','dcor','ajslc',natom,intg=ajslc)
      call chmdealloc('corman2.src','dcor','t1',ncoord,3,crl=t1)
      call chmdealloc('corman2.src','dcor','t2',ncoord,3,crl=t2)
      call chmdealloc('corman2.src','dcor','xyz',3,nset1+nset2,ncoord,cr4=xyz)

    elseIF(LENTR)THEN  !get variances
       DO I=1,NDIM
          XSUM(I)=XSUM(I)/NCOORD
       ENDDO
       DO I2=1,NDIM
          DO I1=I2,NDIM
             CO(I1,I2)=CO(I1,I2)/NCOORD - XSUM(I1)*XSUM(I2)
             ! Could possibly remove next line, if we do make use of the symmetry below
             CO(I2,I1)=CO(I1,I2)
          ENDDO
       ENDDO
    ELSEIF(LDIID)THEN
       DO I2=1,NDIM
          DO I1=I2,NDIM
             CO(I1,I2)=CO(I1,I2)/NCOORD
             CO(I2,I1)=CO(I1,I2)
          ENDDO
       ENDDO
    ELSE
       DO I=1,NATOM
          XSUM(I)=XSUM(I)/NCOORD
          YSUM(I)=YSUM(I)/NCOORD
          ZSUM(I)=ZSUM(I)/NCOORD
          X2SUM(I)=X2SUM(I)/NCOORD
          Y2SUM(I)=Y2SUM(I)/NCOORD
          Z2SUM(I)=Z2SUM(I)/NCOORD
          WSUM(I)= -XSUM(I)*XSUM(I)+X2SUM(I) &
               -YSUM(I)*YSUM(I)+Y2SUM(I) &
               -ZSUM(I)*ZSUM(I)+Z2SUM(I)
          IF(WSUM(I) > RSMALL) THEN
             WSUM(I)=SQRT(WSUM(I))
          ELSE
             WSUM(I)=ONE
          ENDIF
       ENDDO
       !
       ! now complete the covariance matrix
       IF (NCOORD > 0) THEN
          DO I1=1,NSET1
             DO I2=1,NSET2
                CO(I1,I2)=( CO(I1,I2)/NCOORD &
                     - ( XSUM(IND1(I1))*XSUM(IND2(I2)) &
                     +YSUM(IND1(I1))*YSUM(IND2(I2)) &
                     +ZSUM(IND1(I1))*ZSUM(IND2(I2)) ) ) &
                     /( WSUM(IND1(I1))*WSUM(IND2(I2)) )
             ENDDO
          ENDDO
       ENDIF
    endif ! <-- Amitava Roy 12/08/2016

    !
    IF(TIMER > 0) CALL WRTTIM('TIME TO COLLECT COVARIANCES')
    ! now write the covariance matrix to UNIT
    ! LNI change, to make the file easier readable by other programs
    ! output just the actual numbers if MATRix is specified
    !
    !**clbiii 5/01
    ! clbiii changed this back since one doesn't always choose all residues in
    ! selection and therefore need mapping between residue number and position
    ! in map.
    IF(IOLEV <= 0) then
       call chmdealloc('corman2.src','covar2','temp',natom,cr4=temp)
       RETURN
    endif
    IF(LMAT) THEN
       IF(UNIT  >  0)THEN
          !     for now this is not good for the residue averaged stuff...
          IF(LENTR.OR.LDIID)THEN
             WRITE(MATFMT,'(A,I5,A)')'(',NDIM,'G16.8)'
             DO I1=1,NDIM
                WRITE(UNIT,MATFMT)(CO(I1,I2),I2=1,NDIM)
             ENDDO
          ELSE
             WRITE(MATFMT,'(A,I5,A)')'(',NSET2,'G16.8)'
             DO I1=1,NSET1
                WRITE(UNIT,MATFMT)(CO(I1,I2),I2=1,NSET2)
             ENDDO
          ENDIF
       ENDIF
    ELSE IF(UNIT > 0)THEN
       !
       WRITE(UNIT,'(A)') ' '
       WRITE(UNIT,'(A)') ' COVARIANCE MATRIX'
       !
       RIDO='    '
       IF(NRES2 < 1) &
            WRITE(UNIT,'(A,I10)') ' NSET1=',NSET1
       IF(NRES2 > 0) &
            WRITE(UNIT,'(A,I10)') ' NSET1=',NRES1
       DO I1=1,NSET1
          CALL ATOMID(IND1(I1),SID,RID,REN,AC)
          IF(NRES2 > 0) THEN
             IF(RID /= RIDO) THEN
                RIDO=RID
                WRITE(UNIT,'(10X,A,1X,A)') RIDO(1:idleng), &
                     REN(1:idleng)
             ENDIF
          ELSE
             WRITE(UNIT,'(1X,I6,A,4(A,1X),A)') &
                  I1,'=(',SID(1:idleng),RID(1:idleng), &
                  REN(1:idleng),AC(1:idleng),')'
          ENDIF
       ENDDO
       !
       WRITE(UNIT,'(A)') ' '
       !
       RIDO='    '
       IF(LRESI)THEN
          WRITE(UNIT,'(A,I10)') ' NSET2=',IRES2
       ELSE
          WRITE(UNIT,'(A,I10)') ' NSET2=',NSET2
       ENDIF
       DO I2=1,NSET2
          CALL ATOMID(IND2(I2),SID,RID,REN,AC)
          IF(LRESI) THEN
             IF(RID /= RIDO) THEN
                RIDO=RID
                WRITE(UNIT,'(10X,A,1X,A)') RIDO(1:idleng), &
                     REN(1:idleng)
             ENDIF
          ELSE
             WRITE(UNIT,'(1X,I6,A,4(A,1X),A)') &
                  I2,'=(',SID(1:idleng),RID(1:idleng), &
                  REN(1:idleng),AC(1:idleng),')'
          ENDIF
       ENDDO
       WRITE(UNIT,'(A)') ' '
       !
       DO I1=1,NSET1
          RIDO='    '
          IRES2 = 0
          DO I2=1,NSET2
             IF(LRESI) THEN
                CALL ATOMID(IND2(I2),SID,RID,REN,AC)
                IF(RID /= RIDO) THEN
                   IRES2 = IRES2 + 1
                   IF(I2 /= 1) THEN
                      IF(NRES2 < 1.AND..NOT.LMAT) &
                           WRITE(UNIT,'(I4,1X,I4,1X,G15.8)') &
                           I1, IRES2-1, COCUM/NUMAT
                      CORED(I1,IRES2-1) = COCUM/NUMAT
                   ENDIF
                   RIDO=RID
                   NUMAT = 1
                   COCUM = CO(I1,I2)
                ELSE
                   NUMAT = NUMAT + 1
                   COCUM=COCUM + CO(I1,I2)
                ENDIF
                IF(I2 == NSET2) THEN
                   IF(NRES2 < 1.AND..NOT.LMAT) &
                        WRITE(UNIT,'(I4,1X,I4,1X,G15.8)') &
                        I1, IRES2, COCUM/NUMAT
                   CORED(I1,IRES2) = COCUM/NUMAT
                ENDIF
             ELSE
                IF(.NOT.LMAT) &
                     WRITE(UNIT,'(I4,1X,I4,1X,G15.8)') &
                     I1, I2, CO(I1,I2)
             ENDIF
          ENDDO
          IF (NRES2 < 1) WRITE(UNIT,'(A)') ' '
       ENDDO
       !
       IF(NRES2 > 0) THEN
          DO IRES2=1,NRES2
             RIDO = '    '
             NRES1 = 0
             DO I1=1,NSET1
                CALL ATOMID(IND1(I1),SID,RID,REN,AC)
                IF(RID /= RIDO) THEN
                   NRES1 = NRES1 + 1
                   IF (I1 /= 1.AND..NOT.LMAT) &
                        WRITE(UNIT,'(I4,1X,I4,1X,G15.8)') &
                        NRES1-1, IRES2, COCUM/NUMAT
                   RIDO=RID
                   NUMAT = 1
                   COCUM = CORED(I1,IRES2)
                ELSE
                   NUMAT = NUMAT + 1
                   COCUM=COCUM + CORED(I1,IRES2)
                ENDIF
                IF (I1 == NSET1.AND..NOT.LMAT) WRITE(UNIT, &
                     '(I4,1X,I4,1X,G15.8)') &
                     NRES1, IRES2, COCUM/NUMAT
             ENDDO
          ENDDO
       ENDIF
    ENDIF
    IF (LENTR) THEN
       ! entropy estimate S' from covariance matrix. L. Nilsson September 2002
       ! Schlitter Chem Phys Lett (1993) 215, 617:
       ! S<S'=0.5*Kboltz*ln(det(1+k* T*exp(2)/hbar**2 * M*covar)
       ! Need HBAR in AKMA
       !
       COF1=KBOLTZ*TMPR/(HBAR/KCALMOL/TIMFAC*1D12)**2
       IF(LSPRIM)THEN
          !
          ! Schlitter S'
          COF1=COF1*EXP(TWO)
          COF2=ONE
       ELSE
          !
          ! Andricioaei&Karplus S''
          COF2=ONE/TWELVE
       ENDIF
       DO I1=1,NDIM
          ! NB: Check units for consistency here!!
          ! Symmetry? Put dimensionality checks in calling routine
          ! LINV3F may need unprotection of other routines in imsl.src
          ! same for some constants in consta.f90
          !
          COEFF=AMASS(IND1((I1-1)/3 + 1))*COF1
          DO I2=I1,NDIM
             CO(I1,I2)=COEFF*CO(I1,I2)
             CO(I2,I1)=CO(I1,I2)
          ENDDO
          CO(I1,I1)=CO(I1,I1)+COF2
       ENDDO
       !
       ! Test option, use only diagonal elements
       IF(LDIAG)THEN
          DO I1=1,NDIM
             DO I2=I1+1,NDIM
                CO(I1,I2)=0.0
                CO(I2,I1)=0.0
             ENDDO
          ENDDO
       ENDIF
       !
       IF(LENTRES)THEN
          ! Need to save CO for later use
          DO I1=1,NDIM
             DO I2=1,NDIM
                COSAV(I2,I1)=CO(I2,I1)
             ENDDO
          ENDDO
       ENDIF
       !
       IF(TIMER > 0) CALL WRTTIM('TIME TO ARRANGE COVAR MATRIX')
       IF(LCHOL)THEN
          !
          ! Positive semidefinite, do a cholesky decomposition
          CALL CHOLESKY(CO,NDIM,NDIM,WK,D1,D2,WK(N+1))
       ELSE
          CALL LUDET(CO,NDIM,NDIM,WK,D1,D2)
       ENDIF
       IF(TIMER > 0) CALL WRTTIM('TIME TO COMPUTE DETERMINANT')
       !
       ! NB! entropy units here are kcal/mol/K, for one mol of the selected system
       ! det = D1*2**D2: log (det)= log(D1) + D2*log(2)

       S=0.5*KBOLTZ* (LOG(D1)+D2*LOG(TWO))
       IF(.NOT. LSPRIM) S=S+KBOLTZ
       WRITE(OUTU,'(/5X,A,F6.1,A,I6,A)') &
            'Entropy estimate  at T=',TMPR,'[K] for', &
            NDIM,' degrees of freedom'
       IF(LDIAG) WRITE(OUTU,'(/5X,A)') 'Using diagonal elements only'
       IF(LSPRIM)THEN
          WRITE(OUTU,'(/5X,A)') 'Approximation of Schlitter'
       ELSE
          WRITE(OUTU,'(/5X,A)') &
               'Approximation of Andricioaei & Karplus'
       ENDIF
       WRITE(OUTU,'(/5X,A,G12.5,A,/5X,A,G12.5,A,/5X,A,G12.5,A/)') &
            '                      S=',S,' [kcal/mol/K]', &
            '                     TS=',TMPR*S,' [kcal/mol]', &
            '                S/Ndegf=',S/NDIM,' [kcal/mol/K]'
       call set_param('ENTROPY',S)
       !
       ! On a residue basis
       IF(LENTRES)THEN
          !
          ! Upper triangle of CO-matrix is preserved in Cholesky decomp,and so
          ! can be reused
          SFULL=S
          SUM=0.0
          N=0
          IFIRST=1
          WRITE(OUTU,'(/5X,A,/5X,A,5X,A,5X,A,5X,A,5X,A)') &
               'Entropy [kcal/mol/K]/residue. Other correlations neglected', &
               ' Segid','Resnam',' Resid','Entropy','Atoms/residue'
          CALL ATOMID(IND1(1),SIDO,RIDO,RENO,AC)
          !
          DO I=1,NSET1
             CALL ATOMID(IND1(I),SID,RID,REN,AC)
             IF(RID /= RIDO)THEN
                ! At a new residue; sum up results for previous residue, reset counters
                N=3*N
                DO I1=1,N
                   DO I2=I1,N
                      ! CHECK THAT WE GET RIGHT PART OF CO!!!!!!
                      CO(I1,I2)=COSAV(IFIRST+I1-1,IFIRST+I2-1)
                      CO(I2,I1)=CO(I1,I2)
                   ENDDO
                ENDDO
                IF(LCHOL)THEN
                   CALL CHOLESKY(CO,NMAX,N,WK,D1,D2,WK(N+1))
                ELSE
                   CALL LUDET(CO,NMAX,N,WK,D1,D2)
                ENDIF
                S=0.5*KBOLTZ* (LOG(D1)+D2*LOG(TWO))
                IF(.NOT. LSPRIM) S=S+KBOLTZ
                SUM=SUM+S
                WRITE(OUTU,'(5X,3(2X,A,5X),G12.4,I10)') &
                     SIDO(1:idleng),RENO(1:idleng),RIDO(1:idleng), &
                     S,N/3
                IFIRST=3*I-2
                N=1
                SIDO=SID
                RIDO=RID
                RENO=REN
             ELSE
                N=N+1
             ENDIF
          ENDDO
          !
          ! And the last residue
          N=3*N
          DO I1=1,N
             DO I2=I1,N
                ! CHECK THAT WE GET RIGHT PART OF CO!!!!!!
                CO(I1,I2)=COSAV(IFIRST+I1-1,IFIRST+I2-1)
                CO(I2,I1)=CO(I1,I2)
             ENDDO
          ENDDO
          IF(LCHOL)THEN
             CALL CHOLESKY(CO,NMAX,N,WK,D1,D2,WK(N+1))
          ELSE
             CALL LUDET(CO,NMAX,N,WK,D1,D2)
          ENDIF
          S=0.5*KBOLTZ* (LOG(D1)+D2*LOG(TWO))
          IF(.NOT. LSPRIM) S=S+KBOLTZ
          SUM=SUM+S
          WRITE(OUTU,'(5X,3(2X,A,5X),G12.4,I10)') &
               SIDO(1:idleng),RENO(1:idleng),RIDO(1:idleng),S,N/3
          WRITE(OUTU,'(//5X,A,G12.4/5X,A,G12.4/)') &
               'Sum of residue entropies (Sres)=',SUM, &
               '          Difference Sfull-Sres=',SFULL-SUM
          IF(TIMER > 0) &
               CALL WRTTIM('TIME TO COMPUTE SUB-DETERMINANTS')
       ENDIF
    ENDIF
    call chmdealloc('corman2.src','covar2','temp',natom,cr4=temp)
    RETURN
  END SUBROUTINE COVAR2

  SUBROUTINE DISTMAT()
    !-----------------------------------------------------------------------
    !  Computes distance matrix from a dynamics trajectory for
    !  selected pairs of atoms. The command structure is like
    !  that of most other coordinate manipulation commands other keywords are:
    !
    !      UNIT     the distance matrix will be written to the unit
    !               number specified as an ASCII file unless the TRAJ
    !               keyword is specified, in which case a binary "trajectory" of
    !               the distance matrix will be written.
    !       RESIdue this keyword specifies to compute the distance matrix
    !               for a center of geometry weighted average of residues
    !       NOE     this keyword denotes that the averaging over distances
    !               in the distance matrix should be inverse sixth power
    !               weighted.
    !       TRAJ    write a dynamic trajectory file of the distance matrix
    !       SINGle  process only a single coordinate file
    !       CUTOff  print only those values of the distance matrix which are
    !               smaller than cutoff value
    !       PROJect project out a subset of contacts for printing
    !       UPRJ    read projection matrix from unit UPRJ
    !       PROB    compute the contact probability based on differences
    !               from reference contact map read from UPRB and with
    !               an upperbound tolerance of TOLE
    !       RMSF    Computes the root mean square fluctuation in the distance
    !               matrix from the trajectory. Disables the printing of
    !               the binary file.
    !
    ! SYNTAX:
    !   CHARMM> COORdinates DMAT -
    !           RESIdue_average NOE_weighting -
    !           SINGle -
    !           FIRSt_unit <int> NUNIt <int> BEGIn <int> SKIP <int> -
    !           STOP <int> 2x<atom selection (SET1, SET2)> -
    !           UNIT_for_output <int>  TRAJectory CUTOff <real> -
    !           PROJect UPRJ <int> PROBability UPRB <int> TOLE <real> [RMSF]
    !
    !**********************************************************************C
    !                                                                      C
    !                   Written by C. L. Brooks III                        C
    !                   Winter, 1996                                       C
    !                   Additions by A. Lahiri                             C
    !                   Winter, 1998                                       C
    !                                                                      C
    !**********************************************************************C
    !
  use dimens_fcm
  use number
  use comand
  use select
  use stream
  use string
  use psf
  use coord
  use coordc
  use memory
  use cvio,only:trjspc
  use chutil,only:atomid

    ! local
    INTEGER NSET1, NSET2, UNIT, NRES1, NRES2
    INTEGER RES1, RES2, DUNIT
    INTEGER BEGIN, SKIP, STOP, NUNIT, FIRSTU
    INTEGER UPROJ, UPROB, PROJ, DIFF
    integer,allocatable,target,dimension(:) :: space_sets
    INTEGER,pointer,dimension(:) :: SET1,SET2
    LOGICAL ERROR, LRESI, LNOE, LTRAJ, LSING, LPROB, LPROJ
    LOGICAL LRMSF, LMAT, LREL, LMKPRJ
    character(len=8) RIDO
    character(len=8) SID,RID,REN,AC
    INTEGER I
    real(chm_real) CTOFF, TOLER
    !
    call chmalloc('corman2.src','distmat','space_sets',2*natom,intg=space_sets)
    set1 => space_sets(1:natom)
    set2 => space_sets(natom+1:2*natom)
    LSING=.FALSE.
    LSING=(INDXA(COMLYN,COMLEN,'SING') > 0)
    IF(.NOT.LSING) THEN
       ERROR=.FALSE.
       CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,BEGIN,SKIP,STOP)
    ENDIF
    UNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
    LTRAJ=.FALSE.
    LTRAJ=(INDXA(COMLYN,COMLEN,'TRAJ') > 0)
    LRMSF = (INDXA(COMLYN,COMLEN,'RMSF') > 0)
    LMAT = (INDXA(COMLYN,COMLEN,'MATR') > 0)
    LREL = (INDXA(COMLYN,COMLEN,'RELA') > 0)
    LMKPRJ = (INDXA(COMLYN,COMLEN,'MKPR') > 0)
    IF (LRMSF)THEN
       LTRAJ = .FALSE.
       ! Now check if there are separate specifications for Distance and RMSF output
       DUNIT=GTRMI(COMLYN,COMLEN,'DUNI',-1)
       UNIT=GTRMI(COMLYN,COMLEN,'FUNI',UNIT)
    ELSE
       DUNIT=UNIT
    ENDIF
    IF(PRNLEV >= 3.AND.DUNIT > 0) WRITE(OUTU,'(A,I4)') &
         ' DMAT: Distance matrix output to unit',DUNIT
    IF(LTRAJ) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,I4)') &
            ' DMAT: binary trajectory output to unit',UNIT
    ENDIF
    IF(LRMSF.AND. PRNLEV >= 3) THEN
       IF(.NOT. LREL) THEN
          WRITE(OUTU,'(A,I4)') &
               ' DMAT: RMS distance fluctuations output to unit',UNIT
       ELSE
          WRITE(OUTU,'(A,I4)') &
               ' DMAT: Relative RMS distance fluctuations output to unit',UNIT

       ENDIF
    ENDIF
    IF(LMKPRJ .AND. PRNLEV >= 3) THEN
       WRITE(OUTU,'(A,I5)') &
            ' DMAT: Writing projection matrix to unit', UNIT
    ENDIF
    !
    LNOE=.FALSE.
    LNOE=(INDXA(COMLYN,COMLEN,'NOE') > 0)
    IF(LNOE) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A)') &
            ' DMAT: Inverse r^6 (NOE-like) weighting for matrix averaging'
    ENDIF
    !
    CALL SELCTD(COMLYN,COMLEN,SET1,SET2,X,Y,Z,WMAIN,.TRUE.,ERROR)
    NSET1=NSELCT(NATOM,SET1)
    NSET2=NSELCT(NATOM,SET2)
    IF(PRNLEV >= 3) WRITE(OUTU,'(A,I4)') &
         ' DMAT: Number of atoms selected for SET1',NSET1
    IF(PRNLEV >= 3) WRITE(OUTU,'(A,I4)') &
         ' DMAT: Number of atoms selected for SET2',NSET2
    !
    LRESI=.FALSE.
    LRESI=(INDXA(COMLYN,COMLEN,'RESI') > 0)
    IF(LRESI) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A)') &
            ' DMAT: Distance matrix will be for residue center of geometry'
    ENDIF
    CTOFF = GTRMF(COMLYN,COMLEN,'CUTO',ANUM)
    IF (LRESI) THEN
       NRES2 = 0
       RIDO='    '
       DO I=1,NATOM
          IF(SET2(I) == 1) THEN
             CALL ATOMID(I,SID,RID,REN,AC)
             IF (RID /= RIDO) THEN
                RIDO=RID
                NRES2 = NRES2 + 1
             ENDIF
          ENDIF
       ENDDO
       !
       NRES1 = 0
       RIDO='    '
       DO I=1,NATOM
          IF(SET1(I) == 1) THEN
             CALL ATOMID(I,SID,RID,REN,AC)
             IF (RID /= RIDO) THEN
                RIDO=RID
                NRES1 = NRES1 + 1
             ENDIF
          ENDIF
       ENDDO
       !
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,2X,I5)') &
            ' DMAT: Number of residues in SET1 is', NRES1
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,2X,I5)') &
            ' DMAT: Number of residues in SET2 is', NRES2
    ENDIF
    LPROJ=.FALSE.
    LPROJ=(INDXA(COMLYN,COMLEN,'PROJ') > 0)
    UPROJ = GTRMI(COMLYN,COMLEN,'UPRJ',6)
    IF(LPROJ) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A)') &
            ' DMAT: DISTANCE MATRIX PROJECTION WILL BE COMPUTED'
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,I5)') &
            ' DMAT: DISTANCE MATRIX FOR PROJECTION WILL BE READ FROM UNIT ' &
            , UPROJ
    ENDIF
    LPROB=.FALSE.
    LPROB=(INDXA(COMLYN,COMLEN,'PROB') > 0)
    UPROB = GTRMI(COMLYN,COMLEN,'UPRB',ISTRM)
    TOLER = GTRMF(COMLYN,COMLEN,'TOLE',ANUM)
    IF(LPROB) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A)') &
            ' DMAT: DISTANCE MATRIX PROBABILITY WILL BE COMPUTED'
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,I5)') &
            ' DMAT: DISTANCE MATRIX FOR COMPARISON WILL BE READ FROM UNIT ' &
            , UPROB
    ENDIF
    !
    !
    IF (.NOT.ERROR.AND.NSET1 > 0.AND.NSET2 > 0) THEN
       !
       ! dynamic
       !
       CALL DISTMAT2(UNIT,NATOM, &
            BEGIN,SKIP,STOP,NUNIT, &
            FIRSTU,NSET1,SET1,NSET2,SET2, &
            NRES1, NRES2, &
            LRESI,LTRAJ,LRMSF,LNOE,LSING,X,Y,Z,CTOFF, &
            LPROJ,UPROJ,LPROB,UPROB, &
            TOLER,LMAT,LREL,DUNIT,LMKPRJ)
    ENDIF
    call chmdealloc('corman2.src','distmat','space_sets',2*natom,intg=space_sets)
    RETURN
  END SUBROUTINE DISTMAT

  SUBROUTINE DISTMAT2(UNIT,NATOM, &
       BEGIN,SKIP,STOP,NUNIT,FIRSTU, &
       NSET1,SET1,NSET2,SET2, &
       NRES1,NRES2, &
       LRESI,LTRAJ,LRMSF,LNOE,LSING,X,Y,Z,CTOFF, &
       LPROJ,UPROJ,LPROB,UPROB, &
       TOLER,LMAT,LREL,DUNIT,LMKPRJ)
    !
    !
  use cheq,only: qcg
  use number
  use ctitla
  use stream
  use cvio
#if KEY_CHEQ==1
  use dimens_fcm   
#endif
  use memory    
  use chutil
  use param_store, only: set_param

    INTEGER UNIT
    INTEGER NATOM
    real(chm_real),allocatable,dimension(:),target :: space_rl
    real(chm_real),pointer,dimension(:) :: XCEN=>null(),YCEN=>null(),ZCEN=>null(), &
         WCEN=>null(),XCEN2=>null(),YCEN2=>null(),ZCEN2=>null(), WCEN2=>null(), &
         xl=>null(),yl=>null(),zl=>null()
    REAL(chm_real4),allocatable,dimension(:) :: TEMP

    integer,allocatable,dimension(:),target :: space_int
    INTEGER,pointer,dimension(:) :: FREEAT,ind1,ind2
    INTEGER BEGIN, SKIP, STOP, NUNIT, FIRSTU
    INTEGER UPROJ, UPROB, DUNIT
    INTEGER NSET1, NSET2, NRES1, NRES2
    INTEGER SET1(NSET1),SET2(NSET2)
    real(chm_real) X(*), Y(*), Z(*)
    real(chm_real) :: ctoff,toler
    real(chm_real),allocatable,dimension(:,:) :: CO, PROJ, FL, DIFF
    LOGICAL LRESI, LNOE, LTRAJ, LRMSF, LSING, LPROJ, LPROB, LMAT,LREL
    LOGICAL LMKPRJ
    ! local
    INTEGER NCOORD, I, I1, I2, ISTATS, NFREAT
    INTEGER ISTEP, IUNIT, NDEGF, NFILE, NSAV
    INTEGER IRES2, IRES1
    INTEGER NCONTACT
    character(len=4) TITL
    real(chm_real) COCUM,AV, CTOF1
    real(chm_real) DELTA
    character(len=4),parameter :: HDR='CORD'
    character(len=8) RIDO,RIDO2,SID,RID,REN,AC
    character(len=4),parameter :: HDRD='DIST'
    INTEGER ICNTRL(20)
    LOGICAL ERROR, DONE
    character(len=20) MATFMT
    integer :: siz_rl
    integer :: lineno
    character(len=*), parameter :: BADRES = 'Bad residue number'
    character(len=*), parameter :: PROCNAME = 'DISTMAT2'
    ! begin
    !
   
    call chmalloc('corman2.src','DISTMAT2','space_int',NATOM+nset1+nset2,intg=space_int)
    freeat => space_int(1:natom)
    ind1   => space_int(natom+1:natom+nset1)
    ind2   => space_int(natom+nset1+1:natom+nset1+nset2)
    call chmalloc('corman2.src','DISTMAT2','TEMP',NATOM,cr4=TEMP)
    IF(.NOT. LRESI) THEN
       NRES1 = NSET1
       NRES2 = NSET2
    ENDIF
    siz_rl = 3*natom + 4*(nset1+nset2)
    call chmalloc('corman2.src','DISTMAT2','space_rl',siz_rl,crl=space_rl)
    xl => space_rl(1:natom)
    yl => space_rl(natom+1:2*natom)
    zl => space_rl(2*natom+1:3*natom)
    xcen => space_rl(3*natom+1:3*natom+nset1)
    ycen => space_rl(3*natom+nset1+1:3*natom+2*nset1)
    zcen => space_rl(3*natom+2*nset1+1:3*natom+3*nset1)
    wcen => space_rl(3*natom+3*nset1+1:3*natom+4*nset1)
    xcen2 => space_rl(3*natom+4*nset1        +1:3*natom+4*nset1+  nset2)
    ycen2 => space_rl(3*natom+4*nset1+  nset2+1:3*natom+4*nset1+2*nset2)
    zcen2 => space_rl(3*natom+4*nset1+2*nset2+1:3*natom+4*nset1+3*nset2)
    wcen2 => space_rl(3*natom+4*nset1+3*nset2+1:3*natom+4*nset1+4*nset2)
    call chmalloc('corman2.src','DISTMAT','CO',NRES1,NRES2,crl=CO)
    call chmalloc('corman2.src','DISTMAT','FL',NRES1,NRES2,crl=FL)
    if(lproj) call chmalloc('corman2.src','DISTMAT','PROJ',NRES1,NRES2,crl=PROJ)
    if(lprob) call chmalloc('corman2.src','DISTMAT','DIFF',NRES1,NRES2,crl=DIFF)
    
    IUNIT=FIRSTU
    NCOORD=0
    !
    ! generate index lists
    DO I = 1, NRES1
       WCEN(I) = ZERO
    ENDDO
    DO I = 1, NRES2
       WCEN2(I) = ZERO
    ENDDO
    I1=0
    I2=0
    IRES1 = 0
    RIDO='    '
    IRES2 = 0
    RIDO2='    '
    DO I=1,NATOM
       IF (SET1(I) == 1) THEN
          I1=I1+1
          IND1(I1)=I
          CALL ATOMID(I,SID,RID,REN,AC)
          IF (RID /= RIDO) THEN
             RIDO=RID
             IRES1 = IRES1 + 1
          ENDIF
          WCEN(IRES1) = WCEN(IRES1) + ONE
       ENDIF
       IF (SET2(I) == 1) THEN
          I2=I2+1
          IND2(I2)=I
          CALL ATOMID(I,SID,RID,REN,AC)
          IF (RID /= RIDO2) THEN
             RIDO2=RID
             IRES2 = IRES2 + 1
          ENDIF
          WCEN2(IRES2) = WCEN2(IRES2) + ONE
       ENDIF
    ENDDO
    !
    !
    ! initialize covariance matrix
    DO I1=1,NRES1
       DO I2=1,NRES2
          IF(LRMSF) FL(I1,I2)=ZERO
          CO(I1,I2)=ZERO
          IF(LPROJ) PROJ(I1,I2)=ZERO
          IF(LPROB) DIFF(I1,I2)=ZERO
       ENDDO
    ENDDO
    !
    ! if a projection matrix is to be output the cutoff for contact
    ! counting has to be > 1
    CTOF1=CTOFF
    IF(LMKPRJ) CTOF1=2.0
    !
    !  Check and see whether reference matrices need to be read in
    !
    IF(LPROJ) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,I5)') &
            ' PROJECTION MATRIX READ FROM UNIT ',UPROJ
       DONE = .FALSE.
       lineno = 0
       DO WHILE(.NOT.DONE)
          READ(UPROJ,'(A)') TITL
          lineno = lineno + 1
          IF(TITL == '*   ') DONE = .TRUE.
       ENDDO
1000   READ(UPROJ,*,END=1010) I1, I2, COCUM
       lineno = lineno + 1
       if (I1 < 1 .or. I1 > NRES1) call value_error(PROCNAME, BADRES, &
             I1, UPROJ, lineno)
       if (I2 < 1 .or. I2 > NRES2) call value_error(PROCNAME, BADRES, &
             I2, UPROJ, lineno)
       PROJ(I1,I2) = COCUM
       GOTO 1000
    endif
1010 continue
    !
    !
    IF(LPROB) THEN
       IF(PRNLEV >= 3) WRITE(OUTU,'(A,I5)') &
            ' REFERENCE MATRIX READ FROM UNIT ',UPROB
       DONE = .FALSE.
       lineno = 0
       DO WHILE(.NOT.DONE)
          READ(UPROB,'(A)') TITL
          lineno = lineno + 1
          IF(TITL == '*   ') DONE = .TRUE.
       ENDDO
2000   READ(UPROB,*,END=2010) I1, I2, COCUM
       lineno = lineno + 1
       if (I1 < 1 .or. I1 > NRES1) call value_error(PROCNAME, BADRES, &
             I1, UPROB, lineno)
       if (I2 < 1 .or. I2 > NRES2) call value_error(PROCNAME, BADRES, &
             I2, UPROB, lineno)
       DIFF(I1,I2) = COCUM
       GOTO 2000
    ENDIF
2010 continue
    !
    ISTATS=1
#if KEY_CHEQ==1
    if(QCG)write(outu,'(a/a)') &
         '<DISTMAT2> reading trajectory with CHEQ on.', &
         '      **** CHARGES WILL BE DISCARDED ****  '
#endif 
    DO WHILE(ISTATS  >=  0)
       IF(.NOT.LSING) THEN
          CALL READCV(XL,YL,ZL, &
#if KEY_CHEQ==1
               (/ ZERO /), .FALSE., &  
#endif
               TEMP,NATOM,FREEAT,NFREAT, &
               FIRSTU,NUNIT,IUNIT,NFILE,ISTEP,ISTATS, &
               NDEGF,DELTA,BEGIN,STOP,SKIP, &
               NSAV,HDR,HDR,TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       ELSE
          DO I=1, NATOM
             XL(I) = X(I)
             YL(I) = Y(I)
             ZL(I) = Z(I)
          ENDDO
          ISTATS=-1
       ENDIF
       NCOORD=NCOORD+1
       !
       !
       IF(LRESI) THEN
          I1 = 0
          DO I = 1, NRES1
             XCEN(I) = ZERO
             YCEN(I) = ZERO
             ZCEN(I) = ZERO
             IRES1 = INT(WCEN(I))
             DO I2 = 1, IRES1
                I1 = I1 + 1
                XCEN(I) = XCEN(I) + XL(IND1(I1))
                YCEN(I) = YCEN(I) + YL(IND1(I1))
                ZCEN(I) = ZCEN(I) + ZL(IND1(I1))
             ENDDO
             XCEN(I) = XCEN(I) / WCEN(I)
             YCEN(I) = YCEN(I) / WCEN(I)
             ZCEN(I) = ZCEN(I) / WCEN(I)
          ENDDO
          !
          I1 = 0
          DO I = 1, NRES2
             XCEN2(I) = ZERO
             YCEN2(I) = ZERO
             ZCEN2(I) = ZERO
             IRES2 = INT(WCEN2(I))
             DO I2 = 1, IRES2
                I1 = I1 + 1
                XCEN2(I) = XCEN2(I) + XL(IND2(I1))
                YCEN2(I) = YCEN2(I) + YL(IND2(I1))
                ZCEN2(I) = ZCEN2(I) + ZL(IND2(I1))
             ENDDO
             XCEN2(I) = XCEN2(I) / WCEN2(I)
             YCEN2(I) = YCEN2(I) / WCEN2(I)
             ZCEN2(I) = ZCEN2(I) / WCEN2(I)
          ENDDO
          !  testing purposes
          !
          !            write(6,'(a)')
          !            write(6,'(a)')'************************************************'
          !            write(6,'(a)')' Coordinates for resi average, set 1: x, y, z, w'
          !            do i=1, nres1
          !               write(6,'(3(1x,f8.3))') xcen(i), ycen(i), zcen(i), wcen(i)
          !            enddo
          !
          !            write(6,'(a)')
          !            write(6,'(a)')'************************************************'
          !            write(6,'(a)')' Coordinates for resi average, set 2: x, y, z, w'
          !            do i=1, nres2
          !             write(6,'(3(1x,f8.3))') xcen2(i), ycen2(i), zcen2(i), wcen2(i)
          !            enddo
          !            write(6,'(a)')

       ELSE
          DO I1 = 1, NRES1
             XCEN(I1) = XL(IND1(I1))
             YCEN(I1) = YL(IND1(I1))
             ZCEN(I1) = ZL(IND1(I1))
          ENDDO
          DO I2 = 1, NRES2
             XCEN2(I2) = XL(IND2(I2))
             YCEN2(I2) = YL(IND2(I2))
             ZCEN2(I2) = ZL(IND2(I2))
          ENDDO
       ENDIF
       !
       !     WRITE BINARY FILE TO IUNIT
       !
       IF(LTRAJ) THEN
          IF(IOLEV > 0) THEN
             DO I1=1,NRES1
                DO I2=1,NRES2
                   CO(I1,I2)= (XCEN(I1)-XCEN2(I2))**2 &
                        +(YCEN(I1)-YCEN2(I2))**2 &
                        +(ZCEN(I1)-ZCEN2(I2))**2
                   IF(CO(I1,I2) > ZERO) THEN
                      CO(I1,I2)=SQRT(CO(I1,I2))
                      IF(LNOE) CO(I1,I2) = ONE / CO(I1,I2)**6
                   ENDIF
                ENDDO
             ENDDO
             IF(NCOORD == 1) THEN
                DO I=1,20
                   ICNTRL(I)=0
                ENDDO
                ICNTRL(1) = (STOP - BEGIN)/SKIP + 1
                ICNTRL(2) = BEGIN
                ICNTRL(3) = SKIP
                ICNTRL(4) = STOP - BEGIN + 1
                ICNTRL(5) = NSAV
                ICNTRL(8) = NDEGF
                ICNTRL(9) = NATOM - NFREAT
                ! store bytes of skip * delta in icntrl(10)
                icntrl(10) = transfer(skip * delta, icntrl(10))
                IF(LNOE) THEN
                   ICNTRL(11) = 1
                ELSE
                   ICNTRL(11) = 0
                ENDIF
                IF(LRESI) THEN
                   ICNTRL(12) = 1
                ELSE
                   ICNTRL(12) = 0
                ENDIF
                WRITE(UNIT) HDRD,ICNTRL
                CALL WRTITL(TITLEA,NTITLA,UNIT,-1)
                WRITE(UNIT) NSET1,NSET2
                WRITE(UNIT) (IND1(I1),I1=1,NSET1)
                WRITE(UNIT) (IND2(I2),I2=1,NSET2)
             ENDIF
             !
             WRITE(UNIT) ((CO(I1,I2),I1=1,NRES1),I2=1,NRES2)
          ENDIF
       ELSE
          !
          !
          ! add in the contribution for this coordinate set
          ! NB Important to do first index in inner loop
          DO I2=1,NRES2
             DO I1=1,NRES1
                COCUM = (XCEN(I1)-XCEN2(I2))**2 &
                     +(YCEN(I1)-YCEN2(I2))**2 &
                     +(ZCEN(I1)-ZCEN2(I2))**2
                IF(COCUM > ZERO) THEN
                   COCUM = SQRT(COCUM)
                   IF(LNOE) COCUM = ONE / COCUM**6
                ENDIF
                IF(LPROB) THEN
                   IF( ( ( COCUM - DIFF(I1,I2) ) <= TOLER ) &
                        .AND. ( DIFF(I1,I2)  >  ZERO ) ) THEN
                      !                      write(6,*)' Prob: ',i1,i2,cocum, diff(i1,i2)
                      COCUM = ONE
                   ELSE
                      COCUM = ZERO
                   ENDIF
                ENDIF
                CO(I1,I2) = CO(I1,I2) + COCUM
                IF(LRMSF) FL(I1,I2) = FL(I1,I2) + COCUM*COCUM
             ENDDO
          ENDDO
       ENDIF
    ENDDO
    !
    !
    IF(.NOT.LTRAJ) THEN
       IF (NCOORD > 0) THEN
          DO I2=1,NRES2
             DO I1=1,NRES1
                CO(I1,I2) = CO(I1,I2)/NCOORD
                AV = CO(I1,I2)
                IF(LRMSF) FL(I1,I2) = (FL(I1,I2)/NCOORD)-AV*AV
                IF(LNOE.AND.CO(I1,I2) > ZERO) THEN
                   CO(I1,I2) = ONE/CO(I1,I2)**(ONE/SIX)
                ENDIF
             ENDDO
          ENDDO
          !
          ! NB: with MATRIX output option there is no mapping between residue number
          ! and position in map (one doesn't always choose all residues in
          ! selection and therefore may need this)
          IF(IOLEV > 0) THEN
             IF(.NOT. LMAT)THEN
                WRITE(UNIT,'(A)') '********************************* '
                IF(LNOE) THEN
                   WRITE(UNIT,'(A)') '* DISTANCE MATRIX (NOE WEIGHTING)'
                ELSE
                   WRITE(UNIT,'(A)') '* DISTANCE MATRIX'
                ENDIF
                IF(CTOFF < ANUM) THEN
                   WRITE(UNIT,'(A,F6.2,A)') &
                        '* FILTERED USING CUTOFF OF ', CTOFF, ' A'
                ENDIF
                !
                RIDO='    '
                WRITE(UNIT,'(A,I10)') '* NSET1=',NRES1
                DO I1=1,NSET1
                   CALL ATOMID(IND1(I1),SID,RID,REN,AC)
                   IF(LRESI) THEN
                      IF (RID /= RIDO) THEN
                         RIDO=RID
                         WRITE(UNIT,'(A,10X,A,1X,A)') &
                              '***',RIDO(1:idleng),REN(1:idleng)
                      ENDIF
                   ELSE
                      WRITE(UNIT,'(A,1X,I6,A,4(A,1X),A)') &
                           '***', I1,'=(',SID(1:idleng), &
                           RID(1:idleng),REN(1:idleng), &
                           AC(1:idleng),')'
                   ENDIF
                ENDDO
                !
                WRITE(UNIT,'(A)') '*********************************'
                !
                RIDO='    '
                WRITE(UNIT,'(A,I10)') '* NSET2=',NRES2
                DO I2=1,NSET2
                   CALL ATOMID(IND2(I2),SID,RID,REN,AC)
                   IF(LRESI) THEN
                      IF(RID /= RIDO) THEN
                         RIDO=RID
                         WRITE(UNIT,'(A,10X,A,1X,A)') &
                              '***', RIDO(1:idleng),REN(1:idleng)
                      ENDIF
                   ELSE
                      WRITE(UNIT,'(A,1X,I6,A,4(A,1X),A)') &
                           '***', I2,'=(',SID(1:idleng), &
                           RID(1:idleng),REN(1:idleng), &
                           AC(1:idleng),')'
                   ENDIF
                ENDDO
                WRITE(UNIT,'(A)') '********************************* '
                WRITE(UNIT,'(A)') '*   '
                ! end of non-matrix format output IF(.NOT. LMAT)
             ENDIF
             !
             NCONTACT = 0
             IF(LMAT) WRITE(MATFMT,'(A,I5,A)')'(',NRES2,'G16.8)'
             ! if CO is large the index order should be reversed
             IF(LMAT.AND.LRMSF.AND.DUNIT > 0)THEN
                ! have to write out the distances first; this could be cleaned up...
                DO I1=1,NRES1
                   WRITE(DUNIT,MATFMT)(CO(I1,I2),I2=1,NRES2)
                ENDDO
             ENDIF
             DO I1=1,NRES1
                DO I2=1,NRES2
                   COCUM = ZERO
                   IF(LPROJ) CO(I1,I2) = PROJ(I1,I2)*CO(I1,I2)
                   IF(LMKPRJ)THEN
                      IF(CO(I1,I2) <=  CTOFF &
                           .AND. CO(I1,I2) > ZERO) THEN
                         CO(I1,I2)=ONE
                      ELSE
                         CO(I1,I2)=ZERO
                      ENDIF
                   ENDIF
                   IF(LRMSF)THEN
                      IF(DUNIT >  0 .AND. .NOT. LMAT) &
                           WRITE(DUNIT,'(I4,1X,I4,1X,G15.8)') &
                           I1,I2,CO(I1,I2)
                      IF(LREL.AND.CO(I1,I2) > ZERO)THEN
                         CO(I1,I2) = SQRT(FL(I1,I2))/CO(I1,I2)
                      ELSE
                         CO(I1,I2) = SQRT(FL(I1,I2))
                      ENDIF
                   ENDIF
                   IF((CO(I1,I2) > ZERO) &
                        .AND.(CO(I1,I2) <= CTOF1)) THEN
                      NCONTACT = NCONTACT + 1
                      COCUM = CO(I1,I2)
                   ENDIF
                   IF(.NOT.LMAT)THEN
                      ! When writing projection matrix, only output non-zero elements
                      IF(.NOT.(LMKPRJ.AND.COCUM <= ZERO))THEN
                         WRITE(UNIT,'(I4,1X,I4,1X,G15.8)') &
                              I1, I2, COCUM
                      ENDIF
                   ENDIF
                ENDDO
                IF(LMAT) WRITE(UNIT,MATFMT) (CO(I1,I2),I2=1,NRES2)
             ENDDO
             IF(CTOFF < ANUM) THEN
                IF(PRNLEV >= 3) WRITE(OUTU,'(A,I6)') &
                     ' DISTMAT2: TOTAL NUMBER OF CONTACTS ', NCONTACT
                CALL set_param( 'NCONTACT', NCONTACT )
             ENDIF
          ENDIF
       ENDIF
    ELSE
       IF(PRNLEV >= 3) WRITE(OUTU,101) &
            ICNTRL(1),(ICNTRL(I),I=2,3),UNIT
101    FORMAT(/2X,I5,'   DYNAMICS DISTANCE MATRIX STARTING FROM',/, &
            5X,'STEP NO ',I8,'   FOR EVERY ',I5,'  STEPS',/, &
            5X,'WRITTEN ON UNIT',I5,/)
       CALL VCLOSE (UNIT,'KEEP',ERROR)
    ENDIF
    nullify(XCEN,YCEN,ZCEN, WCEN,XCEN2,YCEN2,ZCEN2, WCEN2,xl,yl,zl)
    return
  END SUBROUTINE DISTMAT2

  SUBROUTINE CHOLESKY(A,NDECL,N,DIAG,D1,D2,DU)
    !
    ! Cholesky decomposition of symmetric positive definite matrix A
    ! Using NxN upper corner of matrix declared as A(NDECL,NDECL).
    ! Only upper triangle is used as input. Lower triangle returns the
    ! lower triangle of symmetric Cholesky matrix L (diagonal elements in
    ! DIAG):
    !                T
    !         A = L L
    !
    !  D=det(A) is returned in component form: D=D1 * 2**D2
    !
    ! L. Nilsson, September 2002
    !
  use number
  use stream

    INTEGER NDECL, N
    real(chm_real) A(NDECL,NDECL),DIAG(N),D1,D2,DU(N)
    !
    INTEGER I,J,K,NNEG
    real(chm_real) S
    real(chm_real),PARAMETER :: SIXTEEN=16_chm_real, SIXTNTH=ONE/SIXTEEN
    !
    ! Variant due to Rolf Karlsson,
    ! should work also for non-positive definite matrices
    ! but no eigenvalue can be zero...
    !
    IF(.FALSE.)THEN
       NNEG=0
       DO I=1,N
          DO K=1,I-1
             DU(K)=DIAG(K) * A(K,I)
          ENDDO
          DO J=I,N
             S=A(J,I)
             DO K=I-1,1,-1
                S=S-DU(K)*A(K,J)
             ENDDO
             IF(I == J)THEN
                DIAG(I)=S
                IF(S <= 0.0)THEN
                   NNEG=NNEG+1
                   write(outu,*)'tchol, i,s',i,s
                ENDIF
             ELSE
                A(J,I)=S/DIAG(I)
             ENDIF
          ENDDO
       ENDDO
       IF(NNEG > 0 .AND. PRNLEV  >=  2) &
            WRITE(OUTU,'(/5X,A,I10,A/)') &
            'TCHOL>',NNEG,' NON-POSITIVE DIAGONAL VALUES FOUND'
    ENDIF
    !
    !      IF(.FALSE.)THEN
    DO I=1,N
       DO J=I,N
          S=A(I,J)
          DO K=I-1,1,-1
             S=S-A(I,K)*A(J,K)
          ENDDO
          IF(I == J)THEN
             IF(S <= ZERO)THEN
                CALL WRNDIE(-2,'<CHOLESKY>','MATRIX NOT POSITIVE DEFINITE')
             ENDIF
             DIAG(I)=SQRT(S)
          ELSE
             A(J,I)=S/DIAG(I)
          ENDIF
       ENDDO
    ENDDO
    !
    !      ENDIF
    !
    ! Determinant is product of (diagonal elements)**2
    ! (unlike in a std LU decomposition L has non-zero diagonal elements,
    !  same as in L-transpose)
    !
    D1=ONE
    D2=ZERO
    DO I=1,N
       D1=D1*DIAG(I)*DIAG(I)
5      IF(ABS(D1)  <=  ONE) GOTO 10
       D1=D1*SIXTNTH
       D2=D2+FOUR
       GOTO 5
10     IF(ABS(D1)  >=  SIXTNTH) GOTO 15
       D1=D1*SIXTEEN
       D2=D2-FOUR
       GOTO 10
15     CONTINUE
    ENDDO
    RETURN
  END SUBROUTINE CHOLESKY
  !
  SUBROUTINE LUDET(CO,NDECL,NDIM,WK,D1,D2)
    !
    ! Compute determinant of CO using LU-decomposition
    ! CO should be positive definite, det(CO)> 0, so we skip all negative diagonal
    ! elements of the U matrix when calculating the determinant,
    ! which is returned as det(CO)=D1*2**D2
    ! NDIM   rank of CO
    ! NDECL  row dimension of CO exactly as declared in calling routine
    ! WK work array of dimnsion NDIM
    ! L. Nilsson, September 2002
    !
  use number
  use stream

    INTEGER NDIM,NDECL
    real(chm_real) CO(NDECL,NDECL),D1,D2,WK(NDIM)
    !
    INTEGER IDUM,IER,NIGNORE,I
    !      integer kk
    real(chm_real),PARAMETER :: SIXTEEN=FOUR*FOUR, SIXTNTH=ONE/SIXTEEN
    !
    ! Just get determinant from IMSL (arg3=4), slower than cholesky, but may allow
    ! recovery from negative diagonal elements (ie, det(CO)<0)
    CALL LINV3F(CO,0,4,NDIM,NDECL,D1,D2,WK,IER)
    IF(IER > 0)THEN
       IF(PRNLEV  > 3) WRITE(OUTU,'(A,I5)') ' IER=', IER
       CALL WRNDIE(-1,'<LUDET>','IMSL IER>0 FROM LINV3F')
    ENDIF
    IF(D1 <= ZERO)THEN
       !
       ! Try to get determinant by using only non-negative eigenvalues
       ! (diagonal elements), similar as when getting thermodynamic quantities
       ! from vibrational frequencies
       !
       NIGNORE=0
       D1=ONE
       D2=ZERO
       DO I=1,NDIM
          IF(CO(I,I)  >  0.00001)THEN
             D1=D1*CO(I,I)
          ELSE
             NIGNORE=NIGNORE+1
          ENDIF
5         IF(ABS(D1)  <=  ONE) GOTO 10
          D1=D1*SIXTNTH
          D2=D2+FOUR
          GOTO 5
10        IF(ABS(D1)  >=  SIXTNTH) GOTO 15
          D1=D1*SIXTEEN
          D2=D2-FOUR
          GOTO 10
15        CONTINUE
       ENDDO
       IF(NIGNORE > 0 .AND. PRNLEV  >=  2) &
            WRITE(OUTU,'(/5X,A,I10,A/)') &
            'LUDET>',NIGNORE,' NON-POSITIVE DIAGONAL VALUES IGNORED'
    ENDIF
    RETURN
  END SUBROUTINE LUDET

end module corman2

