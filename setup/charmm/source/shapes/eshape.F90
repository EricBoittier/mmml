#if KEY_SHAPES==1 /*eshape*/
SUBROUTINE ESHAPE(ESHAP,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
     DD1,IUPT,QSECD)
  !
  use chm_kinds
  use memory
  use number
  use dimens_fcm
  use psf
  use select
  use shapes
  use stream
  !
  implicit none
  !
  real(chm_real) ESHAP
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  real(chm_real),allocatable,dimension(:) :: ARRAY
  real(chm_real),allocatable,dimension(:) :: DERSHP
  real(chm_real),allocatable,dimension(:) :: RADIUS
  !
  INTEGER ISHR,IS,JS,I,J,IPROP
  CHARACTER(len=4) TYPE1,TYPE2
  LOGICAL NEEDR,ERR
  !
  ESHAP=ZERO
  IF(NUMESH.LE.0) RETURN
  !
  ! Check to see if the number of atoms has changed...
  IF(NATOM.NE.SHPATL) THEN
     IS=0
     DO I=1,NUMSHP
        IF(SHPTYP(I).EQ.'RIGI' .OR. SHPTYP(I).EQ.'FLEX') IS=I
     ENDDO
     IF(IS.GT.0) THEN
        CALL WRNDIE(-3,'<ESHAPE>', &
             'The number of atoms has changed. All shapes become static')
        call chmdealloc('eshape.src','ESHAPE','SHPATP',SHPATL,intg=SHPATP)


        SHPATL=NATOM
        call chmalloc('eshape.src','ESHAPE','SHPATP',SHPATL,intg=SHPATP)
        SHPATP(1:SHPATL) = 0

        DO I=1,NUMSHP
           IF(SHPTYP(I).EQ.'RIGI' .OR. SHPTYP(I).EQ.'FLEX') &
                SHPTYP(I)='NONE'
           SHPNAT(I)=0
        ENDDO
     ENDIF
  ENDIF
  !
  IF(QSECD) THEN
     CALL WRNDIE(0,'<ESHAPE>','No second derivatives for shapes')
     RETURN
  ENDIF
  !
  call chmalloc('eshape.src','ESHAPE','ARRAY',NATOM,crl=ARRAY)
  call chmalloc('eshape.src','ESHAPE','DERSHP',LENSHP,crl=DERSHP)

  NEEDR=.FALSE.
  DO IPROP=1,NPRSHP
     IF(SHPPRP(2,IPROP).EQ.'HOMO' .OR. SHPPRP(2,IPROP).EQ.'GAUS') &
          NEEDR=.TRUE.
  ENDDO
  IF(NEEDR) THEN
     call chmalloc('eshape.src','ESHAPE','RADIUS',NATOM,crl=RADIUS)
     call SELPROP('SCA9',RADIUS,NATOM,ERR)
     IF(ERR) THEN
        CALL WRNDIE(-2,'<SHPCOM>','Radius array "SCA9" is not filled')
        RADIUS(1:NATOM)=zero
     ENDIF
  ENDIF
  !
  DO ISHR=1,NUMESH
     IS=0
     JS=0
     DO I=1,NUMSHP
        IF(ESHPNAM(1,ISHR).EQ.NAMSHP(I)) THEN
           IF(IS.NE.0) CALL WRNDIE(-3,'<ESHAPE>', &
                'Duplicate shape names')
           TYPE1=SHPTYP(I)
           IS=I
        ENDIF
        IF(ESHPNAM(2,ISHR).EQ.NAMSHP(I)) THEN
           IF(JS.NE.0) CALL WRNDIE(-3,'<ESHAPE>', &
                'Duplicate shape names')
           TYPE2=SHPTYP(I)
           JS=I
        ENDIF
     ENDDO
     !
     IF(IS*JS.GT.0) THEN
        IF(TYPE1.EQ.'FLEX' .OR. TYPE1.EQ.'RIGI') THEN
           CALL SHPFILL(IS,-1,0,[0],.FALSE.,X,Y,Z,NATOM)
        ENDIF
        IF(TYPE2.EQ.'FLEX' .OR. TYPE2.EQ.'RIGI') THEN
           CALL SHPFILL(JS,-1,0,[0],.FALSE.,X,Y,Z,NATOM)
        ENDIF
        !
        CALL ESHAP2(ESHAP,DX,DY,DZ,X,Y,Z,LENSHP, &
             NPRSHP,NATOM,WGHSHP, &
             IS,TYPE1,PTRSHP(IS)%a, &
             JS,TYPE2,PTRSHP(JS)%a, &
             ORDSHP,SHPATP,INDSHP,SHPPRPE,SHPPRP, &
             ESHPFC(ISHR),RADIUS, &
             ARRAY,DERSHP)
        !
     ENDIF
  ENDDO
  !
  call chmdealloc('eshape.src','ESHAPE','ARRAY',NATOM,crl=ARRAY)
  call chmdealloc('eshape.src','ESHAPE','DERSHP',LENSHP,crl=DERSHP)
  IF(NEEDR) call chmdealloc('eshape.src','ESHAPE','RADIUS',NATOM,crl=RADIUS)
  !
  RETURN
END SUBROUTINE ESHAPE

SUBROUTINE ESHAP2(ESHAP,DX,DY,DZ,X,Y,Z,LENSHP,NPROP,NATOM,WEIGH, &
     IS,TYPE1,SHAPE1,JS,TYPE2,SHAPE2, &
     NORD,SHPSEL,IPINDX,SHPPRPE,SHPPRP, &
     KFORCE,RADIUS,ARRAY,DERSHP)
  !
  use chm_kinds
  use number
  use dimens_fcm
  use select
  use stream
  implicit none
  !
  real(chm_real) ESHAP
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  INTEGER LENSHP,NPROP,NATOM
  real(chm_real) WEIGH(LENSHP,NPROP)
  INTEGER IS,JS
  CHARACTER(len=4) TYPE1,TYPE2
  real(chm_real) SHAPE1(LENSHP,NPROP),SHAPE2(LENSHP,NPROP)
  INTEGER NORD,SHPSEL(NATOM),IPINDX(*)
  INTEGER SHPPRPE(NPROP)
  CHARACTER(len=4) SHPPRP(3,NPROP)
  real(chm_real) KFORCE,RADIUS(NATOM),ARRAY(NATOM),DERSHP(LENSHP)
  !
  INTEGER I,J,IPROP
  real(chm_real) SSQ,DS
  LOGICAL DO1,DO2,ERR
  !
  DO1=(TYPE1.EQ.'FLEX' .OR. TYPE1.EQ.'RIGI')
  DO2=(TYPE2.EQ.'FLEX' .OR. TYPE2.EQ.'RIGI')
  !
  SSQ=ZERO
  DO IPROP=1,NPROP
     DO J=1,LENSHP
        SSQ=SSQ+(SHAPE1(J,IPROP)-SHAPE2(J,IPROP))**2 *WEIGH(J,IPROP)
     ENDDO
     !
     IF(DO1.OR.DO2) THEN
        !       get the propery array
        CALL SELPROP(SHPPRP(1,IPROP),ARRAY,NATOM,ERR)
        IF(ERR) THEN
           CALL WRNDIE(-1,'<SHPCOM>','Bad shape property array name')
           ARRAY(1:NATOM)=zero
        ENDIF
        DO J=1,LENSHP
           DERSHP(J)=KFORCE*WEIGH(J,IPROP)* &
                (SHAPE1(J,IPROP)-SHAPE2(J,IPROP))
        ENDDO
#if KEY_DEBUG==1
        WRITE(6,888) IPROP,DERSHP
888     FORMAT('Property',I4,'  DERSHP:'/,(10F12.4,/))
#endif 
     ENDIF
     !
     ! do forces for first shape...
     !
     IF(DO1) THEN
        CALL ESHAPDER(DX,DY,DZ,X,Y,Z,LENSHP,IS,SHAPE1(1,IPROP), &
             SHPSEL,NATOM,DERSHP, &
             RADIUS,ARRAY,NORD,IPINDX,SHPPRPE(IPROP), &
             SHPPRP(2,IPROP),SHPPRP(3,IPROP))
        !
     ENDIF
     !
     ! do forces for second shape...
     !
     IF(DO2) THEN
        DO J=1,LENSHP
           DERSHP(J)=-DERSHP(J)
        ENDDO
        CALL ESHAPDER(DX,DY,DZ,X,Y,Z,LENSHP,JS,SHAPE2(1,IPROP), &
             SHPSEL,NATOM,DERSHP, &
             RADIUS,ARRAY,NORD,IPINDX,SHPPRPE(IPROP), &
             SHPPRP(2,IPROP),SHPPRP(3,IPROP))
        !
     ENDIF
     !
  ENDDO
  !
  ESHAP=ESHAP+SSQ*HALF*KFORCE
  !
  RETURN
END SUBROUTINE ESHAP2

SUBROUTINE ESHAPDER(DX,DY,DZ,X,Y,Z,LENSHP, &
     ISHAPE,SHAPE1, &
     SHPSEL,NATOM,DERSHP, &
     RADIUS,ARRAY,NORD,IPINDX, &
     IEXP,PFILL,PCENT)
  !
  ! This routine fills a shape descriptor based on simple atom positions.
  !
  !     By Bernard R. Brooks   April, 1995
  !
  use chm_kinds
  use number
  use shapes, only: shapemoment
  use stream
  use machutil,only:die
  implicit none
  !
  real(chm_real)  DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  INTEGER LENSHP
  real(chm_real)  SHAPE1(LENSHP)
  INTEGER ISHAPE,NATOM
  INTEGER SHPSEL(NATOM)
  real(chm_real)  DERSHP(LENSHP),RADIUS(NATOM),ARRAY(NATOM)
  INTEGER NORD
  INTEGER IPINDX(*)
  INTEGER IEXP
  CHARACTER(len=4) PFILL,PCENT
  !
  INTEGER IS,JS,KS,I,IP,IORD,IPT,NP,NP2
  real(chm_real)  WTOT,WCEN,VALN,VALI,VALJ,VALK,DS
  real(chm_real)  XCEN,YCEN,ZCEN,XVAL,YVAL,ZVAL,XREC,YREC,ZREC
  real(chm_real)  DXCN,DYCN,DZCN,RX,RY,RZ,CX,CY,CZ,PX,PY,PZ
  LOGICAL ATZERO,QCENT
  !
  NP=NORD+1
  NP2=NP*NP
  !
  QCENT=(PCENT.EQ.'CENT')
  !
  IF(QCENT) THEN
     IP=1+1          ! X  INDEX
     XCEN=SHAPE1(IPINDX(IP))
     IP=1+NP         ! Y  INDEX
     YCEN=SHAPE1(IPINDX(IP))
     IP=1+NP2        ! Z  INDEX
     ZCEN=SHAPE1(IPINDX(IP))
     WCEN = SHAPE1(1)
     IF(ABS(WCEN).GT.TENM14) WCEN=ONE/WCEN
  ELSE
     XCEN=ZERO
     YCEN=ZERO
     ZCEN=ZERO
     WCEN=ZERO
  ENDIF
  DXCN=ZERO
  DYCN=ZERO
  DZCN=ZERO
  !
  WTOT=ONE
  IF(QCENT .OR. PCENT.EQ.'NORM') THEN
     WTOT = SHAPE1(1)
     IF(ABS(WTOT).GT.TENM14) WTOT=ONE/WTOT
  ENDIF
  !
  ! Fill the grid
  DO I=1,NATOM
     IF(IABS(SHPSEL(I)).EQ.ISHAPE) THEN
        !             calculate the property value for this atom.
        VALN=ARRAY(I)**IEXP
        XVAL=X(I)-XCEN
        YVAL=Y(I)-YCEN
        ZVAL=Z(I)-ZCEN
        IF(MIN(ABS(XVAL),ABS(YVAL),ABS(ZVAL)).GT.TENM14) THEN
           XREC=ONE/XVAL
           YREC=ONE/YVAL
           ZREC=ONE/ZVAL
           ATZERO=.FALSE.
        ELSE
           ATZERO=.TRUE.
        ENDIF
        !
        IPT=0
        IF(PFILL.EQ.'GRID') THEN
           CALL DIE  ! not yet coded...
        ELSE IF(PFILL.EQ.'NONE') THEN
           CONTINUE  ! do nothing
        ELSE IF(PFILL.EQ.'POIN') THEN
           VALI=VALN
           DO IS=0,NORD
              VALJ=VALI
              DO JS=0,NORD
                 VALK=VALJ
                 DO KS=0,NORD
                    IORD=IS+JS+KS
                    IF(IORD.LE.NORD) THEN
                       IPT=IPT+1
                       DS=DERSHP(IPT)*WTOT
                       IF(ATZERO) THEN
                          ! This atom is on a principal axis. Do an explicit calculation
                          IF(IS.GT.1) THEN
                             PX=IS*XVAL**(IS-1)
                             CX=XVAL**IS
                          ELSE IF(IS.EQ.1) THEN
                             PX=ONE
                             CX=XVAL
                          ELSE
                             PX=ZERO
                             CX=ONE
                          ENDIF
                          IF(JS.GT.1) THEN
                             PY=JS*YVAL**(JS-1)
                             CY=YVAL**JS
                          ELSE IF(JS.EQ.1) THEN
                             PY=ONE
                             CY=YVAL
                          ELSE
                             PY=ZERO
                             CY=ONE
                          ENDIF
                          IF(KS.GT.1) THEN
                             PZ=KS*ZVAL**(KS-1)
                             CZ=ZVAL**KS
                          ELSE IF(KS.EQ.1) THEN
                             PZ=ONE
                             CZ=ZVAL
                          ELSE
                             PZ=ZERO
                             CZ=ONE
                          ENDIF
                          RX=DS*VALN*PX*CY*CZ
                          RY=DS*VALN*CX*PY*CZ
                          RZ=DS*VALN*CX*CY*PZ
                       ELSE
                          RX=DS*VALK*IS*XREC
                          RY=DS*VALK*JS*YREC
                          RZ=DS*VALK*KS*ZREC
                       ENDIF
                       !
                       DX(I)=DX(I)+RX
                       DY(I)=DY(I)+RY
                       DZ(I)=DZ(I)+RZ
                       !
                       IF(IORD.GT.1) THEN
                          DXCN=DXCN-RX
                          DYCN=DYCN-RY
                          DZCN=DZCN-RZ
                       ENDIF
                    ENDIF
                    VALK=VALK*ZVAL
                 ENDDO
                 VALJ=VALJ*YVAL
              ENDDO
              VALI=VALI*XVAL
           ENDDO
        ELSE
           DO IS=0,NORD
              DO JS=0,NORD
                 DO KS=0,NORD
                    IORD=IS+JS+KS
                    IF(IORD.LE.NORD) THEN
                       IPT=IPT+1
                       DS=DERSHP(IPT)*WTOT*VALN
                       IF(DS.NE.ZERO) THEN
                          CALL SHAPEMOMENT(PFILL,RADIUS(I),XVAL,YVAL, &
                               ZVAL,IS,JS,KS,VALK, &
                               .TRUE.,RX,RY,RZ)
                          !C                        SHAPEF(IPT)=SHAPEF(IPT)+VALK*VALN
                          DX(I)=DX(I)+RX*DS
                          DY(I)=DY(I)+RY*DS
                          DZ(I)=DZ(I)+RZ*DS
                          IF(IORD.GT.1) THEN
                             DXCN=DXCN-RX*DS
                             DYCN=DYCN-RY*DS
                             DZCN=DZCN-RZ*DS
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDDO
              ENDDO
           ENDDO
        ENDIF
     ENDIF
  ENDDO
  !
  ! do correction for center option higher order terms.
  IF(QCENT) THEN
     DXCN=DXCN*WCEN
     DYCN=DYCN*WCEN
     DZCN=DZCN*WCEN
     DO I=1,NATOM
        IF(IABS(SHPSEL(I)).EQ.ISHAPE) THEN
           VALN=ARRAY(I)**IEXP
           DX(I)=DX(I)+DXCN*VALN
           DY(I)=DY(I)+DYCN*VALN
           DZ(I)=DZ(I)+DZCN*VALN
        ENDIF
     ENDDO
  ENDIF
  !
  return
end SUBROUTINE ESHAPDER
#endif /*  (eshape)*/
SUBROUTINE NULL_ESHAPE
  RETURN
END SUBROUTINE NULL_ESHAPE

