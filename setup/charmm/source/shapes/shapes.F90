module shapes
  use chm_kinds
  use chm_types
  implicit none

#if KEY_SHAPES==1 /*shapefcm*/
!   Shape descriptor common file
!
! Limits:
!     MAXSHP - Maximun mumber of descriptors
!     MNAMSH - Maximum number of characters in the descriptor name
!     MAXORD - Maximum order for shape factors
!     MAXPRP - Maximum number of properties
!     MAXESH - Maximum number of shape restraints
!
! Global values and data:
!     ORDSHP - Order of the shape descriptor
!     LENSHP - Number of elements in the shape descriptor
!     INDSHP - Pointer to index array for shape descriptors
!     WGHSHP - Pointer to weights for each element and property
!     SHPATP - Pointer to atom selection array
!     SHPATL - Length  of atom selection array
!     FACTORAL(*)   - Array to store N!  (for efficiency)
!     DBFACTORAL(*) - Array to store N!!
!
! The descriptors
!     NUMSHP    - Number of active descriptors
!     NUMSHPR   - Number of rigid shapes
!     NAMSHP(*) - Name of each descriptor
!     PTRSHP(*) - Heap pointer for shape descriptors
!     SHPNAT(*) - Number of atoms selcted for each shape
!                 (Negative indicates deselected atoms)
!     SHPTYP(*) - Type of shape: FIXE,FLEX,ROTA,TRAN,MOVE,DYNA
!                          (used with shape restraints)
!          STATic  - Change only upon explicit recalulate
!          DYNAmic - Recalculate every time
!          FOLLow  - May trans/rotate only during optimization (6 degf)
!          NONE    - Invalid or obsolete data
!
! The properties
!     NPRSHP      - Number of properties in each shape descriptor
!     NAMPRP(*)   - Name of property type
!     SHPPRP(1,*) - Property1: CHARMM array name for data source
!     SHPPRP(2,*) - Property2: POIN,HOMO,GAUS,GRID,NONE (fill option)
!          POINt       - Atoms are point sources
!          HOMOgeneous - Atoms are uniform spheres
!          GAUSsian    - Atoms are normalized 3-D gaussians
!          GRID        - Grid points are used (from COOR SEARch)
!          NONE        - Don't fill or modify this data
!     SHPPRP(3,*) - Property3: RAW,NORM,CENT (positioning option)
!
!     SHPPRPE(*)  - Property1 exponent (integer: 1,2,3,...)
!
!
! Variables for computing restraint shape energies.
!     NUMESH       - Number of shape restraint terms
!     ESHPNAM(2,*) - Name of two shapes to restrain
!     ESHPFC(*)    - Force constant for each restraint (def 1.0)
!
  integer, parameter :: MAXSHP=20, MNAMSH=8, MAXORD=10, MAXPRP=10, MAXESH=10

  INTEGER ORDSHP,LENSHP,NPRSHP,NUMSHP,NUMSHPR,NUMESH, &
          SHPPRPE(MAXPRP),SHPATL,SHPNAT(MAXSHP)

  integer,allocatable,dimension(:):: SHPATP
  real(chm_real),allocatable,dimension(:,:):: WGHSHP
  type(chm_ptr_2d),dimension(MAXSHP),save:: PTRSHP
  integer,allocatable,dimension(:):: INDSHP

  real(chm_real) ESHPFC(MAXESH),FACTORAL(-2:MAXORD),DBFACTORAL(-2:MAXORD)

  CHARACTER(len=MNAMSH) NAMSHP(MAXSHP),ESHPNAM(2,MAXESH),NAMPRP(MAXPRP)
  CHARACTER(len=4) SHPTYP(MAXSHP),SHPPRP(3,MAXPRP)

#endif /* (shapefcm)*/

contains

#if KEY_SHAPES==0 /*shapes*/
SUBROUTINE SHPCOM(COMLYN,COMLEN)
  INTEGER COMLEN
  CHARACTER*(*) COMLYN
  CALL WRNDIE(-1,'<SHPCOM>','Shape descriptor code is not compiled')
  return
end SUBROUTINE SHPCOM

#else /* (shapes)*/

subroutine shapes_init
  numshp=0  ! no shapes
  numshpr=0 ! no rigid shapes
  nprshp=0  ! no properties
  ordshp=-1 ! no order
  return
end subroutine shapes_init

SUBROUTINE SHPCOM(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  use memory
  use dimens_fcm
  use vector
  use number
  use corsubs,only:axisx,axisy,axisz,axisr,axiscx,axiscy,axiscz,qaxisc,fndu
  use psf
  use coord
  use coordc
  use select
  use stream
  use string

  integer,allocatable,dimension(:) :: ISLCT

  INTEGER COMLEN
  CHARACTER(len=*) COMLYN
  !
  !
  !
  CHARACTER(len=4) WRD,STYPE
  CHARACTER(len=MNAMSH) SNAME,ONAME
  INTEGER IS,JS,I,J,K,UNIT,NUM,IPROP,IUTIL,ITRY,NTRY,BTRY

  real(chm_real) XCEN,YCEN,ZCEN,FACT,PHI,XDIR,YDIR,ZDIR
  real(chm_real) BTRYVAL,TRYVAL,RJUNK
  real(chm_real) ROTM(3,4),ROTM2(3,4),UMAT(3,4),RN(3),RC(3)
  LOGICAL QWEIG,LAXIS,LOK,QAPPE,QPRINT,QSELE,QPROP,QCOMP,ERR
  real(chm_real),pointer,dimension(:,:):: I_ptr
  real(chm_real),allocatable,dimension(:) :: ARRAY
  !
  WRD=NEXTA4(COMLYN,COMLEN)
  !
  !-----------------------------------------------------------------------
  ! Code to clear the current data structures
  !
  IF(WRD.EQ.'CLEA') THEN
     QPROP=(INDXA(COMLYN,COMLEN,'PROP').GT.0)
     CALL FREESHP(QPROP)
     RETURN
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Specify a new ORDER value
  !
  IF(WRD.EQ.'ORDE') THEN
     IS=NEXTI(COMLYN,COMLEN)
     IF(PRNLEV.GT.2) &
          WRITE(OUTU,22) 'Specify order of shape descriptors'
     IF(IS.EQ.ORDSHP) RETURN    ! same as current order, ignore.
     IF(IS.LE.1 .OR. IS.GT.MAXORD) THEN
        CALL WRNDIE(-2,'<SHPCOM>','Specified order is not valid')
        RETURN
     ENDIF
     IF(NUMSHP.GT.0) CALL WRNDIE(0,'<SHPCOM>', &
          'Cannot modify order of existing shapes - all deleted')
     CALL FREESHP(.FALSE.)  ! don't delete properties
     CALL SETUPSHP(IS)
     RETURN
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Initialize upon first entry or when a new ORDER is specified
  !     setup new order with default value(3).
  IF(ORDSHP.LT.0) CALL SETUPSHP(3)
  !
  IF(NATOM.NE.SHPATL) THEN
     IS=0
     DO I=1,NUMSHP
        IF(SHPTYP(I).EQ.'RIGI' .OR. SHPTYP(I).EQ.'FLEX') IS=I
     ENDDO
     IF(IS.GT.0) THEN
        CALL WRNDIE(0,'<SHPCOM>', &
             'The number of atoms has changed. All shapes become static')
        call chmdealloc('shapes.src','SHPCOM','SHPATP',SHPATL,intg=SHPATP)
        SHPATL=NATOM

        call chmalloc('shapes.src','SHPCOM','SHPATP',SHPATL,intg=SHPATP)
        SHPATP(1:SHPATL) = 0
        DO I=1,NUMSHP
           IF(SHPTYP(I).EQ.'RIGI' .OR. SHPTYP(I).EQ.'FLEX') &
                SHPTYP(I)='NONE'
        ENDDO
     ENDIF
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Fill all of the dynamic shapes (type='FLEX')
  ! (Just in case any coordinate or property value has been modified).
  !
  DO IS=1,NUMSHP
     IF(SHPTYP(IS).EQ.'FLEX') THEN
        CALL SHPFILL(IS,-1,0,[0],.FALSE.,X,Y,Z,NATOM)
     ENDIF
  ENDDO
  !-----------------------------------------------------------------------
  ! Process vector option for trans/rot commands
  IF(WRD.EQ.'ROTA' .OR. WRD.EQ.'TRAN') THEN
     LAXIS=(INDXA(COMLYN,COMLEN,'AXIS').GT.0)
     IF(LAXIS) THEN
        IF(QAXISC) THEN
           RN(1)=AXISX
           RN(2)=AXISY
           RN(3)=AXISZ
           XCEN=AXISCX
           YCEN=AXISCY
           ZCEN=AXISCZ
        ELSE
           CALL WRNDIE(-1,'<CORMAN>', &
                'AXIS requested, but no axis yet defined. - Ignored')
           RETURN
        ENDIF
     ELSE
        FACT=GTRMF(COMLYN,COMLEN,'XDIR',ZERO)
        RN(1)=FACT
        FACT=GTRMF(COMLYN,COMLEN,'YDIR',ZERO)
        RN(2)=FACT
        FACT=GTRMF(COMLYN,COMLEN,'ZDIR',ZERO)
        RN(3)=FACT
        XCEN=GTRMF(COMLYN,COMLEN,'XCEN',ZERO)
        RC(1)=XCEN
        YCEN=GTRMF(COMLYN,COMLEN,'YCEN',ZERO)
        RC(2)=YCEN
        ZCEN=GTRMF(COMLYN,COMLEN,'ZCEN',ZERO)
        RC(2)=ZCEN
     ENDIF
     !
     PHI = DOTVEC(RN,RN,3)
     FACT=GTRMF(COMLYN,COMLEN,'DIST',ANUM)
     IF(FACT.NE.ANUM) THEN
        IF(PHI.LT.0.00001) THEN
           CALL WRNDIE(0,'<CORMAN>','No vector specified')
           RETURN
        ENDIF
        CALL NORMALL(RN,3)
        DO I=1,3
           RN(I)=RN(I)*FACT
        ENDDO
     ENDIF
     !
     FACT=GTRMF(COMLYN,COMLEN,'FACT',ONE)
     XDIR=RN(1)*FACT
     YDIR=RN(2)*FACT
     ZDIR=RN(3)*FACT
  ENDIF
  !-----------------------------------------------------------------------
  ! Start the main parse IF..
  !-----------------------------------------------------------------------
  IF (WRD.EQ.'PRIN') THEN
     ! process PRINT command
     IF(PRNLEV.GT.2) THEN
        WRITE(OUTU,22) 'Print descriptor data file elements'
        SNAME=NEXTA8(COMLYN,COMLEN)
        IUTIL=GTRMI(COMLYN,COMLEN,'ORDE',ORDSHP)
        DO I=1,NUMSHP
           IF(NAMSHP(I).EQ.SNAME .OR. SNAME.EQ.'ALL') THEN
              WRITE(OUTU,21) 'Printing elements of: ',NAMSHP(I), &
                   ' Type: ',SHPTYP(I)
              J=I
              IF(SHPTYP(I).EQ.'FLEX') J=-I
              CALL PRNSHP(OUTU,ORDSHP,NAMSHP(I),PTRSHP(I)%a, &
                   LENSHP,NPRSHP,NAMPRP,SHPATP,SHPATL, &
                   SHPNAT(I),J,SHPPRPE,SHPPRP,IUTIL)
           ENDIF
        ENDDO
        IF (SNAME.EQ.'WEIG') THEN
           !            print the current weighting array
           WRITE(OUTU,21) 'Printing elements of: ','WEIGHT'
           CALL PRNSHP(OUTU,ORDSHP,'WEIGHT',WGHSHP, &
                LENSHP,NPRSHP,NAMPRP,[0],0,0,0,SHPPRPE,SHPPRP, &
                IUTIL)
        ENDIF
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'LIST') THEN
     ! process LIST command
     IF(PRNLEV.GT.2) THEN
        !
        !         general information
        WRITE(OUTU,23) 'The order  of current descriptors is:',ORDSHP
        WRITE(OUTU,23) 'The length of current descriptors is:',LENSHP
        WRITE(OUTU,23) 'The number of current descriptors is:',NUMSHP
        !
        WRITE(OUTU,22) 'List of active descriptors:'
        DO I=1,NUMSHP
           IF(SHPTYP(I).EQ.'STAT' .OR. SHPTYP(I).EQ.'NONE') THEN
              WRITE(OUTU,44) NAMSHP(I),SHPTYP(I), &
                   'No assigned atoms'
           ELSE
              IS=NSELCTV(SHPATL,SHPATP,I)
              IF(IS.NE.SHPNAT(I)) IS=ISIGN(SHPNAT(I),-1)
              IF(IS.GE.0) THEN
                 WRITE(OUTU,44) NAMSHP(I),SHPTYP(I), &
                      'Number of allocated atoms:',IS
              ELSE
                 WRITE(OUTU,44) NAMSHP(I),SHPTYP(I), &
                      'Number of deallocated atoms:',-IS
              ENDIF
           ENDIF
44         FORMAT(5X,'Descriptor: "',A,'"  Type: "',A4,'"   ',A,I6)
        ENDDO
        !
        WRITE(OUTU,23) 'The number of current properties: ',NPRSHP
        DO I=1,NPRSHP
           WRITE(OUTU,45) I,NAMPRP(I),SHPPRP(1,I),SHPPRPE(I), &
                SHPPRP(2,I),SHPPRP(3,I)
45         FORMAT(I3,'  Property: "',A,'"  Data from: "',A4, &
                '"  Exponent:',I3,'  Atoms as: "',A4, &
                '"  Type: "',A4,'"')
        ENDDO
        !
        WRITE(OUTU,23) 'The number of energy restraints: ',NUMESH
        DO I=1,NUMESH
           IS=0
           JS=0
           DO J=1,NUMSHP
              IF(ESHPNAM(1,I).EQ.NAMSHP(J)) IS=J
              IF(ESHPNAM(2,I).EQ.NAMSHP(J)) JS=J
           ENDDO
           IF(IS*JS.GT.0) THEN
              WRITE(OUTU,46) I,ESHPNAM(1,I),SHPTYP(IS), &
                   ESHPNAM(2,I),SHPTYP(JS),ESHPFC(I)
46            FORMAT(I3,'   Active restraint:  First: "',A, &
                   '" type: "',A,'"  Second: "',A,'" type: "',A, &
                   '"  Force constant:',F12.5)
           ELSE
              WRITE(OUTU,47) I,ESHPNAM(1,I),ESHPNAM(2,I)
47            FORMAT(I3,' Inactive restraint:  First: "',A, &
                   '"               Second: "',A,'"')
           ENDIF
        ENDDO
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'REST') THEN
     WRD=NEXTA4(COMLYN,COMLEN)
     IF(WRD.EQ.'CLEA') THEN
        NUMESH=0
        IF(PRNLEV.GT.2) WRITE(OUTU,23) &
             'The number of energy restraints: ',NUMESH
        RETURN
     ENDIF
     !
     FACT=GTRMF(COMLYN,COMLEN,'FORC',ONE)
     SNAME=NEXTA8(COMLYN,COMLEN)
     ONAME=NEXTA8(COMLYN,COMLEN)
     IS=0
     DO I=1,NUMESH
        IF(ESHPNAM(1,I).EQ.SNAME .AND. ESHPNAM(2,I).EQ.ONAME) IS=I
        IF(ESHPNAM(1,I).EQ.ONAME .AND. ESHPNAM(2,I).EQ.SNAME) IS=I
     ENDDO
     !
     IF(WRD.EQ.'ADD') THEN
        IF(IS.GT.0) THEN
           CALL WRNDIE(-2,'<SHPCOM>', &
                'New shape restraint already present')
           RETURN
        ENDIF
        IF(MAXESH.EQ.NUMESH) THEN
           CALL WRNDIE(-2,'<SHPCOM>','Too many shape restraints')
           RETURN
        ENDIF
        NUMESH=NUMESH+1
        ESHPFC(NUMESH)=FACT
        ESHPNAM(1,NUMESH)=SNAME
        ESHPNAM(2,NUMESH)=ONAME
     ELSE IF(WRD.EQ.'MODI') THEN
        IF(IS.EQ.0) THEN
           CALL WRNDIE(-2,'<SHPCOM>','Shape restraint not found')
           RETURN
        ENDIF
        IF(FACT.EQ.ESHPFC(IS)) THEN
           CALL WRNDIE(-2,'<SHPCOM>','Same force constant specified')
        ENDIF
        ESHPFC(IS)=FACT
     ELSE
        CALL WRNDIE(-2,'<SHPCOM>','Unrecognized restrain option')
        RETURN
     ENDIF
     !
     IF(PRNLEV.GT.2) WRITE(OUTU,23) &
          'The number of energy restraints: ',NUMESH
     I=NUMESH
     IS=0
     JS=0
     DO J=1,NUMSHP
        IF(ESHPNAM(1,I).EQ.NAMSHP(J)) IS=J
        IF(ESHPNAM(2,I).EQ.NAMSHP(J)) JS=J
     ENDDO
     IF(IS*JS.GT.0) THEN
        IF(PRNLEV.GT.2) WRITE(OUTU,46) I,ESHPNAM(1,I),SHPTYP(IS), &
             ESHPNAM(2,I),SHPTYP(JS),ESHPFC(I)
     ELSE
        CALL WRNDIE(-2,'<SHPCOM>', &
             'New shape restraint is innactive')
        IF(PRNLEV.GT.2) WRITE(OUTU,47) I,ESHPNAM(1,I),ESHPNAM(2,I)
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'WRIT') THEN
     ! process WRITE command
     IF(PRNLEV.GT.2) WRITE(OUTU,22) 'Write descriptor data file'
     UNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
     CALL SHPWRI(UNIT)
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'READ') THEN
     ! process READ command
     IF(PRNLEV.GT.2) WRITE(OUTU,22) 'Read descriptor data file'
     QAPPE=(INDXA(COMLYN,COMLEN,'APPE').GT.0)
     UNIT=GTRMI(COMLYN,COMLEN,'UNIT',ISTRM)
     CALL SHPREA(UNIT,QAPPE)
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'ROTA') THEN
     ! process ROTATE command
     IF(PRNLEV.GT.2) &
          WRITE(OUTU,22) 'Rotate descriptor data file element'
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.TRUE.)
     IF(IS.LE.0) RETURN
     !
     PHI=GTRMF(COMLYN,COMLEN,'PHI',ZERO)
     CALL FNDU(UMAT,RN,PHI,LOK)
     IF(.NOT.LOK) THEN
        CALL WRNDIE(0,'<SHPCOM>','ROTAte parsing error')
        RETURN
     ENDIF
     !
     DO I=1,3
        ROTM(I,4)=-RC(I)
        DO J=1,3
           ROTM(I,J)=UMAT(J,I)
           ROTM(I,4)=ROTM(I,4)+RC(J)*UMAT(J,I)
        ENDDO
     ENDDO
     !
     !C         ROTM(1,4)=XCEN*UMAT(1,1)+YCEN*UMAT(2,1)+ZCEN*UMAT(3,1)-XCEN
     !C         ROTM(2,4)=XCEN*UMAT(1,2)+YCEN*UMAT(2,2)+ZCEN*UMAT(3,2)-YCEN
     !C         ROTM(3,4)=XCEN*UMAT(1,3)+YCEN*UMAT(2,3)+ZCEN*UMAT(3,3)-ZCEN
     !
     IF(PRNLEV.GT.3) WRITE(OUTU,26) &
          'Rotation operation on shape: ',SNAME,ROTM
     !

     call chmalloc('shapes.src','SHPCOM','I_ptr',LENSHP,MAXPRP,crlp=I_ptr)
     CALL SHAPEROT(ORDSHP,LENSHP,NPRSHP,PTRSHP(IS)%a,I_ptr, &
          ROTM,INDSHP,IS,SHPTYP(IS),SHPATP, &
          X,Y,Z,NATOM,SHPPRP)
     call chmdealloc('shapes.src','SHPCOM','PTRSHP(IS)',LENSHP,MAXPRP,crlp=PTRSHP(IS)%a)
     PTRSHP(IS)%a => I_ptr
     nullify(I_ptr)

     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'TRAN') THEN
     ! process TRANSLATE command
     IF(PRNLEV.GT.2) &
          WRITE(OUTU,22) 'Translate descriptor data file element'
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.TRUE.)
     IF(IS.LE.0) RETURN

     DO I=1,3
        DO J=1,3
           ROTM(J,I)=ZERO
        ENDDO
        ROTM(I,I)=ONE
     ENDDO
     ROTM(1,4)=XDIR
     ROTM(2,4)=YDIR
     ROTM(3,4)=ZDIR
     !
     IF(PRNLEV.GT.3) WRITE(OUTU,26) &
          'Translation operation on shape: ',SNAME,ROTM
     !

     call chmalloc('shapes.src','SHPCOM','I_ptr',LENSHP,MAXPRP,crlp=I_ptr)
     CALL SHAPEROT(ORDSHP,LENSHP,NPRSHP,PTRSHP(IS)%a,I_ptr, &
          ROTM,INDSHP,IS,SHPTYP(IS),SHPATP, &
          X,Y,Z,NATOM,SHPPRP)
     call chmdealloc('shapes.src','SHPCOM','PTRSHP(IS)%a1',LENSHP,MAXPRP,crlp=PTRSHP(IS)%a)
     PTRSHP(IS)%a => I_ptr
     nullify(I_ptr)

     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'CENT') THEN
     ! process CENTER command
     IF(PRNLEV.GT.2) &
          WRITE(OUTU,22) 'Center descriptor data file element'
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.TRUE.)
     IF(IS.LE.0) RETURN
     !
     IPROP=GTRMI(COMLYN,COMLEN,'PROP',1)
     IF(IPROP.LE.0 .OR. IPROP.GT.NPRSHP) THEN
        CALL WRNDIE(-2,'<SHPCOM>','Bad property number')
        RETURN
     ENDIF
     !
     IUTIL=0
     IF(INDXA(COMLYN,COMLEN,'MAJO').GT.0) IUTIL=1
     IF(INDXA(COMLYN,COMLEN,'MINO').GT.0) IUTIL=2
     IF(INDXA(COMLYN,COMLEN,'NORM').GT.0) IUTIL=3
     !
     CALL SHPGETCEN(ORDSHP,PTRSHP(IS)%a,LENSHP,NPRSHP,IPROP, &
          ROTM,INDSHP,IUTIL,SHPPRP(3,IPROP))
     IF(PRNLEV.GT.3) WRITE(OUTU,26) 'First ROTM:  ',SNAME,ROTM
     !
     CALL TRIMA(COMLYN,COMLEN)
     IF(COMLEN.GT.0) THEN
        CALL GETSHPNM(COMLYN,COMLEN,JS,ONAME,.TRUE.)
        IF(JS.GT.0) THEN
           CALL SHPGETCEN(ORDSHP,PTRSHP(JS)%a,LENSHP,NPRSHP,IPROP, &
                ROTM2,INDSHP,0,SHPPRP(3,IPROP))
           IF(PRNLEV.GT.3) WRITE(OUTU,26) 'Second ROTM:  ',ONAME,ROTM2
           !
           !            multiply the two rotation matricies (second inverted)
           !CC             DO I=1,3
           !CC               DO J=1,3
           !CC                 U(J,I)=ZERO
           !CC                 DO K=1,3
           !CC                   U(J,I)=U(J,I)+ROTM(K,I)*ROTM2(K,J)
           !CC                 ENDDO
           !CC               ENDDO
           !CC               RC(I)=ROTM(I,4)-ROTM2(I,4)
           !CC             ENDDO
           !CC             DO I=1,3
           !CC               ROTM(I,4)=ZERO
           !CC               DO J=1,3
           !CC                 ROTM(I,J)=U(I,J)
           !CC                 ROTM(I,4)=ROTM(I,4)+RC(J)*ROTM2(J,I)
           !CC               ENDDO
           !CC             ENDDO
           !
           CALL INVTTRM(ROTM2,ROTM2)
           !
           ! Search for the bestfit (4 possibilities if chiral, 8 if not)
           !
           BTRY=0
           BTRYVAL=ANUM
           NTRY=4
           IF(INDXA(COMLYN,COMLEN,'NONC').GT.0) NTRY=8 ! nonchiral
           DO ITRY=1,NTRY
              DO I=1,3
                 DO J=1,4
                    UMAT(I,J)=ZERO
                 ENDDO
                 UMAT(I,I)=ONE
              ENDDO
              IF(MOD(ITRY+1,2).EQ.1) UMAT(3,3)=MINONE
              IF(MOD(ITRY/2,2).EQ.1) UMAT(2,2)=MINONE
              IF(MOD((ITRY+1)/4,2).EQ.1) UMAT(1,1)=MINONE
              CALL MULTTRM(ROTM2,UMAT,UMAT)
              CALL MULTTRM(UMAT,ROTM,UMAT)

              call chmalloc('shapes.src','SHPCOM','I_prt',LENSHP,MAXPRP,crlp=I_ptr)
              CALL SHAPEROT(ORDSHP,LENSHP,NPRSHP,PTRSHP(IS)%a, &
                   I_ptr,UMAT,INDSHP, &
                   IS,'NONE',[0],[ZERO],[ZERO],[ZERO],0,SHPPRP)
              CALL SHAPEMAN('COMP',-1,ORDSHP,I_ptr,PTRSHP(JS)%a, &
                   LENSHP,NPRSHP,WGHSHP,ZERO,TRYVAL)
              call chmdealloc('shapes.src','SHPCOM','I_ptr',LENSHP,MAXPRP,crlp=I_ptr)

              IF(TRYVAL.LT.BTRYVAL) THEN
                 BTRYVAL=TRYVAL
                 BTRY=ITRY
              ENDIF
           ENDDO
           !
           ITRY=BTRY
           DO I=1,3
              DO J=1,4
                 UMAT(I,J)=ZERO
              ENDDO
              UMAT(I,I)=ONE
           ENDDO
           IF(MOD(ITRY+1,2).EQ.1) UMAT(3,3)=MINONE
           IF(MOD(ITRY/2,2).EQ.1) UMAT(2,2)=MINONE
           IF(MOD((ITRY+1)/4,2).EQ.1) UMAT(1,1)=MINONE
           CALL MULTTRM(ROTM2,UMAT,UMAT)
           CALL MULTTRM(UMAT,ROTM,ROTM)
           !
           !CCC             CALL MULTTRM(ROTM2,ROTM,ROTM)
           !
        ENDIF
        IF(PRNLEV.GT.3) THEN
           WRITE(OUTU,22) 'Bestfit operation on shape: ',SNAME
           WRITE(OUTU,26) 'Relative to current shape:  ',ONAME,ROTM
        ENDIF
     ELSE
        IF(PRNLEV.GT.3) WRITE(OUTU,26) &
             'Center to origin shape: ',SNAME,ROTM
     ENDIF
     !
     call chmalloc('shapes.src','SHPCOM','I_ptr',LENSHP,MAXPRP,crlp=I_ptr)
     CALL SHAPEROT(ORDSHP,LENSHP,NPRSHP,PTRSHP(IS)%a,I_ptr, &
          ROTM,INDSHP,IS,SHPTYP(IS),SHPATP, &
          X,Y,Z,NATOM,SHPPRP)
     call chmdealloc('shapes.src','SHPCOM','PTRSHP(IS)%a',LENSHP,MAXPRP,crlp=PTRSHP(IS)%a)

     PTRSHP(IS)%a => I_ptr
     nullify(I_ptr)
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'STAT') THEN
     ! Calculate and print some shape properties

     continue

     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'AXIS') THEN
     ! Create an axis for a particular shape (as in COOR LSQP)
     IF(PRNLEV.GT.2) &
          WRITE(OUTU,22) 'Statistics for a descriptor'
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.TRUE.)
     IF(IS.EQ.0) RETURN
     !
     IPROP=GTRMI(COMLYN,COMLEN,'PROP',1)
     IF(IPROP.LE.0 .OR. IPROP.GT.NPRSHP) THEN
        CALL WRNDIE(-2,'<SHPCOM>','Bad property number')
        RETURN
     ENDIF
     !
     IUTIL=0
     IF(INDXA(COMLYN,COMLEN,'MAJO').GT.0) IUTIL=1
     IF(INDXA(COMLYN,COMLEN,'MINO').GT.0) IUTIL=2
     IF(INDXA(COMLYN,COMLEN,'NORM').GT.0) IUTIL=3
     !
     DO I=1,NUMSHP
        IF(I.EQ.IS .OR. IS.EQ.-1) THEN
           CALL SHPGETCEN(ORDSHP,PTRSHP(IS)%a,LENSHP,NPRSHP,IPROP, &
                ROTM,INDSHP,IUTIL,SHPPRP(3,IPROP))
        ENDIF
     ENDDO
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'COMP') THEN
     ! process FIT command
     IF(PRNLEV.GT.2) WRITE(OUTU,22) 'Compare two descriptors'
     IPROP=GTRMI(COMLYN,COMLEN,'PROP',-1)
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.TRUE.)
     CALL GETSHPNM(COMLYN,COMLEN,JS,SNAME,.TRUE.)
     DO I=1,NUMSHP
        IF(I.EQ.IS .OR. IS.EQ.-1) THEN
           DO J=1,NUMSHP
              IF(J.EQ.JS .OR. JS.EQ.-1) THEN
                 CALL SHAPEMAN(WRD,IPROP,ORDSHP,PTRSHP(IS)%a, &
                      PTRSHP(JS)%a,LENSHP,NPRSHP,WGHSHP, &
                      ZERO,RJUNK)
              ENDIF
           ENDDO
        ENDIF
     ENDDO
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'SUM' .OR. WRD.EQ.'DIFF' .OR. WRD.EQ.'COPY') THEN
     ! process manipulate commands
     IF(PRNLEV.GT.2) THEN
        WRITE(OUTU,22) WRD//' two descriptor data file elements'
        WRITE(OUTU,22) 'First element is modified'
     ENDIF
     IPROP=GTRMI(COMLYN,COMLEN,'PROP',-1)
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.TRUE.)
     CALL GETSHPNM(COMLYN,COMLEN,JS,SNAME,.TRUE.)
     IF(IS.LE.0 .OR. JS.LE.0) RETURN
     CALL SHAPEMAN(WRD,IPROP,ORDSHP,PTRSHP(IS)%a, &
          PTRSHP(JS)%a,LENSHP,NPRSHP,WGHSHP, &
          ZERO,RJUNK)
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'SCAL') THEN
     ! process manipulate commands
     IPROP=GTRMI(COMLYN,COMLEN,'PROP',-1)
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.TRUE.)
     PHI=NEXTF(COMLYN,COMLEN)
     DO I=1,NUMSHP
        IF(I.EQ.IS .OR. IS.EQ.-1) THEN
           IF(PRNLEV.GT.2) WRITE(OUTU,22) NAMSHP(I),' is scaled'
           CALL SHAPEMAN(WRD,IPROP,ORDSHP,PTRSHP(I)%a, &
                PTRSHP(I)%a,LENSHP,NPRSHP,WGHSHP, &
                PHI,RJUNK)
        ENDIF
     ENDDO
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'DUPL') THEN
     ! process DUPLicate command
     IF(PRNLEV.GT.2) THEN
        WRITE(OUTU,22) 'Duplicate a descriptor data file element'
        WRITE(OUTU,22) 'First element is new'
     ENDIF
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.FALSE.)
     CALL GETSHPNM(COMLYN,COMLEN,JS,WRD,.TRUE.)
     IF(JS.LE.0) RETURN
     IF(IS.EQ.0) THEN
        IF(NUMSHP.EQ.MAXSHP) THEN
           CALL WRNDIE(-2,'<SHPCOM>','Too many shape descriptors')
           RETURN
        ENDIF
        NUMSHP=NUMSHP+1
        NAMSHP(NUMSHP)=SNAME
        SHPTYP(NUMSHP)=SHPTYP(JS)
        IF(SHPTYP(JS).EQ.'FLEX' .OR. SHPTYP(JS).EQ.'RIGI') THEN
           SHPTYP(NUMSHP)='NONE'
           CALL WRNDIE(1,'<SHPCOM>', &
                'New duplicate is given type: NONE')
        ENDIF

        call chmalloc('shapes.src','SHPCOM','PTRSHP(NUMSHP)',LENSHP,MAXPRP,crlp=PTRSHP(NUMSHP)%a)
        PTRSHP(NUMSHP)%a=ZERO
        IS=NUMSHP
     ELSE
        CALL WRNDIE(-1,'<SHPCOM>', &
             'Cannot DUPLicate to an existing descriptor, use COPY')
        RETURN
     ENDIF
     WRD='COPY'
     CALL SHAPEMAN(WRD,-1,ORDSHP,PTRSHP(IS)%a, &
          PTRSHP(JS)%a,LENSHP,NPRSHP,WGHSHP, &
          ZERO,RJUNK)
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'WEIG') THEN
     ! process WEIGHT command
     QPRINT=(INDXA(COMLYN,COMLEN,'PRIN').GT.0 .AND. PRNLEV.GT.2)
     IF(PRNLEV.GT.2) WRITE(OUTU,22) 'Weight descriptor data file'
     CALL SHPWGH(COMLYN,COMLEN,ORDSHP,WGHSHP, &
          LENSHP,NPRSHP,INDSHP)
     !     print the current weighting array
     IF(QPRINT) THEN
        CALL PRNSHP(OUTU,ORDSHP,'WEIGHT',WGHSHP,LENSHP, &
             NPRSHP,NAMPRP,[0],0,0,0,SHPPRPE,SHPPRP,ORDSHP)
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'DELE') THEN
     ! process DELETE command
     IF(PRNLEV.GT.2) &
          WRITE(OUTU,22) 'Delete descriptor from data file.'
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.TRUE.)
     IF(IS.EQ.-1) THEN
        NUM=NUMSHP
        DO I=1,NUM
           CALL DELSHAPE(NUMSHP,SHPATP,NATOM)
        ENDDO
     ENDIF
     IF(IS.LE.0) RETURN
     CALL DELSHAPE(IS,SHPATP,NATOM)
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'DESC') THEN
     ! process add or modify a descriptor
     IF(PRNLEV.GT.2) WRITE(OUTU,22) &
          'DESCriptor command: Edit or create a descriptor.'
     !
     call chmalloc('shapes.src','SHPCOM','ISLCT',NATOM,intg=ISLCT)

     QSELE=(INDX(COMLYN,COMLEN,'SELE',4).GT.0)
     IF(QSELE) THEN
        call SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     ENDIF
     !
     CALL GETSHPNM(COMLYN,COMLEN,IS,SNAME,.FALSE.)
     STYPE=NEXTA4(COMLYN,COMLEN)
     QCOMP=(INDXA(COMLYN,COMLEN,'COMP').GT.0)
     !
     IF(IS.EQ.-1) THEN
        ! edit all descriptors
        IF(PRNLEV.GT.2) WRITE(OUTU,22) 'Editing all descriptors'
        DO I=1,NUMSHP
           CALL EDITSHAPE(I,STYPE,SHPATP,ISLCT,QSELE,IMOVE,NATOM)
        ENDDO
     ELSE IF(IS.EQ.0) THEN
        ! add a new descritor
        IF(NUMSHP.GE.MAXSHP) THEN
           CALL WRNDIE(-2,'<SHPCOM>', &
                'Too many shape descriptors - Cannot add')
           RETURN
        ENDIF
        IF(.NOT.QSELE) THEN
           ISLCT(1:NATOM) = 0
           QSELE=.TRUE.
        ENDIF
        NUMSHP=NUMSHP+1
        IS=NUMSHP
        NAMSHP(IS)=SNAME
        SHPTYP(IS)='NONE'

        call chmalloc('shapes.src','SHPCOM','PTRSHP(IS)',LENSHP,MAXPRP,crlp=PTRSHP(IS)%a)
        PTRSHP(IS)%a=ZERO
        IF(STYPE.EQ.' ') THEN
           CALL WRNDIE(-2,'<SHPCOM>', &
                'No shape type specified for new shape')
           STYPE='NONE'
        ENDIF
        !
        CALL EDITSHAPE(IS,STYPE,SHPATP,ISLCT,QSELE,IMOVE,NATOM)
        !            fill a new shape
        IF(PRNLEV.GT.2) WRITE(OUTU,22) &
             'Add a new descriptor named: ',SNAME, &
             '  Type: ',SHPTYP(IS)
     ELSE
        ! edit one descriptor
        IF(PRNLEV.GT.2) WRITE(OUTU,22) &
             'Editing descriptor named: '//SNAME
        CALL EDITSHAPE(IS,STYPE,SHPATP,ISLCT,QSELE,IMOVE,NATOM)
     ENDIF
     !
     IF(QCOMP) THEN
        CALL SHPFILL(IS,-1,1,ISLCT,.TRUE., &
             XCOMP,YCOMP,ZCOMP,NATOM)
     ELSE
        CALL SHPFILL(IS,-1,1,ISLCT,.TRUE.,X,Y,Z,NATOM)
     ENDIF
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD.EQ.'PROP') THEN
     ! add or modify a property
     IPROP=NEXTI(COMLYN,COMLEN)
     IF(IPROP.LE.0 .OR. IPROP.GT.NPRSHP+1) THEN
        CALL WRNDIE(-2,'<SHPCOM>','Bad property number specified')
        RETURN
     ELSE IF(IPROP.GT.MAXPRP) THEN
        CALL WRNDIE(-2,'<SHPCOM>','Too many properties')
        RETURN
     ENDIF
     ! parse property name
     NAMPRP(IPROP)=NEXTA8(COMLYN,COMLEN)
     !
     ! parse array name to obtain atom data
     STYPE=NEXTA4(COMLYN,COMLEN)
     call chmalloc('shapes.src','SHPCOM','ARRAY',NATOM,crl=ARRAY)
     CALL SELPROP(STYPE,ARRAY,NATOM,ERR)
     call chmdealloc('shapes.src','SHPCOM','ARRAY',NATOM,crl=ARRAY)
     IF(ERR) THEN
        CALL WRNDIE(-2,'<SHPCOM>','Bad first property specified')
        STYPE='ZERO'
     ENDIF
     SHPPRP(1,IPROP)=STYPE
     !
     ! parse exponent
     I=GTRMI(COMLYN,COMLEN,'EXPO',SHPPRPE(IPROP))
     IF(I.GT.0) THEN
        SHPPRPE(IPROP)=I
     ELSE
        CALL WRNDIE(-2,'<SHPCOM>','Bad exponent specified')
        SHPPRPE(IPROP)=1
     ENDIF
     !
     ! parse atom option
     WRD=NEXTA4(COMLYN,COMLEN)
     IF(WRD.EQ.'POIN' .OR. WRD.EQ.'HOMO' .OR. WRD.EQ.'GAUS' .OR. &
          WRD.EQ.'GRID' .OR. WRD.EQ.'NONE') THEN
        SHPPRP(2,IPROP)=WRD
     ELSE
        CALL WRNDIE(-2,'<SHPCOM>','Bad second property specified')
        SHPPRP(2,IPROP)='NONE'
     ENDIF
     !
     ! parse centering option
     WRD=NEXTA4(COMLYN,COMLEN)
     IF(WRD.EQ.' ') WRD='RAW '
     IF(WRD.EQ.'RAW ' .OR. WRD.EQ.'CENT' .OR. WRD.EQ.'NORM') THEN
        SHPPRP(3,IPROP)=WRD
     ELSE
        CALL WRNDIE(-2,'<SHPCOM>','Bad third property specified')
        SHPPRP(3,IPROP)='NONE'
     ENDIF
     !
     IF(IPROP.GT.NPRSHP) THEN
        !  Add a new property
        IF(PRNLEV.GT.2) WRITE(OUTU,22) &
             'PROP command: A new property added named: '//NAMPRP(IPROP)
        NPRSHP=IPROP
     ELSE
        !  Edit a current property
        IF(PRNLEV.GT.2) WRITE(OUTU,22) &
             'PROP command: Editing an existing property'
     ENDIF
     ! Recalulate elements for this property for all non-static descriptors
     CALL SHPFILL(-1,IPROP,0,[0],.TRUE.,X,Y,Z,NATOM)
     !
     !-----------------------------------------------------------------------
  ELSE
     CALL WRNDIE(-1,'<SHPCOM>','Unrecognized SHAPe command keyname')
  ENDIF
  !-----------------------------------------------------------------------
21 FORMAT(/'SHAPES: ',5A)
22 FORMAT('SHAPES: ',5A)
23 FORMAT('SHAPES: ',A,I6)
25 FORMAT('        ',A,I6)
26 FORMAT('SHAPES: ',2A,4(/,15X,3F12.5))
  !

  IF(allocated(ISLCT)) call chmdealloc('shapes.src','SHPCOM','ISLCT',NATOM,intg=ISLCT)
  IF(allocated(ARRAY)) call chmdealloc('shapes.src','SHPCOM','ARRAY',NATOM,crl=ARRAY)

  RETURN
END SUBROUTINE SHPCOM

SUBROUTINE FREESHP(QPROP)
  !
  ! Free and clear the SHAPE data structure
  !
  use memory
  use number
  use stream

  LOGICAL QPROP
  !
  INTEGER I
  !
  ! free space for shapes
  IF(NUMSHP.GT.0) THEN
     !        delete all existing shape factors
     DO I=1,NUMSHP
        call chmdealloc('shapes.src','FREESHP','PTRSHP(I)',LENSHP,MAXPRP,crlp=PTRSHP(I)%a)
     ENDDO
  ENDIF
  ! Free space for wrighting array and index array
  IF(ORDSHP.GE.0) THEN
     I=(ORDSHP+1)**3
     call chmdealloc('shapes.src','FREESHP','INDSHP',I,intg=INDSHP)
     call chmdealloc('shapes.src','FREESHP','WGHSHP',LENSHP,MAXPRP,crl=WGHSHP)
     IF(SHPATL.NE.0) THEN
        call chmdealloc('shapes.src','FREESHP','SHPATP',SHPATL,intg=SHPATP)
        SHPATL=0
     ENDIF
  ENDIF
  !
  NUMSHP=0            ! no shapes
  NUMSHPR=0           ! no rigid shapes
  ORDSHP=-1           ! no order
  IF(QPROP) NPRSHP=0  ! no properties
  !
  IF(PRNLEV.GT.2) WRITE(OUTU,22) 'Shape descriptor data cleared.'
22 FORMAT('SHAPES: ',A,'.')
  !
  RETURN
END SUBROUTINE FREESHP

SUBROUTINE SETUPSHP(ORD)
  !-----------------------------------------------------------------------
  ! Initialize upon first entry or when a new ORDER is specified
  !  or when reading shapes from a file.  This routine assumes
  !  that the data structues have been already cleared.
  !-----------------------------------------------------------------------
  !
  use memory
  use dimens_fcm
  use number
  use stream
  use psf

  INTEGER ORD
  !
  INTEGER I,N,MX
  !
  ! set default shape order and length
  ORDSHP=ORD
  LENSHP=((ORDSHP+1)*(ORDSHP+2)*(ORDSHP+3))/6
  IF(PRNLEV.GT.2) WRITE(OUTU,44) ORDSHP,LENSHP
44 FORMAT('SHAPES: New order is:',I5,' New length is:',I5)
  I=(ORDSHP+1)**3

  call chmalloc('shapes.src','SETUPSHP','INDSHP',I,intg=INDSHP)
  CALL MKSHPI(ORDSHP,INDSHP)
  call chmalloc('shapes.src','SETUPSHP','WGHSHP',LENSHP,MAXPRP,crl=WGHSHP)
  WGHSHP=ONE
  SHPATL=NATOM
  call chmalloc('shapes.src','SETUPSHP','SHPATP',SHPATL,intg=SHPATP)
  SHPATP(1:SHPATL) = 0
  !
  ! setup factorial arrays (N! and N!!)
  !
  DO N=-2,ORDSHP
     MX=1
     DO I=1, N
        MX=MX*I
     ENDDO
     FACTORAL(N)=MX
     MX=1
     DO I=N,1,-2
        MX=MX*I
     ENDDO
     DBFACTORAL(N)=MX
  ENDDO
  !
  ! clear the properties only when none exist...
  IF(NPRSHP.EQ.0) THEN
     DO I=1,MAXPRP
        SHPPRPE(I)=1
        NAMSHP(I)=' '
        SHPPRP(1,I)='ZERO'
        SHPPRP(2,I)='NONE'
        SHPPRP(3,I)='NONE'
     ENDDO
  ENDIF
  !
  NUMSHP=0
  NUMSHPR=0
  NUMESH=0
  !
  RETURN
END SUBROUTINE SETUPSHP

SUBROUTINE MKSHPI(NORD,IPINDX)
  !
  ! Create shape index array for pointing into descriptors
  !
  use number
  use stream

  INTEGER NORD
  INTEGER IPINDX(*)
  !
  INTEGER NP,NP2,IPT,IS,JS,KS,IP
  !
  NP=NORD+1
  NP2=NP*NP
  !
  IPT=0
  DO IS=0,NORD
     DO JS=0,NORD
        DO KS=0,NORD
           IP=IS+JS*NP+KS*NP2+1
           IF(IS+JS+KS.LE.NORD) THEN
              IPT=IPT+1
              IPINDX(IP)=IPT
           ELSE
              IPINDX(IP)=0
           ENDIF
        ENDDO
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE MKSHPI

SUBROUTINE GETSHPNM(COMLYN,COMLEN,ISHP,SNAME,QWARN)
  !
  !  Parse the selection of a shape factor
  !
  use string

  CHARACTER(len=*) COMLYN
  INTEGER COMLEN,ISHP
  CHARACTER(len=*) SNAME
  LOGICAL QWARN
  !
  INTEGER IS,I
  !
  SNAME=NEXTA8(COMLYN,COMLEN)
  IS=0
  IF(SNAME.EQ.'ALL') IS=-1
  DO I=1,NUMSHP
     IF(NAMSHP(I).EQ.SNAME) THEN
        IF(IS.EQ.-1) CALL WRNDIE(-2,'<GETSHPNM>','ALL is reserved')
        IS=I
     ENDIF
  ENDDO
  IF(IS.EQ.0 .AND. QWARN) CALL WRNDIE(-2,'<GETSHPNM>', &
       'Could not find specified shape descriptor')
  ISHP=IS
  RETURN
END SUBROUTINE GETSHPNM

SUBROUTINE EDITSHAPE(IS,STYPE,SHPSEL, &
     ISLCT,QSELE,IMOVE,NATOM)
  !
  ! Edit (or add a new) descriptor.
  !
  !     By Bernard R. Brooks   February, 1998
  !
  use number
  use stream

  INTEGER IS
  CHARACTER(len=4) STYPE
  INTEGER NATOM
  INTEGER SHPSEL(NATOM),ISLCT(NATOM),IMOVE(NATOM)
  LOGICAL QSELE,NOTFREE
  !
  !
  INTEGER I,J,JS,KS
  LOGICAL QOLD,QNEW
  !
  QOLD=.FALSE.
  QNEW=.FALSE.
  IF(SHPTYP(IS).EQ.'FLEX' .OR. SHPTYP(IS).EQ.'RIGI') QOLD=.TRUE.
  IF(SHPTYP(IS).EQ.'RIGI') NUMSHPR=NUMSHPR-1
  IF(STYPE.EQ.'FLEX' .OR. STYPE.EQ.'RIGI') QNEW=.TRUE.
  KS=IS
  IF(STYPE.EQ.'FLEX') KS=-IS
  !
  IF (STYPE.EQ.'STAT' .OR. STYPE.EQ.'FLEX' .OR. &
       STYPE.EQ.'RIGI' .OR. STYPE.EQ.'NONE') THEN
     SHPTYP(IS)=STYPE
  ELSE IF(STYPE.NE.' ') THEN
     CALL WRNDIE(-2,'<EDITSHAPE>','Bad shape type specified')
     RETURN
  ENDIF
  IF(SHPTYP(IS).EQ.'RIGI') NUMSHPR=NUMSHPR+1
  !
  ! Remove the old atom selection for active shapes
  IF(QOLD) THEN
     DO I=1,NATOM
        IF(IABS(SHPSEL(I)).EQ.IS) SHPSEL(I)=0
     ENDDO
  ENDIF
  !
  ! Add the new selection to the shape atom array
  NOTFREE=.FALSE.
  IF(QNEW) THEN
     SHPNAT(IS)=0
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
           IF(SHPSEL(I).NE.0) THEN
              !               remove this atom from previous shape
              !               make previous shape's atom list inactive
              JS=IABS(SHPSEL(I))
              IF(SHPNAT(JS).GT.0) SHPNAT(JS)=-SHPNAT(JS)
              IF(PRNLEV.GE.3) WRITE(OUTU,45) -SHPNAT(JS),NAMSHP(JS)
              DO J=I,NATOM
                 IF(IABS(SHPSEL(J)).EQ.JS) SHPSEL(J)=0
              ENDDO
           ENDIF
           SHPSEL(I)=KS
           SHPNAT(IS)=SHPNAT(IS)+1
           IF(STYPE.EQ.'RIGI') THEN
              IF(IMOVE(I).NE.0) THEN
                 NOTFREE=.TRUE.
              ENDIF
           ENDIF
        ENDIF
     ENDDO
45   FORMAT('WARNING: Deselecting',I6,' atoms from descriptor: ',A)
  ENDIF
  !
  IF(NOTFREE) CALL WRNDIE(-2,'<EDITSHAPE>', &
       'Fixed atom or lonepair in rigid shape.  Expect problems..')
  !
  RETURN
END SUBROUTINE EDITSHAPE

SUBROUTINE DELSHAPE(ISHAPE,SHPSEL,NATOM)
  !
  ! DELETE a descriptor
  !
  !     By Bernard R. Brooks   February, 1998
  !
  use memory
  use number
  use stream

  INTEGER ISHAPE,NATOM
  INTEGER SHPSEL(NATOM)
  !
  !
  INTEGER I,IS
  !
  IS=ISHAPE  ! must make a copy
  DO I=1,NATOM
     IF(IABS(SHPSEL(I)).EQ.IS) SHPSEL(I)=0
     IF(SHPSEL(I).GT.IS) SHPSEL(I)=SHPSEL(I)-1
     IF(SHPSEL(I).LT.-IS) SHPSEL(I)=SHPSEL(I)+1
  ENDDO
  !
  call chmdealloc('shapes.src','DELSHAPE','PTRSHP(IS)',LENSHP,MAXPRP,crlp=PTRSHP(IS)%a)
  NUMSHP=NUMSHP-1
  IF(SHPTYP(IS).EQ.'RIGI') NUMSHPR=NUMSHPR-1

  DO I=IS,NUMSHP
     PTRSHP(I)=PTRSHP(I+1)
     NAMSHP(I)=NAMSHP(I+1)
     SHPNAT(I)=SHPNAT(I+1)
     SHPTYP(I)=SHPTYP(I+1)
  ENDDO
  !
  RETURN
END SUBROUTINE DELSHAPE

SUBROUTINE SHPWGH(COMLYN,COMLEN,NORD,WEIGHT, &
     LENSHP,NPROP,IPINDX)
  !
  ! This routine parses the specification of the weighting array
  !
  !   SHAPe WEIGht [ PRINt ]  -
  !
  !    repeat( { [ ELEMent  element-spec ] } [ ORDEr int ] [ SCALe ] real )
  !
  !    element-spec::=  iiii
  !       i ::= { *     }
  !             { digit }
  !
  !     The elements are: x exponent, y exponent, z exponent, and prop#
  !
  !     (element-spec examples:  020*  ***1  1101  1002  **21  )
  !
  !     By Bernard R. Brooks   December 5, 1995
  !
  use number
  use stream
  use string

  INTEGER COMLEN
  CHARACTER(len=*) COMLYN
  INTEGER NORD,IPINDX(*),LENSHP,NPROP
  real(chm_real) WEIGHT(LENSHP,NPROP)
  !
  INTEGER II,IS,JS,KS,LS,I,IORD,IPT,JORD,MODE,IT,JT,KT,LT,IO
  real(chm_real) VAL
  LOGICAL QSCALE
  CHARACTER(len=4) WRD,WSET
  !
100 CONTINUE
  CALL TRIMA(COMLYN,COMLEN)
  IF(COMLEN.EQ.0) RETURN
  !
  MODE=0
  QSCALE=.FALSE.
  IO=-1
  IT=-1
  JT=-1
  KT=-1
  LT=-1
  WSET='****'
  !
200 CONTINUE
  WRD=CURRA4(COMLYN,COMLEN)

  IF(WRD.EQ.'ELEM') THEN
     MODE=1
     WRD=NEXTA4(COMLYN,COMLEN)
     WRD=NEXTA4(COMLYN,COMLEN)
     IF(WRD(1:1).NE.'*') READ(WRD(1:1),'(I1)') IT
     IF(WRD(2:2).NE.'*') READ(WRD(2:2),'(I1)') JT
     IF(WRD(3:3).NE.'*') READ(WRD(3:3),'(I1)') KT
     IF(WRD(4:4).NE.'*') READ(WRD(4:4),'(I1)') LT
     WSET=WRD
  ELSE IF(WRD.EQ.'ORDE') THEN
     MODE=1
     WRD=NEXTA4(COMLYN,COMLEN)
     IO=NEXTI(COMLYN,COMLEN)
  ELSE IF(WRD.EQ.'SCAL') THEN
     QSCALE=.TRUE.
     WRD=NEXTA4(COMLYN,COMLEN)
  ELSE
     IF(MODE.EQ.0) THEN
        CALL WRNDIE(0,'<SHPWGH>','Error in parsing WEIGht command')
        RETURN
     ELSE
        VAL=NEXTF(COMLYN,COMLEN)
        MODE=-MODE
     ENDIF
  ENDIF
  !
  IF(MODE.GE.0) GOTO 200
  !
  ! for each element, see if it should be modified
  DO LS=1,NPROP
     IPT=0
     DO IS=0,NORD
        DO JS=0,NORD
           DO KS=0,NORD
              IORD=IS+JS+KS
              IF(IORD.LE.NORD) THEN
                 IPT=IPT+1
                 !        check to see if the order is correct
                 IF(IORD.EQ.IO .OR. IO.LT.0) THEN  ! order OK
                    IF(IT.LT.0 .OR. IS.EQ.IT) THEN
                       IF(JT.LT.0 .OR. JS.EQ.JT) THEN
                          IF(KT.LT.0 .OR. KS.EQ.KT) THEN
                             IF(LT.LT.0 .OR. LS.EQ.LT) THEN
                                IF(QSCALE) THEN
                                   WEIGHT(IPT,LS)=WEIGHT(IPT,LS)*VAL
                                ELSE
                                   WEIGHT(IPT,LS)=VAL
                                ENDIF
                                IF(PRNLEV.GT.5) &
                                     WRITE(OUTU,25) IS,JS,KS,LS,WEIGHT(IPT,LS)
25                              FORMAT('SHAPES: Weight for:',4I2,' set to',F15.6)
                             ENDIF
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
              ENDIF
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  !
  ! loop back to parse next set
  GOTO 100
  !
END SUBROUTINE SHPWGH

SUBROUTINE PRNSHP(UNIT,NORD,SHNAM,SHAPEF,LENSHP,NPROP,NAMPRP, &
     SHPSEL,NAT,NSEL,MODE,IEXP,PDAT,PORD)
  !
  ! This routine prints the values of a shape descriptor
  !             (or weighting array)
  !
  !     By Bernard R. Brooks   April, 1995
  !
  use dimens_fcm
  use number
  use select
  use psf
  use stream
  use machutil,only:die

  INTEGER UNIT,NORD
  CHARACTER(len=*) SHNAM
  INTEGER LENSHP,NPROP,NAT,NSEL
  real(chm_real) SHAPEF(LENSHP,NPROP)
  CHARACTER(len=*) NAMPRP(NPROP)
  INTEGER SHPSEL(NAT)
  INTEGER MODE,IEXP(NPROP)
  CHARACTER(len=4) PDAT(3,NPROP)
  INTEGER PORD
  !
  INTEGER II,IS,JS,KS,LS,I,IORD,IPT,JORD
  real(chm_real) W,WTOT,VALI,VALJ,VALK,VALL
  !
  IF(PRNLEV.LT.3) RETURN
  !
  IF(MODE.NE.0) THEN
     IF(NATOM.NE.NAT) CALL DIE ! Bad coding if here
     II=NSELCTV(NATOM,SHPSEL,MODE)
     IF(II.EQ.0 .AND. NSEL.EQ.0) THEN
        WRITE(OUTU,700)
700     FORMAT(' No atoms belong to this shape')
     ELSE IF(II.EQ.NSEL) THEN
        WRITE(OUTU,701) II
701     FORMAT(' The number atoms belong to this shape:',I5)
        CALL PRNTATSL(SHPSEL,NATOM,MODE,RESID,RES,IBASE,ATYPE, &
             SEGID,NICTOT,NSEG)
     ELSE IF(II.GE.0 .AND. NSEL.GE.0) THEN
        WRITE(OUTU,702) II,NSEL
702     FORMAT(' The wrong number atoms belong to this shape:',I5)
        CALL PRNTATSL(SHPSEL,NATOM,MODE,RESID,RES,IBASE,ATYPE, &
             SEGID,NICTOT,NSEG)
     ELSE
        WRITE(OUTU,703) -NSEL
703     FORMAT(' Number of inactive atoms belong to this shape:',I5)
     ENDIF
  ENDIF
  !
  WRITE(UNIT,24) NAMPRP
24 FORMAT(/' Properties:        ',10A15)
  WRITE(UNIT,25) (PDAT(1,I),IEXP(I),I=1,NPROP)
25 FORMAT(' Array and exponent:',7X,10(A4,I4,7X))
  WRITE(UNIT,26) (PDAT(2,I),PDAT(3,I),I=1,NPROP)
26 FORMAT(' Other properties:  ',7X,10(A4,1X,A4,6X))
  WRITE(UNIT,27)
  !
  DO JORD=0,PORD
     IPT=0
     DO IS=0,NORD
        DO JS=0,NORD
           DO KS=0,NORD
              IORD=IS+JS+KS
              IF(IORD.LE.NORD) THEN
                 IPT=IPT+1
                 IF(IORD.EQ.JORD) THEN
                    WRITE(UNIT,27) SHNAM,IS,JS,KS, &
                         (SHAPEF(IPT,LS),LS=1,NPROP)
27                  FORMAT(A11,3I3,10F15.4)
                 ENDIF
              ENDIF
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE PRNSHP

SUBROUTINE SHPWRI(UNIT)
  !
  ! This routine writes a shape descriptor file
  !
  !     By Bernard R. Brooks   April, 1995
  !
  use number
  use ctitla
  use stream

  INTEGER UNIT
  !
  INTEGER I
  !
  IF(IOLEV.GT.0) THEN
     CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
     CALL WRTITL(TITLEA,NTITLA,UNIT,0)
     WRITE(UNIT,22) 'INTEGERS: NUMSHP,ORDSHP,LENSHP,NPRSHP'
     WRITE(UNIT,28)  NUMSHP,ORDSHP,LENSHP,NPRSHP
28   FORMAT(16I5)
     WRITE(UNIT,22) 'PROPERTIES: I,NAMSHP(I),SHPPRP(*,I),SHPPRPE(I)'
     DO I=1,NPRSHP
        WRITE(UNIT,32) I,NAMPRP(I),SHPPRP(1,I),SHPPRP(2,I), &
             SHPPRP(3,I),SHPPRPE(I)
32      FORMAT(I3,2X,A8,2X,A4,2X,A4,2X,A4,I4)
     ENDDO
     WRITE(UNIT,22) 'WEIGHTING: WGHSHP'
     CALL WRARRAY(WGHSHP,LENSHP,NPRSHP,UNIT,.TRUE.)
     WRITE(UNIT,22) 'DESCRIPTORS: I,NAMSHP(I),SHPTYP(I)'
     DO I=1,NUMSHP
        WRITE(UNIT,35) I,NAMSHP(I),SHPTYP(I)
35      FORMAT(I5,5X,A,2X,A4)
        CALL WRARRAY(PTRSHP(I)%a,LENSHP,NPRSHP,UNIT,.TRUE.)
     ENDDO
  ENDIF
  RETURN
22 FORMAT(A)
END SUBROUTINE SHPWRI

SUBROUTINE SHPREA(UNIT,QAPPE)
  !
  ! This routine reads a shape descriptor file
  !
  !     By Bernard R. Brooks   April, 1995
  !
  use memory
  use number
  use ctitla
  use stream

  INTEGER UNIT
  LOGICAL QAPPE, OK
  !
  INTEGER I,NSH,OSH,LSH,NPR,IT,J,PRPE
  CHARACTER(len=MNAMSH) WORD
  CHARACTER(len=4) PRP1,PRP2,PRP3,ATYPEI
  real(chm_real),pointer,dimension(:,:):: I_ptr
  !
  IF(IOLEV.GT.0) THEN
     !
     IF(NUMSHP.EQ.0) QAPPE=.FALSE.
     IF(.NOT.QAPPE) CALL FREESHP(.TRUE.)
     !
     CALL RDTITL(TITLEB,NTITLB,UNIT,0)
     CALL WRTITL(TITLEB,NTITLB,OUTU,1)
     !
     READ(UNIT,28)
     READ(UNIT,28) NSH,OSH,LSH,NPR
28   FORMAT(16I5)
     !
     IF(QAPPE) THEN
        ! Append the new descriptors to the existing set.
        IF(OSH.NE.ORDSHP) THEN
           !         the order is different
           CALL WRNDIE(-3,'<SHPREA>', &
                'Shape order does not match existing descriptors')
           RETURN
        ENDIF
        IF(PRNLEV.GT.2) WRITE(OUTU,42)
42      FORMAT(' SHPREA: The weighting array is replaced.')
     ELSE
        CALL SETUPSHP(OSH)
     ENDIF
     !
     ! Read the properties
     READ(UNIT,28)
     IF(QAPPE) THEN
        OK=.TRUE.
        IF(NPRSHP.NE.NPR) OK=.FALSE.
        DO I=1,NPR
           READ(UNIT,32) WORD,PRP1,PRP2,PRP3,PRPE
           IF(NAMPRP(I)  .NE.WORD) OK=.FALSE.
           IF(SHPPRP(1,I).NE.PRP1) OK=.FALSE.
           IF(SHPPRP(2,I).NE.PRP2) OK=.FALSE.
           IF(SHPPRP(3,I).NE.PRP3) OK=.FALSE.
           IF(SHPPRPE(I) .NE.PRPE) OK=.FALSE.
           IF(.NOT.OK) THEN
              WRITE(OUTU,31) 'Found:    ',WORD,PRP1,PRP2,PRP3,PRPE
              WRITE(OUTU,31) 'Expected: ',NAMPRP(I),SHPPRP(1,I), &
                   SHPPRP(2,I),SHPPRP(3,I),SHPPRPE(I)
31            FORMAT(A,A8,2X,A4,2X,A4,2X,A4,I4)
           ENDIF
        ENDDO
        IF(.NOT.OK) THEN
           CALL WRNDIE(-3,'<SHPREA>', &
                'Shape properties do not match existing descriptors')
        ENDIF
     ELSE
        NPRSHP=NPR
        DO I=1,NPR
           READ(UNIT,32) NAMPRP(I),SHPPRP(1,I),SHPPRP(2,I), &
                SHPPRP(2,I),SHPPRPE(I)
32         FORMAT(5X,A8,2X,A4,2X,A4,2X,A4,I4)
        ENDDO
     ENDIF
     !
     ! Read the weighting array
     READ(UNIT,28)
     IF(QAPPE) THEN
        !         ignore waiting array on read append
        call chmalloc('shapes.src','SHPREA','I_ptr',LSH,NPR,crlp=I_ptr)
        call RDARRAY(I_ptr,LSH,NPR,UNIT,.TRUE.)
        call chmdealloc('shapes.src','SHPREA','I_ptr',LSH,NPR,crlp=I_ptr)
     ELSE
        call RDARRAY(WGHSHP,LENSHP,NPRSHP,UNIT,.TRUE.)
     ENDIF
     !
     ! Read a new set of descriptors.
     !
     READ(UNIT,28)
     DO I=1,NSH
        READ(UNIT,35) WORD,ATYPEI
        IT=-1
        DO J=1,NUMSHP
           IF(WORD(1:LSH).EQ.NAMSHP(J)) IT=J
        ENDDO
        IF(IT.GT.0) THEN
           IF(WRNLEV.GT.3) WRITE(OUTU,49) WORD(1:LSH)
49         FORMAT(' WARNING:: SHPREA:: Duplicate shape descriptor', &
                ' named "',A,'" is replaced.')
           IF(SHPTYP(IT).EQ.'RIGI') NUMSHPR=NUMSHPR-1
        ELSE
           NUMSHP=NUMSHP+1
           IT=NUMSHP

           call chmalloc('shapes.src','SHPREA','PTRSHP(NUMSHP)', &
                LENSHP,MAXPRP,crlp=PTRSHP(NUMSHP)%a)
           PTRSHP(NUMSHP)%a(1:LENSHP,1:MAXPRP)=zero
        ENDIF
        !
        NAMSHP(IT)=WORD(1:LSH)
        IF (ATYPEI == 'RIGI' .OR. ATYPEI == 'FLEX') ATYPEI='NONE'
        SHPTYP(I)=ATYPEI
35      FORMAT(10X,A,2X,A4)
        CALL RDARRAY(PTRSHP(IT)%a,LENSHP,NPRSHP,UNIT,.TRUE.)
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE SHPREA

SUBROUTINE WRARRAY(ARRAY,LENGTH,COUNT,UNIT,QCARD)
  !
  ! This routine writes a real(chm_real) array to UNIT
  !
  INTEGER LENGTH,UNIT,COUNT
  real(chm_real) ARRAY(LENGTH,COUNT)
  LOGICAL QCARD
  !
  IF(QCARD) THEN
     WRITE(UNIT,55) ARRAY
55   FORMAT(5D15.8)
  ELSE
     WRITE(UNIT) ARRAY
  ENDIF
  RETURN
END SUBROUTINE WRARRAY

SUBROUTINE RDARRAY(ARRAY,LENGTH,COUNT,UNIT,QCARD)
  !
  ! This routine reads a real(chm_real) array from UNIT
  !
  INTEGER LENGTH,UNIT,COUNT
  real(chm_real) ARRAY(LENGTH,COUNT)
  LOGICAL QCARD
  !
  IF(QCARD) THEN
     READ(UNIT,55) ARRAY
55   FORMAT(5D15.8)
  ELSE
     READ(UNIT) ARRAY
  ENDIF
  RETURN
END SUBROUTINE RDARRAY


SUBROUTINE SHPFILL(IS,IP,ISX,ISLCT,QPRINT,X,Y,Z,NATOM)
  !
  ! This routine fills a shape descriptor by number
  !
  !     By Bernard R. Brooks   February, 1998
  !
  use memory
  use number
  use dimens_fcm
  use select
  use stream

  INTEGER IS,IP,ISX
  INTEGER NATOM
  INTEGER ISLCT(NATOM)
  LOGICAL QPRINT
  real(chm_real) X(*),Y(*),Z(*)
  !
  INTEGER IPROP,ISHAPE
  LOGICAL LPRINT,ERR,NEEDR

  real(chm_real),allocatable,dimension(:) :: ARRAY
  real(chm_real),allocatable,dimension(:) :: RADIUS
  !
  LPRINT=.FALSE.
  IF(PRNLEV.GT.2) LPRINT=QPRINT
  IF(PRNLEV.GT.6) LPRINT=.TRUE.
  !
  IF(NPRSHP.LE.0 .OR. NUMSHP.LE.0) RETURN
  !
  IF(LPRINT) THEN
     IF(IS.LT.0) THEN
        WRITE(OUTU,22) 'Fill all descriptors'
     ELSE
        WRITE(OUTU,22) 'Fill descriptor named: '//NAMSHP(IS)
     ENDIF
     !
     IF(IP.LT.0) THEN
        WRITE(OUTU,22) 'Fill all properties'
     ELSE
        WRITE(OUTU,22) 'Fill property named: '//NAMPRP(IP)
     ENDIF
  ENDIF
  !
  call chmalloc('shapes.src','SHPFILL','RADIUS',NATOM,crl=RADIUS)
  call chmalloc('shapes.src','SHPFILL','ARRAY',NATOM,crl=ARRAY)
  !
  ! Fill the radius array (only if needed)
  NEEDR=.FALSE.
  DO IPROP=1,NPRSHP
     IF(SHPPRP(2,IPROP).EQ.'HOMO' .OR. SHPPRP(2,IPROP).EQ.'GAUS') &
          NEEDR=.TRUE.
  ENDDO
  IF(NEEDR) THEN
     CALL SELPROP('SCA9',RADIUS,NATOM,ERR)
     IF(ERR) THEN
        CALL WRNDIE(-2,'<SHPCOM>','Radius array "SCA9" is not filled')
        RADIUS(1:NATOM)=zero
     ENDIF
  ENDIF
  !
  ! Fill the elements of a shape
  DO IPROP=1,NPRSHP
     IF(IP.EQ.IPROP .OR. IP.LT.0) THEN
        CALL SELPROP(SHPPRP(1,IPROP),ARRAY,NATOM,ERR)
        IF(ERR) THEN
           CALL WRNDIE(-1,'<SHPCOM>','Bad shape property array name')
           ARRAY(1:NATOM)=zero
        ENDIF
        !
        DO ISHAPE=1,NUMSHP
           IF(IS.EQ.ISHAPE .OR. IS.LT.0) THEN
              IF(ISX.EQ.1) THEN

                 CALL SHPFILL2(ISX,NATOM,ISLCT,X,Y,Z, &
                      RADIUS,ARRAY,ORDSHP, &
                      INDSHP,PTRSHP(ISHAPE)%a, &
                      LENSHP,NPRSHP,IPROP,SHPPRPE(IPROP), &
                      SHPPRP(2,IPROP),SHPPRP(3,IPROP))
              ELSE
                 IF(SHPTYP(ISHAPE).EQ.'FLEX' .OR. &
                      SHPTYP(ISHAPE).EQ.'RIGI') THEN
                    CALL SHPFILL2(ISHAPE,NATOM,SHPATP,X,Y,Z, &
                         RADIUS,ARRAY,ORDSHP, &
                         INDSHP,PTRSHP(ISHAPE)%a , &
                         LENSHP,NPRSHP,IPROP,SHPPRPE(IPROP), &
                         SHPPRP(2,IPROP),SHPPRP(3,IPROP))
                 ENDIF
              ENDIF
           ENDIF
        ENDDO

     ENDIF
  ENDDO
  !
22 FORMAT('SHPFILL: ',2A)
  !

  IF(allocated(RADIUS)) call chmdealloc('shapes.src','SHPFILL','RADIUS',NATOM,crl=RADIUS)
  IF(allocated(ARRAY)) call chmdealloc('shapes.src','SHPFILL','ARRAY',NATOM,crl=ARRAY)

  RETURN
END SUBROUTINE SHPFILL

SUBROUTINE SHPFILL2(ISHAPE,NATOM,SHPSEL,X,Y,Z,RADIUS,ARRAY, &
     NORD,IPINDX,SHAPEF,LENSHP,NPROP,IPROP, &
     IEXP,PFILL,PCENT)
  !
  ! This routine fills a shape descriptor based on simple atom positions.
  !
  !     By Bernard R. Brooks   April, 1995
  !
  use number
  use stream
  use machutil,only:die

  INTEGER ISHAPE,NATOM
  INTEGER SHPSEL(NATOM)
  real(chm_real) X(*),Y(*),Z(*),RADIUS(*)
  real(chm_real) ARRAY(*)
  INTEGER NORD
  INTEGER IPINDX(*)
  INTEGER LENSHP,NPROP
  real(chm_real) SHAPEF(LENSHP,NPROP)
  INTEGER IPROP,IEXP
  CHARACTER(len=4) PFILL,PCENT
  !
  INTEGER IS,JS,KS,I,IP,IORD,IPT,NP,NP2
  real(chm_real) W,WTOT,VALI,VALJ,VALK
  real(chm_real) XCEN,YCEN,ZCEN,XVAL,YVAL,ZVAL
  !
  NP=NORD+1
  NP2=NP*NP
  !
  ! Zero the grid
  IPT=0
  DO IS=0,NORD
     DO JS=0,NORD
        DO KS=0,NORD
           IORD=IS+JS+KS
           IF(IORD.LE.NORD) THEN
              IPT=IPT+1
              SHAPEF(IPT,IPROP)=ZERO
           ENDIF
        ENDDO
     ENDDO
  ENDDO
  !
  W=ZERO
  XCEN=ZERO
  YCEN=ZERO
  ZCEN=ZERO
  IF(PCENT.EQ.'CENT') THEN
     DO I=1,NATOM
        IF(IABS(SHPSEL(I)).EQ.ISHAPE) THEN
           VALI=ARRAY(I)**IEXP
           W=W + VALI
           XCEN=XCEN+X(I)*VALI
           YCEN=YCEN+Y(I)*VALI
           ZCEN=ZCEN+Z(I)*VALI
        ENDIF
     ENDDO
     IF(ABS(W).GT.TENM5) THEN
        W=ONE/W
        XCEN=XCEN*W
        YCEN=YCEN*W
        ZCEN=ZCEN*W
     ELSE
        PCENT='NONE'
        XCEN=ZERO
        YCEN=ZERO
        ZCEN=ZERO
     ENDIF
  ENDIF
  !
  ! Fill the grid
  DO I=1,NATOM
     IF(IABS(SHPSEL(I)).EQ.ISHAPE) THEN
        !             calculate the property value for this atom.
        IPT=0
        VALI=ARRAY(I)**IEXP
        XVAL=X(I)-XCEN
        YVAL=Y(I)-YCEN
        ZVAL=Z(I)-ZCEN
        !
        IF(PFILL.EQ.'GRID') THEN
           CALL DIE  ! not yet coded...
        ELSE IF(PFILL.EQ.'NONE') THEN
           CONTINUE  ! do nothing
        ELSE IF(PFILL.EQ.'POIN') THEN
           DO IS=0,NORD
              VALJ=VALI
              DO JS=0,NORD
                 VALK=VALJ
                 DO KS=0,NORD
                    IORD=IS+JS+KS
                    IF(IORD.LE.NORD) THEN
                       IPT=IPT+1
                       SHAPEF(IPT,IPROP)=SHAPEF(IPT,IPROP)+VALK
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
                       CALL SHAPEMOMENT(PFILL,RADIUS(I),XVAL,YVAL, &
                            ZVAL,IS,JS,KS,VALK, &
                            .FALSE.,ZERO,ZERO,ZERO)
                       SHAPEF(IPT,IPROP)=SHAPEF(IPT,IPROP)+VALK*VALI
                    ENDIF
                 ENDDO
              ENDDO
           ENDDO
        ENDIF
     ENDIF
  ENDDO
  !
  ! Normalize the shape array for either the center or normalize subtype
  IF(PCENT.EQ.'CENT' .OR. PCENT.EQ.'NORM') THEN
     W = SHAPEF(1,IPROP)
     IF(ABS(W).GT.TENM5) THEN
        W=ONE/W
        DO I=2,LENSHP
           SHAPEF(I,IPROP)=SHAPEF(I,IPROP)*W
        ENDDO
     ELSE
        PCENT='NONE'
     ENDIF
     IF(PCENT.EQ.'CENT') THEN
        IP=1+1          ! X  INDEX
        SHAPEF(IPINDX(IP),IPROP)=XCEN
        IP=1+NP         ! Y  INDEX
        SHAPEF(IPINDX(IP),IPROP)=YCEN
        IP=1+NP2        ! Z  INDEX
        SHAPEF(IPINDX(IP),IPROP)=ZCEN
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE SHPFILL2

SUBROUTINE SHAPEMOMENT(STYPE,RADIUS,X,Y,Z,IS,JS,KS,SHV, &
     QFORCE,SHDX,SHDY,SHDZ)
  !
  ! This routine computes a shape moment based on simple atom positions.
  !
  !   X,Y,Z      coordinates of the atom
  !   IS,JS,KS   index of moment
  !   STYPE      index of type of distribution
  !                'GAUS'    gaussian distribution
  !                'HOMO'    spherical distribution
  !                'POIN'**  point particles
  !                'GRID'**  grid based fill
  !                'NONE'**  do not fill
  !                   (**::= not done here in this routine)
  !   RADIUS     parameter for distribution
  !   SHV        moment
  !
  !     By Yuhong Zhang   April, 1996
  !
  use number

  CHARACTER(len=4) STYPE
  real(chm_real)  RADIUS,X,Y,Z
  INTEGER IS,JS,KS
  real(chm_real)  SHV
  LOGICAL QFORCE
  real(chm_real)  SHDX,SHDY,SHDZ
  !
  !
  INTEGER IX,IY,IZ
  real(chm_real)  CX,CY,CZ,CV
  real(chm_real)  DX,DY,DZ
  !
  SHV=ZERO
  IF(QFORCE) THEN
     SHDX=ZERO
     SHDY=ZERO
     SHDZ=ZERO
  ENDIF
  !
  IF(RADIUS.LT.TENM5) THEN
     IF(IS.GT.1) THEN
        DX=IS*X**(IS-1)
        CX=X**IS
     ELSE IF(IS.EQ.1) THEN
        DX=ONE
        CX=X
     ELSE
        DX=ZERO
        CX=ONE
     ENDIF
     IF(JS.GT.1) THEN
        DY=JS*Y**(JS-1)
        CY=Y**JS
     ELSE IF(JS.EQ.1) THEN
        DY=ONE
        CY=Y
     ELSE
        DY=ZERO
        CY=ONE
     ENDIF
     IF(KS.GT.1) THEN
        DZ=KS*Z**(KS-1)
        CZ=Z**KS
     ELSE IF(KS.EQ.1) THEN
        DZ=ONE
        CZ=Z
     ELSE
        DZ=ZERO
        CZ=ONE
     ENDIF
     SHV =CX*CY*CZ
     IF(QFORCE) THEN
        SHDX=DX*CY*CZ
        SHDY=CX*DY*CZ
        SHDZ=CX*CY*DZ
     ENDIF
  ELSE
     DO IX=0,IS,2
        !
        CX=RADIUS**IX
        IF(IX.EQ.IS) THEN
           DX=ZERO
        ELSE IF(IS-IX.EQ.1) THEN
           CX=CX*IS
           DX=CX
           CX=CX*X
        ELSE IF(IS-IX.EQ.2) THEN
           CX=CX*X*IS*(IS-1)*HALF
           DX=CX*2
           CX=CX*X
        ELSE
           CX=CX*FACTORAL(IS)/FACTORAL(IX)/FACTORAL(IS-IX)
           DX=CX*(IS-IX)*X**(IS-IX-1)
           CX=CX*X**(IS-IX)
        ENDIF
        !
        DO IY=0,JS,2
           !
           CY=RADIUS**IY
           IF(IY.EQ.JS) THEN
              DY=ZERO
           ELSE IF(JS-IY.EQ.1) THEN
              CY=CY*JS
              DY=CY
              CY=CY*Y
           ELSE IF(JS-IY.EQ.2) THEN
              CY=CY*Y*JS*(JS-1)*HALF
              DY=CY*2
              CY=CY*Y
           ELSE
              CY=CY*FACTORAL(JS)/FACTORAL(IY)/FACTORAL(JS-IY)
              DY=CY*(JS-IY)*Y**(JS-IY-1)
              CY=CY*Y**(JS-IY)
           ENDIF
           !
           DO IZ=0,KS,2
              !
              CZ=RADIUS**IZ
              IF(IZ.EQ.KS) THEN
                 DZ=ZERO
              ELSE IF(KS-IZ.EQ.1) THEN
                 CZ=CZ*KS
                 DZ=CZ
                 CZ=CZ*Z
              ELSE IF(KS-IZ.EQ.2) THEN
                 CZ=CZ*Z*KS*(KS-1)*HALF
                 DZ=CZ*2
                 CZ=CZ*Z
              ELSE
                 CZ=CZ*FACTORAL(KS)/FACTORAL(IZ)/FACTORAL(KS-IZ)
                 DZ=CZ*(KS-IZ)*Z**(KS-IZ-1)
                 CZ=CZ*Z**(KS-IZ)
              ENDIF
              !
              CALL VOLUMINTEG(STYPE,IX,IY,IZ,CV)
              SHV=SHV+CX*CY*CZ*CV
              IF(QFORCE) THEN
                 SHDX=SHDX+DX*CY*CZ*CV
                 SHDY=SHDY+CX*DY*CZ*CV
                 SHDZ=SHDZ+CX*CY*DZ*CV
              ENDIF
              !
              !
#if KEY_DEBUG==1
              write(6,123) IS,JS,KS,SHV,CX,CY,CZ,CV,DX,DY,DZ

              write(6,124) IX,radius**ix,X**(IS-IX),FACTORAL(IS), &
                   FACTORAL(IX),FACTORAL(IS-IX)
              write(6,124) IY,radius**iY,Y**(JS-IY),FACTORAL(JS), &
                   FACTORAL(IY),FACTORAL(JS-IY)
              write(6,124) IZ,radius**iZ,Z**(KS-IZ),FACTORAL(KS), &
                   FACTORAL(IZ),FACTORAL(KS-IZ)
123           format(3I3,3X,9E12.4)
124           format(9X,I3,6E12.4)
#endif 
              !
           ENDDO
        ENDDO
     ENDDO
  ENDIF

  RETURN
END SUBROUTINE SHAPEMOMENT

SUBROUTINE VOLUMINTEG(STYPE,IX,IY,IZ,CV)
  !
  ! This routine computes volume integration
  !
  !     By Yuhong Zhang   April, 1996
  !
  use number
  use machutil,only:die

  CHARACTER(len=4) STYPE
  INTEGER IX,IY,IZ
  real(chm_real)  CV
  !
  !
  INTEGER K
  real(chm_real)  CR,CT,CP,C1,C2
  !
  !-----------------------------------------------------------------------
  ! Gaussian convolution
  IF (STYPE == 'GAUS') THEN
     CV=DBFACTORAL(IX-1)*DBFACTORAL(IY-1)*DBFACTORAL(IZ-1)
     !
     !      write(6,123) IX,IY,IZ,CV,DBFACTORAL(IX-1),DBFACTORAL(IY-1),
     !     &                         DBFACTORAL(IZ-1)
     ! 123  format(3I3,6E12.4)
     !
     !-----------------------------------------------------------------------
     ! Uniform spherical distribution
  ELSE IF (STYPE == 'HOMO') THEN
     CR=THREE/(IX+IY+IZ+3)
     CT=ZERO
     DO K=0,(IX+IY)/2
        C1=FACTORAL(IX/2+IY/2)/FACTORAL(K)/FACTORAL(IX/2+IY/2-K)
        C2=ONE/(IZ+2*K+1)
        IF( INT(K/2)*2 .EQ. K ) THEN
           CT=CT+C1*C2
        ELSE
           CT=CT-C1*C2
        ENDIF
     ENDDO
     CP=ZERO
     DO K=0,IY/2
        C1=FACTORAL(IY/2)/FACTORAL(K)/FACTORAL(IY/2-K)
        C2=DBFACTORAL(IX+2*K-1)/DBFACTORAL(IX+2*K)
        IF( INT(K/2)*2 .EQ. K ) THEN
           CP=CP+C1*C2
        ELSE
           CP=CP-C1*C2
        ENDIF
     ENDDO
     CV=CR*CT*CP
#if KEY_DEBUG==1
     WRITE(6,85) IX,IY,IZ,CR,CT,CP,CV
85   FORMAT(3I3,4F10.5)
#endif 
     !-----------------------------------------------------------------------
  ELSE
     CALL DIE ! bad shape type
     CV=ZERO
  ENDIF
  !
  RETURN
END SUBROUTINE VOLUMINTEG

SUBROUTINE SHAPEROT(NORD,LENSHP,NPROP,SHAPEI,SHAPEF, &
     ROTM,IPINDX,ISHAPE,STYPE,SHPSEL,X,Y,Z,NAT, &
     SHPPRP)
  !
  ! This routine rotates/translates a shape descriptor.
  !
  !  NORD    - Order of shape factor
  !  LENSHP  - Length of the shape vectors
  !  NPROP   - Number of properties
  !  SHAPEI  - Original shape factor
  !  SHAPEF  - New shape factor
  !  ROTM    - The trans/rot matrix (3,4),first 3x3 as rot,(*,4) as trans
  !  IPINDX  - shape factor pointer index
  !  ISHAPE  - Current shape number
  !  STYPE   - Type of shape
  !  SHPSEL  - Shape selection array
  !  X,Y,Z   - Current coordinates
  !  NAT     - Number of atoms
  !  SHPPRP  - Centering option for each property
  !
  !     By Bernard R. Brooks   November 9, 1994
  !
  use number
  use stream

  INTEGER NORD,LENSHP,NPROP
  real(chm_real) SHAPEI(LENSHP,NPROP),SHAPEF(LENSHP,NPROP)
  real(chm_real) ROTM(3,4)
  INTEGER IPINDX(*)
  INTEGER ISHAPE,NAT
  CHARACTER(len=4) STYPE
  INTEGER SHPSEL(NAT)
  real(chm_real) X(NAT),Y(NAT),Z(NAT)
  CHARACTER(len=4) SHPPRP(3,NPROP)
  !
  INTEGER, PARAMETER :: MAXORD=6
  INTEGER ISINDX(0:MAXORD * 3),JLEV(0:MAXORD)
  INTEGER IS,JS,KS,NI,NJ,NK,I,J,IORD,JORD,ILEV,IP,JP,IPROP
  INTEGER NP,NP2,NP3
  real(chm_real) S,T,SCALEF,XN,YN,ZN
  real(chm_real) ROTMX(3,4)
  real(chm_real) XCEN,YCEN,ZCEN
  CHARACTER(len=4) PCENT
  !
  NP=NORD+1
  NP2=NP*NP
  NP3=NP2*NP
  !
#if KEY_DEBUG==1
  IP=1+NP*(1+NP*(1+NP))
  write(OUTU,45) nord
45 format(' nord=',i5)
  write(OUTU,47) rotm
  write(OUTU,46) (ipindx(i),i=1,ip)
46 format(20I4)
  write(OUTU,47) (shapei(i,1),i=1,5)
47 format(10F12.5)
#endif 
  !
  DO IPROP=1,NPROP
     !
     DO I=1,4
        DO J=1,3
           ROTMX(J,I)=ROTM(J,I)
        ENDDO
     ENDDO
     PCENT=SHPPRP(3,IPROP)
     IF(PCENT.EQ.'CENT' .OR. PCENT.EQ.'NORM') THEN
        SCALEF=SHAPEI(1,IPROP)
        SHAPEI(1,IPROP)=ONE
     ENDIF
     IF(PCENT.EQ.'CENT') THEN
        DO J=1,3
           ROTMX(J,4)=ZERO
        ENDDO
        IP=1+1          ! X  INDEX
        XCEN=SHAPEI(IPINDX(IP),IPROP)
        SHAPEI(IPINDX(IP),IPROP)=ZERO
        IP=1+NP         ! Y  INDEX
        YCEN=SHAPEI(IPINDX(IP),IPROP)
        SHAPEI(IPINDX(IP),IPROP)=ZERO
        IP=1+NP2        ! Z  INDEX
        ZCEN=SHAPEI(IPINDX(IP),IPROP)
        SHAPEI(IPINDX(IP),IPROP)=ZERO
     ENDIF
     !
     DO IS=0,NORD
        ISINDX(IS)=1
        DO JS=0,NORD
           IF(JS.GT.0) ISINDX(IS+JS)=2
           DO KS=0,NORD
              JORD=IS+JS+KS-1
              IF(KS.GT.0) ISINDX(IS+JS+KS)=3
              IORD=IS+JS+KS
              IF(IORD.LE.NORD) THEN
                 S=ZERO
                 DO I=0,JORD
                    JLEV(I)=1
                 ENDDO
200              CONTINUE
                 !
                 NI=0
                 NJ=0
                 NK=0
                 T=ONE
                 DO I=0,JORD
                    IF(JLEV(I).EQ.1) NI=NI+1
                    IF(JLEV(I).EQ.2) NJ=NJ+1
                    IF(JLEV(I).EQ.3) NK=NK+1
                    T=T*ROTMX(ISINDX(I+1),JLEV(I))
                 ENDDO
                 IP=NI+NJ*NP+NK*NP2+1
                 IF(IP.GT.NP3 .OR. IP.LE.0) then
                    write(OUTU, *) 'Bad IP pointer'
                    write (OUTU, *) is, js, ks, iord, jord, nord
                    write (OUTU, *) NI,NJ,NP,NK,NP2,NP3
                    write (OUTU, *) 'ip = ', ip, ' np3 = ', np3
                    stop 'bad IP add pointer'
                 end if

                 JP=IPINDX(IP)
                 IF(JP.GT.LENSHP .OR. JP.LE.0) STOP 'BAD JP ADD POINTER'
                 S=S+T*SHAPEI(JP,IPROP)
                 !
#if KEY_DEBUG==1
                 write(OUTU,145) is,js,ks,ni,nj,nk,ip, &
                      ipindx(ip),T,SHAPEI(IPINDX(IP),IPROP),S
145              format('element:',3I3,4x,3i3,4x,2i6,3F12.5)
#endif 
                 !
                 ILEV=0
300              CONTINUE
                 JLEV(ILEV)=JLEV(ILEV)+1
                 IF(JLEV(ILEV).GT.4) THEN
                    JLEV(ILEV)=1
                    ILEV=ILEV+1
                    IF(ILEV.LE.JORD) GOTO 300
                 ENDIF
                 IF(ILEV.LE.JORD) GOTO 200
                 IP=IS+JS*NP+KS*NP2+1
                 IF(IP.GT.NP3 .OR. IP.LE.0) STOP 'BAD IP SET POINTER'
                 JP=IPINDX(IP)
                 IF(JP.GT.LENSHP .OR. JP.LE.0) STOP 'BAD JP SET POINTER'
                 SHAPEF(JP,IPROP)=S
              ENDIF
           ENDDO
        ENDDO
     ENDDO
     !
     IF(PCENT.EQ.'CENT' .OR. PCENT.EQ.'NORM') THEN
        SHAPEF(1,IPROP)=SCALEF
     ENDIF
     IF(PCENT.EQ.'CENT') THEN
        XN=ROTMX(1,1)*XCEN+ROTMX(1,2)*YCEN+ROTMX(1,3)*ZCEN+ROTM(1,4)
        YN=ROTMX(2,1)*XCEN+ROTMX(2,2)*YCEN+ROTMX(2,3)*ZCEN+ROTM(2,4)
        ZN=ROTMX(3,1)*XCEN+ROTMX(3,2)*YCEN+ROTMX(3,3)*ZCEN+ROTM(3,4)
        IP=1+1          ! X  INDEX
        SHAPEF(IPINDX(IP),IPROP)=XN
        IP=1+NP         ! Y  INDEX
        SHAPEF(IPINDX(IP),IPROP)=YN
        IP=1+NP2        ! Z  INDEX
        SHAPEF(IPINDX(IP),IPROP)=ZN
     ENDIF
     !
  ENDDO ! DO(IPROP)
  !
  ! Now rotate atoms of this shape if type='rigi' (rigid)
  !
  IF (STYPE == 'RIGI') THEN
     DO I=1,NAT
        IF(IABS(SHPSEL(I)).EQ.ISHAPE) THEN
           XN=ROTMX(1,1)*X(I)+ROTMX(1,2)*Y(I)+ROTMX(1,3)*Z(I)+ROTM(1,4)
           YN=ROTMX(2,1)*X(I)+ROTMX(2,2)*Y(I)+ROTMX(2,3)*Z(I)+ROTM(2,4)
           ZN=ROTMX(3,1)*X(I)+ROTMX(3,2)*Y(I)+ROTMX(3,3)*Z(I)+ROTM(3,4)
           X(I)=XN
           Y(I)=YN
           Z(I)=ZN
        ENDIF
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE SHAPEROT

SUBROUTINE SHAPEMAN(WRD,IPROP,NORD,SHAPEI,SHAPEJ,LENSHP,NPROP, &
     WEIGHT,FACT,RESULT)
  !
  ! This routine manipulates two shape descriptors.
  !
  !  WRD     - the name of the current manipulation command
  !  NORD    - Order of shape factor
  !  SHAPEI  - Modified shape factor
  !  SHAPEJ  - Other shape factor
  !  WEIGHT  - Weight for each factor in fitting
  !  FACT    - scale factor
  !
  !  Valid operations are:
  !      SUM:   I <- I+J
  !      DIFF:  I <- I-J
  !      DUPL:  I <-   J
  !
  !     By Bernard R. Brooks   October 31, 1995
  !
  use number
  use stream
  use param_store, only: set_param

  CHARACTER(len=4) WRD
  INTEGER NORD, IPROP, LENSHP, NPROP
  real(chm_real) SHAPEI(LENSHP,NPROP),SHAPEJ(LENSHP,NPROP)
  real(chm_real) WEIGHT(LENSHP,NPROP), FACT, RESULT
  !
  INTEGER IS,JS,KS,IPT,IORD,IP
  real(chm_real) SSQ,SSQTOT
  !
  SSQTOT=ZERO
  DO IP=1,NPROP
     IF(IP.EQ.IPROP .OR. IPROP.LT.0) THEN
        !
        IPT=0
        SSQ=ZERO
        DO IS=0,NORD
           DO JS=0,NORD
              DO KS=0,NORD
                 IORD=IS+JS+KS
                 IF(IORD.LE.NORD) THEN
                    IPT=IPT+1
                    IF(WRD(1:3).EQ.'SUM') THEN
                       SHAPEI(IPT,IP)=SHAPEJ(IPT,IP)+SHAPEI(IPT,IP)
                    ELSEIF(WRD.EQ.'DIFF') THEN
                       SHAPEI(IPT,IP)=SHAPEI(IPT,IP)-SHAPEJ(IPT,IP)
                    ELSEIF(WRD.EQ.'COPY') THEN
                       SHAPEI(IPT,IP)=SHAPEJ(IPT,IP)
                    ELSEIF(WRD.EQ.'SCAL') THEN
                       SHAPEI(IPT,IP)=SHAPEI(IPT,IP)*FACT
                    ELSEIF(WRD.EQ.'COMP') THEN
                       SSQ=SSQ+(SHAPEI(IPT,IP)-SHAPEJ(IPT,IP))**2 &
                            *WEIGHT(IPT,IP)
                    ENDIF
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        !
        SSQTOT=SSQTOT+SSQ
        IF(WRD.EQ.'COMP') THEN
           IF(SSQ.LE.0.0) SSQ=0.0
           IF(PRNLEV.GT.2) WRITE(OUTU,65) IP,SQRT(SSQ)
65         FORMAT(' SHAPES:: FIT> Weighted rms for property',I4, &
                ' is:',F20.6)
        ENDIF
        !
     ENDIF
  ENDDO  ! DO(IP)
  !
  SSQ=SSQTOT
  IF(WRD.EQ.'COMP') THEN
     IF(SSQ.LE.0.0) SSQ=0.0
     RESULT=SQRT(SSQ)
     IF(PRNLEV.GT.2) WRITE(OUTU,75) RESULT
75   FORMAT(' SHAPES:: FIT> Total weighted rms fit for all:  ', &
          F20.6)
     call set_param('SFIT',RESULT)
  ENDIF
  !
  RETURN
END SUBROUTINE SHAPEMAN

SUBROUTINE SHPGETCEN(NORD,SHAPEI,LENSHP,NPROP,IPROP,ROTM,IPINDX, &
     QAXIS,PCENT)
  !
  !  This routine calculates that trans/rotation matric required to center
  !  a particular shape descriptor to the origin and align along the principal
  !  axis.  Charges are ignored for this operation.
  !
  !           By Bernard R. Brooks   November 6, 1995
  !
  use number
  use corsubs, only: &
    axisx,axisy,axisz,axisr, &
    axiscx,axiscy,axiscz,qaxisc,fndrot
  use param_store, only: set_param
  use stream

  INTEGER NORD, LENSHP, NPROP, IPROP
  real(chm_real) SHAPEI(LENSHP,NPROP)
  real(chm_real) ROTM(3,4)
  INTEGER IPINDX(*)
  INTEGER QAXIS
  CHARACTER(len=4) PCENT
  !
  !
  real(chm_real) U(9),UM(3,3),SCR(24),AMOM(6),RN(3),CM(3)
  EQUIVALENCE(U,UM)
  real(chm_real) DET,PHI,RNORM
  INTEGER NP,NP2
  INTEGER I,J,IPT,IP
  LOGICAL LPRNT
  !
  NP=NORD+1
  NP2=NP*NP
  LPRNT=PRNLEV.GT.2
  !
  IF(PCENT.EQ.'CENT' .OR. PCENT.EQ.'NORM') THEN
     RNORM=ONE
  ELSE
     RNORM=SHAPEI(1,IPROP)
     IF(RNORM.LE.TENM5) RNORM=ONE
     RNORM=ONE/RNORM
  ENDIF
  !
  IF(PCENT.EQ.'CENT') THEN
     CM(1)=ZERO
     CM(2)=ZERO
     CM(3)=ZERO
  ELSE
     IP=1+1          ! X  INDEX
     CM(1)=-SHAPEI(IPINDX(IP),IPROP)*RNORM
     IP=1+NP         ! Y  INDEX
     CM(2)=-SHAPEI(IPINDX(IP),IPROP)*RNORM
     IP=1+NP2        ! Z  INDEX
     CM(3)=-SHAPEI(IPINDX(IP),IPROP)*RNORM
  ENDIF
  !
  IP=2+1          ! XX INDEX
  AMOM(1)=RNORM*SHAPEI(IPINDX(IP),IPROP)-CM(1)**2
  IP=1+NP+1       ! XY INDEX
  AMOM(2)=RNORM*SHAPEI(IPINDX(IP),IPROP)-CM(1)*CM(2)
  IP=1+NP2+1      ! XZ INDEX
  AMOM(3)=RNORM*SHAPEI(IPINDX(IP),IPROP)-CM(1)*CM(3)
  IP=2*NP+1       ! YY INDEX
  AMOM(4)=RNORM*SHAPEI(IPINDX(IP),IPROP)-CM(2)**2
  IP=NP+NP2+1     ! YZ INDEX
  AMOM(5)=RNORM*SHAPEI(IPINDX(IP),IPROP)-CM(2)*CM(3)
  IP=NP2*2+1      ! ZZ INDEX
  AMOM(6)=RNORM*SHAPEI(IPINDX(IP),IPROP)-CM(3)**2
  !
  IF(PCENT.EQ.'CENT') THEN
     IP=1+1          ! X  INDEX
     CM(1)=-SHAPEI(IPINDX(IP),IPROP)*RNORM
     IP=1+NP         ! Y  INDEX
     CM(2)=-SHAPEI(IPINDX(IP),IPROP)*RNORM
     IP=1+NP2        ! Z  INDEX
     CM(3)=-SHAPEI(IPINDX(IP),IPROP)*RNORM
  ENDIF
  !
  IF(LPRNT) WRITE(OUTU,104) CM
104 FORMAT(' CENTER:',4X,3F12.5)
  IF(LPRNT) WRITE(OUTU,105) AMOM
105 FORMAT(' MOMENTS:'/3F16.8,/16X,2F16.8,/32X,F16.8/)
  !
  CALL DIAGQ(3,3,AMOM,U,SCR(4),SCR(7),SCR(10),SCR(13),SCR(1), &
       SCR(16),SCR(19),SCR(22),0)
  !
  IF(LPRNT) WRITE(OUTU,106) SCR(3),SCR(2),SCR(1)
106 FORMAT(' EIGENVALUES:',3F16.8)
  !
  DO I=1,3
     DET=U(I)
     U(I)=U(I+6)
     U(I+6)=DET
  ENDDO
  !      DO I=1,3
  !         IPT=(I-1)*3
  !         DET=U(IPT+1)
  !         U(IPT+1)=U(IPT+3)
  !         U(IPT+3)=DET
  !         IF(U(IPT+I).LT.ZERO) THEN
  !            DO J=1,3
  !               IPT=IPT+1
  !               U(IPT)=-U(IPT)
  !            ENDDO
  !         ENDIF
  !      ENDDO
  DET=U(1)*(U(5)*U(9)-U(6)*U(8))+U(2)*(U(6)*U(7)-U(4)*U(9))+ &
       U(3)*(U(4)*U(8)-U(5)*U(7))
  IF(DET.LT.ZERO) THEN
     U(7)=-U(7)
     U(8)=-U(8)
     U(9)=-U(9)
     DET=-DET
  ENDIF
  IF(WRNLEV.GT.2 .AND. ABS(DET-ONE).GT.1.D-4) WRITE(OUTU,203)DET
203 FORMAT(/' ***** WARNING ***** FROM SHPGETCEN.', &
       ' ROTATION MATRIX IS NOT UNITARY.'/,' DETERMINANT=',F14.8/)
  IF(LPRNT) WRITE(OUTU,205) U
205 FORMAT(' Transpose of the rotation matrix',3(/1X,3F12.6))
  !
  IF(QAXIS.GT.0) THEN
     QAXISC=.TRUE.
     AXISCX= CM(1)
     AXISCY= CM(2)
     AXISCZ= CM(3)
     AXISR = ONE
     AXISX = UM(1,QAXIS)
     AXISY = UM(2,QAXIS)
     AXISZ = UM(3,QAXIS)
     call set_param('XAXI',AXISX)
     call set_param('YAXI',AXISY)
     call set_param('ZAXI',AXISZ)
     call set_param('RAXI',AXISR)
     call set_param('XCEN',AXISCX)
     call set_param('YCEN',AXISCY)
     call set_param('ZCEN',AXISCZ)
  ENDIF
  !
  CALL FNDROT(U,RN,PHI,LPRNT)
  IPT=0
  DO I=1,3
     DO J=1,3
        IPT=IPT+1
        ROTM(I,J)=U(IPT)
     ENDDO
  ENDDO
  DO I=1,3
     ROTM(I,4)=ZERO
     DO J=1,3
        ROTM(I,4)=ROTM(I,4)+ROTM(I,J)*CM(J)
     ENDDO
  ENDDO
  !
  !      IF(LPRNT) WRITE(OUTU,305) ROTM
  ! 305  FORMAT(' ROT MATRIX:'/12X,3F16.8,/12X,3F16.8,
  !     &                     /12X,3F16.8,/12X,3F16.8/)
  !
  RETURN
END SUBROUTINE SHPGETCEN

SUBROUTINE INVTTRM(A,C)
  !
  ! This routine inverts a rotation translation matrix
  !          -1
  !     C = A
  !
  ! Do a copy at the end to allow duplicate use of arrays - BRB
  !
  use number

  real(chm_real) A(3,4),C(3,4)
  !
  INTEGER I,J
  real(chm_real) TEMP(3,4)
  !
  DO I=1,3
     TEMP(I,4)=ZERO
     DO J=1,3
        TEMP(J,I)=A(I,J)
        TEMP(I,4)=TEMP(I,4)-A(J,4)*A(J,I)
     ENDDO
  ENDDO
  !
  DO I=1,4
     DO J=1,3
        C(J,I)=TEMP(J,I)
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE INVTTRM
#endif /* (shapes)*/

end module shapes

