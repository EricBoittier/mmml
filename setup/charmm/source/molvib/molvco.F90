module molvco_m
  implicit none
contains

#if KEY_MOLVIB==0
SUBROUTINE MOLVCO(COMLYN,COMLEN)
  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN

  CALL WRNDIE(-1,'<MOLVCO>','MOLVIB is not compiled.')
  RETURN
END SUBROUTINE MOLVCO
#else /**/
SUBROUTINE MOLVCO(COMLYN,COMLEN)
  !
  !     This is the front-end routine for the MOLVIB module.
  !     Function:
  !      - read the MOLVIB card, determine dimensons and control parameters
  !      - allocate AX, AM, FX, FS  on HEAP
  !      - call second-level front-end: MOLVCP
  !----------------------------------------------------------------------------
  !      K.Kuczera,     Cambridge, MA,    23-Apr-1991                         |
  !      Victor Anisimov,2004: Drude support
  !      Victor Anisimov,2005: LP support
  !----------------------------------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use psf
  use stream
  use string
  use number
  use lonepr
  implicit none
  !
  ! .Passed variables.
  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN
  !
  ! .Local variables.
  !
  ! ..MOLVIB dimension variables..
  !   NIC0 - number of primitive IC's (R) to be read in after the 'IC' card
  !   NIC  - number of IC's S1=U1*R after transformation with first U matrix
  !   NQ   - number of IC's S2=U2*S1 after transformation with second U matrix
  !   NQM  - basic dimension passed to MOLVIB - all two-dim arrays will be
  !          (NQM,NQM); NQM = max(NAT3,NIC0,NIC,NQ)
  !   NAT3 = 3*NATOM (NATOM is CHARMM atom counter from psf.f90)
  !   IZMAX - max. no of molecules in unit cell ('CRYS' only); default=10
  !   MAXSYB - max. no. of symbols for PED analysis; default=NQ
  !   NGMAX  - max. no. of groups in PED analysis; default=NQ
  !   NBLMAX - max. no. of symmetry blocks in G,F for PED analysis; default=NQ
  !   NAT - separate atom counter, kept in case no CHARMM topology/psf
  !         is present; NAT and NATOM are forced equal
  !         (NAT atoms, from total NATOM, will be passed to MOLVIB in
  !         the presence of Drudes.)
  !
  INTEGER NQ,NIC,NIC0,NQM,NAT,NAT3,NQM2,IZMAX,MAXSYB,NGMAX,NBLMAX
  real(chm_real) FINI
  !
  ! ..Control..
  LOGICAL LSDC,LTOP,LPRF
  !
  ! ..Temporary..
  real(chm_real),allocatable,dimension(:) :: IXM
  real(chm_real),allocatable,dimension(:) :: IAM
  real(chm_real),allocatable,dimension(:) :: IFX
  real(chm_real),allocatable,dimension(:) :: IFS
  !
  ! ...Print header
  !
  WRITE(OUTU,*)
  WRITE(OUTU,*) '  >>>>>> Entering MOLVIB module of CHARMM <<<<<<'
  WRITE(OUTU,*)
  !
  ! ... LSDC flag: if true, FX will be calculated
  ! ... LTOP flag: if true, no topology/psf information from CHARMM
  ! ...            will be used
  ! ... LPRF flag: if true, FX will be printed out before MOLVIB call
  !
  LSDC = (INDXA(COMLYN,COMLEN,'SECO').GT.0)
  LTOP = (INDXA(COMLYN,COMLEN,'NOTO').GT.0)
  LPRF = (INDXA(COMLYN,COMLEN,'PRIN').GT.0)
  IF(LTOP) THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) '    MLVCOM: CHARMM RTF/PSF will be ignored', &
          ' inside MOLVIB'
     WRITE(OUTU,*)
  ELSE
     WRITE(OUTU,*)
     WRITE(OUTU,*) '    MLVCOM: CHARMM RTF/PSF will be used', &
          ' inside MOLVIB'
     WRITE(OUTU,*)
  END IF
  IF(LSDC) THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) '    MLVCOM: CHARMM second derivatives will be'
     WRITE(OUTU,*) '     evaluated and passed directly to MOLVIB'
     WRITE(OUTU,*)
  END IF
  !
  ! ... Read FINIte step size (for numerical second differentiation)
  FINI=GTRMF(COMLYN,COMLEN,'FINI',PT01)
  !
  ! ... Determine MOLVIB dimensions
  !
  NQ   = GTRMI(COMLYN,COMLEN,'NDI1',-1)
  NIC  = GTRMI(COMLYN,COMLEN,'NDI2',-1)
  NIC0 = GTRMI(COMLYN,COMLEN,'NDI3',-1)
  NAT  = GTRMI(COMLYN,COMLEN,'NATO',-1)
  !
  IZMAX  = GTRMI(COMLYN,COMLEN,'IZMA',10)
  MAXSYB = GTRMI(COMLYN,COMLEN,'MAXS',NQ)
  NGMAX  = GTRMI(COMLYN,COMLEN,'NGMA',NQ)
  NBLMAX = GTRMI(COMLYN,COMLEN,'NBLM',NQ)
  ! ...Check consistency
  WRITE(OUTU,*) ' MOLVIB dimensions have been set'
  WRITE(OUTU,*)
  WRITE(OUTU,'(A,4I8)') '   NQ,NIC,NIC0,NAT =',NQ,NIC,NIC0,NAT
  WRITE(OUTU,'(A,4I8)') '   IZMAX,MAXSYB,NGMAX,NBLMAX =', &
       IZMAX,MAXSYB,NGMAX,NBLMAX
  IF(NQ.LE.0 .OR. NIC.LE.0 .OR. NIC0.LE.0) THEN
     WRITE(OUTU,*) ' ****ERROR: DIMENSION NOT SET'
     WRITE(OUTU,'(A,3I8)') ' NDI1, NDI2, NDI3 =',NQ,NIC,NIC0
     CALL WRNDIE(-4,'<MLVCOM>','CHECK MOLVIB CARD')
  END IF
  IF(NQ.GT.NIC .OR. NQ.GT.NIC0 .OR. NIC.GT.NIC0) THEN
     WRITE(OUTU,*) ' ****WARNING: INCONSISTENT DIMENSIONS'
     WRITE(OUTU,'(A,3I8)') ' NDI1, NDI2, NDI3 =',NQ,NIC,NIC0
     CALL WRNDIE(-1,'<MLVCOM>','CHECK MOLVIB CARD')
  END IF
  IF(IZMAX.LE.0 .OR. MAXSYB.LE.0 .OR. NGMAX.LE.0 .OR. &
       NBLMAX.LE.0) THEN
     WRITE(OUTU,*) ' ****WARNING: INCONSISTENT DIMENSIONS'
     WRITE(OUTU,'(A,4I8)') ' IZMAX,MAXSYB,NGMAX,NBLMAX =', &
          IZMAX,MAXSYB,NGMAX,NBLMAX
     CALL WRNDIE(-1,'<MLVCOM>','CHECK MOLVIB CARD')
  END IF
  ! ...Set NAT3, NQM, NQM2
  IF(LTOP) THEN
     NATOM=NAT
  ELSE
     NAT=NATOM-NUMLP-NDRUDE
  ENDIF
  NAT3 = 3*NAT
  IF(NATOM.LE.0 .OR. NAT3.LE.0) THEN
     WRITE(OUTU,*) ' ****ERROR: NUMBER OF ATOMS ILLEGAL'
     WRITE(OUTU,*) ' NATOM, NAT, NAT3 =',NATOM,NAT,NAT3
     CALL WRNDIE(-4,'<MLVCOM>','CHECK MOLVIB CARD')
  END IF
  NQM=NIC0
  IF(NIC.GT.NQM)  NQM = NIC
  IF(NQ.GT.NQM)   NQM = NQ
  IF(NAT3.GT.NQM) NQM = NAT3
  NQM2 = NQM**2
  WRITE(OUTU,*)
  WRITE(OUTU,*) ' The maximum dimension has been set to NQM =',NQM
  WRITE(OUTU,*)
  !
  ! ...Allocate space on HEAP for arrays which are needed before MOLVIB call:
  ! ...  XM(NAT3) - cartesian coordinates
  ! ...  AM(NAT3) - atomic masses
  ! ...  FX(NQM,NQM) - cartesian force constants
  ! ...  FS  - here used as work array
  ! ...and call MLVCON - second-level front end for MOLVIB
  !
  call chmalloc('molvco.src','MOLVCO','IXM',NAT3,crl=IXM)
  call chmalloc('molvco.src','MOLVCO','IAM',NAT3,crl=IAM)
  call chmalloc('molvco.src','MOLVCO','IFX',NQM2,crl=IFX)
  call chmalloc('molvco.src','MOLVCO','IFS',NQM2,crl=IFS)
  !
  call MOLVCP(NQ,NIC,NIC0,IZMAX,MAXSYB,NGMAX,NBLMAX,NAT,NAT3,NQM,NQM2, &
       LSDC,LTOP,LPRF,IXM,IAM,IFX,IFS,FINI)
  !
  call chmdealloc('molvco.src','MOLVCO','IXM',NAT3,crl=IXM)
  call chmdealloc('molvco.src','MOLVCO','IAM',NAT3,crl=IAM)
  call chmdealloc('molvco.src','MOLVCO','IFX',NQM2,crl=IFX)
  call chmdealloc('molvco.src','MOLVCO','IFS',NQM2,crl=IFS)
  !
  RETURN
END SUBROUTINE MOLVCO

SUBROUTINE MOLVCP(NQ,NIC,NIC0,IZMAX,MAXSYB,NGMAX,NBLMAX, &
     NAT,NAT3,NQM,NQM2,LSDC,LTOP,LPRF,XM,AM,FX,FS,FINI)
  !
  !     This is the second-level front-end routine for the MOLVIB module.
  !     Function:
  !     - (If needed) call ENERGY to get cartesian force constants
  !     - translate CHARMM coordinates and masess to MOLVIB ones
  !     - allocate space on HEAP
  !     - call MOLVIB
  !     - free space on HEAP
  !----------------------------------------------------------------------------
  !      K.Kuczera,     Cambridge, MA,    23-Apr-1991                         |
  !----------------------------------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use coord
  use memory
  use psf
  use stream
  implicit none
  !
  ! .Passed variables.
  !
  INTEGER NQ,NIC,NIC0,NQM,NAT,NAT3,NQM2
  INTEGER IZMAX,MAXSYB,NGMAX,NBLMAX
  LOGICAL LSDC,LTOP,LPRF
  real(chm_real)  XM(NAT3),AM(NAT3),FX(NQM,NQM),FS(NQM,NQM)
  real(chm_real) FINI
  !
  ! .Local variables.
  !
  INTEGER I,I1,I2,I3
  character(len=4),allocatable,dimension(:) :: ATYP,SBLOC
  character(len=8),allocatable,dimension(:) :: SYMB,SPED

  real(chm_real),allocatable,dimension(:) :: IBMAT
  real(chm_real),allocatable,dimension(:) :: IG
  real(chm_real),allocatable,dimension(:) :: ILS
  real(chm_real),allocatable,dimension(:) :: ILX
  real(chm_real),allocatable,dimension(:) :: IU1
  real(chm_real),allocatable,dimension(:) :: IU2
  real(chm_real),allocatable,dimension(:) :: IW1
  real(chm_real),allocatable,dimension(:) :: IW2
  real(chm_real),allocatable,dimension(:) :: IDD
  real(chm_real),allocatable,dimension(:) :: IEXPF
  real(chm_real),allocatable,dimension(:) :: ICALF
  real(chm_real),allocatable,dimension(:) :: IV1
  real(chm_real),allocatable,dimension(:) :: IV2
  real(chm_real),allocatable,dimension(:) :: IV3
  real(chm_real),allocatable,dimension(:) :: IV4
  real(chm_real),allocatable,dimension(:) :: IV5
  real(chm_real),allocatable,dimension(:) :: IV6
  real(chm_real),allocatable,dimension(:) :: IV7
  real(chm_real),allocatable,dimension(:) :: IPRIN
  real(chm_real),allocatable,dimension(:) :: ITOMM
  integer,allocatable,dimension(:) :: IICTYP
  integer,allocatable,dimension(:) :: IIATI
  integer,allocatable,dimension(:) :: IIATJ
  integer,allocatable,dimension(:) :: IIATK
  integer,allocatable,dimension(:) :: IIATL
  integer,allocatable,dimension(:) :: IINDX
  integer,allocatable,dimension(:) :: IMNAT
  integer,allocatable,dimension(:) :: ILGRUP
  integer,allocatable,dimension(:) :: IIGRUP
  integer,allocatable,dimension(:) :: IKSYMB
  integer,allocatable,dimension(:) :: IIPTA
  integer,allocatable,dimension(:) :: IIPTB
  integer,allocatable,dimension(:) :: IIBLOC
  logical,allocatable,dimension(:) :: IQSEL
  !
  ! ... Calculate second derivatives of energy with respect to present cartesian
  ! ... coordinates, if needed
  !
  IF(LSDC .AND. .NOT.LTOP) THEN
     CALL GETFX0(FX,FS,NAT,NAT3,NQM,LPRF,FINI)
  END IF
  !
  ! ...To pass cartesian coordinates and masses to MOLVIB - rearrange them
  ! ...Do not copy Drudes and Lone-Pairs
  !
  IF(.NOT.LTOP) THEN
     IF(ISDRUDE(1)) THEN
        CALL WRNDIE(-1,'<MOLVCP>', &
             'Drude particle cannot be first in the coordinate table')
     ENDIF
     I1 = 0
     DO I=1,NATOM
        IF(ISDRUDE(I)) THEN
           ! ...Add Drude masses back to real atoms
           AM(I1)=AM(I1)+AMASS(I)
           AM(I2)=AM(I2)+AMASS(I)
           AM(I3)=AM(I3)+AMASS(I)
        ELSEIF(IMOVE(I).EQ.0) THEN
           I1=I1+1
           I2=I1+NAT
           I3=I2+NAT
           XM(I1)=X(I)
           XM(I2)=Y(I)
           XM(I3)=Z(I)
           AM(I1)=AMASS(I)
           AM(I2)=AMASS(I)
           AM(I3)=AMASS(I)
        ENDIF
     ENDDO
     !
     WRITE(OUTU,*)
     WRITE(OUTU,*) '  CHARMM cartesian coordinates passed to MOLVIB'
     WRITE(OUTU,*)
     WRITE(OUTU,*) '   Atom #       X         Y         Z', &
          '       MASS  '
     DO I=1,NAT
        WRITE(OUTU,921) I,XM(I),XM(I+NAT),XM(I+2*NAT),AM(I)
     END DO
     !
     WRITE(OUTU,*)
     WRITE(OUTU,*) '  Atomic masses passed to MOLVIB:'
     WRITE(OUTU,*)
     WRITE(OUTU,922)  (AM(I),I=1,NAT3)
     !
  END IF
  !
921 FORMAT(3X,I3,3X,4F10.5)
922 FORMAT(1X,8F9.5,12(/1X,8F9.5))
  !
  ! ...Allocate space on HEAP for MOLVIB arrays
  !
  ! ...real(chm_real) arrays
  call chmalloc('molvco.src','MOLVCP','IBMAT',NQM2,crl=IBMAT)
  call chmalloc('molvco.src','MOLVCP','IG',NQM2,crl=IG)
  call chmalloc('molvco.src','MOLVCP','ILS',NQM2,crl=ILS)
  call chmalloc('molvco.src','MOLVCP','ILX',NQM2,crl=ILX)
  call chmalloc('molvco.src','MOLVCP','IU1',NQM2,crl=IU1)
  call chmalloc('molvco.src','MOLVCP','IU2',NQM2,crl=IU2)
  call chmalloc('molvco.src','MOLVCP','IW1',NQM2,crl=IW1)
  call chmalloc('molvco.src','MOLVCP','IW2',NQM2,crl=IW2)
  !
  call chmalloc('molvco.src','MOLVCP','IDD',NQM,crl=IDD)
  call chmalloc('molvco.src','MOLVCP','IEXPF',NQM,crl=IEXPF)
  call chmalloc('molvco.src','MOLVCP','ICALF',NQM,crl=ICALF)
  call chmalloc('molvco.src','MOLVCP','IV1',NQM,crl=IV1)
  call chmalloc('molvco.src','MOLVCP','IV2',NQM,crl=IV2)
  call chmalloc('molvco.src','MOLVCP','IV3',NQM,crl=IV3)
  call chmalloc('molvco.src','MOLVCP','IV4',NQM,crl=IV4)
  call chmalloc('molvco.src','MOLVCP','IV5',NQM,crl=IV5)
  call chmalloc('molvco.src','MOLVCP','IV6',NQM,crl=IV6)
  call chmalloc('molvco.src','MOLVCP','IV7',NQM,crl=IV7)
  !
  call chmalloc('molvco.src','MOLVCP','IPRIN',3*IZMAX,crl=IPRIN)
  call chmalloc('molvco.src','MOLVCP','ITOMM',IZMAX,crl=ITOMM)
  !
  ! ...Integer arrays
  call chmalloc('molvco.src','MOLVCP','IICTYP',NQM,intg=IICTYP)
  call chmalloc('molvco.src','MOLVCP','IIATI',NQM,intg=IIATI)
  call chmalloc('molvco.src','MOLVCP','IIATJ',NQM,intg=IIATJ)
  call chmalloc('molvco.src','MOLVCP','IIATK',NQM,intg=IIATK)
  call chmalloc('molvco.src','MOLVCP','IIATL',NQM,intg=IIATL)
  call chmalloc('molvco.src','MOLVCP','IINDX',NQM,intg=IINDX)
  !
  call chmalloc('molvco.src','MOLVCP','IMNAT',IZMAX,intg=IMNAT)
  !
  call chmalloc('molvco.src','MOLVCP','ILGRUP',NGMAX,intg=ILGRUP)
  call chmalloc('molvco.src','MOLVCP','IIGRUP',NGMAX*MAXSYB,&
       intg=IIGRUP)
  call chmalloc('molvco.src','MOLVCP','IKSYMB',MAXSYB,intg=IKSYMB)
  call chmalloc('molvco.src','MOLVCP','IIPTA',NBLMAX,intg=IIPTA)
  call chmalloc('molvco.src','MOLVCP','IIPTB',NBLMAX,intg=IIPTB)
  call chmalloc('molvco.src','MOLVCP','IIBLOC',NBLMAX,intg=IIBLOC)
  !
  ! ...Logical array
  call chmalloc('molvco.src','MOLVCP','IQSEL',NBLMAX,log=IQSEL)

  call chmalloc('molvco.src','MOLVCP','atyp',Nat3,ch4=atyp)
  call chmalloc('molvco.src','MOLVCP','sbloc',nblmax,ch4=sbloc)
  call chmalloc('molvco.src','MOLVCP','symb',maxsyb,ch8=symb)
  call chmalloc('molvco.src','MOLVCP','sped',nblmax,ch8=sped)
  !
  ! ...Call MOLVIB
  !
  call MOLVIB(NQ,NIC,NIC0,IZMAX,MAXSYB,NGMAX,NBLMAX,NAT,NAT3,NQM,ISTRM, &
       OUTU,XM,AM,FX,IBMAT,IG,FS,ILS,ILX,IU1,IU2,IW1, &
       IW2,IDD,IEXPF,ICALF,IV1,IV2,IV3,IV4,IV5, &
       IV6,IV7,IPRIN,ITOMM,IICTYP,IIATI,IIATJ, &
       IIATK,IIATL,IINDX,IMNAT,ILGRUP,IIGRUP,IKSYMB, &
       IIPTA,IIPTB,IIBLOC,IQSEL,ATYP, &
       SBLOC,SYMB,SPED)
  !
  ! ...Free HEAP space
  !
  call chmdealloc('molvco.src','MOLVCP','IBMAT',NQM2,crl=IBMAT)
  call chmdealloc('molvco.src','MOLVCP','IG',NQM2,crl=IG)
  call chmdealloc('molvco.src','MOLVCP','ILS',NQM2,crl=ILS)
  call chmdealloc('molvco.src','MOLVCP','ILX',NQM2,crl=ILX)
  call chmdealloc('molvco.src','MOLVCP','IU1',NQM2,crl=IU1)
  call chmdealloc('molvco.src','MOLVCP','IU2',NQM2,crl=IU2)
  call chmdealloc('molvco.src','MOLVCP','IW1',NQM2,crl=IW1)
  call chmdealloc('molvco.src','MOLVCP','IW2',NQM2,crl=IW2)
  !
  call chmdealloc('molvco.src','MOLVCP','IDD',NQM,crl=IDD)
  call chmdealloc('molvco.src','MOLVCP','IEXPF',NQM,crl=IEXPF)
  call chmdealloc('molvco.src','MOLVCP','ICALF',NQM,crl=ICALF)
  call chmdealloc('molvco.src','MOLVCP','IV1',NQM,crl=IV1)
  call chmdealloc('molvco.src','MOLVCP','IV2',NQM,crl=IV2)
  call chmdealloc('molvco.src','MOLVCP','IV3',NQM,crl=IV3)
  call chmdealloc('molvco.src','MOLVCP','IV4',NQM,crl=IV4)
  call chmdealloc('molvco.src','MOLVCP','IV5',NQM,crl=IV5)
  call chmdealloc('molvco.src','MOLVCP','IV6',NQM,crl=IV6)
  call chmdealloc('molvco.src','MOLVCP','IV7',NQM,crl=IV7)
  !
  call chmdealloc('molvco.src','MOLVCP','IPRIN',3*IZMAX,crl=IPRIN)
  call chmdealloc('molvco.src','MOLVCP','ITOMM',IZMAX,crl=ITOMM)
  !
  call chmdealloc('molvco.src','MOLVCP','IICTYP',NQM,intg=IICTYP)
  call chmdealloc('molvco.src','MOLVCP','IIATI',NQM,intg=IIATI)
  call chmdealloc('molvco.src','MOLVCP','IIATJ',NQM,intg=IIATJ)
  call chmdealloc('molvco.src','MOLVCP','IIATK',NQM,intg=IIATK)
  call chmdealloc('molvco.src','MOLVCP','IIATL',NQM,intg=IIATL)
  !
  call chmdealloc('molvco.src','MOLVCP','IMNAT',IZMAX,intg=IMNAT)
  !
  call chmdealloc('molvco.src','MOLVCP','ILGRUP',NGMAX,intg=ILGRUP)
  call chmdealloc('molvco.src','MOLVCP','IIGRUP',NGMAX*MAXSYB, &
       intg=IIGRUP)
  call chmdealloc('molvco.src','MOLVCP','IKSYMB',MAXSYB,intg=IKSYMB)
  call chmdealloc('molvco.src','MOLVCP','IIPTA',NBLMAX,intg=IIPTA)
  call chmdealloc('molvco.src','MOLVCP','IIPTB',NBLMAX,intg=IIPTB)
  call chmdealloc('molvco.src','MOLVCP','IIBLOC',NBLMAX,intg=IIBLOC)
  !
  call chmdealloc('molvco.src','MOLVCP','IQSEL',NBLMAX,log=IQSEL)
  !
  !
  call chmdealloc('molvco.src','MOLVCP','atyp',Nat3,ch4=atyp)
  call chmdealloc('molvco.src','MOLVCP','sbloc',nblmax,ch4=sbloc)
  call chmdealloc('molvco.src','MOLVCP','symb',maxsyb,ch8=symb)
  call chmdealloc('molvco.src','MOLVCP','sped',nblmax,ch8=sped)
  !
  ! ...Pass any coordinate transformations to main program
  ! ...Place Drudes at the center of the corresponding atom by applying
  !    coordinate duplication
  ! ...Standard numeration of atoms in CHARMM assumes the sequence: real
  !    atom, its Drude, etc
  !
  IF(.NOT.LTOP) THEN
     ! ...First atom requires special handling (test of Drude not being first atom)
     IF(ISDRUDE(1)) THEN
        CALL WRNDIE(-1,'<MOLVCP>', &
             'Drude particle cannot be first in the coordinate table')
     ENDIF
     ! ...The following data have to be computed prior to starting the loop
     I1 = 1
     I2 = 1 + NAT
     I3 = 1 + NAT + NAT
     DO I=1,NATOM
        IF(ISDRUDE(I).AND.I.GT.1) THEN
           ! ...If the I-th particle is Drude then take coordinates of its parent
           X(I) = X(I-1)
           Y(I) = X(I-1)
           Z(I) = X(I-1)
        ELSEIF(IMOVE(I).EQ.0) THEN
           ! ...Do not update coordinates of Lone Pairs
           ! ...Check limits
           IF(I3.GT.NAT3) THEN
              CALL WRNDIE(-1,'<MOLVCP>','Wrong atom passed to MOLVIB')
           ELSE
              X(I) = XM(I1)
              Y(I) = XM(I2)
              Z(I) = XM(I3)
              I1 = I1 + 1
              I2 = I2 + 1
              I3 = I3 + 1
           ENDIF
        ENDIF
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE MOLVCP

SUBROUTINE GETFX0(FX,FS,NAT,NAT3,NQM,LPRF,FINI)
  !
  !... First step on the way to the cartesian second derivatives.
  !
  !
  use chm_kinds
  use chm_types
  use number,only:zero
  use dimens_fcm
  use memory
  use comand
  !
  use bases_fcm
  use coord
  use image
  use psf
  implicit none
  !
  ! .Passed variables.
  INTEGER NAT,NAT3,NATOM3,NQM
  real(chm_real)  FX(NQM,NQM),FS(NQM,NQM)
  LOGICAL LPRF
  ! . Local variables.
  INTEGER ISPACE
  real(chm_real) FINI
  real(chm_real),allocatable,dimension(:) :: IDD1
  !
  !... Update nonbonded energy list
  !
  CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
       .TRUE.,.TRUE.,.TRUE.,.TRUE., &
       0,[zero],[zero],[zero],[zero],[zero],[zero])
  !
  !... Allocate space for upper triangle of second derivative matrix
  !... stored in one-dim array DD1
  !
  NATOM3=NATOM*3
  ISPACE=(NATOM3*(NATOM3+1))/2
  IF(NATOM3.GT.600) THEN
     CALL WRNDIE(-1,'<GETFX0>','NATOM3 for second derivatives is >600')
  END IF
  call chmalloc('molvco.src','GETFX0','IDD1',ISPACE,crl=IDD1)
  !
  call GETFX1(X,Y,Z,BNBND,BIMAG,IDD1,FX,FS,NAT,NAT3,NATOM3,NQM,LPRF, &
       FINI,ISPACE)
  !
  call chmdealloc('molvco.src','GETFX0','IDD1',ISPACE,crl=IDD1)
  !
  RETURN
END SUBROUTINE GETFX0

SUBROUTINE GETFX1(X,Y,Z,BNBND,BIMAG,DD1,FX,FS,NAT,NAT3,NATOM3, &
     NQM,LPRF,FINI,ISPACE)
  !
  ! ... Finally we have the second derivatives
  ! ... Numerical finite differences for Drudes, Victor Anisimov 2004
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use memory
  use deriv
  use energym
  use parallel,only: gcomb
  use stream
  use psf
  use number
  use vibcom, only: gensd2
  implicit none
  !
  ! .Passed variables.
  INTEGER NAT,NAT3,NATOM3,NQM,ISPACE
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG

  real(chm_real) :: X(:), Y(:), Z(:)
  real(chm_real) FX(NQM,NQM),FS(NQM,NQM),DD1(ISPACE)
  LOGICAL LPRF
  INTEGER I,J,M,II,JJ,IINDEX,JINDEX
  INTEGER IATOM,JATOM
  real(chm_real) FINI

  real(chm_real),allocatable,dimension(:) :: XREF
  real(chm_real),allocatable,dimension(:) :: YREF
  real(chm_real),allocatable,dimension(:) :: ZREF
  real(chm_real),allocatable,dimension(:) :: DXF
  real(chm_real),allocatable,dimension(:) :: DYF
  real(chm_real),allocatable,dimension(:) :: DZF
  !
  !.Local variables.
  !...Conversion factor for force constants
  real(chm_real) :: FACT=143.94_chm_real
  !
  ! ...Put upper triangle of second derivative matrix into DD1
  !

  II=(NAT3*(NAT3+1))/2
  DO I=1,II
     DD1(I)=0.0
  END DO
  !
  IF(QDRUDE) THEN
     ! Drude particles are present: Calculate second derivatives through
     ! finite differences
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
     call chmalloc('molvco.src','GETFX1','XREF',NATOM,crl=XREF)
     call chmalloc('molvco.src','GETFX1','YREF',NATOM,crl=YREF)
     call chmalloc('molvco.src','GETFX1','ZREF',NATOM,crl=ZREF)
     call chmalloc('molvco.src','GETFX1','DXF',NATOM,crl=DXF)
     call chmalloc('molvco.src','GETFX1','DYF',NATOM,crl=DYF)
     call chmalloc('molvco.src','GETFX1','DZF',NATOM,crl=DZF)
     call GENSD2(NATOM3,X,Y,Z,XREF,YREF,ZREF,DD1,& 
          DXF,DYF,DZF,FINI,.TRUE., &
          3*NATOM,.FALSE.) ! JZ_UW12
     call chmdealloc('molvco.src','GETFX1','XREF',NATOM,crl=XREF)
     call chmdealloc('molvco.src','GETFX1','YREF',NATOM,crl=YREF)
     call chmdealloc('molvco.src','GETFX1','ZREF',NATOM,crl=ZREF)
     call chmdealloc('molvco.src','GETFX1','DXF',NATOM,crl=DXF)
     call chmdealloc('molvco.src','GETFX1','DYF',NATOM,crl=DYF)
     call chmdealloc('molvco.src','GETFX1','DZF',NATOM,crl=DZF)
  ELSE
     ! General (non-Drude) case: Analytical second derivatives
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DD1)
  ENDIF
  !
#if KEY_PARALLEL==1
  CALL VDGBR(DX,DY,DZ,1)
  CALL GCOMB(DD1,II)
#endif 
  !
  !... Unpack DD1 into FX and remove Drudes
  !
  II=0
  IINDEX=0
  DO I=1,NATOM3
     IATOM = (I-1)/3 + 1
     IF(.NOT.ISDRUDE(IATOM).AND.IMOVE(IATOM).EQ.0) THEN
        JINDEX=IINDEX
        IINDEX=IINDEX+1
        DO J=I,NATOM3
           JATOM = (J-1)/3 + 1
           II=II+1
           IF(.NOT.ISDRUDE(JATOM).AND.IMOVE(JATOM).EQ.0) THEN
              JINDEX=JINDEX+1
              FX(IINDEX,JINDEX)=DD1(II)
           ENDIF
        END DO
     ELSE
        II=II+NATOM3-I+1
     ENDIF
  END DO
  !
  !... Symmetrize FX
  !
  DO I=1,NAT3
     II=I+1
     DO J=II,NAT3
        FX(J,I)=FX(I,J)
     END DO
  END DO
  !
  !... test printout
  IF(LPRF) THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' GETFX1: the cartesian force constants FX'
     WRITE(OUTU,*) ' symmetrized, unpermuted, in kcal/(mol*A**3)'
     WRITE(OUTU,*)
     DO I=1,NAT3
        WRITE(OUTU,1010) I,(FX(I,J),J=1,NAT3)
     END DO
  END IF
  !
  !...Permute rows and columns (corresponds to change in ordering of X,Y,Z
  !...between CHARMM and MOLVIB
  !
  II=0
  DO I=1,NAT
     DO M=1,3
        II=II+1
        IINDEX=(M-1)*NAT+I
        DO J=1,NAT3
           FS(IINDEX,J)=FX(II,J)
        END DO
     END DO
  END DO
  !
  JJ=0
  DO J=1,NAT
     DO M=1,3
        JJ=JJ+1
        JINDEX=(M-1)*NAT+J
        DO I=1,NAT3
           FX(I,JINDEX)=FS(I,JJ)
        END DO
     END DO
  END DO
  !
  !... Change units - from CHARMM: kcal/(mol*A**3) to MOLVIB: aJ/A**3
  !
  DO I=1,NAT3
     DO J=1,NAT3
        FX(I,J)=FX(I,J)/FACT
     END DO
  END DO
  !
  ! ...test printout
  IF(LPRF) THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' GETFX1: the cartesian force constants FX'
     WRITE(OUTU,*) '   symmetrized and permuted, in aJ/A**3'
     WRITE(OUTU,*)
     DO I=1,NAT3
        WRITE(OUTU,1010) I,(FX(I,J),J=1,NAT3)
     END DO
  END IF
  !
1010 FORMAT(1X,I4,2X,8F9.4,12(/7X,8F9.4))
  !
  RETURN
END SUBROUTINE GETFX1
#endif /*  MOLVIB*/
end module molvco_m

