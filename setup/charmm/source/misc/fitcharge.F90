module fitchg
  use chm_kinds
  implicit none

contains

#if KEY_FITCHG==0 /*fitchg*/
SUBROUTINE FITCHARGE
  !-----------------------------------------------------------------------
  CALL WRNDIE(-1,'<FITCHARGE>','FITCHG code is not compiled.')
  return
end SUBROUTINE FITCHARGE

#else /* (fitchg)*/

SUBROUTINE FITCHARGE
  !-----------------------------------------------------------------------
  !
  !  Purpose:
  !     This SUBROUTINE adjusts atomic charges from electrostatic potential.
  !     The total charge of every _residue_ is preserved.
  !
  !  Author:
  !     Guillaume Lamoureux (2000-2001)
  !  Modifications:
  !     Victor Anisimov (2003-2005)
  !
  !  Usage:
  !     FITCHARGE -
  !         [TEST] -
  !         [{ EQUIvalent <atom_selection> }] -
  !         [ RESTraint [RESP|REPD] [PARAbolic|HYPErbolic] -
  !           [BHYP <REAL>] [DHYP <REAL>] -
  !           [FLAT <REAL>] [DFLAt <REAL>] ] -
  !         -
  !         <atom_selection_1> <atom_selection_2> -
  !         [NITEr <int>] [TOLErance <REAL>] [COUNter <int>] -
  !         UPOT <int> [UOUT <int>] -
  !         -
  !         [NCONf  <int>] -   Multi-conformations fit
  !         [NPERt  <int> [...]] -   Multi-perturbations fit
  !         [UPPOt  <int>] -
  !         [UCOOrd <int>] [ALTInput] -
  !         [UPRT   <int>] -    CHARMM stream file unit for final printout
  !         [TPRT   <int>] -    CHARMM stream file unit for thole printout
  !         [ASCAle <REAL>] -   Polarizability (ALPHA) scaling factor
  !         [RDIPole <REAL>]    Reference dipole for charge scaling
  !         [VTHOLE]            keyword implies that thole radii are also fitted
  !
  !     The EQUI block allows explicit equivalences to be stated.
  !     For each EQUI keyword, the selected atoms are made equivalent.
  !
  !     The first atom selection specifies the atoms to fit. If no atoms
  !     are selected, it is assumed that the potential should be
  !     computed as is, without any charge adjustment.
  !
  !     The second atom selection specifies the atoms contributing to the
  !     electrostatic potential.
  !
  !     All other (non-selected) atoms are contributing to the potential
  !     energy, and are considered as a perturbation.
  !
  !     For non-polarizable molecules (i.e. no Drude oscillators), the
  !     command simplifies to:
  !     FITCHARGE -
  !         [{ EQUIvalent <atom_selection> }] -
  !         [ RESTraint [RESP|REPD] [PARAbolic|HYPErbolic] -
  !           [B <REAL>] [FLAT <REAL>] ] -
  !         -
  !         <atom_selection_1> <atom_selection_2> -
  !         [NITEr <int>] [TOLErance <REAL>] [COUNter <int>] -
  !         UPOT <int> [UOUT <int>] -
  !         [NCONf <int>] -
  !         [UCOOrd <int>] [ALTInput] -
  !         [UPRT   <int>] -
  !         [TPRT   <int>] -
  !         [ASCAle <REAL>] -
  !         [RDIPole <REAL>]
  !
  !     ------------------------------------------------------------------
  !     Keyword    Default
  !     ------------------------------------------------------------------
  !     TEST        no      Set to "yes" performs test on CHARMM and QM MESP;
  !                         No fitting will be performed.
  !
  !     EQUIValent          All atoms in the selection are made equivalent.
  !                         This block can be repeated.
  !
  !     RESTraint           Switches on restraints on charges.
  !                         The charges are restrainted to their actual values.
  !                         The restriction forces are taken from WMAIN.
  !
  !     RESP       yes      A modified RESP restraint
  !                         (Bayly et al, JPC 97 (40), 10269, 1993)
  !     REPD                The REPD restraint of Henchman and Essex
  !                         (J.Comp.Chem. 20 (5), 483, 1999)
  !
  !     PARAbolic           Parabolic shape: S(q) = q^2
  !                         The default shape for REPD.
  !
  !     HYPErbolic          Hyperbolic shape: S(q) = sqrt(q^2 + B^2) - B
  !                         The default shape for RESP.
  !
  !     BHYP       0.1      The LHYPER stiffness (in electrons)
  !     DHYP       BHYP     ... for the Drudes
  !
  !     FLAT       0        Creates a zero penalty from -FLAT to +FLAT
  !     DFLAt      FLAT     ... for the Drudes
  !
  !     NITEr      50       Maximum number of Levenberg-Marquardt iterations
  !
  !     TOLErance  1.0E-4   Relative tolerance on the convergence of the
  !                         chi^2 (for Levenberg-Marquardt algorithm)
  !
  !     COUNter    2        Number of iterations under the tolerance it
  !                         needs to converge
  !
  !     UPOT                Input unit for the grid and the unperturbed
  !                         potential. The FORMAT should be:
  !                            Number of lines:  NGRID(ICONF)
  !                            Format:  Xgrid Ygrid Zgrid Potential (4f15.6)
  !                         For NPERT>1, units UPOT+1, UPOT+2, ..., UPOT+NPERT-1
  !                         will also be read. They should have been open.
  !
  !     UOUT       OUTU     Output unit for the unperturbed potential
  !                         (due to the first selection). The FORMAT is:
  !                         For NPERT=1:
  !                            Number of lines:  sum_ICONF NGRID(ICONF)
  !                            Format:  Xgrid Ygrid Zgrid Pot MPot (5f15.6)
  !                         For NPERT>1:
  !                            Number of lines:  sum_ICONF sum_ipert
  !                                              npert(ICONF)*NGRID(ICONF)
  !                            Format:  Pot MPot (2f15.6)
  !
  !     NCONf      1        Number of molecular conformations.
  !                         NCONf INTEGERs should be following, each one
  !                         being the number of perturbed geometries of the
  !                         conformation.
  !
  !     NPERt      1        Number of perturbations for each conformation.
  !
  !     UPPOt               Input unit for the perturbed potential.
  !                         The FORMAT is:
  !                            Number of lines:  npert(ICONF)*NGRID(ICONF)
  !                            Format:  Potential (1f15.6)
  !                         For NCONF>1, units UPPOT+1, UPPOT+2, ...,
  !                         UPPOT+NCONF-1
  !                         will also be read.
  !
  !     ALTInput            Switches on the alternative input for coordinates.
  !                         (Useful when many conformations/perturbations are
  !                         to be read.)
  !                         In this mode, the coordinates of the atoms of the
  !                         second selection are read from UCOOrd, for each
  !                         conformation and perturbation.
  !
  !     UCOOrd     -1       First coordinates file unit.
  !                         If ALTINPUT is OFF:
  !                            Coordinates are in CHARMM FORMAT.
  !                            ConFORMATion 1: units UCOORD to UCOORD+NPERT(1)-1
  !                            ConFORMATion 2: units UCOORD+NPERT(1)
  !                                            to UCOORD+NPERT(1)+NPERT(2)-1
  !                            etc.
  !                         If ALTINPUT is ON:
  !                            All coordinates are in UCOORD, in XYZ FORMAT.
  !
  !     UPRT       -1       Final charge and polarizability printout unit.
  !                            The CHARMM stream file will be created.
  !
  !     ASCAle      0.0     Polarizability (ALPHA) scaling factor
  !
  !     RDIPole     0.0     Reference dipole for charge scaling
  !
  !     THOLECT    THOLEI   Fill array of variable thole's based on chem types
  !
  !     DRUDEY     FALSE    Fill array id'ing chem types that have a drude
  !
  !    KDRUDEFIT    0.0     Fill atom length array with drude force constants
  !
  !     VTHOLE     FALSE    If true then the thole radii used for shielding the
  !                         electrostatic interactions of 1-2, 1-3 pairs are also fitted
  !-----------------------------------------------------------------------

  use comand
  use coord
  use dimens_fcm
  use memory
  use number
  use param
  use psf
  use stream
  use string

  implicit none

  !     ... Local variables
  integer,allocatable,dimension(:) :: ISLCT1
  integer,allocatable,dimension(:) :: ISLCT2
  integer,allocatable,dimension(:) :: EQUIV
  integer,allocatable,dimension(:) :: CONSTC
  integer,allocatable,dimension(:) :: FITC
  integer,allocatable,dimension(:) :: MULTIP
  integer,allocatable,dimension(:) :: CONSTR
  integer,allocatable,dimension(:) :: AFRAGM
  real(chm_real),allocatable,dimension(:) :: THOLECT
  logical,allocatable,dimension(:) :: DRUDEY
  real(chm_real),allocatable,dimension(:) :: KDRUDEFIT
  real(chm_real),allocatable,dimension(:) :: MPOT
  real(chm_real),allocatable,dimension(:) :: APOT
  real(chm_real),allocatable,dimension(:) :: mpot0
  real(chm_real),allocatable,dimension(:) :: apot0
  real(chm_real),allocatable,dimension(:) :: GRIDX
  real(chm_real),allocatable,dimension(:) :: GRIDY
  real(chm_real),allocatable,dimension(:) :: GRIDZ

  INTEGER NCONF, UPOT, UPPOT
  INTEGER MXNGRD, MXNGEO, ICONF, IGRID, IUNIT, IGEO, NGEO
  LOGICAL FITALP
  !
  !     ...allocate stack
  call chmalloc('fitcharge.src','FITCHARGE','ISLCT1',NATOM,intg=ISLCT1)
  call chmalloc('fitcharge.src','FITCHARGE','ISLCT2',NATOM,intg=ISLCT2)
  call chmalloc('fitcharge.src','FITCHARGE','EQUIV',NATOM,intg=EQUIV)
  call chmalloc('fitcharge.src','FITCHARGE','CONSTC',NATOM,intg=CONSTC)
  call chmalloc('fitcharge.src','FITCHARGE','FITC',NATOM,intg=FITC)
  call chmalloc('fitcharge.src','FITCHARGE','MULTIP',NATOM,intg=MULTIP)
  call chmalloc('fitcharge.src','FITCHARGE','CONSTR',NATOM,intg=CONSTR)
  call chmalloc('fitcharge.src','FITCHARGE','AFRAGM',NATOM*NATOM,intg=AFRAGM)
  call chmalloc('fitcharge.src','FITCHARGE','THOLECT',NATC,crl=THOLECT)
  call chmalloc('fitcharge.src','FITCHARGE','DRUDEY',NATC,log=DRUDEY)
  call chmalloc('fitcharge.src','FITCHARGE','KDRUDEFIT',NATOM,crl=KDRUDEFIT)
  !
  !          Parsing UPOT keyword
  !
  UPOT = GTRMI(COMLYN,COMLEN,'UPOT',-1)
  IF(UPOT.EQ.-1) THEN
     CALL WRNDIE(-4,'FITCHARGE','Unspecified UPOT')
  ELSE
     IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
          'FITCHARGE> Potential unit (UPOT):',UPOT
  ENDIF
  !
  !          Parsing NCONF keyword
  !
  NCONF=GTRMI(COMLYN,COMLEN,'NCON',1)
  IF(NCONF.LT.1) THEN
     CALL WRNDIE(-4,'FITCHARGE','NCONF too low!')
  ENDIF
  IF(NCONF.GT.1) THEN
     IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3,A,I3)') &
          'FITCHARGE> Additional potential units (UPOT):', &
          UPOT+1,' to',UPOT+NCONF-1
  ENDIF
  !
  !          Parsing UPPOT keyword
  !
  UPPOT = GTRMI(COMLYN,COMLEN,'UPPO',-1)
  IF(UPPOT.EQ.-1) THEN
     FITALP=.FALSE.
     !           Non-polarizable fitting
  ELSE
     IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
          'FITCHARGE> Perturbed potential unit (UPPO):',UPPOT
     !           Polarizability will be determined
     FITALP=.TRUE.
  ENDIF
  !
  !          Get number of grid points from UPOT
  !          Get number of ions from UPPOT
  !
  MXNGRD = 0
  MXNGEO = 0
  DO ICONF=1,NCONF
     !          grid coordinates
     IUNIT=UPOT+ICONF-1
     IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
          'FITCHARGE> Read potential from unit',IUNIT
     IGRID = LNCOUNT(IUNIT)
     IF(IGRID.GT.MXNGRD) MXNGRD = IGRID
     IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
          '           Number of gridpoints:',IGRID
     !          perturbed potential
     IF(UPPOT.GT.0) THEN
        IUNIT=UPPOT+ICONF-1
        IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
             'FITCHARGE> Read perturbed potential from unit',IUNIT
        IGEO = LNCOUNT(IUNIT)
        NGEO = IGEO/IGRID
        IF(NGEO*IGRID .NE. IGEO) THEN
           CALL WRNDIE(-4,'FITCHARGE','Wrong number of ions!')
        ENDIF
        IF(NGEO.GT.MXNGEO) MXNGEO = NGEO
     ENDIF
  ENDDO
  MXNGEO = MXNGEO * NCONF
  !
  !          Allocate grid and potential arrays
  call chmalloc('fitcharge.src','FITCHARGE','MPOT',MXNGRD*MXNGEO,crl=MPOT)
  call chmalloc('fitcharge.src','FITCHARGE','APOT',MXNGRD*MXNGEO,crl=APOT)
  call chmalloc('fitcharge.src','FITCHARGE','mpot0',MXNGRD*NCONF,crl=mpot0)
  call chmalloc('fitcharge.src','FITCHARGE','apot0',MXNGRD*NCONF,crl=apot0)
  call chmalloc('fitcharge.src','FITCHARGE','GRIDX',MXNGRD*NCONF,crl=GRIDX)
  call chmalloc('fitcharge.src','FITCHARGE','GRIDY',MXNGRD*NCONF,crl=GRIDY)
  call chmalloc('fitcharge.src','FITCHARGE','GRIDZ',MXNGRD*NCONF,crl=GRIDZ)
  mpot0 = ZERO
  !
  call DO_FIT(NATOM,ISLCT1,ISLCT2,EQUIV,CONSTC,FITC, &
       MULTIP,CONSTR,AFRAGM,THOLECT,DRUDEY,KDRUDEFIT, &
       NCONF,UPOT,UPPOT,FITALP,MXNGRD,MPOT,APOT,mpot0,apot0, &
       GRIDX,GRIDY,GRIDZ,MXNGEO)
  !
  !     Update Thole FUNCTION and relax Drudes upon leaving FITCHARGE
  call SCFDIP(KDRUDEFIT)
  !
  !     ...free stack
  call chmdealloc('fitcharge.src','FITCHARGE','ISLCT1',NATOM,intg=ISLCT1)
  call chmdealloc('fitcharge.src','FITCHARGE','ISLCT2',NATOM,intg=ISLCT2)
  call chmdealloc('fitcharge.src','FITCHARGE','EQUIV',NATOM,intg=EQUIV)
  call chmdealloc('fitcharge.src','FITCHARGE','CONSTC',NATOM,intg=CONSTC)
  call chmdealloc('fitcharge.src','FITCHARGE','FITC',NATOM,intg=FITC)
  call chmdealloc('fitcharge.src','FITCHARGE','MULTIP',NATOM,intg=MULTIP)
  call chmdealloc('fitcharge.src','FITCHARGE','CONSTR',NATOM,intg=CONSTR)
  call chmdealloc('fitcharge.src','FITCHARGE','AFRAGM',NATOM*NATOM,intg=AFRAGM)
  call chmdealloc('fitcharge.src','FITCHARGE','THOLECT',NATC,crl=THOLECT)
  call chmdealloc('fitcharge.src','FITCHARGE','DRUDEY',NATC,log=DRUDEY)
  call chmdealloc('fitcharge.src','FITCHARGE','KDRUDEFIT',NATOM,crl=KDRUDEFIT)
  call chmdealloc('fitcharge.src','FITCHARGE','MPOT',MXNGRD*MXNGEO,crl=MPOT)
  call chmdealloc('fitcharge.src','FITCHARGE','APOT',MXNGRD*MXNGEO,crl=APOT)
  call chmdealloc('fitcharge.src','FITCHARGE','mpot0',MXNGRD*NCONF,crl=mpot0)
  call chmdealloc('fitcharge.src','FITCHARGE','apot0',MXNGRD*NCONF,crl=apot0)
  call chmdealloc('fitcharge.src','FITCHARGE','GRIDX',MXNGRD*NCONF,crl=GRIDX)
  call chmdealloc('fitcharge.src','FITCHARGE','GRIDY',MXNGRD*NCONF,crl=GRIDY)
  call chmdealloc('fitcharge.src','FITCHARGE','GRIDZ',MXNGRD*NCONF,crl=GRIDZ)
  !
  RETURN
END SUBROUTINE FITCHARGE

INTEGER FUNCTION LNCOUNT(IUNIT)
  INTEGER IUNIT
  CHARACTER(len=256) STR
  LNCOUNT = 0
100 READ(IUNIT,'(A80)',ERR=200,END=300) STR
  LNCOUNT = LNCOUNT + 1
  GOTO 100
200 CALL WRNDIE(-4,'FITCHARGE','Unavailable unit!')
300 CONTINUE
  REWIND(UNIT=IUNIT)
  RETURN
END FUNCTION LNCOUNT


SUBROUTINE DO_FIT(NATOM_,ISLCT1,ISLCT2, &
     EQUIV,CONSTC,FITCL,MULTIP,CONSTR,AFRAGM, &
     THOLECT,DRUDEY,KDRUDEFIT,NCONF,UPOT,UPPOT,FITALP,MXNGRD, &
     MPOT,APOT,mpot0,apot0,GRIDX,GRIDY,GRIDZ,MXNGEO)
  !
  !     Engine for FITCHARGE
  !
  use bases_fcm
  use dimens_fcm
  use memory
  use number
  use select
  use stream
  use string
  use comand
  use psf
  use coord
  use consta

  !     ... Local parameters

  INTEGER MXNGEO,MXNGRD
  INTEGER, PARAMETER :: MXNCONF=10

  !     ... Arguments
  INTEGER NATOM_
  INTEGER ISLCT1(NATOM_), ISLCT2(NATOM_)
  INTEGER EQUIV(NATOM_)
  !        For two equivalent atoms i and j, we have
  !        EQUIV(i) = equiv(j). Missing values are -1.
  INTEGER CONSTC(*)
  INTEGER FITCL(*)
  INTEGER MULTIP(*)
  INTEGER CONSTR(*)
  INTEGER AFRAGM(NATOM_,NATOM_)
  real(chm_real)  THOLECT(*)
  LOGICAL DRUDEY(*)
  real(chm_real)  KDRUDEFIT(*)
  !     ... Local variables
  INTEGER I,J
  !        Number of selected atoms
  INTEGER NSEL
  INTEGER IMODE
  !        First coordinates unit
  INTEGER UCOO
  INTEGER NBEGIN,NSKIP,NSTOP
  !        Input/output units
  INTEGER UPOT,UPPOT,UOUT,UPRT,TPRT
  LOGICAL ALTINP

  !     ... Local variables
  !        Number of conformations
  INTEGER NCONF
  INTEGER ICONF
  !        Number of perturbed geometries for each conformation
  INTEGER NPERT_(MXNCONF)
  INTEGER IPERT
  !        Total number of geometries
  INTEGER NGEO
  INTEGER IGEO

  !        Number of gridpoints for each conformation
  INTEGER NGRID(MXNCONF)
  INTEGER IGRID
  !        stack pointers
  real(chm_real)  GRIDX(MXNGRD,NCONF)
  real(chm_real)  GRIDY(MXNGRD,NCONF)
  real(chm_real)  GRIDZ(MXNGRD,NCONF)
  !        Unperturbed ab initio potential for each conformation
  real(chm_real)  apot0(MXNGRD,NCONF)
  !        Unperturbed model potential for each conformation
  real(chm_real)  mpot0(MXNGRD,NCONF)
  !        Perturbed ab initio potential
  real(chm_real)  APOT(MXNGRD,MXNGEO)
  !        Perturbed model potential
  real(chm_real)  MPOT(MXNGRD,MXNGEO)

  LOGICAL FITALP, TEST, QVTHOLE
  LOGICAL LRESTR, RESP,REPD, LPARAB,LHYPER
  real(chm_real)  BHYPER,DHYPER, FLAT,DFLAT
  real(chm_real)  WRESTR(NATOM_)
  !     ... Local variables for EQUI statements
  !        Index counting EQUI keywords
  INTEGER IEQUIV
  INTEGER NTYPES,NCONSTR,CTYPES,TTYPES
  INTEGER EQNUM1,EQNUM2
  !        for Levenberg-Marquardt algorithm
  INTEGER NITER
  real(chm_real)  TOLERA
  INTEGER COUNTER
  !
  real(chm_real)  ASCALE, RDIPOL, RELERR, ABSERR
  INTEGER NFRAGM,NHATFR,NCATFR
  real(chm_real)  AWEIGHT, AXX, AYY, AZZ
  !
  !        stack pointers  size: (NATOM,NGEO)
  real(chm_real),allocatable,dimension(:) :: ALLX
  real(chm_real),allocatable,dimension(:) :: ALLY
  real(chm_real),allocatable,dimension(:) :: ALLZ
  real(chm_real),allocatable,dimension(:) :: GRIDX1
  real(chm_real),allocatable,dimension(:) :: GRIDY1
  real(chm_real),allocatable,dimension(:) :: GRIDZ1

  TEST=INDXA(COMLYN,COMLEN,'TEST').GT.0
  IF(PRNLEV.GT.2) THEN
     IF(TEST) THEN
        WRITE(OUTU,'(A)')  &
             'FITCHARGE> Charge and polarizability test mode'
     ELSE
        WRITE(OUTU,'(A)')  &
             'FITCHARGE> Atomic charges adjustment'
     ENDIF
  ENDIF

  DO I=1,NATOM
     WRESTR(I) = WMAIN(i)
     KDRUDEFIT(I) = ZERO
  ENDDO

  DO I=1,NATOM
     IF(ISDRUDE(I)) THEN
        KDRUDEFIT(I-1) = CCELEC*CG(I)**2/(2*abs(ALPHADP(I-1)))
     ENDIF
  ENDDO
  ASCALE=GTRMF(COMLYN,COMLEN,'ASCA',ZERO)
  RDIPOL=GTRMF(COMLYN,COMLEN,'RDIP',ZERO)
  AWEIGHT=GTRMF(COMLYN,COMLEN,'AWEI',ZERO)
  AXX=GTRMF(COMLYN,COMLEN,'AXX',ZERO)
  AYY=GTRMF(COMLYN,COMLEN,'AYY',ZERO)
  AZZ=GTRMF(COMLYN,COMLEN,'AZZ',ZERO)

  !
  !     Parsing EQUI block
  !
  ! Reset equivalences to all-unique state
  DO I=1,NATOM
     EQUIV(I)=-1
     ! Reset zero-charge fragments
     DO J=1,NATOM
        AFRAGM(J,I)=0
     ENDDO
  ENDDO
  !
  ! Obtain zero-charge constrained fragments
  !
  NFRAGM=0
  DO I=1,NATOM
     IF(INDXA(COMLYN,COMLEN,'FRAG').GT.0) THEN
        !               Selection of fragment atoms
        IMODE = 0
        CALL SELRPN(COMLYN,COMLEN,ISLCT1,NATOMT,0,IMODE, &
             .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
             .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
        IF(IMODE.NE.0) THEN
           CALL WRNDIE(0,'<SELRPN>','Fragment selection error')
           GOTO 20
        ENDIF
        NFRAGM=NFRAGM+1
        !
        !               Read in carbon atom number
        NCATFR=0
        DO J=1,NATOM
           IF(ISLCT1(J).EQ.1 .AND.INDEX(ATYPE(J),'C').GT.0) THEN
              NCATFR=NCATFR+1
              AFRAGM(NCATFR+1,NFRAGM)=J
           ENDIF
        ENDDO
        IF(NCATFR.NE.1) THEN
           CALL WRNDIE(0,'<FRAGMN>', &
                'One Carbon atom must be present in fragment selection')
           NFRAGM=NFRAGM-1
           GOTO 20
        ENDIF
        !
        !               Read in hydrogen atom numbers
        NHATFR=0
        DO J=1,NATOM
           IF(ISLCT1(J).EQ.1 .AND.INDEX(ATYPE(J),'H').GT.0) THEN
              NHATFR=NHATFR+1
              AFRAGM(NHATFR+2,NFRAGM)=J
           ENDIF
        ENDDO
        IF(NHATFR.GE.1) THEN
           AFRAGM(1,NFRAGM)=NHATFR
        ELSE
           CALL WRNDIE(0,'<FRAGMN>', &
                'No hydrogen atoms specified in selection')
           NFRAGM=NFRAGM-1
           GOTO 20
        ENDIF
     ELSE
        GOTO 30
     ENDIF
20   CONTINUE
  ENDDO
30 CONTINUE
  !
  ! Some printing
  IF(NFRAGM.GT.0) THEN
     WRITE(OUTU,'(A)')  &
          'Zero-charge constraint for alkanes will be applied to:'
     DO I=1,NFRAGM
        WRITE(OUTU,'(A,I3,A,A4,A,20A4)')  &
             'Fragment:',I,'  Carbon= ',ATYPE(AFRAGM(2,I)), &
             '  Hydrogens= ',(ATYPE(AFRAGM(J,I)),J=3,AFRAGM(1,I)+2)
     ENDDO
  ENDIF
  !
  ! Build the equivalence list
  !     NATOM is maximum possible number of equivalence groups/classes
  !     Equivalence (class/group) numbers will start from 1
  EQNUM1=1
  DO IEQUIV=1,NATOM
     IF(INDXA(COMLYN,COMLEN,'EQUI').GT.0) THEN
        !               Selection of equivalent atoms
        IMODE = 0
        CALL SELRPN(COMLYN,COMLEN,ISLCT1,NATOMT,0,IMODE, &
             .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
             .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
        IF(IMODE.NE.0) THEN
           CALL WRNDIE(0,'<SELRPN>','Atom selection parsing error')
        ENDIF
        !
        !           Fill in "equiv" list
        DO I=1,NATOM
           IF(ISLCT1(I).EQ.1) THEN
              !                 Atom (I) is in the selection
              IF(EQUIV(I).NE.-1) THEN
                 !                    The atom is already equivalent to some
                 !                    others, so we update the equivalence classes
                 !                    of those others.
                 EQNUM2 = EQUIV(I)
                 DO J=1,NATOM
                    IF(EQUIV(J).EQ.EQNUM2) THEN
                       EQUIV(J)=EQNUM1
                    ENDIF
                 ENDDO
              ELSE
                 EQUIV(I)=EQNUM1
              ENDIF
           ENDIF
        ENDDO
        EQNUM1=EQNUM1+1
     ELSE
        !           No more equivalences defined
        GOTO 50
     ENDIF
  ENDDO
50 CONTINUE
  !
  ! Assign unique equivalence numbers to each remaining atom
  DO I=1,NATOM
     IF(EQUIV(I).EQ.-1) THEN
        EQUIV(I)=EQNUM1
        EQNUM1=EQNUM1+1
     ENDIF
  ENDDO
  WRITE(OUTU,'(A)') 'Equivalence classes:'
  WRITE(OUTU,'(A)') '    Index  TYPE  MTYPE    Class'
103 FORMAT(3X,I7,A6,I6,5X,I4)
  DO I=1,NATOM
     WRITE(OUTU,103) I,ATYPE(i),IAC(i),EQUIV(I)
  ENDDO
  !
  !     Parsing RESTRAINT switch
  !
  LRESTR=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'REST').GT.0) LRESTR=.TRUE.
  !
  !     Parsing RESTRAINT options
  !
  RESP=.TRUE.
  REPD=.FALSE.
  LPARAB=.TRUE.
  LHYPER=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'RESP').GT.0) THEN
     !        Default options are good
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'REPD').GT.0) THEN
     RESP=.FALSE.
     REPD=.TRUE.
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'PARA').GT.0) THEN
     LPARAB=.TRUE.
     LHYPER=.FALSE.
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'HYPE').GT.0) THEN
     LPARAB=.FALSE.
     LHYPER=.TRUE.
  ENDIF
  BHYPER=GTRMF(COMLYN,COMLEN,'BHYP',PTONE)
  DHYPER=GTRMF(COMLYN,COMLEN,'DHYP',BHYPER)
  FLAT=GTRMF(COMLYN,COMLEN,'FLAT',ZERO)
  DFLAT=GTRMF(COMLYN,COMLEN,'DFLA',FLAT)
  !
  !     Some printing
  IF(LRESTR.AND.PRNLEV.GT.2) THEN
     WRITE(OUTU,'(A)') 'FITCHARGE> Charge restraints'
     IF(REPD) THEN
        WRITE(OUTU,'(A)') '           REPD scheme'
     ELSE
        WRITE(OUTU,'(A)') '           RESP-like scheme'
     ENDIF
     IF(LPARAB) THEN
        WRITE(OUTU,'(A)') '           Parabolic restraint shape'
     ELSEIF(LHYPER) THEN
        WRITE(OUTU,'(A)') '           Hyperbolic restraint shape'
        WRITE(OUTU,'(A,F12.6)') '             BHYP =',BHYPER
        WRITE(OUTU,'(A,F12.6)') '             DHYP =',DHYPER
     ENDIF
     IF(FLAT.NE.ZERO) &
          WRITE(OUTU,'(A,F12.6)') '           FLAT  =',FLAT
     IF(DFLAT.NE.ZERO) &
          WRITE(OUTU,'(A,F12.6)') '           DFLAT =',DFLAT
  ENDIF
  !
  !     Parsing first selection (atoms to fit)
  !
  IMODE = 0
  CALL SELRPN(COMLYN,COMLEN,ISLCT1,NATOMT,1,IMODE, &
       .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
       .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
  IF(IMODE.NE.0) &
       CALL WRNDIE(0,'<SELRPN>','Atom selection parsing error')
  !
  !     Parsing second selection (atoms contributing to the potential)
  !
  IMODE=0
  CALL SELRPN(COMLYN,COMLEN,ISLCT2,NATOMT,1,IMODE, &
       .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
       .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
  IF(IMODE.NE.0) &
       CALL WRNDIE(0,'<SELRPN>','Atom selection parsing error')
  NSEL=0
  DO I=1,NATOM
     IF(ISLCT2(I).EQ.1) NSEL=NSEL+1
  ENDDO
  IF(NSEL.EQ.0) CALL WRNDIE(-4,'FITCHARGE','No atoms selected')
  !
  !     Parsing NITER and TOLER
  !
  NITER=GTRMI(COMLYN,COMLEN,'NITE',50)
  TOLERA= GTRMF(COMLYN,COMLEN,'TOLE',PT0001)
  counter= GTRMI(COMLYN,COMLEN,'COUN',2)
  IF(PRNLEV.GT.2) THEN
     WRITE(OUTU,'(A,F12.6)') &
          'FITCHARGE> Chi^2 tolerance (TOLE)   =',TOLERA
     WRITE(OUTU,'(A,I4)') &
          'FITCHARGE> Tolerance counter (COUN) =',counter
     WRITE(OUTU,'(A,I4)') &
          'FITCHARGE> Maximum number of iterations (NITE) =',niter
  ENDIF
  !
  !     Parsing input/output units for potential
  !
  UOUT=GTRMI(COMLYN,COMLEN,'UOUT',-1)
  UPRT=GTRMI(COMLYN,COMLEN,'UPRT',-1)
  TPRT=GTRMI(COMLYN,COMLEN,'TPRT',-1)
  !
  !     Parsing NPERT line
  !
  NPERT_(1) = GTRMI(COMLYN,COMLEN,'NPER',1)
  IF(NPERT_(1).LT.1) NPERT_(1)=1
  NGEO = NPERT_(1)
  IF(NCONF.GT.1) THEN
     DO ICONF=2,NCONF
        IF(NDRUDE.GT.0) THEN
           !             polarizability fitting
           !             read number of perturbation points for the given conformation
           NPERT_(ICONF) = NEXTI(COMLYN,COMLEN)
           IF(NPERT_(ICONF).LT.1) THEN
              CALL WRNDIE(-4,'FITCHARGE','All NPERT have to be >0 !')
           ENDIF
        ELSE
           !             static charge fitting
           NPERT_(ICONF)=1
        ENDIF
        NGEO = NGEO + NPERT_(ICONF)
     ENDDO
  ENDIF
  IF(NGEO.GT.MXNGEO) THEN
     CALL WRNDIE(-4,'FITCHARGE','MXNGEO too low!')
  ENDIF
  !
  !     Parsing of ALTInput
  !
  ALTINP=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'ALTI').GT.0) ALTINP = .TRUE.
  !
  !     Parsing UCOO keyword
  !
  UCOO=GTRMI(COMLYN,COMLEN,'UCOO',-1)
  IF(NGEO.GT.1) THEN
     !           We have to read external coordinates
     IF(UCOO.EQ.-1) CALL WRNDIE(-4,'FITCHARGE','Unspecified UCOO')
     IF(PRNLEV.GT.2) THEN
        IF(ALTINP) THEN
           WRITE(OUTU,'(A)') &
                'FITCHARGE> Alternative input for coordinates'
           WRITE(OUTU,'(A,I3)') &
                'FITCHARGE> Coordinate unit (UCOO): ',UCOO
        ELSE
           WRITE(OUTU,'(A,I3,A,I3)')  &
                'FITCHARGE> Coordinate units (UCOO):', &
                UCOO,' to',UCOO+NGEO-1
        ENDIF
     ENDIF
  ELSE
     !           We use actual coordinates
     IF(PRNLEV.GT.2) WRITE(OUTU,'(A)') &
          'FITCHARGE> Actual coordinates will be used.'
  ENDIF

  call chmalloc('fitcharge.src','DO_FIT','ALLX',natom*ngeo,crl=ALLX)
  call chmalloc('fitcharge.src','DO_FIT','ALLY',natom*ngeo,crl=ALLY)
  call chmalloc('fitcharge.src','DO_FIT','ALLZ',natom*ngeo,crl=ALLZ)
  call chmalloc('fitcharge.src','DO_FIT','GRIDX1',2*MXNGRD,crl=GRIDX1)
  call chmalloc('fitcharge.src','DO_FIT','GRIDY1',2*MXNGRD,crl=GRIDY1)
  call chmalloc('fitcharge.src','DO_FIT','GRIDZ1',2*MXNGRD,crl=GRIDZ1)
  !
  !     Read grid and electrostatic potential from UPOT
  !
  DO ICONF=1,NCONF
     IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
          'FITCHARGE> Read potential from unit', &
          UPOT+ICONF-1
     DO IGRID=1,MXNGRD
        READ(UPOT+ICONF-1,*,ERR=100,END=200) &
             GRIDX(igrid,ICONF), GRIDY(igrid,ICONF), &
             GRIDZ(igrid,ICONF), apot0(igrid,ICONF)
     ENDDO
     GOTO 200
100  CALL WRNDIE(-4,'FITCHARGE','Unavailable unit!')
200  CONTINUE
     NGRID(ICONF) = igrid - 1
     IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
          '           Number of gridpoints:',NGRID(ICONF)
  ENDDO

  !     Examine Equivalences
  CALL EXEQUI(ISLCT1,EQUIV, &
       NTYPES, FITCL, NCONSTR, CONSTC, &
       MULTIP, CONSTR, WRESTR )

  !
  !    Define charge types as ntypes and add thole types to ntypes
  CALL COUNTTH(TTYPES,DRUDEY)

  CTYPES = NTYPES
  QVTHOLE = (INDXA(COMLYN,COMLEN,'VTHO').GT.0)
  IF (QVTHOLE) THEN
     NTYPES = CTYPES + TTYPES
  ENDIF
  !
  !        Read coordinates
  !

  call READCO(ALTINP,NATOM_,ngeo,NCONF,ALLX,ALLY,ALLZ,UCOO)

  IF(.NOT.FITALP) THEN
     !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
     !        Fit REAL atoms only (no Drude oscillators)
     !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
     call DO_ALP(NATOM_,NGEO,GRIDX,GRIDY,GRIDZ,NGRID,NCONF,apot0,mpot0, &
          APOT,MPOT,FITALP,NPERT_,ISLCT1,ISLCT2,EQUIV,NTYPES,CTYPES,CONSTC, &
          NCONSTR,FITCL,MULTIP,CONSTR,ALLX,ALLY,ALLZ,LRESTR,REPD, &
          RESP,LPARAB,LHYPER,BHYPER,DHYPER,FLAT,DFLAT,WRESTR,NITER,TOLERA, &
          counter,UPRT,TPRT,ASCALE,RDIPOL,NFRAGM,AFRAGM,AWEIGHT,AXX,AYY,AZZ, &
          TEST,THOLECT,DRUDEY,KDRUDEFIT,QVTHOLE,MXNGRD,GRIDX1,GRIDY1, &
          GRIDZ1,MXNGEO)

     !        Write gridpoints and electrostatic potentials
105  FORMAT(3F12.6,2F14.8)
     IF(UOUT.GT.0) THEN
        IF(TEST) THEN
           IGEO=1
           IPERT=1
           DO ICONF=1,NCONF
              DO IGRID=1,NGRID(ICONF)
                 WRITE(UOUT,501) IGRID, &
                      apot0(IGRID,IGEO),mpot0(IGRID,IGEO), &
                      APOT(IGRID,IPERT),MPOT(IGRID,IPERT), &
                      APOT(IGRID,IGEO)-apot0(IGRID,IPERT), &
                      MPOT(IGRID,IGEO)-mpot0(IGRID,IPERT)
                 IPERT=IPERT+1
              ENDDO
              IGEO=IGEO+1
           ENDDO
        ELSE
           WRITE(UOUT,'(A11,A12,A12,A13,A14)')  &
                'GRIDX','GRIDY','GRIDZ','QMESP','CHESP'
           DO ICONF=1,NCONF
              WRITE(UOUT,105) (GRIDX(IGRID,ICONF),GRIDY(IGRID,ICONF), &
                   GRIDZ(IGRID,ICONF),apot0(IGRID,ICONF), &
                   mpot0(IGRID,ICONF),IGRID=1,NGRID(ICONF))
           ENDDO
        ENDIF
     ENDIF
  ELSE
     !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
     !        Fit all atoms (including Drude particles)
     !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
     !
     !        Read perturbed potential
     !
     IGEO = 1
     DO ICONF=1,NCONF
        IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
             'FITCHARGE> Read perturbed potential from unit', &
             UPPOT+ICONF-1
        DO IPERT=1,NPERT_(ICONF)
           IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
                '           Read perturbation',ipert
           READ(UPPOT+ICONF-1,*,ERR=300,END=400) &
                (APOT(igrid,IGEO), igrid = 1,NGRID(ICONF))
           IGEO = IGEO + 1
        ENDDO
        GOTO 400
300     CALL WRNDIE(-4,'FITCHARGE','Unavailable unit!')
400     CONTINUE
     ENDDO
     !
     !        Adjust polarizabilities
     !
     call DO_ALP(NATOM_,NGEO,GRIDX,GRIDY,GRIDZ,NGRID,NCONF,apot0,mpot0, &
          APOT,MPOT,FITALP,NPERT_,ISLCT1,ISLCT2,EQUIV,NTYPES,CTYPES,CONSTC, &
          NCONSTR,FITCL,MULTIP,CONSTR,ALLX,ALLY,ALLZ,LRESTR,REPD, &
          RESP,LPARAB,LHYPER,BHYPER,DHYPER,FLAT,DFLAT,WRESTR,NITER,TOLERA, &
          counter,UPRT,TPRT,ASCALE,RDIPOL,NFRAGM,AFRAGM,AWEIGHT,AXX,AYY,AZZ, &
          TEST,THOLECT,DRUDEY,KDRUDEFIT,QVTHOLE,MXNGRD,GRIDX1,GRIDY1, &
          GRIDZ1,MXNGEO)
     !
     !        Write electrostatic potentials
     !
501  FORMAT(I6, 3(2X,2F8.4))
502  FORMAT(3F10.5, 2F14.8, F10.4, 3I6)
     IF(UOUT.GT.0) THEN
        IF(TEST) THEN
           IGEO=1
           IPERT=1
           DO ICONF=1,NCONF
              DO IGRID=1,NGRID(ICONF)
                 WRITE(UOUT,501) IGRID, &
                      apot0(IGRID,IGEO),mpot0(IGRID,IGEO), &
                      APOT(IGRID,IPERT),MPOT(IGRID,IPERT), &
                      APOT(IGRID,IGEO)-apot0(IGRID,IPERT), &
                      MPOT(IGRID,IGEO)-mpot0(IGRID,IPERT)
                 IPERT=IPERT+1
              ENDDO
              IGEO=IGEO+1
           ENDDO
        ELSE
           WRITE(UOUT,'(A,A)')  &
                '    GRIDX     GRIDY     GRIDZ       QMESP          CHESP', &
                '      ABSERR  ICONF  IGEO  IGRID'
           !
           !             Unperturbed potential
           IGEO=1
           DO ICONF=1,NCONF
              DO IGRID=1,NGRID(ICONF)
                 IF(abs(apot0(IGRID,IGEO)).GT.1.0D-6) THEN
                    ABSERR=ABS(mpot0(IGRID,IGEO)-apot0(IGRID,IGEO))
                    RELERR=ABS((mpot0(IGRID,IGEO)-apot0(IGRID,IGEO)) / &
                         apot0(IGRID,IGEO))
                 ELSE
                    ABSERR=0.0D0
                    RELERR=0.0D0
                 ENDIF
                 WRITE(UOUT,502) GRIDX(IGRID,ICONF), &
                      GRIDY(IGRID,ICONF),GRIDZ(IGRID,ICONF), &
                      apot0(IGRID,IGEO),mpot0(IGRID,IGEO),ABSERR, &
                      ICONF,0,IGRID
              ENDDO
              IGEO=IGEO+1
           ENDDO
           !
           !             Perturbed potential
           IGEO=1
           DO ICONF=1,NCONF
              DO IPERT=1,NPERT_(ICONF)
                 DO IGRID=1,NGRID(ICONF)
                    IF(abs(APOT(IGRID,IGEO)).GT.1.0D-6) THEN
                       ABSERR=ABS(MPOT(IGRID,IGEO)-APOT(IGRID,IGEO))
                       RELERR=ABS((MPOT(IGRID,IGEO)-APOT(IGRID,IGEO)) / &
                            APOT(IGRID,IGEO))
                    ELSE
                       ABSERR=0.0D0
                       RELERR=0.0D0
                    ENDIF
                    WRITE(UOUT,502) GRIDX(IGRID,ICONF), &
                         GRIDY(IGRID,ICONF),GRIDZ(IGRID,ICONF), &
                         APOT(IGRID,IGEO), MPOT(IGRID,IGEO), ABSERR, &
                         ICONF,IGEO,IGRID
                 ENDDO
                 IGEO=IGEO+1
              ENDDO
           ENDDO
        ENDIF
     ENDIF
     !
  ENDIF
  !
  call chmdealloc('fitcharge.src','DO_FIT','ALLX',natom*ngeo,crl=ALLX)
  call chmdealloc('fitcharge.src','DO_FIT','ALLY',natom*ngeo,crl=ALLY)
  call chmdealloc('fitcharge.src','DO_FIT','ALLZ',natom*ngeo,crl=ALLZ)
  call chmdealloc('fitcharge.src','DO_FIT','GRIDX1',2*MXNGRD,crl=GRIDX1)
  call chmdealloc('fitcharge.src','DO_FIT','GRIDY1',2*MXNGRD,crl=GRIDY1)
  call chmdealloc('fitcharge.src','DO_FIT','GRIDZ1',2*MXNGRD,crl=GRIDZ1)
  !
  RETURN
END SUBROUTINE DO_FIT


SUBROUTINE READCO(ALTINP,NATOM_,ngeo_,NCONF,ALLX,ALLY,ALLZ,UCOO)
  use dimens_fcm
  use comand
  use stream
  use psf
  use coord
  use ctitla
  use coorio_mod,only:cread
  use memory
  !     Arguments
  !  ALTINP               INPUT
  !             .TRUE. to read all coordinates from one file (in XYZ FORMAT)
  !  NATOM_               INPUT
  !  ngeo_                INPUT
  !  NCONF                INPUT
  !  ALLX(NATOM_,ngeo_)   I/O
  !  ALLY(NATOM_,ngeo_)   I/O
  !  ALLZ(NATOM_,ngeo_)   I/O
  !  UCOO                 INPUT   First coordinates unit
  !

  LOGICAL ALTINP
  INTEGER NATOM_
  INTEGER ngeo_
  INTEGER NCONF
  real(chm_real)  ALLX(NATOM_,ngeo_)
  real(chm_real)  ALLY(NATOM_,ngeo_)
  real(chm_real)  ALLZ(NATOM_,ngeo_)
  INTEGER UCOO

  !     Locals
  INTEGER i,IGEO,IUCOO
  INTEGER coord_FORMAT
  INTEGER offset
  INTEGER icntrl(20)
  character(len=80) cr_line
  !        stack pointers
  INTEGER,allocatable,dimension(:) :: free_stack
  INTEGER free_stack_size
  logical lfree
  integer,allocatable,dimension(:) :: ISLCT3
  !
  ! copy CHARMM coordinates X,Y,Z to the fitcharge repository
  !
  IF(UCOO.EQ.-1) THEN
     DO I=1,NATOM_
        ALLX(I,1)=X(I)
        ALLY(I,1)=Y(I)
        ALLZ(I,1)=Z(I)
     ENDDO
     RETURN
  ENDIF
  !
  IF(ALTINP) THEN
     !        Read all coordinates from one file
     DO IGEO=1,NGEO_
        IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3,A,I3)') &
             'FITCHARGE> Read geometry ',IGEO,' from unit ',UCOO
        !
        DO I=1,NATOM_
           ! LPs must be present in FITCHARGE input if only NCONF > 1
           IF(INDEX(ATYPE(I),'LP').GT.0 .AND. NCONF.EQ.1) THEN
              ALLX(I,IGEO)=X(I)
              ALLY(I,IGEO)=Y(I)
              ALLZ(I,IGEO)=Z(I)
              GOTO 400
           ENDIF
           IF(ISDRUDE(I)) THEN
              ALLX(I,IGEO) = ALLX(I-1,IGEO)
              ALLY(I,IGEO) = ALLY(I-1,IGEO)
              ALLZ(I,IGEO) = ALLZ(I-1,IGEO)
              GOTO 400
           ENDIF
           READ(UCOO,*,err=450,END=250) &
                ALLX(I,IGEO),ALLY(I,IGEO),ALLZ(I,IGEO)
           GOTO 400
250        IF(I.EQ.1) THEN
              UCOO=UCOO+1
              READ(UCOO,*,ERR=450,END=350) &
                   ALLX(I,IGEO),ALLY(I,IGEO),ALLZ(I,IGEO)
              GOTO 400
           ENDIF
350        CALL WRNDIE(-4,'FITCHARGE','Unavailable geometry!')
400        CONTINUE
        ENDDO
        GOTO 500
450     CALL WRNDIE(-4,'FITCHARGE','Unavailable unit!')
500     CONTINUE
     ENDDO
  ELSE
     !        Read from separate files; 1 for CARD, -1 for PDB
     !        This mode is not compatible with LPs
     coord_FORMAT=1

     !        Select all atoms
     call chmalloc('fitcharge.src','READCO','ISLCT3',NATOM,intg=ISLCT3)
     ISLCT3 = 1
     offset = 0
     call chmalloc('fitcharge.src','READCO','free_stack',NATOM+1,intg=free_stack)
     free_stack_size = NATOM+1
     lfree=.false.     !MFC do not know what was intended for this variable
     !MFC CREAD should get this variable not free_stack_size
     !MFC do not know how this worked before.....
     IGEO = 1
     DO IUCOO=UCOO,UCOO+NGEO_-1
        IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') &
             'FITCHARGE> Read coordinates in unit',IUCOO
        CALL CREAD(IUCOO,TITLEB,NTITLB,icntrl,X,Y,Z,WMAIN,NATOM, &
             coord_FORMAT,ISLCT3, &
             offset,RES,NRES,ATYPE,IBASE, &
             1,free_stack,SEGID,RESID,NICTOT,NSEGT,.FALSE., &
             lfree,cr_line,80,0,.FALSE.,0)
        !             free_stack_size,cr_line,80,0,.FALSE.)    !MFC incorrect
        DO I=1,NATOM
           IF(ISDRUDE(i)) THEN
              ALLX(I,IGEO)=ALLX(I-1,IGEO)
              ALLY(I,IGEO)=ALLY(I-1,IGEO)
              ALLZ(I,IGEO)=ALLZ(i-1,IGEO)
           ELSE
              ALLX(I,IGEO)=X(I)
              ALLY(I,IGEO)=Y(I)
              ALLZ(I,IGEO)=Z(I)
           ENDIF
        ENDDO
        IGEO = IGEO + 1
     ENDDO

     call chmdealloc('fitcharge.src','READCO','ISLCT3',NATOM,intg=ISLCT3)
     call chmdealloc('fitcharge.src','READCO','free_stack',NATOM+1, &
          intg=free_stack)

  ENDIF

  RETURN
END SUBROUTINE READCO


SUBROUTINE DO_ALP(NATOM_,ngeo_, &
     GRIDX,GRIDY,GRIDZ,NGRID,NCONF, &
     apot0,mpot0,APOT,MPOT,FITALP,NPERT_, ISLCT1,ISLCT2, &
     equiv,NTYPES,CTYPES,CONSTC,NCONSTR,FITCL, &
     MULTIP,CONSTR,ALLX,ALLY,ALLZ,RESTR,REPD,RESP, &
     LPARAB,LHYPER,BHYPER,DHYPER,FLAT,DFLAT,WRESTR, &
     niter,TOLERA,counter,UPRT,TPRT,ASCALE,RDIPOL,NFRAGM,AFRAGM, &
     AWEIGHT,AXX,AYY,AZZ,TEST,THOLECT,DRUDEY,KDRUDEFIT, &
     QVTHOLE,MXNGRD,GRIDX1,GRIDY1,GRIDZ1,MXNGEO)
  !
  !  Purpose:
  !     Fit charges to minimize the residues of the potential of
  !     different geometries (pot) calculated at given gridpoints
  !     (GRIDX,GRIDY,GRIDZ).
  !
  !  Arguments:
  !
  !     GRIDX, GRIDY, GRIDZ
  !                  (input)  REAL array, dimension (NGRID(ICONF),nconf)
  !                           Coordinates of gridpoints for each conformation
  !
  !     NGRID        (input)  INTEGER array, dimension (nconf)
  !                           Number of gridpoints for each conformation
  !
  !     nconf        (input)  INTEGER
  !                           Number of conformations
  !
  !     pot          (input)  REAL array, dimension (NGRID(ICONF),ngeo)
  !                           Ab initio perturbed potential
  !
  !     MPOT         (output) REAL array, dimension (NGRID(ICONF),ngeo)
  !                           Model perturbed potential
  !
  !     NPERT_       (input)  INTEGER
  !                           Number of perturbed geometries for each
  !                           conformation
  !
  !     ISLCT1       (input)  INTEGER array, dimension (NATOM)
  !                           Atoms to fit
  !
  !     ISLCT2       (input)  INTEGER array, dimension (NATOM)
  !                           Atom contributing to the potential
  !
  !     equiv        (input)  INTEGER array, dimension (NATOM)
  !                           Equivalence classes of the atoms
  !
  !     ALLX, ALLY, ALLZ
  !                  (input/output)
  !                           REAL array, dimension (NATOM,ngeo)
  !                           Atomic coordinates of each perturbed geometry
  !
  !     RESTR    (input)  LOGICAL
  !                           .TRUE. if some restraints are to be applied
  !
  !     REPD         (input)  LOGICAL
  !                           .TRUE. to use REPD penalty
  !
  !     resp         (input)  LOGICAL
  !                           .TRUE. to use RESP penalty
  !
  !     LPARAB    (input)  LOGICAL
  !                           .TRUE. for LPARAB shape
  !
  !     LHYPER   (input)  LOGICAL
  !                           .TRUE. for LHYPER shape
  !
  !     BHYPER       (input)  REAL
  !                           b value for the LHYPER shape (in electrons)
  !     DHYPER      (input)  REAL
  !                           idem for Drudes
  !
  !     FLAT         (input)  REAL
  !                           FLAT PARAMETER (in electrons)
  !     DFLAT        (input)  REAL
  !                           idem for Drudes
  !
  !     WRESTR   (input)  REAL array, dimension (NATOM)
  !                           Restraints on charges/polarizabilities
  !
  !     niter        (input)  INTEGER
  !                           Maximum number of L-M iterations
  !
  !     TOLERA    (input)  REAL
  !                           Relative tolerance on the convergence of the chi^2
  !
  !     counter      (input)  INTEGER
  !                           Number of iterations under the tolerance it
  !                           needs to converge
  !
  !     uprt         (input)  INTEGER
  !                           Output unit for final charges and polarizabilities
  !
  !     ascale       (input)  REAL
  !                           Polarizability scaling factor
  !
  !     RDIPOL      (input)  REAL
  !                           Reference dipole moment for charge scaling
  !
  !     TEST        (input)  LOGICAL
  !                           Enter test mode if TEST==.T.; Do not perform fitting;
  !                           CHARMM and QM electrostatic potential will be compared
  !
  use memory
  use stream
  use dimens_fcm
  use psf
  use consta
  use coord

  !     Local parameters
  INTEGER MXNGEO, MXNGRD
  INTEGER, PARAMETER :: MXNCONF=10

  !     Arguments
  INTEGER NATOM_, ngeo_
  INTEGER nconf
  real(chm_real)  GRIDX(MXNGRD,NCONF)
  real(chm_real)  GRIDY(MXNGRD,NCONF)
  real(chm_real)  GRIDZ(MXNGRD,NCONF)
  real(chm_real)  GRIDX1(MXNGRD)
  real(chm_real)  GRIDY1(MXNGRD)
  real(chm_real)  GRIDZ1(MXNGRD)
  INTEGER NGRID(MXNCONF)
  real(chm_real)  apot0(MXNGRD,NCONF)
  real(chm_real)  mpot0(MXNGRD,NCONF)
  real(chm_real)  APOT(MXNGRD,MXNGEO)
  real(chm_real)  MPOT(MXNGRD,MXNGEO)
  INTEGER NPERT_(MXNCONF)
  INTEGER ISLCT1(NATOM_)
  INTEGER ISLCT2(NATOM_)

  INTEGER EQUIV(NATOM_)
  INTEGER NTYPES
  INTEGER CTYPES
  !     INTEGER FITCL(*)      Equivalence classes to fit
  !     INTEGER CONSTC(*)    Equivalence classes to constraint
  !     INTEGER NCONSTR
  !     INTEGER MULTIP(*)   Number of atoms equivalent to the constrained
  !                         atom (including the constrained atom itself)
  INTEGER FITCL(*)
  INTEGER CONSTC(*)
  INTEGER NCONSTR
  INTEGER MULTIP(*)
  INTEGER CONSTR(*)

  real(chm_real)  ALLX(NATOM_,ngeo_)
  real(chm_real)  ALLY(NATOM_,ngeo_)
  real(chm_real)  ALLZ(NATOM_,ngeo_)
  LOGICAL RESTR,REPD,RESP,LPARAB,LHYPER
  real(chm_real)  BHYPER, DHYPER, FLAT, DFLAT
  real(chm_real)  WRESTR(NATOM_)
  INTEGER NITER
  real(chm_real)  TOLERA
  INTEGER counter
  INTEGER UPRT
  INTEGER TPRT
  real(chm_real)  ASCALE,RDIPOL
  real(chm_real)  AWEIGHT,AXX,AYY,AZZ
  INTEGER NFRAGM,AFRAGM(NATOM_,NATOM_)
  LOGICAL TEST, FITALP
  real(chm_real)  THOLECT(*)
  LOGICAL DRUDEY(*)
  real(chm_real)  KDRUDEFIT(*)
  LOGICAL QVTHOLE

  !     Local variables
  INTEGER i, j, igrid
  !        Atom indices
  INTEGER ia, ja
  INTEGER ir, cia
  INTEGER ITYPE, JTYPE
  LOGICAL new_type, new_type_in_res

  !     Local variables for multi-conformation fit
  INTEGER ICONF, IGEO
  LOGICAL ACCUML

  !     Local variables for LM algorithm
  real(chm_real)  lambda
  INTEGER counter2, iter
  real(chm_real)  chi2, chi22, chi2restr, chi2restr2, conv
  INTEGER npoints
  real(chm_real)  CALCG
  !
  real(chm_real),allocatable,dimension(:) :: ALPHA
  real(chm_real),allocatable,dimension(:) :: ALPHA2
  real(chm_real),allocatable,dimension(:) :: beta
  real(chm_real),allocatable,dimension(:) :: beta2
  real(chm_real),allocatable,dimension(:) :: da
  real(chm_real),allocatable,dimension(:) :: EPSILN
  real(chm_real),allocatable,dimension(:) :: DPOT
  !        pointer to target charges
  real(chm_real),allocatable,dimension(:) :: TARGET
  !
  if (prnlev >= 2) then
     WRITE(OUTU,'(A)') ' '
     WRITE(OUTU,'(A)') 'FITCHARGE> Restrained nonlinear regression'
     WRITE(OUTU,'(A)') '           Levenberg-Marquardt algorithm'
     IF(NCONF.GT.1) THEN
        WRITE(OUTU,'(A,I3,A)') '           (for',nconf,' conformations)'
     ENDIF
  endif

  CHI22 = 0.0D0

  !     Keep target charges
  call chmalloc('fitcharge.src','DO_ALP','ALPHA',NTYPES*NTYPES,crl=ALPHA)
  call chmalloc('fitcharge.src','DO_ALP','ALPHA2',NTYPES*NTYPES,crl=ALPHA2)
  call chmalloc('fitcharge.src','DO_ALP','beta',NTYPES,crl=beta)
  call chmalloc('fitcharge.src','DO_ALP','beta2',NTYPES,crl=beta2)
  call chmalloc('fitcharge.src','DO_ALP','da',NTYPES,crl=da)
  call chmalloc('fitcharge.src','DO_ALP','EPSILN',NTYPES,crl=EPSILN)
  call chmalloc('fitcharge.src','DO_ALP','DPOT',NTYPES*MXNGRD,crl=DPOT)
  IF(RESTR) THEN
     call chmalloc('fitcharge.src','DO_ALP','TARGET',NATOM,crl=TARGET)
     do i=1,natom
        TARGET(i) = CG(i)
     enddo
  ENDIF

  !
  !     Check if the shape is allowed (for REPD only)
  !
  IF(RESTR.AND.REPD) THEN
     IF(LHYPER) THEN
        DO I=1,NATOM
           IF(ISDRUDE(i).and.CG(i).NE.0.0D0) THEN
              CALL WRNDIE(-4,'FITCHARGE', &
                   'REPD with LHYPER shape requires'// &
                   ' zero reference polarizabilities')
           ENDIF
        ENDDO
     ENDIF
     IF(DFLAT.NE.0.0D0) THEN
        DO I=1,NATOM
           IF(ISDRUDE(i).and.CG(i).NE.0.0D0) THEN
              CALL WRNDIE(-4,'FITCHARGE', &
                   'REPD with nonzero DFLAT requires'// &
                   ' zero reference polarizabilities')
           ENDIF
        ENDDO
     ENDIF
  ENDIF

  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !     Levenberg-Marquardt algorithm
  !     (Nomenclature from Numerical Recipes)
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

  NPOINTS=0
  DO ICONF=1,NCONF
     NPOINTS=NPOINTS+NPERT_(ICONF)*NGRID(ICONF)
  ENDDO

  !     Start (iteration zero)
  lambda = 0.05d0
  iter = 0
  counter2 = 0

  IGEO = 1
  ACCUML = .FALSE.
  DO ICONF=1,NCONF
     DO IGRID=1,NGRID(ICONF)
        GRIDX1(IGRID)=GRIDX(IGRID,ICONF)
        GRIDY1(IGRID)=GRIDY(IGRID,ICONF)
        GRIDZ1(IGRID)=GRIDZ(IGRID,ICONF)
     ENDDO
     !
     ! Contribution of unperturbed potential
     ! vam: This part requires improvement
     IF(NPERT_(ICONF).EQ.1 .OR. TEST) THEN
        IF(FITALP) THEN
           !            discharge CAL atom
           CALCG = CG(NATOM)
           CG(NATOM) = 0.0D0
        ENDIF
        call POTCHI2(NATOM_,ngeo_,ISLCT1,ISLCT2,equiv,ACCUML,CONSTC,FITCL, &
             MULTIP,CONSTR,NTYPES,CTYPES,GRIDX1,GRIDY1,GRIDZ1,NGRID(ICONF),ALLX, &
             ALLY,ALLZ,apot0,mpot0,IGEO,IGEO,CHI2,BETA,ALPHA,RESTR,REPD, &
             RESP,LPARAB,LHYPER,BHYPER,DHYPER,FLAT,DFLAT,WRESTR,TARGET, &
             chi2restr,NFRAGM,AFRAGM,AWEIGHT,AXX,AYY,AZZ,TEST,THOLECT,DRUDEY, &
             KDRUDEFIT,QVTHOLE,MXNGRD,DPOT,MXNGEO,EPSILN)
        ACCUML = .TRUE.
        !          restore charge on CAL atom
        IF(FITALP) CG(NATOM) = CALCG
     ENDIF
     !
     ! Contribution of perturbed potential
     IF(NPERT_(ICONF).GT.1) &
          CALL POTCHI2( NATOM_,ngeo_, &
          ISLCT1,ISLCT2,equiv, ACCUML, &
          CONSTC,FITCL, MULTIP,CONSTR,NTYPES,CTYPES, &
          GRIDX1,GRIDY1,GRIDZ1,NGRID(ICONF), &
          ALLX,ALLY,ALLZ, APOT,MPOT,IGEO,IGEO+NPERT_(ICONF)-1, &
          CHI2,BETA,ALPHA, &
          RESTR,REPD,RESP,LPARAB,LHYPER, &
          BHYPER,DHYPER,FLAT,DFLAT,WRESTR, &
          TARGET,chi2restr,NFRAGM,AFRAGM, &
          AWEIGHT,AXX,AYY,AZZ,TEST,THOLECT,DRUDEY,KDRUDEFIT, &
          QVTHOLE,MXNGRD,DPOT,MXNGEO,EPSILN)
     IGEO = IGEO + NPERT_(ICONF)
     ACCUML = .TRUE.
  ENDDO
  CALL ITEROUT(iter,ISLCT1,NPOINTS,NTYPES,RESTR,CHI2,chi2restr, &
       KDRUDEFIT)
  !
  IF(NTYPES.EQ.0) RETURN
  IF(TEST) GOTO 200

  DO ITER=1,NITER
     if (prnlev >= 2) WRITE(OUTU,"(a,f10.6)") 'lambda =',lambda
#if KEY_PATHSCALE==1
     CALL &  
#endif
     FLUSH (OUTU)

     !                Solution of "ALPHA*da=beta"
     call LEVMARQ(ALPHA,BETA,DA,NTYPES,lambda)
     !                                 Modify charges by "da"
     call MODCG(da,.TRUE.,CTYPES,CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
          NFRAGM,AFRAGM)
     call MODTH(da,.TRUE.,NTYPES,CTYPES,THOLECT,DRUDEY,QVTHOLE)

     !                                 Compute new chi2
     IGEO = 1
     ACCUML = .FALSE.
     DO ICONF=1,NCONF
        DO IGRID=1,NGRID(ICONF)
           GRIDX1(IGRID)=GRIDX(IGRID,ICONF)
           GRIDY1(IGRID)=GRIDY(IGRID,ICONF)
           GRIDZ1(IGRID)=GRIDZ(IGRID,ICONF)
        ENDDO
        !
        ! Contribution of unperturbed potential
        ! vam: This part requires improvement
        IF(NPERT_(ICONF).EQ.1 .OR. TEST) THEN
           IF(FITALP) THEN
              !               discharge CAL atom
              CALCG = CG(NATOM)
              CG(NATOM) = 0.0D0
           ENDIF
           call POTCHI2(NATOM_,ngeo_,ISLCT1,ISLCT2,equiv,ACCUML,CONSTC,FITCL, &
                MULTIP,CONSTR,NTYPES,CTYPES,GRIDX1,GRIDY1,GRIDZ1, &
                NGRID(ICONF),ALLX, &
                ALLY,ALLZ,apot0,mpot0,IGEO,IGEO,CHI22,BETA2,ALPHA2,RESTR,REPD, &
                RESP,LPARAB,LHYPER,BHYPER,DHYPER,FLAT,DFLAT,WRESTR,TARGET, &
                chi2restr2,NFRAGM,AFRAGM,AWEIGHT,AXX,AYY,AZZ, &
                TEST,THOLECT,DRUDEY, &
                KDRUDEFIT,QVTHOLE,MXNGRD,DPOT,MXNGEO,EPSILN)
           ACCUML = .TRUE.
           !             restore charge on CAL atom
           IF(FITALP) CG(NATOM) = CALCG
        ENDIF
        !
        ! Contribution of perturbed potential
        IF(NPERT_(ICONF).GT.1) &
             CALL POTCHI2( NATOM_,ngeo_, &
             ISLCT1,ISLCT2,equiv, ACCUML, &
             CONSTC,FITCL, MULTIP,CONSTR,NTYPES,CTYPES, &
             GRIDX1,GRIDY1,GRIDZ1,NGRID(ICONF), &
             ALLX,ALLY,ALLZ, APOT,MPOT,IGEO,IGEO+NPERT_(ICONF)-1, &
             CHI22,BETA2,ALPHA2, &
             RESTR,REPD,RESP,LPARAB,LHYPER, &
             BHYPER,DHYPER,FLAT,DFLAT,WRESTR, &
             TARGET,chi2restr2,NFRAGM,AFRAGM, &
             AWEIGHT,AXX,AYY,AZZ,TEST,THOLECT,DRUDEY,KDRUDEFIT, &
             QVTHOLE,MXNGRD,DPOT,MXNGEO,EPSILN)
        IGEO=IGEO+NPERT_(ICONF)
        ACCUML = .TRUE.
     ENDDO

     !        Beginning of iteration

     CONV=2.0D0*ABS(CHI22-CHI2)/(CHI22+CHI2)
     if (prnlev >= 2) then
        WRITE(OUTU,"(a,f10.6)") 'chi2   =', chi2
        WRITE(OUTU,"(a,f10.6)") 'chi22  =', chi22
        WRITE(OUTU,"(a,f10.6)") 'conver =', conv
     endif

     IF(CONV.LT.TOLERA) THEN
        counter2 = counter2 + 1
     ELSE
        counter2 = 0
     ENDIF
     IF(chi22.LT.chi2) THEN
        lambda = lambda / 10.0d0
        beta = beta2
        alpha = alpha2
        chi2 = chi22
        chi2restr = chi2restr2
     ELSE
        lambda = lambda * 10.0d0
        !                                 Go back to previous charges
        call MODCG(da,.FALSE.,CTYPES,CONSTC,FITCL,MULTIP,CONSTR,equiv, &
             NFRAGM,AFRAGM)
        call MODTH(da,.FALSE.,NTYPES,CTYPES,THOLECT,DRUDEY,QVTHOLE)
     ENDIF

     CALL ITEROUT(ITER,ISLCT1,NPOINTS,NTYPES,RESTR,CHI2,chi2restr, &
          KDRUDEFIT)
#if KEY_PATHSCALE==1
     CALL &  
#endif
     FLUSH (OUTU)
     !                                 Test convergence of iteration
     IF(counter2.GE.counter) THEN
        if (prnlev >= 2) WRITE(OUTU,'(A)') 'FITALP> Convergent LM algorithm'
        CALL CROUND(ISLCT1,ISLCT2,UPRT,TPRT,EQUIV,ASCALE,RDIPOL, &
             iter, npoints,NTYPES, chi2,chi2restr, lambda, &
             THOLECT, DRUDEY, KDRUDEFIT, QVTHOLE)
        GOTO 200
     ENDIF
  ENDDO
  if (prnlev >= 2) WRITE(OUTU,'(A)') 'FITALP> Non-convergent LM algorithm'
200 CONTINUE
  !
  !     Free memory
  call chmdealloc('fitcharge.src','DO_ALP','ALPHA',NTYPES*NTYPES,crl=ALPHA)
  call chmdealloc('fitcharge.src','DO_ALP','ALPHA2',NTYPES*NTYPES,crl=ALPHA2)
  call chmdealloc('fitcharge.src','DO_ALP','beta',NTYPES,crl=beta)
  call chmdealloc('fitcharge.src','DO_ALP','beta2',NTYPES,crl=beta2)
  call chmdealloc('fitcharge.src','DO_ALP','da',NTYPES,crl=da)
  call chmdealloc('fitcharge.src','DO_ALP','EPSILN',NTYPES,crl=EPSILN)
  call chmdealloc('fitcharge.src','DO_ALP','DPOT',NTYPES*MXNGRD,crl=DPOT)
  IF(RESTR) call chmdealloc('fitcharge.src','DO_ALP','TARGET',NATOM,crl=TARGET)

  RETURN
END SUBROUTINE DO_ALP
!
!
! Examine Equivalences
SUBROUTINE EXEQUI(ISLCT1,EQUIV, &
     NTYPES,FITGRP,NCONST,CONSTC,MULTIP,CATOM,WRESTR)
  use dimens_fcm
  use comand
  use psf
  use stream
  use chutil,only:getres
  !
  ! ISLCT1(*)    Selection of atoms to fit
  ! EQUIV(*)     Equivalence classes of atoms
  ! WRESTR(*)    Weighting constants for individual atoms
  ! NTYPES       Number of non-equivalent types to fit
  ! FITGRP(*)    List of atoms for fitting (size=NATOM)
  ! NCONST       Number of constraints to apply
  ! CONSTC(*)    List of constrained classes (size=NATOM)
  ! MULTIP(*)    Number of atoms equivalent to the constrained atom (size=nconst)
  ! CATOM(*)     Index of the constrained atom, within each residue (size=NATOM)
  !
  !        Input Arguments
  INTEGER ISLCT1(*)
  INTEGER EQUIV(*)
  real(chm_real)  WRESTR(*)
  !        Output Arguments
  INTEGER NTYPES
  INTEGER FITGRP(*)
  INTEGER NCONST
  INTEGER CONSTC(*)
  INTEGER MULTIP(*)
  INTEGER CATOM(*)
  !
  !        Local variables
  INTEGER IA,IR,ITYPE,CIA
  LOGICAL NEWRES, REDUND
  !
  !     Find all the redundancies and constraints in the fit
  IF(PRNLEV.GT.2) WRITE(OUTU,'(A)') &
       '    TYPE  MTYPE    COMMENT     CLASS'
120 FORMAT(3X,A6,I6,5X,A,I6,G15.8)
  NTYPES=0
  NCONST=0
  DO IR=1,NRES
     NEWRES=.TRUE.
     DO IA=IBASE(IR)+1,IBASE(IR+1)
        IF(ISLCT1(IA).EQ.1) THEN
           !
           !              Find new class (group) type
           DO ITYPE=1,NTYPES
              !     Check that the atom group type EQUIV(IA) is already in FITGRP
              IF(SAMECL(EQUIV(IA),FITGRP(ITYPE))) THEN
                 IF(PRNLEV.GT.2) WRITE(OUTU,120) &
                      ATYPE(IA),IAC(IA),'Redundant  ', &
                      FITGRP(ITYPE),WRESTR(IA)
                 !                    Jump to the END of the IA loop
                 GOTO 150
              ENDIF
           ENDDO
           !
           !     This is new atom group type (class), which is not present in FITGRP
           IF(NEWRES .AND. .NOT.ISDRUDE(IA) .AND.  &
                .NOT.ISDRUDE(IA+1)) THEN
              !   place charge constraint on first encountered hydrogen atom of new residue
              NCONST=NCONST+1
              CONSTC(NCONST)=EQUIV(IA)
              MULTIP(NCONST)=1
              IF(PRNLEV.GT.2) WRITE(OUTU,120) &
                   ATYPE(IA),IAC(IA),'Constrained', &
                   CONSTC(NCONST),WRESTR(IA)
              NEWRES=.FALSE.
           ELSE
              REDUND = .FALSE.
              IF (NCONST > 0) THEN
                 REDUND = SAMECL(EQUIV(IA),CONSTC(NCONST))
              ENDIF
              IF (REDUND) THEN
                 MULTIP(NCONST)=MULTIP(NCONST)+1
                 IF(PRNLEV.GT.2) WRITE(OUTU,120) &
                      ATYPE(IA),IAC(IA),'Redundant c', &
                      CONSTC(NCONST),WRESTR(IA)
              ELSE
                 NTYPES=NTYPES+1
                 FITGRP(NTYPES)=EQUIV(IA)
                 IF(PRNLEV.GT.2) WRITE(OUTU,120) &
                      ATYPE(IA),IAC(IA),'Free       ', &
                      FITGRP(NTYPES),WRESTR(IA)
              ENDIF
           ENDIF
        ENDIF
150     CONTINUE
     ENDDO
  ENDDO

  !     For each atom, find the index of the constrained atom
  !       within the same residue
  !     Only one constrained atom per residue is allowed
  DO IA=1,NATOM
     CATOM(IA)=-1
  ENDDO
  IF(NCONST.GT.0) THEN
     DO IA=1,NATOM
        DO CIA=IBASE(GETRES(IA,IBASE,NRES))+1, &
             IBASE(GETRES(IA,IBASE,NRES)+1)
           IF(ISLCT1(CIA).EQ.1) THEN
              DO ITYPE=1,NCONST
                 IF(SAMECL(EQUIV(CIA),CONSTC(ITYPE))) THEN
                    CATOM(IA)=CIA
                    GOTO 111
                 ENDIF
              ENDDO
           ENDIF
        ENDDO
111     CONTINUE
     ENDDO
     DO IA=1,NATOM
        IF(CATOM(IA).LE.0 .OR. CATOM(IA).GT.NATOM) CATOM(IA)=IA
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE EXEQUI

SUBROUTINE COUNTTH(TTYPES,DRUDEY)
  use dimens_fcm
  use psf
  use param

  !     Arguments
  !  TTYPEs       Number of thole parameter types
  !
  LOGICAL  DRUDEY(*)
  INTEGER TTYPES

  !     Local variables
  INTEGER I,J

  TTYPES = 0

  !     Figure out atom types with drudes
  DO I=1,NATC
     DRUDEY(I) = .FALSE.
  ENDDO

  DO I=1,NATOM
     IF(ISDRUDE(I+1)) THEN
        DRUDEY(IAC(I)) = .TRUE.
     ENDIF
  ENDDO

  DO I=1,NATC
     IF (DRUDEY(I)) THEN
        TTYPES = TTYPES + 1
     ENDIF
  ENDDO

  RETURN

END SUBROUTINE COUNTTH

SUBROUTINE MODTH( DA, ADD, NTYPES, CTYPES, THOLECT,  &
     DRUDEY, QVTHOLE)
  use dimens_fcm
  use comand
  use psf
  use stream
  use param

  !     Arguments
  !  DA(*)        INPUT   Charge/Thole increments
  !  ADD          INPUT  .TRUE. to add increments
  !                      .FALSE. to substract increments
  !  NTYPES       INPUT  Number of charge/thole mini variables
  !  CTYPES       INPUT  Number of charge mini variables

  real(chm_real)  DA(*)
  real(chm_real)  THOLECT(*)
  LOGICAL  DRUDEY(*)
  LOGICAL ADD, QVTHOLE
  INTEGER NTYPES
  INTEGER CTYPES

  !     Local variables
  INTEGER I,J,K
  INTEGER IX,NXI,NXIMAX,COUNTER

  IF (.NOT.QVTHOLE) RETURN

  !     Initialize
  K = CTYPES

  !     Revise the thole parameters

  DO I = 1,NATOM
     THOLECT(IAC(I)) = THOLEI(I)
  ENDDO

  DO I = 1,NATC
     IF (DRUDEY(I)) THEN
        K = K + 1
        IF (ADD) THEN
           THOLECT(I) = THOLECT(I) + DA(K)
           IF(THOLECT(I).LT.0.1) THEN
              THOLECT(I)=THOLECT(I) - DA(K)
           ENDIF
        ELSE
           THOLECT(I) = THOLECT(I) - DA(K)
           IF(THOLECT(I).LT.0.1) THEN
              THOLECT(I)=THOLECT(I) + DA(K)
           ENDIF
        ENDIF
     ENDIF
  ENDDO

  DO I = 1,NATOM
     THOLEI(I) = THOLECT(IAC(I))
  ENDDO

  RETURN

END SUBROUTINE MODTH

SUBROUTINE MODCG( da, add, &
     NTYPES, CONSTC, FITCL, &
     MULTIP, CONSTR, EQUIV, NFRAGM, AFRAGM)
  use dimens_fcm
  use comand
  use psf
  use stream

  !     Arguments
  !  da(*)        INPUT  Charge increments (size=NTYPES)
  !  add          INPUT  .TRUE. to add increments
  !                      .FALSE. to substract increments
  !  NTYPES       INPUT  Number of non-equivalent types to fit
  !  CONSTC(*)    INPUT
  !  FITCL(*)     INPUT
  !  MULTIP(*)    INPUT  Number of atoms equivalent to the constrained atom
  !                      (size=NCONSTR)
  !  CONSTR(*)    INPUT  Index of the constrained atom, within each residue
  !                      (size=NATOM)
  !  EQUIV(NATOM) INPUT  Equivalence classes of atoms
  !
  real(chm_real)  da(*)
  LOGICAL add
  INTEGER NTYPES
  INTEGER CONSTC(*)
  INTEGER FITCL(*)
  INTEGER MULTIP(*)
  INTEGER CONSTR(*)
  INTEGER EQUIV(NATOM)
  INTEGER NFRAGM,AFRAGM(NATOM,NATOM)

  !     Local variables
  INTEGER IA,ITYPE,CIA,ICONST,NCONST
  INTEGER IFRAGM,IFATOM,IHEAVY,NHATOM
  real(chm_real)  TOTCH,DIFFCH
  !
  !     Temporary use opposite increments
  IF(.NOT.ADD) THEN
     DO ITYPE=1,NTYPES
        DA(ITYPE)=-DA(ITYPE)
     ENDDO
  ENDIF
  !
  ! Calculate total charge of molecule
  TOTCH=0.0D0
  DO IA=1,NATOM
     TOTCH=TOTCH+CG(IA)
  ENDDO
  !
  DO IA=1,NATOM
     !        Skip constrained and equivalent atoms
     IF(IA.EQ.CONSTR(IA)) GOTO 200
     IF(SAMECL(EQUIV(IA),EQUIV(CONSTR(IA)))) GOTO 200
     !
     !        Skip dependent (Hydrogen) atoms of constant-charge fragments;
     !          their charge will not be fitted
     DO IFRAGM=1,NFRAGM
        NHATOM=AFRAGM(1,IFRAGM)
        DO IFATOM=3,NHATOM+2
           IF(AFRAGM(IFATOM,IFRAGM).EQ.IA) GOTO 200
        ENDDO
     ENDDO
     !
     DO ITYPE=1,NTYPES
        IF(SAMECL(EQUIV(IA),FITCL(ITYPE))) THEN
           IF(ISDRUDE(IA)) THEN
              !                    Drude particle:
              CG(IA)=CG(IA)+DA(ITYPE)
              !                    Real corresponding atom:
              CG(IA-1)=CG(IA-1)-DA(ITYPE)
           ELSE
              CG(IA)=CG(IA)+DA(ITYPE)
              !                 Modify constrained charges to keep constant
              !                   total charge of each residue.
              IF(CONSTR(IA).NE.-1) THEN
                 CIA=CONSTR(IA)
                 DO ICONST=1,NATOM
                    IF(CONSTC(ICONST).EQ.EQUIV(CIA)) THEN
                       CG(CIA)=CG(CIA)-DA(ITYPE)/MULTIP(ICONST)
                       GOTO 100
                    ENDIF
                 ENDDO
              ENDIF
           ENDIF
           GOTO 100
        ENDIF
     ENDDO
100  CONTINUE
     !
     !        Perform fragment charge neutralization
     DO IFRAGM=1,NFRAGM
        IF(IA.EQ.AFRAGM(2,IFRAGM)) THEN
           !              This is Carbon atom from the fixed-charge fragment
           !              Calculate H-atom charge to neutralize the heavy atom charge
           IHEAVY=AFRAGM(2,IFRAGM)
           NHATOM=AFRAGM(1,IFRAGM)
           !              Set H-atom charge
           CG(AFRAGM(3,IFRAGM))=-(CG(IHEAVY)+CG(IHEAVY+1)) &
                / FLOAT(NHATOM)
           !              Apply this charge to other H-atoms of the fragment
           DO IFATOM=4,NHATOM+2
              CG(AFRAGM(IFATOM,IFRAGM))=CG(AFRAGM(3,IFRAGM))
           ENDDO
           GOTO 200
        ENDIF
     ENDDO
     !
200  CONTINUE
  ENDDO
  !
  !     Set charges on atoms equivalent to constrained atoms
  DO IA=1,NATOM
     IF(CONSTR(IA).NE.-1 .AND. IA.NE.CONSTR(IA) .AND. &
          SAMECL(EQUIV(IA),EQUIV(CONSTR(IA)))) THEN
        CG(IA)=CG(CONSTR(IA))
     ENDIF
  ENDDO
  !
  ! Apply total charge preservation
  IF(NFRAGM.GT.0) THEN
     !        New scheme with fragment charge preservation
     !        Calculate number of constraints
     NCONST=1
     DO IA=1,NATOM
        IF(CONSTR(IA).NE.-1 .AND. IA.NE.CONSTR(IA) .AND. &
             SAMECL(EQUIV(IA),EQUIV(CONSTR(IA)))) THEN
           NCONST=NCONST+1
        ENDIF
     ENDDO
     DO IA=1,NATOM
        TOTCH=TOTCH-CG(IA)
     ENDDO
     DIFFCH=TOTCH/FLOAT(NCONST)
     DO IA=1,NATOM
        IF(CONSTR(IA).NE.-1 .AND. IA.NE.CONSTR(IA) .AND. &
             SAMECL(EQUIV(IA),EQUIV(CONSTR(IA)))) THEN
           CG(IA)=CG(IA)+DIFFCH
        ENDIF
     ENDDO
     CG(CONSTR(1))=CG(CONSTR(1))+DIFFCH
  ELSE
     !        Old scheme with residue charge preservation
     DO IA=1,NATOM
        IF(CONSTR(IA).NE.-1 .AND. IA.NE.CONSTR(IA) .AND. &
             SAMECL(EQUIV(IA),EQUIV(CONSTR(Ia)))) THEN
           CG(ia)=CG(CONSTR(IA))
        ENDIF
     ENDDO
  ENDIF
  !
  !     Go back to original increments
  IF(.NOT.ADD) THEN
     DO ITYPE=1,NTYPES
        DA(ITYPE)=-DA(ITYPE)
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE MODCG

SUBROUTINE ITEROUT(ITER,ISLCT1,N,NTYPES,RESTR,CHI2,chi2restr, &
     KDRUDEFIT)
  use dimens_fcm
  use comand
  use psf
  use consta
  use stream
  !     ...Arguments
  !    ITER             INPUT  Iteration number
  !    ISLCT1(NATOM)    INPUT  Selection of atoms to fit
  !    n                INPUT  Total number of gridpoints
  !    NTYPES           INPUT  Number of parameters to fit
  !    RESTR        INPUT  .TRUE. if chi2restr.NE.0
  !    chi2             INPUT  Total chi2
  !    chi2restr        INPUT  chi2 associated to restraints
  !
  INTEGER ITER
  INTEGER ISLCT1(NATOM)
  INTEGER N
  INTEGER NTYPES
  LOGICAL RESTR,DRUDESLCT
  real(chm_real)  CHI2
  real(chm_real)  chi2restr
  real(chm_real)  KDRUDEFIT(*)
  !     ...Local variables
  INTEGER ia

  if (prnlev >= 2) then
     WRITE(OUTU,'(A)') ' '
     WRITE(OUTU,'(A,I4,A)') '**** LM Iteration',iter,' ****'
  endif

2000 FORMAT('Type  Charge    Alpha (kdrude=',f6.1,')')
2001 FORMAT(a4,f10.6)
2002 FORMAT(a4,f10.6,f10.6)
2003 FORMAT(a4,'    -     ',f10.6)

  DO IA=1,NATOM
     IF(ISLCT1(IA)==1) THEN
        DRUDESLCT=ISDRUDE(IA)
        IF(DRUDESLCT) THEN
           DRUDESLCT=ISLCT1(IA-1)==1
        ENDIF
        if( .NOT.DRUDESLCT ) then
           IF(IA.LT.NATOM .AND. ISDRUDE(IA+1)) THEN
              !                                 ... is polarizable atom
              if (prnlev >= 2) then
                 WRITE(OUTU,2002) ATYPE(IA),CG(IA+1)+CG(IA), &
                      CCELEC * CG(IA+1)**2 / (2.0*KDRUDEFIT(IA))
              endif
           ELSEIF(ISDRUDE(IA)) THEN
              !                                 ... is Drude particle
              if (prnlev >= 2) then
                 WRITE(OUTU,2003) ATYPE(IA), &
                      CCELEC * CG(IA)**2 / (2.0*KDRUDEFIT(IA))
              endif
           ELSE
               !                                 ... is non-polarizable atom
              if (prnlev >= 2) then
                 WRITE(OUTU,2001) ATYPE(IA),CG(IA)
              endif
           ENDIF
        endif
     ENDIF
  ENDDO
  IF(RESTR) THEN
     if (prnlev >= 2) then
        WRITE(OUTU,'(a,F12.8)') 'chi2 (potential) =',chi2 - chi2restr
        WRITE(OUTU,'(a,F12.8)') 'chi2 (restraint) =',chi2restr
     endif
  ENDIF
  if (prnlev >= 2) then
     WRITE(OUTU,"(A,F10.6)") &
          'RMSE   =',SQRT((chi2-chi2restr)/(n-NTYPES))
  endif
  !
  RETURN
END SUBROUTINE ITEROUT

LOGICAL FUNCTION SAMECL(CLASS1,CLASS2)
  ! Equivalence classes
  INTEGER CLASS1, CLASS2
  IF(CLASS1.EQ.CLASS2 .AND. CLASS1.GE.0) THEN
     SAMECL=.TRUE.
  ELSE
     SAMECL=.FALSE.
  ENDIF
  RETURN
END FUNCTION SAMECL

SUBROUTINE POTCHI2( NATOM_,ngeo_, &
     ISLCT1,ISLCT2,equiv, ACCUML, &
     CONSTC,FITCL,MULTIP,CONSTR,NTYPES,CTYPES, &
     GRIDX,GRIDY,GRIDZ,NGRID, &
     ALLX,ALLY,ALLZ, APOT,MPOT,IGEObegin,IGEOEND, &
     CHI2,BETA,ALPHA,RESTR,REPD,RESP,LPARAB,LHYPER, &
     BHYPER,DHYPER,FLAT,DFLAT,WRESTR, &
     TARGET,chi2restr,NFRAGM,AFRAGM, &
     AWEIGHT,AXX,AYY,AZZ,TEST,THOLECT,DRUDEY,KDRUDEFIT, &
     QVTHOLE,MXNGRD,DPOT,MXNGEO,EPSILN)
  !
  !  Purpose:
  !     Compute chi2 (and derivatives) between ab initio potential
  !     (pot) and model potential (MPOT).
  !     The self-consistent dipoles are computed.
  !
  !  Arguments:
  !
  !     ISLCT1    (input)  INTEGER array, dimension (NATOM)
  !                        Drude particles to fit
  !
  !     ISLCT2    (input)  INTEGER array, dimension (NATOM)
  !                        Atom contributing to the perturbed ab initio
  !                        potential
  !
  !     equiv     (input)  INTEGER array, dimension (NATOM)
  !                        Equivalence classes of the atoms
  !
  !     ACCUML    (input)  LOGICAL
  !                        Flag to ACCUML chi2 (and derivates)
  !                        or to reset (if .FALSE.)
  !
  !     CONSTC    (input)  INTEGER array, dimension (NATOM)
  !     FITCL     (input)  INTEGER array, dimension (NTYPES)
  !     MULTIP    (input)  INTEGER array, dimension (NTYPES)
  !     CONSTR    (input)  INTEGER array, dimension (NTYPES)
  !     NTYPES    (input)  INTEGER
  !
  !     GRIDX, GRIDY, GRIDZ
  !               (input)  REAL array, dimension (NGRID)
  !                        Coordinates of gridpoints
  !     NGRID     (input)  INTEGER
  !                            Number of gridpoints
  !     ALLX, ALLY, ALLZ
  !               (input)  REAL array, dimension (NATOM,ngeo)
  !
  !     pot       (input)  REAL array, dimension (NGRID,ngeo)
  !                        Ab initio perturbed potential
  !     MPOT      (output) REAL array, dimension (NGRID,ngeo)
  !                        Model perturbed potential
  !     IGEObegin (input)  INTEGER
  !                        Perturbed geometry to start with
  !     IGEOEND   (input)  INTEGER
  !                        Perturbed geometry to END with
  !
  !     chi2      (output) REAL
  !     beta      (output) REAL array, dimension (NTYPES)
  !     ALPHA     (output) REAL array, dimension (NTYPES,NTYPES)
  !
  !     RESTR     (input)  LOGICAL
  !                        .TRUE. if some restraint terms are to be added
  !     REPD      (input)  LOGICAL
  !     resp      (input)  LOGICAL
  !     LPARAB    (input)  LOGICAL
  !     LHYPER    (input)  LOGICAL
  !     BHYPER    (input)  REAL
  !     DHYPER    (input)  REAL
  !     FLAT      (input)  REAL
  !     DFLAT     (input)  REAL
  !     WRESTR    (input)  REAL array, dimension (NATOM)
  !                        Restraint forces (see code)
  !     TARGET    (input)  REAL array, dimension (NATOM)
  !                        Target charges
  !     chi2restr (output) REAL
  !     TEST      (input)  LOGICAL
  !                        Switch ON or OFF test mode
  !
  use dimens_fcm
  use psf
  use coord
  use memory

  !     ...Local parameters
  INTEGER MXNGEO, MXNGRD
  !     ...Arguments
  INTEGER NATOM_, ngeo_
  INTEGER ISLCT1(NATOM_), ISLCT2(NATOM_)
  INTEGER EQUIV(NATOM_)
  LOGICAL ACCUML
  INTEGER CONSTC(*)
  INTEGER FITCL(*)
  INTEGER MULTIP(*)
  INTEGER CONSTR(*)
  INTEGER NTYPES
  INTEGER CTYPES
  real(chm_real)  GRIDX(*), GRIDY(*), GRIDZ(*)
  INTEGER NGRID
  real(chm_real)  ALLX(NATOM_,NGEO_)
  real(chm_real)  ALLY(NATOM_,NGEO_)
  real(chm_real)  ALLZ(NATOM_,NGEO_)
  real(chm_real)  APOT(MXNGRD,MXNGEO)
  real(chm_real)  MPOT(MXNGRD,MXNGEO)
  INTEGER IGEObegin, IGEOEND
  real(chm_real)  CHI2,BETA(NTYPES),ALPHA(NTYPES,NTYPES),  &
       EPSILN(NTYPES)
  LOGICAL RESTR,REPD,RESP,LPARAB, LHYPER, TEST
  real(chm_real)  BHYPER,DHYPER, FLAT,DFLAT
  real(chm_real)  WRESTR(NATOM_)
  real(chm_real)  TARGET(*)
  real(chm_real)  chi2restr
  INTEGER NFRAGM,AFRAGM(NATOM_,NATOM_)
  real(chm_real)  AWEIGHT,AXX,AYY,AZZ
  real(chm_real)  THOLECT(*)
  LOGICAL  DRUDEY(*)
  real(chm_real)  KDRUDEFIT(*)
  LOGICAL QVTHOLE

  !     Local variables
  real(chm_real)  eps,eps2
  real(chm_real)  delta
  real(chm_real)  DPOT(NTYPES,MXNGRD)
  !                                 Potential derivatives
  INTEGER ngeo
  real(chm_real)  chi2r,dchi2r,ddchi2r
  INTEGER ia, ITYPE, ja, JTYPE, IGEO, igrid
  real(chm_real)  dist2
  INTEGER ir, cia
  !            pointed to copy charges
  real(chm_real),allocatable,dimension(:) :: COPYCG

  NFRAGM=0

  IF(.NOT.ACCUML) THEN
     CHI2=0.0D0
     DO ITYPE=1,NTYPES
        BETA(ITYPE)=0.0D0
        DO JTYPE=1,NTYPES
           ALPHA(ITYPE,JTYPE)=0.0D0
        ENDDO
     ENDDO
     chi2restr=0.0D0
  ENDIF
  DO IGEO=IGEObegin,IGEOEND
     !                                Set positions of this geometry
     DO IA=1,NATOM
        X(IA)=ALLX(IA,IGEO)
        Y(IA)=ALLY(IA,IGEO)
        Z(IA)=ALLZ(IA,IGEO)
     ENDDO

     !        Skip numerical differentiation in the TEST mode
     IF(TEST) GOTO 150

     DO ITYPE=1,NTYPES
        EPSILN(ITYPE)=0.0D0
     ENDDO
     DO ITYPE=1,NTYPES
        !                                Choose charge variation
        EPS=0.0D0
        DO IA=1,NATOM
           IF(ISLCT1(IA).EQ.1 .AND. IA.NE.CONSTR(IA) .AND. &
                .NOT.SAMECL(EQUIV(IA),EQUIV(CONSTR(IA))) .AND. &
                SAMECL(EQUIV(IA),FITCL(ITYPE))) THEN
              EPS=1.0D-4
              GOTO 100
           ENDIF
        ENDDO
100     CONTINUE
        !                                Choose the thole variation
        IF (ITYPE.GT.CTYPES) EPS=1.0D-5
        !                                "+" increment
        EPSILN(ITYPE)=EPS
        CALL MODCG( EPSILN,.TRUE.,CTYPES, &
             CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
             NFRAGM,AFRAGM)
        CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
             DRUDEY, QVTHOLE)
        CALL SCFDIP(KDRUDEFIT)
        CALL POTGRID( GRIDX,GRIDY,GRIDZ,NGRID, &
             ISLCT2,MPOT,IGEO,MXNGRD,MXNGEO)
        DO IGRID=1,NGRID
           DPOT(ITYPE,igrid) = MPOT(igrid,IGEO) / (2.0*eps)
        ENDDO
        !                                "-" increment
        EPSILN(ITYPE) = -2.0*eps
        CALL MODCG( EPSILN,.TRUE.,CTYPES, &
             CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
             NFRAGM,AFRAGM)
        CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
             DRUDEY, QVTHOLE)
        CALL SCFDIP(KDRUDEFIT)
        CALL POTGRID( GRIDX,GRIDY,GRIDZ,NGRID, &
             ISLCT2,MPOT,IGEO,MXNGRD,MXNGEO)
        DO IGRID=1,NGRID
           DPOT(ITYPE,igrid) = DPOT(ITYPE,igrid) - &
                MPOT(igrid,IGEO) / (2.0*eps)
        ENDDO

        EPSILN(ITYPE) = eps
        CALL MODCG( EPSILN,.TRUE.,CTYPES, &
             CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
             NFRAGM,AFRAGM)
        CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
             DRUDEY, QVTHOLE)
        EPSILN(ITYPE)=0.0D0
     ENDDO
     !                                "0" increment (original charges)
150  CONTINUE
     CALL SCFDIP(KDRUDEFIT)
     CALL POTGRID( GRIDX,GRIDY,GRIDZ,NGRID, &
          ISLCT2,MPOT,IGEO,MXNGRD,MXNGEO)
     DO IGRID=1,NGRID
        delta = APOT(igrid,IGEO) - MPOT(igrid,IGEO)
        chi2 = chi2 + delta**2
        DO ITYPE=1,NTYPES
           beta(ITYPE) = beta(ITYPE) + delta * DPOT(ITYPE,igrid)
           DO JTYPE=1,NTYPES
              ALPHA(ITYPE,JTYPE) = ALPHA(ITYPE,JTYPE) + &
                   DPOT(ITYPE,igrid) * DPOT(JTYPE,igrid)
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  !
  IF(TEST) RETURN
  !
  !     Addition of the restraints terms
  !
  IF(RESTR) THEN
     call chmalloc('fitcharge.src','POTCHI2','COPYCG',NATOM,crl=COPYCG)
     do ia=1,natom
        copyCG(ia) = CG(ia)
     enddo
  ENDIF
  IF(RESTR.AND.REPD) THEN
     !           See the REPD paper [Henchman99]
     DO IGEO=IGEObegin,IGEOEND
        DO IA=1,NATOM
           X(IA)=ALLX(ia,IGEO)
           Y(IA)=ALLY(ia,IGEO)
           Z(IA)=ALLZ(ia,IGEO)
        ENDDO
        !           Compute chi2restr
        chi2r = Chi2REPD( NATOM,ISLCT1,CG,TARGET,ISDRUDE, REPD &
             ,LPARAB,LHYPER, BHYPER,DHYPER,FLAT,DFLAT, &
             WRESTR, GRIDX,GRIDY,GRIDZ,NGRID,X,Y,Z, &
             AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT )
        chi2restr = chi2restr + chi2r
        !           Compute beta
        DO ITYPE=1,NTYPES
           EPSILN(ITYPE)=0.0D0
        ENDDO
        DO ITYPE=1,NTYPES
           !              Choose charge variation
           EPS=0.D0
           DO IA=1,NATOM
              IF(ISLCT1(IA).EQ.1 .AND. IA.NE.CONSTR(IA) .AND. &
                   .NOT.SAMECL(EQUIV(IA),EQUIV(CONSTR(IA))) .AND. &
                   SAMECL(EQUIV(IA),FITCL(ITYPE))) THEN
                 EPS=1.0D-4
                 GOTO 200
              ENDIF
           ENDDO
200        CONTINUE
           !                                Choose the thole variation
           IF (ITYPE.GT.CTYPES) EPS=1.0D-5
           !              +
           EPSILN(ITYPE) = eps
           CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           chi2r = Chi2REPD( NATOM,ISLCT1,CG,TARGET,ISDRUDE, REPD &
                ,LPARAB,LHYPER, BHYPER,DHYPER,FLAT,DFLAT, &
                WRESTR, GRIDX,GRIDY,GRIDZ,NGRID,X,Y,Z, &
                AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT )
           dchi2r = chi2r / (2.0*eps)
           CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           EPSILN(ITYPE)=0.0D0
           !              -
           EPSILN(ITYPE) = -eps
           CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           chi2r = Chi2REPD( NATOM,ISLCT1,CG,TARGET,ISDRUDE, REPD &
                ,LPARAB,LHYPER, BHYPER,DHYPER,FLAT,DFLAT, &
                WRESTR, GRIDX,GRIDY,GRIDZ,NGRID,X,Y,Z, &
                AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT )
           dchi2r = dchi2r - chi2r / (2.0*eps)
           CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           EPSILN(ITYPE)=0.0D0
           !              beta
           beta(ITYPE) = beta(ITYPE) - 0.5 * dchi2r
        ENDDO
        !           Compute ALPHA
        DO ITYPE=1,NTYPES
           EPSILN(ITYPE)=0.0D0
        ENDDO
        DO ITYPE=1,NTYPES
           !              Choose charge variation
           EPS=0.0D0
           DO IA=1,NATOM
              IF(ISLCT1(IA).EQ.1 .AND. IA.NE.CONSTR(IA) .AND. &
                   .NOT.SAMECL(EQUIV(IA),EQUIV(CONSTR(IA))) .AND. &
                   SAMECL(EQUIV(ia),FITCL(ITYPE))) THEN
                 EPS=1.0D-4
                 GOTO 300
              ENDIF
           ENDDO
300        CONTINUE
           !                                Choose the thole variation
           IF (ITYPE.GT.CTYPES) EPS=1.0D-5
           DO JTYPE=1,ITYPE
              !                 compute only the lower triangle
              !                 Choose charge variation
              EPS2=0.0D0
              DO IA=1,NATOM
                 IF(ISLCT1(IA).EQ.1 .AND. IA.NE.CONSTR(IA) .AND. &
                      .NOT.SAMECL(EQUIV(IA),EQUIV(CONSTR(IA))).AND. &
                      SAMECL(EQUIV(ia),FITCL(JTYPE))) THEN
                    EPS2=1.0D-5
                    GOTO 400
                 ENDIF
              ENDDO
400           CONTINUE
              !                                Choose the thole variation
              IF (JTYPE.GT.CTYPES) EPS2=1.0D-6
              !                 ++
              EPSILN(ITYPE)=0.0D0
              EPSILN(JTYPE)=0.0D0
              EPSILN(ITYPE) = EPSILN(ITYPE) + eps
              EPSILN(JTYPE) = EPSILN(JTYPE) + eps2
              CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                   CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                   NFRAGM,AFRAGM)
              CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                   DRUDEY, QVTHOLE)
              chi2r = Chi2REPD( NATOM,ISLCT1,CG,TARGET,ISDRUDE, &
                   REPD,LPARAB,LHYPER, BHYPER,DHYPER,FLAT &
                   ,DFLAT, WRESTR, GRIDX,GRIDY,GRIDZ,NGRID,X,Y,Z &
                   ,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
              ddchi2r = chi2r / (4.0*eps*eps2)
              CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                   CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                   NFRAGM,AFRAGM)
              CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
                   DRUDEY, QVTHOLE)
              !                 --
              EPSILN(ITYPE)=0.0D0
              EPSILN(JTYPE)=0.0D0
              EPSILN(ITYPE) = EPSILN(ITYPE) - eps
              EPSILN(JTYPE) = EPSILN(JTYPE) - eps2
              CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                   CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                   NFRAGM,AFRAGM)
              CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                   DRUDEY, QVTHOLE)
              chi2r = Chi2REPD( NATOM,ISLCT1,CG,TARGET,ISDRUDE, &
                   REPD,LPARAB,LHYPER, BHYPER,DHYPER,FLAT &
                   ,DFLAT, WRESTR, GRIDX,GRIDY,GRIDZ,NGRID,X,Y,Z &
                   ,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
              ddchi2r = ddchi2r + chi2r / (4.0*eps*eps2)
              CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                   CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                   NFRAGM,AFRAGM)
              CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
                   DRUDEY, QVTHOLE)
              !                 +-
              EPSILN(ITYPE)=0.0D0
              EPSILN(JTYPE)=0.0D0
              EPSILN(ITYPE) = EPSILN(ITYPE) + eps
              EPSILN(JTYPE) = EPSILN(JTYPE) - eps2
              CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                   CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                   NFRAGM,AFRAGM)
              CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                   DRUDEY, QVTHOLE)
              chi2r = Chi2REPD( NATOM,ISLCT1,CG,TARGET,ISDRUDE, &
                   REPD,LPARAB,LHYPER, BHYPER,DHYPER,FLAT &
                   ,DFLAT, WRESTR, GRIDX,GRIDY,GRIDZ,NGRID,X,Y,Z &
                   ,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
              ddchi2r = ddchi2r - chi2r / (4.0*eps*eps2)
              CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                   CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                   NFRAGM,AFRAGM)
              CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
                   DRUDEY, QVTHOLE)
              !                 -+
              EPSILN(ITYPE)=0.0D0
              EPSILN(JTYPE)=0.0D0
              EPSILN(ITYPE) = EPSILN(ITYPE) - eps
              EPSILN(JTYPE) = EPSILN(JTYPE) + eps2
              CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                   CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                   NFRAGM,AFRAGM)
              CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                   DRUDEY, QVTHOLE)
              chi2r = Chi2REPD( NATOM,ISLCT1,CG,TARGET,ISDRUDE, &
                   REPD,LPARAB,LHYPER, BHYPER,DHYPER,FLAT &
                   ,DFLAT, WRESTR, GRIDX,GRIDY,GRIDZ,NGRID,X,Y,Z &
                   ,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
              ddchi2r = ddchi2r - chi2r / (4.0*eps*eps2)
              CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                   CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                   NFRAGM,AFRAGM)
              CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
                   DRUDEY, QVTHOLE)
              !                 ALPHA
              ALPHA(ITYPE,JTYPE) = ALPHA(ITYPE,JTYPE) + 0.5*ddchi2r
              IF(JTYPE.NE.ITYPE) THEN
                 !                       upper triangle
                 ALPHA(JTYPE,ITYPE) = ALPHA(JTYPE,ITYPE) &
                      + 0.5*ddchi2r
              ENDIF
           ENDDO
        ENDDO
     ENDDO
  ELSEIF(RESTR.AND.RESP) THEN
     !           See the RESP paper [Bayly93]
     ngeo = 1 + IGEOEND - IGEObegin
     !        Compute chi2restr
     chi2r = ngeo*NGRID * Chi2RESP( NATOM,ISLCT1,CG ,TARGET &
          ,ISDRUDE, REPD,LPARAB,LHYPER, BHYPER ,DHYPER ,FLAT &
          ,DFLAT, WRESTR,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT )
     chi2restr = chi2restr + chi2r
     !        Compute beta
     DO ITYPE=1,NTYPES
        EPSILN(ITYPE)=0.0D0
     ENDDO
     DO ITYPE=1,NTYPES
        !           Choose charge variation
        EPS=0.0D0
        DO IA=1,NATOM
           IF(ISLCT1(IA).EQ.1 .AND. IA.NE.CONSTR(IA) .AND. &
                .NOT.SAMECL(EQUIV(IA),EQUIV(CONSTR(IA))) .AND. &
                SAMECL(EQUIV(ia),FITCL(ITYPE))) THEN
              EPS=1.0D-4
              GOTO 500
           ENDIF
        ENDDO
500     CONTINUE
        !                                Choose the thole variation
        IF (ITYPE.GT.CTYPES) EPS=1.0D-5
        !           +
        EPSILN(ITYPE) = eps
        CALL MODCG( EPSILN,.TRUE.,CTYPES, &
             CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
             NFRAGM,AFRAGM)
        CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
             DRUDEY, QVTHOLE)
        chi2r = ngeo*NGRID * Chi2RESP( NATOM,ISLCT1,CG ,TARGET &
             ,ISDRUDE, REPD,LPARAB,LHYPER, BHYPER ,DHYPER &
             ,FLAT,DFLAT, WRESTR,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
        dchi2r = chi2r / (2.0*eps)
        CALL MODCG( EPSILN,.FALSE.,CTYPES, &
             CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
             NFRAGM,AFRAGM)
        CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
             DRUDEY, QVTHOLE)
        EPSILN(ITYPE)=0.0D0
        !           -
        EPSILN(ITYPE) = -eps
        CALL MODCG( EPSILN,.TRUE.,CTYPES, &
             CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
             NFRAGM,AFRAGM)
        CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
             DRUDEY, QVTHOLE)
        chi2r = ngeo*NGRID * Chi2RESP( NATOM,ISLCT1,CG ,TARGET &
             ,ISDRUDE, REPD,LPARAB,LHYPER, BHYPER ,DHYPER &
             ,FLAT,DFLAT, WRESTR,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
        dchi2r = dchi2r - chi2r / (2.0*eps)
        CALL MODCG( EPSILN,.FALSE.,CTYPES, &
             CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
             NFRAGM,AFRAGM)
        CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
             DRUDEY, QVTHOLE)
        EPSILN(ITYPE)=0.0D0
        !           beta
        beta(ITYPE) = beta(ITYPE) - 0.5*dchi2r
     ENDDO
     !        Compute ALPHA
     DO ITYPE=1,NTYPES
        EPSILN(ITYPE)=0.0D0
     ENDDO
     DO ITYPE=1,NTYPES
        !           Choose charge variation
        EPS=0.0D0
        DO IA=1,NATOM
           IF(ISLCT1(IA).EQ.1 .AND. IA.NE.CONSTR(IA) .AND. &
                .NOT.SAMECL(EQUIV(IA),EQUIV(CONSTR(IA))) .AND. &
                SAMECL(EQUIV(ia),FITCL(ITYPE))) THEN
              EPS=1.0D-4
              GOTO 600
           ENDIF
        ENDDO
600     CONTINUE
        !                                Choose the thole variation
        IF (ITYPE.GT.CTYPES) EPS=1.0D-5
        DO JTYPE=1,ITYPE
           !              compute only the lower triangle
           !              Choose charge variation
           EPS2=0.0D0
           DO IA=1,NATOM
              IF(ISLCT1(IA).NE.1) GOTO 650
              IF(IA.EQ.CONSTR(IA)) GOTO 650
              IF(SAMECL(EQUIV(IA),EQUIV(CONSTR(IA)))) GOTO 650
              IF(.NOT.SAMECL(EQUIV(ia),FITCL(JTYPE))) GOTO 650
              EPS2=1.0D-5
              GOTO 750
650           CONTINUE
           ENDDO
750        CONTINUE
           !                                Choose the thole variation
           IF (JTYPE.GT.CTYPES) EPS2=1.0D-6
           !              ++
           EPSILN(ITYPE)=0.0D0
           EPSILN(JTYPE)=0.0D0
           EPSILN(ITYPE) = EPSILN(ITYPE) + eps
           EPSILN(JTYPE) = EPSILN(JTYPE) + eps2
           CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           chi2r = ngeo*NGRID * Chi2RESP( NATOM,ISLCT1,CG ,TARGET &
                ,ISDRUDE, REPD,LPARAB,LHYPER, BHYPER ,DHYPER &
                ,FLAT,DFLAT, WRESTR,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
           ddchi2r = chi2r / (4.0*eps*eps2)
           CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           !              --
           EPSILN(ITYPE)=0.0D0
           EPSILN(JTYPE)=0.0D0
           EPSILN(ITYPE) = EPSILN(ITYPE) - eps
           EPSILN(JTYPE) = EPSILN(JTYPE) - eps2
           CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           chi2r = ngeo*NGRID * Chi2RESP( NATOM,ISLCT1,CG ,TARGET &
                ,ISDRUDE, REPD,LPARAB,LHYPER, BHYPER ,DHYPER &
                ,FLAT,DFLAT, WRESTR,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
           ddchi2r = ddchi2r + chi2r / (4.0*eps*eps2)
           CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           !              +-
           EPSILN(ITYPE)=0.0D0
           EPSILN(JTYPE)=0.0D0
           EPSILN(ITYPE) = EPSILN(ITYPE) + eps
           EPSILN(JTYPE) = EPSILN(JTYPE) - eps2
           CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           chi2r = ngeo*NGRID * Chi2RESP( NATOM,ISLCT1,CG ,TARGET &
                ,ISDRUDE, REPD,LPARAB,LHYPER, BHYPER ,DHYPER &
                ,FLAT,DFLAT, WRESTR,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
           ddchi2r = ddchi2r - chi2r / (4.0*eps*eps2)
           CALL MODCG( EPSILN,.FALSE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           !              -+
           EPSILN(ITYPE)=0.0D0
           EPSILN(JTYPE)=0.0D0
           EPSILN(ITYPE) = EPSILN(ITYPE) - eps
           EPSILN(JTYPE) = EPSILN(JTYPE) + eps2
           CALL MODCG( EPSILN,.TRUE.,CTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.TRUE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           chi2r = ngeo*NGRID * Chi2RESP( NATOM,ISLCT1,CG ,TARGET &
                ,ISDRUDE, REPD,LPARAB,LHYPER, BHYPER ,DHYPER &
                ,FLAT,DFLAT, WRESTR,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT)
           ddchi2r = ddchi2r - chi2r / (4.0*eps*eps2)
           CALL MODCG(EPSILN,.FALSE.,NTYPES, &
                CONSTC,FITCL,MULTIP,CONSTR,EQUIV, &
                NFRAGM,AFRAGM)
           CALL MODTH( EPSILN,.FALSE., NTYPES, CTYPES, THOLECT, &
                DRUDEY, QVTHOLE)
           !              ALPHA
           ALPHA(ITYPE,JTYPE)=ALPHA(ITYPE,JTYPE)+0.5*ddchi2r
           IF(JTYPE.NE.ITYPE) THEN
              !                    upper triangle
              ALPHA(JTYPE,ITYPE) = ALPHA(JTYPE,ITYPE) + 0.5*ddchi2r
           ENDIF
        ENDDO
     ENDDO
  ENDIF
  IF(RESTR) THEN
     do ia=1,natom
        CG(ia) = copyCG(ia)
     enddo
     call chmdealloc('fitcharge.src','POTCHI2','COPYCG',NATOM,crl=COPYCG)
  ENDIF
  CHI2=CHI2+chi2restr

  RETURN
END SUBROUTINE POTCHI2


FUNCTION Chi2REPD( NATOM,ISLCT1,CG,TARGET,ISDRUDE, &
     REPD,LPARAB,LHYPER, BHYPER,DHYPER,FLAT,DFLAT, &
     WRESTR,GRIDX,GRIDY,GRIDZ,NGRID,X,Y,Z, &
     AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT ) result(chi2repd_rslt)

  !     Arguments
  real(chm_real) :: chi2repd_rslt
  INTEGER NATOM
  INTEGER ISLCT1(NATOM)
  real(chm_real)  CG(*),TARGET(*)
  LOGICAL ISDRUDE(*)
  LOGICAL REPD,LPARAB,LHYPER
  real(chm_real)  BHYPER,DHYPER,FLAT,DFLAT
  real(chm_real)  AWEIGHT,AXX,AYY,AZZ
  real(chm_real)  WRESTR(*)

  real(chm_real)  GRIDX(*),GRIDY(*),GRIDZ(*) ! FORMAT 1..NGRID
  INTEGER NGRID
  real(chm_real)  X(*), Y(*), Z(*)
  real(chm_real)  KDRUDEFIT(*)

  !     Local variables
  INTEGER i,ia
  INTEGER g
  real(chm_real)  rr2,r2,rd2
  real(chm_real)  shp,SHP2,SHP3,SHP4

  CALL SCFDIP(KDRUDEFIT)

  Chi2repd_Rslt=0.0D0
  DO IA=1,NATOM
     IF(ISLCT1(IA).NE.1) GOTO 100
     IF(WRESTR(IA).EQ.0.0D0) GOTO 100

     !        Compute 1/(r_gi)^2 or 1/(r_pgi)^2
     RR2=0.0D0
     IF(ISDRUDE(i)) THEN
        DO G=1,NGRID
           rd2 = (X(i)-GRIDX(g))**2 + (Y(i)-GRIDY(g))**2 &
                + (Z(i)-GRIDZ(g))**2
           r2 = (X(i-1)-GRIDX(g))**2 + (Y(i-1)-GRIDY(g))**2 &
                + (Z(i-1)-GRIDZ(g))**2
           rr2 = rr2 + (1.0/sqrt(rd2) - 1/sqrt(r2))**2
        ENDDO
     ELSE
        DO G=1,NGRID
           r2 = (X(i-1)-GRIDX(g))**2 + (Y(i-1)-GRIDY(g))**2 &
                + (Z(i-1)-GRIDZ(g))**2
           rr2 = rr2 + 1.0/r2
        ENDDO
     ENDIF

     CALL PENLTY( CG,TARGET,ISDRUDE,ia,NATOM, &
          REPD,LPARAB,LHYPER, &
          BHYPER,DHYPER,FLAT,DFLAT, shp )
     Chi2repd_Rslt = Chi2repd_Rslt + ABS(WRESTR(ia)) * shp * rr2
100  CONTINUE
  ENDDO
  !
  !     Add polarizability restraint
  !      SHP2 = ZERO
  !      SHP3 = ZERO
  !      SHP4 = ZERO
  !      CALL ALPHAPENLTY(LPARAB,LHYPER,SHP2,SHP3,SHP4,AXX,AYY,AZZ,
  !     $                 KDRUDEFIT)
  !      Chi2repd_Rslt=Chi2repd_Rslt+ AWEIGHT*(SHP2 + SHP3 + SHP4)*rr2
  !
  RETURN
END FUNCTION Chi2REPD


FUNCTION Chi2RESP( NATOM,ISLCT1,CG,TARGET,ISDRUDE, &
     REPD,LPARAB,LHYPER, BHYPER,DHYPER,FLAT,DFLAT, &
     WRESTR,AWEIGHT,AXX,AYY,AZZ,KDRUDEFIT ) result(chi2resp_rslt)
  real(chm_real) :: chi2resp_rslt
  !     ... Arguments
  INTEGER NATOM
  INTEGER ISLCT1(NATOM)
  real(chm_real)  CG(*),TARGET(*)
  LOGICAL ISDRUDE(*)
  LOGICAL REPD,LPARAB,LHYPER
  real(chm_real)  BHYPER,DHYPER,FLAT,DFLAT
  real(chm_real)  WRESTR(*)
  real(chm_real)  AWEIGHT,AXX,AYY,AZZ
  real(chm_real)  KDRUDEFIT(*)
  !     ... Local variables
  INTEGER ia
  real(chm_real)  SHP,SHP2,SHP3,SHP4
  !
  Chi2resp_Rslt=0.0D0
  DO IA=1,NATOM
     IF(ISLCT1(IA).NE.1) GOTO 100
     IF(WRESTR(IA).EQ.0.0D0) GOTO 100
     CALL PENLTY(CG,TARGET,ISDRUDE,IA,NATOM, &
          REPD,LPARAB,LHYPER, &
          BHYPER,DHYPER,FLAT,DFLAT,SHP)
     Chi2resp_Rslt=Chi2resp_Rslt+ABS(WRESTR(IA))*SHP
100  CONTINUE
  ENDDO
  !
  !     Add polarizability restraint
  !      SHP2 = ZERO
  !      SHP3 = ZERO
  !      SHP4 = ZERO
  !      CALL ALPHAPENLTY(LPARAB,LHYPER,SHP2,SHP3,SHP4,AXX,AYY,AZZ,
  !     $                 KDRUDEFIT)
  !      Chi2resp_Rslt=Chi2resp_Rslt+ AWEIGHT*(SHP2 + SHP3 + SHP4)
  !
  RETURN
END FUNCTION Chi2RESP

SUBROUTINE POTGRID(GRIDX,GRIDY,GRIDZ,NGRID,ISLCT,MPOT,IGEO, &
     MXNGRD,MXNGEO)
  !  Purpose:
  !     Compute electrostatic potential created by selected atoms
  !
  !  Arguments:
  !
  !     GRIDX, GRIDY, GRIDZ
  !     NGRID
  !
  !     ISLCT       (input)  REAL array, dimension (NATOM)
  !                          Atoms contributing to the potential
  !
  !     MPOT        (output) REAL array, dimension (NGRID,ngeo)
  !                          Model potential
  !
  !     IGEO        (input)  INTEGER
  !                          Geometry index (to fill "MPOT")
  use dimens_fcm
  use psf
  use coord
  !     ...Local parameters
  INTEGER MXNGEO, MXNGRD
  !     ...Arguments
  real(chm_real)  GRIDX(MXNGRD),GRIDY(MXNGRD),GRIDZ(MXNGRD)
  INTEGER NGRID
  INTEGER ISLCT(*)
  real(chm_real)  MPOT(MXNGRD,MXNGEO)
  INTEGER IGEO
  !     ...Local variables
  INTEGER IA, IGRID
  real(chm_real)  DIST
  !
  DO IGRID=1,NGRID
     MPOT(IGRID,IGEO)=0.0D0
     DO IA=1,NATOM
        IF(ISLCT(IA).EQ.1) THEN
           DIST=SQRT((X(IA)-GRIDX(IGRID))**2 + &
                (Y(IA)-GRIDY(IGRID))**2 + (Z(IA)-GRIDZ(IGRID))**2 )
           MPOT(IGRID,IGEO)=MPOT(IGRID,IGEO)+CG(IA) / DIST
        ENDIF
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE POTGRID

SUBROUTINE PENLTY(CG,TARGET,ISDRUDE,I,NATOM, &
     REPD,LPARAB,LHYPER,BHYPER,DHYPER,FLAT,DFLAT,SHP)
  !     ...Arguments
  real(chm_real)  CG(*),TARGET(*)
  LOGICAL ISDRUDE(*)
  INTEGER I,NATOM
  LOGICAL REPD,LPARAB,LHYPER
  real(chm_real)  BHYPER,DHYPER,FLAT,DFLAT
  real(chm_real)  SHP
  !     ...Local variables
  real(chm_real)  Q,Q2,B,FL
  !
  IF(I.LT.NATOM.AND.ISDRUDE(I+1)) THEN
     IF(LHYPER) THEN
        ! Restrain to zero
        Q=CG(I)+CG(I+1)
     ELSE
        ! Restrain to target charges
        Q=(CG(I)+CG(I+1))-(TARGET(I)+TARGET(I+1))
     ENDIF
  ELSE
     IF(ISDRUDE(I).AND.REPD) THEN
        !              Problem is CG is too small...
        Q=CG(I)-TARGET(I)**2 / CG(I)
        IF(ABS(CG(I)).LT.0.000001D0) THEN
           CALL WRNDIE(-4,'<Penalty>','CG is too small')
        ENDIF
     ELSE
        IF(LHYPER) THEN
           ! Restrain to zero; applied when hyperbolic penalty
           Q=CG(I)
        ELSE
           ! Restrain to target charges; applied when parabolic penalty
           Q=CG(I)-TARGET(I)
        ENDIF
     ENDIF
  ENDIF
  IF(ISDRUDE(I)) THEN
     FL=DFLAT
  ELSE
     FL=FLAT
  ENDIF
  IF(Q.LT.-FL) THEN
     Q2=Q+FL
  ELSEIF(Q.GT.FL) THEN
     Q2=Q-FL
  ELSE
     Q2=0.0D0
  ENDIF
  !
  IF(LPARAB) THEN
     SHP=Q2*Q2
  ELSEIF(LHYPER) THEN
     IF(ISDRUDE(I)) THEN
        B=DHYPER
     ELSE
        B=BHYPER
     ENDIF
     SHP=SQRT(Q2*Q2+B*B)-B
  ENDIF
  RETURN
END SUBROUTINE PENLTY

SUBROUTINE ALPHAPENLTY(LPARAB,LHYPER,SHP2,SHP3,SHP4,AXX,AYY,AZZ, &
     KDRUDEFIT)
  !     Compute a restraint on fitting for polarizability tensor
  use dimens_fcm
  use psf
  use consta
  use coord
  use number
  !     Arguments
  real(chm_real) SHP2,SHP3,SHP4,CGTEMP,XTEMP,YTEMP,ZTEMP
  real(chm_real) AXX,AYY,AZZ
  LOGICAL LPARAB,LHYPER
  real(chm_real)  DIP(4)
  real(chm_real)  DIP1(4)
  real(chm_real)  MALP(4)
  INTEGER IA
  real(chm_real) KDRUDEFIT(*)
  !
  IF(LPARAB) THEN
     !
     !  Save test charge magnitude and position
     CGTEMP = CG(NATOM)
     XTEMP = X(NATOM)
     YTEMP = Y(NATOM)
     ZTEMP = Z(NATOM)
     !  Relax Drudes
     CG(NATOM)=ZERO
     CALL SCFDIP(KDRUDEFIT)
     !
     !  Calculate Dipole moment
     DIP(1)=ZERO
     DIP(2)=ZERO
     DIP(3)=ZERO
     DO IA=1,NATOM
        DIP(1)=DIP(1)+X(IA)*CG(IA)
        DIP(2)=DIP(2)+Y(IA)*CG(IA)
        DIP(3)=DIP(3)+Z(IA)*CG(IA)
     ENDDO
     DIP(1)=DIP(1)*DEBYEC
     DIP(2)=DIP(2)*DEBYEC
     DIP(3)=DIP(3)*DEBYEC
     !
     !  Calculate the molecular polarizability
     CG(NATOM) = 100000.0
     X(NATOM) = 1000.0
     Y(NATOM) = 0.0
     Z(NATOM) = 0.0
     !  Relax Drudes
     CALL SCFDIP(KDRUDEFIT)
     !
     CG(NATOM)=ZERO
     DIP1(1)=ZERO
     DO IA=1,NATOM
        DIP1(1)=DIP1(1)+X(IA)*CG(IA)
     ENDDO
     DIP1(1)=DIP1(1)*DEBYEC

     CG(NATOM) = 100000.0
     X(NATOM) = 0.0
     Y(NATOM) = 1000.0
     Z(NATOM) = 0.0
     !  Relax Drudes
     CALL SCFDIP(KDRUDEFIT)
     !
     CG(NATOM)=ZERO
     DIP1(2)=ZERO
     DO IA=1,NATOM
        DIP1(2)=DIP1(2)+Y(IA)*CG(IA)
     ENDDO
     DIP1(2)=DIP1(2)*DEBYEC

     CG(NATOM) = 100000.0
     X(NATOM) = 0.0
     Y(NATOM) = 0.0
     Z(NATOM) = 1000.0
     !  Relax Drudes
     CALL SCFDIP(KDRUDEFIT)
     !
     CG(NATOM)=ZERO
     DIP1(3)=ZERO
     DO IA=1,NATOM
        DIP1(3)=DIP1(3)+Z(IA)*CG(IA)
     ENDDO
     DIP1(3)=DIP1(3)*DEBYEC

     !  ALPHA components
     MALP(1)=(DIP1(1) - DIP(1))/0.48
     MALP(2)=(DIP1(2) - DIP(2))/0.48
     MALP(3)=(DIP1(3) - DIP(3))/0.48
     MALP(4) = (MALP(1) + MALP(2) + MALP(3))/3
     !
     !  Reset the original test charge magnitude and position
     CG(NATOM) = CGTEMP
     X(NATOM) = XTEMP
     Y(NATOM) = YTEMP
     Z(NATOM) = ZTEMP
     SHP2=(ABS(MALP(1)) - ABS(AXX))**2
     SHP3=(ABS(MALP(2)) - ABS(AYY))**2
     SHP4=(ABS(MALP(3)) - ABS(AZZ))**2
  ENDIF
  RETURN
END SUBROUTINE ALPHAPENLTY

SUBROUTINE CONDSV(SV,SM,N,M,NMAX,MMAX,CONDNUM)
  use stream
  !     Purpose:
  !     Remove singular modes having "too small" singular values.
  !
  !     Arguments:
  !     sv         (input) REAL array, dimension (n)
  !                        Singular values (output of "svdcmp")
  !     sm         (input/output) REAL array, dimension (m,n)
  !                        Matrix of singular modes (output of "svdcmp")
  !     n,m        (input) INTEGER
  !                        Logical dimensions of arrays
  !     nmax,mmax  (input) INTEGER
  !                        Physical dimensions of arrays
  !     condnum    (input) REAL
  !                        Condition number
  !
  !     Note:
  !     From "xsvbksb.for" (Numerical Recipes)
  !
  !     ...Arguments
  INTEGER N,M,NMAX,MMAX
  real(chm_real)  SV(MMAX)
  real(chm_real)  SM(NMAX,MMAX)
  real(chm_real)  CONDNUM
  !     ...Local variables
  INTEGER I,J
  real(chm_real)  MIN_SV,MAX_SV
  !
  MAX_SV=0.0D0
  DO I=1,N
     !        Singular value = sv(i)
     IF(SV(I).GT.MAX_SV) MAX_SV=SV(I)
     !        Singular mode:
     !        WRITE(OUTU,'(7F15.6)') (SM(J,I),J =1,M)
  ENDDO
  !     Condition number = condnum
  MIN_SV=CONDNUM*MAX_SV
  !     All modes with singular value lower than MIN_SV will be removed
  DO I=1,N
     IF(SV(I).LT.MIN_SV) THEN
        SV(I)=0.0D0
        if (prnlev >= 2) then
           WRITE(OUTU,'(A,I4,A)')  'Removing singular mode',i,' ...'
           WRITE(OUTU,'(A)') 'Singular value ='
           WRITE(OUTU,'(F16.8)') sv
        endif
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE CONDSV

SUBROUTINE ALPRIME(ALPHAPRIME,ALPHA,N,LAMBDA)
  !     ...Arguments
  INTEGER N
  real(chm_real) ALPHAPRIME(N,N)
  real(chm_real) ALPHA(N,N)
  real(chm_real) LAMBDA
  !     ...Locals
  INTEGER I,J
  !     Begin
  DO I=1,N
     DO J=1,N
        ALPHAPRIME(I,J)=ALPHA(I,J)
     ENDDO
     ALPHAPRIME(I,I)=ALPHAPRIME(I,I) * (1.0+LAMBDA)
  ENDDO
  RETURN
END SUBROUTINE ALPRIME

SUBROUTINE CROUND(ISLCT1,ISLCT2,UPRT,TPRT,EQUIV,ASCALE,RDIPOL, &
     ITER,NGRIDP,NTYPES,CHI2,CHI2RESTR,LAMBDA, &
     THOLECT,DRUDEY,KDRUDEFIT, QVTHOLE)
  !
  ! Perform charge rounding
  use dimens_fcm
  use psf
  use stream
  use consta
  use coord
  use number
  use param
  !        Selection of atoms to fit
  INTEGER ISLCT1(NATOM), ISLCT2(NATOM)
  INTEGER UPRT
  INTEGER TPRT
  INTEGER EQUIV(NATOM)
  real(chm_real)  ASCALE, RDIPOL
  !        Number of LM iterations
  !        Total number of gridpoints
  !        Number of parameters to fit
  !        Total chi2
  !        chi2 associated to restraints
  INTEGER I, ITER, NITER
  INTEGER NGRIDP
  INTEGER NTYPES
  real(chm_real)  CHI2
  real(chm_real)  CHI2RESTR
  real(chm_real)  LAMBDA
  INTEGER COUNTER
  real(chm_real)  THOLECT(*)
  LOGICAL DRUDEY(*)
  real(chm_real)  KDRUDEFIT(*)
  LOGICAL QVTHOLE
  !
  INTEGER IA,JA,NONEQ
  real(chm_real)  SUMR,SUMT,CHARGE,ALPHA,TOTALC,QSCALE
  real(chm_real)  DIPA(4),DIPB(4),DIPC(4)
  CHARACTER(len=16) STRING
  CHARACTER(len=20) FORMAT1
  CHARACTER(len=20) FORMAT2
  CHARACTER(len=20) FORMAT3
  CHARACTER(len=20) FORMAT4
  CHARACTER(len=20) FORMAT5
  !
  ! FORMAT selection: f7.3 or f12.8
  !     FORMAT1='(F12.8)'
  !     FORMAT2='(A4,2(2X,F12.8))'
  !     FORMAT3='(A,F12.8)'
  !     FORMAT4='(A,F12.8,A,A4,A)'
  !     FORMAT5='(A,A4,A,F12.8,A)'
  !
  FORMAT1='(F7.3)'
  FORMAT2='(A4,2(2X,F7.3))'
  FORMAT3='(A,F7.3)'
  FORMAT4='(A,F7.3,A,A4,A)'
  FORMAT5='(A,A4,A,F7.3,A)'
  !
  STRING=' '
  !
  !  Remove charge from CAL atoms
  !  Calculate total molecular charge
  TOTALC=ZERO
  DO IA=1,NATOM
     IF(INDEX(ATYPE(IA),'CAL').GT.0) CG(IA)=ZERO
     IF(ISLCT2(IA).EQ.1) TOTALC=TOTALC+CG(IA)
  ENDDO
  !  Perform rounding
  WRITE(STRING,FORMAT1) TOTALC
  READ(STRING,FORMAT1) TOTALC
  !
  !  Calculate dipole moment before (DIPB) scaling (rounding)
  CALL FCDIPO(DIPB,KDRUDEFIT)
  !
  IF(NDRUDE.GT.0) THEN
     !  Polarizability scaling and rounding of Drudes included into Selection2
     IF(ABS(ASCALE) .LT. 0.1D0) ASCALE=1.0D0
     DO IA=1,NATOM-1
        IF(.NOT.ISDRUDE(IA).AND.ISDRUDE(IA+1).AND.ISLCT2(IA).EQ.1)THEN
           ALPHA=ASCALE*CCELEC*CG(IA+1)**2 / (2.0D0*KDRUDEFIT(IA))
           WRITE(STRING,FORMAT1) ALPHA
           READ(STRING,FORMAT1) ALPHA
           CHARGE=SIGN(SQRT(ABS( &
                ALPHA*TWO*KDRUDEFIT(IA) / CCELEC)), CG(IA+1))
           CG(IA)=CG(IA)+CG(IA+1)-CHARGE
           CG(IA+1)=CHARGE
        ENDIF
     ENDDO
     !
     !  Print current charges and polarizabilities to stdout
     if (prnlev >= 2) then
        WRITE(OUTU,'(/A)') ' After polarizability scaling and rounding:'
     endif
     DO IA=1,NATOM
        IF(ATYPE(IA).NE.'CAL' .AND. .NOT.ISDRUDE(IA)) THEN
           IF(IA.LT.NATOM .AND. ISDRUDE(IA+1)) THEN
              !                     ... polarizable atom
              ALPHA=CCELEC * CG(IA+1)**2 / (2.0*KDRUDEFIT(IA))
              if (prnlev >= 2) then
                 WRITE(OUTU,'(A4,2F10.6)') ATYPE(IA),CG(IA)+CG(IA+1),ALPHA
              endif
           ELSE
              !                     ... non-polarizable atom
              if (prnlev >= 2) then
                 WRITE(OUTU,'(A4,2F10.6)') ATYPE(IA),CG(IA)
              endif
           ENDIF
        ENDIF
     ENDDO
  ENDIF
  !
  !  Calculate dipole moment after (DIPA) polarizability scaling (rounding)
  CALL FCDIPO(DIPA,KDRUDEFIT)
  !
  IF(RDIPOL.GT.0.01D0 .AND. DIPA(4).GT.0.01D0) THEN
     QSCALE=RDIPOL/DIPA(4)
  ELSE
     QSCALE=1.0D0
  ENDIF
  !
  !  Find first nonequivalent atom.
  !  This atom will be assigned to absorb the charge rounding error
  NONEQ=0
  loop100: DO IA=1,NATOM
     IF(ISLCT1(IA).EQ.1.AND..NOT.ISDRUDE(IA)) THEN
        IF(NONEQ.EQ.0) NONEQ=IA
        DO JA=1,NATOM
           IF(EQUIV(JA).EQ.EQUIV(IA).AND.JA.NE.IA) cycle loop100
        ENDDO
        NONEQ=IA
        exit loop100
     ENDIF
  enddo loop100
  !
  !  Scale and round atomic charges in Selection2 group
  SUMR=ZERO
  IA=1
  do while(IA.LE.NATOM)
     IF(ISLCT2(IA).EQ.1 .AND. .NOT.ISDRUDE(IA)) THEN
        IF(IA.LT.NATOM .AND. ISDRUDE(IA+1)) THEN
           !              ... polarizable atom
           WRITE(STRING,FORMAT1) (CG(IA)+CG(IA+1)) * QSCALE
           READ(STRING,FORMAT1) CHARGE
           CG(IA)=CHARGE-CG(IA+1)
           IA=IA+1
        ELSE
           !              ... non-polarizable atom
           WRITE(STRING,FORMAT1) CG(IA)*QSCALE
           READ(STRING,FORMAT1) CHARGE
           CG(IA)=CHARGE
        ENDIF
        SUMR=SUMR+CHARGE
     ENDIF
     IA=IA+1
  enddo
  !
  !  Neutralize accummulated rounding error to preserve charge in ISLCT1 group
  CHARGE=TOTALC-SUMR
  IF(ABS(CHARGE).GE.1.0D-6 .AND. NONEQ.GT.0) THEN
     CG(NONEQ)=CG(NONEQ)+CHARGE
  ENDIF
  !
  !  Print final charges and polarizabilities to stdout
  SUMT=ZERO
  if (prnlev >= 2) then
     WRITE(OUTU,'(/A)') ' Final atomic charges and polarizabilities'
  endif
  DO IA=1,NATOM
     IF(ISLCT2(IA).EQ.1.AND..NOT.ISDRUDE(IA)) THEN
        IF(IA.LT.NATOM.AND.ISDRUDE(IA+1)) THEN
           !                     ... polarizable atom
           ALPHA=CCELEC*CG(IA+1)**2 / (2.0*KDRUDEFIT(IA))
           if (prnlev >= 2) then
              WRITE(OUTU,FORMAT2) ATYPE(IA),CG(IA)+CG(IA+1),ALPHA
           endif
           SUMT=SUMT+CG(IA)+CG(IA+1)
        ELSE
           !                     ... non-polarizable atom
           if (prnlev >= 2) then
              WRITE(OUTU,FORMAT2) ATYPE(IA),CG(IA)
           endif
           SUMT=SUMT+CG(IA)
        ENDIF
     ENDIF
  ENDDO
  !
  if (prnlev >= 2) then
     WRITE(OUTU,FORMAT3) ' Total Charge =',SUMT
     IF(ABS(CHARGE).LT.1.D-6) THEN
        WRITE(OUTU,FORMAT3) ' Charge on variable atoms = ',SUMR
     ELSE
        IF(NONEQ.GT.0) THEN
           WRITE(OUTU,FORMAT4) ' Charge difference = ', CHARGE, &
                ' was added to the atom ',ATYPE(NONEQ)
        ELSE
           WRITE(OUTU,FORMAT4) ' Charge difference = ', CHARGE, &
                ' remains uncompensated due to equivalence constraints'
        ENDIF
     ENDIF
  endif
  !
  !  Calculate final dipole moment
  CALL FCDIPO(DIPC,KDRUDEFIT)
  if (prnlev >= 2) then
     WRITE(OUTU,'(3(/A,4F10.4))') &
          'Dipole before       scaling/rounding: ',(DIPB(IA),IA=1,4), &
          'Dipole after ALPHA  scaling/rounding: ',(DIPA(IA),IA=1,4), &
          'Dipole after charge scaling/rounding: ',(DIPC(IA),IA=1,4)
  endif
  !
  if (prnlev >= 2) then
     WRITE(OUTU,'(A,3F9.4)') &
          'Scaling parameters: aScale, qScale, rDipole: ', &
          ASCALE,QSCALE,RDIPOL
  endif
  !
  IF(UPRT.LE.0) RETURN
  !
  !  Print final data to stream file
  WRITE(UPRT,'(A /A,I2 /A,F8.5 /A,F8.5 /A,F6.3,A,A /A,4F8.4 /A/A)') &
       '* Fitted charges and polarizabilities', &
       '* Number of LM iter = ',ITER, &
       '* RMSE              = ',SQRT((CHI2-CHI2RESTR)/(NGRIDP-NTYPES)), &
       '* Lambda            = ', LAMBDA, &
       '* Charge difference = ', CHARGE, &
       ' was added to the atom ',ATYPE(NONEQ), &
       '* Dipole (FIXED ATOMS, RELAXED DRUDES)=',(DIPC(I),I=1,4), &
       '*',' '
  !
  !  Print charges
  WRITE(UPRT,'(/A)') '! Fitted charges'
  DO IA=1,NATOM
     IF(ATYPE(IA).NE.'CAL' .AND. .NOT.ISDRUDE(IA)) THEN
        IF(IA.LT.NATOM .AND. ISDRUDE(IA+1)) THEN
           !                     ... polarizable atom
           WRITE(UPRT,FORMAT4) 'scalar charge set ',CG(IA)+CG(IA+1), &
                ' select resn @residue .and. type ',ATYPE(IA),' END'
        ELSE
           !                     ... non-polarizable atom
           WRITE(UPRT,FORMAT4) 'scalar charge set ',CG(IA), &
                ' select resn @residue .and. type ',ATYPE(IA),' END'
        ENDIF
     ENDIF
  ENDDO
  !
  !  Print polarizabilities
  WRITE(UPRT,'(/A)') '! Fitted polarizabilities'
  DO IA=1,NATOM
     IF(ATYPE(IA).NE.'CAL' .AND. .NOT.ISDRUDE(IA)) THEN
        IF(IA.LT.NATOM .AND. ISDRUDE(IA+1)) THEN
           !                     ... polarizable atom
           ALPHA=-CCELEC*CG(IA+1)**2 / (2.0*KDRUDEFIT(IA))
           WRITE(UPRT,FORMAT4) &
                'scalar wmain  set ',ALPHA, &
                '  select resn @residue .and. type ',ATYPE(IA),' END'
        ELSE
           !                     ... non-polarizable atom
           WRITE(UPRT,FORMAT4) &
                'scalar wmain  set ', 0.0d0, &
                '  select resn @residue .and. type ',ATYPE(IA),' END'
        ENDIF
     ENDIF
  ENDDO
  !
  !   Print the tholei parameters
  WRITE(UPRT,'(/A)') '! Fitted tholei parameters'
  DO IA = 1,NATC
     IF (DRUDEY(IA).AND.QVTHOLE) THEN
        WRITE(UPRT,'(A,F7.3,A,A7,A)') &
             'scalar wcomp  set ',THOLECT(IA), &
             '  select resn @residue .and. chem ',ATC(IA),' END'
     ENDIF
  ENDDO
  !
  WRITE(UPRT,'(/A/A/A//A)') &
       '!Manual scaling:', &
       '!scalar wmain  mult @ASCALE select segid @residue END', &
       '!scalar charge mult @QSCALE select segid @residue END','RETURN'

  !
  IF(TPRT.LE.0) RETURN

  RETURN
END SUBROUTINE CROUND


SUBROUTINE SVDCMP2(A,M,N,MP,NP,W,V)
  !  (C) Copr. 1986-92 Numerical Recipes Software +).
  use number
  use stream
  INTEGER M,MP,N,NP
  real(chm_real) A(MP,NP),V(NP,NP),W(NP)
  INTEGER, PARAMETER :: NMAX=500
  !U    USES pythag
  INTEGER I,ITS,J,JJ,K,L,NM
  real(chm_real) ANORM,C,F,G,H,S,SCALE,X,Y,Z,RV1(NMAX)
  G=0.0D0
  SCALE=0.0D0
  ANORM=0.0D0
  loop25: DO I=1,N
     L=I+1
     RV1(I)=SCALE*G
     G=0.0D0
     S=0.0D0
     SCALE=0.0D0
     IF(I.LE.M)THEN
        DO K=I,M
           SCALE=SCALE+ABS(A(K,I))
        enddo
        IF(SCALE.NE.0.0D0)THEN
           DO K=I,M
              A(K,I)=A(K,I)/SCALE
              S=S+A(K,I)*A(K,I)
           enddo
           F=A(I,I)
           G=-SIGN(SQRT(S),F)
           H=F*G-S
           A(I,I)=F-G
           DO J=L,N
              S=0.0D0
              DO K=I,M
                 S=S+A(K,I)*A(K,J)
              enddo
              F=S/H
              DO K=I,M
                 A(K,J)=A(K,J)+F*A(K,I)
              enddo
           enddo
           A(i:m,I)=SCALE*A(i:m,I)
        ENDIF
     ENDIF
     W(I)=SCALE*G
     G=0.0D0
     S=0.0D0
     SCALE=0.0D0
     IF((I.LE.M).AND.(I.NE.N))THEN
        DO K=L,N
           SCALE=SCALE+ABS(A(I,K))
        enddo
        IF(SCALE.NE.0.0D0)THEN
           DO K=L,N
              A(I,K)=A(I,K)/SCALE
              S=S+A(I,K)*A(I,K)
           enddo
           F=A(I,L)
           G=-SIGN(SQRT(S),F)
           H=F*G-S
           A(I,L)=F-G

           RV1(l:n)=A(I,l:n)/H

           DO J=L,M
              S=0.0D0
              DO K=L,N
                 S=S+A(J,K)*A(I,K)
              enddo
              DO K=L,N
                 A(J,K)=A(J,K)+S*RV1(K)
              enddo
           enddo
           DO K=L,N
              A(I,K)=SCALE*A(I,K)
           enddo
        ENDIF
     ENDIF
     ANORM=MAX(ANORM,(ABS(W(I))+ABS(RV1(I))))
  enddo loop25
  DO I=N,1,-1
     IF(I.LT.N)THEN
        IF(G.NE.0.0D0)THEN
           V(l:n,I)=(A(I,l:n)/A(I,L))/G
           DO J=L,N
              S=0.0D0
              DO K=L,N
                 S=S+A(I,K)*V(K,J)
              enddo
              V(l:n,J)=V(l:n,J)+S*V(l:n,I)
           enddo
        ENDIF
        DO J=L,N
           V(I,J)=0.0D0
           V(J,I)=0.0D0
        enddo
     ENDIF
     V(I,I)=1.0D0
     G=RV1(I)
     L=I
  enddo
  DO I=MIN(M,N),1,-1
     L=I+1
     G=W(I)
     A(I,l:n)=0.0D0
     IF(G.NE.0.0D0)THEN
        G=1.0/G
        DO J=L,N
           S=0.0D0
           DO K=L,M
              S=S+A(K,I)*A(K,J)
           enddo
           F=(S/A(I,I))*G
           DO K=I,M
              A(K,J)=A(K,J)+F*A(K,I)
           enddo
        enddo
        A(i:m,I)=A(i:m,I)*G
     ELSE
        A(i:m,I)=zero
     ENDIF
     A(I,I)=A(I,I)+one
  enddo
  loop49: DO K=N,1,-1
     loop48: DO ITS=1,30
        loop41: DO L=K,1,-1
           NM=L-1
           IF((ABS(RV1(L))+ANORM).EQ.ANORM)  GOTO 2
           IF((ABS(W(NM))+ANORM).EQ.ANORM)  exit loop41
        enddo loop41
        C=0.0D0
        S=1.0D0
        loop43: DO I=L,K
           F=S*RV1(I)
           RV1(I)=C*RV1(I)
           IF((ABS(F)+ANORM).EQ.ANORM) exit loop43
           G=W(I)
           H=PYTHAG(F,G)
           W(I)=H
           H=1.0D0/H
           C=(G*H)
           S=-(F*H)
           DO J=1,M
              Y=A(J,NM)
              Z=A(J,I)
              A(J,NM)=(Y*C)+(Z*S)
              A(J,I)=-(Y*S)+(Z*C)
           enddo
        enddo loop43
2       Z=W(K)
        IF(L.EQ.K)THEN
           IF(Z.LT.0.0D0)THEN
              W(K)=-Z
              V(1:n,K)=-V(1:n,K)
           ENDIF
           GOTO 3
        ENDIF
        IF(ITS.EQ.30) THEN
           if (prnlev >= 2) then
              WRITE(OUTU,'(A)') &
                   'Warning: FITCHARGE: no convergence in svdcmp2'
           endif
        ENDIF
        X=W(L)
        NM=K-1
        Y=W(NM)
        G=RV1(NM)
        H=RV1(K)
        F=((Y-Z)*(Y+Z)+(G-H)*(G+H))/(2.0D0*H*Y)
        G=PYTHAG(F,ONE)
        F=((X-Z)*(X+Z)+H*((Y/(F+SIGN(G,F)))-H))/X
        C=1.0D0
        S=1.0D0
        DO J=L,NM
           I=J+1
           G=RV1(I)
           Y=W(I)
           H=S*G
           G=C*G
           Z=PYTHAG(F,H)
           RV1(J)=Z
           C=F/Z
           S=H/Z
           F= (X*C)+(G*S)
           G=-(X*S)+(G*C)
           H=Y*S
           Y=Y*C
           DO JJ=1,N
              X=V(JJ,J)
              Z=V(JJ,I)
              V(JJ,J)= (X*C)+(Z*S)
              V(JJ,I)=-(X*S)+(Z*C)
           enddo
           Z=PYTHAG(F,H)
           W(J)=Z
           IF(Z.NE.0.0D0)THEN
              Z=1.0D0/Z
              C=F*Z
              S=H*Z
           ENDIF
           F= (C*G)+(S*Y)
           X=-(S*G)+(C*Y)
           DO JJ=1,M
              Y=A(JJ,J)
              Z=A(JJ,I)
              A(JJ,J)= (Y*C)+(Z*S)
              A(JJ,I)=-(Y*S)+(Z*C)
           enddo
        enddo
        RV1(L)=0.0D0
        RV1(K)=F
        W(K)=X
     enddo loop48
3    CONTINUE
  enddo loop49
  RETURN
END SUBROUTINE SVDCMP2


SUBROUTINE SVBKSB(U,W,V,M,N,MP,NP,B,X)
  !  (C) Copr. 1986-92 Numerical Recipes Software +).
  INTEGER M,MP,N,NP
  real(chm_real) B(MP),U(MP,NP),V(NP,NP),W(NP),X(NP)
  INTEGER, PARAMETER :: NMAX=500
  INTEGER I,J,JJ
  real(chm_real) S,TMP(NMAX)
  DO J=1,N
     S=0.D0
     IF(W(J).NE.0.)THEN
        DO I=1,M
           S=S+U(I,J)*B(I)
        enddo
        S=S/W(J)
     ENDIF
     TMP(J)=S
  enddo
  DO J=1,N
     S=0.D0
     DO JJ=1,N
        S=S+V(J,JJ)*TMP(JJ)
     enddo
     X(J)=S
  enddo
  RETURN
END SUBROUTINE SVBKSB

FUNCTION PYTHAG(A,B)
  !  (C) Copr. 1986-92 Numerical Recipes Software +).
  real(chm_real) A,B,PYTHAG
  real(chm_real) ABSA,ABSB
  ABSA=ABS(A)
  ABSB=ABS(B)
  IF(ABSA.GT.ABSB) THEN
     PYTHAG=ABSA*SQRT(1.+(ABSB/ABSA)**2)
     !        PYTHAG=SQRT(ABSA*ABSA + ABSB*ABSB)
  ELSE
     IF(ABSB.EQ.0.D0) THEN
        PYTHAG=0.D0
     ELSE
        PYTHAG=ABSB*SQRT(1.+(ABSA/ABSB)**2)
        !          PYTHAG=SQRT(ABSA*ABSA + ABSB*ABSB)
     ENDIF
  ENDIF
  RETURN
END FUNCTION PYTHAG

SUBROUTINE SCFDIP(KDRUDEFIT)
  !  Purpose:
  !     Change position of Drude particles to minimize potential energy
  !     Minimize energy with ABNR algorithm.
  !
  use dimens_fcm
  use number
  use coord
  use psf
  use memory
  use minmiz_module, only: minmiz
  use stream
  use string

  real(chm_real)  KDRUDEFIT(*)
  !     ... Local variables
  INTEGER IA
  CHARACTER(len=80) COMLYN2
  INTEGER COMLEN2
  !         Copy of original IMOVE
  integer,allocatable,dimension(:) :: IMOVES
  !         Copy of original PRNLEV
  INTEGER PRNLEVS
  !
  IF(NDRUDE.EQ.0) RETURN
  !
  CALL UTHOLE(KDRUDEFIT)
  !
  !     Let move only Drude particles
  call chmalloc('fitcharge.src','SCFDIP','IMOVES',NATOM,intg=IMOVES)
  do ia=1,natom
     IMOVES(ia) = IMOVE(ia)
  enddo
  DO IA=1,NATOM
     !            fixed
     IMOVE(IA)=1
     !            mobile
     IF(ISDRUDE(IA)) THEN
        IMOVE(IA)=0
        !            reset Drude coordinates
        X(IA)=X(IA-1)
        Y(IA)=Y(IA-1)
        Z(IA)=Z(IA-1)
     ENDIF
  ENDDO
  PRNLEVS=PRNLEV
  PRNLEV=0
  !
  !     Minimize energy
  !      COMLYN2='ABNR STEP 0.001 NSTEPS 1000 CUTNB 9999. CTOFNB 9990.'
  COMLYN2='ABNR STEP 0.001 NSTEPS 1000 CUTNB 9999. CTOFNB 9990.'
  COMLEN2=STRLNG(COMLYN2)
  CALL MINMIZ(COMLYN2,COMLEN2)
  !
  !     Restore original settings
  DO IA=1,NATOM
     IMOVE(IA)=IMOVES(IA)
  ENDDO
  PRNLEV=PRNLEVS
  call chmdealloc('fitcharge.src','SCFDIP','IMOVES',NATOM,intg=IMOVES)
  !
  RETURN
END SUBROUTINE SCFDIP


SUBROUTINE FCDIPO(DIP,KDRUDEFIT)
  ! Calculate dipole moment for FITCHARGE routine
  use dimens_fcm
  use psf
  use consta
  use coord
  use number

  real(chm_real)  DIP(4)
  real(chm_real)  KDRUDEFIT(*)
  INTEGER IA
  !
  !  Relax Drudes
  CALL SCFDIP(KDRUDEFIT)
  !
  !  Calculate Dipole moment
  DIP(1)=ZERO
  DIP(2)=ZERO
  DIP(3)=ZERO
  DO IA=1,NATOM
     DIP(1)=DIP(1)+X(IA)*CG(IA)
     DIP(2)=DIP(2)+Y(IA)*CG(IA)
     DIP(3)=DIP(3)+Z(IA)*CG(IA)
  ENDDO
  DIP(1)=DIP(1)*DEBYEC
  DIP(2)=DIP(2)*DEBYEC
  DIP(3)=DIP(3)*DEBYEC
  DIP(4)=SQRT(DIP(1)*DIP(1)+DIP(2)*DIP(2)+DIP(3)*DIP(3))
  !
  RETURN
END SUBROUTINE FCDIPO


SUBROUTINE UTHOLE(KDRUDEFIT)
  !  Purpose:
  !     Update Thole screening FUNCTION
  !     This routine must be invoked each time the atomic or
  !          Drude charges are modified
  !     Small changes in charges do not require Thole update
  !          (for example: numerical differentiation)
  !
  use dimens_fcm
  use number
  use consta
  use psf
  !
  real(chm_real)  KDRUDEFIT(*)
  !
  INTEGER IA
  ! Thole constant removed from this update command
  ! enters into ETHOLE energy term in drude.src
  !
  IF(NDRUDE.GT.0) THEN
     !      IF(NDRUDE.GT.0.and.QTHOLE) THEN
     !        Update ALPHADP
     DO IA=1,NATOM
        ALPHADP(IA)=ZERO
     ENDDO
     DO IA=1,NATOM
        IF(ISDRUDE(IA)) THEN
           ALPHADP(IA-1)=CCELEC * CG(IA)**2 / (2.0*KDRUDEFIT(IA-1))
        ENDIF
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE UTHOLE


! Levenberg-Marquardt Non-linear minimization
SUBROUTINE LEVMARQ(ALPHA,BETA,DA,N,LAMBDA)
  use memory
  use stream

  INTEGER N
  real(chm_real) ALPHA(N,N)
  real(chm_real) BETA(N)
  real(chm_real) DA(N)
  real(chm_real) LAMBDA
  real(chm_real) SVDZERO
  integer i,j
  real(chm_real),allocatable,dimension(:) :: ALPRIM
  real(chm_real),allocatable,dimension(:) :: W_NR
  real(chm_real),allocatable,dimension(:) :: V_NR
  !
  call chmalloc('fitcharge.src','LEVMARQ','ALPRIM',N*N,crl=ALPRIM)
  call chmalloc('fitcharge.src','LEVMARQ','W_NR',N,crl=W_NR)
  call chmalloc('fitcharge.src','LEVMARQ','V_NR',N*N,crl=V_NR)
  !
  SVDZERO = 1.0E-10
  call ALPRIME(ALPRIM,ALPHA,N,LAMBDA)
  call SVDCMP2(ALPRIM,N,N,N,N,W_NR,V_NR)
  call CONDSV(W_NR,ALPRIM,N,N,N,N,SVDZERO)
  call SVBKSB(ALPRIM,W_NR,V_NR,N,N,N,N,BETA,DA)
  !
  call chmdealloc('fitcharge.src','LEVMARQ','ALPRIM',N*N,crl=ALPRIM)
  call chmdealloc('fitcharge.src','LEVMARQ','W_NR',N,crl=W_NR)
  call chmdealloc('fitcharge.src','LEVMARQ','V_NR',N*N,crl=V_NR)
  !
  RETURN
END SUBROUTINE LEVMARQ
#endif /* (fitchg)*/

#if KEY_FITCHG==0 /*fitparam*/
SUBROUTINE FITPARAM
  !-----------------------------------------------------------------------
  CALL WRNDIE(-1,'<FITPARAM>','FITCHG code is not compiled.')
  return
end SUBROUTINE FITPARAM

#else /* (fitparam)*/

SUBROUTINE FITPARAM
  !-----------------------------------------------------------------------
  use dimens_fcm
  use comand
  use stream
  use string
  use memory
  use number

  real(chm_real)  LAMBDA
  real(chm_real)  CHI2, CHI20, CONV, CHI2RESTR, CHI2RESTR2
  INTEGER ITER, COUNTER2
  INTEGER I,J
  real(chm_real)  DELTA, TOLERA
  character(len=20),dimension(:),allocatable :: labels,labexp
  INTEGER NITER,COUNTER,NPAR,NEXP,NGRP,NDIP,NDER,UEXP,UFUN
  LOGICAL LANTOINE
  CHARACTER(len=256) FGUES,FPARM,FEXPD,FFVAL,FEXPW,FMULT,FREST,FREXT
  INTEGER      FGUESLEN, FPARMLEN, FEXPDLEN, FFVALLEN, FEXPWLEN
  INTEGER      FMULTLEN, FRESTLEN

  real(chm_real),allocatable,dimension(:) :: ALPHA
  real(chm_real),allocatable,dimension(:) :: ALPHA2
  real(chm_real),allocatable,dimension(:) :: DA
  real(chm_real),allocatable,dimension(:) :: BETA
  real(chm_real),allocatable,dimension(:) :: BETA2
  real(chm_real),allocatable,dimension(:) :: PARAMS
  real(chm_real),allocatable,dimension(:) :: EXPDATA
  real(chm_real),allocatable,dimension(:) :: FVALUE
  real(chm_real),allocatable,dimension(:) :: EXPWEIGHT
  real(chm_real),allocatable,dimension(:) :: DERIVA
  real(chm_real),allocatable,dimension(:) :: WEIGHT
  real(chm_real),allocatable,dimension(:) :: RESTRAINT
  integer,allocatable,dimension(:) :: MULTIPL
  integer,allocatable,dimension(:) :: GROUP
  integer,allocatable,dimension(:) :: IDIPOLE
  real(chm_real),allocatable,dimension(:) :: DIPEXP
  real(chm_real),allocatable,dimension(:) :: DIPMOD
  real(chm_real),allocatable,dimension(:) :: DIPDER
  real(chm_real),allocatable,dimension(:) :: XVALUE
  !
  if (prnlev >= 2) then
     WRITE(OUTU,'(A)') ' '
     WRITE(OUTU,'(A)')  &
          'FITPARAM> Restrained fitting of nonlinear function'
  endif
  !
  !     Start (iteration zero)
  LAMBDA = 0.05D0
  ITER = 0
  COUNTER2 = 0
  CHI20 = 0.0D0
  !
  TOLERA= GTRMF(COMLYN,COMLEN,'TOLE',PT0001)
  NITER=GTRMI(COMLYN,COMLEN,'NITE',100)
  COUNTER=GTRMI(COMLYN,COMLEN,'COUN',2)
  NPAR=GTRMI(COMLYN,COMLEN,'NPAR',0)
  NEXP=GTRMI(COMLYN,COMLEN,'NEXP',0)
  NGRP=GTRMI(COMLYN,COMLEN,'NGRP',1)
  NDIP=GTRMI(COMLYN,COMLEN,'NDIP',0)
  UFUN=UEXP+1
  !
  IF(NEXP.EQ.0) THEN
     if (prnlev >= 2) then
        WRITE(OUTU,'(A)') "Error: no parameters to optimize"
     endif
     STOP
  ENDIF
  IF(NEXP.EQ.0) THEN
     if (prnlev >= 2) then
        WRITE(OUTU,'(A)') "Error: no data to fit"
     endif
     STOP
  ENDIF
  !
  ! Allocate memory
  !
  call chmalloc('fitcharge.src','FITPARAM','ALPHA',NPAR*NPAR,crl=ALPHA)
  call chmalloc('fitcharge.src','FITPARAM','ALPHA2',NPAR*NPAR,crl=ALPHA2)
  call chmalloc('fitcharge.src','FITPARAM','DA',NPAR,crl=DA)
  call chmalloc('fitcharge.src','FITPARAM','BETA',NPAR,crl=BETA)
  call chmalloc('fitcharge.src','FITPARAM','BETA2',NPAR,crl=BETA2)
  call chmalloc('fitcharge.src','FITPARAM','PARAMS',NPAR,crl=PARAMS)
  call chmalloc('fitcharge.src','FITPARAM','EXPDATA',NEXP,crl=EXPDATA)
  call chmalloc('fitcharge.src','FITPARAM','FVALUE',NEXP,crl=FVALUE)
  call chmalloc('fitcharge.src','FITPARAM','EXPWEIGHT',NEXP,crl=EXPWEIGHT)
  call chmalloc('fitcharge.src','FITPARAM','DERIVA',NEXP*NPAR,crl=DERIVA)
  call chmalloc('fitcharge.src','FITPARAM','WEIGHT',NPAR,crl=WEIGHT)
  call chmalloc('fitcharge.src','FITPARAM','RESTRAINT',NPAR,crl=RESTRAINT)
  call chmalloc('fitcharge.src','FITPARAM','MULTIPL',NPAR,intg=MULTIPL)
  call chmalloc('fitcharge.src','FITPARAM','GROUP',NGRP,intg=GROUP)
  IF(NDIP.GT.0) THEN
     NDER = NDIP * NPAR
     call chmalloc('fitcharge.src','FITPARAM','IDIPOLE',NDIP,intg=IDIPOLE)
     call chmalloc('fitcharge.src','FITPARAM','DIPEXP',3*NDIP,crl=DIPEXP)
     call chmalloc('fitcharge.src','FITPARAM','DIPMOD',3*NDIP,crl=DIPMOD)
     call chmalloc('fitcharge.src','FITPARAM','DIPDER',3*NDER,crl=DIPDER)
  ENDIF
  GROUP = 0
  !
  ! Check for Antoine parameter fitting
  !
  LANTOINE=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'ANTO').GT.0) LANTOINE = .TRUE.
  IF(LANTOINE .AND. NPAR.NE.2 .AND. NPAR.NE.3) THEN
     if (prnlev >= 2) then
        WRITE(OUTU,'(A,I3)')  &
             "Error: Wrong number of Antoine parameters: ",NPAR
        WRITE(OUTU,'(A)') "Can be 2 or 3 only"
     endif
     STOP
  ENDIF
  !
  ! Allocate memory for exp.data, param.labels, and exp.labels
  !
  call chmalloc('fitcharge.src','FITPARAM','XVALUE',NEXP,crl=XVALUE)
  !     ...allocate character*20 array on C-stack
  call chmalloc('fitcharge.src','FITPARAM','labels',Npar,ch20=labels)
  call chmalloc('fitcharge.src','FITPARAM','labels',NEXP,ch20=labexp)
  !
  ! Mandatory Input Files
  !
  CALL GETFNAME('GUES',FGUES,256,FGUESLEN,.TRUE.)
  CALL GETFNAME('EXPD',FEXPD,256,FEXPDLEN,.TRUE.)
  CALL GETFNAME('PARM',FPARM,256,FPARMLEN,.TRUE.)
  IF(.NOT.LANTOINE) THEN
     CALL GETFNAME('FVAL',FFVAL,256,FFVALLEN,.FALSE.)
     IF(INDXA(COMLYN,COMLEN,'REXT').EQ.0) THEN
        if (prnlev >= 2) then
           WRITE(OUTU,'(A)')  &
                "Error: No ext.script file name. See keyword REXT"
        endif
        STOP
     ELSE
        CALL GETBETW(COMLYN,'"','"',FREXT)
     ENDIF
  ENDIF
  !
  ! Optional Input Files
  !
  CALL GETFNAME('EXPW',FEXPW,256,FEXPWLEN,.FALSE.)
  CALL GETFNAME('MULT',FMULT,256,FMULTLEN,.FALSE.)
  CALL GETFNAME('REST',FREST,256,FRESTLEN,.FALSE.)
  !
  ! Read in experimental data,initial parameter values, and restraint weights
  call REXPAR(NEXP,NPAR,EXPDATA,PARAMS,WEIGHT,LABELS, &
       MULTIPL,GROUP,NGRP,LABEXP,EXPWEIGHT,XVALUE, &
       FGUES,FEXPD,FEXPW,FMULT,FREST,NDIP,IDIPOLE,DIPEXP)
  RESTRAINT = PARAMS
  !
  DO ITER=1,NITER
     !             Compute derivatives, Alpha and Beta
     call COMPDERI(UFUN,ALPHA,BETA,NEXP,NPAR,EXPDATA,FVALUE, &
          DERIVA,CHI2,LAMBDA,PARAMS,WEIGHT,RESTRAINT,ITER, &
          LABELS,MULTIPL,GROUP,NGRP,LABEXP,EXPWEIGHT, &
          XVALUE,LANTOINE,FFVAL,FREXT,FPARM,NDIP,NDER,IDIPOLE, &
          DIPEXP,DIPDER,DIPMOD)
     !
     CONV=2.0D0*ABS(CHI20-CHI2)/(CHI20+CHI2)
     !                        Count the number of successful iterations
     IF(CONV.LT.TOLERA) THEN
        COUNTER2 = COUNTER2 + 1
        !                        Test convergence of iteration
        IF(COUNTER2 .GE. COUNTER) THEN
           if (prnlev >= 2) then
              WRITE(OUTU,'(A)') "FITPARAM> Convergent LM algorithm"
           endif
           GOTO 200
        ENDIF
     ELSE
        COUNTER2 = 0
     ENDIF

     IF(ITER.GT.1) THEN
        if(CHI2 .LE. CHI20) then
           LAMBDA = LAMBDA / 10.0D0
           CHI20 = CHI2
           CHI2RESTR = CHI2RESTR2
           !                          Save Alpha and Beta to be used when next iteration fails
           beta2 = beta
           alpha2 = alpha
        ELSE
           LAMBDA = LAMBDA * 10.0D0
           IF(LAMBDA .GT. 999999.9) GOTO 100
           if (prnlev >= 2) then
              WRITE(OUTU,'(A,I3,A,F13.6)')  &
                   " Iteration ",iter," failed; increasing lambda = ",lambda
              !                          Go back to previous charges; restore Alpha and Beta
           endif
           beta = beta2
           alpha = alpha2
        ENDIF
     ELSE
        CHI20 = CHI2
     ENDIF
     !                          Solution of "ALPHA*da=beta"
     call LEVMARQ(ALPHA,BETA,DA,NPAR,LAMBDA)
     !                          Modify parameters by "da"
     call UPDPAR(PARAMS,DA,NPAR,.TRUE.,LABELS,FPARM)
  ENDDO
100 CONTINUE
  if (prnlev >= 2) then
     WRITE(OUTU,'(A)') "FITPARAM> Non-convergent LM algorithm"
  endif
200 CONTINUE
  !
  !     Free memory
  call chmdealloc('fitcharge.src','FITPARAM','ALPHA',NPAR*NPAR,crl=ALPHA)
  call chmdealloc('fitcharge.src','FITPARAM','ALPHA2',NPAR*NPAR,crl=ALPHA2)
  call chmdealloc('fitcharge.src','FITPARAM','DA',NPAR,crl=DA)
  call chmdealloc('fitcharge.src','FITPARAM','BETA',NPAR,crl=BETA)
  call chmdealloc('fitcharge.src','FITPARAM','BETA2',NPAR,crl=BETA2)
  call chmdealloc('fitcharge.src','FITPARAM','PARAMS',NPAR,crl=PARAMS)
  call chmdealloc('fitcharge.src','FITPARAM','EXPDATA',NEXP,crl=EXPDATA)
  call chmdealloc('fitcharge.src','FITPARAM','FVALUE',NEXP,crl=FVALUE)
  call chmdealloc('fitcharge.src','FITPARAM','EXPWEIGHT',NEXP,crl=EXPWEIGHT)
  call chmdealloc('fitcharge.src','FITPARAM','DERIVA',NEXP*NPAR,crl=DERIVA)
  call chmdealloc('fitcharge.src','FITPARAM','WEIGHT',NPAR,crl=WEIGHT)
  call chmdealloc('fitcharge.src','FITPARAM','RESTRAINT',NPAR,crl=RESTRAINT)
  call chmdealloc('fitcharge.src','FITPARAM','MULTIPL',NPAR,intg=MULTIPL)
  call chmdealloc('fitcharge.src','FITPARAM','GROUP',NGRP,intg=GROUP)
  IF(NDIP.GT.0) THEN
     NDER = NDIP * NPAR
     call chmdealloc('fitcharge.src','FITPARAM','IDIPOLE',NDIP,intg=IDIPOLE)
     call chmdealloc('fitcharge.src','FITPARAM','DIPEXP',3*NDIP,crl=DIPEXP)
     call chmdealloc('fitcharge.src','FITPARAM','DIPMOD',3*NDIP,crl=DIPMOD)
     call chmdealloc('fitcharge.src','FITPARAM','DIPDER',3*NDER,crl=DIPDER)
  ENDIF
  call chmdealloc('fitcharge.src','FITPARAM','XVALUE',NEXP,crl=XVALUE)
  call chmdealloc('fitcharge.src','FITPARAM','labels',Npar,ch20=labels)
  call chmdealloc('fitcharge.src','FITPARAM','labels',NEXP,ch20=labexp)
  !
  RETURN
END SUBROUTINE FITPARAM

SUBROUTINE REXPAR(NEXP,NPAR,EXPDATA,PARAMS,WEIGHT,LABELS,MULTIPL, &
     GROUP,NGRP,LABEXP,EXPWEIGHT,XVALUE, &
     FGUES,FEXPD,FEXPW,FMULT,FREST, &
     NDIP,IDIPOLE,DIPEXP)
  ! Read in experimental data and initial value of parameters
  use stream
  use number
  use machio,only:vopen
  INTEGER UNIT,NEXP,NPAR,NGRP,NDIP,I,J,K,IDIP
  real(chm_real)  EXPDATA(NEXP),PARAMS(NPAR),WEIGHT(NPAR)
  CHARACTER(len=20) LABELS(NPAR)
  CHARACTER(len=20) LABEXP(NEXP)
  INTEGER MULTIPL(NPAR), GROUP(NGRP)
  INTEGER IDIPOLE(NDIP)
  real(chm_real)  DIPEXP(3,NDIP)
  real(chm_real)  EXPWEIGHT(NEXP)
  real(chm_real)  XVALUE(NEXP)
  CHARACTER(len=256) FGUES, FEXPD, FEXPW, FMULT, FREST
  LOGICAL ERR, LDIPO
  INTEGER OLDPRN
  CHARACTER(len=256) STR
  real(chm_real)  NULL
  !
  OLDPRN=PRNLEV
  PRNLEV=0
  UNIT=10
  NULL=0.0d0
  !
  ! Read in Experimental weights
  CALL RDRSCALAR(UNIT,NEXP,EXPWEIGHT,ONE,FEXPW)
  !
  ! Read in Experimental data
  CALL VOPEN(UNIT,FEXPD,'FORMATTED','READ',ERR,0)
  IF(ERR) THEN
     WRITE(outu,'(A,A)') "Error: file is not present ",FEXPD
     STOP
  ENDIF
  WRITE(OUTU,'(A)') "FITP> Experimental data:"
  WRITE(OUTU,'(A)') "                       weight    value"
  IDIP=0
  DO I=1,NEXP
10   READ(UNIT,'(A80)') STR
     IF(STR(1:1).EQ.'*' .OR. STR(1:1).EQ.'!' .OR. STR(1:1).EQ.'#') &
          GOTO 10
     XVALUE(I) = NULL
     CALL RDSTR(STR,XVALUE(I),EXPDATA(I),LABEXP(I), &
          LDIPO,IDIP,NDIP,DIPEXP)
     IF(LDIPO) THEN
        IDIPOLE(IDIP) = I
        WRITE(OUTU,'(A20,F9.2,3F12.6)')  &
             LABEXP(I),EXPWEIGHT(I),(DIPEXP(K,IDIP),K=1,3)
     ELSE
        IF(XVALUE(I).EQ.NULL) THEN
           WRITE(OUTU,'(A20,F9.2,F12.6)')  &
                LABEXP(I),EXPWEIGHT(I),EXPDATA(I)
        ELSE
           WRITE(OUTU,'(A20,F9.2,2F12.6)')  &
                LABEXP(I),EXPWEIGHT(I),XVALUE(I),EXPDATA(I)
        ENDIF
     ENDIF
  ENDDO
  !         check the number of dipole moments
  IF(IDIP.NE.NDIP) THEN
     WRITE(OUTU,'(A,2I4)')  &
          "Error: wrong number of dipole moments ",idip,ndip
     STOP
  ENDIF
  CALL VCLOSE(UNIT,'KEEP',ERR)
  !
  ! Parameter values
  CALL VOPEN(UNIT,FGUES,'FORMATTED','READ',ERR,0)
  IF(ERR) THEN
     WRITE(OUTU,'(A,A)') "Error: file is not present ",FGUES
     STOP
  ENDIF
  DO I=1,NPAR
20   READ(UNIT,'(A80)') STR
     IF(STR(1:1).EQ.'*' .OR. STR(1:1).EQ.'!' .OR. STR(1:1).EQ.'#')  &
          GOTO 20
     CALL RDSTRPAR(STR,LABELS(I),PARAMS(I))
  ENDDO
  CALL VCLOSE(UNIT,'KEEP',ERR)
  !
  ! Restraint weights
  CALL RDRSCALAR(UNIT,NPAR,WEIGHT,ZERO,FREST)
  !
  ! Parameter multiplicity values
  K=1
  CALL VOPEN(UNIT,FMULT,'FORMATTED','READ',ERR,0)
  IF(ERR) THEN
     DO I=1,NPAR
        MULTIPL(I) = 1.0D0
     ENDDO
  ELSE
     DO I=1,NPAR
        READ(UNIT,'(A80)') STR
        IF(INDEX(STR,"group").GT.0.OR.INDEX(STR,"GROUP").GT.0) THEN
           READ(UNIT,'(A80)') STR
           IF(I.GT.1) K=K+1
        ENDIF
        GROUP(K)=I
        READ(STR,*) MULTIPL(I)
     ENDDO
  ENDIF
  CALL VCLOSE(UNIT,'KEEP',ERR)
  !
  PRNLEV=OLDPRN
  RETURN
END SUBROUTINE REXPAR


SUBROUTINE COMPDERI(UFUN,ALPHA,BETA,NEXP,NPAR, &
     EXPDATA,FVALUE,DERIVA,CHI2,LAMBDA,PARAMS,WEIGHT,RESTRAINT,ITER, &
     LABELS,MULTIPL,GROUP,NGRP,LABEXP,EXPWEIGHT,XVALUE,LANTOINE, &
     FFVAL,FREXT,FPARM, &
     NDIP,NDER,IDIPOLE,DIPEXP,DIPDER,DIPMOD)
  use dimens_fcm
  use stream

  INTEGER UFUN,NEXP,NPAR,NGRP,NDIP,NDER
  real(chm_real)  ALPHA(NPAR,NPAR),BETA(NPAR)
  real(chm_real)  EXPDATA(NEXP),FVALUE(NEXP),DERIVA(NEXP,NPAR), &
       XVALUE(NEXP)
  real(chm_real)  CHI2, RMS
  real(chm_real)  LAMBDA
  real(chm_real)  PARAMS(NPAR), WEIGHT(NPAR), RESTRAINT(NPAR)
  INTEGER MULTIPL(NPAR), GROUP(NGRP)
  CHARACTER(len=256) FFVAL, FREXT, FPARM
  INTEGER ITER, IDIP, JDIP
  CHARACTER(len=20) LABELS(NPAR)
  CHARACTER(len=20) LABEXP(NEXP)
  CHARACTER(len=20) ALABEL
  INTEGER IDIPOLE(NDIP)
  real(chm_real)  DIPEXP(3,NDIP), DIPMOD(3,NDIP), DIPDER(3,NDER)
  real(chm_real)  EXPWEIGHT(NEXP)
  LOGICAL LANTOINE
  CHARACTER(len=256) STR
  LOGICAL LDIPO,LDNDIFF
  !
  CHARACTER(len=256) COMLYN2
  INTEGER COMLEN2
  LOGICAL LUSED,ERR
  INTEGER I,J,K, LINE, OLDPRN
  real(chm_real)  DELTA, SUM, DIFF
  !
  ! Zero the arrays
  DO J=1,NPAR
     BETA(J)=0.0D0
     DO K=1,NPAR
        ALPHA(K,J)=0.0D0
     ENDDO
  ENDDO
  !
  if (prnlev >= 2) WRITE(OUTU,'(A/A,I3)') "--------------","Iteration: ",iter
  ! Compute function value and derivatives
  IF(LANTOINE) THEN
     CALL ANTOINE(UFUN,NEXP,NPAR,XVALUE,FVALUE,DERIVA,PARAMS)
  ELSE
     CALL RUNEXTP(UFUN,NEXP,NPAR, &
          FVALUE,DERIVA,NDIP,NDER,FFVAL,FREXT,FPARM, &
          IDIPOLE,DIPEXP,DIPDER,DIPMOD)
  ENDIF

  ! Compute Alpha, Beta, RMS, and chi^2
  IDIP = 1
  CHI2 = 0.0D0
  RMS  = 0.0D0
  DO I=1,NEXP
     IF(NDIP.GT.0 .AND. IDIP.LE.NDIP) THEN
        IF(I.EQ.IDIPOLE(IDIP)) THEN
           DIFF=(DIPEXP(1,IDIP)*DIPMOD(1,IDIP) +  &
                DIPEXP(2,IDIP)*DIPMOD(2,IDIP) +  &
                DIPEXP(3,IDIP)*DIPMOD(3,IDIP) ) / SQRT( &
                (DIPEXP(1,IDIP)**2+DIPEXP(2,IDIP)**2+DIPEXP(3,IDIP)**2) * &
                (DIPMOD(1,IDIP)**2+DIPMOD(2,IDIP)**2+DIPMOD(3,IDIP)**2) )
           DIFF = 1.0D0 - DIFF
           IDIP = IDIP + 1
        ELSE
           DIFF = EXPDATA(I) - FVALUE(I)
        ENDIF
     ELSE
        DIFF = EXPDATA(I) - FVALUE(I)
     ENDIF
     CHI2  = CHI2 + EXPWEIGHT(I) * DIFF ** 2
     RMS   = RMS  + DIFF ** 2
     DELTA = EXPWEIGHT(I) * DIFF
     DO J=1,NPAR
        ! Add parabolic restraints
        BETA(J) = BETA(J) + DELTA * DERIVA(I,J) -  &
             NEXP * WEIGHT(J) * (PARAMS(J)-RESTRAINT(J))
        DO K=1,NPAR
           ALPHA(J,K) = ALPHA(J,K) + DERIVA(I,J) * DERIVA(I,K)
        ENDDO
        ! Add parabolic restraints
        ALPHA(J,J) = ALPHA(J,J) + NEXP * WEIGHT(J)
     ENDDO
  ENDDO
  RMS = SQRT(RMS / NEXP)
  ! Add parabolic restraints to chi^2
  DO J=1,NPAR
     CHI2 = CHI2 + NEXP * WEIGHT(J) * (PARAMS(J)-RESTRAINT(J))**2
  ENDDO
  !
  !             Print status of the current iteration
  K=1
  SUM=0.0D0
  DO I=1,NPAR
     if (prnlev >= 2) WRITE(OUTU,'(I3,A1,2X,A20,F16.8,I5)')  &
          I, ")", LABELS(I), PARAMS(I), MULTIPL(I)
     SUM = SUM - PARAMS(I) * FLOAT(MULTIPL(I))
     IF(I.EQ.GROUP(K)) THEN
        if (prnlev >= 2) WRITE(OUTU,'(A26,F16.8)') "sum", SUM
        SUM=0.0D0
        K=K+1
     ENDIF
  ENDDO
  if (prnlev >= 2) then
     WRITE(OUTU,'(A,F13.6)') "lambda = ",LAMBDA
     WRITE(OUTU,'(A,F13.6)') "chi2   = ",CHI2
     WRITE(OUTU,'(A,F13.6)') "RMSD   = ",RMS
     IF(NDIP.EQ.0) THEN
        WRITE(OUTU,'(20X,A)')   "    exp    model   diff"
     ELSE
        WRITE(OUTU,'(20X,A,A)') "    exp    model   diff", &
             "        expDipoleXYZ      modelDipoleXYZ    Angle"
     ENDIF
  endif
  IDIP = 1
  DO I = 1, NEXP
     LDNDIFF=.false.
     IF (NDIP > 0 .AND. IDIP <= NDIP) THEN
        LDNDIFF = (I == IDIPOLE(IDIP))
     ENDIF
     IF (LDNDIFF) THEN
        DIFF=(DIPEXP(1,IDIP)*DIPMOD(1,IDIP) +  &
             DIPEXP(2,IDIP)*DIPMOD(2,IDIP) +  &
             DIPEXP(3,IDIP)*DIPMOD(3,IDIP) ) / SQRT( &
             (DIPEXP(1,IDIP)**2+DIPEXP(2,IDIP)**2+DIPEXP(3,IDIP)**2) * &
             (DIPMOD(1,IDIP)**2+DIPMOD(2,IDIP)**2+DIPMOD(3,IDIP)**2) )
        DIFF = ACOS(DIFF) * 180.0D0 / 3.14159265358D0
        if (prnlev >= 2) WRITE(OUTU,'(A20,24X,2X,3(3F6.2,2X),F6.2)')  &
             LABEXP(I), &
             (DIPEXP(J,IDIP),J=1,3), (DIPMOD(J,IDIP),J=1,3), DIFF
        IDIP = IDIP + 1
     ELSE
        DIFF = FVALUE(I) - EXPDATA(I)
        if (prnlev >= 2) WRITE(OUTU,'(A20,3F8.3)')  &
             LABEXP(I),EXPDATA(I),FVALUE(I), DIFF
     ENDIF
  ENDDO
#if KEY_PATHSCALE==1
  CALL &  
#endif
  FLUSH (OUTU)
  RETURN
END SUBROUTINE COMPDERI


SUBROUTINE RDSTR(STR,XVALUE,YVALUE,LABEL,LDIPO,IDIP,NDIP,DIPOLE)
  use stream

  CHARACTER(len=256) STR
  real(chm_real)       XVALUE, YVALUE
  CHARACTER(len=20) LABEL
  LOGICAL      LDIPO
  INTEGER      IDIP,NDIP
  real(chm_real)       DIPOLE(3,NDIP)
  INTEGER      NSUBS,I,J,K
  !
  ! Count the number of decimal points
  NSUBS = 0
  DO I=1,80
     IF(STR(I:I).EQ.'.') NSUBS = NSUBS + 1
     IF(STR(I:I).EQ.'!') GOTO 10
  ENDDO
10 CONTINUE
  !
  ! Check for comments
  K = INDEX(STR,'!')
  IF(K.EQ.0) K = INDEX(STR,'#')
  IF(K.GT.0) THEN
     K=K-1
  ELSE
     K=80
  ENDIF
  !
  ! Check whether we deal with a scalar or a vector
  IF(NSUBS.EQ.3 .AND. NDIP.GT.0 .AND. IDIP.LT.NDIP) THEN
     !           read dipole moment components
     XVALUE = 0.0D0
     YVALUE = 0.0D0
     IF(IDIP.EQ.NDIP) THEN
        if (prnlev >= 2) WRITE(OUTU,'(A,I3)')  &
             "Number of dipoles exceeded limit ", ndip
        STOP
     ELSE
        IDIP = IDIP + 1
     ENDIF
     READ(STR(:K),*) (DIPOLE(J,IDIP),J=1,3)
     LDIPO=.TRUE.
  ELSEIF(NSUBS.EQ.2) THEN
     !           read two values
     READ(STR(:K),*) XVALUE, YVALUE
     LDIPO=.FALSE.
  ELSE
     !           read a scalar value
     READ(STR(:K),*) YVALUE
     LDIPO=.FALSE.
  ENDIF
  !
  ! Read comments
  K = INDEX(STR,'!')
  IF(K.EQ.0) K = INDEX(STR,'#')
  IF(K.GT.0 .AND. K.LT.80) THEN
     K=K+1
     READ(STR(K:),'(A20)') LABEL
  ELSE
     LABEL = " "
  ENDIF
  RETURN
END SUBROUTINE RDSTR


SUBROUTINE DIPERR(LINE,STR,NDIP,IDIP,IDIPOLE)
  use stream

  CHARACTER(len=256) STR
  INTEGER      LINE, NDIP, IDIP
  INTEGER      IDIPOLE(NDIP)
  if (prnlev >= 2) then
     WRITE(OUTU,'(A)') "Reading the file of derivatives"
     WRITE(OUTU,'(A)') "Error: wrong order of computed properties"
     WRITE(OUTU,'(A,I3)') "Dipole moment index: ", IDIPOLE(IDIP)
     WRITE(OUTU,'(A,I3)') "Total number of dipole moments: ",NDIP
     WRITE(OUTU,'(A,I3)') "Error is found at the line number: ",LINE
     WRITE(OUTU,'(A)')  STR
  endif
  STOP
END SUBROUTINE DIPERR


SUBROUTINE UPDPAR(PARAMS,DA,NPAR,SIGN,LABELS,FPARM)
  use stream
  use machio, only: vopen
  use param_store, only: set_param

  implicit none

  INTEGER NPAR, I
  real(chm_real)  PARAMS(NPAR), DA(NPAR)
  LOGICAL SIGN
  CHARACTER(len=20) LABELS(NPAR)
  INTEGER UNIT, OLDPRN
  LOGICAL ERR
  CHARACTER(len=256) FPARM, PSAVE
  OLDPRN=PRNLEV
  PRNLEV=0
  IF(SIGN) THEN
     DO I=1,NPAR
        PARAMS(I) = PARAMS(I) + DA(I)
     ENDDO
  ELSE
     DO I=1,NPAR
        PARAMS(I) = PARAMS(I) - DA(I)
     ENDDO
  ENDIF
  ! Save new parameters to file
  UNIT=10
  CALL VOPEN(UNIT,FPARM,'FORMATTED','WRITE',ERR,0)
  DO I=1,NPAR
     WRITE(UNIT,'(A20,F16.8)') LABELS(I), PARAMS(I)
     !
     !         Save parameters in CHARMM variables
     !
     IF(I.LE.9) THEN
        WRITE(PSAVE,'(A4,I1)') "FPAR",I
     ELSE
        WRITE(PSAVE,'(A4,I2)') "FPAR",I
     ENDIF
     call set_param(PSAVE,PARAMS(I))
  ENDDO
  CALL VCLOSE(UNIT,'KEEP',ERR)
  PRNLEV=OLDPRN
  RETURN
END SUBROUTINE UPDPAR


SUBROUTINE RDRSCALAR(UNIT,N,RSCALAR,DVALUE,FNAME)
  ! Read an array of scalar vlues of REAL type
  ! Assign default value if the file is not present
  ! Ignore comments starting from '*', '!', or '#' character
  !
  use stream
  use machio,only:vopen

  INTEGER       UNIT,N
  real(chm_real)        RSCALAR(N)
  real(chm_real)        DVALUE
  CHARACTER(len=*) FNAME
  CHARACTER(len=256) STR
  INTEGER       I
  LOGICAL       ERR
  !
  CALL VOPEN(UNIT,FNAME,'FORMATTED','READ',ERR,0)
  IF(ERR) THEN
     DO I=1,N
        RSCALAR(I) = DVALUE
     ENDDO
  ELSE
     DO I=1,N
10      READ(UNIT,*) STR
        IF(STR(1:1).EQ.'*' .OR.STR(1:1).EQ.'!' .OR.STR(1:1).EQ.'#') &
             GOTO 10
        READ(STR,*) RSCALAR(I)
     ENDDO
  ENDIF
  CALL VCLOSE(UNIT,'KEEP',ERR)
  !
  RETURN
END SUBROUTINE RDRSCALAR


SUBROUTINE RUNEXTP(UFUN,NEXP,NPAR, &
     FVALUE,DERIVA,NDIP,NDER,FFVAL,FREXT,FPARM, &
     IDIPOLE,DIPEXP,DIPDER,DIPMOD)
  use dimens_fcm
  use stream
  use machio,only:vopen
  use string

  implicit none

  INTEGER UFUN,NEXP,NPAR,NDIP,NDER
  real(chm_real)  FVALUE(NEXP), DERIVA(NEXP,NPAR)
  CHARACTER(len=256) FFVAL, FREXT, FPARM
  INTEGER IDIP, JDIP
  CHARACTER(len=20) ALABEL
  INTEGER IDIPOLE(NDIP)
  real(chm_real)  DIPEXP(3,NDIP), DIPMOD(3,NDIP), DIPDER(3,NDER)
  real(chm_real)  UNUSED
  CHARACTER(len=256) STR
  LOGICAL LDIPO
  !
  CHARACTER(len=256) COMLYN2
  INTEGER COMLEN2
  LOGICAL LUSED,ERR
  INTEGER I,J, LINE, OLDPRN
  !
  ! execute external program to compute function value and derivatives
  ! using the current parameters
  OLDPRN=PRNLEV
  PRNLEV=0
  LUSED=.TRUE.
  WRITE(COMLYN2,'(A5,A1,A,A1)') 'SYST ','"',TRIM(FREXT),'"'
  COMLEN2=STRLNG(COMLYN2)
  CALL MISCOM(COMLYN2,MXCMSZ,COMLEN2,LUSED)
  !
  ! Read in current function values
  CALL VOPEN(UFUN,FFVAL,'FORMATTED','READ',ERR,0)
  IF(ERR) THEN
     WRITE(OUTU,'(A)') "Error: cannot find derivatives"
     STOP
  ENDIF
  IDIP=0
  LINE=0
  DO I=1,NEXP
     LINE = LINE + 1
     READ(UFUN,'(A80)') STR
     CALL RDSTR(STR,UNUSED,FVALUE(I),ALABEL,LDIPO,IDIP,NDIP,DIPMOD)
     IF(LDIPO .AND. NDIP.GT.0 .AND. IDIPOLE(IDIP).NE.I) THEN
        CALL DIPERR(LINE,STR,NDIP,IDIP,IDIPOLE)
     ENDIF
  ENDDO
  ! Read in derivatives
  IDIP=0
  DO J=1,NPAR
     JDIP = 0
     DO I=1,NEXP
        READ(UFUN,'(A80)') STR
        LINE = LINE + 1
        CALL RDSTR(STR,UNUSED,DERIVA(I,J),ALABEL, &
             LDIPO,IDIP,NDER,DIPDER)
        IF(LDIPO.AND.NDIP.GT.0.AND.JDIP.LE.NDIP.AND.IDIP.LE.NDER) THEN
           ! Derivative of dot product of model dipole (variable) with exp.dipole (constant)
           JDIP = JDIP + 1
           DERIVA(I,J)= -( 1.0d0 -   &
                (DIPEXP(1,JDIP)*DIPMOD(1,JDIP)+  &
                DIPEXP(2,JDIP)*DIPMOD(2,JDIP)+  &
                DIPEXP(3,JDIP)*DIPMOD(3,JDIP)) / SQRT( &
                (DIPEXP(1,JDIP)**2+DIPEXP(2,JDIP)**2+DIPEXP(3,JDIP)**2)* &
                (DIPMOD(1,JDIP)**2+DIPMOD(2,JDIP)**2+DIPMOD(3,JDIP)**2)) ) * &
                (DIPEXP(1,JDIP)*DIPDER(1,IDIP)+  &
                DIPEXP(2,JDIP)*DIPDER(2,IDIP)+  &
                DIPEXP(3,JDIP)*DIPDER(3,IDIP)) / SQRT( &
                (DIPEXP(1,JDIP)**2+DIPEXP(2,JDIP)**2+DIPEXP(3,JDIP)**2) * &
                (DIPDER(1,IDIP)**2+DIPDER(2,IDIP)**2+DIPDER(3,IDIP)**2) )
        ENDIF
     ENDDO
  ENDDO
  CALL VCLOSE(UFUN,'KEEP',ERR)
  PRNLEV=OLDPRN
  RETURN
END SUBROUTINE RUNEXTP


SUBROUTINE ANTOINE(UFUN,NEXP,NPAR,XVALUE,YVALUE,DERIVA,PARAMS)
  use stream

  INTEGER UFUN,NEXP,NPAR
  real(chm_real)  XVALUE(NEXP), YVALUE(NEXP)
  real(chm_real)  DERIVA(NEXP,NPAR), PARAMS(NPAR)
  INTEGER I,J
  !
  ! Compute function values:  Y = A - B / (T + C)
  IF(NPAR.EQ.3) THEN
     DO I=1,NEXP
        YVALUE(I) = PARAMS(1) - PARAMS(2) / (XVALUE(I) + PARAMS(3))
     ENDDO
  ELSEIF(NPAR.EQ.2) THEN
     DO I=1,NEXP
        YVALUE(I) = PARAMS(1) - PARAMS(2) / XVALUE(I)
     ENDDO
  ELSE
     if (prnlev >= 2) then
        WRITE(outu,'(A,I3)')  &
             "Error: Wrong number of Antoine parameters: ",NPAR
        WRITE(outu,'(A)') "Can be 2 or 3 only"
     endif
     STOP
  ENDIF
  !
  ! Compute derivatives
  IF(NPAR.EQ.3) THEN
     DO I=1,NEXP
        DERIVA(I,1) =  1.0D0
        DERIVA(I,2) = -1.0D0 / (XVALUE(I) + PARAMS(3))
        DERIVA(I,3) =  PARAMS(2) / (XVALUE(I) + PARAMS(3)) ** 2
     ENDDO
  ELSE
     DO I=1,NEXP
        DERIVA(I,1) =  1.0D0
        DERIVA(I,2) = -1.0D0 / XVALUE(I)
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE ANTOINE


SUBROUTINE GETFNAME(KEYWD,FNAME,MAXLEN,NAMLEN,QEXIT)
  use dimens_fcm
  use comand
  use stream
  use string

  CHARACTER(len=4)   KEYWD
  CHARACTER(len=*) FNAME
  INTEGER       MAXLEN, NAMLEN
  LOGICAL       QEXIT
  ! local variables
  INTEGER UNIT
  LOGICAL QOPEN, QFORM, QWRITE, ERR
  INTEGER OLDPRN
  CHARACTER(len=256)  FNSTR
  !
  ! inquire file name by unit
  ! return variables: file name, file name length: FNAME, NAMLEN
  !
  OLDPRN=PRNLEV
  PRNLEV=0
  NAMLEN=0
  FNAME=""
  UNIT=GTRMI(COMLYN,COMLEN,KEYWD,-1)
  IF(UNIT.GT.0) THEN
     CALL VINQRE('UNIT',FNSTR,MAXLEN,NAMLEN,QOPEN,QFORM,QWRITE,UNIT)
     CALL VCLOSE(UNIT,'KEEP',ERR)
     !         add double quotes to protect character case
     WRITE(FNAME,'(A1,A,A1)') '"',TRIM(FNSTR),'"'
     NAMLEN=NAMLEN+2
  ELSEIF(QEXIT) THEN
     WRITE(OUTU,'(A,A)') "Error: File not found. See keyword ", KEYWD
     STOP
  ENDIF
  ! restore original print level
  PRNLEV=OLDPRN
  RETURN
END SUBROUTINE GETFNAME


SUBROUTINE RDSTRPAR(STR,LABEL,PARAM)
  use stream

  CHARACTER(len=256) STR
  CHARACTER(len=20) LABEL
  real(chm_real)       PARAM
  INTEGER      I
  !
  ! Find position of decimal point
  I=INDEX(STR,'.')
  ! Move pointer left until space character is encountered
10 I=I-1
  IF(I.GT.1 .AND. STR(I:I).NE.' ') GOTO 10
  !
  ! Read Label (first) part of string
  IF(I.GE.1) THEN
     READ(STR(:I),'(a20)') LABEL
  ELSE
     if (prnlev >= 2) then
        WRITE(OUTU,'(A)') "Error: Wrong file format"
        WRITE(OUTU,'(A)') STR
     endif
     STOP
  ENDIF
  !
  ! Read Numeric (second) part of string
  READ(STR(I:),*) PARAM
  !
  RETURN
END SUBROUTINE RDSTRPAR

#endif /* (fitparam)*/

end module fitchg

