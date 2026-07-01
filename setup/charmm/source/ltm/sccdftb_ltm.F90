module sccdftb
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_SCCDFTB==1 /*sccdftb_fcm*/
!
!     Defines the data necessary for a QM/MM  calculation on a system.
!
!     Variable  Index    Purpose 
! 
!     MAXCEN             Maximum number of SCCDFTB atoms allowed
!
      integer,PARAMETER :: MAXCEN = 650,MAXN3 = 1950
!

!---mfc--- Needs allocation routines

      real(chm_real)  SCCZIN(maxn3)
      INTEGER NSCCTC,ICHSCC
      INTEGER SCCTYP(maxn3)
      !real(chm_real)  SCCMASS(maxn3)
      CHARACTER(len=10) :: SCCATM(maxn3)
!
!
!     NSCCTC         Number of SCCDFTB atoms
!     ICHSCC         Charge of SCCDFTB part.
!     SCCTYP         Atom type in SCCDFTB, we take from the WMAIN array
!     SCCZIN         Nuc. charge of QM atoms
!     SCCATM         Element name of QM atoms
!     MAXCEN         Maximum number of atoms that SCCDFTB works with
!     MAXPTC         SCCDFTB variable - defines the maximum no. of ptc.
!     COMMON/EXTCHR/ A common block from SCCDFTB that stores information
!                    on MM(CHARMM) atoms, coordinates, charge values
!                    and # of MM atoms.
!     D{X,Y,Z}TBMM   Forces from QM electron wavefunc. to MM atoms
!
!     PARAMETER (MAXPTC=30*MAXCEN)
      INTEGER,PARAMETER :: MAXPTC=251200
!
      real(chm_real) :: CPTC(3,maxptc),ZPTC(maxptc)
      INTEGER NPTC
      CHARACTER(len=2) :: EXTFLAG
!     FOR PARALLEL
      INTEGER MYSCCID


      real(chm_real),dimension(maxptc) :: DXTBMM,DYTBMM,DZTBMM


!PATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINT
!PATHINT     NSCCRP   # of Replica --- usually the primary is deleted!
!PATHINT     LSCCRP   if replica is invoked

      integer,PARAMETER :: MXQM2=100, MXRP=50, MX_STORE=5000
      INTEGER NSCCRP
      LOGICAL LSCCRP
      INTEGER IQMLST(MXQM2,MXRP)

!     Centroid information
!     XYZCMD         coordinate of centroid
!     FRCCMD         froce      on centroid (may not be useful here)
!     NCNSCD         # of constraints
!     IATCNS         List of atoms w/ the cons (limited to 100 for now)
!     WTTCNS         List of atoms w/ the cons (limited to 3 for now)
!     FCNSCD         Force constant for   the constraint (limited to 100)
!     RCNSCD0        Equilibrium value of the constraint (limited to 100)
!     LCNSCD         IF INVOKE CONSTRAINT
!     UNTSCD         FILE UNIT # for constraint output

      real(chm_real) :: XYZCMD(3,mxqm2),FRCCMD(3,mxqm2)
      real(chm_real) :: WTTCNS(3),FCNSCD,RCNSCD0
      INTEGER NCNSCD,IATCNS(100)
      INTEGER UNTSCD
      LOGICAL LCNSCD
      INTEGER NFRQSC1,NFRQSCW,ISCDWRT,JSCDWRT
      real(chm_real) R1SCD(MX_STORE)
      real(chm_real) R2SCD(MX_STORE)

!PATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINTCPATHINT

!     =================================================================
!     QC_UW04: Add Mulliken and eWald
      real(chm_real) QMULIK(MAXCEN)
      real(chm_real) QMULI2(MXQM2,MXRP)
      LOGICAL LMULIK  
!     ---------------------------------------------------------------- 
!     PBC RELATED COMMON BLOCK FROM SCC (dylcao)
      real(chm_real) boxsiz(3,3), xinvbox(3,3), xnullvec(3)
      integer nlat(3) 
      logical period
      integer iuptbox 

!     If choose to optimize para (which is recommended)
      logical LSCOEWD 
      real(chm_real) kappascc
      integer kmxscc,ksqmxscc

!     K-tab
      integer,parameter :: maxkscc=5000
      real(chm_real) kxvec(maxkscc)
      real(chm_real) kyvec(maxkscc)
      real(chm_real) kzvec(maxkscc)
      real(chm_real) kvec (maxkscc)
      integer nkvec
!     XIAO_QC_UW0609
      integer kxv(maxkscc)
      integer kyv(maxkscc)
      integer kzv(maxkscc)

!     SCC LST2 for ewald real space sum
      integer,parameter :: MAXPTR=5000
      real(chm_real) :: CPTR(3,maxptr),ZPTR(maxptr)
      INTEGER NPTR
      INTEGER IMM2LST(maxptr)

      real(chm_real),dimension(maxptr) :: DXTBM2,DYTBM2,DZTBM2
      LOGICAL LSKIMG

!     DIV scheme (Xiao_PHK_QC_UW0609)
      LOGICAL QSCCDIV

!     JZ_UW12: SMBP
      LOGICAL SMBP_GRAD,SCC_CALLED

!     JZ_UW12: read/write hessian
      LOGICAL QSCCREADHESS,QSCCSAVEHESS
!     Guishan_Zheng 03/17/2013
      integer :: igeometry,IMD              ! number of MD steps 
 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     More SCC unique info moved to sccdftbsrc_ltm
!
! icntce is the count of calling Energy
! icntnb is same as INBFRQ. 
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

contains

  subroutine sccdftb_init()
    nscctc=0
    return
  end subroutine sccdftb_init
#endif /* (sccdftb_fcm)*/
!
end module sccdftb

