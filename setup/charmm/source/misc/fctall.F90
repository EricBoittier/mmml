Module facts_module

  use  chm_kinds
  use  chm_types
  use  dimens_fcm
  implicit none

  ! Author: Originally Urs Haberthuer. extended by Francois Marchand
  ! Modified: Francois Marchand
  !           parallelization
  !           FACTS list space management
  !           Solute-solvent dispersion term
  !
  ! Modified: Francois Marchand
  ! Tue Jan 26 10:53:51 CET 2010:
  ! - f95 conversion
  ! - facts_block
  ! - facts_salt
  ! - facts_fix
  !
  ! Modified: Francois Marchand
  ! Thu Jul  1 20:53:00 CEST 2010
  !
  ! modified to be integrated in c36a4
  ! Things still missing or to care:
  !       - Is 'fctscr' necessary? / Who use this? / Cannot it be removed?
  !       - Completion of 'fctprt'
  !             - Salt term
  !             - remove 'fctscr' from 'fctprt'
  !             - print at each energy call (don't call 'FCTINI' at each frame -> memory)
  !       - Automatic detection of image -> allocation of image pair list
  !
  !
  ! Modified: Francois Marchand
  ! Mon Jun  6 18:38:06 CEST 2011
  ! - FACTS compatibility with INTEraction command
  !      can be used to compute screened electrostatic interaction between 2 selected groups
  !      can also compute atomic properties (self-energies & nonpolar (gamma*surface_area) ) on selected groups
  !      can write files with all atomic values for electrostatic self, screened and nonpolar energies
  !
  !
#if KEY_FACTS==1 /*factvars*/
   real(kind=chm_real), parameter :: fctcel = 332.0716d0
   !!!   integer, parameter :: fctmat = chsize  !?? should not be replaced by natom
   integer, parameter :: fctmvw = 250        ! Number of VdW radii

   integer,save  :: mxfcab, mxfcib

   logical, save :: fctrun, fctsfe, fctscr, fctpsl, fctpin ,&
                    fctpsr, fctpal, fctpbt, fctpvw, fctpfr ,&
                    ifast

   logical, save :: fctaim, fctscro, fctslfo, doscreen, qslfunit, qnplunit, qscrunit
   integer,save  :: selfunit, npolunit, scrnunit

   real(kind=chm_real),save ::  fctiem,   fcties, fcttau, fctpco ,&
                                fctqos,   fctuos, fctvos, fctwos ,&
                                fctbin,   fctikp                 ,&
                                fct1ln,   fct2ln, fct3ln         ,&
                                fct2cn,   fct3cn                 ,&
                                fct2ci,   fct3ci                 ,&
                                fct2fn,   fct3fn                 ,&
                                fctvdwwrd                        ,&
                                fctkappa,fctscreen,fcttai,fctk2ew

   ! Direct fixed allocation
   real(kind=chm_real), save :: fctcb1(fctmvw), fctcb2(fctmvw)
   real(kind=chm_real), save :: fctca0(fctmvw)
   real(kind=chm_real), save :: fctca1(fctmvw), fctca2(fctmvw)
   real(kind=chm_real), save :: fctca3(fctmvw)
   real(kind=chm_real), save :: fctcd1(fctmvw), fctcd2(fctmvw)
   real(kind=chm_real), save :: fctcc0(fctmvw)
   real(kind=chm_real), save :: fctcc1(fctmvw), fctcc2(fctmvw)
   real(kind=chm_real), save :: fctcc3(fctmvw)
   real(kind=chm_real), save :: fctcf1(fctmvw), fctcf2(fctmvw)
   real(kind=chm_real), save :: fctce0(fctmvw)
   real(kind=chm_real), save :: fctce1(fctmvw), fctce2(fctmvw)
   real(kind=chm_real), save :: fctce3(fctmvw)
   real(kind=chm_real), save :: fctch1(fctmvw), fctch2(fctmvw)
   real(kind=chm_real), save :: fctcg0(fctmvw)
   real(kind=chm_real), save :: fctcg1(fctmvw), fctcg2(fctmvw)
   real(kind=chm_real), save :: fctcg3(fctmvw)
   real(kind=chm_real), save :: fctqun(fctmvw)
   real(kind=chm_real), save :: fct1cn(fctmvw), fct1ci(fctmvw)
   real(kind=chm_real), save :: fct1fn(fctmvw)
   real(kind=chm_real), save :: fctvol(fctmvw)

   ! Dynamic allocation

   real(kind=chm_real), allocatable, dimension(:), save :: fctcgs, fctnpc, fctvdwfac, fctvdwalp
   real(kind=chm_real), allocatable, dimension(:), save :: fctslfw

   integer, allocatable, dimension(:), save :: fctidx

   ! FACTS Interaction pair lists
   type FactsDataStructure

     integer, dimension(:), allocatable  :: fct1ilo
     integer, dimension(:), allocatable  :: fct1jnb

     integer, dimension(:), allocatable  :: fct2ilo
     integer, dimension(:), allocatable  :: fct2jnb

     integer, dimension(:), allocatable  :: fct3ilo
     integer, dimension(:), allocatable  :: fct3jnb

     integer, dimension(:), allocatable  :: fct4ilo
     integer, dimension(:), allocatable  :: fct4jnb

   end type FactsDataStructure

   type(FactsDataStructure),target,save  :: FCTBND, FCTBNDC

#endif /* (factvars)*/
!--------------------------------------------------------------------
   contains

#if KEY_FACTS==0 /*factz1*/
  SUBROUTINE FCTINI(comlyn,comlen)
    character(len=*), intent(inout) :: comlyn
    integer         , intent(inout) :: comlen
    CALL WRNDIE(-1,'<FACTS>','FACTS CODE NOT COMPILED')
    RETURN
  end SUBROUTINE FCTINI
#else /*  (factz1)*/
  subroutine facts_init()
    fctrun=.false.
    return
  end subroutine facts_init

!--------------------------------------------------------------------
   SUBROUTINE fctini(comlyn,comlen)

   ! This routine parses the FACTS commands in the CHARMM input file and
   ! initializes FACTS.
   !
   ! Author: Urs Haberthuer.
   ! Modified: Francois Marchand
   !           parallelization
   !           FACTS list space management
   !           Solute-solvent dispersion term
   !
   use chm_kinds
   use chm_types
   use dimens_fcm
   use string

   use bases_fcm
   use consta
   use coord
   use eutil
   use image
   use inbnd
   use number
   use param
   use psf
   use stream
#if KEY_PARALLEL==1
   use parallel
#if KEY_STRINGM==1
   use multicom_aux                   /* VO string method*/
#endif
#endif 
   use memory
       implicit none

   character(len=*),intent(inout) :: comlyn
   integer         ,intent(inout) :: comlen

   ! Variables declaration
   real(kind=chm_real) ::     fcteps  ,&
      fctap0, fctap1, fctap2, fctap3  ,&
      fctbt0, fctbt1, fctbt2, fctbt3  ,&
      fctcsl, fctcil, fctcic          ,&
      cutmax, fctsz                   ,&
      fctkps                          ,&
      fctdif                          ,&
      fctpol,      fctnpl             ,&
      fctvdwk,     fctvdwrho          ,&
      fctvdweps,   fctvdwsig, fctvdwa ,&
      fctvdwepsot, fctvdwsigot        ,&
      fctvdwepsht, fctvdwsight        ,&
      fctvdwepsox, fctvdwsigox        ,&
      fctvdwepshx, fctvdwsighx        ,&
      fctconc    , fcttemp,   saltfact,&
      fctdflt

   real(kind=chm_real), allocatable, dimension(:) :: &
        fctrvw, fctcsc, fctscf, fctnps, fctvdwalpha

   character(len=8) :: fctvdwepsotn,fctvdwepshtn
   character(len=1) :: fctdum

   integer :: i,j
   integer :: fctcps,fctfps, fctnnv,fctnav, fctnst
   ! new
   integer :: iacmax,ierr,ierr2, fctszi
   logical :: fctavw, fctcho, fcterr
   ! -------------------------------------------------------------------
   ! Array allocation - local
   ierr2=0
   iacmax=0
   do i=1, natom
      iacmax=max(iac(i),iacmax)
   enddo

   if(allocated(fctcgs))write(outu,*)    "FCTCGS    already allocated"
   if(allocated(fctslfw))  write(outu,*) "FCTSLFW   already allocated"
   if(allocated(fctnpc))write(outu,*)    "FCTNPC    already allocated"
   if(allocated(fctvdwalp))write(outu,*) "FCTVDWFAC already allocated"
   if(allocated(fctvdwalp))write(outu,*) "FCTVDWALP already allocated"

   call chmalloc('fctall.src','FCTINI','fctcgs'   , natom, crl=fctcgs   , ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','fctslfw'  , natom, crl=fctslfw  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','fctnpc'   , natom, crl=fctnpc   , ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','fctvdwfac', natom, crl=fctvdwfac, ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','fctvdwalp', natom, crl=fctvdwalp, ierr=ierr2, qdie=.true.)

   if(ierr2 /= 0) then
      call wrndie(-4,"FCTINI(fctall.src)",&
         "Cannot allocate FCTCGS,...")
   endif

   if(allocated(fctidx))write(outu,*) "FCTIDX already allocated"
   call chmalloc('fctall.src','fctini','fctidx', natom, intg=fctidx, ierr=ierr2, qdie=.true.)
   if(ierr2 /= 0) then
      call wrndie(-4,"FCTINI(fctall.src)",&
         "Cannot allocate FCTIDX")
   endif

   if(allocated(fctrvw))write(outu,*)      "FCTRVW      already allocated"
   if(allocated(fctcsc))write(outu,*)      "FCTCSC      already allocated"
   if(allocated(fctscf))write(outu,*)      "FCTSCF      already allocated"
   if(allocated(fctnps))write(outu,*)      "FCTNPS      already allocated"
   if(allocated(fctvdwalpha))write(outu,*) "FCTVDWALPHA already allocated"

   call chmalloc('fctall.src','FCTINI','fctrvw'     , fctmvw, crl=fctrvw     , ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','fctcsc'     , fctmvw, crl=fctcsc     , ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','fctscf'     , fctmvw, crl=fctscf     , ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','fctnps'     , fctmvw, crl=fctnps     , ierr=ierr2, qdie=.true.)
   ! VO : below make sure fctvdwalpha accommodates all atom types :
   call chmalloc('fctall.src','FCTINI','fctvdwalpha', max(fctmvw,maxval(iac)), crl=fctvdwalpha, ierr=ierr2, qdie=.true.)

   if(ierr2 /= 0) then
      call wrndie(-4,"FCTINI(fctall.src)",&
         "Cannot allocate FCTRVW,...")
   endif

   ! Nullify
   fctidx    = 0

   fctcgs    = zero
   fctslfw   = zero
   fctnpc    = zero
   fctvdwfac = zero
   fctvdwalp = zero

   fctrvw      = zero
   fctcsc      = zero
   fctscf      = zero
   fctnps      = zero
   fctvdwalpha = zero
! -------------------------------------------------------------------

! Set the flag to invoke FACTS.

      fctrun=.true.

!  Obtain the flag and UNIT number to write atomic Self Energies to a file

   selfunit=gtrmi(comlyn,comlen,'SLFU',-1)
   qslfunit=(selfunit > 0)

!  Obtain the flag and UNIT number to write atomic Screen Energies to a file

   scrnunit=gtrmi(comlyn,comlen,'SCRU',-1)
   qscrunit=(selfunit > 0)

!  Obtain the flag and UNIT number to write atomic Non-polar Energies to a file

   npolunit=gtrmi(comlyn,comlen,'NPOU',-1)
   qnplunit=(npolunit > 0)

! Obtain the flag for calculating only the screening part and not
! the self-solvation part of the electrostatic solvation.

   fctscro=(indxa(comlyn,comlen,'SCRO')  >  0)

! Obtain the flag for calculating only the self-solvation part and not
! the screening  part of the electrostatic solvation.

   fctslfo=(indxa(comlyn,comlen,'SLFO')  >  0)

! Obtain the flag for calculating only the screening (FCTSCR) and not
! the solvation (FCTSFE).

      fctscr=(indxa(comlyn,comlen,'TSCR')  >  0)

! Obtain the flag for printing atomic self terms.

      fctpsl=(indxa(comlyn,comlen,'TPSL')  >  0)

! Obtain the flag for printing screened interaction energies.

      fctpin=(indxa(comlyn,comlen,'TPIN')  >  0)

! Obtain the flag for printing atomic solvent accessible surface areas.

      fctpsr=(indxa(comlyn,comlen,'TPSR')  >  0)

! Obtain the flag for printing alpha.

      fctpal=(indxa(comlyn,comlen,'TPAL')  >  0)

! Obtain the flag for printing beta.

      fctpbt=(indxa(comlyn,comlen,'TPBT')  >  0)

! Obtain the flag for printing VdW

      fctpvw=(indxa(comlyn,comlen,'TPVW')  >  0)

! Obtain the flag for printing forces.

      fctpfr=(indxa(comlyn,comlen,'TPFR')  >  0)

! Obtain the flag for adding non native van der Waals radii.

      fctavw=(indxa(comlyn,comlen,'TAVW')  >  0)

! Obtain the flag for setting up the image lists.

   fctaim=.false.
   if(NTRANS > 0) then
      fctaim=.true.
   endif

! Obtain the CHARMM parameter set.

   fctcps=gtrmi(comlyn,comlen,'TCPS',22)

   if(fctcps == 19)then
      ! fctdflt = two
      fctdflt = one
   else
      fctdflt = one
   endif

! Obtain the dielectric constant of the macromolecular region, i.e., the
! internal dielectric constant.

   fcteps=gtrmf(comlyn,comlen,'TEPS',fctdflt)

! Obtain the FACTS parameter set number.
! FM: set the default to 3

   fctfps=gtrmi(comlyn,comlen,'TFPS',3)

! Obtain alpha.

   fctap0=gtrmf(comlyn,comlen,'GAMM',1.5d-02)
   fctap1=gtrmf(comlyn,comlen,'TAP1',zero)
   fctap2=gtrmf(comlyn,comlen,'TAP2',zero)
   fctap3=gtrmf(comlyn,comlen,'TAP3',zero)

! Obtain beta.

   fctbt0=gtrmf(comlyn,comlen,'TBT0',zero)
   fctbt1=gtrmf(comlyn,comlen,'TBT1',zero)
   fctbt2=gtrmf(comlyn,comlen,'TBT2',zero)
   fctbt3=gtrmf(comlyn,comlen,'TBT3',zero)

! Obtain the salt concentration, the medium temperature
! and the SALT scaling factor

   fctconc=gtrmf(comlyn ,comlen,'CONC', zero)
   fcttemp=gtrmf(comlyn ,comlen,'TEMP', 2.98d+02)
   ! saltfact=gtrmf(comlyn,comlen,'SLTF', one)
   ! Salt factor set to 0.3
   saltfact=gtrmf(comlyn,comlen,'SLTF', three/ten)

! Set the flag for calculating the screening part of the electrostatic solvation.

   doscreen=.true.
   if(fctslfo) then
      doscreen=.false.
   endif

! Set the flag for calculating the solvation (FCTSFE) and not the
! screening (FCTSCR).

   if (fctscr) then
      fctsfe=.false.
   else
      fctsfe=.true.
   endif

! Process the choice of the FACTS parameter set.

   if(fcteps /= eps)then
      if(wrnlev >= 2) write(outu,101) fcteps, eps
101      FORMAT( /,  &
          ' ***** WARNING from FCTINI ***** ',/,/,&
          ' FCTINI: Solute internal dielectric constant TEPS ', F6.2,/, &
          ' is different from non-bonded dielectric constant EPS ',F6.2)

      call wrndie(-1,"FCTINI(fctall.src)",&
         "FACTS dielectric constant TEPS is not equall to non-bonded EPS")
   endif

   ! Mutually exclusive options
   if(fctscro .and. fctslfo) then
      if(wrnlev >= 2) write(outu,102)
102   FORMAT( /,  &
         ' ***** WARNING from FCTINI ***** ',/,/,&
         ' FCTINI: SLFO and SCRO options are mutually exclusive')

      call wrndie(-1,"FCTINI(fctall.src)",&
         "Incompatible SLFO and SCRO options")
   endif

! Process hard parameters.

      fctcho=.false.

      if ((fctcps == 19    )  .and. &
          (fcteps == one   )  .and. &
          (fctfps == 1     )) then

! This FACTS parameter set was derived with no boundary conditions.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/one
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 7

! Define the number of actual van der Waals radii.

         fctnav    = 7

! Define the native van der Waals radii.

         fctrvw(1 )= 1.0000d0
         fctrvw(2 )= 1.6000d0
         fctrvw(3 )= 1.8900d0
         fctrvw(4 )= 2.1000d0
         fctrvw(5 )= 2.1650d0
         fctrvw(6 )= 2.2350d0
         fctrvw(7 )= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =11.0d0
         fctcil    = 9.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.750653e+01
         fctcsc(2 )= 0.888280e+01
         fctcsc(3 )= 0.100000e+02
         fctcsc(4 )= 0.996775e+01
         fctcsc(5 )= 0.100000e+02
         fctcsc(6 )= 0.980689e+01
         fctcsc(7 )= 0.100000e+02
         fctcic    = 7.5d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.146119e+03
         fctcb2(1 )=-0.123030e+01
         fctca0(1 )=-0.145713e+03
         fctca1(1 )= 0.128031e+03
         fctca2(1 )= 0.565022e-02
         fctca3(1 )= 0.190511e+03

         fctcb1(2 )=-0.333497e+03
         fctcb2(2 )=-0.951523e+00
         fctca0(2 )=-0.102064e+03
         fctca1(2 )= 0.101863e+03
         fctca2(2 )= 0.223184e-02
         fctca3(2 )= 0.575798e+03

         fctcb1(3 )= 0.134968e+03
         fctcb2(3 )=-0.895481e+00
         fctca0(3 )=-0.786321e+02
         fctca1(3 )= 0.720024e+02
         fctca2(3 )= 0.241746e-02
         fctca3(3 )= 0.109742e+04

         fctcb1(4 )=-0.126041e+03
         fctcb2(4 )=-0.766938e+00
         fctca0(4 )=-0.757741e+02
         fctca1(4 )= 0.757721e+02
         fctca2(4 )= 0.185735e-02
         fctca3(4 )= 0.113839e+04

         fctcb1(5 )=-0.873195e+02
         fctcb2(5 )=-0.530376e+00
         fctca0(5 )=-0.698119e+02
         fctca1(5 )= 0.698099e+02
         fctca2(5 )= 0.272477e-02
         fctca3(5 )= 0.126616e+04

         fctcb1(6 )= 0.199651e+01
         fctcb2(6 )=-0.794886e+00
         fctca0(6 )=-0.697933e+02
         fctca1(6 )= 0.697913e+02
         fctca2(6 )= 0.224315e-02
         fctca3(6 )= 0.115681e+04

         fctcb1(7 )= 0.161465e+03
         fctcb2(7 )=-0.698050e+00
         fctca0(7 )=-0.612286e+02
         fctca1(7 )= 0.612266e+02
         fctca2(7 )= 0.233284e-02
         fctca3(7 )= 0.139336e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.193261e+04
         fctcd2(1 )=-0.129440e+01
         fctcc0(1 )= 0.163304e+02
         fctcc1(1 )=-0.163284e+02
         fctcc2(1 )= 0.427427e-02
         fctcc3(1 )=-0.127882e+04

         fctcd1(2 )=-0.377745e+04
         fctcd2(2 )=-0.355457e+01
         fctcc0(2 )= 0.551873e+02
         fctcc1(2 )=-0.551853e+02
         fctcc2(2 )= 0.160205e-02
         fctcc3(2 )=-0.343534e+04

         fctcd1(3 )=-0.139512e+05
         fctcd2(3 )=-0.268920e+01
         fctcc0(3 )= 0.393976e+02
         fctcc1(3 )=-0.391533e+02
         fctcc2(3 )= 0.731149e-03
         fctcc3(3 )=-0.765844e+04

         fctcd1(4 )=-0.226779e+04
         fctcd2(4 )=-0.202926e+01
         fctcc0(4 )= 0.167488e+03
         fctcc1(4 )=-0.167463e+03
         fctcc2(4 )= 0.188775e-02
         fctcc3(4 )=-0.233942e+04

         fctcd1(5 )=-0.152804e+04
         fctcd2(5 )=-0.131676e+01
         fctcc0(5 )= 0.103612e+03
         fctcc1(5 )=-0.103610e+03
         fctcc2(5 )= 0.208470e-02
         fctcc3(5 )=-0.975892e+03

         fctcd1(6 )=-0.149749e+04
         fctcd2(6 )=-0.144059e+01
         fctcc0(6 )= 0.637879e+02
         fctcc1(6 )=-0.637859e+02
         fctcc2(6 )= 0.288320e-02
         fctcc3(6 )=-0.626834e+03

         fctcd1(7 )= 0.385298e+03
         fctcd2(7 )=-0.212571e+01
         fctcc0(7 )= 0.123362e+03
         fctcc1(7 )=-0.123360e+03
         fctcc2(7 )= 0.397702e-02
         fctcc3(7 )=-0.249603e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.146119e+03
         fctcf2(1 )=-0.123030e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )=-0.333497e+03
         fctcf2(2 )=-0.951523e+00
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.134968e+03
         fctcf2(3 )=-0.895481e+00
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )=-0.126041e+03
         fctcf2(4 )=-0.766938e+00
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )=-0.873195e+02
         fctcf2(5 )=-0.530376e+00
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.199651e+01
         fctcf2(6 )=-0.794886e+00
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.161465e+03
         fctcf2(7 )=-0.698050e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.146119e+03
         fctch2(1 )=-0.123030e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )=-0.333497e+03
         fctch2(2 )=-0.951523e+00
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.134968e+03
         fctch2(3 )=-0.895481e+00
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )=-0.126041e+03
         fctch2(4 )=-0.766938e+00
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )=-0.873195e+02
         fctch2(5 )=-0.530376e+00
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.199651e+01
         fctch2(6 )=-0.794886e+00
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.161465e+03
         fctch2(7 )=-0.698050e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one

! Define the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero

      endif

      if ((fctcps == 19    )  .and. &
          (fcteps == one   )  .and. &
          (fctfps == 2     )) then

! This FACTS parameter set was derived with boundary conditions at
! negative and positive infinity.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/one
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 7

! Define the number of actual van der Waals radii.

         fctnav    = 7

! Define the native van der Waals radii.

         fctrvw(1 )= 1.0000d0
         fctrvw(2 )= 1.6000d0
         fctrvw(3 )= 1.8900d0
         fctrvw(4 )= 2.1000d0
         fctrvw(5 )= 2.1650d0
         fctrvw(6 )= 2.2350d0
         fctrvw(7 )= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =11.0d0
         fctcil    = 9.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.758192e+01
         fctcsc(2 )= 0.888282e+01
         fctcsc(3 )= 0.100000e+02
         fctcsc(4 )= 0.998903e+01
         fctcsc(5 )= 0.100000e+02
         fctcsc(6 )= 0.981312e+01
         fctcsc(7 )= 0.100000e+02
         fctcic    = 7.5d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.241162e+03
         fctcb2(1 )=-0.116943e+01
         fctca0(1 )=-0.163921e+03
         fctca1(1 )= 0.163919e+03
         fctca2(1 )= 0.354894e-02
         fctca3(1 )= 0.167028e+03

         fctcb1(2 )=-0.306833e+03
         fctcb2(2 )=-0.965681e+00
         fctca0(2 )=-0.102450e+03
         fctca1(2 )= 0.102448e+03
         fctca2(2 )= 0.223295e-02
         fctca3(2 )= 0.576944e+03

         fctcb1(3 )= 0.277598e+03
         fctcb2(3 )=-0.100806e+01
         fctca0(3 )=-0.867305e+02
         fctca1(3 )= 0.867285e+02
         fctca2(3 )= 0.195957e-02
         fctca3(3 )= 0.107884e+04

         fctcb1(4 )=-0.438434e+02
         fctcb2(4 )=-0.829504e+00
         fctca0(4 )=-0.780575e+02
         fctca1(4 )= 0.780555e+02
         fctca2(4 )= 0.179271e-02
         fctca3(4 )= 0.110891e+04

         fctcb1(5 )= 0.391840e+03
         fctcb2(5 )=-0.860529e+00
         fctca0(5 )=-0.757139e+02
         fctca1(5 )= 0.757119e+02
         fctca2(5 )= 0.264801e-02
         fctca3(5 )= 0.120150e+04

         fctcb1(6 )= 0.230231e+03
         fctcb2(6 )=-0.964971e+00
         fctca0(6 )=-0.733426e+02
         fctca1(6 )= 0.733406e+02
         fctca2(6 )= 0.216898e-02
         fctca3(6 )= 0.111092e+04

         fctcb1(7 )= 0.738064e+03
         fctcb2(7 )=-0.116688e+01
         fctca0(7 )=-0.693111e+02
         fctca1(7 )= 0.693091e+02
         fctca2(7 )= 0.208109e-02
         fctca3(7 )= 0.125876e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.530974e+04
         fctcd2(1 )=-0.283595e+01
         fctcc0(1 )= 0.723823e+02
         fctcc1(1 )=-0.723803e+02
         fctcc2(1 )= 0.948699e-03
         fctcc3(1 )=-0.634299e+04

         fctcd1(2 )=-0.721736e+04
         fctcd2(2 )=-0.588016e+01
         fctcc0(2 )= 0.113097e+03
         fctcc1(2 )=-0.113095e+03
         fctcc2(2 )= 0.721988e-03
         fctcc3(2 )=-0.833669e+04

         fctcd1(3 )=-0.449150e+03
         fctcd2(3 )=-0.184751e+01
         fctcc0(3 )= 0.136020e+03
         fctcc1(3 )=-0.136018e+03
         fctcc2(3 )= 0.311294e-02
         fctcc3(3 )=-0.799607e+03

         fctcd1(4 )=-0.293013e+04
         fctcd2(4 )=-0.246201e+01
         fctcc0(4 )= 0.153938e+03
         fctcc1(4 )=-0.153936e+03
         fctcc2(4 )= 0.167988e-02
         fctcc3(4 )=-0.293529e+04

         fctcd1(5 )=-0.625795e+03
         fctcd2(5 )=-0.142591e+01
         fctcc0(5 )= 0.159709e+03
         fctcc1(5 )=-0.159707e+03
         fctcc2(5 )= 0.247873e-02
         fctcc3(5 )=-0.715735e+03

         fctcd1(6 )=-0.329967e+03
         fctcd2(6 )=-0.156074e+01
         fctcc0(6 )= 0.166042e+03
         fctcc1(6 )=-0.166040e+03
         fctcc2(6 )= 0.317832e-02
         fctcc3(6 )=-0.525907e+03

         fctcd1(7 )= 0.290940e+03
         fctcd2(7 )=-0.210886e+01
         fctcc0(7 )= 0.178131e+03
         fctcc1(7 )=-0.178129e+03
         fctcc2(7 )= 0.370899e-02
         fctcc3(7 )=-0.433283e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.241162e+03
         fctcf2(1 )=-0.116943e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )=-0.306833e+03
         fctcf2(2 )=-0.965681e+00
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.277598e+03
         fctcf2(3 )=-0.100806e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )=-0.438434e+02
         fctcf2(4 )=-0.829504e+00
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )= 0.391840e+03
         fctcf2(5 )=-0.860529e+00
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.230231e+03
         fctcf2(6 )=-0.964971e+00
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.738064e+03
         fctcf2(7 )=-0.116688e+01
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.241162e+03
         fctch2(1 )=-0.116943e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )=-0.306833e+03
         fctch2(2 )=-0.965681e+00
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.277598e+03
         fctch2(3 )=-0.100806e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )=-0.438434e+02
         fctch2(4 )=-0.829504e+00
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )= 0.391840e+03
         fctch2(5 )=-0.860529e+00
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.230231e+03
         fctch2(6 )=-0.964971e+00
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.738064e+03
         fctch2(7 )=-0.116688e+01
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = zero

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one

! Define the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero

      ENDIF

      if ((fctcps == 19    )  .and. &
          (fcteps == one   )  .and. &
          (fctfps == 3     )) then

! This FACTS parameter set was derived with boundary conditions at zero
! and positive infinity.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/one
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 7

! Define the number of actual van der Waals radii.

         fctnav    = 7

! Define the native van der Waals radii.

         fctrvw(1 )= 1.0000d0
         fctrvw(2 )= 1.6000d0
         fctrvw(3 )= 1.8900d0
         fctrvw(4 )= 2.1000d0
         fctrvw(5 )= 2.1650d0
         fctrvw(6 )= 2.2350d0
         fctrvw(7 )= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =11.0d0
         fctcil    = 9.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.753702e+01
         fctcsc(2 )= 0.879248e+01
         fctcsc(3 )= 0.100000e+02
         fctcsc(4 )= 0.996445e+01
         fctcsc(5 )= 0.100000e+02
         fctcsc(6 )= 0.977048e+01
         fctcsc(7 )= 0.100000e+02
         fctcic    = 7.5d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )= 0.166082e+03
         fctcb2(1 )=-0.137007e+01
         fctca0(1 )=-0.450100e+03
         fctca1(1 )= 0.450098e+03
         fctca2(1 )= 0.331383e-02
         fctca3(1 )=-0.168158e+03

         fctcb1(2 )= 0.278024e+03
         fctcb2(2 )=-0.134664e+01
         fctca0(2 )=-0.175090e+03
         fctca1(2 )= 0.175088e+03
         fctca2(2 )= 0.198357e-02
         fctca3(2 )= 0.173352e+03

         fctcb1(3 )= 0.603837e+03
         fctcb2(3 )=-0.120153e+01
         fctca0(3 )=-0.103788e+03
         fctca1(3 )= 0.103786e+03
         fctca2(3 )= 0.179918e-02
         fctca3(3 )= 0.903844e+03

         fctcb1(4 )= 0.496612e+03
         fctcb2(4 )=-0.120905e+01
         fctca0(4 )=-0.983751e+02
         fctca1(4 )= 0.983731e+02
         fctca2(4 )= 0.160127e-02
         fctca3(4 )= 0.840541e+03

         fctcb1(5 )= 0.542264e+03
         fctcb2(5 )=-0.941220e+00
         fctca0(5 )=-0.792235e+02
         fctca1(5 )= 0.792215e+02
         fctca2(5 )= 0.261332e-02
         fctca3(5 )= 0.117531e+04

         fctcb1(6 )= 0.554837e+03
         fctcb2(6 )=-0.120060e+01
         fctca0(6 )=-0.832693e+02
         fctca1(6 )= 0.832673e+02
         fctca2(6 )= 0.204106e-02
         fctca3(6 )= 0.979828e+03

         fctcb1(7 )= 0.100346e+04
         fctcb2(7 )=-0.138605e+01
         fctca0(7 )=-0.769089e+02
         fctca1(7 )= 0.769069e+02
         fctca2(7 )= 0.192988e-02
         fctca3(7 )= 0.114552e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.193261e+04
         fctcd2(1 )=-0.129440e+01
         fctcc0(1 )= 0.163304e+02
         fctcc1(1 )=-0.163284e+02
         fctcc2(1 )= 0.427427e-02
         fctcc3(1 )=-0.127882e+04

         fctcd1(2 )=-0.377745e+04
         fctcd2(2 )=-0.355457e+01
         fctcc0(2 )= 0.551873e+02
         fctcc1(2 )=-0.551853e+02
         fctcc2(2 )= 0.160205e-02
         fctcc3(2 )=-0.343534e+04

         fctcd1(3 )=-0.139512e+05
         fctcd2(3 )=-0.268920e+01
         fctcc0(3 )= 0.393976e+02
         fctcc1(3 )=-0.391533e+02
         fctcc2(3 )= 0.731149e-03
         fctcc3(3 )=-0.765844e+04

         fctcd1(4 )=-0.226779e+04
         fctcd2(4 )=-0.202926e+01
         fctcc0(4 )= 0.167488e+03
         fctcc1(4 )=-0.167463e+03
         fctcc2(4 )= 0.188775e-02
         fctcc3(4 )=-0.233942e+04

         fctcd1(5 )=-0.152804e+04
         fctcd2(5 )=-0.131676e+01
         fctcc0(5 )= 0.103612e+03
         fctcc1(5 )=-0.103610e+03
         fctcc2(5 )= 0.208470e-02
         fctcc3(5 )=-0.975892e+03

         fctcd1(6 )=-0.149749e+04
         fctcd2(6 )=-0.144059e+01
         fctcc0(6 )= 0.637879e+02
         fctcc1(6 )=-0.637859e+02
         fctcc2(6 )= 0.288320e-02
         fctcc3(6 )=-0.626834e+03

         fctcd1(7 )= 0.385298e+03
         fctcd2(7 )=-0.212571e+01
         fctcc0(7 )= 0.123362e+03
         fctcc1(7 )=-0.123360e+03
         fctcc2(7 )= 0.397702e-02
         fctcc3(7 )=-0.249603e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )= 0.166082e+03
         fctcf2(1 )=-0.137007e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )= 0.278024e+03
         fctcf2(2 )=-0.134664e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.603837e+03
         fctcf2(3 )=-0.120153e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )= 0.496612e+03
         fctcf2(4 )=-0.120905e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )= 0.542264e+03
         fctcf2(5 )=-0.941220e+00
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.554837e+03
         fctcf2(6 )=-0.120060e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.100346e+04
         fctcf2(7 )=-0.138605e+01
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )= 0.166082e+03
         fctch2(1 )=-0.137007e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )= 0.278024e+03
         fctch2(2 )=-0.134664e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.603837e+03
         fctch2(3 )=-0.120153e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )= 0.496612e+03
         fctch2(4 )=-0.120905e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )= 0.542264e+03
         fctch2(5 )=-0.941220e+00
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.554837e+03
         fctch2(6 )=-0.120060e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.100346e+04
         fctch2(7 )=-0.138605e+01
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one

! Define the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero

      endif

      if ((fctcps == 19    )  .and. &
          (fcteps == 1.35d0)  .and. &
          (fctfps == 1     )) then

! This FACTS parameter set was derived with no boundary conditions.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/1.35d0
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 7

! Define the number of actual van der Waals radii.

         fctnav    = 7

! Define the native van der Waals radii.

         fctrvw(1 )= 1.0000d0
         fctrvw(2 )= 1.6000d0
         fctrvw(3 )= 1.8900d0
         fctrvw(4 )= 2.1000d0
         fctrvw(5 )= 2.1650d0
         fctrvw(6 )= 2.2350d0
         fctrvw(7 )= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =11.0d0
         fctcil    = 9.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.712626e+01
         fctcsc(2 )= 0.860849e+01
         fctcsc(3 )= 0.893099e+01
         fctcsc(4 )= 0.940148e+01
         fctcsc(5 )= 0.999124e+01
         fctcsc(6 )= 0.943040e+01
         fctcsc(7 )= 0.100000e+02
         fctcic    = 7.5d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.190622e+03
         fctcb2(1 )=-0.117281e+01
         fctca0(1 )=-0.107228e+03
         fctca1(1 )= 0.923060e+02
         fctca2(1 )= 0.595917e-02
         fctca3(1 )= 0.134519e+03

         fctcb1(2 )=-0.791076e+02
         fctcb2(2 )=-0.118545e+01
         fctca0(2 )=-0.931852e+02
         fctca1(2 )= 0.931832e+02
         fctca2(2 )= 0.195182e-02
         fctca3(2 )= 0.342099e+03

         fctcb1(3 )=-0.311741e+02
         fctcb2(3 )=-0.790763e+00
         fctca0(3 )=-0.620319e+02
         fctca1(3 )= 0.620299e+02
         fctca2(3 )= 0.178036e-02
         fctca3(3 )= 0.897620e+03

         fctcb1(4 )=-0.311741e+02
         fctcb2(4 )=-0.790763e+00
         fctca0(4 )=-0.620319e+02
         fctca1(4 )= 0.620299e+02
         fctca2(4 )= 0.178036e-02
         fctca3(4 )= 0.897620e+03

         fctcb1(5 )=-0.174917e+03
         fctcb2(5 )=-0.565937e+00
         fctca0(5 )=-0.536194e+02
         fctca1(5 )= 0.536174e+02
         fctca2(5 )= 0.217793e-02
         fctca3(5 )= 0.122633e+04

         fctcb1(6 )= 0.723640e+02
         fctcb2(6 )=-0.872348e+00
         fctca0(6 )=-0.543674e+02
         fctca1(6 )= 0.543654e+02
         fctca2(6 )= 0.220468e-02
         fctca3(6 )= 0.100378e+04

         fctcb1(7 )= 0.229882e+03
         fctcb2(7 )=-0.769184e+00
         fctca0(7 )=-0.487544e+02
         fctca1(7 )= 0.487524e+02
         fctca2(7 )= 0.188294e-02
         fctca3(7 )= 0.134929e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.193261e+04
         fctcd2(1 )=-0.129440e+01
         fctcc0(1 )= 0.163304e+02
         fctcc1(1 )=-0.163284e+02
         fctcc2(1 )= 0.427427e-02
         fctcc3(1 )=-0.127882e+04

         fctcd1(2 )=-0.377745e+04
         fctcd2(2 )=-0.355457e+01
         fctcc0(2 )= 0.551873e+02
         fctcc1(2 )=-0.551853e+02
         fctcc2(2 )= 0.160205e-02
         fctcc3(2 )=-0.343534e+04

         fctcd1(3 )=-0.139512e+05
         fctcd2(3 )=-0.268920e+01
         fctcc0(3 )= 0.393976e+02
         fctcc1(3 )=-0.391533e+02
         fctcc2(3 )= 0.731149e-03
         fctcc3(3 )=-0.765844e+04

         fctcd1(4 )=-0.226779e+04
         fctcd2(4 )=-0.202926e+01
         fctcc0(4 )= 0.167488e+03
         fctcc1(4 )=-0.167463e+03
         fctcc2(4 )= 0.188775e-02
         fctcc3(4 )=-0.233942e+04

         fctcd1(5 )=-0.152804e+04
         fctcd2(5 )=-0.131676e+01
         fctcc0(5 )= 0.103612e+03
         fctcc1(5 )=-0.103610e+03
         fctcc2(5 )= 0.208470e-02
         fctcc3(5 )=-0.975892e+03

         fctcd1(6 )=-0.149749e+04
         fctcd2(6 )=-0.144059e+01
         fctcc0(6 )= 0.637879e+02
         fctcc1(6 )=-0.637859e+02
         fctcc2(6 )= 0.288320e-02
         fctcc3(6 )=-0.626834e+03

         fctcd1(7 )= 0.385298e+03
         fctcd2(7 )=-0.212571e+01
         fctcc0(7 )= 0.123362e+03
         fctcc1(7 )=-0.123360e+03
         fctcc2(7 )= 0.397702e-02
         fctcc3(7 )=-0.249603e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.190622e+03
         fctcf2(1 )=-0.117281e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )=-0.791076e+02
         fctcf2(2 )=-0.118545e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )=-0.311741e+02
         fctcf2(3 )=-0.790763e+00
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )=-0.311741e+02
         fctcf2(4 )=-0.790763e+00
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )=-0.174917e+03
         fctcf2(5 )=-0.565937e+00
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.723640e+02
         fctcf2(6 )=-0.872348e+00
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.229882e+03
         fctcf2(7 )=-0.769184e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.190622e+03
         fctch2(1 )=-0.117281e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )=-0.791076e+02
         fctch2(2 )=-0.118545e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )=-0.311741e+02
         fctch2(3 )=-0.790763e+00
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )=-0.311741e+02
         fctch2(4 )=-0.790763e+00
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )=-0.174917e+03
         fctch2(5 )=-0.565937e+00
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.723640e+02
         fctch2(6 )=-0.872348e+00
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.229882e+03
         fctch2(7 )=-0.769184e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one

! Define the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero

      endif

      if ((fctcps == 19    )  .and. &
          (fcteps == 1.35d0)  .and. &
          (fctfps == 2     )) then

! This FACTS parameter set was derived with boundary conditions at
! negative and positive infinity.

         fctcho    = .true.

         call wrndie(-4,'<FCTINI>','FACTS PARAMETER SET MISSING.')

      endif

      if ((fctcps == 19    )  .and. &
          (fcteps == 1.35d0)  .and. &
          (fctfps == 3     )) then

! This FACTS parameter set was derived with boundary conditions at zero
! and positive infinity.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/1.35d0
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 7

! Define the number of actual van der Waals radii.

         fctnav    = 7

! Define the native van der Waals radii.

         fctrvw(1 )= 1.0000d0
         fctrvw(2 )= 1.6000d0
         fctrvw(3 )= 1.8900d0
         fctrvw(4 )= 2.1000d0
         fctrvw(5 )= 2.1650d0
         fctrvw(6 )= 2.2350d0
         fctrvw(7 )= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =11.0d0
         fctcil    = 9.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.712391e+01
         fctcsc(2 )= 0.848390e+01
         fctcsc(3 )= 0.863555e+01
         fctcsc(4 )= 0.939229e+01
         fctcsc(5 )= 0.995147e+01
         fctcsc(6 )= 0.941218e+01
         fctcsc(7 )= 0.100000e+02
         fctcic    = 7.5d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )= 0.135120e+03
         fctcb2(1 )=-0.132289e+01
         fctca0(1 )=-0.609686e+03
         fctca1(1 )= 0.609684e+03
         fctca2(1 )= 0.329844e-02
         fctca3(1 )=-0.423607e+03

         fctcb1(2 )= 0.222798e+03
         fctcb2(2 )=-0.132650e+01
         fctca0(2 )=-0.204518e+03
         fctca1(2 )= 0.204516e+03
         fctca2(2 )= 0.169364e-02
         fctca3(2 )=-0.315813e+03

         fctcb1(3 )= 0.332295e+03
         fctcb2(3 )=-0.110545e+01
         fctca0(3 )=-0.807187e+02
         fctca1(3 )= 0.807167e+02
         fctca2(3 )= 0.154687e-02
         fctca3(3 )= 0.588528e+03

         fctcb1(4 )= 0.332295e+03
         fctcb2(4 )=-0.110545e+01
         fctca0(4 )=-0.807187e+02
         fctca1(4 )= 0.807167e+02
         fctca2(4 )= 0.154687e-02
         fctca3(4 )= 0.588528e+03

         fctcb1(5 )= 0.384086e+03
         fctcb2(5 )=-0.921494e+00
         fctca0(5 )=-0.614733e+02
         fctca1(5 )= 0.614713e+02
         fctca2(5 )= 0.209863e-02
         fctca3(5 )= 0.109215e+04

         fctcb1(6 )= 0.422800e+03
         fctcb2(6 )=-0.115653e+01
         fctca0(6 )=-0.643363e+02
         fctca1(6 )= 0.643343e+02
         fctca2(6 )= 0.198577e-02
         fctca3(6 )= 0.837368e+03

         fctcb1(7 )= 0.809985e+03
         fctcb2(7 )=-0.128354e+01
         fctca0(7 )=-0.607427e+02
         fctca1(7 )= 0.607407e+02
         fctca2(7 )= 0.157274e-02
         fctca3(7 )= 0.106104e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.193261e+04
         fctcd2(1 )=-0.129440e+01
         fctcc0(1 )= 0.163304e+02
         fctcc1(1 )=-0.163284e+02
         fctcc2(1 )= 0.427427e-02
         fctcc3(1 )=-0.127882e+04

         fctcd1(2 )=-0.377745e+04
         fctcd2(2 )=-0.355457e+01
         fctcc0(2 )= 0.551873e+02
         fctcc1(2 )=-0.551853e+02
         fctcc2(2 )= 0.160205e-02
         fctcc3(2 )=-0.343534e+04

         fctcd1(3 )=-0.139512e+05
         fctcd2(3 )=-0.268920e+01
         fctcc0(3 )= 0.393976e+02
         fctcc1(3 )=-0.391533e+02
         fctcc2(3 )= 0.731149e-03
         fctcc3(3 )=-0.765844e+04

         fctcd1(4 )=-0.226779e+04
         fctcd2(4 )=-0.202926e+01
         fctcc0(4 )= 0.167488e+03
         fctcc1(4 )=-0.167463e+03
         fctcc2(4 )= 0.188775e-02
         fctcc3(4 )=-0.233942e+04

         fctcd1(5 )=-0.152804e+04
         fctcd2(5 )=-0.131676e+01
         fctcc0(5 )= 0.103612e+03
         fctcc1(5 )=-0.103610e+03
         fctcc2(5 )= 0.208470e-02
         fctcc3(5 )=-0.975892e+03

         fctcd1(6 )=-0.149749e+04
         fctcd2(6 )=-0.144059e+01
         fctcc0(6 )= 0.637879e+02
         fctcc1(6 )=-0.637859e+02
         fctcc2(6 )= 0.288320e-02
         fctcc3(6 )=-0.626834e+03

         fctcd1(7 )= 0.385298e+03
         fctcd2(7 )=-0.212571e+01
         fctcc0(7 )= 0.123362e+03
         fctcc1(7 )=-0.123360e+03
         fctcc2(7 )= 0.397702e-02
         fctcc3(7 )=-0.249603e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )= 0.135120e+03
         fctcf2(1 )=-0.132289e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )= 0.222798e+03
         fctcf2(2 )=-0.132650e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.332295e+03
         fctcf2(3 )=-0.110545e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )= 0.332295e+03
         fctcf2(4 )=-0.110545e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )= 0.384086e+03
         fctcf2(5 )=-0.921494e+00
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.422800e+03
         fctcf2(6 )=-0.115653e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.809985e+03
         fctcf2(7 )=-0.128354e+01
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )= 0.135120e+03
         fctch2(1 )=-0.132289e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )= 0.222798e+03
         fctch2(2 )=-0.132650e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.332295e+03
         fctch2(3 )=-0.110545e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )= 0.332295e+03
         fctch2(4 )=-0.110545e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )= 0.384086e+03
         fctch2(5 )=-0.921494e+00
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.422800e+03
         fctch2(6 )=-0.115653e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.809985e+03
         fctch2(7 )=-0.128354e+01
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one

! Define the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero

      endif

      if ((fctcps == 19    )  .and. &
          (fcteps == two   )  .and. &
          (fctfps == 1     )) then

! This FACTS parameter set was derived with no boundary conditions.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/two
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 7

! Define the number of actual van der Waals radii.

         fctnav    = 7

! Define the native van der Waals radii.

         fctrvw(1 )= 1.0000d0
         fctrvw(2 )= 1.6000d0
         fctrvw(3 )= 1.8900d0
         fctrvw(4 )= 2.1000d0
         fctrvw(5 )= 2.1650d0
         fctrvw(6 )= 2.2350d0
         fctrvw(7 )= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =11.0d0
         fctcil    = 9.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.738180e+01
         fctcsc(2 )= 0.853395e+01
         fctcsc(3 )= 0.902031e+01
         fctcsc(4 )= 0.956589e+01
         fctcsc(5 )= 0.100000e+02
         fctcsc(6 )= 0.941392e+01
         fctcsc(7 )= 0.100000e+02
         fctcic    = 7.5d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.211263e+03
         fctcb2(1 )=-0.120766e+01
         fctca0(1 )=-0.711374e+02
         fctca1(1 )= 0.610582e+02
         fctca2(1 )= 0.562288e-02
         fctca3(1 )= 0.143854e+03

         fctcb1(2 )=-0.161267e+03
         fctcb2(2 )=-0.102811e+01
         fctca0(2 )=-0.537462e+02
         fctca1(2 )= 0.516011e+02
         fctca2(2 )= 0.241634e-02
         fctca3(2 )= 0.448738e+03

         fctcb1(3 )= 0.243420e+03
         fctcb2(3 )=-0.935265e+00
         fctca0(3 )=-0.389775e+02
         fctca1(3 )= 0.339534e+02
         fctca2(3 )= 0.365658e-02
         fctca3(3 )= 0.803406e+03

         fctcb1(4 )=-0.218802e+03
         fctcb2(4 )=-0.614302e+00
         fctca0(4 )=-0.389967e+02
         fctca1(4 )= 0.389947e+02
         fctca2(4 )= 0.182234e-02
         fctca3(4 )= 0.999663e+03

         fctcb1(5 )=-0.245572e+03
         fctcb2(5 )=-0.434034e+00
         fctca0(5 )=-0.349332e+02
         fctca1(5 )= 0.349312e+02
         fctca2(5 )= 0.239099e-02
         fctca3(5 )= 0.124506e+04

         fctcb1(6 )= 0.773192e+02
         fctcb2(6 )=-0.856250e+00
         fctca0(6 )=-0.365072e+02
         fctca1(6 )= 0.365052e+02
         fctca2(6 )= 0.221413e-02
         fctca3(6 )= 0.990781e+03

         fctcb1(7 )= 0.187208e+03
         fctcb2(7 )=-0.735135e+00
         fctca0(7 )=-0.321476e+02
         fctca1(7 )= 0.321456e+02
         fctca2(7 )= 0.196335e-02
         fctca3(7 )= 0.134031e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.193261e+04
         fctcd2(1 )=-0.129440e+01
         fctcc0(1 )= 0.163304e+02
         fctcc1(1 )=-0.163284e+02
         fctcc2(1 )= 0.427427e-02
         fctcc3(1 )=-0.127882e+04

         fctcd1(2 )=-0.377745e+04
         fctcd2(2 )=-0.355457e+01
         fctcc0(2 )= 0.551873e+02
         fctcc1(2 )=-0.551853e+02
         fctcc2(2 )= 0.160205e-02
         fctcc3(2 )=-0.343534e+04

         fctcd1(3 )=-0.110519e+05
         fctcd2(3 )=-0.276237e+01
         fctcc0(3 )= 0.399838e+02
         fctcc1(3 )=-0.398897e+02
         fctcc2(3 )= 0.918397e-03
         fctcc3(3 )=-0.610549e+04

         fctcd1(4 )=-0.732548e+03
         fctcd2(4 )=-0.210098e+01
         fctcc0(4 )= 0.160195e+03
         fctcc1(4 )=-0.160146e+03
         fctcc2(4 )= 0.355838e-02
         fctcc3(4 )=-0.100191e+04

         fctcd1(5 )=-0.152804e+04
         fctcd2(5 )=-0.131676e+01
         fctcc0(5 )= 0.103612e+03
         fctcc1(5 )=-0.103610e+03
         fctcc2(5 )= 0.208470e-02
         fctcc3(5 )=-0.975892e+03

         fctcd1(6 )=-0.174048e+04
         fctcd2(6 )=-0.124393e+01
         fctcc0(6 )= 0.590390e+02
         fctcc1(6 )=-0.590370e+02
         fctcc2(6 )= 0.302113e-02
         fctcc3(6 )=-0.643231e+03

         fctcd1(7 )= 0.386309e+03
         fctcd2(7 )=-0.211465e+01
         fctcc0(7 )= 0.123365e+03
         fctcc1(7 )=-0.123363e+03
         fctcc2(7 )= 0.395129e-02
         fctcc3(7 )=-0.249600e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.211263e+03
         fctcf2(1 )=-0.120766e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )=-0.161267e+03
         fctcf2(2 )=-0.102811e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.243420e+03
         fctcf2(3 )=-0.935265e+00
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )=-0.218802e+03
         fctcf2(4 )=-0.614302e+00
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )=-0.245572e+03
         fctcf2(5 )=-0.434034e+00
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.773192e+02
         fctcf2(6 )=-0.856250e+00
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.187208e+03
         fctcf2(7 )=-0.735135e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.211263e+03
         fctch2(1 )=-0.120766e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )=-0.161267e+03
         fctch2(2 )=-0.102811e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.243420e+03
         fctch2(3 )=-0.935265e+00
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )=-0.218802e+03
         fctch2(4 )=-0.614302e+00
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )=-0.245572e+03
         fctch2(5 )=-0.434034e+00
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.773192e+02
         fctch2(6 )=-0.856250e+00
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.187208e+03
         fctch2(7 )=-0.735135e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one

! Define the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero

      endif

      if ((fctcps == 19    )  .and. &
          (fcteps == two   )  .and. &
          (fctfps == 2     )) then

! This FACTS parameter set was derived with boundary conditions at
! negative and positive infinity.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/two
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 7

! Define the number of actual van der Waals radii.

         fctnav    = 7

! Define the native van der Waals radii.

         fctrvw(1 )= 1.0000d0
         fctrvw(2 )= 1.6000d0
         fctrvw(3 )= 1.8900d0
         fctrvw(4 )= 2.1000d0
         fctrvw(5 )= 2.1650d0
         fctrvw(6 )= 2.2350d0
         fctrvw(7 )= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =11.0d0
         fctcil    = 9.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.739987e+01
         fctcsc(2 )= 0.852134e+01
         fctcsc(3 )= 0.917499e+01
         fctcsc(4 )= 0.956171e+01
         fctcsc(5 )= 0.100000e+02
         fctcsc(6 )= 0.941730e+01
         fctcsc(7 )= 0.100000e+02
         fctcic    = 7.5d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.317834e+03
         fctcb2(1 )=-0.112417e+01
         fctca0(1 )=-0.809028e+02
         fctca1(1 )= 0.809008e+02
         fctca2(1 )= 0.339314e-02
         fctca3(1 )= 0.116192e+03

         fctcb1(2 )=-0.393021e+03
         fctcb2(2 )=-0.816398e+00
         fctca0(2 )=-0.505642e+02
         fctca1(2 )= 0.505622e+02
         fctca2(2 )= 0.226296e-02
         fctca3(2 )= 0.518247e+03

         fctcb1(3 )= 0.243318e+03
         fctcb2(3 )=-0.955083e+00
         fctca0(3 )=-0.428057e+02
         fctca1(3 )= 0.428037e+02
         fctca2(3 )= 0.254306e-02
         fctca3(3 )= 0.862465e+03

         fctcb1(4 )=-0.263481e+03
         fctcb2(4 )=-0.573028e+00
         fctca0(4 )=-0.385251e+02
         fctca1(4 )= 0.385231e+02
         fctca2(4 )= 0.183784e-02
         fctca3(4 )= 0.101234e+04

         fctcb1(5 )= 0.193948e+03
         fctcb2(5 )=-0.749211e+00
         fctca0(5 )=-0.373685e+02
         fctca1(5 )= 0.373665e+02
         fctca2(5 )= 0.235559e-02
         fctca3(5 )= 0.118302e+04

         fctcb1(6 )= 0.468378e+02
         fctcb2(6 )=-0.827901e+00
         fctca0(6 )=-0.361981e+02
         fctca1(6 )= 0.361961e+02
         fctca2(6 )= 0.222188e-02
         fctca3(6 )= 0.999628e+03

         fctcb1(7 )= 0.438806e+03
         fctcb2(7 )=-0.945418e+00
         fctca0(7 )=-0.342084e+02
         fctca1(7 )= 0.342064e+02
         fctca2(7 )= 0.185526e-02
         fctca3(7 )= 0.126483e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.146127e+04
         fctcd2(1 )=-0.152459e+01
         fctcc0(1 )= 0.723823e+02
         fctcc1(1 )=-0.723803e+02
         fctcc2(1 )= 0.278988e-02
         fctcc3(1 )=-0.183088e+04

         fctcd1(2 )=-0.595514e+04
         fctcd2(2 )=-0.520513e+01
         fctcc0(2 )= 0.113097e+03
         fctcc1(2 )=-0.113095e+03
         fctcc2(2 )= 0.883239e-03
         fctcc3(2 )=-0.681803e+04

         fctcd1(3 )=-0.287917e+03
         fctcd2(3 )=-0.185258e+01
         fctcc0(3 )= 0.136020e+03
         fctcc1(3 )=-0.136018e+03
         fctcc2(3 )= 0.422977e-02
         fctcc3(3 )=-0.565971e+03

         fctcd1(4 )=-0.201670e+04
         fctcd2(4 )=-0.229274e+01
         fctcc0(4 )= 0.153938e+03
         fctcc1(4 )=-0.153936e+03
         fctcc2(4 )= 0.221780e-02
         fctcc3(4 )=-0.207534e+04

         fctcd1(5 )=-0.625795e+03
         fctcd2(5 )=-0.142591e+01
         fctcc0(5 )= 0.159709e+03
         fctcc1(5 )=-0.159707e+03
         fctcc2(5 )= 0.247873e-02
         fctcc3(5 )=-0.715735e+03

         fctcd1(6 )=-0.153128e+03
         fctcd2(6 )=-0.158520e+01
         fctcc0(6 )= 0.166042e+03
         fctcc1(6 )=-0.166040e+03
         fctcc2(6 )= 0.394806e-02
         fctcc3(6 )=-0.355894e+03

         fctcd1(7 )= 0.290940e+03
         fctcd2(7 )=-0.210886e+01
         fctcc0(7 )= 0.178131e+03
         fctcc1(7 )=-0.178129e+03
         fctcc2(7 )= 0.370899e-02
         fctcc3(7 )=-0.433283e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.317834e+03
         fctcf2(1 )=-0.112417e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )=-0.393021e+03
         fctcf2(2 )=-0.816398e+00
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.243318e+03
         fctcf2(3 )=-0.955083e+00
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )=-0.263481e+03
         fctcf2(4 )=-0.573028e+00
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )= 0.193948e+03
         fctcf2(5 )=-0.749211e+00
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.468378e+02
         fctcf2(6 )=-0.827901e+00
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.438806e+03
         fctcf2(7 )=-0.945418e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.317834e+03
         fctch2(1 )=-0.112417e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )=-0.393021e+03
         fctch2(2 )=-0.816398e+00
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.243318e+03
         fctch2(3 )=-0.955083e+00
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )=-0.263481e+03
         fctch2(4 )=-0.573028e+00
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )= 0.193948e+03
         fctch2(5 )=-0.749211e+00
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.468378e+02
         fctch2(6 )=-0.827901e+00
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.438806e+03
         fctch2(7 )=-0.945418e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = zero

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one

! Define the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero

      endif

      if ((fctcps == 19    )  .and. &
          (fcteps == two   )  .and. &
          (fctfps == 3     )) then

! This FACTS parameter set was derived with boundary conditions at zero
! and positive infinity.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/two
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 7

! Define the number of actual van der Waals radii.

         fctnav    = 7

! Define the native van der Waals radii.

         fctrvw(1 )= 1.0000d0
         fctrvw(2 )= 1.6000d0
         fctrvw(3 )= 1.8900d0
         fctrvw(4 )= 2.1000d0
         fctrvw(5 )= 2.1650d0
         fctrvw(6 )= 2.2350d0
         fctrvw(7 )= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =11.0d0
         fctcil    = 9.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.739032e+01
         fctcsc(2 )= 0.846133e+01
         fctcsc(3 )= 0.917618e+01
         fctcsc(4 )= 0.958532e+01
         fctcsc(5 )= 0.100000e+02
         fctcsc(6 )= 0.939675e+01
         fctcsc(7 )= 0.100000e+02
         fctcic    = 7.5d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )= 0.147839e+03
         fctcb2(1 )=-0.134703e+01
         fctca0(1 )=-0.416362e+03
         fctca1(1 )= 0.416360e+03
         fctca2(1 )= 0.305516e-02
         fctca3(1 )=-0.465533e+03

         fctcb1(2 )= 0.223513e+03
         fctcb2(2 )=-0.129504e+01
         fctca0(2 )=-0.110968e+03
         fctca1(2 )= 0.110966e+03
         fctca2(2 )= 0.185690e-02
         fctca3(2 )=-0.957731e+02

         fctcb1(3 )= 0.469945e+03
         fctcb2(3 )=-0.114207e+01
         fctca0(3 )=-0.505772e+02
         fctca1(3 )= 0.505752e+02
         fctca2(3 )= 0.233135e-02
         fctca3(3 )= 0.731837e+03

         fctcb1(4 )= 0.347165e+03
         fctcb2(4 )=-0.108875e+01
         fctca0(4 )=-0.525372e+02
         fctca1(4 )= 0.525352e+02
         fctca2(4 )= 0.154118e-02
         fctca3(4 )= 0.656211e+03

         fctcb1(5 )= 0.431914e+03
         fctcb2(5 )=-0.894286e+00
         fctca0(5 )=-0.401683e+02
         fctca1(5 )= 0.401663e+02
         fctca2(5 )= 0.229638e-02
         fctca3(5 )= 0.112840e+04

         fctcb1(6 )= 0.420734e+03
         fctcb2(6 )=-0.113993e+01
         fctca0(6 )=-0.430259e+02
         fctca1(6 )= 0.430239e+02
         fctca2(6 )= 0.200764e-02
         fctca3(6 )= 0.830804e+03

         fctcb1(7 )= 0.810694e+03
         fctcb2(7 )=-0.126414e+01
         fctca0(7 )=-0.401094e+02
         fctca1(7 )= 0.401074e+02
         fctca2(7 )= 0.164588e-02
         fctca3(7 )= 0.106769e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.193261e+04
         fctcd2(1 )=-0.129440e+01
         fctcc0(1 )= 0.163304e+02
         fctcc1(1 )=-0.163284e+02
         fctcc2(1 )= 0.427427e-02
         fctcc3(1 )=-0.127882e+04

         fctcd1(2 )=-0.377745e+04
         fctcd2(2 )=-0.355457e+01
         fctcc0(2 )= 0.551873e+02
         fctcc1(2 )=-0.551853e+02
         fctcc2(2 )= 0.160205e-02
         fctcc3(2 )=-0.343534e+04

         fctcd1(3 )=-0.110519e+05
         fctcd2(3 )=-0.276237e+01
         fctcc0(3 )= 0.399838e+02
         fctcc1(3 )=-0.398897e+02
         fctcc2(3 )= 0.918397e-03
         fctcc3(3 )=-0.610549e+04

         fctcd1(4 )=-0.732548e+03
         fctcd2(4 )=-0.210098e+01
         fctcc0(4 )= 0.160195e+03
         fctcc1(4 )=-0.160146e+03
         fctcc2(4 )= 0.355838e-02
         fctcc3(4 )=-0.100191e+04

         fctcd1(5 )=-0.152804e+04
         fctcd2(5 )=-0.131676e+01
         fctcc0(5 )= 0.103612e+03
         fctcc1(5 )=-0.103610e+03
         fctcc2(5 )= 0.208470e-02
         fctcc3(5 )=-0.975892e+03

         fctcd1(6 )=-0.174048e+04
         fctcd2(6 )=-0.124393e+01
         fctcc0(6 )= 0.590390e+02
         fctcc1(6 )=-0.590370e+02
         fctcc2(6 )= 0.302113e-02
         fctcc3(6 )=-0.643231e+03

         fctcd1(7 )= 0.386309e+03
         fctcd2(7 )=-0.211465e+01
         fctcc0(7 )= 0.123365e+03
         fctcc1(7 )=-0.123363e+03
         fctcc2(7 )= 0.395129e-02
         fctcc3(7 )=-0.249600e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )= 0.147839e+03
         fctcf2(1 )=-0.134703e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )= 0.223513e+03
         fctcf2(2 )=-0.129504e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.469945e+03
         fctcf2(3 )=-0.114207e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )= 0.347165e+03
         fctcf2(4 )=-0.108875e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )= 0.431914e+03
         fctcf2(5 )=-0.894286e+00
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.420734e+03
         fctcf2(6 )=-0.113993e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.810694e+03
         fctcf2(7 )=-0.126414e+01
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )= 0.147839e+03
         fctch2(1 )=-0.134703e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )= 0.223513e+03
         fctch2(2 )=-0.129504e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.469945e+03
         fctch2(3 )=-0.114207e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )= 0.347165e+03
         fctch2(4 )=-0.108875e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )= 0.431914e+03
         fctch2(5 )=-0.894286e+00
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.420734e+03
         fctch2(6 )=-0.113993e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.810694e+03
         fctch2(7 )=-0.126414e+01
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one

! Define the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero

      endif

      if ((fctcps == 22    )  .and. &
          (fcteps == one   )  .and. &
          (fctfps == 1     )) then

! This FACTS parameter set was derived with no boundary conditions.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/one
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der Waals radii.

         fctnnv    = 19

! Define the number of actual van der Waals radii.

         fctnav    = 19

! Define the native van der Waals radii.

         fctrvw(1 )= 0.2245d0
         fctrvw(2 )= 0.4500d0
         fctrvw(3 )= 0.7000d0
         fctrvw(4 )= 0.9000d0
         fctrvw(5 )= 1.3200d0
         fctrvw(6 )= 1.3582d0
         fctrvw(7 )= 1.4680d0
         fctrvw(8 )= 1.7000d0
         fctrvw(9 )= 1.7700d0
         fctrvw(10)= 1.8000d0
         fctrvw(11)= 1.8500d0
         fctrvw(12)= 1.9750d0
         fctrvw(13)= 1.9924d0
         fctrvw(14)= 2.0000d0
         fctrvw(15)= 2.0600d0
         fctrvw(16)= 2.1750d0
         fctrvw(17)= 2.2350d0
         fctrvw(18)= 2.2750d0
         fctrvw(19)= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =12.0d0
         fctcil    =14.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.714752e+01
         fctcsc(2 )= 0.642627e+01
         fctcsc(3 )= 0.600000e+01
         fctcsc(4 )= 0.711831e+01
         fctcsc(5 )= 0.820532e+01
         fctcsc(6 )= 0.837924e+01
         fctcsc(7 )= 0.802782e+01
         fctcsc(8 )= 0.854745e+01
         fctcsc(9 )= 0.850252e+01
         fctcsc(10)= 0.837057e+01
         fctcsc(11)= 0.852379e+01
         fctcsc(12)= 0.884069e+01
         fctcsc(13)= 0.920590e+01
         fctcsc(14)= 0.881363e+01
         fctcsc(15)= 0.986535e+01
         fctcsc(16)= 0.891107e+01
         fctcsc(17)= 0.600000e+01
         fctcsc(18)= 0.952518e+01
         fctcsc(19)= 0.600000e+01
         fctcic    =12.0d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.210603e+03
         fctcb2(1 )=-0.199137e+01
         fctca0(1 )=-0.150967e+03
         fctca1(1 )= 0.127403e+03
         fctca2(1 )= 0.510164e-02
         fctca3(1 )=-0.478652e+02

         fctcb1(2 )= 0.152309e+03
         fctcb2(2 )=-0.133090e+01
         fctca0(2 )=-0.174143e+03
         fctca1(2 )= 0.156700e+03
         fctca2(2 )= 0.813065e-02
         fctca3(2 )= 0.244034e+03

         fctcb1(3 )= 0.152309e+03
         fctcb2(3 )=-0.133090e+01
         fctca0(3 )=-0.174143e+03
         fctca1(3 )= 0.156700e+03
         fctca2(3 )= 0.813065e-02
         fctca3(3 )= 0.244034e+03

         fctcb1(4 )= 0.152309e+03
         fctcb2(4 )=-0.133090e+01
         fctca0(4 )=-0.174143e+03
         fctca1(4 )= 0.156700e+03
         fctca2(4 )= 0.813065e-02
         fctca3(4 )= 0.244034e+03

         fctcb1(5 )= 0.495330e+02
         fctcb2(5 )=-0.138532e+01
         fctca0(5 )=-0.127149e+03
         fctca1(5 )= 0.121317e+03
         fctca2(5 )= 0.303488e-02
         fctca3(5 )= 0.337407e+03

         fctcb1(6 )=-0.223768e+03
         fctcb2(6 )=-0.138928e+01
         fctca0(6 )=-0.110988e+03
         fctca1(6 )= 0.103587e+03
         fctca2(6 )= 0.300192e-02
         fctca3(6 )= 0.405079e+03

         fctcb1(7 )= 0.383667e+02
         fctcb2(7 )=-0.501591e+00
         fctca0(7 )=-0.104285e+03
         fctca1(7 )= 0.949888e+02
         fctca2(7 )= 0.503257e-02
         fctca3(7 )= 0.642381e+03

         fctcb1(8 )=-0.220010e+03
         fctcb2(8 )=-0.966769e+00
         fctca0(8 )=-0.968233e+02
         fctca1(8 )= 0.961271e+02
         fctca2(8 )= 0.235122e-02
         fctca3(8 )= 0.666119e+03

         fctcb1(9 )=-0.223037e+03
         fctcb2(9 )=-0.836065e+00
         fctca0(9 )=-0.929564e+02
         fctca1(9 )= 0.929544e+02
         fctca2(9 )= 0.243993e-02
         fctca3(9 )= 0.693910e+03

         fctcb1(10)= 0.853399e+01
         fctcb2(10)=-0.109161e+01
         fctca0(10)=-0.123827e+03
         fctca1(10)= 0.123825e+03
         fctca2(10)= 0.185613e-02
         fctca3(10)= 0.347902e+03

         fctcb1(11)=-0.289292e+03
         fctcb2(11)=-0.697267e+00
         fctca0(11)=-0.932613e+02
         fctca1(11)= 0.932593e+02
         fctca2(11)= 0.202205e-02
         fctca3(11)= 0.715165e+03

         fctcb1(12)= 0.507041e+03
         fctcb2(12)=-0.108130e+01
         fctca0(12)=-0.791668e+02
         fctca1(12)= 0.688771e+02
         fctca2(12)= 0.354600e-02
         fctca3(12)= 0.895937e+03

         fctcb1(13)= 0.519116e+01
         fctcb2(13)=-0.105654e+01
         fctca0(13)=-0.845685e+02
         fctca1(13)= 0.845665e+02
         fctca2(13)= 0.187023e-02
         fctca3(13)= 0.915891e+03

         fctcb1(14)= 0.147227e+03
         fctcb2(14)=-0.811304e+00
         fctca0(14)=-0.858924e+02
         fctca1(14)= 0.858904e+02
         fctca2(14)= 0.196363e-02
         fctca3(14)= 0.900140e+03

         fctcb1(15)= 0.257422e+02
         fctcb2(15)=-0.894245e+00
         fctca0(15)=-0.763464e+02
         fctca1(15)= 0.763444e+02
         fctca2(15)= 0.192986e-02
         fctca3(15)= 0.130541e+04

         fctcb1(16)= 0.620536e+02
         fctcb2(16)=-0.943820e+00
         fctca0(16)=-0.754147e+02
         fctca1(16)= 0.754127e+02
         fctca2(16)= 0.224894e-02
         fctca3(16)= 0.979966e+03

         fctcb1(17)= 0.746028e+03
         fctcb2(17)=-0.130263e+01
         fctca0(17)=-0.735409e+02
         fctca1(17)= 0.731524e+02
         fctca2(17)= 0.182580e-02
         fctca3(17)= 0.121299e+04

         fctcb1(18)= 0.746028e+03
         fctcb2(18)=-0.130263e+01
         fctca0(18)=-0.735409e+02
         fctca1(18)= 0.731524e+02
         fctca2(18)= 0.182580e-02
         fctca3(18)= 0.121299e+04

         fctcb1(19)= 0.746028e+03
         fctcb2(19)=-0.130263e+01
         fctca0(19)=-0.735409e+02
         fctca1(19)= 0.731524e+02
         fctca2(19)= 0.182580e-02
         fctca3(19)= 0.121299e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.448975e+04
         fctcd2(1 )=-0.977422e+01
         fctcc0(1 )= 0.331627e+02
         fctcc1(1 )=-0.331607e+02
         fctcc2(1 )= 0.276020e-01
         fctcc3(1 )=-0.719487e+04

         fctcd1(2 )=-0.978643e+04
         fctcd2(2 )=-0.688345e+01
         fctcc0(2 )= 0.872021e+01
         fctcc1(2 )=-0.871821e+01
         fctcc2(2 )= 0.143223e-02
         fctcc3(2 )=-0.709153e+04

         fctcd1(3 )=-0.978643e+04
         fctcd2(3 )=-0.688345e+01
         fctcc0(3 )= 0.872021e+01
         fctcc1(3 )=-0.871821e+01
         fctcc2(3 )= 0.143223e-02
         fctcc3(3 )=-0.709153e+04

         fctcd1(4 )=-0.978643e+04
         fctcd2(4 )=-0.688345e+01
         fctcc0(4 )= 0.872021e+01
         fctcc1(4 )=-0.871821e+01
         fctcc2(4 )= 0.143223e-02
         fctcc3(4 )=-0.709153e+04

         fctcd1(5 )=-0.192740e+03
         fctcd2(5 )=-0.176685e+01
         fctcc0(5 )= 0.100230e+03
         fctcc1(5 )=-0.100228e+03
         fctcc2(5 )= 0.630842e-02
         fctcc3(5 )=-0.578701e+03

         fctcd1(6 )=-0.728029e+03
         fctcd2(6 )=-0.225798e+01
         fctcc0(6 )= 0.163762e+03
         fctcc1(6 )=-0.163760e+03
         fctcc2(6 )= 0.327572e-02
         fctcc3(6 )=-0.141485e+04

         fctcd1(7 )=-0.345948e+04
         fctcd2(7 )=-0.203757e+01
         fctcc0(7 )= 0.458058e+02
         fctcc1(7 )=-0.458038e+02
         fctcc2(7 )= 0.193741e-02
         fctcc3(7 )=-0.240496e+04

         fctcd1(8 )=-0.193196e+04
         fctcd2(8 )=-0.166026e+01
         fctcc0(8 )= 0.650084e+02
         fctcc1(8 )=-0.650064e+02
         fctcc2(8 )= 0.245596e-02
         fctcc3(8 )=-0.135370e+04

         fctcd1(9 )=-0.309785e+04
         fctcd2(9 )=-0.119474e+01
         fctcc0(9 )= 0.721270e+02
         fctcc1(9 )=-0.721250e+02
         fctcc2(9 )= 0.171363e-02
         fctcc3(9 )=-0.208454e+04

         fctcd1(10)=-0.227929e+05
         fctcd2(10)=-0.825630e+01
         fctcc0(10)= 0.315402e+02
         fctcc1(10)=-0.313502e+02
         fctcc2(10)= 0.567394e-03
         fctcc3(10)=-0.142830e+05

         fctcd1(11)=-0.193388e+04
         fctcd2(11)=-0.156243e+01
         fctcc0(11)= 0.746559e+02
         fctcc1(11)=-0.746539e+02
         fctcc2(11)= 0.268963e-02
         fctcc3(11)=-0.128236e+04

         fctcd1(12)= 0.278812e+03
         fctcd2(12)=-0.224176e+01
         fctcc0(12)= 0.179508e+03
         fctcc1(12)=-0.179460e+03
         fctcc2(12)= 0.501248e-02
         fctcc3(12)=-0.346746e+03

         fctcd1(13)=-0.609707e+04
         fctcd2(13)=-0.203276e+00
         fctcc0(13)= 0.160296e+03
         fctcc1(13)=-0.160221e+03
         fctcc2(13)= 0.882224e-03
         fctcc3(13)=-0.537504e+04

         fctcd1(14)=-0.811741e+04
         fctcd2(14)=-0.217625e+01
         fctcc0(14)= 0.168481e+03
         fctcc1(14)=-0.168287e+03
         fctcc2(14)= 0.113765e-02
         fctcc3(14)=-0.672543e+04

         fctcd1(15)=-0.105332e+05
         fctcd2(15)=-0.124008e+01
         fctcc0(15)= 0.166467e+03
         fctcc1(15)=-0.166465e+03
         fctcc2(15)= 0.471945e-03
         fctcc3(15)=-0.969033e+04

         fctcd1(16)=-0.211060e+05
         fctcd2(16)=-0.454142e+00
         fctcc0(16)= 0.130754e+03
         fctcc1(16)=-0.130750e+03
         fctcc2(16)= 0.326802e-03
         fctcc3(16)=-0.152890e+05

         fctcd1(17)=-0.639876e+03
         fctcd2(17)=-0.199623e+01
         fctcc0(17)= 0.178300e+03
         fctcc1(17)=-0.178298e+03
         fctcc2(17)= 0.302069e-02
         fctcc3(17)=-0.103577e+04

         fctcd1(18)=-0.639876e+03
         fctcd2(18)=-0.199623e+01
         fctcc0(18)= 0.178300e+03
         fctcc1(18)=-0.178298e+03
         fctcc2(18)= 0.302069e-02
         fctcc3(18)=-0.103577e+04

         fctcd1(19)=-0.639876e+03
         fctcd2(19)=-0.199623e+01
         fctcc0(19)= 0.178300e+03
         fctcc1(19)=-0.178298e+03
         fctcc2(19)= 0.302069e-02
         fctcc3(19)=-0.103577e+04

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.210603e+03
         fctcf2(1 )=-0.199137e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )= 0.152309e+03
         fctcf2(2 )=-0.133090e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.152309e+03
         fctcf2(3 )=-0.133090e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )= 0.152309e+03
         fctcf2(4 )=-0.133090e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )= 0.495330e+02
         fctcf2(5 )=-0.138532e+01
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )=-0.223768e+03
         fctcf2(6 )=-0.138928e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.383667e+02
         fctcf2(7 )=-0.501591e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctcf1(8 )=-0.220010e+03
         fctcf2(8 )=-0.966769e+00
         fctce0(8 )= fctap0
         fctce1(8 )= fctap1
         fctce2(8 )= fctap2
         fctce3(8 )= fctap3

         fctcf1(9 )=-0.223037e+03
         fctcf2(9 )=-0.836065e+00
         fctce0(9 )= fctap0
         fctce1(9 )= fctap1
         fctce2(9 )= fctap2
         fctce3(9 )= fctap3

         fctcf1(10)= 0.853399e+01
         fctcf2(10)=-0.109161e+01
         fctce0(10)= fctap0
         fctce1(10)= fctap1
         fctce2(10)= fctap2
         fctce3(10)= fctap3

         fctcf1(11)=-0.289292e+03
         fctcf2(11)=-0.697267e+00
         fctce0(11)= fctap0
         fctce1(11)= fctap1
         fctce2(11)= fctap2
         fctce3(11)= fctap3

         fctcf1(12)= 0.507041e+03
         fctcf2(12)=-0.108130e+01
         fctce0(12)= fctap0
         fctce1(12)= fctap1
         fctce2(12)= fctap2
         fctce3(12)= fctap3

         fctcf1(13)= 0.519116e+01
         fctcf2(13)=-0.105654e+01
         fctce0(13)= fctap0
         fctce1(13)= fctap1
         fctce2(13)= fctap2
         fctce3(13)= fctap3

         fctcf1(14)= 0.147227e+03
         fctcf2(14)=-0.811304e+00
         fctce0(14)= fctap0
         fctce1(14)= fctap1
         fctce2(14)= fctap2
         fctce3(14)= fctap3

         fctcf1(15)= 0.257422e+02
         fctcf2(15)=-0.894245e+00
         fctce0(15)= fctap0
         fctce1(15)= fctap1
         fctce2(15)= fctap2
         fctce3(15)= fctap3

         fctcf1(16)= 0.620536e+02
         fctcf2(16)=-0.943820e+00
         fctce0(16)= fctap0
         fctce1(16)= fctap1
         fctce2(16)= fctap2
         fctce3(16)= fctap3

         fctcf1(17)= 0.746028e+03
         fctcf2(17)=-0.130263e+01
         fctce0(17)= fctap0
         fctce1(17)= fctap1
         fctce2(17)= fctap2
         fctce3(17)= fctap3

         fctcf1(18)= 0.746028e+03
         fctcf2(18)=-0.130263e+01
         fctce0(18)= fctap0
         fctce1(18)= fctap1
         fctce2(18)= fctap2
         fctce3(18)= fctap3

         fctcf1(19)= 0.746028e+03
         fctcf2(19)=-0.130263e+01
         fctce0(19)= fctap0
         fctce1(19)= fctap1
         fctce2(19)= fctap2
         fctce3(19)= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.210603e+03
         fctch2(1 )=-0.199137e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )= 0.152309e+03
         fctch2(2 )=-0.133090e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.152309e+03
         fctch2(3 )=-0.133090e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )= 0.152309e+03
         fctch2(4 )=-0.133090e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )= 0.495330e+02
         fctch2(5 )=-0.138532e+01
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )=-0.223768e+03
         fctch2(6 )=-0.138928e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.383667e+02
         fctch2(7 )=-0.501591e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctch1(8 )=-0.220010e+03
         fctch2(8 )=-0.966769e+00
         fctcg0(8 )= fctbt0
         fctcg1(8 )= fctbt1
         fctcg2(8 )= fctbt2
         fctcg3(8 )= fctbt3

         fctch1(9 )=-0.223037e+03
         fctch2(9 )=-0.836065e+00
         fctcg0(9 )= fctbt0
         fctcg1(9 )= fctbt1
         fctcg2(9 )= fctbt2
         fctcg3(9 )= fctbt3

         fctch1(10)= 0.853399e+01
         fctch2(10)=-0.109161e+01
         fctcg0(10)= fctbt0
         fctcg1(10)= fctbt1
         fctcg2(10)= fctbt2
         fctcg3(10)= fctbt3

         fctch1(11)=-0.289292e+03
         fctch2(11)=-0.697267e+00
         fctcg0(11)= fctbt0
         fctcg1(11)= fctbt1
         fctcg2(11)= fctbt2
         fctcg3(11)= fctbt3

         fctch1(12)= 0.507041e+03
         fctch2(12)=-0.108130e+01
         fctcg0(12)= fctbt0
         fctcg1(12)= fctbt1
         fctcg2(12)= fctbt2
         fctcg3(12)= fctbt3

         fctch1(13)= 0.519116e+01
         fctch2(13)=-0.105654e+01
         fctcg0(13)= fctbt0
         fctcg1(13)= fctbt1
         fctcg2(13)= fctbt2
         fctcg3(13)= fctbt3

         fctch1(14)= 0.147227e+03
         fctch2(14)=-0.811304e+00
         fctcg0(14)= fctbt0
         fctcg1(14)= fctbt1
         fctcg2(14)= fctbt2
         fctcg3(14)= fctbt3

         fctch1(15)= 0.257422e+02
         fctch2(15)=-0.894245e+00
         fctcg0(15)= fctbt0
         fctcg1(15)= fctbt1
         fctcg2(15)= fctbt2
         fctcg3(15)= fctbt3

         fctch1(16)= 0.620536e+02
         fctch2(16)=-0.943820e+00
         fctcg0(16)= fctbt0
         fctcg1(16)= fctbt1
         fctcg2(16)= fctbt2
         fctcg3(16)= fctbt3

         fctch1(17)= 0.746028e+03
         fctch2(17)=-0.130263e+01
         fctcg0(17)= fctbt0
         fctcg1(17)= fctbt1
         fctcg2(17)= fctbt2
         fctcg3(17)= fctbt3

         fctch1(18)= 0.746028e+03
         fctch2(18)=-0.130263e+01
         fctcg0(18)= fctbt0
         fctcg1(18)= fctbt1
         fctcg2(18)= fctbt2
         fctcg3(18)= fctbt3

         fctch1(19)= 0.746028e+03
         fctch2(19)=-0.130263e+01
         fctcg0(19)= fctbt0
         fctcg1(19)= fctbt1
         fctcg2(19)= fctbt2
         fctcg3(19)= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one
         fctqun(8 )= one
         fctqun(9 )= one
         fctqun(10)= one
         fctqun(11)= one
         fctqun(12)= one
         fctqun(13)= one
         fctqun(14)= one
         fctqun(15)= one
         fctqun(16)= one
         fctqun(17)= one
         fctqun(18)= one
         fctqun(19)= one

! Define the scaling factors for the charges and for the fctnps factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one
         fctscf(8 )= one
         fctscf(9 )= one
         fctscf(10)= one
         fctscf(11)= one
         fctscf(12)= one
         fctscf(13)= one
         fctscf(14)= one
         fctscf(15)= one
         fctscf(16)= one
         fctscf(17)= one
         fctscf(18)= one
         fctscf(19)= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero
         fctnps(8 )= zero
         fctnps(9 )= zero
         fctnps(10)= zero
         fctnps(11)= zero
         fctnps(12)= zero
         fctnps(13)= zero
         fctnps(14)= zero
         fctnps(15)= zero
         fctnps(16)= zero
         fctnps(17)= zero
         fctnps(18)= zero
         fctnps(19)= zero

      endif

      if ((fctcps == 22    )  .and. &
          (fcteps == one   )  .and. &
          (fctfps == 2     )) then

! This FACTS parameter set was derived with boundary conditions at
! negative and positive infinity.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/one
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der waals radii.

         fctnnv    = 19

! Define the number of actual van der waals radii.

         fctnav    = 19

! Define the native van der waals radii.

         fctrvw(1 )= 0.2245d0
         fctrvw(2 )= 0.4500d0
         fctrvw(3 )= 0.7000d0
         fctrvw(4 )= 0.9000d0
         fctrvw(5 )= 1.3200d0
         fctrvw(6 )= 1.3582d0
         fctrvw(7 )= 1.4680d0
         fctrvw(8 )= 1.7000d0
         fctrvw(9 )= 1.7700d0
         fctrvw(10)= 1.8000d0
         fctrvw(11)= 1.8500d0
         fctrvw(12)= 1.9750d0
         fctrvw(13)= 1.9924d0
         fctrvw(14)= 2.0000d0
         fctrvw(15)= 2.0600d0
         fctrvw(16)= 2.1750d0
         fctrvw(17)= 2.2350d0
         fctrvw(18)= 2.2750d0
         fctrvw(19)= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =12.0d0
         fctcil    =14.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.714652e+01
         fctcsc(2 )= 0.747510e+01
         fctcsc(3 )= 0.600000e+01
         fctcsc(4 )= 0.713710e+01
         fctcsc(5 )= 0.821901e+01
         fctcsc(6 )= 0.840905e+01
         fctcsc(7 )= 0.801510e+01
         fctcsc(8 )= 0.855916e+01
         fctcsc(9 )= 0.850423e+01
         fctcsc(10)= 0.833995e+01
         fctcsc(11)= 0.851735e+01
         fctcsc(12)= 0.886189e+01
         fctcsc(13)= 0.920688e+01
         fctcsc(14)= 0.882151e+01
         fctcsc(15)= 0.987366e+01
         fctcsc(16)= 0.891042e+01
         fctcsc(17)= 0.600000e+01
         fctcsc(18)= 0.952250e+01
         fctcsc(19)= 0.600000e+01
         fctcic    =12.0d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.148587e+03
         fctcb2(1 )=-0.196968e+01
         fctca0(1 )=-0.158448e+03
         fctca1(1 )= 0.136063e+03
         fctca2(1 )= 0.498271e-02
         fctca3(1 )=-0.379617e+02

         fctcb1(2 )= 0.740137e+02
         fctcb2(2 )=-0.134808e+01
         fctca0(2 )=-0.182134e+03
         fctca1(2 )= 0.182132e+03
         fctca2(2 )= 0.561322e-02
         fctca3(2 )= 0.226916e+03

         fctcb1(3 )= 0.740137e+02
         fctcb2(3 )=-0.134808e+01
         fctca0(3 )=-0.182134e+03
         fctca1(3 )= 0.182132e+03
         fctca2(3 )= 0.561322e-02
         fctca3(3 )= 0.226916e+03

         fctcb1(4 )= 0.740137e+02
         fctcb2(4 )=-0.134808e+01
         fctca0(4 )=-0.182134e+03
         fctca1(4 )= 0.182132e+03
         fctca2(4 )= 0.561322e-02
         fctca3(4 )= 0.226916e+03

         fctcb1(5 )=-0.774918e+02
         fctcb2(5 )=-0.133410e+01
         fctca0(5 )=-0.124182e+03
         fctca1(5 )= 0.124180e+03
         fctca2(5 )= 0.267342e-02
         fctca3(5 )= 0.353111e+03

         fctcb1(6 )=-0.140441e+03
         fctcb2(6 )=-0.145382e+01
         fctca0(6 )=-0.120690e+03
         fctca1(6 )= 0.120688e+03
         fctca2(6 )= 0.249519e-02
         fctca3(6 )= 0.403313e+03

         fctcb1(7 )= 0.696019e+02
         fctcb2(7 )=-0.585486e+00
         fctca0(7 )=-0.111663e+03
         fctca1(7 )= 0.111661e+03
         fctca2(7 )= 0.408956e-02
         fctca3(7 )= 0.639992e+03

         fctcb1(8 )=-0.250171e+03
         fctcb2(8 )=-0.949265e+00
         fctca0(8 )=-0.964239e+02
         fctca1(8 )= 0.964219e+02
         fctca2(8 )= 0.230868e-02
         fctca3(8 )= 0.674423e+03

         fctcb1(9 )=-0.234194e+03
         fctcb2(9 )=-0.827453e+00
         fctca0(9 )=-0.926106e+02
         fctca1(9 )= 0.926086e+02
         fctca2(9 )= 0.243729e-02
         fctca3(9 )= 0.695557e+03

         fctcb1(10)=-0.500339e+03
         fctcb2(10)=-0.521275e+00
         fctca0(10)=-0.910671e+02
         fctca1(10)= 0.910651e+02
         fctca2(10)= 0.215451e-02
         fctca3(10)= 0.628435e+03

         fctcb1(11)=-0.477266e+03
         fctcb2(11)=-0.513087e+00
         fctca0(11)=-0.886058e+02
         fctca1(11)= 0.886038e+02
         fctca2(11)= 0.206548e-02
         fctca3(11)= 0.760419e+03

         fctcb1(12)= 0.118886e+03
         fctcb2(12)=-0.717360e+00
         fctca0(12)=-0.829978e+02
         fctca1(12)= 0.829958e+02
         fctca2(12)= 0.252700e-02
         fctca3(12)= 0.959470e+03

         fctcb1(13)=-0.951296e+02
         fctcb2(13)=-0.979302e+00
         fctca0(13)=-0.822730e+02
         fctca1(13)= 0.822710e+02
         fctca2(13)= 0.189426e-02
         fctca3(13)= 0.943618e+03

         fctcb1(14)= 0.321039e+02
         fctcb2(14)=-0.698167e+00
         fctca0(14)=-0.819603e+02
         fctca1(14)= 0.819583e+02
         fctca2(14)= 0.202218e-02
         fctca3(14)= 0.950325e+03

         fctcb1(15)= 0.268844e+03
         fctcb2(15)=-0.103372e+01
         fctca0(15)=-0.795732e+02
         fctca1(15)= 0.795712e+02
         fctca2(15)= 0.189113e-02
         fctca3(15)= 0.126722e+04

         fctcb1(16)= 0.631751e+02
         fctcb2(16)=-0.941390e+00
         fctca0(16)=-0.753658e+02
         fctca1(16)= 0.753638e+02
         fctca2(16)= 0.225164e-02
         fctca3(16)= 0.980461e+03

         fctcb1(17)= 0.673429e+03
         fctcb2(17)=-0.123948e+01
         fctca0(17)=-0.720531e+02
         fctca1(17)= 0.720511e+02
         fctca2(17)= 0.184714e-02
         fctca3(17)= 0.124320e+04

         fctcb1(18)= 0.673429e+03
         fctcb2(18)=-0.123948e+01
         fctca0(18)=-0.720531e+02
         fctca1(18)= 0.720511e+02
         fctca2(18)= 0.184714e-02
         fctca3(18)= 0.124320e+04

         fctcb1(19)= 0.673429e+03
         fctcb2(19)=-0.123948e+01
         fctca0(19)=-0.720531e+02
         fctca1(19)= 0.720511e+02
         fctca2(19)= 0.184714e-02
         fctca3(19)= 0.124320e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.448975e+04
         fctcd2(1 )=-0.977422e+01
         fctcc0(1 )= 0.331627e+02
         fctcc1(1 )=-0.331607e+02
         fctcc2(1 )= 0.276020e-01
         fctcc3(1 )=-0.719487e+04

         fctcd1(2 )=-0.188978e+04
         fctcd2(2 )=-0.476442e+01
         fctcc0(2 )= 0.664761e+02
         fctcc1(2 )=-0.664741e+02
         fctcc2(2 )= 0.208082e-02
         fctcc3(2 )=-0.314490e+04

         fctcd1(3 )=-0.188978e+04
         fctcd2(3 )=-0.476442e+01
         fctcc0(3 )= 0.664761e+02
         fctcc1(3 )=-0.664741e+02
         fctcc2(3 )= 0.208082e-02
         fctcc3(3 )=-0.314490e+04

         fctcd1(4 )=-0.188978e+04
         fctcd2(4 )=-0.476442e+01
         fctcc0(4 )= 0.664761e+02
         fctcc1(4 )=-0.664741e+02
         fctcc2(4 )= 0.208082e-02
         fctcc3(4 )=-0.314490e+04

         fctcd1(5 )=-0.227334e+05
         fctcd2(5 )=-0.123295e+02
         fctcc0(5 )= 0.929710e+02
         fctcc1(5 )=-0.929690e+02
         fctcc2(5 )= 0.257980e-03
         fctcc3(5 )=-0.275568e+05

         fctcd1(6 )=-0.338361e+05
         fctcd2(6 )=-0.273739e+02
         fctcc0(6 )= 0.956008e+02
         fctcc1(6 )=-0.955988e+02
         fctcc2(6 )= 0.156308e-03
         fctcc3(6 )=-0.419913e+05

         fctcd1(7 )=-0.345638e+03
         fctcd2(7 )=-0.157073e+01
         fctcc0(7 )= 0.103364e+03
         fctcc1(7 )=-0.103362e+03
         fctcc2(7 )= 0.524066e-02
         fctcc3(7 )=-0.488295e+03

         fctcd1(8 )=-0.428198e+03
         fctcd2(8 )=-0.161616e+01
         fctcc0(8 )= 0.120763e+03
         fctcc1(8 )=-0.120761e+03
         fctcc2(8 )= 0.384137e-02
         fctcc3(8 )=-0.598349e+03

         fctcd1(9 )=-0.100031e+04
         fctcd2(9 )=-0.144611e+01
         fctcc0(9 )= 0.126278e+03
         fctcc1(9 )=-0.126276e+03
         fctcc2(9 )= 0.276368e-02
         fctcc3(9 )=-0.105029e+04

         fctcd1(10)=-0.152781e+05
         fctcd2(10)=-0.745997e+01
         fctcc0(10)= 0.128680e+03
         fctcc1(10)=-0.128678e+03
         fctcc2(10)= 0.487732e-03
         fctcc3(10)=-0.143218e+05

         fctcd1(11)=-0.368958e+03
         fctcd2(11)=-0.160743e+01
         fctcc0(11)= 0.132732e+03
         fctcc1(11)=-0.132730e+03
         fctcc2(11)= 0.453277e-02
         fctcc3(11)=-0.479033e+03

         fctcd1(12)=-0.197722e+04
         fctcd2(12)=-0.337536e+01
         fctcc0(12)= 0.143139e+03
         fctcc1(12)=-0.143137e+03
         fctcc2(12)= 0.184412e-02
         fctcc3(12)=-0.246645e+04

         fctcd1(13)=-0.262962e+04
         fctcd2(13)=-0.923843e+00
         fctcc0(13)= 0.144619e+03
         fctcc1(13)=-0.144617e+03
         fctcc2(13)= 0.158436e-02
         fctcc3(13)=-0.251924e+04

         fctcd1(14)=-0.177357e+05
         fctcd2(14)=-0.200780e+01
         fctcc0(14)= 0.145267e+03
         fctcc1(14)=-0.145265e+03
         fctcc2(14)= 0.573638e-03
         fctcc3(14)=-0.138185e+05

         fctcd1(15)=-0.166663e+02
         fctcd2(15)=-0.174062e+01
         fctcc0(15)= 0.150440e+03
         fctcc1(15)=-0.150438e+03
         fctcc2(15)= 0.306503e-02
         fctcc3(15)=-0.567656e+03

         fctcd1(16)= 0.140386e+02
         fctcd2(16)=-0.199711e+01
         fctcc0(16)= 0.160606e+03
         fctcc1(16)=-0.160604e+03
         fctcc2(16)= 0.460762e-02
         fctcc3(16)=-0.407165e+03

         fctcd1(17)= 0.457199e+03
         fctcd2(17)=-0.244355e+01
         fctcc0(17)= 0.169717e+03
         fctcc1(17)=-0.169715e+03
         fctcc2(17)= 0.434189e-02
         fctcc3(17)=-0.444777e+03

         fctcd1(18)= 0.457199e+03
         fctcd2(18)=-0.244355e+01
         fctcc0(18)= 0.169717e+03
         fctcc1(18)=-0.169715e+03
         fctcc2(18)= 0.434189e-02
         fctcc3(18)=-0.444777e+03

         fctcd1(19)= 0.457199e+03
         fctcd2(19)=-0.244355e+01
         fctcc0(19)= 0.169717e+03
         fctcc1(19)=-0.169715e+03
         fctcc2(19)= 0.434189e-02
         fctcc3(19)=-0.444777e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.148587e+03
         fctcf2(1 )=-0.196968e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )= 0.740137e+02
         fctcf2(2 )=-0.134808e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.740137e+02
         fctcf2(3 )=-0.134808e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )= 0.740137e+02
         fctcf2(4 )=-0.134808e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )=-0.774918e+02
         fctcf2(5 )=-0.133410e+01
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )=-0.140441e+03
         fctcf2(6 )=-0.145382e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.696019e+02
         fctcf2(7 )=-0.585486e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctcf1(8 )=-0.250171e+03
         fctcf2(8 )=-0.949265e+00
         fctce0(8 )= fctap0
         fctce1(8 )= fctap1
         fctce2(8 )= fctap2
         fctce3(8 )= fctap3

         fctcf1(9 )=-0.234194e+03
         fctcf2(9 )=-0.827453e+00
         fctce0(9 )= fctap0
         fctce1(9 )= fctap1
         fctce2(9 )= fctap2
         fctce3(9 )= fctap3

         fctcf1(10)=-0.500339e+03
         fctcf2(10)=-0.521275e+00
         fctce0(10)= fctap0
         fctce1(10)= fctap1
         fctce2(10)= fctap2
         fctce3(10)= fctap3

         fctcf1(11)=-0.477266e+03
         fctcf2(11)=-0.513087e+00
         fctce0(11)= fctap0
         fctce1(11)= fctap1
         fctce2(11)= fctap2
         fctce3(11)= fctap3

         fctcf1(12)= 0.118886e+03
         fctcf2(12)=-0.717360e+00
         fctce0(12)= fctap0
         fctce1(12)= fctap1
         fctce2(12)= fctap2
         fctce3(12)= fctap3

         fctcf1(13)=-0.951296e+02
         fctcf2(13)=-0.979302e+00
         fctce0(13)= fctap0
         fctce1(13)= fctap1
         fctce2(13)= fctap2
         fctce3(13)= fctap3

         fctcf1(14)= 0.321039e+02
         fctcf2(14)=-0.698167e+00
         fctce0(14)= fctap0
         fctce1(14)= fctap1
         fctce2(14)= fctap2
         fctce3(14)= fctap3

         fctcf1(15)= 0.268844e+03
         fctcf2(15)=-0.103372e+01
         fctce0(15)= fctap0
         fctce1(15)= fctap1
         fctce2(15)= fctap2
         fctce3(15)= fctap3

         fctcf1(16)= 0.631751e+02
         fctcf2(16)=-0.941390e+00
         fctce0(16)= fctap0
         fctce1(16)= fctap1
         fctce2(16)= fctap2
         fctce3(16)= fctap3

         fctcf1(17)= 0.673429e+03
         fctcf2(17)=-0.123948e+01
         fctce0(17)= fctap0
         fctce1(17)= fctap1
         fctce2(17)= fctap2
         fctce3(17)= fctap3

         fctcf1(18)= 0.673429e+03
         fctcf2(18)=-0.123948e+01
         fctce0(18)= fctap0
         fctce1(18)= fctap1
         fctce2(18)= fctap2
         fctce3(18)= fctap3

         fctcf1(19)= 0.673429e+03
         fctcf2(19)=-0.123948e+01
         fctce0(19)= fctap0
         fctce1(19)= fctap1
         fctce2(19)= fctap2
         fctce3(19)= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.148587e+03
         fctch2(1 )=-0.196968e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )= 0.740137e+02
         fctch2(2 )=-0.134808e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.740137e+02
         fctch2(3 )=-0.134808e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )= 0.740137e+02
         fctch2(4 )=-0.134808e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )=-0.774918e+02
         fctch2(5 )=-0.133410e+01
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )=-0.140441e+03
         fctch2(6 )=-0.145382e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.696019e+02
         fctch2(7 )=-0.585486e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctch1(8 )=-0.250171e+03
         fctch2(8 )=-0.949265e+00
         fctcg0(8 )= fctbt0
         fctcg1(8 )= fctbt1
         fctcg2(8 )= fctbt2
         fctcg3(8 )= fctbt3

         fctch1(9 )=-0.234194e+03
         fctch2(9 )=-0.827453e+00
         fctcg0(9 )= fctbt0
         fctcg1(9 )= fctbt1
         fctcg2(9 )= fctbt2
         fctcg3(9 )= fctbt3

         fctch1(10)=-0.500339e+03
         fctch2(10)=-0.521275e+00
         fctcg0(10)= fctbt0
         fctcg1(10)= fctbt1
         fctcg2(10)= fctbt2
         fctcg3(10)= fctbt3

         fctch1(11)=-0.477266e+03
         fctch2(11)=-0.513087e+00
         fctcg0(11)= fctbt0
         fctcg1(11)= fctbt1
         fctcg2(11)= fctbt2
         fctcg3(11)= fctbt3

         fctch1(12)= 0.118886e+03
         fctch2(12)=-0.717360e+00
         fctcg0(12)= fctbt0
         fctcg1(12)= fctbt1
         fctcg2(12)= fctbt2
         fctcg3(12)= fctbt3

         fctch1(13)=-0.951296e+02
         fctch2(13)=-0.979302e+00
         fctcg0(13)= fctbt0
         fctcg1(13)= fctbt1
         fctcg2(13)= fctbt2
         fctcg3(13)= fctbt3

         fctch1(14)= 0.321039e+02
         fctch2(14)=-0.698167e+00
         fctcg0(14)= fctbt0
         fctcg1(14)= fctbt1
         fctcg2(14)= fctbt2
         fctcg3(14)= fctbt3

         fctch1(15)= 0.268844e+03
         fctch2(15)=-0.103372e+01
         fctcg0(15)= fctbt0
         fctcg1(15)= fctbt1
         fctcg2(15)= fctbt2
         fctcg3(15)= fctbt3

         fctch1(16)= 0.631751e+02
         fctch2(16)=-0.941390e+00
         fctcg0(16)= fctbt0
         fctcg1(16)= fctbt1
         fctcg2(16)= fctbt2
         fctcg3(16)= fctbt3

         fctch1(17)= 0.673429e+03
         fctch2(17)=-0.123948e+01
         fctcg0(17)= fctbt0
         fctcg1(17)= fctbt1
         fctcg2(17)= fctbt2
         fctcg3(17)= fctbt3

         fctch1(18)= 0.673429e+03
         fctch2(18)=-0.123948e+01
         fctcg0(18)= fctbt0
         fctcg1(18)= fctbt1
         fctcg2(18)= fctbt2
         fctcg3(18)= fctbt3

         fctch1(19)= 0.673429e+03
         fctch2(19)=-0.123948e+01
         fctcg0(19)= fctbt0
         fctcg1(19)= fctbt1
         fctcg2(19)= fctbt2
         fctcg3(19)= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = zero

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one
         fctqun(8 )= one
         fctqun(9 )= one
         fctqun(10)= one
         fctqun(11)= one
         fctqun(12)= one
         fctqun(13)= one
         fctqun(14)= one
         fctqun(15)= one
         fctqun(16)= one
         fctqun(17)= one
         fctqun(18)= one
         fctqun(19)= one

! Define the scaling factors for the charges and for the fctnps factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one
         fctscf(8 )= one
         fctscf(9 )= one
         fctscf(10)= one
         fctscf(11)= one
         fctscf(12)= one
         fctscf(13)= one
         fctscf(14)= one
         fctscf(15)= one
         fctscf(16)= one
         fctscf(17)= one
         fctscf(18)= one
         fctscf(19)= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero
         fctnps(8 )= zero
         fctnps(9 )= zero
         fctnps(10)= zero
         fctnps(11)= zero
         fctnps(12)= zero
         fctnps(13)= zero
         fctnps(14)= zero
         fctnps(15)= zero
         fctnps(16)= zero
         fctnps(17)= zero
         fctnps(18)= zero
         fctnps(19)= zero

      endif

      if ((fctcps == 22    )  .and. &
          (fcteps == one   )  .and. &
          (fctfps == 3     )) then

! This FACTS parameter set was derived with boundary conditions at zero
! and positive infinity.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/one
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der waals radii.

         fctnnv    = 19

! Define the number of actual van der waals radii.

         fctnav    = 19

! Define the native van der waals radii.

         fctrvw(1 )= 0.2245d0
         fctrvw(2 )= 0.4500d0
         fctrvw(3 )= 0.7000d0
         fctrvw(4 )= 0.9000d0
         fctrvw(5 )= 1.3200d0
         fctrvw(6 )= 1.3582d0
         fctrvw(7 )= 1.4680d0
         fctrvw(8 )= 1.7000d0
         fctrvw(9 )= 1.7700d0
         fctrvw(10)= 1.8000d0
         fctrvw(11)= 1.8500d0
         fctrvw(12)= 1.9750d0
         fctrvw(13)= 1.9924d0
         fctrvw(14)= 2.0000d0
         fctrvw(15)= 2.0600d0
         fctrvw(16)= 2.1750d0
         fctrvw(17)= 2.2350d0
         fctrvw(18)= 2.2750d0
         fctrvw(19)= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =12.0d0
         fctcil    =14.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.714752e+01
         fctcsc(2 )= 0.770910e+01
         fctcsc(3 )= 0.600000e+01
         fctcsc(4 )= 0.715806e+01
         fctcsc(5 )= 0.807040e+01
         fctcsc(6 )= 0.826072e+01
         fctcsc(7 )= 0.800356e+01
         fctcsc(8 )= 0.850524e+01
         fctcsc(9 )= 0.842862e+01
         fctcsc(10)= 0.834750e+01
         fctcsc(11)= 0.853229e+01
         fctcsc(12)= 0.889733e+01
         fctcsc(13)= 0.914446e+01
         fctcsc(14)= 0.881827e+01
         fctcsc(15)= 0.983259e+01
         fctcsc(16)= 0.889709e+01
         fctcsc(17)= 0.600000e+01
         fctcsc(18)= 0.954038e+01
         fctcsc(19)= 0.600000e+01
         fctcic    =12.0d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.210603e+03
         fctcb2(1 )=-0.199137e+01
         fctca0(1 )=-0.150967e+03
         fctca1(1 )= 0.127403e+03
         fctca2(1 )= 0.510164e-02
         fctca3(1 )=-0.478652e+02

         fctcb1(2 )= 0.196721e+03
         fctcb2(2 )=-0.136295e+01
         fctca0(2 )=-0.277468e+03
         fctca1(2 )= 0.277466e+03
         fctca2(2 )= 0.499224e-02
         fctca3(2 )= 0.129670e+03

         fctcb1(3 )= 0.196721e+03
         fctcb2(3 )=-0.136295e+01
         fctca0(3 )=-0.277468e+03
         fctca1(3 )= 0.277466e+03
         fctca2(3 )= 0.499224e-02
         fctca3(3 )= 0.129670e+03

         fctcb1(4 )= 0.196721e+03
         fctcb2(4 )=-0.136295e+01
         fctca0(4 )=-0.277468e+03
         fctca1(4 )= 0.277466e+03
         fctca2(4 )= 0.499224e-02
         fctca3(4 )= 0.129670e+03

         fctcb1(5 )= 0.327651e+03
         fctcb2(5 )=-0.143178e+01
         fctca0(5 )=-0.218418e+03
         fctca1(5 )= 0.218416e+03
         fctca2(5 )= 0.264085e-02
         fctca3(5 )= 0.104488e+03

         fctcb1(6 )= 0.322880e+03
         fctcb2(6 )=-0.144317e+01
         fctca0(6 )=-0.184482e+03
         fctca1(6 )= 0.184480e+03
         fctca2(6 )= 0.258321e-02
         fctca3(6 )= 0.246814e+03

         fctcb1(7 )= 0.208357e+03
         fctcb2(7 )=-0.761629e+00
         fctca0(7 )=-0.122088e+03
         fctca1(7 )= 0.122086e+03
         fctca2(7 )= 0.397343e-02
         fctca3(7 )= 0.596760e+03

         fctcb1(8 )= 0.272497e+03
         fctcb2(8 )=-0.121046e+01
         fctca0(8 )=-0.128202e+03
         fctca1(8 )= 0.128200e+03
         fctca2(8 )= 0.222000e-02
         fctca3(8 )= 0.499978e+03

         fctcb1(9 )= 0.219247e+03
         fctcb2(9 )=-0.102267e+01
         fctca0(9 )=-0.114914e+03
         fctca1(9 )= 0.114912e+03
         fctca2(9 )= 0.243609e-02
         fctca3(9 )= 0.584393e+03

         fctcb1(10)= 0.226825e+03
         fctcb2(10)=-0.129999e+01
         fctca0(10)=-0.199105e+03
         fctca1(10)= 0.199103e+03
         fctca2(10)= 0.162982e-02
         fctca3(10)=-0.104865e+03

         fctcb1(11)= 0.183621e+03
         fctcb2(11)=-0.108865e+01
         fctca0(11)=-0.139478e+03
         fctca1(11)= 0.139476e+03
         fctca2(11)= 0.169541e-02
         fctca3(11)= 0.327275e+03

         fctcb1(12)= 0.410314e+03
         fctcb2(12)=-0.973425e+00
         fctca0(12)=-0.935102e+02
         fctca1(12)= 0.935082e+02
         fctca2(12)= 0.236404e-02
         fctca3(12)= 0.874027e+03

         fctcb1(13)= 0.510548e+03
         fctcb2(13)=-0.141151e+01
         fctca0(13)=-0.110269e+03
         fctca1(13)= 0.110267e+03
         fctca2(13)= 0.170081e-02
         fctca3(13)= 0.633798e+03

         fctcb1(14)= 0.469654e+03
         fctcb2(14)=-0.111225e+01
         fctca0(14)=-0.108712e+03
         fctca1(14)= 0.108710e+03
         fctca2(14)= 0.171827e-02
         fctca3(14)= 0.651603e+03

         fctcb1(15)= 0.667407e+03
         fctcb2(15)=-0.123469e+01
         fctca0(15)=-0.892306e+02
         fctca1(15)= 0.892286e+02
         fctca2(15)= 0.183562e-02
         fctca3(15)= 0.114889e+04

         fctcb1(16)= 0.500111e+03
         fctcb2(16)=-0.132339e+01
         fctca0(16)=-0.898055e+02
         fctca1(16)= 0.898035e+02
         fctca2(16)= 0.203303e-02
         fctca3(16)= 0.812753e+03

         fctcb1(17)= 0.106765e+04
         fctcb2(17)=-0.156808e+01
         fctca0(17)=-0.854426e+02
         fctca1(17)= 0.854406e+02
         fctca2(17)= 0.161973e-02
         fctca3(17)= 0.103900e+04

         fctcb1(18)= 0.106765e+04
         fctcb2(18)=-0.156808e+01
         fctca0(18)=-0.854426e+02
         fctca1(18)= 0.854406e+02
         fctca2(18)= 0.161973e-02
         fctca3(18)= 0.103900e+04

         fctcb1(19)= 0.106765e+04
         fctcb2(19)=-0.156808e+01
         fctca0(19)=-0.854426e+02
         fctca1(19)= 0.854406e+02
         fctca2(19)= 0.161973e-02
         fctca3(19)= 0.103900e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.448975e+04
         fctcd2(1 )=-0.977422e+01
         fctcc0(1 )= 0.331627e+02
         fctcc1(1 )=-0.331607e+02
         fctcc2(1 )= 0.276020e-01
         fctcc3(1 )=-0.719487e+04

         fctcd1(2 )=-0.978643e+04
         fctcd2(2 )=-0.688345e+01
         fctcc0(2 )= 0.872021e+01
         fctcc1(2 )=-0.871821e+01
         fctcc2(2 )= 0.143223e-02
         fctcc3(2 )=-0.709153e+04

         fctcd1(3 )=-0.978643e+04
         fctcd2(3 )=-0.688345e+01
         fctcc0(3 )= 0.872021e+01
         fctcc1(3 )=-0.871821e+01
         fctcc2(3 )= 0.143223e-02
         fctcc3(3 )=-0.709153e+04

         fctcd1(4 )=-0.978643e+04
         fctcd2(4 )=-0.688345e+01
         fctcc0(4 )= 0.872021e+01
         fctcc1(4 )=-0.871821e+01
         fctcc2(4 )= 0.143223e-02
         fctcc3(4 )=-0.709153e+04

         fctcd1(5 )=-0.192740e+03
         fctcd2(5 )=-0.176685e+01
         fctcc0(5 )= 0.100230e+03
         fctcc1(5 )=-0.100228e+03
         fctcc2(5 )= 0.630842e-02
         fctcc3(5 )=-0.578701e+03

         fctcd1(6 )=-0.728029e+03
         fctcd2(6 )=-0.225798e+01
         fctcc0(6 )= 0.163762e+03
         fctcc1(6 )=-0.163760e+03
         fctcc2(6 )= 0.327572e-02
         fctcc3(6 )=-0.141485e+04

         fctcd1(7 )=-0.345948e+04
         fctcd2(7 )=-0.203757e+01
         fctcc0(7 )= 0.458058e+02
         fctcc1(7 )=-0.458038e+02
         fctcc2(7 )= 0.193741e-02
         fctcc3(7 )=-0.240496e+04

         fctcd1(8 )=-0.193196e+04
         fctcd2(8 )=-0.166026e+01
         fctcc0(8 )= 0.650084e+02
         fctcc1(8 )=-0.650064e+02
         fctcc2(8 )= 0.245596e-02
         fctcc3(8 )=-0.135370e+04

         fctcd1(9 )=-0.309785e+04
         fctcd2(9 )=-0.119474e+01
         fctcc0(9 )= 0.721270e+02
         fctcc1(9 )=-0.721250e+02
         fctcc2(9 )= 0.171363e-02
         fctcc3(9 )=-0.208454e+04

         fctcd1(10)=-0.227929e+05
         fctcd2(10)=-0.825630e+01
         fctcc0(10)= 0.315402e+02
         fctcc1(10)=-0.313502e+02
         fctcc2(10)= 0.567394e-03
         fctcc3(10)=-0.142830e+05

         fctcd1(11)=-0.193388e+04
         fctcd2(11)=-0.156243e+01
         fctcc0(11)= 0.746559e+02
         fctcc1(11)=-0.746539e+02
         fctcc2(11)= 0.268963e-02
         fctcc3(11)=-0.128236e+04

         fctcd1(12)= 0.278812e+03
         fctcd2(12)=-0.224176e+01
         fctcc0(12)= 0.179508e+03
         fctcc1(12)=-0.179460e+03
         fctcc2(12)= 0.501248e-02
         fctcc3(12)=-0.346746e+03

         fctcd1(13)=-0.609707e+04
         fctcd2(13)=-0.203276e+00
         fctcc0(13)= 0.160296e+03
         fctcc1(13)=-0.160221e+03
         fctcc2(13)= 0.882224e-03
         fctcc3(13)=-0.537504e+04

         fctcd1(14)=-0.811741e+04
         fctcd2(14)=-0.217625e+01
         fctcc0(14)= 0.168481e+03
         fctcc1(14)=-0.168287e+03
         fctcc2(14)= 0.113765e-02
         fctcc3(14)=-0.672543e+04

         fctcd1(15)=-0.105332e+05
         fctcd2(15)=-0.124008e+01
         fctcc0(15)= 0.166467e+03
         fctcc1(15)=-0.166465e+03
         fctcc2(15)= 0.471945e-03
         fctcc3(15)=-0.969033e+04

         fctcd1(16)=-0.211060e+05
         fctcd2(16)=-0.454142e+00
         fctcc0(16)= 0.130754e+03
         fctcc1(16)=-0.130750e+03
         fctcc2(16)= 0.326802e-03
         fctcc3(16)=-0.152890e+05

         fctcd1(17)=-0.639876e+03
         fctcd2(17)=-0.199623e+01
         fctcc0(17)= 0.178300e+03
         fctcc1(17)=-0.178298e+03
         fctcc2(17)= 0.302069e-02
         fctcc3(17)=-0.103577e+04

         fctcd1(18)=-0.639876e+03
         fctcd2(18)=-0.199623e+01
         fctcc0(18)= 0.178300e+03
         fctcc1(18)=-0.178298e+03
         fctcc2(18)= 0.302069e-02
         fctcc3(18)=-0.103577e+04

         fctcd1(19)=-0.639876e+03
         fctcd2(19)=-0.199623e+01
         fctcc0(19)= 0.178300e+03
         fctcc1(19)=-0.178298e+03
         fctcc2(19)= 0.302069e-02
         fctcc3(19)=-0.103577e+04

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.210603e+03
         fctcf2(1 )=-0.199137e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )= 0.196721e+03
         fctcf2(2 )=-0.136295e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.196721e+03
         fctcf2(3 )=-0.136295e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )= 0.196721e+03
         fctcf2(4 )=-0.136295e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )= 0.327651e+03
         fctcf2(5 )=-0.143178e+01
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )= 0.322880e+03
         fctcf2(6 )=-0.144317e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.208357e+03
         fctcf2(7 )=-0.761629e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctcf1(8 )= 0.272497e+03
         fctcf2(8 )=-0.121046e+01
         fctce0(8 )= fctap0
         fctce1(8 )= fctap1
         fctce2(8 )= fctap2
         fctce3(8 )= fctap3

         fctcf1(9 )= 0.219247e+03
         fctcf2(9 )=-0.102267e+01
         fctce0(9 )= fctap0
         fctce1(9 )= fctap1
         fctce2(9 )= fctap2
         fctce3(9 )= fctap3

         fctcf1(10)= 0.226825e+03
         fctcf2(10)=-0.129999e+01
         fctce0(10)= fctap0
         fctce1(10)= fctap1
         fctce2(10)= fctap2
         fctce3(10)= fctap3

         fctcf1(11)= 0.183621e+03
         fctcf2(11)=-0.108865e+01
         fctce0(11)= fctap0
         fctce1(11)= fctap1
         fctce2(11)= fctap2
         fctce3(11)= fctap3

         fctcf1(12)= 0.410314e+03
         fctcf2(12)=-0.973425e+00
         fctce0(12)= fctap0
         fctce1(12)= fctap1
         fctce2(12)= fctap2
         fctce3(12)= fctap3

         fctcf1(13)= 0.510548e+03
         fctcf2(13)=-0.141151e+01
         fctce0(13)= fctap0
         fctce1(13)= fctap1
         fctce2(13)= fctap2
         fctce3(13)= fctap3

         fctcf1(14)= 0.469654e+03
         fctcf2(14)=-0.111225e+01
         fctce0(14)= fctap0
         fctce1(14)= fctap1
         fctce2(14)= fctap2
         fctce3(14)= fctap3

         fctcf1(15)= 0.667407e+03
         fctcf2(15)=-0.123469e+01
         fctce0(15)= fctap0
         fctce1(15)= fctap1
         fctce2(15)= fctap2
         fctce3(15)= fctap3

         fctcf1(16)= 0.500111e+03
         fctcf2(16)=-0.132339e+01
         fctce0(16)= fctap0
         fctce1(16)= fctap1
         fctce2(16)= fctap2
         fctce3(16)= fctap3

         fctcf1(17)= 0.106765e+04
         fctcf2(17)=-0.156808e+01
         fctce0(17)= fctap0
         fctce1(17)= fctap1
         fctce2(17)= fctap2
         fctce3(17)= fctap3

         fctcf1(18)= 0.106765e+04
         fctcf2(18)=-0.156808e+01
         fctce0(18)= fctap0
         fctce1(18)= fctap1
         fctce2(18)= fctap2
         fctce3(18)= fctap3

         fctcf1(19)= 0.106765e+04
         fctcf2(19)=-0.156808e+01
         fctce0(19)= fctap0
         fctce1(19)= fctap1
         fctce2(19)= fctap2
         fctce3(19)= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.210603e+03
         fctch2(1 )=-0.199137e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )= 0.196721e+03
         fctch2(2 )=-0.136295e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.196721e+03
         fctch2(3 )=-0.136295e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )= 0.196721e+03
         fctch2(4 )=-0.136295e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )= 0.327651e+03
         fctch2(5 )=-0.143178e+01
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )= 0.322880e+03
         fctch2(6 )=-0.144317e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.208357e+03
         fctch2(7 )=-0.761629e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctch1(8 )= 0.272497e+03
         fctch2(8 )=-0.121046e+01
         fctcg0(8 )= fctbt0
         fctcg1(8 )= fctbt1
         fctcg2(8 )= fctbt2
         fctcg3(8 )= fctbt3

         fctch1(9 )= 0.219247e+03
         fctch2(9 )=-0.102267e+01
         fctcg0(9 )= fctbt0
         fctcg1(9 )= fctbt1
         fctcg2(9 )= fctbt2
         fctcg3(9 )= fctbt3

         fctch1(10)= 0.226825e+03
         fctch2(10)=-0.129999e+01
         fctcg0(10)= fctbt0
         fctcg1(10)= fctbt1
         fctcg2(10)= fctbt2
         fctcg3(10)= fctbt3

         fctch1(11)= 0.183621e+03
         fctch2(11)=-0.108865e+01
         fctcg0(11)= fctbt0
         fctcg1(11)= fctbt1
         fctcg2(11)= fctbt2
         fctcg3(11)= fctbt3

         fctch1(12)= 0.410314e+03
         fctch2(12)=-0.973425e+00
         fctcg0(12)= fctbt0
         fctcg1(12)= fctbt1
         fctcg2(12)= fctbt2
         fctcg3(12)= fctbt3

         fctch1(13)= 0.510548e+03
         fctch2(13)=-0.141151e+01
         fctcg0(13)= fctbt0
         fctcg1(13)= fctbt1
         fctcg2(13)= fctbt2
         fctcg3(13)= fctbt3

         fctch1(14)= 0.469654e+03
         fctch2(14)=-0.111225e+01
         fctcg0(14)= fctbt0
         fctcg1(14)= fctbt1
         fctcg2(14)= fctbt2
         fctcg3(14)= fctbt3

         fctch1(15)= 0.667407e+03
         fctch2(15)=-0.123469e+01
         fctcg0(15)= fctbt0
         fctcg1(15)= fctbt1
         fctcg2(15)= fctbt2
         fctcg3(15)= fctbt3

         fctch1(16)= 0.500111e+03
         fctch2(16)=-0.132339e+01
         fctcg0(16)= fctbt0
         fctcg1(16)= fctbt1
         fctcg2(16)= fctbt2
         fctcg3(16)= fctbt3

         fctch1(17)= 0.106765e+04
         fctch2(17)=-0.156808e+01
         fctcg0(17)= fctbt0
         fctcg1(17)= fctbt1
         fctcg2(17)= fctbt2
         fctcg3(17)= fctbt3

         fctch1(18)= 0.106765e+04
         fctch2(18)=-0.156808e+01
         fctcg0(18)= fctbt0
         fctcg1(18)= fctbt1
         fctcg2(18)= fctbt2
         fctcg3(18)= fctbt3

         fctch1(19)= 0.106765e+04
         fctch2(19)=-0.156808e+01
         fctcg0(19)= fctbt0
         fctcg1(19)= fctbt1
         fctcg2(19)= fctbt2
         fctcg3(19)= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one
         fctqun(8 )= one
         fctqun(9 )= one
         fctqun(10)= one
         fctqun(11)= one
         fctqun(12)= one
         fctqun(13)= one
         fctqun(14)= one
         fctqun(15)= one
         fctqun(16)= one
         fctqun(17)= one
         fctqun(18)= one
         fctqun(19)= one

! Define the scaling factors for the charges and for the fctnps factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one
         fctscf(8 )= one
         fctscf(9 )= one
         fctscf(10)= one
         fctscf(11)= one
         fctscf(12)= one
         fctscf(13)= one
         fctscf(14)= one
         fctscf(15)= one
         fctscf(16)= one
         fctscf(17)= one
         fctscf(18)= one
         fctscf(19)= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero
         fctnps(8 )= zero
         fctnps(9 )= zero
         fctnps(10)= zero
         fctnps(11)= zero
         fctnps(12)= zero
         fctnps(13)= zero
         fctnps(14)= zero
         fctnps(15)= zero
         fctnps(16)= zero
         fctnps(17)= zero
         fctnps(18)= zero
         fctnps(19)= zero

      endif

      if ((fctcps == 22    )  .and. &
          (fcteps == one   )  .and. &
          (fctfps == 4     )) then

! this FACTS parameter set was derived with no boundary conditions.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/one
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der waals radii.

         fctnnv    = 19

! Define the number of actual van der waals radii.

         fctnav    = 19

! Define the native van der waals radii.

         fctrvw(1 )= 0.2245d0
         fctrvw(2 )= 0.4500d0
         fctrvw(3 )= 0.7000d0
         fctrvw(4 )= 0.9000d0
         fctrvw(5 )= 1.3200d0
         fctrvw(6 )= 1.3582d0
         fctrvw(7 )= 1.4680d0
         fctrvw(8 )= 1.7000d0
         fctrvw(9 )= 1.7700d0
         fctrvw(10)= 1.8000d0
         fctrvw(11)= 1.8500d0
         fctrvw(12)= 1.9750d0
         fctrvw(13)= 1.9924d0
         fctrvw(14)= 2.0000d0
         fctrvw(15)= 2.0600d0
         fctrvw(16)= 2.1750d0
         fctrvw(17)= 2.2350d0
         fctrvw(18)= 2.2750d0
         fctrvw(19)= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =12.0d0
         fctcil    =14.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.714652e+01
         fctcsc(2 )= 0.986561e+01
         fctcsc(3 )= 0.600000e+01
         fctcsc(4 )= 0.712021e+01
         fctcsc(5 )= 0.814320e+01
         fctcsc(6 )= 0.838075e+01
         fctcsc(7 )= 0.802833e+01
         fctcsc(8 )= 0.853973e+01
         fctcsc(9 )= 0.850368e+01
         fctcsc(10)= 0.837127e+01
         fctcsc(11)= 0.849777e+01
         fctcsc(12)= 0.884036e+01
         fctcsc(13)= 0.920827e+01
         fctcsc(14)= 0.882238e+01
         fctcsc(15)= 0.986489e+01
         fctcsc(16)= 0.891896e+01
         fctcsc(17)= 0.600000e+01
         fctcsc(18)= 0.953929e+01
         fctcsc(19)= 0.600000e+01
         fctcic    =12.0d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.148587e+03
         fctcb2(1 )=-0.196968e+01
         fctca0(1 )=-0.158448e+03
         fctca1(1 )= 0.136063e+03
         fctca2(1 )= 0.498271e-02
         fctca3(1 )=-0.379617e+02

         fctcb1(2 )= 0.148938e+03
         fctcb2(2 )=-0.132070e+01
         fctca0(2 )=-0.173338e+03
         fctca1(2 )= 0.155882e+03
         fctca2(2 )= 0.811466e-02
         fctca3(2 )= 0.244496e+03

         fctcb1(3 )= 0.148938e+03
         fctcb2(3 )=-0.132070e+01
         fctca0(3 )=-0.173338e+03
         fctca1(3 )= 0.155882e+03
         fctca2(3 )= 0.811466e-02
         fctca3(3 )= 0.244496e+03

         fctcb1(4 )= 0.148938e+03
         fctcb2(4 )=-0.132070e+01
         fctca0(4 )=-0.173338e+03
         fctca1(4 )= 0.155882e+03
         fctca2(4 )= 0.811466e-02
         fctca3(4 )= 0.244496e+03

         fctcb1(5 )=-0.305104e+03
         fctcb2(5 )=-0.111371e+01
         fctca0(5 )=-0.108983e+03
         fctca1(5 )= 0.101877e+03
         fctca2(5 )= 0.317461e-02
         fctca3(5 )= 0.367529e+03

         fctcb1(6 )=-0.221576e+03
         fctcb2(6 )=-0.138737e+01
         fctca0(6 )=-0.110915e+03
         fctca1(6 )= 0.103524e+03
         fctca2(6 )= 0.299797e-02
         fctca3(6 )= 0.405182e+03

         fctcb1(7 )= 0.369364e+02
         fctcb2(7 )=-0.498136e+00
         fctca0(7 )=-0.104247e+03
         fctca1(7 )= 0.949737e+02
         fctca2(7 )= 0.502617e-02
         fctca3(7 )= 0.642582e+03

         fctcb1(8 )=-0.341674e+03
         fctcb2(8 )=-0.883105e+00
         fctca0(8 )=-0.943883e+02
         fctca1(8 )= 0.942109e+02
         fctca2(8 )= 0.232945e-02
         fctca3(8 )= 0.680033e+03

         fctcb1(9 )=-0.221713e+03
         fctcb2(9 )=-0.836988e+00
         fctca0(9 )=-0.929244e+02
         fctca1(9 )= 0.929224e+02
         fctca2(9 )= 0.243538e-02
         fctca3(9 )= 0.693472e+03

         fctcb1(10)= 0.865806e+01
         fctcb2(10)=-0.108666e+01
         fctca0(10)=-0.123349e+03
         fctca1(10)= 0.123347e+03
         fctca2(10)= 0.185919e-02
         fctca3(10)= 0.351961e+03

         fctcb1(11)=-0.459976e+03
         fctcb2(11)=-0.523950e+00
         fctca0(11)=-0.887323e+02
         fctca1(11)= 0.887303e+02
         fctca2(11)= 0.208277e-02
         fctca3(11)= 0.755828e+03

         fctcb1(12)= 0.503292e+03
         fctcb2(12)=-0.107570e+01
         fctca0(12)=-0.790794e+02
         fctca1(12)= 0.687793e+02
         fctca2(12)= 0.354875e-02
         fctca3(12)= 0.896420e+03

         fctcb1(13)=-0.114883e+03
         fctcb2(13)=-0.964277e+00
         fctca0(13)=-0.819019e+02
         fctca1(13)= 0.818999e+02
         fctca2(13)= 0.189699e-02
         fctca3(13)= 0.948533e+03

         fctcb1(14)=-0.258760e+01
         fctcb2(14)=-0.665425e+00
         fctca0(14)=-0.810234e+02
         fctca1(14)= 0.810214e+02
         fctca2(14)= 0.203711e-02
         fctca3(14)= 0.962221e+03

         fctcb1(15)=-0.562214e+02
         fctcb2(15)=-0.844556e+00
         fctca0(15)=-0.755199e+02
         fctca1(15)= 0.755179e+02
         fctca2(15)= 0.193532e-02
         fctca3(15)= 0.131543e+04

         fctcb1(16)=-0.874508e+02
         fctcb2(16)=-0.806689e+00
         fctca0(16)=-0.731234e+02
         fctca1(16)= 0.730867e+02
         fctca2(16)= 0.228109e-02
         fctca3(16)= 0.101119e+04

         fctcb1(17)= 0.715275e+03
         fctcb2(17)=-0.127219e+01
         fctca0(17)=-0.728567e+02
         fctca1(17)= 0.726676e+02
         fctca2(17)= 0.182543e-02
         fctca3(17)= 0.123203e+04

         fctcb1(18)= 0.715275e+03
         fctcb2(18)=-0.127219e+01
         fctca0(18)=-0.728567e+02
         fctca1(18)= 0.726676e+02
         fctca2(18)= 0.182543e-02
         fctca3(18)= 0.123203e+04

         fctcb1(19)= 0.715275e+03
         fctcb2(19)=-0.127219e+01
         fctca0(19)=-0.728567e+02
         fctca1(19)= 0.726676e+02
         fctca2(19)= 0.182543e-02
         fctca3(19)= 0.123203e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.448975e+04
         fctcd2(1 )=-0.977422e+01
         fctcc0(1 )= 0.331627e+02
         fctcc1(1 )=-0.331607e+02
         fctcc2(1 )= 0.276020e-01
         fctcc3(1 )=-0.719487e+04

         fctcd1(2 )=-0.252195e+04
         fctcd2(2 )=-0.310804e+01
         fctcc0(2 )= 0.878806e+01
         fctcc1(2 )=-0.878606e+01
         fctcc2(2 )= 0.469353e-02
         fctcc3(2 )=-0.187500e+04

         fctcd1(3 )=-0.252195e+04
         fctcd2(3 )=-0.310804e+01
         fctcc0(3 )= 0.878806e+01
         fctcc1(3 )=-0.878606e+01
         fctcc2(3 )= 0.469353e-02
         fctcc3(3 )=-0.187500e+04

         fctcd1(4 )=-0.252195e+04
         fctcd2(4 )=-0.310804e+01
         fctcc0(4 )= 0.878806e+01
         fctcc1(4 )=-0.878606e+01
         fctcc2(4 )= 0.469353e-02
         fctcc3(4 )=-0.187500e+04

         fctcd1(5 )=-0.192740e+03
         fctcd2(5 )=-0.176685e+01
         fctcc0(5 )= 0.100230e+03
         fctcc1(5 )=-0.100228e+03
         fctcc2(5 )= 0.630842e-02
         fctcc3(5 )=-0.578701e+03

         fctcd1(6 )=-0.730774e+03
         fctcd2(6 )=-0.226116e+01
         fctcc0(6 )= 0.163737e+03
         fctcc1(6 )=-0.163735e+03
         fctcc2(6 )= 0.324340e-02
         fctcc3(6 )=-0.143345e+04

         fctcd1(7 )=-0.358859e+04
         fctcd2(7 )=-0.205353e+01
         fctcc0(7 )= 0.455314e+02
         fctcc1(7 )=-0.455294e+02
         fctcc2(7 )= 0.186269e-02
         fctcc3(7 )=-0.251209e+04

         fctcd1(8 )=-0.195091e+04
         fctcd2(8 )=-0.167523e+01
         fctcc0(8 )= 0.650048e+02
         fctcc1(8 )=-0.650028e+02
         fctcc2(8 )= 0.241968e-02
         fctcc3(8 )=-0.138528e+04

         fctcd1(9 )=-0.296035e+04
         fctcd2(9 )=-0.147375e+01
         fctcc0(9 )= 0.793192e+02
         fctcc1(9 )=-0.793172e+02
         fctcc2(9 )= 0.165563e-02
         fctcc3(9 )=-0.221748e+04

         fctcd1(10)=-0.117464e+05
         fctcd2(10)=-0.539960e+01
         fctcc0(10)= 0.317380e+02
         fctcc1(10)=-0.315388e+02
         fctcc2(10)= 0.104259e-02
         fctcc3(10)=-0.750000e+04

         fctcd1(11)=-0.189210e+04
         fctcd2(11)=-0.160476e+01
         fctcc0(11)= 0.747670e+02
         fctcc1(11)=-0.747650e+02
         fctcc2(11)= 0.269841e-02
         fctcc3(11)=-0.128355e+04

         fctcd1(12)= 0.288531e+03
         fctcd2(12)=-0.222335e+01
         fctcc0(12)= 0.173162e+03
         fctcc1(12)=-0.173110e+03
         fctcc2(12)= 0.506917e-02
         fctcc3(12)=-0.326051e+03

         fctcd1(13)=-0.153676e+04
         fctcd2(13)=-0.115195e+01
         fctcc0(13)= 0.392391e+02
         fctcc1(13)=-0.392371e+02
         fctcc2(13)= 0.242002e-02
         fctcc3(13)=-0.860972e+03

         fctcd1(14)=-0.806211e+04
         fctcd2(14)=-0.227039e+01
         fctcc0(14)= 0.167153e+03
         fctcc1(14)=-0.166955e+03
         fctcc2(14)= 0.113533e-02
         fctcc3(14)=-0.675647e+04

         fctcd1(15)=-0.860544e+04
         fctcd2(15)=-0.636356e+00
         fctcc0(15)= 0.334933e+02
         fctcc1(15)=-0.334913e+02
         fctcc2(15)= 0.100623e-02
         fctcc3(15)=-0.364055e+04

         fctcd1(16)=-0.596334e+03
         fctcd2(16)=-0.186455e+01
         fctcc0(16)= 0.632048e+02
         fctcc1(16)=-0.632028e+02
         fctcc2(16)= 0.391227e-02
         fctcc3(16)=-0.451094e+03

         fctcd1(17)=-0.666671e+03
         fctcd2(17)=-0.201932e+01
         fctcc0(17)= 0.179900e+03
         fctcc1(17)=-0.179898e+03
         fctcc2(17)= 0.295319e-02
         fctcc3(17)=-0.108272e+04

         fctcd1(18)=-0.666671e+03
         fctcd2(18)=-0.201932e+01
         fctcc0(18)= 0.179900e+03
         fctcc1(18)=-0.179898e+03
         fctcc2(18)= 0.295319e-02
         fctcc3(18)=-0.108272e+04

         fctcd1(19)=-0.666671e+03
         fctcd2(19)=-0.201932e+01
         fctcc0(19)= 0.179900e+03
         fctcc1(19)=-0.179898e+03
         fctcc2(19)= 0.295319e-02
         fctcc3(19)=-0.108272e+04

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.148587e+03
         fctcf2(1 )=-0.196968e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )= 0.148938e+03
         fctcf2(2 )=-0.132070e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.148938e+03
         fctcf2(3 )=-0.132070e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )= 0.148938e+03
         fctcf2(4 )=-0.132070e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )=-0.305104e+03
         fctcf2(5 )=-0.111371e+01
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )=-0.221576e+03
         fctcf2(6 )=-0.138737e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.369364e+02
         fctcf2(7 )=-0.498136e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctcf1(8 )=-0.341674e+03
         fctcf2(8 )=-0.883105e+00
         fctce0(8 )= fctap0
         fctce1(8 )= fctap1
         fctce2(8 )= fctap2
         fctce3(8 )= fctap3

         fctcf1(9 )=-0.221713e+03
         fctcf2(9 )=-0.836988e+00
         fctce0(9 )= fctap0
         fctce1(9 )= fctap1
         fctce2(9 )= fctap2
         fctce3(9 )= fctap3

         fctcf1(10)= 0.865806e+01
         fctcf2(10)=-0.108666e+01
         fctce0(10)= fctap0
         fctce1(10)= fctap1
         fctce2(10)= fctap2
         fctce3(10)= fctap3

         fctcf1(11)=-0.459976e+03
         fctcf2(11)=-0.523950e+00
         fctce0(11)= fctap0
         fctce1(11)= fctap1
         fctce2(11)= fctap2
         fctce3(11)= fctap3

         fctcf1(12)= 0.503292e+03
         fctcf2(12)=-0.107570e+01
         fctce0(12)= fctap0
         fctce1(12)= fctap1
         fctce2(12)= fctap2
         fctce3(12)= fctap3

         fctcf1(13)=-0.114883e+03
         fctcf2(13)=-0.964277e+00
         fctce0(13)= fctap0
         fctce1(13)= fctap1
         fctce2(13)= fctap2
         fctce3(13)= fctap3

         fctcf1(14)=-0.258760e+01
         fctcf2(14)=-0.665425e+00
         fctce0(14)= fctap0
         fctce1(14)= fctap1
         fctce2(14)= fctap2
         fctce3(14)= fctap3

         fctcf1(15)=-0.562214e+02
         fctcf2(15)=-0.844556e+00
         fctce0(15)= fctap0
         fctce1(15)= fctap1
         fctce2(15)= fctap2
         fctce3(15)= fctap3

         fctcf1(16)=-0.874508e+02
         fctcf2(16)=-0.806689e+00
         fctce0(16)= fctap0
         fctce1(16)= fctap1
         fctce2(16)= fctap2
         fctce3(16)= fctap3

         fctcf1(17)= 0.715275e+03
         fctcf2(17)=-0.127219e+01
         fctce0(17)= fctap0
         fctce1(17)= fctap1
         fctce2(17)= fctap2
         fctce3(17)= fctap3

         fctcf1(18)= 0.715275e+03
         fctcf2(18)=-0.127219e+01
         fctce0(18)= fctap0
         fctce1(18)= fctap1
         fctce2(18)= fctap2
         fctce3(18)= fctap3

         fctcf1(19)= 0.715275e+03
         fctcf2(19)=-0.127219e+01
         fctce0(19)= fctap0
         fctce1(19)= fctap1
         fctce2(19)= fctap2
         fctce3(19)= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.148587e+03
         fctch2(1 )=-0.196968e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )= 0.148938e+03
         fctch2(2 )=-0.132070e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.148938e+03
         fctch2(3 )=-0.132070e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )= 0.148938e+03
         fctch2(4 )=-0.132070e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )=-0.305104e+03
         fctch2(5 )=-0.111371e+01
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )=-0.221576e+03
         fctch2(6 )=-0.138737e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.369364e+02
         fctch2(7 )=-0.498136e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctch1(8 )=-0.341674e+03
         fctch2(8 )=-0.883105e+00
         fctcg0(8 )= fctbt0
         fctcg1(8 )= fctbt1
         fctcg2(8 )= fctbt2
         fctcg3(8 )= fctbt3

         fctch1(9 )=-0.221713e+03
         fctch2(9 )=-0.836988e+00
         fctcg0(9 )= fctbt0
         fctcg1(9 )= fctbt1
         fctcg2(9 )= fctbt2
         fctcg3(9 )= fctbt3

         fctch1(10)= 0.865806e+01
         fctch2(10)=-0.108666e+01
         fctcg0(10)= fctbt0
         fctcg1(10)= fctbt1
         fctcg2(10)= fctbt2
         fctcg3(10)= fctbt3

         fctch1(11)=-0.459976e+03
         fctch2(11)=-0.523950e+00
         fctcg0(11)= fctbt0
         fctcg1(11)= fctbt1
         fctcg2(11)= fctbt2
         fctcg3(11)= fctbt3

         fctch1(12)= 0.503292e+03
         fctch2(12)=-0.107570e+01
         fctcg0(12)= fctbt0
         fctcg1(12)= fctbt1
         fctcg2(12)= fctbt2
         fctcg3(12)= fctbt3

         fctch1(13)=-0.114883e+03
         fctch2(13)=-0.964277e+00
         fctcg0(13)= fctbt0
         fctcg1(13)= fctbt1
         fctcg2(13)= fctbt2
         fctcg3(13)= fctbt3

         fctch1(14)=-0.258760e+01
         fctch2(14)=-0.665425e+00
         fctcg0(14)= fctbt0
         fctcg1(14)= fctbt1
         fctcg2(14)= fctbt2
         fctcg3(14)= fctbt3

         fctch1(15)=-0.562214e+02
         fctch2(15)=-0.844556e+00
         fctcg0(15)= fctbt0
         fctcg1(15)= fctbt1
         fctcg2(15)= fctbt2
         fctcg3(15)= fctbt3

         fctch1(16)=-0.874508e+02
         fctch2(16)=-0.806689e+00
         fctcg0(16)= fctbt0
         fctcg1(16)= fctbt1
         fctcg2(16)= fctbt2
         fctcg3(16)= fctbt3

         fctch1(17)= 0.715275e+03
         fctch2(17)=-0.127219e+01
         fctcg0(17)= fctbt0
         fctcg1(17)= fctbt1
         fctcg2(17)= fctbt2
         fctcg3(17)= fctbt3

         fctch1(18)= 0.715275e+03
         fctch2(18)=-0.127219e+01
         fctcg0(18)= fctbt0
         fctcg1(18)= fctbt1
         fctcg2(18)= fctbt2
         fctcg3(18)= fctbt3

         fctch1(19)= 0.715275e+03
         fctch2(19)=-0.127219e+01
         fctcg0(19)= fctbt0
         fctcg1(19)= fctbt1
         fctcg2(19)= fctbt2
         fctcg3(19)= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = zero

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one
         fctqun(8 )= one
         fctqun(9 )= one
         fctqun(10)= one
         fctqun(11)= one
         fctqun(12)= one
         fctqun(13)= one
         fctqun(14)= one
         fctqun(15)= one
         fctqun(16)= one
         fctqun(17)= one
         fctqun(18)= one
         fctqun(19)= one

! Define the scaling factors for the charges and for the fctnps factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one
         fctscf(8 )= one
         fctscf(9 )= one
         fctscf(10)= one
         fctscf(11)= one
         fctscf(12)= one
         fctscf(13)= one
         fctscf(14)= one
         fctscf(15)= one
         fctscf(16)= one
         fctscf(17)= one
         fctscf(18)= one
         fctscf(19)= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero
         fctnps(8 )= zero
         fctnps(9 )= zero
         fctnps(10)= zero
         fctnps(11)= zero
         fctnps(12)= zero
         fctnps(13)= zero
         fctnps(14)= zero
         fctnps(15)= zero
         fctnps(16)= zero
         fctnps(17)= zero
         fctnps(18)= zero
         fctnps(19)= zero

      endif

      if ((fctcps == 22    )  .and. &
          (fcteps == two   )  .and. &
          (fctfps == 1     )) then

! this FACTS parameter set was derived with no boundary conditions.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/two
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der waals radii.

         fctnnv    = 19

! Define the number of actual van der waals radii.

         fctnav    = 19

! Define the native van der waals radii.

         fctrvw(1 )= 0.2245d0
         fctrvw(2 )= 0.4500d0
         fctrvw(3 )= 0.7000d0
         fctrvw(4 )= 0.9000d0
         fctrvw(5 )= 1.3200d0
         fctrvw(6 )= 1.3582d0
         fctrvw(7 )= 1.4680d0
         fctrvw(8 )= 1.7000d0
         fctrvw(9 )= 1.7700d0
         fctrvw(10)= 1.8000d0
         fctrvw(11)= 1.8500d0
         fctrvw(12)= 1.9750d0
         fctrvw(13)= 1.9924d0
         fctrvw(14)= 2.0000d0
         fctrvw(15)= 2.0600d0
         fctrvw(16)= 2.1750d0
         fctrvw(17)= 2.2350d0
         fctrvw(18)= 2.2750d0
         fctrvw(19)= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =12.0d0
         fctcil    =14.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.706222e+01
         fctcsc(2 )= 0.767241e+01
         fctcsc(3 )= 0.600000e+01
         fctcsc(4 )= 0.670164e+01
         fctcsc(5 )= 0.824601e+01
         fctcsc(6 )= 0.829473e+01
         fctcsc(7 )= 0.749910e+01
         fctcsc(8 )= 0.845824e+01
         fctcsc(9 )= 0.844679e+01
         fctcsc(10)= 0.835774e+01
         fctcsc(11)= 0.844792e+01
         fctcsc(12)= 0.909484e+01
         fctcsc(13)= 0.926378e+01
         fctcsc(14)= 0.878606e+01
         fctcsc(15)= 0.999486e+01
         fctcsc(16)= 0.890732e+01
         fctcsc(17)= 0.600000e+01
         fctcsc(18)= 0.971278e+01
         fctcsc(19)= 0.600000e+01
         fctcic    =12.0d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.275971e+03
         fctcb2(1 )=-0.200203e+01
         fctca0(1 )=-0.751192e+02
         fctca1(1 )= 0.639494e+02
         fctca2(1 )= 0.474256e-02
         fctca3(1 )=-0.941027e+02

         fctcb1(2 )=-0.594710e+03
         fctcb2(2 )=-0.213036e+01
         fctca0(2 )=-0.698183e+02
         fctca1(2 )= 0.698163e+02
         fctca2(2 )= 0.564903e-02
         fctca3(2 )=-0.131023e+03

         fctcb1(3 )=-0.594710e+03
         fctcb2(3 )=-0.213036e+01
         fctca0(3 )=-0.698183e+02
         fctca1(3 )= 0.698163e+02
         fctca2(3 )= 0.564903e-02
         fctca3(3 )=-0.131023e+03

         fctcb1(4 )=-0.594710e+03
         fctcb2(4 )=-0.213036e+01
         fctca0(4 )=-0.698183e+02
         fctca1(4 )= 0.698163e+02
         fctca2(4 )= 0.564903e-02
         fctca3(4 )=-0.131023e+03

         fctcb1(5 )=-0.973434e+02
         fctcb2(5 )=-0.130741e+01
         fctca0(5 )=-0.583544e+02
         fctca1(5 )= 0.560110e+02
         fctca2(5 )= 0.291132e-02
         fctca3(5 )= 0.357392e+03

         fctcb1(6 )=-0.191605e+03
         fctcb2(6 )=-0.130440e+01
         fctca0(6 )=-0.563360e+02
         fctca1(6 )= 0.537271e+02
         fctca2(6 )= 0.290104e-02
         fctca3(6 )= 0.407967e+03

         fctcb1(7 )= 0.511143e+02
         fctcb2(7 )=-0.881330e+00
         fctca0(7 )=-0.573999e+02
         fctca1(7 )= 0.573979e+02
         fctca2(7 )= 0.408694e-02
         fctca3(7 )= 0.467901e+03

         fctcb1(8 )=-0.258749e+03
         fctcb2(8 )=-0.943276e+00
         fctca0(8 )=-0.481314e+02
         fctca1(8 )= 0.481294e+02
         fctca2(8 )= 0.227343e-02
         fctca3(8 )= 0.635441e+03

         fctcb1(9 )=-0.266819e+03
         fctcb2(9 )=-0.799658e+00
         fctca0(9 )=-0.469718e+02
         fctca1(9 )= 0.469698e+02
         fctca2(9 )= 0.222330e-02
         fctca3(9 )= 0.658810e+03

         fctcb1(10)= 0.120345e+03
         fctcb2(10)=-0.115631e+01
         fctca0(10)=-0.873248e+02
         fctca1(10)= 0.873228e+02
         fctca2(10)= 0.158464e-02
         fctca3(10)=-0.276594e+02

         fctcb1(11)=-0.392515e+03
         fctcb2(11)=-0.546163e+00
         fctca0(11)=-0.449052e+02
         fctca1(11)= 0.449032e+02
         fctca2(11)= 0.203911e-02
         fctca3(11)= 0.722270e+03

         fctcb1(12)= 0.132837e+04
         fctcb2(12)=-0.192121e+01
         fctca0(12)=-0.485138e+02
         fctca1(12)= 0.413458e+02
         fctca2(12)= 0.371541e-02
         fctca3(12)= 0.759156e+03

         fctcb1(13)= 0.397096e+01
         fctcb2(13)=-0.107441e+01
         fctca0(13)=-0.430531e+02
         fctca1(13)= 0.430511e+02
         fctca2(13)= 0.173781e-02
         fctca3(13)= 0.879763e+03

         fctcb1(14)= 0.431695e+02
         fctcb2(14)=-0.675727e+00
         fctca0(14)=-0.410881e+02
         fctca1(14)= 0.410861e+02
         fctca2(14)= 0.197575e-02
         fctca3(14)= 0.925435e+03

         fctcb1(15)=-0.588491e+02
         fctcb2(15)=-0.853175e+00
         fctca0(15)=-0.378659e+02
         fctca1(15)= 0.378639e+02
         fctca2(15)= 0.179140e-02
         fctca3(15)= 0.132067e+04

         fctcb1(16)= 0.112463e+03
         fctcb2(16)=-0.964233e+00
         fctca0(16)=-0.382644e+02
         fctca1(16)= 0.382624e+02
         fctca2(16)= 0.216212e-02
         fctca3(16)= 0.951170e+03

         fctcb1(17)= 0.130860e+04
         fctcb2(17)=-0.177237e+01
         fctca0(17)=-0.661707e+02
         fctca1(17)= 0.661687e+02
         fctca2(17)= 0.118151e-02
         fctca3(17)= 0.406434e+03

         fctcb1(18)= 0.130860e+04
         fctcb2(18)=-0.177237e+01
         fctca0(18)=-0.661707e+02
         fctca1(18)= 0.661687e+02
         fctca2(18)= 0.118151e-02
         fctca3(18)= 0.406434e+03

         fctcb1(19)= 0.130860e+04
         fctcb2(19)=-0.177237e+01
         fctca0(19)=-0.661707e+02
         fctca1(19)= 0.661687e+02
         fctca2(19)= 0.118151e-02
         fctca3(19)= 0.406434e+03

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.597528e+04
         fctcd2(1 )=-0.132626e+02
         fctcc0(1 )= 0.331627e+02
         fctcc1(1 )=-0.331607e+02
         fctcc2(1 )= 0.268702e-01
         fctcc3(1 )=-0.945280e+04

         fctcd1(2 )=-0.952570e+04
         fctcd2(2 )=-0.892289e+01
         fctcc0(2 )= 0.875739e+01
         fctcc1(2 )=-0.875539e+01
         fctcc2(2 )= 0.150488e-02
         fctcc3(2 )=-0.731427e+04

         fctcd1(3 )=-0.952570e+04
         fctcd2(3 )=-0.892289e+01
         fctcc0(3 )= 0.875739e+01
         fctcc1(3 )=-0.875539e+01
         fctcc2(3 )= 0.150488e-02
         fctcc3(3 )=-0.731427e+04

         fctcd1(4 )=-0.952570e+04
         fctcd2(4 )=-0.892289e+01
         fctcc0(4 )= 0.875739e+01
         fctcc1(4 )=-0.875539e+01
         fctcc2(4 )= 0.150488e-02
         fctcc3(4 )=-0.731427e+04

         fctcd1(5 )=-0.192740e+03
         fctcd2(5 )=-0.176685e+01
         fctcc0(5 )= 0.100230e+03
         fctcc1(5 )=-0.100228e+03
         fctcc2(5 )= 0.630842e-02
         fctcc3(5 )=-0.578701e+03

         fctcd1(6 )=-0.676447e+03
         fctcd2(6 )=-0.222803e+01
         fctcc0(6 )= 0.163762e+03
         fctcc1(6 )=-0.163760e+03
         fctcc2(6 )= 0.345894e-02
         fctcc3(6 )=-0.133154e+04

         fctcd1(7 )=-0.281663e+04
         fctcd2(7 )=-0.255432e+01
         fctcc0(7 )= 0.492274e+02
         fctcc1(7 )=-0.492254e+02
         fctcc2(7 )= 0.224746e-02
         fctcc3(7 )=-0.218549e+04

         fctcd1(8 )=-0.193271e+04
         fctcd2(8 )=-0.164887e+01
         fctcc0(8 )= 0.650194e+02
         fctcc1(8 )=-0.650174e+02
         fctcc2(8 )= 0.248175e-02
         fctcc3(8 )=-0.135331e+04

         fctcd1(9 )=-0.311522e+04
         fctcd2(9 )=-0.124537e+01
         fctcc0(9 )= 0.735675e+02
         fctcc1(9 )=-0.735655e+02
         fctcc2(9 )= 0.169442e-02
         fctcc3(9 )=-0.214416e+04

         fctcd1(10)=-0.222684e+05
         fctcd2(10)=-0.810128e+01
         fctcc0(10)= 0.315852e+02
         fctcc1(10)=-0.313963e+02
         fctcc2(10)= 0.579209e-03
         fctcc3(10)=-0.139485e+05

         fctcd1(11)=-0.193136e+04
         fctcd2(11)=-0.155601e+01
         fctcc0(11)= 0.746558e+02
         fctcc1(11)=-0.746538e+02
         fctcc2(11)= 0.270264e-02
         fctcc3(11)=-0.128202e+04

         fctcd1(12)= 0.288801e+03
         fctcd2(12)=-0.229942e+01
         fctcc0(12)= 0.174431e+03
         fctcc1(12)=-0.174344e+03
         fctcc2(12)= 0.455283e-02
         fctcc3(12)=-0.404115e+03

         fctcd1(13)=-0.421160e+04
         fctcd2(13)=-0.459151e+00
         fctcc0(13)= 0.346466e+02
         fctcc1(13)=-0.346446e+02
         fctcc2(13)= 0.138939e-02
         fctcc3(13)=-0.209693e+04

         fctcd1(14)=-0.811683e+04
         fctcd2(14)=-0.234357e+01
         fctcc0(14)= 0.167665e+03
         fctcc1(14)=-0.167472e+03
         fctcc2(14)= 0.113712e-02
         fctcc3(14)=-0.675676e+04

         fctcd1(15)=-0.627618e+04
         fctcd2(15)=-0.153047e+01
         fctcc0(15)= 0.104380e+03
         fctcc1(15)=-0.104378e+03
         fctcc2(15)= 0.732719e-03
         fctcc3(15)=-0.522811e+04

         fctcd1(16)=-0.178801e+05
         fctcd2(16)=-0.310708e+01
         fctcc0(16)= 0.172766e+03
         fctcc1(16)=-0.172687e+03
         fctcc2(16)= 0.348553e-03
         fctcc3(16)=-0.151252e+05

         fctcd1(17)=-0.633935e+03
         fctcd2(17)=-0.200029e+01
         fctcc0(17)= 0.178299e+03
         fctcc1(17)=-0.178297e+03
         fctcc2(17)= 0.292989e-02
         fctcc3(17)=-0.106603e+04

         fctcd1(18)=-0.633935e+03
         fctcd2(18)=-0.200029e+01
         fctcc0(18)= 0.178299e+03
         fctcc1(18)=-0.178297e+03
         fctcc2(18)= 0.292989e-02
         fctcc3(18)=-0.106603e+04

         fctcd1(19)=-0.633935e+03
         fctcd2(19)=-0.200029e+01
         fctcc0(19)= 0.178299e+03
         fctcc1(19)=-0.178297e+03
         fctcc2(19)= 0.292989e-02
         fctcc3(19)=-0.106603e+04

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.275971e+03
         fctcf2(1 )=-0.200203e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )=-0.594710e+03
         fctcf2(2 )=-0.213036e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )=-0.594710e+03
         fctcf2(3 )=-0.213036e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )=-0.594710e+03
         fctcf2(4 )=-0.213036e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )=-0.973434e+02
         fctcf2(5 )=-0.130741e+01
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )=-0.191605e+03
         fctcf2(6 )=-0.130440e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )= 0.511143e+02
         fctcf2(7 )=-0.881330e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctcf1(8 )=-0.258749e+03
         fctcf2(8 )=-0.943276e+00
         fctce0(8 )= fctap0
         fctce1(8 )= fctap1
         fctce2(8 )= fctap2
         fctce3(8 )= fctap3

         fctcf1(9 )=-0.266819e+03
         fctcf2(9 )=-0.799658e+00
         fctce0(9 )= fctap0
         fctce1(9 )= fctap1
         fctce2(9 )= fctap2
         fctce3(9 )= fctap3

         fctcf1(10)= 0.120345e+03
         fctcf2(10)=-0.115631e+01
         fctce0(10)= fctap0
         fctce1(10)= fctap1
         fctce2(10)= fctap2
         fctce3(10)= fctap3

         fctcf1(11)=-0.392515e+03
         fctcf2(11)=-0.546163e+00
         fctce0(11)= fctap0
         fctce1(11)= fctap1
         fctce2(11)= fctap2
         fctce3(11)= fctap3

         fctcf1(12)= 0.132837e+04
         fctcf2(12)=-0.192121e+01
         fctce0(12)= fctap0
         fctce1(12)= fctap1
         fctce2(12)= fctap2
         fctce3(12)= fctap3

         fctcf1(13)= 0.397096e+01
         fctcf2(13)=-0.107441e+01
         fctce0(13)= fctap0
         fctce1(13)= fctap1
         fctce2(13)= fctap2
         fctce3(13)= fctap3

         fctcf1(14)= 0.431695e+02
         fctcf2(14)=-0.675727e+00
         fctce0(14)= fctap0
         fctce1(14)= fctap1
         fctce2(14)= fctap2
         fctce3(14)= fctap3

         fctcf1(15)=-0.588491e+02
         fctcf2(15)=-0.853175e+00
         fctce0(15)= fctap0
         fctce1(15)= fctap1
         fctce2(15)= fctap2
         fctce3(15)= fctap3

         fctcf1(16)= 0.112463e+03
         fctcf2(16)=-0.964233e+00
         fctce0(16)= fctap0
         fctce1(16)= fctap1
         fctce2(16)= fctap2
         fctce3(16)= fctap3

         fctcf1(17)= 0.130860e+04
         fctcf2(17)=-0.177237e+01
         fctce0(17)= fctap0
         fctce1(17)= fctap1
         fctce2(17)= fctap2
         fctce3(17)= fctap3

         fctcf1(18)= 0.130860e+04
         fctcf2(18)=-0.177237e+01
         fctce0(18)= fctap0
         fctce1(18)= fctap1
         fctce2(18)= fctap2
         fctce3(18)= fctap3

         fctcf1(19)= 0.130860e+04
         fctcf2(19)=-0.177237e+01
         fctce0(19)= fctap0
         fctce1(19)= fctap1
         fctce2(19)= fctap2
         fctce3(19)= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.275971e+03
         fctch2(1 )=-0.200203e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )=-0.594710e+03
         fctch2(2 )=-0.213036e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )=-0.594710e+03
         fctch2(3 )=-0.213036e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )=-0.594710e+03
         fctch2(4 )=-0.213036e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )=-0.973434e+02
         fctch2(5 )=-0.130741e+01
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )=-0.191605e+03
         fctch2(6 )=-0.130440e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )= 0.511143e+02
         fctch2(7 )=-0.881330e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctch1(8 )=-0.258749e+03
         fctch2(8 )=-0.943276e+00
         fctcg0(8 )= fctbt0
         fctcg1(8 )= fctbt1
         fctcg2(8 )= fctbt2
         fctcg3(8 )= fctbt3

         fctch1(9 )=-0.266819e+03
         fctch2(9 )=-0.799658e+00
         fctcg0(9 )= fctbt0
         fctcg1(9 )= fctbt1
         fctcg2(9 )= fctbt2
         fctcg3(9 )= fctbt3

         fctch1(10)= 0.120345e+03
         fctch2(10)=-0.115631e+01
         fctcg0(10)= fctbt0
         fctcg1(10)= fctbt1
         fctcg2(10)= fctbt2
         fctcg3(10)= fctbt3

         fctch1(11)=-0.392515e+03
         fctch2(11)=-0.546163e+00
         fctcg0(11)= fctbt0
         fctcg1(11)= fctbt1
         fctcg2(11)= fctbt2
         fctcg3(11)= fctbt3

         fctch1(12)= 0.132837e+04
         fctch2(12)=-0.192121e+01
         fctcg0(12)= fctbt0
         fctcg1(12)= fctbt1
         fctcg2(12)= fctbt2
         fctcg3(12)= fctbt3

         fctch1(13)= 0.397096e+01
         fctch2(13)=-0.107441e+01
         fctcg0(13)= fctbt0
         fctcg1(13)= fctbt1
         fctcg2(13)= fctbt2
         fctcg3(13)= fctbt3

         fctch1(14)= 0.431695e+02
         fctch2(14)=-0.675727e+00
         fctcg0(14)= fctbt0
         fctcg1(14)= fctbt1
         fctcg2(14)= fctbt2
         fctcg3(14)= fctbt3

         fctch1(15)=-0.588491e+02
         fctch2(15)=-0.853175e+00
         fctcg0(15)= fctbt0
         fctcg1(15)= fctbt1
         fctcg2(15)= fctbt2
         fctcg3(15)= fctbt3

         fctch1(16)= 0.112463e+03
         fctch2(16)=-0.964233e+00
         fctcg0(16)= fctbt0
         fctcg1(16)= fctbt1
         fctcg2(16)= fctbt2
         fctcg3(16)= fctbt3

         fctch1(17)= 0.130860e+04
         fctch2(17)=-0.177237e+01
         fctcg0(17)= fctbt0
         fctcg1(17)= fctbt1
         fctcg2(17)= fctbt2
         fctcg3(17)= fctbt3

         fctch1(18)= 0.130860e+04
         fctch2(18)=-0.177237e+01
         fctcg0(18)= fctbt0
         fctcg1(18)= fctbt1
         fctcg2(18)= fctbt2
         fctcg3(18)= fctbt3

         fctch1(19)= 0.130860e+04
         fctch2(19)=-0.177237e+01
         fctcg0(19)= fctbt0
         fctcg1(19)= fctbt1
         fctcg2(19)= fctbt2
         fctcg3(19)= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one
         fctqun(8 )= one
         fctqun(9 )= one
         fctqun(10)= one
         fctqun(11)= one
         fctqun(12)= one
         fctqun(13)= one
         fctqun(14)= one
         fctqun(15)= one
         fctqun(16)= one
         fctqun(17)= one
         fctqun(18)= one
         fctqun(19)= one

! Define the scaling factors for the charges and for the fctnps factors.
! they only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one
         fctscf(8 )= one
         fctscf(9 )= one
         fctscf(10)= one
         fctscf(11)= one
         fctscf(12)= one
         fctscf(13)= one
         fctscf(14)= one
         fctscf(15)= one
         fctscf(16)= one
         fctscf(17)= one
         fctscf(18)= one
         fctscf(19)= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero
         fctnps(8 )= zero
         fctnps(9 )= zero
         fctnps(10)= zero
         fctnps(11)= zero
         fctnps(12)= zero
         fctnps(13)= zero
         fctnps(14)= zero
         fctnps(15)= zero
         fctnps(16)= zero
         fctnps(17)= zero
         fctnps(18)= zero
         fctnps(19)= zero

      endif

      if ((fctcps == 22    )  .and. &
          (fcteps == two   )  .and. &
          (fctfps == 2     )) then

! This FACTS parameter set was derived with boundary conditions at
! negative and positive infinity.

         fctcho    = .true.

! Define some constants related to the dielectric medium.

         fctiem    = fctcel/two
         fcties    = fctcel/78.5d0
         fcttau    = fctiem-fcties
         fctpco    = four/(fcttau*fcttau)

! Define the number of native van der waals radii.

         fctnnv    = 19

! Define the number of actual van der waals radii.

         fctnav    = 19

! Define the native van der waals radii.

         fctrvw(1 )= 0.2245d0
         fctrvw(2 )= 0.4500d0
         fctrvw(3 )= 0.7000d0
         fctrvw(4 )= 0.9000d0
         fctrvw(5 )= 1.3200d0
         fctrvw(6 )= 1.3582d0
         fctrvw(7 )= 1.4680d0
         fctrvw(8 )= 1.7000d0
         fctrvw(9 )= 1.7700d0
         fctrvw(10)= 1.8000d0
         fctrvw(11)= 1.8500d0
         fctrvw(12)= 1.9750d0
         fctrvw(13)= 1.9924d0
         fctrvw(14)= 2.0000d0
         fctrvw(15)= 2.0600d0
         fctrvw(16)= 2.1750d0
         fctrvw(17)= 2.2350d0
         fctrvw(18)= 2.2750d0
         fctrvw(19)= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

         fctcsl    =12.0d0
         fctcil    =14.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

         fctcsc(1 )= 0.708767e+01
         fctcsc(2 )= 0.729938e+01
         fctcsc(3 )= 0.600000e+01
         fctcsc(4 )= 0.679166e+01
         fctcsc(5 )= 0.825820e+01
         fctcsc(6 )= 0.831263e+01
         fctcsc(7 )= 0.751347e+01
         fctcsc(8 )= 0.845933e+01
         fctcsc(9 )= 0.845001e+01
         fctcsc(10)= 0.827363e+01
         fctcsc(11)= 0.844284e+01
         fctcsc(12)= 0.921829e+01
         fctcsc(13)= 0.925872e+01
         fctcsc(14)= 0.878370e+01
         fctcsc(15)= 0.100000e+02
         fctcsc(16)= 0.890240e+01
         fctcsc(17)= 0.600000e+01
         fctcsc(18)= 0.954464e+01
         fctcsc(19)= 0.600000e+01
         fctcic    =12.0d0

! Define the parameters for calculating atomic solvation energies.

         fctcb1(1 )=-0.483656e+03
         fctcb2(1 )=-0.204915e+01
         fctca0(1 )=-0.700328e+02
         fctca1(1 )= 0.585122e+02
         fctca2(1 )= 0.440294e-02
         fctca3(1 )=-0.165164e+03

         fctcb1(2 )= 0.549650e+02
         fctcb2(2 )=-0.170556e+01
         fctca0(2 )=-0.898920e+02
         fctca1(2 )= 0.898900e+02
         fctca2(2 )= 0.709650e-02
         fctca3(2 )= 0.113554e+03

         fctcb1(3 )= 0.549650e+02
         fctcb2(3 )=-0.170556e+01
         fctca0(3 )=-0.898920e+02
         fctca1(3 )= 0.898900e+02
         fctca2(3 )= 0.709650e-02
         fctca3(3 )= 0.113554e+03

         fctcb1(4 )= 0.549650e+02
         fctcb2(4 )=-0.170556e+01
         fctca0(4 )=-0.898920e+02
         fctca1(4 )= 0.898900e+02
         fctca2(4 )= 0.709650e-02
         fctca3(4 )= 0.113554e+03

         fctcb1(5 )=-0.106521e+03
         fctcb2(5 )=-0.132303e+01
         fctca0(5 )=-0.612900e+02
         fctca1(5 )= 0.612880e+02
         fctca2(5 )= 0.256643e-02
         fctca3(5 )= 0.341634e+03

         fctcb1(6 )=-0.151213e+03
         fctcb2(6 )=-0.134286e+01
         fctca0(6 )=-0.595662e+02
         fctca1(6 )= 0.595642e+02
         fctca2(6 )= 0.255313e-02
         fctca3(6 )= 0.404864e+03

         fctcb1(7 )=-0.276787e+02
         fctcb2(7 )=-0.792865e+00
         fctca0(7 )=-0.551109e+02
         fctca1(7 )= 0.551089e+02
         fctca2(7 )= 0.406121e-02
         fctca3(7 )= 0.482670e+03

         fctcb1(8 )=-0.296518e+03
         fctcb2(8 )=-0.913736e+00
         fctca0(8 )=-0.475899e+02
         fctca1(8 )= 0.475879e+02
         fctca2(8 )= 0.227454e-02
         fctca3(8 )= 0.642012e+03

         fctcb1(9 )=-0.356797e+03
         fctcb2(9 )=-0.727135e+00
         fctca0(9 )=-0.457078e+02
         fctca1(9 )= 0.457058e+02
         fctca2(9 )= 0.223982e-02
         fctca3(9 )= 0.677236e+03

         fctcb1(10)=-0.518968e+03
         fctcb2(10)=-0.387026e+00
         fctca0(10)=-0.449460e+02
         fctca1(10)= 0.449440e+02
         fctca2(10)= 0.215001e-02
         fctca3(10)= 0.626869e+03

         fctcb1(11)=-0.489213e+03
         fctcb2(11)=-0.445345e+00
         fctca0(11)=-0.437312e+02
         fctca1(11)= 0.437292e+02
         fctca2(11)= 0.206363e-02
         fctca3(11)= 0.745653e+03

         fctcb1(12)= 0.276081e+03
         fctcb2(12)=-0.980170e+00
         fctca0(12)=-0.409634e+02
         fctca1(12)= 0.409614e+02
         fctca2(12)= 0.223955e-02
         fctca3(12)= 0.102502e+04

         fctcb1(13)=-0.200621e+03
         fctcb2(13)=-0.921209e+00
         fctca0(13)=-0.406057e+02
         fctca1(13)= 0.406037e+02
         fctca2(13)= 0.178926e-02
         fctca3(13)= 0.940900e+03

         fctcb1(14)= 0.420018e+01
         fctcb2(14)=-0.634365e+00
         fctca0(14)=-0.404514e+02
         fctca1(14)= 0.404494e+02
         fctca2(14)= 0.199922e-02
         fctca3(14)= 0.941249e+03

         fctcb1(15)= 0.161564e+03
         fctcb2(15)=-0.980790e+00
         fctca0(15)=-0.392732e+02
         fctca1(15)= 0.392712e+02
         fctca2(15)= 0.176127e-02
         fctca3(15)= 0.128210e+04

         fctcb1(16)= 0.887817e+01
         fctcb2(16)=-0.867689e+00
         fctca0(16)=-0.371967e+02
         fctca1(16)= 0.371947e+02
         fctca2(16)= 0.220477e-02
         fctca3(16)= 0.977038e+03

         fctcb1(17)= 0.609835e+03
         fctcb2(17)=-0.116869e+01
         fctca0(17)=-0.355617e+02
         fctca1(17)= 0.355597e+02
         fctca2(17)= 0.180182e-02
         fctca3(17)= 0.124516e+04

         fctcb1(18)= 0.609835e+03
         fctcb2(18)=-0.116869e+01
         fctca0(18)=-0.355617e+02
         fctca1(18)= 0.355597e+02
         fctca2(18)= 0.180182e-02
         fctca3(18)= 0.124516e+04

         fctcb1(19)= 0.609835e+03
         fctcb2(19)=-0.116869e+01
         fctca0(19)=-0.355617e+02
         fctca1(19)= 0.355597e+02
         fctca2(19)= 0.180182e-02
         fctca3(19)= 0.124516e+04

         fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

         fctcd1(1 )=-0.597528e+04
         fctcd2(1 )=-0.132626e+02
         fctcc0(1 )= 0.331627e+02
         fctcc1(1 )=-0.331607e+02
         fctcc2(1 )= 0.268702e-01
         fctcc3(1 )=-0.945280e+04

         fctcd1(2 )=-0.337531e+04
         fctcd2(2 )=-0.825278e+01
         fctcc0(2 )= 0.664761e+02
         fctcc1(2 )=-0.664741e+02
         fctcc2(2 )= 0.134909e-02
         fctcc3(2 )=-0.540283e+04

         fctcd1(3 )=-0.337531e+04
         fctcd2(3 )=-0.825278e+01
         fctcc0(3 )= 0.664761e+02
         fctcc1(3 )=-0.664741e+02
         fctcc2(3 )= 0.134909e-02
         fctcc3(3 )=-0.540283e+04

         fctcd1(4 )=-0.337531e+04
         fctcd2(4 )=-0.825278e+01
         fctcc0(4 )= 0.664761e+02
         fctcc1(4 )=-0.664741e+02
         fctcc2(4 )= 0.134909e-02
         fctcc3(4 )=-0.540283e+04

         fctcd1(5 )=-0.227334e+05
         fctcd2(5 )=-0.123295e+02
         fctcc0(5 )= 0.929710e+02
         fctcc1(5 )=-0.929690e+02
         fctcc2(5 )= 0.257980e-03
         fctcc3(5 )=-0.275568e+05

         fctcd1(6 )=-0.970016e+04
         fctcd2(6 )=-0.931682e+01
         fctcc0(6 )= 0.956008e+02
         fctcc1(6 )=-0.955988e+02
         fctcc2(6 )= 0.518505e-03
         fctcc3(6 )=-0.122744e+05

         fctcd1(7 )=-0.156406e+04
         fctcd2(7 )=-0.229863e+01
         fctcc0(7 )= 0.103364e+03
         fctcc1(7 )=-0.103362e+03
         fctcc2(7 )= 0.259637e-02
         fctcc3(7 )=-0.174776e+04

         fctcd1(8 )=-0.428198e+03
         fctcd2(8 )=-0.161616e+01
         fctcc0(8 )= 0.120763e+03
         fctcc1(8 )=-0.120761e+03
         fctcc2(8 )= 0.384137e-02
         fctcc3(8 )=-0.598349e+03

         fctcd1(9 )=-0.100680e+04
         fctcd2(9 )=-0.145899e+01
         fctcc0(9 )= 0.126278e+03
         fctcc1(9 )=-0.126276e+03
         fctcc2(9 )= 0.278499e-02
         fctcc3(9 )=-0.105848e+04

         fctcd1(10)=-0.584761e+04
         fctcd2(10)=-0.400291e+01
         fctcc0(10)= 0.128680e+03
         fctcc1(10)=-0.128678e+03
         fctcc2(10)= 0.117228e-02
         fctcc3(10)=-0.561052e+04

         fctcd1(11)=-0.368958e+03
         fctcd2(11)=-0.160743e+01
         fctcc0(11)= 0.132732e+03
         fctcc1(11)=-0.132730e+03
         fctcc2(11)= 0.453277e-02
         fctcc3(11)=-0.479033e+03

         fctcd1(12)=-0.386416e+04
         fctcd2(12)=-0.475745e+01
         fctcc0(12)= 0.143139e+03
         fctcc1(12)=-0.143137e+03
         fctcc2(12)= 0.111939e-02
         fctcc3(12)=-0.466434e+04

         fctcd1(13)=-0.228226e+03
         fctcd2(13)=-0.128129e+01
         fctcc0(13)= 0.144619e+03
         fctcc1(13)=-0.144617e+03
         fctcc2(13)= 0.354868e-02
         fctcc3(13)=-0.596089e+03

         fctcd1(14)=-0.330023e+05
         fctcd2(14)= 0.291087e+02
         fctcc0(14)= 0.145267e+03
         fctcc1(14)=-0.145265e+03
         fctcc2(14)= 0.283263e-03
         fctcc3(14)=-0.195503e+05

         fctcd1(15)=-0.261470e+03
         fctcd2(15)=-0.169021e+01
         fctcc0(15)= 0.150440e+03
         fctcc1(15)=-0.150438e+03
         fctcc2(15)= 0.265060e-02
         fctcc3(15)=-0.770449e+03

         fctcd1(16)= 0.140386e+02
         fctcd2(16)=-0.199711e+01
         fctcc0(16)= 0.160606e+03
         fctcc1(16)=-0.160604e+03
         fctcc2(16)= 0.460762e-02
         fctcc3(16)=-0.407165e+03

         fctcd1(17)= 0.457199e+03
         fctcd2(17)=-0.244355e+01
         fctcc0(17)= 0.169717e+03
         fctcc1(17)=-0.169715e+03
         fctcc2(17)= 0.434189e-02
         fctcc3(17)=-0.444777e+03

         fctcd1(18)= 0.457199e+03
         fctcd2(18)=-0.244355e+01
         fctcc0(18)= 0.169717e+03
         fctcc1(18)=-0.169715e+03
         fctcc2(18)= 0.434189e-02
         fctcc3(18)=-0.444777e+03

         fctcd1(19)= 0.457199e+03
         fctcd2(19)=-0.244355e+01
         fctcc0(19)= 0.169717e+03
         fctcc1(19)=-0.169715e+03
         fctcc2(19)= 0.434189e-02
         fctcc3(19)=-0.444777e+03

         fctuos    = zero

! Define the parameters for calculating alpha.

         fctcf1(1 )=-0.483656e+03
         fctcf2(1 )=-0.204915e+01
         fctce0(1 )= fctap0
         fctce1(1 )= fctap1
         fctce2(1 )= fctap2
         fctce3(1 )= fctap3

         fctcf1(2 )= 0.549650e+02
         fctcf2(2 )=-0.170556e+01
         fctce0(2 )= fctap0
         fctce1(2 )= fctap1
         fctce2(2 )= fctap2
         fctce3(2 )= fctap3

         fctcf1(3 )= 0.549650e+02
         fctcf2(3 )=-0.170556e+01
         fctce0(3 )= fctap0
         fctce1(3 )= fctap1
         fctce2(3 )= fctap2
         fctce3(3 )= fctap3

         fctcf1(4 )= 0.549650e+02
         fctcf2(4 )=-0.170556e+01
         fctce0(4 )= fctap0
         fctce1(4 )= fctap1
         fctce2(4 )= fctap2
         fctce3(4 )= fctap3

         fctcf1(5 )=-0.106521e+03
         fctcf2(5 )=-0.132303e+01
         fctce0(5 )= fctap0
         fctce1(5 )= fctap1
         fctce2(5 )= fctap2
         fctce3(5 )= fctap3

         fctcf1(6 )=-0.151213e+03
         fctcf2(6 )=-0.134286e+01
         fctce0(6 )= fctap0
         fctce1(6 )= fctap1
         fctce2(6 )= fctap2
         fctce3(6 )= fctap3

         fctcf1(7 )=-0.276787e+02
         fctcf2(7 )=-0.792865e+00
         fctce0(7 )= fctap0
         fctce1(7 )= fctap1
         fctce2(7 )= fctap2
         fctce3(7 )= fctap3

         fctcf1(8 )=-0.296518e+03
         fctcf2(8 )=-0.913736e+00
         fctce0(8 )= fctap0
         fctce1(8 )= fctap1
         fctce2(8 )= fctap2
         fctce3(8 )= fctap3

         fctcf1(9 )=-0.356797e+03
         fctcf2(9 )=-0.727135e+00
         fctce0(9 )= fctap0
         fctce1(9 )= fctap1
         fctce2(9 )= fctap2
         fctce3(9 )= fctap3

         fctcf1(10)=-0.518968e+03
         fctcf2(10)=-0.387026e+00
         fctce0(10)= fctap0
         fctce1(10)= fctap1
         fctce2(10)= fctap2
         fctce3(10)= fctap3

         fctcf1(11)=-0.489213e+03
         fctcf2(11)=-0.445345e+00
         fctce0(11)= fctap0
         fctce1(11)= fctap1
         fctce2(11)= fctap2
         fctce3(11)= fctap3

         fctcf1(12)= 0.276081e+03
         fctcf2(12)=-0.980170e+00
         fctce0(12)= fctap0
         fctce1(12)= fctap1
         fctce2(12)= fctap2
         fctce3(12)= fctap3

         fctcf1(13)=-0.200621e+03
         fctcf2(13)=-0.921209e+00
         fctce0(13)= fctap0
         fctce1(13)= fctap1
         fctce2(13)= fctap2
         fctce3(13)= fctap3

         fctcf1(14)= 0.420018e+01
         fctcf2(14)=-0.634365e+00
         fctce0(14)= fctap0
         fctce1(14)= fctap1
         fctce2(14)= fctap2
         fctce3(14)= fctap3

         fctcf1(15)= 0.161564e+03
         fctcf2(15)=-0.980790e+00
         fctce0(15)= fctap0
         fctce1(15)= fctap1
         fctce2(15)= fctap2
         fctce3(15)= fctap3

         fctcf1(16)= 0.887817e+01
         fctcf2(16)=-0.867689e+00
         fctce0(16)= fctap0
         fctce1(16)= fctap1
         fctce2(16)= fctap2
         fctce3(16)= fctap3

         fctcf1(17)= 0.609835e+03
         fctcf2(17)=-0.116869e+01
         fctce0(17)= fctap0
         fctce1(17)= fctap1
         fctce2(17)= fctap2
         fctce3(17)= fctap3

         fctcf1(18)= 0.609835e+03
         fctcf2(18)=-0.116869e+01
         fctce0(18)= fctap0
         fctce1(18)= fctap1
         fctce2(18)= fctap2
         fctce3(18)= fctap3

         fctcf1(19)= 0.609835e+03
         fctcf2(19)=-0.116869e+01
         fctce0(19)= fctap0
         fctce1(19)= fctap1
         fctce2(19)= fctap2
         fctce3(19)= fctap3

         fctvos    = zero

! Define the parameters for calculating beta.

         fctch1(1 )=-0.483656e+03
         fctch2(1 )=-0.204915e+01
         fctcg0(1 )= fctbt0
         fctcg1(1 )= fctbt1
         fctcg2(1 )= fctbt2
         fctcg3(1 )= fctbt3

         fctch1(2 )= 0.549650e+02
         fctch2(2 )=-0.170556e+01
         fctcg0(2 )= fctbt0
         fctcg1(2 )= fctbt1
         fctcg2(2 )= fctbt2
         fctcg3(2 )= fctbt3

         fctch1(3 )= 0.549650e+02
         fctch2(3 )=-0.170556e+01
         fctcg0(3 )= fctbt0
         fctcg1(3 )= fctbt1
         fctcg2(3 )= fctbt2
         fctcg3(3 )= fctbt3

         fctch1(4 )= 0.549650e+02
         fctch2(4 )=-0.170556e+01
         fctcg0(4 )= fctbt0
         fctcg1(4 )= fctbt1
         fctcg2(4 )= fctbt2
         fctcg3(4 )= fctbt3

         fctch1(5 )=-0.106521e+03
         fctch2(5 )=-0.132303e+01
         fctcg0(5 )= fctbt0
         fctcg1(5 )= fctbt1
         fctcg2(5 )= fctbt2
         fctcg3(5 )= fctbt3

         fctch1(6 )=-0.151213e+03
         fctch2(6 )=-0.134286e+01
         fctcg0(6 )= fctbt0
         fctcg1(6 )= fctbt1
         fctcg2(6 )= fctbt2
         fctcg3(6 )= fctbt3

         fctch1(7 )=-0.276787e+02
         fctch2(7 )=-0.792865e+00
         fctcg0(7 )= fctbt0
         fctcg1(7 )= fctbt1
         fctcg2(7 )= fctbt2
         fctcg3(7 )= fctbt3

         fctch1(8 )=-0.296518e+03
         fctch2(8 )=-0.913736e+00
         fctcg0(8 )= fctbt0
         fctcg1(8 )= fctbt1
         fctcg2(8 )= fctbt2
         fctcg3(8 )= fctbt3

         fctch1(9 )=-0.356797e+03
         fctch2(9 )=-0.727135e+00
         fctcg0(9 )= fctbt0
         fctcg1(9 )= fctbt1
         fctcg2(9 )= fctbt2
         fctcg3(9 )= fctbt3

         fctch1(10)=-0.518968e+03
         fctch2(10)=-0.387026e+00
         fctcg0(10)= fctbt0
         fctcg1(10)= fctbt1
         fctcg2(10)= fctbt2
         fctcg3(10)= fctbt3

         fctch1(11)=-0.489213e+03
         fctch2(11)=-0.445345e+00
         fctcg0(11)= fctbt0
         fctcg1(11)= fctbt1
         fctcg2(11)= fctbt2
         fctcg3(11)= fctbt3

         fctch1(12)= 0.276081e+03
         fctch2(12)=-0.980170e+00
         fctcg0(12)= fctbt0
         fctcg1(12)= fctbt1
         fctcg2(12)= fctbt2
         fctcg3(12)= fctbt3

         fctch1(13)=-0.200621e+03
         fctch2(13)=-0.921209e+00
         fctcg0(13)= fctbt0
         fctcg1(13)= fctbt1
         fctcg2(13)= fctbt2
         fctcg3(13)= fctbt3

         fctch1(14)= 0.420018e+01
         fctch2(14)=-0.634365e+00
         fctcg0(14)= fctbt0
         fctcg1(14)= fctbt1
         fctcg2(14)= fctbt2
         fctcg3(14)= fctbt3

         fctch1(15)= 0.161564e+03
         fctch2(15)=-0.980790e+00
         fctcg0(15)= fctbt0
         fctcg1(15)= fctbt1
         fctcg2(15)= fctbt2
         fctcg3(15)= fctbt3

         fctch1(16)= 0.887817e+01
         fctch2(16)=-0.867689e+00
         fctcg0(16)= fctbt0
         fctcg1(16)= fctbt1
         fctcg2(16)= fctbt2
         fctcg3(16)= fctbt3

         fctch1(17)= 0.609835e+03
         fctch2(17)=-0.116869e+01
         fctcg0(17)= fctbt0
         fctcg1(17)= fctbt1
         fctcg2(17)= fctbt2
         fctcg3(17)= fctbt3

         fctch1(18)= 0.609835e+03
         fctch2(18)=-0.116869e+01
         fctcg0(18)= fctbt0
         fctcg1(18)= fctbt1
         fctcg2(18)= fctbt2
         fctcg3(18)= fctbt3

         fctch1(19)= 0.609835e+03
         fctch2(19)=-0.116869e+01
         fctcg0(19)= fctbt0
         fctcg1(19)= fctbt1
         fctcg2(19)= fctbt2
         fctcg3(19)= fctbt3

         fctwos    = zero

! Define the initial value for the measure of symmetry.

         fctbin    = zero

! Define the factor for the exponential function in the Generalized Born
! formula.

         fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

         fctqun(1 )= one
         fctqun(2 )= one
         fctqun(3 )= one
         fctqun(4 )= one
         fctqun(5 )= one
         fctqun(6 )= one
         fctqun(7 )= one
         fctqun(8 )= one
         fctqun(9 )= one
         fctqun(10)= one
         fctqun(11)= one
         fctqun(12)= one
         fctqun(13)= one
         fctqun(14)= one
         fctqun(15)= one
         fctqun(16)= one
         fctqun(17)= one
         fctqun(18)= one
         fctqun(19)= one

! Define the scaling factors for the charges and for the fctnps factors.
! they only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

         fctscf(1 )= one
         fctscf(2 )= one
         fctscf(3 )= one
         fctscf(4 )= one
         fctscf(5 )= one
         fctscf(6 )= one
         fctscf(7 )= one
         fctscf(8 )= one
         fctscf(9 )= one
         fctscf(10)= one
         fctscf(11)= one
         fctscf(12)= one
         fctscf(13)= one
         fctscf(14)= one
         fctscf(15)= one
         fctscf(16)= one
         fctscf(17)= one
         fctscf(18)= one
         fctscf(19)= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

         fctnps(1 )= zero
         fctnps(2 )= zero
         fctnps(3 )= zero
         fctnps(4 )= zero
         fctnps(5 )= zero
         fctnps(6 )= zero
         fctnps(7 )= zero
         fctnps(8 )= zero
         fctnps(9 )= zero
         fctnps(10)= zero
         fctnps(11)= zero
         fctnps(12)= zero
         fctnps(13)= zero
         fctnps(14)= zero
         fctnps(15)= zero
         fctnps(16)= zero
         fctnps(17)= zero
         fctnps(18)= zero
         fctnps(19)= zero

      endif

   if ((fctcps == 22    )  .and. &
       (fcteps == two   )  .and. &
       (fctfps == 3     )) then

! This FACTS parameter set was derived with boundary conditions at zero
! and positive infinity.

      fctcho    = .true.

! Define some constants related to the dielectric medium.

      fctiem    = fctcel/two
      fcties    = fctcel/78.5d0
      fcttau    = fctiem-fcties
      fctpco    = four/(fcttau*fcttau)

! Define the number of native van der waals radii.

      fctnnv    = 19

! Define the number of actual van der waals radii.

      fctnav    = 19

! Define the native van der waals radii.

      fctrvw(1 )= 0.2245d0
      fctrvw(2 )= 0.4500d0
      fctrvw(3 )= 0.7000d0
      fctrvw(4 )= 0.9000d0
      fctrvw(5 )= 1.3200d0
      fctrvw(6 )= 1.3582d0
      fctrvw(7 )= 1.4680d0
      fctrvw(8 )= 1.7000d0
      fctrvw(9 )= 1.7700d0
      fctrvw(10)= 1.8000d0
      fctrvw(11)= 1.8500d0
      fctrvw(12)= 1.9750d0
      fctrvw(13)= 1.9924d0
      fctrvw(14)= 2.0000d0
      fctrvw(15)= 2.0600d0
      fctrvw(16)= 2.1750d0
      fctrvw(17)= 2.2350d0
      fctrvw(18)= 2.2750d0
      fctrvw(19)= 2.3650d0

! Define the cutoffs for the FACTS self and interaction energy pair
! lists.

      fctcsl    =12.0d0
      fctcil    =14.0d0

! Define the cutoffs for the FACTS self and interaction energy
! calculations.

      fctcsc(1 )= 0.706222e+01
      fctcsc(2 )= 0.780050e+01
      fctcsc(3 )= 0.600000e+01
      fctcsc(4 )= 0.682946e+01
      fctcsc(5 )= 0.810049e+01
      fctcsc(6 )= 0.816195e+01
      fctcsc(7 )= 0.751745e+01
      fctcsc(8 )= 0.841095e+01
      fctcsc(9 )= 0.835799e+01
      fctcsc(10)= 0.834481e+01
      fctcsc(11)= 0.846507e+01
      fctcsc(12)= 0.925994e+01
      fctcsc(13)= 0.921611e+01
      fctcsc(14)= 0.878033e+01
      fctcsc(15)= 0.998814e+01
      fctcsc(16)= 0.890286e+01
      fctcsc(17)= 0.600000e+01
      fctcsc(18)= 0.957919e+01
      fctcsc(19)= 0.600000e+01
      fctcic    =12.0d0

! Define the parameters for calculating atomic solvation energies.

      fctcb1(1 )=-0.275971e+03
      fctcb2(1 )=-0.200203e+01
      fctca0(1 )=-0.751192e+02
      fctca1(1 )= 0.639494e+02
      fctca2(1 )= 0.474256e-02
      fctca3(1 )=-0.941027e+02

      fctcb1(2 )= 0.191552e+03
      fctcb2(2 )=-0.138179e+01
      fctca0(2 )=-0.110797e+03
      fctca1(2 )= 0.110795e+03
      fctca2(2 )= 0.793172e-02
      fctca3(2 )= 0.183897e+03

      fctcb1(3 )= 0.191552e+03
      fctcb2(3 )=-0.138179e+01
      fctca0(3 )=-0.110797e+03
      fctca1(3 )= 0.110795e+03
      fctca2(3 )= 0.793172e-02
      fctca3(3 )= 0.183897e+03

      fctcb1(4 )= 0.191552e+03
      fctcb2(4 )=-0.138179e+01
      fctca0(4 )=-0.110797e+03
      fctca1(4 )= 0.110795e+03
      fctca2(4 )= 0.793172e-02
      fctca3(4 )= 0.183897e+03

      fctcb1(5 )= 0.323610e+03
      fctcb2(5 )=-0.142148e+01
      fctca0(5 )=-0.115058e+03
      fctca1(5 )= 0.115056e+03
      fctca2(5 )= 0.252914e-02
      fctca3(5 )= 0.517585e+02

      fctcb1(6 )= 0.297152e+03
      fctcb2(6 )=-0.138207e+01
      fctca0(6 )=-0.920407e+02
      fctca1(6 )= 0.920387e+02
      fctca2(6 )= 0.261762e-02
      fctca3(6 )= 0.231737e+03

      fctcb1(7 )= 0.183798e+03
      fctcb2(7 )=-0.101401e+01
      fctca0(7 )=-0.663349e+02
      fctca1(7 )= 0.663329e+02
      fctca2(7 )= 0.385091e-02
      fctca3(7 )= 0.413215e+03

      fctcb1(8 )= 0.254877e+03
      fctcb2(8 )=-0.119338e+01
      fctca0(8 )=-0.658550e+02
      fctca1(8 )= 0.658530e+02
      fctca2(8 )= 0.217134e-02
      fctca3(8 )= 0.441010e+03

      fctcb1(9 )= 0.188084e+03
      fctcb2(9 )=-0.989163e+00
      fctca0(9 )=-0.610204e+02
      fctca1(9 )= 0.610184e+02
      fctca2(9 )= 0.219217e-02
      fctca3(9 )= 0.498842e+03

      fctcb1(10)= 0.199228e+03
      fctcb2(10)=-0.123164e+01
      fctca0(10)=-0.119719e+03
      fctca1(10)= 0.119717e+03
      fctca2(10)= 0.149176e-02
      fctca3(10)=-0.341234e+03

      fctcb1(11)= 0.168275e+03
      fctcb2(11)=-0.104456e+01
      fctca0(11)=-0.719168e+02
      fctca1(11)= 0.719148e+02
      fctca2(11)= 0.166129e-02
      fctca3(11)= 0.264377e+03

      fctcb1(12)= 0.614438e+03
      fctcb2(12)=-0.125042e+01
      fctca0(12)=-0.473819e+02
      fctca1(12)= 0.473799e+02
      fctca2(12)= 0.205712e-02
      fctca3(12)= 0.900995e+03

      fctcb1(13)= 0.491726e+03
      fctcb2(13)=-0.140993e+01
      fctca0(13)=-0.577249e+02
      fctca1(13)= 0.577229e+02
      fctca2(13)= 0.156399e-02
      fctca3(13)= 0.552216e+03

      fctcb1(14)= 0.450903e+03
      fctcb2(14)=-0.107514e+01
      fctca0(14)=-0.550524e+02
      fctca1(14)= 0.550504e+02
      fctca2(14)= 0.167531e-02
      fctca3(14)= 0.608224e+03

      fctcb1(15)= 0.640084e+03
      fctcb2(15)=-0.122445e+01
      fctca0(15)=-0.451333e+02
      fctca1(15)= 0.451313e+02
      fctca2(15)= 0.167454e-02
      fctca3(15)= 0.113603e+04

      fctcb1(16)= 0.469141e+03
      fctcb2(16)=-0.128110e+01
      fctca0(16)=-0.451129e+02
      fctca1(16)= 0.451109e+02
      fctca2(16)= 0.196058e-02
      fctca3(16)= 0.789179e+03

      fctcb1(17)= 0.102722e+04
      fctcb2(17)=-0.152321e+01
      fctca0(17)=-0.429235e+02
      fctca1(17)= 0.429215e+02
      fctca2(17)= 0.155246e-02
      fctca3(17)= 0.101446e+04

      fctcb1(18)= 0.102722e+04
      fctcb2(18)=-0.152321e+01
      fctca0(18)=-0.429235e+02
      fctca1(18)= 0.429215e+02
      fctca2(18)= 0.155246e-02
      fctca3(18)= 0.101446e+04

      fctcb1(19)= 0.102722e+04
      fctcb2(19)=-0.152321e+01
      fctca0(19)=-0.429235e+02
      fctca1(19)= 0.429215e+02
      fctca2(19)= 0.155246e-02
      fctca3(19)= 0.101446e+04

      fctqos    = zero

! Define the parameters for calculating atomic solvent accessible
! surface areas.

      fctcd1(1 )=-0.597528e+04
      fctcd2(1 )=-0.132626e+02
      fctcc0(1 )= 0.331627e+02
      fctcc1(1 )=-0.331607e+02
      fctcc2(1 )= 0.268702e-01
      fctcc3(1 )=-0.945280e+04

      fctcd1(2 )=-0.952570e+04
      fctcd2(2 )=-0.892289e+01
      fctcc0(2 )= 0.875739e+01
      fctcc1(2 )=-0.875539e+01
      fctcc2(2 )= 0.150488e-02
      fctcc3(2 )=-0.731427e+04

      fctcd1(3 )=-0.952570e+04
      fctcd2(3 )=-0.892289e+01
      fctcc0(3 )= 0.875739e+01
      fctcc1(3 )=-0.875539e+01
      fctcc2(3 )= 0.150488e-02
      fctcc3(3 )=-0.731427e+04

      fctcd1(4 )=-0.952570e+04
      fctcd2(4 )=-0.892289e+01
      fctcc0(4 )= 0.875739e+01
      fctcc1(4 )=-0.875539e+01
      fctcc2(4 )= 0.150488e-02
      fctcc3(4 )=-0.731427e+04

      fctcd1(5 )=-0.192740e+03
      fctcd2(5 )=-0.176685e+01
      fctcc0(5 )= 0.100230e+03
      fctcc1(5 )=-0.100228e+03
      fctcc2(5 )= 0.630842e-02
      fctcc3(5 )=-0.578701e+03

      fctcd1(6 )=-0.676447e+03
      fctcd2(6 )=-0.222803e+01
      fctcc0(6 )= 0.163762e+03
      fctcc1(6 )=-0.163760e+03
      fctcc2(6 )= 0.345894e-02
      fctcc3(6 )=-0.133154e+04

      fctcd1(7 )=-0.281663e+04
      fctcd2(7 )=-0.255432e+01
      fctcc0(7 )= 0.492274e+02
      fctcc1(7 )=-0.492254e+02
      fctcc2(7 )= 0.224746e-02
      fctcc3(7 )=-0.218549e+04

      fctcd1(8 )=-0.193271e+04
      fctcd2(8 )=-0.164887e+01
      fctcc0(8 )= 0.650194e+02
      fctcc1(8 )=-0.650174e+02
      fctcc2(8 )= 0.248175e-02
      fctcc3(8 )=-0.135331e+04

      fctcd1(9 )=-0.311522e+04
      fctcd2(9 )=-0.124537e+01
      fctcc0(9 )= 0.735675e+02
      fctcc1(9 )=-0.735655e+02
      fctcc2(9 )= 0.169442e-02
      fctcc3(9 )=-0.214416e+04

      fctcd1(10)=-0.222684e+05
      fctcd2(10)=-0.810128e+01
      fctcc0(10)= 0.315852e+02
      fctcc1(10)=-0.313963e+02
      fctcc2(10)= 0.579209e-03
      fctcc3(10)=-0.139485e+05

      fctcd1(11)=-0.193136e+04
      fctcd2(11)=-0.155601e+01
      fctcc0(11)= 0.746558e+02
      fctcc1(11)=-0.746538e+02
      fctcc2(11)= 0.270264e-02
      fctcc3(11)=-0.128202e+04

      fctcd1(12)= 0.288801e+03
      fctcd2(12)=-0.229942e+01
      fctcc0(12)= 0.174431e+03
      fctcc1(12)=-0.174344e+03
      fctcc2(12)= 0.455283e-02
      fctcc3(12)=-0.404115e+03

      fctcd1(13)=-0.421160e+04
      fctcd2(13)=-0.459151e+00
      fctcc0(13)= 0.346466e+02
      fctcc1(13)=-0.346446e+02
      fctcc2(13)= 0.138939e-02
      fctcc3(13)=-0.209693e+04

      fctcd1(14)=-0.811683e+04
      fctcd2(14)=-0.234357e+01
      fctcc0(14)= 0.167665e+03
      fctcc1(14)=-0.167472e+03
      fctcc2(14)= 0.113712e-02
      fctcc3(14)=-0.675676e+04

      fctcd1(15)=-0.627618e+04
      fctcd2(15)=-0.153047e+01
      fctcc0(15)= 0.104380e+03
      fctcc1(15)=-0.104378e+03
      fctcc2(15)= 0.732719e-03
      fctcc3(15)=-0.522811e+04

      fctcd1(16)=-0.178801e+05
      fctcd2(16)=-0.310708e+01
      fctcc0(16)= 0.172766e+03
      fctcc1(16)=-0.172687e+03
      fctcc2(16)= 0.348553e-03
      fctcc3(16)=-0.151252e+05

      fctcd1(17)=-0.633935e+03
      fctcd2(17)=-0.200029e+01
      fctcc0(17)= 0.178299e+03
      fctcc1(17)=-0.178297e+03
      fctcc2(17)= 0.292989e-02
      fctcc3(17)=-0.106603e+04

      fctcd1(18)=-0.633935e+03
      fctcd2(18)=-0.200029e+01
      fctcc0(18)= 0.178299e+03
      fctcc1(18)=-0.178297e+03
      fctcc2(18)= 0.292989e-02
      fctcc3(18)=-0.106603e+04

      fctcd1(19)=-0.633935e+03
      fctcd2(19)=-0.200029e+01
      fctcc0(19)= 0.178299e+03
      fctcc1(19)=-0.178297e+03
      fctcc2(19)= 0.292989e-02
      fctcc3(19)=-0.106603e+04

      fctuos    = zero

! Define the parameters for calculating alpha.

      fctcf1(1 )=-0.275971e+03
      fctcf2(1 )=-0.200203e+01
      fctce0(1 )= fctap0
      fctce1(1 )= fctap1
      fctce2(1 )= fctap2
      fctce3(1 )= fctap3

      fctcf1(2 )= 0.191552e+03
      fctcf2(2 )=-0.138179e+01
      fctce0(2 )= fctap0
      fctce1(2 )= fctap1
      fctce2(2 )= fctap2
      fctce3(2 )= fctap3

      fctcf1(3 )= 0.191552e+03
      fctcf2(3 )=-0.138179e+01
      fctce0(3 )= fctap0
      fctce1(3 )= fctap1
      fctce2(3 )= fctap2
      fctce3(3 )= fctap3

      fctcf1(4 )= 0.191552e+03
      fctcf2(4 )=-0.138179e+01
      fctce0(4 )= fctap0
      fctce1(4 )= fctap1
      fctce2(4 )= fctap2
      fctce3(4 )= fctap3

      fctcf1(5 )= 0.323610e+03
      fctcf2(5 )=-0.142148e+01
      fctce0(5 )= fctap0
      fctce1(5 )= fctap1
      fctce2(5 )= fctap2
      fctce3(5 )= fctap3

      fctcf1(6 )= 0.297152e+03
      fctcf2(6 )=-0.138207e+01
      fctce0(6 )= fctap0
      fctce1(6 )= fctap1
      fctce2(6 )= fctap2
      fctce3(6 )= fctap3

      fctcf1(7 )= 0.183798e+03
      fctcf2(7 )=-0.101401e+01
      fctce0(7 )= fctap0
      fctce1(7 )= fctap1
      fctce2(7 )= fctap2
      fctce3(7 )= fctap3

      fctcf1(8 )= 0.254877e+03
      fctcf2(8 )=-0.119338e+01
      fctce0(8 )= fctap0
      fctce1(8 )= fctap1
      fctce2(8 )= fctap2
      fctce3(8 )= fctap3

      fctcf1(9 )= 0.188084e+03
      fctcf2(9 )=-0.989163e+00
      fctce0(9 )= fctap0
      fctce1(9 )= fctap1
      fctce2(9 )= fctap2
      fctce3(9 )= fctap3

      fctcf1(10)= 0.199228e+03
      fctcf2(10)=-0.123164e+01
      fctce0(10)= fctap0
      fctce1(10)= fctap1
      fctce2(10)= fctap2
      fctce3(10)= fctap3

      fctcf1(11)= 0.168275e+03
      fctcf2(11)=-0.104456e+01
      fctce0(11)= fctap0
      fctce1(11)= fctap1
      fctce2(11)= fctap2
      fctce3(11)= fctap3

      fctcf1(12)= 0.614438e+03
      fctcf2(12)=-0.125042e+01
      fctce0(12)= fctap0
      fctce1(12)= fctap1
      fctce2(12)= fctap2
      fctce3(12)= fctap3

      fctcf1(13)= 0.491726e+03
      fctcf2(13)=-0.140993e+01
      fctce0(13)= fctap0
      fctce1(13)= fctap1
      fctce2(13)= fctap2
      fctce3(13)= fctap3

      fctcf1(14)= 0.450903e+03
      fctcf2(14)=-0.107514e+01
      fctce0(14)= fctap0
      fctce1(14)= fctap1
      fctce2(14)= fctap2
      fctce3(14)= fctap3

      fctcf1(15)= 0.640084e+03
      fctcf2(15)=-0.122445e+01
      fctce0(15)= fctap0
      fctce1(15)= fctap1
      fctce2(15)= fctap2
      fctce3(15)= fctap3

      fctcf1(16)= 0.469141e+03
      fctcf2(16)=-0.128110e+01
      fctce0(16)= fctap0
      fctce1(16)= fctap1
      fctce2(16)= fctap2
      fctce3(16)= fctap3

      fctcf1(17)= 0.102722e+04
      fctcf2(17)=-0.152321e+01
      fctce0(17)= fctap0
      fctce1(17)= fctap1
      fctce2(17)= fctap2
      fctce3(17)= fctap3

      fctcf1(18)= 0.102722e+04
      fctcf2(18)=-0.152321e+01
      fctce0(18)= fctap0
      fctce1(18)= fctap1
      fctce2(18)= fctap2
      fctce3(18)= fctap3

      fctcf1(19)= 0.102722e+04
      fctcf2(19)=-0.152321e+01
      fctce0(19)= fctap0
      fctce1(19)= fctap1
      fctce2(19)= fctap2
      fctce3(19)= fctap3

      fctvos    = zero

! Define the parameters for calculating beta.

      fctch1(1 )=-0.275971e+03
      fctch2(1 )=-0.200203e+01
      fctcg0(1 )= fctbt0
      fctcg1(1 )= fctbt1
      fctcg2(1 )= fctbt2
      fctcg3(1 )= fctbt3

      fctch1(2 )= 0.191552e+03
      fctch2(2 )=-0.138179e+01
      fctcg0(2 )= fctbt0
      fctcg1(2 )= fctbt1
      fctcg2(2 )= fctbt2
      fctcg3(2 )= fctbt3

      fctch1(3 )= 0.191552e+03
      fctch2(3 )=-0.138179e+01
      fctcg0(3 )= fctbt0
      fctcg1(3 )= fctbt1
      fctcg2(3 )= fctbt2
      fctcg3(3 )= fctbt3

      fctch1(4 )= 0.191552e+03
      fctch2(4 )=-0.138179e+01
      fctcg0(4 )= fctbt0
      fctcg1(4 )= fctbt1
      fctcg2(4 )= fctbt2
      fctcg3(4 )= fctbt3

      fctch1(5 )= 0.323610e+03
      fctch2(5 )=-0.142148e+01
      fctcg0(5 )= fctbt0
      fctcg1(5 )= fctbt1
      fctcg2(5 )= fctbt2
      fctcg3(5 )= fctbt3

      fctch1(6 )= 0.297152e+03
      fctch2(6 )=-0.138207e+01
      fctcg0(6 )= fctbt0
      fctcg1(6 )= fctbt1
      fctcg2(6 )= fctbt2
      fctcg3(6 )= fctbt3

      fctch1(7 )= 0.183798e+03
      fctch2(7 )=-0.101401e+01
      fctcg0(7 )= fctbt0
      fctcg1(7 )= fctbt1
      fctcg2(7 )= fctbt2
      fctcg3(7 )= fctbt3

      fctch1(8 )= 0.254877e+03
      fctch2(8 )=-0.119338e+01
      fctcg0(8 )= fctbt0
      fctcg1(8 )= fctbt1
      fctcg2(8 )= fctbt2
      fctcg3(8 )= fctbt3

      fctch1(9 )= 0.188084e+03
      fctch2(9 )=-0.989163e+00
      fctcg0(9 )= fctbt0
      fctcg1(9 )= fctbt1
      fctcg2(9 )= fctbt2
      fctcg3(9 )= fctbt3

      fctch1(10)= 0.199228e+03
      fctch2(10)=-0.123164e+01
      fctcg0(10)= fctbt0
      fctcg1(10)= fctbt1
      fctcg2(10)= fctbt2
      fctcg3(10)= fctbt3

      fctch1(11)= 0.168275e+03
      fctch2(11)=-0.104456e+01
      fctcg0(11)= fctbt0
      fctcg1(11)= fctbt1
      fctcg2(11)= fctbt2
      fctcg3(11)= fctbt3

      fctch1(12)= 0.614438e+03
      fctch2(12)=-0.125042e+01
      fctcg0(12)= fctbt0
      fctcg1(12)= fctbt1
      fctcg2(12)= fctbt2
      fctcg3(12)= fctbt3

      fctch1(13)= 0.491726e+03
      fctch2(13)=-0.140993e+01
      fctcg0(13)= fctbt0
      fctcg1(13)= fctbt1
      fctcg2(13)= fctbt2
      fctcg3(13)= fctbt3

      fctch1(14)= 0.450903e+03
      fctch2(14)=-0.107514e+01
      fctcg0(14)= fctbt0
      fctcg1(14)= fctbt1
      fctcg2(14)= fctbt2
      fctcg3(14)= fctbt3

      fctch1(15)= 0.640084e+03
      fctch2(15)=-0.122445e+01
      fctcg0(15)= fctbt0
      fctcg1(15)= fctbt1
      fctcg2(15)= fctbt2
      fctcg3(15)= fctbt3

      fctch1(16)= 0.469141e+03
      fctch2(16)=-0.128110e+01
      fctcg0(16)= fctbt0
      fctcg1(16)= fctbt1
      fctcg2(16)= fctbt2
      fctcg3(16)= fctbt3

      fctch1(17)= 0.102722e+04
      fctch2(17)=-0.152321e+01
      fctcg0(17)= fctbt0
      fctcg1(17)= fctbt1
      fctcg2(17)= fctbt2
      fctcg3(17)= fctbt3

      fctch1(18)= 0.102722e+04
      fctch2(18)=-0.152321e+01
      fctcg0(18)= fctbt0
      fctcg1(18)= fctbt1
      fctcg2(18)= fctbt2
      fctcg3(18)= fctbt3

      fctch1(19)= 0.102722e+04
      fctch2(19)=-0.152321e+01
      fctcg0(19)= fctbt0
      fctcg1(19)= fctbt1
      fctcg2(19)= fctbt2
      fctcg3(19)= fctbt3

      fctwos    = zero

! Define the initial value for the measure of symmetry.

      fctbin    = one

! Define the factor for the exponential function in the Generalized Born
! formula.

      fctkps    = four

! Define the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

      fctqun(1 )= one
      fctqun(2 )= one
      fctqun(3 )= one
      fctqun(4 )= one
      fctqun(5 )= one
      fctqun(6 )= one
      fctqun(7 )= one
      fctqun(8 )= one
      fctqun(9 )= one
      fctqun(10)= one
      fctqun(11)= one
      fctqun(12)= one
      fctqun(13)= one
      fctqun(14)= one
      fctqun(15)= one
      fctqun(16)= one
      fctqun(17)= one
      fctqun(18)= one
      fctqun(19)= one

! Define the scaling factors for the charges and for the fctnps factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

      fctscf(1 )= one
      fctscf(2 )= one
      fctscf(3 )= one
      fctscf(4 )= one
      fctscf(5 )= one
      fctscf(6 )= one
      fctscf(7 )= one
      fctscf(8 )= one
      fctscf(9 )= one
      fctscf(10)= one
      fctscf(11)= one
      fctscf(12)= one
      fctscf(13)= one
      fctscf(14)= one
      fctscf(15)= one
      fctscf(16)= one
      fctscf(17)= one
      fctscf(18)= one
      fctscf(19)= one

! Define the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

      fctnps(1 )= zero
      fctnps(2 )= zero
      fctnps(3 )= zero
      fctnps(4 )= zero
      fctnps(5 )= zero
      fctnps(6 )= zero
      fctnps(7 )= zero
      fctnps(8 )= zero
      fctnps(9 )= zero
      fctnps(10)= zero
      fctnps(11)= zero
      fctnps(12)= zero
      fctnps(13)= zero
      fctnps(14)= zero
      fctnps(15)= zero
      fctnps(16)= zero
      fctnps(17)= zero
      fctnps(18)= zero
      fctnps(19)= zero

   endif

   if (.not. fctcho) then
      call wrndie(-4,'<FCTINI>','No valid FACTS parameter set.')
   endif

! Process the user input.

   if (fctcps == 19) then

! Obtain the Cutoffs for the FACTS Self and Interaction Energy Pair
! Lists.

      fctcsl    =gtrmf(comlyn,comlen,'TCSL',fctcsl    )
      fctcil    =gtrmf(comlyn,comlen,'TCIL',fctcil    )

! Obtain the cutoffs for the FACTS self and interaction energy
! calculations.

      fctcsc(1 )=gtrmf(comlyn,comlen,'SC01',fctcsc(1 ))
      fctcsc(2 )=gtrmf(comlyn,comlen,'SC02',fctcsc(2 ))
      fctcsc(3 )=gtrmf(comlyn,comlen,'SC03',fctcsc(3 ))
      fctcsc(4 )=gtrmf(comlyn,comlen,'SC04',fctcsc(4 ))
      fctcsc(5 )=gtrmf(comlyn,comlen,'SC05',fctcsc(5 ))
      fctcsc(6 )=gtrmf(comlyn,comlen,'SC06',fctcsc(6 ))
      fctcsc(7 )=gtrmf(comlyn,comlen,'SC07',fctcsc(7 ))
      fctcic    =gtrmf(comlyn,comlen,'TCIC',fctcic    )

! Obtain the parameters for calculating atomic solvation energies.

      fctcb1(1 )=gtrmf(comlyn,comlen,'B101',fctcb1(1 ))
      fctcb2(1 )=gtrmf(comlyn,comlen,'B201',fctcb2(1 ))
      fctca0(1 )=gtrmf(comlyn,comlen,'A001',fctca0(1 ))
      fctca1(1 )=gtrmf(comlyn,comlen,'A101',fctca1(1 ))
      fctca2(1 )=gtrmf(comlyn,comlen,'A201',fctca2(1 ))
      fctca3(1 )=gtrmf(comlyn,comlen,'A301',fctca3(1 ))

      fctcb1(2 )=gtrmf(comlyn,comlen,'B102',fctcb1(2 ))
      fctcb2(2 )=gtrmf(comlyn,comlen,'B202',fctcb2(2 ))
      fctca0(2 )=gtrmf(comlyn,comlen,'A002',fctca0(2 ))
      fctca1(2 )=gtrmf(comlyn,comlen,'A102',fctca1(2 ))
      fctca2(2 )=gtrmf(comlyn,comlen,'A202',fctca2(2 ))
      fctca3(2 )=gtrmf(comlyn,comlen,'A302',fctca3(2 ))

      fctcb1(3 )=gtrmf(comlyn,comlen,'B103',fctcb1(3 ))
      fctcb2(3 )=gtrmf(comlyn,comlen,'B203',fctcb2(3 ))
      fctca0(3 )=gtrmf(comlyn,comlen,'A003',fctca0(3 ))
      fctca1(3 )=gtrmf(comlyn,comlen,'A103',fctca1(3 ))
      fctca2(3 )=gtrmf(comlyn,comlen,'A203',fctca2(3 ))
      fctca3(3 )=gtrmf(comlyn,comlen,'A303',fctca3(3 ))

      fctcb1(4 )=gtrmf(comlyn,comlen,'B104',fctcb1(4 ))
      fctcb2(4 )=gtrmf(comlyn,comlen,'B204',fctcb2(4 ))
      fctca0(4 )=gtrmf(comlyn,comlen,'A004',fctca0(4 ))
      fctca1(4 )=gtrmf(comlyn,comlen,'A104',fctca1(4 ))
      fctca2(4 )=gtrmf(comlyn,comlen,'A204',fctca2(4 ))
      fctca3(4 )=gtrmf(comlyn,comlen,'A304',fctca3(4 ))

      fctcb1(5 )=gtrmf(comlyn,comlen,'B105',fctcb1(5 ))
      fctcb2(5 )=gtrmf(comlyn,comlen,'B205',fctcb2(5 ))
      fctca0(5 )=gtrmf(comlyn,comlen,'A005',fctca0(5 ))
      fctca1(5 )=gtrmf(comlyn,comlen,'A105',fctca1(5 ))
      fctca2(5 )=gtrmf(comlyn,comlen,'A205',fctca2(5 ))
      fctca3(5 )=gtrmf(comlyn,comlen,'A305',fctca3(5 ))

      fctcb1(6 )=gtrmf(comlyn,comlen,'B106',fctcb1(6 ))
      fctcb2(6 )=gtrmf(comlyn,comlen,'B206',fctcb2(6 ))
      fctca0(6 )=gtrmf(comlyn,comlen,'A006',fctca0(6 ))
      fctca1(6 )=gtrmf(comlyn,comlen,'A106',fctca1(6 ))
      fctca2(6 )=gtrmf(comlyn,comlen,'A206',fctca2(6 ))
      fctca3(6 )=gtrmf(comlyn,comlen,'A306',fctca3(6 ))

      fctcb1(7 )=gtrmf(comlyn,comlen,'B107',fctcb1(7 ))
      fctcb2(7 )=gtrmf(comlyn,comlen,'B207',fctcb2(7 ))
      fctca0(7 )=gtrmf(comlyn,comlen,'A007',fctca0(7 ))
      fctca1(7 )=gtrmf(comlyn,comlen,'A107',fctca1(7 ))
      fctca2(7 )=gtrmf(comlyn,comlen,'A207',fctca2(7 ))
      fctca3(7 )=gtrmf(comlyn,comlen,'A307',fctca3(7 ))

      fctqos    =gtrmf(comlyn,comlen,'TQOS',fctqos    )

! Obtain the parameters for calculating atomic solvent accessible
! surface areas.

      fctcd1(1 )=gtrmf(comlyn,comlen,'D101',fctcd1(1 ))
      fctcd2(1 )=gtrmf(comlyn,comlen,'D201',fctcd2(1 ))
      fctcc0(1 )=gtrmf(comlyn,comlen,'C001',fctcc0(1 ))
      fctcc1(1 )=gtrmf(comlyn,comlen,'C101',fctcc1(1 ))
      fctcc2(1 )=gtrmf(comlyn,comlen,'C201',fctcc2(1 ))
      fctcc3(1 )=gtrmf(comlyn,comlen,'C301',fctcc3(1 ))

      fctcd1(2 )=gtrmf(comlyn,comlen,'D102',fctcd1(2 ))
      fctcd2(2 )=gtrmf(comlyn,comlen,'D202',fctcd2(2 ))
      fctcc0(2 )=gtrmf(comlyn,comlen,'C002',fctcc0(2 ))
      fctcc1(2 )=gtrmf(comlyn,comlen,'C102',fctcc1(2 ))
      fctcc2(2 )=gtrmf(comlyn,comlen,'C202',fctcc2(2 ))
      fctcc3(2 )=gtrmf(comlyn,comlen,'C302',fctcc3(2 ))

      fctcd1(3 )=gtrmf(comlyn,comlen,'D103',fctcd1(3 ))
      fctcd2(3 )=gtrmf(comlyn,comlen,'D203',fctcd2(3 ))
      fctcc0(3 )=gtrmf(comlyn,comlen,'C003',fctcc0(3 ))
      fctcc1(3 )=gtrmf(comlyn,comlen,'C103',fctcc1(3 ))
      fctcc2(3 )=gtrmf(comlyn,comlen,'C203',fctcc2(3 ))
      fctcc3(3 )=gtrmf(comlyn,comlen,'C303',fctcc3(3 ))

      fctcd1(4 )=gtrmf(comlyn,comlen,'D104',fctcd1(4 ))
      fctcd2(4 )=gtrmf(comlyn,comlen,'D204',fctcd2(4 ))
      fctcc0(4 )=gtrmf(comlyn,comlen,'C004',fctcc0(4 ))
      fctcc1(4 )=gtrmf(comlyn,comlen,'C104',fctcc1(4 ))
      fctcc2(4 )=gtrmf(comlyn,comlen,'C204',fctcc2(4 ))
      fctcc3(4 )=gtrmf(comlyn,comlen,'C304',fctcc3(4 ))

      fctcd1(5 )=gtrmf(comlyn,comlen,'D105',fctcd1(5 ))
      fctcd2(5 )=gtrmf(comlyn,comlen,'D205',fctcd2(5 ))
      fctcc0(5 )=gtrmf(comlyn,comlen,'C005',fctcc0(5 ))
      fctcc1(5 )=gtrmf(comlyn,comlen,'C105',fctcc1(5 ))
      fctcc2(5 )=gtrmf(comlyn,comlen,'C205',fctcc2(5 ))
      fctcc3(5 )=gtrmf(comlyn,comlen,'C305',fctcc3(5 ))

      fctcd1(6 )=gtrmf(comlyn,comlen,'D106',fctcd1(6 ))
      fctcd2(6 )=gtrmf(comlyn,comlen,'D206',fctcd2(6 ))
      fctcc0(6 )=gtrmf(comlyn,comlen,'C006',fctcc0(6 ))
      fctcc1(6 )=gtrmf(comlyn,comlen,'C106',fctcc1(6 ))
      fctcc2(6 )=gtrmf(comlyn,comlen,'C206',fctcc2(6 ))
      fctcc3(6 )=gtrmf(comlyn,comlen,'C306',fctcc3(6 ))

      fctcd1(7 )=gtrmf(comlyn,comlen,'D107',fctcd1(7 ))
      fctcd2(7 )=gtrmf(comlyn,comlen,'D207',fctcd2(7 ))
      fctcc0(7 )=gtrmf(comlyn,comlen,'C007',fctcc0(7 ))
      fctcc1(7 )=gtrmf(comlyn,comlen,'C107',fctcc1(7 ))
      fctcc2(7 )=gtrmf(comlyn,comlen,'C207',fctcc2(7 ))
      fctcc3(7 )=gtrmf(comlyn,comlen,'C307',fctcc3(7 ))

      fctuos    =gtrmf(comlyn,comlen,'TUOS',fctuos    )

! Obtain the parameters for calculating alpha.

      fctcf1(1 )=gtrmf(comlyn,comlen,'F101',fctcf1(1 ))
      fctcf2(1 )=gtrmf(comlyn,comlen,'F201',fctcf2(1 ))
      fctce0(1 )=gtrmf(comlyn,comlen,'E001',fctce0(1 ))
      fctce1(1 )=gtrmf(comlyn,comlen,'E101',fctce1(1 ))
      fctce2(1 )=gtrmf(comlyn,comlen,'E201',fctce2(1 ))
      fctce3(1 )=gtrmf(comlyn,comlen,'E301',fctce3(1 ))

      fctcf1(2 )=gtrmf(comlyn,comlen,'F102',fctcf1(2 ))
      fctcf2(2 )=gtrmf(comlyn,comlen,'F202',fctcf2(2 ))
      fctce0(2 )=gtrmf(comlyn,comlen,'E002',fctce0(2 ))
      fctce1(2 )=gtrmf(comlyn,comlen,'E102',fctce1(2 ))
      fctce2(2 )=gtrmf(comlyn,comlen,'E202',fctce2(2 ))
      fctce3(2 )=gtrmf(comlyn,comlen,'E302',fctce3(2 ))

      fctcf1(3 )=gtrmf(comlyn,comlen,'F103',fctcf1(3 ))
      fctcf2(3 )=gtrmf(comlyn,comlen,'F203',fctcf2(3 ))
      fctce0(3 )=gtrmf(comlyn,comlen,'E003',fctce0(3 ))
      fctce1(3 )=gtrmf(comlyn,comlen,'E103',fctce1(3 ))
      fctce2(3 )=gtrmf(comlyn,comlen,'E203',fctce2(3 ))
      fctce3(3 )=gtrmf(comlyn,comlen,'E303',fctce3(3 ))

      fctcf1(4 )=gtrmf(comlyn,comlen,'F104',fctcf1(4 ))
      fctcf2(4 )=gtrmf(comlyn,comlen,'F204',fctcf2(4 ))
      fctce0(4 )=gtrmf(comlyn,comlen,'E004',fctce0(4 ))
      fctce1(4 )=gtrmf(comlyn,comlen,'E104',fctce1(4 ))
      fctce2(4 )=gtrmf(comlyn,comlen,'E204',fctce2(4 ))
      fctce3(4 )=gtrmf(comlyn,comlen,'E304',fctce3(4 ))

      fctcf1(5 )=gtrmf(comlyn,comlen,'F105',fctcf1(5 ))
      fctcf2(5 )=gtrmf(comlyn,comlen,'F205',fctcf2(5 ))
      fctce0(5 )=gtrmf(comlyn,comlen,'E005',fctce0(5 ))
      fctce1(5 )=gtrmf(comlyn,comlen,'E105',fctce1(5 ))
      fctce2(5 )=gtrmf(comlyn,comlen,'E205',fctce2(5 ))
      fctce3(5 )=gtrmf(comlyn,comlen,'E305',fctce3(5 ))

      fctcf1(6 )=gtrmf(comlyn,comlen,'F106',fctcf1(6 ))
      fctcf2(6 )=gtrmf(comlyn,comlen,'F206',fctcf2(6 ))
      fctce0(6 )=gtrmf(comlyn,comlen,'E006',fctce0(6 ))
      fctce1(6 )=gtrmf(comlyn,comlen,'E106',fctce1(6 ))
      fctce2(6 )=gtrmf(comlyn,comlen,'E206',fctce2(6 ))
      fctce3(6 )=gtrmf(comlyn,comlen,'E306',fctce3(6 ))

      fctcf1(7 )=gtrmf(comlyn,comlen,'F107',fctcf1(7 ))
      fctcf2(7 )=gtrmf(comlyn,comlen,'F207',fctcf2(7 ))
      fctce0(7 )=gtrmf(comlyn,comlen,'E007',fctce0(7 ))
      fctce1(7 )=gtrmf(comlyn,comlen,'E107',fctce1(7 ))
      fctce2(7 )=gtrmf(comlyn,comlen,'E207',fctce2(7 ))
      fctce3(7 )=gtrmf(comlyn,comlen,'E307',fctce3(7 ))

      fctvos    =gtrmf(comlyn,comlen,'TVOS',fctvos    )

! Obtain the parameters for calculating beta.

      fctch1(1 )=gtrmf(comlyn,comlen,'H101',fctch1(1 ))
      fctch2(1 )=gtrmf(comlyn,comlen,'H201',fctch2(1 ))
      fctcg0(1 )=gtrmf(comlyn,comlen,'G001',fctcg0(1 ))
      fctcg1(1 )=gtrmf(comlyn,comlen,'G101',fctcg1(1 ))
      fctcg2(1 )=gtrmf(comlyn,comlen,'G201',fctcg2(1 ))
      fctcg3(1 )=gtrmf(comlyn,comlen,'G301',fctcg3(1 ))

      fctch1(2 )=gtrmf(comlyn,comlen,'H102',fctch1(2 ))
      fctch2(2 )=gtrmf(comlyn,comlen,'H202',fctch2(2 ))
      fctcg0(2 )=gtrmf(comlyn,comlen,'G002',fctcg0(2 ))
      fctcg1(2 )=gtrmf(comlyn,comlen,'G102',fctcg1(2 ))
      fctcg2(2 )=gtrmf(comlyn,comlen,'G202',fctcg2(2 ))
      fctcg3(2 )=gtrmf(comlyn,comlen,'G302',fctcg3(2 ))

      fctch1(3 )=gtrmf(comlyn,comlen,'H103',fctch1(3 ))
      fctch2(3 )=gtrmf(comlyn,comlen,'H203',fctch2(3 ))
      fctcg0(3 )=gtrmf(comlyn,comlen,'G003',fctcg0(3 ))
      fctcg1(3 )=gtrmf(comlyn,comlen,'G103',fctcg1(3 ))
      fctcg2(3 )=gtrmf(comlyn,comlen,'G203',fctcg2(3 ))
      fctcg3(3 )=gtrmf(comlyn,comlen,'G303',fctcg3(3 ))

      fctch1(4 )=gtrmf(comlyn,comlen,'H104',fctch1(4 ))
      fctch2(4 )=gtrmf(comlyn,comlen,'H204',fctch2(4 ))
      fctcg0(4 )=gtrmf(comlyn,comlen,'G004',fctcg0(4 ))
      fctcg1(4 )=gtrmf(comlyn,comlen,'G104',fctcg1(4 ))
      fctcg2(4 )=gtrmf(comlyn,comlen,'G204',fctcg2(4 ))
      fctcg3(4 )=gtrmf(comlyn,comlen,'G304',fctcg3(4 ))

      fctch1(5 )=gtrmf(comlyn,comlen,'H105',fctch1(5 ))
      fctch2(5 )=gtrmf(comlyn,comlen,'H205',fctch2(5 ))
      fctcg0(5 )=gtrmf(comlyn,comlen,'G005',fctcg0(5 ))
      fctcg1(5 )=gtrmf(comlyn,comlen,'G105',fctcg1(5 ))
      fctcg2(5 )=gtrmf(comlyn,comlen,'G205',fctcg2(5 ))
      fctcg3(5 )=gtrmf(comlyn,comlen,'G305',fctcg3(5 ))

      fctch1(6 )=gtrmf(comlyn,comlen,'H106',fctch1(6 ))
      fctch2(6 )=gtrmf(comlyn,comlen,'H206',fctch2(6 ))
      fctcg0(6 )=gtrmf(comlyn,comlen,'G006',fctcg0(6 ))
      fctcg1(6 )=gtrmf(comlyn,comlen,'G106',fctcg1(6 ))
      fctcg2(6 )=gtrmf(comlyn,comlen,'G206',fctcg2(6 ))
      fctcg3(6 )=gtrmf(comlyn,comlen,'G306',fctcg3(6 ))

      fctch1(7 )=gtrmf(comlyn,comlen,'H107',fctch1(7 ))
      fctch2(7 )=gtrmf(comlyn,comlen,'H207',fctch2(7 ))
      fctcg0(7 )=gtrmf(comlyn,comlen,'G007',fctcg0(7 ))
      fctcg1(7 )=gtrmf(comlyn,comlen,'G107',fctcg1(7 ))
      fctcg2(7 )=gtrmf(comlyn,comlen,'G207',fctcg2(7 ))
      fctcg3(7 )=gtrmf(comlyn,comlen,'G307',fctcg3(7 ))

      fctwos    =gtrmf(comlyn,comlen,'TWOS',fctwos    )

! Obtain the initial value for the measure of symmetry.

      fctbin    =gtrmf(comlyn,comlen,'TBIN',fctbin    )

! Obtain the factor for the exponential function in the Generalized Born
! formula.

      fctkps    =gtrmf(comlyn,comlen,'TKPS',fctkps    )

! Obtain the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

      fctqun(1 )=gtrmf(comlyn,comlen,'UN01',fctqun(1 ))
      fctqun(2 )=gtrmf(comlyn,comlen,'UN02',fctqun(2 ))
      fctqun(3 )=gtrmf(comlyn,comlen,'UN03',fctqun(3 ))
      fctqun(4 )=gtrmf(comlyn,comlen,'UN04',fctqun(4 ))
      fctqun(5 )=gtrmf(comlyn,comlen,'UN05',fctqun(5 ))
      fctqun(6 )=gtrmf(comlyn,comlen,'UN06',fctqun(6 ))
      fctqun(7 )=gtrmf(comlyn,comlen,'UN07',fctqun(7 ))

! Obtain the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

      fctscf(1 )=gtrmf(comlyn,comlen,'CF01',fctscf(1 ))
      fctscf(2 )=gtrmf(comlyn,comlen,'CF02',fctscf(2 ))
      fctscf(3 )=gtrmf(comlyn,comlen,'CF03',fctscf(3 ))
      fctscf(4 )=gtrmf(comlyn,comlen,'CF04',fctscf(4 ))
      fctscf(5 )=gtrmf(comlyn,comlen,'CF05',fctscf(5 ))
      fctscf(6 )=gtrmf(comlyn,comlen,'CF06',fctscf(6 ))
      fctscf(7 )=gtrmf(comlyn,comlen,'CF07',fctscf(7 ))

! Obtain the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

      fctnps(1 )=gtrmf(comlyn,comlen,'PS01',fctnps(1 ))
      fctnps(2 )=gtrmf(comlyn,comlen,'PS02',fctnps(2 ))
      fctnps(3 )=gtrmf(comlyn,comlen,'PS03',fctnps(3 ))
      fctnps(4 )=gtrmf(comlyn,comlen,'PS04',fctnps(4 ))
      fctnps(5 )=gtrmf(comlyn,comlen,'PS05',fctnps(5 ))
      fctnps(6 )=gtrmf(comlyn,comlen,'PS06',fctnps(6 ))
      fctnps(7 )=gtrmf(comlyn,comlen,'PS07',fctnps(7 ))

   endif

   if (fctcps == 22) then

! Obtain the cutoffs for the FACTS self and interaction energy pair
! lists.

      fctcsl    =gtrmf(comlyn,comlen,'TCSL',fctcsl    )
      fctcil    =gtrmf(comlyn,comlen,'TCIL',fctcil    )

! Obtain the cutoffs for the FACTS self and interaction energy
! calculations.

      fctcsc(1 )=gtrmf(comlyn,comlen,'SC01',fctcsc(1 ))
      fctcsc(2 )=gtrmf(comlyn,comlen,'SC02',fctcsc(2 ))
      fctcsc(3 )=gtrmf(comlyn,comlen,'SC03',fctcsc(3 ))
      fctcsc(4 )=gtrmf(comlyn,comlen,'SC04',fctcsc(4 ))
      fctcsc(5 )=gtrmf(comlyn,comlen,'SC05',fctcsc(5 ))
      fctcsc(6 )=gtrmf(comlyn,comlen,'SC06',fctcsc(6 ))
      fctcsc(7 )=gtrmf(comlyn,comlen,'SC07',fctcsc(7 ))
      fctcsc(8 )=gtrmf(comlyn,comlen,'SC08',fctcsc(8 ))
      fctcsc(9 )=gtrmf(comlyn,comlen,'SC09',fctcsc(9 ))
      fctcsc(10)=gtrmf(comlyn,comlen,'SC10',fctcsc(10))
      fctcsc(11)=gtrmf(comlyn,comlen,'SC11',fctcsc(11))
      fctcsc(12)=gtrmf(comlyn,comlen,'SC12',fctcsc(12))
      fctcsc(13)=gtrmf(comlyn,comlen,'SC13',fctcsc(13))
      fctcsc(14)=gtrmf(comlyn,comlen,'SC14',fctcsc(14))
      fctcsc(15)=gtrmf(comlyn,comlen,'SC15',fctcsc(15))
      fctcsc(16)=gtrmf(comlyn,comlen,'SC16',fctcsc(16))
      fctcsc(17)=gtrmf(comlyn,comlen,'SC17',fctcsc(17))
      fctcsc(18)=gtrmf(comlyn,comlen,'SC18',fctcsc(18))
      fctcsc(19)=gtrmf(comlyn,comlen,'SC19',fctcsc(19))
      fctcic    =gtrmf(comlyn,comlen,'TCIC',fctcic    )

! Obtain the parameters for calculating atomic solvation energies.

      fctcb1(1 )=gtrmf(comlyn,comlen,'B101',fctcb1(1 ))
      fctcb2(1 )=gtrmf(comlyn,comlen,'B201',fctcb2(1 ))
      fctca0(1 )=gtrmf(comlyn,comlen,'A001',fctca0(1 ))
      fctca1(1 )=gtrmf(comlyn,comlen,'A101',fctca1(1 ))
      fctca2(1 )=gtrmf(comlyn,comlen,'A201',fctca2(1 ))
      fctca3(1 )=gtrmf(comlyn,comlen,'A301',fctca3(1 ))

      fctcb1(2 )=gtrmf(comlyn,comlen,'B102',fctcb1(2 ))
      fctcb2(2 )=gtrmf(comlyn,comlen,'B202',fctcb2(2 ))
      fctca0(2 )=gtrmf(comlyn,comlen,'A002',fctca0(2 ))
      fctca1(2 )=gtrmf(comlyn,comlen,'A102',fctca1(2 ))
      fctca2(2 )=gtrmf(comlyn,comlen,'A202',fctca2(2 ))
      fctca3(2 )=gtrmf(comlyn,comlen,'A302',fctca3(2 ))

      fctcb1(3 )=gtrmf(comlyn,comlen,'B103',fctcb1(3 ))
      fctcb2(3 )=gtrmf(comlyn,comlen,'B203',fctcb2(3 ))
      fctca0(3 )=gtrmf(comlyn,comlen,'A003',fctca0(3 ))
      fctca1(3 )=gtrmf(comlyn,comlen,'A103',fctca1(3 ))
      fctca2(3 )=gtrmf(comlyn,comlen,'A203',fctca2(3 ))
      fctca3(3 )=gtrmf(comlyn,comlen,'A303',fctca3(3 ))

      fctcb1(4 )=gtrmf(comlyn,comlen,'B104',fctcb1(4 ))
      fctcb2(4 )=gtrmf(comlyn,comlen,'B204',fctcb2(4 ))
      fctca0(4 )=gtrmf(comlyn,comlen,'A004',fctca0(4 ))
      fctca1(4 )=gtrmf(comlyn,comlen,'A104',fctca1(4 ))
      fctca2(4 )=gtrmf(comlyn,comlen,'A204',fctca2(4 ))
      fctca3(4 )=gtrmf(comlyn,comlen,'A304',fctca3(4 ))

      fctcb1(5 )=gtrmf(comlyn,comlen,'B105',fctcb1(5 ))
      fctcb2(5 )=gtrmf(comlyn,comlen,'B205',fctcb2(5 ))
      fctca0(5 )=gtrmf(comlyn,comlen,'A005',fctca0(5 ))
      fctca1(5 )=gtrmf(comlyn,comlen,'A105',fctca1(5 ))
      fctca2(5 )=gtrmf(comlyn,comlen,'A205',fctca2(5 ))
      fctca3(5 )=gtrmf(comlyn,comlen,'A305',fctca3(5 ))

      fctcb1(6 )=gtrmf(comlyn,comlen,'B106',fctcb1(6 ))
      fctcb2(6 )=gtrmf(comlyn,comlen,'B206',fctcb2(6 ))
      fctca0(6 )=gtrmf(comlyn,comlen,'A006',fctca0(6 ))
      fctca1(6 )=gtrmf(comlyn,comlen,'A106',fctca1(6 ))
      fctca2(6 )=gtrmf(comlyn,comlen,'A206',fctca2(6 ))
      fctca3(6 )=gtrmf(comlyn,comlen,'A306',fctca3(6 ))

      fctcb1(7 )=gtrmf(comlyn,comlen,'B107',fctcb1(7 ))
      fctcb2(7 )=gtrmf(comlyn,comlen,'B207',fctcb2(7 ))
      fctca0(7 )=gtrmf(comlyn,comlen,'A007',fctca0(7 ))
      fctca1(7 )=gtrmf(comlyn,comlen,'A107',fctca1(7 ))
      fctca2(7 )=gtrmf(comlyn,comlen,'A207',fctca2(7 ))
      fctca3(7 )=gtrmf(comlyn,comlen,'A307',fctca3(7 ))

      fctcb1(8 )=gtrmf(comlyn,comlen,'B108',fctcb1(8 ))
      fctcb2(8 )=gtrmf(comlyn,comlen,'B208',fctcb2(8 ))
      fctca0(8 )=gtrmf(comlyn,comlen,'A008',fctca0(8 ))
      fctca1(8 )=gtrmf(comlyn,comlen,'A108',fctca1(8 ))
      fctca2(8 )=gtrmf(comlyn,comlen,'A208',fctca2(8 ))
      fctca3(8 )=gtrmf(comlyn,comlen,'A308',fctca3(8 ))

      fctcb1(9 )=gtrmf(comlyn,comlen,'B109',fctcb1(9 ))
      fctcb2(9 )=gtrmf(comlyn,comlen,'B209',fctcb2(9 ))
      fctca0(9 )=gtrmf(comlyn,comlen,'A009',fctca0(9 ))
      fctca1(9 )=gtrmf(comlyn,comlen,'A109',fctca1(9 ))
      fctca2(9 )=gtrmf(comlyn,comlen,'A209',fctca2(9 ))
      fctca3(9 )=gtrmf(comlyn,comlen,'A309',fctca3(9 ))

      fctcb1(10)=gtrmf(comlyn,comlen,'B110',fctcb1(10))
      fctcb2(10)=gtrmf(comlyn,comlen,'B210',fctcb2(10))
      fctca0(10)=gtrmf(comlyn,comlen,'A010',fctca0(10))
      fctca1(10)=gtrmf(comlyn,comlen,'A110',fctca1(10))
      fctca2(10)=gtrmf(comlyn,comlen,'A210',fctca2(10))
      fctca3(10)=gtrmf(comlyn,comlen,'A310',fctca3(10))

      fctcb1(11)=gtrmf(comlyn,comlen,'B111',fctcb1(11))
      fctcb2(11)=gtrmf(comlyn,comlen,'B211',fctcb2(11))
      fctca0(11)=gtrmf(comlyn,comlen,'A011',fctca0(11))
      fctca1(11)=gtrmf(comlyn,comlen,'A111',fctca1(11))
      fctca2(11)=gtrmf(comlyn,comlen,'A211',fctca2(11))
      fctca3(11)=gtrmf(comlyn,comlen,'A311',fctca3(11))

      fctcb1(12)=gtrmf(comlyn,comlen,'B112',fctcb1(12))
      fctcb2(12)=gtrmf(comlyn,comlen,'B212',fctcb2(12))
      fctca0(12)=gtrmf(comlyn,comlen,'A012',fctca0(12))
      fctca1(12)=gtrmf(comlyn,comlen,'A112',fctca1(12))
      fctca2(12)=gtrmf(comlyn,comlen,'A212',fctca2(12))
      fctca3(12)=gtrmf(comlyn,comlen,'A312',fctca3(12))

      fctcb1(13)=gtrmf(comlyn,comlen,'B113',fctcb1(13))
      fctcb2(13)=gtrmf(comlyn,comlen,'B213',fctcb2(13))
      fctca0(13)=gtrmf(comlyn,comlen,'A013',fctca0(13))
      fctca1(13)=gtrmf(comlyn,comlen,'A113',fctca1(13))
      fctca2(13)=gtrmf(comlyn,comlen,'A213',fctca2(13))
      fctca3(13)=gtrmf(comlyn,comlen,'A313',fctca3(13))

      fctcb1(14)=gtrmf(comlyn,comlen,'B114',fctcb1(14))
      fctcb2(14)=gtrmf(comlyn,comlen,'B214',fctcb2(14))
      fctca0(14)=gtrmf(comlyn,comlen,'A014',fctca0(14))
      fctca1(14)=gtrmf(comlyn,comlen,'A114',fctca1(14))
      fctca2(14)=gtrmf(comlyn,comlen,'A214',fctca2(14))
      fctca3(14)=gtrmf(comlyn,comlen,'A314',fctca3(14))

      fctcb1(15)=gtrmf(comlyn,comlen,'B115',fctcb1(15))
      fctcb2(15)=gtrmf(comlyn,comlen,'B215',fctcb2(15))
      fctca0(15)=gtrmf(comlyn,comlen,'A015',fctca0(15))
      fctca1(15)=gtrmf(comlyn,comlen,'A115',fctca1(15))
      fctca2(15)=gtrmf(comlyn,comlen,'A215',fctca2(15))
      fctca3(15)=gtrmf(comlyn,comlen,'A315',fctca3(15))

      fctcb1(16)=gtrmf(comlyn,comlen,'B116',fctcb1(16))
      fctcb2(16)=gtrmf(comlyn,comlen,'B216',fctcb2(16))
      fctca0(16)=gtrmf(comlyn,comlen,'A016',fctca0(16))
      fctca1(16)=gtrmf(comlyn,comlen,'A116',fctca1(16))
      fctca2(16)=gtrmf(comlyn,comlen,'A216',fctca2(16))
      fctca3(16)=gtrmf(comlyn,comlen,'A316',fctca3(16))

      fctcb1(17)=gtrmf(comlyn,comlen,'B117',fctcb1(17))
      fctcb2(17)=gtrmf(comlyn,comlen,'B217',fctcb2(17))
      fctca0(17)=gtrmf(comlyn,comlen,'A017',fctca0(17))
      fctca1(17)=gtrmf(comlyn,comlen,'A117',fctca1(17))
      fctca2(17)=gtrmf(comlyn,comlen,'A217',fctca2(17))
      fctca3(17)=gtrmf(comlyn,comlen,'A317',fctca3(17))

      fctcb1(18)=gtrmf(comlyn,comlen,'B118',fctcb1(18))
      fctcb2(18)=gtrmf(comlyn,comlen,'B218',fctcb2(18))
      fctca0(18)=gtrmf(comlyn,comlen,'A018',fctca0(18))
      fctca1(18)=gtrmf(comlyn,comlen,'A118',fctca1(18))
      fctca2(18)=gtrmf(comlyn,comlen,'A218',fctca2(18))
      fctca3(18)=gtrmf(comlyn,comlen,'A318',fctca3(18))

      fctcb1(19)=gtrmf(comlyn,comlen,'B119',fctcb1(19))
      fctcb2(19)=gtrmf(comlyn,comlen,'B219',fctcb2(19))
      fctca0(19)=gtrmf(comlyn,comlen,'A019',fctca0(19))
      fctca1(19)=gtrmf(comlyn,comlen,'A119',fctca1(19))
      fctca2(19)=gtrmf(comlyn,comlen,'A219',fctca2(19))
      fctca3(19)=gtrmf(comlyn,comlen,'A319',fctca3(19))

      fctqos    =gtrmf(comlyn,comlen,'TQOS',fctqos    )

! Obtain the parameters for calculating atomic solvent accessible
! surface areas.

      fctcd1(1 )=gtrmf(comlyn,comlen,'D101',fctcd1(1 ))
      fctcd2(1 )=gtrmf(comlyn,comlen,'D201',fctcd2(1 ))
      fctcc0(1 )=gtrmf(comlyn,comlen,'C001',fctcc0(1 ))
      fctcc1(1 )=gtrmf(comlyn,comlen,'C101',fctcc1(1 ))
      fctcc2(1 )=gtrmf(comlyn,comlen,'C201',fctcc2(1 ))
      fctcc3(1 )=gtrmf(comlyn,comlen,'C301',fctcc3(1 ))

      fctcd1(2 )=gtrmf(comlyn,comlen,'D102',fctcd1(2 ))
      fctcd2(2 )=gtrmf(comlyn,comlen,'D202',fctcd2(2 ))
      fctcc0(2 )=gtrmf(comlyn,comlen,'C002',fctcc0(2 ))
      fctcc1(2 )=gtrmf(comlyn,comlen,'C102',fctcc1(2 ))
      fctcc2(2 )=gtrmf(comlyn,comlen,'C202',fctcc2(2 ))
      fctcc3(2 )=gtrmf(comlyn,comlen,'C302',fctcc3(2 ))

      fctcd1(3 )=gtrmf(comlyn,comlen,'D103',fctcd1(3 ))
      fctcd2(3 )=gtrmf(comlyn,comlen,'D203',fctcd2(3 ))
      fctcc0(3 )=gtrmf(comlyn,comlen,'C003',fctcc0(3 ))
      fctcc1(3 )=gtrmf(comlyn,comlen,'C103',fctcc1(3 ))
      fctcc2(3 )=gtrmf(comlyn,comlen,'C203',fctcc2(3 ))
      fctcc3(3 )=gtrmf(comlyn,comlen,'C303',fctcc3(3 ))

      fctcd1(4 )=gtrmf(comlyn,comlen,'D104',fctcd1(4 ))
      fctcd2(4 )=gtrmf(comlyn,comlen,'D204',fctcd2(4 ))
      fctcc0(4 )=gtrmf(comlyn,comlen,'C004',fctcc0(4 ))
      fctcc1(4 )=gtrmf(comlyn,comlen,'C104',fctcc1(4 ))
      fctcc2(4 )=gtrmf(comlyn,comlen,'C204',fctcc2(4 ))
      fctcc3(4 )=gtrmf(comlyn,comlen,'C304',fctcc3(4 ))

      fctcd1(5 )=gtrmf(comlyn,comlen,'D105',fctcd1(5 ))
      fctcd2(5 )=gtrmf(comlyn,comlen,'D205',fctcd2(5 ))
      fctcc0(5 )=gtrmf(comlyn,comlen,'C005',fctcc0(5 ))
      fctcc1(5 )=gtrmf(comlyn,comlen,'C105',fctcc1(5 ))
      fctcc2(5 )=gtrmf(comlyn,comlen,'C205',fctcc2(5 ))
      fctcc3(5 )=gtrmf(comlyn,comlen,'C305',fctcc3(5 ))

      fctcd1(6 )=gtrmf(comlyn,comlen,'D106',fctcd1(6 ))
      fctcd2(6 )=gtrmf(comlyn,comlen,'D206',fctcd2(6 ))
      fctcc0(6 )=gtrmf(comlyn,comlen,'C006',fctcc0(6 ))
      fctcc1(6 )=gtrmf(comlyn,comlen,'C106',fctcc1(6 ))
      fctcc2(6 )=gtrmf(comlyn,comlen,'C206',fctcc2(6 ))
      fctcc3(6 )=gtrmf(comlyn,comlen,'C306',fctcc3(6 ))

      fctcd1(7 )=gtrmf(comlyn,comlen,'D107',fctcd1(7 ))
      fctcd2(7 )=gtrmf(comlyn,comlen,'D207',fctcd2(7 ))
      fctcc0(7 )=gtrmf(comlyn,comlen,'C007',fctcc0(7 ))
      fctcc1(7 )=gtrmf(comlyn,comlen,'C107',fctcc1(7 ))
      fctcc2(7 )=gtrmf(comlyn,comlen,'C207',fctcc2(7 ))
      fctcc3(7 )=gtrmf(comlyn,comlen,'C307',fctcc3(7 ))

      fctcd1(8 )=gtrmf(comlyn,comlen,'D108',fctcd1(8 ))
      fctcd2(8 )=gtrmf(comlyn,comlen,'D208',fctcd2(8 ))
      fctcc0(8 )=gtrmf(comlyn,comlen,'C008',fctcc0(8 ))
      fctcc1(8 )=gtrmf(comlyn,comlen,'C108',fctcc1(8 ))
      fctcc2(8 )=gtrmf(comlyn,comlen,'C208',fctcc2(8 ))
      fctcc3(8 )=gtrmf(comlyn,comlen,'C308',fctcc3(8 ))

      fctcd1(9 )=gtrmf(comlyn,comlen,'D109',fctcd1(9 ))
      fctcd2(9 )=gtrmf(comlyn,comlen,'D209',fctcd2(9 ))
      fctcc0(9 )=gtrmf(comlyn,comlen,'C009',fctcc0(9 ))
      fctcc1(9 )=gtrmf(comlyn,comlen,'C109',fctcc1(9 ))
      fctcc2(9 )=gtrmf(comlyn,comlen,'C209',fctcc2(9 ))
      fctcc3(9 )=gtrmf(comlyn,comlen,'C309',fctcc3(9 ))

      fctcd1(10)=gtrmf(comlyn,comlen,'D110',fctcd1(10))
      fctcd2(10)=gtrmf(comlyn,comlen,'D210',fctcd2(10))
      fctcc0(10)=gtrmf(comlyn,comlen,'C010',fctcc0(10))
      fctcc1(10)=gtrmf(comlyn,comlen,'C110',fctcc1(10))
      fctcc2(10)=gtrmf(comlyn,comlen,'C210',fctcc2(10))
      fctcc3(10)=gtrmf(comlyn,comlen,'C310',fctcc3(10))

      fctcd1(11)=gtrmf(comlyn,comlen,'D111',fctcd1(11))
      fctcd2(11)=gtrmf(comlyn,comlen,'D211',fctcd2(11))
      fctcc0(11)=gtrmf(comlyn,comlen,'C011',fctcc0(11))
      fctcc1(11)=gtrmf(comlyn,comlen,'C111',fctcc1(11))
      fctcc2(11)=gtrmf(comlyn,comlen,'C211',fctcc2(11))
      fctcc3(11)=gtrmf(comlyn,comlen,'C311',fctcc3(11))

      fctcd1(12)=gtrmf(comlyn,comlen,'D112',fctcd1(12))
      fctcd2(12)=gtrmf(comlyn,comlen,'D212',fctcd2(12))
      fctcc0(12)=gtrmf(comlyn,comlen,'C012',fctcc0(12))
      fctcc1(12)=gtrmf(comlyn,comlen,'C112',fctcc1(12))
      fctcc2(12)=gtrmf(comlyn,comlen,'C212',fctcc2(12))
      fctcc3(12)=gtrmf(comlyn,comlen,'C312',fctcc3(12))

      fctcd1(13)=gtrmf(comlyn,comlen,'D113',fctcd1(13))
      fctcd2(13)=gtrmf(comlyn,comlen,'D213',fctcd2(13))
      fctcc0(13)=gtrmf(comlyn,comlen,'C013',fctcc0(13))
      fctcc1(13)=gtrmf(comlyn,comlen,'C113',fctcc1(13))
      fctcc2(13)=gtrmf(comlyn,comlen,'C213',fctcc2(13))
      fctcc3(13)=gtrmf(comlyn,comlen,'C313',fctcc3(13))

      fctcd1(14)=gtrmf(comlyn,comlen,'D114',fctcd1(14))
      fctcd2(14)=gtrmf(comlyn,comlen,'D214',fctcd2(14))
      fctcc0(14)=gtrmf(comlyn,comlen,'C014',fctcc0(14))
      fctcc1(14)=gtrmf(comlyn,comlen,'C114',fctcc1(14))
      fctcc2(14)=gtrmf(comlyn,comlen,'C214',fctcc2(14))
      fctcc3(14)=gtrmf(comlyn,comlen,'C314',fctcc3(14))

      fctcd1(15)=gtrmf(comlyn,comlen,'D115',fctcd1(15))
      fctcd2(15)=gtrmf(comlyn,comlen,'D215',fctcd2(15))
      fctcc0(15)=gtrmf(comlyn,comlen,'C015',fctcc0(15))
      fctcc1(15)=gtrmf(comlyn,comlen,'C115',fctcc1(15))
      fctcc2(15)=gtrmf(comlyn,comlen,'C215',fctcc2(15))
      fctcc3(15)=gtrmf(comlyn,comlen,'C315',fctcc3(15))

      fctcd1(16)=gtrmf(comlyn,comlen,'D116',fctcd1(16))
      fctcd2(16)=gtrmf(comlyn,comlen,'D216',fctcd2(16))
      fctcc0(16)=gtrmf(comlyn,comlen,'C016',fctcc0(16))
      fctcc1(16)=gtrmf(comlyn,comlen,'C116',fctcc1(16))
      fctcc2(16)=gtrmf(comlyn,comlen,'C216',fctcc2(16))
      fctcc3(16)=gtrmf(comlyn,comlen,'C316',fctcc3(16))

      fctcd1(17)=gtrmf(comlyn,comlen,'D117',fctcd1(17))
      fctcd2(17)=gtrmf(comlyn,comlen,'D217',fctcd2(17))
      fctcc0(17)=gtrmf(comlyn,comlen,'C017',fctcc0(17))
      fctcc1(17)=gtrmf(comlyn,comlen,'C117',fctcc1(17))
      fctcc2(17)=gtrmf(comlyn,comlen,'C217',fctcc2(17))
      fctcc3(17)=gtrmf(comlyn,comlen,'C317',fctcc3(17))

      fctcd1(18)=gtrmf(comlyn,comlen,'D118',fctcd1(18))
      fctcd2(18)=gtrmf(comlyn,comlen,'D218',fctcd2(18))
      fctcc0(18)=gtrmf(comlyn,comlen,'C018',fctcc0(18))
      fctcc1(18)=gtrmf(comlyn,comlen,'C118',fctcc1(18))
      fctcc2(18)=gtrmf(comlyn,comlen,'C218',fctcc2(18))
      fctcc3(18)=gtrmf(comlyn,comlen,'C318',fctcc3(18))

      fctcd1(19)=gtrmf(comlyn,comlen,'D119',fctcd1(19))
      fctcd2(19)=gtrmf(comlyn,comlen,'D219',fctcd2(19))
      fctcc0(19)=gtrmf(comlyn,comlen,'C019',fctcc0(19))
      fctcc1(19)=gtrmf(comlyn,comlen,'C119',fctcc1(19))
      fctcc2(19)=gtrmf(comlyn,comlen,'C219',fctcc2(19))
      fctcc3(19)=gtrmf(comlyn,comlen,'C319',fctcc3(19))

      fctuos    =gtrmf(comlyn,comlen,'TUOS',fctuos    )

! Obtain the parameters for calculating alpha.

      fctcf1(1 )=gtrmf(comlyn,comlen,'F101',fctcf1(1 ))
      fctcf2(1 )=gtrmf(comlyn,comlen,'F201',fctcf2(1 ))
      fctce0(1 )=gtrmf(comlyn,comlen,'E001',fctce0(1 ))
      fctce1(1 )=gtrmf(comlyn,comlen,'E101',fctce1(1 ))
      fctce2(1 )=gtrmf(comlyn,comlen,'E201',fctce2(1 ))
      fctce3(1 )=gtrmf(comlyn,comlen,'E301',fctce3(1 ))

      fctcf1(2 )=gtrmf(comlyn,comlen,'F102',fctcf1(2 ))
      fctcf2(2 )=gtrmf(comlyn,comlen,'F202',fctcf2(2 ))
      fctce0(2 )=gtrmf(comlyn,comlen,'E002',fctce0(2 ))
      fctce1(2 )=gtrmf(comlyn,comlen,'E102',fctce1(2 ))
      fctce2(2 )=gtrmf(comlyn,comlen,'E202',fctce2(2 ))
      fctce3(2 )=gtrmf(comlyn,comlen,'E302',fctce3(2 ))

      fctcf1(3 )=gtrmf(comlyn,comlen,'F103',fctcf1(3 ))
      fctcf2(3 )=gtrmf(comlyn,comlen,'F203',fctcf2(3 ))
      fctce0(3 )=gtrmf(comlyn,comlen,'E003',fctce0(3 ))
      fctce1(3 )=gtrmf(comlyn,comlen,'E103',fctce1(3 ))
      fctce2(3 )=gtrmf(comlyn,comlen,'E203',fctce2(3 ))
      fctce3(3 )=gtrmf(comlyn,comlen,'E303',fctce3(3 ))

      fctcf1(4 )=gtrmf(comlyn,comlen,'F104',fctcf1(4 ))
      fctcf2(4 )=gtrmf(comlyn,comlen,'F204',fctcf2(4 ))
      fctce0(4 )=gtrmf(comlyn,comlen,'E004',fctce0(4 ))
      fctce1(4 )=gtrmf(comlyn,comlen,'E104',fctce1(4 ))
      fctce2(4 )=gtrmf(comlyn,comlen,'E204',fctce2(4 ))
      fctce3(4 )=gtrmf(comlyn,comlen,'E304',fctce3(4 ))

      fctcf1(5 )=gtrmf(comlyn,comlen,'F105',fctcf1(5 ))
      fctcf2(5 )=gtrmf(comlyn,comlen,'F205',fctcf2(5 ))
      fctce0(5 )=gtrmf(comlyn,comlen,'E005',fctce0(5 ))
      fctce1(5 )=gtrmf(comlyn,comlen,'E105',fctce1(5 ))
      fctce2(5 )=gtrmf(comlyn,comlen,'E205',fctce2(5 ))
      fctce3(5 )=gtrmf(comlyn,comlen,'E305',fctce3(5 ))

      fctcf1(6 )=gtrmf(comlyn,comlen,'F106',fctcf1(6 ))
      fctcf2(6 )=gtrmf(comlyn,comlen,'F206',fctcf2(6 ))
      fctce0(6 )=gtrmf(comlyn,comlen,'E006',fctce0(6 ))
      fctce1(6 )=gtrmf(comlyn,comlen,'E106',fctce1(6 ))
      fctce2(6 )=gtrmf(comlyn,comlen,'E206',fctce2(6 ))
      fctce3(6 )=gtrmf(comlyn,comlen,'E306',fctce3(6 ))

      fctcf1(7 )=gtrmf(comlyn,comlen,'F107',fctcf1(7 ))
      fctcf2(7 )=gtrmf(comlyn,comlen,'F207',fctcf2(7 ))
      fctce0(7 )=gtrmf(comlyn,comlen,'E007',fctce0(7 ))
      fctce1(7 )=gtrmf(comlyn,comlen,'E107',fctce1(7 ))
      fctce2(7 )=gtrmf(comlyn,comlen,'E207',fctce2(7 ))
      fctce3(7 )=gtrmf(comlyn,comlen,'E307',fctce3(7 ))

      fctcf1(8 )=gtrmf(comlyn,comlen,'F108',fctcf1(8 ))
      fctcf2(8 )=gtrmf(comlyn,comlen,'F208',fctcf2(8 ))
      fctce0(8 )=gtrmf(comlyn,comlen,'E008',fctce0(8 ))
      fctce1(8 )=gtrmf(comlyn,comlen,'E108',fctce1(8 ))
      fctce2(8 )=gtrmf(comlyn,comlen,'E208',fctce2(8 ))
      fctce3(8 )=gtrmf(comlyn,comlen,'E308',fctce3(8 ))

      fctcf1(9 )=gtrmf(comlyn,comlen,'F109',fctcf1(9 ))
      fctcf2(9 )=gtrmf(comlyn,comlen,'F209',fctcf2(9 ))
      fctce0(9 )=gtrmf(comlyn,comlen,'E009',fctce0(9 ))
      fctce1(9 )=gtrmf(comlyn,comlen,'E109',fctce1(9 ))
      fctce2(9 )=gtrmf(comlyn,comlen,'E209',fctce2(9 ))
      fctce3(9 )=gtrmf(comlyn,comlen,'E309',fctce3(9 ))

      fctcf1(10)=gtrmf(comlyn,comlen,'F110',fctcf1(10))
      fctcf2(10)=gtrmf(comlyn,comlen,'F210',fctcf2(10))
      fctce0(10)=gtrmf(comlyn,comlen,'E010',fctce0(10))
      fctce1(10)=gtrmf(comlyn,comlen,'E110',fctce1(10))
      fctce2(10)=gtrmf(comlyn,comlen,'E210',fctce2(10))
      fctce3(10)=gtrmf(comlyn,comlen,'E310',fctce3(10))

      fctcf1(11)=gtrmf(comlyn,comlen,'F111',fctcf1(11))
      fctcf2(11)=gtrmf(comlyn,comlen,'F211',fctcf2(11))
      fctce0(11)=gtrmf(comlyn,comlen,'E011',fctce0(11))
      fctce1(11)=gtrmf(comlyn,comlen,'E111',fctce1(11))
      fctce2(11)=gtrmf(comlyn,comlen,'E211',fctce2(11))
      fctce3(11)=gtrmf(comlyn,comlen,'E311',fctce3(11))

      fctcf1(12)=gtrmf(comlyn,comlen,'F112',fctcf1(12))
      fctcf2(12)=gtrmf(comlyn,comlen,'F212',fctcf2(12))
      fctce0(12)=gtrmf(comlyn,comlen,'E012',fctce0(12))
      fctce1(12)=gtrmf(comlyn,comlen,'E112',fctce1(12))
      fctce2(12)=gtrmf(comlyn,comlen,'E212',fctce2(12))
      fctce3(12)=gtrmf(comlyn,comlen,'E312',fctce3(12))

      fctcf1(13)=gtrmf(comlyn,comlen,'F113',fctcf1(13))
      fctcf2(13)=gtrmf(comlyn,comlen,'F213',fctcf2(13))
      fctce0(13)=gtrmf(comlyn,comlen,'E013',fctce0(13))
      fctce1(13)=gtrmf(comlyn,comlen,'E113',fctce1(13))
      fctce2(13)=gtrmf(comlyn,comlen,'E213',fctce2(13))
      fctce3(13)=gtrmf(comlyn,comlen,'E313',fctce3(13))

      fctcf1(14)=gtrmf(comlyn,comlen,'F114',fctcf1(14))
      fctcf2(14)=gtrmf(comlyn,comlen,'F214',fctcf2(14))
      fctce0(14)=gtrmf(comlyn,comlen,'E014',fctce0(14))
      fctce1(14)=gtrmf(comlyn,comlen,'E114',fctce1(14))
      fctce2(14)=gtrmf(comlyn,comlen,'E214',fctce2(14))
      fctce3(14)=gtrmf(comlyn,comlen,'E314',fctce3(14))

      fctcf1(15)=gtrmf(comlyn,comlen,'F115',fctcf1(15))
      fctcf2(15)=gtrmf(comlyn,comlen,'F215',fctcf2(15))
      fctce0(15)=gtrmf(comlyn,comlen,'E015',fctce0(15))
      fctce1(15)=gtrmf(comlyn,comlen,'E115',fctce1(15))
      fctce2(15)=gtrmf(comlyn,comlen,'E215',fctce2(15))
      fctce3(15)=gtrmf(comlyn,comlen,'E315',fctce3(15))

      fctcf1(16)=gtrmf(comlyn,comlen,'F116',fctcf1(16))
      fctcf2(16)=gtrmf(comlyn,comlen,'F216',fctcf2(16))
      fctce0(16)=gtrmf(comlyn,comlen,'E016',fctce0(16))
      fctce1(16)=gtrmf(comlyn,comlen,'E116',fctce1(16))
      fctce2(16)=gtrmf(comlyn,comlen,'E216',fctce2(16))
      fctce3(16)=gtrmf(comlyn,comlen,'E316',fctce3(16))

      fctcf1(17)=gtrmf(comlyn,comlen,'F117',fctcf1(17))
      fctcf2(17)=gtrmf(comlyn,comlen,'F217',fctcf2(17))
      fctce0(17)=gtrmf(comlyn,comlen,'E017',fctce0(17))
      fctce1(17)=gtrmf(comlyn,comlen,'E117',fctce1(17))
      fctce2(17)=gtrmf(comlyn,comlen,'E217',fctce2(17))
      fctce3(17)=gtrmf(comlyn,comlen,'E317',fctce3(17))

      fctcf1(18)=gtrmf(comlyn,comlen,'F118',fctcf1(18))
      fctcf2(18)=gtrmf(comlyn,comlen,'F218',fctcf2(18))
      fctce0(18)=gtrmf(comlyn,comlen,'E018',fctce0(18))
      fctce1(18)=gtrmf(comlyn,comlen,'E118',fctce1(18))
      fctce2(18)=gtrmf(comlyn,comlen,'E218',fctce2(18))
      fctce3(18)=gtrmf(comlyn,comlen,'E318',fctce3(18))

      fctcf1(19)=gtrmf(comlyn,comlen,'F119',fctcf1(19))
      fctcf2(19)=gtrmf(comlyn,comlen,'F219',fctcf2(19))
      fctce0(19)=gtrmf(comlyn,comlen,'E019',fctce0(19))
      fctce1(19)=gtrmf(comlyn,comlen,'E119',fctce1(19))
      fctce2(19)=gtrmf(comlyn,comlen,'E219',fctce2(19))
      fctce3(19)=gtrmf(comlyn,comlen,'E319',fctce3(19))

      fctvos    =gtrmf(comlyn,comlen,'TVOS',fctvos    )

! Obtain the parameters for calculating beta.

      fctch1(1 )=gtrmf(comlyn,comlen,'H101',fctch1(1 ))
      fctch2(1 )=gtrmf(comlyn,comlen,'H201',fctch2(1 ))
      fctcg0(1 )=gtrmf(comlyn,comlen,'G001',fctcg0(1 ))
      fctcg1(1 )=gtrmf(comlyn,comlen,'G101',fctcg1(1 ))
      fctcg2(1 )=gtrmf(comlyn,comlen,'G201',fctcg2(1 ))
      fctcg3(1 )=gtrmf(comlyn,comlen,'G301',fctcg3(1 ))

      fctch1(2 )=gtrmf(comlyn,comlen,'H102',fctch1(2 ))
      fctch2(2 )=gtrmf(comlyn,comlen,'H202',fctch2(2 ))
      fctcg0(2 )=gtrmf(comlyn,comlen,'G002',fctcg0(2 ))
      fctcg1(2 )=gtrmf(comlyn,comlen,'G102',fctcg1(2 ))
      fctcg2(2 )=gtrmf(comlyn,comlen,'G202',fctcg2(2 ))
      fctcg3(2 )=gtrmf(comlyn,comlen,'G302',fctcg3(2 ))

      fctch1(3 )=gtrmf(comlyn,comlen,'H103',fctch1(3 ))
      fctch2(3 )=gtrmf(comlyn,comlen,'H203',fctch2(3 ))
      fctcg0(3 )=gtrmf(comlyn,comlen,'G003',fctcg0(3 ))
      fctcg1(3 )=gtrmf(comlyn,comlen,'G103',fctcg1(3 ))
      fctcg2(3 )=gtrmf(comlyn,comlen,'G203',fctcg2(3 ))
      fctcg3(3 )=gtrmf(comlyn,comlen,'G303',fctcg3(3 ))

      fctch1(4 )=gtrmf(comlyn,comlen,'H104',fctch1(4 ))
      fctch2(4 )=gtrmf(comlyn,comlen,'H204',fctch2(4 ))
      fctcg0(4 )=gtrmf(comlyn,comlen,'G004',fctcg0(4 ))
      fctcg1(4 )=gtrmf(comlyn,comlen,'G104',fctcg1(4 ))
      fctcg2(4 )=gtrmf(comlyn,comlen,'G204',fctcg2(4 ))
      fctcg3(4 )=gtrmf(comlyn,comlen,'G304',fctcg3(4 ))

      fctch1(5 )=gtrmf(comlyn,comlen,'H105',fctch1(5 ))
      fctch2(5 )=gtrmf(comlyn,comlen,'H205',fctch2(5 ))
      fctcg0(5 )=gtrmf(comlyn,comlen,'G005',fctcg0(5 ))
      fctcg1(5 )=gtrmf(comlyn,comlen,'G105',fctcg1(5 ))
      fctcg2(5 )=gtrmf(comlyn,comlen,'G205',fctcg2(5 ))
      fctcg3(5 )=gtrmf(comlyn,comlen,'G305',fctcg3(5 ))

      fctch1(6 )=gtrmf(comlyn,comlen,'H106',fctch1(6 ))
      fctch2(6 )=gtrmf(comlyn,comlen,'H206',fctch2(6 ))
      fctcg0(6 )=gtrmf(comlyn,comlen,'G006',fctcg0(6 ))
      fctcg1(6 )=gtrmf(comlyn,comlen,'G106',fctcg1(6 ))
      fctcg2(6 )=gtrmf(comlyn,comlen,'G206',fctcg2(6 ))
      fctcg3(6 )=gtrmf(comlyn,comlen,'G306',fctcg3(6 ))

      fctch1(7 )=gtrmf(comlyn,comlen,'H107',fctch1(7 ))
      fctch2(7 )=gtrmf(comlyn,comlen,'H207',fctch2(7 ))
      fctcg0(7 )=gtrmf(comlyn,comlen,'G007',fctcg0(7 ))
      fctcg1(7 )=gtrmf(comlyn,comlen,'G107',fctcg1(7 ))
      fctcg2(7 )=gtrmf(comlyn,comlen,'G207',fctcg2(7 ))
      fctcg3(7 )=gtrmf(comlyn,comlen,'G307',fctcg3(7 ))

      fctch1(8 )=gtrmf(comlyn,comlen,'H108',fctch1(8 ))
      fctch2(8 )=gtrmf(comlyn,comlen,'H208',fctch2(8 ))
      fctcg0(8 )=gtrmf(comlyn,comlen,'G008',fctcg0(8 ))
      fctcg1(8 )=gtrmf(comlyn,comlen,'G108',fctcg1(8 ))
      fctcg2(8 )=gtrmf(comlyn,comlen,'G208',fctcg2(8 ))
      fctcg3(8 )=gtrmf(comlyn,comlen,'G308',fctcg3(8 ))

      fctch1(9 )=gtrmf(comlyn,comlen,'H109',fctch1(9 ))
      fctch2(9 )=gtrmf(comlyn,comlen,'H209',fctch2(9 ))
      fctcg0(9 )=gtrmf(comlyn,comlen,'G009',fctcg0(9 ))
      fctcg1(9 )=gtrmf(comlyn,comlen,'G109',fctcg1(9 ))
      fctcg2(9 )=gtrmf(comlyn,comlen,'G209',fctcg2(9 ))
      fctcg3(9 )=gtrmf(comlyn,comlen,'G309',fctcg3(9 ))

      fctch1(10)=gtrmf(comlyn,comlen,'H110',fctch1(10))
      fctch2(10)=gtrmf(comlyn,comlen,'H210',fctch2(10))
      fctcg0(10)=gtrmf(comlyn,comlen,'G010',fctcg0(10))
      fctcg1(10)=gtrmf(comlyn,comlen,'G110',fctcg1(10))
      fctcg2(10)=gtrmf(comlyn,comlen,'G210',fctcg2(10))
      fctcg3(10)=gtrmf(comlyn,comlen,'G310',fctcg3(10))

      fctch1(11)=gtrmf(comlyn,comlen,'H111',fctch1(11))
      fctch2(11)=gtrmf(comlyn,comlen,'H211',fctch2(11))
      fctcg0(11)=gtrmf(comlyn,comlen,'G011',fctcg0(11))
      fctcg1(11)=gtrmf(comlyn,comlen,'G111',fctcg1(11))
      fctcg2(11)=gtrmf(comlyn,comlen,'G211',fctcg2(11))
      fctcg3(11)=gtrmf(comlyn,comlen,'G311',fctcg3(11))

      fctch1(12)=gtrmf(comlyn,comlen,'H112',fctch1(12))
      fctch2(12)=gtrmf(comlyn,comlen,'H212',fctch2(12))
      fctcg0(12)=gtrmf(comlyn,comlen,'G012',fctcg0(12))
      fctcg1(12)=gtrmf(comlyn,comlen,'G112',fctcg1(12))
      fctcg2(12)=gtrmf(comlyn,comlen,'G212',fctcg2(12))
      fctcg3(12)=gtrmf(comlyn,comlen,'G312',fctcg3(12))

      fctch1(13)=gtrmf(comlyn,comlen,'H113',fctch1(13))
      fctch2(13)=gtrmf(comlyn,comlen,'H213',fctch2(13))
      fctcg0(13)=gtrmf(comlyn,comlen,'G013',fctcg0(13))
      fctcg1(13)=gtrmf(comlyn,comlen,'G113',fctcg1(13))
      fctcg2(13)=gtrmf(comlyn,comlen,'G213',fctcg2(13))
      fctcg3(13)=gtrmf(comlyn,comlen,'G313',fctcg3(13))

      fctch1(14)=gtrmf(comlyn,comlen,'H114',fctch1(14))
      fctch2(14)=gtrmf(comlyn,comlen,'H214',fctch2(14))
      fctcg0(14)=gtrmf(comlyn,comlen,'G014',fctcg0(14))
      fctcg1(14)=gtrmf(comlyn,comlen,'G114',fctcg1(14))
      fctcg2(14)=gtrmf(comlyn,comlen,'G214',fctcg2(14))
      fctcg3(14)=gtrmf(comlyn,comlen,'G314',fctcg3(14))

      fctch1(15)=gtrmf(comlyn,comlen,'H115',fctch1(15))
      fctch2(15)=gtrmf(comlyn,comlen,'H215',fctch2(15))
      fctcg0(15)=gtrmf(comlyn,comlen,'G015',fctcg0(15))
      fctcg1(15)=gtrmf(comlyn,comlen,'G115',fctcg1(15))
      fctcg2(15)=gtrmf(comlyn,comlen,'G215',fctcg2(15))
      fctcg3(15)=gtrmf(comlyn,comlen,'G315',fctcg3(15))

      fctch1(16)=gtrmf(comlyn,comlen,'H116',fctch1(16))
      fctch2(16)=gtrmf(comlyn,comlen,'H216',fctch2(16))
      fctcg0(16)=gtrmf(comlyn,comlen,'G016',fctcg0(16))
      fctcg1(16)=gtrmf(comlyn,comlen,'G116',fctcg1(16))
      fctcg2(16)=gtrmf(comlyn,comlen,'G216',fctcg2(16))
      fctcg3(16)=gtrmf(comlyn,comlen,'G316',fctcg3(16))

      fctch1(17)=gtrmf(comlyn,comlen,'H117',fctch1(17))
      fctch2(17)=gtrmf(comlyn,comlen,'H217',fctch2(17))
      fctcg0(17)=gtrmf(comlyn,comlen,'G017',fctcg0(17))
      fctcg1(17)=gtrmf(comlyn,comlen,'G117',fctcg1(17))
      fctcg2(17)=gtrmf(comlyn,comlen,'G217',fctcg2(17))
      fctcg3(17)=gtrmf(comlyn,comlen,'G317',fctcg3(17))

      fctch1(18)=gtrmf(comlyn,comlen,'H118',fctch1(18))
      fctch2(18)=gtrmf(comlyn,comlen,'H218',fctch2(18))
      fctcg0(18)=gtrmf(comlyn,comlen,'G018',fctcg0(18))
      fctcg1(18)=gtrmf(comlyn,comlen,'G118',fctcg1(18))
      fctcg2(18)=gtrmf(comlyn,comlen,'G218',fctcg2(18))
      fctcg3(18)=gtrmf(comlyn,comlen,'G318',fctcg3(18))

      fctch1(19)=gtrmf(comlyn,comlen,'H119',fctch1(19))
      fctch2(19)=gtrmf(comlyn,comlen,'H219',fctch2(19))
      fctcg0(19)=gtrmf(comlyn,comlen,'G019',fctcg0(19))
      fctcg1(19)=gtrmf(comlyn,comlen,'G119',fctcg1(19))
      fctcg2(19)=gtrmf(comlyn,comlen,'G219',fctcg2(19))
      fctcg3(19)=gtrmf(comlyn,comlen,'G319',fctcg3(19))

      fctwos    =gtrmf(comlyn,comlen,'TWOS',fctwos    )

! Obtain the initial value for the measure of symmetry.

      fctbin    =gtrmf(comlyn,comlen,'TBIN',fctbin    )

! Obtain the factor for the exponential function in the Generalized Born
! formula.

      fctkps    =gtrmf(comlyn,comlen,'TKPS',fctkps    )

! Obtain the quenching factors for the atomic solvation energies. They
! affect both atomic solvation and interaction energies.

      fctqun(1 )=gtrmf(comlyn,comlen,'UN01',fctqun(1 ))
      fctqun(2 )=gtrmf(comlyn,comlen,'UN02',fctqun(2 ))
      fctqun(3 )=gtrmf(comlyn,comlen,'UN03',fctqun(3 ))
      fctqun(4 )=gtrmf(comlyn,comlen,'UN04',fctqun(4 ))
      fctqun(5 )=gtrmf(comlyn,comlen,'UN05',fctqun(5 ))
      fctqun(6 )=gtrmf(comlyn,comlen,'UN06',fctqun(6 ))
      fctqun(7 )=gtrmf(comlyn,comlen,'UN07',fctqun(7 ))
      fctqun(8 )=gtrmf(comlyn,comlen,'UN08',fctqun(8 ))
      fctqun(9 )=gtrmf(comlyn,comlen,'UN09',fctqun(9 ))
      fctqun(10)=gtrmf(comlyn,comlen,'UN10',fctqun(10))
      fctqun(11)=gtrmf(comlyn,comlen,'UN11',fctqun(11))
      fctqun(12)=gtrmf(comlyn,comlen,'UN12',fctqun(12))
      fctqun(13)=gtrmf(comlyn,comlen,'UN13',fctqun(13))
      fctqun(14)=gtrmf(comlyn,comlen,'UN14',fctqun(14))
      fctqun(15)=gtrmf(comlyn,comlen,'UN15',fctqun(15))
      fctqun(16)=gtrmf(comlyn,comlen,'UN16',fctqun(16))
      fctqun(17)=gtrmf(comlyn,comlen,'UN17',fctqun(17))
      fctqun(18)=gtrmf(comlyn,comlen,'UN18',fctqun(18))
      fctqun(19)=gtrmf(comlyn,comlen,'UN19',fctqun(19))

! Obtain the scaling factors for the charges and for the FCTNPS factors.
! They only affect atomic solvation energies and the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies. They do not affect interaction energies.

      fctscf(1 )=gtrmf(comlyn,comlen,'CF01',fctscf(1 ))
      fctscf(2 )=gtrmf(comlyn,comlen,'CF02',fctscf(2 ))
      fctscf(3 )=gtrmf(comlyn,comlen,'CF03',fctscf(3 ))
      fctscf(4 )=gtrmf(comlyn,comlen,'CF04',fctscf(4 ))
      fctscf(5 )=gtrmf(comlyn,comlen,'CF05',fctscf(5 ))
      fctscf(6 )=gtrmf(comlyn,comlen,'CF06',fctscf(6 ))
      fctscf(7 )=gtrmf(comlyn,comlen,'CF07',fctscf(7 ))
      fctscf(8 )=gtrmf(comlyn,comlen,'CF08',fctscf(8 ))
      fctscf(9 )=gtrmf(comlyn,comlen,'CF09',fctscf(9 ))
      fctscf(10)=gtrmf(comlyn,comlen,'CF10',fctscf(10))
      fctscf(11)=gtrmf(comlyn,comlen,'CF11',fctscf(11))
      fctscf(12)=gtrmf(comlyn,comlen,'CF12',fctscf(12))
      fctscf(13)=gtrmf(comlyn,comlen,'CF13',fctscf(13))
      fctscf(14)=gtrmf(comlyn,comlen,'CF14',fctscf(14))
      fctscf(15)=gtrmf(comlyn,comlen,'CF15',fctscf(15))
      fctscf(16)=gtrmf(comlyn,comlen,'CF16',fctscf(16))
      fctscf(17)=gtrmf(comlyn,comlen,'CF17',fctscf(17))
      fctscf(18)=gtrmf(comlyn,comlen,'CF18',fctscf(18))
      fctscf(19)=gtrmf(comlyn,comlen,'CF19',fctscf(19))

! Obtain the factors to obtain the nonpolar contributions that are
! proportional to the unit charge atomic solvation energies.

      fctnps(1 )=gtrmf(comlyn,comlen,'PS01',fctnps(1 ))
      fctnps(2 )=gtrmf(comlyn,comlen,'PS02',fctnps(2 ))
      fctnps(3 )=gtrmf(comlyn,comlen,'PS03',fctnps(3 ))
      fctnps(4 )=gtrmf(comlyn,comlen,'PS04',fctnps(4 ))
      fctnps(5 )=gtrmf(comlyn,comlen,'PS05',fctnps(5 ))
      fctnps(6 )=gtrmf(comlyn,comlen,'PS06',fctnps(6 ))
      fctnps(7 )=gtrmf(comlyn,comlen,'PS07',fctnps(7 ))
      fctnps(8 )=gtrmf(comlyn,comlen,'PS08',fctnps(8 ))
      fctnps(9 )=gtrmf(comlyn,comlen,'PS09',fctnps(9 ))
      fctnps(10)=gtrmf(comlyn,comlen,'PS10',fctnps(10))
      fctnps(11)=gtrmf(comlyn,comlen,'PS11',fctnps(11))
      fctnps(12)=gtrmf(comlyn,comlen,'PS12',fctnps(12))
      fctnps(13)=gtrmf(comlyn,comlen,'PS13',fctnps(13))
      fctnps(14)=gtrmf(comlyn,comlen,'PS14',fctnps(14))
      fctnps(15)=gtrmf(comlyn,comlen,'PS15',fctnps(15))
      fctnps(16)=gtrmf(comlyn,comlen,'PS16',fctnps(16))
      fctnps(17)=gtrmf(comlyn,comlen,'PS17',fctnps(17))
      fctnps(18)=gtrmf(comlyn,comlen,'PS18',fctnps(18))
      fctnps(19)=gtrmf(comlyn,comlen,'PS19',fctnps(19))
   
   endif

! Process soft parameters.

! Assign to each atom the FACTS index.

   fcterr=.false.
   do i=1,natom
      fctidx(i)=0
      do j=1,fctnav
         if (wmain(i) == fctrvw(j)) then
            fctidx(i)=j
         endif
      enddo
      if (fctidx(i) == 0) then
         if (.not. fctavw) then
            if (prnlev >= 2) then
               write(outu,'(A39,E11.4,A10,I6,A18)')             &
                    ' FCTINI> Error: Van der Waals radius of', &
                    wmain(i),                                  &
                    ' A of atom',                              &
                    i,                                         &
                    ' is not supported.'
            endif
            fcterr=.true.
         else
            fctnav=fctnav+1
            if (fctnav  >  fctmvw) then
               call wrndie(-4,'<FCTINI>','Too many vdW radii.')
               return
            endif
            fctrvw(fctnav)=wmain(i)
            fctidx(i)=fctnav
            if (prnlev >= 2) then
               write(outu,'(A43,A10,E11.4,A2,A17)')                 &
                    ' FCTINI> FACTS parameters for van der Waals', &
                    ' radius of',                                  &
                    wmain(i),                                      &
                    ' A',                                          &
                    ' are being added.'
            endif
         endif
      endif
   enddo
   if (fcterr) then
      call wrndie(-4,'<FCTINI>','Missing FACTS parameter(s).')
      return
   endif

! Add FACTS parameters of non native van der Waals radii.

   if (fctavw) then
      do i=fctnnv+1,fctnav
         fctnst=1
         fctdif=dabs(fctrvw(fctnst)-fctrvw(i))
         do j=2,fctnnv
            if (dabs(fctrvw(j)-fctrvw(i))  <  fctdif) then
               fctnst=j
               fctdif=dabs(fctrvw(fctnst)-fctrvw(i))
            endif
         enddo
         fctcsc(i)=fctcsc(fctnst)
         fctcb1(i)=fctcb1(fctnst)
         fctcb2(i)=fctcb2(fctnst)
         fctca0(i)=fctca0(fctnst)*(fctrvw(fctnst)/fctrvw(i))
         fctca1(i)=fctca1(fctnst)*(fctrvw(fctnst)/fctrvw(i))
         fctca2(i)=fctca2(fctnst)
         fctca3(i)=fctca3(fctnst)
         fctcd1(i)=fctcd1(fctnst)
         fctcd2(i)=fctcd2(fctnst)
         fctcc0(i)=fctcc0(fctnst)*(fctrvw(i)/fctrvw(fctnst)) &
                                 *(fctrvw(i)/fctrvw(fctnst))
         fctcc1(i)=fctcc1(fctnst)*(fctrvw(i)/fctrvw(fctnst)) &
                                 *(fctrvw(i)/fctrvw(fctnst))
         fctcc2(i)=fctcc2(fctnst)
         fctcc3(i)=fctcc3(fctnst)
         fctcf1(i)=fctcf1(fctnst)
         fctcf2(i)=fctcf2(fctnst)
         fctce0(i)=fctce0(fctnst)
         fctce1(i)=fctce1(fctnst)
         fctce2(i)=fctce2(fctnst)
         fctce3(i)=fctce3(fctnst)
         fctch1(i)=fctch1(fctnst)
         fctch2(i)=fctch2(fctnst)
         fctcg0(i)=fctcg0(fctnst)
         fctcg1(i)=fctcg1(fctnst)
         fctcg2(i)=fctcg2(fctnst)
         fctcg3(i)=fctcg3(fctnst)
         fctqun(i)=fctqun(fctnst)
         fctscf(i)=fctscf(fctnst)
         fctnps(i)=fctnps(fctnst)
      enddo
   endif

! Obtain the reciprocal value of FCTKPS.

   fctikp=one/fctkps

! Obtain the charges squared and scaled for the calculations of atomic
! solvation energies.

   ! do i=1,natom
   !    fctcgs(i)= fctscf(fctidx(i))*cg(i)*cg(i)
   ! enddo

   if(fctscro)then
   ! Screening only -> nullify fctcgs(i) / only affect Self-Solvation
      do i=1,natom
         fctcgs(i)  = zero*cg(i)*cg(i)
         ! fctcgs2(i) = fctscf(fctidx(i))*cg(i)*cg(i)
         fctslfw(i) = zero
      enddo
   else
      do i=1,natom
         fctcgs(i)= fctscf(fctidx(i))*cg(i)*cg(i)
         ! fctcgs2(i)= fctscf(fctidx(i))*cg(i)*cg(i)
         fctslfw(i) = one
      enddo
   endif

! Obtain FCTNPS squared and scaled for the calculations of the nonpolar
! contributions that are proportional to the unit charge atomic
! solvation energies.

   do i=1,natom
      fctnpc(i)=-fctscf(fctidx(i))*fctnps(fctidx(i))*fctnps(fctidx(i))
   enddo

! Obtain the cutoffs squared for the FACTS self, interaction, and
! screening energy pair lists.

   fct1ln=fctcsl*fctcsl
   fct2ln=fctcil*fctcil
   fct3ln=ctonnb*ctonnb

! Obtain the cutoffs squared and their reciprocal values for the FACTS
! self, interaction, and screening energy calculations.

   do i=1,fctnav
      fct1cn(i)=fctcsc(i)*fctcsc(i)
      fct1ci(i)=one/fct1cn(i)
   enddo
   fct2cn=fctcic*fctcic
   fct2ci=one/fct2cn
   fct3cn=ctofnb*ctofnb
   fct3ci=one/fct3cn

! Obtain the constants that correspond to the cutoffs for the FACTS
! self, interaction, and screening energy calculations.

   do i=1,fctnav
      fct1fn(i)=-four/fct1cn(i)
   enddo
   fct2fn=-four/fct2cn
   fct3fn=-four/fct3cn

! Obtain the van der Waals volumes.

   do i=1,fctnav
      fctvol(i)=(four/three)*pi*fctrvw(i)*fctrvw(i)*fctrvw(i)
   enddo

! ===================================================================
! Van der Waals solute-solvent dispersion terms as
! implemented by Gallichio

   do i=1,fctmvw
         fctvdwalpha(i)=zero
   enddo 

   fctvdwalpha(  1)=gtrmf(comlyn,comlen,'VDW001',zero)
   fctvdwalpha(  2)=gtrmf(comlyn,comlen,'VDW002',zero)
   fctvdwalpha(  3)=gtrmf(comlyn,comlen,'VDW003',zero)
   fctvdwalpha(  4)=gtrmf(comlyn,comlen,'VDW004',zero)
   fctvdwalpha(  5)=gtrmf(comlyn,comlen,'VDW005',zero)
   fctvdwalpha(  6)=gtrmf(comlyn,comlen,'VDW006',zero)
   fctvdwalpha(  7)=gtrmf(comlyn,comlen,'VDW007',zero)
   fctvdwalpha(  8)=gtrmf(comlyn,comlen,'VDW008',zero)
   fctvdwalpha(  9)=gtrmf(comlyn,comlen,'VDW009',zero)
   fctvdwalpha( 10)=gtrmf(comlyn,comlen,'VDW010',zero)
   fctvdwalpha( 11)=gtrmf(comlyn,comlen,'VDW011',zero)
   fctvdwalpha( 12)=gtrmf(comlyn,comlen,'VDW012',zero)
   fctvdwalpha( 13)=gtrmf(comlyn,comlen,'VDW013',zero)
   fctvdwalpha( 14)=gtrmf(comlyn,comlen,'VDW014',zero)
   fctvdwalpha( 15)=gtrmf(comlyn,comlen,'VDW015',zero)
   fctvdwalpha( 16)=gtrmf(comlyn,comlen,'VDW016',zero)
   fctvdwalpha( 17)=gtrmf(comlyn,comlen,'VDW017',zero)
   fctvdwalpha( 18)=gtrmf(comlyn,comlen,'VDW018',zero)
   fctvdwalpha( 19)=gtrmf(comlyn,comlen,'VDW019',zero)
   fctvdwalpha( 20)=gtrmf(comlyn,comlen,'VDW020',zero)
   fctvdwalpha( 21)=gtrmf(comlyn,comlen,'VDW021',zero)
   fctvdwalpha( 22)=gtrmf(comlyn,comlen,'VDW022',zero)
   fctvdwalpha( 23)=gtrmf(comlyn,comlen,'VDW023',zero)
   fctvdwalpha( 24)=gtrmf(comlyn,comlen,'VDW024',zero)
   fctvdwalpha( 25)=gtrmf(comlyn,comlen,'VDW025',zero)
   fctvdwalpha( 26)=gtrmf(comlyn,comlen,'VDW026',zero)
   fctvdwalpha( 27)=gtrmf(comlyn,comlen,'VDW027',zero)
   fctvdwalpha( 28)=gtrmf(comlyn,comlen,'VDW028',zero)
   fctvdwalpha( 29)=gtrmf(comlyn,comlen,'VDW029',zero)
   fctvdwalpha( 30)=gtrmf(comlyn,comlen,'VDW030',zero)
   fctvdwalpha( 31)=gtrmf(comlyn,comlen,'VDW031',zero)
   fctvdwalpha( 32)=gtrmf(comlyn,comlen,'VDW032',zero)
   fctvdwalpha( 33)=gtrmf(comlyn,comlen,'VDW033',zero)
   fctvdwalpha( 34)=gtrmf(comlyn,comlen,'VDW034',zero)
   fctvdwalpha( 35)=gtrmf(comlyn,comlen,'VDW035',zero)
   fctvdwalpha( 36)=gtrmf(comlyn,comlen,'VDW036',zero)
   fctvdwalpha( 37)=gtrmf(comlyn,comlen,'VDW037',zero)
   fctvdwalpha( 38)=gtrmf(comlyn,comlen,'VDW038',zero)
   fctvdwalpha( 39)=gtrmf(comlyn,comlen,'VDW039',zero)
   fctvdwalpha( 40)=gtrmf(comlyn,comlen,'VDW040',zero)
   fctvdwalpha( 41)=gtrmf(comlyn,comlen,'VDW041',zero)
   fctvdwalpha( 42)=gtrmf(comlyn,comlen,'VDW042',zero)
   fctvdwalpha( 43)=gtrmf(comlyn,comlen,'VDW043',zero)
   fctvdwalpha( 44)=gtrmf(comlyn,comlen,'VDW044',zero)
   fctvdwalpha( 45)=gtrmf(comlyn,comlen,'VDW045',zero)
   fctvdwalpha( 46)=gtrmf(comlyn,comlen,'VDW046',zero)
   fctvdwalpha( 47)=gtrmf(comlyn,comlen,'VDW047',zero)
   fctvdwalpha( 48)=gtrmf(comlyn,comlen,'VDW048',zero)
   fctvdwalpha( 49)=gtrmf(comlyn,comlen,'VDW049',zero)
   fctvdwalpha( 50)=gtrmf(comlyn,comlen,'VDW050',zero)
   fctvdwalpha( 51)=gtrmf(comlyn,comlen,'VDW051',zero)
   fctvdwalpha( 52)=gtrmf(comlyn,comlen,'VDW052',zero)
   fctvdwalpha( 53)=gtrmf(comlyn,comlen,'VDW053',zero)
   fctvdwalpha( 54)=gtrmf(comlyn,comlen,'VDW054',zero)
   fctvdwalpha( 55)=gtrmf(comlyn,comlen,'VDW055',zero)
   fctvdwalpha( 56)=gtrmf(comlyn,comlen,'VDW056',zero)
   fctvdwalpha( 57)=gtrmf(comlyn,comlen,'VDW057',zero)
   fctvdwalpha( 58)=gtrmf(comlyn,comlen,'VDW058',zero)
   fctvdwalpha( 59)=gtrmf(comlyn,comlen,'VDW059',zero)
   fctvdwalpha( 60)=gtrmf(comlyn,comlen,'VDW060',zero)
   fctvdwalpha( 61)=gtrmf(comlyn,comlen,'VDW061',zero)
   fctvdwalpha( 62)=gtrmf(comlyn,comlen,'VDW062',zero)
   fctvdwalpha( 63)=gtrmf(comlyn,comlen,'VDW063',zero)
   fctvdwalpha( 64)=gtrmf(comlyn,comlen,'VDW064',zero)
   fctvdwalpha( 65)=gtrmf(comlyn,comlen,'VDW065',zero)
   fctvdwalpha( 66)=gtrmf(comlyn,comlen,'VDW066',zero)
   fctvdwalpha( 67)=gtrmf(comlyn,comlen,'VDW067',zero)
   fctvdwalpha( 68)=gtrmf(comlyn,comlen,'VDW068',zero)
   fctvdwalpha( 69)=gtrmf(comlyn,comlen,'VDW069',zero)
   fctvdwalpha( 70)=gtrmf(comlyn,comlen,'VDW070',zero)
   fctvdwalpha( 71)=gtrmf(comlyn,comlen,'VDW071',zero)
   fctvdwalpha( 72)=gtrmf(comlyn,comlen,'VDW072',zero)
   fctvdwalpha( 73)=gtrmf(comlyn,comlen,'VDW073',zero)
   fctvdwalpha( 74)=gtrmf(comlyn,comlen,'VDW074',zero)
   fctvdwalpha( 75)=gtrmf(comlyn,comlen,'VDW075',zero)
   fctvdwalpha( 76)=gtrmf(comlyn,comlen,'VDW076',zero)
   fctvdwalpha( 77)=gtrmf(comlyn,comlen,'VDW077',zero)
   fctvdwalpha( 78)=gtrmf(comlyn,comlen,'VDW078',zero)
   fctvdwalpha( 79)=gtrmf(comlyn,comlen,'VDW079',zero)
   fctvdwalpha( 80)=gtrmf(comlyn,comlen,'VDW080',zero)
   fctvdwalpha( 81)=gtrmf(comlyn,comlen,'VDW081',zero)
   fctvdwalpha( 82)=gtrmf(comlyn,comlen,'VDW082',zero)
   fctvdwalpha( 83)=gtrmf(comlyn,comlen,'VDW083',zero)
   fctvdwalpha( 84)=gtrmf(comlyn,comlen,'VDW084',zero)
   fctvdwalpha( 85)=gtrmf(comlyn,comlen,'VDW085',zero)
   fctvdwalpha( 86)=gtrmf(comlyn,comlen,'VDW086',zero)
   fctvdwalpha( 87)=gtrmf(comlyn,comlen,'VDW087',zero)
   fctvdwalpha( 88)=gtrmf(comlyn,comlen,'VDW088',zero)
   fctvdwalpha( 89)=gtrmf(comlyn,comlen,'VDW089',zero)
   fctvdwalpha( 90)=gtrmf(comlyn,comlen,'VDW090',zero)
   fctvdwalpha( 91)=gtrmf(comlyn,comlen,'VDW091',zero)
   fctvdwalpha( 92)=gtrmf(comlyn,comlen,'VDW092',zero)
   fctvdwalpha( 93)=gtrmf(comlyn,comlen,'VDW093',zero)
   fctvdwalpha( 94)=gtrmf(comlyn,comlen,'VDW094',zero)
   fctvdwalpha( 95)=gtrmf(comlyn,comlen,'VDW095',zero)
   fctvdwalpha( 96)=gtrmf(comlyn,comlen,'VDW096',zero)
   fctvdwalpha( 97)=gtrmf(comlyn,comlen,'VDW097',zero)
   fctvdwalpha( 98)=gtrmf(comlyn,comlen,'VDW098',zero)
   fctvdwalpha( 99)=gtrmf(comlyn,comlen,'VDW099',zero)
   fctvdwalpha(100)=gtrmf(comlyn,comlen,'VDW100',zero)

   do i=1,natom
      fctvdwalp(i) = fctvdwalpha(iac(i))
   enddo

! FCTVDWK = -16/3 * Pi
   fctvdwk   =-0.167552e+02
! FCTVDWRHO: number density of water
   fctvdwrho = 0.334280e-01
! FCTVDWWRD: water radius
   fctvdwwrd = 1.4d0

   fctvdwepsotn = 'OT'
   fctvdwepshtn = 'HT'

   do i=1,natc
      if (atc(i)==fctvdwepsotn) then
         fctvdwepsot = eff(itc(i))
         fctvdwsigot = vdwr(itc(i))
      endif
      if (atc(i)==fctvdwepshtn) then
         fctvdwepsht = eff(itc(i))
         fctvdwsight = vdwr(itc(i))
      endif
   enddo

   do i=1,natom
      fctvdweps = eff(itc(iac(i)))
      fctvdwsig = vdwr(itc(iac(i)))

      fctvdwepsox = sqrt(abs(fctvdwepsot*fctvdweps))
      fctvdwsigox = fctvdwsigot + fctvdwsig
      fctvdwsigox = fctvdwsigox**6

      fctvdwepshx = sqrt(abs(fctvdwepsht*fctvdweps))
      fctvdwsighx = fctvdwsight + fctvdwsig
      fctvdwsighx = fctvdwsighx**6

      fctvdwfac(i)= fctvdwk*fctvdwrho         &
                     *fctvdwepsox*fctvdwsigox &
                     *fctvdwalp(i) 
   enddo

! =====================================================================
! Salt Stuff
! ----------
   !   fctconc, fcttemp,fctkappa,fctscreen
   fctkappa = zero
   if(fctconc > zero) then
      fctkappa = sqrt(fcties/(2530.362733d0*fctconc/fcttemp))
      ! fctscreen = one / fctkappa
      fctscreen = saltfact / fctkappa

      fctk2ew = half*fctscreen*fcties
   endif

   fcttai = one/fcttau
   ! write(*,*) "fctkappa", fctkappa, "fctscreen", fctscreen
! =====================================================================
!  Check is fast code should be used
! fast part :
! No Salt contribution (conc =0.0)
! No Gallicchio Solute-Solvent vdW interaction (fctvdwfac(i) = 0.0)
! Simple surface tension (no sigmoidal function of surface tension fctce1(fctidx(i)=0)
! No fixed atoms

   ifast=.true.

   if(fctkappa /= zero) ifast=.false.

   do i=1,natom
      if(fctce1(fctidx(i)) /= zero) ifast=.false.
      if(fctvdwfac(i) /= zero) ifast=.false.
   enddo
! =====================================================================
! FACTS Interaction Lists Allocation
! ----------------------------------
! Allocate space on the heap for the FACTS lists.
! Based on estimates instead of (N*(N-1)/2)
! Estimate half of maximum space for a dense system (water)
! assuming 10 A**3 for each atom. 0.20944=4/3*PI/10/2/2

! Allocate space on the heap for the real lists.

   mxfcab=0

   cutmax=max(max(fctcil,fctcsl),cutnb)
   fctsz=nbscal*cutmax**3*natom*0.10472
   fctszi=nint(fctsz)
   if (fctsz > mxfcab) mxfcab=int(fctsz)
   if (mxfcab  >  ((natom-1)*natom/2)) then
      mxfcab=(natom-1)*natom/2
   endif
   mxfcib=2*mxfcab*imscal

   if(allocated(FCTBND%fct1ilo))write(outu,*) "FCT1IL already allocated"
   if(allocated(FCTBND%fct1jnb))write(outu,*) "FCT1IB already allocated"
   if(allocated(FCTBND%fct2ilo))write(outu,*) "FCT2IL already allocated"
   if(allocated(FCTBND%fct2jnb))write(outu,*) "FCT2IB already allocated"

   call chmalloc('fctall.src','FCTINI','FCT1ILO', natom,  intg=FCTBND%fct1ilo, ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','FCT1JNB', mxfcab, intg=FCTBND%fct1jnb, ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','FCT2ILO', natom,  intg=FCTBND%fct2ilo, ierr=ierr2, qdie=.true.)
   call chmalloc('fctall.src','FCTINI','FCT2JNB', mxfcab, intg=FCTBND%fct2jnb, ierr=ierr2, qdie=.true.)

   if (fctaim) then
      ! write(outu,*) 'Setup of FACTS image lists'

      if(allocated(FCTBND%fct3ilo))write(outu,*) "FCT3ILO already allocated"
      if(allocated(FCTBND%fct3jnb))write(outu,*) "FCT3JNB already allocated"
      if(allocated(FCTBND%fct4ilo))write(outu,*) "FCT4ILO already allocated"
      if(allocated(FCTBND%fct4jnb))write(outu,*) "FCT4JNB already allocated"

      call chmalloc('fctall.src','FCTINI','FCT3ILO', 26*natom, intg=FCTBND%fct3ilo, ierr=ierr2, qdie=.true.)
      call chmalloc('fctall.src','FCTINI','FCT3JNB', mxfcib  , intg=FCTBND%fct3jnb, ierr=ierr2, qdie=.true.)
      call chmalloc('fctall.src','FCTINI','FCT4ILO', 26*natom, intg=FCTBND%fct4ilo, ierr=ierr2, qdie=.true.)
      call chmalloc('fctall.src','FCTINI','FCT4JNB', mxfcib  , intg=FCTBND%fct4jnb, ierr=ierr2, qdie=.true.)
   endif

! Make sure that all the lists are filled and up to date.

   call gete0('ENER',fctdum,0)

! Print some data.
! =================================
#if KEY_PARALLEL==1
#if KEY_STRINGM==1 /* VO stringm : limit output*/
   if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then 
#else /**/
   if (mynod==0) then
#endif /*  VO stringm*/
#endif 
! =================================
   write(outu,'(A1)')     &
       ''
   write(outu,'(A45)')    &
       ' FCTINI> FACTS 6.03 successfully initialized.'
   if (fctsfe) then
      write(outu,'(A32)') &
          ' FCTINI> FACTS solvation chosen.'
   else
      write(outu,'(A32)') &
          ' FCTINI> FACTS screening chosen.'
   endif

   write(outu,'(A1)')       &
       ''
   write(outu,'(A30,I3)')   &
       ' FCTINI> CHARMM parameter set:',fctcps
   write(outu,'(A38,F5.2)') &
       ' FCTINI> Internal dielectric constant:',fcteps
   write(outu,'(A43)')      &
       ' FCTINI> External dielectric constant: 78.5'
   write(outu,'(A29,I2)')   &
       ' FCTINI> FACTS parameter set:',fctfps
   write(outu,'(A40,F8.4,A18)')   &
       ' FCTINI> Surface tension coeff. (gamm): ',fctap0, '  [kcal/(molxA^2)]'

   if(fctconc > zero)then
   write(outu,'(A1)')                                         &
      ''
      write(outu,'(A33,F6.3,A4)')                             &
         ' FCTINI> Salt concentration:     ',fctconc,' [M]'
      write(outu,'(A33,F6.1,A4)')                             &
         ' FCTINI> Temperature:            ',fcttemp,' [K]'
      write(outu,'(A33,F6.3,A4)')                             &
         ' FCTINI> Debye screening length: ',fctkappa,' [A]'
   endif


   write(outu,'(A1)')    &
       ''
   write(outu,'(A36)')   &
       ' FCTINI> FACTS solvation parameters:'
   write(outu,'(A1)')    &
       ''
   write(outu,'(4A14)')  &
       '   Line number', &
       '    vdW radius', &
       '            B1', &
       '            B2'
   write(outu,'(A1)') &
       ''
   do i=1,fctnav
      write(outu,'(I14,3E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctcb1(i),             &
          fctcb2(i)
   enddo
   write(outu,'(A1)')    &
       ''
   write(outu,'(6A14)')  &
       '   Line number', &
       '    vdW radius', &
       '            A0', &
       '            A1', &
       '            A2', &
       '            A3'
   write(outu,'(A1)')    &
       ''
   do i=1,fctnav
      write(outu,'(I14,5E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctca0(i),             &
          fctca1(i),             &
          fctca2(i),             &
          fctca3(i)
   enddo

   write(outu,'(A1)')        &
       ''
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTQOS =',fctqos

   write(outu,'(A1)')        &
       ''
   write(outu,'(A34)')       &
       ' FCTINI> FACTS surface parameters:'
   write(outu,'(A1)')    &
       ''
   write(outu,'(4A14)')  &
       '   Line number', &
       '    vdW radius', &
       '            D1', &
       '            D2'
   write(outu,'(A1)')    &
       ''
   do i=1,fctnav
      write(outu,'(I14,3E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctcd1(i),             &
          fctcd2(i)
   enddo
   write(outu,'(A1)')    &
       ''
   write(outu,'(6A14)')  &
       '   Line number', &
       '    vdW radius', &
       '            C0', &
       '            C1', &
       '            C2', &
       '            C3'
   write(outu,'(A1)')    &
       ''
   do i=1,fctnav
      write(outu,'(I14,5E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctcc0(i),             &
          fctcc1(i),             &
          fctcc2(i),             &
          fctcc3(i)
   enddo

   write(outu,'(A1)')        &
       ''
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTUOS =',fctuos

   write(outu,'(A1)')    &
       ''
   write(outu,'(A32)')   &
       ' FCTINI> FACTS alpha parameters:'
   write(outu,'(A1)')    &
       ''
   write(outu,'(4A14)')  &
       '   Line number', &
       '    vdW radius', &
       '            F1', &
       '            F2'
   write(outu,'(A1)')    &
       ''
   do i=1,fctnav
      write(outu,'(I14,3E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctcf1(i),             &
          fctcf2(i)
   enddo
   write(outu,'(A1)')    &
       ''
   write(outu,'(6A14)')  &
       '   Line number', &
       '    vdW radius', &
       '    E0 (gamma)', &
       '            E1', &
       '            E2', &
       '            E3'
   write(outu,'(A1)')    &
       ''
   do i=1,fctnav
      write(outu,'(I14,5E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctce0(i),             &
          fctce1(i),             &
          fctce2(i),             &
          fctce3(i)
   enddo

   write(outu,'(A1)')        &
       ''
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTVOS =',fctvos

   write(outu,'(A1)')        &
       ''
   write(outu,'(A31)')       &
       ' FCTINI> FACTS beta parameters:'
   write(outu,'(A1)')        &
       ''
   write(outu,'(4A14)')      &
       '   Line number',     &
       '    vdW radius',     &
       '            H1',     &
       '            H2'
   write(outu,'(A1)')        &
       ''
   do i=1,fctnav
      write(outu,'(I14,3E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctch1(i),             &
          fctch2(i)
   enddo
   write(outu,'(A1)')    &
       ''
   write(outu,'(6A14)')  &
       '   Line number', &
       '    vdW radius', &
       '            G0', &
       '            G1', &
       '            G2', &
       '            G3'
   write(outu,'(A1)')    &
       ''
   do i=1,fctnav
      write(outu,'(I14,5E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctcg0(i),             &
          fctcg1(i),             &
          fctcg2(i),             &
          fctcg3(i)
   enddo

   write(outu,'(A1)')        &
       ''
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTWOS =',fctwos

   write(outu,'(A1)')        &
       ''
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTBIN =',fctbin

   write(outu,'(A1)')        &
       ''
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTKPS =',fctkps

   write(outu,'(A1)')        &
       ''
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTCSL =',fctcsl
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTCIL =',fctcil

   write(outu,'(A1)')        &
       ''
   write(outu,'(A23)')       &
       ' FCTINI> FACTS cutoffs:'
   write(outu,'(A1)')        &
       ''
   write(outu,'(3A14)')      &
       '   Line number',     &
       '    vdW radius',     &
       '        FCTCSC'
   write(outu,'(A1)')        &
       ''
   do i=1,fctnav
      write(outu,'(I14,2E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctcsc(i)
   enddo

   write(outu,'(A1)')        &
       ''
   write(outu,'(A17,E13.6)') &
       ' FCTINI> FCTCIC =',fctcic

   write(outu,'(A1)')        &
       ''
   write(outu,'(A23)')       &
       ' FCTINI> FACTS factors:'
   write(outu,'(A1)')        &
       ''
   write(outu,'(5A14)')      &
       '   Line number',     &
       '    vdW radius',     &
       '        FCTQUN',     &
       '        FCTSCF',     &
       '        FCTNPS'
      write(outu,'(A1)')     &
       ''
   do i=1,fctnav
      write(outu,'(I14,4E14.6)') &
          i,                     &
          fctrvw(i),             &
          fctqun(i),             &
          fctscf(i),             &
          fctnps(i)
   enddo

   write(outu,'(A1)')     &
       ''
   write(outu,'(A21)')    &
       ' FCTINI> FACTS flags:'
   write(outu,'(A1)')     &
       ''
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTRUN: ',fctrun
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTSFE: ',fctsfe
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTSCR: ',fctscr
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTPSL: ',fctpsl
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTPIN: ',fctpin
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTPSR: ',fctpsr
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTPAL: ',fctpal
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTPBT: ',fctpbt
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTPVW: ',fctpvw
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTPFR: ',fctpfr
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTAVW: ',fctavw
   write(outu,'(A17,L1)') &
       ' FCTINI> FCTAIM: ',fctaim

   write(outu,'(A1)')     &
       ''
   write(outu,'(A59,A25)') &
       ' FCTINI> Do not change the default values unless you really', &
       ' know what you are doing.'
! =================================
#if KEY_PARALLEL==1
   endif
   call psync()
#endif 
! =================================

   call gete0('ENER',fctdum,0)

! Print additional data if required.

   if (fctpsl .or. fctpsr .or. fctpal  .or.          &
       fctpbt .or. fctpvw .or. fctpfr) then
         call fctprt(fctpol                         ,&
                     fctnpl                         ,&
                     natom                          ,&
                     bnbnd%inblo ,bnbnd%jnb         ,&
                     fctbnd%fct1ilo,fctbnd%fct1jnb  ,&
                     fctbnd%fct2ilo,fctbnd%fct2jnb  ,&
                     natim                          ,&
                     bimag%imblo ,bimag%imjnb       ,&
                     bimag%imattr                   ,&
                     fctbnd%fct3ilo,fctbnd%fct3jnb  ,&  ! ?????????????
                     fctbnd%fct4ilo,fctbnd%fct4jnb)     ! ?????????????
   endif

   ! Array Deallocation
   call chmdealloc('fctall.src','FCTINI','fctrvw'     , fctmvw, crl=fctrvw)
   call chmdealloc('fctall.src','FCTINI','fctcsc'     , fctmvw, crl=fctcsc)
   call chmdealloc('fctall.src','FCTINI','fctscf'     , fctmvw, crl=fctscf)
   call chmdealloc('fctall.src','FCTINI','fctnps'     , fctmvw, crl=fctnps)
   call chmdealloc('fctall.src','FCTINI','fctvdwalpha', fctmvw, crl=fctvdwalpha)

   RETURN
   END SUBROUTINE fctini
! ##ENDIF (factz1)

! ##IF FACTS (factz2)
   SUBROUTINE fctene(fctpol         ,&
                     fctnpl         ,&
                     reanat         ,&
                     reablo,reanbo  ,&
                     fct1ll,fct1lb  ,&
                     fct2ll,fct2lb  ,&
                     imanat         ,&
                     imablo,imanbo  ,&
                     imattr         ,&
                     fct3ll,fct3lb  ,&
                     fct4ll,fct4lb)

! This routine calculates the FACTS solvation free energy and its
! derivative with respect to Cartesian coordinates.
!
! Author: Urs Haberthuer.

   use chm_kinds
   use dimens_fcm

   use coord
   use deriv

   use inbnd
   use number
   use psf
#if KEY_PARALLEL==1
   use parallel
#endif 
      implicit none

   real(kind=chm_real), intent(out)  :: fctpol, fctnpl
   integer, intent(in) :: reanat, imanat
   integer, intent(in) :: imattr(*)
   integer, intent(in) :: reablo(*), reanbo(*)
   integer, intent(in) :: fct1ll(*), fct1lb(*)
   integer, intent(in) :: fct2ll(*), fct2lb(*)
   integer, intent(in) :: imablo(*), imanbo(*)
   integer, intent(in) :: fct3ll(*), fct3lb(*)
   integer, intent(in) :: fct4ll(*), fct4lb(*)

   integer             :: a,b,u,v,i,j
   logical             :: qmove

   real(kind=chm_real) :: vux,vuy,vuz
   real(kind=chm_real) :: fctdvu,fctivu,fctsvu

   real(kind=chm_real) :: fct01vuss,fct02vuss,fct03vuss,fct04vuss
   real(kind=chm_real) :: fct05vuss,fct06vuss,fct07vuss
   real(kind=chm_real) :: fct01uvss,fct02uvss,fct03uvss,fct04uvss
   real(kind=chm_real) :: fct05uvss,fct06uvss,fct07uvss

#if KEY_STRINGM==1 /* VO when stringm is used, communication is usually faster with fewer calls to gcomb, so combine arrays below:*/
  real(kind=chm_real), target, dimension(natom,18) :: FCTCOMM
  real(kind=chm_real), pointer, dimension(:) :: FCT01SS,FCT02SS,FCT03SS,&
      &                                         FCT01XS,FCT01YS,FCT01ZS,&
      &                                         FCT02XS,FCT02YS,FCT02ZS,&
      &                                         FCT03XS,FCT03YS,FCT03ZS,&
      &                                         FCT01XX,FCT01XY,FCT01XZ,&
      &                                         FCT01YY,FCT01YZ,&
      &                                         FCT01ZZ
#else /*  VO stringm*/
   real(kind=chm_real) :: fct01ss(natom),fct02ss(natom),fct03ss(natom)
   real(kind=chm_real) :: fct01xs(natom),fct01ys(natom),fct01zs(natom)
   real(kind=chm_real) :: fct02xs(natom),fct02ys(natom),fct02zs(natom)
   real(kind=chm_real) :: fct03xs(natom),fct03ys(natom),fct03zs(natom)
   real(kind=chm_real) :: fct01xx(natom),fct01xy(natom),fct01xz(natom)
   real(kind=chm_real) :: fct01yy(natom),fct01yz(natom)
   real(kind=chm_real) :: fct01zz(natom)
#endif /*  VO stringm*/
   real(kind=chm_real) :: fct04xs,fct04ys,fct04zs

   real(kind=chm_real) :: fcttp1,fcttp2,fcttp3,fcttp4
   real(kind=chm_real) :: fcttp5,fcttp6,fcttp7,fcttp8

   real(kind=chm_real) :: fctis1(natom),fctis2(natom)
   real(kind=chm_real) :: fctmov(natom),fctmos(natom)

   real(kind=chm_real) :: fctqsg(natom)
   real(kind=chm_real) :: fctisg(natom)
   real(kind=chm_real) :: fctqdv(natom),fctqds(natom)
   real(kind=chm_real) :: fctusg(natom)
   real(kind=chm_real) :: fctudv(natom),fctuds(natom)
   real(kind=chm_real) :: fctvsg(natom)
   real(kind=chm_real) :: fctvdv(natom),fctvds(natom)
   real(kind=chm_real) :: fctwsg(natom)
   real(kind=chm_real) :: fctwdv(natom),fctwds(natom)

   real(kind=chm_real) :: fctpl1,fctpl2
   real(kind=chm_real) :: fctnp1,fctnp2,fctnp3

   real(kind=chm_real) :: fctqsx(natom),fctqsy(natom),fctqsz(natom)
   real(kind=chm_real) :: fctusx(natom),fctusy(natom),fctusz(natom)
   real(kind=chm_real) :: fctvsx(natom),fctvsy(natom),fctvsz(natom)
   real(kind=chm_real) :: fctwsx(natom),fctwsy(natom),fctwsz(natom)

   real(kind=chm_real) :: fct01vu,fct02vu,fct03vu,fct04vu
   real(kind=chm_real) :: fct05vu,fct06vu,fct07vu
   real(kind=chm_real) :: fct08vu,fct09vu,fct10vu,fct11vu
   real(kind=chm_real) :: fct12vu,fct13vu,fct14vu

   ! Salt Stuff
   real(kind=chm_real) :: fctslt01, fctslt02, fctslt03
   real(kind=chm_real) :: fctqsgslt
   real(kind=chm_real) :: fctcgslt(natom)

   real(kind=chm_real) :: fctggg(natom), fctkkk(natom)
   real(kind=chm_real) :: fcthhh(natom)

   real(kind=chm_real) :: fctssx(natom),fctssy(natom),fctssz(natom)
   real(kind=chm_real) :: fctsix(natom),fctsiy(natom),fctsiz(natom)
   real(kind=chm_real) :: fctiix(natom),fctiiy(natom),fctiiz(natom)

   real(kind=chm_real) :: fct02vuxs,fct02vuys,fct02vuzs
   real(kind=chm_real) :: fct02uvxs,fct02uvys,fct02uvzs
   real(kind=chm_real) :: fct03vuxs,fct03vuys,fct03vuzs
   real(kind=chm_real) :: fct03uvxs,fct03uvys,fct03uvzs
   real(kind=chm_real) :: fct04vuxs,fct04vuys,fct04vuzs
   real(kind=chm_real) :: fct04uvxs,fct04uvys,fct04uvzs
   real(kind=chm_real) :: fct01vuxx,fct01vuxy,fct01vuxz
   real(kind=chm_real) ::           fct01vuyy,fct01vuyz
   real(kind=chm_real) ::                     fct01vuzz
   real(kind=chm_real) :: fct01uvxx,fct01uvxy,fct01uvxz
   real(kind=chm_real) ::           fct01uvyy,fct01uvyz
   real(kind=chm_real) ::                     fct01uvzz

   real(kind=chm_real) :: fctqsvux,fctqsvuy,fctqsvuz
   real(kind=chm_real) :: fctqsuvx,fctqsuvy,fctqsuvz
   real(kind=chm_real) :: fctusvux,fctusvuy,fctusvuz
   real(kind=chm_real) :: fctusuvx,fctusuvy,fctusuvz
   real(kind=chm_real) :: fctvsvux,fctvsvuy,fctvsvuz
   real(kind=chm_real) :: fctvsuvx,fctvsuvy,fctvsuvz
   real(kind=chm_real) :: fctwsvux,fctwsvuy,fctwsvuz
   real(kind=chm_real) :: fctwsuvx,fctwsuvy,fctwsuvz

   real(kind=chm_real) :: fctvdw01,fctvdw02
   real(kind=chm_real) :: fctvdwen(natom),fctvdwdf(natom)

   real(kind=chm_real) :: fctself(natom), fctnpol(natom), fctscreene(natom)
! ===================================================================
#if KEY_STRINGM==1 /* VO stringm : reduce number of calls to gcomb*/
    FCT01SS=>FCTCOMM(:,1)
    FCT02SS=>FCTCOMM(:,2)
    FCT03SS=>FCTCOMM(:,3)
    FCT01XS=>FCTCOMM(:,4)
    FCT01YS=>FCTCOMM(:,5)
    FCT01ZS=>FCTCOMM(:,6)
    FCT02XS=>FCTCOMM(:,7)
    FCT02YS=>FCTCOMM(:,8)
    FCT02ZS=>FCTCOMM(:,9)
    FCT03XS=>FCTCOMM(:,10)
    FCT03YS=>FCTCOMM(:,11)
    FCT03ZS=>FCTCOMM(:,12)
    FCT01XX=>FCTCOMM(:,13)
    FCT01XY=>FCTCOMM(:,14)
    FCT01XZ=>FCTCOMM(:,15)
    FCT01YY=>FCTCOMM(:,16)
    FCT01YZ=>FCTCOMM(:,17)
    FCT01ZZ=>FCTCOMM(:,18)
    FCTCOMM=zero
#endif 
! ===================================================================
   if (fctsfe) then
      qmove=.false.

      fctpl1  = zero
      fctpl2  = zero
      fctnp1  = zero
      fctnp2  = zero
      fctnp3  = zero

#if KEY_STRINGM==0 /* VO initialized above*/
      fct01ss = zero

      fct03ss = zero
      fct01xs = zero
      fct01ys = zero
      fct01zs = zero
      fct02xs = zero
      fct02ys = zero
      fct02zs = zero
      fct03xs = zero
      fct03ys = zero
      fct03zs = zero
      fct01xx = zero
      fct01xy = zero
      fct01xz = zero
      fct01yy = zero
      fct01yz = zero
      fct01zz = zero
#endif 
      fctggg  = zero
      fctkkk  = zero

      fctssx  = zero
      fctssy  = zero
      fctssz  = zero
      fctsix  = zero
      fctsiy  = zero
      fctsiz  = zero
      fctiix  = zero
      fctiiy  = zero
      fctiiz  = zero

      fcthhh  = zero
      fctvsg  = zero
      fctusg  = zero

      fctself = zero
      fctnpol = zero
      fctscreene = zero

      do i=1, reanat
! ==========================================
#if KEY_PARALLEL==1
         fctqsg(i) =zero
         if(mynod==0)then
            fct02ss(i)=fctbin
         else
            fct02ss(i)=zero
         endif
#else /**/
         fct02ss(i)=fctbin
#endif 
! ==========================================
         if(imove(i)>0) ifast=.false.
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            u=i
            v=fct1lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct01vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u)))
               fct02vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu
               fct03vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))
               fct06vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctivu
               fct01ss(u)= fct01ss(u)+fct01vuss
               fct02ss(u)= fct02ss(u)+fct02vuss
               fct03ss(u)= fct03ss(u)+fct03vuss
               fct01xs(u)= fct01xs(u)+fct03vuss*vux
               fct01ys(u)= fct01ys(u)+fct03vuss*vuy
               fct01zs(u)= fct01zs(u)+fct03vuss*vuz
               fct02xs(u)= fct02xs(u)+fct05vuss*vux
               fct02ys(u)= fct02ys(u)+fct05vuss*vuy
               fct02zs(u)= fct02zs(u)+fct05vuss*vuz
               fct03xs(u)= fct03xs(u)+(-fct04vuss+fct06vuss)*vux
               fct03ys(u)= fct03ys(u)+(-fct04vuss+fct06vuss)*vuy
               fct03zs(u)= fct03zs(u)+(-fct04vuss+fct06vuss)*vuz
               fct01xx(u)= fct01xx(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01xy(u)= fct01xy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01xz(u)= fct01xz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01yy(u)= fct01yy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01yz(u)= fct01yz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01zz(u)= fct01zz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct01uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v)))
               fct02uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu
               fct03uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))
               fct06uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctivu
               fct01ss(v)= fct01ss(v)+fct01uvss
               fct02ss(v)= fct02ss(v)+fct02uvss
               fct03ss(v)= fct03ss(v)+fct03uvss
               fct01xs(v)= fct01xs(v)-fct03uvss*vux
               fct01ys(v)= fct01ys(v)-fct03uvss*vuy
               fct01zs(v)= fct01zs(v)-fct03uvss*vuz
               fct02xs(v)= fct02xs(v)-fct05uvss*vux
               fct02ys(v)= fct02ys(v)-fct05uvss*vuy
               fct02zs(v)= fct02zs(v)-fct05uvss*vuz
               fct03xs(v)= fct03xs(v)-(-fct04uvss+fct06uvss)*vux
               fct03ys(v)= fct03ys(v)-(-fct04uvss+fct06uvss)*vuy
               fct03zs(v)= fct03zs(v)-(-fct04uvss+fct06uvss)*vuz
               fct01xx(v)= fct01xx(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01xy(v)= fct01xy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01xz(v)= fct01xz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01yy(v)= fct01yy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01yz(v)= fct01yz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01zz(v)= fct01zz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct01vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u)))
               fct02vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu
               fct03vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))
               fct06vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctivu
               fct01ss(u)= fct01ss(u)+fct01vuss
               fct02ss(u)= fct02ss(u)+fct02vuss
               fct03ss(u)= fct03ss(u)+fct03vuss
               fct01xs(u)= fct01xs(u)+fct03vuss*vux
               fct01ys(u)= fct01ys(u)+fct03vuss*vuy
               fct01zs(u)= fct01zs(u)+fct03vuss*vuz
               fct02xs(u)= fct02xs(u)+fct05vuss*vux
               fct02ys(u)= fct02ys(u)+fct05vuss*vuy
               fct02zs(u)= fct02zs(u)+fct05vuss*vuz
               fct03xs(u)= fct03xs(u)+(-fct04vuss+fct06vuss)*vux
               fct03ys(u)= fct03ys(u)+(-fct04vuss+fct06vuss)*vuy
               fct03zs(u)= fct03zs(u)+(-fct04vuss+fct06vuss)*vuz
               fct01xx(u)= fct01xx(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01xy(u)= fct01xy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01xz(u)= fct01xz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01yy(u)= fct01yy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01yz(u)= fct01yz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01zz(u)= fct01zz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct01uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v)))
               fct02uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu
               fct03uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))
               fct06uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctivu
               fct01ss(v)= fct01ss(v)+fct01uvss
               fct02ss(v)= fct02ss(v)+fct02uvss
               fct03ss(v)= fct03ss(v)+fct03uvss
               fct01xs(v)= fct01xs(v)-fct03uvss*vux
               fct01ys(v)= fct01ys(v)-fct03uvss*vuy
               fct01zs(v)= fct01zs(v)-fct03uvss*vuz
               fct02xs(v)= fct02xs(v)-fct05uvss*vux
               fct02ys(v)= fct02ys(v)-fct05uvss*vuy
               fct02zs(v)= fct02zs(v)-fct05uvss*vuz
               fct03xs(v)= fct03xs(v)-(-fct04uvss+fct06uvss)*vux
               fct03ys(v)= fct03ys(v)-(-fct04uvss+fct06uvss)*vuy
               fct03zs(v)= fct03zs(v)-(-fct04uvss+fct06uvss)*vuz
               fct01xx(v)= fct01xx(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01xy(v)= fct01xy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01xz(v)= fct01xz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01yy(v)= fct01yy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01yz(v)= fct01yz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01zz(v)= fct01zz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz
            endif
         enddo
      enddo

! ==========================================
#if KEY_PARALLEL==1
#if KEY_STRINGM==1 /* VO with string, usually faster to reduce with one call*/
      call gcombr(fctcomm,18*reanat)
#else /**/
      call gcomb(fct01ss,reanat)
      call gcomb(fct02ss,reanat)
      call gcomb(fct03ss,reanat)
      call gcomb(fct01xs,reanat)
      call gcomb(fct01ys,reanat)
      call gcomb(fct01zs,reanat)
      call gcomb(fct02xs,reanat)
      call gcomb(fct02ys,reanat)
      call gcomb(fct02zs,reanat)
      call gcomb(fct03xs,reanat)
      call gcomb(fct03ys,reanat)
      call gcomb(fct03zs,reanat)
      call gcomb(fct01xx,reanat)
      call gcomb(fct01xy,reanat)
      call gcomb(fct01xz,reanat)
      call gcomb(fct01yy,reanat)
      call gcomb(fct01yz,reanat)
      call gcomb(fct01zz,reanat)
#endif 
#endif 
! ==========================================

      if(ifast) then
      ! fast part :
      ! No Salt contribution (conc =0.0)
      ! No Gallicchio Solute-Solvent vdW interaction (fctvdwfac(i) = 0.0)
      ! Simple surface tension (no sigmoidal function of surface tension fctce1(fctidx(i)=0)
      ! No fixed atoms

         do i=1,reanat
            fcttp1   = sqrt( fct01xs(i)*fct01xs(i)  &
                            +fct01ys(i)*fct01ys(i)  &
                            +fct01zs(i)*fct01zs(i))
            fcttp2   = fct02ss(i)

            fctis1(i)= one/fcttp1
            fctis2(i)= one/fcttp2

            fctmov(i)= fct01ss(i)
            fctmos(i)= fcttp1*fctis2(i)

            fcttp1   =                   fctmov(i)           &
                      +fctcb1(fctidx(i))          *fctmos(i) &
                      +fctcb2(fctidx(i))*fctmov(i)*fctmos(i) &
                      +fctqos
            fcttp2   = one                                   &
                      +fctcb2(fctidx(i))          *fctmos(i)
            fcttp3   = fctcb1(fctidx(i))                     &
                      +fctcb2(fctidx(i))*fctmov(i)
            fcttp4   = fctca2(fctidx(i))*(fcttp1-fctca3(fctidx(i)))
            fcttp5   = dexp(-fcttp4)
            fcttp6   = one/(one+fcttp5)
            fcttp7   = fctca0(fctidx(i))                     &
                      +fctca1(fctidx(i))*fcttp6
            fcttp8   = fctca1(fctidx(i))*fctca2(fctidx(i))   &
                      /(one/fcttp5+two+fcttp5)
            fctqsg(i)= fcttp7
            fctqdv(i)= fcttp2*fcttp8
            fctqds(i)= fcttp3*fcttp8
            if (fctqsg(i) > -0.001d0) then
               call wrndie(-4,'<FCTENE>','Solvation energy too high.')
            endif
            fctqsg(i)= fctqsg(i)*fctqun(fctidx(i))
            fctqdv(i)= fctqdv(i)*fctqun(fctidx(i))
            fctqds(i)= fctqds(i)*fctqun(fctidx(i))
            ! SASA Sigmoidal
            fcttp1   =                   fctmov(i)           &
                      +fctcd1(fctidx(i))          *fctmos(i) &
                      +fctcd2(fctidx(i))*fctmov(i)*fctmos(i) &
                      +fctuos
            fcttp2   = one                                   &
                      +fctcd2(fctidx(i))          *fctmos(i)
            fcttp3   = fctcd1(fctidx(i))                     &
                      +fctcd2(fctidx(i))*fctmov(i)
            fcttp4   = fctcc2(fctidx(i))*(fcttp1-fctcc3(fctidx(i)))
            fcttp5   = dexp(-fcttp4)
            fcttp6   = one/(one+fcttp5)
            fcttp7   = fctcc0(fctidx(i))                     &
                      +fctcc1(fctidx(i))*fcttp6
            fcttp8   = fctcc1(fctidx(i))*fctcc2(fctidx(i))   &
                      /(one/fcttp5+two+fcttp5)
            fctusg(i)= fcttp7
            fctudv(i)= fcttp2*fcttp8
            fctuds(i)= fcttp3*fcttp8
            if (fctusg(i) < 0.001d0) then
               call wrndie(-4,'<FCTENE>','Surface too low.')
            endif
            ! Surface Tension Sigmoidal

            fcttp7   = fctce0(fctidx(i))
            fctvsg(i)= fcttp7
            ! A.P. Volumetric Correction Sigmoidal
            ! g0, g1, g2 ,g3
            fcttp1   =                   fctmov(i)           &
                      +fctch1(fctidx(i))          *fctmos(i) &
                      +fctch2(fctidx(i))*fctmov(i)*fctmos(i) &
                      +fctwos
            fcttp2   = one                                   &
                      +fctch2(fctidx(i))          *fctmos(i)
            fcttp3   = fctch1(fctidx(i))                     &
                      +fctch2(fctidx(i))*fctmov(i)
            fcttp4   = fctcg2(fctidx(i))*(fcttp1-fctcg3(fctidx(i)))
            fcttp5   = dexp(-fcttp4)
            fcttp6   = one/(one+fcttp5)
            fcttp7   = fctcg0(fctidx(i))                     &
                      +fctcg1(fctidx(i))*fcttp6
            fcttp8   = fctcg1(fctidx(i))*fctcg2(fctidx(i))   &
                      /(one/fcttp5+two+fcttp5)
            fctwsg(i)= fcttp7
            fctwdv(i)= fcttp2*fcttp8
            fctwds(i)= fcttp3*fcttp8

            fctisg(i)= one/fctqsg(i)
            ! VdW TERM ===================================================
            ! ============================================================
            ! Salt term
               fctqsgslt   = fctqsg(i)
               fctcgslt(i) = fctcgs(i)
            ! ============================================================
            fct04xs  =(fct03ss(i)*fct01xs(i)          &
                      +fct01xs(i)*fct01xx(i)          &
                      +fct01ys(i)*fct01xy(i)          &
                      +fct01zs(i)*fct01xz(i))         &
                      *fctis1(i) *fctis2(i)           &
                      -fct03xs(i)*fctis2(i)*fctmos(i)
            fct04ys  =(fct03ss(i)*fct01ys(i)          &
                      +fct01xs(i)*fct01xy(i)          &
                      +fct01ys(i)*fct01yy(i)          &
                      +fct01zs(i)*fct01yz(i))         &
                      *fctis1(i) *fctis2(i)           &
                      -fct03ys(i)*fctis2(i)*fctmos(i)
            fct04zs  =(fct03ss(i)*fct01zs(i)          &
                      +fct01xs(i)*fct01xz(i)          &
                      +fct01ys(i)*fct01yz(i)          &
                      +fct01zs(i)*fct01zz(i))         &
                      *fctis1(i) *fctis2(i)           &
                      -fct03zs(i)*fctis2(i)*fctmos(i)

            fctqsx(i)=-fct02xs(i)*fctqdv(i)-fct04xs*fctqds(i)
            fctqsy(i)=-fct02ys(i)*fctqdv(i)-fct04ys*fctqds(i)
            fctqsz(i)=-fct02zs(i)*fctqdv(i)-fct04zs*fctqds(i)
            fctusx(i)=-fct02xs(i)*fctudv(i)-fct04xs*fctuds(i)
            fctusy(i)=-fct02ys(i)*fctudv(i)-fct04ys*fctuds(i)
            fctusz(i)=-fct02zs(i)*fctudv(i)-fct04zs*fctuds(i)
! ==========================================
#if KEY_PARALLEL==1
         if(mynod==0)then
#endif 
! ==========================================
            ! fctpl1   = fctpl1+fctcgs(i)*fctqsgslt
            fctself(i)= fctcgs(i)*fctqsgslt

            fctnpol(i) = fctnpc(i)*fctqsg(i) + fctusg(i)*fctvsg(i)
            ! fctnp1   = fctnp1+fctnpc(i)*fctqsg(i)
            ! fctnp2   = fctnp2+fctusg(i)*fctvsg(i)
            ! ! fctnp3   = fctnp3+fctwsg(i)+fctvdwen(i)

            fctssx(i)=  fctssx(i)                        &
                      +(fctcgslt(i)+fctnpc(i))*fctqsx(i) &
                      +             fctvsg(i) *fctusx(i)
            fctssy(i)=  fctssy(i)                        &
                      +(fctcgslt(i)+fctnpc(i))*fctqsy(i) &
                      +             fctvsg(i) *fctusy(i)
            fctssz(i)=  fctssz(i)                        &
                      +(fctcgslt(i)+fctnpc(i))*fctqsz(i) &
                      +             fctvsg(i) *fctusz(i)
! ==========================================
#if KEY_PARALLEL==1
         endif
#endif 
! ==========================================
         enddo

      if(doscreen)then  ! (doscreen)
         ! Primary Atoms Screened Interactions
         ! -----------------------------------
         do i=1,reanat
            if (i == 1) then
               a=1
            else
               a=fct2ll(i-1)+1
            endif
            b=fct2ll(i)
            do j=a,b
               u=i
               v=fct2lb(j)
               vux=x(v)-x(u)
               vuy=y(v)-y(u)
               vuz=z(v)-z(u)
               fctdvu=vux*vux+vuy*vuy+vuz*vuz
               if (fctdvu < fct2cn) then
                  fct01vu  = one/fctdvu
                  fct02vu  = one-fctdvu*fct2ci
                  fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
                  fct04vu  = dexp(-fctikp*fct03vu)
                  fct05vu  = fct04vu/fct03vu
                  fct06vu  = fct05vu+fctikp*fct04vu
                  fct07vu  = one/(one+fct05vu)
                  fct08vu  =-cg(v)*cg(u)*fcttau*fct02vu              &
                            *sqrt(fct01vu*fct07vu)
                  fct09vu  = (fct01vu*fct02vu*(-one+fct06vu*fct07vu) &
                            +fct2fn)*fct08vu
                  fct10vu  = half*fct02vu*fct06vu*fct07vu*fct08vu
                  fctpl2   = fctpl2+fct02vu*fct08vu
                  ! Atomic Screening Energy Array
                  fctscreene(u) = fctscreene(u)+fct02vu*fct08vu*half
                  fctscreene(v) = fctscreene(v)+fct02vu*fct08vu*half
                  ! ---
                  fctggg(u)= fctggg(u)-fct10vu
                  fctiix(u)= fctiix(u)                               &
                            -fct09vu*vux+fct10vu*fctisg(u)*fctqsx(u)
                  fctiiy(u)= fctiiy(u)                               &
                            -fct09vu*vuy+fct10vu*fctisg(u)*fctqsy(u)
                  fctiiz(u)= fctiiz(u)                               &
                            -fct09vu*vuz+fct10vu*fctisg(u)*fctqsz(u)
                  fctggg(v)= fctggg(v)-fct10vu
                  fctiix(v)= fctiix(v)                               &
                            +fct09vu*vux+fct10vu*fctisg(v)*fctqsx(v)
                  fctiiy(v)= fctiiy(v)                               &
                            +fct09vu*vuy+fct10vu*fctisg(v)*fctqsy(v)
                  fctiiz(v)= fctiiz(v)                               &
                            +fct09vu*vuz+fct10vu*fctisg(v)*fctqsz(v)
               endif
            enddo
         enddo

         ! Image Screened Interactions
         ! ---------------------------
         do i=reanat+1,imanat
            if (i == reanat+1) then
               a=1
            else
               a=fct4ll(i-1)+1
            endif
            b=fct4ll(i)
            do j=a,b
               u=i
               v=fct4lb(j)
               vux=x(v)-x(u)
               vuy=y(v)-y(u)
               vuz=z(v)-z(u)
               fctdvu=vux*vux+vuy*vuy+vuz*vuz
               u=imattr(i)
               if (fctdvu < fct2cn) then
                  fct01vu  = one/fctdvu
                  fct02vu  = one-fctdvu*fct2ci
                  fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
                  fct04vu  = dexp(-fctikp*fct03vu)
                  fct05vu  = fct04vu/fct03vu
                  fct06vu  = fct05vu+fctikp*fct04vu
                  fct07vu  = one/(one+fct05vu)
                  fct08vu  =-cg(v)*cg(u)*fcttau*fct02vu              &
                            *sqrt(fct01vu*fct07vu)
                  fct09vu  = (fct01vu*fct02vu*(-one+fct06vu*fct07vu) &
                             +fct2fn)*fct08vu
                  fct10vu  = half*fct02vu*fct06vu*fct07vu*fct08vu
                  fctpl2   = fctpl2+fct02vu*fct08vu
                  ! Atomic Screening Energy Array
                  fctscreene(u) = fctscreene(u)+fct02vu*fct08vu*half
                  fctscreene(v) = fctscreene(v)+fct02vu*fct08vu*half
                  ! ---
                  fctggg(u)= fctggg(u)-fct10vu
                  fctiix(u)= fctiix(u)                               &
                            -fct09vu*vux+fct10vu*fctisg(u)*fctqsx(u)
                  fctiiy(u)= fctiiy(u)                               &
                            -fct09vu*vuy+fct10vu*fctisg(u)*fctqsy(u)
                  fctiiz(u)= fctiiz(u)                               &
                            -fct09vu*vuz+fct10vu*fctisg(u)*fctqsz(u)
                  fctggg(v)= fctggg(v)-fct10vu
                  fctiix(v)= fctiix(v)                               &
                            +fct09vu*vux+fct10vu*fctisg(v)*fctqsx(v)
                  fctiiy(v)= fctiiy(v)                               &
                            +fct09vu*vuy+fct10vu*fctisg(v)*fctqsy(v)
                  fctiiz(v)= fctiiz(v)                               &
                            +fct09vu*vuz+fct10vu*fctisg(v)*fctqsz(v)
               endif
            enddo
         enddo
      endif  ! (doscreen)
! ==========================================
#if KEY_PARALLEL==1
      call gcomb(fctggg,reanat)
#endif 
! ==========================================
         do i=1,reanat
            ! Original
            ! fcthhh(i)= fctcgslt(i)+fctnpc(i)-fctisg(i)*fctggg(i)+fctvdwdf(i)
            fcthhh(i)= fctcgslt(i)+fctnpc(i)-fctisg(i)*fctggg(i)
         enddo

         do i=1,reanat
            if (i == 1) then
               a=1
            else
               a=fct1ll(i-1)+1
            endif
            b=fct1ll(i)
            do j=a,b
               u=i
               v=fct1lb(j)
               vux=x(v)-x(u)
               vuy=y(v)-y(u)
               vuz=z(v)-z(u)
               fctdvu=vux*vux+vuy*vuy+vuz*vuz
               if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                   (fctdvu < fct1cn(fctidx(v)))) then
                  fctivu=one/fctdvu
                  fctsvu=sqrt(fctivu)
               endif
               if (fctdvu < fct1cn(fctidx(u))) then
                  fct03vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fctivu
                  fct04vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fctsvu*fctivu
                  fct05vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fct1fn(fctidx(u))
                  fct06vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fct1fn(fctidx(u))              &
                             *fctsvu
                  fct07vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fct1fn(fctidx(u))              &
                             *fctivu
                  fct02vuxs = fct05vuss*vux
                  fct02vuys = fct05vuss*vuy
                  fct02vuzs = fct05vuss*vuz
                  fct03vuxs = (-fct04vuss+fct06vuss)*vux
                  fct03vuys = (-fct04vuss+fct06vuss)*vuy
                  fct03vuzs = (-fct04vuss+fct06vuss)*vuz
                  fct01vuxx = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
                  fct01vuxy = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
                  fct01vuxz = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
                  fct01vuyy = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
                  fct01vuyz = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
                  fct01vuzz = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

                  fct04vuxs =(fct03vuss *fct01xs(u)          &
                             +fct01xs(u)*fct01vuxx           &
                             +fct01ys(u)*fct01vuxy           &
                             +fct01zs(u)*fct01vuxz)          &
                             *fctis1(u) *fctis2(u)           &
                             -fct03vuxs *fctis2(u)*fctmos(u)
                  fct04vuys =(fct03vuss *fct01ys(u)          &
                             +fct01xs(u)*fct01vuxy           &
                             +fct01ys(u)*fct01vuyy           &
                             +fct01zs(u)*fct01vuyz)          &
                             *fctis1(u) *fctis2(u)           &
                             -fct03vuys *fctis2(u)*fctmos(u)
                  fct04vuzs =(fct03vuss *fct01zs(u)          &
                             +fct01xs(u)*fct01vuxz           &
                             +fct01ys(u)*fct01vuyz           &
                             +fct01zs(u)*fct01vuzz)          &
                             *fctis1(u) *fctis2(u)           &
                             -fct03vuzs *fctis2(u)*fctmos(u)

                  fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
                  fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
                  fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
                  fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
                  fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
                  fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)

                  fctsix(v) = fctsix(v)          &
                             +fcthhh(u)*fctqsvux &
                             +fctvsg(u)*fctusvux
                  fctsiy(v) = fctsiy(v)          &
                             +fcthhh(u)*fctqsvuy &
                             +fctvsg(u)*fctusvuy
                  fctsiz(v) = fctsiz(v)          &
                             +fcthhh(u)*fctqsvuz &
                             +fctvsg(u)*fctusvuz
               endif
               if (fctdvu < fct1cn(fctidx(v))) then
                  fct03uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fctivu
                  fct04uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fctsvu*fctivu
                  fct05uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fct1fn(fctidx(v))
                  fct06uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fct1fn(fctidx(v))              &
                             *fctsvu
                  fct07uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fct1fn(fctidx(v))              &
                             *fctivu
                  fct02uvxs =-fct05uvss*vux
                  fct02uvys =-fct05uvss*vuy
                  fct02uvzs =-fct05uvss*vuz
                  fct03uvxs =-(-fct04uvss+fct06uvss)*vux
                  fct03uvys =-(-fct04uvss+fct06uvss)*vuy
                  fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
                  fct01uvxx = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
                  fct01uvxy = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
                  fct01uvxz = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
                  fct01uvyy = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
                  fct01uvyz = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
                  fct01uvzz = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

                  fct04uvxs =(fct03uvss *fct01xs(v)          &
                             +fct01xs(v)*fct01uvxx           &
                             +fct01ys(v)*fct01uvxy           &
                             +fct01zs(v)*fct01uvxz)          &
                             *fctis1(v) *fctis2(v)           &
                             -fct03uvxs *fctis2(v)*fctmos(v)
                  fct04uvys =(fct03uvss *fct01ys(v)          &
                             +fct01xs(v)*fct01uvxy           &
                             +fct01ys(v)*fct01uvyy           &
                             +fct01zs(v)*fct01uvyz)          &
                             *fctis1(v) *fctis2(v)           &
                             -fct03uvys *fctis2(v)*fctmos(v)
                  fct04uvzs =(fct03uvss *fct01zs(v)          &
                             +fct01xs(v)*fct01uvxz           &
                             +fct01ys(v)*fct01uvyz           &
                             +fct01zs(v)*fct01uvzz)          &
                             *fctis1(v) *fctis2(v)           &
                             -fct03uvzs *fctis2(v)*fctmos(v)

                  fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
                  fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
                  fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
                  fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
                  fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
                  fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)

                  fctsix(u) = fctsix(u)          &
                             +fcthhh(v)*fctqsuvx &
                             +fctvsg(v)*fctusuvx
                  fctsiy(u) = fctsiy(u)          &
                             +fcthhh(v)*fctqsuvy &
                             +fctvsg(v)*fctusuvy
                  fctsiz(u) = fctsiz(u)          &
                             +fcthhh(v)*fctqsuvz &
                             +fctvsg(v)*fctusuvz
               endif
            enddo
         enddo

         do i=reanat+1,imanat
            if (i == reanat+1) then
               a=1
            else
               a=fct3ll(i-1)+1
            endif
            b=fct3ll(i)
            do j=a,b
               u=i
               v=fct3lb(j)
               vux=x(v)-x(u)
               vuy=y(v)-y(u)
               vuz=z(v)-z(u)
               fctdvu=vux*vux+vuy*vuy+vuz*vuz
               u=imattr(i)
               if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                   (fctdvu < fct1cn(fctidx(v)))) then
                  fctivu=one/fctdvu
                  fctsvu=sqrt(fctivu)
               endif
               if (fctdvu < fct1cn(fctidx(u))) then
                  fct03vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fctivu
                  fct04vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fctsvu*fctivu
                  fct05vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fct1fn(fctidx(u))
                  fct06vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fct1fn(fctidx(u))              &
                             *fctsvu
                  fct07vuss = fctvol(fctidx(v))              &
                             *(one-fctdvu*fct1ci(fctidx(u))) &
                             *fct1fn(fctidx(u))              &
                             *fctivu
                  fct02vuxs = fct05vuss*vux
                  fct02vuys = fct05vuss*vuy
                  fct02vuzs = fct05vuss*vuz
                  fct03vuxs = (-fct04vuss+fct06vuss)*vux
                  fct03vuys = (-fct04vuss+fct06vuss)*vuy
                  fct03vuzs = (-fct04vuss+fct06vuss)*vuz
                  fct01vuxx = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
                  fct01vuxy = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
                  fct01vuxz = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
                  fct01vuyy = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
                  fct01vuyz = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
                  fct01vuzz = half*fct07vuss                         &
                             *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

                  fct04vuxs =(fct03vuss *fct01xs(u)          &
                             +fct01xs(u)*fct01vuxx           &
                             +fct01ys(u)*fct01vuxy           &
                             +fct01zs(u)*fct01vuxz)          &
                             *fctis1(u) *fctis2(u)           &
                             -fct03vuxs *fctis2(u)*fctmos(u)
                  fct04vuys =(fct03vuss *fct01ys(u)          &
                             +fct01xs(u)*fct01vuxy           &
                             +fct01ys(u)*fct01vuyy           &
                             +fct01zs(u)*fct01vuyz)          &
                             *fctis1(u) *fctis2(u)           &
                             -fct03vuys *fctis2(u)*fctmos(u)
                  fct04vuzs =(fct03vuss *fct01zs(u)          &
                             +fct01xs(u)*fct01vuxz           &
                             +fct01ys(u)*fct01vuyz           &
                             +fct01zs(u)*fct01vuzz)          &
                             *fctis1(u) *fctis2(u)           &
                             -fct03vuzs *fctis2(u)*fctmos(u)

                  fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
                  fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
                  fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
                  fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
                  fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
                  fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)

                  fctsix(v) = fctsix(v)          &
                             +fcthhh(u)*fctqsvux &
                             +fctvsg(u)*fctusvux
                  fctsiy(v) = fctsiy(v)          &
                             +fcthhh(u)*fctqsvuy &
                             +fctvsg(u)*fctusvuy
                  fctsiz(v) = fctsiz(v)          &
                             +fcthhh(u)*fctqsvuz &
                             +fctvsg(u)*fctusvuz
               endif
               if (fctdvu < fct1cn(fctidx(v))) then
                  fct03uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fctivu
                  fct04uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fctsvu*fctivu
                  fct05uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fct1fn(fctidx(v))
                  fct06uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fct1fn(fctidx(v))              &
                             *fctsvu
                  fct07uvss = fctvol(fctidx(u))              &
                             *(one-fctdvu*fct1ci(fctidx(v))) &
                             *fct1fn(fctidx(v))              &
                             *fctivu
                  fct02uvxs =-fct05uvss*vux
                  fct02uvys =-fct05uvss*vuy
                  fct02uvzs =-fct05uvss*vuz
                  fct03uvxs =-(-fct04uvss+fct06uvss)*vux
                  fct03uvys =-(-fct04uvss+fct06uvss)*vuy
                  fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
                  fct01uvxx = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
                  fct01uvxy = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
                  fct01uvxz = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
                  fct01uvyy = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
                  fct01uvyz = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
                  fct01uvzz = half*fct07uvss                         &
                             *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

                  fct04uvxs =(fct03uvss *fct01xs(v)          &
                             +fct01xs(v)*fct01uvxx           &
                             +fct01ys(v)*fct01uvxy           &
                             +fct01zs(v)*fct01uvxz)          &
                             *fctis1(v) *fctis2(v)           &
                             -fct03uvxs *fctis2(v)*fctmos(v)
                  fct04uvys =(fct03uvss *fct01ys(v)          &
                             +fct01xs(v)*fct01uvxy           &
                             +fct01ys(v)*fct01uvyy           &
                             +fct01zs(v)*fct01uvyz)          &
                             *fctis1(v) *fctis2(v)           &
                             -fct03uvys *fctis2(v)*fctmos(v)
                  fct04uvzs =(fct03uvss *fct01zs(v)          &
                             +fct01xs(v)*fct01uvxz           &
                             +fct01ys(v)*fct01uvyz           &
                             +fct01zs(v)*fct01uvzz)          &
                             *fctis1(v) *fctis2(v)           &
                             -fct03uvzs *fctis2(v)*fctmos(v)

                  fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
                  fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
                  fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
                  fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
                  fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
                  fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)

                  fctsix(u) = fctsix(u)          &
                             +fcthhh(v)*fctqsuvx &
                             +fctvsg(v)*fctusuvx
                  fctsiy(u) = fctsiy(u)          &
                             +fcthhh(v)*fctqsuvy &
                             +fctvsg(v)*fctusuvy
                  fctsiz(u) = fctsiz(u)          &
                             +fcthhh(v)*fctqsuvz &
                             +fctvsg(v)*fctusuvz
               endif
            enddo
         enddo
! ---------------------------------------------------------------------
      else ! (ifast)
! Now slow part
! ---------------------------------------------------------------------
         do i=1,reanat
            fcttp1   = sqrt( fct01xs(i)*fct01xs(i)  &
                            +fct01ys(i)*fct01ys(i)  &
                            +fct01zs(i)*fct01zs(i))
            fcttp2   = fct02ss(i)

            fctis1(i)= one/fcttp1
            fctis2(i)= one/fcttp2

            fctmov(i)= fct01ss(i)
            fctmos(i)= fcttp1*fctis2(i)

            fcttp1   =                   fctmov(i)           &
                      +fctcb1(fctidx(i))          *fctmos(i) &
                      +fctcb2(fctidx(i))*fctmov(i)*fctmos(i) &
                      +fctqos
            fcttp2   = one                                   &
                      +fctcb2(fctidx(i))          *fctmos(i)
            fcttp3   = fctcb1(fctidx(i))                     &
                      +fctcb2(fctidx(i))*fctmov(i)
            fcttp4   = fctca2(fctidx(i))*(fcttp1-fctca3(fctidx(i)))
            fcttp5   = dexp(-fcttp4)
            fcttp6   = one/(one+fcttp5)
            fcttp7   = fctca0(fctidx(i))                     &
                      +fctca1(fctidx(i))*fcttp6
            fcttp8   = fctca1(fctidx(i))*fctca2(fctidx(i))   &
                      /(one/fcttp5+two+fcttp5)
            fctqsg(i)= fcttp7
            fctqdv(i)= fcttp2*fcttp8
            fctqds(i)= fcttp3*fcttp8
            if (fctqsg(i) > -0.001d0) then
               call wrndie(-4,'<FCTENE>','Solvation energy too high.')
            endif
            fctqsg(i)= fctqsg(i)*fctqun(fctidx(i))
            fctqdv(i)= fctqdv(i)*fctqun(fctidx(i))
            fctqds(i)= fctqds(i)*fctqun(fctidx(i))
            ! SASA Sigmoidal
            fcttp1   =                   fctmov(i)           &
                      +fctcd1(fctidx(i))          *fctmos(i) &
                      +fctcd2(fctidx(i))*fctmov(i)*fctmos(i) &
                      +fctuos
            fcttp2   = one                                   &
                      +fctcd2(fctidx(i))          *fctmos(i)
            fcttp3   = fctcd1(fctidx(i))                     &
                      +fctcd2(fctidx(i))*fctmov(i)
            fcttp4   = fctcc2(fctidx(i))*(fcttp1-fctcc3(fctidx(i)))
            fcttp5   = dexp(-fcttp4)
            fcttp6   = one/(one+fcttp5)
            fcttp7   = fctcc0(fctidx(i))                     &
                      +fctcc1(fctidx(i))*fcttp6
            fcttp8   = fctcc1(fctidx(i))*fctcc2(fctidx(i))   &
                      /(one/fcttp5+two+fcttp5)
            fctusg(i)= fcttp7
            fctudv(i)= fcttp2*fcttp8
            fctuds(i)= fcttp3*fcttp8
            if (fctusg(i) < 0.001d0) then
               call wrndie(-4,'<FCTENE>','Surface too low.')
            endif
            ! Surface Tension Sigmoidal
            ! e0 == gamma [kcal/(A^2*mol)]
            fcttp1   =                   fctmov(i)           &
                      +fctcf1(fctidx(i))          *fctmos(i) &
                      +fctcf2(fctidx(i))*fctmov(i)*fctmos(i) &
                      +fctvos
            fcttp2   = one                                   &
                      +fctcf2(fctidx(i))          *fctmos(i)
            fcttp3   = fctcf1(fctidx(i))                     &
                      +fctcf2(fctidx(i))*fctmov(i)
            fcttp4   = fctce2(fctidx(i))*(fcttp1-fctce3(fctidx(i)))
            fcttp5   = dexp(-fcttp4)
            fcttp6   = one/(one+fcttp5)
            fcttp7   = fctce0(fctidx(i))                     &
                      +fctce1(fctidx(i))*fcttp6
            fcttp8   = fctce1(fctidx(i))*fctce2(fctidx(i))   &
                      /(one/fcttp5+two+fcttp5)
            fctvsg(i)= fcttp7
            fctvdv(i)= fcttp2*fcttp8
            fctvds(i)= fcttp3*fcttp8
            ! A.P. Volumetric Correction Sigmoidal
            ! g0, g1, g2 ,g3
            fcttp1   =                   fctmov(i)           &
                      +fctch1(fctidx(i))          *fctmos(i) &
                      +fctch2(fctidx(i))*fctmov(i)*fctmos(i) &
                      +fctwos
            fcttp2   = one                                   &
                      +fctch2(fctidx(i))          *fctmos(i)
            fcttp3   = fctch1(fctidx(i))                     &
                      +fctch2(fctidx(i))*fctmov(i)
            fcttp4   = fctcg2(fctidx(i))*(fcttp1-fctcg3(fctidx(i)))
            fcttp5   = dexp(-fcttp4)
            fcttp6   = one/(one+fcttp5)
            fcttp7   = fctcg0(fctidx(i))                     &
                      +fctcg1(fctidx(i))*fcttp6
            fcttp8   = fctcg1(fctidx(i))*fctcg2(fctidx(i))   &
                      /(one/fcttp5+two+fcttp5)
            fctwsg(i)= fcttp7
            fctwdv(i)= fcttp2*fcttp8
            fctwds(i)= fcttp3*fcttp8

            fctisg(i)= one/fctqsg(i)
            ! VdW TERM ===================================================
            fctvdw01    = fcttau*fctisg(i)/two
            fctvdw02    = one/(-fctvdw01+fctvdwwrd)
            fctvdwen(i) = fctvdwfac(i)*fctvdw02**3
            fctvdwdf(i) =-three*fctvdwen(i)*fctvdw02*fctvdw01*fctisg(i)
            ! Salt TERM ==================================================
            ! GBSW like
            ! ---------
            fctslt01    = half*fctscreen*fcttau*fctisg(i)
            fctslt02    = fcties*dexp(fctslt01)
            fctslt03    = (fctiem-fctslt02)*fcttai

            fctqsgslt   = fctslt03*fctqsg(i)
            fctcgslt(i) = (fctiem+(fctslt01-one)*fctslt02)*fcttai*fctcgs(i)
            ! ============================================================
            fct04xs  =(fct03ss(i)*fct01xs(i)          &
                      +fct01xs(i)*fct01xx(i)          &
                      +fct01ys(i)*fct01xy(i)          &
                      +fct01zs(i)*fct01xz(i))         &
                      *fctis1(i) *fctis2(i)           &
                      -fct03xs(i)*fctis2(i)*fctmos(i)
            fct04ys  =(fct03ss(i)*fct01ys(i)          &
                      +fct01xs(i)*fct01xy(i)          &
                      +fct01ys(i)*fct01yy(i)          &
                      +fct01zs(i)*fct01yz(i))         &
                      *fctis1(i) *fctis2(i)           &
                      -fct03ys(i)*fctis2(i)*fctmos(i)
            fct04zs  =(fct03ss(i)*fct01zs(i)          &
                      +fct01xs(i)*fct01xz(i)          &
                      +fct01ys(i)*fct01yz(i)          &
                      +fct01zs(i)*fct01zz(i))         &
                      *fctis1(i) *fctis2(i)           &
                      -fct03zs(i)*fctis2(i)*fctmos(i)

            fctqsx(i)=-fct02xs(i)*fctqdv(i)-fct04xs*fctqds(i)
            fctqsy(i)=-fct02ys(i)*fctqdv(i)-fct04ys*fctqds(i)
            fctqsz(i)=-fct02zs(i)*fctqdv(i)-fct04zs*fctqds(i)
            fctusx(i)=-fct02xs(i)*fctudv(i)-fct04xs*fctuds(i)
            fctusy(i)=-fct02ys(i)*fctudv(i)-fct04ys*fctuds(i)
            fctusz(i)=-fct02zs(i)*fctudv(i)-fct04zs*fctuds(i)
            ! May be put in slow
            fctvsx(i)=-fct02xs(i)*fctvdv(i)-fct04xs*fctvds(i)
            fctvsy(i)=-fct02ys(i)*fctvdv(i)-fct04ys*fctvds(i)
            fctvsz(i)=-fct02zs(i)*fctvdv(i)-fct04zs*fctvds(i)
            fctwsx(i)=-fct02xs(i)*fctwdv(i)-fct04xs*fctwds(i)
            fctwsy(i)=-fct02ys(i)*fctwdv(i)-fct04ys*fctwds(i)
            fctwsz(i)=-fct02zs(i)*fctwdv(i)-fct04zs*fctwds(i)
! ==========================================
#if KEY_PARALLEL==1
         if(mynod==0)then
#endif 
! ==========================================
            if(imove(i)==0) then
               ! fctpl1   = fctpl1+fctcgs(i)*fctqsgslt
               fctself(i)= fctcgs(i)*fctqsgslt

               fctnpol(i) = fctnpc(i)*fctqsg(i)+fctusg(i)*fctvsg(i)+fctwsg(i)+fctvdwen(i)
               ! fctnp1   = fctnp1+fctnpc(i)*fctqsg(i)
               ! fctnp2   = fctnp2+fctusg(i)*fctvsg(i)
               ! fctnp3   = fctnp3+fctwsg(i)+fctvdwen(i)

               fctssx(i)=  fctssx(i)                                    &
                         +(fctvdwdf(i)+fctcgslt(i)+fctnpc(i))*fctqsx(i) &
                         +                         fctvsg(i) *fctusx(i) &
                         +                         fctusg(i) *fctvsx(i) &
                         +                                    fctwsx(i)
               fctssy(i)=  fctssy(i)                                    &
                         +(fctvdwdf(i)+fctcgslt(i)+fctnpc(i))*fctqsy(i) &
                         +                         fctvsg(i) *fctusy(i) &
                         +                         fctusg(i) *fctvsy(i) &
                         +                                    fctwsy(i)
               fctssz(i)=  fctssz(i)                                    &
                         +(fctvdwdf(i)+fctcgslt(i)+fctnpc(i))*fctqsz(i) &
                         +                         fctvsg(i) *fctusz(i) &
                         +                         fctusg(i) *fctvsz(i) &
                         +                                    fctwsz(i)
            endif
! ==========================================
#if KEY_PARALLEL==1
         endif
#endif 
! ==========================================
         enddo

      if(doscreen)then  ! (doscreen)
         ! Primary Atoms Screened Interactions
         ! -----------------------------------
         do i=1,reanat
            if (i == 1) then
               a=1
            else
               a=fct2ll(i-1)+1
            endif
            b=fct2ll(i)
            do j=a,b
               u=i
               v=fct2lb(j)
               vux=x(v)-x(u)
               vuy=y(v)-y(u)
               vuz=z(v)-z(u)
               fctdvu=vux*vux+vuy*vuy+vuz*vuz
               if (fctdvu < fct2cn) then
                  ! Salt
                  fct01vu  = one/fctdvu
                  fct02vu  = one-fctdvu*fct2ci
                  fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
                  fct04vu  = dexp(-fctikp*fct03vu)
                  fct05vu  = fct04vu/fct03vu
                  fct06vu  = fct05vu+fctikp*fct04vu
                  fct07vu  = one/(one+fct05vu)

                  ! 1/f_GB
                  fct08vu  = sqrt(fct01vu*fct07vu)
                  fct09vu  = -cg(v)*cg(u)*fct02vu*fct08vu

                  ! fcties = 1. / 78.5
                  fct10vu = dexp(-fctscreen/fct08vu)*fcties
                  fct11vu = (fctiem-fct10vu)*fct09vu*fct02vu

                  ! fct12vu: (xi-xj) components
                  fct12vu = fct09vu*(fct08vu*(fctikp*fct04vu-one)        &
                           *fct02vu                                      &
                           *(fct08vu*(fctiem-fct10vu)-fctscreen*fct10vu) &
                           +(fctiem-fct10vu)*fct2fn)
                  ! fct13vu: (QSXi) components
                  fct13vu = half*fct09vu*fct02vu*fct08vu*fctdvu          &
                           *fct06vu*(fct08vu*(fctiem-fct10vu)            &
                           -fctscreen*fct10vu)

                  fctpl2   = fctpl2+fct11vu
                  ! Atomic Screening Energy Array
                  fctscreene(u) = fctscreene(u)+fct11vu*half
                  fctscreene(v) = fctscreene(v)+fct11vu*half
                  ! ---
                  fctggg(u)= fctggg(u)-fct13vu
                  fctggg(v)= fctggg(v)-fct13vu
                  fctiix(u)= fctiix(u)                               &
                            -fct12vu*vux+fct13vu*fctisg(u)*fctqsx(u)
                  fctiiy(u)= fctiiy(u)                               &
                            -fct12vu*vuy+fct13vu*fctisg(u)*fctqsy(u)
                  fctiiz(u)= fctiiz(u)                               &
                            -fct12vu*vuz+fct13vu*fctisg(u)*fctqsz(u)
                  fctiix(v)= fctiix(v)                               &
                            +fct12vu*vux+fct13vu*fctisg(v)*fctqsx(v)
                  fctiiy(v)= fctiiy(v)                               &
                            +fct12vu*vuy+fct13vu*fctisg(v)*fctqsy(v)
                  fctiiz(v)= fctiiz(v)                               &
                            +fct12vu*vuz+fct13vu*fctisg(v)*fctqsz(v)
               endif
            enddo
         enddo

         do i=reanat+1,imanat
            if (i == reanat+1) then
               a=1
            else
               a=fct4ll(i-1)+1
            endif
            b=fct4ll(i)
            do j=a,b
               u=i
               v=fct4lb(j)
               vux=x(v)-x(u)
               vuy=y(v)-y(u)
               vuz=z(v)-z(u)
               fctdvu=vux*vux+vuy*vuy+vuz*vuz
               u=imattr(i)
               if (fctdvu < fct2cn) then
                  ! Salt
                  fct01vu  = one/fctdvu
                  fct02vu  = one-fctdvu*fct2ci
                  fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
                  fct04vu  = dexp(-fctikp*fct03vu)
                  fct05vu  = fct04vu/fct03vu
                  fct06vu  = fct05vu+fctikp*fct04vu
                  fct07vu  = one/(one+fct05vu)

                  ! 1/f_GB
                  fct08vu  = sqrt(fct01vu*fct07vu)
                  fct09vu  = -cg(v)*cg(u)*fct02vu*fct08vu

                  ! fcties = 1. / 78.5
                  fct10vu = dexp(-fctscreen/fct08vu)*fcties
                  fct11vu = (fctiem-fct10vu)*fct09vu*fct02vu

                  ! fct12vu: (xi-xj) components
                  fct12vu = fct09vu*(fct08vu*(fctikp*fct04vu-one)        &
                           *fct02vu                                      &
                           *(fct08vu*(fctiem-fct10vu)-fctscreen*fct10vu) &
                           +(fctiem-fct10vu)*fct2fn)
                  ! fct13vu: (QSXi) components
                  fct13vu = half*fct09vu*fct02vu*fct08vu*fctdvu          &
                           *fct06vu*(fct08vu*(fctiem-fct10vu)            &
                           -fctscreen*fct10vu)

                  fctpl2   = fctpl2+fct11vu
                  ! Atomic Screening Energy Array
                  fctscreene(u) = fctscreene(u)+fct11vu*half
                  fctscreene(v) = fctscreene(v)+fct11vu*half
                  ! ---
                  fctggg(u)= fctggg(u)-fct13vu
                  fctggg(v)= fctggg(v)-fct13vu
                  fctiix(u)= fctiix(u)                               &
                            -fct12vu*vux+fct13vu*fctisg(u)*fctqsx(u)
                  fctiiy(u)= fctiiy(u)                               &
                            -fct12vu*vuy+fct13vu*fctisg(u)*fctqsy(u)
                  fctiiz(u)= fctiiz(u)                               &
                            -fct12vu*vuz+fct13vu*fctisg(u)*fctqsz(u)
                  fctiix(v)= fctiix(v)                               &
                            +fct12vu*vux+fct13vu*fctisg(v)*fctqsx(v)
                  fctiiy(v)= fctiiy(v)                               &
                            +fct12vu*vuy+fct13vu*fctisg(v)*fctqsy(v)
                  fctiiz(v)= fctiiz(v)                               &
                            +fct12vu*vuz+fct13vu*fctisg(v)*fctqsz(v)
               endif
            enddo
         enddo
      endif  ! (doscreen)
! ==========================================
#if KEY_PARALLEL==1
      call gcomb(fctggg,reanat)
#endif 
! ==========================================
      do i=1,reanat
         ! Original
         ! fcthhh(i)= fctcgslt(i)+fctnpc(i)-fctisg(i)*fctggg(i)+fctvdwdf(i)
         fctkkk(i)=-fctisg(i)*fctggg(i)
         fcthhh(i)= fctcgslt(i)+fctnpc(i)+fctvdwdf(i)
      enddo

         do i=1,reanat
            if (i == 1) then
               a=1
            else
               a=fct1ll(i-1)+1
            endif
            b=fct1ll(i)
            do j=a,b
               u=i
               v=fct1lb(j)
               vux=x(v)-x(u)
               vuy=y(v)-y(u)
               vuz=z(v)-z(u)
               fctdvu=vux*vux+vuy*vuy+vuz*vuz
               if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                   (fctdvu < fct1cn(fctidx(v)))) then
                  fctivu=one/fctdvu
                  fctsvu=sqrt(fctivu)
               endif
               if (fctdvu < fct1cn(fctidx(u))) then
                  if(imove(v)==0) then
                     fct03vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fctivu
                     fct04vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fctsvu*fctivu
                     fct05vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fct1fn(fctidx(u))
                     fct06vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fct1fn(fctidx(u))              &
                                *fctsvu
                     fct07vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fct1fn(fctidx(u))              &
                                *fctivu
                     fct02vuxs = fct05vuss*vux
                     fct02vuys = fct05vuss*vuy
                     fct02vuzs = fct05vuss*vuz
                     fct03vuxs = (-fct04vuss+fct06vuss)*vux
                     fct03vuys = (-fct04vuss+fct06vuss)*vuy
                     fct03vuzs = (-fct04vuss+fct06vuss)*vuz
                     fct01vuxx = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
                     fct01vuxy = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
                     fct01vuxz = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
                     fct01vuyy = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
                     fct01vuyz = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
                     fct01vuzz = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

                     fct04vuxs =(fct03vuss *fct01xs(u)          &
                                +fct01xs(u)*fct01vuxx           &
                                +fct01ys(u)*fct01vuxy           &
                                +fct01zs(u)*fct01vuxz)          &
                                *fctis1(u) *fctis2(u)           &
                                -fct03vuxs *fctis2(u)*fctmos(u)
                     fct04vuys =(fct03vuss *fct01ys(u)          &
                                +fct01xs(u)*fct01vuxy           &
                                +fct01ys(u)*fct01vuyy           &
                                +fct01zs(u)*fct01vuyz)          &
                                *fctis1(u) *fctis2(u)           &
                                -fct03vuys *fctis2(u)*fctmos(u)
                     fct04vuzs =(fct03vuss *fct01zs(u)          &
                                +fct01xs(u)*fct01vuxz           &
                                +fct01ys(u)*fct01vuyz           &
                                +fct01zs(u)*fct01vuzz)          &
                                *fctis1(u) *fctis2(u)           &
                                -fct03vuzs *fctis2(u)*fctmos(u)

                     fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
                     fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
                     fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
                     fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
                     fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
                     fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)
                     ! May be put in slow
                     fctvsvux  = fct02vuxs*fctvdv(u)+fct04vuxs*fctvds(u)
                     fctvsvuy  = fct02vuys*fctvdv(u)+fct04vuys*fctvds(u)
                     fctvsvuz  = fct02vuzs*fctvdv(u)+fct04vuzs*fctvds(u)
                     fctwsvux  = fct02vuxs*fctwdv(u)+fct04vuxs*fctwds(u)
                     fctwsvuy  = fct02vuys*fctwdv(u)+fct04vuys*fctwds(u)
                     fctwsvuz  = fct02vuzs*fctwdv(u)+fct04vuzs*fctwds(u)

                     fctsix(v) = fctsix(v)          &
                                +fctkkk(u)*fctqsvux
                     fctsiy(v) = fctsiy(v)          &
                                +fctkkk(u)*fctqsvuy
                     fctsiz(v) = fctsiz(v)          &
                                +fctkkk(u)*fctqsvuz

                     if(imove(u)==0) then
                        fctsix(v) = fctsix(v)          &
                                   +fcthhh(u)*fctqsvux &
                                   +fctvsg(u)*fctusvux &
                                   +fctusg(u)*fctvsvux &
                                   +          fctwsvux
                        fctsiy(v) = fctsiy(v)          &
                                   +fcthhh(u)*fctqsvuy &
                                   +fctvsg(u)*fctusvuy &
                                   +fctusg(u)*fctvsvuy &
                                   +          fctwsvuy
                        fctsiz(v) = fctsiz(v)          &
                                   +fcthhh(u)*fctqsvuz &
                                   +fctvsg(u)*fctusvuz &
                                   +fctusg(u)*fctvsvuz &
                                   +          fctwsvuz
                     endif
                  endif
               endif
               if (fctdvu < fct1cn(fctidx(v))) then
                  if(imove(u)==0) then
                     fct03uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fctivu
                     fct04uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fctsvu*fctivu
                     fct05uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fct1fn(fctidx(v))
                     fct06uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fct1fn(fctidx(v))              &
                                *fctsvu
                     fct07uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fct1fn(fctidx(v))              &
                                *fctivu
                     fct02uvxs =-fct05uvss*vux
                     fct02uvys =-fct05uvss*vuy
                     fct02uvzs =-fct05uvss*vuz
                     fct03uvxs =-(-fct04uvss+fct06uvss)*vux
                     fct03uvys =-(-fct04uvss+fct06uvss)*vuy
                     fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
                     fct01uvxx = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
                     fct01uvxy = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
                     fct01uvxz = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
                     fct01uvyy = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
                     fct01uvyz = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
                     fct01uvzz = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

                     fct04uvxs =(fct03uvss *fct01xs(v)          &
                                +fct01xs(v)*fct01uvxx           &
                                +fct01ys(v)*fct01uvxy           &
                                +fct01zs(v)*fct01uvxz)          &
                                *fctis1(v) *fctis2(v)           &
                                -fct03uvxs *fctis2(v)*fctmos(v)
                     fct04uvys =(fct03uvss *fct01ys(v)          &
                                +fct01xs(v)*fct01uvxy           &
                                +fct01ys(v)*fct01uvyy           &
                                +fct01zs(v)*fct01uvyz)          &
                                *fctis1(v) *fctis2(v)           &
                                -fct03uvys *fctis2(v)*fctmos(v)
                     fct04uvzs =(fct03uvss *fct01zs(v)          &
                                +fct01xs(v)*fct01uvxz           &
                                +fct01ys(v)*fct01uvyz           &
                                +fct01zs(v)*fct01uvzz)          &
                                *fctis1(v) *fctis2(v)           &
                                -fct03uvzs *fctis2(v)*fctmos(v)

                     fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
                     fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
                     fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
                     fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
                     fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
                     fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)
                     fctvsuvx  = fct02uvxs*fctvdv(v)+fct04uvxs*fctvds(v)
                     fctvsuvy  = fct02uvys*fctvdv(v)+fct04uvys*fctvds(v)
                     fctvsuvz  = fct02uvzs*fctvdv(v)+fct04uvzs*fctvds(v)
                     fctwsuvx  = fct02uvxs*fctwdv(v)+fct04uvxs*fctwds(v)
                     fctwsuvy  = fct02uvys*fctwdv(v)+fct04uvys*fctwds(v)
                     fctwsuvz  = fct02uvzs*fctwdv(v)+fct04uvzs*fctwds(v)

                     fctsix(u) = fctsix(u)          &
                                +fctkkk(v)*fctqsuvx
                     fctsiy(u) = fctsiy(u)          &
                                +fctkkk(v)*fctqsuvy
                     fctsiz(u) = fctsiz(u)          &
                                +fctkkk(v)*fctqsuvz

                     if(imove(v)==0) then
                        fctsix(u) = fctsix(u)          &
                                   +fcthhh(v)*fctqsuvx &
                                   +fctvsg(v)*fctusuvx &
                                   +fctusg(v)*fctvsuvx &
                                   +          fctwsuvx
                        fctsiy(u) = fctsiy(u)          &
                                   +fcthhh(v)*fctqsuvy &
                                   +fctvsg(v)*fctusuvy &
                                   +fctusg(v)*fctvsuvy &
                                   +          fctwsuvy
                        fctsiz(u) = fctsiz(u)          &
                                   +fcthhh(v)*fctqsuvz &
                                   +fctvsg(v)*fctusuvz &
                                   +fctusg(v)*fctvsuvz &
                                   +          fctwsuvz
                     endif
                  endif
               endif
            enddo
         enddo

         do i=reanat+1,imanat
            if (i == reanat+1) then
               a=1
            else
               a=fct3ll(i-1)+1
            endif
            b=fct3ll(i)
            do j=a,b
               u=i
               v=fct3lb(j)
               vux=x(v)-x(u)
               vuy=y(v)-y(u)
               vuz=z(v)-z(u)
               fctdvu=vux*vux+vuy*vuy+vuz*vuz
               u=imattr(i)
               if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                   (fctdvu < fct1cn(fctidx(v)))) then
                  fctivu=one/fctdvu
                  fctsvu=sqrt(fctivu)
               endif
               if (fctdvu < fct1cn(fctidx(u))) then
                  if(imove(v)==0) then
                     fct03vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fctivu
                     fct04vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fctsvu*fctivu
                     fct05vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fct1fn(fctidx(u))
                     fct06vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fct1fn(fctidx(u))              &
                                *fctsvu
                     fct07vuss = fctvol(fctidx(v))              &
                                *(one-fctdvu*fct1ci(fctidx(u))) &
                                *fct1fn(fctidx(u))              &
                                *fctivu
                     fct02vuxs = fct05vuss*vux
                     fct02vuys = fct05vuss*vuy
                     fct02vuzs = fct05vuss*vuz
                     fct03vuxs = (-fct04vuss+fct06vuss)*vux
                     fct03vuys = (-fct04vuss+fct06vuss)*vuy
                     fct03vuzs = (-fct04vuss+fct06vuss)*vuz
                     fct01vuxx = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
                     fct01vuxy = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
                     fct01vuxz = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
                     fct01vuyy = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
                     fct01vuyz = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
                     fct01vuzz = half*fct07vuss                         &
                                *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

                     fct04vuxs =(fct03vuss *fct01xs(u)          &
                                +fct01xs(u)*fct01vuxx           &
                                +fct01ys(u)*fct01vuxy           &
                                +fct01zs(u)*fct01vuxz)          &
                                *fctis1(u) *fctis2(u)           &
                                -fct03vuxs *fctis2(u)*fctmos(u)
                     fct04vuys =(fct03vuss *fct01ys(u)          &
                                +fct01xs(u)*fct01vuxy           &
                                +fct01ys(u)*fct01vuyy           &
                                +fct01zs(u)*fct01vuyz)          &
                                *fctis1(u) *fctis2(u)           &
                                -fct03vuys *fctis2(u)*fctmos(u)
                     fct04vuzs =(fct03vuss *fct01zs(u)          &
                                +fct01xs(u)*fct01vuxz           &
                                +fct01ys(u)*fct01vuyz           &
                                +fct01zs(u)*fct01vuzz)          &
                                *fctis1(u) *fctis2(u)           &
                                -fct03vuzs *fctis2(u)*fctmos(u)

                     fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
                     fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
                     fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
                     fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
                     fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
                     fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)
                     fctvsvux  = fct02vuxs*fctvdv(u)+fct04vuxs*fctvds(u)
                     fctvsvuy  = fct02vuys*fctvdv(u)+fct04vuys*fctvds(u)
                     fctvsvuz  = fct02vuzs*fctvdv(u)+fct04vuzs*fctvds(u)
                     fctwsvux  = fct02vuxs*fctwdv(u)+fct04vuxs*fctwds(u)
                     fctwsvuy  = fct02vuys*fctwdv(u)+fct04vuys*fctwds(u)
                     fctwsvuz  = fct02vuzs*fctwdv(u)+fct04vuzs*fctwds(u)

                     fctsix(v) = fctsix(v)          &
                                +fctkkk(u)*fctqsvux
                     fctsiy(v) = fctsiy(v)          &
                                +fctkkk(u)*fctqsvuy
                     fctsiz(v) = fctsiz(v)          &
                                +fctkkk(u)*fctqsvuz

                     if(imove(u)==0) then
                        fctsix(v) = fctsix(v)          &
                                   +fcthhh(u)*fctqsvux &
                                   +fctvsg(u)*fctusvux &
                                   +fctusg(u)*fctvsvux &
                                   +          fctwsvux
                        fctsiy(v) = fctsiy(v)          &
                                   +fcthhh(u)*fctqsvuy &
                                   +fctvsg(u)*fctusvuy &
                                   +fctusg(u)*fctvsvuy &
                                   +          fctwsvuy
                        fctsiz(v) = fctsiz(v)          &
                                   +fcthhh(u)*fctqsvuz &
                                   +fctvsg(u)*fctusvuz &
                                   +fctusg(u)*fctvsvuz &
                                   +          fctwsvuz
                     endif
                  endif
               endif
               if (fctdvu < fct1cn(fctidx(v))) then
                  if(imove(u)==0) then
                     fct03uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fctivu
                     fct04uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fctsvu*fctivu
                     fct05uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fct1fn(fctidx(v))
                     fct06uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fct1fn(fctidx(v))              &
                                *fctsvu
                     fct07uvss = fctvol(fctidx(u))              &
                                *(one-fctdvu*fct1ci(fctidx(v))) &
                                *fct1fn(fctidx(v))              &
                                *fctivu
                     fct02uvxs =-fct05uvss*vux
                     fct02uvys =-fct05uvss*vuy
                     fct02uvzs =-fct05uvss*vuz
                     fct03uvxs =-(-fct04uvss+fct06uvss)*vux
                     fct03uvys =-(-fct04uvss+fct06uvss)*vuy
                     fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
                     fct01uvxx = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
                     fct01uvxy = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
                     fct01uvxz = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
                     fct01uvyy = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
                     fct01uvyz = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
                     fct01uvzz = half*fct07uvss                         &
                                *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

                     fct04uvxs =(fct03uvss *fct01xs(v)          &
                                +fct01xs(v)*fct01uvxx           &
                                +fct01ys(v)*fct01uvxy           &
                                +fct01zs(v)*fct01uvxz)          &
                                *fctis1(v) *fctis2(v)           &
                                -fct03uvxs *fctis2(v)*fctmos(v)
                     fct04uvys =(fct03uvss *fct01ys(v)          &
                                +fct01xs(v)*fct01uvxy           &
                                +fct01ys(v)*fct01uvyy           &
                                +fct01zs(v)*fct01uvyz)          &
                                *fctis1(v) *fctis2(v)           &
                                -fct03uvys *fctis2(v)*fctmos(v)
                     fct04uvzs =(fct03uvss *fct01zs(v)          &
                                +fct01xs(v)*fct01uvxz           &
                                +fct01ys(v)*fct01uvyz           &
                                +fct01zs(v)*fct01uvzz)          &
                                *fctis1(v) *fctis2(v)           &
                                -fct03uvzs *fctis2(v)*fctmos(v)

                     fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
                     fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
                     fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
                     fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
                     fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
                     fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)
                     fctvsuvx  = fct02uvxs*fctvdv(v)+fct04uvxs*fctvds(v)
                     fctvsuvy  = fct02uvys*fctvdv(v)+fct04uvys*fctvds(v)
                     fctvsuvz  = fct02uvzs*fctvdv(v)+fct04uvzs*fctvds(v)
                     fctwsuvx  = fct02uvxs*fctwdv(v)+fct04uvxs*fctwds(v)
                     fctwsuvy  = fct02uvys*fctwdv(v)+fct04uvys*fctwds(v)
                     fctwsuvz  = fct02uvzs*fctwdv(v)+fct04uvzs*fctwds(v)

                     fctsix(u) = fctsix(u)          &
                                +fctkkk(v)*fctqsuvx
                     fctsiy(u) = fctsiy(u)          &
                                +fctkkk(v)*fctqsuvy
                     fctsiz(u) = fctsiz(u)          &
                                +fctkkk(v)*fctqsuvz

                     if(imove(v)==0) then
                        fctsix(u) = fctsix(u)          &
                                   +fcthhh(v)*fctqsuvx &
                                   +fctvsg(v)*fctusuvx &
                                   +fctusg(v)*fctvsuvx &
                                   +          fctwsuvx
                        fctsiy(u) = fctsiy(u)          &
                                   +fcthhh(v)*fctqsuvy &
                                   +fctvsg(v)*fctusuvy &
                                   +fctusg(v)*fctvsuvy &
                                   +          fctwsuvy
                        fctsiz(u) = fctsiz(u)          &
                                   +fcthhh(v)*fctqsuvz &
                                   +fctvsg(v)*fctusuvz &
                                   +fctusg(v)*fctvsuvz &
                                   +          fctwsuvz
                     endif
                  endif
               endif
            enddo
         enddo

      endif ! (ifast)

! ================================================
      do i=1,reanat
         fctpl1 = fctpl1 + fctself(i) * fctslfw(i)
         fctnpl = fctnpl + fctnpol(i) * fctslfw(i)

         dx(i)=dx(i)+fctssx(i)+fctsix(i)+fctiix(i)
         dy(i)=dy(i)+fctssy(i)+fctsiy(i)+fctiiy(i)
         dz(i)=dz(i)+fctssz(i)+fctsiz(i)+fctiiz(i)
      enddo

      fctpol=fctpl1+fctpl2
      ! fctnpl=fctnp1+fctnp2+fctnp3

      ! ----
      ! write(*,'(A,F12.5,A,F12.5,A,F12.5,A,F12.5)') &
      ! 'FCTPOL  ',fctpol,'  FCTPL1  ',fctpl1,'  fctpl2  ',fctpl2,'  fctnpl  ',fctnpl
      ! ----
      if(qslfunit)then
         do i=1,reanat
            write(selfunit,'(F10.5,X)', advance="no" ) fctself(i) ! *fctslfw(i)
         enddo
         write(selfunit,'(/)', advance="no" )
         ! write(11,'(/)',advance="no")
      endif

      if(qnplunit)then
         do i=1,reanat
            write(npolunit,'(F10.5,X)', advance="no" ) fctnpol(i) ! *fctslfw(i)
         enddo
         write(npolunit,'(/)', advance="no" )
         ! write(11,'(/)',advance="no")
      endif

      if(qscrunit)then
         do i=1,reanat
            write(scrnunit,'(F10.5,X)', advance="no" ) fctscreene(i) ! *fctslfw(i)
         enddo
         write(scrnunit,'(/)', advance="no" )
         ! write(11,'(/)',advance="no")
      endif
      ! ---
   endif

! =====================================================================
! =====================================================================

   if (fctscr) then

      fctpl1  = zero
      fctpl2  = zero
      fctnp1  = zero
      fctnp2  = zero
      fctnp3  = zero

#if KEY_STRINGM==1 /* VO*/
      fctcomm=zero
#else /**/
      fct01ss = zero

      fct03ss = zero
      fct01xs = zero
      fct01ys = zero
      fct01zs = zero
      fct02xs = zero
      fct02ys = zero
      fct02zs = zero
      fct03xs = zero
      fct03ys = zero
      fct03zs = zero
      fct01xx = zero
      fct01xy = zero
      fct01xz = zero
      fct01yy = zero
      fct01yz = zero
      fct01zz = zero
#endif 
      fctggg  = zero

      fctssx  = zero
      fctssy  = zero
      fctssz  = zero
      fctsix  = zero
      fctsiy  = zero
      fctsiz  = zero
      fctiix  = zero
      fctiiy  = zero
      fctiiz  = zero

      do i=1,reanat
! ==========================================
#if KEY_PARALLEL==1
         fctqsg(i) =zero
         if(mynod==0)then
            fct02ss(i)=fctbin
         else
            fct02ss(i)=zero
         endif
#else /**/
         fct02ss(i)=fctbin
#endif 
! ==========================================
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            u=i
            v=fct1lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct01vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u)))
               fct02vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu
               fct03vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctivu
               fct04vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu*fctivu
               fct05vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))
               fct06vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctsvu
               fct07vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctivu
               fct01ss(u)= fct01ss(u)+fct01vuss
               fct02ss(u)= fct02ss(u)+fct02vuss
               fct03ss(u)= fct03ss(u)+fct03vuss
               fct01xs(u)= fct01xs(u)+fct03vuss*vux
               fct01ys(u)= fct01ys(u)+fct03vuss*vuy
               fct01zs(u)= fct01zs(u)+fct03vuss*vuz
               fct02xs(u)= fct02xs(u)+fct05vuss*vux
               fct02ys(u)= fct02ys(u)+fct05vuss*vuy
               fct02zs(u)= fct02zs(u)+fct05vuss*vuz
               fct03xs(u)= fct03xs(u)+(-fct04vuss+fct06vuss)*vux
               fct03ys(u)= fct03ys(u)+(-fct04vuss+fct06vuss)*vuy
               fct03zs(u)= fct03zs(u)+(-fct04vuss+fct06vuss)*vuz
               fct01xx(u)= fct01xx(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01xy(u)= fct01xy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01xz(u)= fct01xz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01yy(u)= fct01yy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01yz(u)= fct01yz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01zz(u)= fct01zz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct01uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v)))
               fct02uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu
               fct03uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctivu
               fct04uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu*fctivu
               fct05uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))
               fct06uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctsvu
               fct07uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctivu
               fct01ss(v)= fct01ss(v)+fct01uvss
               fct02ss(v)= fct02ss(v)+fct02uvss
               fct03ss(v)= fct03ss(v)+fct03uvss
               fct01xs(v)= fct01xs(v)-fct03uvss*vux
               fct01ys(v)= fct01ys(v)-fct03uvss*vuy
               fct01zs(v)= fct01zs(v)-fct03uvss*vuz
               fct02xs(v)= fct02xs(v)-fct05uvss*vux
               fct02ys(v)= fct02ys(v)-fct05uvss*vuy
               fct02zs(v)= fct02zs(v)-fct05uvss*vuz
               fct03xs(v)= fct03xs(v)-(-fct04uvss+fct06uvss)*vux
               fct03ys(v)= fct03ys(v)-(-fct04uvss+fct06uvss)*vuy
               fct03zs(v)= fct03zs(v)-(-fct04uvss+fct06uvss)*vuz
               fct01xx(v)= fct01xx(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01xy(v)= fct01xy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01xz(v)= fct01xz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01yy(v)= fct01yy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01yz(v)= fct01yz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01zz(v)= fct01zz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct01vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u)))
               fct02vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu
               fct03vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctivu
               fct04vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu*fctivu
               fct05vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))
               fct06vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctsvu
               fct07vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctivu
               fct01ss(u)= fct01ss(u)+fct01vuss
               fct02ss(u)= fct02ss(u)+fct02vuss
               fct03ss(u)= fct03ss(u)+fct03vuss
               fct01xs(u)= fct01xs(u)+fct03vuss*vux
               fct01ys(u)= fct01ys(u)+fct03vuss*vuy
               fct01zs(u)= fct01zs(u)+fct03vuss*vuz
               fct02xs(u)= fct02xs(u)+fct05vuss*vux
               fct02ys(u)= fct02ys(u)+fct05vuss*vuy
               fct02zs(u)= fct02zs(u)+fct05vuss*vuz
               fct03xs(u)= fct03xs(u)+(-fct04vuss+fct06vuss)*vux
               fct03ys(u)= fct03ys(u)+(-fct04vuss+fct06vuss)*vuy
               fct03zs(u)= fct03zs(u)+(-fct04vuss+fct06vuss)*vuz
               fct01xx(u)= fct01xx(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01xy(u)= fct01xy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01xz(u)= fct01xz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01yy(u)= fct01yy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01yz(u)= fct01yz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01zz(u)= fct01zz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct01uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v)))
               fct02uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu
               fct03uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctivu
               fct04uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu*fctivu
               fct05uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))
               fct06uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctsvu
               fct07uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctivu
               fct01ss(v)= fct01ss(v)+fct01uvss
               fct02ss(v)= fct02ss(v)+fct02uvss
               fct03ss(v)= fct03ss(v)+fct03uvss
               fct01xs(v)= fct01xs(v)-fct03uvss*vux
               fct01ys(v)= fct01ys(v)-fct03uvss*vuy
               fct01zs(v)= fct01zs(v)-fct03uvss*vuz
               fct02xs(v)= fct02xs(v)-fct05uvss*vux
               fct02ys(v)= fct02ys(v)-fct05uvss*vuy
               fct02zs(v)= fct02zs(v)-fct05uvss*vuz
               fct03xs(v)= fct03xs(v)-(-fct04uvss+fct06uvss)*vux
               fct03ys(v)= fct03ys(v)-(-fct04uvss+fct06uvss)*vuy
               fct03zs(v)= fct03zs(v)-(-fct04uvss+fct06uvss)*vuz
               fct01xx(v)= fct01xx(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01xy(v)= fct01xy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01xz(v)= fct01xz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01yy(v)= fct01yy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01yz(v)= fct01yz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01zz(v)= fct01zz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz
            endif
         enddo
      enddo

! ==========================================
#if KEY_PARALLEL==1
#if KEY_STRINGM==1 /* VO with stringm usually faster to reduce in one call*/
      call gcombr(fctcomm,18*reanat)
#else /**/
      call gcomb(fct01ss,reanat)
      call gcomb(fct02ss,reanat)
      call gcomb(fct03ss,reanat)
      call gcomb(fct01xs,reanat)
      call gcomb(fct01ys,reanat)
      call gcomb(fct01zs,reanat)
      call gcomb(fct02xs,reanat)
      call gcomb(fct02ys,reanat)
      call gcomb(fct02zs,reanat)
      call gcomb(fct03xs,reanat)
      call gcomb(fct03ys,reanat)
      call gcomb(fct03zs,reanat)
      call gcomb(fct01xx,reanat)
      call gcomb(fct01xy,reanat)
      call gcomb(fct01xz,reanat)
      call gcomb(fct01yy,reanat)
      call gcomb(fct01yz,reanat)
      call gcomb(fct01zz,reanat)
#endif 
#endif 
! ==========================================

      do i=1,reanat
         fcttp1   = sqrt( fct01xs(i)*fct01xs(i)    &
                           +fct01ys(i)*fct01ys(i)  &
                           +fct01zs(i)*fct01zs(i))
         fcttp2   = fct02ss(i)

         fctis1(i)= one/fcttp1
         fctis2(i)= one/fcttp2

         fctmov(i)= fct01ss(i)
         fctmos(i)= fcttp1*fctis2(i)

         fcttp1   =                   fctmov(i)                   &
                   +fctcb1(fctidx(i))          *fctmos(i)         &
                   +fctcb2(fctidx(i))*fctmov(i)*fctmos(i)         &
                   +fctqos
         fcttp2   = one                                           &
                   +fctcb2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcb1(fctidx(i))                             &
                   +fctcb2(fctidx(i))*fctmov(i)
         fcttp4   = fctca2(fctidx(i))*(fcttp1-fctca3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctca0(fctidx(i))                             &
                   +fctca1(fctidx(i))*fcttp6
         fcttp8   = fctca1(fctidx(i))*fctca2(fctidx(i))           &
                   /(one/fcttp5+two+fcttp5)
         fctqsg(i)= fcttp7
         fctqdv(i)= fcttp2*fcttp8
         fctqds(i)= fcttp3*fcttp8
         if (fctqsg(i) > -0.001d0) then
            call wrndie(-4,'<FCTENE>','Solvation energy too high.')
         endif
         fctqsg(i)= fctqsg(i)*fctqun(fctidx(i))
         fctqdv(i)= fctqdv(i)*fctqun(fctidx(i))
         fctqds(i)= fctqds(i)*fctqun(fctidx(i))

         fcttp1   =                   fctmov(i)                  &
                   +fctcd1(fctidx(i))          *fctmos(i)        &
                   +fctcd2(fctidx(i))*fctmov(i)*fctmos(i)        &
                   +fctuos
         fcttp2   = one                                          &
                   +fctcd2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcd1(fctidx(i))                            &
                   +fctcd2(fctidx(i))*fctmov(i)
         fcttp4   = fctcc2(fctidx(i))*(fcttp1-fctcc3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctcc0(fctidx(i))                            &
                   +fctcc1(fctidx(i))*fcttp6
         fcttp8   = fctcc1(fctidx(i))*fctcc2(fctidx(i))          &
                   /(one/fcttp5+two+fcttp5)
         fctusg(i)= fcttp7
         fctudv(i)= fcttp2*fcttp8
         fctuds(i)= fcttp3*fcttp8
         if (fctusg(i) < 0.001d0) then
            call wrndie(-4,'<FCTENE>','Surface too low.')
         endif

         fcttp1   =                   fctmov(i)           &
                   +fctcf1(fctidx(i))          *fctmos(i) &
                   +fctcf2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctvos
         fcttp2   = one                                   &
                     +fctcf2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcf1(fctidx(i))                     &
                   +fctcf2(fctidx(i))*fctmov(i)
         fcttp4   = fctce2(fctidx(i))*(fcttp1-fctce3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctce0(fctidx(i))                     &
                   +fctce1(fctidx(i))*fcttp6
         fcttp8   = fctce1(fctidx(i))*fctce2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctvsg(i)= fcttp7
         fctvdv(i)= fcttp2*fcttp8
         fctvds(i)= fcttp3*fcttp8

         fcttp1   =                   fctmov(i)           &
                   +fctch1(fctidx(i))          *fctmos(i) &
                   +fctch2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctwos
         fcttp2   = one                                   &
                   +fctch2(fctidx(i))          *fctmos(i)
         fcttp3   = fctch1(fctidx(i))                     &
                   +fctch2(fctidx(i))*fctmov(i)
         fcttp4   = fctcg2(fctidx(i))*(fcttp1-fctcg3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctcg0(fctidx(i))                     &
                   +fctcg1(fctidx(i))*fcttp6
         fcttp8   = fctcg1(fctidx(i))*fctcg2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctwsg(i)= fcttp7
         fctwdv(i)= fcttp2*fcttp8
         fctwds(i)= fcttp3*fcttp8

         fctisg(i)= one/fctqsg(i)

         ! VdW TERM ================================================
         fctvdw01    = fcttau*fctisg(i)/two
         fctvdw02    = one/(-fctvdw01+fctvdwwrd)
         fctvdwen(i) = fctvdwfac(i)*fctvdw02**3
         fctvdwdf(i) =-three*fctvdwen(i)*fctvdw02*fctvdw01*fctisg(i)
         ! =========================================================

         fct04xs  = (fct03ss(i)*fct01xs(i)          &
                    +fct01xs(i)*fct01xx(i)          &
                    +fct01ys(i)*fct01xy(i)          &
                    +fct01zs(i)*fct01xz(i))         &
                    *fctis1(i) *fctis2(i)           &
                    -fct03xs(i)*fctis2(i)*fctmos(i)
         fct04ys  = (fct03ss(i)*fct01ys(i)          &
                    +fct01xs(i)*fct01xy(i)          &
                    +fct01ys(i)*fct01yy(i)          &
                    +fct01zs(i)*fct01yz(i))         &
                    *fctis1(i) *fctis2(i)           &
                    -fct03ys(i)*fctis2(i)*fctmos(i)
         fct04zs  = (fct03ss(i)*fct01zs(i)          &
                    +fct01xs(i)*fct01xz(i)          &
                    +fct01ys(i)*fct01yz(i)          &
                    +fct01zs(i)*fct01zz(i))         &
                    *fctis1(i) *fctis2(i)           &
                    -fct03zs(i)*fctis2(i)*fctmos(i)

         fctqsx(i)=-fct02xs(i)*fctqdv(i)-fct04xs*fctqds(i)
         fctqsy(i)=-fct02ys(i)*fctqdv(i)-fct04ys*fctqds(i)
         fctqsz(i)=-fct02zs(i)*fctqdv(i)-fct04zs*fctqds(i)
         fctusx(i)=-fct02xs(i)*fctudv(i)-fct04xs*fctuds(i)
         fctusy(i)=-fct02ys(i)*fctudv(i)-fct04ys*fctuds(i)
         fctusz(i)=-fct02zs(i)*fctudv(i)-fct04zs*fctuds(i)
         fctvsx(i)=-fct02xs(i)*fctvdv(i)-fct04xs*fctvds(i)
         fctvsy(i)=-fct02ys(i)*fctvdv(i)-fct04ys*fctvds(i)
         fctvsz(i)=-fct02zs(i)*fctvdv(i)-fct04zs*fctvds(i)
         fctwsx(i)=-fct02xs(i)*fctwdv(i)-fct04xs*fctwds(i)
         fctwsy(i)=-fct02ys(i)*fctwdv(i)-fct04ys*fctwds(i)
         fctwsz(i)=-fct02zs(i)*fctwdv(i)-fct04zs*fctwds(i)

! ==========================================
#if KEY_PARALLEL==1
         if(mynod==0)then
#endif 
! ==========================================
         fctnp1   = fctnp1+fctnpc(i)*fctqsg(i)
         fctnp2   = fctnp2+fctusg(i)*fctvsg(i)
         fctnp3   = fctnp3+fctwsg(i)+fctvdwen(i)

         fctssx(i)=  fctssx(i)                        &
                   +(fctvdwdf(i)+fctnpc(i))*fctqsx(i) &
                   +             fctvsg(i) *fctusx(i) &
                   +             fctusg(i) *fctvsx(i) &
                   +                        fctwsx(i)
         fctssy(i)=  fctssy(i)                        &
                   +(fctvdwdf(i)+fctnpc(i))*fctqsy(i) &
                   +             fctvsg(i) *fctusy(i) &
                   +             fctusg(i) *fctvsy(i) &
                   +                        fctwsy(i)
         fctssz(i)=  fctssz(i)                        &
                   +(fctvdwdf(i)+fctnpc(i))*fctqsz(i) &
                   +             fctvsg(i) *fctusz(i) &
                   +             fctusg(i) *fctvsz(i) &
                   +                        fctwsz(i)
! ==========================================
#if KEY_PARALLEL==1
         endif
#endif 
! ==========================================
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=reablo(i-1)+1
         endif
         b=reablo(i)
         do j=a,b
            u=i
            v=abs(reanbo(j))
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if (fctdvu < fct3cn) then
               fct01vu  = one/fctdvu
               fct02vu  = one-fctdvu*fct3ci
               fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
               fct04vu  = dexp(-fctikp*fct03vu)
               fct05vu  = fct04vu/fct03vu
               fct06vu  = fct05vu+fctikp*fct04vu
               fct07vu  = one/(one+fct05vu)
               fct08vu  =-cg(v)*cg(u)*fcttau*fct02vu              &
                         *sqrt(fct01vu*fct07vu)
               if(reanbo(j) < 0) fct08vu=fct08vu*e14fac
               fct09vu  = (fct01vu*fct02vu*(-one+fct06vu*fct07vu) &
                          +fct2fn)*fct08vu
               fct10vu  = half*fct02vu*fct06vu*fct07vu*fct08vu
               fctpl2   = fctpl2+fct02vu*fct08vu
               fctggg(u)= fctggg(u)-fct10vu
               fctggg(v)= fctggg(v)-fct10vu
               fctiix(u)= fctiix(u)                               &
                         -fct09vu*vux+fct10vu*fctisg(u)*fctqsx(u)
               fctiiy(u)= fctiiy(u)                               &
                         -fct09vu*vuy+fct10vu*fctisg(u)*fctqsy(u)
               fctiiz(u)= fctiiz(u)                               &
                         -fct09vu*vuz+fct10vu*fctisg(u)*fctqsz(u)
               fctiix(v)= fctiix(v)                               &
                         +fct09vu*vux+fct10vu*fctisg(v)*fctqsx(v)
               fctiiy(v)= fctiiy(v)                               &
                         +fct09vu*vuy+fct10vu*fctisg(v)*fctqsy(v)
               fctiiz(v)= fctiiz(v)                               &
                         +fct09vu*vuz+fct10vu*fctisg(v)*fctqsz(v)
               endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=imablo(i-1)+1
         endif
         b=imablo(i)
         do j=a,b
            u=i
            v=abs(imanbo(j))
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if (fctdvu < fct3cn) then
               fct01vu  = one/fctdvu
               fct02vu  = one-fctdvu*fct3ci
               fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
               fct04vu  = dexp(-fctikp*fct03vu)
               fct05vu  = fct04vu/fct03vu
               fct06vu  = fct05vu+fctikp*fct04vu
               fct07vu  = one/(one+fct05vu)
               fct08vu  =-cg(v)*cg(u)*fcttau*fct02vu              &
                         *sqrt(fct01vu*fct07vu)
               if(imanbo(j) < 0) fct08vu=fct08vu*e14fac
               fct09vu  = (fct01vu*fct02vu*(-one+fct06vu*fct07vu) &
                          +fct2fn)*fct08vu
               fct10vu  = half*fct02vu*fct06vu*fct07vu*fct08vu
               fctpl2   = fctpl2+fct02vu*fct08vu
               fctggg(u)= fctggg(u)-fct10vu
               fctggg(v)= fctggg(v)-fct10vu
               fctiix(u)= fctiix(u)                               &
                         -fct09vu*vux+fct10vu*fctisg(u)*fctqsx(u)
               fctiiy(u)= fctiiy(u)                               &
                         -fct09vu*vuy+fct10vu*fctisg(u)*fctqsy(u)
               fctiiz(u)= fctiiz(u)                               &
                         -fct09vu*vuz+fct10vu*fctisg(u)*fctqsz(u)
               fctiix(v)= fctiix(v)                               &
                         +fct09vu*vux+fct10vu*fctisg(v)*fctqsx(v)
               fctiiy(v)= fctiiy(v)                               &
                         +fct09vu*vuy+fct10vu*fctisg(v)*fctqsy(v)
               fctiiz(v)= fctiiz(v)                               &
                         +fct09vu*vuz+fct10vu*fctisg(v)*fctqsz(v)
               endif
         enddo
      enddo

! ========================================
#if KEY_PARALLEL==1
          call gcomb(fctggg,reanat)
#endif 
! ========================================

      do i=1,reanat
         fcthhh(i)=fctnpc(i)-fctisg(i)*fctggg(i)+fctvdwdf(i)
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            u=i
            v=fct1lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct03vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctivu
               fct04vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu*fctivu
               fct05vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))
               fct06vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctsvu
               fct07vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctivu
               fct02vuxs = fct05vuss*vux
               fct02vuys = fct05vuss*vuy
               fct02vuzs = fct05vuss*vuz
               fct03vuxs = (-fct04vuss+fct06vuss)*vux
               fct03vuys = (-fct04vuss+fct06vuss)*vuy
               fct03vuzs = (-fct04vuss+fct06vuss)*vuz
               fct01vuxx = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01vuxy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01vuxz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01vuyy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01vuyz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01vuzz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

               fct04vuxs = (fct03vuss *fct01xs(u)          &
                           +fct01xs(u)*fct01vuxx           &
                           +fct01ys(u)*fct01vuxy           &
                           +fct01zs(u)*fct01vuxz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuxs *fctis2(u)*fctmos(u)
               fct04vuys = (fct03vuss *fct01ys(u)          &
                           +fct01xs(u)*fct01vuxy           &
                           +fct01ys(u)*fct01vuyy           &
                           +fct01zs(u)*fct01vuyz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuys *fctis2(u)*fctmos(u)
               fct04vuzs = (fct03vuss *fct01zs(u)          &
                           +fct01xs(u)*fct01vuxz           &
                           +fct01ys(u)*fct01vuyz           &
                           +fct01zs(u)*fct01vuzz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuzs *fctis2(u)*fctmos(u)

               fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
               fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
               fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
               fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
               fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
               fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)
               fctvsvux  = fct02vuxs*fctvdv(u)+fct04vuxs*fctvds(u)
               fctvsvuy  = fct02vuys*fctvdv(u)+fct04vuys*fctvds(u)
               fctvsvuz  = fct02vuzs*fctvdv(u)+fct04vuzs*fctvds(u)
               fctwsvux  = fct02vuxs*fctwdv(u)+fct04vuxs*fctwds(u)
               fctwsvuy  = fct02vuys*fctwdv(u)+fct04vuys*fctwds(u)
               fctwsvuz  = fct02vuzs*fctwdv(u)+fct04vuzs*fctwds(u)

               fctsix(v) = fctsix(v)          &
                          +fcthhh(u)*fctqsvux &
                          +fctvsg(u)*fctusvux &
                          +fctusg(u)*fctvsvux &
                          +          fctwsvux
               fctsiy(v) = fctsiy(v)          &
                          +fcthhh(u)*fctqsvuy &
                          +fctvsg(u)*fctusvuy &
                          +fctusg(u)*fctvsvuy &
                          +          fctwsvuy
               fctsiz(v) = fctsiz(v)          &
                          +fcthhh(u)*fctqsvuz &
                          +fctvsg(u)*fctusvuz &
                          +fctusg(u)*fctvsvuz &
                          +          fctwsvuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct03uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctivu
               fct04uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu*fctivu
               fct05uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))
               fct06uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctsvu
               fct07uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctivu
               fct02uvxs =-fct05uvss*vux
               fct02uvys =-fct05uvss*vuy
               fct02uvzs =-fct05uvss*vuz
               fct03uvxs =-(-fct04uvss+fct06uvss)*vux
               fct03uvys =-(-fct04uvss+fct06uvss)*vuy
               fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
               fct01uvxx = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01uvxy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01uvxz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01uvyy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01uvyz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01uvzz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

               fct04uvxs = (fct03uvss *fct01xs(v)          &
                           +fct01xs(v)*fct01uvxx           &
                           +fct01ys(v)*fct01uvxy           &
                           +fct01zs(v)*fct01uvxz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvxs *fctis2(v)*fctmos(v)
               fct04uvys = (fct03uvss *fct01ys(v)          &
                           +fct01xs(v)*fct01uvxy           &
                           +fct01ys(v)*fct01uvyy           &
                           +fct01zs(v)*fct01uvyz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvys *fctis2(v)*fctmos(v)
               fct04uvzs = (fct03uvss *fct01zs(v)          &
                           +fct01xs(v)*fct01uvxz           &
                           +fct01ys(v)*fct01uvyz           &
                           +fct01zs(v)*fct01uvzz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvzs *fctis2(v)*fctmos(v)

               fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
               fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
               fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
               fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
               fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
               fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)
               fctvsuvx  = fct02uvxs*fctvdv(v)+fct04uvxs*fctvds(v)
               fctvsuvy  = fct02uvys*fctvdv(v)+fct04uvys*fctvds(v)
               fctvsuvz  = fct02uvzs*fctvdv(v)+fct04uvzs*fctvds(v)
               fctwsuvx  = fct02uvxs*fctwdv(v)+fct04uvxs*fctwds(v)
               fctwsuvy  = fct02uvys*fctwdv(v)+fct04uvys*fctwds(v)
               fctwsuvz  = fct02uvzs*fctwdv(v)+fct04uvzs*fctwds(v)

               fctsix(u) = fctsix(u)          &
                          +fcthhh(v)*fctqsuvx &
                          +fctvsg(v)*fctusuvx &
                          +fctusg(v)*fctvsuvx &
                          +          fctwsuvx
               fctsiy(u) = fctsiy(u)          &
                          +fcthhh(v)*fctqsuvy &
                          +fctvsg(v)*fctusuvy &
                          +fctusg(v)*fctvsuvy &
                          +          fctwsuvy
               fctsiz(u) = fctsiz(u)          &
                          +fcthhh(v)*fctqsuvz &
                          +fctvsg(v)*fctusuvz &
                          +fctusg(v)*fctvsuvz &
                          +          fctwsuvz
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct03vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctivu
               fct04vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu*fctivu
               fct05vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))
               fct06vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctsvu
               fct07vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctivu
               fct02vuxs = fct05vuss*vux
               fct02vuys = fct05vuss*vuy
               fct02vuzs = fct05vuss*vuz
               fct03vuxs = (-fct04vuss+fct06vuss)*vux
               fct03vuys = (-fct04vuss+fct06vuss)*vuy
               fct03vuzs = (-fct04vuss+fct06vuss)*vuz
               fct01vuxx = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01vuxy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01vuxz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01vuyy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01vuyz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01vuzz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

               fct04vuxs = (fct03vuss *fct01xs(u)          &
                           +fct01xs(u)*fct01vuxx           &
                           +fct01ys(u)*fct01vuxy           &
                           +fct01zs(u)*fct01vuxz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuxs *fctis2(u)*fctmos(u)
               fct04vuys = (fct03vuss *fct01ys(u)          &
                           +fct01xs(u)*fct01vuxy           &
                           +fct01ys(u)*fct01vuyy           &
                           +fct01zs(u)*fct01vuyz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuys *fctis2(u)*fctmos(u)
               fct04vuzs = (fct03vuss *fct01zs(u)          &
                           +fct01xs(u)*fct01vuxz           &
                           +fct01ys(u)*fct01vuyz           &
                           +fct01zs(u)*fct01vuzz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuzs *fctis2(u)*fctmos(u)

               fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
               fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
               fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
               fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
               fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
               fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)
               fctvsvux  = fct02vuxs*fctvdv(u)+fct04vuxs*fctvds(u)
               fctvsvuy  = fct02vuys*fctvdv(u)+fct04vuys*fctvds(u)
               fctvsvuz  = fct02vuzs*fctvdv(u)+fct04vuzs*fctvds(u)
               fctwsvux  = fct02vuxs*fctwdv(u)+fct04vuxs*fctwds(u)
               fctwsvuy  = fct02vuys*fctwdv(u)+fct04vuys*fctwds(u)
               fctwsvuz  = fct02vuzs*fctwdv(u)+fct04vuzs*fctwds(u)

               fctsix(v) =  fctsix(v)          &
                           +fcthhh(u)*fctqsvux &
                           +fctvsg(u)*fctusvux &
                           +fctusg(u)*fctvsvux &
                           +          fctwsvux
               fctsiy(v) =  fctsiy(v)          &
                           +fcthhh(u)*fctqsvuy &
                           +fctvsg(u)*fctusvuy &
                           +fctusg(u)*fctvsvuy &
                           +          fctwsvuy
               fctsiz(v) =  fctsiz(v)          &
                           +fcthhh(u)*fctqsvuz &
                           +fctvsg(u)*fctusvuz &
                           +fctusg(u)*fctvsvuz &
                           +          fctwsvuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct03uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctivu
               fct04uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu*fctivu
               fct05uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))
               fct06uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctsvu
               fct07uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctivu
               fct02uvxs =-fct05uvss*vux
               fct02uvys =-fct05uvss*vuy
               fct02uvzs =-fct05uvss*vuz
               fct03uvxs =-(-fct04uvss+fct06uvss)*vux
               fct03uvys =-(-fct04uvss+fct06uvss)*vuy
               fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
               fct01uvxx = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01uvxy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01uvxz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01uvyy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01uvyz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01uvzz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

               fct04uvxs = (fct03uvss *fct01xs(v)          &
                           +fct01xs(v)*fct01uvxx           &
                           +fct01ys(v)*fct01uvxy           &
                           +fct01zs(v)*fct01uvxz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvxs *fctis2(v)*fctmos(v)
               fct04uvys = (fct03uvss *fct01ys(v)          &
                           +fct01xs(v)*fct01uvxy           &
                           +fct01ys(v)*fct01uvyy           &
                           +fct01zs(v)*fct01uvyz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvys *fctis2(v)*fctmos(v)
               fct04uvzs = (fct03uvss *fct01zs(v)          &
                           +fct01xs(v)*fct01uvxz           &
                           +fct01ys(v)*fct01uvyz           &
                           +fct01zs(v)*fct01uvzz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvzs *fctis2(v)*fctmos(v)

               fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
               fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
               fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
               fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
               fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
               fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)
               fctvsuvx  = fct02uvxs*fctvdv(v)+fct04uvxs*fctvds(v)
               fctvsuvy  = fct02uvys*fctvdv(v)+fct04uvys*fctvds(v)
               fctvsuvz  = fct02uvzs*fctvdv(v)+fct04uvzs*fctvds(v)
               fctwsuvx  = fct02uvxs*fctwdv(v)+fct04uvxs*fctwds(v)
               fctwsuvy  = fct02uvys*fctwdv(v)+fct04uvys*fctwds(v)
               fctwsuvz  = fct02uvzs*fctwdv(v)+fct04uvzs*fctwds(v)

               fctsix(u) =  fctsix(u)          &
                           +fcthhh(v)*fctqsuvx &
                           +fctvsg(v)*fctusuvx &
                           +fctusg(v)*fctvsuvx &
                           +          fctwsuvx
               fctsiy(u) =  fctsiy(u)          &
                           +fcthhh(v)*fctqsuvy &
                           +fctvsg(v)*fctusuvy &
                           +fctusg(v)*fctvsuvy &
                           +          fctwsuvy
               fctsiz(u) =  fctsiz(u)          &
                           +fcthhh(v)*fctqsuvz &
                           +fctvsg(v)*fctusuvz &
                           +fctusg(v)*fctvsuvz &
                           +          fctwsuvz
            endif
         enddo
      enddo

      do i=1,reanat
         dx(i)=dx(i)+fctssx(i)+fctsix(i)+fctiix(i)
         dy(i)=dy(i)+fctssy(i)+fctsiy(i)+fctiiy(i)
         dz(i)=dz(i)+fctssz(i)+fctsiz(i)+fctiiz(i)
      enddo

      fctpol=fctpl1+fctpl2
      fctnpl=fctnp1+fctnp2+fctnp3

   endif
     !
#if KEY_STRINGM==1 /* VO nullify pointer arrays*/
   nullify(FCT01SS,FCT02SS,FCT03SS,FCT01XS,FCT01YS,FCT01ZS,&
  &                                FCT02XS,FCT02YS,FCT02ZS,&
  &                                FCT03XS,FCT03YS,FCT03ZS,&
  &                                FCT01XX,FCT01XY,FCT01XZ,&
  &                                        FCT01YY,FCT01YZ,&
  &                                                FCT01ZZ)
#endif 
    !
   RETURN
   END Subroutine fctene

! =========================================================
! =========================================================

   SUBROUTINE fctprt(fctpol         ,&
                     fctnpl         ,&
                     reanat         ,&
                     reablo,reanbo  ,&
                     fct1ll,fct1lb  ,&
                     fct2ll,fct2lb  ,&
                     imanat         ,&
                     imablo,imanbo  ,&
                     imattr         ,&
                     fct3ll,fct3lb  ,&
                     fct4ll,fct4lb)

! This routine prints data calculated by FACTS.
!
! Author: Urs Haberthuer.

   use  chm_kinds
   use  dimens_fcm

   use  coord
   use  deriv

   use  inbnd
   use  number
   use  psf

   use stream
! ##IF PARALLEL
!   use  parallel
! ##ENDIF

       implicit none

   real(kind=chm_real), intent(out)  :: fctpol, fctnpl
   integer, intent(in) :: reanat, imanat
   integer, intent(in) :: imattr(*)
   integer, intent(in) :: reablo(*), reanbo(*)
   integer, intent(in) :: fct1ll(*), fct1lb(*)
   integer, intent(in) :: fct2ll(*), fct2lb(*)
   integer, intent(in) :: imablo(*), imanbo(*)
   integer, intent(in) :: fct3ll(*), fct3lb(*)
   integer, intent(in) :: fct4ll(*), fct4lb(*)

   integer             :: a,b,u,v,i,j
   logical             :: qmove

   real(kind=chm_real) :: vux,vuy,vuz
   real(kind=chm_real) :: fctdvu,fctivu,fctsvu

   real(kind=chm_real) :: fct01vuss,fct02vuss,fct03vuss,fct04vuss
   real(kind=chm_real) :: fct05vuss,fct06vuss,fct07vuss
   real(kind=chm_real) :: fct01uvss,fct02uvss,fct03uvss,fct04uvss
   real(kind=chm_real) :: fct05uvss,fct06uvss,fct07uvss
   real(kind=chm_real) :: fct01ss(natom),fct02ss(natom),fct03ss(natom)
   real(kind=chm_real) :: fct01xs(natom),fct01ys(natom),fct01zs(natom)
   real(kind=chm_real) :: fct02xs(natom),fct02ys(natom),fct02zs(natom)
   real(kind=chm_real) :: fct03xs(natom),fct03ys(natom),fct03zs(natom)
   real(kind=chm_real) :: fct04xs,fct04ys,fct04zs
   real(kind=chm_real) :: fct01xx(natom),fct01xy(natom),fct01xz(natom)
   real(kind=chm_real) :: fct01yy(natom),fct01yz(natom)
   real(kind=chm_real) :: fct01zz(natom)

   real(kind=chm_real) :: fcttp1,fcttp2,fcttp3,fcttp4
   real(kind=chm_real) :: fcttp5,fcttp6,fcttp7,fcttp8

   real(kind=chm_real) :: fctis1(natom),fctis2(natom)
   real(kind=chm_real) :: fctmov(natom),fctmos(natom)

   real(kind=chm_real) :: fctqsg(natom)
   real(kind=chm_real) :: fctisg(natom)
   real(kind=chm_real) :: fctqdv(natom),fctqds(natom)
   real(kind=chm_real) :: fctusg(natom)
   real(kind=chm_real) :: fctudv(natom),fctuds(natom)
   real(kind=chm_real) :: fctvsg(natom)
   real(kind=chm_real) :: fctvdv(natom),fctvds(natom)
   real(kind=chm_real) :: fctwsg(natom)
   real(kind=chm_real) :: fctwdv(natom),fctwds(natom)

   real(kind=chm_real) :: fctpl1,fctpl2
   real(kind=chm_real) :: fctnp1,fctnp2,fctnp3

   real(kind=chm_real) :: fctqsx(natom),fctqsy(natom),fctqsz(natom)
   real(kind=chm_real) :: fctusx(natom),fctusy(natom),fctusz(natom)
   real(kind=chm_real) :: fctvsx(natom),fctvsy(natom),fctvsz(natom)
   real(kind=chm_real) :: fctwsx(natom),fctwsy(natom),fctwsz(natom)

   real(kind=chm_real) :: fct01vu,fct02vu,fct03vu,fct04vu
   real(kind=chm_real) :: fct05vu,fct06vu,fct07vu
   real(kind=chm_real) :: fct08vu,fct09vu,fct10vu,fct11vu
   real(kind=chm_real) :: fct12vu,fct13vu,fct14vu

   real(kind=chm_real) :: fctggg(natom)
   real(kind=chm_real) :: fcthhh(natom)

   real(kind=chm_real) :: fctssx(natom),fctssy(natom),fctssz(natom)
   real(kind=chm_real) :: fctsix(natom),fctsiy(natom),fctsiz(natom)
   real(kind=chm_real) :: fctiix(natom),fctiiy(natom),fctiiz(natom)

   real(kind=chm_real) :: fct02vuxs,fct02vuys,fct02vuzs
   real(kind=chm_real) :: fct02uvxs,fct02uvys,fct02uvzs
   real(kind=chm_real) :: fct03vuxs,fct03vuys,fct03vuzs
   real(kind=chm_real) :: fct03uvxs,fct03uvys,fct03uvzs
   real(kind=chm_real) :: fct04vuxs,fct04vuys,fct04vuzs
   real(kind=chm_real) :: fct04uvxs,fct04uvys,fct04uvzs
   real(kind=chm_real) :: fct01vuxx,fct01vuxy,fct01vuxz
   real(kind=chm_real) ::           fct01vuyy,fct01vuyz
   real(kind=chm_real) ::                     fct01vuzz
   real(kind=chm_real) :: fct01uvxx,fct01uvxy,fct01uvxz
   real(kind=chm_real) ::           fct01uvyy,fct01uvyz
   real(kind=chm_real) ::                     fct01uvzz

   real(kind=chm_real) :: fctqsvux,fctqsvuy,fctqsvuz
   real(kind=chm_real) :: fctqsuvx,fctqsuvy,fctqsuvz
   real(kind=chm_real) :: fctusvux,fctusvuy,fctusvuz
   real(kind=chm_real) :: fctusuvx,fctusuvy,fctusuvz
   real(kind=chm_real) :: fctvsvux,fctvsvuy,fctvsvuz
   real(kind=chm_real) :: fctvsuvx,fctvsuvy,fctvsuvz
   real(kind=chm_real) :: fctwsvux,fctwsvuy,fctwsvuz
   real(kind=chm_real) :: fctwsuvx,fctwsuvy,fctwsuvz

   real(kind=chm_real) :: fctvdw01,fctvdw02
   real(kind=chm_real) :: fctvdwen(natom),fctvdwdf(natom)

   if (fctsfe) then
      qmove=.false.

      fctpl1=zero
      fctpl2=zero
      fctnp1=zero
      fctnp2=zero
      fctnp3=zero

      fct01ss = zero

      fct02ss = fctbin

      fct03ss = zero
      fct01xs = zero
      fct01ys = zero
      fct01zs = zero
      fct02xs = zero
      fct02ys = zero
      fct02zs = zero
      fct03xs = zero
      fct03ys = zero
      fct03zs = zero
      fct01xx = zero
      fct01xy = zero
      fct01xz = zero
      fct01yy = zero
      fct01yz = zero
      fct01zz = zero

      fctggg  = zero

      fctssx  = zero
      fctssy  = zero
      fctssz  = zero
      fctsix  = zero
      fctsiy  = zero
      fctsiz  = zero
      fctiix  = zero
      fctiiy  = zero
      fctiiz  = zero

      do i=1,reanat
         if(imove(i)>0) qmove=.true.
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            u=i
            v=fct1lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct01vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u)))
               fct02vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu
               fct03vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))
               fct06vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctivu
               fct01ss(u)= fct01ss(u)+fct01vuss
               fct02ss(u)= fct02ss(u)+fct02vuss
               fct03ss(u)= fct03ss(u)+fct03vuss
               fct01xs(u)= fct01xs(u)+fct03vuss*vux
               fct01ys(u)= fct01ys(u)+fct03vuss*vuy
               fct01zs(u)= fct01zs(u)+fct03vuss*vuz
               fct02xs(u)= fct02xs(u)+fct05vuss*vux
               fct02ys(u)= fct02ys(u)+fct05vuss*vuy
               fct02zs(u)= fct02zs(u)+fct05vuss*vuz
               fct03xs(u)= fct03xs(u)+(-fct04vuss+fct06vuss)*vux
               fct03ys(u)= fct03ys(u)+(-fct04vuss+fct06vuss)*vuy
               fct03zs(u)= fct03zs(u)+(-fct04vuss+fct06vuss)*vuz
               fct01xx(u)= fct01xx(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01xy(u)= fct01xy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01xz(u)= fct01xz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01yy(u)= fct01yy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01yz(u)= fct01yz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01zz(u)= fct01zz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct01uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v)))
               fct02uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu
               fct03uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))
               fct06uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctivu
               fct01ss(v)= fct01ss(v)+fct01uvss
               fct02ss(v)= fct02ss(v)+fct02uvss
               fct03ss(v)= fct03ss(v)+fct03uvss
               fct01xs(v)= fct01xs(v)-fct03uvss*vux
               fct01ys(v)= fct01ys(v)-fct03uvss*vuy
               fct01zs(v)= fct01zs(v)-fct03uvss*vuz
               fct02xs(v)= fct02xs(v)-fct05uvss*vux
               fct02ys(v)= fct02ys(v)-fct05uvss*vuy
               fct02zs(v)= fct02zs(v)-fct05uvss*vuz
               fct03xs(v)= fct03xs(v)-(-fct04uvss+fct06uvss)*vux
               fct03ys(v)= fct03ys(v)-(-fct04uvss+fct06uvss)*vuy
               fct03zs(v)= fct03zs(v)-(-fct04uvss+fct06uvss)*vuz
               fct01xx(v)= fct01xx(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01xy(v)= fct01xy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01xz(v)= fct01xz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01yy(v)= fct01yy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01yz(v)= fct01yz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01zz(v)= fct01zz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct01vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u)))
               fct02vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu
               fct03vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))
               fct06vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctivu
               fct01ss(u)= fct01ss(u)+fct01vuss
               fct02ss(u)= fct02ss(u)+fct02vuss
               fct03ss(u)= fct03ss(u)+fct03vuss
               fct01xs(u)= fct01xs(u)+fct03vuss*vux
               fct01ys(u)= fct01ys(u)+fct03vuss*vuy
               fct01zs(u)= fct01zs(u)+fct03vuss*vuz
               fct02xs(u)= fct02xs(u)+fct05vuss*vux
               fct02ys(u)= fct02ys(u)+fct05vuss*vuy
               fct02zs(u)= fct02zs(u)+fct05vuss*vuz
               fct03xs(u)= fct03xs(u)+(-fct04vuss+fct06vuss)*vux
               fct03ys(u)= fct03ys(u)+(-fct04vuss+fct06vuss)*vuy
               fct03zs(u)= fct03zs(u)+(-fct04vuss+fct06vuss)*vuz
               fct01xx(u)= fct01xx(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01xy(u)= fct01xy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01xz(u)= fct01xz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01yy(u)= fct01yy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01yz(u)= fct01yz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01zz(u)= fct01zz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct01uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v)))
               fct02uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu
               fct03uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))
               fct06uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctivu
               fct01ss(v)= fct01ss(v)+fct01uvss
               fct02ss(v)= fct02ss(v)+fct02uvss
               fct03ss(v)= fct03ss(v)+fct03uvss
               fct01xs(v)= fct01xs(v)-fct03uvss*vux
               fct01ys(v)= fct01ys(v)-fct03uvss*vuy
               fct01zs(v)= fct01zs(v)-fct03uvss*vuz
               fct02xs(v)= fct02xs(v)-fct05uvss*vux
               fct02ys(v)= fct02ys(v)-fct05uvss*vuy
               fct02zs(v)= fct02zs(v)-fct05uvss*vuz
               fct03xs(v)= fct03xs(v)-(-fct04uvss+fct06uvss)*vux
               fct03ys(v)= fct03ys(v)-(-fct04uvss+fct06uvss)*vuy
               fct03zs(v)= fct03zs(v)-(-fct04uvss+fct06uvss)*vuz
               fct01xx(v)= fct01xx(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01xy(v)= fct01xy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01xz(v)= fct01xz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01yy(v)= fct01yy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01yz(v)= fct01yz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01zz(v)= fct01zz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz
            endif
         enddo
      enddo

      do i=1,reanat
         fcttp1   = sqrt( fct01xs(i)*fct01xs(i)  &
                         +fct01ys(i)*fct01ys(i)  &
                         +fct01zs(i)*fct01zs(i))
         fcttp2   = fct02ss(i)

         fctis1(i)= one/fcttp1
         fctis2(i)= one/fcttp2

         fctmov(i)= fct01ss(i)
         fctmos(i)= fcttp1*fctis2(i)

         fcttp1   =                   fctmov(i)           &
                   +fctcb1(fctidx(i))          *fctmos(i) &
                   +fctcb2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctqos
         fcttp2   = one                                   &
                   +fctcb2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcb1(fctidx(i))                     &
                   +fctcb2(fctidx(i))*fctmov(i)
         fcttp4   = fctca2(fctidx(i))*(fcttp1-fctca3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctca0(fctidx(i))                     &
                   +fctca1(fctidx(i))*fcttp6
         fcttp8   = fctca1(fctidx(i))*fctca2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctqsg(i)= fcttp7
         fctqdv(i)= fcttp2*fcttp8
         fctqds(i)= fcttp3*fcttp8
         if (fctqsg(i) > -0.001d0) then
            call wrndie(-4,'<FCTPRT>','Solvation energy too high.')
         endif
         fctqsg(i)= fctqsg(i)*fctqun(fctidx(i))
         fctqdv(i)= fctqdv(i)*fctqun(fctidx(i))
         fctqds(i)= fctqds(i)*fctqun(fctidx(i))

         fcttp1   =                   fctmov(i)                  &
                   +fctcd1(fctidx(i))          *fctmos(i)        &
                   +fctcd2(fctidx(i))*fctmov(i)*fctmos(i)        &
                   +fctuos
         fcttp2   = one                                          &
                   +fctcd2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcd1(fctidx(i))                            &
                   +fctcd2(fctidx(i))*fctmov(i)
         fcttp4   = fctcc2(fctidx(i))*(fcttp1-fctcc3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctcc0(fctidx(i))                            &
                   +fctcc1(fctidx(i))*fcttp6
         fcttp8   = fctcc1(fctidx(i))*fctcc2(fctidx(i))          &
                   /(one/fcttp5+two+fcttp5)
         fctusg(i)= fcttp7
         fctudv(i)= fcttp2*fcttp8
         fctuds(i)= fcttp3*fcttp8
         if (fctusg(i) < 0.001d0) then
            call wrndie(-4,'<FCTPRT>','Surface too low.')
         endif

         fcttp1   =                   fctmov(i)           &
                   +fctcf1(fctidx(i))          *fctmos(i) &
                   +fctcf2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctvos
         fcttp2   = one                                   &
                   +fctcf2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcf1(fctidx(i))                     &
                   +fctcf2(fctidx(i))*fctmov(i)
         fcttp4   = fctce2(fctidx(i))*(fcttp1-fctce3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctce0(fctidx(i))                     &
                   +fctce1(fctidx(i))*fcttp6
         fcttp8   = fctce1(fctidx(i))*fctce2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctvsg(i)= fcttp7
         fctvdv(i)= fcttp2*fcttp8
         fctvds(i)= fcttp3*fcttp8

         fcttp1   =                   fctmov(i)           &
                   +fctch1(fctidx(i))          *fctmos(i) &
                   +fctch2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctwos
         fcttp2   = one                                   &
                   +fctch2(fctidx(i))          *fctmos(i)
         fcttp3   = fctch1(fctidx(i))                     &
                   +fctch2(fctidx(i))*fctmov(i)
         fcttp4   = fctcg2(fctidx(i))*(fcttp1-fctcg3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctcg0(fctidx(i))                     &
                   +fctcg1(fctidx(i))*fcttp6
         fcttp8   = fctcg1(fctidx(i))*fctcg2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctwsg(i)= fcttp7
         fctwdv(i)= fcttp2*fcttp8
         fctwds(i)= fcttp3*fcttp8

         fctisg(i)= one/fctqsg(i)

         ! VdW TERM ================================================
         fctvdw01    = fcttau*fctisg(i)/two
         fctvdw02    = one/(-fctvdw01+fctvdwwrd)
         fctvdwen(i) = fctvdwfac(i)*fctvdw02**3
         fctvdwdf(i) =-three*fctvdwen(i)*fctvdw02*fctvdw01*fctisg(i)
         ! =========================================================
         fct04xs  =(fct03ss(i)*fct01xs(i)          &
                   +fct01xs(i)*fct01xx(i)          &
                   +fct01ys(i)*fct01xy(i)          &
                   +fct01zs(i)*fct01xz(i))         &
                   *fctis1(i) *fctis2(i)           &
                   -fct03xs(i)*fctis2(i)*fctmos(i)
         fct04ys  =(fct03ss(i)*fct01ys(i)          &
                   +fct01xs(i)*fct01xy(i)          &
                   +fct01ys(i)*fct01yy(i)          &
                   +fct01zs(i)*fct01yz(i))         &
                   *fctis1(i) *fctis2(i)           &
                   -fct03ys(i)*fctis2(i)*fctmos(i)
         fct04zs  =(fct03ss(i)*fct01zs(i)          &
                   +fct01xs(i)*fct01xz(i)          &
                   +fct01ys(i)*fct01yz(i)          &
                   +fct01zs(i)*fct01zz(i))         &
                   *fctis1(i) *fctis2(i)           &
                   -fct03zs(i)*fctis2(i)*fctmos(i)

         fctqsx(i)=-fct02xs(i)*fctqdv(i)-fct04xs*fctqds(i)
         fctqsy(i)=-fct02ys(i)*fctqdv(i)-fct04ys*fctqds(i)
         fctqsz(i)=-fct02zs(i)*fctqdv(i)-fct04zs*fctqds(i)
         fctusx(i)=-fct02xs(i)*fctudv(i)-fct04xs*fctuds(i)
         fctusy(i)=-fct02ys(i)*fctudv(i)-fct04ys*fctuds(i)
         fctusz(i)=-fct02zs(i)*fctudv(i)-fct04zs*fctuds(i)
         fctvsx(i)=-fct02xs(i)*fctvdv(i)-fct04xs*fctvds(i)
         fctvsy(i)=-fct02ys(i)*fctvdv(i)-fct04ys*fctvds(i)
         fctvsz(i)=-fct02zs(i)*fctvdv(i)-fct04zs*fctvds(i)
         fctwsx(i)=-fct02xs(i)*fctwdv(i)-fct04xs*fctwds(i)
         fctwsy(i)=-fct02ys(i)*fctwdv(i)-fct04ys*fctwds(i)
         fctwsz(i)=-fct02zs(i)*fctwdv(i)-fct04zs*fctwds(i)

         if(imove(i)==0) then                                        ! # # imove
         fctpl1   = fctpl1+fctcgs(i)*fctqsg(i)

         fctnp1   = fctnp1+fctnpc(i)*fctqsg(i)
         fctnp2   = fctnp2+fctusg(i)*fctvsg(i)
         fctnp3   = fctnp3+fctwsg(i)+fctvdwen(i)

         fctssx(i)=  fctssx(i)                                  &
                   +(fctvdwdf(i)+fctcgs(i)+fctnpc(i))*fctqsx(i) &
                   +                       fctvsg(i) *fctusx(i) &
                   +                       fctusg(i) *fctvsx(i) &
                   +                                  fctwsx(i)
         fctssy(i)=  fctssy(i)                                  &
                   +(fctvdwdf(i)+fctcgs(i)+fctnpc(i))*fctqsy(i) &
                   +                       fctvsg(i) *fctusy(i) &
                   +                       fctusg(i) *fctvsy(i) &
                   +                                  fctwsy(i)
         fctssz(i)=  fctssz(i)                                  &
                   +(fctvdwdf(i)+fctcgs(i)+fctnpc(i))*fctqsz(i) &
                   +                       fctvsg(i) *fctusz(i) &
                   +                       fctusg(i) *fctvsz(i) &
                   +                                  fctwsz(i)
         endif
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct2ll(i-1)+1
         endif
         b=fct2ll(i)
         do j=a,b
            u=i
            v=fct2lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if (fctdvu < fct2cn) then
               fct01vu  = one/fctdvu
               fct02vu  = one-fctdvu*fct2ci
               fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
               fct04vu  = dexp(-fctikp*fct03vu)
               fct05vu  = fct04vu/fct03vu
               fct06vu  = fct05vu+fctikp*fct04vu
               fct07vu  = one/(one+fct05vu)
               fct08vu  =-cg(v)*cg(u)*fcttau*fct02vu              &
                         *sqrt(fct01vu*fct07vu)
               fct09vu  = (fct01vu*fct02vu*(-one+fct06vu*fct07vu) &
                          +fct2fn)*fct08vu
               fct10vu  = half*fct02vu*fct06vu*fct07vu*fct08vu
               fctpl2   = fctpl2+fct02vu*fct08vu
               fctggg(u)= fctggg(u)-fct10vu
               fctggg(v)= fctggg(v)-fct10vu
               fctiix(u)= fctiix(u)                               &
                         -fct09vu*vux+fct10vu*fctisg(u)*fctqsx(u)
               fctiiy(u)= fctiiy(u)                               &
                         -fct09vu*vuy+fct10vu*fctisg(u)*fctqsy(u)
               fctiiz(u)= fctiiz(u)                               &
                         -fct09vu*vuz+fct10vu*fctisg(u)*fctqsz(u)
               fctiix(v)= fctiix(v)                               &
                         +fct09vu*vux+fct10vu*fctisg(v)*fctqsx(v)
               fctiiy(v)= fctiiy(v)                               &
                         +fct09vu*vuy+fct10vu*fctisg(v)*fctqsy(v)
               fctiiz(v)= fctiiz(v)                               &
                         +fct09vu*vuz+fct10vu*fctisg(v)*fctqsz(v)
               endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct4ll(i-1)+1
         endif
         b=fct4ll(i)
         do j=a,b
            u=i
            v=fct4lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if (fctdvu < fct2cn) then
               fct01vu  = one/fctdvu
               fct02vu  = one-fctdvu*fct2ci
               fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
               fct04vu  = dexp(-fctikp*fct03vu)
               fct05vu  = fct04vu/fct03vu
               fct06vu  = fct05vu+fctikp*fct04vu
               fct07vu  = one/(one+fct05vu)
               fct08vu  =-cg(v)*cg(u)*fcttau*fct02vu              &
                           *sqrt(fct01vu*fct07vu)
               fct09vu  = (fct01vu*fct02vu*(-one+fct06vu*fct07vu) &
                           +fct2fn)*fct08vu
               fct10vu  = half*fct02vu*fct06vu*fct07vu*fct08vu
               fctpl2   = fctpl2+fct02vu*fct08vu
               fctggg(u)= fctggg(u)-fct10vu
               fctggg(v)= fctggg(v)-fct10vu
               fctiix(u)= fctiix(u)                               &
                         -fct09vu*vux+fct10vu*fctisg(u)*fctqsx(u)
               fctiiy(u)= fctiiy(u)                               &
                         -fct09vu*vuy+fct10vu*fctisg(u)*fctqsy(u)
               fctiiz(u)= fctiiz(u)                               &
                         -fct09vu*vuz+fct10vu*fctisg(u)*fctqsz(u)
               fctiix(v)= fctiix(v)                               &
                         +fct09vu*vux+fct10vu*fctisg(v)*fctqsx(v)
               fctiiy(v)= fctiiy(v)                               &
                         +fct09vu*vuy+fct10vu*fctisg(v)*fctqsy(v)
               fctiiz(v)= fctiiz(v)                               &
                         +fct09vu*vuz+fct10vu*fctisg(v)*fctqsz(v)
               endif
         enddo
      enddo

      do i=1,reanat
         fcthhh(i)= fctcgs(i)+fctnpc(i)-fctisg(i)*fctggg(i)+fctvdwdf(i)
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            u=i
            v=fct1lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               if(imove(v)==0) then                                    ! # # imove
               fct03vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))
               fct06vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctivu
               fct02vuxs = fct05vuss*vux
               fct02vuys = fct05vuss*vuy
               fct02vuzs = fct05vuss*vuz
               fct03vuxs = (-fct04vuss+fct06vuss)*vux
               fct03vuys = (-fct04vuss+fct06vuss)*vuy
               fct03vuzs = (-fct04vuss+fct06vuss)*vuz
               fct01vuxx = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01vuxy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01vuxz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01vuyy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01vuyz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01vuzz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

               fct04vuxs =(fct03vuss *fct01xs(u)          &
                          +fct01xs(u)*fct01vuxx           &
                          +fct01ys(u)*fct01vuxy           &
                          +fct01zs(u)*fct01vuxz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuxs *fctis2(u)*fctmos(u)
               fct04vuys =(fct03vuss *fct01ys(u)          &
                          +fct01xs(u)*fct01vuxy           &
                          +fct01ys(u)*fct01vuyy           &
                          +fct01zs(u)*fct01vuyz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuys *fctis2(u)*fctmos(u)
               fct04vuzs =(fct03vuss *fct01zs(u)          &
                          +fct01xs(u)*fct01vuxz           &
                          +fct01ys(u)*fct01vuyz           &
                          +fct01zs(u)*fct01vuzz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuzs *fctis2(u)*fctmos(u)

               fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
               fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
               fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
               fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
               fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
               fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)
               fctvsvux  = fct02vuxs*fctvdv(u)+fct04vuxs*fctvds(u)
               fctvsvuy  = fct02vuys*fctvdv(u)+fct04vuys*fctvds(u)
               fctvsvuz  = fct02vuzs*fctvdv(u)+fct04vuzs*fctvds(u)
               fctwsvux  = fct02vuxs*fctwdv(u)+fct04vuxs*fctwds(u)
               fctwsvuy  = fct02vuys*fctwdv(u)+fct04vuys*fctwds(u)
               fctwsvuz  = fct02vuzs*fctwdv(u)+fct04vuzs*fctwds(u)

               fctsix(v) = fctsix(v)          &
                          +fcthhh(u)*fctqsvux &
                          +fctvsg(u)*fctusvux &
                          +fctusg(u)*fctvsvux &
                          +          fctwsvux
               fctsiy(v) = fctsiy(v)          &
                          +fcthhh(u)*fctqsvuy &
                          +fctvsg(u)*fctusvuy &
                          +fctusg(u)*fctvsvuy &
                          +          fctwsvuy
               fctsiz(v) = fctsiz(v)          &
                          +fcthhh(u)*fctqsvuz &
                          +fctvsg(u)*fctusvuz &
                          +fctusg(u)*fctvsvuz &
                          +          fctwsvuz
               endif                                                   ! # # imove
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               if(imove(u)==0) then                                    ! # # imove
               fct03uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))
               fct06uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctivu
               fct02uvxs =-fct05uvss*vux
               fct02uvys =-fct05uvss*vuy
               fct02uvzs =-fct05uvss*vuz
               fct03uvxs =-(-fct04uvss+fct06uvss)*vux
               fct03uvys =-(-fct04uvss+fct06uvss)*vuy
               fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
               fct01uvxx = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01uvxy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01uvxz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01uvyy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01uvyz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01uvzz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

               fct04uvxs = (fct03uvss *fct01xs(v)          &
                           +fct01xs(v)*fct01uvxx           &
                           +fct01ys(v)*fct01uvxy           &
                           +fct01zs(v)*fct01uvxz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvxs *fctis2(v)*fctmos(v)
               fct04uvys = (fct03uvss *fct01ys(v)          &
                           +fct01xs(v)*fct01uvxy           &
                           +fct01ys(v)*fct01uvyy           &
                           +fct01zs(v)*fct01uvyz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvys *fctis2(v)*fctmos(v)
               fct04uvzs = (fct03uvss *fct01zs(v)          &
                           +fct01xs(v)*fct01uvxz           &
                           +fct01ys(v)*fct01uvyz           &
                           +fct01zs(v)*fct01uvzz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvzs *fctis2(v)*fctmos(v)

               fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
               fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
               fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
               fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
               fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
               fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)
               fctvsuvx  = fct02uvxs*fctvdv(v)+fct04uvxs*fctvds(v)
               fctvsuvy  = fct02uvys*fctvdv(v)+fct04uvys*fctvds(v)
               fctvsuvz  = fct02uvzs*fctvdv(v)+fct04uvzs*fctvds(v)
               fctwsuvx  = fct02uvxs*fctwdv(v)+fct04uvxs*fctwds(v)
               fctwsuvy  = fct02uvys*fctwdv(v)+fct04uvys*fctwds(v)
               fctwsuvz  = fct02uvzs*fctwdv(v)+fct04uvzs*fctwds(v)

               fctsix(u) = fctsix(u)          &
                          +fcthhh(v)*fctqsuvx &
                          +fctvsg(v)*fctusuvx &
                          +fctusg(v)*fctvsuvx &
                          +          fctwsuvx
               fctsiy(u) = fctsiy(u)          &
                          +fcthhh(v)*fctqsuvy &
                          +fctvsg(v)*fctusuvy &
                          +fctusg(v)*fctvsuvy &
                          +          fctwsuvy
               fctsiz(u) = fctsiz(u)          &
                          +fcthhh(v)*fctqsuvz &
                          +fctvsg(v)*fctusuvz &
                          +fctusg(v)*fctvsuvz &
                          +          fctwsuvz
               endif                                                   ! # # imove
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               if(imove(v)==0) then                                    ! # # imove
               fct03vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctivu
               fct04vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu*fctivu
               fct05vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))
               fct06vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctsvu
               fct07vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctivu
               fct02vuxs = fct05vuss*vux
               fct02vuys = fct05vuss*vuy
               fct02vuzs = fct05vuss*vuz
               fct03vuxs = (-fct04vuss+fct06vuss)*vux
               fct03vuys = (-fct04vuss+fct06vuss)*vuy
               fct03vuzs = (-fct04vuss+fct06vuss)*vuz
               fct01vuxx = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01vuxy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01vuxz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01vuyy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01vuyz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01vuzz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

               fct04vuxs =( fct03vuss *fct01xs(u)          &
                           +fct01xs(u)*fct01vuxx           &
                           +fct01ys(u)*fct01vuxy           &
                           +fct01zs(u)*fct01vuxz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuxs *fctis2(u)*fctmos(u)
               fct04vuys =( fct03vuss *fct01ys(u)          &
                           +fct01xs(u)*fct01vuxy           &
                           +fct01ys(u)*fct01vuyy           &
                           +fct01zs(u)*fct01vuyz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuys *fctis2(u)*fctmos(u)
               fct04vuzs =( fct03vuss *fct01zs(u)          &
                           +fct01xs(u)*fct01vuxz           &
                           +fct01ys(u)*fct01vuyz           &
                           +fct01zs(u)*fct01vuzz)          &
                           *fctis1(u) *fctis2(u)           &
                           -fct03vuzs *fctis2(u)*fctmos(u)

               fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
               fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
               fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
               fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
               fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
               fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)
               fctvsvux  = fct02vuxs*fctvdv(u)+fct04vuxs*fctvds(u)
               fctvsvuy  = fct02vuys*fctvdv(u)+fct04vuys*fctvds(u)
               fctvsvuz  = fct02vuzs*fctvdv(u)+fct04vuzs*fctvds(u)
               fctwsvux  = fct02vuxs*fctwdv(u)+fct04vuxs*fctwds(u)
               fctwsvuy  = fct02vuys*fctwdv(u)+fct04vuys*fctwds(u)
               fctwsvuz  = fct02vuzs*fctwdv(u)+fct04vuzs*fctwds(u)

               fctsix(v) =  fctsix(v)          &
                           +fcthhh(u)*fctqsvux &
                           +fctvsg(u)*fctusvux &
                           +fctusg(u)*fctvsvux &
                           +          fctwsvux
               fctsiy(v) =  fctsiy(v)          &
                           +fcthhh(u)*fctqsvuy &
                           +fctvsg(u)*fctusvuy &
                           +fctusg(u)*fctvsvuy &
                           +          fctwsvuy
               fctsiz(v) =  fctsiz(v)          &
                           +fcthhh(u)*fctqsvuz &
                           +fctvsg(u)*fctusvuz &
                           +fctusg(u)*fctvsvuz &
                           +          fctwsvuz
               endif                                                   ! # # imove
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               if(imove(u)==0) then                                    ! # # imove
               fct03uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))
               fct06uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctivu
               fct02uvxs =-fct05uvss*vux
               fct02uvys =-fct05uvss*vuy
               fct02uvzs =-fct05uvss*vuz
               fct03uvxs =-(-fct04uvss+fct06uvss)*vux
               fct03uvys =-(-fct04uvss+fct06uvss)*vuy
               fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
               fct01uvxx = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01uvxy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01uvxz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01uvyy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01uvyz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01uvzz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

               fct04uvxs = (fct03uvss *fct01xs(v)          &
                           +fct01xs(v)*fct01uvxx           &
                           +fct01ys(v)*fct01uvxy           &
                           +fct01zs(v)*fct01uvxz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvxs *fctis2(v)*fctmos(v)
               fct04uvys = (fct03uvss *fct01ys(v)          &
                           +fct01xs(v)*fct01uvxy           &
                           +fct01ys(v)*fct01uvyy           &
                           +fct01zs(v)*fct01uvyz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvys *fctis2(v)*fctmos(v)
               fct04uvzs = (fct03uvss *fct01zs(v)          &
                           +fct01xs(v)*fct01uvxz           &
                           +fct01ys(v)*fct01uvyz           &
                           +fct01zs(v)*fct01uvzz)          &
                           *fctis1(v) *fctis2(v)           &
                           -fct03uvzs *fctis2(v)*fctmos(v)

               fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
               fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
               fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
               fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
               fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
               fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)
               fctvsuvx  = fct02uvxs*fctvdv(v)+fct04uvxs*fctvds(v)
               fctvsuvy  = fct02uvys*fctvdv(v)+fct04uvys*fctvds(v)
               fctvsuvz  = fct02uvzs*fctvdv(v)+fct04uvzs*fctvds(v)
               fctwsuvx  = fct02uvxs*fctwdv(v)+fct04uvxs*fctwds(v)
               fctwsuvy  = fct02uvys*fctwdv(v)+fct04uvys*fctwds(v)
               fctwsuvz  = fct02uvzs*fctwdv(v)+fct04uvzs*fctwds(v)

               fctsix(u) =  fctsix(u)          &
                           +fcthhh(v)*fctqsuvx &
                           +fctvsg(v)*fctusuvx &
                           +fctusg(v)*fctvsuvx &
                           +          fctwsuvx
               fctsiy(u) =  fctsiy(u)          &
                           +fcthhh(v)*fctqsuvy &
                           +fctvsg(v)*fctusuvy &
                           +fctusg(v)*fctvsuvy &
                           +          fctwsuvy
               fctsiz(u) =  fctsiz(u)          &
                           +fcthhh(v)*fctqsuvz &
                           +fctvsg(v)*fctusuvz &
                           +fctusg(v)*fctvsuvz &
                           +          fctwsuvz
               endif                                                   ! # # imove
            endif
         enddo
      enddo

      fctpol=fctpl1+fctpl2
      fctnpl=fctnp1+fctnp2+fctnp3

      if (fctpsl) then
         do i=1,reanat
            wmain(i)=fctqsg(i)/fctqun(fctidx(i))
         enddo
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(A32)')                                      &
            ' FCTPRT> Atomic Self Terms (ST):'
         write(outu,'(A32)')                                      &
            ' FCTPRT> ======================='
         write(outu,'(A1)')                                       &
            ''
         ! write(outu,'(2A64)')
         write(outu,'(2A)')                                       &
            ' FCTPRT> ST unit charge: elec. Self energy',         &
            ' Term with unit charge'
         write(outu,'(A1)')                                       &
            ''
         ! write(outu,'(2A71)')
         write(outu,'(2A)')                                       &
            ' FCTPRT> ST real charge: elec. Self energy',         &
            ' Term with real/actual charge'
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(3A)')                                       &
            ' FCTPRT> ST npol charge: nonpolar contribution',     &
            ' proportional to the unit charge atomic',            &
            ' self energy'
         write(outu,'(2A)')                                       &
            ' FCTPRT> ST Energy: Self Term energy',               &
            ' (ST real charge + ST npol charge)'
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(10A17)')                                    &
            '      Atom number',                                  &
            '      FACTS index',                                  &
            '           Volume',                                  &
            '         Symmetry',                                  &
            '                C',                                  &
            '   Quenching fac.',                                  &
            '   ST unit charge',                                  &
            '   ST real charge',                                  &
            '   ST npol charge',                                  &
            '        ST Energy'
         write(outu,'(A1)')                                       &
            ''
         do i=1,reanat
            write(outu,'(A8,I9,I17,8F17.5)')                      &
               ' FCTSLF:',                                        &
               i,                                                 &
               fctidx(i),                                         &
               fctmov(i),                                         &
               fctmos(i),                                         &
                                 fctmov(i)                        &
              +fctcb1(fctidx(i))          *fctmos(i)              &
              +fctcb2(fctidx(i))*fctmov(i)*fctmos(i)              &
              +fctqos,                                            &
               fctqun(fctidx(i)),                                 &
               fctqsg(i),                                         &
               fctcgs(i)*fctqsg(i),                               &
               fctnpc(i)*fctqsg(i),                               &
               (fctcgs(i)+fctnpc(i))*fctqsg(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
      endif

      if (fctpsr) then
         do i=1,reanat
            wmain(i)=fctusg(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(A49)')                                      &
            ' FCTPRT> Atomic solvent accessible surface areas:'
         write(outu,'(A49)')                                      &
            ' FCTPRT> ========================================'
         write(outu,'(2A)')                                       &
            ' FCTPRT> SAS-Energy: Surface related nonpolar',      &
            ' contribution to the energy'
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(7A17)')                                     &
            '      Atom number',                                  &
            '      FACTS index',                                  &
            '           Volume',                                  &
            '         Symmetry',                                  &
            '                D',                                  &
            '             SASA',                                  &
            '       SAS-Energy'
         write(outu,'(A1)')                                       &
            ''
         do i=1,reanat
            write(outu,'(A8,I9,I17,5F17.5)')                      &
               ' FCTSRF:',                                        &
               i,                                                 &
               fctidx(i),                                         &
               fctmov(i),                                         &
               fctmos(i),                                         &
                                 fctmov(i)                        &
              +fctcd1(fctidx(i))          *fctmos(i)              &
              +fctcd2(fctidx(i))*fctmov(i)*fctmos(i)              &
              +fctuos,                                            &
               fctusg(i),                                         &
               fctusg(i)*fctvsg(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
      endif

      if (fctpal) then
         do i=1,reanat
            wmain(i)=fctvsg(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(A15)')                                      &
            ' FCTPRT> Alpha:'
         write(outu,'(A15)')                                      &
            ' FCTPRT> ======'
         write(outu,'(2A)')                                       &
            ' FCTPRT> Alpha: Surface tension sigmoidal function'
         write(outu,'(2A)')                                       &
            ' FCTPRT> SAS-Energy: Surface related nonpolar',      &
            ' contribution to the energy'
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(7A17)')                                     &
            '      Atom number',                                  &
            '      FACTS index',                                  &
            '           Volume',                                  &
            '         Symmetry',                                  &
            '                E',                                  &
            '            Alpha',                                  &
            '       SAS-Energy'
         write(outu,'(A1)')                                       &
            ''
         do i=1,reanat
            write(outu,'(A8,I9,I17,5F17.5)')                      &
               ' FCTALP:',                                        &
               i,                                                 &
               fctidx(i),                                         &
               fctmov(i),                                         &
               fctmos(i),                                         &
                                 fctmov(i)                        &
              +fctcf1(fctidx(i))          *fctmos(i)              &
              +fctcf2(fctidx(i))*fctmov(i)*fctmos(i)              &
              +fctvos,                                            &
               fctvsg(i),                                         &
               fctusg(i)*fctvsg(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
      endif

      if (fctpbt) then
         do i=1,reanat
            wmain(i)=fctwsg(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(A14)')                                      &
            ' FCTPRT> Beta:'
         write(outu,'(A14)')                                      &
            ' FCTPRT> ====='
         write(outu,'(2A)')                                       &
            ' FCTPRT> Beta: Volume related nonpolar',             &
            ' contribution to the energy'
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(6A17)')                                     &
            '      Atom number',                                  &
            '      FACTS index',                                  &
            '           Volume',                                  &
            '         Symmetry',                                  &
            '                G',                                  &
            '             Beta'
         write(outu,'(A1)')                                       &
            ''
         do i=1,reanat
            write(outu,'(A8,I9,I17,4F17.5)')                      &
               ' FCTBET:',                                        &
               i,                                                 &
               fctidx(i),                                         &
               fctmov(i),                                         &
               fctmos(i),                                         &
                                 fctmov(i)                        &
              +fctch1(fctidx(i))          *fctmos(i)              &
              +fctch2(fctidx(i))*fctmov(i)*fctmos(i)              &
              +fctwos,                                            &
               fctwsg(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
      endif
      ! VdW-PRINTING ============================================
      if (fctpvw) then
         do i=1,reanat
            wmain(i)=fctwsg(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(A33)')                                      &
            ' FCTPRT> Solute-Solvent VdW term:'
         write(outu,'(A33)')                                      &
            ' FCTPRT> ========================'
         write(outu,'(A26)')                                      &
            ' FCTPRT> R_GB: Born radius'
         write(outu,'(A30)')                                      &
            ' FCTPRT> Alpha: scaling factor'
         write(outu,'(2A)')                                       &
            ' FCTPRT> VdW-Energy: solute-solvent',                &
            ' dispersion term'
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(7A17)')                                     &
            '      Atom number',                                  &
            '      FACTS index',                                  &
            '           Volume',                                  &
            '         Symmetry',                                  &
            '             R_GB',                                  &
            '            Alpha',                                  &
            '       VdW-Energy'
         write(outu,'(A1)')                                       &
            ''
         do i=1,reanat
            write(outu,'(A8,I9,I17,5F17.5)')                      &
               ' FCTVDW:',                                        &
               i,                                                 &
               fctidx(i),                                         &
               fctmov(i),                                         &
               fctmos(i),                                         &
              -fcttau*fctisg(i)/two,                              &
               fctvdwalp(i),                                      &
               fctvdwen(i)
         enddo
         write(outu,'(A1)')                                       &
            ''
      endif
      ! =========================================================
      if (fctpfr) then
         do i=1,reanat
            wmain(i)=sqrt(                                        &
               (fctssx(i)+fctsix(i)+fctiix(i))                    &
               *(fctssx(i)+fctsix(i)+fctiix(i))                    &
               +(fctssy(i)+fctsiy(i)+fctiiy(i))                    &
               *(fctssy(i)+fctsiy(i)+fctiiy(i))                    &
               +(fctssz(i)+fctsiz(i)+fctiiz(i))                    &
               *(fctssz(i)+fctsiz(i)+fctiiz(i)))
         enddo
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(A33)')                                      &
            ' FCTPRT> Atomic solvation forces:'
         write(outu,'(A33)')                                      &
            ' FCTPRT> ========================'
         write(outu,'(A1)')                                       &
            ''
         write(outu,'(6A17)')                                     &
            '      Atom number',                                  &
            '      FACTS index',                                  &
            '               DX',                                  &
            '               DY',                                  &
            '               DZ',                                  &
            '   Absolute value'
         write(outu,'(A1)')                                       &
            ''
         do i=1,reanat
            write(outu,'(A8,I9,I17,4F17.5)')                      &
               ' FCTFRC:',                                        &
               i,                                                 &
               fctidx(i),                                         &
               fctssx(i)+fctsix(i)+fctiix(i),                     &
               fctssy(i)+fctsiy(i)+fctiiy(i),                     &
               fctssz(i)+fctsiz(i)+fctiiz(i),                     &
               sqrt(                                              &
               (fctssx(i)+fctsix(i)+fctiix(i))                    &
              *(fctssx(i)+fctsix(i)+fctiix(i))                    &
              +(fctssy(i)+fctsiy(i)+fctiiy(i))                    &
              *(fctssy(i)+fctsiy(i)+fctiiy(i))                    &
              +(fctssz(i)+fctsiz(i)+fctiiz(i))                    &
              *(fctssz(i)+fctsiz(i)+fctiiz(i)))
         enddo
         write(outu,'(A1)')                                       &
            ''
      endif
   endif

   if (fctscr) then

      fctpl1=zero
      fctpl2=zero
      fctnp1=zero
      fctnp2=zero
      fctnp3=zero

      do i=1,reanat
         fct01ss(i)=zero
         fct02ss(i)=fctbin
         fct03ss(i)=zero
         fct01xs(i)=zero
         fct01ys(i)=zero
         fct01zs(i)=zero
         fct02xs(i)=zero
         fct02ys(i)=zero
         fct02zs(i)=zero
         fct03xs(i)=zero
         fct03ys(i)=zero
         fct03zs(i)=zero
         fct01xx(i)=zero
         fct01xy(i)=zero
         fct01xz(i)=zero
         fct01yy(i)=zero
         fct01yz(i)=zero
         fct01zz(i)=zero

         fctggg(i) =zero

         fctssx(i) =zero
         fctssy(i) =zero
         fctssz(i) =zero
         fctsix(i) =zero
         fctsiy(i) =zero
         fctsiz(i) =zero
         fctiix(i) =zero
         fctiiy(i) =zero
         fctiiz(i) =zero
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            u=i
            v=fct1lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct01vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u)))
               fct02vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu
               fct03vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctivu
               fct04vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu*fctivu
               fct05vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))
               fct06vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctsvu
               fct07vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctivu
               fct01ss(u)= fct01ss(u)+fct01vuss
               fct02ss(u)= fct02ss(u)+fct02vuss
               fct03ss(u)= fct03ss(u)+fct03vuss
               fct01xs(u)= fct01xs(u)+fct03vuss*vux
               fct01ys(u)= fct01ys(u)+fct03vuss*vuy
               fct01zs(u)= fct01zs(u)+fct03vuss*vuz
               fct02xs(u)= fct02xs(u)+fct05vuss*vux
               fct02ys(u)= fct02ys(u)+fct05vuss*vuy
               fct02zs(u)= fct02zs(u)+fct05vuss*vuz
               fct03xs(u)= fct03xs(u)+(-fct04vuss+fct06vuss)*vux
               fct03ys(u)= fct03ys(u)+(-fct04vuss+fct06vuss)*vuy
               fct03zs(u)= fct03zs(u)+(-fct04vuss+fct06vuss)*vuz
               fct01xx(u)= fct01xx(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01xy(u)= fct01xy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01xz(u)= fct01xz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01yy(u)= fct01yy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01yz(u)= fct01yz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01zz(u)= fct01zz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct01uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v)))
               fct02uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu
               fct03uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctivu
               fct04uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu*fctivu
               fct05uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))
               fct06uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctsvu
               fct07uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctivu
               fct01ss(v)= fct01ss(v)+fct01uvss
               fct02ss(v)= fct02ss(v)+fct02uvss
               fct03ss(v)= fct03ss(v)+fct03uvss
               fct01xs(v)= fct01xs(v)-fct03uvss*vux
               fct01ys(v)= fct01ys(v)-fct03uvss*vuy
               fct01zs(v)= fct01zs(v)-fct03uvss*vuz
               fct02xs(v)= fct02xs(v)-fct05uvss*vux
               fct02ys(v)= fct02ys(v)-fct05uvss*vuy
               fct02zs(v)= fct02zs(v)-fct05uvss*vuz
               fct03xs(v)= fct03xs(v)-(-fct04uvss+fct06uvss)*vux
               fct03ys(v)= fct03ys(v)-(-fct04uvss+fct06uvss)*vuy
               fct03zs(v)= fct03zs(v)-(-fct04uvss+fct06uvss)*vuz
               fct01xx(v)= fct01xx(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01xy(v)= fct01xy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01xz(v)= fct01xz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01yy(v)= fct01yy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01yz(v)= fct01yz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01zz(v)= fct01zz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct01vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u)))
               fct02vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu
               fct03vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctivu
               fct04vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fctsvu*fctivu
               fct05vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))
               fct06vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctsvu
               fct07vuss =  fctvol(fctidx(v))              &
                           *(one-fctdvu*fct1ci(fctidx(u))) &
                           *fct1fn(fctidx(u))              &
                           *fctivu
               fct01ss(u)= fct01ss(u)+fct01vuss
               fct02ss(u)= fct02ss(u)+fct02vuss
               fct03ss(u)= fct03ss(u)+fct03vuss
               fct01xs(u)= fct01xs(u)+fct03vuss*vux
               fct01ys(u)= fct01ys(u)+fct03vuss*vuy
               fct01zs(u)= fct01zs(u)+fct03vuss*vuz
               fct02xs(u)= fct02xs(u)+fct05vuss*vux
               fct02ys(u)= fct02ys(u)+fct05vuss*vuy
               fct02zs(u)= fct02zs(u)+fct05vuss*vuz
               fct03xs(u)= fct03xs(u)+(-fct04vuss+fct06vuss)*vux
               fct03ys(u)= fct03ys(u)+(-fct04vuss+fct06vuss)*vuy
               fct03zs(u)= fct03zs(u)+(-fct04vuss+fct06vuss)*vuz
               fct01xx(u)= fct01xx(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01xy(u)= fct01xy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01xz(u)= fct01xz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01yy(u)= fct01yy(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01yz(u)= fct01yz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01zz(u)= fct01zz(u)+half*fct07vuss              &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct01uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v)))
               fct02uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu
               fct03uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctivu
               fct04uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fctsvu*fctivu
               fct05uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))
               fct06uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctsvu
               fct07uvss =  fctvol(fctidx(u))              &
                           *(one-fctdvu*fct1ci(fctidx(v))) &
                           *fct1fn(fctidx(v))              &
                           *fctivu
               fct01ss(v)= fct01ss(v)+fct01uvss
               fct02ss(v)= fct02ss(v)+fct02uvss
               fct03ss(v)= fct03ss(v)+fct03uvss
               fct01xs(v)= fct01xs(v)-fct03uvss*vux
               fct01ys(v)= fct01ys(v)-fct03uvss*vuy
               fct01zs(v)= fct01zs(v)-fct03uvss*vuz
               fct02xs(v)= fct02xs(v)-fct05uvss*vux
               fct02ys(v)= fct02ys(v)-fct05uvss*vuy
               fct02zs(v)= fct02zs(v)-fct05uvss*vuz
               fct03xs(v)= fct03xs(v)-(-fct04uvss+fct06uvss)*vux
               fct03ys(v)= fct03ys(v)-(-fct04uvss+fct06uvss)*vuy
               fct03zs(v)= fct03zs(v)-(-fct04uvss+fct06uvss)*vuz
               fct01xx(v)= fct01xx(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01xy(v)= fct01xy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01xz(v)= fct01xz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01yy(v)= fct01yy(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01yz(v)= fct01yz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01zz(v)= fct01zz(v)+half*fct07uvss              &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz
            endif
         enddo
      enddo

      do i=1,reanat
         fcttp1   = sqrt( fct01xs(i)*fct01xs(i)  &
                         +fct01ys(i)*fct01ys(i)  &
                         +fct01zs(i)*fct01zs(i))
         fcttp2   = fct02ss(i)

         fctis1(i)= one/fcttp1
         fctis2(i)= one/fcttp2

         fctmov(i)= fct01ss(i)
         fctmos(i)= fcttp1*fctis2(i)

         fcttp1   =                   fctmov(i)           &
                   +fctcb1(fctidx(i))          *fctmos(i) &
                   +fctcb2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctqos
         fcttp2   = one                                   &
                   +fctcb2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcb1(fctidx(i))                     &
                   +fctcb2(fctidx(i))*fctmov(i)
         fcttp4   = fctca2(fctidx(i))*(fcttp1-fctca3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctca0(fctidx(i))                     &
                   +fctca1(fctidx(i))*fcttp6
         fcttp8   = fctca1(fctidx(i))*fctca2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctqsg(i)= fcttp7
         fctqdv(i)= fcttp2*fcttp8
         fctqds(i)= fcttp3*fcttp8
         if (fctqsg(i) > -0.001d0) then
            call wrndie(-4,'<FCTPRT>','Solvation energy too high.')
         endif
         fctqsg(i)= fctqsg(i)*fctqun(fctidx(i))
         fctqdv(i)= fctqdv(i)*fctqun(fctidx(i))
         fctqds(i)= fctqds(i)*fctqun(fctidx(i))

         fcttp1   =                   fctmov(i)           &
                   +fctcd1(fctidx(i))          *fctmos(i) &
                   +fctcd2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctuos
         fcttp2   = one                                   &
                   +fctcd2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcd1(fctidx(i))                     &
                   +fctcd2(fctidx(i))*fctmov(i)
         fcttp4   = fctcc2(fctidx(i))*(fcttp1-fctcc3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctcc0(fctidx(i))                     &
                   +fctcc1(fctidx(i))*fcttp6
         fcttp8   = fctcc1(fctidx(i))*fctcc2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctusg(i)= fcttp7
         fctudv(i)= fcttp2*fcttp8
         fctuds(i)= fcttp3*fcttp8
         if (fctusg(i) < 0.001d0) then
            call wrndie(-4,'<FCTPRT>','Surface too low.')
         endif

         fcttp1   =                   fctmov(i)           &
                   +fctcf1(fctidx(i))          *fctmos(i) &
                   +fctcf2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctvos
         fcttp2   = one                                   &
                   +fctcf2(fctidx(i))          *fctmos(i)
         fcttp3   = fctcf1(fctidx(i))                     &
                   +fctcf2(fctidx(i))*fctmov(i)
         fcttp4   = fctce2(fctidx(i))*(fcttp1-fctce3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctce0(fctidx(i))                     &
                   +fctce1(fctidx(i))*fcttp6
         fcttp8   = fctce1(fctidx(i))*fctce2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctvsg(i)= fcttp7
         fctvdv(i)= fcttp2*fcttp8
         fctvds(i)= fcttp3*fcttp8

         fcttp1   =                   fctmov(i)           &
                   +fctch1(fctidx(i))          *fctmos(i) &
                   +fctch2(fctidx(i))*fctmov(i)*fctmos(i) &
                   +fctwos
         fcttp2   = one                                   &
                   +fctch2(fctidx(i))          *fctmos(i)
         fcttp3   = fctch1(fctidx(i))                     &
                   +fctch2(fctidx(i))*fctmov(i)
         fcttp4   = fctcg2(fctidx(i))*(fcttp1-fctcg3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctcg0(fctidx(i))                     &
                   +fctcg1(fctidx(i))*fcttp6
         fcttp8   = fctcg1(fctidx(i))*fctcg2(fctidx(i))   &
                   /(one/fcttp5+two+fcttp5)
         fctwsg(i)= fcttp7
         fctwdv(i)= fcttp2*fcttp8
         fctwds(i)= fcttp3*fcttp8

         fctisg(i)= one/fctqsg(i)

         fctnp1   = fctnp1+fctnpc(i)*fctqsg(i)
         fctnp2   = fctnp2+fctusg(i)*fctvsg(i)
         fctnp3   = fctnp3+          fctwsg(i)

         fct04xs  =(fct03ss(i)*fct01xs(i)          &
                   +fct01xs(i)*fct01xx(i)          &
                   +fct01ys(i)*fct01xy(i)          &
                   +fct01zs(i)*fct01xz(i))         &
                   *fctis1(i) *fctis2(i)           &
                   -fct03xs(i)*fctis2(i)*fctmos(i)
         fct04ys  =(fct03ss(i)*fct01ys(i)          &
                   +fct01xs(i)*fct01xy(i)          &
                   +fct01ys(i)*fct01yy(i)          &
                   +fct01zs(i)*fct01yz(i))         &
                   *fctis1(i) *fctis2(i)           &
                   -fct03ys(i)*fctis2(i)*fctmos(i)
         fct04zs  =(fct03ss(i)*fct01zs(i)          &
                   +fct01xs(i)*fct01xz(i)          &
                   +fct01ys(i)*fct01yz(i)          &
                   +fct01zs(i)*fct01zz(i))         &
                   *fctis1(i) *fctis2(i)           &
                   -fct03zs(i)*fctis2(i)*fctmos(i)

         fctqsx(i)=-fct02xs(i)*fctqdv(i)-fct04xs*fctqds(i)
         fctqsy(i)=-fct02ys(i)*fctqdv(i)-fct04ys*fctqds(i)
         fctqsz(i)=-fct02zs(i)*fctqdv(i)-fct04zs*fctqds(i)
         fctusx(i)=-fct02xs(i)*fctudv(i)-fct04xs*fctuds(i)
         fctusy(i)=-fct02ys(i)*fctudv(i)-fct04ys*fctuds(i)
         fctusz(i)=-fct02zs(i)*fctudv(i)-fct04zs*fctuds(i)
         fctvsx(i)=-fct02xs(i)*fctvdv(i)-fct04xs*fctvds(i)
         fctvsy(i)=-fct02ys(i)*fctvdv(i)-fct04ys*fctvds(i)
         fctvsz(i)=-fct02zs(i)*fctvdv(i)-fct04zs*fctvds(i)
         fctwsx(i)=-fct02xs(i)*fctwdv(i)-fct04xs*fctwds(i)
         fctwsy(i)=-fct02ys(i)*fctwdv(i)-fct04ys*fctwds(i)
         fctwsz(i)=-fct02zs(i)*fctwdv(i)-fct04zs*fctwds(i)

         fctssx(i)= fctssx(i)           &
                   +fctnpc(i)*fctqsx(i) &
                   +fctvsg(i)*fctusx(i) &
                   +fctusg(i)*fctvsx(i) &
                   +          fctwsx(i)
         fctssy(i)= fctssy(i)           &
                   +fctnpc(i)*fctqsy(i) &
                   +fctvsg(i)*fctusy(i) &
                   +fctusg(i)*fctvsy(i) &
                   +          fctwsy(i)
         fctssz(i)= fctssz(i)           &
                   +fctnpc(i)*fctqsz(i) &
                   +fctvsg(i)*fctusz(i) &
                   +fctusg(i)*fctvsz(i) &
                   +          fctwsz(i)
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=reablo(i-1)+1
         endif
         b=reablo(i)
         do j=a,b
            u=i
            v=abs(reanbo(j))
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if (fctdvu < fct3cn) then
               fct01vu  = one/fctdvu
               fct02vu  = one-fctdvu*fct3ci
               fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
               fct04vu  = dexp(-fctikp*fct03vu)
               fct05vu  = fct04vu/fct03vu
               fct06vu  = fct05vu+fctikp*fct04vu
               fct07vu  = one/(one+fct05vu)
               fct08vu  =-cg(v)*cg(u)*fcttau*fct02vu              &
                         *sqrt(fct01vu*fct07vu)
               if(reanbo(j) < 0) fct08vu=fct08vu*e14fac
               fct09vu  = (fct01vu*fct02vu*(-one+fct06vu*fct07vu) &
                         +fct2fn)*fct08vu
               fct10vu  = half*fct02vu*fct06vu*fct07vu*fct08vu
               fctpl2   = fctpl2+fct02vu*fct08vu
               fctggg(u)= fctggg(u)-fct10vu
               fctggg(v)= fctggg(v)-fct10vu
               fctiix(u)= fctiix(u)                               &
                         -fct09vu*vux+fct10vu*fctisg(u)*fctqsx(u)
               fctiiy(u)= fctiiy(u)                               &
                         -fct09vu*vuy+fct10vu*fctisg(u)*fctqsy(u)
               fctiiz(u)= fctiiz(u)                               &
                         -fct09vu*vuz+fct10vu*fctisg(u)*fctqsz(u)
               fctiix(v)= fctiix(v)                               &
                         +fct09vu*vux+fct10vu*fctisg(v)*fctqsx(v)
               fctiiy(v)= fctiiy(v)                               &
                         +fct09vu*vuy+fct10vu*fctisg(v)*fctqsy(v)
               fctiiz(v)= fctiiz(v)                               &
                         +fct09vu*vuz+fct10vu*fctisg(v)*fctqsz(v)
               endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=imablo(i-1)+1
         endif
         b=imablo(i)
         do j=a,b
            u=i
            v=abs(imanbo(j))
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if (fctdvu < fct3cn) then
               fct01vu  = one/fctdvu
               fct02vu  = one-fctdvu*fct3ci
               fct03vu  = fctpco*fctqsg(v)*fctdvu*fctqsg(u)
               fct04vu  = dexp(-fctikp*fct03vu)
               fct05vu  = fct04vu/fct03vu
               fct06vu  = fct05vu+fctikp*fct04vu
               fct07vu  = one/(one+fct05vu)
               fct08vu  =-cg(v)*cg(u)*fcttau*fct02vu              &
                         *sqrt(fct01vu*fct07vu)
               if(imanbo(j) < 0) fct08vu=fct08vu*e14fac
               fct09vu  = (fct01vu*fct02vu*(-one+fct06vu*fct07vu) &
                          +fct2fn)*fct08vu
               fct10vu  = half*fct02vu*fct06vu*fct07vu*fct08vu
               fctpl2   = fctpl2+fct02vu*fct08vu
               fctggg(u)= fctggg(u)-fct10vu
               fctggg(v)= fctggg(v)-fct10vu
               fctiix(u)= fctiix(u)                               &
                         -fct09vu*vux+fct10vu*fctisg(u)*fctqsx(u)
               fctiiy(u)= fctiiy(u)                               &
                         -fct09vu*vuy+fct10vu*fctisg(u)*fctqsy(u)
               fctiiz(u)= fctiiz(u)                               &
                         -fct09vu*vuz+fct10vu*fctisg(u)*fctqsz(u)
               fctiix(v)= fctiix(v)                               &
                         +fct09vu*vux+fct10vu*fctisg(v)*fctqsx(v)
               fctiiy(v)= fctiiy(v)                               &
                         +fct09vu*vuy+fct10vu*fctisg(v)*fctqsy(v)
               fctiiz(v)= fctiiz(v)                               &
                         +fct09vu*vuz+fct10vu*fctisg(v)*fctqsz(v)
               endif
         enddo
      enddo

      do i=1,reanat
         fcthhh(i)=fctnpc(i)-fctisg(i)*fctggg(i)
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            u=i
            v=fct1lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct03vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))
               fct06vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctivu
               fct02vuxs = fct05vuss*vux
               fct02vuys = fct05vuss*vuy
               fct02vuzs = fct05vuss*vuz
               fct03vuxs = (-fct04vuss+fct06vuss)*vux
               fct03vuys = (-fct04vuss+fct06vuss)*vuy
               fct03vuzs = (-fct04vuss+fct06vuss)*vuz
               fct01vuxx = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01vuxy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01vuxz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01vuyy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01vuyz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01vuzz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

               fct04vuxs =(fct03vuss *fct01xs(u)          &
                          +fct01xs(u)*fct01vuxx           &
                          +fct01ys(u)*fct01vuxy           &
                          +fct01zs(u)*fct01vuxz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuxs *fctis2(u)*fctmos(u)
               fct04vuys =(fct03vuss *fct01ys(u)          &
                          +fct01xs(u)*fct01vuxy           &
                          +fct01ys(u)*fct01vuyy           &
                          +fct01zs(u)*fct01vuyz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuys *fctis2(u)*fctmos(u)
               fct04vuzs =(fct03vuss *fct01zs(u)          &
                          +fct01xs(u)*fct01vuxz           &
                          +fct01ys(u)*fct01vuyz           &
                          +fct01zs(u)*fct01vuzz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuzs *fctis2(u)*fctmos(u)

               fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
               fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
               fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
               fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
               fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
               fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)
               fctvsvux  = fct02vuxs*fctvdv(u)+fct04vuxs*fctvds(u)
               fctvsvuy  = fct02vuys*fctvdv(u)+fct04vuys*fctvds(u)
               fctvsvuz  = fct02vuzs*fctvdv(u)+fct04vuzs*fctvds(u)
               fctwsvux  = fct02vuxs*fctwdv(u)+fct04vuxs*fctwds(u)
               fctwsvuy  = fct02vuys*fctwdv(u)+fct04vuys*fctwds(u)
               fctwsvuz  = fct02vuzs*fctwdv(u)+fct04vuzs*fctwds(u)

               fctsix(v) = fctsix(v)          &
                          +fcthhh(u)*fctqsvux &
                          +fctvsg(u)*fctusvux &
                          +fctusg(u)*fctvsvux &
                          +          fctwsvux
               fctsiy(v) = fctsiy(v)          &
                          +fcthhh(u)*fctqsvuy &
                          +fctvsg(u)*fctusvuy &
                          +fctusg(u)*fctvsvuy &
                          +          fctwsvuy
               fctsiz(v) = fctsiz(v)          &
                          +fcthhh(u)*fctqsvuz &
                          +fctvsg(u)*fctusvuz &
                          +fctusg(u)*fctvsvuz &
                          +          fctwsvuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct03uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))
               fct06uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctivu
               fct02uvxs =-fct05uvss*vux
               fct02uvys =-fct05uvss*vuy
               fct02uvzs =-fct05uvss*vuz
               fct03uvxs =-(-fct04uvss+fct06uvss)*vux
               fct03uvys =-(-fct04uvss+fct06uvss)*vuy
               fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
               fct01uvxx = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01uvxy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01uvxz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01uvyy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01uvyz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01uvzz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

               fct04uvxs =(fct03uvss *fct01xs(v)          &
                          +fct01xs(v)*fct01uvxx           &
                          +fct01ys(v)*fct01uvxy           &
                          +fct01zs(v)*fct01uvxz)          &
                          *fctis1(v) *fctis2(v)           &
                          -fct03uvxs *fctis2(v)*fctmos(v)
               fct04uvys =(fct03uvss *fct01ys(v)          &
                          +fct01xs(v)*fct01uvxy           &
                          +fct01ys(v)*fct01uvyy           &
                          +fct01zs(v)*fct01uvyz)          &
                          *fctis1(v) *fctis2(v)           &
                          -fct03uvys *fctis2(v)*fctmos(v)
               fct04uvzs =(fct03uvss *fct01zs(v)          &
                          +fct01xs(v)*fct01uvxz           &
                          +fct01ys(v)*fct01uvyz           &
                          +fct01zs(v)*fct01uvzz)          &
                          *fctis1(v) *fctis2(v)           &
                          -fct03uvzs *fctis2(v)*fctmos(v)

               fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
               fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
               fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
               fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
               fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
               fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)
               fctvsuvx  = fct02uvxs*fctvdv(v)+fct04uvxs*fctvds(v)
               fctvsuvy  = fct02uvys*fctvdv(v)+fct04uvys*fctvds(v)
               fctvsuvz  = fct02uvzs*fctvdv(v)+fct04uvzs*fctvds(v)
               fctwsuvx  = fct02uvxs*fctwdv(v)+fct04uvxs*fctwds(v)
               fctwsuvy  = fct02uvys*fctwdv(v)+fct04uvys*fctwds(v)
               fctwsuvz  = fct02uvzs*fctwdv(v)+fct04uvzs*fctwds(v)

               fctsix(u) = fctsix(u)          &
                          +fcthhh(v)*fctqsuvx &
                          +fctvsg(v)*fctusuvx &
                          +fctusg(v)*fctvsuvx &
                          +          fctwsuvx
               fctsiy(u) = fctsiy(u)          &
                          +fcthhh(v)*fctqsuvy &
                          +fctvsg(v)*fctusuvy &
                          +fctusg(v)*fctvsuvy &
                          +          fctwsuvy
               fctsiz(u) = fctsiz(u)          &
                          +fcthhh(v)*fctqsuvz &
                          +fctvsg(v)*fctusuvz &
                          +fctusg(v)*fctvsuvz &
                          +          fctwsuvz
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            fctdvu=vux*vux+vuy*vuy+vuz*vuz
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(u)))  .or. &
                (fctdvu < fct1cn(fctidx(v)))) then
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(u))) then
               fct03vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))
               fct06vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(v))              &
                          *(one-fctdvu*fct1ci(fctidx(u))) &
                          *fct1fn(fctidx(u))              &
                          *fctivu
               fct02vuxs = fct05vuss*vux
               fct02vuys = fct05vuss*vuy
               fct02vuzs = fct05vuss*vuz
               fct03vuxs = (-fct04vuss+fct06vuss)*vux
               fct03vuys = (-fct04vuss+fct06vuss)*vuy
               fct03vuzs = (-fct04vuss+fct06vuss)*vuz
               fct01vuxx = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vux
               fct01vuxy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuy
               fct01vuxz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vux*vuz
               fct01vuyy = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuy
               fct01vuyz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuy*vuz
               fct01vuzz = half*fct07vuss                         &
                          *(one+fct1cn(fctidx(u))*fctivu)*vuz*vuz

               fct04vuxs =(fct03vuss *fct01xs(u)          &
                          +fct01xs(u)*fct01vuxx           &
                          +fct01ys(u)*fct01vuxy           &
                          +fct01zs(u)*fct01vuxz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuxs *fctis2(u)*fctmos(u)
               fct04vuys =(fct03vuss *fct01ys(u)          &
                          +fct01xs(u)*fct01vuxy           &
                          +fct01ys(u)*fct01vuyy           &
                          +fct01zs(u)*fct01vuyz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuys *fctis2(u)*fctmos(u)
               fct04vuzs =(fct03vuss *fct01zs(u)          &
                          +fct01xs(u)*fct01vuxz           &
                          +fct01ys(u)*fct01vuyz           &
                          +fct01zs(u)*fct01vuzz)          &
                          *fctis1(u) *fctis2(u)           &
                          -fct03vuzs *fctis2(u)*fctmos(u)

               fctqsvux  = fct02vuxs*fctqdv(u)+fct04vuxs*fctqds(u)
               fctqsvuy  = fct02vuys*fctqdv(u)+fct04vuys*fctqds(u)
               fctqsvuz  = fct02vuzs*fctqdv(u)+fct04vuzs*fctqds(u)
               fctusvux  = fct02vuxs*fctudv(u)+fct04vuxs*fctuds(u)
               fctusvuy  = fct02vuys*fctudv(u)+fct04vuys*fctuds(u)
               fctusvuz  = fct02vuzs*fctudv(u)+fct04vuzs*fctuds(u)
               fctvsvux  = fct02vuxs*fctvdv(u)+fct04vuxs*fctvds(u)
               fctvsvuy  = fct02vuys*fctvdv(u)+fct04vuys*fctvds(u)
               fctvsvuz  = fct02vuzs*fctvdv(u)+fct04vuzs*fctvds(u)
               fctwsvux  = fct02vuxs*fctwdv(u)+fct04vuxs*fctwds(u)
               fctwsvuy  = fct02vuys*fctwdv(u)+fct04vuys*fctwds(u)
               fctwsvuz  = fct02vuzs*fctwdv(u)+fct04vuzs*fctwds(u)

               fctsix(v) = fctsix(v)          &
                          +fcthhh(u)*fctqsvux &
                          +fctvsg(u)*fctusvux &
                          +fctusg(u)*fctvsvux &
                          +          fctwsvux
               fctsiy(v) = fctsiy(v)          &
                          +fcthhh(u)*fctqsvuy &
                          +fctvsg(u)*fctusvuy &
                          +fctusg(u)*fctvsvuy &
                          +          fctwsvuy
               fctsiz(v) = fctsiz(v)          &
                          +fcthhh(u)*fctqsvuz &
                          +fctvsg(u)*fctusvuz &
                          +fctusg(u)*fctvsvuz &
                          +          fctwsvuz
            endif
            if (fctdvu < fct1cn(fctidx(v))) then
               fct03uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))
               fct06uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(u))              &
                          *(one-fctdvu*fct1ci(fctidx(v))) &
                          *fct1fn(fctidx(v))              &
                          *fctivu
               fct02uvxs =-fct05uvss*vux
               fct02uvys =-fct05uvss*vuy
               fct02uvzs =-fct05uvss*vuz
               fct03uvxs =-(-fct04uvss+fct06uvss)*vux
               fct03uvys =-(-fct04uvss+fct06uvss)*vuy
               fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
               fct01uvxx = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vux
               fct01uvxy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuy
               fct01uvxz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vux*vuz
               fct01uvyy = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuy
               fct01uvyz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuy*vuz
               fct01uvzz = half*fct07uvss                         &
                          *(one+fct1cn(fctidx(v))*fctivu)*vuz*vuz

               fct04uvxs =(fct03uvss *fct01xs(v)          &
                          +fct01xs(v)*fct01uvxx           &
                          +fct01ys(v)*fct01uvxy           &
                          +fct01zs(v)*fct01uvxz)          &
                          *fctis1(v) *fctis2(v)           &
                          -fct03uvxs *fctis2(v)*fctmos(v)
               fct04uvys =(fct03uvss *fct01ys(v)          &
                          +fct01xs(v)*fct01uvxy           &
                          +fct01ys(v)*fct01uvyy           &
                          +fct01zs(v)*fct01uvyz)          &
                          *fctis1(v) *fctis2(v)           &
                          -fct03uvys *fctis2(v)*fctmos(v)
               fct04uvzs =(fct03uvss *fct01zs(v)          &
                          +fct01xs(v)*fct01uvxz           &
                          +fct01ys(v)*fct01uvyz           &
                          +fct01zs(v)*fct01uvzz)          &
                          *fctis1(v) *fctis2(v)           &
                          -fct03uvzs *fctis2(v)*fctmos(v)

               fctqsuvx  = fct02uvxs*fctqdv(v)+fct04uvxs*fctqds(v)
               fctqsuvy  = fct02uvys*fctqdv(v)+fct04uvys*fctqds(v)
               fctqsuvz  = fct02uvzs*fctqdv(v)+fct04uvzs*fctqds(v)
               fctusuvx  = fct02uvxs*fctudv(v)+fct04uvxs*fctuds(v)
               fctusuvy  = fct02uvys*fctudv(v)+fct04uvys*fctuds(v)
               fctusuvz  = fct02uvzs*fctudv(v)+fct04uvzs*fctuds(v)
               fctvsuvx  = fct02uvxs*fctvdv(v)+fct04uvxs*fctvds(v)
               fctvsuvy  = fct02uvys*fctvdv(v)+fct04uvys*fctvds(v)
               fctvsuvz  = fct02uvzs*fctvdv(v)+fct04uvzs*fctvds(v)
               fctwsuvx  = fct02uvxs*fctwdv(v)+fct04uvxs*fctwds(v)
               fctwsuvy  = fct02uvys*fctwdv(v)+fct04uvys*fctwds(v)
               fctwsuvz  = fct02uvzs*fctwdv(v)+fct04uvzs*fctwds(v)

               fctsix(u) = fctsix(u)           &
                           +fcthhh(v)*fctqsuvx &
                           +fctvsg(v)*fctusuvx &
                           +fctusg(v)*fctvsuvx &
                           +          fctwsuvx
               fctsiy(u) = fctsiy(u)           &
                           +fcthhh(v)*fctqsuvy &
                           +fctvsg(v)*fctusuvy &
                           +fctusg(v)*fctvsuvy &
                           +          fctwsuvy
               fctsiz(u) = fctsiz(u)           &
                           +fcthhh(v)*fctqsuvz &
                           +fctvsg(v)*fctusuvz &
                           +fctusg(v)*fctvsuvz &
                           +          fctwsuvz
            endif
         enddo
      enddo

      fctpol=fctpl1+fctpl2
      fctnpl=fctnp1+fctnp2+fctnp3

   endif

   RETURN
   END Subroutine fctprt


! -------------------------------------------------------------
! END OF FACTS
!       end module FACTS
! ##ELSE
!       subroutine null_fctall
!       return
!       end subroutine null_fctall
! ##ENDIF


!===================================================================
!===================================================================
   SUBROUTINE FCTNBA(NNNB,JNB,MAXJNB,INBLO,X,Y,Z,        &
                     INB14,IBLO14,CUTNB,WRNMIN,CMPLTD,   &
                     FCT1AC,FCT2AC,                      &
                     FCT1LL,FCT1LB,                      &
                     FCT2LL,FCT2LB,                      &
                     RSCMX,RSCMY,RSCMZ,                  &
                     RSXMAX,RSYMAX,RSZMAX,RSDISP,        &
                     ATSX,ATSY,ATSZ)
!-----------------------------------------------------------------------
!     THIS ROUTINE CONSTRUCTS THE ATOM BASED NONBONDED LISTS
!
!     22-AUG-1981  By Bernard R. Brooks
!     Overhauled (group lists removed) - BRB - October 25, 1996
!
  use chm_kinds
  use dimens_fcm
  use number

  ! use exclm

  use psf
  use stream
  use timerm
#if KEY_PARALLEL==1
  use parallel      
#endif
#if KEY_REPLICA==1
  use replica_mod   
#endif

  use chutil,  only:initia,atomid
  use machutil,only:die,timre,timrb
      implicit none

   integer, intent(inout)            :: NNNB     , MAXJNB
   integer, intent(inout)            :: JNB(*)   , INBLO(*)
   real(kind=chm_real), intent(in)   :: X(*),Y(*),Z(*)
   integer, intent(inout)            :: INB14(*) , IBLO14(*)
   real(kind=chm_real), intent(in)   :: CUTNB    , WRNMIN
   logical                           :: CMPLTD
   integer, intent(inout)            :: FCT1AC   , FCT2AC
   integer, intent(inout)            :: FCT1LL(*), FCT1LB(*)
   integer, intent(inout)            :: FCT2LL(*), FCT2LB(*)
   real(kind=chm_real),intent(inout) :: RSCMX(*) , RSCMY(*) , RSCMZ(*)
   real(kind=chm_real),intent(inout) :: RSXMAX(*), RSYMAX(*), RSZMAX(*)
   integer, intent(inout)            :: RSDISP(*)
   real(kind=chm_real),intent(inout) :: ATSX(*)  , ATSY(*)  , ATSZ(*)

#if KEY_PARALLEL==1 /*pardecl*/
   integer               :: imynod
#endif /* (pardecl)*/
!
#if KEY_REPLICA==1 /*repdecl*/
!# <caves>-Aug-4-1993 (Leo Caves)
   integer              :: iRepNo, iRepID
#endif /* (repdecl)   REPLICA*/

   integer              :: itmp, jtmp
   logical              :: ltmp, qerr
   logical, external    :: qinlist
   !rcz
   integer              :: i,j,is,iq,nat,ngat,irs,nxi,nximax
   integer              :: jrs,js,jq,irst,jrst,inbx,ix14,ix14p,igrpmx
   real(kind=chm_real)  :: ctnbsq,wminsq
   real(kind=chm_real)  :: xmin,xmax,ymin,ymax,zmin,zmax,xd,yd,zd
   real(kind=chm_real)  :: xd1,yd1,zd1,r2,r,xi,yi,zi,dxt,dyt,dzt,grpmxs
   logical              :: movefg,qmove,doit
   character (len=8)    :: siddn,riddn,resdn,acdn,siddn2,riddn2,resdn2,acdn2

   real(kind=chm_real)  :: fctlcs, fcthcs
   integer              :: fctinc
   ! integer              :: a, b
!-----------------------------------------------------------------------
   IF (TIMER > 0) CALL TIMRB
#if KEY_REPLICA==1 /*repsetup*/
   IF (qRep) nRepXA = 0
#endif /* (repsetup)  REPLICA*/
   CMPLTD=.FALSE.
   CTNBSQ=CUTNB*CUTNB
   FCTLCS=MIN(MIN(FCT1LN,FCT2LN),CTNBSQ)
   FCTHCS=MAX(MAX(FCT1LN,FCT2LN),CTNBSQ)
   WMINSQ=WRNMIN*WRNMIN
   QMOVE=.FALSE.
   !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
   GRPMXS=ZERO
   IGRPMX=1

   ! Store the current atom configuration.
   ATSX(1:natom)=X(1:natom)
   ATSY(1:natom)=Y(1:natom)
   ATSZ(1:natom)=Z(1:natom)
   ! Find geometric center for each group
   DO I=1,NGRP
      IS=IGPBS(I)+1
      IQ=IGPBS(I+1)
      NAT=IQ-IS+1
      IF(NAT <= 0) CALL DIE
      XMIN=X(IS)
      XMAX=XMIN
      YMIN=Y(IS)
      YMAX=YMIN
      ZMIN=Z(IS)
      ZMAX=ZMIN

      IF(IMOVE(IS) > 0) QMOVE=.TRUE.
      DO J=IS+1,IQ
         XMIN=MIN(X(J),XMIN)
         YMIN=MIN(Y(J),YMIN)
         ZMIN=MIN(Z(J),ZMIN)
         XMAX=MAX(X(J),XMAX)
         YMAX=MAX(Y(J),YMAX)
         ZMAX=MAX(Z(J),ZMAX)

         IF(IMOVE(J) > 0) QMOVE=.TRUE.
      ENDDO
      XD=HALF*(XMIN+XMAX)
      YD=HALF*(YMIN+YMAX)
      ZD=HALF*(ZMIN+ZMAX)
      ! Size of rectangular box surrounding group.
      RSXMAX(I)=XMAX-XD
      RSYMAX(I)=YMAX-YD
      RSZMAX(I)=ZMAX-ZD
      !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
      IF(RSXMAX(I) > GRPMXS) THEN
         GRPMXS=RSXMAX(I)
         IGRPMX=IS
      ENDIF
      IF(RSYMAX(I) > GRPMXS) THEN
         GRPMXS=RSYMAX(I)
         IGRPMX=IS
      ENDIF
      IF(RSZMAX(I) > GRPMXS) THEN
         GRPMXS=RSZMAX(I)
         IGRPMX=IS
      ENDIF
      !brb..07-FEB-99
      ! center of group defined by the box
      RSCMX(I)=XD
      RSCMY(I)=YD
      RSCMZ(I)=ZD
   ENDDO

   !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
   GRPMXS=GRPMXS*TWO
   IF(GRPMXS > TWELVE) THEN
      IF(WRNLEV >= -1) write(outu,137) GRPMXS,IGRPMX
137   FORMAT(                                                            &
             ' NBONDA>>  Maximum group spatial extent (12A) exceeded.',/ &
             '   Size is',F12.2,' Angstroms and starts with atom:',I8,/  &
             '   Please check group boundary definitions.')
      !yw...to run testcases         CALL DIEWRN(-1)
   ENDIF
   !brb..07-FEB-99
   !
   ! Now decide how to treat each residue pair using a rectangular
   ! search and store the disposition in rsdisp.
   NNNB=0
   NGAT=0
   FCT1AC=0
   FCT2AC=0

! Expand control section
! Do IMOVE expansion of code
! Do REPLICA expansion of code
#if KEY_DEBUG != 1
#undef FACTS_DEBUG

    IF(QMOVE) THEN

#define FACTS_IMOVE 1
      IF(QREP) THEN
#define FACTS_REPLICA 1
#include "fctnba.inc"
#undef FACTS_REPLICA
      ELSE
#include "fctnba.inc"
      ENDIF

#undef FACTS_IMOVE

    ELSE

      IF(QREP) THEN
#define FACTS_REPLICA 1
#include "fctnba.inc"
#undef FACTS_REPLICA
      ELSE
#include "fctnba.inc"
      ENDIF
         
    ENDIF

#else /* KEY_DEBUG */
      
#define FACTS_DEBUG 1
#define FACTS_IMOVE 1
#undef FACTS_REPLICA
#include "fctnba.inc"
#undef FACTS_IMOVE
#undef FACTS_DEBUG
      
#endif /* KEY_DEBUG */

   CMPLTD=.TRUE.
!
245 FORMAT(' WARNING: ATOMS',4(1X,A),               &
           ' AND',4(1X,A),' ONLY',F5.2,' A. APART')

! Termination of the routine.
   IF(PRNLEV >= 5) THEN
      write(outu,720)
720   FORMAT(/' General atom nonbond list generation found:')
      ! Atom lists
      write(outu,733) NNNB
733   FORMAT(I9,' ATOM PAIRS WERE FOUND FOR ATOM LIST')

      write(outu,736) NGAT
736   FORMAT(I9,' GROUP PAIRS REQUIRED ATOM SEARCHES'/)
      ! FACTS interaction lists
      write(outu,737) FCT1AC
737   FORMAT(I9,' ATOM PAIRS WERE FOUND FOR FACTS ATOM LIST 1')
      write(outu,738) FCT2AC
738   FORMAT(I9,' ATOM PAIRS WERE FOUND FOR FACTS ATOM LIST 2')
   ENDIF
!
!
#if KEY_REPLICA==1 /*repprint*/
   !# <caves>-Aug-4-1993 (Leo Caves)
   IF (PRNLEV >= 5.AND.qRep) THEN
      write(outu,'(I9,A/)') nRepXA,' REPLICA ATOM  PAIRS EXCLUDED'
   ENDIF ! PRNLEV
#endif /* (repprint)  REPLICA*/
   IF (TIMER==1) THEN
      IF(PRNLEV >= 2) write(outu,830) 'TOTAL TIME IN NBONDA'
830     FORMAT(1X,A)
      CALL TIMRE
      CALL TIMRB
   ENDIF
!

! Print the nonbond exclusion pair list.

!      DO I=1,NATOM
!         IF (I == 1) THEN
!            A=1
!         ELSE
!            A=IBLO14(I-1)+1
!         ENDIF
!         B=IBLO14(I)
!         DO J=A,B
!            PRINT *,'I INB14(J) =',I,INB14(J)
!         ENDDO
!      ENDDO
!      PRINT *,'IBLO14(NATOM) =',IBLO14(NATOM)

! Print the nonbond pair list.

!      DO I=1,NATOM
!         IF (I == 1) THEN
!            A=1
!         ELSE
!            A=INBLO(I-1)+1
!         ENDIF
!         B=INBLO(I)
!         DO J=A,B
!            PRINT *,'I JNB(J) =',I,JNB(J)
!         ENDDO
!      ENDDO
!      PRINT *,'INBLO(NATOM) =',INBLO(NATOM)

! Print the FACTS pair list 1 (all pairs, all entries positive).

!      DO I=1,NATOM
!         IF (I == 1) THEN
!            A=1
!         ELSE
!            A=FCT1LL(I-1)+1
!         ENDIF
!         B=FCT1LL(I)
!         DO J=A,B
!            PRINT *,'I FCT1LB(J) =',I,FCT1LB(J)
!         ENDDO
!      ENDDO
!      PRINT *,'FCT1LL(NATOM) =',FCT1LL(NATOM)

! Print the FACTS pair list 2 (both atoms of each pair charged, all
! entries positive).

!      DO I=1,NATOM
!         IF (I == 1) THEN
!            A=1
!         ELSE
!            A=FCT2LL(I-1)+1
!         ENDIF
!         B=FCT2LL(I)
!         DO J=A,B
!            PRINT *,'I FCT2LB(J) =',I,FCT2LB(J)
!         ENDDO
!      ENDDO
!      PRINT *,'FCT2LL(NATOM) =',FCT2LL(NATOM)

   RETURN
   END SUBROUTINE FCTNBA
! -------------------------------------------------------------------

   SUBROUTINE FCTNMA(X,Y,Z,MXJNB,IMINB,IMIBLO,                      &
                     NIMNB, IMJNB, IMBLO, NIMNBS,IMJNBS,IMBLOS,     &
                     NTRANS,NIMGRP,IMATTR,IMATPT,LIMINV,IMINV,      &
                     CUTNB,WRNMIN,CMPLTD,                           &
                     FCT3AC,FCT4AC,                                 &
                     FCT3LL,FCT3LB,                                 &
                     FCT4LL,FCT4LB,                                 &
                     RSCMX,RSCMY,RSCMZ,RSXMAX,RSYMAX,RSZMAX,RSDISP)
!
! THIS ROUTINE CONSTRUCTS THE IMAGE ATOM BASED NONBONDED LISTS
!
!     By Bernard R. Brooks  22-AUG-1981
!     Overhauled (group lists removed) - BRB - October 25, 1996
!
  use chm_kinds
  use dimens_fcm
  use number

  ! use exclm

  use psf
  use stream
  use timerm
#if KEY_PARALLEL==1
  use parallel                      
#endif
#if KEY_REPLICA==1
  use replica_mod                   
#endif

  use chutil,only:initia,atomid
  use machutil,only:timrb,timre,die
      implicit none

   integer, intent(inout)            :: NIMNB,MXJNB,NIMNBS,NTRANS,NIMGRP
   integer, intent(inout)            :: FCT3AC,FCT4AC
   real(kind=chm_real), intent(in)   :: CUTNB,WRNMIN
   logical, external                 :: qinlist
   !
   real(kind=chm_real), intent(in)   :: X(*),Y(*),Z(*)
   integer, intent(inout)            :: IMBLO(*),IMBLOS(*)
   integer, intent(inout)            :: IMJNB(*),IMJNBS(*)
   integer, intent(inout)            :: IMINB(*)
   integer, intent(inout)            :: IMIBLO(*)
   integer, intent(inout)            :: IMATTR(*),IMATPT(*)
   integer, intent(inout)            :: RSDISP(*)
   integer, intent(inout)            :: IMINV(*)
   integer, intent(inout)            :: FCT3LL(*),FCT3LB(*)
   integer, intent(inout)            :: FCT4LL(*),FCT4LB(*)
   !
   real(kind=chm_real),intent(inout) :: RSCMX(*),RSCMY(*),RSCMZ(*)
   real(kind=chm_real),intent(inout) :: RSXMAX(*),RSYMAX(*),RSZMAX(*)

#if KEY_PARALLEL==1
   integer               :: imynod
#endif 
#if KEY_REPLICA==1 /*repdecl*/
   integer              :: iRepNo, iRepID
#endif /* (repdecl)  REPLICA*/
!
   integer              :: ITMP,JTMP
   integer              :: NAT,NGAT,NGPX,I,J,IS,IQ
   integer              :: IRS,JRS,KRS,KRSX,ITRANS
   integer              :: NXI,NXIMAX,JS,JQ,INBX,IX14,IX14P,NGRPX
   integer              :: FCTINC
   ! integer              :: A,B  ! (used ???)

   logical              :: LTMP   ! , QINLIST
   logical              :: CMPLTD,LIMINV,MOVEFG,QMOVE,LSELF,LIMALX,DOIT

   real(kind=chm_real)  :: CTNBSQ,WMINSQ,R2
   real(kind=chm_real)  :: XD,YD,ZD,XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX,XD1,YD1,ZD1,XI,YI,ZI
   real(kind=chm_real)  :: FCTLCS,FCTHCS
   character (len=8)    :: SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
! ----------------------------------------------------------------------------------
   IF (TIMER > 0) CALL TIMRB
#if KEY_REPLICA==1 /*repsetup*/
   IF (qRep) nRepXA = 0
#endif /* (repsetup)  REPLICA*/
   CMPLTD=.FALSE.
   CTNBSQ=CUTNB*CUTNB
   FCTLCS=MIN(MIN(FCT1LN,FCT2LN),CTNBSQ)
   FCTHCS=MAX(MAX(FCT1LN,FCT2LN),CTNBSQ)
   WMINSQ=WRNMIN*WRNMIN
   QMOVE=.FALSE.
   ! Fill in null values for the "real" atom interactions.
   IMBLO(1:natom)  = 0
   IMBLOS(1:natom) = 0

   ! Find geometric center for each group
   ! The arrays for the first NGRP groups are assumed to be already
   ! correctly filled from the call to NBONDG.
   ! write(*,*) 'NGRP',NGRP,'NIMGRP',NIMGRP

   DO I=NGRP+1,NIMGRP
      IS=IGPBS(I)+1
      IQ=IGPBS(I+1)
      NAT=IQ-IS+1
      IF(NAT <= 0) CALL DIE
      XMIN=X(IS)
      XMAX=XMIN
      YMIN=Y(IS)
      YMAX=YMIN
      ZMIN=Z(IS)
      ZMAX=ZMIN
      DO J=IS,IQ
         XMIN=MIN(X(J),XMIN)
         YMIN=MIN(Y(J),YMIN)
         ZMIN=MIN(Z(J),ZMIN)
         XMAX=MAX(X(J),XMAX)
         YMAX=MAX(Y(J),YMAX)
         ZMAX=MAX(Z(J),ZMAX)
   
         IF(IMOVE(J) > 0) QMOVE=.TRUE.
      ENDDO
      XD=HALF*(XMIN+XMAX)
      YD=HALF*(YMIN+YMAX)
      ZD=HALF*(ZMIN+ZMAX)
      ! Size of rectangular box surrounding group
      RSXMAX(I)=XMAX-XD
      RSYMAX(I)=YMAX-YD
      RSZMAX(I)=ZMAX-ZD
      ! Center of group defined by the box
      RSCMX(I)=XD
      RSCMY(I)=YD
      RSCMZ(I)=ZD
      IF(IMOVEG(I) > 0) QMOVE=.TRUE.
   ENDDO
! Now decide how to treat each residue pair using a rectangular
! search and store the disposition in rsdisp.
   NIMNB  = 0
   NIMNBS = 0
   FCT3AC = 0
   FCT4AC = 0

   NGAT=0
   NGPX=0
   IRS=NGRP

   do i=1,natom
      if(imove(i)>0) qmove=.true.
   enddo

! Expand control section
! Do IMOVE expansion of code
! Do REPLICA expansion of code
#if KEY_DEBUG != 1
#undef FACTS_DEBUG

    IF(QMOVE) THEN

#define FACTS_IMOVE 1
      IF(QREP) THEN
#define FACTS_REPLICA 1
#include "fctnma.inc"
#undef FACTS_REPLICA
      ELSE
#include "fctnma.inc"
      ENDIF

#undef FACTS_IMOVE

    ELSE

      IF(QREP) THEN
#define FACTS_REPLICA 1
#include "fctnma.inc"
#undef FACTS_REPLICA
      ELSE
#include "fctnma.inc"
      ENDIF
         
    ENDIF

#else /* KEY_DEBUG */
      
#define FACTS_DEBUG 1
#define FACTS_IMOVE 1
#undef FACTS_REPLICA
#include "fctnma.inc"
#undef FACTS_IMOVE
#undef FACTS_DEBUG
      
#endif /* KEY_DEBUG */

245 FORMAT(' WARNING: ATOMS',4(1X,A), &
           ' AND',4(1X,A),' ONLY',F5.2,' A. APART')
!
   CMPLTD=.TRUE.
!
   IF(PRNLEV >= 5) THEN
      write(outu,720)
720   FORMAT(/' Image nonbond list generation found:')
      ! Atom lists
      write(outu,731) NIMNB
731   FORMAT(I9,' ATOM PAIRS WERE FOUND FOR ATOM LIST')
      write(outu,732) NIMNBS
732   FORMAT(I9,' ATOM PAIRS WERE FOUND FOR ATOM SELF LIST')
      write(outu,739) NGAT
739   FORMAT(I9,' GROUP PAIRS REQUIRED ATOM SEARCHES'/)
      ! FACTS IMAGE LISTS
      write(outu,740) FCT3AC
740   FORMAT(I9,' ATOM PAIRS WERE FOUND FOR FACTS ATOM LIST 3')
      write(outu,741) FCT4AC
741   FORMAT(I9,' ATOM PAIRS WERE FOUND FOR FACTS ATOM LIST 4')
   ENDIF
#if KEY_REPLICA==1 /*repprint*/
!# <caves>-Aug-4-1993 (Leo Caves)
   IF (PRNLEV >= 5.AND.qRep) THEN
      write(outu,'(I9,A/)') nRepXA,' REPLICA ATOM  PAIRS EXCLUDED'
   ENDIF ! PRNLEV
#endif /* (repprint)  REPLICA*/
   IF (TIMER==1) THEN
      IF(PRNLEV >= 2) write(outu,130) 'TOTAL TIME IN NBONDMA'
130   FORMAT(1X,A)
      CALL TIMRE
      CALL TIMRB
   ENDIF
!
   IF(PRNLEV > 8) THEN
      write(outu,888) 'NIMNB ',NIMNB
      write(outu,888) 'IMBLO ',(IMBLO(I),I=1,NATOMT)
      write(outu,888) 'IMJNB ',(IMJNB(I),I=1,NIMNB)
      write(outu,888) 'NIMNBS',NIMNBS
      write(outu,888) 'IMBLOS',(IMBLOS(I),I=1,NATOMT)
      write(outu,888) 'IMJNBS',(IMJNBS(I),I=1,NIMNBS)

888   FORMAT(2X,A6,': '/,(20I5))
   ENDIF
! ------------------------------------------------------------------
! Print the FACTS pair list 3 (all pairs, all entries positive).

!      DO I=1,NATOMT
!         IF (I == 1) THEN
!            A=1
!         ELSE
!            A=FCT3LL(I-1)+1
!         ENDIF
!         B=FCT3LL(I)
!         DO J=A,B
!            PRINT *,'I FCT3LB(J) =',I,FCT3LB(J)
!         ENDDO
!      ENDDO
!      PRINT *,'FCT3LL(NATOMT) =',FCT3LL(NATOMT)

! Print the FACTS pair list 4 (both atoms of each pair charged, all
! entries positive).

!      DO I=1,NATOMT
!         IF (I == 1) THEN
!            A=1
!         ELSE
!            A=FCT4LL(I-1)+1
!         ENDIF
!         B=FCT4LL(I)
!         DO J=A,B
!            PRINT *,'I FCT4LB(J) =',I,FCT4LB(J)
!         ENDDO
!      ENDDO
!      PRINT *,'FCT4LL(NATOMT) =',FCT4LL(NATOMT)
! ------------------------------------------------------------------
   RETURN
   END SUBROUTINE FCTNMA
! ------------------------------------------------------------------

   SUBROUTINE FCTGROW_ints(filez, subroutinz, arrayz, member, newsize)

   ! -------------------------------------------------
   use chm_kinds
   use chm_types
   use stream

   use memory
       implicit none
   !--------------------------------------------------
   character(len=*), intent(in)                  :: filez, subroutinz, arrayz
   integer,dimension(:), allocatable, intent(in) :: member
   integer,intent(in)                            :: newsize

   integer                                       :: currsize
   integer                                       :: ierr,ierr2
   !
   currsize = size(member)

   if (allocated(member)) then
      ! write(outu,*) 'FCTGROW_ints:', member,' is allocated '
      if (newsize > size(member)) then
         write(outu,*)'FCTGROW_ints: (newsize > size(member))', newsize, size(member)

         call chmdealloc(filez,subroutinz,arrayz, currsize, intg=member, ierr=ierr2, qdie=.true.)
         call chmalloc  (filez,subroutinz,arrayz, newsize , intg=member, ierr=ierr2, qdie=.true.)
      endif
   else
      write(outu,*)'FCTGROW_ints:',  member,' is NOT allocated '
      write(outu,*)'FCTGROW_ints: newsize > size(member)', newsize, size(member)
      call chmalloc(filez,subroutinz,arrayz, newsize, intg=member, ierr=ierr2, qdie=.true.)
   endif
   RETURN
   END SUBROUTINE FCTGROW_ints

! ##ENDIF  (factz2)
#endif /*  (factz1)*/
end module facts_module
