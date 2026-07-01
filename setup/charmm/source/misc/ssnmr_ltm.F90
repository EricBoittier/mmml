module ssnmr
   use chm_kinds
   use dimens_fcm
   use chm_types

   implicit none

#if KEY_NOMISC==0 && KEY_SSNMR==1
   logical :: qssnmr, cs_initialized = .false.

   integer :: csmax = 1000, nmrmx = 20

   integer csnum, csnm2, numcat

   ! depends on nmrmx
   real(chm_real), allocatable, dimension(:, :) :: sigma ! 3 x nmrmx
   real(chm_real), allocatable, dimension(:) :: phics, nudc
   integer, allocatable, dimension(:, :) ::  stoag ! 2 x nmrmx
   logical, allocatable, dimension(:) :: lconsnu, ldcabs

   ! size csmax allocatable arrays
   real(chm_real), allocatable, dimension(:) :: forcs, expcs
   logical, allocatable, dimension(:) :: ldipc
   integer, allocatable, dimension(:) :: csipt, csinm, cslis

   ! for analysis
   logical lanal
   real rmsdval,rmsdvdc,iunij

   ! for soft asymptote, size csmax allocatable arrays
   logical, allocatable, dimension(:) :: lsofta
   real(chm_real), allocatable, dimension(:) :: &
        kmincs, plcs, kmaxcs, pucs, &
        rsw, expt, knew, rswl

contains
!***********************************************************************
! Sets up 15N Chemical Shift constraint force field.
! & N-H dipolar coupling constraint
!***********************************************************************
! Authors: Jinhyuk Lee and Wonpil Im (2006)
!          jinhyuk@ku.edu  wonpil@ku.edu
! @ Center for Bioinformatics in Univ. of Kansas
!
! Reference: Jinhyuk Lee, Jianhan Chen, Charles L Brooks III, and
!            Wonpil Im, J. Magn. Reson., 193, 68-76 (2008)
!
! Bug, Error, Problems, Suggestion etc reports welcome
!
! [SYNTAX]
!
! Define SSNMR restraint potentials
!
! ccs
! exps s11 (real) s22 (real) s33 (real) phi (real-degs) nudc (real) DCCO DCABs
! assign atom-selection
!        [DIPC] forc (real) exp (real) [asymptotic]
! end
!
! [asymptotic] : use Asymtotic potentials in behalf of harmonic thing
! alot of options to define
!
! It dosen't support "absolute observables" routine
! but it supports "changable nu_0" routine
!
! Simple analasis routine
! if RMSD (= sqrt (1/N sum_i^N (exp-calc)^2) <= val (val>=0), store a value ($SRMV=1) to use later
!
! ccs
! print anal diff val
! end
!
! Reset all ssnmr restraints
!
! ccs
! reset
! end
!
! OPTIONS
!
! DCABS - DC use absolute observables, back-calculation makes negative to positives
!       - (default) : available both signs
! DCCO  - nu_0 uses the constant value defined in nudc
!       - (default) : chagable nu_0
! DIPC  - DC assignment
!
! s11 s22 s33 phi - CS shift observables
!
!
  subroutine ccs_iniall()
    use memory, only: chmalloc

    implicit none

    if (allocated(forcs)) call ccs_uniniall()

    csmax = 1000
    csnum = 0
    csnm2 = 0

    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'forcs', csmax, crl=forcs)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'expcs', csmax, crl=expcs)

    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'ldipc', csmax, log=ldipc)

    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'csipt', csmax, intg=csipt)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'csinm', csmax, intg=csinm)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'cslis', csmax, intg=cslis)

    ! for soft asymptote
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'kmincs', csmax, crl=kmincs)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'plcs', csmax, crl=plcs)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'kmaxcs', csmax, crl=kmaxcs)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'pucs', csmax, crl=pucs)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'rsw', csmax, crl=rsw)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'expt', csmax, crl=expt)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'knew', csmax, crl=knew)
    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'rswl', csmax, crl=rswl)

    call chmalloc('ssnmr_ltm.src', 'ccs_iniall', 'lsofta', csmax, log=lsofta)

    call nmr_init()

    cs_initialized = .true.
  end subroutine ccs_iniall

  subroutine ccs_uniniall()
    use memory, only: chmdealloc

    implicit none

    cs_initialized = .false.

    numcat = 0
    csnum = 0
    csnm2 = 0
    lanal = .false.

    if (.not. allocated(cslis)) return

    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'forcs', csmax, crl=forcs)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'expcs', csmax, crl=expcs)

    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'ldipc', csmax, log=ldipc)

    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'csipt', csmax, intg=csipt)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'csinm', csmax, intg=csinm)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'cslis', csmax, intg=cslis)

    ! for soft asymptote
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'kmincs', csmax, crl=kmincs)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'plcs', csmax, crl=plcs)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'kmaxcs', csmax, crl=kmaxcs)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'pucs', csmax, crl=pucs)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'rsw', csmax, crl=rsw)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'expt', csmax, crl=expt)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'knew', csmax, crl=knew)
    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'rswl', csmax, crl=rswl)

    call chmdealloc('ssnmr_ltm.src', 'ccs_uniniall', 'lsofta', csmax, log=lsofta)

    csmax = 1000

    call nmr_uninit()
  end subroutine ccs_uniniall

  subroutine ccs_add_storage()
    use memory, only: chmrealloc

    implicit none

    csmax = 2 * csmax

    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'forcs', csmax, crl=forcs)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'expcs', csmax, crl=expcs)

    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'ldipc', csmax, log=ldipc)

    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'csipt', csmax, intg=csipt)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'csinm', csmax, intg=csinm)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'cslis', csmax, intg=cslis)

    ! for soft asymptote
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'kmincs', csmax, crl=kmincs)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'plcs', csmax, crl=plcs)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'kmaxcs', csmax, crl=kmaxcs)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'pucs', csmax, crl=pucs)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'rsw', csmax, crl=rsw)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'expt', csmax, crl=expt)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'knew', csmax, crl=knew)
    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'rswl', csmax, crl=rswl)

    call chmrealloc('ssnmr_ltm.src', 'ccs_add_storage', 'lsofta', csmax, log=lsofta)
  end subroutine ccs_add_storage

  subroutine nmr_init()
    use memory, only: chmalloc

    implicit none

    nmrmx = 20
    numcat = 0

    call chmalloc('ssnmr_ltm.src', 'nmr_init', 'sigma', 3, nmrmx, crl=sigma)
    call chmalloc('ssnmr_ltm.src', 'nmr_init', 'phics', nmrmx, crl=phics)
    call chmalloc('ssnmr_ltm.src', 'nmr_init', 'nudc', nmrmx, crl=nudc)

    call chmalloc('ssnmr_ltm.src', 'nmr_init', 'stoag', 2, nmrmx, intg=stoag)

    call chmalloc('ssnmr_ltm.src', 'nmr_init', 'lconsnu', nmrmx, log=lconsnu)
    call chmalloc('ssnmr_ltm.src', 'nmr_init', 'ldcabs', nmrmx, log=ldcabs)
  end subroutine nmr_init

  subroutine nmr_uninit()
    use memory, only: chmdealloc

    implicit none

    nmrmx = 20
    numcat = 0

    call chmdealloc('ssnmr_ltm.src', 'nmr_uninit', 'sigma', &
         3, nmrmx, crl=sigma)
    call chmdealloc('ssnmr_ltm.src', 'nmr_uninit', 'phics', &
         nmrmx, crl=phics)
    call chmdealloc('ssnmr_ltm.src', 'nmr_uninit', 'nudc', &
         nmrmx, crl=nudc)

    call chmdealloc('ssnmr_ltm.src', 'nmr_uninit', 'stoag', &
         2, nmrmx, intg=stoag)

    call chmdealloc('ssnmr_ltm.src', 'nmr_uninit', 'lconsnu', &
         nmrmx, log=lconsnu)
    call chmdealloc('ssnmr_ltm.src', 'nmr_uninit', 'ldcabs', &
         nmrmx, log=ldcabs)
  end subroutine nmr_uninit

  subroutine nmr_add_storage()
    use memory, only: chmrealloc

    implicit none

    nmrmx = 2 * nmrmx

    call chmrealloc('ssnmr_ltm.src', 'nmr_add_storage', 'sigma', &
         3, nmrmx, crl=sigma)
    call chmrealloc('ssnmr_ltm.src', 'nmr_add_storage', 'phics', &
         nmrmx, crl=phics)
    call chmrealloc('ssnmr_ltm.src', 'nmr_add_storage', 'nudc', &
         nmrmx, crl=nudc)

   call chmrealloc('ssnmr_ltm.src', 'nmr_add_storage', 'stoag', &
        2, nmrmx, intg=stoag)

   call chmrealloc('ssnmr_ltm.src', 'nmr_add_storage', 'lconsnu', &
        nmrmx, log=lconsnu)
   call chmrealloc('ssnmr_ltm.src', 'nmr_add_storage', 'ldcabs', &
        nmrmx, log=ldcabs)
  end subroutine nmr_add_storage

  subroutine csset
      use dimens_fcm
      use psf
      use comand
      use string
      use memory
      implicit none

      call csset2

   return
   end subroutine csset

#else /* KEY_NOMISC, KEY_SSNMR */

 contains

   subroutine csset
      call WRNDIE(-1,'<CHARMM>','SSNMR code is not compiled.')
   return
   end subroutine csset
#endif /* KEY_NOMISC, KEY_SSNMR */

end module ssnmr
