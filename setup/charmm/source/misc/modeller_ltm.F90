module modeller
   use chm_kinds
   use dimens_fcm
   use chm_types

   implicit none

#if KEY_NOMISC==0 && KEY_MODELLER==1
   logical :: qmodel, md_initialized = .false.

   integer :: mdmax = 50000, mdmx2 = 150000, maxncmd = 200, maxmin = 1089
   integer :: mdmaxt = 20000, maxmint = 34

   INTEGER MDNUM,MDNM2,numcatmd,cdnum

   integer, allocatable, dimension(:, :) :: stoagmd
   integer, allocatable, dimension(:) :: MDIPT,MDINM,MDLIS, mdjpt,mdjnm,tnmin, tnpos, md2cd


   real(chm_real), allocatable, dimension(:, :) :: mdmin, mdpos,smdpos, mdsig
   real(chm_real), allocatable, dimension(:) :: formd, ulsig, lder,hder,lval,hval,inte,rsm, knem,expm,pucm

   real(chm_real), allocatable, dimension(:) :: iunijmd, sxmin,sxmax,symin, symax,sxint,syint
   real(chm_real), allocatable, dimension(:, :) :: ytwo, ytwo2

   real(chm_real), allocatable, dimension(:,:,:,:,:) :: ccl
!   real(chm_real) ccl(mdmaxt,maxmint,maxmint,4,4)
!   real(chm_real), allocatable, dimension(mdmaxt,maxmint,maxmint,4,4) :: ccl

   logical lanamd
   logical, allocatable, dimension(:) :: lperiod, ldspline,laspline,lrotspline, lmhar, lgmha, loqua, lsgar,lsper,lsgaa,ldspa, lfbsp, lcubi

contains
! Modeller energy functions
!C 1. Multiple Gaussian Function NOE like distance restraint potential
!C
!C Authors: Jinhyuk Lee @ KOBIC (Korean Bioinformatimd Center)
!C                        KRIBB (Korea Research Institute for Bioscience and
!Biotechnology)
!C          jinhyuk@kribb.re.kr / mack97hyuk@gmail.com
!C
!c 02/26/2009
!c
!c mod
!c mgau
!c assign double atom-selections -
!c        forc (real) nmin (integer) min (nmin) * (real) sigma (nmin) * (real)
!c end
!c
!c mod
!c mper
!c assign sele (four-atoms selections) end -
!c        forc (real) nmin (integer) min (nmin) * (real) -
!c                                  weigh (nmin) * (real)
!c end
!c
!c Reset all modeller energy restraints
!c mod
!c reset
!c end
!c
!c OPTIONS
!c forc - force constant
!c nmin - number of minimums
!c min - minimum distances
!c sigma - deviations
!c

   subroutine mod_iniall()
     use memory, only: chmalloc

     implicit none

!   local
     integer iiii

     if (allocated(formd)) call mod_uniniall()

     mdmax = 50000
     mdnum = 0
     mdnm2 = 0
     mdmaxt = 20000
     maxmint = 34
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'mdipt', mdmax, intg=mdipt)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'mdinm', mdmax, intg=mdinm)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'mdlis', mdmx2, intg=mdlis)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'mdjpt', mdmax, intg=mdjpt)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'mdjnm', mdmax, intg=mdjnm)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'tnmin', mdmax, intg=tnmin)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'tnpos', mdmax, intg=tnpos)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'md2cd', mdmax, intg=md2cd)
    
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'stoagmd',200,2,intg=stoagmd)
    ! for soft asymptote
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'formd', mdmax, crl=formd)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'mdmin', mdmax,maxmin, crl=mdmin)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'mdpos', mdmax,maxmin, crl=mdpos)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'smdpos', mdmax,maxmin, crl=smdpos)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'mdsig', mdmax,maxmin,crl=mdsig)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'ulsig', mdmax,crl=ulsig)

    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lder', mdmax, crl=lder)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'hder', mdmax, crl=hder)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lval', mdmax, crl=lval)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'hval', mdmax, crl=hval)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'inte', mdmax, crl=inte)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'rsm', mdmax, crl=rsm)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'knem', mdmax, crl=knem)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'expm', mdmax, crl=expm)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'pucm', mdmax, crl=pucm)

    call chmalloc('modeller_ltm.src', 'mod_iniall', 'sxmin', mdmax, crl=sxmin)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'sxmax', mdmax, crl=sxmax)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'symin', mdmax, crl=symin)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'symax', mdmax, crl=symax)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'sxint', mdmax, crl=sxint)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'syint', mdmax, crl=syint)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'ytwo', mdmax,maxmin, crl=ytwo)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'ytwo2', mdmax,maxmin, crl=ytwo2)
!    call chmalloc('modeller_ltm.src', 'mod_iniall', 'ccl', mdmaxt,maxmint,4,4,crl=ytwo2)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'ccl',20000,34,34,4,4,crl=ccl)


    call chmalloc('modeller_ltm.src', 'mod_iniall', 'ldspline', maxncmd, log=ldspline)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lperiod', maxncmd, log=lperiod)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'laspline', maxncmd, log=laspline)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lrotspline', maxncmd, log=lrotspline)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lmhar', mdmax, log=lmhar)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lgmha', mdmax, log=lgmha)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'loqua', mdmax, log=loqua)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lsgar', mdmax, log=lsgar)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lsper', mdmax, log=lsper)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lsgaa', mdmax, log=lsgaa)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'ldspa', mdmax, log=ldspa)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lfbsp', mdmax, log=lfbsp)
    call chmalloc('modeller_ltm.src', 'mod_iniall', 'lcubi', mdmax, log=lcubi)

    do iiii = 1,maxncmd
        lperiod(iiii)=.false.
        ldspline(iiii)=.false.
        laspline(iiii)=.false.
        lrotspline(iiii)=.false.
    enddo

    md_initialized = .true.
   end subroutine mod_iniall


   subroutine mod_uniniall()
     use memory, only: chmdealloc

     implicit none

     md_initialized = .false.

     numcatmd = 0
     mdnum = 0
     mdnm2 = 0
     lanamd= .false.
     mdmax = 50000
     maxmin = 1089
     maxmint = 34
     mdmaxt = 20000

    if (.not. allocated(mdlis)) return

    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'mdipt', mdmax, intg=mdipt)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'mdinm', mdmax, intg=mdinm)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'mdlis', mdmx2, intg=mdlis)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'mdjpt', mdmax, intg=mdjpt)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'mdjnm', mdmax, intg=mdjnm)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'tnmin', mdmax, intg=tnmin)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'tnpos', mdmax, intg=tnpos)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'md2cd', mdmax, intg=md2cd)

    call chmdealloc('modeller_ltm.src', 'mod_iniall','stoagmd',200,2,intg=stoagmd)

    ! for soft asymptote
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'formd', mdmax, crl=formd)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'mdmin', mdmax, maxmin, crl=mdmin)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'mdpos', mdmax, maxmin, crl=mdpos)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'smdpos', mdmax, maxmin, crl=smdpos)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'mdsig',mdmax, maxmin, crl=mdsig)

    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'ulsig', mdmax, crl=ulsig)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lder', mdmax, crl=lder)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'hder', mdmax, crl=hder)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lval', mdmax, crl=lval)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'hval', mdmax, crl=hval)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'inte', mdmax, crl=inte)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'rsm', mdmax, crl=rsm)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'knem', mdmax, crl=knem)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'expm', mdmax, crl=expm)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'pucm', mdmax, crl=pucm)

    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'sxmin', mdmax, crl=sxmin)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'sxmax', mdmax, crl=sxmax)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'symin', mdmax, crl=symin)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'symax', mdmax, crl=symax)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'sxint', mdmax, crl=sxint)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'syint', mdmax, crl=syint)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'ytwo', mdmax,maxmin,crl=ytwo)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'ytwo2', mdmax,maxmin,crl=ytwo2)
!   call chmdealloc('modeller_ltm.src', 'mod_iniall', 'ccl', mdmaxt,maxmint,4,4,crl=ccl)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall','ccl',20000,34,34,4,4,crl=ccl)


    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'ldspline', maxncmd,log=ldspline)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lperiod', maxncmd,log=lperiod)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'laspline', maxncmd,log=laspline)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lrotspline', maxncmd,log=lrotspline)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lmhar', mdmax, log=lmhar)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lgmha', mdmax, log=lgmha)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'loqua', mdmax, log=loqua)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lsgar', mdmax, log=lsgar)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lsper', mdmax, log=lsper)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lsgaa', mdmax, log=lsgaa)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'ldspa', mdmax, log=ldspa)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lfbsp', mdmax, log=lfbsp)
    call chmdealloc('modeller_ltm.src', 'mod_uniniall', 'lcubi', mdmax, log=lcubi)


    end subroutine mod_uniniall

 subroutine mdset
     use dimens_fcm
     use psf
     use comand
     use string
     use memory
     implicit none

     call mdset2

    return
 end subroutine mdset

#else /* KEY_NOMISC, KEY_MODELLER */

 contains

   subroutine mdset
      call WRNDIE(-1,'<CHARMM>','MODELLER code is not compiled.')
   return
   end subroutine mdset
#endif /* KEY_NOMISC, KEY_MODELLER */

end module modeller

