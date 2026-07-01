module dimens_fcm
  implicit none
    !-----------------------------------------------------------------------
    !
    !
    ! This common file contains all useful dimensioning information.
    !                                   BRB - 01/12/89
    !
    !-----------------------------------------------------------------------
    !     Standard Size
    !
    !   XXLARGE  -   ~360,000 atoms
    !
    !   The actual size varies by machine type.
    !   It is listed in the header of the CHARMM output file.
    !
  type chsize_t
     logical :: q = .false.
     integer :: n = 0
  end type chsize_t

  type chsizes
     type(chsize_t) :: chsize,maxa,maxb,maxt,maxp,maximp,maxnb,maxpad,maxres,maxseg, &
#if KEY_CMAP==1
          maxcrt, &
#endif
          maxshk,maxaim,maxgrp, &
          maxnbf, maxitc, maxcn, maxpauto
  end type chsizes

  type(chsizes), save :: new_chsize
  logical, save, private :: sizes_frozen = .false.

  integer, save :: chsize = 6 * 60120

    !  CHRSIZ - Length of the character_stack (should be at least as
    !           large as MAXA).
    !  MAXPAD - Maximum number of donors or acceptors.
    !  MAXRES - Maximum number of residues.
    !
    !  MAXLP  - Maximum number of lone-pair atoms (typ 100
    !  MAXLPH - Maximum number of lone-pair hosts (typ 500
    integer,save :: chrsiz,maxres
    integer,save :: maxlp,maxlph,maxpad

#if KEY_MMFF==1
    ! MMFF-specific information from (MSI/Merck version of) dimens.fcm
    !
    integer,parameter :: MAXDEFI=250       ! maximum number of atom types
    !                                      ! not to be confused with maximum
    !                                      ! allowed atom type MAXATC
    !
    !  NAME0  = LENGTH OF CHARACTER STRING FOR AN ATOM NAME AS RETURNED
    !           BY FUNCTION NAME OR XNAME
    !  NAMEQ0 = LENGTH OF CHARACTER STRING FOR A PARTIALLY OR FULLY
    !           QUALIFIED ATOM NAME AS RETURNED BY FUNCTION QNAME OR XQNAME
    !  NRES0  = LENGTH OF CHARACTER STRING FOR A RESIDUE NAME FIELD
    !  KRES0  = LENGTH OF CHARACTER STRING FOR A RESIDUE TYPE (E.G., "ALA")
    !
    integer,parameter :: NAME0=4,NAMEQ0=10,NRES0=4,KRES0=4
    !
    integer,parameter :: MaxAtN=55

    integer,parameter :: MAXAUX = 10 ! AuxPar(MAXAUX)
#endif

    !-----------------------------------------------------------------------
    !  FROM:  comand.fcm
    !
    !  MXCMSZ - The maximum command length (inluding all continuation lines)
    !
    integer,parameter :: MXCMSZ = 5000

    !-----------------------------------------------------------------------
    !  FROM:  etable.fcm
    !
    !  MAXATB - Maximum number of atom types for table uasge. Should
    !           probably match MAXATC.
    !
    integer,parameter :: MAXATB = 200

    !-----------------------------------------------------------------------
    !  FROM:  graph.fcm
    !
    !  IATBMX - Maximum number of bonds for any single atom.
    !
#if KEY_BLOCK==1 /*ldm*/
    integer,parameter :: IATBMX = 32 ! RLH - 16 wasn't enough
#else /**/
    integer,parameter :: IATBMX = 8
#endif

    !-----------------------------------------------------------------------
    !  FROM:  hbond.fcm
    !
    !  MAXHB - The maximum number of active hydrogen bonds.
    !      Note: Hydrogen bonds removed by post processing of hbond list
    !      for the BEST and HBEXcl option also count against this total.
    !
    integer,parameter :: MAXHB = 14000
    !
    !-----------------------------------------------------------------------
    !  FROM:  image.fcm
    !
    !  MAXTRN - The maximum number of image transformations.
    !
    integer,parameter :: MAXTRN = 5000
    !
    !  MAXSYM - The maximum number of crystal symmetry operations allowed.
    !           The maximum number ever needed in a crystal is 192 but it
    !           is conceivable that in bizarre cases one may require more.
    !               (such as for some finite space groups
    !
    integer,parameter :: MAXSYM = 192
    !-----------------------------------------------------------------------
    !  FROM:  noe.fcm
    !
    !  NOEMAX - The maximum number of NOE atoms and restraints.
    !  NOEMX2 - The maximum number of NOE atoms.
    !         -- all noe storage has been using noemax and not noemx2 in noem

    integer :: NOEMAX = 2000

    ! integer, parameter :: NOEMX2 = 4000
    !     -- was never actually used for anything in noem

    !-----------------------------------------------------------------------
    !  FROM:  param.fcm
    !
    !  MAXATC - Maximum number of different atom types.
    !  MAXCB  - Maximum number of bond parameters.
    !  MAXCT  - Maximum number of angle parameters.
    !  MAXCP  - Maximum number of dihedral parameters.
#if KEY_CMAP==1
    !  MAXCTP - Maximum number of cross-term maps
#endif
    !  MAXCI  - Maximum number of improper dihedral parameters.
    !  MAXCN  - Maximum number of vdw lookup values
    !  MAXCH  - Maximum number of hydrogen bond parameters.
    !  MAXNBF - Maximum number of nonbond fixes (vdw).
#if KEY_FLEXPARM==1
    !  MAXACTEQV- Maximum number of atom equivalences
#endif

#if KEY_CGENFF==1
    integer,parameter :: MAXATC = 1400, MAXCB = 3000, MAXCH = 6400, MAXCI = 1200, &
         MAXCP = 20000, MAXCT = 50000 !, MAXITC = 500
#elif KEY_MMFF==1 || KEY_CFF==1
    integer,parameter :: MAXATC = 500, MAXCB = 1500, MAXCH = 3200, MAXCI = 600, &
         MAXCP  = 3000, MAXCT = 50000 !,MAXITC = 500
#elif KEY_YAMMP==1
    integer,parameter :: MAXATC = 1500, MAXCB = 2000, MAXCH = 300, MAXCI = 1000, &
         MAXCP  = 1000, MAXCT = 50000 !, MAXITC=  200
#else
    integer,parameter :: MAXATC = 1000, MAXCB = 3000, MAXCH = 6400, MAXCI = 1200, &
         MAXCP  = 3000, MAXCT = 50000   !, MAXITC=  200
#endif

!    integer,parameter :: MAXCN = MAXITC*(MAXITC+1)/2

#if KEY_FLEXPARM==1
    integer,parameter :: MAXACTEQV = 40
#endif

        !  FROM:  resdist.fcm
    !
    !  REDMAX - The maximum number of distance restraints.
    !  REDMX2 - The maximum number of specified atom pairs.
    !
    !-----------------------------------------------------------------------
    !  FROM:  shake.fcm
    !
    !  MAXSHK - The maximum number of SHAKE constraints.
    !
    integer,parameter :: REDMAX = 50
    integer,parameter :: REDMX2 = 200

    integer,save :: maxa,maxb,maxt,maxp,maximp,maxnb,maxcrt,maxseg
    integer,save :: maxaim,maxgrp
    integer,save :: maxnbf
    integer, save :: maxitc, maxcn, maxpauto

    !-----------------------------------------------------------------------
    !  FROM:  shake.fcm
    !
    !  MAXSHK - The maximum number of SHAKE constraints.
    !
    integer,save :: maxshk

    !
    !-----------------------------------------------------------------------
    !  FROM:  sbound.fcm
    !
    !  NMFTAB - Maximum number of distance lookup points for any table.
    !  NMCTAB - Maximum number of boundary tables.
    !  NMCATM - Maximum number of atoms for any boundary table.
    !  NSPLIN - Order of fit (3=cubic spline).
    !

    integer,parameter :: NMFTAB = 200, NMCTAB = 3, NMCATM = 12000, NSPLIN = 3

    !
    !
    !-----------------------------------------------------------------------
    !  FROM:  string.fcm
    !
    !  SCRMAX - The maximum string length. Should match MXCMSZ.
    !

    integer,parameter :: SCRMAX = 5000

    !
    !-----------------------------------------------------------------------

  contains

    subroutine set_dimen(size_rec, new_size)
      type(chsize_t), intent(inout) :: size_rec
      integer, intent(in) :: new_size

      if (sizes_frozen) then
         call wrndie(-4, '<set_dimen>', 'Too late to change dimensions')
      else if (new_size <= 0) then
         call wrndie(-4, '<set_dimen>', 'Specified dimensions must be > 0')
      else
         size_rec%n = new_size
         size_rec%q = .true.
      endif
    end subroutine set_dimen

    subroutine clear_dimen(size_rec)
      type(chsize_t), intent(out) :: size_rec

      size_rec%n = 0
      size_rec%q = .false.
    end subroutine clear_dimen

    integer function get_dimen(size_rec, default_size)
      type(chsize_t), intent(in) :: size_rec
      integer, intent(in) :: default_size

      if (size_rec%q) then
         get_dimen = size_rec%n
      else
         get_dimen = default_size
      endif
    end function get_dimen

    subroutine set_chsize(size_arg)
      integer, intent(in) :: size_arg

      call set_dimen(new_chsize%chsize, size_arg)
      call set_dimen(new_chsize%maxa, size_arg)
      call set_dimen(new_chsize%maxb, size_arg)
      call set_dimen(new_chsize%maxt, size_arg*2)
      call set_dimen(new_chsize%maxp, size_arg*3)
      call set_dimen(new_chsize%maximp, size_arg/2)
      call set_dimen(new_chsize%maxnb,  size_arg/4)
      call set_dimen(new_chsize%maxpad, size_arg)
      call set_dimen(new_chsize%maxres, size_arg/3)
      call set_dimen(new_chsize%maxseg, size_arg/8)
#if KEY_CMAP==1
      call set_dimen(new_chsize%maxcrt, size_arg/3)
#endif
      call set_dimen(new_chsize%maxshk, size_arg)
      call set_dimen(new_chsize%maxaim, 2*size_arg)
      call set_dimen(new_chsize%maxgrp, 2*size_arg/3)
! BIOVIA Code Start
#if KEY_LICENSE==1
      call set_dimen(new_chsize%maxnbf, 1000)
      call set_dimen(new_chsize%maxitc, 500)
      call set_dimen(new_chsize%maxcn, 500*(500+1)/2)
#else
! BIOVIA Code End
      call set_dimen(new_chsize%maxnbf, size_arg/360)
      call set_dimen(new_chsize%maxitc, size_arg/360)
      call set_dimen(new_chsize%maxcn, size_arg/360*(size_arg/360+1)/2)
      call set_dimen(new_chsize%maxpauto, size_arg/8)
! BIOVIA Code Start
#endif
! BIOVIA Code End
    end subroutine set_chsize

    subroutine clear_chsize()
      call clear_dimen(new_chsize%chsize)
      call clear_dimen(new_chsize%maxa)
      call clear_dimen(new_chsize%maxb)
      call clear_dimen(new_chsize%maxt)
      call clear_dimen(new_chsize%maxp)
      call clear_dimen(new_chsize%maximp)
      call clear_dimen(new_chsize%maxnb)
      call clear_dimen(new_chsize%maxpad)
      call clear_dimen(new_chsize%maxres)
      call clear_dimen(new_chsize%maxseg)
#if KEY_CMAP==1
      call clear_dimen(new_chsize%maxcrt)
#endif
      call clear_dimen(new_chsize%maxshk)
      call clear_dimen(new_chsize%maxaim)
      call clear_dimen(new_chsize%maxgrp)
      call clear_dimen(new_chsize%maxnbf)
      call clear_dimen(new_chsize%maxitc)
      call clear_dimen(new_chsize%maxcn)
      call clear_dimen(new_chsize%maxpauto)
    end subroutine clear_chsize

    subroutine freeze_dimens()
      sizes_frozen = .true.
    end subroutine freeze_dimens

    subroutine set_dimens()
      integer :: imsize

      !--- Check if chsize has been specified.
      chsize = get_dimen(new_chsize%chsize, chsize)
      !
      !-----------------------------------------------------------------------
      !  FROM:  cstack
      !
      !  CHRSIZ - Length of the character_stack (should be at least as
      !           large as MAXA).
      !
      CHRSIZ = CHSIZE
      !
      !
      !-----------------------------------------------------------------------
      !  FROM:  lonepr
      !
      !  MAXLP  - Maximum number of lone-pair atoms (typ 100
      !  MAXLPH - Maximum number of lone-pair hosts (typ 500
      !
#if KEY_LONEPAIR==1 /*lonepair_max*/
      MAXLP  = CHSIZE/5
      MAXLPH = CHSIZE*4/5
#endif /* (lonepair_max)*/

      !-----------------------------------------------------------------------
      !  FROM:  psf
      !
      !  MAXA   - Maximum number of atoms.
      !  MAXAIM - Maximum number of atoms including image atoms.
      !  MAXB   - Maximum number of bonds.
      !  MAXT   - Maximum number of angles (thetas).
      !  MAXP   - Maximum number of dihedrals (phis).
      !  MAXIMP - Maximum number of improper dihedrals.
#if KEY_CMAP==1
      !  MAXCRT - Maximum number of cross terms
#endif
      !  MAXNB  - Maximum number of explicit nonbond exclusions.
      !  MAXSEG - Maximum number of segments.
      !  MAXGRP - Maximum number of groups.
      !  MAXSHK - The maximum number of SHAKE constraints.
      !  MAXPAUTO - Maximum number of specified autogen angles & dihedrals
      !
      !-----------------------------------------------------------------------

      maxa = get_dimen(new_chsize%maxa, chsize)
      maxb = get_dimen(new_chsize%maxb, chsize)
      maxt = get_dimen(new_chsize%maxt, chsize*2)
      maxp = get_dimen(new_chsize%maxp, chsize*3)
      maximp = get_dimen(new_chsize%maximp, chsize/2)
      maxnb  = get_dimen(new_chsize%maxnb,  chsize/4)
      maxpad = get_dimen(new_chsize%maxpad, chsize)
      maxres = get_dimen(new_chsize%maxres, chsize/3)
      maxseg = get_dimen(new_chsize%maxseg, chsize/8)
#if KEY_CMAP==1
      maxcrt = get_dimen(new_chsize%maxcrt, chsize/3)
#endif
      maxshk = get_dimen(new_chsize%maxshk, chsize)
      maxpauto = get_dimen(new_chsize%maxpauto, chsize)

      imsize = 2*chsize
      maxaim = get_dimen(new_chsize%maxaim, imsize)
      maxgrp = get_dimen(new_chsize%maxgrp, imsize/3)
      !-----------------------------------------------------------------------
      !  FROM:  param
      !
      !  MAXNBF   - Maximum number of NB FIXES.

      ! nbfix array size
      ! maxnbf = get_dimen(new_chsize%maxnbf, chsize/360)
      maxnbf = 100000
      ! set maxitc and maxcn arrays
! BIOVIA Code Start
#if KEY_LICENSE==1
      maxitc = get_dimen(new_chsize%maxitc, chsize/120)
      maxcn = get_dimen(new_chsize%maxcn, chsize/120*(chsize/120+1)/2)
#else
! BIOVIA Code End
      maxitc = get_dimen(new_chsize%maxitc, chsize/360)
      maxcn = get_dimen(new_chsize%maxcn, chsize/360*(chsize/360+1)/2)
! BIOVIA Code Start
#endif
! BIOVIA Code End
      call clear_chsize()
      return
    end subroutine set_dimens

end module dimens_fcm
