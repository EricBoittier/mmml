module domdec_bonded

#if KEY_DOMDEC==1 /* domdec_main */
  use chm_kinds
  use dimens_fcm
  use mpi
  use domdec_bonded_types,only:bondlist_t, anglelist_t, dihelist_t, list14_t, list14thole_t, &
       hyperlist_t
#if KEY_CMAP==1
  use domdec_bonded_types,only:cmaplist_t
#endif
  ! Use statements for domdec_bonded_kernel.inc:
  use domdec_common,only:hboxx, hboxy, hboxz
  use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
  use number,only:zero
  use iso_c_binding
  use psf,only:nbond, ntheta, nphi, nimphi
#if KEY_DOMDEC_GPU==1
  use domdec_common,only:q_gpu, gpu_code_version
  use nblist_util,only:alloc_gpu, dealloc_gpu
#endif
  implicit none
  private

  ! Bonded interaction lists
  type(bondlist_t), pointer, dimension(:) :: bondlist
  type(bondlist_t), pointer, dimension(:) :: ureyblist
  type(anglelist_t), pointer, dimension(:) :: anglelist
  type(dihelist_t), pointer, dimension(:) :: dihelist
  type(dihelist_t), pointer, dimension(:) :: imdihelist
#if KEY_CMAP==1
  type(cmaplist_t), allocatable, dimension(:) :: cmaplist
#endif
  type(list14_t), pointer, dimension(:) :: in14list
  type(list14_t), pointer, dimension(:) :: ex14list
  type(list14thole_t), allocatable, dimension(:) :: ex14tholelist
  type(hyperlist_t), allocatable, dimension(:) :: hyperlist

#if KEY_DOMDEC_GPU==1
  ! Logicals that tell wether above lists are allocated in pinned host memory
  logical :: q_bondlist_pinned = .false.
  logical :: q_ureyblist_pinned = .false.
  logical :: q_anglelist_pinned = .false.
  logical :: q_dihelist_pinned = .false.
  logical :: q_imdihelist_pinned = .false.
  logical :: q_in14list_pinned = .false.
  logical :: q_ex14list_pinned = .false.
#endif

  ! Logicals that determine if the above lists are current or not
  logical :: q_bondlist_current = .false.
  logical :: q_ureyblist_current = .false.
  logical :: q_anglelist_current = .false.
  logical :: q_dihelist_current = .false.
  logical :: q_imdihelist_current = .false.
  logical :: q_in14list_current = .false.
  logical :: q_ex14list_current = .false.
#if KEY_CMAP==1
  logical :: q_cmaplist_current = .false.
#endif
  logical :: q_ex14tholelist_current = .false.
  logical :: q_hyperlist_current = .false.

  ! Tables for bonds, angles, and dihedrals
  integer, pointer, dimension(:) :: bondtbl, angletbl, dihetbl, imdihetbl, cmaptbl, in14tbl, ex14tbl
  integer, pointer, dimension(:) :: ex14tholetbl, hypertbl
  ! Number of bonds, angles, and dihedrals in the local linked list
  integer nbondtbl, nangletbl, ndihetbl, nimdihetbl, ncmaptbl, nin14tbl, nex14tbl
  integer nex14tholetbl, nhypertbl

  ! Logical flag that determines if bonded interactions are used or not
  logical q_bonded

  ! 1-4 interactions & exclusions pair list
  integer nin14, nex14
  integer, allocatable, dimension(:) :: in14i, in14j, ex14i, ex14j

  ! Thole 1-4 exclusion pair list
  integer nex14thole
  integer, allocatable, dimension(:) :: ex14tholei, ex14tholej
  real(chm_real), allocatable, dimension(:) :: ex14thole_aa

  ! Drude hyper polarization
  integer nhyper
  integer, allocatable, dimension(:) :: hyperi

  ! Public subroutines
  public init_bonded, uninit_bonded, check_home_box
  public build_bonded_ll
  public get_bonded_ind, get_bonded_num, set_bonded_alloc_len
  public set_bonded_ntbl
  public bonded_associate_tbl
  public check_bonded_14, check_bonded_14_totals

  public build_bondlist_ps, build_bondlist_pd
  public build_anglelist_ps, build_anglelist_pd
  public build_dihelist_ps, build_dihelist_pd
  public build_xx14list_ps, build_xx14list_pd
  public build_ex14tholelist_ps, build_ex14tholelist_pd

  public realloc_bondlist, realloc_ureyblist
  public realloc_anglelist, realloc_dihelist, realloc_imdihelist
  public realloc_in14list, realloc_ex14list

  ! Public variables
  public q_bonded
  public bondtbl, angletbl, dihetbl, imdihetbl, in14tbl, ex14tbl, cmaptbl, &
       nbondtbl, nangletbl, ndihetbl, nimdihetbl, nin14tbl, nex14tbl, ncmaptbl
  public nin14, in14i, in14j, nex14, ex14i, ex14j
  public bondlist, ureyblist, anglelist, dihelist, imdihelist, in14list, ex14list
  public q_bondlist_current, q_ureyblist_current, q_anglelist_current, &
       q_dihelist_current, q_imdihelist_current, q_in14list_current, q_ex14list_current
#if KEY_CMAP==1
  public cmaplist
  public q_cmaplist_current
#endif
  public ex14tholei, ex14tholej, ex14thole_aa, hyperi
  public nex14tholetbl, ex14tholetbl, ex14tholelist
  public nhypertbl, hypertbl, hyperlist
  public q_ex14tholelist_current, q_hyperlist_current

#endif /* domdec_main */

contains

#if KEY_DOMDEC==1 /* domdec_main */
  ! *
  ! * Initializes bonded interactions
  ! *
  subroutine init_bonded()
    use psf,only:ib, jb, nbond, it, jt, kt, ntheta, natom, &
#if KEY_CMAP==1
            i1ct, j1ct, k1ct, l1ct, i2ct, j2ct, k2ct, l2ct, ncrterm, &
#endif
            ip, jp, kp, lp, nphi, im, jm, km, lm, nimphi, &
            qdrude, qhyper, isdrude
    use domdec_common,only:nthread
    use bases_fcm,only:bnbnd
    use inbnd,only:nnb14
    implicit none
    ! Variables
    integer i

    ! Build 1-4 interaction & exclusion pair lists
    call build_14_pairlist(natom, nnb14, bnbnd%inb14, bnbnd%iblo14, qdrude, isdrude, &
         in14i, in14j, nin14, ex14i, ex14j, nex14, ex14tholei, ex14tholej, ex14thole_aa, nex14thole)

    if (qdrude .and. qhyper) then
       call build_hyper_list(natom, isdrude, hyperi, nhyper)
    else
       nhyper = 0
    endif

    q_bonded = .false.
    if (nbond > 0 .or. ntheta > 0 .or. nphi > 0 .or. nimphi > 0 .or. &
#if KEY_CMAP==1
         ncrterm > 0 .or. &
#endif
         nin14 > 0 .or. nex14 > 0 .or. nex14thole > 0 .or. nhyper > 0) then
       q_bonded = .true.
    endif

!!$    ! Allocate bondlist
!!$    if (allocated(bondlist)) then
!!$       if (size(bondlist) < nbond) then
!!$          deallocate(bondlist)
!!$       endif
!!$    endif
!!$
!!$    if (.not.allocated(bondlist)) then
!!$       allocate(bondlist(nbond))
!!$    endif
!!$
!!$    ! Allocate ureyblist
!!$    if (allocated(ureyblist)) then
!!$       if (size(ureyblist) < ntheta) then
!!$          deallocate(ureyblist)
!!$       endif
!!$    endif
!!$
!!$    if (.not.allocated(ureyblist)) then
!!$       allocate(ureyblist(ntheta))
!!$    endif
!!$
!!$    ! Allocate anglelist
!!$    if (allocated(anglelist)) then
!!$       if (size(anglelist) < ntheta) then
!!$          deallocate(anglelist)
!!$       endif
!!$    endif
!!$
!!$    if (.not.allocated(anglelist)) then
!!$       allocate(anglelist(ntheta))
!!$    endif
!!$
!!$    ! Allocate dihelist
!!$    if (allocated(dihelist)) then
!!$       if (size(dihelist) < nphi) then
!!$          deallocate(dihelist)
!!$       endif
!!$    endif
!!$
!!$    if (.not.allocated(dihelist)) then
!!$       allocate(dihelist(nphi))
!!$    endif
!!$
!!$    ! Allocate imdihelist
!!$    if (allocated(imdihelist)) then
!!$       if (size(imdihelist) < nimphi) then
!!$          deallocate(imdihelist)
!!$       endif
!!$    endif
!!$
!!$    if (.not.allocated(imdihelist)) then
!!$       allocate(imdihelist(nimphi))
!!$    endif
!!$
!!$    ! Allocate in14list
!!$    if (allocated(in14list)) then
!!$       if (size(in14list) < nin14) then
!!$          deallocate(in14list)
!!$       endif
!!$    endif
!!$
!!$    if (.not.allocated(in14list)) then
!!$       allocate(in14list(nin14))
!!$    endif
!!$
!!$    ! Allocate ex14list
!!$    if (allocated(ex14list)) then
!!$       if (size(ex14list) < nex14) then
!!$          deallocate(ex14list)
!!$       endif
!!$    endif
!!$
!!$    if (.not.allocated(ex14list)) then
!!$       allocate(ex14list(nex14))
!!$    endif

    ! Allocate ex14tholelist
    if (allocated(ex14tholelist)) then
       if (size(ex14tholelist) < nex14thole) then
          deallocate(ex14tholelist)
       endif
    endif

    if (.not.allocated(ex14tholelist)) then
       allocate(ex14tholelist(nex14thole))
    endif

    ! Allocate hyperlist
    if (allocated(hyperlist)) then
       if (size(hyperlist) < nhyper) then
          deallocate(hyperlist)
       endif
    endif

    if (.not.allocated(hyperlist)) then
       allocate(hyperlist(nhyper))
    endif

    return
  end subroutine init_bonded

  ! *
  ! * Uninitialize bonded interactions
  ! *
  subroutine uninit_bonded()
    use memory,only:chmdealloc
    implicit none
    integer i, j

    call dealloc_bondlist()
    call dealloc_ureyblist()
    call dealloc_anglelist()
    call dealloc_dihelist()
    call dealloc_imdihelist()
    call dealloc_in14list()
    call dealloc_ex14list()

!!$    ! Dellocate bondlist
!!$    if (allocated(bondlist)) then
!!$       deallocate(bondlist)
!!$    endif
!!$
!!$    ! Deallocate ureyblist
!!$    if (allocated(ureyblist)) then
!!$       deallocate(ureyblist)
!!$    endif
!!$
!!$    ! Deallocate anglelist
!!$    if (allocated(anglelist)) then
!!$       deallocate(anglelist)
!!$    endif
!!$
!!$    ! Deallocate dihelist
!!$    if (allocated(dihelist)) then
!!$       deallocate(dihelist)
!!$    endif
!!$
!!$    ! Deallocate imdihelist
!!$    if (allocated(imdihelist)) then
!!$       deallocate(imdihelist)
!!$    endif
!!$
!!$    ! Deallocate in14list
!!$    if (allocated(in14list)) then
!!$       deallocate(in14list)
!!$    endif
!!$
!!$    ! Deallocate ex14list
!!$    if (allocated(ex14list)) then
!!$       deallocate(ex14list)
!!$    endif

    ! Deallocate ex14tholelist
    if (allocated(ex14tholelist)) then
       deallocate(ex14tholelist)
    endif

    ! Deallocate hyperlist
    if (allocated(hyperlist)) then
       deallocate(hyperlist)
    endif

    ! in14i
    if (allocated(in14i)) then
       call chmdealloc('domdec_bonded.src','uninit_bonded','in14i',size(in14i),intg=in14i)
    endif

    ! in14j
    if (allocated(in14j)) then
       call chmdealloc('domdec_bonded.src','uninit_bonded','in14j',size(in14j),intg=in14j)
    endif

    ! ex14i
    if (allocated(ex14i)) then
       call chmdealloc('domdec_bonded.src','uninit_bonded','ex14i',size(ex14i),intg=ex14i)
    endif

    ! ex14j
    if (allocated(ex14j)) then
       call chmdealloc('domdec_bonded.src','uninit_bonded','ex14j',size(ex14j),intg=ex14j)
    endif

    ! ex14tholei
    if (allocated(ex14tholei)) then
       call chmdealloc('domdec_bonded.src','uninit_bonded','ex14tholei',&
            size(ex14tholei),intg=ex14tholei)
    endif

    ! ex14tholej
    if (allocated(ex14tholej)) then
       call chmdealloc('domdec_bonded.src','uninit_bonded','ex14tholej',&
            size(ex14tholej),intg=ex14tholej)
    endif

    ! ex14thole_aa
    if (allocated(ex14thole_aa)) then
       call chmdealloc('domdec_bonded.src','uninit_bonded','ex14thole_aa',&
            size(ex14thole_aa),crl=ex14thole_aa)
    endif

    ! hyperi
    if (allocated(hyperi)) then
       call chmdealloc('domdec_bonded.src','uninit_bonded','hyperi',&
            size(hyperi),intg=hyperi)
    endif

    nullify(bondtbl, angletbl, dihetbl, imdihetbl, in14tbl, ex14tbl, cmaptbl,&
         ex14tholetbl, hypertbl)

    nin14 = 0
    nex14 = 0
    nex14thole = 0
    nhyper = 0

    return
  end subroutine uninit_bonded

  ! *
  ! * Associate bond, angle, etc. tables
  ! *
  subroutine bonded_associate_tbl(tbl)
    use nblist_types,only:intarray_t
    use domdec_bonded_types,only:TYPE_BOND, TYPE_ANGLE, TYPE_DIHE, TYPE_IMDIHE, TYPE_IN14,&
         TYPE_EX14, TYPE_CMAP, TYPE_EX14THOLE, TYPE_HYPER
    implicit none
    ! Input
    type(intarray_t), intent(in), target, dimension(:) :: tbl

    nullify(bondtbl, angletbl, dihetbl, imdihetbl, in14tbl, ex14tbl, cmaptbl, &
         ex14tholetbl, hypertbl)
    bondtbl   => tbl(TYPE_BOND)%array
    angletbl  => tbl(TYPE_ANGLE)%array
    dihetbl   => tbl(TYPE_DIHE)%array
    imdihetbl => tbl(TYPE_IMDIHE)%array
    in14tbl   => tbl(TYPE_IN14)%array
    ex14tbl   => tbl(TYPE_EX14)%array
    cmaptbl   => tbl(TYPE_CMAP)%array
    ex14tholetbl => tbl(TYPE_EX14THOLE)%array
    hypertbl => tbl(TYPE_HYPER)%array

    return
  end subroutine bonded_associate_tbl

  ! *
  ! * Checks that there are correct number of bonded interactions,
  ! * and 1-4 interactions and exclusions
  ! *
  subroutine check_bonded_14()
    implicit none
    ! Variables
    integer nbond_sum, nangle_sum, ndihe_sum, nimdihe_sum, nin14_sum, nex14_sum
#if KEY_CMAP==1
    integer ncmap_sum
#endif
    integer nex14thole_sum, nhyper_sum
    integer tbl(9), ntbl

    ! Sum up the numbers
    ntbl = 0
    if (q_bonded) then
       ntbl = ntbl + 1
       tbl(ntbl) = nbondtbl
       ntbl = ntbl + 1
       tbl(ntbl) = nangletbl
       ntbl = ntbl + 1
       tbl(ntbl) = ndihetbl
       ntbl = ntbl + 1
       tbl(ntbl) = nimdihetbl
       ntbl = ntbl + 1
       tbl(ntbl) = nin14tbl
       ntbl = ntbl + 1
       tbl(ntbl) = nex14tbl
#if KEY_CMAP==1
       ntbl = ntbl + 1
       tbl(ntbl) = ncmaptbl
#endif
       ntbl = ntbl + 1
       tbl(ntbl) = nex14tholetbl
       ntbl = ntbl + 1
       tbl(ntbl) = nhypertbl
    endif

    if (ntbl == 0) return

    call igcomb(tbl, ntbl)

    ntbl = 0
    if (q_bonded) then
       ntbl = ntbl + 1
       nbond_sum = tbl(ntbl)
       ntbl = ntbl + 1
       nangle_sum = tbl(ntbl)
       ntbl = ntbl + 1
       ndihe_sum = tbl(ntbl)
       ntbl = ntbl + 1
       nimdihe_sum = tbl(ntbl)
       ntbl = ntbl + 1
       nin14_sum = tbl(ntbl)
       ntbl = ntbl + 1
       nex14_sum = tbl(ntbl)
#if KEY_CMAP==1
       ntbl = ntbl + 1
       ncmap_sum = tbl(ntbl)
#endif
       ntbl = ntbl + 1
       nex14thole_sum = tbl(ntbl)
       ntbl = ntbl + 1
       nhyper_sum = tbl(ntbl)
    endif

    call check_bonded_14_totals(nbond_sum, nangle_sum, ndihe_sum, nimdihe_sum, &
         nin14_sum, nex14_sum, &
#if KEY_CMAP==1
         ncmap_sum, &
#endif
         nex14thole_sum, nhyper_sum)

    return
  end subroutine check_bonded_14

  ! *
  ! * Checks the bonded and 1-4 interaction & exclusion totals
  ! *
  subroutine check_bonded_14_totals(nbond_sum, nangle_sum, ndihe_sum, nimdihe_sum, &
       nin14_sum, nex14_sum, &
#if KEY_CMAP==1
       ncmap_sum, &
#endif
       nex14thole_sum, nhyper_sum)
    use stream,only:outu,prnlev
    use psf,only:nbond, ntheta, &
#if KEY_CMAP==1
         ncrterm, &
#endif
         nphi, nimphi
    implicit none
    ! Input
    integer, intent(in) :: nbond_sum, nangle_sum, ndihe_sum, nimdihe_sum
    integer, intent(in) :: nin14_sum, nex14_sum
#if KEY_CMAP==1
    integer, intent(in) :: ncmap_sum
#endif
    integer, intent(in) :: nex14thole_sum, nhyper_sum

    ! Check the numbers:
    if (q_bonded) then
       if (nbond_sum /= nbond .or. nangle_sum /= ntheta .or. &
#if KEY_CMAP==1
            ncmap_sum /= ncrterm .or. &
#endif
            ndihe_sum /= nphi .or. nimdihe_sum /= nimphi) then
          if (prnlev > 2) then
#if KEY_CMAP==1
             write (outu,'(a,5i8)') 'correct: nbond,ntheta,nphi,nimphi,ncmap=',&
                  nbond, ntheta, nphi, nimphi, ncrterm
             write (outu,'(a,5i8)') 'actual:  nbond,ntheta,nphi,nimphi,ncmap=',&
                  nbond_sum,nangle_sum,ndihe_sum,nimdihe_sum,ncmap_sum
#else
             write (outu,'(a,4i8)') 'correct: nbond,ntheta,nphi,nimphi=',&
                  nbond, ntheta, nphi, nimphi
             write (outu,'(a,4i8)') 'actual:  nbond,ntheta,nphi,nimphi=',&
                  nbond_sum,nangle_sum,ndihe_sum,nimdihe_sum
#endif
          endif          
          call debug_bonded(nbond_sum, nangle_sum, ndihe_sum, nimdihe_sum, ncmap_sum)
          call wrndie(-5,'<domdec_bonded>','Incorrect number of bonded interactions')
       endif
    endif
    
    if (nin14_sum /= nin14) then
       write (outu,'(a,i8,a,i8)') 'incorrect: ',nin14_sum,' correct: ',nin14
       call wrndie(-5,'<domdec_bonded>','Incorrect number of 1-4 interactions')
    endif

    if (nex14_sum /= nex14) then
       write (outu,'(a,i8,a,i8)') 'incorrect: ',nex14_sum,' correct: ',nex14
       call wrndie(-5,'<domdec_bonded>','Incorrect number of 1-4 exclusions')
    endif

    if (nex14thole_sum /= nex14thole) then
       write (outu,'(a,i8,a,i8)') 'incorrect: ',nex14thole_sum,' correct: ',nex14thole
       call wrndie(-5,'<domdec_bonded>','Incorrect number of Thole 1-4 exclusions')
    endif

    if (nhyper_sum /= nhyper) then
       write (outu,'(a,i8,a,i8)') 'incorrect: ',nhyper_sum,' correct: ',nhyper
       call wrndie(-5,'<domdec_bonded>','Incorrect number of hyper polarizations')
    endif

    return
  end subroutine check_bonded_14_totals

  ! *
  ! * Checks if the coordinates x,y,z are within home box of this node
  ! * NOTE: (x,y,z) are fractional coordinates between 0 and 1
  ! *
  subroutine check_home_box(x, y, z, inhome)
    use domdec_common,only:homeix, homeiy, homeiz
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x, y, z
    logical, intent(out) :: inhome
    ! Variables
    integer ix, iy, iz

    call coord_to_boxind(x, y, z, ix, iy, iz)
    if (ix == homeix .and. iy == homeiy .and. iz == homeiz) then
       inhome = .true.
    else
       inhome = .false.
    endif

    return
  end subroutine check_home_box

  ! *
  ! * Builds hyper polarizability list
  ! *
  subroutine build_hyper_list(natom, isdrude, hyperi, nhyper)
    use memory,only:chmalloc, chmdealloc
    implicit none
    ! Input / Output
    integer, intent(in) :: natom
    logical, intent(in) :: isdrude(:)
    integer, allocatable, dimension(:), intent(inout) :: hyperi
    integer, intent(out) :: nhyper
    ! Variables
    integer i, ii_hyper

    nhyper = 0
    do i=1,natom
       if (isdrude(i)) then
          nhyper = nhyper + 1
       endif
    enddo

    if (allocated(hyperi)) then
       if (size(hyperi) < nhyper) then
          call chmdealloc('domdec_bonded.src','build_hyper_list','hyperi',&
               size(hyperi),intg=hyperi)
       endif
    endif
    if (.not.allocated(hyperi)) then
       call chmalloc('domdec_bonded.src','build_hyper_list','hyperi',&
            nhyper,intg=hyperi)
    endif

    ii_hyper = 0
    do i=1,natom
       if (isdrude(i)) then
          ii_hyper = ii_hyper + 1
          hyperi(ii_hyper) = i
       endif
    enddo

    return
  end subroutine build_hyper_list

  ! *
  ! * Builds (and allocates): in14i(1:nin14), in14j(1:nin14), ex14i(1:nex14), ex14j(1:nex14)
  ! * ex14tholei(1:nex14thole), ex14tholej(1:nex14thole)
  ! *
  subroutine build_14_pairlist(natom, nnb14, inb14, iblo14, qdrude, isdrude, &
       in14i, in14j, nin14, ex14i, ex14j, nex14, ex14tholei, ex14tholej, ex14thole_aa, nex14thole)
    use memory,only:chmalloc, chmdealloc
    use stream,only:outu
    use psf,only:alphadp, tholei
    implicit none
    ! Input / Output
    integer, intent(in) :: natom, nnb14, inb14(:), iblo14(:)
    logical, intent(in) :: qdrude, isdrude(:)
    integer, allocatable, dimension(:), intent(inout) :: in14i, in14j, ex14i, ex14j
    integer, intent(out) :: nin14, nex14
    integer, allocatable, dimension(:), intent(inout) :: ex14tholei, ex14tholej
    real(chm_real), allocatable, dimension(:), intent(inout) :: ex14thole_aa
    integer, intent(out) :: nex14thole
    ! Variables
    integer i, j, k, ii_in14, ii_ex14, ii_ex14thole
    integer nxi, nximax
    real(chm_real) alpha1, alpha2, aa

    ! Count the number of 1-4 interactions & exclusions (nin14, nex14, nex14thole)
    nin14 = 0
    nex14 = 0
    nex14thole = 0
    do i=1,natom
       if (i > 1) then
          nxi = iblo14(i-1) + 1
       else
          nxi = 1
       endif
       nximax = iblo14(i)
       do k = nxi,nximax
          if (inb14(k) < 0) then
             ! 1-4 interaction
             nin14 = nin14 + 1
          else
             ! 1-4 exclusion
             nex14 = nex14 + 1
             if (qdrude) then
                if (isdrude(i) .and. isdrude(abs(inb14(k)))) then
                   nex14thole = nex14thole + 1
                endif
             endif
          endif
       enddo
    enddo

    ! Sanity check
    if (nin14+nex14 /= nnb14) then
       write (outu,'(a,2i8)') 'nin14+nex14,nnb14=',nin14+nex14,nnb14
       call wrndie(-5,'<domdec_bonded>',&
            'build_14_pairlist: Found incorrect number of 1-4 exclusions')
    endif

    if (allocated(in14i)) then
       if (size(in14i) < nin14) then
          call chmdealloc('domdec_bonded.src','build_14_pairlist','in14i',size(in14i),intg=in14i)
          call chmdealloc('domdec_bonded.src','build_14_pairlist','in14j',size(in14j),intg=in14j)
       endif
    endif

    if (allocated(ex14i)) then
       if (size(ex14i) < nex14) then
          call chmdealloc('domdec_bonded.src','build_14_pairlist','ex14i',size(ex14i),intg=ex14i)
          call chmdealloc('domdec_bonded.src','build_14_pairlist','ex14j',size(ex14j),intg=ex14j)
       endif
    endif

    if (.not.allocated(in14i)) then
       call chmalloc('domdec_bonded.src','build_14_pairlist','in14i',nin14,intg=in14i)
       call chmalloc('domdec_bonded.src','build_14_pairlist','in14j',nin14,intg=in14j)
    endif

    if (.not.allocated(ex14i)) then
       call chmalloc('domdec_bonded.src','build_14_pairlist','ex14i',nex14,intg=ex14i)
       call chmalloc('domdec_bonded.src','build_14_pairlist','ex14j',nex14,intg=ex14j)
    endif

    if (allocated(ex14tholei)) then
       if (size(ex14tholei) < nex14thole) then
          call chmdealloc('domdec_bonded.src','build_14_pairlist','ex14tholei',&
               size(ex14tholei),intg=ex14tholei)
          call chmdealloc('domdec_bonded.src','build_14_pairlist','ex14tholej',&
               size(ex14tholej),intg=ex14tholej)
          call chmdealloc('domdec_bonded.src','build_14_pairlist','ex14thole_aa',&
               size(ex14thole_aa),crl=ex14thole_aa)
       endif
    endif
    if (.not.allocated(ex14tholei)) then
       call chmalloc('domdec_bonded.src','build_14_pairlist','ex14tholei',&
            nex14thole,intg=ex14tholei)
       call chmalloc('domdec_bonded.src','build_14_pairlist','ex14tholej',&
            nex14thole,intg=ex14tholej)
       call chmalloc('domdec_bonded.src','build_14_pairlist','ex14thole_aa',&
            nex14thole,crl=ex14thole_aa)
    endif

    ii_in14 = 0
    ii_ex14 = 0
    ii_ex14thole = 0
    do i=1,natom
       if (i > 1) then
          nxi = iblo14(i-1) + 1
       else
          nxi = 1
       endif
       nximax = iblo14(i)
       do k = nxi,nximax
          j = abs(inb14(k))
          if (inb14(k) < 0) then
             ! 1-4 interaction
             ii_in14 = ii_in14 + 1
             in14i(ii_in14) = i
             in14j(ii_in14) = j
          else
             ! 1-4 exclusion
             ii_ex14 = ii_ex14 + 1
             ex14i(ii_ex14) = i
             ex14j(ii_ex14) = j
             if (qdrude) then
                if (i < natom .and. j < natom) then
                   if (isdrude(i+1) .and. isdrude(j+1)) then
                      ii_ex14thole = ii_ex14thole + 1
                      ex14tholei(ii_ex14thole) = i
                      ex14tholej(ii_ex14thole) = j
                      alpha1 = alphadp(i)
                      alpha2 = alphadp(j)
                      aa = alpha1*alpha2/(tholei(i) + tholei(j))**6
                      ex14thole_aa(ii_ex14thole) = aa
                   endif
                endif
             endif
          endif
       enddo
    enddo

    return
  end subroutine build_14_pairlist

  ! *
  ! * Builds bonded lists for each atom
  ! * Only the atom with the lowest index has the list
  ! * NOTE: only called in initialization
  ! *
  subroutine build_bonded_ll(ntmp, storage_size, ll_ind, ll_head, ll_data)
    use psf,only:nbond, ntheta, nphi, nimphi, ib, jb, it, jt, kt, &
#if KEY_CMAP==1
            i1ct, j1ct, k1ct, l1ct, i2ct, j2ct, k2ct, l2ct, ncrterm, &
#endif
            ip, jp, kp, lp, im, jm, km, lm
    use domdec_bonded_types,only:ll_type, add_to_list, TYPE_BOND, TYPE_ANGLE, TYPE_DIHE, &
         TYPE_IMDIHE, TYPE_IN14, TYPE_EX14, TYPE_CMAP, TYPE_EX14THOLE, TYPE_HYPER
    implicit none
    ! Input / Output
    integer, intent(inout) :: ntmp(:,:), storage_size
    integer, intent(inout) :: ll_ind, ll_head(:,:)
    type(ll_type), intent(inout) :: ll_data(:)
    ! Variables
    integer ii, t

    do ii=1,nbond
       t = min(ib(ii),jb(ii))
       if (t <= 0) cycle
       call add_to_list(TYPE_BOND, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

    do ii=1,ntheta
       t = min(it(ii),jt(ii),kt(ii))
       if (t <= 0) cycle
       call add_to_list(TYPE_ANGLE, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

    do ii=1,nphi
       t = min(ip(ii),jp(ii),kp(ii),lp(ii))
       call add_to_list(TYPE_DIHE, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

    do ii=1,nimphi
       t = min(im(ii),jm(ii),km(ii),lm(ii))
       call add_to_list(TYPE_IMDIHE, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

    do ii=1,nin14
       t = min(in14i(ii),in14j(ii))
       call add_to_list(TYPE_IN14, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

    do ii=1,nex14
       t = min(ex14i(ii),ex14j(ii))
       call add_to_list(TYPE_EX14, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

#if KEY_CMAP==1
    do ii=1,ncrterm
       t = min(i1ct(ii),j1ct(ii),k1ct(ii),l1ct(ii),i2ct(ii),j2ct(ii),k2ct(ii),l2ct(ii))
       call add_to_list(TYPE_CMAP, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo
#endif

    do ii=1,nex14thole
       t = min(ex14tholei(ii),ex14tholej(ii))
       call add_to_list(TYPE_EX14THOLE, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

    do ii=1,nhyper
       t = hyperi(ii)
       call add_to_list(TYPE_HYPER, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

    return
  end subroutine build_bonded_ll

  ! *
  ! * Debug dynamic load balanced bonded interactions
  ! * nbondw, nanglew, ndihew, nimdihew = actual (wrong) bond counts
  ! *
  subroutine debug_bonded(nbondw, nanglew, ndihew, nimdihew, ncmapw)
    use psf,only:nbond,ntheta,nphi, nimphi, &
#if KEY_CMAP==1
         ncrterm, &
         i1ct, j1ct, k1ct, l1ct, i2ct, j2ct, k2ct, l2ct,&
#endif
         ib, jb, it, jt, kt, ip, jp, kp, lp, im, jm, km, lm
    use parallel
    use mpi
    use memory
    use stream
    use groupxfast,only:invgroup, groupcenter
    use domdec_common
    use domdec_dr_common,only:comm_direct
    use domdec_dlb
    implicit none
    ! Input
    integer, intent(in) :: nbondw, nanglew, ndihew, nimdihew, ncmapw
    ! Variables
    integer i, j, k, ind(4), ig, jg, ierror
    character(4) mynodstr
    integer ntbl, nfound, foundint(2), node(2)
    integer, allocatable, dimension(:) :: tbl, tmp, disp, recvcount
    integer nmissing
    integer, parameter :: max_nmissing = 1000
    integer, allocatable, dimension(:) :: missing
    logical ok, foundit

    if (q_load_balance) then
       call write_nodeb()
    endif

    if (mynod == 0) then
       ntbl = max(nbondw, nanglew, ndihew, nimdihew)
       call chmalloc('domdec_bonded.src','debug_bonded','tbl',ntbl,intg=tbl)
       call chmalloc('domdec_bonded.src','debug_bonded','tmp',ntbl,intg=tmp)
    endif

    call chmalloc('domdec_bonded.src','debug_dlb_bonded','missing',4*max_nmissing,intg=missing)

    call chmalloc('domdec_bonded.src','debug_bonded','disp',ndirect,intg=disp)
    call chmalloc('domdec_bonded.src','debug_bonded','recvcount',ndirect,intg=recvcount)

    ! BONDS
    ! Send all bonds to root node
    call mpi_gather(nbondtbl, 1, MPI_INTEGER, recvcount, 1, MPI_INTEGER, 0, comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
         'Error in mpi_gather in debug_dlb_bonded')

    if (mynod == 0) then
       disp(1) = 0
       do i=2,ndirect
          disp(i) = disp(i-1) + recvcount(i-1)
       enddo
    endif

    call mpi_gatherv(bondtbl, nbondtbl, MPI_INTEGER, tbl, recvcount, disp, MPI_INTEGER, 0, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
         'Error in mpi_gatherv in debug_dlb_bonded')

    if (mynod == 0) then
       open (121,file='bond.txt')
       do i=1,nbondw
          write (121,'(i8)') tbl(i)
       enddo
       close(121)
       if (nbondw > nbond) then
          ! Sort the table and look for duplicates
          tmp(1:nbondw) = tbl(1:nbondw)
          call qsort_ints(tmp, nbondw)
          do i=2,nbondw
             if (tmp(i-1) == tmp(i)) then
                j = tmp(i)
                ig = invgroup(ib(j))
                jg = invgroup(jb(j))
                write (outu,'(a,3i6)') 'bondtbl, duplicate entry: j,ib(j),jb(j)=',j,ib(j),jb(j)
                write (outu,'(a,i6,3f8.3)') 'ig,xyzf=',ig,groupcenter(1,ig)*invx,&
                     groupcenter(2,ig)*invy,groupcenter(3,ig)*invz
                ! Look for the node numbers
                nfound = 0
                do k=1,ntbl
                   if (tbl(k) == j) then
                      nfound = nfound + 1
                      foundint(nfound) = k
                      if (nfound == 2) exit
                   endif
                enddo
                do nfound=1,2
                   node(nfound) = ndirect
                   do k=2,ndirect
                      if (foundint(nfound) <= disp(k)+1) then
                         node(nfound) = k-1
                         exit
                      endif
                   enddo
                enddo
                write (outu,'(a,2i7,2i3)') 'indices,nodes=',foundint(1:2),node(1:2)
             endif
          enddo
       elseif (nbondw < nbond) then
          ! Find missing bond(s)
          ! tbl = list with missing bond(s)
          ! ib,jb = correct bond(s)
          do i=1,nbond
             foundit = .false.
             do j=1,nbondw
                if (i == tbl(j)) then
                   foundit = .true.
                   exit
                endif
             enddo
             if (.not.foundit) then
                write (outu,'(a,3i8)') 'Missing bond i,ib,jb=',i,ib(i),jb(i)
                write (outu,'(a,2i8)') 'groups:',invgroup(ib(i)),invgroup(jb(i))
             endif
          enddo
       endif
    endif

    ! ANGLES
    ! Send all angles to root node
    call mpi_gather(nangletbl, 1, MPI_INTEGER, recvcount, 1, MPI_INTEGER, 0, comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
         'Error in mpi_gather in debug_dlb_bonded')

    if (mynod == 0) then
       disp(1) = 0
       do i=2,ndirect
          disp(i) = disp(i-1) + recvcount(i-1)
       enddo
    endif

    call mpi_gatherv(angletbl, nangletbl, MPI_INTEGER, tbl, recvcount, disp, MPI_INTEGER, 0, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
         'Error in mpi_gatherv in debug_dlb_bonded')

    if (mynod == 0) then
       if (nanglew > ntheta) then
          ! Sort the table and look for duplicates
          nmissing = 0
       elseif (nanglew < ntheta) then
          ! Find missing angle(s)
          ! tbl = list with missing angle(s)
          ! it,jt,kt = correct angle(s)
          nmissing = 0
          do i=1,ntheta
             foundit = .false.
             do j=1,nanglew
                if (i == tbl(j)) then
                   foundit = .true.
                   exit
                endif
             enddo
             if (.not.foundit) then
                write (outu,'(a,4i8)') 'Missing angle i,it,jt,kt=',i,it(i),jt(i),kt(i)
                write (outu,'(3f8.3)') groupcenter(1:3,invgroup(it(i)))
                write (outu,'(3f8.3)') groupcenter(1:3,invgroup(jt(i)))
                write (outu,'(3f8.3)') groupcenter(1:3,invgroup(kt(i)))
                if (nmissing < max_nmissing) then
                   nmissing = nmissing + 1
                   missing((nmissing-1)*3+1:nmissing*3) = (/ it(i),jt(i),kt(i) /)
                endif
             endif
          enddo
       endif
    endif

!!$    call mpi_bcast(nmissing,1,mpi_integer,0,comm_charmm,ierror)
!!$    if (nmissing > 0) then
!!$       call mpi_bcast(missing,3*nmissing,mpi_integer,0,comm_charmm,ierror)
!!$       ! Check which node has these atoms
!!$       do i=1,nmissing
!!$          if (sum(homezone(missing((i-1)*3+1:(i-1)*3+3))) /= 0) then
!!$             write (*,'(a,i4,3i3)') 'partial angle at: mynod,homezone=',&
!!$                  mynod,homezone(missing((i-1)*3+1:(i-1)*3+3))
!!$          endif
!!$       enddo
!!$    endif

    ! DIHEDRALS
    ! Send all dihedrals to root node
    call mpi_gather(ndihetbl, 1, MPI_INTEGER, recvcount, 1, MPI_INTEGER, 0, comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
         'Error in mpi_gather in debug_dlb_bonded')

    if (mynod == 0) then
       disp(1) = 0
       do i=2,ndirect
          disp(i) = disp(i-1) + recvcount(i-1)
       enddo
    endif

    call mpi_gatherv(dihetbl, ndihetbl, MPI_INTEGER, tbl, recvcount, disp, MPI_INTEGER, 0, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
         'Error in mpi_gatherv in debug_dlb_bonded')

    if (mynod == 0) then
       if (ndihew > nphi) then
          ! Sort the table and look for duplicates
       elseif (ndihew < nphi) then
          ! Find missing dihedral(s)
          ! tbl = list with missing dihedral(s)
          ! ip,jp,kp,lp = correct dihedral(s)
          nmissing = 0
          do i=1,nphi
             foundit = .false.
             do j=1,ndihew
                if (i == tbl(j)) then
                   foundit = .true.
                   exit
                endif
             enddo
             if (.not.foundit) then
                write (outu,'(a,5i7)') 'Missing dihedral i,ip,jp,kp,lp=',&
                     i,ip(i),jp(i),kp(i),lp(i)
                write (outu,'(3f8.3)') groupcenter(1:3,invgroup(ip(i)))
                write (outu,'(3f8.3)') groupcenter(1:3,invgroup(jp(i)))
                write (outu,'(3f8.3)') groupcenter(1:3,invgroup(kp(i)))
                write (outu,'(3f8.3)') groupcenter(1:3,invgroup(lp(i)))
                if (nmissing < max_nmissing) then
                   nmissing = nmissing + 1
                   missing((nmissing-1)*4+1:nmissing*4) = (/ ip(i),jp(i),kp(i),lp(i) /)
                endif
             endif
          enddo
       endif
    endif

    ! IMPROPER DIHEDRALS
    ! Send all dihedrals to root node
    call mpi_gather(nimdihetbl, 1, MPI_INTEGER, recvcount, 1, MPI_INTEGER, 0, comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
         'Error in mpi_gather in debug_dlb_bonded')

    if (mynod == 0) then
       disp(1) = 0
       do i=2,ndirect
          disp(i) = disp(i-1) + recvcount(i-1)
       enddo
    endif

    call mpi_gatherv(imdihetbl, nimdihetbl, MPI_INTEGER, tbl, recvcount, disp, MPI_INTEGER, 0, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
         'Error in mpi_gatherv in debug_dlb_bonded')

    if (mynod == 0) then
       if (nimdihew > nimphi) then
          ! Sort the table and look for duplicates
       elseif (nimdihew < nimphi) then
          ! Find missing improper dihedral(s)
          ! tbl = list with missing improper dihedral(s)
          ! im,jm,km,lm = correct improper dihedral(s)
          do i=1,nimphi
             foundit = .false.
             do j=1,nimdihew
                if (i == tbl(j)) then
                   foundit = .true.
                   exit
                endif
             enddo
             if (.not.foundit) then
                write (outu,'(a,5i6)') 'Missing im-dihedral i,im,jm,km,lm=',&
                     i,im(i),jm(i),km(i),lm(i)
             endif
          enddo
       endif
    endif

!!$##IF CMAP
!!$    ! CMAPs
!!$    ! Send all cmaps to root node
!!$    call mpi_gather(ncmaptbl, 1, MPI_INTEGER, recvcount, 1, MPI_INTEGER, 0, comm_direct, ierror)
!!$    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
!!$         'Error in mpi_gather in debug_dlb_bonded')
!!$
!!$    if (mynod == 0) then
!!$       disp(1) = 0
!!$       do i=2,ndirect
!!$          disp(i) = disp(i-1) + recvcount(i-1)
!!$       enddo
!!$    endif
!!$
!!$    call mpi_gatherv(cmaptbl, ncmaptbl, MPI_INTEGER, tbl, recvcount, disp, MPI_INTEGER, 0, &
!!$         comm_direct, ierror)
!!$    if (ierror /= mpi_success) call wrndie(-5,'<domdec_bonded>',&
!!$         'Error in mpi_gatherv in debug_dlb_bonded')
!!$
!!$    if (mynod == 0) then
!!$       if (ncmapw > ncrterm) then
!!$          ! Sort the table and look for duplicates
!!$       elseif (ncmapw < ncrterm) then
!!$          ! Find missing CMAP(s)
!!$          ! tbl = list with missing CMAP(s)
!!$          ! im,jm,km,lm = correct CMAP(s)
!!$          do i=1,ncrterm
!!$             foundit = .false.
!!$             do j=1,ncmapw
!!$                if (i == tbl(j)) then
!!$                   foundit = .true.
!!$                   exit
!!$                endif
!!$             enddo
!!$             if (.not.foundit) then
!!$                write (outu,'(a,i6,8i6)') 'Missing CMAPs i,ijkl=',&
!!$                     i,i1ct(i),j1ct(i),k1ct(i),l1ct(i),i2ct(i),j2ct(i),k2ct(i),l2ct(i)
!!$             endif
!!$          enddo
!!$       endif
!!$    endif
!!$##ENDIF

    ! Deallocate
    if (mynod == 0) then
       call chmdealloc('domdec_bonded.src','debug_bonded','tbl',ntbl,intg=tbl)
       call chmdealloc('domdec_bonded.src','debug_bonded','tmp',ntbl,intg=tmp)
    endif

    call chmdealloc('domdec_bonded.src','debug_bonded','missing',4*max_nmissing,intg=missing)

    call mpi_barrier(comm_direct, ierror)

    if (ndirect <= 10) then
       write(mynodstr,'(i1)') mynod
    elseif (ndirect <= 100) then
       write(mynodstr,'(i2)') mynod
    elseif (ndirect <= 1000) then
       write(mynodstr,'(i3)') mynod
    endif

    if (nanglew /= ntheta) then
       open(56,file='ijk_'//trim(mynodstr)//'.txt')
       do i=1,nangletbl
          j = angletbl(i)
          ind(1:3) = (/ it(j), jt(j), kt(j) /)
          call qsort_ints(ind, 3)
          write (56,'(3i6)') ind(1:3)
       enddo
       close(56)
    endif

    if (ndihew /= nphi) then
       open(56,file='ijkl_'//trim(mynodstr)//'.txt')
       do i=1,ndihetbl
          j = dihetbl(i)
          ind(1:4) = (/ ip(j), jp(j), kp(j), lp(j) /)
          call qsort_ints(ind, 4)
          write (56,'(4i6)') ind(1:4)
       enddo
       close(56)
    endif

    if (nimdihew /= nimphi) then
       open(56,file='ijkl2_'//trim(mynodstr)//'.txt')
       do i=1,nimdihetbl
          j = imdihetbl(i)
          ind(1:4) = (/ im(j), jm(j), km(j), lm(j) /)
          call qsort_ints(ind, 4)
          write (56,'(4i6)') ind(1:4)
       enddo
       close(56)
    endif

    call chmdealloc('domdec_bonded.src','debug_bonded','disp',ndirect,intg=disp)
    call chmdealloc('domdec_bonded.src','debug_bonded','recvcount',ndirect,intg=recvcount)

    return
  end subroutine debug_bonded

  ! *
  ! * Converts coordinates to box index
  ! * NOTE: (x,y,z) are fractional coordinates between 0 and 1
  ! *
  subroutine coord_to_boxind(x, y, z, ix, iy, iz)
    use number
    use stream,only:outu
    use domdec_common,only:nx, ny, nz, frx, fry, frz, nx_comm, ny_comm, homeix, homeiy, homeiz
    use domdec_dlb,only:q_load_balance, nodefrx, nodefry, nodefrz
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x, y, z
    integer, intent(out) :: ix, iy, iz
    ! Variables
    real(chm_real) xt, yt, zt
    integer dix, diy

    if (q_load_balance) then
       ! Just some initialization values
       iy = -1
       iz = -1
       ! x
       xt = x
       ix = 0
       do while (xt >= zero .and. ix < nx)
          ix = ix + 1
          xt = xt - nodefrx(ix)
       enddo
       dix = ix - homeix
       if (dix < -nx_comm) then
          dix = dix + nx
       elseif (dix > nx_comm) then
          dix = dix - nx
       endif
       if (dix < -nx_comm .or. dix > nx_comm) return
       ! y
       yt = y
       iy = 0
       do while (yt >= zero .and. iy < ny)
          iy = iy + 1
          yt = yt - nodefry(iy,dix)
       enddo
       diy = iy - homeiy
       if (diy < -ny_comm) then
          diy = diy + ny
       elseif (diy > ny_comm) then
          diy = diy - ny
       endif
       if (diy < -ny_comm .or. diy > ny_comm) return
       ! z
       zt = z
       iz = 0
       do while (zt >= zero .and. iz < nz)
          iz = iz + 1
          zt = zt - nodefrz(iz,diy,dix)
       enddo
    else
       ! x
       xt = x
       ix = 0
       do while (xt >= zero .and. ix < nx)
          ix = ix + 1
          xt = xt - frx
       enddo
       ! y
       yt = y
       iy = 0
       do while (yt >= zero .and. iy < ny)
          iy = iy + 1
          yt = yt - fry
       enddo
       ! z
       zt = z
       iz = 0
       do while (zt >= zero .and. iz < nz)
          iz = iz + 1
          zt = zt - frz
       enddo
    endif

    return
  end subroutine coord_to_boxind

#define BONDLIST
#define DOUBLE_PREC
#define KERNEL_NAME build_bondlist_pd
#include "domdec_bonded_kernel.inc"
#undef BONDLIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define ANGLELIST
#define DOUBLE_PREC
#define KERNEL_NAME build_anglelist_pd
#include "domdec_bonded_kernel.inc"
#undef ANGLELIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define DIHELIST
#define DOUBLE_PREC
#define KERNEL_NAME build_dihelist_pd
#include "domdec_bonded_kernel.inc"
#undef DIHELIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define XX14LIST
#define DOUBLE_PREC
#define KERNEL_NAME build_xx14list_pd
#include "domdec_bonded_kernel.inc"
#undef XX14LIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define EX14THOLELIST
#define DOUBLE_PREC
#define KERNEL_NAME build_ex14tholelist_pd
#include "domdec_bonded_kernel.inc"
#undef EX14THOLELIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define ISHIFT
#define DOUBLE_PREC
#define KERNEL_NAME calc_ishift_pd
#include "domdec_bonded_kernel.inc"
#undef ISHIFT
#undef DOUBLE_PREC
#undef KERNEL_NAME

! -------------- single precision -------------

#define BONDLIST
#define SINGLE_PREC
#define KERNEL_NAME build_bondlist_ps
#include "domdec_bonded_kernel.inc"
#undef BONDLIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#define ANGLELIST
#define SINGLE_PREC
#define KERNEL_NAME build_anglelist_ps
#include "domdec_bonded_kernel.inc"
#undef ANGLELIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#define DIHELIST
#define SINGLE_PREC
#define KERNEL_NAME build_dihelist_ps
#include "domdec_bonded_kernel.inc"
#undef DIHELIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#define XX14LIST
#define SINGLE_PREC
#define KERNEL_NAME build_xx14list_ps
#include "domdec_bonded_kernel.inc"
#undef XX14LIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#define EX14THOLELIST
#define SINGLE_PREC
#define KERNEL_NAME build_ex14tholelist_ps
#include "domdec_bonded_kernel.inc"
#undef EX14THOLELIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#define ISHIFT
#define SINGLE_PREC
#define KERNEL_NAME calc_ishift_ps
#include "domdec_bonded_kernel.inc"
#undef ISHIFT
#undef SINGLE_PREC
#undef KERNEL_NAME

#define MEMORY
#define LISTNAME bondlist
#define REALLOC_NAME realloc_bondlist
#define ALLOC_NAME alloc_bondlist
#define DEALLOC_NAME dealloc_bondlist
#define TYPENAME bondlist_t
#define BOOLNAME q_bondlist_pinned
#define NCAP nbond
#include "domdec_bonded_kernel.inc"
#undef LISTNAME
#undef REALLOC_NAME
#undef ALLOC_NAME
#undef DEALLOC_NAME
#undef TYPENAME
#undef BOOLNAME
#undef NCAP
#undef MEMORY

#define MEMORY
#define LISTNAME ureyblist
#define REALLOC_NAME realloc_ureyblist
#define ALLOC_NAME alloc_ureyblist
#define DEALLOC_NAME dealloc_ureyblist
#define TYPENAME bondlist_t
#define BOOLNAME q_ureyblist_pinned
#define NCAP ntheta
#include "domdec_bonded_kernel.inc"
#undef LISTNAME
#undef REALLOC_NAME
#undef ALLOC_NAME
#undef DEALLOC_NAME
#undef TYPENAME
#undef BOOLNAME
#undef NCAP
#undef MEMORY

#define MEMORY
#define LISTNAME anglelist
#define REALLOC_NAME realloc_anglelist
#define ALLOC_NAME alloc_anglelist
#define DEALLOC_NAME dealloc_anglelist
#define TYPENAME anglelist_t
#define BOOLNAME q_anglelist_pinned
#define NCAP ntheta
#include "domdec_bonded_kernel.inc"
#undef LISTNAME
#undef REALLOC_NAME
#undef ALLOC_NAME
#undef DEALLOC_NAME
#undef TYPENAME
#undef BOOLNAME
#undef NCAP
#undef MEMORY

#define MEMORY
#define LISTNAME dihelist
#define REALLOC_NAME realloc_dihelist
#define ALLOC_NAME alloc_dihelist
#define DEALLOC_NAME dealloc_dihelist
#define TYPENAME dihelist_t
#define BOOLNAME q_dihelist_pinned
#define NCAP nphi
#include "domdec_bonded_kernel.inc"
#undef LISTNAME
#undef REALLOC_NAME
#undef ALLOC_NAME
#undef DEALLOC_NAME
#undef TYPENAME
#undef BOOLNAME
#undef NCAP
#undef MEMORY

#define MEMORY
#define LISTNAME imdihelist
#define REALLOC_NAME realloc_imdihelist
#define ALLOC_NAME alloc_imdihelist
#define DEALLOC_NAME dealloc_imdihelist
#define TYPENAME dihelist_t
#define BOOLNAME q_imdihelist_pinned
#define NCAP nimphi
#include "domdec_bonded_kernel.inc"
#undef LISTNAME
#undef REALLOC_NAME
#undef ALLOC_NAME
#undef DEALLOC_NAME
#undef TYPENAME
#undef BOOLNAME
#undef NCAP
#undef MEMORY

#define MEMORY
#define LISTNAME in14list
#define REALLOC_NAME realloc_in14list
#define ALLOC_NAME alloc_in14list
#define DEALLOC_NAME dealloc_in14list
#define TYPENAME list14_t
#define BOOLNAME q_in14list_pinned
#define NCAP nin14
#include "domdec_bonded_kernel.inc"
#undef LISTNAME
#undef REALLOC_NAME
#undef ALLOC_NAME
#undef DEALLOC_NAME
#undef TYPENAME
#undef BOOLNAME
#undef NCAP
#undef MEMORY

#define MEMORY
#define LISTNAME ex14list
#define REALLOC_NAME realloc_ex14list
#define ALLOC_NAME alloc_ex14list
#define DEALLOC_NAME dealloc_ex14list
#define TYPENAME list14_t
#define BOOLNAME q_ex14list_pinned
#define NCAP nex14
#include "domdec_bonded_kernel.inc"
#undef LISTNAME
#undef REALLOC_NAME
#undef ALLOC_NAME
#undef DEALLOC_NAME
#undef TYPENAME
#undef BOOLNAME
#undef NCAP
#undef MEMORY

  ! *
  ! * Returns bonded atom indices
  ! *
  subroutine get_bonded_ind(atom_ind, bond_type, ind, nind)
    use psf,only:ib, jb, it, jt, kt, &
#if KEY_CMAP==1
            i1ct, j1ct, k1ct, l1ct, i2ct, j2ct, k2ct, l2ct, &
#endif
            ip, jp, kp, lp, im, jm, km, lm
    use domdec_bonded_types,only:q_solvent
    implicit none
    ! Input / Output
    integer, intent(in) :: atom_ind, bond_type
    integer, intent(out) :: ind(:), nind

    call get_bond_ind(atom_ind, bond_type, &
         ib, jb, it, jt, kt, ip, jp, kp, lp, &
         im, jm, km, lm, in14i, in14j, ex14i, ex14j, &
#if KEY_CMAP==1
         i1ct, j1ct, k1ct, l1ct, i2ct, j2ct, k2ct, l2ct, &
#endif
         ex14tholei, ex14tholej, hyperi, ind, nind)

    ! For solvents, we do not need to add the atom indices
    if (q_solvent(ind(1))) then
       nind = 0
    endif

    return
  end subroutine get_bonded_ind

  ! *
  ! * Returns bond ii atom indices to ind(1:8)
  ! *
  subroutine get_bond_ind(ii, d, &
       ib, jb, it, jt, kt, ip, jp, kp, lp, &
       im, jm, km, lm, in14i, in14j, ex14i, ex14j, &
#if KEY_CMAP==1
       i1ct, j1ct, k1ct, l1ct, i2ct, j2ct, k2ct, l2ct, &
#endif
       ex14tholei, ex14tholej, hyperi, ind, nind)
    use domdec_bonded_types,only:TYPE_BOND, TYPE_ANGLE, TYPE_DIHE, TYPE_IMDIHE,&
         TYPE_IN14, TYPE_EX14, TYPE_CMAP, TYPE_EX14THOLE, TYPE_HYPER
    implicit none
    ! Input / Output
    integer, intent(in) :: ii, d
    integer, intent(in) :: ib(:), jb(:)
    integer, intent(in) :: it(:), jt(:), kt(:)
    integer, intent(in) :: ip(:), jp(:), kp(:), lp(:)
    integer, intent(in) :: im(:), jm(:), km(:), lm(:)
    integer, intent(in) :: in14i(:), in14j(:)
    integer, intent(in) :: ex14i(:), ex14j(:)
#if KEY_CMAP==1
    integer, intent(in) :: i1ct(:), j1ct(:), k1ct(:), l1ct(:)
    integer, intent(in) :: i2ct(:), j2ct(:), k2ct(:), l2ct(:)
#endif
    integer, intent(in) :: ex14tholei(:), ex14tholej(:), hyperi(:)
    integer, intent(out) :: ind(:), nind

    if (d == TYPE_BOND) then
       ind(1) = ib(ii)
       ind(2) = jb(ii)
       nind = 2
    else if (d == TYPE_ANGLE) then
       ind(1) = it(ii)
       ind(2) = jt(ii)
       ind(3) = kt(ii)
       nind = 3
    else if (d == TYPE_DIHE) then
       ind(1) = ip(ii)
       ind(2) = jp(ii)
       ind(3) = kp(ii)
       ind(4) = lp(ii)
       nind = 4
    else if (d == TYPE_IMDIHE) then
       ind(1) = im(ii)
       ind(2) = jm(ii)
       ind(3) = km(ii)
       ind(4) = lm(ii)
       nind = 4
    else if (d == TYPE_IN14) then
       ind(1) = in14i(ii)
       ind(2) = in14j(ii)
       nind = 2
    else if (d == TYPE_EX14) then
       ind(1) = ex14i(ii)
       ind(2) = ex14j(ii)
       nind = 2
#if KEY_CMAP==1
    else if (d == TYPE_CMAP) then
       ind(1) = i1ct(ii)
       ind(2) = j1ct(ii)
       ind(3) = k1ct(ii)
       ind(4) = l1ct(ii)
       ind(5) = i2ct(ii)
       ind(6) = j2ct(ii)
       ind(7) = k2ct(ii)
       ind(8) = l2ct(ii)
       nind = 8
#endif
    else if (d == TYPE_EX14THOLE) then
       ind(1) = ex14tholei(ii)
       ind(2) = ex14tholej(ii)
       nind = 2
    else if (d == TYPE_HYPER) then
       ind(1) = hyperi(ii)
       nind = 1
    endif

    return
  end subroutine get_bond_ind

  ! *
  ! * Set allocation lengths for bonded interactions
  ! *
  subroutine set_bonded_alloc_len(alloc_len)
    use psf,only:nbond, ntheta, nphi, &
#if KEY_CMAP==1
         ncrterm, &
#endif
         nimphi
    use domdec_bonded_types,only:TYPE_BOND, TYPE_ANGLE, TYPE_DIHE, TYPE_IMDIHE,&
         TYPE_IN14, TYPE_EX14, TYPE_CMAP, TYPE_EX14THOLE, TYPE_HYPER
    implicit none
    ! Input / Output
    integer, intent(out) :: alloc_len(:)

    alloc_len(TYPE_BOND) = nbond
    alloc_len(TYPE_ANGLE) = ntheta
    alloc_len(TYPE_DIHE) = nphi
    alloc_len(TYPE_IMDIHE) = nimphi
    alloc_len(TYPE_IN14) = nin14
    alloc_len(TYPE_EX14) = nex14
#if KEY_CMAP==1
    alloc_len(TYPE_CMAP) = ncrterm
#else
    alloc_len(TYPE_CMAP) = 0
#endif
    alloc_len(TYPE_EX14THOLE) = nex14thole
    alloc_len(TYPE_HYPER) = nhyper

    return
  end subroutine set_bonded_alloc_len

  ! *
  ! * Returns the total number of bonded interactions
  ! *
  integer function get_bonded_num()
    use psf,only:nbond, ntheta, nphi, &
#if KEY_CMAP==1
         ncrterm, &
#endif
         nimphi
    implicit none

    get_bonded_num = nbond + ntheta + nphi + nimphi + nin14 + nex14 + nex14thole + nhyper
#if KEY_CMAP==1
    get_bonded_num = get_bonded_num + ncrterm
#endif

    return
  end function get_bonded_num

  ! *
  ! * Sets nbondtbl, nangletbl, etc..
  ! *
  subroutine set_bonded_ntbl(ntbl)
    use domdec_bonded_types,only:TYPE_BOND, TYPE_ANGLE, TYPE_DIHE, TYPE_IMDIHE,&
         TYPE_IN14, TYPE_EX14, TYPE_CMAP, TYPE_EX14THOLE, TYPE_HYPER
    implicit none
    ! Input / Output
    integer, intent(in) :: ntbl(:)

    nbondtbl   = ntbl(TYPE_BOND)
    nangletbl  = ntbl(TYPE_ANGLE)
    ndihetbl   = ntbl(TYPE_DIHE)
    nimdihetbl = ntbl(TYPE_IMDIHE)
    nin14tbl   = ntbl(TYPE_IN14)
    nex14tbl   = ntbl(TYPE_EX14)
    ncmaptbl   = ntbl(TYPE_CMAP)
    nex14tholetbl = ntbl(TYPE_EX14THOLE)
    nhypertbl = ntbl(TYPE_HYPER)
    
    q_bondlist_current = .false.
    q_ureyblist_current = .false.
    q_anglelist_current = .false.
    q_dihelist_current = .false.
    q_imdihelist_current = .false.
    q_in14list_current = .false.
    q_ex14list_current = .false.
#if KEY_CMAP==1
    q_cmaplist_current = .false.
#endif
    q_ex14tholelist_current = .false.
    q_hyperlist_current = .false.

    return
  end subroutine set_bonded_ntbl

#endif /* domdec_main */

end module domdec_bonded
