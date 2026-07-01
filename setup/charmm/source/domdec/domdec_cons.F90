module domdec_cons

#if KEY_DOMDEC==1

  ! *
  ! * Domain decomposition constrains and restraints handling
  ! *

  use chm_kinds
  use dimens_fcm
  implicit none
  private

  ! Public subroutines
  public init_cons, uninit_cons, print_cons

contains

  ! *
  ! * Prints out coordinates/forces that are in the cons list
  ! *
  subroutine print_cons(x, y, z)
    use psf,only:natom
    use domdec_common,only:q_cons, ncons
    implicit none
    ! Input
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    ! Variables
    integer i

    if (ncons == 0 .or. .not.allocated(q_cons)) return

    do i=1,natom
       if (q_cons(i)) then
          write (*,'(i7,3f12.5)') i, x(i), y(i), z(i)
       endif
    enddo

    return
  end subroutine print_cons

  ! *
  ! * Sets q_cons(i) to .true. and increases ncons if q_cons(i) was false
  ! *
  subroutine set_q_cons(i)
    use domdec_common,only:q_cons, ncons
    implicit none
    ! Input
    integer, intent(in) :: i

    if (.not.q_cons(i)) ncons = ncons + 1
    q_cons(i) = .true.

    return
  end subroutine set_q_cons

  ! *
  ! * Initializes constraints / restraints
  ! *
  subroutine init_cons()
    use domdec_common,only:q_cons, ncons, conslist
    use psf,only:natom
    use memory,only:chmalloc, chmdealloc
#if KEY_ADUMB==1
    use umb,only:numphi, ium, jum, kum, lum
#endif
#if KEY_DMCONS==1
    use dmcons,only:dmc_atomlist, dmc_natomlist, ndmc
#endif
#if KEY_RXNCOR==1
    use rxncom,only:ngraph, deftyp, refnod, iataa, POINT_TYPE
#endif
    use cnst_fcm,only:ncsphi, ics, jcs, kcs, lcs
#if KEY_NOMISC==0
    use noem,only:noenm2, noelis
    use resdist_ltm,only:rednm2, redilis
#endif
#if KEY_DENBIAS==1
    use denbias, only : nsel,nsel2,adlist,adlist2
    !use denbias, only : nsel,adlist
#endif
#if KEY_HMCOM==1
    use cstran_mod,only:nhmcm, ihmcm, inhmcm, phmcm
#endif
    implicit none
    ! Variables
    integer i, j, n, igraph

    if (allocated(q_cons)) then
       if (size(q_cons) < natom) then
          call chmdealloc('domdec_cons.src','init_cons','q_cons',size(q_cons),log=q_cons)
       endif
    endif

    if (.not.allocated(q_cons)) then
       call chmalloc('domdec_cons.src','init_cons','q_cons',natom,log=q_cons)
    endif

    q_cons(1:natom) = .false.
    ncons = 0

#if KEY_DMCONS==1
    ! DMCONS
    do j=1,ndmc
       do i=1,dmc_natomlist(j)
          call set_q_cons(dmc_atomlist(j,i))
       enddo
    enddo
#endif

#if KEY_ADUMB==1
    ! ADUMB
    do i=1,numphi
       !JMS 9/2012 -- integrate adaptive umbrella with DOMDEC
       call set_q_cons(ium(i))
       call set_q_cons(jum(i))
       call set_q_cons(kum(i))
       call set_q_cons(lum(i))
    enddo
#endif

#if KEY_RXNCOR==1
    ! RXNCOR
    do j=1,ngraph
       if (deftyp(j) == POINT_TYPE) then
          n = refnod(1,j)
          igraph = refnod(2,j)
          do i=1,n
             call set_q_cons(iataa(igraph)%a(i))
          enddo
       endif
    enddo
#endif

    ! CONS DIHE
    do i=1,ncsphi
       call set_q_cons(ics(i))
       call set_q_cons(jcs(i))
       call set_q_cons(kcs(i))
       call set_q_cons(lcs(i))
    enddo

#if KEY_NOMISC==0
    ! NOE
    do i=1,noenm2
       call set_q_cons(noelis(i))
    enddo

    ! REDCNS
    do i=1,rednm2
       call set_q_cons(redilis(1,i))
       call set_q_cons(redilis(2,i))
    enddo
#endif

#if KEY_DENBIAS==1
    do i=1,nsel
       call set_q_cons(adlist(i))
    enddo 
    do i=1,nsel2
       call set_q_cons(adlist2(i))
    enddo 
#endif

#if KEY_HMCOM==1
    ! CONS HMCM
    do j=1,nhmcm
       n = ihmcm(j)
       do i=0,inhmcm(j)-1
          call set_q_cons(phmcm(n+i))
       enddo
    enddo
#endif

    ! Done with q_cons, now build conslist
    ! Warning: We have a natom loop here!
    !          (Should be OK though, since this is only done in initialization)
    if (ncons > 0) then
       if (allocated(conslist)) then
          if (size(conslist) < ncons) then
             call chmdealloc('domdec_cons.src','init_cons','conslist',ncons,intg=conslist)
          endif
       endif
       if (.not.allocated(conslist)) then
          call chmalloc('domdec_cons.src','init_cons','conslist',ncons,intg=conslist)
       endif
       j = 0
       do i=1,natom
          if (q_cons(i)) then
             j = j + 1
             conslist(j) = i
          endif
       enddo
       ! Sanity check
       if (j /= ncons) then
          call wrndie(-5,'<domdec_cons>','Error building conslist')
       endif
    endif

    return
  end subroutine init_cons

  ! *
  ! * Uninitializes constraints / restraints
  ! *
  subroutine uninit_cons()
    use domdec_common,only:q_cons, ncons, conslist
    use memory,only:chmdealloc
    implicit none

    if (allocated(q_cons)) then
       call chmdealloc('domdec_cons.src','uninit_cons','q_cons',size(q_cons),log=q_cons)
    endif

    if (allocated(conslist)) then
       call chmdealloc('domdec_cons.src','uninit_cons','conslist',size(conslist),intg=conslist)
    endif

    ncons = 0

    return
  end subroutine uninit_cons

#endif

end module domdec_cons
