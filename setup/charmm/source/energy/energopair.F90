module gopair
  use chm_kinds
  use chm_types
  use stream
  
  implicit none
  
  private
  
  real(chm_real), allocatable, dimension(:), save :: go_eps, go_rmin
  integer, allocatable, dimension(:), save :: igopair, jgopair
  integer, save :: ngopair
  logical, save :: qgopair, qgopair_eten, qgopair_etsr, qgopair_upinb, qgopair_pbc, qgopair_noexcl
  
  public :: qgopair, ngopair, qgopair_upinb, GoPair_initialize, GoPair_setup, &
       GoPair_exclusions, EGoPair
  
contains
  
  subroutine GoPair_deallocate()
    use memory
    integer :: n
    
    if(allocated(go_eps)) n = size(go_eps)
    call chmdealloc('energygopair.src','GoPair_deallocate','go_eps',n,crl=go_eps)
    call chmdealloc('energygopair.src','GoPair_deallocate','go_rmin',n,crl=go_rmin)
    call chmdealloc('energygopair.src','GoPair_deallocate','igopair',n,intg=igopair)
    call chmdealloc('energygopair.src','GoPair_deallocate','jgopair',n,intg=jgopair)
    qgopair = .false.
    qgopair_eten = .false.
    qgopair_etsr = .false.
    ngopair = 0
    qgopair_upinb = .true.
  end subroutine GoPair_deallocate

  subroutine GoPair_allocate()
    use memory

    call chmalloc('energygopair.src','GoPair_dllocate','go_eps',ngopair,crl=go_eps)
    call chmalloc('energygopair.src','GoPair_allocate','go_rmin',ngopair,crl=go_rmin)
    call chmalloc('energygopair.src','GoPair_allocate','igopair',ngopair,intg=igopair)
    call chmalloc('energygopair.src','GoPair_allocate','jgopair',ngopair,intg=jgopair)
  end subroutine GoPair_allocate

  subroutine GoPair_initialize()
    qgopair = .false.
    ngopair = 0
    qgopair_eten = .false.
    qgopair_etsr = .false.
    qgopair_upinb = .false.
    qgopair_pbc = .false.
  end subroutine GoPair_initialize

  subroutine GoPair_setup(comlyn, comlen)
    use string, only : indxa, gtrmi
    use psf, only : natom

    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen

    integer :: GoPairUnit
    if(natom <= 0) call wrndie(-1,'<GoPair_setup>', &
         'Go pairs must be set-up after psf is built')
    if ( indxa(comlyn,comlen, "ETEN") > 0 ) then
       if ( indxa(comlyn,comlen, "ON") > 0 ) then
          qgopair_eten = .true.
          if(prnlev>2) write(outu,'(10x,a)') &
               ' GoPair_setup> ETEN functionality turned on'
       else if ( indxa(comlyn,comlen, "OFF") > 0 ) then
          qgopair_eten = .false.
          if(prnlev>2) write(outu,'(10x,a)') &
               ' GoPair_setup> ETEN functionality turned off'
       else
          qgopair_eten = .true.
          if(prnlev>2) write(outu,'(10x,a)') &
               ' GoPair_setup> ETEN functionality turned on'
       endif
       qgopair_upinb = .true.
    endif
    if ( indxa(comlyn,comlen, "ETSR") > 0 ) then
       if ( indxa(comlyn,comlen, "ON") > 0 ) then
          qgopair_etsr = .true.
          if(prnlev>2) write(outu,'(10x,a)') &
               ' GoPair_setup> ETSR functionality turned on'
       else if ( indxa(comlyn,comlen, "OFF") > 0 ) then
          qgopair_etsr = .false.
          if(prnlev>2) write(outu,'(10x,a)') &
               ' GoPair_setup> ETSR functionality turned off'
       else
          qgopair_etsr = .true.
          if(prnlev>2) write(outu,'(10x,a)') &
               ' GoPair_setup> ETSR functionality turned on'
       endif
       qgopair_upinb = .true.
    endif
    qgopair_pbc =  ( indxa(comlyn,comlen, "PBC") > 0 ) 
    qgopair_noexcl =  ( indxa(comlyn,comlen, "noex") > 0 )
    if(qgopair_noexcl.and. prnlev >=2) write(outu,'(10x,a,/,10x, a)') &
         '  GoPair_setup> GoPair functionality will use used but no exclusions will be made between', &
         '                GoPair list and normal nonboded list'
   if( indxa(comlyn,comlen, 'ON') > 0 ) then
       if(ngopair <= 0 .or. .not. allocated(go_eps)) then
          call wrndie(-1, '<GoPair_setup>', 'Go pairs not setup, use read')
       else
          qgopair = .true.
          qgopair_upinb = .true.
       endif
    else if ( indxa(comlyn,comlen, 'OFF') > 0 ) then
       qgopair = .false.
       qgopair_upinb = .true.
       if(prnlev>2) write(outu,'(10x,a)') &
            ' GoPair_setup> GoPair functionality turned off'
    else if( indxa(comlyn,comlen, 'CLEA') > 0 ) then
       call GoPair_deallocate
       if(prnlev>2) write(outu,'(10x,a)') &
            ' GoPair_setup> GoPair data structures cleared'
    else if( indxa(comlyn,comlen, 'READ') > 0 ) then
       GoPairUnit = gtrmi(comlyn, comlen, 'UNIT', -1)
       if(GoPairUnit <=0) GoPairUnit = istrm
       if(iolev > 0) then 
          call GoPair_read(GoPairUnit)
       endif
    endif
    if(qgopair_pbc .and. prnlev >=2) write(outu,'(10x,a,/,10x, a)') &
         '  GoPair_setup> GoPair functionality will use Periodic Boundary Conditions', &
         '                w/ minimum image convention as set-up via Crystal/Image facility'
  end subroutine GoPair_setup

  subroutine GoPair_read(GoPairUnit)
    use psf, only : natom
    use dimens_fcm, only : mxcmsz
    use comand
    use string, only : nextf, indxa
    use coord, only : x, y, z, wmain
    use select

    integer, intent(in) :: GoPairUnit

    real(chm_real) :: eps, rmin
    integer :: i, ip, jp, iatm, icnt, jcnt, prnlev_current
    integer :: islct(natom), jslct(natom)
    logical :: ldobynu, eof, err

    eof = .false.
    call rdcmnd(comlyn,mxcmsz,comlen,GoPairUnit,eof,.true.,.false., &
         'GoPair_read> ')
    ldobynu = .false.
    ldobynu = ( indxa(comlyn,comlen,'BYNU') > 0 )
    ngopair = nextf(comlyn,comlen)
    call GoPair_allocate()
    do i = 1, ngopair
       if (ldobynu) then
          read(GoPairUnit,*) ip, jp, eps, rmin
       else
          call rdcmnd(comlyn,mxcmsz,comlen,GoPairUnit,eof,.true.,.false., &
               'GoPair_read> ')
          prnlev_current = prnlev
          prnlev = 0
          call selctd(comlyn,comlen,islct,jslct,x,y,z,wmain,.false.,err)
          prnlev = prnlev_current
          ip = 0
          jp = 0
          icnt = 0
          jcnt = 0
          iatm = 1
          do while((ip==0 .or. jp==0) .and. iatm <= natom)
             if(islct(iatm) /= 0) then
                ip=iatm
                icnt = icnt + 1
             endif
             if(jslct(iatm) /= 0) then
                jp=iatm
                jcnt = jcnt + 1
             endif
             iatm = iatm + 1
          end do
          if(icnt+jcnt>2) call wrndie(-1,'<GoPair_read>', &
               ' More than one atom selected in GoPair parsing')
          eps = nextf(comlyn,comlen)
          rmin = nextf(comlyn,comlen)
       endif
       if(ip <= natom .and. jp <= natom .and. ip > 0 .and. jp > 0 ) then
          igopair(i) = ip
          jgopair(i) = jp
          go_eps(i) = eps
          go_rmin(i) = rmin
       endif
    enddo
    qgopair = .true.
    qgopair_upinb = .true.
    if(prnlev >= 2) &
         write(outu,'(/,10x,a,i5,a)')'GoPair_setup> Setup GoPair model for ', &
         ngopair, ' pairs'
    if(prnlev >= 6) then
       do i = 1, ngopair
          write(outu,*) igopair(i), jgopair(i), go_eps(i), go_rmin(i)
       enddo
    endif
    return
  end subroutine GoPair_read

  subroutine GoPair_exclusions(ipk, jpk, npair)
    integer, intent(inout) :: ipk(*), jpk(*), npair

    integer :: i
    qgopair_upinb = .false.
    if(qgopair_noexcl) return
    do i = 1, ngopair

       npair = npair + 1
       ipk(npair) = igopair(i)
       jpk(npair) = jgopair(i)

    enddo

  end subroutine GoPair_exclusions

  subroutine EGoPair(eGo, dx, dy, dz, x, y, z)
    use number
    use image, only : xucell
    use inbnd, only : ctofnb

    real(chm_real), intent(inout) :: eGo, dx(*), dy(*), dz(*)
    real(chm_real), intent(in) :: x(*), y(*), z(*)

    real(chm_real) :: EGoPr, dxij, dyij, dzij, df
    real(chm_real) :: r2, sig2, sig6, sig12, fac, swtmp
    real(chm_real) :: xinv, yinv, zinv
    real(chm_real) :: ctofnb2
    integer :: ipair, i, j

    if( (xucell(4)/=ninety .and. xucell(5) /= ninety .and. xucell(6) /= ninety) &
         .and. qgopair_pbc) call wrndie(-1, 'EGoPair','Not using 90 degree lattice-type')
    ctofnb2 = ctofnb * ctofnb
    if(qgopair_pbc) then
       xinv = One / xucell(1)
       yinv = One / xucell(2)
       zinv = One / xucell(3)
    endif
    do ipair = 1, ngopair
       i = igopair(ipair)
       j = jgopair(ipair)

       dxij = x(i)-x(j)
       dyij = y(i)-y(j)
       dzij = z(i)-z(j)
       
       if(qgopair_pbc) then
          dxij = xinv * dxij
          dyij = yinv * dyij
          dzij = zinv * dzij
          dxij = dxij - nint(dxij)
          dyij = dyij - nint(dyij)
          dzij = dzij - nint(dzij)
          dxij = dxij * xucell(1)
          dyij = dyij * xucell(2)
          dzij = dzij * xucell(3) 
      endif

       r2 = dxij*dxij + dyij*dyij + dzij*dzij
       if (r2 <= ctofnb2) then
          r2 = 1 / r2
          
          sig2 = go_rmin(ipair) * go_rmin(ipair) * r2
          sig6 = sig2 * sig2 * sig2
          sig12 = sig6 * sig6
          
          if (qgopair_etsr) then
             fac = four / (nine * sig2)
             swtmp = (fac * fac)**3

             ! fac = ( 2r/3sigma )^2

             EGoPr = -go_eps(ipair) * (thirtn*sig12 &
                  - nine*two*sig6*sig2*sig2 &
                  + four*sig6) &
                  / ( one + swtmp )             
             df = go_eps(ipair) * r2 * twelve &
                  / (one + swtmp) &
                  *(( thirtn*sig12 - fiftn*sig6*sig2*sig2 + two*sig6 ) &
                  + (swtmp)/(one + swtmp) &
                  * (thirtn*sig12 - (nine+nine)*sig6*sig2*sig2 &
                  + four*sig6))
          else if(qgopair_eten) then
             EGoPr = -go_eps(ipair) * (thirtn*sig12 &
                  - nine*two*sig6*sig2*sig2 &
                  + four*sig6)
             df = -go_eps(ipair) * r2 * twelve * ( fiftn*sig6*sig2*sig2 &
                  - thirtn*sig12 - two*sig6 )
          else
             EGoPr = -go_eps(ipair) * ( sig12 - sig6 - sig6 )
             df = -go_eps(ipair) * r2 * twelve * ( sig6 - sig12 )
          endif
          dx(i) = dx(i) + dxij * df
          dy(i) = dy(i) + dyij * df
          dz(i) = dz(i) + dzij * df
          
          dx(j) = dx(j) - dxij * df
          dy(j) = dy(j) - dyij * df
          dz(j) = dz(j) - dzij * df
          
          eGo = eGo + EGoPr
       endif
    enddo

  end subroutine EGoPair
end module gopair
