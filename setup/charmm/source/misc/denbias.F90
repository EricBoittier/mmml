! *
! * Developed by Vahid Mirjalili/Michael Feig 
! * (Michigan State University)
! *
module denbias

   use chm_kinds
   use chm_types
   implicit none

   logical, save :: qdenbias_init = .false., qdenbias = .false.
   
   logical, save :: qrad, qz
   real(chm_real), save :: rcyl, zup, zlow, rw, zw, tden, kforc
   real(chm_real), save :: vol_cyl
   
   integer, allocatable, dimension(:),save :: adlist, adlist2, flist, flist2
   real(chm_real),allocatable,dimension(:),save :: fmx, fmy, fmz, fmx2, fmy2, fmz2
   logical, save :: qdsel = .false.
   integer, save :: nsel, nsel2
   integer, save :: nstp

#if KEY_DENBIAS==1
   contains
!  ========================================== 
subroutine denbias_set(COMLYN,COMLEN)
use exfunc
use stream
use dimens_fcm
use psf
use number
use string
use memory
use coord
use contrl
#if KEY_PARALLEL==1
use parallel
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:domdec_system_changed
#endif

   implicit none
   character(len=*),intent(inout) :: COMLYN
   integer,intent(inout) :: COMLEN
   real(chm_real) :: dbias_ener

#if KEY_PARALLEL==1
   if (MYNOD .eq. 0) then
#endif
   write(outu,*) 'denbias: DenBias implementation'
#if KEY_PARALLEL==1
   endif
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
   if (.not. qdenbias_init) &
      call denbias_init()

   if (indxa(comlyn,comlen,'CLEA').gt.0) then
    if (PRNLEV.gt.2) write(OUTU,*) '<DENBIAS> Clearing the memory'
      call denbias_free()
   else if (indxa(comlyn,comlen,'ENER').gt.0) then
      if (qdenbias) then
        call denbias_vol_density(dbias_ener, .false., .false.)
      else
        call WRNDIE(-5,'<denbias.src> denbias_set()', &
          'Error in DenBias: Energy requested but DBIAs is not setup!!!') 
      endif
   else if (indxa(comlyn,comlen,'ANAL').gt.0) then
      if (qdenbias) then
        call denbias_vol_density(dbias_ener, .false., .true.)
      else
        call WRNDIE(-5,'<denbias.src> denbias_set()', &
          'Error in DenBias: Analysis requested but DBIAs is not setup!!!')
      endif
   else
      call denbias_process_stream(COMLYN,COMLEN)

      qdenbias = .TRUE.
      if (vol_cyl <= 0) then
         vol_cyl = (4.0*atan(1.0)) * rcyl*rcyl * (zup-zlow) * 0.001
         ! vol_cyl is in units of nm^3
      endif


      call denbias_vol_density(dbias_ener, .true., .true.)
#if KEY_PARALLEL==1
      if (MYNOD .eq. 0) then
#endif
      write(outu,'(A,F7.3,A)') 'denbias: Cylinder Volume: ',vol_cyl,'(nm^3)'
      write(outu,'(A,F7.3)') 'denbias: DenBias starting ener: ',dbias_ener
#if KEY_PARALLEL==1
      endif
#endif
   endif

end subroutine denbias_set
!  ==========================================
!  ==========================================
   subroutine denbias_init()
 use exfunc
 use stream
 use number
 use param
 use psf
 use memory

   implicit none
   integer :: i


   nstp = 19

   call chmalloc('denbias.src','denbias_init','adlist', NATOM, intg=adlist)
   call chmalloc('denbias.src','denbias_init','fmx',    NATOM, crl=fmx)
   call chmalloc('denbias.src','denbias_init','fmy',    NATOM, crl=fmy)
   call chmalloc('denbias.src','denbias_init','fmz',    NATOM, crl=fmz)
   call chmalloc('denbias.src','denbias_init','flist', NATOM, intg=flist)

   call chmalloc('denbias.src','denbias_init','adlist2', NATOM, intg=adlist2)
   call chmalloc('denbias.src','denbias_init','fmx2',    NATOM, crl=fmx2)
   call chmalloc('denbias.src','denbias_init','fmy2',    NATOM, crl=fmy2)
   call chmalloc('denbias.src','denbias_init','fmz2',    NATOM, crl=fmz2)
   call chmalloc('denbias.src','denbias_init','flist2', NATOM, intg=flist2)

   nsel = 0
   do i=1, NATOM
      adlist(i) = 0
      fmx(i) = 0
      fmy(i) = 0
      fmz(i) = 0
   enddo

   qdenbias_init = .true.

   end subroutine denbias_init
!  ==========================================
subroutine denbias_free()
use exfunc
use stream
use number
use param
use psf
use memory
#if KEY_PARALLEL==1
use parallel
#endif

   implicit none

   if (allocated(adlist)) &
     call chmdealloc('denbias.src','denbias_init','adlist', NATOM, intg=adlist)
   if (allocated(fmx)) &
     call chmdealloc('denbias.src','denbias_init','fmx',    NATOM, crl=fmx)
   if (allocated(fmy)) &
     call chmdealloc('denbias.src','denbias_init','fmy',    NATOM, crl=fmy)
   if (allocated(fmz)) &
     call chmdealloc('denbias.src','denbias_init','fmz',    NATOM, crl=fmz)
   if (allocated(flist)) &
     call chmdealloc('denbias.src','denbias_init','flist', NATOM, intg=flist)

   if (allocated(adlist2)) &
     call chmdealloc('denbias.src','denbias_init','adlist2', NATOM, intg=adlist2)
   if (allocated(fmx2)) &
     call chmdealloc('denbias.src','denbias_init','fmx2',    NATOM, crl=fmx2)
   if (allocated(fmy2)) &
     call chmdealloc('denbias.src','denbias_init','fmy2',    NATOM, crl=fmy2)
   if (allocated(fmz2)) &
     call chmdealloc('denbias.src','denbias_init','fmz2',    NATOM, crl=fmz2)
   if (allocated(flist2)) &
     call chmdealloc('denbias.src','denbias_init','flist2', NATOM, intg=flist2)


   nsel = 0
   nsel2 = 0

   qdenbias = .FALSE.
   qdenbias_init = .FALSE.

   nstp = 0

#if KEY_PARALLEL==1
   if (MYNOD .eq. 0) then
#endif
   write(outu,*) 'denbias: module memory freed!'
#if KEY_PARALLEL==1
   endif
#endif

end subroutine denbias_free
!  ==========================================

   subroutine denbias_process_stream(COMLYN,COMLEN)
use exfunc
use stream
use dimens_fcm
use psf
use number
use string
use memory
use coord
use chutil,only : getres, getseg, atomid
use contrl

use cons_rmsd,only: setrmsd0
use cnst_fcm,only: allocate_cnst

use chm_kinds
use code
use bases_fcm
use coordc
use select

#if KEY_PARALLEL==1
use parallel
#endif
 
   implicit none
   character(len=*),intent(inout) :: COMLYN
   integer,intent(inout) :: COMLEN

   integer :: i
   logical :: QERR

   integer,allocatable,dimension(:),target :: ispace
   integer,pointer,dimension(:) :: islct,jslct

   call chmalloc('denbias.src','denbias_process_stream','ispace',2*NATOM,intg=ispace)
   islct => ispace(1:natom)
   jslct => ispace(natom+1:2*natom)

#if KEY_PARALLEL==1
   if (MYNOD .eq. 0) then
#endif
   write(outu,*) 'denbias: denbias_process_stream'
#if KEY_PARALLEL==1
   endif
#endif

   rcyl   = GtrmF(Comlyn,Comlen, 'RCYL', FOUR)
   rw     = GtrmF(Comlyn,Comlen, 'RW', ONE)
   zup    = GtrmF(Comlyn,Comlen, 'ZUP', TEN)
   zlow   = GtrmF(Comlyn,Comlen, 'ZLOW', ZERO)
   zw     = GtrmF(Comlyn,Comlen, 'ZW', ONE)
   tden   = GtrmF(Comlyn,Comlen, 'TDEN', ZERO)  ! target density

   kforc   = GtrmF(Comlyn,Comlen, 'FORC', ONE)
   vol_cyl = GtrmF(Comlyn,Comlen, 'VOLU', -1.0)


   if (INDXA(COMLYN,COMLEN,'ASEL').gt.0) then    !! single atom selection 
       call selcta(comlyn,comlen,islct,x,y,z,wmain,.true.)
       qdsel = .false.
       nsel2 = 0
   else if (INDXA(COMLYN,COMLEN,'DSEL').gt.0) then !! double atom selection
       call SELCTD(COMLYN,COMLEN,islct,jslct,X,Y,Z,wmain,.true.,QERR)
       qdsel = .true.
   else
       call WRNDIE(-5,'<denbias.src> denbias_set()', &
       'Error in DenBias: Atom Selection Error')
   endif

   do i=1,natom
       if (islct(i) > 0) then
          nsel = nsel + 1
          adlist(nsel) = i
       endif
   enddo

   if (qdsel) then
       do i=1,natom
          if (jslct(i) > 0) then
             nsel2 = nsel2 + 1
             adlist2(nsel2) = i
          endif
       enddo
   endif
   
#if KEY_PARALLEL==1
   if (MYNOD .eq. 0) then
#endif
   write(outu,'(4X,A,1x,I5)') 'DENBIAS> NSEL1   ', nsel
   write(outu,'(4X,A,1x,I5)') 'DENBIAS> NSEL2   ', nsel2
   write(outu,'(4X,A,F7.2,2x,A)') 'DENBIAS> RCYL   ', rcyl, ' (Angstroms)'
   write(outu,'(4X,A,F7.2,2x,A)') 'DENBIAS> ZUP    ', zup, ' (Angstroms)'
   write(outu,'(4X,A,F7.2,2x,A)') 'DENBIAS> ZLOW   ', zlow, ' (Angstroms)'
   write(outu,'(4X,A,F7.2,2x,A)') 'DENBIAS> RW     ', rw, ' (Angstroms)'
   write(outu,'(4X,A,F7.2,2x,A)') 'DENBIAS> ZW     ', zw , ' (Angstroms)'
   write(outu,'(4X,A,F10.5,2x,A)') 'DENBIAS> TDEN   ', tden, ' (1/nm^3)'
   write(outu,'(4X,A,F10.3,2x,A)') 'DENBIAS> FORC   ', &
                   kforc, ' (kcal/mol/nm^6)'
#if KEY_PARALLEL==1
   endif
#endif

end subroutine denbias_process_stream
!  ==========================================
!  ==========================================
subroutine denbias_vol_density(edenbias, qforc, qprint)
   ! Calculates the volume density of the selected atoms
   ! that exist in the cylinder (centered at zero, along Z-axis)
   ! and computes the forces for each corresponding atom 
   ! to adjust the target (desired) density (tden)
use exfunc
use stream
use dimens_fcm
use psf
use number
use string
use memory
use coord
use contrl
use consta
use deriv      !  DX,DY,DZ - Force components
#if KEY_PARALLEL==1
use parallel
#endif
 
   implicit none
   logical, intent(in) :: qforc, qprint
   real(chm_real),intent(out) :: edenbias

   integer :: i, ai, nf, nf2
   real(chm_real) :: vr, vsum, vsum2
   real(chm_real) :: rin, rinSq, rout, routSq, cf1r, cf2r
   real(chm_real) :: vz, zup_in, zup_out, zlow_in, zlow_out
   real(chm_real) :: cf1z, cf2z, dza, dzaSq, dzaCub, dvol_z
   real(chm_real) :: xa, ya, za, ra, rdSq, dr, drSq, drCub
   real(chm_real) :: dvol_r, vden, vden2, vdiff
   logical::qfi
   
   vsum = ZERO
   nf = 0
   do i=1,nsel
      fmx(i) = 0.0
      fmy(i) = 0.0
      fmz(i) = 0.0
      flist(i) = 0
   enddo



   nstp = nstp + 1

   rin = rcyl - rw
   rinSq = rin * rin
   rout = rcyl + rw
   routSq = rout * rout

   zup_in   = zup  - zw
   zup_out  = zup  + zw
   zlow_in  = zlow + zw
   zlow_out = zlow - zw

   cf1r = THREE/(FOUR * rw)
   cf2r = ONE / (FOUR * rw*rw*rw)
   cf1z = THREE/(FOUR * zw)
   cf2z = ONE / (FOUR * zw*zw*zw)

#if KEY_PARALLEL==1
   if (MyNOD .eq. 0) then
#endif
   do i=1, nsel
      qfi = .false.

      ai = adlist(i)
      xa = X(ai)
      ya = Y(ai)
      za = Z(ai)
      if (za <= zup_out .and. za>=zlow_out) then
         rdSq = xa*xa + ya*ya
         if (rdSq <= routSq) then

            if (rdSq <= rinSq) then
               vr = ONE
               dvol_r = ZERO
            else if (rdSq <= routSq) then
              if (za <= zup_out .and. za>= zlow_out) then
                ra = sqrt(rdSq)
                dr = ra-rcyl
                drSq = dr*dr
                drCub = drSq * dr
                vr = HALF - cf1r*dr  +  cf2r*drCub

                dvol_r = -cf1r + 3.0*cf2r*drSq
                qfi = .true.
              endif
            endif

            if (za <= zup_in .and. za>=zlow_in) then
               vz = ONE
               dvol_z = ZERO
            else 
               if (rdSq <= routSq) then
                  if (za > zup_in) then 
                     dza = za - zup
                  else if (za < zlow_in) then
                     dza = zlow - za
                  endif

                  dzaSq = dza*dza
                  dzaCub = dzaSq*dza
                  vz = HALF - cf1z*dza + cf2z*dzaCub
                  
                  dvol_z = -cf1z + 3.0*cf2z*dzaSq
                  if (za < zlow_in) dvol_z = -dvol_z
                   
              endif
            endif

            if (qfi) then
               nf = nf  +1
               flist(nf) = ai
               fmx(nf) = dvol_r*xa/ra * vz 
               fmy(nf) = dvol_r*ya/ra * vz 
               fmz(nf) = vr * dvol_z
            endif

            vsum = vsum + vr * vz

         endif
      endif ! if za
   enddo 
   vden = vsum / vol_cyl
#if KEY_PARALLEL==1
   endif
#endif
   !!!edenbias = kforc*(vden-tden)*(vden-tden) !! moved to after processing selection 2

 ! ************************************
 ! *********** Selection 2 ************
 ! ************************************

  if (qdsel ) then
   vsum2 = ZERO
   nf2 = 0
   do i=1,nsel2
      fmx2(i) = 0.0
      fmy2(i) = 0.0
      fmz2(i) = 0.0
      flist2(i) = 0
   enddo

#if KEY_PARALLEL==1
   if (MyNOD .eq. 0) then
#endif
   do i=1, nsel2
      qfi = .false.

      ai = adlist2(i)
      xa = X(ai)
      ya = Y(ai)
      za = Z(ai)
      if (za <= zup_out .and. za>=zlow_out) then
         rdSq = xa*xa + ya*ya
         if (rdSq <= routSq) then

            if (rdSq <= rinSq) then
               vr = ONE
               dvol_r = ZERO
            else if (rdSq <= routSq) then
              if (za <= zup_out .and. za>= zlow_out) then
                ra = sqrt(rdSq)
                dr = ra-rcyl
                drSq = dr*dr
                drCub = drSq * dr
                vr = HALF - cf1r*dr  +  cf2r*drCub

                dvol_r = -cf1r + 3.0*cf2r*drSq
                qfi = .true.
              endif
            endif

            if (za <= zup_in .and. za>=zlow_in) then
               vz = ONE
               dvol_z = ZERO
            else
              if (rdSq <= routSq) then
                  if (za > zup_in) then
                     dza = za - zup
                  else if (za < zlow_in) then
                     dza = zlow - za
                  endif

                  dzaSq = dza*dza
                  dzaCub = dzaSq*dza
                  vz = HALF - cf1z*dza + cf2z*dzaCub

                  dvol_z = -cf1z + 3.0*cf2z*dzaSq
                  if (za < zlow_in) dvol_z = -dvol_z
              endif
            endif

            if (qfi) then
               nf2 = nf2  +1
               flist2(nf2) = ai
               fmx2(nf2) =  dvol_r*xa/ra * vz
               fmy2(nf2) =  dvol_r*ya/ra * vz
               fmz2(nf2) = vr * dvol_z

            endif

            vsum2 = vsum2 + vr * vz

         endif
      endif
   enddo
   vden2 = vsum2 / vol_cyl
#if KEY_PARALLEL==1
   endif
#endif
  endif ! if qdsel

  vdiff = vden - vden2


  edenbias = HALF * kforc*(vdiff-tden)*(vdiff-tden)

#if KEY_PARALLEL==1   
  if (MYNOD /= 0) then
  edenbias = 0
  endif
#endif


  !!!! *** Write the values: *** !!!

#if KEY_PARALLEL==1
   if (MYNOD .eq. 0) then
#endif
   if (qprint) then
      write(outu,'(10x,A,3(1x,F14.8),A)')'VolDen> ', &
            vden, vden2, vdiff, ' (1/nm^3)'

   !else
     ! if ((nstp .eq. 50) .or. (.not. qforc)) then
     !  write(outu,'(10x,A,4(1x,F10.4))')'C39-Debug> ', &
     !      vden,vden2,vdiff, edenbias
     !  nstp = 0
     ! endif
   endif

#if KEY_PARALLEL==1
   endif
#endif

   do i=1, nf
      ai = flist(i)

      DX(ai) = DX(ai) + kforc*(vdiff-tden)*fmx(i)/vol_cyl
      DY(ai) = DY(ai) + kforc*(vdiff-tden)*fmy(i)/vol_cyl
      DZ(ai) = DZ(ai) + kforc*(vdiff-tden)*fmz(i)/vol_cyl


      !write(25,'(A,I4,3(1x,F12.6))')'F-X ', ai,X(ai), fmx(i), DX(ai)
      !write(25,'(A,I4,3(1x,F12.6))')'F-Y ', ai,Y(ai), fmy(i), DY(ai)
      !write(25,'(A,I4,3(1x,F12.6))')'F-Z ', ai,Z(ai), fmz(i), DZ(ai)
   enddo

   if (qdsel) then  !! forces on 2nd selection
     do i=1, nf2
        ai = flist2(i)
        DX(ai) = DX(ai) - kforc*(vdiff-tden)*fmx2(i)/vol_cyl
        DY(ai) = DY(ai) - kforc*(vdiff-tden)*fmy2(i)/vol_cyl
        DZ(ai) = DZ(ai) - kforc*(vdiff-tden)*fmz2(i)/vol_cyl
      !write(26,'(A,I4,3(1x,F12.6))')'F-X ', ai,X(ai), fmx2(i), DX(ai)
      !write(26,'(A,I4,3(1x,F12.6))')'F-Y ', ai,Y(ai), fmy2(i), DY(ai)
      !write(26,'(A,I4,3(1x,F12.6))')'F-Z ', ai,Z(ai), fmz2(i), DZ(ai)
     enddo
   endif


end subroutine denbias_vol_density


!  ==========================================
!  ==========================================


#endif
!  =========================================
  end module denbias

