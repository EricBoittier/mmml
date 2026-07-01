module evb_mod
#if KEY_ENSEMBLE==1
  use chm_kinds      ! number kinds
  use dimens_fcm      ! useful dimensioning information.
  ! ------------------------------------------------------------------------------------------
  ! DESCRIPTION
  ! ------------------------------------------------------------------------------------------
  ! This module implements Multi Reference Emperical Valence Bond calcuations. 
  !
  ! Multiple states are entered into CHARMM via different topology files (and possibly
  ! differrent parameter files).
  !
  ! Each state is assigned to a different MPI process.  This is set up in subroutine ENSNENSEM
  ! in ensemble.src
  !
  ! EVBSETUP: reads in parameters from the input file and calls EVB_ALLOCATE which allocates
  !           memory and sets certain parameters. It is called from subroutine ENSCMD in
  !           ensemble.src.
  ! EVB_DEALLOCATE: is called from subroutine STOPCH (which is launched at the STOP command)
  ! EVB:  gathers potential energies from all states, diagonalizes EVB Hamiltonian and 
  !       calculates forces. This is called from subroutine OLD_ENERGY in energy.src    
  ! 
  ! Originally coded by D. Glowacki (September 2010), with contributions from M. O'Connor, 
  !   J. Harvey, R. Arbon
  ! 
  ! ------------------------------------------------------------------------------------------
  ! PUBLIC VARIABLES
  ! ------------------------------------------------------------------------------------------
  logical :: qevb
  ! ------------------------------------------------------------------------------------------
  ! COMMON VARIABLES AND ARRAYS
  ! 
  ! evbiidx, evbjidx - contains the row/column indices of the various coupling elements. 
  ! esel, esel2 - the atom selections used to define the bonds for 1D and 2D Gaussian elements
  ! nread - number of coupling elements.
  ! coutype - coupling type (CONSstant or GAUSsian)
  ! couprm - values of the various coupling elements. 
  ! prnlev_high/low - two levels for output. 
  ! ------------------------------------------------------------------------------------------
  integer(chm_int8), dimension(:), allocatable :: evbiidx, evbjidx
  integer(chm_int8), dimension(:,:), allocatable :: esel, esel2
  integer(chm_int8) :: nread
  integer :: dielev, evbu
  character(len=4), dimension(:), allocatable :: coutype
  real(chm_real), dimension(:,:), allocatable :: couprm
  logical :: prnlev_low, prnlev_high, header_flag
  integer(chm_int4) :: buflen
  integer(chm_int4) :: maxcoup

  ! ------------------------------------------------------------------------------------------
  ! EVB_SETUP ARRAYS
  ! 
  ! emslct  - scratch array for atom selection 
  ! ------------------------------------------------------------------------------------------
  integer(chm_int4), dimension(:), allocatable :: emslct

  ! ------------------------------------------------------------------------------------------
  ! EVB ARRAYS
  ! evbmat - EVB Hamiltonian matrix
  ! uptri - upper tri EVB matrix
  ! revecs - eigenvector matrix
  ! evals - eigenvalue array
  ! smevec - smallest eigenvector
  ! s1-7 - scratch arrays for matrix diagonalization
  ! rcdxdip - MPI buffers for dipole calculation 
  ! dis1/2 - distance for Gauss' coupling elements
  ! dmdz - gradient coupling matrices
  ! ------------------------------------------------------------------------------------------
  real(chm_real), dimension(:,:), allocatable :: evbmat, revecs       
  real(chm_real), dimension(:), allocatable :: uptri, evals, smevec  
  real(chm_real), dimension(:), allocatable :: s1,s2,s3,s4,s5,s6,s7
  real(chm_real), dimension(:), allocatable :: rcdxdip, rcdydip, rcdzdip     
  real(chm_real), dimension(:), allocatable :: dis1, dis2  
  real(chm_real), dimension(:,:), allocatable :: dmdx, dmdy, dmdz     
  ! ------------------------------------------------------------------------------------------
  ! PUBLIC VS PRIVATE ARRAYS
  ! 
  ! ------------------------------------------------------------------------------------------

  public :: qevb
  ! common
  private :: evbiidx, evbjidx, esel, esel2, nread, coutype, couprm, prnlev_high, prnlev_low
  private :: evbu, header_flag, buflen, maxcoup
  ! evb_setup
  private :: emslct
  ! evb 
  private :: evbmat, revecs, uptri, evals, smevec, s1,s2,s3,s4,s5,s6,s7, rcdxdip, rcdydip, rcdzdip
  private :: dis1, dis2, dmdx, dmdy, dmdz


contains

! ------------------------------------------------------------------------------------------
! EVB_ALLOCATE
! ------------------------------------------------------------------------------------------
! Allocates arrays for EVB routines
! 

subroutine evb_allocate

use memory
use chm_kinds      
use ensemble, only: nensem, ensdx, ensdy, ensdz, ensh
use psf, only: natom
use dimens_fcm, only: maxaim 
use stream, only: prnlev, iolev

implicit none

buflen = natom*nensem
maxcoup  = nensem*(nensem-1)*100

call chmalloc('evb_ltm.src','evb_allocate','evbiidx',maxcoup,ci8=evbiidx)  
call chmalloc('evb_ltm.src','evb_allocate','evbjidx',maxcoup,ci8=evbjidx)  
call chmalloc('evb_ltm.src','evb_allocate','esel',maxcoup,2,ci8=esel)  
call chmalloc('evb_ltm.src','evb_allocate','esel2',maxcoup,2,ci8=esel2) 
call chmalloc('evb_ltm.src','evb_allocate','coutype',maxcoup,ch4=coutype) 
call chmalloc('evb_ltm.src','evb_allocate','couprm',maxcoup,9,crl=couprm)
call chmalloc('evb_ltm.src','evb_allocate','emslct',maxaim,ci4=emslct) 
call chmalloc('evb_ltm.src','evb_allocate','ensdx',buflen,crl=ensdx)
call chmalloc('evb_ltm.src','evb_allocate','ensdy',buflen,crl=ensdy)
call chmalloc('evb_ltm.src','evb_allocate','ensdz',buflen,crl=ensdz)
call chmalloc('evb_ltm.src','evb_allocate','ensh',nensem,crl=ensh)
call chmalloc('evb_ltm.src','evb_allocate','evbmat',nensem,nensem,crl=evbmat)
call chmalloc('evb_ltm.src','evb_allocate','uptri',nensem*(nensem+1)/2,crl=uptri)
call chmalloc('evb_ltm.src','evb_allocate','revecs',nensem,nensem,crl=revecs)
call chmalloc('evb_ltm.src','evb_allocate','evals',nensem,crl=evals)
call chmalloc('evb_ltm.src','evb_allocate','smevec',nensem,crl=smevec)
call chmalloc('evb_ltm.src','evb_allocate','s1',nensem,crl=s1)
call chmalloc('evb_ltm.src','evb_allocate','s2',nensem,crl=s2)
call chmalloc('evb_ltm.src','evb_allocate','s3',nensem,crl=s3)
call chmalloc('evb_ltm.src','evb_allocate','s4',nensem,crl=s4)
call chmalloc('evb_ltm.src','evb_allocate','s5',nensem,crl=s5)
call chmalloc('evb_ltm.src','evb_allocate','s6',nensem,crl=s6)
call chmalloc('evb_ltm.src','evb_allocate','s7',nensem,crl=s7)
call chmalloc('evb_ltm.src','evb_allocate','rcdxdip',nensem,crl=rcdxdip)
call chmalloc('evb_ltm.src','evb_allocate','rcdydip',nensem,crl=rcdydip)
call chmalloc('evb_ltm.src','evb_allocate','rcdzdip',nensem,crl=rcdzdip)
call chmalloc('evb_ltm.src','evb_allocate','dis1',nensem*(nensem+1)/2,crl=dis1)
call chmalloc('evb_ltm.src','evb_allocate','dis2',nensem*(nensem+1)/2,crl=dis2)
call chmalloc('evb_ltm.src','evb_allocate','dmdx',nensem,nensem,crl=dmdx)
call chmalloc('evb_ltm.src','evb_allocate','dmdy',nensem,nensem,crl=dmdy)
call chmalloc('evb_ltm.src','evb_allocate','dmdz',nensem,nensem,crl=dmdz)

prnlev_low = iolev > 0 .and. prnlev > 2
prnlev_high = iolev > 0 .and. prnlev > 6
dielev = -5
evbu = -1
header_flag = .true.

end subroutine evb_allocate

! ------------------------------------------------------------------------------------------
! EVB_DEALLOCATE
! ------------------------------------------------------------------------------------------
! Deallocates arrays for EVB routines
! 

subroutine evb_deallocate

use memory
use chm_kinds      
use ensemble, only: nensem, ensdx, ensdy, ensdz, ensh
use psf, only: natom
use dimens_fcm, only: maxaim 

implicit none


call chmdealloc('evb_ltm.src','evb_deallocate','evbiidx',maxcoup,ci8=evbiidx)  
call chmdealloc('evb_ltm.src','evb_deallocate','evbjidx',maxcoup,ci8=evbjidx)  
call chmdealloc('evb_ltm.src','evb_deallocate','esel',maxcoup,2,ci8=esel)  
call chmdealloc('evb_ltm.src','evb_deallocate','esel2',maxcoup,2,ci8=esel2) 
call chmdealloc('evb_ltm.src','evb_deallocate','coutype',maxcoup,ch4=coutype) 
call chmdealloc('evb_ltm.src','evb_deallocate','couprm',maxcoup,9,crl=couprm)
call chmdealloc('evb_ltm.src','evb_deallocate','emslct',maxaim,ci4=emslct) 
call chmdealloc('evb_ltm.src','evb_deallocate','ensdx',buflen,crl=ensdx)
call chmdealloc('evb_ltm.src','evb_deallocate','ensdy',buflen,crl=ensdy)
call chmdealloc('evb_ltm.src','evb_deallocate','ensdz',buflen,crl=ensdz)
call chmdealloc('evb_ltm.src','evb_deallocate','ensh',nensem,crl=ensh)
call chmdealloc('evb_ltm.src','evb_deallocate','evbmat',nensem,nensem,crl=evbmat)
call chmdealloc('evb_ltm.src','evb_deallocate','uptri',nensem*(nensem+1)/2,crl=uptri)
call chmdealloc('evb_ltm.src','evb_deallocate','revecs',nensem,nensem,crl=revecs)
call chmdealloc('evb_ltm.src','evb_deallocate','evals',nensem,crl=evals)
call chmdealloc('evb_ltm.src','evb_deallocate','smevec',nensem,crl=smevec)
call chmdealloc('evb_ltm.src','evb_deallocate','s1',nensem,crl=s1)
call chmdealloc('evb_ltm.src','evb_deallocate','s2',nensem,crl=s2)
call chmdealloc('evb_ltm.src','evb_deallocate','s3',nensem,crl=s3)
call chmdealloc('evb_ltm.src','evb_deallocate','s4',nensem,crl=s4)
call chmdealloc('evb_ltm.src','evb_deallocate','s5',nensem,crl=s5)
call chmdealloc('evb_ltm.src','evb_deallocate','s6',nensem,crl=s6)
call chmdealloc('evb_ltm.src','evb_deallocate','s7',nensem,crl=s7)
call chmdealloc('evb_ltm.src','evb_deallocate','rcdxdip',nensem,crl=rcdxdip)
call chmdealloc('evb_ltm.src','evb_deallocate','rcdydip',nensem,crl=rcdydip)
call chmdealloc('evb_ltm.src','evb_deallocate','rcdzdip',nensem,crl=rcdzdip)
call chmdealloc('evb_ltm.src','evb_deallocate','dis1',nensem*(nensem+1)/2,crl=dis1)
call chmdealloc('evb_ltm.src','evb_deallocate','dis2',nensem*(nensem+1)/2,crl=dis2)
call chmdealloc('evb_ltm.src','evb_deallocate','dmdx',nensem,nensem,crl=dmdx)
call chmdealloc('evb_ltm.src','evb_deallocate','dmdy',nensem,nensem,crl=dmdy)
call chmdealloc('evb_ltm.src','evb_deallocate','dmdz',nensem,nensem,crl=dmdz)

end subroutine evb_deallocate


! ------------------------------------------------------------------------------------------
! EVBSETUP
! ------------------------------------------------------------------------------------------
! Sets up EVB calculation
! This calls EVB_ALLOCATE which sets up various arrays and then parses all the user commands. 
! The parsed commands define the diagonal and off-diagonal elements of the EVB matrix.
! The most important arrays are listed as common in the module header. 
! See docs for examples of the commands that can be entered.   

subroutine evbsetup(comlyn, comlen) 

use stream         
use psf
use ensemble       
use number          
use coord           
use string           
use memory
use select

implicit none

! Arguments

character(len=*) :: comlyn
integer(chm_int4) :: comlen  

! Local variables

integer(chm_int4) :: i,j,ii,jj,kk,ng,gi,nsel
integer(chm_int4) :: maxii,maxjj 
integer(chm_int4) :: ncouple,itmp,nodeidx,ctr
logical :: joffset,jcouple,found,evbcons,evbgaus,eof
real(chm_real) :: smalla, smallb, smallc,shftval
character(len = 4) :: wrd
logical :: onedcrd, twodcrd

qevb = .true.         

! Allocate arrays

call evb_allocate

! Parameters for reading the matrix coupling elements
! ncouple - number of coupling elements
! maxii/jj - maximum index number i.e. the extent of the EVB Hamiltonian matrix. 

ncouple=nensem*(nensem-1)/2
maxii=nensem-1
maxjj=nensem-1

! Print out information to the user

if (prnlev_low) then
  write(outu,'(a)') ' Using Multi-Reference MM (EVB).'
  write(outu,'(a,i3,a)') ' Ensemble EVB setting all ',ncouple,' elem&
    &ents of the coupling matrix to a default value of zero.'
  write(outu,'(a)')' Ensemble EVB setting all energy shifts to the d&
    &efault value of zero.'
  write(outu,'(a,i3,a,i3,a)')' Energy will be the lowest eigenvalu&
    &e of a ',nensem,' by ',nensem,' coupling matrix.'
endif

! Zero out the array that holds the energy shifts. This is declared in 
! module ensemble

do i=1,nensem
  expoff(i) = 0.0
end do

! ------------------------------------------------------------------------
! MAIN COMMAND READING LOOP
! ------------------------------------------------------------------------

eof=.false.
call rdcmnd(comlyn,mxcmsz,comlen,istrm,eof,.true., .true.,'EVB> ')     
wrd=curra4(comlyn,comlen)
ctr=0
nread=0
do while ((wrd.ne.'ENVB').and.(.not.eof)) !READ COMMAND
  ctr=ctr+1

! Read in diagonal element energy shifts

  if(wrd.eq.'SHFT') then !COUP/SHIFT
    wrd=nexta4(comlyn,comlen)
    nodeidx=nexti(comlyn,comlen)
    shftval=nextf(comlyn,comlen)        
    if (nodeidx.lt.0) then !NODEIDX
      if (prnlev_low) then
        write(outu,'(a,i3,a)') ' Node index ', nodeidx ,' is negative! It must be zero or greater.'
      end if
      call wrndie(dielev,'ENSEMBLE>','Node index is negative!') 
    else if (nodeidx.gt.nensem-1) then !NODEIDX
      if (prnlev_low) then
        write(outu,'(a,i3,a,i3)') ' Node index ', nodeidx ,' is greater than the maximum allowable node index of ', nensem-1
      end if 
      call wrndie(dielev,'ENSEMBLE>','Node index is too large!')         
    else !NODEIDX
      expoff(nodeidx+1)=shftval
      if (prnlev_low) then
        write(outu,'(a,i3,a,f12.5)') ' Node ', nodeidx,' will be shifted in energy by ', expoff(nodeidx+1)
      end if
    end if

! Read in off-diagonal coupling elements. 

  else if(wrd.eq.'COUP') then !COUP/SHIFT

    wrd=nexta4(comlyn,comlen)
    if (prnlev_low) write(outu,'(a)') ' '
    nread=nread+1
    ii = nexti(comlyn,comlen)
    jj = nexti(comlyn,comlen)

! > Error check indices

    if(ii.gt.maxii) then !IDXCHK
      if (prnlev_low) then
        write(outu,'(a,i3,a,i3)')'The node index ', ii,' is &
          &greater than the maximum allowable node index of ',maxii
      end if 
      call wrndie(dielev,'ENSEMBLE>','Node index too large!')
    else if(jj.gt.maxjj) then !IDXCHK
      if (prnlev_low) then
        write(outu,'(a,i3,a,i3)')'The node index ', jj,' is &
          &greater than the maximum allowable node index of ',maxjj
      end if
      call wrndie(dielev,'ENSEMBLE>','Node index too large!')        

    else if(ii.lt.0) then !IDXCHK
      if (prnlev_low) then
        write(outu,'(a,i3,a)')'The node index ', ii,' is negative!&
          & It must be zero or greater.'
      end if
      call wrndie(dielev,'ENSEMBLE>','Node index is negative!')
    else if(jj.lt.0) then !IDXCHK
      if (prnlev_low) then
        write(outu,'(a,i3,a)')'The node index ', jj,' is negative!&
          & It must be zero or greater.'
      end if
      call wrndie(dielev,'ENSEMBLE>','Node index is negative!')
    else if(ii.eq.jj) then !IDXCHK
      if (prnlev_low) then
        write(outu,'(a,i3,a,i3,a)')'The specified node indices ', ii &
          &,' and ', jj, ' are equal.'
      end if
      call wrndie(dielev,'ENSEMBLE>','Node indices are equal.')
    end if !IDXCHK

! > Swap indices if they're given wrong way around

    if(ii.gt.jj) then
      itmp=ii
      ii=jj
      jj=itmp
    end if

    evbiidx(nread)=ii
    evbjidx(nread)=jj
    evbcons=.false.
    evbgaus=.false.
    onedcrd=.false.
    twodcrd=.false.
    if (prnlev_low) then
      write(outu,'(a,i3,a,i3,a)')' Reading terms for inclusion in EVB &
        &coupling element ',ii,' -',jj,' ...'
    end if 

! > Determine type of coupling (CONStant of GAUSSian)

    wrd=nexta4(comlyn,comlen)

    if (wrd.eq.'CONS') then
      evbcons=.true.
    else if (wrd.eq.'GAUS') then
      evbgaus=.true.
    end if

! > Read in constant coupling

    if (evbcons) then !CONS/GAUSS
      couprm(nread,1) = nextf(comlyn,comlen)
      if (prnlev_low) then
        write(outu,'(a,f12.5)')' Specified term is a constant with value: ',couprm(nread,1)
      end if  
      coutype(nread)='CONS'

! > Read in Gaussian coupling

    else if (evbgaus) then !CONS/GAUSS
      wrd=nexta4(comlyn,comlen)
      if (wrd.eq.'ONED') then
        onedcrd=.true.
      else if (wrd.eq.'TWOD') then
        twodcrd=.true.
      end if

! >>> Determine whether Gaussian is 1D or 2D

      if(onedcrd) then !1D/2D
        onedcrd=.true.
        coutype(nread)='ONED'
        if (prnlev_low) then
          write(outu,'(a)')' Specified term is a 1-D gaussian.'
          write(outu,'(a)')' It depends on a distance R1:'
          write(outu,'(a)')' A*exp(-(R1-R01)**2/(2C1**2))'
        end if
      else if (twodcrd) then !1D/2D
        twodcrd=.true.
        coutype(nread)='TWOD'
        if (prnlev_low) then
          write(outu,'(a)')' Specified term is a 2-D Gaussian.'
          write(outu,'(a)')' It depends on two distances R1 & R2:'
          write(outu,'(a)')' A*exp(-(a[R1-R01]**2 + 2b[R1-R01][R2-R02] +&
            & c[R2-R02]**2))'
          write(outu,'(a)')' where:'
          write(outu,'(a)')' a = [cos(THETA)]**2/(2*C1**2) + [sin(THETA)&
            &]**2/(2*C2**2) '
          write(outu,'(a)')' b = -sin(2*THETA)/(4*C1**2) + sin(2*THETA)&
            &/(4*C2**2) '
          write(outu,'(a)')' c = [sin(THETA)]**2/(2*C1**2) + [cos(THETA)&
            &]**2/(2*C2**2) '
        end if
      else !1D/2D
        if (prnlev_low) then
          write(outu,'(a)')' Unrecognized input option for Gaussian co&
            &upling element...exiting.'
        end if 
        call wrndie(dielev,'ENSEMBLE>','For EVB with Gaussian coupling y&
          &ou must specify the keyword ONED or TWOD')
      end if !1D/2D

! >>> Read the 1D Gaussian parameters

      if (prnlev_low) write(outu,'(a)') ' Reading A, R01, C1 parameters'
      couprm(nread,1) = nextf(comlyn,comlen)
      if (prnlev_low) write(outu,'(a,f12.5)')' A   : ',couprm(nread,1)
      couprm(nread,2) = nextf(comlyn,comlen)
      if (prnlev_low) write(outu,'(a,f12.5)')' R01 : ',couprm(nread,2)
      couprm(nread,3) = nextf(comlyn,comlen)
      if (prnlev_low) write(outu,'(a,f12.5)')' C1  : ',couprm(nread,3)

! >>> Read in the first atom selection; this is always required

      emslct=0.0d0
      if (onedcrd.or.twodcrd) then !1DOR2D
        if (prnlev_low) write(outu,'(a)') ' Reading atom selection to define R1'
        call selcta(comlyn,comlen,emslct,x,y,z,wmain,.true.)
        nsel = 0
        do ii=1,maxaim
          if (emslct(ii) .eq. 1) then
            nsel = nsel + 1
            esel(nread,nsel) = ii
          end if
        end do
        if(nsel.lt.2.or.nsel.gt.2) then
          if (prnlev_low) then
            write(outu,'(a)') ' EVB with Gaussian coupling elements re&
              &quires specification of 2 atoms'
          endif 
          call wrndie(dielev,'ENSEMBLE>','For EVB with Gaussian coupling&
            & you must select 2 atoms')
        else
          if (prnlev_low) then
            write(outu,'(a)') ' Atom selections:'
            do kk=1,nsel
              write(outu,'(a,i6)')' Atom ',esel(nread,kk)
            end do
          end if
        end if

! >>>>> Reset the elements of emslct to zero

        emslct(nread)=0
        ! if there's no second selection, set esel2 elements to zero
        if (onedcrd) then
          esel2(nread,1)=0
          esel2(nread,2)=0

! >>>>> Now read the additional 2D Gaussian parameters: THETA, R02, C2

        else if(twodcrd) then

          if (prnlev_low) write(outu,'(a)') ' Reading THETA, R02, C2 parameters'
          couprm(nread,4) = nextf(comlyn,comlen)
          if (prnlev_low) write(outu,'(a,f12.5)') ' THETA : ',couprm(nread,4)
          couprm(nread,5) = nextf(comlyn,comlen)
          if (prnlev_low) write(outu,'(a,f12.5)') ' R02   : ',couprm(nread,5)
          couprm(nread,6) = nextf(comlyn,comlen)
          if (prnlev_low) write(outu,'(a,f12.5)') ' C2    : ',couprm(nread,6)

! >>>>>>> Derive the parameters for 2D Gaussian coupling

          smalla=((cos(couprm(nread,4)))**2)/(2.0d0*couprm(nread,3)**2) &
            &+((sin(couprm(nread,4)))**2)/(2.0d0*couprm(nread,6)**2)

          smallb=-1.0d0*sin(2.0d0*couprm(nread,4))/   &
          &(4.0d0*couprm(nread,3)**2)               &
          &+sin(2.0d0*couprm(nread,4))/(4.0d0*couprm(nread,6)**2)

          smallc=((sin(couprm(nread,4)))**2)/(2.0d0*couprm(nread,3)**2) &
          &+((cos(couprm(nread,4)))**2)/(2.0d0*couprm(nread,6)**2)

          couprm(nread,7)=smalla
          couprm(nread,8)=smallb
          couprm(nread,9)=smallc
          if (prnlev_low) then
            write(outu,'(a,f12.5)')' Derived parameter a =',smalla
            write(outu,'(a,f12.5)')' Derived parameter b =',smallb
            write(outu,'(a,f12.5)')' Derived parameter c =',smallc
          endif

! >>>>>>> Read in second atom selection for second bond distance. 

          if (prnlev_low) write(outu,'(a)') ' Atom selection to define R2:'
          call selcta(comlyn,comlen,emslct,x,y,z,wmain,.true.)
          nsel = 0
          do ii=1,maxaim
            if (emslct(ii) .eq. 1) then
              nsel = nsel + 1
              esel2(nread,nsel) = ii
            end if
          end do

          if (nsel.lt.2.or.nsel.gt.2) then
            if (prnlev_low) then
              write(outu,'(a)') ' EVB with Gaussian coupling elements re&
                &quires specification of 2 atoms'
            endif
            call wrndie(dielev,'ENSEMBLE>','For EVB with Gaussian coupling&
              & you must select 2 atoms')
          else
            if (prnlev_low) then 
              write(outu,'(a)')' Atom selections:'
              do kk=1,nsel
                write(outu,'(a,i6)')' Atom ',esel2(nread,kk)
              end do

            end if  

          end if

        endif

      end if !1DOR2D  

    end if !CONS/GAUSS

! Read in unit to save dipole output to

  else if (wrd.eq.'EVBS') then

    ! get save unit
      evbu=gtrmi(comlyn,comlen,'UNIT',-1)
      if ((iolev.ge.0).and.(evbu.gt.0)) then
        if (prnlev_low) then
          write(outu, '(a,i5)') 'Saving EVB output to UNIT ', evbu
        end if 
      else if ((iolev.ge.0).and.(evbu.eq.-1)) then
        if (prnlev_low) then
          write(outu,'(a)') ' Unrecognised unit - not saving EVB output.' 
        end if
        call wrndie(dielev,'ENSEMBLE>','Unrecognised unit number') 
      end if

  end if !COUP/SHIFT

! Get the next line

  call rdcmnd(comlyn,mxcmsz,comlen,istrm,eof,.true., &
    & .true.,'EVB> ')   
  wrd=curra4(comlyn,comlen)

! Break out of command loop if ENVB is reached. 

  if(wrd.eq.'ENVB') then !ENVB
    wrd=nexta4(comlyn,comlen)
    if (prnlev_low) write(outu,'(a)')'ENVB...end of EVB input block'
  end if !ENVB

end do ! READ COMMAND 

! Provide summary

if (prnlev_low) then
  write(outu,'(a)') ' '
  write(outu,'(a)') ' ---------------------------------------------'
  write(outu,'(a)') ' Energy shift summary: '
  do i=1,nensem
    write(outu,'(a,i3,a,f12.5)') ' Node = ', i-1,' Shift = ', expoff &
      &(i)
  end do
  write(outu,'(a)') ' ---------------------------------------------'
  write(outu,'(a)') ' '

! Write out the matrix coupling elements to check input is correct

  write(outu,'(a)') ' '
  write(outu,'(a)') ' ---------------------------------------------'
  write(outu,'(a)') ' Coupling matrix element summary: '
  do ii=1,nensem
    do jj=ii+1,nensem
      found=.false.
      write(outu,'(a)') ' '
      write(outu,'(a,i3,a,i3,a)')' > Element which couples nodes ', &
      &ii-1,' & ', jj-1,' is a sum of the following terms: '
      do j=1,nread
        if ((ii-1).eq.evbiidx(j).and.(jj-1).eq.evbjidx(j)) then
          found=.true.
          if (coutype(j).eq.'CONS') then
            write(outu,'(a,f12.5)')' >> Constant   : ',couprm(j,1)
          else if (coutype(j).eq.'ONED') then
            write(outu,'(a,a,f12.5,a,f12.5,a,f12.5)')' >> 1D Gaussian: ', &
              &'A: ', couprm(j,1),' R01: ',couprm(j,2),' C1: ',couprm(j,3)
            write(outu,'(a,i6,a,i6)')' >>> Depends on one distance : ',esel(j,1), &
              &' - ',esel(j,2)
          else if (coutype(j).eq.'TWOD') then
            write(outu,'(a,a,f12.5,a,f12.5,a,f12.5,a,f12.5,a,f12.5,a, f12.5)') &
              &' >> 2D Gaussian: ','A: ',couprm(j,1),' R01: ',couprm(j,2),' C1: ', couprm(j,3)&
              &,' THETA: ', couprm(j,4),' R02: ',couprm(j,5),' C2: ',couprm(j,6)
            write(outu,'(a,i6,a,i6,a,i6,a,i6)') ' >>> Depends on two distances: ',&
              &esel(j,1),' - ',esel(j,2),' and ',esel2(j,1),' - ',esel2(j,2)               
          end if     
        end if
      end do

      if (.not.found) then
       write(outu,'(a)')' Default value is constant: 0.0000 '            
      end if

    end do
  end do
  write(outu,'(a)') ' ---------------------------------------------'
  write(outu,'(a)') ' '
endif

        
  


  end subroutine evbsetup

! ------------------------------------------------------------------------------------------
! EVB
! ------------------------------------------------------------------------------------------
! Calculates EVB energies and forces:
! 1) Potential energies (hk) from all other MPI processes (EVB states) are gathered 
! 2) EVB matrix (evbmat) is set up using gathered energies and the coupling/shift parameters
!     setup in EVBSETUP.
! 3) EVB matrix is diagonalized and the smallest eigenvalue (the potential energy) 
!     and eigenvector (smevec) are recorded.
! 4) The forces are calculated using the Feynmann-Hellman theorem.
!

 subroutine evb(qx, qy, qz, dxk, dyk, dzk, hk, natom)
! ---------------------------------------------------------------------


use param_store, only: get_param, find_param
use stream          
use ensemble         
use mpi            
use contrl          
use number          
use chm_kinds
use new_timer, only: timer_stop, timer_start, T_EVB, T_EVB_Comms, T_EVB_Energy, T_EVB_Forces


implicit none

! Arguments:
! qx/y/z - positions
! dx/y/zk - forces
! hk - potential energy
! natom - number of atoms

real(chm_real) :: qx(*), qy(*), qz(*)                
real(chm_real) :: dxk(*), dyk(*), dzk (*)            
real(chm_real) :: hk                                 
integer(chm_int4) :: natom                    

! Local variables:
!  various counteers etc. 

integer(chm_int4) :: i,j,ctr,ii,jj,gi       
integer(chm_int4) :: m,ng,ceidx              
integer(chm_int4) :: ierror                 
real(chm_real) :: tmph(1),csqd 
                                  
! dummy variables to hold various parts of force/energy calculation

real(chm_real) :: dxevb,dyevb,dzevb          
real(chm_real) :: xdip(1),ydip(1),zdip(1)       
real(chm_real) :: ddis1dx,ddis1dy,ddis1dz       
real(chm_real) :: ddis2dx,ddis2dy,ddis2dz
real(chm_real) :: dgddis1,dgddis2
real(chm_real) :: ga,gr01,gc1,gr02,gau,smalla,smallb,smallc
real(chm_real) :: gautrm, gtrm2
real(chm_real) :: diffmx,diffmy,diffmz
logical :: found

call timer_start(T_EVB)

tmph(1) = hk

! Get dipole moment if it's been set and user has requested saving the 
! state average dipole moment values. 

if ((iolev.ge.0).and.(evbu.gt.0)) then

  call find_param('XDIP',xdip(1), found)

  if (found) then

    call get_param('XDIP',xdip(1))
    call get_param('YDIP',ydip(1))
    call get_param('ZDIP',zdip(1))

  else

    if (prnlev_low) then
      write(outu,'(a)') ' Dipole moment missing - please set before calling ENERgy'
    end if
    call wrndie(dielev,'ENSEMBLE>','Dipole moment missing')

  end if

end if 

call timer_start(T_EVB_Comms)

! ------------------------------------------------------------------------
! COMMUNICATION
! ------------------------------------------------------------------------
! Gather forces (d[x/y/z]k), energies(tmph), dipole moments ([x/y/z]dip) from each process and distribute them to all 
! other processes. 

call mpi_barrier(mpi_comm_world, ierror)
call mpi_allgather(dxk(1),natom,mpi_double_precision,ensdx(1), &
  natom,mpi_double_precision,mpi_comm_world,ierror)
call mpi_allgather(dyk(1),natom,mpi_double_precision,ensdy(1), &
  natom,mpi_double_precision,mpi_comm_world,ierror)
call mpi_allgather(dzk(1),natom,mpi_double_precision,ensdz(1), &
  natom,mpi_double_precision,mpi_comm_world,ierror)
call mpi_allgather(tmph(1),1,mpi_double_precision,ensh(1), &
  1,mpi_double_precision,mpi_comm_world,ierror)
call mpi_allgather(xdip,1,mpi_double_precision,rcdxdip(1), &
  1,mpi_double_precision,mpi_comm_world,ierror)
call mpi_allgather(ydip,1,mpi_double_precision,rcdydip(1), &
  1,mpi_double_precision,mpi_comm_world,ierror)
call mpi_allgather(zdip,1,mpi_double_precision,rcdzdip(1), &
  1,mpi_double_precision,mpi_comm_world,ierror)

call timer_stop(T_EVB_Comms)

call timer_start(T_EVB_Energy)

! ------------------------------------------------------------------------
! CALCULATE ENERGY
!------------------------------------------------------------------------
! Initialise EVB matrix

do ii=1,nensem
  evbmat(ii,ii)=0.0
  do jj=ii+1,nensem
    evbmat(ii,jj)=0.0
    evbmat(jj,ii)=0.0
  enddo
enddo

! Reads in various matrix elements to EVB matrix 

do ii=1,nensem

! Read in diagonal elements (SHFTs)

  evbmat(ii,ii)=ensh(ii)+expoff(ii)

! Read in off-diagonal elements

  do jj=ii+1,nensem

    evbmat(ii,jj)=0.0d0

! > Loop over total number of coupling elements

    do j=1,nread      

! >>> Check if coupling element belongs in this part of the matrix

      if((ii-1).eq.evbiidx(j).and.(jj-1).eq.evbjidx(j)) then

! >>>>> Read in constant coupling

        if(coutype(j).eq.'CONS') then

          evbmat(ii,jj)=evbmat(ii,jj) + couprm(j,1)

! >>>>> Read in Gaussian couplings.  The 1D Gaussian parameters are common
!       to both 1D and 2D so read in regardless of 1D/2D

        else if(coutype(j).eq.'ONED'.or.coutype(j).eq.'TWOD') then 

          ga=couprm(j,1)
          gr01=couprm(j,2)
          gc1=couprm(j,3)
          dis1(j)=sqrt((qx(esel(j,1))-qx(esel(j,2)))**2 &
            &+(qy(esel(j,1))-qy(esel(j,2)))**2 &
            &+(qz(esel(j,1))-qz(esel(j,2)))**2)
  
! >>>>>>> Add 1D Gaussian term

          if(coutype(j).eq.'ONED') then

            gau=ga*exp(((dis1(j)-gr01)**2)/(-2.0d0*gc1**2))
            evbmat(ii,jj)=evbmat(ii,jj) + gau

! >>>>>>> Read in extra 2D Gaussian parameters and resultant coupling
!         term into EVB matrix. 

          else if(coutype(j).eq.'TWOD') then

            dis2(j)=sqrt((qx(esel2(j,1))-qx(esel2(j,2)))**2 &
              &+(qy(esel2(j,1))-qy(esel2(j,2)))**2 &
              &+(qz(esel2(j,1))-qz(esel2(j,2)))**2)
  
            gr02=couprm(j,5)
            
            smalla=couprm(j,7)
            smallb=couprm(j,8)
            smallc=couprm(j,9)

            gau=ga*exp(-1.0d0*(smalla*(dis1(j)-gr01)**2 &
            &+2.0d0*smallb*(dis1(j)-gr01)*(dis2(j)-gr02) & 
            &+smallc*(dis2(j)-gr02)**2))

            evbmat(ii,jj)=evbmat(ii,jj) + gau

          endif

        endif     

      endif

    enddo

! > Make matrix symettric

    evbmat(jj,ii)=evbmat(ii,jj)

  enddo

enddo

! Diagonalize matrix

ctr=1
do i=1,nensem
  do j=i,nensem
    uptri(ctr)=evbmat(i,j)
    ctr=ctr+1
  enddo
enddo

call diagq(nensem,nensem,uptri,revecs,s1,s2,s3,s4,evals,s5,s6,s7,0)

! Find lowest eigenvalue

hk=evals(1)
ctr=1
do i=1,nensem
  if(evals(i).lt.hk) then
    hk=evals(i)
    ctr=i
  endif
enddo

! Get the eigenvector corresponding to the smallest eigenvalue

do i=1,nensem
  smevec(i)=revecs(i,ctr)
enddo     

call timer_stop(T_EVB_Energy)

call timer_start(T_EVB_Forces)

! ------------------------------------------------------------------------
! CALCULATE FORCES/GRADIENTS
! ------------------------------------------------------------------------
! Get the gradients using the formula from 
! P. Lancaster, Numerische Mathematik 6, 377-387 (1964). this is identical
! to using the hellman-feynmann approach.
!
! If M is evbmat, its eigenvector matrix is u, and its eigenvalue
! matrix is D (i.e., D=(U^T)*M*U), then dD/dp=(U^T)*(dM/dp)*(U), 
! where M depends on some parameter p. in this case, p represents the 
! cartesian directions - x,y,z, so that we have dM/dx, dM/dy, and dM/dz.
!
! For a two state system, this method gives results identical
! to those obtained with an analytic formula, but is extendable to 
! multi-state systems where analytic forms are not available.
! NB:
! Analytic 2-state gradients (e.g., dxk) would be calculated as follows:
! dxk(i)=0.5*(ensdx(i)+ensdx(natom+i)-tmpfac*(ensdx(i)-ensdx(natom+i)))
! where tmpfac=(v1-v2)/sqrt((v1-v2)**2+4*ensebeta**2)
! J.N.Harvey version of gradients routine

! Add diagonal part of forces

do i=1,natom

  dxevb=0.0d0
  dyevb=0.0d0
  dzevb=0.0d0

  do ii=1,nensem
  
    dxevb=dxevb+smevec(ii)**2*ensdx(natom*(ii-1)+i)
    dyevb=dyevb+smevec(ii)**2*ensdy(natom*(ii-1)+i)
    dzevb=dzevb+smevec(ii)**2*ensdz(natom*(ii-1)+i)
  
  enddo
  
  dxk(i)=dxevb
  dyk(i)=dyevb
  dzk(i)=dzevb

enddo

! Loop over the coupling terms. Note constant terms don't contribute to off-diagonal gradients. 

do ii= 1, nread

! 1D coupling terms

  if(coutype(ii).eq.'ONED') then

    ddis1dx = (qx(esel(ii, 1)) - qx(esel(ii, 2)))/dis1(ii)
    ddis1dy = (qy(esel(ii, 1)) - qy(esel(ii, 2)))/dis1(ii)
    ddis1dz = (qz(esel(ii, 1)) - qz(esel(ii, 2)))/dis1(ii)

    gau=couprm(ii,1)*exp(-(dis1(ii)-couprm(ii,2))**2/(2.d0*couprm(ii,3)**2))
    gautrm=-gau*(dis1(ii)-couprm(ii,2))/(couprm(ii,3)**2)

! > Calculate derivative wrt correct eigenvector component

    gtrm2=gautrm*smevec(evbiidx(ii)+1)*smevec(evbjidx(ii)+1)*2.d0
    
! > Apply to relevant atoms

    dxk(esel(ii,1))=dxk(esel(ii,1))+gtrm2*ddis1dx
    dyk(esel(ii,1))=dyk(esel(ii,1))+gtrm2*ddis1dy
    dzk(esel(ii,1))=dzk(esel(ii,1))+gtrm2*ddis1dz
    dxk(esel(ii,2))=dxk(esel(ii,2))-gtrm2*ddis1dx
    dyk(esel(ii,2))=dyk(esel(ii,2))-gtrm2*ddis1dy
    dzk(esel(ii,2))=dzk(esel(ii,2))-gtrm2*ddis1dz

! 1D coupling terms 

  else if(coutype(ii).eq.'TWOD') then

    ddis1dx = (qx(esel(ii, 1)) - qx(esel(ii, 2)))/dis1(ii)
    ddis1dy = (qy(esel(ii, 1)) - qy(esel(ii, 2)))/dis1(ii)
    ddis1dz = (qz(esel(ii, 1)) - qz(esel(ii, 2)))/dis1(ii)
    ddis2dx = (qx(esel2(ii, 1)) - qx(esel2(ii, 2)))/dis2(ii)
    ddis2dy = (qy(esel2(ii, 1)) - qy(esel2(ii, 2)))/dis2(ii)
    ddis2dz = (qz(esel2(ii, 1)) - qz(esel2(ii, 2)))/dis2(ii)

    ga=couprm(ii,1)
    gr01=couprm(ii,2)
    gr02=couprm(ii,5)

    smalla=couprm(ii,7)
    smallb=couprm(ii,8)
    smallc=couprm(ii,9)

    gau=ga*exp(-1.0d0*(smalla*(dis1(ii)-gr01)**2&
    &          +2.0d0*smallb*(dis1(ii)-gr01)*(dis2(ii)-gr02)&
    &          +smallc*(dis2(ii)-gr02)**2))

! > Calculate the chain rule gradients:

    dgddis1=-2.0d0*smalla*(dis1(ii)-gr01)&
    &      -2.0d0*smallb*(dis2(ii)-gr02)
    dgddis2=-2.0d0*smallb*(dis1(ii)-gr01)&
    &      -2.0d0*smallc*(dis2(ii)-gr02)

    diffmx=gau*dgddis2*ddis2dx
    diffmy=gau*dgddis2*ddis2dy
    diffmz=gau*dgddis2*ddis2dz

    gtrm2 = smevec(evbiidx(ii)+1)*smevec(evbjidx(ii)+1)*2.d0        
    
    dxk(esel2(ii,1))=dxk(esel2(ii,1))+gtrm2*diffmx
    dyk(esel2(ii,1))=dyk(esel2(ii,1))+gtrm2*diffmy
    dzk(esel2(ii,1))=dzk(esel2(ii,1))+gtrm2*diffmz
    dxk(esel2(ii,2))=dxk(esel2(ii,2))-gtrm2*diffmx
    dyk(esel2(ii,2))=dyk(esel2(ii,2))-gtrm2*diffmy
    dzk(esel2(ii,2))=dzk(esel2(ii,2))-gtrm2*diffmz
    
    diffmx=gau*dgddis1*ddis1dx
    diffmy=gau*dgddis1*ddis1dy
    diffmz=gau*dgddis1*ddis1dz

    dxk(esel(ii,1))=dxk(esel(ii,1))+gtrm2*diffmx
    dyk(esel(ii,1))=dyk(esel(ii,1))+gtrm2*diffmy
    dzk(esel(ii,1))=dzk(esel(ii,1))+gtrm2*diffmz
    dxk(esel(ii,2))=dxk(esel(ii,2))-gtrm2*diffmx
    dyk(esel(ii,2))=dyk(esel(ii,2))-gtrm2*diffmy
    dzk(esel(ii,2))=dzk(esel(ii,2))-gtrm2*diffmz

  endif

enddo

call timer_stop(T_EVB_Forces)

! Output state average dipole moments

if ((iolev.ge.0).and.(evbu.gt.0)) then

  if (header_flag) then
    write(evbu,'(a)')'important states: index, c**2, energy, x,y,z &
      & dipole vectors >>> state averaged energy, x,y,z dipole vectors'
    header_flag=.false.
  end if

  xdip=0.0
  ydip=0.0
  zdip=0.0

  do i=1,nensem
  
    csqd=smevec(i)*smevec(i)
  
    if(csqd .ge. 0.00001) then
  
      write(evbu,'(i3,1x,f12.5$,1x,f12.5$,1x,f12.5$,1x,f12.5$,1x,f12.5$)') &
        &i,csqd,ensh(i)+expoff(i),rcdxdip(i),rcdydip(i),rcdzdip(i)
  
      xdip = xdip + csqd*rcdxdip(i)
      ydip = ydip + csqd*rcdydip(i)
      zdip = zdip + csqd*rcdzdip(i)

    endif

  enddo

  write(evbu,'(3x,a,1x,f12.5$,1x,f12.5$,1x,f12.5$,1x,f12.5$)')'>>>',&
    &hk,xdip,ydip,zdip
  write(evbu,'(a)'),' '

endif

call timer_stop(T_EVB)

end subroutine evb

#endif

end module evb_mod

