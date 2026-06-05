module grapemod
  !********************************************************************
  !     As of November 2008 this code is hardcoded for GPU
  !     using shift cutoff method for both electrostatic
  !     and VDW interactions. It works for both periodic and non periodic.
  !     In case of periodic it works best with ewald, but maybe
  !     non-ewald is also OK (not tested, but programmed)
  !     1-4 parameter tricks used in CHARMM should now be OK!
  !********************************************************************
#if KEY_GRAPE==1 /*mdgrape*/

  use chm_kinds
  implicit none

  integer*4, allocatable, dimension(:),target :: numex,natex,iacsor
  integer*4, allocatable, dimension(:) :: bckptr
  real(chm_real),allocatable,dimension(:,:),target :: posgrp,rslgrp
  real(chm_real),allocatable,dimension(:,:) :: singrp,cosgrp
  real(chm_real),allocatable,dimension(:) :: ksdx,ksdy,ksdz,rslgrpot
  real(chm_real),allocatable,dimension(:),target :: cgloc
  integer,allocatable,dimension(:) :: gpuinblo
  ! how many exclusions per atom, including image atoms:
  integer, parameter :: lenexlist = 20 ! 10 maybe not enough?

  ! pointers for single parameter in the calls to MR3 routines
  real(chm_real), pointer, dimension(:,:) :: pgpuposgrp
  real(chm_real), pointer, dimension(:) :: pgpucg
  integer,pointer, dimension(:) :: pgpuiacsor,pgpunatex,pgpunumex
  !
  ! absolute position of the last element in the exclusion list
  ! per each real atom (not image)
  integer, allocatable, dimension(:) :: numexa
  !
  ! flags to decide when to execute the GRAPE/GPU code
  !
  logical :: qgpuene=.true.
  !
  real(chm_real) ene_c, ene_v, vdw_im, ele_im
  !
  ! Flags for various parallel strategies: block, interleaving, spacial
  ! This is cleaner then using pref.dat keywords!
  ! They are defined in the gpualloc()
  ! gpualloc() does the restructuring of atoms/gpu info
  logical :: qpargblock, qpargintrlv, qpargspac
  !
  ! split calculations stuff
  integer :: igpungr, igpungk, igpunch
  logical :: qgpurs, qgpuks
  !
  integer, allocatable, dimension(:) :: gpumap,mapgpu
#if KEY_PARALLEL==1
  integer*4, allocatable, dimension(:),target :: piacsor,numexj
  integer*4, allocatable, dimension(:),target :: natexj,numexp,numexpd
  integer*4, allocatable, dimension(:),target :: natexp,natexpd
  integer, allocatable, dimension(:) :: numexap,invmap_interleave,map_interleave
  real(chm_real),allocatable,dimension(:),target :: pcggrp
  real(chm_real),allocatable,dimension(:,:),target :: pposgrp
#endif 
  !
  ! fmm stuff
  integer :: save_igrape
  logical :: fmm_export_not_yet = .true.
  !
contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  SUBROUTINE ENBGRAP(NPRTCL,LELECX,X,Y,Z, DX,DY,DZ,ENB,EEL,INB14,IBLO14, &
#if KEY_FLUCQ==1
       QFLUC,FQCFOR,   & 
#endif
       LUSED)
    !----------------------------------------------------------------------
    !     This is the version of the nonboned energy terms for the
    !     GRAPE (GRAvity PipE) machine
    !
    !     For testing purposes it includes also no cutoff method for
    !     CHARMM without nonbond list (NCNLNB).
    !
    !     Compile this code with the GRAPE in pref.dat.
    !
    !     Add "grape <int>" to the nonbond specifications in the energy or dynamics
    !     commands. <int> can be:
    !                -1   no MDGRAPE in use
    !                 0   normal use
    !                 1   use fast library routines
    !                 2   host emulation (for control)
    !                 3   for pressure, include image atoms from charmm 
    !               +10   perform ewald sum on the GPU (add 10 to the above)
    !
    !     Subroutines in this file:
    !
    !     NCNLNB - does the straight nonbond calculation between all
    !              the atom pairs without the list
    !
    !     NCCORR - corrects the nonbond energies and forces for the
    !              pairs included in the exclusion list
    !              It had a variety of modes of operation but they
    !              are now done in the GRAPE library! Cleaned in June 2006
    !
    !     VDWGRAP - prepares, sends, and gets the data from/to GRAPE for
    !               Van der Waals interactions
    !
    !     EWGRAP -  prepares, sends, and gets the data from/to GRAPE for Ewald.
    !
    !     ELJGRP -  non-periodic energy and forces
    !
    !     GRAPEINI  initialize the board
    !
    !     GRAPEFIN  release the board
    !
    !     HEWEXC -  calculates Ewald elelctrostatic exclusion
    !
    !     BONDGRAP - interface between MDGRAPE and CHARMM parameter structure
    !                plus some bonding info. Currently slow: needs to be
    !                incorporated into parameter routines directly for efficiency
    !
    !     
    !     April   9, 1998   Milan Hodoscek
    !     August 25, 1998   Milan Hodoscek - added Ewald code
    !     June   20, 2001   Milan Hodoscek - added support for GRAPE-II
    !     June   12, 2006   Milan Hodoscek, Tetsu Narumi, Makoto Taiji
    !                       - added support for MDGRAPE-3
    !                       - lot of cleanup (for older stuff look at versions of
    !                                             CHARMM before and including c33a2)
    !
    !     November   2008   - Support for GPU: CUDA library
    !                         by Tetsu Narumi, Ryuji Sakamaki, and Milan Hodoscek
    !
    !     November   2011   - Code reorganization for parallel
    !                         by Tetsu Narumi, Milan Hodoscek
    !
    !     November   2012   - Support for switching cuttof methods
    !                       - Initial FMM support added
    !                         by Tetsu Narumi, Rio Yokota, Milan Hodoscek
    !
    !     October    2014   - keyword cleanups, for older code see c40a1 or older
    !                         Now GRAPE is for all the code, CUDAPME & MDGRAPE3 are no longer
    !
    !     November   2015   - Added new EXAFMM interface
    !                         by Rio Yokota & Milan Hodoscek
    !
    !----------------------------------------------------------------------
    !
    !
    use number
    use dimens_fcm
    use psf
    use param
    use inbnd,only: eps,lelec,lvdw,lcons,lshft,lvshft,lfswt,lvfswt
    use consta
    use exfunc
    use stream
    use grape
    use image
    use ewald
    use energym
    use contrl
    use machutil
    !
    LOGICAL LELECX,LUSED
    INTEGER NPRTCL,INB14(*),IBLO14(*)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) DX(*),DY(*),DZ(*),ENB,EEL
    !
#if KEY_FLUCQ==1
    LOGICAL QFLUC    
    real(chm_real) FQCFOR(*) 
#endif
    !
    real(chm_real) CGF
    real(chm_real) telaps,tstart
    INTEGER IOFF(MAXATC),I,J,MODE,IRET
    LOGICAL QENEGRAP
    logical cons_ele_shift,cons_vdw_shift,cons_ele_switch,cons_vdw_switch

    !
    tstart = eclock()
    !
#if KEY_FLUCQ==1
    IF (QFLUC) THEN
       ! No FlucQ implementation for GRAPE
       LUSED=.FALSE.
       RETURN
    ENDIF
#endif 
    LUSED=.TRUE.
    J=0
    DO I=1,MAXATC
       IOFF(I)=J
       J=J+I
    ENDDO
    !
    CGF=ZERO
    IF(LELECX.AND.(EPS /= ZERO)) CGF=CCELEC/EPS
    !
    !     Nocutoff (NOLIst): used for GPU accuracy checks
    !
    IF (LNOCUT) THEN
       CALL NCNLNB(X,Y,Z,CGF,DX,DY,DZ,ENB,EEL,IOFF)
       CALL NCCORR(X,Y,Z,CGF,DX,DY,DZ,ENB,EEL, &
            INB14,IBLO14,IOFF)
       RETURN
    ENDIF
    !******************
    !*    GRAPE/GPU:  *
    !******************
    !
    ! Cut-off method checking:
    ! constant eps shift & switch are the only one working currently
    ! no force method yet
    !
    cons_ele_shift  = lcons .and. lshft         .and. (.not.lfswt)
    cons_vdw_shift  = lcons .and. lvshft        .and. (.not.lvfswt)
    cons_ele_switch = lcons .and. (.not.lshft)  .and. (.not.lfswt)
    cons_vdw_switch = lcons .and. (.not.lvshft) .and. (.not.lvfswt)
    ! ....
    ! more can be specified similarly when supported
    if((.not.(cons_ele_shift.or.cons_ele_switch)).and.lelec) then
    !if((.not.cons_ele_shift).and.lelec) then
       CALL WRNDIE(-5,'<ENBGRAP>','Unsupported cutoff method for electrostatics')
       return
    endif
    if((.not.(cons_vdw_shift.or.cons_vdw_switch)).and.lvdw) then
    !if((.not.cons_vdw_shift).and.lvdw) then
       CALL WRNDIE(-5,'<ENBGRAP>','Unsupported cutoff method for VDW')
       return
    endif
    !
    QGRIM=.FALSE.
    IF(NTRANS > 0) THEN
       QGRIM=.TRUE.
       ! This routine is called from 2 places:
       ! image routines + standard routines
       ! qgpuene prevents extra calculations:
       if(.not.qgpuene) return        ! we already did everything
                                      ! when it was called from energy()
    ENDIF
    !
    QENEGRAP=.FALSE.
    !
    !
    !     mdstep counting is in dynamc.src (we cannot do it here)
    !     there maybe an energy call or something in the input script
    !     which modifies the counter
    IF(MDSTEP == 0)QENEGRAP=.TRUE.
    IF(MOD(MDSTEP+1,NPRINT) == 0)QENEGRAP=.TRUE.
    !
    !
    ! allocate some memory for global vars. Change whenever update is called
    call gpualloc
    !
    IF(QGRIM.or.lfmm) THEN
       CALL EWGRAP(X,Y,Z,CGF,DX,DY,DZ,EEL,ENB,QENEGRAP,IOFF, &
#if KEY_FLUCQ==1
            QFLUC,FQCFOR,                                             & 
#endif
            INB14,IBLO14)
    ELSE
       CALL ELJGRP(X,Y,Z,CGF,DX,DY,DZ,EEL,ENB,QENEGRAP)
    ENDIF
    !
    call gpudealloc
    !
    telaps=eclock()-tstart
    mdgelj=mdgelj+telaps
    !
    RETURN
  END SUBROUTINE ENBGRAP
  !
  subroutine gpualloc

    use grape
    use parallel
    use psf
    use image
    use coord
    use stream
    use memory
    use bases_fcm, only: bimag

    integer npart,ipt,jpt,kpt,iorig,iend,istart,i,j,k,m
    integer iatom
    real(chm_real) :: xmin,xmax,xbox,xlow,xup
    real(chm_real) :: ymin,ymax,ybox,ylow,yup
    real(chm_real) :: zmin,zmax,zbox,zlow,zup
    integer :: bx,by,bz
    integer, allocatable,dimension(:) :: nx,ny,nz
    integer,parameter :: maxgpus = 16
    !
    ! we need this routine only when update was called
    ! and before the first energy calculation!
    ! qgpupd is controlled by upimag(), should be in the calling routine!!!
    !
    ! control of the parallel method used in the GPU.
    ! The choice here is only useful for developing the code
    ! Users don't need to change this, hopefully... (compile time)
    ! In case we use no parallel, these need to be all false
#if KEY_PARALLEL==0
    qpargblock  = .false.
    qpargintrlv = .false.
    qpargspac   = .false.
#else /**/
    qpargblock  = .false.
    qpargintrlv = .false.     ! now we use interleave
    qpargspac   = .true.      ! now we use spacial decomposition
    if(lfmm) qpargspac   = .false. ! temp
#endif 

    !
    ! how many particles we have in the system
    npart=natom
    if(mod(igrape,10) == 3 ) npart=natim
    !
    if(natim==0)npart=natom
    !write(outu,'(a,3(2x,i0))')'gpualloc>npart,natom=',npart,natom
    call chmalloc('grape.src','gpualloc','posgrp',3,npart,crl=posgrp)
    call chmalloc('grape.src','gpualloc','rslgrp',3,npart,crl=rslgrp)
    call chmalloc('grape.src','gpualloc','rslgrpot',npart,crl=rslgrpot)
    call chmalloc('grape.src','gpualloc','cgloc',npart,crl=cgloc)
#if KEY_PARALLEL==1
    call chmalloc('grape.src','gpualloc','pposgrp',3,npart,crl=pposgrp)
    call chmalloc('grape.src','gpualloc','pcggrp',npart,crl=pcggrp)
#endif 
    !
    !write(*,*)'gpualloc>qgpupd=',qgpupd
    !
    ! we should return here because of the performance issues!
    if(.not.qgpupd)return

    if(qgpupd)then
       qgpupd=.false.
    endif

    ! sending new image atom info to GPU.
    ! image stuff for pressure. Let's keep the old GPU stuff (faster) as
    ! igrape = 11. Use igrape = 13 for the pressure stuff.

    !write(outu,'(a,3(2x,i0))')'gpualloc>natom,natim,ntrans=',natom,natim,ntrans
    if(mod(igrape,10) == 3 ) then
       ! extend natex list for image atoms, just a copy of the real stuff
#if KEY_PARALLEL==1
       ! different for parallel:
       ipt=numexap(natom)
       do i = natom+1, natim
          iorig = bimag%imattr(i)
          iacsor(i) = iacsor(iorig) ! piacsor comes later...
          numexp(i) = numexp(iorig)
          iend = numexap(iorig)
          istart = 1
          if (iorig > 1) istart = numexap(iorig-1) + 1
          do j = istart, iend
             ipt=ipt+1
             if(ipt.gt.2*natim*lenexlist) &
                  CALL WRNDIE(-5,'<gpualloc>','NATEX index overflow.')
             natexp(ipt) = natexp(j)
          enddo
       enddo
#else /**/
       ipt=numexa(natom)

       do i = natom+1, natim
          iorig = bimag%imattr(i)
          numex(i) = numex(iorig)
          iacsor(i) = iacsor(iorig)
          iend = numexa(iorig)
          istart = 1
          if (iorig > 1) istart = numexa(iorig-1) + 1
          do j = istart, iend
             ipt=ipt+1
             if(ipt.gt.2*natim*lenexlist) &
                  CALL WRNDIE(-5,'<gpualloc>','NATEX index overflow.')
             natex(ipt) = natex(j)
          enddo
       enddo
#endif 
    endif
    ! distribute the numexp, and natexp here, then it is good for grape 11? or 13
#if KEY_PARALLEL==1 /*distr_natexp*/
    ! if needed then setup a different parallel scheme.
    ! the check is inside the routine!
    call cpugpusplitset()
    ! In some of the splited parts this code does not need to be executed
    ! problem - FIXME
    !if (.not.qgpurs) then
    !   call cpugpusplitset()
    !   return
    !endif
    !
    if(allocated(gpumap)) deallocate(gpumap)
    if(allocated(mapgpu)) deallocate(mapgpu)
    call chmalloc('grape.src','gpualloc','gpumap',npart,intg=gpumap)
    call chmalloc('grape.src','gpualloc','mapgpu',npart,intg=mapgpu)
    gpumap(1:npart)=0
    mapgpu(1:npart)=0
    !
    ! Spacial decomposition
    !
    !write(*,*)'GPUALLOC>qpargspac=',qpargspac
    If (qpargspac) then
       ! we use nx,ny,nz to divide the space
       call chmalloc('grape.src','gpualloc','nx',maxgpus,intg=nx)
       call chmalloc('grape.src','gpualloc','ny',maxgpus,intg=ny)
       call chmalloc('grape.src','gpualloc','nz',maxgpus,intg=nz)
       ! we do the fixed division here. For now up to numnod=16
       ! some of these are bad for load balance
       ! I guess there is a way to produce this on the fly !!???
       ! + add the dynamic load balance
       nx=[1,2,3,2,5,3,7,2,3,5,11,4,13,7,5,4]
       ny=[1,1,1,2,1,2,1,2,3,2,1,3,1,2,3,2]
       nz=[1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2]
       ! It would be better to have a general formula for the above!!
       ! Should be easy to setup. Note that the above is usually bad
       ! unless it is 1,2,4, and 8 where it is perfect if the system
       ! is isotropically dense: the space is always symmetrically devided,
       ! ie the ratio of th real/image atoms is the same for every sub-box.
       !
       ! For now we need to protect for the limited size of this table:
       if (numnod > maxgpus) &
            call wrndie(-5,'<GPUALLOC>','Currently only 16 GPUs supported')
       ! now find wich box this process should work on (maybe not ideal code?)
       do i=0,nx(numnod)-1
          do j=0,ny(numnod)-1
             do k=0,nz(numnod)-1
                m=i+j*nx(numnod)+k*nx(numnod)*ny(numnod)
                if(mynod == m) then
                   bx=i
                   by=j
                   bz=k
                endif
             enddo
          enddo
       enddo
       !
       ! not sure which is better here: npart or natom ???
       ! in some cases it doesn't matter (eg, numnod=4)
       ! in the case of natom we must bring corresponding
       ! image atoms in the right GPU/process
       xmin=minval(x(1:npart))
       xmax=maxval(x(1:npart))+0.1 ! border extended in one direction
       ymin=minval(y(1:npart))
       ymax=maxval(y(1:npart))+0.1 ! border extended in one direction
       zmin=minval(z(1:npart))
       zmax=maxval(z(1:npart))+0.1 ! border extended in one direction
       xbox=(xmax-xmin)/nx(numnod)
       ybox=(ymax-ymin)/ny(numnod)
       zbox=(zmax-zmin)/nz(numnod)
       xlow = xmin + bx*xbox
       ylow = ymin + by*ybox
       zlow = zmin + bz*zbox
       xup = xlow + xbox
       if (bx == (nx(numnod)-1)) xup = xmax ! take the buffer zone, too
       yup = ylow + ybox
       if (by == (ny(numnod)-1)) yup = ymax
       zup = zlow + zbox
       if (bz == (nz(numnod)-1)) zup = zmax
       ipt=0   ! all particles
       kpt=0   ! image particles only: for load balance report
       do i = 1, npart
          if((xlow <= x(i)).and.(x(i) < xup) .and. &
               (ylow <= y(i)).and.(y(i) < yup) .and. &
               (zlow <= z(i)).and.(z(i) < zup) ) then
             ipt=ipt+1
             gpumap(i)=ipt
             mapgpu(ipt)=i
             if(i>natom)kpt=kpt+1
          endif
       enddo
       ! do some checking for load balance and total
       !write(outu,'(a,i0,a,i0,a,i0,a,i0,a,f7.2)')'the process ',mynod, ' has ',ipt, &
       !     ' atoms of which there are ',kpt,' image atoms, out of total ',npart, &
       !     ' percentage: ',100.0*ipt/npart
       write(outu,'(a,i0,a,i0,a,i0,a,i0,a,f7.2,a)')'GPU ',mynod, ' has ',ipt, &
            ' atoms of which there are ',kpt,' image atoms, out of total ',npart, &
            ' :',100.0*ipt/npart,' %'
       ! prepare natexpd: this is actually good for any of the parallel method ???
       ! so it should go out of this if-block ???
       jpt=0
       do i = 1, ipt
          iatom=mapgpu(i)
          istart=1
          if(iatom>1)istart=numexap(iatom-1)+1
          kpt=0
          do j=istart,numexap(iatom)
             jpt=jpt+1
             natexpd(jpt)=natexp(j)
             kpt=kpt+1
          enddo
          numexpd(i)=kpt
       enddo
       !call print_natex(50+mynod,npart,numexp,natexp)
       !call print_natex(60+mynod,ipt,numexpd,natexpd,mapgpu)
       call chmdealloc('grape.src','gpualloc','nx',maxgpus,intg=nx)
       call chmdealloc('grape.src','gpualloc','ny',maxgpus,intg=ny)
       call chmdealloc('grape.src','gpualloc','nz',maxgpus,intg=nz)
    endif   !qpargspac
    !
    ! Interleaving (do i = mynodp,npart,numnod)
    !
    if (qpargintrlv) then
       ! we need to resize this array here FIXME: get rid of these two! gpumap,mapgpu are the same
       if(allocated(invmap_interleave)) &
            deallocate(invmap_interleave,map_interleave)
       call chmalloc('grape.src','gpualloc','invmap_interleave',npart,intg=invmap_interleave)
       call chmalloc('grape.src','gpualloc','map_interleave',npart,intg=map_interleave)
       !
       ipt=0
       invmap_interleave(1:npart)=-1
       do i = mynodp, npart, numnod
          ipt=ipt+1
          gpumap(i)=ipt             ! to use in X,DX,... copy
          mapgpu(ipt)=i             ! reverse gpu map (for image atoms)
                                    ! actually this is the only one really needed :-)
                                    ! but the code uses gpumap()
          map_interleave(ipt)=i     ! to fillup DX arrays
          invmap_interleave(i)=ipt  ! to reduce natexp
       enddo
       jpt=0
       do i = 1, ipt
          iatom=map_interleave(i)
          istart=1
          if(iatom>1)istart=numexap(iatom-1)+1
          kpt=0
          do j=istart,numexap(iatom)
             !          if(invmap_interleave(natexp(j)) > 0) then
             jpt=jpt+1
             natexpd(jpt)=natexp(j)
             kpt=kpt+1
             !          endif
          enddo
          numexpd(i)=kpt
       enddo
       !call print_natex(50+mynod,npart,numexp,natexp)
       !call print_natex(60+mynod,ipt,numexpd,natexpd,map_interleave)
    endif
    ! we can restore the parallel setup here...
    call cpugpusplitclear()
    !
#endif /* (distr_natexp)*/

    return
  end subroutine gpualloc

  subroutine cpugpusplitset()
    use parallel
    use grape
    use stream
    integer :: i
    ! this is for general parallel split for some of the 
    ! energy/force term calculations in a separate set of
    ! processors or GPUs
    ! We get the data for this from PARA GPUS input script command
    if(qgpusplit) then
       qgpurs = .false.
       qgpuks = .false.
    else
       qgpurs = .true.
       qgpuks = .true.
    endif
    if (.not.qgpusplit) return
#if KEY_PARALLEL==1
    !
    ! First set is for GPUs
    if (mynodg < igpungr) then
       numnod=igpungr
       ! no need for mynod=mynodg
       do i=0,numnod
          ippmap(i)=i
       enddo
       qgpurs = .true.
    endif
    ! GPU kspace can only use one GPU
    if ( (igpungk == 1) .and. (mynodg == igpungr) ) then
       mynod = 0
       numnod = 1
       do i=0,numnod
          ippmap(i)=i+igpungr
       enddo
       qgpuks = .true.
    endif
    ! The rest is for CPU host calculations only
    if ( (igpunch > 0) .and. (mynodg >= igpungr+igpungk) ) then
       mynod=mynodg-igpungr-igpungk
       numnod=numnodg-igpungr-igpungk
       do i=0,numnod
          ippmap(i)=i+igpungr+igpungk
       enddo
       ! we do this on the host, is this good for any case???
       qgpuks=.true.
    endif
    ! the same for every case
    mynodp=mynod+1
    noddim=nptwo()
    do i = 1,maxnode
       inode(i)=mod(i,numnod)
    enddo
    ! we are after update, so we need new iparpt here
    !jnb and inblo are not used in parupdate - just dummy parameters
    !
    call parupdate([0],[0])
    !
    ! checking printouts:
    !write(outu,'(a,i0,a,i0,a,i0,a,i0)')'CPU/GPU split>mynodg ',mynodg, &
    !     ' numnodg ',numnodg,' mynod ',mynod,' numnod ', numnod
    !write(outu,'(a,i0,a,i0,a,i0)')'CPU/GPU split>mynodg ',mynodg, &
    !     ' mynod ', mynod,' ippmap(mynod) = ', ippmap(mynod)
#endif 
    return
  end subroutine cpugpusplitset

  subroutine cpugpusplitclear()
    use parallel
    use grape
    ! it makes a standard run for parallel CHARMM 
    integer:: i
    if(.not.qgpusplit) return
#if KEY_PARALLEL==1
    MYNOD=MYNODG
    NUMNOD=NUMNODG
    MYNODP=MYNOD+1
    NODDIM=NPTWO()
    CALL CUBE(MYNOD,NUMNOD,IPPMAP)
    !
    !     Lets do the INODE array, too:
    !
    DO I=1,MAXNODE
       INODE(I)=MOD(I,NUMNOD)
    ENDDO
    ! we are after update, so we need new iparpt here
    !jnb and inblo are not used in parupdate - just dummy parameters
    call parupdate([0],[0])
#endif 
    !
    return
  end subroutine cpugpusplitclear

  subroutine print_natex(unit,n,num,atom,map)
    integer unit,n,num(:),atom(:)
    integer,optional::map(:)
    integer i,ist,l
    ist=0
    do i=1,n
       ist=ist+num(i)
    enddo
    write(unit,'(a,3i8)')'number of pairs=',ist
    ist=0
    do i=1,n
       l=i
       if(present(map))l=map(i)
       write(unit,'(a,3i8)')'i,atom,num(i)=',i,l,num(i)
       write(unit,'(10i8)')atom(ist+1:ist+num(i))
       ist=ist+num(i)
    enddo
    return
  end subroutine print_natex

  subroutine gpudealloc

    use psf
    use image
    use parallel
    use grape
    use memory
    use stream

    integer npart

    npart = natom
    if(mod(igrape,10) == 3 ) npart=natim
    if(natim==0) npart = natom
    !write(outu,'(a,3(2x,i0))')'gpudealloc>npart=',npart
    call chmdealloc('grape.src','gpudealloc','posgrp',3,npart,crl=posgrp)
    call chmdealloc('grape.src','gpudealloc','rslgrp',3,npart,crl=rslgrp)
    call chmdealloc('grape.src','gpudealloc','rslgrpot',npart,crl=rslgrpot)
    call chmdealloc('grape.src','gpudealloc','cgloc',npart,crl=cgloc)
#if KEY_PARALLEL==1
    call chmdealloc('grape.src','gpudealloc','pposgrp',3,npart,crl=pposgrp)
    call chmdealloc('grape.src','gpudealloc','pcggrp',npart,crl=pcggrp)
    ! maybe put gpumap,mapgpu also here ???
#endif 

    return
  end subroutine gpudealloc

  SUBROUTINE GRAPEINI(BNBND)
    !----------------------------------------------------------------------
    !
    !     Initialize GRAPE
    !
    use chm_kinds
    use chm_types
    use exfunc
    !
    use dimens_fcm
    use number
    use inbnd
    use psf
    use param
    use code
    use grape
    use image
    use memory
    use contrl
    use parallel
    use stream
    use ctitla, only: ntitla,titlea
    use chutil, only: atomid
    use coord
    implicit none
    !
    type(nonbondDataStructure) BNBND
    INTEGER IRET,I
    !
#if 1
    CHARACTER(len=100) mdm_table_ver
    CHARACTER(len=100) lj, ljpot, grav, gravpot
    CHARACTER(len=100) real, realpot
    INTEGER*4 n_table
    INTEGER*4 n_lj, n_ljpot, n_grav, n_gravpot
    INTEGER*4 n_real, n_realpot,izer,ione
    integer*4 tgrav,tgravpot,tlj,tljpot,treal,trealpot
#endif 
    CHARACTER(len=8) SID,RID,REN,AC
    integer ires
    integer fmmimag,ista,iend,fmmverbose
    real(chm_real) fmmtheta
    integer, allocatable, dimension(:) :: islct,icntrl
    integer ipar,jpar,kpar,lpar,ii,jj,kk,ll,ic
    real(chm_real), allocatable, dimension(:) :: emass
    real(chm_real), allocatable, dimension(:,:) :: cbond,rbond
    real(chm_real), allocatable, dimension(:,:,:) :: cangle,aangle
    !
    write(*,*)'GRAPEINI>entering...'
    IF(.NOT.QGRAPINI)THEN
       ! fmm igrape trickery
       save_igrape=igrape
       if(lfmm) igrape=3
       qgpupd=.true.
#if 1
       CALL MR3init
#else /**/
       CALL MR1init
#endif 
       !write(outu,*)'grapeini>ntrans,igrape,natom=',ntrans,igrape,natom
       IF(NTRANS > 0)THEN
!          call chmalloc('grape.src','grapeini','cosgrp',3,natom,crl=cosgrp)
          call chmalloc('grape.src','grapeini','singrp',3,natom,crl=singrp)
          if(mod(igrape,10) == 3 ) then
             ! for parallel we need to send kspace to the last CPU /already coded/
             ! but then we need to ignore that one!!!
             ! But in this case we need the forces&energy from KSPACE via send/receive!
             ! Also we need to take care of parallel/parallel -> introduce new
             ! global variables gpunumnod,gpumynod???
             ! or do the command in the beginning of the run; before repd
             ! something like: paral split 8 1
             ! so do something like this:
#if 0 /* prll-dothislater==1 */
             !if(numnodg > 1) numnod = numnodg-1     
#endif
             ! above is not enough, we need for the last one to specify
             ! if(mynodg == (numnodg-1) {numnod=1 ; mynod=0}
             !write(outu,*)'grapeini>allocate ksdx,...'
             ! PROBLEM: call chmalloc('grape.src','grapeini','gpuinblo',4*natom,intg=gpuinblo)
             ! we use factor 2 for updates that can have more image atoms than initially
             call chmalloc('grape.src','grapeini','gpuinblo',2*natim,intg=gpuinblo)
             call chmalloc('grape.src','grapeini','ksdx',natom,crl=ksdx)
             call chmalloc('grape.src','grapeini','ksdy',natom,crl=ksdy)
             call chmalloc('grape.src','grapeini','ksdz',natom,crl=ksdz)
          endif
       ENDIF
    ENDIF
    QGRAPINI=.true.
    MDSTEP=0
#if 1
    call getenv('MDM_TABLE_VER', mdm_table_ver)

    n_table = index(mdm_table_ver, ' ') - 1
    if (n_table  ==  0) then
       mdm_table_ver = '00'
       n_table = 2
    end if

    lj = 'lj'//mdm_table_ver(1:2)
    ljpot = 'ljpot'//mdm_table_ver(1:2)
    grav = 'grav'//mdm_table_ver(1:2)
    gravpot = 'gravpot'//mdm_table_ver(1:2)
    real = 'real'//mdm_table_ver(1:2)
    realpot = 'realpot'//mdm_table_ver(1:2)

    n_lj = index(lj,' ') - 1
    n_ljpot = index(ljpot,' ') - 1
    n_grav = index(grav,' ') - 1
    n_gravpot = index(gravpot,' ') - 1
    n_real = index(real,' ') - 1
    n_realpot = index(realpot,' ') - 1

    tlj=2
    tljpot=3
    tgrav=0
    tgravpot=1
    treal=6
    trealpot=7
    izer=0
    ione=1
    call MR3SetTable_nf(lj,     tlj, izer, n_lj)
    call MR3SetTable_nf(ljpot,  tljpot, ione, n_ljpot)
    call MR3SetTable_nf(grav,   tgrav, izer, n_grav)
    call MR3SetTable_nf(gravpot,tgravpot, ione, n_gravpot)
    call MR3SetTable_nf(real,   treal, izer, n_real)
    call MR3SetTable_nf(realpot,trealpot, ione, n_realpot)
#else /**/
    n_grav=4
    n_gravpot=7
    n_lj=2
    N_ljpot=5
    n_real=4
    n_realpot=7
    tgrav=500
    tgravpot=501
    tlj=502
    tljpot=503
    treal=506
    trealpot=507
    izer=0
    ione=1
    call MR1SetTable_emu_nf('grav',tgrav,izer,n_grav)
    call MR1SetTable_emu_nf('gravpot',tgravpot,ione,n_gravpot)
    call MR1SetTable_emu_nf('lj',tlj,izer,n_lj)
    call MR1SetTable_emu_nf('ljpot',tljpot,ione,n_ljpot)
    call MR1SetTable_emu_nf('real',treal,izer,n_real)
    call MR1SetTable_emu_nf('realpot',trealpot,ione,n_realpot)
#endif 
    !
    !
    !     Intialize internal timings
    mdgini=zero
    mdgelj=zero
    mdglj=zero
    mdgrs=zero
    mdgks=zero
    mdgex=zero
    mdg14=zero
    !
    !     This routine fills natex and numex arrays from INB14, IBLO14
    !     It also prepares non-bond parameters for MRXnnn library
    CALL BONDGRAP(BNBND%INB14,BNBND%IBLO14)
    ! space for image atom mapping array
    qgpupd = .true.   ! we can always perform update in the beginning
    !
    if (lfmm) then
       !write(*,*)'GRAPEINI>entering FMM code'
       ! level of periodic tree in FMM
       fmmimag=3     ! =0 -> no images
       fmmtheta=0.4   ! theta: small is slow
       fmmverbose=0   ! time profiling: 0=no, 1=yes
       call fmm_init(fmmimag,fmmtheta,fmmverbose)
       if(allocated(fmmcpu))deallocate(fmmcpu)
       call chmalloc('grape.src','GRAPEINI','fmmcpu',natom,intg=fmmcpu)
       ista=1
       iend=natom
       call split_range(ista, iend, mynod, numnod)
       fmmcpu(1:natom)=0 ! needs to be initialized ???
       fmmcpu(ista:iend)=1 ! no overlap!!
       ! export the data for test_charmm.f90 in exafmm-dev/wrappers
       ! box sizes
       ! coordinates
       ! numex, natex
       ! vdw parameters:rscale==frscale,gscale,fgscale,iacsor
       ! some flags trickery
       export_tofmm: if (fmm_export_not_yet.and.(save_igrape == 7)) then
          fmm_export_not_yet=.false.
          write(*,*)'GRAPEINI>starting write: coor, psf, param'
          call chmalloc('grape.src','grapeini','islct',natom,intg=islct)
          call chmalloc('grape.src','grapeini','icntrl',20,intg=icntrl)
          islct(1:natom)=1
          write(7,'(''* SIZE = '',3(f0.5,1x))')xucell(1),xucell(2),xucell(3)
          ! the following cannot be used for circular dependency problem?
          !call cwrite(7,titlea,ntitla,icntrl,x,y,z,cg, &
          !     res,atype,ibase,nres,natom,islct,2,0,0,.false.,-1)
          ! extract the EXT code from cwrite and put it here:
          call wrtitl(titlea,ntitla,7,0)
          write(7,'(i10,2x,a)') natom,'EXT'
          simplified_cwrite: do ires=1,nres
             do i=ibase(ires)+1,ibase(ires+1)
                call atomid(i,sid,rid,ren,ac)
                write(7,'(2I10,2X,A8,2X,A8,3F20.10,2X,A8,2X,A8,F20.10)') &
                     I,IRES,RES(IRES),ATYPE(I),X(I),Y(I),Z(I),SID,RID,cg(i)
             enddo
          enddo simplified_cwrite
          call chmdealloc('grape.src','grapeini','islct',natom,intg=islct)
          call chmdealloc('grape.src','grapeini','icntrl',20,intg=icntrl)
          write(7,'(a)')'NUMEX'
          write(7,'(10i10)')numexp(1:natom)
          write(7,'(a,2x,i0)')'NATEX',sum(numexp(1:natom))
          write(7,'(10i10)')natexp(1:sum(numexp(1:natom)))
          write(7,'(''RSCALE==FRSCALE'',2x,i0)')idist
          write(7,'(5f20.10)')rscale(1:idist**2)
          write(7,'(''GSCALE'',2x,i0)')idist
          write(7,'(5f20.10)')gscale(1:idist**2)
          write(7,'(''FGSCALE'',2x,i0)')idist
          write(7,'(5f20.10)')fgscale(1:idist**2)
          write(7,'(''IACSOR'')')
          write(7,'(20i5)')iacsor(1:natom)
          ! PSF stuff for dynamics
          write(7,'(''BONDS'',2x,i0)')nbond
          write(7,'(10i10)')(ib(i),jb(i),i=1,nbond)
          write(7,'(''ANGLES'',2x,i0)')ntheta
          write(7,'(9i10)')(it(i),jt(i),kt(i),i=1,ntheta)
          ! Parameters stuff for dynamics
          ! based on the same principles as for rscale,gscale, ie make
          ! 2, 3 and 4 dimensional matrices. Maybe for torsions this could be a problem???
          ! they are basically empty :-(
          call chmalloc('grape.src','grapeini','cbond',idist,idist,crl=cbond)
          call chmalloc('grape.src','grapeini','rbond',idist,idist,crl=rbond)
          call chmalloc('grape.src','grapeini','aangle',idist,idist,idist,crl=aangle)
          call chmalloc('grape.src','grapeini','cangle',idist,idist,idist,crl=cangle)
          call chmalloc('grape.src','grapeini','emass',idist,crl=emass)
          cbond=zero
          rbond=zero
          aangle=zero
          cangle=zero
          emass=zero
          ! these loops are not very efficient, but stil general and not so bad:
          ! they write the same value in the same place many times !??
          do i = 1, nbond
             ii=ib(i)
             jj=jb(i)
             ic=icb(i)
             if(ic==0) cycle
             ipar=iacsor(ii)
             jpar=iacsor(jj)
             cbond(ipar,jpar)=cbc(ic)
             rbond(ipar,jpar)=cbb(ic)
          enddo
          do i = 1, ntheta
             ii=it(i)
             jj=jt(i)
             kk=kt(i)
             ic=ict(i)
             if(ic==0) cycle
             ipar=iacsor(ii)
             jpar=iacsor(jj)
             kpar=iacsor(kk)
             cangle(ipar,jpar,kpar)=ctc(ic)
             aangle(ipar,jpar,kpar)=ctb(ic)
          enddo
          ! iacsor must be defined before this: call non grape 7 energy first
          do i=1, natom
             emass(iacsor(i))=amass(i)
          enddo
          write(7,'(''RBOND'',2x,i0)')idist
          write(7,'(5f20.10)')rbond(1:idist,1:idist)
          write(7,'(''CBOND'',2x,i0)')idist
          write(7,'(5f20.10)')cbond(1:idist,1:idist)
          write(7,'(''AANGLE'',2x,i0)')idist
          write(7,'(5f20.10)')aangle(1:idist,1:idist,1:idist)
          write(7,'(''CANGLE'',2x,i0)')idist
          write(7,'(5f20.10)')cangle(1:idist,1:idist,1:idist)
          write(7,'(''MASS'',2x,i0)')idist
          write(7,'(5f20.10)')emass(1:idist)
          call chmdealloc('grape.src','grapeini','cbond',idist,idist,crl=cbond)
          call chmdealloc('grape.src','grapeini','rbond',idist,idist,crl=rbond)
          call chmdealloc('grape.src','grapeini','aangle',idist,idist,idist,crl=aangle)
          call chmdealloc('grape.src','grapeini','cangle',idist,idist,idist,crl=cangle)
          call chmdealloc('grape.src','grapeini','emass',idist,crl=emass)
          ! we also want the velocities!!!
          ! get them in the restart code!
       endif export_tofmm
    endif
!
    RETURN
    contains
      subroutine split_range(ista, iend, isplit, nsplit)
      integer size, remainder,ista,iend,isplit,nsplit,increment
      size = iend - ista + 1
      increment = size / nsplit
      remainder = mod(size, nsplit)
      ista = ista + isplit * increment + min(isplit,remainder)
      iend = ista + increment - 1
      if(remainder.gt.isplit) iend = iend + 1
      return
    end subroutine split_range

  END SUBROUTINE GRAPEINI
  !
  SUBROUTINE GRAPEFIN
    !----------------------------------------------------------------------
    !
    use chm_kinds
    use exfunc
    !
    use dimens_fcm
    use psf
    use grape
    use image
    use memory
    use parallel
    use stream
    implicit none
    integer llen1,llen2
    !
    IF(QGRAPINI)THEN
#if 1
       Call MR3Free
#else /**/
       Call MR1Free
#endif 
       llen1 = natom
       llen1 = 2*natim  ! So that grape==11 works in parallel, too
       if (mod(igrape,10) == 3) llen1 = 2*natim
       if(natim==0)llen1=2*natom
       llen2 = llen1 * lenexlist
       IF(NTRANS > 0)THEN
          if(allocated(singrp)) then !.and.allocated(cosgrp)) then
             !call chmdealloc('grape.src','grapeini','cosgrp',3,10*natom,crl=cosgrp)
             !write(outu,*)'grapefin>deallocate singrp,...'
             call chmdealloc('grape.src','grapefin','singrp',3,natom,crl=singrp)
          endif
          if(mod(igrape,10) == 3 ) then
             ! Put back numnod since we are done with KSPACE split
             ! if(numnodg > numnod) numnod = numnodg  
             !write(outu,*)'grapefin>deallocate ksdx,...natom=',natom
             !write(outu,*)'grapefin>deallocate ksdx,...allocted ?=', &
             !     allocated(ksdx),allocated(ksdy),allocated(ksdz)
             call chmdealloc('grape.src','grapefin','ksdx',natom,crl=ksdx)
             call chmdealloc('grape.src','grapefin','ksdy',natom,crl=ksdy)
             call chmdealloc('grape.src','grapefin','ksdz',natom,crl=ksdz)
          endif
       ENDIF
    ENDIF
    QGRAPINI=.FALSE.
    !
    !write(outu,*)'grapefin>deallocate numex,...'
    if(allocated(numex).and.allocated(natex).and.allocated(iacsor)) then
       call chmdealloc('grape.src','grapefin','numex',llen1,ci4=numex)
       call chmdealloc('grape.src','grapefin','numexa',llen1,intg=numexa)
       call chmdealloc('grape.src','grapefin','iacsor',llen1,ci4=iacsor)
       call chmdealloc('grape.src','grapefin','natex',llen2,ci4=natex)
    endif
    !write(outu,*)'grapefin>deallocate bckptr,...'
    if(allocated(bckptr)) &
         call chmdealloc('grape.src','grapefin','bckptr',llen2,ci4=bckptr)
    !
#if KEY_PARALLEL==1
    if(allocated(numexp).and.allocated(natexp).and.allocated(piacsor)) then
       call chmdealloc('grape.src','grapefin','natexp',llen2,ci4=natexp)
       call chmdealloc('grape.src','grapefin','numexp',llen1,ci4=numexp)
       call chmdealloc('grape.src','grapefin','natexpd',llen2,ci4=natexpd)
       call chmdealloc('grape.src','grapefin','numexpd',llen1,ci4=numexpd)
       call chmdealloc('grape.src','grapefin','numexap',llen1,intg=numexap)
       call chmdealloc('grape.src','grapefin','piacsor',llen1,ci4=piacsor)
    endif
    if(allocated(numexj).and.allocated(natexj)) then
       call chmdealloc('grape.src','grapefin','natexj',llen2,ci4=natexj)
       call chmdealloc('grape.src','grapefin','numexj',llen1,ci4=numexj)
    endif
#endif 
    !
    IF(.NOT.LGRAPE)RETURN
    !     Print out the timings
#if KEY_PARALLEL==0
    write(outu,99)           
#endif
#if KEY_PARALLEL==1
    write(outu,97)mynod      
#endif
98  format(68('='))
99  format(20('='),' MDGRAPE timings in seconds ',20('='))
97  format(12('='),' MDGRAPE timings in seconds for process',i5,12('='))
    write(outu,100)
100 format('    Initial   Non-bond     VDW     R-Space', &
         '   K-Space     EwExc   nbadd14')
    write(outu,'(7f10.2)')mdgini,mdgelj,mdglj,mdgrs,mdgks,mdgex,mdg14
    write(outu,98)
    !
    RETURN
  END SUBROUTINE GRAPEFIN
  !
  SUBROUTINE NCNLNB(X,Y,Z,CGF,DX,DY,DZ,ENB,EEL,IOFF)
    !----------------------------------------------------------------------
    !
    !     This routine calculates the nonbond energy without any cutoffs.
    !     It also doesn't use any nonbond list. For efficiency reasons
    !     it includes also excluded atom pairs, which has to be corrected
    !     by calling NCCORR subroutine after this.
    !
    !     October 2008: introduce cutoff (plain cutoff; no shift or switch)
    !                   specify both nolist and cnolist
    !
    use chm_kinds
    use number
    use dimens_fcm
    use psf
    use param
    use parallel
    use grape
    !
    implicit none
    !C      IMPLICIT NONE
    !
    real(chm_real) X(*),Y(*),Z(*),CGF,EPS
    real(chm_real) DX(*),DY(*),DZ(*),ENB,EEL
    INTEGER IOFF(*)
    !
    INTEGER I,J,I1,J1,IACI,IC
    real(chm_real) XFGRAP,YFGRAP,ZFGRAP,XX,YY,ZZ,BD,BD2
    real(chm_real) XGRAPE,XGRAP3,XGRAP4,BEL
    real(chm_real) FGRAP,FGRAPX,FGRAPY,FGRAPZ,CT2NL
    !
    EEL=ZERO
    ENB=ZERO
    !
    ! This is nolist cutoff!
    !
    CT2NL = CTOFNBNL*CTOFNBNL
    !
#if KEY_PARALLEL==1
    DO I=MYNODP,NATOM-1,NUMNOD
#else /**/
    DO I=1,NATOM-1
#endif 
       I1=ITC(IAC(I))
       IACI=IOFF(I1)
       XFGRAP=ZERO
       YFGRAP=ZERO
       ZFGRAP=ZERO
       DO J=I+1,NATOM
          XX=X(J)-X(I)
          YY=Y(J)-Y(I)
          ZZ=Z(J)-Z(I)
          BD2=MAX(RSMALL,XX*XX+YY*YY+ZZ*ZZ)
          IF((BD2 <= CT2NL).OR.(.NOT.QCNOLIST))THEN
             BD=SQRT(BD2)
             J1=ITC(IAC(J))
             IF (I1 < J1) THEN
                IC=IOFF(J1)+I1
             ELSE
                IC=IACI+J1
             ENDIF
             XGRAPE=CNBA(IC)/BD2
             XGRAP3=XGRAPE*XGRAPE*XGRAPE
             ENB=ENB+CNBB(IC)*(XGRAP3*XGRAP3-XGRAP3-XGRAP3)
             XGRAP4=XGRAP3*XGRAPE
             BEL=CGF*CG(I)*CG(J)/BD2
             EEL=EEL+BEL*BD
             FGRAP=-TWELVE*CNBB(IC)/CNBA(IC)*(XGRAP4*XGRAP3-XGRAP4)
             FGRAP=FGRAP-BEL/BD
             FGRAPX=FGRAP*XX
             FGRAPY=FGRAP*YY
             FGRAPZ=FGRAP*ZZ
             DX(J)=DX(J)+FGRAPX
             DY(J)=DY(J)+FGRAPY
             DZ(J)=DZ(J)+FGRAPZ
             XFGRAP=XFGRAP+FGRAPX
             YFGRAP=YFGRAP+FGRAPY
             ZFGRAP=ZFGRAP+FGRAPZ
          ENDIF    ! QCNOLIST
       ENDDO
       DX(I)=DX(I)-XFGRAP
       DY(I)=DY(I)-YFGRAP
       DZ(I)=DZ(I)-ZFGRAP
    ENDDO
    !
    RETURN
  END SUBROUTINE NCNLNB
  !
  SUBROUTINE NCCORR(X,Y,Z,CGF,DX,DY,DZ,ENB,EEL,INB14,IBLO14,IOFF)
    !----------------------------------------------------------------------
    !     This subroutine calculates elelctrostatics and VDW for atom
    !     pairs in the exclusion list INB14. There are various modes.
    !
    use chm_kinds
    use exfunc
    use number
    use dimens_fcm
    use psf
    use param
    use inbnd, only: e14fac
    use consta
    use grape
    use ewald
    use parallel
    !
    implicit none
    !
    real(chm_real) X(*),Y(*),Z(*),CGF,DX(*),DY(*),DZ(*),ENB,EEL
    INTEGER INB14(*),IBLO14(*),IOFF(*)
    !
    real(chm_real) DERFC
    !
    INTEGER I,J,MM,K,L,NXI,NXIMAX,I1,J1,JJ,IC,IACI
    real(chm_real) DUMMY,XMAX,FMAX,SIGFAC
    real(chm_real) XX,YY,ZZ,BD,BD2,BEL,XGRAPE,XGRAP3,XGRAP4
    real(chm_real) FGRAP,XFGRAP,YFGRAP,ZFGRAP,GFX,GFY,GFZ,GEL,GVDW
    real(chm_real) ENE,DFRS,CH,AR,AR2,EFAC,EEB,EE14,CEVDW,CEVDW4
    real(chm_real) TVDW,CT2NL
    INTEGER IXVAL
    !
    SIGFAC=TWO**(ONE/THREE)
    ! 
    EEB=ZERO
    EE14=ZERO
    CEVDW=ZERO
    CEVDW4=ZERO
    !
    !     Nolist cutoff
    CT2NL = CTOFNBNL*CTOFNBNL
    !
#if KEY_PARALLEL==1
    DO I=MYNODP,NATOM,NUMNOD
#else /**/
    DO I=1,NATOM
#endif 
       IF(I > 1)THEN
          NXI=IBLO14(I-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=IBLO14(I)
       I1=ITC(IAC(I))
       IACI=IOFF(I1)
       IF(NXI > NXIMAX)GOTO 200
       XFGRAP=ZERO
       YFGRAP=ZERO
       ZFGRAP=ZERO
       DO K=NXI,NXIMAX
          JJ=INB14(K)
!!!!            write(*,'(a,3i7)')'nccorr>i,jj=',i,jj
          J=ABS(JJ)
          XX=X(J)-X(I)
          YY=Y(J)-Y(I)
          ZZ=Z(J)-Z(I)
          BD2=MAX(RSMALL,XX*XX+YY*YY+ZZ*ZZ)
          IF((BD2 <= CT2NL).OR.(.NOT.QCNOLIST))THEN
             BD=SQRT(BD2)
             J1=ITC(IAC(J))
             IF (I1 < J1) THEN
                IC=IOFF(J1)+I1
!!!               write(*,'(a,3i7)')'nccorr:i1<j1>i1,j1,ic=',i1,j1,ic
             ELSE
                IC=IACI+J1
!!!               write(*,'(a,3i7)')'nccorr:i1>=1>i1,j1,ic=',i1,j1,ic
             ENDIF
             !
             XGRAPE=CNBA(IC)/BD2
             XGRAP3=XGRAPE*XGRAPE*XGRAPE
             ENB=ENB-CNBB(IC)*(XGRAP3*XGRAP3-XGRAP3-XGRAP3)
             XGRAP4=XGRAP3*XGRAPE
             BEL=CGF*CG(I)*CG(J)/BD2
             EEL=EEL-BEL*BD
             FGRAP=-TWELVE*CNBB(IC)/CNBA(IC)*(XGRAP4*XGRAP3-XGRAP4)
             FGRAP=FGRAP-BEL/BD
             DX(J)=DX(J)-FGRAP*XX
             DY(J)=DY(J)-FGRAP*YY
             DZ(J)=DZ(J)-FGRAP*ZZ
             XFGRAP=XFGRAP-FGRAP*XX
             YFGRAP=YFGRAP-FGRAP*YY
             ZFGRAP=ZFGRAP-FGRAP*ZZ
             !
             !           After the correction check for 1-4 VDW:
             !           use CNBA(IC+MAXCN) to calculate correct number.
             !
             IF(JJ < 0)THEN
                XGRAPE=CNBA(IC+MAXCN)/BD2
                XGRAP3=XGRAPE*XGRAPE*XGRAPE
                XGRAP4=XGRAP3*XGRAPE
                TVDW=CNBB(IC+MAXCN)*(XGRAP3*XGRAP3-TWO*XGRAP3)
                !SIGNPROBLEM
                CEVDW4=CEVDW4+TVDW
                ENB=ENB+TVDW
                IF(QGRIM) THEN
                   CH=CGF*CG(I)*CG(J)*E14FAC
                   AR=BD*KAPPA
                   AR2=AR*AR
                   EFAC=DERFC(BD*KAPPA)
                   ENE=CH*EFAC/BD
                   DFRS=(ENE+TWO*CH*KAPPA/SQRT(PI)*EXP(-AR2))/BD2
                ELSE
                   ENE=CGF*CG(I)*CG(J)/BD
                   DFRS=ENE/BD2
                ENDIF
                EE14=EE14+ENE
                EEL=EEL+ENE
                FGRAP=-TWELVE*CNBB(IC+MAXCN)/CNBA(IC+MAXCN) &
                     *(XGRAP4*XGRAP3-XGRAP4)
                FGRAP=FGRAP-DFRS
                DX(J)=DX(J)+FGRAP*XX
                DY(J)=DY(J)+FGRAP*YY
                DZ(J)=DZ(J)+FGRAP*ZZ
                XFGRAP=XFGRAP+FGRAP*XX
                YFGRAP=YFGRAP+FGRAP*YY
                ZFGRAP=ZFGRAP+FGRAP*ZZ
             ENDIF
          ENDIF  ! QCNOLIST
       ENDDO
       DX(I)=DX(I)-XFGRAP
       DY(I)=DY(I)-YFGRAP
       DZ(I)=DZ(I)-ZFGRAP
200    CONTINUE
    ENDDO
    !
    RETURN
  END SUBROUTINE NCCORR
  !
  SUBROUTINE NBADD14(X,Y,Z,DX,DY,DZ,CGF,EEL,ENB,INB14,IBLO14,IOFF)
    !----------------------------------------------------------------------
    !     This subroutine calculates 1-4 VDW + ELE for atom pairs in the 1-4
    !     exclusion list INB14. Currently for shift+switch
    !
    use exfunc
    use number
    use dimens_fcm
    use psf
    use param
    use inbnd, only: ctonnb,ctofnb,e14fac,lcons,lshft,lvshft,lfswt
    use consta
    use grape
    use ewald
    use energym,only:qeterm,vdw,elec,imvdw,imelec
    use parallel
    !
    !
    REAL(chm_real) :: X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CGF
    real(chm_real) :: EEL,ENB
    INTEGER INB14(*),IBLO14(*),IOFF(*)
    !
    INTEGER I,J,MM,K,L,NXI,NXIMAX,I1,J1,JJ,IC,IACI,iperi,ist
    real(chm_real) :: DUMMY,XMAX,FMAX,SIGFAC
    real(chm_real) :: XX,YY,ZZ,BD,BD2,BEL,XGRAPE,XGRAP3,XGRAP4
    real(chm_real) :: FGRAP,XFGRAP,YFGRAP,ZFGRAP,GFX,GFY,GFZ,GEL,GVDW
    real(chm_real) :: TVDW
    real(chm_real) :: cacx,cbcx,cadx,cbdx,ctof2,ccdx,ch
    real(chm_real) :: ca,cc,tr2,tr6,ccnba,ccnbb,ccnbc,ccnbd
    real(chm_real) :: fsh,ctrof2,c4rof2,r1,ene,tfelec,spi,rk
    real(chm_real) :: cut2,rul3,rul12,rijl,riju,fsw,dfsw,ttvdw
    !
    !
    !      SIGFAC=TWO**(ONE/THREE)
    !
    CUT2=CTONNB*CTONNB
    CTOF2=CTOFNB*CTOFNB
    RUL3=1.0/(CTOF2-CUT2)**3
    RUL12=TWELVE*RUL3
    CTROF2=-ONE/CTOF2
    C4ROF2=FOUR*CTROF2
    spi=one/sqrt(pi)
    IF(CTOFNB < fifty) THEN
       CACX=-2.0/CTOF2**9
       CBCX=1.0/CTOF2**6
       CADX=-1.0/CTOF2**6
       CBDX=1.0/CTOF2**3
       CCDX=CTOF2**3
    ELSE
       CACX=zero
       CBCX=zero
       CADX=zero
       CBDX=zero
       CCDX=zero
    ENDIF

#if KEY_PARALLEL==1
    if(lfmm) then
       ist=1
       iperi=1
    else
       ist=mynodp
       iperi=numnod
    endif
    DO I=ist,NATOM,iperi
       if(lfmm)then
          if(fmmcpu(i)/=1) cycle
       endif
#else
    DO I=1,NATOM
#endif 
       IF(I > 1)THEN
          NXI=IBLO14(I-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=IBLO14(I)
       I1=ITC(IAC(I))
       IACI=IOFF(I1)
       IF(NXI <= NXIMAX)THEN
          XFGRAP=ZERO
          YFGRAP=ZERO
          ZFGRAP=ZERO
          DO K=NXI,NXIMAX
             JJ=INB14(K)
             IF(JJ < 0)THEN
                J=ABS(JJ)
                XX=X(J)-X(I)
                YY=Y(J)-Y(I)
                ZZ=Z(J)-Z(I)
                BD2=MAX(RSMALL,XX*XX+YY*YY+ZZ*ZZ)
!
!                 This code only works if 1-4 atoms are
!                 closer than ctofnb. Please fix!!
!                 Also in case of ewald we need image stuff
!                 here to, if one of the atoms is image !???
!
                J1=ITC(IAC(J))
                IF (I1 < J1) THEN
                   IC=IOFF(J1)+I1
                ELSE
                   IC=IACI+J1
                ENDIF
                tr2=one/bd2
                tr6=tr2*tr2*tr2
                r1=sqrt(tr2)
                ! ele
                tfelec=zero
                ene=zero
                qeterm_elec: if(qeterm(elec))then
                   fmm_14: if(lfmm) then ! no smoothing for FMM yet :-(
                      ! no need for images: 1-4 is never split (check?!)
                      ene = e14fac*cgf*cg(i)*cg(j)*r1
                      tfelec = -ene*tr2
                   else
                   if (lewald) then
                      ! in case of ewald e14fac is maybe useless ???
                      ch=e14fac*cgf*cg(i)*cg(j)
                      rk=kappa/r1
                      ene=ch*(one-erf(rk))*r1
                      tfelec=-(ene+two*ch*kappa*spi*exp(-rk*rk))*tr2
                   else
                      if (lshft) then                ! shift
                         fsh=one+bd2*ctrof2
                         ch=e14fac*cgf*cg(i)*cg(j)*r1*fsh
                         ene=ch*fsh
                         tfelec= -ene*tr2+c4rof2*ch
                      else                               ! switch
                         if(bd2>cut2) then
                            rijl=cut2-bd2
                            riju=ctof2-bd2
                            fsw=riju*riju*(riju-three*rijl)*rul3
                            dfsw=rijl*riju*rul12
                            ene = e14fac*cgf*cg(i)*cg(j)*r1*fsw
                            tfelec = -ene*tr2 + e14fac*cgf*cg(i)*cg(j)*r1*dfsw
                         else
                            ene = e14fac*cgf*cg(i)*cg(j)*r1
                            tfelec = -ene*tr2
                         endif
                      endif
                   endif
                   endif fmm_14
                endif qeterm_elec
                ! vdw (do we need image part too here?)
                fgrap=zero
                tvdw=zero
                qeterm_vdw: if(qeterm(vdw)) then
                   ccnba=cnbb(ic+maxcn)*cnba(ic+maxcn)**6
                   ccnbb=two*cnbb(ic+maxcn)*cnba(ic+maxcn)**3
                   ccnbc=cacx*ccnba+cbcx*ccnbb
                   ccnbd=cadx*ccnba+cbdx*ccnbb+ccdx*ccnbc
                   ca=ccnba*tr6*tr6
                   cc=ccnbc*bd2*bd2*bd2
                   if (lvshft) then                     ! shift
                      tvdw=ca-ccnbb*tr6-cc+ccnbd
                      ! sum up for forces
                      fgrap=minsix*(tvdw+ca+cc)*tr2!+tfelec
                   else                                 ! switch
                      if(bd2>cut2) then
                         rijl=cut2-bd2
                         riju=ctof2-bd2
                         fsw=riju*riju*(riju-three*rijl)*rul3
                         dfsw=rijl*riju*rul12
                         ttvdw=ca-ccnbb*tr6
                         tvdw=ttvdw*fsw
                         fgrap=ttvdw*dfsw+minsix*tr2*(ca+tvdw)*fsw!+tfelec
                      else
                         tvdw=ca-ccnbb*tr6
                         fgrap=minsix*tr2*(ca+tvdw)!+tfelec
                      endif
                   endif
                endif qeterm_vdw
                !straight cut:
                !                  XGRAPE=CNBA(IC+MAXCN)/BD2
                !                  XGRAP3=XGRAPE*XGRAPE*XGRAPE
                !                  XGRAP4=XGRAP3*XGRAPE
                !                  TVDW=CNBB(IC+MAXCN)*(XGRAP3*XGRAP3-TWO*XGRAP3)
                !                  FGRAP=-TWELVE*CNBB(IC+MAXCN)/CNBA(IC+MAXCN)
                !     $                 *(XGRAP4*XGRAP3-XGRAP4)
                !
                ! add to the total results
                ENB=ENB+TVDW
                EEL=EEL+ENE
                fgrap=fgrap+tfelec    ! this routine got qeterm() separated
                DX(J)=DX(J)+FGRAP*XX
                DY(J)=DY(J)+FGRAP*YY
                DZ(J)=DZ(J)+FGRAP*ZZ
                XFGRAP=XFGRAP+FGRAP*XX
                YFGRAP=YFGRAP+FGRAP*YY
                ZFGRAP=ZFGRAP+FGRAP*ZZ
             ENDIF
          ENDDO
          DX(I)=DX(I)-XFGRAP
          DY(I)=DY(I)-YFGRAP
          DZ(I)=DZ(I)-ZFGRAP
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE NBADD14
  !       
  SUBROUTINE HEWEXC(X,Y,Z,CGF,DX,DY,DZ,EWEXCL,INB14,IBLO14)
    !----------------------------------------------------------------------
    !     This subroutine calculates Ewald elelctrostatic exclusion
    !     term in the exclusion list INB14.
    !
    use chm_kinds
    use exfunc
    use number
    use consta
    use dimens_fcm
    use psf
    use ewald
    use grape
    use machutil
    use parallel
    !
    implicit none
    !C      IMPLICIT NONE
    !
    real(chm_real) X(*),Y(*),Z(*),CGF,E14FAC,DX(*),DY(*),DZ(*),EWEXCL
    INTEGER INB14(*),IBLO14(*)
    !
    real(chm_real) DERF
    !
    real(chm_real) XX,YY,ZZ,EWEB,XFGRAP,YFGRAP,ZFGRAP
    real(chm_real) BD2,BD,CH,AR,AR2,EFAC,ENE,DFRS
    real(chm_real) TSTART,TELAP
    INTEGER I,NXI,NXIMAX,k,j
    !
    TSTART=eclock()
    EWEB=ZERO
#if KEY_PARALLEL==1
    DO I=MYNODP,NATOM,NUMNOD
#else /**/
    DO I=1,NATOM
#endif 
       IF(I > 1)THEN
          NXI=IBLO14(I-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=IBLO14(I)
       XFGRAP=ZERO
       YFGRAP=ZERO
       ZFGRAP=ZERO
       DO K=NXI,NXIMAX
          J=INB14(K)
          IF(J > 0) THEN
             XX=X(J)-X(I)
             YY=Y(J)-Y(I)
             ZZ=Z(J)-Z(I)
             BD2=MAX(RSMALL,XX*XX+YY*YY+ZZ*ZZ)
             BD=SQRT(BD2)
             CH=CGF*CG(I)*CG(J)
             AR=BD*KAPPA
             AR2=AR*AR
             EFAC=DERF(BD*KAPPA)
             ENE=-CH*EFAC/BD

             !SIGNPROBLEM
             EWEB=EWEB+ENE
             DFRS=-(ENE+CH*TWO*KAPPA/SQRT(PI)*EXP(-AR2))/BD2
             DX(J)=DX(J)+DFRS*XX
             DY(J)=DY(J)+DFRS*YY
             DZ(J)=DZ(J)+DFRS*ZZ
             XFGRAP=XFGRAP+DFRS*XX
             YFGRAP=YFGRAP+DFRS*YY
             ZFGRAP=ZFGRAP+DFRS*ZZ
          ENDIF
       ENDDO
       DX(I)=DX(I)-XFGRAP
       DY(I)=DY(I)-YFGRAP
       DZ(I)=DZ(I)-ZFGRAP
    ENDDO
    EWEXCL=EWEB
    TELAP=eclock()-TSTART
    MDGEX=MDGEX+TELAP
    !
    RETURN
  END SUBROUTINE HEWEXC
  !
  SUBROUTINE BONDGRAP(INB14,IBLO14)
    !----------------------------------------------------------------------
    !     This routine modifies the bonding info for GRAPE,
    !     and fills the parameter info.
    !
    use chm_kinds
    use number
    use exfunc
    use stream
    use dimens_fcm
    use psf
    use rtf
    use param
    use grape
    use image
    use memory
    use parallel
    use machutil
    !
    implicit none
    !

    INTEGER INB14(*),IBLO14(*)
    !
    INTEGER I,J,NXI,IPT,JPT,K,L,JJ,IOFF(MAXATC)
    INTEGER IDSTN(G2MAXTYP2),MAPTYP(MAXATC)
    INTEGER ITY,JTY,I1,J1,IACI,IC,IPTR,K9,L9
    INTEGER JPTP,JJX,JX,IX,llen1,llen2
    integer nparam(maxatc), bombflag, nparam_sum
    integer*4 natom4
    integer,allocatable,dimension(:) :: numexja
    character(len=6) cparam(maxatc)
    logical new_type_found
    !
    real(chm_real) SIGFAC
    real(chm_real) tstart, telaps
    !
    !
    write(*,*)'igrape,natom,natim=',igrape,natom,natim
    llen1 = natom
    llen1 = 2*natim    ! So that grape==11 works in parallel, too
    if (mod(igrape,10) == 3) llen1 = 2*natim
    if(natim==0)llen1=2*natom  ! for testing if no images
    llen2 = llen1 * lenexlist
    if(.not. allocated(numex)) then
    call chmalloc('grape.src','bondgrap','numex',llen1,ci4=numex)
    call chmalloc('grape.src','bondgrap','numexa',llen1,intg=numexa)
    call chmalloc('grape.src','bondgrap','iacsor',llen1,ci4=iacsor)
    call chmalloc('grape.src','bondgrap','bckptr',llen2,ci4=bckptr)
    call chmalloc('grape.src','bondgrap','natex',llen2,ci4=natex)
#if KEY_PARALLEL==1
    call chmalloc('grape.src','bondgrap','natexp',llen2,ci4=natexp)
    call chmalloc('grape.src','bondgrap','numexp',llen1,ci4=numexp)
    call chmalloc('grape.src','bondgrap','natexp',llen2,ci4=natexpd)
    call chmalloc('grape.src','bondgrap','numexpd',llen1,ci4=numexpd)
    call chmalloc('grape.src','bondgrap','numexap',llen1,intg=numexap)
    call chmalloc('grape.src','bondgrap','natexj',llen2,ci4=natexj)
    call chmalloc('grape.src','bondgrap','numexj',llen1,ci4=numexj)
    call chmalloc('grape.src','bondgrap','piacsor',llen1,ci4=piacsor)
#endif 
    endif
    !
    natom4=natom
    bombflag=0
    SIGFAC=TWO**(ONE/THREE)
    !
    J=0
    DO I=1,MAXATC
       IOFF(I)=J
       J=J+I
    ENDDO
    !
    tstart=eclock()
    IPT=0
    JPTP=0
    DO I=1, NATOM
       IF(I > 1)THEN
          NXI=IBLO14(I-1)+1
       ELSE
          NXI=1
       ENDIF
       JPT=0
       DO JJ = NXI, IBLO14(I)
          J=INB14(JJ)
          !            IF(J > 0)THEN  ! when no nbadd14()
          !    Parallel not yet!
#if 0  /* prll-notgood */
          IF(MYNOD == MOD(JPTP,NUMNOD))THEN      
#endif
             JPT=JPT+1
             IPT=IPT+1
             NATEX(IPT)=ABS(J)
#if 0 /* prll-notgood */
          ENDIF                                  
          JPTP=JPTP+1                            
#endif
          !            ENDIF
       ENDDO
       IF(IPT > llen2)CALL WRNDIE(-5,'<BONDGRAP>', &
            'NATEX index overflow.')
       NUMEX(I)=JPT
       numexa(i) = ipt  ! absolute position of the last element in natex
    ENDDO
    telaps=eclock()-tstart
    mdgini=mdgini+telaps
#if KEY_PARALLEL==1
    !write(outu,*)'BONDGRAP>after 1.loop:me,time=',mynod,telaps  
#endif
#if 1
    DO I=1, llen1
       BCKPTR(I)=I
    ENDDO
    call MR3_setup_exlist(NATOM4, BCKPTR)
#endif 
    !
    !     For parallel add lower triangle part of the matrix to the
    !     existing NUMEX, NATEX
    !
#if KEY_PARALLEL==1
    if(mod(igrape,10) == 3) then    ! this is needed only for parallel (3,13,...)
    ! First find the numexjs
    tstart=eclock()
    ipt=1
    numexj(1:natom)=0
    DO I=1,NATOM
       IF(NUMEX(I) > 0)THEN
          DO JX=IPT,IPT+NUMEX(I)-1
             numexj(natex(jx))=numexj(natex(jx))+1
          ENDDO
          IPT=IPT+NUMEX(I)
       ENDIF
    ENDDO
    ! we need absolute positions to collect j-s
    call chmalloc('grape.src','bondgrap','numexja',llen1,ci4=numexja)
    numexja(1)=0
    do i=1,natom-1
       numexja(i+1)=numexja(i)+numexj(i)
    enddo
    ! fillup natexj-s
    ipt=1
    natexj(1:llen2)=-1
    DO I=1,NATOM
       IF(NUMEX(I) > 0)THEN
          DO JX=IPT,IPT+NUMEX(I)-1
             j=natex(jx)
             do ix=1,numexj(j)
                if(natexj(numexja(j)+ix) == -1) then
                   natexj(numexja(j)+ix)=i
                   exit  ! exits from do ix
                endif
             enddo
          ENDDO
          IPT=IPT+NUMEX(I)
       ENDIF
    ENDDO
    call chmdealloc('grape.src','bondgrap','numexja',llen1,ci4=numexja)
    telaps=eclock()-tstart
    mdgini=mdgini+telaps
    !write(outu,*)'BONDGRAP>after 2a.loop:me,time=',mynod,telaps
    !
    IPTR=0
    IPT=1
    JPT=1
    tstart=eclock()
    DO I=1,NATOM
       DO J=IPT,IPT-1+NUMEX(I)
          IPTR=IPTR+1
          NATEXP(IPTR)=NATEX(J)
       ENDDO
       DO J=JPT,JPT-1+NUMEXJ(I)
          IPTR=IPTR+1
          NATEXP(IPTR)=NATEXJ(J)
       ENDDO
       NUMEXP(I)=NUMEXJ(I)+NUMEX(I)
       IPT=IPT+NUMEX(I)
       JPT=JPT+NUMEXJ(I)
       NUMEXAP(I) = IPTR
    ENDDO
    telaps=eclock()-tstart
    mdgini=mdgini+telaps
    endif
    !
#endif 
    !
    !     Do the mapping between the standard types and the packed ones:
    !     GRAPE-II supports only 32 distinct types, while this was not a
    !     problem on GRAPE-I
    !     For the performance reason put this outside the forces loop
    !     NOTE: BE carefull -> this is a problem IDIST goes out of bounds!!!
    IDIST=1
    IDSTN(1)=iac(1)
    tstart=eclock()
    DO I=1,NATOM
       new_type_found=.true.
       !write(*,'(a,i5,40i3)')'i,iac(i)=',i,iac(i),idstn(1:idist)
       DO J=1,IDIST
          IF(IDSTN(J) == IAC(I)) then
!!! debug:             if(bombflag == 2)write(outu,*) ' problem',i,j,iac(i)
!!! old             GOTO 111
             new_type_found=.false.
             exit  ! j loop
          endif
       ENDDO
       if(new_type_found) then
          IDIST=IDIST+1
          IF(IDIST > G2MAXTYP)bombflag=1
          IDSTN(IDIST)=IAC(I)
       endif
!!! old 111    CONTINUE
    ENDDO
!!! old    IDIST=IDIST-1
    if(prnlev > 2) WRITE(OUTU,'(A,I5)')'Number of distinct atom types =',IDIST
    !
    !     Parameter statistics:
    !
    do i=1,maxatc
       nparam(i)=0
       cparam(i)='      '
    enddo
    do i=1,natom
       do j=1,idist
          if(iac(i) == idstn(j))then
             if(idstn(j) > maxatc)then
                WRITE(OUTU,'(A,I5)')'Something went wrong',IDSTN(j)
                CALL WRNDIE(-5,'<ENBGRAP>','Wrong statistics')
             endif
             nparam(idstn(j))=nparam(idstn(j))+1
             cparam(idstn(j))=atct(iac(i))
!!! old             goto 222
             exit
          endif
       enddo
!!! old 222    continue
    enddo
    if(prnlev > 2)then
       nparam_sum=0
       write(outu,*)'Parameter statistics:',idist
       write(outu,'(''type #  Par.name     Par.code    # atoms'')')
       do i=1,idist
          write(outu,'(i5,4x,a6,2i12)') &
               i,cparam(idstn(i)),idstn(i),nparam(idstn(i))
          nparam_sum=nparam_sum+nparam(idstn(i))
       enddo
       if(natom /= nparam_sum)write(outu,*) &
            'problem with parameter assignments'
    endif
    !
    IF(bombflag == 1)then
       WRITE(OUTU,'(A,I5)')'Number of distinct atom types =',IDIST
       CALL WRNDIE(-5,'<ENBGRAP>','Too many distinct atom types')
    ENDIF
    !
    telaps=eclock()-tstart
    mdgini=mdgini+telaps
#if KEY_PARALLEL==1
    !write(outu,*)'BONDGRAP>after 4.loop:me,time=',mynod,telaps 
#endif
    !
    !     We need the invert of IDSTN
    !
    DO I=1,NATC
       MAPTYP(I)=-1
    ENDDO
    DO I=1,IDIST
       MAPTYP(IDSTN(I))=I
    ENDDO
    !
    tstart=eclock()
    if (idist**2 > g2maxtyp2) &
         call wrndie(-5,'<ENBGRAP>','parameter overflow.')
    rscale(1:idist**2)=zero
    gscale(1:idist**2)=zero
    frscale(1:idist**2)=zero
    fgscale(1:idist**2)=zero
    !
    IACSOR(1:NATOM)=MAPTYP(IAC(1:NATOM))
    ! This loop can be further cleaned up  ??
    DO ITY=1,NATC
       K9=MAPTYP(ITY)
       if(k9 == -1) cycle
       I1=ITC(ITY)
       IACI=IOFF(I1)
       DO JTY=1,NATC
          L9=MAPTYP(JTY)
          if(l9 == -1) cycle
          J1=ITC(JTY)
          IF(I1 < J1)THEN
             IC=IOFF(J1)+I1
          ELSE
             IC=IACI+J1
          ENDIF
          IF((L9 > 0).AND.(K9 > 0)) THEN
             IF(CNBA(IC) > RSMALL) THEN
                rscale(k9+(l9-1)*idist)=SIGFAC/CNBA(IC)
                frscale(k9+(l9-1)*idist)=SIGFAC/CNBA(IC)
                fgscale(k9+(l9-1)*idist)= &
                     SIGFAC*TWELVE*TWO*CNBB(IC)/CNBA(IC)
             ENDIF
             gscale(k9+(l9-1)*idist)=FOUR*CNBB(IC)
             rscale(l9+(k9-1)*idist)=rscale(k9+(l9-1)*idist)
             gscale(l9+(k9-1)*idist)=gscale(k9+(l9-1)*idist)
             frscale(l9+(k9-1)*idist)=frscale(k9+(l9-1)*idist)
             fgscale(l9+(k9-1)*idist)=fgscale(k9+(l9-1)*idist)
          ENDIF
       ENDDO
    ENDDO
    telaps=eclock()-tstart
    mdgini=mdgini+telaps
#if KEY_PARALLEL==1
    !write(outu,*)'BONDGRAP>after 6.loop:me,time=',mynod,telaps  
#endif
    !
    RETURN
  END SUBROUTINE BONDGRAP
  !
  !
  SUBROUTINE EWGRAP(X,Y,Z,CGF,DX,DY,DZ,EEL,ENB,QENE,IOFF,   &
#if KEY_FLUCQ==1
       QFLUC,FQCFOR,                                        & 
#endif
       INB14,IBLO14)
    !----------------------------------------------------------------------
    !     This is the version of the Ewald energy terms for the
    !      GRAPE (GRAvity PipE) machine
    !
    !$ use omp_lib
    use chm_kinds
    use number
    use exfunc
    use fast, only: iacnb
    use dimens_fcm
    use psf
    use param
    use inbnd, only: lvfswt,lvshft,ctonnb,ctofnb,e14fac
    use consta
    use coordc ! for fmm_partition communication, later do the xold here...
    use grape
    use ewald
    use image
    use energym
    use machutil
    use parallel
    use stream
    use bases_fcm, only: bimag
    !
    !
    real(chm_real) X(*),Y(*),Z(*),CGF,DX(*),DY(*),DZ(*),EEL,ENB
    LOGICAL QENE
    INTEGER IOFF(*),INB14(*),IBLO14(*),istart,iend,iorig,istartp
    !
#if KEY_FLUCQ==1
    LOGICAL QFLUC                                             
    real(chm_real) FQCFOR(*)                                          
#endif
    !
    real(chm_real) FMAX,XMAX,DUMMY
    real(chm_real) TSTART,TELAPS,ttstart
    real(chm_real) ALPHA2, ALPHA3, EKEL, B, VFACT
    INTEGER I,J,OVFLOW,IPTR,IPT,IOPT,KX,KY,KZ
    INTEGER KSY,KSZ,KSQ
    real(chm_real) RKX,RKY,RKZ,RKSQ,ESELF
    INTEGER, PARAMETER :: MAXKVEC=10000
    real(chm_real) KVECI(3,MAXKVEC),EWSIN(MAXKVEC)
    real(chm_real) EWCOS(MAXKVEC),KVEC(MAXKVEC)
    real(chm_real) CS(3,MAXKVEC),CC(3,MAXKVEC)
    real(chm_real) TDX,TDY,TDZ, CELL(3,3),STRESS(3,3),PI4,TPOT
    real(chm_real) fmmgrms,xxfmm
    INTEGER KVECIT(3,MAXKVEC)
    !
    INTEGER*4 MODOP,iperiod,natom4,ioper,igrim,igrim2,tblno,tblno2
    INTEGER*4 NPT,natim4
    INTEGER*4 flagregexlist,potflag,idummy,vdwmode
    integer npart, nparticles, iatom, iat, lifrsta
    INTEGER*4 ldim(3), ciflag, modop1
    real(chm_real) vol(3), rcut, skinnb,eel_14,enb_14,cgepsilon
    real(chm_real), allocatable, dimension(:) :: tmpdx,tmpdy,tmpdz,lcg
    real(chm_real), allocatable, dimension(:) :: posgrp_main
    real(chm_real), allocatable, dimension(:) :: posgrp_comp
    logical llelecx,llvdwx,llused
    integer orig_igrape ! for fmm trickery
    integer fmmksize
    real(chm_real) fmmalpha,fmmsigma,fmmcutoff
    !
    orig_igrape=igrape
    igrape=save_igrape
    PI4=ONE/PI/FOUR
    ALPHA2=kappa*kappa
    ALPHA3=kappa*ALPHA2
    FMAX=MEGA
    !     Good for cubic only!
    !      IF(XTLTYP /= 'CUBI')CALL WRNDIE(-5,'<EWGRAP>','CUBIC only.')
    XMAX=HALF*XUCELL(1)
    DUMMY=ONE
    TSTART=eclock()
    !
    !! currently not in use here: singrp(1:3,1:natim) = zero
    npart=natom
    if(mod(igrape,10) == 3) npart=natim
    if(natim==0) npart = natom
    !write(*,*) 'ewgrap>npart,natom,natim,igrape=',npart,natom,natim,igrape
    posgrp(1,1:npart) = x(1:npart)
    posgrp(2,1:npart) = y(1:npart)
    posgrp(3,1:npart) = z(1:npart)
    !
    iperiod=1
    natom4=natom
    natim4=natim

    call cpugpusplitset

    !     Perform normal case: VDW & Coulomb are separate calls
    !
    ! currently we support shift(vdwmode=1) and switch(vdwmode=2)
    
    if(lvshft) then
       vdwmode=1
    else
       vdwmode=2
    endif
    call MR3_set_vdwmode(vdwmode)
    call MR3_set_rcut2(CTOFNB*CTOFNB)
    call MR3_set_cuton(CTONNB)

!!! Start this routine with the check for FMM
!!! Perform all the FMM in this block of code

    FMM: if (lfmm) then


!!! MAYBE NOT:
       ! we need to take care of the charge=zero cases here
       ! more efficient should be in FMM library!??
       call chmalloc('grape.src','EWGRAP','lcg',natom,crl=lcg)
       lcg(1:natom)=cg(1:natom)
!       do i=1,natom
!          if(abs(lcg(i))<rsmall)lcg(i)=rsmall
!       enddo
!!!
       xxfmm=eclock()
       ! second time instead of posgrp should be velocities
       !write(*,*)'before partition: igrape, orig_igrape, save_igrape ',igrape, orig_igrape, save_igrape
       !write(*,*)'before partition: natom4, natom, npart ',natom4, natom, npart
       ! for fmm_partition communication, later do the xold here...
       if(.not.allocated(posgrp_comp)) &
            call chmalloc('grape.src','ewgrap','posgrp_comp',3*npart,crl=posgrp_comp)
       posgrp_comp(1:npart) = xcomp(1:npart)
       posgrp_comp(npart+1:2*npart) = ycomp(1:npart)
       posgrp_comp(2*npart+1:3*npart) = zcomp(1:npart)
       
       call fmm_partition(natom, fmmcpu, posgrp, lcg, posgrp_comp, xucell(1))
       if(allocated(posgrp_comp)) &
            call chmdealloc('grape.src','ewgrap','posgrp',3*npart,crl=posgrp_comp)
       xxfmm=eclock()-xxfmm
       !write(*,*)'me,FMM_partition-time=',mynod,xxfmm

       FMM_electrostatic: if (qeterm(elec)) then
          rslgrp(1:3,1:natom)=zero
          rslgrpot(1:natom)=zero
          xxfmm=eclock()
          ! for now this is too slow, developing new code ...
!          goto 12321
          call fmm_coulomb(natom, fmmcpu, posgrp, lcg, rslgrpot, rslgrp, xucell(1))
!          write(*,*)'me,FMM-sum_of_forces=',mynod,sum(rslgrp(1:3,1:natom))
          ! to check with ewald (not particle mesh) in FMM library
          !fmmksize=11
          !fmmalpha=ten/xucell(1)
          !fmmsigma=fourth/pi
          !fmmcutoff=ten
          !call ewald_coulomb(natom4, fmmcpu, posgrp, lcg, rslgrpot, rslgrp, &
          !     fmmksize, fmmalpha, fmmsigma, fmmcutoff, xucell(1))
          call coulomb_exclusion(natom, fmmcpu, posgrp, lcg, rslgrpot, rslgrp, xucell(1),numexp,natexp)
          xxfmm=eclock()-xxfmm
!          write(*,*)'me,FMM-sum_of_forces(after exc.)=',mynod,sum(rslgrp(1:3,1:natom))
          !write(*,*)'me,FMM-time=',mynod,xxfmm
!          call gcomb(xxfmm,1)
          !write(*,*)'FMM-average-time=',xxfmm/numnod
!12321     continue
          xxfmm=zero
          ipt=0
          do i=1,natom
             if(fmmcpu(i)/=1) cycle
             ipt=ipt+1
             xxfmm=xxfmm+rslgrpot(i)!*cg(i)
             dx(i)=dx(i)+rslgrp(1,i)
             dy(i)=dy(i)+rslgrp(2,i)
             dz(i)=dz(i)+rslgrp(3,i)
          enddo
          xxfmm=xxfmm*half!*cgf
          !write(*,*)'me,FMM-elec=',mynod,xxfmm
          eel=xxfmm
          GRPELE=EEL
          ene_c=eel
       endif FMM_electrostatic

       !VDW
       FMM_vdw: if (qeterm(vdw)) then
          rslgrp(1:3,1:natom)=zero
          rslgrpot(1:natom)=zero

          call fmm_vanderwaals(natom,fmmcpu,iacsor,posgrp,rslgrpot,rslgrp,ctonnb,ctofnb,&
               xucell(1),idist,rscale,gscale,fgscale)
          !write(200+mynod,*)'vdw potential:'
          !write(200+mynod,'(5f15.7)')rslgrpot(1:natom)
          call vanderwaals_exclusion(natom,fmmcpu,iacsor,posgrp,rslgrpot,rslgrp,ctonnb,ctofnb,&
               xucell(1),idist,rscale,gscale,fgscale,numexp,natexp)

          !write(300+mynod,*)'vdw exclusion potential:'
          !write(300+mynod,'(5f15.7)')rslgrpot(1:natom)
          
          enb=zero
          do i=1,natom
             if(fmmcpu(i)/=1) cycle
             enb=enb+rslgrpot(i)
             dx(i)=dx(i)+rslgrp(1,i)
             dy(i)=dy(i)+rslgrp(2,i)
             dz(i)=dz(i)+rslgrp(3,i)
          enddo
          enb=half*enb
          ene_v = enb
          GRPVDW=ENB
       endif FMM_vdw

       if(igrape==7) then
          write(*,*)'fmm>me,ene_c,ene_v=',mynod,ene_c,ene_v
       else
          !write(*,*)'fmm>me,ene_c,ene_v=',mynod,ene_c,ene_v
       endif

       ! 1-4 corrections:
       eel_14=zero
       enb_14=zero
       call nbadd14(x,y,z,dx,dy,dz,cgf,eel_14,enb_14,inb14,iblo14,ioff)
       if(igrape==7) then
          write(*,*)'fmm>me,lewald,eel_14,enb_14=',mynod,lewald,eel_14,enb_14
       else
          !write(*,*)'fmm>me,lewald,eel_14,enb_14=',mynod,lewald,eel_14,enb_14
       endif
       ene_c=ene_c+eel_14
       ene_v=ene_v+enb_14
       eel=eel+eel_14
       enb=enb+enb_14

!!! End of FMM
       igrape=orig_igrape ! FMM trickery: probably not needed anymore???
       call chmdealloc('grape.src','EWGRAP','lcg',natom,crl=lcg)

       RETURN

    endif FMM

    if(mod(igrape,10) == 0)then

       ioper=507
       !
       !     Real space energy:
       ene_v = zero
       ene_c = zero
       if(lfmm)qene=.true.
       IF(QENE) THEN
          rslgrp(1:3,1:natom) = zero
          rslgrpot(1:natom) = zero
          MODOP=7
#if 1
          qeterm_elec_energy: if (qeterm(elec)) then   ! also we could check for qeterm(IMELEC)
             IF(.NOT.LFMM) THEN
                call MR3calccoulomb_ij_exlist( &
                     NATOM4,posgrp,cg,rslgrp, &
                     NATOM4,posgrp,cg, kappa, &
                     modop, xucell(1), iperiod, &
                     numex, natex )
                write(outu,*)'After MR3> elec. energy = ', sum(rslgrp(1,1:natom))
                write(outu,*)'After MR3> elec. energy = ', rslgrp(1,1:100)
             ELSE
                write(outu,*)'EWGRAP-beforeFMM>grape,qene=',igrape,qene
                !write(55,'(4f20.10)')(rscale(i),gscale(i),frscale(i),fgscale(i),i=1,idist**2)
                !write(55,'(2i8,4f15.8)')(i,iacsor(i),posgrp(1,i),posgrp(2,i),posgrp(3,i),cg(i),i=1,natom)
                !write(55,'(2i10)')(numex(i),natex(i),i=1,natom)
                !call FMMcalccoulomb_ij( &
                !     NATOM4,posgrp,cg,rslgrp, &
                !     NATOM4,posgrp,cg, kappa, &
                !     modop, xucell(1), iperiod)
!!                call FMMcalccoulomb_ij_exlist( &
!!                     NATOM4,posgrp,cg,rslgrp, &
!!                     NATOM4,posgrp,cg, kappa, &
!!                     modop, xucell(1), iperiod, numex, natex )
                call chmalloc('grape.src','EWGRAP','lcg',natom,crl=lcg)
                ! we need to take care of the charge=zero cases here
                ! more effifient should be in FMM library!
                lcg(1:natom)=cg(1:natom)
                !@!! maybe a problem, because not zero total charge
                !@!! need to subtract a small charge from any neighbor charge
                !@!! in case of rsmall no changes on 80,000 atoms :-)
                !@!cgepsilon=rsmall
                !@!do i=1,natom
                !@!   if (abs(lcg(i)) < cgepsilon) then
                !@!      lcg(i)=rsmall
                !@!      !if (i<natom) then
                !@!      !   lcg(i+1)=lcg(i+1)-cgepsilon
                !@!      !else
                !@!      !   lcg(i-1)=lcg(i-1)-cgepsilon
                !@!      !endif
                !@!   endif
                !@!enddo
                ! this works OK
                do i=1,natom
                   if(abs(lcg(i))<rsmall)lcg(i)=rsmall
                enddo
!!!!
                !xxfmm=zero
                !ipt=0
                !do i=1,natom
                !   ipt=ipt+1
                !   xxfmm=xxfmm+rslgrpot(i)*cg(i)
                !   write(*,*)'i=',i,' pot=',rslgrpot(i),posgrp(1,i),posgrp(2,i),posgrp(3,i),lcg(i)
                !enddo
!!!!
!!! doesn't work:                write(*,*)'before FMM>qgpupd: ',qgpupd
!!! doesn't work:                if(qgpupd) call fmm_partition(natom4, fmmcpu, posgrp, lcg, xucell(1))
                xxfmm=eclock()
                !call fmm_partition(natom4, fmmcpu, posgrp, lcg, xucell(1))
                xxfmm=eclock()-xxfmm
                write(*,*)'me,FMM_partition-time=',mynod,xxfmm
                xxfmm=eclock()
!                call fmm_coulomb(natom4, fmmcpu, posgrp, lcg, rslgrpot, rslgrp, xucell(1))
!                call coulomb_exclusion(natom4, fmmcpu, posgrp, lcg, rslgrpot, rslgrp, xucell(1),numexp,natexp)
                !call ewald_coulomb(natom4, fmmcpu, posgrp, lcg, rslgrpot, rslgrp, &
                !     11, 10.0/xucell(1), 0.25/pi, 10.0, xucell(1))
                !call coulomb_exclusion(natom4, fmmcpu, posgrp, lcg, rslgrpot, rslgrp, xucell(1),numexp,natexp)
                xxfmm=eclock()-xxfmm
                write(*,*)'me,FMM-time=',mynod,xxfmm
                call gcomb(xxfmm,1)
                write(*,*)'FMM-average-time=',xxfmm/numnod
                call chmdealloc('grape.src','EWGRAP','lcg',natom,crl=lcg)
                xxfmm=zero
                ipt=0
                do i=1,natom
                   if(fmmcpu(i)/=1) cycle
                   ipt=ipt+1
                   xxfmm=xxfmm+rslgrpot(i)*cg(i)
                   !write(*,*)'mynod=',mynod,' i=',i,' pot=',rslgrpot(i)
                enddo
                xxfmm=half*cgf*xxfmm
                eel=xxfmm
                !xxfmm=half*cgf*sum(rslgrpot(1:natom))
!                write(outu,*)'After FMM> xucell,xmin,xmax,ymin,ymax,zmin,zmax ', &
!                     xucell(1),cmin(x(1:natom)),cmax(x(1:natom)),cmin(y(1:natom)),cmax(y(1:natom)),&
!                     cmin(z(1:natom)),cmax(z(1:natom))
!                write(outu,*)'After FMM>n, elec. energy = ',ipt, xxfmm
                call gcomb(xxfmm,1)
                call igcomb(ipt,1)
                if(prnlev>2)write(outu,*)'After FMM>n, Total elec. energy = ', ipt, xxfmm
                fmmgrms=zero
                ipt=0
                do i=1,natom
                   if(fmmcpu(i)/=1) cycle
                   ipt=ipt+1
                   fmmgrms=fmmgrms+rslgrp(1,i)**2+rslgrp(2,i)**2+rslgrp(3,i)**2
                enddo
                call gcomb(fmmgrms,1)
                fmmgrms=ccelec*sqrt(fmmgrms/natom/three)*half
                !if(prnlev>2)write(outu,*)'After FMM> elec. force GRMS = ', fmmgrms
                !write(outu,*)'After FMM> elec. forces = ', cgf*rslgrp(1:3,1:100)
!                write(outu,*)'After FMM> elec. energy = ', cgf*sum(rslgrp(1,1:natom))
!                write(outu,*)'After FMM> elec. energy = ', rslgrp(1,1:100)
!                rslgrpot(1:natom)=zero
!                kmaxx=15
!                kappa=0.33
!                call fmmewald(natom4, posgrp, cg, rslgrpot, rslgrp, kmaxx, kappa, xucell(1))
!                xxfmm=cgf*sum(rslgrpot(1:natom))
!                write(outu,*)'After FMM> elec. energy(ewald) = ', xxfmm
                !call mpi_finalize(i)
                !stop
             ENDIF
          endif qeterm_elec_energy
#else
          !     call MR1calccoulomb(posgrp,NATOM,cg,kappa,
          !     +     MODOP,xucell(1),1,0,rslgrp)
          call MR1calccoulomb_ij(NATOM4,posgrp,cg,rslgrp, &
               NATOM4,posgrp,cg,kappa,MODOP,xucell(1),iperiod)
          call MR1calccoulomb_nlist_emu(posgrp,natom4,cg,kappa, &
               ioper,xucell(1),iperiod,numex,natex,-ONE,rslgrp)
#endif
          if(.not.lfmm)then
             EEL=ZERO
             DO I=1,NATOM
                eel=eel+cgf*rslgrp(1,i)
             ENDDO
          endif
       ENDIF ! if(QENE)
       !     
       !     Put real space electrostatic in IMELEC:
       IF(QENE) THEN
          GRPELE=EEL
          ene_c=eel
!          EEL=ZERO
       ENDIF
       !     
       !     Real space forces:
       !
       ! in the case of FMM forces are already calculated
       if(.not.lfmm)rslgrp(1:3,1:natom)=zero
       MODOP=6
       ioper=506
#if 1
       qeterm_elec_force: if (qeterm(elec)) then   ! also we could check for qeterm(IMELEC)
          IF(.NOT.LFMM) THEN
             call MR3calccoulomb_ij_exlist( &
                  NATOM4,posgrp,cg,rslgrp, &
                  NATOM4,posgrp,cg, kappa, &
                  modop, xucell(1), iperiod, &
                  numex, natex )
!             fmmgrms=zero  ! debug printout...
!             do i=1,natom
!                fmmgrms=fmmgrms+rslgrp(1,i)**2+rslgrp(2,i)**2+rslgrp(3,i)**2
!             enddo
!             fmmgrms=ccelec*sqrt(fmmgrms/natom/three)
!             write(outu,*)'After MR3> elec. force GRMS = ', fmmgrms
!             write(outu,*)'After MR3> elec. force sum (x) = ', sum(rslgrp(1,1:natom))
!             write(outu,*)'After MR3> elec. force 100 (x) = ', rslgrp(1,1:100)
          ELSE
!!             call FMMcalccoulomb_ij_exlist( &
!!                  NATOM4,posgrp,cg,rslgrp, &
!!                  NATOM4,posgrp,cg, kappa, &
!!                  modop, xucell(1), iperiod, numex, natex )
             !call FMMcalccoulomb_ij( &
             !     NATOM4,posgrp,cg,rslgrp, &
             !     NATOM4,posgrp,cg, kappa, &
             !     modop, xucell(1), iperiod )
             fmmgrms=zero
             do i=1,natom
                fmmgrms=fmmgrms+rslgrp(1,i)**2+rslgrp(2,i)**2+rslgrp(3,i)**2
             enddo
             fmmgrms=ccelec*sqrt(fmmgrms/natom/three)
             write(outu,*)'After FMM> elec. force GRMS = ', fmmgrms
             !write(outu,*)'After FMM> elec. force 100 (x) = ', rslgrp(1,1:100)
          ENDIF
       endif qeterm_elec_force
#else
       !      call MR1calccoulomb(posgrp,NATOM,cg,kappa,
       !     +     MODOP,xucell(1),1,0,rslgrp)
       call MR1calccoulomb_ij(NATOM4,posgrp,cg,rslgrp, &
            NATOM4,posgrp,cg,kappa,MODOP,xucell(1),iperiod)
       call MR1calccoulomb_nlist_emu(posgrp,natom4,cg,kappa, &
            ioper,xucell(1),iperiod,numex,natex,-ONE,rslgrp)
#endif 
       IPTR=0
       if(lfmm)then
          DO I=1,NATOM
             dx(i)=dx(i)+cgf*rslgrp(1,i)*cg(i)
             dy(i)=dy(i)+cgf*rslgrp(2,i)*cg(i)
             dz(i)=dz(i)+cgf*rslgrp(3,i)*cg(i)
          ENDDO
       else
          DO I=1,NATOM
             dx(i)=dx(i)-cgf*rslgrp(1,i)
             dy(i)=dy(i)-cgf*rslgrp(2,i)
             dz(i)=dz(i)-cgf*rslgrp(3,i)
          ENDDO
       endif      
       !     
       TELAPS=eclock()-TSTART
       MDGRS=MDGRS+TELAPS
       !     
       !     BEGIN Van der Waals   ****************************************************
       !
       TSTART=eclock()
       !
       !      IF(QETERM(VDW))
       !     $        CALL VDWGRAP(X,Y,Z,DX,DY,DZ,ENB,IOFF,QENE,IOFF
#if 0
       !     $        PPOS,                                  
#endif
       !     $        POSGRP,RSLGRP)
       !
       FMAX=MEGA
       XMAX=THOSND
       IF(QGRIM)XMAX=XUCELL(1)
       DUMMY=ONE
       !
       !     Prepare parameters for VDW in GRAPE
       !
       !
       posgrp(1,1:natom) = x(1:natom)
       posgrp(2,1:natom) = y(1:natom)
       posgrp(3,1:natom) = z(1:natom)
       rslgrp(1:3,1:natom) = zero
       istart=1
#if KEY_PARALLEL==1
       istart=mynod*((natom+numnod-1)/numnod)+1
       iend=mynodp*((natom+numnod-1)/numnod)
       if(iend > natom)iend=natom
       IPTR=1
       DO I=istart,iend
          pposgrp(1,iptr)=x(i)
          pposgrp(2,iptr)=y(i)
          pposgrp(3,iptr)=z(i)
          PIACSOR(iptr)=IACSOR(i)
          pcggrp(iptr)=cg(i)
          iptr=iptr+1
       ENDDO
       NPT=IPTR-1
#endif 

       !
       !     VDW Energy:
       !
       IF(QGRIM) THEN
          IGRIM=1
       ELSE
          IGRIM=0
       ENDIF
       IGRIM2=IGRIM+2
       !
       IF(QENE) THEN
          tblno=3
#if 1
          qeterm_vdw_e: if (qeterm(VDW)) then   ! also we could check for qeterm(IMVDW)
!*             IF(.NOT.LFMM) THEN
          call MR3calcvdw_ij_exlist( &
#if KEY_PARALLEL==0
               NATOM4,posgrp,iacsor,                                 &
#endif
#if KEY_PARALLEL==1
               NPT,pposgrp,piacsor,                                  & 
#endif
               rslgrp(1,istart),NATOM4,posgrp,iacsor, &
               IDIST,gscale,rscale, &
               tblno, xmax, igrim, &
               numex, natex )
                !write(outu,*)'After MR3> vdw. energy sum = ', sum(rslgrp(1,1:natom))
                !write(outu,*)'After MR3> vdw. energy 100 (x) = ', rslgrp(1,1:100)
!*             ELSE
!!                call FMMcalcvdw_ij_exlist( &
!!                     NATOM4,posgrp,iacsor,                                 & !##.not.PARALLEL
!!                     NPT,pposgrp,piacsor,                                  & !##PARALLEL
!!                     rslgrp(1,istart),NATOM4,posgrp,iacsor, &
!!                     IDIST,gscale,rscale, &
!!                     tblno, xmax, igrim, numex, natex )
!*                write(outu,*)'After FMM> vdw. energy sum = ', sum(rslgrp(1,1:natom))
!*                write(outu,*)'After FMM> vdw. energy 100 (x) = ', rslgrp(1,1:100)
!*             ENDIF
          endif qeterm_vdw_e
#else /**/
          CALL MR1calcvdw_ij( &
#if KEY_PARALLEL==0
               NATOM4,posgrp,IACSOR, & 
#endif
#if KEY_PARALLEL==1
               NPT,pposgrp,PIACSOR,  & 
#endif
               rslgrp(1,istart),NATOM4,posgrp,IACSOR, &
               IDIST,GSCALE,RSCALE,tblno,xmax,IGRIM)
#endif 
          !
#if 0
          tblno=503
          CALL MR1calcvdw_nlist_emu2(posgrp,NATOM4,IACSOR,IDIST,GSCALE, &
#if KEY_PARALLEL==0
               RSCALE,tblno,xmax,IGRIM,numex,natex,-ONE,rslgrp)        
#endif
#if KEY_PARALLEL==1
               RSCALE,tblno,xmax,IGRIM2,numexp,natexp,-ONE,rslgrp)     
#endif
#endif 
          !
       ENDIF
       !
       IF(QENE)THEN
          ENB=ZERO
          DO I=1,NATOM
             ENB=ENB+rslgrp(1,I)
          ENDDO
       ENDIF
       !
       !     Images: Put enb to zero and store everything to IMVDW
       IF(QGRIM) THEN
          ene_v = enb
          GRPVDW=ENB
          !          ENB=ZERO
       ENDIF
       !
       !     VDW Forces:
       !
       rslgrp(1:3,1:natom)=zero
       TBLNO=2
#if 1
       qeterm_vdw_f: if (qeterm(vdw)) then  ! we could also check for qeterm(IMVDW)
!!!*          IF(.NOT.LFMM) THEN
          call MR3calcvdw_ij_exlist( &
#if KEY_PARALLEL==0
               NATOM4,posgrp,iacsor,                                     & 
#endif
#if KEY_PARALLEL==1
               NPT,pposgrp,piacsor,                                      & 
#endif
               rslgrp(1,istart),NATOM4,posgrp,iacsor, &
               IDIST,fgscale,frscale, &
               tblno, xmax, igrim, &
               numex, natex )
             fmmgrms=zero
             do i=1,natom
                fmmgrms=fmmgrms+rslgrp(1,i)**2+rslgrp(2,i)**2+rslgrp(3,i)**2
             enddo
             fmmgrms=sqrt(fmmgrms/natom/three)
             !write(outu,*)'After MR3> vdw. forces GRMS = ', fmmgrms
             !write(outu,*)'After MR3> vdw. forces 100 (x) = ', rslgrp(1,1:100)
!*          ELSE
!!             call FMMcalcvdw_ij_exlist( &
!!                  NATOM4,posgrp,iacsor,                                 & !##.not.PARALLEL
!!                  NPT,pposgrp,piacsor,                                  & !##PARALLEL
!!                  rslgrp(1,istart),NATOM4,posgrp,iacsor, &
!!                  IDIST,fgscale,frscale, &
!!                  tblno, xmax, igrim, numex, natex)
!*             fmmgrms=zero
!*             do i=1,natom
!*                fmmgrms=fmmgrms+rslgrp(1,i)**2+rslgrp(2,i)**2+rslgrp(3,i)**2
!*             enddo
!*             fmmgrms=sqrt(fmmgrms/natom/three)
!*             write(outu,*)'After FMM> vdw. forces GRMS = ', fmmgrms
!*             write(outu,*)'After FMM> vdw. forces 100 (x) = ', rslgrp(1,1:100)
!*          ENDIF
       endif qeterm_vdw_f
#else /**/
       CALL MR1calcvdw_ij( &
#if KEY_PARALLEL==0
            NATOM4,posgrp,IACSOR,rslgrp,          & 
#endif
#if KEY_PARALLEL==1
            NPT,pposgrp,PIACSOR,rslgrp(1,istart), & 
#endif
            NATOM4,posgrp,IACSOR,IDIST,FGSCALE,FRSCALE, &
            tblno,XMAX,IGRIM)
       TBLNO2=502
       CALL MR1calcvdw_nlist_emu2(posgrp,NATOM4,IACSOR,IDIST,FGSCALE,FRSCALE, &
#if KEY_PARALLEL==0
            tblno2,xmax,IGRIM,numex,natex,-ONE,rslgrp)      
#endif
#if KEY_PARALLEL==1
            tblno2,xmax,IGRIM2,numexp,natexp,-ONE,rslgrp)   
#endif
#endif 
       !
       DO I=1,NATOM
          DX(I)=DX(I)-rslgrp(1,I)
          DY(I)=DY(I)-rslgrp(2,I)
          DZ(I)=DZ(I)-rslgrp(3,I)
       ENDDO
       !
       !     END VDW   ****************************************************
       !
    endif
    !     End of case igrape = 0
    !
    !     Start mod(igrape,10)=1 - fastest +++
    !
!!*!!$OMP PARALLEL PRIVATE(TSTART,TELAPS) asdfasdfadf
!!*!!$OMP SECTIONS
!!*!!$OMP SECTION 
    !!old      if(igrape == 1 .or. igrape.eq.2) then
    !     here comes fast routine
    IF(QENE) THEN
       potflag=1
    ELSE
       potflag=0
    ENDIF
    rslgrp(1:3,1:npart)=zero
    istart=1
    IPTR=0
#if KEY_PARALLEL==1
    ! this is for parallel decomposition in blocks
    if (qpargblock) then
       ! make this the same as the rest of parallel methods
       ! by using gpumap & mapgpu
       istart=mynod*((npart+numnod-1)/numnod)+1
       iend=mynodp*((npart+numnod-1)/numnod)
       if(iend > npart)iend=npart
       IPTR=1
       DO I=istart,iend
          pposgrp(1,iptr)=x(i)
          pposgrp(2,iptr)=y(i)
          pposgrp(3,iptr)=z(i)
          PIACSOR(iptr)=IACSOR(i)
          pcggrp(iptr) = cg(i)
          iptr=iptr+1
       ENDDO
       NPT=IPTR-1
    endif
    istart=1     ! needed sometimes...
    IPTR=1
    DO I=1,npart
       if( gpumap(i) > 0 ) then
          pposgrp(1,iptr)=x(i)
          pposgrp(2,iptr)=y(i)
          pposgrp(3,iptr)=z(i)
          PIACSOR(iptr)=IACSOR(i)
          pcggrp(iptr) = cg(i)
          iptr=iptr+1
       endif
    ENDDO
    NPT=IPTR-1
#endif 
    modop = 6
    ! we use normal cutoff methods for periodic non PME calculations
    if(.not.lewald) modop = 0
    xmax = xucell(1)
    flagregexlist=2+256+512
    flagregexlist=flagregexlist+64
    if(mod(igrape,10) == 1) then
       ene_c = ZERO
       ene_v = ZERO
       rcut=CTOFNB
       skinnb=0.0d0
       modop1=modop+1
       do i=1,3
          vol(i)=xucell(i)
          ldim(i)=int(vol(i)/(rcut+skinnb))
       enddo
       call MR3_set_ldim(ldim)
       call MR3_set_coulomb_vdw_factor(cgf)
       call MR3calccoulomb_vdw_exlist(NATOM4,posgrp,cg,iacsor,IDIST, &
            rslgrp(1,istart),kappa,fgscale,gscale,rscale, &
            modop,numex,natex,xmax,ene_c,ene_v, &
            BCKPTR,potflag,flagregexlist)
       !call MR3_get_forces_overlap(rslgrp(1,istart))
    else if (mod(igrape,10) == 3) then
       ene_c = ZERO
       ene_v = ZERO
       call MR3_set_coulomb_vdw_factor(cgf)
       if(1.eq.0) then
          call MR3calccoulomb_vdw_ij_exlist( &
               NATIM4,posgrp,cg,iacsor,rslgrp(1,istart), &
               NATOM4,posgrp,cg,iacsor, &
               IDIST,fgscale,gscale,rscale,kappa, &
               modop,xmax,ene_c,ene_v, &
               idummy,potflag,idummy,numex,natex)
       else 
          rcut=CTOFNB
          skinnb=0.0d0
          ciflag=0
#if KEY_PARALLEL==1
          ciflag=ciflag+2           
#endif
          modop1=modop+1
          do i=1,3
             vol(i)=xucell(i)
             ldim(i)=int(vol(i)/(rcut+skinnb))
          enddo
          !write(*,*) 'volume is ',vol(1),vol(2),vol(3)
          !write(*,*) 'ldim is ',ldim(1),ldim(2),ldim(3)
          !write(*,*) 'rcut ',rcut,' skinnb ',skinnb

          ! Choosing the parallel method...
          ! do we need to ever nullify() these pointers ???
          istartp=1
#if KEY_PARALLEL==1
          if(qpargblock) then
             nparticles=natim4
             if(istart > 1) istartp = numexap(istart-1)+1
          else if(qpargintrlv.or.qpargspac) then
             nparticles=npt
             pgpuposgrp => pposgrp
             pgpucg => pcggrp
             pgpuiacsor => piacsor
             pgpunumex => numexpd
             pgpunatex => natexpd
          else   ! not parallel
#endif 
             nparticles=natim4
             cgloc(1:nparticles)=cg(1:nparticles)
             pgpuposgrp => posgrp
             pgpucg => cgloc
             pgpuiacsor => iacsor
             pgpunumex => numex
             pgpunatex => natex
#if KEY_PARALLEL==1
          endif                                      
#endif
          !
          if(qene.and.qgpurs) then
             call MR3calccoulomb_vdw_ij_ci_exlist( &
                  nparticles,pgpuposgrp,pgpucg,pgpuiacsor,rslgrp(1,istart), &
                  NATOM4,posgrp,cg,iacsor,IDIST,gscale,rscale,kappa, &
                  modop1,rcut,skinnb,vol,ldim,pgpunumex,pgpunatex,ciflag)

             !do i = 1,npart
             !   write(*,*)mynod,rslgrp(1,i),rslgrp(2,i)
             !   write(*,*)mynod,numexp(i),pposgrp(1,i),pposgrp(2,i),pposgrp(3,i)
             !   write(*,*)mynod,numex(i),posgrp(1,i),posgrp(2,i),posgrp(3,i)
             !enddo
             !do i = 1,2*npart
             !   write(*,*)mynod,natex(i),natexp(i)
             !enddo
             
             ene_c = zero
             ene_v = zero
             vdw_im = zero
             ele_im = zero
#if KEY_PARALLEL==1
#if 0 /* parallel-block */
!             if (iend > natom) then            
!                do i = istart,iend             
#endif
                !the new code is the same for all: block,interleave,spacial
             do i = 1,npt
                iatom=mapgpu(i)
                if (iatom>natom) then
                   vdw_im = vdw_im + half*rslgrp(2,i)
                   ele_im = ele_im + half*rslgrp(1,i)
                endif
             enddo
!             endif ! to finish parallel-block above
#else /**/
             do i = natom+1,natim
                vdw_im = vdw_im + half*rslgrp(2,i)
                ele_im = ele_im + half*rslgrp(1,i)
                do j=1,2  ! was 3
                   rslgrp(j,bimag%imattr(i)) = rslgrp(j,bimag%imattr(i)) + rslgrp(j,i)
                   rslgrp(j,i) = zero
                enddo
             enddo
#endif 
             !write(outu,*)'EWGRAP>me,Image VDW,ELE = ',mynod,vdw_im,ele_im
#if KEY_PARALLEL==1
             do i=1,npt
                ene_c = ene_c + rslgrp(1,i)
                ene_v = ene_v + rslgrp(2,i)
             enddo
#else /**/
             do i=1,natom
                ene_c = ene_c + rslgrp(1,i)
                ene_v = ene_v + rslgrp(2,i)
             enddo
#endif 
             !write(*,*)'EWGRAP>me,ene_c,ene_v = ',mynod,ene_c, ene_v
             rslgrp(1:3,1:npart)=zero
          endif   ! end for energy calculation
          !
          ! next call is for overlap of force calculations
          ciflag=ciflag+64
          if(qgpurs) &
               call MR3calccoulomb_vdw_ij_ci_exlist( &
               nparticles,pgpuposgrp,pgpucg,pgpuiacsor,rslgrp(1,istart), &
               NATOM4,posgrp,cg,iacsor,IDIST,fgscale,rscale,kappa, &
               modop,rcut,skinnb,vol,ldim,pgpunumex,pgpunatex,ciflag)
       endif
    else if (mod(igrape,10) == 2) then
       idummy=0
       ene_c = zero
       ene_v = zero
       call MR3_set_coulomb_vdw_factor(cgf)
       call MR3calccoulomb_vdw_ij_exlist_host( &
            NATOM4,posgrp,cg,iacsor,rslgrp(1,istart), &
            NATOM4,posgrp,cg,iacsor, &
            IDIST,fgscale,gscale,rscale,kappa, &
            modop,xmax,ene_c,ene_v, &
            idummy,potflag,idummy,numex,natex)
    endif
    !         write(*,*) 'After MR3calccoulomb_vdw_exlist'
    !write(outu,*)'fmm is here-1? ene_c,ene_v,eel,enb=',ene_c,ene_v,eel,enb
    if(.not.(mod(igrape,10)==0)) ene_c = half*ene_c
    ene_v = half*ene_v
    eel = ene_c*HALF    ! why half here ?? 
    enb = ene_v*HALF
    !write(outu,*)'fmm is here-2? ene_c,ene_v,eel,enb=',ene_c,ene_v,eel,enb
    IF(QENE) THEN
       GRPELE=EEL
       ! we keep eel, since we report just one number from GPU
       !EEL=ZERO
    ENDIF
    IF(QGRIM) THEN
       GRPVDW=ENB
       ! we keep enb, since we report just one number from GPU
       !ENB=ZERO
    ENDIF

    if(mod(igrape,10) == 2) then
       DO I=1,NATOM
          DX(I)=DX(I)-rslgrp(1,I)
          DY(I)=DY(I)-rslgrp(2,I)
          DZ(I)=DZ(I)-rslgrp(3,I)
       ENDDO
    endif

    !
    TELAPS=eclock()-TSTART
    MDGLJ=MDGLJ+TELAPS
    !
    !     The following can be overlapped with the GRAPE code
    !
    !     K space - use PME from host
    !     For CUDA library it is decided in PME code which one is used
    !     if igrap/10 == 1 then use GPU else use host.

    !
    !     Add correction:
    if(lewald.and.qgpurs.and.(.not.lfmm)) &
         CALL HEWEXC(X,Y,Z,CGF,DX,DY,DZ,ETERM(EWEXCL),INB14,IBLO14)
    !write(*,*)'EWGRAP>ewexcl=',eterm(ewexcl), ' mynodg=',mynodg,' qgpuks=',qgpurs
!
!     Sometimes 1-4 vdw parameters are different than the rest. For
!     this to work we need to include 1-4 VDW. But we do this by
!     extending numex,natex arrays for 1-4 interaction. Then we need
!     to do also electrostatic, since we dont want separate
!     numex&natex for electrostatics and vdw!!
!
!     Shift is currently the only one supported for cutoff
!
!     We need to work on this further....
!
!     This is now always called so the code is good only for shift! But
!     so is the library that we link with !!!
!
!     For now this is OK, but send some control to the library, ie is it
!     shift or nocutoff. In case of shift send the ctofnb value!
!
!
      tstart=eclock()
      eel_14=zero
      enb_14=zero
      llelecx=.true.
      llvdwx=.true.
      lifrsta=1
#if KEY_PARALLEL==1
      if(qgpurs) &                 
#endif
      call nbadd14(x,y,z,dx,dy,dz,cgf,eel_14,enb_14,inb14,iblo14,ioff)
      ! the following call didn't work very well
      ! It may need special versions of inb14,iblo14, and iacnb ???
      ! Currently it produces some huge numbers
      !call enbaexp(enb_14,eel_14,llelecx,llvdwx,lifrsta,natom, &
      !cg,inb14,iblo14,iacnb,llused)
      ! what about ENBFS8 ??

      !write(*,*)'nbadd14>qgpurs,cgf,eel,enb=',qgpurs,cgf,eel_14,enb_14
      fmmgrms=zero
      do i=1,natom
         fmmgrms=fmmgrms+dx(i)**2+dy(i)**2+dz(i)**2
      enddo
      fmmgrms=sqrt(fmmgrms/natom/three)
      !write(*,*)'nbadd14>total? grms=',fmmgrms
      !write(*,*)'nbadd14>total? dx(1-100)=',dx(1:100)
      !write(*,*)'nbadd14>eel,enb=',eel_14,enb_14
      ene_c=ene_c+eel_14
      ene_v=ene_v+enb_14
      eel=eel+eel_14
      enb=enb+enb_14
      telaps=eclock() - tstart
      mdg14 = mdg14 + telaps

!!*!!$OMP SECTION
    TSTART=eclock()
    ! Because of the pressure calculations and because the kspace 
    ! calculations are overlapped with rspace calculations we need
    ! local arrays here....
    if(mod(igrape,10) == 3) then
       ksdx(1:natom)=zero
       ksdy(1:natom)=zero
       ksdz(1:natom)=zero
       if(lewald.and.qgpuks) &
            CALL KSPACE(ETERM(EWKSUM),ETERM(EWSELF),ETERM(EWQCOR), &
            ETERM(EWUTIL),QETERM(EWKSUM),QETERM(EWSELF), &
            QETERM(EWQCOR),QETERM(EWUTIL), &
            X,Y,Z,ksDX,ksDY,ksDZ,NATOM,CG,CGTOT &
#if KEY_FLUCQ==1
            ,QFLUC,FQCFOR         & 
#endif
            )
    else
       if(lewald.and.(.not.lfmm)) &
            CALL KSPACE(ETERM(EWKSUM),ETERM(EWSELF),ETERM(EWQCOR), &
            ETERM(EWUTIL),QETERM(EWKSUM),QETERM(EWSELF), &
            QETERM(EWQCOR),QETERM(EWUTIL), &
            X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT &
#if KEY_FLUCQ==1
            ,QFLUC,FQCFOR         & 
#endif
            )
    endif

    !write(outu,'(a,9f12.5)')'ewgrap>after kspace: ewvirial=',ewvirial

    !
    TELAPS=eclock()-TSTART
    MDGKS=MDGKS+TELAPS
!!*!!$OMP END SECTIONS
!!*!!$OMP END PARALLEL
    !
    if( (mod(igrape,10) /= 2) .or. (mod(igrape,10) /= 0) ) then
       TSTART=eclock()
       if(qgpurs) &
            call MR3_get_forces_overlap(rslgrp(1,istart))
! The special image handling is done in call transi()
!  at the end of energy() routine...
#if KEY_PARALLEL==1
       if (qgpurs) then
#endif
          ipt=1
          DO I=1,npart
#if KEY_PARALLEL==1
             if (gpumap(i)>0) then
#endif
                DX(I)=DX(I)-rslgrp(1,Ipt)
                DY(I)=DY(I)-rslgrp(2,Ipt)
                DZ(I)=DZ(I)-rslgrp(3,Ipt)
                ipt=ipt+1
#if KEY_PARALLEL==1
             endif
#endif
          ENDDO
#if KEY_PARALLEL==1
       endif
#endif
       TELAPS=eclock()-TSTART
       MDGRS=MDGRS+TELAPS
    endif

    call cpugpusplitclear

    igrape=orig_igrape ! FMM trickery

    RETURN

    !
    ! OLD, kept here for easy comparison&debugging
    !
    !     K space Ewald sum
    !
    !        Prepare K vectors:
    !
    B = ONE/FOUR/KAPPA/KAPPA
    VFACT = EIGHT*PI/(XUCELL(1)**3)
    IPT = 0
    IF(KMAXX > 14)CALL WRNDIE(-5,'<EWGRAP>', &
         'Maximum number of K vectors exceeded.')
    DO KX = 0, KMAXX
       RKX = TWOPI*KX/XUCELL(1)
       IF(KX == 0) THEN
          KSY = 0
       ELSE
          KSY = -KMAXY
       ENDIF
       DO KY = KSY, KMAXY
          RKY = TWOPI*KY/XUCELL(1)
          IF(KX == 0.AND.KY.EQ.0) THEN
             KSZ = 1
          ELSE
             KSZ = -KMAXZ
          ENDIF
          DO KZ = KSZ, KMAXZ
             RKZ = TWOPI*KZ/XUCELL(1)
             KSQ = KX*KX + KY*KY + KZ*KZ
             IF (KSQ <= KSQMAX.AND. KSQ /= 0) THEN
                IPT = IPT + 1
                RKSQ = RKX*RKX + RKY*RKY + RKZ*RKZ
                KVEC(IPT) = VFACT*EXP((-B*RKSQ))/RKSQ
                KVECIT(1,IPT) = KX
                KVECIT(2,IPT) = KY
                KVECIT(3,IPT) = KZ
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    IF(IPT > MAXKVEC)CALL WRNDIE(-5,'<ELGRAP>','K vector overflow.')
    !     K space forces & energy:
    MODOP=9
    DO I=1,3
       DO J=1,3
          CELL(I,J)=ZERO
          IF(I == J)CELL(I,J)=XUCELL(1)
       ENDDO
    ENDDO
    !
    singrp(1:3,1:natom) = zero
    tpot=zero
#if 1
    CALL MR3calcewald & 
#endif
#if 0
    CALL MR1calcewald & 
#endif
         (KVECIT,IPT,posgrp,NATOM,CG,KAPPA,pi4,cell, &
         singrp,tpot,stress)
    ESELF=ZERO
    !SIGNPROBLEM
    DO I=1,NATOM
       tdx=-cgf*singrp(1,I)
       tdy=-cgf*singrp(2,I)
       tdz=-cgf*singrp(3,I)
       dx(i)=dx(i)+tdx
       dy(i)=dy(i)+tdy
       dz(i)=dz(i)+tdz
       ESELF=ESELF+CG(I)**2
    ENDDO
    ETERM(EWSELF)=-KAPPA*CCELEC/SQRT(PI)*ESELF
    EKEL=ZERO
    EKEL=EKEL+cgf*tpot
    !
    ETERM(EWKSUM)=EKEL
    !
    RETURN
  contains
    real(chm_real) function cmin(a)
      real(chm_real) a(:),big
      integer i,n
      n=size(a)
      big=10.0**10
      do i = 1, n
         if (a(i)<big) big=a(i)
      enddo
      cmin = big
      return
    end function cmin
    real(chm_real) function cmax(a)
      real(chm_real) a(:),big
      integer i,n
      n=size(a)
      big=-10.0**10
      do i = 1, n
         if (a(i)>big) big=a(i)
      enddo
      cmax = big
      return
    end function cmax
  END SUBROUTINE EWGRAP
  !
  SUBROUTINE ELJGRP(X,Y,Z,CGF,DX,DY,DZ,EEL,ENB,QENE)
    !----------------------------------------------------------------------
    !     This is the version of the nonboned energy terms for the
    !      GRAPE (GRAvity PipE) machine
    !
    use chm_kinds
    use number
    use dimens_fcm
    use psf
    use param
    use inbnd, only: e14fac, ctofnb
    use consta
    use grape
    use parallel
    !
    implicit none
    !C      IMPLICIT NONE
    !
    real(chm_real) X(*),Y(*),Z(*),CGF,DX(*),DY(*),DZ(*),EEL,ENB
    LOGICAL QENE
    !
    real(chm_real) FMAX,XMAX,DUMMY
    INTEGER I,J,MODOP,OVFLOW,IPTR,istart,iend
    real(chm_real) ERSCALE
    real(chm_real) fe(3,MAXA)
    real(chm_real) ene_c, ene_v
    integer*4 natom4,TBLNO,idummy,potflag,flagregexlist,npt,tblno2
    !
    natom4=natom
    !
    ERSCALE=ONE
    FMAX=MEGA
    XMAX=THOSND
    DUMMY=ONE
    istart=1
    !
    DO I=1,NATOM
       posgrp(1,i)=x(i)
       posgrp(2,i)=y(i)
       posgrp(3,i)=z(i)
       rslgrp(1,i)=zero
       rslgrp(2,i)=zero
       rslgrp(3,i)=zero
    ENDDO
#if KEY_PARALLEL==1
    istart=mynod*((natom+numnod-1)/numnod)+1
    iend=mynodp*((natom+numnod-1)/numnod)
    if(iend > natom)iend=natom
    IPTR=1
    DO I=istart,iend
       pposgrp(1,iptr)=x(i)
       pposgrp(2,iptr)=y(i)
       pposgrp(3,iptr)=z(i)
       pcggrp(iptr)=cgf*cg(i)
       piacsor(iptr)=iacsor(i)
       iptr=iptr+1
    ENDDO
    NPT=IPTR-1
#endif 
    !
    !     Check if we need energy?
    potflag=0
    IF(QENE) potflag=1
    !
#if 1
    !      call MR3calcvdw_ij_emu(
    !     $     NATOM, posgrp, iacsor, rslgrp,
    !     $     NATOM, posgrp, iacsor,
    !     $     idist,
    !     $     fgscale,
    !     $     rscale, tblno, xmax, 0 )

    !C      write(*,*)'ELJGRP>igrape=',igrape
    ene_c=ZERO
    ene_v=ZERO
    flagregexlist=2+256 
    flagregexlist=flagregexlist+64
    !      flagregexlist=0
    TBLNO=0
    !
    idummy=0
    call MR3_set_rcut2(CTOFNB*CTOFNB);
    if (mod(igrape,10) == 2) then
       ! this may work only in parallel ????
#if 1
       call MR3_set_coulomb_vdw_factor(cgf)
#endif 
       call MR3calccoulomb_vdw_ij_exlist_host( &
#if 0
            NATOM4,posgrp,pcggrp,iacsor,rslgrp(1,istart),        & 
#endif
#if 1
            NATOM4,posgrp,cg,iacsor,rslgrp(1,istart),         & 
#endif
            NATOM4,posgrp,cg,iacsor, &
            IDIST,fgscale,gscale,rscale,erscale, &
            TBLNO,xmax,ene_c,ene_v, &
            idummy,potflag,idummy,numex,natex)
       eel = ene_c*HALF
       enb = ene_v*HALF
    endif
    !
    if (mod(igrape,10) == 1) then
       call MR3_set_coulomb_vdw_factor(cgf)
       call MR3calccoulomb_vdw_exlist(NATOM4,posgrp,cg,iacsor,IDIST, &
            rslgrp(1,istart),erscale,fgscale,gscale,rscale, &
            TBLNO,numex,natex,xmax,ene_c,ene_v, &
            BCKPTR,potflag,flagregexlist)
       call MR3_get_forces_overlap(rslgrp(1,istart))
       eel = ene_c*HALF
       enb = ene_v*HALF
    endif

    if(mod(igrape,10) == 0) then
       TBLNO=0
       call MR3calccoulomb_ij_exlist( &
            NATOM4,posgrp,cg, rslgrp,  &
            NATOM4,posgrp,cg, erscale, &
            tblno, xmax, idummy, &
            numex, natex )
       do i=1, natom
          do j=1, 3
             rslgrp(j,i) = rslgrp(j,i) * cg(i) * cgf
          enddo
       enddo
       !     write(*,*) sum(1), sum(2), sum(3)

       TBLNO=2
       call MR3calcvdw_ij_exlist( &
            NATOM4,posgrp,iacsor, &
            rslgrp(1,istart),NATOM4,posgrp,iacsor, &
            IDIST,fgscale,rscale, &
            tblno, xmax, idummy, &
            numex, natex )

       if (QENE) then
          DO I=1,NATOM
             fe(1,i)=zero
             fe(2,i)=zero
             fe(3,i)=zero
          ENDDO
          TBLNO=1
          call MR3calccoulomb_ij_exlist( &
               NATOM4,posgrp,cg,fe, &
               NATOM4,posgrp,cg, erscale, &
               tblno, xmax, idummy, &
               numex, natex )
          do i=1, natom
             eel = eel + fe(1,i) * cg(i) * cgf
             !     write(*,*) fe(1,i), cg(i), eel
          enddo

          DO I=1,NATOM
             fe(1,i)=zero
             fe(2,i)=zero
             fe(3,i)=zero
          ENDDO
          TBLNO=3
          call MR3calcvdw_ij_exlist( &
               NATOM4,posgrp,iacsor, &
               fe(1,1),NATOM4,posgrp,iacsor, &
               IDIST,gscale,rscale, &
               tblno, xmax, idummy, &
               numex, natex )
          do i=1, natom
             enb = enb + fe(1,i)
          enddo

          eel = eel*HALF 
          enb = enb*HALF 

       endif
    endif
#else /**/
    !C-CC      call MR3calccoulomb_vdw_exlist(
    !C-CC     $     NATOM,posgrp,cg,iacsor,idist,
    !C-CC     $     rslgrp(1,istart), erscale,
    !C-CC     $     fgscale,gscale,rscale,
    !C-CC     $     numex, natex, 
    !C-CC     $     xmax,EEL,ENB,bckptr,1, 258)
    !C-CC
    !C-CCc      call MR3calccoulomb_vdw_ij_exlist(
#if 1
    !C-CCc     $     NATOM,posgrp,cgcgf,iacsor,                             
#endif
#if 0
    !C-CCc     $     NPT,pposgrp,pcg,piacsor,                            
#endif
    !C-CCc     $     rslgrp(1,istart),NATOM,posgrp,cg,iacsor,
    !C-CCc     $     IDIST,fgscale,gscale,rscale,
    !C-CCc     $     erscale, tblno,
    !C-CCc     $     xmax,EEL,ENB,0,potflag,2,
#if 1
    !C-CCc     $     numex, natex )                                   
#endif
#if 0
    !C-CCc     $     numexp, natexp )                                 
#endif
    !C-CC##ENDIF
    !C-CC##ELSE
    tblno=2
    call MR1calccoulomb_vdw_ij( &
#if 1
         NATOM4,posgrp,cg,iacsor,                              & 
#endif
#if 0
         NPT,pposgrp,pcggrp,piacsor,                             & 
#endif
         rslgrp(1,istart),NATOM4,posgrp,cg,iacsor, &
         IDIST,fgscale,gscale,rscale, &
         xmax,EEL,ENB,idummy,potflag,tblno)

    !
    tblno=500
#if 0
    tblno2=2                                                     
#endif
#if 1
    tblno2=0                                                     
#endif
    call MR1calccoulomb_nlist_emu(posgrp,natom4,cg,one, &
#if 1
         tblno,xmax,tblno2,numex,natex,-cgf,rslgrp)                 & 
#endif
#if 0
         tblno,xmax,tblno2,numexp,natexp,-cgf,rslgrp)              
#endif
    !
    tblno=502
    CALL MR1calcvdw_nlist_emu2(posgrp,NATOM4,IACSOR,IDIST,FGSCALE,FRSCALE, &
#if 1
         tblno,xmax,tblno2,numex,natex,-ONE,rslgrp)    & 
#endif
#if 0
         tblno,xmax,tblno2,numexp,natexp,-ONE,rslgrp)  
#endif
#endif 
    !
    DO I=1,NATOM
       dx(i)=dx(i)-rslgrp(1,i)
       dy(i)=dy(i)-rslgrp(2,i)
       dz(i)=dz(i)-rslgrp(3,i)
    ENDDO
    !
#if 0
    IF (QENE) THEN
       !
       DO i=1,natom
          rslgrp(1,i)=zero
          rslgrp(2,i)=zero
          rslgrp(3,i)=zero
       enddo
       !
#if 1
       tblno=500                                                       
#endif
#if 0
       tblno=501                                                       
#endif
#if 0
       tblno2=2                                                        
#endif
#if 1
       tblno2=0                                                        
#endif
       call MR1calccoulomb_nlist_emu(posgrp,natom4,cg,one, &
#if 1
            tblno,xmax,tblno2,numex,natex,-cgf,rslgrp)                    & 
#endif
#if 0
            tblno,xmax,tblno2,numexp,natexp,-cgf,rslgrp)                 
#endif
       !     
       DO i=1,natom
          EEL=EEL+rslgrp(1,i)
       enddo
       !
       DO i=1,natom
          rslgrp(1,i)=zero
          rslgrp(2,i)=zero
          rslgrp(3,i)=zero
       enddo
       !
#if 1
       tblno=502                                                       
#endif
#if 0
       tblno=503                                                       
#endif
#if 0
       tblno2=2                                                        
#endif
#if 1
       tblno2=0                                                        
#endif
       CALL MR1calcvdw_nlist_emu2(posgrp,NATOM4,IACSOR,IDIST, &
            GSCALE,RSCALE, &
#if 1
            tblno,xmax,tblno2,numex,natex,-ONE,rslgrp)                    & 
#endif
#if 0
            tblno,xmax,tblno2,numexp,natexp,-ONE,rslgrp)                 
#endif
       !
       DO i=1,natom
          ENB=ENB+rslgrp(1,i)
       enddo
       !
       EEL=EEL*HALF
       ENB=ENB*HALF
       !
    ENDIF
#endif 
    !
    RETURN
  END SUBROUTINE ELJGRP
  !
#if 0
  subroutine MDVIS_INIT(x,y,z)
    !
    use chm_kinds
    use number
    use stream
    use dimens_fcm
    use psf
    use rtf,only:atct
    use param
    use grape
    use parallel
    !
    implicit none
    !
    real(chm_real) x(*),y(*),z(*)
    integer str_len

    MDVPORT=6100
    str_len=51
    call MR3_sock_init_charmm(MDVSOCK, MDVPORT, x,y,z, &
         ATCT, IAC, NATOM, &
         str_len, RES, IBASE, NRES)
  end subroutine MDVIS_INIT
#endif /* */
  !
  subroutine grpprn(n,a)
    !----------------------------------------------------------------------
    use chm_kinds
    use stream
    implicit none
    integer n,i,j,k,rest,num
    real(chm_real) a(n,n)
    !
    num=8
    do i=1,n,num
       rest=num
       if(i+num-1 > n) rest=n-i+1
       write(outu,*)i
       do j=1,n
          write(outu,'(10f10.4)')(a(k,j),k=i,i+rest-1)
       enddo
    enddo
  end subroutine grpprn
  !
#else /*  (mdgrape)*/
contains
  SUBROUTINE NULL_GRAPE
    RETURN
  END SUBROUTINE NULL_GRAPE
#endif /* (mdgrape)*/
#if KEY_PARALLEL==0
! No FMM if not parallel
    subroutine fmm_init()
      return
    end subroutine fmm_init
    subroutine fmm_partition(n,ic,x,q,v,s)
      use chm_kinds
      integer n, ic(:)
      real(chm_real) x(:,:), q(:), v(:), s
      return
    end subroutine fmm_partition
    subroutine fmm(n,ic,x,q,e,f,s)
      use chm_kinds
      integer n, ic(:)
      real(chm_real) x(:,:), q(:), e(:), f(:,:), s
      return
    end subroutine fmm
#endif
  !
#if KEY_LIBGRAPE==1
  subroutine mr3init
    return
  end subroutine mr3init
  subroutine mr3settable_nf(pot,tpot,one,npot)
    character(len=100) pot
    integer*4 tpot,one,npot
    return
  end subroutine mr3settable_nf
  subroutine mr3free()
    return
  end subroutine mr3free
  subroutine mr3_setup_exlist(l,a)
    integer*4 l,a(:)
    return
  end subroutine mr3_setup_exlist
  subroutine mr3calccoulomb_ij_exlist(&
       ni,c1,qi,r,&
       nj,c2,qj,ers,&
       t,xm,f,&
       nux,nax)
    integer*4 ni,nj,idum,idum0,t,id,f,nux(*),nax(*)
    real(chm_real), dimension(:) :: qi,qj
    real(chm_real), dimension(:,:) :: c1,c2,r
    real(chm_real) xm,ec,ev,ers
    return
  end subroutine mr3calccoulomb_ij_exlist
  subroutine mr3calccoulomb_vdw_exlist( &
       ni,c1,qi,ia,id,r,&
       ers,fgs,gs,rs,&
       t,nux,nax,xm,ec,ev,&
       idum,f,idum0)
    integer*4 ni,idum(*),idum0,t,ia(*),id,f,nux(*),nax(*)
    real(chm_real), dimension(:) :: qi,fgs,gs,rs
    real(chm_real), dimension(:,:) :: c1
    real(chm_real) xm,ec,ev,r,ers
    return
  end subroutine mr3calccoulomb_vdw_exlist
  subroutine mr3calccoulomb_vdw_ij_exlist( &
       ni,c1,qi,ia,r,&
       nj,c2,qj,ib,&
       id,fgs,gs,rs,ers,&
       t,xm,ec,ev,&
       idum,f,idum0,nux,nax)
    integer*4 ni,nj,idum,idum0,t,ia(*),ib(*),id,f,nux(*),nax(*)
    real(chm_real), dimension(:) :: qi,qj,fgs,gs,rs
    real(chm_real), dimension(:,:) :: c1,c2
    real(chm_real) xm,ec,ev,r,ers
    return
  end subroutine mr3calccoulomb_vdw_ij_exlist
  subroutine mr3calccoulomb_vdw_ij_exlist_host( &
       ni,c1,qi,ia,r,&
       nj,c2,qj,ib,&
       id,fgs,gs,rs,ers,&
       t,xm,ec,ev,&
       idum,f,idum0,nux,nax)
    integer*4 ni,nj,idum,idum0,t,ia(*),ib(*),id,f,nux(*),nax(*)
    real(chm_real), dimension(:) :: qi,qj,fgs,gs,rs
    real(chm_real), dimension(:,:) :: c1,c2
    real(chm_real) xm,ec,ev,r,ers
    return
  end subroutine mr3calccoulomb_vdw_ij_exlist_host
  subroutine mr3calcvdw_ij_exlist( &
       ni,c1,ia,r,&
       nj,c2,ib,&
       id,fgs,gs,&
       t,xm,f,nux,nax)
    integer*4 ni,nj,idum,idum0,t,ia(*),ib(*),id,f,nux(*),nax(*)
    real(chm_real) fgs(*),gs(*)
    real(chm_real), dimension(:,:) :: c1,c2
    real(chm_real) xm,ec,ev,ers,r
    return
  end subroutine mr3calcvdw_ij_exlist
  subroutine mr3calccoulomb_ij_exlist_host
    return
  end subroutine mr3calccoulomb_ij_exlist_host
  subroutine MR3calccoulomb_vdw_ij_ci_exlist( &
       nparticles,pgpuposgrp,pgpucg,pgpuiacsor,rslgrp, &
       NATOM4,posgrp,cg,iacsor,IDIST,gscale,rscale,kappa, &
       modop1,rcut,skinnb,vol,ldim,pgpunumex,pgpunatex,ciflag)
    integer nparticles
    integer*4 natom4, idist, ciflag,modop1,ldim(3),iacsor(*)
    real(chm_real) r, kappa, rslgrp,rcut,skinnb,vol(3)
    real(chm_real), pointer, dimension(:,:) :: pgpuposgrp
    real(chm_real), pointer, dimension(:) :: pgpucg
    integer,pointer, dimension(:) :: pgpuiacsor,pgpunatex,pgpunumex
    real(chm_real), dimension(:,:) :: posgrp
    real(chm_real), dimension(:) :: cg,gscale,rscale
    return
  end subroutine MR3calccoulomb_vdw_ij_ci_exlist
  !
  !  some more routines
  !
  subroutine mr3_set_coulomb_vdw_factor(s)
    real(chm_real) s
    return
  end subroutine mr3_set_coulomb_vdw_factor
  subroutine mr3_get_forces_overlap(f)
    real(chm_real) f
    return
  end subroutine mr3_get_forces_overlap
  !C
  subroutine m1presetmode()
    return
  end subroutine m1presetmode
  subroutine m1setautoscale()
    return
  end subroutine m1setautoscale
  subroutine m1setxjn()
    return
  end subroutine m1setxjn
  subroutine m1setqxn()
    return
  end subroutine m1setqxn
  subroutine m1setcjn()
    return
  end subroutine m1setcjn
  subroutine m1free()
    return
  end subroutine m1free
  subroutine m1setconstb()
    return
  end subroutine m1setconstb
  subroutine m1setcellsize()
    return
  end subroutine m1setcellsize
  subroutine m1setgrscalen()
    return
  end subroutine m1setgrscalen
  subroutine mr1init()
    return
  end subroutine mr1init
  subroutine mr1free()
    return
  end subroutine mr1free
  subroutine mr1calccoulomb_ij()
    return
  end subroutine mr1calccoulomb_ij
  subroutine mr1calccoulomb_vdw_ij()
    return
  end subroutine mr1calccoulomb_vdw_ij
  subroutine mr1calccoulomb_nlist_emu()
    return
  end subroutine mr1calccoulomb_nlist_emu
  subroutine mr1calcvdw()
    return
  end subroutine mr1calcvdw
  subroutine mr1calcvdw_ij()
    return
  end subroutine mr1calcvdw_ij
  subroutine mr1calcewald()
    return
  end subroutine mr1calcewald
  subroutine mr1calcvdw_nlist_emu2()
    return
  end subroutine mr1calcvdw_nlist_emu2
  subroutine mr1settable_emu_nf()
    return
  end subroutine mr1settable_emu_nf
  real(chm_real) function derfc(x)
    real(chm_real) x
    derfc=1.0
    return
  end function derfc
  real(chm_real) function derf(x)
    real(chm_real) x
    derf=1.0
    return
  END function derf
  subroutine mr3_set_rcut2(c)
    real(chm_real) c
    return
  end subroutine mr3_set_rcut2
  subroutine mr3_set_ldim(l)
    integer*4 l(3)
  end subroutine mr3_set_ldim
  subroutine MR3_set_vdwmode(vdwmode)
    integer*4 vdwmode
    return
  end subroutine MR3_set_vdwmode
  subroutine MR3_set_cuton(CT)
    real(chm_real) ct
    return
  end subroutine MR3_set_cuton

#endif 

end module grapemod

#if KEY_LIBGRAPE==1
  subroutine gpupme_ &
       (numatoms,coor,charge,ftmp,ptmp,nfft1,nfft2,nfft3,alpha, &
       bsp_mod1,bsp_mod2,bsp_mod3,volume,recip,ccelec,&
       forder,m,virial_gpu)
    use chm_kinds
    integer m,numatoms,nfft1,nfft2,nfft3,forder
    real(chm_real) bsp_mod1,bsp_mod2,bsp_mod3
    real(chm_real) coor(*),charge(*),volume,recip,alpha,ccelec,virial_gpu(*)
    real(chm_real) ftmp(*),ptmp(*)
    return
  end subroutine gpupme_
#endif
! circular dependency problem (either stays here or goes in the pme.src)
#if KEY_GRAPE==1
subroutine cuda_pme &
     (volume,alpha,recip,x,y,z,charge,numatoms, &
#if KEY_CHEQ==1
     dch, qcg, &  
#endif
     fx,fy,fz,eer)
  !****************************************************************
  !    this routine calculates Particle Mesh Ewald Summation
  !    on NVIDIA's Graphics Processing Unit (GPU) with CUDA, 
  !    implemented into CHARMM 
  !          by Ryuji Sakamaki, Keio Univ., Japan, Jul-2008
  !             (sakamaki@atwired.com)
  !****************************************************************
  !     Input
  !       numatoms     : number of atoms
  !       volume       : volume of the unit cell
  !       recip        : reciprocal unit cell vectors
  !       x,y,z,charge : atomic coordinates, and charges
  !       alpha        : ewald convergence parameter (kappa)
  !     Output
  !       fx,fy,fz     : atomic forces
  !       dch          : atomic potential energy of reciprocal space
  !       eer          : total potential enery of reciprocal space
  !----------------------------------------------------------------
  use pmeutil
  use consta
  use number
  use ewald_1m,only: ewvirial
  use parallel
  use grape, only: qgpusplit

  implicit none
  integer       , intent(in)    :: numatoms
  real(chm_real), intent(in)    :: volume, alpha, recip(9)
  real(chm_real), intent(in)    :: x(*), y(*), z(*), charge(*)
  real(chm_real), intent(inout) :: fx(*), fy(*), fz(*)
#if KEY_CHEQ==1
  real(chm_real), intent(inout) :: DCH(*)
  logical, intent(in) :: QCG
#endif 
  real(chm_real), intent(out)   :: eer
  real(chm_real)                :: tmp
  real(chm_real), allocatable   :: coor(:,:), ptmp(:), ftmp(:,:)
  real(chm_real)                :: virial_gpu(9) ! added by Ryuji
  integer                       :: i

  ! Cuda PME not parallel :-((
  ! this should be good for all the cases. In the case of split we need to communicate, too
  if(.not.qgpusplit) then
#if KEY_PARALLEL==1
     if(mynodg /= numnodg-1) return   
#endif
  endif

  allocate(coor(3,numatoms),ftmp(3,numatoms),ptmp(numatoms))
  tmp = 0.5d0*(volume**(1.d0/3.d0))
  do i = 1, numatoms
     coor(1,i) = x(i) + tmp
     coor(2,i) = y(i) + tmp
     coor(3,i) = z(i) + tmp
     ftmp(1:3,i) = 0.d0
     ptmp(i)   = 0.d0       
  end do
  !
  virial_gpu=zero
  call gpupme_ &
       (numatoms,coor,charge,ftmp,ptmp,nfft1,nfft2,nfft3,alpha, &
       bsp_mod1,bsp_mod2,bsp_mod3,volume,recip,ccelec,forder,-1,virial_gpu) ! was -1 ???
!  call vggcalcpme_charmm_ &
!       (numatoms,coor,charge,ftmp,ptmp,nfft1,nfft2,nfft3,alpha, &
!       bsp_mod1,bsp_mod2,bsp_mod3,volume,recip,ccelec,forder,-1,virial_gpu) ! was -1 ???
  do i = 1, 9
     ewvirial(i) = virial_gpu(i)*ccelec*volume
  enddo
  !----------------------------------------------------------------
  !       last integer indicates GPU ID.
  !       if you use shell environment variable VG_DEVICEID, set -1.
  !----------------------------------------------------------------
  !ccc   ##ENDIF
  eer = 0.d0
  do i = 1, numatoms
     fx(i) = fx(i) + ftmp(1,i)
     fy(i) = fy(i) + ftmp(2,i)
     fz(i) = fz(i) + ftmp(3,i)
#if KEY_CHEQ==1
     if (QCG .and. charge(i) /= 0.d0) then
        dch(i) = dch(i) + ptmp(i) / charge(i)
     end if
#endif 
     eer = eer + ptmp(i)
  end do
  eer = eer * 0.5d0 / ccelec
  deallocate(coor,ftmp,ptmp)

  return
end subroutine cuda_pme
#endif 

