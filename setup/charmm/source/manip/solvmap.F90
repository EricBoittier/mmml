! Calculates solvation map 
!
! by Wonmuk Hwang, 2014,2017
!
module solvmap
    use chm_kinds
    use dimens_fcm
    use memory
    use psf,only: ndon,nacc,ihd1,iacc ! for hbonds
!    use cheq,only:qcg         !##CHEQ
    use number
    use ctitla
    use stream
    use cvio, only:readcv
    use param_store
    !  use image
    use corsubs,only:pkmass,rotls1,fndu
    use chutil,only:qhavcrd,getres
#if KEY_MPI==1
    use parallel, only: mynod,mynodp,numnod,comm_charmm,psnd4,psnd8
!    use parallel, only: mynod,mynodp,numnod,comm_charmm,mpi_integer_size, &
!       mpi_real8_size
    use mpi
#endif

    implicit none

! common private variables
#if KEY_MPI==0
    integer, private::   mynod=0,mynodp=1,numnod=1
#endif
  integer, private:: idum1,idum2
  integer, private:: nx,ny,nz,npoint ! number of cells in each dir
  integer, private:: ix,iy,iz ! index
  integer, private:: i,j,k,m
  real(chm_real), private:: x0,y0,z0,x1,y1,z1
  real(chm_real), private:: xmin,xmax,ymin,ymax,zmin,zmax
  character(len=4), private:: hdr1='COOR', hdr2='CORD'
  
  integer, private:: nfile,nslct,nsolv,ncoord,nframe,iunit
  integer, private:: nfreat,istep,ndegf,nsavv
  real(chm_real), private:: delta,rdum
  real(chm_real), private:: tol
  logical, private:: error, flag, flag1
  
contains

!-------------------------------------------------------
subroutine slvden(x,y,z,wmain,natom,nunit,firstu,nskip,nbegn,nstop,islct, &
        atype,res,rcut,ounit,outmode,orient,xcomp,ycomp,zcomp,wcomp,&
        lmass,lwmain,lnoro,amass,asolv)
! Calculate solvent density

! argument vars
  real(chm_real) x(*),y(*),z(*),wmain(*)
  integer natom,nunit,firstu,nskip,nbegn,nstop,ounit,outmode
  integer islct(*)
  character(len=*) atype(*)
  real (chm_real) res,rcut
  logical orient,lmass,lwmain,lnoro
  real(chm_real) xcomp(*),ycomp(*),zcomp(*),wcomp(*),amass(*)
  character(len=8) asolv ! solvent atom name 

  ! local vars
  integer, allocatable,dimension(:) :: solvent,solute ! section subset
  real(chm_real4),allocatable,dimension(:) :: itemp
  integer,allocatable,dimension(:) :: ifreat
  real(chm_real),allocatable,dimension(:) :: sdensity,smap_tmp
!  real(chm_real),allocatable,dimension(:) :: CCG        !##CHEQ

  ! For coord orientation
  real(chm_real), allocatable, dimension(:) :: bmass
  integer istats,npr
  integer,allocatable,dimension(:,:) :: atompr

  ! mpi related
  integer chunksize(numnod),offset(numnod),myoffset,ierr
  !integer status(MPI_STATUS_SIZE)
#if KEY_MPI==1  
  call mpi_barrier(COMM_CHARMM,ierr)
#endif

  ! set search box size, etc
  if ( mynod .eq. 0 ) then  
     call get_dimension(x,y,z,natom,nskip,nbegn,nstop,islct, &
        atype,asolv,res,rcut,orient,xcomp,ycomp,zcomp)
     flag1 = .true. ! assume comp coord to exist
  endif
  !nframe=1+(nstop-nbegn)/nskip ! number of frames to use
     
#if KEY_MPI==1
  ! broadcast common parameters
  call mpi_barrier(COMM_CHARMM,ierr)
  call psnd8(xmax,1); call psnd8(ymax,1); call psnd8(zmax,1);
  call psnd8(xmin,1); call psnd8(ymin,1); call psnd8(zmin,1);
  call psnd4(nx,1); call psnd4(ny,1); call psnd4(nz,1);
  call psnd4(npoint,1); call psnd4(nslct,1); call psnd4(nsolv,1);
  call mpi_barrier(COMM_CHARMM,ierr)
#endif

  call chmalloc('solvmap.src','slvden','solute',nslct,intg=solute)
  call chmalloc('solvmap.src','slvden','solvent',nsolv,intg=solvent)
  call chmalloc('solvmap.src','slvden','smap_tmp',npoint,cr8=smap_tmp)
  call chmalloc('solvmap.src','slvden','sdensity',npoint,cr8=sdensity)

#if KEY_MPI==1
  call mpi_barrier(COMM_CHARMM,ierr)
#endif

  ! System setup for the master node
  call chmalloc('solvmap.src','slvden','itemp',natom,cr4=itemp)
!  call chmalloc('solvmap.src','slvden','CCG',natom,crl=CCG)      !##CHEQ
  if(orient) then
     call chmalloc('solvmap.src','slvden','atompr',2,natom,intg=atompr)
     call chmalloc('solvmap.src','slvden','bmass',natom,crl=bmass)
  else
     call chmalloc('solvmap.src','slvden','atompr',1,1,intg=atompr)
     call chmalloc('solvmap.src','slvden','bmass',1,crl=bmass)
  endif
  write(outu,'(A)') 'debug' ! debug


  ! Setup the orient option assuming that the reference structure is 
  !     in the COMP set
  if(orient)then
     npr=0
     do i = 1,natom
        if (islct(i) == 1) then
           npr=npr+1 ! npr: npair
           atompr(1,npr)=i; atompr(2,npr)=i ! atompr: ref atom indices
        endif
     enddo
     call pkmass(amass,amass,bmass,atompr,npr,lmass,lwmain,wmain)
  endif
  j=0;  k=0;  ncoord=0 ! ncoord: number of coord frames
  do i=1,natom ! register selected atom index
     if(islct(i) == 1) then
        j=j+1; solute(j)=i
     endif
     if (atype(i)==asolv) then
        k=k+1; solvent(k)=i
     endif
  enddo
  iunit=firstu; istats=1
  idum1=nsolv/numnod;   idum2=mod(nsolv,numnod)

  write(outu,'(A,I3)') 'numnod= ', numnod ! debug
  
  do i=1,numnod ! uniformly distribute atoms among nodes
     if (i<=idum2) then
        chunksize(i) = idum1+1
     else 
        chunksize(i) = idum1
     endif
  enddo
  sdensity = 0.0 ! initialize all elements of sdensity
  smap_tmp = 0.0 ! temporary array to be used across nodes
  nfreat=natom ! to avoid nfreat from being read (see cvio.src: readcv())
  offset(1) =  1 ;

#if KEY_MPI==1  
  do i = 2, numnod
     j = i -1
     offset(i) = offset(j) + chunksize(j)
  end do
#endif
  
  if (mynod .eq. 0) then 
!     write(OUTU,82) NX,NY,NZ ! debug
!82   format('SMAP: nx= ',I8,' ny= ', I8,' nz= ', I8 ,' cells willl be used.')

     write(outu,80) nslct, nsolv
80   format('SMAP:',I8, ' solute atoms and ', I8, &
          ' solvent atoms used for analysis.')
  endif ! if (mynod .eq. 0)

#if KEY_MPI==1
  call mpi_barrier(COMM_CHARMM,ierr)
#endif

  do while(istats >= 0)   ! Go over the trajectory
     if (mynod .eq. 0) then 
        call readcv(x,y,z, &
#if KEY_CHEQ==1
            (/ ZERO /), .FALSE., &  
#endif
!             ccg,qcg,                        & !##CHEQ
             itemp,natom,ifreat,nfreat, &
             firstu,nunit,iunit,nfile, &
             istep,istats,ndegf,delta, &
             nbegn,nstop,nskip,nsavv,hdr1,hdr2, &
             titleb,ntitlb,.false., (/ zero /), .false.)
        ncoord=ncoord+1
!        write (outu,'(/A,I10)') 'SMAP-SDEN: Working on frame ',ncoord
        if (ncoord == 1) tol = 0.1 / real( nfile )
        if (orient) then
           ! Orient the main coord set with respect to the comp set.
           if (ncoord == 1 .and. .not. qhavcrd(natom,islct,xcomp)) then
              flag1 = .false.
              call wrndie(1,'<SLVDEN>', &
                   'NO COMP SET FOR ORIENT! USING THE FIRST FRAME INSTEAD.')
              do i =1,natom
                 xcomp(i)=x(i); ycomp(i)=y(i); zcomp(i)=z(i)
              enddo
           endif
           if ( flag1 .eqv. .true.) then
              CALL ROTLS1(XCOMP,YCOMP,ZCOMP,X,Y,Z,NATOM, &
                   ATOMPR,NPR,BMASS,.false.,lnoro)
           endif
        ENDIF
        !write (outu,'(A,I6)') 'zz, ncoord= ',ncoord
     endif ! mynod .eq. 0     
#if KEY_MPI==1
     call mpi_barrier(COMM_CHARMM,ierr)
     call psnd4(ncoord,1); call psnd4(istats,1)
     if ( ncoord == 1 ) call psnd8(tol,1)
     call psnd8(x,natom); call psnd8(y,natom); call psnd8(z,natom)
     call mpi_barrier(COMM_CHARMM,ierr) 
#endif

     ! Calculate density on each node
     i = mynodp
     do j=offset(i),offset(i)+chunksize(i)-1
        k=solvent(j); x0=x(k); y0=y(k); z0=z(k)
        flag=.false.
        if ((x0>xmin) .and. (x0<xmax)) then
           if ((y0>ymin) .and. (y0<ymax)) then
              if ((z0>zmin) .and. (z0<zmax)) then           
                 flag=.true.
              endif
           endif
        endif

        if (flag .eqv. .true. ) then ! count density
           ! i{x,y,z} <= 0 : can happen due to movement
           ix=ceiling((x0-xmin)/res); if ( ix .le. 0) cycle 
           iy=ceiling((y0-ymin)/res); if ( iy .le. 0) cycle 
           iz=ceiling((z0-zmin)/res); if ( iz .le. 0) cycle 
           if ((ix > nx) .or. (iy > ny) .or. (iz > nz)) then
              cycle ! This can happen because the molecule moves. Skip.
           endif
           !k = (ix-1)*ny*nz + (iy-1)*nz + iz
           k = (iz-1)*ny*nx + (iy-1)*nx + ix 
           if (k> nx*ny*nz) then ! this should not happen
              write(outu,'(a,3I5)') 'SMAP-SDEN: ix,iy,iz = ',ix,iy,iz
              call wrndie(-2,'<SLVDEN>', &
                   'DENSITY ARRAY INDEX OUT OF RANGE.')
           endif
           smap_tmp(k) = smap_tmp(k) + 1.0
        endif
     enddo
#if KEY_MPI==1     
     call mpi_barrier(COMM_CHARMM,ierr)
#endif     
     !if (ncoord == nframe) exit
  enddo ! do while(istats >= 0)

#if KEY_MPI==1     
  ! combine all contributions to density
  call mpi_barrier(COMM_CHARMM,ierr)
  !  call mpi_allreduce(smap_tmp,sdensity,npoint,mpi_double_precision, &
  call mpi_allreduce(smap_tmp,sdensity,npoint,mpi_double_precision, &
       MPI_SUM,COMM_CHARMM,ierr)
  call mpi_barrier(COMM_CHARMM,ierr)
#else
  do k=1,npoint
     sdensity(k)=smap_tmp(k)
  enddo
#endif  
  
 ! Calculate and write out density
  if ( mynod .eq. 0 ) then
     do k = 1, npoint
        !if (sdensity(k) < tol) then
        !   sdensity(k) = -9999.0
        !else
        sdensity(k) = sdensity(k)/(dble(ncoord)* res**3)
        !endif
     enddo
     call write_data(sdensity,ounit,outmode,res,rcut,'density (A^-3)')

     !call vclose(ounit,'KEEP',error)
     if (flag1 .eqv. .false.) then ! reset comp set
        xcomp(1:natom) = anum; ycomp(1:natom) = anum; zcomp(1:natom) = anum;
     endif
  endif !  if ( mynod .eq. 0 ) then
        
  call chmdealloc('solvmap.src','slvden','itemp',natom,cr4=itemp)
!  call chmdealloc('solvmap.src','slvden','CCG',natom,crl=CCG)      !##CHEQ
  if(orient) then
     call chmdealloc('solvmap.src','slvden','atompr',2,natom,intg=atompr)
     call chmdealloc('solvmap.src','slvden','bmass',natom,crl=bmass)
  else
     call chmdealloc('solvmap.src','slvden','atompr',1,1,intg=atompr)
     call chmdealloc('solvmap.src','slvden','bmass',1,crl=bmass)
  endif

  call chmdealloc('solvmap.src','slvden','solute',nslct,intg=solute)
  call chmdealloc('solvmap.src','slvden','solvent',nsolv,intg=solvent)
  call chmdealloc('solvmap.src','slvden','smap_tmp',npoint,cr8=smap_tmp)
  call chmdealloc('solvmap.src','slvden','sdensity',npoint,cr8=sdensity)
end subroutine slvden

!-------------------------------------------------------
subroutine slv_hbond(x,y,z,wmain,natom,nunit,firstu,nskip,nbegn,nstop,islct, &
        atype,res,rcut,ounit,oslv,oslt,otot,outmode,orient,xcomp,ycomp,zcomp,&
        wcomp,lmass,lwmain,lnoro,amass,cg,cuthb,dcut,ibase,nres,asolv)
! Calculate number of hbonds

! argument vars
  real(chm_real) x(*),y(*),z(*),wmain(*),cg(*)
  integer natom,nunit,firstu,nskip,nbegn,nstop
  integer ounit,oslv,oslt,otot,outmode,nres
  integer islct(*),ibase(*)
  character(len=*) atype(*)
  real (chm_real) res,rcut,cuthb,dcut
  logical orient,lmass,lwmain,lnoro
  real(chm_real) xcomp(*),ycomp(*),zcomp(*),wcomp(*),amass(*)
  character(len=8) asolv ! solvent atom name   

! local vars
  real x2,y2,z2
  integer, allocatable,dimension(:) :: solute 
  integer, allocatable,dimension(:,:) :: solvent
  integer, allocatable,dimension(:) :: slt_dnr,slt_acc
  real (chm_real),allocatable,dimension(:) :: nvisit,nvisit_tmp,sdensity
  real(chm_real4),allocatable,dimension(:) :: itemp
  integer,allocatable,dimension(:) :: ifreat
  real(chm_real),allocatable,dimension(:) :: nhb_slv,nhb_slt,nhb_tot
  real(chm_real),allocatable,dimension(:) :: nhb_slv_tmp,nhb_slt_tmp
!  real(chm_real),allocatable,dimension(:) :: CCG        !##CHEQ

  ! For coord orientation
  real(chm_real), allocatable, dimension(:) :: bmass
  integer istats,npr
  integer,allocatable,dimension(:,:) :: atompr

  ! mpi related
  integer chunksize(numnod),offset(numnod),myoffset,ierr
  !integer status(MPI_STATUS_SIZE)

  ! for hbond analysis
  LOGICAL,allocatable,dimension(:) :: AFLAG,DFLAG
  integer n,p,ndonor,nacceptor
  real cuthb2

#if KEY_MPI==1
  call mpi_barrier(COMM_CHARMM,ierr)
#endif
  
  ! set search box size, etc
  if ( mynod .eq. 0 ) then  
     call get_dimension(x,y,z,natom,nskip,nbegn,nstop,islct, &
        atype,asolv,res,rcut,orient,xcomp,ycomp,zcomp)
     flag1 = .true. ! assume comp coord to exist
  endif
  !nframe=1+(nstop-nbegn)/nskip ! number of frames to use

  !! get number of hb donors and acceptors
  !ndonor=0; nacceptor=0;
  !do i = 1, natom
  !   if (atype(i)(1:1)=='H') then
  !      if ((atype(i) .ne. 'H1') .and. (atype(i) .ne. 'H2')) then 
  !         if (cg(i)>cutq) then ! not wat hydrogen
  !            ndonor=ndonor+1
  !         endif
  !      endif
  !   else if ((atype(i)(1:1)=='O') .or. (atype(i)(1:1)=='N')) then
  !      if (atype(i) .ne. 'OH2') then
  !         nacceptor=nacceptor+1
  !      endif
  !   endif
  !enddo

  
#if KEY_MPI==1
  ! broadcast common parameters
  call mpi_barrier(COMM_CHARMM,ierr)
  call psnd8(xmax,1); call psnd8(ymax,1); call psnd8(zmax,1);
  call psnd8(xmin,1); call psnd8(ymin,1); call psnd8(zmin,1);
  call psnd4(nx,1); call psnd4(ny,1); call psnd4(nz,1);
  call psnd4(npoint,1); call psnd4(nslct,1); call psnd4(nsolv,1);
  call mpi_barrier(COMM_CHARMM,ierr)
#endif

  call chmalloc('solvmap.src','slv_hbond','solute',nslct,intg=solute)
  call chmalloc('solvmap.src','slv_hbond','solvent',nsolv,3,intg=solvent)
  call chmalloc('solvmap.src','slv_hbond','nhb_slv',npoint,cr8=nhb_slv)
  call chmalloc('solvmap.src','slv_hbond','nhb_slt',npoint,cr8=nhb_slt)
  call chmalloc('solvmap.src','slv_hbond','nhb_tot',npoint,cr8=nhb_tot)
  call chmalloc('solvmap.src','slv_hbond','nhb_slv_tmp',npoint,cr8=nhb_slv_tmp)
  call chmalloc('solvmap.src','slv_hbond','nhb_slt_tmp',npoint,cr8=nhb_slt_tmp)
  call chmalloc('solvmap.src','slv_hbond','nvisit',npoint,cr8=nvisit)
  call chmalloc('solvmap.src','slv_hbond','nvisit_tmp',npoint,cr8=nvisit_tmp)
  call chmalloc('solvmap.src','slv_hbond','sdensity',npoint,cr8=sdensity)
  call chmalloc('solvmap.src','slv_hbond','AFLAG',NATOM,log=AFLAG)
  call chmalloc('solvmap.src','slv_hbond','DFLAG',NATOM,log=DFLAG)

#if KEY_MPI==1
  call mpi_barrier(COMM_CHARMM,ierr)
#endif
  
  ! System setup for the master node
  call chmalloc('solvmap.src','slv_hbond','itemp',natom,cr4=itemp)
  if(orient) then
     call chmalloc('solvmap.src','slv_hbond','atompr',2,natom,intg=atompr)
     call chmalloc('solvmap.src','slv_hbond','bmass',natom,crl=bmass)
  else
     call chmalloc('solvmap.src','slv_hbond','atompr',1,1,intg=atompr)
     call chmalloc('solvmap.src','slv_hbond','bmass',1,crl=bmass)
  endif

  ! Setup the orient option assuming that the reference structure is 
  !     in the COMP set
  if(orient)then
     npr=0
     do i = 1,natom
        if (islct(i) == 1) then
           npr=npr+1 ! npr: npair
           atompr(1,npr)=i; atompr(2,npr)=i ! atompr: ref atom indices
        endif
     enddo
     call pkmass(amass,amass,bmass,atompr,npr,lmass,lwmain,wmain)
  endif

  ! setup hb donor/acceptor flag
  DO I=1,NATOM
     AFLAG(I)=.FALSE.
     DFLAG(I)=.FALSE.
  ENDDO
  DO I=1,NACC
     AFLAG(IACC(I))=.TRUE.
  ENDDO
  DO I=1,NDON
     DFLAG(IHD1(I))=.TRUE.
  ENDDO

  j=0;  k=0;  ncoord=0 ! ncoord: number of coord frames
  do i=1,natom ! register selected atom index
     if(islct(i) == 1) then
        j=j+1; solute(j)=i
     endif 
     if (atype(i)==asolv) then
        k=k+1; solvent(k,1)=i
        do m=1,4 ! Considering TIP5 water, look upto 5 atoms
           ! check for different residue
           idum1=i+m
           if (idum1>natom) exit
           if (getres(idum1,ibase,nres) .ne. getres(i,ibase,nres)) cycle;
           if (atype(idum1)=='H1') then
              solvent(k,2)=idum1
           else if (atype(idum1)=='H2') then
              solvent(k,3)=idum1
           else if (atype(idum1)=='OH2') then
              solvent(k,1)=idum1 ! make sure oxygen is the first atom
           else  
              continue
           endif
        enddo
     endif
  enddo

  j=0; k=0; m=0;  ndonor=0; nacceptor=0;
  do i = 1, natom ! count ndonor,nacceptor
     if ((dflag(i).eqv. .true.) .and. (islct(i)==1)) then 
           ndonor=ndonor+1
        !endif
     endif
     if ((aflag(i).eqv. .true.) .and. (islct(i)==1)) then 
           nacceptor=nacceptor+1
        !endif
     endif
  enddo

  !write(outu,'(A,2I8)') 'Ndonor,naaceptor= ',ndonor,nacceptor ! debug
  
#if KEY_MPI==1
  call mpi_barrier(COMM_CHARMM,ierr)
  call psnd4(ndonor,1); call psnd4(nacceptor,1) 
  call mpi_barrier(COMM_CHARMM,ierr)
#endif
  call chmalloc('solvmap.src','slv_hbond','slt_dnr',ndonor,intg=slt_dnr)
  call chmalloc('solvmap.src','slv_hbond','slt_acc',nacceptor,intg=slt_acc)
#if KEY_MPI==1
  call mpi_barrier(COMM_CHARMM,ierr)
#endif
  
  do i = 1, natom
     if ((dflag(i).eqv. .true.) .and. (islct(i)==1)) then 
           j=j+1; slt_dnr(j)=i ! solute donor
        !endif
     endif
     if ((aflag(i).eqv. .true.) .and. (islct(i)==1)) then 
           m=m+1; slt_acc(m)=i ! solute acceptor
        !endif
     endif
  enddo

  iunit=firstu; istats=1
  idum1=nsolv/numnod;   idum2=mod(nsolv,numnod)
  cuthb2 = cuthb * cuthb
  do i=1,numnod ! uniformly distribute atoms among nodes
     if (i<=idum2) then
        chunksize(i) = idum1+1
     else 
        chunksize(i) = idum1
     endif
  enddo
  nhb_slv = 0.0;  nhb_slt = 0.0;  nhb_slv_tmp = 0.0;  nhb_slt_tmp = 0.0;
  nvisit = 0.; nvisit_tmp = 0.; nhb_tot=0.0; sdensity=0.0
  nfreat=natom ! to avoid nfreat from being read (see cvio.src: readcv())
  offset(1) =  1 ;

#if KEY_MPI==1
  do i = 2, numnod
     j = i -1
     offset(i) = offset(j) + chunksize(j)
  end do
#endif
  
  if (mynod .eq. 0) then 
!     write(OUTU,82) NX,NY,NZ
!82   format('SMAP: nx= ',I8,' ny= ', I8,' nz= ', I8 ,' cells willl be used.')
     write(outu,80) nslct, nsolv
80   format('SMAP:',I8, ' solute atoms and ', I8, &
          ' solvent molecules used for analysis.')
     if (dcut>0.0) write(outu,83) dcut
83   format('SMAP: Points with water density > ',F6.3, &
          ' A^-3 considered for calculation.')
  endif ! if (mynod .eq. 0)

#if KEY_MPI==1
  call mpi_barrier(COMM_CHARMM,ierr)
#endif
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  do while(istats >= 0)   ! Go over the trajectory
     if (mynod .eq. 0) then 
        call readcv(x,y,z, &
#if KEY_CHEQ==1
            (/ ZERO /), .FALSE., &  
#endif
             itemp,natom,ifreat,nfreat, &
             firstu,nunit,iunit,nfile, &
             istep,istats,ndegf,delta, &
             nbegn,nstop,nskip,nsavv,hdr1,hdr2, &
             titleb,ntitlb,.false., (/ zero /), .false.)
        ncoord=ncoord+1
        if (ncoord == 1) tol = 0.1 / real( nfile )
        if (orient) then
           ! Orient the main coord set with respect to the comp set.
           if (ncoord == 1 .and. .not. qhavcrd(natom,islct,xcomp)) then
              flag1 = .false.
              call wrndie(1,'<SLV_HBOND>', &
                   'NO COMP SET FOR ORIENT! USING THE FIRST FRAME INSTEAD.')
              do i =1,natom
                 xcomp(i)=x(i); ycomp(i)=y(i); zcomp(i)=z(i)
              enddo
           endif
           if ( flag1 .eqv. .true.) then
              CALL ROTLS1(XCOMP,YCOMP,ZCOMP,X,Y,Z,NATOM, &
                   ATOMPR,NPR,BMASS,.false.,lnoro)
           endif
        ENDIF
     endif ! mynod .eq. 0     
#if KEY_MPI==1
     call mpi_barrier(COMM_CHARMM,ierr)
     call psnd4(ncoord,1); call psnd4(istats,1)
     if ( ncoord == 1 ) call psnd8(tol,1)
     call psnd8(x,natom); call psnd8(y,natom); call psnd8(z,natom)
     call mpi_barrier(COMM_CHARMM,ierr) 
#endif
     
     ! Count number of visits and hbonds on each node
     i = mynodp
     do j=offset(i),offset(i)+chunksize(i)-1
        k=solvent(j,1); x0=x(k); y0=y(k); z0=z(k)
        flag=.false.
        if ((x0>xmin) .and. (x0<xmax)) then
           if ((y0>ymin) .and. (y0<ymax)) then
              if ((z0>zmin) .and. (z0<zmax)) then           
                 flag=.true.
              endif
           endif
        endif
        if (flag .eqv. .true. ) then ! measure hbond
           ! i{x,y,z} <= 0 : can happen due to movement
           ix=ceiling((x0-xmin)/res); if ( ix .le. 0) cycle 
           iy=ceiling((y0-ymin)/res); if ( iy .le. 0) cycle 
           iz=ceiling((z0-zmin)/res); if ( iz .le. 0) cycle 
           if ((ix > nx) .or. (iy > ny) .or. (iz > nz)) then
              cycle ! This can happen because the molecule moves. Skip.
           endif
           k = (iz-1)*ny*nx + (iy-1)*nx + ix 
           if (k> nx*ny*nz) then ! this should not happen
              write(outu,'(a,3I5)') 'SLV_HBOND: ix,iy,iz = ',ix,iy,iz
              call wrndie(-2,'<SLV_HBOND>', &
                   'ARRAY INDEX OUT OF RANGE.')
           endif
           nvisit_tmp(k) = nvisit_tmp(k) + 1.0 ! count number of visits

           ! count number of hbonds
           do m=1,ndonor ! current water oxygen-solute donor
              n=slt_dnr(m); x1=x(n); y1=y(n);z1=z(n)
              if ((abs(x0-x1)>cuthb).or.(abs(y0-y1)>cuthb).or. &
                   (abs(z0-z1)>cuthb)) cycle; 
              rdum=((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
              if (rdum<cuthb2) then
                 nhb_slt_tmp(k) = nhb_slt_tmp(k) + 1.0
              endif
           enddo
           do m=1,nsolv ! current water oxygen- other water hydrogen
              if ( m .eq. j) cycle; ! skip self
              n=solvent(m,2); x1=x(n); y1=y(n);z1=z(n)
              if ((abs(x0-x1)>cuthb).or.(abs(y0-y1)>cuthb).or. &
                   (abs(z0-z1)>cuthb)) cycle; 
              rdum=((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
              if (rdum<cuthb2) then
                 nhb_slv_tmp(k) = nhb_slv_tmp(k) + 1.0
              endif
              n=solvent(m,3); x1=x(n); y1=y(n);z1=z(n)
              if ((abs(x0-x1)>cuthb).or.(abs(y0-y1)>cuthb).or. &
                   (abs(z0-z1)>cuthb)) cycle; 
              rdum=((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
              if (rdum<cuthb2) then
                 nhb_slv_tmp(k) = nhb_slv_tmp(k) + 1.0
              endif
           enddo
           p=solvent(j,2); x2=x(p); y2=y(p); z2=z(p)
           do m=1,nacceptor ! current water hydrogen -solute acceptor 
              n=slt_acc(m); x1=x(n); y1=y(n);z1=z(n)
              if ((abs(x2-x1)>cuthb).or.(abs(y2-y1)>cuthb).or. &
                   (abs(z2-z1)>cuthb)) cycle; 
              rdum=((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
              if (rdum<cuthb2) then
                 nhb_slt_tmp(k) = nhb_slt_tmp(k) + 1.0
              endif
           enddo
           do m=1,nsolv ! current water hydrogen-other water oxygen
              if ( m .eq. j) cycle;
              n=solvent(m,1); x1=x(n); y1=y(n);z1=z(n)
              if ((abs(x2-x1)>cuthb).or.(abs(y2-y1)>cuthb).or. &
                   (abs(z2-z1)>cuthb)) cycle; 
              rdum=((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
              if (rdum<cuthb2) then
                 nhb_slv_tmp(k) = nhb_slv_tmp(k) + 1.0
              endif
           enddo

           p=solvent(j,3); x2=x(p); y2=y(p); z2=z(p)
           do m=1,nacceptor ! current water hydrogen -solute acceptor 
              n=slt_acc(m); x1=x(n); y1=y(n);z1=z(n)
              if ((abs(x2-x1)>cuthb).or.(abs(y2-y1)>cuthb).or. &
                   (abs(z2-z1)>cuthb)) cycle; 
              rdum=((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
              if (rdum<cuthb2) then
                 nhb_slt_tmp(k) = nhb_slt_tmp(k) + 1.0
              endif
           enddo
           do m=1,nsolv ! current water hydrogen-other water oxygen
              if ( m .eq. j) cycle;
              n=solvent(m,1); x1=x(n); y1=y(n);z1=z(n)
              if ((abs(x2-x1)>cuthb).or.(abs(y2-y1)>cuthb).or. &
                   (abs(z2-z1)>cuthb)) cycle; 
              rdum=((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
              if (rdum<cuthb2) then
                 nhb_slv_tmp(k) = nhb_slv_tmp(k) + 1.0
              endif
           enddo
        endif ! if (flag .eqv. .true. ) then ! measure hbond
     enddo
#if KEY_MPI==1     
     call mpi_barrier(COMM_CHARMM,ierr)
#endif     
     !if (ncoord == nframe) exit
  enddo ! do while(istats >= 0)

#if KEY_MPI==1
  ! combine all contributions to density
  call mpi_barrier(COMM_CHARMM,ierr)
  call mpi_allreduce(nhb_slt_tmp,nhb_slt,npoint,mpi_double_precision, &
       MPI_SUM,COMM_CHARMM,ierr)
  call mpi_allreduce(nhb_slv_tmp,nhb_slv,npoint,mpi_double_precision, &
       MPI_SUM,COMM_CHARMM,ierr)
  call mpi_allreduce(nvisit_tmp,nvisit,npoint,mpi_double_precision, &
       MPI_SUM,COMM_CHARMM,ierr)
  call mpi_barrier(COMM_CHARMM,ierr)
#else
  do k=1,npoint
     nhb_slt(k)=nhb_slt_tmp(k)
     nhb_slv(k)=nhb_slv_tmp(k)
     nvisit(k)=nvisit_tmp(k)
  enddo
#endif
  
 ! Calculate and write out density
  if ( mynod .eq. 0 ) then
     do k = 1, npoint ! make nvisit as solvent density
        if (nvisit(k) < tol) then
           sdensity(k) = 0.0
        else
           sdensity(k) = nvisit(k)/(dble(ncoord)* res**3)
        endif
     enddo
     
     do k = 1, npoint
        if (nvisit(k) < tol) then
           nhb_slt(k) = 0.0;  nhb_slv(k) = 0.0
           cycle
        endif
        if (nhb_slt(k) < tol) then
           nhb_slt(k) = 0.0
        else
           if (sdensity(k) > dcut) then 
              nhb_slt(k) = nhb_slt(k)/nvisit(k)
           else
              nhb_slt(k) = 0.0
           endif
        endif
        if (nhb_slv(k) < tol) then
           nhb_slv(k) = 0.0
        else
           if (sdensity(k) > dcut) then 
              nhb_slv(k) = nhb_slv(k)/nvisit(k)
           else 
              nhb_slv(k) = 0.9
           endif
        endif
     enddo
     if (otot .ne. -1) then
        do k = 1, npoint
           nhb_tot(k)=nhb_slt(k)+nhb_slv(k)
        enddo
     endif
     
     call write_data(sdensity,ounit,outmode,res,rcut,'water density (A^-3)')
     call write_data(nhb_slt,oslt,outmode,res,rcut,'nHb w/ solute')
     call write_data(nhb_slv,oslv,outmode,res,rcut,'nHb w/ water')
     call write_data(nhb_tot,otot,outmode,res,rcut,'nHb w/ (water+soute)')

     if (flag1 .eqv. .false.) then ! reset comp set
        xcomp(1:natom) = anum; ycomp(1:natom) = anum; zcomp(1:natom) = anum;
     endif
  endif !  if ( mynod .eq. 0 ) then

  call chmdealloc('solvmap.src','slv_hbond','itemp',natom,cr4=itemp)
!  call chmdealloc('solvmap.src','slv_hbond','CCG',natom,crl=CCG)      !##CHEQ
  if(orient) then
     call chmdealloc('solvmap.src','slv_hbond','atompr',2,natom,intg=atompr)
     call chmdealloc('solvmap.src','slv_hbond','bmass',natom,crl=bmass)
  else
     call chmdealloc('solvmap.src','slv_hbond','atompr',1,1,intg=atompr)
     call chmdealloc('solvmap.src','slv_hbond','bmass',1,crl=bmass)
  endif
  call chmdealloc('solvmap.src','slv_hbond','solute',nslct,intg=solute)
  call chmdealloc('solvmap.src','slv_hbond','solvent',nsolv,3,intg=solvent)
  call chmdealloc('solvmap.src','slv_hbond','slt_dnr',ndonor,intg=slt_dnr)
  call chmdealloc('solvmap.src','slv_hbond','slt_acc',nacceptor,intg=slt_acc)
  call chmdealloc('solvmap.src','slv_hbond','nhb_slv',npoint,cr8=nhb_slv)
  call chmdealloc('solvmap.src','slv_hbond','nhb_slt',npoint,cr8=nhb_slt)
  call chmdealloc('solvmap.src','slv_hbond','nhb_tot',npoint,cr8=nhb_tot)
  call chmdealloc('solvmap.src','slv_hbond','nhb_slv_tmp',npoint,cr8=nhb_slv_tmp)
  call chmdealloc('solvmap.src','slv_hbond','nhb_slt_tmp',npoint,cr8=nhb_slt_tmp)
  call chmdealloc('solvmap.src','slv_hbond','nvisit',npoint,cr8=nvisit)
  call chmdealloc('solvmap.src','slv_hbond','nvisit_tmp',npoint,cr8=nvisit_tmp)
  call chmdealloc('solvmap.src','slv_hbond','sdensity',npoint,cr8=sdensity)
end subroutine slv_hbond

!-------------------------------------------------------!

subroutine slv_dift(x,y,z,wmain,natom,nunit,firstu,nskip,nbegn,nstop, &
     islct,atype,res,rcut,ounit,ounit1,outmode,orient,xcomp,ycomp,zcomp,wcomp,&
     lmass,lwmain,lnoro,amass,asolv,wdist,dcut,hcut)
! Calculate translational diffusion coefficient
  use vector

  ! argument vars
  real(chm_real) x(*),y(*),z(*),wmain(*)
  integer natom,nunit,firstu,nskip,nbegn,nstop,ounit,ounit1,outmode
  integer islct(*)
  character(len=*) atype(*)
  real (chm_real) res,rcut,wdist,dcut,hcut
  logical orient,lmass,lwmain,lnoro
  real(chm_real) xcomp(*),ycomp(*),zcomp(*),wcomp(*),amass(*)
  character(len=8) asolv ! solvent atom name
  
  ! local vars
  integer numvisit
  real rcut2,wdist2,dt
  integer, allocatable,dimension(:) :: solvent,solute ! section subset
  real(chm_real4),allocatable,dimension(:) :: itemp
  integer,allocatable,dimension(:) :: ifreat
  integer,allocatable,dimension(:) :: list_cell,list_solv
  real (chm_real),allocatable,dimension(:) :: nvisit,nvisit_tmp,sdensity
  real(chm_real),allocatable,dimension(:) :: disp2,disp2_tmp
  real(chm_real),allocatable,dimension(:,:) :: pos_tmp
!  real(chm_real),allocatable,dimension(:) :: CCG        !##CHEQ

  ! For coord orientation
  real(chm_real), allocatable, dimension(:) :: bmass
  real(chm_real) xcm0,ycm0,zcm0,phi0,xave,yave,zave
  real(chm_real) U(9),RN(3)
  logical lok
  integer istats, npr
  integer,allocatable,dimension(:,:) :: atompr

  ! mpi related
  integer chunksize(numnod),offset(numnod),myoffset,tag1,tag2, ierr
  !integer status(MPI_STATUS_SIZE)

  ! Initialize
  do i = 1, 3
     rn(i) = 0.0
  enddo

  if ( mynod .eq. 0 ) then  
     call get_dimension(x,y,z,natom,nskip,nbegn,nstop,islct, &
        atype,asolv,res,rcut,orient,xcomp,ycomp,zcomp)
     flag1 = .true. ! assume comp coord to exist
     wdist2=wdist**2
  endif

#if KEY_MPI==1
  ! broadcast common parameters
  call mpi_barrier(COMM_CHARMM,ierr)
  call psnd8(xmax,1); call psnd8(ymax,1); call psnd8(zmax,1);
  call psnd8(xmin,1); call psnd8(ymin,1); call psnd8(zmin,1);
  call psnd4r(wdist2,1) ! possibly bug
  call psnd4(nx,1); call psnd4(ny,1); call psnd4(nz,1);
  call psnd4(npoint,1); call psnd4(nslct,1); call psnd4(nsolv,1);
  call mpi_barrier(COMM_CHARMM,ierr)
#endif

  call chmalloc('solvmap.src','slv_dift','solute',nslct,intg=solute)
  call chmalloc('solvmap.src','slv_dift','solvent',nsolv,intg=solvent)
  call chmalloc('solvmap.src','slv_dift','list_cell',npoint,intg=list_cell)
  call chmalloc('solvmap.src','slv_dift','list_solv',npoint,intg=list_solv)
  call chmalloc('solvmap.src','slv_dift','nvisit',npoint,cr8=nvisit)
  call chmalloc('solvmap.src','slv_dift','nvisit_tmp',npoint,cr8=nvisit_tmp)
  call chmalloc('solvmap.src','slv_dift','sdensity',npoint,cr8=sdensity)  
  call chmalloc('solvmap.src','slv_dift','disp2',npoint,cr8=disp2)
  call chmalloc('solvmap.src','slv_dift','disp2_tmp',npoint,cr8=disp2_tmp)
  call chmalloc('solvmap.src','slv_dift','pos_tmp',3,npoint,cr8=pos_tmp)

#if KEY_MPI==1  
  call mpi_barrier(COMM_CHARMM,ierr)
#endif

  ! System setup for the master node
  call chmalloc('solvmap.src','slv_dift','itemp',natom,cr4=itemp)
!  call chmalloc('solvmap.src','slv_dift','CCG',natom,crl=CCG)      !##CHEQ
  if(orient) then
     call chmalloc('solvmap.src','slv_dift','atompr',2,natom,intg=atompr)
     call chmalloc('solvmap.src','slv_dift','bmass',natom,crl=bmass)
  else
     call chmalloc('solvmap.src','slv_dift','atompr',1,1,intg=atompr)
     call chmalloc('solvmap.src','slv_dift','bmass',1,crl=bmass)
  endif

  ! Setup the orient option assuming that the reference structure is 
  !     in the COMP set
  if(orient)then
     npr=0
     do i = 1,natom
        if (islct(i) == 1) then
           npr=npr+1; atompr(1,npr)=i; atompr(2,npr)=i 
        endif
     enddo
     call pkmass(amass,amass,bmass,atompr,npr,lmass,lwmain,wmain)
  endif
  j=0;  k=0;  ncoord=0 ! ncoord: number of coord frames
  do i=1,natom ! register selected atom index
     if(islct(i) == 1) then
        j=j+1; solute(j)=i
     endif
     if (atype(i)==asolv) then
        k=k+1; solvent(k)=i
     endif
  enddo
  iunit=firstu; istats=1
  idum1=nsolv/numnod;     idum2=mod(nsolv,numnod)
  rcut2 = rcut * rcut
  do i=1,numnod ! uniformly distribute atoms among nodes
     if (i<=idum2) then
        chunksize(i) = idum1+1
     else 
        chunksize(i) = idum1
     endif
  enddo
  nvisit = 0.; nvisit_tmp = 0.; list_cell = 0; list_solv = 0
  disp2 = 0.; disp2_tmp = 0.; pos_tmp = 0.; sdensity=0.0
  nfreat=natom ! to avoid nfreat from being read (see cvio.src: readcv())
  offset(1) =  1 ! send each node portion of the array 

#if KEY_MPI==1
  do i = 2, numnod
     j = i -1
     offset(i) = offset(j) + chunksize(j)
  end do
#endif
  
  if (mynod .eq. 0) then 
     write(outu,80) nslct, nsolv
80   format('SMAP:',I8, ' solute atoms and ', I8, &
          ' solvent atoms used for analysis.')
     if (dcut>0.0) write(outu,83) dcut
83   format('SMAP: Points with water density > ',F6.3, &
          ' A^-3 considered for calculation.')
  endif ! if (mynod .eq. 0)

#if KEY_MPI==1
  call mpi_barrier(COMM_CHARMM,ierr)
#endif
  
  do while(istats >= 0) ! Go over the trajectory
     if (mynod .eq. 0) then 
        call readcv(x,y,z, &
#if KEY_CHEQ==1
            (/ ZERO /), .FALSE., &  
#endif
!             ccg,qcg,                        & !##CHEQ
             itemp,natom,ifreat,nfreat, &
             firstu,nunit,iunit,nfile, &
             istep,istats,ndegf,delta, &
             nbegn,nstop,nskip,nsavv,hdr1,hdr2, &
             titleb,ntitlb,.false., (/ zero /), .false.)
        ncoord=ncoord + 1
!        write (outu,'(/A,I10)') 'SMAP-DIFT: Working on frame ',ncoord
        if (ncoord == 1 ) then
           tol = 0.1 / real( nfile )
           call get_param('TIMFAC',rdum)           
           dt = rdum * delta * real(nskip)
           write (outu,'(A,f8.4,A)') 'SMAP: Coordinate saving&
                & frequency = ',dt,' ps'
        endif
        if ( (orient) .and. ( ncoord > 1 ) )then
           ! From frame 2, orient coord using the matrix from the previous 
           ! frame and measure solvent displacement
           rdum = dotvec(rn,rn,3)
           ! no rotation if the previous coord wasn't oriented (can happen
           ! if it is the reference coord.
           if ( rdum < TENM8 ) then 
              rn=0.0;
              rn(1) = 1.0; phi0 = 0.0;
           endif
           call fndu(u,rn,phi0,lok)

           if(.not. lok) then
              call wrndie(-1,'<SLV_DIFT>','PARSING ERROR')
           endif
           !call get_cm(x,y,z,xave,yave,zave,solute,nslct,bmass)
        endif ! ( (orient) .and. ( ncoord > 1 ) )then
        do i=1,natom
           if(x(i) == anum) then
              call wrndie(-1,'<SLV_DIFT>','MISSING COORDINATES')
           endif
        enddo
     endif ! (mynod .eq. 0) then 

#if KEY_MPI==1
     call mpi_barrier(COMM_CHARMM,ierr)
     if (orient) then
        call psnd8(u,9); call psnd8(rn,3); ! call psnd8(phi0,1)
        call psnd8(xave,1); call psnd8(yave,1); call psnd8(zave,1);
     endif
     call psnd4(ncoord,1); call psnd4(istats,1)
     CALL PSND8(X,NATOM); CALL PSND8(Y,NATOM); CALL PSND8(Z,NATOM)
     call mpi_barrier(COMM_CHARMM,ierr)
#endif     

     ! calculate displacement squared in previous sites
     ! before orienting the current coordinate
     if (ncoord > 1 ) then 
        do j = 1, numvisit
           i = list_solv(j); k = list_cell(j)
           if (orient) then
              ! [xyz]ave: CM of the ref atom in the previous frame
              ! u: rotation matrix in the previous frame
              ! [xyz]cm0: CM of the ref atom in the comp set
              ! [xyz]1: coord in the current frame oriented as in prev frame
              x0=x(i)-xave; y0=y(i)-yave; z0=z(i)-zave
              x1=u(1)*x0 + u(2)*y0 + u(3)*z0 + xcm0
              y1=u(4)*x0 + u(5)*y0 + u(6)*z0 + ycm0
              z1=u(7)*x0 + u(8)*y0 + u(9)*z0 + zcm0
           else
              x1 = x(i); y1= y(i); z1 = z(i)
           endif
           x0 = pos_tmp(1,j) - x1 ! pos_tmp: coord of prev frame
           y0 = pos_tmp(2,j) - y1
           z0 = pos_tmp(3,j) - z1
           rdum =  x0**2 + y0**2 + z0**2
           if ((rdum>wdist2) .and. (wdist >0.)) then
              write(outu,'(a,I6,a,F10.4)') 'Solvent ',i,' displacement ' &
                   ,sqrt(rdum)
              call wrndie(-1,'<SLV_DIFT>',&
                   'Solvent moved more than warning distance.')
           endif
           disp2_tmp(k) = disp2_tmp(k) + rdum
        enddo
     endif

#if KEY_MPI==1     
     call mpi_barrier(COMM_CHARMM,ierr)
#endif     

     if (mynod .eq. 0 ) then
        if (orient) then
           ! Orient the main coord set with respect to the comp set.
           if (ncoord == 1 .and. .not. qhavcrd(natom,islct,xcomp)) then
              flag1 = .false.
              call wrndie(1,'<SLV_DIFT>', &
                   'NO COMP SET FOR ORIENT! USING THE FIRST FRAME INSTEAD.')
              do i =1,natom
                 xcomp(i)=x(i); ycomp(i)=y(i); zcomp(i)=z(i)
              enddo
           endif
           
           call get_cm(x,y,z,xave,yave,zave,solute,nslct,bmass)
           CALL ROTLS1(XCOMP,YCOMP,ZCOMP,X,Y,Z,NATOM, &
                ATOMPR,NPR,BMASS,.true.,lnoro)
           call get_param('XAXI',rn(1))
           call get_param('YAXI',rn(2))
           call get_param('ZAXI',rn(3))
           call get_param('THET',phi0)
           phi0=-phi0 ! flip sign due to the inverted sense of coor rota
           
           if (ncoord==1) then
              ! center of mass of solute in the comp set
              call get_cm(xcomp,ycomp,zcomp,xcm0,ycm0,zcm0,solute,nslct,bmass)
              !if (ncoord .eq. 1) then ! debug
              !   write (outu,'(a,3F12.5)') 'cm0: ',xcm0,ycm0,zcm0
              !endif
           endif
        endif ! (orient) then
     endif ! mynod .eq. 0     
#if KEY_MPI==1
     call mpi_barrier(COMM_CHARMM,ierr)
     if ( ncoord == 1 ) then
        call psnd8(tol,1); call psnd8(delta,1)
        call psnd8(xcm0,1); call psnd8(ycm0,1); call psnd8(zcm0,1);
     endif
     call psnd8(rn,3); call psnd8(phi0,1);
     call psnd8(xave,1); call psnd8(yave,1); call psnd8(zave,1);
     CALL PSND8(X,NATOM); CALL PSND8(Y,NATOM); CALL PSND8(Z,NATOM)
     call mpi_barrier(COMM_CHARMM,ierr) 
#endif
     
     ! identify visited sites 
     i = mynodp; numvisit = 0. ; list_cell = 0; list_solv = 0
     do j=offset(i),offset(i)+chunksize(i)-1
        m=solvent(j); x0=x(m); y0=y(m); z0=z(m)
        flag=.false.
        if ((x0>xmin) .and. (x0<xmax)) then
           if ((y0>ymin) .and. (y0<ymax)) then
              if ((z0>zmin) .and. (z0<zmax)) then           
                 flag=.true.
              endif
           endif
        endif

        if (flag .eqv. .true. ) then ! count density
           ! i{x,y,z} <= 0 : can happen due to movement
           ix=ceiling((x0-xmin)/res); if ( ix .le. 0) cycle 
           iy=ceiling((y0-ymin)/res); if ( iy .le. 0) cycle 
           iz=ceiling((z0-zmin)/res); if ( iz .le. 0) cycle 
           if ((ix > nx) .or. (iy > ny) .or. (iz > nz)) then
              cycle ! This can happen because the molecule moves. Skip.
           endif
           k = (iz-1)*ny*nx + (iy-1)*nx + ix 
           if (k> nx*ny*nz) then ! this should not happen
              write(outu,'(a,3I5)') 'SMAP: ix,iy,iz = ',ix,iy,iz
              call wrndie(-2,'<SLV_DIFT>', &
                   'DENSITY ARRAY INDEX OUT OF RANGE.')
           endif
           if (istats >= 0) then ! skip the last frame (no more displacement)
              ! k=cell index, m=solvent index 
              nvisit_tmp(k)=nvisit_tmp(k) + 1.0
              numvisit = numvisit + 1;  ! visit count for the current node
              list_solv(numvisit) = m; list_cell(numvisit) = k
              pos_tmp(1,numvisit) = x(m);pos_tmp(2,numvisit) = y(m);
              pos_tmp(3,numvisit) = z(m);
           endif
        endif
     enddo
#if KEY_MPI==1     
     call mpi_barrier(COMM_CHARMM,ierr)
#endif     
     !if (ncoord == nframe) exit
  enddo ! do while(istats >= 0)

#if KEY_MPI==1  
! combine all contributions
  call mpi_allreduce(nvisit_tmp,nvisit,npoint,mpi_double_precision, &
       MPI_SUM,COMM_CHARMM,ierr)
  call mpi_allreduce(disp2_tmp,disp2,npoint,mpi_double_precision, &
       MPI_SUM,COMM_CHARMM,ierr)
  call mpi_barrier(COMM_CHARMM,ierr)
#else
  do k=1,npoint
     disp2(k)=disp2_tmp(k)
     nvisit(k)=nvisit_tmp(k)
  enddo
#endif

  if ( mynod .eq. 0 ) then  ! Calculate and write out diffusion coeff.
     do k = 1, npoint ! make nvisit as solvent density
        if (nvisit(k) < tol) then
           sdensity(k) = 0.0
        else
           sdensity(k) = nvisit(k)/(dble(ncoord)* res**3)
        endif
     enddo

     do k = 1, npoint
        if (nvisit(k) < tol) then
           disp2(k) = 0.0 ! -9999.0
        else
           disp2(k) = disp2(k) / ( nvisit(k) * 6.0 * dt )
           if ((dcut>0.0).and.(sdensity(k)<dcut)) disp2(k)=0.0
           if ((hcut>0.0) .and. (disp2(k)>hcut)) disp2(k)=0.0
        endif
     enddo

     call write_data(sdensity,ounit,outmode,res,rcut,'water density (A^-3)')
     call write_data(disp2,ounit1,outmode,res,rcut,'D_trans(A^2/ps)')

     if (flag1 .eqv. .false.) then ! reset comp set
        xcomp(1:natom) = anum; ycomp(1:natom) = anum; zcomp(1:natom) = anum;
     endif
  endif ! ( mynod .eq. 0 )

  call chmdealloc('solvmap.src','slv_dift','itemp',natom,cr4=itemp)
!  call chmdealloc('solvmap.src','slv_dift','CCG',natom,crl=CCG)      !##CHEQ
  if(orient) then
     call chmdealloc('solvmap.src','slv_dift','atompr',2,natom,intg=atompr)
     call chmdealloc('solvmap.src','slv_dift','bmass',natom,crl=bmass)
  else
     call chmdealloc('solvmap.src','slv_dift','atompr',1,1,intg=atompr)
     call chmdealloc('solvmap.src','slv_dift','bmass',1,crl=bmass)
  endif

  call chmdealloc('solvmap.src','slv_dift','solute',nslct,intg=solute)
  call chmdealloc('solvmap.src','slv_dift','solvent',nsolv,intg=solvent)
  call chmdealloc('solvmap.src','slv_dift','list_cell',npoint,intg=list_cell)
  call chmdealloc('solvmap.src','slv_dift','list_solv',npoint,intg=list_solv)
  call chmdealloc('solvmap.src','slv_dift','nvisit',npoint,cr8=nvisit)
  call chmdealloc('solvmap.src','slv_dift','nvisit_tmp',npoint,cr8=nvisit_tmp)
  call chmdealloc('solvmap.src','slv_dift','sdensity',npoint,cr8=sdensity)
  call chmdealloc('solvmap.src','slv_dift','disp2',npoint,cr8=disp2)
  call chmdealloc('solvmap.src','slv_dift','disp2_tmp',npoint,cr8=disp2_tmp)
  call chmdealloc('solvmap.src','slv_dift','pos_tmp',3,npoint,cr8=pos_tmp)

end subroutine slv_dift

!-------------------------------------------------------!

subroutine get_dimension(x,y,z,natom,nskip,nbegn,nstop,islct, &
        atype,asolv,res,rcut,orient,xcomp,ycomp,zcomp)
! prepare for solvmap calculation by finding search box size, etc.
! if orient is true, work w/ comp set. If comp set is absent, use the
! current coord.

! argument vars
  real(chm_real) x(*),y(*),z(*)
  integer natom,nskip,nbegn,nstop,iflag
  integer islct(*)
  character(len=*) atype(*)
  real (chm_real) res,rcut,rdum1
  real(chm_real) xmin0,xmax0,ymin0,ymax0,zmin0,zmax0
  logical orient, flag
  real(chm_real) xcomp(*),ycomp(*),zcomp(*)
  character(len=8) asolv
  
  flag = .false. 
  ! Find span of the selected atoms and box size
  nslct=0;  nsolv=0 ! Number of selected / solvent molecules
  ! Selected region size
  xmin=anum ; ymin=anum ; zmin=anum
  xmax=-anum; ymax=-anum; zmax=-anum
  ! Entire system size
  !xmin0=anum ; ymin0=anum ; zmin0=anum
  !xmax0=-anum; ymax0=-anum; zmax0=-anum
  
  if(orient)then
     if (.not. qhavcrd(natom,islct,xcomp)) then
        write (outu,'(A)') &
             'SMAP: No comp coord for orient! Using initially read coord&
             & to determine search box.'
        flag = .true.
        do i =1,natom
           xcomp(i)=x(i); ycomp(i)=y(i); zcomp(i)=z(i)
        enddo
     endif
  else ! no orient. Just use the first frame
     flag = .true.
     do i =1,natom
        if(xcomp(i) == anum) then
           call wrndie(-1,'<GET_DIMENSION>', &
                'UNDEFINED COORDINATES IN THE MAIN SET.')
        endif
        xcomp(i)=x(i); ycomp(i)=y(i); zcomp(i)=z(i)
     enddo
  endif

  iflag=0
  do i=1,natom
     if (atype(i)==asolv) then
        !if (islct(i) == 1) then 
        !   write (outu,'(A)') 'SMAP: Solvent atom selected as solute. Check&
        !        & whether this is what you wanted.' 
        !endif
        nsolv=nsolv+1
     endif
     if(xcomp(i) == anum) then
        iflag=1
        cycle ! skip this atom
     endif
     if(islct(i) == 1) then
        nslct=nslct+1
        if(xcomp(i) > xmax) xmax=xcomp(i); if(xcomp(i) < xmin) xmin=xcomp(i)
        if(ycomp(i) > ymax) ymax=ycomp(i); if(ycomp(i) < ymin) ymin=ycomp(i)
        if(zcomp(i) > zmax) zmax=zcomp(i); if(zcomp(i) < zmin) zmin=zcomp(i)
     endif
  enddo
  if (iflag==1) then
     call wrndie(-1,'<GET_DIMENSION>', 'UNDEFINED COORDINATES IN THE COMP SET.')
  endif
  
  ! Determine the search box
  xmax=xmax+rcut; ymax=ymax+rcut; zmax=zmax+rcut
  xmin=xmin-rcut; ymin=ymin-rcut; zmin=zmin-rcut

  !! limit the box within the entire system size
  !if (xmin<xmin0) xmin=xmin0;  if (xmax>xmax0) xmax=xmax0
  !if (ymin<ymin0) ymin=ymin0;  if (ymax>ymax0) ymax=ymax0
  !if (zmin<zmin0) zmin=zmin0;  if (zmax>zmax0) zmax=zmax0

  nx=ceiling((xmax-xmin)/res) ! number of cells in the box
  ny=ceiling((ymax-ymin)/res)
  nz=ceiling((zmax-zmin)/res)
  npoint=nx*ny*nz

!  ! shift the search box so that designated ceter is at the center of the cell
!  if (xcen<anum) then
!     rdum1=xcen-xmin
!     rdum=rdum1-res*real(ceiling(rdum1/res))
!     xmin=xmin+rdum !-0.5*res
!  endif
!  if (ycen<anum) then
!     rdum1=ycen-ymin
!     rdum=rdum1-res*real(ceiling(rdum1/res))
!     ymin=ymin+rdum ! -0.5*res
!  endif
!  if (zcen<anum) then
!     rdum1=zcen-zmin
!     rdum=rdum1-res*real(ceiling(rdum1/res))
!     zmin=zmin+rdum ! -0.5*res
!  endif
  
  ! reset comp set
  if (flag) then
     xcomp(1:natom) = anum; ycomp(1:natom) = anum; zcomp(1:natom) = anum;
  endif
end subroutine get_dimension

!-------------------------------------------------------

subroutine write_data(adata,ounit,outmode,res,rcut,label)
! write array adata to ounit, with label for the measured quantity
! adata must be prepared such that elements to be omitted are set to
! -9999. Since MRC (CCP4) store data in 4 bytes, need to convert htme.
! 
! outmode: output format (1: EMAP, 2: card)

  ! argument vars
  real(chm_real) adata(*)
  real (chm_real) res,rcut,amin,amax,amean,rms
  integer ounit,outmode
  character(len=*) label
  ! for emdb format
  integer*4 ncs,nrs,nss,idum,MAPC,MAPR,MAPS  ! n{c,r,s}s: origin in grid units
  integer*4 nx4,ny4,nz4
  integer*4 ISPG,NSYMBT,LSKFLG,NLABL
  real(chm_real4) XL,YL,ZL,alpha,beta,gamma,rdum4
  real(chm_real4) amin4,amax4,amean4,rms4
  real(chm_real4) SKWMAT(9),SKWTRN(3),EXTRA(15)
  real(chm_real8) anum0,anum1
  character*4 MAP,MACHST
  character*6 cdum1,cdum2
  CHARACTER*80 LABEL_N(10)

  if (ounit .eq. -1) return
  amean=0.d0; rms=0.d0
  NLABL=2
  if (mynod .eq. 0 )  then
     x0=xmin+res*0.5; y0=ymin+res*0.5; z0=zmin+res*0.5
     ! get min,max, mean
     anum0 = 1 - anum; anum1 = -anum0
     amin=anum; amax=-anum; amean=0.0; rms=0.0
     idum1 = 0
     do k = 1, npoint
        rdum=adata(k) 
        !if (rdum > anum0 ) then
        if (rdum > tol ) then ! only consider positive nonzero values
           if (amin > rdum) amin = rdum
           if (amax < rdum) amax = rdum
           amean = amean + rdum
           rms = rms + rdum**2
           idum1 = idum1 + 1
        endif
     enddo
     if (idum1 .eq. 0) then
        call wrndie(-2,'<WRITE_DATA>', 'NO CELLS WITH POSITIVE DATA VALUE.')
     endif
     amean = amean / real(idum1)
     rms =rms / real(idum1) - amean*amean ;

     if ( rms < 0.0) then
        call wrndie(-2,'<WRITE_DATA>', 'NEGATIVE RMS OF MAP VALUES.') 
     endif
     rms = sqrt(rms)

     write(outu,'(a)') ''
     write(outu,'(a,a)') 'SMAP: ',label ! \r: carriage return
     write(outu,'(a)') 'SMAP: Search range for solvmap calculation:'
     write(outu,'(a,F10.4,a,F10.4,a,F10.4)') 'SMAP: xmin= ',xmin, &
          ' ymin= ', ymin, ' zmin= ',zmin
     write(outu,'(a,F10.4,a,F10.4,a,F10.4)') 'SMAP: xmax= ',xmax, &
          ' ymax= ', ymax, ' zmax= ',zmax
     write(outu,'(a,i5,a,i5,a,i5,a,i5)') 'SMAP: nx= ',nx,' ny= ',ny, &
          ' nz= ',nz, '  NCOORD= ',ncoord
     write(outu,'(a,f7.3,a,f10.3)') 'SMAP: binsize= ',res, ' Rcut= ',rcut
     write(outu,'(a,F10.4,a,F10.4,a,F10.4,a,F10.4)') &
          'SMAP: min= ',amin,'  max= ',amax,' avg= ',amean,'  std= ',rms

     select case (outmode)
        case (2)  ! write to CARD
           write(ounit,'(a)') '# Search range for solvmap calculation:'
           write(ounit,'(a,F10.4,a,F10.4,a,F10.4)') '# xmin= ',xmin, &
                ' ymin= ', ymin, ' zmin= ',zmin
           write(ounit,'(a,F10.4,a,F10.4,a,F10.4)') '# xmax= ',xmax, &
                ' ymax= ', ymax, ' zmax= ',zmax
           write(ounit,'(a,i5,a,i5,a,i5,a,i5)') '# nx= ',nx,' ny= ',ny, &
                ' nz= ',nz, '  NCOORD= ',ncoord
           write(ounit,'(a,f7.3,a,f10.3)') '# binsize= ',res, ' Rcut= ',rcut
           write(ounit,'(a,a15)') '# ix       x      iy       y      iz       z       ',   label
81         FORMAT(I4,' ',F10.4,' ',I4,' ',F10.4,' ',I4,' ',F10.4,' ',ES15.5E3)
           do iz = 1, nz
              do iy = 1,ny
                 do ix = 1,nx 
                    k = (iz-1)*nx*ny + (iy-1)*nx + ix
                    if (adata(k) > -9998) then
                       write(ounit,81) ix,x0,iy,y0,iz,z0, adata(k)
                    endif
                    x0=x0+res
                 enddo
                 y0=y0+res; x0=xmin+res*0.5
              enddo
              z0=z0+res; y0=ymin+res*0.5
           enddo
!           do ix = 1, nx
!              do iy = 1,ny
!                 do iz = 1,nz
!                    k = (ix-1)*ny*nz + (iy-1)*nz + iz
!                    if (adata(k) > -9998) then
!                       write(ounit,81) ix,x0,iy,y0,iz,z0, adata(k)
!                    endif
!                    z0=z0+res
!                 enddo
!                 y0=y0+res; z0=zmin+res*0.5
!              enddo
!              x0=x0+res; y0=ymin+res*0.5
!           enddo
!           !  case (2)  ! write to CARD

        case default
           idum=2
           nx4=nx; ny4=ny; nz4=nz
           amin4=amin; amax4=amax; amean4=amean; rms4=rms
           ncs=ceiling(x0/res); nrs=ceiling(y0/res); nss=ceiling(z0/res)
           ncs=ncs-1; nrs=nrs-1; nss=nss-1
           rdum=xmax-xmin; XL=rdum;
           rdum=ymax-ymin; YL=rdum;
           rdum=zmax-zmin; ZL=rdum;
           alpha=90.0; beta=90.0; gamma=90.0;
           mapc=1 ; mapr=2 ; maps=3
           ISPG=1 ; NSYMBT=0 ; LSKFLG=0
           SKWMAT = (/ 0,0,0, 0,0,0, 0,0,0 /); SKWTRN = (/ 0,0,0 /)
           do k = 1, 15
              EXTRA(k) = 0.0
           enddo
           MAP='MAP' ; MACHST='DA'
           LABEL_N(1)='::::CHARMM::::SOLVMAP::::' // label//'::::'
           write(cdum1,'(F6.3)') res; write(cdum2,'(F6.3)') rcut; 
           LABEL_N(2)='BINSIZE= '// cdum1 //':::: RCUT= '// cdum2
           do k= 3,10
              LABEL_N(k)=''
           enddo

           rdum4 = 0.0
           WRITE(ounit)NX4,NY4,NZ4       !1,2,3
           WRITE(ounit)idum ! MODE                   ! 4
           WRITE(ounit)NCS,NRS,NSS            ! 5,6,7
           WRITE(ounit)NX4,NY4,NZ4               ! 8,9,10
           WRITE(ounit)XL,YL,ZL               ! 11,12,13
           WRITE(ounit)ALPHA,BETA,GAMMA       ! 14,15,16
           WRITE(ounit)MAPC,MAPR,MAPS         ! 17,18,19
           WRITE(ounit)AMIN4,AMAX4,AMEAN4        ! 20,21,22
           WRITE(ounit) ISPG,NSYMBT,LSKFLG     ! 23,24,25
           WRITE(ounit)SKWMAT                 ! 26-34
           WRITE(ounit)SKWTRN                 ! 35-37
           WRITE(ounit)EXTRA                  ! 38-52
           WRITE(ounit)MAP               ! 53
           WRITE(ounit)MACHST                 ! 54
           WRITE(ounit)RMS4                   ! 55
           WRITE(ounit) NLABL                  ! 56
           WRITE(ounit)(LABEL_N(I),I=1,10)      ! 57-256

           do k = 1, npoint
              if ((adata(k) > anum0 ) .and. (adata(k) < anum1 )) then
                 amin4=adata(k) ! cast into real
                 write(ounit) amin4
              else 
                 write(ounit) rdum4
              endif
           enddo

     end select
  endif
  call vclose(ounit,'KEEP',error)         
  end subroutine write_data

!-------------------------------------------------------

subroutine get_cm(x,y,z,xave,yave,zave,solute,nslct,bmass)
! Get center of mass of atoms in solute array, weighed by bmass
! both solute and bmass have size nslct. soute contains selected atom indexes 
! and bmass contains the corresponding weight

  ! argument vars
  real(chm_real) x(*),y(*),z(*),xave,yave,zave
  integer nslct, solute(*)
  real(chm_real) bmass(*)

  ! local var
  real(chm_real) amasst, amassv

  xave=0.0; yave=0.0; zave=0.0; amasst = 0.0
  do i=1,nslct
     j = solute(i)
     amassv=bmass(i);     amasst=amasst+amassv
     xave=xave+x(j)*amassv
     yave=yave+y(j)*amassv
     zave=zave+z(j)*amassv
  enddo
  if(amasst /= zero) then
     xave=xave/amasst
     yave=yave/amasst
     zave=zave/amasst
  else
     call wrndie (-2,'<SMAP>', 'ZERO TOTAL MASS.') 
  endif ! if (mynod .eq. 0 )  then

end subroutine get_cm

end module solvmap
