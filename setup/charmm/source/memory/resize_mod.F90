#ifdef KEY_RESIZE
MODULE resize
  !
  ! This module handles dynamic resizing of arrays, to reduce the memory
  ! footprint of CHARMM. Without resizing, array sizes are fixed at the
  ! runtime (e.g., by using the -chsize command line option or the
  ! DIMEnsion command), leading to spurious memory spaces. With RESIZE,
  ! array sizes are adjusted as structures are built or modified.
  !
  ! NOTE: As of 2022, not all arrays are automatically resized. For other
  ! arrays relying on the maximum sizes (ltm/dimens_ltm.F90), -chsize
  ! should still be used. Yet, reduction in the size of major arrays
  ! including position, internal corrdinate, and force allows the use of a
  ! smaller -chsize.
  !
  ! Author: Wonmuk Hwang, 2022
  !
  use psf
  use coord
  use coordc
  use deriv
  use reallocation
  
  ! drsz0: Resizing arrays every time when NATOM, NRES, NGRP, etc, change
  ! by 1 in a loop can take time. It is faster to resize in increments
  ! drsz0.
  !
  ! rszf: resize factor multiplying the number of elements, to account for
  ! image atoms.
  integer::drsz0, rszf
contains

  subroutine resize_init
    drsz0=10000
    rszf=1
    return
  end subroutine resize_init
  
  subroutine resize_psf(filename,procname,varname,newsize, qrszf)
    ! qrszf: .true. multiply newsize with rszf (for images)
    !        .false. directly use newsize for the array size.  This is
    !        applicable only for natom,nseg,ngrp,nres (updated in image
    !        calls)
    use code,only: icb,ict,icp,ici,icct
    use cheq,only: ech,eha,molt
    use flucqm,only:fqcfor
    use number
    
    character(len=*),intent(in) :: filename,procname,varname
    integer,intent(in) :: newsize
    integer:: nsz0,nsz1
    logical,intent(in) :: qrszf
    ! if (newsize .eq. 0) continue
    nsz0=newsize
    if (qrszf) nsz0=newsize*rszf ! for natom,nseg,ngrp,nres
    ! keep minimum size (needed because for small numbers, fluctuation
    ! in image atom may be more than by 25% set in set_rszf().
    if (nsz0 .lt. drsz0) then
       nsz0=drsz0 
    endif
    ! nsz1: for numbers that do not multiply by rszf but have extra
    ! counts in images
    nsz1=max(newsize,drsz0)
    if (varname == 'NATOM') then
       call chmrealloc(filename,procname,'amass ',nsz0,crl=amass) 
       call chmrealloc(filename,procname,'cg    ',nsz0,crl=cg)
       call chmrealloc(filename,procname,'iac   ',nsz0,intg=iac)
       call chmrealloc(filename,procname,'iblo  ',nsz0,intg=iblo)
       call chmrealloc(filename,procname,'imove ',nsz0,intg=imove)
       call chmrealloc(filename,procname,'icaauto',nsz0,intg=icaauto) ! rszv02
       call chmrealloc(filename,procname,'atype ',nsz0,ch8=atype)
       call chmrealloc(filename,procname,'rsclf ',nsz0,crl=rsclf)
       call chmrealloc(filename,procname,'alphadp',nsz0,crl=alphadp)
       call chmrealloc(filename,procname,'tholei',nsz0,crl=tholei)
       call chmrealloc(filename,procname,'isdrude',nsz0,log=isdrude)
#if KEY_WCA==1
       call chmrealloc(filename,procname,'wca   ',nsz0,crl=wca)
#endif       
#if KEY_LJPME==1
       call chmrealloc(filename,procname,'ljc6  ',nsz0,crl=ljc6)
#endif
       if (nsz0 .gt. size(inb)) then
          ! hwm: check
          !nsz0=max(size(inb),(nsz0+size(inb))/2) ! give additional room
          call chmrealloc(filename,procname,'inb   ',nsz0,intg=inb)
       endif
       !
       ! cheq-related
       call chmrealloc(filename,procname,'molt    ',nsz0,intg=molt)
       call chmrealloc(filename,procname,'ech     ',nsz0, crl=ech)
       call chmrealloc(filename,procname,'eha     ',nsz0, crl=eha)
       
       ! flucq-related
       if (allocated(fqcfor)) &
            call chmrealloc(filename,procname,'FQCFOR',nsz0,crl=FQCFOR)
       
    else if (varname == 'NBOND') then
       nsz0=nsz1 ! hwm: check -> include image bond (nimbon) in resize_array
       call chmrealloc(filename,procname,'ib    ',nsz0,intg=ib)
       call chmrealloc(filename,procname,'jb    ',nsz0,intg=jb)
       call chmrealloc(filename,procname,'icbauto',nsz0,intg=icbauto) ! rszv02
       call chmrealloc(filename,procname,'icb',nsz0,intg=icb) ! rszv02
       !! hwm: check
       !if ( nsz0 .gt. size(idon)) then
       !   call chmrealloc(filename,procname,'idon  ',nsz0,intg=idon)       
       !   call chmrealloc(filename,procname,'ihd1  ',nsz0,intg=ihd1)
       !endif
    else if (varname == 'NDON') then
       nsz0=max(newsize,2) ! don't use rszf regardless of qrszf
       call chmrealloc(filename,procname,'idon  ',nsz0,intg=idon)       
       call chmrealloc(filename,procname,'ihd1  ',nsz0,intg=ihd1)       
    else if (varname == 'NACC') then    
       nsz0=max(newsize,2) ! don't use rszf regardless of qrszf
       call chmrealloc(filename,procname,'iacc  ',nsz0,intg=iacc)       
       call chmrealloc(filename,procname,'iac1  ',nsz0,intg=iac1)       
    else if (varname == 'NCRTERM') then
#if KEY_CMAP==1
       nsz0=nsz1  ! don't use rszf regardless of qrszf
       call chmrealloc(filename,procname,'i1ct  ',nsz0,intg=i1ct)       
       call chmrealloc(filename,procname,'j1ct  ',nsz0,intg=j1ct)       
       call chmrealloc(filename,procname,'k1ct  ',nsz0,intg=k1ct)       
       call chmrealloc(filename,procname,'l1ct  ',nsz0,intg=l1ct)       
       call chmrealloc(filename,procname,'i2ct  ',nsz0,intg=i2ct)       
       call chmrealloc(filename,procname,'j2ct  ',nsz0,intg=j2ct)       
       call chmrealloc(filename,procname,'k2ct  ',nsz0,intg=k2ct)       
       call chmrealloc(filename,procname,'l2ct  ',nsz0,intg=l2ct)
       call chmrealloc(filename,procname,'icct  ',nsz0,intg=icct) ! rszv02
#endif
    else if (varname == 'NGRP') then
       call chmrealloc(filename,procname,'igpbs ',nsz0+1,intg=igpbs)
       call chmrealloc(filename,procname,'igptyp',nsz0,intg=igptyp)
       call chmrealloc(filename,procname,'imoveg',nsz0,intg=imoveg)
    else if (varname == 'NNB') then
       ! inb also resized by NATOM above.
       nsz0=max(newsize,2) ! don't use rszf regardless of qrszf
       call chmrealloc(filename,procname,'inb   ',nsz0,intg=inb)
    else if (varname == 'NIMPHI') then
       nsz0=nsz1 ! don't use rszf regardless of qrszf
       call chmrealloc(filename,procname,'im    ',nsz0,intg=im)    
       call chmrealloc(filename,procname,'jm    ',nsz0,intg=jm)    
       call chmrealloc(filename,procname,'km    ',nsz0,intg=km)    
       call chmrealloc(filename,procname,'lm    ',nsz0,intg=lm)
       call chmrealloc(filename,procname,'ici   ',nsz0,intg=ici) ! rszv02
    else if (varname == 'NPHI') then
       nsz0=nsz1
       call chmrealloc(filename,procname,'ip    ',nsz0,intg=ip)       
       call chmrealloc(filename,procname,'jp    ',nsz0,intg=jp)       
       call chmrealloc(filename,procname,'kp    ',nsz0,intg=kp)       
       call chmrealloc(filename,procname,'lp    ',nsz0,intg=lp)
       call chmrealloc(filename,procname,'icp   ',nsz0,intg=icp) ! rszv02
    else if (varname == 'NTHETA') then
       ! *2: To accommodate in msmmpt.F90, around line 10632, IT(J)
       ! where J>NTHETA (found via c47test/ms_cluster.inp)
       nsz0=nsz1 ! newsize*2 ! hwm: check > verified it's needed.
       call chmrealloc(filename,procname,'it    ',nsz0,intg=it)       
       call chmrealloc(filename,procname,'jt    ',nsz0,intg=jt)       
       call chmrealloc(filename,procname,'kt    ',nsz0,intg=kt)
       call chmrealloc(filename,procname,'ict   ',nsz0,intg=ict) ! rszv02
    else if (varname == 'NRES') then
       call chmrealloc(filename,procname,'ibase ',nsz0+1,intg=ibase)
       call chmrealloc(filename,procname,'res   ',nsz0,ch8=res)
       call chmrealloc(filename,procname,'resid ',nsz0,ch8=resid)
    else if (varname == 'NSEG') then
       call chmrealloc(filename,procname,'nictot',nsz0+1,intg=nictot)
       call chmrealloc(filename,procname,'segid ',nsz0,ch8=segid)
    endif
    return
  end subroutine resize_psf

  subroutine resize_coord(filename,procname,newsize,qrszf)
    character(len=*),intent(in) :: filename,procname
    integer,intent(in) :: newsize
    integer:: nsz0
    logical,intent(in) :: qrszf
    !if (newsize .eq. 0) return ! do nothing
    nsz0=newsize
    if (qrszf) nsz0=newsize*rszf
    if (nsz0 .lt. drsz0) then
       nsz0=drsz0 ! keep minimum size (needed in some cases, like ABNR)
    endif
    call chmrealloc(filename,procname,'x ',nsz0,crl=x)
    call chmrealloc(filename,procname,'y ',nsz0,crl=y)
    call chmrealloc(filename,procname,'z ',nsz0,crl=z)
    call chmrealloc(filename,procname,'wmain ',nsz0,crl=wmain)
    return
  end subroutine resize_coord
  
  subroutine resize_coordc(filename,procname,newsize,qrszf)
    character(len=*),intent(in) :: filename,procname
    integer,intent(in) :: newsize
    integer:: nsz0
    logical,intent(in) :: qrszf
    !if (newsize .eq. 0) return ! do nothing
    nsz0=newsize
    if (qrszf) nsz0=newsize*rszf
    if (nsz0 .lt. drsz0) then
       nsz0=drsz0 ! keep minimum size (needed in some cases, like ABNR)
    endif
    call chmrealloc(filename,procname,'xcomp ',nsz0,crl=xcomp)
    call chmrealloc(filename,procname,'ycomp ',nsz0,crl=ycomp)
    call chmrealloc(filename,procname,'zcomp ',nsz0,crl=zcomp)
    call chmrealloc(filename,procname,'wcomp ',nsz0,crl=wcomp)
#if KEY_COMP2==1
    call chmrealloc(filename,procname,'xcomp2',nsz0,crl=xcomp2)
    call chmrealloc(filename,procname,'ycomp2',nsz0,crl=ycomp2)
    call chmrealloc(filename,procname,'zcomp2',nsz0,crl=zcomp2)
    call chmrealloc(filename,procname,'wcomp2',nsz0,crl=wcomp2)
#endif
    return
  end subroutine resize_coordc

  subroutine resize_deriv(filename,procname,newsize,qrszf)
    character(len=*),intent(in) :: filename,procname
    integer,intent(in) :: newsize
    integer:: nsz0
    logical,intent(in) :: qrszf
    !if (newsize .eq. 0) return ! do nothing
    nsz0=newsize
    if (qrszf) nsz0=newsize*rszf
    if (nsz0 .lt. drsz0) then
       nsz0=drsz0 ! keep minimum size (needed in some cases, like ABNR)
    endif
    call chmrealloc(filename,procname,'dx ',nsz0,crl=dx)
    call chmrealloc(filename,procname,'dy ',nsz0,crl=dy)
    call chmrealloc(filename,procname,'dz ',nsz0,crl=dz)
    return
  end subroutine resize_deriv

  subroutine resize_bimag(bimag)
    ! bimag can be from bases_fcm or from other modules
    use chm_types ! Has imageDataStructure
    type(imageDataStructure) BIMAG
    integer:: nsz0

    nsz0=max(1,size(cg)) ! atom-based
    if (nsz0/=size(bimag%IMATTR)) then
       call chmrealloc('resize_mod.F90','resize_bimag','bimag%IMATTR',nsz0,intg=bimag%IMATTR)  
       call chmrealloc('resize_mod.F90','resize_bimag','bimag%IMINB',nsz0,intg=bimag%IMINB)
#if KEY_IMCUBES==1
       call chmrealloc('resize_mod.F90','resize_bimag','bimag%IMIBLO',nsz0*2,intg=bimag%IMIBLO)
#else /**/
       call chmrealloc('resize_mod.F90','resize_bimag','bimag%IMIBLO',nsz0,intg=bimag%IMIBLO)
#endif
    endif
    nsz0=max(1,size(imoveg)) ! group-based
    if (nsz0/=size(bimag%IMING)) then
       call chmrealloc('resize_mod.F90','resize_bimag','bimag%IMING',nsz0,intg=bimag%IMING)
       call chmrealloc('resize_mod.F90','resize_bimag','bimag%IMIGLO',nsz0,intg=bimag%IMIGLO)
    endif
    return
  end subroutine resize_bimag

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine set_rszf(filename,procname,qimchng)
    ! Sets the resizing factor rszf. It is for the case when the crystal is
    ! present but no information is available regarding the number of image
    ! atoms (i.e., image updating has not been done; MKIMAT/MKIMAT2 not
    ! called.  Using cutim (image cutoff) and xucell(*) (crystal
    ! dimension), the factor (rszf) multiplying arrays containing image
    ! atoms is determined.
    ! In case image info is available (from previous calls), attempt
    ! to use it.
    
    use image
    use number
    use stream
    use bases_fcm,only: bimag
    implicit none
    character(len=*),intent(in) :: filename,procname
    logical,intent(in):: qimchng ! whether there has been changes in CUTIM
    !logical:: qset
    integer i,j,k,rszf0,i0
    integer IQ,IS,NAT,XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX
    real(chm_real):: r0(3),r(3),cut0,rdum0,rdum1,rdum2,rtot

    if (ntrans .eq. 0 ) then
       return ! no image present.
    endif
    rszf0=rszf

    ! Determine the effective cutim (cut0) including the group size.  This
    ! block is based on upimag.F90:MKIMAT2()
    rdum1=0.
    DO I=1,NGRP
       IS=IGPBS(I)+1
       IQ=IGPBS(I+1)
       NAT=IQ-IS+1
       IF(NAT.LE.0) CALL WRNDIE(-5,'<SET_RSZF>', &
            'Group with no atoms found')
       XMIN=X(IS)
       XMAX=XMIN
       YMIN=Y(IS)
       YMAX=YMIN
       ZMIN=Z(IS)
       ZMAX=ZMIN
       DO J=IS+1,IQ
          IF(X(J).LT.XMIN) XMIN=X(J)
          IF(Y(J).LT.YMIN) YMIN=Y(J)
          IF(Z(J).LT.ZMIN) ZMIN=Z(J)
          IF(X(J).GT.XMAX) XMAX=X(J)
          IF(Y(J).GT.YMAX) YMAX=Y(J)
          IF(Z(J).GT.ZMAX) ZMAX=Z(J)
       ENDDO
       !  Size of rectangular box surrounding group
       r0(1)=(XMAX-XMIN)*0.5 ! half of x-width of group
       r0(2)=(YMAX-YMIN)*0.5 ! half of y-width
       r0(3)=(ZMAX-ZMIN)*0.5 ! half of z-width
       rdum1 =max(r0(1),r0(2),r0(3)) !largest dimension of any group
    ENDDO
    
    if (cutim<13.5) then
       cut0=13.5 ! 13.5: image_module.F90:image_init (default)
    else if (cutim>100.) then ! too large
       cut0=100. 
    else
       cut0=cutim
    endif
    cut0=cut0+rdum1

    ! Determine volume fraction of the region around the primary cell
    ! within cut0 (image cutoff).

    ! 0) Case of orthorhombic image (to develop general expression)
    !
    ! For each face of the cell, there is a slice of thickness cut0,
    ! its volume fraction is cut0*(1/Lx+1/Ly+1/Lz).
    !
    ! Around each slice, there are 4 bars with cross-sectional area cut0^2
    ! and length equal to that of the edge of the slice.  A pair of bars on
    ! the opposing edges have the same volume and they are each shared by 4
    ! cells -> Multiply the volume fraction by 2.
    !
    ! At each corner of the cell there is a cube of volume cut0^3. For
    ! each face, there are 4 corners, each shared by 8 cells ->
    ! Multiply the volume fraction by 2.

    ! 1) Triclinic: Difference from the orthorhombic case is

    ! Volume of the unit cell is multiplied by a factor
    ! sqrt(1-\sum_i\cos^2\theta_i+\prod_i\cos\theta_i) (i=1,2,3)
    !
    ! Distance from the boundary is the normal distance, which is
    ! cut0/\sin\theta_i. Ignore this here, which won't affect the
    ! estimation.

    rdum2=1.D0
    do i=1,6 ! find if any of xucell(i) is zero
       rdum2=rdum2*xucell(i)
    enddo
    
    if (rdum2 > PT001) then ! check if unit cell is defined
       rtot = cut0*( 1.d0/xucell(1)+1.d0/xucell(2)+1.d0/xucell(3)) ! each face
       rdum0= 1.d0/(xucell(1)*xucell(2))+ 1.d0/(xucell(2)*xucell(3)) &
            +1.d0/(xucell(3)*xucell(1)) ! vol. fraction of bars
       rtot=rtot+2.d0*cut0*cut0*rdum0 ! 2*: explained above
       rdum1=1.d0 ! unit cell vol
       do i=1,3
          rdum1=rdum1*xucell(i)
       enddo
       rtot=rtot+2.d0*cut0*cut0*cut0/rdum1 ! Cubes in the corner.

       do i=1,3 ! xucell(4:6): in degrees
          r0(i)=cos(3.1415926358979D0*xucell(i+3)/180.D0) 
       enddo
       rdum1=1.D0 ! unit cell volume factor
       do i=1,3
          rdum1=rdum1-r0(i)*r0(i)
       enddo
       rdum1=rdum1+2*r0(1)*r0(2)*r0(3)
       rdum1=sqrt(rdum1)
       rtot=rtot/rdum1
       rszf=max(ceiling(rtot),2)
    else 
       rszf=1 ! unit cell undefined
    endif
    
    IF(PRNLEV.GE.4) WRITE(OUTU,*) '     <RESIZE>: Array sizes adjusted:'
13  FORMAT(12X,A16,' from ',I9,' to ',I9)
    i0=0 ! number of resized arrays

    ! Atom-based arrays
    if ((natim .le. natom) .or. qimchng) then
       k=natom*6/5*rszf
    else 
       k=natim*6/5 ! use existing image
    endif
    j=size(cg)
    if ( k .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NATOM',k, .false.)
       if (j .ne. size(cg)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Atom:',j,size(cg)
          i0=i0+1
       endif
          
    endif
    j=size(x)
    if ( k .ne. j) then
       call resize_coord('resize_mod.F90','set_rszf',k, .false.)
       call resize_coordc('resize_mod.F90','set_rszf',k, .false.)
       call resize_deriv('resize_mod.F90','set_rszf',k, .false.)
       if (j .ne. size(x)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Coord:',j,size(x)
          i0=i0+1
       endif
    endif

    ! Group-based arrays
    !i=ngrp*6/5;   if (i .lt. 2) i=2
    !k=max(i*rszf,nimgrp*6/5)
    if ((nimgrp .le. ngrp) .or. qimchng) then
       k=ngrp*rszf*6/5
    else
       k=nimgrp*6/5
    endif
    j=size(imoveg)
    if ( k .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NGRP',k,.false.)
       if (j .ne. size(imoveg)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Group:',j,size(imoveg)
          i0=i0+1
       endif
    endif

    ! Residue-based arrays
    if ((nrest .le. nres) .or.  qimchng) then
       k=nres*6/5*rszf
    else
       k=nrest*6/5
    endif
    !i=nres*6/5;  if (i .lt. 2) i=2
    !k=i*rszf
    j=size(resid)
    if ( k .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NRES',k,.false.)
       if (j .ne. size(resid)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Residue:',j,size(resid)
          i0=i0+1
       endif
    endif

    ! Segment-based arrays
    if ((nsegt .le. nseg) .or.  qimchng) then
       k=nseg*6/5*rszf
    else
       k=nsegt*6/5
    endif
    !i=nseg*rszf*6/5;  if (i .lt. 2) i=2
    !k=i*rszf
    j=size(segid)
    if ( k .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NSEG',k,.false.)
       if (j .ne. size(segid)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Segment:',j,size(segid)
          i0=i0+1
       endif
    endif

    ! Below do not involve rszf. But update to reduce array size
    ! in case a previous operation set their sizes larger than needed.

    !i=(nbond+nimbon)*6/5 ! updated based on BRB's suggestion on 220722
    if ((nbondt2 .le. nbond) .or. qimchng) then
       i=nbond*6/5
    else
       i=nbondt2*6/5
    endif

    if ( i .ne. size(ib)) then
       j=size(ib)
       call resize_psf('resize_mod.F90','set_rszf','NBOND',i,.false.)
       if (j .ne. size(ib)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Bond:',j,size(ib)
          i0=i0+1
       endif
    endif
    i=max(ndon,2)
    j=size(idon)
    if ( i .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NDON',i,.false.)
       if (j .ne. size(idon)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Donor:',j,size(idon)
          i0=i0+1
       endif
    endif
    i=max(nacc,2)
    j=size(iacc)
    if ( i .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NACC',i,.false.)
       if (j .ne. size(iacc)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Acceptor:',j,size(iacc)
          i0=i0+1
       endif
    endif
#if KEY_CMAP==1
    i=(ncrterm+nimcrt)*6/5
    j=size(i1ct)
    if ( i .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NCRTERM',i,.false.)
       if (j .ne. size(i1ct)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Cross-term:',j,size(i1ct)
          i0=i0+1
       endif
    endif
#endif
    i=(ntheta+nimang)*6/5
    j=size(it)
    if ( i .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NTHETA',i, .false.)
       if (j .ne. size(it)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Angle:',j,size(it)
          i0=i0+1
       endif
    endif
    i=(nphi+nimdih)*6/5
    j=size(ip)
    if ( i .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NPHI',i, .false.)
       if (j .ne. size(ip)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Dihedral',j,size(ip)
          i0=i0+1
       endif
    endif
    i=(nimphi+nimimp)*6/5
    j=size(im)
    if ( i .ne. j) then
       call resize_psf('resize_mod.F90','set_rszf','NIMPHI',i, .false.)
       if (j .ne. size(im)) then
          IF(PRNLEV.GE.4) WRITE(OUTU,13) 'Improper',j,size(im)
          i0=i0+1
       endif
    endif
    IF(PRNLEV.GE.4) WRITE(OUTU,15) i0
15  FORMAT(12X,'A total of ',I6,' array families were resized')
    call resize_bimag(bimag)
    return
  end subroutine set_rszf

  subroutine resize_array(filename,procname)
    ! Resize arrays when image info is available. Multiply current number
    ! of elements by 6/5 to allow for 25% fluctuation upon image update
    ! during dynamics run.
    ! 
    use stream
    use image
    use bases_fcm,only: bimag
    character(len=*),intent(in) :: filename,procname
    integer i0,i1
    !logical:: qset
    
    if ( natim .le. natom ) then ! image not properly set
       call set_rszf('resize_mod.F90','resize_array',.false.) 
    else
       i0 = (nthet+nimang)*6/5
       i1 = size(it)
       if (i0 .ne. i1) then ! return ! no change in size
          call resize_psf('resize_mod.F90','resize_array','NTHET',i0,.false.)
          if (i1 .ne. size(it)) then
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Angle', i1,i0
          endif
       endif
       !
       i0 = natim*6/5 ! give a 25% room for fluctuation
       i1 = size(x)
       !if (i0 .eq. i1) return ! no change in size
       if (i0 .ne. i1) then
          call resize_psf('resize_mod.F90','resize_array','NATOM',i0,.false.)
          call resize_coord('resize_mod.F90','resize_array',i0,.false.)
          call resize_coordc('resize_mod.F90','resize_array',i0,.false.)
          call resize_deriv('resize_mod.F90','resize_array',i0,.false.)
          if ( i1 .ne. size(x)) then ! i1 is the old size of the array.
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Atom', i1,i0
          endif
       endif
       !
       i0 = (nbond+nimbon)*6/5
       i1 = size(ib)
       !if (i0 .eq. i1) return ! no change in size
       if (i0 .ne. i1) then !hwm2206
          call resize_psf('resize_mod.F90','resize_array','NBOND',i0,.false.)
          if (i1 .ne. size(ib)) then
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Bond', i1,i0
          endif
       endif
       !
#if KEY_CMAP==1       
       i0 = (ncrterm+nimcrt)*6/5
       i1 = size(i1ct)
       !if (i0 .eq. i1) return ! no change in size
       if (i0 .ne. i1) then !hwm2206
          call resize_psf('resize_mod.F90','resize_array','NCRTERM',i0,.false.)
          if (i1 .ne. size(i1ct)) then
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Cross-term', i1,i0
          endif
       endif
#endif
       !
       i0 = (nphi+nimdih)*6/5
       i1 = size(ip)
       !if (i0 .eq. i1) return ! no change in size
       if (i0 .eq. i1) then
          call resize_psf('resize_mod.F90','resize_array','NPHI',i0,.false.)
          if (i1 .ne. size(ip)) then
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Dihedral', i1,i0
          endif
       endif
       !
       i0 = nimgrp*6/5
       i1 = size(imoveg)
       !if (i0 .eq. i1) return ! no change in size
       if (i0 .ne. i1) then ! hwm2206
          call resize_psf('resize_mod.F90','resize_array','NGRP',i0,.false.)
          if (i1 .ne. size(imoveg)) then
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Group', i1,i0
          endif
       endif
       !
       i0 = (nimphi+nimimp)*6/5
       i1 = size(im)
       !if (i0 .eq. i1) return ! no change in size
       if (i0 .ne. i1) then ! hwm2206
          call resize_psf('resize_mod.F90','resize_array','NIMPHI',i0,.false.)
          if (i1 .ne. size(im)) then
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Improper', i1,i0
          endif
       endif
       !
       i0 = nrest*6/5
       i1 = size(resid)
       !if (i0 .eq. i1) return ! no change in size
       if (i0 .ne. i1) then ! hwm2206
          call resize_psf('resize_mod.F90','resize_array','NRES',i0,.false.)
          if (i1 .ne. size(resid)) then
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Residue', i1,i0
          endif
       endif
       !
       i0 = nsegt*6/5
       i1 = size(segid)
       !if (i0 .eq. i1) return ! no change in size
       if (i0 .ne. i1) then ! hwm2206
          call resize_psf('resize_mod.F90','resize_array','NSEG',i0,.false.)
          if (i1 .ne. size(segid)) then
             IF(PRNLEV.GE.4) WRITE(OUTU,14) 'Segment', i1,i0
          endif
       endif
       !
       call resize_bimag(bimag)
    endif
14  FORMAT(5X,'<RESIZE>:',A8,' array size changed from ',I9,' to ',I9,'.')
    return
  end subroutine resize_array
    
END MODULE resize
#endif /* KEY_RESIZE */
