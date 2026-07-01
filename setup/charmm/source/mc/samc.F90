module samc
!   by Florent Hedin & Markus Meuwly, University of Basel, Switzerland
!   florent.hedin@unibas.ch
!   October, 2011 - January, 2014
!
!   See the following references:
!
!   1) J.D. Doll et al., J. Chem. Phys., 131, 104107, (2009)
!   2) N.Plattner et al., J. Chem. Phys., 133, 044506, (2010)
!   3) F. Hedin et al., J. Chem. Theory Comput.,DOI: 10.1021/ct500529w (2014)

#if KEY_MC==1 /*(mc)*/
#if KEY_SAMC==1 /*(samc)*/
  !modules commonly used by subroutines of this file
  use chm_types, only: chm_ptr, iptr_ptr
  use chm_kinds, only: chm_real, chm_int4
  use stream, only: outu,prnlev
  use memory
  use number
  use mc, only: mmvtyp

  implicit none

  !logicals used for spatial averaging simulations
  !if spatial averaging is used, lsamc is set to TRUE
  !if lsamc is true, lspdone is used: it is set to FALSE at the beginning of each itelation of the MCLOOP
  !     and became TRUE if spatial averaging was applied for the current instance
  logical,save :: lspmv,lsamc,lspdone
  logical,dimension(mmvtyp),save :: lspgroup

  !arrays of spatial averaging parameters
  real(chm_real),dimension(mmvtyp),target,save :: weparr
  integer(chm_int4),dimension(mmvtyp),target,save :: meparr,neparr

  !variables pointing to some parts of the previous arrays
  real(chm_real),pointer,save :: wepsilon
  integer(chm_int4),pointer,save :: mepsilon,nepsilon

  integer(chm_int4),save :: maxmeps,maxneps

  !desamc is the spatial averaging acceptance criterion
  real(chm_real),save :: desamc

  !if unbiasing of spatial averaging is required, lunb set to TRUE
  !     unbiofile is the number of the stream unit for I/O
  !     unb contains the unbiasing factor: it is the biased density rho(epsilon); what is written to unbiofile
  !             is the ratio rho(0)/rho(epsilon) see litterature for more details.
  logical,save :: lunb
  integer(chm_int4),save :: unbiofile
  real(chm_real),save :: unb

  !arrays for storing temporary copies of coordinates (coorfirst,coorafter)
  !     or distribution of coordinates (coorini,coorfin)
  !     or distribution of energies (eneini,enefin)
  real(chm_real),allocatable,dimension(:),target,save :: coorfirst,coorafter
  real(chm_real),allocatable,dimension(:,:),save :: eneini,enefin
  real(chm_real),allocatable,dimension(:,:,:),target,save :: coorini,coorfin

  !pointers to coordinates
  real(chm_real),pointer,dimension(:),save :: spcx,spcy,spcz

#if KEY_ACE==1
  real(chm_real),allocatable,dimension(:),save :: spesr1p,spesr2p
  real(chm_real),allocatable,dimension(:,:),save :: spaceelold,spaceelnew
#endif

  contains
!--------------------------------------------------------------------
  subroutine spgetparamsbygroup(comlyn,comlen,nmvtyp,mvtype,mvcode)
    ! This is called by the MOVEAD subroutine from movead.src for each type of move
    ! Each instance move can use dedicated values for we, me and ne.
    use string, only: gtrmi,gtrmf,indxa

    implicit none

    !received variables
    character(len=*),intent(in)  :: comlyn
    integer(chm_int4),intent(in) :: comlen
    integer(chm_int4),intent(in) :: nmvtyp,mvtype
    character(len=4),intent(in)  :: mvcode

    !local variables
    logical :: lavail

    lspmv = indxa(comlyn,comlen,'SAMC') > 0

    lavail = (mvtype == 1 .or. mvtype == 2 .or. mvtype == 3 .or. mvtype == 4)
    if(lspmv .and. .not. lavail) then
        call wrndie(-5,'<SPINITCHECK>','SAMC IS NOT AVAILABLE FOR THIS TYPE MOVE : '//mvcode)
    endif

    if(lspmv) then

        weparr(nmvtyp) = gtrmf(comlyn,comlen,'WEPS',ZERO)
        meparr(nmvtyp) = gtrmi(comlyn,comlen,'MEPS',0)
        neparr(nmvtyp) = gtrmi(comlyn,comlen,'NEPS',0)
        lspgroup(nmvtyp) = .true.

        if ( (weparr(nmvtyp) <= 0.0) .or. (meparr(nmvtyp) <= 0) .or. (neparr(nmvtyp) <= 0) ) then
        call wrndie(-5,'<SPINITCHECK>','SAMC DEFINED : WEPS, MEPS, NEPS HAVE TO BE GREATER THAN ZERO')
        endif

        if (prnlev .ge. 2) then
            write(outu,'(a,i4,a,f7.3,i4,i4)') ' SPGETPARAMSBYGROUP> FOR NMVTYP = ',nmvtyp,&
                    ' PARAMETERS ARE : ',weparr(nmvtyp),meparr(nmvtyp),neparr(nmvtyp)
        endif

    endif

    return
  end subroutine spgetparamsbygroup
!--------------------------------------------------------------------
  subroutine spinitcheck(iaccpf)
    implicit none

    !received variables
    integer(chm_int4),intent(in) :: iaccpf

    if (iaccpf == 4 .and. .not. lsamc) then
      call wrndie(-5,'<SPINITCHECK>','IACC IS 4 BUT SAMC IS NOT USED : FATAL ERROR')
    endif

    if (lsamc) then

      if(prnlev .ge. 2) then
        write(outu,'(a)') ' SPINITCHECK> SPAT AVG (SAMC) IS DEFINED.'
      endif

      if (iaccpf .ne. 4) then
        call wrndie(-5,'<SPINITCHECK>','SAMC DEFINED : PLEASE ONLY USE 4 FOR IACC')
      endif

    endif

    maxmeps = maxval(meparr)
    maxneps = maxval(neparr)

    if(maxmeps == 0 .or. maxneps == 0) then
        call wrndie(-5,'<SPINITCHECK>','SAMC AND IACC DEFINED ON THE MC COMMAND LINE BUT '&
                    //'NO SAMC PARAMETERS DEFINED FOR THE MOVE SELECTIONS.')
    endif

    if (prnlev .ge. 2) then
        write(outu,'(a,i4,i4)') ' SPINITCHECK> MAX MEPS AND NEPS ARE : ',maxmeps,maxneps
    endif

    wepsilon => null()
    mepsilon => null()
    nepsilon => null()

    return
  end subroutine spinitcheck
!--------------------------------------------------------------------
  subroutine spallocarrays(natom)
    implicit none

    !received variables
    integer(chm_int4),intent(in) :: natom

    call chmalloc('samc.src','spallocarrays','coorfirst',3*natom,crl=coorfirst)
    call chmalloc('samc.src','spallocarrays','coorafter',3*natom,crl=coorafter)
    call chmalloc('samc.src','spallocarrays','coorini',maxmeps,maxneps,3*natom,crl=coorini)
    call chmalloc('samc.src','spallocarrays','coorfin',maxmeps,maxneps,3*natom,crl=coorfin)
    call chmalloc('samc.src','spallocarrays','eneini',maxmeps,maxneps,crl=eneini)
    call chmalloc('samc.src','spallocarrays','enefin',maxmeps,maxneps,crl=enefin)

    return
  end subroutine spallocarrays
!--------------------------------------------------------------------
  subroutine spdeallocarrays(natom)
    implicit none

    !received variables
    integer(chm_int4),intent(in) :: natom

    call chmdealloc('samc.src','spdeallocarrays','coorfirst',3*natom,crl=coorfirst)
    call chmdealloc('samc.src','spdeallocarrays','coorafter',3*natom,crl=coorafter)
    call chmdealloc('samc.src','spdeallocarrays','coorini',maxmeps,maxneps,3*natom,crl=coorini)
    call chmdealloc('samc.src','spdeallocarrays','coorfin',maxmeps,maxneps,3*natom,crl=coorfin)
    call chmdealloc('samc.src','spdeallocarrays','eneini',maxmeps,maxneps,crl=eneini)
    call chmdealloc('samc.src','spdeallocarrays','enefin',maxmeps,maxneps,crl=enefin)

    return
  end subroutine spdeallocarrays
!--------------------------------------------------------------------
  subroutine spcopycoordinates(natom,x,y,z,linitial)
    implicit none

    !received variables
    integer(chm_int4),intent(in) :: natom
    real(chm_real),dimension(natom),intent(in) :: x,y,z
    logical,intent(in) :: linitial

    if(linitial)then
      coorfirst(1:natom) = x(:)
      coorfirst(natom+1:2*natom) = y(:)
      coorfirst(2*natom+1:3*natom) = z(:)
    else
      coorafter(1:natom) = x(:)
      coorafter(natom+1:2*natom) = y(:)
      coorafter(2*natom+1:3*natom) = z(:)
    endif

    return
  end subroutine spcopycoordinates
!--------------------------------------------------------------------
  subroutine spgaussmove(iseed,stddev,dx,thresh)
    use rndnum
    use clcg_mod,only:bmgaus
    implicit none

    !received variables
    integer(chm_int4),intent(in) :: iseed
    real(chm_real),intent(in) :: thresh,stddev
    real(chm_real),dimension(3),intent(out) :: dx

    !local variables
    real(chm_real) :: norm2

    do
      dx(1) = bmgaus(stddev,iseed)*thresh
      dx(2) = bmgaus(stddev,iseed)*thresh
      dx(3) = bmgaus(stddev,iseed)*thresh
      norm2 = dx(1)*dx(1) + dx(2)*dx(2) + dx(3)*dx(3)
      if ( norm2 <= thresh*thresh ) exit
    enddo

    return
  end subroutine spgaussmove
!--------------------------------------------------------------------
  subroutine spgenerate(natom,aniso,idx,mdxp,jmvtyp,ipivtp,imvngp,mvtype,nlimit,iseed,linitial)
    use rndnum
    use clcg_mod,only:bmgaus
    use mcmvrtrn
    use corsubs,only:fndu
    implicit none

    !received variables
    integer(chm_int4),intent(in) :: natom
    logical,intent(in) :: aniso
    integer(chm_int4),intent(in) :: idx
    type(chm_ptr),intent(in) :: mdxp
    integer(chm_int4),intent(in) :: jmvtyp
    type(iptr_ptr),intent(in) :: ipivtp,imvngp
    integer(chm_int4),intent(in) :: mvtype,nlimit
    integer(chm_int4),intent(in) :: iseed
    logical,intent(in) :: linitial

    !local variables
    integer(chm_int4) :: i,j,k,l,m,n
    integer(chm_int4) ng,nn,firs,last
    real(chm_real) :: u(9),rn(3),dx(3),thresh
    integer(chm_int4) :: iaf,ial,iatom
    logical :: lok,qmvloc

    real(chm_real),pointer,dimension(:) :: coortomod
    real(chm_real),pointer,dimension(:,:,:) :: coormod
    integer(chm_int4),pointer,dimension(:) :: tempp,tempp3
    real(chm_real),pointer :: tempp2

    if(linitial)then
      coortomod => coorfirst
      coormod => coorini
    else
      coortomod => coorafter
      coormod => coorfin
    endif

    tempp => imvngp%a(idx)%a    !moving group for a given idx
    tempp2 => mdxp%a(idx)       !move limit for a given idx
    thresh = tempp2

    spcx => null()
    spcy => null()
    spcz => null()

    lok = .true.
    qmvloc = .true.

    do i=1,mepsilon
      do j=1,nepsilon

        !coordinates copy and pointers use
        coormod(i,j,:)=coortomod(:)
        spcx => coormod(i,j,1:natom)
        spcy => coormod(i,j,natom+1:2*natom)
        spcz => coormod(i,j,2*natom+1:3*natom)
        !depending of type of move, generate spatial averaging configurations
        if (mvtype == 1) then
          !rtrn
          dx(1) = bmgaus(wepsilon,iseed)*thresh
          dx(2) = bmgaus(wepsilon,iseed)*thresh
          dx(3) = bmgaus(wepsilon,iseed)*thresh
          call trnall(spcx,spcy,spcz,tempp,dx(1),dx(2),dx(3))
        else if (mvtype == 2) then
          !rrot
          tempp3 => ipivtp%a(idx)%a !pivots for a given idx (i.e. for finding list of moving atoms)
          call spmkrrot(dx(1),spcx,spcy,spcz,tempp,tempp3(1), &
            thresh, iseed, wepsilon)
        else if (mvtype == 3) then
          !single cartesian atom move
          dx(1) = bmgaus(wepsilon,iseed)*thresh
          dx(2) = bmgaus(wepsilon,iseed)*thresh
          dx(3) = bmgaus(wepsilon,iseed)*thresh
          iatom = tempp(3)
          call rigtrn(spcx,spcy,spcz,iatom,iatom,dx(1),dx(2),dx(3))
        else if (mvtype == 4) then
          !dihedral rotation
          tempp3 => ipivtp%a(idx)%a
          rn(1) = spcx(tempp3(2)) - spcx(tempp3(1))
          rn(2) = spcy(tempp3(2)) - spcy(tempp3(1))
          rn(3) = spcz(tempp3(2)) - spcz(tempp3(1))
          dx(1) = bmgaus(wepsilon,iseed) * thresh
          call fndu(u,rn,dx(1),lok)
          n = tempp(2)
          do l = 4, n, 2
            iaf = tempp(l - 1)
            ial = tempp(l)
            do m=iaf, ial
              call applyu(u,spcx,spcy,spcz,m,spcx(tempp3(2)),spcy(tempp3(2)),spcz(tempp3(2)))
            enddo
          enddo
        endif

      enddo
    enddo

    return
  end subroutine spgenerate
!--------------------------------------------------------------------
  subroutine spgetcriterion(beta,natom,lacemc)
    implicit none

    !received variables
    real(chm_real),intent(in) :: beta
    integer(chm_int4),intent(in) :: natom
    logical,intent(in) :: lacemc

    !local variables
    integer(chm_int4) :: i,j
    real(chm_real) :: sm_old,sm_new,delta_tot
    real(chm_real),dimension(mepsilon) :: deltam_boltz
    real(chm_real) :: sigma2

#if KEY_ACE==1
    if(lacemc)then
        eneini = eneini + spaceelold
        enefin = enefin + spaceelnew
    endif
#endif

    eneini=exp(-beta*eneini)
    enefin=exp(-beta*enefin)

    !then calculate for each set sm and deltam
    do i=1,mepsilon
      sm_old = ZERO
      sm_new = ZERO
      do j=1,nepsilon
        sm_old = sm_old + eneini(i,j)
        sm_new = sm_new + enefin(i,j)
      enddo
      deltam_boltz(i) = -log(sm_new/sm_old)
    enddo

    !delta_tot is the average energy resulting from spatial averaging
    delta_tot = ZERO
    do i=1,mepsilon
      delta_tot = delta_tot + deltam_boltz(i)
    enddo
    delta_tot = (ONE/mepsilon)*delta_tot

    !sigma2 is the variance
    sigma2 = ZERO
    do i=1,mepsilon
      sigma2 = sigma2 + (deltam_boltz(i)-delta_tot)*(deltam_boltz(i)-delta_tot)
    enddo
    sigma2 = (ONE/(mepsilon*(mepsilon-ONE)))*sigma2

    !so the acc/rej parameter is delta_tot + sigma2/2
    desamc = delta_tot + (sigma2/TWO)
    unb = desamc

    return
  end subroutine spgetcriterion
!--------------------------------------------------------------------
  subroutine spmkrrot(dx,x,y,z,imvng,ipvt,rmdx,iseed,stddev)
  !       Make a rigid rotations to the atoms between IAF and IAL.
  !       original mkrrot by Aaron R. Dinner
  !       this modified version by Florent Hedin
      use clcg_mod,only:bmgaus
      use consta
      use psf
      use mcmvrtrn
      use corsubs,only:fndu

      integer ipvt, iseed, imvng(:)
      real(chm_real)  dx, x(*), y(*), z(*), rmdx

      integer i, j, k, l, iaf, ial, ng, nn
      real(chm_real)  rn(3), u(9), xp, yp, zp, mt
      logical lok

      real(chm_real) stddev

      call sprosphr(stddev,iseed,rn)

      dx = bmgaus(stddev,iseed)*rmdx

      call fndu(u,rn,dx,lok)
      if (.not. lok) call wrndie(-5,'<SPMKRROT>','INTERNAL ERROR')

      if (ipvt > 0) then
        xp = x(ipvt)
        yp = y(ipvt)
        zp = z(ipvt)
      else
        call mvgcom(xp,yp,zp,imvng,x,y,z)
      endif

      ng = imvng(1)
      j = ng + 2
      do i = 2, ng
        nn = imvng(i)
        do k = j, nn, 2
          iaf = imvng(k-1)
          ial = imvng(k)
          do l = iaf, ial
            call applyu(u,x,y,z,l,xp,yp,zp)
          enddo
        enddo
        j = nn + 2
      enddo

      return
  end subroutine spmkrrot
!--------------------------------------------------------------------
  subroutine sprosphr(stddev,iseed,v)
  !       Returns a random vector on a unit sphere.
  !       original rosphr by Aaron R. Dinner
  !       this modified version by Florent Hedin
    use rndnum
    use clcg_mod,only:bmgaus
    implicit none

    !received variables
    real(chm_real),intent(in) :: stddev
    integer(chm_int4),intent(in) :: iseed
    real(chm_real),dimension(3),intent(out) :: v

    do
      v(1) = bmgaus(stddev,iseed)
      v(2) = bmgaus(stddev,iseed)
      v(3) = v(1)*v(1) + v(2)*v(2)
      if (v(3)<=one) exit
    enddo

    v(1) = two*v(1)*sqrt(one - v(3))
    v(2) = two*v(2)*sqrt(one - v(3))
    v(3) = one - two*v(3)

    return
  end subroutine sprosphr
!--------------------------------------------------------------------
  logical function spacpt(beta,iseed)
    use clcg_mod,only:random
    implicit none

    !passed variables
    real(chm_real),intent(in) :: beta
    integer(chm_int4),intent(in) :: iseed

    !local variables
    real(chm_real) :: alpha

    alpha = random(iseed)

    if (desamc <= ZERO) then
      spacpt = .true.
    else
      spacpt = (exp(-beta*desamc) >= alpha)
    endif

    return
  end function spacpt
!--------------------------------------------------------------------
#if KEY_ACE==1 /* (ace_sp) */
  subroutine spaceallocarrays(natom)
    implicit none

    !received variables
    integer(chm_int4),intent(in) :: natom

    call chmalloc('samc.src','spaceallocarrays','spesr1p',natom,crl=spesr1p)
    call chmalloc('samc.src','spaceallocarrays','spesr2p',natom,crl=spesr2p)
    call chmalloc('samc.src','spaceallocarrays','spaceelold',maxmeps,maxneps,crl=spaceelold)
    call chmalloc('samc.src','spaceallocarrays','spaceelnew',maxmeps,maxneps,crl=spaceelnew)

    return
  end subroutine spaceallocarrays
!--------------------------------------------------------------------
  subroutine spacedeallocarrays(natom)
    implicit none

    !received variables
    integer(chm_int4),intent(in) :: natom

    call chmdealloc('samc.src','spaceallocarrays','spesr1p',natom,crl=spesr1p)
    call chmdealloc('samc.src','spaceallocarrays','spesr2p',natom,crl=spesr2p)
    call chmdealloc('samc.src','spaceallocarrays','spaceelold',maxmeps,maxneps,crl=spaceelold)
    call chmdealloc('samc.src','spaceallocarrays','spaceelnew',maxmeps,maxneps,crl=spaceelnew)

    return
  end subroutine spacedeallocarrays
!--------------------------------------------------------------------
  subroutine spacecontribution(inblo,jnb,ifrsta,natomx, &   !for GTESLF
             ces1,ces2,sig2i,mue4,cswit,c2ofnb,c2onnb, &    !for GTESLF
             ibcut,ncutbc,ecut,ibchg,nacebc, &              !for ADESLF
             esmcfx,esarr,bsarr,cg2arr,rsys,notadd, &               !for MCACEE
             mcblo,mca14,fact2,fact2h,eturn,eturnh,meref,merefh,natom)    !for MCACEE

    use mcace,only:stupbc,gteslf,adeslf,mcacee

    implicit none

    !received variables
    !GTSELF
    integer(chm_int4),dimension(:),intent(in) :: inblo,jnb
    integer(chm_int4),intent(in) :: ifrsta, natomx
    real(chm_real),dimension(:,:),intent(in) :: ces1,sig2i,mue4
    real(chm_real),dimension(:),intent(in) :: ces2
    logical,intent(in) :: cswit
    real(chm_real),intent(in) :: c2ofnb, c2onnb
    !ADESLF
    integer(chm_int4),dimension(:),intent(in) :: ibcut,ibchg
    real(chm_real),intent(in) :: ecut
    integer(chm_int4),intent(in) :: ncutbc,nacebc
    !MCACEE
    real(chm_real),dimension(:),intent(in) :: esmcfx,esarr,bsarr,cg2arr
    real(chm_real),intent(in) :: rsys
    integer(chm_int4),dimension(:),intent(in) :: notadd
    type(chm_iptr),dimension(:),intent(in) :: mcblo,mca14
    real(chm_real),intent(in) :: fact2,fact2h,eturn,eturnh,meref,merefh

    integer(chm_int4),intent(in) :: NATOM

    !local variables
    integer(chm_int4) :: i,j
    integer(chm_int4) :: offset

    !-----------------------

    spcx => null()
    spcy => null()
    spcz => null()

    spaceelold=ZERO
    spaceelnew=ZERO

    do i=1,mepsilon
        do j=1,nepsilon

            spesr1p = ZERO
            spesr2p = ZERO

            spcx => coorini(i,j,1:natom)
            spcy => coorini(i,j,natom+1:2*natom)
            spcz => coorini(i,j,2*natom+1:3*natom)

            call stupbc(ibchg,nacebc,notadd,inblo, &
                 jnb,ifrsta,natomx)

            call gteslf(spesr1p,inblo,jnb,ifrsta,natomx, &
                 spcx,spcy,spcz,ces1,ces2,sig2i,mue4, &
                 cswit,c2ofnb,c2onnb)

            spcx => coorfin(i,j,1:natom)
            spcy => coorfin(i,j,natom+1:2*natom)
            spcz => coorfin(i,j,2*natom+1:3*natom)

            call gteslf(spesr2p,inblo,jnb,ifrsta,natomx, &
                 spcx,spcy,spcz,ces1,ces2,sig2i,mue4, &
                 cswit,c2ofnb,c2onnb)

            call adeslf(ibcut,ncutbc,ecut,spesr1p,spesr2p, &
                 ibchg,nacebc,1)

            call mcacee(spaceelnew(i,j),ifrsta,natomx,esmcfx,esarr, &
                 bsarr,cg2arr,rsys,notadd,mcblo,mca14,ibchg,nacebc,ibcut, &
                 ncutbc,fact2,fact2h,eturn,eturnh,meref,merefh,cswit)

            call adeslf(ibcut,ncutbc,ecut,spesr2p,spesr1p, &
                 ibchg,nacebc,0)

            call mcacee(spaceelold(i,j),ifrsta,natomx,esmcfx,esarr, &
                 bsarr,cg2arr,rsys,notadd,mcblo,mca14,ibchg,nacebc,ibcut, &
                 ncutbc,fact2,fact2h,eturn,eturnh,meref,merefh,cswit)

        enddo
    enddo


    return
  end subroutine
!--------------------------------------------------------------------
#endif /* (ace_sp) */
!--------------------------------------------------------------------
#endif /* (samc) */
#endif /* (mc) */
end module samc
