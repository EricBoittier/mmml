module cons_rmsd

  use chm_kinds
  use chm_types
  implicit none

  logical qrmsd,qrmsddyn
  integer urmsd
  integer :: irmsdcll
  integer :: ntrmsd
  integer :: maxalloc,maxrmsd,ierr_allocate

  logical,allocatable,dimension(:) :: &
       lrmsdnrot, lrmsdntra, lrmsddiff,lrmsdhpla,lprtrmsd

  real(chm_real),allocatable, dimension(:),save ::  &
       kfrmsd,rmsd0,rmsdb,drmsd0,rmsdnorm

  integer,allocatable,dimension(:) :: nrmsd
  type(chm_iptr),allocatable,dimension(:) :: irmsd
  type(chm_ptr),allocatable,dimension(:) :: xrmsd, yrmsd, zrmsd, wrmsd

contains

  !-----------------------------------------------------------------------
  !        SETRMSD0
  !-----------------------------------------------------------------------
  subroutine setrmsd0(islct,jslct,lcomp,x,y,z,wmain, &
       xcomp,ycomp,zcomp,wcomp)

    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use psf
    use select
    use stream
    use string
    use memory
    use cnst_fcm

    implicit none

    integer,intent(inout),dimension(natom) :: islct,jslct
    real(chm_real) x(*),y(*),z(*),wmain(*)
    real(chm_real) xcomp(*),ycomp(*),zcomp(*),wcomp(*)

    !-----------------------------------------------------------------------
    ! Local variables
    CHARACTER(len=4)   WRD
    INTEGER I, IMODE
    REAL(CHM_REAL)  DROFF, KFORCE, DRMSD,NORM
    LOGICAL LMASS, LCOMP, LNOROT, LNOTRAN, LRELA, LHPLA, LPRT
    REAL(CHM_REAL)  DRBFF
    integer icount
#if KEY_RMSDDBL==1
    logical :: ierror
    integer :: inum,jnum
#endif 

!----Lack of initialisation of qrmsddyn ---------------
!----Note: qrmsddyn set to true in DYNA(dynopt) if activated and never set back
!---- to false unless dyna is called again without RMSD keyword. ABlondel jun2013.
    qrmsddyn=.false.
    if_keyword: if(indxa(comlyn,comlen,'SHOW') .gt. 0) then
       call prnrmsd(natom)
       return
    elseif (indxa(comlyn,comlen,'RESE').gt.0.or. &
         indxa(comlyn,comlen,'CLEA').gt.0) then   if_keyword
       if(qrmsd)then
          if(prnlev.gt.2) WRITE(OUTU,100)'RSMD restraint cleared '
          do i=1,ntrmsd
             icount = nrmsd(i)
             call chmdealloc('rmsd.src','setrmsd0','XRMSD(i)',icount,crlp=XRMSD(i)%a)
             call chmdealloc('rmsd.src','setrmsd0','YRMSD(i)',icount,crlp=YRMSD(i)%a)
             call chmdealloc('rmsd.src','setrmsd0','ZRMSD(i)',icount,crlp=ZRMSD(i)%a)
             call chmdealloc('rmsd.src','setrmsd0','WRMSD(i)',icount,crlp=WRMSD(i)%a)
             call chmdealloc('rmsd.src','setrmsd0','IRMSD(i)',icount,intgp=IRMSD(i)%a)
          enddo

          deallocate(xrmsd,yrmsd,zrmsd,wrmsd,stat=ierr_allocate)
          if(ierr_allocate.ne.0) &
               CALL WrnDie(-1,'<rmsd.src>setrmsd0', &
               'Failed to deallocate memory for x,y,z,wrmsd')
          deallocate(lprtrmsd,kfrmsd,rmsd0,rmsdb,drmsd0,nrmsd,rmsdnorm, &
               irmsd,stat=ierr_allocate)
          if(ierr_allocate.ne.0) &
               CALL WrnDie(-1,'<rmsd.src>setrmsd0', &
               'Failed to deallocate 2nd set of rmsd arrays')
          deallocate(lrmsdnrot,lrmsdntra,lrmsddiff,lrmsdhpla,stat=ierr_allocate)
          if(ierr_allocate.ne.0) &
               CALL WrnDie(-1,'<rmsd.src>setrmsd0', &
               'Failed to deallocate 3rd set RMSD arrays')
          ntrmsd = 0
          qrmsd    = .false.
          qcnstr   = .false.

       else

          call wrndie(0,'<rmsd>','rmsd restraint not setup')

       endif

    else   if_keyword          ! get parameter values

       if(.not.qrmsd)then      ! initialize if necessary
          if(indxa(comlyn,comlen,'MAXR') .gt. 0) then
             if(wrnlev.ge.2 .and. prnlev .gt. 2) write(outu,200)
200          format(' Warning: maxrmsd replaced by maxn, default=10')
          endif

          maxrmsd = gtrmi(comlyn,comlen,'MAXN',10)
          if(prnlev.gt.2) &
               write(outu,100) 'RMSD initialized, maxn =',maxrmsd
100       format(1x,a,i5)

          allocate(xrmsd(maxrmsd),yrmsd(maxrmsd),zrmsd(maxrmsd), &
               wrmsd(maxrmsd),lprtrmsd(maxrmsd),kfrmsd(maxrmsd), &
               rmsd0(maxrmsd), &
               rmsdb(maxrmsd),drmsd0(maxrmsd),rmsdnorm(maxrmsd),nrmsd(maxrmsd), &
               irmsd(maxrmsd),lrmsdnrot(maxrmsd),lrmsdntra(maxrmsd), &
               lrmsddiff(maxrmsd),lrmsdhpla(maxrmsd),stat=ierr_allocate)
          if(ierr_allocate.ne.0) &
               CALL WrnDie(-1,'<rmsd.src>setrmsd0', &
               'Failed to allocate RMSD arrays')
          ntrmsd = 0
          qrmsd  = .true.
          qcnstr = .true.
          call allocate_cnst(natom)
       else
          if(indxa(comlyn,comlen,'MAXR').gt.0)then
             call wrndie(-1,'<rmsd>','must reset first ')
          endif
       endif

       droff  = gtrmf(comlyn,comlen,'OFFS',zero)
       drbff  = gtrmf(comlyn,comlen,'BOFF',zero)

       if(indxa(comlyn,comlen,'NPRT').gt.0) then
          lprt = .false.
       else
          lprt = .true.
       endif
       kforce = gtrmf(comlyn,comlen,'FORC',zero)
       drmsd  = gtrmf(comlyn,comlen,'DRMS',zero)

       lmass= indxa(comlyn,comlen,'MASS') .gt. 0
       if(lmass .and. prnlev .gt.2)  &
            write(outu,100) 'mass weighting will be used '

       lnorot= indxa(comlyn,comlen,'NORO') .gt. 0
       if(lnorot.and.prnlev.gt.2)  &
            write(outu,100) 'no rotation will be used '

       lnotran= indxa(comlyn,comlen,'NOTR') .gt. 0
       if(lnotran.and.prnlev.gt.2)  &
            write(outu,100) 'no translation will be used '

       lrela= indxa(comlyn,comlen,'RELA') .gt. 0

       lhpla= indxa(comlyn,comlen,'HPLA') .gt. 0
       norm=one
       if (indxa(comlyn,comlen,'NORM') .gt. 0 ) norm=minone

       !C Took out the old Single Selection Scheme
       !C We wanted the option for different selections of the system for
       !C applying the contraint and for fitting the reference structures.
       !
       !  uncommented the old  single selection scheme

       imode=0                 !implies default = all atoms selected
#if KEY_RMSDDBL==0 /*rmsddbl*/
       call selrpn(comlyn,comlen,islct,natom,1,imode, &
            .false.,1,' ',0,resid,res,ibase,segid,nictot,nseg, &
            .true.,x,y,z,.true.,1,wmain)
       if(imode.ne.0)then
          call wrndie(-1,'<SETRMSD0(rmsd.f)>', &
               'atom selection parsing error')
       endif
#else /*       (rmsddbl)*/
       ! MSF commented out
       !
       ! New Double Selection Scheme - uses the same selection for both islct
       ! and jslct if only one selection is given in the restart file.
       ! islct = apply constraint energy
       ! jslct = used for fitting reference structures
       call selctd(comlyn,comlen,islct,jslct,x,y,z,wmain, &
            .true.,ierror)
       if(ierror) call wrndie(-5,'<rmsd>','error in selection')

       inum=0
       jnum=0
       do i = 1, natom
          if (islct(i).eq.1) inum = inum + 1
          if (jslct(i).eq.1) jnum = jnum + 1
       enddo
       if (prnlev.ge.2) then
          write(outu,*) 'rmsd constraint: ', inum,  &
               ' atoms selected for constraint application'
          write(outu,*) 'rmsd constraint: ', jnum,  &
               ' atoms selected for fitting reference structures'
       endif
#endif /*     (rmsddbl)*/
       !-------------------- Relative RMSD -----------------------------
       if(lrela)then

          if(prnlev.gt.2) then 
             if(.not.lhpla)then
                  write(outu,100)  &
                  'Relative RMSD [rmsd(main)-rmsd(comp)] '
             else
                  write(outu,100)  &
                  'Relative RMSD [rmsd(main)^2-rmsd(comp)^2] '
                  if(norm.eq.minone) write(outu,*)  &
                  'Normalised as if ref. points at distance 1:',&
                  '[(R-(R1+R2)/2).(R1-R2)/|R1-R2|]^2.'
             endif
          endif

          if(prnlev.gt.2) write(outu, '(a)') &
               ' #1 Reference coordinates set to main coordinates.'

          call setrmsd1(natom,x,y,z,islct, &
#if KEY_RMSDDBL==1
               jslct,                                          & 
#endif
               ntrmsd,amass,lmass,kforce,droff,drbff,drmsd, &
               lnorot,lnotran,lrela,lhpla,maxalloc,norm,maxrmsd,lprt)

          if(prnlev.gt.2) write(outu, '(a)') &
               '#2 Reference coordinates set to comparison coordinates.'

          call setrmsd1(natom,xcomp,ycomp,zcomp,islct, &
#if KEY_RMSDDBL==1
               jslct,                                          & 
#endif
               ntrmsd, &
               amass,lmass,kforce,droff,drbff,drmsd, &
               lnorot,lnotran,.false.,.false.,maxalloc,norm,maxrmsd,lprt)

       else

          if(lcomp)then
             if(prnlev.gt.2) write(outu, '(2a)') &
                  ' reference coordinates set to ', &
                  'comparison coordinates.'
             call setrmsd1(natom,xcomp,ycomp,zcomp,islct, &
#if KEY_RMSDDBL==1
                  jslct,                                          & 
#endif
                  ntrmsd, &
                  amass,lmass,kforce,droff,drbff,drmsd, &
                  lnorot,lnotran,lrela,lhpla,maxalloc,norm,maxrmsd,lprt)

          else
             if(prnlev.gt.2) write(outu, '(a)') &
                  ' reference coordinates set to main coordinates.'
             call setrmsd1(natom,x,y,z,islct, &
#if KEY_RMSDDBL==1
                  jslct,                                          & 
#endif
                  ntrmsd, &
                  amass,lmass,kforce,droff,drbff,drmsd, &
                  lnorot,lnotran,lrela,lhpla,maxalloc,norm,maxrmsd,lprt)

          endif
       endif
    endif    if_keyword

    return
  end subroutine setrmsd0

  !-----------------------------------------------------------------------
  !        SETRMSD1
  !-----------------------------------------------------------------------
  subroutine setrmsd1(natom,x,y,z,islct, &
#if KEY_RMSDDBL==1
       jslct, &  
#endif
       ntrmsd, &
       amass,lmass,kforce,droff,drbff,drmsd, &
       lnorot,lnotran,lrela,lhpla,maxalloc,norm,maxrmsd,lprt)

    use stream
    use memory

    integer,intent(in) :: natom,maxrmsd
    real(chm_real),intent(in) ::  drbff,kforce,droff, drmsd,norm
    logical,intent(in) :: lmass,lprt,lnorot,lnotran,lrela,lhpla

    real(chm_real), dimension(natom) :: x,y,z,amass
    integer, intent(in),dimension(natom) :: islct
#if KEY_RMSDDBL==1
    integer, intent(in),dimension(natom) :: jslct        
#endif
    integer,intent(out) :: ntrmsd
    integer,intent(inout) :: maxalloc

    !------ Local variables --------------------
    integer i,icount,j,ioff,n

    ntrmsd = ntrmsd + 1
    if(ntrmsd.gt.maxrmsd) then
       CALL WRNDIE(0,'<RMSD>','Max no. of RMSD restraits exceeded.')
    endif

    icount = 0
    do i=1,NATOM
       if(islct(i).eq.1)then
          icount=icount+1
       endif
    enddo
    if(icount.eq.0)then
       return
    endif

    call chmalloc('rmsd.src','setrmsd1','IRMSD(ntrmsd)',icount,intgp=IRMSD(ntrmsd)%a)
    call chmalloc('rmsd.src','setrmsd1','XRMSD(ntrmsd)',icount,crlp=XRMSD(ntrmsd)%a)
    call chmalloc('rmsd.src','setrmsd1','YRMSD(ntrmsd)',icount,crlp=YRMSD(ntrmsd)%a)
    call chmalloc('rmsd.src','setrmsd1','ZRMSD(ntrmsd)',icount,crlp=ZRMSD(ntrmsd)%a)
    call chmalloc('rmsd.src','setrmsd1','WRMSD(ntrmsd)',icount,crlp=WRMSD(ntrmsd)%a)
    rmsdb(ntrmsd) = drbff
    nrmsd(ntrmsd) = icount
    lprtrmsd(ntrmsd) = lprt
    kfrmsd(ntrmsd) = kforce
    rmsd0(ntrmsd)  = droff
    drmsd0(ntrmsd) = drmsd
    rmsdnorm(ntrmsd) = norm
    lrmsdnrot(ntrmsd)=LNOROT
    lrmsdntra(ntrmsd)=LNOTRAN
    lrmsddiff(ntrmsd)=LRELA
    lrmsdhpla(ntrmsd)=LHPLA

    ioff = 0
    do i=1,NATOM
       if (islct(i) == 1) then
          ioff=ioff+1
          irmsd(ntrmsd)%a(ioff) = i
          xrmsd(ntrmsd)%a(ioff) = x(i)
          yrmsd(ntrmsd)%a(ioff) = y(i)
          zrmsd(ntrmsd)%a(ioff) = z(i)
          if(lmass)then
             wrmsd(ntrmsd)%a(ioff) = amass(i)
          else
             wrmsd(ntrmsd)%a(ioff) = 1.0d0
          endif
       endif
    enddo

    if(maxalloc.lt.icount) maxalloc = icount


    return
  end subroutine setrmsd1

  ! =================================================================
  !        PRNRMSD
  ! =================================================================
  subroutine prnrmsd(natom)
    ! MSF rmsd \./
    !     &                 LRMSDDIFF)
    !    &                 LRMSDDIFF,
    !    &                 LRMSDZETA,CFZETA,INRTRMSD)
    ! MSF End

    use stream
    integer,intent(in) :: natom

    !     Local variables
    integer i, j, j1, j2, icount, n

    if(prnlev.gt.2) write(outu,*)
    if(prnlev.gt.2) write(outu,'(a,i5)')  &
         ' Total number of RMSD restraints:',ntrmsd

    do i=1,ntrmsd

       if(prnlev.gt.2) then
          write(outu,*)
          write(outu,'(a,i5)') ' RMSD restraint:', i

          write(outu,'(a,i5,a,f10.3,a,f10.3,a,f10.3)') &
               ' Number of atoms=',nrmsd(i), &
               '    force=',kfrmsd(i), &
               '    offset=',rmsd0(i), &
               '    flat-bottom=',rmsdb(i)

          if(lprtrmsd(i)) then
             write(outu,'(a)') ' Output this RMSD in Dynamics '
          else
             write(outu,'(a)') ' Not output this RMSD in Dynamics '
          endif
          if(lrmsdntra(i)) write(outu,'(a)') ' No rotation '
          if(lrmsdntra(i)) write(outu,'(a)') ' No translation '
          if(lrmsdhpla(i)) write(outu,'(a)')  &
               ' Relative [rmsd1^2-rmsd2^2] '
          if(lrmsddiff(i).and..not.lrmsdhpla(i)) write(outu,'(a)')  &
               ' Relative [rmsd1-rmsd2] '

          write(outu,*)
          write(outu,'(a)') &
               '    #   atom    xref      yref      zref     weight'

          do j=1,nrmsd(i)
             write(outu,'(2i5,1x,4f10.3)') j, irmsd(i)%a(j), &
                  xrmsd(i)%a(j), &
                  yrmsd(i)%a(j), &
                  zrmsd(i)%a(j), &
                  wrmsd(i)%a(j)
          enddo
       endif
    enddo

    return
  end subroutine prnrmsd

  ! =================================================================
  !         ECNST3
  ! =================================================================
  subroutine ecnst3(x,y,z,atompr,lprint,erms,dx,dy,dz, &
       bmass,ra,rb,rb2,dra,drb,drb2,drtemp)
    !-----------------------------------------------------------------------
    !     Multiple Root-Means-Square-Deviation (RMSD) restraints
    !     Written by Benoit Roux (April 2000) using the harmonic restraints
    !     with best fit of B.R. Brooks (1997)
    !
    use number
    use stream

    use energym
    real(chm_real),intent(in) ::  x(*),y(*),z(*)
    integer,intent(out) :: atompr(2,*)
    logical,intent(in) :: lprint
    real(chm_real),intent(inout) ::  erms
    real(chm_real),intent(inout) ::  dx(*),dy(*),dz(*)
    !
    !------- Temporary space arrays ----------
    real(chm_real),intent(out) :: bmass(*)
    real(chm_real),intent(out) ::  ra(3,*),rb(3,*)
    real(chm_real),intent(out) ::  dra(3,*),drb(3,*),drtemp(3,*)
    real(chm_real),intent(out) ::  rb2(3,*)
    real(chm_real),intent(out) ::  drb2(3,*)

    !-------- Local variables and arrays----------------
    real(chm_real) :: deva(3,3)
    real(chm_real) :: derms,tmass,rmst,norm
    real(chm_real) :: rmsv,d0,kforce
    real(chm_real) :: u(3,3),du(3,3,3,3)
    integer npair
    integer :: i, j, j1, j2, icount, n
    logical :: lnorot,lnotrn,lrela,lhpla

    real(chm_real) :: db
    integer :: idx

    !------ Local variables and arrays for relative RMSD------------
    real(chm_real) :: deva2(3,3)
    real(chm_real) :: rmst2, rmsv2, diff

    if(LPRINT.and.prnlev.gt.2)then
       write(outu,*)
       write(outu,'(a,i5)') ' Total number of RMSD restraints:',ntrmsd
    endif
    lrela = .false.

    if(QRMSDDYN) THEN ! for time-dependent RMSD
     IF(QDYNCALL)THEN
          irmsdcll=irmsdcll+1
     ENDIF
    endif

    bigloop: do i=1,ntrmsd
       kforce=kfrmsd(i)
       d0=rmsd0(i)
       db=rmsdb(i)
       norm=rmsdnorm(i)
       lnorot=lrmsdnrot(i)
       lnotrn=lrmsdntra(i)
       lrela=lrmsddiff(i)
       lhpla=lrmsdhpla(i)

       tmass=0.0

       if(i.gt.1) then
          if(lrmsddiff(i-1)) cycle bigloop  !skip second set of a relative constraint
       endif

       if(LPRINT)then
          if(prnlev.gt.2) write(outu,*)
          ! HJW---
          !           write(outu,'(a,i5,a,i5,a,f10.3,a,f10.3)')
          if(prnlev.gt.2)  &
               write(outu,'(a,i5,a,i5,a,f10.3,a,f10.3,a,f10.3)') &
               ' RMSD restraint:',i, &
               '    number of atoms=',nrmsd(i), &
               '    force=',kfrmsd(i), &
               '    offset=',rmsd0(i), &
               '    bottom_offset=',rmsdb(i)
          ! ---HJW

          if(lhpla.and.prnlev.gt.2) write(outu,'(a)')  &
               ' Relative [rmsd1^2-rmsd2^2]'
          if(lrela.and..not.lhpla.and.prnlev.gt.2) write(outu,'(a)')  &
               ' Relative [rmsd1-rmsd2]'
          if(prnlev.gt.2) then
             write(outu, *)
             write(outu,'(a)') &
                  '    #   atom    xref      yref      zref     weight'// &
                  '         x         y         z   '
          endif
       endif

       do j=1,nrmsd(i)
          idx = irmsd(i)%a(j)
          RA(1,j)=x(idx)
          RA(2,j)=y(idx)
          RA(3,j)=z(idx)
          RB(1,j) = xrmsd(i)%a(j)
          RB(2,j) = yrmsd(i)%a(j)
          RB(3,j) = zrmsd(i)%a(j)
          BMASS(j) = wrmsd(i)%a(j)
          TMASS=TMASS+bmass(j)
          ATOMPR(1,j)=idx
          ATOMPR(2,j)=idx
          if (LPRINT .and. prnlev > 2) then
             write(outu,'(2i5,1x,2(4f10.3,4x))') j, idx, &
                  rb(1,j),rb(2,j),rb(3,j) ,bmass(j), &
                  x(idx), y(idx), z(idx)
          endif
       enddo

       call ecbstf2(nrmsd(i),lnorot,lnotrn,lprint,bmass, &
            tmass,rmst,ra,rb,deva,zero)

       if(rmst.lt.rsmall) rmst=zero
       if(lhpla)then
          rmsv=rmst/tmass
       else
          rmsv=sqrt(rmst/tmass)
       endif

       if(LRELA)then
          if (LPRINT .and. prnlev > 2) write(outu,'(a)') ' second set '
          if (nrmsd(i) /= nrmsd(i+1)) &
               CALL WRNDIE(0,'<RMSD>','Relative RMSD setup incorrect')
          do j=1,nrmsd(i+1)
             idx = irmsd(i+1)%a(j)
             RB2(1,j) = xrmsd(i+1)%a(j)
             RB2(2,j) = yrmsd(i+1)%a(j)
             RB2(3,j) = zrmsd(i+1)%a(j)
             if (LPRINT .and. prnlev > 2) then
                write(outu,'(2i5,1x,2(4f10.3,4x))') j, idx, &
                     rb(1,j),rb(2,j),rb(3,j),bmass(j), &
                     x(idx), y(idx), z(idx)
             endif
          enddo

          if(norm.eq.minone)then
             call ecbstf2(nrmsd(i+1),lnorot,lnotrn,lprint,bmass, &
             tmass,rmst2,rb,rb2,deva2,zero)
             if(rmst2.lt.rsmall)then
                norm=zero
                CALL WRNDIE(0,'<RMSD>',&
                    'Ref. points at zero distance: setup incorrect')
             else
                norm=half/sqrt(rmst2/tmass)
             endif
             rmsdnorm(i)=norm
             write(outu,*)'RMSD HyperPlane Normalisation factor: ',&
                           norm
             write(outu,*)''
          endif

          rmsv=rmsv*norm
          d0=d0*norm

          call ecbstf2(nrmsd(i+1),lnorot,lnotrn,lprint,bmass, &
               tmass,rmst2,ra,rb2,deva2,zero)
          if(rmst2.lt.rsmall) rmst2=zero
          if(lhpla)then
             rmsv2=rmst2*norm/tmass
          else
             rmsv2=sqrt(rmst2/tmass)
          endif

       endif  ! if(LRELA)then

       if(.not.LRELA)then

          if(db.ge.0) then                                                 ! add biasing force if outside |db|
              if(abs(rmsv-d0).gt.db)  erms=erms+kforce*(abs(rmsv-d0)-db)**2
          else                                                             ! add biasing force if rmsv < -db
              if(rmsv.lt.-db)  erms=erms+kforce*(rmsv-d0)**2
          endif

          !------- Calculate force on all atoms
          if(rmsv.gt.rsmall)then

              if(db.ge.0) then
                  if(rmsv.gt.d0+db) then                                   ! add biasing force if outside |db|
                      derms=two*kforce*(rmsv-d0-db)/rmsv  ! [dE/d(rmsv)](rmsv-d0)/rmsv
                  else if(rmsv.lt.d0-db) then
                      derms=two*kforce*(rmsv-d0+db)/rmsv  ! [dE/d(rmsv)](rmsv-d0)/rmsv
                  else
                        derms=zero
                  endif
              else
                  if(rmsv.lt.-db) then                                   ! add biasing force if rmsv < -db
                        derms=two*kforce*(rmsv-d0)/rmsv  ! [dE/d(rmsv)](rmsv-d0)/rmsv
                  else
                         derms=zero
                  endif
              endif
          else
             derms=zero
          endif

          if(LPRINT.and.prnlev.gt.2)then
             write(outu,*) 'tmass  = ',tmass
             write(outu,*) 'rmst   = ',rmst
             write(outu,*) 'kforce = ',kforce
             write(outu,*) 'offset = ',d0
             write(outu,*) 'bottom = ',db
             write(outu,*) 'rmsv   = ',rmsv
             write(outu,*) 'erms   = ',erms
             write(outu,*) 'derms  = ',derms
          endif

          if(abs(derms).gt.rsmall) then
             call ecforc(atompr,nrmsd(i),lnorot,lnotrn,lprint, &
                  .true.,dx,dy,dz,.false., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
                  derms,bmass,tmass,ra,rb,dra,drb,drtemp,deva)!

          endif ! if(abs(derms).gt.rsmall)
       else

          diff = rmsv - rmsv2
!          write(6,*) 'HPLA0: ',erms
          if(db.ge.0) then                                               ! add biasing force if outside |db|
                  if(abs(diff-d0).gt.db) then
                     erms=erms+kforce*(abs(diff-d0)-db)**2 ! Calculate energy (AB: same for HPLA).
                  else
                     erms=zero
                  endif
          else                                                           ! add biasing force if inside |db|
                  if(diff.lt.-db) then
                      erms=erms+kforce*(diff-d0)**2 ! Calculate energy (AB: same for HPLA).
                  else
                     erms=zero
                  endif
           endif
!          write(6,*) 'HPLA: ',erms,rmsv,rmsv2!,diff,d0,db
          !----- Calculate force on all atoms (AB: first dist; if HPLA: no singularity).
          if(db.ge.0) then                                               ! add biasing force if outside |db|
             if(diff.gt.d0+db) then
                 derms=two*kforce*(diff-d0-db) ! [dE/d(rmsv)](rmsv-d0)[if!HPLA: /rmsv]
             else if(diff.lt.d0-db) then
                 derms=two*kforce*(diff-d0+db) ! [dE/d(rmsv)](rmsv-d0)[if!HPLA: /rmsv]
             else
                 derms=zero
             endif
          else                                                           ! add biasing force if inside |db|
             if(diff.lt.-db) then
                 derms=two*kforce*(diff-d0) ! [dE/d(rmsv)](rmsv-d0)[if!HPLA: /rmsv]
             else
                 derms=zero
             endif
          endif
          if(lhpla) then
             derms=(derms+derms)*norm
          else
             if(rmsv.gt.rsmall)then
                derms=derms/rmsv
             else
                derms=zero
             endif
          endif
!          write(6,*)'Derms1: ',derms
          if(LPRINT.and.prnlev.gt.2)then
             write(outu,*) 'tmass  = ',tmass
             write(outu,*) 'rmst1  = ',rmst
             write(outu,*) 'rmst2  = ',rmst2
             write(outu,*) 'kforce = ',kforce
             write(outu,*) 'offset = ',d0
             if(lhpla)then
                write(outu,*) 'rmsv^2   = ',rmsv
                write(outu,*) 'rmsv2^2  = ',rmsv2
             else
                write(outu,*) 'rmsv   = ',rmsv
                write(outu,*) 'rmsv2  = ',rmsv2
             endif
             write(outu,*) 'erms   = ',erms
             write(outu,*) 'derms1 = ',derms
          endif

          if(abs(derms).gt.rsmall) then
             call ecforc(atompr,nrmsd(i),lnorot,lnotrn,lprint, &
                  .true.,dx,dy,dz,.false., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
                  derms,bmass,tmass,ra,rb,dra,drb,drtemp,deva)

          endif ! if(abs(derms).gt.rsmall)
!-force for second distance.
          if(db.ge.0) then                                               ! add biasing force if outside |db|
             if(diff.gt.d0+db) then
                 derms=-two*kforce*(diff-d0-db)
             else if(diff.lt.d0-db) then
                 derms=-two*kforce*(diff-d0+db)
             else
                 derms=zero
             endif
          else                                                           ! add biasing force if diff < -db
             if(diff.lt.-db) then
                derms=-two*kforce*(diff-d0)
             else
                derms=zero
             endif
          endif
          if(lhpla)then
             derms=(derms+derms)*norm
          else
             if(rmsv2.gt.rsmall)then
                derms=derms/rmsv2
             else
                derms=zero
             endif ! if(rmsv2.gt.rsmall)
          endif
!          write(6,*)'Derms2: ',derms

          if(LPRINT)then
             if(prnlev.gt.2) write(outu,*) 'derms2 = ',derms !(AB: derms1->derms2)
          endif

          if(abs(derms).gt.rsmall) then
             call ecforc(atompr,nrmsd(i+1),lnorot,lnotrn,lprint, &
                  .true.,dx,dy,dz,.false., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
                  derms,bmass,tmass,ra,rb2,dra,drb2,drtemp,deva2)
          endif ! if(abs(derms).gt.rsmall)
       endif

       !
       ! For time-dependent forcing with RMSD restraints during MD
       !
       if(qrmsddyn)then
          if(urmsd.gt.0 .and. lprtrmsd(i))then
             if(lrela)then
               IF(QDYNCALL)THEN
                if(prnlev.gt.2)  &
                     write(urmsd,'(i10,f15.5,f15.5,f15.5,f15.5,f15.5)') &
                     irmsdcll,rmsv,rmsv2,diff,d0
               ENDIF
             else
              IF(QDYNCALL)THEN
                if(prnlev.gt.2) write(urmsd,'(i10,f15.5,f15.5)') &
                     irmsdcll,rmsv,d0
              ENDIF
             endif
          endif
          rmsd0(i)=rmsd0(i)+drmsd0(i)
       endif
    enddo bigloop

    return
  end subroutine ecnst3


  ! =================================================================
  !                ECBSTF2
  ! =================================================================
  SUBROUTINE ECBSTF2(NPAIR,LNOROT,LNOTRN,LPRINTP,BMASS, &
       TMASS,RMST,RA,RB,DEVA,EVWID)
    !
    use number
    use stream
    use corsubs,only:frotu
    !
    INTEGER NPAIR
    LOGICAL LNOROT,LNOTRN,LPRINTP
    !
    ! Temporary space arrays
    REAL(CHM_REAL) BMASS(NPAIR)
    REAL(CHM_REAL) TMASS,RMST
    REAL(CHM_REAL) RA(3,NPAIR),RB(3,NPAIR)
    REAL(CHM_REAL) DEVA(3,3)
    REAL(CHM_REAL) EVWID
    !
    ! Local variables and arrays
    REAL(CHM_REAL) EVA(3),U(3,3),ATOT,BTOT
    REAL(CHM_REAL) R(3,3)
    REAL(CHM_REAL) CMA(3),CMB(3),CMC(3)
    REAL(CHM_REAL) RMSV,ATMP,BTMP
    INTEGER K,I,J
    LOGICAL QEVW,LPRINT
    !
    IF(TMASS.LT.RSMALL) RETURN ! don't process nonpositive total weight.
    !
    LPRINT=LPRINTP
    IF(PRNLEV.LE.2)LPRINT=.FALSE.
    !
    IF(LPRINT.and.prnlev.gt.2) write(OUTU,56)  &
         ' The BMASS value:',BMASS
56  format(A/,(10X,3F12.5))

    ! Compute centers of mass (or weight)
    DO I=1,3
       CMA(I)=0.0
       CMB(I)=0.0
       CMC(I)=0.0
    ENDDO
    !
    IF(.NOT.LNOTRN) THEN
       DO K=1,NPAIR
          DO I=1,3
             CMA(I)=CMA(I)+RA(I,K)*BMASS(K)
             CMB(I)=CMB(I)+RB(I,K)*BMASS(K)
          ENDDO
       ENDDO

       DO I=1,3
          CMA(I)=CMA(I)/TMASS
          CMB(I)=CMB(I)/TMASS
          CMC(I)=CMA(I)-CMB(I)
       ENDDO
    ENDIF

    DO K=1,NPAIR
       DO I=1,3
          RA(I,K)=RA(I,K)-CMA(I)
          RB(I,K)=RB(I,K)-CMB(I)
       ENDDO
    ENDDO
    !
    IF(LPRINT.and.prnlev.gt.2) write(OUTU,56) ' The RA value:',RA
    IF(LPRINT.and.prnlev.gt.2) write(OUTU,56) ' The RB value:',RB
    !
    IF (LPRINT.and.prnlev.gt.2) THEN
       WRITE(OUTU,44) CMB
       WRITE(OUTU,45) CMA
       WRITE(OUTU,46) CMC
    ENDIF
44  FORMAT(' CENTER OF ATOMS BEFORE TRANSLATION',3F12.5)
45  FORMAT(' CENTER OF REFERENCE COORDINATE SET',3F12.5)
46  FORMAT(' NET TRANSLATION OF ROTATED ATOMS  ',3F12.5)
    !
    IF (LNOROT) THEN
       !
       RMST=0.0
       DO K=1,NPAIR
          DO I=1,3
             RMST=RMST+(RA(I,K)-RB(I,K))**2*BMASS(K)
          ENDDO
       ENDDO
       !
    ELSE
       !
       !       compute rotation matrix from lagrangian
       !
       ATOT=0.0
       BTOT=0.0
       DO I=1,3
          DO J=1,3
             R(I,J)=0.0
          ENDDO
       ENDDO
       DO K=1,NPAIR
          DO I=1,3
             ATOT=ATOT + RA(I,K)*RA(I,K)*BMASS(K)
             BTOT=BTOT + RB(I,K)*RB(I,K)*BMASS(K)
             DO J=1,3
                R(I,J)=R(I,J) + RA(I,K)*RB(J,K)*BMASS(K)
             ENDDO
          ENDDO
       ENDDO

       CALL FROTU(R,EVA,DEVA,U,EVWID,QEVW,LPRINT)

       IF(LPRINT.and.prnlev.gt.2)  &
            write(6,56) ' The ATOT,BTOT and EV(*) values:', &
            ATOT,BTOT,EVA(1),EVA(2),EVA(3)

       RMST=ATOT+BTOT-2.0*(EVA(1)+EVA(2)+EVA(3))

       IF(QEVW) THEN
          ! Note: Do not use conventional (higher accuracy) method in case
          ! the width parameter (EVWID) is employed.
          RMST=ATOT+BTOT-2.0*(EVA(1)+EVA(2)+EVA(3))
       ELSE
          RMST=0.0
          DO K=1,NPAIR
             ATMP=0.0
             DO I=1,3
                BTMP=RA(I,K)
                DO J=1,3
                   BTMP=BTMP-U(I,J)*RB(J,K)
                ENDDO
                ATMP=ATMP+BTMP**2
             ENDDO
             RMST=RMST+ATMP*BMASS(K)
          ENDDO
       ENDIF
       !
    ENDIF

    RMSV=SQRT(RMST/TMASS)

    RETURN
  END SUBROUTINE ECBSTF2

end module cons_rmsd

