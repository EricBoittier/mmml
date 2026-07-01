module tmd
  use chm_types
  use dimens_fcm

#if KEY_TMD==1 /*tmd_fcm*/

  implicit none
  ! TMD.FCM
  !
  !   IXTAR  -
  !   IYTAR  -
  !   IZTAR  -
  !   ITRCM  -
  !   ISTMD  -
  !
  ! MSF added IXTAR2,IYTAR2,IZTAR2,JSTMD   MSF tmd
  !

  !---mfc--- Needs allocation routines

  real(chm_real),allocatable,dimension(:) :: ixtar
  real(chm_real),allocatable,dimension(:) :: iytar
  real(chm_real),allocatable,dimension(:) :: iztar
  real(chm_real),allocatable,dimension(:) :: itrcm
  integer,allocatable,dimension(:) :: istmd
  integer,allocatable,dimension(:) :: jstmd
  real(chm_real),allocatable,dimension(:) :: ixtar2
  real(chm_real),allocatable,dimension(:) :: iytar2
  real(chm_real),allocatable,dimension(:) :: iztar2
  real(chm_real),allocatable,dimension(:) :: ix1tmd
  real(chm_real),allocatable,dimension(:) :: iy1tmd
  real(chm_real),allocatable,dimension(:) :: iz1tmd
  real(chm_real),allocatable,dimension(:) :: ix2tmd
  real(chm_real),allocatable,dimension(:) :: iy2tmd
  real(chm_real),allocatable,dimension(:) :: iz2tmd
  real(chm_real),allocatable,dimension(:) :: itmdx
  real(chm_real),allocatable,dimension(:) :: itmdy
  real(chm_real),allocatable,dimension(:) :: itmdz
  real(chm_real),allocatable,dimension(:) :: itmdxx,itmdyy,itmdzz
  real(chm_real),allocatable,dimension(:) :: ixgtmd,iygtmd,izgtmd
  !
  real(chm_real) TMDFRMS,DRHO,TMDRHOF
  integer :: inrt = -999
  INTEGER TMDNTAR,TMDNTAR2,TMDFRST,TMDLAST, &
       TMDFRST2,TMDLAST2
  LOGICAL QTMD
  logical qzeta
  real(chm_real) czeta,ztol,zetatg
  integer zmxitr

  !%% AvdV addition
  real(chm_real) tmdfmax2,tmdbmax2
  integer itmd,ftmd,isump
  logical tmdwr,tmdcalc
  !%% AvdV end addition
  !
#endif /* (tmd_fcm)*/
  !
contains

#if KEY_TMD==1 /*if_tmd*/
  subroutine tmd_iniall()
    qtmd=.false.
    return
  end subroutine tmd_iniall


  SUBROUTINE TMDINIT(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     This routine interprets commands initializing the
    !     targeted molecular dynamics (TMD) method.
    !
    !     Author: Jianpeng Ma (8/96)
    !     Modified by Arjan van der Vaart
    !
    !-----------------------------------------------------------------------
    !
    use number
    use psf
    use coord
    !
    use string
    use stream
    use memory
    use select
    character(len=*) COMLYN
    INTEGER COMLEN
    !
    INTEGER ETMD
    !      DATA INRT /1000/
    !                               
    qtmd=.true.
    !
    ! Allocate space for the TMD target structure.
    call chmalloc('tmd.src','TMDINIT','ixtar',NATOM,crl=ixtar)
    call chmalloc('tmd.src','TMDINIT','iytar',NATOM,crl=iytar)
    call chmalloc('tmd.src','TMDINIT','iztar',NATOM,crl=iztar)
    call chmalloc('tmd.src','TMDINIT','itrcm',NATOM,crl=itrcm)
    call chmalloc('tmd.src','TMDINIT','istmd',NATOM,intg=istmd)
    call chmalloc('tmd.src','TMDINIT','jstmd',NATOM,intg=jstmd)
    !
    INRT=GTRMI(COMLYN,COMLEN,'INRT',INRT)
    DRHO=GTRMF(COMLYN,COMLEN,'DINC',ZERO)
    QZETA= INDXA(COMLYN,COMLEN,'ZETA')  >  0
    CZETA= GTRMF(COMLYN,COMLEN,'CZET',ONE)
    ZTOL = GTRMF(COMLYN,COMLEN,'ZTOL',RSMALL)
    ZMXITR=GTRMI(COMLYN,COMLEN,'ZMIT',1000)
    tmdfrms=gtrmf(comlyn,comlen,'FRMS',ZERO)
    tmdfrms=abs(tmdfrms)
    if(tmdfrms < tenm6)tmdfrms=tenm6
    !%%AvdV addition
    itmd=gtrmi(comlyn,comlen,'ITMD',-1)
    ftmd=gtrmi(comlyn,comlen,'FTMD',-1)
    isump=gtrmi(comlyn,comlen,'SUMP',0)
    tmdfmax2=gtrmf(comlyn,comlen,'MAXF',-ONE)
    if(tmdfmax2 > 0.0D0)tmdfmax2=tmdfmax2*tmdfmax2
    tmdbmax2=gtrmf(comlyn,comlen,'MAXB',-ONE)
    if(tmdbmax2 > 0.0D0)tmdbmax2=tmdbmax2*tmdbmax2
    etmd=gtrmi(comlyn,comlen,'ENER',0)
    tmdcalc=.false.
    if((itmd > 0).and.(etmd /= 0))tmdcalc=.true.
    !%%AvdV end addition
    !
    if(qzeta)then
       call chmalloc('tmd.src','TMDINIT','ixtar2',NATOM,crl=ixtar2)
       call chmalloc('tmd.src','TMDINIT','iytar2',NATOM,crl=iytar2)
       call chmalloc('tmd.src','TMDINIT','iztar2',NATOM,crl=iztar2)
       call chmalloc('tmd.src','TMDINIT','ix1tmd',NATOM,crl=ix1tmd)
       call chmalloc('tmd.src','TMDINIT','iy1tmd',NATOM,crl=iy1tmd)
       call chmalloc('tmd.src','TMDINIT','iz1tmd',NATOM,crl=iz1tmd)
       call chmalloc('tmd.src','TMDINIT','ix2tmd',NATOM,crl=ix2tmd)
       call chmalloc('tmd.src','TMDINIT','iy2tmd',NATOM,crl=iy2tmd)
       call chmalloc('tmd.src','TMDINIT','iz2tmd',NATOM,crl=iz2tmd)
       call chmalloc('tmd.src','TMDINIT','ixgtmd',NATOM,crl=ixgtmd)
       call chmalloc('tmd.src','TMDINIT','iygtmd',NATOM,crl=iygtmd)
       call chmalloc('tmd.src','TMDINIT','izgtmd',NATOM,crl=izgtmd)
       itmd=-1
       tmdfmax2=-1.0D0
       !%%AvdV addition
    elseif(itmd > 0)then
       call chmalloc('tmd.src','TMDINIT','itmdx',natom,crl=itmdx)
       call chmalloc('tmd.src','TMDINIT','itmdy',natom,crl=itmdy)
       call chmalloc('tmd.src','TMDINIT','itmdz',natom,crl=itmdz)
       if(tmdcalc)then
          if(prnlev > 2) &
               write(itmd,'("# istep -- Max |p_i| -- Sum |p_i| --", &
               " Sum |x_i| -- Sum |p_i| cos(p_i,F_i) -- rho", &
               " -- Delta ENER")')
          call chmalloc('tmd.src','TMDINIT','itmdxx',natom,crl=itmdxx)
          call chmalloc('tmd.src','TMDINIT','itmdyy',natom,crl=itmdyy)
          call chmalloc('tmd.src','TMDINIT','itmdzz',natom,crl=itmdzz)
       else
          if(prnlev > 2) &
               write(itmd,'("# istep -- Max |p_i| -- Sum |p_i| --", &
               " Sum |x_i| -- Sum |p_i| cos(p_i,F_i) -- rho")')
       endif
       !temporarily until parallel problem is solved!
       !%%AvdV end addition
    endif
    !
    ! To handle the atom selection for the least-square fitting for
    ! stop the rotation. Note here the selection is done in terms of
    ! the main set of coordinates, rather than the target coordinates
    ! although they are the same.
    !
    call seltmd(comlyn,comlen,istmd,jstmd) 
    !
    RETURN
  END SUBROUTINE TMDINIT

  SUBROUTINE SELTMD(COMLYN,COMLEN,ISLCT,JSLCT)
    use number
    use psf
    use coord
    use select
    character(len=*) COMLYN
    INTEGER COMLEN
    INTEGER ISLCT(*),JSLCT(*)
    INTEGER IMODE,K
    !
    LOGICAL IERROR
    !

    !     double selection:
    !     1. Selection of Fitting area, using in Subroutine STOPRTTMD
    !     This is consistent with purpose of the original single selection
    !     2. Selection of TMD area, used in Subroutine SHIFTZ
    CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN, &
         .TRUE.,IERROR)
    IF(IERROR) CALL WRNDIE(-5,'<SELTMD>','ERROR IN SELECTION')
    !     
    TMDNTAR=0
    TMDFRST=NATOM+1
    DO K=1,NATOM
       IF(ISLCT(K) == 1) THEN
          TMDNTAR=TMDNTAR+1
          TMDFRST=MIN(K,TMDFRST)
          TMDLAST=K
       ENDIF
    ENDDO
    IF(TMDNTAR == 0) THEN
       CALL WRNDIE(0,'<TMDINIT>','ZERO ATOMS SELECTED FOR FIT')
    ENDIF
    TMDNTAR2=0
    TMDFRST2=NATOM+1
    DO K=1,NATOM
       IF(JSLCT(K) == 1) THEN
          TMDNTAR2=TMDNTAR2+1
          TMDFRST2=MIN(K,TMDFRST2)
          TMDLAST2=K
       ENDIF
    ENDDO
    IF(TMDNTAR2 == 0) THEN
       CALL WRNDIE(0,'<TMDINIT>','ZERO ATOMS SELECTED FOR TMD')
    ENDIF

    RETURN
  END SUBROUTINE SELTMD

  subroutine shift_tmd(x_new,y_new,z_new,x_old,y_old,z_old,xf,yf,zf, &
       amass,natom,tol_corr, &
       jslct, &
       bnbnd,bimag, &
       teterm,teprop,tepress,tmde, &
       tmpx,tmpy,tmpz,istep,tmdsh1)
    !%% AvdV added bnbnd--tmdsh1
    !
    !     This routine implements the Targeted Molecular Dynamics
    !     Simulation (TMD) algorithm:
    !     J. Schlitter et al. Molecular Simulation 10,291(1993)
    !
    !     Author: Jianpeng Ma, May 1996
    !
    !%%   MD-TMD addition by AvdV, Aug. 2004
    !
    use number
    use parallel
    !%% AvdV addition
    use deriv
    use energym
    use comand
    use coord
    use contrl
    use eutil
    !%% AvdV end addition
    integer i,natom
    integer jslct(*)
    real(chm_real) x_old(*),y_old(*),z_old(*)
    real(chm_real) x_new(*),y_new(*),z_new(*)
    real(chm_real) xf(*),yf(*),zf(*)
    real(chm_real) delta_x,delta_y,delta_z
    real(chm_real) dis2,phi,aa,bb,gamma1,gamma2,gamma
    real(chm_real) mass_tol,tol_corr
    real(chm_real) amass(*)
    integer atlast,atfrst,istart,iend,j
    !%% AvdV addition
    real(chm_real) tmpx(*),tmpy(*),tmpz(*),teterm(*),teprop(*), &
         tepress(*),tmde
    integer istep   !!,bnbnd(*),bimag(*)
    type(nonbondDataStructure) bnbnd
    type(imageDataStructure) bimag

    logical tmdsh1
    real(chm_real) gfmax2,gbmax2,sumpx,aa1
    !%% AvdV end addition

    !
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
    ATFRST=1+IPARPT(MYNOD)
    ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
    ATFRST=1
    ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /* (paramain)*/
    ATFRST=1
    ATLAST=NATOM
#endif /* (paramain)*/
    !
    !     Set the limits for each process
    istart=tmdfrst2
    if(tmdfrst2 <= atfrst)istart=atfrst
    if(tmdfrst2 > atlast)istart=0
    iend=tmdlast2
    if(tmdlast2 > atlast)iend=atlast
    if(tmdlast2 < atfrst)iend=0
    !
    mass_tol=zero
    aa=zero
    bb=zero
    dis2=zero
    tol_corr=zero
    !%% AvdV addition
    gfmax2=zero
    sumpx=zero

    IF (tmdwr.and.tmdsh1) THEN                       
       do i=1,natom
          tmpx(i)=x(i)
          tmpy(i)=y(i)
          tmpz(i)=z(i)
       enddo
       do i=istart,iend
          x(i)=x_new(i)
          y(i)=y_new(i)
          z(i)=z_new(i)
       enddo
       IF(tmdcalc) THEN                       
          !     .     perform one extra energy call before applying TMD 
          !     .     to calculate the potential energy change upon TMD
          do i=1,lenent
             teterm(i)=eterm(i)
          enddo
          do i=1,lenenp
             teprop(i)=eprop(i)
          enddo
          do i=1,lenenv
             tepress(i)=epress(i)
          enddo
          !C
          !C   PARALLEL FIX ??? these X,Y,Z are not for the whole molecule! 
          !      you must first broadcast them...
          !C
          !            do i=istart,iend
          do i=1,natom
             tmpx(i)=x(i)
             tmpy(i)=y(i)
             tmpz(i)=z(i)
          enddo
          do i=istart,iend
             x(i)=x_new(i)
             y(i)=y_new(i)
             z(i)=z_new(i)
          enddo
          !
#if KEY_PARALLEL==1
          CALL VDGBR(x,y,z,1)
#endif 
          !     We need to get rid of extra printouts... SEEMS TO NOT BE NEEDED ????
          !            CALL UPDATE('',0,x,y,z,wmain,
          !     $           .true.,.true.,.true.,.true.,.true.,0,0,0,0,0,0,0)
          CALL GETE(x,y,z,x,y,z,0)
          !
          do i=1,natom
             x(i)=tmpx(i)
             y(i)=tmpy(i)
             z(i)=tmpz(i)
          enddo
          do i=istart,iend
             tmpx(i)=x_new(i)
             tmpy(i)=y_new(i)
             tmpz(i)=z_new(i)
          enddo
          !            if(mynod == 0)
          !     $     CALL printe(6,eprop,eterm,'xxxx','ENR',.false.,1,1,1,.true.)
          !     .           we are only interested in the change in potential energy 
          tmde=eprop(3)
          !            write(*,*)'SHIFT>me,tmde=',mynod,tmde
          !            write(*,111)'SHIFT>me,x_new=',mynod,(x_new(i),i=1,12)
          !            write(*,111)'SHIFT>me,tmpx= ',mynod,(tmpx(i),i=1,12)
111       format(a15,i2,12f8.5)
       endif
    endif
    !
    !%% AvdV end addition
    !
    IF((IEND /= 0).AND.(ISTART.NE.0)) THEN
       !
       ! Mass weighting the coordinates to avoid the translation of COM.
       do i=istart,iend
          if(jslct(i) == 1)then
             mass_tol=mass_tol+amass(i)
          endif
       enddo
       !
    ENDIF
    !
#if KEY_PARALLEL==1
    call gcomb(mass_tol,1)                   
#endif
    !
    IF((IEND /= 0).AND.(ISTART.NE.0)) THEN
       !
       do i=istart,iend
          if(jslct(i) == 1)then
             amass(i)=sqrt(amass(i)/mass_tol)
          endif
       enddo
       !
       do i=istart,iend
          if(jslct(i) == 1)then
             x_old(i)=x_old(i)*amass(i)
             y_old(i)=y_old(i)*amass(i)
             z_old(i)=z_old(i)*amass(i)
             x_new(i)=x_new(i)*amass(i)
             y_new(i)=y_new(i)*amass(i)
             z_new(i)=z_new(i)*amass(i)
             xf(i)=xf(i)*amass(i)
             yf(i)=yf(i)*amass(i)
             zf(i)=zf(i)*amass(i)
          endif
       enddo
       !
       ! Construct the quadratic equation of Schlitter's method.
       !%% AvdV addition
       if(tmdfmax2 >= zero)then
          if(isump == 0)then
             !     .        maximum displacement per atom
             do i=istart,iend
                if(jslct(i) == 1)then
                   aa1=(x_old(i)-xf(i))**2+ &
                        (y_old(i)-yf(i))**2+ &
                        (z_old(i)-zf(i))**2
                   aa=aa+aa1
                   gfmax2=max(gfmax2,aa1/(amass(i)*amass(i)))
                   bb=bb+(x_old(i)-xf(i))*(x_new(i)-xf(i))+ &
                        (y_old(i)-yf(i))*(y_new(i)-yf(i))+ &
                        (z_old(i)-zf(i))*(z_new(i)-zf(i))
                endif
             enddo
          else
             !     .        maximum total displacement
             do i=istart,iend
                if(jslct(i) == 1)then
                   aa1=(x_old(i)-xf(i))**2+ &
                        (y_old(i)-yf(i))**2+ &
                        (z_old(i)-zf(i))**2
                   aa=aa+aa1
                   gfmax2=gfmax2+(sqrt(aa1)/amass(i))
                   bb=bb+(x_old(i)-xf(i))*(x_new(i)-xf(i))+ &
                        (y_old(i)-yf(i))*(y_new(i)-yf(i))+ &
                        (z_old(i)-zf(i))*(z_new(i)-zf(i))
                endif
             enddo
          endif
          if(tmdbmax2 >= zero)then
             !     .        calculate Sum p . F 
             !     .        note that in CHARMM, dx=grad(E)=-F
             do i=istart,iend
                if(jslct(i) == 1)then
                   sumpx=sumpx &
                        +(((x_old(i)-xf(i))*dx(i) &
                        +(y_old(i)-yf(i))*dy(i) &
                        +(z_old(i)-zf(i))*dz(i)) &
                        /(amass(i)*sqrt(dx(i)*dx(i)+dy(i)*dy(i) &
                        +dz(i)*dz(i))))
                endif
             enddo
          endif
       else
          !%% AvdV end addition
          do i=istart,iend
             if(jslct(i) == 1)then
                aa=aa+(x_old(i)-xf(i))**2+ &
                     (y_old(i)-yf(i))**2+ &
                     (z_old(i)-zf(i))**2
                bb=bb+(x_old(i)-xf(i))*(x_new(i)-xf(i))+ &
                     (y_old(i)-yf(i))*(y_new(i)-yf(i))+ &
                     (z_old(i)-zf(i))*(z_new(i)-zf(i))
             endif
          enddo
          !%% AvdV addition
       endif
       !%% AvdV end addition
       bb=two*bb
       !
       do i=istart,iend
          if(jslct(i) == 1)then
             dis2=dis2+(x_new(i)-xf(i))**2+ &
                  (y_new(i)-yf(i))**2+ &
                  (z_new(i)-zf(i))**2
          endif
       enddo
       !
    ENDIF
    !
#if KEY_PARALLEL==1
    !     Sum for parallel:
    call gcomb(bb,1)
    call gcomb(aa,1)
    call gcomb(dis2,1)
    !%% AvdV addition
    if(tmdfmax2 >= 0.0D0)then
       if(isump == 0)then
          call gcombmax(gfmax2)
       else
          call gcomb(gfmax2,1)
       endif
       call gcomb(sumpx,1)
    endif
    !%% AvdV end addition
#endif 
    !
    !%% AvdV addition
    if(tmdfmax2 >= zero)then
       if(isump /= 0)then
          gfmax2=gfmax2*gfmax2
       endif
       gbmax2=tmdbmax2/gfmax2
       gfmax2=tmdfmax2/gfmax2
       if(tmdbmax2 < zero)then
          gamma=gfmax2
       else
          !     .     Here: sumpx=-sumpx
          if((sumpx*bb) < zero)then
             gamma=gbmax2
          else
             gamma=gfmax2
          endif
       endif
       gamma=sign(sqrt(gamma),-bb)
       phi=-aa*gamma*gamma-bb*gamma
       tmdrhof=sqrt(dis2-phi)
    else
       !%% AvdV end addition
100    phi=dis2-tmdrhof**2
    !
    ! Solving the quadratic equation.
    !%% AvdV slight change for efficiency
       gamma=sqrt(bb**2-four*aa*phi)
       gamma1=(-bb+gamma)/(two*aa)
       gamma2=(-bb-gamma)/(two*aa)
       !%% AvdV end change
       if(abs(gamma1)  <  abs(gamma2)) then
          gamma=gamma1
       else
          gamma=gamma2
       endif
       !%% AvdV addition
    endif
    !%% AvdV end addition
    !
    ! Correct the coordinates. Be sure to remove the mass weighting
    ! on the passed-in coordinate array. Recover the mass array.
    !
    !
    IF((IEND /= 0).AND.(ISTART.NE.0)) THEN
       do i=istart,iend
          if(jslct(i) == 1)then
             xf(i)=xf(i)/amass(i)
             yf(i)=yf(i)/amass(i)
             zf(i)=zf(i)/amass(i)
             x_old(i)=x_old(i)/amass(i)
             y_old(i)=y_old(i)/amass(i)
             z_old(i)=z_old(i)/amass(i)
             x_new(i)=x_new(i)/amass(i)
             y_new(i)=y_new(i)/amass(i)
             z_new(i)=z_new(i)/amass(i)
             amass(i)=mass_tol*amass(i)**2
          endif
       enddo
       !
       do i=istart,iend
          if(jslct(i) == 1)then
             delta_x=gamma*(x_old(i)-xf(i))
             delta_y=gamma*(y_old(i)-yf(i))
             delta_z=gamma*(z_old(i)-zf(i))
             x_new(i)=x_new(i)+delta_x
             y_new(i)=y_new(i)+delta_y
             z_new(i)=z_new(i)+delta_z
             tol_corr=tol_corr+delta_x**2+delta_y**2+delta_z**2
          endif
       enddo
    ENDIF
    !
#if KEY_PARALLEL==1
    call gcomb(tol_corr,1)                               
#endif
    tol_corr=sqrt(tol_corr)
    !
    return
  end subroutine shift_tmd

  subroutine stoprt_tmd(x1,y1,z1,natom1,amass1,x2,y2,z2, &
       natom2,amass2, &
       atomin,wmain,islct)
    !
    !     This routine is for stopping the rotation during the
    !     TMD method. It makes use of the existing CHARMM source
    !     codes to achieve the goal. The fundamental algorithm
    !     is based on Kabsch's method: Acta Cryst. A32, 922 (1976).
    !
    !     x1,y1,z1 are the fixed structure. (natom)
    !     x2,y2,z2 are the moving structure. (ntar)
    !     Note: We let the target following the structure, so the
    !           numerical integration is not discretized.
    !
    !     The ROTLSQ routine called by this routine is taken from
    !     /source/manip/rotlsq.src. It further calls a few routines.
    !
    !     Author: Jianpeng Ma, June 1996
    !
    use stream
    use corsubs

    !
    integer natom1,natom2,npr,i,natom
    integer atomin(2,*),islct(*)
    real(chm_real) x1(*),y1(*),z1(*),x2(*),y2(*),z2(*)
    real(chm_real) amass1(*),amass2(*),wmain(*)
    logical lmass,lweig,lnorot,lprint
    !
    lmass=.true.  ! using mass weighing always
    lweig=.false.
    lnorot=.false.
    if(prnlev >= 5)then
       lprint=.true.
    else
       lprint=.false.
    endif

    npr=0
    do i=tmdfrst,tmdlast
       if(islct(i)  ==  1) then
          npr=npr+1
          atomin(1,npr)=i
          atomin(2,npr)=i
       endif
    enddo

    !
    call rotlsq(x1,y1,z1,natom1,x2,y2,z2,natom2,atomin,npr, &
         lmass,amass1,amass2,lweig,wmain,lnorot,lprint)
    !
    return
  end subroutine stoprt_tmd

  subroutine inidis(xtar,ytar,ztar,natom,amass,islct)
    !
    !     This routine computes the initial mass weighted rms distance
    !     for the TMD. The mean mass <m> in the original literature
    !     is replaced by the total mass.
    !
    !     Author: Jianpeng Ma, June 1996
    !
    use coord
    use coordc
    integer i,natom
    integer islct(*)
    real(chm_real) xtar(*),ytar(*),ztar(*)
    real(chm_real) amass(*)
    real(chm_real) mass_tol
    !
    ! Mass weighting the coordinates.
    mass_tol=0.
    do i=tmdfrst, tmdlast
       if(islct(i) == 1)then
          mass_tol=mass_tol+amass(i)
       endif
    enddo
    do i=tmdfrst,tmdlast
       if(islct(i) == 1)then
          amass(i)=amass(i)/mass_tol
       endif
    enddo
    !
    tmdrhof=0.0
    do i=tmdfrst,tmdlast
       if(islct(i) == 1)then
          tmdrhof=tmdrhof+amass(i)*((x(i)-xtar(i))**2+ &
               (y(i)-ytar(i))**2+ &
               (z(i)-ztar(i))**2)
          amass(i)=mass_tol*amass(i)
       endif
    enddo
    tmdrhof=sqrt(tmdrhof)
    !
    return
  END subroutine inidis

  !=====================================================================
  subroutine inizet(xtar,ytar,ztar,XTAR2,YTAR2,ZTAR2, &
       amass,islct,JSLCT)
    !=====================================================================
    !
    !
    !  tmdrhof  is the current zeta function for the current coordinates
    !  zetatg is the 'final' zeta, when the coordinates are at the 2nd target
    !         So in the evaluation, rmsd1=rmsd0 and rmsd2=0.0
    !
    !     Author: Mark Formaneck, May 2004
    !
    use coord
    use coordc
    use stream
    use number

    integer i
    integer islct(*),jslct(*),icount,jcount
    real(chm_real)  xtar(*), ytar(*), ztar(*)
    real(chm_real)  xtar2(*),ytar2(*),ztar2(*)
    real(chm_real)  amass(*)

    real(chm_real)  rmsd0,rmsd1,rmsd2
    real(chm_real)  tmass
    !
    ! rmsd0 - target1 to target2
    ! rmsd1 - current structure to target1
    ! rmsd2 - current structure to target2
    !
    rmsd0=ZERO
    rmsd1=ZERO
    rmsd2=ZERO
    tmass=ZERO
    do i=tmdfrst2, tmdlast2
       if(jslct(i)  ==  1) then
          tmass = tmass + amass(i)
          rmsd0=rmsd0+((xtar(i)-xtar2(i))**2+ &
               (ytar(i)-ytar2(i))**2+ &
               (ztar(i)-ztar2(i))**2)*amass(i)
          rmsd1=rmsd1+((x(i)-xtar(i))**2+ &
               (y(i)-ytar(i))**2+ &
               (z(i)-ztar(i))**2)*amass(i)
          rmsd2=rmsd2+((x(i)-xtar2(i))**2+ &
               (y(i)-ytar2(i))**2+ &
               (z(i)-ztar2(i))**2)*amass(i)
       endif
    enddo

    rmsd0 = sqrt(rmsd0/tmass)
    rmsd1 = sqrt(rmsd1/tmass)
    rmsd2 = sqrt(rmsd2/tmass)

    zetatg =(-ONE/(ONE+exp(-czeta*rmsd0))) &
         +(ONE/(ONE+exp(-czeta*0.0)))
    tmdrhof  =(-ONE/(ONE+exp(-czeta*rmsd1))) &
         +(ONE/(ONE+exp(-czeta*rmsd2)))

    if (prnlev >= 2) then
       WRITE(OUTU,*) 'TMD(inizet): rmsd1, rmsd2  ',  &
            rmsd1, rmsd2
       write(OUTU,*) '  '
       write(OUTU,'((8x,A,F12.7,/),(8x,A,F12.7,/))') &
            ' TMD(inizet): Initial value of Zeta = ', tmdrhof, &
            ' TMD(inizet): Target  value of Zeta = ', zetatg
       write(OUTU,*) 'TMD: ', tmdntar,  &
            ' atoms selected for fitting targets.'
       write(OUTU,*) 'TMD: ', tmdntar2,  &
            ' atoms selected for TMD constraint.'
    endif

    return
  END subroutine inizet

  !=======================================================================
  subroutine shiftz(x_new,y_new,z_new,x_old,y_old,z_old,xt1,yt1,zt1, &
       xt2,yt2,zt2, &
       amass,tol_corr, &
       JSLCT, &
       xdiff1,ydiff1,zdiff1, &
       xdiff2,ydiff2,zdiff2, &
       lagrx, lagry, lagrz, lagsum)
    !=======================================================================

    ! Used to minimize the Lagrange Undetermined Multiplier, lambda
    ! gamma is actually minimixed, where:  gamma = (1/2)*lambda*delt**2
    ! The constraint used is (zeta - tmdrhof) = 0
    ! The function (zeta - tmdrhof)**2 is minimized
    ! This is to assure that the steps taken in the minimization go towards
    ! (zeta - tmdrhof) = 0
    ! Otherwise, for the the non-squared function, convergence may be a 
    ! problem.
    !
    ! PRESHZ is called before, for calculation lagrx, lagry, lagrz, lagsum.
    !
    ! tmdrhof is the Current rho0  (rho0=rho0-drho calculated before cons iteration)
    !
    !     Author: Mark Formaneck, May 2004
    !
    use number
    !C  use parallel  ! - to be worked on later
    use stream

    real(chm_real)  x_new(*),y_new(*),z_new(*)
    real(chm_real)  x_old(*),y_old(*),z_old(*)
    real(chm_real)  xt1(*),yt1(*),zt1(*)
    real(chm_real)  xt2(*),yt2(*),zt2(*)
    real(chm_real)  amass(*), tol_corr
    integer jslct(*)
    real(chm_real)  xdiff1(*),ydiff1(*),zdiff1(*)
    real(chm_real)  xdiff2(*),ydiff2(*),zdiff2(*)
    real(chm_real)  lagrx(*), lagry(*), lagrz(*), LAGSUM


    integer i
    integer MXITER, istart, iend
    real(chm_real)  TOL, GAMMA

    real(chm_real)  rmsd1n, rmsd2n
    real(chm_real)  exrm1n, exrm2n
    real(chm_real)  dnom1n, dnom2n
    real(chm_real)  difnw1, difnw2

    integer niter
    real(chm_real)  ZETA
    real(chm_real)  drmsd1, drmsd2, DZETA, DGAMMA
    real(chm_real)  delt_x, delt_y, delt_z

    real(chm_real)  tmass, fact
    real(chm_real)  zcons

    ! =================================================================
    ! Tolerance is for the change in GAMMA undetermined multiplier iteration
    !    Set value is = 1.0e-6
    ! Tolerance for contraint (zeta-tmdrhof = 0) is ZTOL
    !    Default = 1.0e-10 (rsmall)
    !
    ! There is a separate tolerance for the overall rms diff for the
    ! change of x_new -> x_new + dx.
    ! This uses tol_corr, which is checked outside this subroutine,
    ! because it is also affected by SHAKE.  SHAKE and TMD are both
    ! iterated, until tol_diff is small enough.
    !
    ! Default  MXITER = 1000
    MXITER = zmxitr 

    ! Starting GAMMA is 0.0 - initial guess is equivalent to no constraint
    GAMMA=ZERO
    !===============================================================
    ! PARALLEL
    ! These variables will have to be dealt with in the parallelization
    ! Not implemented
    istart=tmdfrst2
    iend  =tmdlast2

    ! ================================================================
    ! ARRAYS
    !
    ! Read in the configuration data
    ! x_old - at time t
    ! x_new - at time (t+delt), solved without constraint
    !         or the product of the last iteration of this subroutine
    !         (and SHAKE)
    ! xt1 - first  target structure
    ! xt2 - second target structure
    !
    ! =====================================================================

    ! Calculation of Quantities NOT dependent on GAMMA - not updated

    difnw1=ZERO
    difnw2=ZERO

    tmass = ZERO
    DO i = istart, iend
       if(jslct(i)  ==  1) then

          tmass = tmass + amass(i)

          xdiff1(i) = x_new(i) - xt1(i)
          ydiff1(i) = y_new(i) - yt1(i)
          zdiff1(i) = z_new(i) - zt1(i)
          xdiff2(i) = x_new(i) - xt2(i)
          ydiff2(i) = y_new(i) - yt2(i)
          zdiff2(i) = z_new(i) - zt2(i)

          difnw1=difnw1+((xdiff1(i))*LAGRX(i) &
               + (ydiff1(i))*LAGRY(i) &
               + (zdiff1(i))*LAGRZ(i))
          difnw2=difnw2+((xdiff2(i))*LAGRX(i) &
               + (ydiff2(i))*LAGRY(i) &
               + (zdiff2(i))*LAGRZ(i))

       endif
    ENDDO
    !
    ! ==================================================================
    ! Loop for Updating GAMMA

    niter = 0
10  continue
    niter = niter + 1

    rmsd1n=ZERO    
    rmsd2n=ZERO    

    DO i = istart, iend
       if(jslct(i)  ==  1) then
          rmsd1n = rmsd1n +((xdiff1(i) + gamma*lagrx(i)/amass(i))**2 &
               + (ydiff1(i) + gamma*lagry(i)/amass(i))**2 &
               + (zdiff1(i) + gamma*lagrz(i)/amass(i))**2) &
               * amass(i)
          rmsd2n = rmsd2n +((xdiff2(i) + gamma*lagrx(i)/amass(i))**2 &
               + (ydiff2(i) + gamma*lagry(i)/amass(i))**2 &
               + (zdiff2(i) + gamma*lagrz(i)/amass(i))**2) &
               * amass(i)
       endif
    ENDDO

    rmsd1n=DSQRT(rmsd1n/tmass)
    rmsd2n=DSQRT(rmsd2n/tmass)

    if(prnlev >= 9) write(outu,*) 'TMD(shiftz):  NEW RMSDs  ',  &
         rmsd1n, rmsd2n

    exrm1n =  EXP(-czeta*rmsd1n)
    dnom1n =  rmsd1n*(TWO+exrm1n+ONE/exrm1n)
    exrm2n =  EXP(-czeta*rmsd2n)
    dnom2n =  rmsd2n*(TWO+exrm2n+ONE/exrm2n)

    zeta=(-(ONE)/(1.0+exrm1n) &
         +(ONE)/(1.0+exrm2n))

    drmsd1=(difnw1 + gamma*lagsum)
    drmsd2=(difnw2 + gamma*lagsum)
    dzeta=TWO*(zeta-tmdrhof) &
         *(-czeta/tmass)*(drmsd1/dnom1n-drmsd2/dnom2n)

    zcons=zeta-tmdrhof
    dgamma= -(zeta-tmdrhof)**2/dzeta
    gamma=gamma + dgamma
    if(prnlev >= 9) write(outu,*) ' niter,gamma,dgamma,zeta,dzeta ',  &
         niter,gamma,dgamma,zeta,dzeta 
    IF (niter > mxiter)    GOTO 20  !
    IF (abs(zcons) < ztol) GOTO 30  ! z-z0 = 0 constraint satified
    ! within tolerance (ztol)
    GOTO 10

20  continue
    if (prnlev >= 2) then
       write(outu,*) '*** TMD Constraint Error: Subroutine SHIFTZ'
       write(outu,*) '***    Exceeded Maximum gamma Interations = ', &
            mxiter
       write(outu,*) '  '
       write(outu,*) '  '
    endif
    call wrndie(-5,'<TMD:shiftz>','TMD constraint is NOT Converged')
30  continue
    IF (abs(dgamma)  >  0.000001d0) THEN
       if(PRNLEV >= 9) THEN
          WRITE(outu,*) 'ZSHIFT: zeta tolerance satified -- ', ztol
          WRITE(outu,*) '      : but gamma step size still large '
          WRITE(outu,*) '      : dgamma ', dgamma,' > 1.0e-6'
       endif
       GOTO 10   ! check that the step size is small
       ! if not, try again
    ENDIF
    ! ========================================================================

    IF(prnlev > 5)THEN
       write(outu,*) ' TMD monitor (r1,r2,z): ', rmsd1n, rmsd2n, zeta
    ENDIF


    ! Update the new coordinates
    ! Check Tolerance - check is in dynamc/dynamc.src

    tol_corr = ZERO
    IF((IEND /= 0).AND.(ISTART.NE.0)) THEN
       do i=istart,iend
          if(jslct(i) == 1) then
             delt_x=gamma*lagrx(i)/amass(i)
             delt_y=gamma*lagry(i)/amass(i)
             delt_z=gamma*lagrz(i)/amass(i)
             x_new(i)=x_new(i)+delt_x
             y_new(i)=y_new(i)+delt_y
             z_new(i)=z_new(i)+delt_z
             tol_corr=tol_corr+delt_x**2+delt_y**2+delt_z**2
          endif
       enddo
    ENDIF
    !
#if KEY_PARALLEL==1
    !      call gcomb(tol_corr,1)                               
#endif
    tol_corr=sqrt(tol_corr)
    if(prnlev >= 9) write(outu,*) 'tol_corr in ShiftZ = ', tol_corr
    !
    ! ==========================================================================
    !
    return
  end subroutine shiftz

  !=======================================================================
  subroutine preshz(x_old,y_old,z_old,xt1,yt1,zt1, &
       xt2,yt2,zt2, &
       amass, &
       JSLCT, &
       lagrx, lagry, lagrz, lagsum)
    !=======================================================================

    ! Preparation for shiftZ subroutine
    ! Quantities calculated here are functions of OLD and TARGET coordinates
    ! which are not changed in the iterations to find GAMMA
    ! nor in the iterations between shiftz and shake (holonoma)

    use number
    !C  use parallel  ! - to be worked on later
    use stream

    real(chm_real)  x_old(*),y_old(*),z_old(*)
    real(chm_real)  xt1(*),yt1(*),zt1(*)
    real(chm_real)  xt2(*),yt2(*),zt2(*),amass(*)
    integer jslct(*)
    real(chm_real)  lagrx(*), lagry(*), lagrz(*), LAGSUM


    integer i
    integer istart, iend

    real(chm_real)  rmsd1o, rmsd2o
    real(chm_real)  exrm1o, exrm2o
    real(chm_real)  dnom1o, dnom2o

    real(chm_real)  tmass, fact

    ! PARALLEL
    ! These variables will have to be dealt with in the parallelization
    ! Not implemented
    istart=tmdfrst2
    iend  =tmdlast2

    rmsd1o = ZERO
    rmsd2o = ZERO

    tmass=ZERO
    DO i = tmdfrst2, tmdlast2
       if(jslct(i)  ==  1) then
          tmass = tmass + amass(i)
          rmsd1o = rmsd1o + ((x_old(i) - xt1(i))**2 &
               +  (y_old(i) - yt1(i))**2 &
               +  (z_old(i) - zt1(i))**2)*amass(i)
          rmsd2o = rmsd2o + ((x_old(i) - xt2(i))**2 &
               +  (y_old(i) - yt2(i))**2 &
               +  (z_old(i) - zt2(i))**2)*amass(i)
       endif
    ENDDO

    rmsd1o=DSQRT(rmsd1o/tmass)
    rmsd2o=DSQRT(rmsd2o/tmass)

    if(prnlev >= 9) write(outu,*) 'TMD(shiftz):  OLD RMSDs',  &
         rmsd1o, rmsd2o

    exrm1o =  exp(-czeta*rmsd1o)
    dnom1o =  rmsd1o*(TWO+exrm1o+ONE/exrm1o)
    exrm2o =  exp(-czeta*rmsd2o)
    dnom2o =  rmsd2o*(TWO+exrm2o+ONE/exrm2o)

    LAGSUM=ZERO

    DO i = istart, iend
       if(jslct(i)  ==  1) then
          fact = czeta*amass(i)/tmass
          LAGRX(i) = fact*(-(x_old(i)-xt1(i))/dnom1o &
               +(x_old(i)-xt2(i))/dnom2o)

          LAGRY(i) = fact*(-(y_old(i)-yt1(i))/dnom1o &
               +(y_old(i)-yt2(i))/dnom2o)

          LAGRZ(i) = fact*(-(z_old(i)-zt1(i))/dnom1o &
               +(z_old(i)-zt2(i))/dnom2o)

          LAGSUM = LAGSUM + (LAGRX(I)**2 + LAGRY(I)**2 &
               +  LAGRZ(I)**2) / amass(i)

       endif
    ENDDO

    return
    !
  end subroutine preshz

  subroutine tmdprint(outu)
    !     printing of TMD stuff
    !
    integer outu

    !
    write(outu,30)
    if(qzeta)then
       write(outu,190) czeta
       write(outu,200) tmdrhof
       write(outu,210) zetatg
    else
       write(outu,40) tmdrhof
       !%% AvdV addition
       if(tmdfmax2 >= 0.0d0)then
          if(isump == 0)then
             write(outu,80) sqrt(tmdfmax2)
          else
             write(outu,90) sqrt(tmdfmax2)
          endif
       else
          !%% AvdV end addition
          write(outu,100) drho
          !%% AvdV addition
       endif
       if(tmdbmax2 >= 0.0D0)then
          if(isump == 0)then
             write(outu,110) sqrt(tmdbmax2)
          else
             write(outu,120) sqrt(tmdbmax2)
          endif
       endif
       !%% AvdV end addition
    endif
    write(outu,157) inrt
    write(outu,160) tmdfrms
    write(outu,170) tmdntar
    write(outu,175) tmdntar2
    !%% AvdV addition
    if(.not.qzeta)then
       if(itmd > 0)then
          write(outu,180) itmd,ftmd
          if(tmdcalc)then
             write(outu,185)
          endif
       endif
    endif
    !%% AvdV end addition
    write(outu,'(" ")')

    !      
30  FORMAT(' DCNTRL> Targeted molecular dynamics requested.')
40  FORMAT(8X,' Distance with target structure = ',F12.7,' A')
    !%% AvdV addition
80  FORMAT(8X,' Maximum forward displacement per atom = ', &
         F15.10,' A')
90  FORMAT(8X,' Maximum total forward displacement = ',F15.10,' A')
    !%% AvdV end addition
100 FORMAT(8X,' Distance increment = ',F12.7,' A')
    !%% AvdV addition
110 FORMAT(8X,' Favor perturbation along unperturbed direction ', &
         /,8X,'   else maximum backward displacement per atom = ', &
         F15.10,' A')
120 FORMAT(8X,' Favor perturbation along unperturbed direction ', &
         /,8X,'   else maximum total backward displacement = ', &
         F15.10,' A')
    !%% AvdV end addition
157 FORMAT(8X,' Step frequency for stopping rotation  = ',I5)
160 FORMAT(8X,' Stop when distance with target is less than ', &
         F15.10,' A')
170 FORMAT(8X,' Total number of TMD atoms to fit = ',I5)
175 FORMAT(8X,' Total number of TMD atoms to perturb = ',I5)
    !%% AvdV addition
180 FORMAT(8X,' Write TMD analysis file to unit ',I3, &
         /,8X,'   with frequency of ',i5)
185 FORMAT(8X,'   and calculate Delta ENER')
    !%% AvdV end addition
190 FORMAT(8X,' Using ZETA form of TMD constraint',/, &
         8X,' Exponential Factor for Zeta (default 1.0) = ',F12.7)
200 FORMAT(8X,' Current Value for Zeta  = ',F12.7)
210 FORMAT(8X,' Target  Value for Zeta  = ',F12.7)
    ! 

    return
  end subroutine tmdprint

  subroutine freetmd
    !     This routine creates TMD arrays with dimension 1, so that routines will
    !     not complain.
    !     by Arjan van der Vaart
    use number
    use psf
    use coord
    use memory

    if(qtmd)then
       !     .  TMD was active: delete all associated arrays
       !         deallocate(ixtar,iytar,iztar)
       !         deallocate(itrcm,istmd,jstmd)
       call chmdealloc('tmd.src','freetmd','ixtar',natom,crl=ixtar)
       call chmdealloc('tmd.src','freetmd','iytar',natom,crl=iytar)
       call chmdealloc('tmd.src','freetmd','iztar',natom,crl=iztar)
       call chmdealloc('tmd.src','freetmd','itrcm',natom,crl=itrcm)
       call chmdealloc('tmd.src','freetmd','istmd',natom,intg=istmd)
       call chmdealloc('tmd.src','freetmd','istmd',natom,intg=jstmd)
       if(qzeta)then
          !            deallocate(ixtar2,iytar2,iztar2)
          !            deallocate(ix1tmd,iy1tmd,iz1tmd)
          !            deallocate(ix2tmd,iy2tmd,iz2tmd)
          !            deallocate(ixgtmd,iygtmd,izgtmd)
          call chmdealloc('tmd.src','freetmd','ixtar2',NATOM,crl=ixtar2)
          call chmdealloc('tmd.src','freetmd','iytar2',NATOM,crl=iytar2)
          call chmdealloc('tmd.src','freetmd','iztar2',NATOM,crl=iztar2)
          call chmdealloc('tmd.src','freetmd','ix1tmd',NATOM,crl=ix1tmd)
          call chmdealloc('tmd.src','freetmd','iy1tmd',NATOM,crl=iy1tmd)
          call chmdealloc('tmd.src','freetmd','iz1tmd',NATOM,crl=iz1tmd)
          call chmdealloc('tmd.src','freetmd','ix2tmd',NATOM,crl=ix2tmd)
          call chmdealloc('tmd.src','freetmd','iy2tmd',NATOM,crl=iy2tmd)
          call chmdealloc('tmd.src','freetmd','iz2tmd',NATOM,crl=iz2tmd)
          call chmdealloc('tmd.src','freetmd','ixgtmd',NATOM,crl=ixgtmd)
          call chmdealloc('tmd.src','freetmd','iygtmd',NATOM,crl=iygtmd)
          call chmdealloc('tmd.src','freetmd','izgtmd',NATOM,crl=izgtmd)
          !%% AvdV addition
       elseif(itmd > 0)then
          !            deallocate(itmdx,itmdy,itmdz)
          call chmdealloc('tmd.src','freetmd','itmdx',NATOM,crl=itmdx)
          call chmdealloc('tmd.src','freetmd','itmdy',NATOM,crl=itmdy)
          call chmdealloc('tmd.src','freetmd','itmdz',NATOM,crl=itmdz)
          if(tmdcalc)then
             !               deallocate(itmdxx,itmdyy,itmdzz)
             call chmdealloc('tmd.src','freetmd','itmdxx',NATOM,crl=itmdxx)
             call chmdealloc('tmd.src','freetmd','itmdyy',NATOM,crl=itmdyy)
             call chmdealloc('tmd.src','freetmd','itmdzz',NATOM,crl=itmdzz)
          endif
          !%% AvdV end addition
       endif
    endif
    !
    !     Deactivate TMD
    qtmd=.false.      
    qzeta=.false.
    tmdcalc=.false.
    tmdwr=.false.
    !

    return
  end subroutine freetmd
  !
  !%% AvdV addition

  subroutine tmdan(xtar,ytar,ztar, &
       tmpx,tmpy,tmpz,bnbnd,bimag, &
       tmde,teterm,teprop,tepress, &
       jslct,istep,atfrst,atlast,tx,ty,tz)

    use energym
    use eutil
    use coord
    use coordc
    use deriv
    use consta
    !%% AvdV addition
    use parallel
    use stream
    use psf
    !%% AvdV end addition
    !     arguments:
    real(chm_real) xtar(*),ytar(*),ztar(*), &
         tmpx(*),tmpy(*),tmpz(*), &
         teterm(*),teprop(*),tepress(*),tmde
    integer jslct(*),istep,atfrst,atlast   !!,bnbnd(*),bimag(*)
    type(nonbondDataStructure) bnbnd
    type(imageDataStructure) bimag

    !     locals:
    real(chm_real) tmdptot,tmdxtot,tmdcost,tmdpmax, &
         tmpxx,tmpyy,tmpzz,tmdp, &
         tmdx,tmddot,ri,l1
    real(chm_real) tx(*),ty(*),tz(*)
    integer i,istart,iend

#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
    ATFRST=1+IPARPT(MYNOD)
    ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
    ATFRST=1
    ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /* (paramain)*/
    ATFRST=1
    ATLAST=NATOM
#endif /* (paramain)*/
    !
    !     Set the limits for each process
    istart=tmdfrst2
    if(tmdfrst2 <= atfrst)istart=atfrst
    if(tmdfrst2 > atlast)istart=0
    iend=tmdlast2
    if(tmdlast2 > atlast)iend=atlast
    if(tmdlast2 < atfrst)iend=0

    tmdptot=0.0D0
    tmdxtot=0.0D0
    tmdcost=0.0D0
    tmdpmax=0.0D0
    IF((IEND /= 0).AND.(ISTART.NE.0)) THEN
       do I=istart,iend
          if(jslct(i) == 1)then
             tmpxx=x(i)-tmpx(i)
             tmpyy=y(i)-tmpy(i)
             tmpzz=z(i)-tmpz(i)
             !     .        tmdp = |p_i|
             tmdp=sqrt(tmpxx*tmpxx+tmpyy*tmpyy+tmpzz*tmpzz)
             !     .        tmdpmax = Max |p_i|
             tmdpmax=max(tmdpmax,tmdp)
             !     .        tmdptot = Sum |p_i|
             tmdptot=tmdptot+tmdp
             tmpx(i)=tmpx(i)-xcomp(i)
             tmpy(i)=tmpy(i)-ycomp(i)
             tmpz(i)=tmpz(i)-zcomp(i)
             !     .        tmdx = |x_i| (unperturbed)
             tmdx=sqrt(tmpx(i)*tmpx(i)+tmpy(i)*tmpy(i) &
                  +tmpz(i)*tmpz(i))
             !     .        tmdxtot = Sum |x_i| (unperturbed)
             tmdxtot=tmdxtot+tmdx
             !     .        tmddot = |p_i| cos(p_i,F_i) 
             !     .             (perturbation component along unperturbed force)
             tmddot=tmpxx*dx(i)+tmpyy*dy(i)+tmpzz*dz(i)
             tmddot=tmddot/sqrt(dx(i)*dx(i) &
                  +dy(i)*dy(i)+dz(i)*dz(i))
             !     .        tmdcost = Sum |p_i| cos(p_i,F_i) 
             !     .             (perturbation component along unperturbed force)
             tmdcost=tmdcost+tmddot
          endif
       enddo
    endif

#if KEY_PARALLEL==1
    !     Sum for parallel:
    call gcomb(tmdptot,1)
    call gcomb(tmdxtot,1)
    call gcomb(tmdcost,1)
    call gcombmax(tmdpmax)
#endif 
    tmdcost=-tmdcost

    IF(tmdcalc) THEN                       
       !     .  perform one extra energy call to calculate the potential 
       !     .  energy change upon TMD. It is important that the pairlist
       !     .  remains the same; the reason why we have 2 extra energy 
       !     .  calls total (rather than just 1) when tmdcalc is active
       IF((IEND /= 0).AND.(ISTART.NE.0)) THEN
          do I=1,natom
             tx(i)=x(i)
             ty(i)=y(i)
             tz(i)=z(i)
          enddo
       endif
#if KEY_PARALLEL==1
       call vdgbr(x,y,z,1)
#endif 
       CALL GETE(x,y,z,x,y,z,0)
       do I=1,natom
          x(i)=tx(i)
          y(i)=ty(i)
          z(i)=tz(i)
       enddo
       !
       !
       !     .  we are only interested in the change in potential energy 
       !     .  bring the energies back to its original values
       tmde=eprop(3)-tmde
       !c         write(*,*)'TMDAN>me,tmde=',mynod,tmde
       do i=1,lenent
          eterm(i)=teterm(i)
       enddo
       do i=1,lenenp
          eprop(i)=teprop(i)
       enddo
       do i=1,lenenv
          epress(i)=tepress(i)
       enddo
    endif

    !     i;  Max |p_i|; Sum |p_i| ; Sum |x_i|; Sum |p_i| cos(p_i,F_i); rho0 (; delta ENER)
    if(prnlev > 2)then
       if(tmdcalc)then
          write(itmd,'(i8,6(1x,F13.6))') &
               istep,tmdpmax,tmdptot,tmdxtot,tmdcost,tmdrhof, &
               tmde
       else
          write(itmd,'(i8,5(1x,F13.6))') &
               istep,tmdpmax,tmdptot,tmdxtot,tmdcost,tmdrhof
       endif
    endif
    !%% AvdV end addition
    return
  end subroutine tmdan


#else /* (if_tmd)*/
  SUBROUTINE TMDINIT(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     This routine interprets commands initializing the
    !     targeted molecular dynamics (TMD) method.
    !
    !     Author: Jianpeng Ma (8/96)
    !     Modified by Arjan van der Vaart
    !
    !-----------------------------------------------------------------------
    character(len=*) COMLYN
    INTEGER COMLEN
    CALL WRNDIE(0,'<TMD>','TMD code not compiled')
    RETURN
  END SUBROUTINE TMDINIT
#endif /* (if_tmd)*/

end module tmd

