!     Routines using Isotropic Periodicall Sum (IPS) to calculate 
!     long-range interactions
!  
!      By Xiongwu Wu, 8/15/2003 at NHLBI/NIH 
!
module aips_module
  use chm_kinds
  implicit none

  real(chm_real),allocatable,dimension(:),save :: wnb
  real(chm_real),allocatable,dimension(:),save :: elearray,vdwarray
  real(chm_real),allocatable,dimension(:),save :: elexx,elexy, &
       elexz,eleyy,eleyz,elezz,vdwxx,vdwxy,vdwxz,vdwyy,vdwyz,vdwzz
  real(chm_real),allocatable,dimension(:) :: qarray,warray
  real(chm_real),allocatable,dimension(:) :: FR1,FR2,FR3,CG1
  real(chm_real),save :: ctensor

  integer,save :: nattot,ierr_allocate,alloc_err

contains
#if KEY_NBIPS==1 /*nbips_main*/

  !****************************************************************
  !                        AIPSFIN
  !****************************************************************
  SUBROUTINE AIPSFIN(ENBAIPS,EELAIPS,IFRSTA,ILASTA,NATOM, &
       LVDWX,LELECX,LVIPS,LEIPS, &
       EPS,CG,X,Y,Z,DX,DY,DZ) 
    !-----------------------------------------------------------------------
    !    Calculate grid based anisotropic IPS interaction
    !
    use pmeutil,only:nfft1,nfft2,nfft3,sizfftab,sizffwrk,forder, &
         fft1_table,fft2_table,fft3_table,ffwork, &
         mxystart,mxyslabs,mxzslabs,get_fftdims &
         ,pll_fft_setup,allocate_bspline,deallocate_bspline &
         ,FFT3D0RC
    use pme_module, only:fft_forwardrc,fft_backrc, &
         tmpy,alpha,beta
    use new_timer,only:timer_start,timer_stop,timer_stpstrt,  & 
         T_ipsafunc, T_ipsagrid,T_ipsasum,T_ipsaforce,T_ipsafft 
    use dimens_fcm
    use exfunc
    use number
    use consta
    use stream
    use energym
    use image
    use nbips
    use parallel
    use memory
    integer,allocatable,dimension(:) :: lmy_ks
    LOGICAL LVDW,LELEC,LVIPS,LEIPS,LVDWX,LELECX
    INTEGER IFRSTA,ILASTA,NATOM
    real(chm_real) ENBAIPS,EELAIPS,EPS,CGF,CG(*)
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    real(chm_real),allocatable,dimension(:) :: xr,yr,zr
    real(chm_real)  XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX
    real(chm_real)  XTLINV(6),RECIP(3,3)
    INTEGER SIZ_Q
    INTEGER THETA1,THETA2,THETA3,DTHETA1,DTHETA2,DTHETA3
    INTEGER LATM
    !
    real(chm_real) scale,BL,BLC,VBOX
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3,NFFTABLE,NFFWORK
    INTEGER I,J,NN
    INTEGER IGOOD, KBOT, KTOP
    !
    LOGICAL OK,QMEM
    real(chm_real)  ATEMP(5)                  
    !
    LVDW=LVDWX.AND.LVIPS
    LELEC=LELECX.AND.LEIPS
    CGF=CCELEC/EPS
    NATTOT=NATOM
    !
    !  Finite system
    !  Orient the system to have the inertia moments along x, y, z axis
    allocate(xr(nattot),yr(nattot),zr(nattot),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate xr,yr,zr"
    CALL AIPSROT(NATOM,X,Y,Z,XR,YR,ZR, &
         XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX,URIPS,UVIPS,QIPSUPD)
    QMEM=.FALSE.
    BL=XMAX-XMIN
    CALL IPSGRID(QIPSFIN,MIPSX,MIPSO,GIPSX,BL,NN,BLC)
    XTLABC(1)=BLC
    QMEM=QMEM.OR.(NFFT1 /= NN)
    NFFT1=NN
    BL=(YMAX-YMIN)
    CALL IPSGRID(QIPSFIN,MIPSY,MIPSO,GIPSY,BL,NN,BLC)
    XTLABC(3)=BLC
    QMEM=QMEM.OR.(NFFT2 /= NN)
    NFFT2=NN
    BL=(ZMAX-ZMIN)
    CALL IPSGRID(QIPSFIN,MIPSZ,MIPSO,GIPSZ,BL,NN,BLC)
    XTLABC(6)=BLC
    QMEM=QMEM.OR.(NFFT3 /= NN)
    NFFT3=NN
    XTLABC(2)=ZERO
    XTLABC(4)=ZERO
    XTLABC(5)=ZERO
    RECIP(1,1) = URIPS(1)/XTLABC(1)
    RECIP(2,1) = URIPS(2)/XTLABC(1)
    RECIP(3,1) = URIPS(3)/XTLABC(1)
    RECIP(1,2) = URIPS(4)/XTLABC(3)
    RECIP(2,2) = URIPS(5)/XTLABC(3)
    RECIP(3,2) = URIPS(6)/XTLABC(3)
    RECIP(1,3) = URIPS(7)/XTLABC(6)
    RECIP(2,3) = URIPS(8)/XTLABC(6)
    RECIP(3,3) = URIPS(9)/XTLABC(6)
    !
    VBOX=XTLABC(1)*XTLABC(3)*XTLABC(6)
    ! 
    !
    IF(ABS(ONE-VBIPS/VBOX) > DVBIPS)THEN
       QIPSUPD=.TRUE.
       VBIPS=VBOX
    ELSE
       QIPSUPD=.FALSE.
    ENDIF
    !
    !   Get memory for scratch arrays, free them after summation.
    !
    FORDER=MIPSO
    CALL GET_FFTDIMS( &
         NFFTDIM1,NFFTDIM2,NFFTDIM3,NFFTABLE,NFFWORK)
    SCALE = ONE
    !
    IF(QMEM) &
         call pll_fft_setup(nfftdim1,nfftdim2,nfftdim3)
#if KEY_PARALLEL==1
    SIZ_Q = max(2*NFFTDIM1*NFFTDIM2*mxyslabs,  &
         2*NFFTDIM1*NFFTDIM3*mxzslabs)
#else /**/
    SIZ_Q = 2*NFFTDIM1*NFFTDIM2*NFFTDIM3
#endif 
    !
    allocate(qarray(siz_q),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray"
    allocate(warray(siz_q),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate warray"
    allocate(fr1(nattot),fr2(nattot),fr3(nattot),cg1(nattot), &
         stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate fr arrays"
    call allocate_bspline(natom,nattot)
    allocate(tmpy(2*nfftdim1),alpha(nfft1),beta(nfft1), &
         stat=alloc_err)
    if(alloc_err /= 0 ) &
         write(0,*)"unable to allocate tmpy,alpha,beta"

    !  Change grid size to for FFT
    IF(QMEM)THEN
       !    Initiate FFT setting
       CALL AIPS_SETUP(NATOM,1)
       !
       call FFT3D0rc(0,scale,qarray, &
            nfftdim1,nfftdim2, &
            tmpy,alpha,beta)
       IF(allocated(elearray))THEN
          deallocate(elearray,vdwarray,stat=alloc_err)
          if(alloc_err /= 0 ) write(0,*)"unable to deallocate elearray"
       ENDIF
       allocate(elearray(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) then
          ! if(allocated(elearray)) write(0,*) "already allocated: elearray", size(elearray)
          write(0,*)"unable to allocate elearray", alloc_err
       endif
       allocate(vdwarray(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate vdwarray"
       !  end of memory allocation
    ENDIF
    IPSSIZ=SIZ_Q
    IF(QMEM.OR.QIPSUPD)THEN
       call timer_start(T_ipsafunc)     
       ! Update energy function grid. stored at the image part of Q and W
       elearray(1:SIZ_Q)=zero
       vdwarray(1:SIZ_Q)=zero
#if KEY_PARALLEL==1
       KBOT = MXYSTART(MYNOD) + 1
       KTOP = MXYSTART(MYNOD) + MXYSLABS
#else /**/
       kbot = 1
       ktop = nfft3
#endif 
       CALL FIN_ENG_GRID(LVDW,LELEC,kbot, ktop, &
            NFFTDIM1,NFFTDIM2,NFFTDIM3, &
            XTLABC,CGF)
       call timer_stpstrt(T_ipsafunc,T_ipsafft)     
       IF(LELEC)THEN
          call fft_backrc( &
               ELEARRAY, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
       ENDIF
       IF(LVDW)THEN
          call fft_backrc( &
               VDWARRAY, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
       ENDIF
       call timer_stop(T_ipsafft)     
    ENDIF
    !       make array for keeping track of atoms important to
    !         this processor
    latm=nattot
    call chmalloc('enbips.src','AIPSFIN','lmy_ks',latm,intg=lmy_ks)

    !-------- symmetrical case ------------------------------
    !        fill frac coords and thetas in fill_ch_grid
    !              use min image charge array: cg
    call timer_start(T_ipsagrid)     
    QARRAY(1:SIZ_Q)=zero
    WARRAY(1:SIZ_Q)=zero                          
    CALL FIN_IPS_GRID( &
         igood, kbot, ktop, &
         NATOM,CG, &
         XR,YR,ZR,XTLABC, &
         NFFTDIM1,NFFTDIM2, &
         LMY_KS,LATM, &
#if KEY_PARALLEL==1
         mxyslabs &           
#endif
#if KEY_PARALLEL==0
         NFFTDIM3 &           
#endif
         )
    call timer_stpstrt(T_ipsagrid,T_ipsafft)     
    IF(LELEC)THEN
       call fft_backrc( &
            qarray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
    ELSE
       qarray(1:SIZ_Q)=zero
    ENDIF
    IF(LVDW)THEN
       call fft_backrc( &
            warray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
    ELSE
       warray(1:SIZ_Q)=zero
    ENDIF
    ! Calculate grid IPS potentials
    call timer_stpstrt(T_ipsafft,T_ipsasum)     
    CALL FIN_IPS_SUM(ENBAIPS,EELAIPS,LVDW,LELEC,CGF,EPROP(VOLUME), &
         NFFTDIM1,NFFTDIM2,NFFTDIM3, &
         XTLABC)    
    !  Forward FFT to get potential surface
    !
    call timer_stpstrt(T_ipsasum,T_ipsafft)     
    IF(LELEC)THEN
       call fft_forwardrc( &
            qarray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
    ELSE
       qarray(1:SIZ_Q)=zero
    ENDIF
    IF(LVDW)THEN
       call fft_forwardrc( &
            warray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
    ELSE
       warray(1:SIZ_Q)=zero
    ENDIF
    call timer_stpstrt(T_ipsafft,T_ipsaforce)     
    !  Calculate forces on atoms by B-spline interpolation
    CALL FIN_IPS_GRAD( &
         IGOOD, KBOT, KTOP, &
         NATOM,CG,RECIP, &
         X,Y,Z,DX,DY,DZ, &
         FR1,FR2,FR3, &
         NFFTDIM1,NFFTDIM2,NFFTDIM3, &
         LMY_KS,LATM, &
         1,XTLABC)
    !
    !=======================================================================
    !   Main loop end
    !=======================================================================
    !
    call timer_stop(T_ipsaforce)     
    call chmdealloc('enbips.src','AIPSFIN','LMY_KS',LATM,intg=LMY_KS)
    !
    deallocate(tmpy,alpha,beta,stat=alloc_err)
    if(alloc_err /= 0 ) &
         write(0,*)"unable to deallocate tmpy,alpha,beta"
    deallocate(xr,yr,zr,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate xr,yr,zr"
    deallocate(fr1,fr2,fr3,cg1,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate fr1,2,3"
    deallocate(qarray,warray,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray"
    call deallocate_bspline()
    !  Print out the long rang corretions parameters
    ATEMP(1)=ENBAIPS
    ATEMP(2)=EELAIPS
    ATEMP(3)=PIPSVIR(1)
    ATEMP(4)=PIPSVIR(5)
    ATEMP(5)=PIPSVIR(9)
#if KEY_PARALLEL==1 /*EIPSS*/
    IF(NUMNOD > 1)THEN
       CALL GCOMB(ATEMP,5)
    ENDIF
#endif /* (EIPSS)*/
    IF(PRNLEV >= 6)THEN
#if KEY_PARALLEL==1
       IF(MYNOD == 0)THEN                               
#endif
          WRITE(OUTU,'("  ENBGRID,EELGRID  = ",6E14.7)') &
               (ATEMP(I),I=1,2)
          WRITE(OUTU,'("  PIPSXX  = ",3E14.7)') &
               ATEMP(3),ATEMP(4),ATEMP(5)
#if KEY_PARALLEL==1
       ENDIF                               
#endif
    ENDIF
    RETURN 
  END SUBROUTINE AIPSFIN


  SUBROUTINE AIPSROT(NAT,X,Y,Z,XR,YR,ZR, &
       XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX,U,UV,LROT)
    !-----------------------------------------------------------------------
    !  This routine rotate system using matrix UMATRIX.  
    !  UMATRIX is generated when LROT is true to orient a system on axises
    !
    use number
    use stream
    !
    INTEGER NAT
    real(chm_real) X(*),Y(*),Z(*),XR(*),YR(*),ZR(*)
    real(chm_real) XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX,U(9),UV(9)
    LOGICAL LROT
    !
    real(chm_real) XC,YC,ZC,XN,YN,ZN,XI,YI,ZI
    INTEGER NPR,N,I,J,IPT
    real(chm_real) SCR(24),AMOM(6),EV(3)
    real(chm_real) XX,XY,XZ,YY,YZ,ZZ,DET
    LOGICAL OK
    !
    N=NAT
    !
    ! Calculate the center of mass
    !
    XC=ZERO
    YC=ZERO
    ZC=ZERO
    DO I=1,N
       XC=XC+X(I)
       YC=YC+Y(I)
       ZC=ZC+Z(I)
    ENDDO
    XC=XC/N
    YC=YC/N
    ZC=ZC/N
    !
    ! Process best rotation calculation
    !
    XX=ZERO
    XY=ZERO
    XZ=ZERO
    YY=ZERO
    YZ=ZERO
    ZZ=ZERO
    DO I=1,N
       XI=X(I)-XC
       YI=Y(I)-YC
       ZI=Z(I)-ZC
       XR(I)=XI
       YR(I)=YI
       ZR(I)=ZI
       XX=XX+XI*XI
       XY=XY+XI*YI
       XZ=XZ+XI*ZI
       YY=YY+YI*YI
       YZ=YZ+YI*ZI
       ZZ=ZZ+ZI*ZI
    ENDDO
    !
    AMOM(1)=ZZ
    AMOM(2)=YZ
    AMOM(3)=XZ
    AMOM(4)=YY
    AMOM(5)=XY
    AMOM(6)=XX
    !
    CALL DIAGQ(3,3,AMOM,U,SCR(4),SCR(7),SCR(10),SCR(13),EV, &
         SCR(16),SCR(19),SCR(22),0)
    !
    DO I=1,3
       DET=U(I)
       U(I)=U(I+6)
       U(I+6)=DET
    ENDDO
    DO I=1,3
       IPT=(I-1)*3
       DET=U(IPT+1)
       U(IPT+1)=U(IPT+3)
       U(IPT+3)=DET
       IF(U(IPT+I) < ZERO) THEN
          DO J=1,3
             IPT=IPT+1
             U(IPT)=-U(IPT)
          ENDDO
       ENDIF
    ENDDO
    DET=U(1)*(U(5)*U(9)-U(6)*U(8))+U(2)*(U(6)*U(7)-U(4)*U(9))+ &
         U(3)*(U(4)*U(8)-U(5)*U(7))
    IF(DET < ZERO) THEN
       U(7)=-U(7)
       U(8)=-U(8)
       U(9)=-U(9)
       DET=-DET
    ENDIF
    IF(ABS(DET-ONE) > 1.D-4) WRITE(OUTU,203) DET
203 FORMAT(/' ***** WARNING ***** FROM LSQP. ROTATION MATRIX IS', &
         ' NOT UNITARY.'/,' DETERMINANT=',F14.8/)
    !
    CALL INVT33(UV,U,OK)
    IF(.NOT.OK)THEN
       DO I=1,9
          U(I)=ZERO
          UV(I)=ZERO
       ENDDO
       U(1)=ONE
       U(5)=ONE
       U(9)=ONE
       UV(1)=ONE
       UV(5)=ONE
       UV(9)=ONE
    ENDIF
    XMIN=ZERO
    XMAX=ZERO
    YMIN=ZERO
    YMAX=ZERO
    ZMIN=ZERO
    ZMAX=ZERO
    DO I=1,N
       XI=XR(I)
       YI=YR(I)
       ZI=ZR(I)
       XN=U(1)*XI+U(2)*YI+U(3)*ZI
       YN=U(4)*XI+U(5)*YI+U(6)*ZI
       ZN=U(7)*XI+U(8)*YI+U(9)*ZI
       XR(I)=XN
       YR(I)=YN
       ZR(I)=ZN
       XMIN=HALF*(XN+XMIN-ABS(XMIN-XN))
       XMAX=HALF*(XN+XMAX+ABS(XMAX-XN))
       YMIN=HALF*(YN+YMIN-ABS(YMIN-YN))
       YMAX=HALF*(YN+YMAX+ABS(YMAX-YN))
       ZMIN=HALF*(ZN+ZMIN-ABS(ZMIN-ZN))
       ZMAX=HALF*(ZN+ZMAX+ABS(ZMAX-ZN))
    ENDDO
    RETURN
  END SUBROUTINE AIPSROT



  SUBROUTINE FIN_ENG_GRID(LVDW,LELEC,KBOT,KTOP, &
       NFFTDIM1,NFFTDIM2,NFFTDIM3, &
       XTLABC,CGF)    
    ! Calculate grid IPS potentials
    use pmeutil,only:nfft1,nfft2,nfft3
    use number
    use dimens_fcm
    use consta
    use nbips
    !
    LOGICAL LVDW,LELEC
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    INTEGER KBOT,KTOP,MRCX,MRCY,MRCZ
    real(chm_real)  XTLABC(6),VBOX,CGF
    !
    INTEGER IBINX0,IBINY0,IBINZ0
    INTEGER IBINX1,IBINY1,IBINZ1,IXYZ0,IXYZ1
    INTEGER I000,I100,I010,I001,I110,I011,I101,I111
    real(chm_real) RIPSC,RIPSC2,RIPSCR,RIPSC2R,RIPSC6R,FIPS
    real(chm_real) WRK11,WRK21,WRK31,WRK12,WRK22,WRK32
    real(chm_real) WRK13,WRK23,WRK33,WRK14,WRK24,WRK34
    real(chm_real) XBIN,YBIN,ZBIN
    real(chm_real) EELIJ,ENBIJ
    real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ
    real(chm_real)  R1,R2,R2R
    real(chm_real) U1,U2,U4,U8,U6R,U12R
    real(chm_real) PE,DPE,PVC,DPVC,FIJ,DEIJ,DVIJ,PEC,PVCC
    integer rcskip,nfftdimrc
    rcskip=1
    nfftdimrc=nfft1+4
    ! Grid parameters
    XBIN=XTLABC(1)/NFFT1
    YBIN=XTLABC(3)/NFFT2
    ZBIN=XTLABC(6)/NFFT3
    MRCX=NFFT1/2
    MRCY=NFFT2/2
    MRCZ=NFFT3/2
    !  Cutoff grids
    loop300: DO IBINZ1=-MRCZ,MRCZ
       ZIJ=-(IBINZ1)*ZBIN
       IBINZ0=MOD(IBINZ1+100*NFFT3,NFFT3)+1
#if KEY_PARALLEL==1
       IF(IBINZ0 < KBOT .OR. IBINZ0  >  KTOP) cycle loop300    
#endif
       IBINZ0=(IBINZ0-KBOT)*NFFTDIMRC*NFFTDIM2
       loop400: DO IBINY1=-MRCY,MRCY
          YIJ=-(IBINY1)*YBIN
          IBINY0=IBINZ0+MOD(IBINY1+100*NFFT2,NFFT2)*NFFTDIMRC
          loop500: DO IBINX1=-MRCX,MRCX
             XIJ=-(IBINX1)*XBIN
             IBINX0=IBINY0+MOD(IBINX1+100*NFFT1,NFFT1)
             R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
             U2=R2*RIPS2R
             IF(U2 < ONE)THEN
                IF(R2 > RSMALL)THEN
                   R2R=ONE/R2
                ELSE
                   R2R=ZERO
                ENDIF
                !  Electrostatic IPS
                !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                ! 
                PE=-U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                     +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))+PIPSEC
                EELIJ=PE
                !  Lennard-Jones IPS
                U4=U2*U2
                U6R=ONE/U4/U2
                !  L-J r6 term
                !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                !
                PVC=-AIPSVC(0)-U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                     +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))+PIPSVCC
                !  L-J r12 term --neglected
                !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                !
                ENBIJ=-PVC
             ELSE 
                !  Electrostatic IPS
                !   etr1=1/r
                !   detr1/dr*r=-1/r
                ! 
                U1=SQRT(U2)
                PE=ONE/U1
                EELIJ=PE
                !  Lennard-Jones IPS
                U4=U2*U2
                U6R=ONE/U4/U2
                !  L-J r6 term
                !   etr6=1/r6
                !   detr6/dr*r1=-6/r6
                !
                PVC=U6R
                !  L-J r12 term --neglected
                !   etr12=1/r12
                !   detr12/dr*r1=-12/r12
                !
                ENBIJ=-PVC
             ENDIF
             EELIJ=EELIJ*RIPSR
             ENBIJ=ENBIJ*RIPS6R
             I000=IBINX0+1
             VDWARRAY(I000)=VDWARRAY(I000)+ENBIJ
             ELEARRAY(I000)=ELEARRAY(I000)+EELIJ
          enddo loop500
       enddo loop400
    enddo loop300
    !
    EIPSANB=ZERO
    EIPSAEL=ZERO
    RETURN
  END SUBROUTINE FIN_ENG_GRID

  !***********************************************************************
  !                 +----------------------------+
  !                 |        FIN_IPS_GRID        |
  !                 +----------------------------+
  !***********************************************************************
  SUBROUTINE FIN_IPS_GRID( &
       igood, kbot, ktop, &
       NATOM,CHARGE, &
       X,Y,Z,XTLABC, &
       NFFTDIM1,NFFTDIM2, &
       MY_KS,LATM, &
       NFFTDIM3)
    !
    ! This routine fills the charge grid, Q, and vdw grid, W.
    !
    !---------------------------------------------------------------------
    ! INPUT:
    !      numatoms:  number of atoms
    !      charge: the array of atomic charges
    !      wnb: the array of atomic vdw weight
    !      theta1,theta2,theta3: the spline coeff arrays
    !      fr1,fr2,fr3 the scaled and shifted fractional coords
    !      nfft1,nfft2,nfft3: the charge grid dimensions
    !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
    !      order: the order of spline interpolation
    ! OUTPUT:
    !      Q the charge grid
    !      W the vdw grid
    !---------------------------------------------------------------------
    !
    use pmeutil,only:nfft1,nfft2,nfft3,forder, &
         theta1,theta2,theta3, &
         dtheta1,dtheta2,dtheta3,fill_bspline &
#if KEY_PARALLEL==1
         ,mxystart,mxyslabs & 
#endif
         ;                    !   Ends continuation when not PARALLEL
    use number
#if KEY_PARALLEL==1
    use parallel
    use dimens_fcm
#endif 
    INTEGER ORDER,NATOM
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    real(chm_real) CHARGE(*)
    real(chm_real) X(*),Y(*),Z(*),XTLABC(6)
    !
    INTEGER N,NTOT,ITH1,ITH2,ITH3,I,J,K,KQ,IPT1,IPT2,IPT3,IPT
    real(chm_real) PRODA,PRODAA,PRODB,PRODBB

    INTEGER ENUMTASKS,ITASK,KDEL,KBOT0
    real(chm_real) fr1n,fr2n,fr3n,w
    integer igoody, igdt
    integer igood, kbot, ktop
    LOGICAL QFILL
    integer latm,my_ks(latm)
    integer rcskip,nfftdimrc
    rcskip=1
    nfftdimrc=nfft1+4
    !
    ORDER=FORDER
    igood=0
    igoody=0
#if KEY_PARALLEL==1
    KBOT0 = MXYSTART(MYNOD)
    KBOT = KBOT0 + 1
    KTOP = KBOT0 + MXYSLABS
#else /**/
    kbot0 = 0
    kbot = 1
    ktop = nfft3
#endif 
    !
    !------------------------------------------
    !          MFC NOTE: THERE COULD BE A PRE-FILTER HERE TO ELIMINATE
    !                      MOST OF THE ATOMS AND KEEP A LIST OF APPROPRIATE
    !                      ATOMS EITHER FOR EACH PE, EACH X-Y SLAB,
    !                      OR EACH Q ELEMENT
    !          MFC NOte Note: Looks like I am doing that filter now....
    !------------------------------------------
    loop110: DO N = 1,NATOM
       QFILL = .TRUE.
       w = Z(n)/XTLABC(6)
       fr3n = nfft3*(w - anint(w) + HALF)
       K = INT(FR3N) - ORDER + 1 + NFFT3
       IF(K > NFFT3) K=K-NFFT3
#if KEY_PARALLEL==1
       IF ( K+ORDER  <  KBOT .OR. K  >=  KTOP )cycle loop110
#endif 
       !
       IGOOD=IGOOD+1
       MY_KS(IGOOD)=N
       IGOODY=IGOOD
       W = X(N)/XTLABC(1)
       FR1N = NFFT1*(W - ANINT(W) + HALF)
       W = Y(N)/XTLABC(3)
       FR2N = NFFT2*(W - ANINT(W) + HALF)
       FR1(IGOOD)=FR1N
       FR2(IGOOD)=FR2N
       FR3(IGOOD)=FR3N
       IGDT=IGOOD
       W = FR1N-INT(FR1N)
       CALL FILL_BSPLINE(W,ORDER,THETA1(1,IGOOD),DTHETA1(1,IGDT))
       W = FR2N-INT(FR2N)
       CALL FILL_BSPLINE(W,ORDER,THETA2(1,IGOOD),DTHETA2(1,IGDT))
       W = FR3N-INT(FR3N)
       CALL FILL_BSPLINE(W,ORDER,THETA3(1,IGOOD),DTHETA3(1,IGDT))
       !
       DO ITH3 = 1,ORDER
          K=K+1
          IF(K > NFFT3) K=K-NFFT3
          KQ=K
#if KEY_PARALLEL==1
          IF ( K  >=  KBOT .AND. K  <=  KTOP )THEN
             KQ = K - KBOT0
#endif 
             PRODA = THETA3(ITH3,IGOOD)*CHARGE(N)
             PRODB = THETA3(ITH3,IGOOD)*WNB(N)
             !
             J = INT(FR2N) - ORDER + 1 + NFFT2
             IPT1 = (KQ-1)*NFFTDIM2 - 1
             !
             I = INT(FR1N) - ORDER + 1 + NFFT1
             IF(I >= NFFT1) I=I-NFFT1
             !
             DO ITH2 = 1,ORDER
                J=J+1
                IF(J > NFFT2) J=J-NFFT2
                PRODAA = THETA2(ITH2,IGOOD)*PRODA
                PRODBB = THETA2(ITH2,IGOOD)*PRODB
                IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                IPT3= IPT2 + rcskip*(NFFT1-I)
                !
                DO ITH1 = 1,ORDER
                   QARRAY(IPT2) = QARRAY(IPT2)+THETA1(ITH1,IGOOD)*PRODAA
                   WARRAY(IPT2) = WARRAY(IPT2)+THETA1(ITH1,IGOOD)*PRODBB
                   IPT2=IPT2+rcskip
                   IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                ENDDO
             ENDDO
#if KEY_PARALLEL==1
          ENDIF
#endif 
          !       (check to see if space overflow)
          if ( igood  > latm) &
               CALL WRNDIE(-5,'<IPS>' &
               ,'FIN_IPS_GRID igood  > LATM ')
       ENDDO
    enddo loop110
    !
    igood=igoody
    RETURN
  END SUBROUTINE FIN_IPS_GRID

  SUBROUTINE FIN_IPS_SUM(ENBAIPS,EELAIPS,LVDW,LELEC,CGF,VOLUME, &
       NFFTDIM1,NFFTDIM2,NFFTDIM3, &
       XTLABC)    
    ! Calculate grid IPS potentials
    use pmeutil,only:mxzslabs,mxzstart, &
         nfft1,nfft2,nfft3,bsp_mod1,bsp_mod2,bsp_mod3
    use number
    use dimens_fcm
    use consta
    use parallel
    use nbips
    !
    LOGICAL LVDW,LELEC
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    INTEGER MRCX2,MRCY2,MRCZ2
    real(chm_real) ENBAIPS,EELAIPS,CGF,VOLUME
    real(chm_real) XTLABC(6)
    real(chm_real) D1R,D1M,D2R,D2M
    INTEGER K,K2Q,K10,K20,K30,K0,K1,K2,K3,M1,M2,M3,IND,JND,INDTOP
    INTEGER NF1,NF2,NF3,K3Q
    INTEGER IPT1,IPT2,IPT3,IPT3I
    !
    real(chm_real) CFACT,CFACT0,CFACT1,CFACT2,CFACT3,ESTR,STRUQ,STRUW
    real(chm_real) QI,QJ,WI,WJ,PQI,PWI,PQJ,PWJ,PQIJ,PWIJ,EELIJ,ENBIJ
    INTEGER MC1,MC2,MC3
    !
    CFACT=HALF/(NFFT1*NFFT2*NFFT3)
    !
    NF1 = NFFT1/2
    IF(2*NF1 < NFFT1) NF1 = NF1+1
    NF2 = NFFT2/2
    IF(2*NF2 < NFFT2) NF2 = NF2+1
    NF3 = NFFT3/2
    IF(2*NF3 < NFFT3) NF3 = NF3+1
    !  Separate F(Q) and F(W) and put into convolution array.
    !  Accumlate total energies and virals
    EELAIPS=ZERO
    ENBAIPS=ZERO
    IPT1=1
    DO K2Q = 1, MXZSLABS
       K2=K2Q
#if KEY_PARALLEL==1
       IF(MYNOD > 0) K2 = K2Q + MXZSTART(MYNOD)     
#endif
       Mc2=MOD(NFFT2-K2+1,NFFT2)+1
       IPT2=IPT1
       CFACT1=BSP_MOD2(K2)
       DO K1 = 1, NF1+1
          IPT3=IPT2
          CFACT2=CFACT1*BSP_MOD1(K1)
          mc1=MOD(NFFT1-K1+1,NFFT1)+1
          IF(K1+K2 == 2) THEN
             K30=2
          ELSE
             K30=1
          ENDIF
          CFACT0=ONE+(NFFT1+K1)/(NFFT1+2)
          loop920: DO K3 = 1,NFFT3
             CFACT3=CFACT2*BSP_MOD3(K3)
             Mc3=MOD(NFFT3-K3+1,NFFT3)+1
             IF(mc1 == k1.and.mc2.eq.k2.and.mc3.eq.k3)THEN
                CFACT0=CFACT3
             ELSE
                CFACT0=TWO*CFACT3
             ENDIF
             QI=QARRAY(IPT3) 
             QJ=QARRAY(IPT3+1)
             PQI=ELEARRAY(IPT3) 
             PQJ=ELEARRAY(IPT3+1)
             QARRAY(IPT3) = CFACT3*(QI*PQI-QJ*PQJ)
             QARRAY(IPT3+1) = CFACT3*(QI*PQJ+QJ*PQI)
             WI=WARRAY(IPT3) 
             WJ=WARRAY(IPT3+1)
             PWI=VDWARRAY(IPT3) 
             PWJ=VDWARRAY(IPT3+1)
             WARRAY(IPT3) = CFACT3*(WI*PWI-WJ*PWJ)
             WARRAY(IPT3+1) =CFACT3*(WI*PWJ+WJ*PWI)
             IF(K1 == MC1.AND. &
                  (K2 > NFFT2/2+1.OR.(K2 == MC2.AND.K3.GT.NFFT3/2+1)) &
                  ) THEN
                IPT3=IPT3+2
                cycle loop920
             ENDIF
             IPT3=IPT3+2
             !
             STRUQ = CFACT0*CGF*(QI*QI + QJ*QJ)
             ESTR =  PQI * STRUQ
             EELAIPS=EELAIPS+ESTR
             !
             STRUW = CFACT0*(WI*WI + WJ*WJ)
             ESTR = PWI * STRUW
             ENBAIPS=ENBAIPS+ESTR
             !
          enddo loop920
          IPT2=IPT2+NFFT3*2
       ENDDO
       IPT1=IPT1+NFFT3*NFFTDIM1*2
    ENDDO
    !
    EELAIPS=CFACT*EELAIPS
    ENBAIPS=CFACT*ENBAIPS
    !      write(*,*)"AIPSSYS:",eelaips,enbaips
    !      write(*,*)"AIPSSYS:",(vir(k),k=1,6)
    !
    RETURN
  END SUBROUTINE FIN_IPS_SUM


  !***********************************************************************
  !                 +----------------------------+
  !                 |        FIN_IPS_GRAD            |
  !                 +----------------------------+
  !***********************************************************************
  SUBROUTINE FIN_IPS_GRAD( &
       igood, kbot, ktop, &
       NUMATOMS,CHARGE, &
       RECIP,X,Y,Z,FX,FY,FZ, &
       FR1,FR2,FR3, &
       NFFTDIM1,NFFTDIM2,NFFTDIM3, &
       my_ks,latm,XNSYMM,XTLABC)

    use pmeutil,only: mxystart,mxyslabs,nfft1,nfft2,nfft3,FORDER, &
         theta1,theta2,theta3,dtheta1,dtheta2,dtheta3
    use number
    use dimens_fcm
    use consta
    use parallel
    use nbips
    !
    !
    INTEGER NUMATOMS,ORDER
    INTEGER XNSYMM
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    !...##IF PARALLEL
    !      real(chm_real) Q(2,NFFT3,NFFTDIM1,MXZSLABS)
    !...##ELSE
    !      real(chm_real) Q(2,NFFTDIM1,NFFTDIM2,NFFTDIM3)
    !...##ENDIF
    real(chm_real) ENBAIPS,EELAIPS,RECIP(9),XTLABC(6)
    real(chm_real) FR1(*),FR2(*),FR3(*)
    real(chm_real) X(*),Y(*),Z(*),FX(*),FY(*),FZ(*)
    real(chm_real) CHARGE(*)
    integer latm,my_ks(latm)
    !
    integer igoo,ig
    real(chm_real) VAL0,VAL1,VAL2,VAL3,VAL0A,VAL1A,VAL2A,VAL3A
    INTEGER IPT1,IPT2,IPT3
    !
    INTEGER KQ
    integer igood, kbot, ktop
    !
    INTEGER N,ITH1,ITH2,ITH3,I,J,K,M1,M2
    real(chm_real) CGI,VDWI,QI,WI,QWI
    real(chm_real) EELI,ENBI,F1,F2,F3,TERM,CFACT,CFACT1,CFACT2
    real(chm_real) XI,YI,ZI,DXI,DYI,DZI
    integer rcskip,nfftdimrc
    !
    rcskip=1
    if(nfft1/2 /= (nfft1+1)/2)then
       CALL WRNDIE(-5,'<AIPS fill charge grid>' &
            ,'fftx dimension not even ')
    endif
    nfftdimrc=nfft1+4
    !
    ORDER=FORDER
    CFACT=ONE/(NFFT1*NFFT2*NFFT3)
    CFACT1=CFACT*CCELEC
    CFACT2=CFACT
    !
    ENBAIPS=ZERO
    EELAIPS=ZERO
    do ig = 1,igood
       n=my_ks(ig)
       if(xnsymm == 1)then
          igoo=ig
       else
          igoo=n
       endif
       CGI = CFACT1*CHARGE(N)
       VDWI = CFACT2*WNB(N)
       F1 = ZERO
       F2 = ZERO
       F3 = ZERO
       EELI = ZERO
       ENBI = ZERO
       K = INT(FR3(igoo)) - ORDER + 1 + NFFT3
       !
       DO ITH3 = 1,ORDER
          K=K+1
          IF(K > NFFT3) K=K-NFFT3
          KQ=K
#if KEY_PARALLEL==1
          IF ( K  >=  KBOT .AND. K  <=  KTOP )THEN
             KQ = K - MXYSTART(MYNOD)
#endif 
             VAL0A =  THETA3(ITH3,ig)
             VAL1A =  NFFT1 * THETA3(ITH3,ig)
             VAL2A =  NFFT2 * THETA3(ITH3,ig)
             VAL3A =  NFFT3 * DTHETA3(ITH3,igoo)
             !
             J = INT(FR2(igoo)) - ORDER + 1 + NFFT2
             IPT1=(KQ-1)*NFFTDIM2 -1
             !
             I = INT(FR1(igoo)) - ORDER + 1 + NFFT1
             IF(I >= NFFT1) I=I-NFFT1
             !
             DO ITH2 = 1,ORDER
                J=J+1
                IF(J > NFFT2) J=J-NFFT2
                !
                VAL0= VAL0A * THETA2(ITH2,ig)
                VAL1= VAL1A * THETA2(ITH2,ig)
                VAL2= VAL2A * DTHETA2(ITH2,igoo)
                VAL3= VAL3A * THETA2(ITH2,ig)

                IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                IPT3= IPT2 + rcskip*(NFFT1-I)
                !
                !
                DO ITH1 = 1,ORDER
                   !
                   !       force is negative of grad
                   QI=CGI*QARRAY(IPT2)
                   WI=VDWI*WARRAY(IPT2)
                   QWI=QI+WI
                   EELI = EELI + QI * VAL0 * THETA1(ITH1,ig)
                   ENBI = ENBI + WI * VAL0 * THETA1(ITH1,ig)
                   F1 = F1 - QWI * VAL1 * DTHETA1(ITH1,igoo)
                   F2 = F2 - QWI * VAL2 * THETA1(ITH1,ig)
                   F3 = F3 - QWI * VAL3 * THETA1(ITH1,ig)
                   !
                   !
                   IPT2=IPT2+rcskip
                   IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                ENDDO
             ENDDO
#if KEY_PARALLEL==1
          ENDIF
#endif 
       ENDDO
       !
       DXI =  - (RECIP(1)*F1+RECIP(4)*F2+RECIP(7)*F3)
       DYI =  - (RECIP(2)*F1+RECIP(5)*F2+RECIP(8)*F3)
       DZI =  - (RECIP(3)*F1+RECIP(6)*F2+RECIP(9)*F3)
       FX(N) = FX(N) +DXI
       FY(N) = FY(N) +DYI
       FZ(N) = FZ(N) +DZI
       EELAIPS=EELAIPS+EELI 
       ENBAIPS=ENBAIPS+ENBI 
       XI=X(N)
       YI=Y(N)
       ZI=Z(N)
       PIPSVIR(1)=PIPSVIR(1)-XI*DXI
       PIPSVIR(2)=PIPSVIR(2)-XI*DYI
       PIPSVIR(3)=PIPSVIR(3)-XI*DZI
       PIPSVIR(4)=PIPSVIR(4)-YI*DXI
       PIPSVIR(5)=PIPSVIR(5)-YI*DYI
       PIPSVIR(6)=PIPSVIR(6)-YI*DZI
       PIPSVIR(7)=PIPSVIR(7)-ZI*DXI
       PIPSVIR(8)=PIPSVIR(8)-ZI*DYI
       PIPSVIR(9)=PIPSVIR(9)-ZI*DZI
    ENDDO
    EELAIPS=HALF*EELAIPS 
    ENBAIPS=HALF*ENBAIPS
    RETURN
  END SUBROUTINE FIN_IPS_GRAD



  SUBROUTINE AIPSPBC(ENBAIPS,EELAIPS,IFRSTA,ILASTA,NATOM, &
       LEWALDX,LVDWX,LELECX,LVIPS,LEIPS, &
       EPS,CG,X,Y,Z,DX,DY,DZ)
    !-----------------------------------------------------------------------
    !    Calculate grid based anisotropic IPS interaction
    !
    use pmeutil,only:nfft1,nfft2,nfft3,sizfftab,sizffwrk,forder, &
         fft1_table,fft2_table,fft3_table,ffwork, &
         mxystart,mxyslabs,mxzslabs,get_fftdims &
         ,pll_fft_setup,allocate_bspline,deallocate_bspline &
         ,FFT3D0RC,GET_SC_FRACT
    use pme_module, only:fft_forwardrc,fft_backrc, &
         tmpy,alpha,beta
    use new_timer,only:timer_start,timer_stop,timer_stpstrt,  & 
         T_ipsafunc, T_ipsagrid,T_ipsasum,T_ipsaforce,T_ipsafft 
    use dimens_fcm
    use exfunc
    use number
    use consta
    use stream
    use energym
    use image
    use nbips
    use parallel
    use memory
    integer,allocatable,dimension(:) :: lmy_ks
    LOGICAL LVDW,LELEC,LVIPS,LEIPS,LVDWX,LELECX,LEWALDX
    INTEGER IFRSTA,ILASTA,NATOM
    real(chm_real) ENBAIPS,EELAIPS,EPS,CGF,CG(*)
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    real(chm_real)  XMIN,YMIN,ZMIN,XMAX,YMAX,ZMAX
    real(chm_real)  XTLINV(6),RECIP(3,3)
    INTEGER SIZ_Q
    INTEGER LATM
    !
    real(chm_real) scale,BL,BLC,VBOX
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3,NFFTABLE,NFFWORK
    INTEGER I,J,NN
    INTEGER IGOOD, KBOT, KTOP
    !
    LOGICAL OK,QMEM
    real(chm_real)  ATEMP(5)                  
    !
    LVDW=LVDWX.AND.LVIPS
    LELEC=LELECX.AND.LEIPS.AND..NOT.LEWALDX
    CGF=CCELEC/EPS
    !
    NATTOT=NATOM*XNSYMM
    CALL INVT33S(XTLINV,XTLABC,OK)
    !
    RECIP(1,1) = XTLINV(1)
    RECIP(2,2) = XTLINV(3)
    RECIP(3,3) = XTLINV(6)
    RECIP(1,2) = XTLINV(2)
    RECIP(2,1) = XTLINV(2)
    RECIP(1,3) = XTLINV(4)
    RECIP(3,1) = XTLINV(4)
    RECIP(2,3) = XTLINV(5)
    RECIP(3,2) = XTLINV(5)
    QMEM=.FALSE.
    IF(.NOT.LEWALDX)THEN
       CALL IPSGRID(QIPSFIN,MIPSX,MIPSO,GIPSX,XTLABC(1),NN,BLC)
       QMEM=QMEM.OR.(NFFT1 /= NN)
       IF(QMEM)NFFT1=NN
       CALL IPSGRID(QIPSFIN,MIPSY,MIPSO,GIPSY,XTLABC(3),NN,BLC)
       QMEM=QMEM.OR.(NFFT2 /= NN)
       IF(QMEM)NFFT2=NN
       CALL IPSGRID(QIPSFIN,MIPSZ,MIPSO,GIPSZ,XTLABC(6),NN,BLC)
       QMEM=QMEM.OR.(NFFT3 /= NN)
       IF(QMEM)NFFT3=NN
       FORDER=MIPSO
    ENDIF
    !
    VBOX=EPROP(VOLUME)
    !
    IF(ABS(ONE-VBIPS/VBOX) > DVBIPS)THEN
       ! Do update if volume change is large and QIPSUPD=mod(nconf,nipsfrq)==0
       VBIPS=VBOX
    ELSE
       ! No update if volume change is small
       QIPSUPD=.FALSE.
    ENDIF
    !
    !-------------------------------------------------------------------
    ! INPUT
    !      NATOM is number of atoms
    !      FORDER is the order of B-spline interpolation
    !      x,y,z:   atomic coords
    !      CG  atomic charges
    !      XTLINV=recip: array of reciprocal unit cell vectors
    !      VOLUME: the volume of the unit cell
    !      KAPPA=ewald_coeff:   ewald convergence parameter
    !      NFFT1,NFFT2,NFFT3: the dimensions of the charge grid array
    ! OUTPUT
    !      siz_Q=3d charge grid array
    !      dx,dy,dz: forces incremented by k-space sum
    !
    !   All pointers are the integer names of the real(chm_real) variables that
    !   will be filled
    !
    !   Get memory for scratch arrays, free them after summation.
    !
    CALL GET_FFTDIMS( &
         NFFTDIM1,NFFTDIM2,NFFTDIM3,NFFTABLE,NFFWORK)
    SCALE = ONE
    !
    IF(QMEM) &
         call pll_FFT_setup(nfftdim1,nfftdim2,nfftdim3)
    SIZ_Q = max(2*NFFTDIM1*NFFTDIM2*mxyslabs, &
         2*NFFTDIM1*NFFTDIM3*mxzslabs)
    !
    allocate(qarray(siz_q),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray"
    allocate(warray(siz_q),stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray"
    allocate(fr1(nattot),fr2(nattot),fr3(nattot),cg1(nattot), &
         stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to allocate fr arrays"
    call allocate_bspline(natom,nattot)
    allocate(tmpy(2*nfftdim1),alpha(nfft1),beta(nfft1))
    if(alloc_err /= 0 ) &
         write(0,*)"unable to allocate tmpy,alpha,beta"

    !  Change grid size to for FFT
    IF(QMEM)THEN
       !    Initiate FFT setting
       CALL AIPS_SETUP(NATOM,XNSYMM)
    ENDIF
    !
    call FFT3D0rc(0,scale,qarray, &
         nfftdim1,nfftdim2, &
         tmpy,alpha,beta)
    IF(QMEM.OR.QIPSUPD)THEN
       call timer_start(T_ipsafunc)     
       IF(allocated(elearray))THEN
          deallocate(elearray,vdwarray,stat=alloc_err)
          if(alloc_err /= 0 ) write(0,*)"unable to deallocate elearray"
          deallocate(elexx,elexy,elexz,eleyy,eleyz,elezz,stat=alloc_err)
          if(alloc_err /= 0 ) write(0,*)"unable to deallocate elexx"
          deallocate(vdwxx,vdwxy,vdwxz,vdwyy,vdwyz,vdwzz,stat=alloc_err)
          if(alloc_err /= 0 ) write(0,*)"unable to deallocate vdwxx"
       ENDIF
       allocate(elearray(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate elearray"
       allocate(elexx(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate elexx"
       allocate(elexy(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate elexy"
       allocate(elexz(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate elexz"
       allocate(eleyy(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate eleyy"
       allocate(eleyz(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate eleyz"
       allocate(elezz(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate elezz"
       allocate(vdwarray(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate vdwarray"
       allocate(vdwxx(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate vdwxx"
       allocate(vdwxy(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate vdwxy"
       allocate(vdwxz(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate vdwxz"
       allocate(vdwyy(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate vdwyy"
       allocate(vdwyz(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate vdwyz"
       allocate(vdwzz(siz_q),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate vdwzz"
       !  end of memory allocation
       IPSSIZ=SIZ_Q
       ! Update energy function grid. stored at the image part of Q and W
       elearray(1:SIZ_Q)=zero
       vdwarray(1:SIZ_Q)=zero
       elexx(1:SIZ_Q)=zero
       elexy(1:SIZ_Q)=zero
       elexz(1:SIZ_Q)=zero
       eleyy(1:SIZ_Q)=zero
       eleyz(1:SIZ_Q)=zero
       elezz(1:SIZ_Q)=zero
       vdwxx(1:SIZ_Q)=zero
       vdwxy(1:SIZ_Q)=zero
       vdwxz(1:SIZ_Q)=zero
       vdwyy(1:SIZ_Q)=zero
       vdwyz(1:SIZ_Q)=zero
       vdwzz(1:SIZ_Q)=zero
#if KEY_PARALLEL==1
       KBOT = MXYSTART(MYNOD) + 1
       KTOP = MXYSTART(MYNOD) + MXYSLABS
#else /**/
       kbot = 1
       ktop = nfft3
#endif 
       IF(QIPS2D)THEN
          CALL PBC_IPS2D_ENG(LVDW,LELEC,kbot, ktop, &
               NFFTDIM1,NFFTDIM2,NFFTDIM3, &
               XTLABC,EPROP(VOLUME),CGF)
       ELSE
          CALL PBC_IPS_ENG(LVDW,LELEC,kbot, ktop, &
               NFFTDIM1,NFFTDIM2,NFFTDIM3, &
               XTLABC,EPROP(VOLUME),CGF)
       ENDIF
       call timer_stpstrt(T_ipsafunc,T_ipsafft)     
       IF(LELEC)THEN
          call fft_backrc( &
               elearray, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               elexx, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               elexy, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               elexz, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               eleyy, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               eleyz, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               elezz, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
       ENDIF
       IF(LVDW)THEN
          call fft_backrc( &
               vdwarray, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               vdwxx, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               vdwxy, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               vdwxz, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               vdwyy, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               vdwyz, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
          call fft_backrc( &
               vdwzz, &
               nfftdim1,nfftdim2,nfftdim3,nffwork)
       ENDIF
       call timer_stop(T_ipsafft)     
    ENDIF
    !
    call timer_start(T_ipsagrid)     
    if(XNSYMM > 1) &
         CALL GET_SC_FRACT(FR1,FR2,FR3,CG1, &
         NATOM,NATTOT,X,Y,Z,RECIP, &
         XNSYMM,MAXSYM,XSYMOP,CG)
    !       make array for keeping track of atoms important to
    !         this processor
    latm=nattot
    call chmalloc('enbips.src','AIPSPBC','lmy_ks',latm,intg=lmy_ks)
    !-------- symmetrical case ------------------------------
    !        fill frac coords and thetas in fill_ch_grid
    !              use min image charge array: cg
    qarray(1:SIZ_Q)=zero
    warray(1:SIZ_Q)=zero                          
    if(xnsymm == 1) then
       CALL PBC_IPS_GRID( &
            igood, kbot, ktop, &
            NATTOT,CG, &
            X,Y,Z,recip, &
            NATOM,xnsymm, &
            FR1,FR2,FR3, &
            NFFTDIM1,NFFTDIM2, &
            LMY_KS,LATM, &
#if KEY_PARALLEL==1
            mxyslabs)          
#endif
#if KEY_PARALLEL==0
       NFFTDIM3)          
#endif
    ELSE
       !-------- asymmetrical case ------------------------------
       !        fill frac coords  in GET_SC_FRACT but thetas in fill_ch_grid
       !           use the whole unit cell charge array: cg1
       CALL PBC_IPS_GRID( &
            igood, kbot, ktop, &
            NATTOT,CG1, &
            X,Y,Z,recip, &
            NATOM,xnsymm, &
            FR1,FR2,FR3, &
            NFFTDIM1,NFFTDIM2, &
            LMY_KS,LATM, &
#if KEY_PARALLEL==1
            mxyslabs)         
#endif
#if KEY_PARALLEL==0
       NFFTDIM3)         
#endif
    endif
    call timer_stpstrt(T_ipsagrid,T_ipsafft)     
    IF(LELEC)THEN
       call fft_backrc( &
            qarray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
    ELSE
       qarray(1:SIZ_Q)=zero
    ENDIF
    IF(LVDW)THEN
       call fft_backrc( &
            warray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
    ELSE
       warray(1:SIZ_Q)=zero
    ENDIF
    ! Calculate grid IPS potentials
    call timer_stpstrt(T_ipsafft,T_ipsasum)     
    CALL PBC_IPS_SUM(ENBAIPS,EELAIPS,LVDW,LELEC,CGF,EPROP(VOLUME), &
         NFFTDIM1,NFFTDIM2,NFFTDIM3, &
         XTLABC)
    !  Forward FFT to get potential surface
    !
    call timer_stpstrt(T_ipsasum,T_ipsafft)     
    IF(LELEC)THEN
       call fft_forwardrc( &
            qarray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
    ENDIF
    IF(LVDW)THEN
       call fft_forwardrc( &
            warray, &
            nfftdim1,nfftdim2,nfftdim3,nffwork)
    ENDIF
    !  Calculate forces on atoms by B-spline interpolation
    call timer_stpstrt(T_ipsafft,T_ipsaforce)     
    CALL PBC_IPS_GRAD( &
         IGOOD, KBOT, KTOP, &
         NATOM,CG,RECIP, &
         DX,DY,DZ, &
         FR1,FR2,FR3, &
         NFFTDIM1,NFFTDIM2,NFFTDIM3, &
         LMY_KS,LATM, &
         XNSYMM,XTLABC)
    call timer_stop(T_ipsaforce)     
    !
    !=======================================================================
    !   Main loop end
    !=======================================================================
    !
    call chmdealloc('enbips.src','AIPSPBC','LMY_KS',LATM,intg=LMY_KS)
    !
    !
    deallocate(tmpy,alpha,beta,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*) &
         "unable to deallocate tmpy,alpha,beta"
    deallocate(fr1,fr2,fr3,cg1,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate fr1,2,3"
    deallocate(qarray,warray,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray"
    call deallocate_bspline()
    !  Print out the long rang corretions parameters
    !write(outu,'("aipspbc:",6I6,10F10.2)')IFRSTA,ILASTA,natom,nfftdim1,nfftdim2,nfftdim3,enbaips,eelaips
    ATEMP(1)=ENBAIPS
    ATEMP(2)=EELAIPS
    ATEMP(3)=PIPSVIR(1)
    ATEMP(4)=PIPSVIR(5)
    ATEMP(5)=PIPSVIR(9)
#if KEY_PARALLEL==1 /*EIPSS*/
    IF(NUMNOD > 1)THEN
       CALL GCOMB(ATEMP,5)
    ENDIF
#endif /* (EIPSS)*/
    IF(PRNLEV >= 6)THEN
#if KEY_PARALLEL==1
       IF(MYNOD == 0)THEN                               
#endif
          WRITE(OUTU,'("  ENBGRID,EELGRID  = ",6E14.7)') &
               (ATEMP(I),I=1,2)
          WRITE(OUTU,'("  PIPSXX  = ",3E14.7)') &
               ATEMP(3),ATEMP(4),ATEMP(5)
#if KEY_PARALLEL==1
       ENDIF                               
#endif
    ENDIF
    RETURN 
  END SUBROUTINE AIPSPBC

  !****************************************************************
  !                        IPSGRID
  !****************************************************************
  SUBROUTINE IPSGRID(QFIN,MIPS,MORDER,GIPS,BOXL,NGRID,BIPS)
    use number
    use parallel
    LOGICAL  QFIN
    real(chm_real)  GIPS,BOXL,BIPS
    INTEGER  MIPS,MORDER,NGRID
    INTEGER  M,N,NN
    INTEGER, PARAMETER :: MMAX=1000
    !
    IF(MIPS > 0)THEN
       IF(QFIN)THEN 
          M=2*(MIPS+MORDER)
       ELSE
          M=2*(MIPS/2)
          IF(M < MORDER)M=MORDER
       ENDIF
    ELSE
       IF(QFIN)THEN 
          M=2*(INT(BOXL/GIPS)+1+MORDER)
       ELSE
          M=2*((INT(BOXL/GIPS)+1)/2)
          IF(M < MORDER)M=MORDER
       ENDIF
    ENDIF
#if KEY_PARALLEL==1
    IF(M < NUMNOD)M=NUMNOD
#endif 
10  N=M
20  IF(N == 2*(N/2))THEN
       N=N/2
       GOTO 20
    ENDIF
30  IF(N == 3*(N/3))THEN
       N=N/3
       GOTO 30
    ENDIF
50  IF(N == 5*(N/5))THEN
       N=N/5
       GOTO 50
    ENDIF
    !      NN=(N-(N/7)*7)*(N-(N/11)*11)*(N-(N/13)*13)*(N-(N/17)*17)*
    !     &   (N-(N/23)*23)*(N-(N/29)*29)*(N-(N/31)*31)*(N-(N/37)*37)*
    !     &   (N-(N/41)*41)*(N-(N/43)*43)*(N-(N/47)*47)*(N-(N/51)*51)
    !      IF(NN == 0.or.n > 51)THEN
    IF(N > 1)THEN
       M=M+2
       GOTO 10
    ENDIF
    IF(M > MMAX)CALL WRNDIE(-5,'<IPSGRID>' &
         ,'Acceptable grid size is too big! ')
    NGRID=M
    IF(QFIN)THEN 
       IF(MIPS > 0)THEN 
          BIPS=NGRID*(AINT(BOXL/GIPS)+ONE)*GIPS/MIPS
       ELSE
          BIPS=NGRID*GIPS
       ENDIF
    ELSE
       BIPS=BOXL
    ENDIF
    RETURN
  END SUBROUTINE IPSGRID


  !****************************************************************
  !                        AIPS_SETUP
  !****************************************************************
  SUBROUTINE AIPS_SETUP(NATOM,XNSYMM)
    !-----------------------------------------------------------------------
    !     This routine allocates space and defines variables for
    !     the Particle Mesh Ewald Summation
    !     original code by Tom Darden, implemented into CHARMM
    !     by Scott Feller and Bernie Brooks, NIH, Jan-1996
    !     Parallel 3D fft routines from Michael Crowley, Pittsburgh
    !     Supercomputing Center
    !-----------------------------------------------------------------------
    !
    use pmeutil,only:nfft1,nfft2,nfft3,sizfftab,sizffwrk,forder, &
         fft1_table,fft2_table,fft3_table,ffwork, &
         bsp_mod1,bsp_mod2,bsp_mod3, &
         mxyslabs,mxzslabs,get_fftdims &
         ,pll_fft_setup,load_bsp_mod
!!$#if KEY_COLFFT==1
!!$    use colfft,only:colfft_init,colfft_uninit 
!!$#endif
    use exfunc 
    use stream
#if KEY_DOMDEC==1
    use domdec_dr_common,only:nrecip
    use domdec_r2d_comm,only:n_grid_atom
    use domdec_common,only:q_domdec
#endif 
#if KEY_PARALLEL==1
    use parallel  
#endif
    !
    INTEGER NATOM,XNSYMM
    !
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3,nfftable,NFFWORK
    INTEGER SIZ_Q,SIZHEAP,SIZTHETA,SIZDTHETA

    INTEGER,PARAMETER :: MAXORDER=25, MAXNFFT=2000

    CALL GET_FFTDIMS(NFFTDIM1,NFFTDIM2,NFFTDIM3,NFFTABLE,NFFWORK)

    !          tripled for 3 FFTs
    SIZFFTAB = 3*NFFTABLE
    !          doubled for complex
    SIZFFWRK = 2*NFFWORK

    SIZTHETA  = NATOM*XNSYMM*FORDER
    SIZDTHETA = NATOM*FORDER
#if KEY_PARALLEL==1
    SIZ_Q = max(2*NFFTDIM1*NFFTDIM2*mxyslabs,  &
         2*NFFTDIM1*NFFTDIM3*mxzslabs)
#else /**/
    SIZ_Q = 2*NFFTDIM1*NFFTDIM2*NFFTDIM3
#endif 
    !
    SIZHEAP = NFFT1+NFFT2+NFFT3+SIZFFTAB + &
         SIZ_Q+3*SIZTHETA+3*SIZDTHETA+SIZFFWRK+4*NATOM*XNSYMM
    !
    !     allocate long-term space
    if(allocated(fft1_table)) &
         deallocate(fft1_table,fft2_table,fft3_table,ffwork)
    allocate(fft1_table(4*nfftdim1), fft2_table(4*nfftdim2), &
         fft3_table(4*nfftdim3),ffwork(sizffwrk))

#if KEY_COLFFT==1 /*colfft*/
!!$    call colfft_uninit()
!!$#if KEY_DOMDEC==1
!!$    if (q_domdec) then
!!$       call colfft_init(nrecip,n_grid_atom)
!!$    else
!!$#endif 
!!$#if KEY_PARALLEL==1
!!$       call colfft_init(numnod,natom)  
!!$#endif
!!$#if KEY_PARALLEL==0
!!$       call colfft_init(1,natom)   
!!$#endif
!!$#if KEY_DOMDEC==1
!!$    endif  
!!$#endif
#else /* (colfft)*/
    call pll_fft_setup(nfftdim1,nfftdim2,nfftdim3)
#endif /* (colfft)*/
    !
    !
    if(allocated(bsp_mod1)) &
         deallocate(bsp_mod1,bsp_mod2,bsp_mod3)
    allocate(bsp_mod1(nfft1),bsp_mod2(nfft2),bsp_mod3(nfft3))
    !
    CALL LOAD_BSP_MOD
    !  Self AIPS interaction

    RETURN
  END subroutine aips_setup


  !***********************************************************************
  !                 +----------------------------+
  !                 |        FILL_IPS_GRID        |
  !                 +----------------------------+
  !***********************************************************************
  SUBROUTINE PBC_IPS_GRID( &
       igood, kbot, ktop, &
       NUMATOMS,CHARGE, &
       X,Y,Z,recip,NATOM,xnsymm, &
       FR1,FR2,FR3, &
       NFFTDIM1,NFFTDIM2, &
       MY_KS,LATM, &
       NFFTDIM3)
    !
    ! This routine fills the charge grid, Q, and vdw grid, W.
    !
    !---------------------------------------------------------------------
    ! INPUT:
    !      numatoms:  number of atoms
    !      charge: the array of atomic charges
    !      wnb: the array of atomic vdw weight
    !      theta1,theta2,theta3: the spline coeff arrays
    !      fr1,fr2,fr3 the scaled and shifted fractional coords
    !      nfft1,nfft2,nfft3: the charge grid dimensions
    !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
    !      order: the order of spline interpolation
    !---------------------------------------------------------------------
    !
    use pmeutil,only:nfft1,nfft2,nfft3,forder, &
         theta1,theta2,theta3, &
         dtheta1,dtheta2,dtheta3,fill_bspline &
#if KEY_PARALLEL==1
         ,mxystart,mxyslabs & 
#endif
         ;                    !    Ends continuation when not PARALLEL
    use number
#if KEY_PARALLEL==1
    use parallel
    use dimens_fcm
#endif 
    INTEGER NUMATOMS,ORDER,NATOM
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    real(chm_real) FR1(NUMATOMS),FR2(NUMATOMS),FR3(NUMATOMS)
    real(chm_real) CHARGE(NUMATOMS)
    real(chm_real) X(*),Y(*),Z(*), RECIP(9)
    !
    INTEGER N,NTOT,ITH1,ITH2,ITH3,I,J,K,KQ,IPT1,IPT2,IPT3,IPT
    real(chm_real) PRODA,PRODAA,PRODB,PRODBB

    INTEGER ENUMTASKS,ITASK,KDEL,KBOT0
    real(chm_real) fr1n,fr2n,fr3n,w
    integer xnsymm,igoody, igdt
    integer igood, kbot, ktop
    LOGICAL QFILL
    integer latm,my_ks(latm)
    integer rcskip,nfftdimrc
    rcskip=1
    if(nfft1/2 /= (nfft1+1)/2)then
       CALL WRNDIE(-5,'<IPSPBC fill charge grid>' &
            ,'fftx dimension not even ')
    endif
    nfftdimrc=nfft1+4
    !
    order=forder
    igood=0
    igoody=0
#if KEY_PARALLEL==1
    ENUMTASKS = 1
    KDEL = NFFT3/ENUMTASKS
    IF ( KDEL  ==  0 )KDEL = 1
    !
    KBOT0 = MXYSTART(MYNOD)
    KBOT = KBOT0 + 1
    KTOP = KBOT0 + MXYSLABS
#else /**/
    kbot0 = 0
    kbot = 1
    ktop = nfft3
#endif 
    !
    !
    !------------------------------------------
    !          MFC NOTE: THERE COULD BE A PRE-FILTER HERE TO ELIMINATE
    !                      MOST OF THE ATOMS AND KEEP A LIST OF APPROPRIATE
    !                      ATOMS EITHER FOR EACH PE, EACH X-Y SLAB,
    !                      OR EACH Q ELEMENT
    !          MFC NOte Note: Looks like I am doing that filter now....
    !------------------------------------------
    DO N = 1,NUMATOMS
       QFILL = .TRUE.
       if(XNSYMM > 1)then
          fr3n=fr3(n)
       else
          w = X(n)*recip(7)+Y(n)*recip(8)+Z(n)*recip(9)
          fr3n = nfft3*(w - anint(w) + HALF)
       endif
       K = INT(FR3N) - ORDER + 1 + NFFT3
       !
       DO ITH3 = 1,ORDER
          K=K+1
          IF(K > NFFT3) K=K-NFFT3
          KQ=K
#if KEY_PARALLEL==1
          IF ( K  >=  KBOT .AND. K  <=  KTOP )THEN
             KQ = K - KBOT0
#endif 
             IF(QFILL) THEN
                QFILL = .FALSE.
                IGOOD=IGOOD+1
                MY_KS(IGOOD)=N
                IF(N <= NATOM)IGOODY=IGOOD
                IF(XNSYMM == 1)THEN
                   W = X(N)*RECIP(1)+Y(N)*RECIP(2)+Z(N)*RECIP(3)
                   FR1N = NFFT1*(W - ANINT(W) + HALF)
                   W = X(N)*RECIP(4)+Y(N)*RECIP(5)+Z(N)*RECIP(6)
                   FR2N = NFFT2*(W - ANINT(W) + HALF)
                   FR1(IGOOD)=FR1N
                   FR2(IGOOD)=FR2N
                   FR3(IGOOD)=FR3N
                   IGDT=IGOOD
                ELSE
                   FR1N=FR1(N)
                   FR2N=FR2(N)
                   IGDT=MIN(N,NATOM+1)
                ENDIF
                W = FR1N-INT(FR1N)
                CALL FILL_BSPLINE(W,ORDER,THETA1(1,IGOOD),DTHETA1(1,IGDT))
                W = FR2N-INT(FR2N)
                CALL FILL_BSPLINE(W,ORDER,THETA2(1,IGOOD),DTHETA2(1,IGDT))
                W = FR3N-INT(FR3N)
                CALL FILL_BSPLINE(W,ORDER,THETA3(1,IGOOD),DTHETA3(1,IGDT))
             ENDIF
             PRODA = THETA3(ITH3,IGOOD)*CHARGE(N)
             PRODB = THETA3(ITH3,IGOOD)*WNB(N)
             !
             J = INT(FR2N) - ORDER + 1 + NFFT2
             IPT1 = (KQ-1)*NFFTDIM2 - 1
             !
             I = INT(FR1N) - ORDER + 1 + NFFT1
             IF(I >= NFFT1) I=I-NFFT1
             !
             DO ITH2 = 1,ORDER
                J=J+1
                IF(J > NFFT2) J=J-NFFT2
                PRODAA = THETA2(ITH2,IGOOD)*PRODA
                PRODBB = THETA2(ITH2,IGOOD)*PRODB
                IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                IPT3= IPT2 + rcskip*(NFFT1-I)
                !
                DO ITH1 = 1,ORDER
                   QARRAY(IPT2) = QARRAY(IPT2)+THETA1(ITH1,IGOOD)*PRODAA
                   WARRAY(IPT2) = WARRAY(IPT2)+THETA1(ITH1,IGOOD)*PRODBB
                   IPT2=IPT2+rcskip
                   IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                ENDDO
             ENDDO
#if KEY_PARALLEL==1
          ENDIF
#endif 
          !       (check to see if space overflow)
          if ( igood  > latm) &
               CALL WRNDIE(-5,'<AIPS>' &
               ,'FILL_IPS_GRID igood  > LATM ')
       ENDDO
    ENDDO
    !
    igood=igoody
    RETURN
  END SUBROUTINE PBC_IPS_GRID

  SUBROUTINE PBC_IPS_SUM(ENBAIPS,EELAIPS,LVDW,LELEC,CGF,VOLUME, &
       NFFTDIM1,NFFTDIM2,NFFTDIM3, &
       XTLABC)    
    ! Calculate grid IPS potentials
    use pmeutil,only:mxzslabs,mxzstart, &
         nfft1,nfft2,nfft3,bsp_mod1,bsp_mod2,bsp_mod3
    use number
    use dimens_fcm
    !
    use consta
    use parallel
    use nbips
    LOGICAL LVDW,LELEC
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    INTEGER MRCX2,MRCY2,MRCZ2
    real(chm_real) ENBAIPS,EELAIPS,CGF,VOLUME
    real(chm_real) XTLABC(6)
    real(chm_real) D1R,D1M,D2R,D2M
    INTEGER K,K2Q,K10,K20,K30,K0,K1,K2,K3,M1,M2,M3,IND,JND,INDTOP
    INTEGER NF1,NF2,NF3,K3Q
    INTEGER IPT1,IPT2,IPT3,IPT3I
    !
    INTEGER IBINX,IBINY,IBINZ,IBINX0,IBINY0,IBINZ0
    INTEGER ISYMX,ISYMY,ISYMZ
    INTEGER IBINX1,IBINY1,IBINZ1,IXYZ0,IXYZ1
    INTEGER IX,IY,IZ
    real(chm_real) WRK11,WRK21,WRK31,WRK12,WRK22,WRK32
    real(chm_real) WRK13,WRK23,WRK33,WRK14,WRK24,WRK34
    real(chm_real) XBIN,YBIN,ZBIN
    real(chm_real) CFACT,CFACT0,CFACT1,CFACT2,CFACT3,CFACT4, &
         CTENS1,CTENS2
    real(chm_real) ESTR,STRUQ,STRUW
    real(chm_real) QI,QJ,WI,WJ,PQI,PWI,PQJ,PWJ,PQIJ,PWIJ,EELIJ,ENBIJ
    real(chm_real) QXX,QXY,QXZ,QYY,QYZ,QZZ,WXX,WXY,WXZ,WYY,WYZ,WZZ
    real(chm_real) VIR(6)
    INTEGER MC1,MC2,MC3
    !
    CFACT=HALF/(NFFT1*NFFT2*NFFT3)
    !
    CTENS1=CTENSOR
    CTENS2=ONE-CTENS1
    !
    NF1 = NFFT1/2
    IF(2*NF1 < NFFT1) NF1 = NF1+1
    NF2 = NFFT2/2
    IF(2*NF2 < NFFT2) NF2 = NF2+1
    NF3 = NFFT3/2
    IF(2*NF3 < NFFT3) NF3 = NF3+1
    !  Separate F(Q) and F(W) and put into convolution array.
    !  Accumlate total energies and virals
    EELAIPS=ZERO
    ENBAIPS=ZERO
    DO K = 1,6
       VIR(K) = ZERO
    ENDDO
    !
    IPT1=1
    DO K2Q = 1, MXZSLABS
       K2=K2Q
#if KEY_PARALLEL==1
       IF(MYNOD > 0) K2 = K2Q + MXZSTART(MYNOD)      
#endif
       Mc2=MOD(NFFT2-K2+1,NFFT2)+1
       IPT2=IPT1
       CFACT1=BSP_MOD2(K2)
       DO K1 = 1, NF1+1
          IPT3=IPT2
          CFACT2=CFACT1*BSP_MOD1(K1)
          mc1=MOD(NFFT1-K1+1,NFFT1)+1
          loop910: DO K3 = 1,NFFT3
             CFACT3=CFACT2*BSP_MOD3(K3)
             Mc3=MOD(NFFT3-K3+1,NFFT3)+1
             !                  IF(mc1 == k1.and.mc2.eq.k2.and.mc3.eq.k3)THEN
             CFACT0=CTENS1+CTENS2*CFACT3
             CFACT4=CTENS2+CTENS1*CFACT3
             IF(mc1 /= k1)THEN
                CFACT0=TWO*CFACT0
             ENDIF
             QI=QARRAY(IPT3) 
             QJ=QARRAY(IPT3+1)
             PQI=ELEARRAY(IPT3) 
             PQJ=ELEARRAY(IPT3+1)
             QARRAY(IPT3) = CFACT3*(QI*PQI-QJ*PQJ)
             QARRAY(IPT3+1) = CFACT3*(QI*PQJ+QJ*PQI)
             WI=WARRAY(IPT3) 
             WJ=WARRAY(IPT3+1)
             PWI=VDWARRAY(IPT3) 
             PWJ=VDWARRAY(IPT3+1)
             WARRAY(IPT3) = CFACT3*(WI*PWI-WJ*PWJ)
             WARRAY(IPT3+1) =CFACT3*(WI*PWJ+WJ*PWI)
             QXX = ELEXX(IPT3)
             WXX = VDWXX(IPT3)
             QXY = ELEXY(IPT3)
             WXY = VDWXY(IPT3)
             QXZ = ELEXZ(IPT3)
             WXZ = VDWXZ(IPT3)
             QYY = ELEYY(IPT3)
             WYY = VDWYY(IPT3)
             QYZ = ELEYZ(IPT3)
             WYZ = VDWYZ(IPT3)
             QZZ = ELEZZ(IPT3)
             WZZ = VDWZZ(IPT3)
             IPT3=IPT3+2
             !
             STRUQ = CFACT0*CGF*(QI*QI + QJ*QJ)
             ESTR =  CFACT4* PQI * STRUQ
             EELAIPS=EELAIPS+ESTR
             !
             STRUW = CFACT0*(WI*WI + WJ*WJ)
             ESTR = CFACT4* PWI * STRUW
             ENBAIPS=ENBAIPS+ESTR
             !
             VIR(1)=VIR(1)+QXX*STRUQ+WXX*STRUW
             VIR(2)=VIR(2)+QXY*STRUQ+WXY*STRUW
             VIR(3)=VIR(3)+QXZ*STRUQ+WXZ*STRUW
             VIR(4)=VIR(4)+QYY*STRUQ+WYY*STRUW
             VIR(5)=VIR(5)+QYZ*STRUQ+WYZ*STRUW
             VIR(6)=VIR(6)+QZZ*STRUQ+WZZ*STRUW
             !
          enddo loop910
          IPT2=IPT2+NFFT3*2
       ENDDO
       IPT1=IPT1+NFFT3*NFFTDIM1*2
    ENDDO
    !
    EELAIPS=CFACT*EELAIPS
    ENBAIPS=CFACT*ENBAIPS
    DO K = 1,6
       VIR(K) =CFACT*VIR(K)
    ENDDO
    PIPSVIR(1) = PIPSVIR(1) - VIR(1)
    PIPSVIR(2) = PIPSVIR(2) - VIR(2)
    PIPSVIR(3) = PIPSVIR(3) - VIR(3)
    PIPSVIR(4) = PIPSVIR(4) - VIR(2)
    PIPSVIR(5) = PIPSVIR(5) - VIR(4)
    PIPSVIR(6) = PIPSVIR(6) - VIR(5)
    PIPSVIR(7) = PIPSVIR(7) - VIR(3)
    PIPSVIR(8) = PIPSVIR(8) - VIR(5)
    PIPSVIR(9) = PIPSVIR(9) - VIR(6)
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN                             
#endif
       EELAIPS=EELAIPS+EIPSAEL
       ENBAIPS=ENBAIPS+EIPSANB
       PIPSVIR(1) = PIPSVIR(1)+VIRAIPS
       PIPSVIR(5) = PIPSVIR(5)+VIRAIPS
       PIPSVIR(9) = PIPSVIR(9)+VIRAIPS
#if KEY_PARALLEL==1
    ENDIF                                          
#endif
    RETURN
  END SUBROUTINE PBC_IPS_SUM


  SUBROUTINE PBC_IPS_ENG(LVDW,LELEC,KBOT,KTOP, &
       NFFTDIM1,NFFTDIM2,NFFTDIM3, &
       XTLABC,VBOX,CGF)    
    ! Calculate grid IPS potentials
    use pmeutil,only:mxzslabs,mxzstart, &
         nfft1,nfft2,nfft3
    use number
    use dimens_fcm
    use consta
    use nbips
    use vector

    LOGICAL LVDW,LELEC
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    INTEGER KBOT,KTOP,MRCX,MRCY,MRCZ
    real(chm_real)  XTLABC(6),VBOX,CGF
    !
    INTEGER IBINX,IBINY,IBINZ,IBINX0,IBINY0,IBINZ0
    INTEGER IBINX1,IBINY1,IBINZ1,IXYZ0,IXYZ1
    INTEGER I000,NF1,NF2,NF3
    real(chm_real) RIPSC,RIPSC2,RIPSCR,RIPSC2R,RIPSC6R,FIPS
    real(chm_real) WRK11,WRK21,WRK31,WRK12,WRK22,WRK32
    real(chm_real) WRK13,WRK23,WRK33,WRK14,WRK24,WRK34
    real(chm_real) AX(3),AY(3),AZ(3),PC(3)
    real(chm_real) XBIN,YBIN,ZBIN
    real(chm_real) EELIJ,ENBIJ,EELIJC,ENBIJC
    real(chm_real)  XI,YI,ZI,XJ,YJ,ZJ,XIJ,YIJ,ZIJ
    real(chm_real)  XI0,YI0,ZI0,XIJ0,YIJ0,ZIJ0
    real(chm_real)  R1,R2,R2R
    real(chm_real) U1,U2,U4,U8,U6R,U12R
    real(chm_real) PE,DPE,DEIJ,PEC,PDEC
    real(chm_real) PVC,DPVC,DVIJ,PVCC,PDVCC,VIPSEC,VIPSVCC
    real(chm_real) CIPSE(0:6),CIPSDE(0:6),DIPSE(6),DIPSDE(6)
    real(chm_real) CIPSVC(0:6),CIPSDVC(0:6),DIPSVC(6),DIPSDVC(6)
    integer rcskip,nfftdimrc
    rcskip=1
    nfftdimrc=nfft1+4
    ! Convert to fractional coordinates
    WRK11=XTLABC(1)
    WRK21=XTLABC(2)
    WRK31=XTLABC(4)
    WRK12=XTLABC(2)
    WRK22=XTLABC(3)
    WRK32=XTLABC(5)
    WRK13=XTLABC(4)
    WRK23=XTLABC(5)
    WRK33=XTLABC(6)
    XBIN=ONE/NFFT1
    YBIN=ONE/NFFT2
    ZBIN=ONE/NFFT3
    !  Calculate anisotropic radius
    IF(RAIPS > ZERO)THEN
       RIPSC=RAIPS
    ELSE
       !  Define RIPSC twice the longest boxsize
       RIPSC=TWO*ABS(XTLABC(1))
       IF(RIPSC<TWO*ABS(XTLABC(3)))RIPSC=TWO*ABS(XTLABC(3))
       IF(RIPSC<TWO*ABS(XTLABC(6)))RIPSC=TWO*ABS(XTLABC(6))
       !   Make RIPSC constant within certain range of box sizes
       RIPSC=ANINT(RIPSC)
    ENDIF
    RIPSCR=ONE/RIPSC
    RIPSC2=RIPSC*RIPSC
    RIPSC2R=ONE/RIPSC2
    RIPSC6R=RIPSC2R*RIPSC2R*RIPSC2R
    !  For the time being assume the lattice is orth. so maximum number of grid
    !    can be calculated in the following way
    NF1=NFFT1/2
    NF2=NFFT2/2
    NF3=NFFT3/2
    ! Find the lattice range that enclose the cutoff sphere
    !    x-y plan is: pxy=cross(vx,vy); 
    !        plan distance is: rz=dot(vz,pxy);
    !        plan numbers: 2*ripsc/rz + 1
    AX(1)=WRK11
    AX(2)=WRK12
    AX(3)=WRK13
    AY(1)=WRK21
    AY(2)=WRK22
    AY(3)=WRK23
    AZ(1)=WRK31
    AZ(2)=WRK32
    AZ(3)=WRK33
    CALL CROSS3(AX,AY,PC)
    CALL NORMALL(PC,3)
    CALL DOTPR(PC,AZ,3,ZIJ)
    MRCZ=INT(RIPSC*NFFT3/ABS(ZIJ)+ONE)
    CALL CROSS3(AY,AZ,PC)
    CALL NORMALL(PC,3)
    CALL DOTPR(PC,AX,3,XIJ)
    MRCX=INT(RIPSC*NFFT1/ABS(XIJ)+ONE)
    CALL CROSS3(AZ,AX,PC)
    CALL NORMALL(PC,3)
    CALL DOTPR(PC,AY,3,YIJ) 
    MRCY=INT(RIPSC*NFFT2/ABS(YIJ)+ONE)
    !  Set pressure tensor calculation parameter based on RIPSC
    CTENSOR=ONE
    !      IF(MRCX/NFFT1 >= 2 .AND.MRCY/NFFT2.GE.2
    !     &   .AND. MRCZ/NFFT3 >= 2 )CTENSOR=ZERO
    IF(2*MRCX <= NFFT1 .AND. 2*MRCY.LE.NFFT2 &
         .AND. 2*MRCZ <= NFFT3 )CTENSOR=ZERO
    ! Precalculate IPS constants
    U2=RIPS2/RIPSC2
    U4=U2*U2
    U8=U4*U4
    !  electrostatic parameters
    CIPSE(0)=ZERO
    CIPSE(1)=FIVE*SEVEN/EIGHT/TWO
    CIPSE(2)=-THREE*SEVEN/EIGHT/TWO
    CIPSE(3)=FIVE/EIGHT/TWO
    CIPSE(4)=ZERO
    CIPSE(5)=ZERO
    CIPSE(6)=ZERO
    CIPSE(6)=(ONE-TWO*CIPSE(1)-FOUR*CIPSE(2)-SIX*CIPSE(3) &
         -EIGHT*CIPSE(4)-TEN*CIPSE(5))/TWELVE
    PEC=FIVE*SEVEN/EIGHT/TWO
    PEC=ONE+CIPSE(0)+CIPSE(1)+CIPSE(2)+CIPSE(3) &
         +CIPSE(4)+CIPSE(5)+CIPSE(6)
    DIPSE(1)=TWO*CIPSE(1)
    DIPSE(2)=FOUR*CIPSE(2)
    DIPSE(3)=SIX*CIPSE(3)
    DIPSE(4)=EIGHT*CIPSE(4)
    DIPSE(5)=TEN*CIPSE(5)
    DIPSE(6)=TWELVE*CIPSE(6)
    CIPSDE(0)=0.0
    CIPSDE(1)=CIPSE(1)*U2*RIPSCR-AIPSE(1)*RIPSR
    CIPSDE(2)=CIPSE(2)*U4*RIPSCR-AIPSE(2)*RIPSR
    CIPSDE(3)=CIPSE(3)*U4*U2*RIPSCR-AIPSE(3)*RIPSR
    CIPSDE(4)=CIPSE(4)*U8*RIPSCR-AIPSE(4)*RIPSR
    CIPSDE(5)=CIPSE(5)*U8*U2*RIPSCR-AIPSE(5)*RIPSR
    CIPSDE(6)=CIPSE(6)*U8*U4*RIPSCR-AIPSE(6)*RIPSR
    PDEC=PIPSEC*RIPSR-PEC*RIPSCR
    DIPSDE(1)=TWO*CIPSDE(1)
    DIPSDE(2)=FOUR*CIPSDE(2)
    DIPSDE(3)=SIX*CIPSDE(3)
    DIPSDE(4)=EIGHT*CIPSDE(4)
    DIPSDE(5)=TEN*CIPSDE(5)
    DIPSDE(6)=TWELVE*CIPSDE(6)
    !   r6 parameters
    CIPSVC(0)=0.4376632
    CIPSVC(1)=0.5394569
    CIPSVC(2)=0.4012847
    CIPSVC(3)=0.2117008
    CIPSVC(4)=0.1607817
    CIPSVC(5)=0.0510378
    CIPSVC(6)=0.0091897
    CIPSVC(6)=(SIX-TWO*CIPSVC(1)-FOUR*CIPSVC(2)-SIX*CIPSVC(3) &
         -EIGHT*CIPSVC(4)-TWELVE*CIPSVC(5))/EIGHT/TWO
    PVCC=2.8111148
    PVCC=ONE+CIPSVC(0)+CIPSVC(1)+CIPSVC(2)+CIPSVC(3) &
         +CIPSVC(4)+CIPSVC(5)+CIPSVC(6)

    DIPSVC(1)=TWO*CIPSVC(1)
    DIPSVC(2)=FOUR*CIPSVC(2)
    DIPSVC(3)=SIX*CIPSVC(3)
    DIPSVC(4)=EIGHT*CIPSVC(4)
    DIPSVC(5)=TWELVE*CIPSVC(5)
    DIPSVC(6)=TWO*EIGHT*CIPSVC(6)
    CIPSDVC(0)=CIPSVC(0)*RIPSC6R-AIPSVC(0)*RIPS6R
    CIPSDVC(1)=CIPSVC(1)*U2*RIPSC6R-AIPSVC(1)*RIPS6R
    CIPSDVC(2)=CIPSVC(2)*U4*RIPSC6R-AIPSVC(2)*RIPS6R
    CIPSDVC(3)=CIPSVC(3)*U4*U2*RIPSC6R-AIPSVC(3)*RIPS6R
    CIPSDVC(4)=CIPSVC(4)*U8*RIPSC6R-AIPSVC(4)*RIPS6R
    CIPSDVC(5)=CIPSVC(5)*U8*U4*RIPSC6R-AIPSVC(5)*RIPS6R
    CIPSDVC(6)=CIPSVC(6)*U8*U8*RIPSC6R-AIPSVC(6)*RIPS6R
    DIPSDVC(1)=TWO*CIPSDVC(1)
    DIPSDVC(2)=FOUR*CIPSDVC(2)
    DIPSDVC(3)=SIX*CIPSDVC(3)
    DIPSDVC(4)=EIGHT*CIPSDVC(4)
    DIPSDVC(5)=TWELVE*CIPSDVC(5)
    DIPSDVC(6)=TWO*EIGHT*CIPSDVC(6)
    PDVCC=PIPSVCC*RIPS6R-PVCC*RIPSC6R
    !      VIPSEC=PEC*RIPSCR
    !      VIPSVCC=-PIPSVCC*RIPSC6R
    VIPSEC=ZERO
    VIPSVCC=ZERO
    !  Cutoff grids
    loop300: DO  IBINZ1=-MRCZ,MRCZ
       ZI=-IBINZ1*ZBIN
       IBINZ=MOD(IBINZ1+100*NFFT3,NFFT3)+1
#if KEY_PARALLEL==1
       IF(IBINZ < KBOT .OR. IBINZ  >  KTOP)cycle loop300    
#endif
       IBINZ0=(IBINZ-KBOT)*NFFTDIMRC*NFFTDIM2
       loop400: DO IBINY1=-MRCY,MRCY
          YI=-IBINY1*YBIN
          IBINY=MOD(IBINY1+100*NFFT2,NFFT2)
          IBINY0=IBINZ0+IBINY*NFFTDIMRC
          loop500: DO IBINX1=-MRCX,MRCX
             XI=-IBINX1*XBIN
             IBINX=MOD(IBINX1+100*NFFT1,NFFT1)
             IBINX0=IBINY0+IBINX
             I000=IBINX0+1
             XIJ=XI*WRK11+YI*WRK12+ZI*WRK13
             YIJ=XI*WRK21+YI*WRK22+ZI*WRK23
             ZIJ=XI*WRK31+YI*WRK32+ZI*WRK33
             R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
             EELIJ=ZERO
             ENBIJ=ZERO
             IF(R2 > RIPSC2) cycle loop500
             IF(R2 < RIPS2)THEN
                U2=R2*RIPS2R
                R2R=R2/(R2*R2+RSMALL)
                !  Electrostatic IPS
                !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                ! 
                PE=U2*(CIPSDE(1)+U2*(CIPSDE(2)+U2*(CIPSDE(3) &
                     +U2*(CIPSDE(4)+U2*(CIPSDE(5)+U2*CIPSDE(6))))))+PDEC
                DPE=U2*(DIPSDE(1)+U2*(DIPSDE(2)+U2*(DIPSDE(3) &
                     +U2*(DIPSDE(4)+U2*(DIPSDE(5)+U2*DIPSDE(6))))))
                EELIJ=EELIJ+PE
                DEIJ=DPE
                !  Lennard-Jones IPS
                U4=U2*U2
                U6R=ONE/U4/U2
                U12R=U6R*U6R
                !  L-J r6 term
                !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                !
                PVC=CIPSDVC(0)+U2*(CIPSDVC(1)+U2*(CIPSDVC(2)+U2*(CIPSDVC(3) &
                     +U2*(CIPSDVC(4)+U4*(CIPSDVC(5)+U4*CIPSDVC(6))))))+PDVCC
                DPVC=U2*(DIPSDVC(1)+U2*(DIPSDVC(2)+U2*(DIPSDVC(3) &
                     +U2*(DIPSDVC(4)+U4*(DIPSDVC(5)+U4*DIPSDVC(6))))))
                !  L-J r12 term --neglected
                !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                !
                ENBIJ=ENBIJ-PVC
                DVIJ=-DPVC
             ELSE 
                R2R=ONE/R2
                U2=R2*RIPSC2R
                !  Electrostatic IPS
                !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                ! 
                U1=SQRT(U2)
                PE=ONE/U1+U2*(CIPSE(1)+U2*(CIPSE(2)+U2*(CIPSE(3) &
                     +U2*(CIPSE(4)+U2*(CIPSE(5)+U2*CIPSE(6))))))-PEC
                DPE=-ONE/U1+U2*(DIPSE(1)+U2*(DIPSE(2)+U2*(DIPSE(3) &
                     +U2*(DIPSE(4)+U2*(DIPSE(5)+U2*DIPSE(6))))))
                EELIJ=EELIJ+PE*RIPSCR
                DEIJ=DPE*RIPSCR
                !  Lennard-Jones IPS
                U4=U2*U2
                U6R=ONE/U4/U2
                U12R=U6R*U6R
                !  L-J r6 term
                !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                !
                PVC=U6R+CIPSVC(0)+U2*(CIPSVC(1)+U2*(CIPSVC(2)+U2*(CIPSVC(3) &
                     +U2*(CIPSVC(4)+U4*(CIPSVC(5)+U4*CIPSVC(6))))))-PVCC
                DPVC=-SIX*U6R+U2*(DIPSVC(1)+U2*(DIPSVC(2)+U2*(DIPSVC(3) &
                     +U2*(DIPSVC(4)+U4*(DIPSVC(5)+U4*DIPSVC(6))))))
                !  L-J r12 term --neglected
                !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                !
                ENBIJ=ENBIJ-PVC*RIPSC6R
                DVIJ=-DPVC*RIPSC6R
             ENDIF
             !  Add boundary tensor
             DVIJ=(DVIJ-VIPSVCC)*R2R
             DEIJ=(DEIJ-VIPSEC)*R2R
             VDWXX(I000)=VDWXX(I000)+XIJ*XIJ*DVIJ
             VDWXY(I000)=VDWXY(I000)+XIJ*YIJ*DVIJ
             VDWYY(I000)=VDWYY(I000)+YIJ*YIJ*DVIJ
             VDWXZ(I000)=VDWXZ(I000)+XIJ*ZIJ*DVIJ
             VDWYZ(I000)=VDWYZ(I000)+YIJ*ZIJ*DVIJ
             VDWZZ(I000)=VDWZZ(I000)+ZIJ*ZIJ*DVIJ
             ELEXX(I000)=ELEXX(I000)+XIJ*XIJ*DEIJ
             ELEXY(I000)=ELEXY(I000)+XIJ*YIJ*DEIJ
             ELEYY(I000)=ELEYY(I000)+YIJ*YIJ*DEIJ
             ELEXZ(I000)=ELEXZ(I000)+XIJ*ZIJ*DEIJ
             ELEYZ(I000)=ELEYZ(I000)+YIJ*ZIJ*DEIJ
             ELEZZ(I000)=ELEZZ(I000)+ZIJ*ZIJ*DEIJ
             VDWARRAY(I000)=VDWARRAY(I000)+ENBIJ
             ELEARRAY(I000)=ELEARRAY(I000)+EELIJ
          enddo loop500
       enddo loop400
    enddo loop300
    !
    FIPS=FOUR*PI*RIPSC2*RIPSC/THREE/VBOX
    IF(LVDW)THEN 
       !  Only the 6th-term uses ripsc as cutoff
       EIPSANB=-FIPS*CIJSUM*PVCC*RIPSC6R
    ELSE
       EIPSANB=ZERO
    ENDIF
    IF(LELEC)THEN 
       EIPSAEL=FIPS*CGF*CGSUM*CGSUM*PEC/RIPSC/TWO
    ELSE
       EIPSAEL=ZERO
    ENDIF
    VIRAIPS=EIPSANB+EIPSAEL
    RETURN
  END SUBROUTINE PBC_IPS_ENG


  !****************************************************************
  !                        PBC_IPS2D_ENG
  !****************************************************************

  SUBROUTINE PBC_IPS2D_ENG(LVDW,LELEC,KBOT,KTOP, &
       NFFTDIM1,NFFTDIM2,NFFTDIM3, &
       XTLABC,VBOX,CGF)    
    ! Calculate grid IPS potentials
    use pmeutil,only:mxzslabs,mxzstart, &
         nfft1,nfft2,nfft3
    use number
    use dimens_fcm
    use consta
    use nbips
    use vector

    LOGICAL LVDW,LELEC
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    INTEGER KBOT,KTOP,MRCX,MRCY,MRCZ
    real(chm_real)  XTLABC(6),VBOX,CGF
    !
    INTEGER IBINX0,IBINY0,IBINZ0,IBINX1,IBINY1,IBINZ1
    INTEGER IBINX,IBINY,IBINZ,IXYZ0,IXYZ1
    INTEGER I000,NF1,NF2,NF3
    real(chm_real) RIPSC,RIPSC2,RIPSCR,RIPSC2R,RIPSC4R,RIPSC6R,RIPSC8R
    real(chm_real) WRK11,WRK21,WRK31,WRK12,WRK22,WRK32
    real(chm_real) WRK13,WRK23,WRK33,WRK14,WRK24,WRK34
    real(chm_real) AX(3),AY(3),AZ(3),PC(3)
    real(chm_real) XBIN,YBIN,ZBIN
    real(chm_real) EELIJ,ENBIJ,EELIJC,ENBIJC
    real(chm_real) SBOX,HBOX,FIPSU,FIPSH,FIPS,ENBIJU,ENBIJH, &
         EELIJU,EELIJH
    real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,X2,Y2,Z2,IPSX1,IPSY1,IPSZ1
    real(chm_real)  R1V,R2V,R6V,RU2,RH2,R2R
    real(chm_real)  R1,R2,R6,R12,U2,U4,U6,U12,H2,H4,H6,H12
    real(chm_real)  RC1,RC2,RC6,PU1,PU2,PU3,PU6,PU12, &
         PH1,PH2,PH3,PH6,PH12
    real(chm_real)  PU1N,PU6N,PU12N,PH1N,PH6N,PH12N
    real(chm_real)  PE,PE0,PE1,DPE,DPEU,DPEH
    real(chm_real)  PVC,PVCS0,PVC0,PVC1,DPVC,DPVCU,DPVCH,DPVCSU,DPVCSH
    real(chm_real)  PVAS0,PVA0,PVA1,DPVA,DPVAU,DPVAH,DPVASU,DPVASH
    real(chm_real)  FIJ,DEIJ,DVIJ,PEC,PVCC
    real(chm_real)  DUX,DUY,DUZ,DHX,DHY,DHZ
    real(chm_real)  DELU,DELH,DELX,DELY,DELZ,DVCU,DVCH,DVCX,DVCY,DVCZ
    real(chm_real) DELUC,DELHC
    real(chm_real) CIPSEC,CIPSE(0:5),CIPSEU(0:4),CIPSEH(0:4)
    real(chm_real) DIPSE(5),DIPSEU(4),DIPSEH(4)
    real(chm_real) CIPSVCC,CIPSVC(0:5),CIPSVCU(0:4),CIPSVCH(0:4)
    real(chm_real) DIPSVC(5),DIPSVCU(4),DIPSVCH(4)
    real(chm_real) ROOT2
    LOGICAL QU,QH
    integer i,j,ndata
    integer rcskip,nfftdimrc
    rcskip=1
    nfftdimrc=nfft1+4
    ndata=nfftdimrc*nfftdim2*nfft3
    ! Convert to fractional coordinates
    WRK11=XTLABC(1)
    WRK21=XTLABC(2)
    WRK31=XTLABC(4)
    WRK12=XTLABC(2)
    WRK22=XTLABC(3)
    WRK32=XTLABC(5)
    WRK13=XTLABC(4)
    WRK23=XTLABC(5)
    WRK33=XTLABC(6)
    XBIN=ONE/NFFT1
    YBIN=ONE/NFFT2
    ZBIN=ONE/NFFT3
    IPSX1=ONE-IPSX
    IPSY1=ONE-IPSY
    IPSZ1=ONE-IPSZ
    !  Calculate anisotropic radius
    IF(RAIPS > ZERO)THEN
       RIPSC=RAIPS
    ELSE
       !  Define RIPSC twice the longest boxsize
       RIPSC=TWO*ABS(XTLABC(1))
       IF(RIPSC<TWO*ABS(XTLABC(3)))RIPSC=TWO*ABS(XTLABC(3))
       IF(RIPSC<TWO*ABS(XTLABC(6)))RIPSC=TWO*ABS(XTLABC(6))
       !   Make RIPSC constant within certain range of box sizes
       RIPSC=ANINT(RIPSC)
    ENDIF
    RIPSCR=ONE/RIPSC
    RIPSC2=RIPSC*RIPSC
    RIPSC2R=ONE/RIPSC2
    RIPSC4R=RIPSC2R*RIPSC2R
    RIPSC6R=RIPSC2R*RIPSC2R*RIPSC2R
    RIPSC8R=RIPSC6R*RIPSC2R
    !  For the time being assume the lattice is orth. so maximum number of grid
    !    can be calculated in the following way
    !  A Minumum full PBC boxe to include boundary energies uniformly
    NF1=NFFT1/2
    NF2=NFFT2/2
    NF3=NFFT3/2
    ! Find the lattice range that enclose the cutoff cylinder hight:2rc,radius: rc
    !    x-y plan is: pxy=cross(vx,vy); 
    !        plan distance to origin is: rz=rc*(COS(S)+SIN(S));
    !          s is the angle between pxy and z axis cos(s)=pxy(z)/|pxy|
    !        plan grid numbers: rz*nfft3/dot(<pc>,lz) + 1
    ROOT2=SQRT(TWO)
    AX(1)=WRK11
    AX(2)=WRK12
    AX(3)=WRK13
    AY(1)=WRK21
    AY(2)=WRK22
    AY(3)=WRK23
    AZ(1)=WRK31
    AZ(2)=WRK32
    AZ(3)=WRK33
    CALL CROSS3(AX,AY,PC)
    CALL NORMALL(PC,3)
    R2R=ABS(PC(3))+SQRT(ONE-PC(3)*PC(3))
    CALL DOTPR(PC,AZ,3,ZIJ)
    MRCZ=NF3*INT(RIPSC*TWO*R2R/ABS(ZIJ)+ONE-ONE/NFFT3)
    CALL CROSS3(AY,AZ,PC)
    CALL NORMALL(PC,3)
    R2R=ABS(PC(3))+SQRT(ONE-PC(3)*PC(3))
    CALL DOTPR(PC,AX,3,XIJ)
    MRCX=NF1*INT(RIPSC*TWO*R2R/ABS(XIJ)+ONE-ONE/NFFT1)
    CALL CROSS3(AZ,AX,PC)
    CALL NORMALL(PC,3)
    R2R=ABS(PC(3))+SQRT(ONE-PC(3)*PC(3))
    CALL DOTPR(PC,AY,3,YIJ)
    MRCY=NF2*INT(RIPSC*TWO*R2R/ABS(YIJ)+ONE-ONE/NFFT2)
    !
    !  Set pressure tensor calculation parameter based on RIPSC
    CTENSOR=ONE
    !      IF(MRCX/NFFT1 >= 2 .AND.MRCY/NFFT2.GE.2
    !     &   .AND. MRCZ/NFFT3 >= 2 )CTENSOR=ZERO
    IF(2*MRCX <= NFFT1 .AND. 2*MRCY.LE.NFFT2 &
         .AND. 2*MRCZ <= NFFT3 )CTENSOR=ZERO
    !   Calculate boundary energy scaling factors
    SBOX=IPSX*IPSY*XTLABC(1)*XTLABC(3)*MRCX*MRCY/NF1/NF2 &
         +IPSY*IPSZ*XTLABC(3)*XTLABC(6)*MRCZ*MRCY/NF3/NF2 &
         +IPSX*IPSZ*XTLABC(1)*XTLABC(6)*MRCX*MRCZ/NF1/NF3
    HBOX=IPSX1*XTLABC(1)*MRCX/NF1 &
         +IPSY1*XTLABC(3)*MRCY/NF2 &
         +IPSZ1*XTLABC(6)*MRCZ/NF3
    FIPSH=PI*RIPSC2/SBOX
    FIPSU=TWO*RIPSC/HBOX
    FIPS=FIPSU*FIPSH
    ! Precalculate IPS constants
    ROOT2=SQRT(TWO)
    !------------------------------------------------------------
    !  1/r parameters
    !     CIPSEC -- boundary cornor energy
    CIPSEC=1.1566523/RIPSC
    !     AIPSEU(0:4), DIPSEU(1:4)-- uc boundary  energy term
    CIPSEU(0)=2.0642083
    CIPSEU(1)=2.6873108/RIPSC
    CIPSEU(2)=-2.9268456*RIPSC2R
    CIPSEU(3)=2.1226370*RIPSC*RIPSC4R
    CIPSEU(4)=-0.6314084*RIPSC4R
    DIPSEU(1)=CIPSEU(1)
    DIPSEU(2)=TWO*CIPSEU(2)
    DIPSEU(3)=THREE*CIPSEU(3)
    DIPSEU(4)=FOUR*CIPSEU(4)
    !     AIPSEH(0:4), DIPSEH(1:4) -- hc boundary  energies
    CIPSEH(0)=2.0182029
    CIPSEH(1)=0.5703608/RIPSC
    CIPSEH(2)=-0.17669854*RIPSC2R
    CIPSEH(3)=0.04280393*RIPSC*RIPSC4R
    CIPSEH(4)=-0.1205994*RIPSC4R
    DIPSEH(1)=CIPSEH(1)
    DIPSEH(2)=TWO*CIPSEH(2)
    DIPSEH(3)=THREE*CIPSEH(3)
    DIPSEH(4)=FOUR*CIPSEH(4)
    !     CIPSE(0:5), DIPSE(1:5) -- u-h surface  energies
    CIPSE(0)=-1.6413432/RIPSC
    CIPSE(1)=3.979949*RIPSC2R
    CIPSE(2)=-5.2368556*RIPSC*RIPSC4R
    CIPSE(3)=6.058640*RIPSC4R
    CIPSE(4)=-3.9314972*RIPSC*RIPSC6R
    CIPSE(5)=1.2358496*RIPSC6R
    DIPSE(1)=CIPSE(1)
    DIPSE(2)=TWO*CIPSE(2)
    DIPSE(3)=THREE*CIPSE(3)
    DIPSE(4)=FOUR*CIPSE(4)
    DIPSE(5)=FIVE*CIPSE(5)
    !
    !------------------------------------------------------------
    ! vdw 1/r^6 parameters
    !     PIPSVCC -- boundary cornor energy
    CIPSVCC=0.6676714*RIPSC6R
    !     AIPSVCU(0:4), DIPSEU(1:4)-- uc boundary  energy term
    CIPSVCU(0)=2.093048
    CIPSVCU(1)=0.0354176/RIPSC
    CIPSVCU(2)=-0.0212502*RIPSC2R
    CIPSVCU(3)=0.1228670*RIPSC*RIPSC4R
    CIPSVCU(4)=-0.0389980*RIPSC4R
    DIPSVCU(1)=CIPSVCU(1)
    DIPSVCU(2)=TWO*CIPSVCU(2)
    DIPSVCU(3)=THREE*CIPSVCU(3)
    DIPSVCU(4)=FOUR*CIPSVCU(4)
    !     AIPSVCH(0:4), DIPSVCH(1:4) -- hc boundary  energies
    CIPSVCH(0)=1.9238283
    CIPSVCH(1)=0.1913090/RIPSC
    CIPSVCH(2)=-0.0518235*RIPSC2R
    CIPSVCH(3)=-0.1356389*RIPSC*RIPSC4R
    CIPSVCH(4)=0.00986502*RIPSC4R
    DIPSVCH(1)=CIPSVCH(1)
    DIPSVCH(2)=TWO*CIPSVCH(2)
    DIPSVCH(3)=THREE*CIPSVCH(3)
    DIPSVCH(4)=FOUR*CIPSVCH(4)
    !     AIPSVC(0:5), DIPSVC(1:5) -- u-h surface  energies
    CIPSVC(0)=0.3737736*RIPSC6R
    CIPSVC(1)=0.3273148*RIPSC6R
    CIPSVC(2)=0.5613918*RIPSC6R
    CIPSVC(3)=0.17277327*RIPSC6R*RIPSC4R
    CIPSVC(4)=ZERO
    CIPSVC(5)=ZERO
    DIPSVC(1)=TWO*CIPSVC(1)
    DIPSVC(2)=TWO*CIPSVC(2)
    DIPSVC(3)=FOUR*CIPSVC(3)
    !
    !        ENBIJC=-CIPSVCC
    !        EELIJC=CIPSEC
    !  Assume overall neutral
    EELIJC=ZERO
    ENBIJC=ZERO
    !------------------------------------------------------------
    ! vdw 1/r12 parameters ---- IGNORED
    !  Cutoff grids
    loop300: DO IBINZ1=-MRCZ,MRCZ-1
       ZI=-IBINZ1*ZBIN
       IBINZ0=MOD(IBINZ1+100*NFFT3,NFFT3)+1
#if KEY_PARALLEL==1
       IF(IBINZ0 < KBOT .OR. IBINZ0  >  KTOP) cycle loop300    
#endif
       IBINZ0=(IBINZ0-KBOT)*NFFTDIMRC*NFFTDIM2
       loop400: DO IBINY1=-MRCY,MRCY-1
          YI=-IBINY1*YBIN
          IBINY0=IBINZ0+MOD(IBINY1+100*NFFT2,NFFT2)*NFFTDIMRC
          loop500: DO IBINX1=-MRCX,MRCX-1
             XI=-IBINX1*XBIN
             IBINX0=IBINY0+MOD(IBINX1+100*NFFT1,NFFT1)
             I000=IBINX0+1
             XIJ=XI*WRK11+YI*WRK12+ZI*WRK13
             YIJ=XI*WRK21+YI*WRK22+ZI*WRK23
             ZIJ=XI*WRK31+YI*WRK32+ZI*WRK33
             X2=XIJ*XIJ
             Y2=YIJ*YIJ
             Z2=ZIJ*ZIJ
             R2=X2+Y2+Z2
             RU2=IPSX*X2+IPSY*Y2+IPSZ*Z2
             RH2=R2-RU2
             EELIJ=FIPS*EELIJC
             ENBIJ=FIPS*ENBIJC
             QU=RU2 > RIPSC2
             QH=RH2 > RIPSC2
             !  Skip grids outside both U and H cutoff
             IF(QU.AND.QH)GOTO 650
             DELU=ZERO
             DELH=ZERO
             DVCU=ZERO
             DVCH=ZERO
             DUX=IPSX*XIJ
             DUY=IPSY*YIJ
             DUZ=IPSZ*ZIJ
             DHX=XIJ-DUX
             DHY=YIJ-DUY
             DHZ=ZIJ-DUZ            
             IF(QH)GOTO 610
             !  HC boundary energies.  For all grids within the U cutoff
             RC2=RIPSC2+RH2
             RC1=SQRT(RC2)
             PU2=RC2/RIPSC2/TWO
             !  Electrostatic 2D1D IPS: E=1/r
             !   e1hc=(1-PU1)^2 PE
             !                    PU1=r/rc  PU2=r2/rc2
             !                    PE=(a0+a1 r+a2 r2+a3 r3+a4 r4)/r
             !                    DPE=(a1 r+2 a2 r2+3 a3 r3+4 a4 r4)/r
             !   de1uc/dh=h((1-PU1)^2 DPE-(1-PU2)PE)/r2
             !
             PU1=RC1/RIPSC/ROOT2
             PU1N=ONE-PU1
             PE=CIPSEU(0)/RC1+CIPSEU(1)+RC1*(CIPSEU(2)+CIPSEU(3)*RC1 &
                  +CIPSEU(4)*RC2)
             DPE=DIPSEU(1)+RC1*(DIPSEU(2)+DIPSEU(3)*RC1 &
                  +DIPSEU(4)*RC2)
             EELIJU=PU1N*PU1N*PE
             EELIJ=EELIJ+FIPSH*EELIJU
             DELH=(PU1N*PU1N*DPE-(ONE-PU2)*PE)/RC2
             !          DELH=ZERO
             !  VDW 
             RC6=RC2*RC2*RC2
             PU6=PU2*PU2*PU2
             PU12=PU6*PU6 
             PU6N=ONE-PU6
             PU12N=ONE-PU12
             !  L-J r6 term
             !   e6uc=(1-PU6)^2 PVC
             !                    PU6=r6/rc6  
             !                    PVC=(a0+a1 r+a2 r2+a3 r3+a4 r4)/r6
             !                    DPVC=(a1 r+2 a2 r2+3 a3 r3+4 a4 r4)/r6
             !   de6uc/dh=h((1-PU6)^2 DPVC-(1-PU12)PVC)/r2
             !
             PVC=(CIPSVCU(0)+RC1*(CIPSVCU(1) &
                  +RC1*(CIPSVCU(2)+RC1*CIPSVCU(3)+CIPSVCU(4)*RC2)))/RC6
             DPVC=RC1*(DIPSVCU(1) &
                  +RC1*(DIPSVCU(2)+RC1*DIPSVCU(3)+DIPSVCU(4)*RC2))/RC6
             DVCH=PU6N*PU6N*DPVC-SIX*PU12N*PVC
             !  L-J r12 term --Ignored
             !   e12uc=(1-PU12)^2 PVA
             !                    PH12=r12/rc12  
             !                    PV1=(a0+a1 r+a2 r2+a3 r3+a4 r4)/r12
             !                    DPV1=(a1 r+2 a2 r2+3 a3 r3+4 a4 r4)/r12
             !   de12uc/dh=h((1-PU12)^2 DPVA-(1-PU24)PVA)/r2
             !
             ENBIJU=-(PU6N*PU6N*PVC)
             ENBIJ=ENBIJ+FIPSH*ENBIJU
             DVCH=-(PU6N*PU6N*DPVC-SIX*PU12N*PVC)/RC2
610          CONTINUE
             IF(QU)GOTO 630
             !  UC boundary energies.  For all grids within the H cutoff
             RC2=RU2+RIPSC2
             RC1=SQRT(RC2)
             PH2=RC2/RIPSC2/TWO
             !  Electrostatic 2D1D IPS: E=1/r
             !   e1hc=(1-PH1)^2 PE
             !                    PH1=r/rc  PH2=r2/rc2
             !                    PE=(a0+a1 r+a2 r2+a3 r3+a4 r4)/R
             !                    DPE=a1 +2 a2 r+3 a3 r2+4 a4 r3
             !   de1hc/du=u((1-PHR)^2 DPE-(1-PH)PE)/r3
             !
             PH1=RC1/RIPSC/ROOT2
             PH1N=ONE-PH1
             PE=CIPSEH(0)/RC1+CIPSEH(1)+RC1*(CIPSEH(2)+CIPSEH(3)*RC1 &
                  +CIPSEH(4)*RC2)
             DPE=DIPSEH(1)+RC1*(DIPSEH(2)+DIPSEH(3)*RC1 &
                  +DIPSEH(4)*RC2)
             EELIJH=PH1N*PH1N*PE
             EELIJ=EELIJ+FIPSU*EELIJH
             DELU=(PH1N*PH1N*DPE-(ONE-PH2)*PE)/RC2
             !          DELU=ZERO
             !  VDW
             RC6=RC2*RC2*RC2
             PH6=PH2*PH2*PH2
             PH12=PH6*PH6
             PH6N=ONE-PH6
             PH12N=ONE-PH12
             !  L-J r6 term
             !   e6hc=(1-PH6)^2 PVC
             !                    PH6=r6/rc6  
             !                    PVC=(a0+a1 r+a2 r2+a3 r3+a4 r4)/r6
             !                    DPVC=(a1 r+2 a2 r2+3 a3 r3+4 a4 r4)/r6
             !   de6hc/du=u((1-PH6)^2 DPVC-(1-PH12)PVC)/r2
             !
             PVC=(CIPSVCH(0)+RC1*(CIPSVCH(1) &
                  +RC1*(CIPSVCH(2)+RC1*CIPSVCH(3)+CIPSVCH(4)*RC2)))/RC6
             DPVC=RC1*(DIPSVCH(1) &
                  +RC1*(DIPSVCH(2)+RC1*DIPSVCH(3)+DIPSVCH(4)*RC2))/RC6
             !  L-J r12 term  --Ignored
             !   e12hc=(1-PH12)^2 PVA
             !                    PH12=r12/rc12  
             !                    PV1=(a0+a1 r+a2 r2+a3 r3+a4 r4)/r12
             !                    DPV1=(a1 r+2 a2 r2+3 a3 r3+4 a4 r4)/r12
             !   de12hc/du=u((1-PH12)^2 DPVA-(1-PH24)PVA)/r2
             !
             ENBIJH=-(PH6N*PH6N*PVC)
             ENBIJ=ENBIJ+FIPSU*ENBIJH
             DVCU=-(PH6N*PH6N*DPVC-SIX*PH12N*PVC)/RC2
630          CONTINUE
             IF(QU.OR.QH)GOTO 650
             !  U-H surface energies.  For all grids within the U and H cutoff
             !       To avoid R2=0
             R1=R2/SQRT(R2+RSMALL)
             R1V=R1/(R2+RSMALL)
             R2V=R2/(R2*R2+RSMALL)
             R6V=R2V*R2V*R2V
             IF(R2 < RIPS2)THEN
                U2=R2*RIPS2R
                !  Electrostatic IPS
                !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                ! 
                PE=-U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                     +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))+PIPSEC
                DPE=-U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                     +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                EELIJ=EELIJ+PE*RIPSR-R1V
                DEIJ=(DPE*RIPSR+R1V)*R2V
                !  Lennard-Jones IPS
                U4=U2*U2
                !  L-J r6 term
                !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                !
                PVC=-AIPSVC(0)-U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                     +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))+PIPSVCC
                DPVC=-U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                     +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
                !  L-J r12 term --neglected
                !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                !
                ENBIJ=ENBIJ-PVC*RIPS6R+R6V
                DVIJ=-(DPVC*RIPS6R+SIX*R6V)*R2V
             ELSE 
                DEIJ=ZERO
                DVIJ=ZERO
             ENDIF
             !  The 2+1D IPS
             !   u2=ipsx*x2+ipsy*y2+ipsz*z2
             !   h2=(1-ipsx)*x2+(1-ipsy)*y2+(1-ipsz)*z2
             !   r2=u2+h2
             !   de/dx=de/du*du/dx+de/dh*dh/dx
             !   du/dx=ipsx*x/u
             !   dh/dx=(1-ipsx)*x/h
             U2=RU2
             H2=RH2
             U4=U2*U2
             U6=U4*U2
             U12=U6*U6
             H4=H2*H2
             H6=H4*H2
             H12=H6*H6
             R2=U2+H2
             R6=R2*R2*R2
             R12=R6*R6
             PU1=R2/(U2+RIPSC2)
             PU2=PU1*PU1
             PU3=PU2*PU1
             PU6=PU3*PU3
             PH1=R2/(RIPSC2+H2)
             PH2=PH1*PH1
             PH3=PH2*PH1
             PH6=PH3*PH3
             PE0=(ONE-PU1)*(ONE-PH1)
             !  Electrostatic 2D1D IPS: E=s/r
             !   et1uh=(1-PU)^2(1-PH)^2 PE
             !                    PU=(U2+H2)/(U2+HC2)   PH=(U2+H2)/(UC2+H2)
             !                    PE=1/r+a0+a1 r+a2 r2+a3 r3+a4 r4+a5 r5
             !                    DPE=(-1/r+a1 r+a2 r2+a3 r3+a4 r4+a5 r5)/r2
             !   det1uh/du=u(1-PU)(1-PH)((1-PU)(1-PH) DPE-4 rc2(PU-PU2)PH PE/r4)
             !   det1uh/dh=h(1-PU)(1-PH)((1-PU)(1-PH) DPE-4 rc2(PH-PH2)PU PE/r4)
             !
             PE1=R1V+CIPSE(0)+R1*(CIPSE(1)+R1*(CIPSE(2) &
                  +R1*(CIPSE(3)+R1*CIPSE(4)+R2*CIPSE(5))))
             DPE=-R1V+R1*(DIPSE(1)+R1*(DIPSE(2)+R1*(DIPSE(3) &
                  +R1*(DIPSE(4)+R1*DIPSE(5)))))
             DPEU=PE0*(PE0*DPE &
                  -EIGHT*PE1*RIPSC2*(PU1-PU2)*PH1*R2V)
             DPEH=PE0*(PE0*DPE &
                  -EIGHT*PE1*RIPSC2*PU1*(PH1-PH2)*R2V)
             EELIJ=EELIJ+PE0*PE0*PE1
             DELU=DELU+DEIJ+DPEU*R2V
             DELH=DELH+DEIJ+DPEH*R2V
             !  L-J r6 term
             !   e6s=(1-PU3)^2(1-PH3)^2/r6
             !   e6uh=(1-PU2)^2(1-PH2)^2 PVC
             !                    PU3=((U2+H2)/(U2+HC2))^3   PH3=((U2+H2)/(UC2+H2))^3
             !                    PVC=a0+a1 PH u2/r2+a2 PU h2/r2+a3 r4
             !                    DPVCU=2a1 PH/r2-2a2 PU2 h2/r4+4a3 r2
             !                    DPVCH=-2a1 PH2 u2/r4+2a2 PU/r2+4a3 r2
             ! 
             !   de6s/du=-6u(1-PU3)(1-PH3)(2(PU3-PU4)(1-PH3)+(1-PU3)(1+PH3))/r8)
             !   de6s/dh=-6h(1-PU3)(1-PH3)(2(PH3-PH4)(1-PU3)+(1+PU3)(1-PH3))/r8)
             !
             !   de6uh/du=u(1-PU2)(1-PH2)((1-PU2)(1-PH2)DPVCU
             !                  -8(PH2(1-PU2)+(PU2-PU3)(1-PH2))PVC/r2)
             !   de6uh/dh=h(1-PU2)(1-PH2)((1-PU2)(1-PH2)DPVCH
             !                  -8(PU2(1-PH2)+(PH2-PH3)(1-PU2))PVC/r2)
             !
             PVCS0=(ONE-PU3)*(ONE-PH3)
             PVC0=(ONE-PU2)*(ONE-PH2)
             PVC1=CIPSVC(0)+ &
                  (CIPSVC(1)*PH1*U2+CIPSVC(2)*PU1*H2+CIPSVC(3)*R6)*R2V
             DPVCSU=-SIX*PVCS0*(TWO*(PU3-PU2*PU2)*(ONE-PH3)+ &
                  (ONE-PU3)*(ONE+PH3))*R6V
             DPVCU=PVC0*( &
                  PVC0*((DIPSVC(1)*PH1-DIPSVC(2)*PU2*H2*R2V)+DIPSVC(3)*R2*R2) &
                  -EIGHT*(PH2*(ONE-PU2)+(PU2-PU3)*(ONE-PH2))*PVC1)
             DPVCSH=-SIX*PVCS0*(TWO*(ONE-PU3)*(PH3-PH2*PH2)+ &
                  (ONE-PH3)*(ONE+PU3))*R6V
             DPVCH=PVC0*( &
                  PVC0*((-DIPSVC(1)*PH2*U2*R2V+DIPSVC(2)*PU1)+DIPSVC(3)*R2*R2) &
                  -EIGHT*(PU2*(ONE-PH2)+(PH2-PH3)*(ONE-PU2))*PVC1)
             !  L-J r12 term --ignored
             !   e12s=(1-PU6)^2(1-PH6)^2/r12
             !   e12uh=(1-PU)^4(1-PH)^4 PVA
             !                    PU6=((U2+H2)/(U2+HC2))^6   PH6=((U2+H2)/(UC2+H2))^6
             !                    PVA=a0+a1 PH2 u4/r4+a2 PU6 h12/r12+a3 r10
             !                    DPVAU=4 a1 PH2 u2/r4-12a2 PU7 h12/r14+10a3 r8
             !                    DPVAH=-4a1 PH4 u4/r6+12a2 PU6 h10/r12+10a3 r8
             !
             !   de12s/du=-12u(1-PU6)(1-PH6)(2(PU6-PU7)(1-PH6)+(1-PU6)(1+PH6))/r14)
             !   de12s/dh=-12h(1-PU6)(1-PH6)(2(PH6-PH7)(1-PU6)+(1+PU6)(1-PH6))/r14)
             !
             !   de12uh/du=u(1-PU)^4(1-PH)^3((1-PH)DPVAU
             !                  -8(PU(1-PH)+PH)PVA/r2)
             !   de12uh/dh=h(1-PU)^3(1-PH)^4((1-PU)DPVAH
             !                  -8(PH(1-PU)+PU)PVA/r2)
             !
             ENBIJ=ENBIJ-(PVCS0*PVCS0*R6V+PVC0*PVC0*PVC1)
             DVCU=DVCU+DVIJ-(DPVCSU+DPVCU)*R2V
             DVCH=DVCH+DVIJ-(DPVCSH+DPVCH)*R2V
             ! Derivatives of x, y, z
             DVCX=DVCU*DUX+DVCH*DHX
             DVCY=DVCU*DUY+DVCH*DHY
             DVCZ=DVCU*DUZ+DVCH*DHZ
             VDWXX(I000)=VDWXX(I000)+XIJ*DVCX &
                  -IPSX*ENBIJU-IPSX1*ENBIJH-ENBIJC
             VDWXY(I000)=VDWXY(I000)+XIJ*DVCY
             VDWYY(I000)=VDWYY(I000)+YIJ*DVCY &
                  -IPSY*ENBIJU-IPSY1*ENBIJH-ENBIJC
             VDWXZ(I000)=VDWXZ(I000)+XIJ*DVCZ
             VDWYZ(I000)=VDWYZ(I000)+YIJ*DVCZ
             VDWZZ(I000)=VDWZZ(I000)+ZIJ*DVCZ &
                  -IPSZ*ENBIJU-IPSZ1*ENBIJH-ENBIJC
             DELX=DELU*DUX+DELH*DHX
             DELY=DELU*DUY+DELH*DHY
             DELZ=DELU*DUZ+DELH*DHZ
             ELEXX(I000)=ELEXX(I000)+XIJ*DELX &
                  -IPSX*EELIJU-IPSX1*EELIJH-EELIJC
             ELEXY(I000)=ELEXY(I000)+XIJ*DELY
             ELEYY(I000)=ELEYY(I000)+YIJ*DELY &
                  -IPSY*EELIJU-IPSY1*EELIJH-EELIJC
             ELEXZ(I000)=ELEXZ(I000)+XIJ*DELZ
             ELEYZ(I000)=ELEYZ(I000)+YIJ*DELZ
             ELEZZ(I000)=ELEZZ(I000)+ZIJ*DELZ &
                  -IPSZ*EELIJU-IPSZ1*EELIJH-EELIJC
650          CONTINUE
             VDWARRAY(I000)=VDWARRAY(I000)+ENBIJ
             ELEARRAY(I000)=ELEARRAY(I000)+EELIJ
          enddo loop500
       enddo loop400
    enddo loop300
    ! Correct the vdw interaction at u=0 and h=0:enbij(0,0)=4/ripsc6
    VDWARRAY(1)=VDWARRAY(1)+FOUR*RIPSC6R
    !
    FIPS=TWO*PI*RIPSC2*RIPSC/VBOX
    IF(LVDW)THEN 
       !  Only the 6th-term uses ripsc as cutoff
       EIPSANB=-FIPS*CIJSUM*CIPSVCC
    ELSE
       EIPSANB=ZERO
    ENDIF
    IF(LELEC)THEN 
       EIPSAEL=FIPS*CGF*CGSUM*CGSUM*CIPSEC/TWO
    ELSE
       EIPSAEL=ZERO
    ENDIF
    VIRAIPS=EIPSANB+EIPSAEL
    RETURN
  END SUBROUTINE PBC_IPS2D_ENG



  !***********************************************************************
  !                 +----------------------------+
  !                 |        GRAD_SUM            |
  !                 +----------------------------+
  !***********************************************************************
  SUBROUTINE PBC_IPS_GRAD( &
       igood, kbot, ktop, &
       NUMATOMS,CHARGE, &
       RECIP,FX,FY,FZ, &
       FR1,FR2,FR3, &
       NFFTDIM1,NFFTDIM2,NFFTDIM3, &
       my_ks,latm,XNSYMM,XTLABC)

    use pmeutil,only: mxystart,mxyslabs,nfft1,nfft2,nfft3,forder, &
         theta1,theta2,theta3,dtheta1,dtheta2,dtheta3
    use number
    use dimens_fcm
    use consta
    use parallel
    use nbips
    !
    !
    INTEGER NUMATOMS,ORDER
    INTEGER XNSYMM
    INTEGER NFFTDIM1,NFFTDIM2,NFFTDIM3
    real(chm_real) RECIP(9),XTLABC(6)
    real(chm_real) FR1(*),FR2(*),FR3(*)
    real(chm_real) FX(*),FY(*),FZ(*)
    real(chm_real) CHARGE(*)
    integer latm,my_ks(latm)
    !
    integer igoo,ig
    real(chm_real) VAL0,VAL1,VAL2,VAL3,VAL0A,VAL1A,VAL2A,VAL3A
    INTEGER IPT1,IPT2,IPT3
    !
    INTEGER KQ
    integer igood, kbot, ktop
    !
    INTEGER N,ITH1,ITH2,ITH3,I,J,K,M1,M2
    INTEGER IX,IY,IZ,IXYZ0
    real(chm_real) CGI,VDWI,QI,WI,QWI
    real(chm_real) F1,F2,F3,TERM,CFACT,CFACT1,CFACT2
    integer rcskip,nfftdimrc
    !
    rcskip=1
    if(nfft1/2 /= (nfft1+1)/2)then
       CALL WRNDIE(-5,'<AIPS fill charge grid>' &
            ,'fftx dimension not even ')
    endif
    nfftdimrc=nfft1+4
    !
    order=forder
    CFACT=ONE/(NFFT1*NFFT2*NFFT3)
    CFACT1=CFACT*CCELEC/XNSYMM
    CFACT2=CFACT/XNSYMM
    !
    do ig = 1,igood
       n=my_ks(ig)
       if(xnsymm == 1)then
          igoo=ig
       else
          igoo=n
       endif
       CGI = CFACT1*CHARGE(N)
       VDWI = CFACT2*WNB(N)
       F1 = ZERO
       F2 = ZERO
       F3 = ZERO
       K = INT(FR3(igoo)) - ORDER + 1 + NFFT3
       !
       DO ITH3 = 1,ORDER
          K=K+1
          IF(K > NFFT3) K=K-NFFT3
          KQ=K
#if KEY_PARALLEL==1
          IF ( K  >=  KBOT .AND. K  <=  KTOP )THEN
             KQ = K - MXYSTART(MYNOD)
#endif 
             VAL1A =  NFFT1 * THETA3(ITH3,ig)
             VAL2A =  NFFT2 * THETA3(ITH3,ig)
             VAL3A =  NFFT3 * DTHETA3(ITH3,igoo)
             !
             J = INT(FR2(igoo)) - ORDER + 1 + NFFT2
             IPT1=(KQ-1)*NFFTDIM2 -1
             !
             I = INT(FR1(igoo)) - ORDER + 1 + NFFT1
             IF(I >= NFFT1) I=I-NFFT1
             !
             DO ITH2 = 1,ORDER
                J=J+1
                IF(J > NFFT2) J=J-NFFT2
                !
                VAL1= VAL1A * THETA2(ITH2,ig)
                VAL2= VAL2A * DTHETA2(ITH2,igoo)
                VAL3= VAL3A * THETA2(ITH2,ig)

                IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                IPT3= IPT2 + rcskip*(NFFT1-I)
                !
                DO ITH1 = 1,ORDER
                   !
                   !       force is negative of grad
                   QI=CGI*QARRAY(IPT2)
                   WI=VDWI*WARRAY(IPT2)
                   QWI=QI+WI
                   F1 = F1 - QWI * VAL1 * DTHETA1(ITH1,igoo)
                   F2 = F2 - QWI * VAL2 * THETA1(ITH1,ig)
                   F3 = F3 - QWI * VAL3 * THETA1(ITH1,ig)
                   !
                   IPT2=IPT2+rcskip
                   IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                ENDDO
             ENDDO
#if KEY_PARALLEL==1
          ENDIF
#endif 
       ENDDO
       !
       FX(N) = FX(N) - (RECIP(1)*F1+RECIP(4)*F2+RECIP(7)*F3)
       FY(N) = FY(N) - (RECIP(2)*F1+RECIP(5)*F2+RECIP(8)*F3)
       FZ(N) = FZ(N) - (RECIP(3)*F1+RECIP(6)*F2+RECIP(9)*F3)
    ENDDO
    RETURN
  END SUBROUTINE PBC_IPS_GRAD


  SUBROUTINE IPSSYS(NATOM,NATC,CG, &
       CNBA,CNBB,IAC,ITC)
    !-----------------------------------------------------------------------
    !    Calculate boundary energies for 3D IPS systems
    !
    use dimens_fcm
    use number
    use consta
    use stream
    use image
    use memory
    use nbips
#if KEY_PARALLEL==1 /*pll*/
    use parallel
#endif /* (pll)*/
    INTEGER NATOM,NATC
    real(chm_real) CG(*),CNBA(*), CNBB(*)
    INTEGER IAC(*), ITC(*)
    INTEGER, PARAMETER :: MXTYPE=500
    INTEGER,allocatable,dimension(:) :: NSUMIT
    real(chm_real),allocatable,dimension(:) :: CISUM
    real(chm_real)  CISUMI
    INTEGER I,J,ITI,ITJ,ITMAX,IC,NITI,NITJ,ITEMP
    real(chm_real)  CGI,CGIJ,AIJ,CIJ,ANIJ
    real(chm_real)  SIG2,SIG6,SIG12
    !
    call chmalloc('enbips.src','IPSSYS','nsumit',natc,intg=nsumit)
    call chmalloc('enbips.src','IPSSYS','cisum',natc,crl=cisum)
    DO ITI=1,NATC
       NSUMIT(ITI)=0
       CISUM(ITI)=ZERO
    ENDDO
    CGSUM=ZERO
    CG2SUM=ZERO
    ITMAX=0
    DO I=1,NATOM
       CGSUM=CGSUM+CG(I)
       CG2SUM=CG2SUM+CG(I)*CG(I)
       ITI=ITC(IAC(I))
       IF(ITI > ITMAX)ITMAX=ITI
       NSUMIT(ITI)=NSUMIT(ITI)+1
    ENDDO
    IF(ITMAX > MXTYPE)STOP 'INCREASE MXTYPE IN IPSSYS3D!'
    C2SUM=ZERO
    A2SUM=ZERO
    CIJSUM=ZERO
    AIJSUM=ZERO
    !  system energy is calculated based on all pairs
    DO ITI=1,ITMAX
       IC=ITI*(ITI-1)/2+ITI
       SIG2=CNBA(IC)
       SIG6=SIG2*SIG2*SIG2
       SIG12=SIG6*SIG6
       AIJ=CNBB(IC)*SIG12
       CIJ=TWO*CNBB(IC)*SIG6
       NITI=NSUMIT(ITI)
       ANIJ=NITI*NITI/TWO
       C2SUM=C2SUM+CIJ*NITI
       A2SUM=A2SUM+AIJ*NITI
       CISUMI=CISUM(ITI)+CIJ*NITI
       CIJSUM=CIJSUM+CIJ*ANIJ
       AIJSUM=AIJSUM+AIJ*ANIJ
       DO ITJ=ITI+1,ITMAX
          NITJ=NSUMIT(ITJ)
          IC=ITJ*(ITJ-1)/2+ITI
          SIG2=CNBA(IC)
          SIG6=SIG2*SIG2*SIG2
          SIG12=SIG6*SIG6
          AIJ=CNBB(IC)*SIG12
          CIJ=TWO*CNBB(IC)*SIG6
          ANIJ=NITI*NITJ
          CIJSUM=CIJSUM+CIJ*ANIJ
          AIJSUM=AIJSUM+AIJ*ANIJ
          CISUMI=CISUMI+CIJ*NITJ
          CISUM(ITJ)=CISUM(ITJ)+CIJ*NITI
       ENDDO
       CISUM(ITI)=CISUMI
    ENDDO
    !      CGIJSUM=CGSUM*CGSUM/TWO
    !write(*,*)'IPSSYS:',mynod,cgsum,cg2sum,a2sum,aijsum,c2sum,cijsum
    IF(QAIPS)THEN
       if(allocated(wnb)) &
            deallocate(wnb)
       allocate(wnb(natom),stat=alloc_err)
       if(alloc_err /= 0 ) write(0,*)"unable to allocate wnb"
       DO I=1,NATOM
          ITI=ITC(IAC(I))
          !   pair interaction part
          WNB(I)=CISUM(ITI)/SQRT(CIJSUM*TWO)
       ENDDO
    ENDIF
    call chmdealloc('enbips.src','IPSSYS','nsumit',natc,intg=nsumit)
    call chmdealloc('enbips.src','IPSSYS','cisum',natc,crl=cisum)
    RETURN 
  END SUBROUTINE IPSSYS


  SUBROUTINE EEXIPS(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC, &
       LVDW,LELEC,LVIPS,LEIPS,LCONS,INB,IBLO,CG,CNBA,CNBB,IAC,ITC,MAXROW, &
       EPS,E14FAC,VBOX,DX,DY,DZ, &
       X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,ICALL)
    !
    !   3D IPS interaction between excluded atom pairs
    !   This routine must be called first to update IPS parameters when needed
    !    and to inital electrostatic and vdw energies
    !
    !
    ! input/output
    use dimens_fcm
    use consta
    use number
    use stream
    use nbips
    use image
    use parallel
#if KEY_PERT == 1
    use pert,only:qpert,qpssp
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
#endif
    real(chm_real) ENB, EEL
    INTEGER IFRSTA,ILASTA,NATOM, NATC
    INTEGER IBLO(*),INB(*),ICALL
    real(chm_real) CG(*),CNBA(*), CNBB(*)
    INTEGER IAC(*), ITC(*),MAXROW
    real(chm_real)  EPS, E14FAC,VBOX,FIPS
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*)
    LOGICAL QECONTX
    real(chm_real) ECONTX(*)
    LOGICAL LVDW,LELEC,LVDWX,LELECX,LVIPS,LEIPS,LCONS
    real(chm_real)  DD1(*)
    INTEGER IUPT(*)
    LOGICAL LSECD
     ! local
    INTEGER I,J,K,IC,ICX,ITI,ITJ,II,JJ,IADD
    INTEGER  ILAST, IFIRST,LJROW
    real(chm_real) ESCALE,VSCALE
    real(chm_real) DXI, DYI, DZI,DF,DFX,DFY,DFZ
    real(chm_real) ENBIJ,EELIJ,ECONTI
    real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,R2
    real(chm_real)  CGF,CGI,CGIJ,AIJ,CIJ,AEXIJ,CEXIJ,SIG2,SIG6,SIG12
    real(chm_real)  U1,U2,U4,U6R,U12R
    real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
    real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,DDF
    ! begin
    IF(QIPSEXP)THEN
      CALL EEXIPSEXP(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC, &
         LVDW,LELEC,LVIPS,LEIPS,LCONS,INB,IBLO,CG,CNBA,CNBB,IAC,ITC,MAXROW, &
         EPS,E14FAC,VBOX,DX,DY,DZ, &
         X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,ICALL)
      RETURN
    ENDIF
    LVDWX=LVDW.AND.LVIPS
    LELECX=LELEC.AND.LEIPS
    CGF=CCELEC/EPS
    QIPSUPD=MOD(NIPSCNT,NIPSFRQ) == 0
    NIPSCNT=NIPSCNT+ICALL
    !
    ! check to see if IPS parameters need to be updated

    IF (QIPSINI &
#if KEY_PERT == 1
         .OR. QPERT &
#endif
         )THEN
       QIPSFIN=QIPSFIN.OR.NTRANS == 0
       IF(QIPSFIN)THEN
          IF(.NOT.QAIPS)THEN
             ! Finite system with 3D IPS only
             !       No shift for L-J IPS energies
             PIPSVC0=AIPSVC(0)
             PIPSVA0=AIPSVA(0)
             PIPSVCC=ZERO
             PIPSVAC=ZERO
          ELSE
             !            GRIDIPS=GRIDIPS*TWO
          ENDIF
       ENDIF
       CALL IPSSYS(NATOM,NATC,CG, &
            CNBA,CNBB,IAC,ITC)
       QIPSINI=.FALSE.
    ENDIF
    !
    IF(QIPSFIN)THEN
    !     EIPSSNB=ZERO
    !   EIPSSEL=ZERO
    !   VIRIPS=ZERO
      FIPS=ONE
    ELSE
      FIPS=FOUR*PI*RIPS2*RIPS/THREE/VBOX
    ENDIF
#if KEY_PARALLEL==1
       IF(MYNOD == 0)THEN                
#endif
          IF(QAIPS)THEN
             ! Only the vdw 12th-term is considered here
             EIPSSNB=FIPS*AIJSUM*PIPSVAC*RIPS12R
             EIPSSEL=ZERO
          ELSE
             EIPSSNB=FIPS*(AIJSUM*(PIPSVAC)*RIPS6R &
                  -CIJSUM*(PIPSVCC))*RIPS6R
             EIPSSEL=FIPS*CGF*CGSUM*CGSUM*PIPSEC*RIPSR/TWO
          ENDIF
          IF(.NOT.LELEC)EIPSSEL=ZERO
          IF(.NOT.LVDW)EIPSSNB=ZERO
#if KEY_PARALLEL==1
       ELSE
          EIPSSNB=ZERO
          EIPSSEL=ZERO
       ENDIF
#endif 
       VIRIPS=(EIPSSNB+EIPSSEL)
    !ENDIF
    ! apply option for netcharge boundary contribution
    IF(.NOT.QIPSNETCG)THEN
      !  ignoring net charge boudnary ele ips energy
      EIPSSEL=ZERO
    ENDIF
    !
    ENB=EIPSSNB
    EEL=EIPSSEL
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN                
#endif
         ! add self-term
         IF(LVDW)ENB=EIPSSNB+HALF*(A2SUM*PIPSVA0*rips6r-C2SUM*PIPSVC0)*rips6r
         IF(LELEC)EEL=EIPSSEL+HALF*CG2SUM*PIPSE0*CGF*RIPSR
#if KEY_PARALLEL==1
    ENDIF
#endif 
    DO I=1,9
       PIPSVIR(I)=ZERO
    ENDDO
    PIPSVIR(1)=VIRIPS
    PIPSVIR(5)=VIRIPS
    PIPSVIR(9)=VIRIPS
#if KEY_DOMDEC==1
    if (q_domdec) RETURN
#endif
    !IF(QPSSP)RETURN
    !  Evenly split long rang correction to each atom
    !=======================================================================
    !
    !=======================================================================
    !   Main loop begin
    !=======================================================================
    ILAST = 0
    EELIJ=ZERO
    ENBIJ=ZERO
    IF(IFRSTA > 1)ILAST=IBLO(IFRSTA-1)
    DO I=IFRSTA,ILASTA
       !        define index list J
       IFIRST = ILAST+1
       ILAST=IBLO(I)
       IF(LELECX)THEN
          IF(LCONS)THEN
            CGI=CG(I)*CGF*RIPSR
          ELSE
             CGI=CG(I)*CGF*RIPS2R
          ENDIF
          IF(QECONTX) THEN
            CGIJ=CG(I)*CGI
            ! charge self iteraction.  Already calculated
            EELIJ=HALF*CGIJ*PIPSE0
            ECONTX(I)=ECONTX(I)+EELIJ
          ENDIF
       ENDIF
       IF(LVDWX)THEN
          ITI=ITC(IAC(I))
          IF(QECONTX) THEN
            IC=ITI*(ITI-1)/2+ITI
            SIG2=CNBA(IC)*RIPS2R
            SIG6=SIG2*SIG2*SIG2
            SIG12=SIG6*SIG6
            AIJ=CNBB(IC)*SIG12
            CIJ=TWO*CNBB(IC)*SIG6
           ! Atom i long range referece and self interaction. Already calculated
            ENBIJ=HALF*(AIJ*PIPSVA0-CIJ*PIPSVC0)
            ECONTX(I)=ECONTX(I)+ENBIJ
          ENDIF
       ENDIF
       !
       DXI=DX(I)
       DYI=DY(I)
       DZI=DZ(I)
       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
       loop100: DO K=IFIRST,ILAST
          J = INB(K)
          IF(J < 0)THEN
             J=-J
             ESCALE=E14FAC
             VSCALE=ONE
             LJROW=MAXROW
          ELSE
             ESCALE=ZERO
             VSCALE=ZERO
             LJROW=0
          ENDIF
          XIJ=XI-X(J)
          YIJ=YI-Y(J)
          ZIJ=ZI-Z(J)
          R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
          U2=R2*RIPS2R
          IF(LELECX)THEN
             !  Electrostatic IPS
             !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
             !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
             ! 
             CGIJ=CGI*CG(J)
             IF(LCONS)THEN
                U1=SQRT(U2)
                PE=ESCALE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                   +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                DPE=-ESCALE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                   +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
             ELSE
                PE=ESCALE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                   +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                DPE=-TWO*ESCALE/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                   +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
             ENDIF
             EELIJ=CGIJ*(PE-PIPSEC)
             EEL=EEL+EELIJ
             DF=-CGIJ*DPE/R2
             IF(LSECD) THEN
                IF(LCONS)THEN
                   DDPE=TWO*ESCALE/U1+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                      +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                ELSE
                   DDPE=SIX*ESCALE/U2+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                      +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                ENDIF
                DDF=CGIJ*DDPE/R2/R2
             ENDIF
          ELSE
             DF=ZERO
             DDF=ZERO
          ENDIF
          IF(LVDWX)THEN
             ITJ=ITC(IAC(J))
             IF(ITI > ITJ)THEN
                IC=ITI*(ITI-1)/2+ITJ
             ELSE
                IC=ITJ*(ITJ-1)/2+ITI
             ENDIF
             SIG2=CNBA(IC)*RIPS2R
             SIG6=SIG2*SIG2*SIG2
             SIG12=SIG6*SIG6
             AIJ=CNBB(IC)*SIG12
             CIJ=TWO*CNBB(IC)*SIG6
             ICX=IC+LJROW
             SIG2=CNBA(ICX)*RIPS2R
             SIG6=SIG2*SIG2*SIG2
             SIG12=SIG6*SIG6
             AEXIJ=CNBB(ICX)*SIG12
             CEXIJ=TWO*CNBB(ICX)*SIG6
             ! debuging domdec
             !  AIJ=AEXIJ
             !  CIJ=CEXIJ
               AEXIJ=VSCALE*AEXIJ
               CEXIJ=VSCALE*CEXIJ
             !
             U4=U2*U2
             U6R=ONE/U4/U2
             U12R=U6R*U6R
             !  L-J r6 term
             !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
             !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
             !
             PVC=CEXIJ*U6R+CIJ*(AIPSVC(0)+U2*(AIPSVC(1) &
                  +U2*(AIPSVC(2)+U2*(AIPSVC(3)+U2*(AIPSVC(4) &
                  +U4*(AIPSVC(5)+U4*AIPSVC(6))))))-PIPSVCC)
             DPVC=-SIX*CEXIJ*U6R+CIJ*(U2*(BIPSVC(1) &
                  +U2*(BIPSVC(2)+U2*(BIPSVC(3)+U2*(BIPSVC(4) &
                  +U4*(BIPSVC(5)+U4*BIPSVC(6)))))))
             !  L-J r12 term 
             !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
             !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
             !
             PVA=AEXIJ*U12R+AIJ*(AIPSVA(0)+U2*(AIPSVA(1) &
                  +U2*(AIPSVA(2)+U2*(AIPSVA(3)+U4*(AIPSVA(4) &
                  +U4*(AIPSVA(5)+U4*AIPSVA(6))))))-PIPSVAC)
             DPVA=-TWELVE*AEXIJ*U12R+AIJ*(U2*(BIPSVA(1) &
                  +U2*(BIPSVA(2)+U2*(BIPSVA(3)+U4*(BIPSVA(4) &
                  +U4*(BIPSVA(5)+U4*BIPSVA(6)))))))
             ENBIJ=PVA-PVC
             ENB=ENB+ENBIJ
             DF=DF-(DPVA-DPVC)/R2
           IF(LSECD) THEN
              DDPVC=42.0D0*CEXIJ*U6R+CIJ*U2*(BBIPSVC(1)+U2*(BBIPSVC(2)+U2*(BBIPSVC(3) &
                    +U2*(BBIPSVC(4)+U4*(BBIPSVC(5)+U4*BBIPSVC(6))))))
              DDPVA=156.0D0*AEXIJ*U12R+AIJ*U2*(BBIPSVA(1)+U2*(BBIPSVA(2)+U2*(BBIPSVA(3) &
                    +U4*(BBIPSVA(4)+U4*(BBIPSVA(5)+U4*BBIPSVA(6))))))
              DDF=DDF+(DDPVA-DDPVC)/R2/R2
           ENDIF
          ENDIF
          ! Calculate total forces
          DFX=DF*XIJ
          DFY=DF*YIJ
          DFZ=DF*ZIJ
          DXI=DXI-DFX
          DYI=DYI-DFY
          DZI=DZI-DFZ
          DX(J)=DX(J)+DFX
          DY(J)=DY(J)+DFY
          DZ(J)=DZ(J)+DFZ
         !
          IF(QECONTX) THEN
             ECONTI=HALF*(EELIJ+ENBIJ)
             ECONTX(I)=ECONTX(I)+ECONTI
             ECONTX(J)=ECONTX(J)+ECONTI
          ENDIF
                 IF(LSECD) THEN
                    !
                       !
                       DDF=DDF+DF/R2
                       !
                       !     NOW UPDATE DERIVATIVE MATRICIES
                       !
                       AXX=XIJ*XIJ*DDF-DF
                       AYY=YIJ*YIJ*DDF-DF
                       AZZ=ZIJ*ZIJ*DDF-DF
                       AXY=XIJ*YIJ*DDF
                       AXZ=XIJ*ZIJ*DDF
                       AYZ=YIJ*ZIJ*DDF
                          II=3*I-2
                          JJ=3*J-2
                          !
                          IADD=IUPT(II)+II
                          DD1(IADD)=DD1(IADD)+AXX
                          IADD=IUPT(II+1)+II+1
                          DD1(IADD)=DD1(IADD)+AYY
                          IADD=IUPT(II+2)+II+2
                          DD1(IADD)=DD1(IADD)+AZZ
                          IADD=IUPT(II)+II+1
                          DD1(IADD)=DD1(IADD)+AXY
                          IADD=IUPT(II)+II+2
                          DD1(IADD)=DD1(IADD)+AXZ
                          IADD=IUPT(II+1)+II+2
                          DD1(IADD)=DD1(IADD)+AYZ
                          !
                          IADD=IUPT(JJ)+JJ
                          DD1(IADD)=DD1(IADD)+AXX
                          IADD=IUPT(JJ+1)+JJ+1
                          DD1(IADD)=DD1(IADD)+AYY
                          IADD=IUPT(JJ+2)+JJ+2
                          DD1(IADD)=DD1(IADD)+AZZ
                          IADD=IUPT(JJ)+JJ+1
                          DD1(IADD)=DD1(IADD)+AXY
                          IADD=IUPT(JJ)+JJ+2
                          DD1(IADD)=DD1(IADD)+AXZ
                          IADD=IUPT(JJ+1)+JJ+2
                          DD1(IADD)=DD1(IADD)+AYZ
                          !
                          IF (JJ < II) THEN
                             IADD=IUPT(JJ)+II
                             DD1(IADD)=DD1(IADD)-AXX
                             IADD=IUPT(JJ+1)+II+1
                             DD1(IADD)=DD1(IADD)-AYY
                             IADD=IUPT(JJ+2)+II+2
                             DD1(IADD)=DD1(IADD)-AZZ
                             IADD=IUPT(JJ)+II+1
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(JJ+1)+II
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(JJ)+II+2
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(JJ+2)+II
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(JJ+1)+II+2
                             DD1(IADD)=DD1(IADD)-AYZ
                             IADD=IUPT(JJ+2)+II+1
                             DD1(IADD)=DD1(IADD)-AYZ
                          ELSE
                             IADD=IUPT(II)+JJ
                             DD1(IADD)=DD1(IADD)-AXX
                             IADD=IUPT(II+1)+JJ+1
                             DD1(IADD)=DD1(IADD)-AYY
                             IADD=IUPT(II+2)+JJ+2
                             DD1(IADD)=DD1(IADD)-AZZ
                             IADD=IUPT(II+1)+JJ
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(II)+JJ+1
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(II+2)+JJ
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(II)+JJ+2
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(II+2)+JJ+1
                             DD1(IADD)=DD1(IADD)-AYZ
                             IADD=IUPT(II+1)+JJ+2
                             DD1(IADD)=DD1(IADD)-AYZ
                          ENDIF
                          !
                    !
                 ENDIF  ! (LSECD)
       enddo loop100
       DX(I) =  DXI
       DY(I) =  DYI
       DZ(I) =  DZI
    ENDDO
    !=======================================================================
    !   Main loop end
    !=======================================================================
    RETURN
  END SUBROUTINE EEXIPS


  SUBROUTINE IPSSYSEXP(NATOM,NATC,CG, &
       CNBA,CNBB,IAC,ITC)
    !-----------------------------------------------------------------------
    !    Calculate boundary energies for 3D IPS systems
    !
    use dimens_fcm
    use number
    use consta
    use stream
    use image
    use memory
    use nbips
#if KEY_PARALLEL==1 /*pll*/
    use parallel
#endif /* (pll)*/
    INTEGER NATOM,NATC
    real(chm_real) CG(*),CNBA(*), CNBB(*)
    INTEGER IAC(*), ITC(*)
    INTEGER, PARAMETER :: MXTYPE=500
    INTEGER,allocatable,dimension(:) :: NSUMIT
    real(chm_real),allocatable,dimension(:) :: CISUM
    real(chm_real)  CISUMI
    INTEGER I,J,ITI,ITJ,ITMAX,IC,NITI,NITJ,ITEMP
    real(chm_real)  CGI,CGIJ,AIJ,CIJ,ANIJ
    real(chm_real)  SIG,SIG2,ASC,ASC2,EXPASC,EXPASC2,CEXP,SINH,COSH
    real(chm_real)  EIPSVAC,DIPSVAC,EIPSVCC,DIPSVCC
    !
    call chmalloc('enbips.src','IPSSYS','nsumit',natc,intg=nsumit)
    call chmalloc('enbips.src','IPSSYS','cisum',natc,crl=cisum)
    DO ITI=1,NATC
       NSUMIT(ITI)=0
       CISUM(ITI)=ZERO
    ENDDO
    CGSUM=ZERO
    CG2SUM=ZERO
    ITMAX=0
    DO I=1,NATOM
       CGSUM=CGSUM+CG(I)
       CG2SUM=CG2SUM+CG(I)*CG(I)
       ITI=ITC(IAC(I))
       IF(ITI > ITMAX)ITMAX=ITI
       NSUMIT(ITI)=NSUMIT(ITI)+1
    ENDDO
    IF(ITMAX > MXTYPE)STOP 'INCREASE MXTYPE IN IPSSYS3D!'
    SRIPS=RBIG
    A2SUM=ZERO
    AIJSUM=ZERO
    !  system energy is calculated based on all pairs
    DO ITI=1,ITMAX
       IC=ITI*(ITI-1)/2+ITI
       SIG2=CNBA(IC)
       IF(SIG2<RSMALL)CYCLE
       SIG=SQRT(SIG2)
       IF(SRIPS>RIPS/SIG)SRIPS=RIPS/SIG
       AIJ=CNBB(IC)
       NITI=NSUMIT(ITI)
       ANIJ=NITI*NITI/TWO
       A2SUM=A2SUM+AIJ*NITI
       AIJSUM=AIJSUM+AIJ*ANIJ*SIG*SIG2
       DO ITJ=ITI+1,ITMAX
          NITJ=NSUMIT(ITJ)
          IC=ITJ*(ITJ-1)/2+ITI
          SIG2=CNBA(IC)
          IF(SIG2<RSMALL)CYCLE
          SIG=SQRT(SIG2)
          IF(SRIPS>RIPS/SIG)SRIPS=RIPS/SIG
          AIJ=CNBB(IC)
          ANIJ=NITI*NITJ
          AIJSUM=AIJSUM+AIJ*ANIJ*SIG*SIG2
       ENDDO
    ENDDO
    ! including local region volume
    AIJSUM=FOUR*PI*SRIPS**3*AIJSUM/THREE
    call chmdealloc('enbips.src','IPSSYSRN','nsumit',natc,intg=nsumit)
    call chmdealloc('enbips.src','IPSSYSRN','cisum',natc,crl=cisum)
      ! expips=exp(-a s) + exp(a(s-2sc))-2exp(-a sc)
      ! expipsc=2exp(-a sc)+12exp(-a sc)/(a sc)^2+6exp(-2a sc)/(a sc)^3
      !        =exp(-a sc)(2+(12+6exp(-a sc)/(a sc))/(a sc)^2)
      !     s=r/rm  sc=rc/rm
      ASC=AEXPPOT*SRIPS
      ASC2=ASC*ASC
      EXPASC=EXP(-ASC)
      EXPASC2=EXPASC*EXPASC
      AIPSVA(2)=ZERO
      AIPSVA(3)=ZERO
      ! exp(-2 a sc)
      AIPSVA(4)=EXPASC2
      ! -2exp(-a sc)
      AIPSVA(5)=-TWO*EXPASC
      PIPSVAC=EXPASC*(TWO+(TWELVE+SIX*EXPASC/ASC)/ASC2)
      PIPSVA0=EXPASC*(EXPASC-TWO)
      ASC=CEXPPOT*SRIPS
      ASC2=ASC*ASC
      EXPASC=EXP(-ASC)
      EXPASC2=EXPASC*EXPASC
      AIPSVC(2)=ZERO
      AIPSVC(3)=ZERO
      ! exp(-2 a sc)
      AIPSVC(4)=EXPASC2
      ! -2exp(-a sc)
      AIPSVC(5)=-TWO*EXPASC
      PIPSVCC=EXPASC*(TWO+(TWELVE+SIX*EXPASC/ASC)/ASC2)
      PIPSVC0=EXPASC*(EXPASC-TWO)
      ! derivative parameters
      BIPSVA(1)=-AEXPPOT*AIPSVA(1)
      BIPSVA(2)=AEXPPOT*AIPSVA(2)
      BIPSVA(3)=AEXPPOT*AIPSVA(3)
      BIPSVA(4)=AEXPPOT*AIPSVA(4)
      BIPSVA(5)=AEXPPOT*AIPSVA(5)
      BIPSVC(1)=-CEXPPOT*AIPSVC(1)
      BIPSVC(2)=CEXPPOT*AIPSVC(2)
      BIPSVC(3)=CEXPPOT*AIPSVC(3)
      BIPSVC(4)=CEXPPOT*AIPSVC(4)
      BIPSVC(5)=CEXPPOT*AIPSVC(5)
  
    !  Print out the NBIPS parameters
    IF(PRNLEV >= 4)THEN
        WRITE(OUTU,1010) AEXPPOT,CEXPPOT
        WRITE(OUTU,1020) RIPS,SRIPS
        WRITE(OUTU,1250)(AIPSVA(I),I=0,5)
        WRITE(OUTU,1251)(BIPSVA(I),I=1,1)
        WRITE(OUTU,1150)(AIPSVC(I),I=0,5)
        WRITE(OUTU,1151)(BIPSVC(I),I=1,1)
        WRITE(OUTU,2117) PIPSVAC,PIPSVA0,PIPSVCC,PIPSVC0
        !WRITE(OUTU,2118) EIPSVAC,DIPSVAC,EIPSVCC,DIPSVCC
     ENDIF
1010 FORMAT (' Exponents AEXPPOT and CEXPPOT:',2F15.6)
1020 FORMAT (' RIPS and SRIPS:',2F15.6)
1150 FORMAT("  AIPSVC:",8(1X,E14.6))
1151 FORMAT("  BIPSVC:",15X,8(1X,E14.6))
1250 FORMAT("  AIPSVA:",8(1X,E14.6))
1251 FORMAT("  BIPSVA:",15X,8(1X,E14.6))
2117 FORMAT ("PIPSVAC,PIPSVA0,PIPSVCC,PIPSVC0=",4E15.6)
2118 FORMAT ("EIPSVAC,DIPSVAC,EIPDVCC,DIPDVCC=",4E15.6)
    RETURN
  END SUBROUTINE IPSSYSEXP


  SUBROUTINE EEXIPSEXP(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC, &
       LVDW,LELEC,LVIPS,LEIPS,LCONS,INB,IBLO,CG,CNBA,CNBB,IAC,ITC,MAXROW, &
       EPS,E14FAC,VBOX,DX,DY,DZ, &
       X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,ICALL)
    !
    !   3D IPS interaction between excluded atom pairs
    !   This routine must be called first to update IPS parameters when needed
    !    and to inital electrostatic and vdw energies
    !
    !
    ! input/output
    use dimens_fcm
    use consta
    use number
    use stream
    use nbips
    use image
    use parallel
    use pert,only:qpert,qpssp
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
#endif
    real(chm_real) ENB, EEL
    INTEGER IFRSTA,ILASTA,NATOM, NATC
    INTEGER IBLO(*),INB(*),ICALL
    real(chm_real) CG(*),CNBA(*), CNBB(*)
    INTEGER IAC(*), ITC(*),MAXROW
    real(chm_real)  EPS, E14FAC,VBOX,FIPS
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*)
    LOGICAL QECONTX
    real(chm_real) ECONTX(*)
    LOGICAL LVDW,LELEC,LVDWX,LELECX,LVIPS,LEIPS,LCONS
    real(chm_real)  DD1(*)
    INTEGER IUPT(*)
    LOGICAL LSECD
     ! local
    INTEGER I,J,K,IC,ICX,ITI,ITJ,II,JJ,IADD
    INTEGER  ILAST, IFIRST,LJROW
    real(chm_real) ESCALE,VSCALE
    real(chm_real) DXI, DYI, DZI,DF,DFX,DFY,DFZ
    real(chm_real) ENBIJ,EELIJ,ECONTI
    real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,R1,R2
    real(chm_real)  CGF,CGI,CGIJ,AIJ,CIJ,AEXIJ,CEXIJ,SIG2,SIG6,SIG12
    real(chm_real)  U1,U2,EXPAREX,EXPCREX,DFIJ
    real(chm_real)  SIG,SIGEX,AKR,AKRC,CKR,CKRC,EXPAR,EXPCR,EXPARC,EXPCRC
    real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
    real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,DDF
    ! begin
    LVDWX=LVDW.AND.LVIPS
    LELECX=LELEC.AND.LEIPS
    CGF=CCELEC/EPS
    QIPSUPD=MOD(NIPSCNT,NIPSFRQ) == 0
    NIPSCNT=NIPSCNT+ICALL
    !
    ! check to see if IPS parameters need to be updated
    IF(QIPSINI.OR.QPERT)THEN
       QIPSFIN=QIPSFIN.OR.NTRANS == 0
       ! full DE calculation no need precalculation parameters
       CALL IPSSYSEXP(NATOM,NATC,CG, &
            CNBA,CNBB,IAC,ITC)
       QIPSINI=.FALSE.
    ENDIF
    !
    IF(QIPSFIN)THEN
       !EIPSSNB=ZERO
       !EIPSSEL=ZERO
       !VIRIPS=ZERO
      FIPS=ONE
    ELSE
       FIPS=FOUR*PI*RIPS2*RIPS/THREE/VBOX
    ENDIF
#if KEY_PARALLEL==1
       IF(MYNOD == 0)THEN
#endif
          IF(QAIPS)THEN
            call wrndie(-2,'<AIPS>', &                                      
            ' IPS/DFFT not implemented for COLFFT--Use 3D IPS (add "PXYZ")!')  
          ELSE
            ! boundary energies: Ec=Sum(ecij)Vc/Vbox-Sum(ecii)/2
            !EIPSSNB=(AIJSUM/VBOX-HALF*A2SUM)*(AIPSVA(1)*PIPSVAC-AIPSVC(1)*PIPSVCC)
            EIPSSNB=(AIJSUM/VBOX)*(AIPSVA(1)*PIPSVAC-AIPSVC(1)*PIPSVCC)
            EIPSSEL=FIPS*CGF*CGSUM*CGSUM*PIPSEC*RIPSR/TWO
          ENDIF
          IF(.NOT.LELEC)EIPSSEL=ZERO
          IF(.NOT.LVDW)EIPSSNB=ZERO
#if KEY_PARALLEL==1
       ELSE
          EIPSSNB=ZERO
          EIPSSEL=ZERO
       ENDIF
#endif
    !ENDIF
    ! apply option for netcharge boundary contribution
    IF(.NOT.QIPSNETCG)THEN
      !  ignoring net charge boudnary ele ips energy
      EIPSSEL=ZERO
    ENDIF
    !
    VIRIPS=(EIPSSNB+EIPSSEL)
    ENB=EIPSSNB
    EEL=EIPSSEL
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN
#endif
         ! add self-term
         ! self term is calculated with full DB potential
         !IF(LVDW)ENB=EIPSSNB+HALF*A2SUM*(PIPSVA0-PIPSVAC)
         IF(LELEC)EEL=EIPSSEL+HALF*CG2SUM*PIPSE0*CGF*RIPSR
#if KEY_PARALLEL==1
    ENDIF
#endif
    DO I=1,9
       PIPSVIR(I)=ZERO
    ENDDO
    PIPSVIR(1)=VIRIPS
    PIPSVIR(5)=VIRIPS
    PIPSVIR(9)=VIRIPS
    !write(*,*)'EEXIPS0= ',mynod,SRIPS,FIPS,EIPSSNB,EIPSSEL, ENB,EEL
#if KEY_DOMDEC==1
    if (q_domdec) RETURN
#endif
    !IF(QPSSP)RETURN
    !write(*,*)'EEXIPS= ',mynod,FIPS,EIPSSNB,EIPSSEL, ENB,EEL
    !  Evenly split long rang correction to each atom
    !=======================================================================
    !
    !=======================================================================
    !   Main loop begin
    !=======================================================================
    ILAST = 0
    IF(IFRSTA > 1)ILAST=IBLO(IFRSTA-1)
    DO I=IFRSTA,ILASTA
       !        define index list J
       IFIRST = ILAST+1
       ILAST=IBLO(I)
       IF(LELECX)THEN
          IF(LCONS)THEN
            CGI=CG(I)*CGF*RIPSR
          ELSE
             CGI=CG(I)*CGF*RIPS2R
          ENDIF
          IF(QECONTX) THEN
            CGIJ=CG(I)*CGI
            ! charge self iteraction.  Already calculated
            EELIJ=HALF*CGIJ*PIPSE0
            ECONTX(I)=ECONTX(I)+EELIJ
            EEL=EEL+EELIJ
         ENDIF
       ENDIF
       IF(LVDWX)THEN
          ITI=ITC(IAC(I))
          IC=ITI*(ITI-1)/2+ITI
          AIJ=CNBB(IC)
          ! half self interaction. 
          ENBIJ=HALF*AIJ*(AIPSVA(1)*PIPSVA0-AIPSVC(1)*PIPSVC0)
          ENB=ENB+ENBIJ
          IF(QECONTX) THEN
            ECONTX(I)=ECONTX(I)+ENBIJ
         ENDIF
       ENDIF
       DXI=DX(I)
       DYI=DY(I)
       DZI=DZ(I)
       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
       loop100: DO K=IFIRST,ILAST
          J = INB(K)
          IF(J < 0)THEN
             J=-J
             ESCALE=E14FAC
             VSCALE=ONE
             LJROW=MAXROW
          ELSE
             ESCALE=ZERO
             VSCALE=ZERO
             LJROW=0
          ENDIF
          XIJ=XI-X(J)
          YIJ=YI-Y(J)
          ZIJ=ZI-Z(J)
          R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
          R1=SQRT(R2)
          U2=R2*RIPS2R
          U1=R1*RIPSR
          IF(LELECX)THEN
             !  Electrostatic IPS
             !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
             !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
             !
             CGIJ=CGI*CG(J)
             IF(LCONS)THEN
                PE=ESCALE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                   +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                DPE=-ESCALE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                   +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
             ELSE
                PE=ESCALE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                   +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                DPE=-ESCALE/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                   +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
             ENDIF
             EELIJ=CGIJ*(PE-PIPSEC)
             EEL=EEL+EELIJ
             DF=-CGIJ*DPE/R2
          ELSE
             DF=ZERO
             DDF=ZERO
          ENDIF
          IF(LVDWX)THEN
             ITJ=ITC(IAC(J))
             IF(ITI > ITJ)THEN
                IC=ITI*(ITI-1)/2+ITJ
             ELSE
                IC=ITJ*(ITJ-1)/2+ITI
             ENDIF
             SIG2=CNBA(IC)
             IF(SIG2>RSMALL)THEN
             SIG=SQRT(SIG2)
             AIJ=CNBB(IC)
             ICX=IC+LJROW
             SIG2=CNBA(ICX)
             IF(SIG2>RSMALL)THEN
             SIGEX=SQRT(SIG2)            
             AEXIJ=VSCALE*CNBB(ICX)
             IF(R1<SRIPS*SIGEX)THEN
               AKR=AIPSVA(0)*R1/SIGEX
               !AKRC=AIPSVA(0)*RIPS/SIG
               EXPAR=EXP(-AKR)
               !EXPARC=EXP(-AKRC)
               EXPAREX=EXP(-AIPSVA(0)*R1/SIGEX)
               ! energy
               ENBIJ=AIPSVA(1)*(AEXIJ*EXPAREX+AIJ*(AIPSVA(4)/EXPAR+AIPSVA(5)))
               ! force 
               DFIJ=BIPSVA(1)*(AEXIJ*EXPAREX/SIGEX-AIJ*(AIPSVA(4)/EXPAR/SIG))
               CKR=AIPSVC(0)*R1/SIG
               !CKRC=AIPSVC(0)*RIPS/SIG
               EXPCR=EXP(-CKR)
               !EXPCRC=EXP(-CKRC)
               EXPCREX=EXP(-AIPSVC(0)*R1/SIGEX)
               ! energy
               ENBIJ=ENBIJ-AIPSVC(1)*(AEXIJ*EXPCREX+AIJ*(AIPSVC(4)/EXPCR+AIPSVC(5)))
               ! force
               DFIJ=DFIJ-BIPSVC(1)*(AEXIJ*EXPCREX/SIGEX-AIJ*(AIPSVC(4)/EXPCR/SIG))
               ENB=ENB+ENBIJ
               DF=DF-DFIJ/R1
            ENDIF
            ENDIF
            ENDIF
          ENDIF
          ! Calculate total forces
          DFX=DF*XIJ
          DFY=DF*YIJ
          DFZ=DF*ZIJ
          DXI=DXI-DFX
          DYI=DYI-DFY
          DZI=DZI-DFZ
          DX(J)=DX(J)+DFX
          DY(J)=DY(J)+DFY
          DZ(J)=DZ(J)+DFZ
         !
       enddo loop100
       DX(I) =  DXI
       DY(I) =  DYI
       DZ(I) =  DZI
    ENDDO
    !=======================================================================
    !   Main loop end
    !=======================================================================
    !write(*,*)'EEXIPS1= ',mynod,FIPS,EIPSSNB,EIPSSEL, ENB,EEL
    RETURN
  END SUBROUTINE EEXIPSEXP



  SUBROUTINE EIPSSYS(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC, &
       LVDW,LELEC,LVIPS,LEIPS,LCONS,LDBEXP,INB,IBLO,  &
       CG,RSCLF,CNBA,CNBB,IAC,ITC,MAXROW, &
       EPS,E14FAC,VBOX,CGS,CG2S,AIIS,CIIS,AIJS,CIJS, &
       DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,ICALL)
    !
    !   3D IPS interaction between excluded atom pairs
    !   This routine must be called first to update IPS parameters when needed
    !    and to inital electrostatic and vdw energies
    !
    !
    ! input/output
    use dimens_fcm
    use consta
    use number
    use stream
    use nbips
    use image
    use parallel
  use sftc          !Cc New PBLOCK
#if KEY_PERT==1
  use pert
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
#endif
    real(chm_real) ENB, EEL
    INTEGER IFRSTA,ILASTA,NATOM, NATC
    INTEGER IBLO(*),INB(*),ICALL
    real(chm_real) CG(*),RSCLF(*),CNBA(*), CNBB(*)
    INTEGER IAC(*), ITC(*),MAXROW
    real(chm_real)  EPS, E14FAC,VBOX,FIPS
    real(chm_real)  CGS,CG2S,AIIS,CIIS,AIJS,CIJS
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*)
    LOGICAL QECONTX
    real(chm_real) ECONTX(*)
    LOGICAL LVDW,LELEC,LVIPS,LEIPS,LCONS,LDBEXP
    LOGICAL LLJIPSX,LELECX,LDEIPSX
    real(chm_real)  DD1(*)
    INTEGER IUPT(*)
    LOGICAL LSECD
     ! local
    INTEGER I,J,K,IC,ICX,ITI,ITJ,II,JJ,IADD
    INTEGER  ILAST, IFIRST,LJROW
    real(chm_real) ESCALE,VSCALE
    real(chm_real) DXI, DYI, DZI,DF,DFX,DFY,DFZ
    real(chm_real) ENBIJ,EELIJ,ECONTI
    real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,R2
    real(chm_real)  CGF,CGI,CGIJ,AIJ,CIJ,AEXIJ,CEXIJ,SIG2,SIG6,SIG12
    real(chm_real)  U1,U2,U4,U6R,U12R
    real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
    real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,DDF
    real(chm_real)  SIG,AKR,CKR,EXPAR,EXPCR,ENBA,ENBC,DFA,DFC,DFIJ
    real(chm_real)  SIGEX,EXPAREX,EXPCREX,SRIPS2
#if KEY_PERT==1
  !     REACTANT AND PRODUCT PART CALCULATED BY
  !     BY SEPARATE ROUTINE (SEE EPERT.SRC)
  !c
#if KEY_BLOCK==1
  IF (TQPSSP.or.QBPSSP) THEN
#else /**/
  IF (TQPSSP) THEN
#endif 
        CALL  ESSIPSPERT(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC,  &
             LVDW,LELEC,LVIPS,LEIPS,LCONS,LDBEXP,INB,IBLO, &
             CG,RSCLF,CNBA,CNBB,IAC,ITC,MAXROW,&
             EPS,E14FAC,VBOX,CGS,CG2S,AIIS,CIIS,AIJS,CIJS,  &
             DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,ICALL)
     !
     RETURN
  ENDIF
#endif /* KEY_PERT */
    ! begin
    LLJIPSX=LVDW.AND.LVIPS.AND..NOT.LDBEXP
    LDEIPSX=LVDW.AND.LVIPS.AND.LDBEXP
    LELECX=LELEC.AND.LEIPS
    CGF=CCELEC/EPS
    SRIPS2=SRIPS*SRIPS
    QIPSUPD=MOD(NIPSCNT,NIPSFRQ) == 0
    NIPSCNT=NIPSCNT+ICALL
    !
    ! check to see if IPS parameters need to be updated

    IF (QIPSINI )THEN
       QIPSFIN=QIPSFIN.OR.NTRANS == 0
       QIPSINI=.FALSE.
    ENDIF
    !
    IF(QIPSFIN)THEN
      EIPSSNB=ZERO
      EIPSSEL=ZERO
      VIRIPS=ZERO
      FIPS=ONE
    ELSE
      FIPS=FOUR*PI*RIPS2*RIPS/THREE/VBOX
#if KEY_PARALLEL==1
       IF(MYNOD == 0)THEN                
#endif
          IF(QAIPS)THEN
            IF(LDEIPSX)THEN 
               call wrndie(-2,'<AIPS>', &                                      
               ' IPS/DFFT not implemented for DE IPS (add "PXYZ")!')  
            ELSE IF(LVIPS)THEN
                ! Only the vdw 12th-term is considered here
              EIPSSNB=FIPS*AIJS*PIPSVAC*RIPS12R
            ELSE
               EIPSSNB=ZERO
            ENDIF
            EIPSSEL=ZERO
         ELSE
            IF(LDEIPSX)THEN 
               EIPSSNB=(AIJS/VBOX)*(PIPSVAC-PIPSVCC)
            ELSE IF(LVIPS)THEN
              EIPSSNB=FIPS*(AIJS*(PIPSVAC)*RIPS6R &
                  -CIJS*(PIPSVCC))*RIPS6R
            ELSE
               EIPSSNB=ZERO
            ENDIF
            IF(LEIPS)THEN
               EIPSSEL=FIPS*CGF*CGS*CGS*PIPSEC*RIPSR/TWO
            ELSE
               EIPSSEL=ZERO
            ENDIF
          ENDIF
          IF(.NOT.LELEC)EIPSSEL=ZERO
          IF(.NOT.LVDW)EIPSSNB=ZERO
#if KEY_PARALLEL==1
       ELSE
          EIPSSNB=ZERO
          EIPSSEL=ZERO
       ENDIF
#endif 
       VIRIPS=(EIPSSNB+EIPSSEL)
    ENDIF
    ! apply option for netcharge boundary contribution
    IF(.NOT.QIPSNETCG)THEN
      !  ignoring net charge boudnary ele ips energy
      EIPSSEL=ZERO
    ENDIF 
    !
    ENB=EIPSSNB
    EEL=EIPSSEL
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN                
#endif
         ! add self-term
         IF(LVDW)THEN
            IF(LDEIPSX)THEN 
               ENB=EIPSSNB+HALF*AIIS*(PIPSVA0-PIPSVC0)
            ELSE IF(LVIPS)THEN
               ENB=EIPSSNB+HALF*(AIIS*PIPSVA0*rips6r-CIIS*PIPSVC0)*rips6r
            ENDIF
         ENDIF
         IF(LELEC.AND.LEIPS)THEN
            IF(LCONS)THEN 
               EEL=EIPSSEL+HALF*CG2S*PIPSE0*CGF*RIPSR
            ELSE
               EEL=EIPSSEL+HALF*CG2S*PIPSE0*CGF*RIPS2R
            ENDIF
         ENDIF
#if KEY_PARALLEL==1
    ENDIF
#endif 
    DO I=1,9
       PIPSVIR(I)=ZERO
    ENDDO
    PIPSVIR(1)=VIRIPS
    PIPSVIR(5)=VIRIPS
    PIPSVIR(9)=VIRIPS
#if KEY_DOMDEC==1
    if (q_domdec) RETURN
#endif
!IF(QPSSP)RETURN
    !  Evenly split long rang correction to each atom
    !=======================================================================
    !
    !=======================================================================
    !   Main loop begin
    !=======================================================================
    ILAST = 0
    EELIJ=ZERO
    ENBIJ=ZERO
    IF(IFRSTA > 1)ILAST=IBLO(IFRSTA-1)
    DO I=IFRSTA,ILASTA
       !        define index list J
       IFIRST = ILAST+1
       ILAST=IBLO(I)
       IF(LELECX)THEN
          IF(LCONS)THEN
            CGI=CG(I)*CGF*RIPSR
          ELSE
            CGI=CG(I)*CGF*RIPS2R
          ENDIF
          IF(QECONTX) THEN
            CGIJ=CG(I)*CGI
            ! charge self iteraction.  Already calculated
            EELIJ=HALF*CGIJ*PIPSE0
            ECONTX(I)=ECONTX(I)+EELIJ
          ENDIF
       ENDIF 
       IF(LVIPS)THEN
          ITI=ITC(IAC(I))
          IF(QECONTX) THEN
            IC=ITI*(ITI-1)/2+ITI
            IF(LDEIPSX)THEN 
               AIJ=CNBB(IC)
               ENBIJ=HALF*AIJ*(PIPSVA0-PIPSVC0)
            ELSE
              SIG2=CNBA(IC)*RIPS2R
              SIG6=SIG2*SIG2*SIG2
              SIG12=SIG6*SIG6
              AIJ=CNBB(IC)*SIG12
              CIJ=TWO*CNBB(IC)*SIG6
              ! Atom i long range referece and self interaction. Already calculated
              ENBIJ=HALF*(AIJ*PIPSVA0-CIJ*PIPSVC0)
            ENDIF
            ECONTX(I)=ECONTX(I)+ENBIJ
          ENDIF
       ENDIF
       !
       DXI=DX(I)
       DYI=DY(I)
       DZI=DZ(I)
       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
       loop100: DO K=IFIRST,ILAST
          J = INB(K)
          ! skip 1-4 pairs while will be calculated in nonbond routines
          IF(J < 0)THEN
            J=-J
            ESCALE=E14FAC
            VSCALE=ONE
            LJROW=MAXROW
         ELSE
            ESCALE=ZERO
            VSCALE=ZERO
            LJROW=0
         ENDIF
          XIJ=XI-X(J)
          YIJ=YI-Y(J)
          ZIJ=ZI-Z(J)
          R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
          U2=R2*RIPS2R
          IF(LELECX)THEN
             !  Electrostatic IPS
             !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
             !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
             ! 
             CGIJ=CGI*CG(J)
             IF(LCONS)THEN
               U1=SQRT(U2)
               PE=ESCALE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                  +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
               DPE=-ESCALE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                  +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
            ELSE
               PE=ESCALE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                  +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
               DPE=-ESCALE/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                  +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
            ENDIF
            EELIJ=CGIJ*(PE-PIPSEC)
            EEL=EEL+EELIJ
            DF=-CGIJ*DPE/R2
            !debugwxw
            !if(j<5)write(*,*)"DEBUGWXW1: ",j,escale,cgij,u1,pe,dpe,eelij,df,eel
            IF(LSECD) THEN
               IF(LCONS)THEN
                  DDPE=TWO*ESCALE/U1+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                     +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
               ELSE
                  DDPE=SIX*ESCALE/U2+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                     +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
               ENDIF
               DDF=CGIJ*DDPE/R2/R2
            ENDIF
         ELSE
            DF=ZERO
            DDF=ZERO
          ENDIF
          IF(LLJIPSX)THEN
             ITJ=ITC(IAC(J))
             IF(ITI > ITJ)THEN
                IC=ITI*(ITI-1)/2+ITJ
             ELSE
                IC=ITJ*(ITJ-1)/2+ITI
             ENDIF

             SIG2=CNBA(IC)*RIPS2R
             SIG6=SIG2*SIG2*SIG2
             SIG12=SIG6*SIG6
             AIJ=CNBB(IC)*SIG12
             CIJ=TWO*CNBB(IC)*SIG6
             ICX=IC+LJROW
             SIG2=CNBA(ICX)*RIPS2R
             SIG6=SIG2*SIG2*SIG2
             SIG12=SIG6*SIG6
             AEXIJ=CNBB(ICX)*SIG12
             CEXIJ=TWO*CNBB(ICX)*SIG6
             ! debuging domdec
             !  AIJ=AEXIJ
             !  CIJ=CEXIJ
               AEXIJ=VSCALE*AEXIJ
               CEXIJ=VSCALE*CEXIJ
             !
             U4=U2*U2
             U6R=ONE/U4/U2
             U12R=U6R*U6R
             !  L-J r6 term
             !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
             !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
             !
             PVC=CEXIJ*U6R+CIJ*(AIPSVC(0)+U2*(AIPSVC(1) &
                  +U2*(AIPSVC(2)+U2*(AIPSVC(3)+U2*(AIPSVC(4) &
                  +U4*(AIPSVC(5)+U4*AIPSVC(6))))))-PIPSVCC)
             DPVC=-SIX*CEXIJ*U6R+CIJ*(U2*(BIPSVC(1) &
                  +U2*(BIPSVC(2)+U2*(BIPSVC(3)+U2*(BIPSVC(4) &
                  +U4*(BIPSVC(5)+U4*BIPSVC(6)))))))
             !  L-J r12 term 
             !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
             !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
             !
             PVA=AEXIJ*U12R+AIJ*(AIPSVA(0)+U2*(AIPSVA(1) &
                  +U2*(AIPSVA(2)+U2*(AIPSVA(3)+U4*(AIPSVA(4) &
                  +U4*(AIPSVA(5)+U4*AIPSVA(6))))))-PIPSVAC)
             DPVA=-TWELVE*AEXIJ*U12R+AIJ*(U2*(BIPSVA(1) &
                  +U2*(BIPSVA(2)+U2*(BIPSVA(3)+U4*(BIPSVA(4) &
                  +U4*(BIPSVA(5)+U4*BIPSVA(6)))))))
             ENBIJ=PVA-PVC
             ENB=ENB+ENBIJ
             DF=DF-(DPVA-DPVC)/R2
             IF(LSECD) THEN
              DDPVC=42.0D0*CEXIJ*U6R+CIJ*U2*(BBIPSVC(1)+U2*(BBIPSVC(2)+U2*(BBIPSVC(3) &
                    +U2*(BBIPSVC(4)+U4*(BBIPSVC(5)+U4*BBIPSVC(6))))))
              DDPVA=156.0D0*AEXIJ*U12R+AIJ*U2*(BBIPSVA(1)+U2*(BBIPSVA(2)+U2*(BBIPSVA(3) &
                    +U4*(BBIPSVA(4)+U4*(BBIPSVA(5)+U4*BBIPSVA(6))))))
              DDF=DDF+(DDPVA-DDPVC)/R2/R2
             ENDIF
            ELSE IF(LDEIPSX)THEN
              ITJ=ITC(IAC(J))
              IF(ITI > ITJ)THEN
                 IC=ITI*(ITI-1)/2+ITJ
              ELSE
                 IC=ITJ*(ITJ-1)/2+ITI
              ENDIF
              SIG2=CNBA(IC)
               IF(SIG2>RSMALL.AND.R2<SRIPS2*SIG2)THEN
                 SIG=SQRT(SIG2/R2)
                 AIJ=CNBB(IC)
                 ICX=IC+LJROW
                 SIGEX=SQRT(CNBA(ICX)/R2)
                 AEXIJ=VSCALE*CNBB(ICX)
                 AKR=AIPSVA(0)/SIG
                 EXPAR=EXP(-AKR)
                 EXPAREX=EXP(-AIPSVA(0)/SIGEX)
                 ! energy
                 ENBA=AEXIJ*AIPSVA(1)*EXPAREX+AIJ*AIPSVA(1)*(AIPSVA(4)/EXPAR+AIPSVA(5))
                 ! force 
                 DFA=AEXIJ*BIPSVA(1)*EXPAREX/SIGEX-AIJ*BIPSVA(1)*(AIPSVA(4)/EXPAR/SIG)
                 CKR=AIPSVC(0)/SIG
                 !CKRC=AIPSVC(0)*RIPS/SIG
                 EXPCR=EXP(-CKR)
                 EXPCREX=EXP(-AIPSVC(0)/SIGEX)
                 ! energy
                 ENBC=AEXIJ*AIPSVC(1)*EXPCREX+AIJ*AIPSVC(1)*(AIPSVC(4)/EXPCR+AIPSVC(5))
                 ENBIJ=ENBA-ENBC
                 ! force
                 DFC=AEXIJ*BIPSVC(1)*EXPCREX/SIGEX-AIJ*BIPSVC(1)*(AIPSVC(4)/EXPCR/SIG)
                 DFIJ=(DFA-DFC)/R2
                 ENB=ENB+ENBIJ
                 DF=DF-DFIJ
               ENDIF 
            ENDIF
          ! Calculate total forces
          DFX=DF*XIJ
          DFY=DF*YIJ
          DFZ=DF*ZIJ
          DXI=DXI-DFX
          DYI=DYI-DFY
          DZI=DZI-DFZ
          DX(J)=DX(J)+DFX
          DY(J)=DY(J)+DFY
          DZ(J)=DZ(J)+DFZ
           !
          IF(QECONTX) THEN
             ECONTI=HALF*(EELIJ+ENBIJ)
             ECONTX(I)=ECONTX(I)+ECONTI
             ECONTX(J)=ECONTX(J)+ECONTI
          ENDIF
          IF(LSECD) THEN
            !
               !
               DDF=DDF+DF/R2
               !
               !     NOW UPDATE DERIVATIVE MATRICIES
               !
               AXX=XIJ*XIJ*DDF-DF
               AYY=YIJ*YIJ*DDF-DF
               AZZ=ZIJ*ZIJ*DDF-DF
               AXY=XIJ*YIJ*DDF
               AXZ=XIJ*ZIJ*DDF
               AYZ=YIJ*ZIJ*DDF
                  II=3*I-2
                  JJ=3*J-2
                  !
                  IADD=IUPT(II)+II
                  DD1(IADD)=DD1(IADD)+AXX
                  IADD=IUPT(II+1)+II+1
                  DD1(IADD)=DD1(IADD)+AYY
                  IADD=IUPT(II+2)+II+2
                  DD1(IADD)=DD1(IADD)+AZZ
                  IADD=IUPT(II)+II+1
                  DD1(IADD)=DD1(IADD)+AXY
                  IADD=IUPT(II)+II+2
                  DD1(IADD)=DD1(IADD)+AXZ
                  IADD=IUPT(II+1)+II+2
                  DD1(IADD)=DD1(IADD)+AYZ
                  !
                  IADD=IUPT(JJ)+JJ
                  DD1(IADD)=DD1(IADD)+AXX
                  IADD=IUPT(JJ+1)+JJ+1
                  DD1(IADD)=DD1(IADD)+AYY
                  IADD=IUPT(JJ+2)+JJ+2
                  DD1(IADD)=DD1(IADD)+AZZ
                  IADD=IUPT(JJ)+JJ+1
                  DD1(IADD)=DD1(IADD)+AXY
                  IADD=IUPT(JJ)+JJ+2
                  DD1(IADD)=DD1(IADD)+AXZ
                  IADD=IUPT(JJ+1)+JJ+2
                  DD1(IADD)=DD1(IADD)+AYZ
                  !
                  IF (JJ < II) THEN
                     IADD=IUPT(JJ)+II
                     DD1(IADD)=DD1(IADD)-AXX
                     IADD=IUPT(JJ+1)+II+1
                     DD1(IADD)=DD1(IADD)-AYY
                     IADD=IUPT(JJ+2)+II+2
                     DD1(IADD)=DD1(IADD)-AZZ
                     IADD=IUPT(JJ)+II+1
                     DD1(IADD)=DD1(IADD)-AXY
                     IADD=IUPT(JJ+1)+II
                     DD1(IADD)=DD1(IADD)-AXY
                     IADD=IUPT(JJ)+II+2
                     DD1(IADD)=DD1(IADD)-AXZ
                     IADD=IUPT(JJ+2)+II
                     DD1(IADD)=DD1(IADD)-AXZ
                     IADD=IUPT(JJ+1)+II+2
                     DD1(IADD)=DD1(IADD)-AYZ
                     IADD=IUPT(JJ+2)+II+1
                     DD1(IADD)=DD1(IADD)-AYZ
                  ELSE
                     IADD=IUPT(II)+JJ
                     DD1(IADD)=DD1(IADD)-AXX
                     IADD=IUPT(II+1)+JJ+1
                     DD1(IADD)=DD1(IADD)-AYY
                     IADD=IUPT(II+2)+JJ+2
                     DD1(IADD)=DD1(IADD)-AZZ
                     IADD=IUPT(II+1)+JJ
                     DD1(IADD)=DD1(IADD)-AXY
                     IADD=IUPT(II)+JJ+1
                     DD1(IADD)=DD1(IADD)-AXY
                     IADD=IUPT(II+2)+JJ
                     DD1(IADD)=DD1(IADD)-AXZ
                     IADD=IUPT(II)+JJ+2
                     DD1(IADD)=DD1(IADD)-AXZ
                     IADD=IUPT(II+2)+JJ+1
                     DD1(IADD)=DD1(IADD)-AYZ
                     IADD=IUPT(II+1)+JJ+2
                     DD1(IADD)=DD1(IADD)-AYZ
                  ENDIF
                  !
            !
         ENDIF  ! (LSECD)
enddo loop100
       DX(I) =  DXI
       DY(I) =  DYI
       DZ(I) =  DZI
    ENDDO
    !=======================================================================
    !   Main loop end
    !=======================================================================
    RETURN
  END SUBROUTINE EIPSSYS

  SUBROUTINE EIPSSYSEXP(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC, &
       LVDW,LELEC,LVIPS,LEIPS,LCONS,INB,IBLO,CG,CNBA,CNBB,IAC,ITC,MAXROW, &
       EPS,E14FAC,VBOX,CGS,CG2S,AIIS,CIIS,AIJS,CIJS, &
       X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,ICALL)
    !
    !   3D IPS interaction between excluded atom pairs
    !   This routine must be called first to update IPS parameters when needed
    !    and to inital electrostatic and vdw energies
    !
    !
    ! input/output
    use dimens_fcm
    use consta
    use number
    use stream
    use nbips
    use image
    use parallel
    use pert,only:qpert,qpssp
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
#endif
    real(chm_real) ENB, EEL
    INTEGER IFRSTA,ILASTA,NATOM, NATC
    INTEGER IBLO(*),INB(*),ICALL
    real(chm_real) CG(*),CNBA(*), CNBB(*)
    INTEGER IAC(*), ITC(*),MAXROW
    real(chm_real)  EPS, E14FAC,VBOX,FIPS
    real(chm_real)  CGS,CG2S,AIIS,CIIS,AIJS,CIJS
    real(chm_real) X(*),Y(*),Z(*)
    LOGICAL QECONTX
    real(chm_real) ECONTX(*)
    LOGICAL LVDW,LELEC,LVDWX,LELECX,LVIPS,LEIPS,LCONS
    real(chm_real)  DD1(*)
    INTEGER IUPT(*)
    LOGICAL LSECD
     ! local
    INTEGER I,J,K,IC,ICX,ITI,ITJ,II,JJ,IADD
    INTEGER  ILAST, IFIRST,LJROW
    real(chm_real) ESCALE,VSCALE
    real(chm_real) DXI, DYI, DZI,DF,DFX,DFY,DFZ
    real(chm_real) ENBIJ,EELIJ,ECONTI
    real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,R1,R2
    real(chm_real)  CGF,CGI,CGIJ,AIJ,CIJ,AEXIJ,CEXIJ,SIG2,SIG6,SIG12
    real(chm_real)  U1,U2,EXPAREX,EXPCREX,DFIJ
    real(chm_real)  SIG,SIGEX,AKR,AKRC,CKR,CKRC,EXPAR,EXPCR,EXPARC,EXPCRC
    real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
    real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,DDF
    ! begin
    LVDWX=LVDW.AND.LVIPS
    LELECX=LELEC.AND.LEIPS
    CGF=CCELEC/EPS
    QIPSUPD=MOD(NIPSCNT,NIPSFRQ) == 0
    NIPSCNT=NIPSCNT+ICALL
    !
    ! check to see if IPS parameters need to be updated
    IF(QIPSINI)THEN
       QIPSFIN=QIPSFIN.OR.NTRANS == 0
       ! full DE calculation no need precalculation parameters
       QIPSINI=.FALSE.
    ENDIF
    !
    IF(QIPSFIN)THEN
       !EIPSSNB=ZERO
       !EIPSSEL=ZERO
       !VIRIPS=ZERO
      FIPS=ONE
    ELSE
       FIPS=FOUR*PI*RIPS2*RIPS/THREE/VBOX
    ENDIF
#if KEY_PARALLEL==1
       IF(MYNOD == 0)THEN
#endif
          IF(QAIPS)THEN
            call wrndie(-2,'<AIPS>', &                                      
            ' IPS/DFFT not implemented for COLFFT--Use 3D IPS (add "PXYZ")!')  
          ELSE
            ! boundary energies: Ec=Sum(ecij)Vc/Vbox-Sum(ecii)/2
            !EIPSSNB=(AIJSUM/VBOX-HALF*A2SUM)*(AIPSVA(1)*PIPSVAC-AIPSVC(1)*PIPSVCC)
            EIPSSNB=(AIJS/VBOX)*(AIPSVA(1)*PIPSVAC-AIPSVC(1)*PIPSVCC)
            EIPSSEL=FIPS*CGF*CGS*CGS*PIPSEC*RIPSR/TWO
          ENDIF
          IF(.NOT.LELEC)EIPSSEL=ZERO
          IF(.NOT.LVDW)EIPSSNB=ZERO
#if KEY_PARALLEL==1
       ELSE
          EIPSSNB=ZERO
          EIPSSEL=ZERO
       ENDIF
#endif
    !ENDIF
    ! apply option for netcharge boundary contribution
    IF(.NOT.QIPSNETCG)THEN
      !  ignoring net charge boudnary ele ips energy
      EIPSSEL=ZERO
    ENDIF
    !
    VIRIPS=(EIPSSNB+EIPSSEL)
    ENB=EIPSSNB
    EEL=EIPSSEL
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN
#endif
         ! add self-term
         ! self term is calculated with full DB potential
         IF(LVDW)ENB=EIPSSNB+HALF*AIIS*(AIPSVA(1)*PIPSVA0-AIPSVC(1)*PIPSVA0)
         IF(LELEC)EEL=EIPSSEL+HALF*CG2S*PIPSE0*CGF*RIPSR
#if KEY_PARALLEL==1
    ENDIF
#endif
    DO I=1,9
       PIPSVIR(I)=ZERO
    ENDDO
    PIPSVIR(1)=VIRIPS
    PIPSVIR(5)=VIRIPS
    PIPSVIR(9)=VIRIPS
    !write(*,*)'EEXIPS0= ',mynod,SRIPS,FIPS,EIPSSNB,EIPSSEL, ENB,EEL
#if KEY_DOMDEC==1
    if (q_domdec) RETURN
#endif
    !IF(QPSSP)RETURN
    !write(*,*)'EEXIPS= ',mynod,FIPS,EIPSSNB,EIPSSEL, ENB,EEL
    !  Evenly split long rang correction to each atom
    !=======================================================================
    !
    !=======================================================================
    !   Main loop begin
    !=======================================================================
    ILAST = 0
    IF(IFRSTA > 1)ILAST=IBLO(IFRSTA-1)
    DO I=IFRSTA,ILASTA
       !        define index list J
       IFIRST = ILAST+1
       ILAST=IBLO(I)
       IF(LELECX)THEN
          IF(LCONS)THEN
            CGI=CG(I)*CGF*RIPSR
          ELSE
             CGI=CG(I)*CGF*RIPS2R
          ENDIF
          IF(QECONTX) THEN
            CGIJ=CG(I)*CGI
            ! charge self iteraction.  Already calculated
            EELIJ=HALF*CGIJ*PIPSE0
            ECONTX(I)=ECONTX(I)+EELIJ
            !EEL=EEL+EELIJ
         ENDIF
       ENDIF
       IF(LVDWX)THEN
          ITI=ITC(IAC(I))
          IC=ITI*(ITI-1)/2+ITI
          AIJ=CNBB(IC)
          ! half self interaction. 
          ENBIJ=HALF*AIJ*(AIPSVA(1)*PIPSVA0-AIPSVC(1)*PIPSVC0)
          !ENB=ENB+ENBIJ
          IF(QECONTX) THEN
            ECONTX(I)=ECONTX(I)+ENBIJ
         ENDIF
       ENDIF
       loop100: DO K=IFIRST,ILAST
          J = INB(K)
          IF(J < 0)CYCLE loop100
          IF(LELECX)THEN
             !  Electrostatic IPS
             !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
             !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
             !
             CGIJ=CGI*CG(J)
             PE=AIPSE(0)
             EELIJ=CGIJ*(PE-PIPSEC)
             EEL=EEL+EELIJ
          ENDIF
          IF(LVDWX)THEN
             ITJ=ITC(IAC(J))
             IF(ITI > ITJ)THEN
                IC=ITI*(ITI-1)/2+ITJ
             ELSE
                IC=ITJ*(ITJ-1)/2+ITI
             ENDIF
             AIJ=CNBB(IC)
             ! energy
             ENBIJ=AIJ*(AIPSVA(1)*(AIPSVA(4)+AIPSVA(5))-AIPSVC(1)*(AIPSVC(4)+AIPSVC(5)))
             ENB=ENB+ENBIJ
          ENDIF
         !
       enddo loop100
    ENDDO
    !=======================================================================
    !   Main loop end
    !=======================================================================
    !write(*,*)'EEXIPS1= ',mynod,FIPS,EIPSSNB,EIPSSEL, ENB,EEL
    RETURN
  END SUBROUTINE EIPSSYSEXP


  
SUBROUTINE IPSTERMS(NATOM,NATC,CG, &
   CNBA,CNBB,IAC,ITC,CGS,CG2S,AIIS,CIIS,AIJS,CIJS)
!-----------------------------------------------------------------------
!    Calculate boundary energies for 3D IPS systems
!
use dimens_fcm
use number
use consta
use stream
use image
use memory
use nbips
#if KEY_PARALLEL==1 /*pll*/
use parallel
#endif /* (pll)*/
INTEGER NATOM,NATC
real(chm_real) CG(*),CNBA(*), CNBB(*)
INTEGER IAC(*), ITC(*)
INTEGER, PARAMETER :: MXTYPE=500
INTEGER,allocatable,dimension(:) :: NSUMIT
real(chm_real),allocatable,dimension(:) :: CISUM
real(chm_real)  CISUMI
INTEGER I,J,ITI,ITJ,ITMAX,IC,NITI,NITJ,ITEMP
real(chm_real)  CGI,CGIJ,AIJ,CIJ,ANIJ
real(chm_real)  SIG,SIG2,SIG6,SIG12
real(chm_real)  CGS,CG2S,AIIS,CIIS,AIJS,CIJS
!
call chmalloc('enbips.src','IPSSYS','nsumit',natc,intg=nsumit)
call chmalloc('enbips.src','IPSSYS','cisum',natc,crl=cisum)
DO ITI=1,NATC
   NSUMIT(ITI)=0
   CISUM(ITI)=ZERO
ENDDO
CGS=ZERO
CG2S=ZERO
ITMAX=0
DO I=1,NATOM
   CGS=CGS+CG(I)
   CG2S=CG2S+CG(I)*CG(I)
   ITI=ITC(IAC(I))
   IF(ITI > ITMAX)ITMAX=ITI
   NSUMIT(ITI)=NSUMIT(ITI)+1
ENDDO
IF(ITMAX > MXTYPE)STOP 'INCREASE MXTYPE IN IPSSYS3D!'
CIIS=ZERO
AIIS=ZERO
CIJS=ZERO
AIJS=ZERO
!  system energy is calculated based on all pairs
DO ITI=1,ITMAX
   IC=ITI*(ITI-1)/2+ITI
   SIG2=CNBA(IC)
   NITI=NSUMIT(ITI)
   ANIJ=NITI*NITI/TWO
   IF(QIPSEXP)THEN
      SIG=SQRT(SIG2)
      IF(SIG2>RSMALL.AND.SRIPS>RIPS/SIG)SRIPS=RIPS/SIG
      AIJ=CNBB(IC)
      AIIS=AIIS+AIJ*NITI
      AIJS=AIJS+AIJ*ANIJ*SIG*SIG2
   ELSE
     SIG6=SIG2*SIG2*SIG2
     SIG12=SIG6*SIG6
     AIJ=CNBB(IC)*SIG12
     CIJ=TWO*CNBB(IC)*SIG6
     CIIS=CIIS+CIJ*NITI
     AIIS=AIIS+AIJ*NITI
     CISUMI=CISUM(ITI)+CIJ*NITI
     CIJS=CIJS+CIJ*ANIJ
     AIJS=AIJS+AIJ*ANIJ
   ENDIF
   DO ITJ=ITI+1,ITMAX
      NITJ=NSUMIT(ITJ)
      IC=ITJ*(ITJ-1)/2+ITI
      ANIJ=NITI*NITJ
      SIG2=CNBA(IC)
      IF(QIPSEXP)THEN
         SIG=SQRT(SIG2)
         IF(SIG2>RSMALL.AND.SRIPS>RIPS/SIG)SRIPS=RIPS/SIG
         AIJ=CNBB(IC)
         AIJS=AIJS+AIJ*ANIJ*SIG*SIG2
      ELSE
         SIG6=SIG2*SIG2*SIG2
         SIG12=SIG6*SIG6
         AIJ=CNBB(IC)*SIG12
         CIJ=TWO*CNBB(IC)*SIG6
         CIJS=CIJS+CIJ*ANIJ
         AIJS=AIJS+AIJ*ANIJ
         CISUMI=CISUMI+CIJ*NITJ
         CISUM(ITJ)=CISUM(ITJ)+CIJ*NITI
      ENDIF
   ENDDO
   CISUM(ITI)=CISUMI
ENDDO
IF(QIPSEXP)THEN
   !AIIS=FOUR*PI*SRIPS**3*AIIS/THREE
   AIJS=FOUR*PI*SRIPS**3*AIJS/THREE
ENDIF
   !      CGIJSUM=CGSUM*CGSUM/TWO
!write(*,*)'IPSSYS:',mynod,cgsum,cg2sum,a2sum,aijsum,c2sum,cijsum
IF(QAIPS.AND..NOT.QIPSEXP)THEN
   if(allocated(wnb)) &
        deallocate(wnb)
   allocate(wnb(natom),stat=alloc_err)
   if(alloc_err /= 0 ) write(0,*)"unable to allocate wnb"
   DO I=1,NATOM
      ITI=ITC(IAC(I))
      !   pair interaction part
      WNB(I)=CISUM(ITI)/SQRT(CIJSUM*TWO)
   ENDDO
ENDIF
call chmdealloc('enbips.src','IPSSYS','nsumit',natc,intg=nsumit)
call chmdealloc('enbips.src','IPSSYS','cisum',natc,crl=cisum)
RETURN 
END SUBROUTINE IPSTERMS


end module aips_module


!================================================================================
!================================================================================
!         Non module routines follow
!================================================================================
!================================================================================


SUBROUTINE IPSINPUT(COMLYN,COMLEN,QIPS1,LVIPS,LEIPS,QGROU,QVGROU, &
     QEWALD,QSHIF,QFSHI,QSWIT,QFSWT,QVSWI,QVFSWT,QVSHI,QDBEXP &
#if KEY_LRVDW==1
     ,QLRVDW                                                  & 
#endif
     )
  !
  !-----------------------------------------------------------------------
  !     Set up parameters for IPS calculation 
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use consta
  use econtmod
  use fast
  use stream
  use string
  use nbips

  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  LOGICAL QIPS1,LEIPS,LVIPS
  LOGICAL QGROU,QVGROU
  LOGICAL QEWALD,QSHIF,QFSHI,QSWIT,QFSWT,QVSWI,QVFSWT,QVSHI,QDBEXP
#if KEY_LRVDW==1
  LOGICAL QLRVDW                                            
#endif
  INTEGER I
  !  PARSE commend line
  !
  !   If "IPS" is in command line, only use IPS routines for calcualtion.
  !   To combine IPS and other methods, "IPS" should not be present.
  !   IF EIPS or VIPS define using IPS method for electrostatic or L-J calculation.
  !   IF PXY (or other 2D flags) is in command line, 2D IPS will be used for 
  !       electrostatic (EIPS) or L-J (VIPS) calculation.
  !
  QIPS1=.FALSE. 
  IF(INDXA(COMLYN,COMLEN,'IPS') > 0) QIPSONLY=.TRUE.
  IF(INDXA(COMLYN,COMLEN,'FINI') > 0) QIPSFIN=.TRUE.
  QIPSNETCG=QIPSNETCG.OR.(INDXA(COMLYN,COMLEN,'NETCG') > 0)
  IF(INDXA(COMLYN,COMLEN,'PXYZ') > 0 &
       .OR. INDXA(COMLYN,COMLEN,'PXZY') > 0 &
       .OR. INDXA(COMLYN,COMLEN,'PYZX') > 0 &
       .OR. INDXA(COMLYN,COMLEN,'PYXZ') > 0 &
       .OR. INDXA(COMLYN,COMLEN,'PZXY') > 0 &
       .OR. INDXA(COMLYN,COMLEN,'PZYX') > 0)THEN
     QAIPS=.FALSE.
     NIPSFRQ=10000
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'PXY') > 0 &
       .OR. INDXA(COMLYN,COMLEN,'PYX') > 0)THEN
     IPSZ=ZERO
     QIPS2D=.TRUE. 
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'PYZ') > 0 &
       .OR. INDXA(COMLYN,COMLEN,'PZY') > 0)THEN
     IPSX=ZERO
     QIPS2D=.TRUE. 
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'PXZ') > 0 &
       .OR. INDXA(COMLYN,COMLEN,'PZX') > 0)THEN
     IPSY=ZERO
     QIPS2D=.TRUE. 
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'PX') > 0)THEN
     IPSX=ZERO
     QIPS2D=.TRUE. 
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'PY') > 0)THEN
     IPSY=ZERO
     QIPS2D=.TRUE. 
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'PZ') > 0)THEN
     IPSZ=ZERO
     QIPS2D=.TRUE. 
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'EIPS') > 0.OR.QIPSONLY)THEN
     LEIPS=.TRUE.
     QEWALD=.FALSE.
     QSHIF=.FALSE.
     QFSHI=.FALSE.
     QSWIT=.FALSE.
     QFSWT=.FALSE.
  ELSE
     IF(QEWALD.OR.QSHIF.OR.QFSHI.OR.QSWIT.OR.QFSWT)LEIPS=.FALSE.
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'VIPS') > 0.OR.QIPSONLY)THEN
     LVIPS=.TRUE.
     QVSWI=.FALSE.
     QVFSWT=.FALSE.
     QVSHI=.FALSE.
#if KEY_LRVDW==1
     QLRVDW=.FALSE.                                      
#endif
  ELSE
#if KEY_LRVDW==1
     IF(QVSWI.OR.QVFSWT.OR.QVSHI.OR.QLRVDW)LVIPS=.FALSE. 
#endif
  ENDIF
  QIPS=LEIPS.OR.LVIPS
  IF(QGROU.OR.QVGROU)QIPSONLY=.FALSE.
  !IF(.NOT.QIPS) RETURN
  IF(FASTER == 1.AND.QEWALD.AND.LVIPS)CALL WRNDIE(-2,'<IPSINPUT>', &
       'FAST must be on or off to use VIPS  with Ewald.')
  IF(QDBEXP)THEN
   QVSWI=.FALSE.
   QVFSWT=.FALSE.
   QVSHI=.FALSE.
#if KEY_LRVDW==1
   QLRVDW=.FALSE.                                      
#endif
  ENDIF
#if KEY_COLFFT==1
  IF(QIPS.AND.QAIPS)THEN
     call wrndie(-2,'<AIPS>', &                                      
     ' IPS/DFFT not implemented for COLFFT--Use 3D IPS (add "PXYZ")!')  
     QAIPS=.FALSE.
  ENDIF
#endif 
  AEXPPOT=GTRMF(COMLYN,COMLEN,'AEXPPOT',AEXPPOT)
  CEXPPOT=GTRMF(COMLYN,COMLEN,'CEXPPOT',CEXPPOT)
  IF(PRNLEV >= 2)THEN
     IF(LEIPS)WRITE(OUTU,910)

     IF(LVIPS.AND..NOT.QDBEXP)WRITE(OUTU,920)
     IF(LVIPS.AND.QDBEXP)WRITE(OUTU,921)
     IF(QIPS.AND..NOT.QAIPS)WRITE(OUTU,930)
     IF(QIPS.AND.QAIPS)WRITE(OUTU,940)
     IF(QIPSFIN)WRITE(OUTU,950)
910  FORMAT(' Use IPS for electrostatic energy ')
920  FORMAT(' Use IPS for Lennard-Jones vdw energy ')
921  FORMAT(' Use IPS for double exponential vdw energy ')
930  FORMAT(' Only 3D IPS  is used ')
940  FORMAT(' IPS/DFFT algorithm is used ')
950  FORMAT(' IPS finite form is used ')
  ENDIF
  QIPS1=QIPS
  RIPS=GTRMF(COMLYN,COMLEN,'RIPS',RIPS)
  RAIPS=GTRMF(COMLYN,COMLEN,'RAIPS',RAIPS)
  DVBIPS=GTRMF(COMLYN,COMLEN,'DVBIPS',DVBIPS)
  MIPSX=GTRMI(COMLYN,COMLEN,'MIPSX',MIPSX)
  MIPSY=GTRMI(COMLYN,COMLEN,'MIPSY',MIPSY)
  MIPSZ=GTRMI(COMLYN,COMLEN,'MIPSZ',MIPSZ)
  MIPSO=GTRMI(COMLYN,COMLEN,'MIPSO',MIPSO)
  NIPSFRQ=GTRMI(COMLYN,COMLEN,'NIPSFRQ',NIPSFRQ)
 !
  RETURN
END SUBROUTINE IPSINPUT

SUBROUTINE IPSSET(CUTNB,CTOFNB,LCONS,LVIPS,LEIPS,LDBEXP)
  !-----------------------------------------------------------------------
  !     Set up parameters for IPS calculation 
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use consta
  use parallel
  use stream
  use param
  use aips_module,only: ipsterms                
  use pert
  use psf
  use nbips
  implicit none
  real(chm_real)  CUTNB,CTOFNB,GP
  real(chm_real)  ASC,ASC2,EXPASC,EXPASC2
  LOGICAL LCONS,LVIPS,LEIPS,LDBEXP
  !
  INTEGER I
  !
#if KEY_VSCODE==1
!     mpi debug
    i=0
    do while (i==0)
      call sleep(5)
    enddo
#endif /* KEY_VSCODE */
  QIPSINI=.TRUE.
  ! vdw potential parameters
  QIPSEXP=LVIPS.AND.LDBEXP
  IF(QIPSEXP.AND.QAIPS)THEN
     call wrndie(-2,'<AIPS>', &                                      
     ' IPS/DFFT not implemented for DBEXP IPS (add "PXYZ")!')  
     QAIPS=.FALSE.
  ENDIF
  !
  IF(RIPS <= ZERO)THEN
     RIPS=CTOFNB
  ELSE IF(RIPS /= CTOFNB)THEN
     CALL WRNDIE(-2,'<IPSSET>', &
          ' CTOFNB is set to RIPS.')
     CTOFNB=RIPS
     IF(CUTNB <= RIPS)CUTNB=CTOFNB+TWO
  ENDIF
  RIPS2=RIPS*RIPS
  RIPSR=ONE/RIPS
  RIPS2R=RIPSR*RIPSR
  RIPS6R=RIPS2R*RIPS2R*RIPS2R
  RIPS12R=RIPS6R*RIPS6R
  SRIPS=RBIG
  ! Initialize IPS count
  NIPSCNT=0 
  IPSUPD=-1000
  ! calculate IPS system variables
  IF(QPERT)THEN
    CALL IPSTERMP(NATOMP,NATC,CG,CNBA,CNBB,IAC,ITC,CGSUM,CG2SUM,A2SUM,C2SUM,AIJSUM,CIJSUM,  &
                  PCGSUM,PCG2SUM,PA2SUM,PC2SUM,PAIJSUM,PCIJSUM)
    CALL IPSTERMP(NATOMP,NATC,PPCG,CNBA,CNBB,PPIAC,ITC,CGSUM,CG2SUM,A2SUM,C2SUM,AIJSUM,CIJSUM,  &
                  RCGSUM,RCG2SUM,RA2SUM,RC2SUM,RAIJSUM,RCIJSUM)
  ELSE
    CALL IPSTERMS(NATOM,NATC,CG,CNBA,CNBB,IAC,ITC,CGSUM,CG2SUM,A2SUM,C2SUM,AIJSUM,CIJSUM)
  ENDIF

  !   3D IPS parameters
  !
  !  Electrostatic IPS parameters
  !  Define IPS parameter according to input instruction
  IF(LCONS)THEN
     !
     IF(QAIPS)THEN
        AIPSE(0)=ZERO
        AIPSE(1)=FIVE*SEVEN/EIGHT/TWO
        AIPSE(2)=-THREE*SEVEN/EIGHT/TWO
        AIPSE(3)=FIVE/EIGHT/TWO
        AIPSE(4)=ZERO
        AIPSE(5)=ZERO
        AIPSE(6)=ZERO
        PIPSEC=FIVE*SEVEN/EIGHT/TWO
     ELSE
        AIPSE(0)=ZERO
        AIPSE(1)=FIVE*SEVEN/EIGHT/TWO
        AIPSE(2)=-THREE*SEVEN/EIGHT/TWO
        AIPSE(3)=FIVE/EIGHT/TWO
        AIPSE(4)=ZERO
        AIPSE(5)=ZERO
        AIPSE(6)=ZERO
        PIPSEC=FIVE*SEVEN/EIGHT/TWO
     ENDIF
     PIPSEC=ONE+AIPSE(0)+AIPSE(1)+AIPSE(2)+AIPSE(3) &
          +AIPSE(4)+AIPSE(5)+AIPSE(6)
     PIPSE0=-PIPSEC
     BIPSE(1)=TWO*AIPSE(1)
     BIPSE(2)=FOUR*AIPSE(2)
     BIPSE(3)=SIX*AIPSE(3)
     BIPSE(4)=EIGHT*AIPSE(4)
     BIPSE(5)=TEN*AIPSE(5)
     BIPSE(6)=TWELVE*AIPSE(6)
     BBIPSE(1)=BIPSE(1)
     BBIPSE(2)=THREE*BIPSE(2)
     BBIPSE(3)=FIVE*BIPSE(3)
     BBIPSE(4)=SEVEN*BIPSE(4)
     BBIPSE(5)=NINE*BIPSE(5)
     BBIPSE(6)=ELEVEN*BIPSE(6)
  ELSE
     ! distance dependent dielectric constant
     !QIPSONLY=.FALSE. 
     AIPSE(0)=ZERO
     AIPSE(1)=SIX
     AIPSE(2)=-FOUR
     AIPSE(3)=ONE
     AIPSE(4)=ZERO
     AIPSE(5)=ZERO 
     AIPSE(6)=ZERO 
     AIPSE(6)=(TWO-TWO*AIPSE(1)-FOUR*AIPSE(2)-SIX*AIPSE(3) &
          -EIGHT*AIPSE(4)-TEN*AIPSE(5))/TWELVE
     PIPSEC=FOUR
     PIPSEC=ONE+AIPSE(0)+AIPSE(1)+AIPSE(2)+AIPSE(3) &
          +AIPSE(4)+AIPSE(5)+AIPSE(6)
     PIPSE0=-PIPSEC
     BIPSE(1)=TWO*AIPSE(1)
     BIPSE(2)=FOUR*AIPSE(2)
     BIPSE(3)=SIX*AIPSE(3)
     BIPSE(4)=EIGHT*AIPSE(4)
     BIPSE(5)=TEN*AIPSE(5)
     BIPSE(6)=TWELVE*AIPSE(6)
     BBIPSE(1)=BIPSE(1)
     BBIPSE(2)=THREE*BIPSE(2)
     BBIPSE(3)=FIVE*BIPSE(3)
     BBIPSE(4)=SEVEN*BIPSE(4)
     BBIPSE(5)=NINE*BIPSE(5)
     BBIPSE(6)=ELEVEN*BIPSE(6)
  ENDIF
  IF(LDBEXP)THEN
    ! set up DE potential parameters
     ! calculate exponential potential parameters
    AIPSVA=ZERO
    AIPSVC=ZERO
    BIPSVA=ZERO
    BIPSVC=ZERO
    ASC=AEXPPOT*SRIPS
    ASC2=ASC*ASC
    EXPASC=EXP(-ASC)
    EXPASC2=EXPASC*EXPASC
    AIPSVA(0)=AEXPPOT
    AIPSVA(1)=CEXPPOT*EXP(AEXPPOT)/(AEXPPOT-CEXPPOT)
    AIPSVA(2)=ZERO
    AIPSVA(3)=ZERO
    ! exp(-2 a sc)
    AIPSVA(4)=EXPASC2
    ! -2exp(-a sc)
    AIPSVA(5)=-TWO*EXPASC
    PIPSVAC=AIPSVA(1)*EXPASC*(TWO+(TWELVE+SIX*EXPASC/ASC)/ASC2)
    PIPSVA0=AIPSVA(1)*(AIPSVA(4)+AIPSVA(5))
    ASC=CEXPPOT*SRIPS
    ASC2=ASC*ASC
    EXPASC=EXP(-ASC)
    EXPASC2=EXPASC*EXPASC
    AIPSVC(0)=CEXPPOT
    AIPSVC(1)=AEXPPOT*EXP(CEXPPOT)/(AEXPPOT-CEXPPOT)
    AIPSVC(2)=ZERO
    AIPSVC(3)=ZERO
    ! exp(-2 a sc)
    AIPSVC(4)=EXPASC2
    ! -2exp(-a sc)
    AIPSVC(5)=-TWO*EXPASC 
    PIPSVCC=AIPSVC(1)*EXPASC*(TWO+(TWELVE+SIX*EXPASC/ASC)/ASC2)
    PIPSVC0=AIPSVC(1)*(AIPSVC(4)+AIPSVC(5))
    ! derivative parameters
    BIPSVA(1)=-AEXPPOT*AIPSVA(1)
    BIPSVA(2)=AEXPPOT*AIPSVA(2)
    BIPSVA(3)=AEXPPOT*AIPSVA(3)
    BIPSVA(4)=AEXPPOT*AIPSVA(4)
    BIPSVA(5)=AEXPPOT*AIPSVA(5)
    BIPSVC(1)=-CEXPPOT*AIPSVC(1)
    BIPSVC(2)=CEXPPOT*AIPSVC(2)
    BIPSVC(3)=CEXPPOT*AIPSVC(3)
    BIPSVC(4)=CEXPPOT*AIPSVC(4)
    BIPSVC(5)=CEXPPOT*AIPSVC(5)
 ELSE
    !  L-J r6 term IPS parameters
    IF(QAIPS)THEN
     AIPSVC(0)=0.4376632
     AIPSVC(1)=0.5394569
     AIPSVC(2)=0.4012847
     AIPSVC(3)=0.2117008
     AIPSVC(4)=0.1607817
     AIPSVC(5)=0.0510378
     AIPSVC(6)=0.0091897
     PIPSVCC=2.8111148
     ! rationalized parameters
     AIPSVC(0)=109.0/249.0
     AIPSVC(1)=43.0/79.0
     AIPSVC(2)=7.0/16.0
     AIPSVC(3)=0.0
     AIPSVC(4)=999.0/2528.0
     AIPSVC(5)=0.0
     AIPSVC(6)=0.0
     AIPSVC(6)=(SIX-TWO*AIPSVC(1)-FOUR*AIPSVC(2)-SIX*AIPSVC(3) &
          -EIGHT*AIPSVC(4)-TWELVE*AIPSVC(5))/EIGHT/TWO
     PIPSVCC=1142321.0/629472.0
    ELSE
     AIPSVC(0)=0.4376632
     AIPSVC(1)=0.5394569
     AIPSVC(2)=0.4012847
     AIPSVC(3)=0.2117008
     AIPSVC(4)=0.1607817
     AIPSVC(5)=0.0510378
     AIPSVC(6)=0.0091897
     PIPSVCC=2.8111148
     ! rationalized parameters
     AIPSVC(0)=109.0/249.0
     AIPSVC(1)=43.0/79.0
     AIPSVC(2)=7.0/16.0
     AIPSVC(3)=0.0
     AIPSVC(4)=999.0/2528.0
     AIPSVC(5)=0.0
     AIPSVC(6)=0.0
     AIPSVC(6)=(SIX-TWO*AIPSVC(1)-FOUR*AIPSVC(2)-SIX*AIPSVC(3) &
          -EIGHT*AIPSVC(4)-TWELVE*AIPSVC(5))/EIGHT/TWO
     PIPSVCC=1142321.0/629472.0
    ENDIF
  PIPSVCC=ONE+AIPSVC(0)+AIPSVC(1)+AIPSVC(2)+AIPSVC(3) &
       +AIPSVC(4)+AIPSVC(5)+AIPSVC(6)
  PIPSVC0=AIPSVC(0)-PIPSVCC
  BIPSVC(1)=TWO*AIPSVC(1)
  BIPSVC(2)=FOUR*AIPSVC(2)
  BIPSVC(3)=SIX*AIPSVC(3)
  BIPSVC(4)=EIGHT*AIPSVC(4)
  BIPSVC(5)=TWELVE*AIPSVC(5)
  BIPSVC(6)=TWO*EIGHT*AIPSVC(6)
  BBIPSVC(1)=BIPSVC(1)
  BBIPSVC(2)=THREE*BIPSVC(2)
  BBIPSVC(3)=FIVE*BIPSVC(3)
  BBIPSVC(4)=SEVEN*BIPSVC(4)
  BBIPSVC(5)=ELEVEN*BIPSVC(5)
  BBIPSVC(6)=15.0D0*BIPSVC(6)
  !  L-J r12 term IPS parameters
  AIPSVA(0)=0.0063536
  AIPSVA(1)=0.0373915
  AIPSVA(2)=0.1079737
  AIPSVA(3)=0.1682118
  AIPSVA(4)=0.5965498
  AIPSVA(5)=-0.0524821
  AIPSVA(6)=0.2918502
  PIPSVAC=2.1558484
     ! rationalized parameters
  AIPSVA(0)=1.0/157.0
  AIPSVA(1)=1.0/20.0
  AIPSVA(2)=0.0
  AIPSVA(3)=4.0/9.0
  AIPSVA(4)=0.0
  AIPSVA(5)=277.0/420.0
  AIPSVA(6)=0.0
  AIPSVA(6)=(TWELVE-TWO*AIPSVA(1)-FOUR*AIPSVA(2)-SIX*AIPSVA(3) &
       -TEN*AIPSVA(4)-TWO*SEVEN*AIPSVA(5))/NINE/TWO
  PIPSVAC=114769.0/98910.0
  PIPSVAC=ONE+AIPSVA(0)+AIPSVA(1)+AIPSVA(2)+AIPSVA(3) &
       +AIPSVA(4)+AIPSVA(5)+AIPSVA(6)
  PIPSVA0=AIPSVA(0)-PIPSVAC
  BIPSVA(1)=TWO*AIPSVA(1)
  BIPSVA(2)=FOUR*AIPSVA(2)
  BIPSVA(3)=SIX*AIPSVA(3)
  BIPSVA(4)=TEN*AIPSVA(4)
  BIPSVA(5)=TWO*SEVEN*AIPSVA(5)
  BIPSVA(6)=TWO*NINE*AIPSVA(6)
  BBIPSVA(1)=BIPSVA(1)
  BBIPSVA(2)=THREE*BIPSVA(2)
  BBIPSVA(3)=FIVE*BIPSVA(3)
  BBIPSVA(4)=NINE*BIPSVA(4)
  BBIPSVA(5)=13.0D0*BIPSVA(5)
  BBIPSVA(6)=17.0D0*BIPSVA(6)
  ENDIF   !QDBEXP
  !  Print out the NBIPS parameters
  IF(PRNLEV >= 4)THEN
     WRITE(OUTU,1010)IPSX,IPSY,IPSZ
     WRITE(OUTU,1020)NIPSFRQ
     WRITE(OUTU,1030)RIPS
     IF(QAIPS)THEN
        IF(RAIPS > ZERO)THEN
           WRITE(OUTU,1040)RAIPS
        ELSE
           WRITE(OUTU,1041)
        ENDIF
     ELSE
        WRITE(OUTU,1042)
     ENDIF
     IF(LEIPS)THEN
        WRITE(OUTU,2115) PIPSEC,PIPSE0
        WRITE(OUTU,1050)(AIPSE(I),I=0,6)
     ENDIF
     IF(LDBEXP)THEN
    !  Print out the DBEXP parameters
            WRITE(OUTU,1110) AEXPPOT,CEXPPOT
            IF(QIPSEXP)THEN
               WRITE(OUTU,1025) RIPS,SRIPS
               WRITE(OUTU,1250)(AIPSVA(I),I=0,5)
               WRITE(OUTU,1251)(BIPSVA(I),I=1,5)
               WRITE(OUTU,1150)(AIPSVC(I),I=0,5)
               WRITE(OUTU,1151)(BIPSVC(I),I=1,5)
               WRITE(OUTU,2118) PIPSVAC,PIPSVA0,PIPSVCC,PIPSVC0
            ELSE
                  WRITE(OUTU,1250)(AIPSVA(I),I=0,1)
                  WRITE(OUTU,1251)(BIPSVA(I),I=1,1)
                  WRITE(OUTU,1150)(AIPSVC(I),I=0,1)
                  WRITE(OUTU,1151)(BIPSVC(I),I=1,1)
            ENDIF
1110 FORMAT (' DE exponents AEXPPOT and CEXPPOT:',2F15.6)
1025 FORMAT (' DE IPS parameters RIPS and SRIPS:',2F15.6)
1150 FORMAT("  AIPSVC:",8(1X,E14.6))
1151 FORMAT("  BIPSVC:",15X,8(1X,E14.6))
1250 FORMAT("  AIPSVA:",8(1X,E14.6))
1251 FORMAT("  BIPSVA:",15X,8(1X,E14.6))
2118 FORMAT ("DE PIPSVAC,PIPSVA0,PIPSVCC,PIPSVC0=",4E15.6)
       ELSE IF(LVIPS)THEN
        WRITE(OUTU,2116) PIPSVCC,PIPSVC0
        WRITE(OUTU,1350)(AIPSVC(I),I=0,6)
        WRITE(OUTU,2117) PIPSVAC,PIPSVA0
        WRITE(OUTU,1351)(AIPSVA(I),I=0,6)
      ENDIF
  ENDIF
2115 FORMAT (' IPS parameters for electrostatic energy '/ &
       5X,"PIPSEC,PIPSE0=",2E15.6)
2116 FORMAT (' IPS parameters for L-J   r6 term'/ &
       5X,"PIPSVCC,PIPSVC0=",2E15.6)
2117 FORMAT (' IPS parameters for L-J   r12 term'/ &
       5X,"PIPSVAC,PIPSVA0=",2E15.6)
1010 FORMAT("  Homogenous index in X, Y, Z direction:", 3(F4.0))
1020 FORMAT("  Anisotropic grid update frequency: ",I6)
1030 FORMAT("  Isotropic radius:   ",F10.7)
1040 FORMAT("  Anisotropic radius: ",F10.7)
1041 FORMAT("  Anisotropic radius is determined by box size. ")
1042 FORMAT("  System is assumed purely isotropic. ")
1050 FORMAT("  AIPSE: ",8F10.7)
1350 FORMAT("  AIPSVC:",8F10.7)
1351 FORMAT("  AIPSVA:",8F10.7)
  ! Allocate storage for system forces and energies
  IF(QAIPS)THEN
     GIPSX=MIN(RIPS/FOUR,GIPSX)
     GIPSY=MIN(RIPS/FOUR,GIPSY)
     GIPSZ=MIN(RIPS/FOUR,GIPSZ)
     !  Grid number must not be smaller than order
     IF(MIPSX > 0.AND.MIPSX < MIPSO)MIPSX=MIPSO
     IF(MIPSY > 0.AND.MIPSY < MIPSO)MIPSY=MIPSO
     IF(MIPSZ > 0.AND.MIPSZ < MIPSO)MIPSZ=MIPSO
#if KEY_PARALLEL==1
     !  Grid number must not be smaller than number of cpus
     IF(PRNLEV >= 4)THEN
        IF(MIPSX > 0.AND.MIPSX < NUMNOD)THEN
           WRITE(OUTU,1081)"MIPSX",NUMNOD
           MIPSX=NUMNOD
        ENDIF
        IF(MIPSY > 0.AND.MIPSY < NUMNOD)THEN
           WRITE(OUTU,1081)"MIPSY",NUMNOD
           MIPSY=NUMNOD
        ENDIF
        IF(MIPSZ > 0.AND.MIPSZ < NUMNOD)THEN
           WRITE(OUTU,1081)"MIPSZ",NUMNOD
           MIPSZ=NUMNOD
        ENDIF
     ENDIF
1081 FORMAT('<IPSSET> ',A5,' is set to cpu number:',I4)
#endif 
  ENDIF
  RETURN
END SUBROUTINE IPSSET



SUBROUTINE IPSINIT
  !-----------------------------------------------------------------------
  !     Set up parameters for IPS calculation 
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use consta
  use parallel
  use stream
  use nbips
  implicit none
  !
  INTEGER I 
  !
  QIPSINI=.TRUE.
  !
  QIPS=.FALSE. 
  QIPSONLY=.FALSE.
  QAIPS=.TRUE. 
  QIPS2D=.FALSE. 
  QIPSNETCG=.FALSE.
  QIPSEXP=.FALSE.
  QIPSFIN=.FALSE.
  IPSX=ONE
  IPSY=ONE
  IPSZ=ONE
  NIPSFRQ=1
  ! Initialize IPS count
  NIPSCNT=0 
  IPSUPD=-1000
  ! Allocate storage for system forces and energies
  GIPSX=THREE
  GIPSY=THREE
  GIPSZ=THREE
  VBIPS=ZERO
  !  Grid number must not be smaller than order
  MIPSO=6
  MIPSX=MIPSO
  MIPSY=MIPSO
  MIPSZ=MIPSO
  ! initial values
  RIPS=-ONE
  RAIPS=-ONE
  DVBIPS=RSMALL
  ! exp(-a*r)
  SRIPS=RBIG
  AEXPPOT=17.47D0
  CEXPPOT=4.099D0

  RETURN
END SUBROUTINE IPSINIT

SUBROUTINE ENBIPS(ENB,EEL,IFRSTA,NATOM,LVDW,LELEC,LCONS, &
     JNB,INBLO,CG,RSCLF,CNBA,CNBB,MAXROW,IAC,ITC,NATC, &
     EPS,DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD  &
#if KEY_IMCUBES==1
     ,lbycbim                        & 
#endif
     )
  !
  !   3D IPS Nbonded interaction
  !
  !
  use chm_kinds
  ! input/output
  use dimens_fcm
  use consta
  use number
  use memory
  use inbnd,only:E14FAC
  use param,only:MAXCN
  use nbips
  use sftc          !Cc New PBLOCK
#if KEY_PERT==1
  use pert
#endif
  implicit none
  real(chm_real) ENB, EEL
  INTEGER IFRSTA,NATOM
  INTEGER INBLO(*),JNB(*)
  real(chm_real) CG(*),RSCLF(*),CNBA(*), CNBB(*)
  INTEGER MAXROW
  INTEGER IAC(*),ITC(*)
  INTEGER NATC
  integer,allocatable,dimension(:) :: ioff
  real(chm_real)  EPS
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  LOGICAL LVDW,LELEC,LCONS
  real(chm_real)  DD1(*)
  INTEGER IUPT(*)
  LOGICAL LSECD
#if KEY_IMCUBES==1
  logical lbycbim                                
#endif

  ! local
  INTEGER I,J,K,IACI,IACJ,IC,IC0,ITI,ITJ,II,JJ,IADD
  INTEGER ILAST, IFIRST
  real(chm_real) DXI, DYI, DZI ,DF,DFX,DFY,DFZ
  real(chm_real) ENBIJ,EELIJ,ECONTI
  real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,R2
  real(chm_real)  CGF,CGI,CGIJ,AIJ,CIJ,SIG2,SIG6,SIG12
  real(chm_real) U1,U2,U4,U6R,U12R
  real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
  real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,DDF
  !
#if KEY_PERT==1
  !     REACTANT AND PRODUCT PART CALCULATED BY
  !     BY SEPARATE ROUTINE (SEE EPERT.SRC)
  !c
#if KEY_BLOCK==1
  IF (TQPSSP.or.QBPSSP) THEN
#else /**/
  IF (TQPSSP) THEN
#endif 
     call chmalloc('enbips.src','ENBOND','ioff',natc,intg=ioff)
     CALL ESSNBA(ENB,EEL,IFRSTA,NATOM,JNB,INBLO,CG,RSCLF,CNBA,CNBB, &
          MAXROW,IAC,ITC,NATC,IOFF,LELEC,LVDW,LCONS,.FALSE.,.FALSE.,.FALSE., &
          .FALSE.,QIPSEXP, &
          .TRUE.,.TRUE.,                                          & 
          DX,DY,DZ,X,Y,Z,RIPS,RIPS,EPS,E14FAC, &
          QECONTX,ECONTX,DD1,IUPT,LSECD)
     !
     RETURN
  ENDIF
#endif /* KEY_PERT */
  ! begin
  IF(QIPSEXP)THEN
    CALL ENBIPSEXP(ENB,EEL,IFRSTA,NATOM,LVDW,LELEC,LCONS, &
     JNB,INBLO,CG,RSCLF,CNBA,CNBB,IAC,ITC, &
     EPS,DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD  &
#if KEY_IMCUBES==1
     ,lbycbim                        & 
#endif
     )
    RETURN
  ENDIF
  !
  ! Setup constants for use in inner loops
  CGF=CCELEC/EPS
  !=======================================================================
  !
  !=======================================================================
  !   Main loop begin
  !=======================================================================
  ILAST = 0
  EELIJ=ZERO
  ENBIJ=ZERO
  DO I=IFRSTA,NATOM
#if KEY_IMCUBES==1
     if(lbycbim)ilast=inblo(I+natom)
#endif 
     !        define index list J
     IFIRST = ILAST+1
     ILAST=INBLO(I)
     IF(LCONS)THEN
       CGI=CG(I)*CGF*RIPSR
     ELSE
       CGI=CG(I)*CGF*RIPS2R
     ENDIF
     ITI=ITC(IAC(I))
     IC=ITI*(ITI-1)/2+ITI
     IACI=ITI*(ITI-1)/2
     DXI=DX(I)
     DYI=DY(I)
     DZI=DZ(I)
     XI=X(I)
     YI=Y(I)
     ZI=Z(I)
     loop100: DO K=IFIRST,ILAST
        J = JNB(K)
        IF(J <= 0) cycle loop100
        XIJ=XI-X(J)
        YIJ=YI-Y(J)
        ZIJ=ZI-Z(J)
        R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
        IF(R2 >= RIPS2) cycle loop100
        U2=R2*RIPS2R
        IF(LELEC)THEN
           !  Electrostatic IPS
           !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
           !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
           ! 
           CGIJ=CGI*CG(J)
           IF(LCONS)THEN
             U1=SQRT(U2)
             PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
             DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
           ELSE
             PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
             DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
           ENDIF
           EELIJ=CGIJ*(PE-PIPSEC)
           EEL=EEL+EELIJ
           DF=-CGIJ*DPE/R2
           IF(LSECD) THEN
              IF(LCONS)THEN
                 DDPE=TWO/U1+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                    +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                 DDF=CGIJ*DDPE/R2/R2
              ELSE
                 DDPE=SIX/U2+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                    +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
                 DDF=CGIJ*DDPE/R2/R2
              ENDIF
           ENDIF
        ELSE
           DF=ZERO
           DDF=ZERO
        ENDIF
        IF(LVDW)THEN
           !  Lennard-Jones IPS
           ITJ=ITC(IAC(J))
           IF(ITI > ITJ)THEN
            IC=ITI*(ITI-1)/2+ITJ
           ELSE
            IC=ITJ*(ITJ-1)/2+ITI
           ENDIF
           SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*RIPS2R
           SIG6=SIG2*SIG2*SIG2
           SIG12=SIG6*SIG6
           AIJ=CNBB(IC)*SIG12
           CIJ=TWO*CNBB(IC)*SIG6
           U4=U2*U2
           U6R=ONE/U4/U2
           U12R=U6R*U6R
           !  L-J r6 term
           !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
           !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
           !
           PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))
           DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
           !  L-J r12 term 
           !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
           !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
           !
           PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))
           DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
           ENBIJ=AIJ*(PVA-PIPSVAC)-CIJ*(PVC-PIPSVCC)
           ENB=ENB+ENBIJ
           DF=DF-(AIJ*DPVA-CIJ*DPVC)/R2
           IF(LSECD) THEN
              DDPVC=42.0D0*U6R+(U2*(BBIPSVC(1)+U2*(BBIPSVC(2)+U2*(BBIPSVC(3) &
                    +U2*(BBIPSVC(4)+U4*(BBIPSVC(5)+U4*BBIPSVC(6)))))))
              DDPVA=156.0D0*U12R+(U2*(BBIPSVA(1)+U2*(BBIPSVA(2)+U2*(BBIPSVA(3) &
                    +U4*(BBIPSVA(4)+U4*(BBIPSVA(5)+U4*BBIPSVA(6)))))))
              DDF=DDF+(AIJ*DDPVA-CIJ*DDPVC)/R2/R2
           ENDIF
        ENDIF
        ! Calculate total forces
        DFX=DF*XIJ
        DFY=DF*YIJ
        DFZ=DF*ZIJ
        DXI=DXI-DFX
        DYI=DYI-DFY
        DZI=DZI-DFZ
        DX(J)=DX(J)+DFX
        DY(J)=DY(J)+DFY
        DZ(J)=DZ(J)+DFZ
        IF(QECONTX) THEN
             ECONTI=HALF*(EELIJ+ENBIJ)
             ECONTX(I)=ECONTX(I)+ECONTI
             ECONTX(J)=ECONTX(J)+ECONTI
        ENDIF
                 IF(LSECD) THEN
                    !
                       !
                       DDF=DDF+DF/R2
                       !
                       !     NOW UPDATE DERIVATIVE MATRICIES
                       !
                       AXX=XIJ*XIJ*DDF-DF
                       AYY=YIJ*YIJ*DDF-DF
                       AZZ=ZIJ*ZIJ*DDF-DF
                       AXY=XIJ*YIJ*DDF
                       AXZ=XIJ*ZIJ*DDF
                       AYZ=YIJ*ZIJ*DDF
                          II=3*I-2
                          JJ=3*J-2
                          !
                          IADD=IUPT(II)+II
                          DD1(IADD)=DD1(IADD)+AXX
                          IADD=IUPT(II+1)+II+1
                          DD1(IADD)=DD1(IADD)+AYY
                          IADD=IUPT(II+2)+II+2
                          DD1(IADD)=DD1(IADD)+AZZ
                          IADD=IUPT(II)+II+1
                          DD1(IADD)=DD1(IADD)+AXY
                          IADD=IUPT(II)+II+2
                          DD1(IADD)=DD1(IADD)+AXZ
                          IADD=IUPT(II+1)+II+2
                          DD1(IADD)=DD1(IADD)+AYZ
                          !
                          IADD=IUPT(JJ)+JJ
                          DD1(IADD)=DD1(IADD)+AXX
                          IADD=IUPT(JJ+1)+JJ+1
                          DD1(IADD)=DD1(IADD)+AYY
                          IADD=IUPT(JJ+2)+JJ+2
                          DD1(IADD)=DD1(IADD)+AZZ
                          IADD=IUPT(JJ)+JJ+1
                          DD1(IADD)=DD1(IADD)+AXY
                          IADD=IUPT(JJ)+JJ+2
                          DD1(IADD)=DD1(IADD)+AXZ
                          IADD=IUPT(JJ+1)+JJ+2
                          DD1(IADD)=DD1(IADD)+AYZ
                          !
                          IF (JJ < II) THEN
                             IADD=IUPT(JJ)+II
                             DD1(IADD)=DD1(IADD)-AXX
                             IADD=IUPT(JJ+1)+II+1
                             DD1(IADD)=DD1(IADD)-AYY
                             IADD=IUPT(JJ+2)+II+2
                             DD1(IADD)=DD1(IADD)-AZZ
                             IADD=IUPT(JJ)+II+1
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(JJ+1)+II
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(JJ)+II+2
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(JJ+2)+II
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(JJ+1)+II+2
                             DD1(IADD)=DD1(IADD)-AYZ
                             IADD=IUPT(JJ+2)+II+1
                             DD1(IADD)=DD1(IADD)-AYZ
                          ELSE
                             IADD=IUPT(II)+JJ
                             DD1(IADD)=DD1(IADD)-AXX
                             IADD=IUPT(II+1)+JJ+1
                             DD1(IADD)=DD1(IADD)-AYY
                             IADD=IUPT(II+2)+JJ+2
                             DD1(IADD)=DD1(IADD)-AZZ
                             IADD=IUPT(II+1)+JJ
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(II)+JJ+1
                             DD1(IADD)=DD1(IADD)-AXY
                             IADD=IUPT(II+2)+JJ
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(II)+JJ+2
                             DD1(IADD)=DD1(IADD)-AXZ
                             IADD=IUPT(II+2)+JJ+1
                             DD1(IADD)=DD1(IADD)-AYZ
                             IADD=IUPT(II+1)+JJ+2
                             DD1(IADD)=DD1(IADD)-AYZ
                          ENDIF
                          !
                    !
                 ENDIF  ! (LSECD)
     enddo loop100
     DX(I) =  DXI
     DY(I) =  DYI
     DZ(I) =  DZI
  ENDDO
  !=======================================================================
  !   Main loop end
  !=======================================================================
  return
end SUBROUTINE ENBIPS


  
SUBROUTINE ENBIPSEXP(ENB,EEL,IFRSTA,NATOM,LVDW,LELEC,LCONS, &
     JNB,INBLO,CG,RSCLF,CNBA,CNBB,IAC,ITC, &
     EPS,DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD  &
#if KEY_IMCUBES==1
     ,lbycbim                        & 
#endif
     )
  !
  !   3D IPS Nbonded interaction
  !
  !
  use chm_kinds
  ! input/output
  use dimens_fcm
  use consta
  use number
  use inbnd,only:E14FAC
  use param,only:MAXCN
  use nbips
  implicit none
  real(chm_real) ENB, EEL
  INTEGER IFRSTA,NATOM
  INTEGER INBLO(*),JNB(*)
  real(chm_real) CG(*),RSCLF(*),CNBA(*), CNBB(*)
  INTEGER IAC(*), ITC(*)
  real(chm_real)  EPS
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  LOGICAL LVDW,LELEC,LCONS
  real(chm_real)  DD1(*)
  INTEGER IUPT(*)
  LOGICAL LSECD
#if KEY_IMCUBES==1
  logical lbycbim                                
#endif

  ! local
  INTEGER I,J,K,IACI,IACJ,IC,ITI,ITJ,II,JJ,IADD
  INTEGER ILAST, IFIRST
  real(chm_real) DXI, DYI, DZI ,DF,DFX,DFY,DFZ
  real(chm_real) ENBIJ,EELIJ,ECONTI
  real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,R1,R2
  real(chm_real)  CGF,CGI,CGIJ,AIJ,CIJ,SIG2,SIG6,SIG12
  real(chm_real) U1,U2,DFIJ,PE,DPE
  real(chm_real)  SIG,SIGEX,AKR,AKRC,CKR,CKRC,EXPAR,EXPCR,EXPARC,EXPCRC
  real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,DDF
  !
  ! begin
  !
  !write(*,*)'ENBIPSEXP0= ',ENB,EEL,(AIPSVC(I),AIPSVA(I),I=0,6),PIPSVCC,PIPSVAC
  ! Setup constants for use in inner loops
  CGF=CCELEC/EPS
  !=======================================================================
  !
  !=======================================================================
  !   Main loop begin
  !=======================================================================
  ILAST = 0
  EELIJ=ZERO
  ENBIJ=ZERO
  DO I=IFRSTA,NATOM
#if KEY_IMCUBES==1
     if(lbycbim)ilast=inblo(I+natom)
#endif 
     !        define index list J
     IFIRST = ILAST+1
     ILAST=INBLO(I)
     IF(LCONS)THEN
       CGI=CG(I)*CGF*RIPSR
     ELSE
       CGI=CG(I)*CGF*RIPS2R
     ENDIF
     ITI=ITC(IAC(I))
     IC=ITI*(ITI-1)/2+ITI
     IACI=ITI*(ITI-1)/2
     DXI=DX(I)
     DYI=DY(I)
     DZI=DZ(I)
     XI=X(I)
     YI=Y(I)
     ZI=Z(I)
     loop100: DO K=IFIRST,ILAST
        J = JNB(K)
        IF(J <= 0) cycle loop100
        XIJ=XI-X(J)
        YIJ=YI-Y(J)
        ZIJ=ZI-Z(J)
        R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
        IF(R2 >= RIPS2) cycle loop100
        R1=SQRT(R2)
        U2=R2*RIPS2R
        U1=R1*RIPSR
        IF(LELEC)THEN
           !  Electrostatic IPS
           !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
           !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
           !
           CGIJ=CGI*CG(J)
           IF(LCONS)THEN
             PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
             DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
           ELSE
             PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
             DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
           ENDIF
           EELIJ=CGIJ*(PE-PIPSEC)
           EEL=EEL+EELIJ
           DF=-CGIJ*DPE/R2
        ELSE
           DF=ZERO
           DDF=ZERO
        ENDIF
        IF(LVDW)THEN
           !  exponential potential IPS
           ITJ=ITC(IAC(J))
           IF(ITI > ITJ)THEN
              IC=ITI*(ITI-1)/2+ITJ
           ELSE
              IC=ITJ*(ITJ-1)/2+ITI
           ENDIF
           SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)
           IF(SIG2>RSMALL)THEN
             SIG=SQRT(SIG2)
            IF(R1<SRIPS*SIG)THEN
            AKR=AIPSVA(0)*R1/SIG
            !AKRC=AIPSVA(0)*RIPS/SIG
            EXPAR=EXP(-AKR)
            !EXPARC=EXP(-AKRC)
            ! energy
            ENBIJ=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
            ! force 
            DFIJ=BIPSVA(1)*(EXPAR-AIPSVA(4)/EXPAR)/SIG
            CKR=AIPSVC(0)*R1/SIG
            !CKRC=AIPSVC(0)*RIPS/SIG
            EXPCR=EXP(-CKR)
            !EXPCRC=EXP(-CKRC)
            ! energy
            ENBIJ=ENBIJ-AIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR+AIPSVC(5))
            ! force
            DFIJ=DFIJ-BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG
            ENB=ENB+CNBB(IC)*ENBIJ
            DF=DF-CNBB(IC)*DFIJ/R1
           !write(*,*)'ENBIPSRNIJ= ',
           !I,J,R1,SINHAR,COSHAR,SINHCR,COSHCR,ENBIJ,DFIJ,ENB,DF
            ENDIF
          ENDIF
        ENDIF
        ! Calculate total forces
        DFX=DF*XIJ
        DFY=DF*YIJ
        DFZ=DF*ZIJ
        DXI=DXI-DFX
        DYI=DYI-DFY
        DZI=DZI-DFZ
        DX(J)=DX(J)+DFX
        DY(J)=DY(J)+DFY
        DZ(J)=DZ(J)+DFZ
     enddo loop100
     DX(I) =  DXI
     DY(I) =  DYI
     DZ(I) =  DZI
  ENDDO
  !write(*,*)'ENBIPSEXP1= ',ENB,EEL
  !=======================================================================
  !   Main loop end
  !=======================================================================
  return
end SUBROUTINE ENBIPSEXP

#else /* (nbips_main)*/

SUBROUTINE NBIPS_NULL
  RETURN  
END SUBROUTINE NBIPS_NULL

end module aips_module
#endif /* (nbips_main)*/


