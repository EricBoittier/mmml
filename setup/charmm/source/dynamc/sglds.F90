module sgld
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_SGLD==1 /*vars*/

!  Self Guided Langevin Dynamics (SGLD)
!
!        By Xiongwu Wu and Bernard Brooks, NIH, July 2003
!                                       Modified Dec 2005
!                                       Modified July 2010
!                                       Modified Jan 2013
!                                       Modified July 2015
!                                       Modified July 2020
!
!      variables and commom areas for SGLD simulation
!
!     QSGLD - logical flag for SGLDg
!     QSGMD - logical flag for SGMDg simulation
!     QSGBZ - logical flag to enforce Boltzmann distribution
!
!    SGLD ARRAYS (HEAP INDEX):
!     SGX1,SGY1,SGZ1   ! position 1st order average
!     SGX2,SGY2,SGZ2   ! position 2nd order average
!     SGVX,SGVY,SGVZ   ! velocity averages
!     SGDX,SGDY,SGDZ   ! Interaction forces
!     SGKX,SGKY,SGKZ   ! random force average arrays
!     SGWT             ! weighting factor for guiding forces. Using SCALAR to manipulate.
!     SGGAMMA          ! apparent friction constants
!     SGBETA           ! energy conservation factor
!     SGFP             ! average force momentum product 
!     SGPP             ! average momentum momentum product 
!
!    SGLD APPLYING ATOM RANGE
!      ISGSTA     ! Begining atom index applying sgld
!      ISGEND     ! Ending atom index applying sgld
!      SGTYPE - SG structure size. 1-atom; 2-also bonded and bond angle atoms; 
!               3-also dihedral angle atoms; 4-atoms within SGSIZE cutoff
!
!    SGLD VARIABLES
!     SGAVG0  ! Local average remains
!     SGAVG1  ! Local average factor, SGAVG1=1-SGAVG0
!     SGAVP0  ! Convergency control average remains
!     SGAVP1  ! Convergency control average factor, SGAVP1=1-SGAVP0
!     TSGAVG  ! Local average time, ps
!     TSGAVP  ! Convergency control average time, ps
!     SGFT    ! Momentum guiding factor
!     SGFTI   ! Current momentum guiding factor
!     SGFF    ! force guiding factor
!     SGFFI   ! Current force guiding factor
!     SGFD    ! SGLD-GLE momentum guiding factor
!     SGFDI   ! Current SGLD-GLE momentum guiding factor
!     TEMPSG  ! Guiding temperature to set SGFT and SGFF
!     SGSIZE  ! Local average length, angstrom
!
  real(chm_real),allocatable,dimension(:) :: SGWT,SGGAMMA,SGFP,SGPP,SGBETA
  real(chm_real),allocatable,dimension(:) :: SGX1,SGY1,SGZ1,SGX2,SGY2,SGZ2
  real(chm_real),allocatable,dimension(:) :: SGVX,SGVY,SGVZ,SGDX,SGDY,SGDZ
  real(chm_real),allocatable,dimension(:) :: SGKX,SGKY,SGKZ
  integer ISGSTA,ISGEND,NDEGFSG,NSGSUM,NSGOUT
  integer,allocatable,dimension(:) :: NATSG,IDATSG,IDXATSG
  real(chm_real),allocatable,dimension(:) :: FMSG 

  integer SGTYPE,NGRIDX,NGRIDY,NGRIDZ,NGRIDXY,NGRIDYZ,NGRIDZX,NGRIDXYZ
  real(chm_real) SGSIZE,GXMIN,GXMAX,GYMIN,GYMAX,GZMIN,GZMAX
  real(chm_real),allocatable,dimension(:) :: RHOM,RHOVX,RHOVY,RHOVZ,RHOAX,RHOAY,RHOAZ 
  

  real(chm_real) TEMPSG,TSGAVG,TSGAVP,SGAVG0,SGAVG1,SGAVP0,SGAVP1
  real(chm_real) SGFT,SGFTI,SGFF,SGFFI,SGFD,SGFDI,SGSCAL,FSGLDG,PSGLDG
  real(chm_real) TSGSET,TEMPLF,TEMPHF
  real(chm_real) EPOTLF,EPOTLF2,EPOTHF
  real(chm_real) SGEXP,SUMSGWT
  real(chm_real) SGSUM(20),SGSUM2(20)

  LOGICAL QSGLD,QSGMD,QSGBZ
  LOGICAL QSGCOM,QSGGAMMA
#endif /* (vars)*/

contains
#if KEY_SGLD==1 /*sgld_main*/
  SUBROUTINE PSGLD(NATOM,TIMEST,X,Y,Z,VX,VY,VZ)
    !
    !    This routine allocate memory for Average arrays
    !
  use chm_kinds
  use cnst_fcm
  use memory
  use stream, only:prnlev,outu
  use psf, only:amass
  use bases_fcm, only: bnbnd
  use inbnd, only: nnb14
  implicit none

    INTEGER NATOM
    real(chm_real) TIMEST
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) VX(*),VY(*),VZ(*)
    !
    if(prnlev>0)write(OUTU,'("     SGMD-SGLD VERSION: 01-1-2023     ")')
    !  Allocate memory for SGLD arrays
    call chmalloc('sglds.src','PSGLD','SGX1',NATOM,crl=SGX1)
    call chmalloc('sglds.src','PSGLD','SGY1',NATOM,crl=SGY1)
    call chmalloc('sglds.src','PSGLD','SGZ1',NATOM,crl=SGZ1)
    call chmalloc('sglds.src','PSGLD','SGX2',NATOM,crl=SGX2)
    call chmalloc('sglds.src','PSGLD','SGY2',NATOM,crl=SGY2)
    call chmalloc('sglds.src','PSGLD','SGZ2',NATOM,crl=SGZ2)
    call chmalloc('sglds.src','PSGLD','SGVX',NATOM,crl=SGVX)
    call chmalloc('sglds.src','PSGLD','SGVY',NATOM,crl=SGVY)
    call chmalloc('sglds.src','PSGLD','SGVZ',NATOM,crl=SGVZ)
    call chmalloc('sglds.src','PSGLD','SGDX',NATOM,crl=SGDX)
    call chmalloc('sglds.src','PSGLD','SGDY',NATOM,crl=SGDY)
    call chmalloc('sglds.src','PSGLD','SGDZ',NATOM,crl=SGDZ)
    call chmalloc('sglds.src','PSGLD','SGKX',NATOM,crl=SGKX)
    call chmalloc('sglds.src','PSGLD','SGKY',NATOM,crl=SGKY)
    call chmalloc('sglds.src','PSGLD','SGKZ',NATOM,crl=SGKZ)
    call chmalloc('sglds.src','PSGLD','SGFP',NATOM,crl=SGFP)
    call chmalloc('sglds.src','PSGLD','SGPP',NATOM,crl=SGPP)
    call chmalloc('sglds.src','PSGLD','SGBETA',NATOM,crl=SGBETA)
    call chmalloc('sglds.src','PSGLD','NATSG',NATOM,intg=NATSG)
    call chmalloc('sglds.src','PSGLD','IDATSG',NATOM+2*NNB14,intg=IDATSG)
    call chmalloc('sglds.src','PSGLD','IDXATSG',NATOM,intg=IDXATSG)
    call chmalloc('sglds.src','PSGLD','FMSG',NATOM,crl=FMSG)
    
    CALL SGINIT(NATOM,ISGSTA,ISGEND,TIMEST,TSGAVG, &
         BNBND%IBLO14,BNBND%INB14,FBETA,AMASS,X,Y,Z,VX,VY,VZ)

    !     Caculate initial constants
    RETURN
  END SUBROUTINE PSGLD

  SUBROUTINE SGFREE
    !
    !      This routinefREE memory space for Average arrays
    !
  use chm_kinds
  use psf, only:NATOM
  use memory
  use inbnd, only: nnb14

    !WXW    free storage for SGLD substructure data
    call chmdealloc('sglds.src','SGFREE','SGX1',NATOM,crl=SGX1)
    call chmdealloc('sglds.src','SGFREE','SGY1',NATOM,crl=SGY1)
    call chmdealloc('sglds.src','SGFREE','SGZ1',NATOM,crl=SGZ1)
    call chmdealloc('sglds.src','SGFREE','SGX2',NATOM,crl=SGX2)
    call chmdealloc('sglds.src','SGFREE','SGY2',NATOM,crl=SGY2)
    call chmdealloc('sglds.src','SGFREE','SGZ2',NATOM,crl=SGZ2)
    call chmdealloc('sglds.src','SGFREE','SGVX',NATOM,crl=SGVX)
    call chmdealloc('sglds.src','SGFREE','SGVY',NATOM,crl=SGVY)
    call chmdealloc('sglds.src','SGFREE','SGVZ',NATOM,crl=SGVZ)
    call chmdealloc('sglds.src','SGFREE','SGDX',NATOM,crl=SGDX)
    call chmdealloc('sglds.src','SGFREE','SGDY',NATOM,crl=SGDY)
    call chmdealloc('sglds.src','SGFREE','SGDZ',NATOM,crl=SGDZ)
    call chmdealloc('sglds.src','SGFREE','SGKX',NATOM,crl=SGKX)
    call chmdealloc('sglds.src','SGFREE','SGKY',NATOM,crl=SGKY)
    call chmdealloc('sglds.src','SGFREE','SGKZ',NATOM,crl=SGKZ)
    call chmdealloc('sglds.src','PSGLD','SGFP',NATOM,crl=SGFP)
    call chmdealloc('sglds.src','PSGLD','SGPP',NATOM,crl=SGPP)
    call chmdealloc('sglds.src','SGFREE','SGBETA',NATOM,crl=SGBETA)
    call chmdealloc('sglds.src','SGFREE','NATSG',NATOM,intg=NATSG)
    call chmdealloc('sglds.src','SGFREE','IDATSG',NATOM+2*NNB14,intg=IDATSG)
    call chmdealloc('sglds.src','SGFREE','IDXATSG',NATOM,intg=IDXATSG)
    call chmdealloc('sglds.src','SGFREE','FMSG',NATOM,crl=FMSG)
    RETURN
  END SUBROUTINE SGFREE

  SUBROUTINE ALLOCATE_SGWT(NATOM)
    !
    !    This routine allocate memory for Average arrays
    !
  use chm_kinds
  use cnst_fcm
  use memory
  use stream, only:prnlev,outu
  implicit none

    INTEGER NATOM
    !
    !  Allocate memory for SGLD array SGWT
    call chmalloc('sglds.src','ALLOCATE_SGWT','SGWT',NATOM,crl=SGWT)
    SGWT=1.0D0
    RETURN
  END SUBROUTINE ALLOCATE_SGWT

  SUBROUTINE ALLOCATE_SGGAMMA(NATOM)
    !
    !    This routine allocate memory for Average arrays
    !
  use chm_kinds
  use cnst_fcm
  use memory
  use stream, only:prnlev,outu
  implicit none

    INTEGER NATOM
    !
    !  Allocate memory for SGLD array SGGAMMA
    call chmalloc('sglds.src','ALLOCATE_SGGAMMA','SGGAMMA',NATOM,crl=SGGAMMA)
    QSGGAMMA=.TRUE.
    SGGAMMA=0.0D0
    RETURN
  END SUBROUTINE ALLOCATE_SGGAMMA

 SUBROUTINE SGINIT(NATOM,ISGSTA,ISGEND,TIMEST,TSGAVG, &
       IBLO,INB,FBETA,AMASS,X,Y,Z,VX,VY,VZ)  !  ,AVGVX,AVGVY,AVGVZ,AVGV2,AVGG2)
    !
    !    This routine perform initiation for average arrays
    !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use stream
  use shake
  implicit none
    INTEGER NATOM,ISGSTA,ISGEND
    INTEGER I,J,K,II,NDEGFT,IX
    INTEGER IBLO(*),INB(*)
    real(chm_real) AMASS(*),X(*),Y(*),Z(*)
    real(chm_real) VX(*),VY(*),VZ(*),FBETA(*)
!    real(chm_real) SGVX(*),SGVY(*),SGVZ(*),SGV2(*),SGG2(*)
    real(chm_real) XI,YI,ZI,VXI,VYI,VZI
    real(chm_real) TIMEST,TSGAVG,VFACT
    real(chm_real) EGG,GAMM,DELTA,AMASSI,AMASSJ,RFD,FACT1,FACT2
    INTEGER IFRSTA,ILASTA,IFRST,ILAST
    INTEGER NATSGI,IDXATSGI,JFRSTA,JFRST,JLAST
#if KEY_PARALLEL==1
!
#if KEY_VSCODE==1
!     mpi debug
    i=0
    do while (i==0)
      call sleep(5)
    enddo
#endif /* KEY_VSCODE */
#endif /* KEY_PARALLEL */
!
  ! allocate SGWT
    IF(.not.allocated(sgwt))call allocate_sgwt(natom)
    ! allocate SGGAMMA
    IF(.not.allocated(sggamma)) call allocate_sggamma(natom)
    !WXW    Assign initial values
    NSGOUT=6
    DELTA=TIMEST/TIMFAC
    ! average over bonded structures
    ! Fill SG structure arrays
    IFRSTA=1
    ILASTA=NATOM
    ILAST = 0
    IF(IFRSTA > 1)ILAST=IBLO(IFRSTA-1)
    IDXATSGI=0
    DO I=IFRSTA,ILASTA
       ! the 1st atom is itself
       IDXATSG(I)=IDXATSGI
       NATSGI=1
       IDXATSGI=IDXATSGI+1
       IDATSG(IDXATSGI)=I
       AMASSI=AMASS(I)
       IF(SGTYPE==1.OR.SGTYPE>3)THEN
        NATSG(I)=NATSGI
        FMSG(I)=AMASS(I)/AMASSI
        CYCLE 
       ENDIF 
       ! identify excluded atoms from previous list
       JLAST = 0
       IF(IFRSTA > 1)JLAST=IBLO(IFRSTA-1)
       DO J=IFRSTA,I-1
         JFRST =JLAST + 1
         JLAST = IDXATSG(J) + NATSG(J)
         DO K=JFRST,JLAST
           IF(IDATSG(K)==I)THEN
             NATSGI=NATSGI+1
             IDXATSGI=IDXATSGI+1
             IDATSG(IDXATSGI)=J
             AMASSI=AMASSI+AMASS(J)
           ENDIF
         ENDDO
       ENDDO
       !  identify excluded atoms from exlusion list
       IFRST = ILAST+1
       ILAST=IBLO(I)
       loop100: DO K=IFRST,ILAST
          J = INB(K)
          IF(J < 0)THEN
             J=-J
             IF(SGTYPE<3)CYCLE
          ENDIF
          NATSGI=NATSGI+1
          IDXATSGI=IDXATSGI+1
          IDATSG(IDXATSGI)=J
          AMASSI=AMASSI+AMASS(J)
       enddo loop100
       NATSG(I)=NATSGI
       FMSG(I)=AMASSI
    ENDDO       
    NDEGFSG=0
    NDEGFT=0
    SUMSGWT=ZERO
    DO I=1,NATOM
       II=0
       IF(I>=ISGSTA .AND. I<=ISGEND)II=1
       IF(QSHAKE)THEN
          NDEGFT=NDEGFT+IDGF2(I)
          NDEGFSG=NDEGFSG+IDGF2(I)*II
       ELSE
          NDEGFT=NDEGFT+3
          NDEGFSG=NDEGFSG+3*II
       ENDIF
       AMASSI=AMASS(I)
       SGFP(I)=ZERO
       !SGPP(I)=AMASSI*(VX(I)*VX(I)+VY(I)*VY(I)+VZ(I)*VZ(I))/DELTA
       SGPP(I)=ZERO
       SGBETA(I)=ZERO
       SUMSGWT=SUMSGWT+SGWT(I)
       !WXW    initialize arrays
            ! average over SG structures
       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
            NATSGI=NATSG(I)
            IF(NATSGI>1)THEN
              XI=XI*AMASSI
              YI=YI*AMASSI
              ZI=ZI*AMASSI
              DO IX=2,NATSGI-1
              J=IDATSG(IDXATSG(I)+IX)
              AMASSJ=AMASS(J)
              XI=XI+X(J)*AMASSJ 
              YI=YI+Y(J)*AMASSJ  
              ZI=ZI+Z(J)*AMASSJ 
              ENDDO 
              XI=XI/FMSG(I)
              YI=YI/FMSG(I)
              ZI=ZI/FMSG(I)
            ENDIF
       SGX1(I)=XI
       SGY1(I)=YI
       SGZ1(I)=ZI
       SGX2(I)=XI
       SGY2(I)=YI
       SGZ2(I)=ZI
       SGVX(I)=ZERO
       SGVY(I)=ZERO
       SGVZ(I)=ZERO
       SGKX(I)=ZERO
       SGKY(I)=ZERO
       SGKZ(I)=ZERO
    ENDDO
    IF(SGTYPE>3)THEN
      GXMIN=RBIG
      GXMAX=-RBIG
      GYMIN=RBIG
      GYMAX=-RBIG
      GZMIN=RBIG
      GZMAX=-RBIG
      CALL MAPINIT(1,natom,X,Y,Z)
    ENDIF
    !write(OUTU,'("DEBUGI ",10E14.6)')(SGPP(I),I=1,10)
    IF(QSHAKE)NDEGFSG=NDEGFSG/2
    SGFTI=SGFT
    SGFFI=ZERO
    SGFDI=ZERO
    IF(SGFF*SGFF>RSMALL)SGFFI=SGFF
    IF(SGFD*SGFD>RSMALL)SGFDI=SGFD
    IF(TSGSET<RSMALL)TSGSET=300.0D0
    IF(QSGBZ)THEN
      IF(SGFTI*SGFTI>RSMALL)THEN
          IF(SGFTI*SGFTI>RSMALL)THEN
            FACT1=SQRT(NINE*NINE-TWELVE*SGFTI*SGFTI*SGFTI)-NINE
            FACT2=(ABS(FACT1)*THREE/TWO)**(ONE/THREE)
            SGFFI=-ONE-SIGN(ONE,FACT1)*(FACT2/THREE+SGFTI/FACT2)
          ELSE
            SGFFI=ZERO
          ENDIF
      ELSE 
          SGFTI=(ONE+SGFFI)*(ONE+SGFFI)-ONE/(ONE+SGFFI)
      ENDIF
    ENDIF
    ! GLE force constant
    FSGLDG=-ONE+SQRT(ONE-SGFDI)
    ! canonical reweighting force constant 
    !PSGLDG=SQRT(ONE+SGFTI)-ONE
    !PSGLDG=SIGN(SQRT(ONE+ABS(SGFTI))-ONE,SGFTI)
    IF(SGFTI*SGFTI>RSMALL)THEN
      FACT1=SQRT(NINE*NINE-TWELVE*SGFTI*SGFTI*SGFTI)-NINE
      FACT2=(ABS(FACT1)*THREE/TWO)**(ONE/THREE)
      PSGLDG=-ONE-SIGN(ONE,FACT1)*(FACT2/THREE+SGFTI/FACT2)
    ELSE
      PSGLDG=ZERO
    ENDIF
    IF(TEMPSG>RSMALL)THEN
    ! when tempsg is set
      FACT1=ONE-TSGSET/TEMPSG
      IF(SGFTI*SGFTI>RSMALL)THEN
        SGFFI=PSGLDG-FACT1
      ELSE 
        PSGLDG=SGFFI+FACT1
        SGFTI=(ONE+PSGLDG)*(ONE+PSGLDG)-ONE/(ONE+PSGLDG)
      ENDIF
    ELSE
      TEMPSG=TSGSET/(ONE-PSGLDG+SGFFI)
    ENDIF
    EPOTLF=TWO*RBIG
    SGEXP=ZERO
    NSGSUM=0
    DO I=1,NSGOUT
      SGSUM(I)=ZERO
      SGSUM2(I)=ZERO
    ENDDO

    IF(PRNLEV >= 2 .AND. TEMPSG>RSMALL)WRITE(OUTU,970)TEMPSG
    IF(PRNLEV >= 2)WRITE(OUTU,980)SGFTI,PSGLDG
    IF(PRNLEV >= 2)WRITE(OUTU,981)SGFFI
    IF(PRNLEV >= 2)WRITE(OUTU,982)SGFDI,FSGLDG
    IF(PRNLEV >= 2)WRITE(OUTU,989)

970  FORMAT('  Guiding temperature TEMPSG is:',F10.4, ' K')
980  FORMAT('  Momentum guiding factor SGFTI, PSGLDG is set to:',F8.4,F10.4)
981  FORMAT('  Force guiding factor SGFFI is set to:',F8.4 )
982  FORMAT('  SGLD-GLE momentum guiding factor SGFDI, FSGLDG is set to:',F8.4,F10.4)
989  FORMAT(' ............End of SGMD/SGLD Parameters......'/)
    RETURN
  END SUBROUTINE SGINIT

  SUBROUTINE SGFAVG(ATFRST,ATLAST,EPOT,IMOVE,DX,DY,DZ)
    !
    !     This routine calculates the local average forces
    !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use reawri
  use econtmod, only:qecont,econt
  use machutil,only:eclock
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec,natoml,atoml
#endif
#if KEY_PARALLEL==1
  use parallel
#endif
  implicit none
    INTEGER ATFRST,ATLAST
    real(chm_real)  EPOT,VIRINT,VOLUME
    INTEGER IMOVE(*)
    real(chm_real)  DX(*),DY(*),DZ(*)
    !
    !
    INTEGER I,IFRST,ILAST,IA
    real(chm_real) EPSUM,GAMAVG,DXI,DYI,DZI,TMP,TMPS(20)
#if KEY_PARALLEL==1
    real(chm_real) TIMMER
#endif
    !
    IFRST=ATFRST
    ILAST=ATLAST
    IF(ISGSTA > IFRST) IFRST=ISGSTA
    IF(ISGEND < ILAST) ILAST=ISGEND
    !
    do ia = atfrst,atlast
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif
       IF(I<ISGSTA .OR. I>ISGEND)CYCLE
       IF(IMOVE(I) == 0) THEN
          DXI=DX(I)
          DYI=DY(I)
          DZI=DZ(I)
          SGDX(I)=DXI
          SGDY(I)=DYI
          SGDZ(I)=DZI
       ENDIF
    ENDDO
    IF(QECONT)THEN
      ! when sgld is applied to only a part of a system
      EPSUM=ZERO
      DO I=IFRST,ILAST
        EPSUM=EPSUM+ECONT(I)
      ENDDO
#if KEY_PARALLEL==1
      CALL PSYNC()
      TIMMER=ECLOCK()
      CALL GCOMB(EPSUM,1)
      TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
      TIMMER=ECLOCK()
#endif
    ELSE
      EPSUM=EPOT
    ENDIF
    IF(EPOTLF>RBIG)THEN
       EPOTLF=EPSUM
       EPOTHF=EPSUM
    ELSE
       EPOTLF=SGAVG0*EPOTLF+SGAVG1*EPSUM
       EPOTHF=SGAVG0*EPOTHF+SGAVG1*EPOTLF
    ENDIF
    !GAMAVG=SUM(SGGAMMA*SGWT)/SUMSGWT
    GAMAVG=SUM(SGBETA*SGWT)/SUMSGWT
    !ACGAMMA=ACGAMMA+GAMAVG
    SGEXP=-(PSGLDG-SGFFI)*(EPOTLF-EPOTHF)/(KBOLTZ*TSGSET)
    !
    ! Guiding variable averages
    NSGSUM=NSGSUM+1
    TMPS(1)=SGSCAL
    TMPS(2)=TEMPLF
    TMPS(3)=TEMPHF
    !TMPS(4)=GAMAVG
    TMPS(4)=EPOTLF
    TMPS(5)=EPOTHF
    TMPS(6)=SGEXP
    DO I=1,NSGOUT
       TMP=TMPS(I)
       SGSUM(I)=SGSUM(I)+TMP
       SGSUM2(I)=SGSUM2(I)+TMP*TMP
    ENDDO
    RETURN
  END SUBROUTINE SGFAVG


  subroutine sgld_ave_params()
    use chm_kinds, only: chm_real
    use param_store, only: set_param

    implicit none

    integer i
    real(chm_real) tmps(20)

    do i = 1,  nsgout
       tmps(i) = sgsum(i) / nsgsum
    end do

    call set_param('SGGAM', TMPS(1))
    call set_param('TEMPLF', TMPS(2))
    call set_param('TEMPHF', TMPS(3))
    call set_param('EPOTLF', TMPS(4))
    call set_param('EPOTLLF', TMPS(5))
    call set_param('SGEXP', TMPS(6))
  end subroutine sgld_ave_params

  SUBROUTINE PRNTSG(OUTU,QHEAD,SGTYPE,QINIT)
    !
    !     This routine calculates the guiding forces and adds them
    !     to the systematic forces.
    !         Assum SGFT*gamma=constant for all atoms
    !
  use chm_kinds
  use dimens_fcm
  use number
  use param_store, only: set_param

  implicit none

    INTEGER OUTU,I
    LOGICAL QHEAD,QINIT
    CHARACTER(len=4) SGTYPE
    CHARACTER(len=10) MARK
    CHARACTER(len=70) LABEL
    real(chm_real) TMP,TMPS(20),GAMAVG
  !WXW    determine data type to be printed
    IF(QSGLD)THEN
       MARK=SGTYPE//' SGLD'
    ELSE IF(QSGMD)THEN
       MARK=SGTYPE//' SGMD'
    ELSE
       RETURN
    ENDIF
    IF(SGTYPE.EQ.'AVER')THEN
       DO I=1,NSGOUT
          TMPS(I)=SGSUM(I)/NSGSUM
       ENDDO
    ELSE IF(SGTYPE.EQ.'FLUC')THEN
       DO I=1,NSGOUT
          TMP=SGSUM(I)/NSGSUM
          TMPS(I)=SGSUM2(I)/NSGSUM-TMP*TMP
          IF(TMPS(I)>ZERO)THEN
              TMPS(I)=SQRT(TMPS(I))
          ELSE
              TMPS(I)=ZERO
          ENDIF
       ENDDO
    ELSE IF(SGTYPE.EQ.'DYNA')THEN
       TMPS(1)=SGSCAL
       TMPS(2)=TEMPLF
       TMPS(3)=TEMPHF
       TMPS(4)=EPOTLF
       TMPS(5)=EPOTHF
       TMPS(6)=SGEXP
    ELSE
      TMPS(1)=SGSCAL
      TMPS(2)=TEMPLF
      TMPS(3)=TEMPHF
      TMPS(4)=EPOTLF
      TMPS(5)=EPOTHF
      TMPS(6)=SGEXP
    ENDIF
    IF(QHEAD)THEN
      WRITE(OUTU,'(A9,": "," SGGAMMA ","  TEMPLF ",&
&          "  TEMPHF","        EPOTLF ", "      EPOTLLF ",   "        SGEXP")')MARK
    ENDIF
    WRITE(OUTU,'(A9,"> ",F8.4,X,F8.2,X,F8.2,F14.4,F14.4,F14.4)')&
&               MARK,(TMPS(I),I=1,NSGOUT)
    IF(QINIT)THEN
       NSGSUM=0
       DO I=1,NSGOUT
          SGSUM(I)=ZERO
          SGSUM2(I)=ZERO
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE PRNTSG

  SUBROUTINE SGLDW(ATFRST,ATLAST,DELTA,NDEGF,TEMPI, &
                   IMOVE,AMASS,X,Y,Z,VX,VY,VZ,DX,DY,DZ)
    !
    !     This routine calculates the guiding forces and adds them
    !     to the systematic forces.
    !         Assum SGFT*gamma=constant for all atoms
    !
    use chm_kinds
    use dimens_fcm
    use number
    use stream
    use consta
    use cnst_fcm
#if KEY_PARALLEL==1
    use parallel
#endif
    use machutil,only:eclock
    use prssre
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml
#endif
    implicit none
    !
    INTEGER ATFRST,ATLAST,NDEGF
    real(chm_real)  DELTA,TEMPI
    INTEGER IMOVE(*)
    real(chm_real)  AMASS(*),X(*),Y(*),Z(*),VX(*),VY(*),VZ(*), &
         DX(*),DY(*),DZ(*)
    !
#if KEY_PARALLEL==1
    real(chm_real) GCARR(31),TIMMER
#endif
    !
    INTEGER I,J,K,IFRST,ILAST,IA,IX,NATSGI
    real(chm_real) XI,YI,ZI,X1I,Y1I,Z1I,X2I,Y2I,Z2I
    real(chm_real) VXI,VYI,VZI,VXT,VYT,VZT
    real(chm_real) FXI,FYI,FZI,GXI,GYI,GZI,DXI,DYI,DZI
    real(chm_real) SGXI,SGYI,SGZI,SGXJ,SGYJ,SGZJ,SGXK,SGYK,SGZK
    real(chm_real) GX2I,GY2I,GZ2I,GX2J,GY2J,GZ2J,GX2K,GY2K,GZ2K
    real(chm_real) GAM,GAMM,FACT0,FACT1,FACT2,DTEMP,SGFTJ
    real(chm_real) AMASSI,AMASSJ,DELTA2,SGGV,SGVV,EKIN,EKINSG,SUMGAM
    !
    DELTA2=DELTA*DELTA
    !
    IFRST=ATFRST
    ILAST=ATLAST
    IF(ISGSTA > IFRST) IFRST=ISGSTA
    IF(ISGEND < ILAST) ILAST=ISGEND
    !
    ! build spatial average maps
    IF(SGTYPE>3)THEN
      CALL MAPBUILD(ATFRST,ATLAST,DELTA,IMOVE,AMASS,X,Y,Z)
    ENDIF
    !
    EKIN=ZERO
    EKINSG=ZERO
    SUMGAM=ZERO
    !
    !WXW Calculate constraint factor to eliminate energy input from guiding force
    do ia = atfrst,atlast
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif
       IF(IMOVE(I) == 0) THEN
          AMASSI=AMASS(I)
          XI=X(I)
          YI=Y(I)
          ZI=Z(I)
          VXI=VX(I)
          VYI=VY(I)
          VZI=VZ(I)
          DXI=DX(I)
          DYI=DY(I)
          DZI=DZ(I)
          IF(I>=ISGSTA .AND. I<=ISGEND)THEN
            GAM=TIMFAC*FBETA(I)*DELTA
            GAMM=TWO/(TWO+GAM)
            ! average over SG structures
            NATSGI=NATSG(I)
            IF(NATSGI>1)THEN
              XI=XI*AMASSI
              YI=YI*AMASSI
              ZI=ZI*AMASSI
              DO IX=2,NATSGI-1
              J=IDATSG(IDXATSG(I)+IX)
              AMASSJ=AMASS(J)
              XI=XI+X(J)*AMASSJ 
              YI=YI+Y(J)*AMASSJ  
              ZI=ZI+Z(J)*AMASSJ 
              ENDDO 
              XI=XI/FMSG(I)
              YI=YI/FMSG(I)
              ZI=ZI/FMSG(I)
            ENDIF
            ! Obtain local averages
            IF(SGTYPE>3)THEN
              ! spatial average using map
              CALL MAPVALUES(AMASSI,XI,YI,ZI,GXI,GYI,GZI,SGXJ,SGYJ,SGZJ)
              GXI=GXI*DELTA
              GYI=GYI*DELTA
              GZI=GZI*DELTA
            ELSE
            X1I=SGAVG0*SGX1(I)+SGAVG1*XI
            Y1I=SGAVG0*SGY1(I)+SGAVG1*YI
            Z1I=SGAVG0*SGZ1(I)+SGAVG1*ZI
            X2I=SGAVG0*SGX2(I)+SGAVG1*X1I
            Y2I=SGAVG0*SGY2(I)+SGAVG1*Y1I
            Z2I=SGAVG0*SGZ2(I)+SGAVG1*Z1I
            SGX1(I)=X1I
            SGY1(I)=Y1I
            SGZ1(I)=Z1I
            SGX2(I)=X2I
            SGY2(I)=Y2I
            SGZ2(I)=Z2I
            ! avg(v)
            FACT1=TIMFAC*DELTA/TSGAVG
            GXI=FACT1*(XI-X1I)
            GYI=FACT1*(YI-Y1I)
            GZI=FACT1*(ZI-Z1I)
            VXT=(GXI-SGAVG0*SGVX(I))/SGAVG1
            VYT=(GYI-SGAVG0*SGVY(I))/SGAVG1
            VZT=(GZI-SGAVG0*SGVZ(I))/SGAVG1
            SGVX(I)=GXI
            SGVY(I)=GYI
            SGVZ(I)=GZI
            ! average force deviation 
            FACT1=-TIMFAC*AMASSI/DELTA/TSGAVG
            FACT2=-TIMFAC*DELTA/TSGAVG
            SGXJ=FACT1*(VXT+FACT2*(TWO*XI-THREE*X1I+X2I))
            SGYJ=FACT1*(VYT+FACT2*(TWO*YI-THREE*Y1I+Y2I))
            SGZJ=FACT1*(VZT+FACT2*(TWO*ZI-THREE*Z1I+Z2I))
            ENDIF /* SGTYPE */
            ! SGLD-GLE
            !random force averages
            SGXK=SGAVG0*SGKX(I)+SGAVG1*(DXI-SGDX(I))
            SGYK=SGAVG0*SGKY(I)+SGAVG1*(DYI-SGDY(I))
            SGZK=SGAVG0*SGKZ(I)+SGAVG1*(DZI-SGDZ(I))
            SGKX(I)=SGXK
            SGKY(I)=SGYK
            SGKZ(I)=SGZK
            FACT1=SGFDI*SGWT(I)*GAM*AMASSI/DELTA2
            FACT2=FSGLDG*SGWT(I)
            !DXI=DXI-FACT1*GXI+FACT2*SGXK
            !DYI=DYI-FACT1*GYI+FACT2*SGYK
            !DZI=DZI-FACT1*GZI+FACT2*SGZK
            SGXI=FACT1*GXI-FACT2*SGXK
            SGYI=FACT1*GYI-FACT2*SGYK
            SGZI=FACT1*GZI-FACT2*SGZK
            ! <avg(f-avg(f))avg(v)>
            SGFP(I)=SGAVP0*SGFP(I)+SGAVP1*(SGXJ*GXI+SGYJ*GYI+SGZJ*GZI)
            ! <avg(p)avg(v)>
            SGPP(I)=SGAVP0*SGPP(I)+SGAVP1*AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)/DELTA
            ! force friction constant
            IF(QSGGAMMA.AND.SGPP(I)>RSMALL)SGGAMMA(I)=SGFP(I)/SGPP(I)/TIMFAC
            SUMGAM=SUMGAM+SGGAMMA(I)
            ! guiding force
            FACT1=SGFTI*SGWT(I)*SGGAMMA(I)*TIMFAC*AMASSI*SGWT(I)/DELTA
            FACT2=SGFFI*SGWT(I)
            SGXI=SGXI+FACT1*GXI-FACT2*SGXJ
            SGYI=SGYI+FACT1*GYI-FACT2*SGYJ
            SGZI=SGZI+FACT1*GZI-FACT2*SGZJ
            ! <g.v>
            !FACT1=HALF*DELTA2/AMASSI
            !VXT=VXI-FACT1*(DXI-SGXI)
            !VYT=VYI-FACT1*(DYI-SGYI)
            !VZT=VZI-FACT1*(DZI-SGZI)
            ! Using current velocities to avoid SHAKE complication
            VXT=VXI
            VYT=VYI
            VZT=VZI
            SGGV=(SGXI*VXT+SGYI*VYT+SGZI*VZT)
            ! <v.v>
            SGVV=TWO*AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)/DELTA-SGGV*DELTA
            ! kinetic energies
            EKIN=EKIN+AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)
            EKINSG=EKINSG+AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)
            ! energy conservation friction constant
            IF(ABS(SGGV)<TEN*ABS(SGVV))THEN
              SGBETA(I)=(TWO+GAM)*SGGV/SGVV/TIMFAC
            ELSE
              SGBETA(I)=ZERO
            ENDIF
            SGXI=DXI-SGXI
            SGYI=DYI-SGYI
            SGZI=DZI-SGZI
            SGSCAL=TIMFAC*SGBETA(I)*DELTA
            GAMM=ONE+HALF*(GAM+SGSCAL)
            FACT1=(ONE+HALF*GAM)/GAMM
            FACT2=AMASSI*SGSCAL/GAMM/DELTA2
            SGXI=FACT1*SGXI+FACT2*VXI
            SGYI=FACT1*SGYI+FACT2*VYI
            SGZI=FACT1*SGZI+FACT2*VZI
            DX(I)=SGXI
            DY(I)=SGYI
            DZ(I)=SGZI
          ENDIF
       ENDIF
    ENDDO
    !
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
    GCARR(1)=EKIN
    GCARR(2)=EKINSG
    GCARR(3) = SUMGAM
    CALL GCOMB(GCARR,3)
    EKIN=GCARR(1)
    EKINSG=GCARR(2)
    SUMGAM=GCARR(3)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    TIMMER=ECLOCK()
#endif
    TEMPLF=EKINSG/(KBOLTZ*NDEGFSG)/DELTA/DELTA
    TEMPHF=EKIN/(KBOLTZ*NDEGFSG)/DELTA/DELTA-TEMPLF
    SGSCAL=SUMGAM/(ISGEND-ISGSTA+1)
    !
    RETURN
  END SUBROUTINE SGLDW

  SUBROUTINE SGMDW(ATFRST,ATLAST,DELTA,NDEGF,TEMPI, &
                   IMOVE,AMASS,X,Y,Z,VX,VY,VZ,DX,DY,DZ)
    !
    !     This routine calculates the guiding forces and adds them
    !     to the systematic forces.
    !         Assum SGFT*gamma=constant for all atoms
    !
    use chm_kinds
    use clcg_mod,only:random
    use dimens_fcm
    use number
    use stream
    use consta
    use parallel
    use machutil,only:eclock
    use prssre
    use rndnum
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml
#endif
    implicit none
    !
    INTEGER ATFRST,ATLAST,NDEGF
    real(chm_real)  DELTA,TEMPI
    INTEGER IMOVE(*)
    real(chm_real)  AMASS(*)
    real(chm_real)  X(*),Y(*),Z(*),VX(*),VY(*),VZ(*), &
         DX(*),DY(*),DZ(*)
    !
    INTEGER IG,IA
#if KEY_PARALLEL==1
    real(chm_real) GCARR(30),TIMMER
#endif
    !
    INTEGER I,J,K,IFRST,ILAST,IX,NATSGI
    real(chm_real) DELTA2,GAM
    real(chm_real) XI,YI,ZI,X1I,Y1I,Z1I,X2I,Y2I,Z2I
    real(chm_real) VXI,VYI,VZI,VXT,VYT,VZT
    real(chm_real) GXI,GYI,GZI,DXI,DYI,DZI
    real(chm_real) FXI,FYI,FZI,SGXI,SGYI,SGZI,SGXJ,SGYJ,SGZJ,SGXK,SGYK,SGZK
    real(chm_real) GX2I,GY2I,GZ2I,GX2J,GY2J,GZ2J,GX2K,GY2K,GZ2K
    real(chm_real) FACT0,FACT1,FACT2,AMASSI,AMASSJ,SGFTJ
    real(chm_real) A,B,RDX,RDY,RDZ,SGGV,SGVV,EKIN,EKINSG,SUMGAM
    !
    DELTA2=DELTA*DELTA
    !
    IFRST=ATFRST
    ILAST=ATLAST
    IF(ISGSTA > IFRST) IFRST=ISGSTA
    IF(ISGEND < ILAST) ILAST=ISGEND
    !
    ! build spatial average maps
    IF(SGTYPE>3)THEN
      CALL MAPBUILD(ATFRST,ATLAST,DELTA,IMOVE,AMASS,X,Y,Z)
    ENDIF
    !
    EKIN=ZERO
    EKINSG=ZERO
    SUMGAM=ZERO
    !
    !WXW Calculate constraint factor to eliminate energy input from guiding force
    do ia = atfrst,atlast
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif
       IF(IMOVE(I) == 0) THEN
          AMASSI=AMASS(I)
          XI=X(I)
          YI=Y(I)
          ZI=Z(I)
          VXI=VX(I)
          VYI=VY(I)
          VZI=VZ(I)
          DXI=DX(I)
          DYI=DY(I)
          DZI=DZ(I)
          IF(I>=ISGSTA .AND. I<=ISGEND)THEN
            ! average over SG structures
            NATSGI=NATSG(I)
            IF(NATSGI>1)THEN
              XI=XI*AMASSI
              YI=YI*AMASSI
              ZI=ZI*AMASSI
              DO IX=2,NATSGI-1
              J=IDATSG(IDXATSG(I)+IX)
              AMASSJ=AMASS(J)
              XI=XI+X(J)*AMASSJ 
              YI=YI+Y(J)*AMASSJ  
              ZI=ZI+Z(J)*AMASSJ 
              ENDDO 
              XI=XI/FMSG(I)
              YI=YI/FMSG(I)
              ZI=ZI/FMSG(I)
            ENDIF
            ! Obtain local averages
            IF(SGTYPE>3)THEN
              ! spatial average using map
              CALL MAPVALUES(AMASSI,XI,YI,ZI,GXI,GYI,GZI,SGXJ,SGYJ,SGZJ)
              GXI=GXI*DELTA
              GYI=GYI*DELTA
              GZI=GZI*DELTA
            ELSE
            X1I=SGAVG0*SGX1(I)+SGAVG1*XI
            Y1I=SGAVG0*SGY1(I)+SGAVG1*YI
            Z1I=SGAVG0*SGZ1(I)+SGAVG1*ZI
            X2I=SGAVG0*SGX2(I)+SGAVG1*X1I
            Y2I=SGAVG0*SGY2(I)+SGAVG1*Y1I
            Z2I=SGAVG0*SGZ2(I)+SGAVG1*Z1I
            SGX1(I)=X1I
            SGY1(I)=Y1I
            SGZ1(I)=Z1I
            SGX2(I)=X2I
            SGY2(I)=Y2I
            SGZ2(I)=Z2I
            ! avg(v)
            FACT1=TIMFAC*DELTA/TSGAVG
            GXI=FACT1*(XI-X1I)
            GYI=FACT1*(YI-Y1I)
            GZI=FACT1*(ZI-Z1I)
            VXT=(GXI-SGAVG0*SGVX(I))/SGAVG1
            VYT=(GYI-SGAVG0*SGVY(I))/SGAVG1
            VZT=(GZI-SGAVG0*SGVZ(I))/SGAVG1
            SGVX(I)=GXI
            SGVY(I)=GYI
            SGVZ(I)=GZI
            ! avg(f-avg(f))
            FACT1=-TIMFAC*AMASSI/DELTA/TSGAVG
            FACT2=-TIMFAC*DELTA/TSGAVG
            SGXJ=FACT1*(VXT+FACT2*(TWO*XI-THREE*X1I+X2I))
            SGYJ=FACT1*(VYT+FACT2*(TWO*YI-THREE*Y1I+Y2I))
            SGZJ=FACT1*(VZT+FACT2*(TWO*ZI-THREE*Z1I+Z2I))
            ENDIF /* SGTYPE */
            ! <avg(f-avg(f))avg(v)>
            SGFP(I)=SGAVP0*SGFP(I)+SGAVP1*(SGXJ*GXI+SGYJ*GYI+SGZJ*GZI)
            ! <avg(p)avg(v)>
            SGPP(I)=SGAVP0*SGPP(I)+SGAVP1*AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)/DELTA
            ! apparent friction constant
            IF(QSGGAMMA.AND.SGPP(I)>RSMALL)SGGAMMA(I)=SGFP(I)/SGPP(I)/TIMFAC
            SUMGAM=SUMGAM+SGGAMMA(I)
            ! guiding force
            FACT1=SGFTI*SGWT(I)*SGGAMMA(I)*TIMFAC*AMASSI/DELTA
            FACT2=SGFFI*SGWT(I)
            SGXJ=FACT1*GXI-FACT2*SGXJ
            SGYJ=FACT1*GYI-FACT2*SGYJ
            SGZJ=FACT1*GZI-FACT2*SGZJ
            ! <g.v>
            !FACT1=HALF*DELTA2/AMASSI
            !VXT=VXI-FACT1*(DXI-SGXJ)
            !VYT=VYI-FACT1*(DYI-SGYJ)
            !VZT=VZI-FACT1*(DZI-SGZJ)
            ! Using current velocities to avoid SHAKE complication
            VXT=VXI
            VYT=VYI
            VZT=VZI
            SGGV=(SGXJ*VXT+SGYJ*VYT+SGZJ*VZT)
            ! <v.v>
            SGVV=TWO*AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)/DELTA-SGGV*DELTA
            ! kinetic energies
            EKIN=EKIN+AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)
            EKINSG=EKINSG+AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)
            ! energy conservation friction constant
            IF(ABS(SGGV)<TEN*ABS(SGVV))THEN
              SGBETA(I)=TWO*SGGV/SGVV/TIMFAC
            ELSE
              SGBETA(I)=ZERO
            ENDIF
            !SGBETA(I)=SGGV(I)/SGVV(I)/TIMFAC
            SGXI=DXI-SGXJ
            SGYI=DYI-SGYJ
            SGZI=DZI-SGZJ
            SGSCAL=TIMFAC*SGBETA(I)*DELTA
            FACT0=TWO/(TWO+SGSCAL)
            FACT1=AMASSI/DELTA2
            FACT2=FACT1*FACT0*SGSCAL
            SGXI=FACT0*SGXI+FACT2*VXI
            SGYI=FACT0*SGYI+FACT2*VYI
            SGZI=FACT0*SGZI+FACT2*VZI
            DX(I)=SGXI
            DY(I)=SGYI
            DZ(I)=SGZI
          ENDIF
       ENDIF
    ENDDO
    !
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
    GCARR(1)=EKIN
    GCARR(2)=EKINSG
    GCARR(3) = SUMGAM
    CALL GCOMB(GCARR,3)
    EKIN=GCARR(1)
    EKINSG=GCARR(2)
    SUMGAM=GCARR(3)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    TIMMER=ECLOCK()
#endif
    TEMPLF=EKINSG/(KBOLTZ*NDEGFSG)/DELTA/DELTA
    TEMPHF=EKIN/(KBOLTZ*NDEGFSG)/DELTA/DELTA-TEMPLF
    SGSCAL=SUMGAM/(ISGEND-ISGSTA+1)
    !
    RETURN
  END SUBROUTINE SGMDW

  SUBROUTINE SGLDG(ATFRST,ATLAST,DELTA,NDEGF, &
               INLCKP,QNHLANG,isDrude,NHGAMMA,NHGAMMAD,  &
                   IMOVE,AMASS,X,Y,Z,VX,VY,VZ,DX,DY,DZ)
    !
    !     This routine calculates the guiding forces and adds them
    !     to the systematic forces.
    !         Assum SGFT*gamma=constant for all atoms
    !
    use chm_kinds
    use dimens_fcm
    use number
    use stream
    use consta
    use cnst_fcm
#if KEY_PARALLEL==1
    use parallel
#endif
    use machutil,only:eclock
    use prssre
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml
#endif
    implicit none
    !
    INTEGER ATFRST,ATLAST,NDEGF
    real(chm_real)  DELTA,NHGAMMA,NHGAMMAD
    INTEGER INLCKP(*)
    LOGICAL QNHLANG(*),isDrude(*)
    INTEGER IMOVE(*)
    real(chm_real)  AMASS(*),X(*),Y(*),Z(*),VX(*),VY(*),VZ(*), &
         DX(*),DY(*),DZ(*)
    !
#if KEY_PARALLEL==1
    real(chm_real) GCARR(31),TIMMER
#endif
    !
    INTEGER I,J,K,IFRST,ILAST,IA,TI,IX,NATSGI
    real(chm_real) XI,YI,ZI,X1I,Y1I,Z1I,X2I,Y2I,Z2I
    real(chm_real) VXI,VYI,VZI,VXT,VYT,VZT
    real(chm_real) FXI,FYI,FZI,GXI,GYI,GZI,DXI,DYI,DZI
    real(chm_real) SGXI,SGYI,SGZI,SGXJ,SGYJ,SGZJ,SGXK,SGYK,SGZK
    real(chm_real) GX2I,GY2I,GZ2I,GX2J,GY2J,GZ2J,GX2K,GY2K,GZ2K
    real(chm_real) GAM,GAMM,FACT0,FACT1,FACT2,DTEMP,SGFDJ,SGFTJ
    real(chm_real) AMASSI,AMASSJ,DELTA2,SGGV,SGVV,EKIN,EKINSG,SUMGAM
    !
    DELTA2=DELTA*DELTA
    !
    IFRST=ATFRST
    ILAST=ATLAST
    IF(ISGSTA > IFRST) IFRST=ISGSTA
    IF(ISGEND < ILAST) ILAST=ISGEND
    !
    ! build spatial average maps
    IF(SGTYPE>3)THEN
      CALL MAPBUILD(ATFRST,ATLAST,DELTA,IMOVE,AMASS,X,Y,Z)
    ENDIF
    !
    EKIN=ZERO
    EKINSG=ZERO
    SUMGAM=ZERO
    !WXW Calculate constraint factor to eliminate energy input from guiding force
    do ia = atfrst,atlast
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif
       IF(IMOVE(I) == 0) THEN
          AMASSI=AMASS(I)
          XI=X(I)
          YI=Y(I)
          ZI=Z(I)
          VXI=VX(I)
          VYI=VY(I)
          VZI=VZ(I)
          DXI=DX(I)
          DYI=DY(I)
          DZI=DZ(I)
          IF(I>=ISGSTA .AND. I<=ISGEND)THEN
            ti = INLCKP(i)
            j = i + 1
            if (QNHLANG(ti).and.(isDrude(i).or.isDrude(j)))then
              GAM=NHGAMMAD*DELTA
            else
              GAM=NHGAMMA*DELTA
            endif
            GAMM=TWO/(TWO+GAM)
            ! average over SG structures
            NATSGI=NATSG(I)
            IF(NATSGI>1)THEN
              XI=XI*AMASSI
              YI=YI*AMASSI
              ZI=ZI*AMASSI
              DO IX=2,NATSGI-1
              J=IDATSG(IDXATSG(I)+IX)
              AMASSJ=AMASS(J)
              XI=XI+X(J)*AMASSJ 
              YI=YI+Y(J)*AMASSJ  
              ZI=ZI+Z(J)*AMASSJ 
              ENDDO 
              XI=XI/FMSG(I)
              YI=YI/FMSG(I)
              ZI=ZI/FMSG(I)
            ENDIF
            ! Obtain local averages
            IF(SGTYPE>3)THEN
              ! spatial average using map
              CALL MAPVALUES(AMASSI,XI,YI,ZI,GXI,GYI,GZI,SGXI,SGYI,SGZI)
            ELSE
            ! avg(r)
            X1I=SGAVG0*SGX1(I)+SGAVG1*XI
            Y1I=SGAVG0*SGY1(I)+SGAVG1*YI
            Z1I=SGAVG0*SGZ1(I)+SGAVG1*ZI
            SGX1(I)=X1I
            SGY1(I)=Y1I
            SGZ1(I)=Z1I
            ! avg(avg(r))
            X2I=SGAVG0*SGX2(I)+SGAVG1*X1I
            Y2I=SGAVG0*SGY2(I)+SGAVG1*Y1I
            Z2I=SGAVG0*SGZ2(I)+SGAVG1*Z1I
            SGX2(I)=X2I
            SGY2(I)=Y2I
            SGZ2(I)=Z2I
            ! avg(v)
            FACT1=TIMFAC/TSGAVG
            GXI=FACT1*(XI-X1I)
            GYI=FACT1*(YI-Y1I)
            GZI=FACT1*(ZI-Z1I)
            ! v
            VXT=(GXI-SGAVG0*SGVX(I))/SGAVG1
            VYT=(GYI-SGAVG0*SGVY(I))/SGAVG1
            VZT=(GZI-SGAVG0*SGVZ(I))/SGAVG1
            SGVX(I)=GXI
            SGVY(I)=GYI
            SGVZ(I)=GZI
            ! avg(F-avg(F))
            FACT1=-TIMFAC*AMASSI/TSGAVG
            FACT2=-TIMFAC/TSGAVG
            SGXI=FACT1*(VXT+FACT2*(TWO*XI-THREE*X1I+X2I))
            SGYI=FACT1*(VYT+FACT2*(TWO*YI-THREE*Y1I+Y2I))
            SGZI=FACT1*(VZT+FACT2*(TWO*ZI-THREE*Z1I+Z2I))
            ENDIF /* SGTYPE */
            ! SGLD-GLE
            !random force averages
            SGXK=SGAVG0*SGKX(I)+SGAVG1*(DXI-SGDX(I))
            SGYK=SGAVG0*SGKY(I)+SGAVG1*(DYI-SGDY(I))
            SGZK=SGAVG0*SGKZ(I)+SGAVG1*(DZI-SGDZ(I))
            SGKX(I)=SGXK
            SGKY(I)=SGYK
            SGKZ(I)=SGZK
            FACT1=SGFDI*SGWT(I)*GAM*AMASSI/DELTA
            FACT2=FSGLDG*SGWT(I)
            !DXI=DXI-FACT1*GXI+FACT2*SGXK
            !DYI=DYI-FACT1*GYI+FACT2*SGYK
            !DZI=DZI-FACT1*GZI+FACT2*SGZK
            SGXJ=FACT1*GXI-FACT2*SGXK
            SGYJ=FACT1*GYI-FACT2*SGYK
            SGZJ=FACT1*GZI-FACT2*SGZK
            ! <avg(F-avg(F))avg(P)>
            SGFP(I)=SGAVP0*SGFP(I)+SGAVP1*(SGXI*GXI+SGYI*GYI+SGZI*GZI)
            ! <avg(P)avg(P)>
            SGPP(I)=SGAVP0*SGPP(I)+SGAVP1*AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)
            !SGGAMMA(I)
            IF(QSGGAMMA.AND.SGPP(I)>RSMALL)SGGAMMA(I)=SGFP(I)/SGPP(I)/TIMFAC
            SUMGAM=SUMGAM+SGGAMMA(I)
            !guiding forces
            FACT1=SGFTI*SGGAMMA(I)*TIMFAC*AMASSI*SGWT(I)
            FACT2=SGFFI*SGWT(I)
            SGXJ=SGXJ+FACT1*GXI-FACT2*SGXI
            SGYJ=SGYJ+FACT1*GYI-FACT2*SGYI
            SGZJ=SGZJ+FACT1*GZI-FACT2*SGZI
            ! <g.v>
            !FACT1=HALF*DELTA/AMASSI
            !VXT=VXI-FACT1*(DXI-SGXJ)
            !VYT=VYI-FACT1*(DYI-SGYJ)
            !VZT=VZI-FACT1*(DZI-SGZJ)
            ! Using current velocities to avoid SHAKE complication
            VXT=VXI
            VYT=VYI
            VZT=VZI
            SGGV=(SGXJ*VXI+SGYJ*VYI+SGZJ*VZI)
            ! <v.v>
            SGVV=TWO*AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)-SGGV*DELTA
            ! kinetic energies
            EKIN=EKIN+AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)
            EKINSG=EKINSG+AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)
            ! energy conservation friction constant
            IF(ABS(SGGV)<TEN*ABS(SGVV))THEN
              SGBETA(I)=(TWO+GAM)*SGGV/SGVV/TIMFAC
            ELSE
              SGBETA(I)=ZERO
            ENDIF
            DXI=DXI-SGXJ
            DYI=DYI-SGYJ
            DZI=DZI-SGZJ
            SGSCAL=TIMFAC*SGBETA(I)*DELTA
            GAMM=ONE+HALF*(GAM+SGSCAL)
            FACT1=(ONE+HALF*GAM)/GAMM
            FACT2=AMASSI*SGSCAL/GAMM/DELTA
            DXI=FACT1*DXI+FACT2*VXI
            DYI=FACT1*DYI+FACT2*VYI
            DZI=FACT1*DZI+FACT2*VZI
            DX(I)=DXI
            DY(I)=DYI
            DZ(I)=DZI
          ENDIF
       ENDIF
    ENDDO
    !
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
    GCARR(1)=EKIN
    GCARR(2)=EKINSG
    GCARR(3) = SUMGAM
    CALL GCOMB(GCARR,3)
    EKIN=GCARR(1)
    EKINSG=GCARR(2)
    SUMGAM=GCARR(3)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    TIMMER=ECLOCK()
#endif
    TEMPLF=EKINSG/(KBOLTZ*NDEGFSG)
    TEMPHF=EKIN/(KBOLTZ*NDEGFSG)-TEMPLF
    SGSCAL=SUMGAM/(ISGEND-ISGSTA+1)
!
    RETURN
  END SUBROUTINE SGLDG

  SUBROUTINE SGMDG(ATFRST,ATLAST,DELTA,NDEGF,MAXNOS,INLCKP,RTMPR,SNHV, &
                   IMOVE,AMASS,X,Y,Z,VX,VY,VZ,DX,DY,DZ)
    !
    !     This routine calculates the guiding forces and adds them
    !     to the systematic forces.
    !         Assum SGFT*gamma=constant for all atoms
    !
    use chm_kinds
    use clcg_mod,only:random
    use dimens_fcm
    use number
    use stream
    use consta
    use parallel
    use machutil,only:eclock
    use rndnum
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml
#endif
    implicit none
    !
    integer MAXNOS            ! INPUT   Maximum value for NOBL
    INTEGER ATFRST,ATLAST,NDEGF
    real(chm_real)  DELTA
    INTEGER IMOVE(*)
    integer,dimension(:) :: INLCKP  ! INPUT
    real(chm_real)  SNHV(MAXNOS)      ! I/O     NH velocities (zeta)
    real(chm_real)  RTMPR(MAXNOS)     ! INPUT   NH target temperatures
    real(chm_real)  AMASS(*)
    real(chm_real)  X(*),Y(*),Z(*),VX(*),VY(*),VZ(*), &
         DX(*),DY(*),DZ(*)
    !
    INTEGER IG,TI
#if KEY_PARALLEL==1
    real(chm_real) GCARR(30),TIMMER
#endif
    !
    INTEGER I,J,K,IA,IFRST,ILAST,IX,NATSGI
    real(chm_real) DELTA2,GAMM
    real(chm_real) XI,YI,ZI,X1I,Y1I,Z1I,X2I,Y2I,Z2I
    real(chm_real) VXI,VYI,VZI,VXT,VYT,VZT
    real(chm_real) GXI,GYI,GZI,DXI,DYI,DZI
    real(chm_real) FXI,FYI,FZI,SGXI,SGYI,SGZI,SGXJ,SGYJ,SGZJ,SGXK,SGYK,SGZK
    real(chm_real) GX2I,GY2I,GZ2I,GX2J,GY2J,GZ2J,GX2K,GY2K,GZ2K
    real(chm_real) FACT0,FACT1,FACT2,AMASSI,AMASSJ,SGFTJ
    real(chm_real) A,B,RDX,RDY,RDZ,SGGV,SGVV,EKIN,EKINSG,SUMGAM
    !
    DELTA2=DELTA*DELTA
    !
    IFRST=ATFRST
    ILAST=ATLAST
    IF(ISGSTA > IFRST) IFRST=ISGSTA
    IF(ISGEND < ILAST) ILAST=ISGEND
    !
    ! build spatial average maps
    IF(SGTYPE>3)THEN
      CALL MAPBUILD(ATFRST,ATLAST,DELTA,IMOVE,AMASS,X,Y,Z)
    ENDIF
    !
    EKIN=ZERO
    EKINSG=ZERO
    SUMGAM=ZERO
    !
    !WXW Calculate constraint factor to eliminate energy input from guiding force
    do ia = atfrst,atlast
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif
       IF(IMOVE(I) == 0) THEN
          AMASSI=AMASS(I)
          XI=X(I)
          YI=Y(I)
          ZI=Z(I)
          VXI=VX(I)
          VYI=VY(I)
          VZI=VZ(I)
          DXI=DX(I)
          DYI=DY(I)
          DZI=DZ(I)
          IF(I>=ISGSTA .AND. I<=ISGEND)THEN
            TI = INLCKP(I)
            FACT0=SNHV(TI)*AMASSI  ! instantaneous friction
            SGFTJ=SGFTI*SGWT(I)
            ! average over SG structures
            NATSGI=NATSG(I)
            IF(NATSGI>1)THEN
              XI=XI*AMASSI
              YI=YI*AMASSI
              ZI=ZI*AMASSI
              DO IX=2,NATSGI-1
              J=IDATSG(IDXATSG(I)+IX)
              AMASSJ=AMASS(J)
              XI=XI+X(J)*AMASSJ 
              YI=YI+Y(J)*AMASSJ  
              ZI=ZI+Z(J)*AMASSJ 
              ENDDO 
              XI=XI/FMSG(I)
              YI=YI/FMSG(I)
              ZI=ZI/FMSG(I)
            ENDIF
            ! Obtain local averages
            IF(SGTYPE>3)THEN
              ! spatial average using map
              CALL MAPVALUES(AMASSI,XI,YI,ZI,GXI,GYI,GZI,SGXI,SGYI,SGZI)
            ELSE
            X1I=SGAVG0*SGX1(I)+SGAVG1*XI
            Y1I=SGAVG0*SGY1(I)+SGAVG1*YI
            Z1I=SGAVG0*SGZ1(I)+SGAVG1*ZI
            X2I=SGAVG0*SGX2(I)+SGAVG1*X1I
            Y2I=SGAVG0*SGY2(I)+SGAVG1*Y1I
            Z2I=SGAVG0*SGZ2(I)+SGAVG1*Z1I
            SGX1(I)=X1I
            SGY1(I)=Y1I
            SGZ1(I)=Z1I
            SGX2(I)=X2I
            SGY2(I)=Y2I
            SGZ2(I)=Z2I
            ! avg(v)
            FACT1=TIMFAC/TSGAVG
            GXI=FACT1*(XI-X1I)
            GYI=FACT1*(YI-Y1I)
            GZI=FACT1*(ZI-Z1I)
            ! v
            VXT=(GXI-SGAVG0*SGVX(I))/SGAVG1
            VYT=(GYI-SGAVG0*SGVY(I))/SGAVG1
            VZT=(GZI-SGAVG0*SGVZ(I))/SGAVG1
            SGVX(I)=GXI
            SGVY(I)=GYI
            SGVZ(I)=GZI
            ! avg(F-avg(F))
            FACT1=-TIMFAC*AMASSI/TSGAVG
            FACT2=-TIMFAC/TSGAVG
            SGXI=FACT1*(VXT+FACT2*(TWO*XI-THREE*X1I+X2I))
            SGYI=FACT1*(VYT+FACT2*(TWO*YI-THREE*Y1I+Y2I))
            SGZI=FACT1*(VZT+FACT2*(TWO*ZI-THREE*Z1I+Z2I))
            ENDIF /* SGTYPE */
            ! <avg(F-avg(F))avg(P)>
            SGFP(I)=SGAVP0*SGFP(I)+SGAVP1*(SGXI*GXI+SGYI*GYI+SGZI*GZI)
            ! <avg(P)avg(P)>
            SGPP(I)=SGAVP0*SGPP(I)+SGAVP1*AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)
            ! apparent fraction constant
            IF(QSGGAMMA.AND.SGPP(I)>RSMALL)SGGAMMA(I)=SGFP(I)/SGPP(I)/TIMFAC
            SUMGAM=SUMGAM+SGGAMMA(I)
            !guiding forces
            FACT1=SGFTI*SGWT(I)*SGGAMMA(I)*TIMFAC*AMASSI
            FACT2=SGFFI*SGWT(I)
            SGXJ=FACT1*GXI-FACT2*SGXI
            SGYJ=FACT1*GYI-FACT2*SGYI
            SGZJ=FACT1*GZI-FACT2*SGZI
            ! <g.v>
            !FACT1=HALF*DELTA/AMASSI
            !VXT=VXI-FACT1*(DXI-SGXJ)
            !VYT=VYI-FACT1*(DYI-SGYJ)
            !VZT=VZI-FACT1*(DZI-SGZJ)
            ! Using current velocities to avoid SHAKE complication
            VXT=VXI
            VYT=VYI
            VZT=VZI
            SGGV=(SGXJ*VXT+SGYJ*VYT+SGZJ*VZT)
            ! <v.v>
            SGVV=TWO*AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)-SGGV*DELTA
            ! kinetic energies
            EKIN=EKIN+AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)
            EKINSG=EKINSG+AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)
           ! energy conservation friction constant
            IF(ABS(SGGV)<TEN*ABS(SGVV))THEN
              SGBETA(I)=TWO*SGGV/SGVV/TIMFAC
            ELSE
              SGBETA(I)=ZERO
            ENDIF
            DXI=DXI-SGXJ
            DYI=DYI-SGYJ
            DZI=DZI-SGZJ
            SGSCAL=TIMFAC*SGBETA(I)*DELTA
            GAMM=ONE+HALF*SGSCAL
            FACT1=ONE/GAMM
            FACT2=AMASSI*SGSCAL/GAMM/DELTA
            DXI=FACT1*DXI+FACT2*VXI
            DYI=FACT1*DYI+FACT2*VYI
            DZI=FACT1*DZI+FACT2*VZI
            DX(I)=DXI
            DY(I)=DYI
            DZ(I)=DZI
          ENDIF
       ENDIF
    ENDDO
    !
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
    GCARR(1)=EKIN
    GCARR(2)=EKINSG
    GCARR(3) = SUMGAM
    CALL GCOMB(GCARR,3)
    EKIN=GCARR(1)
    EKINSG=GCARR(2)
    SUMGAM=GCARR(3)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    TIMMER=ECLOCK()
#endif
    TEMPLF=EKINSG/(KBOLTZ*NDEGFSG)
    TEMPHF=EKIN/(KBOLTZ*NDEGFSG)-TEMPLF
    SGSCAL=SUMGAM/(ISGEND-ISGSTA+1)
    !
    RETURN
  END SUBROUTINE SGMDG

!#if KEY_SGGRID==1

  SUBROUTINE SGLDM(ATFRST,ATLAST,DELTA,NDEGF,TEMPI, &
                   IMOVE,AMASS,X,Y,Z,VX,VY,VZ,DX,DY,DZ)
    !
    !     This routine uses spatial average to calculate the guiding forces 
    !     and adds them to the systematic forces.
    !         Assum SGFT*gamma=constant for all atoms
    !
    use chm_kinds
    use dimens_fcm
    use number
    use stream
    use consta
    use cnst_fcm
#if KEY_PARALLEL==1
    use parallel
#endif
    use machutil,only:eclock
    use prssre
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml
#endif
    implicit none
    !
    INTEGER ATFRST,ATLAST,NDEGF
    real(chm_real)  DELTA,TEMPI
    INTEGER IMOVE(*)
    real(chm_real)  AMASS(*),X(*),Y(*),Z(*),VX(*),VY(*),VZ(*), &
         DX(*),DY(*),DZ(*)
    !
#if KEY_PARALLEL==1
    real(chm_real) GCARR(31),TIMMER
#endif
    !
    INTEGER I,J,K,IFRST,ILAST,IA,IX,NATSGI
    real(chm_real) XI,YI,ZI,X1I,Y1I,Z1I,X2I,Y2I,Z2I
    real(chm_real) VXI,VYI,VZI,VXT,VYT,VZT
    real(chm_real) FXI,FYI,FZI,GXI,GYI,GZI,DXI,DYI,DZI
    real(chm_real) SGXI,SGYI,SGZI,SGXJ,SGYJ,SGZJ,SGXK,SGYK,SGZK
    real(chm_real) GX2I,GY2I,GZ2I,GX2J,GY2J,GZ2J,GX2K,GY2K,GZ2K
    real(chm_real) GAM,GAMM,FACT0,FACT1,FACT2,DTEMP,SGFTJ
    real(chm_real) AMASSI,AMASSJ,DELTA2,SGGV,SGVV,EKIN,EKINSG,SUMGAM
    !
    DELTA2=DELTA*DELTA
    !
    IFRST=ATFRST
    ILAST=ATLAST
    IF(ISGSTA > IFRST) IFRST=ISGSTA
    IF(ISGEND < ILAST) ILAST=ISGEND
    !
    ! build spatial average maps
    IF(SGTYPE>3)THEN
      CALL MAPBUILD(ATFRST,ATLAST,DELTA,IMOVE,AMASS,X,Y,Z)
    ENDIF
    !
    EKIN=ZERO
    EKINSG=ZERO
    
    SUMGAM=ZERO
    !
    !WXW Calculate constraint factor to eliminate energy input from guiding force
    do ia = atfrst,atlast
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif
       IF(IMOVE(I) == 0) THEN
          AMASSI=AMASS(I)
          XI=X(I)
          YI=Y(I)
          ZI=Z(I)
          VXI=VX(I)
          VYI=VY(I)
          VZI=VZ(I)
          DXI=DX(I)
          DYI=DY(I)
          DZI=DZ(I)
          IF(I>=ISGSTA .AND. I<=ISGEND)THEN
            GAM=TIMFAC*FBETA(I)*DELTA
            GAMM=TWO/(TWO+GAM)
            ! average over SG structures
            NATSGI=NATSG(I)
            IF(NATSGI>1)THEN
              XI=XI*AMASSI
              YI=YI*AMASSI
              ZI=ZI*AMASSI
              DO IX=2,NATSGI-1
              J=IDATSG(IDXATSG(I)+IX)
              AMASSJ=AMASS(J)
              XI=XI+X(J)*AMASSJ 
              YI=YI+Y(J)*AMASSJ  
              ZI=ZI+Z(J)*AMASSJ 
              ENDDO 
              XI=XI/FMSG(I)
              YI=YI/FMSG(I)
              ZI=ZI/FMSG(I)
            ENDIF
            ! Obtain local averages
            IF(SGTYPE>3)THEN
              ! spatial average using map
              CALL MAPVALUES(AMASSI,XI,YI,ZI,GXI,GYI,GZI,SGXJ,SGYJ,SGZJ)
            ELSE
            X1I=SGAVG0*SGX1(I)+SGAVG1*XI
            Y1I=SGAVG0*SGY1(I)+SGAVG1*YI
            Z1I=SGAVG0*SGZ1(I)+SGAVG1*ZI
            X2I=SGAVG0*SGX2(I)+SGAVG1*X1I
            Y2I=SGAVG0*SGY2(I)+SGAVG1*Y1I
            Z2I=SGAVG0*SGZ2(I)+SGAVG1*Z1I
            SGX1(I)=X1I
            SGY1(I)=Y1I
            SGZ1(I)=Z1I
            SGX2(I)=X2I
            SGY2(I)=Y2I
            SGZ2(I)=Z2I
            ! avg(v)
            FACT1=TIMFAC*DELTA/TSGAVG
            GXI=FACT1*(XI-X1I)
            GYI=FACT1*(YI-Y1I)
            GZI=FACT1*(ZI-Z1I)
            VXT=(GXI-SGAVG0*SGVX(I))/SGAVG1
            VYT=(GYI-SGAVG0*SGVY(I))/SGAVG1
            VZT=(GZI-SGAVG0*SGVZ(I))/SGAVG1
            SGVX(I)=GXI
            SGVY(I)=GYI
            SGVZ(I)=GZI
            ! average force deviation 
            FACT1=-TIMFAC*AMASSI/DELTA/TSGAVG
            FACT2=-TIMFAC*DELTA/TSGAVG
            SGXJ=FACT1*(VXT+FACT2*(TWO*XI-THREE*X1I+X2I))
            SGYJ=FACT1*(VYT+FACT2*(TWO*YI-THREE*Y1I+Y2I))
            SGZJ=FACT1*(VZT+FACT2*(TWO*ZI-THREE*Z1I+Z2I))
            ENDIF /* SGTYPE */
            ! SGLD-GLE
            !random force averages
            SGXK=SGAVG0*SGKX(I)+SGAVG1*(DXI-SGDX(I))
            SGYK=SGAVG0*SGKY(I)+SGAVG1*(DYI-SGDY(I))
            SGZK=SGAVG0*SGKZ(I)+SGAVG1*(DZI-SGDZ(I))
            SGKX(I)=SGXK
            SGKY(I)=SGYK
            SGKZ(I)=SGZK
            FACT1=SGFDI*SGWT(I)*GAM*AMASSI/DELTA2
            FACT2=FSGLDG*SGWT(I)
            !DXI=DXI-FACT1*GXI+FACT2*SGXK
            !DYI=DYI-FACT1*GYI+FACT2*SGYK
            !DZI=DZI-FACT1*GZI+FACT2*SGZK
            SGXI=FACT1*GXI-FACT2*SGXK
            SGYI=FACT1*GYI-FACT2*SGYK
            SGZI=FACT1*GZI-FACT2*SGZK

            ! <avg(f-avg(f))avg(v)>
            SGFP(I)=SGAVP0*SGFP(I)+SGAVP1*(SGXJ*GXI+SGYJ*GYI+SGZJ*GZI)
            ! <avg(p)avg(v)>
            SGPP(I)=SGAVP0*SGPP(I)+SGAVP1*AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)/DELTA
            ! force friction constant
            IF(QSGGAMMA.AND.SGPP(I)>RSMALL)SGGAMMA(I)=SGFP(I)/SGPP(I)/TIMFAC
            SUMGAM=SUMGAM+SGGAMMA(I)
            ! guiding force
            FACT1=SGFTI*SGWT(I)*SGGAMMA(I)*TIMFAC*AMASSI*SGWT(I)/DELTA
            FACT2=SGFFI*SGWT(I)
            SGXI=SGXI+FACT1*GXI-FACT2*SGXJ
            SGYI=SGYI+FACT1*GYI-FACT2*SGYJ
            SGZI=SGZI+FACT1*GZI-FACT2*SGZJ
            ! <g.v>
            !FACT1=HALF*DELTA2/AMASSI
            !VXT=VXI-FACT1*(DXI-SGXI)
            !VYT=VYI-FACT1*(DYI-SGYI)
            !VZT=VZI-FACT1*(DZI-SGZI)
            ! Using current velocities to avoid SHAKE complication
            VXT=VXI
            VYT=VYI
            VZT=VZI
            SGGV=(SGXI*VXT+SGYI*VYT+SGZI*VZT)
            ! <v.v>
            SGVV=TWO*AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)/DELTA-SGGV*DELTA
            ! kinetic energies
            EKIN=EKIN+AMASSI*(VXT*VXT+VYT*VYT+VZT*VZT)
            EKINSG=EKINSG+AMASSI*(GXI*GXI+GYI*GYI+GZI*GZI)
            ! energy conservation friction constant
            IF(ABS(SGGV)<TEN*ABS(SGVV))THEN
              SGBETA(I)=(TWO+GAM)*SGGV/SGVV/TIMFAC
            ELSE
              SGBETA(I)=ZERO
            ENDIF
            SGXI=DXI-SGXI
            SGYI=DYI-SGYI
            SGZI=DZI-SGZI
            SGSCAL=TIMFAC*SGBETA(I)*DELTA
            GAMM=ONE+HALF*(GAM+SGSCAL)
            FACT1=(ONE+HALF*GAM)/GAMM
            FACT2=AMASSI*SGSCAL/GAMM/DELTA2
            SGXI=FACT1*SGXI+FACT2*VXI
            SGYI=FACT1*SGYI+FACT2*VYI
            SGZI=FACT1*SGZI+FACT2*VZI
            DX(I)=SGXI
            DY(I)=SGYI
            DZ(I)=SGZI
          ENDIF
       ENDIF
    ENDDO
    !
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
    GCARR(1)=EKIN
    GCARR(2)=EKINSG
    GCARR(3) = SUMGAM
    CALL GCOMB(GCARR,3)
    EKIN=GCARR(1)
    EKINSG=GCARR(2)
    SUMGAM=GCARR(3)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    TIMMER=ECLOCK()
#endif
    TEMPLF=EKINSG/(KBOLTZ*NDEGFSG)/DELTA/DELTA
    TEMPHF=EKIN/(KBOLTZ*NDEGFSG)/DELTA/DELTA-TEMPLF
    SGSCAL=SUMGAM/(ISGEND-ISGSTA+1)
    !
    RETURN
  END SUBROUTINE SGLDM

  SUBROUTINE MAPBUILD(atfrst,atlast,DELTA,IMOVE,AMASS,X,Y,Z)
  !-----------------------------------------------------------------------
  !     This routine build velocity and acceleration maps
  !
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use stream
  use parallel
  use machutil,only:eclock
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec,natoml,atoml
#endif
  implicit none
  INTEGER atfrst,atlast,IMOVE(*)
  real(chm_real) X(*),Y(*),Z(*),AMASS(*)
  real(chm_real) DELTA,XI,YI,ZI,AMASSI
  INTEGER X0,Y0,Z0,X1,Y1,Z1
  INTEGER I000,I100,I010,I001,I110,I101,I011,I111
  real(chm_real) GX,GY,GZ,A0,B0,C0,A1,B1,C1
  real(chm_real) ABC000,ABC100,ABC010,ABC001,ABC110,ABC101,ABC011,ABC111
  INTEGER IA,I
    real(chm_real) X1I,Y1I,Z1I,X2I,Y2I,Z2I
    real(chm_real) GXI,GYI,GZI,VXT,VYT,VZT,FXI,FYI,FZI
    real(chm_real) FACT1,FACT2,RHOMI
#if KEY_PARALLEL==1
    real(chm_real) TIMMER
#endif
  !
  ! check mapsize
  CALL MAPINIT(atfrst,atlast,X,Y,Z)
  ! interpolate structure to protein map
  !WXW Calculate constraint factor to eliminate energy input from guiding force
    do ia = atfrst,atlast
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif
       IF(IMOVE(I) == 0) THEN
          AMASSI=AMASS(I)
          XI=X(I)
          YI=Y(I)
          ZI=Z(I)
          IF(I>=ISGSTA .AND. I<=ISGEND)THEN
            X1I=SGAVG0*SGX1(I)+SGAVG1*XI
            Y1I=SGAVG0*SGY1(I)+SGAVG1*YI
            Z1I=SGAVG0*SGZ1(I)+SGAVG1*ZI
            X2I=SGAVG0*SGX2(I)+SGAVG1*X1I
            Y2I=SGAVG0*SGY2(I)+SGAVG1*Y1I
            Z2I=SGAVG0*SGZ2(I)+SGAVG1*Z1I
            SGX1(I)=X1I
            SGY1(I)=Y1I
            SGZ1(I)=Z1I
            SGX2(I)=X2I
            SGY2(I)=Y2I
            SGZ2(I)=Z2I
            ! avg(v)
            FACT1=TIMFAC/TSGAVG
            GXI=FACT1*(XI-X1I)
            GYI=FACT1*(YI-Y1I)
            GZI=FACT1*(ZI-Z1I)
            VXT=(GXI-SGAVG0*SGVX(I))/SGAVG1
            VYT=(GYI-SGAVG0*SGVY(I))/SGAVG1
            VZT=(GZI-SGAVG0*SGVZ(I))/SGAVG1
            SGVX(I)=GXI
            SGVY(I)=GYI
            SGVZ(I)=GZI
            ! average acceleration deviation 
            FACT1=-TIMFAC/TSGAVG
            FACT2=-TIMFAC/TSGAVG
            FXI=FACT1*(VXT+FACT2*(TWO*XI-THREE*X1I+X2I))
            FYI=FACT1*(VYT+FACT2*(TWO*YI-THREE*Y1I+Y2I))
            FZI=FACT1*(VZT+FACT2*(TWO*ZI-THREE*Z1I+Z2I))
            GX=(XI-GXMIN)/SGSIZE+ONE
            GY=(YI-GYMIN)/SGSIZE+ONE
            GZ=(ZI-GZMIN)/SGSIZE+ONE
            X0=INT(GX)
            Y0=INT(GY)
            Z0=INT(GZ)
            IF(X0<1.OR.Y0<1.OR.Z0<1)THEN
              STOP 'Position outside of lower boundary'
            ENDIF
            X1=X0+1
            Y1=Y0+1
            Z1=Z0+1
            IF(X1>NGRIDX.OR.Y1>NGRIDY.OR.Z1>NGRIDZ)THEN
              write(outu,*)'I= ',i,xi,yi,zi,x1,y1,z1
              STOP 'Position outside of higher boundary'
            ENDIF
            A0=X1-GX
            B0=Y1-GY
            C0=Z1-GZ
            A1=ONE-A0
            B1=ONE-B0
            C1=ONE-C0
            ABC000=A0*B0*C0*AMASSI
            ABC100=A1*B0*C0*AMASSI
            ABC010=A0*B1*C0*AMASSI
            ABC001=A0*B0*C1*AMASSI
            ABC110=A1*B1*C0*AMASSI
            ABC011=A0*B1*C1*AMASSI
            ABC101=A1*B0*C1*AMASSI
            ABC111=A1*B1*C1*AMASSI
            I000=X0+NGRIDX*(Y0-1+NGRIDY*(Z0-1))
            I100=I000+1
            I010=I000+NGRIDX
            I001=I000+NGRIDXY
            I110=I010+1
            I101=I001+1
            I011=I001+NGRIDX
            I111=I011+1
            RHOM(I000)=RHOM(I000)+ABC111
            RHOM(I100)=RHOM(I100)+ABC011
            RHOM(I010)=RHOM(I010)+ABC101
            RHOM(I001)=RHOM(I001)+ABC110
            RHOM(I110)=RHOM(I110)+ABC001
            RHOM(I101)=RHOM(I101)+ABC010
            RHOM(I011)=RHOM(I011)+ABC100
            RHOM(I111)=RHOM(I111)+ABC000
            RHOVX(I000)=RHOVX(I000)+ABC111*GXI
            RHOVY(I000)=RHOVY(I000)+ABC111*GYI
            RHOVZ(I000)=RHOVZ(I000)+ABC111*GZI
            RHOVX(I100)=RHOVX(I100)+ABC011*GXI
            RHOVY(I100)=RHOVY(I100)+ABC011*GYI
            RHOVZ(I100)=RHOVZ(I100)+ABC011*GZI
            RHOVX(I010)=RHOVX(I010)+ABC101*GXI
            RHOVY(I010)=RHOVY(I010)+ABC101*GYI
            RHOVZ(I010)=RHOVZ(I010)+ABC101*GZI
            RHOVX(I001)=RHOVX(I001)+ABC110*GXI
            RHOVY(I001)=RHOVY(I001)+ABC110*GYI
            RHOVZ(I001)=RHOVZ(I001)+ABC110*GZI
            RHOVX(I110)=RHOVX(I110)+ABC001*GXI
            RHOVY(I110)=RHOVY(I110)+ABC001*GYI
            RHOVZ(I110)=RHOVZ(I110)+ABC001*GZI
            RHOVX(I101)=RHOVX(I101)+ABC010*GXI
            RHOVY(I101)=RHOVY(I101)+ABC010*GYI
            RHOVZ(I101)=RHOVZ(I101)+ABC010*GZI
            RHOVX(I011)=RHOVX(I011)+ABC100*GXI
            RHOVY(I011)=RHOVY(I011)+ABC100*GYI
            RHOVZ(I011)=RHOVZ(I011)+ABC100*GZI
            RHOVX(I111)=RHOVX(I111)+ABC000*GXI
            RHOVY(I111)=RHOVY(I111)+ABC000*GYI
            RHOVZ(I111)=RHOVZ(I111)+ABC000*GZI

            RHOAX(I000)=RHOAX(I000)+ABC111*FXI
            RHOAY(I000)=RHOAY(I000)+ABC111*FYI
            RHOAZ(I000)=RHOAZ(I000)+ABC111*FZI
            RHOAX(I100)=RHOAX(I100)+ABC011*FXI
            RHOAY(I100)=RHOAY(I100)+ABC011*FYI
            RHOAZ(I100)=RHOAZ(I100)+ABC011*FZI
            RHOAX(I010)=RHOAX(I010)+ABC101*FXI
            RHOAY(I010)=RHOAY(I010)+ABC101*FYI
            RHOAZ(I010)=RHOAZ(I010)+ABC101*FZI
            RHOAX(I001)=RHOAX(I001)+ABC110*FXI
            RHOAY(I001)=RHOAY(I001)+ABC110*FYI
            RHOAZ(I001)=RHOAZ(I001)+ABC110*FZI
            RHOAX(I110)=RHOAX(I110)+ABC001*FXI
            RHOAY(I110)=RHOAY(I110)+ABC001*FYI
            RHOAZ(I110)=RHOAZ(I110)+ABC001*FZI
            RHOAX(I101)=RHOAX(I101)+ABC010*FXI
            RHOAY(I101)=RHOAY(I101)+ABC010*FYI
            RHOAZ(I101)=RHOAZ(I101)+ABC010*FZI
            RHOAX(I011)=RHOAX(I011)+ABC100*FXI
            RHOAY(I011)=RHOAY(I011)+ABC100*FYI
            RHOAZ(I011)=RHOAZ(I011)+ABC100*FZI
            RHOAX(I111)=RHOAX(I111)+ABC000*FXI
            RHOAY(I111)=RHOAY(I111)+ABC000*FYI
            RHOAZ(I111)=RHOAZ(I111)+ABC000*FZI
       ENDIF  /* ISGSTA */
     ENDIF /* IMOVE */
  enddo 
    !
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
    CALL GCOMB(RHOVX,NGRIDXYZ)
    CALL GCOMB(RHOVY,NGRIDXYZ)
    CALL GCOMB(RHOVZ,NGRIDXYZ)
    CALL GCOMB(RHOAX,NGRIDXYZ)
    CALL GCOMB(RHOAY,NGRIDXYZ)
    CALL GCOMB(RHOAZ,NGRIDXYZ)
    CALL GCOMB(RHOM,NGRIDXYZ)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    TIMMER=ECLOCK()
#endif
  ! remove net momentum and forces
  IF(QSGCOM)THEN
    GXI=ZERO
    GYI=ZERO
    GZI=ZERO
    FXI=ZERO
    FYI=ZERO
    FZI=ZERO
  ELSE
    AMASSI=SUM(RHOM)
    GXI=SUM(RHOVX)/AMASSI
    GYI=SUM(RHOVY)/AMASSI
    GZI=SUM(RHOVZ)/AMASSI
    FXI=SUM(RHOAX)/AMASSI
    FYI=SUM(RHOAY)/AMASSI
    FZI=SUM(RHOAZ)/AMASSI
  ENDIF
  DO I=1,NGRIDXYZ
    RHOMI=RHOM(I)
    IF(RHOMI>RSMALL)THEN
      RHOVX(I)=RHOVX(I)/RHOMI
      RHOVY(I)=RHOVY(I)/RHOMI
      RHOVZ(I)=RHOVZ(I)/RHOMI
      RHOAX(I)=RHOAX(I)/RHOMI
      RHOAY(I)=RHOAY(I)/RHOMI
      RHOAZ(I)=RHOAZ(I)/RHOMI
    ENDIF
  ENDDO
  IF(.NOT. QSGCOM)THEN
    RHOVX=RHOVX-GXI
    RHOVY=RHOVY-GYI
    RHOVZ=RHOVZ-GZI
    RHOAX=RHOAX-FXI
    RHOAY=RHOAY-FYI
    RHOAZ=RHOAZ-FZI
  ENDIF
  RETURN
END SUBROUTINE MAPBUILD

SUBROUTINE MAPVALUES(AMASSI,XI,YI,ZI,GXI,GYI,GZI,FXI,FYI,FZI)
  !-----------------------------------------------------------------------
  !     This routine to find map values at input position
  !     throuth linear interpolation
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  real(chm_real) XI,YI,ZI,AMASSI
  real(chm_real) GXI,GYI,GZI,FXI,FYI,FZI
  INTEGER X0,Y0,Z0,X1,Y1,Z1
  INTEGER I000,I100,I010,I001,I110,I101,I011,I111
  real(chm_real) GX,GY,GZ,A0,B0,C0,A1,B1,C1
  real(chm_real) ABC000,ABC100,ABC010,ABC001,ABC110,ABC101,ABC011,ABC111
     GX=(XI-GXMIN)/SGSIZE+ONE
     GY=(YI-GYMIN)/SGSIZE+ONE
     GZ=(ZI-GZMIN)/SGSIZE+ONE
     X0=INT(GX)
     Y0=INT(GY)
     Z0=INT(GZ)
     IF(X0<1.OR.Y0<1.OR.Z0<1)THEN
       STOP 'Position outside of lower boundary'
     ENDIF
     X1=X0+1
     Y1=Y0+1
     Z1=Z0+1
     IF(X1>NGRIDX.OR.Y1>NGRIDY.OR.Z1>NGRIDZ)THEN
       STOP 'Position outside of higher boundary'
     ENDIF
     A0=X1-GX
     B0=Y1-GY
     C0=Z1-GZ
     A1=ONE-A0
     B1=ONE-B0
     C1=ONE-C0
     ABC000=A0*B0*C0
     ABC100=A1*B0*C0
     ABC010=A0*B1*C0
     ABC001=A0*B0*C1
     ABC110=A1*B1*C0
     ABC011=A0*B1*C1
     ABC101=A1*B0*C1
     ABC111=A1*B1*C1
     I000=X0+NGRIDX*(Y0-1+NGRIDY*(Z0-1))
     I100=I000+1
     I010=I000+NGRIDX
     I001=I000+NGRIDXY
     I110=I010+1
     I101=I001+1
     I011=I001+NGRIDX
     I111=I011+1
     GXI=ABC111*RHOVX(I000)+ABC110*RHOVX(I001)+ABC101*RHOVX(I010)+ABC011*RHOVX(I100)+     &
     ABC100*RHOVX(I011)+ABC010*RHOVX(I101)+ABC001*RHOVX(I110)+ABC000*RHOVX(I111)
     GYI=ABC111*RHOVY(I000)+ABC110*RHOVY(I001)+ABC101*RHOVY(I010)+ABC011*RHOVY(I100)+     &
     ABC100*RHOVY(I011)+ABC010*RHOVY(I101)+ABC001*RHOVY(I110)+ABC000*RHOVY(I111)
     GZI=ABC111*RHOVZ(I000)+ABC110*RHOVZ(I001)+ABC101*RHOVZ(I010)+ABC011*RHOVZ(I100)+     &
     ABC100*RHOVZ(I011)+ABC010*RHOVZ(I101)+ABC001*RHOVZ(I110)+ABC000*RHOVZ(I111)
     FXI=ABC111*RHOAX(I000)+ABC110*RHOAX(I001)+ABC101*RHOAX(I010)+ABC011*RHOAX(I100)+     &
     ABC100*RHOAX(I011)+ABC010*RHOAX(I101)+ABC001*RHOAX(I110)+ABC000*RHOAX(I111)
     FYI=ABC111*RHOAY(I000)+ABC110*RHOAY(I001)+ABC101*RHOAY(I010)+ABC011*RHOAY(I100)+     &
     ABC100*RHOAY(I011)+ABC010*RHOAY(I101)+ABC001*RHOAY(I110)+ABC000*RHOAY(I111)
     FZI=ABC111*RHOAZ(I000)+ABC110*RHOAZ(I001)+ABC101*RHOAZ(I010)+ABC011*RHOAZ(I100)+     &
     ABC100*RHOAZ(I011)+ABC010*RHOAZ(I101)+ABC001*RHOAZ(I110)+ABC000*RHOAZ(I111)
     FXI=FXI*AMASSI
     FYI=FYI*AMASSI
     FZI=FZI*AMASSI
     RETURN
  END SUBROUTINE MAPVALUES
  

  SUBROUTINE MAPINIT(atfrst,atlast,X,Y,Z)
  !-----------------------------------------------------------------------
  !     This routine build and initialize maps
  !       (Periodic boundary condition is not considered)
  !
  !-----------------------------------------------------------------------
  use chm_kinds
  use cnst_fcm
  use memory
  use number
  use parallel
  use machutil,only:eclock
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec,natoml,atoml
#endif
  implicit none
  integer atfrst,atlast
  real(chm_real) X(*),Y(*),Z(*)

  integer I,IA
  real(chm_real) XI,YI,ZI
  real(chm_real) XMAX,XMIN,YMAX,YMIN,ZMAX,ZMIN,FACT
  logical rmap
#if KEY_PARALLEL==1
    real(chm_real) TIMMER
#endif
  XMIN=RBIG
  XMAX=-RBIG
  YMIN=RBIG
  YMAX=-RBIG
  ZMIN=RBIG
  ZMAX=-RBIG
    do ia = atfrst,atlast
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif

       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
       IF(XI>XMAX)XMAX=XI
       IF(XI<XMIN)XMIN=XI
       IF(YI>YMAX)YMAX=YI
       IF(YI<YMIN)YMIN=YI
       IF(ZI>ZMAX)ZMAX=ZI
       IF(ZI<ZMIN)ZMIN=ZI
  ENDDO
#if KEY_PARALLEL==1
    CALL PSYNC()
    TIMMER=ECLOCK()
  CALL GCOMBMAX(XMAX)
  CALL GCOMBMAX(YMAX)
  CALL GCOMBMAX(ZMAX)
  FACT=-XMIN
  CALL GCOMBMAX(FACT)
  XMIN=-FACT
  FACT=-YMIN
  CALL GCOMBMAX(FACT)
  YMIN=-FACT
  FACT=-ZMIN
  CALL GCOMBMAX(FACT)
  ZMIN=-FACT
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    TIMMER=ECLOCK()
#endif
  RMAP=(XMAX>GXMAX.OR.XMIN<GXMIN.OR.YMAX>GYMAX.OR.YMIN<GYMIN.OR.ZMAX>GZMAX.OR.ZMIN<GZMIN)
  IF(RMAP)THEN
    ! deallocate maps
    if(allocated(rhom))call chmdealloc('sglds.src','MAPCHECK','RHOM',NGRIDXYZ,crl=RHOM)
    if(allocated(rhovx))call chmdealloc('sglds.src','MAPCHECK','RHOVX',NGRIDXYZ,crl=RHOVX)
    if(allocated(rhovy))call chmdealloc('sglds.src','MAPCHECK','RHOVY',NGRIDXYZ,crl=RHOVY)
    if(allocated(rhovz))call chmdealloc('sglds.src','MAPCHECK','RHOVZ',NGRIDXYZ,crl=RHOVZ)
    if(allocated(rhoax))call chmdealloc('sglds.src','MAPCHECK','RHOAX',NGRIDXYZ,crl=RHOAX)
    if(allocated(rhoay))call chmdealloc('sglds.src','MAPCHECK','RHOAY',NGRIDXYZ,crl=RHOAY)
    if(allocated(rhoaz))call chmdealloc('sglds.src','MAPCHECK','RHOAZ',NGRIDXYZ,crl=RHOAZ)

    GXMAX=SGSIZE*(INT(XMAX/SGSIZE)+ONE)
    GXMIN=SGSIZE*(INT(XMIN/SGSIZE)-ONE)
    GYMAX=SGSIZE*(INT(YMAX/SGSIZE)+ONE)
    GYMIN=SGSIZE*(INT(YMIN/SGSIZE)-ONE)
    GZMAX=SGSIZE*(INT(ZMAX/SGSIZE)+ONE)
    GZMIN=SGSIZE*(INT(ZMIN/SGSIZE)-ONE)
    NGRIDX=INT((GXMAX-GXMIN)/SGSIZE)+1
    NGRIDY=INT((GYMAX-GYMIN)/SGSIZE)+1
    NGRIDZ=INT((GZMAX-GZMIN)/SGSIZE)+1
    NGRIDXY=NGRIDX*NGRIDY
    NGRIDYZ=NGRIDY*NGRIDZ
    NGRIDZX=NGRIDZ*NGRIDX
    NGRIDXYZ=NGRIDX*NGRIDY*NGRIDZ
        ! allocate sgmaps
    IF(.not.allocated(RHOM))call chmalloc('sglds.src','SGINIT','RHOM',NGRIDXYZ,crl=RHOM)
    IF(.not.allocated(RHOVX))call chmalloc('sglds.src','SGINIT','RHOVX',NGRIDXYZ,crl=RHOVX)
    IF(.not.allocated(RHOVY))call chmalloc('sglds.src','SGINIT','RHOVY',NGRIDXYZ,crl=RHOVY)
    IF(.not.allocated(RHOVZ))call chmalloc('sglds.src','SGINIT','RHOVZ',NGRIDXYZ,crl=RHOVZ)
    IF(.not.allocated(RHOAX))call chmalloc('sglds.src','SGINIT','RHOAX',NGRIDXYZ,crl=RHOAX)
    IF(.not.allocated(RHOAY))call chmalloc('sglds.src','SGINIT','RHOAY',NGRIDXYZ,crl=RHOAY)
    IF(.not.allocated(RHOAZ))call chmalloc('sglds.src','SGINIT','RHOAZ',NGRIDXYZ,crl=RHOAZ)
  ENDIF
    RHOM=ZERO
    RHOVX=ZERO
    RHOVY=ZERO
    RHOVZ=ZERO
    RHOAX=ZERO
    RHOAY=ZERO
    RHOAZ=ZERO
    RETURN
  END SUBROUTINE MAPINIT
!#endif /* (sggrid) */

#endif /* (sgld_main)*/
  SUBROUTINE NULL_SGLD
    RETURN
  END SUBROUTINE NULL_SGLD
end module sgld
