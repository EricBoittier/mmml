module scalar_module
  use chm_kinds
  implicit none

contains

SUBROUTINE SCALAR
  !-----------------------------------------------------------------------
  !     THIS ROUTINE PARSES THE KEYNAME FROM THE SCALAR COMMAND
  !
  !     BRB 9-DEC-1983
  !
  use genborn,only: alph_gb,sigx_gb,sigy_gb,sigz_gb,t_gb,      & 
       gb_atm,qanalys                                         

#if KEY_CHEQ==1
  use cheq,only:ech,eha, DCH,SUMDCH,DDCH                        
#endif
#if KEY_FLUCQ==1
  use flucqm,only: fqcfor       
#endif

  use chm_kinds
  use dimens_fcm
  use bases_fcm
  use psf
  use coord
  use coordc
  use cnst_fcm
  use deriv
  use fourdm
#if KEY_ACE==1
  use ace_module,only:bsolv,cgiacp,cgsacp      
#endif
#if KEY_PERT==1
  use pert       
#endif
  use econtmod
  use comand
  use param
#if KEY_SGLD==1
  use sgld     
#endif
  use stream
  use string
#if KEY_FLUCQ==1
  use flucq    
#endif
  use gcmc
  !EPO variable LJ cutoff
  use varcutm
  use memory

  use shake, only: qshake
  
  implicit none
  real(chm_real),allocatable,dimension(:) :: iw,jw
  integer,allocatable,dimension(:) :: FLAGS,iwork
#if KEY_PERT==1 /*pert0*/
  INTEGER IPSF
#endif /* (pert0)*/
  !
  CHARACTER(len=4) WRD,WRD2,OPER
  LOGICAL QSECOND,ERR

#if KEY_PERT==1 /*pertparse*/
  IPSF=GTRMI(COMLYN,COMLEN,'PSF',1)
  IF(IPSF == 0 .AND. PRNLEV >= 2) WRITE(OUTU,'(A)') &
       '  SCALAR> PERT reference psf will be used.'
#endif /* (pertparse)*/

  WRD=NEXTA4(COMLYN,COMLEN)
  OPER=NEXTA4(COMLYN,COMLEN)
  WRD2=' '
#if KEY_SGLD==1
  if( WRD == 'SGWT'  .and. .not. allocated(sgwt)) &
       call allocate_sgwt(natom)
  if( WRD == 'SGGA'  .and. .not. allocated(sggamma)) &
       call allocate_sggamma(natom)
#endif 
#if KEY_GCMC==1
  if((WRD == 'GCMC' .or. WRD == 'GCBL') .and. .not. allocated(gcmcon)) &
       call allocate_gcmc
#endif 
#if KEY_FOURD==1
   if((WRD == 'FDIM' .or. WRD == 'FDCO' .or. WRD == 'FDEQ') .and. &
      .not. allocated(fdim)) &
        call allocate_fourd_ltm
#endif 

   if (wrd == 'MOVE' .and. qshake) then
      call wrndie(-1, '<SCALAR>', &
           'shake already active and may affect fixed atoms')
   end if

  ! Handle the second keyname cases now (needed for back compatibility)
  IF(OPER == 'COPY' .OR. OPER == 'SUM' .OR. OPER == '=' .OR. &
       OPER == 'PROD' .OR. OPER == 'RECA') THEN
     WRD2=NEXTA4(COMLYN,COMLEN)
     QSECOND=.TRUE.
  ELSE IF(OPER == 'STOR'.OR.OPER == '+STO'.OR.OPER == '*STO') THEN
     WRD2=WRD
     WRD=NEXTA4(COMLYN,COMLEN)
     QSECOND=.TRUE.
  ELSE
     QSECOND=.FALSE.
  ENDIF
#if KEY_SGLD==1
  if( WRD2 == 'SGWT'  .and. .not. allocated(sgwt)) &
       call allocate_sgwt(natom)
  if( WRD2 == 'SGGAMMA'  .and. .not. allocated(sggamma)) &
       call allocate_sggamma(natom)
#endif 
#if KEY_GCMC==1
  if((WRD2 == 'GCMC' .or. WRD2 == 'GCBL') .and. .not. allocated(gcmcon)) &
       call allocate_gcmc
#endif 
#if KEY_FOURD==1
  if((WRD2 == 'FDIM' .or. WRD2 == 'FDCO' .or. WRD2 =='FDEQ') .and. &
     .not. allocated(fdim)) &
       call allocate_fourd_ltm
#endif 

  ! ECONT array could be used, better allocate if not done so yet. cb3
  if(.not.allocated(econt)) call allocate_econt
  !
  call chmalloc('scalar.src','SCALAR','IW',NATOM,crl=IW)
  call chmalloc('scalar.src','SCALAR','JW',NATOM,crl=JW)
  !
  ERR=.FALSE.  ! do not print and report some scalar selection errors
#if KEY_PERT==1 /*pert1*/
  IF(IPSF == 0)THEN
     CALL SCAFIL(WRD,IW,PERTDX,PERTDY,PERTDZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
          DCH,ECH,EHA,               & 
#endif
          XCOMP,YCOMP,ZCOMP,WCOMP,AMASS,PPCG,PPRSCLF,IMOVE,PPIAC,ITC, &
          PRREFX,PRREFY,PRREFZ,PRKCNST, &
          ALP,EFF,VDWR,ECONT,ICAAUTO, &
          NATOM, ERR &
          , Alph_Gb, SigX_Gb, SigY_Gb,SigZ_Gb, T_Gb, GB_Atm, QAnalys &   
#if KEY_ACE==1
          ,EFVOL,CHRAD,ACEHYD,BSOLV,CGIACP,CGSACP & 
#endif
#if KEY_FLUCQ==1
          ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC &  
#endif
#if KEY_GCMC==1
          ,qgcmc,GCMCON,GCBLKR                    & 
#endif
#if KEY_WCA==1
          ,PPWCA              & 
#endif
#if KEY_SGLD==1
          ,SGWT,SGGAMMA              & 
#endif
#if KEY_COMP2==1
          ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2    & 
#endif
          )
  ELSE
#endif /* (pert1)*/
     CALL SCAFIL(WRD,IW,DX,DY,DZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
          DCH,ECH,EHA,                    & 
#endif
          XCOMP,YCOMP,ZCOMP,WCOMP,AMASS,CG,RSCLF,IMOVE,IAC,ITC, &
          REFX,REFY,REFZ,KCNSTR, &
          ALP,EFF,VDWR,ECONT,ICAAUTO, &
          NATOM, ERR &
          , Alph_Gb, SigX_Gb, SigY_Gb,SigZ_Gb, T_Gb, GB_Atm, Qanalys & 
#if KEY_ACE==1
          ,EFVOL,CHRAD,ACEHYD,BSOLV,CGIACP,CGSACP &  
#endif
#if KEY_FLUCQ==1
          ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC & 
#endif
#if KEY_GCMC==1
          ,qgcmc,GCMCON,GCBLKR       & 
#endif
#if KEY_WCA==1
          ,WCA                 & 
#endif
#if KEY_SGLD==1
          ,SGWT,SGGAMMA              & 
#endif
#if KEY_COMP2==1
          ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2    & 
#endif
          )
#if KEY_PERT==1 /*pert2*/
  ENDIF
#endif /* (pert2)*/
  IF(ERR)  THEN
     call chmdealloc('scalar.src','SCALAR','IW',NATOM,crl=IW)
     call chmdealloc('scalar.src','SCALAR','JW',NATOM,crl=JW)
     RETURN
  ENDIF

  IF(QSECOND) THEN
     ERR=.TRUE.  ! print and report any errors
#if KEY_PERT==1 /*pert3*/
     IF(IPSF == 0)THEN
        CALL SCAFIL(WRD2,JW,PERTDX, &
             PERTDY,PERTDZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
             DCH,ECH,EHA,               & 
#endif
             XCOMP,YCOMP,ZCOMP,WCOMP,AMASS, &
             PPCG,PPRSCLF,IMOVE, &
             PPIAC,ITC, &
             PRREFX,PRREFY,PRREFZ,PRKCNST, &
             ALP,EFF,VDWR,ECONT,ICAAUTO, &
             NATOM, ERR &
             , Alph_Gb, SigX_Gb, SigY_Gb,SigZ_Gb, T_Gb, GB_atm, Qanalys &  
#if KEY_ACE==1
             ,EFVOL,CHRAD,ACEHYD,BSOLV,CGIACP,CGSACP &  
#endif
#if KEY_FLUCQ==1
             ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC &    
#endif
#if KEY_GCMC==1
             ,qgcmc,GCMCON,GCBLKR          & 
#endif
#if KEY_WCA==1
             ,PPWCA    & 
#endif
#if KEY_SGLD==1
             ,SGWT,SGGAMMA              & 
#endif
#if KEY_COMP2==1
             ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2    & 
#endif
             )
     ELSE
#endif /* (pert3)*/
        CALL SCAFIL(WRD2,JW,DX,DY,DZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
             DCH,ECH,EHA,                    & 
#endif
             XCOMP,YCOMP,ZCOMP,WCOMP,AMASS,CG,RSCLF,IMOVE,IAC,ITC, &
             REFX,REFY,REFZ,KCNSTR, &
             ALP,EFF,VDWR,ECONT,ICAAUTO, &
             NATOM, ERR &
             , Alph_Gb, SigX_Gb, SigY_Gb,SigZ_Gb, T_Gb, GB_atm, Qanalys &   
#if KEY_ACE==1
             ,EFVOL,CHRAD,ACEHYD,BSOLV,CGIACP,CGSACP &   
#endif
#if KEY_FLUCQ==1
             ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC & 
#endif
#if KEY_GCMC==1
             ,qgcmc,GCMCON,GCBLKR       & 
#endif
#if KEY_WCA==1
             ,WCA                 & 
#endif
#if KEY_SGLD==1
             ,SGWT,SGGAMMA              & 
#endif
#if KEY_COMP2==1
             ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2    & 
#endif
             )
#if KEY_PERT==1
     ENDIF         
#endif
     IF(ERR)  THEN
        call chmdealloc('scalar.src','SCALAR','IW',NATOM,crl=IW)
        call chmdealloc('scalar.src','SCALAR','JW',NATOM,crl=JW)
        RETURN
     ENDIF
  ENDIF
  call chmalloc('scalar.src','SCALAR','FLAGS',NATOM,intg=FLAGS)
  call chmalloc('scalar.src','SCALAR','IWORK',NATOM,intg=IWORK)
  CALL SCALA2(COMLYN,COMLEN,OPER,IW,JW,FLAGS,X,Y,Z,IWORK)
#if KEY_PERT==1 /*pert5*/
  IF(IPSF == 0)THEN
     CALL SCARET(WRD,IW,PERTDX, &
          PERTDY,PERTDZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
          DCH,ECH,EHA,               & 
#endif
          XCOMP,YCOMP,ZCOMP,WCOMP,AMASS,PPCG, &
          CGTOTP,PPRSCLF,IMOVE, &
          PPIAC, &
          PRREFX,PRREFY,PRREFZ,PRKCNST, &
          ECONT,ICAAUTO, &
          NATOM, ERR &
          , Alph_Gb, SigX_Gb, SigY_Gb,SigZ_Gb, T_Gb, GB_atm, Qanalys &   
#if KEY_FLUCQ==1
          ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC &   
#endif
#if KEY_GCMC==1
          ,QGCMC,GCMCON,GCBLKR       & 
#endif
#if KEY_WCA==1
          ,PPWCA       & 
#endif
#if KEY_SGLD==1
          ,SGWT,QSGGAMMA,SGGAMMA              & 
#endif
#if KEY_COMP2==1
          ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2  & 
#endif
          )
  ELSE
#endif /* (pert5)*/
     CALL SCARET(WRD,IW,DX,DY,DZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
          DCH,ECH,EHA,                    & 
#endif
          XCOMP,YCOMP,ZCOMP,WCOMP,AMASS,CG,CGTOT,RSCLF,IMOVE, &
          IAC, &
          REFX,REFY,REFZ,KCNSTR, &
          ECONT,ICAAUTO, &
          NATOM, ERR &
          , Alph_Gb, SigX_Gb, SigY_Gb,SigZ_Gb, T_Gb, GB_atm, Qanalys &   
#if KEY_FLUCQ==1
          ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC &  
#endif
#if KEY_GCMC==1
          ,QGCMC,GCMCON,GCBLKR       & 
#endif
#if KEY_WCA==1
          ,WCA                       & 
#endif
#if KEY_SGLD==1
          ,SGWT,QSGGAMMA,SGGAMMA              & 
#endif
#if KEY_COMP2==1
          ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2  & 
#endif
          )
#if KEY_PERT==1
  ENDIF       
#endif

  call chmdealloc('scalar.src','SCALAR','IW',NATOM,crl=IW)
  call chmdealloc('scalar.src','SCALAR','JW',NATOM,crl=JW)
  call chmdealloc('scalar.src','SCALAR','FLAGS',NATOM,intg=FLAGS)
  call chmdealloc('scalar.src','SCALAR','IWORK',NATOM,intg=IWORK)
  RETURN
END SUBROUTINE SCALAR


!=======================================================================
!              SCAFIL
!=======================================================================
SUBROUTINE SCAFIL(WRD,WORK,DX,DY,DZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
     DCH,ECH,EHA,              & 
#endif
     XCOMP,YCOMP,ZCOMP,WCOMP,AMASS,CG,RSCLF,IMOVE,IAC,ITC, &
     REFX,REFY,REFZ,KCNSTR, &
     ALP,EFF,VDWR,ECONT, ICAAUTO, NATOM, ERR &
     , Alph_Gb, SigX_Gb, SigY_Gb,SigZ_Gb, T_Gb, GBAtm, Qanalys &    
#if KEY_ACE==1
     ,EFVOL,CHRAD,ACEHYD,BSARR,CGIACE,CGSACE &    
#endif
#if KEY_FLUCQ==1
     ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC &   
#endif
#if KEY_GCMC==1
     ,qgcmc,GCMCON,GCBLKR       & 
#endif
#if KEY_WCA==1
     ,WCA                 & 
#endif
#if KEY_SGLD==1
     ,SGWT,SGGAMMA              & 
#endif
#if KEY_COMP2==1
     ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2    & 
#endif
     )
  !
  !-----------------------------------------------------------------------
  !     THIS ROUTINE FILLS THE WORK ARRAY WITH THE SELECTED ARRAY
  !
  !     BRB 5-NOV-1984
  !
  use chm_kinds
  use cnst_fcm, only: fbeta, allocate_cnst
  use stream
  use number
  use fourdm
  use storage
  use surface
  use pert
  use varcutm
#if KEY_MNDO97==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  use dimens_fcm
    use mndo97
#if KEY_MNDO97==1
    use qm1_info, only : qm_control_r
#endif
#endif 
  !
  implicit none

#if KEY_CHEQ==1
  real(chm_real) DCH(*),ECH(*),EHA(*)   
#endif
  CHARACTER(len=4) WRD
  real(chm_real) DX(*),DY(*),DZ(*),WORK(:)
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),WCOMP(*)
#if KEY_COMP2==1
  real(chm_real) XCOMP2(*),YCOMP2(*),ZCOMP2(*),WCOMP2(*)
#endif 
  INTEGER IMOVE(*),IAC(*),ITC(*),ICAAUTO(*)
  real(chm_real) AMASS(*),CG(*),RSCLF(*)
#if KEY_WCA==1
  real(chm_real) WCA(*)                         
#endif
#if KEY_SGLD==1
  real(chm_real) SGWT(*),SGGAMMA(*)                         
#endif
  real(chm_real), allocatable :: REFX(:),REFY(:),REFZ(:),KCNSTR(:)
  real(chm_real) ALP(*),EFF(*),VDWR(*),ECONT(*)
  INTEGER NATOM
  LOGICAL ERR
  real(chm_real) Alph_GB(*), SigX_GB(*), SigY_GB(*), SigZ_GB(*), &
       T_GB(*), GBAtm(*)
  Logical QAnalys
#if KEY_ACE==1
  real(chm_real) EFVOL(*),CHRAD(*),ACEHYD(*),BSARR(*)
  real(chm_real) CGIACE(*),CGSACE(*)
#endif 
#if KEY_FLUCQ==1
  real(chm_real) FQCHI(*),FQZETA(*),FQJZ(*), &
       FQCFOR(*),FQOLDQ(*),FQCHMA(*)
  INTEGER FQPRIN(*)
  LOGICAL QFLUC
#endif 
#if KEY_GCMC==1
  LOGICAL qgcmc, GCMCON(:),GCBLKR(:)     
#endif
  INTEGER I,IST
  integer lev
  LOGICAL QPERR
  CHARACTER(len=1) CIST
  !
  QPERR=ERR  ! should selection array errors be reported?
  !                ! on "fill" mode, these are ignored.
  ERR=.FALSE.
  IF (WRD == 'WMAI') THEN
     DO I=1,NATOM
        WORK(I)=WMAIN(I)
     ENDDO
  ELSE IF (WRD == 'WCOM') THEN
     DO I=1,NATOM
        WORK(I)=WCOMP(I)
     ENDDO
#if KEY_COMP2==1
  ELSE IF (WRD == 'WCM2') THEN
     DO I=1,NATOM
        WORK(I)=WCOMP2(I)
     ENDDO
#endif 
  ELSE IF (WRD == 'CONS') THEN
     call copy_dyn(work, kcnstr, 'KCNSTR')
  ELSE IF (WRD == 'MASS') THEN
     DO I=1,NATOM
        WORK(I)=AMASS(I)
     ENDDO
  ELSE IF (WRD == 'CHAR') THEN
     DO I=1,NATOM
        WORK(I)=CG(I)
     ENDDO
  ELSE IF (WRD == 'VARC') THEN
     call get_varcut(WORK, NATOM)
  ELSE IF (WRD == 'CQMM') THEN
#if KEY_GAMESSUK==1 || KEY_GAMESS==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
     DO I=1,NATOM
        WORK(I)=CGQMMM(I)
     ENDDO
#elif KEY_MNDO97==1
     DO I=1,NATOM
        WORK(I)=qm_control_r%CGQMMM(I)
     ENDDO
#endif
  else if (wrd == 'FBET') then
     call allocate_cnst(NATOM)  ! special case for Langevin dynamics
     call copy_dyn(work, fbeta, 'FBETA')
  ELSE IF (WRD == 'X   ') THEN
     DO I=1,NATOM
        WORK(I)=X(I)
     ENDDO
  ELSE IF (WRD == 'Y   ') THEN
     DO I=1,NATOM
        WORK(I)=Y(I)
     ENDDO
  ELSE IF (WRD == 'Z   ') THEN
     DO I=1,NATOM
        WORK(I)=Z(I)
     ENDDO
  ELSE IF (WRD == 'XCOM') THEN
     DO I=1,NATOM
        WORK(I)=XCOMP(I)
     ENDDO
  ELSE IF (WRD == 'YCOM') THEN
     DO I=1,NATOM
        WORK(I)=YCOMP(I)
     ENDDO
  ELSE IF (WRD == 'ZCOM') THEN
     DO I=1,NATOM
        WORK(I)=ZCOMP(I)
     ENDDO
#if KEY_COMP2==1
  ELSE IF (WRD == 'XCM2') THEN
     DO I=1,NATOM
        WORK(I)=XCOMP2(I)
     ENDDO
  ELSE IF (WRD == 'YCM2') THEN
     DO I=1,NATOM
        WORK(I)=YCOMP2(I)
     ENDDO
  ELSE IF (WRD == 'ZCM2') THEN
     DO I=1,NATOM
        WORK(I)=ZCOMP2(I)
     ENDDO
#endif 
     ! SAPATEL
#if KEY_CHEQ==1
  ELSE IF (WRD == 'DCH ') THEN
     DO I=1,NATOM
        WORK(I)=DCH(I)
     ENDDO
  ELSE IF (WRD == 'ECH ') THEN
     DO I=1,NATOM
        WORK(I)=ECH(I)
     ENDDO
  ELSE IF (WRD == 'EHA ') THEN
     DO I=1,NATOM
        WORK(I)=EHA(I)
     ENDDO
#endif 
     ! SAPATEL
  ELSE IF (WRD == 'DX  ') THEN
     DO I=1,NATOM
        WORK(I)=DX(I)
     ENDDO
  ELSE IF (WRD == 'DY  ') THEN
     DO I=1,NATOM
        WORK(I)=DY(I)
     ENDDO
  ELSE IF (WRD == 'DZ  ') THEN
     DO I=1,NATOM
        WORK(I)=DZ(I)
     ENDDO
  ELSE IF (WRD == 'XREF') THEN
     call copy_dyn(work, refx, 'REFX', ANUM)
  ELSE IF (WRD == 'YREF') THEN
     call copy_dyn(work, refy, 'REFY', ANUM)
  ELSE IF (WRD == 'ZREF') THEN
     call copy_dyn(work, refz, 'REFZ', ANUM)
  ELSE IF (WRD == 'ECON') THEN
     DO I=1,NATOM
        WORK(I)=ECONT(I)
     ENDDO
  ELSE IF (WRD == 'EPCO') THEN
#if KEY_PERT==1
     IF(QPERT) THEN
        DO I=1,NATOM
           WORK(I) = PERTEPC(I)
        ENDDO
     ELSE
#endif 
        CALL WRNDIE(-1,'<SCAFIL>','PERT code not active')
        ERR=.TRUE.  ! always report this error
        WORK(1:NATOM)=zero
#if KEY_PERT==1
     ENDIF  
#endif
  ELSE IF (WRD == 'MOVE') THEN
     DO I=1,NATOM
        WORK(I)=IMOVE(I)
     ENDDO
  ELSE IF (WRD == 'TYPE') THEN
     DO I=1,NATOM
        WORK(I)=IAC(I)
     ENDDO
  ELSE IF (WRD == 'AUTO') THEN
     DO I=1,NATOM
        WORK(I)=ICAAUTO(I)
     ENDDO
  ELSE IF (WRD == 'RSCA') THEN
     DO I=1,NATOM
        !         note: the RSCLF array is stored in squared form.
        WORK(I)=SQRT(ABS(RSCLF(I)))
     ENDDO
#if KEY_WCA==1
     ! Weeks, Chandler, Anderson decomposition of Lennard-Jones Potential
  ELSE IF (WRD == 'WCAD') THEN
     DO I=1,NATOM
        WORK(I)=WCA(I)
     ENDDO
#endif 
  ELSE IF (WRD == 'ALPH') THEN
     DO I=1,NATOM
        WORK(I)=ALP(ITC(IAC(I)))
     ENDDO
  ELSE IF (WRD == 'EFFE') THEN
     DO I=1,NATOM
        WORK(I)=EFF(ITC(IAC(I)))
     ENDDO
  ELSE IF (WRD == 'RADI') THEN
     DO I=1,NATOM
        WORK(I)=VDWR(ITC(IAC(I)))
     ENDDO
#if KEY_ASPENER==1
  ELSE IF (WRD == 'IGNO') THEN
     call copy_dyn_i(work, ignore, 'IGNORE')
  ELSE IF (WRD == 'ASPV') THEN
     call copy_dyn(work, aspv, 'ASPV')
  ELSE IF (WRD == 'VDWS') THEN
     call copy_dyn(work, vdw_surf, 'VDW_SURF')
#else /**/
  ELSE IF (WRD == 'IGNO' .OR. WRD == 'ASPV' .OR. WRD == 'VDWS') THEN
     CALL WRNDIE(-1, '<SCAFIL>', 'ASP code not compiled')
     ERR = .TRUE.
     WORK(1:NATOM) = ZERO
#endif 
#if KEY_FOURD==1
  ELSE IF (WRD == 'FDIM') THEN
     DO I=1,NATOM
        WORK(I)=FDIM(I)
     ENDDO
  ELSE IF (WRD == 'FDCO') THEN
     DO I=1,NATOM
        WORK(I)=FDCOMP(I)
     ENDDO
  ELSE IF (WRD == 'FDEQ') THEN
     DO I=1,NATOM
        WORK(I)=FDEQ(I)
     ENDDO
#else /**/
  ELSE IF(WRD == 'FDIM' .OR. WRD == 'FDCO' .OR. WRD == 'FDEQ') THEN
     CALL WRNDIE(-1,'<SCAFIL>','4D code not compiled')
     ERR=.TRUE.   ! always report this error
     WORK(1:NATOM)=zero
#endif 
  Else if ( Wrd  ==  'GBAL' ) then
     Do i=1, natom
        Work(i) = Alph_GB(i)
     Enddo
  Else if ( Wrd  ==  'SIGX' ) then
     Do i=1, natom
        Work(i) = SigX_GB(i)
     Enddo
  Else if ( Wrd  ==  'SIGY' ) then
     Do i=1, natom
        Work(i) = SigY_GB(i)
     Enddo
  Else if ( Wrd  ==  'SIGZ' ) then
     Do i=1, natom
        Work(i) = SigZ_GB(i)
     Enddo
  Else if ( Wrd  ==  'T_GB' ) then
     Do i=1, natom
        Work(i) = T_GB(i)
     Enddo
  Else if ( Wrd  ==  'GBAT' ) then
     If ( Qanalys ) Then
        Do i=1, natom
           Work(i) = GBAtm(i)
        Enddo
     Else
        Call Wrndie(-1,'<SCAFIL>', &
             'Analysis key needed on GBORN command line' )
        ERR = .True.
     Endif
#if KEY_ACE==1
  ELSE IF (WRD == 'VOLU') THEN
     DO I=1,NATOM
        WORK(I)=EFVOL(IAC(I))
     ENDDO
  ELSE IF (WRD == 'QRAD') THEN
     DO I=1,NATOM
        WORK(I)=CHRAD(IAC(I))
     ENDDO
  ELSE IF (WRD == 'SIGM') THEN
     DO I=1,NATOM
        WORK(I)=ACEHYD(IAC(I))
     ENDDO
  ELSE IF (WRD == 'BSOL') THEN
     DO I=1,NATOM
        WORK(I)=BSARR(I)
     ENDDO
  ELSE IF (WRD == 'QIAC') THEN
     DO I=1,NATOM
        WORK(I)=CGIACE(I)
     ENDDO
  ELSE IF (WRD == 'QSAC') THEN
     DO I=1,NATOM
        WORK(I)=CGSACE(I)
     ENDDO
#else /*  ACE*/
  ELSE IF(WRD == 'VOLU'.OR.WRD == 'QRAD'.OR.WRD == 'SIGM'.OR. &
       WRD == 'BSOL'.OR.WRD == 'QIAC'.OR.WRD == 'QSAC') THEN
     CALL WRNDIE(-1,'<SCAFIL>','ACE code not compiled')
     ERR=.TRUE.
     WORK(1:NATOM)=zero
#endif 
#if KEY_FLUCQ==1
  ELSE IF (WRD == 'FQCH') THEN
     DO I=1,NATOM
        WORK(I)=FQCHI(IAC(I))
     ENDDO
  ELSE IF (WRD == 'FQMA') THEN
     DO I=1,NATOM
        WORK(I)=FQCHMA(IAC(I))
     ENDDO
  ELSE IF (WRD == 'FQPR') THEN
     DO I=1,NATOM
        WORK(I)=FQPRIN(IAC(I))
     ENDDO
  ELSE IF (WRD == 'FQZE') THEN
     DO I=1,NATOM
        WORK(I)=FQZETA(IAC(I))
     ENDDO
  ELSE IF (WRD == 'FQJZ') THEN
     DO I=1,NATOM
        WORK(I)=FQJZ(IAC(I))
     ENDDO
  ELSE IF (WRD == 'FQCF') THEN
     IF (QFLUC) THEN
        DO I=1,NATOM
           WORK(I)=FQCFOR(I)
        ENDDO
     ELSE
        CALL WRNDIE(-1,'<SCALAR>','FlucQ code not activated!')
        ERR=.TRUE.
     ENDIF
  ELSE IF (WRD == 'FQOL') THEN
     IF (QFLUC) THEN
        DO I=1,NATOM
           WORK(I)=FQOLDQ(I)
        ENDDO
     ELSE
        CALL WRNDIE(-1,'<SCALAR>','FlucQ code not activated!')
        ERR=.TRUE.
     ENDIF
#endif 
#if KEY_GCMC==1
  ELSE IF (WRD == 'GCMC') THEN
     DO I=1,NATOM
        IF (GCMCON(I)) THEN
           WORK(I)=ONE
        ELSE
           WORK(I)=ZERO
        ENDIF
     ENDDO
  ELSE IF (WRD == 'GCBL') THEN
     DO I=1,NATOM
        IF (GCBLKR(I)) THEN
           WORK(I)=ONE
        ELSE
           WORK(I)=ZERO
        ENDIF
     ENDDO
#else /**/
  ELSE IF(WRD == 'GCMC' .OR. WRD == 'GCBL') THEN
     CALL WRNDIE(-1,'<SCAFIL>','GCMC code not compiled')
     ERR=.TRUE.
     WORK(1:NATOM)=zero
#endif 
#if KEY_SGLD==1
  ELSE IF (WRD == 'SGWT') THEN
     DO I=1,NATOM
        WORK(I)=SGWT(I)
     ENDDO
  ELSE IF (WRD == 'SGGA') THEN
     DO I=1,NATOM
        WORK(I)=SGGAMMA(I)
     ENDDO
#endif 
     !
     ! Special arrays for atom selection and other usage
     !
  ELSE IF(WRD(1:3) == 'SCA' .OR. WRD(2:4) == '   ') THEN
     IF(WRD(1:3) == 'SCA') CIST=WRD(4:4)
     IF(WRD(2:4) == '   ') CIST=WRD(1:1)
     IST=-1
     READ(CIST,'(I1)',ERR=250) IST
250  CONTINUE
     IF(IST < 1 .OR. IST > MAXSTO) THEN
        CALL WRNDIE(-1,'<SCALAR>','Store number out of range')
        ERR=.TRUE.  ! always report this error
        WORK(1:NATOM) = FMARK
     ELSE IF (.not. allocated (ptrsto(ist)%a)) then
        IF(QPERR) THEN ! ignore this error in fill mode
           CALL WRNDIE(-1,'<SCALAR>','Store empty')
           ERR=.TRUE.
        ENDIF
        WORK(1:NATOM) = FMARK
     ELSE IF (ptrSTO(IST)%len /= NATOM) THEN
        IF(QPERR) THEN ! ignore this error in fill mode
           CALL WRNDIE(-1,'<SCALAR>','Dim. mismatch')
           ERR=.TRUE.
        ENDIF
        WORK(1:NATOM) = FMARK
     ELSE
        work(1:natom) = ptrsto(ist)%a(1:natom)
     ENDIF
  ELSE IF (WRD == 'ZERO') THEN
     WORK(1:NATOM)=zero
  ELSE IF (WRD == 'ONE ') THEN
     WORK(1:NATOM) = ONE
     !
     ! End of special arrays
     !
  ELSE
     CALL WRNDIE(-1,'<SCAFIL>','UNRECOGNIZED SCALAR KEYNAME')
     ERR=.TRUE.
  ENDIF
  !
  RETURN
contains

  subroutine copy_dyn(dest, src, sname, fb_arg)
    real(chm_real), intent(out) :: dest(:)
    real(chm_real), allocatable, intent(in) :: src(:)
    character(len=*), intent(in) :: sname
    real(chm_real), intent(in), optional :: fb_arg
    real(chm_real) :: fallback

    fallback = ZERO
    if (present(fb_arg)) fallback = fb_arg

    if (allocated(src)) then
       if (size(src) == NATOM) then
          dest(1:NATOM) = src(1:NATOM)
       else
          dest(1:NATOM) = fallback
          if (WRNLEV >= 2) write (OUTU, '(2A)') &
               sname, " size changed, operation not performed"
       endif
    else
       dest(1:NATOM) = fallback
       if (WRNLEV >= 2) write (OUTU, '(2A)') &
            sname, " not yet filled, operation not performed"
    endif
  end subroutine copy_dyn

  subroutine copy_dyn_i(dest, src, sname)
    real(chm_real), intent(out) :: dest(:)
    integer, allocatable, intent(in) :: src(:)
    character(len=*), intent(in) :: sname

    if (allocated(src)) then
       if (size(src) == NATOM) then
          dest(1:NATOM) = src(1:NATOM)
       else
          dest(1:NATOM) = ZERO
          if (WRNLEV >= 2) write (OUTU, '(2A)') &
               sname, " size changed, operation not performed"
       endif
    else
       dest(1:NATOM) = ZERO
       if (WRNLEV >= 2) write (OUTU, '(2A)') &
            sname, " not yet filled, operation not performed"
    endif
  end subroutine copy_dyn_i

END SUBROUTINE SCAFIL

!-----------------------------------------------------------------
!          SCARET
!-----------------------------------------------------------------
SUBROUTINE SCARET(WRD,WORK,DX,DY,DZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
     DCH,ECH,EHA,         & 
#endif
     XCOMP,YCOMP,ZCOMP,WCOMP,AMASS,CG,CGTOT,RSCLF,IMOVE, &
     IAC, &
     REFX,REFY,REFZ,KCNSTR, &
     ECONT, ICAAUTO, NATOM, ERR &
     , Alph_Gb, SigX_Gb, SigY_Gb,SigZ_Gb, T_Gb, GBAtm, Qanalys &  
#if KEY_FLUCQ==1
     ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC & 
#endif
#if KEY_GCMC==1
     ,QGCMC,GCMCON,GCBLKR       & 
#endif
#if KEY_WCA==1
     ,WCA                       & 
#endif
#if KEY_SGLD==1
     ,SGWT,QSGGAMMA,SGGAMMA                       & 
#endif
#if KEY_COMP2==1
     ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2  & 
#endif
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE RETURNS THE SELECTED ARRAY WITH THE WORK ARRAY
  !
  !     BRB 5-NOV-1984
  !
  use chm_kinds
  use number
  use dimens_fcm
  use stream
  use code
  use fourdm
  use storage
  use surface
  use pert
  use varcutm
#if KEY_MNDO97==1
  use mndo97
  use qm1_info, only : qm_control_r
#endif 
  use cnst_fcm, only: fbeta, allocate_cnst
  implicit none
  CHARACTER(len=4) WRD
  real(chm_real) DX(*),DY(*),DZ(*),WORK(:)
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  ! SAPATEL
#if KEY_CHEQ==1
  real(chm_real) DCH(*),ECH(*),EHA(*)
#endif 
  ! SAPATEL
  real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),WCOMP(*)
#if KEY_COMP2==1
  real(chm_real) XCOMP2(*),YCOMP2(*),ZCOMP2(*),WCOMP2(*)
#endif 
  INTEGER IMOVE(*),IAC(*),ICAAUTO(*)
  real(chm_real) AMASS(*),CG(*),CGTOT,RSCLF(*)
#if KEY_WCA==1
  real(chm_real) WCA(*)                   
#endif
#if KEY_SGLD==1
  Logical QSGGAMMA
  real(chm_real) SGWT(*),SGGAMMA(*)                   
#endif
  real(chm_real), allocatable :: REFX(:),REFY(:),REFZ(:),KCNSTR(:)
  real(chm_real) ECONT(*)
  INTEGER NATOM
  LOGICAL ERR
  real(chm_real) Alph_GB(*), SigX_GB(*), SigY_GB(*), SigZ_GB(*), &
       T_GB(*), GBAtm(*)
  Logical QAnalys
#if KEY_FLUCQ==1
  real(chm_real) FQCHI(*),FQZETA(*),FQJZ(*), &
       FQCFOR(*),FQOLDQ(*),FQCHMA(*)
  INTEGER FQPRIN(*)
  LOGICAL QFLUC
#endif 
#if KEY_GCMC==1
  LOGICAL QGCMC,GCMCON(:),GCBLKR(:)        
#endif
  !
  INTEGER I,IST
  real(chm_real)  QTOT
  CHARACTER(len=1) CIST
  !
  ERR=.FALSE.
  IF (WRD == 'WMAI') THEN
     DO I=1,NATOM
        WMAIN(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'WCOM') THEN
     DO I=1,NATOM
        WCOMP(I)=WORK(I)
     ENDDO
#if KEY_COMP2==1
  ELSE IF (WRD == 'WCM2') THEN
     DO I=1,NATOM
        WCOMP2(I)=WORK(I)
     ENDDO
#endif 
  ELSE IF (WRD == 'CONS') THEN
     if(size(kcnstr) /= natom) call allocate_cnst(natom)
     kcnstr(1:natom)=work(1:natom)
  ELSE IF (WRD == 'MASS') THEN
     DO I=1,NATOM
        AMASS(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'VARC') THEN
     call set_varcut(WORK, NATOM)
  ELSE IF (WRD == 'CHAR') THEN
     QTOT=0.0
     DO I=1,NATOM
        CG(I)=WORK(I)
        QTOT=QTOT+CG(I)
     ENDDO
     IF(ABS(ANINT(QTOT)-QTOT) > PT0001) THEN
        ! SAPATEL
#if KEY_CHEQ==1
        if (prnlev >= 2) WRITE(OUTU,433) QTOT
433     FORMAT(/,' Warning from SCALAR: The sum of charges (', &
             F12.6,') is not an integer',/)
#else /**/
        IF(WRNLEV >= 2 .and. prnlev >= 2) WRITE(OUTU,433) QTOT
433     FORMAT(/,' Warning from SCALAR: The sum of charges (', &
             F12.6,') is not an integer',/)
        CALL WRNDIE(0,'<SCARET>','Total charge not an integer')
#endif 
        ! SAPATEL
     ENDIF

     CGTOT=QTOT
  ELSE IF (WRD == 'CQMM') THEN
#if KEY_MNDO97==1
     QTOT=0.0
     DO I=1,NATOM
        qm_control_r%CGQMMM(I)=WORK(I)
        QTOT=QTOT+qm_control_r%CGQMMM(I)
        if (prnlev >= 2) WRITE(OUTU,435) I, CG(I), qm_control_r%CGQMMM(I)
     ENDDO
435  FORMAT(/,'<SCARET> Charge for atom # (',i8,') changed from (', &
          F12.6,') to (',F12.6,') ',/)
     IF(ABS(ANINT(QTOT)-QTOT) > PT0001) THEN
        IF(WRNLEV >= 2 .and. prnlev >= 2) WRITE(OUTU,434) QTOT
434     FORMAT(/,' Warning from SCALAR: The sum of charges (', &
             F12.6,') is not an integer',/)
        CALL WRNDIE(0,'<SCARET>', &
             'Total MM charge for QMMM not an integer')
     ENDIF
#endif 
  else if (wrd == 'FBET') then
     if(size(fbeta) /= natom) call allocate_cnst(natom)
     fbeta(1:natom)=work(1:natom)
  ELSE IF (WRD == 'X   ') THEN
     DO I=1,NATOM
        X(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'Y   ') THEN
     DO I=1,NATOM
        Y(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'Z   ') THEN
     DO I=1,NATOM
        Z(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'XCOM') THEN
     DO I=1,NATOM
        XCOMP(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'YCOM') THEN
     DO I=1,NATOM
        YCOMP(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'ZCOM') THEN
     DO I=1,NATOM
        ZCOMP(I)=WORK(I)
     ENDDO
#if KEY_COMP2==1
  ELSE IF (WRD == 'XCM2') THEN
     DO I=1,NATOM
        XCOMP2(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'YCM2') THEN
     DO I=1,NATOM
        YCOMP2(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'ZCM2') THEN
     DO I=1,NATOM
        ZCOMP2(I)=WORK(I)
     ENDDO
#endif 
     ! SAPATEL
#if KEY_CHEQ==1
  ELSE IF (WRD == 'DCH ') THEN
     DO I=1,NATOM
        DCH(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'ECH ') THEN
     DO I=1,NATOM
        ECH(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'EHA ') THEN
     DO I=1,NATOM
        EHA(I)=WORK(I)
     ENDDO
#endif 
     ! SAPATEL
  ELSE IF (WRD == 'DX  ') THEN
     DO I=1,NATOM
        DX(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'DY  ') THEN
     DO I=1,NATOM
        DY(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'DZ  ') THEN
     DO I=1,NATOM
        DZ(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'XREF') THEN
     if (size(refx) /= natom) call allocate_cnst(natom)
     refx(1:natom) = work(1:natom)
  ELSE IF (WRD == 'YREF') THEN
     if (size(refy) /= natom) call allocate_cnst(natom)
     refy(1:natom) = work(1:natom)
  ELSE IF (WRD == 'ZREF') THEN
     if (size(refz) /= natom) call allocate_cnst(natom)
     refz(1:natom) = work(1:natom)
  ELSE IF (WRD == 'ECON') THEN
     DO I=1,NATOM
        ECONT(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'EPCO') THEN
#if KEY_PERT==1
     IF(QPERT) THEN
        DO I=1,NATOM
           PERTEPC(I) = WORK(I)
        ENDDO
     ELSE
#endif 
        ERR=.TRUE.
#if KEY_PERT==1
     ENDIF  
#endif
  ELSE IF (WRD == 'MOVE') THEN
     DO I=1,NATOM
        IMOVE(I)=NINT(WORK(I))
     ENDDO
  ELSE IF (WRD == 'TYPE') THEN
     DO I=1,NATOM
        IAC(I)=NINT(WORK(I))
     ENDDO
     MUSTUP=.TRUE.
  ELSE IF (WRD == 'AUTO') THEN
     DO I=1,NATOM
        ICAAUTO(I)=NINT(WORK(I))
     ENDDO
     MUSTUP=.TRUE.
  ELSE IF (WRD == 'RSCA') THEN
     DO I=1,NATOM
        !         note: the RSCLF array is stored in squared form.
        RSCLF(I)=WORK(I)**2
     ENDDO
#if KEY_WCA==1
     ! Weeks, Chandler, Anderson decomposition of Lennard-Jones Potential
  ELSE IF (WRD == 'WCAD') THEN
     DO I=1,NATOM
        WCA(I)=WORK(I)
     ENDDO
#endif 
  ELSE IF (WRD == 'ALPH') THEN
     RETURN
  ELSE IF (WRD == 'EFFE') THEN
     RETURN
  ELSE IF (WRD == 'RADI') THEN
     RETURN
#if KEY_ASPENER==1
  ELSE IF (WRD == 'IGNO') THEN
     if(.not.allocated(ignore)) then
        call allocate_surface(natom)
     else if(size(ignore) /= natom) then
        call allocate_surface(natom)
     endif
     DO I=1,NATOM
        IGNORE(I)=NINT(WORK(I))
     ENDDO
  ELSE IF (WRD == 'ASPV') THEN
     if(.not.allocated(aspv)) then
        call allocate_surface(natom)
     else if(size(aspv) /= natom) then
        call allocate_surface(natom)
     endif
     DO I=1,NATOM
        ASPV(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'VDWS') THEN
     if(.not.allocated(vdw_surf)) then
        call allocate_surface(natom)
     else if(size(vdw_surf) /= natom) then
        call allocate_surface(natom)
     endif
     DO I=1,NATOM
        VDW_SURF(I)=WORK(I)
     ENDDO
#endif 
#if KEY_FOURD==1
  ELSE IF (WRD == 'FDIM') THEN
     DO I=1,NATOM
        FDIM(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'FDCO') THEN
     DO I=1,NATOM
        FDCOMP(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'FDEQ') THEN
     DO I=1,NATOM
        FDEQ(I)=WORK(I)
     ENDDO
#endif 
     !
  Else if ( Wrd  ==  'GBAL' ) then
     Do i=1, natom
        Alph_GB(i) = Work(i)
     Enddo
  Else if ( Wrd  ==  'SIGX' ) then
     Do i=1, natom
        SigX_GB(i) = Work(i)
     Enddo
  Else if ( Wrd  ==  'SIGY' ) then
     Do i=1, natom
        SigY_GB(i) = Work(i)
     Enddo
  Else if ( Wrd  ==  'SIGZ' ) then
     Do i=1, natom
        SigZ_GB(i) = Work(i)
     Enddo
  Else if ( Wrd  ==  'T_GB' ) then
     Do i=1, natom
        T_GB(i) = Work(i)
     Enddo
  Else if ( Wrd  ==  'GBAT' ) then
     If ( Qanalys ) Then
        Do i=1, natom
           Work(i) = GBAtm(i)
        Enddo
     Else
        Call Wrndie(-1,'<SCAFIL>', &
             'Analysis key needed on GBORN command line' )
        ERR = .True.
     Endif
#if KEY_FLUCQ==1
  ELSE IF (WRD == 'FQCH') THEN
     RETURN
  ELSE IF (WRD == 'FQMA') THEN
     RETURN
  ELSE IF (WRD == 'FQPR') THEN
     RETURN
  ELSE IF (WRD == 'FQZE') THEN
     RETURN
  ELSE IF (WRD == 'FQJZ') THEN
     RETURN
  ELSE IF (WRD == 'FQCF') THEN
     IF (QFLUC) THEN
        DO I=1,NATOM
           FQCFOR(I)=WORK(I)
        ENDDO
     ELSE
        CALL WRNDIE(-1,'<SCAFIL>','FlucQ code not activated!')
        ERR=.TRUE.
     ENDIF
  ELSE IF (WRD == 'FQOL') THEN
     IF (QFLUC) THEN
        DO I=1,NATOM
           FQOLDQ(I)=WORK(I)
        ENDDO
     ELSE
        CALL WRNDIE(-1,'<SCAFIL>','FlucQ code not activated!')
        ERR=.TRUE.
     ENDIF
#endif 
#if KEY_GCMC==1
  ELSE IF (WRD == 'GCMC') THEN
     QGCMC = .FALSE.
     DO I=1,NATOM
        GCMCON(I) = (WORK(I)  >  RSMALL)
        IF (.NOT. GCMCON(I)) QGCMC = .TRUE.
     ENDDO
  ELSE IF (WRD == 'GCBL') THEN
     DO I=1,NATOM
        GCBLKR(I) = (WORK(I)  >  RSMALL)
     ENDDO
#else /**/
  ELSE IF(WRD == 'GCMC' .OR. WRD == 'GCBL') THEN
     CALL WRNDIE(-1,'<SCAFIL>','GCMC code not compiled')
     ERR=.TRUE.
     WORK(1:NATOM)=zero
#endif 
#if KEY_SGLD==1
     ! SGLD guiding factors
  ELSE IF (WRD == 'SGWT') THEN
     DO I=1,NATOM
        SGWT(I)=WORK(I)
     ENDDO
  ELSE IF (WRD == 'SGGA') THEN
     QSGGAMMA=.FALSE.
     DO I=1,NATOM
        SGGAMMA(I)=WORK(I)
     ENDDO
#endif 
     !
     ! Special arrays for atom selection and other usage
     !
  ELSE IF(WRD(1:3) == 'SCA' .OR. WRD(2:4) == '   ') THEN
     IF(WRD(1:3) == 'SCA') CIST=WRD(4:4)
     IF(WRD(2:4) == '   ') CIST=WRD(1:1)
     IST=-1
     READ(CIST,'(I1)',ERR=250) IST
250  CONTINUE
     IF(IST < 1 .OR. IST > MAXSTO) THEN
        CALL WRNDIE(1,'<SCALAR>','Store number out of range')
        ERR=.TRUE.
     ELSE
        IF(ptrsto(IST)%len /= NATOM) THEN
           call resize_array('scalar.src','scaret/resize_array','ptrsto(ist)%a', &
                ptrsto(ist),natom)
        ENDIF
        ptrsto(ist)%a(1:natom) = work(1:natom)
     ENDIF
  ELSE IF (WRD == 'ZERO') THEN
     RETURN
  ELSE IF (WRD == 'ONE ') THEN
     RETURN
     !
     ! End of special arrays
     !
  ELSE
     CALL WRNDIE(-1,'<SCARET>','Unrecognized scalar keyname')
     ERR=.TRUE.
     RETURN
  ENDIF
  !
  IF(ERR) THEN
     CALL WRNDIE(-1,'<SCARET>', &
          'Destination array unavailable due to compile options')
     RETURN
  ENDIF
  !
  DO I=1,NATOM
     IF(WORK(I) == FMARK) THEN
        CALL WRNDIE(0,'<SCARET>', &
             'Some returned array elements are undefined')
        RETURN
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE SCARET

SUBROUTINE SCALA2(COMLYN,COMLEN,WRD,ARRAY,ARRAY2, &
     ISLCT,X,Y,Z,IWORK)
  !-----------------------------------------------------------------------
  !     By Axel Brunger, 17-MAR-83,
  !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc, only: orderr
  use select
  use storage
  use stream
  use string
  use psf
  use hbondm
  use gcmc
  use clcg_mod, only: ranumb
#if KEY_CHEQ==1
  use corman3, only: histogram, histogram2     
#endif
  use chutil
  use param_store, only: set_param
  use parallel,only: psnd8

  implicit none

  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  CHARACTER(len=4) WRD
  real(chm_real) ARRAY(:),ARRAY2(:)
  INTEGER ISLCT(:)
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER IWORK(:)
  !
  real(chm_real) RNUMBR, RNUMBR2, RNORM, TOL
  real(chm_real) ARMIN,ARMAX,ARAVE,ARWGHT,ARTOT,ARVAR
  real(chm_real), allocatable,dimension(:) :: TMPARR
  !
  INTEGER   I,J,ISEG,IRES,IST,NSL,INUMBR,ICODE,IS,IQ,LLEN
  LOGICAL   LTOTAL
  CHARACTER(len=40) LINE
  !
  TOL=TENM14
  !
  CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,ARRAY,.TRUE.)
  !
  !-----------------------------------------------------------------------
  IF (WRD == 'COPY' .OR. WRD == '=' .OR. WRD == 'RECA' .OR. &
       WRD == 'STOR') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=ARRAY2(I)
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'SUM' .OR. WRD == '+STO') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=ARRAY(I)+ARRAY2(I)
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'PROD' .OR. WRD == '*STO') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=ARRAY(I)*ARRAY2(I)
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'READ') THEN
     INUMBR=NEXTI(COMLYN,COMLEN)
     IF(IOLEV > 0) THEN
        DO I=1,NATOM
           IF (ISLCT(I) == 1) THEN
              READ(INUMBR,'(A)') LINE
              LLEN=40
              ARRAY(I)=NEXTF(LINE,LLEN)
           ENDIF
        ENDDO
     ENDIF
#if KEY_PARALLEL==1
     CALL PSND8(ARRAY,NATOM)
#endif 
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'WRIT') THEN
     INUMBR=NEXTI(COMLYN,COMLEN)
     IF(IOLEV > 0) THEN
        DO I=1,NATOM
           IF(ISLCT(I) == 1) THEN
              IF(FMTLEN > 0) THEN
                 WRITE(INUMBR,FMTWD(1:FMTLEN)) ARRAY(I)
              ELSE
                 WRITE(INUMBR,'(1PG14.6)') ARRAY(I)
              ENDIF
           ENDIF
        ENDDO
     ENDIF
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'HWRI') THEN
     INUMBR=NEXTI(COMLYN,COMLEN)
     IF(IOLEV > 0 .AND. (NSELCT(NATOM,ISLCT) > 0)) THEN
        CALL CHMALLOC('scalar.src','scala2','tmparr',natom,crl=tmparr)
        J=0
        DO I=1,NATOM
           IF(ISLCT(I) == 1) THEN
              J=J+1
              TMPARR(J)=ARRAY(I)
           ENDIF
        ENDDO
        IF(PRNLEV >= 2) WRITE(OUTU,'(A,I6,A,I6)') 'Writing ',J,' items to unit ',INUMBR  
        WRITE(INUMBR,'(G14.6,999(G14.6))') TMPARR(1:J)
        CALL CHMDEALLOC('scalar.src','scala2','tmparr',natom,crl=tmparr)
     ENDIF
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'SET ') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=RNUMBR
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'LT  ') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     RNUMBR2=NEXTF(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1 .AND. ARRAY(i) <= RNUMBR) ARRAY(I)=RNUMBR2
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'GT  ') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     RNUMBR2=NEXTF(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1 .AND. ARRAY(i) >= RNUMBR) ARRAY(I)=RNUMBR2
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'ADD ') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=ARRAY(I)+RNUMBR
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'RECI') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           IF (ABS(ARRAY(I)) < TOL) THEN
              CALL WRNDIE(0,'<SCALAR>','Element zero. No operation')
           ELSE
              ARRAY(I)=1.0/ARRAY(I)
           ENDIF
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'MULT') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=RNUMBR*ARRAY(I)
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'DIVI') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     IF (ABS(RNUMBR) < TOL) THEN
        CALL WRNDIE(-1,'<SCALAR>','CANT DIVIDE BY ZERO')
     ELSE
        DO I=1,NATOM
           IF (ISLCT(I) == 1) ARRAY(I)=ARRAY(I)/RNUMBR
        ENDDO
     ENDIF
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'SIGN') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           IF(ARRAY(I) > 0.0) ARRAY(I)=1.0
           IF(ARRAY(I) < 0.0) ARRAY(I)=-1.0
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'INDE') THEN
     IS=0
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           IS=IS+1
           ARRAY(I)=IS
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'INTE') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=INT(ARRAY(I))
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'NORM') THEN
     RNORM=0.0
     DO I=1,NATOM
        IF (ISLCT(I) == 1) RNORM=RNORM+ARRAY(I)*ARRAY(I)
     ENDDO
     IF (RNORM < TOL) THEN
        CALL WRNDIE(0,'<SCALAR>','NORM is zero. No operation ' &
             //'performed')
     ELSE
        RNORM=SQRT(RNORM)
        DO I=1,NATOM
           IF (ISLCT(I) == 1) ARRAY(I)=ARRAY(I)/RNORM
        ENDDO
     ENDIF
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'ABS ') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=ABS(ARRAY(I))
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'LOG ') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           IF (ARRAY(I) < TOL) THEN
              CALL WRNDIE(0,'<SCALAR>','Element negative. No ' &
                   //'operation')
           ELSE
              ARRAY(I)=LOG(ARRAY(I))
           ENDIF
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'EXP ') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=EXP(ARRAY(I))
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'POWE') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           IF (ARRAY(I) < TOL) THEN
              CALL WRNDIE(0,'<SCALAR>','Element negative. No ' &
                   //'operation')
           ELSE
              ARRAY(I)=ARRAY(I)**RNUMBR
           ENDIF
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'IPOW') THEN
     INUMBR=NEXTI(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           ARRAY(I)=ARRAY(I)**INUMBR
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'POW2') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           ARRAY(I)=ARRAY(I)**2
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'SQRT') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           IF (ARRAY(I) < ZERO) THEN
              CALL WRNDIE(0,'<SCALAR>','Element negative. No ' &
                   //'operation')
           ELSE
              ARRAY(I)=SQRT(ARRAY(I))
           ENDIF
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'RAND') THEN
     DO I=1,NATOM
        IF (ISLCT(I) == 1) THEN
           ARRAY(I)=RANUMB()
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'MIN ') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=MIN(RNUMBR,ARRAY(I))
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'MAX ') THEN
     RNUMBR=NEXTF(COMLYN,COMLEN)
     DO I=1,NATOM
        IF (ISLCT(I) == 1) ARRAY(I)=MAX(RNUMBR,ARRAY(I))
     ENDDO
     !
     ! ----------------------------------------------------------------------
     ! SAPATEL
#if KEY_CHEQ==1
  ELSE IF (WRD == 'QHIS') THEN
     CALL HISTOGRAM(ARRAY,COMLYN,COMLEN)
  ELSE IF (WRD == 'FQHI') THEN
     CALL HISTOGRAM2(ISLCT,COMLYN,COMLEN)
#endif 
     !
     ! SAPATEL
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'SHOW') THEN
335  FORMAT(' ( ',4(A,1X),')  ',G12.5)
     IF(INDXA(COMLYN,COMLEN,'SORT') > 0) THEN
        CALL SORTP(NATOM,IWORK,ORDERR,ARRAY,1,0,0,0,0,0,0)
        DO IS=1,NATOM
           I=IWORK(IS)
           IF(ISLCT(I) == 1) THEN
              IRES=GETRES(I,IBASE,NRES)
              ISEG=GETSEG(IRES,NICTOT,NSEG)
              IF(PRNLEV >= 2) WRITE(OUTU,335) &
                   SEGID(ISEG)(1:idleng),RES(IRES)(1:idleng), &
                   RESID(IRES)(1:idleng),ATYPE(I)(1:idleng), &
                   ARRAY(I)
           ENDIF
        ENDDO
     ELSE
        DO ISEG=1,NSEG
           DO IRES=NICTOT(ISEG)+1,NICTOT(ISEG+1)
              DO I=IBASE(IRES)+1,IBASE(IRES+1)
                 IF (ISLCT(I) == 1) THEN
                    IF(PRNLEV >= 2) WRITE(OUTU,335) &
                         SEGID(ISEG)(1:idleng),RES(IRES)(1:idleng), &
                         RESID(IRES)(1:idleng),ATYPE(I)(1:idleng), &
                         ARRAY(I)
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
     ENDIF
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'STAT') THEN
     IF(INDXA(COMLYN,COMLEN,'WEIG') > 0) THEN
        IST=NEXTI(COMLYN,COMLEN)
        IF(IST < 1 .OR. IST > MAXSTO) THEN
           CALL WRNDIE(-1,'<SCALAR>','Store number out of range')
           RETURN
        ENDIF
        IF(.not. allocated(PTRSTO(IST)%a)) THEN
           CALL WRNDIE(-1,'<SCALAR>','Store empty')
           RETURN
        ELSE IF(ptrsto(ist)%len /= NATOM) THEN
           CALL WRNDIE(-1,'<SCALAR>','Dim. mismatch')
           RETURN
        ELSE
           array2(1:natom) = ptrsto(ist)%a(1:natom)
        ENDIF
     ELSE
        array2(1:natom)=one   
     ENDIF
     !
     IF(INDXA(COMLYN,COMLEN,'MASS') > 0) THEN
        DO I=1,NATOM
           ARRAY2(I)=ARRAY2(I)*AMASS(I)
        ENDDO
     ENDIF
     NSL=0
     ARMIN=9999.0
     ARMAX=-9999.0
     ARAVE=0.0
     ARWGHT=0.0
     DO I=1,NATOM
        IF(ISLCT(I) == 1) THEN
           ARAVE=ARAVE+ARRAY(I)*ARRAY2(I)
           ARWGHT=ARWGHT+ARRAY2(I)
           IF(ARRAY(I) < ARMIN) ARMIN=ARRAY(I)
           IF(ARRAY(I) > ARMAX) ARMAX=ARRAY(I)
           NSL=NSL+1
        ENDIF
     ENDDO
     !
     ARTOT=ARAVE
     ARAVE=0.0
     IF(ARWGHT /= 0.0) ARAVE=ARTOT/ARWGHT
     !
     ARVAR=0.0
     DO I=1,NATOM
        IF(ISLCT(I) == 1) THEN
           ARVAR=ARVAR+(ARAVE-ARRAY(I))**2*ARRAY2(I)
        ENDIF
     ENDDO
     IF(ARWGHT /= 0.0) THEN
        ARVAR=ARVAR/ARWGHT
        IF(ARVAR > 0.0) THEN
           ARVAR=SQRT(ARVAR)
        ELSE
           ARVAR=0.0
        ENDIF
     ELSE
        ARVAR=0.0
     ENDIF
     !
     IF(PRNLEV >= 2) WRITE(OUTU,88) NSL,ARMIN,ARMAX,ARWGHT, &
          ARAVE,ARVAR,ARTOT
88   FORMAT(' Statistics for',I5,' selected atoms:'/ &
          '       minimum = ',1PG14.6,'  maximum = ',1PG14.6, &
          ' weight = ',1PG14.6/ &
          '       average = ',1PG14.6,'  variance= ',1PG14.6, &
          ' total  = ',1PG14.6)
     !
     call set_param('SMIN',ARMIN)
     call set_param('SMAX',ARMAX)
     call set_param('SAVE',ARAVE)
     call set_param('SWEI',ARWGHT)
     call set_param('STOT',ARTOT)
     call set_param('SVAR',ARVAR)
     CALL set_param('NSEL',NSL)
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'AVER' .OR. WRD == 'TOTA') THEN
     LTOTAL=(WRD == 'TOTA')
     ICODE=0
     IF(INDXA(COMLYN,COMLEN,'BYGR') > 0) ICODE=4
     IF(INDXA(COMLYN,COMLEN,'BYRE') > 0) ICODE=3
     IF(INDXA(COMLYN,COMLEN,'BYSE') > 0) ICODE=2
     IF(INDXA(COMLYN,COMLEN,'ALL') > 0) ICODE=1
     IF(INDXA(COMLYN,COMLEN,'WEIG') > 0) THEN
        !---- Procedures GET-STORE-NUMBER / FETCH-SAVED-VECTOR
        IST=NEXTI(COMLYN,COMLEN)
        IF(IST < 1 .OR. IST > MAXSTO) THEN
           CALL WRNDIE(-1,'<SCALAR>','Store number out of range')
           RETURN
        ENDIF
        IF(.not. allocated(PTRSTO(IST)%a) ) THEN
           CALL WRNDIE(-1,'<SCALAR>','Store empty')
           RETURN
        ELSE IF (ptrSTO(IST)%len /= NATOM) THEN
           CALL WRNDIE(-1,'<SCALAR>','Dim. mismatch')
           RETURN
        ELSE
           array2(1:natom) = ptrsto(ist)%a(1:natom)
        ENDIF
     ELSE
        ARRAY2(1:NATOM) = ONE
     ENDIF
     !
     IF(INDXA(COMLYN,COMLEN,'MASS') > 0) THEN
        DO I=1,NATOM
           ARRAY2(I)=ARRAY2(I)*AMASS(I)
        ENDDO
     ENDIF
     IF (ICODE == 1) THEN
        !         DO FOR ALL ATOMS
        IS=1
        IQ=NATOM
        !---- Procedure AVERAGE-THESE-ATOMS
        ARAVE=0.0
        ARWGHT=0.0
        NSL=0
        DO I=IS,IQ
           IF (ISLCT(I) == 1) THEN
              ARAVE=ARAVE+ARRAY(I)*ARRAY2(I)
              ARWGHT=ARWGHT+ARRAY2(I)
              NSL=NSL+1
           ENDIF
        ENDDO
        IF(NSL > 0) THEN
           IF(LTOTAL) ARWGHT=ONE
           IF(ABS(ARWGHT) > TOL) THEN
              ARAVE=ARAVE/ARWGHT
              DO I=IS,IQ
                 IF(ISLCT(I) == 1) ARRAY(I)=ARAVE
              ENDDO
           ENDIF
        ENDIF
        !----
     ELSE IF (ICODE == 2) THEN
        !     DO FOR EACH SEGMENT
        ISEG=0
        IF (ISEG >= NSEG) GOTO 420
400     CONTINUE
        ISEG=ISEG+1
        IS=IBASE(NICTOT(ISEG)+1)+1
        IQ=IBASE(NICTOT(ISEG+1)+1)
        !---- Procedure AVERAGE-THESE-ATOMS
        ARAVE=0.0
        ARWGHT=0.0
        NSL=0
        DO I=IS,IQ
           IF (ISLCT(I) == 1) THEN
              ARAVE=ARAVE+ARRAY(I)*ARRAY2(I)
              ARWGHT=ARWGHT+ARRAY2(I)
              NSL=NSL+1
           ENDIF
        ENDDO
        IF(NSL > 0) THEN
           IF(LTOTAL) ARWGHT=ONE
           IF(ABS(ARWGHT) > TOL) THEN
              ARAVE=ARAVE/ARWGHT
              DO I=IS,IQ
                 IF(ISLCT(I) == 1) ARRAY(I)=ARAVE
              ENDDO
           ENDIF
        ENDIF
        !----
        IF (.NOT.(ISEG >= NSEG)) GOTO 400
420     CONTINUE
     ELSE IF (ICODE == 3) THEN
        !         DO FOR EACH RESUDUE
        IRES=0
        IF (IRES >= NRES) GOTO 450
430     CONTINUE
        IRES=IRES+1
        IS=IBASE(IRES)+1
        IQ=IBASE(IRES+1)
        !---- Procedure AVERAGE-THESE-ATOMS
        ARAVE=0.0
        ARWGHT=0.0
        NSL=0
        DO I=IS,IQ
           IF (ISLCT(I) == 1) THEN
              ARAVE=ARAVE+ARRAY(I)*ARRAY2(I)
              ARWGHT=ARWGHT+ARRAY2(I)
              NSL=NSL+1
           ENDIF
        ENDDO
        IF(NSL > 0) THEN
           IF(LTOTAL) ARWGHT=ONE
           IF(ABS(ARWGHT) > TOL) THEN
              ARAVE=ARAVE/ARWGHT
              DO I=IS,IQ
                 IF(ISLCT(I) == 1) ARRAY(I)=ARAVE
              ENDDO
           ENDIF
        ENDIF
        !----
        IF (.NOT.(IRES >= NRES)) GOTO 430
450     CONTINUE
     ELSE IF (ICODE == 4) THEN
        !     DO FOR EACH GROUP
        IRES=0
        IF (IRES >= NGRP) GOTO 480
470     CONTINUE
        IRES=IRES+1
        IS=IGPBS(IRES)+1
        IQ=IGPBS(IRES+1)
        !---- AVERAGE-THESE-ATOMS
        ARAVE=0.0
        ARWGHT=0.0
        NSL=0
        DO I=IS,IQ
           IF (ISLCT(I) == 1) THEN
              ARAVE=ARAVE+ARRAY(I)*ARRAY2(I)
              ARWGHT=ARWGHT+ARRAY2(I)
              NSL=NSL+1
           ENDIF
        ENDDO
        IF(NSL > 0) THEN
           IF(LTOTAL) ARWGHT=ONE
           IF(ABS(ARWGHT) > TOL) THEN
              ARAVE=ARAVE/ARWGHT
              DO I=IS,IQ
                 IF(ISLCT(I) == 1) ARRAY(I)=ARAVE
              ENDDO
           ENDIF
        ENDIF
        !----
        IF (.NOT.(IRES >= NGRP)) GOTO 470
480     CONTINUE
     ELSE
        CALL WRNDIE(0,'<SCALA2>','Averaging field not specified')
     ENDIF
     !
     !-----------------------------------------------------------------------
  ELSE IF (WRD == 'HBCO') THEN
     ! Process Hydrogen Bond count request
     !
     DO I=1,NATOM
        IF(ISLCT(I) == 1) ARRAY(I)=0
     ENDDO
     DO I=1,NHB
        IF(KHB(I) > 0) THEN
           IF(ISLCT(KHB(I)) == 1) ARRAY(KHB(I))=ARRAY(KHB(I))+1
        ELSE IF(IHB(I) > 0) THEN
           IF(ISLCT(IHB(I)) == 1) ARRAY(IHB(I))=ARRAY(IHB(I))+1
        ENDIF
        IF(JHB(I) > 0) THEN
           IF(ISLCT(JHB(I)) == 1) ARRAY(JHB(I))=ARRAY(JHB(I))+1
        ENDIF
     ENDDO
     !
     !-----------------------------------------------------------------------
  ELSE
     CALL WRNDIE(1,'<SCALAR>','Unrecognized option')
  ENDIF
  !
  RETURN
END SUBROUTINE SCALA2

end module scalar_module

! outside module to avoid dependency cycle
! callers should declare explicit interface
SUBROUTINE SELPROP(WPROP,ARRAY,NFLAGS,ERR)
  !-----------------------------------------------------------------------
  !     This routine fills the propery array based on the selected key.
  !
  !     BRB 21-DEC-1996
  !
  use chm_kinds
  use dimens_fcm
  use bases_fcm
  use psf
  use coord
  use coordc
  use cnst_fcm
  use deriv
  use econtmod
  use flucq
  use param
  use gcmc
  use varcutm
  use scalar_module
  use ace_module
  use gb_common
  use cheq
  use flucqm
#if KEY_SGLD==1
  use sgld                 
#endif
  implicit none
  !
  character(len=4) WPROP
  real(chm_real) ARRAY(:)
  INTEGER NFLAGS
  LOGICAL ERR
  ERR=.TRUE.  ! request print and report of any error
  ! all those USEs are just to provide args to SCAFIL :P
  CALL SCAFIL(WPROP,ARRAY,DX,DY,DZ,X,Y,Z,WMAIN, &
#if KEY_CHEQ==1
       DCH,ECH,EHA,                    & 
#endif
       XCOMP,YCOMP,ZCOMP,WCOMP,AMASS,CG,RSCLF,IMOVE,IAC,ITC, &
       REFX,REFY,REFZ,KCNSTR, &
       ALP,EFF,VDWR,ECONT,ICAAUTO, &
       NFLAGS, ERR &
       , Alph_Gb, SigX_Gb, SigY_Gb, SigZ_Gb, T_Gb, GB_Atm, Qanalys &  
#if KEY_ACE==1
       ,EFVOL,CHRAD,ACEHYD,BSOLV ,CGIACP,CGSACP &   
#endif
#if KEY_FLUCQ==1
       ,FQCHI,FQZETA,FQPRIN,FQCHMA,FQJZ,FQCFOR,FQOLDQ,QFLUC &   
#endif
#if KEY_GCMC==1
       ,qgcmc,GCMCON,GCBLKR       & 
#endif
#if KEY_WCA==1
       ,WCA                 & 
#endif
#if KEY_SGLD==1
       ,SGWT,SGGAMMA                 & 
#endif
#if KEY_COMP2==1
       ,XCOMP2,YCOMP2,ZCOMP2,WCOMP2 & 
#endif
       )
  !
  IF(ERR) CALL WRNDIE(-2,'<SELPROP>', &
       'Error encountered in processing property array selection')
  !
  RETURN
END SUBROUTINE SELPROP

