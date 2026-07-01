module gbmv
  use chm_kinds
  use gb_common,only:eps_gb,igentype, &
     alph_gb, &
     rad   => r_gb, &
     t     => t_gb

  implicit none

  logical,save :: qgbmv
  logical,private,save :: qhybrid  ! variable used for hybrid explicit/implicit phmd
  logical,private,save :: qexclgb  ! variable used to exclude selected residues for HDGB. BD/MF

  integer, allocatable, dimension(:),save :: adlistgb !BD/MF
  integer(int_byte),allocatable,dimension(:) ::  &
       value,grid,grid2,surf_atom,surfx

  real(chm_real),allocatable,dimension(:),save :: wtx,wty,wtz, &
       wt1,wt2,wtr,wts,xcp,ycp,zcp,rtmp,vtmp, &
       gamma,f,g,pox,poy,poz, &
       dx_gb,dy_gb,dz_gb, &
       wt4
!MS/MF
#if KEY_HDGBVDW==1
  real(chm_real),allocatable,dimension(:),save ::dz_gbvdw
  real(chm_real),allocatable,dimension(:),save :: pvdw
  real(chm_real),allocatable,dimension(:),save :: dp
  real(chm_real),allocatable,dimension(:),save :: ddd_hdgb
#endif

  real(chm_real),allocatable,dimension(:),save ::  &
       AX_GB,AY_GB,AZ_GB, &
       BX_GB,BY_GB,BZ_GB, &
       CX_GB,CY_GB,CZ_GB, &
       VX1,VY1,VZ1,RA, &
       RB,R7D,TEMP1_GB,TEMP2_GB, &
       GCUTR

  integer,allocatable,dimension(:),save :: alist,index,ptr, &
       xindex,blist,clist,dlist,d

  integer, parameter :: maxtmp = 100

  ! This common block file contains the control parameters
  ! for generalized Born calculations using a molecular volume
  ! formalism of
  ! M.S. Lee, F. Salsbury Jr., and C.L. Brooks III, JCP 2002.
  ! doi:10.1063/1.1480013
  ! See also M.S. Lee, M. Feig, F. Salsbury Jr., and C.L. Brooks III, JCC 2003.
  ! doi:10.1002/jcc.10272
  !
  ! The variables include:
  !
  ! Parameters for the GB equations
  !
  ! Note the polarization energy Gpol is given by:
  !                                                    q q
  !                             N   N                   i j
  !   G   =  -C  (1-1/eps){1/2 sum sum ------------------------------------ }
  !    pol     el              i=1 j=1 [r^2 + alpha *alpha exp(-D  )]^(0.5)
  !                                      ij        i      j      ij
  !
  integer,parameter :: GBNUM=200,MAXAT1=50000,MAXRADG=50

  INTEGER,save ::  &
       NGBAtom,IMPULSE_GB, &
       GBIter,ML,MemMult, &
       Surf_Wt_GB,MaxWeights,MaxTable,MaxSurf,NSurf, &
       NAngular,NWeights,NumR, &
       NX,NY,NZ,NXY,NGridGB,NTable,Offset, &
       MaxTablePerNode, &
       NPHI_GB,NGridGB_AL, &
       NGridGB_GB,NX_GB,NY_GB,NZ_GB,OFFSET_GB,GBStepCtr,ALFRQ, &
       STYPE,MTYPE_GB,CORR_GB, &
       DRSTEP,DRFRQ, NXP_GB,NYP_GB,NZP_GB, CUBIC_VSA_GB, &
       MODSTILL,GCUT,NRADG, &
       value_gb,grid_gb,grid2_gb,surf_atom_gb,nmvsel, &
       FAST_GB,SGBSTEP,SGBFRQ,CUTNUM

  LOGICAL,save :: GBGridReset,GBMVGrid,QWeight,FixA,ShowGrid_GB, &
       ECOMP_GB,EVDW_GB,ESURF_GB,CONV_GB

  real(chm_real),save :: &
       Beta,Shift,Lambda1,P1,P2,WPROBE,RShift,EShift, &
       Cent_X,Cent_Y,Cent_Z,dn,dn_inv,dndiag,XMIN,YMIN,ZMIN,XMAX, &
       YMAX,ZMAX,StillFac,P3,P4,P5,P6,P7,BufR,BufR2, &
       TOL_GB,SLOPE,On_X,Off_X, &
       EMP_GB,HSX1,HSX2, &
       Thresh1,Thresh2,Ext,CutA,KAPPA_GB,UOLD(3,3),TT, &
       XMIN_GB,XMAX_GB,YMIN_GB,YMAX_GB,ZMIN_GB,ZMAX_GB, &
       SA_GB,SB_GB,Surface_Area,SON_GB,SOFF_GB, &
       A1,A2,A3, T1, S2, MS0, MS1, MS2, MS3, &
       tWtScale,tWtScale2, & ! RunGBMV1F aux vars (final 3 lines)
       tR2C,tR2CI,tR2CIT,tR20,tR20I,tR2BUF, &
       SXDGB

  ! SJT/MF VDW
  real(chm_real),allocatable,dimension(:),save :: avdw, tvdw, &
       gmvdw, gmasp
  LOGICAL,save :: QGBVDW, QGBASP
  ! SJT/MF

  INTEGER,parameter :: MAXC_HDGB=300

  real(chm_real),save ::  A4, A5, &
       EPSIJ_HDGB,  &
       EPS0_HDGB, EPSW_HDGB, WEPS_HDGB,EPSMIN_HDGB, EPSMAX_HDGB, &
       ZS_HDGB, ZM_HDGB, ZT_HDGB, ST0_HDGB, WST_HDGB, &
       HSPL, ZMINSPL, ZMAXSPL, &
       HSPLNP, ZMINSPLNP, ZMAXSPLNP,NP0_HDGB,NPMIN_HDGB,NPW_HDGB, &
       cspl(4,maxc_hdgb), csplnp(4,maxc_hdgb)

  real(chm_real),allocatable,dimension(:),save :: r_hdgb, dr_hdgb

  ! are these variables used??? (MF)
  real(chm_real),save :: hspl_hdgb, zminspl_hdgb, zmaxspl_hdgb

  ! dimension GBNUM
  real(chm_real),allocatable,dimension(:),save :: &
       ra_hdgb, r7d_hdgb, az_hdgb, &
       cz_hdgb, bx_hdgb

  ! dimension MAXAT1
  real(chm_real),allocatable,dimension(:),save :: &
       rdist_hdgb,  &
       eps_hdgb, deps_hdgb, a3_hdgb, shift_hdgb, p_hdgb, s_hdgb, &
       surft_hdgb, dsurft_hdgb, atsa_hdgb

  LOGICAL,save :: QHDGBRC, QHDNOSW
#if KEY_HDGBVDW==1
! MS/MF

  real(chm_real),save :: &
       HSPLD,D0_HDGB, &
       DW_HDGB,DMIN_HDGB,DMAX_HDGB, &
       cspld(4,maxc_hdgb)
  real(chm_real),allocatable,dimension(:),save :: dc_hdgb,ddc_hdgb
  real(chm_real),allocatable,dimension(:), save:: dc2_hdgb,ddc2_hdgb
  real(chm_real),allocatable,dimension(:),save :: dp_hdgb,ddp_hdgb, &
       d_hdgb,dd_hdgb, &
       do_hdgb,ddo_hdgb, &
       dn_hdgb,ddn_hdgb
   INTEGER,save :: UNDP_HDGB,UNDC2_HDGB,UNDC_HDGB, &
       UNDO_HDGB,UNDN_HDGB

   real(chm_real),save :: A_CT3, A_CY, A_CPH, A_CC, &
       A_CA,  A_CT1, A_CT2, A_CM, &
       A_HS, A_HO, A_HP, A_HM, A_HY, A_HR, A_NH, A_NY, &
       A_OH, A_OY, A_OR, A_SM, A_SC, A_P, A_S

   real(chm_real),save :: DON_HDGB, &
       DWN_HDGB,DMINN_HDGB,DMAXN_HDGB, &
       DOC_HDGB,DWC_HDGB,DMINC_HDGB,DMAXC_HDGB, &
       DOC2_HDGB,DWC2_HDGB,DMINC2_HDGB,DMAXC2_HDGB, &
       DOP_HDGB,DWP_HDGB,DMINP_HDGB,DMAXP_HDGB, &
       DOO_HDGB,DWO_HDGB,DMINO_HDGB,DMAXO_HDGB, &
       ZMIN_C,ZMIN_O,ZMIN_C2,ZMIN_P,ZMIN_N, &
       ZMAX_C,ZMAX_O,ZMAX_C2,ZMAX_P,ZMAX_N

   real(chm_real),save :: &
       CSPLDN(4,maxc_hdgb),CSPLDC(4,maxc_hdgb), &
       CSPLDP(4,maxc_hdgb),CSPLDC2(4,maxc_hdgb), &
       CSPLDO(4,maxc_hdgb)


! MS/MF
#endif
#if KEY_DHDGB==1
!AP/MF
  REAL(chm_real),save :: &
       CIRCLERAD,CIRCLEMRAD, &
       MIR_FHDGB,MAR_FHDGB, &
       INC_FHDGB,NPS_FHDGB
  INTEGER,SAVE :: NDT_FHDGB
  INTEGER,parameter :: INCT_FHDGB=6
  INTEGER,PARAMETER :: INC_R0=2.50D0
  REAL(chm_real),save :: &
       INC_RAD, &
       MEMSPL(4,INCT_FHDGB),INCTH_FHDGB,THETAZ(INCT_FHDGB)
  INTEGER,PARAMETER :: PRO_LENGTH=600
  integer,parameter :: prov_length=6000
  REAL(chm_real),save :: EPS_DEF_B,NP_DEF_B, &
       EPS_DEF_A_SPL(4,PRO_LENGTH),EPS_DEF_C_SPL(4,PRO_LENGTH), &
       EPS_DEF_E_SPL(4,PRO_LENGTH), &
       NP_DEF_A_SPL(4,PRO_LENGTH),NP_DEF_C_SPL(4,PRO_LENGTH),NP_DEF_E_SPL(4,PRO_LENGTH), &
       HSPL_EPS_DEF_A,HSPL_EPS_DEF_C, &
       HSPL_EPS_DEF_E, &
       HSPL_NP_DEF_A,HSPL_NP_DEF_C, &
       HSPL_NP_DEF_E, &
       ZMIN_EPS_DEF_A,ZMAX_EPS_DEF_A, &
       ZMIN_EPS_DEF_C,ZMAX_EPS_DEF_C, &
       ZMIN_EPS_DEF_E,ZMAX_EPS_DEF_E, &
       ZMIN_NP_DEF_A,ZMAX_NP_DEF_A, &
       ZMIN_NP_DEF_C,ZMAX_NP_DEF_C, &
       ZMIN_NP_DEF_E,ZMAX_NP_DEF_E, &
       MIN_EPS_DEF_A,MAX_EPS_DEF_A, &
       MIN_EPS_DEF_C,MAX_EPS_DEF_C, &
       MIN_EPS_DEF_E,MAX_EPS_DEF_E, &
       MIN_NP_DEF_A,MAX_NP_DEF_A, &
       MIN_NP_DEF_C,MAX_NP_DEF_C, &
       MIN_NP_DEF_E,MAX_NP_DEF_E, &
       SEPS_C,SNP_C,P_DHDGB
  LOGICAL,SAVE :: QSPLINE,QPROVIDE,QWRTCENT,QDHDGBDEBUG,QWRITES
  LOGICAL,SAVE :: QSMALL
  INTEGER,SAVE :: STARTATM,ATMEND
  REAL(chm_real),save :: READCX(PROV_LENGTH),READCY(PROV_LENGTH), &
       PENALTY_OFF,PENALTY_PREFACT,PENALTY_FACT, &
       PENALTY_OFFL,PENALTY_PREFACTL,PENALTY_FACTL, &
       INNERC,LAMF, &
       MAXEXP
  INTEGER,SAVE :: UN_FHDGB
  integer,save :: length_fhdgb
  integer,parameter :: maxlen_fhdgb=256
  logical,save :: qopen_fhdgb, qform_fhdgb, qwrite_fhdgb, error_fhdgb
  ! BIOVIA Code Start : Bug fix
  character(len=maxlen_fhdgb),save :: name_fhdgb
  ! BIOVIA Code End
  REAL,save::LOOKUP_COEFF(4*7**5)
  INTEGER,SAVE :: LOOK_SIZE
#endif

contains

  Subroutine Gbmv_Set(comlyn, comlen)
    !-------------------------------------------------------------
    ! This routine starts the molecular volume Gen. Born code
    ! by preparing the necessary structures
    ! First developed by Michael S. Lee (MSL) 05/01
    !-------------------------------------------------------------

    use exfunc
    use stream
    use dimens_fcm
    use consta
    use psf
    use memory
    use number
    use coord
    use param
    use string
    use surface
    use select
#if KEY_PARALLEL==1
    use parallel
#endif 
#if KEY_DHDGB==1
!AP/MF
    use dhdgb,only:sdef,samass,QFHDGB,MITHICK_FHDGB,MATHICK_FHDGB,FRIC_DHDGB,MAXNUMS
#endif
    Character(len=*) :: Comlyn
    Integer Comlen,I,J,K,WType
    real(chm_real) R,r1,rtemp
    Integer TINX
    integer gbmv_sel(natom)
    integer islctgb(natom) !BD/MF
    logical qhbok  ! logical to check whether hybrid is ok

    ! SJT/MF
    Integer UNEPS_HDGB, UNNP_HDGB, N_HDGB, NNP_HDGB, NHDGB
    real(chm_real)  X_HDGB(300), A_HDGB(300)
    real(chm_real)  XNP_HDGB(300), ANP_HDGB(300)
    Integer IX1_HDGB
    real(chm_real)  ZI, EPSI, DEPSI, DST, ST
    LOGICAL QEPSOUT, QNPOUT

    integer :: length_hdgb
    integer,parameter :: maxlen_hdgb=256
    character(len=maxlen_hdgb) :: name_hdgb
    logical qopen_hdgb, qform_hdgb, qwrite_hdgb, error_hdgb

#if KEY_HDGBVDW==1
! MS/MF
    real*8  XD_HDGB(300), AD_HDGB(300)
    real*8 DI, DDI
    LOGICAL QDPOUT,QDC2OUT,QDCOUT,QDOOUT,QDNOUT

! MS/MF
#endif
#if KEY_DHDGB==1
!AP/MF
    INTEGER UNEPS_DEF_A, UNEPS_DEF_C,UNEPS_DEF_E
    INTEGER UNNP_DEF_A, UNNP_DEF_C,UNNP_DEF_E
    INTEGER NEPS_DEF_A, NEPS_DEF_C,NEPS_DEF_E
    INTEGER NNP_DEF_A, NNP_DEF_C,NNP_DEF_E
    INTEGER UN_PROVIDE
!    Integer UN_FHDGB,LOOKUP_DIM,TEMP_FHDGB
    INTEGER last
    REAL(chm_real) DUM1(NATOM),DUM2(NATOM),DUM3(NATOM)
    REAL(chm_real) DUM4(NATOM),DUM5(NATOM)
    integer length_eps_a,length_eps_c,length_eps_e
    integer length_np_a,length_np_c,length_np_e
    character name_eps_def_a*(maxlen_hdgb)
    character name_eps_def_c*(maxlen_hdgb)
    character name_eps_def_e*(maxlen_hdgb)
    character name_np_def_a*(maxlen_hdgb)
    character name_np_def_c*(maxlen_hdgb)
    character name_np_def_e*(maxlen_hdgb)
    character name_prov*(maxlen_hdgb)
    logical qopen_epsa, qform_epsa, qwrite_epsa, error_epsa
    logical qopen_epsc, qform_epsc, qwrite_epsc, error_epsc
    logical qopen_epse, qform_epse, qwrite_epse, error_epse
    logical qopen_npa, qform_npa, qwrite_npa, error_npa
    logical qopen_npc, qform_npc, qwrite_npc, error_npc
    logical qopen_npe, qform_npe, qwrite_npe, error_npe
    logical qopen_prov,qform_prov,qwrite_prov
    integer length
    REAL(chm_real) AEPS_A(PRO_LENGTH),AEPS_C(PRO_LENGTH)
    REAL(chm_real) AEPS_E(PRO_LENGTH)
    REAL(chm_real) ANP_A(PRO_LENGTH),ANP_C(PRO_LENGTH)
    REAL(chm_real) ANP_E(PRO_LENGTH)
    REAL(chm_real) XEPS_A(PRO_LENGTH),XEPS_C(PRO_LENGTH),XEPS_E(PRO_LENGTH)
    REAL(chm_real) XNP_A(PRO_LENGTH),XNP_C(PRO_LENGTH),XNP_E(PRO_LENGTH)
#endif

    ALFRQ   = GtrmI(Comlyn,Comlen, 'UPDA',-1)
    IF (ALFRQ .GT. -1) THEN
       IF (PRNLEV .GE. 5) &
            WRITE(OUTU,*) &
            ' GBMV alphas will update every ',ALFRQ,' steps.'
       return
    endif

    clear_blk: if (indxa(comlyn,comlen,'CLEA').gt.0) then
       if (allocated(adlistgb)) &
          call chmdealloc('gbmvmodule.src','gbmv_set','adlistgb',NATOM,intg=adlistgb) !BD/MF
       if (.not. qgbmv) then
          call wrndie(-1,'<gbmvmodule.src>Gbmv_SET', &
               'Called GBMV CLEAR w/o being active.')
          return
       endif

       if (.not. gbmvgrid) then

          if (allocated(alist)) &
               call chmdealloc('gbmvmodule.src','Gbmv_Set','alist',maxtable,intg=alist)

          ! MFC/IVK bugfix 5 Array used as bytes thus 1/4, add one for
          ! non-multiple of 4 for MaxTable, need some extra spots.
          if (allocated(value)) &
               call chmdealloc('gbmvmodule.src','Gbmv_Set','value',maxtable,iby=value)
          if (allocated(surf_atom)) &
               call chmdealloc('gbmvmodule.src','Gbmv_Set','surf_atom',maxsurf,iby=surf_atom)
          if (allocated(index)) then
             call chmdealloc('gbmvmodule.src','Gbmv_Set','index',ngridgb,intg=index)
             call chmdealloc('gbmvmodule.src','Gbmv_Set','ptr',ngridgb,intg=ptr)
          endif

          ! MF extra temporary storage for optimized version
          IF (FAST_GB.GT.0) THEN
             call chmdealloc('gbmvmodule.src','Gbmv_Set','surfx',maxsurf,iby=surfx)
             call chmdealloc('gbmvmodule.src','Gbmv_Set','rtmp',9*nmvsel,crl=rtmp)
             call chmdealloc('gbmvmodule.src','Gbmv_Set','vtmp',11*nmvsel,crl=vtmp)

             call chmdealloc('gbmvmodule.src','Gbmv_Set','xindex',maxsurf*6,intg=xindex)
             call chmdealloc('gbmvmodule.src','Gbmv_Set','blist',maxtable,intg=blist)
             call chmdealloc('gbmvmodule.src','Gbmv_Set','clist',maxtable*4,intg=clist)
             call chmdealloc('gbmvmodule.src','Gbmv_Set','dlist',maxtable,intg=dlist)
          ENDIF

          call chmdealloc('gbmvmodule.src','Gbmv_Set','gamma',nmvsel,crl=gamma)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','f',nmvsel,crl=f)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','g',nmvsel,crl=g)
       ELSE
          call chmdealloc('gbmvmodule.src','Gbmv_Set','grid',ngridgb,iby=grid)
          if (allocated(grid2)) then
             if (CONV_GB) then
                call chmdealloc('gbmvmodule.src','Gbmv_Set','grid2',ngridgb,iby=grid2)
             else
                call chmdealloc('gbmvmodule.src','Gbmv_Set','grid2',100,iby=grid2)
             endif
          endif
          if (allocated(pox)) then
             call chmdealloc('gbmvmodule.src','Gbmv_Set','pox',ml,crl=pox)
             call chmdealloc('gbmvmodule.src','Gbmv_Set','poy',ml,crl=poy)
             call chmdealloc('gbmvmodule.src','Gbmv_Set','poz',ml,crl=poz)
             i = 1.1D0 * (FOUR/THREE) * PI * (WPROBE/DN)**3
             call chmdealloc('gbmvmodule.src','Gbmv_Set','d',i,intg=d)
          endif
       ENDIF
       if (allocated(alph_gb)) &
            call chmdealloc('gbmvmodule.src','Gbmv_Set','alph_gb',nmvsel,crl=alph_gb)
       if (allocated(t)) &
            call chmdealloc('gbmvmodule.src','Gbmv_Set','t',nmvsel,crl=t)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','rad',nmvsel,crl=rad)

       ! SJT/MF VDW
       if (allocated(tvdw)) &
            call chmdealloc('gbmvmodule.src','Gbmv_Set','tvdw',nmvsel,crl=tvdw)
       if (allocated(avdw)) &
            call chmdealloc('gbmvmodule.src','Gbmv_Set','avdw',nmvsel,crl=avdw)
       if (allocated(gmvdw)) &
            call chmdealloc('gbmvmodule.src','Gbmv_Set','gmvdw',nmvsel,crl=gmvdw)
       if (allocated(gmasp)) &
            call chmdealloc('gbmvmodule.src','Gbmv_Set','gmasp',nmvsel,crl=gmasp)

       if (allocated(r_hdgb)) &
            call chmdealloc('gbmvmodule.src','Gbmv_Set','r_hdgb',nmvsel,crl=r_hdgb)
       if (allocated(dr_hdgb)) &
            call chmdealloc('gbmvmodule.src','Gbmv_Set','dr_hdgb',nmvsel,crl=dr_hdgb)

       if (allocated(ra_hdgb)) then
          call chmdealloc('gbmvmodule.src','Gbmv_Set','ra_hdgb',gbnum,crl=ra_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','r7d_hdgb',gbnum,crl=r7d_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','az_hdgb',gbnum,crl=az_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','cz_hdgb',gbnum,crl=cz_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','bx_hdgb',gbnum,crl=bx_hdgb)
       endif
       if (allocated(rdist_hdgb)) then
          call chmdealloc('gbmvmodule.src','Gbmv_Set','rdist_hdgb',nmvsel,crl=rdist_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','eps_hdgb',nmvsel,crl=eps_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','deps_hdgb',nmvsel,crl=deps_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','a3_hdgb',nmvsel,crl=a3_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','shift_hdgb',nmvsel,crl=shift_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','p_hdgb',nmvsel,crl=p_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','s_hdgb',nmvsel,crl=s_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','surft_hdgb',nmvsel,crl=surft_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','dsurft_hdgb',nmvsel,crl=dsurft_hdgb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','atsa_hdgb',nmvsel,crl=atsa_hdgb)
       endif

!MS/MF
#if KEY_HDGBVDW==1
       if (allocated(pvdw)) then
          call chmdealloc('gbmvmodule.src','Gbmv_Set','pvdw',nmvsel,crl=pvdw)
       endif
       if (allocated(dp)) then
          call chmdealloc('gbmvmodule.src','Gbmv_Set','dp',nmvsel,crl=dp)
       endif
       if (allocated(ddd_hdgb)) then
          call chmdealloc('gbmvmodule.src','Gbmv_Set','ddd_hdgb',nmvsel,crl=ddd_hdgb)
       endif
       if (allocated(dx_gb)) then
          call chmdealloc('gbmvmodule.src','Gbmv_Set','dz_gbvdw',nmvsel,crl=dz_gbvdw)
       endif
       if (allocated(dn_hdgb)) then
         call chmdealloc('gbmvmodule.src','Gbmv_Set','dn_hdgb',nmvsel,crl=dn_hdgb)
         call chmdealloc('gbmvmodule.src','Gbmv_Set','ddn_hdgb',nmvsel,crl=ddn_hdgb)
       endif
       if (allocated(dc2_hdgb)) then
         call chmdealloc('gbmvmodule.src','Gbmv_Set','dc2_hdgb',nmvsel,crl=dc2_hdgb)
         call chmdealloc('gbmvmodule.src','Gbmv_Set','ddc2_hdgb',nmvsel,crl=ddc2_hdgb)
       endif
       if (allocated(dc_hdgb)) then
         call chmdealloc('gbmvmodule.src','Gbmv_Set','dc_hdgb',nmvsel,crl=dc_hdgb)
         call chmdealloc('gbmvmodule.src','Gbmv_Set','ddc_hdgb',nmvsel,crl=ddc_hdgb)
       endif
       if (allocated(do_hdgb)) then
         call chmdealloc('gbmvmodule.src','Gbmv_Set','do_hdgb',nmvsel,crl=do_hdgb)
         call chmdealloc('gbmvmodule.src','Gbmv_Set','ddo_hdgb',nmvsel,crl=ddo_hdgb)
       endif
       if (allocated(dp_hdgb)) then
         call chmdealloc('gbmvmodule.src','Gbmv_Set','dp_hdgb',nmvsel,crl=dp_hdgb)
         call chmdealloc('gbmvmodule.src','Gbmv_Set','ddp_hdgb',nmvsel,crl=ddp_hdgb)
       endif
#endif

       ! SJT/MF VDW

       if (allocated(dx_gb)) then
          call chmdealloc('gbmvmodule.src','Gbmv_Set','dx_gb',nmvsel,crl=dx_gb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','dy_gb',nmvsel,crl=dy_gb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','dz_gb',nmvsel,crl=dz_gb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','wt4',maxweights,crl=wt4)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','AX_GB',gbnum,crl=AX_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','AY_GB',gbnum,crl=AY_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','AZ_GB',gbnum,crl=AZ_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','BX_GB',gbnum,crl=BX_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','BY_GB',gbnum,crl=BY_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','BZ_GB',gbnum,crl=BZ_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','CX_GB',gbnum,crl=CX_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','CY_GB',gbnum,crl=CY_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','CZ_GB',gbnum,crl=CZ_GB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','VX1',gbnum,crl=VX1)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','VY1',gbnum,crl=VY1)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','VZ1',gbnum,crl=VZ1)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','RA',gbnum,crl=RA)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','RB',gbnum,crl=RB)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','R7D',gbnum,crl=R7D)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','temp1_gb',gbnum,crl=temp1_gb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','temp2_gb',gbnum,crl=temp2_gb)
          call chmdealloc('gbmvmodule.src','Gbmv_Set','GCUTR',maxradg,crl=GCUTR)
       endif

       call chmdealloc('gbmvmodule.src','Gbmv_Set','xcp',nmvsel,crl=xcp)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','ycp',nmvsel,crl=ycp)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','zcp',nmvsel,crl=zcp)

       call chmdealloc('gbmvmodule.src','Gbmv_Set','wtx',maxweights,crl=wtx)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','wty',maxweights,crl=wty)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','wtz',maxweights,crl=wtz)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','wt1',maxweights,crl=wt1)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','wt2',maxweights,crl=wt2)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','wtr',maxweights,crl=wtr)
       call chmdealloc('gbmvmodule.src','Gbmv_Set','wts',maxweights,crl=wts)

       QGBMV = .false.
       IGenType = 0
       RETURN
    ENDIF  clear_blk

    ! Read in parameters

    GBMVGrid = (INDXA(COMLYN,COMLEN,'GRID').gt.0)
    QWeight =  (INDXA(Comlyn,Comlen,'WEIGHT').gt.0)
    FixA    =  (INDXA(Comlyn,Comlen,'FIXA').gt.0)
    ECOMP_GB = (INDXA(Comlyn,Comlen,'COUT').gt.0)
    EVDW_GB  = (INDXA(Comlyn,Comlen,'VDW').gt.0)
    ESURF_GB  = (INDXA(Comlyn,Comlen,'SURF').gt.0)
    MTYPE_GB = 2
    IF (INDXA(Comlyn,Comlen,'ARITH') .GT. 0) MTYPE_GB = 1
    IF (INDXA(Comlyn,Comlen,'GEO') .GT. 0) MTYPE_GB = 2
    IF (PRNLEV .GE. 5) THEN
       IF (MTYPE_GB .EQ. 1) WRITE(OUTU,*)'Arithmetic Cross-Term'
       IF (MTYPE_GB .EQ. 2) WRITE(OUTU,*)'Geometric Cross-Term'
    ENDIF
    dn      = GtrmF(Comlyn,Comlen, 'DN',ONE)
    nphi_GB = GtrmI(Comlyn,Comlen, 'NPHI',26)
    CUTNUM  = GtrmI(Comlyn,Comlen, 'CUTNUM',maxtmp)
    CORR_GB = GtrmI(Comlyn,Comlen, 'CORR',1)
    A1      = GtrmF(Comlyn,Comlen, 'A1',ONE)
    A2      = GtrmF(Comlyn,Comlen, 'A2',ZERO)
    A3      = GtrmF(Comlyn,Comlen, 'A3',ZERO)

    call chmalloc('gbmvmodule.src','gbmv_set','adlistgb',NATOM,intg=adlistgb) !BD/MF

!BD/MF
    qexclgb = (INDXA(COMLYN,COMLEN,'EXCLGB').gt.0)
    if (qexclgb) then
        nmvsel = 0
        do i=1,NATOM
           adlistgb(i) = 0
        enddo
        nmvsel = 0
        call selcta(comlyn,comlen,islctgb,x,y,z,wmain,.true.)
        do i=1,natom
           if (islctgb(i) .eq. 1) then
              nmvsel = nmvsel + 1
              adlistgb(nmvsel) = i
           endif
        enddo
#if KEY_PARALLEL==1
      if (MYNOD .eq. 0) then
#endif
        write(outu,'(4X,A,1x,I5)') 'GBMV> exclude selection   ', nmvsel
#if KEY_PARALLEL==1
      endif
#endif
    else
      nmvsel = natom
    endif

    !-Hybrid: select atoms for GB radii calculation---------------
    ! Qhybrid: solute GB and explicit solvent

     Qhybrid  = ( INDXA(COMLYN,COMLEN,'HYBRID') > 0 )
     if (Qhybrid) then
        CALL SELCTA(COMLYN,COMLEN,GBMV_SEL,X,Y,Z,WMAIN,.TRUE.)
        nmvsel = 0
        qhbok = .true.
        do i=1,natom
            if (gbmv_sel(i) .eq. 1) then
                nmvsel = nmvsel + 1
                qhbok = (nmvsel == i) .and. qhbok
            endif
        enddo
        QGBASP = .false.
        QGBVDW = .false.
        if (nmvsel == natom .or. .not. qhbok) &
          call wrndie(-4,'<gbmv>', &
          'Problem with hybrid setup: not 1st segment or all atoms selected')
     else
        nmvsel = natom
     end if

    ! SJT/MF extra parameters for HDGB
    A4      = GtrmF(Comlyn,Comlen, 'A4',ZERO)
    A5      = GtrmF(Comlyn,Comlen, 'A5',ZERO)

    CutA    = GtrmF(Comlyn,Comlen, 'CUTA',TWENTY)

    ! MF new option to allow selection of radial integration
    !    grid size
    !
    ! GCUT 1  : original grid
    ! GCUT 2  : finer grid for small R
    ! GCUT 3  : custom grid (specify spacing explicitly
    !                        with RADG option)
    !
    ! default grid (GCUT=1) corresponds to:
    !
    ! RADG 23 0.1  0.2  0.3  0.4  0.5  0.75 1.0  1.25
    !         1.5  1.75 2.0  2.5  3.0  3.5  4.0  5.0
    !         6.0  7.0  8.0 10.0 12.0 16.0 20.0
    !
    ! finer grid (GCUT=2) corresponds to:
    !
    ! RADG 29 0.1  0.2  0.3  0.4  0.5  0.6  0.8  1.0
    !         1.2  1.4  1.6  1.8  2.0  2.2  2.4  2.6
    !         2.8  3.0  3.2  3.7  4.1  5.1  6.1  7.1
    !         8.1 10.1 12.1 16.1 20.1

    GCUT    = GtrmI(Comlyn,Comlen, 'GCUT', 1)

    IF (GCUT .EQ. 3) THEN
       TINX=INDXA(COMLYN,COMLEN,'RADG')
       IF (TINX .GT. 0) THEN
          CALL NXTWDA(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN,TINX)
          NRADG=DECODI(SWDTCH,SWDLEN)
          IF (NRADG.GT.MAXRADG) THEN
             CALL WRNDIE(1,'<gbmvmodule.src>Gbmv_set', &
                  'radial integration grid steps too large (increase MAXRADG)')
          ELSE
             DO I=1,NRADG
                CALL NXTWDA(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN,TINX)
                GCUTR(I)=DECODF(SWDTCH,SWDLEN)
             ENDDO
             DO I=NRADG+1,MAXRADG
                GCUTR(I)=CUTA
             ENDDO
          ENDIF
       ELSE
          CALL WRNDIE(1,'<gbmvmodule.src>Gbmv_SET', &
               'use RADG options to specify radial grid when GCUT=3')
       ENDIF
    ENDIF
    ! MF end of change

    Eps_GB  = GtrmF(Comlyn,Comlen, 'EPSILON',eighty)
    CUBIC_VSA_GB = GtrmI(Comlyn,Comlen,'CUBIC',0)
    Shift   = GtrmF(Comlyn,Comlen, 'SHIF',PT25)
    EShift  = GtrmF(Comlyn,Comlen, 'ESHI',ZERO)
    SLOPE   = GtrmF(Comlyn,Comlen, 'SLOP',ONE)
    rtemp=1.4D0
    WPROBE  = GtrmF(Comlyn,Comlen, 'WATR',rtemp)
    CutA    = GtrmI(Comlyn,Comlen, 'CUTA',20)
    WType   = GtrmI(Comlyn,Comlen, 'WTYP',2)
    Tol_GB  = GtrmF(Comlyn,Comlen, 'TOL',TENM8)
    TT      = GtrmF(Comlyn,Comlen, 'TT',MINONE)
    SA_GB   = GtrmF(Comlyn,Comlen, 'SA',ZERO)
    SB_GB   = GtrmF(Comlyn,Comlen, 'SB',ZERO)
    rtemp=1.2D0
    SON_GB  = GtrmF(Comlyn,Comlen, 'SON',rtemp)
    rtemp=1.5D0
    SOFF_GB = GtrmF(Comlyn,Comlen, 'SOFF',rtemp)
    KAPPA_GB = GtrmF(Comlyn,Comlen,'KAPPA',ZERO)
    P6      = GtrmF(Comlyn,Comlen, 'P6',FOUR)
    P7      = GtrmF(Comlyn,Comlen, 'P7',ZERO)
    IF (P6 .LE. ZERO) P6 = FOUR
    StillFac = P6
    P6      = ONE/P6

    NXP_GB   = GtrmI(Comlyn,Comlen, 'NX',0)
    NYP_GB   = GtrmI(Comlyn,Comlen, 'NY',0)
    NZP_GB   = GtrmI(Comlyn,Comlen, 'NZ',0)

    IF ((PRNLEV .GE. 5) .AND. (SA_GB .NE. ZERO)) THEN
       WRITE(OUTU,'(A)') ' SASA term requested with'
       WRITE(OUTU,'(A,F8.5,A,F8.5,A)') &
            ' SA =',SA_GB,' kcal/(mol*A**2) and SB =', &
            SB_GB,' kcal/mol.'
    ENDIF

    IF (TT .EQ. MINONE) TT = TWO * DSQRT(TWO)

    IF (CORR_GB .NE. 2) THEN
       ESHIFT = ESHIFT / (CCELEC*HALF*(ONE-ONE/EPS_GB))
    ENDIF

    IF (CutA .lt. 1.0D-5) CutA = TWENTY

    gbmvgrid_blk: IF (GBMVGrid) THEN
       IGenType = 21
       ML      = GtrmI(Comlyn,Comlen, 'ML',5000)
       SHOWGRID_GB = (INDXA(Comlyn,Comlen,'SHOW').gt.0)
       CONV_GB     = (INDXA(Comlyn,Comlen,'CONV').gt.0)
       IF (PRNLEV .GE. 5) THEN
          WRITE(OUTU,*)'Running grid-based GBMV: No forces available'
          WRITE(OUTU,'(A,I5)')' # surface pts. / atom = ',ML
       ENDIF
    ELSE gbmvgrid_blk
       rtemp=-20.0D0
       Beta =    GtrmF(Comlyn,Comlen, 'BETA',rtemp)
       rtemp=0.45D0
       P1      = GtrmF(Comlyn,Comlen, 'P1',rtemp)
       rtemp=1.25D0
       P2      = GtrmF(Comlyn,Comlen, 'P2',rtemp)
       rtemp=0.7D0
       P3      = GtrmF(Comlyn,Comlen, 'P3',rtemp)
       P4      = GtrmF(Comlyn,Comlen, 'P4',ZERO)
       rtemp=0.16D0
       P5      = GtrmF(Comlyn,Comlen, 'P5',rtemp)
       rtemp=-0.125D0
       HSX1    = GtrmF(Comlyn,Comlen, 'HSX1',rtemp)
       HSX2    = GtrmF(Comlyn,Comlen, 'HSX2',PT25)
       rtemp=1.9D0
       ON_X    = GtrmF(Comlyn,Comlen, 'ONX',rtemp)
       rtemp=2.1D0
       OFF_X   = GtrmF(Comlyn,Comlen, 'OFFX',rtemp)
       Lambda1 = GtrmF(Comlyn,Comlen, 'LAMBDA1',HALF)
       BufR    = GtrmF(Comlyn,Comlen, 'BUFR',HALF)
       EMP_GB  = GtrmF(Comlyn,Comlen, 'EMP',MINONE)
       MemMult = GtrmI(Comlyn,Comlen, 'MEM',10)
       ALFRQ   = GtrmI(Comlyn,Comlen, 'ALFRQ',1)
       DRFRQ   = GtrmI(Comlyn,Comlen, 'DRFRQ',1)

       MODSTILL = GtrmI(Comlyn,Comlen, 'MODSTILL',0)
       MS0      = GtrmF(Comlyn,Comlen, 'MS0',ONE)
       MS1      = GtrmF(Comlyn,Comlen, 'MS1',ZERO)
       MS2      = GtrmF(Comlyn,Comlen, 'MS2',ZERO)
       MS3      = GtrmF(Comlyn,Comlen, 'MS3',ZERO)
       ! SJT/MF
       QGBVDW   = (INDXA(Comlyn,Comlen, 'GBVD').gt.0)

       ! MF
       QGBASP   = (INDXA(Comlyn,Comlen, 'GBASP').gt.0)

       ! SJT/MF extra options for HDGB
       UNEPS_HDGB = GtrmI(Comlyn,Comlen, 'UNEPS', -1)
       UNNP_HDGB  = GtrmI(Comlyn,Comlen, 'UNNP',-1)
       ZS_HDGB    = GtrmF(Comlyn,Comlen, 'ZS', 0.5D0)
       ZM_HDGB    = GtrmF(Comlyn,Comlen, 'ZM', 9.2D0)
       ZT_HDGB    = GtrmF(Comlyn,Comlen, 'ZT', 25.D0)
       ST0_HDGB   = GtrmF(Comlyn,Comlen, 'ST0', 0.32D0)
       QEPSOUT    = (INDXA(Comlyn,Comlen,'EPSOUT').gt.0)
       QNPOUT    = (INDXA(Comlyn,Comlen,'NPOUT').gt.0)
       QHDGBRC    = (INDXA(Comlyn,Comlen,'HDGBRC').gt.0)
       QHDNOSW    = (INDXA(Comlyn,Comlen,'HDNOSW').gt.0)

#if KEY_HDGBVDW==1
! MS/MF extra options for vdw interactions in the membrane

         UNDP_HDGB = GtrmI(Comlyn,Comlen, 'UNDP', -1)
         UNDC_HDGB = GtrmI(Comlyn,Comlen, 'UNDC', -1)
         UNDC2_HDGB = GtrmI(Comlyn,Comlen, 'UND2C', -1)
         UNDO_HDGB = GtrmI(Comlyn,Comlen, 'UNDO', -1)
         UNDN_HDGB = GtrmI(Comlyn,Comlen, 'UNDN', -1)

         A_CT3= GtrmF(Comlyn,Comlen, 'A_CT3',0.25D0 )
         A_CY= GtrmF(Comlyn,Comlen, 'A_CY',0.20D0 )
         A_CPH= GtrmF(Comlyn,Comlen, 'A_CPH',0.40D0 )
         A_CC= GtrmF(Comlyn,Comlen, 'A_CC',0.40D0 )
         A_CA= GtrmF(Comlyn,Comlen, 'A_CA',0.20D0 )
         A_CT1= GtrmF(Comlyn,Comlen, 'A_CT1',0.55D0 )
         A_CT2= GtrmF(Comlyn,Comlen, 'A_CT2',0.45D0 )
         A_CM= GtrmF(Comlyn,Comlen, 'A_CM',0.70D0 )

         A_HS= GtrmF(Comlyn,Comlen, 'A_HS',0.50D0 )
         A_HO= GtrmF(Comlyn,Comlen, 'A_HO',0.75D0 )
         A_HY= GtrmF(Comlyn,Comlen, 'A_HY',0.20D0 )
         A_HM= GtrmF(Comlyn,Comlen, 'A_HM',0.675D0 )
         A_HP= GtrmF(Comlyn,Comlen, 'A_HP',0.15D0 )
         A_HR= GtrmF(Comlyn,Comlen, 'A_HR',0.475D0 )

!         WRITE (*,*) 'Double DEBUG' ,A_HO,A_HP,A_HM
         A_NH= GtrmF(Comlyn,Comlen, 'A_NH',0.225D0 )
         A_NY= GtrmF(Comlyn,Comlen, 'A_NY',0.15D0 )

         A_OH= GtrmF(Comlyn,Comlen, 'A_OH',0.75D0 )
         A_OY= GtrmF(Comlyn,Comlen, 'A_OY',0.725D0 )
         A_OR= GtrmF(Comlyn,Comlen, 'A_OR',0.425D0 )

         A_SM= GtrmF(Comlyn,Comlen, 'A_SM',0.175D0 )
         A_SC= GtrmF(Comlyn,Comlen, 'A_SC',0.425D0 )
         A_S=  GtrmF(Comlyn,Comlen, 'A_S',0.225D0 )
         A_P= GtrmF(Comlyn,Comlen, 'A_P',0.375D0 )

!DEBUG
!         WRITE (*,*) 'DEBUG'
!         WRITE (*,*)'A_CT3 ', A_CT3,'A_CY ',A_CY,'A_CPH ',A_CPH
!         WRITE (*,*) 'A_CC ',A_CC,'A_CA ',A_CA,'A_Ct1' ,A_CT1, &
!              'A_CT2' ,A_CT2,'A_CM ',A_CM,'A_HS ',A_HS, &
!              'A_HO ',A_HO,'A_HM ',A_HM,'A_HP ',A_HP,  &
!              'A_HR ',A_HR,'A_HY ',A_HY,'A_NH ',A_NH,'A_NY ',A_NY, &
!              'A_OH ',A_OH,'A_OR ',A_OR,'A_SM',A_SM,'A_SC',A_SC, &
!              'A_P ' , A_P

         QDPOUT    = (INDXA(Comlyn,Comlen,'DPOUT').gt.0)
         QDCOUT    = (INDXA(Comlyn,Comlen,'DCOUT').gt.0)
         QDC2OUT    = (INDXA(Comlyn,Comlen,'DC2OUT').gt.0)
         QDOOUT    = (INDXA(Comlyn,Comlen,'DOOUT').gt.0)
         QDNOUT    = (INDXA(Comlyn,Comlen,'DNOUT').gt.0)

!End MS/MF
#endif
#if KEY_DHDGB==1
       QFHDGB = (INDXA(Comlyn,Comlen,'QDHDGB').gt.0)
       IF (QFHDGB) THEN
           UN_FHDGB  = GtrmI(Comlyn,Comlen, 'UNFHDGB', -1)
           UNEPS_DEF_A = GtrmI(Comlyn,Comlen, 'UNEPS_DEF_A', -1)
           UNEPS_DEF_C = GtrmI(Comlyn,Comlen, 'UNEPS_DEF_C', -1)
           UNEPS_DEF_E = GtrmI(Comlyn,Comlen, 'UNEPS_DEF_E', -1)
           UNNP_DEF_A = GtrmI(Comlyn,Comlen, 'UNNP_DEF_A', -1)
           UNNP_DEF_C = GtrmI(Comlyn,Comlen, 'UNNP_DEF_C', -1)
           UNNP_DEF_E = GtrmI(Comlyn,Comlen, 'UNNP_DEF_E', -1)
           UN_PROVIDE = GtrmI(Comlyn,Comlen, 'UN_PROV', -1)
           EPS_DEF_B = GtrmF(Comlyn,Comlen,'EPS_DEF_B', 0.9145D0)
           NP_DEF_B = GtrmF(Comlyn,Comlen,'NP_DEF_B', 0.9145D0)
!          QSPLINE = (INDXA(Comlyn,Comlen,'SPLINE').gt.0)
           QDHDGBDEBUG = (INDXA(Comlyn,Comlen,'DHDGBDEB').gt.0)
           QWRTCENT = (INDXA(Comlyn,Comlen,'WRTCENT').gt.0)
           NPS_FHDGB = GtrmI(Comlyn,Comlen,'TOTPNTS', 11)
           NDT_FHDGB = GtrmI(Comlyn,Comlen,'ANGDIM', 5)
           MIR_FHDGB = GtrmF(Comlyn,Comlen,'MINR0', 2.50D0)
           MAR_FHDGB = GtrmF(Comlyn,Comlen,'MAXR0', 12.50D0)
           INC_FHDGB = GtrmF(Comlyn,Comlen,'INCR0', 2.5D0)
           INCTH_FHDGB = GtrmF(Comlyn,Comlen,'INCTH', 2.5D0)
           CIRCLERAD = GtrmF(Comlyn,Comlen,'CIRCLERAD', 7.5D0)
           SAMASS = GtrmF(Comlyn,Comlen,'SMASS',50000.D0)
           FRIC_DHDGB=GtrmF(Comlyn,Comlen,'DHDGBFR',0.0D0)
           SDEF(1)= GtrmF(Comlyn,Comlen,'S1',25.0D0)
           SDEF(2)= GtrmF(Comlyn,Comlen,'S2',25.0D0)
           SDEF(3)= GtrmF(Comlyn,Comlen,'S3',25.0D0)
           SDEF(4)= GtrmF(Comlyn,Comlen,'S4',25.0D0)
           SDEF(5)= GtrmF(Comlyn,Comlen,'S5',25.0D0)
           SDEF(6)= GtrmF(Comlyn,Comlen,'S6',25.0D0)
           SDEF(7)= GtrmF(Comlyn,Comlen,'S7',25.0D0)
           SDEF(8)= GtrmF(Comlyn,Comlen,'S8',25.0D0)
           SDEF(9)= GtrmF(Comlyn,Comlen,'S9',25.0D0)
           SDEF(10)= GtrmF(Comlyn,Comlen,'S10',25.0D0)
           CIRCLEMRAD = CIRCLERAD
           SEPS_C= GtrmF(Comlyn,Comlen,'SEPSC', -1.0639425D0)
           SNP_C= GtrmF(Comlyn,Comlen,'SINPC', -1.0639425D0)
           P_DHDGB = GtrmF(Comlyn,Comlen,'WFACT', 1.0D0)
           QWRITES=(INDXA(Comlyn,Comlen,'WRTS').gt.0)
           STARTATM=GtrmI(Comlyn,Comlen,'STARTATM', 0)
           ATMEND=GtrmI(Comlyn,Comlen,'ENDATM', 0)
           PENALTY_OFF=GtrmF(Comlyn,Comlen,'P_OFF', -6.0D0)
           PENALTY_FACT=GtrmF(Comlyn,Comlen,'P_FACT', 1.0D0)
           PENALTY_PREFACT=GtrmF(Comlyn,Comlen,'P_PREFFACT', 1.0D0)
           PENALTY_OFFL=GtrmF(Comlyn,Comlen,'P_OFFL', 1.0D0)
           PENALTY_FACTL=GtrmF(Comlyn,Comlen,'P_FACTL', 4.0D0)
           PENALTY_PREFACTL=GtrmF(Comlyn,Comlen,'P_PREFFACTL', 1.0D0)
           INNERC = GtrmF(Comlyn,Comlen,'INNERC',1.50D0)
           LAMF = GtrmF(Comlyn,Comlen,'LAMF',8.00D0)
           MAXEXP=GtrmF(Comlyn,Comlen,'MAXEXP', -1.50D0)
           IF (UN_PROVIDE .GT. -1) THEN
               QPROVIDE=.TRUE.
           ENDIF
       ENDIF
#endif
       ! MF extra options for optimized GBMV
       FAST_GB  = GtrmI(Comlyn,Comlen, 'FAST',0)
       SXDGB    = GtrmF(Comlyn,Comlen, 'SXD',ZERO)
       SGBFRQ   = GtrmI(Comlyn,Comlen, 'SGBFRQ',1)

       IF (INDXA(Comlyn,Comlen,'IMP').gt.0) THEN
          IMPULSE_GB = 1
          IF (PRNLEV.GE.5) THEN
             WRITE(OUTU,*) 'Using IMP method.'
          ENDIF
       ELSEIF (INDXA(Comlyn,Comlen,'LIMP').gt.0) THEN
          IMPULSE_GB = 2
          IF (PRNLEV.GE.5) THEN
             WRITE(OUTU,*) 'Using LIMP method.'
          ENDIF
       ELSEIF (INDXA(Comlyn,Comlen,'SIMP').gt.0) THEN
          IMPULSE_GB = 4
          IF (PRNLEV.GE.5) THEN
             WRITE(OUTU,*) 'Using SIMP method.'
          ENDIF
       ELSEIF (EMP_GB .GT. 1D-10) THEN
          IMPULSE_GB = 3
          IF (PRNLEV.GE.5) THEN
             WRITE(OUTU,'(A,F8.4)') &
                  ' Using EMP method with value of',EMP_GB
          ENDIF
       ELSE
          IMPULSE_GB = 5
          IF (PRNLEV.GE.5) THEN
             WRITE(OUTU,*) 'Constant force multiple time step'
          ENDIF
       ENDIF

       EMP_GB  = ONE/EMP_GB

       IF (BufR .lt. 1.0D-5) BufR = 0.5D0

       prnl_gt_5: IF (PRNLEV .GE. 5) THEN
          IF (P3 .EQ. ZERO) THEN
             WRITE(OUTU,*) &
                  'Running Analytical GBMV: Method I (Original)'
          ELSE
             WRITE(OUTU,*) &
                  'Running Analytical GBMV: Method II (VSA)'
          ENDIF

          ! SJT/MF VDW
          IF (QGBVDW) THEN
             WRITE(OUTU,*) 'VDW DISPERSION: ON'
          ENDIF
          ! SJT/MF VDW

          ! MF
          IF (QGBASP) THEN
             WRITE(OUTU,*) 'Variable ASP term ON.'
             WRITE(OUTU,*) 'Reading ASP coefficients from ASPValue'
          ENDIF
#if KEY_DHDGB==1
!AP/MF
          IF (QFHDGB) THEN
             if ((NDT_FHDGB .ne. 3) .and. &
                 (NDT_FHDGB .ne. 5)) then
                 CALL WrnDie(1,'<gbmvmodule.src>GBMV_Set', &
                                'DHDGB only works with ANGDIM 3 or 5')
                 QFHDGB=.false.
             endif
             if (NDT_FHDGB .eq. 3 ) then
                QSMALL=.TRUE.
             endif   
             if (CORR_GB .eq. 3) then
                if (QFHDGB) THEN
                 WRITE (OUTU,*)'MEMBRANE FLUCTUATION ON'
                endif
             else
                 CALL WrnDie(1,'<gbmvmodule.src>GBMV_Set', &
                                'DHDGB is only compatible with corrGB 3')
                 QFHDGB= .false.
             endif
         ENDIF
#endif
          ! SJT/MF
          ! HDGB2
          IF ((CORR_GB .eq. 3) .or. &
               (CORR_GB .eq. 4) .or. &
               (CORR_GB .eq. 5)) THEN
             ! HDGB2
             WRITE(OUTU,*)'with Heterogeneous Dielectric GB Mode'
             WRITE(OUTU,*)'(HDGB 2.0)'
             IF (QHDGBRC) THEN
                WRITE(OUTU,*) 'Radius Modulation: ON'
             ELSE
                WRITE(OUTU,*) 'Radius Modulation: OFF'
             END IF
             ! HDGB2
             IF (CORR_GB .eq. 3) THEN
                WRITE(OUTU,*) 'Membrane dielectric modulation: ON'
#if KEY_HDGBVDW==1
! MS/MF
!                  WRITE (OUTU,*) 'DEBUG-> checkpoint 1'
                  IF ( QGBVDW ) THEN
                     WRITE(OUTU,*) &
                      'VDW interactions in the membrane: ON'
                     WRITE(OUTU,*)   UNDN_HDGB, &
                              UNDC_HDGB, UNDC2_HDGB, UNDO_HDGB, &
                              UNDP_HDGB

                  ENDIF
#endif

             ELSE IF (CORR_GB .eq. 4) THEN
                WRITE(OUTU,*) 'Spherical dielectric modulation: ON'
             ELSE IF (CORR_GB .eq. 5) THEN
                WRITE(OUTU,*) 'Cylindrical dielectric modulation: ON'
             END IF
             IF (QHDNOSW) THEN
                WRITE(OUTU,*) 'SA modulation: OFF'
             ELSE
                WRITE(OUTU,*) 'SA modulation: ON'
             END IF
             ! HDGB2
          ENDIF
          IF (CORR_GB .eq. 0) THEN
             WRITE(OUTU,*)'CFA Correction factor: (1/R^5)^(1/2)'
             WRITE(OUTU,155) Shift,EShift,TT
          ELSEIF (CORR_GB .eq. 1) THEN
             WRITE(OUTU,*)'CFA Correction factor: (1/R^7)^(1/4)'
             WRITE(OUTU,157) Slope,Shift
          ELSEIF (CORR_GB .eq. 2) THEN
             WRITE(OUTU,*) &
                  'CFA Correction factor: (1/R^5)^(1/2)+(1/R^7)^(1/4)'
             WRITE(OUTU,158) A1,A2,A3,EShift,Shift,Slope
          ELSEIF ((CORR_GB .eq. 3) .or. &
               (CORR_GB .eq. 4) .or. &
               (CORR_GB .eq. 5)) THEN
             WRITE(OUTU,*) &
                  'CFA Correction factor: (1/R^5)^(1/2)+(1/R^7)^(1/4)'
             WRITE(OUTU,161) A1,A2,A3
             WRITE(OUTU,162) A4,A5,Slope
             WRITE(OUTU,163) UNEPS_HDGB, UNNP_HDGB, &
                  ZS_HDGB, ZM_HDGB, ZT_HDGB, ST0_HDGB
          ENDIF

          write(outu,140)p1,p2,p3,p4
          write(outu,145)p5,one/p6
          write(outu,150)lambda1,beta
          write(outu,160)cuta,bufr,dn,memmult
          write(outu,170)alfrq,one/emp_gb,kappa_gb
          write(outu,110)wprobe,eps_gb
          WRITE(OUTU,'(A,F6.3,A,F6.3)') &
               ' Hard sphere function from ',HSX1,' to ',HSX2
          WRITE(OUTU,'(A,F6.3,A,F6.3)') &
               ' Tail function switched from ',On_X,' to ',Off_X
          ! MF
          write(outu,'(a,i2)') ' GCUT = ', GCUT
          if(modstill.eq.1) then
             write(outu,180)ms0,ms1,ms2,ms3
          endif
       endif  prnl_gt_5
       igentype = 20
    endif gbmvgrid_blk

    if (gbmvgrid) then
       maxweights = 50000
    else
       maxweights = 5000
    endif

140 FORMAT(' P1     = ',F8.5,' P2    = ',F8.5, &
         ' P3     = ',F8.5,' P4    = ',F8.5)
145 FORMAT(' P5     = ',F8.5,' P6    = ',F8.5)
150 FORMAT(' Lambda = ',F8.5,' Beta  = ',F8.2)
155 FORMAT(' Shift  = ',F8.5,' Eshift= ',F8.5, &
         ' TT     = ',F8.5)
157 FORMAT(' Slope  = ',F8.5,' Shift = ',F8.5)
158 FORMAT(' A1     = ',F8.5,' A2    = ',F8.5, &
         ' A3     = ',F8.5,' Eshift= ',F8.5, &
         ' Shift  = ',F8.5,' Slope = ',F8.5)
160 FORMAT(' CutA   = ',F8.5,' Bufr  = ',F8.5, &
         ' DN     = ',F8.5,' Mem   = ',I8)
    ! SJT/MF
161 FORMAT(' A1     = ',F8.5,' A2    = ',F8.5, &
         ' A3     = ',F8.5)
162 FORMAT(' A4     = ',F8.5,' A5    = ',F8.5, &
         ' Slope  = ',F8.5)
163 FORMAT(' UNEPS  = ',  I8,' UNNP  = ', I8/, &
         ' ZS     = ',F8.5,' ZM    = ',F8.5, &
         ' ZT     = ',F8.5,' ST0   = ',F8.5)
#if KEY_HDGBVDW==1
!! MS/MF
165  FORMAT(' UNDN  = ', I8/, &
            ' UNDC  = ',  I8,' UNDC2  = ', I8/, &
            ' UNDO  = ',  I8,' UNDP  = ', I8)
#endif

164 FORMAT(' HSPL   = ',F8.5,' ZMIN  = ',F8.5, &
         ' ZMAX   = ',F8.5/, &
         ' MIN    = ',F8.5,' MAX   = ',F8.5)
194  FORMAT(' HSPL   = ',F8.5,' ZMIN  = ',F8.5, &
         ' ZMAX   = ',F8.5/,                    &
         ' ZERO   = ',F8.5,' FINAL = ',F8.5,    &
         ' MIN    = ',F8.5,' MAX   = ',F8.5)

110 FORMAT(' WProbe = ',F8.5,' EPS_GB= ',F8.5)
170 FORMAT(' ALFRQ  = ',I8,  ' EMP   = ',F8.5, &
         ' KAPPA  = ',F8.5)
180 FORMAT(' MS0    = ',F8.5,' MS1   = ',F8.5, &
         ' MS2    = ',F8.5,' MS3   = ',F8.5)

    dn_inv = ONE/dn
    dndiag = DSQRT(THREE)*dn/TWO
    NGBAtom = nmvsel
    gbmv_blk: IF (.not. QGBMV) THEN

       ! Allocate arrays for spherical weights

       call chmalloc('gbmvmodule.src','Gbmv_Set','wtx',maxweights,crl=wtx)
       call chmalloc('gbmvmodule.src','Gbmv_Set','wty',maxweights,crl=wty)
       call chmalloc('gbmvmodule.src','Gbmv_Set','wtz',maxweights,crl=wtz)

       call chmalloc('gbmvmodule.src','Gbmv_Set','wt1',maxweights,crl=wt1)
       call chmalloc('gbmvmodule.src','Gbmv_Set','wt2',maxweights,crl=wt2)
       call chmalloc('gbmvmodule.src','Gbmv_Set','wt4',maxweights,crl=wt4)
       call chmalloc('gbmvmodule.src','Gbmv_Set','wtr',maxweights,crl=wtr)
       call chmalloc('gbmvmodule.src','Gbmv_Set','wts',maxweights,crl=wts)
       call chmalloc('gbmvmodule.src','Gbmv_Set','xcp',nmvsel,crl=xcp)
       call chmalloc('gbmvmodule.src','Gbmv_Set','ycp',nmvsel,crl=ycp)
       call chmalloc('gbmvmodule.src','Gbmv_Set','zcp',nmvsel,crl=zcp)

       xcp = zero
       ycp = zero
       zcp = zero

       ! Get spherical integration weights

       CALL BuildGBWeight(WType)

       ! Allocate some atom-based arrays

       !-------- genborn module ---------------
       call chmalloc('gbmvmodule.src','Gbmv_Set','alph_gb',nmvsel,crl=alph_gb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','rad',nmvsel,crl=rad)
       ! SJT/MF VDW
       call chmalloc('gbmvmodule.src','Gbmv_Set','tvdw',nmvsel,crl=tvdw)
       call chmalloc('gbmvmodule.src','Gbmv_Set','avdw',nmvsel,crl=avdw)
       call chmalloc('gbmvmodule.src','Gbmv_Set','gmvdw',nmvsel,crl=gmvdw)
       call chmalloc('gbmvmodule.src','Gbmv_Set','gmasp',nmvsel,crl=gmasp)
       ! SJT/MF VDW

#if KEY_HDGBVDW==1
       call chmalloc('gbmvmodule.src','Gbmv_Set','dz_gbvdw',nmvsel,crl=dz_gbvdw)
       call chmalloc('gbmvmodule.src','Gbmv_Set','pvdw',nmvsel,crl=pvdw)                                                                               
       call chmalloc('gbmvmodule.src','Gbmv_Set','dp',nmvsel,crl=dp)
       call chmalloc('gbmvmodule.src','Gbmv_Set','ddd_hdgb',nmvsel,crl=ddd_hdgb)
#endif
       call chmalloc('gbmvmodule.src','Gbmv_Set','dx_gb',nmvsel,crl=dx_gb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','dy_gb',nmvsel,crl=dy_gb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','dz_gb',nmvsel,crl=dz_gb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','AX_GB',gbnum,crl=AX_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','AY_GB',gbnum,crl=AY_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','AZ_GB',gbnum,crl=AZ_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','BX_GB',gbnum,crl=BX_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','BY_GB',gbnum,crl=BY_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','BZ_GB',gbnum,crl=BZ_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','CX_GB',gbnum,crl=CX_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','CY_GB',gbnum,crl=CY_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','CZ_GB',gbnum,crl=CZ_GB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','VX1',gbnum,crl=VX1)
       call chmalloc('gbmvmodule.src','Gbmv_Set','VY1',gbnum,crl=VY1)
       call chmalloc('gbmvmodule.src','Gbmv_Set','VZ1',gbnum,crl=VZ1)
       call chmalloc('gbmvmodule.src','Gbmv_Set','RA',gbnum,crl=RA)
       call chmalloc('gbmvmodule.src','Gbmv_Set','RB',gbnum,crl=RB)
       call chmalloc('gbmvmodule.src','Gbmv_Set','R7D',gbnum,crl=R7D)
       call chmalloc('gbmvmodule.src','Gbmv_Set','temp1_gb',gbnum,crl=temp1_gb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','temp2_gb',gbnum,crl=temp2_gb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','GCUTR',MAXRADG,crl=GCUTR)

       ! SJT/MF
       call chmalloc('gbmvmodule.src','Gbmv_Set','r_hdgb',nmvsel,crl=r_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','dr_hdgb',nmvsel,crl=dr_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','rdist_hdgb',nmvsel,crl=rdist_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','eps_hdgb',nmvsel,crl=eps_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','deps_hdgb',nmvsel,crl=deps_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','a3_hdgb',nmvsel,crl=a3_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','shift_hdgb',nmvsel,crl=shift_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','p_hdgb',nmvsel,crl=p_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','s_hdgb',nmvsel,crl=s_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','surft_hdgb',nmvsel,crl=surft_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','dsurft_hdgb',nmvsel,crl=dsurft_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','atsa_hdgb',nmvsel,crl=atsa_hdgb)

       call chmalloc('gbmvmodule.src','Gbmv_Set','ra_hdgb',gbnum,crl=ra_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','r7d_hdgb',gbnum,crl=r7d_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','az_hdgb',gbnum,crl=az_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','cz_hdgb',gbnum,crl=cz_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','bx_hdgb',gbnum,crl=bx_hdgb)
       ! SJT/MF

#if KEY_HDGBVDW==1
! MS/MF
       call chmalloc('gbmvmodule.src','Gbmv_Set','dn_hdgb',nmvsel,crl=dn_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','ddn_hdgb',nmvsel,crl=ddn_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','dc2_hdgb',nmvsel,crl=dc2_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','ddc2_hdgb',nmvsel,crl=ddc2_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','dc_hdgb',nmvsel,crl=dc_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','ddc_hdgb',nmvsel,crl=ddc_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','do_hdgb',nmvsel,crl=do_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','ddo_hdgb',nmvsel,crl=ddo_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','dp_hdgb',nmvsel,crl=dp_hdgb)
       call chmalloc('gbmvmodule.src','Gbmv_Set','ddp_hdgb',nmvsel,crl=ddp_hdgb)


! MS/MF
#endif



       ! Allocate Grids and Tables

       ! SJT/MF
       CALL SetRadiusHDGB()

       ! in case of no HDGB, do the following
!       IF (.not. QWeight) THEN
!          DO i=1,nmvsel
!             WMAIN(i) = VDWR(ITC(IAC(I)))
!          ENDDO
!       ELSE
!          IF (PRNLEV .GE. 5)  &
!               WRITE(OUTU,*) ' Using Radii from WMAIN array'
!       ENDIF
       ! SJT/MF

       CALL FindExtents(1)
       IF (PRNLEV .GE. 5) WRITE(OUTU,200) NX,NY,NZ


       grid_blk: IF (.not. GBMVGrid) THEN
          !----- genborn module -------------------
          call chmalloc('gbmvmodule.src','Gbmv_Set','t',nmvsel,crl=t)
          call chmalloc('gbmvmodule.src','Gbmv_Set','gamma',nmvsel,crl=gamma)
          call chmalloc('gbmvmodule.src','Gbmv_Set','f',nmvsel,crl=f)
          call chmalloc('gbmvmodule.src','Gbmv_Set','g',nmvsel,crl=g)

          ! MF allocate extra storage, pre-calculate some quantities
          IF (FAST_GB.GT.0) THEN
             call chmalloc('gbmvmodule.src','Gbmv_Set','rtmp',9*nmvsel,crl=rtmp)
             call chmalloc('gbmvmodule.src','Gbmv_Set','vtmp',11*nmvsel,crl=vtmp)
          ENDIF

          IF (PRNLEV .GE. 6) WRITE(OUTU,210) NGridGB, MaxTable
          r1 = MAXTABLE
          r1 = r1 / 200000
          ! more than 50% of probed volume is surface??
          MaxSurf = NWeights * nmvsel
#if KEY_PARALLEL==1
          ! surface will be split amongst processors
          !
          ! MFC/IVK   bugfix 7
          ! MaxSurf = MaxSurf / NumNod
#endif 
          IF (PRNLEV .GE. 5) THEN
             WRITE(OUTU,125)' Table Size = ',r1,' MB'
             WRITE(OUTU,125)' Surf. Pts  = ',MaxSurf/1D6,' MB'
             WRITE(OUTU,125)' Grid Size  = ',NGridGB*8/1D6,' MB'
          ENDIF
125       FORMAT (A,F10.2,A)

          ! mfc/ivk  bugfix8 see bugfix 5
          call chmalloc('gbmvmodule.src','Gbmv_Set','surf_atom',maxsurf,iby=surf_atom)

          call chmalloc('gbmvmodule.src','Gbmv_Set','alist',maxtable,intg=alist)

          ! MF allocate extra temporary storage for fast GBMV
          IF (FAST_GB.GT.0) THEN
             call chmalloc('gbmvmodule.src','Gbmv_Set','surfx',maxsurf,iby=surfx)
             call chmalloc('gbmvmodule.src','Gbmv_Set','xindex',maxsurf*6,intg=xindex)
             call chmalloc('gbmvmodule.src','Gbmv_Set','blist',maxtable,intg=blist)
             call chmalloc('gbmvmodule.src','Gbmv_Set','clist',maxtable*4,intg=clist)
             call chmalloc('gbmvmodule.src','Gbmv_Set','dlist',maxtable,intg=dlist)
          ENDIF

          ! mfc/ivk  bugfix9 see bugfix 6
          call chmalloc('gbmvmodule.src','Gbmv_Set','value',maxtable,iby=value)

          call chmalloc('gbmvmodule.src','Gbmv_Set','index',ngridgb,intg=index)
          call chmalloc('gbmvmodule.src','Gbmv_Set','ptr',ngridgb,intg=ptr)

          GBStepCtr = 0
          DRSTEP = 0          ! Step to update derivative points
       ELSE grid_blk  ! .not.gbmvgrid above, gbmvgrid below
          IF (PRNLEV .GE. 5) WRITE(OUTU,220) NGridGB
          call chmalloc('gbmvmodule.src','Gbmv_Set','grid',ngridgb,iby=grid)
          IF (CONV_GB) THEN
             call chmalloc('gbmvmodule.src','Gbmv_Set','grid2',ngridgb,iby=grid2)
          ELSE
             call chmalloc('gbmvmodule.src','Gbmv_Set','grid2',100,iby=grid2)
          ENDIF

          i = 1.1D0 * (FOUR/THREE) * PI * (WPROBE/DN)**3 ! # estimated probe points
          IF (PRNLEV .GE. 5) &
               WRITE(OUTU,'(a,I10)') &
               ' 110% of # estimated probe points = ',i

          call chmalloc('gbmvmodule.src','Gbmv_Set','d',i,intg=d)
          call chmalloc('gbmvmodule.src','Gbmv_Set','pox',ml,crl=pox)
          call chmalloc('gbmvmodule.src','Gbmv_Set','poy',ml,crl=poy)
          call chmalloc('gbmvmodule.src','Gbmv_Set','poz',ml,crl=poz)

       ENDIF grid_blk

200    FORMAT(' Grid Extents: ',I4,' X ',I4,' X ',I4)
210    FORMAT(' Grid size: ',I12,'   Table Size = ',I12)
220    FORMAT(' Grid size: ',I12)

       QGBMV = .true.
       GBGridReset = .true.

       ! SJT/MF VDW
       IF (QGBVDW) THEN
          CALL SetVDWA()
          CALL SetVDWG(SA_GB)
       ENDIF
       ! SJT/MF VDW

       ! MF
#if KEY_ASPENER==1
       IF (QGBASP) THEN
          DO I=1,nmvsel
             GMASP(I)=ASPV(I)
          ENDDO
       ELSE   !!YMC/MF
          DO I=1,nmvsel
             GMASP(I)=SA_GB
          ENDDO
       ENDIF
#else /**/
       DO I=1,nmvsel
          GMASP(I)=SA_GB
       ENDDO
#endif 
   
       ! MF precalculate some quantities
       IF (FAST_GB.GT.0) THEN
          tR2C   =  ONE + DEXP(BETA * (TWO-LAMBDA1))
          tR2CI  =  ONE/tR2C
          tR2CIT =  tR2CI+TOL_GB
          tR20 = ONE + DEXP(BETA * (0.0D0-LAMBDA1))
          tR20I = ONE/tR20

          EXT = LOG(TOL_GB)/BETA
          THRESH1 = LAMBDA1 - EXT
          THRESH2 = LAMBDA1 + EXT
          IF (THRESH1 .LT. ZERO) THRESH1 = 1.0D-16

          tR2BUF = (BufR*0.99)**2

          CALL SetGBMVRad(WMAIN)

          CALL CalcTRad(WMAIN,nmvsel,9)
          SGBSTEP=0
       ENDIF

       ! SJT/MF Set up the dielectric profile
       IF ((CORR_GB .eq. 3) .or. &
            (CORR_GB .eq. 4) .or. &
            (CORR_GB .eq. 5)) THEN

          IF (UNEPS_HDGB .NE. -1) THEN
             ! Read a user-defined eps profile
#if KEY_PARALLEL==1
             IF (MYNOD .EQ. 0) THEN
#endif 
                ! Check at least if the file is opened.
                CALL VINQRE('UNIT',NAME_HDGB,MAXLEN_HDGB,LENGTH_HDGB, &
                     QOPEN_HDGB,QFORM_HDGB,QWRITE_HDGB, &
                     UNEPS_HDGB)
                IF (.NOT. QOPEN_HDGB) THEN
                   CALL WRNDIE(0,'<SetEpsProfile>','UNIT NOT OPEN')
                END IF

                READ(UNEPS_HDGB,*) N_HDGB, HSPL
                DO I = 1, N_HDGB
                   READ(UNEPS_HDGB,*) X_HDGB(I), A_HDGB(I)
                END DO
                CALL VCLOSE(UNEPS_HDGB,'KEEP',ERROR_HDGB)
#if KEY_PARALLEL==1
             END IF

             CALL PSND4(N_HDGB, 1)
             CALL PSND8(HSPL, 1)
             CALL PSND8(X_HDGB, 300)
             CALL PSND8(A_HDGB, 300)
#endif 
          END IF

          CALL SetEpsProfile(UNEPS_HDGB,  &
               EPS0_HDGB, EPSW_HDGB,EPSMIN_HDGB,EPSMAX_HDGB, &
               N_HDGB, X_HDGB, A_HDGB)

#if KEY_PARALLEL==1
          IF (MYNOD .EQ. 0) THEN
#endif 
             IF (PRNLEV .GE. 5) THEN
                WRITE(OUTU,*) 'EPS Profile Setup'
                WRITE(OUTU,194) HSPL,  &
                     ZMINSPL, ZMAXSPL, &
                     EPS0_HDGB, EPSW_HDGB, EPSMIN_HDGB, EPSMAX_HDGB
             END IF

             IF (QEPSOUT) THEN
                WRITE(OUTU,*) 'HDGB: EPS PROFILE:'
                IX1_HDGB = Int((ZMAXSPL +5.D0)/HSPL)
                DO I = -IX1_HDGB, IX1_HDGB
                   ZI = HSPL * I
                   CALL CalEpsHDGB(EPSI, DEPSI, ZI,  &
                        EPS0_HDGB, EPSW_HDGB,EPSMIN_HDGB,EPSMAX_HDGB,I)
                   WRITE(OUTU,'(3(f25.16,1x))') ZI, EPSI, DEPSI
                END DO
             END IF
#if KEY_PARALLEL==1
          END IF
#endif 

          IF (UNNP_HDGB .NE. -1) THEN
             ! Read a user-defined hydrophobic profile
#if KEY_PARALLEL==1
             IF (MYNOD .EQ. 0) THEN
#endif 
                ! Check at least if the file is opened.
                CALL VINQRE('UNIT',NAME_HDGB,MAXLEN_HDGB,LENGTH_HDGB, &
                     QOPEN_HDGB,QFORM_HDGB,QWRITE_HDGB, &
                     UNNP_HDGB)
                IF (.NOT. QOPEN_HDGB) THEN
                   CALL WRNDIE(0,'<SetNPProfile>','UNIT NOT OPEN')
                END IF

                READ(UNNP_HDGB,*) NNP_HDGB, HSPLNP
                DO I = 1, NNP_HDGB
                   READ(UNNP_HDGB,*) XNP_HDGB(I), ANP_HDGB(I)
                END DO
                CALL VCLOSE(UNNP_HDGB,'KEEP',ERROR_HDGB)
#if KEY_PARALLEL==1
             END IF

             CALL PSND4(NNP_HDGB, 1)
             CALL PSND8(HSPLNP, 1)
             CALL PSND8(XNP_HDGB, 300)
             CALL PSND8(ANP_HDGB, 300)
#endif 
          ELSE
             HSPLNP=0.15D0
             NNP_HDGB=250
             DO I = 1, NNP_HDGB
                XNP_HDGB(I)=DBLE(I-1)*HSPLNP
                CALL CalSurfT_HDGB(ANP_HDGB(I), DST, &
                     XNP_HDGB(I), ONE, QHDNOSW)
             END DO
          END IF
          CALL SetNPProfile(NP0_HDGB,NPW_HDGB,NPMIN_HDGB,NNP_HDGB,  &
               XNP_HDGB, ANP_HDGB)


#if KEY_PARALLEL==1
          IF (MYNOD .EQ. 0) THEN
#endif 
             IF (PRNLEV .GE. 5) THEN
                WRITE(OUTU,*) 'Non-polar Profile Setup'
                WRITE(OUTU,164) HSPLNP,  &
                     ZMINSPLNP, ZMAXSPLNP, &
                     NPMIN_HDGB,NPW_HDGB


             END IF

             IF (QNPOUT) THEN
                WRITE(OUTU,*) 'HDGB: NON-POLAR PROFILE:'
                IX1_HDGB = Int((ZMAXSPLNP +5.D0)/HSPLNP)
                DO I = -IX1_HDGB, IX1_HDGB
                   ZI = HSPLNP * I
                   CALL CalSurfTspl_HDGB(ST, DST, ZI, &
                        ZERO, ONE, ZERO, ONE,I)
                   WRITE(OUTU,'(3(f25.16,1x))') ZI, ST, DST
                END DO
             END IF
#if KEY_PARALLEL==1
          END IF
#endif
#if KEY_DHDGB==1
!AP/MF
! opens the file containing the cubic SPline coefficients of the deforemation energy look up table
! please note that if you use tables for NDT_FHDGB=5 (qsmall false) the coefficients will be read form the file throughout the simulations
! if you use NDT_FHDGB=3 (qsmall true) the coeficients will be read here and saved
       IF (QFHDGB) THEN
          IF (UN_FHDGB > -1) THEN
            IF (.NOT. QSMALL) THEN
             ! Check at least if the file is opened.
                CALL VINQRE('UNIT',NAME_FHDGB,MAXLEN_HDGB, &
                  LENGTH_FHDGB, &
                  QOPEN_FHDGB,QFORM_FHDGB,QWRITE_FHDGB, &
                  UN_FHDGB)
                IF (.NOT. QOPEN_FHDGB) THEN
                 CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
                ELSE
                 WRITE(OUTU,*) 'UNIT OPEND'
                ENDIF
            ELSE
                LOOK_SIZE=(4*7)**3
! IF QSMALL THEN FOR 7 INTERVALS OF (0:5:35) S THERE ARE 4 SPLINE COEFFS (IT IS CUBIC SPLINE) AND WE HAVE 3 S VALUES FOR EACH LEAFLET.
              ! Check at least if the file is opened.
                CALL VINQRE('UNIT',NAME_FHDGB,MAXLEN_HDGB, &
                  LENGTH_FHDGB, &
                  QOPEN_FHDGB,QFORM_FHDGB,QWRITE_FHDGB, &
                  UN_FHDGB)
                IF (.NOT. QOPEN_FHDGB) THEN
                 CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
                ELSE
                 WRITE(OUTU,*) 'UNIT OPEND'
                 DO I=1,LOOK_SIZE
                    READ(UN_FHDGB,REC=I) LOOKUP_COEFF(I)   
                 ENDDO
                ENDIF
            ENDIF
          ENDIF
! IF THE CENTER OF CIRCLES ARE PROVIDED READ THEM
          IF (QPROVIDE) THEN
              WRITE(OUTU,*) 'PROVIDED CENTERS WILL BE USED'
              CALL VINQRE('UNIT',NAME_PROV,MAXLEN_HDGB, &
                          LENGTH_FHDGB, &
                          QOPEN_PROV,QFORM_PROV,QWRITE_PROV, &
                          UN_PROVIDE)
              IF (.NOT. QOPEN_PROV) THEN
                  CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
              ELSE
                  WRITE(OUTU,*) 'UNIT OPEND'
              ENDIF
                 READ(UN_PROVIDE,*) LENGTH
              DO I=1,LENGTH
                 READ(UN_PROVIDE,*) READCX(I),READCY(I)
              ENDDO
              CALL VCLOSE(UN_PROVIDE,'KEEP',ERROR_HDGB)
          ENDIF
!READ COEFF EPS_A
          CALL VINQRE('UNIT',NAME_EPS_DEF_A,MAXLEN_HDGB, &
                       LENGTH_EPS_A, &
                       QOPEN_EPSA,QFORM_EPSA,QWRITE_EPSA, &
                       UNEPS_DEF_A)
          IF (.NOT. QOPEN_EPSA) THEN
              CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
          ENDIF
          READ(UNEPS_DEF_A,*) NEPS_DEF_A, HSPL_EPS_DEF_A
          DO I = 1, NEPS_DEF_A
             READ(UNEPS_DEF_A,*) XEPS_A(I), AEPS_A(I)
          ENDDO
          CALL VCLOSE(UNEPS_DEF_A,'KEEP',ERROR_EPSA)
          CALL SET_UP_COEFF(XEPS_A,AEPS_A,NEPS_DEF_A, &
                            EPS_DEF_A_SPL, &
                            ZMIN_EPS_DEF_A,ZMAX_EPS_DEF_A, &
                            MIN_EPS_DEF_A,MAX_EPS_DEF_A)
!FINISH READING EPA_A
!READ COEFF EPS_C
         CALL VINQRE('UNIT',NAME_EPS_DEF_C,MAXLEN_HDGB, &
                       LENGTH_EPS_C, &
                       QOPEN_EPSC,QFORM_EPSC,QWRITE_EPSC, &
                       UNEPS_DEF_C)
         IF (.NOT. QOPEN_EPSC) THEN
              CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
         ENDIF
         READ(UNEPS_DEF_C,*) NEPS_DEF_C, HSPL_EPS_DEF_C
         DO I = 1, NEPS_DEF_C
            READ(UNEPS_DEF_C,*) XEPS_C(I), AEPS_C(I)
         ENDDO
         CALL VCLOSE(UNEPS_DEF_C,'KEEP',ERROR_EPSC)
         CALL SET_UP_COEFF(XEPS_C,AEPS_C,NEPS_DEF_C, &
                           EPS_DEF_C_SPL, &
                           ZMIN_EPS_DEF_C,ZMAX_EPS_DEF_C, &
                           MIN_EPS_DEF_C,MAX_EPS_DEF_C)
!READ COEFF EPS_E
        CALL VINQRE('UNIT',NAME_EPS_DEF_E,MAXLEN_HDGB, &
                     LENGTH_EPS_E, &
                     QOPEN_EPSE,QFORM_EPSE,QWRITE_EPSE, &
                     UNEPS_DEF_E)
        IF (.NOT. QOPEN_EPSE) THEN
             CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
        ENDIF
        READ(UNEPS_DEF_E,*) NEPS_DEF_E, HSPL_EPS_DEF_E
        DO I = 1, NEPS_DEF_E
           READ(UNEPS_DEF_E,*) XEPS_E(I), AEPS_E(I)
        ENDDO
        CALL VCLOSE(UNEPS_DEF_E,'KEEP',ERROR_EPSE)
        CALL SET_UP_COEFF(XEPS_E,AEPS_E,NEPS_DEF_E, &
                          EPS_DEF_E_SPL, &
                          ZMIN_EPS_DEF_E,ZMAX_EPS_DEF_E, &
                          MIN_EPS_DEF_E,MAX_EPS_DEF_E)
!FINISH READING EPS_E
!READ COEFF NP_A
        CALL VINQRE('UNIT',NAME_NP_DEF_A,MAXLEN_HDGB, &
                     LENGTH_NP_A, &
                     QOPEN_NPA,QFORM_NPA,QWRITE_NPA, &
                     UNNP_DEF_A)
        IF (.NOT. QOPEN_NPA) THEN
             CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
        ENDIF
        READ(UNNP_DEF_A,*) NNP_DEF_A, HSPL_NP_DEF_A
        DO I = 1, NNP_DEF_A
           READ(UNNP_DEF_A,*) XNP_A(I), ANP_A(I)
        ENDDO
        CALL VCLOSE(UNNP_DEF_A,'KEEP',ERROR_NPA)
        CALL SET_UP_COEFF(XNP_A,ANP_A,NNP_DEF_A, &
                          NP_DEF_A_SPL, &
                          ZMIN_NP_DEF_A,ZMAX_NP_DEF_A, &
                          MIN_NP_DEF_A,MAX_NP_DEF_A)
!FINISH READING NP_A
!READ COEFF NP_C
        CALL VINQRE('UNIT',NAME_NP_DEF_C,MAXLEN_HDGB, &
                     LENGTH_NP_C, &
                     QOPEN_NPC,QFORM_NPC,QWRITE_NPC, &
                     UNNP_DEF_C)
        IF (.NOT. QOPEN_EPSC) THEN
             CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
        ENDIF
        READ(UNNP_DEF_C,*) NNP_DEF_C, HSPL_NP_DEF_C
        DO I = 1, NNP_DEF_C
           READ(UNNP_DEF_C,*) XNP_C(I), ANP_C(I)
        ENDDO
        CALL VCLOSE(UNNP_DEF_C,'KEEP',ERROR_NPC)
        CALL SET_UP_COEFF(XNP_C,ANP_C,NNP_DEF_C, &
                          NP_DEF_C_SPL, &
                          ZMIN_NP_DEF_C,ZMAX_NP_DEF_C, &
                          MIN_NP_DEF_C,MAX_NP_DEF_C)
!FINISH READING NP_C
!READ COEFF NP_E
        CALL VINQRE('UNIT',NAME_NP_DEF_E,MAXLEN_HDGB, &
                     LENGTH_NP_E, &
                     QOPEN_NPE,QFORM_NPE,QWRITE_NPE,&
                     UNNP_DEF_E)
        IF (.NOT. QOPEN_EPSE) THEN
             CALL WRNDIE(0,'<SETMEMFLUCT>','UNIT NOT OPEN')
        ENDIF
        READ(UNNP_DEF_E,*) NNP_DEF_E, HSPL_NP_DEF_E
        DO I = 1, NNP_DEF_E
           READ(UNNP_DEF_E,*) XNP_E(I), ANP_E(I)
        ENDDO
        CALL VCLOSE(UNNP_DEF_E,'KEEP',ERROR_NPE)
        CALL SET_UP_COEFF(XNP_E,ANP_E,NNP_DEF_E, &
                          NP_DEF_E_SPL, &
                          ZMIN_NP_DEF_E,ZMAX_NP_DEF_E, &
                          MIN_NP_DEF_E,MAX_NP_DEF_E)
!FINISH READING NP_E
!FINISH CHECKING
        DO I=1,NATOM
           DUM1(I)=0.0D0
           DUM2(I)=0.0D0
           DUM3(I)=0.0D0
           DUM4(I)=0.0D0
           DUM5(I)=0.0D0
        ENDDO
!1090 format(' INCRAD*', F8.5, ' DEF_ENER', F8.5)
! ENDIF TO IF QFHDGB
       ENDIF
#endif 
       END IF
#if KEY_HDGBVDW==1
!MS/MF set up the density profiles for vdw interactions in the np contribution
         IF ((CORR_GB .EQ. 3 ) .and. QGBVDW  ) THEN
!              WRITE(6,*) 'here we go'
!              Read a user-defined density profile
            IF (UNDC_HDGB .NE. -1) THEN 
              CALL SetVdw(UNDC_HDGB, &
                   QDCOUT,DC_HDGB,DDC_HDGB, &
                   DOC_HDGB,DWC_HDGB,DMINC_HDGB,DMAXC_HDGB, &
                   ZMIN_C,ZMAX_C)
              CSPLDC=CSPLD

!            WRITE (6,*) 'check DC'!, DC_HDGB !,DDC_HDGB
            ENDIF
           IF (UNDC2_HDGB .NE. -1) THEN
               CALL SetVdw(UNDC2_HDGB, &
                  QDC2OUT,DC2_HDGB,DDC2_HDGB, &
                  DOC2_HDGB,DWC2_HDGB,DMINC2_HDGB,DMAXC2_HDGB, &
                  ZMIN_C2,ZMAX_C2)
               CSPLDC2=CSPLD

!            WRITE (6,*) 'check DC2'!, DC2_HDGB !, DDC2_HDGB
           ENDIF
           IF (UNDP_HDGB .NE. -1) THEN
               CALL SetVdw(UNDP_HDGB, &
                           QDPOUT,DP_HDGB,DDP_HDGB, &
                       DOP_HDGB,DWP_HDGB,DMINP_HDGB,DMAXP_HDGB, &
                       ZMIN_P,ZMAX_P)
                CSPLDP=CSPLD
           ENDIF
!            WRITE (6,*) 'checkpoint 1 ', UNDO_HDGB
           IF (UNDO_HDGB .NE. -1) THEN
               CALL SetVdw(UNDO_HDGB, &
                           QDOOUT,DO_HDGB,DDO_HDGB, &
                       DOO_HDGB,DWO_HDGB,DMINO_HDGB,DMAXO_HDGB, &
                       ZMIN_O,ZMAX_O)
                CSPLDO=CSPLD
           ENDIF
!           WRITE (6,*) 'check UNDN_HDGB ',UNDN_HDGB
           IF (UNDN_HDGB .NE. -1) THEN
               CALL SetVdw(UNDN_HDGB, &
                           QDNOUT,DN_HDGB,DDN_HDGB, &
                       DON_HDGB,DWN_HDGB,DMINN_HDGB,DMAXN_HDGB, &
                       ZMIN_N,ZMAX_N)
                CSPLDN=CSPLD
!            WRITE (6,*) 'check DN'!, DN_HDGB !,DDN_HDGB

           ENDIF
         END IF

! END MS/MF
#endif




    ENDIF gbmv_blk

    RETURN
  END Subroutine Gbmv_Set


  !=======================================================================
  ! BUILDGBWEIGHTS
  !=======================================================================
  Subroutine BuildGBWeight(WType)
    !-------------------------------------------------------------
    ! This routine builds the spherical integration weights
    ! used in the molecular volume GB code
  use exfunc
  use number
  use consta

  use stream
  use dimens_fcm
  use psf
  use coord

    INTEGER i,j,k,IPHI,ITHETA,NPHIZ,NTHETAZ
    real(chm_real) dPHI, dTHETA, PHIZ, THETA,S3
    real(chm_real) R,R1,R2,R3,C1,C2,C4,SP,VX,VY,VZ,CP1,A,B
    real(chm_real) PTS1(3,110),WTS1(110),C5,C6
    INTEGER WType,NLEB,POS,IX,IY,IZ,LebOffSet
    real(chm_real) Ang(42)
    real(chm_real) ME,PE,MF,PF
    LOGICAL Rotate
    SAVE Ang,Wts1,Pts1
    PARAMETER(PE=1.61803398925D0,ME=-1.61803398925D0)
    PARAMETER(PF=0.61803398925D0,MF=-0.61803398925D0)

    S3 = DSQRT(THREE)

    DATA  (Ang(j),j=1,42) &
         / ONE,ZERO,ZERO,   ZERO,ONE,ZERO,   ZERO,ZERO,ONE, &
         MINONE,ZERO,ZERO,ZERO,MINONE,ZERO,ZERO,ZERO,MINONE, &
         ONE,ONE,ONE,     MINONE,ONE,ONE,  ONE,MINONE,ONE, &
         ONE,ONE,MINONE, MINONE,MINONE,ONE,MINONE,ONE,MINONE, &
         ONE,MINONE,MINONE,MINONE,MINONE,MINONE /

    ! Octahedron

    DATA ((Pts1(i,j),i=1,3),j=1,6) &
         /1.0000000000000000D0,0.0000000000000000D0,0.0000000000000000D0, &
         -1.0000000000000000D0,0.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,1.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,-1.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,0.0000000000000000D0,1.0000000000000000D0, &
         0.0000000000000000D0,0.0000000000000000D0,-1.0000000000000000D0/
    DATA (Wts1(j),j=1,6) &
         /0.1666666666666667D0,0.1666666666666667D0,0.1666666666666667D0, &
         0.1666666666666667D0,0.1666666666666667D0,0.1666666666666667D0/


    ! Lebedev grid, N = 26

    DATA ((Pts1(i,j),i=1,3),j=7,32) &
         /1.0000000000000000D0,0.0000000000000000D0,0.0000000000000000D0, &
         -1.0000000000000000D0,0.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,1.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,-1.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,0.0000000000000000D0,1.0000000000000000D0, &
         0.0000000000000000D0,0.0000000000000000D0,-1.0000000000000000D0, &
         0.7071067811865476D0,0.7071067811865476D0,0.0000000000000000D0, &
         0.7071067811865476D0,-0.7071067811865476D0,0.0000000000000000D0, &
         -0.7071067811865476D0,0.7071067811865476D0,0.0000000000000000D0, &
         -0.7071067811865476D0,-0.7071067811865476D0,0.0000000000000000D0, &
         0.7071067811865476D0,0.0000000000000000D0,0.7071067811865476D0, &
         0.7071067811865476D0,0.0000000000000000D0,-0.7071067811865476D0, &
         -0.7071067811865476D0,0.0000000000000000D0,0.7071067811865476D0, &
         -0.7071067811865476D0,0.0000000000000000D0,-0.7071067811865476D0, &
         0.0000000000000000D0,0.7071067811865476D0,0.7071067811865476D0, &
         0.0000000000000000D0,0.7071067811865476D0,-0.7071067811865476D0, &
         0.0000000000000000D0,-0.7071067811865476D0,0.7071067811865476D0, &
         0.0000000000000000D0,-0.7071067811865476D0,-0.7071067811865476D0, &
         0.5773502691896257D0,0.5773502691896257D0,0.5773502691896257D0, &
         0.5773502691896257D0,0.5773502691896257D0,-0.5773502691896257D0, &
         0.5773502691896257D0,-0.5773502691896257D0,0.5773502691896257D0, &
         0.5773502691896257D0,-0.5773502691896257D0,-0.5773502691896257D0, &
         -0.5773502691896257D0,0.5773502691896257D0,0.5773502691896257D0, &
         -0.5773502691896257D0,0.5773502691896257D0,-0.5773502691896257D0, &
         -0.5773502691896257D0,-0.5773502691896257D0,0.5773502691896257D0, &
         -0.5773502691896257D0,-0.5773502691896257D0,-0.5773502691896257D0/

    DATA (Wts1(j),j=7,32) &
         /0.0476190476190476D0,0.0476190476190476D0,0.0476190476190476D0, &
         0.0476190476190476D0,0.0476190476190476D0,0.0476190476190476D0, &
         0.0380952380952381D0,0.0380952380952381D0,0.0380952380952381D0, &
         0.0380952380952381D0,0.0380952380952381D0,0.0380952380952381D0, &
         0.0380952380952381D0,0.0380952380952381D0,0.0380952380952381D0, &
         0.0380952380952381D0,0.0380952380952381D0,0.0380952380952381D0, &
         0.0321428571428571D0,0.0321428571428571D0,0.0321428571428571D0, &
         0.0321428571428571D0,0.0321428571428571D0,0.0321428571428571D0, &
         0.0321428571428571D0,0.0321428571428571D0/

    ! Lebedev grid, N = 38

    DATA ((Pts1(i,j),i=1,3),j=33,70) &
         /1.0000000000000000D0,0.0000000000000000D0,0.0000000000000000D0, &
         -1.0000000000000000D0,0.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,1.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,-1.0000000000000000D0,0.0000000000000000D0, &
         0.0000000000000000D0,0.0000000000000000D0,1.0000000000000000D0, &
         0.0000000000000000D0,0.0000000000000000D0,-1.0000000000000000D0, &
         0.5773502691896257D0,0.5773502691896257D0,0.5773502691896257D0, &
         0.5773502691896257D0,0.5773502691896257D0,-0.5773502691896257D0, &
         0.5773502691896257D0,-0.5773502691896257D0,0.5773502691896257D0, &
         0.5773502691896257D0,-0.5773502691896257D0,-0.5773502691896257D0, &
         -0.5773502691896257D0,0.5773502691896257D0,0.5773502691896257D0, &
         -0.5773502691896257D0,0.5773502691896257D0,-0.5773502691896257D0, &
         -0.5773502691896257D0,-0.5773502691896257D0,0.5773502691896257D0, &
         -0.5773502691896257D0,-0.5773502691896257D0,-0.5773502691896257D0, &
         0.8880738339771153D0,0.4597008433809831D0,0.0000000000000000D0, &
         0.8880738339771153D0,-0.4597008433809831D0,0.0000000000000000D0, &
         -0.8880738339771153D0,0.4597008433809831D0,0.0000000000000000D0, &
         -0.8880738339771153D0,-0.4597008433809831D0,0.0000000000000000D0, &
         0.0000000000000000D0,0.8880738339771153D0,0.4597008433809831D0, &
         0.0000000000000000D0,0.8880738339771153D0,-0.4597008433809831D0, &
         0.0000000000000000D0,-0.8880738339771153D0,0.4597008433809831D0, &
         0.0000000000000000D0,-0.8880738339771153D0,-0.4597008433809831D0, &
         0.4597008433809831D0,0.0000000000000000D0,0.8880738339771153D0, &
         -0.4597008433809831D0,0.0000000000000000D0,0.8880738339771153D0, &
         0.4597008433809831D0,0.0000000000000000D0,-0.8880738339771153D0, &
         -0.4597008433809831D0,0.0000000000000000D0,-0.8880738339771153D0, &
         0.4597008433809831D0,0.8880738339771153D0,0.0000000000000000D0, &
         -0.4597008433809831D0,0.8880738339771153D0,0.0000000000000000D0, &
         0.4597008433809831D0,-0.8880738339771153D0,0.0000000000000000D0, &
         -0.4597008433809831D0,-0.8880738339771153D0,0.0000000000000000D0, &
         0.8880738339771153D0,0.0000000000000000D0,0.4597008433809831D0, &
         0.8880738339771153D0,0.0000000000000000D0,-0.4597008433809831D0, &
         -0.8880738339771153D0,0.0000000000000000D0,0.4597008433809831D0, &
         -0.8880738339771153D0,0.0000000000000000D0,-0.4597008433809831D0, &
         0.0000000000000000D0,0.4597008433809831D0,0.8880738339771153D0, &
         0.0000000000000000D0,-0.4597008433809831D0,0.8880738339771153D0, &
         0.0000000000000000D0,0.4597008433809831D0,-0.8880738339771153D0, &
         0.0000000000000000D0,-0.4597008433809831D0,-0.8880738339771153D0/

    DATA (Wts1(j),j=33,70) &
         /0.0095238095238095D0,0.0095238095238095D0,0.0095238095238095D0, &
         0.0095238095238095D0,0.0095238095238095D0,0.0095238095238095D0, &
         0.0321428571428571D0,0.0321428571428571D0,0.0321428571428571D0, &
         0.0321428571428571D0,0.0321428571428571D0,0.0321428571428571D0, &
         0.0321428571428571D0,0.0321428571428571D0,0.0285714285714286D0, &
         0.0285714285714286D0,0.0285714285714286D0,0.0285714285714286D0, &
         0.0285714285714286D0,0.0285714285714286D0,0.0285714285714286D0, &
         0.0285714285714286D0,0.0285714285714286D0,0.0285714285714286D0, &
         0.0285714285714286D0,0.0285714285714286D0,0.0285714285714286D0, &
         0.0285714285714286D0,0.0285714285714286D0,0.0285714285714286D0, &
         0.0285714285714286D0,0.0285714285714286D0,0.0285714285714286D0, &
         0.0285714285714286D0,0.0285714285714286D0,0.0285714285714286D0, &
         0.0285714285714286D0,0.0285714285714286D0/

    ! Icosahedron

    DATA ((Pts1(i,j),i=1,3),j=71,82) &
         /ZERO,ONE,PE,ZERO,MINONE,PE,ZERO,ONE,ME,ZERO,MINONE,ME, &
         PE,ZERO,ONE,PE,ZERO,MINONE,ME,ZERO,ONE,ME,ZERO,MINONE, &
         ONE,PE,ZERO,ONE,ME,ZERO,MINONE,PE,ZERO,MINONE,ME,ZERO/

    ! Dodecahedron

    DATA ((Pts1(i,j),i=1,3),j=83,102) &
         /ZERO,ME,PF,ZERO,PE,PF,MF,ZERO,PE,PF,ZERO,PE,ZERO,ME,MF, &
         ZERO,PE,MF,PF,ZERO,ME,MF,ZERO,ME,PE,MF,ZERO,PE,PF,ZERO, &
         ME,MF,ZERO,ME,PF,ZERO,MINONE,ONE,ONE,MINONE,MINONE,ONE, &
         MINONE,MINONE,MINONE,MINONE,ONE,MINONE,ONE,ONE,ONE, &
         ONE,MINONE,ONE,ONE,ONE,MINONE,ONE,MINONE,MINONE/

    DO I=6,13
       Ang(I*3+1)=Ang(I*3+1)/S3
       Ang(I*3+2)=Ang(I*3+2)/S3
       Ang(I*3+3)=Ang(I*3+3)/S3
    ENDDO

    Rotate = .TRUE.
    J = 0
    IF (WTYPE .GE. TEN) THEN

       R1 = ONE/(PT25)**(0.75D0)
       R2 = ONE/(TWENTY)**(0.75D0)
       R3 = (R1-R2)/15.0D0
       DO WHILE (R .LE. CUTA)
          R = ONE/(R1**(0.75D0))
          IF (R .GT. ZERO) THEN
             J = J + 1
             WTR(J) = R
             ! WRITE(OUTU,*)J,R
             R1 = R1 - R3
          ELSE
             R=CUTA+ONE
          ENDIF
       ENDDO
       NumR = J-1
       WTYPE = WTYPE - 10

    ELSE

       ! MF following lines rewritten to allow different radial
       !    integration grids
       !
       ! GCUT=1: original grid (default)
       ! GCUT=2: finer grid for small R
       ! GCUT=3: custom grid

       IF (GCUT .eq. 3) THEN
          DO WHILE (GCUTR(J).LT.CutA)
             J=J+1
             WTR(J)=GCUTR(J)
          ENDDO
       ELSE
          R = 0.1
          DO WHILE (R .LT. CutA)
             J = J + 1
             WTR(J) = R
             IF (GCUT .EQ. 1) THEN
                IF (R .LT. HALF) THEN
                   R = R + 0.1
                ELSEIF (R .LT. TWO) THEN
                   R = R + 0.25
                ELSEIF (R .LT. FOUR) THEN
                   R = R + 0.5
                ELSEIF (R .LT. EIGHT) THEN
                   R = R + 1.0
                ELSEIF (R .LT. TWELVE) THEN
                   R = R + 2.0
                ELSE
                   R = R + 4.0
                ENDIF
             ELSEIF (GCUT.EQ.2) THEN
                IF (R .LT. HALF+0.1) THEN
                   R = R + 0.1
                ELSEIF (R .LT. THREE+0.1) THEN
                   R = R + 0.2
                ELSEIF (R .LT. FOUR) THEN
                   R = R + 0.5
                ELSEIF (R .LT. EIGHT) THEN
                   R = R + 1.0
                ELSEIF (R .LT. TWELVE) THEN
                   R = R + 2.0
                ELSE
                   R = R + 4.0
                ENDIF
             ELSE
                Call WRNDIE (-1,'<BuildGBWeight>', &
                     '  Invalid GCUT value. (choose 1, 2, or 3)')
             ENDIF
          ENDDO
       ENDIF
       NumR = J-1
    ENDIF
    ! MF end of change


    IF (WTYPE .EQ. 0) THEN
       Rotate = .FALSE.
       IF (.not. GBMVGrid) THEN
          WTYPE=2
          NPHI_GB=20
          NumR = (CutA - 4) + FOUR/HALF - ONE
          R = 0.5
          DO J=1,NumR+1
             WTR(J) = R
             IF (R .lt. FOUR) THEN
                R = R + 0.5
             ELSE
                R = R + 1.0
             ENDIF
          ENDDO
       ELSE
          WTYPE=1
          NPHI_GB = 8
          NumR = (EIGHT*TWO)/DN + (CUTA - EIGHT) + 1.0D-12
          R = DN*0.5
          DO J=1,NumR+1
             WTR(J) = R
             IF (R .lt. EIGHT) THEN
                R = R + DN*0.5
             ELSE
                R = R + 1.0
             ENDIF
          ENDDO
       ENDIF
    ENDIF

    IF (WTYPE .EQ. 1) THEN    ! Spherical polar
       NPHIZ = NPHI_GB
       dPHI = PI / NPHIZ
       IF (PRNLEV .GE. 5) WRITE(OUTU,'(A,I3,A)') &
            ' Spherical-polar integration grid of order', &
            NPHIZ,' selected.'
    ELSEIF (WTYPE .EQ. 2) THEN ! Lebedev
       NLEB = NPHI_GB
       IF (NLEB .eq. 6) THEN
          LebOffSet = 1
          IF (PRNLEV .GE. 5) &
               WRITE(OUTU,'(A,I3,A)')' Lebedev integration grid size'
       ELSEIF (NLEB .eq. 26) THEN
          LebOffSet = 7
          IF (PRNLEV .GE. 5) &
               WRITE(OUTU,'(A,I3,A)')' Lebedev integration grid size 26'
       ELSEIF (NLEB .eq. 38) THEN
          LebOffSet = 33
          IF (PRNLEV .GE. 5) &
               WRITE(OUTU,'(A,I3,A)')' Lebedev integration grid size 38'
       ELSEIF (NLEB .eq. 12) THEN
          IF (PRNLEV .GE. 5) &
               WRITE(OUTU,*)'Icosahedron integration grid selected.'
          LebOffSet = 71
          C2 = ONE/DSQRT(((ONE+DSQRT(FIVE))/TWO)**2+ONE)
          DO I=71,82
             PTS1(1,I)=PTS1(1,I)*C2
             PTS1(2,I)=PTS1(2,I)*C2
             PTS1(3,I)=PTS1(3,I)*C2
             WTS1(I)=ONE/TWELVE
          ENDDO
       ELSEIF (NLEB .eq. 20) THEN
          IF (PRNLEV .GE. 5) &
               WRITE(OUTU,*)'Dodecahedron integration grid selected.'
          LebOffSet = 83
          C2=ONE/S3
          DO I=83,102
             PTS1(1,I)=PTS1(1,I)*C2
             PTS1(2,I)=PTS1(2,I)*C2
             PTS1(3,I)=PTS1(3,I)*C2
             WTS1(I)=ONE/TWENTY
          ENDDO
       ! CUDA-GBMV supports five types of Lebedev angular  integration
       ELSEIF (NLEB .eq. 50 .OR. NLEB .eq. 6 .OR. NLEB .eq. 14 .OR. NLEB .eq. 26 .OR. NLEB .eq. 38) THEN
          LebOffSet = 33
          IF (PRNLEV .GE. 5) &
               WRITE(OUTU,'(A,I3,A)')' Lebedev integration grid size for GPU calculations: ', NLEB
       ELSE
          Call WRNDIE (-1,'<BuildGBWeight>', &
               '  Lebedev grid of NPHI not supported.')
       ENDIF


    ELSE                      ! WTYPE = 3 ?

       IF (PRNLEV .GE. 5) &
            WRITE(OUTU,'(A)')' Alternating 6/8 grid selected.'

    ENDIF

    k = 1
    R3 = -ONE/(R)

    DO J=1,NumR
       R = WTR(J)
       R1 = WTR(J+1)
       R2 = HALF*(R+R1)
       C2 = ONE/R - ONE/R1
       C4 = HALF/(R*R) - HALF/(R1*R1)
       C6 = (ONE/(R*R*R*R) - ONE/(R1*R1*R1*R1))/FOUR
       IF (WType .EQ. ONE) THEN
          PHIZ = DPHI*HALF
          DO IPHI = 1,NPHIZ
             SP = dsin(PHIZ)
             VZ = dcos(PHIZ)

             NTHETAZ = (NPHIZ * 2 - 1) * DABS(SP) + 1.0
             dTHETA = 2.0 * PI / NTHETAZ
             C1 = dTHETA / (4.0 * PI)

             CP1 = dcos(PHIZ - dPHI*HALF) - dcos(PHIZ + dPHI * HALF)
             THETA = ZERO

             DO ITHETA = 1,NTHETAZ
                VX = dcos(THETA) * SP
                VY = dsin(THETA) * SP
                WT1(k) = c2 * c1 * cp1
                WT2(k) = c4 * c1 * cp1
                !                 WT3(k) = c5 * c1 * cp1
                WT4(k) = c6 * c1 * cp1
                WTX(k) = VX * R2
                WTY(k) = VY * R2
                WTZ(k) = VZ * R2
                WTS(k) = R
                R3 = R3 + c2*c1*cp1
                k = k + 1
                THETA = THETA + dTHETA
             ENDDO

             PHIZ = PHIZ + dPHI
          ENDDO
       ELSEIF (WTYPE .EQ. 2) THEN !Lebedev
          DO I = 0,NLEB-1
             WT1(k+I) = C2 * WTS1(I+LebOffSet)
             WT2(k+I) = C4 * WTS1(I+LebOffSet)
             WT4(k+I) = C6 * WTS1(I+LebOffSet)
             WTS(k+I) = R
             IF (Rotate) THEN
                IF (MOD(J,2).EQ.0) THEN
                   WTX(k+I) = PTS1(1,I+LebOffSet) * R2
                   WTY(k+I) = PTS1(2,I+LebOffSet) * R2
                   WTZ(k+I) = PTS1(3,I+LebOffSet) * R2
                ELSE
                   WTY(k+I) = -PTS1(1,I+LebOffSet) * R2
                   WTX(k+I) = PTS1(2,I+LebOffSet) * R2
                   WTZ(k+I) = PTS1(3,I+LebOffSet) * R2
                ENDIF
             ELSE
                WTX(k+I) = PTS1(1,I+LebOffSet) * R2
                WTY(k+I) = PTS1(2,I+LebOffSet) * R2
                WTZ(k+I) = PTS1(3,I+LebOffSet) * R2
             ENDIF
          ENDDO
          k = k + NLEB
       ELSEIF (WTYPE .EQ. 3) THEN !Alt. oct/cube
          IF (MOD(J,2) .EQ. 0) THEN ! Do oct
             DO I = 0,5
                WT1(k+I) = C2 / SIX
                WT2(k+I) = C4 / SIX
                WTS(k+I) = R
                WTX(k+I) = Ang(I*3+1) * R2
                WTY(k+I) = Ang(I*3+2) * R2
                WTZ(k+I) = Ang(I*3+3) * R2
             ENDDO
          ELSE                ! Do cube
             k = k + SIX
             DO I = 6,13
                WT1(k+I-6) = C2 / EIGHT
                WT2(k+I-6) = C4 / EIGHT
                WTS(k+I-6) = R
                WTX(k+I-6) = Ang(I*3+1) * R2
                WTY(k+I-6) = Ang(I*3+2) * R2
                WTZ(k+I-6) = Ang(I*3+3) * R2
             ENDDO
             k = k + EIGHT
          ENDIF
       ENDIF

       ! WRITE(OUTU,*) 'Sum = ',-ONE/R3
    ENDDO

    NWeights = k - 1

    NAngular = NWeights / NumR

#if KEY_GBMVDEBUG==1
    OPEN(1,file='weights.xyz',status='unknown')
    DO I=1,NWeights
       WRITE(1,100) WTX(I),WTY(I),WTZ(I)
    ENDDO
100 FORMAT ('H ',F10.3,1X,F10.3,1X,F10.3)
    CLOSE(1)
#endif 

    IF (PRNLEV .GE. 5) WRITE(OUTU,110) NWeights
110 FORMAT(' Number of integration points = ',I5)
    RETURN
  END subroutine BuildGBWeight

  Subroutine BuildGBLookup()
    !-------------------------------------------------------------
    !  Build two lookup tables:
    !  1) Grid of points which intersect core of an atom
    !  2) Grid of lists telling which atoms are in vicinity

  use exfunc
  use stream
  use dimens_fcm
  use param
  use psf
  use number
  use coord
  use parallel

    integer(int_byte) :: ipass
    real(chm_real) Sum, TempA(maxtmp)
    Integer i,j,k,PS,POS,nt,ii,jj,prev
    real(chm_real) R1,R2,R3,CX,CY,CZ,VX,VY,VZ, &
         XDIFF,YDIFF,ZDIFF,VR,dn1,RFac
    Integer AX,AY,AZ,IX,IY,IZ,AX1,AX2,AY1,AY2, &
         AZ1,AZ2
    INTEGER NodeStart,NodeEnd         ! MFC/IVK bugfix 1 nodeend

#if KEY_PARALLEL==1
    NodeStart = (MaxTable * (MyNod))  / NumNod + 1 ! MFC/ivk bugfix 2
    NodeEnd   = (MaxTable * (MyNodP)) / NumNod     ! MFC/ivk bugfix 2
    DO I=1,MaxTable
       AList(I) = 0
    ENDDO
#else /**/
    NodeStart = 1
    NodeEnd   = MaxTable
#endif 


    ! Build lookup grids

    ! WRITE(OUTU,*) 'Building lookup grid...'
    GBGridReset = .false.

    DO I=1,NGridGB
       INDEX(I) = 0
       PTR(I) = 0
    ENDDO

    IF (P3 .LE. ZERO) THEN
       RFac = (log(1.0D-8)/(log(Lambda1)*P1))**(PT25)
    ENDIF
    DO IPass = 1,2
       ! WRITE(OUTU,*) 'Pass #',IPass
       DO I=1,nmvsel
          CX = X(i) - XMIN
          CY = Y(i) - YMIN
          CZ = Z(i) - ZMIN
          AX = CX * dn_inv
          AY = CY * dn_inv
          AZ = CZ * dn_inv
          VR = WMAIN(i)
          IF (P3 .GT. ZERO) THEN
             R1 = (VR + Off_X) + dndiag + BufR
          ELSE
             R1 = RFac * VR + dndiag + BufR
          ENDIF
          K = R1 * dn_inv + ONE
          R1 = R1 * R1
          R3 = VR*VR
          AX1=AX-K
          AX2=AX+K
          AY1=AY-K
          AY2=AY+K
          AZ1=AZ-K
          AZ2=AZ+K
          IF (AX1.LT.0) AX1=0
          IF (AY1.LT.0) AY1=0
          IF (AZ1.LT.0) AZ1=0
          IF (AX2.GT.(NX-1)) AX2=NX-1
          IF (AY2.GT.(NY-1)) AY2=NY-1
          IF (AZ2.GT.(NZ-1)) AZ2=NZ-1

#if KEY_AVIMODS==1
           
             IZ = AZ1
             DO WHILE ((MOD(IZ,NumNod).EQ.(MyNodP-1)) .AND. (IZ .LE. AZ2))
                VZ = IZ * dn + HALF*dn
                ZDIFF = VZ*VZ + CZ*CZ -2*VZ*CZ

                DO IX = AX1,AX2
                   VX = IX * dn + HALF*dn
                   XDIFF = VX*VX + CX*CX - 2*VX*CX

                   DO IY = AY1,AY2
                      VY = IY * dn + HALF*dn
                      YDIFF = VY*VY + CY*CY - 2*VY*CY

                      R2 = XDIFF + YDIFF + ZDIFF
                      POS = IX + IY * NX + IZ * NXY - Offset
                      IF (R2 .LT. R1) THEN
                         IF (IPASS .EQ. 2) THEN
                            AList(Index(POS)+PTR(POS)) = I
                            !                          Value(Index(POS)+PTR(POS)) = 100*(r1-r2)/r1
                            Value(Index(POS)+PTR(POS)) = 100*(r2-r1)/r1
                            if(Index(POS)+PTR(POS) > maxtable ) then
                               CALL WrnDie(-1,'<BuildGBLookup>', &
                                    'v index > maxtable')
                            endif
                         ENDIF
                         PTR(POS) = PTR(POS) + 1
                      ENDIF
                   ENDDO ! iy
                ENDDO  ! ix
          IZ = IZ+1
          ENDDO    !iz = az1,az2

#else /**/

          DO IZ = AZ1,AZ2
#if KEY_PARALLEL==1
             IF (MOD(IZ,NumNod).EQ.(MyNodP-1)) THEN     
#endif
                VZ = IZ * dn + HALF*dn
                ZDIFF = VZ - CZ
                ZDIFF = ZDIFF*ZDIFF
                DO IX = AX1,AX2
                   VX = IX * dn + HALF*dn
                   XDIFF = VX - CX
                   XDIFF = XDIFF*XDIFF
                   DO IY = AY1,AY2
                      VY = IY * dn + HALF*dn
                      YDIFF = VY - CY
                      YDIFF = YDIFF*YDIFF
                      R2 = XDIFF + YDIFF + ZDIFF
                      POS = IX + IY * NX + IZ * NXY - Offset
                      IF (R2 .LT. R1) THEN
                         IF (IPASS .EQ. 2) THEN
                            AList(Index(POS)+PTR(POS)) = I
                            Value(Index(POS)+PTR(POS)) = 100*(r2-r1)/r1
                            if(Index(POS)+PTR(POS) > maxtable ) then
                               CALL WrnDie(-1,'<BuildGBLookup>', &
                                    'v index > maxtable')
                            endif
                         ENDIF
                         PTR(POS) = PTR(POS) + 1
                      ENDIF
                   ENDDO ! iy
                ENDDO  ! ix
#if KEY_PARALLEL==1
             ENDIF         
#endif
          ENDDO    !iz = az1,az2

#endif 

       ENDDO   ! i=1,natom
       IF (IPASS .EQ. 1) THEN
          prev = NodeStart

#if KEY_AVIMODS==1
          IZ = 0

            DO WHILE (MOD(IZ,NumNod).EQ.(MyNodP-1) .AND. (IZ .LT. NZ))
                DO IX = 0,NX-1
                   DO IY = 0,NY-1
                      POS = IX + IY * NX + IZ * NXY - Offset
                      INDEX(POS) = prev
                      !
                      ! MFC/IVK bugfix 3, mfc variation takes care of both checks
                      !                   in the published bugfix.
                      !--- Check if adding on the next ptr(pos) entries
                      !--- will overrun the allocated space.
                      IF (INDEX(POS)+ptr(pos)-1 .GT. NodeEnd)  &
                           CALL WrnDie(-1,'<BuildGBLookup>', &
                           'Lookup grid exceeded bounds. Increase MEM.')

                      prev = prev + PTR(POS)
                      PTR(POS) = 0
                   ENDDO         ! iy
                ENDDO            ! ix
          IZ = IZ+1
          ENDDO               ! iz

#else /**/

          DO IZ = 0,NZ-1
#if KEY_PARALLEL==1
             IF (MOD(IZ,NumNod).EQ.(MyNodP-1)) THEN  
#endif
                DO IX = 0,NX-1
                   DO IY = 0,NY-1
                      POS = IX + IY * NX + IZ * NXY - Offset
                      INDEX(POS) = prev
                      !
                      ! MFC/IVK bugfix 3, mfc variation takes care of both checks
                      !                   in the published bugfix.
                      !--- Check if adding on the next ptr(pos) entries
                      !--- will overrun the allocated space.
                      IF (INDEX(POS)+ptr(pos)-1 .GT. NodeEnd)  &
                           CALL WrnDie(-1,'<BuildGBLookup>', &
                           'Lookup grid exceeded bounds. Increase MEM.')

                      prev = prev + PTR(POS)
                      PTR(POS) = 0
                   ENDDO         ! iy
                ENDDO            ! ix
#if KEY_PARALLEL==1
             ENDIF            
#endif
          ENDDO               ! iz

#endif 

       ENDIF                  ! ipass == 1
    ENDDO                     !IPass loop

    NTable = prev - NodeStart + 1
    IF (PRNLEV .GE. 6) WRITE(OUTU,*)'Total tabulated atoms = ',NTable

    ! Sort lookup grid for adjacent atoms

    ! WRITE(OUTU,*) 'Sorting lookup table...'
    DO IZ = 0,NZ-1
#if KEY_PARALLEL==1
       IF (MOD(IZ,NumNod).EQ.(MyNodP-1)) THEN    
#endif
          DO IX = 0,NX-1
             DO IY = 0,NY-1
                POS = IX + IY * NX + IZ * NXY - Offset
                if (ptr(pos) > maxtmp) then
                   CALL WrnDie(-4,'<BuildGBLookup>', &
                        'ptr(pos) exceeds tempa array size maxtmp')
                else IF (ptr(pos) .NE. 0) THEN
                   JJ = INDEX(POS)
                   DO i=1,ptr(pos)
                      TEMPA(i) = VALUE(i+jj-1)
                   ENDDO
                   call qcksrt(ptr(pos),tempa,alist(jj))
                ENDIF
             ENDDO
          ENDDO
#if KEY_PARALLEL==1
       ENDIF                    
#endif
    ENDDO

#if KEY_PARALLEL==1
    DO POS = 1,MaxTable,512000
       IF (POS .gt. (MaxTable-512000)) THEN
          CALL GBOR(AList(POS),MaxTable-POS+1)
       ELSE
          CALL GBOR(AList(POS),512000)
       ENDIF
    ENDDO
    DO POS = 1,NGridGB,512000
       IF (POS .gt. (NGridGB-512000)) THEN
          CALL GBOR(Index(POS),NGridGB-POS+1)
          CALL GBOR(Ptr(POS),  NGridGB-POS+1)
       ELSE
          CALL GBOR(Index(POS),512000)
          CALL GBOR(Ptr(POS),  512000)
       ENDIF
    ENDDO
#endif 

    RETURN
  END Subroutine BuildGBLookup

  ! MF optimized routine invoked with FAST option
  ! parallel version is not recommended
  !
  Subroutine BuildGBLookupF()

    !-------------------------------------------------------------
    !  Build two lookup tables:
    !  1) Grid of points which intersect core of an atom
    !  2) Grid of lists telling which atoms are in vicinity
    !
    use exfunc
    use stream
    use dimens_fcm
    use param
    use psf
    use number
    use coord
    use parallel
    real(chm_real) Sum, TempA(maxtmp)
    Integer i,j,k,PS,POS,nt,ii,jj,prev,ipos
    real(chm_real) R1,R2,R3,CX,CY,CZ,VX,VY,VZ, &
         XDIFF,YDIFF,ZDIFF,VR,dn1,RFac,TR1,TR2
    Integer AX,AY,AZ,IX,IY,IZ,AX1,AX2,AY1,AY2, &
         AZ1,AZ2
    INTEGER NodeStart,NodeEnd         ! MFC/IVK bugfix 1 nodeend

    !...#if KEY_PARALLEL==1
    !      NodeStart = (MaxTable * (MyNod))  / NumNod + 1 ! MFC/ivk bugfix 2
    !      NodeEnd   = (MaxTable * (MyNodP)) / NumNod     ! MFC/ivk bugfix 2
    !      DO I=1,MaxTable
    !         AList(I) = 0
    !      ENDDO
    !...#else
    NodeStart = 1
    NodeEnd   = MaxTable
    !...#endif

    ! Build lookup grids

    GBGridReset = .false.

    DO I=1,NGridGB
       INDEX(I) = 0
       PTR(I) = 0
    ENDDO

    ! MF pass #1
    DO I=1,nmvsel
       CX = X(i) - XMIN
       CY = Y(i) - YMIN
       CZ = Z(i) - ZMIN
       AX = CX * dn_inv
       AY = CY * dn_inv
       AZ = CZ * dn_inv
       VR = WMAIN(i)
       R1 = (VR + Off_X) + dndiag + BufR

       K = R1 * dn_inv + ONE
       R1 = R1 * R1
       R3 = VR*VR

#if KEY_AVIMODS==1 /*init*/

       IF ((AX-K) .LT. 0) THEN
         AX1=0
       ELSE
         AX1=AX-K
       ENDIF

       IF ((AY-K) .LT. 0) THEN
         AY1=0
       ELSE
         AY1=AY-K
       ENDIF

       IF ((AZ-K) .LT. 0) THEN
         AZ1=0
       ELSE
         AZ1=AZ-K
       ENDIF

       IF ((AX+K) .GT. (NX-1)) THEN
         AX2=NX-1
       ELSE
         AX2=AX+K
       ENDIF

       IF ((AY+K) .GT. (NY-1)) THEN
         AY2=NY-1
       ELSE
         AY2=AY+K
       ENDIF

       IF ((AZ+K) .GT. (NZ-1)) THEN
         AZ2=NZ-1
       ELSE
         AZ2=AZ+K
       ENDIF

#else /* (init)*/

       AX1=AX-K
       AX2=AX+K
       AY1=AY-K
       AY2=AY+K
       AZ1=AZ-K
       AZ2=AZ+K
       IF (AX1.LT.0) AX1=0
       IF (AY1.LT.0) AY1=0
       IF (AZ1.LT.0) AZ1=0
       IF (AX2.GT.(NX-1)) AX2=NX-1
       IF (AY2.GT.(NY-1)) AY2=NY-1
       IF (AZ2.GT.(NZ-1)) AZ2=NZ-1

#endif /* (init)*/

#if KEY_AVIMODS==1 /*loops*/

       DO IZ = AZ1,AZ2

          VZ = IZ * dn + HALF*dn

          ZDIFF = VZ*VZ - 2*CZ*VZ + CZ*CZ

          DO IX = AX1,AX2
             VX = IX * dn + HALF*dn
             XDIFF = VX*VX - 2*CX*VX + CX*CX

             TR2=XDIFF+ZDIFF

                IY=AY1
                DO WHILE((TR2.LT.R1) .AND. (IY .LE. AY2))

                IPOS=IX+IZ*NXY-Offset

                   VY = IY * dn + HALF*dn
                   YDIFF = VY*VY - 2*CY*VY + CY*CY

                   R2 = TR2 + YDIFF
                   IF (R2 .LT. R1) THEN
                      POS = IY * NX + IPOS
                      PTR(POS) = PTR(POS) + 1
                   ENDIF
                 IY = IY+1
                END DO      ! iy new do while loop

          ENDDO            ! ix
       ENDDO                  !iz = az1,az2

#else /* (loops)*/


       DO IZ = AZ1,AZ2
#if KEY_PARALLEL==1
          ! IF (MOD(IZ,NumNod).EQ.(MyNodP-1)) THEN 
#endif
          VZ = IZ * dn + HALF*dn
          ZDIFF = VZ - CZ
          ZDIFF = ZDIFF*ZDIFF
          DO IX = AX1,AX2
             VX = IX * dn + HALF*dn
             XDIFF = VX - CX
             XDIFF = XDIFF*XDIFF
             TR2=XDIFF+ZDIFF
             IF (TR2.LT.R1) THEN
                IPOS=IX+IZ*NXY-Offset
                DO IY = AY1,AY2
                   VY = IY * dn + HALF*dn
                   YDIFF = VY - CY
                   YDIFF = YDIFF*YDIFF
                   R2 = TR2 + YDIFF
                   IF (R2 .LT. R1) THEN
                      POS = IY * NX + IPOS
                      PTR(POS) = PTR(POS) + 1
                   ENDIF
                ENDDO      ! iy
             ENDIF
          ENDDO            ! ix
#if KEY_PARALLEL==1
          ! ENDIF               
#endif
       ENDDO                  !iz = az1,az2

#endif /* (loops)*/

    ENDDO                     ! i=1,natom
    prev = NodeStart
    DO IZ = 0,NZ-1
#if KEY_PARALLEL==1
       ! IF (MOD(IZ,NumNod).EQ.(MyNodP-1)) THEN 
#endif
       DO IX = 0,NX-1
          DO IY = 0,NY-1
             POS = IX + IY * NX + IZ * NXY - Offset
             INDEX(POS) = prev
             prev = prev + PTR(POS)
             PTR(POS) = 0
          ENDDO         ! iy
       ENDDO            ! ix
#if KEY_PARALLEL==1
       ! ENDIF               
#endif
    ENDDO                  ! iz

    ! MFC/IVK bugfix 3, mfc variation takes care of both checks
    !                   in the published bugfix.
    ! MF, moved outside previous loop and changed slightly
    !--- Check if allocated space large enough
    IF (prev-1 .GT. NodeEnd)  &
         CALL WrnDie(-1,'<BuildGBLookupF>', &
         'Lookup grid exceeded bounds. Increase MEM.')

    ! MF pass #2
    DO I=1,nmvsel
       CX = X(i) - XMIN
       CY = Y(i) - YMIN
       CZ = Z(i) - ZMIN
       AX = CX * dn_inv
       AY = CY * dn_inv
       AZ = CZ * dn_inv
       VR = WMAIN(i)
       R1 = (VR + Off_X) + dndiag + BufR

       K = R1 * dn_inv + ONE
       R1 = R1 * R1
       TR1=100/R1
       R3 = VR*VR

#if KEY_AVIMODS==1 /*init*/

       IF ((AX-K) .LT. 0) THEN
         AX1=0
       ELSE
         AX1=AX-K
       ENDIF

       IF ((AY-K) .LT. 0) THEN
         AY1=0
       ELSE
         AY1=AY-K
       ENDIF

       IF ((AZ-K) .LT. 0) THEN
         AZ1=0
       ELSE
         AZ1=AZ-K
       ENDIF

       IF ((AX+K) .GT. (NX-1)) THEN
         AX2=NX-1
       ELSE
         AX2=AX+K
       ENDIF

       IF ((AY+K) .GT. (NY-1)) THEN
         AY2=NY-1
       ELSE
         AY2=AY+K
       ENDIF

       IF ((AZ+K) .GT. (NZ-1)) THEN
         AZ2=NZ-1
       ELSE
         AZ2=AZ+K
       ENDIF


#else /* (init)*/


       AX1=AX-K
       AX2=AX+K
       AY1=AY-K
       AY2=AY+K
       AZ1=AZ-K
       AZ2=AZ+K
       IF (AX1.LT.0) AX1=0
       IF (AY1.LT.0) AY1=0
       IF (AZ1.LT.0) AZ1=0
       IF (AX2.GT.(NX-1)) AX2=NX-1
       IF (AY2.GT.(NY-1)) AY2=NY-1
       IF (AZ2.GT.(NZ-1)) AZ2=NZ-1

#endif /* (init)*/

       DO IZ = AZ1,AZ2
#if KEY_PARALLEL==1
          ! IF (MOD(IZ,NumNod).EQ.(MyNodP-1)) THEN     
#endif
          VZ = IZ * dn + HALF*dn

#if KEY_AVIMODS==1 /*loops*/

          ZDIFF = VZ*VZ - 2*VZ*CZ + CZ*CZ
          DO IX = AX1,AX2
             VX = IX * dn + HALF*dn
             XDIFF = VX*VX - 2*CX*VX + CX*CX

             TR2=XDIFF+ZDIFF

             IY=AY1
                IPOS=IX+IZ*NXY-Offset
             DO WHILE((TR2.LT.R1) .AND. (IY .LE. AY2))
                   VY = IY*dn + HALF*dn
                   YDIFF = VY*VY - 2*CY*VY + CY*CY

                   R2 = TR2 + YDIFF
                   IF (R2 .LT. R1) THEN
                      POS = IY * NX + IPOS
                      AList(Index(POS)+PTR(POS)) = I
                      Value(Index(POS)+PTR(POS)) = TR1*(r2-r1)
                      PTR(POS) = PTR(POS) + 1
                   ENDIF
              IY=IY+1
              ENDDO      ! iy -- new do while loop
                
#else /* (loops)*/


          ZDIFF = VZ - CZ
          ZDIFF = ZDIFF*ZDIFF
          DO IX = AX1,AX2
             VX = IX * dn + HALF*dn
             XDIFF = VX - CX
             XDIFF = XDIFF*XDIFF
             TR2=XDIFF+ZDIFF
             IF (TR2.LT.R1) THEN
                IPOS=IX+IZ*NXY-Offset
                DO IY = AY1,AY2
                   VY = IY*dn + HALF*dn
                   YDIFF = VY - CY
                   YDIFF = YDIFF*YDIFF
                   R2 = TR2 + YDIFF
                   IF (R2 .LT. R1) THEN
                      POS = IY * NX + IPOS
                      AList(Index(POS)+PTR(POS)) = I
                      Value(Index(POS)+PTR(POS)) = TR1*(r2-r1)
                      PTR(POS) = PTR(POS) + 1
                   ENDIF
                ENDDO      ! iy
             ENDIF

#endif /* (loops)*/

          ENDDO  ! ix
#if KEY_PARALLEL==1
          ! ENDIF         
#endif
       ENDDO    !iz = az1,az2
    ENDDO   ! i=1,natom

    NTable = prev - NodeStart + 1
    IF (PRNLEV .GE. 6) WRITE(OUTU,*)'Total tabulated atoms = ',NTable

    ! Sort lookup grid for adjacent atoms

    ! WRITE(OUTU,*) 'Sorting lookup table...'
    DO IZ = 0,NZ-1
#if KEY_PARALLEL==1
       ! IF (MOD(IZ,NumNod).EQ.(MyNodP-1)) THEN    
#endif
       DO IX = 0,NX-1
          DO IY = 0,NY-1
             POS = IX + IY * NX + IZ * NXY - Offset
             if (ptr(pos) > maxtmp) then
                CALL WrnDie(-4,'<BuildGBLookupF>', &
                     'ptr(pos) exceeds tempa array size maxtmp')
             else IF (ptr(pos) .NE. 0) THEN
                JJ = INDEX(POS)
                DO i=1,ptr(pos)
                   TEMPA(i) = VALUE(i+jj-1)
                ENDDO
                call qcksrt(ptr(pos),tempa,alist(jj))
             ENDIF
          ENDDO
       ENDDO
#if KEY_PARALLEL==1
       ! ENDIF                    
#endif
    ENDDO

    RETURN
  END subroutine BuildGBLookupF

  real(chm_real) FUNCTION segpar_gb(raneti,ranetj,aij)
    !------------------------------------------------------------------------
    ! Segment part formula (used in MAYER)
    !
    real(chm_real) raneti,ranetj,aij,v1,v2

    v1=raneti+ranetj
    v2=abs(raneti-ranetj)
    if(aij.gt.v1)then
       segpar_gb=0.
    elseif(aij.lt.v2)then
       if(raneti.gt.ranetj)then
          segpar_gb=0.
       else
          segpar_gb=1.
       endif
    else
       if (abs(aij).gt.1D-8) then
          segpar_gb=(ranetj**2-(raneti-aij)**2)/(4*aij*raneti)
       else
          segpar_gb=10.
       endif
    endif
    return
  end FUNCTION segpar_gb

  Subroutine BuildGBGrid()
    !-------------------------------------------------------------
    ! Build a detailed MV grid of system
    ! (mimicking PB REEN w/ 1.4 Angstrom water probe)
    ! for use with GB Grid option

  use clcg_mod,only:random
  use exfunc
  use stream
  use dimens_fcm
  use param
  use psf
  use number
  use coord
  use parallel

    Integer ll(2000)
    Integer i,j,k,PS,POS,nt
    real(chm_real) R1,R2,R3,CX,CY,CZ,VX,VY,VZ, &
         XDIFF,YDIFF,ZDIFF,VR,dn1, &
         RI,RJ,RM,XI,XJ,YI,YJ,ZI,ZJ,wrsf
    real(chm_real) rl(2000)
    Integer AX,AY,AZ,IX,IY,IZ,CTR,nctr, &
         m,k2,j2,pnum,nl,ISEED,ML2
    LOGICAL Check,Check2

    IF (PRNLEV .GE. 5) WRITE(OUTU,*) 'Building GB grid...'

    DO I=1,NGridGB
       Grid(I) = 0
    ENDDO

    ! Build solute volume

    j2 = 0
    WRSF = zero
    DO I=1,nmvsel
       CX = X(i) - XMIN
       CY = Y(i) - YMIN
       CZ = Z(i) - ZMIN
       AX = CX * dn_inv
       AY = CY * dn_inv
       AZ = CZ * dn_inv
       VR = WMAIN(i) + WPROBE
       IF (WRSF .lt. VR) WRSF = VR
       R3 = VR*VR
       k = VR * dn_inv + TWO
       DO IZ = AZ-k,AZ+k
          VZ = IZ * dn
          ZDIFF = VZ - CZ
          ZDIFF = ZDIFF*ZDIFF
          DO IY = AY-k,AY+k
             VY = IY * dn
             YDIFF = VY - CY
             YDIFF = YDIFF*YDIFF
             DO IX = AX-k,AX+k
                VX = IX * dn
                XDIFF = VX - CX
                XDIFF = XDIFF*XDIFF
                R2 = XDIFF + YDIFF + ZDIFF
                IF (R2 .LT. R3) THEN
                   POS = IX + IY * NX + IZ * NXY - Offset
                   Grid(POS) = 1
                   j2 = j2 + 1
                ENDIF
             ENDDO   ! ix
          ENDDO      ! iy
       ENDDO         ! iz
    ENDDO            ! i atoms
    WRSF = ML/(WRSF**2) + 1.0D-6
    ! Remove re-entrant volume

    k = wprobe * dn_inv + 2.0
    R2 = wprobe*wprobe
    pnum = 0
    DO IX = -k,k
       XDIFF = IX * dn
       XDIFF = XDIFF*XDIFF
       DO IY = -k,k
          YDIFF = IY * dn
          YDIFF = YDIFF*YDIFF
          DO IZ = -k,k
             ZDIFF = IZ * dn
             ZDIFF = ZDIFF*ZDIFF
             R1 = XDIFF + YDIFF + ZDIFF
             IF (R1 .LT. R2) THEN
                PNUM = PNUM + 1
                D(PNUM) = IX + IY*NX + IZ*NXY
             ENDIF
          ENDDO ! iz
       ENDDO    ! iy
    ENDDO       ! ix
    IF (PRNLEV .GE. 5) WRITE(OUTU,*)'# of probe points = ',pnum

    ISEED = 12345

    IF (ML .GT. 0) THEN
       DO I = 1,ML
          XI = TWO*RANDOM(ISEED)-ONE
          YI = TWO*RANDOM(ISEED)-ONE
          ZI = TWO*RANDOM(ISEED)-ONE
          R1 = XI*XI + YI*YI + ZI*ZI
          R1 = ONE/DSQRT(R1)
          POX(i) = XI*R1
          POY(i) = YI*R1
          POZ(i) = ZI*R1
       ENDDO

       J2 = 0
       DO I = 1,nmvsel
          CHECK = .true.
          NL = 0
          XI = X(i) - XMIN
          YI = Y(i) - YMIN
          ZI = Z(i) - ZMIN
          RI = WMAIN(i) + WPROBE
          j = 0
          DO WHILE (j .lt. nmvsel)
             j = j + 1
             XJ = X(j) - XMIN
             YJ = Y(j) - YMIN
             ZJ = Z(j) - ZMIN
             RJ = WMAIN(j) + WPROBE
             R1 = DSQRT((XI-XJ)**2 + &
                  (YI-YJ)**2 + &
                  (ZI-ZJ)**2 )
             R2 = SEGPAR_GB(RI,RJ,R1)
             IF (R2 .eq. ONE) THEN
                Check = .false.
                j = nmvsel
             ENDIF
             IF (R2 .ne. ZERO) THEN
                nl = nl + 1
                ll(nl) = j
                rl(nl) = r2
             ENDIF
          ENDDO   ! while j < natom
          IF ((NL .GT. 0).AND.(CHECK)) THEN
             ML2 = int(wrsf*ri**2)
             DO K2 = 1,ML2
                CX = XI + RI * POX(k2)
                CY = YI + RI * POY(k2)
                CZ = ZI + RI * POZ(k2)
                CHECK2 = .true.
                j = 0
                DO WHILE (j .lt. nl)
                   j = j + 1
                   M = ll(j)
                   R1 =  (CX-X(m)+XMIN)**2 &
                        +(CY-Y(m)+YMIN)**2 &
                        +(CZ-Z(m)+ZMIN)**2
                   RM = WMAIN(m) + WPROBE
                   IF ((R1-RM**2) .lt. -1.0D-12) THEN
                      ! IF (PRNLEV .GT. 6) WRITE(OUTU,*)R1,RM**2
                      CHECK2 = .false.
                      j = nl
                   ENDIF
                ENDDO

                IF (CHECK2) THEN
                   J2 = J2 + 1
                   AX = CX * dn_inv
                   AY = CY * dn_inv
                   AZ = CZ * dn_inv
                   POS = AX + AY * NX + AZ * NXY - Offset
                   DO K = 1,PNUM
                      Grid(POS+D(k)) = 0
                   ENDDO
                ENDIF

             ENDDO   ! k2
          ENDIF      ! nl>0
       ENDDO         ! i atoms
       IF (PRNLEV .GE. 5) WRITE(OUTU,*)'# of carvings = ',j2
    ENDIF
    IF (CONV_GB) THEN
       IF (PRNLEV .GE. 5) WRITE(OUTU,*)'Convolving...'

       Do K=2,NZ-2
          DO J=2,NY-2
             DO I=2,NX-2
                POS = I + J * NX + K * NXY - OFFSET
                J2 = Grid(POS+1)  +Grid(POS-1)+ &
                     Grid(POS+NX) +Grid(POS-NX)+ &
                     Grid(POS+NXY)+Grid(POS-NXY) + 2*Grid(POS)
                Grid2(POS) = (J2+6)/8
             ENDDO
          ENDDO
       ENDDO

       DO K=2,NZ-2
          K2 = K*NXY - OFFSET
          DO J=2,NY-2
             J2 = J*NX + K2
             DO I=2,NX-2
                POS = I + J2
                Grid(POS)=Grid2(POS)
             ENDDO
          ENDDO
       ENDDO
    ENDIF
#if KEY_GBMVDEBUG==1
    IF (SHOWGRID_GB) THEN
       OPEN(31,file='grid.xyz',status='unknown')
       OPEN(32,file='params.txt',status='unknown')

       WRITE(32,'(2I10)')NX,NZ
       WRITE(32,'(2F10.5)')XMIN,ZMIN

       VX = dn + XMIN
       VY = dn + YMIN
       VZ = dn + ZMIN

       IF (PRNLEV .GE. 5) WRITE(OUTU,110)VX,VY,VZ
       VX = NX * dn + XMIN
       VY = NY * dn + YMIN
       VZ = NZ * dn + ZMIN
       IF (PRNLEV .GE. 5) WRITE(OUTU,110)VX,VY,VZ
       WRITE(32,'(2F10.5)')VX,VZ
       WRITE(32,'(F10.5)')dn
       CLOSE(32)

       J = NY/2
       IF (PRNLEV .GE. 5) WRITE(OUTU,*)J
       DO I=1,NX
          ! DO J=1,NY
          DO K=1,NZ
             POS = I + J * NX + K * NXY - OFFSET
             ! R1 = Grid(POS) + Grid(POS-1) + Grid(POS+1) +
             ! c        Grid(POS-NX)+Grid(POS+NX) +
             ! c        Grid(POS-Nxy)+Grid(POS+Nxy)
             ! WRITE(1,'(F5.3)') R1

             WRITE(1,'(I6)') Grid(POS)

             ! IF (Grid(POS) .eq. 2) THEN
             !    WRITE(31,100)I,J,K
             ! ENDIF

          ENDDO
          ! ENDDO
       ENDDO

100    FORMAT ('H ',I5,1X,I5,1X,I5)
110    FORMAT (3(F10.5,1X))
       CLOSE(31)
    ENDIF
#endif /* GBMVDEBUG*/

1010 RETURN
  END Subroutine BuildGBGrid

  SUBROUTINE FindExtents(K)
    !-------------------------------------------------------------
    ! Find the extents of the current system
    !-------------------------------------------------------------

  use exfunc
  use stream
  use dimens_fcm
  use param
  use psf
  use number
  use deriv
  use coord
  use consta
#if KEY_PARALLEL==1
  use parallel
#endif 
    INTEGER i,k
    real(chm_real) R,RMAX,VR,SUM,RFac

    XMAX = -ANUM
    YMAX = -ANUM
    ZMAX = -ANUM
    XMIN = ANUM
    YMIN = ANUM
    ZMIN = ANUM
    RMAX = -ANUM
    IF (.not. GBMVGrid) THEN
       SUM = ZERO
       IF (P3 .LE. ZERO) THEN
          RFac = (log(1.0D-8)/(log(Lambda1)*P1))**(PT25)
       ENDIF
    ENDIF
    DO I = 1,nmvsel
       IF (X(I).GT.XMAX) XMAX=X(I)
       IF (X(I).LT.XMIN) XMIN=X(I)
       IF (Y(I).GT.YMAX) YMAX=Y(I)
       IF (Y(I).LT.YMIN) YMIN=Y(I)
       IF (Z(I).GT.ZMAX) ZMAX=Z(I)
       IF (Z(I).LT.ZMIN) ZMIN=Z(I)
       VR = WMAIN(i)
       IF (RMAX.LT.VR) RMAX=VR
       IF (.not. GBMVGrid) THEN
          IF (P3 .GT. ZERO) THEN
             SUM=SUM+FOUR*PI/THREE*((VR+Off_X+dndiag+BufR)/DN)**3
          ELSE
             SUM=SUM+FOUR*PI/THREE*((VR*RFac+dndiag+BufR)/DN)**3
          ENDIF
       ENDIF
    ENDDO
    IF (GBMVGrid) THEN
       RMAX = RMAX + WPROBE*8
    ELSE
       IF (P3 .GT. ZERO) THEN
          RMAX = RMAX + (Off_X+dndiag+BufR) + DN
       ELSE
          RMAX = RMAX*RFac + dndiag + BufR + DN
       ENDIF
    ENDIF
    XMIN = XMIN - RMAX
    YMIN = YMIN - RMAX
    ZMIN = ZMIN - RMAX
    XMAX = XMAX + RMAX
    YMAX = YMAX + RMAX
    ZMAX = ZMAX + RMAX

    NX = (XMAX-XMIN) * dn_inv + 1
    NY = (YMAX-YMIN) * dn_inv + 1
    NZ = (ZMAX-ZMIN) * dn_inv + 1

    IF(NXP_GB.GT.NX) NX=NXP_GB
    IF(NYP_GB.GT.NY) NY=NYP_GB
    IF(NZP_GB.GT.NZ) NZ=NZP_GB

    NXY = NX * NY
    OFFSET = -1 !NX + NXY
    IF (.not. GBMVGrid) THEN
       NGridGB = 2 * NXY * NZ
    ELSE
       NGridGB = NXY * NZ
    ENDIF
    IF ((.not. GBMVGrid).and.(k .eq. 1)) THEN
       VR = MemMult
       VR = VR/100.0D0
#if KEY_PARALLEL==1
       ! MFC/IVK bugfix 4  optional and probably circumvents a problem in the
       !    parallel implementation. Checking it out, this change in place till
       !    verified since it does no harm.
       VR = MemMult+NumNod
       VR = VR/100.0D0
#endif 
       MaxTable = Sum * (ONE + VR)
       ! WRITE(OUTU,*)'MaxTable = ',MaxTable,'Sum=',Sum
    ENDIF

    RETURN
  END SUBROUTINE FindExtents

  ! Returns the last element of WTR <= the given radius,
  ! or WTR(1) if there is no such element.
  real(chm_real) function lastWtrLE(radius)
    real(chm_real),intent(in) :: radius
    integer :: j

    lastWtrLE = Wtr(1)
    do j = NumR, 1, -1
       if (Wtr(j) <= radius) then
          lastWtrLE = Wtr(j)
          exit
       endif
    enddo
  end function lastWtrLE

  ! MF set radii
  !
  Subroutine SetGBMVRad(WMAIN)
    ! initialize radius temp array
  use number
  use exfunc
  use stream
  use dimens_fcm
  use param
  use psf
    real(chm_real) WMAIN(*)
    INTEGER i

    tWtScale = ONE / ((WTR(1)+WTR(2)) * HALF)
    tWtScale2 = ONE/(ONE / WTR(1) - ONE / WTR(2))

    DO i=1,nmvsel
       IF (QWeight) THEN
          Rad(i) = WMAIN(i)
       ELSE
          Rad(i) = VDWR(ITC(IAC(I)))
          WMAIN(i) = Rad(i)
          ! IF (WMAIN(i) .LT. -HSX1) WMAIN(i) = -HSX1
       ENDIF

       Rad(i) = lastWtrLE(Rad(i))
    ENDDO

    RETURN
  END subroutine SetGBMVRad

  ! MF precalculate some quantities that are derived from
  !    the atomic radii for fast GBMV

  Subroutine CalcTRad(WMAIN,nmvsel,NK)
    ! initialize radius temp array
  use number
    real(chm_real) WMAIN(*)
    INTEGER nmvsel
    INTEGER NK
    INTEGER i,ki
    DO i=1,nmvsel
       ki=(i-1)*NK
       rtmp(ki+1)=WMAIN(i)

       rtmp(ki+2)=(WMAIN(i)+HSX1)*(WMAIN(i)+HSX1)
       rtmp(ki+3)=(WMAIN(i)+HSX2)*(WMAIN(i)+HSX2)
       rtmp(ki+4)=(WMAIN(i)+On_X)*(WMAIN(i)+On_X)
       rtmp(ki+5)=(WMAIN(i)+Off_X)*(WMAIN(i)+Off_X)
       rtmp(ki+6)=ONE/(rtmp(ki+5)-rtmp(ki+4))
       rtmp(ki+7)=ONE/(rtmp(ki+3)-rtmp(ki+2))
       rtmp(ki+8)=ONE/(P2*WMAIN(i)+P1)
       rtmp(ki+9)=ONE-WMAIN(i)*WMAIN(i)*rtmp(ki+8)
    ENDDO
    RETURN
  END Subroutine CalcTRad

  Subroutine RunGBMV(GBEnergy, &
       JNbl,INbl,Inbl14,INblo14, &
       QInte,ISlct,JSlct)
    !------------------------------------------------------------
    ! Called from Energy, this re-allocates memory if
    ! necessary and then calls the alph_gb, energy, and force
    ! builder.
    !------------------------------------------------------------
  use exfunc
  use stream
  use dimens_fcm
  use psf
  use number
  use deriv
  use coord
  use consta
  use memory
  use param ! for HDGB
#if KEY_DHDGB==1
!AP/MF
  use dhdgb
  use derivdhdgb
#endif
    real(chm_real) GBEnergy
    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)
    Integer ISlct(*), JSlct(*)
    Logical QInte

    ! SJT/MF
    Integer i
    real(chm_real) X1, X2
    ! SJT/MF

    NX_GB = NX
    NY_GB = NY
    NZ_GB = NZ
    OFFSET_GB = OFFSET
    NGridGB_GB = NGridGB
    XMIN_GB = XMIN
    YMIN_GB = YMIN
    ZMIN_GB = ZMIN
    XMAX_GB = XMAX
    YMAX_GB = YMAX
    ZMAX_GB = ZMAX

    ! SJT/MF
    DO I = 1, nmvsel
       IF (QHDGBRC) THEN
          CALL CorrectRadius(Rad(I), DR_HDGB(I), R_HDGB(I), Z(I))
       ELSE
          Rad(I) = R_HDGB(I)
       END IF

       WMAIN(I) = Rad(I)

       ! WRITE(OUTU,('(a,1x,5(f10.4))')) 'Radius: ', Z(i), WMAIN(i),
       !     &      DR_HDGB(i), R_HDGB(i), Rad(i)

       IF (P3 .GT. ZERO) THEN
          X1 = WMAIN(i)+HSX2
          X1 = X1*X1
          X2 = WMAIN(i)+HSX1
          X2 = X2*X2
          Gamma(i) = ONE/(X1-X2)
       ELSE
          Gamma(i) = P1*log(lambda1)/(Rad(i)**4)
       ENDIF

       Rad(i) = lastWtrLE(Rad(i))
    ENDDO
    ! SJT/MF

    CALL FindExtents(0) ! Find new grid size, not table size

    IF (NGridGB .GT. (NGridGB_GB*2)) THEN
       IF (PRNLEV .GE. 1) THEN
          WRITE(OUTU,*) 'Readjusting grid size'
          WRITE(OUTU,'(A,I10,A,I10,A,I10)') &
               'New Extents: ',NX,' x ',NY,' x ',NZ
          WRITE(OUTU,*) 'Grid size',NGridGB

          if(NGridGB .GT. (NGridGB_GB*5)) THEN
             write(outu,*)"Asking for too large a grid change"
             write(outu,*)"Exiting."
             CALL STOPCH('DJP')
          endif
       ENDIF

       call chmrealloc('gbmvmodule.src','RunGBMV','index',ngridgb,intg=index)
       call chmrealloc('gbmvmodule.src','RunGBMV','ptr',ngridgb,intg=ptr)

       NGridGB_GB = NGridGB
       GBGridReset = .true.

    ENDIF

    IF (FAST_GB.EQ.0) THEN
       ! MF call original routine
       Call RunGBMV1(GBEnergy,JNbl,INbl,Inbl14,INblo14, &
            QInte,Islct,Jslct)
    ELSE
       ! MF call fast GBMV routine
       Call RunGBMV1F(GBEnergy,JNbl,INbl,Inbl14,INblo14, &
            QInte,Islct,Jslct)
    ENDIF
    RETURN
  END Subroutine RunGBMV

  Subroutine RunGBMV1(GBEnergy, &
       JNbl,INbl,Inbl14,INblo14, &
       QInte,Islct,Jslct)
    !------------------------------------------------------------
    !  Build alph_gb array and some derivative components
    !------------------------------------------------------------
  use new_timer,only:timer_start,timer_stop,      & 
     T_gb_radii,T_gb_energ,T_gb_force          
  use exfunc
  use stream
  use dimens_fcm
  use param
  use psf
  use number
  use deriv
  use coord
  use consta
  use inbnd
  use energym
#if KEY_PARALLEL==1
  use parallel
#endif
#if KEY_DHDGB==1
!AP/MF
 use dhdgb,only:sdef,MITHICK_FHDGB,MATHICK_FHDGB,FRIC_DHDGB,MAXNUMS,QFHDGB,TOTALS
 use derivdhdgb
#endif 
    real(chm_real) RE,RG,RI,RJ,R1,R2,R3,R4,R5,R1A,R6,W2,R7,RH
    real(chm_real) R,SX,SY,SZ,S,VX,VY,VZ,TX,TY,TZ,AI,AJ
    real(chm_real) C1,C2,C3,C4,C5,Q,D,A,CX,CY,CZ,Factor,SALT,FGB,TA2
    real(chm_real) XDIFF,YDIFF,ZDIFF,Rx,Ry,Rz,Impulse,E_Alpha
    real(chm_real) GBEnergy,RMax,XE1,XE2,YE1,YE2,ZE1,ZE2
    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)
    Integer i,j,k,l,PS,IX,IY,IZ,POS,ii,ctr,k2,sum,k1,iii,m
    INTEGER num,i2
    real(chm_real) C2OfNb
    Integer ITemp, NPr, jpr
    real(chm_real) C2OnNb, Rul3, Rul12, RijL, RijU, FSw, DFSw
    real(chm_real) Rul3_X, Rul12_X, RijL_X, RijU_X, FSw_X, DFSw_X
    real(chm_real) B,C,BX,BY,BZ
    real(chm_real) dMX,dMY,dMZ,X1,X2,X3,X4,X5,X6,X0
    Logical LOuter, Switch,QUpdateAlpha
    real(chm_real) TOTDIFF,Sum1
    real(chm_real) HSVAL,TOL2,RTWOPI,S1,Area1,Force_Scale
    real(chm_real) WtScale,FOURPI,WtScale2,Surf5,EMPA_GB,EMPB_GB
    LOGICAL OK
    Integer ISlct(*), JSlct(*)
    Logical QInte,LOK
    real(chm_real) RR1,RRA
    ! SJT/MF VDW
    real(chm_real) EPSWVDW, SIGWVDW, GVDW
    real(chm_real) X1VDW, X2VDW, X3VDW, X4VDW
    real(chm_real) EPSIW, SIGIW
    real(chm_real) Gnp_VDW, Gnp_ASP
    ! SJT/MF VDW

    ! SJT/MF
    real(chm_real) Gnp_HDGB
    real(chm_real) X0_HDGB, X1_HDGB, X2_HDGB, X3_HDGB
    real(chm_real) dMZ_HDGB, TZ_HDGB
    real(chm_real) R4_HDGB
    real(chm_real) S2_HDGB, Area1_HDGB
    real(chm_real) ST_HDGB
    real(chm_real) VZ1_HDGB
#if KEY_HDGBVDW==1
! MS/MF
     real(chm_real) EPSNVDW,SIGNVDW,EPSC2VDW,SIGC2VDW,SIGPVDW
     real(chm_real) EPSCVDW,SIGCVDW,EPSOVDW,SIGOVDW,EPSPVDW
     real(chm_real) GVDW_HDGB
     real(chm_real) SIG_I,EPS_I
     real(chm_real) At_HDGB
!MS/MF
#endif
#if KEY_DHDGB==1
!AP/MF
    real(chm_real) ZIP(NATOM)
    real(chm_real) XSDEF(INCT_FHDGB-1),XSDEFM(INCT_FHDGB-1),YSDEF(INCT_FHDGB-1),YSDEFM(INCT_FHDGB-1)
    real(chm_real) LOWR0,UPR0,LOWMR0,UPMR0,INTRAD,INTMRAD,R0
    INTEGER DIM,MAXLINE,MINLINE,LINE,RLL
    real(chm_real) RCHECK
    INTEGER NUMS,ALLCA
    INTEGER START
    real(chm_real) SUP(INCT_FHDGB)
    real(chm_real) SUPMAX,MINDISTUP,supmin
    real(chm_real) SLO(INCT_FHDGB)
    real(chm_real) SLOMAX,MINDISTLO,slomin
    real(chm_real) DEF_ENERUP,DEF_ENERLO,CRICLEMRAD,CRICLERAD
    real(chm_real) DE_UP(INCT_FHDGB),DE_LO(INCT_FHDGB)
    real(chm_real) MEMSPLUP(4,INCT_FHDGB),MEMSPLLO(4,INCT_FHDGB)
    real(chm_real) C4_CSAPE(INCT_FHDGB-1,INCT_FHDGB)
    real(chm_real) C3_CSAPE(INCT_FHDGB-1,INCT_FHDGB)
    real(chm_real) C2_CSAPE(INCT_FHDGB-1,INCT_FHDGB)
    real(chm_real) C1_CSAPE(INCT_FHDGB-1,INCT_FHDGB)
    real(chm_real) C4_CSAPE_small(INCT_FHDGB-1,INCT_FHDGB)
    real(chm_real) C3_CSAPE_small(INCT_FHDGB-1,INCT_FHDGB)
    real(chm_real) C2_CSAPE_small(INCT_FHDGB-1,INCT_FHDGB)
    real(chm_real) C1_CSAPE_small(INCT_FHDGB-1,INCT_FHDGB)
    real(chm_real) SFHDGB(NATOM),DEPSP(NATOM),DGAMMAP(NATOM)
    real(chm_real) DEPSPL(NATOM),DGAMMAPL(NATOM)
    real(chm_real) SFHDGBLO(NATOM)
    real(chm_real) SFHDGB_OLD(NATOM),SFHDGBLO_OLD(NATOM)
    real(chm_real) A_EPS(NATOM),C_EPS(NATOM),A_NP(NATOM),C_NP(NATOM)
    real(chm_real) E_EPS(NATOM),E_NP(NATOM)
    real(chm_real) MA_EPS(NATOM),MC_EPS(NATOM),MA_NP(NATOM),MC_NP(NATOM)
    real(chm_real) ME_EPS(NATOM),ME_NP(NATOM)
    real(chm_real) DA_EPS_DZ(NATOM),DC_EPS_DZ(NATOM)
    real(chm_real) DE_EPS_DZ(NATOM)
    real(chm_real) MDA_EPS_DZ(NATOM),MDC_EPS_DZ(NATOM)
    real(chm_real) MDE_EPS_DZ(NATOM)
    real(chm_real) DA_NP_DZ(NATOM),DC_NP_DZ(NATOM)
    real(chm_real) DE_NP_DZ(NATOM)
    real(chm_real) MDA_NP_DZ(NATOM),MDC_NP_DZ(NATOM)
    real(chm_real) MDE_NP_DZ(NATOM)
    real(chm_real) EPS_S0_HDGB(NATOM),DEPS_S0_HDGB(NATOM)
    real(chm_real) EPS_S0_HDGBL(NATOM),DEPS_S0_HDGBL(NATOM)
    real(chm_real) SurfT_S0_HDGB(NATOM),DSurfT_S0_HDGB(NATOM)
    real(chm_real) SurfT_S0_HDGBL(NATOM),DSurfT_S0_HDGBL(NATOM)
    real(chm_real) DS_DTHETA(NATOM),DTHETADX(NATOM,NATOM)
    real(chm_real) DS_DTHETALO(NATOM)
    real(chm_real) DTHETADY(NATOM,NATOM)
    real(chm_real) DTHETADZ(NATOM,NATOM),DRDX(NATOM,NATOM)
    real(chm_real) DRDY(NATOM,NATOM),DRDZ(NATOM,NATOM)
    real(chm_real) SUM_DX_GB,SUM_DX
    real(chm_real) SUM_DY_GB,SUM_DY
    real(chm_real) SUM_DZ_GB,SUM_DZ
    real(chm_real) DS_DSUP(INCT_FHDGB-1),DS_DSLO(INCT_FHDGB-1),CO(NATOM)
    real(chm_real) DGnp_DS_ATM(NATOM,2*(INCT_FHDGB-1))
    real(chm_real) DGB_DS_ATM(NATOM,2*(INCT_FHDGB-1))
    real(chm_real) DGB_DS(2*(INCT_FHDGB-1))
    real(chm_real) DGnp_DS(2*(INCT_FHDGB-1))
    real(chm_real) DX_DEF(NATOM),DY_DEF(NATOM)
    real(chm_real) DDEF_DRUP,DDEF_DRLO
    real(chm_real) DSI_DSJ(NATOM,2*(INCT_FHDGB-1))
    real(chm_real) DE_DS(2*(INCT_FHDGB-1)),DISTFHDGB
    real(chm_real) test(natom)
    real(chm_real) CENTERX(NATOM),CENTERY(NATOM)
    real(chm_real) SUMZI,SUMSU,SUMSL,SUMEPS,SUMGAM
    real(chm_real) AVGSDEF,AVGSDEFL,LAM1_DHDGB(NATOM),LAM2_DHDGB(NATOM)
    real(chm_real) LAM1L_DHDGB(NATOM),LAM2L_DHDGB(NATOM)
    real(chm_real) DLAM1_DR_DHDGB(NATOM),DLAM1L_DR_DHDGB(NATOM)
    real(chm_real) R_DHDGB
    INTEGER COUNTATM
    !,DE_DS_LO(INCT_FHDGB-1)
    DATA ((C4_CSAPE(I,J),J=1,INCT_FHDGB),I=1,INCT_FHDGB-1) &
           /0.36668007958577D0, -0.59586207614247D0, &
         0.27501813054642D0, -0.00000232102317D0, &
         -0.27500884626904D0, 0.22917503330248D0, &
         -0.22886800869784D0, 0.59504075594768D0, &
         -0.59503943144699D0, 0.27463968986002D0, &
         -0.00000198672470D0, -0.04577101893818D0, &
         0.04583613398973D0, -0.27501680422862D0, &
         0.59585544444791D0, -0.59585544444791D0, &
         0.27501680422862D0, -0.04583613398973D0, &
         0.04583381296656D0, 0.00000198945032D0, &
         -0.27501647265577D0, 0.59585577602076D0, &
         -0.59585710233857D0, 0.22918199655670D0, &
         -0.22917503330248D0, 0.27500884626904D0, &
         0.00000232102317D0, -0.27501813054642D0, &
         0.59586207614247D0, -0.36668007958577D0/
    DATA ((C3_CSAPE(I,J),J=1,INCT_FHDGB),I=1,INCT_FHDGB-1) &
           /-0.69083110046579D0, 0.86355019114919D0, &
         -0.17272367174366D0, -0.17269618501754D0, &
         0.86352215494490D0, -0.69082138886710D0, &
         0.69044717046513D0, -1.38063237729317D0, &
         0.86250604795546D0, -0.17224554498198D0, &
         -0.17269732784665D0, 0.17262203170121D0, &
         -0.17270680470747D0, 0.86354464408898D0, &
         -1.38165271084635D0, 0.86352979754349D0, &
         -0.17271867424447D0, 0.00000374816582D0, &
         -0.00000086960866D0, -0.17270482067047D0, &
         0.86354306489181D0, -1.38165271086305D0, &
         0.86353137680747D0, -0.17271604055710D0, &
         0.17271020991305D0, -0.17271128774104D0, &
         -0.17270482068731D0, 0.86354431430856D0, &
         -1.38165730546797D0, 0.69081888967471D0/
    DATA ((C2_CSAPE(I,J),J=1,INCT_FHDGB),I=1,INCT_FHDGB-1) &
            /-0.50639609280114D0, 0.65107698992693D0, &
           -0.21701990650738D0, 0.21701990650738D0, &
           -0.65107698992693D0, 0.50639609280114D0, &
           -0.50641074649938D0, 0.00000392500776D0, &
           0.65109059698004D0, -0.21702356985906D0, &
           0.21702095315252D0, -0.14468115878187D0, &
           0.14468482213355D0, -0.65109269027031D0, &
           0.00000026165608D0, 0.65109164362517D0, &
           -0.21702409317122D0, 0.07234005602672D0, &
           -0.07234005602672D0, 0.21702409317122D0, &
           -0.65109164362517D0, -0.00000026165608D0, &
           0.65109269027031D0, -0.14468482213355D0, &
           0.14468115878187D0, -0.21702095315252D0, &
           0.21702356985906D0, -0.65109059698004D0, &
            -0.00000392500776D0, 0.50641074649938D0/
    DATA ((C1_CSAPE(I,J),J=1,INCT_FHDGB),I=1,INCT_FHDGB-1) &
           /1.0D0,0.0D0,0.0D0,0.0D0,0.0D0,0.0D0, &
           0.0D0,1.0D0,0.0D0,0.0D0,0.0D0,0.0D0, &
           0.0D0,0.0D0,1.0D0,0.0D0,0.0D0,0.0D0, &
           0.0D0,0.0D0,0.0D0,1.0D0,0.0D0,0.0D0, &
           0.0D0,0.0D0,0.0D0,0.0D0,1.0D0,0.0D0/
    DATA ((C4_CSAPE_small(I,J),J=1,INCT_FHDGB),I=1,INCT_FHDGB-1) &
          /0.802947883087865D0,  -0.726956995356112D0, &
          -0.000000000000000D0,  -0.075990887731753D0, &
           0.000000000000000D0,  0.000000000000000D0, &
          -0.151981775463507D0,   0.726956995356112D0, &
          -0.726956995356112D0,   0.151981775463507D0, &
           0.000000000000000D0,  0.000000000000000D0, &
           1.030920546283125D0,   0.000000000000000D0, &
          -0.227972663195260D0,  -0.802947883087865D0, &
           0.000000000000000D0,  0.000000000000000D0, &
           0.000000000000000D0,  0.000000000000000D0, &
           0.000000000000000D0,  0.000000000000000D0, &
           0.000000000000000D0,  0.000000000000000D0, &
           0.000000000000000D0,  0.000000000000000D0, & 
           0.000000000000000D0,  0.000000000000000D0, &
           0.000000000000000D0,  0.000000000000000D0/
    DATA ((C3_CSAPE_small(I,J),J=1,INCT_FHDGB),I=1,INCT_FHDGB-1) &
         /-1.318408156229428D0,   0.976449161436538D0, &
          0.227972663195260D0,   0.113986331597630D0,&
          0.000000000000000D0,  0.000000000000000D0,&
          0.341958994792890D0,  -1.204421824631798D0,&
          0.976449161436538D0,  -0.113986331597630D0,&
          0.000000000000000D0,  0.000000000000000D0,&
          -1.546380819424688D0,   0.227972663195260D0,&
          0.227972663195260D0,  1.090435493034168D0,&
          0.000000000000000D0,  0.000000000000000D0,&
          0.000000000000000D0,  0.000000000000000D0,&
          0.000000000000000D0,  0.000000000000000D0,&
          0.000000000000000D0,  0.000000000000000D0,&
          0.000000000000000D0,  0.000000000000000D0,&
          0.000000000000000D0,  0.000000000000000D0,&
          0.000000000000000D0,  0.000000000000000D0/
    DATA ((C2_CSAPE_small(I,J),J=1,INCT_FHDGB),I=1,INCT_FHDGB-1) &
          /0.079577471545948D0,   0.477464829275686D0,&
          -0.477464829275686D0,  -0.079577471545948D0,&
           0.000000000000000D0,    0.000000000000000D0,&
           -0.397887357729738D0,  -0.000000000000000D0,&
           0.477464829275686D0,  -0.079577471545948D0,&
           0.000000000000000D0,    0.000000000000000D0,&
           0.079577471545948D0,  -0.477464829275686D0,&
           0.000000000000000D0,   0.397887357729738D0,&
           0.000000000000000D0,    0.000000000000000D0,&
           0.000000000000000D0,    0.000000000000000D0,&
           0.000000000000000D0,    0.000000000000000D0,&
           0.000000000000000D0,    0.000000000000000D0,&
           0.000000000000000D0,    0.000000000000000D0,&
           0.000000000000000D0,    0.000000000000000D0,&
           0.000000000000000D0,    0.000000000000000D0/
    DATA ((C1_CSAPE_small(I,J),J=1,INCT_FHDGB),I=1,INCT_FHDGB-1) &
           /1.0D0,0.0D0,0.0D0,0.0D0,0.0D0,0.0D0, &
           0.0D0,1.0D0,0.0D0,0.0D0,0.0D0,0.0D0, &
           0.0D0,0.0D0,1.0D0,0.0D0,0.0D0,0.0D0, &
           0.0D0,0.0D0,0.0D0,0.0D0,0.0D0,0.0D0, &
           0.0D0,0.0D0,0.0D0,0.0D0,0.0D0,0.0D0/
 
    

#endif
    !---------------------------------
    ! Compute Alphas for every atom
    !---------------------------------
    SAVE Force_Scale

    call timer_start(T_GB_RADII) 

    ! Generate radii and exponent values

    RTWOPI = ONE/TWOPI
    FOURPI = TWO*TWOPI
    Factor = -CCELEC * (ONE/EPS - ONE/EPS_GB)
    Switch = LCons .and. .not. LShft .and. .not. LFSwt

    WtScale = ONE / ((WTR(1)+WTR(2)) * HALF)
    WtScale2 = ONE/(ONE / WTR(1) - ONE / WTR(2))

    ! Prepare function cutoff variables

    TOL2 = 1.0D-16
    ! LAMBDA1 = log(TOL2)/Beta

    EXT = LOG(TOL_GB)/BETA
    THRESH1 = LAMBDA1 - EXT
    THRESH2 = LAMBDA1 + EXT
    IF (THRESH1 .LT. ZERO) THRESH1 = 1.0D-16
    ! WRITE(OUTU,*)'T1,T2',THRESH1,THRESH2

    HSVAL = ONE               !LAMBDA1*(2.0D0)
    ! WRITE(OUTU,*)'L,H',LAMBDA1,HSVAL

    IF (IMPULSE_GB .EQ. 3) THEN
       EMPA_GB = DEXP(-EMP_GB)
       EMPB_GB = ALFRQ
       EMPB_GB = EMPB_GB*(ONE-EMPA_GB)/(ONE-EMPA_GB**ALFRQ)
    ENDIF

    ! in case of no HDGB, we would have done the following:
    ! Rotate integration weights to molecular frame
    ! NYI!
!    DO i=1,nmvsel
!
!       IF (QWeight) THEN
!          Rad(i) = WMAIN(i)
!       ELSE
!          Rad(i) = VDWR(ITC(IAC(I)))
!          WMAIN(i) = Rad(i)
!          ! IF (WMAIN(i) .LT. -HSX1) WMAIN(i) = -HSX1
!       ENDIF
!
!       IF (P3 .GT. ZERO) THEN
!          X1 = WMAIN(i)+HSX2
!          X1 = X1*X1
!          X2 = WMAIN(i)+HSX1
!          X2 = X2*X2
!          Gamma(i) = ONE/(X1-X2)
!       ELSE
!          Gamma(i) = P1*log(lambda1)/(Rad(i)**4)
!       ENDIF
!
!       Rad(i) = lastWtrLE(Rad(i))
!
!    ENDDO

    ! Generate grid-based lookup tables if asked or heuristically

    R2 = (BufR*0.99)**2

    TOTDIFF = ZERO
    DO I=1,nmvsel
       R1 = (X(I)-XCP(I))*(X(I)-XCP(I))+ &
            (Y(I)-YCP(I))*(Y(I)-YCP(I))+ &
            (Z(I)-ZCP(I))*(Z(I)-ZCP(I))
       TOTDIFF = TOTDIFF + R1
       IF (R1 .GT. R2) THEN
          GBGridReset = .true.
          EXIT
       ENDIF
    ENDDO

    GBStepCtr=GBStepCtr+1
    QUpdateAlpha = .NOT.((TOTDIFF .LT. 1.0D-20).AND.(FIXA))
    QUpdateAlpha = QUpdateAlpha .AND. (GBStepCtr .eq. 1)
    IF (GBStepCtr .ge. ALFRQ) THEN
       GBStepCtr=0
    ENDIF

    IF (DRSTEP .eq. 0 .or. DRSTEP .ge. DRFRQ) THEN
       DRSTEP=1
       ! MFC?IVK bugfix 10 Array dimensioned 1:MaxSurf
       ! DO I=0,NAtom*NWeights-1
       IF (nmvsel*nweights .gt. MaxSurf)  &
            CALL WrnDie(-1,'<RunGBMV1>', &
            'MaxSurf too small for natom*nweights')
       DO I=1,nmvsel*NWeights
          Surf_Atom(I) = 0
       ENDDO
    ELSE
       DRSTEP = DRSTEP + 1
    ENDIF

    IF (GBGridReset.and.QUpdateAlpha) THEN
       DO I=1,nmvsel
          XCP(I) = X(I)
          YCP(I) = Y(I)
          ZCP(I) = Z(I)
       ENDDO
       NGridGB = NGridGB_GB
       CALL BuildGBLookup()
    ELSE
       XMIN=XMIN_GB
       YMIN=YMIN_GB
       ZMIN=ZMIN_GB
       XMAX=XMAX_GB
       YMAX=YMAX_GB
       ZMAX=ZMAX_GB
       NX = NX_GB
       NY = NY_GB
       NZ = NZ_GB
       NXY = NX*NY
       OFFSET = OFFSET_GB
       NGridGB = NGridGB_GB
    ENDIF

    IF (QUpdateAlpha) THEN
       ! WRITE(OUTU,*) 'Getting alphas...'
       l = 0

       NSurf = 0
       R4 = log(thresh2)
       Surface_Area = ZERO

       ! SJT/MF
       Gnp_HDGB = ZERO
       Gnp_VDW  = ZERO
       DO i=1,nmvsel
          T(i) = ZERO
#if KEY_HDGBVDW==1
          DP(i) = ZERO
          DDD_HDGB(I)=ZERO
#endif
          Alph_Gb(i) = ZERO
          F(i) = ZERO
          G(i) = ZERO
          DX_GB(I)=ZERO
          DY_GB(I)=ZERO
          DZ_GB(I)=ZERO

#if KEY_HDGBVDW==1
          DZ_GBVdw(I)=ZERO
#endif

          ! SJT/MF
          P_HDGB(I)=ZERO
          AtSA_HDGB(I)=ZERO
#if KEY_DHDGB==1
!AP/MF
          DX_DEF(I)=ZERO
          DY_DEF(I)=ZERO
          DEPSPL(I)=ZERO
          DEPSP(I)=ZERO
          DGAMMAPL(I)=ZERO
          DGAMMAP(I)=ZERO
#endif
       ENDDO
#if KEY_DHDGB==1
!AP/MF
          DO J=1,TOTALS
             DO I=1,nmvsel
                DGB_DS_ATM(I,J)=ZERO
                DGnp_DS_ATM(I,J)=ZERO
             ENDDO
             DE_DS(J)=ZERO
             DS_DHDGB(J)=ZERO
          ENDDO
#endif
#if KEY_GBMVDEBUG==1
       IF (ECOMP_GB) OPEN(1,file='comp.txt',status='unknown')
       IF (ESURF_GB) OPEN(2,file='surf_atoms.txt',status='unknown')

       IF (EVDW_GB) THEN
          OPEN(1,file='atoms.xyz',status='unknown')
          DO I=1,nmvsel
             WRITE(1,'(5(F10.5,1X)') &
                  X(I),Y(I),Z(I),CCNBA(I),CCNBB(I)
          ENDDO
          CLOSE(1)
          OPEN(1,file='surf.xyz',status='unknown')
       ENDIF

       OPEN (2,file='volume.xyz',status='unknown')
#endif /*                         GBMVDEBUG*/
#if KEY_DHDGB==1
!AP/MF
       DEF_ENERUP=0.0D0
       DEF_ENERLO=0.0D0
#endif
       ! SJT/MF
       if (CORR_GB .eq. 3) then
#if KEY_DHDGB==1
!AP/MF
         IF (QFHDGB) THEN
             NUMS=NDT_FHDGB
!INITIALIZE THETA VALUES
             DO I=1,INCT_FHDGB
                THETAZ(I)=0.0D0
             ENDDO
             DO I=1,NDT_FHDGB+1
                THETAZ(I)=(I-1)*TWOPI/(NDT_FHDGB)
             ENDDO
             AVGSDEF=0.0D0
             AVGSDEFL=0.0D0
             DO I=1,INCT_FHDGB-1
                SUP(I)=SDEF(I)
!TEST
!                write(outu,*)'SUP(I)=',SUP(I)
                AVGSDEF=AVGSDEF+SUP(I)/NUMS
                SLO(I)=SDEF(I+INCT_FHDGB-1)
                AVGSDEFL=AVGSDEFL+SLO(I)/NUMS
             ENDDO
             SUP(NDT_FHDGB+1)=SUP(1)
             SLO(NDT_FHDGB+1)=SLO(1)

             SUP(INCT_FHDGB)=SUP(1)
             SLO(INCT_FHDGB)=SLO(1)

             CALL FIND_DEF_ENER(UN_FHDGB,CIRCLERAD,SUP, &
                                 THETAZ, &
                                 DEF_ENERUP,DE_UP, &
                                 PENALTY_OFF,PENALTY_OFFL, &
                                 PENALTY_PREFACT,PENALTY_PREFACTL, &
                                 PENALTY_FACT,PENALTY_FACTL)
             CALL FIND_DEF_ENER(UN_FHDGB,CIRCLEMRAD,SLO, &
                                 THETAZ, &
                                 DEF_ENERLO,DE_LO, &
                                 PENALTY_OFF,PENALTY_OFFL, &
                                 PENALTY_PREFACT,PENALTY_PREFACTL, &
                                 PENALTY_FACT,PENALTY_FACTL)
!if we decide to introduce variation of the radius later)
             DDEF_DRUP=DE_UP(1)
             DDEF_DRLO=DE_LO(1)
!!!!
             DDEF_DRUP=0.0D0
             DDEF_DRLO=0.0D0
             DO I=1,TOTALS
                DE_DS(I)=0.0D0
             ENDDO
             DO I=1,MAXNUMS
                DE_DS(I)=DE_UP(I+1)
                DE_DS(I+MAXNUMS)=DE_LO(I+1)
!                write(outu,*)'DE_DS(I)',DE_DS(I)
!                write(outu,*)'DE_DSlo(I)',DE_DS(I+MAXNUMS)
             ENDDO
             DO I=1,nmvsel
                DO J=1,TOTALS
                   DSI_DSJ(I,J)=ZERO
                ENDDO
             ENDDO
             DO I=1,nmvsel
                DO J=1,nmvsel
                   DTHETADX(I,J)=0.0D0
                   DTHETADY(I,J)=0.0D0
                   DTHETADZ(I,J)=0.0D0
                   DRDX(I,J)=0.0D0
                   DRDY(I,J)=0.0D0
                   DRDZ(I,J)=0.0D0
                ENDDO
                IF (.NOT. QPROVIDE) THEN
                    CENTERX(I)=0.0D0
                    CENTERY(I)=0.0D0
                ELSE
                   CENTERX(I)=READCX(I)
                   CENTERY(I)=READCY(I)
                ENDIF
             ENDDO
             IF (.NOT. QPROVIDE) THEN
                 CALL GETCIRCLE(CENTERX,CENTERY,&
                                 DTHETADX,DTHETADY,DTHETADZ,DRDX,DRDY,DRDZ)
             ENDIF
             IF (QWRTCENT) THEN
                 DO I=1,nmvsel
                    WRITE(17,990) CENTERX(I),CENTERY(I)
                 ENDDO
990  FORMAT(f8.5,2x,f8.5)
             ENDIF
             IF (.NOT. QSMALL) THEN
                CALL CALC_CSAPE_SPL(SUP,C4_CSAPE,C3_CSAPE, &
                                 C2_CSAPE,MEMSPLUP)
                CALL CALC_CSAPE_SPL(SLO,C4_CSAPE,C3_CSAPE, &
                                 C2_CSAPE,MEMSPLLO)
             ELSE
                CALL CALC_CSAPE_SPL(SUP,C4_CSAPE_SMALL,C3_CSAPE_SMALL, &
                                 C2_CSAPE_SMALL,MEMSPLUP)
                CALL CALC_CSAPE_SPL(SLO,C4_CSAPE_SMALL,C3_CSAPE_SMALL, &
                                 C2_CSAPE_SMALL,MEMSPLLO)
             ENDIF
             SUMZI=0.0D0
             SUMSU=0.0D0
             SUMSL=0.0D0
             SUMEPS=0.0D0
             SUMGAM=0.0D0
             DO I=1,nmvsel
               IF (.NOT. QSMALL) THEN
                CALL CalsFHDGB(X(I),Y(I),test(i),CENTERX(I),CENTERY(I), &
                               SUP, &
                               MEMSPLUP,SFHDGB_OLD(I),DS_DTHETA(I), &
                               DS_DSUP,C4_CSAPE,C3_CSAPE, &
                               C2_CSAPE,C1_CSAPE)
                CALL CalsFHDGB(X(I),Y(I),test(i),CENTERX(I),CENTERY(I), &
                               SLO, &
                               MEMSPLLO,SFHDGBLO_OLD(I),DS_DTHETALO(I), &
                               DS_DSLO,C4_CSAPE,C3_CSAPE, &
                               C2_CSAPE,C1_CSAPE)
               ELSE
                CALL CalsFHDGB(X(I),Y(I),test(i),CENTERX(I),CENTERY(I), &
                               SUP, &
                               MEMSPLUP,SFHDGB_OLD(I),DS_DTHETA(I), &
                               DS_DSUP,C4_CSAPE_SMALL,C3_CSAPE_SMALL, &
                               C2_CSAPE_SMALL,C1_CSAPE_SMALL)
                CALL CalsFHDGB(X(I),Y(I),test(i),CENTERX(I),CENTERY(I), &
                               SLO, &
                               MEMSPLLO,SFHDGBLO_OLD(I),DS_DTHETALO(I), &
                               DS_DSLO,C4_CSAPE_SMALL,C3_CSAPE_SMALL, &
                               C2_CSAPE_SMALL,C1_CSAPE_SMALL)
!                 write(outu,*)'SFHDGB=',SFHDGB_OLD(I)
!                 write(outu,*)'SFHDGBLO=',SFHDGBLO_OLD(I)
               ENDIF
                R_DHDGB=DSQRT((X(I)-CENTERX(I))**2+(Y(I)-CENTERY(I))**2)
                CALL CALC_S_LAMBDA(R_DHDGB, &
                                   SFHDGB_OLD(I),AVGSDEF,INNERC,LAMF, &
                                   SFHDGB(I),LAM1_DHDGB(I))
                CALL CALC_S_LAMBDA(R_DHDGB, &
                                   SFHDGBLO_OLD(I),AVGSDEFL,INNERC,LAMF, &
                                   SFHDGBLO(I),LAM1L_DHDGB(I))
                DLAM1_DR_DHDGB(I)=-LAMF*(1.0D0/LAM1_DHDGB(I)-1.0D0)/ &
                                   (1.0D0/LAM1_DHDGB(I)**2)
                DLAM1L_DR_DHDGB(I)=-LAMF*(1.0D0/LAM1L_DHDGB(I)-1.0D0)/ &
                                   (1.0D0/LAM1L_DHDGB(I)**2)
! I may have to double check this!
! TEST
!                do j=1,INCT_FHDGB-1
!                   write(outu,*) 'DS_DSUP(J)=',DS_DSUP(J)
!                   write(outu,*) 'DS_Dlo(J)=',DS_DSLO(J)
!                enddo
                DO J=1,INCT_FHDGB-1
                   DSI_DSJ(I,J)=(1.0D0-LAM1_DHDGB(I))*DS_DSUP(J)+ &
                   LAM1_DHDGB(I)*1.0D0/(NDT_FHDGB)
                   IF ( J .GT. NDT_FHDGB) THEN
                       DSI_DSJ(I,J)=0.D0 
                   ENDIF
                   DS_DSUP(J)=0.0D0
                ENDDO
                DO J=1,INCT_FHDGB-1
                   DSI_DSJ(I,J+INCT_FHDGB-1)=(1.0D0-LAM1L_DHDGB(I))*DS_DSLO(J)+ &
                   LAM1L_DHDGB(I)*1.0D0/(NDT_FHDGB)
                   IF ( J .GT. NDT_FHDGB) THEN
                       DSI_DSJ(I,J+INCT_FHDGB-1)=0.D0
                   ENDIF 
                   DS_DSLO(J)=0.0D0
                ENDDO
!TEST
!                if (i .eq. 1) then
!                    do j=1,10
!                       write(outu,*) DSI_DSJ(i,j)      
!                    enddo      
!                endif   
!___________CALCULATE THE PROFILES FOR NONDEFORM MEMBRANE________
                CALL CalEpsHDGB(EPS_S0_HDGB(i), DEPS_S0_HDGB(i), &
                                Z(i),EPS0_HDGB, EPSW_HDGB,EPSMIN_HDGB,EPSMAX_HDGB,i)
                CALL CalEpsHDGB(EPS_S0_HDGBL(i), DEPS_S0_HDGBL(i), &
                                -Z(i),EPS0_HDGB, EPSW_HDGB,EPSMIN_HDGB,EPSMAX_HDGB,i)
                CALL CalSurfTspl_HDGB(SurfT_S0_HDGB(i), &
                                      DSurfT_S0_HDGB(i), &
                                Z(i),NP0_HDGB,NPW_HDGB,NPMIN_HDGB ,GMASP(i),i)
                IF (GMASP(i) .NE. 0.0D0) THEN
                    SurfT_S0_HDGB(i)=SurfT_S0_HDGB(i)/GMASP(i)
                    DSurfT_S0_HDGB(i)=DSurfT_S0_HDGB(i)/GMASP(i)
                ENDIF
                CALL CalSurfTspl_HDGB(SurfT_S0_HDGBL(i), &
                                      DSurfT_S0_HDGBL(i), &
                                      -Z(i),NP0_HDGB,NPW_HDGB,NPMIN_HDGB ,GMASP(i),i)
                IF (GMASP(i) .NE. 0.0D0) THEN
                    SurfT_S0_HDGBL(i)=SurfT_S0_HDGBL(i)/GMASP(i)
                    DSurfT_S0_HDGBL(i)=DSurfT_S0_HDGBL(i)/GMASP(i)
                ENDIF
!________________________________________________________________
!CALCULATE THE COEFFICENTS AND DERIVATIVES OF THE DEFORMATION EQUATION
               CALL CAL_COEFF(EPS_DEF_A_SPL,ZMIN_EPS_DEF_A, &
                              ZMAX_EPS_DEF_A,MIN_EPS_DEF_A, &
                              MAX_EPS_DEF_A,Z(I), &
                              HSPL_EPS_DEF_A,A_EPS(I),MA_EPS(I), &
                              DA_EPS_DZ(I),MDA_EPS_DZ(I),0.0D0)
               CALL CAL_COEFF(EPS_DEF_C_SPL,ZMIN_EPS_DEF_C, &
                              ZMAX_EPS_DEF_C,MIN_EPS_DEF_C, &
                              MAX_EPS_DEF_C,Z(I), &
                              HSPL_EPS_DEF_C,C_EPS(I),MC_EPS(I), &
                              DC_EPS_DZ(I),MDC_EPS_DZ(I),SEPS_C)
               CALL CAL_COEFF(EPS_DEF_E_SPL,ZMIN_EPS_DEF_E, &
                              ZMAX_EPS_DEF_E,MIN_EPS_DEF_E, &
                              MAX_EPS_DEF_E,Z(I), &
                              HSPL_EPS_DEF_E,E_EPS(I),ME_EPS(I), &
                              DE_EPS_DZ(I),MDE_EPS_DZ(I),0.0D0)
               CALL CAL_COEFF(NP_DEF_A_SPL,ZMIN_NP_DEF_A, &
                              ZMAX_NP_DEF_A,MIN_NP_DEF_A, &
                              MAX_NP_DEF_A,Z(I), &
                              HSPL_NP_DEF_A,A_NP(I),MA_NP(I), &
                              DA_NP_DZ(I),MDA_NP_DZ(I),0.0D0)
               CALL CAL_COEFF(NP_DEF_C_SPL,ZMIN_NP_DEF_C, &
                              ZMAX_NP_DEF_C,MIN_NP_DEF_C, &
                              MAX_NP_DEF_C,Z(I), &
                              HSPL_NP_DEF_C,C_NP(I),MC_NP(I), &
                              DC_NP_DZ(I),MDC_NP_DZ(I),SNP_C)
               CALL CAL_COEFF(NP_DEF_E_SPL,ZMIN_NP_DEF_E, &
                              ZMAX_NP_DEF_E,MIN_NP_DEF_E, &
                              MAX_NP_DEF_E,Z(I), &
                              HSPL_NP_DEF_E,E_NP(I),ME_NP(I), &
                              DE_NP_DZ(I),MDE_NP_DZ(I),0.0D0)
!____CALCULATE THE DEFORM EPS AND GAMMA AND THE DERIVATIVES (DZ)______
               CALL CALPRO_DEF(P_DHDGB,Z(I),A_EPS(I), &
                               C_EPS(I),E_EPS(I), &
                               EPS_DEF_B,0.0D0, &
                               MA_EPS(I),MC_EPS(I),ME_EPS(I), &
                               DA_EPS_DZ(I), &
                               DC_EPS_DZ(I), &
                               DE_EPS_DZ(I), &
                               MDA_EPS_DZ(I), &
                               MDC_EPS_DZ(I), &
                               MDE_EPS_DZ(I), &
                               SFHDGB(I),SFHDGBLO(I), &
                               EPS_S0_HDGB(i),DEPS_S0_HDGB(I), &
                               EPS_S0_HDGBL(i),DEPS_S0_HDGBL(I), &
                               EPS_HDGB(i),DEPS_HDGB(I), &
                               DEPSP(I),DEPSPL(I))
               CALL CALPRO_DEF(P_DHDGB,Z(I),A_NP(I), &
                               C_NP(I),E_NP(I),NP_DEF_B, &
                               1.50D0, &
                               MA_NP(I),MC_NP(I),ME_NP(I), &
                               DA_NP_DZ(I), &
                               DC_NP_DZ(I), &
                               DE_NP_DZ(I), &
                               MDA_NP_DZ(I), &
                               MDC_NP_DZ(I), &
                               MDE_NP_DZ(I), &
                               SFHDGB(I),SFHDGBLO(I), &
                               SurfT_S0_HDGB(i),DSurfT_S0_HDGB(i), &
                               SurfT_S0_HDGBL(i),DSurfT_S0_HDGBL(i), &
                               SurfT_HDGB(i),DSurfT_HDGB(i), &
                               DGAMMAP(I),DGAMMAPL(I))
               DGAMMAP(I)=DGAMMAP(I)*GMASP(i)
               SurfT_HDGB(i)=SurfT_HDGB(i)*GMASP(i)
               DSurfT_HDGB(i)=DSurfT_HDGB(i)*GMASP(i)
               DGAMMAPL(I)=DGAMMAPL(I)*GMASP(i)
               IF (QWRITES) THEN
                   COUNTATM=ATMEND-STARTATM+1
                   IF ((I .GE. STARTATM) .AND. &
                       (I .LE. ATMEND)) THEN
                       SUMZI=SUMZI+Z(I)/COUNTATM
                       SUMSU=SUMSU+SFHDGB(I)/COUNTATM
                       SUMSL=SUMSL+SFHDGBLO(I)/COUNTATM
                       SUMEPS=SUMEPS+EPS_HDGB(i)/COUNTATM
                       SUMGAM=SUMGAM+(SurfT_HDGB(i)/GMASP(i))/COUNTATM
                   ENDIF
               ENDIF
             ENDDO
             IF (QWRITES) THEN
                WRITE(145,145)SUMZI,SUMSU,SUMSL,SUMEPS,SUMGAM
145  FORMAT(5(F8.5,2X))
             ENDIF
         ELSE
#endif
#if KEY_DHDGB==1
!AP/MF
           DO i = 1, nmvsel
              CALL CalEpsHDGB(EPS_HDGB(i), DEPS_HDGB(i), Z(i),  &
                   EPS0_HDGB, EPSW_HDGB, EPSMIN_HDGB, EPSMAX_HDGB,i)
              CALL CalSurfTspl_HDGB(SurfT_HDGB(i), DSurfT_HDGB(i), &
                   Z(i),NP0_HDGB,NPW_HDGB,NPMIN_HDGB,GMASP(i),i)  ! YMC/MF
           ENDDO
        ENDIF
#else /**/
          DO i = 1, nmvsel
             CALL CalEpsHDGB(EPS_HDGB(i), DEPS_HDGB(i), Z(i),  &
                  EPS0_HDGB, EPSW_HDGB, EPSMIN_HDGB, EPSMAX_HDGB,i)
             CALL CalSurfTspl_HDGB(SurfT_HDGB(i), DSurfT_HDGB(i), &
                  Z(i),NP0_HDGB,NPW_HDGB,NPMIN_HDGB,GMASP(i),i)  ! YMC/MF
          ENDDO
#endif
          ! SJT/MF HDGB2
       else if (CORR_GB .eq. 4) then
          DO i = 1, nmvsel
             RDIST_HDGB(i) = DSQRT(X(i)**2+Y(i)**2+Z(i)**2)
             CALL CalEpsHDGB(EPS_HDGB(i), DEPS_HDGB(i), RDIST_HDGB(i), &
                  EPS0_HDGB, EPSW_HDGB, EPSMIN_HDGB, EPSMAX_HDGB,i)
             CALL CalSurfTspl_HDGB(SurfT_HDGB(i), DSurfT_HDGB(i), &
                  RDIST_HDGB(i),NP0_HDGB,NPW_HDGB,NPMIN_HDGB,GMASP(i),i) !YMC/MF 
          ENDDO
       else if (CORR_GB .eq. 5) then
          DO i = 1, nmvsel
             RDIST_HDGB(i) = DSQRT(X(i)**2+Y(i)**2)
             CALL CalEpsHDGB(EPS_HDGB(i), DEPS_HDGB(i), RDIST_HDGB(i), &
                  EPS0_HDGB, EPSW_HDGB, EPSMIN_HDGB, EPSMAX_HDGB,i)
             CALL CalSurfTspl_HDGB(SurfT_HDGB(i), DSurfT_HDGB(i), &
                  RDIST_HDGB(i),NP0_HDGB,NPW_HDGB,NPMIN_HDGB,GMASP(i),i) !YMC/MF
          ENDDO
          ! SJT/MF HDGB2
       endif
       I2 = 0
#if KEY_PARALLEL==1
       DO i = MyNodP, nmvsel, NumNod
#else /**/
       DO i=1,nmvsel
#endif 
          CX = X(i) - XMIN
          CY = Y(i) - YMIN
          CZ = Z(i) - ZMIN

          RI = Rad(i)
          RE = ONE / RI
          RG = HALF / (RI*RI)
          RH = ONE / (FOUR*(RI*RI*RI*RI))
          Area1 = FOURPI*(WMAIN(i)+1.4D0)*(WMAIN(i)+1.4D0)*WtScale2
          Surface_Area = Surface_Area + Area1/WtScale2
          Surf5 = Area1/WtScale2
          ! SJT/MF
          AtSA_HDGB(i) = Area1/WtScale2
          Area1_HDGB = TWO*FOURPI*(WMAIN(i)+1.4D0)*WtScale2
          S2_HDGB  = ZERO
          VZ1_HDGB = ZERO

          XE1 = (X(i)-XMIN)
          XE2 = (XMAX-X(i))
          YE1 = (Y(i)-YMIN)
          YE2 = (YMAX-Y(i))
          ZE1 = (Z(i)-ZMIN)
          ZE2 = (ZMAX-Z(i))

          IF (XE1 .gt. XE2) THEN
             RMAX = XE1*XE1
          ELSE
             RMAX = XE2*XE2
          ENDIF

          IF (YE1 .gt. YE2) THEN
             RMAX = RMAX + YE1*YE1
          ELSE
             RMAX = RMAX + YE2*YE2
          ENDIF

          IF (ZE1 .gt. ZE2) THEN
             RMAX = RMAX + ZE1*ZE1
          ELSE
             RMAX = RMAX + ZE2*ZE2
          ENDIF

          RMAX = DSQRT(RMAX)

          ! Do Solvent-Accessible-Surface-Area (SASA) term

          IF (SA_GB .NE. ZERO) THEN
             R4 = (WMAIN(i)+1.4D0) * WtScale
             R4_HDGB = WMAIN(i)+1.4D0
             DO j=1,NAngular
                Rx = R4 * WTX(j) + cx
                Ry = R4 * WTY(j) + cy
                Rz = R4 * WTZ(j) + cz
                IX = Rx * dn_inv
                IY = Ry * dn_inv
                IZ = Rz * dn_inv
                R7 = ZERO
                IF ((IX .ge. 0) .and. (IY .ge. 0) .and.  &
                     (IZ .ge. 0) .and. &
                     (IX .lt. Nx) .and. (IY .lt. Ny)  &
                     .and. (IZ .lt. Nz)) THEN
                   POS = IX + IY * Nx + IZ * Nxy - OFFSET
                   RX = RX + XMIN
                   RY = RY + YMIN
                   RZ = RZ + ZMIN
                   PS = Index(POS)
                   k  = AList(PS)
                   num = PTR(POS)+PS
                   ctr = 0
                   DO WHILE ((ps .lt. num) .and. (R7 .lt. TWO))
                      IF (k .ne. i) THEN
                         VX = RX - X(k)
                         VY = RY - Y(k)
                         VZ = RZ - Z(k)
                         R2 = VX * VX + VY * VY + VZ * VZ
                         X4 = WMAIN(k)+SON_GB
                         X4 = X4*X4
                         IF (R2 .LT. X4) THEN
                            R7 = TWO ! Hard-sphere
                            ctr = 0
                         ELSE
                            X3 = WMAIN(k)+SOFF_GB
                            X3 = X3*X3
                            IF (R2 .LT. X3) THEN ! Accumulate atoms
                               ctr = ctr + 1
                               X5 = ONE/(X3-X4)
                               X2 = (R2-X4)*X5
                               X1 = X2*X2
                               X3 = X1*x2
                               FSw_X = ONE +  &
                                    X3*(X2*(15.0D0-SIX*X2)-TEN)
                               DfSw_X = SIXTY*X1*(TWO*X2-X1-ONE)*X5
                               R7 = R7 + TWO*FSw_X
                               VX1(ctr) = VX
                               VY1(ctr) = VY
                               VZ1(ctr) = VZ
                               BX_GB(CTR) = DfSw_X
                               BX_HDGB(CTR) = ((WMAIN(k)+SON_GB) &
                                    +X2*(SOFF_GB-SON_GB))*DfSw_X
                               TEMP1_GB(ctr) = k
                            ENDIF
                         ENDIF ! Within hard shell
                      ENDIF
                      PS = PS + 1 ! move to next atom
                      IF (PS .lt. num) k = AList(PS)
                   ENDDO      !Loop over atoms
#if KEY_GBMVDEBUG==1
                   IF (EVDW_GB) THEN
                      IF (R7.LT.ONE) WRITE(1,'(4(F10.5,1X))')  &
                           RX,RY,RZ,ONE-R7
                   ENDIF
#endif 

                   IF (R7 .GT. ZERO) THEN

                      IF (R7 .LT. ONE) THEN
                         X2 = R7
                         X1 = X2*X2
                         X3 = X1*X2
                         ! S  = ONE + X3*(X2*(15.0D0-SIX*X2)-TEN)
                         S  = -X3*(X2*(15.0D0-SIX*X2)-TEN)
                         S1 = -SIXTY*X1*(TWO*X2-X1-ONE)
                      ELSE
                         S = ONE
                         S1 = ZERO
                      ENDIF

                      ! SJT/MF
                      Surface_Area = Surface_Area - Area1*S*WT1(J)
                      Surf5 = Surf5 - Area1*S*WT1(J)
                      if ((CORR_GB .eq. 3) .or. &
                           (CORR_GB .eq. 4) .or. &
                           (CORR_GB .eq. 5)) THEN
                         S1 = S1*WT1(J)*Area1*SurfT_HDGB(I)
                         AtSA_HDGB(I)=AtSA_HDGB(I)-Area1*S*WT1(J)
                         if (QHDGBRC) then
                            S2_HDGB = S2_HDGB &
                                 - WT1(J)*Area1_HDGB*S
                         endif
                      else if (QGBVDW) then
                         S1 = S1*WT1(J)*Area1*GMVDW(I)
                      else if (QGBASP) then
                         S1 = S1*WT1(J)*Area1*GMASP(I)
                      else
                         S1 = S1*WT1(J)*Area1*SA_GB
                      endif

                      IF (CTR .gt. 0) THEN
                         DO II=1,ctr
                            k2 = TEMP1_GB(ii)
                            S = S1 * BX_GB(ii)
                            DX_GB(i) = DX_GB(i) - S*VX1(ii)
                            DY_GB(i) = DY_GB(i) - S*VY1(ii)
                            DZ_GB(i) = DZ_GB(i) - S*VZ1(ii)
                            DX_GB(k2) = DX_GB(k2) + S*VX1(ii)
                            DY_GB(k2) = DY_GB(k2) + S*VY1(ii)
                            DZ_GB(k2) = DZ_GB(k2) + S*VZ1(ii)

                            IF (QHDGBRC) THEN
                               VZ1_HDGB = VZ1_HDGB &
                                    - S &
                                    *(VX1(ii)*WTX(j)+VY1(ii)*WTY(j) &
                                    +VZ1(ii)*WTZ(j))*WtScale
                               DZ_GB(k2)=DZ_GB(k2) &
                                    +S1*BX_HDGB(ii)*DR_HDGB(k2)
                            ENDIF
                         ENDDO
                      ENDIF
                   ENDIF
                ENDIF         !Inside grid
             ENDDO            !Do j=1,Nangular
             ! SJT/MF
             if (CORR_GB .eq. 3) then
                Gnp_HDGB = Gnp_HDGB + SurfT_HDGB(i)*AtSA_HDGB(i)
                DZ_GB(i) = DZ_GB(i) + DSurfT_HDGB(i)*AtSA_HDGB(i)
#if KEY_DHDGB==1
!AP/MF
                IF (QFHDGB) THEN
                    DX_DEF(I)=0.D0
                    DY_DEF(I)=0.D0
                    DO J=1,INCT_FHDGB-1
                       DGnp_DS_ATM(I,J) = AtSA_HDGB(i)* &
                                          DGAMMAP(I)*DSI_DSJ(I,J)
                    ENDDO
                    DO J=INCT_FHDGB,TOTALS
                       DGnp_DS_ATM(I,J) = AtSA_HDGB(i)* &
                                          DGAMMAPL(I)*DSI_DSJ(I,J)
                    ENDDO
                ENDIF

#endif
                if (QHDGBRC) then
                   DZ_GB(i) = DZ_GB(i) &
                        + (EIGHT*PI*R4_HDGB+S2_HDGB) &
                        * SurfT_HDGB(i) * DR_HDGB(i) &
                        + VZ1_HDGB * DR_HDGB(i)
                endif
                ! SJT/MF HDGB2
             else if (CORR_GB .eq. 4) then
                Gnp_HDGB = Gnp_HDGB + SurfT_HDGB(i)*AtSA_HDGB(i)
                X0_HDGB = DSurfT_HDGB(i)*AtSA_HDGB(i)/RDIST_HDGB(i)
                DX_GB(i) = DX_GB(i) + X0_HDGB*X(i)
                DY_GB(i) = DY_GB(i) + X0_HDGB*Y(i)
                DZ_GB(i) = DZ_GB(i) + X0_HDGB*Z(i)
             else if (CORR_GB .eq. 5) then
                Gnp_HDGB = Gnp_HDGB + SurfT_HDGB(i)*AtSA_HDGB(i)
                X0_HDGB = DSurfT_HDGB(i)*AtSA_HDGB(i)/RDIST_HDGB(i)
                DX_GB(i) = DX_GB(i) + X0_HDGB*X(i)
                DY_GB(i) = DY_GB(i) + X0_HDGB*Y(i)
                ! SJT/MF HDGB2
             endif

             if (QGBVDW) then
                Gnp_VDW = Gnp_VDW + Surf5*GMVDW(i)
             else if (QGBASP) then
                Gnp_ASP = Gnp_ASP + Surf5*GMASP(i)
             endif
          ENDIF               ! Do SASA
#if KEY_GBMVDEBUG==1
          IF (ESURF_GB) WRITE(2,'(I4,I4,I4,1X,F10.6)')  & 
#endif
#if KEY_GBMVDEBUG==1
               I,IAC(I),ITC(IAC(I)),Surf5 
#endif
          ! Do GB alpha part

          DO j=1,NWeights
             I2 = I2 + 1
             IF ((DRSTEP .EQ. 1).OR. &
                  (SURF_ATOM(I2).EQ.-1)) THEN
                R = WTS(j)
                S = ZERO
                IF ((R .ge. RI).and.(R .lt. RMAX)) THEN
                   ! IF (R .ge. RI) THEN
                   Rx = WtX(j) + cx
                   Ry = WtY(j) + cy
                   Rz = WtZ(j) + cz
                   IX = Rx * dn_inv
                   IY = Ry * dn_inv
                   IZ = Rz * dn_inv
                   R1 = ZERO
                   IF ((IX .ge. 0) .and. (IY .ge. 0) .and.  &
                        (IZ .ge. 0) .and. &
                        (IX .lt. Nx) .and. (IY .lt. Ny) .and.  &
                        (IZ .lt. Nz)) THEN
                      POS = IX + IY * Nx + IZ * Nxy - OFFSET
                      RX = RX + XMIN
                      RY = RY + YMIN
                      RZ = RZ + ZMIN
                      ctr = 0
                      R1 = ZERO
                      R7 = ZERO
                      PS = Index(POS)
                      k  = AList(PS)
                      num = PTR(POS)+PS
                      ! WRITE(OUTU,*)i,POS,k,PS
                      IF (P3 .LE. ZERO) THEN ! Do Original GBMV
                         DO WHILE ((PS .lt. num) .and.  &
                              (R1 .lt. thresh2))
                            l = l + 1
                            VX = RX - X(k)
                            VY = RY - Y(k)
                            VZ = RZ - Z(k)
                            R2 = VX * VX + VY * VY + VZ * VZ
                            R3 = GAMMA(k)*r2*r2
                            IF (R3 .GT. -23) THEN
                               R1 = R1 + dexp(R3)
                            ENDIF
                            PS = PS + 1
                            if (PS .lt. num) k = AList(PS)
                         ENDDO
                      ELSE    ! Do VSA
                         DO WHILE ((ps .lt. num) .and. (R7 .lt. TWO))
                            VX = RX - X(k)
                            VY = RY - Y(k)
                            VZ = RZ - Z(k)
                            R2 = VX * VX + VY * VY + VZ * VZ
                            X1 = WMAIN(k)+HSX1
                            X1 = X1*X1
                            IF (R2 .LT. X1) THEN
                               R7 = TWO ! Hard-sphere
                               ctr = 0
                            ELSE
                               X1 = WMAIN(k)+Off_X
                               X1 = X1*X1
                               IF (R2 .LT. X1) THEN ! Accumulate atoms
                                  ctr = ctr + 1 ! that exist in VSA region
                                  VX1(ctr)=VX
                                  VY1(ctr)=VY
                                  VZ1(ctr)=VZ
                                  RA(ctr)=R2
                                  RB(ctr)=WMAIN(k)
                               ENDIF
                               PS = PS + 1 ! move to next atom
                               if (ps .lt. num) k = AList(PS)
                            ENDIF
                         ENDDO
                         ! Check i nter-atom region
                         IF (ctr.gt.0) THEN
                            l = l + ctr
                            R1 = ZERO
                            R1A = ZERO
                            SX = ZERO
                            SY = ZERO
                            SZ = ZERO
                            R7 = ZERO
                            DO ii=1,ctr
                               X1 = RB(ii) ! R(atom)
                               R2 = RA(ii) ! R2

                               IF (CUBIC_VSA_GB .ne. 0) THEN
                                  T1 = (X1+Off_X)*(X1+Off_X)
                                  S2 = (R2-T1)/(X1*X1-T1)
                                  R3 = S2*S2*S2
                               ELSE
                                  X5 = P2*X1 + P1 ! P2 * R(ii) + P1
                                  X4 = X1*X1
                                  X3 = R2 - X4
                                  X6 = X5+X3
                                  R3 = X5/X6
                                  R3 = R3*R3
                               ENDIF

                               ! Hard- sphere tail and start of VSA
                               X3 = X1+HSX2
                               X3 = X3*X3
                               IF (R2 .LT. X3) THEN
                                  X4 = X1+HSX1
                                  X4 = X4*X4
                                  X2 = (R2-X4)/(X3-X4)
                                  X3 = X2*X2*x2
                                  FSw_X = ONE  &
                                       + X3*(X2*(15.0D0-SIX*X2)-TEN)
                                  R7 = R7 + TWO*FSw_X
                                  R3 = R3 * (ONE-FSw_X)
                               ELSE
                                  IF (CUBIC_VSA_GB .eq. 0) THEN
                                     X4 = X1+On_X ! R + On
                                     X4 = X4*X4
                                     IF (R2 .GT. X4) THEN ! VSA Tail
                                        X2 = X1+Off_X ! R + Off_X
                                        X2 = X2*X2
                                        X3 = (R2-X4)/(X2-X4)
                                        X2 = X3*X3*X3
                                        FSw_X = ONE +  &
                                             X2* &
                                             (X3*(15.0D0-SIX*X3)-TEN)
                                        R3 = R3 * FSw_X
                                     ENDIF
                                  ENDIF
                               ENDIF
                               R1 = R1 + R3
                               VX = VX1(ii)*R3
                               VY = VY1(ii)*R3
                               VZ = VZ1(ii)*R3
                               SX = SX + VX
                               SY = SY + VY
                               SZ = SZ + VZ
                               R1A = R1A + VX*VX + VY*VY + VZ*VZ
                            ENDDO
                            R4 = SX*SX + SY*SY + SZ*SZ
                            IF (R4 .LT. 1.0D-18) R4=1.0D-18
                            R1 = R1 * P3 * R1A/(R4+P7)
                         ENDIF
                         R1 = R1 + R7 ! Hard-sphere
                      ENDIF   ! Choice between original GBMV vs. VSA
                      R2 = ONE + DEXP(BETA * (R1-LAMBDA1))
                      S = ONE/R2
                      IF (S .GT. ZERO) THEN
                         RE = RE - S * Wt1(j)
                         RG = RG - S * Wt2(j)
                         RH = RH - S * Wt4(j)
                      ENDIF
                      IF (DRSTEP.EQ.1) THEN
                         IF ((R1 .GT. THRESH1).AND. &
                              (R1 .LT. THRESH2)) THEN
                            SURF_ATOM(I2) = -1
                         ELSE
                            SURF_ATOM(I2) = S + TOL_GB
                         ENDIF
                      ELSE
                         Surf_Atom(I2) = -1
                      ENDIF
#if KEY_GBMVDEBUG==1
                      ! IF (R1 .GT. THRESH1) WRITE(2,*) 'H',Rx,Ry,Rz  
#endif
                   ENDIF
                ENDIF
             ELSE             ! Re-use value of integration point
                IF (Surf_Atom(I2).GT. ZERO) THEN
                   RE = RE - Wt1(j)
                   RG = RG - Wt2(j)
                   RH = RH - Wt4(j)
                ENDIF
             ENDIF
          ENDDO               !DO j=1,NWeights

          ! SJT/MF
          IF (CORR_GB .EQ. 0) THEN
             F(i) = DSQRT(RG)
             Alph_Gb(i) = -ONE/(RE - TT * F(i) + ESHIFT) + SHIFT
          ELSEIF (CORR_GB .eq. 1) THEN
             F(i) = RH
             Alph_Gb(i) = SLOPE/((ONE-DSQRT(HALF))*RE+RH**(PT25))  &
                  + SHIFT
          ELSEIF (CORR_GB .eq. 2) THEN
             F(i) = DSQRT(RG)
             G(i) = RH
             Alph_Gb(i)=SLOPE/(A1*RE+A2*F(i)+A3*RH**(PT25)+ESHIFT) &
                  + SHIFT
          ELSEIF ((CORR_GB .eq. 3) .or. &
               (CORR_GB .eq. 4) .or. &
               (CORR_GB .eq. 5)) THEN
             ! Compute the Born radii via Eq. (15) in
             ! J.Chem.Phys. (2004) 120: 903.
             ! C0->A1 C1->A3 D->A4 E->A5
             F(i) = DSQRT(RG)
             G(i) = RH
             A3_HDGB(i) = A3 * THREE * EPS_HDGB(i) /  &
                  (THREE * EPS_HDGB(i) + TWO * EPS)
             SHIFT_HDGB(i) = A4 + A5/(EPS_HDGB(i) + ONE)
             Alph_Gb(i)      = SLOPE/(A1*RE+A3_HDGB(i)*RH**(PT25))  &
                  + SHIFT_HDGB(i)
             S_HDGB(i) = (Alph_Gb(i) - SHIFT_HDGB(i))**2 &
                  * RH**(PT25) &
                  * A3 * SIX * EPS  &
                  / (THREE * EPS_HDGB(i) + TWO * EPS)**2 &
                  + A5 / (EPS_HDGB(i) + ONE)**2
          ENDIF

#if KEY_GBMVDEBUG==1
          IF (ECOMP_GB) WRITE(1,210) RE,DSQRT(RG),DSQRT(DSQRT(RH)) 
#endif
       ENDDO                  !Do i=1, Natom

210    FORMAT(3(F25.21,1X))
       R1 = L
       R2 = NSURF
       !99   0   FORMAT('Weight # ',I4, ' Atom:',I4)
#if KEY_GBMVDEBUG==1
       !     IF (ECOMP_GB) CLOSE(1)  
#endif
#if KEY_GBMVDEBUG==1
       IF (EVDW_GB) CLOSE(1)  
#endif

#if KEY_PARALLEL==1
       ! WRITE(OUTU,990)MyNodP,NSurf,l
       ! 990   FORMAT('Node ',I2,': surface/volume points =',I10,I10)
#endif 

#if KEY_PARALLEL==1
       R1 = R1 * NumNod
       R2 = R2 * NumNod
#endif 
       R1 = R1/(NWeights*nmvsel)
       R2 = R2/(NWeights*nmvsel)*HUNDRD
#if KEY_GBMVDEBUG==1
       IF (PRNLEV .GE. 6) THEN
          WRITE(OUTU,1000) R1,R2
       ENDIF
1000   FORMAT ('# atoms per IP: ',F6.3,'    Surface: ',F6.2,'%')
#endif /*                         GMBMVDEBUG*/

#if KEY_PARALLEL==1
       ! CALL GComb(F,NAtom)
       CALL GComb(Alph_Gb,nmvsel)
#endif 
    ENDIF                     !IF (QUpdateAlpha) THEN

    call timer_stop(T_GB_RADII) 

    if (Qhybrid) return  ! Hybrid

    !-------------------------------------------------
    ! Forming energy and some derivative components
    !-------------------------------------------------
    ! WRITE(OUTU,*) 'Forming energy...'

    call timer_start(T_GB_ENERG) 

    ! First, do excluded list
    SALT = Factor
    STILLFAC = ONE/P5
    GBEnergy = ZERO
    ITemp = 0
#if KEY_PARALLEL==1 /*paraexcl*/
    DO i=MyNodP, nmvsel, NumNod
#else /* (paraexcl)*/
    DO i=1,nmvsel
#endif /* (paraexcl)*/
       !AvdV..bugfix 16-Jul-2004
       If (i .ne. 1) ITemp = INblo14(i-1)
       !AvdV..
       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          IF (k .gt. 0) THEN
             LOK = .TRUE.
             If (QInte) LOK = (Islct(i).eq.1).and.(Jslct(j).eq.1)
             IF (LOK) THEN
                XDIFF = X(i) - X(j)
                YDIFF = Y(i) - Y(j)
                ZDIFF = Z(i) - Z(j)
                R1 = XDIFF * XDIFF + YDIFF * YDIFF + ZDIFF * ZDIFF
                IF (MTYPE_GB .EQ. 2) THEN
                   A = Alph_Gb(i) * Alph_Gb(j)
                ELSE
                   A = Alph_Gb(i) + Alph_Gb(j)
                   A = A*A/FOUR
                ENDIF
                Q = CG(i) * CG(j)

                X3 = R1/A
                D = DEXP(-P6*X3)

                FGB = DSQRT(R1+A*D)
                C2 = ONE/FGB
                C1 = C2*C2


                IF (P4 .NE. ZERO) THEN
                   X1 = DSQRT(P5*A)
                   X2 = DSQRT(R1)
                   C3 = -P4*(R1*X2)*(ONE/(A*A))*DEXP(-X2/X1)
                ELSE
                   C3 = ZERO
                ENDIF

                IF (KAPPA_GB .ne. ZERO) THEN
                   C4 = DEXP(-KAPPA_GB*FGB)/EPS_GB
                   SALT = -CCELEC*(ONE/EPS - C4)
                ENDIF

                ! SJT/MF
                IF (MODSTILL.EQ.0) THEN
                   if ((CORR_GB .eq. 3) .or. &
                        (CORR_GB .eq. 4) .or. &
                        (CORR_GB .eq. 5)) then
                      EPSIJ_HDGB = HALF*(EPS_HDGB(i) + EPS_HDGB(j))
                      GBEnergy = GBEnergy  &
                           + Q * (C2+C3)* &
                           (-CCELEC * (ONE/EPS - ONE/EPSIJ_HDGB))
                      P_HDGB(i) = P_HDGB(i) + (ONE/EPSIJ_HDGB**2)*Q*C2
                      P_HDGB(j) = P_HDGB(j) + (ONE/EPSIJ_HDGB**2)*Q*C2
                   else
                      GBEnergy = GBEnergy + Q * (C2+C3)*SALT
                   endif
                ELSE
                   RR1=DSQRT(R1)
                   RRA=DSQRT(A)
                   FGB=DSQRT(R1+(A+MS1*R1*RR1/RRA+MS2*RR1*RRA &
                        +MS3*R1 )*D)
                   GBEnergy = GBEnergy + Q * MS0 / FGB *SALT
                ENDIF

                IF (KAPPA_GB .ne. ZERO) THEN
                   C2 = Q * HALF * C1 * (C2*SALT+CCELEC*KAPPA_GB*C4) &
                        + Q*C3*HALF*C2*CCELEC*KAPPA_GB*C4
                   !(1/2f)*Q*k*KAPPA*exp(KAP*F)/eps_gb
                ELSE if ((CORR_GB .eq. 3) .or. &
                         (CORR_GB .eq. 4) .or. &
                         (CORR_GB .eq. 5)) then
                      C2 = Q * C1 * C2 * Half *  &
                           (-CCELEC * (ONE/EPS - ONE/EPSIJ_HDGB))
                else
                      C2 = Q * C1 * C2 * Half * SALT
                ENDIF

                X4 = ONE + P6*X3

                C5 = C2 * D * X4
                IF (MTYPE_GB .EQ. 2) THEN
                   RI = C5 * Alph_Gb(j)
                   RJ = C5 * Alph_Gb(i)
                ELSE
                   TA2 = HALF*(Alph_Gb(i)+Alph_Gb(j))
                   RI = C5 * TA2
                   RJ = C5 * TA2
                ENDIF

                C2 = C2 * TWO * (P6*D-ONE)

                IF (P4 .NE. ZERO) THEN
                   X3 = Q * C3 * (X2/(TWO*A*X1)-TWO/A) * SALT
                   RI = RI - X3 * Alph_Gb(j)
                   RJ = RJ - X3 * Alph_Gb(i)
                   C3 = Q * TWO * C3 *  &
                        ((1.5D0/R1)-HALF/(X2*X1)) * SALT
                   C2 = C2 + C3
                ENDIF

                VX = XDIFF * C2
                VY = YDIFF * C2
                VZ = ZDIFF * C2
                T(i) = T(i) - RI
                DX(I) = DX(I) + VX
                DY(I) = DY(I) + VY
                DZ(I) = DZ(I) + VZ
                T(j) = T(j) - RJ
                DX(J) = DX(J) - VX
                DY(J) = DY(J) - VY
                DZ(J) = DZ(J) - VZ
                if(i == 1 .or. j == 1)then
                endif
             ENDIF
          ENDIF
       ENDDO
       !AvdV..bugfix 16-Jul-2004       ITemp = INblo14(i)
    ENDDO                     !DO i=1,Natom


    ! Next, Do non-bonded list

    C2OfNB = CtOfNB * CtOfNB
    FSw = ONE
    DFSw = ZERO
    IF (Switch) THEN
       C2OnNb = CtOnNb * CtOnNb
       If (CtOfNb .gt. CtOnNb) Then
          Rul3 = One / (C2OfNb - C2OnNb)**3
          Rul12 = Twelve * Rul3
       Endif
    Endif

    ITemp = 0
    DO i=1,nmvsel-1
#if KEY_IMCUBES==1
       if (lbycbim) ITemp = INbl(I+nmvsel) 
#endif
       if (CG(i).ne.ZERO) THEN
          NPr = INbl(i) - ITemp
          Do jpr = 1, NPr
             k = JNbl(Itemp+jpr)
             j = Abs(k)
             IF (CG(j) .ne. zero) THEN
                LOK = .TRUE.
                If (QInte) LOK = (Islct(i).eq.1).and.(Jslct(j).eq.1)
                IF (LOK) THEN
                   XDIFF = X(i) - X(j)
                   YDIFF = Y(i) - Y(j)
                   ZDIFF = Z(i) - Z(j)
                   R1 = XDIFF * XDIFF + YDIFF * YDIFF + ZDIFF * ZDIFF

                   ! The Electrostatic Switch Function

                   If (R1 .lt. C2OfNb) Then

                      FSw = One
                      DFSw = Zero
                      If (Switch) Then
                         LOuter = (R1 .gt. C2OnNb)
                         If (LOuter) Then
                            RijL = C2OnNb - R1
                            RijU = C2OfNb - R1
                            FSw = RijU * RijU *  &
                                 (RijU - Three * RijL) * Rul3
                            DfSw = RijL * RijU * Rul12 / FSw
                         Endif
                      Endif

                      IF (MTYPE_GB .EQ. 2) THEN
                         A = Alph_Gb(i) * Alph_Gb(j)
                      ELSE
                         A = Alph_Gb(i) + Alph_Gb(j)
                         A = A*A/FOUR
                      ENDIF

                      Q = CG(i) * CG(j)

                      X3 = R1/A
                      D = DEXP(-P6*X3)

                      FGB = DSQRT(R1+A*D)
                      C2 = ONE/FGB
                      C1 = C2*C2

                      IF (P4 .NE. ZERO) THEN
                         X1 = DSQRT(P5*A)
                         X2 = DSQRT(R1)
                         C3 = -P4*(R1*X2)*(ONE/(A*A))*DEXP(-X2/X1)
                      ELSE
                         C3 = ZERO
                      ENDIF

                      IF (KAPPA_GB .ne. ZERO) THEN
                         C4 = DEXP(-KAPPA_GB*FGB)/EPS_GB
                         SALT = -CCELEC*(ONE/EPS - C4)
                      ENDIF

                      ! SJT/MF
                      IF (MODSTILL.EQ.0) THEN
                         if ((CORR_GB .eq. 3) .or. &
                              (CORR_GB .eq. 4) .or. &
                              (CORR_GB .eq. 5)) then
                            EPSIJ_HDGB = HALF &
                                 * (EPS_HDGB(i)+EPS_HDGB(j))
                            S = Q * (C2+C3) * FSw * &
                                 (-CCELEC * (ONE/EPS - ONE/EPSIJ_HDGB))
                            GBEnergy = GBEnergy + S
                            P_HDGB(i) = P_HDGB(i)  &
                                 + (ONE/EPSIJ_HDGB**2)*FSw*Q*C2
                            P_HDGB(j) = P_HDGB(j)  &
                                 + (ONE/EPSIJ_HDGB**2)*FSw*Q*C2
                         else
                            S = Q * (C2+C3) * FSw * SALT
                            GBEnergy = GBEnergy + S
                         endif
                      ELSE
                         RR1=DSQRT(R1)
                         RRA=DSQRT(A)
                         FGB=DSQRT( R1+ &
                              (A+MS1*R1*RR1/RRA+MS2*RR1*RRA+MS3*R1)*D)
                         GBEnergy = GBEnergy +  &
                              Q * MS0 / FGB * FSw *SALT
                      ENDIF

                      IF (KAPPA_GB .ne. ZERO) THEN
                         C2 = Q * FSw * HALF * C1 *  &
                              (C2*SALT+CCELEC*KAPPA_GB*C4) &
                              + Q*C3*HALF*C2*CCELEC*KAPPA_GB*C4
                         !(1/2f)*Q*k*KAPPA*exp(KAP*F)/eps_gb
                      ELSE if ((CORR_GB .eq. 3) .or. &
                               (CORR_GB .eq. 4) .or. &
                               (CORR_GB .eq. 5)) then
                            C2 = Q * C1 * C2 * Half * FSw * &
                                 (-CCELEC * (ONE/EPS - ONE/EPSIJ_HDGB))
                      else
                            C2 = Q * C1 * C2 * Half * SALT * FSw
                      ENDIF

                      X4 = ONE + P6*X3

                      C5 = C2 * D * X4

                      IF (MTYPE_GB .EQ. 2) THEN
                         RI = C5 * Alph_Gb(j)
                         RJ = C5 * Alph_Gb(i)
                      ELSE
                         TA2 = HALF*(Alph_Gb(i)+Alph_Gb(j))
                         RI = C5 * TA2
                         RJ = RI
                      ENDIF

                      C2 = C2 * TWO * (P6*D-ONE)

                      IF (P4 .NE. ZERO) THEN
                         X3 = Q * C3 * (X2/(TWO*A*X1)-TWO/A)  &
                              * SALT * FSw
                         RI = RI - X3 * Alph_Gb(j)
                         RJ = RJ - X3 * Alph_Gb(i)
                         C3 = Q * TWO * C3 *  &
                              ((1.5D0/R1)-HALF/(X2*X1)) * SALT * FSw
                         C2 = C2 + C3
                      ENDIF

                      IF (Switch .and. LOuter) THEN
                         C2 = C2 + DFSw * S
                      ENDIF
                      VX = XDIFF * C2
                      VY = YDIFF * C2
                      VZ = ZDIFF * C2
                      T(i) = T(i) - RI
                      DX(I) = DX(I) + VX
                      DY(I) = DY(I) + VY
                      DZ(I) = DZ(I) + VZ
                      T(j) = T(j) - RJ
                      DX(J) = DX(J) - VX
                      DY(J) = DY(J) - VY
                      DZ(J) = DZ(J) - VZ
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
       ENDIF
       ITemp = INbl(i)
    ENDDO                     !DO i=1,Natom-1

    ! Finally, the self terms

#if KEY_PARALLEL==1
    DO i=MyNodP,nmvsel,NumNod
#else /**/
    DO i=1,nmvsel
#endif 
       IF (IMOVE(i).eq.0) THEN
          LOK = .true.
          IF (QInte) LOK = (Islct(i).eq.1).and.(Jslct(i).eq.1)
          IF (LOK) THEN
             Q = CG(i) * CG(i)
             C1 = ONE / Alph_Gb(i)
             IF (KAPPA_GB .ne. ZERO) THEN
                C4 = DEXP(-KAPPA_GB*Alph_Gb(i))/EPS_GB
                SALT = -CCELEC*(ONE/EPS - C4)
             ENDIF
             ! SJT/MF
             if ((CORR_GB .eq. 3) .or. &
                  (CORR_GB .eq. 4) .or. &
                  (CORR_GB .eq. 5)) then
                C2 = Q * abs(C1) * Half * &
                     (-CCELEC * (ONE/EPS - ONE/EPS_HDGB(i)))
                GBEnergy = GBEnergy + C2
                P_HDGB(i) = P_HDGB(i)  &
                     + (ONE/EPS_HDGB(i)**2)*Q*abs(C1)
             else
                C2 = Q * abs(C1) * Half * SALT
                IF(MODSTILL.EQ.0) THEN
                   GBEnergy = GBEnergy + C2
                ELSE
                   GBEnergy = GBEnergy + MS0 * C2
                ENDIF
             endif

             RI = C1 * C2
             IF (KAPPA_GB .ne. ZERO) THEN
                RI = RI + Q*HALF*CCELEC*(KAPPA_GB*C4) * C1
             ENDIF
             T(i) = T(i) - RI
          ENDIF
       ENDIF
    ENDDO

    ! SJT/MF VDW
    !
    ! VDW DISPERSION TERM (May, 2006)
    !
    ! E. Gallicchio and RM Levy, J Comput Chem 25: 479--499, 2004
    !
#if KEY_HDGBVDW==1
! MS/MF

      IF (QGBVDW ) THEN
       IF ( (UNDN_HDGB .NE. -1) .or. &
          (UNDO_HDGB .NE. -1) .or. &
          (UNDC_HDGB .NE. -1) .or. &
          (UNDC2_HDGB .NE. -1) .or. &
          (UNDP_HDGB .NE. -1) ) then


!lipid H 85
         EPSNVDW = 0.03070D0
         SIGNVDW = 1.220D0

!lipid C tail 84
         EPSC2VDW = 0.057D0
         SIGC2VDW = 2.019D0
!lipid C 83
         EPSCVDW = 0.068D0
         SIGCVDW = 2.035D0
!water 81
         EPSOVDW = 0.1520D0
         SIGOVDW = 1.7700D0
!lipid O 82
         EPSPVDW = 0.11D0
         SIGPVDW = 1.675D0

         GVDW    = ZERO
#if KEY_PARALLEL==1
        DO i=MyNodP,nmvsel,NumNod
#else /**/
        DO I = 1, nmvsel
#endif
!         DDD_HDGB(I)=ZERO
!         WRITE(6,*) 'derivative>> ' ,I , DDD_HDGB(I)
           SIG_I = VDWR(ITC(IAC(I)))
           EPS_I = ABS(EFF(ITC(IAC(I))))
           IF (UNDN_HDGB .NE. -1) THEN
          ! write (*,*) 'rad check', Alph_GB(I),i
             CALL CalDenspl_HDGB(DN_HDGB(I),DDN_HDGB(I),Z(i), &
               DON_HDGB, DWN_HDGB,DMINN_HDGB,DMAXN_HDGB,UNDN_HDGB, &
               ZMIN_N,ZMAX_N,i)
             CALL CalVdw(EPSNVDW, SIGNVDW, EPS_I, SIG_I, &
               GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DN_HDGB(I), &
               UNDN_HDGB,Z(I),DDN_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DN_HDGB(I) .ne. 0 ).and.(DDN_HDGB(I) .ne. 0)) THEN
                DDD_HDGB(I)= DDD_HDGB(I) &
                       - GVDW_HDGB/DN_HDGB(I)*DDN_HDGB(I)

             ELSE
                DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF
           ENDIF

           IF (UNDC2_HDGB .NE. -1) THEN
             CALL CalDenspl_HDGB(DC2_HDGB(I),DDC2_HDGB(I),Z(i), &
              DOC2_HDGB, DWC2_HDGB,DMINC2_HDGB,DMAXC2_HDGB, &
              UNDC2_HDGB,ZMIN_C2,ZMAX_C2,i)
             CALL CalVdw(EPSC2VDW, SIGC2VDW, EPS_I, SIG_I, &
              GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DC2_HDGB(I), &
              UNDC2_HDGB,Z(I),DDC2_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DC2_HDGB(I) .ne. 0 ).and.(DDC2_HDGB(I) .ne. 0)) THEN
               DDD_HDGB(I)= DDD_HDGB(I) &
                       - GVDW_HDGB/DC2_HDGB(I)*DDC2_HDGB(I)
             ELSE
               DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF

           ENDIF

           IF (UNDC_HDGB .NE. -1) THEN
             CALL CalDenspl_HDGB(DC_HDGB(I),DDC_HDGB(I),Z(i), &
              DOC_HDGB, DWC_HDGB,DMINC_HDGB,DMAXC_HDGB, &
              UNDC_HDGB,ZMIN_C,ZMAX_C,i)
             CALL CalVdw(EPSCVDW, SIGCVDW, EPS_I, SIG_I, &
              GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DC_HDGB(I), &
              UNDC_HDGB,Z(I),DDC_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DC_HDGB(I) .ne. 0 ).and.(DDC_HDGB(I) .ne. 0)) THEN
                 DDD_HDGB(I)= DDD_HDGB(I) &
                         - GVDW_HDGB/DC_HDGB(I)*DDC_HDGB(I)
             ELSE
                 DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF
           ENDIF

           IF (UNDO_HDGB .NE. -1) THEN
             CALL CalDenspl_HDGB(DO_HDGB(I),DDO_HDGB(I),Z(i), &
              DOO_HDGB, DWO_HDGB,DMINO_HDGB,DMAXO_HDGB, &
              UNDO_HDGB,ZMIN_O,ZMAX_O,i)
             CALL CalVdw(EPSOVDW, SIGOVDW, EPS_I, SIG_I, &
              GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DO_HDGB(I), &
              UNDO_HDGB,Z(I),DDO_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DO_HDGB(I) .ne. 0 ).and.(DDO_HDGB(I) .ne. 0)) THEN
                 DDD_HDGB(I)= DDD_HDGB(I) &
                         - GVDW_HDGB/DO_HDGB(I)*DDO_HDGB(I)
             ELSE
                 DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF
           ENDIF

           IF (UNDP_HDGB .NE. -1) THEN
             CALL CalDenspl_HDGB(DP_HDGB(I),DDP_HDGB(I),Z(i), &
              DOP_HDGB, DWP_HDGB,DMINP_HDGB,DMAXP_HDGB, &
              UNDP_HDGB,ZMIN_P,ZMAX_P,i)
             CALL CalVdw(EPSPVDW, SIGPVDW, EPS_I, SIG_I, &
              GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DP_HDGB(I), &
              UNDP_HDGB,Z(I),DDP_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DP_HDGB(I) .ne. 0 ).and.(DDP_HDGB(I) .ne. 0)) THEN
                 DDD_HDGB(I)= DDD_HDGB(I) &
                         - GVDW_HDGB/DP_HDGB(I)*DDP_HDGB(I)
             ELSE
                 DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF
           ENDIF
!        WRITE(6,*) 'GB Energy-> ' ,GVDW

         END DO
!         WRITE(6,*) 'GB Energy>> ' ,GVDW

       ELSE

        X1VDW   = EIGHT*PI*(0.033428D0)/THREE
        EPSWVDW = 0.1520D0
        SIGWVDW = 1.7682D0
        GVDW    = ZERO
        DO I = 1, nmvsel
            EPSIW = SQRT(ABS(EFF(ITC(IAC(I))))*EPSWVDW)
            SIGIW = VDWR(ITC(IAC(I))) + SIGWVDW
            SIGIW = SIGIW*SIGIW
            SIGIW = SIGIW*SIGIW*SIGIW
            X2VDW = ONE/(Alph_GB(I)+1.4D0)
            X3VDW = X2VDW*X2VDW*X2VDW
            X4VDW = X1VDW*AVDW(I)*EPSIW*SIGIW
            GVDW  = GVDW &
                 - X4VDW*X3VDW
            TVDW(I) = THREE*X4VDW*X3VDW*X2VDW
            T(I) = T(I) + TVDW(I)
           WRITE(6,*) 'GB i ->  ' ,GVDW,I

        END DO

       ENDIF
      ENDIF
! MS/MF
#else

    IF (QGBVDW) THEN
       X1VDW   = EIGHT*PI*(0.033428D0)/THREE
       EPSWVDW = 0.1520D0
       SIGWVDW = 1.7682D0
       GVDW    = ZERO
       DO I = 1, nmvsel
          EPSIW = SQRT(ABS(EFF(ITC(IAC(I))))*EPSWVDW)
          SIGIW = VDWR(ITC(IAC(I))) + SIGWVDW
          SIGIW = SIGIW*SIGIW
          SIGIW = SIGIW*SIGIW*SIGIW
          X2VDW = ONE/(Alph_GB(I)+1.4D0)
          X3VDW = X2VDW*X2VDW*X2VDW
          X4VDW = X1VDW*AVDW(I)*EPSIW*SIGIW
          GVDW  = GVDW  &
               - X4VDW*X3VDW
          TVDW(I) = THREE*X4VDW*X3VDW*X2VDW
          T(I) = T(I) + TVDW(I)
       END DO
    END IF

    ! SJT VDW
!MS/MF
#endif

    ! SJT/MF
#if KEY_PARALLEL==1
    CALL GComb(T,nmvsel)
    if ((CORR_GB .eq. 3) .or. &
         (CORR_GB .eq. 4) .or. &
         (CORR_GB .eq. 5)) then
       CALL GComb(P_HDGB,nmvsel)
    endif
#endif 

    ! WRITE(OUTU,*) 'GB Energy =',GBEnergy

    call timer_stop(T_GB_ENERG) 

    IF (QUpdateAlpha) THEN
       !------------------------------------------
       ! Compute forces
       !------------------------------------------

       call timer_start(T_GB_FORCE) 

       DFSw_X = ZERO
       FSw_X = ONE
       l = 0

       I2 = 0
#if KEY_PARALLEL==1
       DO i=MyNodP, nmvsel, NumNod
#else /**/
       DO i=1,nmvsel
#endif 
          CX = X(i) - XMIN
          CY = Y(i) - YMIN
          CZ = Z(i) - ZMIN

          E_ALPHA = T(i)


          ! SJT/MF
          if (CORR_GB .eq. 3) then
             DZ(i) = DZ(i) - HALF*CCELEC*P_HDGB(i)*DEPS_HDGB(i) &
                  - E_ALPHA * S_HDGB(i) * DEPS_HDGB(i)
#if KEY_DHDGB==1
!AP/MF
             IF (QFHDGB) THEN
                 DO J=1,INCT_FHDGB-1
                    DGB_DS_ATM(I,J)=-(HALF*CCELEC*P_HDGB(i) &
                    + E_ALPHA * S_HDGB(i))*DEPSP(I)*DSI_DSJ(I,J)
                 ENDDO
                 DO J=INCT_FHDGB,TOTALS
                    DGB_DS_ATM(I,J)=-(HALF*CCELEC*P_HDGB(i) &
                    + E_ALPHA * S_HDGB(i))*DEPSPL(I)*DSI_DSJ(I,J)
                 ENDDO
             ENDIF

#endif
#if KEY_HDGBVDW==1
! MS/MF
             IF (QGBVDW) THEN
               IF ((UNDO_HDGB .NE. -1) .or. &
                   (UNDC_HDGB .NE. -1) .or. &
                   (UNDC2_HDGB .NE. -1) .or. &
                   (UNDP_HDGB .NE. -1) .or. &
                   (UNDN_HDGB .NE. -1) ) THEN 
                   DZ_GBVdw(i) = DZ_GBVdw(i) + DDD_HDGB(i)
               ENDIF
            ENDIF 
! MS/MF
#endif

             ! HDGB2
          else if (CORR_GB .eq. 4) then
             X0_HDGB = (HALF*CCELEC*P_HDGB(i)*DEPS_HDGB(i) &
                  + E_ALPHA * S_HDGB(i) * DEPS_HDGB(i))/RDIST_HDGB(i)
             DX(i) = DX(i) - X0_HDGB*X(i)
             DY(i) = DY(i) - X0_HDGB*Y(i)
             DZ(i) = DZ(i) - X0_HDGB*Z(i)
          else if (CORR_GB .eq. 5) then
             X0_HDGB = (HALF*CCELEC*P_HDGB(i)*DEPS_HDGB(i) &
                  + E_ALPHA * S_HDGB(i) * DEPS_HDGB(i))/RDIST_HDGB(i)
             DX(i) = DX(i) - X0_HDGB*X(i)
             DY(i) = DY(i) - X0_HDGB*Y(i)
             ! HDGB2
          endif

          DO J=1,NWeights
             I2 = I2 + 1
             IF (Surf_Atom(I2).EQ.-1) THEN
                Rx = WtX(j) + cx
                Ry = WtY(j) + cy
                Rz = WtZ(j) + cz
                IX = Rx * dn_inv
                IY = Ry * dn_inv
                IZ = Rz * dn_inv

                POS = IX + IY * Nx + IZ * Nxy - OFFSET
                RX = RX + XMIN
                RY = RY + YMIN
                RZ = RZ + ZMIN
                PS = Index(POS)
                k =  AList(PS)
                num = PTR(POS)+PS
                R7 = ZERO
                R1 = ZERO
                IF (P3 .LE. ZERO) THEN ! Original GBMV
                   R1 = ZERO
                   TX = ZERO
                   TY = ZERO
                   TZ = ZERO
                   l = l+1
                   ctr = 0
                   DO WHILE ((ps .lt. num) .and. (R1 .lt. thresh2))
                      VX = RX - X(k) ! Rm + Xi - Xj
                      VY = RY - Y(k)
                      VZ = RZ - Z(k)
                      R2 = VX*VX + VY*VY + VZ*VZ !|Rm + Xi - Xj|^2
                      R3 = GAMMA(k)*R2*R2
                      IF (R3 .gt. -23) THEN
                         R3 = DEXP(R3)
                         R1 = R1 + R3
                         R2 = R3 * FOUR * R2 * GAMMA(k)
                         VX = VX * R2
                         VY = VY * R2
                         VZ = VZ * R2
                         ctr = ctr + 1
                         BX_GB(ctr) = VX
                         BY_GB(ctr) = VY
                         BZ_GB(ctr) = VZ
                         TEMP1_GB(ctr) = k
                         TX = TX + VX
                         TY = TY + VY
                         TZ = TZ + VZ
                      ENDIF
                      PS = PS + 1
                      if (ps .lt. num) k = AList(PS)
                   ENDDO
                ELSE          ! VSA GBMV
                   ! Find hard-spheres; accumulate info
                   ctr = 0
                   DO WHILE ((ps .lt. num) .and. (R7 .lt. TWO))
                      l = l + 1
                      VX = RX - X(k)
                      VY = RY - Y(k)
                      VZ = RZ - Z(k)
                      R2 = VX * VX + VY * VY + VZ * VZ
                      X1 = WMAIN(k)+HSX1
                      X2 = X1*X1
                      IF (R2 .LT. X2) THEN
                         R7 = TWO ! Hard-sphere
                         ctr = 0
                      ELSE
                         X2 = WMAIN(k)+Off_X
                         X2 = X2*X2
                         IF (R2 .LT. X2) THEN ! Accumulate atoms
                            ctr = ctr + 1 ! that exist in VSA region
                            TEMP1_GB(ctr) = k
                            VX1(ctr)=VX
                            VY1(ctr)=VY
                            VZ1(ctr)=VZ
                            RA(ctr)=R2
                            RB(ctr)=WMAIN(k)
                            RA_HDGB(ctr)=DR_HDGB(k)
                         ENDIF
                         PS = PS + 1 ! move to next atom
                         if (ps .lt. num) k = AList(PS)
                      ENDIF
                   ENDDO

                   ! Check inter-atom region

                   IF (ctr.gt.0) THEN
                      SX = ZERO
                      SY = ZERO
                      SZ = ZERO
                      A = ZERO
                      C = ZERO
                      R7 = ZERO
                      R1 = ZERO
                      DO iii=1,ctr
                         R2 = RA(iii)
                         X0 = RB(iii) ! R
                         X0_HDGB = RA_HDGB(iii) ! DR
                         IF (CUBIC_VSA_GB .ne. 0) THEN
                            T1 = (X0+Off_X)*(X0+Off_X)
                            S2 = (R2-T1)/(X0*X0-T1)
                            R3 = S2*S2*S2
                            X1 = SIX*S2*S2/(X0*X0-T1)

                            ! VSA fn.
                         ELSE
                            X5 = P2*X0 + P1 ! P2 * R(ii) + P1
                            X2 = X0*X0
                            X3 = R2 - X2
                            X4 = X5+X3
                            X4 = ONE/X4
                            R3 = X5*X4
                            R3 = R3*R3
                            X1 = -FOUR*R3*X4
                            ! STJ/MF
                            IF (QHDGBRC) THEN
                               X1_HDGB = TWO*(P2/X5-(P2-TWO*X0)*X4) &
                                    *R3
                            END IF
                            ! STJ/MF
                         ENDIF

                         !     Hard-sphere tail
                         X3 = X0+HSX2
                         X3 = X3*X3
                         IF (R2 .LT. X3) THEN
                            X2 = X0+HSX1
                            X2 = X2*X2

                            X5 = ONE/(X3-X2)
                            X4 = (R2-X2)*X5

                            X2 = X4*X4
                            X3 = X2*X4
                            FSw_X = ONE + X3*(X4*(15.0D0-SIX*X4)-TEN)
                            ! SJT/MF
                            X2_HDGB = ((X0+HSX1) &
                                 + X4*(HSX2-HSX1))
                            X3_HDGB = SIXTY*X2*(TWO*X4-X2-ONE)
                            DfSw_X = X3_HDGB*X5
                            R7 = R7 + TWO*FSw_X
                            R7D(iii) = TWO*DfSw_X
                            ! SJT/MF
                            R7D_HDGB(iii) = -TWO*X3_HDGB*X2_HDGB*X5
                            DfSw_X = DfSw_X * R3
                            R3 = R3 * (ONE-FSw_X)
                            X1 = X1 * (ONE-FSw_X) - DfSw_X
                            ! STJ/MF
                            IF (QHDGBRC) THEN
                               X1_HDGB = X1_HDGB*(ONE-FSw_X) &
                                    + X2_HDGB*DfSw_X
                            END IF
                            ! STJ/MF
                         ELSE
                            R7D(iii) = ZERO
                            R7D_HDGB(iii) = ZERO
                         ENDIF
                         !     VSA tail
                         IF (CUBIC_VSA_GB .eq.0) THEN
                            X4 = X0+On_X ! R + 1.9
                            X4 = X4*X4
                            IF (R2 .GT. X4) THEN ! VSA fn
                               X2 = X0+Off_X ! R + 2.1
                               X2 = X2*X2

                               X5 = ONE/(X2-X4)
                               X3 = (R2-X4)*X5

                               X2 = X3*X3
                               X4 = X2*X3

                               FSw_X = ONE +  &
                                    X4*(X3*(15.0D0-SIX*X3)-TEN)
                               DfSw_X = SIXTY*X2*(TWO*X3-X2-ONE)*X5

                               DfSw_X = DfSw_X * R3
                               R3 = R3 * FSw_X
                               X1 = X1 * FSw_X + DfSw_X
                               ! STJ/MF
                               IF (QHDGBRC) THEN
                                  X1_HDGB = X1_HDGB*FSw_X &
                                       - ((X0+On_X) &
                                       + X3*(Off_X-On_X))*DfSw_X
                               END IF
                               ! STJ/MF
                            ENDIF
                         ENDIF
                         VX = VX1(iii)
                         VY = VY1(iii)
                         VZ = VZ1(iii)

                         dMX = X1 * VX
                         dMY = X1 * VY
                         dMZ = X1 * VZ

                         VX = VX * R3
                         VY = VY * R3
                         VZ = VZ * R3

                         SX = SX + VX
                         SY = SY + VY
                         SZ = SZ + VZ

                         A = A + VX*VX + VY*VY + VZ*VZ
                         C = C + R3

                         !--------< dC/dx >-------------

                         CX_GB(iii) = -dMX
                         CY_GB(iii) = -dMY
                         CZ_GB(iii) = -dMZ

                         !--------< dA/dx >-------------

                         X1 = TWO*R3

                         AX_GB(iii) = -X1*(R2*dMX + VX)
                         AY_GB(iii) = -X1*(R2*dMY + VY)
                         AZ_GB(iii) = -X1*(R2*dMZ + VZ)

                         !--------< dB/dx >-------------

                         BX_GB(iii) = R3
                         ! STJ/MF
                         IF (QHDGBRC) THEN
                            dMZ_HDGB = X1_HDGB
                            CZ_HDGB(iii) = -dMZ_HDGB
                            AZ_HDGB(iii) = -X1*R2*dMZ_HDGB
                         END IF
                         ! STJ/MF
                      ENDDO
                   ENDIF
                   IF (CTR .GT. 0) THEN
                      B = SX*SX + SY*SY + SZ*SZ
                      IF (B .LT. 1.0D-18) B=1.0D-18
                      R1 = C * P3 * A/B
                   ENDIF
                   R1 = R1 + R7 ! Hard-sphere
                ENDIF         ! P3 check

                R2 = ONE + DEXP(BETA * (R1-LAMBDA1))
                IMPULSE = -BETA * (R2 - ONE) / (R2*R2)

                R2 = IMPULSE * E_ALPHA
                ! SJT/MF
                IF (CORR_GB .EQ. 0) THEN
                   S = (Alph_Gb(i)-shift)*(Alph_Gb(i)-shift)
                   R1 = R2 * Wt1(j) * S
                   R3 = -TT * HALF * R2 * Wt2(j) * S / F(i)
                ELSEIF (CORR_GB .EQ. 1) THEN
                   S = (Alph_Gb(i)-shift)*(Alph_Gb(i)-shift)/(SLOPE)
                   R1 = -(ONE-DSQRT(HALF))*R2 * Wt1(j) * S
                   R3 = -PT25 * R2 * Wt4(j) * S * F(i)**(-0.75D0)
                ELSEIF (CORR_GB .EQ. 2) THEN
                   S = (Alph_Gb(i)-shift)*(Alph_Gb(i)-shift)/(SLOPE)
                   R1 = -A1*R2*Wt1(j)*S
                   R3 = S*(-A2*HALF*R2*Wt2(j)/F(i)- &
                        A3*PT25*R2*Wt4(j)*G(i)**(-0.75D0))
                ELSEIF ((CORR_GB .EQ. 3) .or. &
                     (CORR_GB .EQ. 4) .or. &
                     (CORR_GB .EQ. 5)) THEN
                   S = (Alph_Gb(i)-Shift_HDGB(i))* &
                        (Alph_Gb(i)-Shift_HDGB(i))/(SLOPE)
                   R1 = -A1*R2*Wt1(j)*S
                   R3 = S*(-A2*HALF*R2*Wt2(j)/F(i)- &
                        A3_HDGB(i)*PT25*R2*Wt4(j)*G(i)**(-0.75D0))
                ENDIF

                R3 = R3 + R1

                IF (P3 .GT. ZERO) THEN
                   IF (CTR .GT. 0) THEN
                      R4 = R3
                      R3 = -R4 * P3
                      X1 = ONE/B
                      X2 = A*X1*R3
                      X3 = C*X1
                      X4 = X2*X3*TWO
                      X3 = X3*R3
                   ENDIF
                ELSE
                   DX_GB(i) = DX_GB(i) - R3*TX
                   DY_GB(i) = DY_GB(i) - R3*TY
                   DZ_GB(i) = DZ_GB(i) - R3*TZ
                ENDIF

                IF (CTR .GT. 0) THEN
                   DO K = 1,CTR
                      k2 = TEMP1_GB(k)
                      IF (P3 .GT. 0) THEN
                         X1 = VX1(k)*SX+VY1(k)*SY+VZ1(k)*SZ
                         X1_HDGB = X1

                         TX = BX_GB(k)*SX - CX_GB(k)*X1
                         TY = BX_GB(k)*SY - CY_GB(k)*X1
                         TZ = BX_GB(k)*SZ - CZ_GB(k)*X1

                         VX = CX_GB(k)*X2 + AX_GB(k)*X3 + X4*TX
                         VY = CY_GB(k)*X2 + AY_GB(k)*X3 + X4*TY
                         VZ = CZ_GB(k)*X2 + AZ_GB(k)*X3 + X4*TZ

                         X1 = R4*R7D(k)

                         VX = VX + X1*VX1(k)
                         VY = VY + X1*VY1(k)
                         VZ = VZ + X1*VZ1(k)

                         DX_GB(i)  = DX_GB(i)  - VX
                         DY_GB(i)  = DY_GB(i)  - VY
                         DZ_GB(i)  = DZ_GB(i)  - VZ

                         DX_GB(k2) = DX_GB(k2) + VX
                         DY_GB(k2) = DY_GB(k2) + VY
                         DZ_GB(k2) = DZ_GB(k2) + VZ
                         ! SJT/MF
                         IF (QHDGBRC) THEN
                            TZ_HDGB = - CZ_HDGB(k)*X1_HDGB
                            IF (i .EQ. k2) THEN
                               DZ_GB(i) = DZ_GB(i)-(R4*R7D_HDGB(k) &
                                    + CZ_HDGB(k)*X2+AZ_HDGB(k)*X3  &
                                    + X4*TZ_HDGB) &
                                    * RA_HDGB(k)
                            ELSE
                               DZ_GB(k2) = DZ_GB(k2)-(R4*R7D_HDGB(k) &
                                    + CZ_HDGB(k)*X2+AZ_HDGB(k)*X3  &
                                    + X4*TZ_HDGB) &
                                    * RA_HDGB(k)
                            END IF
                         END IF
                         ! SJT/MF
                      ELSE
                         DX_GB(k2) = DX_GB(k2) + R3*BX_GB(k)
                         DY_GB(k2) = DY_GB(k2) + R3*BY_GB(k)
                         DZ_GB(k2) = DZ_GB(k2) + R3*BZ_GB(k)
                      ENDIF
                   ENDDO
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       call timer_stop(T_GB_FORCE) 
    ENDIF                     !IF (QUpdateAlpha) THEN

    IF (IMPULSE_GB .EQ. 5) THEN
       FORCE_SCALE=ONE
    ELSEIF (IMPULSE_GB .EQ. 3) THEN ! emp method
       IF (QUpdateAlpha) THEN
          FORCE_SCALE = EMPB_GB ! Start with biggest
       ELSE
          FORCE_SCALE = FORCE_SCALE * EMPA_GB ! Decrease each step
       ENDIF
    ELSEIF (IMPULSE_GB .EQ. 2) THEN !limp method
       IF (QUpdateAlpha) THEN
          FORCE_SCALE = ALFRQ
       ELSE
          FORCE_SCALE = ZERO
       ENDIF
    ELSEIF (IMPULSE_GB .EQ. 4) THEN !simp method
       FORCE_SCALE = ONE/ALFRQ
    ELSE
       IF (QUpdateAlpha) THEN !imp method by default
          FORCE_SCALE = ONE
       ELSE
          FORCE_SCALE = 0.0
       ENDIF
    ENDIF
#if KEY_DHDGB==1
!AP/MF
    IF (QFHDGB) THEN
!        DO I=1,nmvsel
        DO I=1,natom
           SUM_DX_GB=0.D0
           SUM_DY_GB=0.D0
           SUM_DZ_GB=0.D0
           SUM_DX=0.D0
           SUM_DY=0.D0
           SUM_DZ=0.D0
           DO J=1,nmvsel
              SUM_DX_GB=SUM_DX_GB+AtSA_HDGB(j)* &
                        DGAMMAP(J)*DTHETADX(J,I)*DS_DTHETA(J) &
                        *(1.0D0-LAM1_DHDGB(J)) &
                        +AtSA_HDGB(j)* &
                        DGAMMAP(J)*DLAM1_DR_DHDGB(J)* &
                        (AVGSDEF-SFHDGB_OLD(J)) &
                        *DRDX(J,I) &
                        +AtSA_HDGB(j)* &
                        DGAMMAPL(J)*DTHETADX(J,I)*DS_DTHETALO(J) &
                        *(1.0D0-LAM1L_DHDGB(J)) &
                        +AtSA_HDGB(j)* &
                        DGAMMAPL(J)*DLAM1L_DR_DHDGB(J)* &
                        (AVGSDEFL-SFHDGBLO_OLD(J)) &
                        *DRDX(J,I)
              SUM_DY_GB=SUM_DY_GB+AtSA_HDGB(j)* &
                        DGAMMAP(J)*DTHETADY(J,I)*DS_DTHETA(J) &
                        *(1.0D0-LAM1_DHDGB(J)) &
                        +AtSA_HDGB(j)* &
                        DGAMMAP(J)*DLAM1_DR_DHDGB(J)* &
                        (AVGSDEF-SFHDGB_OLD(J)) &
                        *DRDY(J,I) &
                        +AtSA_HDGB(j)* &
                        DGAMMAPL(J)*DTHETADY(J,I)*DS_DTHETALO(J) &
                        *(1.0D0-LAM1L_DHDGB(J)) &
                        +AtSA_HDGB(j)* &
                        DGAMMAPL(J)*DLAM1L_DR_DHDGB(J)* &
                        (AVGSDEFL-SFHDGBLO_OLD(J)) &
                        *DRDY(J,I)
              SUM_DZ_GB=SUM_DZ_GB+AtSA_HDGB(j)* &
                        DGAMMAP(J)*DTHETADZ(J,I)*DS_DTHETA(J) &
                        *(1.0D0-LAM1_DHDGB(J)) &
                        +AtSA_HDGB(j)* &
                        DGAMMAP(J)*DLAM1_DR_DHDGB(J)* &
                        (AVGSDEF-SFHDGB_OLD(J)) &
                        *DRDZ(J,I) &
                        +AtSA_HDGB(j)* &
                        DGAMMAPL(J)*DTHETADZ(J,I)*DS_DTHETALO(J) &
                        *(1.0D0-LAM1L_DHDGB(J)) &
                        +AtSA_HDGB(j)* &
                        DGAMMAPL(J)*DLAM1L_DR_DHDGB(J)* &
                        (AVGSDEFL-SFHDGBLO_OLD(J)) &
                        *DRDZ(J,I)
              SUM_DX=SUM_DX- (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSP(J) &
                     *DS_DTHETA(J)*DTHETADX(J,I) &
                     *(1.0D0-LAM1_DHDGB(J)) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSP(J) &
                     *DLAM1_DR_DHDGB(J)* &
                     (AVGSDEF-SFHDGB_OLD(J)) &
                     *DRDX(J,I) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSPL(J) &
                     *DS_DTHETALO(J)*DTHETADX(J,I) &
                     *(1.0D0-LAM1L_DHDGB(J)) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSPL(J) &
                     *DLAM1L_DR_DHDGB(J)* &
                     (AVGSDEFL-SFHDGBLO_OLD(J)) &
                     *DRDX(J,I)
              SUM_DY=SUM_DY- (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSP(J) &
                     *DS_DTHETA(J)*DTHETADY(J,I) &
                     *(1.0D0-LAM1_DHDGB(J)) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSP(J) &
                     *DLAM1_DR_DHDGB(J)* &
                     (AVGSDEF-SFHDGB_OLD(J)) &
                     *DRDY(J,I) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSPL(J) &
                     *DS_DTHETALO(J)*DTHETADY(J,I) &
                     *(1.0D0-LAM1L_DHDGB(J)) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSPL(J) &
                     *DLAM1L_DR_DHDGB(J)* &
                     (AVGSDEFL-SFHDGBLO_OLD(J)) &
                     *DRDY(J,I)
             SUM_DZ=SUM_DZ- (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSP(J) &
                     *DS_DTHETA(J)*DTHETADZ(J,I) &
                     *(1.0D0-LAM1_DHDGB(J)) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSP(J) &
                     *DLAM1_DR_DHDGB(J)* &
                     (AVGSDEF-SFHDGB_OLD(J)) &
                     *DRDZ(J,I) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSPL(J) &
                     *DS_DTHETALO(J)*DTHETADZ(J,I) &
                     *(1.0D0-LAM1L_DHDGB(J)) &
                     - (HALF*CCELEC*P_HDGB(J) &
                     + T(J) * S_HDGB(J))*DEPSPL(J) &
                     *DLAM1L_DR_DHDGB(J)* &
                     (AVGSDEFL-SFHDGBLO_OLD(J)) &
                     *DRDZ(J,I)
           ENDDO
           DX_GB(I)=DX_GB(I)+SUM_DX_GB
           DY_GB(I)=DY_GB(I)+SUM_DY_GB
           DZ_GB(I)=DZ_GB(I)+SUM_DZ_GB
           DX(I)=DX(I)+SUM_DX
           DY(I)=DY(I)+SUM_DY
           DZ(I)=DZ(I)+SUM_DZ
        ENDDO
    ENDIF
#endif
    ! write(*,*) 'force_scale: ',force_scale
    IF (FORCE_SCALE .GT. 1E-10) THEN
       DO I=1,nmvsel
          DX(I)=DX(I)+FORCE_SCALE*DX_GB(I)
          DY(I)=DY(I)+FORCE_SCALE*DY_GB(I)
          DZ(I)=DZ(I)+FORCE_SCALE*DZ_GB(I)
       ENDDO
    ENDIF
#if KEY_HDGBVDW==1
! MS/MF
      IF (QGBVDW) THEN
       IF ((UNDO_HDGB .NE. -1) .or. &
          (UNDP_HDGB .NE. -1) .or. &
          (UNDC_HDGB .NE. -1) .or. &
          (UNDC2_HDGB .NE. -1) .or. &
          (UNDN_HDGB .NE. -1) )  then
          DO I=1,nmvsel  
            DZ(I)=DZ(I)+DZ_GBVdw(I)
          ENDDO
       ENDIF
      ENDIF
! MS/MF
#endif
#if KEY_DHDGB==1
!AP/MF
    IF(QFHDGB) THEN
       DO J=1,TOTALS
          DGB_DS(J)=0.D0
          DGnp_DS(J)=0.D0
          DS_DHDGB(J)=0.D0
       ENDDO
    ENDIF
    IF (QFHDGB) THEN
        DO J=1,TOTALS
           DO I=1,nmvsel
              DGB_DS(J)=DGB_DS(J)+DGB_DS_ATM(I,J)
              DGnp_DS(J)=DGnp_DS(J)+DGnp_DS_ATM(I,J)
           ENDDO
        ENDDO
        DO J=1,TOTALS
           DS_DHDGB(J)=DE_DS(J)+DGB_DS(J)+DGnp_DS(J)
! TEST
!           write(outu,*) "DS_DHDGB(J)",DS_DHDGB(J)
        ENDDO
    ENDIF
#endif
    ! IF ((.NOT. (IMPULSE_GB .OR. IMPULSE2_GB)).OR.(QUpdateAlpha)) THEN
    !    IF (IMPULSE2_GB) THEN
    !       FORCE_SCALE = ALFRQ
    !    ELSE IF (EMP_GB .LE. 1D+10) THEN
    !       FORCE_SCALE = ONE
    !    ENDIF
    !    IF (IMPULSE3_GB) THEN
    !       FORCE_SCALE = ALFRQ
    !       FORCE_SCALE = ONE/FORCE_SCALE
    !    ENDIF
    !    IF ((.not. IMPULSE2_GB) .AND. (EMP_GB .GT. ZERO)) THEN
    !    ENDIF
    ! c  WRITE(2,*)DX_GB(1),DY_GB(1),DZ_GB(1)
    ! ENDIF
    ! SJT/MF
    if ((CORR_GB .eq. 3) .or. &
         (CORR_GB .eq. 4) .or. &
         (CORR_GB .eq. 5)) then
#if KEY_HDGBVDW==1
        if (  (QGBVDW) .and.( ( UNDO_HDGB .NE. -1) .or. &
          (UNDP_HDGB .NE. -1) .or. (UNDC_HDGB .NE. -1) .or. &
          (UNDC2_HDGB .NE. -1) .or. (UNDN_HDGB .NE. -1) ) ) then
              ETERM(ASP) =  GVDW +Gnp_HDGB
        else
#endif
#if KEY_DHDGB==1
!AP/MF
       IF (QFHDGB) THEN
           ETERM(DEFE)=DEF_ENERUP+DEF_ENERLO
       ENDIF
#endif 
       ETERM(ASP) = Gnp_HDGB
#if KEY_HDGBVDW==1
        endif
#endif

    else
       if (QGBVDW) then
          ETERM(ASP) = Gnp_VDW + GVDW
       else if (QGBASP) then
          ETERM(ASP) = Gnp_ASP
       else
          ETERM(ASP) = Surface_Area * SA_GB
       endif
    endif
#if KEY_GBMVDEBUG==1
    IF (ESURF_GB) CLOSE(2)    
#endif

#if KEY_PARALLEL==1
    IF (MyNodP.eq.1) ETERM(ASP) = ETERM(ASP) + SB_GB
#else /**/
    ETERM(ASP) = ETERM(ASP) + SB_GB
#endif 

    ! WRITE(OUTU,909) Surface_Area
909 FORMAT('Surface area = ',F10.4)
    RETURN
  END Subroutine RunGBMV1

  ! MF optimized fast GBMV version
  Subroutine RunGBMV1F(GBEnergy, &
       JNbl,INbl,Inbl14,INblo14, &
       QInte,Islct,Jslct)
    !------------------------------------------------------------
    ! Build alpha array and some derivative components
    !------------------------------------------------------------
  use new_timer,only:timer_start,timer_stop,    & 
      T_gb_radii,T_gb_energ,T_gb_force        

  use exfunc
  use stream
  use dimens_fcm
  use param
  use psf
  use number
  use deriv
  use coord
  use consta
  use inbnd
  use energym
#if KEY_PARALLEL==1
  use parallel
#endif 

    real(chm_real) RE,RG,RI,RJ,R1,R2,R3,R4,R5,R1A,R6,W2,R7,RH
    real(chm_real) R,SX,SY,SZ,S,VX,VY,VZ,TX,TY,TZ,AI,AJ
    real(chm_real) C1,C2,C3,C4,C5,Q,D,A,CX,CY,CZ,Factor,SALT,FGB,TA2
    real(chm_real) XDIFF,YDIFF,ZDIFF,Rx,Ry,Rz,Impulse,E_Alpha
    real(chm_real) GBEnergy,RMax,XE1,XE2,YE1,YE2,ZE1,ZE2
    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)
    Integer i,j,k,l,PS,IX,IY,IZ,POS,ii,ctr,k2,sum,k1,iii,m
    INTEGER num,i2
    real(chm_real) C2OfNb
    Integer ITemp, NPr, jpr
    real(chm_real) C2OnNb, Rul3, Rul12, RijL, RijU, FSw, DFSw
    real(chm_real) B,C,BX,BY,BZ
    real(chm_real) Rul3_X, Rul12_X, RijL_X, RijU_X, FSw_X, DFSw_X
    real(chm_real) dMX,dMY,dMZ,X1,X2,X3,X4,X5,X6,X0
    Logical LOuter, Switch,QUpdateAlpha
    real(chm_real) TOTDIFF,Sum1
    real(chm_real) TOL2,RTWOPI,S1,Area1,Force_Scale
    real(chm_real) FOURPI,Surf5,EMPA_GB,EMPB_GB
    LOGICAL OK
    Integer ISlct(*), JSlct(*)
    Logical QInte,LOK
    real(chm_real) RR1,RRA
    real(chm_real) WK
    real(chm_real) XW1,XW2,XW3,XW4,XW5,XW6,XW7,XW8,XW9
    real(chm_real) iVX,iVY,iVZ
    Integer ki,ictr,ia1,ia2,ia3,ierr,i3
    Integer ii2,ii22,ii24,ii26,binx,cinx,dinx
    Integer sxs,gotit,si2,si3,si1,si0
    Integer ips

    ! SJT/MF VDW
    real(chm_real) EPSWVDW, SIGWVDW, GVDW
    real(chm_real) X1VDW, X2VDW, X3VDW, X4VDW
    real(chm_real) EPSIW, SIGIW
    real(chm_real) Gnp_VDW, Gnp_ASP
    ! SJT/MF VDW
    real(chm_real) Gnp_HDGB
    real(chm_real) X0_HDGB
#if KEY_HDGBVDW==1
! MS/MF
     real(chm_real) EPSNVDW,SIGNVDW,EPSC2VDW,SIGC2VDW,SIGPVDW
     real(chm_real) EPSCVDW,SIGCVDW,EPSOVDW,SIGOVDW,EPSPVDW
     real(chm_real) GVDW_HDGB
     real(chm_real) SIG_I,EPS_I
     real(chm_real) At_HDGB
!MS/MF
#endif

    !---------------------------------
    ! Compute Alphas for every atom
    !---------------------------------
    SAVE Force_Scale

    call timer_start(T_GB_RADII) 

    ! Generate radii and exponent values

    Switch = LCons .and. .not. LShft .and. .not. LFSwt

    IF (IMPULSE_GB .EQ. 3) THEN
       EMPA_GB = DEXP(-EMP_GB)
       EMPB_GB = ALFRQ
       EMPB_GB = EMPB_GB*(ONE-EMPA_GB)/(ONE-EMPA_GB**ALFRQ)
    ENDIF

    DO i=1,nmvsel
       Gamma(i)=rtmp(i*9-2)
    ENDDO

    TOTDIFF = ZERO
    DO I=1,nmvsel
       R1 = (X(I)-XCP(I))*(X(I)-XCP(I))+ &
            (Y(I)-YCP(I))*(Y(I)-YCP(I))+ &
            (Z(I)-ZCP(I))*(Z(I)-ZCP(I))
       TOTDIFF = TOTDIFF + R1
       IF (R1 .GT. tR2BUF) THEN
          GBGridReset = .true.
          EXIT
       ENDIF
    ENDDO

    GBStepCtr=GBStepCtr+1
    QUpdateAlpha = .NOT.((TOTDIFF .LT. 1.0D-20).AND.(FIXA))
    QUpdateAlpha = QUpdateAlpha .AND. (GBStepCtr .eq. 1)
    IF (GBStepCtr .ge. ALFRQ) THEN
       GBStepCtr=0
    ENDIF

    IF (DRSTEP .eq. 0 .or. DRSTEP .ge. DRFRQ) THEN
       DRSTEP=1
       ! MFC?IVK bugfix 10 Array dimensioned 1:MaxSurf
       ! DO I=0,NAtom*NWeights-1
       IF (nmvsel*nweights .gt. MaxSurf)  &
            CALL WrnDie(-1,'<RunGBMV1>', &
            'MaxSurf too small for natom*nweights')
       DO I=1,nmvsel*NWeights
          Surf_Atom(I) = 0
       ENDDO
    ELSE
       DRSTEP = DRSTEP + 1
    ENDIF

    SGBSTEP=SGBSTEP+1
    IF (SGBSTEP.GT.SGBFRQ) THEN
       SGBSTEP=1
    ENDIF

    IF (GBGridReset.and.QUpdateAlpha) THEN
       DO I=1,nmvsel
          XCP(I) = X(I)
          YCP(I) = Y(I)
          ZCP(I) = Z(I)
       ENDDO
       NGridGB = NGridGB_GB
       CALL BuildGBLookupF()
    ELSE
       XMIN=XMIN_GB
       YMIN=YMIN_GB
       ZMIN=ZMIN_GB
       XMAX=XMAX_GB
       YMAX=YMAX_GB
       ZMAX=ZMAX_GB
       NX = NX_GB
       NY = NY_GB
       NZ = NZ_GB
       NXY = NX*NY
       OFFSET = OFFSET_GB
       NGridGB = NGridGB_GB
    ENDIF

    IF (QUpdateAlpha) THEN

       ! WRITE(OUTU,*) 'Getting alphas...'
       l = 0

       NSurf = 0
       R4 = log(thresh2)
       Surface_Area = ZERO
       Gnp_HDGB = ZERO
       Gnp_VDW  = ZERO
       DO i=1,nmvsel
          T(i) = ZERO
#if KEY_HDGBVDW==1
          DP(i) = ZERO
          DDD_HDGB(I)=ZERO
#endif
          Alph_Gb(i) = ZERO
          F(i) = ZERO
          G(i) = ZERO
          DX_GB(I)=ZERO
          DY_GB(I)=ZERO
          DZ_GB(I)=ZERO
#if KEY_HDGBVDW==1
          DZ_GBVdw(I)=ZERO
#endif
          P_HDGB(I)=ZERO
          AtSA_HDGB(I)=ZERO
       ENDDO

#if KEY_GBMVDEBUG==1
       IF (ECOMP_GB) OPEN(1,file='comp.txt',status='unknown')
       IF (ESURF_GB) OPEN(2,file='surf_atoms.txt',status='unknown')

       IF (EVDW_GB) THEN
          OPEN(1,file='atoms.xyz',status='unknown')
          DO I=1,nmvsel
             WRITE(1,'(5(F10.5,1X)')
             X(I),Y(I),Z(I),CCNBA(I),CCNBB(I)
          ENDDO
          CLOSE(1)
          OPEN(1,file='surf.xyz',status='unknown')
       ENDIF

       OPEN (2,file='volume.xyz',status='unknown')
#endif /*                         GBMVDEBUG*/

       if (CORR_GB .eq. 3) then
          DO i = 1, nmvsel
             CALL CalEpsHDGB(EPS_HDGB(i), DEPS_HDGB(i), Z(i),  &
                  EPS0_HDGB, EPSW_HDGB,EPSMIN_HDGB, EPSMAX_HDGB,i)
             CALL CalSurfTspl_HDGB(SurfT_HDGB(i), DSurfT_HDGB(i), &
                  Z(i),NP0_HDGB,NPW_HDGB,NPMIN_HDGB ,GMASP(i),i)  ! YMC/MF
          ENDDO
          ! HDGB2
       else if (CORR_GB .eq. 4) then
          DO i = 1, nmvsel
             RDIST_HDGB(i) = DSQRT(X(i)**2+Y(i)**2+Z(i)**2)
             CALL CalEpsHDGB(EPS_HDGB(i), DEPS_HDGB(i), RDIST_HDGB(i), &
                  EPS0_HDGB, EPSW_HDGB,EPSMIN_HDGB, EPSMAX_HDGB,i)
             CALL CalSurfTspl_HDGB(SurfT_HDGB(i), DSurfT_HDGB(i), &
                  RDIST_HDGB(i),NP0_HDGB,NPW_HDGB, &
                  NPMIN_HDGB, GMASP(i),i)       ! YMC/MF
          ENDDO
       else if (CORR_GB .eq. 5) then
          DO i = 1, nmvsel
             RDIST_HDGB(i) = DSQRT(X(i)**2+Y(i)**2)
             CALL CalEpsHDGB(EPS_HDGB(i), DEPS_HDGB(i), RDIST_HDGB(i), &
                  EPS0_HDGB, EPSW_HDGB,EPSMIN_HDGB, EPSMAX_HDGB,i)
             CALL CalSurfTspl_HDGB(SurfT_HDGB(i), DSurfT_HDGB(i), &
                  RDIST_HDGB(i),NP0_HDGB,NPW_HDGB, &
                  NPMIN_HDGB, GMASP(i),i)      ! YMC/MF
          ENDDO
          ! HDGB2
       endif

       I2 = 0

       Binx=1
       Cinx=1
       Dinx=1

       SXS=0

       ! Do Solvent-Accessible-Surface-Area (SASA) term

       FOURPI = TWO*TWOPI

#if KEY_PARALLEL==1
       DO i = MyNodP, nmvsel, NumNod
#else /**/
       DO i=1,nmvsel
#endif 
          CX = X(i) - XMIN
          CY = Y(i) - YMIN
          CZ = Z(i) - ZMIN

          Area1 = FOURPI*(WMAIN(i)+1.4D0)*(WMAIN(i)+1.4D0)
          Surface_Area = Surface_Area + Area1
          Surf5 = Area1
          AtSA_HDGB(i) = Area1
          Area1=Area1*tWtScale2

          IF (SA_GB .NE. ZERO) THEN
             R4 = (WMAIN(i)+1.4D0) * tWtScale
             DO j=1,NAngular
                Rx = R4 * WTX(j) + cx
                Ry = R4 * WTY(j) + cy
                Rz = R4 * WTZ(j) + cz
                IX = Rx * dn_inv
                IY = Ry * dn_inv
                IZ = Rz * dn_inv
                R7 = ZERO
                IF ((IX .ge. 0) .and. (IY .ge. 0) .and.  &
                     (IZ .ge. 0) .and. &
                     (IX .lt. Nx) .and. (IY .lt. Ny)  &
                     .and. (IZ .lt. Nz)) THEN
                   POS = IX + IY * Nx + IZ * Nxy - OFFSET
                   RX = RX + XMIN
                   RY = RY + YMIN
                   RZ = RZ + ZMIN
                   PS = Index(POS)
                   k  = AList(PS)
                   num = PTR(POS)+PS
                   ctr = 0
                   DO WHILE ((ps .lt. num) .and. (R7 .lt. TWO))
                      IF (k .ne. i) THEN
                         VX = RX - X(k)
                         VY = RY - Y(k)
                         VZ = RZ - Z(k)
                         R2 = VX * VX + VY * VY + VZ * VZ
                         X4 = WMAIN(k)+SON_GB
                         X4 = X4*X4
                         IF (R2 .LT. X4) THEN
                            R7 = TWO ! Hard-sphere
                            ctr = 0
                         ELSE
                            X3 = WMAIN(k)+SOFF_GB
                            X3 = X3*X3
                            IF (R2 .LT. X3) THEN ! Accumulate atoms
                               ctr = ctr + 1
                               X5 = ONE/(X3-X4)
                               X2 = (R2-X4)*X5
                               X1 = X2*X2
                               X3 = X1*x2
                               FSw_X = ONE +  &
                                    X3*(X2*(15.0D0-SIX*X2)-TEN)
                               DfSw_X = SIXTY*X1*(TWO*X2-X1-ONE)*X5
                               R7 = R7 + TWO*FSw_X

                               ictr=(ctr-1)*4

                               VTMP(ictr+1)=DfSw_X
                               VTMP(ictr+2)=VX
                               VTMP(ictr+3)=VY
                               VTMP(ictr+4)=VZ

                               TEMP1_GB(ctr) = k
                            ENDIF
                         ENDIF ! Within hard shell
                      ENDIF
                      PS = PS + 1 ! move to next atom
                      IF (PS .lt. num) k = AList(PS)
                   ENDDO      !Loop over atoms
#if KEY_GBMVDEBUG==1
                   IF (EVDW_GB) THEN
                      IF (R7.LT.ONE) WRITE(1,'(4(F10.5,1X))')  &
                           RX,RY,RZ,ONE-R7
                   ENDIF
#endif 

                   IF (R7 .GT. ZERO) THEN

                      IF (R7 .LT. ONE) THEN
                         X2 = R7
                         X1 = X2*X2
                         X3 = X1*X2
                         ! S  = ONE + X3*(X2*(15.0D0-SIX*X2)-TEN)
                         S  = -X3*(X2*(15.0D0-SIX*X2)-TEN)
                         S1 = -SIXTY*X1*(TWO*X2-X1-ONE)
                      ELSE
                         S = ONE
                         S1 = ZERO
                      ENDIF

                      Surface_Area = Surface_Area - Area1*S*WT1(J)
                      Surf5 = Surf5 - Area1*S*WT1(J)
                      if ((CORR_GB .eq. 3) .or. &
                           (CORR_GB .eq. 4) .or. &
                           (CORR_GB .eq. 5)) THEN
                         S1 = S1*WT1(J)*Area1*SurfT_HDGB(I)
                         AtSA_HDGB(I)=AtSA_HDGB(I)-Area1*S*WT1(J)
                      else if (QGBVDW) then
                         S1 = S1*WT1(J)*Area1*GMVDW(I)
                      else if (QGBASP) then
                         S1 = S1*WT1(J)*Area1*GMASP(I)
                      else
                         S1 = S1*WT1(J)*Area1*SA_GB
                      endif

                      IF (CTR .gt. 0) THEN
                         DO II=1,ctr
                            ictr=(ii-1)*4
                            k2 = TEMP1_GB(ii)

                            S = S1 * VTMP(ictr+1)
                            DX_GB(i) = DX_GB(i) - S*VTMP(ictr+2)
                            DY_GB(i) = DY_GB(i) - S*VTMP(ictr+3)
                            DZ_GB(i) = DZ_GB(i) - S*VTMP(ictr+4)
                            DX_GB(k2) = DX_GB(k2) + S*VTMP(ictr+2)
                            DY_GB(k2) = DY_GB(k2) + S*VTMP(ictr+3)
                            DZ_GB(k2) = DZ_GB(k2) + S*VTMP(ictr+4)

                         ENDDO
                      ENDIF
                   ENDIF
                ENDIF         !Inside grid
             ENDDO            !Do j=1,Nangular

             if (CORR_GB .eq. 3) then
                Gnp_HDGB = Gnp_HDGB + SurfT_HDGB(i)*AtSA_HDGB(i)
                DZ_GB(i) = DZ_GB(i) + DSurfT_HDGB(i)*AtSA_HDGB(i)
                ! HDGB2
             else if (CORR_GB .eq. 4) then
                Gnp_HDGB = Gnp_HDGB + SurfT_HDGB(i)*AtSA_HDGB(i)
                X0_HDGB = DSurfT_HDGB(i)*AtSA_HDGB(i)/RDIST_HDGB(i)
                DX_GB(i) = DX_GB(i) + X0_HDGB*X(i)
                DY_GB(i) = DY_GB(i) + X0_HDGB*Y(i)
                DZ_GB(i) = DZ_GB(i) + X0_HDGB*Z(i)
             else if (CORR_GB .eq. 5) then
                Gnp_HDGB = Gnp_HDGB + SurfT_HDGB(i)*AtSA_HDGB(i)
                X0_HDGB = DSurfT_HDGB(i)*AtSA_HDGB(i)/RDIST_HDGB(i)
                DX_GB(i) = DX_GB(i) + X0_HDGB*X(i)
                DY_GB(i) = DY_GB(i) + X0_HDGB*Y(i)
                ! HDGB2
             endif

             if (QGBVDW) then
                Gnp_VDW = Gnp_VDW + Surf5*GMVDW(i)
             else if (QGBASP) then
                Gnp_ASP = Gnp_ASP + Surf5*GMASP(i)
             endif
          ENDIF               ! Do SASA
#if KEY_GBMVDEBUG==1
          IF (ESURF_GB) WRITE(2,'(I4,I4,I4,1X,F10.6)')  & 
               I,IAC(I),ITC(IAC(I)),Surf5 
#endif

       ENDDO

       ! *******************************************************************************
       ! Do GB alpha part

       IF (DRSTEP.EQ.1 .AND. SGBSTEP.EQ.1) THEN

#if KEY_PARALLEL==1
          DO i = MyNodP, nmvsel, NumNod
#else /**/
          DO i=1,nmvsel
#endif 
             CX = X(i) - XMIN
             CY = Y(i) - YMIN
             CZ = Z(i) - ZMIN

             RI = Rad(i)
             RE = ONE / RI
             RG = HALF / (RI*RI)
             RH = ONE / (FOUR*(RI*RI*RI*RI))

             XE1 = (X(i)-XMIN)
             XE2 = (XMAX-X(i))
             YE1 = (Y(i)-YMIN)
             YE2 = (YMAX-Y(i))
             ZE1 = (Z(i)-ZMIN)
             ZE2 = (ZMAX-Z(i))

             IF (XE1 .gt. XE2) THEN
                RMAX = XE1*XE1
             ELSE
                RMAX = XE2*XE2
             ENDIF

             IF (YE1 .gt. YE2) THEN
                RMAX = RMAX + YE1*YE1
             ELSE
                RMAX = RMAX + YE2*YE2
             ENDIF

             IF (ZE1 .gt. ZE2) THEN
                RMAX = RMAX + ZE1*ZE1
             ELSE
                RMAX = RMAX + ZE2*ZE2
             ENDIF

             RMAX = DSQRT(RMAX)

             DO j=1,NWeights
                I2 = I2 + 1

                SURFX(i2)=0

                R = WTS(j)
                S = ZERO

                IF ((R .ge. RI).and.(R .lt. RMAX)) THEN
                   Rx = WtX(j) + cx
                   Ry = WtY(j) + cy
                   Rz = WtZ(j) + cz
                   IX = Rx * dn_inv
                   IY = Ry * dn_inv
                   IZ = Rz * dn_inv
                   R1 = ZERO

                   IF ((IX .ge. 0) .and. (IY .ge. 0) .and.  &
                        (IZ .ge. 0) .and. &
                        (IX .lt. Nx) .and. (IY .lt. Ny) .and.  &
                        (IZ .lt. Nz)) THEN

                      II2=(I2-1)*6
                      XIndex(II2+1)=Binx
                      XIndex(II2+3)=Cinx
                      XIndex(II2+5)=Dinx

                      RX = RX + XMIN
                      RY = RY + YMIN
                      RZ = RZ + ZMIN
                      R7 = ZERO
                      R1A = ZERO
                      SX = ZERO
                      SY = ZERO
                      SZ = ZERO
                      POS = IX + IY * Nx + IZ * Nxy - OFFSET
                      PS = Index(POS)
                      num = PTR(POS)+PS

                      GOTIT=1
                      DO WHILE (ps .lt. num)
                         k=AList(PS)
                         VX = RX - X(k)
                         VY = RY - Y(k)
                         VZ = RZ - Z(k)

                         ki=(k-1)*9
                         XW1=rtmp(ki+2) ! (WK+HSX1)^2
                         R2 = VX * VX + VY * VY + VZ * VZ

                         IF (R2 .LT. (XW1-SXDGB)) THEN
                            RE = RE - tR2CI*Wt1(j)
                            RG = RG - tR2CI*Wt2(j)
                            RH = RH - tR2CI*Wt4(j)
                            SURF_ATOM(I2) = 1
                            SURFX(I2)=3
                            GOTIT=99
                            ps=num+1
                         ELSE
                            IF (R2.LT.XW1) THEN
                               RE = RE - tR2CI*Wt1(j)
                               RG = RG - tR2CI*Wt2(j)
                               RH = RH - tR2CI*Wt4(j)
                               SURF_ATOM(I2) = 1
                               SURFX(I2)=2
                               GOTIT=99
                               ps=num+1
                            ELSE
                               IF (R2 .LT. rtmp(ki+5)) THEN
                                  R3=ONE/(rtmp(ki+9)+rtmp(ki+8)*R2)
                                  R3=R3*R3

                                  IF (R2 .LT. rtmp(ki+3)) THEN
                                     X2 = (R2-XW1)*rtmp(ki+7)
                                     X3 = X2*X2*x2
                                     FSw_X = ONE  &
                                          + X3*(X2*(15.0D0-SIX*X2)-TEN)
                                     R7 = R7 + TWO*FSw_X
                                     R3 = R3 * (ONE-FSw_X)

                                     BList(Binx)=k
                                     Binx=Binx+1
                                  ELSE
                                     XW3=rtmp(ki+4) ! (WK+On_X)^2
                                     IF (R2 .GT. XW3) THEN ! VSA Tail
                                        X2 = (R2-XW3)*rtmp(ki+6)
                                        X3 = X2*X2*X2
                                        FSw_X = ONE +  &
                                             X3*(X2*(15.0D0-SIX*X2)-TEN)
                                        R3 = R3 * FSw_X
                                        DList(Dinx)=k
                                        Dinx=Dinx+1
                                     ELSE
                                        CList(Cinx)=k
                                        Cinx=Cinx+1
                                     ENDIF
                                  ENDIF

                                  R1=R1+R3
                                  VX=VX*R3
                                  VY=VY*R3
                                  VZ=VZ*R3
                                  SX=SX+VX
                                  SY=SY+VY
                                  SZ=SZ+VZ
                                  R1A=R1A+R3*R3*R2
                               ENDIF
                               IF (GOTIT.eq.1) THEN
                                  IF (R2.LT.rtmp(ki+5)+SXDGB) THEN
                                     GOTIT=2
                                  ENDIF
                               ENDIF
                               PS=PS+1
                            ENDIF
                         ENDIF
                      ENDDO

                      IF (PS.LE.NUM) THEN
                         R4 = SX*SX + SY*SY + SZ*SZ
                         IF (R4 .LT. 1.0D-18) R4=1.0D-18
                         R1 = R1 * P3 * R1A/(R4+P7)
                         R1 = R1 + R7 ! Hard-sphere

                         R2 = ONE + DEXP(BETA * (R1-LAMBDA1))

                         S = ONE/R2

                         RE = RE - S * Wt1(j)
                         RG = RG - S * Wt2(j)
                         RH = RH - S * Wt4(j)

                         IF ((R1 .GT. THRESH1).AND. &
                              (R1 .LT. THRESH2)) THEN
                            SURF_ATOM(I2) = -1
                         ELSE
                            SURF_ATOM(I2) = S + TOL_GB
                         ENDIF

                         SURFX(I2)=GOTIT

                         XIndex(II2+2)=Binx-1
                         XIndex(II2+4)=Cinx-1
                         XIndex(II2+6)=Dinx-1
                      ENDIF
                   ENDIF
                ENDIF
             ENDDO            !DO j=1,NWeights

             IF (CORR_GB .EQ. 0) THEN
                F(i) = DSQRT(RG)
                Alph_Gb(i) = -ONE/(RE - TT * F(i) + ESHIFT) + SHIFT
             ELSEIF (CORR_GB .eq. 1) THEN
                F(i) = RH
                Alph_Gb(i) = SLOPE/((ONE-DSQRT(HALF))*RE+RH**(PT25))  &
                     + SHIFT
             ELSEIF (CORR_GB .eq. 2) THEN
                F(i) = DSQRT(RG)
                G(i) = RH
                Alph_Gb(i)=SLOPE/(A1*RE+A2*F(i)+A3*RH**(PT25)+ESHIFT) &
                     + SHIFT
             ELSEIF ((CORR_GB .eq. 3) .or. &
                  (CORR_GB .eq. 4) .or. &
                  (CORR_GB .eq. 5)) THEN
                ! Compute the Born radii via Eq. (15) in
                ! J.Chem.Phys. (2004) 120: 903.
                ! C0->A1 C1->A3 D->A4 E->A5
                F(i) = DSQRT(RG)
                G(i) = RH
                A3_HDGB(i) = A3 * THREE * EPS_HDGB(i) /  &
                     (THREE * EPS_HDGB(i) + TWO * EPS)
                SHIFT_HDGB(i) = A4 + A5/(EPS_HDGB(i) + ONE)
                Alph_Gb(i)      = SLOPE/(A1*RE+A3_HDGB(i)*RH**(PT25))  &
                     + SHIFT_HDGB(i)
                S_HDGB(i) = (Alph_Gb(i) - SHIFT_HDGB(i))**2 &
                     * RH**(PT25) &
                     * A3 * SIX * EPS  &
                     / (THREE * EPS_HDGB(i) + TWO * EPS)**2 &
                     + A5 / (EPS_HDGB(i) + ONE)**2
             ENDIF

#if KEY_GBMVDEBUG==1
             IF (ECOMP_GB) WRITE(1,210) RE,DSQRT(RG),DSQRT(DSQRT(RH)) 
#endif
          ENDDO               !Do i=1, Natom

       ELSEIF (DRSTEP.EQ.1 .AND. SGBSTEP.GT.1) THEN
#if KEY_PARALLEL==1
          DO i = MyNodP, nmvsel, NumNod
#else /**/
          DO i=1,nmvsel
#endif 
             CX = X(i) - XMIN
             CY = Y(i) - YMIN
             CZ = Z(i) - ZMIN

             RI = Rad(i)
             RE = ONE / RI
             RG = HALF / (RI*RI)
             RH = ONE / (FOUR*(RI*RI*RI*RI))

             DO j=1,NWeights
                I2 = I2 + 1
                SXS=SURFX(I2)
                IF (SXS.EQ.3) THEN
                   RE = RE - tR2CI*Wt1(j)
                   RG = RG - tR2CI*Wt2(j)
                   RH = RH - tR2CI*Wt4(j)
                   SURF_ATOM(I2) = 1
                ELSE
                   IF (SXS.EQ.2) THEN
                      R = WTS(j)
                      S = ZERO

                      Rx = WtX(j) + cx
                      Ry = WtY(j) + cy
                      Rz = WtZ(j) + cz
                      IX = Rx * dn_inv
                      IY = Ry * dn_inv
                      IZ = Rz * dn_inv
                      R1 = ZERO

                      II2=(I2-1)*6
                      XIndex(II2+1)=Binx
                      XIndex(II2+3)=Cinx
                      XIndex(II2+5)=Dinx

                      RX = RX + XMIN
                      RY = RY + YMIN
                      RZ = RZ + ZMIN
                      R7 = ZERO
                      R1A = ZERO
                      SX = ZERO
                      SY = ZERO
                      SZ = ZERO
                      POS = IX + IY * Nx + IZ * Nxy - OFFSET
                      PS = Index(POS)
                      num = PTR(POS)+PS

                      DO WHILE (ps .lt. num)
                         k=AList(PS)
                         VX = RX - X(k)
                         VY = RY - Y(k)
                         VZ = RZ - Z(k)

                         R2 = VX * VX + VY * VY + VZ * VZ

                         ki=(k-1)*9
                         XW1=rtmp(ki+2) ! (WK+HSX1)^2
                         XW3=rtmp(ki+4)
                         IF (R2.GE.rtmp(ki+5)) THEN
                         ELSEIF (R2.GE.XW3) THEN
                            R3=ONE/(rtmp(ki+9)+rtmp(ki+8)*R2)
                            R3=R3*R3
                            X2 = (R2-XW3)*rtmp(ki+6)
                            X3 = X2*X2*X2
                            FSw_X = ONE +  &
                                 X3*(X2*(15.0D0-SIX*X2)-TEN)
                            R3 = R3*FSw_X
                            R1=R1+R3
                            SX=SX+VX*R3
                            SY=SY+VY*R3
                            SZ=SZ+VZ*R3
                            R1A=R1A+R3*R3*R2
                            DList(Dinx)=k
                            Dinx=Dinx+1
                         ELSEIF (R2.GE.rtmp(ki+3)) THEN
                            R3=ONE/(rtmp(ki+9)+rtmp(ki+8)*R2)
                            R3=R3*R3
                            R1=R1+R3
                            SX=SX+VX*R3
                            SY=SY+VY*R3
                            SZ=SZ+VZ*R3
                            R1A=R1A+R3*R3*R2
                            CList(Cinx)=k
                            Cinx=Cinx+1
                         ELSEIF (R2.GE.XW1) THEN
                            R3=ONE/(rtmp(ki+9)+rtmp(ki+8)*R2)
                            R3=R3*R3
                            X2 = (R2-XW1)*rtmp(ki+7)
                            X3 = X2*X2*x2
                            FSw_X = ONE  &
                                 + X3*(X2*(15.0D0-SIX*X2)-TEN)
                            R7 = R7 + TWO*FSw_X
                            R3 = R3*(ONE-FSw_X)
                            R1=R1+R3
                            SX=SX+VX*R3
                            SY=SY+VY*R3
                            SZ=SZ+VZ*R3
                            R1A=R1A+R3*R3*R2
                            BList(Binx)=k
                            Binx=Binx+1
                         ELSE
                            RE = RE - tR2CI*Wt1(j)
                            RG = RG - tR2CI*Wt2(j)
                            RH = RH - tR2CI*Wt4(j)
                            SURF_ATOM(I2) = 1
                            PS=NUM
                         ENDIF
                         PS=PS+1
                      ENDDO

                      IF (PS.LE.NUM) THEN
                         R4 = SX*SX + SY*SY + SZ*SZ
                         IF (R4 .LT. 1.0D-18) R4=1.0D-18
                         R1 = R1 * P3 * R1A/(R4+P7)
                         R1 = R1 + R7 ! Hard-sphere

                         R2 = ONE + DEXP(BETA * (R1-LAMBDA1))

                         S = ONE/R2

                         RE = RE - S * Wt1(j)
                         RG = RG - S * Wt2(j)
                         RH = RH - S * Wt4(j)

                         IF ((R1 .GT. THRESH1).AND. &
                              (R1 .LT. THRESH2)) THEN
                            SURF_ATOM(I2) = -1
                         ELSE
                            SURF_ATOM(I2) = S + TOL_GB
                         ENDIF

                         XIndex(II2+2)=Binx-1
                         XIndex(II2+4)=Cinx-1
                         XIndex(II2+6)=Dinx-1
                      ENDIF
                   ELSE
                      IF (SXS.EQ.1) THEN
                         RE=RE-tR20i*Wt1(j)
                         RG=RG-tR20i*Wt2(j)
                         RH=RH-tR20i*Wt4(j)
                         SURF_ATOM(I2)=tR20i+TOL_GB
                      ENDIF
                   ENDIF
                ENDIF
             ENDDO            !DO j=1,NWeights

             IF (CORR_GB .EQ. 0) THEN
                F(i) = DSQRT(RG)
                Alph_Gb(i) = -ONE/(RE - TT * F(i) + ESHIFT) + SHIFT
             ELSEIF (CORR_GB .eq. 1) THEN
                F(i) = RH
                Alph_Gb(i) = SLOPE/((ONE-DSQRT(HALF))*RE+RH**(PT25))  &
                     + SHIFT
             ELSEIF (CORR_GB .eq. 2) THEN
                F(i) = DSQRT(RG)
                G(i) = RH
                Alph_Gb(i)=SLOPE/(A1*RE+A2*F(i)+A3*RH**(PT25)+ESHIFT) &
                     + SHIFT
             ELSEIF ((CORR_GB .eq. 3) .or. &
                  (CORR_GB .eq. 4) .or. &
                  (CORR_GB .eq. 5)) THEN
                ! Compute the Born radii via Eq. (15) in
                ! J.Chem.Phys. (2004) 120: 903.
                ! C0->A1 C1->A3 D->A4 E->A5
                F(i) = DSQRT(RG)
                G(i) = RH
                A3_HDGB(i) = A3 * THREE * EPS_HDGB(i) /  &
                     (THREE * EPS_HDGB(i) + TWO * EPS)
                SHIFT_HDGB(i) = A4 + A5/(EPS_HDGB(i) + ONE)
                Alph_Gb(i)      = SLOPE/(A1*RE+A3_HDGB(i)*RH**(PT25))  &
                     + SHIFT_HDGB(i)
                S_HDGB(i) = (Alph_Gb(i) - SHIFT_HDGB(i))**2 &
                     * RH**(PT25) &
                     * A3 * SIX * EPS  &
                     / (THREE * EPS_HDGB(i) + TWO * EPS)**2 &
                     + A5 / (EPS_HDGB(i) + ONE)**2
             ENDIF

#if KEY_GBMVDEBUG==1
             IF (ECOMP_GB) WRITE(1,210) RE,DSQRT(RG),DSQRT(DSQRT(RH)) 
#endif
          ENDDO               !Do i=1, Natom
       ELSE
          ! DRSTEP>1
#if KEY_PARALLEL==1
          DO i = MyNodP, nmvsel, NumNod
#else /**/
          DO i=1,nmvsel
#endif 
             CX = X(i) - XMIN
             CY = Y(i) - YMIN
             CZ = Z(i) - ZMIN

             RI = Rad(i)
             RE = ONE / RI
             RG = HALF / (RI*RI)
             RH = ONE / (FOUR*(RI*RI*RI*RI))

             DO j=1,NWeights
                I2 = I2 + 1

                IF (surf_atom(i2).eq.-1) THEN
                   R = WTS(j)
                   S = ZERO

                   Rx = WtX(j) + cx
                   Ry = WtY(j) + cy
                   Rz = WtZ(j) + cz
                   IX = Rx * dn_inv
                   IY = Ry * dn_inv
                   IZ = Rz * dn_inv
                   R1 = ZERO

                   IF ((IX .ge. 0) .and. (IY .ge. 0) .and.  &
                        (IZ .ge. 0) .and. &
                        (IX .lt. Nx) .and. (IY .lt. Ny) .and.  &
                        (IZ .lt. Nz)) THEN


                      II2=(I2-1)*6
                      XIndex(II2+1)=Binx
                      XIndex(II2+3)=Cinx
                      XIndex(II2+5)=Dinx

                      RX = RX + XMIN
                      RY = RY + YMIN
                      RZ = RZ + ZMIN
                      R7 = ZERO
                      R1A = ZERO
                      SX = ZERO
                      SY = ZERO
                      SZ = ZERO
                      POS = IX + IY * Nx + IZ * Nxy - OFFSET
                      PS = Index(POS)
                      num = PTR(POS)+PS-1

                      DO ips=ps,num
                         k=AList(iPS)

                         VX = RX - X(k)
                         VY = RY - Y(k)
                         VZ = RZ - Z(k)

                         R2 = VX * VX + VY * VY + VZ * VZ

                         ki=(k-1)*9
                         XW1=rtmp(ki+2) ! (WK+HSX1)^2

                         IF (R2 .LT. rtmp(ki+5)) THEN
                            R3=ONE/(rtmp(ki+9)+rtmp(ki+8)*R2)
                            R3=R3*R3
                            IF (R2 .LT. rtmp(ki+3)) THEN
                               X2 = (R2-XW1)*rtmp(ki+7)
                               X3 = X2*X2*x2
                               FSw_X = ONE  &
                                    + X3*(X2*(15.0D0-SIX*X2)-TEN)
                               R7 = R7 + TWO*FSw_X
                               R3 = R3 * (ONE-FSw_X)

                               BList(Binx)=k
                               Binx=Binx+1
                            ELSE
                               XW3=rtmp(ki+4) ! (WK+On_X)^2
                               IF (R2 .GT. XW3) THEN ! VSA Tail
                                  X2 = (R2-XW3)*rtmp(ki+6)
                                  X3 = X2*X2*X2
                                  FSw_X = ONE +  &
                                       X3*(X2*(15.0D0-SIX*X2)-TEN)
                                  R3 = R3 * FSw_X
                                  DList(Dinx)=k
                                  Dinx=Dinx+1
                               ELSE
                                  CList(Cinx)=k
                                  Cinx=Cinx+1
                               ENDIF
                            ENDIF

                            R1=R1+R3
                            VX=VX*R3
                            VY=VY*R3
                            VZ=VZ*R3
                            SX=SX+VX
                            SY=SY+VY
                            SZ=SZ+VZ
                            R1A=R1A+R3*R3*R2
                         ENDIF
                      ENDDO

                      XIndex(II2+2)=Binx-1
                      XIndex(II2+4)=Cinx-1
                      XIndex(II2+6)=Dinx-1

                      R4 = SX*SX + SY*SY + SZ*SZ
                      IF (R4 .LT. 1.0D-18) R4=1.0D-18
                      R1 = R1 * P3 * R1A/(R4+P7)
                      R1 = R1 + R7 ! Hard-sphere

                      R2 = ONE + DEXP(BETA * (R1-LAMBDA1))
                      S = ONE/R2
                      RE = RE - S * Wt1(j)
                      RG = RG - S * Wt2(j)
                      RH = RH - S * Wt4(j)
                   ENDIF
                ELSE
                   IF (surf_atom(i2).gt.zero) THEN
                      RE=RE-tR2CI*Wt1(j)
                      RG=RG-tR2CI*Wt2(j)
                      RH=RH-tR2CI*Wt4(j)
                   ENDIF
                ENDIF
             ENDDO            !DO j=1,NWeights

             IF (CORR_GB .EQ. 0) THEN
                F(i) = DSQRT(RG)
                Alph_Gb(i) = -ONE/(RE - TT * F(i) + ESHIFT) + SHIFT
             ELSEIF (CORR_GB .eq. 1) THEN
                F(i) = RH
                Alph_Gb(i) = SLOPE/((ONE-DSQRT(HALF))*RE+RH**(PT25))  &
                     + SHIFT
             ELSEIF (CORR_GB .eq. 2) THEN
                F(i) = DSQRT(RG)
                G(i) = RH
                Alph_Gb(i)=SLOPE/(A1*RE+A2*F(i)+A3*RH**(PT25)+ESHIFT) &
                     + SHIFT
             ELSEIF ((CORR_GB .eq. 3) .or. &
                  (CORR_GB .eq. 4) .or. &
                  (CORR_GB .eq. 5)) THEN
                ! Compute the Born radii via Eq. (15) in
                ! J.Chem.Phys. (2004) 120: 903.
                ! C0->A1 C1->A3 D->A4 E->A5
                F(i) = DSQRT(RG)
                G(i) = RH
                A3_HDGB(i) = A3 * THREE * EPS_HDGB(i) /  &
                     (THREE * EPS_HDGB(i) + TWO * EPS)
                SHIFT_HDGB(i) = A4 + A5/(EPS_HDGB(i) + ONE)
                Alph_Gb(i)      = SLOPE/(A1*RE+A3_HDGB(i)*RH**(PT25))  &
                     + SHIFT_HDGB(i)
                S_HDGB(i) = (Alph_Gb(i) - SHIFT_HDGB(i))**2 &
                     * RH**(PT25) &
                     * A3 * SIX * EPS  &
                     / (THREE * EPS_HDGB(i) + TWO * EPS)**2 &
                     + A5 / (EPS_HDGB(i) + ONE)**2
             ENDIF

#if KEY_GBMVDEBUG==1
             IF (ECOMP_GB) WRITE(1,210) RE,DSQRT(RG),DSQRT(DSQRT(RH)) 
#endif
          ENDDO               !Do i=1, Natom
       ENDIF

       ! end of GB alpha calculation
       ! ************************************************************

210    FORMAT(3(F25.21,1X))
       R1 = L
       R2 = NSURF
       !99   0   FORMAT('Weight # ',I4, ' Atom:',I4)
#if KEY_GBMVDEBUG==1
       ! IF (ECOMP_GB) CLOSE(1)  
       IF (EVDW_GB) CLOSE(1)  
#endif

#if KEY_PARALLEL==1
       ! WRITE(OUTU,990)MyNodP,NSurf,l
       ! 990   FORMAT('Node ',I2,': surface/volume points =',I10,I10)
#endif 

#if KEY_PARALLEL==1
       R1 = R1 * NumNod
       R2 = R2 * NumNod
#endif 
       R1 = R1/(NWeights*nmvsel)
       R2 = R2/(NWeights*nmvsel)*HUNDRD
#if KEY_GBMVDEBUG==1
       IF (PRNLEV .GE. 6) THEN
          WRITE(OUTU,1000) R1,R2
       ENDIF
1000   FORMAT ('# atoms per IP: ',F6.3,'    Surface: ',F6.2,'%')
#endif /*                         GMBMVDEBUG*/

#if KEY_PARALLEL==1
       CALL GComb(Alph_Gb,nmvsel)
#endif 
    ENDIF                     !IF (QUpdateAlpha) THEN

    call timer_stop(T_GB_RADII) 

    !-------------------------------------------------
    ! Forming energy and some derivative components
    !-------------------------------------------------
    ! WRITE(OUTU,*) 'Forming energy...'

    call timer_start(T_GB_ENERG) 

    ! First, do excluded list
    SALT = -CCELEC * (ONE/EPS - ONE/EPS_GB)
    STILLFAC = ONE/P5
    GBEnergy = ZERO
    ITemp = 0
#if KEY_PARALLEL==1 /*paraexcl*/
    DO i=MyNodP, nmvsel, NumNod
#else /* (paraexcl)*/
    DO i=1,nmvsel
#endif /* (paraexcl)*/
       !AvdV..bugfix 16-Jul-2004
       If (i .ne. 1) ITemp = INblo14(i-1)
       !AvdV..
       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          IF (k .gt. 0) THEN
             LOK = .TRUE.
             If (QInte) LOK = (Islct(i).eq.1).and.(Jslct(j).eq.1)
             IF (LOK) THEN
                XDIFF = X(i) - X(j)
                YDIFF = Y(i) - Y(j)
                ZDIFF = Z(i) - Z(j)
                R1 = XDIFF * XDIFF + YDIFF * YDIFF + ZDIFF * ZDIFF
                A = Alph_Gb(i) * Alph_Gb(j)

                Q = CG(i) * CG(j)

                X3 = R1/A
                D = DEXP(-P6*X3)

                FGB = DSQRT(R1+A*D)
                C2 = ONE/FGB
                C1 = C2*C2

                C3 = ZERO

                IF (KAPPA_GB .ne. ZERO) THEN
                   C4 = DEXP(-KAPPA_GB*FGB)/EPS_GB
                   SALT = -CCELEC*(ONE/EPS - C4)
                ENDIF

                IF (MODSTILL.EQ.0) THEN
                   if ((CORR_GB .eq. 3) .or. &
                        (CORR_GB .eq. 4) .or. &
                        (CORR_GB .eq. 5)) then
                      EPSIJ_HDGB = HALF*(EPS_HDGB(i) + EPS_HDGB(j))
                      GBEnergy = GBEnergy  &
                           + Q * (C2+C3)* &
                           (-CCELEC * (ONE/EPS - ONE/EPSIJ_HDGB))
                      P_HDGB(i) = P_HDGB(i) + (ONE/EPSIJ_HDGB**2)*Q*C2
                      P_HDGB(j) = P_HDGB(j) + (ONE/EPSIJ_HDGB**2)*Q*C2
                   else
                      GBEnergy = GBEnergy + Q * (C2+C3)*SALT
                   endif
                ELSE
                   RR1=DSQRT(R1)
                   RRA=DSQRT(A)
                   FGB=DSQRT(R1+(A+MS1*R1*RR1/RRA+MS2*RR1*RRA &
                        +MS3*R1 )*D)
                   GBEnergy = GBEnergy + Q * MS0 / FGB *SALT
                ENDIF

                IF (KAPPA_GB .ne. ZERO) THEN
                   C2 = Q * HALF * C1 * (C2*SALT+CCELEC*KAPPA_GB*C4) &
                        + Q*C3*HALF*C2*CCELEC*KAPPA_GB*C4
                   !(1/2f)*Q*k*KAPPA*exp(KAP*F)/eps_gb
                ELSE
                   if ((CORR_GB .eq. 3) .or. &
                        (CORR_GB .eq. 4) .or. &
                        (CORR_GB .eq. 5)) then
                      C2 = Q * C1 * C2 * Half *  &
                           (-CCELEC * (ONE/EPS - ONE/EPSIJ_HDGB))
                   else
                      C2 = Q * C1 * C2 * Half * SALT
                   endif
                ENDIF

                X4 = ONE + P6*X3

                C5 = C2 * D * X4
                RI = C5 * Alph_Gb(j)
                RJ = C5 * Alph_Gb(i)

                C2 = C2 * TWO * (P6*D-ONE)

                VX = XDIFF * C2
                VY = YDIFF * C2
                VZ = ZDIFF * C2
                T(i) = T(i) - RI
                DX(I) = DX(I) + VX
                DY(I) = DY(I) + VY
                DZ(I) = DZ(I) + VZ
                T(j) = T(j) - RJ
                DX(J) = DX(J) - VX
                DY(J) = DY(J) - VY
                DZ(J) = DZ(J) - VZ
             ENDIF
          ENDIF
       ENDDO
       !AvdV..bugfix 16-Jul-2004       ITemp = INblo14(i)
    ENDDO                     !DO i=1,Natom

    ! Next, Do non-bonded list

    C2OfNB = CtOfNB * CtOfNB
    FSw = ONE
    DFSw = ZERO
    IF (Switch) THEN
       C2OnNb = CtOnNb * CtOnNb
       If (CtOfNb .gt. CtOnNb) Then
          Rul3 = One / (C2OfNb - C2OnNb)**3
          Rul12 = Twelve * Rul3
       Endif
    Endif

    ITemp = 0
    DO i=1,nmvsel-1
#if KEY_IMCUBES==1
       if (lbycbim) ITemp = INbl(I+nmvsel) 
#endif
       if (CG(i).ne.ZERO) THEN
          NPr = INbl(i) - ITemp
          Do jpr = 1, NPr
             k = JNbl(Itemp+jpr)
             j = Abs(k)
             IF (CG(j) .ne. zero) THEN
                LOK = .TRUE.
                If (QInte) LOK = (Islct(i).eq.1).and.(Jslct(j).eq.1)
                IF (LOK) THEN
                   XDIFF = X(i) - X(j)
                   YDIFF = Y(i) - Y(j)
                   ZDIFF = Z(i) - Z(j)
                   R1 = XDIFF * XDIFF + YDIFF * YDIFF + ZDIFF * ZDIFF

                   ! The Electrostatic Switch Function

                   If (R1 .lt. C2OfNb) Then

                      FSw = One
                      DFSw = Zero
                      If (Switch) Then
                         LOuter = (R1 .gt. C2OnNb)
                         If (LOuter) Then
                            RijL = C2OnNb - R1
                            RijU = C2OfNb - R1
                            FSw = RijU * RijU *  &
                                 (RijU - Three * RijL) * Rul3
                            DfSw = RijL * RijU * Rul12 / FSw
                         Endif
                      Endif

                      A = Alph_Gb(i) * Alph_Gb(j)

                      Q = CG(i) * CG(j)

                      X3 = R1/A
                      D = DEXP(-P6*X3)

                      FGB = DSQRT(R1+A*D)
                      C2 = ONE/FGB
                      C1 = C2*C2

                      C3 = ZERO

                      IF (KAPPA_GB .ne. ZERO) THEN
                         C4 = DEXP(-KAPPA_GB*FGB)/EPS_GB
                         SALT = -CCELEC*(ONE/EPS - C4)
                      ENDIF

                      IF (MODSTILL.EQ.0) THEN
                         if ((CORR_GB .eq. 3) .or. &
                              (CORR_GB .eq. 4) .or. &
                              (CORR_GB .eq. 5)) then
                            EPSIJ_HDGB = HALF &
                                 * (EPS_HDGB(i)+EPS_HDGB(j))
                            S = Q * (C2+C3) * FSw * &
                                 (-CCELEC * (ONE/EPS - ONE/EPSIJ_HDGB))
                            GBEnergy = GBEnergy + S
                            P_HDGB(i) = P_HDGB(i)  &
                                 + (ONE/EPSIJ_HDGB**2)*FSw*Q*C2
                            P_HDGB(j) = P_HDGB(j)  &
                                 + (ONE/EPSIJ_HDGB**2)*FSw*Q*C2
                         else
                            S = Q * (C2+C3) * FSw * SALT
                            GBEnergy = GBEnergy + S
                         endif
                      ELSE
                         RR1=DSQRT(R1)
                         RRA=DSQRT(A)
                         FGB=DSQRT( R1+ &
                              (A+MS1*R1*RR1/RRA+MS2*RR1*RRA+MS3*R1)*D)
                         GBEnergy = GBEnergy +  &
                              Q * MS0 / FGB * FSw *SALT
                      ENDIF

                      IF (KAPPA_GB .ne. ZERO) THEN
                         C2 = Q * FSw * HALF * C1 *  &
                              (C2*SALT+CCELEC*KAPPA_GB*C4) &
                              + Q*C3*HALF*C2*CCELEC*KAPPA_GB*C4
                         !(1/2f)*Q*k*KAPPA*exp(KAP*F)/eps_gb
                      ELSE if ((CORR_GB .eq. 3) .or. &
                               (CORR_GB .eq. 4) .or. &
                               (CORR_GB .eq. 5)) then
                        C2 = Q * C1 * C2 * Half * FSw * &
                             (-CCELEC * (ONE/EPS - ONE/EPSIJ_HDGB))
                      else
                        C2 = Q * C1 * C2 * Half * SALT * FSw
                      ENDIF

                      X4 = ONE + P6*X3

                      C5 = C2 * D * X4

                      RI = C5 * Alph_Gb(j)
                      RJ = C5 * Alph_Gb(i)

                      C2 = C2 * TWO * (P6*D-ONE)

                      IF (Switch .and. LOuter) THEN
                         C2 = C2 + DFSw * S
                      ENDIF
                      VX = XDIFF * C2
                      VY = YDIFF * C2
                      VZ = ZDIFF * C2
                      T(i) = T(i) - RI
                      DX(I) = DX(I) + VX
                      DY(I) = DY(I) + VY
                      DZ(I) = DZ(I) + VZ
                      T(j) = T(j) - RJ
                      DX(J) = DX(J) - VX
                      DY(J) = DY(J) - VY
                      DZ(J) = DZ(J) - VZ
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
       ENDIF
       ITemp = INbl(i)
    ENDDO                     !DO i=1,Natom-1

    ! Finally, the self terms

#if KEY_PARALLEL==1
    DO i=MyNodP,nmvsel,NumNod
#else /**/
    DO i=1,nmvsel
#endif 
       IF (IMOVE(i).eq.0) THEN
          LOK = .true.
          IF (QInte) LOK = (Islct(i).eq.1).and.(Jslct(i).eq.1)
          IF (LOK) THEN
             Q = CG(i) * CG(i)
             C1 = ONE / Alph_Gb(i)
             IF (KAPPA_GB .ne. ZERO) THEN
                C4 = DEXP(-KAPPA_GB*Alph_GB(i))/EPS_GB
                SALT = -CCELEC*(ONE/EPS - C4)
             ENDIF

             if ((CORR_GB .eq. 3) .or. &
                  (CORR_GB .eq. 4) .or. &
                  (CORR_GB .eq. 5)) then
                C2 = Q * abs(C1) * Half * &
                     (-CCELEC * (ONE/EPS - ONE/EPS_HDGB(i)))
                GBEnergy = GBEnergy + C2
                P_HDGB(i) = P_HDGB(i)  &
                     + (ONE/EPS_HDGB(i)**2)*Q*abs(C1)
             else
                C2 = Q * abs(C1) * Half * SALT
                IF(MODSTILL.EQ.0) THEN
                   GBEnergy = GBEnergy + C2
                ELSE
                   GBEnergy = GBEnergy + MS0 * C2
                ENDIF
             endif

             RI = C1 * C2
             IF (KAPPA_GB .ne. ZERO) THEN
                RI = RI + Q*HALF*CCELEC*(KAPPA_GB*C4) * C1
             ENDIF
             T(i) = T(i) - RI
          ENDIF
       ENDIF
    ENDDO

    ! SJT/MF VDW
    !
    ! VDW DISPERSION TERM (May, 2006)
    !
    ! E. Gallicchio and RM Levy, J Comput Chem 25: 479--499, 2004
    !
#if KEY_HDGBVDW==1
! MS/MF
      IF (QGBVDW ) THEN
       IF ( (UNDN_HDGB .NE. -1) .or. &
          (UNDO_HDGB .NE. -1) .or. &
          (UNDC_HDGB .NE. -1) .or. &
          (UNDC2_HDGB .NE. -1) .or. &
          (UNDP_HDGB .NE. -1) ) then

!lipid H 85
         EPSNVDW = 0.0307D0
         SIGNVDW = 1.2200D0
!lipid C tail 84
         EPSC2VDW = 0.0570D0
         SIGC2VDW = 2.0190D0
!lipid C 83
         EPSCVDW = 0.0680D0
         SIGCVDW = 2.0350D0
!water 81
         EPSOVDW = 0.1520D0
         SIGOVDW = 1.7700D0
!lipid O 82
         EPSPVDW = 0.1100D0
         SIGPVDW = 1.6750D0

         GVDW    = ZERO
         
#if KEY_PARALLEL==1
         DO i=MyNodP,nmvsel,NumNod
#else /**/
         DO I = 1, nmvsel
#endif
!         DDD_HDGB(I)=ZERO
!         WRITE(6,*) 'derivative>> ' ,I , DDD_HDGB(I)
           SIG_I = VDWR(ITC(IAC(I)))
           EPS_I = ABS(EFF(ITC(IAC(I))))

           IF (UNDN_HDGB .NE. -1) THEN

             CALL CalDenspl_HDGB(DN_HDGB(I),DDN_HDGB(I),Z(i), &
               DON_HDGB, DWN_HDGB,DMINN_HDGB,DMAXN_HDGB,UNDN_HDGB, &
               ZMIN_N,ZMAX_N,i)
             CALL CalVdw(EPSNVDW, SIGNVDW, EPS_I, SIG_I, &
               GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DN_HDGB(I), &
               UNDN_HDGB,Z(I),DDN_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DN_HDGB(I) .ne. 0 ).and.(DDN_HDGB(I) .ne. 0)) THEN
                DDD_HDGB(I)= DDD_HDGB(I) &
                       - GVDW_HDGB/DN_HDGB(I)*DDN_HDGB(I) 
         
             ELSE
                DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF
           ENDIF
 
           IF (UNDC2_HDGB .NE. -1) THEN
             CALL CalDenspl_HDGB(DC2_HDGB(I),DDC2_HDGB(I),Z(i), &
              DOC2_HDGB, DWC2_HDGB,DMINC2_HDGB,DMAXC2_HDGB, &
              UNDC2_HDGB,ZMIN_C2,ZMAX_C2,i)
             CALL CalVdw(EPSC2VDW, SIGC2VDW, EPS_I, SIG_I, &
              GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DC2_HDGB(I), &
              UNDC2_HDGB,Z(I),DDC2_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DC2_HDGB(I) .ne. 0 ).and.(DDC2_HDGB(I) .ne. 0)) THEN
               DDD_HDGB(I)= DDD_HDGB(I) &
                       - GVDW_HDGB/DC2_HDGB(I)*DDC2_HDGB(I)
             ELSE
               DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF

           ENDIF

           IF (UNDC_HDGB .NE. -1) THEN
             CALL CalDenspl_HDGB(DC_HDGB(I),DDC_HDGB(I),Z(i), &
              DOC_HDGB, DWC_HDGB,DMINC_HDGB,DMAXC_HDGB, &
              UNDC_HDGB,ZMIN_C,ZMAX_C,i)
             CALL CalVdw(EPSCVDW, SIGCVDW, EPS_I, SIG_I, &
              GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DC_HDGB(I), &
              UNDC_HDGB,Z(I),DDC_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DC_HDGB(I) .ne. 0 ).and.(DDC_HDGB(I) .ne. 0)) THEN
                 DDD_HDGB(I)= DDD_HDGB(I) &
                         - GVDW_HDGB/DC_HDGB(I)*DDC_HDGB(I)
             ELSE
                 DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF
           ENDIF

           IF (UNDO_HDGB .NE. -1) THEN
             CALL CalDenspl_HDGB(DO_HDGB(I),DDO_HDGB(I),Z(i), &
              DOO_HDGB, DWO_HDGB,DMINO_HDGB,DMAXO_HDGB, &
              UNDO_HDGB,ZMIN_O,ZMAX_O,i)
             CALL CalVdw(EPSOVDW, SIGOVDW, EPS_I, SIG_I, &
              GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DO_HDGB(I), &
              UNDO_HDGB,Z(I),DDO_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             T(I)= T(I) + TVDW(I)
             DP(I)=DP(I)+PVDW(I)
             IF ((DO_HDGB(I) .ne. 0 ).and.(DDO_HDGB(I) .ne. 0)) THEN
                 DDD_HDGB(I)= DDD_HDGB(I) &
                         - GVDW_HDGB/DO_HDGB(I)*DDO_HDGB(I)
             ELSE
                 DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF
           ENDIF

           IF (UNDP_HDGB .NE. -1) THEN
             CALL CalDenspl_HDGB(DP_HDGB(I),DDP_HDGB(I),Z(i), &
              DOP_HDGB, DWP_HDGB,DMINP_HDGB,DMAXP_HDGB, &
              UNDP_HDGB,ZMIN_P,ZMAX_P,i)
             CALL CalVdw(EPSPVDW, SIGPVDW, EPS_I, SIG_I, &
              GVDW_HDGB ,Alph_GB(I), AVDW(I), TVDW(I),DP_HDGB(I), &
              UNDP_HDGB,Z(I),DDP_HDGB(I),PVDW(I) )
             GVDW = GVDW - GVDW_HDGB
             
             DP(I)=DP(I)+PVDW(I)
             IF ((DP_HDGB(I) .ne. 0 ).and.(DDP_HDGB(I) .ne. 0)) THEN
                 DDD_HDGB(I)= DDD_HDGB(I) &
                         - GVDW_HDGB/DP_HDGB(I)*DDP_HDGB(I)
             ELSE
                 DDD_HDGB(I)= DDD_HDGB(I)+0
             ENDIF
           ENDIF

         END DO         
!         WRITE(6,*)'GB Energy>> ' ,GVDW

       ELSE

        X1VDW   = EIGHT*PI*(0.033428D0)/THREE
        EPSWVDW = 0.1520D0
        SIGWVDW = 1.7682D0
        GVDW    = ZERO
        DO I = 1, nmvsel
            EPSIW = SQRT(ABS(EFF(ITC(IAC(I))))*EPSWVDW)
            SIGIW = VDWR(ITC(IAC(I))) + SIGWVDW
            SIGIW = SIGIW*SIGIW
            SIGIW = SIGIW*SIGIW*SIGIW
            X2VDW = ONE/(Alph_GB(I)+1.4D0)
            X3VDW = X2VDW*X2VDW*X2VDW
            X4VDW = X1VDW*AVDW(I)*EPSIW*SIGIW
            GVDW  = GVDW &
                 - X4VDW*X3VDW
            TVDW(I) = THREE*X4VDW*X3VDW*X2VDW
            T(I) = T(I) + TVDW(I)
           WRITE(6,*) 'GB i ->  ' ,GVDW,I

        END DO

       ENDIF
      ENDIF
! MS/MF
#else


    IF (QGBVDW) THEN
       X1VDW   = EIGHT*PI*(0.033428D0)/THREE
       EPSWVDW = 0.1520D0
       SIGWVDW = 1.7682D0
       GVDW    = ZERO
       DO I = 1, nmvsel
          EPSIW = SQRT(ABS(EFF(ITC(IAC(I))))*EPSWVDW)
          SIGIW = VDWR(ITC(IAC(I))) + SIGWVDW
          SIGIW = SIGIW*SIGIW
          SIGIW = SIGIW*SIGIW*SIGIW
          X2VDW = ONE/(Alph_GB(I)+1.4D0)
          X3VDW = X2VDW*X2VDW*X2VDW
          X4VDW = X1VDW*AVDW(I)*EPSIW*SIGIW
          GVDW  = GVDW  &
               - X4VDW*X3VDW
          TVDW(I) = THREE*X4VDW*X3VDW*X2VDW
          T(I) = T(I) + TVDW(I)
       END DO
    END IF

    ! SJT VDW
!MS/MF
#endif

#if KEY_PARALLEL==1
    CALL GComb(T,nmvsel)
    if ((CORR_GB .eq. 3) .or. &
         (CORR_GB .eq. 4) .or. &
         (CORR_GB .eq. 5)) then
       CALL GComb(P_HDGB,nmvsel)
    endif
#endif 
    ! WRITE(OUTU,*) 'GB Energy =',GBEnergy

    call timer_stop(T_GB_ENERG) 

    IF (QUpdateAlpha) THEN
       !------------------------------------------
       ! Compute forces
       !------------------------------------------

       call timer_start(T_GB_FORCE) 

       DFSw_X = ZERO
       FSw_X = ONE
       l = 0

       I2 = 0
#if KEY_PARALLEL==1
       DO i=MyNodP, nmvsel, NumNod
#else /**/
       DO i=1,nmvsel
#endif 

          CX = X(i) - XMIN
          CY = Y(i) - YMIN
          CZ = Z(i) - ZMIN

          E_ALPHA = T(i)

          if (CORR_GB .eq. 3) then
             DZ(i) = DZ(i) - HALF*CCELEC*P_HDGB(i)*DEPS_HDGB(i) &
                  - E_ALPHA * S_HDGB(i) * DEPS_HDGB(i)
#if KEY_HDGBVDW==1
! MS/MF
                if ( (UNDO_HDGB .NE. -1) .or. &
                    (UNDC_HDGB .NE. -1) .or. &
                    (UNDC2_HDGB .NE. -1) .or. &
                    (UNDP_HDGB .NE. -1) .or. &
                    (UNDN_HDGB .NE. -1) ) then
                   DZ_GBVdw(i) = DZ_GBVdw(i) + DDD_HDGB(i) 
               endif
! MS/MF
#endif

             ! HDGB2
          else if (CORR_GB .eq. 4) then
             X0_HDGB = (HALF*CCELEC*P_HDGB(i)*DEPS_HDGB(i) &
                  + E_ALPHA * S_HDGB(i) * DEPS_HDGB(i))/RDIST_HDGB(i)
             DX(i) = DX(i) - X0_HDGB*X(i)
             DY(i) = DY(i) - X0_HDGB*Y(i)
             DZ(i) = DZ(i) - X0_HDGB*Z(i)
          else if (CORR_GB .eq. 5) then
             X0_HDGB = (HALF*CCELEC*P_HDGB(i)*DEPS_HDGB(i) &
                  + E_ALPHA * S_HDGB(i) * DEPS_HDGB(i))/RDIST_HDGB(i)
             DX(i) = DX(i) - X0_HDGB*X(i)
             DY(i) = DY(i) - X0_HDGB*Y(i)
             ! HDGB2
          endif

          DO J=1,NWeights
             I2 = I2 + 1
             IF (Surf_Atom(I2).EQ.-1) THEN
                Rx = WtX(j) + cx + xmin
                Ry = WtY(j) + cy + ymin
                Rz = WtZ(j) + cz + zmin

                ctr = 0

                SX = ZERO
                SY = ZERO
                SZ = ZERO
                A = ZERO
                C = ZERO
                R7 = ZERO
                R1 = ZERO

                ! (R2 .LT. rtmp(ki+5)) .AND. (R2 .LT. rtmp(ki+3))

                II2=(I2-1)*6

                DO ps=XIndex(II2+1),XIndex(II2+2)
                   k=BList(PS)

                   VX = RX - X(k)
                   VY = RY - Y(k)
                   VZ = RZ - Z(k)
                   R2 = VX * VX + VY * VY + VZ * VZ

                   ki=(k-1)*9
                   WK=rtmp(ki+1)

                   ctr = ctr + 1
                   ictr=(ctr-1)*7

                   TEMP1_GB(ctr) = k

                   VTMP(ictr+1)=VX
                   VTMP(ictr+2)=VY
                   VTMP(ictr+3)=VZ

                   X5 = P2*WK + P1 ! P2 * R(ii) + P1
                   X2 = WK*WK
                   X3 = R2 - X2
                   X4 = X5+X3
                   X4 = ONE/X4
                   R3 = X5*X4
                   R3 = R3*R3
                   X1 = -FOUR*R3*X4

                   ! Hard-sphere tail
                   X5=rtmp(ki+7)
                   X4 = (R2-rtmp(ki+2))*X5

                   X2 = X4*X4
                   X3 = X2*X4
                   FSw_X=ONE +X3*(X4*(15.0D0-SIX*X4)-TEN)
                   DfSw_X = SIXTY*X2*(TWO*X4-X2-ONE)*X5

                   R7 = R7 + TWO*FSw_X
                   VTMP(ictr+4)=TWO*DfSw_X

                   DfSw_X = DfSw_X * R3
                   X1 = X1 * (ONE-FSw_X) - DfSw_X
                   R3 = R3 * (ONE-FSw_X)

                   VTMP(ictr+5)=X1
                   VTMP(ictr+6)=R3
                   VTMP(ictr+7)=R2

                   SX = SX + VX*R3
                   SY = SY + VY*R3
                   SZ = SZ + VZ*R3

                   a=a+r3*r3*r2
                   C = C + R3
                ENDDO

                ! (R2 .LT. rtmp(ki+5)) .AND. (R2 .GE. rtmp(ki+3) .AND. (R2.LE.rtmp(ki+4))

                ! most common!!!
                DO ps=XIndex(II2+3),XIndex(II2+4)
                   k=CList(PS)

                   VX = RX - X(k)
                   VY = RY - Y(k)
                   VZ = RZ - Z(k)
                   R2 = VX * VX + VY * VY + VZ * VZ

                   ki=(k-1)*9
                   WK=rtmp(ki+1)

                   ctr = ctr + 1 ! that exist in VSA region
                   ictr=(ctr-1)*7

                   TEMP1_GB(ctr) = k

                   VTMP(ictr+1)=VX
                   VTMP(ictr+2)=VY
                   VTMP(ictr+3)=VZ

                   X5 = P2*WK + P1
                   X2 = WK*WK
                   X3 = R2 - X2
                   X4 = X5+X3
                   X4 = ONE/X4
                   R3 = X5*X4
                   R3 = R3*R3
                   X1 = -FOUR*R3*X4

                   VTMP(ictr+4)=ZERO
                   VTMP(ictr+5)=X1
                   VTMP(ictr+6)=R3
                   VTMP(ictr+7)=R2

                   SX = SX + VX*R3
                   SY = SY + VY*R3
                   SZ = SZ + VZ*R3

                   a=a+r3*r3*r2
                   C = C + R3
                ENDDO

                ! (R2 .LT. rtmp(ki+5)) .AND. (R2.GT.rtmp(ki+4))

                DO ps=XIndex(II2+5),XIndex(II2+6)
                   k=DList(PS)

                   VX = RX - X(k)
                   VY = RY - Y(k)
                   VZ = RZ - Z(k)
                   R2 = VX * VX + VY * VY + VZ * VZ

                   ki=(k-1)*9
                   WK=rtmp(ki+1)
                   ctr = ctr + 1 ! that exist in VSA region
                   ictr=(ctr-1)*7

                   TEMP1_GB(ctr) = k

                   VTMP(ictr+1)=VX
                   VTMP(ictr+2)=VY
                   VTMP(ictr+3)=VZ

                   X5 = P2*WK + P1 ! P2 * R(ii) + P1
                   X2 = WK*WK
                   X3 = R2 - X2
                   X4 = X5+X3
                   X4 = ONE/X4
                   R3 = X5*X4
                   R3 = R3*R3
                   X1 = -FOUR*R3*X4

                   VTMP(ictr+4)=ZERO
                   ! VSA tail
                   X4=rtmp(ki+4)
                   X5=rtmp(ki+6)
                   X3 = (R2-X4)*X5

                   X2 = X3*X3
                   X4 = X2*X3

                   FSw_X = ONE +  &
                        X4*(X3*(15.0D0-SIX*X3)-TEN)
                   DfSw_X = SIXTY*X2*(TWO*X3-X2-ONE)*X5

                   DfSw_X = DfSw_X * R3
                   R3 = R3 * FSw_X
                   X1 = X1 * FSw_X + DfSw_X

                   VTMP(ictr+5)=X1
                   VTMP(ictr+6)=R3
                   VTMP(ictr+7)=R2

                   SX = SX + VX*R3
                   SY = SY + VY*R3
                   SZ = SZ + VZ*R3

                   a=a+r3*r3*r2
                   C = C + R3
                ENDDO

                ! Check inter-atom region

                IF (ctr.gt.0) THEN
                   B = SX*SX + SY*SY + SZ*SZ
                   IF (B .LT. 1.0D-18) B=1.0D-18
                   R1 = C * P3 * A/B
                ENDIF

                R1 = R1 + R7  ! Hard-sphere

                R2 = ONE + DEXP(BETA * (R1-LAMBDA1))
                IMPULSE = -BETA * (R2 - ONE) / (R2*R2)

                R2 = IMPULSE * E_ALPHA

                IF (CORR_GB .EQ. 0) THEN
                   S = (Alph_Gb(i)-shift)*(Alph_Gb(i)-shift)
                   R1 = R2 * Wt1(j) * S
                   R3 = -TT * HALF * R2 * Wt2(j) * S / F(i)
                ELSEIF (CORR_GB .EQ. 1) THEN
                   S = (Alph_Gb(i)-shift)*(Alph_Gb(i)-shift)/(SLOPE)
                   R1 = -(ONE-DSQRT(HALF))*R2 * Wt1(j) * S
                   R3 = -PT25*R2*Wt4(j) * S * F(i)**(-0.75D0)
                ELSEIF (CORR_GB .EQ. 2) THEN
                   S = (Alph_Gb(i)-shift)*(Alph_Gb(i)-shift)/(SLOPE)
                   R1 = -A1*R2*Wt1(j)*S
                   R3 = S*(-A2*HALF*R2*Wt2(j)/F(i)- &
                        A3*PT25*R2*Wt4(j)*G(i)**(-0.75D0))
                ELSEIF ((CORR_GB .EQ. 3) .or. &
                     (CORR_GB .EQ. 4) .or. &
                     (CORR_GB .EQ. 5)) THEN
                   S = (Alph_Gb(i)-Shift_HDGB(i))* &
                        (Alph_Gb(i)-Shift_HDGB(i))/(SLOPE)
                   R1 = -A1*R2*Wt1(j)*S
                   R3 = S*(-A2*HALF*R2*Wt2(j)/F(i)- &
                        A3_HDGB(i)*PT25*R2*Wt4(j)*G(i)**(-0.75D0))
                ENDIF

                R3 = R3 + R1

                IF (CTR .GT. 0) THEN
                   R4 = R3
                   R3 = -R4 * P3
                   X1 = ONE/B
                   X2 = A*X1*R3
                   X3 = C*X1
                   X4 = X2*X3*TWO
                   X3 = X3*R3

                   DO K = 1,CTR
                      ictr=(k-1)*7

                      iVX=VTMP(ictr+1)
                      iVY=VTMP(ictr+2)
                      iVZ=VTMP(ictr+3)
                      X5=R4*VTMP(ictr+4)
                      X1=VTMP(ictr+5)
                      R3=VTMP(ictr+6)
                      R2=VTMP(ictr+7)

                      dMX = X1 * iVX
                      dMY = X1 * iVY
                      dMZ = X1 * iVZ

                      X6=iVX*SX+iVY*SY+iVZ*SZ
                      TX=R3*SX+dMX*X6
                      TY=R3*SY+dMY*X6
                      TZ=R3*SZ+dMZ*X6

                      VX=iVX*R3+R2*dMX
                      VY=iVY*R3+R2*dMY
                      VZ=iVZ*R3+R2*dMZ

                      X1=-TWO*R3*X3
                      VX=X1*VX-X2*dMX
                      VY=X1*VY-X2*dMY
                      VZ=X1*VZ-X2*dMZ

                      VX=VX+X4*TX+X5*iVX
                      VY=VY+X4*TY+X5*iVY
                      VZ=VZ+X4*TZ+X5*iVZ

                      DX_GB(i)  = DX_GB(i)  - VX
                      DY_GB(i)  = DY_GB(i)  - VY
                      DZ_GB(i)  = DZ_GB(i)  - VZ

                      k2 = TEMP1_GB(k)
                      DX_GB(k2) = DX_GB(k2) + VX
                      DY_GB(k2) = DY_GB(k2) + VY
                      DZ_GB(k2) = DZ_GB(k2) + VZ

                   ENDDO

                ENDIF
             ENDIF
          ENDDO
       ENDDO

       call timer_stop(T_GB_FORCE) 

    ENDIF                     !IF (QUpdateAlpha) THEN

    IF (IMPULSE_GB .EQ. 5) THEN
       FORCE_SCALE=ONE
    ELSEIF (IMPULSE_GB .EQ. 3) THEN ! emp method
       IF (QUpdateAlpha) THEN
          FORCE_SCALE = EMPB_GB ! Start with biggest
       ELSE
          FORCE_SCALE = FORCE_SCALE * EMPA_GB ! Decrease each step
       ENDIF
    ELSEIF (IMPULSE_GB .EQ. 2) THEN !limp method
       IF (QUpdateAlpha) THEN
          FORCE_SCALE = ALFRQ
       ELSE
          FORCE_SCALE = ZERO
       ENDIF
    ELSEIF (IMPULSE_GB .EQ. 4) THEN !simp method
       FORCE_SCALE = ONE/ALFRQ
    ELSE
       IF (QUpdateAlpha) THEN !imp method by default
          FORCE_SCALE = ONE
       ELSE
          FORCE_SCALE = 0.0
       ENDIF
    ENDIF

    ! write(*,*) 'force_scale: ',force_scale
    IF (FORCE_SCALE .GT. 1E-10) THEN
       DO I=1,nmvsel
          DX(I)=DX(I)+FORCE_SCALE*DX_GB(I)
          DY(I)=DY(I)+FORCE_SCALE*DY_GB(I)
          DZ(I)=DZ(I)+FORCE_SCALE*DZ_GB(I)
#if KEY_HDGBVDW==1
! MS/MF
        if (  (QGBVDW) .and.( ( UNDO_HDGB .NE. -1) .or. &
          (UNDP_HDGB .NE. -1) .or. (UNDC_HDGB .NE. -1) .or. &
          (UNDC2_HDGB .NE. -1) .or. (UNDN_HDGB .NE. -1) ) ) then

            DZ(I)=DZ(I)+DZ_GBVdw(I)
        endif
! MS/MF
#endif

       ENDDO
    ENDIF

    ! IF ((.NOT. (IMPULSE_GB .OR. IMPULSE2_GB)).OR.(QUpdateAlpha)) THEN
    !    IF (IMPULSE2_GB) THEN
    !       FORCE_SCALE = ALFRQ
    !    ELSE IF (EMP_GB .LE. 1D+10) THEN
    !       FORCE_SCALE = ONE
    !    ENDIF
    !    IF (IMPULSE3_GB) THEN
    !       FORCE_SCALE = ALFRQ
    !       FORCE_SCALE = ONE/FORCE_SCALE
    !    ENDIF
    !    IF ((.not. IMPULSE2_GB) .AND. (EMP_GB .GT. ZERO)) THEN
    !    ENDIF
    !    c       WRITE(2,*)DX_GB(1),DY_GB(1),DZ_GB(1)
    ! ENDIF

    if ((CORR_GB .eq. 3) .or. &
         (CORR_GB .eq. 4) .or. &
         (CORR_GB .eq. 5)) then
#if KEY_HDGBVDW==1
        if (  (QGBVDW) .and.( ( UNDO_HDGB .NE. -1) .or. &
          (UNDP_HDGB .NE. -1) .or. (UNDC_HDGB .NE. -1) .or. &
          (UNDC2_HDGB .NE. -1) .or. (UNDN_HDGB .NE. -1) ) ) then
              ETERM(ASP) =  GVDW +Gnp_HDGB             
        else
#endif


       ETERM(ASP) = Gnp_HDGB

#if KEY_HDGBVDW==1
        endif
#endif

    else if (QGBVDW) then
       ETERM(ASP) = Gnp_VDW + GVDW
    else if (QGBASP) then
       ETERM(ASP) = Gnp_ASP
    else
       ETERM(ASP) = Surface_Area * SA_GB
    endif

#if KEY_GBMVDEBUG==1
    IF (ESURF_GB) CLOSE(2)    
#endif

#if KEY_PARALLEL==1
    IF (MyNodP.eq.1) ETERM(ASP) = ETERM(ASP) + SB_GB
#else /**/
    ETERM(ASP) = ETERM(ASP) + SB_GB
#endif 

    ! WRITE(OUTU,909) Surface_Area
909       FORMAT('Surface area = ',F10.4)

    RETURN
  END   Subroutine RunGBMV1F

  Subroutine RunGBMVGrid(GBEnergy, &
       JNbl,INbl,Inbl14,INblo14)
    !------------------------------------------------------------
    ! Called from Energy, this re-allocates memory if
    ! necessary and then calls the alpha, and energy
    ! builder.
    !------------------------------------------------------------
  use exfunc
  use stream
  use dimens_fcm
  use psf
  use number
  use coord
  use consta
  use memory

    real(chm_real) GBEnergy
    INTEGER OLDNGrid
    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)

    OLDNGrid = NGridGB

    CALL FindExtents(0)

    call chmrealloc('gbmvmodule.src','RunGBMVGrid','grid',NGridGB+4,iby=grid)
    if(conv_gb)then
       call chmrealloc('gbmvmodule.src','RunGBMVGrid','grid2',ngridgb+4,iby=grid2)
    else
       call chmrealloc('gbmvmodule.src','RunGBMVGrid','grid2',100,iby=grid2)
    endif

    IF (OLDNGrid .ne. NGridGB) THEN
       IF (PRNLEV .GE. 5) THEN
          WRITE(OUTU,*) 'Readjusting grid size'
          WRITE(OUTU,*) 'New Extents: nx , ny , nz = ',NX,NY,NZ
          WRITE(OUTU,*) 'Grid size',NGridGB
       ENDIF
    ENDIF
    GBGridReset = .true.

    Call       RunGBMVGrid1(GBEnergy,JNbl,INbl,Inbl14,INblo14)
    RETURN
  END Subroutine RunGBMVGrid

  Subroutine RunGBMVGrid1(GBEnergy, JNbl,INbl,Inbl14,INblo14)
    !------------------------------------------------------------
    ! Build alpha array and some derivative components
    !------------------------------------------------------------
  use exfunc
  use stream
  use dimens_fcm
  use param
  use psf
  use number
  use coord
  use consta
  use inbnd
#if KEY_PARALLEL==1
  use parallel
#endif 
    real(chm_real) RE,RG,RH,RI,RJ,R1,R3,R4,R5,R1A,R6
    real(chm_real) R,SX,SY,SZ,S,VX,VY,VZ,TX,TY,TZ
    real(chm_real) C1,C2,C3,C4,Q,DD,A,CX,CY,CZ
    real(chm_real) XDIFF,YDIFF,ZDIFF,Rx,Ry,Rz,Impulse,E_Alpha
    real(chm_real) GBEnergy,RMax,XE1,XE2,YE1,YE2,ZE1,ZE2
    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)
    Integer i,j,k,l,PS,IX,IY,IZ,POS,ii,k2
    real(chm_real) C2OfNb
    Integer ITemp, NPr, jpr
    real(chm_real) C2OnNb, Rul3, Rul12, RijL, RijU, FSw, DFSw
    Logical LOuter, Switch
    real(chm_real) Salt,TOTDIFF
    !---------------------------------
    ! Compute Alphas for every atom
    !---------------------------------

    ! Generate radii and exponent values

    Switch = LCons .and. .not. LShft .and. .not. LFSwt

    DO i=1,nmvsel
       IF (QWeight) THEN
          Rad(i) = WMAIN(i)
       ELSE
          Rad(i) = VDWR(ITC(IAC(I)))
          WMAIN(i) = Rad(i)
       ENDIF

       j = NumR
       DO WHILE (WTR(j) .gt. Rad(i))
          j = j - 1
       ENDDO
       if (j .ge. 1) THEN
          Rad(i) = Wtr(j)
       else
          Rad(i) = Wtr(1)
       ENDIF
    ENDDO

    TOTDIFF = ZERO
    DO I=1,nmvsel
       R1 = (X(I)-XCP(I))*(X(I)-XCP(I))+ &
            (Y(I)-YCP(I))*(Y(I)-YCP(I))+ &
            (Z(I)-ZCP(I))*(Z(I)-ZCP(I))
       TOTDIFF = TOTDIFF + R1
    ENDDO

    ! don't redo alphas if structure same.
    IF (.not. ((TOTDIFF .LT. 1D-20).AND.(FIXA))) THEN

       DO I=1,nmvsel
          XCP(I) = X(I)
          YCP(I) = Y(I)
          ZCP(I) = Z(I)
       ENDDO

       CALL BuildGBGrid()

       DO i=1,nmvsel
          Alph_Gb(i) = ZERO
       ENDDO

#if KEY_GBMVDEBUG==1
       IF (ECOMP_GB) OPEN(1,file='comp.txt',status='unknown') 
#endif

#if KEY_PARALLEL==1
       DO i = MyNodP, nmvsel, NumNod
#else /**/
       DO i=1,nmvsel
#endif 
          CX = X(i) - XMIN
          CY = Y(i) - YMIN
          CZ = Z(i) - ZMIN

          RI = Rad(i)
          RE = ONE / RI
          RG = HALF / (RI*RI)
          RH = ONE/((RI*RI*RI*RI)*FOUR)

          XE1 = (X(i)-XMIN)
          XE2 = (XMAX-X(i))
          YE1 = (Y(i)-YMIN)
          YE2 = (YMAX-Y(i))
          ZE1 = (Z(i)-ZMIN)
          ZE2 = (ZMAX-Z(i))
          IF (XE1 .gt. XE2) THEN
             RMAX = XE1*XE1
          ELSE
             RMAX = XE2*XE2
          ENDIF
          IF (YE1 .gt. YE2) THEN
             RMAX = RMAX + YE1*YE1
          ELSE
             RMAX = RMAX + YE2*YE2
          ENDIF
          IF (ZE1 .gt. ZE2) THEN
             RMAX = RMAX + ZE1*ZE1
          ELSE
             RMAX = RMAX + ZE2*ZE2
          ENDIF
          RMAX = DSQRT(RMAX)
          DO j=1,NWeights
             R = WTS(j)
             IF ((R .ge. RI).and.(R .lt. RMAX)) THEN
                Rx = WtX(j) + cx
                Ry = WtY(j) + cy
                Rz = WtZ(j) + cz
                IX = Rx * dn_inv + 1D-12 !+ HALF
                IY = Ry * dn_inv + 1D-12 !+ HALF
                IZ = Rz * dn_inv + 1D-12 !+ HALF
                IF ((IX .ge. 0) .and. (IY .ge. 0)  &
                     .and. (IZ .ge. 0) .and. &
                     (IX .lt. Nx) .and. (IY .lt. Ny)  &
                     .and. (IZ .lt. Nz)) THEN
                   POS = IX + IY * Nx + IZ * Nxy - OFFSET
                   IF (Grid(POS) .GE. 1) THEN
                      RE = RE - Wt1(j)
                      RG = RG - Wt2(j)
                      RH = RH - Wt4(j)
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
          IF (CORR_GB .EQ. 1) THEN
             Alph_Gb(i) = SLOPE/((ONE-DSQRT(HALF))*RE +  &
                  DSQRT(DSQRT(RH))) &
                  + SHIFT
          ELSEIF (CORR_GB .EQ. 0) THEN
             Alph_Gb(i) = -ONE/(RE - TT * DSQRT(RG)+ESHIFT) + SHIFT
          ELSE
             Alph_Gb(i)=SLOPE/ &
                  (A1*RE+A2*DSQRT(RG)+A3*RH**(PT25)+ESHIFT) &
                  +SHIFT
          ENDIF
#if KEY_GBMVDEBUG==1
          IF (ECOMP_GB) WRITE(1,'(4(F15.10))')RE,RG,ZERO,RH 
#endif
       ENDDO

#if KEY_PARALLEL==1
       CALL GComb(Alph_Gb,nmvsel)
#endif 

       !-------------------------------------------------
       ! Forming energy
       !-------------------------------------------------
#if KEY_GBMVDEBUG==1
       CLOSE(1)               
#endif
    ENDIF                     ! update alpha question

    ! WRITE(OUTU,*) 'Forming energy...'

    ! First, do excluded list
    GBEnergy = ZERO
    ITemp = 0

#if KEY_PARALLEL==1 /*paraexcl*/
    DO i=MyNodP, nmvsel, NumNod
#else /* (paraexcl)*/
    DO i=1,nmvsel
#endif /* (paraexcl)*/

       If (i .ne. 1) ITemp = INblo14(i - 1)
       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          IF (k .gt. 0) THEN
             XDIFF = X(i) - X(j)
             YDIFF = Y(i) - Y(j)
             ZDIFF = Z(i) - Z(j)
             R1 = XDIFF * XDIFF + YDIFF * YDIFF + ZDIFF * ZDIFF
             IF (MTYPE_GB .EQ. 2) THEN
                A = Alph_Gb(i) * Alph_Gb(j)
             ELSE
                A = Alph_Gb(i) + Alph_Gb(j)
                A = A*A/FOUR
             ENDIF
             Q = CG(i) * CG(j)
             DD = DEXP(-P6*R1/A)
             C1 = ONE / (R1 + A*DD)
             C2 = DSQRT(C1)
             SALT = -CCELEC*(ONE/EPS - DEXP(-KAPPA_GB/C2)/EPS_GB)
             GBEnergy = GBEnergy + Q * C2 * SALT
          ENDIF
       ENDDO
    ENDDO

    ! Next, Do non-bonded list

    C2OfNB = CtOfNB * CtOfNB
    FSw = ONE
    DFSw = ZERO
    IF (Switch) THEN
       C2OnNb = CtOnNb * CtOnNb
       If (CtOfNb .gt. CtOnNb) Then
          Rul3 = One / (C2OfNb - C2OnNb)**3
          Rul12 = Twelve * Rul3
       Endif
    Endif


    ITemp = 0
    DO i=1,nmvsel-1
#if KEY_IMCUBES==1
       if (lbycbim) ITemp = INbl(I+nmvsel) 
#endif
       if (CG(i).ne.ZERO) THEN
          R3 = ZERO
          NPr = INbl(i) - ITemp
          Do jpr = 1, NPr
             k = JNbl(Itemp+jpr)
             j = Abs(k)
             IF (CG(j) .ne. zero) THEN
                XDIFF = X(i) - X(j)
                YDIFF = Y(i) - Y(j)
                ZDIFF = Z(i) - Z(j)
                R1 = XDIFF * XDIFF + YDIFF * YDIFF + ZDIFF * ZDIFF

                ! The Electrostatic Switch Function

                If (R1 .lt. C2OfNb) Then

                   FSw = One
                   DFSw = Zero
                   If (Switch) Then
                      LOuter = (R1 .gt. C2OnNb)
                      If (LOuter) Then
                         RijL = C2OnNb - R1
                         RijU = C2OfNb - R1
                         FSw = RijU * RijU *  &
                              (RijU - Three * RijL) * Rul3
                         DfSw = RijL * RijU * Rul12 / FSw
                      Endif
                   Endif

                   IF (MTYPE_GB .EQ. 2) THEN
                      A = Alph_Gb(i) * Alph_Gb(j)
                   ELSE
                      A = Alph_Gb(i) + Alph_Gb(j)
                      A = A*A/FOUR
                   ENDIF

                   Q = CG(i) * CG(j)
                   DD = DEXP(-P6*R1/A)
                   C1 = ONE / (R1 + A*DD)
                   C2 = DSQRT(C1)
                   SALT = -CCELEC*(ONE/EPS -  &
                        DEXP(-KAPPA_GB/C2)/EPS_GB)
                   S = Q * C2 * FSw * SALT
                   GBEnergy = GBEnergy + S
                ENDIF
             ENDIF
          ENDDO
       ENDIF
       ITemp = INbl(i)
    ENDDO

    ! Finally, the self terms

#if KEY_PARALLEL==1
    DO i=MyNodP,nmvsel,NumNod
#else /**/
    DO i=1,nmvsel
#endif 
       Q = CG(i) * CG(i)
       C2 = ONE / Alph_Gb(i)
       SALT = -CCELEC*(ONE/EPS - DEXP(-KAPPA_GB/C2)/EPS_GB)
       GBEnergy = GBEnergy + Q * C2 * Half * SALT
    ENDDO

    ! WRITE(OUTU,*) 'GB Energy =',GBEnergy

    return
  end Subroutine RunGBMVGrid1

  ! SJT/MF VDW
  SUBROUTINE SetVDWA()
  use dimens_fcm
  use psf
  use param
  use stream

    !-----------------------------------------------------------------------
    !
    ! The nonpolar parameters are taken from
    !    E. Gallicchio and RM Levy, J Comput Chem 25: 479--499,2004
    !
    !-----------------------------------------------------------------------

    INTEGER I

    if (prnlev > 2) WRITE(OUTU,*) 'Setting up the coefficients for VDW term'

#if KEY_HDGBVDW==1
!MS/MF
      DO I = 1, nmvsel

         IF (ATC(IAC(I))(1:1) .EQ. 'C') THEN
           IF ((ATC(IAC(I))(1:2) .EQ. 'CA') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C145') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C514') ) THEN
               avdw(I) = A_CA
!               WRITE(OUTU,*) 'CA: ',ATC(IAC(I))(1:4)

            ELSE IF((ATC(IAC(I))(1:3) .EQ. 'CT3' ) &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C135') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C209')) THEN
!               WRITE(OUTU,*) 'CT3: ',ATC(IAC(I))(1:4)
               avdw(I) = A_CT3
            ELSE IF((ATC(IAC(I))(1:2) .EQ. 'CM')) THEN
!               WRITE(OUTU,*) 'CM: ',ATC(IAC(I))(1:4)
               avdw(I) = A_CM
            ELSE IF((ATC(IAC(I))(1:3) .EQ. 'CPT' ) &
              .or. (ATC(IAC(I))(1:2) .EQ. 'CY') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C501') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C502') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C166') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C503')) THEN
!               WRITE(OUTU,*)'CPT: ', ATC(IAC(I))(1:4)
               avdw(I) = A_CY
            ELSE IF ((ATC(IAC(I))(1:3) .EQ. 'CPH' ) &
!His
              .or. (ATC(IAC(I))(1:4) .EQ. 'C506' ) &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C507' ) &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C508' )) THEN
               avdw(I) = A_CPH
!               WRITE(OUTU,*)'CPH: ', ATC(IAC(I))(1:4)

            ELSE IF((ATC(IAC(I))(1:2) .EQ. 'CC' ) &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C235') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C236')) THEN
               avdw(I) = A_CC
!               WRITE(OUTU,*) 'CC:' ,ATC(IAC(I))(1:4)
           ELSE IF((ATC(IAC(I))(1:3) .EQ. 'CT1' ) &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C158') &
              .or. (ATC(IAC(I))(1:4) .EQ. 'C137')) THEN
               avdw(I) = A_CT1
!               WRITE(OUTU,*) 'CT1: ', ATC(IAC(I))(1:4)

            ELSE
!               WRITE(OUTU,*) 'CT2: ',ATC(IAC(I))(1:4)
               avdw(I) = A_CT2
            ENDIF
!            WRITE (*,*) 'Debug -> Carbon atom', ATC(IAC(I))(1:4), &
!                'a values:', avdw(I)

         ELSE IF (ATC(IAC(I))(1:1) .EQ. 'H') THEN

            IF  ((ATC(IAC(I))(1:4) .EQ. 'H204' ) &
              .or. (ATC(IAC(I))(1:2) .EQ. 'HS')) THEN
               avdw(I) = A_HS
!               WRITE(OUTU,*) 'HS: ',ATC(IAC(I))(1:4)

            ELSE IF  ((ATC(IAC(I))(1:4) .EQ. 'H155' )) THEN
!               avdw(I) = 1.30D0
               avdw(I) = A_HO
!               WRITE(OUTU,*) 'HO: ',ATC(IAC(I))(1:4)

            ELSE IF  ((ATC(IAC(I))(1:4) .EQ. 'H504' )) THEN
!H on NE in TRP
!               avdw(I) = 0.70D0
               avdw(I) = A_HY
!               WRITE(OUTU,*) 'HT: ',ATC(IAC(I))(1:4)
            ELSE IF  ((ATC(IAC(I))(1:2) .EQ. 'H ')) THEN
!               avdw(I) = 1.50D0
               avdw(I) = A_HM
!               WRITE(OUTU,*) 'H Meth: ',ATC(IAC(I))(1:4)
            ELSE IF  ((ATC(IAC(I))(1:4) .EQ. 'H146' ) &
              .or. (ATC(IAC(I))(1:2) .EQ. 'HP')) THEN
!Benzene H - 12 site
!               avdw(I) = 0.50D0
               avdw(I) = A_HP
!               WRITE(OUTU,*) 'HP: ',ATC(IAC(I))(1:4)
           ELSE

!for amino acids:

!               avdw(I) = 0.75D0
               avdw(I) = A_HR
!               WRITE(OUTU,*) 'H ALIPHATIC ANALOGS: ',ATC(IAC(I))(1:4)

            END IF
!            WRITE (*,*) 'Debug -> H atom:',ATC(IAC(I))(1:4),&
!                 'a value:', avdw(I)

         ELSE IF (ATC(IAC(I))(1:1) .EQ. 'N') THEN
            IF  ((ATC(IAC(I))(1:4) .EQ. 'N237' ) &
               .or. (ATC(IAC(I))(1:2) .EQ. 'NH')) THEN
!               WRITE(OUTU,*) 'NH: ',ATC(IAC(I))(1:4)
!               avdw(I) = 0.30D0
               avdw(I) = A_NH
            ELSE
!               WRITE(OUTU,*) 'NY: ',ATC(IAC(I))(1:4)
!               avdw(I) = 0.25D0
               avdw(I) = A_NY
            END IF
!            WRITE (*,*) 'Debug->N atom:',ATC(IAC(I))(1:4),&
!                'a value:',avdw(I)

         ELSE IF (ATC(IAC(I))(1:1) .EQ. 'O') THEN
           IF  ((ATC(IAC(I))(1:2) .EQ. 'OH' ) &
              .or. (ATC(IAC(I))(1:4) .EQ. 'O154')) THEN
!               avdw(I) = 1.50D0
               avdw(I) = A_OH
           ELSE IF  ((ATC(IAC(I))(1:4) .EQ. 'O167' ))  THEN
!oxygen on Tyr
!               avdw(I) = 0.80D0
               avdw(I) = A_OY
!           ELSE IF  ((ATC(IAC(I))(1:4) .EQ. 'O236' ))  THEN
!oxygen on Gln and Asn 
            ELSE
!              avdw(I) = 1.00D0
               avdw(I) = A_OR
           END IF
!           WRITE (*,*)'Debug->oxygen atom:',ATC(IAC(I))(1:4),&
!             'a value:',avdw(I)
         ELSE IF (ATC(IAC(I))(1:1) .EQ. 'P') THEN
!            avdw(I) = 0.75D0
             avdw(I) = A_P
         ELSE IF (ATC(IAC(I))(1:1) .EQ. 'S') THEN 
          IF (ATC(IAC(I))(1:4) .EQ. 'S200') THEN
!!Cys
!!            avdw(I) = 0.70D0
               avdw(I) = A_SC
          ELSE IF (ATC(IAC(I))(1:4) .EQ. 'S202') THEN
               avdw(I) = A_SM
          ELSE  
               avdw(I) = A_S 
          END IF       
!           WRITE (*,*)'Debug->S atom:',ATC(IAC(I))(1:4),&
!                  'a value:',avdw(I)
         ELSE
            CALL WRNDIE (-1,'<SetVDWA>', &
                        '  VDW SET UP FAIL (TYPE NOT FOUND)')
            WRITE (*,*) 'DEBUG a:' ,ATC(IAC(I))(1:4)
         END IF

!               WRITE(6,*) 'check for VDWA in sub: ',avdw(I),ATC(IAC(I))
      END DO

#else

    DO I = 1, nmvsel

       IF (ATC(IAC(I))(1:1) .EQ. 'C') THEN
          IF (ATC(IAC(I))(1:2) .EQ. 'CT') THEN
             avdw(I) = 0.70D0
          ELSE IF (ATC(IAC(I))(1:2) .EQ. 'CN') THEN
             avdw(I) = 1.15D0
          ELSE
             avdw(i) = 0.80D0
          END IF
       ELSE IF (ATC(IAC(I))(1:1) .EQ. 'H') THEN
          avdw(I) = 0.80D0
       ELSE IF (ATC(IAC(I))(1:1) .EQ. 'N') THEN
          avdw(I) = 0.75D0
       ELSE IF (ATC(IAC(I))(1:1) .EQ. 'O') THEN
          avdw(I) = 0.80D0
       ELSE IF (ATC(IAC(I))(1:1) .EQ. 'P') THEN
          avdw(I) = 0.85D0
       ELSE IF (ATC(IAC(I))(1:1) .EQ. 'S') THEN
          avdw(I) = 0.75D0
       ELSE
          CALL WRNDIE (-1,'<SetVDWA>', &
               '  VDW SET UP FAIL (TYPE NOT FOUND)')
       END IF


    END DO

!MS/MF
#endif
    RETURN
  END SUBROUTINE SetVDWA

  SUBROUTINE SetVDWG(G0)
  use dimens_fcm
  use psf
  use param
  use stream

    real(chm_real) G0

    INTEGER I

    if (prnlev > 2) WRITE(OUTU,*) 'Setting surface tension for hydrogen atoms to zero'

    DO I = 1, nmvsel
       IF (ATC(IAC(I))(1:1) .EQ. 'H') THEN
          GMVDW(i) = 0.0D0
       ELSE
          GMVDW(I) = G0
       END IF
    END DO

    ! SJT VDW
    ! MF/SJT
    RETURN
  END SUBROUTINE SetVDWG

  SUBROUTINE SetEpsProfile(UNEPS, EPS0, EPSW, EPSMIN, EPSMAX, N, X, A)
    !-----------------------------------------------------------------------
    ! Set up the dielectric constant profile along the z axis
    ! as a spline interpolating function.
    ! An input file format is rather restricted, that is,
    ! the dielectric constant must be sampled at equal interval HSPL.
    !
    !    UNEPS   unit number of an input file (Default : -1)
    !    CSPL    spline coefficients
    !    MAXC_HDGB    maximum size of array
    !    HSPL    grid size (A)
    !    ZMINSPL Z minimum value (A)
    !    ZMAXSPL Z maximum value (A)
    !    EPS0    eps for |Z| < ZMINSPL
    !    EPSW    eps for |Z| > ZMAXSPL
    !    EPSMIN  minimum value of eps
    !    EPSMAX  maximum value of eps
    !
    !-----------------------------------------------------------------------
    INTEGER UNEPS, N
    real(chm_real)  EPS0, EPSW
    real(chm_real)  X(300), A(300)

    INTEGER I, J
    real(chm_real)  ALPHA(300), DF0, DFN, EPSMIN, EPSMAX
    real(chm_real)  H(300), L(300), M(300), Z(300)
    real(chm_real)  B(300), C(300), D(300)
    real(chm_real)  COEF(4,50)

    DATA ((COEF(I,J), I = 1, 4), J = 1, 50) &
         /-0.000348659847D0, 0.000305904095D0, 0.000000000000D0,  &
         0.000000000000D0, 0.000316427536D0, -0.000217085676D0, &
         0.000044409210D0, 0.000032893543D0, -0.000195156005D0,  &
         0.000257555628D0, 0.000064644186D0, 0.000040380171D0, &
         0.000093740020D0, -0.000035178379D0, 0.000175832811D0,  &
         0.000112696670D0,  -0.000077312547D0, 0.000105431651D0, &
         0.000210959447D0, 0.000203535983D0, 0.000104523308D0,  &
         -0.000010537169D0, 0.000258406688D0, 0.000325709551D0, &
         -0.000039425231D0, 0.000146247793D0, 0.000326262000D0,  &
         0.000465344016D0, 0.000002774277D0, 0.000087109947D0, &
         0.000442940870D0, 0.000660108810D0, 0.000079543087D0,  &
         0.000091271363D0, 0.000532131525D0, 0.000903703517D0, &
         -0.000016354599D0, 0.000210585994D0, 0.000683060203D0,  &
         0.001202530006D0, 0.000113621226D0, 0.000186054095D0, &
         0.000881380248D0, 0.001594662281D0, -0.000069137164D0,  &
         0.000356485935D0, 0.001152650263D0, 0.002096068582D0, &
         0.000397462971D0, 0.000252780188D0, 0.001457283324D0,  &
         0.002752873051D0, 0.000031074448D0, 0.000848974645D0, &
         0.002008160740D0, 0.003594392632D0, 0.001205578353D0,  &
         0.000895586317D0, 0.002880441221D0, 0.004814600969D0, &
         -0.000201098780D0, 0.002703953846D0, 0.004680211302D0,  &
         0.006629415453D0, -0.000902016948D0, 0.002402305676D0, &
         0.007233341063D0, 0.009620372218D0, -0.000068587828D0,  &
         0.001049280254D0, 0.008959134028D0, 0.013724867050D0, &
         0.001276424204D0, 0.000946398512D0, 0.009956973411D0,  &
         0.018458180650D0, -0.001019450870D0, 0.002861034818D0, &
         0.011860690076D0, 0.023832820009D0, 0.001893100108D0,  &
         0.001331858512D0, 0.013957136741D0, 0.030350992392D0, &
         -0.001046312614D0, 0.004171508674D0, 0.016708820334D0,  &
         0.037899162904D0, 0.003466172002D0, 0.002602039753D0, &
         0.020095594547D0, 0.047165661163D0, -0.005488815012D0,  &
         0.007801297756D0, 0.025297263302D0, 0.058297239875D0, &
         0.004746017928D0, -0.000431924762D0, 0.028981949799D0,  &
         0.072210094089D0, 0.005800661582D0, 0.006687102130D0, &
         0.032109538483D0, 0.087186340039D0, -0.005883113557D0,  &
         0.015388094502D0, 0.043147136799D0, 0.105637967510D0, &
         0.009430077696D0, 0.006563424167D0, 0.054122896133D0,  &
         0.130323170340D0, 0.001252404164D0, 0.020708540710D0, &
         0.067758878572D0, 0.160204234161D0, 0.005859776572D0,  &
         0.022587146955D0, 0.089406722404D0, 0.199417359145D0, &
         0.019317939582D0, 0.031376811813D0, 0.116388701788D0,  &
         0.250499979157D0, -0.010957260850D0, 0.060353721186D0, &
         0.162253968288D0, 0.318953275452D0, 0.063126195543D0,  &
         0.043917829911D0, 0.214389743836D0, 0.413799032286D0, &
         -0.249468650409D0, 0.138607123226D0, 0.305652220405D0,  &
         0.539864136125D0, 0.111218229739D0, -0.235595852387D0, &
         0.257157855824D0, 0.696158445833D0, 0.050421414968D0,  &
         -0.068768507778D0, 0.104975675742D0, 0.779740689366D0, &
         -0.034552144511D0, 0.006863614674D0, 0.074023229190D0,  &
         0.821339077163D0, 0.048099616080D0, -0.044964602093D0, &
         0.054972735480D0, 0.855747577363D0, -0.043272177653D0,  &
         0.027184822028D0, 0.046082845448D0, 0.878005246590D0, &
         0.032895919355D0, -0.037723444452D0, 0.040813534235D0,  &
         0.902433852614D0, -0.020524554015D0, 0.011620434581D0, &
         0.027762029300D0, 0.917521748538D0, 0.028301128136D0,  &
         -0.019166396442D0, 0.023989048369D0, 0.931742302581D0, &
         -0.037751019206D0, 0.023285295762D0, 0.026048498029D0,  &
         0.942482868673D0, 0.035689583790D0, -0.033341233047D0, &
         0.021020529386D0, 0.956609564227D0, -0.027830825125D0,  &
         0.020193142638D0, 0.014446484181D0, 0.963245718632D0, &
         0.028457176070D0, -0.021553095050D0, 0.013766507975D0,  &
         0.972038393241D0, -0.024171311879D0, 0.021132669054D0, &
         0.013556294977D0, 0.977090520475D0, 0.008033126202D0,  &
         -0.015124298764D0, 0.016560480122D0, 0.986130421242D0, &
         0.008396981157D0, -0.003074609461D0, 0.007461026010D0,  &
         0.991633727387D0, -0.026940019589D0, 0.009520862275D0, &
         0.010684152417D0, 0.995645210672D0/

    DO I = 1, MAXC_HDGB
       DO J = 1, 4
          CSPL(J,I) = 0.D0
       END DO
    END DO

    IF (UNEPS .EQ. -1) THEN
       ! Defalut EPS profile
       ! Based on the three-dielectric model of DPPC
       HSPL    = 0.5D0
       ZMINSPL = 0.0D0
       ZMAXSPL = 25.0D0
       EPS0    = 2.258D0
       EPSMIN = 2.258D0
       EPSMAX = 80.D0
       EPSW    = 80.D0
       DO I = 1, 50
          CSPL(1,i) = COEF(1,i)
          CSPL(2,i) = COEF(2,i)
          CSPL(3,i) = COEF(3,i)
          CSPL(4,i) = COEF(4,i)
       END DO
    ELSE
       ! Construct the spline function

       ! Shift and normalize the data
       EPS0 = A(1)
       EPSMIN  = A(1)
       EPSMAX  = A(1)
       !AP/MF
       DO I=1, N
          IF (A(I) .lt. EPSMIN) THEN
             EPSMIN = A(I)
          END IF
          IF (A(I) .gt. EPSMAX) THEN
             EPSMAX = A(I)
          END IF
       END DO
       !END AP/MF

       EPSW    = A(N)
       ZMINSPL = X(1)
       ZMAXSPL = X(N)
       DO I = 1, N
          A(I) = A(I) - EPSMIN
       END DO
       DO I = 1, N
          !mf A(I) = A(I) / A(N)
          A(I) = A(I) / (EPSMAX-EPSMIN)
       END DO

       ! Clamped Cubic Spline Algorithm
       DF0  = 0.D0
       DFN  = 0.D0
       DO I = 1, N-1
          H(I) = X(I+1) - X(I)
       END DO
       ALPHA(1) = 3.0D0*(A(2)-A(1))/H(1)-3*DF0
       ALPHA(N) = 3.0D0*DFN-3.0D0*(A(N)-A(N-1))/H(N-1)
       DO I = 2, N-1
          ALPHA(I) = 3.D0*(A(I+1)*H(I-1)-A(I)*(X(I+1)-X(I-1)) &
               +A(I-1)*H(I))/(H(I)*H(I-1))
       ENDDO
       L(1)  = 2.0D0*H(1)
       M(1)  = 0.5D0
       Z(1)  = ALPHA(1)/L(1)
       DO I = 2, N-1
          L(I) = 2.0D0*(X(I+1)-X(I-1))-H(I-1)*M(I-1)
          M(I) = H(I)/L(I)
          Z(I) = (ALPHA(I)-H(I-1)*Z(I-1))/L(I)
       ENDDO
       L(N) = H(N-1)*(2-M(N-1))
       Z(N) = (ALPHA(N)-H(N-1)*Z(N-1))/L(N)
       C(N) = Z(N)
       DO J = N-1, 1, -1
          C(J) = Z(J)-M(J)*C(J+1)
          B(J) = (A(J+1)-A(J))/H(J)-H(J)*(C(J+1)+2.D0*C(J))/3.0D0
          D(J) = (C(J+1)-C(J))/(3.0D0*H(J))
       ENDDO
       DO I = 1, N-1
          CSPL(1,I) = D(I)
          CSPL(2,I) = C(I)
          CSPL(3,I) = B(I)
          CSPL(4,I) = A(I)
          ! WRITE(6,'(1x,I2,4(1X,f20.12))') I, (CSPL(J,I), J=4,1,-1)
       END DO

    END IF

    RETURN
  END SUBROUTINE SetEpsProfile

  SUBROUTINE  SetNPProfile(NP0,NPW,NPMIN,N, X, A)
    !-----------------------------------------------------------------------
    ! Set up the non-polar profile (to be multiplied by surface tension
    ! in water) along the z axis as a spline interpolating function.
    ! An input file format is rather restricted, that is,
    ! the dielectric constant must be sampled at equal interval HSPL.
    !
    !    CSPLNP    spline coefficients
    !    MAXC_HDGB maximum size of array
    !    ZMINSPLNP Z minimum value (A)
    !    ZMAXSPLNP Z maximum value (A)
    !
    !-----------------------------------------------------------------------
    INTEGER UNEPS, N
    real(chm_real)  X(300), A(300)

    real(chm_real)  NP0,NPW,NPMIN

    real(chm_real)  NPMAX

    INTEGER I, J
    real(chm_real)  ALPHA(300), DF0, DFN
    real(chm_real)  H(300), L(300), M(300), Z(300)
    real(chm_real)  B(300), C(300), D(300)

    DO I = 1, MAXC_HDGB
       DO J = 1, 4
          CSPLNP(J,I) = 0.D0
       END DO
    END DO

    ! Construct the spline function

    ! Shift and normalize the data
    NP0 = A(1)
    NPMIN=A(1)
    NPMAX=A(1)
    ! AP/MF
    DO I=1, N
       IF (A(I) .lt. NPMIN) THEN
          NPMIN = A(I)
       END IF
       IF (A(I) .gt. NPMAX) THEN
          NPMAX = A(I)
       END IF
    END DO
    ! END AP/MF

    NPW    = A(N)

    ZMINSPLNP = X(1)
    ZMAXSPLNP = X(N)

    IF (NPMAX-NPMIN .LT. 0.00001) THEN
      DO I=1,N
        CSPLNP(4,I)=NPMIN
      ENDDO
    ELSE
      DO I = 1, N
        A(I) = A(I) - NPMIN
      END DO

      DO I = 1, N
        A(I) = A(I) / A(N)
      END DO

      ! Clamped Cubic Spline Algorithm
      DF0  = 0.D0
      DFN  = 0.D0
      DO I = 1, N-1
         H(I) = X(I+1) - X(I)
      END DO
      ALPHA(1) = 3.0D0*(A(2)-A(1))/H(1)-3*DF0
      ALPHA(N) = 3.0D0*DFN-3.0D0*(A(N)-A(N-1))/H(N-1)
      DO I = 2, N-1
         ALPHA(I) = 3.D0*(A(I+1)*H(I-1)-A(I)*(X(I+1)-X(I-1)) &
              +A(I-1)*H(I))/(H(I)*H(I-1))
      ENDDO
      L(1)  = 2.0D0*H(1)
      M(1)  = 0.5D0
      Z(1)  = ALPHA(1)/L(1)
      DO I = 2, N-1
         L(I) = 2.0D0*(X(I+1)-X(I-1))-H(I-1)*M(I-1)
         M(I) = H(I)/L(I)
         Z(I) = (ALPHA(I)-H(I-1)*Z(I-1))/L(I)
      ENDDO
      L(N) = H(N-1)*(2-M(N-1))
      Z(N) = (ALPHA(N)-H(N-1)*Z(N-1))/L(N)
      C(N) = Z(N)
      DO J = N-1, 1, -1
         C(J) = Z(J)-M(J)*C(J+1)
         B(J) = (A(J+1)-A(J))/H(J)-H(J)*(C(J+1)+2.D0*C(J))/3.0D0
         D(J) = (C(J+1)-C(J))/(3.0D0*H(J))
      ENDDO
      DO I = 1, N-1
         CSPLNP(1,I) = D(I)
         CSPLNP(2,I) = C(I)
         CSPLNP(3,I) = B(I)
         CSPLNP(4,I) = A(I)
         ! WRITE(6,'(1x,I2,4(1X,f20.12))') I, (CSPLNP(J,I), J=4,1,-1)
      END DO
    ENDIF

    RETURN
  END SUBROUTINE SetNPProfile

!________________________________
#if KEY_HDGBVDW==1
!MS, MF

      SUBROUTINE  SetDenProfile(D0,DW,DMIN,DMAX,N, X, A ,ZMINSPLD,ZMAXSPLD)
!-----------------------------------------------------------------------
!     Set up the dielectric constant profile along the z axis
!     as a spline interpolating function.
!     An input file format is rather restricted, that is,
!     the dielectric constant must be sampled at equal interval HSPL.
!
!        CSPLD    spline coefficients
!        MAXC_HDGB    maximum size of array
!        HSPL    grid size (A)
!        ZMINSPLD Z minimum value (A)
!        ZMAXSPLD Z maximum value (A)
!        D0    eps for |Z| < ZMINSPLD
!        DW    eps for |Z| > ZMAXSPLD
!        DMIN  minimum value of density
!        DMAX  max value of density
      INTEGER  N
      REAL*8  X(300), A(300)

      REAL*8  D0,DW,DMIN,DMAX,ZMINSPLD,ZMAXSPLD
      INTEGER I, J
      REAL*8  ALPHA(300), DF0, DFN
      REAL*8  H(300), L(300), M(300), Z(300)
      REAL*8  B(300), C(300), D(300)

      DO I = 1, MAXC_HDGB
         DO J = 1, 4
           CSPLD(J,I) = 0.D0
         END DO
      END DO

!!     Construct the spline function

!!     Shift and normalize the data
      D0 = A(1)
      DMIN=A(1)
      DMAX=A(N)

      DO I=1, N
        IF (A(I) .lt. DMIN) THEN
           DMIN = A(I)
        END IF
        IF (A(I) .gt. DMAX) THEN
           DMAX = A(I)
        END IF
      END DO

      DW    = A(N)
      ZMINSPLD = X(1)
      ZMAXSPLD = X(N)

!      WRITE (6,*) 'Min Max',ZMINSPLD,ZMAXSPLD

      DO I = 1, N
         A(I) = A(I) - DMIN
      END DO

      DO I = 1, N
         A(I) = A(I) / DMAX
      END DO

!!     Clamped Cubic Spline Algorithm
      DF0  = 0.D0
      DFN  = 0.D0
      DO I = 1, N-1
         H(I) = X(I+1) - X(I)
      END DO
      ALPHA(1) = 3.0D0*(A(2)-A(1))/H(1)-3*DF0
      ALPHA(N) = 3.0D0*DFN-3.0D0*(A(N)-A(N-1))/H(N-1)
      DO I = 2, N-1
         ALPHA(I) = 3.D0*(A(I+1)*H(I-1)-A(I)*(X(I+1)-X(I-1)) &
             +A(I-1)*H(I))/(H(I)*H(I-1))

!      WRITE(6,*) 'test alfa'
      ENDDO
      L(1)  = 2.0D0*H(1)
      M(1)  = 0.5D0
      Z(1)  = ALPHA(1)/L(1)
      DO I = 2, N-1
         L(I) = 2.0D0*(X(I+1)-X(I-1))-H(I-1)*M(I-1)
         M(I) = H(I)/L(I)
         Z(I) = (ALPHA(I)-H(I-1)*Z(I-1))/L(I)
      ENDDO
      L(N) = H(N-1)*(2-M(N-1))
      Z(N) = (ALPHA(N)-H(N-1)*Z(N-1))/L(N)
      C(N) = Z(N)
      DO J = N-1, 1, -1
         C(J) = Z(J)-M(J)*C(J+1)
         B(J) = (A(J+1)-A(J))/H(J)-H(J)*(C(J+1)+2.D0*C(J))/3.0D0
         D(J) = (C(J+1)-C(J))/(3.0D0*H(J))

      ENDDO
      DO I = 1, N-1
         CSPLD(1,I) = D(I)
         CSPLD(2,I) = C(I)
         CSPLD(3,I) = B(I)
         CSPLD(4,I) = A(I)

      END DO

!      WRITE(6,*) 'DO, DW,min and Max',D0,DW,DMIN,DMAX

      RETURN
      END  SUBROUTINE SetDenProfile

!END MS/MF
#endif
#if KEY_DHDGB==1
!AP/MF
  SUBROUTINE SET_UP_COEFF(X_INP,Y_INP,N_INP,SPLINES,XMIN_INP, &
                          XMAX_INP, &
                          YMIN_INP,YMAX_INP)
!_____________________________________________________
!calculates the spline of the deformation coefficients
!_____________________________________________________
    INTEGER N_INP,I,J,NUM
    REAL(chm_real) X_INP(PRO_LENGTH),Y_INP(PRO_LENGTH)
    REAL(chm_real) XMIN_INP,XMAX_INP
    REAL(chm_real) YMIN_INP,YMAX_INP, SPLINES(4,PRO_LENGTH)
    REAL(chm_real) A(PRO_LENGTH),B(PRO_LENGTH),D(PRO_LENGTH)
    REAL(chm_real) H(PRO_LENGTH),ALPHA(PRO_LENGTH)
    REAL(chm_real) C(PRO_LENGTH),L(PRO_LENGTH),M(PRO_LENGTH)
    REAL(chm_real) Z(PRO_LENGTH)
    REAL(chm_real) DFN,DF0,SLOPE
    DFN=0.D0
    DF0=0.D0
    NUM=N_INP-1
    XMIN_INP=X_INP(1)
    XMAX_INP=X_INP(N_INP)
    YMIN_INP=Y_INP(1)
    YMAX_INP=Y_INP(N_INP)
    DO I=1,N_INP
       DO J=1,4
          SPLINES(J,I)=0.D0
       ENDDO
    ENDDO
    DO I=1,N_INP
       A(I)=Y_INP(I)
    ENDDO
    DO I=1,N_INP-1
       H(I)=X_INP(I+1)-X_INP(I)
    ENDDO
    ALPHA(1)=3.0D0*(A(2)-A(1))/H(1)-3.0D0*DF0
    ALPHA(N_INP) = 3.0D0*DFN-3.0D0*(A(N_INP)-A(N_INP-1))/H(N_INP-1)
    DO I=2,N_INP-1
       ALPHA(I)=3.D0*(A(I+1)*H(I-1)-A(I)*(X_INP(I+1)-X_INP(I-1)) &
             +A(I-1)*H(I))/(H(I)*H(I-1))
    ENDDO
    L(1)=2.0D0*H(1)
    M(1)=0.5D0
    Z(1)=ALPHA(1)/L(1)
    DO I=2,N_INP-1
       L(I)=2.0D0*(X_INP(I+1)-X_INP(I-1))-H(I-1)*M(I-1)
       M(I)=H(I)/L(I)
       Z(I)=(ALPHA(I)-H(I-1)*Z(I-1))/L(I)
    ENDDO
    L(N_INP)=H(N_INP-1)*(2.0D0-M(N_INP-1))
    Z(N_INP)=(ALPHA(N_INP)-H(N_INP-1)*Z(N_INP-1))/L(N_INP)
    C(N_INP)=Z(N_INP)
    DO J=N_INP-1,1,-1
       C(J)=Z(J)-M(J)*C(J+1)
       B(J)=(A(J+1)-A(J))/H(J)-H(J)*(C(J+1)+2.0D0*C(J))/3.0D0
       D(J)=(C(J+1)-C(J))/(3.0D0*H(J))
    ENDDO
    DO I=1,N_INP-1
       SPLINES(1,I)=D(I)
       SPLINES(2,I)=C(I)
       SPLINES(3,I)=B(I)
       SPLINES(4,I)=A(I)
    ENDDO
  END SUBROUTINE SET_UP_COEFF
  SUBROUTINE CAL_COEFF(SPLINES,XMIN_INP,XMAX_INP, &
             AMIN_INP,AMAX_INP,Z_I, &
             HSPL_COEFF,COEFF_I,MCOEFF_I, &
             dCOEFF_Idz,MdCOEFF_Idz,SLOPE)
!_____________________________________________________________________
!CALCULATES A COEFFICIENT AND DCOEFFICIENT/DZ FROM ITS SPLINE FUNCTION
!_____________________________________________________________________
  REAL(chm_real) SPLINES(4,PRO_LENGTH)
  REAL(chm_real) XMIN_INP,XMAX_INP,Z_I,HSPL_COEFF,COEFF_I,MCOEFF_I
  REAL(chm_real) AMIN_INP,AMAX_INP,DIFF,DELL,SLOPE
  REAL(chm_real) dCOEFF_Idz,MdCOEFF_Idz,MZ_I
  INTEGER I
  COEFF_I=0.0d0
  dCOEFF_IdZ=0.0d0
  IF (Z_I .lt. XMIN_INP) THEN
      COEFF_I=AMIN_INP
      dCOEFF_IdZ=0.D0
  ELSEIF (Z_I .lt. XMAX_INP) THEN
      DIFF = Z_I - XMIN_INP
      I = DINT(DIFF/HSPL_COEFF)+1
      DELL = Z_I - XMIN_INP - (I-1)*HSPL_COEFF
      COEFF_I = SPLINES(1,I)*DELL**3+SPLINES(2,I)*DELL**2+ &
              SPLINES(3,I)*DELL+SPLINES(4,I)
      dCOEFF_IdZ = 3*SPLINES(1,I)*DELL**2+2*SPLINES(2,I)*DELL+ &
              SPLINES(3,I)
  ELSE
      DIFF = Z_I - XMAX_INP
      COEFF_I=AMAX_INP+SLOPE*DIFF
      dCOEFF_IdZ=SLOPE
  ENDIF
  MZ_I=-Z_I
  IF (MZ_I .lt. XMIN_INP) THEN
      MCOEFF_I=AMIN_INP
      MdCOEFF_IdZ=0.D0
  ELSEIF (MZ_I .lt. XMAX_INP) THEN
      DIFF = MZ_I - XMIN_INP
      I = DINT(DIFF/HSPL_COEFF)+1
      DELL = MZ_I - XMIN_INP - (I-1)*HSPL_COEFF
      MCOEFF_I = SPLINES(1,I)*DELL**3+SPLINES(2,I)*DELL**2+ &
             SPLINES(3,I)*DELL+SPLINES(4,I)
      MdCOEFF_IdZ = -(3*SPLINES(1,I)*DELL**2+2*SPLINES(2,I)*DELL+ &
             SPLINES(3,I))
  ELSE
     DIFF = MZ_I - XMAX_INP
     MCOEFF_I=AMAX_INP+SLOPE*DIFF
     MdCOEFF_IdZ=-SLOPE
  ENDIF
  END SUBROUTINE CAL_COEFF
  SUBROUTINE CALPRO_DEF(P,Z_I,A_COEF,C_COEFF,E_COEFF,B_COEFF,OFFS, &
             MA_COEF,MC_COEFF,ME_COEFF, &
             DA_COEFDZ,DC_COEFFDZ,DE_COEFFDZ, &
             MDA_COEFDZ,MDC_COEFFDZ,MDE_COEFFDZ, &
             S_FHDGB,S_FHDGBLO,NODEF,DNODEF_DZ, &
             MNODEF,MDNODEF_DZ, &
             PROU,DPROU_DZ,DPROU_DS,DPROU_DSL)
    REAL(chm_real) Z_I,A_COEF,C_COEFF,E_COEFF,B_COEFF,OFFS
    REAL(chm_real) MA_COEF,MC_COEFF,ME_COEFF
    REAL(chm_real) DA_COEFDZ,DC_COEFFDZ,DE_COEFFDZ
    REAL(chm_real) MDA_COEFDZ,MDC_COEFFDZ,MDE_COEFFDZ
    REAL(chm_real) S_FHDGB,S_FHDGBLO,NODEF,DNODEF_DZ
    REAL(chm_real) MNODEF,MDNODEF_DZ
    REAL(chm_real) PROU,DPROU_DZ,DPROU_DS,DPROU_DSL
    REAL(chm_real) PRO,MPRO,DPRO_DZ,DPRO_DS,MDPRO_DZ,MDPRO_DS
    REAL(chm_real) DEFORM,DENUM,DENUMWI
    REAL(chm_real) P,WI,DWI_DZ,DWI_DS,DWI_DSL
    REAL(chm_real) DEL1,DEL2,SUMDEL,DELWEIGHT
    REAL(chm_real) DDEL1_DZ,MDDEL1_DZ,DEL1DENUM
!HOW TO CALCULATE DEFORMED PROFILE WITH ONLY 1.5 EXPANSION ALLOWED
!EPS(Z,U)=EPS(Z,0)+SUMDEL
!DELWEIGHT=1/(1+EXP(-2*(DEFORM-MAXEXP)))
!DEL1=A/(1+EXP(-B*(MAXEXP-C)))-E
!DEL2=A/(1+EXP(-B*(DEFORM-C)))-E-DEL1
!SUMDEL=DELWEIGHT*DEL2+DEL1
!______CALCULATING EPS AND THE DERIVATIVES_______________
    DEFORM=ZMAXSPL-S_FHDGB
    DENUM=1.0D0+DEXP(-B_COEFF*(DEFORM-C_COEFF+OFFS))
    DEL1DENUM=1.0D0+DEXP(-B_COEFF*(MAXEXP-C_COEFF+OFFS))
    DEL1=A_COEF/DEL1DENUM-E_COEFF
    DEL2=A_COEF/DENUM-E_COEFF-DEL1
    DELWEIGHT=1.0D0/(1.0D0+DEXP(-2.0D0*(DEFORM-MAXEXP)))
    PRO=NODEF+DELWEIGHT*DEL2+DEL1
    DDEL1_DZ=(((DA_COEFDZ*DEL1DENUM)- &
             (B_COEFF*DC_COEFFDZ*(DEL1DENUM-1))*A_COEF)/ &
              DEL1DENUM**2-DE_COEFFDZ)
    DPRO_DZ=(((DA_COEFDZ*DENUM)- &
            (B_COEFF*DC_COEFFDZ*(DENUM-1))*A_COEF)/ &
             DENUM**2-DE_COEFFDZ)*DELWEIGHT+ &
             DDEL1_DZ*(1.0D0-DELWEIGHT) &
             +DNODEF_DZ
    DPRO_DS=(-A_COEF*B_COEFF*(DENUM-1)/DENUM**2)*DELWEIGHT &
             -DEL2*2.0D0*DEXP(-2.0D0*(DEFORM-MAXEXP))/ &
             (1.0D0+DEXP(-2.0D0*(DEFORM-MAXEXP)))**2
!_____CALCULATING EPSL AND THE DERIVATIVES_______________
    DEFORM=ZMAXSPL-S_FHDGBLO
    DENUM=1.0D0+DEXP(-B_COEFF*(DEFORM-MC_COEFF+OFFS))
    DEL1DENUM=1.0D0+DEXP(-B_COEFF*(MAXEXP-MC_COEFF+OFFS))
    DEL1=MA_COEF/DEL1DENUM-ME_COEFF
    DEL2=MA_COEF/DENUM-ME_COEFF-DEL1
    DELWEIGHT=1.0D0/(1.0D0+DEXP(-2.0D0*(DEFORM-MAXEXP)))
    MPRO=MNODEF+DELWEIGHT*DEL2+DEL1
    MDDEL1_DZ=(((MDA_COEFDZ*DEL1DENUM)- &
              (B_COEFF*MDC_COEFFDZ*(DEL1DENUM-1))*MA_COEF) &
              /DEL1DENUM**2-MDE_COEFFDZ)
    MDPRO_DZ=(((MDA_COEFDZ*DENUM)- &
              (B_COEFF*MDC_COEFFDZ*(DENUM-1))*MA_COEF) &
              /DENUM**2-MDE_COEFFDZ)*DELWEIGHT+ &
              MDDEL1_DZ*(1.0D0-DELWEIGHT) &
              -MDNODEF_DZ
    MDPRO_DS=(-MA_COEF*B_COEFF*(DENUM-1)/DENUM**2)*DELWEIGHT &
             -DEL2*2.0D0*DEXP(-2.0D0*(DEFORM-MAXEXP))/ &
              (1.0D0+DEXP(-2.0D0*(DEFORM-MAXEXP)))**2
!______CALCULATING THE WEIGHT_______________________________

    DENUMWI=1+DEXP(-P*(Z_I-(S_FHDGB-S_FHDGBLO)/2.0D0))
    WI=1.0D0/DENUMWI
    PROU=WI*PRO+(1-WI)*MPRO
    DWI_DZ=P*(DENUMWI-1.0D0)/DENUMWI**2
    DWI_DS=-P*(0.50D0)*(DENUMWI-1.0D0)/DENUMWI**2
    DWI_DSL=P*(0.50D0)*(DENUMWI-1.0D0)/DENUMWI**2
    DPROU_DZ=DWI_DZ*(PRO-MPRO)+DPRO_DZ*WI+(1.0D0-WI)*MDPRO_DZ
    DPROU_DS=DWI_DS*(PRO-MPRO)+DPRO_DS*WI
    DPROU_DSL=DWI_DSL*(PRO-MPRO)+(1.0D0-WI)*MDPRO_DS
  END SUBROUTINE CALPRO_DEF
#endif

  SUBROUTINE CalEpsHDGB(EPS_HDGB_I, DEPS_HDGB_I, Z_I, EPS0, EPSW,  &
       EPSMIN, EPSMAX,natomgb)
    !-----------------------------------------------------------------------
    ! Assign the effective dielectric constant and its derivative
    ! via spline function
    !-----------------------------------------------------------------------
    real(chm_real) EPS_HDGB_I, DEPS_HDGB_I, Z_I, EPS0, EPSW,EPSMIN, EPSMAX
    INTEGER i, j, natomgb
    real(chm_real)  ZDEL, Z_I_ABS, X1

    Z_I_ABS = DABS(Z_I)

    IF (Z_I_ABS .lt. ZMINSPL) THEN
       EPS_HDGB_I  = EPS0
       DEPS_HDGB_I = 0.D0
    ELSE IF (Z_I_ABS .lt. ZMAXSPL) THEN
       !mf X1   = EPSW - EPSMIN
       X1 = EPSMAX - EPSMIN
       i    = DINT((Z_I_ABS -ZMINSPL) / HSPL ) + 1
       ZDEL = DABS(Z_I - ZMINSPL) - HSPL * (i - 1)

       EPS_HDGB_I  = CSPL(1,i) * ZDEL**3 + CSPL(2,i) * ZDEL**2 &
            + CSPL(3,i) * ZDEL    + CSPL(4,i)
       DEPS_HDGB_I = 3.D0 * CSPL(1,i) * ZDEL**2  &
            + 2.D0 * CSPL(2,i) * ZDEL &
            + CSPL(3,i)

       EPS_HDGB_I  = X1*EPS_HDGB_I + EPSMIN
       DEPS_HDGB_I = X1*DEPS_HDGB_I
    ELSE
       EPS_HDGB_I  = EPSW
       DEPS_HDGB_I = 0.D0
    ENDIF
!BD/MF
    if (qexclgb) then
      IF (ANY( adlistgb==natomgb ) ) then
        EPS_HDGB_I  = EPSW
        DEPS_HDGB_I = 0.D0
      ENDIF
    endif
    IF (Z_I .lt. 0) THEN
       DEPS_HDGB_I = - DEPS_HDGB_I
    ENDIF

    RETURN
  END SUBROUTINE CalEpsHDGB

  SUBROUTINE CalSurfTspl_HDGB(NP_HDGB_I, DNP_HDGB_I, Z_I, &
       NP0, NPW,NPMIN, SA,natomgb)
    !-----------------------------------------------------------------------
    ! Calculate the non-polar profile and its derivative
    ! via spline function
    !-----------------------------------------------------------------------
    real(chm_real) NP_HDGB_I, DNP_HDGB_I, Z_I, NP0, NPW, SA,NPMIN
    INTEGER i, j, natomgb
    real(chm_real)  ZDEL, Z_I_ABS, X1

    Z_I_ABS = DABS(Z_I)

    IF (Z_I_ABS .lt. ZMINSPLNP) THEN
       NP_HDGB_I  = NP0*SA
       DNP_HDGB_I = 0.D0
    ELSE IF (Z_I_ABS .lt. ZMAXSPLNP) THEN
       X1   = NPW - NPMIN
       i    = DINT( (Z_I_ABS-ZMINSPLNP) / HSPLNP ) + 1
       ZDEL = DABS(Z_I - ZMINSPLNP) - HSPLNP * (i - 1)
       NP_HDGB_I  = CSPLNP(1,i) * ZDEL**3 + CSPLNP(2,i) * ZDEL**2 &
            + CSPLNP(3,i) * ZDEL    + CSPLNP(4,i)

       DNP_HDGB_I = 3.D0 * CSPLNP(1,i) * ZDEL**2  &
            + 2.D0 * CSPLNP(2,i) * ZDEL &
            + CSPLNP(3,i)
       NP_HDGB_I  = (X1*NP_HDGB_I + NPMIN)*SA
       DNP_HDGB_I = (X1*DNP_HDGB_I)*SA
    ELSE
       NP_HDGB_I  = NPW*SA
       DNP_HDGB_I = 0.D0
    ENDIF
!BD/MF
    if (qexclgb) then
      IF (ANY( adlistgb==natomgb ) ) then
         NP_HDGB_I  = NPW*SA
         DNP_HDGB_I = 0.D0
      ENDIF
    endif
    IF (Z_I .lt. 0) THEN
       DNP_HDGB_I = - DNP_HDGB_I
    ENDIF

    RETURN
  END SUBROUTINE CalSurfTspl_HDGB


  SUBROUTINE CalSurfT_HDGB(SurfT_HDGB_I, DSurfT_HDGB_I, Z_I, WST, QHDNOSW)
    !-----------------------------------------------------------------------
    ! Assign the surface tension and its derivative
    !-----------------------------------------------------------------------
  use number
    real(chm_real) SurfT_HDGB_I, DSurfT_HDGB_I, Z_I, WST
    LOGICAL QHDNOSW

    INTEGER i, j
    real(chm_real) ZDEL, Z_I_ABS, Z_I2
    real(chm_real) X1, X2, X3, X4
    real(chm_real) ST1

    IF (.NOT. QHDNOSW) THEN
      ! Compute the surface tension via analytic function
      Z_I_ABS = DABS(Z_I)
      ST1 = 1.D0 - ST0_HDGB
      IF (Z_I_ABS .lt. ZS_HDGB) THEN
         SurfT_HDGB_I  = 0.D0
         DSurfT_HDGB_I = 0.D0
      ELSE IF (Z_I_ABS .lt. ZM_HDGB) THEN
         X1 = (ZM_HDGB - ZS_HDGB)**3
         X2 = Z_I_ABS - ZS_HDGB
         SurfT_HDGB_I  = ST0_HDGB*(X2**2)*(3.D0*ZM_HDGB  &
         - 2.D0*Z_I_ABS - ZS_HDGB)/X1
         DSurfT_HDGB_I = ST0_HDGB*6.D0*X2*(ZM_HDGB - Z_I_ABS)/X1
         SurfT_HDGB_I  = WST*SurfT_HDGB_I
         DSurfT_HDGB_I = WST*DSurfT_HDGB_I
      ELSE IF (Z_I_ABS .lt. ZT_HDGB) THEN
         X1 = ZM_HDGB*ZM_HDGB
         X2 = ZT_hdgb*ZT_hdgb
         X3 = (X2-X1)**3
         X4 = Z_I_ABS**2
         SurfT_HDGB_I  = ST1*((X4-X1)**2) &
         *(3.D0*X2-2.D0*X4-X1)/X3 + ST0_HDGB
         DSurfT_HDGB_I = ST1*12.D0*Z_I_ABS*(X4-X1)*(X2-X4)/X3
         SurfT_HDGB_I  = WST*SurfT_HDGB_I
         DSurfT_HDGB_I = WST*DSurfT_HDGB_I
      ELSE
         SurfT_HDGB_I = WST
         DSurfT_HDGB_I = 0.d0
      ENDIF
      
      IF (Z_I .lt. 0) THEN
         DSurfT_HDGB_I = - DSurfT_HDGB_I
      ENDIF
    ELSE
       SurfT_HDGB_I = WST
       DSurfT_HDGB_I = 0.d0
    ENDIF

    RETURN
  END SUBROUTINE CalSurfT_HDGB

!_____________________________________________
#if KEY_HDGBVDW==1
!M/S,M/F
  SUBROUTINE CalDenspl_HDGB(D_HDGB_I, DD_HDGB_I, Z_I, &
          D0, DW,DMIN,DMAX,UND,ZMINSPLD,ZMAXSPLD,natomgb)

!-----------------------------------------------------------------------
!     Calculate the density profile and its derivative
!     via spline function for the vdw atraction in membrane
!-----------------------------------------------------------------------

      REAL*8 D_HDGB_I, DD_HDGB_I, Z_I
      REAL*8  D0, DW,DMIN, DMAX,ZMINSPLD,ZMAXSPLD
      INTEGER i, j ,k, UND, natomgb
      REAL*8  ZDEL, Z_I_ABS, X1 ,t

      Z_I_ABS = DABS(Z_I)
! First
      IF (Z_I_ABS .lt. ZMINSPLD) THEN
         D_HDGB_I  = D0
         DD_HDGB_I = 0.D0
      ELSE IF (Z_I_ABS .lt. ZMAXSPLD) THEN
         X1   = DMAX - DMIN
         i    = DINT((Z_I_ABS -ZMINSPLD) / HSPLD ) + 1
         ZDEL = DABS(Z_I_ABS - ZMINSPLD) - HSPLD * (i - 1)

         IF ( UND .eq. 81) THEN
            D_HDGB_I  = CSPLDO(1,i) * ZDEL**3 + CSPLDO(2,i) * ZDEL**2 &
                    + CSPLDO(3,i) * ZDEL    + CSPLDO(4,i)
            DD_HDGB_I = 3.D0 * CSPLDO(1,i) * ZDEL**2 &
                    + 2.D0 * CSPLDO(2,i) * ZDEL &
                    + CSPLDO(3,i)
         ELSEIF (UND .eq. 82) THEN
            D_HDGB_I  = CSPLDP(1,i) * ZDEL**3 + CSPLDP(2,i) * ZDEL**2 &
                    + CSPLDP(3,i) * ZDEL    + CSPLDP(4,i)
            DD_HDGB_I = 3.D0 * CSPLDP(1,i) * ZDEL**2 &
                    + 2.D0 * CSPLDP(2,i) * ZDEL &
                    + CSPLDP(3,i)

        ELSEIF (UND .eq. 83) THEN
            D_HDGB_I  = CSPLDC(1,i) * ZDEL**3 + CSPLDC(2,i) * ZDEL**2 &
                    + CSPLDC(3,i) * ZDEL    + CSPLDC(4,i)
            DD_HDGB_I = 3.D0 * CSPLDC(1,i) * ZDEL**2 &
                    + 2.D0 * CSPLDC(2,i) * ZDEL &
                    + CSPLDC(3,i)

         ELSEIF (UND .eq. 84) THEN
            D_HDGB_I  = CSPLDC2(1,i) * ZDEL**3 + CSPLDC2(2,i) * ZDEL**2 &
                    + CSPLDC2(3,i) * ZDEL    + CSPLDC2(4,i)
            DD_HDGB_I = 3.D0 * CSPLDC2(1,i) * ZDEL**2 &
                    + 2.D0 * CSPLDC2(2,i) * ZDEL &
                    + CSPLDC2(3,i)

         ELSEIF (UND .eq. 85) THEN
            D_HDGB_I  = CSPLDN(1,i) * ZDEL**3 + CSPLDN(2,i) * ZDEL**2 &
                    + CSPLDN(3,i) * ZDEL    + CSPLDN(4,i)
            DD_HDGB_I = 3.D0 * CSPLDN(1,i) * ZDEL**2 &
                    + 2.D0 * CSPLDN(2,i) * ZDEL &
                    + CSPLDN(3,i)

         ENDIF

         D_HDGB_I  = X1*D_HDGB_I + DMIN
         DD_HDGB_I = X1*DD_HDGB_I
      ELSE
         D_HDGB_I  = DW
         DD_HDGB_I = 0.D0
      ENDIF
!BD/MF
      if (qexclgb) then
        IF (ANY( adlistgb==natomgb ) ) then
           D_HDGB_I  = DW
           DD_HDGB_I = 0.D0
        ENDIF
      endif
      IF (Z_I .lt. 0) THEN
         DD_HDGB_I = - DD_HDGB_I
      ENDIF

      RETURN

!    if(allocated(dummy) deallocate(dummy)

  END  SUBROUTINE CalDenspl_HDGB


! END M/S , M/F

#endif

  SUBROUTINE SetRadiusHDGB()
  use dimens_fcm
  use coord
  use param
  use psf
  use stream

    INTEGER I

    IF (.NOT. QWEIGHT) THEN
       DO I = 1, nmvsel
          R_HDGB(I) = VDWR(ITC(IAC(I)))
          IF (QHDGBRC) THEN
             CALL CorrectRadius(WMAIN(I), DR_HDGB(I), R_HDGB(I), Z(I))
          ELSE
             WMAIN(I) = R_HDGB(I)
          END IF
       END DO
    ELSE
       IF (PRNLEV .GE. 5)  &
            WRITE(OUTU,*) ' Using Radii from WMAIN array'
       DO I = 1, nmvsel
          R_HDGB(I) = WMAIN(I)
          IF (QHDGBRC) THEN
             CALL CorrectRadius(WMAIN(I), DR_HDGB(I), R_HDGB(I), Z(I))
          END IF
       END DO
    ENDIF

    RETURN

  END SUBROUTINE SetRadiusHDGB


  SUBROUTINE CorrectRadius(R_I, DR_I, R0_I, Z_I)
    !-----------------------------------------------------------------------
    ! Calculating the effective radius upon insertion to an implicit
    ! membrane.
    ! Not compatible with FAST option.
    !-----------------------------------------------------------------------
  use number
    real(chm_real) R_I, DR_I, R0_I, Z_I

    INTEGER i, j
    real(chm_real) ZDEL, Z_I_ABS
    real(chm_real) X1, X2
    real(chm_real) COEF(4,3)
    SAVE COEF

    DATA ((COEF(i,j), i = 1, 4), j = 1, 3) &
         /0.001461453281D0, -0.010247693175D0, &
         0.000000000000D0, 1.211080000000D0, &
         -0.000262781601D0, 0.002905386350D0, &
         -0.022026920475D0, 1.158310000000D0, &
         0.000009455254D0, 0.000146179543D0, &
         -0.011346439848D0, 1.105540000000D0/

    ! Compute the correction via spline function
    Z_I_ABS = DABS(Z_I)
    IF (Z_I_ABS .lt. 3.D0) THEN
       X1 = 1.21108D0
       X2 = 0.D0
    ELSE IF (Z_I_ABS .lt. 6.D0) THEN
       i = 1
       ZDEL = Z_I_ABS - 3.D0
       X1 = COEF(1,i) * ZDEL**3 + COEF(2,i) * ZDEL**2 &
            + COEF(3,i) * ZDEL    + COEF(4,i)
       X2 = 3.D0 * COEF(1,i) * ZDEL**2  &
            + 2.D0 * COEF(2,i) * ZDEL &
            + COEF(3,i)
    ELSE IF (Z_I_ABS .lt. 9.5D0) THEN
       i = 2
       ZDEL = Z_I_ABS - 6.D0
       X1 = COEF(1,i) * ZDEL**3 + COEF(2,i) * ZDEL**2 &
            + COEF(3,i) * ZDEL    + COEF(4,i)
       X2 = 3.D0 * COEF(1,i) * ZDEL**2  &
            + 2.D0 * COEF(2,i) * ZDEL &
            + COEF(3,i)
    ELSE IF (Z_I_ABS .lt. 25.D0) THEN
       i = 3
       ZDEL = Z_I_ABS - 9.5D0
       X1 = COEF(1,i) * ZDEL**3 + COEF(2,i) * ZDEL**2 &
            + COEF(3,i) * ZDEL    + COEF(4,i)
       X2 = 3.D0 * COEF(1,i) * ZDEL**2  &
            + 2.D0 * COEF(2,i) * ZDEL &
            + COEF(3,i)
    ELSE
       X1 = 1.D0
       X2 = 0.D0
    ENDIF

    R_I  = R0_I * X1
    DR_I = R0_I * X2

    IF (Z_I .lt. 0) THEN
       DR_I = - DR_I
    ENDIF

    RETURN
  END SUBROUTINE CorrectRadius


#if KEY_HDGBVDW==1
! MS/MF


  SUBROUTINE SetVdw (UND_HDGB, &
               QDOUT, D_HDGB, DD_HDGB, &
               D0_HDGB,DW_HDGB,DMIN_HDGB,DMAX_HDGB, &
               ZMINSPLD,ZMAXSPLD )
!-----------------------------------------------------------------------
!     Calculating the VDW dispersion term for different atom types in the
!     membrane.
!
!-----------------------------------------------------------------------
 use exfunc
 use stream
! use dimens
 use consta
 use psf
! use heap
 use number
 use coord
 use param
 use string
 use surface
#if KEY_PARALLEL==1
 use parallel
#endif

      INTEGER UND_HDGB
      LOGICAL QDOUT
      Real*8 D0_HDGB,DW_HDGB,DMIN_HDGB,DMAX_HDGB
      Real*8 ZMINSPLD,ZMAXSPLD
      Real*8  D_HDGB(300),  DD_HDGB(300)

      INTEGER  IX1_HDGB
      Real*8  XD_HDGB(300), AD_HDGB(300)
      Real*8   DI ,ZI, DDI
      integer maxlen_hdgb, length_hdgb, ND_HDGB
      parameter(maxlen_hdgb=256)
      character name_hdgb*(maxlen_hdgb)
      logical qopen_hdgb, qform_hdgb, qwrite_hdgb, error_hdgb

      integer I

#if KEY_PARALLEL==1
              IF (MYNOD .EQ. 0) THEN
#endif
!                 Check at least if the file is opened.

!                  WRITE(OUTU,*)'name check1',NAME_HDGB
!                  WRITE(OUTU,*)'open check',MAXLEN_HDGB, &
!                       LENGTH_HDGB,QOPEN_HDGB,QFORM_HDGB,QWRITE_HDGB, &
!                         UND_HDGB
!                  WRITE(OUTU,*)'name check2',NAME_HDGB
                  CALL VINQRE('UNIT',NAME_HDGB,MAXLEN_HDGB,LENGTH_HDGB, &
                             QOPEN_HDGB,QFORM_HDGB,QWRITE_HDGB, &
                             UND_HDGB)

!                   WRITE(OUTU,*)'(TEST 2 FOR unit)',MAXLEN_HDGB, &
!                       LENGTH_HDGB,QOPEN_HDGB,QFORM_HDGB,QWRITE_HDGB, &
!                         UND_HDGB
!                   WRITE(OUTU,*)'name check',NAME_HDGB
                   IF (.NOT. QOPEN_HDGB) THEN
                     CALL WRNDIE(0,'<SetDenProfile>','UNIT NOT OPEN')
                   END IF

                   READ(UND_HDGB,*) ND_HDGB, HSPLD
                   DO I = 1, ND_HDGB
                      READ(UND_HDGB,*) XD_HDGB(I), AD_HDGB(I)
                   END DO
                   CALL VCLOSE(UND_HDGB,'KEEP',ERROR_HDGB)

#if KEY_PARALLEL==1
              END IF

              CALL PSND4(ND_HDGB, 1)
              CALL PSND8(HSPLD, 1)
              CALL PSND8(XD_HDGB, 300)
              CALL PSND8(AD_HDGB, 300)
#endif
              CALL SetDenProfile( &
                         D0_HDGB, DW_HDGB,DMIN_HDGB,DMAX_HDGB, &
                         ND_HDGB, XD_HDGB, AD_HDGB, &
                         ZMINSPLD,ZMAXSPLD )
!              WRITE(OUTU,*)  'DEBUG -> setting up Den profile'
!              WRITE(OUTU,*) UND_HDGB,D0_HDGB,DW_HDGB,DMIN_HDGB
              DO i = 1, nmvsel
                CALL CalDenspl_HDGB(D_HDGB(i), DD_HDGB(i), Z(i), &
                            D0_HDGB, DW_HDGB,DMIN_HDGB,DMAX_HDGB, &
                            UND_HDGB,ZMINSPLD,ZMAXSPLD,i)
!                WRITE (*,*)' Den check' , D_HDGB(i), DD_HDGB(i)
              ENDDO

#if KEY_PARALLEL==1
              IF (MYNOD .EQ. 0) THEN
#endif

                IF (PRNLEV .GE. 5) THEN
                  !WRITE(OUTU,*) 'Density Profile Setup'
                  !WRITE(OUTU,*) HSPLD, &
                  !               ZMINSPLD, ZMAXSPLD, &
                  !               DMIN_HDGB, DW_HDGB
                END IF
!This part writes the profile in the output!
                IF (QDOUT) THEN
                  WRITE(OUTU,*) 'HDGB: Density PROFILE:'
                  IX1_HDGB = Int((ZMAXSPLD +5.D0)/HSPLD)
                  DO I = -IX1_HDGB, IX1_HDGB
                     WRITE(*,*) 'HSPLD:' , HSPLD,I
                     ZI = HSPLD * I
                     CALL CalDenspl_HDGB(DI, DDI, ZI, &
                                 D0_HDGB, DW_HDGB,DMIN_HDGB,DMAX_HDGB, &
                                 UND_HDGB,ZMINSPLD,ZMAXSPLD,i)
                     WRITE(OUTU,'(3(f25.16,1x))') ZI, DI, DDI
                  END DO
                END IF


#if KEY_PARALLEL==1
              END IF
#endif


     RETURN
  END   SUBROUTINE SetVdw
!-------------------

  SUBROUTINE CalVdw(EPSVDW, SIGVDW, EPS_I, SIG_I, GVDW , &
                           Alph_GB , AVDW ,TTVDW,D_HDGB, &
                           UND,Z,DD_HDGB,PPVDW)
!-----------------------------------------------------------------------
!     Calculating the VDW dispersion term for different atom types in the
!     membrane.
!
!----------------------------------------------------------------------

 use number
 use consta
 use stream

      REAL*8 EPSVDW, SIGVDW, EPS_I, SIG_I, GVDW ,Alph_GB
      REAL*8 AVDW,TTVDW,X1VDW,D_HDGB,Z,DD_HDGB,PPVDW
      INTEGER UND

      REAL*8 SIGIW,EPSIW,X2VDW,X3VDW,X4VDW


!compute
!      WRITE (6,*) 'D check', D_HDGB,EPS_I,EPSVDW, SIG_I,SIGVDW
      X1VDW   = EIGHT*PI*(D_HDGB)/THREE
      EPSIW = SQRT (EPSVDW * EPS_I)
      SIGIW = SIGVDW + SIG_I
      SIGIW = SIGIW*SIGIW
      SIGIW = SIGIW*SIGIW*SIGIW

!81 water oxygens
!82 lipid oxygens
!83 lipic c
!84 lipid c2
!85 lipid H2

      if ( UND .EQ. 81) then
         X2VDW = ONE/(Alph_GB+1.40D0)
      else if (UND .EQ. 82) then
         X2VDW = ONE/(Alph_GB+1.5D0)
      else if (UND .EQ. 83) then
         X2VDW = ONE/(Alph_GB+2.2D0)
      else if (UND .EQ. 84) then
         X2VDW = ONE/(Alph_GB+2.20D0)
      else if (UND .EQ. 85) then
         X2VDW = ONE/(Alph_GB+1.50D0)
      endif

      X3VDW = X2VDW*X2VDW*X2VDW
      X4VDW = X1VDW*AVDW*EPSIW*SIGIW
      GVDW  = X4VDW*X3VDW
! dF(density, Ci)/d(Ci)
      TTVDW  = THREE*X4VDW*X3VDW*X2VDW
!dF(density,Ci)/d(density) * (density)'
      DD_HDGB=DD_HDGB
      PPVDW = X4VDW*X3VDW*(DD_HDGB)/(D_HDGB)
      !WRITE (*,*) 'check',Alph_GB,TTVDW,X2VDW,X3VDW,X4VDW 
     RETURN
  END   SUBROUTINE CalVdw

! MS/MF
#endif
#if KEY_DHDGB==1
!AP/MF
  SUBROUTINE GETCIRCLE(CX_2,CY_2,DTHETA_DX, &
                       DTHETA_DY,DTHETA_DZ, &
                       DR_DX,DR_DY,DR_DZ)
use dimens_fcm
use coord
use param
use psf
use stream
INTEGER I,J,ATMNUM,ATMNUM2,M
    REAL(chm_real) PI,TWOPI
    PARAMETER (PI=3.14159265358979323846D0,TWOPI=2.0D0*PI)
    REAL(chm_real) DTHETA_DX(NATOM,NATOM),DTHETA_DY(NATOM,NATOM)
    REAL(chm_real) DTHETA_DZ(NATOM,NATOM)
    REAL(chm_real) DR_DX(NATOM,NATOM),DR_DY(NATOM,NATOM)
    REAL(chm_real) DR_DZ(NATOM,NATOM)
    REAL(chm_real) CX_2(NATOM),CY_2(NATOM),SUMX(NATOM),SUMY(NATOM)
    REAL(chm_real) DCXDZ(NATOM,NATOM),DCYDZ(NATOM,NATOM),DERIVZ
    REAL(chm_real) WIJ(NATOM,NATOM),SUMWIJ(NATOM)
    REAL(chm_real) DELTAIJ,DELTAMJ,DELTAMI
    REAL(chm_real) PREFACT,PREFACT2,PREFACT3,FACTOR
    REAL(chm_real) TERM(NATOM,NATOM),ST(NATOM),STX(NATOM),STY(NATOM)
    PARAMETER (PREFACT2=1/(225.0D0*0.1590D0))
    PREFACT=1/DSQRT(0.1590D0*TWOPI)
    PREFACT3=2.0D0*PREFACT2*PREFACT
!___________________________________________________
    DO I=1,NATOM
       SUMX(I)=0.0D0
       SUMY(I)=0.0D0
       SUMWIJ(I)=0.0D0
       ST(I)=0.0D0
       STX(I)=0.0D0
       STY(I)=0.0D0
       DO J=1,NATOM
          ATMNUM=I
          ATMNUM2=J
          WIJ(I,J)=PREFACT*DEXP(-(Z(ATMNUM)-Z(ATMNUM2))**2*PREFACT2)
          SUMWIJ(I)=SUMWIJ(I)+WIJ(I,J)
          SUMX(I)=SUMX(I)+WIJ(I,J)*X(J)
          SUMY(I)=SUMY(I)+WIJ(I,J)*Y(J)
          TERM(I,J)=(-1.0D0)*(Z(I)-Z(J))* &
                     DEXP(-(Z(I)-Z(J))**2*PREFACT2) &
                     *PREFACT3
          ST(I)=ST(I)+TERM(I,J)
          STX(I)=STX(I)+TERM(I,J)*X(J)
          STY(I)=STY(I)+TERM(I,J)*Y(J)
       ENDDO
        CX_2(I)=SUMX(I)/SUMWIJ(I)
        CY_2(I)=SUMY(I)/SUMWIJ(I)
    ENDDO
    DO I=1,NATOM
       ATMNUM=I
       DO J=1,NATOM
          ATMNUM2=J
          IF (ATMNUM .EQ. ATMNUM2) THEN
              DELTAIJ=1.0D0
              FACTOR=-1.0D0
          ELSE
              DELTAIJ=0.0D0
              FACTOR=1.0D0
          ENDIF
          DTHETA_DX(I,J)=-((DELTAIJ-(WIJ(I,J)/SUMWIJ(I))) &
                          *(Y(I)-CY_2(I)))/ &
                          ((X(I)-CX_2(I))**2+(Y(I)-CY_2(I))**2)
          DR_DX(I,J)=((DELTAIJ-(WIJ(I,J)/SUMWIJ(I))) &
                          *(X(I)-CX_2(I)))/ &
                           (DSQRT((X(I)-CX_2(I))**2+(Y(I)-CY_2(I))**2))
          DTHETA_DY(I,J)=((DELTAIJ-(WIJ(I,J)/SUMWIJ(I))) &
                          *(X(I)-CX_2(I)))/ &
                          ((X(I)-CX_2(I))**2+(Y(I)-CY_2(I))**2)
          DR_DY(I,J)=((DELTAIJ-(WIJ(I,J)/SUMWIJ(I))) &
                          *(Y(I)-CY_2(I)))/ &
                      (DSQRT((X(I)-CX_2(I))**2+(Y(I)-CY_2(I))**2))
         IF (i .ne. j) THEN
            DCXDZ(I,J)=(-TERM(I,J)*X(J)*SUMWIJ(I)+TERM(I,J)*SUMX(I))/ &
                       (SUMWIJ(I)**2)
            DCYDZ(I,J)=(-TERM(I,J)*Y(J)*SUMWIJ(I)+TERM(I,J)*SUMY(I))/ &
                       (SUMWIJ(I)**2)
            DTHETA_DZ(I,J)=(-DCYDZ(I,J)*(X(I)-CX_2(I))+DCXDZ(I,J)* &
               (Y(I)-CY_2(I)))/((X(I)-CX_2(I))**2+(Y(I)-CY_2(I))**2)
            DR_DZ(I,J)=(-DCXDZ(I,J)*(X(I)-CX_2(I)) &
                        -DCYDZ(I,J)*(Y(I)-CY_2(I)))/ &
                        (DSQRT((X(I)-CX_2(I))**2+(Y(I)-CY_2(I))**2))
          ENDIF
          IF (((X(ATMNUM)-CX_2(I))**2+(Y(ATMNUM)-CY_2(I))**2) &
                .EQ. 0.0D0) THEN
               DTHETA_DX(I,J)=0.0d0
               DTHETA_Dy(I,J)=0.0d0
               DTHETA_Dz(I,J)=0.0d0
               DR_DX(I,J)=0.0d0
               DR_Dy(I,J)=0.0d0
               DR_Dz(I,J)=0.0d0
          ENDIF
       ENDDO
        DCXDZ(I,I)=(STX(I)*SUMWIJ(I)-ST(I)*SUMX(I))/(SUMWIJ(I)**2)
        DCYDZ(I,I)=(STY(I)*SUMWIJ(I)-ST(I)*SUMY(I))/(SUMWIJ(I)**2)
        DTHETA_DZ(I,I)=(-DCYDZ(I,I)*(X(I)-CX_2(I))+DCXDZ(I,I)* &
             (Y(I)-CY_2(I)))/((X(I)-CX_2(I))**2+(Y(I)-CY_2(I))**2)
        DR_DZ(I,I)=(-DCXDZ(I,I)*(X(I)-CX_2(I)) &
                   -DCYDZ(I,I)*(Y(I)-CY_2(I)))/ &
                   (DSQRT((X(I)-CX_2(I))**2+(Y(I)-CY_2(I))**2))
       IF (((X(i)-CX_2(I))**2+(Y(i)-CY_2(I))**2) &
            .EQ. 0.0D0) THEN
            DTHETA_DX(I,i)=0.0d0
            DTHETA_Dy(I,i)=0.0d0
            DTHETA_Dz(I,i)=0.0d0
            DR_DX(I,i)=0.0d0
            DR_Dy(I,i)=0.0d0
            DR_Dz(I,i)=0.0d0
       ENDIF
    ENDDO
    RETURN
    END SUBROUTINE GETCIRCLE
    SUBROUTINE CALC_S_LAMBDA(R,SOLD,SBAR,INNER, &
               LAMBDAFACT,SNEW,LAMBDA)
    REAL(chm_real) R,SOLD,SBAR,INNER
    REAL(chm_real) LAMBDAFACT,SNEW,LAMBDA
    REAL(chm_real) DLAMB_DR
      LAMBDA=1.0D0/(1.0D0+DEXP(LAMBDAFACT*(R-INNER)))
      SNEW=LAMBDA*SBAR+(1-LAMBDA)*SOLD
    RETURN
    END SUBROUTINE CALC_S_LAMBDA
    SUBROUTINE FIND_DEF_ENER(SPLTAB_NAME,CIRRAD, &
               DEFARRAY4,THETA, &
               DEF_ENER,DE, &
               PENALTY_OFF,PENALTY_OFFL, &
               PENALTY_PREFACT,PENALTY_PREFACTL, &
               PENALTY_FACT,PENALTY_FACTL)
use stream
use dhdgb,only:MATHICK_FHDGB,MITHICK_FHDGB,MITHICK_FHDGB
    implicit none
    REAL(chm_real) DEFARRAY(INCT_FHDGB),DEFARRAY2(INCT_FHDGB)
    REAL(chm_real) DEFARRAY3(INCT_FHDGB+1),DEFARRAY4(INCT_FHDGB)
    REAL(chm_real) THETA(INCT_FHDGB)
    REAL(chm_real) DEF_ENER,DFN,DF0,CIRRAD
    INTEGER I,J,DIST,LINEUP,LINELO,N,NUM,CO,K,CAF,QSCAPE
    REAL(chm_real) LUP(INCT_FHDGB),LO(INCT_FHDGB)
    REAL(chm_real) A(INCT_FHDGB+1),B(INCT_FHDGB),D(INCT_FHDGB)
    REAL(chm_real) H(INCT_FHDGB),ALPHA(INCT_FHDGB)
    REAL(chm_real) C(INCT_FHDGB+1),L(INCT_FHDGB+1),M(INCT_FHDGB+1)
    REAL(chm_real) Z(INCT_FHDGB+1),E(2**(INCT_FHDGB)),temp
    REAL(chm_real) VAL(INCT_FHDGB),VALLO(INCT_FHDGB),VALLUP(INCT_FHDGB)
    REAL(chm_real) C1,C2,C3,C4,C5,C6,TERM(INCT_FHDGB+1)
    REAL(chm_real) DE(INCT_FHDGB),DRIV,DJ_DK,T(INCT_FHDGB)
    integer SPLTAB_NAME
!
!    LOGICAL QSPL
    REAL(chm_real) DELTA(5),X(5)
    REAL COEFFS(1024)
    INTEGER L_I,M_I,IERROR,LU,I_I,J_I,K_I
    INTEGER G(5),BASE,OFFSET(1024),COUNT1
    REAL(chm_real) Y5(256),Y4(64),Y3(16),Y2(4),DY5(256),DY4(64),DY3(16)
    REAL(chm_real) DY2(4),DY
    REAL(chm_real) PENALTY,PENALTY_DENUM,PENALTY_NUM
    REAL(chm_real) PENALTY_OFF,PENALTY_PREFACT
    REAL(chm_real) PENALTY_FACT,OFF
    REAL(chm_real) PENALTY2
    REAL(chm_real) PENALTY_OFFL,PENALTY_PREFACTL
    REAL(chm_real) PENALTY_FACTL,OFF_L,MAXDEF,MATHICK
    INTEGER SIZEOFREAL,MAXGRID,INC_EXTEND,besh
    DO I=1,INCT_FHDGB
         DEFARRAY(I)=DEFARRAY4(I)
    ENDDO
!    besh=(4*7)**5
    besh = (4*7)**NDT_FHDGB
    SIZEOFREAL=4
    LU=18
! DIMENSION OF E IS BASED ON HAVING 5 INDEPENDENT S VALUES
    NUM=INCT_FHDGB-1
    N=INCT_FHDGB
    VALLO(1)=DINT((ABS(CIRRAD-MIR_FHDGB))/INC_FHDGB)*INC_FHDGB+ &
             MIR_FHDGB
    VALLUP(1)=VALLO(1)+INC_FHDGB
    IF (VALLO(1) .LE. MIR_FHDGB) THEN
        VALLO(1)=MIR_FHDGB
        VALLUP(1)=VALLO(1)+INC_FHDGB
    ENDIF
    T(1)=(CIRRAD-VALLO(1))/(VALLUP(1)-VALLO(1))
    DEFARRAY3(1)=CIRRAD
    DO I=1,INCT_FHDGB
         DEFARRAY2(I)=DEFARRAY(I)
         DEFARRAY3(I+1)=DEFARRAY(I)
    ENDDO
    DO I=2,INCT_FHDGB
       CAF=I-1
           IF (DEFARRAY(CAF).GT. MATHICK_FHDGB) THEN
               DEFARRAY(CAF)=MATHICK_FHDGB- &
               (DEFARRAY(CAF)-INT(DEFARRAY(CAF)/MATHICK_FHDGB)* &
               MATHICK_FHDGB)
           ENDIF
       DIST=DINT((DEFARRAY(CAF)-MITHICK_FHDGB)/INCTH_FHDGB)
       VALLO(I)=DIST*INCTH_FHDGB+MITHICK_FHDGB
       VALLUP(I)=VALLO(I)+INCTH_FHDGB
        IF (VALLO(I) .GE. MATHICK_FHDGB) THEN
            VALLUP(I)=MATHICK_FHDGB
            VALLO(I)=VALLUP(I)-INCTH_FHDGB
        ENDIF
        IF (VALLUP(I) .LE. MITHICK_FHDGB) THEN
            VALLO(I)=MITHICK_FHDGB
            VALLUP(I)=VALLO(I)+INCTH_FHDGB
        ENDIF
        T(I)=(DEFARRAY(CAF)-VALLO(I))/(VALLUP(I)-VALLO(I))
     ENDDO
     LINEUP=0
     LINELO=0
     I=0
     DEF_ENER=0
     DO J=1,INCT_FHDGB-1
        DE(J)=0.0D0
     ENDDO
!CALCULATE SPLINE
         MAXDEF=10.0D0
         MATHICK=MATHICK_FHDGB+MAXDEF
         INC_EXTEND=5
         MAXGRID=DINT(DABS(MATHICK)/INC_EXTEND)+1
         BASE=0
         DO I=1,INCT_FHDGB-1
             X(I)=DABS(DEFARRAY2(I))-MATHICK_FHDGB
!              X(I)=DEFARRAY2(I)-MATHICK
!             IF (X(I) .GT. 0.D0) THEN
!                 X(I)=-X(I)
            IF (DEFARRAY2(I) .LT. MITHICK_FHDGB ) THEN
                X(I)=-MATHICK_FHDGB
            ENDIF
          IF (DEFARRAY2(I) .LT. MITHICK_FHDGB ) THEN
              X(I)=-MATHICK_FHDGB
          ENDIF
         ENDDO
         DO I=INCT_FHDGB-1,1,-1
          IF (I .GT. NDT_FHDGB) THEN
              G(I)=1
          ELSE
            G(I)=DINT(DABS(-MATHICK_FHDGB-X(I))/INC_EXTEND)+1
            IF (G(I) .GE. MAXGRID) THEN
                G(I)=MAXGRID-1
            ENDIF
            DELTA(I)=MATHICK_FHDGB-(G(I)-1)*INC_EXTEND+X(I)
            IF (DEFARRAY2(I) .LT. MITHICK_FHDGB) THEN
                DELTA(I)=-(MITHICK_FHDGB-DEFARRAY2(I))
            ENDIF
          ENDIF
            BASE=BASE+(G(I)-1)*28**(I-1)
         ENDDO
         COUNT1=0
     
     IF (.NOT. QSMALL) THEN     
      ERRORCHECK: IF (.NOT. QOPEN_FHDGB )THEN
           CALL WRNDIE(0,'<SETMEMFLUCT>','SP_TAB NOT OPEN')
      ELSE 
       DO I_I=1,22,7
        DO J_I=1,22,7
         DO K_I=1,22,7
          DO L_I=1,22,7
           DO M_I=1,22,7
              COUNT1=COUNT1+1
              OFFSET(COUNT1)=(M_I-1)+(L_I-1)*28+ &
                             (K_I-1)*28**2+(J_I-1)*28**3+  &
                             (I_I-1)*28**4+1+BASE
              if (OFFSET(COUNT1) .GT. besh) then
                  write(outu,*) 'ERROR: DIMENSIONS OF THE LOOKUP TABLE DONOT MATCH'
              endif
              READ(SPLTAB_NAME,REC=OFFSET(COUNT1)) COEFFS(COUNT1)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDIF ERRORCHECK
     ELSE
!IF QSMALL
         DO K_I=1,22,7
          DO L_I=1,22,7
           DO M_I=1,22,7
              COUNT1=COUNT1+1
              OFFSET(COUNT1)=(M_I-1)+(L_I-1)*28+ &
                             (K_I-1)*28**2+1+BASE
              if (OFFSET(COUNT1) .GT. besh) then
                  write(outu,*)'ERROR: DIMENSIONS OF THE LOOKUP TABLE DONOT MATCH'
              endif
              COEFFS(COUNT1)=LOOKUP_COEFF(OFFSET(COUNT1))
           ENDDO
          ENDDO
         ENDDO
!IF QSMALL       
     ENDIF
       DO I=1,256
        IF (.NOT. QSMALL) THEN
          Y5(I)=COEFFS(I)*DELTA(5)**3+COEFFS(I+256)*DELTA(5)**2+ &
                COEFFS(I+512)*DELTA(5)+COEFFS(I+768)
          DY5(i)=3*COEFFS(I)*DELTA(5)**2+2*COEFFS(I+256)*DELTA(5)+ &
                COEFFS(I+512)
        ELSE
          Y5(I)=0.0D0
          DY5(i)=0.0D0
        ENDIF
       ENDDO
       DO I=1,64
        IF (.NOT. QSMALL) THEN  
          Y4(I)=Y5(I)*DELTA(4)**3+Y5(I+64)*DELTA(4)**2+ &
                Y5(I+128)*DELTA(4)+Y5(I+192)
          DY4(I)=3*Y5(I)*DELTA(4)**2+2*Y5(I+64)*DELTA(4)+ &
                 Y5(I+128)
        ELSE
          Y4(I)=0.0D0
          DY4(I)=0.0D0 
        ENDIF 
       ENDDO
       DO I=1,16
        IF (.NOT. QSMALL) THEN 
          Y3(I)=Y4(I)*DELTA(3)**3+Y4(I+16)*DELTA(3)**2+ &
               Y4(I+32)*DELTA(3)+Y4(I+48)
          DY3(I)=3*Y4(I)*DELTA(3)**2+2*Y4(I+16)*DELTA(3)+ &
                Y4(I+32)
        ELSE
          Y3(I)=COEFFS(I)*DELTA(3)**3+COEFFS(I+16)*DELTA(3)**2+ &
                COEFFS(I+32)*DELTA(3)+COEFFS(I+48)
          DY3(I)=3*COEFFS(I)*DELTA(3)**2+2*COEFFS(I+16)*DELTA(3)+ &
                COEFFS(I+32)  
        ENDIF
      ENDDO
      DO I=1,4
          Y2(I)=Y3(I)*DELTA(2)**3+Y3(I+4)*DELTA(2)**2+ &
               Y3(I+8)*DELTA(2)+Y3(I+12)
          DY2(I)=3*Y3(I)*DELTA(2)**2+2*Y3(I+4)*DELTA(2)+ &
                 Y3(I+8)
      ENDDO
          DEF_ENER=Y2(1)*DELTA(1)**3+Y2(2)*DELTA(1)**2+ &
                   Y2(3)*DELTA(1)+Y2(4)
!END CALCULATE THE SPLINES
!CALCULATE THE DERIVATIVES OF THE DEFORMATION ENERGY WITH RESPECT TO
!DELTA(1) TO DELTA(5)
      DO I=1,64
         DY5(I)=DY5(I)*DELTA(4)**3+DY5(I+64)*DELTA(4)**2+ &
                DY5(I+128)*DELTA(4)+DY5(I+192)
      ENDDO
      DO I=1,16
         DY5(I)=DY5(I)*DELTA(3)**3+DY5(I+16)*DELTA(3)**2+ &
                DY5(I+32)*DELTA(3)+DY5(I+48)
      ENDDO
      DO I=1,4
         DY5(I)=DY5(I)*DELTA(2)**3+DY5(I+4)*DELTA(2)**2+ &
                DY5(I+8)*DELTA(2)+DY5(I+12)
      ENDDO
      DE(INCT_FHDGB)=0.d0
      DE(INCT_FHDGB)=DY5(1)*DELTA(1)**3+DY5(2)*DELTA(1)**2+ &
                     DY5(3)*DELTA(1)+DY5(4)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      DO I=1,16
         DY4(I)=DY4(I)*DELTA(3)**3+DY4(I+16)*DELTA(3)**2+ &
                DY4(I+32)*DELTA(3)+DY4(I+48)
      ENDDO
      DO I=1,4
         DY4(I)=DY4(I)*DELTA(2)**3+DY4(I+4)*DELTA(2)**2+ &
                DY4(I+8)*DELTA(2)+DY4(I+12)
      ENDDO
      DE(INCT_FHDGB-1)=DY4(1)*DELTA(1)**3+DY4(2)*DELTA(1)**2+ &
                       DY4(3)*DELTA(1)+DY4(4)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      DO I=1,4
         DY3(I)=DY3(I)*DELTA(2)**3+DY3(I+4)*DELTA(2)**2+ &
                DY3(I+8)*DELTA(2)+DY3(I+12)
      ENDDO
      DE(INCT_FHDGB-2)=DY3(1)*DELTA(1)**3+DY3(2)*DELTA(1)**2+ &
                       DY3(3)*DELTA(1)+DY3(4)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      DE(INCT_FHDGB-3)=DY2(1)*DELTA(1)**3+DY2(2)*DELTA(1)**2+ &
                       DY2(3)*DELTA(1)+DY2(4)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      DE(INCT_FHDGB-4)=3*Y2(1)*DELTA(1)**2+2*Y2(2)*DELTA(1)+ &
                       Y2(3)
!      DO I =1, INCT_FHDGB
!         write(outu,*) DE(I)
!      ENDDO
!END OF DERIVATIVE CALCULATION
!THIS PART CALCULATES THE EXTRA PENALTY CONSIDERED FOR EXPANSION
      OFF=MATHICK_FHDGB+PENALTY_OFF+10.0D0
      OFF_L=MITHICK_FHDGB+PENALTY_OFFL
!        write(outu,*)DEF_ENER
      DO I=2,INCT_FHDGB
!          PENALTY_DENUM=1.0D0+DEXP(PENALTY_FACT*(OFF-DEFARRAY2(I-1)))
!          PENALTY_NUM=PENALTY_PREFACT*(OFF-DEFARRAY2(I-1))**2
!          PENALTY=PENALTY_NUM/PENALTY_DENUM
         PENALTY=PENALTY_PREFACT* &
                 DEXP(PENALTY_FACT*(DEFARRAY2(I-1)-OFF))
         PENALTY2=PENALTY_PREFACTL* &
                  DEXP(PENALTY_FACTL*(-DEFARRAY2(I-1)-OFF_L))
         DEF_ENER=DEF_ENER+PENALTY+PENALTY2
         DE(I)=DE(I)+PENALTY_FACT*PENALTY-PENALTY_FACTL*PENALTY2
!          DE(I)=DE(I)+(-2.0D0*PENALTY_PREFACT*(OFF-DEFARRAY2(I-1))*
!     &    PENALTY_DENUM+PENALTY_FACT*PENALTY_NUM
!     &    *(PENALTY_DENUM-1.0D0))/
!    &    PENALTY_DENUM**2
        IF (QSMALL) THEN
           IF (I .GT. NDT_FHDGB+1) THEN
               DE(I) = 0.0D0
           ENDIF
        ENDIF  
      ENDDO
     RETURN
    END SUBROUTINE FIND_DEF_ENER
    SUBROUTINE CALC_CSAPE_SPL(DEFARRAY2,C4_CS,C3_CS,C2_CS,MEMSPL)
use dhdgb,only:MATHICK_FHDGB
! TEST
use stream
     REAL(chm_real) DEFARRAY2(INCT_FHDGB),MEMSPL(4,INCT_FHDGB)
     REAL(chm_real) DEFARRAY(INCT_FHDGB)
     REAL(chm_real) C4_CS(INCT_FHDGB-1,INCT_FHDGB)
     REAL(chm_real) C3_CS(INCT_FHDGB-1,INCT_FHDGB)
     REAL(chm_real) C2_CS(INCT_FHDGB-1,INCT_FHDGB)
     INTEGER I,J
     DO I=1,INCT_FHDGB
        DEFARRAY(I)=DEFARRAY2(I)-MATHICK_FHDGB
     ENDDO
     DO I=1,INCT_FHDGB-1
        MEMSPL(1,I)=0.0D0
        MEMSPL(2,I)=0.0D0
        MEMSPL(3,I)=0.0D0
        DO J=1,INCT_FHDGB
           MEMSPL(1,I)=MEMSPL(1,I)+C4_CS(I,J)*DEFARRAY(J)
           MEMSPL(2,I)=MEMSPL(2,I)+C3_CS(I,J)*DEFARRAY(J)
           MEMSPL(3,I)=MEMSPL(3,I)+C2_CS(I,J)*DEFARRAY(J)
        ENDDO
           MEMSPL(4,I)=DEFARRAY(I)
     ENDDO
     RETURN
     END SUBROUTINE CALC_CSAPE_SPL
     SUBROUTINE CalsFHDGB(XATM,YATM,theta,CXAF,CYAF, &
                SINIT,MEMSPL,S,DTHETA,DS,C4_CS,C3_CS, &
                C2_CS,C1_CS)
!XATM,YATM      ATOMIC COORDINATES
!CX,CY      CENTER OF THE CIRCLE
!SINIT          INITIAL VALUES OF MEMBRANE THICKNESS ON THE
!               ENCOMPASSING CIRCLE
!MEMSPL         SPLINE COEFFICIENTS OF S VS THETA
!S              MEMBRANE THICKNESS AT POSITION OF THE ITH ATOM
!DTHETA         DS(I)/DTHETA
!DS             DS(I)/DS(1,...,INCT_FHDGB-1)
use consta
use stream
use dimens_fcm
use param
use number
use dhdgb,only:MATHICK_FHDGB
     REAL(chm_real) XATM,YATM,S,ANG,THETA
     REAL(chm_real) DEL_THETA,DISTX,DISTY,RATIO,CXAF,CYAF
     REAL(chm_real) DTHETA
     REAL(chm_real) MEMSPL(4,INCT_FHDGB-1),SINIT(INCT_FHDGB),INC
     REAL(chm_real) DMEMSPL(4,INCT_FHDGB-1,INCT_FHDGB),DS(INCT_FHDGB-1)
     REAL(chm_real) C4_CS(INCT_FHDGB-1,INCT_FHDGB)
     REAL(chm_real) C3_CS(INCT_FHDGB-1,INCT_FHDGB)
     REAL(chm_real) C2_CS(INCT_FHDGB-1,INCT_FHDGB)
     REAL(chm_real) C1_CS(INCT_FHDGB-1,INCT_FHDGB)
     INTEGER I,J
! I REPRESNETS THETA INTERVAL J REPRESENTS S VALUES
!      DO I=1,INCT_FHDGB-1
!        DO J=1,INCT_FHDGB
!           WRITE(OUTU,*) 'DD(',I,')/DS(',J,')=',DMEMSPL(1,I,J)
!        ENDDO
!      ENDDO
!      DO I=1,INCT_FHDGB
!        DO J=1,INCT_FHDGB
!           WRITE(OUTU,*) 'DA(',I,')/DS(',J,')=',DMEMSPL(4,I,J)
!        ENDDO
!      ENDDO
     DISTX=XATM-CXAF
     DISTY=YATM-CYAF
     THETA=0.0D0
     RATIO=DABS(DISTY/DISTX)
!      write(outu,*)'cx=',cx,'cy=',cy
!      write(outu,*)'ratio=',ratio
     IF ((DISTX .NE. 0.D0) .AND. (DISTY .NE. 0.D0)) THEN
          THETA=ATAN(RATIO)
!           write(outu,*)'atan=',THETA
          IF ((XATM .LT. CXAF) .AND. (YATM .GT. CYAF)) THEN
               THETA=PI-THETA
          ELSEIF ((XATM .GT. CXAF) .AND. (YATM .LT. CYAF)) THEN
                   THETA=TWOPI-THETA
          ELSEIF ((XATM .LT. CXAF) .AND. (YATM .LT. CYAF)) THEN
                   THETA=PI+THETA
          ENDIF
     ELSEIF ((DISTX .EQ. 0.D0) .AND. (YATM .GT. CYAF)) THEN
           THETA=PI/2.D0
     ELSEIF ((DISTX .EQ. 0.D0) .AND. (YATM .LT. CYAF)) THEN
           THETA=3*PI/2.D0
     ELSEIF ((XATM .GT. CXAF) .AND. (DISTY .EQ. 0.D0)) THEN
           THETA=0.D0
     ELSEIF ((XATM .LT. CXAF) .AND. (DISTY .EQ. 0.D0)) THEN
           THETA=PI
     ENDIF
!      write(outu,*)'theta=',THETA
     INC=TWOPI/(NDT_FHDGB)
     I=DINT((THETA-THETAZ(1))/INC)+1
     DEL_THETA=DABS(THETA-THETAZ(1))-INC*(I-1)
     IF (I .EQ. NDT_FHDGB+1) THEN
         I=1
     ENDIF
     IF (I .LT. NDT_FHDGB+1) THEN
         S=MEMSPL(1,i) * DEL_THETA**3 + MEMSPL(2,i) * DEL_THETA**2 &
           + MEMSPL(3,i) * DEL_THETA + MEMSPL(4,i) + MATHICK_FHDGB
         DTHETA=3*MEMSPL(1,i)*DEL_THETA**2+2*MEMSPL(2,i)*DEL_THETA &
           + MEMSPL(3,i)
     ELSE
         S=SINIT(NDT_FHDGB+1)
         DTHETA=3*MEMSPL(1,1)*DEL_THETA**2+2*MEMSPL(2,1)*DEL_THETA &
             + MEMSPL(3,1)
     ENDIF
     DO J=1,INCT_FHDGB-1
        DS(J)=0.0D0
     ENDDO 
     DO J=1,NDT_FHDGB
          DS(J)=C4_CS(I,J)*DEL_THETA**3+C3_CS(I,J)*DEL_THETA**2 &
                +C2_CS(I,J)*DEL_THETA+C1_CS(I,J)
     ENDDO
          DS(1)=DS(1)+(C4_CS(I,NDT_FHDGB+1)*DEL_THETA**3+ &
                C3_CS(I,NDT_FHDGB+1)*DEL_THETA**2 &
                +C2_CS(I,NDT_FHDGB+1)*DEL_THETA+C1_CS(I,NDT_FHDGB+1))
     RETURN
     END SUBROUTINE CalsFHDGB
#endif


end module gbmv

