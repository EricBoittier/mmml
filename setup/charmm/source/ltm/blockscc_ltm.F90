!CHARMM Element source/fcm/blockscc.fcm $Revision: 1.6 $

module blockscc_fcm
  use chm_kinds
#if KEY_SCCDFTB==1
  use sccdftbsrc, only: nndim     /*Puja */
#endif
  integer :: blockscc_fcm_dummy_var
#if KEY_SCCDFTB==1 /*sccdftb*/
#if KEY_BLOCK==1 /*block*/
  !
  INTEGER,PARAMETER :: M1AXN3 = 1950,  M1XQM2=100, M1XRP=50

  LOGICAL,save :: qsccb,qstop,qdtop,qdone,qdone1,qdone2
  logical,save :: qsccupdt,qmcqm,qmcqm1,qmcqm2,qpkac

  INTEGER,save :: nsccl,icntdyn,iavti,tiav,idxbnd,idxphi
  integer,save :: nmoved,idxmoved(M1AXN3),idxdone,idxnbd

  real(chm_real),save ::  cdvdl(6),dvdl,dvdlb,dvdla,dvdlp,dvdle,dvdlv,dvdlip
  real(chm_real),save ::  dvdl0,dvdl1,dvdltmp1,dvdltmp2,dtmp,dtmp1, &
       dvdlub,dvdlcp
  real(chm_real),save ::  dvdlscc,dvdlav,tico
  real(chm_real),save ::  dvdlsccold,eqmeold,depotscc,eqmeold1,eqmeold0
  real(chm_real),save ::  dvdlsccold1,dvdlsccold0,ddvdlscc,pte1,pte2
  real(chm_real),save ::  oldcqm(M1AXN3,3),dvdlv1

  integer,save :: idxqmoved(M1AXN3),idxhyd,idxstp,outpka 

  ! for lamda from guohui (MG_121101: extended to lcolspin)
  real(chm_real),save :: TELEC1,SCFTOL1,scclamd, &
       dvdlold0,dvdlold1,dvdlold2,ddvdl
  INTEGER,save :: MXITSCF1,ichscc1,sccpass
  LOGICAL,save :: LPRVEC1,lcolspin1,qlamda
  real(chm_real),save :: TELEC2,SCFTOL2,scal,ETERM1,ETERM2
  INTEGER,save :: MXITSCF2,ichscc2,sccstep
  LOGICAL,save :: LPRVEC2,lcolspin2
  real(chm_real),save :: unpe1,unpe2

  ! QC: 12/09 - get rid of islct1,islct2 - we don't need to save those stack variables!

  ! for lamda from guohui
  real(chm_real) ,save :: scczin1(m1axn3),scczin2(m1axn3)
  INTEGER        ,save :: scctyp1(m1axn3),scctyp2(m1axn3)
  CHARACTER(len=10),save :: sccatm1(m1axn3),sccatm2(m1axn3)
  integer        ,save :: iqmlst1(m1xqm2,m1xrp),iqmlst2(m1xqm2,m1xrp)
  ! QC: 11/17 - PERT based on Xiya
  ! INTEGER        ,save :: nptc1,nscctc1,nscctc2
  LOGICAL,save           :: qsccpert
  INTEGER,save,allocatable,dimension(:) :: igmsel1,igmsel2
  real(chm_real),save,allocatable,dimension(:) :: charge1,charge2
  INTEGER        ,save :: nptc1,nptc2, nscctc1,nscctc2      !Xiya
  ! QC: 11/17 Done
  !     -------------------------------------------------------------
  ! for parameters from dylcao
  !integer,parameter :: maxtyp=6,nndim=650,maxint=160,maxtab=600
  !MG_121101: nndim is set in sccdftbsrc_ltm and imported here!
  integer,parameter :: nndim2=nndim 
  ! for A  (MG_121101: extended to lcolspin and bookkeeping of charges)
  integer,save :: izp2a(NNDIM2)
  real(chm_real),save :: nela,nelupa,neldowna
  real(chm_real),save :: qmata(nndim),qla(3*nndim),qlupa(3*nndim)
  real(chm_real),save :: qldowna(3*nndim)
  logical,save :: qaflag

  ! for B
  integer,save :: izp2b(NNDIM2)
  real(chm_real),save :: nelb,nelupb,neldownb    
  real(chm_real),save :: qmatb(nndim),qlb(3*nndim),qlupb(3*nndim)
  real(chm_real),save :: qldownb(3*nndim)
  logical,save :: qbflag

  ! Some restart options? 
  integer,save :: icntdyn1,iavti1
  real(chm_real),save ::  dvdlav1,dvdl2
  logical,save :: qsccres

  !     -------------------------------------------------------------
  !     QC_UW04:Cleaned up some
  !     QC_UW10:The following should be unique to SCCDFTB src - so 
  !     move to sccdftbsrc/sccdftbsrc_ltm
  ! integer,save :: izp2(NNDIM), lmax(MAXTYP)
  ! integer,save :: dim(MAXTYP,MAXTYP)
  ! integer,save :: numint(MAXTYP,MAXTYP)
  ! real(chm_real),save :: skhtab(10,MAXTAB,MAXTYP,MAXTYP)
  ! real(chm_real),save :: skstab(10,MAXTAB,MAXTYP,MAXTYP)
  ! real(chm_real),save :: skself(3,MAXTYP), sr(MAXTYP,MAXTYP), nel
  ! real(chm_real),save :: xr(2,MAXINT,MAXTYP,MAXTYP)
  ! real(chm_real),save :: coeff(6,MAXINT,MAXTYP,MAXTYP)
  ! real(chm_real),save :: efkt(3,MAXTYP,MAXTYP),cutoff(MAXTYP,MAXTYP)
  ! real(chm_real),save :: espin(MAXTYP),Edis
  ! real(chm_real),save :: racc,qzeroscc(maxtyp), uhubb(maxtyp)
  ! integer,save :: nbeweg 
  ! logical,save :: dispers,writeout

  !     QC: UW_06 Add Gaussian SRP fit (Haibo)
  ! real(chm_real),save :: gbump(3,maxtyp,maxtyp),ronsw(maxtyp,maxtyp)
  ! real(chm_real),save :: deltar(maxtyp,maxtyp)
  ! logical,save :: lgausw

#endif /* (block)*/
#endif /* (sccdftb)*/
end module blockscc_fcm


