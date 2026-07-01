module zmodule1
 use chm_kinds
 implicit none

 contains
#if KEY_ZEROM==0 /*zeromain*/
SUBROUTINE ZEROM
  CALL WRNDIE(-1,'<CHARMM>','ZERO code (ZEROM kywd) not compiled.')
  return
end SUBROUTINE ZEROM
#else /* (zeromain)*/

SUBROUTINE ZEROM
  !-----------------------------------------------------------------------
  ! Zero-order minimization module
  !
  !          Robert J. Petrella
  !           Department of Chemistry and Chemical Biology,
  !            Harvard University  
  !                                   2002-2006
  !
  !  The zero module performs grid searches (zero-order minimization)
  !  in N-dimensional conformational space.
  !
  !  This first subroutine controls all 'zero' subcommands. The format
  !  of the module is modelled after that of PBEQ0.
  !-----------------------------------------------------------------------
  !    
  !
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use comand
  use psf
  use select
  use stream
  use string
  use coord
  use zdata_mod
  use bases_fcm
  use intcor_module
  use energym
  use memory
  use actclus_mod
#if KEY_PARALLEL==1
  use parallel, only: MYNODP,NUMNOD,NUMNODG,MYNOD,MYNODG,MYNODGP 
  use nbndcc_utilb,only: parstoperr  !temporary
  use cpustruc,only: QCPUONES
  use paral4,only: cpu_commarr
#endif
  use zmod_comm 
  use zmodule2
  use zutil,only: set_randseed,rdzconf,randsel,print_randseed
  use zcs,only: cs_alloc,cs_write,cs_trim,cs_copy
  use zstruc,only: CSR,CSW
  use nbndcc_utilb,only: parstoperr
  !  use cpustruc,only: NUMNODH,IMYKEYP,HOODCOM

  implicit none
  !
  !-----------------------------------------------------------------------
  !  Miscelaneous Local variables
  !  Local variable
  integer,allocatable,dimension(:) :: HPTALI_hv
  integer,allocatable,dimension(:) :: HPTIND_hv
  integer,allocatable,dimension(:) :: HPSALI_hv
  integer,allocatable,dimension(:) :: HPSCUR_hv
  integer,allocatable,dimension(:) :: HPTMPD_hv
  real(chm_real),allocatable,dimension(:) :: HPTMPV_hv
  integer,allocatable,dimension(:) :: HPCMPD_hv
  real(chm_real),allocatable,dimension(:) :: HPCMPV_hv
  integer,allocatable,dimension(:) :: HPSDOF_hv
  real(chm_real),allocatable,dimension(:) :: HPSVAL_hv
  integer,allocatable,dimension(:) :: HPSTMD_hv
  integer,allocatable,dimension(:) :: HPSSSL_hv
  integer,allocatable,dimension(:) :: HPSSS2_hv
  integer,allocatable,dimension(:) :: HPMXPS_hv
  integer,allocatable,dimension(:) :: HPPSIT_hv
  integer,allocatable,dimension(:) :: ISLCT_sv
  integer,allocatable,dimension(:) :: JSLCT_sv

  INTEGER       I,ISLCT,LSTPBI,NTPBI,IMODE,NCEL
  INTEGER       JSLCT
  INTEGER       LISTR,FISTR,DDEFNB,DSUBSP,DDEFDIS
  CHARACTER(len=4)   WRD,WORD,WRD2
  LOGICAL       DONE,EOF,LUSED,OK,QSUBSA,ERR,QMINOR,QDISTA,QLOCREV
  !     & QICCNA 
  real(chm_real)        EGSBP,TEMPF,DISTLT,DISTGT
  !  for reading conformer file:
  INTEGER HPTMPD,HPTMPV,HPCMPD,HPCMPV,HPSDOF,HPSVAL
  INTEGER HPSTMD
  INTEGER RDZCUN,WRZCUN
  !  for adding substructure definitions
  INTEGER HPTALI,HPSALI,HPSCUR,HPTIND
  INTEGER ALIASN
  ! calling for combinations
  INTEGER ZNTAKE,ZPMODE,HPSSSL,HPSSS2 
  INTEGER HPMXPS,HPPSIT,HPSSNM
  ! for parsing distance constraints
  real(chm_real) MINUS1
  ! for parsing 1st order minimizatino
  INTEGER NSTEPSD,NSTEPCG
  ! for dof constraints
  INTEGER DDEFDCON,LOCDOF
  integer :: II,CNF,JJ

! for random selection of conformers
  real(chm_real) :: ZRNDVTOL !tolerance above min for ener cutoff, rand selected confs 
  logical :: QZRNDVCT=.false. !associated flag
  real(chm_real) :: ZRNDECUT !absolute cutoff for  energies, rand selection
  logical :: QZRNDECT=.false. !associated flag
  integer :: ZRANDWU !write unit for randomly selected conformers

! for energy cutoffs in conformer selection:
  real(chm_real),allocatable,dimension(:),save :: ECUTAR  !holds energy cutoffs
  logical :: QZECUTR !flag for ECUTAR present
  real(chm_real) :: ZLVTOL,ZGVTOL  !local and global energy band widths for cutoffs
  logical :: QZLVCT,QZGVCT  !associated flags
  character(len=4) :: NXTA
  real(chm_real) :: ZGECUT 
  logical :: QZGECT,QVERBOSE=.false.
  integer :: ZFILTWU !write unit for filtered conformers

#if KEY_PARALLEL==1
 include 'mpif.h'
 integer,parameter :: NDATA=10
 integer,dimension(NDATA) :: IDATA
 integer(chm_int4) :: IERR, NODE0
#endif
  ! 
  ! misc
  INTEGER DEFCNT  !local definition count
  integer :: SS  

!  write(mynodgp+10,*) 'beginning of zmodule1 NSUBSP is ',NSUBSP
  QZMOD = .TRUE.
  !

#if KEY_PARALLEL==1
  if(MYNODP.eq.1) then  
#endif
  WRITE(OUTU,'(3X,A61)') &
       '*************************************************************'
  WRITE(OUTU,100) &
       'Entering zero module'
  WRITE(OUTU,*)
#if KEY_PARALLEL==1
  endif 
#endif
#if KEY_ACTBOND==0 /*actbond1*/
  WRITE(OUTU,'(A)')  &
       'ACTBOND CODE MUST BE COMPILED FOR Z MODULE TO RUN'
  call WRNDIE(-5,'<ZEROM>', &
       'ACTBOND CODE NOT COMPILED')
#endif /*actbond1*/
  !
100 FORMAT(3X,A)
  !-----------------------------------------------------------------------
1000 CALL XTRANE(COMLYN,COMLEN,'ZERO')

  OK    = .TRUE.
  LUSED = .FALSE.
  DONE  = .FALSE.
  EOF   = .FALSE.
  QSSDIS = .FALSE. 
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
       '  ZERO> ')

  IF(EOF)THEN
     CALL PPSTRM(OK)
     IF(.NOT.OK)  RETURN
  ENDIF
  CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
  IF(LUSED) GOTO 1000

  WRD  = '    '
  WRD=NEXTA4(COMLYN,COMLEN)
  IF(WRD.EQ.'    ') GOTO 1000

  !.......................................................................
  IF (WRD.EQ.'RESE') THEN
     !        -------------
     IF (QZEROMEM) THEN
        CALL CLEARZMEM
        QZEROMEM    = .FALSE.
        QSSDIS = .FALSE.
     else
        WRITE(OUTU,100) 'Z MODULE memory not allocated'
     endif
     !.......................................................................
  ELSE IF(WRD.EQ.'ZMEM') THEN
     !        -------------
     !     DDEFME   number of dof definitions
     !     NVALME   length of master doflist (# of values)
     !     NCNFME   number of conformers 
     !     NSUBME   number of subspaces/substructures
     !     NSATME   number of total atoms in all substructures
     !     MXCNSZ   conformer size (# dofs)
     !     NDISATME  number of atoms in possible distance constraints
     !     NDISTME  number of possible distance constraints
     !     NDOFCME  number of possible dof constraints
     !        -------------
#if KEY_PARALLEL==1 /*mynodgp1*/
    if(MYNODGP.eq.1) then   
#endif
     DDEFME = GTRMI(COMLYN,COMLEN,'NDOF',-1)
     NVALME = GTRMI(COMLYN,COMLEN,'NVAL',-1)
     NCNFME = GTRMI(COMLYN,COMLEN,'NCON',-1)
     NSUBME = GTRMI(COMLYN,COMLEN,'NSUB',-1)
     NSATME = GTRMI(COMLYN,COMLEN,'NATO',-1)
     MXCNSZ = GTRMI(COMLYN,COMLEN,'CSIZ',-1)
     NDISATME = GTRMI(COMLYN,COMLEN,'NDAT',0)
     NDISTME = GTRMI(COMLYN,COMLEN,'NDIS',0)
     NDOFCME = GTRMI(COMLYN,COMLEN,'NDFC',0)
#if KEY_PARALLEL==1
     MAXLNCNT=GTRMI(COMLYN,COMLEN,'MXLN',100000) 
#endif
     if (DDEFME.LE.0) &
          call WRNDIE(-5,'<ZEROM>', &
          'NO MEMORY FOR DOF DEFINITIONS ALLOTTED (ZMEM NDOF)')
     IF (NSUBME.LE.0) &
          CALL WRNDIE(-5,'<ZEROM>', &
          'NO SUBSPACE/SUBSTRUC MEMORY ALLOTTED (ZMEM NSUB)')
     IF (NSATME.LE.0) &
          CALL WRNDIE(-5,'<ZEROM>', &
          'NO SUBSPACE ATOM MEMORY ALLOCATED (ZMEM NATO)')
     IF(NVALME.LE.0) CALL WRNDIE(-5,'<ZEROM>', &
          'NO MEMORY ALLOCATED FOR MAIN DOF VALUE LIST (ZMEM NVAL)')
     IF(NCNFME.LE.0) CALL WRNDIE(-5,'<ZEROM>', &
          'NO MEMORY ALLOCATED FOR CONFORMERS  (ZMEM NCON)')
     IF(NSUBME.LE.0) CALL WRNDIE(-5,'<ZEROM>', &
          'NO MEMORY ALLOCATED FOR SUBUNIT DEFINITIONS (ZMEM NSUB)')
#if KEY_PARALLEL==1 /*mynodgp1*/
   endif 
#endif /*mynodgp1*/

#if KEY_PARALLEL==1
   IDATA = 0  !array
   NODE0 = 0
   if(MYNODGP.eq.1) then
    IDATA(1) = DDEFME
    IDATA(2) = NVALME
    IDATA(3) = NCNFME
    IDATA(4) = NSUBME
    IDATA(5) = NSATME
    IDATA(6) = MXCNSZ
    IDATA(7) = NDISATME
    IDATA(8) = NDISTME
    IDATA(9) = NDOFCME
    IDATA(10) = MAXLNCNT
   endif
   call MPI_BCAST(IDATA,NDATA,MPI_INTEGER,NODE0,MPI_COMM_WORLD,IERR)
   if(MYNODGP.ne.1) then
    DDEFME = IDATA(1)
    NVALME = IDATA(2) 
    NCNFME = IDATA(3)
    NSUBME = IDATA(4)
    NSATME = IDATA(5) 
    MXCNSZ = IDATA(6) 
    NDISATME = IDATA(7) 
    NDISTME = IDATA(8) 
    NDOFCME = IDATA(9)
    MAXLNCNT = IDATA(10)
   endif
#endif
     !
     ! allocate space
     !        -------------
     !     HPMDFL   for MSTDOF master dof list
     !     HPMDFV   for MSTDFV master dof values
     !     MSTDOFW  master dof list, uncompressed
     !     MSTDFVW  master dof value list, uncompressed 
     !     HPLODF   for LODOFL lo master doflist pointer
     !     HPHIDF   for HIDOFL hi master doflist pointer
     !     LODOFLW   for LODOFL lo master doflist pointer, uncompressed
     !     HIDOFLW   for HIDOFL hi master doflist pointer, uncompressed
     !     HPLOCN   for LOCONF lo conformer pointer
     !     HPHICN   for HICONF hi conformer pointer
     !     HPLOIE   for LOICEN lo pointer into ic line entry list
     !     HPHIIE   for HIICEN hi pointer into ic line entry list
     !     HPGMNV   for GMINVAL global minimum dof values in 1 search
     !     HPLDSMN  for LDDSSMN glob min dof vals for loaded ss's
     !     HPLNLS   for LNLIST ic line entries
     !     HPCLST   for COLLST ic column entries
     !     HPSATB   for SATBEG substruc atom lo base array
     !     HPSATE   for SATEND substruc atom hi base array
     !     HPSALS   for SUBATL substruc atom list
     !     HPSIAB   for SIABEG subst atom init lo base array
     !     HPSIAE   for SIAEND subst atom init hi base array
     !     HPINIL   for INIATL subst atom init list  
     !     HPLDSS   for LDSSLS loaded subspaces
     !     HPMINSS  for MINORSS minor subspace list
     !     HPIMISS  for IMINORSS minor subspace flags (1 or 0)
     !     HPALIA   for ALILST list of subspace aliases
     !     HPALRF   for ALIREF list of alias references
     !     HPADDF   for ALLDDF distinct dofs in read conf
     !     HPFLCI   for FLACTI flagging active atoms
     !     HPFLIN   for FLINIT initialization of atom positions
     !     HPDOIN   for DOINIT initialization of atom positions
     !     HPFLINR  for FLINITR initialization of atom positions
     !     HPDOINR  for DOINITR initialization of atom positions
     !     HPINVL   for INIVAL initial values of dofs
     !     HPSSNM   for SSNAME input subspace numbers
     !     HPQICBF  for ZQICBF ic build forward flags
     !     HPDISAB1 for DISATBEG1 distance atom lo base array 
     !     HPDISAE1 for DISATEND1 distance atom hi base array
     !     HPDISAB2 for DISATBEG2 distance atom lo base array
     !     HPDISAE2 for DISATEND2 distance atom hi base array
     !     HPDIATL  for DISTATL distance atom list
     !     HPLDDIS  for LDDISLS loaded distance list
     !     HPDISGAR for DISTGTAR lower distance limit array 
     !     HPDISLAR for DISTLTAR upper distance limit array
     !     HPDFCGAR for DOFCGAR lower ic cons value lim array
     !     HPDFCLAR for DOFCLAR upper ic cons value lim array
     !     HPDFCMP  for DOFCMAP map of dof constraints
     !     HPLDDFC  for LDDOFCON array of loaded ic constraints
     !        -------------
   if(allocated(HPMDFV_hv)) call chmdealloc('zerom1.src','ZEROM','HPMDFV_hv',NVALME,crl=HPMDFV_hv) 
   if(allocated(HPLODF_hv)) call chmdealloc('zerom1.src','ZEROM','HPLODF_hv',NCNFME,intg=HPLODF_hv) 
   if(allocated(HPHIDF_hv)) call chmdealloc('zerom1.src','ZEROM','HPHIDF_hv',NCNFME,intg=HPHIDF_hv) 
! uncompressed arrays
!   if(allocated(LODOFLW)) call chmdealloc('zerom1.src','ZEROM','LODOFLW',NCNFME,intg=LODOFLW) 
!   if(allocated(HIDOFLW)) call chmdealloc('zerom1.src','ZEROM','HIDOFLW',NCNFME,intg=HIDOFLW) 
!   if(allocated(LOCONFW)) call chmdealloc('zerom1.src','ZEROM','LOCONFW',NSUBME,intg=LOCONFW) 
!   if(allocated(HICONFW)) call chmdealloc('zerom1.src','ZEROM','HICONFW',NSUBME,intg=HICONFW) 
!   if(allocated(ZENERGY)) call chmdealloc('zerom1.src','ZEROM','ZENERGY',NCNFME,crl=ZENERGY) 
!
   if(allocated(HPLOCN_hv)) call chmdealloc('zerom1.src','ZEROM','HPLOCN_hv',NSUBME,intg=HPLOCN_hv) 
   if(allocated(HPHICN_hv)) call chmdealloc('zerom1.src','ZEROM','HPHICN_hv',NSUBME,intg=HPHICN_hv) 
   if(allocated(HPLOIE_hv)) call chmdealloc('zerom1.src','ZEROM','HPLOIE_hv',DDEFME,intg=HPLOIE_hv) 
   if(allocated(HPHIIE_hv)) call chmdealloc('zerom1.src','ZEROM','HPHIIE_hv',DDEFME,intg=HPHIIE_hv) 
   if(allocated(HPGMNV_hv)) call chmdealloc('zerom1.src','ZEROM','HPGMNV_hv',DDEFME,crl=HPGMNV_hv)  
   if(allocated(HPLDSMN_hv)) call chmdealloc('zerom1.src','ZEROM','HPLDSMN_hv',DDEFME,crl=HPLDSMN_hv) 
   if(allocated(HPLNLS_hv)) call chmdealloc('zerom1.src','ZEROM','HPLNLS_hv',4*DDEFME,intg=HPLNLS_hv) 
   if(allocated(HPCLST_hv)) call chmdealloc('zerom1.src','ZEROM','HPCLST_hv',4*DDEFME,intg=HPCLST_hv) 
   if(allocated(HPSATB_hv)) call chmdealloc('zerom1.src','ZEROM','HPSATB_hv',NSUBME,intg=HPSATB_hv) 
   if(allocated(HPSATE_hv)) call chmdealloc('zerom1.src','ZEROM','HPSATE_hv',NSUBME,intg=HPSATE_hv) 
   if(allocated(HPSALS_hv)) call chmdealloc('zerom1.src','ZEROM','HPSALS_hv',NSATME,intg=HPSALS_hv) 
   if(allocated(HPSIAB_hv)) call chmdealloc('zerom1.src','ZEROM','HPSIAB_hv',NSUBME,intg=HPSIAB_hv) 
   if(allocated(HPSIAE_hv)) call chmdealloc('zerom1.src','ZEROM','HPSIAE_hv',NSUBME,intg=HPSIAE_hv) 
   if(allocated(HPINIL_hv)) call chmdealloc('zerom1.src','ZEROM','HPINIL_hv',NSATME,intg=HPINIL_hv) 
   if(allocated(HPLDSS_hv)) call chmdealloc('zerom1.src','ZEROM','HPLDSS_hv',NSUBME,intg=HPLDSS_hv) 
   if(allocated(HPMINSS_hv)) call chmdealloc('zerom1.src','ZEROM','HPMINSS_hv',NSUBME,intg=HPMINSS_hv) 
   if(allocated(HPIMISS_hv)) call chmdealloc('zerom1.src','ZEROM','HPIMISS_hv',NSUBME,intg=HPIMISS_hv) 
   if(allocated(HPALIA_hv)) call chmdealloc('zerom1.src','ZEROM','HPALIA_hv',NSUBME,intg=HPALIA_hv) 
   if(allocated(HPALRF_hv)) call chmdealloc('zerom1.src','ZEROM','HPALRF_hv',NSUBME,intg=HPALRF_hv) 
   if(allocated(HPADDF_hv)) call chmdealloc('zerom1.src','ZEROM','HPADDF_hv',DDEFME,intg=HPADDF_hv) 
   if(allocated(HPFLCI_hv)) call chmdealloc('zerom1.src','ZEROM','HPFLCI_hv',NATOM,intg=HPFLCI_hv) 
   if(allocated(HPFLIN_hv)) call chmdealloc('zerom1.src','ZEROM','HPFLIN_hv',NATOM,intg=HPFLIN_hv) 
   if(allocated(HPDOIN_hv)) call chmdealloc('zerom1.src','ZEROM','HPDOIN_hv',NATOM,intg=HPDOIN_hv) 
   if(allocated(HPFLINR_hv)) call chmdealloc('zerom1.src','ZEROM','HPFLINR_hv',NATOM,intg=HPFLINR_hv) 
   if(allocated(HPDOINR_hv)) call chmdealloc('zerom1.src','ZEROM','HPDOINR_hv',NATOM,intg=HPDOINR_hv) 
   if(allocated(HPINVL_hv)) call chmdealloc('zerom1.src','ZEROM','HPINVL_hv',DDEFME,crl=HPINVL_hv) 
   if(allocated(HPSSNM_hv)) call chmdealloc('zerom1.src','ZEROM','HPSSNM_hv',NSUBME,intg=HPSSNM_hv) 
   if(allocated(HPQICBF_hv)) call chmdealloc('zerom1.src','ZEROM','HPQICBF_hv',NSUBME,intg=HPQICBF_hv) 
   if(allocated(HPDISAB1_hv)) call chmdealloc('zerom1.src','ZEROM','HPDISAB1_hv',NDISTME,intg=HPDISAB1_hv) 
   if(allocated(HPDISAE1_hv)) call chmdealloc('zerom1.src','ZEROM','HPDISAE1_hv',NDISTME,intg=HPDISAE1_hv) 
   if(allocated(HPDISAB2_hv)) call chmdealloc('zerom1.src','ZEROM','HPDISAB2_hv',NDISTME,intg=HPDISAB2_hv) 
   if(allocated(HPDISAE2_hv)) call chmdealloc('zerom1.src','ZEROM','HPDISAE2_hv',NDISTME,intg=HPDISAE2_hv) 
   if(allocated(HPDIATL_hv)) call chmdealloc('zerom1.src','ZEROM','HPDIATL_hv',NDISATME,intg=HPDIATL_hv) 
   if(allocated(HPLDDIS_hv)) call chmdealloc('zerom1.src','ZEROM','HPLDDIS_hv',NDISTME,intg=HPLDDIS_hv) 
   if(allocated(HPDISGAR_hv)) call chmdealloc('zerom1.src','ZEROM','HPDISGAR_hv',NDISTME,crl=HPDISGAR_hv) 
   if(allocated(HPDISLAR_hv)) call chmdealloc('zerom1.src','ZEROM','HPDISLAR_hv',NDISTME,crl=HPDISLAR_hv) 
   if(allocated(HPDFCGAR_hv)) call chmdealloc('zerom1.src','ZEROM','HPDFCGAR_hv',NDOFCME,crl=HPDFCGAR_hv) 
   if(allocated(HPDFCLAR_hv)) call chmdealloc('zerom1.src','ZEROM','HPDFCLAR_hv',NDOFCME,crl=HPDFCLAR_hv) 
   if(allocated(HPDFCMP_hv)) call chmdealloc('zerom1.src','ZEROM','HPDFCMP_hv',NDOFCME,intg=HPDFCMP_hv) 
   if(allocated(HPLDDFC_hv)) call chmdealloc('zerom1.src','ZEROM','HPLDDFC_hv',NDOFCME,intg=HPLDDFC_hv) 
 
     call chmalloc('zerom1.src','ZEROM','HPMDFL_hv',NVALME,intg=HPMDFL_hv)
     call chmalloc('zerom1.src','ZEROM','HPMDFV_hv',NVALME,crl=HPMDFV_hv)
     call chmalloc('zerom1.src','ZEROM','HPLODF_hv',NCNFME,intg=HPLODF_hv)
     call chmalloc('zerom1.src','ZEROM','HPHIDF_hv',NCNFME,intg=HPHIDF_hv)
! uncompressed arrays
      call cs_alloc(CSR,NSUBME,NCNFME,NVALME,'zerom1.src','ZEROM')
!
     call chmalloc('zerom1.src','ZEROM','HPLOCN_hv',NSUBME,intg=HPLOCN_hv)
     call chmalloc('zerom1.src','ZEROM','HPHICN_hv',NSUBME,intg=HPHICN_hv)
     call chmalloc('zerom1.src','ZEROM','HPLOIE_hv',DDEFME,intg=HPLOIE_hv)
     call chmalloc('zerom1.src','ZEROM','HPHIIE_hv',DDEFME,intg=HPHIIE_hv)
     call chmalloc('zerom1.src','ZEROM','HPGMNV_hv',DDEFME,crl=HPGMNV_hv)
     call chmalloc('zerom1.src','ZEROM','HPLDSMN_hv',DDEFME,crl=HPLDSMN_hv)
     call chmalloc('zerom1.src','ZEROM','HPLNLS_hv',4*DDEFME,intg=HPLNLS_hv)
     call chmalloc('zerom1.src','ZEROM','HPCLST_hv',4*DDEFME,intg=HPCLST_hv)
     call chmalloc('zerom1.src','ZEROM','HPSATB_hv',NSUBME,intg=HPSATB_hv)
     call chmalloc('zerom1.src','ZEROM','HPSATE_hv',NSUBME,intg=HPSATE_hv)
     call chmalloc('zerom1.src','ZEROM','HPSALS_hv',NSATME,intg=HPSALS_hv)
     call chmalloc('zerom1.src','ZEROM','HPSIAB_hv',NSUBME,intg=HPSIAB_hv)
     call chmalloc('zerom1.src','ZEROM','HPSIAE_hv',NSUBME,intg=HPSIAE_hv)
     call chmalloc('zerom1.src','ZEROM','HPINIL_hv',NSATME,intg=HPINIL_hv)
     call chmalloc('zerom1.src','ZEROM','HPLDSS_hv',NSUBME,intg=HPLDSS_hv)
     call chmalloc('zerom1.src','ZEROM','HPMINSS_hv',NSUBME,intg=HPMINSS_hv)
     call chmalloc('zerom1.src','ZEROM','HPIMISS_hv',NSUBME,intg=HPIMISS_hv)
     call chmalloc('zerom1.src','ZEROM','HPALIA_hv',NSUBME,intg=HPALIA_hv)
     call chmalloc('zerom1.src','ZEROM','HPALRF_hv',NSUBME,intg=HPALRF_hv)
     call chmalloc('zerom1.src','ZEROM','HPADDF_hv',DDEFME,intg=HPADDF_hv)
     call chmalloc('zerom1.src','ZEROM','HPFLCI_hv',NATOM,intg=HPFLCI_hv)
     call chmalloc('zerom1.src','ZEROM','HPFLIN_hv',NATOM,intg=HPFLIN_hv)
     call chmalloc('zerom1.src','ZEROM','HPDOIN_hv',NATOM,intg=HPDOIN_hv)
     call chmalloc('zerom1.src','ZEROM','HPFLINR_hv',NATOM,intg=HPFLINR_hv)
     call chmalloc('zerom1.src','ZEROM','HPDOINR_hv',NATOM,intg=HPDOINR_hv)
     call chmalloc('zerom1.src','ZEROM','HPINVL_hv',DDEFME,crl=HPINVL_hv)
     call chmalloc('zerom1.src','ZEROM','HPSSNM_hv',NSUBME,intg=HPSSNM_hv)
     call chmalloc('zerom1.src','ZEROM','HPQICBF_hv',NSUBME,intg=HPQICBF_hv)
     call chmalloc('zerom1.src','ZEROM','HPDISAB1_hv',NDISTME,intg=HPDISAB1_hv)
     call chmalloc('zerom1.src','ZEROM','HPDISAE1_hv',NDISTME,intg=HPDISAE1_hv)
     call chmalloc('zerom1.src','ZEROM','HPDISAB2_hv',NDISTME,intg=HPDISAB2_hv)
     call chmalloc('zerom1.src','ZEROM','HPDISAE2_hv',NDISTME,intg=HPDISAE2_hv)
     call chmalloc('zerom1.src','ZEROM','HPDIATL_hv',NDISATME,intg=HPDIATL_hv)
     call chmalloc('zerom1.src','ZEROM','HPLDDIS_hv',NDISTME,intg=HPLDDIS_hv)
     call chmalloc('zerom1.src','ZEROM','HPDISGAR_hv',NDISTME,crl=HPDISGAR_hv)
     call chmalloc('zerom1.src','ZEROM','HPDISLAR_hv',NDISTME,crl=HPDISLAR_hv)
     call chmalloc('zerom1.src','ZEROM','HPDFCGAR_hv',NDOFCME,crl=HPDFCGAR_hv)
     call chmalloc('zerom1.src','ZEROM','HPDFCLAR_hv',NDOFCME,crl=HPDFCLAR_hv)
     call chmalloc('zerom1.src','ZEROM','HPDFCMP_hv',NDOFCME,intg=HPDFCMP_hv)
     call chmalloc('zerom1.src','ZEROM','HPLDDFC_hv',NDOFCME,intg=HPLDDFC_hv)
     !
#if KEY_PARALLEL==1
    if(MYNODGP.eq.1) then  
#endif
     WRITE(OUTU,'(6X,A47)')  &
          'Z MODULE ALLOCATED MEMORY FOR A MAXIMUM OF: '
     WRITE(OUTU,'(6X,I14,1x,A15)') DDEFME,'dof definitions'
     WRITE(OUTU,'(6X,I14,1x,A10)') NVALME,'dof values'
     WRITE(OUTU,'(6X,I14,1x,A10)') NCNFME,'conformers'
     WRITE(OUTU,'(6X,I14,1x,A24)') NSUBME,'substructures containing'
     WRITE(OUTU,'(6X,I14,1x,A11)') NSATME,'total atoms'
     WRITE(OUTU,'(6X,I14,1x,A26)') NDISTME,'total distance constraints'
     WRITE(OUTU,'(6X,I14,1x,A31)') NDISATME, &
          'total distance constraint atoms' 
     WRITE(OUTU,'(6X,I14,1x,A26)') NDOFCME,'total dof constraints'
     !
#if KEY_PARALLEL==1
    endif  
#endif
     call ZINITARR(HPMDFL_hv,HPMDFV_hv,HPLODF_hv,HPHIDF_hv,HPLOCN_hv, &
          HPHICN_hv,HPLOIE_hv,HPHIIE_hv,HPLNLS_hv,HPCLST_hv,HPSATB_hv, &
          HPSATE_hv,HPSALS_hv,HPSIAB_hv,HPSIAE_hv,HPINIL_hv,HPLDSS_hv, &
          HPALIA_hv,HPALRF_hv,HPADDF_hv,HPFLCI_hv,HPFLIN_hv,HPDOIN_hv, &
          HPFLINR_hv,HPDOINR_hv,HPINVL_hv,HPSSNM_hv,HPGMNV_hv,HPLDSMN_hv, &
          HPIMISS_hv,HPMINSS_hv,HPQICBF_hv,HPDISAB1_hv,HPDISAE1_hv, &
          HPDISAB2_hv,HPDISAE2_hv,HPDIATL_hv,HPLDDIS_hv,HPDISGAR_hv, &
          HPDISLAR_hv,HPDFCGAR_hv,HPDFCLAR_hv,HPDFCMP_hv,HPLDDFC_hv)
     ! for distances
     !     initialize counters
     NTHDDF = 0 !number of dof definitions
     NSUBDE = 0 !number of substructure definitions 
     NDISTDE = 0 !number of distance constraints
     NDOFCON = 0 !number of internal coordinate constraints
     NALIAS = 0!number of aliases
     NSUBSP = 0 !number of subspaces read 
     NMASTR = 0 !length of master dof list
     NALLDF = 0 !number of dof's read
     NMINORSS = 0!number of minor subspaces
     NZBINS = 0 !number of energy bins
     !
     QZEROMEM = .TRUE.
     !
     !.......................................................................
  ELSE IF (WRD.EQ.'ZCAT') THEN
     QMINOR = .FALSE.
     IF(INDXA(COMLYN,COMLEN,'MINO').GT.0) THEN
        QMINOR = .TRUE.
        call CATEGMINORSS(COMLYN,COMLEN,HPMINSS_hv,HPIMISS_hv,HPALIA_hv)
     ENDIF
     ! .......................................................................
  ELSE IF(WRD.EQ.'ZDEF') THEN
     IF(.NOT.QZEROMEM)  &
          CALL WRNDIE(-5,'<ZEROM>', &
          'NO MEMORY ALLOCATED ')
     DEFCNT = 0
     DDEFNB = GTRMI(COMLYN,COMLEN,'DOFR',-1)
     IF (DDEFNB.GT.0) DEFCNT = DEFCNT +1 
     DSUBSP = GTRMI(COMLYN,COMLEN,'SUBS',-1) 
     IF (DSUBSP.GT.0) DEFCNT = DEFCNT +1
     DDEFDIS = GTRMI(COMLYN,COMLEN,'DIST',-1)
     IF (DDEFDIS.GT.0) DEFCNT = DEFCNT +1
     DDEFDCON = GTRMI(COMLYN,COMLEN,'DFCO',-1)
     IF (DDEFDCON.GT.0) DEFCNT = DEFCNT +1
     !       IF (((DDEFNB.NE.-1).AND.(DSUBSP.NE.-1)) .OR. 
     !     & ((DDEFNB.NE.-1).AND.(DDEFDIS.NE.-1)) .OR. 
     !     & ((DDEFDIS.NE.-1).AND.(DSUBSP.NE.-1)))  THEN
     !         CALL WRNDIE(-5,'<ZEROM>',
     !     &  'ONLY ONE DEFINITION MAY BE SPECFD PER LINE')
     IF (DEFCNT.GT.1)  &
          CALL WRNDIE(-5,'<ZEROM>', &
          'ONLY ONE DEFINITION MAY BE SPECFD PER LINE')
     ! .......
     ! process degree-of-freedom definitions
     ! .......
#if KEY_PARALLEL==1
    if(MYNODGP.eq.1) then
#endif
     if (DDEFNB.NE.-1) then
        if (DDEFNB.NE.(NTHDDF+1))  &
             call WRNDIE(-5,'<ZEROM>', &
             'DOF DEFINITION NUMBERS OUT OF ORDER')
        !       
        !
        CALL ADDDOFDEF(COMLYN,COMLEN,HPLOIE_hv, &
             HPHIIE_hv,HPLNLS_hv,HPCLST_hv, &
             HPINVL_hv,icr_struct%lenic,&
             icr_struct%B1ic,icr_struct%B2ic, &
             icr_struct%T1ic,icr_struct%T2ic, &
             icr_struct%PIC, icr_struct%IAR, &
             icr_struct%JAR, icr_struct%KAR, &
             icr_struct%LAR, icr_struct%TAR)
        if(NICELST.GT.0) QDOFDE = .TRUE.
        if(NICELST.GT.(4*DDEFME))  &
             call WRNDIE(-5,'<ZEROM>', &
             'IC TABLE ENTRIES EXCEED ALLOCATED MEMORY (ZMEM NDOF)')
        IF(NTHDDF.GT.DDEFME)  &
             CALL WRNDIE(-5,'<ZEROM>', &
             'IC TABLE ENTRIES EXCEED ALLOCATED MEMORY (ZMEM NDOF)') 
        ! .......
        ! process substructure definitions
        ! .......
     ELSE IF (DSUBSP.NE.-1) THEN 
        ALIASN = GTRMI(COMLYN,COMLEN,'ALIA',-1)
        IF (DSUBSP.NE.(NSUBDE+1))  &
             CALL WRNDIE(-5,'<ZEROM>', &
             'SUBSTRUC DEFINITION NUMBERS OUT OF ORDER')
        IF(ALIASN.EQ.-1) THEN
           ! parse the atom selections for substructure definition
           ! and initialization
           call chmalloc('zerom1.src','ZEROM','ISLCT_sv',NATOM,intg=ISLCT_sv)
           call chmalloc('zerom1.src','ZEROM','JSLCT_sv',NATOM,intg=JSLCT_sv)
           !     create the i and j atom lists for rmsd calculations
           call SELCTD(COMLYN,COMLEN,ISLCT_sv,JSLCT_sv,X,Y,Z,WMAIN,.TRUE.,ERR)
           ! note that mode=0 and dflt=0 below
           !
           QLOCREV = .FALSE.
           IF (INDXA(COMLYN,COMLEN,'REVE').GT.0) QLOCREV = .TRUE.
           call chmalloc('zerom1.src','ZEROM','HPTALI_hv',NATOM,intg=HPTALI_hv)
           call chmalloc('zerom1.src','ZEROM','HPTIND_hv',NATOM,intg=HPTIND_hv)
           call chmalloc('zerom1.src','ZEROM','HPSALI_hv',NATOM,intg=HPSALI_hv)
           call chmalloc('zerom1.src','ZEROM','HPSCUR_hv',NATOM,intg=HPSCUR_hv)
           HPTALI_hv = 0
           HPTIND_hv = 0
           HPSALI_hv = 0
           HPSCUR_hv = 0
           !
           call ADDSUBSTR(ISLCT_sv,JSLCT_sv,NATOM,HPSATB_hv,HPSATE_hv,HPSALS_hv, &
                HPSIAB_hv,HPSIAE_hv,HPINIL_hv,HPTALI_hv,HPTIND_hv,HPSALI_hv, &
                HPSCUR_hv,HPQICBF_hv,QLOCREV)
           call chmdealloc('zerom1.src','ZEROM','ISLCT_sv',NATOM,intg=ISLCT_sv)
           call chmdealloc('zerom1.src','ZEROM','JSLCT_sv',NATOM,intg=JSLCT_sv)

           call chmdealloc('zerom1.src','ZEROM','HPTALI_hv',NATOM,intg=HPTALI_hv)
           call chmdealloc('zerom1.src','ZEROM','HPSALI_hv',NATOM,intg=HPSALI_hv)
           call chmdealloc('zerom1.src','ZEROM','HPSCUR_hv',NATOM,intg=HPSCUR_hv)
           call chmdealloc('zerom1.src','ZEROM','HPTIND_hv',NATOM,intg=HPTIND_hv)
        ELSE IF (ALIASN.GT.0) THEN !if there is an alias 
           IF(INDXA(COMLYN,COMLEN,'SELE').GT.0)  &
                CALL WRNDIE(-5,'<ZEROM>', &
                'CANNOT BOTH DEFINE AND ALIAS A SUBSTRUCTURE')
           IF(INDXA(COMLYN,COMLEN,'REVE').GT.0) THEN
              WRITE(OUTU,'(A)')  &
                   'REVE KEYWRD INCOMPATIBLE WITH ALIASING'
              CALL WRNDIE(-5,'<ZEROM>', &
                   'CANNOT BOTH DEFINE AND ALIAS A SUBSTRUCTURE')
           ENDIF
           call ALIASASGN(HPSATB_hv,HPSATE_hv,HPALIA_hv,ALIASN,HPALRF_hv, &
                HPSIAB_hv,HPSIAE_hv,HPQICBF_hv)
        ELSE 
           CALL WRNDIE(-5,'<ZEROM>','BAD ALIAS NUMBER')
        ENDIF
        !********************************************************
        ! parse distance constraints
     ELSE IF (DDEFDIS.NE.-1) THEN
        IF (DDEFDIS.NE.(NDISTDE+1)) &
             CALL WRNDIE(-5,'<ZEROM>', &
             'DISTANCE DEFINITION NUMBERS OUT OF ORDER')
        !          WRITE(6,*) 'BEFORE, DISTGT is ',DISTGT
        MINUS1 = -1.0
        DISTGT = GTRMF(COMLYN,COMLEN,'GTHA',MINUS1) 
        DISTLT = GTRMF(COMLYN,COMLEN,'LTHA',MINUS1)
        IF(DISTLT.EQ.MINUS1) THEN
           WRITE(OUTU,'(A)')  &
                '   UPPER BOUND MUST BE SPECIFIED, USE LTHA [real]'
           CALL WRNDIE(-5,'<ZEROM>', &
                'MISSING UPPER DISTANCE CONSTRAINT BOUND')
        ENDIF
        IF(DISTGT.EQ.MINUS1) THEN
           WRITE(OUTU,'(A)') &
                '   LOWER BOUND MUST BE SPECIFIED, USE GTHA [real]'
           CALL WRNDIE(-5,'<ZEROM>', &
                'MISSING LOWER DISTANCE CONSTRAINT BOUND')
        ENDIF
        IF(DISTLT.LT.DISTGT) THEN
           WRITE(OUTU,'(A,I8,A)') '   FOR DISTANCE CONSTRAINT ', &
                NDISTDE+1,' UPPER BOUND IS LOWER THAN HIGHER BOUND'
           CALL WRNDIE(-5,'<ZEROM>', &
                'BAD DISTANCE CONSTRAINT BOUNDS')
        ENDIF
        call chmalloc('zerom1.src','ZEROM','ISLCT_sv',NATOM,intg=ISLCT_sv)
        call chmalloc('zerom1.src','ZEROM','JSLCT_sv',NATOM,intg=JSLCT_sv)
        !     create the i and j atom lists for distance calculations
        call SELCTD(COMLYN,COMLEN,ISLCT_sv,JSLCT_sv,X,Y,Z,WMAIN,.TRUE.,ERR)
        !          
        call ADDDISTCONS(ISLCT_sv,JSLCT_sv,NATOM,HPDISAB1_hv,HPDISAE1_hv, &
             HPDIATL_hv,HPDISAB2_hv,HPDISAE2_hv,DISTGT,DISTLT,HPDISGAR_hv, &
             HPDISLAR_hv)
        call chmdealloc('zerom1.src','ZEROM','ISLCT_sv',NATOM,intg=ISLCT_sv)
        call chmdealloc('zerom1.src','ZEROM','JSLCT_sv',NATOM,intg=JSLCT_sv)

        !**************************************************************
        ! parse dof (internal coordinate) constraints
        !
     ELSE IF (DDEFDCON.NE.-1) THEN
        IF (DDEFDCON.NE.(NDOFCON+1)) &
             CALL WRNDIE(-5,'<ZEROM>', &
             'IC CONSTRAINT DEFINITION NUMBERS OUT OF ORDER')
        !          WRITE(6,*) 'BEFORE, DISTGT is ',DISTGT
        MINUS1 = -1.0
        LOCDOF = GTRMI(COMLYN,COMLEN,'DOF',-1)
        DISTGT = GTRMF(COMLYN,COMLEN,'GTHA',MINUS1)
        DISTLT = GTRMF(COMLYN,COMLEN,'LTHA',MINUS1)
        IF(DISTLT.EQ.MINUS1) THEN
           WRITE(OUTU,'(A)') &
                '   UPPER BOUND MUST BE SPECIFIED, USE LTHA [real]'
           CALL WRNDIE(-5,'<ZEROM>', &
                'MISSING UPPER DISTANCE CONSTRAINT BOUND')
        ENDIF
        IF(DISTGT.EQ.MINUS1) THEN
           WRITE(OUTU,'(A)') &
                '   LOWER BOUND MUST BE SPECIFIED, USE GTHA [real]'
           CALL WRNDIE(-5,'<ZEROM>', &
                'MISSING LOWER DISTANCE CONSTRAINT BOUND')
        ENDIF
        IF(DISTLT.LT.DISTGT) THEN
           WRITE(OUTU,'(A,I8,A)') '   FOR DISTANCE CONSTRAINT ', &
                NDOFCON+1,' UPPER BOUND IS LOWER THAN HIGHER BOUND'
           CALL WRNDIE(-5,'<ZEROM>', &
                'BAD IC CONSTRAINT BOUNDS')
        ENDIF
        IF(DISTLT.EQ.DISTGT) THEN
           WRITE(OUTU,'(A,I8,A)') '   FOR DISTANCE CONSTRAINT ', &
                NDOFCON+1,' UPPER AND LOWER BOUNDS ARE EQUAL'
           CALL WRNDIE(-5,'<ZEROM>', &
                'BAD IC CONSTRAINT BOUNDS')
        ENDIF
        IF((DISTGT.LT.-360).OR.(DISTLT.GT.360)) THEN
           WRITE(OUTU,'(3X,A)')  &
                ' BOUNDS FOR IC CONSTRAINTS MUST BE WITHIN [-360,360] '
           CALL WRNDIE(-5,'<ZEROM>', &
                'BAD IC CONSTRAINT BOUNDS')
        ENDIF
        IF((DISTLT-DISTGT).GT.360) THEN
           WRITE(OUTU,'(3X,A)') &
                ' BOUNDS FOR IC CONSTRAINTS MUST BE WITHIN 360 DEGREES'
           CALL WRNDIE(-5,'<ZEROM>', &
                'BAD IC CONSTRAINT BOUNDS')
        ENDIF
        ! transform bounds to be within [0,720]
        IF(DISTGT.LT.0) THEN
           DISTGT = DISTGT + 360
           DISTLT = DISTLT + 360
        ENDIF
        !
        call ADDDOFCONS(LOCDOF,HPDFCGAR_hv,HPDFCLAR_hv,DISTGT,DISTLT, &
             HPDFCMP_hv)
     endif
#if KEY_PARALLEL==1
    endif 
#endif
     QZDATCH = .true.  !changed data
     !.......................................................................
  else if(WRD.EQ.'ZLOA') then
   if(QZDATCH) call parstoperr('ZEROM','DATA NOT SET. USE ZSET')
     !
!#if KEY_PARALLEL==1
!     if(QZCOMM) call COMM_ZDATA   
!#endif 
     if(.NOT.QZEROMEM) &
          call WRNDIE(-5,'<ZEROM>', &
          'NO MEMORY ALLOCATED ')
     QSUBSA=(INDXA(COMLYN,COMLEN,'SUBS').GT.0)
     QDISTA=(INDXA(COMLYN,COMLEN,'DIST').GT.0)
     QICCNA=(INDXA(COMLYN,COMLEN,'DFCO').GT.0)
     IF((QSUBSA.AND.QDISTA).OR.(QSUBSA.AND.QICCNA).OR. &
          (QDISTA.AND.QICCNA))  &
          CALL WRNDIE(-5,'<ZEROM>', &
          'USE EITHER DIST, SUBS, OR DFCO IN ONE ZLOAD COMMAND')
     IF(QSUBSA) THEN
        QSUBNUL = .FALSE.
        IF (INDXA(COMLYN,COMLEN,'NULL').GT.0) QSUBNUL = .TRUE.
        !       WRITE(6,*) 'QSUBNUL is ',QSUBNUL
        call SUBSPACELOAD(COMLYN,COMLEN,HPLDSS_hv,HPALIA_hv)
        LDSSMNE = 1.0E+25  !stores minimum energy between zload commands
     ELSE IF (QDISTA) THEN
        call DISTCONSLOAD(COMLYN,COMLEN,HPLDDIS_hv)
     ELSE IF (QICCNA) THEN
        call DOFCONSLOAD(COMLYN,COMLEN,HPLDDFC_hv)
     ENDIF
     !.......................................................................
  ELSE IF(WRD.EQ.'ZSPE') THEN
     ! 
     call ZSPECIFICS(HPLOIE_hv,HPHIIE_hv,HPLNLS_hv,HPCLST_hv,HPSATB_hv, &
          HPSATE_hv,HPLDSS_hv,HPSALS_hv,HPALIA_hv,0,HPLOCN_hv,HPHICN_hv, &
          HPLODF_hv,HPHIDF_hv,HPMDFL_hv,HPMDFV_hv,HPMINSS_hv,HPDISAB1_hv, &
          HPDISAE1_hv,HPDISAB2_hv,HPDISAE2_hv,HPDIATL_hv,HPLDDIS_hv, &
          HPDISGAR_hv,HPDISLAR_hv,HPDFCGAR_hv,HPDFCLAR_hv,HPDFCMP_hv, &
          HPLDDFC_hv)
     ! for distances
     !.......................................................................
  ELSE IF(WRD.EQ.'ZREA') THEN
     IF(.NOT.QZEROMEM) &
          CALL WRNDIE(-5,'<ZEROM>', &
          'NO MEMORY ALLOCATED ')
     IF (INDXA(COMLYN,COMLEN,'SSDI').GT.0) THEN
        QSSDIS=.TRUE.
        WRITE(6,*) 'ALLOWING DISORDER IN SUBSPACE NUMBERS'
     ELSE IF (QSSDIS) THEN 
        WRITE(6,*)  &
             'ONCE SSDI INVOKED, MUST BE USED UNTIL ZERO MOD IS RESET'
        CALL WRNDIE(-5,'<ZEROM>', &
             'SSDI FLAG CANNOT GO TRUE->FALSE')
     ENDIF
     QMISSG = .FALSE.
     IF(INDXA(COMLYN,COMLEN,'MISS').GT.0) THEN
        WRITE(6,*) 'ALLOWING GAPS IN CONF NUMBERS '
        QMISSG = .TRUE.
     ENDIF
     QREDUN = .FALSE.
     IF(INDXA(COMLYN,COMLEN,'REDU').GT.0) THEN
        WRITE(6,*) 'ALLOWING A DOF TO APPEAR IN >1 SUBSPACE '
        QREDUN = .TRUE.
     ENDIF
     IF (INDXA(COMLYN,COMLEN,'CONF').GT.0) THEN
        RDZCUN = GTRMI(COMLYN,COMLEN,'RUNI',-1) 
        WRZCUN = GTRMI(COMLYN,COMLEN,'WUNI',-1)
        if (MXCNSZ.LE.0) then
#if KEY_PARALLEL==1
          if(MYNODGP.eq.1) then  
#endif
           WRITE(OUTU,'(2X,A37)')  &
                'WARNING:  NO CONFORMER SIZE SPECIFIED'
           WRITE(OUTU,'(2X,A40)')  &
                'SETTING EQUAL TO DOF DEFINITION ESTIMATE' 
#if KEY_PARALLEL==1
          endif 
#endif
          MXCNSZ = DDEFME
        endif
        if(RDZCUN.LE.0) call WRNDIE(-5,'<ZEROM>', &
             'NO UNIT NUMBER SPECIFIED FOR READING CONFRMR FILE')
        !
        call chmalloc('zerom1.src','ZEROM','HPTMPD_hv',MXCNSZ,intg=HPTMPD_hv)
        call chmalloc('zerom1.src','ZEROM','HPTMPV_hv',MXCNSZ,crl=HPTMPV_hv)
        call chmalloc('zerom1.src','ZEROM','HPCMPD_hv',MXCNSZ,intg=HPCMPD_hv)
        call chmalloc('zerom1.src','ZEROM','HPCMPV_hv',MXCNSZ,crl=HPCMPV_hv)
        call chmalloc('zerom1.src','ZEROM','HPSDOF_hv',MXCNSZ,intg=HPSDOF_hv)
        call chmalloc('zerom1.src','ZEROM','HPSVAL_hv',MXCNSZ,crl=HPSVAL_hv)
        call chmalloc('zerom1.src','ZEROM','HPSTMD_hv',MXCNSZ,intg=HPSTMD_hv)
        call chmalloc('zerom1.src','ZEROM','HPFSTC_hv',NSUBME,intg=HPFSTC_hv)
        HPTMPD_hv = 0
        HPTMPV_hv = 0
        HPCMPD_hv = 0
        HPCMPV_hv = 0
        HPSDOF_hv = 0
        HPSVAL_hv = 0
        HPSTMD_hv = 0
        HPFSTC_hv = 0
        !
        call RDZCONF(HPFSTC_hv,HPTMPD_hv,HPTMPV_hv,HPCMPD_hv,HPCMPV_hv, &
             HPSDOF_hv,HPSVAL_hv,HPSTMD_hv,HPADDF_hv,HPLODF_hv,HPHIDF_hv, &
             HPLOCN_hv,HPHICN_hv,HPMDFL_hv,HPMDFV_hv,RDZCUN,WRZCUN,HPSSNM_hv, &
             HPALIA_hv)
        !
        call chmdealloc('zerom1.src','ZEROM','HPTMPD_hv',MXCNSZ,intg=HPTMPD_hv)
        call chmdealloc('zerom1.src','ZEROM','HPTMPV_hv',MXCNSZ,crl=HPTMPV_hv)
        call chmdealloc('zerom1.src','ZEROM','HPCMPD_hv',MXCNSZ,intg=HPCMPD_hv)
        call chmdealloc('zerom1.src','ZEROM','HPCMPV_hv',MXCNSZ,crl=HPCMPV_hv)
        call chmdealloc('zerom1.src','ZEROM','HPSDOF_hv',MXCNSZ,intg=HPSDOF_hv)
        call chmdealloc('zerom1.src','ZEROM','HPSVAL_hv',MXCNSZ,crl=HPSVAL_hv)
        call chmdealloc('zerom1.src','ZEROM','HPSTMD_hv',MXCNSZ,intg=HPSTMD_hv)
        call chmdealloc('zerom1.src','ZEROM','HPFSTC_hv',NSUBME,intg=HPFSTC_hv)
     endif
     QZDATCH = .true.
     !.......................................................................C
  ELSE IF (WRD.EQ.'ZSEA') THEN
     !
     ZSTEPN = 0
     NZBINS = 0 
     NSSTAG = 1 
     !       
     QZNCUT = .FALSE.
     QRECUT = .FALSE.
     QWZCMP = .FALSE. !compression of output 
     QZMINI = .FALSE.
     QZVCUT = .FALSE. 
     QZCFIX = .FALSE. 
     QZRMSD = .FALSE.
     QZORIEN = .FALSE.
     QZ1STMIN = .FALSE.
     !
#if KEY_PARALLEL==1
! make communication arrays if not done
  if(.not.QCPUONES) call cpu_commarr
#endif
  !make the active part of the ic table
     if(INDXA(COMLYN,COMLEN,'MSD').GT.0) then
        QZRMSD=.TRUE.
        QZORIEN = .TRUE.
        IF(INDXA(COMLYN,COMLEN,'NOOR').GT.0) QZORIEN=.FALSE.
     ELSE IF(INDXA(COMLYN,COMLEN,'NOOR').GT.0) THEN
        WRITE(OUTU,*)  &
             'WARNING: ORIENTATION KEYWORD SPECIFIED WITHOUT MSD KEYWORD' 
     endif
     QZTIME = .false.
     if(INDXA(COMLYN,COMLEN,'TIME').GT.0) QZTIME = .true.
     !
     QREDUSS = .FALSE.  !don't allow overlap of minor and major ss's
     IF(INDXA(COMLYN,COMLEN,'RESS').GT.0) QREDUSS=.TRUE.
     !  cons fix non-initialized atoms or not
     IF (INDXA(COMLYN,COMLEN,'CFIX').GT.0) QZCFIX=.TRUE.
     ZPMODE = GTRMI(COMLYN,COMLEN,'PMOD',1)
     ZNTAKE = GTRMI(COMLYN,COMLEN,'TAKE',-1) 
     ZWRUNI = GTRMI(COMLYN,COMLEN,'WRUN',-1)
     ZTMINO = GTRMI(COMLYN,COMLEN,'MINO',0)
     ZWRTUNI = GTRMI(COMLYN,COMLEN,'WDUN',-1) !for trajectories
     Z1STPRN = GTRMI(COMLYN,COMLEN,'1NPR',-1) !print intrvl sd or conj
     ! Z1STPRN not implemented yet
     QZWRTRJ = .FALSE.
     IF(ZWRTUNI.GT.0) QZWRTRJ = .TRUE.
     ZTNSTEP = GTRMI(COMLYN,COMLEN,'NDST',1000) 
     !
     QMINPRNT = .FALSE.
     ZMINUNI = GTRMI(COMLYN,COMLEN,'MWRU',-1)
     IF(ZMINUNI.GT.0) QMINPRNT = .TRUE.
     QZSWRIT = .FALSE.
     IF(ZWRUNI.GT.1) THEN
        QZSWRIT = .TRUE.
     ELSE
        WRITE(OUTU,'(A)')  &
             ' WARNING: NO UNIT SPECIFIED FOR SEARCH OUTPUT'
     ENDIF
     ZSVFRC = GTRMF(COMLYN,COMLEN,'SVFR',ONE) 
     NSSTAG = GTRMI(COMLYN,COMLEN,'TAG',1) 
     IF (INDXA(COMLYN,COMLEN,'WCOM').GT.0) QWZCMP=.TRUE.
     IF(QWZCMP.AND.QZRMSD) THEN
        WRITE(OUTU,*) &
             '   MSD calculation not currently compatible with WCOMpression'
        CALL WRNDIE(-5,'<ZEROM>', &
             'MSD and WCOMp not currently compatible')
     ENDIF
     ! parse minimization

     Z1MSTP = GTRMF(COMLYN,COMLEN,'MNST',PT02)
     !       WRITE(6,*) 'PARSING: Z1MSTP is ',Z1MSTP
     NSTEPSD = GTRMI(COMLYN,COMLEN,'SD',-1)
     IF (NSTEPSD.GT.0) THEN
        Z1STTYP = 0
        QZ1STMIN = .TRUE.
        Z1MNSTP = NSTEPSD
     ENDIF
     NSTEPCG = GTRMI(COMLYN,COMLEN,'CONJ',-1) 
     IF (NSTEPCG.GT.0) THEN
        Z1STTYP = 1 
        QZ1STMIN = .TRUE.
        Z1MNSTP = NSTEPCG
     ENDIF
     IF((NSTEPSD.GT.0).AND.(NSTEPCG.GT.0)) THEN
        CALL WRNDIE(-5,'<ZEROM>', &
             'CANNOT SPECIFY BOTH SD AND CONJ GRAD METHODS')
     ENDIF
     !
     IF(ZNTAKE.GT.NSSSLD) CALL WRNDIE(-5,'<ZEROM>', &
          'CANNOT TAKE MORE SUBSPACES THAN LOADED')
     !
     !       WRITE(6,*) 'ZNTAKE ',ZNTAKE,' ZTMINO ',ZTMINO
     IF(ZTMINO.GT.NSSSLD) CALL WRNDIE(-5,'<ZEROM>', &
          'CANNOT TAKE MORE MINOR SUBSPACES THAN LOADED')
     IF(ZTMINO.GT.ZNTAKE) CALL WRNDIE(-5,'<ZEROM>', &
          'CANNOT TAKE MORE MINOR SUBSPACES THAN SUBSPACES') 
     IF(ZTMINO.GT.NMINORSS) THEN
        WRITE(6,*) 'TAKING ',ZTMINO,' MINOR SSs, but ONLY ', &
             NMINORSS,' SSs CATEGORIZED AS MINOR '
        CALL WRNDIE(-5,'<ZEROM>', &
             'CANNOT TAKE MORE MINOR SUBSPACES THAN CATEGORIZED') 
     ENDIF
     IF (INDXA(COMLYN,COMLEN,'MINA').GT.0)  &
          QZMINI = .TRUE.
     !       WRITE(6,*) 'QZMINI ',QZMINI
     ! parse "noenergy" keyword
     QZENERG = .TRUE.
     IF (INDXA(COMLYN,COMLEN,'NOEN').GT.0) QZENERG = .FALSE.
     !       WRITE(6,*) 'QZENERG KEYWORD IS ',QZENERG
     IF((.NOT.QZENERG).AND.((NSTEPSD.GT.0).OR. &
          (NSTEPCG.GT.0))) THEN
        WRITE(OUTU,*)  &
             'NOENergy keywrd incompatible w/ SD or CONJ keywrds'
        CALL WRNDIE(-5,'<ZEROM>', &
             'INCOMPATIBLE ZSEArch KEYWORDS')
     ENDIF
     !
     IF (INDX(COMLYN,COMLEN,'RECU',4).GT.0) THEN
        QRECUT = .TRUE.
        RECUTF = GTRMF(COMLYN,COMLEN,'RECU',-1.0D0)
     ENDIF
     IF (INDX(COMLYN,COMLEN,'ECUT',4).GT.0) THEN
        ZENCUT = GTRMF(COMLYN,COMLEN,'ECUT',-1.0D0)
        QZNCUT = .TRUE.
     ENDIF
     IF (INDX(COMLYN,COMLEN,'VCUT',4).GT.0) THEN
        ZVATOL = GTRMF(COMLYN,COMLEN,'VCUT',-1.0D0)
        QZVCUT = .TRUE.
     ENDIF
     !       IF((QWZCMP).AND.(QRECUT.OR.QZNCUT)) THEN
     !         CALL WRNDIE(-5,'<ZEROM>',
     !     & 'CANNOT HAVE CUTOFFS IF COMPRESSING OUTPUT')
     !       ENDIF
     IF (ZVATOL.LT.0) CALL WRNDIE(-5,'<ZEROM>', &
          'CANNOT HAVE NEG TOLERANCE [VCUT < 0]')
     IF (INDX(COMLYN,COMLEN,'NBIN',4).GT.0) THEN
        NZBINS = GTRMI(COMLYN,COMLEN,'NBIN',-1)
        ZBINSZ = GTRMF(COMLYN,COMLEN,'BINS',ZERO)
        IF (NZBINS.EQ.-1) CALL WRNDIE(-5,'<ZEROM>', &
             'NO NUMBER OF BINS SPECIFIED (NBINS) ')
        IF (ZBINSZ.EQ.0) CALL WRNDIE(-5,'<ZEROM>', &
             'NO BIN SIZE SPECIFIED (BINS)')
        QZBINS = .TRUE.
        IF (.NOT.QRECUT) CALL WRNDIE(-5,'<ZEROM>', &
             'RECUT MUST BE SPECIFIED IN ORDER TO BIN ENERGIES')
        TEMPF = NZBINS*ZBINSZ
        IF (TEMPF.GT.RECUTF) CALL WRNDIE(-5,'<ZEROM>', &
             'RECUT THRESHOLD TOO LOW FOR SPECIFIED BINS')
     ENDIF
     !       WRITE(6,*) 'ECUT is ',ZENCUT
     !       WRITE(6,*) 'RECU is ',RECUTF
     !       WRITE(6,*) 'QWZCMP is ',QWZCMP
     !       WRITE(6,*) 'QREDUSS is ',QREDUSS
#if KEY_PARALLEL==1     
     if(MYNODGP.eq.1) then 
#endif
      WRITE(OUTU,*) 'A SEARCH HAS BEEN REQUESTED'
#if KEY_PARALLEL==1
     endif 
#endif
     WORD = 'OUT'
     !       WRITE(6,*) 'QZORIEN ',QZORIEN
     if(QZORIEN) WORD = ''
     if(QZRMSD) then
#if KEY_PARALLEL==1
       if(MYNODGP.eq.1) then  
#endif
        WRITE(OUTU,*) 'MSDs will be calculated WITH',WORD, &
             ' orientation'
#if KEY_PARALLEL==1
       endif  
#endif
     endif
     if (.NOT.QZENERG) then
        if (QZNCUT) then
           WRITE(OUTU,*) &
                'ENERGY CUTOFF INCOMPATIBLE WITH NOENergy KEYWORD'
           CALL WRNDIE(-5,'<ZEROM>', &
                'INCOMPATIBLE ZSEArch KEYWORDS')
        ENDIF
        IF (QZVCUT) THEN
           WRITE(OUTU,*) &
                'VARIABLE ENER CUTOFF INCOMPATIBLE WITH NOENergy KEYWORD'
           CALL WRNDIE(-5,'<ZEROM>', 'BAD ENERGY KEYWORDS')
        ENDIF
        IF (QZMINI) THEN
           WRITE(OUTU,*) &
                'MINAssign KEYWORD INCOMPATIBLE WITH NOENergy KEYWORD'
           CALL WRNDIE(-5,'<ZEROM>', 'BAD ENERGY KEYWORDS')
        ENDIF
        IF (QZCFIX) THEN
           WRITE(OUTU,*) &
                'CFIX KEYWORD INCOMPATIBLE WITH NOENergy KEYWORD'
           call WRNDIE(-5,'<ZEROM>', 'BAD ENERGY KEYWORDS')
        endif
     endif !if not qzenerg
     !
!
     call chmalloc('zerom1.src','ZEROM','HPSSSL_hv',NSUBME,intg=HPSSSL_hv)
     call chmalloc('zerom1.src','ZEROM','HPSSS2_hv',NSUBME,intg=HPSSS2_hv)
     call chmalloc('zerom1.src','ZEROM','HPMXPS_hv',NSUBME,intg=HPMXPS_hv)
     call chmalloc('zerom1.src','ZEROM','HPPSIT_hv',NSUBME,intg=HPPSIT_hv)
     call chmalloc('zerom1.src','ZEROM','HPZBNS_hv',NZBINS,intg=HPZBNS_hv)
     HPSSSL_hv = 0
     HPSSS2_hv = 0
     HPMXPS_hv = 0
     HPPSIT_hv = 0
     HPZBNS_hv = 0
     if(INDXA(COMLYN,COMLEN,'SPEC').GT.0) then
        call ZSPECIFICS(HPLOIE_hv,HPLOIE_hv,HPLNLS_hv,HPCLST_hv,HPSATB_hv, &
             HPSATE_hv,HPLDSS_hv,HPSALS_hv,HPALIA_hv,-5,HPLOCN_hv,HPHICN_hv, &
             HPLODF_hv,HPHIDF_hv,HPMDFL_hv,HPMDFV_hv,HPMINSS_hv,HPDISAB1_hv, &
             HPDISAE1_hv,HPDISAB2_hv,HPDISAE2_hv,HPDIATL_hv,HPLDDIS_hv, &
             HPDISGAR_hv,HPDISLAR_hv,HPDFCGAR_hv,HPDFCLAR_hv,HPDFCMP_hv, &
             HPLDDFC_hv)
     ENDIF
     call ZCOMBOSET(HPSSSL_hv,HPSSS2_hv,ZNTAKE,ZPMODE,HPMXPS_hv,HPPSIT_hv, &
          HPLODF_hv,HPHIDF_hv,HPLOCN_hv,HPHICN_hv,HPMDFL_hv,HPMDFV_hv, &
          HPLDSS_hv,HPALIA_hv,HPALRF_hv,NATOM)
     call chmdealloc('zerom1.src','ZEROM','HPSSSL_hv',NSUBME,intg=HPSSSL_hv)
     call chmdealloc('zerom1.src','ZEROM','HPSSS2_hv',NSUBME,intg=HPSSS2_hv)
     call chmdealloc('zerom1.src','ZEROM','HPMXPS_hv',NSUBME,intg=HPMXPS_hv)
     call chmdealloc('zerom1.src','ZEROM','HPPSIT_hv',NSUBME,intg=HPPSIT_hv)
     call chmdealloc('zerom1.src','ZEROM','HPZBNS_hv',NZBINS,intg=HPZBNS_hv)
     !.......................................................................
  else if(WRD.EQ.'ZFIL') then
    ZLVTOL = 9.999D99
    ZGVTOL = 9.999D99
    ZGECUT = 9.999D99
    if(QZDATCH) call parstoperr('ZEROM','DATA NOT SET. USE ZSET')
!#if KEY_PARALLEL==1
!    if(QZCOMM) call COMM_ZDATA  
!#endif
    if (INDX(COMLYN,COMLEN,'LVCU',4).GT.0) then
      ZLVTOL = GTRMF(COMLYN,COMLEN,'LVCU',ZLVTOL)
      QZLVCT = .TRUE.
    endif
    if (INDX(COMLYN,COMLEN,'VCUT',4).GT.0) then
      ZGVTOL = GTRMF(COMLYN,COMLEN,'VCUT',ZGVTOL)
      QZGVCT = .TRUE.
    endif
    if (INDX(COMLYN,COMLEN,'ECUT',4).GT.0) then
      ZGECUT = GTRMF(COMLYN,COMLEN,'ECUT',ZGECUT)
      QZGECT = .TRUE.
    endif
    ZFILTWU = GTRMI(COMLYN,COMLEN,'WUNI',-1)
    WRD2=NEXTA4(COMLYN,COMLEN)
    if(WRD2.eq.'LECU') then
     QZECUTR = .true.
     if(allocated(ECUTAR)) call chmdealloc('zerom1.src','ZEROM','ECUTAR',NSUBSP,crl=ECUTAR)
     call chmalloc('zerom1.src','ZEROM','ECUTAR',NSUBSP,crl=ECUTAR)
     do while(COMLEN.gt.0) 
      NXTA = CURRA4(COMLYN,COMLEN)
      if(NXTA.eq.'END') then
        NXTA= NEXTA4(COMLYN,COMLEN)
        exit
      endif
      SS = NEXTI(COMLYN,COMLEN)
      if(SS.gt.NSUBSP) then 
       call WRNDIE(-5,'<ZEROM>',' bad subspace number')
      endif
      ECUTAR(SS) = NEXTF(COMLYN,COMLEN) 
      WRITE(6,*) 'SS ',SS,' ECUTAR ',ECUTAR(SS)
     enddo
    endif
    if(QZECUTR) then
!     call cs_trim(CSR,NSUBSP,ecut=ZGECUT,vcut=ZGVTOL,lvcut=ZLVTOL,ecutar=ECUTAR)
     if(QVERBOSE) write(6,*) 'size of CSW%MSTDF before cs_trim1 ',size(CSW%MSTDF)
     call cs_trim(CSW,NSUBSP,ecut=ZGECUT,vcut=ZGVTOL,lvcut=ZLVTOL,ecutar=ECUTAR)
     if(QVERBOSE) write(6,*) 'size of CSW%MSTDF after cs_trim1 ',size(CSW%MSTDF)
    else
!     call cs_trim(CSR,NSUBSP,ecut=ZGECUT,vcut=ZGVTOL,lvcut=ZLVTOL,ecutar=ECUTAR)
     if(QVERBOSE) write(6,*) 'size of CSW%MSTDF before cs_trim2 ',size(CSW%MSTDF)
     call cs_trim(CSW,NSUBSP,ecut=ZGECUT,vcut=ZGVTOL,lvcut=ZLVTOL,ecutar=ECUTAR)
     if(QVERBOSE) write(6,*) 'size of CSW%MSTDF after cs_trim2 ',size(CSW%MSTDF)
    endif
!    if(ZFILTWU.gt.0) call cs_write(CSR,NSUBSP,ZFILTWU)
    if(ZFILTWU.gt.0) call cs_write(CSW,NSUBSP,ZFILTWU)
    
  else if(WRD.EQ.'ZRAN') then
    if(QZDATCH) call parstoperr('ZEROM','DATA NOT SET. USE ZSET')
!#if KEY_PARALLEL==1
!    if(QZCOMM) call COMM_ZDATA
!#endif
    if(allocated(NRANDAR)) deallocate(NRANDAR)
    allocate(NRANDAR(NSUBSP))
    NRANDAR = 0
    if (INDX(COMLYN,COMLEN,'SEED',4).GT.0) then
     INSEED= GTRMI(COMLYN,COMLEN,'SEED',-1)
     QINSEED = .true.
     call set_randseed(INSEED)  !integer
    else if(.not.QINSEED) then 
     call WRNDIE(-5,'<ZEROM>', &
     'integer SEED > 0 not specified ')
    endif
!    INSEED= GTRMI(COMLYN,COMLEN,'SEED',-1)
!    if(INSEED.le.-1) call WRNDIE(-5,'<ZEROM>', &
!       'integer SEED > 0 not specified ')
  
    ZRANDWU = GTRMI(COMLYN,COMLEN,'WUNI',-1)

    QZRNDECT = .false.
    QZRNDVCT = .false.
    
    if (INDX(COMLYN,COMLEN,'ECUT',4).GT.0) then
     ZRNDECUT = GTRMF(COMLYN,COMLEN,'ECUT',-1.0D0)
     QZRNDECT = .TRUE.
    endif
    if (INDX(COMLYN,COMLEN,'VCUT',4).GT.0) then
      ZRNDVTOL = GTRMF(COMLYN,COMLEN,'VCUT',-1.0D0)
      QZRNDVCT = .TRUE.
    endif

    if(QZRNDECT.and.QZRNDVCT) then
     if(QVERBOSE) then
      WRITE(6,*) 'calling cstrim from zrand1, NSUBSP is ',NSUBSP
      WRITE(6,*) 'size of CSW%LOCNF is ',size(CSW%LOCNF)
      WRITE(6,*) 'size of CSW%MSTDF is ',size(CSW%MSTDF)
     endif
     call cs_trim(CSW,NSUBSP,ecut=ZRNDECUT,vcut=ZRNDVTOL)  
    elseif (QZRNDECT) then
     if(QVERBOSE) then
      WRITE(6,*) 'calling cstrim from zrand2, NSUBSP is ',NSUBSP
      WRITE(6,*) 'size of CSW%(LOCONF) is ',size(CSW%LOCNF)
      WRITE(6,*) 'size of CSR%(LOCONF) is ',size(CSR%LOCNF)
     endif
     call cs_trim(CSW,NSUBSP,ecut=ZRNDECUT)
    elseif (QZRNDVCT) then
     if(QVERBOSE) then
      WRITE(6,*) 'calling cstrim from zrand3, NSUBSP is ',NSUBSP
     endif
     call cs_trim(CSW,NSUBSP,vcut=ZRNDVTOL)
    endif

!    if(MYNODGP.eq.1) then  !##PARALLEL
!    call cs_write(CSW,NSUBSP,500)
!    endif !##PARALLEL
!    call parstoperr('<zerom1.src>','after printing CSW,energy cutoff')

    WRD2=NEXTA4(COMLYN,COMLEN)
    if (WRD2.EQ.'SUBS') then
      QALLSSR=.false.
      QALLSSR=(INDXA(COMLYN,COMLEN,'ALLS').GT.0)
      if(QALLSSR) then
       NALLRAND = NEXTI(COMLYN,COMLEN)
       NRANDAR = NALLRAND !array
#if KEY_PARALLEL==1
       if(MYNODGP.eq.1) then 
#endif
       WRITE(6,*) 'RANDOMIZE all subspaces, number of conformers: ',NALLRAND
#if KEY_PARALLEL==1
       endif
#endif
      else
       NRNDCNT = 0
       do while(COMLEN.GT.0) 
        NRNDCNT = NRNDCNT + 1
        if(NRNDCNT.GT.NSUBSP) then
         call WRNDIE(-5,'<ZEROM>', &
         'Number of randomized subspaces larger than number of subspaces')
        endif
        SS = NEXTI(COMLYN,COMLEN)
        if((SS.le.0).or.(SS.gt.NSUBSP)) then
         call WRNDIE(-5,'<ZEROM>', &
         'Invalid subspace number in ZRANdomize command')
        endif
        NRANDAR(SS) = NEXTI(COMLYN,COMLEN)
#if KEY_PARALLEL==1
       if(MYNODGP.eq.1) then
#endif
        WRITE(6,*) 'RANDOMIZE ',NRANDAR(SS),' conformers in SUBSPACE ',SS
#if KEY_PARALLEL==1
       endif 
#endif
        if(NRANDAR(SS).le.0) then
         call WRNDIE(-5,'<ZEROM>', &
         'Invalid number of conformers to be selected in randomization')
        endif
       enddo        
      endif
!           call parstoperr('<ZMERGE>',' STOP ')  !##PARALLEL
!      call randsel(NSUBSP,CSW,NRANDAR,ZRNDECUT,QZRNDECT,ZRNDVTOL,QZRNDVCT,INSEEDAR,OUTSEEDAR)
      call print_randseed 
     if(qverbose) then
      write(6,*) 'in zerom CSW%LODOF(1) is ',CSW%LODOF(1)
      write(6,*) 'in zerom CSR%LODOF(1) is ',CSR%LODOF(1)
      write(6,*) 'size of CSW%LOCNF before randsel',size(CSW%LOCNF)
      write(6,*) 'size of CSW%MSTDF before randsel',size(CSW%MSTDF)
     endif
      call randsel(NSUBSP,CSW,NRANDAR,ZRNDECUT,QZRNDECT,ZRNDVTOL,QZRNDVCT)
10  FORMAT(I14,I14,I14,F14.7,F14.7)
#if KEY_PARALLEL==1
      if(MYNODGP.eq.1) then 
#endif
       if(ZRANDWU.ge.1) then 
        call cs_write(CSW,NSUBSP,ZRANDWU)
!        do II = 1,NSUBSP
!         do CNF = CSW%LOCNF(II),CSW%HICNF(II)
!           do JJ = CSW%LODOF(CNF),CSW%HIDOF(CNF)
!            WRITE(ZRANDWU,10) II,CNF,CSW%MSTDF(JJ),CSW%MSTDV(JJ),CSW%ENERG(CNF)
!            WRITE(MYNODGP+300,10) II,CNF,CSW%MSTDF(JJ),CSW%MSTDV(JJ),CSW%ENERG(CNF)
!          enddo
!         enddo
!        enddo
       endif 
#if KEY_PARALLEL==1
      endif  
#endif
      QZRANDM = .true.
!      call parstoperr('<ZEROM>','after RANDSEL call ')
    endif !if wrd2 eq subs
  else if(WRD.EQ.'ZMIN') then
     ! assign structure to minimum over all searches since last zload command
     if (INDXA(COMLYN,COMLEN,'LOAD').GT.0) then
        call ZREGENMINX(HPSSSL_hv,HPLOCN_hv,HPHICN_hv,HPZNCB_hv,HPLCSS_hv, &
             HPHCSS_hv,HPCBSS_hv,HPSIAB_hv,HPSIAE_hv,HPINIL_hv,NATOM,HPLODF_hv, &
             HPHIDF_hv,HPMDFL_hv,HPMDFV_hv,HPLOIE_hv,HPHIIE_hv,HPLDSMN_hv, &
             LDSSMNE,HPQICBF_hv)
     endif
     !.......................................................................C
     !.......................................................................C
  else if(WRD.EQ.'ZSET') then  !'sets' the data
#if KEY_PARALLEL==1
!    write(6,*) 'MYNODG ',MYNODG,' calling comm_zdata'
    call COMM_ZDATA
    if(MYNODP.eq.1) write(6,'(3X,A26)') 'END OF READ-- SETTING DATA' 
#endif
    call cs_copy(CSR,CSW)
    QZDATCH=.false.
    if(qverbose) then
#if KEY_PARALLEL==1
     if(MYNODP.eq.1) then
#endif
      write(6,*) ' size of CSR%LOCNF ',size(CSR%LOCNF)
      write(6,*) ' size of CSW%LOCNF ',size(CSW%LOCNF)
      write(6,*) ' CSR%LODOF(1) ',CSR%LODOF(1)
      write(6,*) ' CSW%LODOF(1) ',CSW%LODOF(1)
      write(6,*) ' CSR%LODOF(10) ',CSR%LODOF(10)
      write(6,*) ' CSW%LODOF(10) ',CSW%LODOF(10)
      write(6,*) ' CSR%LODOF(100) ',CSR%LODOF(100)
      write(6,*) ' CSW%LODOF(100) ',CSW%LODOF(100)
#if KEY_PARALLEL==1
     endif
#endif
    endif

  else if(WRD.EQ.'END ')THEN
     !            -------------
     call ENDZERO(HPFLCI_hv,NATOM,IMOVE)
     DONE  = .TRUE.

     !.......................................................................
  ELSE
     !
     write(6,*) 'MYNODGP ',MYNODGP,' not a command, comlyn ',comlyn,' comlen ',comlen
     call WRNDIE(-1,'<ZEROM>','NOT A COMMAND:  '//WRD)

     DONE  = .TRUE.
  ENDIF

  !.......................................................................
  IF(DONE) RETURN
  GOTO 1000

  RETURN
END SUBROUTINE ZEROM
!----------------------------------------------------------------------
!----------------------------------------------------------------------
SUBROUTINE ADDDOFDEF(COMLYN,COMLEN,LOICEN, &
     HIICEN,LNLIST,COLLST,INIVAL,LENIC,B1,B2, &
     T1,T2,PIC,IAR,JAR,KAR,LAR,TAR)
  !  adds a dof definition to the list 
  use chm_kinds
  use dimens_fcm
  use psf
  use select
  use stream
  use string
  use zdata_mod
  use memory
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  INTEGER LOICEN(*),HIICEN(*),LNLIST(*),COLLST(*)
  INTEGER LENIC
  real(chm_real) B1(*),B2(*),T1(*),T2(*),PIC(*),INIVAL(*)
  INTEGER IAR(*),JAR(*),KAR(*),LAR(*)
  LOGICAL TAR(*)
  !
  ! local variables
  INTEGER QAT(4),NQAT
  integer, dimension(:),allocatable :: ISLCT
  INTEGER I,J,K,L,IIC,STORIC,ICLSTR,IVL
  LOGICAL FOUND,LPOS,LNEG,T
  LOGICAL DONE,EOF
  CHARACTER(len=4) WRD
  real(chm_real) STORVL
  !
  !  modified from Bernie's internal coordinates code
  !                                --RJP
  call chmalloc('zerom1.src','ADDDOFDEF','ISLCT',NATOM,intg=ISLCT)
  ISLCT = 0
  do I=1,4
     QAT(I)=0
  ENDDO
  WRD=NEXTA4(COMLYN,COMLEN)
  IF(WRD.EQ.'    ') THEN
     CONTINUE
  ELSE IF(WRD.EQ.'DIHE') THEN
     CALL NXTATM(QAT,NQAT,4,COMLYN,COMLEN,ISLCT, &
          SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
     I=QAT(1)
     J=QAT(2)
     K=QAT(3)
     T=(K.LT.0)
     IF(T) K=-K
     L=QAT(4)
     !
     IF(I.LE.0.OR.J.LE.0.OR.K.LE.0.OR.L.LE.0) THEN
        ! ATOM-DOESNT-EXIST
        CALL WRNDIE(-5,'<ADDDOFDEF>', &
             'ATOM IN DIHEDRAL ANGLE DOESNT EXIST')
     ELSE
        !           
        STORIC = 0
        FOUND=.FALSE.
        DO IIC=1,LENIC
           IF(I.EQ.IAR(IIC).AND.L.EQ.LAR(IIC)) THEN
              LPOS=(J.EQ.JAR(IIC).AND.K.EQ.KAR(IIC))
              LNEG=(J.EQ.KAR(IIC).AND.K.EQ.JAR(IIC))
           ELSE
              IF(I.EQ.LAR(IIC).AND.L.EQ.IAR(IIC)) THEN
                 LNEG=(J.EQ.JAR(IIC).AND.K.EQ.KAR(IIC))
                 LPOS=(J.EQ.KAR(IIC).AND.K.EQ.JAR(IIC))
              ELSE
                 LNEG=.FALSE.
                 LPOS=.FALSE.
              ENDIF
           ENDIF
           !
           IF(LNEG) THEN
              IF(PRNLEV.GE.2) WRITE(OUTU,75) IIC
75            FORMAT(15X,'FOUND DIHEDRAL IN IC',I5,' AS OPPOSITE')
              IF(T.NEQV.TAR(IIC).AND.WRNLEV.GE.2) WRITE(OUTU,76)
76            FORMAT(20X,'BUT TYPE VALUES DONT MATCH')
              IF(T.EQV.TAR(IIC)) FOUND=.TRUE.
              IF (STORIC.NE.0) THEN
                 CALL WRNDIE(-5,'<ADDDOFDEF>', &
                      'DIHEDRAL ANGLE APPEARS > ONCE IN IC TABLE')
              ELSE
                 STORIC = IIC
                 STORVL = PIC(IIC)
              ENDIF
           ENDIF
           IF(LPOS) THEN
              IF(PRNLEV.GE.2) WRITE(OUTU,85) IIC
              IF(T.NEQV.TAR(IIC) .AND. WRNLEV.GE.2) &
                   WRITE(OUTU,76)
              IF(T.EQV.TAR(IIC)) FOUND=.TRUE.
85            FORMAT(15X,'FOUND IN IC',I5,' AS POSITIVE')
              IF (STORIC.NE.0) THEN
                 CALL WRNDIE(-5,'<ADDDOFDEF>', &
                      'DIHEDRAL ANGLE APPEARS > ONCE IN IC TABLE')
              ELSE
                 STORIC = IIC
                 STORVL = PIC(IIC)
              ENDIF
           ENDIF
        ENDDO
        IF(.NOT.FOUND) THEN
           IF(PRNLEV.GE.2) THEN
              WRITE(OUTU,'(6X,A10,1x,I6,1x,I6,1x,I6,1x,I6,A10)') &
                   'DIHEDRAL ',I,J,K,L,'NOT FOUND'
           ENDIF
           CALL WRNDIE(-5,'<ADDDOFDEF>', &
                'DIHEDRAL ANGLE FOR DOF DEFINITION NOT FOUND')
        ENDIF
        NTHDDF = NTHDDF + 1
        NICELST = NICELST + 1
        LOICEN(NTHDDF) = NICELST
        HIICEN(NTHDDF) = NICELST
        INIVAL(NTHDDF) = STORVL
        LNLIST(NICELST) = STORIC
        COLLST(NICELST) = 3 
        !         ICTYPE(NICELST) = INTPIC 
        !         WRITE(OUTU,*) 'NTHDDF ',NTHDDF,' LOICEN ',LOICEN(NTHDDF),
        !     & ' HIICEN ',HIICEN(NTHDDF)
        !         WRITE(OUTU,*) 'NICELST ',NICELST,' LNLIST ',STORIC,
        !     & ' COLLST ',COLLST(NICELST)
     ENDIF !if atom does/doesnt exist
     !
     !-----------------------------------------------------------------------
     ! if it's an angle:
     !-----------------------------------------------------------------------
  ELSE IF(WRD.EQ.'ANGL' .OR. WRD.EQ.'THET' .OR.  &
       WRD.EQ.'ANG') THEN
     !        DO IIC = 1,LENIC
     !         WRITE(6,*) 'IIC ',IIC,' B1 ',B1(IIC),' T1 ',T1(IIC)
     !         WRITE(6,*)
     !     & ' IIC ',IIC,' PIC ',PIC(IIC),' T2 ',T2(IIC),' B2 ',B2(IIC)
     !        ENDDO
     ! PROCESS-ANGLE-EDIT
     ! Store original nicelst marker
     CALL NXTATM(QAT,NQAT,3,COMLYN,COMLEN,ISLCT, &
          SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
     I=QAT(1)
     J=QAT(2)
     K=QAT(3)
     !
     IF(I.LE.0.OR.J.LE.0.OR.K.LE.0) THEN
        ! ATOM-DOESNT-EXIST
        IF(WRNLEV.GE.2) THEN 
           CALL WRNDIE(-1,'<ADDDOFDEF>', &
                'atom in angle doesnt exist.')
        ENDIF
     ELSE
        ICLSTR = NICELST
        FOUND=.FALSE.
        DO IIC=1,LENIC
           IVL=0
           IF(I.EQ.IAR(IIC).OR.K.EQ.IAR(IIC)) IVL=IVL+1
           IF(I.EQ.JAR(IIC).OR.K.EQ.JAR(IIC)) IVL=IVL+2
           IF(I.EQ.KAR(IIC).OR.K.EQ.KAR(IIC)) IVL=IVL+4
           IF(I.EQ.LAR(IIC).OR.K.EQ.LAR(IIC)) IVL=IVL+8
           IF(.NOT.((IVL.NE.3.OR.J.NE.KAR(IIC).OR..NOT.TAR(IIC)) &
                .AND.(IVL.NE.5.OR.J.NE.JAR(IIC).OR.TAR(IIC)))) THEN
              FOUND=.TRUE.
              IF(PRNLEV.GE.2) WRITE(OUTU,115) IIC
115           FORMAT(15X,'FOUND IN IC',I5,' ON LEFT  SIDE')
              NICELST = NICELST + 1
              LNLIST(NICELST) = IIC
              COLLST(NICELST) = 2
              STORVL = T2(IIC)
              !            ICTYPE(NICELST) = INTT1
              !            WRITE(OUTU,*) 'NICELST ',NICELST,' LNLIST ',IIC,
              !     & ' COLLST ',COLLST(NICELST)
              !             WRITE(OUTU,*) 'STORVL = ',STORVL
              !              WRITE(OUTU,*) '************************'
           ENDIF
           IF(IVL.EQ.10.AND.J.EQ.KAR(IIC)) THEN
              FOUND=.TRUE.
              IF(PRNLEV.GE.2) WRITE(OUTU,125) IIC
125           FORMAT(15X,'FOUND IN IC',I5,' ON RIGHT SIDE')
              NICELST = NICELST + 1
              LNLIST(NICELST) = IIC
              COLLST(NICELST) = 4
              STORVL = T1(IIC)
              !            WRITE(OUTU,*) 'NICELST ',NICELST,' LNLIST ',IIC,
              !     & ' COLLST ',COLLST(NICELST)
              !            WRITE(OUTU,*) 'STORVL = ',STORVL
              !            WRITE(OUTU,*) '************************'
           ENDIF
        ENDDO
        !
        IF(.NOT.FOUND) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,175)
175        FORMAT(/10X,'ERROR IN EDITIC. INTERNAL ANGLE ', &
                'COORDINATE CANT BE FOUND.'/)
        ENDIF
        NTHDDF = NTHDDF + 1
        LOICEN(NTHDDF) = ICLSTR + 1 
        HIICEN(NTHDDF) = NICELST
        INIVAL(NTHDDF) = STORVL
        !          IF((NTHDDF.EQ.182).OR.(NTHDDF.EQ.183)) THEN
        !          WRITE(OUTU,*) 'NTHDDF ',NTHDDF,' LOICEN ',LOICEN(NTHDDF),
        !     & ' HIICEN ',HIICEN(NTHDDF)
        !          WRITE(OUTU,*) 'INIVAL is ',INIVAL(NTHDDF)
        !          ENDIF
     ENDIF !if atom exists
     !-----------------------------------------------------------------------
     ! if it's a bond:
     !-----------------------------------------------------------------------
  ELSE IF(WRD.EQ.'DIST' .OR. WRD.EQ.'BOND') THEN
     ! process-distance-edit
     CALL NXTATM(QAT,NQAT,2,COMLYN,COMLEN,ISLCT, &
          SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
     I=QAT(1)
     J=QAT(2)
     !
     IF(I.LE.0.OR.J.LE.0) THEN
        !         atom-doesnt-exist
        IF(WRNLEV.GE.2) THEN
           CALL WRNDIE(-1,'<ADDDOFDEF>', &
                'atom in bond doesnt exist.')
        ENDIF
     ELSE
        ICLSTR=NICELST
        FOUND=.FALSE.
        DO IIC=1,LENIC
           IVL=0
           IF(I.EQ.IAR(IIC).OR.J.EQ.IAR(IIC)) IVL=IVL+1
           IF(I.EQ.JAR(IIC).OR.J.EQ.JAR(IIC)) IVL=IVL+2
           IF(I.EQ.KAR(IIC).OR.J.EQ.KAR(IIC)) IVL=IVL+4
           IF(I.EQ.LAR(IIC).OR.J.EQ.LAR(IIC)) IVL=IVL+8
           IF((IVL.EQ.5.AND.TAR(IIC)).OR. &
                (IVL.EQ.3.AND..NOT.TAR(IIC))) THEN
              FOUND=.TRUE.
              IF(PRNLEV.GE.2) WRITE(OUTU,215) IIC
215           FORMAT(15X,'FOUND IN IC',I5,' ON LEFT  SIDE')
              NICELST = NICELST + 1
              LNLIST(NICELST) = IIC
              COLLST(NICELST) = 1
              !            ICTYPE(NICELST) = INTB1
              STORVL = B2(IIC)
              !            WRITE(OUTU,*) 'NICELST ',NICELST,' LNLIST ',IIC,
              !     & ' COLLST ',COLLST(NICELST)
           ENDIF
           IF(IVL.EQ.12) THEN
              FOUND=.TRUE.
              IF(PRNLEV.GE.2) WRITE(OUTU,225) IIC
225           FORMAT(15X,'FOUND IN IC',I5,' ON RIGHT SIDE')
              NICELST = NICELST + 1
              LNLIST(NICELST) = IIC
              COLLST(NICELST) = 5
              STORVL = B1(IIC)
              !            ICTYPE(NICELST) = INTB2
              !            WRITE(OUTU,*) 'NICELST ',NICELST,' LNLIST ',IIC,
              !     & ' COLLST ',COLLST(NICELST)
           ENDIF
        ENDDO
        !
        IF(.NOT.FOUND) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,275)
275        FORMAT(/10X,'ERROR IN EDITIC. INTERNAL DISTANCE ', &
                'COORDINATE CANT BE FOUND.'/)
           CALL WRNDIE(-5,'<ADDDOFDEF>', &
                'BOND NOT FOUND IN IC TABLE')
        ENDIF
        NTHDDF = NTHDDF + 1
        LOICEN(NTHDDF) = ICLSTR + 1
        HIICEN(NTHDDF) = NICELST
        INIVAL(NTHDDF) = STORVL
        !          WRITE(OUTU,*) 'NTHDDF ',NTHDDF,' LOICEN ',LOICEN(NTHDDF),
        !     & ' HIICEN ',HIICEN(NTHDDF)
     ENDIF
  ENDIF  !if dihedral, angle, or bond
  call chmdealloc('zerom1.src','ADDDOFDEF','ISLCT',NATOM,intg=ISLCT)
  RETURN 
 end subroutine adddofdef
!-----------------------------------------------------------------------
 subroutine addsubstr(ISLCT,JSLCT,NATOM,SATBEG,SATEND, &
     SUBATL,SIABEG,SIAEND,INIATL,TEMPAL,TMPIND,SALARR,SCUARR, &
     ZQICBF,QLOCREV)
  !
  !   Adds a substructure definition (a group of atoms) to the list
  ! 
  use chm_kinds
  use dimens_fcm
  use stream
  use exfunc
  use zdata_mod
  implicit none
  !
  INTEGER ISLCT(*),JSLCT(*),NATOM,SATBEG(*),SATEND(*)
  INTEGER SUBATL(*)
  INTEGER SIABEG(*),SIAEND(*),INIATL(*)
  INTEGER ZQICBF(*)
  ! for memory allocation only
  INTEGER TEMPAL(*),TMPIND(*),SALARR(*),SCUARR(*)
  !  local variables
  INTEGER I,NSUBAT,COUNTA,BBB,KKK,COUNTB,SUBS
  INTEGER NINITA
  INTEGER AAA
  LOGICAL SAME,MATCH,QLOCREV
  !
  !
  NSUBAT = 0
  NINITA = 0
  DO I = 1,NATOM
     IF (ISLCT(I).GT.0) THEN
        NSUBAT = NSUBAT + 1  !count for this substructure
        TSUBAT = TSUBAT + 1  !count over all substructures
        IF(TSUBAT.GT.NSATME) &
             CALL WRNDIE(-5,'<ADDSUBSTR>', &
             'SELECTD SUBSTRUC ATOMS EXCEED ALLCTED MEMO (ZMEM NATO)')
        SUBATL(TSUBAT) = I
     ENDIF
     IF (JSLCT(I).GT.0) THEN
        NINITA = NINITA + 1
        TINITA = TINITA + 1
        IF(TINITA.GT.NSATME) &
             CALL WRNDIE(-5,'<ADDSUBSTR>', &
             'SELECTD INITIALIZ ATOMS EXCEED ALLCTED MEMO (ZMEM NATO)')
        INIATL(TINITA) = I
        !        WRITE(6,*) 'ATOM ',I,' IS SET TO BE INITIALIZED ',
        !     & 'TINITA is ',TINITA
     ENDIF
  ENDDO
  IF (NSUBAT.LE.0) THEN 
     CALL WRNDIE(-5,'<ADDSUBSTR>','EMPTY SUBSTRUCTURE ')
     !
     !     IF (NSUBAT.GT.0) THEN
  ELSE
     NSUBDE = NSUBDE + 1
     !
     IF (QLOCREV) THEN
        ZQICBF(NSUBDE) = 0
     ELSE
        ZQICBF(NSUBDE) = 1
     ENDIF
     !
     !        WRITE(6,*) 'SUBSPACE ',NSUBDE,' HAS ZQICBF=',
     !     & ZQICBF(NSUBDE)
     IF(NSUBDE.GT.NSUBME)  &
          CALL WRNDIE(-5,'<ADDSUBSTR>', &
          'NUMBER OF SUBSTRUCS EXCEEDS ALLOCATED MEMO (ZMEM NSUB)')
     SATBEG(NSUBDE) = TSUBAT - NSUBAT + 1 
     SATEND(NSUBDE) = TSUBAT
     SIABEG(NSUBDE) = TINITA - NINITA + 1
     SIAEND(NSUBDE) = TINITA
     !        WRITE(6,*) 'NSUBDE is ',NSUBDE,' SIABEG is ',SIABEG(NSUBDE)
     !        WRITE(6,*) 'NSUBDE is ',NSUBDE,' SIAEND is ',SIAEND(NSUBDE)
     QSUBDE = .TRUE.
  ENDIF
  !      WRITE(6,*) 'NSUBAT ',NSUBAT,' NINITA ',NINITA
  !      WRITE(6,*) 'SIABEG ',SIABEG(NSUBDE),' SIAEND ',
  !     & SIAEND(NSUBDE) 
  !      WRITE(6,*) 'FIRST ATOM INIT ',INIATL(SIABEG(NSUBDE))
  !      WRITE(6,*) 'LAST  ATOM INIT ',INIATL(SIAEND(NSUBDE))
  COUNTB = 0
  DO BBB = SATBEG(NSUBDE),SATEND(NSUBDE)
     COUNTB = COUNTB + 1
     TEMPAL(COUNTB) = SUBATL(BBB)
  ENDDO
  CALL INDIXX(COUNTB,TEMPAL,TMPIND)
  DO BBB = 1,COUNTB
     KKK = TMPIND(BBB)
     SALARR(BBB) = TEMPAL(KKK)
  ENDDO
  SUBS = 1
  DO WHILE (SUBS.LT.NSUBDE)
     SAME = .FALSE.
     COUNTA = 0
     DO AAA = SATBEG(SUBS),SATEND(SUBS)
        COUNTA = COUNTA + 1 
        TEMPAL(COUNTB) = SUBATL(AAA)        
     ENDDO
     !        WRITE(6,*) 'NSUBDE ',NSUBDE,' SUBS ',SUBS,
     !     & ' COUNTA ',COUNTA,' COUNTB ',COUNTB
     IF(COUNTA.EQ.COUNTB) THEN
        !         WRITE(6,*) 'SUBS ',NSUBDE,' AND ',
        !     & SUBS
        CALL INDIXX(COUNTA,TEMPAL,TMPIND)

        DO BBB = 1,COUNTA
           KKK = TMPIND(BBB)
           SCUARR(BBB) = TEMPAL(KKK)
        ENDDO
        MATCH = .TRUE.
        BBB = 1
        DO WHILE((BBB.LE.COUNTA).AND.(MATCH))
           !           WRITE(6,*) SALARR(BBB),' & ',SCUARR(BBB)
           IF (SALARR(BBB).NE.SCUARR(BBB)) THEN
              MATCH = .FALSE.
           ENDIF
           BBB = BBB + 1
        ENDDO
        IF(MATCH) SAME = .TRUE.
     ENDIF
     IF (SAME) WRITE(OUTU,*) &
          'WARNING: SUBSTRUCS ',NSUBDE,' AND ', &
          SUBS,' HAVE COMMON ATOMS '
     SUBS = SUBS + 1
  ENDDO

  RETURN
 end subroutine addsubstr
!-----------------------------------------------------------------------
 subroutine subspaceload(COMLYN,COMLEN,LDSSLS, &
     ALILST)
  !
  !
  !  processes subspace loading 
  !
  use chm_kinds
  use zdata_mod
  use exfunc
  use stream
  use string
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER  COMLEN,LDSSLS(*),ALILST(*)
  !
  INTEGER STORSS,I,J
  LOGICAL DOUBLE
  character(len=4) :: NXTA
  !
  IF(INDXA(COMLYN,COMLEN,'CLEA').GT.0) THEN
     DO I = 1,NSSSLD
        LDSSLS(I) = 0
     ENDDO
     NSSSLD = 0
  ENDIF
  DO WHILE (COMLEN.GT.0)
     NXTA = CURRA4(COMLYN,COMLEN)
     if(NXTA.eq.'END') then 
      NXTA= NEXTA4(COMLYN,COMLEN)
      exit
     endif
     STORSS = NEXTI(COMLYN,COMLEN) 
     IF(STORSS.GT.NSUBDE) THEN
        WRITE(OUTU,*) 'SUBSPACE ',STORSS, &
             ' IS NOT DEFINED '
        CALL WRNDIE(-5,'<SUBSPACELOAD>', &
             'LOADING SUBSPACE THAT IS NOT DEFINED')
     ENDIF
     !        IF(ALILST(STORSS).GT.0) THEN
     !         WRITE(OUTU,*) 'SUBSPACE ',STORSS,
     !     & ' IS AN ALIAS OF SUBSPACE ',ALILST(STORSS)
     !         CALL WRNDIE(-5,'<SUBSPACELOAD>',
     !     & 'LOADING SUBSPACE ALIAS, NOT ALLOWED')
     !        ENDIF
     IF((STORSS.GT.NSUBSP).AND.(.NOT.QSUBNUL)) THEN
        WRITE(OUTU,*) 'SUBSPACE ',STORSS, &
             ' HAS NO CONFORMERS '
        CALL WRNDIE(-5,'<SUBSPACELOAD>', &
             'LOADING EMPTY SUBSPACE (NO CONFORMERS READ)')
     ENDIF
     IF(STORSS.LE.0) THEN
        WRITE(OUTU,*) 'SUBSPACE ',STORSS, &
             ' HAS AN INVALID SUBSPACE NUMBER '
        CALL WRNDIE(-5,'<SUBSPACELOAD>', &
             'INVALID SUBSPACE NUMBER')
     ENDIF
     I = 1
     DOUBLE = .FALSE.
     ! check this logic: I think it may be incorrect:
     DO WHILE((I.LE.NSSSLD).AND.(.NOT.DOUBLE)) 
        IF(STORSS.EQ.LDSSLS(I)) THEN
           WRITE(OUTU,300) 'WARNING: SUBSPACE ',LDSSLS(I), &
                ' LOADED TWICE, IGNORING'
300        FORMAT(2X,A18,I8,A23)
           DOUBLE = .TRUE.
        ENDIF
        I = I + 1
     ENDDO
     IF(.NOT.DOUBLE) THEN
        IF(NSSSLD+1.GT.NSUBME) THEN
           CALL WRNDIE(-5,'<SUBSPACELOAD>', &
                'NUMB OF LOADED SUBSPACES EXCEEDS ALLCTD MEMO (ZMEM NSUB)')
        ELSE 
           NSSSLD = NSSSLD + 1
           LDSSLS(NSSSLD) = STORSS !add to loaded subspace list
        ENDIF
     ENDIF
  ENDDO
  RETURN
 end subroutine subspaceload
! ----------------------------------------------------------------------
 subroutine categminorss(COMLYN,COMLEN,MINORSS,IMINORSS, &
     ALILST) !temporary
  ! makes a list of minor subspaces
  use chm_kinds
  use zdata_mod
  use exfunc
  use stream
  use string
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER  COMLEN

  INTEGER MINORSS(*),IMINORSS(*)
  INTEGER STORSS,I,J,LOCCNT,ALILST(*)
  LOGICAL DOUBLE
  !
  IF((INDXA(COMLYN,COMLEN,'CLEA').GT.0).OR. &
       (INDXA(COMLYN,COMLEN,'NONE').GT.0))  THEN
     DO I = 1,NMINORSS
        MINORSS(I) = 0
     ENDDO
     NMINORSS = 0
     DO I = 1,NSUBME
        IMINORSS(I) = 0
     ENDDO
     WRITE(OUTU,*) 'CLEARING MINOR SUBSPACES'
  ENDIF
  !      
  LOCCNT = 0
  DO WHILE (COMLEN.GT.0)
     STORSS = NEXTI(COMLYN,COMLEN)
     IF(STORSS.GT.NSUBDE) THEN
        WRITE(OUTU,*) 'SUBSPACE ',STORSS, &
             ' IS NOT DEFINED '
        CALL WRNDIE(-5,'<CATEGMINORSS>', &
             'CATEGORIZING SUBSPACE THAT IS NOT DEFINED')
     ENDIF
     !        IF(STORSS.GT.NSUBSP) THEN
     !         WRITE(OUTU,*) 'SUBSPACE ',STORSS,
     !     & ' HAS NO CONFORMERS '
     !         CALL WRNDIE(-5,'<CATEGMINORSS>',
     !     & 'CATEGORIZING EMPTY SUBSPACE (NO CONFORMERS READ)')
     !        ENDIF
     IF(STORSS.LE.0) THEN
        WRITE(OUTU,*) 'SUBSPACE ',STORSS, &
             ' HAS AN INVALID NUMBER (<0)'
        CALL WRNDIE(-5,'<CATEGMINORSS>', &
             'INVALID SUBSPACE NUMBER')
     ENDIF
     I = 1
     DOUBLE = .FALSE.
     ! check this logic: I think it may be incorrect:
     DO WHILE((I.LE.NMINORSS).AND.(.NOT.DOUBLE))
        IF(STORSS.EQ.MINORSS(I)) THEN
           WRITE(OUTU,*) 'WARNING: SUBSPACE ',MINORSS(I), &
                ' CATEGRZED AS MINOR TWICE, IGNORING'
300        FORMAT(2X,A18,I8,A23)
           DOUBLE = .TRUE.
        ENDIF
        I = I + 1
     ENDDO
     IF(.NOT.DOUBLE) THEN
        IF(NMINORSS+1.GT.NSUBME) THEN
           CALL WRNDIE(-5,'<SUBSPACELOAD>', &
                'NUMB OF MINOR SUBSPACES EXCEEDS ALLCTD MEMO (ZMEM NSUB)')
        ELSE
           NMINORSS = NMINORSS + 1
           LOCCNT = LOCCNT + 1
           IMINORSS(STORSS)=1
           MINORSS(NMINORSS) = STORSS !add to minor subspace list
        ENDIF
     ENDIF
  ENDDO
  WRITE(OUTU,'(A15,I8,A20)') '  CATEGORIZING ',LOCCNT, &
       ' SUBSPACES AS MINOR '
  !      DO I = 1,NSUBME
  !        WRITE(6,*) 'SUBSPACE ',I,' REFERENCE ',ALILST(I)
  !      ENDDO
  RETURN
 end subroutine categminorss
!
!
!-----------------------------------------------------------------------
 subroutine zspecifics(LOICEN,HIICEN,LNLIST,COLLST, &
     SATBEG,SATEND,LDSSLS,SUBATL,ALILST,LEVELW,LOCONF, &
     HICONF,LODOFL,HIDOFL,MSTDOF,MSTDFV,MINORSS, &
     ! for distances
     DISATBEG1,DISATEND1,DISATBEG2,DISATEND2,DISTATL,LDDISLS, &
     DISTGTAR,DISTLTAR,DOFCGAR,DOFCLAR,DOFCMAP,LDDOFCON)
  !
  !   writes current zero module specifications to standard output
  !
  use chm_kinds
  use zdata_mod
  use stream
#if KEY_PARALLEL==1
  use parallel,only: MYNODGP 
#endif
  implicit none
  !
  ! passed variables
  INTEGER LOICEN(*),HIICEN(*),LNLIST(*),COLLST(*)
  INTEGER SATBEG(*),SATEND(*),LDSSLS(*),SUBATL(*)
  INTEGER ALILST(*)
  INTEGER LEVELW
  INTEGER LOCONF(*),HICONF(*),LODOFL(*)
  INTEGER HIDOFL(*),MSTDOF(*)
  real(chm_real) MSTDFV(*)
  INTEGER MINORSS(*)
  INTEGER DISATBEG1(*),DISATEND1(*),DISATBEG2(*),DISATEND2(*)
  INTEGER DISTATL(*),LDDISLS(*)
  real(chm_real) DISTGTAR(*),DISTLTAR(*),DOFCGAR(*),DOFCLAR(*)
  INTEGER DOFCMAP(*),LDDOFCON(*)
  ! local variables
  !
  INTEGER I,II,JJ,J,LO,HI,NATM,LO2,HI2,KK
  INTEGER LOCALC,KKK
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then 
#endif
  WRITE(OUTU,'(A)') ' '
  WRITE(OUTU,'(3X,A)')  &
       '*******************SEARCH SPECIFICATIONS********************'
  !
  WRITE(OUTU,'(3X,I10,1X,A17)') NTHDDF,' DOFs DEFINED:'
#if KEY_PARALLEL==1
  endif 
#endif
  DO I = 1,NTHDDF
     LO = LOICEN(I)
     HI = HIICEN(I)
!     WRITE(OUTU,'(6X,A4,I8,A5,I8,A15)')  &
!          'DOF ',I,' HAS ',HI-LO+1,' ENTRIES IN IC '
     DO II = I+1,NTHDDF
        LO2 = LOICEN(II)
        HI2 = HIICEN(II)
        DO J = LO,HI
           DO JJ = LO2,HI2
              IF (LNLIST(J).EQ.LNLIST(JJ)) THEN
                 IF (COLLST(J).EQ.COLLST(JJ)) WRITE(OUTU,100) &
                      'WARNING: DOF DEFINITIONS ',I,' & ',II,' ARE IDENTICAL'
100              FORMAT(2X,A25,I8,A3,I8,A14)
              ENDIF
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  !
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then 
#endif
  WRITE(OUTU,'(3X,I10,1X,A23)') NSUBDE,' SUBSTRUCTURES DEFINED:'
#if KEY_PARALLEL==1
  endif 
#endif
  DO I=1,NSUBDE
     NATM = SATEND(I) - SATBEG(I) + 1 
#if KEY_PARALLEL==1
     if(MYNODGP.eq.1) then 
#endif
     WRITE(OUTU,'(6X,A13,I8,A9,I10,A6)')  &
          'SUBSTRUCTURE ',I,' CONTAINS ',NATM,' ATOMS ' 
#if KEY_PARALLEL==1
     endif 
#endif
     !       DO II = I+1,NSUBDE
     !        DO J = SATBEG(I),SATEND(I)
     !         DO JJ = SATBEG(II),SATEND(II)   
     !           IF (ALILST(II).NE.I) THEN
     !            IF (SUBATL(J).EQ.SUBATL(JJ)) WRITE(OUTU,200)
     !     & 'WARNING: SUBSTRUCS ',I,' & ',II,' HAVE COMMON ATOMS'             
     ! 200  FORMAT(2X,A19,I8,A3,I8,A18)
     !           ENDIF
     !         ENDDO
     !        ENDDO
     !       ENDDO 
  ENDDO
  !
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then 
#endif
   WRITE(OUTU,'(3X,A21)') ' SUBSPACE SIZES:'
   do II = 1,NSUBSP
     WRITE(6,'(6X,A9,1X,I10,1X,A3,1X,I10,1X,A10)')  &
          'SUBSPACE ',II,'HAS',HICONF(II)-LOCONF(II) +1, &
          'CONFORMERS'
   enddo
#if KEY_PARALLEL==1
  endif  
#endif
  !
  !      WRITE(OUTU,'(3X,I10,1X,A18)') NSSSLD,' SUBSPACES LOADED:'
!------------------------------------------------------
! write out error messages
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then 
#endif
  IF(NSSSLD.LE.0) THEN
     WRITE(OUTU,'(13X,A)')  &
          'NO SUBSPACES CURRENTLY LOADED'
     !        CALL WRNDIE(LEVELW,'<ZSPECIFICS>',
     !     & 'NO SUBSPACES ARE LOADED')
  ELSE IF(NSSSLD.EQ.1) THEN 
     WRITE(OUTU,'(10X,A)')  &
          'ONE SUBSPACE IS CURRENTLY LOADED'
  ELSE 
     WRITE(OUTU,'(3X,I10,A)')  &
          NSSSLD,' SUBSPACES ARE CURRENTLY LOADED'
  ENDIF
  DO I=1,NSSSLD
     IF((LDSSLS(I).GT.NSUBSP).OR.(LDSSLS(I).LE.0)) THEN
        WRITE(OUTU,*) I,'TH SUBSPACE LOADED ISNT DEFINED = ', &
             LDSSLS(I)
        IF(.NOT.QSUBNUL) THEN
           CALL WRNDIE(LEVELW,'<ZSPECIFICS>', &
                'BAD SUBSPACE LOADED')
        ENDIF
     ENDIF
     WRITE(OUTU,'(6X,A9,I8,A7)')  &
          'SUBSPACE ',LDSSLS(I),' LOADED'
     DO J = I+1,NSSSLD
        IF(LDSSLS(J).EQ.LDSSLS(I)) WRITE(OUTU,300) &
             'WARNING: SUBSPACE ',LDSSLS(I),' LOADED TWICE' 
300     FORMAT(2X,A18,I8,A13)
     ENDDO
     LOCALC = 0
     !       WRITE(6,*) 'THE CONF OF SS ',LDSSLS(I),
     !     & 'ARE: '
     DO JJ = LOCONF(LDSSLS(I)),HICONF(LDSSLS(I))
        LOCALC = LOCALC + 1
        !         WRITE(6,*) 'CONFORMER ',JJ
        IF (.NOT.QSUBNUL) THEN
           IF ((JJ.LE.0).OR.(JJ.GT.NCONFO)) THEN 
              WRITE(OUTU,*) 'SUBSPACE ',LDSSLS(I), &
                   'HAS UNDEFINED ',LOCALC,'TH CONFORMER= ',JJ
              CALL WRNDIE(LEVELW,'<ZSPECIFICS>', &
                   'BAD CONFORMER')
           ENDIF
        ENDIF
        DO KK = LODOFL(JJ),HIDOFL(JJ)
           !           WRITE(6,*) 'DOF ',MSTDOF(KK),' VAL ',
           !     & MSTDFV(KK)
           IF(MSTDOF(KK).GT.NTHDDF) THEN
              WRITE(OUTU,*) 'CONFORMER ',JJ, &
                   ' HAS AN UNDEFINED DOF ',MSTDOF(KK)
              CALL WRNDIE(LEVELW,'<ZSPECIFICS>', &
                   'UNDEFINED DEGREE OF FREEDOM')
           ENDIF
           !          DO KKK = LOICEN(MSTDOF(KK)),HIICEN(MSTDOF(KK)) 
           !           WRITE(6,*)' IC ENTRY ',KKK,' LINE ',
           !     & LNLIST(KKK),' COLUMN ',COLLST(KKK)
           !          ENDDO
        ENDDO
     ENDDO
  ENDDO
  WRITE(OUTU,'(3X,I10,1X,A)') &
       NMINORSS,' SUBSPACES CATEGORIZED AS MINOR '
  DO KK = 1,NMINORSS
     WRITE(OUTU,'(6X,A9,I8,A7)') 'SUBSPACE ',MINORSS(KK),' MINOR '
  ENDDO
  ! distance constraints
  WRITE(OUTU,'(6X,A22)') ' DISTANCE CONSTRAINTS:'
  IF(NDISTDE.EQ.0) THEN 
     WRITE(OUTU,'(13X,A)') 'NO DISTANCE CONSTRAINTS SPECIFIED'
  ELSE 
     DO II = 1,NDISTDE
        WRITE(6,'(6X,A11,1X,I10,1X,A3,1X,I10,1X,A3,1X,I10,1X,A5)') &
             'CONSTRAINT ',II,'HAS',DISATEND1(II)-DISATBEG1(II)+1,'AND', &
             DISATEND2(II)-DISATBEG2(II)+1,'ATOMS'
     ENDDO
     !
     WRITE(OUTU,'(6X,A20)') ' CONSTRAINTS LOADED:'
     IF(NDISTLD.LE.0) THEN
        WRITE(OUTU,'(13X,A)') &
             'NO DISTANCE CONSTRAINTS CURRENTLY LOADED'
     ELSE IF(NDISTLD.EQ.1) THEN
        WRITE(OUTU,'(13X,A)') &
             'ONE DISTANCE CONSTRAINT CURRENTLY LOADED'
     ELSE
        WRITE(OUTU,'(3X,I10,A)') &
             NDISTLD,' DISTANCE CONSTRAINTS CURRENTLY LOADED'
     ENDIF
     DO I=1,NDISTLD
        IF((LDDISLS(I).GT.NDISTDE).OR.(LDDISLS(I).LE.0)) THEN
           WRITE(OUTU,*) I,'TH CONSTRAINT LOADED ISNT DEFINED = ', &
                LDDISLS(I)
           CALL WRNDIE(LEVELW,'<ZSPECIFICS>', &
                'BAD DISTANCE CONSTRAINT LOADED')
        ENDIF
        WRITE(OUTU,'(6X,A20,I8,1X,A9)') &
             'DISTANCE CONSTRAINT ',LDDISLS(I),'IS LOADED'
        DO J = I+1,NDISTLD
           IF(LDDISLS(J).EQ.LDDISLS(I)) WRITE(OUTU,400) &
                'WARNING: DISTANCE CONSTRAINT ',LDSSLS(I),' LOADED TWICE'
400        FORMAT(2X,A29,I8,A13)
        ENDDO
     ENDDO !over distance constraints
  ENDIF !if distance constraints
  ! ic constraints
  WRITE(OUTU,'(6X,A22)') ' IC (DOF) CONSTRAINTS:'
  IF(NDOFCON.EQ.0) THEN
     WRITE(OUTU,'(13X,A)') 'NO IC CONSTRAINTS SPECIFIED'
  ELSE
     WRITE(OUTU,'(3X,I8,1X,A24)') &
          NDOFCON,'IC CONSTRAINTS SPECIFIED'
     DO II = 1,NDOFCON
        WRITE(6,'(6X,A13,1X,I8,1X,A8,1X,I8)') &
             'IC CONSTRAINT',II,'USES DOF',DOFCMAP(II)
     ENDDO
     WRITE(OUTU,'(I8,1X,A)') &
          NDOFCLD,'IC CONSTRAINTS LOADED'
     DO II = 1,NDOFCLD
        WRITE(6,'(6X,A13,1X,I8,1X,A6)') &
             'IC CONSTRAINT',II,'LOADED '
     ENDDO
  ENDIF !ic constraints or not
  WRITE(OUTU,'(3X,A)') &
       '******************END SEARCH SPECIFICATIONS******************'
#if KEY_PARALLEL==1
  endif 
#endif
  !
  RETURN
 end subroutine zspecifics
!
! -----------------------------------------------------------------
! -----------------------------------------------------------------
 subroutine aliasasgn(SATBEG,SATEND,ALILST,ALIASN, &
     ALIREF,SIABEG,SIAEND,ZQICBF)
  !
  !     assigns subspace/substructure aliases
  !
  use chm_kinds
  use zdata_mod
  use stream
  implicit none
  !
  INTEGER SATBEG(*),SATEND(*),ALILST(*)
  INTEGER ALIREF(*),SIABEG(*),SIAEND(*)
  INTEGER ZQICBF(*)
  INTEGER ALIASN
  !  
  INTEGER BBB,KK
  !     
  IF(ALIASN.GT.NSUBDE) THEN
     IF(ALIASN.EQ.(NSUBDE+1)) THEN
        CALL WRNDIE(-5,'<ALIASASGN>', &
             'SUBSPACE CANNOT BE ALIASED TO ITSELF')
     ELSE
        CALL WRNDIE(-5,'<ALIASASGN>', &
             'SUBSPACE ALIAS REFERENCE DOES NOT EXIST')
     ENDIF
  ELSE !aliasn is < nsubde
     DO BBB = 1,NSUBDE !loop over subspace definitions
        IF (ALILST(BBB).EQ.ALIASN) THEN
           WRITE(OUTU,*) &
                '  WARNING: SAME ALIAS REFERENCE IS USED MORE THAN ONCE'
        ENDIF
     ENDDO
     NSUBDE = NSUBDE + 1
     IF(NSUBDE.GT.NSUBME) &
          CALL WRNDIE(-5,'<ADDSUBSTR>', &
          'NUMBER OF SUBSTRUCS EXCEEDS ALLOCATED MEMO (ZMEM NSUB)')
     WRITE(OUTU,'(6X,A13,I8,A12,I8)') &
          'SUBSTRUCTURE ',NSUBDE,' ALIASED TO ',ALIASN
     ALILST(NSUBDE) = ALIASN !the original subspace reference
     ALIREF(ALIASN) = NSUBDE !the alias ;this can be "multivalued"; careful
     SATBEG(NSUBDE) = SATBEG(ALIASN)
     SATEND(NSUBDE) = SATEND(ALIASN)
     SIABEG(NSUBDE) = SIABEG(ALIASN)
     SIAEND(NSUBDE) = SIAEND(ALIASN)
     ZQICBF(NSUBDE) = ZQICBF(ALIASN)
  ENDIF !IF aliasn >= last sub definition (nsubde) or not
  !      DO KK = 1,NSUBDE
  !        WRITE(6,*) 'SUBSPACE ',KK,' REFERENCE ',ALILST(KK)
  !      ENDDO
  RETURN
 end subroutine aliasasgn
! --------------------------------------------------------------

 subroutine zinitarr(MSTDOF,MSTDFV,LODOFL,HIDOFL,LOCONF, &
     HICONF,LOICEN,HIICEN,LNLIST,COLLST,SATBEG,SATEND,SUBATL, &
     SIABEG,SIAEND,INIATL,LDSSLS,ALILST,ALIREF,ALLDDF, &
     FLACTI,FLINIT,DOINIT,FLINITR,DOINITR,INIVAL,SSNAME,GMINVAL, &
     LDDSSMN,IMINORSS,MINORSS,ZQICBF, &
     ! for distances
     DISATBEG1,DISATEND1,DISATBEG2,DISATEND2,DISTATL,LDDISLS, &
     DISTGTAR,DISTLTAR,DOFCGAR,DOFCLAR,DOFCMAP,LDDOFCON)
!     ) 
  !
  !  Initialize "long-term" arrays
  !
  use chm_kinds
  use dimens_fcm
  use zdata_mod
  use zstruc,only: CSR
  use inbnd
  use stream
  use psf
  use zcs,only: cs_init
  implicit none
  !
!!  INTEGER MSTDOF(*),LODOFL(*),HIDOFL(*)
!!  INTEGER LOCONF(*),HICONF(*),LOICEN(*),HIICEN(*)
!!  INTEGER LNLIST(*),COLLST(*),SATBEG(*),SATEND(*)
!!  INTEGER SUBATL(*),LDSSLS(*),ALILST(*)
!!  INTEGER SIABEG(*),SIAEND(*),INIATL(*),ALIREF(*)
!!  INTEGER ALLDDF(*),FLACTI(*),FLINIT(*),DOINIT(*)
!!  INTEGER FLINITR(*),DOINITR(*)
!!  INTEGER SSNAME(*)
!!  real(chm_real)  MSTDFV(*),INIVAL(*),GMINVAL(*),LDDSSMN(*) 
!!  INTEGER IMINORSS(*),MINORSS(*)
!!  INTEGER DISATBEG1(*),DISATEND1(*),DISATBEG2(*), &
!!       DISATEND2(*),DISTATL(*),LDDISLS(*)
!!  real(chm_real) DISTGTAR(*),DISTLTAR(*)
!!  INTEGER ZQICBF(*)
!!  real(chm_real) DOFCGAR(*),DOFCLAR(*) !for ic const
!!  INTEGER DOFCMAP(*),LDDOFCON(*) !for ic const

  integer,dimension(:) :: MSTDOF,LODOFL,HIDOFL,LOCONF,HICONF,LOICEN,HIICEN, &
    LNLIST,COLLST,SATBEG,SATEND,SUBATL,LDSSLS,ALILST,SIABEG,SIAEND,INIATL,ALIREF, &
    ALLDDF,FLACTI,FLINIT,DOINIT,FLINITR,DOINITR,SSNAME

  integer,dimension(:) :: IMINORSS,MINORSS,DISATBEG1,DISATEND1,DISATBEG2, &
    DISATEND2,DISTATL,LDDISLS,ZQICBF,DOFCMAP,LDDOFCON

  real(chm_real),dimension(:) :: MSTDFV,INIVAL,GMINVAL,LDDSSMN, &
     DISTGTAR,DISTLTAR,DOFCGAR,DOFCLAR

  !
  INTEGER III
  !
  ! check whether bycc has been selected
  !
  IF(.NOT.LBYCC) THEN
     WRITE(OUTU,'(A)') &
          'BYCC non-bonded option must be selected for ZERO.'
     WRITE(OUTU,'(A)') &
          'Clusters must have been made. (command = MKCLusters)'
     CALL WRNDIE(-5,'<ZINITARR>', &
          'BYCC OPTION NOT SELECTED')
  ENDIF
  !
  !      WRITE(6,*) 'INITIALIZING ARRAYS'
  !
     MSTDOF = 0
     MSTDFV = 0
     LODOFL = 0
     HIDOFL = 0
! for uncompressed conf arrays,read in
     call cs_init(CSR)
!     CSR%MSTDF = 0
!     CSR%MSTDV = 0
!     CSR%LODOF = 0
!     CSR%HIDOF = 0
!     CSR%LOCNF = 0
!     CSR%HICNF = 0
!     CSR%ENERG = 0
  !      WRITE(6,*) 'NSUBME is ',NSUBME
!size NSUBME               
     LOCONF = 0
     HICONF = 0
     SATBEG = 0
     SATEND = 0 
     SIABEG = 0
     SIAEND = 0
     LDSSLS = 0
     ALILST = 0
     ALIREF = 0
     SSNAME = 0
     IMINORSS = 0
     MINORSS = 0
     ZQICBF = 1 !ic build is in forward sense by default
     !        WRITE(6,*) 'III ',III,' ZQICBF ',ZQICBF(III)
     !        WRITE(6,*) 'III ',III,' IMINORSS ',IMINORSS(III)
!
! size DDEFME
     LOICEN = 0
     HIICEN = 0
     ALLDDF = 0
     INIVAL = 0
     GMINVAL = 0
     LDDSSMN = 0
  !      
! size 4*DDEFME
     LNLIST = 0    
     COLLST = 0
  !
! size NSATME
     SUBATL = 0
     INIATL = 0
  !
! size NDISTME
     DISATBEG1 = 0
     DISATEND1 = 0        
     DISATBEG2 = 0
     DISATEND2 = 0
     LDDISLS = 0
     DISTLTAR = 0
     DISTGTAR = 0 
  !
!size NDISATME
     DISTATL = 0
  !
! size NATOM
     FLACTI = 0
     FLINIT = 0
     DOINIT = 0
     FLINITR = 0
     DOINITR = 0
  !
! size NDOFCME
     DOFCGAR = 0
     DOFCLAR = 0
     DOFCMAP = 0
     LDDOFCON = 0
  !
  RETURN
 end subroutine zinitarr
!
! -------------------------------------------------------
!
 subroutine adddistcons(ISLCT,JSLCT,NATOM,DISATBEG1,DISATEND1, &
     DISTATL,DISATBEG2,DISATEND2,DISTGT,DISTLT,DISTGTAR,DISTLTAR)
  ! this subroutine adds a possible distance constraint to the list
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use exfunc
  use zdata_mod
#if KEY_PARALLEL==1
  use parallel,only: MYNODGP 
#endif
  implicit none
  !
  INTEGER ISLCT(*),JSLCT(*),NATOM,DISATBEG1(*),DISATEND1(*)
  INTEGER DISTATL(*),DISATBEG2(*),DISATEND2(*)
  real(chm_real) DISTGTAR(*),DISTLTAR(*)
  !  local variables
  INTEGER I,NDISTAT
  INTEGER AAA
  real(chm_real) DISTGT,DISTLT
  !
  NDISTAT = 0
  DO I = 1,NATOM
     IF (ISLCT(I).GT.0) THEN
        NDISTAT = NDISTAT + 1  !count for this substructure
        TDISTAT = TDISTAT + 1  !count over all substructures
        IF(TDISTAT.GT.NDISATME) &
             CALL WRNDIE(-5,'<ADDDISTCONS>', &
             'SELECTD SUBSTRUC ATOMS EXCEED ALLCTED MEMO (ZMEM NATD)')
        DISTATL(TDISTAT) = I
     ENDIF
  ENDDO
  IF (NDISTAT.LE.0) THEN
     CALL WRNDIE(-5,'<ADDDISTCONS>','EMPTY DISTANCE CONSTRAINT ')
  ELSE
     NDISTDE = NDISTDE + 1
     DISTLTAR(NDISTDE) = DISTLT*DISTLT  !upper limit of constraint
     DISTGTAR(NDISTDE) = DISTGT*DISTGT  !lower limit
     !        WRITE(6,*) 'UPPER LIMIT OF DIST CONS ',NDISTDE,' IS ',
     !     & DISTLTAR(NDISTDE)
     !        WRITE(6,*) 'LOWER LIMIT OF DIST CONS ',NDISTDE,' IS ',
     !     & DISTGTAR(NDISTDE)
     IF(NDISTDE.GT.NDISTME) &
          CALL WRNDIE(-5,'<ADDDISTCONS>', &
          'NUMBER OF SUBSTRUCS EXCEEDS ALLOCATED MEMO (ZMEM NDIS)')
     DISATBEG1(NDISTDE) = TDISTAT - NDISTAT + 1
     DISATEND1(NDISTDE) = TDISTAT
     !        WRITE(6,*) 'NDISTDE is ',NDISTDE,' DISATBEG1 is ',
     !     & DISATBEG1(NDISTDE)
     !        WRITE(6,*) 'NDISTDE is ',NDISTDE,' DISATEND1 is ',
     !     & DISATEND1(NDISTDE)
     QDISTDE = .TRUE.
  ENDIF
  NDISTAT = 0
  DO I = 1,NATOM
     IF (JSLCT(I).GT.0) THEN
        NDISTAT = NDISTAT + 1  !count for this substructure
        TDISTAT = TDISTAT + 1  !count over all substructures
        IF(TDISTAT.GT.NDISATME) &
             CALL WRNDIE(-5,'<ADDDISTCONS>', &
             'SELECTD SUBSTRUC ATOMS EXCEED ALLCTED MEMO (ZMEM NATD)')
        DISTATL(TDISTAT) = I
     ENDIF
  ENDDO
  !      WRITE(6,*) 'NUM OF ATOMS FOR DIST CONS ',NDISTDE,' IS ',NDISTAT
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then
#endif
  WRITE(OUTU,'(3X,A,I10)')  &
       'TOTAL NO. OF ATOMS FOR ALL DISTANCE CONSTRAINTS=  ', &
       TDISTAT
#if KEY_PARALLEL==1
  endif 
#endif
  IF (NDISTAT.LE.0) THEN
     CALL WRNDIE(-5,'<ADDDISTCONS>','EMPTY DISTANCE CONSTRAINT ')
  ELSE
     !        NDISTDE = NDISTDE + 1
     !        IF(NDISTDE.GT.NDISTME)
     !     &    CALL WRNDIE(-5,'<ADDSUBSTR>',
     !     & 'NUMBER OF DISTANCES EXCEEDS ALLOCATED MEMO (ZMEM NDIS)')
     DISATBEG2(NDISTDE) = TDISTAT - NDISTAT + 1
     DISATEND2(NDISTDE) = TDISTAT
     !        WRITE(6,*) 'NDISTDE is ',NDISTDE,' DISATBEG2 is ',
     !     & DISATBEG2(NDISTDE)
     !        WRITE(6,*) 'NDISTDE is ',NDISTDE,' DISATEND2 is ',
     !     & DISATEND2(NDISTDE)

     !        WRITE(6,*) 'NDISTDE is ',NDISTDE,' SIABEG is ',SIABEG(NDISTDE)
     !        WRITE(6,*) 'NDISTDE is ',NDISTDE,' SIAEND is ',SIAEND(NDISTDE)
  ENDIF

  !      WRITE(6,*) 'NDISTAT ',NDISTAT,' NINITA ',NINITA
  !      WRITE(6,*) 'FIRST ATOM INIT ',INIATL(SIABEG(NDISTDE))
  !      WRITE(6,*) 'LAST  ATOM INIT ',INIATL(SIAEND(NDISTDE))
  !      COUNTB = 0
  !      DO BBB = DISATBEG(NDISTDE),DISATEND(NDISTDE)
  !         COUNTB = COUNTB + 1
  !         TEMPAL(COUNTB) = DISTATL(BBB)
  !      ENDDO
  !      CALL INDIXX(COUNTB,TEMPAL,TMPIND)
  !      DO BBB = 1,COUNTB
  !          KKK = TMPIND(BBB)
  !          SALARR(BBB) = TEMPAL(KKK)
  !      ENDDO
  !      SUBS = 1
  !      DO WHILE (SUBS.LT.NDISTDE)
  !        SAME = .FALSE.
  !        COUNTA = 0
  !        DO AAA = DISATBEG(SUBS),DISATEND(SUBS)
  !          COUNTA = COUNTA + 1
  !          TEMPAL(COUNTB) = DISTATL(AAA)
  !        ENDDO
  !C        WRITE(6,*) 'NDISTDE ',NDISTDE,' SUBS ',SUBS,
  !C     & ' COUNTA ',COUNTA,' COUNTB ',COUNTB
  !         IF(COUNTA.EQ.COUNTB) THEN
  !C         WRITE(6,*) 'SUBS ',NDISTDE,' AND ',
  !C     & SUBS
  !         CALL INDIXX(COUNTA,TEMPAL,TMPIND)
  !
  !         DO BBB = 1,COUNTA
  !          KKK = TMPIND(BBB)
  !          SCUARR(BBB) = TEMPAL(KKK)
  !         ENDDO
  !         MATCH = .TRUE.
  !         BBB = 1
  !         DO WHILE((BBB.LE.COUNTA).AND.(MATCH))
  !C           WRITE(6,*) SALARR(BBB),' & ',SCUARR(BBB)
  !           IF (SALARR(BBB).NE.SCUARR(BBB)) THEN
  !            MATCH = .FALSE.
  !           ENDIF
  !           BBB = BBB + 1
  !         ENDDO
  !         IF(MATCH) SAME = .TRUE.
  !        ENDIF
  !        IF (SAME) WRITE(OUTU,*)
  !     &  'WARNING: SUBSTRUCS ',NDISTDE,' AND ',
  !     & SUBS,' HAVE COMMON ATOMS '
  !        SUBS = SUBS + 1
  !      ENDDO
  !
  RETURN
 end subroutine adddistcons
!
 subroutine distconsload(COMLYN,COMLEN,LDDISLS)
  use chm_kinds
  use zdata_mod
  use exfunc
  use stream
  use string
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER  COMLEN,LDDISLS(*)
  !
  INTEGER STORDI,I,J
  LOGICAL DOUBLE
  !
  IF(INDXA(COMLYN,COMLEN,'CLEA').GT.0) THEN
     DO I = 1,NDISTLD
        LDDISLS(I) = 0
     ENDDO
     NDISTLD = 0
  ENDIF
  DO WHILE (COMLEN.GT.0)
     STORDI = NEXTI(COMLYN,COMLEN)
     IF(STORDI.GT.NDISTDE) THEN
        WRITE(OUTU,*) 'CONSTRAINT ',STORDI, &
             ' IS NOT DEFINED '
        CALL WRNDIE(-5,'<DISTCONSLOAD>', &
             'LOADING DIST CONSTRAINT THAT IS NOT DEFINED')
     ENDIF
     IF(STORDI.LE.0) THEN
        WRITE(OUTU,*) 'DIST CONSTRAINT ',STORDI, &
             ' HAS AN INVALID CONSTRAINT NUMBER '
        CALL WRNDIE(-5,'<DISTCONSLOAD>', &
             'INVALID DISTANCE CONSTRAINT NUMBER')
     ENDIF
     I = 1
     DOUBLE = .FALSE.
     ! check this logic: I think it may be incorrect:
     DO WHILE((I.LE.NDISTLD).AND.(.NOT.DOUBLE))
        IF(STORDI.EQ.LDDISLS(I)) THEN
           WRITE(OUTU,300) 'WARNING: DIST CONSTRAINT ',LDDISLS(I), &
                ' LOADED TWICE, IGNORING'
300        FORMAT(2X,A18,I8,A23)
           DOUBLE = .TRUE.
        ENDIF
        I = I + 1
     ENDDO
     IF(.NOT.DOUBLE) THEN
        IF(NDISTLD+1.GT.NDISTME) THEN
           CALL WRNDIE(-5,'<DISTCONSLOAD>', &
                'NUMB OF LOADED DIST CONSTRAINTS EXCEEDS ALLCTD MEMO (ZMEM NDIS)')
        ELSE
           NDISTLD = NDISTLD + 1
           LDDISLS(NDISTLD) = STORDI !add to loaded subspace list
        ENDIF
     ENDIF
  ENDDO
  RETURN
 end subroutine distconsload
!
 subroutine dofconsload(COMLYN,COMLEN,LDDOFCON)
  !
  use chm_kinds
  use zdata_mod
  use exfunc
  use stream
  use string
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER  COMLEN,LDDOFCON(*)
  !
  INTEGER STORDI,I,J
  LOGICAL DOUBLE
  !
  IF(INDXA(COMLYN,COMLEN,'CLEA').GT.0) THEN
     DO I = 1,NDOFCLD
        LDDOFCON(I) = 0
     ENDDO
     NDOFCLD = 0
  ENDIF
  DO WHILE (COMLEN.GT.0)
     STORDI = NEXTI(COMLYN,COMLEN)
     IF(STORDI.GT.NDOFCON) THEN
        WRITE(OUTU,*) 'IC CONSTRAINT ',STORDI, &
             ' IS NOT DEFINED '
        CALL WRNDIE(-5,'<DOFCONSLOAD>', &
             'LOADING IC CONSTRAINT THAT IS NOT DEFINED')
     ENDIF
     IF(STORDI.LE.0) THEN
        WRITE(OUTU,*) 'IC CONSTRAINT ',STORDI, &
             ' HAS AN INVALID CONSTRAINT NUMBER '
        CALL WRNDIE(-5,'<DOFCONSLOAD>', &
             'INVALID IC CONSTRAINT NUMBER')
     ENDIF
     I = 1
     DOUBLE = .FALSE.
     ! check this logic: I think it may be incorrect:
     DO WHILE((I.LE.NDOFCLD).AND.(.NOT.DOUBLE))
        IF(STORDI.EQ.LDDOFCON(I)) THEN
           WRITE(OUTU,300) 'WARNING:  IC CONSTRAINT ',LDDOFCON(I), &
                ' LOADED TWICE, IGNORING'
300        FORMAT(2X,A18,I8,A23)
           DOUBLE = .TRUE.
        ENDIF
        I = I + 1
     ENDDO
     IF(.NOT.DOUBLE) THEN
        IF(NDOFCLD+1.GT.NDOFCME) THEN
           CALL WRNDIE(-5,'<DOFCONSLOAD>', &
                'NUMB OF LOADED IC CONSTRAINTS EXCEEDS ALLCTD MEMO (ZMEM NDIS)')
        ELSE
           NDOFCLD = NDOFCLD + 1
           LDDOFCON(NDOFCLD) = STORDI !add to loaded subspace list
           WRITE(6,*) 'LOADING IC CONS ',STORDI
        ENDIF
     ENDIF
  ENDDO
  RETURN
 end subroutine dofconsload
!
 subroutine adddofcons(LOCDOF,DOFCGAR,DOFCLAR,DISTGT, &
     DISTLT,DOFCMAP)
  use chm_kinds
  use dimens_fcm
  use stream
  use exfunc
  use zdata_mod
#if KEY_PARALLEL==1
  use parallel,only: MYNODGP 
#endif
  implicit none
  !
  INTEGER LOCDOF,DOFCMAP(*)
  real(chm_real) DISTGT,DISTLT,DOFCGAR(*),DOFCLAR(*)
  !
  IF ((LOCDOF.LE.0).OR.(LOCDOF.GT.NTHDDF)) THEN
     CALL WRNDIE(-5,'<ADDDOFCONS>','BAD DOF CONSTRAINT NAME')
  ELSE
     NDOFCON = NDOFCON + 1
     IF(NDOFCON.GT.NDOFCME) &
          CALL WRNDIE(-5,'<ADDDISTCONS>', &
          'NUMB OF DOF CONSTRNTS EXCEEDS ALLOCATED MEMO (ZMEM )')
     !
     DOFCMAP(NDOFCON) = LOCDOF  !map of for dof constraints
     DOFCLAR(NDOFCON) = DISTLT  !upper limit of constraint
     DOFCGAR(NDOFCON) = DISTGT  !lower limit
#if KEY_PARALLEL==1
     if(MYNODGP.eq.1) then 
#endif
     WRITE(6,*) 'UPPER LIMIT OF DIST CONS ',NDOFCON,' IS ', &
          DOFCLAR(NDOFCON)
     WRITE(6,*) 'LOWER LIMIT OF DIST CONS ',NDOFCON,' IS ', &
          DOFCGAR(NDOFCON)
#if KEY_PARALLEL==1
     endif 
#endif

     IF(NDOFCON.GT.NDOFCME) &
          CALL WRNDIE(-5,'<ADDDOFCONS>', &
          'NUMBER OF DOF CONSTRAINTS EXCEEDS ALLOCATED MEMO (ZMEM NDCO)')
     QDOFCONS = .TRUE.
  ENDIF
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then 
#endif
  WRITE(OUTU,'(3X,A22,1X,I8,1X,A8,1X,I8)') 'CREATING IC CONSTRAINT', &
       NDOFCON,'with DOF',LOCDOF
  !      WRITE(6,*) 'NUM OF ATOMS FOR DIST CONS ',NDISTDE,' IS ',NDISTAT
#if KEY_PARALLEL==1
  endif 
#endif

  RETURN
 end subroutine adddofcons
!
 subroutine endzero(FLACTI,NATOMX,IMOVE)
  ! clean up at end of zero run
  use chm_kinds
  use stream
  use zdata_mod
#if KEY_PARALLEL==1
  use parallel,only: MYNODGP 
#endif
  implicit none
  INTEGER FLACTI(*),NATOMX,IMOVE(*)
  ! local variables
  INTEGER III
  ! make all atoms active before getting out
  !      WRITE(OUTU,'(3X,A42)') 
  !     & 'MAKING ALL ATOMS ACTIVE BEFORE TERMINATING'
  !      DO III = 1,NATOMX
  !        FLACTI(III) = 1
  !        IMOVE(III) = 0  !unconstraining
  !      ENDDO
  !      CALL ZNBACTV(FLACTI)    
  !      WRITE(OUTU,'(3X,A41)')
  !     & 'UNCONSTRAINING ALL ATOMS (CONS FIX NONE) '
  !      QZEROMEM = .FALSE.
  QNOZMEMCL = .true.  !temporary--this should be parsed
  if(.not.QNOZMEMCL) then
  CALL CLEARZMEM
  endif
  QZMOD = .FALSE.  !leaving zero module
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then 
#endif
   WRITE(OUTU,'(3X,A37)') &
       'Exiting zero module'
   WRITE(OUTU,'(3X,A61)') &
       '*************************************************************'
#if KEY_PARALLEL==1
  endif 
#endif
  !
  RETURN
 end subroutine endzero
#endif /*zeromain*/

 subroutine clearzmem
  use dimens_fcm
  use chm_kinds
  use zdata_mod
  use memory
  use consta
  use number
  use stream
  use psf
  use ztypes
  use zstruc,only: CSR
  use zcs,only: cs_alloc
  implicit none

  WRITE(OUTU,100) 'Z MODULE MEMORY HAS BEEN CLEARED'
  WRITE(OUTU,100)
100 FORMAT(3X,A)
  call chmdealloc('zerom1.src','CLEARZMEM','HPMDFL_hv',NVALME,intg=HPMDFL_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPMDFV_hv',NVALME,crl=HPMDFV_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPLODF_hv',NCNFME,intg=HPLODF_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPHIDF_hv',NCNFME,intg=HPHIDF_hv)
!uncompressed arrays
  call cs_alloc(CSR,NSUBME,NCNFME,NVALME,'zerom1.src','ZEROM',QDEALL=.true.)

!  call chmdealloc('zerom1.src','CLEARZMEM','LODOFLW',NCNFME,intg=HPLODF_hv)
!  call chmdealloc('zerom1.src','CLEARZMEM','HIDOFLW',NCNFME,intg=HPHIDF_hv)
!
  call chmdealloc('zerom1.src','CLEARZMEM','HPLOCN_hv',NSUBME,intg=HPLOCN_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPHICN_hv',NSUBME,intg=HPHICN_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPLOIE_hv',DDEFME,intg=HPLOIE_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPHIIE_hv',DDEFME,intg=HPHIIE_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPLNLS_hv',4*DDEFME,intg=HPLNLS_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPCLST_hv',4*DDEFME,intg=HPCLST_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPSATB_hv',NSUBME,intg=HPSATB_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPSATE_hv',NSUBME,intg=HPSATE_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPSALS_hv',NSATME,intg=HPSALS_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPSIAB_hv',NSUBME,intg=HPSIAB_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPSIAE_hv',NSUBME,intg=HPSIAE_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPINIL_hv',NSATME,intg=HPINIL_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPLDSS_hv',NSUBME,intg=HPLDSS_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPMINSS_hv',NSUBME,intg=HPMINSS_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPIMISS_hv',NSUBME,intg=HPIMISS_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPALIA_hv',NSUBME,intg=HPALIA_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPALRF_hv',NSUBME,intg=HPALRF_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPADDF_hv',DDEFME,intg=HPADDF_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPFLCI_hv',NATOM,intg=HPFLCI_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPFLIN_hv',NATOM,intg=HPFLIN_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDOIN_hv',NATOM,intg=HPDOIN_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPFLINR_hv',NATOM,intg=HPFLINR_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDOINR_hv',NATOM,intg=HPDOINR_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPINVL_hv',DDEFME,crl=HPINVL_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPSSNM_hv',NSUBME,intg=HPSSNM_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPQICBF_hv',NSUBME,intg=HPQICBF_hv)
  ! for distances
  call chmdealloc('zerom1.src','CLEARZMEM','HPDISAB1_hv',NDISTME,intg=HPDISAB1_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDISAE1_hv',NDISTME,intg=HPDISAE1_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDISAB2_hv',NDISTME,intg=HPDISAB2_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDISAE2_hv',NDISTME,intg=HPDISAE2_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDIATL_hv',NDISATME,intg=HPDIATL_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPLDDIS_hv',NDISTME,intg=HPLDDIS_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDISGAR_hv',NDISTME,crl=HPDISGAR_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDISLAR_hv',NDISTME,crl=HPDISLAR_hv)
  ! for ic constraints
  call chmdealloc('zerom1.src','CLEARZMEM','HPDFCGAR_hv',NDOFCME,crl=HPDFCGAR_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDFCLAR_hv',NDOFCME,crl=HPDFCLAR_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPDFCMP_hv',NDOFCME,intg=HPDFCMP_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPLDDFC_hv',NDOFCME,intg=HPLDDFC_hv)

  call chmdealloc('zerom1.src','CLEARZMEM','HPGMNV_hv',DDEFME,crl=HPGMNV_hv)
  call chmdealloc('zerom1.src','CLEARZMEM','HPLDSMN_hv',DDEFME,crl=HPLDSMN_hv)
 end subroutine clearzmem

end module zmodule1
