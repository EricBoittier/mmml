 module zmod_comm
#if KEY_ZEROM==1
  use chm_types
  use dimens_fcm
  use parallel
  use zutil,only: loc_timer
  implicit none
  real(chm_real) :: etime,ctime
!  logical,private :: QZTIME=.false.,qverbose=.false.
  logical,private :: qverbose=.false.,QWRITE=.false.
contains

#if KEY_PARALLEL==1 /*parallel_comm*/
  subroutine COMM_ZDATA
  use parallel
  use zdata_mod  
  use nbndcc_utilb
  use psf,only: NATOM
  use memory
  use ztypes
  use zstruc,only: CSR,CSW
  use zcs,only: cs_init,cs_copy
  use cpustruc,only: NUMHOOD,QCPUONES
  implicit none

  include 'mpif.h'

  integer :: NIALL,NRALL,NQALL,sizeall
  integer,dimension(:),allocatable :: IALL
  integer(chm_int4) :: ierr,node0
  real(chm_real),dimension(:),allocatable :: RALL
  logical,dimension(:),allocatable :: QALL

! communicate integers

!  write(6,*) 'MYNODGP is ',MYNODGP,' NUMHOOD is ',NUMHOOD
!  write(6,*) 'MYNODGP ',MYNODGP,' NUMNODG ',NUMNODG,' MYNODG ',MYNODG
  NUMNOD = NUMNODG  !set global communication
  MYNODP = MYNODG+1
  MYNOD = MYNODG

  NIALL =15
  if(allocated(IALL)) call chmdealloc('zerom_comm.src','COMM_ZDATA','IALL',NIALL,intg=IALL)
  call chmalloc('zerom_comm.src','COMM_ZDATA','IALL',NIALL,intg=IALL)

 !NSUBME is only size taken from ZMEM allocation. Rest correspond to what is actually used.
!  WRITE(6,*) 'MYNODP ',MYNODP,' BEFORE THE BROADCAST, NSUBME is ',NSUBME
!first broadcast some basic data (integers) to give array sizes 
  if(MYNODGP.eq.1) then
   IALL(1) = NCONFO
   IALL(2) = NMASTR
   IALL(3) = NMASTRW
   IALL(4) = NICELST
   IALL(5) = NTHDDF
   IALL(6) = NALLDF
   IALL(7) = TSUBAT
   IALL(8) = TINITA
   IALL(9) = NDISTDE
   IALL(10) = TDISTAT
   IALL(11) = NDOFCON
   IALL(12) = NSUBSP  !# of ss read
   IALL(13) = NSUBDE !#number of ss definitions
   IALL(14) = NALIAS !# of aliases (??used)
   IALL(15) = NMINORSS !# of minor ss 
  else
   IALL = 0
  endif
  NODE0 = 0

  if(QZTIME) call loc_timer('ZCOMM BEF INITBCAST')

  call MPI_BCAST(IALL,NIALL,MPI_INTEGER,NODE0,MPI_COMM_WORLD,ierr)
  
  if(qverbose) write(6,*) 'MYNODGP ',MYNODGP,' after initial broadcast'
  if(MYNODGP.ne.1) then
   NCONFO = IALL(1)
   NMASTR = IALL(2)
   NMASTRW = IALL(3)
   NICELST = IALL(4)
   NTHDDF = IALL(5)
   NALLDF = IALL(6)
   TSUBAT = IALL(7)
   TINITA = IALL(8)
   NDISTDE = IALL(9)
   TDISTAT = IALL(10)
   NDOFCON = IALL(11)
   NSUBSP = IALL(12)
   NSUBDE = IALL(13)
   NALIAS = IALL(14)
   NMINORSS = IALL(15)
  endif
  if(QZTIME) call loc_timer('ZCOMM BEF INTEG') 

!now broadcast the rest of data
  sizeall =  NALLDF + NICELST*8 + NTHDDF*2 + NSUBME*12 + TSUBAT + TINITA +  &
        NDISTDE*4 + TDISTAT + NDOFCON + NCONFO*4 + NMASTR + NMASTRW + 2*NSUBSP

  if(allocated(IALL)) call chmdealloc('zerom_comm.src','COMM_ZDATA','IALL',sizeall,intg=IALL)
  call chmalloc('zerom_comm.src','COMM_ZDATA','IALL',sizeall,intg=IALL)

  IALL = 0
  if(MYNODGP.eq.1) then  !head node
! don't need to communicate these

!   IALL()=NZBINS !in zsearch
!   IALL(1)=NSUBDE  !data (num ss)
!   IALL(2)=NALIAS  !not actually used (??)
!   IALL(3)=NSUBSP !data number of ss read
!   IALL(3)=NMINORSS !data number of minor ss
!   IALL(26)=ZTMINO !in zsearch
!   IALL(27)=ZWRUNI !in zsearch
!   IALL(28)=ZWRTUNI !in zsearch
!   IALL(29)=ZTNSTEP !in zsearch
!   IALL(30)=ZMINUNI !in zsearch
!   IALL(31)=ZSTEPN grid point counter for each search, essentially a result
!   IALL(32)=ZNCOMB ! calculated in Zcomboset, called from zsearch
!   IALL(33)=NSSTAG !in zsearch
!   IALL(34)=NDISTLD  !in zload
!   IALL(35)=NDOFCLD  !in zload
!   IALL(36)=Z1MNSTP  !in zsearch
!   IALL(37)=Z1STPRN  !in zsearch
!   IALL(38)=Z1STTYP !in zsearch

  else !other nodes

   HPLOIE_hv= 0  !data
   HPHIIE_hv= 0  !data
   HPLNLS_hv= 0  !data--ic line entries
   HPCLST_hv= 0  !data--ic column entries 
   HPSATB_hv= 0  !data substr at base array
   HPSATE_hv= 0  !data substr at base array
   HPSALS_hv= 0  !data
   HPSIAB_hv= 0  !data
   HPSIAE_hv= 0  !data
   HPINIL_hv= 0  !data atom initialization list
!   HPLDSS_hv= 0  !loaded subspaces, in zload
   HPMINSS_hv= 0  !data
   HPIMISS_hv= 0 !data
   HPALRF_hv= 0 !data alias references
!   HPFLCI_hv= 0  !this is used internally (active atoms)--no communication
!   HPFLIN_hv= 0 internally, assigned in ZMERGE
!   HPDOIN_hv= 0 assigned in ZMERGE
!   HPFLINR_hv= 0
!   HPDOINR_hv= 0
   HPQICBF_hv= 0  !data assigned in ADDSUBST
   HPDISAB1_hv= 0 !data, distance atom constraints 
   HPDISAE1_hv= 0  !data, distance atom constraints
   HPDISAB2_hv= 0  !data, distance atom constraints
   HPDISAE2_hv= 0  !data, distance atom constraints
   HPDIATL_hv= 0 !data distance atom list
!   HPLDDIS_hv= 0 !loaded dist constraints, assigned in DISTCONSLOAD
   HPDFCMP_hv= 0 !data map of dof constraints ADDDOFCONS
!   HPLDDFC_hv= 0 ! loaded dof constraints: DOFCONSLOAD

   HPADDF_hv= 0 !data list of dof
   HPLODF_hv= 0  !data
   HPHIDF_hv= 0  !data
   HPLOCN_hv= 0  !data
   HPHICN_hv= 0  !data
   HPMDFL_hv= 0  !data (master dof list)
   HPALIA_hv= 0 !data alias assignment
   HPSSNM_hv= 0 !data, assigned in RDZCONF

   call cs_init(CSR,'INTEGER')
!   LODOFLW = 0 
!   HIDOFLW = 0
!   MSTDOFW = 0
!   LOCONFW = 0 !these two are just copies of LOCONF,HICONF
!   HICONFW = 0 ! inefficient, but done for consistency
  endif

  NIALL = 0

  call pack_iarr(IALL,NIALL,HPLOIE_hv,NTHDDF)
  call pack_iarr(IALL,NIALL,HPHIIE_hv,NTHDDF)
  call pack_iarr(IALL,NIALL,HPLNLS_hv,4*NICELST)
  call pack_iarr(IALL,NIALL,HPCLST_hv,4*NICELST)
  call pack_iarr(IALL,NIALL,HPSATB_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPSATE_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPSALS_hv,TSUBAT)
  call pack_iarr(IALL,NIALL,HPSIAB_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPSIAE_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPINIL_hv,TINITA)
!  call pack_iarr(IALL,NIALL,HPLDSS_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPMINSS_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPIMISS_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPALRF_hv,NSUBME)
!  call pack_iarr(IALL,NIALL,HPFLCI_hv,NATOM)
!  call pack_iarr(IALL,NIALL,HPFLIN_hv,NATOM)
!  call pack_iarr(IALL,NIALL,HPDOIN_hv,NATOM)
!  call pack_iarr(IALL,NIALL,HPFLINR_hv,NATOM)
!  call pack_iarr(IALL,NIALL,HPDOINR_hv,NATOM)
  call pack_iarr(IALL,NIALL,HPQICBF_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPDISAB1_hv,NDISTDE)
  call pack_iarr(IALL,NIALL,HPDISAE1_hv,NDISTDE)
  call pack_iarr(IALL,NIALL,HPDISAB2_hv,NDISTDE)
  call pack_iarr(IALL,NIALL,HPDISAE2_hv,NDISTDE)
  call pack_iarr(IALL,NIALL,HPDIATL_hv,TDISTAT)
!  call pack_iarr(IALL,NIALL,HPLDDIS_hv,NDISTME)
  call pack_iarr(IALL,NIALL,HPDFCMP_hv,NDOFCON)
!  call pack_iarr(IALL,NIALL,HPLDDFC_hv,NDOFCME)

!  WRITE(6,*) 'MYNODGP ',MYNODGP,' NATOM ',NATOM,' NSUBME ',NSUBME,' DDEFME ',DDEFME
!  WRITE(6,*) 'MYNODP ',MYNODP,' NSATME ',NSATME
 
! conformer read
  call pack_iarr(IALL,NIALL,HPADDF_hv,NALLDF)
!  call pack_iarr(IALL,NIALL,HPLODF_hv,NCNFME)
!  call pack_iarr(IALL,NIALL,HPHIDF_hv,NCNFME)
  call pack_iarr(IALL,NIALL,HPLODF_hv,NCONFO)
  call pack_iarr(IALL,NIALL,HPHIDF_hv,NCONFO)
  call pack_iarr(IALL,NIALL,HPLOCN_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPHICN_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPMDFL_hv,NMASTR)
  call pack_iarr(IALL,NIALL,HPALIA_hv,NSUBME)
  call pack_iarr(IALL,NIALL,HPSSNM_hv,NSUBME)
!  call pack_iarr(IALL,NIALL,CSR%LODOF,NCNFME)
!  call pack_iarr(IALL,NIALL,CSR%HIDOF,NCNFME)
  call pack_iarr(IALL,NIALL,CSR%LODOF,NCONFO)
  call pack_iarr(IALL,NIALL,CSR%HIDOF,NCONFO)
  call pack_iarr(IALL,NIALL,CSR%LOCNF,NSUBSP)
  call pack_iarr(IALL,NIALL,CSR%HICNF,NSUBSP)
  call pack_iarr(IALL,NIALL,CSR%MSTDF,NMASTRW)

!  wRITE(6,*) 'MYNODGP ',MYNODGP,' BEFORE ICGCOMB, NIALL is ',NIALL,' size(IALL) ',size(IALL)
!  wRITE(6,*) 'MYNODP ',MYNODP,' BEFORE ICGCOMB, NSUBSP is ',NSUBSP,' size(CSR%LOCNF) ',size(CSR%LOCNF), &
!            ' size(CSR%HICNF) ',size(CSR%HICNF)
  if(NIALL.ne.size(IALL)) then
   call parstoperr('<COMM_ZDATA>','incorrect integer array size')
  endif
   call MPI_BCAST(IALL,NIALL,MPI_INTEGER,NODE0,MPI_COMM_WORLD,ierr)
!   write(6,*) 'MYNODGP ',MYNODGP,' after second broadcast'
!  WRITE(6,*) 'MYNODP ',MYNODP,' AFTER ICGCOMB'

  NIALL = 0 

  call unpack_iarr(IALL,NIALL,HPLOIE_hv,NTHDDF)
  call unpack_iarr(IALL,NIALL,HPHIIE_hv,NTHDDF)
  call unpack_iarr(IALL,NIALL,HPLNLS_hv,4*NICELST)
  call unpack_iarr(IALL,NIALL,HPCLST_hv,4*NICELST)
  call unpack_iarr(IALL,NIALL,HPSATB_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPSATE_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPSALS_hv,TSUBAT)
  call unpack_iarr(IALL,NIALL,HPSIAB_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPSIAE_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPINIL_hv,TINITA)
!  call unpack_iarr(IALL,NIALL,HPLDSS_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPMINSS_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPIMISS_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPALRF_hv,NSUBME)
!  call unpack_iarr(IALL,NIALL,HPFLCI_hv,NATOM)
!  call unpack_iarr(IALL,NIALL,HPFLIN_hv,NATOM)
!  call unpack_iarr(IALL,NIALL,HPDOIN_hv,NATOM)
!  call unpack_iarr(IALL,NIALL,HPFLINR_hv,NATOM)
!  call unpack_iarr(IALL,NIALL,HPDOINR_hv,NATOM)
  call unpack_iarr(IALL,NIALL,HPQICBF_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPDISAB1_hv,NDISTDE)
  call unpack_iarr(IALL,NIALL,HPDISAE1_hv,NDISTDE)
  call unpack_iarr(IALL,NIALL,HPDISAB2_hv,NDISTDE)
  call unpack_iarr(IALL,NIALL,HPDISAE2_hv,NDISTDE)
  call unpack_iarr(IALL,NIALL,HPDIATL_hv,TDISTAT)
!  call unpack_iarr(IALL,NIALL,HPLDDIS_hv,NDISTME)
  call unpack_iarr(IALL,NIALL,HPDFCMP_hv,NDOFCON)
!  call unpack_iarr(IALL,NIALL,HPLDDFC_hv,NDOFCME)

! conformer read
  call unpack_iarr(IALL,NIALL,HPADDF_hv,NALLDF)
!  call unpack_iarr(IALL,NIALL,HPLODF_hv,NCNFME)
!  call unpack_iarr(IALL,NIALL,HPHIDF_hv,NCNFME)
  call unpack_iarr(IALL,NIALL,HPLODF_hv,NCONFO)
  call unpack_iarr(IALL,NIALL,HPHIDF_hv,NCONFO)
  call unpack_iarr(IALL,NIALL,HPLOCN_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPHICN_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPMDFL_hv,NMASTR)
  call unpack_iarr(IALL,NIALL,HPALIA_hv,NSUBME)
  call unpack_iarr(IALL,NIALL,HPSSNM_hv,NSUBME)
!  call unpack_iarr(IALL,NIALL,CSR%LODOF,NCNFME)
!  call unpack_iarr(IALL,NIALL,CSR%HIDOF,NCNFME)
  call unpack_iarr(IALL,NIALL,CSR%LODOF,NCONFO)
  call unpack_iarr(IALL,NIALL,CSR%HIDOF,NCONFO)
  call unpack_iarr(IALL,NIALL,CSR%LOCNF,NSUBSP)
  call unpack_iarr(IALL,NIALL,CSR%HICNF,NSUBSP)
  call unpack_iarr(IALL,NIALL,CSR%MSTDF,NMASTRW)
!---------------------------------------------------------------------------------
!---------------------------------------------------------------------------------
 if(QZTIME) call loc_timer('ZCOMM BEF LOGICLS')

! communicate logicals
  NQALL =2
  if(allocated(QALL)) call chmdealloc('zerom_comm.src','COMM_ZDATA','QALL',NQALL,log=QALL)
  call chmalloc('zerom_comm.src','COMM_ZDATA','QALL',NQALL,log=QALL)
  if(allocated(IALL)) call chmdealloc('zerom_comm.src','COMM_ZDATA','IALL',NQALL,intg=IALL)
  call chmalloc('zerom_comm.src','COMM_ZDATA','IALL',NQALL,intg=IALL)
  IALL = 0
  QALL = 0

  if(mynodgp.eq.1) then
!   QALL(1)=QZNCUT
!   QALL(2)=QRECUT
!   QALL(3)=QZBINS in zsearch
!   QALL(4)=QMISSG for reading conf
!   QALL(5)=QREDUN !only for reading conformers
!   QALL(6)=QWZCMP
!   QALL(7)=QZMINI
!   QALL(8)=QZVCUT
!   QALL(9)=QZCFIX
!   QALL(10)=QMINPRNT in zsearch
!   QALL(11)=QREDUSS in zsearch
!   QALL(12)=QZRMSD
!   QALL(13)=QZENERG in zsearch 
!   QALL(14)=QZORIEN
!   QALL(15)=QICBREV  !this is internal
   QALL(1)=QDOFCONS  !need to comm
   QALL(2)=QICFILL !need to comm
!   QALL(17)=QICCNA
!   QALL(18)=QZ1STMIN

   where (QALL)
    IALL = 1
   elsewhere
    IALL = 0
   end where

  endif !mynodgp.eq.1

!  call IGCOMB(IALL,NQALL)
  call MPI_BCAST(IALL,NQALL,MPI_INTEGER,NODE0,MPI_COMM_WORLD,ierr)

!  write(6,*) 'after logicals broadcast'
  if(MYNODGP.ne.1) then
   where (IALL==1)
    QALL = .true.
   elsewhere
    QALL = .false.
   end where
  
   QDOFCONS=QALL(1)
   QICFILL=QALL(2)
  endif
  
  call chmdealloc('zerom_comm.src','COMM_ZDATA','IALL',NQALL,intg=IALL)
  call chmdealloc('zerom_comm.src','COMM_ZDATA','QALL',NQALL,log=QALL)

!  WRITE(6,*) 'MYNODP ',MYNODP,' QDOFCONS ',QDOFCONS
!  WRITE(6,*) 'MYNODP ',MYNODP,' BEGIN COMM_ZDATA5'
!------------------------------------------
! communicate reals
  if(QZTIME) call loc_timer('ZCOMM BEF REALS')

  sizeall = NMASTR + NMASTRW + NTHDDF + NDISTDE*2 + NDOFCON*2 + NCONFO
  if(allocated(RALL)) call chmdealloc('zerom_comm.src','COMM_ZDATA','RALL',sizeall,crl=RALL)
  call chmalloc('zerom_comm.src','COMM_ZDATA','RALL',sizeall,crl=RALL)
  RALL = 0
  if(MYNODGP.eq.1) then
!   RALL(1) = ZENCUT  !in zsearch
!   RALL(2) = RECUTF !in zsearch
!   RALL(3) = ZBINSZ !in zsearch
!   RALL(4) = ZSVFRC !in zsearch
!   RALL(5) = ZVATOL !in zsearch
!!   RALL(6) = LDSSMNE !stores min energy in between zload commands (what to do?)
!   RALL(7) = Z1MSTP !in zsearch
  else !all other nodes
   HPMDFV_hv=0    !master val list-- need to comm
!   HPGMNV_hv= 0  !result--glob min dof vals for loaded ss
!   HPLDSMN_hv= 0 !result--vals for minimum over grid
   HPINVL_hv= 0 !data, set in ADDDOFDEF
   HPDISGAR_hv= 0  !distance constraints (ADDDISTCONS)
   HPDISLAR_hv= 0  !data  ADDDISTCONS
   HPDFCGAR_hv= 0  !data ADDDOFCONS
   HPDFCLAR_hv= 0  !data ADDDOFCONS
   call cs_init(CSR,'REAL')
  endif
 
  NRALL = 0
  call pack_rarr(RALL,NRALL,HPMDFV_hv,NMASTR)
!  call pack_rarr(RALL,NRALL,HPGMNV_hv,DDEFME)
!  call pack_rarr(RALL,NRALL,HPLDSMN_hv,DDEFME)
  call pack_rarr(RALL,NRALL,HPINVL_hv,NTHDDF)
  call pack_rarr(RALL,NRALL,HPDISGAR_hv,NDISTDE)
  call pack_rarr(RALL,NRALL,HPDISLAR_hv,NDISTDE)
  call pack_rarr(RALL,NRALL,HPDFCGAR_hv,NDOFCON)
  call pack_rarr(RALL,NRALL,HPDFCLAR_hv,NDOFCON)
  call pack_rarr(RALL,NRALL,CSR%MSTDV,NMASTRW) 
  call pack_rarr(RALL,NRALL,CSR%ENERG,NCONFO) 

  if(NRALL.ne.size(RALL)) then
   WRITE(6,*) ' NRALL is ',NRALL,' size(RALL) is ',size(RALL)
   WRITE(6,*) 'MYNODP ',MYNODP,' size(CSR%MSTDV) ',size(CSR%MSTDV),' NMASTRW ',NMASTRW
   call parstoperr('<COMM_ZDATA>','incorrect real array size')
  endif

  call MPI_BCAST(RALL,NRALL,MPI_DOUBLE_PRECISION,NODE0,MPI_COMM_WORLD,ierr)
!  call GCOMB(RALL,NRALL)

!  write(6,*) 'after reals broadcast'
  if(MYNODGP.ne.1) then 
!   ZENCUT = RALL(1)
!   RECUTF = RALL(2)
!   ZBINSZ = RALL(3)
!   ZSVFRC = RALL(4)
!   ZVATOL = RALL(5)
!   LDSSMNE = RALL(6)
!   Z1MSTP = RALL(7)

   NRALL =0 
   call unpack_rarr(RALL,NRALL,HPMDFV_hv,NMASTR)
!   call unpack_rarr(RALL,NRALL,HPGMNV_hv,DDEFME)
!   call unpack_rarr(RALL,NRALL,HPLDSMN_hv,DDEFME)
   call unpack_rarr(RALL,NRALL,HPINVL_hv,NTHDDF)
   call unpack_rarr(RALL,NRALL,HPDFCGAR_hv,NDOFCON)
   call unpack_rarr(RALL,NRALL,HPDFCLAR_hv,NDOFCON)
   call unpack_rarr(RALL,NRALL,HPDISGAR_hv,NDISTDE)
   call unpack_rarr(RALL,NRALL,HPDISLAR_hv,NDISTDE)
   call unpack_rarr(RALL,NRALL,CSR%MSTDV,NMASTRW) 
   call unpack_rarr(RALL,NRALL,CSR%ENERG,NCONFO) 
  endif

 goto 10001
  WRITE(10+mynodgp,*) 'QZNCUT=', QZNCUT
  WRITE(10+mynodgp,*) 'QRECUT=', QRECUT
  WRITE(10+mynodgp,*) 'QZBINS=', QZBINS
  WRITE(10+mynodgp,*) 'QMISSG=', QMISSG
  WRITE(10+mynodgp,*) 'QREDUN=', QREDUN
  WRITE(10+mynodgp,*) 'QWZCMP=', QWZCMP
  WRITE(10+mynodgp,*) 'QZMINI=', QZMINI
  WRITE(10+mynodgp,*) 'QZVCUT=', QZVCUT
  WRITE(10+mynodgp,*) 'QZCFIX=', QZCFIX
  WRITE(10+mynodgp,*) 'QMINPRNT=', QMINPRNT
  WRITE(10+mynodgp,*) 'QREDUSS=', QREDUSS
  WRITE(10+mynodgp,*) 'QZRMSD=', QZRMSD
  WRITE(10+mynodgp,*) 'QZENERG=', QZENERG
  WRITE(10+mynodgp,*) 'QZORIEN=', QZORIEN
  WRITE(10+mynodgp,*) 'QICBREV=', QICBREV
  WRITE(10+mynodgp,*) 'QDOFCONS=', QDOFCONS
  WRITE(10+mynodgp,*) 'QICCNA=', QICCNA
  WRITE(10+mynodgp,*) 'DDEFME=', DDEFME
  WRITE(10+mynodgp,*) 'NVALME=', NVALME
  WRITE(10+mynodgp,*) 'NCNFME=', NCNFME
  WRITE(10+mynodgp,*) 'NSUBME=', NSUBME
  WRITE(10+mynodgp,*) 'NSATME=', NSATME
  WRITE(10+mynodgp,*) 'NDISTME=', NDISTME
  WRITE(10+mynodgp,*) 'NDISATME=', NDISATME
  WRITE(10+mynodgp,*) 'NDOFCME=', NDOFCME
  WRITE(10+mynodgp,*) 'NZBINS=', NZBINS
  WRITE(10+mynodgp,*) 'ZBINSZ=', ZBINSZ
  WRITE(10+mynodgp,*) 'MXCNSZ=', MXCNSZ
  WRITE(10+mynodgp,*) 'NTHDDF=', NTHDDF
  WRITE(10+mynodgp,*) 'NICELST=', NICELST
  WRITE(10+mynodgp,*) 'NSSSLD=', NSSSLD
  WRITE(10+mynodgp,*) 'TSUBAT=', TSUBAT
  WRITE(10+mynodgp,*) 'TINITA=', TINITA
  WRITE(10+mynodgp,*) 'TDISTAT=', TDISTAT
  WRITE(10+mynodgp,*) 'NSUBDE=', NSUBDE
  WRITE(10+mynodgp,*) 'NDISTDE=', NDISTDE
  WRITE(10+mynodgp,*) 'NDOFCON=', NDOFCON
  WRITE(10+mynodgp,*) 'NALIAS=', NALIAS
  WRITE(10+mynodgp,*) 'RECUTF=', RECUTF
  WRITE(10+mynodgp,*) 'NCONFO=', NCONFO
  WRITE(10+mynodgp,*) 'NSUBSP=', NSUBSP
  WRITE(10+mynodgp,*) 'NMASTR=', NMASTR
  WRITE(10+mynodgp,*) 'NALLDF=', NALLDF
  WRITE(10+mynodgp,*) 'NMINORSS=', NMINORSS
  WRITE(10+mynodgp,*) 'ZTMINO=', ZTMINO
  WRITE(10+mynodgp,*) 'ZENCUT=', ZENCUT
  WRITE(10+mynodgp,*) 'ZVATOL=', ZVATOL
  WRITE(10+mynodgp,*) 'ZWRUNI=', ZWRUNI
  WRITE(10+mynodgp,*) 'ZWRTUNI=', ZWRTUNI
  WRITE(10+mynodgp,*) 'ZTNSTEP=', ZTNSTEP
  WRITE(10+mynodgp,*) 'ZMINUNI=', ZMINUNI
  WRITE(10+mynodgp,*) 'ZSTEPN=', ZSTEPN
  WRITE(10+mynodgp,*) 'ZSVFRC=', ZSVFRC
  WRITE(10+mynodgp,*) 'ZNCOMB=', ZNCOMB
  WRITE(10+mynodgp,*) 'NSSTAG=', NSSTAG
  WRITE(10+mynodgp,*) 'LDSSMNE=', LDSSMNE
  WRITE(10+mynodgp,*) 'NDISTLD=', NDISTLD
  WRITE(10+mynodgp,*) 'NDOFCLD=', NDOFCLD
  WRITE(10+mynodgp,*) 'Z1MNSTP=', Z1MNSTP
  WRITE(10+mynodgp,*) 'Z1STPRN=', Z1STPRN
  WRITE(10+mynodgp,*) 'Z1STTYP=', Z1STTYP
  WRITE(10+mynodgp,*) 'QZ1STMIN=', QZ1STMIN
  WRITE(10+mynodgp,*) 'Z1MSTP=', Z1MSTP
10001 continue

 call chmdealloc('zerom_comm.src','COMM_ZDATA','RALL',sizeall,crl=RALL)
!call parstoperr('XXXX','stop')
!  call cs_copy(CSR,CSW)  !copy conformer data to working copy
!  QZCOMM = .false.
  if(QZTIME) call loc_timer('ZCOMM END')

  end subroutine COMM_ZDATA

#endif /* (parallel_comm)*/

  subroutine pack_iarr(RNIARR,RNCNT,IARR,CNT)
  integer,dimension(:),intent(inout) :: RNIARR !larger array
  integer,intent(inout) :: RNCNT !running count over larger array
  integer,dimension(:),intent(in) :: IARR !smaller array being packed into larger
  integer,intent(in) :: CNT !how many elements of small array to pack
! local
  integer :: II
  
  do II = 1,CNT
   RNCNT = RNCNT + 1
   RNIARR(RNCNT) = IARR(II)  
!   WRITE(MYNODP+10,*) 'MYNODP ',MYNODP,' II ',II,' RNCNT ',RNCNT,' RNIARR ',RNIARR(RNCNT)
  enddo 
  
  end subroutine pack_iarr

  subroutine unpack_iarr(RNIARR,RNCNT,IARR,CNT)
  integer,dimension(:),intent(in) :: RNIARR
  integer,dimension(:),intent(inout) :: IARR
  integer,intent(inout) :: RNCNT
  integer,intent(in) :: CNT
! local
  integer :: II

  do II = 1,CNT
   IARR(II) = RNIARR(II+RNCNT)
!  WRITE(MYNODP+20,*) 'AFTER MYNODP ',MYNODP,' II ',II,' IARR ',IARR(II),' RUNNING ',II+RNCNT,' RNIARR ',RNIARR(II+RNCNT)
  enddo
  RNCNT = CNT + RNCNT

  end subroutine unpack_iarr

  subroutine pack_rarr(RNRARR,RNCNT,RARR,CNT)
  real(chm_real),dimension(:),intent(inout) :: RNRARR
  real(chm_real),dimension(:),intent(in) :: RARR
  integer,intent(inout) :: RNCNT
  integer,intent(in) :: CNT
! local
  integer :: II

  do II = 1,CNT
   RNCNT = RNCNT + 1
   RNRARR(RNCNT) = RARR(II)
!   WRITE(MYNODP+10,*) 'MYNODP ',MYNODP,' II ',II,' RNCNT ',RNCNT,' RNRARR ',RNRARR(RNCNT)
  enddo

  end subroutine pack_rarr

  subroutine unpack_rarr(RNRARR,RNCNT,RARR,CNT)
  real(chm_real),dimension(:),intent(in) :: RNRARR
  real(chm_real),dimension(:),intent(inout) :: RARR
  integer,intent(inout) :: RNCNT
  integer,intent(in) :: CNT
! local
  integer :: II

  do II = 1,CNT
   RARR(II) = RNRARR(II+RNCNT)
!  WRITE(MYNODP+20,*) 'AFTER MYNODP ',MYNODP,' II ',II,' RARR ',RARR(II),' RUNNING ',II+RNCNT,' RNRARR ',RNRARR(II+RNCNT)
  enddo
  RNCNT = CNT + RNCNT

  end subroutine unpack_rarr
!-----------------------------------------------------------------------------------------------
  subroutine COMBIN_RESLT
#if KEY_PARALLEL==1 /*pararesult*/
  use chm_kinds
  use parallel
  use zdata_mod
  use cpustruc,only: CPUIDENT,CPUONES,HEADCOM,NUMHOOD,IMYKEYP,QCPUONES
  use asynccomg
  use memory
!  use nbndcc_utilb
!  use psf,only: NATOM
  implicit none

#if KEY_PARALLEL==1
  integer,allocatable,dimension(:) :: NCONFSS 
#endif
  integer,allocatable,dimension(:) :: BEGPT,ENDPT
  integer,allocatable,dimension(:) :: BEGINME
  integer,allocatable,dimension(:,:) :: BEGCONF
  logical :: QALLPROCP
  integer :: COUNT,COUNT1,COUNT2,COUNT3
  integer(chm_int4),allocatable,dimension(:),save :: LOCSHAND,LOCRHAND
  type(arofar_i4),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
  type(arofar),allocatable,dimension(:),save :: LOCSBUFR,LOCRBUFR
  integer :: WORKICNT,WORKRCNT
  integer,allocatable,dimension(:) :: WORKIAR
  real(chm_real),allocatable,dimension(:,:) :: WORKRAR

  integer :: NCPUSEND,NCPURECV,RUNNING
  integer,allocatable,dimension(:) :: CPURECV  !number of cpus in this group (not including me)
  integer :: MYINTEGER,NSECT,RESSIZ
  integer,allocatable,dimension(:) :: OTHERSINT,RECVMNY,RECVHI,RECVLST
  real(chm_real),allocatable,dimension(:,:) :: RECVLSTR
  integer,dimension(1) :: CPUSEND,SENDLST,SENDHI,SENDMNY
  integer,dimension(:),allocatable :: SENDLST2
  integer :: MYUNIT,SUM,TEST,BEG,END,NODE !temporary
  integer :: II,JJ,LASTCNF
  integer ::NUMNODES,MYHOOD,MYLOCRNKP
  integer(chm_int4) :: ALLCOMM
  include 'mpif.h'

!---------------------------------
! send size to head node

  ALLCOMM = MPI_COMM_WORLD
  NUMNODES = NUMNOD
!  MYHOOD = MYNODP
  MYLOCRNKP = MYNODP
  if(NUMHOOD.GT.1) then
   ALLCOMM = HEADCOM
   NUMNODES = NUMHOOD
!   MYHOOD = IMYNHDP
   MYLOCRNKP = IMYKEYP
  endif
  
  if(NUMHOOD.eq.1) return
  if((MYLOCRNKP.NE.1).and.NUMHOOD.gt.1) return  !no communication necessary if not head node when neighborhoods 

  if(QVERBOSE) write(6,*) 'entering combin:  mynodp ',MYNODP,' numnod ',numnod,' numnodes ',numnodes
  MYINTEGER = OUTLNCNT 
  if(MYNODP.eq.1) then  !node 0
    if(allocated(CPURECV)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','CPURECV',NUMNODES,intg=CPURECV)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','CPURECV',NUMNODES,intg=CPURECV)
    if(allocated(LOCRHAND)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','LOCRHAND',NUMNODES,intg=LOCRHAND)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','LOCRHAND',NUMNODES,intg=LOCRHAND)
    if(allocated(RECVMNY)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','RECVMNY',NUMNODES,intg=RECVMNY)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','RECVMNY',NUMNODES,intg=RECVMNY)
    if(allocated(RECVHI)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','RECVHI',NUMNODES,intg=RECVHI)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','RECVHI',NUMNODES,intg=RECVHI)
    RECVHI = 0
    RECVMNY = 0
    CPURECV = 0
    LOCRHAND = 0
    NCPURECV = NUMNODES-1
    do II = 1,NUMNODES-1
     CPURECV(II) = II + 1
    enddo
    NCPUSEND = 0
  else  !not node 1
    if(allocated(CPURECV)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','CPURECV',1,intg=CPURECV)  !dummy array
    call chmalloc('zerom_comm.src','COMBIN_RESLT','CPURECV',1,intg=CPURECV)
    if(allocated(LOCSHAND)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','LOCSHAND',NUMNODES,intg=LOCSHAND)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','LOCSHAND',NUMNODES,intg=LOCSHAND)
    CPURECV = 0
    LOCSHAND=0
    NCPURECV = 0
    NCPUSEND = 1
    CPUSEND(1)=1
    SENDLST(1)=MYINTEGER
    SENDHI(1)=1
    SENDMNY(1)=1
  endif
   
  if(QVERBOSE) write(6,*) 'combin2 :  mynodp ',MYNODP,' numnod ',numnod,' numnodes ',numnodes
!  call parstoperr('<COMBIN_RESLT>','before asynclst')
  if(MYNODP.eq.1) then  !just node 0
    if(qverbose) WRITE(6,*) 'ALLCOMM is ',ALLCOMM,' QCPUONES is ',QCPUONES
    call ASYNCLST_RG(NCPURECV,CPURECV,CPUONES,  &
      LOCRHAND,LOCRBUF,plabint=1,PCCATOR=ALLCOMM)
    if(qverbose) WRITE(6,*) 'ALLCOMM after RG is ',ALLCOMM
    call ASYNCLST_WRG(NCPURECV,CPURECV,RECVMNY,CPUIDENT,CPUONES,LOCRHAND, &
      pribuf=LOCRBUF,plabint=2) 
  else
    if(qverbose)  WRITE(6,*) '>>CHECKMYNODP ',MYNODP,' SENDHI(1)',SENDHI(1),' SENDMNY(1) ',SENDMNY(1)
    call ASYNCLST_SG(NCPUSEND,CPUSEND,SENDLST,SENDHI,SENDMNY,  &
     LOCSHAND,LOCSBUF,plabint=3,PCCATOR=ALLCOMM)
    call ASYNCLST_WSG(NCPUSEND,CPUSEND,SENDMNY,LOCSHAND,psibuf=LOCSBUF)
  endif
   
  if(QVERBOSE) write(6,*) 'combin3 :  mynodp ',MYNODP,' numnod ',numnod
  RUNNING = 0
  if(MYNODP.eq.1) then
    NCPURECV = 0
    CPURECV=0
    do II = 1,NUMNODES
     if(RECVMNY(II).GT.0) then
      RUNNING = RUNNING + RECVMNY(II)*3  !since 3 arrays each for integers and reals
      NCPURECV = NCPURECV + 1
      CPURECV(NCPURECV) = II
      RECVHI(II) = RUNNING 
      RECVMNY(II) = RECVMNY(II)*3  !since 3 arrays
     if(QVERBOSE) WRITE(6,*) '>>CHECKMYNODP 1 RECEIVING ',RECVMNY(II),' FROM NODE ',II,' RUNNING IS ',RUNNING
     endif
!     WRITE(6,*) 'MYNODP is ',MYNODP,' set to receive ',RECVMNY(II)*3,' FROM CPU ',II,' OUTLNCNT'
    enddo
  endif

  if(QVERBOSE) write(6,*) 'combin4 :  mynodp ',MYNODP,' numnod ',numnod
! now communicate data
!                NSSTAG_AR,NEWCONF_AR,MSTDOF_AR,CURVAL_AR,MYPOTEN_AR,RMST1_AR

  if(MYNODP.eq.1) then
!   WRITE(6,*) 'MYNODP ',MYNODP,' **RUNNING BEFORE RECVLST',RUNNING,' allocated(RECVLST) ',allocated(RECVLST)
   if(allocated(RECVLST)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','RECVLST',RUNNING,intg=RECVLST)
   call chmalloc('zerom_comm.src','COMBIN_RESLT','RECVLST',RUNNING,intg=RECVLST)

!   WRITE(6,*) 'MYNODP ',MYNODP,' RUNNING ',RUNNING,' allocated(RECVLSTR) ',allocated(RECVLSTR)

   if(allocated(RECVLSTR)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','RECVLSTR',1,RUNNING,crl=RECVLSTR)
   call chmalloc('zerom_comm.src','COMBIN_RESLT','RECVLSTR',1,RUNNING,crl=RECVLSTR)

!     WRITE(6,*) 'NCPURECV here2 is ',NCPURECV,' CPURECV(1) ',CPURECV(1)
!    WRITE(6,*) 'MYNODP ',MYNODP,' SIZE OF RECVLST ',size(RECVLST),' RUNNING ',RUNNING
    RECVLST = -1
  else  !mynodp ne 1

    if(allocated(WORKIAR)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','WORKIAR',OUTLNCNT*3,intg=WORKIAR)
    if(allocated(WORKRAR)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','WORKRAR',1,siz2=OUTLNCNT*3,crl=WORKRAR)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','WORKIAR',OUTLNCNT*3,intg=WORKIAR)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','WORKRAR',1,siz2=OUTLNCNT*3,crl=WORKRAR)

    WORKIAR = 0
    WORKICNT = 0
    WORKRAR = 0
    WORKRCNT = 0

    SUM = 0
!integer arrays
    do II = 1,OUTLNCNT
     WORKICNT = WORKICNT + 1
     WORKIAR(WORKICNT) = NSSTAG_AR(II) 
     SUM = SUM + WORKIAR(WORKICNT)
    enddo
 
    do II = 1,OUTLNCNT
     WORKICNT = WORKICNT + 1
     WORKIAR(WORKICNT) = NEWCONF_AR(II)
     SUM = SUM + WORKIAR(WORKICNT)
    enddo

    do II = 1,OUTLNCNT
     WORKICNT = WORKICNT + 1
     WORKIAR(WORKICNT) = MSTDOF_AR(II)
     SUM = SUM + WORKIAR(WORKICNT)
    enddo

!     do II = 1,OUTLNCNT*3
!     WRITE(6,*) 'MYNODP ',MYNODP,' sending ',WORKIAR(II)
!     enddo
!     WRITE(6,*) 'NCPURECV3 is ',NCPURECV,' CPURECV(1) ',CPURECV(1)
!    TEST = SUM
!    call GCOMB(TEST,1)
!    WRITE(6,*) '>>>GLOBAL count of all WORKIAR vals is ',TEST
    if(allocated(SENDLST2)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','SENDLST2',OUTLNCNT*3,intg=SENDLST2)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','SENDLST2',OUTLNCNT*3,intg=SENDLST2)

!     WRITE(6,*) 'NCPURECV4 is ',NCPURECV,' CPURECV(1) ',CPURECV(1)
! real arrays
    do II = 1,OUTLNCNT
     WORKRCNT = WORKRCNT + 1
     WORKRAR(1,WORKRCNT) =CURVAL_AR(II)
     SENDLST2(WORKRCNT) = WORKRCNT
    enddo

    do II = 1,OUTLNCNT
     WORKRCNT = WORKRCNT + 1
     WORKRAR(1,WORKRCNT) =MYPOTEN_AR(II)
     SENDLST2(WORKRCNT) = WORKRCNT
    enddo

    do II = 1,OUTLNCNT
     WORKRCNT = WORKRCNT + 1
     WORKRAR(1,WORKRCNT) =RMST1_AR(II)
     SENDLST2(WORKRCNT) = WORKRCNT
    enddo
  endif !if mynodp = 1 or not
  if(QZTIME) call loc_timer('COMBIN JUST BEF INT')

  if(QVERBOSE) write(6,*) 'combin5 :  mynodp ',MYNODP,' numnod ',numnod
!integer data
  if(MYNODP.eq.1) then
! post receives
!     WRITE(6,*) 'NCPURECV5 is ',NCPURECV,' CPURECV(1) ',CPURECV(1)
     call ASYNCLST_RG(NCPURECV,CPURECV,RECVMNY,  &
      LOCRHAND,LOCRBUF,plabint=1,PCCATOR=ALLCOMM)
     call ASYNCLST_WRG(NCPURECV,CPURECV,RECVLST,RECVHI,RECVMNY,LOCRHAND, &
      pribuf=LOCRBUF,plabint=2)
  else
! post sends
     if(OUTLNCNT.GT.0) then
!      WRITE(6,*) 'MYNODP ',MYNODP,' NCPUSEND ',NCPUSEND
!      WRITE(6,*) 'MYNODP ',MYNODP,' SENDHI(1)',SENDHI(1),' SENDMNY(1) ',SENDMNY(1)
      SENDHI(1) = OUTLNCNT*3
      SENDMNY(1) = OUTLNCNT*3
!      if(MYNODP.eq.NUMNOD.or. MYNODP.eq.1) then
!      WRITE(6,*) 'MYNODP ',MYNODP,' SENDING ',SENDMNY(1),' TO NODE ',CPUSEND(1)
!      endif
      call ASYNCLST_SG(NCPUSEND,CPUSEND,WORKIAR,SENDHI,SENDMNY,  &
       LOCSHAND,LOCSBUF,plabint=3,PCCATOR=ALLCOMM)
      call ASYNCLST_WSG(NCPUSEND,CPUSEND,SENDMNY,LOCSHAND,psibuf=LOCSBUF)
     endif
  endif

  if(QVERBOSE) write(6,*) 'combin6 :  mynodp ',MYNODP,' numnod ',numnod
  SUM = 0
  if(MYNODP.eq.1) then
    if(QWRITE) WRITE(6,*) 'RUNNING total of lines received is ',RUNNING,' OUTLNCNT',OUTLNCNT
!     do II = 1,RUNNING
!      WRITE(6,*) '>>>RECEIVED data II= ',II,RECVLST(II),' OUTLNCNT '
!     enddo
!     WRITE(6,*) '>>>RECEIVED total data ',SUM,' OUTLNCNT '

    CONFLLEN = int(RUNNING/3)
    if(QWRITE)  WRITE(6,*) 'CONFLLEN of arrays is ',CONFLLEN,' OUTLNCNT'
    if(allocated(NSSTAG_GLB)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','NSSTAG_GLB',CONFLLEN,intg=NSSTAG_GLB)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','NSSTAG_GLB',CONFLLEN,intg=NSSTAG_GLB)
    if(allocated(NEWCONF_GLB)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','NEWCONF_GLB',CONFLLEN,intg=NEWCONF_GLB)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','NEWCONF_GLB',CONFLLEN,intg=NEWCONF_GLB)
    if(allocated(MSTDOF_GLB)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','MSTDOF_GLB',CONFLLEN,intg=MSTDOF_GLB)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','MSTDOF_GLB',CONFLLEN,intg=MSTDOF_GLB)
    NSSTAG_GLB=0
    NEWCONF_GLB=0
    MSTDOF_GLB=0
    WORKICNT = 0
    LASTCNF = 0
    do II = 1,NCPURECV
      NODE = CPURECV(II)
      END = INT(RECVHI(NODE)/3)
      NSECT = INT(RECVMNY(NODE)/3)
      BEG = END - NSECT + 1
!      WRITE(6,*) 'BEG is ',BEG,' END is ',END,' RUNNING ',RUNNING
      do JJ = BEG,END
       WORKICNT=WORKICNT+1
       NSSTAG_GLB(JJ) = RECVLST(WORKICNT) 
!      WRITE(6,*) 'JJ ',JJ,' NSSTAG_GLB ',NSSTAG_GLB(JJ),'WORKICNT1 ',WORKICNT,' RECVLST ',RECVLST(WORKICNT)
      enddo
      do JJ = BEG,END
       WORKICNT=WORKICNT+1
       NEWCONF_GLB(JJ) = RECVLST(WORKICNT) + LASTCNF 
!      WRITE(6,*) 'WORKICNT2 ',WORKICNT,' RECVLST ',RECVLST(WORKICNT)
      enddo
      do JJ = BEG,END
       WORKICNT=WORKICNT+1
       MSTDOF_GLB(JJ) = RECVLST(WORKICNT) 
!      WRITE(6,*) 'WORKICNT3 ',WORKICNT,' RECVLST ',RECVLST(WORKICNT)
      enddo
      LASTCNF = NEWCONF_GLB(END)
    enddo
    RESSIZ = size(NSSTAG_GLB)
!    WRITE(6,*) 'size of result is ',RESSIZ,' RUNNING IS ',RUNNING
!    do II = 1,RESSIZ
!     WRITE(6,*) 'RESULTS: II ',II, NSSTAG_GLB(II),NEWCONF_GLB(II),MSTDOF_GLB(II) 
!    enddo
  endif
  if(QVERBOSE) write(6,*) 'combin7 :  mynodp ',MYNODP,' numnod ',numnod
!-------------------------------------------------------------------
   
  if(QZTIME) call loc_timer('COMBIN JUST BEF REALS')

! now real data
  if(MYNODP.eq.1) then
    do II = 1,RUNNING
     RECVLST(II) = II
    enddo
    call ASYNCLST_RG(NCPURECV,CPURECV,RECVMNY,  &
     LOCRHAND,prrbuf=LOCRBUFR,plabint=1,PCCATOR=ALLCOMM)
!     WRITE(6,*) 'MYNODP ',MYNODP,' NCPURECV ',NCPURECV
!     WRITE(6,*) 'MYNODP ',MYNODP,' RECVHI(1)',RECVHI(1),' RECVMNY(1) ',RECVMNY(1)
!     WRITE(6,*) 'MYNODP ',MYNODP,' RECVHI(2)',RECVHI(2),' RECVMNY(2) ',RECVMNY(2)
!     WRITE(6,*) 'MYNODP ',MYNODP,' RECVHI(3)',RECVHI(3),' RECVMNY(3) ',RECVMNY(3)
!     WRITE(6,*) 'MYNODP ',MYNODP,' RECVHI(4)',RECVHI(4),' RECVMNY(4) ',RECVMNY(4)
!    write(6,*) 'MYNODP ',MYNODP,' sizes: CPURECV ',size(CPURECV),' RECVLST ',size(RECVLST), &
!    'RECVHI ',size(RECVHI),' RECVMNY ',size(RECVMNY)
!    write(6,*) 'MYNODP ',MYNODP,' sizes: LOCRHAND ',size(LOCRHAND),' LOCRBUFR ',size(LOCRBUFR), &
!    ' RECVLSTR ',size(RECVLSTR)
!     call parstoperr('<here>','before WRG ') 
    call ASYNCLST_WRG(NCPURECV,CPURECV,RECVLST,RECVHI,RECVMNY,LOCRHAND, &
     prrbuf=LOCRBUFR,prdata=RECVLSTR,plabint=2)
  else !mynodp ne 1
    if(OUTLNCNT.GT.0) then
     SENDHI(1) = OUTLNCNT*3
     SENDMNY(1) = OUTLNCNT*3
!     WRITE(6,*) 'MYNODP ',MYNODP,' NCPUSEND ',NCPUSEND
!     WRITE(6,*) 'MYNODP ',MYNODP,' SENDHI(1)',SENDHI(1),' SENDMNY(1) ',SENDMNY(1)
!     WRITE(6,*) 'MYNODP ',MYNODP,' SENDING ',SENDMNY(1),' TO NODE ',CPUSEND(1)
     call ASYNCLST_SG(NCPUSEND,CPUSEND,SENDLST2,SENDHI,SENDMNY,  &
      LOCSHAND,psrbuf=LOCSBUFR,PRDATA=WORKRAR,plabint=3,PCCATOR=ALLCOMM)
     call ASYNCLST_WSG(NCPUSEND,CPUSEND,SENDMNY,LOCSHAND,psrbuf=LOCSBUFR)
    endif
  endif
    
  if(QVERBOSE) write(6,*) 'combin8 :  mynodp ',MYNODP,' numnod ',numnod
  if(MYNODP.eq.1) then
    CONFLLEN = int(RUNNING/3)
!    WRITE(6,*) 'CONFLLEN of arrays is ',CONFLLEN
    if(allocated(CURVAL_GLB)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','CURVAL_GLB',CONFLLEN,crl=CURVAL_GLB)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','CURVAL_GLB',CONFLLEN,crl=CURVAL_GLB)
    if(allocated(MYPOTEN_GLB)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','MYPOTEN_GLB',CONFLLEN,crl=MYPOTEN_GLB)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','MYPOTEN_GLB',CONFLLEN,crl=MYPOTEN_GLB)
    if(allocated(RMST1_GLB)) call chmdealloc('zerom_comm.src','COMBIN_RESLT','RMST1_GLB',CONFLLEN,crl=RMST1_GLB)
    call chmalloc('zerom_comm.src','COMBIN_RESLT','RMST1_GLB',CONFLLEN,crl=RMST1_GLB)
    CURVAL_GLB=0
    MYPOTEN_GLB=0
    RMST1_GLB=0
    WORKRCNT = 0
    do II = 1,NCPURECV
      NODE = CPURECV(II)
      END = INT(RECVHI(NODE)/3)
      NSECT = INT(RECVMNY(NODE)/3)
      BEG = END - NSECT + 1
!      WRITE(6,*) 'BEG is ',BEG,' END is ',END,' RUNNING ',RUNNING
      do JJ = BEG,END
       WORKRCNT=WORKRCNT+1
       CURVAL_GLB(JJ) = RECVLSTR(1,WORKRCNT)
!      WRITE(6,*) 'JJ ',JJ,' CURVAL_GLB ',CURVAL_GLB(JJ),'WORKRCNT1 ',WORKRCNT,' RECVLSTR ',RECVLSTR(1,WORKRCNT)
      enddo
      do JJ = BEG,END
       WORKRCNT=WORKRCNT+1
       MYPOTEN_GLB(JJ) = RECVLSTR(1,WORKRCNT)
!      WRITE(6,*) 'WORKRCNT2 ',WORKRCNT,' RECVLSTR ',RECVLSTR(1,WORKRCNT)
      enddo
      do JJ = BEG,END
       WORKRCNT=WORKRCNT+1
       RMST1_GLB(JJ) = RECVLSTR(1,WORKRCNT)
!      WRITE(6,*) 'WORKRCNT3 ',WORKRCNT,' RECVLSTR ',RECVLSTR(1,WORKRCNT)
      enddo
    enddo
    RESSIZ = size(CURVAL_GLB)
!    WRITE(6,*) 'size of result is ',RESSIZ,' RUNNING IS ',RUNNING
!    do II = 1,RESSIZ
!     WRITE(6,*) 'RESULTS: II ',II, CURVAL_GLB(II),MYPOTEN_GLB(II),RMST1_GLB(II)
!    enddo
  endif

  if(QVERBOSE) write(6,*) 'combin9 :  mynodp ',MYNODP,' numnod ',numnod
goto 9991
  if(MYNODP.eq.1) then
!    do II = 1,RUNNING
!      WRITE(6,*) '>>MYNODP ',MYNODP,' RECEIVED ', RECVLSTR(1,II)
!    enddo
!    COUNT = 0
    do II = 1,NCPURECV
     NODE = CPURECV(II)
     END = RECVHI(NODE)  
     BEG = END - RECVMNY(NODE) + 1
     do JJ = BEG,END
      COUNT = COUNT + 1
!      WRITE(6,*) 'FR CPU ',NODE,' RECEIVED ',COUNT,' VAL ',RECVLSTR(1,COUNT)
     enddo
    enddo
  endif
9991 continue

#endif /* (pararesult)*/
 end subroutine COMBIN_RESLT

#endif 
!-------------------------------------------------------------------------------------------------
   end module zmod_comm
