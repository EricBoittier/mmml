#if KEY_SASAE==1
SUBROUTINE SASINI(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This routine initializes SASA.
  !        See source/sasa/sasa.f90 for a description of variables.
  !
  !     Author: Urs Haberthuer.

  use chm_kinds
  use chm_types
  use memory
  use dimens_fcm
  use number
  use bases_fcm
  use consta
  use coord
  use eutil
  use inbnd
  use psf
  use sasa
  use select
  use string

  implicit none

  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN

  !     FNW  - Number of elements in the SASWNB array before cleaning.
  !     SNW  - Number of elements in the SASWNB array after cleaning.
  !     FNI  - Number of elements in the SASINB array before cleaning.
  !     SNI  - Number of elements in the SASINB array after cleaning.
  !     FNX  - Number of elements in the    XNB array before cleaning.
  !     SNX  - Number of elements in the    XNB array after cleaning.
  !     PTBL - Analogous to PSASWBL. See source/sasa/sasa.f90.
  !     PTNB - Analogous to PSASWNB. See source/sasa/sasa.f90.
  !               The TBL/TNB pair list is only a temporary pair list
  !            helping to build the other pair lists.
  !     PXBL - Analogous to PSASWBL. See source/sasa/sasa.f90.
  !     PXNB - Analogous to PSASWNB. See source/sasa/sasa.f90.
  !               The XBL/XNB pair list is a copy of the fixed exclusion
  !            pair list. Depending on the nonbond exclusion mode, these
  !            pairs may have to be excluded from (a copy of) the nonbond
  !            exclusion pair list in order to build the SASIBL/SASINB
  !            pair list. The XBL/XNB pair list is only a temporary pair
  !            list. See the subroutine SASDIN for more details.

  integer,allocatable,dimension(:) :: PXBL
  integer,allocatable,dimension(:) :: PXNB
  integer,allocatable,dimension(:) :: PTBL
  integer,allocatable,dimension(:) :: PTNB

  INTEGER       FNW,SNW,FNI,SNI,FNX,SNX
  INTEGER       I,J
  CHARACTER(len=4) ATT(31)
  CHARACTER(len=1) DUMM

  real(chm_real) RNUMT

  !     If SASINI is not called for the first time, free everything.

  IF (QSASA) THEN
     QSASA  = .FALSE.
     QINFX  = .FALSE.
     QSRFC  = .FALSE.
     QNEWP  = .FALSE.
     call chmdealloc('sasini.src','SASINI','PSASLCT',SASNAT,intg=PSASLCT)
     call chmdealloc('sasini.src','SASINI','PSASIDX',SASNAT,intg=PSASIDX)
     call chmdealloc('sasini.src','SASINI','PSASWBL',SASNAT,intg=PSASWBL)
     call chmdealloc('sasini.src','SASINI','PSASWNB',SASSNW,intg=PSASWNB)
     call chmdealloc('sasini.src','SASINI','PSASIBL',SASNAT,intg=PSASIBL)
     call chmdealloc('sasini.src','SASINI','PSASINB',SASSNI,intg=PSASINB)
     call chmdealloc('sasini.src','SASINI','PSASACS',SASNAT,crl=PSASACS)
  ENDIF

  !     Make sure that all the lists are filled and up to date.

  CALL GETE0('ENER',DUMM,0)

  !     Set the flag for SASA.

  QSASA     =.TRUE.

  !     Get the flag for including the fixed exclusions.

  QINFX     =(INDXA(COMLYN,COMLEN,'INFX') .GT. 0)

  !     Get the flag for storing the solvent accessible surface areas in
  !     WMAIN.

  QSRFC     =(INDXA(COMLYN,COMLEN,'SURF') .GT. 0)

  !     Get the flag to determine which parameter set is to be used.

  QNEWP     =(INDXA(COMLYN,COMLEN,'NEWP') .GT. 0)

  IF (.NOT. QNEWP) THEN

     !     Get the surface-tension like solvation parameters and assign them
     !     to the CHARMM atom types.

     SASSIG(1 )=GTRMF(COMLYN,COMLEN,'S001',ZERO)
     SASSIG(2 )=GTRMF(COMLYN,COMLEN,'S002',ZERO)
     SASSIG(3 )=GTRMF(COMLYN,COMLEN,'S003',FMARK)
     SASSIG(4 )=GTRMF(COMLYN,COMLEN,'S004',FMARK)
     SASSIG(5 )=GTRMF(COMLYN,COMLEN,'S005',FMARK)
     SASSIG(6 )=GTRMF(COMLYN,COMLEN,'S006',FMARK)
     RNUMT=0.0120D0
     SASSIG(7 )=GTRMF(COMLYN,COMLEN,'S007',RNUMT)
     SASSIG(8 )=GTRMF(COMLYN,COMLEN,'S008',RNUMT)
     SASSIG(9 )=GTRMF(COMLYN,COMLEN,'S009',RNUMT)
     SASSIG(10)=GTRMF(COMLYN,COMLEN,'S010',RNUMT)
     SASSIG(11)=GTRMF(COMLYN,COMLEN,'S011',RNUMT)
     SASSIG(12)=GTRMF(COMLYN,COMLEN,'S012',FMARK)
     SASSIG(13)=GTRMF(COMLYN,COMLEN,'S013',RNUMT)
     RNUMT=-0.0600D0
     SASSIG(14)=GTRMF(COMLYN,COMLEN,'S014',RNUMT)
     SASSIG(15)=GTRMF(COMLYN,COMLEN,'S015',RNUMT)
     SASSIG(16)=GTRMF(COMLYN,COMLEN,'S016',FMARK)
     SASSIG(17)=GTRMF(COMLYN,COMLEN,'S017',RNUMT)
     SASSIG(18)=GTRMF(COMLYN,COMLEN,'S018',RNUMT)
     SASSIG(19)=GTRMF(COMLYN,COMLEN,'S019',RNUMT)
     SASSIG(20)=GTRMF(COMLYN,COMLEN,'S020',RNUMT)
     SASSIG(21)=GTRMF(COMLYN,COMLEN,'S021',RNUMT)
     SASSIG(22)=GTRMF(COMLYN,COMLEN,'S022',RNUMT)
     SASSIG(23)=GTRMF(COMLYN,COMLEN,'S023',RNUMT)
     SASSIG(24)=GTRMF(COMLYN,COMLEN,'S024',FMARK)
     SASSIG(25)=GTRMF(COMLYN,COMLEN,'S025',FMARK)
     SASSIG(26)=GTRMF(COMLYN,COMLEN,'S026',FMARK)
     SASSIG(27)=GTRMF(COMLYN,COMLEN,'S027',FMARK)
     RNUMT=0.0120D0
     SASSIG(28)=GTRMF(COMLYN,COMLEN,'S028',RNUMT)
     SASSIG(29)=GTRMF(COMLYN,COMLEN,'S029',RNUMT)
     SASSIG(30)=GTRMF(COMLYN,COMLEN,'S030',FMARK)
     SASSIG(31)=GTRMF(COMLYN,COMLEN,'S031',RNUMT)

     !     Get the radii and assign them to the CHARMM atom types.

     RNUMT=1.1000D0
     SASROV(1 )=GTRMF(COMLYN,COMLEN,'R001',RNUMT)
     SASROV(2 )=GTRMF(COMLYN,COMLEN,'R002',RNUMT)
     SASROV(3 )=GTRMF(COMLYN,COMLEN,'R003',FMARK)
     SASROV(4 )=GTRMF(COMLYN,COMLEN,'R004',FMARK)
     SASROV(5 )=GTRMF(COMLYN,COMLEN,'R005',FMARK)
     SASROV(6 )=GTRMF(COMLYN,COMLEN,'R006',FMARK)
     RNUMT=1.7200D0
     SASROV(7 )=GTRMF(COMLYN,COMLEN,'R007',RNUMT)
     RNUMT=1.8000D0
     SASROV(8 )=GTRMF(COMLYN,COMLEN,'R008',RNUMT)
     RNUMT=1.9000D0
     SASROV(9 )=GTRMF(COMLYN,COMLEN,'R009',RNUMT)
     RNUMT=TWO
     SASROV(10)=GTRMF(COMLYN,COMLEN,'R010',RNUMT)
     RNUMT=1.8000D0
     SASROV(11)=GTRMF(COMLYN,COMLEN,'R011',RNUMT)
     SASROV(12)=GTRMF(COMLYN,COMLEN,'R012',FMARK)
     RNUMT=1.8000D0
     SASROV(13)=GTRMF(COMLYN,COMLEN,'R013',RNUMT)
     RNUMT=1.5500D0
     SASROV(14)=GTRMF(COMLYN,COMLEN,'R014',RNUMT)
     SASROV(15)=GTRMF(COMLYN,COMLEN,'R015',RNUMT)
     SASROV(16)=GTRMF(COMLYN,COMLEN,'R016',FMARK)
     SASROV(17)=GTRMF(COMLYN,COMLEN,'R017',RNUMT)
     RNUMT=1.6000D0
     SASROV(18)=GTRMF(COMLYN,COMLEN,'R018',RNUMT)
     SASROV(19)=GTRMF(COMLYN,COMLEN,'R019',RNUMT)
     SASROV(20)=GTRMF(COMLYN,COMLEN,'R020',RNUMT)
     RNUMT=1.5000D0
     SASROV(21)=GTRMF(COMLYN,COMLEN,'R021',RNUMT)
     RNUMT=1.7000D0
     SASROV(22)=GTRMF(COMLYN,COMLEN,'R022',RNUMT)
     RNUMT=1.5200D0
     SASROV(23)=GTRMF(COMLYN,COMLEN,'R023',RNUMT)
     SASROV(24)=GTRMF(COMLYN,COMLEN,'R024',FMARK)
     SASROV(25)=GTRMF(COMLYN,COMLEN,'R025',FMARK)
     SASROV(26)=GTRMF(COMLYN,COMLEN,'R026',FMARK)
     SASROV(27)=GTRMF(COMLYN,COMLEN,'R027',FMARK)
     RNUMT=1.8000D0
     SASROV(28)=GTRMF(COMLYN,COMLEN,'R028',RNUMT)
     SASROV(29)=GTRMF(COMLYN,COMLEN,'R029',RNUMT)
     SASROV(30)=GTRMF(COMLYN,COMLEN,'R030',FMARK)
     RNUMT=1.7200D0
     SASROV(31)=GTRMF(COMLYN,COMLEN,'R031',RNUMT)

     !     Get the probabilistic parameters and assign them to the CHARMM
     !     atom types.

     RNUMT=1.1280D0
     SASPAT(1 )=GTRMF(COMLYN,COMLEN,'P001',RNUMT)
     SASPAT(2 )=GTRMF(COMLYN,COMLEN,'P002',RNUMT)
     SASPAT(3 )=GTRMF(COMLYN,COMLEN,'P003',FMARK)
     SASPAT(4 )=GTRMF(COMLYN,COMLEN,'P004',FMARK)
     SASPAT(5 )=GTRMF(COMLYN,COMLEN,'P005',FMARK)
     SASPAT(6 )=GTRMF(COMLYN,COMLEN,'P006',FMARK)
     RNUMT=1.5540D0
     SASPAT(7 )=GTRMF(COMLYN,COMLEN,'P007',RNUMT)
     RNUMT=1.2760D0
     SASPAT(8 )=GTRMF(COMLYN,COMLEN,'P008',RNUMT)
     RNUMT=1.0450D0
     SASPAT(9 )=GTRMF(COMLYN,COMLEN,'P009',RNUMT)
     RNUMT=0.8800D0
     SASPAT(10)=GTRMF(COMLYN,COMLEN,'P010',RNUMT)
     RNUMT=1.0730D0
     SASPAT(11)=GTRMF(COMLYN,COMLEN,'P011',RNUMT)
     SASPAT(12)=GTRMF(COMLYN,COMLEN,'P012',FMARK)
     RNUMT=1.2760D0
     SASPAT(13)=GTRMF(COMLYN,COMLEN,'P013',RNUMT)
     RNUMT=1.0280D0
     SASPAT(14)=GTRMF(COMLYN,COMLEN,'P014',RNUMT)
     SASPAT(15)=GTRMF(COMLYN,COMLEN,'P015',RNUMT)
     SASPAT(16)=GTRMF(COMLYN,COMLEN,'P016',FMARK)
     SASPAT(17)=GTRMF(COMLYN,COMLEN,'P017',RNUMT)
     RNUMT=1.2150D0
     SASPAT(18)=GTRMF(COMLYN,COMLEN,'P018',RNUMT)
     SASPAT(19)=GTRMF(COMLYN,COMLEN,'P019',RNUMT)
     SASPAT(20)=GTRMF(COMLYN,COMLEN,'P020',RNUMT)
     RNUMT=0.9260D0
     SASPAT(21)=GTRMF(COMLYN,COMLEN,'P021',RNUMT)
     RNUMT=0.9220D0
     SASPAT(22)=GTRMF(COMLYN,COMLEN,'P022',RNUMT)
     RNUMT=1.0800D0
     SASPAT(23)=GTRMF(COMLYN,COMLEN,'P023',RNUMT)
     SASPAT(24)=GTRMF(COMLYN,COMLEN,'P024',FMARK)
     SASPAT(25)=GTRMF(COMLYN,COMLEN,'P025',FMARK)
     SASPAT(26)=GTRMF(COMLYN,COMLEN,'P026',FMARK)
     SASPAT(27)=GTRMF(COMLYN,COMLEN,'P027',FMARK)
     RNUMT=1.1210D0
     SASPAT(28)=GTRMF(COMLYN,COMLEN,'P028',RNUMT)
     SASPAT(29)=GTRMF(COMLYN,COMLEN,'P029',RNUMT)
     SASPAT(30)=GTRMF(COMLYN,COMLEN,'P030',FMARK)
     RNUMT=1.5540D0
     SASPAT(31)=GTRMF(COMLYN,COMLEN,'P031',RNUMT)

     !     Get the connectivity parameters.

     RNUMT=0.8875D0
     SASFCO    =GTRMF(COMLYN,COMLEN,'FCON',RNUMT)
     RNUMT=0.3516D0
     SASNCO    =GTRMF(COMLYN,COMLEN,'NCON',RNUMT)

  ELSE

     !     Get the surface-tension like solvation parameters and assign them
     !     to the CHARMM atom types.

     SASSIG(1 )=GTRMF(COMLYN,COMLEN,'S001',ZERO)
     SASSIG(2 )=GTRMF(COMLYN,COMLEN,'S002',ZERO)
     SASSIG(3 )=GTRMF(COMLYN,COMLEN,'S003',FMARK)
     SASSIG(4 )=GTRMF(COMLYN,COMLEN,'S004',FMARK)
     SASSIG(5 )=GTRMF(COMLYN,COMLEN,'S005',FMARK)
     SASSIG(6 )=GTRMF(COMLYN,COMLEN,'S006',FMARK)
     RNUMT=0.0120D0
     SASSIG(7 )=GTRMF(COMLYN,COMLEN,'S007',RNUMT)
     SASSIG(8 )=GTRMF(COMLYN,COMLEN,'S008',RNUMT)
     SASSIG(9 )=GTRMF(COMLYN,COMLEN,'S009',RNUMT)
     SASSIG(10)=GTRMF(COMLYN,COMLEN,'S010',RNUMT)
     SASSIG(11)=GTRMF(COMLYN,COMLEN,'S011',RNUMT)
     SASSIG(12)=GTRMF(COMLYN,COMLEN,'S012',FMARK)
     SASSIG(13)=GTRMF(COMLYN,COMLEN,'S013',RNUMT)
     RNUMT=-0.0600D0
     SASSIG(14)=GTRMF(COMLYN,COMLEN,'S014',RNUMT)
     SASSIG(15)=GTRMF(COMLYN,COMLEN,'S015',RNUMT)
     SASSIG(16)=GTRMF(COMLYN,COMLEN,'S016',FMARK)
     SASSIG(17)=GTRMF(COMLYN,COMLEN,'S017',RNUMT)
     SASSIG(18)=GTRMF(COMLYN,COMLEN,'S018',RNUMT)
     SASSIG(19)=GTRMF(COMLYN,COMLEN,'S019',RNUMT)
     SASSIG(20)=GTRMF(COMLYN,COMLEN,'S020',RNUMT)
     SASSIG(21)=GTRMF(COMLYN,COMLEN,'S021',RNUMT)
     SASSIG(22)=GTRMF(COMLYN,COMLEN,'S022',RNUMT)
     SASSIG(23)=GTRMF(COMLYN,COMLEN,'S023',RNUMT)
     SASSIG(24)=GTRMF(COMLYN,COMLEN,'S024',FMARK)
     SASSIG(25)=GTRMF(COMLYN,COMLEN,'S025',FMARK)
     SASSIG(26)=GTRMF(COMLYN,COMLEN,'S026',FMARK)
     SASSIG(27)=GTRMF(COMLYN,COMLEN,'S027',FMARK)
     RNUMT=0.0120D0
     SASSIG(28)=GTRMF(COMLYN,COMLEN,'S028',RNUMT)
     SASSIG(29)=GTRMF(COMLYN,COMLEN,'S029',RNUMT)
     SASSIG(30)=GTRMF(COMLYN,COMLEN,'S030',FMARK)
     SASSIG(31)=GTRMF(COMLYN,COMLEN,'S031',RNUMT)

     !     Get the radii and assign them to the CHARMM atom types.

     RNUMT=0.100001D0
     SASROV(1 )=GTRMF(COMLYN,COMLEN,'R001',RNUMT)
     RNUMT=0.491498D0
     SASROV(2 )=GTRMF(COMLYN,COMLEN,'R002',RNUMT)
     SASROV(3 )=GTRMF(COMLYN,COMLEN,'R003',FMARK)
     SASROV(4 )=GTRMF(COMLYN,COMLEN,'R004',FMARK)
     SASROV(5 )=GTRMF(COMLYN,COMLEN,'R005',FMARK)
     SASROV(6 )=GTRMF(COMLYN,COMLEN,'R006',FMARK)
     RNUMT=1.369460D0
     SASROV(7 )=GTRMF(COMLYN,COMLEN,'R007',RNUMT)
     RNUMT=1.895900D0
     SASROV(8 )=GTRMF(COMLYN,COMLEN,'R008',RNUMT)
     RNUMT=2.286480D0
     SASROV(9 )=GTRMF(COMLYN,COMLEN,'R009',RNUMT)
     RNUMT=2.651430D0
     SASROV(10)=GTRMF(COMLYN,COMLEN,'R010',RNUMT)
     RNUMT=2.075670D0
     SASROV(11)=GTRMF(COMLYN,COMLEN,'R011',RNUMT)
     SASROV(12)=GTRMF(COMLYN,COMLEN,'R012',FMARK)
     RNUMT=1.895900D0
     SASROV(13)=GTRMF(COMLYN,COMLEN,'R013',RNUMT)
     RNUMT=0.323677D0
     SASROV(14)=GTRMF(COMLYN,COMLEN,'R014',RNUMT)
     RNUMT=1.390860D0
     SASROV(15)=GTRMF(COMLYN,COMLEN,'R015',RNUMT)
     SASROV(16)=GTRMF(COMLYN,COMLEN,'R016',FMARK)
     RNUMT=0.100001D0
     SASROV(17)=GTRMF(COMLYN,COMLEN,'R017',RNUMT)
     RNUMT=1.469600D0
     SASROV(18)=GTRMF(COMLYN,COMLEN,'R018',RNUMT)
     RNUMT=1.554210D0
     SASROV(19)=GTRMF(COMLYN,COMLEN,'R019',RNUMT)
     RNUMT=1.552220D0
     SASROV(20)=GTRMF(COMLYN,COMLEN,'R020',RNUMT)
     RNUMT=1.681850D0
     SASROV(21)=GTRMF(COMLYN,COMLEN,'R021',RNUMT)
     RNUMT=1.652130D0
     SASROV(22)=GTRMF(COMLYN,COMLEN,'R022',RNUMT)
     RNUMT=1.548960D0
     SASROV(23)=GTRMF(COMLYN,COMLEN,'R023',RNUMT)
     SASROV(24)=GTRMF(COMLYN,COMLEN,'R024',FMARK)
     SASROV(25)=GTRMF(COMLYN,COMLEN,'R025',FMARK)
     SASROV(26)=GTRMF(COMLYN,COMLEN,'R026',FMARK)
     SASROV(27)=GTRMF(COMLYN,COMLEN,'R027',FMARK)
     RNUMT=2.188920D0
     SASROV(28)=GTRMF(COMLYN,COMLEN,'R028',RNUMT)
     RNUMT=1.878290D0
     SASROV(29)=GTRMF(COMLYN,COMLEN,'R029',RNUMT)
     SASROV(30)=GTRMF(COMLYN,COMLEN,'R030',FMARK)
     RNUMT=1.578670D0
     SASROV(31)=GTRMF(COMLYN,COMLEN,'R031',RNUMT)

     !     Get the probabilistic parameters and assign them to the CHARMM
     !     atom types.

     RNUMT=0.988957D0
     SASPAT(1 )=GTRMF(COMLYN,COMLEN,'P001',RNUMT)
     RNUMT=1.777570D0
     SASPAT(2 )=GTRMF(COMLYN,COMLEN,'P002',RNUMT)
     SASPAT(3 )=GTRMF(COMLYN,COMLEN,'P003',FMARK)
     SASPAT(4 )=GTRMF(COMLYN,COMLEN,'P004',FMARK)
     SASPAT(5 )=GTRMF(COMLYN,COMLEN,'P005',FMARK)
     SASPAT(6 )=GTRMF(COMLYN,COMLEN,'P006',FMARK)
     RNUMT=1.304200D0
     SASPAT(7 )=GTRMF(COMLYN,COMLEN,'P007',RNUMT)
     RNUMT=1.316430D0
     SASPAT(8 )=GTRMF(COMLYN,COMLEN,'P008',RNUMT)
     RNUMT=1.182780D0
     SASPAT(9 )=GTRMF(COMLYN,COMLEN,'P009',RNUMT)
     RNUMT=1.083020D0
     SASPAT(10)=GTRMF(COMLYN,COMLEN,'P010',RNUMT)
     RNUMT=1.134370D0
     SASPAT(11)=GTRMF(COMLYN,COMLEN,'P011',RNUMT)
     SASPAT(12)=GTRMF(COMLYN,COMLEN,'P012',FMARK)
     RNUMT=1.316430D0
     SASPAT(13)=GTRMF(COMLYN,COMLEN,'P013',RNUMT)
     RNUMT=1.130220D0
     SASPAT(14)=GTRMF(COMLYN,COMLEN,'P014',RNUMT)
     RNUMT=1.510700D0
     SASPAT(15)=GTRMF(COMLYN,COMLEN,'P015',RNUMT)
     SASPAT(16)=GTRMF(COMLYN,COMLEN,'P016',FMARK)
     RNUMT=1.592510D0
     SASPAT(17)=GTRMF(COMLYN,COMLEN,'P017',RNUMT)
     RNUMT=1.261620D0
     SASPAT(18)=GTRMF(COMLYN,COMLEN,'P018',RNUMT)
     RNUMT=1.187100D0
     SASPAT(19)=GTRMF(COMLYN,COMLEN,'P019',RNUMT)
     RNUMT=1.053870D0
     SASPAT(20)=GTRMF(COMLYN,COMLEN,'P020',RNUMT)
     RNUMT=1.036120D0
     SASPAT(21)=GTRMF(COMLYN,COMLEN,'P021',RNUMT)
     RNUMT=1.139990D0
     SASPAT(22)=GTRMF(COMLYN,COMLEN,'P022',RNUMT)
     RNUMT=1.096650D0
     SASPAT(23)=GTRMF(COMLYN,COMLEN,'P023',RNUMT)
     SASPAT(24)=GTRMF(COMLYN,COMLEN,'P024',FMARK)
     SASPAT(25)=GTRMF(COMLYN,COMLEN,'P025',FMARK)
     SASPAT(26)=GTRMF(COMLYN,COMLEN,'P026',FMARK)
     SASPAT(27)=GTRMF(COMLYN,COMLEN,'P027',FMARK)
     RNUMT=1.680130D0
     SASPAT(28)=GTRMF(COMLYN,COMLEN,'P028',RNUMT)
     RNUMT=0.907302D0
     SASPAT(29)=GTRMF(COMLYN,COMLEN,'P029',RNUMT)
     SASPAT(30)=GTRMF(COMLYN,COMLEN,'P030',FMARK)
     RNUMT=1.276250D0
     SASPAT(31)=GTRMF(COMLYN,COMLEN,'P031',RNUMT)

     !     Get the connectivity parameters.

     RNUMT=0.342979D0
     SASFCO    =GTRMF(COMLYN,COMLEN,'FCON',RNUMT)
     RNUMT=0.507482D0
     SASNCO    =GTRMF(COMLYN,COMLEN,'NCON',RNUMT)

  ENDIF

  !     CHARMM atom types supported by SASA.

  ATT(1 )   ='H   '
  ATT(2 )   ='HC  '
  ATT(3 )   ='HA  '
  ATT(4 )   ='HT  '
  ATT(5 )   ='LP  '
  ATT(6 )   ='CT  '
  ATT(7 )   ='C   '
  ATT(8 )   ='CH1E'
  ATT(9 )   ='CH2E'
  ATT(10)   ='CH3E'
  ATT(11)   ='CR1E'
  ATT(12)   ='CM  '
  ATT(13)   ='C1ES'
  ATT(14)   ='N   '
  ATT(15)   ='NR  '
  ATT(16)   ='NP  '
  ATT(17)   ='NH1 '
  ATT(18)   ='NH2 '
  ATT(19)   ='NH3 '
  ATT(20)   ='NC2 '
  ATT(21)   ='O   '
  ATT(22)   ='OC  '
  ATT(23)   ='OH1 '
  ATT(24)   ='OH2 '
  ATT(25)   ='OM  '
  ATT(26)   ='OT  '
  ATT(27)   ='OS  '
  ATT(28)   ='S   '
  ATT(29)   ='SH1E'
  ATT(30)   ='FE  '
  ATT(31)   ='CR  '

  !     Actual number of atoms.

  SASNAT    =NATOM

  !     Solvent probe radius.

  SASRSP    =1.4D0

  !     Create a small look-up table.

  DO I=1,SASNPR
     SASRSS(I)=SASROV(I)+SASRSP
     SASASS(I)=FOUR*PI*SASRSS(I)**2
  ENDDO

  DO I=1,SASNPR
     DO J=1,SASNPR
        SASSUM(I,J)=SASRSS(I)+SASRSS(J)
        SASDIF(I,J)=SASROV(I)-SASROV(J)
        SASPRD(I,J)=SASSUM(I,J)*SASDIF(I,J)
     ENDDO
  ENDDO

  !     Allocate space on the heap.

  call chmalloc('sasini.src','SASINI','PSASLCT',NATOM,intg=PSASLCT)
  call chmalloc('sasini.src','SASINI','PSASIDX',NATOM,intg=PSASIDX)
  call chmalloc('sasini.src','SASINI','PSASACS',NATOM,crl=PSASACS)

  !     Get the selected atoms. Only these atoms are considered in the
  !     calculation. All the other atoms are treated as if not existent.

  call SELCTA(COMLYN,COMLEN,PSASLCT,X,Y,Z,WMAIN,.TRUE.)
  !     Assign to every atom the SASA index.

  call SASIND(NATOM,IAC,PSASLCT,PSASIDX,SASSIG,SASROV,SASPAT)

  !     Make a list of all 1-2 pairs. Use exactly as much memory as
  !     necessary. This list is permanent.

  FNW=NBOND

  call chmalloc('sasini.src','SASINI','PTBL',NATOM,intg=PTBL)
  call chmalloc('sasini.src','SASINI','PTNB',FNW,intg=PTNB)

  call SASDWW(NBOND,IB,JB,NATOM,SNW,PTBL,PTNB)

  call SASSRT(NATOM,SNW,PTBL,PTNB)

  call chmalloc('sasini.src','SASINI','PSASWBL',NATOM,intg=PSASWBL)
  call chmalloc('sasini.src','SASINI','PSASWNB',SNW,intg=PSASWNB)

  call SASCPL(NATOM,PTBL,PTNB,PSASWBL,PSASWNB)

  call chmdealloc('sasini.src','SASINI','PTBL',NATOM,intg=PTBL)
  call chmdealloc('sasini.src','SASINI','PTNB',FNW,intg=PTNB)

  SASSNW=SNW

  !     Make a copy of the fixed exclusion pair list and make sure that it
  !     is well ordered. Use exactly as much memory as necessary. This
  !     list is only temporary.

  FNX=IBLO(NATOM)

  call chmalloc('sasini.src','SASINI','PTBL',NATOM,intg=PTBL)
  call chmalloc('sasini.src','SASINI','PTNB',FNX,intg=PTNB)

  call SASCPL(NATOM,IBLO,INB,PTBL,PTNB)

  call SASSRT(NATOM,SNX,PTBL,PTNB)

  call chmalloc('sasini.src','SASINI','PXBL',NATOM,intg=PXBL)
  call chmalloc('sasini.src','SASINI','PXNB',SNX,intg=PXNB)

  call SASCPL(NATOM,PTBL,PTNB,PXBL,PXNB)

  call chmdealloc('sasini.src','SASINI','PTBL',NATOM,intg=PTBL)
  call chmdealloc('sasini.src','SASINI','PTNB',FNX,intg=PTNB)

  !     Make a list of all pairs that SASA needs in addition to the pairs
  !     in the nonbond pair list. Use exactly as much memory as necessary.
  !     This list is permanent.

  CALL SASCIN(NATOM,FNI,BNBND%IBLO14,BNBND%INB14)

  call chmalloc('sasini.src','SASINI','PTBL',NATOM,intg=PTBL)
  call chmalloc('sasini.src','SASINI','PTNB',FNI,intg=PTNB)

  call SASCPL(NATOM,BNBND%IBLO14,BNBND%INB14,PTBL,PTNB)

  call SASSRT(NATOM,SNI,PTBL,PTNB)

  call SASDIN(NBXMOD,NATOM,SNI,PXBL,PXNB,PTBL,PTNB,QINFX)

  call SASSRT(NATOM,SNI,PTBL,PTNB)

  call chmalloc('sasini.src','SASINI','PSASIBL',NATOM,intg=PSASIBL)
  call chmalloc('sasini.src','SASINI','PSASINB',SNI,intg=PSASINB)

  call SASCPL(NATOM,PTBL,PTNB,PSASIBL,PSASINB)

  call chmdealloc('sasini.src','SASINI','PTBL',NATOM,intg=PTBL)
  call chmdealloc('sasini.src','SASINI','PTNB',FNI,intg=PTNB)

  SASSNI=SNI

  !     Free the temporary list.

  call chmdealloc('sasini.src','SASINI','PXBL',NATOM,intg=PXBL)
  call chmdealloc('sasini.src','SASINI','PXNB',SNX,intg=PXNB)

  !     Print some data.

  call SASPRT(NATOM,SASNPR,PSASIBL,PSASINB,SASSIG,SASROV,SASPAT, &
       SASFCO,SASNCO,QINFX,QSRFC,ATT)

  !     Print all the energy and force terms, now including SASA.

  CALL GETE0('ENER',DUMM,0)

  RETURN
END SUBROUTINE SASINI

SUBROUTINE SASIND(NAT, &
     IAC, &
     SASLCT, &
     SASIDX, &
     SASSIG,SASROV,SASPAT)

  !     This routine assigns to every atom the SASA index. If an atom has
  !     not been selected for SASA by the user, its SASA index is set to
  !     0.
  !
  !     Author: Urs Haberthuer. The assignements were checked for errors
  !             by comparing them with the assignements done by a perl
  !             script written by Joannis Apostolakis.

  use chm_kinds
  use number
  use stream
  implicit none

  INTEGER     NAT
  INTEGER     IAC(*)
  INTEGER     SASLCT(*)
  INTEGER     SASIDX(*)
  real(chm_real)      SASSIG(*),SASROV(*),SASPAT(*)


  INTEGER     CON(92)
  INTEGER     I,J
  LOGICAL     ERROR

  !     Initialize list for converting CHARMM atom type numbers to SASA
  !     indices.

  DO I=1,92
     CON(I)=0
  ENDDO

  !     List for converting CHARMM atom type numbers to SASA indices.

  CON(1 )=1
  CON(2 )=2
  CON(3 )=3
  CON(4 )=4
  CON(5 )=5
  CON(10)=6
  CON(11)=7
  CON(12)=8
  CON(13)=9
  CON(14)=10
  CON(15)=11
  CON(16)=12
  CON(17)=13
  CON(31)=14
  CON(32)=15
  CON(33)=16
  CON(38)=17
  CON(39)=18
  CON(40)=19
  CON(41)=20
  CON(51)=21
  CON(52)=22
  CON(55)=23
  CON(56)=24
  CON(57)=25
  CON(58)=26
  CON(59)=27
  CON(81)=28
  CON(82)=29
  CON(91)=30
  CON(92)=31

  !     Flag for errors.

  ERROR=.FALSE.

  !     Go through all the atoms. If there is any CHARMM atom type for
  !     which one or more SASA parameters are missing, the routine will
  !     terminate with an error message.

  DO I=1,NAT
     IF (SASLCT(I) .NE. 0) THEN
        IF (CON(IAC(I)) .EQ. 0) THEN
           WRITE(OUTU,'(A50,I8)') &
                ' SASIND> Error: No SASA parameters for atom number',I
           ERROR=.TRUE.
        ENDIF
        SASIDX(I)=CON(IAC(I))
        IF ((SASSIG(SASIDX(I)) .EQ. FMARK) .OR. &
             (SASROV(SASIDX(I)) .EQ. FMARK) .OR. &
             (SASPAT(SASIDX(I)) .EQ. FMARK)) THEN
           WRITE(OUTU,'(A50,I8)') &
                ' SASIND> Error: No SASA parameters for atom number',I
           ERROR=.TRUE.
        ENDIF
     ELSE
        SASIDX(I)=0
     ENDIF
  ENDDO

  IF (ERROR) THEN
     CALL WRNDIE(-4,'<SASIND>','Missing SASA parameter(s).')
     RETURN
  ENDIF

  RETURN
END SUBROUTINE SASIND

SUBROUTINE SASDWW(NBO, &
     IB,JB, &
     NAT,NNB, &
     IBL,INB)

  !     This routine sets up the pair list of all 1-2 pairs.
  !
  !     Author: Urs Haberthuer.

  use chm_kinds
  implicit none

  INTEGER NBO
  INTEGER IB(*),JB(*)
  INTEGER NAT,NNB
  INTEGER IBL(*),INB(*)

  INTEGER I,J

  NNB=0
  DO I=1,NAT
     IBL(I)=NNB
     DO J=1,NBO
        IF (IB(J) .EQ. I) THEN
           NNB=NNB+1
           INB(NNB)=JB(J)
           IBL(I)=NNB
        ENDIF
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE SASDWW

SUBROUTINE SASCIN(NAT,NNB, &
     IBL,INB)

  !     This routine gets the number of pairs of a pair list that is
  !     indexed in the same way as is, for instance, the nonbond pair
  !     list. The result is stored in the variable NNB.
  !
  !     Author: Urs Haberthuer.

  use chm_kinds
  implicit none

  INTEGER NAT,NNB
  INTEGER IBL(*),INB(*)

  NNB=IBL(NAT)

  RETURN
END SUBROUTINE SASCIN

SUBROUTINE SASDIN(NBX, &
     NAT,NNB, &
     XBL,XNB, &
     SASIBL,SASINB, &
     QINFX)

  !     This routine sets up the pair list of all pairs that SASA needs
  !     in addition to the pairs in the nonbond pair list.
  !        The routine takes a copy of the nonbond exclusion pair list.
  !     It then removes the fixed exclusion pairs from this list if the
  !     user does not set the QINFX flag and if the nonbond exclusion
  !     mode is positive. Finally, it removes the special 1-4 pairs if
  !     there are any.
  !
  !     Author: Urs Haberthuer.

  use chm_kinds
  implicit none

  INTEGER NBX
  INTEGER NAT,NNB
  INTEGER XBL(*),XNB(*)
  INTEGER SASIBL(*),SASINB(*)
  LOGICAL QINFX

  INTEGER N
  INTEGER A,B,C,D
  INTEGER I,J,K,L

  !     Remove the fixed exclusion pairs from the copy of the nonbond
  !     exclusion pair list if the QINFX flag is set to FALSE and if the
  !     nonbond exclusion mode is positive.

  IF (.NOT. QINFX) THEN
     IF (NBX .GE. 1) THEN
        DO I=1,NAT
           IF (I .EQ. 1) THEN
              A=1
              C=1
           ELSE
              A=SASIBL(I-1)+1
              C=XBL(I-1)+1
           ENDIF
           B=SASIBL(I)
           D=XBL(I)
           J=A
           DO WHILE (J .LE. B)
              N=ABS(SASINB(J))
              K=C
              DO WHILE ((K .LE. D) .AND. (N .GT. ABS(XNB(K))))
                 K=K+1
              ENDDO
              IF       ((K .LE. D) .AND. (N .EQ. ABS(XNB(K)))) THEN
                 DO L=J,SASIBL(NAT)-1
                    SASINB(L)=SASINB(L+1)
                 ENDDO
                 DO L=I,NAT
                    SASIBL(L)=SASIBL(L)-1
                 ENDDO
                 B=B-1
                 J=J-1
              ENDIF
              J=J+1
           ENDDO
        ENDDO
     ENDIF
  ENDIF

  !     Remove the special 1-4 pairs if there are any. Special 1-4 pairs
  !     are 1-4 pairs of which the number of their second atom is
  !     negatively entered in the nonbond exclusion pair list. These
  !     pairs are actually added to the nonbond pair list although they
  !     are also in the nonbond exclusion pair list.

  DO I=1,NAT
     IF (I .EQ. 1) THEN
        A=1
     ELSE
        A=SASIBL(I-1)+1
     ENDIF
     B=SASIBL(I)
     J=A
     DO WHILE (J .LE. B)
        IF (SASINB(J) .LT. 0) THEN
           DO L=J,SASIBL(NAT)-1
              SASINB(L)=SASINB(L+1)
           ENDDO
           DO L=I,NAT
              SASIBL(L)=SASIBL(L)-1
           ENDDO
           B=B-1
           J=J-1
        ENDIF
        J=J+1
     ENDDO
  ENDDO

  NNB=SASIBL(NAT)

  RETURN
END SUBROUTINE SASDIN

SUBROUTINE SASCPL(NAT, &
     ORIIBL,ORIINB, &
     COPIBL,COPINB)

  !     This routine copies lists.
  !
  !     Author: Urs Haberthuer.

  use chm_kinds
  implicit none

  INTEGER NAT
  INTEGER ORIIBL(*),ORIINB(*)
  INTEGER COPIBL(*),COPINB(*)

  INTEGER I

  DO I=1,NAT
     COPIBL(I)=ORIIBL(I)
  ENDDO

  DO I=1,ORIIBL(NAT)
     COPINB(I)=ORIINB(I)
  ENDDO

  RETURN
END SUBROUTINE SASCPL

SUBROUTINE SASSRT(NAT,NNB, &
     IBL,INB)

  !     This routine cleans up a pair list that is indexed in the same way
  !     as is, for instance, the nonbond pair list.
  !        The first thing it does is arranging the list such that the
  !     number of the first atom in any pair is smaller than the absolute
  !     value of the number of the second atom in the same pair.
  !        It then sorts the list according to increasing absolute value
  !     of the number of the second atom in each pair.
  !        Finally, the routine eliminates multiple entries.
  !
  !     Author: Urs Haberthuer.

  use chm_kinds
  implicit none

  INTEGER NAT,NNB
  INTEGER IBL(*),INB(*)

  INTEGER SGN,TMP
  INTEGER A,B,C,D
  INTEGER I,J,K

  !     Arrange the list such that the number of the first atom in any
  !     pair is smaller than the absolute value of the number of the
  !     second atom in the same pair.

  DO I=1,NAT
     IF (I .EQ. 1) THEN
        A=1
     ELSE
        A=IBL(I-1)+1
     ENDIF
     B=IBL(I)
     DO J=A,B
        IF (ABS(INB(J)) .LT. I) THEN
           IF (INB(J) .GT. 0) THEN
              TMP=INB(J)
              SGN=1
           ELSE
              TMP=-INB(J)
              SGN=-1
           ENDIF
           IF (TMP .EQ. 1) THEN
              C=2
           ELSE
              C=IBL(TMP-1)+2
           ENDIF
           D=J
           DO K=D,C,-1
              INB(K)=INB(K-1)
           ENDDO
           INB(C-1)=SGN*I
           DO K=TMP,I-1
              IBL(K)=IBL(K)+1
           ENDDO
        ENDIF
     ENDDO
  ENDDO

  !     Sort the list according to increasing absolute value of the number
  !     of the second atom in each pair.

  DO I=1,NAT
     IF (I .EQ. 1) THEN
        A=1
     ELSE
        A=IBL(I-1)+1
     ENDIF
     B=IBL(I)
     DO J=A,B
        DO K=J+1,B
           IF (ABS(INB(J)) .GT. ABS(INB(K))) THEN
              TMP=INB(J)
              INB(J)=INB(K)
              INB(K)=TMP
           ENDIF
        ENDDO
     ENDDO
  ENDDO

  !     Eliminate multiple entries.

  DO I=1,NAT
     IF (I .EQ. 1) THEN
        A=1
     ELSE
        A=IBL(I-1)+1
     ENDIF
     B=IBL(I)
     J=A
     DO WHILE (J .LT. B)
        IF (INB(J) .EQ. INB(J+1)) THEN
           DO K=J+1,IBL(NAT)-1
              INB(K)=INB(K+1)
           ENDDO
           DO K=I,NAT
              IBL(K)=IBL(K)-1
           ENDDO
           B=B-1
           J=J-1
        ENDIF
        J=J+1
     ENDDO
  ENDDO

  NNB=IBL(NAT)

  RETURN
END SUBROUTINE SASSRT

SUBROUTINE SASPRT(NAT,NPR, &
     IBL,INB, &
     SASSIG,SASROV,SASPAT, &
     SASFCO,SASNCO, &
     QINFX, &
     QSRFC, &
     ATT)

  !     This routine prints some data.
  !
  !     Author: Urs Haberthuer.

  use chm_kinds
  use stream
  implicit none

  INTEGER     NAT,NPR
  INTEGER     IBL(*),INB(*)
  real(chm_real)      SASSIG(*),SASROV(*),SASPAT(*)
  real(chm_real)      SASFCO,SASNCO
  LOGICAL     QINFX
  LOGICAL     QSRFC
  CHARACTER(len=4) ATT(*)


  INTEGER     A,B
  INTEGER     I,J

  IF (PRNLEV .GE. 2) THEN
     WRITE(OUTU,'(A44)') &
          ' SASPRT> SASA 1.83 successfully initialized.'
     WRITE(OUTU,'(A34)') &
          ' SASPRT> SASA pair list generated.'
     WRITE(OUTU,'(A47,I8)') &
          ' SASPRT> Number of pairs in the SASA pair list:', &
          IBL(NAT)
     WRITE(OUTU,'(A54,A11)') &
          ' SASPRT> CHARMM atom types and the corresponding SASA ', &
          'parameters:'
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(3A26)') &
          '               Line number', &
          '          CHARMM atom type', &
          '       Solvation parameter'
     WRITE(OUTU,'(A1)') &
          ''
     DO I=1,NPR
        WRITE(OUTU,'(I26,A26,E26.6)') &
             I, &
             ATT(I), &
             SASSIG(I)
     ENDDO
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(3A26)') &
          '               Line number', &
          '          CHARMM atom type', &
          '                    Radius'
     WRITE(OUTU,'(A1)') &
          ''
     DO I=1,NPR
        WRITE(OUTU,'(I26,A26,E26.6)') &
             I, &
             ATT(I), &
             SASROV(I)
     ENDDO
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(3A26)') &
          '               Line number', &
          '          CHARMM atom type', &
          '   Probabilistic parameter'
     WRITE(OUTU,'(A1)') &
          ''
     DO I=1,NPR
        WRITE(OUTU,'(I26,A26,E26.6)') &
             I, &
             ATT(I), &
             SASPAT(I)
     ENDDO
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(A38)') &
          ' SASPRT> SASA connectivity parameters:'
     WRITE(OUTU,'(A28,F9.4)') &
          ' SASPRT> 1-2 pairs:         ',SASFCO
     WRITE(OUTU,'(A28,F9.4)') &
          ' SASPRT> More distant pairs:',SASNCO
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(A18,L1)') &
          ' SASPRT> Surface: ',QSRFC
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(A35,L1)') &
          ' SASPRT> Include fixed exclusions: ',QINFX
  ENDIF

  IF (PRNLEV .GE. 9) THEN
     WRITE(OUTU,'(A24)') &
          ' SASPRT> SASA pair list:'
     WRITE(OUTU,'(A1)') &
          ''
     WRITE(OUTU,'(3A20)') &
          '                Pair', &
          '          First Atom', &
          '         Second Atom'
     WRITE(OUTU,'(A1)') &
          ''
     DO I=1,NAT
        IF (I .EQ. 1) THEN
           A=1
        ELSE
           A=IBL(I-1)+1
        ENDIF
        B=IBL(I)
        DO J=A,B
           WRITE(OUTU,'(I20,I20,I20)') &
                J, &
                I, &
                INB(J)
        ENDDO
     ENDDO
  ENDIF
END SUBROUTINE SASPRT
#else /**/
SUBROUTINE NULL_SASINI
  RETURN
END SUBROUTINE NULL_SASINI
#endif 

