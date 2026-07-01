module cff_fcm
  use chm_kinds
  use dimens_fcm

  ! replaces old EQUIVALENCE
  use param, only: &
       CBOND1 => CBC, CBOND2 => CBB, &
       CTHET1 => CTC, CTHET2 => CTB, &
       CTHET3 => CTUC, CTHET4 => CTUB, &
       CPHI1 => CPC, CPHI2 => CPB, &
       CPHI3 => CPCOS, CPHI4 => CPSIN, &
       COPLN1 => CIC, &
       CNB1 => CNBA, CNB2 => CNBB

  implicit none
  character(len=*),private,parameter :: file_name   ="cff_ltm.src"
!  logical :: ucase

#if KEY_CFF==1 /*cff_main*/
  !     MXTHTH - maximum number of angle-angle's
  !     MXBB   - maximum number of bond-bond-1-3's
  !  INTEGER,parameter ::  MXBB=MAXA*0.31,MXTHTH=MAXA*3.5
  integer ::  MXBB,MXTHTH
  !
  !     MXTTP  - maximum number of angle-angle parameters
  !     MXNBDP - maximum number of non-bond parameters
  integer,parameter :: MXTTP=1500,MXNBDP=8000
  
  !     FORCE - forcefield type (cvff, amber, cff89)
  !     FRCVER - forcefield version to be used.  (If a parameter has a version
  !              number greater than FRCVER it will be ignored.  If the
  !              forcefield file has been maintained properly this provides
  !              the capability of using the forcefield as it existed at any
  !              time in the past.)
  character(len=40),save :: FORCE
  character(len=80),save :: FRCVER
  logical,save :: DECODE
  
  !     IBE  - equivalence atom types for bond parameters
  !     ITE  - equivalence atom types for angle parameters
  !     IPE  - equivalence atom types for torsion parameters
  !     IOE  - equivalence atom types for out-of-plane parameters
  !     INNB - equivalence atom types for non-bond parameters
  ! INTEGER,dimension(maxatc),save :: IBE,ITE,IPE,IOE,INNB
  integer,allocatable,dimension(:),save :: IBE,ITE,IPE,IOE,INNB

  !     IBAE   - automatic equiv. atom types for bond's
  !     INBAE  - automatic equiv. atom types for non-bond's
  !     ITAAE  - automatic equiv. atom types for apex-angle atoms
  !     ITEAE  - automatic equiv. atom types for end-angle atoms
  !     IPCAE  - automatic equiv. atom types for center-torsion atoms
  !     IPEAE  - automatic equiv. atom types for end-torsion atoms
  !     IOCAE  - automatic equiv. atom types for center-out-of-plane atoms
  !     IOEAE  - automatic equiv. atom types for end-out-of-plane atoms
  !INTEGER,dimension(maxatc),save :: &
  !     IBAE,INBAE,ITAAE,ITEAE,IPCAE,IPEAE,IOCAE,IOEAE
  integer,allocatable,dimension(:),save :: &
       IBAE,INBAE,ITAAE,ITEAE,IPCAE,IPEAE,IOCAE,IOEAE

  !     Hydrogen bond parameters - these are not actually used anywhere in
  !     CHARMM except to read the forcefield file
  integer,save :: NHTYP,NACTYP,HTYP(10),ATYP(10)
  real(chm_real),save :: HBDIST,HBANGL

  !     IBBW - list of torsions which have non-zero bond-bond-1-3 parameters
  !            (Most torsions have no bond-bond cross-term energy to calculate.
  !            This is a list of those that do need to be calculated.)
  !     ITTW - list of the 4 atoms in each angle-angle cross-term.
  !     ITFLG - derivative flags for bonds in angles (see below)
  !     IPHFLG - derivative flags for angles in torsions (see below)
  !     ITTFLG - derivative flags for angles in angle-angles (see below)
  !     The derivatives of bonds and angles are stored for reuse in cross-terms.
  !     However, the derivative for a bond in a bond-angle cross term may be
  !     the -ve of that for the stand-alone bond.  Likewise the derivatives
  !     for an angle in a cross term may be reversed from the derivatives of
  !     the angle itself.  The flag arrays indicate when to use the original
  !     derivatives and when to use the modified derivatives.
  integer,allocatable,dimension(:) :: IBBW
  integer,allocatable,dimension(:,:) :: ITTW
  integer(int_byte),allocatable,dimension(:,:) :: ITFLG, IPHFLG, ITTFLG

  !     BL   - list of bond lengths
  !     TH   - list of angle values
  !     PH   - list of torsion angle values
  !     OPLN - list of out-of-plane angle values
  !     DBDX,DBDY,DBDZ - derivatives of bonds wrt coordinates
  !     DTHDX,DTHDY,DTHDZ - derivatives of angles wrt coords.
  real(chm_real),allocatable,dimension(:),save :: BL,TH,PH,OPLN,DBDX,DBDY,DBDZ,COSTH
  real(chm_real),allocatable,dimension(:,:) ::DTHDX,DTHDY,DTHDZ
  
  !     BondType_cff - order of each bond
  integer,save, allocatable, dimension(:) :: BondType_cff
  !     ITBW  - pointer to the 2 bonds in aech angle
  !     IPBW  - pointer to the 3 bonds in each torsion
  !     IPTW  - pointer to the 2 angles in each torsion
  !     ITTTW - pointer to the 2 angles in each angle-angle
  integer,save,allocatable,dimension(:,:) :: ITBW,IPBW,IPTW,ITTTW
  
  !     CNBD1  - the A parameter for vdw
  !     CNBD2  - the B parameter for vdw
  !     MNO    - index array for nonbonds (for atom type i interacting with
  !              atom type j MNO(i,j) gives the index into CNBD1/CNBD2)
  !     NNBTYP - number of parameters in CNBD1/CNBD2
  !real(chm_real),dimension(mxnbdp),save :: CNBD1,CNBD2
  !integer,save ::MNO(MAXATC,MAXATC), NNBTYP
  integer, save ::NNBTYP
  real(chm_real),allocatable,dimension(:),save :: CNBD1,CNBD2
  integer,allocatable,dimension(:,:),save ::MNO
 
  !     NCBO  - number of bond parameters with and w/o bond order
  !     NCTO  - number of angle parameters with and w/o bond order
  !     NCPO  - number of torsion parameters with and w/o bond order
  !     NCAA  - number of angle-angle parameters with and w/o bond order
  !     NTHTH - number of angle-angle interactions
  !     NBB   - number of bond-bond-1-3 interactions
  !     NUMAT3- number of atoms times 3
  integer,save :: NCBO,NCTO,NCPO,NTHTH,NBB,IHYDNB,NCAA,NUMAT3
  
  !     BID1,BID2 - the 2 atom types in each bond parameter
  !     TID1,TID2,TID3 - the 3 atom types in each angle parameter
  !     PID1,PID2,PID3,PID4 -the 4 atom types in each torsion parameter
  !     OPID1,OPID2,OPID3,OPID4 -the 4 atom types in each out-of-plane parameter
  !     TTID1,TTID2,TTID3,TTID4 -the 4 atom types in each angle-angle parameter
  !     PNBID1,PNBID2 - the 2 atom types in each non-bond parameter
  !     ATI - list of atom types
  !     ICTT - pointer to parameter for each angle-angle
  !     BORD - order of each bond
  !     TORD1,TORD2 - order of each bond in each angle
  !     PORD1,PORD2,PORD3 - order of each bond in each torsion
  !     TTORD1,TTORD2,TTORD3 - order of each bond in each angle-angle
  !integer,save :: BID1(MAXCB),BID2(MAXCB), &
  !     TID1(MAXCT),TID2(MAXCT),TID3(MAXCT), &
  !     PID1(MAXCP),PID2(MAXCP),PID3(MAXCP),PID4(MAXCP), &
  !     OPID1(MAXCI),OPID2(MAXCI),OPID3(MAXCI),OPID4(MAXCI), &
  !     TTID1(MXTTP),TTID2(MXTTP),TTID3(MXTTP),TTID4(MXTTP), &
  !     PNBID1(MXNBDP),PNBID2(MXNBDP),ATI(MAXATC)
  integer,allocatable, dimension(:),save :: BID1,BID2, &
       TID1,TID2,TID3,PID1,PID2,PID3,PID4, &
       OPID1,OPID2,OPID3,OPID4,TTID1,TTID2,TTID3,TTID4, &
       PNBID1,PNBID2,ATI
  integer,allocatable,dimension(:) :: ICTT
  
  !real(chm_real),save :: BORD(MAXCB),TORD1(MAXCT),TORD2(MAXCT), &
  !     PORD1(MAXCP),PORD2(MAXCP),PORD3(MAXCP), &
  !     TTORD1(MXTTP),TTORD2(MXTTP),TTORD3(MXTTP)
  real(chm_real),allocatable,dimension(:),save :: BORD,TORD1,TORD2, &
       PORD1,PORD2,PORD3,TTORD1,TTORD2,TTORD3


  !     The following are forcefield parameters which couldn't be mapped
  !     into arrays already existing in CHARMM:
  !     CBOND3,CBOND4 - coefficients for cubic and quartic terms in bond energy
  !     CTHET5 - coefficients for bond-angle energy
  !     CTHET6,CTHET7 - coefficients for cubic and quartic terms in angle energy
  !     CBP11,CBP12,CBP13 - coeff's for bond_1-torsion
  !     CBP21,CBP22,CBP23 - coeff's for bond_2-torsion
  !     CBP31,CBP32,CBP33 - coeff's for bond_3-torsion
  !     CTP11,CTP12,CTP13 - coeff's for angle_1-torsion
  !     CTP21,CTP22,CTP23 - coeff's for angle_2-torsion
  !     CBB2 - coeff's for bond-bond-1-3
  !     CSGN1,CSGN2,CSGN3 - Phi0 values for torsion energy terms
  !     CTT - coefficients for angle-angle energy
  !     CNB5 - used during conversion of r*-epsilon to A-B while loading
  !            forcefield. Looks like it could be a temporary array.
  !real(chm_real),save :: CBOND3(MAXCB), CBOND4(MAXCB), &
  !     CTHET5(MAXCT),CTHET6(MAXCT),CTHET7(MAXCT), &
  !     CBP11(MAXCP),CBP12(MAXCP),CBP13(MAXCP), &
  !     CBP21(MAXCP),CBP22(MAXCP),CBP23(MAXCP), &
  !     CBP31(MAXCP),CBP32(MAXCP),CBP33(MAXCP), &
  !     CTP11(MAXCP),CTP12(MAXCP),CTP13(MAXCP), &
  !     CTP21(MAXCP),CTP22(MAXCP),CTP23(MAXCP),CBB2(MAXCP), &
  !     CSGN1(MAXCP),CSGN2(MAXCP),CSGN3(MAXCP), &
  !     CTT(MXTTP),CNB5(MXNBDP)
  real(chm_real),allocatable,dimension(:),save :: CBOND3, CBOND4, &
       CTHET5,CTHET6,CTHET7,CBP11,CBP12,CBP13, &
       CBP21,CBP22,CBP23,CBP31,CBP32,CBP33, &
       CTP11,CTP12,CTP13,CTP21,CTP22,CTP23,CBB2, &
       CSGN1,CSGN2,CSGN3,CTT,CNB5
  
  integer NTTP,NNONBP,NNBDP
  
  !
contains
  subroutine allocate_cff()
    use memory
    character(len=*),parameter :: routine_name="allocate_cff"
    ! Allocate space for maxatc variables
    call chmalloc(file_name,routine_name,'ibe ',maxatc,intg=ibe)
    call chmalloc(file_name,routine_name,'ite ',maxatc,intg=ite)
    call chmalloc(file_name,routine_name,'ipe ',maxatc,intg=ipe)
    call chmalloc(file_name,routine_name,'ioe ',maxatc,intg=ioe)
    call chmalloc(file_name,routine_name,'innb ',maxatc,intg=innb)

    call chmalloc(file_name,routine_name,'ibae ',maxatc,intg=ibae)
    call chmalloc(file_name,routine_name,'inbae ',maxatc,intg=inbae)
    call chmalloc(file_name,routine_name,'itaae ',maxatc,intg=itaae)
    call chmalloc(file_name,routine_name,'iteae ',maxatc,intg=iteae)
    call chmalloc(file_name,routine_name,'ipcae ',maxatc,intg=ipcae)
    call chmalloc(file_name,routine_name,'ipeae ',maxatc,intg=ipeae)
    call chmalloc(file_name,routine_name,'iocae ',maxatc,intg=iocae)
    call chmalloc(file_name,routine_name,'ioeae ',maxatc,intg=ioeae)

    call chmalloc(file_name,routine_name,'cnbd1 ',mxnbdp,crl=cnbd1)
    call chmalloc(file_name,routine_name,'cnbd2 ',mxnbdp,crl=cnbd2)
    call chmalloc(file_name,routine_name,'mno ',maxatc,maxatc,intg=mno)

    call chmalloc(file_name,routine_name,'bid1',maxcb,intg=bid1)
    call chmalloc(file_name,routine_name,'bid2',maxcb,intg=bid2)
    call chmalloc(file_name,routine_name,'tid1',maxct,intg=tid1)
    call chmalloc(file_name,routine_name,'tid2',maxct,intg=tid2)
    call chmalloc(file_name,routine_name,'tid3',maxct,intg=tid3)
    call chmalloc(file_name,routine_name,'pid1',maxcp,intg=pid1)
    call chmalloc(file_name,routine_name,'pid2',maxcp,intg=pid2)
    call chmalloc(file_name,routine_name,'pid3',maxcp,intg=pid3)
    call chmalloc(file_name,routine_name,'pid4',maxcp,intg=pid4)
    call chmalloc(file_name,routine_name,'opid1',maxci,intg=opid1)
    call chmalloc(file_name,routine_name,'opid2',maxci,intg=opid2)
    call chmalloc(file_name,routine_name,'opid3',maxci,intg=opid3)
    call chmalloc(file_name,routine_name,'opid4',maxci,intg=opid4)
    call chmalloc(file_name,routine_name,'ttid1',mxttp,intg=ttid1)
    call chmalloc(file_name,routine_name,'ttid2',mxttp,intg=ttid2)
    call chmalloc(file_name,routine_name,'ttid3',mxttp,intg=ttid3)
    call chmalloc(file_name,routine_name,'ttid4',mxttp,intg=ttid4)
    call chmalloc(file_name,routine_name,'pnbid1',mxnbdp,intg=pnbid1)
    call chmalloc(file_name,routine_name,'pnbid2',mxnbdp,intg=pnbid2)
    call chmalloc(file_name,routine_name,'ati',maxatc,intg=ati)

    call chmalloc(file_name,routine_name,'bord',maxcb,crl=bord)
    call chmalloc(file_name,routine_name,'tord1',maxct,crl=tord1)
    call chmalloc(file_name,routine_name,'tord2',maxct,crl=tord2)
    call chmalloc(file_name,routine_name,'pord1',maxcp,crl=pord1)
    call chmalloc(file_name,routine_name,'pord2',maxcp,crl=pord2)
    call chmalloc(file_name,routine_name,'pord3',maxcp,crl=pord3)
    call chmalloc(file_name,routine_name,'ttord1',mxttp,crl=ttord1)
    call chmalloc(file_name,routine_name,'ttord2',mxttp,crl=ttord2)
    call chmalloc(file_name,routine_name,'ttord3',mxttp,crl=ttord3)

    call chmalloc(file_name,routine_name,'cbond3',maxcb,crl=cbond3)
    call chmalloc(file_name,routine_name,'cbond4',maxcb,crl=cbond4)
    call chmalloc(file_name,routine_name,'cthet5',maxct,crl=cthet5)
    call chmalloc(file_name,routine_name,'cthet6',maxct,crl=cthet6)
    call chmalloc(file_name,routine_name,'cthet7',maxct,crl=cthet7)
    call chmalloc(file_name,routine_name,'cbp11',maxcp,crl=cbp11)
    call chmalloc(file_name,routine_name,'cbp12',maxcp,crl=cbp12)
    call chmalloc(file_name,routine_name,'cbp13',maxcp,crl=cbp13)
    call chmalloc(file_name,routine_name,'cbp21',maxcp,crl=cbp21)
    call chmalloc(file_name,routine_name,'cbp22',maxcp,crl=cbp22)
    call chmalloc(file_name,routine_name,'cbp23',maxcp,crl=cbp23)
    call chmalloc(file_name,routine_name,'cbp31',maxcp,crl=cbp31)
    call chmalloc(file_name,routine_name,'cbp32',maxcp,crl=cbp32)
    call chmalloc(file_name,routine_name,'cbp33',maxcp,crl=cbp33)
    call chmalloc(file_name,routine_name,'ctp11',maxcp,crl=ctp11)
    call chmalloc(file_name,routine_name,'ctp12',maxcp,crl=ctp12)
    call chmalloc(file_name,routine_name,'ctp13',maxcp,crl=ctp13)
    call chmalloc(file_name,routine_name,'ctp21',maxcp,crl=ctp21)
    call chmalloc(file_name,routine_name,'ctp22',maxcp,crl=ctp22)
    call chmalloc(file_name,routine_name,'ctp23',maxcp,crl=ctp23)
    call chmalloc(file_name,routine_name,'cbb2',maxcp,crl=cbb2)
    call chmalloc(file_name,routine_name,'csgn1',maxcp,crl=csgn1)
    call chmalloc(file_name,routine_name,'csgn2',maxcp,crl=csgn2)
    call chmalloc(file_name,routine_name,'csgn3',maxcp,crl=csgn3)
    call chmalloc(file_name,routine_name,'ctt',mxttp,crl=ctt)
    call chmalloc(file_name,routine_name,'cnb5',mxnbdp,crl=cnb5)

    MXBB=MAXA*0.31
    MXTHTH=MAXA*3.5
    call chmalloc(file_name,routine_name,'itflg ',2,maxt,    iby=itflg)
    call chmalloc(file_name,routine_name,'iphflg',2,maxp,    iby=iphflg)
    call chmalloc(file_name,routine_name,'ittflg',2,mxthth,  iby=ittflg)
    call chmalloc(file_name,routine_name,'bl    ',maxb,      crl=bl)
    call chmalloc(file_name,routine_name,'th    ',maxt,      crl=th)
    call chmalloc(file_name,routine_name,'ph    ',maxp,      crl=ph)
    call chmalloc(file_name,routine_name,'opln  ',maximp,    crl=opln)
    call chmalloc(file_name,routine_name,'dbdx  ',maxb,      crl=dbdx)
    call chmalloc(file_name,routine_name,'dbdy  ',maxb,      crl=dbdy)
    call chmalloc(file_name,routine_name,'dbdz  ',maxb,      crl=dbdz)
    call chmalloc(file_name,routine_name,'dthdx ',3,maxt,    crl=dthdx)
    call chmalloc(file_name,routine_name,'dthdy ',3,maxt,    crl=dthdy)
    call chmalloc(file_name,routine_name,'dthdz ',3,maxt,    crl=dthdz)
    call chmalloc(file_name,routine_name,'bondtype_cff',maxb, &
         intg=bondtype_cff)
    call chmalloc(file_name,routine_name,'costh ',maxt,      crl=costh)
    call chmalloc(file_name,routine_name,'ibbw  ',mxbb,    intg=ibbw)
    call chmalloc(file_name,routine_name,'ittw  ',4,mxthth,intg=ittw)
    call chmalloc(file_name,routine_name,'ictt  ',mxthth,intg=ictt)
    call chmalloc(file_name,routine_name,'itbw  ',2,maxt   ,intg=itbw)
    call chmalloc(file_name,routine_name,'ipbw  ',3,maxp   ,intg=ipbw)
    call chmalloc(file_name,routine_name,'iptw  ',2,maxp   ,intg=iptw)
    call chmalloc(file_name,routine_name,'itttw ',2,mxthth,intg=itttw)

    return
  end subroutine allocate_cff
#endif /* (cff_main)*/


end module cff_fcm

