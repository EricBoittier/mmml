!
! NOTE: this routine is under development.... for full
!       compatibility with many charmm functionalities
!       (i.e. PERT, TSM, IMAGES etc...) arrays now passed
!       via psf.f90 etc... have to be passed as arguments.
!       This will mean change to CODES argument list as well.
!
#if KEY_MMFF==1
! =========================================================
! SUBROUTINE CODES_MM : parameter assignment for MMFF
! =========================================================
SUBROUTINE CODES_MM
  !
  !  01 Dec 95 Jay Banks: added MDSTBN to argument list for CALL KTHETAM.
  !
  use chm_kinds
  use dimens_fcm
  use code
  !#INCLUDE '~/charmm_fcm/consta.f90'
  !#INCLUDE '~/charmm_fcm/coord.f90'
  use exfunc
  use mmffm
  use param
  use psf
  use inbnd
  implicit none
  !#INCLUDE '~/charmm_fcm/stack.f90'
  !#INCLUDE '~/charmm_fcm/stream.f90'
  !#INCLUDE '~/charmm_fcm/ring.f90'
  !
  integer ERROR
  integer, parameter :: Fatal=2, NonFatal=1
  !
  !  CONSTANTS ARE NOW ASSIGNED IN THE FOLLOWING ORDER: TORSION,
  !  STRETCHING, VDW AND BENDING.
  !  MAKE SURE ITAB REFLECTS MULTIPLE BONDS
  CALL TABINT(2,IB,JB,BondType,NATOM,NBOND,ITAB,LOCAT)
  ERROR=0
  CALL KBONDM(ERROR)
  CALL KVDW(ERROR)
  !         CALL KTHETAM(ERROR)
  CALL KTHETAM(AtNum,MTYPE, &
       NBOND,IB,JB,MDBOND,ITAB, &
       NCT,NCTT,NTHETA,IT,JT,KT,LTHETA, &
       MDTHET,ICT,KCT,AnglEq,AnglFC, &
       NCOOP,NCOOPT,ICOOP,KCOOP,OoplFC, &
       NCSB,NCSBT,ICSTBN,KCSTBN,StrbList,STBNP, &
       ERROR,MDSTBN)
  CALL KOMEGAM(ERROR)
  IF(ERROR.EQ.Fatal) THEN
     CALL WRNDIE(-5,'<codes_mm>', &
          ' FATAL ERROR(S) IN ASSIGNMENT OF PARAMETERS')
  ELSE IF(ERROR.EQ.NonFatal) THEN
     CALL WRNDIE(0,'<codes_mm>', &
          ' WARNING: MISSING PARAMETERS DETECTED')
  ENDIF
  !   NOW REGENERATE IN SINGLE-LISTING FORM (AS THOUGH ALL BONDS
  !   WERE SINGLE BONDS), BUT LEAVE LOCAT() SET AT THE NUMBER OF
  !   ATOMS ATTACHED TO EACH GIVEN ATOM (THE FIRST ARGUMENT OF -1
  !   HAS THIS EFFECT
  CALL TABINT(-1,IB,JB,BondType,NATOM,NBOND,ITAB,LOCAT)
  !
  RETURN
END SUBROUTINE CODES_MM

SUBROUTINE MMFF_INIT
  !
  use chm_kinds
  use dimens_fcm
  !
  use ffieldm
  use inbnd
  use param
  use mmffm
  implicit none
  integer i
  do i=1,NPFILES
     not_read(i)=.TRUE.
  enddo
  PARNAME(AnglFile)='AnglFile'
  PARNAME(AromFile)='AromFile'
  PARNAME(BndkFile)='BndkFile'
  PARNAME(BondFile)='BondFile'
  PARNAME(ChrgFile)='ChrgFile'
  PARNAME(DefiFile)='DefiFile'
  PARNAME(DfsbFile)='DfsbFile'
  PARNAME(HdefFile)='HdefFile'
  PARNAME(OoplFile)='OoplFile'
  PARNAME(PbciFile)='PbciFile'
  PARNAME(PropFile)='PropFile'
  PARNAME(StbnFile)='StbnFile'
  PARNAME(SuppFile)='SuppFile'
  PARNAME(SymbFile)='SymbFile'
  PARNAME(TorsFile)='TorsFile'
  PARNAME(VdwFile )='VdwFile '
  !     SALONE=.TRUE.
  AuxPar(cstr)=-2.  ! coefficient of MMFF cubic stretch terms
  AuxPar(cbnd)=-0.4 ! COEFFICIENT OF mmff cubic BENDING TERMS
  AuxPar(THRESH)=0.00001
  DELQ=0.05   ! parameters for buffered coulombic potential
  DELBUF=0.07 ! DEL parameter for buf-14-7 potential
  GAMBUF=0.12 ! GAM parameter for buf-14-7 potential
  !
  RETURN
END SUBROUTINE MMFF_INIT

SUBROUTINE MMFF_IO(MODE,KEY,UNIT,CMPD)
  !
  ! Jay Banks 25 Oct 95: add ##INCLUDE mmff_arrays.f90, change array name
  ! XYZM to XYZ.
  !
  ! Jay Banks 08 Nov 95: remove mmff_arrays.f90, pass X, Y, Z to molin and
  ! molout instead of XYZ.
  use chm_kinds
  use dimens_fcm
  use comand
  use coord
  use ctitla
  use exfunc
  use merck_io
  use mmffm
  use io
  use psf
  use modpsf
  use stream
  use string
  !
  implicit none
  !
  character(len=4) MODE  ! READ | WRITe
  character(len=4) KEY   ! MERCk
  integer     UNIT  ! i/o unit
  !
  character(len=4) sname
  character(len=24) COMLY
  character(len=20) CMPD
  integer COMLE
  logical error
  !
  if(MODE.eq.'READ') then

     !
     ! I-Jen Chen was here.
     !         if(KEY.eq.'MOL ' .or. KEY.eq.'MERC') then ! 'MOL' to be discontinued
     ! and replaced by MERCk
     if(KEY.eq.'MOL '.or.KEY.eq.'MERC'.or.KEY.eq.'MOL2'.or.KEY.eq. &
          'DB  ') then
        ! I-Jen left.
        sname='SNAM'
        DATA_READ=MERCK
        if(INDXA(COMLYN,COMLEN,'APPE').eq.0) then
           COMLY='ATOMS SELE ALL END'
           COMLE=18                   ! delete all atoms
           call deltic(COMLY,COMLE)   ! if not read append
        endif
#if KEY_DEBUG==1
        if(PRNLEV.GT.5) write(outu,'(a)') ' mmff_io> call molin...'
#endif 

        !
        ! I-Jen was here.
        !            call MOLIN(UNIT,SNAME,ERROR,
#ifdef KEY_RESIZE
        call MOLIN(UNIT,SNAME,ERROR,KEY,CMPD)
#else
        call MOLIN(UNIT,SNAME,ERROR,KEY,CMPD, &
             ! I-Jen left.
             NATOM,X,Y,Z,ATNUM,AtNames,NICHG,  & ! ATOMS
             NBOND,IB,JB,BondType,            & ! BONDS
             IGPBS,NGRP,                            & ! GROUPS
             RESID,RES,IBASE,NRES,RESNAME,KRES,     & ! RESIDUES
             SEGID,NICTOT,NSEG)                    ! SEGMENTS
#endif
        if(error) call wrndie(-3,'<mmff_io>','error in molin')
        NATOMT=NATOM
        NREST=NRES
        NSEGT=NSEG
        NGRPT=NGRP
        NBONDT=NBOND
        NBONDT2=NBOND ! hwm: per BRB's suggestion on 220722
        NICTOT(NSEG+1) = NRES
        !
        !           call mmff_setup
        !
     else
        call wrndie(-5,'<mmff_io>','illegal KEY for READ')
     endif
  elseif(MODE.eq.'WRIT') then
     !
     ! I-Jen was here.
     !         if(KEY.eq.'MOL ' .or. KEY.eq.'MERC') then ! 'MOL' to be discontinued
     ! and replaced by MERCk

     if(KEY.eq.'MOL '.or.KEY.eq.'MERC'.or.KEY.eq.'MOL2') then
        ! I-Jen left.
        DATA_WRITTEN=MERCK
        call MOLOUT(TITLEA,UNIT, &
             NATOM,X,Y,Z,ATNUM,AtNames,NICHG,PARTLQ,MTYPE, &
             NBOND,IB,JB,BondType, &
             RESID,RES,IBASE,NRES, &
             SEGID,NICTOT,NSEG)
     else
        call wrndie(-5,'<mmff_io>','illegal KEY for WRITE')
     endif
  else
     call wrndie(-5,'<mmff_io>','illegal MODE')
  endif
  return
END SUBROUTINE MMFF_IO

SUBROUTINE PARRDR_MMFF(UNIT,COMLYN,COMLEN)
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use ffieldm
  use stream
  use string
  implicit none
  INTEGER UNIT,COMLEN
  CHARACTER(len=*) COMLYN
  !
  CHARACTER(len=4) WRD
  integer ISUPP ! supplementary parameter unit
  integer mua   ! ???
  integer nb    ! supplementary ANGL bending parameters
  integer nv    ! supplementary VDW parameters
  integer ns    ! supplementary BOND strech parameters
  integer nq    ! supplementary CHARge parameters
  integer no    ! supplementary OOPL parameters
  integer nsb   ! ???
  integer nt    ! supplementary TORSional parameters
  save mua, nb, no, nq, ns, nsb, nt, nv, ISUPP
  !
  !     if(INDXA(COMLYN,COMLEN,'MMFF').eq.0) then
  !        CALL WRNDIE(-3,'<PARRDR_MMFF>',
  !    $        'using mmff but read parameter missing mmff keyword')
  !     endif
  !
  WRD=NEXTA4(COMLYN,COMLEN)
  if    (WRD.eq.'ANGL') then                        ! MMFFANG.PAR
     call rdthetm(NB,UNIT,ISUPP,Not_Read(AnglFile))
  elseif(WRD.eq.'AROM') then                        ! MMFFAROM.PAR
     call rdarom(UNIT,Not_Read(AromFile))
  elseif(WRD.eq.'BNDK') then                        ! MMFFBNDK.PAR
     call rdbondk(UNIT,Not_Read(BndkFile))
  elseif(WRD.eq.'BOND') then                        ! MMFFBOND.PAR
     call rdbondm(NS,UNIT,ISUPP,Not_Read(BondFile))
  elseif(WRD.eq.'CHRG') then                        ! MMFFCHG.PAR
     call rdchgm(NQ,UNIT,ISUPP,Not_Read(ChrgFile))
  elseif(WRD.eq.'DEFI') then                        ! MMFFDEF.PAR
     call rddefi(UNIT,Not_Read(DefiFile))
  elseif(WRD.eq.'DFSB') then                        ! MMFFDFSB.PAR
     call rddfsb(UNIT,Not_Read(DfsbFile))
  elseif(WRD.eq.'HDEF') then                        ! MMFFHDEF.PAR
     call rdhdef(UNIT,Not_Read(HdefFile))
  elseif(WRD.eq.'OOPL') then                        ! MMFFOOP.PAR
     call rdoopm(NO,UNIT,ISUPP,Not_Read(OoplFile))
  elseif(WRD.eq.'PBCI') then                        ! MMFFPBCI.PAR
     call rdpbci(UNIT,Not_Read(PbciFile))
  elseif(WRD.eq.'PROP') then                        ! MMFFPROP.PAR
     call rdpropm(UNIT,Not_Read(PropFile))
  elseif(WRD.eq.'STBN') then                        ! MMFFSTBN.PAR
     call rdstbnm(UNIT,Not_Read(StbnFile))
  elseif(WRD.eq.'SUPP') then                        ! MMFFSUP.PAR
     READ(UNIT,'(8I5)') NV,NS,MUA,NQ,NB,NO,NSB,NT
     ISUPP=UNIT
     Not_Read(SuppFile)=.FALSE.
  elseif(WRD.eq.'SYMB') then                        ! MMFFSYMB.PAR
     call rdsymb(UNIT,Not_Read(SymbFile))
  elseif(WRD.eq.'TORS') then                        ! MMFFTOR.PAR
     call rdtors(NT,UNIT,ISUPP,Not_Read(TorsFile))
  elseif(WRD.eq.'VDW ') then                        ! MMFFVDW.PAR
     call rdvdwm(NV,UNIT,ISUPP,Not_Read(VdwFile))
  else
     SCRTCH='Unknown parameter file '//WRD
     call wrndie(-3,'<PARRDR_MMFF>',SCRTCH(:27))
  endif
  !
  RETURN
END SUBROUTINE PARRDR_MMFF

! =====================================================================
! mmff_setup : performs mmff psf generation
! =====================================================================
SUBROUTINE mmff_setup
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  !...##INCLUDE '~/charmm_fcm/debug.f90'
  use memory
  use dimens_fcm
  use code
  use exfunc
  use ffieldm
  use mmffm
  use number
  use psf
  use stream
  use string
  implicit none
  !
  integer,allocatable,dimension(:) :: nimph
  integer naddh, NATOM0
  integer contrl(11)
  logical ok
  !
  integer i,warn
  !
  if(NATOM.le.0) return
  !
  ! test if all MMFF parameter files have been read
  !
  ok=.true.
  do i=1,NPFILES
     if(not_read(i)) then
        if(wrnlev.gt.2) write(outu,'(a,a,a)') &
             'mmff_setup> ERROR: ',PARNAME(i),' parameter file not read'
        ok=.false.
     endif
  enddo
  if(.not.ok) call wrndie(-1,'<mmff_setup>', &
       ' some MMFF parameter files not read')
  !
  mtype(1:natom) = 0 ! reset atom types
  !
  !   CARRY OUT THE SETUP ON THE SUBJECT MOLECULE
  !   .................................................................
  !   THIS CODE USES SUBROUTINE SETUP TO DIRECT THE SETUP PROCEDURE
  !   UNDER THE CONTROL OF INTEGER VARIALBES CONTRL(1..11)
  !   WHOSE SIGNIFICANCE IS DESCRIBED IN SUBROUTINE SETUP
  !  ....................................................................
  !   CARRY SETUP THROUGH THE POINT OF DETECTING ANY MISSING HYDROGENS ON
  !   HETEROATOMS (WHICH MUST BE PRESENT, OR MUST BE ADDED, FOR MMFF
  !   TO PROCEED)
  NATOM0=NATOM
  naddh=0
  call chmalloc('mmff.src','mmff_setup','nimph',2*NATOM,intg=nimph)
  contrl(1:3) = 1
  contrl(4:11) = 0
  call SETUPMF(CONTRL,OK,naddh,nimph)
  IF(.NOT.OK) CALL WRNDIE(-5,'<mmff_setup>','ERROR 1')
  contrl(1:11) = 1
  contrl(9)=2
  IF(NATOM.EQ.NATOM0+naddh) THEN
     contrl(1:3)=0
     call SETUPMF(CONTRL,OK,naddh,nimph)
     IF(.NOT.OK) CALL WRNDIE(-5,'<mmff_setup>','ERROR 2')
  ELSE
     call SETUPMF(CONTRL,OK,naddh,nimph)
     IF(.NOT.OK) CALL WRNDIE(-5,'<mmff_setup>','ERROR 3')
  ENDIF
  call chmdealloc('mmff.src','mmff_setup','nimph',2*NATOM,intg=nimph)
  !
  ! RCZ930513 - group/residue/segment treatment should
  !             be later made more
  !             sofisticated and/or merged with charmm
  !
  if(nseg.eq.0) then
     nseg=1
     nictot(1)=0
     nictot(2)=NRES
     segid(1)='MMFF'
  endif
  !
  warn=0
  do i=1,natom
     !
     ! AMASS is not correctly initialized in genpsf...???
     !
     !       if(AMASS(I).eq.ZERO) then ! RCZ931027???
     AMASS(I)=ElementMass(AtNum(I))
     if(AMASS(I).eq.ZERO) warn=warn+1
     !...##IF DEBUG
     !          if(prnlev.gt.5 .or. (prnlev.ge.2 .and. warn.ne.0))
     !     &    write(outu,'(a,2i5,f10.5)')
     !     &   ' mmff_setup> i,AtNum(I),AMASS(I)=',
     !     &                 i,AtNum(I),AMASS(I)
     !...##ENDIF
     !       endif
  enddo
  if(warn.ne.0) call wrndie(-1,'<mmff_setup>','AMASS.eq.ZERO')
  !
  !     CALL PSFSUM(OUTU)
  !...##IF DEBUG
  !C     mustup=.false.    ! removing this causes differences
  !C                       ! in STRBN term ?!?!
  !      call getenv('MUSTUP',SCRTCH)
  !      if(SCRTCH.eq.'TRUE')   mustup=.TRUE.
  !      if(SCRTCH.eq.'FALSE')  mustup=.FALSE.
  !...##ENDIF
  !
  RETURN
END SUBROUTINE mmff_setup

! =========================================================
! SUBROUTINE SETUP : structure perception and atom-type assignment
! =========================================================
SUBROUTINE SETUPMF(CONTRL,OK,naddh,nimph)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: add ##INCLUDE mmff_arrays.f90, change array name
  ! XYZM to XYZ.
  !
  ! Jay Banks 09 Nov 95: delete mmff_arrays.f90, pass X,Y,Z (from
  ! coord.f90) rather than XYZ to HADD.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use memory
  use dimens_fcm
  use code
  use consta
  use coord
  use exfunc
  use inbnd
  use mmffm
  use param
  use psf
  use stream
  use string
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: FORMLQ
  integer CONTRL(*)
  integer NIMPH(*)
  !
  integer i
  integer j
  integer jmax
  integer nadd
  integer ndef
  integer naddh, nmiss
  integer n, npdp
  !
  !RCZ      CHARACTER*1 IFADD
  LOGICAL OK
  !....
  !   THIS ROUTINE DIRECTS VARIOUS ASPECTS OF THE STRUCTURE PERCEPTION AND
  !   PARAMETERIZATION PROCESS
  !   ITS OPERATION IS CONTROLED BY THE VALUES OF THE INTEGER VARIABLES
  !   CONTRL(1..11), WHOSE SIGNIFICANCE IS AS FOLLOWS:
  !
  !   CONTRL(I) =
  !
  !    (1) = 1:  DETECT MISSING (IMPLICIT) HYDROGENS ON CARBON ATOMS
  !              ASSIGN MMFF SYMBOLIC AND NUMERIC ATOM TYPES TO ALL
  !              NON-HYDROGEN ATOMS
  !    (2) = 1:  NOT USED
  !    (3) = 1:  ADD ANY MISSING HYDROGENS ON HETEROATOMS
  !    (4) = 1:  ASSIGN MMFF ATOM TYPES FOR ALL HYDROGEN ATOMS
  !    (5) = 1:  FIND RINGS (NOTE: DOMAIN FINDING IS NO LONGER
  !              NECESSARY FOR RING FINDING)
  !    (6) = 1:  CARRY OUT RING-TYPE PERCEPTION AND PERCEIVE ANY
  !              SINGLE BONDS BETWEEN SP2-HYBRIDIZED CARBON ATOMS
  !    (7) = 1:  PRINTOUT THE CHOICE IN EFFECT FOR THE INTER- AND
  !              INTRAMOLECULAR NONBONDED POTENTIAL
  !    (8) = 1:  ASSIGN THE NON-BONDED POTENTIAL PARAMETERS (VDW AND
  !              ELECTROSTATIC); RECORD POTENTIAL HYDROGEN BONDING
  !              ACCEPTOR AND DONOR ATOMS
  !    (9) = 1:  DETERMINE THE CHAIN STRUCTURE FROM THE SPANNING TREE;
  !              FIND AND ASSIGN THE TORSION POTENTIAL FOR ALL ROTATABLE
  !              (NON-RING) BONDS
  !        = 2:  DETERMINE CHAIN STRUCTURE; REASSIGN TORSION POTENTIAL TO
  !              REFLECT CHANGES MADE IN BOND LENGTHS AND BOND ANGLES AS
  !              A RESULT OF XYZ OPTIMIZATION
  !   (10) = 1:  DETERMINE INTERNAL-VARIABLE SET OF BOND LENGTHS, BOND
  !              ANGLES, OUT-OF-PLANE ANGLES, AND DIHEDRAL ANGLES
  !   (11) = 1:  ASSIGN MMFF PARAMETERS TO THE I10 INTERNAL VARIABLE SET
  !....
  !
  OK=.TRUE.
  !
  !   SET OR RESET RESIDUE NUMBER COUNT TO ZERO SO FUNCTION NUM WILL
  !   DETERMINE OR REDETERMINE THE SLOT NUMBERS OF THE FIRST (IR1) AND
  !   LAST (IR2) ATOMS FOR EACH RESIDUE ACTUALLY PRESENT, AND WILL
  !   ASSIGN IRES APPROPRIATELY
  !
  !     IRES=0
  IF(CONTRL(1).EQ.1) THEN
     !   CHECK FORMAL CHARGES
     CALL TABINT(2,IB,JB,BondType,NATOM,NBOND,ITAB,LOCAT)

     !   DETERMINE NUMBERS OF IMPLICIT H'S ON C'S
     CALL IMPLCT(NIMPH,NATOM,ATNUM,MTYPE,IB,JB,BondType, &
          NBOND,NICHG)
     NMISS=0
     DO I=1,NATOM
        JMAX=LOCAT(I)
        !         WRITE(6,'(7I5)') I,LOCAT(I),(ITAB(J,I),J=1,5)
        IF(NIMPH(I).NE.0) THEN
           if(wrnlev.ge.2) WRITE(OUTU,9900) QNAME(I),NIMPH(I), &
                (AtNames(ITAB(J,I)),J=1,JMAX)
           !    .      (CNAME(ITAB(J,I)),J=1,JMAX)
9900       FORMAT(' SETUPMF> CARBON ',A,' IS MISSING',I2,' HYDROGENS', &
                ' .. ATTACHED ATOMS: ',5A)
        ELSE
           !             WRITE(12,9910) QNAME(I),(CNAME(ITAB(J,I),J=1,JMAX)
9910       FORMAT('   ATOM ',A,23X,5A)
        ENDIF
        NMISS=NMISS+NIMPH(I)
     ENDDO
     IF(NMISS.NE.0) THEN
        write(SCRTCH,'(i5,a)') NMISS,' Missing hydrogens'
        call wrndie(-2,'<SETUPMF>',SCRTCH(:30))
        IF(wrnlev.gt.2) WRITE(outu,91) NMISS
91      FORMAT(/' *** MMFF believes that ',I3,' hydrogens need', &
             ' to be added to carbon atoms')
        !              add any missing h's on carbon atoms
#ifdef KEY_RESIZE
        call HADD(MTYPE,ATNUM,BondType, &
             NADDH,NIMPH,0,OK,ITAB,LOCAT,AtNames,RESNAME,KRES)
#else
        CALL HADD(NATOM,X,Y,Z,MTYPE,ATNUM,IB,JB,BondType, &
             NBOND,naddh,0,NIMPH,OK,ITAB,LOCAT,AtNames, &
             RESNAME,KRES)
#endif        
        IF(naddh.gt.0) then
           !             update psf group and residue boundary arrays for new atoms
           IBASE(NRES+1)=NATOM
           IGPBS(NGRP+1)=NATOM
           IGPTYP(NGRP)=2 ! assume new group has a net charge...
        ENDIF
     ENDIF
     !   ASSIGN MMFF TYPES - FIRST PREZERO IFNPDP ELEMENTS (THESE QUANTITIES
     !   WILL BE SET TO ZERO FOR NITROGENS FOUND TO BE QUATERNARY NITROGENS
     !   IN A PYRIDINE-LIKE RING; IN SOME CASES, THESE ATOM TYPES WILL NEED
     !   TO BE REVISED LATER
     !
     DO N=1,NATOM
        IFNPDP(N)=.FALSE.
     ENDDO
     CALL XTYPE
     !      write(6,*) ' after XTYPE'
     !        DO N=1,NATOM
     !           WRITE(OUTU,'(A,I4,3X,A,I4)')
     !     .     ' N, QNAME(N), MTYPE(N)', N, QNAME(N), MTYPE(N)
     !       ENDDO
  ENDIF
  !      IF(CONTRL(2).EQ.1) THEN
  !      ENDIF
  IF(CONTRL(3).EQ.1) THEN
     !   ADD ANY MISSING H'S ON HETEROATOMS
#ifdef KEY_RESIZE
     call HADD(MTYPE,ATNUM,BondType, &
          NADDH,2,NIMPH,OK,ITAB,LOCAT,AtNames,RESNAME,KRES)
#else
     CALL HADD(NATOM,X,Y,Z,MTYPE,ATNUM,IB,JB,BondType,NBOND, &
          NADD,2,NIMPH,OK,ITAB,LOCAT,AtNames,RESNAME,KRES)
#endif
     IF(nadd.gt.0) then
        !             update psf group and residue boundary arrays for new atoms
        IBASE(NRES+1)=NATOM
        IGPBS(NGRP+1)=NATOM
        IGPTYP(NGRP)=2 ! assume new group has a net charge...
     ENDIF
     !
     IF(.NOT.OK) THEN
        !   AN ILLEGAL ATTEMPT TO ADD HETERO-HYDROGENS IN A NON STAND-ALONE RUN
        !   ENCOUNTERED - RETURN TO HOST PROGRAM (FRODO OR MOGLI; MOLEDIT RUNS
        !   WILL HAVE USED "CALL WRNDIE(-5,'<setup>','ERROR MSG')" TO GET BACK T
        call wrndie(-5,'<SETUPMF>', &
             'ILLEGAL ATTEMPT TO ADD HETERO-HYDROGENS')
        RETURN
     ENDIF
  ENDIF
  IF(CONTRL(4).EQ.1) THEN
     !  REGENERATE BUT SUPPRESS MULTIPLE-BOND LISTING
     CALL TABINT(1,IB,JB,BondType,NATOM,NBOND, &
          ITAB,LOCAT)
  ENDIF
  IF(CONTRL(5).EQ.1) call GenerateRings
  IF(CONTRL(6).EQ.1) THEN
     !   ASSIGN AROMATIC ATOM TYPES TO ANY AROMATIC RINGS DETECTED
     IF(IRINGS.GT.0) CALL RGTYPE
     !   NOW RECOMPUTE TABLE TO INCLUDE HETEROATOM H'S
     CALL TABINT(2,IB,JB,BondType,NATOM,NBOND, &
          ITAB,LOCAT)
     NPDP=0
     !         DO N=1,NATOM
     !           WRITE(OUTU,'(A,I4,3X,A,I4)')
     !     .     ' N, QNAME(N), MTYPE(N)', N, QNAME(N), MTYPE(N)
     !         ENDDO
     IF(IRINGS.GT.0) CALL RETYPE(NPDP)
     IF(NPDP.GT.0) THEN
        !
        !  AT LEAST ONE PYRIDINE-TYPE QUATERNARY NITROGEN IS PRESENT --
        !  REDO THE ATOM TYPING, AS THE ORIGINAL RESULT WILL HAVE BEEN
        !  "WRONG" IF THE FORMAL CHARGE ON THAT NITROGEN HAS BEEN
        !  DELOCALIZED OVER A NEIGHBORING (E.G., ORTHO) TRIGONAL NITROGEN
        !
        DO N=1,NATOM
           PARTLQ(N)=0.
           MTYPE(N)=0
        ENDDO
        CALL XTYPE
        CALL RGTYPE
     ENDIF
     !
     !   ASSIGN MMFF ATOM TYPES FOR THE HYDROGENS
     !        CALL HTYPE(NATOM,ATNUM,ITAB)
     CALL HTYPE
     !  REGENERATE BUT SUPPRESS MULTIPLE-BOND LISTING
     CALL TABINT(1,IB,JB,BondType,NATOM,NBOND, &
          ITAB,LOCAT)
  ENDIF
  IF(CONTRL(7).EQ.1) THEN
     ! DEFINE THE NONBONDED PARAMETERS TO BE USED
     !
     !   FIRST TELL USER WHAT POTENTIALS ARE NOW BEING USED
     !
     NDEF=2
     if(prnlev.ge.2) WRITE(OUTU,9902)
9902 FORMAT(/' INTRAmolecular interactions use the ', &
          'BUFFERED 14-7 potential and parameters'/ &
          ' INTERmolecular interactions also use the BUFFERED ', &
          '14-7 potential and parameters')
     CALL SAYC(' ')
     IF(LCONS) THEN
        CALL SAYC(' ELECTROSTATIC interactions ' &
             //'currently use a'// &
             ' CONSTANT dielectric model')
        if(prnlev.ge.2) WRITE(OUTU,9250) eps
9250    FORMAT(' The current INTRA- and INTERmolecular ', &
             'dielectric constant is:',F7.3)
     else
        CALL SAYC(' ELECTROSTATIC interactions '// &
             'currently use a'// &
             ' DISTANCE-DEPENDENT dielectric model')
        if(prnlev.ge.2) WRITE(OUTU,7010) eps
7010    FORMAT(' The current value for the DISTANCE-DEPENDENT ', &
             'dielectric constant is:',F7.3/)
     ENDIF
  ENDIF
  IF(CONTRL(8).EQ.1) THEN
     !   ASSIGN PARTIAL CHARGES
     !   PARTLQ is initialized in xtype routine and is used in genchgm to
     !   load what mmff calls "formal atomic charges" (FORMLQ); the latter
     !   are used in genchgm to distribute certain negative formal atomic
     !   among the atoms connected to the center assigned that charge in
     !   xtype.
     call chmalloc('mmff.src','SETUPMF','FORMLQ',NATOM,crl=FORMLQ)
     call GENCHGM(IB,JB,FORMLQ,NBOND,NATOM,ATNUM)
     call chmdealloc('mmff.src','SETUPMF','FORMLQ',NATOM,crl=FORMLQ)
  ENDIF
  IF(CONTRL(9).EQ.1) THEN
     !   not used in opti_msi
  ENDIF
  IF(CONTRL(10).EQ.1) THEN
     !   COMPLETE THE MMFF SETUP ON THE SUBJECT MOLECULE
     !  PREPARE AND WRITE ATOMIC BOND TABLE
     CALL BONDTB
     !  SET UP BOND ANGLE LIST (IT,JT,KT,LTHETA)
     CALL THETA
     !  SET UP DIHEDRAL ANGLE LIST (IP,JP,KP,LP)
     CALL GOMEGA
  ENDIF
  IF(CONTRL(11).EQ.1) THEN
     call codes_mm
     !RCZC  CONSTANTS ARE NOW ASSIGNED IN THE FOLLOWING ORDER: TORSION,
     !RCZC  STRETCHING, VDW AND BENDING.
     !RCZC  MAKE SURE ITAB REFLECTS MULTIPLE BONDS
     !RCZ          CALL TABINT(2,IB,JB,BondType,NATOM,NBOND,ITAB,LOCAT)
     !RCZC   INITIALIZE INIT
     !RCZ          INIT=0
     !RCZ             CALL KBONDM(INIT)
     !RCZ             CALL KVDW(INIT)
     !RCZC            CALL KTHETAM(INIT)
     !RCZ             CALL KTHETAM(AtNum,MTYPE,
     !RCZ     &                    NBOND,IB,JB,MDBOND,ITAB,
     !RCZ     &                    NCT,NCTT,NTHETA,IT,JT,KT,LTHETA,
     !RCZ     &                    MDTHET,ICT,KCT,AnglEq,AnglFC,
     !RCZ     &                    NCOOP,NCOOPT,ICOOP,KCOOP,OoplFC,
     !RCZ     &                    NCSB,NCSBT,ICSTBN,KCSTBN,StrbList,STBNP,
     !RCZ     &                    INIT)
     !RCZ             CALL KOMEGAM(INIT)
     !RCZC   MUST STOP IF ONE OR MORE FATAL ERRORS (SIGNIFIED BY INIT=2)
     !RCZC   OCCURRED
     !RCZ          IF(INIT.EQ.2) THEN
     !RCZ             if(wrnlev.ge.2) WRITE(OUTU,4100)
     !RCZC     IF(IZ.NE.0) WRITE(IZ,4100)
     !RCZ 4100        FORMAT(//' ***** EXECUTION ENDING BECAUSE OF FATAL',
     !RCZ     .       ' ERROR(S) IN ASSIGNMENT OF PARAMETERS *****')
     !RCZ             CALL WRNDIE(-5,'<setup>','ERROR MSG')
     !RCZ          ELSE IF(INIT.EQ.1) THEN
     !RCZ            if(wrnlev.ge.2) WRITE(OUTU,4200)
     !RCZC     IF(IZ.NE.0) WRITE(IZ,4200)
     !RCZ 4200       FORMAT(//' *** WARNING: MISSING PARAMETERS DETECTED ***'/
     !RCZ     .      '     CALCULATION WILL PROCEED, BUT ACCURACY IS',
     !RCZ     .      ' UNCERTAIN')
     !RCZ          ENDIF
     !RCZC   NOW REGENERATE IN SINGLE-LISTING FORM (AS THOUGH ALL BONDS
     !RCZC   WERE SINGLE BONDS), BUT LEAVE LOCAT() SET AT THE NUMBER OF
     !RCZC   ATOMS ATTACHED TO EACH GIVEN ATOM (THE FIRST ARGUMENT OF -1
     !RCZC   HAS THIS EFFECT
     !RCZ          CALL TABINT(-1,IB,JB,BondType,NATOM,NBOND,ITAB,LOCAT)
     !RCZC          DO I=1,NATOM
     !RCZC             WRITE(6,'(7I5)') I,LOCAT(I),(ITAB(J,I),J=1,5)
     !RCZC          ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE SETUPMF
#else /**/
SUBROUTINE NULL_mmff
  RETURN
END SUBROUTINE NULL_mmff
#endif 

