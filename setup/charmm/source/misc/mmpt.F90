!Updated on 26/01/2015
!To integrate DPT code with standard MMPT

!Updated on 30/09/2015
!To simulate double proton transfer (DPT), e.g. in formic acid dimer (FAD),
!by using a mix of SDM and SSM potential to avoid tetra-minimums in PES.
!Switch function of sigma=2.639


! --------------------------------------------------------------------
! CHARMM Element source/misc/mmpt.src 1.3
! 11/11/01:
! - bugs fixed and MMPT can run in parallel with good scaling
! 
! 10/11/15:
! - upgrade to Fortran95-compatiable as CHARMM c36a5
! - include a new potentail NLM (see JCP, 133, 064503)
! 
! 09/11/28:
! - Wrapping of mmpt.src and mmpt.fcm into MODULE MMPT.
! - Convert all the fixed size FORTRAN77 arrays to 
!   allocatable ones. 
! - Fixed the bug that transfering protons have to be the fourth
! - Introducing a new potential LPE with the function
!   form V(r,rho,theta)=\sum V(r,rho)P(costheta)
! -------------------------------------------------------------------

module mmpt_fcm

  use chm_kinds
  implicit none

  !====================================================================
  ! MMPT GLOBAL VARIABLES
  !====================================================================
  !     SDMPRMU    - Unit number of MMPT SDM parameter file.
  !     SSMPRMU    - Unit number of MMPT SSM parameter file.
  !     ASMPRMU    - Unit number of MMPT ASM parameter file.
  !     QSDM       - FLAG TO INVOKE SDM POTENTIAL FUNCTION
  !     QSSM       - FLAG TO INVOKE SSM POTENTIAL FUNCTION
  !     QASM       - FLAG TO INVOKE ASM POTENTIAL FUNCTION
  !     QDPT       - FLAG TO INVOKE DPT POTENTIAL FUNCTION
  !     QANL       - FLAG TO INVOKE ANL POTENTIAL FUNCTION
  !     QMMPT      - MIKE DEVEREUX: GLOBAL FLAG TO ACTIVATE MMPT ROUTINES OR NOT
  !     MORE TO BE ADDED
  !      HBRDNUM -> HYDROGEN BRIDGE COUNTER


  INTEGER, SAVE :: SDMPRMU, SSMPRMU, ASMPRMU, LPEPRMU, NLMPRMU, ANLPRMU
  LOGICAL, SAVE :: QSSM, QSDM, QASM, QLPE, QNLM, QDPT, QANL
  LOGICAL, SAVE :: QMMPT = .FALSE.

  INTEGER, SAVE :: NPRMNHN, NPRMOHO, NPRMNHO, NPRMLPE, NPRMNLM, NPRMANL
  real(chm_real),allocatable,dimension(:),save :: PRMNHN, PRMOHO, &
     PRMNHO, PRMLPE, PRMNLM, PRMANL

  INTEGER, SAVE :: HBRDGU, HBRDNUM, ANGLNUM, DIHENUM, &
     IMPRNUM, NONBNUM, BDTRU,BDTRNUM
  INTEGER,ALLOCATABLE,DIMENSION(:),SAVE :: BONDLV,BDLIST
  INTEGER,ALLOCATABLE,DIMENSION(:,:),SAVE :: HBRDATOM, BONDMMPT, &
     ANGLMMPT, DIHEMMPT, IMPRMMPT, NONBMMPT, MMPTCHRG, INITCHRG
  CHARACTER*3,ALLOCATABLE,DIMENSION(:),SAVE :: POTTYPE
  CHARACTER*4,ALLOCATABLE,DIMENSION(:),SAVE :: BONDA,BONDB
  real(chm_real),allocatable,dimension(:),save :: BONDFC1,BONDFC2,BONDEQU1, &
  BONDEQU2

  real(chm_real), SAVE :: SCLOHO,SCLNHN,SCLNHO

  !====================================================================
  ! Leave global array definition and enter module subroutine section
  !====================================================================

contains

#if KEY_MMPT==1
  !====================================================================
  ! INITIALIZATION
  !====================================================================

  SUBROUTINE MMPTINIT

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
#if KEY_PARALLEL==1
    use parallel 
#endif

    INTEGER I,J,K,L,BC,MA,MD,MM, MULTI
    INTEGER ATOM1,ATOM2,ATOM3,ATOM4
    real(chm_real) AFCONST,TMIN,DFCONST,PMIN

    INTEGER AC,ITH
    INTEGER DC, IPHI
    INTEGER IC, STATU
!      CHARACTER*4 STRING 
    INTEGER HB, VAR1, VAR2
    real(chm_real) RX,RY,RZ,DHA
    
!   NEW PARAMETER FILES FOR HYDROGEN AND ACCEPTOR SIDE ATOMS
    INTEGER HPRMU,NAPAR,NDPAR
    INTEGER DUM1, DUM2, DUM3

    allocate(BDLIST(NATOM))

#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0) THEN  
#endif

    if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> MMPTINIT HAS BEEN CALLED.'

!   ALLOCATE THE ARRAIES FOR READING PARA
    CALL ALLOCFIR

    QSSM = .FALSE.
    QSDM = .FALSE. 
    QASM = .FALSE. 
    QLPE = .FALSE.
    QNLM = .FALSE.
    QANL = .FALSE.
    QDPT = .FALSE.
    QMMPT = .TRUE.
    
    HBRDGU = GTRMI(COMLYN,COMLEN,'UHBR',-1)
    IF (HBRDGU.EQ.-1) THEN
      CALL WrnDie (-3,'<MISCOM>',' No hydrogen bridge file for MMPT')
    ELSE
      HBRDNUM=0
      STATU=0
      DO WHILE (.TRUE.)
         HBRDNUM=HBRDNUM+1
         READ(UNIT=HBRDGU, FMT=*, IOSTAT=STATU) HBRDATOM(HBRDNUM,1),&
         HBRDATOM(HBRDNUM,2),HBRDATOM(HBRDNUM,3),POTTYPE(HBRDNUM)
         IF(STATU.NE.0) EXIT
      ENDDO
    ENDIF
    HBRDNUM=HBRDNUM-1   
    
        
    BDTRU = GTRMI(COMLYN,COMLEN,'UBDT',-1)
    BDTRNUM=0
    IF (BDTRU.EQ.-1) THEN
      WRITE (OUTU,*) 'MMPT> No bonding transformation in MMPT.'
    ELSE
      STATU=0
      DO WHILE (STATU.EQ.0)
         BDTRNUM=BDTRNUM+1
         READ(UNIT=BDTRU, FMT=*, IOSTAT=STATU) BONDA(BDTRNUM),BONDB(BDTRNUM), &
         BONDLV(BDTRNUM), BONDFC1(BDTRNUM), BONDEQU1(BDTRNUM),BONDFC2(BDTRNUM), &
         BONDEQU2(BDTRNUM)
         IF (STATU.EQ.0 .and. BONDLV(BDTRNUM).NE.1 .AND. BONDLV(BDTRNUM).NE.2) &
         CALL WrnDie (-3,'<MISCOM>','MMPT>  No beyond the second order of &
         neighbouring bonds.')
!write (outu,*) 'TEST> ',BDTRNUM,BONDA(BDTRNUM),BONDB(BDTRNUM)
      ENDDO
      BDTRNUM=BDTRNUM-1
    ENDIF
    


    if (prnlev >= 2) then
       WRITE(OUTU,*) ' '
       WRITE(OUTU,*) 'MMPT> FOUND',HBRDNUM,' HYDROGEN BOND(S) IN FILE:'
    endif
    DO J=1,HBRDNUM
      if (prnlev >= 2) WRITE(OUTU,*) 'MMPT>',HBRDATOM(J,1),HBRDATOM(J,2),HBRDATOM(J,3),POTTYPE(J)
      IF (POTTYPE(J) .EQ. 'SSM') THEN 
        QSSM = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'SDM') THEN 
        QSDM = .TRUE.
      ENDIF 
      IF (POTTYPE(J) .EQ. 'ASM') THEN 
        QASM = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'LPE') THEN 
        QLPE = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'NLM') THEN 
        QNLM = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'ANL') THEN 
        QANL = .TRUE.
      ENDIF
      IF (POTTYPE(J) .EQ. 'DPT') THEN 
        QDPT = .TRUE.
      ENDIF
    ENDDO

!   PLAUSIBILITY CHECK: DO ATOMS HAVE THE POSSIBLE DONOR, HYDROGEN AND
!   ACCEPTOR ATOM TYPE (IE FIRST AND LAST MUST BE OXYGEN OR NITROGEN
!   ATOMS AND MIDDLE MUST BE A HYDROGEN ATOM)
    DO I=1,HBRDNUM
      IF (ATYPE(HBRDATOM(I,1))(1:1) .EQ. 'N' .or. &
        ATYPE(HBRDATOM(I,1))(1:1) .EQ. 'O' ) THEN 
        CONTINUE
      ELSE
        WRITE(OUTU,*) ' ',ATYPE(HBRDATOM(I,1))(1:1)
        WRITE(OUTU,*) 'MMPT> FIRST ATOM IN HYDROGEN BOND IS NOT A DONOR ATOM'
        STOP
      ENDIF

      IF (ATYPE(HBRDATOM(I,3))(1:1) .EQ. 'N' .OR. &
        ATYPE(HBRDATOM(I,3))(1:1) .EQ. 'O' ) THEN 
        CONTINUE
      ELSE
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MMPT> THIRD ATOM IN HYDROGEN BOND IS NOT AN ACCEPTOR ATOM'
        STOP
      ENDIF

      IF (ATYPE(HBRDATOM(I,2))(1:1) .EQ. 'H') THEN 
        CONTINUE
      ELSE
        WRITE(OUTU,*) 'MMPT> SECOND ATOM IN HYDROGEN BOND IS NOT A HYDROGEN ATOM'
        STOP
      ENDIF
    ENDDO

!      PROCESS COMMAND LINE
!      READ IN PARAMETER SET / Symmetric Double Minimum (SDM)
!      DERIVED FROM N-H...N BOND IN N2H4+ PROTOTYPE PSF; HENCE NHN<->SDM
      IF (QSDM) THEN
        SDMPRMU = GTRMI(COMLYN,COMLEN,'USDM',-1)
        IF (SDMPRMU.EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No SDM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMNHN
            READ(SDMPRMU, *) PRMNHN(I)
20        ENDDO
          if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> SDM parameter set has been read.'
          IF(PRNLEV.GT.7) THEN
            DO I=1, NPRMNHN
              WRITE(OUTU,*) 'MMPT>  PARAM', I, PRMNHN(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

!      READ IN PARAMETER SET / Symmetric Single Minimum (SSM)
!      DERIVED FROM O-H...O BOND IN O2H5+ PROTOTYPE PSF; HENCE OHO<->SSM 
      
      IF (QSSM) THEN
        SSMPRMU = GTRMI(COMLYN,COMLEN,'USSM',-1)
        IF (SSMPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No SSM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMOHO
            READ(SSMPRMU, *) PRMOHO(I)
21        ENDDO
          if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> SSM parameter set has been read.'
          IF(PRNLEV.GT.7) THEN
            DO I=1, NPRMOHO
              WRITE(OUTU,*) 'MMPT>  PARAM', I, PRMOHO(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

!      READ IN PARAMETER SET / Symmetric Single Minimum (SSM)
!      DERIVED FROM O-H...O BOND IN NH4OH2+ PROTOTYPE PSF; HENCE NHO<->ASM 
      IF (QASM) THEN
        ASMPRMU = GTRMI(COMLYN,COMLEN,'UASM',-1)
        IF (ASMPRMU.EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No ASM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMNHO
            READ(ASMPRMU, *) PRMNHO(I)
22        ENDDO
          if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> ASM parameter set has been read.'
          IF(PRNLEV.GT.7) THEN 
            DO I=1, NPRMNHO
              WRITE(OUTU,*) 'MMPT>  PARAM', I, PRMNHO(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

      IF (QLPE) THEN
        LPEPRMU = GTRMI(COMLYN,COMLEN,'ULPE',-1)
        IF (LPEPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No LPE parameter file for MMPT')
        ELSE
          DO I=1 , NPRMLPE
            READ(LPEPRMU, *) PRMLPE(I)
23        ENDDO
          if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> LPE parameter set has been read.'
          IF(PRNLEV.GT.7) THEN
            DO I=1, NPRMLPE
              WRITE(OUTU,*) 'MMPT>  PARAM', I, PRMLPE(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

      IF (QNLM) THEN
        NLMPRMU = GTRMI(COMLYN,COMLEN,'UNLM',-1)
        IF (NLMPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No NLM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMNLM
            READ(NLMPRMU, *) PRMNLM(I)
24        ENDDO
          if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> NLM parameter set has been read.'
          IF(PRNLEV.GT.7) THEN
            DO I=1, NPRMNLM
              WRITE(OUTU,*) 'MMPT>  PARAM', I, PRMNLM(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF
      
!      READ IN PARAMETER SET / Double Proton Transfer (DPT)
!      DERIVED FROM O-H...O BOND IN NH4OH2+ PROTOTYPE PSF; HENCE NHO<->ASM 
      IF (QDPT) THEN
        SDMPRMU = GTRMI(COMLYN,COMLEN,'UDPT1',-1)
        IF (SDMPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No DPT-SDM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMNHN
            READ(SDMPRMU, *) PRMNHN(I)
25        ENDDO
          if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> DPT-SDM parameter set has been read.'
          IF(PRNLEV.GT.7) THEN
            DO I=1, NPRMNHN
              WRITE(OUTU,*) 'MMPT>  PARAM', I, PRMNHN(I)
            ENDDO
          ENDIF
        ENDIF
        
        SSMPRMU = GTRMI(COMLYN,COMLEN,'UDPT2',-1)
        IF (SSMPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No DPT-SSM parameter file for MMPT')
        ELSE
          DO I=1 , NPRMOHO
            READ(SSMPRMU, *) PRMOHO(I)
26        ENDDO
          if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> DPT-SSM parameter set has been read.'
          IF(PRNLEV.GT.7) THEN
            DO I=1, NPRMOHO
              WRITE(OUTU,*) 'MMPT>  PARAM', I, PRMOHO(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF

      
      IF (QANL) THEN
        ANLPRMU = GTRMI(COMLYN,COMLEN,'UANL',-1)
        IF (ANLPRMU .EQ.-1) THEN
          CALL WrnDie (-3,'<MISCOM>',' No ANL parameter file for MMPT')
        ELSE
          DO I=1 , NPRMANL
            READ(ANLPRMU, *) PRMANL(I)
27        ENDDO
          if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> ANL parameter set has been read.'
          IF(PRNLEV.GT.7) THEN
            DO I=1, NPRMANL
              WRITE(OUTU,*) 'MMPT>  PARAM', I, PRMANL(I)
            ENDDO
          ENDIF
        ENDIF
      ENDIF
!!scaling factor to add
      SCLOHO = GTRMF(COMLYN,COMLEN,'SSMS',ONE)
      IF (SCLOHO .EQ. ONE ) THEN
         if (prnlev >= 2) WRITE(OUTU,*)'MMPT> SSM ENERGIES WITHOUT SCALING'
      ELSE
         if (prnlev >= 2) WRITE(OUTU,*)'MMPT> SSM SCALING FACTOR IS ', SCLOHO
      ENDIF
      SCLNHN = GTRMF(COMLYN,COMLEN,'SDMS',ONE)
      IF (SCLNHN .EQ. ONE ) THEN
         if (prnlev >= 2) WRITE(OUTU,*)'MMPT> SDM ENERGIES WITHOUT SCALING'
      ELSE
         if (prnlev >= 2) WRITE(OUTU,*)'MMPT> SDM SCALING FACTOR IS ', SCLNHN
      ENDIF
      SCLNHO = GTRMF(COMLYN,COMLEN,'ADMS',ONE)
      IF (SCLNHO .EQ. ONE ) THEN
         if (prnlev >= 2) WRITE(OUTU,*)'MMPT> ASM ENERGIES WITHOUT SCALING'
      ELSE
         if (prnlev >= 2) WRITE(OUTU,*)'MMPT> ASM SCALING FACTOR IS ', SCLNHO
      ENDIF

!     ALLOCATE THE ARRAIES FOR MMPT BONDS, ANGLES, ETC. 
      CALL ALLOCSEC(HBRDNUM)

!      GET BOND NUMBERS OF DONOR AND H ATOM BONDS
      DO HB=1, HBRDNUM
        DO MM=1, NBOND
          IF (HBRDATOM(HB,1).EQ.IB(MM).AND.HBRDATOM(HB,2).EQ.JB(MM)) THEN
            BONDMMPT(HB,1)=MM
            BONDMMPT(HB,2)=IB(MM)
            BONDMMPT(HB,3)=JB(MM)
          ENDIF
        ENDDO
      ENDDO

!      GET ALL ANGLES AND ANGLE NUMBERS CONNECTED TO H BRIDGE
      ANGLNUM=0
      DO HB=1,HBRDNUM      
        DO ITH=1, NTHETA
!      GET THE ANGLES WHICH NEED TO BE SWITCHED AT 
!      DONOR SIDE (SEARCH FOR ANGLES IN PSF WHICH 
!      INVOLVE H ATOM)
!      SET SWITCH FLAG TO SOMETHING POSITIV
          IF (HBRDATOM(HB,2).EQ.IT(ITH) &
              .OR.HBRDATOM(HB,2).EQ.JT(ITH) &
              .OR.HBRDATOM(HB,2).EQ.KT(ITH)) THEN
             ANGLNUM=ANGLNUM+1
             ANGLMMPT(ANGLNUM,1)=ITH
             ANGLMMPT(ANGLNUM,2)=IT(ITH)
             ANGLMMPT(ANGLNUM,3)=JT(ITH)
             ANGLMMPT(ANGLNUM,4)=KT(ITH)   
             ANGLMMPT(ANGLNUM,5)=1
          ENDIF
        ENDDO
      ENDDO
!      GET THE ANGLES WHICH NEED TO BE SWITCHED AT
!      ACCEPTOR SIDE ( SEARCH FOR ATOMS THAT ARE
!      BONDED TO DONOR ATOM. USE TO SET UP NEW ANGLE)
!      SET SWITCH FLAG TO SOMETHING NEGATIV
      DO HB=1,HBRDNUM
        DO MM=1, NBOND
          IF (HBRDATOM(HB,3).EQ.IB(MM)) THEN
             ANGLNUM=ANGLNUM+1
             ANGLMMPT(ANGLNUM,1)=0
             ANGLMMPT(ANGLNUM,2)=JB(MM)
             ANGLMMPT(ANGLNUM,3)=IB(MM)
             ANGLMMPT(ANGLNUM,4)=HBRDATOM(HB,2)   
             ANGLMMPT(ANGLNUM,5)=-1
          ELSEIF (HBRDATOM(HB,3).EQ.JB(MM)) THEN
             ANGLNUM=ANGLNUM+1
             ANGLMMPT(ANGLNUM,1)=0
             ANGLMMPT(ANGLNUM,2)=IB(MM)
             ANGLMMPT(ANGLNUM,3)=JB(MM)
             ANGLMMPT(ANGLNUM,4)=HBRDATOM(HB,2)   
             ANGLMMPT(ANGLNUM,5)=-1 
          ENDIF
        ENDDO
      ENDDO


!      WE NEED TO GET THE CORRECT FORCE CONSTANTS. 
!      SEARCH FOR ANGLES OF THE SAME KIND I.E. SAME 
!      PARAMETER TYPE CODE (IAC) IN THE SAME ORDER. 
!      TO DO: IF YOU DO NOT HAVE ANY YOU HAVE TO FIND OUT HOW 
!      THE FORCE CONSTANT TABLES ARE CREATED AFTER
!      SYSTEMS HAS BEEN INITIALIZED. PSF -> ICT
      DO I=1,ANGLNUM
        IF (ANGLMMPT(I,1).EQ.0) THEN
          DO ITH=1, NTHETA
            IF (IAC(ANGLMMPT(I,2)).EQ.IAC(IT(ITH))        &
                 .AND.IAC(ANGLMMPT(I,3)).EQ.IAC(JT(ITH))   &
                 .AND.IAC(ANGLMMPT(I,4)).EQ.IAC(KT(ITH))) THEN
                ANGLMMPT(I,1)=ITH
            ENDIF
            IF (IAC(ANGLMMPT(I,4)).EQ.IAC(IT(ITH))       &
                 .AND.IAC(ANGLMMPT(I,3)).EQ.IAC(JT(ITH))  &
                 .AND.IAC(ANGLMMPT(I,2)).EQ.IAC(KT(ITH))) THEN
                ANGLMMPT(I,1)=ITH
            ENDIF
          ENDDO
        ENDIF
      ENDDO
      

!     GET ALL DIHEDRAL ANGLES CONNECTED TO H BRIDGE
      DIHENUM=0
!      GET DIHEDRALS WHICH NEED TO BE SWITCHED AT 
!      DONOR SIDE (SEARCH FOR DIHEDRALS IN PSF WHICH 
!      INVOLVE H ATOM)
!      SET SWITCH FLAG TO SOMETHING POSITIV
      DO HB=1,HBRDNUM      
        DO IPHI=1, NPHI
          IF (HBRDATOM(HB,2).EQ.IP(IPHI)      &
              .OR.HBRDATOM(HB,2).EQ.JP(IPHI)  &
              .OR.HBRDATOM(HB,2).EQ.KP(IPHI)  &
              .OR.HBRDATOM(HB,2).EQ.LP(IPHI)) THEN
             DIHENUM=DIHENUM+1
             DIHEMMPT(DIHENUM,1)=IPHI
             DIHEMMPT(DIHENUM,2)=IP(IPHI)
             DIHEMMPT(DIHENUM,3)=JP(IPHI)
             DIHEMMPT(DIHENUM,4)=KP(IPHI)
             DIHEMMPT(DIHENUM,5)=LP(IPHI)
             DIHEMMPT(DIHENUM,6)=1 
          ENDIF
        ENDDO
      ENDDO

!      GET DIHEDRALS WHICH NEED TO BE SWITCHED AT
!      ACCEPTOR SIDE ( SEARCH FOR ATOMS THAT FORM
!      ANGLE WITH DONOR ATOM. USE TO SET UP NEW DIHEDRAL)
!      SET SWITCH FLAG TO SOMETHING NEGATIV
     DO HB=1,HBRDNUM
         DO ITH=1,NTHETA
           IF (HBRDATOM(HB,3).EQ.IT(ITH) .AND. KT(ITH).NE.HBRDATOM(HB,2)) THEN
               DIHENUM=DIHENUM+1
               DIHEMMPT(DIHENUM,1)=0
               DIHEMMPT(DIHENUM,2)=KT(ITH)
               DIHEMMPT(DIHENUM,3)=JT(ITH)
               DIHEMMPT(DIHENUM,4)=IT(ITH)   
               DIHEMMPT(DIHENUM,5)=HBRDATOM(HB,2)
               DIHEMMPT(DIHENUM,6)=-1
!      NOT SURE WHAT HAPPENS IF ACCEPTOR ATOM IS SOMETHING
!      ELSE THAN FIRST OR LAST ATOM IN ORDER OF ANGLE DEFINITION
! ... IF NEEDED USE THIS WITH MODIFIED ORDER
           ELSEIF (HBRDATOM(HB,3).EQ.KT(ITH) .AND. IT(ITH).NE.HBRDATOM(HB,2)) THEN
               DIHENUM=DIHENUM+1
               DIHEMMPT(DIHENUM,1)=0
               DIHEMMPT(DIHENUM,2)=IT(ITH)
               DIHEMMPT(DIHENUM,3)=JT(ITH)
               DIHEMMPT(DIHENUM,4)=KT(ITH)   
               DIHEMMPT(DIHENUM,5)=HBRDATOM(HB,2)
               DIHEMMPT(DIHENUM,6)=-1 
           ENDIF
         ENDDO
      ENDDO

!      TO GET THE FORCE CONSTANTS
      DO I=1,DIHENUM
        IF (DIHEMMPT(I,1).EQ.0) THEN
!write(outu,*), 'TEST> 1-',IAC(DIHEMMPT(I,2)),IAC(DIHEMMPT(I,3)),IAC(DIHEMMPT(I,4)),IAC(DIHEMMPT(I,5))
!write(outu,*), 'TEST> 2-',   DIHEMMPT(I,2),   DIHEMMPT(I,3),   DIHEMMPT(I,4),   DIHEMMPT(I,5)
          DO IPHI=1, NPHI
!write(outu,*), 'IPHI > 1-',IAC(IP(IPHI)),IAC(JP(IPHI)),IAC(KP(IPHI)),IAC(LP(IPHI))
!write(outu,*), 'IPHI > 2-',IP(IPHI),JP(IPHI),KP(IPHI),LP(IPHI)
            IF (IAC(DIHEMMPT(I,2)).EQ.IAC(IP(IPHI))        &
                .AND.IAC(DIHEMMPT(I,3)).EQ.IAC(JP(IPHI))   &
                .AND.IAC(DIHEMMPT(I,4)).EQ.IAC(KP(IPHI))   &
                .AND.IAC(DIHEMMPT(I,5)).EQ.IAC(LP(IPHI)))  &
               THEN
               DIHEMMPT(I,1)=IPHI
            else if(IAC(DIHEMMPT(I,2)).EQ.IAC(LP(IPHI))        &
               .AND.IAC(DIHEMMPT(I,3)).EQ.IAC(KP(IPHI))   &
               .AND.IAC(DIHEMMPT(I,4)).EQ.IAC(JP(IPHI))   &
                .AND.IAC(DIHEMMPT(I,5)).EQ.IAC(IP(IPHI)))  &
              THEN
              DIHEMMPT(I,1)=IPHI
            ENDIF
          ENDDO
        ENDIF
      ENDDO


!     GET ALL IMPROPER DIHEDRAL ANGLES CONNECTED TO H BRIDGE
      IMPRNUM=0
      DO HB=1,HBRDNUM      
        DO IPHI=1, NIMPHI
!      GET DIHEDRALS WHICH NEED TO BE SWITCHED AT 
!      DONOR SIDE (SEARCH FOR DIHEDRALS IN PSF WHICH 
!      INVOLVE H ATOM)
!      SET SWITCH FLAG TO SOMETHING POSITIV
          IF (HBRDATOM(HB,2).EQ.IM(IPHI)       &
              .OR.HBRDATOM(HB,2).EQ.JM(IPHI)   &
              .OR.HBRDATOM(HB,2).EQ.KM(IPHI)   &
              .OR.HBRDATOM(HB,2).EQ.LM(IPHI)) THEN
             IMPRNUM=IMPRNUM+1
             IMPRMMPT(IMPRNUM,1)=IPHI
             IMPRMMPT(IMPRNUM,2)=IM(IPHI)
             IMPRMMPT(IMPRNUM,3)=JM(IPHI)
             IMPRMMPT(IMPRNUM,4)=KM(IPHI)
             IMPRMMPT(IMPRNUM,5)=LM(IPHI)
             IMPRMMPT(IMPRNUM,6)=1 
          ENDIF
        ENDDO
      ENDDO

!      GET IMPROPERS WHICH NEED TO BE SWITCHED AT
!      ACCEPTOR SIDE ( SEARCH FOR ATOMS THAT FORM
!      ANGLE WITH DONOR ATOM AND ATOM IS MIDDLE ATOM.
!.... USE ANGLE ATOMS TO SET UP NEW IMPROPER)
!      SET SWITCH FLAG TO SOMETHING NEGATIV
      DO HB=1,HBRDNUM
        DO ITH=1,NTHETA
          IF (HBRDATOM(HB,3).EQ.JT(ITH)) THEN
             IMPRNUM=IMPRNUM+1
             IMPRMMPT(IMPRNUM,1)=0
             IMPRMMPT(IMPRNUM,2)=JT(ITH)
             IMPRMMPT(IMPRNUM,3)=IT(ITH)
             IMPRMMPT(IMPRNUM,4)=KT(ITH)   
             IMPRMMPT(IMPRNUM,5)=HBRDATOM(HB,2)
             IMPRMMPT(IMPRNUM,6)=-1
          ENDIF
        ENDDO
      ENDDO

!      TO GET THE FORCE CONSTANTS
      DO I=1,IMPRNUM
        IF (IMPRMMPT(I,1).EQ.0) THEN
          DO IPHI=1, NIMPHI
            IF (IAC(IMPRMMPT(I,2)).EQ.IAC(IM(IPHI))       &
                 .AND.IAC(IMPRMMPT(I,3)).EQ.IAC(JM(IPHI))  &
                 .AND.IAC(IMPRMMPT(I,4)).EQ.IAC(KM(IPHI))  &
                 .AND.IAC(IMPRMMPT(I,5)).EQ.IAC(LM(IPHI))) &
                 THEN
                IMPRMMPT(I,1)=IPHI
            ENDIF
          ENDDO
        ENDIF
      ENDDO


!      IF H ATOM IS CONNECTED TO ATOMS ON ACCEPTOR SIDE SO THAT UNCOMMON
!      INTERACTIONS BETWEEN ATOM TYPES OCCUR THE USER NEEDS TO PROVIDE 
!      THEM: READ MISSING PARAMETER FROM FILE 
!      ANGLES:
!      THEY NEED TO BE ADDED TO CHARMMS INTERNAL STORAGE OF ANGLE 
!      PARAMETER. 
!      NCT NUMBER OF ANGLE PARAMETER
!      CTC FORCE CONSTANT
!      CTB EQUILIBRIUM ANGLE
!      THEY ARE ACCESSED TROUGH ANGLE CODE LOOK UP ARRAY -> ICT 
!     .WHERE ALL NTHETA ANGLE CODES ARE STORED
!      ICT(X) -> CTC(ICT(X)) AND CTB(ICT(X))
!      DIHEDRALS:

      HPRMU = GTRMI(COMLYN,COMLEN,'UHPM',-1)
      IF (HPRMU.EQ.-1) THEN
         if (prnlev >= 2) WRITE(OUTU,*) 'MMPT> NO HBOND PARAMETER FILE PROVIDED!'
      ELSE
!     SET NUMBER OF MISSING ANGLE RESP DIHEDRAL PARAMETER
        MA=1
        MD=1
!     READ NEW ANGLE PARAMETER
        if (prnlev >= 2) WRITE(OUTU,*) 'READING ANGLE PARMS FROM UNIT', HPRMU
!       todo: check if file exists
        READ(UNIT=HPRMU,FMT=*) NAPAR
        if (prnlev >= 2) WRITE(OUTU,*) 'NUMBER OF ANGLE PARMS TO BE READ:',NAPAR
        DO I=1,NAPAR
          if (prnlev >= 2) WRITE(OUTU,*) 'READING LINE',I
          READ(UNIT=HPRMU,FMT=*) ATOM1,ATOM2,ATOM3, AFCONST, TMIN
          if (prnlev >= 2) WRITE(UNIT=OUTU,FMT=*) ATOM1,ATOM2,ATOM3, AFCONST, TMIN            
          DO J=1,ANGLNUM
            IF (ATOM1.EQ.IAC(ANGLMMPT(J,2)).AND. &
                ATOM2.EQ.IAC(ANGLMMPT(J,3)).AND.  &
                ATOM3.EQ.IAC(ANGLMMPT(J,4))) THEN
!     SET ANGLE NUMBER 
                  ANGLMMPT(J,1)=NTHETA+MA
!     EXTEND ICT
                  ICT(NTHETA+MA)=NCT+MA
!     EXTEND CTC AND CTB
                  CTC(NCT+MA)=AFCONST
                  CTB(NCT+MA)=TMIN*PI/180.D0
!     INCREASE MISSING ANGLE COUNTER
                  MA=MA+1
            ENDIF
          ENDDO
        ENDDO
!     READ DIHEDRAL PARAMETER
        NDPAR=0
        READ(UNIT=HPRMU,FMT=*,END=40) NDPAR
        if (prnlev >= 2) WRITE(OUTU,*) 'NUMBER OF DIHE PARMS TO BE READ:',NDPAR
        DO I=1,NDPAR
          READ(UNIT=HPRMU,FMT=*,END=40) ATOM1,ATOM2,ATOM3,ATOM4, DFCONST, MULTI, PMIN                               
          if (prnlev >= 2) WRITE(UNIT=OUTU,FMT=*) ATOM1,ATOM2,ATOM3,ATOM4, DFCONST, MULTI, PMIN                               
          DO J=1,DIHENUM                                             
            IF (ATOM1.EQ.IAC(DIHEMMPT(J,2)).AND.                   &
                  ATOM2.EQ.IAC(DIHEMMPT(J,3)).AND.                  &
                  ATOM3.EQ.IAC(DIHEMMPT(J,4)).AND.                  &
                  ATOM4.EQ.IAC(DIHEMMPT(J,5))) THEN
!     SET DIHEDRAL NUMBER 
                  DIHEMMPT(J,1)=NPHI+MD
!     EXTEND ICP
                  ICP(NPHI+MD)=NCP+MD
!     EXTEND CPC, CPD, AND CPB 
                  CPC(NCP+MD)=DFCONST
                  CPD(NCP+MD)=MULTI
                  CPB(NCP+MD)=PMIN*PI/180.D0
                  CPCOS(NCP+MD)=cos(CPB(NCP+MD))
                  CPSIN(NCP+MD)=sin(CPB(NCP+MD))
!     INCREASE MISSING DIHEDRAL COUNTER
                  MD=MD+1
            ENDIF
          ENDDO
        ENDDO
      ENDIF
  
!95   FORMAT(I4)
!96   FORMAT(3I4,2F8.4)
!97   FORMAT(4I4,F8.4,I4,F8.4)
!98   FORMAT(4I4,2X,F8.4,2X,I4,2X,F8.4)
40   CONTINUE
 
 
!      IF NO FORCE PARAMETER HAVE BEEN FOUND THEN STOP 
      DO I=1,ANGLNUM
         IF (ANGLMMPT(I,1).EQ.0) THEN
            if (prnlev >= 2) then
               WRITE(OUTU,*) '<MMPTINIT> COULD NOT FIND ANGLE PARAMETER.'
               WRITE(OUTU,*) '           PSF AND TYPE:',       &
                    ANGLMMPT(I,2),' ',ATYPE(ANGLMMPT(I,2)),' ',  &
                    IAC(ANGLMMPT(I,2)),                         &
                    ANGLMMPT(I,3),' ',ATYPE(ANGLMMPT(I,3)),' ',  &   
                    IAC(ANGLMMPT(I,3)),                         &
                    ANGLMMPT(I,4),' ',ATYPE(ANGLMMPT(I,4)),' ',  &   
                    IAC(ANGLMMPT(I,4))     
               WRITE(OUTU,*) '           PLEASE PROVIDE PARAMETER!'
            endif
        ENDIF
     ENDDO


     DO I=1,DIHENUM
        IF (DIHEMMPT(I,1).EQ.0) THEN
           if (prnlev >= 2) then
              WRITE(OUTU,*) '<MMPTINIT> COULD NOT FIND DIHEDRAL PARAMETER'
              WRITE(OUTU,*) '           PSF AND TYPE:',       &
                   DIHEMMPT(I,2),' ',ATYPE(DIHEMMPT(I,2)),' ',  &
                   IAC(DIHEMMPT(I,2)),                         &
                   DIHEMMPT(I,3),' ',ATYPE(DIHEMMPT(I,3)),' ',  &   
                   IAC(DIHEMMPT(I,3)),                         &
                   DIHEMMPT(I,4),' ',ATYPE(DIHEMMPT(I,4)),' ',  &   
                   IAC(DIHEMMPT(I,4)),                         &
                   DIHEMMPT(I,5),' ',ATYPE(DIHEMMPT(I,5)),' ',  &   
                   IAC(DIHEMMPT(I,5))      
              WRITE(OUTU,*) '           PLEASE PROVIDE PARAMETER!'
           endif
        ENDIF
      ENDDO

      DO I=1,ANGLNUM
         IF (ANGLMMPT(I,1).EQ.0) THEN
            STOP
         ENDIF
      ENDDO

      DO I=1,DIHENUM
         IF (DIHEMMPT(I,1).EQ.0) THEN
            STOP
         ENDIF
      ENDDO

!     INITIALIZATION OF NON-BOND INTERACTIONS ARE CARRIED OUT
!     IN SEPERATED SOUBROUTINE NBONDINIT
      CALL NBONDINIT

!     FCM
!     FLUCTUATING CHARGE MODEL - COMMENT OUT IN STARNARD MMPT
!      STORE PSF CHARGES FOR DIPOLMOMENT CALCULATIONS
!       WRITE(OUTU,*) '    '
!       WRITE(OUTU,*) 'MMPT> STORED INITIAL PSF CHARGES FOR DIPOL  &
!       CALCULATION'
!       DO I=1, HBRDNUM
!          INITCHRG(I,1)=CG(HBRDATOM(I,1))
!          INITCHRG(I,2)=CG(HBRDATOM(I,2))
!          INITCHRG(I,3)=CG(HBRDATOM(I,3))
!          WRITE(OUTU,*)'', INITCHRG(I,1),INITCHRG(I,2), INITCHRG(I,3)
!       ENDDO


!     EXCLUDE IMPROPERS FOR NOW. SET IMPROPER NUMBER TO ZERO 
      IMPRNUM=0 
!      WRITE(OUTU,*) 'MMPT> IMPROPER DIHEDRALS HAVE BEEN SWITCHED OFF!'

!     PRINT OUT THE MMPTINIT INFORMATIONS
      if (prnlev >= 2) then
         WRITE(OUTU,*) ' ' 
         WRITE(OUTU,*) 'MMPT> ENERGIES AND FORCES OF FOLLOWING '
         WRITE(OUTU,*) '      INTERACTIONS WILL BE REMOVED OR MODIFIED'
         WRITE(OUTU,*) '      '
         WRITE(OUTU,*) '      BONDED TERMS: FLAG  1  MEANS TERM EXISTS'
         WRITE(OUTU,*) '                    FLAG -1  MEANS TERM IS NEW'
         WRITE(OUTU,*) '      BONDS:'
         WRITE(OUTU,*) '      NO    ATOM I     ATOM J' 
         DO I=1, HBRDNUM
            WRITE(OUTU,101) BONDMMPT(I,1),BONDMMPT(I,2),ATYPE(BONDMMPT(I,2)), &
                 BONDMMPT(I,3),ATYPE(BONDMMPT(I,3))                  
         ENDDO
         WRITE(OUTU,*) '      ANGLES:'                                      
         WRITE(OUTU,*) '      NO    ATOM I     ATOM J     ATOM K   FLAG'    
         DO I=1,ANGLNUM                                                     
            WRITE(OUTU,102) ANGLMMPT(I,1),ANGLMMPT(I,2),ATYPE(ANGLMMPT(I,2)),  &
                 ANGLMMPT(I,3),ATYPE(ANGLMMPT(I,3)),                &
                 ANGLMMPT(I,4),ATYPE(ANGLMMPT(I,4)),ANGLMMPT(I,5)
         ENDDO
         WRITE(OUTU,*) '      DIHEDRALS:'
         WRITE(OUTU,*) '      NO    ATOM I    ATOM J      ATOM K     ATOM' &
              ,' L   FLAG'                                           
         DO I=1,DIHENUM                                                      
            WRITE(OUTU,103) DIHEMMPT(I,1),DIHEMMPT(I,2),ATYPE(DIHEMMPT(I,2)),   &
                 DIHEMMPT(I,3),ATYPE(DIHEMMPT(I,3)),                 &
                 DIHEMMPT(I,4),ATYPE(DIHEMMPT(I,4)),                 &
                 DIHEMMPT(I,5),ATYPE(DIHEMMPT(I,5)),DIHEMMPT(I,6)     
         ENDDO
         WRITE(OUTU,*) '      IMPROPERS:'                                    
         WRITE(OUTU,*) '      NO    ATOM I    ATOM J      ATOM K    ATOM'   &
              ,' L   FLAG'                                           
         DO I=1,IMPRNUM                                                      
            WRITE(OUTU,103) IMPRMMPT(I,1),IMPRMMPT(I,2),ATYPE(IMPRMMPT(I,2)),   &
                 IMPRMMPT(I,3),ATYPE(IMPRMMPT(I,3)),                 &
                 IMPRMMPT(I,4),ATYPE(IMPRMMPT(I,4)),                 &
                 IMPRMMPT(I,5),ATYPE(IMPRMMPT(I,5)),IMPRMMPT(I,6)
         ENDDO
         WRITE(OUTU,*) '      '
         WRITE(OUTU,*) '      NONBONDED TERMS: FLAG  1  MEANS TERM IS NEW'
         WRITE(OUTU,*) '                       FLAG -1  MEANS TERM EXISTS'
         WRITE(OUTU,*) '      SPECIAL 1-4 VDW: FLAG  14  MEANS TERM IS NEW'
         WRITE(OUTU,*) '      SPECIAL 1-4 VDW: FLAG -14 MEANS TERM EXISTS'
         WRITE(OUTU,*) '      '
         WRITE(OUTU,*) '      NONBONDED:'
         WRITE(OUTU,*) '      NO    ATOM I    ATOM J    FLAG'
         DO I=1,NONBNUM
            WRITE(OUTU,104) NONBMMPT(I,1),NONBMMPT(I,2),ATYPE(NONBMMPT(I,2)),&
                 NONBMMPT(I,3),ATYPE(NONBMMPT(I,3)),NONBMMPT(I,4)   
         ENDDO
      endif
      
      deallocate(BDLIST)
                                                                       
101 FORMAT('      ',I5,1X,I6,1X,A4,2X,I6,1X,A4)                       
102 FORMAT('      ',I5,1X,I6,1X,A4,2X,I6,1X,A4,2X,I6,1X,A4,1X,I6)     
103 FORMAT('      ',I5,1X,I6,1X,A4,2X,I6,1X,A4,2X,I6,1X,A4,2X,I6,1X,A4,1X,I6)
104 FORMAT('      ',I5,1X,I6,1X,A4,2X,I6,1X,A4,1X,I6)

#if KEY_PARALLEL==1
    ENDIF 
#endif

  END SUBROUTINE MMPTINIT


!      ______________________________________________________________

!      NBONDED INTERACTIONS AFFECTED BY MMPT

  SUBROUTINE NBONDINIT

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image
    use inbnd

    INTEGER H,J,STORE,BDTMP
    INTEGER NBPREV,L,I,NB,HB

    NONBNUM=0

! SET HYDROGEN - ACCEPTOR NONB INTERACTION
    DO H=1,HBRDNUM
      call BDINIT(NATOM)
      BDTMP=BONDDIST(HBRDATOM(H,2),HBRDATOM(H,3))
!write(outu,*) 'TETS> BDIST = ',BDTMP
      IF (BDTMP.GT.3 .OR. (BDTMP.EQ.3 .AND. NBXMOD.EQ.3)) THEN
        NONBNUM=NONBNUM+1
        NONBMMPT(NONBNUM,1)=NONBNUM
        NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
        NONBMMPT(NONBNUM,3)=HBRDATOM(H,3)
        NONBMMPT(NONBNUM,4)=0
      ELSE IF (BDTMP.EQ.3 .AND. NBXMOD.EQ.5) THEN
        NONBNUM=NONBNUM+1
        NONBMMPT(NONBNUM,1)=NONBNUM
        NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
        NONBMMPT(NONBNUM,3)=HBRDATOM(H,3)
        NONBMMPT(NONBNUM,4)=28
      ENDIF
    ENDDO

! SET DONOR - ACCEPTOR NONB INTERACTION
    DO H=1,HBRDNUM
      call BDINIT(NATOM)
      BDTMP=BONDDIST(HBRDATOM(H,1),HBRDATOM(H,3))
      IF (BDTMP.GT.3 .OR. (BDTMP.EQ.3 .AND. NBXMOD.EQ.3)) THEN
        NONBNUM=NONBNUM+1
        NONBMMPT(NONBNUM,1)=NONBNUM
        NONBMMPT(NONBNUM,2)=HBRDATOM(H,1)
        NONBMMPT(NONBNUM,3)=HBRDATOM(H,3)
        NONBMMPT(NONBNUM,4)=0
      ELSE IF (BDTMP.EQ.3 .AND. NBXMOD.EQ.5) THEN
        NONBNUM=NONBNUM+1
        NONBMMPT(NONBNUM,1)=NONBNUM
        NONBMMPT(NONBNUM,2)=HBRDATOM(H,1)
        NONBMMPT(NONBNUM,3)=HBRDATOM(H,3)
        NONBMMPT(NONBNUM,4)=28
      ENDIF   
    ENDDO

! SET ANGLE CONNECTED ATOMS - HYDROGEN NONB INTERACTION
! FOR EACH HBOND SEARCH IN EACH ANGLE TERM THE ATOM WHICH IS NOT
! PART OF HBOND 
! ASSUMES THAT ANGLE SET UP IS ALWAYS OF THE KIND XA-A-H, OR H-A-XA
! ON ACCEPTOR SIDE OR SIMILAR ON DONOR SIDE LIKE XD-D-H, OR H-D-XD
! WHERE A AND D STAND FOR ACEPTOR, RESP DONOR, H IS HYDROGEN ATOM AND
! XD, RESP XA IS THE THIRD ATOM IN THE STRUCTURE FORMING THE ANGLE
!
! SOME NONBONDED INTERACTIONS ARE NOT COMPUTED. CHARMM HAS CERTAIN
! EXCLUSION MODES (CONTROLLED THROUGH NBXMOD KEYWORD): 
! NBXMOD 3 EXCLUDES 1-2 AND 1-3 INTERACTION, IE EXCLUDE NONBONDED 
! CONTRIBUTION FROM ATOMS CONNECTED TROUGHT BONDS OR ANGLES    
! NBXMOD 4 ADDITIONALLY EXCLUDES 1-4 INTERACTIONS, IE ATOMS CONNECTED
! TROUGH DIHEDRAL ANGLES
!
! CORRECT PROCESSING OF EXCLUSION LISTS OF NONBONDED INTERACTIONS HAS 
! BEEN IMPLEMENTED ONLY FOR MODES 3,4,AND 5 
! SHOULD USE THE DEFAULT 5
! ------------------------- NBXMOD 3 -------------------------------

    IF ((NBXMOD .EQ. 3) .OR. (NBXMOD .EQ. 5)) THEN 
      DO I=1,ANGLNUM
         if (prnlev >= 2) WRITE(OUTU,*) I, ANGLMMPT(I,2),ANGLMMPT(I,3),ANGLMMPT(I,4)
      ENDDO

      DO I=1,ANGLNUM
        DO H=1,HBRDNUM
!   HANDLE 1-3 NONB INTERACTION FOR H-A-X ORDER
          IF (ANGLMMPT(I,2).EQ.HBRDATOM(H,2)) THEN
            NONBNUM=NONBNUM+1
            NONBMMPT(NONBNUM,1)=NONBNUM
            NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
            NONBMMPT(NONBNUM,3)=ANGLMMPT(I,4)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE 
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE 
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
            IF (ANGLMMPT(I,5).EQ.1) THEN
              NONBMMPT(NONBNUM,4)=-1
              call BDINIT(NATOM)
              BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,3))+1
              IF (BDTMP.EQ.3 .AND. (NBXMOD .EQ. 5)) THEN
                NONBMMPT(NONBNUM,4)=-14
              ELSE IF (BDTMP.LT.3) THEN
                NONBNUM=NONBNUM-1
              ENDIF
            ELSE
              NONBMMPT(NONBNUM,4)=1
              call BDINIT(NATOM)
              BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,1))+1
              IF (BDTMP.EQ.3 .AND. (NBXMOD .EQ. 5)) THEN
                NONBMMPT(NONBNUM,4)=14
              ELSE IF (BDTMP.LT.3) THEN
                NONBNUM=NONBNUM-1
              ENDIF
            ENDIF
!   HANDLE 1-3 NONB INTERACTION FOR X-A-H ORDER
          ELSE IF (ANGLMMPT(I,4).EQ.HBRDATOM(H,2)) THEN
            NONBNUM=NONBNUM+1
            NONBMMPT(NONBNUM,1)=NONBNUM
            NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
            NONBMMPT(NONBNUM,3)=ANGLMMPT(I,2)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE 
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE 
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
            IF (ANGLMMPT(I,5).EQ.1) THEN
              NONBMMPT(NONBNUM,4)=-1
              call BDINIT(NATOM)
              BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,3))+1
              IF (BDTMP.EQ.3 .AND. (NBXMOD .EQ. 5)) THEN
                NONBMMPT(NONBNUM,4)=-14
              ELSE IF (BDTMP.LT.3) THEN
                NONBNUM=NONBNUM-1
              ENDIF
            ELSE
              NONBMMPT(NONBNUM,4)=1
              call BDINIT(NATOM)
              BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,1))+1
              IF (BDTMP.EQ.3 .AND. (NBXMOD .EQ. 5)) THEN
                NONBMMPT(NONBNUM,4)=14
              ELSE IF (BDTMP.LT.3) THEN
                NONBNUM=NONBNUM-1
              ENDIF
            ENDIF
          ENDIF
         ENDDO
      ENDDO
    ENDIF

!   ------------- EXTENSION NBXMOD 4 ---------------------------------

    IF (NBXMOD .EQ. 4) THEN     
      WRITE (OUTU,*) "WARNING: BONDDIST IS NOT YET TESTED FOR NBXMOD-4"
      DO I=1,DIHENUM
        DO H=1, HBRDNUM

!   HANDLE 1-4 NONB INTERACTION FOR H-A-XA-XB ORDER
          IF (DIHEMMPT(I,2) .EQ. HBRDATOM(H,2)) THEN
             NONBNUM=NONBNUM+1
             NONBMMPT(NONBNUM,1)=NONBNUM
             NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
             NONBMMPT(NONBNUM,3)=DIHEMMPT(I,5)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   DIHEDRAL TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW DIHEDRAL TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE 
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE 
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
            IF (DIHEMMPT(I,6).EQ.1) THEN
              NONBMMPT(NONBNUM,4)=-1
              call BDINIT(NATOM)
              BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,3))+1
              IF (BDTMP.LT.3) THEN
                NONBNUM=NONBNUM-1
              ENDIF
            ELSE
              NONBMMPT(NONBNUM,4)=1
              call BDINIT(NATOM)
              BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,1))+1
              IF (BDTMP.LT.3) THEN
                NONBNUM=NONBNUM-1
              ENDIF
            ENDIF
!   HANDLE 1-4 NONB INTERACTION FOR XA-XB-A-H ORDER
          ELSE IF (DIHEMMPT(I,5).EQ.HBRDATOM(H,2)) THEN
                NONBNUM=NONBNUM+1
                NONBMMPT(NONBNUM,1)=NONBNUM
                NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
                NONBMMPT(NONBNUM,3)=DIHEMMPT(I,2)
!   USE ANGLE FLAGS TO SET NB FLAGS
!   ANGLE TERMS ON DONOR SIDE HAVE FLAG ONE
!   NEW ANGLE TERMS ON ACCEPTOR SIDE HAVE FLAG MINUS ONE 
!   NONB TERMS ON ACCEPTOR SIDE HAVE FLAG ONE 
!   NEW NONB TERMS ON DONOR SIDE HAVE FLAG MINUS ONE
            IF (DIHEMMPT(I,6).EQ.1) THEN
              NONBMMPT(NONBNUM,4)=-1
              call BDINIT(NATOM)
              BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,3))+1
              IF (BDTMP.LT.3) THEN
                NONBNUM=NONBNUM-1
              ENDIF
            ELSE
              NONBMMPT(NONBNUM,4)=1
              call BDINIT(NATOM)
              BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,1))+1
              IF (BDTMP.LT.3) THEN
                NONBNUM=NONBNUM-1
              ENDIF
            ENDIF
          ENDIF
        ENDDO
      ENDDO
    ENDIF


!   ------------- EXTENSION NBXMOD 5 ---------------------------------
!   IF TRANSFERED HYDROGEN ATOM IS CONNECTED TO OTHER ATOMS THROUGH 
!   DIHEDRAL ANGLES *AND* IF THESE ATOM PAIRS USE SPECIAL 1-4 VDW 
!   PARAMETERS THESE 1-4 INTERACTIONS MUST BE SWITCHED FROM SPECIAL
!   TO STANDARD ON THE DONOR SIDE AND VICE VERSA ON THE ACCEPTOR SIDE

    IF (NBXMOD .EQ. 5) THEN     
!   HANDLE 1-4 NONB EXCLUSION (NBXMOD 5)     
      DO I=1,DIHENUM
        DO H=1, HBRDNUM
!   HANDLE 1-4 NONB INTERACTION FOR H-A-XA-XB ORDER
          IF (DIHEMMPT(I,2) .EQ. HBRDATOM(H,2)) THEN
             NONBNUM=NONBNUM+1
             NONBMMPT(NONBNUM,1)=NONBNUM
             NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
             NONBMMPT(NONBNUM,3)=DIHEMMPT(I,5)
!   USE DIHEDRAL FLAGS TO SET NB FLAGS
!   SPECIAL 1-4 TERMS ON DONOR SIDE HAVE FLAG 14
!   NEW SPECIAL TERMS ON ACCEPTOR SIDE HAVE FLAG -14
            IF (DIHEMMPT(I,6).EQ.1) THEN
               NONBMMPT(NONBNUM,4)=14
               call BDINIT(NATOM)
               BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,3))+1
!Write(*,*) "BDIST1>", BDTMP,NONBMMPT(NONBNUM,3),HBRDATOM(H,3)
               IF (BDTMP.GT.3) THEN
                 NONBNUM=NONBNUM+1
                 NONBMMPT(NONBNUM,1)=NONBNUM
                 NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
                 NONBMMPT(NONBNUM,3)=DIHEMMPT(I,5)
                 NONBMMPT(NONBNUM,4)=-1
               ELSE IF (BDTMP.EQ.3) THEN
                 NONBNUM=NONBNUM-1
               ENDIF
            ELSE
               NONBMMPT(NONBNUM,4)=-14
               call BDINIT(NATOM)
               BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,1))+1
!Write(*,*) "BDIST2>", BDTMP,NONBMMPT(NONBNUM,3),HBRDATOM(H,1)
               IF (BDTMP.GT.3) THEN
                 NONBNUM=NONBNUM+1
                 NONBMMPT(NONBNUM,1)=NONBNUM
                 NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
                 NONBMMPT(NONBNUM,3)=DIHEMMPT(I,5)
                 NONBMMPT(NONBNUM,4)=1
               ELSE IF (BDTMP.EQ.3) THEN
                 NONBNUM=NONBNUM-1
               ENDIF
            ENDIF
!   HANDLE 1-4 NONB INTERACTION FOR XA-XB-A-H ORDER
          ELSE IF (DIHEMMPT(I,5).EQ.HBRDATOM(H,2)) THEN
                NONBNUM=NONBNUM+1
                NONBMMPT(NONBNUM,1)=NONBNUM
                NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
                NONBMMPT(NONBNUM,3)=DIHEMMPT(I,2)
!   USE DIHEDRAL FLAGS TO SET NB FLAGS
!   SPECIAL 1-4 TERMS ON DONOR SIDE HAVE FLAG 14
!   NEW SPECIAL 1-4  TERMS ON ACCEPTOR SIDE HAVE FLAG -14
            IF (DIHEMMPT(I,6).EQ.1) THEN
               NONBMMPT(NONBNUM,4)=14
               call BDINIT(NATOM)
               BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,3))+1
!Write(*,*) "BDIST3>", BDTMP,NONBMMPT(NONBNUM,3),HBRDATOM(H,3)
               IF (BDTMP.GT.3) THEN
                 NONBNUM=NONBNUM+1
                 NONBMMPT(NONBNUM,1)=NONBNUM
                 NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
                 NONBMMPT(NONBNUM,3)=DIHEMMPT(I,2)
                 NONBMMPT(NONBNUM,4)=-1
               ELSE IF (BDTMP.EQ.3) THEN
                 NONBNUM=NONBNUM-1
               ENDIF
            ELSE
               NONBMMPT(NONBNUM,4)=-14
               call BDINIT(NATOM)
               BDTMP=BONDDIST(NONBMMPT(NONBNUM,3),HBRDATOM(H,1))+1
!Write(*,*) "BDIST4>", BDTMP,NONBMMPT(NONBNUM,3),HBRDATOM(H,1)
               IF (BDTMP.GT.3) THEN
                 NONBNUM=NONBNUM+1
                 NONBMMPT(NONBNUM,1)=NONBNUM
                 NONBMMPT(NONBNUM,2)=HBRDATOM(H,2)
                 NONBMMPT(NONBNUM,3)=DIHEMMPT(I,2)
                 NONBMMPT(NONBNUM,4)=1
               ELSE IF (BDTMP.EQ.3) THEN
                 NONBNUM=NONBNUM-1
               ENDIF
            ENDIF
          ENDIF
        ENDDO
      ENDDO
    ENDIF


!   SORT NB PAIRS BY ATOM TYPE, I.E. TRANSFERED HYDROGEN
!   COMES SECOND
    DO H=1,HBRDNUM
      DO I=1,NONBNUM
        IF (NONBMMPT(I,2).NE.HBRDATOM(H,2)) THEN
              CONTINUE
        ELSE
          STORE=NONBMMPT(I,2)
          NONBMMPT(I,2)=NONBMMPT(I,3)
          NONBMMPT(I,3)=STORE
        ENDIF
      ENDDO
    ENDDO


  END SUBROUTINE NBONDINIT


! ... _______________________________________________________

  SUBROUTINE EBONDMMPT(MM,I,J,EB,DXBRM,DYBRM,DZBRM)

! ... routine calculates classical bond energy terms
! ... and forces for specific bonds 
!      results or stored in ebmmpt and dxbrm, dxbrm
! ... based on charmm ebond subroutine


    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image
    

    INTEGER MM      
    real(chm_real) EB,RX,RY,RZ,S1,S2,DB,DF,DXBRM,DYBRM,DZBRM
    real(chm_real) R,ERM
    INTEGER I,II,J,IC
    
    ERM=0.0D0
    IC=ICB(MM)

    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)
    S2=RX*RX + RY*RY + RZ*RZ
    S1=SQRT(S2)
 
    DB=S1-CBB(IC)
    DF=CBC(IC)*DB
    EB=DF*DB
 
    R=2.D0/S1
    DF=DF*R 

    DXBRM=DF*RX
    DYBRM=DF*RY
    DZBRM=DF*RZ

    IF(PRNLEV.GT.7) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> BOND ATOMS ATOM I',I,'  ATOM J', J
      WRITE(OUTU,*) 'MMPT> BOND ENERGY',EB
      WRITE(OUTU,*) 'MMPT> BOND FORCE ATOM I',DXBRM,DYBRM,DZBRM
      WRITE(OUTU,*) 'MMPT> BOND FORCE ATOM J',-DXBRM,-DYBRM,-DZBRM
      WRITE(OUTU,*) ' '
      IF(PRNLEV.GT.8) THEN  
        WRITE(OUTU,*) 'MMPT> BOND PARAMETERS USED',CBB(IC),CBC(IC)
        WRITE(OUTU,*) ' '
      ENDIF
    ENDIF

  END SUBROUTINE EBONDMMPT
! ... _______________________________________________________


  SUBROUTINE EANGLMMPT(M,I,J,K,EA,DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)
!     THIS ROUTINE CALCULATES THE ENERGY AND FORCE OF A BOND ANGLE
!     OF ATOMS CONNECTED TO A TRANSFER HYDROGEN ATOM.
!     THE ROUTINE IS BASED ON THE EANGLE SUBROUTINE BY BROOKS.
!     SVEN.LAMMERS@UNIBAS.CH AUG 2003

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image    

    INTEGER M,I,J,K,IC

    real(chm_real) RI2,RJ2,RI,RJ,RIR,RJR,          &
          DXI,DYI,DZI,DXJ,DYJ,DZJ,         &
          DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,   &
          CST, AT, STR, ST2R,              &
          DA,DF,EA,                        &
          DTXI,DTYI,DTZI,DTXJ,DTYJ,DTZJ,   &
          DFX,DGX,DFY,DGY,DFZ,DGZ

    


    real(chm_real) DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK

    EA=0.0D0
    IC=ICT(M)

    DXI=X(I)-X(J)
    DYI=Y(I)-Y(J)
    DZI=Z(I)-Z(J)
    DXJ=X(K)-X(J)
    DYJ=Y(K)-Y(J)
    DZJ=Z(K)-Z(J)
    
    RI2=DXI*DXI+DYI*DYI+DZI*DZI
    RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ
    
    RI=SQRT(RI2)
    RJ=SQRT(RJ2)

    RIR=1.D0/RI
    RJR=1.D0/RJ
    
    DXIR=DXI*RIR
    DYIR=DYI*RIR
    DZIR=DZI*RIR
    DXJR=DXJ*RJR
    DYJR=DYJ*RJR
    DZJR=DZJ*RJR
    
    CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR
    
    AT=ACOS(CST)
    
    DA=AT-CTB(IC)
    DF=CTC(IC)*DA
    
    EA=DF*DA
    DF=DF+DF
 
    ST2R=1.D0/(1.D0-CST*CST)
    STR=SQRT(ST2R)
    DF=-DF*STR    
  
    DTXI=RIR*(DXJR-CST*DXIR)
    DTXJ=RJR*(DXIR-CST*DXJR)
    DTYI=RIR*(DYJR-CST*DYIR)
    DTYJ=RJR*(DYIR-CST*DYJR)
    DTZI=RIR*(DZJR-CST*DZIR)
    DTZJ=RJR*(DZIR-CST*DZJR)
    
    DFX=DF*DTXI
    DGX=DF*DTXJ

    DFY=DF*DTYI
    DGY=DF*DTYJ

    DFZ=DF*DTZI
    DGZ=DF*DTZJ

    DXAI=DFX     
    DXAJ=-DFX-DGX
    DXAK=DGX

    DYAI=DFY     
    DYAJ=-DFY-DGY
    DYAK=DGY

    DZAI=DFZ     
    DZAJ=-DFZ-DGZ
    DZAK=DGZ

     
    IF(PRNLEV.GT.7) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> ANGLE ATOMS ATOM I',I,'  ATOM J', J,'  ATOM K', K
      WRITE(OUTU,*) 'MMPT> ANGLE ENERGY',EA
      WRITE(OUTU,*) 'MMPT> ANGLE FORCE ATOM I',DXAI,DYAI,DZAI
      WRITE(OUTU,*) 'MMPT> ANGLE FORCE ATOM J',DXAJ,DYAJ,DZAJ
      WRITE(OUTU,*) 'MMPT> ANGLE FORCE ATOM K',DXAK,DYAK,DZAK
      WRITE(OUTU,*) ' '
      IF(PRNLEV.GT.8) THEN  
        WRITE(OUTU,*) 'MMPT> ANGLE PARAMETERS USED',CTB(IC),CTC(IC)
        WRITE(OUTU,*) ' '    
      ENDIF
    ENDIF

    RETURN
  END SUBROUTINE EANGLMMPT


!      _______________________________________________________

  SUBROUTINE NBNDMMPT(NB,I,MYJ,INBLO,JNB,MAXROW, &
         DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM,ENBMMPT,EELMMPT)

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image
  
    INTEGER M,I,J,J1,IC,INBLO(*),JNB(*),NB,MYJ,jpr,npr,itemp
    real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,              &
          DXI,DYI,DZI,S,R2,R1,G1,G2,G3,CGF,CGT,CGT2,   &
          EELMMPT,ENBMMPT, DF,ERM

    INTEGER IOFF(193),I1,IACI,MAXROW

    real(chm_real)  DXIRM,DXJRM,    &
           DYIRM,DYJRM,     &
           DZIRM,DZJRM
    real(chm_real) XYZ, CTOFNB,E14FAC,C2OFNB,C2ROF2,CHROF2,SIG2,SIG6,SIG12
    real(chm_real) DXIT,DYIT,DZIT


!     INITIALIZE THE CODE LOOK UP OFFSETS
    J=0
    DO M=1,NATC
      IOFF(M)=J
      J=J+M
    ENDDO
  
!   TODO:  PASSING OF VARIABLES CTOFNB, EPS,CCLEC AND E14FAC DOES
!   NOT WORK. VALUES ARE SET HERE.

    CTOFNB=12.D0
!   CGF EQUALS CCELEC/EPS 
    CGF= 332.0716D0
!   HERE E14FAC IS ALWAYS SET TO ONE
    E14FAC=1.D0

    C2OFNB=CTOFNB*CTOFNB

!   SHIFTED DIELECTRIC COEFFICIENTS
    C2ROF2=MINTWO/C2OFNB
    CHROF2=-HALF/C2OFNB

    EELMMPT=0.D0
    ENBMMPT=0.D0

!   CALCULATE FORCE OF INTERACTION I - J 
    CRXI=X(I)
    CRYI=Y(I)
    CRZI=Z(I)

    I1=ITC(IAC(I))
    IACI=IOFF(I1)

    CGT=CGF*CG(I)
    DF=0.0D0    
    J=MYJ 
    CGT2=CGT 

    J1=ITC(IAC(J))
    IF (I1.LT.J1) THEN
       IC=IOFF(J1)+I1
    ELSE
       IC=IACI+J1
    ENDIF

    DXI=CRXI-X(MYJ)
    DYI=CRYI-Y(MYJ)
    DZI=CRZI-Z(MYJ)
    
    S=DXI*DXI+DYI*DYI+DZI*DZI
    
    R2=1.0/S
    R1 = SQRT(R2)
    
!   VAN DER WAALS    
    SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
    SIG6=SIG2*SIG2*SIG2
    SIG12=SIG6*SIG6
    
    ENBMMPT=(CNBB(IC)*(SIG12-SIG6-SIG6))
    DF=CNBB(IC)*R2*12.D0*(SIG6-SIG12)  
    
!   ELECTROSTATIC    
    G1=CGT2*CG(J)*R1  
    G2=G1*S*C2ROF2
    G3=G2*S*CHROF2
    
    EELMMPT=(G1+G2+G3)
    DF=DF+R2*(G2-G1+THREE*G3) 
          
    DXIT=DXI*DF
    DYIT=DYI*DF
    DZIT=DZI*DF
    
!   ADD -DXIT TO J AND ADD DXIT TO I          
    DXIRM=DXIT
    DYIRM=DYIT
    DZIRM=DZIT
    
    DXJRM=-DXIT
    DYJRM=-DYIT
    DZJRM=-DZIT
    
    IF(PRNLEV.GT.7) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> NONBONDED ATOMS ATOM I',I,' ATOM J',J
      WRITE(OUTU,*) 'MMPT> NONBONDED ENERGY ELSTAT', EELMMPT    &
          ,' VDW', ENBMMPT                                     
      WRITE(OUTU,*) 'MMPT> NONBONDED FORCES ATOM I',DXIRM,DYIRM,DZIRM
      WRITE(OUTU,*) 'MMPT> NONBONDED FORCES ATOM J',DXJRM,DYJRM,DZJRM
      IF(PRNLEV.GT.8) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MMPT> VDW PARAMETER USED:',  &
           RSCLF(I),RSCLF(J),CNBA(IC),CNBB(IC), IC
      ENDIF
    ENDIF

  END SUBROUTINE NBNDMMPT
  
!      _______________________________________________________

  SUBROUTINE EVDW14MMPT(NB,I,MYJ,I14,INBLO,JNB,MAXROW, &
         DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM,EVDWMMPT,EELMMPT)

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image
  
    INTEGER M,I,J,J1,IC,INBLO(*),JNB(*),NB,MYJ,jpr,npr,itemp
    real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,              &
          DXI,DYI,DZI,S,R2,R1,G1,G2,G3,CGF,CGT,CGT2,   &
          EVDWMMPT,EELMMPT, DF,ERM

    INTEGER IOFF(193),I1,IACI,MAXROW,I14

    real(chm_real)  DXIRM,DXJRM,  &
           DYIRM,DYJRM,   &
           DZIRM,DZJRM

    real(chm_real) XYZ, CTOFNB,E14FAC,C2OFNB,C2ROF2,CHROF2,SIG2,SIG6,SIG12
    real(chm_real) DXIT,DYIT,DZIT


!     INITIALIZE THE CODE LOOK UP OFFSETS
    J=0
    DO M=1,NATC
      IOFF(M)=J
      J=J+M
    ENDDO
  
!  TODO:  PASSING OF VARIABLES CTOFNB, EPS,CCLEC AND E14FAC DOES
!   NOT WORK. VALUES ARE SET HERE.
    CTOFNB=12.D0
!   CGF EQUALS CCELEC/EPS 
    CGF= 332.0716D0
!   HERE E14FAC IS ALWAYS SET TO ONE
    E14FAC=1.D0
    C2OFNB=CTOFNB*CTOFNB

!   SHIFTED DIELECTRIC COEFFICIENTS
    C2ROF2=MINTWO/C2OFNB
    CHROF2=-HALF/C2OFNB

    EVDWMMPT=0.D0
    EELMMPT=0.D0

!   CALCULATE FORCE OF INTERACTION I - J 
    CRXI=X(I)
    CRYI=Y(I)
    CRZI=Z(I)

    I1=ITC(IAC(I))
    IACI=IOFF(I1)

    CGT=CGF*CG(I)

    DF=0.0D0
    
    J=MYJ
 
    CGT2=CGT 

!   RETRIEVE VDW PARAMETER FOR ATOM PAIR 
    IF (I14 .EQ. -14) THEN
      J1=ITC(IAC(J))     
      IF (I1.LT.J1) THEN
          IC=IOFF(J1)+I1+MAXROW
      ELSE 
          IC=IACI+J1+MAXROW
      ENDIF  
    ELSE        
      J1=ITC(IAC(J))
      IF (I1.LT.J1) THEN
          IC=IOFF(J1)+I1
      ELSE
          IC=IACI+J1
      ENDIF
    ENDIF
!write(*,*) "I-J-J1-IC-MAXROW",I,J,J1,IC,MAXROW
     
    DXI=CRXI-X(MYJ)
    DYI=CRYI-Y(MYJ)
    DZI=CRZI-Z(MYJ)
    
    S=DXI*DXI+DYI*DYI+DZI*DZI
    
    R2=1.0/S
    R1 = SQRT(R2)
    
!   VAN DER WAALS    
    SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
    SIG6=SIG2*SIG2*SIG2
    SIG12=SIG6*SIG6

    EVDWMMPT=(CNBB(IC)*(SIG12-SIG6-SIG6))
    DF=CNBB(IC)*R2*12.D0*(SIG6-SIG12)  
          
!   ELECTROSTATIC    
    G1=CGT2*CG(J)*R1  
    G2=G1*S*C2ROF2
    G3=G2*S*CHROF2
    
    EELMMPT=(G1+G2+G3)
    DF=DF+R2*(G2-G1+THREE*G3) 
    
    DXIT=DXI*DF
    DYIT=DYI*DF
    DZIT=DZI*DF
    
!   ADD -DXIT TO J AND ADD DXIT TO I
          
    DXIRM=DXIT
    DYIRM=DYIT
    DZIRM=DZIT
    
    DXJRM=-DXIT
    DYJRM=-DYIT
    DZJRM=-DZIT
    
    IF(PRNLEV.GT.7) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> SPECIAL 1-4 ATOMS ATOM I',I,' ATOM J',J
      WRITE(OUTU,*) 'MMPT> NONBONDED ENERGY 1-4 VDW', EVDWMMPT,EELMMPT
      WRITE(OUTU,*) 'MMPT> NONBONDED FORCES ATOM I',DXIRM,DYIRM,DZIRM
      WRITE(OUTU,*) 'MMPT> NONBONDED FORCES ATOM J',DXJRM,DYJRM,DZJRM
      IF(PRNLEV.GT.8) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MMPT> VDW PARAMETER USED:',  &
           RSCLF(I),RSCLF(J),CNBA(IC),CNBB(IC), IC
      ENDIF
    ENDIF

  END SUBROUTINE EVDW14MMPT
!      _______________________________________________________


  SUBROUTINE EPHIMMPT(IPHI,I,J,K,L,NPHI,ICP,CPD,CPCOS,CPSIN,CPC,&
         erm,DXIRM,DXJRM,DXKRM,DXLRM,                               &
             DYIRM,DYJRM,DYKRM,DYLRM,                               &
             DZIRM,DZJRM,DZKRM,DZLRM)

!     VARIABLES
!     CPD Periodicity of the dihedral energy
!     CPB Phase shift (delta) for the dihedral energy
!     CPCOS COS(CPB)
!     CPSIN SIN(CPB)
!     CPC Force constant for the dihedral energy


    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use consta
    use block_fcm
    use number
    use cnst_fcm
    use image      

    INTEGER ICP(*),CPD(*)
    real(chm_real) CPC(*),CPCOS(*),CPSIN(*)
    INTEGER I,J,K,L,IPER,NPER,IPHI,nphi,ic

    real(chm_real) e,ap,arg,erm

    real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ,                        &
          AX,AY,AZ,BX,BY,BZ,                                  &
          RA2,RB2,RG2,RG,                                     &
          RGR,RA2R,RB2R,RABR,                                 &
          CP,SP,                                              &
          DF,DDF,E1,DF1,DDF1,                                 &
          FG,HG,FGA,HGB,GAA,GBB,                              &
          DTFX,DTFY,DTFZ,DTGX,DTGY,DTGZ,DTHX,DTHY,DTHZ,       &
          DFX,DFY,DFZ,DGX,DGY,DGZ,DHX,DHY,DHZ,                &
          CA,SA
    LOGICAL LREP 
    real(chm_real)  DXIRM,DXJRM,DXKRM,DXLRM,      &
           DYIRM,DYJRM,DYKRM,DYLRM,       &
           DZIRM,DZJRM,DZKRM,DZLRM

    ERM=0.0D0

    IC=ICP(IPHI)

    FX=X(I)-X(J)
    FY=Y(I)-Y(J)
    FZ=Z(I)-Z(J)
    GX=X(J)-X(K)
    GY=Y(J)-Y(K)
    GZ=Z(J)-Z(K)
    HX=X(L)-X(K)
    HY=Y(L)-Y(K)
    HZ=Z(L)-Z(K)
    
    AX=FY*GZ-FZ*GY
    AY=FZ*GX-FX*GZ
    AZ=FX*GY-FY*GX
    BX=HY*GZ-HZ*GY
    BY=HZ*GX-HX*GZ
    BZ=HX*GY-HY*GX
  
    RA2=AX*AX+AY*AY+AZ*AZ
    RB2=BX*BX+BY*BY+BZ*BZ
    RG2=GX*GX+GY*GY+GZ*GZ
    RG=SQRT(RG2)   

    RGR=ONE/RG
    RA2R=ONE/RA2
    RB2R=ONE/RB2
    RABR=SQRT(RA2R*RB2R)  

    CP=(AX*BX+AY*BY+AZ*BZ)*RABR
    SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)

!   SETUP PROPER DIHEDRALS
!   CPD  Periodicity of the dihedral energy

    IF (CPD(IC).NE.0) THEN
      E=ZERO
      DF=ZERO
      DDF=ZERO

      IPER=CPD(IC)

      IF (IPER.GE.0) THEN
        LREP=.FALSE.
      ELSE
        LREP=.TRUE.
        IPER=-IPER
      ENDIF

      E1=ONE
      DF1=ZERO

      DO NPER=1,IPER
          DDF1=E1*CP-DF1*SP
          DF1=E1*SP+DF1*CP
          E1=DDF1
      ENDDO

      E1=E1*CPCOS(IC)+DF1*CPSIN(IC)
      DF1=DF1*CPCOS(IC)-DDF1*CPSIN(IC)
      DF1=-IPER*DF1
      DDF1=-IPER*IPER*E1
      E1=ONE+E1

      IF (IPER.EQ.0) THEN
          E1=ONE
          DF1=ZERO
          DDF1=ZERO
      ENDIF


!   SETUP FOR IMPROPER DIHEDRALS
    ELSE
    
!   Calculation  of cos(phi-phi0),sin(phi-phi0) and (Phi-Phi0).
      CA=CP*CPCOS(IC)+SP*CPSIN(IC)
      SA=SP*CPCOS(IC)-CP*CPSIN(IC)
      IF (CA.GT.PTONE ) THEN
          AP=ASIN(SA)
      ELSE
          AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
      ENDIF

      DDF=TWO*CPC(IC)
      DF=DDF*AP
      E=HALF*DF*AP
    ENDIF

    ARG=CPC(IC)
    E=E+ARG*E1
    DF=DF+ARG*DF1
    DDF=DDF+ARG*DDF1

    FG=FX*GX+FY*GY+FZ*GZ
    HG=HX*GX+HY*GY+HZ*GZ
    FGA=FG*RA2R*RGR
    HGB=HG*RB2R*RGR
    GAA=-RA2R*RG
    GBB=RB2R*RG

    DTFX=GAA*AX
    DTFY=GAA*AY
    DTFZ=GAA*AZ
    DTGX=FGA*AX-HGB*BX
    DTGY=FGA*AY-HGB*BY
    DTGZ=FGA*AZ-HGB*BZ
    DTHX=GBB*BX
    DTHY=GBB*BY
    DTHZ=GBB*BZ

    DFX=DF*DTFX
    DFY=DF*DTFY
    DFZ=DF*DTFZ
    DGX=DF*DTGX
    DGY=DF*DTGY
    DGZ=DF*DTGZ
    DHX=DF*DTHX
    DHY=DF*DTHY
    DHZ=DF*DTHZ

    ERM=E

    DXIRM=DFX     
    DXJRM=-DFX+DGX
    DXKRM=-DHX-DGX
    DXLRM=DHX

    DYIRM=DFY     
    DYJRM=-DFY+DGY
    DYKRM=-DHY-DGY
    DYLRM=DHY

    DZIRM=DFZ     
    DZJRM=-DFZ+DGZ
    DZKRM=-DHZ-DGZ
    DZLRM=DHZ
 
    IF(PRNLEV.GT.7) THEN
      IF (CPD(IC).NE.0) THEN          
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) 'MMPT> DIHEDRAL ATOMS ATOM I',I,'  ATOM J', J   &
              ,'  ATOM K', K,'  ATOM L', L                              
        WRITE(OUTU,*) 'MMPT> DIHEDRAL ENERGY',E                         
        WRITE(OUTU,*) 'MMPT> DIHEDRAL FORCE ATOM I',DXIRM,DYIRM,DZIRM   
        WRITE(OUTU,*) 'MMPT> DIHEDRAL FORCE ATOM J',DXJRM,DYJRM,DZJRM   
        WRITE(OUTU,*) 'MMPT> DIHEDRAL FORCE ATOM K',DXKRM,DYKRM,DZKRM   
        WRITE(OUTU,*) 'MMPT> DIHEDRAL FORCE ATOM L',DXLRM,DYLRM,DZLRM   
        WRITE(OUTU,*) ' '                                                                                                                       
          IF(PRNLEV.GT.8) THEN                                            
            WRITE(OUTU,*) 'MMPT> DIHEDRAL PARAMETERS USED', &
                 CPC(IC),CPCOS(IC),CPSIN(IC),CPD(IC)                    
            WRITE(OUTU,*) ' '                                           
          ENDIF                                                          
      ELSE                                                                                                                                                                                                             
        WRITE(OUTU,*) ' '                                               
        WRITE(OUTU,*) 'MMPT> IMPROPER ATOMS ATOM I',I,'  ATOM J', J    &
             ,'  ATOM K', K,'  ATOM L', L
        WRITE(OUTU,*) 'MMPT> IMPROPER ENERGY',E
        WRITE(OUTU,*) 'MMPT> IMPROPER FORCE ATOM I',DXIRM,DYIRM,DZIRM
        WRITE(OUTU,*) 'MMPT> IMPROPER FORCE ATOM J',DXJRM,DYJRM,DZJRM
        WRITE(OUTU,*) 'MMPT> IMPROPER FORCE ATOM K',DXKRM,DYKRM,DZKRM
        WRITE(OUTU,*) 'MMPT> IMPROPER FORCE ATOM L',DXLRM,DYLRM,DZLRM
        WRITE(OUTU,*) ' '          
        IF(PRNLEV.GT.8) THEN  
             WRITE(OUTU,*) 'MMPT> IMPROPER PARAMETERS USED',  & 
                CPC(IC),CPCOS(IC),CPSIN(IC),CPD(IC)
             WRITE(OUTU,*) ' '            
        ENDIF
      ENDIF
    ENDIF

  END SUBROUTINE EPHIMMPT


!      _________________________________________________________________



  SUBROUTINE SWITCHMMPT(I,J,K,SWF,DSWFRNN,DSWFRNH,                &
                           DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
  
!     Force and energy are controlled by a switching function.
!     They are gradually turned off or on according to the relative 
!     distance between the involved donor resp acceptor atom and 
!     the transfer hydrogen atom.

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    use cnst_fcm
    use image

    INTEGER I,J,K
    real(chm_real) RNN,SNN,DXNN,DYNN,DZNN,        &
          RNH,SNH,DXNH,DYNH,DZNH,                 &
          DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ,    &
          SIGMA,KAPPA,DKAPPA,                     &
          THF,SWF,DSWFRNN,DSWFRNH 
 
!   cartesian distances donor - hydrogen

    DXNH=X(I)-X(J)
    DYNH=Y(I)-Y(J)
    DZNH=Z(I)-Z(J)

!   cartesian distances donor - acceptor 

    DXNN=X(I)-X(K)
    DYNN=Y(I)-Y(K)
    DZNN=Z(I)-Z(K)

    SNN=DXNN*DXNN+DYNN*DYNN+DZNN*DZNN
    RNN=SQRT(SNN)
    
    RNH=(DXNH*DXNN+DYNH*DYNN+DZNH*DZNN)/RNN

!   derivatives  
    DRNHX=DXNH/RNN
    DRNHY=DYNH/RNN
    DRNHZ=DZNH/RNN

    DRNNX=DXNN/RNN
    DRNNY=DYNN/RNN
    DRNNZ=DZNN/RNN
  

!   switch function 
!    INSUFFIECIENT APPROXIMATION
!    KAPPA=-3.625D0+2.7D0*RNN
!    DKAPPA=2.7D0

!    IMPROVED KAPPA
    KAPPA=0.4999988862D0*RNN*RNN+0.6014307884D-5*RNN-0.8024645983D-5
    DKAPPA=0.9999977724D0*RNN+0.6014307884D-5
    
    SIGMA=2.D0

    THF=TANH(SIGMA*(RNH*RNN-KAPPA))

    SWF=(THF+1.D0)/2.D0

    DSWFRNH=(1.D0-THF*THF)/2.D0*SIGMA*RNN
!    DSWFRNN=(1.D0-THF*THF)/2.D0*SIGMA*(RNH-DKAPPA)
!    dswfrnn=dswfrnn-DSWFRNH*rnh/rnn

    DSWFRNN=(1.D0-THF*THF)/2.D0*SIGMA*(-1.D0*DKAPPA)
    IF(PRNLEV.GT.7) THEN
      WRITE(OUTU,*) 'MMPT> SWITCH FUNCTION RETURNS',SWF
    ENDIF
  END SUBROUTINE SWITCHMMPT


  SUBROUTINE SWITCH2D(I,J,K,THF,DSWFRNN,DSWFRNH,                &
                           DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
  
!     Force and energy are controlled by a switching function.
!     They are gradually turned off or on according to the relative 
!     distance between the involved donor resp acceptor atom and 
!     the transfer hydrogen atom.

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    
    use cnst_fcm
    use image

    INTEGER I,J,K
    real(chm_real) RNN,SNN,RNH,SNH,       &
          !RNN2,SNN2,RNH2,SNH2,       
          DXNN,DYNN,DZNN,DXNH,DYNH,DZNH,                 &
          !DXNN2,DYNN2,DZNN2,DXNH2,DYNH2,DZNH2,                 
          DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ,    &
          SIGMA,KAPPA,DKAPPA,                     &
          THF,DSWFRNN,DSWFRNH 
 
!   cartesian distances donor - hydrogen

    DXNH=X(I)-X(J)
    DYNH=Y(I)-Y(J)
    DZNH=Z(I)-Z(J)

!   cartesian distances donor - acceptor 

    DXNN=X(I)-X(K)
    DYNN=Y(I)-Y(K)
    DZNN=Z(I)-Z(K)

    SNN=DXNN*DXNN+DYNN*DYNN+DZNN*DZNN
    RNN=SQRT(SNN)
    
    RNH=(DXNH*DXNN+DYNH*DYNN+DZNH*DZNN)/RNN

!   derivatives  
    DRNHX=DXNH/RNN
    DRNHY=DYNH/RNN
    DRNHZ=DZNH/RNN

    DRNNX=DXNN/RNN
    DRNNY=DYNN/RNN
    DRNNZ=DZNN/RNN
  

!   switch function 
!    INSUFFIECIENT APPROXIMATION
!    KAPPA=-3.625D0+2.7D0*RNN
!    DKAPPA=2.7D0

!    IMPROVED KAPPA
    !KAPPA=0.4999988862D0*RNN*RNN+0.6014307884D-5*RNN-0.8024645983D-5
    !DKAPPA=0.9999977724D0*RNN+0.6014307884D-5
    KAPPA=0.5D0*RNN*RNN
    DKAPPA=RNN
    
    SIGMA=2.639D0

    THF=TANH(SIGMA*(RNH*RNN-KAPPA))

    !SWF=THF

    DSWFRNH=(1.D0-THF*THF)*SIGMA*RNN
!    DSWFRNN=(1.D0-THF*THF)/2.D0*SIGMA*(RNH-DKAPPA)
!    dswfrnn=dswfrnn-DSWFRNH*rnh/rnn

    DSWFRNN=(1.D0-THF*THF)*SIGMA*(-1.D0*DKAPPA)
    
  END SUBROUTINE SWITCH2D
! ... _______________________________________________________

  SUBROUTINE EMMPT(EU,X,Y,Z,DX,DY,DZ,NATOMX)
! main subroutine to add MMPT energy and derivatives    
! 
!

    use dimens_fcm
    use number
    use stream
    use psf
    use inbnd
    use param
    use consta
    use contrl
    use code
    use bases_fcm
    use chm_types
    use pbound

    real(chm_real) EU
    INTEGER NATOMX
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
    real(chm_real) DXA(NATOMX),DYA(NATOMX),DZA(NATOMX)
    real(chm_real) DXB(NATOMX),DYB(NATOMX),DZB(NATOMX)
    !type(nonbondDataStructure) BNBND
  
!     SWITCH STUFF
    real(chm_real) SWF,DSWFRNN,DSWFRNH,DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ
    real(chm_real) DSWFRNN1,DSWFRNH1,DRNHX1,DRNHY1,DRNHZ1,DRNNX1,DRNNY1,DRNNZ1
    real(chm_real) DSWFRNN2,DSWFRNH2,DRNHX2,DRNHY2,DRNHZ2,DRNNX2,DRNNY2,DRNNZ2
    real(chm_real) THF1,THF2,TSWF,ETHF,eu1,eu2
    INTEGER DPTFLAG,DPTMOTIF1(3),DPTMOTIF2(3)


!     -----------------------------------------------------------------

  
    real(chm_real) EB,EA, EP,EELMMPT,ENBMMPT,EVDW1MMPT,EVDW2MMPT
    real(chm_real) ERM,DXBRM,DYBRM,DZBRM
    real(chm_real) SUM

    INTEGER I,J,HB,ATMPA,ATMPB,II

    real(chm_real) EIMPROP

    real(chm_real) DXIRM,DYIRM,DZIRM,DXLRM,&
           DXJRM,DYJRM,DZJRM,DYLRM, &
           DXKRM,DYKRM,DZKRM,DZLRM
                                     
    real(chm_real) DXIRM1,DYIRM1,DZIRM1,    &
           DXJRM1,DYJRM1,DZJRM1
                                     
    real(chm_real) DXIRM2,DYIRM2,DZIRM2,    &
           DXJRM2,DYJRM2,DZJRM2

                                     
    real(chm_real) ED,DXDI,DXDJ,DXDK,DXDL,  &
              DYDI,DYDJ,DYDK,DYDL,  &
              DZDI,DZDJ,DZDK,DZDL
                                     
    real(chm_real) EI,DXII,DXIJ,DXIK,DXIL,  &
              DYII,DYIJ,DYIK,DYIL,  &
              DZII,DZIJ,DZIK,DZIL
 
    real(chm_real) DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK
    real(chm_real) DX1,DX2,DY1,DY2,DZ1,DZ2,PRMANGLE,DDSWF
    real(chm_real) FCST1,FCST2,EQUI1,EQUI2


    real(chm_real) XCOM,YCOM,ZCOM,DIPX,DIPY,DIPZ,DIPM 
!     ------------------------------------------------------------
      

    IF(QMMPT)THEN
      EU = 0.d0 

      DXIRM=0.D0
      DYIRM=0.D0
      DZIRM=0.d0

      DXJRM=0.D0
      DYJRM=0.D0
      DZJRM=0.d0

      DXIRM1=0.D0
      DYIRM1=0.D0
      DZIRM1=0.d0

      DXJRM1=0.D0
      DYJRM1=0.D0
      DZJRM1=0.d0

      DXIRM2=0.D0
      DYIRM2=0.D0
      DZIRM2=0.d0

      DXJRM2=0.D0
      DYJRM2=0.D0
      DZJRM2=0.d0

      DXLRM=0.D0
      DYLRM=0.D0
      DZLRM=0.d0

      DXKRM=0.D0
      DYKRM=0.D0
      DZKRM=0.d0

!      REMARKS ON CONSTRAINTS:
! ... IF ALL HYDROGEN BRIDGE ATOMS ARE CONSTRAINT NO MMPT ENERGY AND FORCES
!      ARE CALCULATED

      
!DPT counts both DPT motifs to calculate 2D switch
      DPTFLAG=0
      DO I=1,HBRDNUM                                                
        IF(POTTYPE(I) .EQ. 'DPT') THEN
          IF (DPTFLAG.EQ.0) THEN
            CALL SWITCH2D(HBRDATOM(I,1),HBRDATOM(I,2),HBRDATOM(I,3), &
            THF1,DSWFRNN1,DSWFRNH1,DRNHX1,DRNHY1,DRNHZ1,DRNNX1,DRNNY1,DRNNZ1)
            DPTFLAG=DPTFLAG+1
            DPTMOTIF1(1)=HBRDATOM(I,1)
            DPTMOTIF1(2)=HBRDATOM(I,2)
            DPTMOTIF1(3)=HBRDATOM(I,3)
          ELSE IF (DPTFLAG.EQ.1) THEN
            CALL SWITCH2D(HBRDATOM(I,1),HBRDATOM(I,2),HBRDATOM(I,3), &
            THF2,DSWFRNN2,DSWFRNH2,DRNHX2,DRNHY2,DRNHZ2,DRNNX2,DRNNY2,DRNNZ2)
            DPTFLAG=DPTFLAG+1
            DPTMOTIF2(1)=HBRDATOM(I,1)
            DPTMOTIF2(2)=HBRDATOM(I,2)
            DPTMOTIF2(3)=HBRDATOM(I,3)
          ENDIF
        ENDIF
      ENDDO 

      IF (DPTFLAG.EQ.2) THEN
        TSWF=(THF1*THF2+1.D0)/2.D0
      ENDIF
!WRITE(outu,*) 'TEST> ',thf1,thf2,tswf
      DO I=1,HBRDNUM

        IF (IMOVE(HBRDATOM(I,1)).GT.0 .AND. IMOVE(HBRDATOM(I,2)).GT.0  &
        .AND. IMOVE(HBRDATOM(I,3)).GT.0) EXIT     
        
        
        IF(POTTYPE(I) .EQ. 'SDM') CALL EPTNHN(HBRDATOM(I,1), &
          HBRDATOM(I,2),HBRDATOM(I,3),EU,X,Y,Z,DX,DY,DZ,NATOMX)
        IF(POTTYPE(I) .EQ. 'SSM') CALL EPTOHO(HBRDATOM(I,1), &
          HBRDATOM(I,2),HBRDATOM(I,3),EU,X,Y,Z,DX,DY,DZ,NATOMX)
        IF(POTTYPE(I) .EQ. 'ASM') CALL EPTNHO(HBRDATOM(I,1), &
          HBRDATOM(I,2),HBRDATOM(I,3),EU,X,Y,Z,DX,DY,DZ,NATOMX)
        IF(POTTYPE(I) .EQ. 'LPE') CALL EPTOHOLE(HBRDATOM(I,1), &
          HBRDATOM(I,2),HBRDATOM(I,3),EU,X,Y,Z,DX,DY,DZ,NATOMX)
        IF(POTTYPE(I) .EQ. 'NLM') CALL EPTNL(HBRDATOM(I,1), &
          HBRDATOM(I,2),HBRDATOM(I,3),EU,X,Y,Z,DX,DY,DZ,NATOMX)
        IF(POTTYPE(I) .EQ. 'ANL') CALL EPTANL(HBRDATOM(I,1), &
          HBRDATOM(I,2),HBRDATOM(I,3),EU,X,Y,Z,DX,DY,DZ,NATOMX)
        IF(POTTYPE(I) .EQ. 'DPT' ) THEN                     
          EU1=0.D0
          EU2=0.D0
          DO J=1,NATOMX
            DXA(J)=0.D0
            DYA(J)=0.D0
            DZA(J)=0.D0
            DXB(J)=0.D0
            DYB(J)=0.D0
            DZB(J)=0.D0
          ENDDO                        
          IF (DPTFLAG.NE.2) call WrnDie (-3,'<MISCOM>', &
          'MMPT EMMPT> NEED 2 MOTIFS DEFINED AS DPT-MOTIFS')
          
          CALL EPTNHN(HBRDATOM(I,1),HBRDATOM(I,2),HBRDATOM(I,3),EU1, &
          X,Y,Z,DXA,DYA,DZA,NATOMX)
          CALL EPTOHO(HBRDATOM(I,1),HBRDATOM(I,2),HBRDATOM(I,3),EU2, &
          X,Y,Z,DXB,DYB,DZB,NATOMX)

!WRITE(outu,*) 'TEST> ',eu1,eu2
          EU=EU+EU1*TSWF+EU2*(1.D0-TSWF)
          DO J=1,NATOMX
            DX(J)=DX(J)+DXA(J)*TSWF+DXB(J)*(1.D0-TSWF)
            DY(J)=DY(J)+DYA(J)*TSWF+DYB(J)*(1.D0-TSWF)
            DZ(J)=DZ(J)+DZA(J)*TSWF+DZB(J)*(1.D0-TSWF)
!WRITE(outu,*) 'TEST> ',DX(J),DXA(J),DXB(J),TSWF
          ENDDO
          
          ETHF=0.5D0*(EU1-EU2)*THF2
          
          DX1=DSWFRNH1*DRNNX1*ETHF
          DX2=(DSWFRNN1*DRNNX1+DSWFRNH1*DRNHX1)*ETHF
          DY1=DSWFRNH1*DRNNY1*ETHF
          DY2=(DSWFRNN1*DRNNY1+DSWFRNH1*DRNHY1)*ETHF
          DZ1=DSWFRNH1*DRNNZ1*ETHF
          DZ2=(DSWFRNN1*DRNNZ1+DSWFRNH1*DRNHZ1)*ETHF
          
          DX(HBRDATOM(1,1))=DX(HBRDATOM(1,1))+DX1+DX2
          DX(HBRDATOM(1,2))=DX(HBRDATOM(1,2))-DX1
          DX(HBRDATOM(1,3))=DX(HBRDATOM(1,3))-DX2
          DY(HBRDATOM(1,1))=DY(HBRDATOM(1,1))+DY1+DY2
          DY(HBRDATOM(1,2))=DY(HBRDATOM(1,2))-DY1
          DY(HBRDATOM(1,3))=DY(HBRDATOM(1,3))-DY2
          DZ(HBRDATOM(1,1))=DZ(HBRDATOM(1,1))+DZ1+DZ2
          DZ(HBRDATOM(1,2))=DZ(HBRDATOM(1,2))-DZ1
          DZ(HBRDATOM(1,3))=DZ(HBRDATOM(1,3))-DZ2
          
          ETHF=0.5D0*(EU1-EU2)*THF1
          
          DX1=DSWFRNH2*DRNNX2*ETHF
          DX2=(DSWFRNN2*DRNNX2+DSWFRNH2*DRNHX2)*ETHF
          DY1=DSWFRNH2*DRNNY2*ETHF
          DY2=(DSWFRNN2*DRNNY2+DSWFRNH2*DRNHY2)*ETHF
          DZ1=DSWFRNH2*DRNNZ2*ETHF
          DZ2=(DSWFRNN2*DRNNZ2+DSWFRNH2*DRNHZ2)*ETHF
          
          DX(HBRDATOM(2,1))=DX(HBRDATOM(2,1))+DX1+DX2
          DX(HBRDATOM(2,2))=DX(HBRDATOM(2,2))-DX1
          DX(HBRDATOM(2,3))=DX(HBRDATOM(2,3))-DX2
          DY(HBRDATOM(2,1))=DY(HBRDATOM(2,1))+DY1+DY2
          DY(HBRDATOM(2,2))=DY(HBRDATOM(2,2))-DY1
          DY(HBRDATOM(2,3))=DY(HBRDATOM(2,3))-DY2
          DZ(HBRDATOM(2,1))=DZ(HBRDATOM(2,1))+DZ1+DZ2
          DZ(HBRDATOM(2,2))=DZ(HBRDATOM(2,2))-DZ1
          DZ(HBRDATOM(2,3))=DZ(HBRDATOM(2,3))-DZ2
          
        ENDIF
          
        PRMANGLE=-0.00001D0
        IF(POTTYPE(I) .EQ. 'SDM' .or. POTTYPE(I) .EQ. 'DPT' &
        ) PRMANGLE=PRMNHN(10)
        IF(POTTYPE(I) .EQ. 'SSM') PRMANGLE=PRMOHO(10)
        IF(POTTYPE(I) .EQ. 'ASM') PRMANGLE=PRMNHO(27)
        
        IF (PRMANGLE.gt.0.D0 .and. 1.eq.0) THEN
          CALL SWITCHMMPT(HBRDATOM(I,1),HBRDATOM(I,2),HBRDATOM(I,3), &
          SWF,DSWFRNN,DSWFRNH,DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)

          CALL EPSEUDOMMPT(HBRDATOM(I,2),HBRDATOM(I,1), &
          HBRDATOM(I,3),EA,PRMANGLE*(180.d0/PI)**2,0.D0,0,DXAI,DYAI,DZAI,   &
          DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)
        
          EU=EU-EA*SWF
  !WRITE(outu,*) 'TEST EA1> ',ea

          DX1=DSWFRNH*DRNNX*EA
          DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA
          DY1=DSWFRNH*DRNNY*EA
          DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA
          DZ1=DSWFRNH*DRNNZ*EA
          DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

          DX(HBRDATOM(I,2))=DX(HBRDATOM(I,2))-(DXAI*SWF)
          DX(HBRDATOM(I,1))=DX(HBRDATOM(I,1))-(DXAJ*SWF)
          DX(HBRDATOM(I,3))=DX(HBRDATOM(I,3))-(DXAK*SWF)
          DX(HBRDATOM(I,1))=DX(HBRDATOM(I,1))-DX1-DX2
          DX(HBRDATOM(I,2))=DX(HBRDATOM(I,2))+DX1
          DX(HBRDATOM(I,3))=DX(HBRDATOM(I,3))+DX2

          DY(HBRDATOM(I,2))=DY(HBRDATOM(I,2))-(DYAI*SWF)
          DY(HBRDATOM(I,1))=DY(HBRDATOM(I,1))-(DYAJ*SWF)
          DY(HBRDATOM(I,3))=DY(HBRDATOM(I,3))-(DYAK*SWF)
          DY(HBRDATOM(I,1))=DY(HBRDATOM(I,1))-DY1-DY2
          DY(HBRDATOM(I,2))=DY(HBRDATOM(I,2))+DY1
          DY(HBRDATOM(I,3))=DY(HBRDATOM(I,3))+DY2

          DZ(HBRDATOM(I,2))=DZ(HBRDATOM(I,2))-(DZAI*SWF)
          DZ(HBRDATOM(I,1))=DZ(HBRDATOM(I,1))-(DZAJ*SWF)
          DZ(HBRDATOM(I,3))=DZ(HBRDATOM(I,3))-(DZAK*SWF)
          DZ(HBRDATOM(I,1))=DZ(HBRDATOM(I,1))-DZ1-DZ2
          DZ(HBRDATOM(I,2))=DZ(HBRDATOM(I,2))+DZ1
          DZ(HBRDATOM(I,3))=DZ(HBRDATOM(I,3))+DZ2
          
          CALL EPSEUDOMMPT(HBRDATOM(I,2),HBRDATOM(I,3), &
          HBRDATOM(I,1),EA,PRMANGLE*(180.d0/PI)**2,0.D0,0,DXAI,DYAI,DZAI,   &
          DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)
        
          EU=EU+EA*SWF
  !WRITE(outu,*) 'TEST EA1> ',ea
          
          DX1=DSWFRNH*DRNNX*EA
          DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA
          DY1=DSWFRNH*DRNNY*EA
          DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA
          DZ1=DSWFRNH*DRNNZ*EA
          DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA

          DX(HBRDATOM(I,2))=DX(HBRDATOM(I,2))+(DXAI*SWF)
          DX(HBRDATOM(I,3))=DX(HBRDATOM(I,3))+(DXAJ*SWF)
          DX(HBRDATOM(I,1))=DX(HBRDATOM(I,1))+(DXAK*SWF)
          DX(HBRDATOM(I,1))=DX(HBRDATOM(I,1))+DX1+DX2
          DX(HBRDATOM(I,2))=DX(HBRDATOM(I,2))-DX1
          DX(HBRDATOM(I,3))=DX(HBRDATOM(I,3))-DX2

          DY(HBRDATOM(I,2))=DY(HBRDATOM(I,2))+(DYAI*SWF)
          DY(HBRDATOM(I,3))=DY(HBRDATOM(I,3))+(DYAJ*SWF)
          DY(HBRDATOM(I,1))=DY(HBRDATOM(I,1))+(DYAK*SWF)
          DY(HBRDATOM(I,1))=DY(HBRDATOM(I,1))+DY1+DY2
          DY(HBRDATOM(I,2))=DY(HBRDATOM(I,2))-DY1
          DY(HBRDATOM(I,3))=DY(HBRDATOM(I,3))-DY2

          DZ(HBRDATOM(I,2))=DZ(HBRDATOM(I,2))+(DZAI*SWF)
          DZ(HBRDATOM(I,3))=DZ(HBRDATOM(I,3))+(DZAJ*SWF)
          DZ(HBRDATOM(I,1))=DZ(HBRDATOM(I,1))+(DZAK*SWF)
          DZ(HBRDATOM(I,1))=DZ(HBRDATOM(I,1))+DZ1+DZ2
          DZ(HBRDATOM(I,2))=DZ(HBRDATOM(I,2))-DZ1
          DZ(HBRDATOM(I,3))=DZ(HBRDATOM(I,3))-DZ2
        ENDIF
        
        ERM=0.d0
        IF(POTTYPE(I) .EQ. 'NLM') THEN
          CALL EPSEUDOBOND(HBRDATOM(I,1),HBRDATOM(I,3),ERM, &
          2000.0D0,2.20D0,-1,DXBRM,DYBRM,DZBRM)
        ELSE
          CALL EPSEUDOBOND(HBRDATOM(I,1),HBRDATOM(I,3),ERM, &
          2000.0D0,2.0D0,-1,DXBRM,DYBRM,DZBRM)
        ENDIF
        
        EU = EU + ERM
        if (prnlev>=7 .AND. ERM.GT.0.D0) WRITE (OUTU,*) &
        'MMPT EMMPT> E_DA_PENALTY=',ERM
        
        DX(HBRDATOM(I,1))=DX(HBRDATOM(I,1))+DXBRM
        DY(HBRDATOM(I,1))=DY(HBRDATOM(I,1))+DYBRM
        DZ(HBRDATOM(I,1))=DZ(HBRDATOM(I,1))+DZBRM

        DX(HBRDATOM(I,3))=DX(HBRDATOM(I,3))-DXBRM
        DY(HBRDATOM(I,3))=DY(HBRDATOM(I,3))-DYBRM
        DZ(HBRDATOM(I,3))=DZ(HBRDATOM(I,3))-DZBRM
      ENDDO 


!      POLARIZATION

!      DO I=1,HBRDNUM
!      CALL PLZFCT(I,HBRDATOM(I,1),HBRDATOM(I,2),HBRDATOM(I,3))
!      ENDDO

!      SUM = 0.0D0
!      DO I=1,HBRDNUM
!      SUM = SUM+CG(HBRDATOM(I,1))+CG(HBRDATOM(I,2))+CG(HBRDATOM(I,3))
!      ENDDO

!      WRITE(99,*) SUM


!      CALCULATE THE DIPOLE MOMENT

!      XCOM=ZERO
!      YCOM=ZERO
!      ZCOM=ZERO
!      DIPX=ZERO
!      DIPY=ZERO
!      DIPZ=ZERO


!      DO I=1,NATOM
!         XCOM=XCOM+X(I)
!         YCOM=YCOM+Y(I)
!         ZCOM=ZCOM+Z(I)
!      ENDDO

!      XCOM=XCOM/NATOM
!      YCOM=YCOM/NATOM
!      ZCOM=ZCOM/NATOM
       
!      
!      DO I=1,NATOM
!         DIPX=DIPX+(X(I)-XCOM)*CG(I)
!         DIPY=DIPY+(Y(I)-YCOM)*CG(I)
!         DIPZ=DIPZ+(Z(I)-ZCOM)*CG(I)
!      ENDDO

!      DIPX=DIPX*DEBYEC
!      DIPY=DIPY*DEBYEC
!      DIPZ=DIPZ*DEBYEC

!      DIPM=SQRT(DIPX*DIPX+DIPY*DIPY+DIPZ*DIPZ)

!      WRITE(99,*) DIPX,DIPY,DIPZ,DIPM

!      BONDS
!      REMARKS ON CONSTRAINTS:
!      IF BOTH ATOMS ARE FIXED BY USING CONS FIX NOTHING IS TO BE
!      DONE BECAUSE FORCECONSTANT POINTER POINTS TO ZERO

!if (prnlev.ge.6) WRITE(outu,*) 'EMMPT> EPOT:',eu
EU1=EU
      DO HB=1,HBRDNUM
        CALL EBONDMMPT(BONDMMPT(HB,1),BONDMMPT(HB,2),BONDMMPT(HB,3), &
                       ERM,DXBRM,DYBRM,DZBRM)
 
        EU = EU - ERM

        DX(BONDMMPT(HB,2))=DX(BONDMMPT(HB,2))-DXBRM
        DY(BONDMMPT(HB,2))=DY(BONDMMPT(HB,2))-DYBRM
        DZ(BONDMMPT(HB,2))=DZ(BONDMMPT(HB,2))-DZBRM
        
        DX(BONDMMPT(HB,3))=DX(BONDMMPT(HB,3))+DXBRM
        DY(BONDMMPT(HB,3))=DY(BONDMMPT(HB,3))+DYBRM
        DZ(BONDMMPT(HB,3))=DZ(BONDMMPT(HB,3))+DZBRM
!write (outu,*) 'TEST> ',BDTRNUM
        IF (BDTRNUM.GT.0) THEN
          CALL SWITCHMMPT(HBRDATOM(HB,1),HBRDATOM(HB,2),HBRDATOM(HB,3), &
          SWF,DSWFRNN,DSWFRNH,DRNHX,DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
          
          DO I=1, NBOND
            DO J=1,BDTRNUM
              ATMPA=-1
              ATMPB=-1
              IF ((ATC(IAC(IB(I))).EQ.BONDA(J).AND. &
              ATC(IAC(JB(I))).EQ.BONDB(J)) &
              .OR. (ATC(IAC(JB(I))).EQ.BONDA(J).AND. &
              ATC(IAC(IB(I))).EQ.BONDB(J))) THEN
                IF(BONDLV(J).EQ.1) THEN
                  IF (HBRDATOM(HB,1).EQ.IB(I).OR.HBRDATOM(HB,1).EQ.JB(I)) THEN
                    ATMPA=IB(I)
                    ATMPB=JB(I)
                    FCST1=BONDFC1(J)
                    FCST2=BONDFC2(J)
                    EQUI1=BONDEQU1(J)
                    EQUI2=BONDEQU2(J)
                  ELSE IF (HBRDATOM(HB,3).EQ.IB(I).OR. &
                  HBRDATOM(HB,3).EQ.JB(I)) THEN
                    ATMPA=IB(I)
                    ATMPB=JB(I)
                    FCST1=BONDFC2(J)
                    FCST2=BONDFC1(J)
                    EQUI1=BONDEQU2(J)
                    EQUI2=BONDEQU1(J)
                  ENDIF
                ELSE IF (BONDLV(J).EQ.2) THEN
                  DO II=1,NBOND
                    IF (HBRDATOM(HB,1).EQ.IB(II).AND.(JB(II).EQ.IB(I).OR. &
                    JB(II).EQ.JB(I)) .OR. HBRDATOM(HB,1).EQ.JB(II) .AND. &
                    (IB(II).EQ.IB(I).OR. IB(II).EQ.JB(I))) THEN
                      ATMPA=IB(I)
                      ATMPB=JB(I)
                      FCST1=BONDFC1(J)
                      FCST2=BONDFC2(J)
                      EQUI1=BONDEQU1(J)
                      EQUI2=BONDEQU2(J)
                      GOTO 2285
                    ELSE IF (HBRDATOM(HB,3).EQ.IB(II).AND.(JB(II).EQ.IB(I).OR. &
                    JB(II).EQ.JB(I)) .OR. HBRDATOM(HB,3).EQ.JB(II) .AND. &
                    (IB(II).EQ.IB(I).OR. IB(II).EQ.JB(I))) THEN
                      ATMPA=IB(I)
                      ATMPB=JB(I)
                      FCST1=BONDFC2(J)
                      FCST2=BONDFC1(J)
                      EQUI1=BONDEQU2(J)
                      EQUI2=BONDEQU1(J)
                      GOTO 2285
                    ENDIF
                  ENDDO
                ENDIF


2285 CONTINUE
                IF (ATMPA.GT.0 .AND. ATMPB.GT.0) THEN
!write (outu,*) 'TEST2> ',BONDLV(J),I,ATMPA,ATMPB,SWF,FCST1
                  CALL EBONDMMPT(I,ATMPA,ATMPB,ERM,DXBRM,DYBRM,DZBRM)
           
                  EU = EU - ERM
          
                  DX(ATMPA)=DX(ATMPA)-DXBRM
                  DY(ATMPA)=DY(ATMPA)-DYBRM
                  DZ(ATMPA)=DZ(ATMPA)-DZBRM
                  
                  DX(ATMPB)=DX(ATMPB)+DXBRM
                  DY(ATMPB)=DY(ATMPB)+DYBRM
                  DZ(ATMPB)=DZ(ATMPB)+DZBRM
                
                  CALL EPSEUDOBOND(ATMPA,ATMPB,ERM, &
                  FCST1, EQUI1 ,0,DXBRM,DYBRM,DZBRM)
                  
                  EU = EU + ERM*(1.D0-SWF)
                  DX(ATMPA)=DX(ATMPA)+DXBRM*(1.D0-SWF)
                  DY(ATMPA)=DY(ATMPA)+DYBRM*(1.D0-SWF)
                  DZ(ATMPA)=DZ(ATMPA)+DZBRM*(1.D0-SWF)
                  
                  DX(ATMPB)=DX(ATMPB)-DXBRM*(1.D0-SWF)
                  DY(ATMPB)=DY(ATMPB)-DYBRM*(1.D0-SWF)
                  DZ(ATMPB)=DZ(ATMPB)-DZBRM*(1.D0-SWF)
                  
                  DX1=DSWFRNH*DRNNX*ERM
                  DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*ERM
        
                  DY1=DSWFRNH*DRNNY*ERM
                  DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*ERM
        
                  DZ1=DSWFRNH*DRNNZ*ERM
                  DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*ERM
        
                  DX(HBRDATOM(HB,1))=DX(HBRDATOM(HB,1))-DX1-DX2
                  DX(HBRDATOM(HB,2))=DX(HBRDATOM(HB,2))+DX1
                  DX(HBRDATOM(HB,3))=DX(HBRDATOM(HB,3))+DX2
        
                  DY(HBRDATOM(HB,1))=DY(HBRDATOM(HB,1))-DY1-DY2
                  DY(HBRDATOM(HB,2))=DY(HBRDATOM(HB,2))+DY1
                  DY(HBRDATOM(HB,3))=DY(HBRDATOM(HB,3))+DY2
        
                  DZ(HBRDATOM(HB,1))=DZ(HBRDATOM(HB,1))-DZ1-DZ2
                  DZ(HBRDATOM(HB,2))=DZ(HBRDATOM(HB,2))+DZ1
                  DZ(HBRDATOM(HB,3))=DZ(HBRDATOM(HB,3))+DZ2
                  
                  CALL EPSEUDOBOND(ATMPA,ATMPB,ERM, &
                  FCST2, EQUI2 ,0,DXBRM,DYBRM,DZBRM)
                  
                  EU = EU + ERM*SWF
                  DX(ATMPA)=DX(ATMPA)+DXBRM*SWF
                  DY(ATMPA)=DY(ATMPA)+DYBRM*SWF
                  DZ(ATMPA)=DZ(ATMPA)+DZBRM*SWF
                  
                  DX(ATMPB)=DX(ATMPB)-DXBRM*SWF
                  DY(ATMPB)=DY(ATMPB)-DYBRM*SWF
                  DZ(ATMPB)=DZ(ATMPB)-DZBRM*SWF
                  
                  DX1=DSWFRNH*DRNNX*ERM
                  DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*ERM
        
                  DY1=DSWFRNH*DRNNY*ERM
                  DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*ERM
        
                  DZ1=DSWFRNH*DRNNZ*ERM
                  DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*ERM
        
                  DX(HBRDATOM(HB,1))=DX(HBRDATOM(HB,1))+DX1+DX2
                  DX(HBRDATOM(HB,2))=DX(HBRDATOM(HB,2))-DX1
                  DX(HBRDATOM(HB,3))=DX(HBRDATOM(HB,3))-DX2
        
                  DY(HBRDATOM(HB,1))=DY(HBRDATOM(HB,1))+DY1+DY2
                  DY(HBRDATOM(HB,2))=DY(HBRDATOM(HB,2))-DY1
                  DY(HBRDATOM(HB,3))=DY(HBRDATOM(HB,3))-DY2
        
                  DZ(HBRDATOM(HB,1))=DZ(HBRDATOM(HB,1))+DZ1+DZ2
                  DZ(HBRDATOM(HB,2))=DZ(HBRDATOM(HB,2))-DZ1
                  DZ(HBRDATOM(HB,3))=DZ(HBRDATOM(HB,3))-DZ2
                
                  EU=EU-(EQUI1-EQUI2)**2/(1.D0/((1.D0-SWF)*FCST1)+ &
                  1.D0/(SWF*FCST2))
                  
                  DDSWF=0.D0-(EQUI1-EQUI2)**2*FCST1*FCST2/(SWF*FCST2+ &
                  (1.D0-SWF)*FCST1)**2*((1.D0-SWF)**2*FCST1-SWF**2*FCST2)
                              
                  DX1=DSWFRNH*DRNNX*DDSWF
                  DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*DDSWF
        
                  DY1=DSWFRNH*DRNNY*DDSWF
                  DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*DDSWF
        
                  DZ1=DSWFRNH*DRNNZ*DDSWF
                  DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*DDSWF
                  
                  DX(HBRDATOM(HB,1))=DX(HBRDATOM(HB,1))+(DX1+DX2)
                  DX(HBRDATOM(HB,2))=DX(HBRDATOM(HB,2))-DX1
                  DX(HBRDATOM(HB,3))=DX(HBRDATOM(HB,3))-DX2
                  
                  DY(HBRDATOM(HB,1))=DY(HBRDATOM(HB,1))+(DY1+DY2)
                  DY(HBRDATOM(HB,2))=DY(HBRDATOM(HB,2))-DY1
                  DY(HBRDATOM(HB,3))=DY(HBRDATOM(HB,3))-DY2
                  
                  DZ(HBRDATOM(HB,1))=DZ(HBRDATOM(HB,1))+(DZ1+DZ2)
                  DZ(HBRDATOM(HB,2))=DZ(HBRDATOM(HB,2))-DZ1
                  DZ(HBRDATOM(HB,3))=DZ(HBRDATOM(HB,3))-DZ2
                ENDIF
              ENDIF
            ENDDO
          ENDDO
        ENDIF
      ENDDO
!if(prnlev.ge.7) WRITE(outu,*) 'BDCORR > ',EU-EU1
!      REMARKS ON CONSTRAINTS:
! ... UNCLEAR WHAT HAPPENS TO ANGLES
!      ANGLES
EU1=EU
!WRITE(outu,*) 'TEST EU_BOND> ',eu

      
      DO I=1,ANGLNUM
         CALL EANGLMMPT(ANGLMMPT(I,1),ANGLMMPT(I,2),  &
             ANGLMMPT(I,3),ANGLMMPT(I,4),             &
             EA,DXAI,DYAI,DZAI,                       &
             DXAJ,DYAJ,DZAJ,                          &
             DXAK,DYAK,DZAK)



!      ASSUMES THAT TRANSFERED HYDROGEN ATOM IS ALWAYS FIRST OR FOURTH
!      ATOM IN MMPT ANGLE DEFINITION AND SECOND IN MMPT HYDROGEN BOND 
!      DEFINITION


         DO J=1, HBRDNUM
           IF ((ANGLMMPT(I,4).EQ.HBRDATOM(J,2))            &
          .OR.(ANGLMMPT(I,2).EQ.HBRDATOM(J,2))) THEN        
             CALL SWITCHMMPT(HBRDATOM(J,1),HBRDATOM(J,2),  &
                  HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX, &
                  DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
 !      CHECK IF ANGLE IS IN PSF OR NEW. EITHER REMOVE OR ADD ACCORDING
 !      TO SWITCHING FUNCTION
           EXIT
 !      STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
           ENDIF
         ENDDO  
         
         IF (ANGLMMPT(I,5).GT.0) THEN
            
            EU=EU-EA*SWF
            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> ANGLE ENERGY REMOVED ACCORDING TO SWITCH',  &
                    EA*SWF
            ENDIF
                        
            DX1=DSWFRNH*DRNNX*EA
            DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA
            
            DY1=DSWFRNH*DRNNY*EA
            DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA
         
            DZ1=DSWFRNH*DRNNZ*EA
            DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA



   
!            WRITE(OUTU,*) -(DXAI*SWF),-(DXAJ*SWF+DX1+DX2), &
!                         -(DXAK*SWF-DX1),+DX2
         
            DX(ANGLMMPT(I,2))=DX(ANGLMMPT(I,2))-(DXAI*SWF)
            DX(ANGLMMPT(I,3))=DX(ANGLMMPT(I,3))-(DXAJ*SWF+DX1+DX2)
            DX(ANGLMMPT(I,4))=DX(ANGLMMPT(I,4))-(DXAK*SWF-DX1) 
            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2

            DY(ANGLMMPT(I,2))=DY(ANGLMMPT(I,2))-(DYAI*SWF)
            DY(ANGLMMPT(I,3))=DY(ANGLMMPT(I,3))-(DYAJ*SWF+DY1+DY2)
            DY(ANGLMMPT(I,4))=DY(ANGLMMPT(I,4))-(DYAK*SWF-DY1)
            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
            
            DZ(ANGLMMPT(I,2))=DZ(ANGLMMPT(I,2))-(DZAI*SWF)
            DZ(ANGLMMPT(I,3))=DZ(ANGLMMPT(I,3))-(DZAJ*SWF+DZ1+DZ2)
            DZ(ANGLMMPT(I,4))=DZ(ANGLMMPT(I,4))-(DZAK*SWF-DZ1)
            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
            
         ELSE
            
            EU=EU+EA*SWF
            
            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> ANGLE ENERGY ADDED ACCORDING TO SWITCH', &
                    EA*SWF
            ENDIF
     
            DX1=DSWFRNH*DRNNX*EA
            DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA
            
            DY1=DSWFRNH*DRNNY*EA
            DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA
         
            DZ1=DSWFRNH*DRNNZ*EA
            DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA
         
            DX(ANGLMMPT(I,2))=DX(ANGLMMPT(I,2))+DXAI*SWF
            DX(ANGLMMPT(I,3))=DX(ANGLMMPT(I,3))+DXAJ*SWF-DX2
            DX(ANGLMMPT(I,4))=DX(ANGLMMPT(I,4))+DXAK*SWF-DX1 
            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX1+DX2
            
            DY(ANGLMMPT(I,2))=DY(ANGLMMPT(I,2))+DYAI*SWF
            DY(ANGLMMPT(I,3))=DY(ANGLMMPT(I,3))+DYAJ*SWF-DY2
            DY(ANGLMMPT(I,4))=DY(ANGLMMPT(I,4))+DYAK*SWF-DY1 
            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY1+DY2
         
            DZ(ANGLMMPT(I,2))=DZ(ANGLMMPT(I,2))+DZAI*SWF
            DZ(ANGLMMPT(I,3))=DZ(ANGLMMPT(I,3))+DZAJ*SWF-DZ2
            DZ(ANGLMMPT(I,4))=DZ(ANGLMMPT(I,4))+DZAK*SWF-DZ1 
            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ1+DZ2
            
         ENDIF

      ENDDO

if(prnlev.ge.7) WRITE(outu,*) 'AGLCORR > ',EU-EU1

EU1=EU
!WRITE(outu,*) 'TEST EU_BOND> ',eu

      
      DO I=1,ANGLNUM
        EA=0.D0
        CALL EPSEUDOMMPT(ANGLMMPT(I,2),ANGLMMPT(I,3),ANGLMMPT(I,4),EA, &
        200.D0,75.D0*(PI/180.d0),-1,DXAI,DYAI,DZAI,   &
        DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)

!      ASSUMES THAT TRANSFERED HYDROGEN ATOM IS ALWAYS FIRST OR FOURTH
!      ATOM IN MMPT ANGLE DEFINITION AND SECOND IN MMPT HYDROGEN BOND 
!      DEFINITION
        IF (EA.GT.0.D0) THEN
!write (outu,*) 'TEST> ', EA
          DO J=1, HBRDNUM
            IF ((ANGLMMPT(I,4).EQ.HBRDATOM(J,2))            &
           .OR.(ANGLMMPT(I,2).EQ.HBRDATOM(J,2))) THEN        
              CALL SWITCHMMPT(HBRDATOM(J,1),HBRDATOM(J,2),  &
                   HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX, &
                   DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
  !      CHECK IF ANGLE IS IN PSF OR NEW. EITHER REMOVE OR ADD ACCORDING
  !      TO SWITCHING FUNCTION
            EXIT
  !      STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
            ENDIF
          ENDDO  
         
          IF (ANGLMMPT(I,5).GT.0) THEN
             
             EU=EU+EA*(1.D0-SWF)
                         
             DX1=DSWFRNH*DRNNX*EA
             DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA
             
             DY1=DSWFRNH*DRNNY*EA
             DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA
          
             DZ1=DSWFRNH*DRNNZ*EA
             DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA
 
 !            WRITE(OUTU,*) -(DXAI*SWF),-(DXAJ*SWF+DX1+DX2), &
 !                         -(DXAK*SWF-DX1),+DX2
          
             DX(ANGLMMPT(I,2))=DX(ANGLMMPT(I,2))+(DXAI*(1.D0-SWF))
             DX(ANGLMMPT(I,3))=DX(ANGLMMPT(I,3))+(DXAJ*(1.D0-SWF)-DX1-DX2)
             DX(ANGLMMPT(I,4))=DX(ANGLMMPT(I,4))+(DXAK*(1.D0-SWF)+DX1) 
             DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
 
             DY(ANGLMMPT(I,2))=DY(ANGLMMPT(I,2))+(DYAI*(1.D0-SWF))
             DY(ANGLMMPT(I,3))=DY(ANGLMMPT(I,3))+(DYAJ*(1.D0-SWF)-DY1-DY2)
             DY(ANGLMMPT(I,4))=DY(ANGLMMPT(I,4))+(DYAK*(1.D0-SWF)+DY1) 
             DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
             
             DZ(ANGLMMPT(I,2))=DZ(ANGLMMPT(I,2))+(DZAI*(1.D0-SWF))
             DZ(ANGLMMPT(I,3))=DZ(ANGLMMPT(I,3))+(DZAJ*(1.D0-SWF)-DZ1-DZ2)
             DZ(ANGLMMPT(I,4))=DZ(ANGLMMPT(I,4))+(DZAK*(1.D0-SWF)+DZ1) 
             DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
             
          ELSE
             
             EU=EU+EA*SWF
      
             DX1=DSWFRNH*DRNNX*EA
             DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*EA
             
             DY1=DSWFRNH*DRNNY*EA
             DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*EA
          
             DZ1=DSWFRNH*DRNNZ*EA
             DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*EA
          
             DX(ANGLMMPT(I,2))=DX(ANGLMMPT(I,2))+DXAI*SWF
             DX(ANGLMMPT(I,3))=DX(ANGLMMPT(I,3))+DXAJ*SWF-DX2
             DX(ANGLMMPT(I,4))=DX(ANGLMMPT(I,4))+DXAK*SWF-DX1 
             DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX1+DX2
             
             DY(ANGLMMPT(I,2))=DY(ANGLMMPT(I,2))+DYAI*SWF
             DY(ANGLMMPT(I,3))=DY(ANGLMMPT(I,3))+DYAJ*SWF-DY2
             DY(ANGLMMPT(I,4))=DY(ANGLMMPT(I,4))+DYAK*SWF-DY1 
             DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY1+DY2
          
             DZ(ANGLMMPT(I,2))=DZ(ANGLMMPT(I,2))+DZAI*SWF
             DZ(ANGLMMPT(I,3))=DZ(ANGLMMPT(I,3))+DZAJ*SWF-DZ2
             DZ(ANGLMMPT(I,4))=DZ(ANGLMMPT(I,4))+DZAK*SWF-DZ1 
             DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ1+DZ2
             
          ENDIF
        ENDIF
      ENDDO

if(prnlev.ge.7) WRITE(outu,*) 'AGLPENT > ',EU-EU1
!WRITE(outu,*) 'TEST EU_EANGLE> ',eu

!     DIHEDRALS 
!      REMARKS ON CONSTRAINTS:
! ... UNCLEAR WHAT HAPPENS TO DIHEDRALS

             
      DO I=1,DIHENUM
      CALL EPHIMMPT(DIHEMMPT(I,1),DIHEMMPT(I,2),               &
                   DIHEMMPT(I,3),DIHEMMPT(I,4),DIHEMMPT(I,5),  &
                   NPHI,ICP,CPD,CPCOS,CPSIN,CPC,               &
          ED,DXDI,DXDJ,DXDK,DXDL,                              &
             DYDI,DYDJ,DYDK,DYDL,                              &
             DZDI,DZDJ,DZDK,DZDL)
 




!      THE TRANSFERED HYDROGEN ATOM IS EITHER FOURTH ATOM OR SECOND
!      ATOM IN DIHEDRAL DEFINITION AND ALWAYS SECOND IN HYDROGEN BOND
!      DEFINITION

         DO J=1, HBRDNUM
           IF (DIHEMMPT(I,5).EQ.HBRDATOM(J,2) .OR.         &
              DIHEMMPT(I,2).EQ.HBRDATOM(J,2) ) THEN         
             CALL SWITCHMMPT(HBRDATOM(J,1),HBRDATOM(J,2),  &
                  HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX, &
                  DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
             EXIT
!      STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
           ENDIF
         ENDDO  
          
!      CHECK IF DIHEDRAL IS IN PSF OR NEW. EITHER REMOVE OR ADD ACCORDING
!      TO SWITCHING FUNCTION         

         IF (DIHEMMPT(I,6).GT.0) THEN
            
            EU=EU-ED*SWF
          
            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> DIHEDRAL ENERGY REMOVED ACCORDING &
&                    TO SWITCH',  ED*SWF
            ENDIF

            DX2=DSWFRNH*DRNHX*ED+DSWFRNN*DRNNX*ED
            DX1=DSWFRNH*DRNNX*ED 
            
            DY2=DSWFRNH*DRNHY*ED+DSWFRNN*DRNNY*ED
            DY1=DSWFRNH*DRNNY*ED
            
            DZ2=DSWFRNH*DRNHZ*ED+DSWFRNN*DRNNZ*ED
            DZ1=DSWFRNH*DRNNZ*ED

          IF (DIHEMMPT(I,5).EQ.HBRDATOM(J,2)) THEN
            DX(DIHEMMPT(I,2))=DX(DIHEMMPT(I,2))-(DXDI*SWF)
            DX(DIHEMMPT(I,3))=DX(DIHEMMPT(I,3))-(DXDJ*SWF)
            DX(DIHEMMPT(I,4))=DX(DIHEMMPT(I,4))-(DXDK*SWF+DX1+DX2)
            DX(DIHEMMPT(I,5))=DX(DIHEMMPT(I,5))-(DXDL*SWF-DX1)
            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
            
            DY(DIHEMMPT(I,2))=DY(DIHEMMPT(I,2))-(DYDI*SWF)
            DY(DIHEMMPT(I,3))=DY(DIHEMMPT(I,3))-(DYDJ*SWF)
            DY(DIHEMMPT(I,4))=DY(DIHEMMPT(I,4))-(DYDK*SWF+DY1+DY2)
            DY(DIHEMMPT(I,5))=DY(DIHEMMPT(I,5))-(DYDL*SWF-DY1)
            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
            
            DZ(DIHEMMPT(I,2))=DZ(DIHEMMPT(I,2))-(DZDI*SWF)
            DZ(DIHEMMPT(I,3))=DZ(DIHEMMPT(I,3))-(DZDJ*SWF)
            DZ(DIHEMMPT(I,4))=DZ(DIHEMMPT(I,4))-(DZDK*SWF+DZ1+DZ2)
            DZ(DIHEMMPT(I,5))=DZ(DIHEMMPT(I,5))-(DZDL*SWF-DZ1)
            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
          ENDIF
          IF (DIHEMMPT(I,2).EQ.HBRDATOM(J,2)) THEN   
            DX(DIHEMMPT(I,5))=DX(DIHEMMPT(I,5))-(DXDL*SWF)
            DX(DIHEMMPT(I,4))=DX(DIHEMMPT(I,4))-(DXDK*SWF)
            DX(DIHEMMPT(I,3))=DX(DIHEMMPT(I,3))-(DXDJ*SWF+DX1+DX2)
            DX(DIHEMMPT(I,2))=DX(DIHEMMPT(I,2))-(DXDI*SWF-DX1)
            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
            
            DY(DIHEMMPT(I,5))=DY(DIHEMMPT(I,5))-(DYDL*SWF)
            DY(DIHEMMPT(I,4))=DY(DIHEMMPT(I,4))-(DYDK*SWF)
            DY(DIHEMMPT(I,3))=DY(DIHEMMPT(I,3))-(DYDJ*SWF+DY1+DY2)
            DY(DIHEMMPT(I,2))=DY(DIHEMMPT(I,2))-(DYDI*SWF-DY1)
            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
            
            DZ(DIHEMMPT(I,5))=DZ(DIHEMMPT(I,5))-(DZDL*SWF)
            DZ(DIHEMMPT(I,4))=DZ(DIHEMMPT(I,4))-(DZDK*SWF)
            DZ(DIHEMMPT(I,3))=DZ(DIHEMMPT(I,3))-(DZDJ*SWF+DZ1+DZ2)
            DZ(DIHEMMPT(I,2))=DZ(DIHEMMPT(I,2))-(DZDI*SWF-DZ1)
            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
          ENDIF

            
         ELSE
            
            EU=EU+ED*SWF
      
            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> DIHEDRAL ENERGY ADDED ACCORDING &
&                    TO SWITCH',ED*SWF
            ENDIF

            DX2=DSWFRNH*DRNHX*ED+DSWFRNN*DRNNX*ED
            DX1=DSWFRNH*DRNNX*ED 
            
            DY2=DSWFRNH*DRNHY*ED+DSWFRNN*DRNNY*ED
            DY1=DSWFRNH*DRNNY*ED
            
            DZ2=DSWFRNH*DRNHZ*ED+DSWFRNN*DRNNZ*ED
            DZ1=DSWFRNH*DRNNZ*ED


          IF (DIHEMMPT(I,5).EQ.HBRDATOM(J,2)) THEN
            DX(DIHEMMPT(I,2))=DX(DIHEMMPT(I,2))+DXDI*SWF
            DX(DIHEMMPT(I,3))=DX(DIHEMMPT(I,3))+DXDJ*SWF
            DX(DIHEMMPT(I,4))=DX(DIHEMMPT(I,4))+DXDK*SWF-DX2
            DX(DIHEMMPT(I,5))=DX(DIHEMMPT(I,5))+DXDL*SWF-DX1 
            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX1+DX2
            
            DY(DIHEMMPT(I,2))=DY(DIHEMMPT(I,2))+DYDI*SWF
            DY(DIHEMMPT(I,3))=DY(DIHEMMPT(I,3))+DYDJ*SWF
            DY(DIHEMMPT(I,4))=DY(DIHEMMPT(I,4))+DYDK*SWF-DY2
            DY(DIHEMMPT(I,5))=DY(DIHEMMPT(I,5))+DYDL*SWF-DY1 
            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY1+DY2
            
            DZ(DIHEMMPT(I,2))=DZ(DIHEMMPT(I,2))+DZDI*SWF
            DZ(DIHEMMPT(I,3))=DZ(DIHEMMPT(I,3))+DZDJ*SWF
            DZ(DIHEMMPT(I,4))=DZ(DIHEMMPT(I,4))+DZDK*SWF-DZ2
            DZ(DIHEMMPT(I,5))=DZ(DIHEMMPT(I,5))+DZDL*SWF-DZ1 
            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ1+DZ2           
          ENDIF
          IF (DIHEMMPT(I,2).EQ.HBRDATOM(J,2)) THEN
            DX(DIHEMMPT(I,5))=DX(DIHEMMPT(I,5))+DXDL*SWF
            DX(DIHEMMPT(I,4))=DX(DIHEMMPT(I,4))+DXDK*SWF
            DX(DIHEMMPT(I,3))=DX(DIHEMMPT(I,3))+DXDJ*SWF-DX2
            DX(DIHEMMPT(I,2))=DX(DIHEMMPT(I,2))+DXDI*SWF-DX1 
            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX1+DX2
            
            DY(DIHEMMPT(I,5))=DY(DIHEMMPT(I,5))+DYDL*SWF
            DY(DIHEMMPT(I,4))=DY(DIHEMMPT(I,4))+DYDK*SWF
            DY(DIHEMMPT(I,3))=DY(DIHEMMPT(I,3))+DYDJ*SWF-DY2
            DY(DIHEMMPT(I,2))=DY(DIHEMMPT(I,2))+DYDI*SWF-DY1 
            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY1+DY2
            
            DZ(DIHEMMPT(I,5))=DZ(DIHEMMPT(I,5))+DZDL*SWF
            DZ(DIHEMMPT(I,4))=DZ(DIHEMMPT(I,4))+DZDK*SWF
            DZ(DIHEMMPT(I,3))=DZ(DIHEMMPT(I,3))+DZDJ*SWF-DZ2
            DZ(DIHEMMPT(I,2))=DZ(DIHEMMPT(I,2))+DZDI*SWF-DZ1 
            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ1+DZ2           
          ENDIF 
         ENDIF
      ENDDO


!WRITE(outu,*) 'TEST EU_DIHE> ',eu


!      IMPROPER DIHEDRAL ANGLES
!      REMARKS ON CONSTRAINTS:
! ... UNCLEAR WHAT HAPPENS TO IMPROPER DIHEDRALS

      DO I=1, IMPRNUM
      CALL EPHIMMPT(IMPRMMPT(I,1),IMPRMMPT(I,2),             &
                   IMPRMMPT(I,3),IMPRMMPT(I,4),IMPRMMPT(I,5),&
                   NPHI,ICP,CPD,CPCOS,CPSIN,CPC,             &
         EI ,DXII,DXIJ,DXIK,DXIL,                            &
             DYII,DYIJ,DYIK,DYIL,                            &
             DZII,DZIJ,DZIK,DZIL)

!      ASSUMES THAT TRANSFERED HYDROGEN ATOM IS ALWAYS FOURTH
!      ATOM IN IMPROPER DEFINITION AND SECOND IN HYDROGEN BOND 
!      DEFINITION
         DO J=1, HBRDNUM
           IF (IMPRMMPT(I,5).EQ.HBRDATOM(J,2)) THEN
              CALL SWITCHMMPT(HBRDATOM(J,1),HBRDATOM(J,2),  &
                  HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX,  &
                  DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
              EXIT
 !      STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
           ENDIF
         ENDDO  
         
!      CHECK IF IMPROPER IS IN PSF OR NEW. EITHER REMOVE OR ADD ACCORDING
!      TO SWITCHING FUNCTION
         
       IF (IMPRMMPT(I,6).GT.0) THEN
            
            EU=EU-EI*SWF    
      
            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> IMPROPER ENERGY REMOVED ACCORDING &
&                    TO SWITCH',EI*SWF
            ENDIF
            DX1=DSWFRNH*DRNNX*EI 
            DX2=DSWFRNN*DRNNX*EI+DSWFRNH*DRNHX*EI
            
            DY1=DSWFRNH*DRNNY*EI
            DY2=DSWFRNN*DRNNY*EI+DSWFRNH*DRNHY*EI
            
            DZ1=DSWFRNH*DRNNZ*EI
            DZ2=DSWFRNN*DRNNZ*EI+DSWFRNH*DRNHZ*EI
            
            DX(IMPRMMPT(I,2))=DX(IMPRMMPT(I,2))-(DXII*SWF+DX1+DX2)
            DX(IMPRMMPT(I,3))=DX(IMPRMMPT(I,3))-(DXIJ*SWF)
            DX(IMPRMMPT(I,4))=DX(IMPRMMPT(I,4))-(DXIK*SWF)
            DX(IMPRMMPT(I,5))=DX(IMPRMMPT(I,5))-(DXIL*SWF-DX1)
            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
            
            DY(IMPRMMPT(I,2))=DY(IMPRMMPT(I,2))-(DYII*SWF+DY1+DY2)
            DY(IMPRMMPT(I,3))=DY(IMPRMMPT(I,3))-(DYIJ*SWF)
            DY(IMPRMMPT(I,4))=DY(IMPRMMPT(I,4))-(DYIK*SWF)
            DY(IMPRMMPT(I,5))=DY(IMPRMMPT(I,5))-(DYIL*SWF-DY1)
            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
            
            DZ(IMPRMMPT(I,2))=DZ(IMPRMMPT(I,2))-(DZII*SWF+DZ1+DZ2)
            DZ(IMPRMMPT(I,3))=DZ(IMPRMMPT(I,3))-(DZIJ*SWF)
            DZ(IMPRMMPT(I,4))=DZ(IMPRMMPT(I,4))-(DZIK*SWF)
            DZ(IMPRMMPT(I,5))=DZ(IMPRMMPT(I,5))-(DZIL*SWF-DZ1)
            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
            

            
         ELSE
            
            EU=EU+EI*SWF
    

            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> IMPROPER ENERGY ADDED ACCORDING &
&                    TO SWITCH',EI*SWF
            ENDIF

            DX1=DSWFRNH*DRNNX*EI 
            DX2=DSWFRNN*DRNNX*EI+DSWFRNH*DRNHX*EI
            
            DY1=DSWFRNH*DRNNY*EI
            DY2=DSWFRNN*DRNNY*EI+DSWFRNH*DRNHY*EI
            
            DZ1=DSWFRNH*DRNNZ*EI
            DZ2=DSWFRNN*DRNNZ*EI+DSWFRNH*DRNHZ*EI
                        
            DX(IMPRMMPT(I,2))=DX(IMPRMMPT(I,2))+DXII*SWF-DX2 
            DX(IMPRMMPT(I,3))=DX(IMPRMMPT(I,3))+DXIJ*SWF
            DX(IMPRMMPT(I,4))=DX(IMPRMMPT(I,4))+DXIK*SWF
            DX(IMPRMMPT(I,5))=DX(IMPRMMPT(I,5))+DXIL*SWF-DX1
            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX1+DX2
            
            DY(IMPRMMPT(I,2))=DY(IMPRMMPT(I,2))+DYII*SWF-DY2 
            DY(IMPRMMPT(I,3))=DY(IMPRMMPT(I,3))+DYIJ*SWF
            DY(IMPRMMPT(I,4))=DY(IMPRMMPT(I,4))+DYIK*SWF
            DY(IMPRMMPT(I,5))=DY(IMPRMMPT(I,5))+DYIL*SWF-DY1
            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY1+DY2
            
            DZ(IMPRMMPT(I,2))=DZ(IMPRMMPT(I,2))+DZII*SWF-DZ2 
            DZ(IMPRMMPT(I,3))=DZ(IMPRMMPT(I,3))+DZIJ*SWF
            DZ(IMPRMMPT(I,4))=DZ(IMPRMMPT(I,4))+DZIK*SWF
            DZ(IMPRMMPT(I,5))=DZ(IMPRMMPT(I,5))+DZIL*SWF-DZ1
            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ1+DZ2
      
      
            
         ENDIF
      ENDDO

!WRITE(outu,*) 'TEST EU_IMPR> ',eu
!      REMARKS ON CONSTRAINTS:
! ... UNCLEAR WHAT HAPPENS TO NONBONDED INTERACTION

      EU1=EU
!WRITE(outu,*) "MMPTTEST-00>",NONBNUM,EU1
      DO I=1, NONBNUM

!        STANDARD OR SPECIAL VDW FORCES

        IF (ABS(NONBMMPT(I,4)) .LT. 5) THEN

          CALL NBNDMMPT(NONBMMPT(I,1),NONBMMPT(I,2),NONBMMPT(I,3),&
            BNBND%INBLO,BNBND%JNB,MAXCN,          &
            DXIRM,DYIRM,DZIRM,DXJRM,DYJRM,DZJRM,                &
            ENBMMPT,EELMMPT)


          IF (NONBMMPT(I,4).EQ.0) THEN
!WRITE(outu,*) "MMPTTEST-01>"
             EU=EU-(EELMMPT+ENBMMPT)

             DX(NONBMMPT(I,2))=DX(NONBMMPT(I,2))-DXIRM
             DY(NONBMMPT(I,2))=DY(NONBMMPT(I,2))-DYIRM
             DZ(NONBMMPT(I,2))=DZ(NONBMMPT(I,2))-DZIRM
             
             DX(NONBMMPT(I,3))=DX(NONBMMPT(I,3))-DXJRM
             DY(NONBMMPT(I,3))=DY(NONBMMPT(I,3))-DYJRM
             DZ(NONBMMPT(I,3))=DZ(NONBMMPT(I,3))-DZJRM            
             
          ELSE
!WRITE(outu,*) "MMPTTEST-02>"
            DO J=1, HBRDNUM
              IF (NONBMMPT(I,2).EQ.HBRDATOM(J,2)              &
                 .OR. NONBMMPT(I,3).EQ.HBRDATOM(J,2)) THEN     
                CALL SWITCHMMPT(HBRDATOM(J,1),HBRDATOM(J,2),  &
                     HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX, &
                     DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
                 EXIT
!     ... STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
              ENDIF
            ENDDO  
!WRITE(outu,*) "MMPTTEST-03>"
            IF (NONBMMPT(I,4).EQ.1) THEN
!     .            REMOVE 1-4 INTERACTION AND ADD SWITCHED ENERGIE AND FORCE, IE
!     .            READ AS  -(EELMMPT+ENBMMPT)+(EELMMPT+ENBMMPT)*(1-SWF)
               EU=EU-(EELMMPT+ENBMMPT)*SWF

               DX1=DSWFRNH*DRNNX*(EELMMPT+ENBMMPT)
               DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*(EELMMPT+ENBMMPT)
               
               DY1=DSWFRNH*DRNNY*(EELMMPT+ENBMMPT)
               DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*(EELMMPT+ENBMMPT)
               
               DZ1=DSWFRNH*DRNNZ*(EELMMPT+ENBMMPT)
               DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*(EELMMPT+ENBMMPT)
               
               
               DX(NONBMMPT(I,2))=DX(NONBMMPT(I,2))-(DXIRM*SWF)
               DX(NONBMMPT(I,3))=DX(NONBMMPT(I,3))-(DXJRM*SWF-DX1)
               DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
               DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))-DX2-DX1
               
               DY(NONBMMPT(I,2))=DY(NONBMMPT(I,2))-(DYIRM*SWF)
               DY(NONBMMPT(I,3))=DY(NONBMMPT(I,3))-(DYJRM*SWF-DY1)
               DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
               DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))-DY2-DY1
               
               
               DZ(NONBMMPT(I,2))=DZ(NONBMMPT(I,2))-(DZIRM*SWF)
               DZ(NONBMMPT(I,3))=DZ(NONBMMPT(I,3))-(DZJRM*SWF-DZ1)
               DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
               DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))-DZ2-DZ1
               
               IF(PRNLEV.GT.7) THEN
                  WRITE(OUTU,*) 'MMPT> NONB ENERGY ACCORDING TO SWITCH', &
                       (EELMMPT+ENBMMPT)*(1.D0-SWF)                 
               ENDIF  
!WRITE(outu,*) "MMPTTEST-04>"


! todo change to if -1
                 
             ELSE
                
                EU=EU+(EELMMPT+ENBMMPT)*SWF

                DX1=DSWFRNH*DRNNX*(EELMMPT+ENBMMPT)
                DX2=(DSWFRNN*DRNNX+DSWFRNH*DRNHX)*(EELMMPT+ENBMMPT)
                
                DY1=DSWFRNH*DRNNY*(EELMMPT+ENBMMPT)
                DY2=(DSWFRNN*DRNNY+DSWFRNH*DRNHY)*(EELMMPT+ENBMMPT)
                
                DZ1=DSWFRNH*DRNNZ*(EELMMPT+ENBMMPT)
                DZ2=(DSWFRNN*DRNNZ+DSWFRNH*DRNHZ)*(EELMMPT+ENBMMPT)
                
             
                DX(NONBMMPT(I,2))=DX(NONBMMPT(I,2))+(DXIRM*SWF)
                DX(NONBMMPT(I,3))=DX(NONBMMPT(I,3))+(DXJRM*SWF-DX1)
                DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))-DX2
                DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX2+DX1
                
                DY(NONBMMPT(I,2))=DY(NONBMMPT(I,2))+(DYIRM*SWF)
                DY(NONBMMPT(I,3))=DY(NONBMMPT(I,3))+(DYJRM*SWF-DY1)
                DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))-DY2
                DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY2+DY1
                
                
                DZ(NONBMMPT(I,2))=DZ(NONBMMPT(I,2))+(DZIRM*SWF)
                DZ(NONBMMPT(I,3))=DZ(NONBMMPT(I,3))+(DZJRM*SWF-DZ1)
                DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))-DZ2
                DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ2+DZ1
!WRITE(outu,*) "MMPTTEST-05>"
                IF(PRNLEV.GT.7) THEN
                   WRITE(OUTU,*) 'MMPT> NONB ENERGY ACCORDING TO SWITCH',  &
                        (EELMMPT+ENBMMPT)*SWF                             
                ENDIF                                                     
             ENDIF                                                        
           ENDIF                                                           
                                                                           
        ELSE                                                       
          CALL EVDW14MMPT(NONBMMPT(I,1),NONBMMPT(I,2),NONBMMPT(I,3),&
             -14,BNBND%INBLO,BNBND%JNB,MAXCN,                       &
             DXIRM1,DYIRM1,DZIRM1,DXJRM1,DYJRM1,DZJRM1,             &
             EVDW1MMPT,EELMMPT)
          
          DO J=1, HBRDNUM             
            IF (NONBMMPT(I,2).EQ.HBRDATOM(J,2)                       &
               .OR. NONBMMPT(I,3).EQ.HBRDATOM(J,2)) THEN             
              CALL SWITCHMMPT(HBRDATOM(J,1),HBRDATOM(J,2),          &
                   HBRDATOM(J,3),SWF,DSWFRNN,DSWFRNH,DRNHX,         &
                   DRNHY,DRNHZ,DRNNX,DRNNY,DRNNZ)
               EXIT
!                     STOP SEARCH AFTER CORRECT HYDROGEN BRIGDE IS FOUND
            ENDIF
          ENDDO
          
          IF (NONBMMPT(I,4).EQ.28) THEN
            EU=EU-(EVDW1MMPT+EELMMPT)

            DX(NONBMMPT(I,2))=DX(NONBMMPT(I,2))-(DXIRM1)
            DX(NONBMMPT(I,3))=DX(NONBMMPT(I,3))-(DXJRM1)
            
            DY(NONBMMPT(I,2))=DY(NONBMMPT(I,2))-(DYIRM1)
            DY(NONBMMPT(I,3))=DY(NONBMMPT(I,3))-(DYJRM1)
            
            DZ(NONBMMPT(I,2))=DZ(NONBMMPT(I,2))-(DZIRM1)
            DZ(NONBMMPT(I,3))=DZ(NONBMMPT(I,3))-(DZJRM1)
            
            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> VDW1 ENERGY ACCORDING TO SWITCH',  &
                    (EVDW1MMPT)
            ENDIF
          ELSE IF (NONBMMPT(I,4).EQ.14) THEN

!     .            REMOVE 1-4 INTERACTION AND ADD SWITCHED ENERGIE AND FORCE, IE
!     .            READ AS  -(EVD1WMMPT)+(EVDW1MMPT)*(1-SWF)
            EU=EU-(EVDW1MMPT+EELMMPT)*SWF

            DX1=DSWFRNH*DRNNX*(EVDW1MMPT+EELMMPT)
            DX2=DSWFRNN*DRNNX*(EVDW1MMPT+EELMMPT)+DSWFRNH*DRNHX*(EVDW1MMPT+EELMMPT)
            
            DY1=DSWFRNH*DRNNY*(EVDW1MMPT+EELMMPT)
            DY2=DSWFRNN*DRNNY*(EVDW1MMPT+EELMMPT)+DSWFRNH*DRNHY*(EVDW1MMPT+EELMMPT)
            
            DZ1=DSWFRNH*DRNNZ*(EVDW1MMPT+EELMMPT)
            DZ2=DSWFRNN*DRNNZ*(EVDW1MMPT+EELMMPT)+DSWFRNH*DRNHZ*(EVDW1MMPT+EELMMPT)
            
            
            DX(NONBMMPT(I,2))=DX(NONBMMPT(I,2))-(DXIRM1*SWF)
            DX(NONBMMPT(I,3))=DX(NONBMMPT(I,3))-(DXJRM1*SWF-DX1)
            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))+DX2
            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))-DX2-DX1
            
            DY(NONBMMPT(I,2))=DY(NONBMMPT(I,2))-(DYIRM1*SWF)
            DY(NONBMMPT(I,3))=DY(NONBMMPT(I,3))-(DYJRM1*SWF-DY1)
            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))+DY2
            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))-DY2-DY1
            
            
            DZ(NONBMMPT(I,2))=DZ(NONBMMPT(I,2))-(DZIRM1*SWF)
            DZ(NONBMMPT(I,3))=DZ(NONBMMPT(I,3))-(DZJRM1*SWF-DZ1)
            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))+DZ2
            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))-DZ2-DZ1
            
            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> VDW1 ENERGY ACCORDING TO SWITCH',  &
                    (EVDW1MMPT+EELMMPT)*(1.D0-SWF)                 
            ENDIF
          ELSE
  
  !                 ADD STANDARD VDW FORCE
  
            EU=EU+(EVDW1MMPT+EELMMPT)*SWF
  
            DX1=DSWFRNH*DRNNX*(EVDW1MMPT+EELMMPT)
            DX2=DSWFRNN*DRNNX*(EVDW1MMPT+EELMMPT)+DSWFRNH*DRNHX*(EVDW1MMPT+EELMMPT)
            
            DY1=DSWFRNH*DRNNY*(EVDW1MMPT+EELMMPT)
            DY2=DSWFRNN*DRNNY*(EVDW1MMPT+EELMMPT)+DSWFRNH*DRNHY*(EVDW1MMPT+EELMMPT)
            
            DZ1=DSWFRNH*DRNNZ*(EVDW1MMPT+EELMMPT)
            DZ2=DSWFRNN*DRNNZ*(EVDW1MMPT+EELMMPT)+DSWFRNH*DRNHZ*(EVDW1MMPT+EELMMPT)
            
         
            DX(NONBMMPT(I,2))=DX(NONBMMPT(I,2))+(DXIRM1*SWF)
            DX(NONBMMPT(I,3))=DX(NONBMMPT(I,3))+(DXJRM1*SWF-DX1)
            DX(HBRDATOM(J,3))=DX(HBRDATOM(J,3))-DX2
            DX(HBRDATOM(J,1))=DX(HBRDATOM(J,1))+DX2+DX1
            
            DY(NONBMMPT(I,2))=DY(NONBMMPT(I,2))+(DYIRM1*SWF)
            DY(NONBMMPT(I,3))=DY(NONBMMPT(I,3))+(DYJRM1*SWF-DY1)
            DY(HBRDATOM(J,3))=DY(HBRDATOM(J,3))-DY2
            DY(HBRDATOM(J,1))=DY(HBRDATOM(J,1))+DY2+DY1
            
            
            DZ(NONBMMPT(I,2))=DZ(NONBMMPT(I,2))+(DZIRM1*SWF)
            DZ(NONBMMPT(I,3))=DZ(NONBMMPT(I,3))+(DZJRM1*SWF-DZ1)
            DZ(HBRDATOM(J,3))=DZ(HBRDATOM(J,3))-DZ2
            DZ(HBRDATOM(J,1))=DZ(HBRDATOM(J,1))+DZ2+DZ1
            
            IF(PRNLEV.GT.7) THEN
               WRITE(OUTU,*) 'MMPT> VDW2 ENERGY ACCORDING TO SWITCH', &
                    (EVDW1MMPT+EELMMPT)*SWF                  
            ENDIF

          ENDIF
        ENDIF
      ENDDO
if(prnlev.ge.7) WRITE(outu,*) 'NBCORR > ',eu-eu1
    ENDIF ! IF QMMPT

  END SUBROUTINE EMMPT



!      _______________________________________________________



  SUBROUTINE PLZFCT(I,DA,HA,AA)
 
    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
!    use energym
    use coord
      
      INTEGER I,DA,HA,AA
      real(chm_real) RDA,RDH,RHO,PLZFCTDA,PLZFCTHA,PLZFCTAA

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A 

      RDA = DSQRT((X(AA)-X(DA))**2+(Y(AA)-Y(DA))**2+(Z(AA)-Z(DA))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((X(HA)-X(DA))**2+(Y(HA)-Y(DA))**2+(Z(HA)-Z(DA))**2)


!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]
      
      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)
 
!      FUNCTION OF POLARIZATION OF DONOR ATOM
      PLZFCTDA = EXP(-12.32D0*(RHO+0.17D0))-0.01
!      FUNCTION OF POLARIZATION OF ACCEPTOR ATOM
      PLZFCTAA = EXP(12.32D0*(RHO-1.17D0))-0.01
!      FUNCTION OF POLARIZATION OF HYDROGEN ATOM
!      PLZFCTHA = -PLZFCTDA-PLZFCTAA

      CG(AA)=INITCHRG(I,3)-INITCHRG(I,3)*PLZFCTAA
      CG(DA)=INITCHRG(I,1)-INITCHRG(I,1)*PLZFCTDA
      CG(HA)=INITCHRG(I,2)+INITCHRG(I,1)*PLZFCTDA+INITCHRG(I,3)*PLZFCTAA

     
!      WRITE(OUTU,*)  CG(DA)+CG(HA)+CG(AA)

  END SUBROUTINE PLZFCT


!      _______________________________________________________




  SUBROUTINE EPTOHO(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)

    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
!    use energym
!    use deriv
!    use coord


!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1]
      real(chm_real) RDA, RDA2, RDH, RHO , R1 , R2
!      XA: LOCAL X COORDINATES
!      YA: LOCAL Y COORDINATES
!      ZA: LOCAL Z COORDINATES
      real(chm_real) XA(3), YA(3), ZA(3)
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX) 
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX) 
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3)

      real(chm_real) THETA, COSTHETA
!      PARTIAL DERIVATIVES
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,          &
            DRHODY1, DRHODY2, DRHODY3,           &
            DRHODZ1, DRHODZ2, DRHODZ3

     real(chm_real) DRDADX1, DRDADX2, DRDADX3,           &
            DRDADY1, DRDADY2, DRDADY3,           &
            DRDADZ1, DRDADZ2, DRDADZ3

     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,        &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,        &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3

     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,     &
            DTHETADY1, DTHETADY2, DTHETADY3,     &
            DTHETADZ1, DTHETADZ2, DTHETADZ3


     real(chm_real) DX1, DX2, DX3,                       &
            DY1, DY2, DY3,                       &
            DZ1, DZ2, DZ3

!      PARAMETER

      real(chm_real) DEQ, B, RE, H1, DDEQ, DB, DRE, DH1


      real(chm_real) DOTPROD




!      PASS COORDINATES TO POTENTIAL ROUTINE

      R1=DSQRT((XA(HA)-XA(DA))**2+(YA(HA)-YA(DA))**2+(ZA(HA)-ZA(DA))**2)
      R2=DSQRT((XA(HA)-XA(AA))**2+(YA(HA)-YA(AA))**2+(ZA(HA)-ZA(AA))**2)

      XA(1) = X(DA)
      YA(1) = Y(DA)
      ZA(1) = Z(DA)

      XA(2) = X(HA)
      YA(2) = Y(HA)
      ZA(2) = Z(HA)

      XA(3) = X(AA)
      YA(3) = Y(AA)
      ZA(3) = Z(AA)

!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
             +(YA(3)-YA(1))*(YA(2)-YA(1))    &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))

      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)



!      CONVERSION RAD -> DEG

      THETA = THETA*RADDEG







!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]

      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)



      IF(PRNLEV.GT.7) THEN
      write(outu,*) 'entering OHO mmpt routine'
      WRITE(OUTU,*) 'MMPT>   RDA', RDA
      WRITE(OUTU,*) 'MMPT>   RDH', RDH
      WRITE(OUTU,*) 'MMPT>   RHO', RHO
      WRITE(OUTU,*) 'MMPT>   COS(THETA)', COSTHETA
!      WRITE(OUTU,*) 'MMPT>   THETA[RAD]', THETA*PI/180.D0
      WRITE(OUTU,*) 'MMPT>   THETA', THETA
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER:
      DEQ=PRMOHO(1)*(1-DEXP(-PRMOHO(2)*(RDA-PRMOHO(3))))**2+PRMOHO(4)
      B =PRMOHO(5)+PRMOHO(6)*RDA
      RE=PRMOHO(7)*DEXP(-PRMOHO(8)*RDA)+PRMOHO(9)
      H1=PRMOHO(10)



!      DERIVE PARAMETER FUNCTIONS P(RDA)
      DDEQ= 2*PRMOHO(1)*(1.D0-DEXP(-PRMOHO(2)*(RDA-PRMOHO(3))))    &
            *PRMOHO(2)*DEXP(-PRMOHO(2)*(RDA-PRMOHO(3)))
      DB = PRMOHO(6)
      DRE= -PRMOHO(7)*PRMOHO(8)*DEXP(-PRMOHO(8)*RDA)
      DH1=0.D0


!      POTENTIAL FUNCTION (DOUBLE MORSE POTENTIAL)

      V0 = DEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                            &
         +DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                        &
         -DEQ+PRMOHO(11)

     DV0DRDA = DDEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                       &
              -2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                      &
              *(-DB*(RHO-RE)+B*DRE)*DEXP(-B*(RHO-RE))                 &
              +DDEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                  &
              -2.D0*DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))                 &
              *(-DB*(1.D0-RHO-RE)+B*DRE)*DEXP(-B*(1.D0-RHO-RE))       &
              -DDEQ



     DV0DRHO = 2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                      &
               *B*DEXP(-B*(RHO-RE))                                   &
               -2.D0*DEQ*(1.D0-DEXP(-B*(1-RHO-RE)))                   &
               *B*DEXP(-B*(1-RHO-RE))


!      HARMONIC RESTRAINT:

      HR =  H1*THETA**2

      DHRDTHET = 2.D0*H1*THETA

!      POTENTIAL ENERGY SCALING

      V0=SCLOHO*V0
      DV0DRDA=SCLOHO*DV0DRDA
      DV0DRHO=SCLOHO*DV0DRHO

!      SUM TERMS

      VTOT = V0 + HR

      EU = EU + VTOT



!      CALCULATE PARTIAL DERIVATIVES

      DRHODX1 =(-XA(2)+XA(1))/(RDH*( RDA-1.6))          &
             -((RDH-0.8D0)*  (-XA(3)+XA(1)))      &
             /(( RDA-1.6D0)**2*RDA)
     DRHODX2 =(XA(2)-XA(1))/( RDH*(RDA-1.6D0))
     DRHODX3 =-(RDH-0.8D0)*  (XA(3)-XA(1))        &
             /(( RDA-1.6D0)**2*RDA)

     DRHODY1 =(-YA(2)+YA(1))/(RDH*( RDA-1.6))           &
             -((RDH-0.8D0)*  (-YA(3)+YA(1)))      &
             /(( RDA-1.6D0)**2*RDA)
     DRHODY2 =(YA(2)-YA(1))/( RDH*(RDA-1.6D0))
     DRHODY3 =-(RDH-0.8D0)*  (YA(3)-YA(1))        &
             /(( RDA-1.6D0)**2*RDA)

     DRHODZ1 =(-ZA(2)+ZA(1))/(RDH*( RDA-1.6))           &
             -((RDH-0.8D0)*  (-ZA(3)+ZA(1)))      &
             /(( RDA-1.6D0)**2*RDA)
     DRHODZ2 =(ZA(2)-ZA(1))/( RDH*(RDA-1.6D0))
     DRHODZ3 =-(RDH-0.8D0)*  (ZA(3)-ZA(1))        &
             /(( RDA-1.6D0)**2*RDA)


!      RESET RDA

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)

!      CALCULATE PARTIAL DERIVATIVES

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA

      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA






!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E.
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE
!      D(THETA)/D(U)*D(U)/D(X1)


      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/(DSQRT(1.D0-COSTHETA**2))*180.D0/PI
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                         &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)

     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                         &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1
      DTHETADX2=DTDCOST*DCOSTDX2
      DTHETADX3=DTDCOST*DCOSTDX3

      DTHETADY1=DTDCOST*DCOSTDY1
      DTHETADY2=DTDCOST*DCOSTDY2
      DTHETADY3=DTDCOST*DCOSTDY3

      DTHETADZ1=DTDCOST*DCOSTDZ1
      DTHETADZ2=DTDCOST*DCOSTDZ2
      DTHETADZ3=DTDCOST*DCOSTDZ3


!      CALCULATE FORCES

      DX1 = DV0DRHO*DRHODX1 + DV0DRDA*DRDADX1 + DHRDTHET*DTHETADX1
      DX2 = DV0DRHO*DRHODX2 + DV0DRDA*DRDADX2 + DHRDTHET*DTHETADX2
      DX3 = DV0DRHO*DRHODX3 + DV0DRDA*DRDADX3 + DHRDTHET*DTHETADX3

      DY1 = DV0DRHO*DRHODY1 + DV0DRDA*DRDADY1 + DHRDTHET*DTHETADY1
      DY2 = DV0DRHO*DRHODY2 + DV0DRDA*DRDADY2 + DHRDTHET*DTHETADY2
      DY3 = DV0DRHO*DRHODY3 + DV0DRDA*DRDADY3 + DHRDTHET*DTHETADY3

      DZ1 = DV0DRHO*DRHODZ1 + DV0DRDA*DRDADZ1 + DHRDTHET*DTHETADZ1
      DZ2 = DV0DRHO*DRHODZ2 + DV0DRDA*DRDADZ2 + DHRDTHET*DTHETADZ2
      DZ3 = DV0DRHO*DRHODZ3 + DV0DRDA*DRDADZ3 + DHRDTHET*DTHETADZ3


      IF(PRNLEV.GT.7) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT ENERGY: ', VTOT
      WRITE(OUTU,*) 'MMPT> PT V0: ', V0
      WRITE(OUTU,*) 'MMPT> PT k*theta2: ', HR
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',AA,DX3,DY3,DZ3

      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EPTOHO



!      _______________________________________________________
!      _______________________________________________________


  SUBROUTINE EPTNHN(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)
   
   

    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
!    use energym
!    use deriv
!    use coord
               

!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY 
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM 
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1] 
      real(chm_real) RDA, RDA2, RDH, RHO
!      XA: LOCAL X COORDINATES 
!      YA: LOCAL Y COORDINATES 
!      ZA: LOCAL Z COORDINATES 
      real(chm_real) XA(3), YA(3), ZA(3)
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX) 
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX) 
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3) 

      real(chm_real) THETA, COSTHETA
!      PARTIAL DERIVATIVES 
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,     &
            DRHODY1, DRHODY2, DRHODY3,      &
            DRHODZ1, DRHODZ2, DRHODZ3
                                             
     real(chm_real) DRDADX1, DRDADX2, DRDADX3,      &
            DRDADY1, DRDADY2, DRDADY3,      &
            DRDADZ1, DRDADZ2, DRDADZ3
                                             
     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,   &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,   &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3 
                                             
     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,&
            DTHETADY1, DTHETADY2, DTHETADY3,&
            DTHETADZ1, DTHETADZ2, DTHETADZ3 

                                             
     real(chm_real) DX1, DX2, DX3,                  &
            DY1, DY2, DY3,                  &
            DZ1, DZ2, DZ3    
       
!      PARAMETER

      real(chm_real) DEQ, B, RE, H1, DDEQ, DB, DRE, DH1


      real(chm_real) DOTPROD


!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) =X(DA)
      YA(1) =Y(DA)
      ZA(1) =Z(DA)

      XA(2) =X(HA)
      YA(2) =Y(HA)
      ZA(2) =Z(HA)

      XA(3) =X(AA)
      YA(3) =Y(AA)
      ZA(3) =Z(AA)



!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A 

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))       &
             +(YA(3)-YA(1))*(YA(2)-YA(1))        &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))
     
      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)

           

!      CONVERSION RAD -> DEG

      THETA = THETA*RADDEG





!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]
      
      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)





      IF(PRNLEV.GT.7) THEN
      write(outu,*) 'entering NHN mmpt routine'
      WRITE(OUTU,*) 'MMPT>   RDA', RDA
      WRITE(OUTU,*) 'MMPT>   RDH', RDH
      WRITE(OUTU,*) 'MMPT>   RHO', RHO
      WRITE(OUTU,*) 'MMPT>   COS(THETA)', COSTHETA
!      WRITE(OUTU,*) 'MMPT>   THETA[RAD]', THETA*PI/180.D0
      WRITE(OUTU,*) 'MMPT>   THETA', THETA 
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER:
      DEQ=PRMNHN(1)*(1-DEXP(-PRMNHN(2)*(RDA-PRMNHN(3))))**2+PRMNHN(4)
      B =PRMNHN(5)+PRMNHN(6)*RDA
      RE=PRMNHN(7)*DEXP(-PRMNHN(8)*RDA)+PRMNHN(9)
      H1=PRMNHN(10)  



!      DERIVE PARAMETER FUNCTIONS P(RDA)
      DDEQ= 2*PRMNHN(1)*(1.D0-DEXP(-PRMNHN(2)*(RDA-PRMNHN(3))))   &
            *PRMNHN(2)*DEXP(-PRMNHN(2)*(RDA-PRMNHN(3)))            
     DB = PRMNHN(6)                                                
     DRE= -PRMNHN(7)*PRMNHN(8)*DEXP(-PRMNHN(8)*RDA)                
     DH1=0.D0                                                      
     
                                                                   
!     POTENTIAL FUNKTION (DOUBLE MORSE POTENTIAL)
                                                                   
     V0 = DEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                         &
         +DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                    &
         -DEQ+PRMNHN(11)
                                                                   
     DV0DRDA = DDEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                   &
              -2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                  &
              *(-DB*(RHO-RE)+B*DRE)*DEXP(-B*(RHO-RE))             &
              +DDEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2              &
              -2.D0*DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))             &
              *(-DB*(1.D0-RHO-RE)+B*DRE)*DEXP(-B*(1.D0-RHO-RE))   &
              -DDEQ


                                                                   
     DV0DRHO = 2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                  &
               *B*DEXP(-B*(RHO-RE))                               &
               -2.D0*DEQ*(1.D0-DEXP(-B*(1-RHO-RE)))               &
               *B*DEXP(-B*(1-RHO-RE))




!      HARMONIC RESTRAINT:

      HR =  H1*THETA**2

      DHRDTHET = 2.D0*H1*THETA


!      POTENTIAL ENERGY SCALING

      V0=SCLNHN*V0    
      DV0DRDA=SCLNHN*DV0DRDA    
      DV0DRHO=SCLNHN*DV0DRHO   

!      SUM TERMS

      VTOT = V0 + HR 

      EU = EU + VTOT



!      RESET RDA
    
      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE PARTIAL DERIVATIVES
                                                           
      DRHODX1 =(-XA(2)+XA(1))/(RDH*(RDA-1.6))  &
             -((RDH-0.8D0)*(-XA(3)+XA(1)))     &
             /((RDA-1.6D0)**2*RDA)              
     DRHODX2 =(XA(2)-XA(1))/(RDH*(RDA-1.6D0))   
     DRHODX3 =-(RDH-0.8D0)*(XA(3)-XA(1))       &
             /((RDA-1.6D0)**2*RDA)
                                                
     DRHODY1 =(-YA(2)+YA(1))/(RDH*(RDA-1.6))   &
             -((RDH-0.8D0)*(-YA(3)+YA(1)))     &
             /((RDA-1.6D0)**2*RDA)              
     DRHODY2 =(YA(2)-YA(1))/(RDH*(RDA-1.6D0))   
     DRHODY3 =-(RDH-0.8D0)*(YA(3)-YA(1))       &
             /((RDA-1.6D0)**2*RDA)
                                                
     DRHODZ1 =(-ZA(2)+ZA(1))/(RDH*(RDA-1.6))   &
             -((RDH-0.8D0)*(-ZA(3)+ZA(1)))     &
             /((RDA-1.6D0)**2*RDA)              
     DRHODZ2 =(ZA(2)-ZA(1))/(RDH*(RDA-1.6D0))   
     DRHODZ3 =-(RDH-0.8D0)*(ZA(3)-ZA(1))       &
             /((RDA-1.6D0)**2*RDA)




!      CALCULATE PARTIAL DERIVATIVES 

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA
 
      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA
 


!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E. 
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE 
!      D(THETA)/D(U)*D(U)/D(X1)
      

      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/(DSQRT(1.D0-COSTHETA**2))*180.D0/PI
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                      &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)     
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)
                                                                       
     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)     
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)
                                                                       
     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)     
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1 
      DTHETADX2=DTDCOST*DCOSTDX2 
      DTHETADX3=DTDCOST*DCOSTDX3 

      DTHETADY1=DTDCOST*DCOSTDY1 
      DTHETADY2=DTDCOST*DCOSTDY2 
      DTHETADY3=DTDCOST*DCOSTDY3 

      DTHETADZ1=DTDCOST*DCOSTDZ1 
      DTHETADZ2=DTDCOST*DCOSTDZ2 
      DTHETADZ3=DTDCOST*DCOSTDZ3 


!      CALCULATE FORCES

      DX1 = DV0DRHO*DRHODX1 + DV0DRDA*DRDADX1 + DHRDTHET*DTHETADX1
      DX2 = DV0DRHO*DRHODX2 + DV0DRDA*DRDADX2 + DHRDTHET*DTHETADX2
      DX3 = DV0DRHO*DRHODX3 + DV0DRDA*DRDADX3 + DHRDTHET*DTHETADX3

      DY1 = DV0DRHO*DRHODY1 + DV0DRDA*DRDADY1 + DHRDTHET*DTHETADY1
      DY2 = DV0DRHO*DRHODY2 + DV0DRDA*DRDADY2 + DHRDTHET*DTHETADY2
      DY3 = DV0DRHO*DRHODY3 + DV0DRDA*DRDADY3 + DHRDTHET*DTHETADY3

      DZ1 = DV0DRHO*DRHODZ1 + DV0DRDA*DRDADZ1 + DHRDTHET*DTHETADZ1
      DZ2 = DV0DRHO*DRHODZ2 + DV0DRDA*DRDADZ2 + DHRDTHET*DTHETADZ2
      DZ3 = DV0DRHO*DRHODZ3 + DV0DRDA*DRDADZ3 + DHRDTHET*DTHETADZ3

          
      IF(PRNLEV.GT.7) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT ENERGY: ', VTOT   
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',AA,DX3,DY3,DZ3
 
      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2      
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EPTNHN






!      _______________________________________________________




 
  SUBROUTINE EPTNHO(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)
   
    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
!    use deriv
    use param
!    use energym
!    use coord

!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY 
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM 
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1] 
      real(chm_real) RDA, RDA2, RDH, RHO
!      XA: LOCAL X COORDINATES 
!      YA: LOCAL Y COORDINATES 
!      ZA: LOCAL Z COORDINATES 
      real(chm_real) XA(3), YA(3), ZA(3)
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX)
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX) 
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3) 

      real(chm_real) THETA, COSTHETA
!      PARTIAL DERIVATIVES 
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,       &
            DRHODY1, DRHODY2, DRHODY3,        &
            DRHODZ1, DRHODZ2, DRHODZ3
                                               
     real(chm_real) DRDADX1, DRDADX2, DRDADX3,        &
            DRDADY1, DRDADY2, DRDADY3,        &
            DRDADZ1, DRDADZ2, DRDADZ3
                                               
     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,     &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,     &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3 
                                               
     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,  &
            DTHETADY1, DTHETADY2, DTHETADY3,  &
            DTHETADZ1, DTHETADZ2, DTHETADZ3 


      real(chm_real) DX1, DX2, DX3,                                         &
            DY1, DY2, DY3,                                          &
            DZ1, DZ2, DZ3                                            
                                                                     
!     PARAMETER
                                                                     
     real(chm_real) DEQ1, DEQ2, BETA1, BETA2, REQ1, REQ2, C, T,             &
            DDEQ1, DDEQ2, DBETA1, DBETA2, DREQ1, DREQ2, DC, DT


      real(chm_real) DOTPROD


      integer i

!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) =X(DA)
      YA(1) =Y(DA)
      ZA(1) =Z(DA)

      XA(2) =X(HA)
      YA(2) =Y(HA)
      ZA(2) =Z(HA)

      XA(3) =X(AA)
      YA(3) =Y(AA)
      ZA(3) =Z(AA)



!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A 

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
             +(YA(3)-YA(1))*(YA(2)-YA(1))    &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))
     
      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)

           

!      CONVERSION RAD -> DEG

      THETA = THETA*RADDEG








!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]
      
      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)



      IF(PRNLEV.GT.7) THEN
      write(outu,*) 'entering NHO mmpt routine'
      WRITE(OUTU,*) 'MMPT>   RDA', RDA
      WRITE(OUTU,*) 'MMPT>   RDH', RDH
      WRITE(OUTU,*) 'MMPT>   RHO', RHO
      WRITE(OUTU,*) 'MMPT>   COS(THETA)', COSTHETA
!      WRITE(OUTU,*) 'MMPT>   THETA[RAD]', THETA*PI/180.D0
      WRITE(OUTU,*) 'MMPT>   THETA', THETA 
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER, PARAMETER FUNCTION ARE FUNCTIONS
!      OF RDA


      DEQ1 =PRMNHO(1)*(1.D0-dexp(-PRMNHO(2)*(RDA-PRMNHO(3))))**2       &
          +PRMNHO(4)

                                                                        
     BETA1=PRMNHO(5)/(1.D0+DEXP(-PRMNHO(6)*(RDA-PRMNHO(7))))
                                                                        
!     write(outu,*) 'beta1', beta1                                      
!     write(outu,*) 'using', prmnho(5),prmnho(6),prmnho(7)

                                                                        
     REQ1 =PRMNHO(8)*(1.D0-dexp(-PRMNHO(9)*(RDA-PRMNHO(10))))**2       &
          +PRMNHO(11)
                                                                        
     DEQ2 =PRMNHO(12)*(1.D0-dexp(-PRMNHO(13)*(RDA-PRMNHO(14))))**2     &
          +PRMNHO(15)
                                                                        
     BETA2=PRMNHO(16)/(1.D0+DEXP(-PRMNHO(17)*(RDA-PRMNHO(18))))
                                                                        
     REQ2 =PRMNHO(19)*(1.D0-dexp(-PRMNHO(20)*(RDA-PRMNHO(21))))**2     &
          +PRMNHO(22)

      C    =PRMNHO(23)*(1.D0-dexp(-PRMNHO(24)*(RDA-PRMNHO(25))))**2    &
          +PRMNHO(26)
   
      T    =PRMNHO(27)      
      
      IF(PRNLEV.GT.7) THEN 
      DO I=1, NPRMNHO
         write(outu,*) 'MMPT>  PARAM', I, PRMNHO(I)
      ENDDO
      ENDIF

!      write(outu,*) 'parameter functions return:'
!      write(outu,*) deq1,beta1,req1,deq2,beta2,req2,c,t


!      DERIVE PARAMETER FUNCTIONS P(RDA)


      ddeq1 = 2*prmnho(1)*(1-exp(-prmnho(2)*(rda-prmnho(3))))*         &
               prmnho(2)*exp(-prmnho(2)*(rda-prmnho(3)))
 
                                                                        
     dbeta1 =  prmnho(5)*prmnho(6)*exp(-prmnho(6)*(rda-prmnho(7)))/    &
               (1+exp(-prmnho(6)*(rda-prmnho(7))))**2

                                                                        
!     write(outu,*) 'dbeta1', dbeta1                                    
!     write(outu,*) 'using', prmnho(5),prmnho(6),prmnho(7)


                                                                        
     dreq1 = 2*prmnho(8)*(1-exp(-prmnho(9)*(rda-prmnho(10))))*         &
               prmnho(9)*exp(-prmnho(9)*(rda-prmnho(10)))
                                                                        
     ddeq2 = 2*prmnho(12)*(1-exp(-prmnho(13)*(rda-prmnho(14))))*       &
               prmnho(13)*exp(-prmnho(13)*(rda-prmnho(14)))
                                                                        
     dbeta2 =  prmnho(16)*prmnho(17)*exp(-prmnho(17)*(rda-prmnho(18)))/&
               (1+exp(-prmnho(17)*(rda-prmnho(18))))**2
                                                                        
     dreq2 = 2*prmnho(19)*(1-exp(-prmnho(20)*(rda-prmnho(21))))*       &
               prmnho(20)*exp(-prmnho(20)*(rda-prmnho(21)))

                                                                        
     dc    = 2*prmnho(23)*(1-exp(-prmnho(24)*(rda-prmnho(25))))*       &
               prmnho(24)*exp(-prmnho(24)*(rda-prmnho(25)))
 
     

!      write(outu,*) 'derivative parameter functions return:'
!      write(outu,*) ddeq1,dbeta1,dreq1,ddeq2,dbeta2,dreq2,dc,dt



      

!      POTENTIAL FUNCTION (DOUBLE MORSE POTENTIAL)
! ... in contrast to eptoho or eptnhn we have two independent potentials

      v0 = Deq1*(1.d0-dexp(-beta1*(rho-Req1)))**2                      &
         +Deq2*(1.d0-dexp(-beta2*(Req2-rho)))**2                       &
         -c
 

                                                                        
     dv0drda = dDeq1*(1-exp(-beta1*(rho-Req1)))**2                     &
             -2*Deq1*(1-exp(-beta1*(rho-Req1)))*                       &
              (-dbeta1*(rho-Req1)+beta1*dReq1)*exp(-beta1*(rho-Req1))  &
              +dDeq2*(1-exp(-beta2*(Req2-rho)))**2                     &
             -2*Deq2*(1-exp(-beta2*(Req2-rho)))*                       &
              (-dbeta2*(Req2-rho)-beta2*dReq2)*exp(-beta2*(Req2-rho))  &
             -dc


!      write(outu,*) 'dv0drda', dv0drda


!      write(outu,*) dDeq1*(1-exp(-beta1*(rho-Req1)))**2
!      write(outu,*) 'using',ddeq1,beta1,rho,req1

!      write(outu,*) -2*Deq1*(1-exp(-beta1*(rho-Req1)))

       dv0drho = 2*Deq1*(1-exp(-beta1*(rho-Req1)))*  &
                beta1*exp(-beta1*(rho-Req1))         &
               -2*Deq2*(1-exp(-beta2*(Req2-rho)))*   &
                beta2*exp(-beta2*(Req2-rho))


!      write(outu,*) 'dv0drho', dv0drho



!      HARMONIC RESTRAINT:

      HR =  T*THETA**2

      DHRDTHET = 2.D0*T*THETA

!      POTENTIAL ENERGY SCALING

      V0=SCLOHO*V0    
      DV0DRDA=SCLOHO*DV0DRDA    
      DV0DRHO=SCLOHO*DV0DRHO   

!      SUM TERMS
      
      VTOT = V0 + HR 

      EU = EU + VTOT



!      CALCULATE PARTIAL DERIVATIVES
                                                           
      DRHODX1 =(-XA(2)+XA(1))/(RDH*( RDA-1.6))           &
             -((RDH-0.8D0)*  (-XA(3)+XA(1)))       &
             /(( RDA-1.6D0)**2*RDA)                       
     DRHODX2 =(XA(2)-XA(1))/( RDH*(RDA-1.6D0))            
     DRHODX3 =-(RDH-0.8D0)*  (XA(3)-XA(1))         &
             /(( RDA-1.6D0)**2*RDA)
                                                                
     DRHODY1 =(-YA(2)+YA(1))/(RDH*( RDA-1.6))            &
             -((RDH-0.8D0)*  (-YA(3)+YA(1)))       &
             /(( RDA-1.6D0)**2*RDA)                       
     DRHODY2 =(YA(2)-YA(1))/( RDH*(RDA-1.6D0))            
     DRHODY3 =-(RDH-0.8D0)*  (YA(3)-YA(1))         &
             /(( RDA-1.6D0)**2*RDA)
                                                                
     DRHODZ1 =(-ZA(2)+ZA(1))/(RDH*( RDA-1.6))            &
             -((RDH-0.8D0)*  (-ZA(3)+ZA(1)))       &
             /(( RDA-1.6D0)**2*RDA)                       
     DRHODZ2 =(ZA(2)-ZA(1))/( RDH*(RDA-1.6D0))            
     DRHODZ3 =-(RDH-0.8D0)*  (ZA(3)-ZA(1))         &
             /(( RDA-1.6D0)**2*RDA)


!      RESET RDA
    
      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)

!      CALCULATE PARTIAL DERIVATIVES 

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA
 
      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA


!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E. 
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE 
!      D(THETA)/D(U)*D(U)/D(X1)
      

      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/(DSQRT(1.D0-COSTHETA**2))*180.D0/PI
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)
                                                                        
     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)
                                                                        
     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1 
      DTHETADX2=DTDCOST*DCOSTDX2 
      DTHETADX3=DTDCOST*DCOSTDX3 

      DTHETADY1=DTDCOST*DCOSTDY1 
      DTHETADY2=DTDCOST*DCOSTDY2 
      DTHETADY3=DTDCOST*DCOSTDY3 

      DTHETADZ1=DTDCOST*DCOSTDZ1 
      DTHETADZ2=DTDCOST*DCOSTDZ2 
      DTHETADZ3=DTDCOST*DCOSTDZ3 


!      CALCULATE FORCES

      DX1 = DV0DRHO*DRHODX1 + DV0DRDA*DRDADX1 + DHRDTHET*DTHETADX1
      DX2 = DV0DRHO*DRHODX2 + DV0DRDA*DRDADX2 + DHRDTHET*DTHETADX2
      DX3 = DV0DRHO*DRHODX3 + DV0DRDA*DRDADX3 + DHRDTHET*DTHETADX3

      DY1 = DV0DRHO*DRHODY1 + DV0DRDA*DRDADY1 + DHRDTHET*DTHETADY1
      DY2 = DV0DRHO*DRHODY2 + DV0DRDA*DRDADY2 + DHRDTHET*DTHETADY2
      DY3 = DV0DRHO*DRHODY3 + DV0DRDA*DRDADY3 + DHRDTHET*DTHETADY3

      DZ1 = DV0DRHO*DRHODZ1 + DV0DRDA*DRDADZ1 + DHRDTHET*DTHETADZ1
      DZ2 = DV0DRHO*DRHODZ2 + DV0DRDA*DRDADZ2 + DHRDTHET*DTHETADZ2
      DZ3 = DV0DRHO*DRHODZ3 + DV0DRDA*DRDADZ3 + DHRDTHET*DTHETADZ3

          
      IF(PRNLEV.GT.7) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT ENERGY: ', VTOT   
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',AA,DX3,DY3,DZ3
 
      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2      
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EPTNHO
!      _________________________________________________________________________

  SUBROUTINE EPTOHOLE(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)
   
!     NEW PES FORMAT: THE LEGENDRE EXPANSION
!     SEPT. 2007

    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
!    use deriv
    use param
!    use energym
!    use coord
               

!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY 
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM 
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1] 
      real(chm_real) RDA, RDA2, RDH, RHO,R1,R2
!      XA: LOCAL X COORDINATES 
!      YA: LOCAL Y COORDINATES 
!      ZA: LOCAL Z COORDINATES 
      real(chm_real) XA(3), YA(3), ZA(3)
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX)
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
!      VTOT: TOTAL MMPT ENERGY 


      real(chm_real) VTOT, V0
      INTEGER L
      PARAMETER (L=10)
      real(chm_real) V(L)
      real(chm_real) FX(3), FY(3), FZ(3) 

      real(chm_real) THETA, COSTHETA, SINTHETA
!      PARTIAL DERIVATIVES 
      real(chm_real) DVDRDA, DVDRHO, DVDTHET, DTDCOST
      real(chm_real) DV0DRDA, DV0DRHO
      real(chm_real) DVLDRDA(L), DVLDRHO(L)

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,    &
            DRHODY1, DRHODY2, DRHODY3,     &
            DRHODZ1, DRHODZ2, DRHODZ3
                                            
     real(chm_real) DRDADX1, DRDADX2, DRDADX3,     &
            DRDADY1, DRDADY2, DRDADY3,     &
            DRDADZ1, DRDADZ2, DRDADZ3
                                            
     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,  &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,  &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3 

                                            
     real(chm_real) DX1, DX2, DX3,                 &
            DY1, DY2, DY3,                 &
            DZ1, DZ2, DZ3    
       
!      PARAMETER
      real(chm_real) PRMA0, PRMA1, PRMA2, PRMA3, PRMA4
      real(chm_real) PRMB0(L), PRMB1(L), PRMB2(L)
!      AND THEIR DERIVATIVE TO RDA
      real(chm_real) DPRMA0, DPRMA1, DPRMA2, DPRMA3, DPRMA4
      real(chm_real) PRMA5, DPRMA5
      real(chm_real) DPRMB0(L), DPRMB1(L), DPRMB2(L) 

!     LOOP INDEX FOR CALCULATING POTENTIAL
      INTEGER IL

      real(chm_real) DOTPROD

!     LEGENDRE POLYNOMIALS
      real(chm_real) P, DP
      DIMENSION P(0:L)
      DIMENSION DP(0:L)




!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) = X(DA)
      YA(1) = Y(DA)
      ZA(1) = Z(DA)

      XA(2) = X(HA)
      YA(2) = Y(HA)
      ZA(2) = Z(HA)

      XA(3) = X(AA)
      YA(3) = Y(AA)
      ZA(3) = Z(AA)



!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A 

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
             +(YA(3)-YA(1))*(YA(2)-YA(1))    &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))
     
      COSTHETA = DOTPROD/(RDA*RDH)
!      IF
!      SINTHETA = DSQRT(1.D0-COSTHETA**2) 

!      THETA = DACOS(COSTHETA)

!           

!      CONVERSION RAD -> DEG

!      THETA = THETA*RADDEG






!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]
      
      RHO = (RDH - 0.8D0) / (RDA - 1.6D0)


      IF(PRNLEV.GT.6) THEN
      write(outu,*) 'entering OHO angular dependent mmpt routine'
      WRITE(OUTU,*) 'MMPT>   RDA', RDA
      WRITE(OUTU,*) 'MMPT>   RDH', RDH
      WRITE(OUTU,*) 'MMPT>   RHO', RHO
      WRITE(OUTU,*) 'MMPT>   COS(THETA)', COSTHETA
      WRITE(OUTU,*) 'MMPT>   THETA[RAD]', THETA*PI/180.D0
!      WRITE(OUTU,*) 'MMPT>   SIN(THETA)', SINTHETA 
!      WRITE(OUTU,*) 'MMPT>   RRM', PRMOHO(1)
      ENDIF



!      POTENTIAL FUNCTION

!     ZEROTH ORDER
!      INITIALIZE PARAMETER
!      AND DERIVE PARAMETER FUNCTIONS P(RDA):
      PRMA0=PRMLPE(1)*(TANH(PRMLPE(2)*(RDA-PRMLPE(3)))+PRMLPE(4))
      PRMA1=PRMLPE(5)*(TANH(PRMLPE(6)*(RDA-PRMLPE(7)))+PRMLPE(8))
      PRMA2=PRMLPE(9)*(TANH(PRMLPE(10)*(RDA-PRMLPE(11)))+PRMLPE(12))
      PRMA3=PRMLPE(13)*(TANH(PRMLPE(14)*(RDA-PRMLPE(15)))+PRMLPE(16))
      PRMA4=PRMLPE(17)*(TANH(PRMLPE(18)*(RDA-PRMLPE(19)))+PRMLPE(20))
      PRMA5=PRMLPE(21)*(TANH(PRMLPE(22)*(RDA-PRMLPE(23)))+PRMLPE(24))

      DPRMA0=PRMLPE(1)*PRMLPE(2)                             &
            *(1.D0-(TANH(PRMLPE(2)*(RDA-PRMLPE(3))))**2)      
     DPRMA1=PRMLPE(5)*PRMLPE(6)                              &
            *(1.D0-(TANH(PRMLPE(6)*(RDA-PRMLPE(7))))**2)      
     DPRMA2=PRMLPE(9)*PRMLPE(10)                             &
            *(1.D0-(TANH(PRMLPE(10)*(RDA-PRMLPE(11))))**2)    
     DPRMA3=PRMLPE(13)*PRMLPE(14)                            &
            *(1.D0-(TANH(PRMLPE(14)*(RDA-PRMLPE(15))))**2)    
     DPRMA4=PRMLPE(17)*PRMLPE(18)                            &
            *(1.D0-(TANH(PRMLPE(18)*(RDA-PRMLPE(19))))**2)    
     DPRMA5=PRMLPE(21)*PRMLPE(22)                            &
            *(1.D0-(TANH(PRMLPE(22)*(RDA-PRMLPE(23))))**2)



      V0 = PRMA0*(1.D0-DEXP(-PRMA1*(RHO-PRMA2)))**2             &
         +PRMA0*(1.D0-DEXP(-PRMA1*(1.D0-RHO-PRMA2)))**2         &
         -1.D0*PRMA5                                            &
         +PRMA3*DEXP(-PRMA4*(RHO-0.5D0)**2)
                                                                 
     DV0DRDA = DPRMA0*(1.D0-DEXP(-PRMA1*(RHO-PRMA2)))**2        &
              -2.D0*PRMA0*(1.D0-DEXP(-PRMA1*(RHO-PRMA2)))       &
              *(-DPRMA1*(RHO-PRMA2)+PRMA1*DPRMA2)               &
              *DEXP(-PRMA1*(RHO-PRMA2))                         &
              +DPRMA0*(1.D0-DEXP(-PRMA1*(1.D0-RHO-PRMA2)))**2   &
              -2.D0*PRMA0*(1.D0-DEXP(-PRMA1*(1.D0-RHO-PRMA2)))  &
              *(-DPRMA1*(1.D0-RHO-PRMA2)+PRMA1*DPRMA2)          &
              *DEXP(-PRMA1*(1.D0-RHO-PRMA2))                    &
              -1.D0*DPRMA5                                      &
              +DPRMA3*DEXP(-PRMA4*(RHO-0.5D0)**2)               &
              -1.D0*PRMA3*DPRMA4                                &
              *DEXP(-PRMA4*(RHO-0.5D0)**2)*(RHO-0.5D0)**2

      DV0DRHO = 2.D0*PRMA0*(1.D0-DEXP(-PRMA1*(RHO-PRMA2)))      &
                *PRMA1*DEXP(-PRMA1*(RHO-PRMA2))                 &
               -2.D0*PRMA0*(1.D0-DEXP(-PRMA1*(1.D0-RHO-PRMA2))) &
                *PRMA1*DEXP(-PRMA1*(1.D0-RHO-PRMA2))            &
               -2.D0*PRMA3*PRMA4*(RHO-0.5D0)                    &
                *DEXP(-PRMA4*(RHO-0.5D0)**2)

!      V0=0.D0
!      DV0DRDA = 0.D0
!      DV0DRHO =0.D0

!      HIGHER ORDER

      DO 36 IL=1, L
        PRMB0(IL)=PRMLPE(9*IL+16)
        PRMB1(IL)=PRMLPE(9*IL+17)                                    &
                 *(TANH(PRMLPE(9*IL+18)*(RDA-PRMLPE(9*IL+19)))       &
                   +PRMLPE(9*IL+20))                                  
       PRMB2(IL)=PRMLPE(9*IL+21)+                                    &
                 PRMLPE(9*IL+22)*(RDA-PRMLPE(9*IL+23))**2+           &
                 PRMLPE(9*IL+24)*(RDA-PRMLPE(9*IL+23))**4

                                                                      
       DPRMB1(IL)=PRMLPE(9*IL+17)*PRMLPE(9*IL+18)*                   &
            (1.D0-(TANH(PRMLPE(9*IL+18)*(RDA-PRMLPE(9*IL+19))))**2)   
       DPRMB2(IL)=2.D0*PRMLPE(9*IL+22)*(RDA-PRMLPE(9*IL+23))+        &
               4.0D0*PRMLPE(9*IL+24)*(RDA-PRMLPE(9*IL+23))**3

        V(IL)=0.D0
        DVLDRDA(IL)=0.D0
        DVLDRHO(IL)=0.D0
        IF ((IL.EQ.1).OR.(IL.EQ.3)) THEN
!        Here we directly skip terms that equil zero
        V(IL)=PRMB0(IL)                                              &
             +PRMB1(IL)/(PRMB2(IL)*((RHO-0.5D0)**2+PRMB1(IL)**2))
                                                                      
       DVLDRDA(IL)=DPRMB1(IL)*((RHO-0.5D0)**2-PRMB1(IL)**2)          &
                   /(PRMB2(IL)*((RHO-0.5D0)**2+PRMB1(IL)**2)**2)     &
                   -DPRMB2(IL)*PRMB1(IL)                             &
                   /(((RHO-0.5D0)**2+PRMB1(IL)**2)*PRMB2(IL)**2)
                                                                      
       DVLDRHO(IL)=-2.D0*PRMB1(IL)*(RHO-0.5D0)                       &
                   /(PRMB2(IL)*((RHO-0.5D0)**2+PRMB1(IL)**2)**2)
        ENDIF

 36   ENDDO

      CALL LGND(L, COSTHETA, P, DP)

!     SUM ALL THE TERMS

      VTOT = V0+0.D0
      DVDRDA = DV0DRDA
      DVDRHO = DV0DRHO
      DVDTHET= 0.D0



      DO 37 IL=1, L
        VTOT = VTOT + V(IL)*P(IL)
        DVDRDA = DVDRDA + DVLDRDA(IL)*P(IL)
        DVDRHO = DVDRHO + DVLDRHO(IL)*P(IL)
!       HERE WE ACCTUALLY CALCULATE d(V)/d(cos(theta)) 
        DVDTHET = DVDTHET + DP(IL)*V(IL)
 37   ENDDO

      IF(PRNLEV.GT.7) THEN
      write(outu,*) 'Print every part of potential'
      WRITE(OUTU,*) 'MMPT>   V0', V0
      WRITE(OUTU,*) 'MMPT>   V1', V(1)
      WRITE(OUTU,*) 'MMPT>   V3', V(3)
      WRITE(OUTU,*) 'MMPT>   V', V(2), V(8), V(9), V(10)
      WRITE(OUTU,*) 'MMPT>   V', V(4), V(5), V(6), V(7)
      WRITE(OUTU,*) 'MMPT>   P1', P(1)
      WRITE(OUTU,*) 'MMPT>   P3', P(3)
      WRITE(OUTU,*) 'MMPT>   DP1', DP(1)
      WRITE(OUTU,*) 'MMPT>   DP3', DP(3)
      WRITE(OUTU,*) 'MMPT>   DVDRDA', DVDRDA
      WRITE(OUTU,*) 'MMPT>   DVDRHO', DVDRHO
      WRITE(OUTU,*) 'MMPT>   DVDTHET', DVDTHET
      ENDIF

!      POTENTIAL ENERGY SCALING 
!     AND ALSO ADD MINUS SIGN
      VTOT= -1.D0*SCLOHO*VTOT    
      DVDRDA=-1.D0*SCLOHO*DVDRDA    
      DVDRHO=-1.D0*SCLOHO*DVDRHO   
      DVDTHET=-1.D0*SCLOHO*DVDTHET

!      ADD TO THE TOTAL ENERGY
      


      EU = EU + VTOT



!      CALCULATE PARTIAL DERIVATIVES
                                                           
      DRHODX1 =(-XA(2)+XA(1))/(RDH*( RDA-1.6))       &
             -((RDH-0.8D0)*  (-XA(3)+XA(1)))   &
             /(( RDA-1.6D0)**2*RDA)                   
     DRHODX2 =(XA(2)-XA(1))/( RDH*(RDA-1.6D0))        
     DRHODX3 =-(RDH-0.8D0)*  (XA(3)-XA(1))     &
             /(( RDA-1.6D0)**2*RDA)
                                                            
     DRHODY1 =(-YA(2)+YA(1))/(RDH*( RDA-1.6))        &
             -((RDH-0.8D0)*  (-YA(3)+YA(1)))   &
             /(( RDA-1.6D0)**2*RDA)                   
     DRHODY2 =(YA(2)-YA(1))/( RDH*(RDA-1.6D0))        
     DRHODY3 =-(RDH-0.8D0)*  (YA(3)-YA(1))     &
             /(( RDA-1.6D0)**2*RDA)
                                                            
     DRHODZ1 =(-ZA(2)+ZA(1))/(RDH*( RDA-1.6))        &
             -((RDH-0.8D0)*  (-ZA(3)+ZA(1)))   &
             /(( RDA-1.6D0)**2*RDA)                   
     DRHODZ2 =(ZA(2)-ZA(1))/( RDH*(RDA-1.6D0))        
     DRHODZ3 =-(RDH-0.8D0)*  (ZA(3)-ZA(1))     &
             /(( RDA-1.6D0)**2*RDA)


!      RESET RDA
    
      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)

!      CALCULATE PARTIAL DERIVATIVES 

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA
 
      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA








      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                      &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)     
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)
                                                                       
     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)     
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)
                                                                       
     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)    &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)     
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                   &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)




!      CALCULATE FORCES

      DX1 = DVDRHO*DRHODX1 + DVDRDA*DRDADX1 + DVDTHET*DCOSTDX1
      DX2 = DVDRHO*DRHODX2 + DVDRDA*DRDADX2 + DVDTHET*DCOSTDX2
      DX3 = DVDRHO*DRHODX3 + DVDRDA*DRDADX3 + DVDTHET*DCOSTDX3

      DY1 = DVDRHO*DRHODY1 + DVDRDA*DRDADY1 + DVDTHET*DCOSTDY1
      DY2 = DVDRHO*DRHODY2 + DVDRDA*DRDADY2 + DVDTHET*DCOSTDY2
      DY3 = DVDRHO*DRHODY3 + DVDRDA*DRDADY3 + DVDTHET*DCOSTDY3

      DZ1 = DVDRHO*DRHODZ1 + DVDRDA*DRDADZ1 + DVDTHET*DCOSTDZ1
      DZ2 = DVDRHO*DRHODZ2 + DVDRDA*DRDADZ2 + DVDTHET*DCOSTDZ2
      DZ3 = DVDRHO*DRHODZ3 + DVDRDA*DRDADZ3 + DVDTHET*DCOSTDZ3

          
      IF(PRNLEV.GT.6) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT ENERGY: ', VTOT   
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',AA,DX3,DY3,DZ3
      WRITE(OUTU,*) 'MMPT> D DX1',DRHODX1,DRDADX1
      WRITE(OUTU,*) 'MMPT> D DX2',DRHODX2,DRDADX2
      WRITE(OUTU,*) 'MMPT> D DX3',DRHODX3,DRDADX3
      WRITE(OUTU,*) 'MMPT> D DY1',DRHODY1,DRDADY1
      WRITE(OUTU,*) 'MMPT> D DY2',DRHODY2,DRDADY2
      WRITE(OUTU,*) 'MMPT> D DY3',DRHODY3,DRDADY3
      WRITE(OUTU,*) 'MMPT> D DZ1',DRHODZ1,DRDADZ1
      WRITE(OUTU,*) 'MMPT> D DZ2',DRHODZ2,DRDADZ2
      WRITE(OUTU,*) 'MMPT> D DZ3',DRHODZ3,DRDADZ3
      ENDIF
      IF(PRNLEV.GT.6) THEN
      WRITE(OUTU,*) 'end>   V0, VTOT', V0, VTOT
      ENDIF
      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2      
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EPTOHOLE

! _________________________________________________________________________

  SUBROUTINE EPTNL(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)


    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
    use param
!    use energym
!    use deriv
!    use coord
               

!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY 
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM 
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1] 
      real(chm_real) RDA, RDA2, RDH, RHO
!      XA: LOCAL X COORDINATES 
!      YA: LOCAL Y COORDINATES 
!      ZA: LOCAL Z COORDINATES 
      real(chm_real) XA(3), YA(3), ZA(3)
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX)
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX) 
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3) ,dhrdrda,dhrdrho,dhrdxyz,optts,rah1

      real(chm_real) THETA, COSTHETA, sintheta ,kmin,kts,dmin,dts,dvar
!      PARTIAL DERIVATIVES 
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST, kfin, dfin

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,          &
            DRHODY1, DRHODY2, DRHODY3,           &
            DRHODZ1, DRHODZ2, DRHODZ3
                                                  
     real(chm_real) DRDADX1, DRDADX2, DRDADX3,           &
            DRDADY1, DRDADY2, DRDADY3,           &
            DRDADZ1, DRDADZ2, DRDADZ3
                                                  
     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,        &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,        &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3 
                                                  
     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,     &
            DTHETADY1, DTHETADY2, DTHETADY3,     &
            DTHETADZ1, DTHETADZ2, DTHETADZ3 

                                                  
     real(chm_real) DX1, DX2, DX3,                       &
            DY1, DY2, DY3,                       &
            DZ1, DZ2, DZ3    
       
!      PARAMETER

      real(chm_real) DEQ, B, RE, H1, DDEQ, DB, DRE, DH1


      real(chm_real) DOTPROD




!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) = X(DA)
      YA(1) = Y(DA)
      ZA(1) = Z(DA)

      XA(2) = X(HA)
      YA(2) = Y(HA)
      ZA(2) = Z(HA)

      XA(3) = X(AA)
      YA(3) = Y(AA)
      ZA(3) = Z(AA)



!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A 

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
             +(YA(3)-YA(1))*(YA(2)-YA(1))    &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))
     
      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)
      sintheta=DSQRT(1.D0-COSTHETA**2)
           

!      CONVERSION RAD -> DEG

      THETA = THETA*RADDEG
      dvar=RDH*sintheta






!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]
      
      RHO = (RDH*costheta - 0.8D0) / (RDA - 1.6D0)



      IF(PRNLEV.GT.7) THEN
      write(outu,*) 'entering NLM mmpt routine'
      WRITE(OUTU,*) 'MMPT>   RDA', RDA
      WRITE(OUTU,*) 'MMPT>   RDH', RDH
      WRITE(OUTU,*) 'MMPT>   RHO', RHO
      WRITE(OUTU,*) 'MMPT>   COS(THETA)', COSTHETA
!      WRITE(OUTU,*) 'MMPT>   THETA[RAD]', THETA*PI/180.D0
      WRITE(OUTU,*) 'MMPT>   THETA', THETA 
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER:
      DEQ=PRMNLM(1)*(1-DEXP(-PRMNLM(2)*(RDA-PRMNLM(3))))**2+PRMNLM(4)
      B =PRMNLM(5)+PRMNLM(6)*RDA
      RE=PRMNLM(7)*DEXP(-PRMNLM(8)*RDA)+PRMNLM(9)
      H1=PRMNLM(10) 
      kmin= PRMNLM(10)
      kts=PRMNLM(12)
      dmin=PRMNLM(13)
      dts=PRMNLM(14) 



!      DERIVE PARAMETER FUNCTIONS P(RDA)
      DDEQ= 2*PRMNLM(1)*(1.D0-DEXP(-PRMNLM(2)*(RDA-PRMNLM(3))))  &
            *PRMNLM(2)*DEXP(-PRMNLM(2)*(RDA-PRMNLM(3)))
      DB = PRMNLM(6)
      DRE= -PRMNLM(7)*PRMNLM(8)*DEXP(-PRMNLM(8)*RDA)
      DH1=0.D0
      

!      POTENTIAL FUNCTION (DOUBLE MORSE POTENTIAL)

      V0 = DEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                            &
         +DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                        &
         -DEQ+PRMNLM(11)
                                                                       
     DV0DRDA = DDEQ*(1.D0-DEXP(-B*(RHO-RE)))**2                       &
              -2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                      &
              *(-DB*(RHO-RE)+B*DRE)*DEXP(-B*(RHO-RE))                 &
              +DDEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))**2                  &
              -2.D0*DEQ*(1.D0-DEXP(-B*(1.D0-RHO-RE)))                 &
              *(-DB*(1.D0-RHO-RE)+B*DRE)*DEXP(-B*(1.D0-RHO-RE))       &
              -DDEQ


                                                                       
     DV0DRHO = 2.D0*DEQ*(1.D0-DEXP(-B*(RHO-RE)))                      &
               *B*DEXP(-B*(RHO-RE))                                   &
               -2.D0*DEQ*(1.D0-DEXP(-B*(1-RHO-RE)))                   &
               *B*DEXP(-B*(1-RHO-RE))


!      HARMONIC RESTRAINT:

        kfin=kts+kmin*V0
        dfin=dts+dmin*(rho-0.5d0)**2
      HR = 0.5d0*kfin*(dvar-dfin)**2

      DHRDTHET = kfin*(dvar-dfin)*RDH*costheta

!      POTENTIAL ENERGY SCALING

      V0=SCLOHO*V0    
      DV0DRDA=SCLOHO*DV0DRDA    
      DV0DRHO=SCLOHO*DV0DRHO   

!      SUM TERMS
      
      VTOT = V0 + HR 

      EU = EU + VTOT

! ... RESET RDA
    
      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)

!      CALCULATE PARTIAL DERIVATIVES
                                                           
      DRHODX1 =costheta*(-XA(2)+XA(1))/(RDH*(RDA-1.6))       &
             -((costheta*RDH-0.8D0)*(-XA(3)+XA(1)))          &
             /((RDA-1.6D0)**2*RDA)                            
     DRHODX2 =costheta*(XA(2)-XA(1))/(RDH*(RDA-1.6D0))        
     DRHODX3 =-(costheta*RDH-0.8D0)*(XA(3)-XA(1))            &
             /((RDA-1.6D0)**2*RDA)
                                                              
     DRHODY1 =costheta*(-YA(2)+YA(1))/(RDH*(RDA-1.6))        &
             -((costheta*RDH-0.8D0)*(-YA(3)+YA(1)))          &
             /((RDA-1.6D0)**2*RDA)                            
     DRHODY2 =costheta*(YA(2)-YA(1))/(RDH*(RDA-1.6D0))        
     DRHODY3 =-(costheta*RDH-0.8D0)*(YA(3)-YA(1))            &
             /((RDA-1.6D0)**2*RDA)
                                                              
     DRHODZ1 =costheta*(-ZA(2)+ZA(1))/(RDH*(RDA-1.6))        &
             -((costheta*RDH-0.8D0)*(-ZA(3)+ZA(1)))          &
             /((RDA-1.6D0)**2*RDA)                            
     DRHODZ2 =costheta*(ZA(2)-ZA(1))/(RDH*(RDA-1.6D0))        
     DRHODZ3 =-(costheta*RDH-0.8D0)*(ZA(3)-ZA(1))            &
             /((RDA-1.6D0)**2*RDA)



!      CALCULATE PARTIAL DERIVATIVES 

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA
 
      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA






!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E. 
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE 
!      D(THETA)/D(U)*D(U)/D(X1)
      

      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/(DSQRT(1.D0-COSTHETA**2))
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)        
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)
                                                                         
     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                         &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)        
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)
                                                                         
     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                         &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)      &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)        
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                     &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1 
      DTHETADX2=DTDCOST*DCOSTDX2 
      DTHETADX3=DTDCOST*DCOSTDX3 

      DTHETADY1=DTDCOST*DCOSTDY1 
      DTHETADY2=DTDCOST*DCOSTDY2 
      DTHETADY3=DTDCOST*DCOSTDY3 

      DTHETADZ1=DTDCOST*DCOSTDZ1 
      DTHETADZ2=DTDCOST*DCOSTDZ2 
      DTHETADZ3=DTDCOST*DCOSTDZ3 

!    plus deriv of DthetaDx
      DRHODx1=DRHODx1+DCOSTDx1*RDH/(RDA-1.6d0)
      DRHODy1=DRHODy1+DCOSTDy1*RDH/(RDA-1.6d0)
      DRHODz1=DRHODz1+DCOSTDz1*RDH/(RDA-1.6d0)
      DRHODx2=DRHODx2+DCOSTDx2*RDH/(RDA-1.6d0)
      DRHODy2=DRHODy2+DCOSTDy2*RDH/(RDA-1.6d0)
      DRHODz2=DRHODz2+DCOSTDz2*RDH/(RDA-1.6d0)
      DRHODx3=DRHODx3+DCOSTDx3*RDH/(RDA-1.6d0)
      DRHODy3=DRHODy3+DCOSTDy3*RDH/(RDA-1.6d0)
      DRHODz3=DRHODz3+DCOSTDz3*RDH/(RDA-1.6d0)

!    new deriv of RDH and RHO
        dhrdrda=0.5d0*kmin*(dvar-dfin)**2*DV0DRDA
        dhrdrho=0.5d0*kmin*(dvar-dfin)**2*DV0Drho-kfin*(dvar-dfin)*2d0*dmin*(rho-0.5d0)
        dhrdxyz=kfin*(dvar-dfin)*sintheta/RDH

!      CALCULATE FORCES

      DX1 = (DV0DRHO+dhrdrho)*DRHODX1 + (DV0DRDA+dhrdrda)*DRDADX1   &
      + DHRDTHET*DTHETADX1 +dhrdxyz*(-XA(2)+XA(1))                   
     DX2 = (DV0DRHO+dhrdrho)*DRHODX2 + (DV0DRDA+dhrdrda)*DRDADX2    &
      + DHRDTHET*DTHETADX2 +dhrdxyz*(XA(2)-XA(1))                    
     DX3 = (DV0DRHO+dhrdrho)*DRHODX3 + (DV0DRDA+dhrdrda)*DRDADX3    &
      + DHRDTHET*DTHETADX3 
                                                                     
     Dy1 = (DV0DRHO+dhrdrho)*DRHODy1 + (DV0DRDA+dhrdrda)*DRDADy1    &
      + DHRDTHET*DTHETADy1 +dhrdxyz*(-yA(2)+yA(1))                   
     Dy2 = (DV0DRHO+dhrdrho)*DRHODy2 + (DV0DRDA+dhrdrda)*DRDADy2    &
      + DHRDTHET*DTHETADy2 +dhrdxyz*(yA(2)-yA(1))                    
     Dy3 = (DV0DRHO+dhrdrho)*DRHODy3 + (DV0DRDA+dhrdrda)*DRDADy3    &
      + DHRDTHET*DTHETADy3 
                                                                     
     Dz1 = (DV0DRHO+dhrdrho)*DRHODz1 + (DV0DRDA+dhrdrda)*DRDADz1    &
      + DHRDTHET*DTHETADz1 +dhrdxyz*(-zA(2)+zA(1))                   
     Dz2 = (DV0DRHO+dhrdrho)*DRHODz2 + (DV0DRDA+dhrdrda)*DRDADz2    &
      + DHRDTHET*DTHETADz2 +dhrdxyz*(zA(2)-zA(1))                    
     Dz3 = (DV0DRHO+dhrdrho)*DRHODz3 + (DV0DRDA+dhrdrda)*DRDADz3    &
      + DHRDTHET*DTHETADz3 

          
      IF(PRNLEV.GT.7) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT ENERGY: ', VTOT   
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',AA,DX3,DY3,DZ3
 
      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2      
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EPTNL


  SUBROUTINE EPTANL(DA,HA,AA,EU,X,Y,Z,DX,DY,DZ,NATOMX)
   
    use dimens_fcm
    use number
    use stream
    use consta
    use inbnd
    use psf
!    use deriv
    use param
!    use energym
!    use coord

!      VARIABLE DECLARATIONS
!      EU: GLOBAL USER ENERGY 
      real(chm_real) EU
!      DA: PSF NUMBER OF DONOR ATOM 
!      HA: PSF NUMBER OF HYDROGEN ATOM
!      AA: PSF NUMBER OF ACCEPTOR ATOM
      INTEGER DA, HA, AA, NATOMX
!      RDA: DISTANCE BETWEEN DONOR ATOM D AND ACCEPTOR ATOM A
!      RDH: DISTANCE BETWEEN DONOR ATOM D AND HYDROGEN ATOM A
!      RDA2: RDA SQUARED
!      RHO: RDH MAPPED ON INTERVAL [0,1] 
      real(chm_real) RDA, RDA2, RDH, RHO
!      XA: LOCAL X COORDINATES 
!      YA: LOCAL Y COORDINATES 
!      ZA: LOCAL Z COORDINATES 
      real(chm_real) XA(3), YA(3), ZA(3)
      REAL(CHM_REAL) X(NATOMX),Y(NATOMX),Z(NATOMX)
      REAL(CHM_REAL) DX(NATOMX),DY(NATOMX),DZ(NATOMX) 
!      VTOT: TOTAL MMPT ENERGY
!      V0: ZEROTH ORDER TERM OF LEGENDRE EXPANSION
!      HR: HARMONIC RESTRAINT TO DESCRIBE THETA DEPENDENCE
      real(chm_real) VTOT, V0, HR
      real(chm_real) FX(3), FY(3), FZ(3) ,dhrdrda,dhrdrho,dhrdxyz,optts,rah1

      real(chm_real) THETA, COSTHETA, sintheta ,kmin,kts,dmin,dts,dvar,rhots
!      PARTIAL DERIVATIVES 
      real(chm_real) DV0DRDA, DV0DRHO, DHRDTHET, DTDCOST, kfin, dfin

      real(chm_real) DRHODX1, DRHODX2, DRHODX3,       &
            DRHODY1, DRHODY2, DRHODY3,        &
            DRHODZ1, DRHODZ2, DRHODZ3
                                               
     real(chm_real) DRDADX1, DRDADX2, DRDADX3,        &
            DRDADY1, DRDADY2, DRDADY3,        &
            DRDADZ1, DRDADZ2, DRDADZ3
                                               
     real(chm_real) DCOSTDX1, DCOSTDX2, DCOSTDX3,     &
            DCOSTDY1, DCOSTDY2, DCOSTDY3,     &
            DCOSTDZ1, DCOSTDZ2, DCOSTDZ3 
                                               
     real(chm_real) DTHETADX1, DTHETADX2, DTHETADX3,  &
            DTHETADY1, DTHETADY2, DTHETADY3,  &
            DTHETADZ1, DTHETADZ2, DTHETADZ3 


     real(chm_real) DX1, DX2, DX3,                   &
            DY1, DY2, DY3,                            &
            DZ1, DZ2, DZ3                                            
                                                                     
!     PARAMETER
                                                                     
     real(chm_real) DEQ1, DEQ2, BETA1, BETA2, REQ1, REQ2, C, T,    &
            DDEQ1, DDEQ2, DBETA1, DBETA2, DREQ1, DREQ2, DC, DT


      real(chm_real) DOTPROD


      integer i

!      PASS COORDINATES TO POTENTIAL ROUTINE

      XA(1) =X(DA)
      YA(1) =Y(DA)
      ZA(1) =Z(DA)

      XA(2) =X(HA)
      YA(2) =Y(HA)
      ZA(2) =Z(HA)

      XA(3) =X(AA)
      YA(3) =Y(AA)
      ZA(3) =Z(AA)



!      TRANSFORM FROM CARTESIAN TO INTERNAL COORDINATE DESCRIPTION:

!      CALCULATE THE ABSOLUTE DISTANCE OF D-A 

      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)


!      CALCULATE THE ABSOLUTE DISTANCE OF D-H
      RDH = DSQRT((XA(2)-XA(1))**2+(YA(2)-YA(1))**2+(ZA(2)-ZA(1))**2)



!      DOTPRODUCT OF RDA . RDH

      DOTPROD =(XA(3)-XA(1))*(XA(2)-XA(1))   &
             +(YA(3)-YA(1))*(YA(2)-YA(1))    &
             +(ZA(3)-ZA(1))*(ZA(2)-ZA(1))
     
      COSTHETA = DOTPROD/(RDA*RDH)

      THETA = DACOS(COSTHETA)
      SINTHETA=DSQRT(1.D0-COSTHETA**2)

           

!      CONVERSION RAD -> DEG

!      THETA = THETA*RADDEG
      DVAR=RDH*SINTHETA








!      V_LAMBDA IS A FUNCTION OF RDH AND RDA BUT THE V_LAMBDA ARE DEFINED
!     AS A FUNCTION OF RHO, WHICH IS DEFINED ON THE INTERVAL [0,1]
      
      RHO = (RDH*COSTHETA - 0.8D0) / (RDA - 1.6D0)



      IF(PRNLEV.GT.7) THEN
      write(outu,*) 'entering ANL mmpt routine'
      WRITE(OUTU,*) 'MMPT>   RDA', RDA
      WRITE(OUTU,*) 'MMPT>   RDH', RDH
      WRITE(OUTU,*) 'MMPT>   RHO', RHO
      WRITE(OUTU,*) 'MMPT>   COS(THETA)', COSTHETA
!      WRITE(OUTU,*) 'MMPT>   THETA[RAD]', THETA*PI/180.D0
      WRITE(OUTU,*) 'MMPT>   D', DVAR 
      ENDIF



!      POTENTIAL FUNCTION
!      INITIALIZE PARAMETER, PARAMETER FUNCTION ARE FUNCTIONS
!      OF RDA


      DEQ1 =PRMANL(1)*(1.D0-dexp(-PRMANL(2)*(RDA-PRMANL(3))))**2       &
          +PRMANL(4)

                                                                        
     BETA1=PRMANL(5)/(1.D0+DEXP(-PRMANL(6)*(RDA-PRMANL(7))))
                                                                        
!     write(outu,*) 'beta1', beta1                                      
!     write(outu,*) 'using', PRMANL(5),PRMANL(6),PRMANL(7)

                                                                        
     REQ1 =PRMANL(8)*(1.D0-dexp(-PRMANL(9)*(RDA-PRMANL(10))))**2       &
          +PRMANL(11)
                                                                        
     DEQ2 =PRMANL(12)*(1.D0-dexp(-PRMANL(13)*(RDA-PRMANL(14))))**2     &
          +PRMANL(15)
                                                                        
     BETA2=PRMANL(16)/(1.D0+DEXP(-PRMANL(17)*(RDA-PRMANL(18))))
                                                                        
     REQ2 =PRMANL(19)*(1.D0-dexp(-PRMANL(20)*(RDA-PRMANL(21))))**2     &
          +PRMANL(22)

      C    =PRMANL(23)*(1.D0-dexp(-PRMANL(24)*(RDA-PRMANL(25))))**2    &
          +PRMANL(26)
   
      KMIN=PRMANL(27)
      KTS=PRMANL(28)
      DMIN=PRMANL(29)
      DTS=PRMANL(30)
      RHOTS=PRMANL(31)
      
      IF(PRNLEV.GT.7) THEN 
      DO I=1, NPRMANL
         write(outu,*) 'MMPT>  PARAM', I, PRMANL(I)
      ENDDO
      ENDIF

!      write(outu,*) 'parameter functions return:'
!      write(outu,*) deq1,beta1,req1,deq2,beta2,req2,c,t


!      DERIVE PARAMETER FUNCTIONS P(RDA)


      ddeq1 = 2*PRMANL(1)*(1-exp(-PRMANL(2)*(rda-PRMANL(3))))*         &
               PRMANL(2)*exp(-PRMANL(2)*(rda-PRMANL(3)))
 
                                                                        
     dbeta1 =  PRMANL(5)*PRMANL(6)*exp(-PRMANL(6)*(rda-PRMANL(7)))/    &
               (1+exp(-PRMANL(6)*(rda-PRMANL(7))))**2

                                                                        
!     write(outu,*) 'dbeta1', dbeta1                                    
!     write(outu,*) 'using', PRMANL(5),PRMANL(6),PRMANL(7)


                                                                        
     dreq1 = 2*PRMANL(8)*(1-exp(-PRMANL(9)*(rda-PRMANL(10))))*         &
               PRMANL(9)*exp(-PRMANL(9)*(rda-PRMANL(10)))
                                                                        
     ddeq2 = 2*PRMANL(12)*(1-exp(-PRMANL(13)*(rda-PRMANL(14))))*       &
               PRMANL(13)*exp(-PRMANL(13)*(rda-PRMANL(14)))
                                                                        
     dbeta2 =  PRMANL(16)*PRMANL(17)*exp(-PRMANL(17)*(rda-PRMANL(18)))/&
               (1+exp(-PRMANL(17)*(rda-PRMANL(18))))**2
                                                                        
     dreq2 = 2*PRMANL(19)*(1-exp(-PRMANL(20)*(rda-PRMANL(21))))*       &
               PRMANL(20)*exp(-PRMANL(20)*(rda-PRMANL(21)))

                                                                        
     dc    = 2*PRMANL(23)*(1-exp(-PRMANL(24)*(rda-PRMANL(25))))*       &
               PRMANL(24)*exp(-PRMANL(24)*(rda-PRMANL(25)))
 
     

!      write(outu,*) 'derivative parameter functions return:'
!      write(outu,*) ddeq1,dbeta1,dreq1,ddeq2,dbeta2,dreq2,dc,dt



      

!      POTENTIAL FUNCTION (DOUBLE MORSE POTENTIAL)
! ... in contrast to eptoho or eptnhn we have two independent potentials

      v0 = Deq1*(1.d0-dexp(-beta1*(rho-Req1)))**2                      &
         +Deq2*(1.d0-dexp(-beta2*(Req2-rho)))**2                       &
         -c
 

                                                                        
     dv0drda = dDeq1*(1-exp(-beta1*(rho-Req1)))**2                     &
             -2*Deq1*(1-exp(-beta1*(rho-Req1)))*                       &
              (-dbeta1*(rho-Req1)+beta1*dReq1)*exp(-beta1*(rho-Req1))  &
              +dDeq2*(1-exp(-beta2*(Req2-rho)))**2                     &
             -2*Deq2*(1-exp(-beta2*(Req2-rho)))*                       &
              (-dbeta2*(Req2-rho)-beta2*dReq2)*exp(-beta2*(Req2-rho))  &
             -dc


!      write(outu,*) 'dv0drda', dv0drda


!      write(outu,*) dDeq1*(1-exp(-beta1*(rho-Req1)))**2
!      write(outu,*) 'using',ddeq1,beta1,rho,req1

!      write(outu,*) -2*Deq1*(1-exp(-beta1*(rho-Req1)))

       dv0drho = 2*Deq1*(1-exp(-beta1*(rho-Req1)))*  &
                beta1*exp(-beta1*(rho-Req1))         &
               -2*Deq2*(1-exp(-beta2*(Req2-rho)))*   &
                beta2*exp(-beta2*(Req2-rho))


!      write(outu,*) 'dv0drho', dv0drho



!      HARMONIC RESTRAINT:
      kfin=kts+kmin*V0
      dfin=dts+dmin*(rho-RHOTS)**2
      HR = 0.5d0*kfin*(dvar-dfin)**2
      DHRDTHET = kfin*(dvar-dfin)*RDH*costheta
!HR =  T*THETA**2
!
!DHRDTHET = 2.D0*T*THETA

!      POTENTIAL ENERGY SCALING

      V0=SCLOHO*V0    
      DV0DRDA=SCLOHO*DV0DRDA    
      DV0DRHO=SCLOHO*DV0DRHO   

!      SUM TERMS
      
      VTOT = V0 + HR 

      EU = EU + VTOT


! ... RESET RDA
    
      RDA = DSQRT((XA(3)-XA(1))**2+(YA(3)-YA(1))**2+(ZA(3)-ZA(1))**2)
!      CALCULATE PARTIAL DERIVATIVES
                                                           
      DRHODX1 =costheta*(-XA(2)+XA(1))/(RDH*(RDA-1.6))       &
             -((costheta*RDH-0.8D0)*(-XA(3)+XA(1)))          &
             /((RDA-1.6D0)**2*RDA)                            
     DRHODX2 =costheta*(XA(2)-XA(1))/(RDH*(RDA-1.6D0))        
     DRHODX3 =-(costheta*RDH-0.8D0)*(XA(3)-XA(1))            &
             /((RDA-1.6D0)**2*RDA)
                                                              
     DRHODY1 =costheta*(-YA(2)+YA(1))/(RDH*(RDA-1.6))        &
             -((costheta*RDH-0.8D0)*(-YA(3)+YA(1)))          &
             /((RDA-1.6D0)**2*RDA)                            
     DRHODY2 =costheta*(YA(2)-YA(1))/(RDH*(RDA-1.6D0))        
     DRHODY3 =-(costheta*RDH-0.8D0)*(YA(3)-YA(1))            &
             /((RDA-1.6D0)**2*RDA)
                                                              
     DRHODZ1 =costheta*(-ZA(2)+ZA(1))/(RDH*(RDA-1.6))        &
             -((costheta*RDH-0.8D0)*(-ZA(3)+ZA(1)))          &
             /((RDA-1.6D0)**2*RDA)                            
     DRHODZ2 =costheta*(ZA(2)-ZA(1))/(RDH*(RDA-1.6D0))        
     DRHODZ3 =-(costheta*RDH-0.8D0)*(ZA(3)-ZA(1))            &
             /((RDA-1.6D0)**2*RDA)



!      CALCULATE PARTIAL DERIVATIVES 

      DRDADX1 = (-XA(3)+XA(1))/RDA
      DRDADX2 = 0.D0
      DRDADX3 = (XA(3)-XA(1))/RDA

      DRDADY1 = (-YA(3)+YA(1))/RDA
      DRDADY2 = 0.D0
      DRDADY3 = (YA(3)-YA(1))/RDA
 
      DRDADZ1 = (-ZA(3)+ZA(1))/RDA
      DRDADZ2 = 0.D0
      DRDADZ3 = (ZA(3)-ZA(1))/RDA


!      PARTIAL DERIVATIVE WAS D(COS(THETA))/D(X1) BUT NOW IS D(THETA)/D(X1) I.E. 
!      D(ARCCOS(COS(THETA)))/D(X1)
!      USE CHAIN RULE AND SUBSTITUTE U = COS(THETA) THEN SOLVE 
!      D(THETA)/D(U)*D(U)/D(X1)
      

      IF(THETA.EQ.ZERO) THEN
         DTDCOST=ZERO
      ELSE
         DTDCOST= -1.D0/SINTHETA
      ENDIF



      DCOSTDX1=(-XA(2)+2.D0*XA(1)-XA(3))/RDA/RDH                       &
             -(DOTPROD*(-2.D0*XA(3)+2.D0*XA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*XA(2)+2.D0*XA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDX2=(XA(3)-XA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*XA(2)-2.D0*XA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDX3=(XA(2)-XA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*XA(3)-2.D0*XA(1)))/(2.D0*RDA**3*RDH)
                                                                        
     DCOSTDY1=(-YA(2)+2.D0*YA(1)-YA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*YA(3)+2.D0*YA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*YA(2)+2.D0*YA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDY2=(YA(3)-YA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*YA(2)-2.D0*YA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDY3=(YA(2)-YA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*YA(3)-2.D0*YA(1)))/(2.D0*RDA**3*RDH)
                                                                        
     DCOSTDZ1=(-ZA(2)+2.D0*ZA(1)-ZA(3))/RDA/RDH                        &
             -(DOTPROD*(-2.D0*ZA(3)+2.D0*ZA(1)))/(2.D0*RDA**3*RDH)     &
             -(DOTPROD*(-2.D0*ZA(2)+2.D0*ZA(1)))/(2.D0*RDA*RDH**3)      
     DCOSTDZ2=(ZA(3)-ZA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*ZA(2)-2.D0*ZA(1)))/(2.D0*RDA*RDH**3)       
     DCOSTDZ3=(ZA(2)-ZA(1))/RDA/RDH                                    &
             -(DOTPROD*(2.D0*ZA(3)-2.D0*ZA(1)))/(2.D0*RDA**3*RDH)

      DTHETADX1=DTDCOST*DCOSTDX1 
      DTHETADX2=DTDCOST*DCOSTDX2 
      DTHETADX3=DTDCOST*DCOSTDX3 

      DTHETADY1=DTDCOST*DCOSTDY1 
      DTHETADY2=DTDCOST*DCOSTDY2 
      DTHETADY3=DTDCOST*DCOSTDY3 

      DTHETADZ1=DTDCOST*DCOSTDZ1 
      DTHETADZ2=DTDCOST*DCOSTDZ2 
      DTHETADZ3=DTDCOST*DCOSTDZ3 
      
!    plus deriv of DthetaDx
      DRHODx1=DRHODx1+DCOSTDx1*RDH/(RDA-1.6d0)
      DRHODy1=DRHODy1+DCOSTDy1*RDH/(RDA-1.6d0)
      DRHODz1=DRHODz1+DCOSTDz1*RDH/(RDA-1.6d0)
      DRHODx2=DRHODx2+DCOSTDx2*RDH/(RDA-1.6d0)
      DRHODy2=DRHODy2+DCOSTDy2*RDH/(RDA-1.6d0)
      DRHODz2=DRHODz2+DCOSTDz2*RDH/(RDA-1.6d0)
      DRHODx3=DRHODx3+DCOSTDx3*RDH/(RDA-1.6d0)
      DRHODy3=DRHODy3+DCOSTDy3*RDH/(RDA-1.6d0)
      DRHODz3=DRHODz3+DCOSTDz3*RDH/(RDA-1.6d0)

!    new deriv of RDH and RHO
        dhrdrda=0.5d0*kmin*(dvar-dfin)**2*DV0DRDA
        dhrdrho=0.5d0*kmin*(dvar-dfin)**2*DV0Drho-kfin*(dvar-dfin)*2d0*dmin*(rho-RHOTS)
        dhrdxyz=kfin*(dvar-dfin)*sintheta/RDH

!      CALCULATE FORCES

      DX1 = (DV0DRHO+dhrdrho)*DRHODX1 + (DV0DRDA+dhrdrda)*DRDADX1   &
      + DHRDTHET*DTHETADX1 +dhrdxyz*(-XA(2)+XA(1))                   
     DX2 = (DV0DRHO+dhrdrho)*DRHODX2 + (DV0DRDA+dhrdrda)*DRDADX2    &
      + DHRDTHET*DTHETADX2 +dhrdxyz*(XA(2)-XA(1))                    
     DX3 = (DV0DRHO+dhrdrho)*DRHODX3 + (DV0DRDA+dhrdrda)*DRDADX3    &
      + DHRDTHET*DTHETADX3 
                                                                     
     Dy1 = (DV0DRHO+dhrdrho)*DRHODy1 + (DV0DRDA+dhrdrda)*DRDADy1    &
      + DHRDTHET*DTHETADy1 +dhrdxyz*(-yA(2)+yA(1))                   
     Dy2 = (DV0DRHO+dhrdrho)*DRHODy2 + (DV0DRDA+dhrdrda)*DRDADy2    &
      + DHRDTHET*DTHETADy2 +dhrdxyz*(yA(2)-yA(1))                    
     Dy3 = (DV0DRHO+dhrdrho)*DRHODy3 + (DV0DRDA+dhrdrda)*DRDADy3    &
      + DHRDTHET*DTHETADy3 
                                                                     
     Dz1 = (DV0DRHO+dhrdrho)*DRHODz1 + (DV0DRDA+dhrdrda)*DRDADz1    &
      + DHRDTHET*DTHETADz1 +dhrdxyz*(-zA(2)+zA(1))                   
     Dz2 = (DV0DRHO+dhrdrho)*DRHODz2 + (DV0DRDA+dhrdrda)*DRDADz2    &
      + DHRDTHET*DTHETADz2 +dhrdxyz*(zA(2)-zA(1))                    
     Dz3 = (DV0DRHO+dhrdrho)*DRHODz3 + (DV0DRDA+dhrdrda)*DRDADz3    &
      + DHRDTHET*DTHETADz3 

          
      IF(PRNLEV.GT.7) THEN

      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT ENERGY: ', VTOT   
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',DA,DX1,DY1,DZ1
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',HA,DX2,DY2,DZ2
      WRITE(OUTU,*) 'MMPT> PT FORCES ON',AA,DX3,DY3,DZ3
 
      ENDIF

      DX(DA)=DX(DA)+DX1
      DX(HA)=DX(HA)+DX2      
      DX(AA)=DX(AA)+DX3
      DY(DA)=DY(DA)+DY1
      DY(HA)=DY(HA)+DY2
      DY(AA)=DY(AA)+DY3
      DZ(DA)=DZ(DA)+DZ1
      DZ(HA)=DZ(HA)+DZ2
      DZ(AA)=DZ(AA)+DZ3

  END SUBROUTINE EPTANL

!      _________________________________________________________________________

!      LEGENDRE POLYNOMIALS 
  SUBROUTINE LGND(LNUM,X,P,DP)
!  SUBROUTINE TO GENERATE LEGENDRE POLYNOMIALS P_L(X)
!      FOR L = 0,1,...,LNUM WITH GIVEN X.
     
      real(chm_real) X, P, DP 
      DIMENSION P(0:LNUM)
      DIMENSION DP(0:LNUM)
      INTEGER L, LNUM
      
      P(0) = 1.D0
      P(1) = X
      DO L=1, LNUM-1
        P(L+1) = ((2.D0*DBLE(L)+1.D0)*X*P(L)-DBLE(L)*P(L-1))/DBLE(L+1)
      ENDDO

!      RECURSIVE RELATION FOR DERIVATIVES
    
      DP(0) = 0.D0
      DP(1) = 1.D0 

   
!      IF COS THETA = 1 THEN USE RESULTS DIRECTLY AND AVOID 
!      DIVISION 
      IF (X.EQ.1.D0) THEN
         DP(2) =  3.D0
         DP(3) =  6.D0
         DP(4) = 10.D0
         DP(5) = 15.D0
         DP(6) = 21.D0
         DP(7) = 28.D0
         DP(8) = 36.D0
         DP(9) = 45.D0
         DP(10)= 55.D0
         DP(11)= 66.D0
      ELSE

      DO L=2, LNUM
        DP(L) = DBLE(L)*(X*P(L)-P(L-1))/(X**2-1)
!        WRITE(22,*) L, P(L), P(L-1), X,(X**2-1), DP(L) 
      ENDDO
      ENDIF 
  
      RETURN 
  END SUBROUTINE LGND

  SUBROUTINE ALLOCFIR

    INTEGER NHBNUM, NHBNUM3, NHBNUM4, NHBNUM7

    NPRMOHO=11
    NPRMNHN=11
    NPRMNHO=27
    NPRMLPE=114
    NPRMNLM=14
    NPRMANL=31
    NHBNUM=200

    ALLOCATE(PRMNHN(NPRMNHN),       &
            PRMOHO(NPRMOHO),        &
            PRMNHO(NPRMNHO),        &
            PRMLPE(NPRMLPE),        &
            PRMNLM(NPRMNLM),        &
            PRMANL(NPRMANL),        &
            POTTYPE(NHBNUM),        &
            HBRDATOM(NHBNUM,3),      &
            BONDA(20),      &
            BONDB(20),      &
            BONDLV(20),      &
            BONDFC1(20),      &
            BONDFC2(20),      &
            BONDEQU1(20),      &
            BONDEQU2(20)      &
            )

    RETURN
  END SUBROUTINE ALLOCFIR

  SUBROUTINE ALLOCSEC(NHBNUM)

    INTEGER NHBNUM, NHBNUM3, NHBNUM4, NHBNUM7

    NHBNUM3=NHBNUM*6
    NHBNUM4=NHBNUM*18
    NHBNUM7=NHBNUM*30

    ALLOCATE(BONDMMPT(NHBNUM,3),     &
            MMPTCHRG(NHBNUM,3),      &
            INITCHRG(NHBNUM,3),      &
            ANGLMMPT(NHBNUM3,7),     &
            DIHEMMPT(NHBNUM4,6),     &
            IMPRMMPT(NHBNUM4,6),     &
            NONBMMPT(NHBNUM7,4) &
            )

    RETURN
  END SUBROUTINE ALLOCSEC

  SUBROUTINE EPSEUDOMMPT(I,J,K,EA,FCST,EQUIDEG,MODE,DXAI,DYAI,DZAI, &
  DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK)
!     THIS ROUTINE CALCULATES THE PSEUDO-ENERGY OF COUPLED D-H-A
! IF D-H-A>90, EA=0, d(EA)=0
! IF D-H-A<90, EA = 45.0 * (theta-90.0)**2
! Z.XU at UNIBAS.CH

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    
    use cnst_fcm
    use image

    INTEGER M,I,J,K,MODE

    real(chm_real) RI2,RJ2,RI,RJ,RIR,RJR,          &
          DXI,DYI,DZI,DXJ,DYJ,DZJ,         &
          DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,   &
          CST, AT, STR, ST2R,              &
          DA,DF,EA,                        &
          DTXI,DTYI,DTZI,DTXJ,DTYJ,DTZJ,   &
          DFX,DGX,DFY,DGY,DFZ,DGZ
          
    real(chm_real) DXAI,DYAI,DZAI,DXAJ,DYAJ,DZAJ,DXAK,DYAK,DZAK
    REAL(CHM_REAL) FCST,EQUI,EQUIDEG

    EA=0.0D0
    
    EQUI=EQUIDEG
    
    DXI=X(I)-X(J)
    DYI=Y(I)-Y(J)
    DZI=Z(I)-Z(J)
    DXJ=X(K)-X(J)
    DYJ=Y(K)-Y(J)
    DZJ=Z(K)-Z(J)

    RI2=DXI*DXI+DYI*DYI+DZI*DZI
    RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ

    RI=SQRT(RI2)
    RJ=SQRT(RJ2)

    RIR=1.D0/RI
    RJR=1.D0/RJ

    DXIR=DXI*RIR
    DYIR=DYI*RIR
    DZIR=DZI*RIR
    DXJR=DXJ*RJR
    DYJR=DYJ*RJR
    DZJR=DZJ*RJR

    CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR

    AT=ACOS(CST)
    
    IF (MODE.EQ.0) THEN
      CONTINUE
    ELSE IF (MODE.EQ.-1) THEN
      IF (AT.GE.EQUI) THEN
        EA=0.D0
        DXAI=0.D0
        DXAJ=0.D0
        DXAK=0.D0
  
        DYAI=0.D0
        DYAJ=0.D0
        DYAK=0.D0
  
        DZAI=0.D0
        DZAJ=0.D0
        DZAK=0.D0
        
        RETURN
      ENDIF
    ELSE
      IF (AT.LE.EQUI) THEN
        EA=0.D0
        DXAI=0.D0
        DXAJ=0.D0
        DXAK=0.D0
  
        DYAI=0.D0
        DYAJ=0.D0
        DYAK=0.D0
  
        DZAI=0.D0
        DZAJ=0.D0
        DZAK=0.D0
        
        RETURN
      ENDIF
    ENDIF
      
      
      
    DA=AT-EQUI
    DF=FCST*DA

    EA=DF*DA
    DF=DF+DF

    ST2R=1.D0/(1.D0-CST*CST)
    STR=SQRT(ST2R)
    DF=-DF*STR

    DTXI=RIR*(DXJR-CST*DXIR)
    DTXJ=RJR*(DXIR-CST*DXJR)
    DTYI=RIR*(DYJR-CST*DYIR)
    DTYJ=RJR*(DYIR-CST*DYJR)
    DTZI=RIR*(DZJR-CST*DZIR)
    DTZJ=RJR*(DZIR-CST*DZJR)

    DFX=DF*DTXI
    DGX=DF*DTXJ

    DFY=DF*DTYI
    DGY=DF*DTYJ

    DFZ=DF*DTZI
    DGZ=DF*DTZJ

    DXAI=DFX
    DXAJ=-DFX-DGX
    DXAK=DGX

    DYAI=DFY
    DYAJ=-DFY-DGY
    DYAK=DGY

    DZAI=DFZ
    DZAJ=-DFZ-DGZ
    DZAK=DGZ


    IF(PRNLEV.GT.7) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-ANGLE ATOMS (I, J, K):',I,J,K
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-ANGLE ENERGY',EA
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-ANGLE FORCE ATOM I',DXAI,DYAI,DZAI
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-ANGLE FORCE ATOM J',DXAJ,DYAJ,DZAJ
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-ANGLE FORCE ATOM K',DXAK,DYAK,DZAK
    ENDIF

    RETURN
  END SUBROUTINE EPSEUDOMMPT

  SUBROUTINE EPSEUDOBOND(I,J,EB,FCST,EQUI,MODE,DXBRM,DYBRM,DZBRM)

! ... routine calculates classical bond energy terms
! ... and forces for specific bonds
!      results or stored in ebmmpt and dxbrm, dxbrm
! ... based on charmm ebond subroutine

    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
    
    use cnst_fcm
    use image

    real(chm_real) EB,RX,RY,RZ,S1,S2,DB,DF,DXBRM,DYBRM,DZBRM
    real(chm_real) R,ERM,FCST,EQUI
    INTEGER I,II,J,IC,MODE

    ERM=0.0D0

    RX=X(I)-X(J)
    RY=Y(I)-Y(J)
    RZ=Z(I)-Z(J)
    S2=RX*RX + RY*RY + RZ*RZ
    S1=SQRT(S2)

    DB=S1-EQUI  !equillibrium for penalty
        
    IF (MODE.EQ.0) THEN
      CONTINUE
    ELSE IF (MODE.EQ.-1) THEN
      IF (DB.GE.0.D0) THEN
        EB=0.D0
        DXBRM=0.D0
        DYBRM=0.D0
        DZBRM=0.D0
        
        RETURN
      ENDIF
    ELSE
      IF (DB.LE.0.D0) THEN
        EB=0.D0
        DXBRM=0.D0
        DYBRM=0.D0
        DZBRM=0.D0
        
        RETURN
      ENDIF
    ENDIF
    
    
    DF=FCST*DB !force constant for penalty
    EB=DF*DB

    R=2.D0/S1
    DF=DF*R

    DXBRM=DF*RX
    DYBRM=DF*RY
    DZBRM=DF*RZ

    IF(PRNLEV.GT.7) THEN
      WRITE(OUTU,*) ' '
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-BOND ATOMS ATOM I',I,'  ATOM J', J
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-BOND ENERGY',EB
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-BOND FORCE ATOM I',DXBRM,DYBRM,DZBRM
      WRITE(OUTU,*) 'MMPT EXTERN> PSEUDO-BOND FORCE ATOM J',-DXBRM,-DYBRM,-DZBRM
      WRITE(OUTU,*) ' '
    ENDIF

  END SUBROUTINE EPSEUDOBOND
  
  SUBROUTINE BDINIT(NATOMX)
  
    INTEGER I,NATOMX
    
    DO I=1,NATOMX
      BDLIST(I)=-1
    ENDDO
  END SUBROUTINE BDINIT
  
  FUNCTION BONDDIST(ATA,ATB) RESULT(RV)
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_fcm
    use number
#if KEY_PARALLEL==1
    use parallel 
#endif
    
    IMPLICIT NONE
    INTEGER :: RV
    INTEGER I,BCHECK,ATA,ATB
    
    
    RV=999999
    BCHECK=0
    
    
    DO I=1,NATOM
      BDLIST(I)=-1
    ENDDO
    BDLIST(ATA)=0
    
    do WHILE (BCHECK.EQ.0)
      BCHECK=-1
      DO I=1,NBOND
        IF (BDLIST(IB(I)).GE.0) THEN
          IF (BDLIST(JB(I)).GE.0) THEN
            IF ((BDLIST(JB(I))-BDLIST(IB(I))).GT.1) THEN
              BDLIST(JB(I))=BDLIST(IB(I))+1
              BCHECK=0
            ELSE IF ((BDLIST(IB(I))-BDLIST(JB(I))).GT.1) THEN
              BDLIST(IB(I))=BDLIST(JB(I))+1
              BCHECK=0
            ENDIF
          ELSE
            BDLIST(JB(I))=BDLIST(IB(I))+1
            BCHECK=0
          ENDIF
        ELSE IF (BDLIST(JB(I)).GE.0) THEN
          BDLIST(IB(I))=BDLIST(JB(I))+1
          BCHECK=0
        ENDIF
      ENDDO
    ENDDO
    
    if (BDLIST(ATB).gt.0) then
      RV=BDLIST(ATB)
    endif
            
          

!    DO I=1, NBOND
!      IF (IB(I).EQ.ATA.OR.(JB(I).EQ.ATA)) THEN
!        IF (IB(I).EQ.ATA) THEN
!          A0=IB(I)
!          A1=JB(I)
!        ELSE
!          A0=JB(I)
!          A1=IB(I)
!        ENDIF
!        IF (A1.EQ.ATB) THEN
!          RV=1
!          BDLIST(A1)=1
!        ELSE
!          IF (BDLIST(A1).EQ.0) THEN
!            BDLIST(A1)=1
!            TMP=BONDDIST(A1,ATB)+1
!            IF (TMP.LT.RV) THEN
!              RV=TMP
!            ENDIF
!          ENDIF
!        ENDIF
!      ENDIF
!    ENDDO
!write(*,*) 'TEST1>',BDLIST
!write(*,*) 'BDTEST01>',RV,ATA,ATB
!write(*,*) 'BDTEST02>',BDLIST(1),BDLIST(2),BDLIST(3),BDLIST(4),BDLIST(5) &
!,BDLIST(6),BDLIST(7),BDLIST(8),BDLIST(9)
  END FUNCTION BONDDIST
#else /**/
  SUBROUTINE NULL_MMPT
        RETURN
  END SUBROUTINE NULL_MMPT
#endif 

!=====================================================================
! End of module MMPT
!=====================================================================
end module mmpt_fcm

