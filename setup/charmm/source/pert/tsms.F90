module tsms_mod
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_TSM==1 /*tsms_fcm*/
  ! TSMS.FCM
  !
  ! SCALARS FOR PERTURBATION CALCULATION
  !
  ! DNTXY:           DNT = DON'T PERTURB
  !                  X=P,R P=PRODUCT; R=REACTANT
  !                  Y=B,A,P,I -> BOND (STRETCH), ANGLE, PHI, IMPROPER PHI
  !  E.G. DNTPB WHEN SET TO TRUE MEANS DON'T PERTURB PRODUCT BOND ANGLE
  !       INTERACTION TERMS.
  !
  !  LMDAST:         LAMBDA SET
  !  SLOWST:         SET SLOW GROWTH (LMDAST AND SLOWST ARE MUTUALLY EXCLUSIVE)
  !  SUBRCT:         WHETHER TO SUBTRACT FORCES DUE TO REACTANT ATOMS
  !                  ON ENVIRONMENT ATOMS WHEN USING DNTXY SWITCHES.
  !                  THIS IS USED PRIMARILY FOR ENDPOINT CALCULATIONS.
  !  SUBPRD:         WHETHER TO SUBTRACT FORCES DUE TO PRODUCT ATOMS
  !                  ON ENVIRONMENT ATOMS WHEN USING DNTXY SWITCHES.
  !  GCM:            FLAG WHETHER CENTER OF MASS GLUE OPTION WAS SPECIFIED.
  !  GATOMS:         FLAG WHETHER ATOM DIST. GLUE OPTION WAS SPECIFIED.
  !  SAVEP:          FLAG WHETHER TO STORE V(I) AND V(II) EVERYTIME COORD.
  !                  ARE STORED.
  !  GSUBR,GSUBP:    SUBTRACT GLUE FORCES FROM REACTANT, PRODUCT AND/OR
  !                      ENVIRONMENT ATOMS, RESPECTIVELY.
  !  PIGSET:         FLAG FOR PIGGYBACK OPTION.
  !  QTSM:           FLAG for TSM
  !

  ! NUMBER OF LOGICAL VARIABLES IN COMMON /LPERT/
  integer,PARAMETER :: DNTEL=22
  LOGICAL DNTPB,DNTRB,DNTPA,DNTRA,DNTPP,DNTRP,DNTPI,DNTRI
  LOGICAL LMDAST,SLOWST,SUBRCT,SUBPRD,GCM,GATOMS,SAVEP
  LOGICAL GSUBR,GSUBP,NOKER,NOKEP,PIGSET,UPPERT,QTSM
  !
  ! LAMBDA: PERTURBATION PARAMETER
  ! LPOWER: POWER OF LAMBDA AND 1-LAMBDA
  ! GFORCE: CONSTRAINT FORCE FOR GLUE OPTION
  ! GMIN:   CONSTRAINT DISTANCE FOR GLUE OPTION
  ! GATOMR,GATOMP: REACTANT ATOM AND PRODUCT ATOM NUMBERS
  !                FOR GLUE ATOM CONSTRAINT
  ! NRCTAT: NUMBER OF REACTANT ATOMS
  ! NPRDAT: NUMBER OF PRODUCT ATOMS
  ! NENVAT: NUMBER OF ENVIRONMENT ATOMS (NATOM - (NRCTAT+NPRDAT))
  ! VPRTTR: REACTANT PERTURBATION POTENTIAL ENERGY ALL INTERACTIONS
  ! VPRTTP: PRODUCT PERTURBATION POTENTIAL ENERGY "
  ! VPRTNR: REACTANT PERTURBATION POTENTIAL ENERGY VDW + ELEC
  ! VPRTNP: PRODUCT PERTURBATION POTENTIAL ENERGY "
  ! VPRTVR: REACTANT PERTURBATION POTENTIAL ENERGY VDW ONLY
  ! VPRTVP: PRODUCT PERTURBATION POTENTIAL ENERGY "
  ! VPRTER: REACTANT PERTURBATION POTENTIAL ENERGY ELECTROSTATIC ONLY
  ! VPRTEP: PRODUCT PERTURBATION POTENTIAL ENERGY "
  ! PUNITX:  FORTRAN UNIT NUMBER FOR WRITING THE PERTURBED POTENTIALS
  ! PERFRQ: FREQUENCY OF PRINTING OUT PERTURBATION POTENTIALS
  ! ASLOW:  HELMHOLTZ FREE ENERGY ACCUMULATOR FOR SLOW GROWTH
  ! DLMBDA: DELTA_LAMBDA
  ! SLTEMP: TEMPERATURE FOR ASLOW
  ! LMFROM: FOR SLOW GROWTH BEGINNING LAMBDA
  ! LMTO:   FOR SLOW GROWTH ENDING LAMBDA
  ! PIGGY:  PIGGYBACK OPTION ATOM THAT IS ACTUALLY EVOLVED.
  ! BACK:   ""   ATOM THAT GOES ALONG FOR THE RIDE.
  ! APIGG:  STORAGE FOR ATOMIC MASS OF ATOM PIGGY. SO WE CAN LAMBDA FACTOR
  !         AMASS(PIGGY)
  !  NCOLO:            NUMBER OF COLOCATED ATOMS (MAX=20)
  !  ACOLO:            COLOCATED ATOM LIST (OF ATOM NUMBERS)
  !  CCOLO:            PRODUCT CHARGES OF COLOCATED ATOM
  !  NPUMB:           UMBRELLA SAMPLING NUMBER OF DIHEDRAL ANGLES
  !  PUMBDH:          INTEGER ARRAY OF DIHEDRAL ANGLES FOR U.S.
  !  PUMTYP:          INTEGER ARRAY OF TYPES (1=REAC,2=PROD,3=ENV)
  !  VPUMB:           STORES VSURROGATE
  !  VPUMBS:          STORES VSURROGATE-VACTUAL
  !  NPIGG:           NUMBER OF PIGGY-BACK PAIRS
  !
  !
  !! INTEGER BPERT(50),LPERT(50)
#if KEY_TSM==1
    !  FROM:  tsms.fcm
    !  MXPIGG - The maximum number of "piggyback atom pairs".
    integer,parameter :: MXPIGG=500
    integer,parameter :: MXCOLO=20,MXPUMB=20
    !
#endif 
    !-----------------------------------------------------------------------
  INTEGER GATOMR,GATOMP,LPOWER
  INTEGER NRCTAT,NPRDAT,NENVAT,NLAMDA,PUNITX,PERFRQ
  INTEGER PIGGY(MXPIGG),BACK(MXPIGG)
  real(chm_real)  LAMBDA,GFORCE,GMIN,VPRTTR,VPRTTP,VPRTNR,VPRTNP, &
       VPRTVR,VPRTVP,VPRTER,VPRTEP,SLTEMP,ASLOW, &
       LMFROM,LMTO,DLMBDA,APIGG(MXPIGG)
  INTEGER NCOLO,NPUMB,NPIGG
  real(chm_real) CCOLO(MXCOLO)
  INTEGER PUMBDH(MXPUMB),ACOLO(MXCOLO)
  INTEGER PUMTYP(MXPUMB)
  real(chm_real) VPUMB(MXPUMB),VPUMBS(MXPUMB)

#endif /* (tsms_fcm)*/
  !
end module tsms_mod

#if KEY_TSM==1 /*tsms_main*/
SUBROUTINE TSMS
  ! VAX VERSION.  Except for the use of include files this is the
  !               same code as the CRAY version.
  ! This file corresponds to the CRAY version of 13 April, 1987.
  ! contains perturbation subroutines and FORTRAN-77 versions
  ! of the FLECS routines DCNTRL and DYNAMC
  ! This code will run on the cray once the contents of the include files are
  ! inserted and the ministration of crayf77 are employed.
  ! Written by Stephen Fleischman.
  !
  ! Changed from USER command to PERT command in main charmm command parser.
  ! Also included all ic perturbation and constraint set-up in this routine.
  ! The command descriptions below were changed to reflect the additions.
  ! DT 6-Mar-89
  !
  ! START OF DECLARATIONS FROM INCLUDE FILES
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use select
  use bases_fcm
  use consta
  use comand
  use coord
  use ctitla
  use energym
  use inbnd
  use image
  use param
  use psf
  use stream
  use string
  use timerm
  use memory
  use tsms_mod
  use tsmh
  use icpert
  use icfix
  use machutil,only:die
!---   use nbutil_module,only:gtnbct
  implicit none
  !
  ! END OF DECLARATIONS FROM INCLUDE FILES
  !
  !
  !     THE FOLLOWING ARE LOCAL VARIABLES
  !
  LOGICAL   EOF,GO,ERR,CHKRPL,OK
  CHARACTER(LEN=4)  WRD
  INTEGER I,J
  CHARACTER(LEN=4) WINIT
  CHARACTER(LEN=80) ELINE
  LOGICAL RSET,PSET,CHKNONE
  LOGICAL LPRICF,LPRICP
  INTEGER ICP
  real(chm_real) RYAN
  !
  !...memory arrays
  real(chm_real),allocatable,dimension(:) :: ihdudi,ihvali,ihdedi,ihdadi,ihaflu,ihbdad,ihbafl
  real(chm_real),allocatable,dimension(:) :: ihgufl,ihgsfl,ihssum,ihbssu,ihsflu,ihusum,ihuflu
  real(chm_real),allocatable,dimension(:) :: ihbufl,ihbsfl,ihde2e,ihdee2,ihbbu,ihbbs
  type(chm_ptr),allocatable,dimension(:) :: ihibxp,ihibyp,ihibzp
  real(chm_real),allocatable,dimension(:) :: ihxtem,ihytem,ihztem,ihrmsv,ihmast
  
  integer ierr_allocate
  !
  ! ...This sets the maximum number of path points in PUICA ...KK
  INTEGER,PARAMETER :: MXPSIZ=800
  ! for running ave !RJP
  real(chm_real) TEMPER
  !
  !                       Commands for perturbation setup.
  !
  !
  !1. REACtant atom_selection_list | NONE     specify reactant atoms
  !               NONE - no reactant atoms (growing something in)
  !
  !2. PRODuct atom_selction_list   | NONE     specify product atoms
  !               NONE - no product atoms (growing something out)
  !
  !3. LAMBda <real> [ POWEr <int> ]           set perturbation parameter
  !        POWEr sets potential E(lambda) = (1-lambda)**NEr + lambda**NEp
  !                             (default 1) LAMBda must come first.
  !
  !4. DONT {REACtant} {internal_energy_spec} [SUBTract]
  !        {PRODuct} {internal_energy_spec}
  !        internal_energy_spec :== BOND THETa|ANGLe PHI|DIHEd IMPHi|IMPR
  !        (Any combination of the above is permissable. At least one
  !         must be specified).
  !         SUBTract: subtract the forces off of the environment atoms.
  !
  !6. SAVE [UNIT <integer>] [FREQ <integer>]
  !        save V(I) and V(II) on UNIT
  !
  !7. SAVIc [ICUNit <integer>] [ICFReq <integer>] [NWINdows <integer>]
  !        save ic perturbation energies on ICUN (default freq 10)
  !        NWIN is the number of ic perturbation windows (default 1)
  !
  !8. COLO atom_spec PCHArge <real>
  !        atom_spec ::= segid resnum type
  !        PCHArge product charge, reactant charge is assumed to be given
  !
  !9. SLOW TEMP <real> LFROm <real> LTO <real>
  !        TEMP temperature for use in calculating delta A
  !
  !10. PIGGyback PIGGy atom_spec BACK atom_spec
  !         atom_spec ::= segid resnum type
  !         "Piggybacks" the BACK atom on the PIGGY atom.  Only one such pair
  !         is currently allowed.  All subsequent invocations of this command
  !         merely replaces the previous one.
  !
  !11. UMBRella 4x( atom_spec) VACTual <real>
  !         atom_spec: segid resnum type
  !
  !12. FIX  {ic-spec} [TOLI <real>]
  !         set up ic constraints; see icfix.flx, subroutine icfset,
  !         for more description
  !
  !13. MAXI <integer>
  !         maximum number of iterations for ic constraints (default 500)
  !
  !14. MOVE {ic-spec} 2x{sele-spec} BY <real>
  !         set up ic perturbations; see icpert.flx, subroutine icpset,
  !         for more description
  !
  !15. END finish setup
  !
  ! Note: must have non-bonded exclusions between reactant and product atoms
  !       in rtf.
  !*********************************************************
  !     END DECLARATIONS -- EXECUTION BEGINS
  !*********************************************************
  !
  LNPRIC=.FALSE.  !reset with each tsm (RJP)
  WRD=NEXTA4(COMLYN,COMLEN)
  IF(WRD == 'CLEA') THEN
     CALL TSMCLEAR
     !...free up scratch arrays and reset storage flag QPRTSS
     IF(QPRTSS) THEN
        DO I=1,12
           call chmdealloc('tsms.src','TSMS','IHPCFTI(I)',MAXA,crlp=IHPCFTI(I)%a)
        END DO
     END IF
     QPRTSS=.FALSE.
     !...reset flags for conformational TI
     QCFTI=.FALSE.
     QCFTM=.FALSE.
     !
     RETURN
  ELSE IF (WRD == 'POST') THEN
     CALL TSMP
     RETURN
  ELSE
     CALL TRIME(COMLYN,COMLEN)
     IF(COMLEN /= 0) THEN
        CALL XTRANE(COMLYN,COMLEN,'TSMS')
        CALL DIEWRN(0)
     END IF
  ENDIF
  ! initialize data structures
  CALL TSMINIT(.true.)    !mfc allocate arrays herexs
  !
  ! ...initialize scratch arrays and set storage flag
  IF(.NOT.QPRTSS) THEN
     DO I=1,12
        call chmalloc('tsms.src','TSMS','IHPCFTI(I)',MAXA,crlp=IHPCFTI(I)%a)
     ENDDO
     QPRTSS=.TRUE.
  ENDIF

  !     allocate memory for data structure


  NRCTAT = 0
  NPRDAT = 0
  NENVAT = NATOM
  PERFRQ = 0
  !!been specified.  Glue command cannot be entered until they are.
  RSET = .FALSE.
  PSET = .FALSE.
  NCOLO = 0
  NPUMB = 0
  NPIGG = 0
  LPRICF = .FALSE.
  LPRICP = .FALSE.
  CALL PIGGINI(NATOM,PIGGLS,BACKLS)
  GO = .TRUE.
  EOF = .FALSE.
  iunicp=-1
  isvicp=-1
50 IF(.NOT.GO) GOTO 100
  ERR = .FALSE.
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
       .TRUE.,'TSMS> ')
  IF (EOF) THEN
     IF (NSTRM  ==  1) THEN
        GO = .FALSE.
        GOTO 100
     ELSE
        CALL PPSTRM(OK)
        EOF=.FALSE.
     END IF
  END IF
  WRD=NEXTA4(COMLYN,COMLEN)
  IF (WRD  ==  '    ') THEN
     CONTINUE
     !
     ! ...Start section for Conformational TI, KK 19-Mar-1997
     !
     ! ... CFTI: Conformational Thermodynamic Integration, 1-Dimensional
  ELSE IF (WRD == 'CFTI') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' <<Conformational thermodynamic integration' &
          ,' activated>>'
     WRITE(OUTU,*)
     QCFTI=.TRUE.
     QCFTM=.FALSE.
     !
     ! ... CFTI: Conformational free energy analysis,  1-Dimensional
  ELSE IF (WRD  ==  'CFTA') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' <<Conformational thermodynamic integration' &
          ,' analysis activated>>'
     WRITE(OUTU,*)
     CALL BLCFTI(COMLYN,COMLEN)
     !
     ! ...  CFTI: Conformational free energy analysis 1-Dimensional
  ELSE IF (WRD  ==  'CFTJ') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' <<Conformational thermodynamic integration' &
          ,' analysis activated>>'
     WRITE(OUTU,*)
     CALL BLCFTJ(COMLYN,COMLEN)
     !
     ! ... CFTM: Conformational thermodynamic integration, Multi-dimensional
  ELSE IF (WRD == 'CFTM') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' << Multidimensional Conformational Thermody' &
          ,'namic Integration activated>>'
     WRITE(OUTU,*)
     QCFTM=.TRUE.
     QICWR=.FALSE.
     QCFTI=.FALSE.
     !
     ! ... CFTM: Multidimensional TI analysis
     ! ... Free energy gradients along each coordinate and the path.
  ELSE IF (WRD  ==  'CFTB') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' << Multidimensional Conformational Thermody' &
          ,'namic Integration analysis active>>'
     WRITE(OUTU,*)

     call chmalloc('tsms.src','TSMS','IHDUDI',NICP,crl=IHDUDI)
     call chmalloc('tsms.src','TSMS','IHVALI',NICP,crl=IHVALI)
     call chmalloc('tsms.src','TSMS','IHDEDI',2*NICP+1,crl=IHDEDI)
     call chmalloc('tsms.src','TSMS','IHDADI',2*NICP+1,crl=IHDADI)
     call chmalloc('tsms.src','TSMS','IHAFLU',2*NICP+1,crl=IHAFLU)
     call chmalloc('tsms.src','TSMS','IHBDAD',2*NICP+1,crl=IHBDAD)
     call chmalloc('tsms.src','TSMS','IHBAFL',2*NICP+1,crl=IHBAFL)

     CALL BLCFTM(COMLYN,COMLEN,IHDUDI,IHVALI, &
          IHDEDI,IHDADI,IHAFLU,IHBDAD,IHBAFL)
          
     call chmdealloc('tsms.src','TSMS','IHDUDI',NICP,crl=IHDUDI)
     call chmdealloc('tsms.src','TSMS','IHVALI',NICP,crl=IHVALI)
     call chmdealloc('tsms.src','TSMS','IHDEDI',2*NICP+1,crl=IHDEDI)
     call chmdealloc('tsms.src','TSMS','IHDADI',2*NICP+1,crl=IHDADI)
     call chmdealloc('tsms.src','TSMS','IHAFLU',2*NICP+1,crl=IHAFLU)
     call chmdealloc('tsms.src','TSMS','IHBDAD',2*NICP+1,crl=IHBDAD)
     call chmdealloc('tsms.src','TSMS','IHBAFL',2*NICP+1,crl=IHBAFL)

     !
     ! ... CFTM: Multidimensional TI analysis
     ! ... Free energy, internal energy and entropy
     ! ... gradients along each coordinate and the path : use COR files
  ELSE IF (WRD  ==  'CFTS') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' << Multidimensional Conformational Thermody' &
          ,'namic Integration analysis active>>'
     WRITE(OUTU,*)
     call chmalloc('tsms.src','TSMS','IHDUDI',NICP,crl=IHDUDI)
     call chmalloc('tsms.src','TSMS','IHVALI',NICP,crl=IHVALI)
     call chmalloc('tsms.src','TSMS','IHGUFL',NICP,crl=IHGUFL)
     call chmalloc('tsms.src','TSMS','IHGSFL',NICP,crl=IHGSFL)
     call chmalloc('tsms.src','TSMS','IHDEDI',2*NICP+1,crl=IHDEDI)
     call chmalloc('tsms.src','TSMS','IHDADI',2*NICP+1,crl=IHDADI)
     call chmalloc('tsms.src','TSMS','IHAFLU',2*NICP+1,crl=IHAFLU)
     call chmalloc('tsms.src','TSMS','IHBDAD',2*NICP+1,crl=IHBDAD)
     call chmalloc('tsms.src','TSMS','IHBAFL',2*NICP+1,crl=IHBAFL)
     call chmalloc('tsms.src','TSMS','IHSSUM',2*NICP+1,crl=IHSSUM)
     call chmalloc('tsms.src','TSMS','IHBSSU',2*NICP+1,crl=IHBSSU)
     call chmalloc('tsms.src','TSMS','IHSFLU',2*NICP+1,crl=IHSFLU)
     call chmalloc('tsms.src','TSMS','IHUSUM',2*NICP+1,crl=IHUSUM)
     call chmalloc('tsms.src','TSMS','IHUFLU',2*NICP+1,crl=IHUFLU)
     call chmalloc('tsms.src','TSMS','IHBUFL',2*NICP+1,crl=IHBUFL)
     call chmalloc('tsms.src','TSMS','IHBSFL',2*NICP+1,crl=IHBSFL)
     call chmalloc('tsms.src','TSMS','IHDE2E',2*NICP+1,crl=IHDE2E)
     call chmalloc('tsms.src','TSMS','IHDEE2',2*NICP+1,crl=IHDEE2)
     call chmalloc('tsms.src','TSMS','IHBBU',2*NICP+1,crl=IHBBU)
     call chmalloc('tsms.src','TSMS','IHBBS',2*NICP+1,crl=IHBBS)
     CALL BLCFTS(COMLYN,COMLEN,IHDUDI,IHVALI, &
          IHGUFL,IHGSFL,IHDEDI,IHDADI, &
          IHAFLU,IHBDAD,IHBAFL,IHSSUM, &
          IHBSSU,IHSFLU,IHUSUM,IHUFLU, &
          IHBUFL,IHBSFL,IHDE2E,IHDEE2, &
          IHBBU,IHBBS)
     call chmdealloc('tsms.src','TSMS','IHDUDI',NICP,crl=IHDUDI)
     call chmdealloc('tsms.src','TSMS','IHVALI',NICP,crl=IHVALI)
     call chmdealloc('tsms.src','TSMS','IHGUFL',NICP,crl=IHGUFL)
     call chmdealloc('tsms.src','TSMS','IHGSFL',NICP,crl=IHGSFL)
     call chmdealloc('tsms.src','TSMS','IHDEDI',2*NICP+1,crl=IHDEDI)
     call chmdealloc('tsms.src','TSMS','IHDADI',2*NICP+1,crl=IHDADI)
     call chmdealloc('tsms.src','TSMS','IHAFLU',2*NICP+1,crl=IHAFLU)
     call chmdealloc('tsms.src','TSMS','IHBDAD',2*NICP+1,crl=IHBDAD)
     call chmdealloc('tsms.src','TSMS','IHBAFL',2*NICP+1,crl=IHBAFL)
     call chmdealloc('tsms.src','TSMS','IHSSUM',2*NICP+1,crl=IHSSUM)
     call chmdealloc('tsms.src','TSMS','IHBSSU',2*NICP+1,crl=IHBSSU)
     call chmdealloc('tsms.src','TSMS','IHSFLU',2*NICP+1,crl=IHSFLU)
     call chmdealloc('tsms.src','TSMS','IHUSUM',2*NICP+1,crl=IHUSUM)
     call chmdealloc('tsms.src','TSMS','IHUFLU',2*NICP+1,crl=IHUFLU)
     call chmdealloc('tsms.src','TSMS','IHBUFL',2*NICP+1,crl=IHBUFL)
     call chmdealloc('tsms.src','TSMS','IHBSFL',2*NICP+1,crl=IHBSFL)
     call chmdealloc('tsms.src','TSMS','IHDE2E',2*NICP+1,crl=IHDE2E)
     call chmdealloc('tsms.src','TSMS','IHDEE2',2*NICP+1,crl=IHDEE2)
     call chmdealloc('tsms.src','TSMS','IHBBU',2*NICP+1,crl=IHBBU)
     call chmdealloc('tsms.src','TSMS','IHBBS',2*NICP+1,crl=IHBBS)
     !
     ! ... CFTM: Multidimensional TI analysis
     ! ... Free energy, internal energy and entropy
     ! ... gradients along each coordinate and the path. ICP files are used
     !     instead of COR files
  ELSE IF (WRD  ==  'CFTC') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' << Multidimensional Conformational Thermody' &
          ,'namic Integration analysis active>>'
     WRITE(OUTU,*)
     call chmalloc('tsms.src','TSMS','IHVALI',NICP,crl=IHVALI)
     call chmalloc('tsms.src','TSMS','IHGUFL',NICP,crl=IHGUFL)
     call chmalloc('tsms.src','TSMS','IHGSFL',NICP,crl=IHGSFL)
     call chmalloc('tsms.src','TSMS','IHDEDI',2*NICP+1,crl=IHDEDI)
     call chmalloc('tsms.src','TSMS','IHDADI',2*NICP+1,crl=IHDADI)
     call chmalloc('tsms.src','TSMS','IHAFLU',2*NICP+1,crl=IHAFLU)
     call chmalloc('tsms.src','TSMS','IHBDAD',2*NICP+1,crl=IHBDAD)
     call chmalloc('tsms.src','TSMS','IHBAFL',2*NICP+1,crl=IHBAFL)
     call chmalloc('tsms.src','TSMS','IHSSUM',2*NICP+1,crl=IHSSUM)
     call chmalloc('tsms.src','TSMS','IHBSSU',2*NICP+1,crl=IHBSSU)
     call chmalloc('tsms.src','TSMS','IHSFLU',2*NICP+1,crl=IHSFLU)
     call chmalloc('tsms.src','TSMS','IHUSUM',2*NICP+1,crl=IHUSUM)
     call chmalloc('tsms.src','TSMS','IHUFLU',2*NICP+1,crl=IHUFLU)
     call chmalloc('tsms.src','TSMS','IHBUFL',2*NICP+1,crl=IHBUFL)
     call chmalloc('tsms.src','TSMS','IHBSFL',2*NICP+1,crl=IHBSFL)
     call chmalloc('tsms.src','TSMS','IHDE2E',2*NICP+1,crl=IHDE2E)
     call chmalloc('tsms.src','TSMS','IHDEE2',2*NICP+1,crl=IHDEE2)
     call chmalloc('tsms.src','TSMS','IHBBU',2*NICP+1,crl=IHBBU)
     call chmalloc('tsms.src','TSMS','IHBBS',2*NICP+1,crl=IHBBS)

     CALL BLCFTC(COMLYN,COMLEN,IHVALI, &
          IHGUFL,IHGSFL,IHDEDI,IHDADI, &
          IHAFLU,IHBDAD,IHBAFL,IHSSUM, &
          IHBSSU,IHSFLU,IHUSUM,IHUFLU, &
          IHBUFL,IHBSFL,IHDE2E,IHDEE2, &
          IHBBU,IHBBS)

     call chmdealloc('tsms.src','TSMS','IHVALI',NICP,crl=IHVALI)
     call chmdealloc('tsms.src','TSMS','IHGUFL',NICP,crl=IHGUFL)
     call chmdealloc('tsms.src','TSMS','IHGSFL',NICP,crl=IHGSFL)
     call chmdealloc('tsms.src','TSMS','IHDEDI',2*NICP+1,crl=IHDEDI)
     call chmdealloc('tsms.src','TSMS','IHDADI',2*NICP+1,crl=IHDADI)
     call chmdealloc('tsms.src','TSMS','IHAFLU',2*NICP+1,crl=IHAFLU)
     call chmdealloc('tsms.src','TSMS','IHBDAD',2*NICP+1,crl=IHBDAD)
     call chmdealloc('tsms.src','TSMS','IHBAFL',2*NICP+1,crl=IHBAFL)
     call chmdealloc('tsms.src','TSMS','IHSSUM',2*NICP+1,crl=IHSSUM)
     call chmdealloc('tsms.src','TSMS','IHBSSU',2*NICP+1,crl=IHBSSU)
     call chmdealloc('tsms.src','TSMS','IHSFLU',2*NICP+1,crl=IHSFLU)
     call chmdealloc('tsms.src','TSMS','IHUSUM',2*NICP+1,crl=IHUSUM)
     call chmdealloc('tsms.src','TSMS','IHUFLU',2*NICP+1,crl=IHUFLU)
     call chmdealloc('tsms.src','TSMS','IHBUFL',2*NICP+1,crl=IHBUFL)
     call chmdealloc('tsms.src','TSMS','IHBSFL',2*NICP+1,crl=IHBSFL)
     call chmdealloc('tsms.src','TSMS','IHDE2E',2*NICP+1,crl=IHDE2E)
     call chmdealloc('tsms.src','TSMS','IHDEE2',2*NICP+1,crl=IHDEE2)
     call chmdealloc('tsms.src','TSMS','IHBBU',2*NICP+1,crl=IHBBU)
     call chmdealloc('tsms.src','TSMS','IHBBS',2*NICP+1,crl=IHBBS)
     !
     ! ... CFTM: define number of coordinates
  ELSE IF (WRD  ==  'NCOR') THEN
     NICP=GTRMI(COMLYN, COMLEN, 'NUMB', 0)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' The number of coordinates are: ',NICP
     WRITE(OUTU,*)
     !
     ! ... CFTM: path analysis in BLCFTM & BLCFTS   (Y. Wang, 04/16/96)
  ELSE IF (WRD  ==  'DIRE') THEN
     YLAMBD=GTRMI(COMLYN, COMLEN, 'LAMB',  -999)
     IF (YLAMBD  >  -999) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' Setting up the path of reaction:'
        WRITE(OUTU,*) '  --number of step is',YLAMBD
        READ(ISTRM,*)  (DIRV(J), J=1,NICP)
        WRITE(OUTU,*) ' --The direction of the conformational ' &
             , 'change is:'
        RYAN = ZERO
        DO I = 1,NICP
           RYAN = RYAN + DIRV(I)**2
        END DO
        RYAN = SQRT(RYAN)
        DO I = 1,NICP
           DIRV(I) = DIRV(I)/RYAN
        END DO
        WRITE(OUTU,110) (DIRV(J), J=1,NICP)
        WRITE(OUTU,*)
     END IF
110  FORMAT(1X,5(F13.7,1X))
#if KEY_TRAVEL==1
     !
     ! ... PUIC : Reaction path under Internal constraints.
  ELSE IF (WRD  ==  'PUIC') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' << Begin Path Search Under The Internal ' &
          ,'Constraints>>'
     WRITE(OUTU,*)
     
     allocate(ihibxp(mxpsiz),ihibyp(mxpsiz),ihibzp(mxpsiz),stat=ierr_allocate)
     if(ierr_allocate /= 0) &
         CALL WrnDie(-1,'<tsms.src>puica', &
                      'Failed to allocate memory for ihibxp,ihibyp,ihibzp')
     call chmalloc('tsms.src','TSMS','IHXTEM',NATOM,crl=IHXTEM)
     call chmalloc('tsms.src','TSMS','IHYTEM',NATOM,crl=IHYTEM)
     call chmalloc('tsms.src','TSMS','IHZTEM',NATOM,crl=IHZTEM)
     call chmalloc('tsms.src','TSMS','IHRMSV',MXPSIZ,crl=IHRMSV)
     call chmalloc('tsms.src','TSMS','IHMAST',MXPSIZ,crl=IHMAST)

     CALL PUICA(COMLYN,COMLEN,MXPSIZ,IHIBXP,IHIBYP, &
          IHIBZP,IHXTEM,IHYTEM,IHZTEM, &
          IHRMSV,IHMAST)

     deallocate(ihibxp,ihibyp,ihibzp,stat=ierr_allocate)
     if(ierr_allocate /= 0) &
         CALL WrnDie(-1,'<tsms.src>puica', &
                      'Failed to deallocate memory for ihibxp,ihibyp,ihibzp')
     call chmdealloc('tsms.src','TSMS','IHXTEM',NATOM,crl=IHXTEM)
     call chmdealloc('tsms.src','TSMS','IHYTEM',NATOM,crl=IHYTEM)
     call chmdealloc('tsms.src','TSMS','IHZTEM',NATOM,crl=IHZTEM)
     call chmdealloc('tsms.src','TSMS','IHRMSV',MXPSIZ,crl=IHRMSV)
     call chmdealloc('tsms.src','TSMS','IHMAST',MXPSIZ,crl=IHMAST)

#endif 
     !
     ! ... CFTM: Group analysis in BLCFTM & BLCFTS
  ELSE IF (WRD  ==  'CFTG') THEN
     NGRUP =GTRMI(COMLYN, COMLEN, 'NGRU',      0)
     IF(NGRUP > 0) THEN
        IF(NGRUP > MXICP) CALL WRNDIE(-4,'<TSMS>', &
             'Too many groups, increase MXICP')
        WRITE(OUTU,*) '  Setting up group contribution analysis:'
        WRITE(OUTU,*) '  --number of groups is',NGRUP
        WRITE(OUTU,*) '  --group numbers of the coordinates'
        READ(ISTRM,*)  (LGRUP(J),J=1,NICP)
        WRITE(OUTU,'(20I4)')  (LGRUP(J),J=1,NICP)
        WRITE(OUTU,*) '  --group symbols'
        READ(ISTRM,'(20A4)')  (GSYM(J),J=1,NGRUP)
        WRITE(OUTU,'(20A4)')  (GSYM(J),J=1,NGRUP)
        DO J=1,NGRUP
           NGRU(J)=0
        END DO
        DO I=1,NICP
           J=LGRUP(I)
           NGRU(J)=NGRU(J)+1
        END DO
     END IF
     !
     ! ...End section for Conformational TI, KK 19-Mar-1997
     !
  ELSE IF (WRD == 'REAC') THEN
     !              !reactant selection
     IF(CHKNONE(NATOM,REACLS,COMLYN,COMLEN)) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,*) 'No reactant atoms.'
        RSET = .TRUE.
        ERR = .FALSE.
        NRCTAT = 0
     ELSE
        CALL SELCTA(COMLYN,COMLEN,REACLS,X,Y,Z, &
             WMAIN,ERR)
     END IF
     IF(.NOT.ERR) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,*) 'Reactant atom list'
        CALL PRNSEL(REACLS,NATOM)

        CALL MAKSKP(REACLS,BSKIPR, &
             ASKIPR,UBSKIPR, &
             PSKIPR,ISKIPR &
#if KEY_CMAP==1
             ,CTSKIPR &     
#endif
        )

        !***end of clbii mode for ub
        RSET = .TRUE.
        NRCTAT = NSELCT(NATOM,REACLS)
     ELSE
        ERR = .TRUE.
        CALL WRNDIE(0,'<TSMS>', &
             'Selection unsuccessful try again.')
     END IF
     IF(.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIEWRN(0)
        END IF
     END IF
  ELSE IF (WRD == 'PROD') THEN
     !              !product selection
     IF(CHKNONE(NATOM,PRODLS,COMLYN,COMLEN)) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,*) 'No product atoms.'
        PSET = .TRUE.
        ERR = .FALSE.
        NPRDAT = 0
     ELSE
        CALL SELCTA(COMLYN,COMLEN,PRODLS,X,Y,Z, &
             WMAIN,ERR)
     END IF
     IF(.NOT.ERR) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,*) 'Product atom list'
        CALL PRNSEL(PRODLS,NATOM)

        CALL MAKSKP(PRODLS,BSKIPP, &
             ASKIPP,UBSKIPP, &
             PSKIPP,ISKIPP &
#if KEY_CMAP==1
             ,CTSKIPP &    
#endif
        )

        !***end of clbii mode for ub
        PSET = .TRUE.
        NPRDAT = NSELCT(NATOM,PRODLS)
     ELSE
        ERR = .TRUE.
        CALL WRNDIE(0,'<TSMS>', &
             'Selection unsuccessful try again.')
     END IF
     IF(.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIEWRN(0)
        END IF
     END IF
  ELSE IF (WRD == 'LAMB') THEN
     !   lambda specification
     LAMBDA = NEXTF(COMLYN,COMLEN)
     IF(LAMBDA < 0.0 .OR. LAMBDA > 1.0) THEN
        !   error code for illegal lambda
        ERR = .TRUE.
        CALL WRNDIE(0,'<TSMS>', &
             'Lambda must be between 0. and 1. Renter it.')
     ELSE
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,F7.4)') &
             'Lambda set to ',LAMBDA
        LMDAST = .TRUE.
        QTSM = .TRUE.
     END IF
     IF(.NOT.ERR) THEN
        IF(INDX(COMLYN,COMLEN,'POWE',4) /= 0) THEN
           LPOWER = GTRMI(COMLYN,COMLEN,'POWE',0)
           IF (LPOWER <= 0) THEN
              CALL WRNDIE(0,'<TSMS>', &
                   'Error parsing power of lambda or zero or negative value')
              ERR = .TRUE.
              LMDAST = .FALSE.
              QTSM = .FALSE.
              IF(WRNLEV >= 2) WRITE(OUTU,*)'lambda not set'
           END IF
        ELSE
           LPOWER = 1
        END IF
        IF (.NOT.ERR .AND. PRNLEV >= 2) WRITE(OUTU, &
             '(1X,A,I4)') 'power of lambda set to ',LPOWER
     END IF
     IF (.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIEWRN(0)
        END IF
     END IF
  ELSE IF (WRD == 'DONT') THEN
     !       perturbation exemption
     WRD = NEXTA4(COMLYN,COMLEN)
     IF (WRD == 'REAC') THEN
        IF(RSET) THEN
           CALL PARDNT('REAC')
        ELSE
           CALL WRNDIE(0,'<TSMS>', &
                'Reactant list has not been specified')
        END IF
     ELSE IF(WRD == 'PROD') THEN
        IF(PSET) THEN
           CALL PARDNT('PROD')
        ELSE
           CALL WRNDIE(0,'<TSMS>', &
                'Product list has not been specified')
        END IF
     ELSE
        ERR = .TRUE.
        CALL WRNDIE(0,'<TSMS>', &
             'In DONT command only REAC or PROD allowed')
     END IF
     IF(.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIEWRN(0)
        END IF
     END IF
  ELSE IF (WRD == 'UMBR') THEN
     IF(.NOT.(RSET.AND.PSET)) THEN
        CALL WRNDIE(0,'<TSMS>', &
             'UMBRella command issued before REAC or PROD command')
     ELSE
        ERR = .FALSE.
        NPUMB = NPUMB+1
        IF (NPUMB < MXPUMB) THEN
           CALL PUMBR(COMLYN,COMLEN,PSKIPR, &
                PSKIPP,PUMBDH, &
                PUMTYP,VPUMB,NPUMB,ERR)
           IF (ERR) THEN
              CALL WRNDIE(0,'<TSMS>', &
                   'UMBrella command error')
              NPUMB = NPUMB-1
           END IF
        ELSE
           IF(WRNLEV >= 2) WRITE(ELINE,'(A,I5,A)') &
                'UMBRella error, exceeded maximum number (',MXPUMB,').'
           CALL WRNDIE(0,'<TSMS>',ELINE)
           IF(WRNLEV >= 2) WRITE(OUTU,*) &
                'All values have been removed.'
           NPUMB = 0
        END IF
        IF(.NOT.ERR) THEN
           CALL TRIME(COMLYN,COMLEN)
           IF(COMLEN /= 0) THEN
              CALL XTRANE(COMLYN,COMLEN,'TSMS')
              CALL DIEWRN(0)
           END IF
        END IF
     END IF
  ELSE IF (WRD == 'PIGG') THEN
     IF(RSET.AND.PSET) THEN
        CALL PIGG(COMLYN,COMLEN,REACLS, &
             PRODLS,NPIGG,PIGGY,BACK, &
             PIGSET)
        IF (.NOT.PIGSET) THEN
           ERR = .TRUE.
        ENDIF
     ELSE
        ERR = .TRUE.
        CALL WRNDIE(0,'<TSMS>', &
             'PIGG: either reaclist or prodlist not specified')
     ENDIF
     IF(.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIE
        END IF
     END IF
  ELSE IF (WRD == 'SAVE') THEN
     IF(INDX(COMLYN,COMLEN,'UNIT',4) /= 0) THEN
        PUNITX = GTRMI(COMLYN,COMLEN,'UNIT',-1)
        IF(PUNITX < 0) THEN
           ERR = .TRUE.
           CALL WRNDIE(0,'<TSMS>', &
                'error in parsing unit in save command')
        ELSE
           IF(INDX(COMLYN,COMLEN,'FREQ',4) /= 0) THEN
              PERFRQ = GTRMI(COMLYN,COMLEN,'FREQ',-1)
              IF(PERFRQ < 0) CALL WRNDIE(0,'<TSMS>', &
                   'error in parsing printing frequency')
           ELSE
              IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
                   'perfrq set to 10 (default)'
              PERFRQ = 10
           END IF
        END IF
     ELSE
        PUNITX = -1
        ERR = .TRUE.
        CALL WRNDIE(0,'<TSMS>', &
             'did not find UNIT specifier in SAVE command.')
     END IF
     IF (PUNITX >= 0.AND.PERFRQ >= 0) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I8,A,I8,A)') &
             'V(I) and V(II) will be saved on unit ',PUNITX, &
             ' every ',PERFRQ,' steps.'
        SAVEP = .TRUE.
     END IF
     IF(.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIEWRN(0)
        END IF
     END IF
  ELSE IF (WRD == 'SAVI') THEN
     IF(INDX(COMLYN,COMLEN,'ICUN',4) /= 0) THEN
        IUNICP = GTRMI(COMLYN,COMLEN,'ICUN',-1)
        IF(IUNICP < 0) THEN
           ERR = .TRUE.
           CALL WRNDIE(0,'<TSMS>', &
                'error parsing ICUN in SAVI command.')
        ELSE
           IF(INDX(COMLYN,COMLEN,'ICFR',4) /= 0) THEN
              ISVICP = GTRMI(COMLYN,COMLEN,'ICFR',-1)
              IF(ISVICP < 0) THEN
                 CALL WRNDIE(0,'<TSMS>', &
                      'error parsing SAVI frequency')
              END IF
              IF(ISVICP == 0) THEN
                 CALL WRNDIE(0,'<TSMS>', &
                      'zero SAVI frequency')
              END IF
           ELSE
              IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
                   'SAVI frequency set to 10 (default)'
              ISVICP = 10
           END IF
        END IF
     ELSE
        IUNICP = -1
        ERR = .TRUE.
        CALL WRNDIE(0,'<TSMS>', &
             'did not find ICUN specifier in SAVI command')
     END IF
     IF(INDX(COMLYN,COMLEN,'NWIN',4) > 0) THEN
        ICPINC = GTRMI(COMLYN,COMLEN,'NWIN',-1)
        IF(ICPINC < 0) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'error parsing number of windows.')
        END IF
        IF(ICPINC == 0) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'number of windows must be greater than zero')
        END IF
     ELSE
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
             'number of windows set to 10 (default)'
        ICPINC = 1
     END IF
     ! SUPP suppresses printing of ic values during run (LNPRIC .TRUE.) -RJP
     IF(INDXA(COMLYN,COMLEN,'SUPP') > 0) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
             'IC values will not be printed'
        LNPRIC =.TRUE.
     ENDIF
     ! **** For running averages ************************ RJP
     LRUNNA = .FALSE.
     IF(INDXA(COMLYN,COMLEN,'RUNA') > 0) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
             'Running average for free energy requested'
        LRUNNA = .TRUE.
        TEMPER = GTRMF(COMLYN,COMLEN,'TEMP',THRHUN)
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,1X,F8.3)') &
             'Temperature for averaging = ',TEMPER
        KTEMPE = TEMPER*KBOLTZ
        RUNITN = GTRMI(COMLYN,COMLEN,'RUNI',-1)
        IF (RUNITN == -1) THEN
           CALL WRNDIE(-5,'<TSMS>', &
                'no unit no. specified for run ave outpt')
        ELSE
           IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,1X,I4)') &
                'Averages to be written to unit ',RUNITN
        ENDIF
        RPRCYC = GTRMI(COMLYN,COMLEN,'RPRI',-1)
        IF (RPRCYC == -1) THEN
           CALL WRNDIE(-5,'<TSMS>', &
                'no print cycle specified for run ave')
        ENDIF
        PEVERY = GTRMI(COMLYN,COMLEN,'PEVE',1)
        !                INITIALIZE minima and sums and counters
        TSMSTP = 0
        DO I = 1,ICPINC
           MINFOR(I) = 9999999
           MINBAC(I) = 9999999
           EAVFOR(I) = 0
           EAVBAC(I) = 0
           SUMFOR(I) = 0
           SUMBAC(I) = 0
        ENDDO
        DO I = 1,NICP
           ICAVER(I) = 0
        ENDDO
     ELSE  !(if not running ave)
        IF (PEVERY /= 1) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
                'No running ave requested. Printing all perturbations.'
           PEVERY = 1
        ENDIF !if pevery ne 1
     ENDIF  !if running averages

     !  end section for running averages *******************
     !
     IF (IUNICP >= 0.AND.ISVICP > 0.AND.ICPINC > 0) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I3,A,I6,A)') &
             'ic pert. energies will be saved on unit ',IUNICP,' every ', &
             ISVICP*PEVERY,' steps.'    !RJP
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,I5,A)') &
             ICPINC,' windows will be used.'
     ELSE
        CALL WRNDIE(0,'<TSMS>', &
             'ic pert. energies will not be saved.')
     END IF
     IF(.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIEWRN(0)
        END IF
     END IF
  ELSE IF (WRD == 'COLO') THEN
     ERR = .FALSE.
     IF(NCOLO < MXCOLO) THEN
        NCOLO = NCOLO + 1
        CALL COLO(COMLYN,COMLEN,ACOLO,CCOLO,NCOLO,ERR)
        IF(ERR) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'COLO command error')
           NCOLO = NCOLO - 1
        END IF
     ELSE
        CALL WRNDIE(0,'<TSMS>', &
             'COLOcate error, exceeded maximum number.')
     END IF
     IF(.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIEWRN(0)
        END IF
     END IF
  ELSE IF (WRD == 'SLOW') THEN
     ERR = .FALSE.
     IF(INDX(COMLYN,COMLEN,'TEMP',4) /= 0) THEN
        SLTEMP = GTRMF(COMLYN,COMLEN,'TEMP',MINONE)
        IF(SLTEMP < 0.0) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'temperature not found or unacceptable')
           ERR = .TRUE.
        END IF
     ELSE
        IF(PRNLEV >= 2) WRITE(OUTU,*) 'sltemp set to 298K.'
        SLTEMP = 298.
     END IF
     IF(INDX(COMLYN,COMLEN,'POWE',4) /= 0) THEN
        LPOWER = GTRMI(COMLYN,COMLEN,'POWE',0)
        IF(LPOWER <= 0) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'Error parsing power of lambda or zero or negative value')
           ERR = .TRUE.
           LMDAST = .FALSE.
           QTSM = .FALSE.
           IF(WRNLEV >= 2) WRITE(OUTU,*) 'lambda not set'
        END IF
     ELSE
        LPOWER = 1
     END IF
     LMFROM = 0.
     LMTO = 1.
     IF(.NOT.ERR.AND.(INDX(COMLYN,COMLEN,'LFRO',4) /= 0)) THEN
        LMFROM = GTRMF(COMLYN,COMLEN,'LFRO',MINONE)
        IF(LMFROM < 0.0) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'Error in parsing LFROm or negative value.')
           ERR = .TRUE.
        ELSE IF (LMFROM > 1.0) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'Error in LFROm, value greater than 1.')
           ERR = .TRUE.
        END IF
     END IF
     IF(.NOT.ERR.AND.(INDX(COMLYN,COMLEN,'LTO',3) /= 0)) THEN
        LMTO = GTRMF(COMLYN,COMLEN,'LTO',MINONE)
        IF(LMTO < 0.0) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'Error in parsing LTO or negative value.')
           ERR = .TRUE.
        ELSE IF (LMTO > 1.0) THEN
           CALL WRNDIE(0,'<TSMS>', &
                'Error in LFROm, value greater than 1.')
           ERR = .TRUE.
        END IF
     END IF
     IF(.NOT.ERR) THEN
        CALL TRIME(COMLYN,COMLEN)
        IF(COMLEN /= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'TSMS')
           CALL DIEWRN(0)
        END IF
     END IF
     IF(.NOT.ERR) THEN
        SLOWST = .TRUE.
        QTSM = .TRUE.
        LAMBDA = LMFROM
     END IF
  ELSE IF (WRD == 'FIX') THEN
     CALL ICFSET
     IF (NICF > 0) LPRICF = .TRUE.
  ELSE IF (WRD == 'MAXI') THEN
     MAXI = NEXTI(COMLYN,COMLEN)
     IF(MAXI <= 0) CALL WRNDIE(0,'<TSMS>', &
          'error parsing number of iterations')
  ELSE IF (WRD == 'MOVE') THEN
     CALL ICPSET
     IF (NICP > 0) LPRICP = .TRUE.
  ELSE IF (WRD == 'END') THEN
     ERR = .FALSE.
     ! pert exit
     GO = .FALSE.
     NENVAT = NATOM -(NRCTAT+NPRDAT)
     IF(.NOT.(LMDAST.OR.SLOWST).AND.NICP+NICF == 0) THEN
        CALL WRNDIE(0,'<TSMS>', &
             'Leaving pert. setup without setting lambda or slow growth.')
        RSET = .FALSE.
        PSET = .FALSE.
        NRCTAT = 0
        NPRDAT = 0
        NCOLO = 0
        ERR = .TRUE.
     ELSE IF (SLOWST.AND.LMDAST) THEN
        CALL WRNDIE(0,'<TSMS>', &
             'Cannot set both lambda and slow growth.')
        RSET = .FALSE.
        PSET = .FALSE.
        NRCTAT = 0
        NPRDAT = 0
        NCOLO = 0
        ERR = .TRUE.
     ELSE IF (NRCTAT+NPRDAT+NCOLO == 0.AND. &
          NICP+NICF == 0) THEN
        CALL WRNDIE(0,'<TSMS>', &
             'Neither reactant, product or colo atoms were selected')
        RSET = .FALSE.
        PSET = .FALSE.
        ERR = .TRUE.
     END IF
     IF (.NOT.ERR) THEN
        ERR = CHKRPL(REACLS,PRODLS,NATOM)
        IF (ERR) CALL WRNDIE(0,'<TSMS>', &
             'Found same atom selected in both reactant and product list')
        IF (NCOLO > 0) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,*) &
                'Co-locate option product charges'
           IF(PRNLEV >= 2) WRITE(OUTU,*) &
                'atom           product charge      reactant charge'
           IF(PRNLEV >= 2) WRITE(OUTU, &
                '(I5,9X,F9.4,11X,F9.4)') &
                (ACOLO(I),CCOLO(I),CG(ACOLO(I)),I=1,NCOLO)
        END IF
        IF (NPUMB > 0) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,*) &
                'Dihedral angles selected for umbrella sampling:'
           IF(PRNLEV >= 2) WRITE(OUTU,'(1X,10(I4,2X))') &
                (PUMBDH(I),I=1,NPUMB)
           IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
                'Use SAVE command to set output unit and frequency'
        END IF
        IF(NPIGG > 0) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,*) &
                'Piggy Back pairs (piggy - back):'
           !                   now make final lists.
           CALL PIGGFIN(PIGGLS,BACKLS,NPIGG,PIGGY,BACK)
        END IF
     END IF
     IF(LPRICF) THEN
        CALL PRNICF
        IICF = 1
        LPRICF = .FALSE.
     END IF
     IF(LPRICP) THEN
        !***CLBIII modification as per DJT
        !
        !     Fill selection arrays for interaction energy calculations.
        !
        
        call chmalloc('tsms.src','TSMS','JSLICP',NATOM,intg=JSLICP)
        call chmalloc('tsms.src','TSMS','ISLICP',NATOM,intg=ISLICP)

        CALL INITSL(ISLICP,JSLICP,NATOM)
        DO ICP = 1,NICP

           CALL GETISL(ICPMV1(ICP)%a,ISLICP, &
                NMOV1(ICP))
           IF(LMOV2(ICP)) CALL GETISL(ICPMV2(ICP)%a, &
                ISLICP,NMOV2(ICP))
        enddo
        CALL PRNICP
        IICP = 1
        LPRICP = .FALSE.
     END IF
     ! This will force update of the exclusion lists (necessary for Ewald).
     WINIT = 'INIT'
     J=4
     CALL GTNBCT(WINIT,J,BNBND)
  ELSE
     CALL WRNDIE(0,'<TSMS>', &
          'Unknown command.  Try again.')
  END IF
  ! end do
  GOTO 50
100 CONTINUE
  IF(EOF) THEN
     ! program termination
     CALL WRNDIE(-4,'<TSMS>', &
          'Eof found during input for perturbation set up.')
     QTSM = .FALSE.
     LMDAST = .FALSE.
     SLOWST = .FALSE.
  END IF
  RETURN
END SUBROUTINE TSMS

LOGICAL FUNCTION CHKNONE(N,SLCT,ST,STLEN)

  use chm_kinds
  use dimens_fcm
  use string
  use stream
  implicit none
  INTEGER N,I,STLEN,STSCR
  CHARACTER(len=*) :: ST
  character(len=4) :: WD
  INTEGER SLCT(*)
  ! This function is used in processing the reac and product commands.
  ! It checks to see if the next word after reac or prod is 'NONE'.
  ! If not, the original command string is undisturbed. If so, the word
  ! is removed.
  SCRTCH = ST
  STSCR = STLEN
  WD = NEXTA4(SCRTCH,STSCR)
  IF(WD == 'NONE') THEN
     !        remove word from command line string
     WD = NEXTA4(ST,STLEN)
     SLCT(1:n) = 0
     CHKNONE = .TRUE.
  ELSE
     CHKNONE = .FALSE.
  END IF
  RETURN
END FUNCTION CHKNONE

LOGICAL FUNCTION CHKRPL(REACLS,PRODLS,NATOM)
  ! checks for redundancies in reactant and product lists
  !
  use chm_kinds
  use stream
  use chutil,only:atomid
  implicit none
  INTEGER REACLS(*),PRODLS(*)
  INTEGER NATOM,I
  CHARACTER(LEN=8) :: SID,RID,REN,AC

  CHKRPL = .FALSE.
  DO I = 1,NATOM
     IF (REACLS(I) == 1.AND.PRODLS(I) == 1) THEN
        CALL ATOMID(I,SID,RID,REN,AC)
        IF(WRNLEV >= 2) WRITE(OUTU,'(1X,A,I8,7A)') &
             'atom # ',I,' (',SID(1:idleng),RID(1:idleng), &
             REN(1:idleng),AC(1:idleng),')', &
             ' was found in both the reactant and product lists.'
        CHKRPL = .TRUE.
     END IF
  enddo
  RETURN
END FUNCTION CHKRPL


SUBROUTINE LOGINI
  !
  ! I have replaced this clever routine with explicit initialization
  ! to make it run on SGI  KK, 04/02/97
  use chm_kinds
  use dimens_fcm
  use tsms_mod
  implicit none
  !
  DNTPB = .FALSE.
  DNTRB = .FALSE.
  DNTPA = .FALSE.
  DNTRA = .FALSE.
  DNTPP = .FALSE.
  DNTRP = .FALSE.
  DNTPI = .FALSE.
  DNTRI = .FALSE.
  LMDAST = .FALSE.
  SLOWST = .FALSE.
  SUBRCT = .FALSE.
  SUBPRD = .FALSE.
  GCM = .FALSE.
  GATOMS = .FALSE.
  SAVEP = .FALSE.
  GSUBR = .FALSE.
  GSUBP = .FALSE.
  NOKER = .FALSE.
  NOKEP = .FALSE.
  PIGSET = .FALSE.
  UPPERT = .FALSE.
  QTSM = .FALSE.
  !
  RETURN
END SUBROUTINE LOGINI

SUBROUTINE MAKSKP(SLCT,BSKIP,ASKIP,UBSKIP,PSKIP,ISKIP &
#if KEY_CMAP==1
     ,CTSKIP &       
#endif
  )
  !  Set up perturbation skip lists

  ! BEGINNING OF DECLARATIONS SPECIFIED BY INCLUDE STATEMENTS
  use chm_kinds
  use dimens_fcm
  use psf
  use stream
  implicit none
  ! END OF DECLARATIONS SPECIFIED BY INCLUDE STATEMENTS

  INTEGER SLCT(*),BSKIP(*),ASKIP(*),PSKIP(*),ISKIP(*),I
  INTEGER UBSKIP(*)

#if KEY_CMAP==1
  INTEGER CTSKIP(*)
#endif 

  IF(PRNLEV >= 2) WRITE(OUTU,*) 'Makskp: bond selection'
  DO I=1,NBOND
     IF(SLCT(IB(I)) == 1 .OR. SLCT(JB(I)) == 1) THEN
        BSKIP(I)=0
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I4,A,I4,A,I4)') &
             'bond ',I,' atoms ',IB(I),'-',JB(I)
     ELSE
        BSKIP(I)=1
     END IF
  enddo
  IF(PRNLEV >= 2) WRITE(OUTU,*) 'Makskp: theta selection'
  DO I = 1,NTHETA
     IF(SLCT(IT(I)) == 1 .OR. SLCT(JT(I)) == 1 .OR. &
          SLCT(KT(I)) == 1) THEN
        ASKIP(I) = 0
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I4,A,I4,A,I4,A,I4)') &
             'angle ',I,' atoms ',IT(I),'-',JT(I),'-',KT(I)
     ELSE
        ASKIP(I) = 1
     END IF
  enddo

  !***mod clbiii to incorporate ub terms into tsm/10-95

  IF(PRNLEV >= 2) WRITE(OUTU,*) 'Makskp: ureyb selection'
  DO I = 1,NTHETA
     IF(SLCT(IT(I)) == 1 .OR. SLCT(KT(I)) == 1) THEN
        UBSKIP(I) = 0
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I4,A,I4,A,I4,A,I4)') &
             'ureyb ',I,' atoms ',IT(I),'-',KT(I)
     ELSE
        UBSKIP(I) = 1
     END IF
  enddo
  !***end of mod clbiii to incorporate ub terms into tsm/10-95

  IF(PRNLEV >= 2) WRITE(OUTU,*) 'Makskp: phi selection'
  DO I = 1,NPHI
     IF (SLCT(IP(I)) == 1 .OR. SLCT(JP(I)) == 1 .OR. &
          SLCT(KP(I)) == 1 .OR. SLCT(LP(I)) == 1) THEN
        PSKIP(I) = 0
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I4,A,I4,A,I4,A,I4,A,I4)') &
             'phi ',I,' atoms ',IP(I),'-',JP(I),'-',KP(I),'-',LP(I)
     ELSE
        PSKIP(I) = 1
     END IF
  enddo

  IF(PRNLEV >= 2) WRITE(OUTU,*) 'Makskp: imphi selection'
  DO I = 1,NIMPHI
     IF (SLCT(IM(I)) == 1 .OR. SLCT(JM(I)) == 1 .OR. &
          SLCT(KM(I)) == 1 .OR. SLCT(LM(I)) == 1) THEN
        ISKIP(I) = 0
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I4,A,I4,A,I4,A,I4,A,I4)') &
             'imphi ',I,' atoms ',IM(I),'-',JM(I),'-',KM(I),'-',LM(I)
     ELSE
        ISKIP(I) = 1
     END IF
  enddo


#if KEY_CMAP==1
  IF(PRNLEV >= 2) WRITE(OUTU,*) 'Makskp: cmap selection'
  DO I = 1,NCRTERM
     IF (SLCT(I1CT(I)) == 1 .OR. SLCT(J1CT(I)) == 1 .OR. &
          SLCT(K1CT(I)) == 1 .OR. SLCT(L1CT(I)) == 1 .OR. &
          SLCT(I2CT(I)) == 1 .OR. SLCT(J2CT(I)) == 1 .OR. &
          SLCT(K2CT(I)) == 1 .OR. SLCT(L2CT(I)) == 1) THEN
        CTSKIP(I) = 0
        IF(PRNLEV >= 2) &
             WRITE(OUTU, &
             '(1X,A,I4,A,I4,A,I4,A,I4,A,I4,/,A,I4,A,I4,A,I4,A,I4)') &
             'cmap ',I,' atoms ',I1CT(I),'-',J1CT(I),'-',K1CT(I),'-', &
             L1CT(I),I2CT(I),'-',J2CT(I),'-',K2CT(I),'-',L2CT(I)
     ELSE
        CTSKIP(I) = 1
     END IF
  enddo
#endif 

  RETURN
END SUBROUTINE MAKSKP

SUBROUTINE PARDNT(OPT)

  use chm_kinds
  use dimens_fcm
  use tsms_mod
  use tsmh
  use stream
  use string
  use comand

  implicit none

  CHARACTER(LEN=4) :: OPT,WD
  LOGICAL GO,REAC
  INTEGER COUNT

  IF (OPT == 'REAC') THEN
     REAC = .TRUE.
  ELSE IF (OPT == 'PROD') THEN
     REAC = .FALSE.
  ELSE
     CALL WRNDIE(-5,'<PARDNT>', &
          'Programmer error: opt was neither REAC or PROD')
     RETURN
  END  IF
  COUNT = 0
  GO = .TRUE.
  !     do while(go)
50 IF(.NOT.GO) GOTO 100
  WD = NEXTA4(COMLYN,COMLEN)
  IF (STRLNG(WD) == 0) THEN
     GO = .FALSE.
  ELSE  IF (WD == 'BOND') THEN
     IF(REAC) THEN
        DNTRB = .TRUE.
     ELSE
        DNTPB = .TRUE.
     END IF
  ELSE IF (WD == 'THET'.OR.WD == 'ANGL') THEN
     IF(REAC) THEN
        DNTRA = .TRUE.
     ELSE
        DNTPA = .TRUE.
     END IF
  ELSE IF (WD == 'PHI '.OR.WD == 'DIHE') THEN
     IF(REAC) THEN
        DNTRP = .TRUE.
     ELSE
        DNTPP = .TRUE.
     END IF
  ELSE IF (WD == 'IMPH'.OR.WD == 'IMPR') THEN
     IF(REAC) THEN
        DNTRI = .TRUE.
     ELSE
        DNTPI = .TRUE.
     END IF
  ELSE IF (WD == 'SUB ') THEN
     IF(REAC) THEN
        SUBRCT = .TRUE.
     ELSE
        SUBPRD = .TRUE.
     END IF
  ELSE
     !              CALL LOGINI(DNTPB,DNTEL)
     CALL LOGINI
     IF(WRNLEV >= 2) WRITE(OUTU,'(1X,A,A)') &
          'PARDNT error with command: ',WD
     CALL WRNDIE(0,'<PARDNT>', &
          'Illegal energy option in DONT command. Try again')
     IF(WRNLEV >= 2) WRITE(OUTU,*) &
          'All DONT switches set to false.'
     RETURN
  END IF
  IF(GO) COUNT = COUNT+1
  !     end do
  GOTO 50
100 CONTINUE
  IF (COUNT > 0) THEN
     IF(SUBRCT.AND.(.NOT.(DNTRB.OR.DNTRA.OR.DNTRP.OR.DNTRI))) THEN
        CALL WRNDIE(0,'<PARDNT>', &
             ' Selected subrct w.o./ any donts')
     ELSE IF(SUBPRD.AND.(.NOT.(DNTPB.OR.DNTPA.OR.DNTPP.OR. &
          DNTPP.OR.DNTPI))) THEN
        CALL WRNDIE(0,'<PARDNT>', &
             ' Selected subprd w.o./ any donts')
     ELSE
        IF(PRNLEV >= 2) WRITE(OUTU,*) 'DONT perturb list'
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,L1,A,L1,A,L1,A,L1)') &
             'REACT:  BOND ',DNTRB,' THETA ',DNTRA,' PHI ',DNTRP, &
             ' IMPHI ',DNTRI
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,L1,A,L1,A,L1,A,L1)') &
             'PROD:  BOND ',DNTPB,' THETA ',DNTPA,' PHI ',DNTPP, &
             ' IMPHI ',DNTPI
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,L1,A,L1)') &
             'Subtract option, REAC: ',SUBRCT,' PROD: ',SUBPRD
     END IF
  ELSE
     CALL WRNDIE(0,'<PARDNT>',' Empty option list')
  END IF
  RETURN
END SUBROUTINE PARDNT

SUBROUTINE PRNSEL(LST,NLST)
  !!diagnostic printout of atom selections.
  !
  use chm_kinds
  use stream
  implicit none
  CHARACTER(len=4),dimension(20) :: LINE
  INTEGER LST(*)
  INTEGER NLST,I,POS

  POS = 0

  DO I = 1,NLST
     IF(LST(I) == 1) THEN
        POS = POS + 1
        WRITE(LINE(POS),'(I4)') I
        IF(POS == 20) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,'(20A4)') LINE
           POS = 0
        END IF
     END IF
  enddo
  IF (POS /= 0 .AND. PRNLEV >= 2) WRITE(OUTU,'(20A4)') &
       (LINE(I),I=1,POS)
  RETURN
END SUBROUTINE PRNSEL

SUBROUTINE PIGG(COMLYN,COMLEN,REACLS,PRODLS, &
     NPIGG,PIGGY,BACK,PIGSET)

  use chm_kinds
  use dimens_fcm
  use number
  use psf
  use coord
  use consta
  use stream
  use string
  use chutil,only:getatn
  use tsms_mod,only:MXPIGG
  implicit none

  LOGICAL PIGSET
  CHARACTER(LEN=*) COMLYN
  INTEGER COMLEN,I,PIGGY(*),BACK(*),NPIGG
  INTEGER REACLS(*),PRODLS(*)
  INTEGER POS,WDLEN,PIGGYT,BACKT
  CHARACTER(LEN=8) :: ARES,ASEG,AATOM

  PIGSET = .TRUE.
  ! get reac and prod piggyback atoms
  POS = INDXA(COMLYN,COMLEN,'PIGG')
  IF (POS == 0) POS = INDXA(COMLYN,COMLEN,'REAC')
  IF (POS == 0) THEN
     CALL WRNDIE(0,'<PIGG>', &
          'Error in piggyback command no REAC or PIGGy command')
     PIGSET =.FALSE.
     RETURN
  END IF
  CALL NXTWDA(COMLYN,COMLEN,ASEG,8,WDLEN,POS)
  CALL NXTWDA(COMLYN,COMLEN,ARES,8,WDLEN,POS)
  CALL NXTWDA(COMLYN,COMLEN,AATOM,8,WDLEN,POS)
  PIGGYT = GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE, &
       NICTOT,NSEGT)
  IF(PIGGYT <= 0.OR.PIGGYT > NATOM) THEN
     CALL WRNDIE(0,'<PIGG>', &
          'error in atom entry in PIGG REAC (PIGGy) command')
     PIGSET = .FALSE.
     RETURN
  END IF
  IF(REACLS(PIGGYT) /= 1) THEN
     CALL WRNDIE(0,'<PIGG>', &
          'error: PIGG REAC (PIGGy) atom not a reactant atom.')
     PIGSET = .FALSE.
     RETURN
  END IF
  ! product atom
  POS = INDXA(COMLYN,COMLEN,'BACK')
  IF (POS == 0) POS = INDXA(COMLYN,COMLEN,'PROD')
  IF (POS == 0) THEN
     CALL WRNDIE(0,'<PIGG>', &
          'Error in piggyback command no PROD or BACK command')
     PIGSET =.FALSE.
     RETURN
  END IF
  CALL NXTWDA(COMLYN,COMLEN,ASEG,8,WDLEN,POS)
  CALL NXTWDA(COMLYN,COMLEN,ARES,8,WDLEN,POS)
  CALL NXTWDA(COMLYN,COMLEN,AATOM,8,WDLEN,POS)
  BACKT = GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE, &
       NICTOT,NSEGT)
  IF(BACKT <= 0.OR.BACKT > NATOM) THEN
     CALL WRNDIE(0,'<PIGG>', &
          'error in atom entry in PIGG PROD (BACK) command')
     PIGSET = .FALSE.
     RETURN
  END IF
  IF(PRODLS(BACKT) /= 1) THEN
     CALL WRNDIE(0,'<PIGG>', &
          'error: PIGG PROD (BACK) atom not a product atom.')
     PIGSET = .FALSE.
     RETURN
  END IF
  !     check for duplicates
  DO I =1,NPIGG
     IF(PIGGYT == PIGGY(I)) THEN
        CALL WRNDIE(0,'<PIGG>', &
             'Piggy atom already specified as a piggy atom. Ignored.')
        PIGSET = .FALSE.
        RETURN
     ELSEIF(PIGGYT == BACK(I)) THEN
        CALL WRNDIE(0,'<PIGG>', &
             'Piggy atom already specified as a back atom. Ignored.')
        PIGSET = .FALSE.
        RETURN
     ELSEIF(BACKT == PIGGY(I)) THEN
        CALL WRNDIE(0,'<PIGG>', &
             'Back atom already specified as a piggy atom. Ignored.')
        PIGSET = .FALSE.
        RETURN
     ELSEIF(BACKT == BACK(I)) THEN
        CALL WRNDIE(0,'<PIGG>', &
             'Back atom already specified as a back atom. Ignored.')
        PIGSET = .FALSE.
        RETURN
     ENDIF
  enddo
  IF (NPIGG < MXPIGG) THEN
     NPIGG = NPIGG+1
     PIGGY(NPIGG) = PIGGYT
     BACK(NPIGG) = BACKT
  ELSE
     CALL WRNDIE(0,'<PIGG>', &
          'error: number of piggy back pairs exceeded maximum')
     PIGSET = .FALSE.
     RETURN
  ENDIF
  X(BACKT) = X(PIGGYT)
  Y(BACKT) = Y(PIGGYT)
  Z(BACKT) = Z(PIGGYT)
  !     this will prevent the back atom from being counted in center of
  !     mass; moment of inertia; and kinetic energy.
  !     the back atom always gets set to the piggy atom coordinates
  !     and velocities.
  IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I5,A,/,1X,A,I5,A)') &
       'PIGGyback: The coordinates of atom ',BACKT,' have been moved', &
       'to those of atom ',PIGGYT,'.'
  RETURN
END SUBROUTINE PIGG


SUBROUTINE COLO(COMLYN,COMLEN,ACOLO,CCOLO,NCOLS,ERR)

  use chm_kinds
  use dimens_fcm
  use psf
  use stream
  use string
  use chutil,only:getatn
  implicit none

  CHARACTER(LEN=*) COMLYN
  INTEGER COMLEN,J,NCOLS,ACOLO(*)
  real(chm_real) CCOLO(*)
  LOGICAL ERR
  real(chm_real) CHARGE
  CHARACTER(LEN=8) :: ARES,ASEG,AATOM
  real(chm_real),PARAMETER :: BIGNUM=-9999999.0D0,BIGGER=-9999999.99D0

  ERR = .FALSE.
  ASEG = NEXTA8(COMLYN,COMLEN)
  ARES = NEXTA8(COMLYN,COMLEN)
  AATOM = NEXTA8(COMLYN,COMLEN)
  J = GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE, &
       NICTOT,NSEGT)
  IF(J <= 0.OR.J > NATOM) THEN
     CALL WRNDIE(0,'<COLO>', &
          'error in atom entry in COLO command')
     ERR = .TRUE.
     RETURN
  END IF
  CHARGE = GTRMF(COMLYN,COMLEN,'PCHA',BIGGER)
  IF (CHARGE <= BIGNUM) THEN
     CALL WRNDIE(0,'<COLO>', &
          'error in PCHArge input')
     ERR= .TRUE.
     RETURN
  ENDIF
  ACOLO(NCOLS) = J
  CCOLO(NCOLS) = CHARGE
  IF (INDX(COMLYN,COMLEN,'RCHA',4) /= 0) THEN
     CHARGE = GTRMF(COMLYN,COMLEN,'RCHA',BIGGER)
     IF (CHARGE <= BIGNUM) THEN
        CALL WRNDIE(0,'<COLO>', &
             'error in RCHArge input')
        ERR= .TRUE.
        RETURN
     ENDIF
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,1X,A,1X,A,1X,A)') &
          'Changing the reactant charge for colocated atom ',ASEG,ARES,AATOM
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,G9.4,A,G9.4)') &
          'The old charge was: ',CG(J),' the new charge is: ',CHARGE
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
          'This permanently changes the charge for the rest of this job.'
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A)') &
          'The old charge can be restored using a scalar command.'
     CG(J) = CHARGE
  END IF
  RETURN
END SUBROUTINE COLO

SUBROUTINE PUMBR(COMLYN,COMLEN,PSKIPR,PSKIPP,PUMBDH,PUMTYP, &
     VPUMB,NPUMBL,ERR)

  use chm_kinds
  use dimens_fcm
  use psf
  use param
  use stream
  use string
  use chutil,only:getatn

  implicit none

  CHARACTER(LEN=*) COMLYN
  INTEGER COMLEN,PUMBDH(*),IPUMB(4),I,NPUMBL
  INTEGER PUMTYP(*),PSKIPR(*),PSKIPP(*)
  real(chm_real) VPUMB(*),VTEMP
  LOGICAL ERR,FOUND
  INTEGER LPUMST
  CHARACTER(len=20) :: PUMSTR
  CHARACTER(LEN=8) :: ARES,ASEG,AATOM
  real(chm_real),PARAMETER :: BIGNUM=-9999999.0D0,BIGGER=-9999999.99D0

  FOUND = .FALSE.
  ERR = .FALSE.
  DO I = 1,4
     ASEG = NEXTA8(COMLYN,COMLEN)
     ARES = NEXTA8(COMLYN,COMLEN)
     AATOM = NEXTA8(COMLYN,COMLEN)
     IPUMB(I) = GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE, &
          NICTOT,NSEGT)
     IF(IPUMB(I) <= 0.OR.IPUMB(I) > NATOM) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,'(A,I1,A)') &
             'error in ',I,'th atom entry in UMBRella command.'
        CALL DIEWRN(0)
        ERR = .TRUE.
        RETURN
     END IF
  enddo
  !
  ! now check dihedral angle list
  loop200: DO I = 1,NPHI
     FOUND = ( ( IPUMB(1) == IP(I).AND.IPUMB(2) == JP(I).AND. &
          IPUMB(3) == KP(I).AND.IPUMB(4) == LP(I) ) .OR. &
          ( IPUMB(1) == LP(I).AND.IPUMB(2) == KP(I).AND. &
          IPUMB(3) == JP(I).AND.IPUMB(4) == IP(I) ) )
     IF(FOUND) THEN
        PUMBDH(NPUMBL) = I
        IF(PSKIPR(I) == 0) THEN
           PUMTYP(NPUMBL) = 1
           PUMSTR =' reactant'
           LPUMST = 9
        ELSE IF(PSKIPP(I) == 0) THEN
           PUMTYP(NPUMBL) = 2
           PUMSTR =' product'
           LPUMST = 8
        ELSE
           PUMTYP(NPUMBL) = 3
           PUMSTR ='n enviroment'
           LPUMST = 13
        END IF
        IF(PRNLEV >= 2) WRITE(OUTU, &
             '(1X,A,I4,A,3(I4,''-''),I4,/,1X,A)') &
             'Selected dihedral angle # ',I,' consisting of atoms ', &
             IPUMB,' for umbrella processing.'
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,A,A)') &
             'This is a',PUMSTR(1:LPUMST),' term.'
     END IF
     ! exit do
     IF (FOUND) exit loop200
  enddo loop200

  ! did we find something?
  IF(.NOT.FOUND) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,'(A)') '<PUMBR>'
     IF(WRNLEV >= 2) WRITE(OUTU,'(A,/,3(I4,''-''),I4)') &
          'Could not find dihedral angle consisting of atoms ',IPUMB
     CALL DIEWRN(0)
     ERR = .TRUE.
  END IF
  ! now get vactual
  IF(INDX(COMLYN,COMLEN,'VACT',4) /= 0) THEN
     VPUMB(NPUMBL) = GTRMF(COMLYN,COMLEN,'VACT',BIGGER)
     IF(VPUMB(NPUMBL) <= BIGNUM) THEN
        CALL WRNDIE(0,'<PUMBR>', &
             'VACTual is unacceptable')
        ERR = .TRUE.
        RETURN
     END IF
     VTEMP = VPUMB(NPUMBL)
     IF(WRNLEV >= 2) WRITE(OUTU,'(1X,A,G15.5)') 'VACTual: ',VTEMP
  ELSE
     CALL WRNDIE(0,'<PUMBR>','VACTual specifier is missing')
     ERR = .TRUE.
     RETURN
  END IF
  RETURN
END SUBROUTINE PUMBR

SUBROUTINE TSMCLEAR
  !
  use chm_kinds
  use chm_types
  use memory
  use dimens_fcm
  use bases_fcm
  use energym
  use inbnd
  use image
  use psf
  use tsms_mod
  use tsmh
  use icfix
  use icpert
  use datstr,only:freedt_nbond,freedt_image
  implicit none
  !
  INTEGER I
  !
  !   clears tsm method data structures and flags
  !
  if(allocated(REACLS)) &
       call chmdealloc('tsms.src','TSMCLEAR','REACLS',siz1=SIZE(REACLS),intg=REACLS)
  if(allocated(PRODLS))  &
       call chmdealloc('tsms.src','TSMCLEAR','PRODLS',siz1=SIZE(PRODLS),intg=PRODLS)
  if(allocated(BSKIPR))  &
       call chmdealloc('tsms.src','TSMCLEAR','BSKIPR',siz1=SIZE(BSKIPR),intg=BSKIPR)
  if(allocated(BSKIPP)) &
       call chmdealloc('tsms.src','TSMCLEAR','BSKIPP',siz1=SIZE(BSKIPP),intg=BSKIPP)
  if(allocated(ASKIPR)) &
       call chmdealloc('tsms.src','TSMCLEAR','ASKIPR',siz1=SIZE(ASKIPR),intg=ASKIPR)
  if(allocated(ASKIPP)) &
       call chmdealloc('tsms.src','TSMCLEAR','ASKIPP',siz1=SIZE(ASKIPP),intg=ASKIPP)
  if(allocated(UBSKIPR)) &
       call chmdealloc('tsms.src','TSMCLEAR','UBSKIPR',siz1=SIZE(UBSKIPR),intg=UBSKIPR)
  if(allocated(UBSKIPP)) &
       call chmdealloc('tsms.src','TSMCLEAR','UBSKIPP',siz1=SIZE(UBSKIPP),intg=UBSKIPP)
#if KEY_CMAP==1
  if(allocated(CTSKIPR))  &                               
#endif
#if KEY_CMAP==1
       call chmdealloc('tsms.src','TSMCLEAR','CTSKIPR', & 
#endif
#if KEY_CMAP==1
       siz1=SIZE(CTSKIPR),intg=CTSKIPR)                   
#endif
#if KEY_CMAP==1
  if(allocated(CTSKIPP))  call chmdealloc('tsms.src','TSMCLEAR','CTSKIPP',  &  
#endif
#if KEY_CMAP==1
       siz1=SIZE(CTSKIPP),intg=CTSKIPP)     
#endif
  if(allocated(PSKIPR))  &
       call chmdealloc('tsms.src','TSMCLEAR','PSKIPR',siz1=SIZE(PSKIPR),intg=PSKIPR)
  if(allocated(PSKIPP)) &
       call chmdealloc('tsms.src','TSMCLEAR','PSKIPP',siz1=SIZE(PSKIPP),intg=PSKIPP)
  if(allocated(ISKIPR)) &
       call chmdealloc('tsms.src','TSMCLEAR','ISKIPR',siz1=SIZE(ISKIPR),intg=ISKIPR)
  if(allocated(ISKIPP)) &
       call chmdealloc('tsms.src','TSMCLEAR','ISKIPP',siz1=SIZE(ISKIPP),intg=ISKIPP)
  if(allocated(PIGGLS)) &
       call chmdealloc('tsms.src','TSMCLEAR','PIGGLS',siz1=SIZE(PIGGLS),intg=PIGGLS)
  if(allocated(BACKLS)) &
       call chmdealloc('tsms.src','TSMCLEAR','BACKLS',siz1=SIZE(BACKLS),intg=BACKLS)



  CALL FREEDT_nbond(BNBNDR)
  CALL FREEDT_image(BIMAGR)
  CALL FREEDT_nbond(BNBNDP)
  CALL FREEDT_image(BIMAGP)
  !      CALL LOGINI(DNTPB,DNTEL)
  CALL LOGINI
  PIGSET = .FALSE.
  LMDAST = .FALSE.
  SLOWST = .FALSE.
  UPPERT = .TRUE.
  QTSM = .FALSE.
  IF(IICP > 0) THEN
     call chmdealloc('tsms.src','TSMCLEAR','islicp',natom,intg=islicp)
     call chmdealloc('tsms.src','TSMCLEAR','jslicp',natom,intg=jslicp)
     DO I=1,NICP
        call chmdealloc('tsms.src','TSMCLEAR','ICPMV1(I)',NMOV1(I),intgp=ICPMV1(I)%a)
        IF(LMOV2(I)) THEN
           call chmdealloc('tsms.src','TSMCLEAR','ICPMV2(I)',NMOV2(I),intgp=ICPMV2(I)%a)
        END IF
     enddo
  END IF
  NICP = 0
  IICP = 0
  NICF = 0
  IICF = 0
  MAXI = 500
  RETURN
END SUBROUTINE TSMCLEAR

SUBROUTINE TSMINIT(allocate_arrays)
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use inbnd
  use image
  use psf
  use tsms_mod
  use tsmh
  use icfix
  use icpert
  implicit none
  !!     initialization of flags
  !!SETS ALL VARIABLES IN COMMON TO .FALSE.
  logical allocate_arrays

  CALL LOGINI

  if(allocate_arrays) then
     call tsminit_allocate('REACLS',NATOM,REACLS)
     call tsminit_allocate('PRODLS',NATOM,PRODLS)
     call tsminit_allocate('BSKIPR',NBOND,BSKIPR)
     call tsminit_allocate('BSKIPP',NBOND,BSKIPP)
     call tsminit_allocate('ASKIPR',NTHETA,ASKIPR)
     call tsminit_allocate('ASKIPP',NTHETA,ASKIPP)
     call tsminit_allocate('UBSKIPR',NTHETA,UBSKIPR)
     call tsminit_allocate('UBSKIPP',NTHETA,UBSKIPP)
#if KEY_CMAP==1
     call tsminit_allocate('CTSKIPR',NCRTERM,CTSKIPR)     
#endif
#if KEY_CMAP==1
     call tsminit_allocate('CTSKIPP',NCRTERM,CTSKIPP)     
#endif
     call tsminit_allocate('PSKIPR',NPHI,PSKIPR)
     call tsminit_allocate('PSKIPP',NPHI,PSKIPP)
     call tsminit_allocate('ISKIPR',NIMPHI,ISKIPR)
     call tsminit_allocate('ISKIPP',NIMPHI,ISKIPP)
     call tsminit_allocate('PIGGLS',NATOM,PIGGLS)
     call tsminit_allocate('BACKLS',NATOM,BACKLS)
  endif

  UPPERT = .TRUE.
  RETURN

  contains
    subroutine tsminit_allocate(name,size_new,array)
      character(len=*) name
      integer,allocatable,dimension(:) :: array
      integer size_new,size_now

      if (allocated(array)) then
         size_now=size(array)
         if (size_now == size_new) return
         call chmrealloc('tsms.src','TSMINIT',name,size_new,intg=array)
      else
         call chmalloc('tsms.src','TSMINIT',name,size_new,intg=array)
      end if

      array(1:size_new) = 0
      return
    end subroutine tsminit_allocate

END SUBROUTINE TSMINIT

SUBROUTINE PIGGSHK1(ICONB,IBTMP,JBTMP,ITTMP,JTTMP,KTTMP,AMTMP, &
     BACKLS,BACKPTR)
  use chm_kinds
  use dimens_fcm
  use number
  use psf
  use code
  use tsms_mod
  use stream
  implicit none
  !
  INTEGER ICONB
  INTEGER IBTMP(*),JBTMP(*),BACKPTR(*),BACKLS(*)
  INTEGER ITTMP(*),JTTMP(*),KTTMP(*)
  real(chm_real) AMTMP(*)
  INTEGER IX,JX,KX
  INTEGER I,J
  !
  ! Restructure bond and angle lists for shake when piggy back is being used.
  !
  IF(ICONB <= 0) RETURN
  !
  ! store bond list (will be restored later).
  !
  DO I = 1,NBOND
     IBTMP(I) = IB(I)
     JBTMP(I) = JB(I)
  ENDDO
  !
  !     store masses and initialize back atom list and map
  !
  DO I = 1,NATOM
     AMTMP(I) = AMASS(I)
     BACKPTR(I) = 0
  ENDDO
  !
  !     now setup back list and map into piggy atoms.
  DO I = 1,NPIGG
     BACKPTR(BACK(I)) = PIGGY(I)
     !        To make sure that piggy-back atoms that combine a hydrogen
     !        with a heavy atom are not considered hydrogens for SHAKE purposes
     !        the masses will be averaged.
     AMASS(PIGGY(I)) = HALF*(AMTMP(PIGGY(I))+AMTMP(BACK(I)))
  ENDDO
  !     We will convert all bonds involving "back atoms" to their equivalent
  !     "piggy atoms".
  DO I = 1,NBOND
     IX = IBTMP(I)
     JX = JBTMP(I)
     !        This will make the search for duplicates easier.
     IF(IX > 0.AND.JX > 0) THEN
        IF(BACKLS(IX) == 1) IB(I) = BACKPTR(IX)
        IF(BACKLS(JX) == 1) JB(I) = BACKPTR(JX)
     ENDIF
  ENDDO
  !     Search and zero duplicates.
  DO I = 1,NBOND-1
     IX=IB(I)
     JX=JB(I)
     DO J= I+1,NBOND
        IF((IX == IB(J).AND.JX == JB(J)) .OR. &
             (IX == JB(J).AND.JX == IB(J))) THEN
           IB(J) = 0
           JB(J) = 0
        ENDIF
     ENDDO
  ENDDO
  !     Angles?
  IF(ICONB <= 2) RETURN

  DO I = 1,NTHETA
     ITTMP(I) = IT(I)
     JTTMP(I) = JT(I)
     JTTMP(I) = KT(I)
  ENDDO
  !
  !     We will convert all bonds involving "back atoms" to their equivalent
  !     "piggy atoms".
  DO I = 1,NTHETA
     IX = ITTMP(I)
     JX = JTTMP(I)
     KX = JTTMP(I)
     !        This will make the search for duplicates easier.
     IF(IX > 0.AND.JX > 0.AND.KX.GT.0) THEN
        IF(BACKLS(IX) == 1) IT(I) = BACKPTR(IX)
        IF(BACKLS(JX) == 1) JT(I) = BACKPTR(JX)
        IF(BACKLS(KX) == 1) KT(I) = BACKPTR(KX)
     ENDIF
  ENDDO
  !     Search and zero duplicates.
  DO I = 1,NTHETA-1
     IX=IT(I)
     JX=JT(I)
     KX=KT(I)
     DO J= I+1,NTHETA
        IF((IX == IT(J).AND.JX == JT(J).AND.KX.EQ.KT(J)).OR. &
             (KX == IT(J).AND.JX == JT(J).AND.IX.EQ.KT(J))) THEN
           IT(I) = 0
           JT(I) = 0
           KT(I) = 0
        ENDIF
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE PIGGSHK1

SUBROUTINE PIGGSHK2(ICONB,IBTMP,JBTMP,ITTMP,JTTMP,KTTMP,AMTMP)
  use chm_kinds
  use dimens_fcm
  use number
  use psf
  use code
  use tsms_mod
  use tsmh
  use stream
  implicit none
  !
  INTEGER ICONB
  INTEGER IBTMP(*),JBTMP(*)
  INTEGER ITTMP(*),JTTMP(*),KTTMP(*)
  real(chm_real) AMTMP(*)
  INTEGER I
  !
  ! restore bond and theta lists.
  !
  IF(ICONB <= 0) RETURN
  !
  DO I = 1,NBOND
     IB(I) = IBTMP(I)
     JB(I) = JBTMP(I)
  ENDDO
  !
  !     store masses and initialize back atom list and map
  !
  DO I = 1,NATOM
     AMASS(I) = AMTMP(I)
  ENDDO
  !     Angles?
  IF(ICONB <= 2) RETURN
  !
  DO I = 1,NTHETA
     IT(I) = ITTMP(I)
     JT(I) = JTTMP(I)
     JT(I) = KTTMP(I)
  ENDDO
  RETURN
END SUBROUTINE PIGGSHK2


SUBROUTINE PIGGINI(NATOM,PIGGLS,BACKLS)
  use chm_kinds
  use stream
  implicit none
  INTEGER NATOM,PIGGLS(*),BACKLS(*)
  INTEGER I

  PIGGLS(1:natom) = 0
  BACKLS(1:natom) = 0

  RETURN
END SUBROUTINE PIGGINI


SUBROUTINE PIGGFIN(PIGGLS,BACKLS,NPIGG,PIGGY,BACK)
  use chm_kinds
  use dimens_fcm
  use stream
  use shake
  implicit none
  INTEGER PIGGLS(*),BACKLS(*)
  INTEGER NPIGG,PIGGY(*),BACK(*)
  INTEGER IPIGG

  DO IPIGG = 1,NPIGG
     PIGGLS(PIGGY(IPIGG)) = 1
     BACKLS(BACK(IPIGG)) = 1
  enddo
  DO IPIGG=1,NPIGG
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,I5,A,I5,A,I5)') &
          'pair: ',IPIGG,' atoms ',PIGGY(IPIGG),'-',BACK(IPIGG)
  enddo
  !     Issue warning if shake is set.
  IF(QSHAKE) CALL WRNDIE(5,'<TSMS>', &
       'REISSUE THE SHAKE COMMAND OR '// &
       'INCORRECT RESULTS WILL BE OBTAINED.')
  RETURN
END SUBROUTINE PIGGFIN


#else /* (tsms_main)*/
SUBROUTINE TSMS
  CALL WRNDIE(-5,'<TSMS>','TSM code is not compiled.')
  return
end SUBROUTINE TSMS
#endif /* (tsms_main)*/

