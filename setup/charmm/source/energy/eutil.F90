module eutil

contains

  SUBROUTINE GETE(X,Y,Z,XREF,YREF,ZREF,MASSW, &
       qomm_arg, qblade_arg &
#if KEY_DHDGB==1
               ,SDEF &
#endif
)
  !-----------------------------------------------------------------------
  ! THIS ROUTINE CALLS ENERGY WITH THE STANDARD ARRAYS AND FILLS
  ! ARRAYS LIKE DX,DY,DZ AND ALL THE ENERGY TERMS.
  ! IT HAS A SHORT CALLING SEQUENCE FOR SIMPLICITY.
  ! B R BROOKS - 10/82
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc
  use number
  use psf
  use bases_fcm
  use deriv
  use image
  use shake
  use memory
  use energym
  use pme_module,only:qpme
  use holonom,only:holonoma,holonomf
#if KEY_OPENMM==1
  use omm_main, only: omm_eterm_mask
#endif /* KEY_OPENMM */

#if KEY_BLADE==1
  use blade_main, only: blade_eterm_mask
#endif /* KEY_BLADE */
  
  use block_ltm, only: NOFORC
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec, q_split
  use domdec_d2d_comm,only:copy_to_all
  use domdec_dr_common,only:q_direct_node, q_recip_node, copy_to_recip, &
      start_split_direct_recip, stop_split_direct_recip
  use domdec_d2r_comm,only:send_coord_to_recip, send_stop_recip
  use domdec_bonded,only:check_bonded_14
#if KEY_LONEPAIR==1
  use domdec_lonepair,only:check_lonepr
#endif
  use domdec_aniso,only:check_aniso
  use energy_util,only:energy_recip
  use domdec_local,only:update_local_coord
#endif
#if KEY_DHDGB==1
  use derivdhdgb
  use dhdgb,only:qfhdgb
#endif
  implicit none

  real(chm_real) :: X(:), Y(:), Z(:)
  real(chm_real) :: XREF(:), YREF(:), ZREF(:)
#if KEY_DHDGB==1
  real(chm_real), intent (in), optional :: SDEF(:)
#endif
  INTEGER MASSW
  logical, intent(in), optional :: qomm_arg, qblade_arg
  real(chm_real),dimension(NATOM) :: IX, IY, IZ
  real(chm_real),dimension(NATOM) :: IXR, IYR, IZR
  LOGICAL LMASS,QOK
  INTEGER NSHKIT
  integer ierror, i
  logical :: want_openmm, want_blade = .false.
  logical :: qeterm_bak(LENENT)

#if KEY_DOMDEC==1
  if (q_domdec) then
     call start_split_direct_recip()
     if (.not.q_direct_node .and. q_recip_node) then
        ! Pure reciprocal nodes stay here and exit after energy is done
        call energy_recip(x, y, z, dx, dy, dz, .false.)
        call copy_to_recip(dx, dy, dz)
        call stop_split_direct_recip()
        return
     endif
  endif
#endif

  want_openmm = .false.
  if (present(qomm_arg)) want_openmm = qomm_arg

  ! XXX adding argument to ENERGY would be better but too hard - mkg 2012
#if KEY_OPENMM==1
  if (want_openmm) then
     qeterm_bak = QETERM
     QETERM = QETERM .and. .not. omm_eterm_mask()
  endif
#endif /* KEY_OPENMM */

  if (present(qblade_arg)) want_blade = qblade_arg
  
#if KEY_BLADE==1
  if (want_blade) then
     qeterm_bak = QETERM
     QETERM = QETERM .and. .not. blade_eterm_mask()
  endif
#endif /* KEY_BLADE */
  
  IF(QHOLO) THEN

     LMASS=(MASSW.EQ.1)
     IX(1:NATOM) = X(1:NATOM)
     IY(1:NATOM) = Y(1:NATOM)
     IZ(1:NATOM) = Z(1:NATOM)
     IXR(1:NATOM) = XREF(1:NATOM)
     IYR(1:NATOM) = YREF(1:NATOM)
     IZR(1:NATOM) = ZREF(1:NATOM)

     if (.not. want_blade) & ! call energy doesn't need positions for blade
     CALL HOLONOMA(X,Y,Z,IXR,IYR,IZR, &
          LMASS,.FALSE.,QOK)

#if KEY_DOMDEC==1
     if (q_domdec) then
        ! Communicate coordinates to recip CPUs
        if (q_split) then
           call send_coord_to_recip(x, y, z, .true.)
        endif
        ! Locate xyzq_loc
        call update_local_coord(x, y, z)
     endif
#endif

     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1 &
#if KEY_DHDGB==1
          ,SDEFin=SDEF,DS_DHDGBout=DS_DHDGB &
#endif
)

     if (.not. want_blade) & ! these forces should be zero anyways for blade
     CALL HOLONOMF(DX,DY,DZ,X,Y,Z,LMASS,.FALSE.,QOK)

     X(1:NATOM) = IX(1:NATOM)
     Y(1:NATOM) = IY(1:NATOM)
     Z(1:NATOM) = IZ(1:NATOM)

  ELSE

#if KEY_DOMDEC==1
     if (q_domdec) then
        ! Communicate coordinates to recip CPUs
        if (q_split) then
           call send_coord_to_recip(x, y, z, .true.)
        endif
        ! Locate xyzq_loc
        call update_local_coord(x, y, z)
     endif
#endif

     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1 &
#if KEY_DHDGB==1
          ,SDEFin=SDEF,DS_DHDGBout=DS_DHDGB &
#endif
)
     !! Bernie suggested to move this endif below the broadcast:
  ENDIF
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
  if(.NOT.NOFORC) then
        CALL VDGBR(DX,DY,DZ,1)
  endif
#if KEY_DOMDEC==1
     endif
#endif
#endif
!  ENDIF ! here: Bernie's suggestion

  if (want_openmm) then
     QETERM = qeterm_bak
  endif

  if (want_blade) then
     QETERM = qeterm_bak
  endif
  
  ! -------------------------------------------------------------------------------
  ! Check the number of bonded interactions in domdec
#if KEY_DOMDEC==1
  if (q_domdec) then
     call copy_to_all(dx, dy, dz)
     call check_bonded_14()
#if KEY_LONEPAIR==1
     call check_lonepr()
#endif
     call check_aniso()
  endif
#endif
  ! -------------------------------------------------------------------------------

#if KEY_DOMDEC==1
  if (q_domdec) then
     if (q_split) then
        call send_stop_recip()
        call copy_to_recip(dx, dy, dz)
     endif
     call stop_split_direct_recip()
  endif
#endif

  RETURN
END SUBROUTINE GETE

SUBROUTINE FILUPT(IUPT,NDIM)
  !-----------------------------------------------------------------------
  ! THIS ROUTINE FILLS THE SECOND DERIVATIVE UPPER TRIANGLE
  ! POINTER OFFSET ARRAY.
  !
  ! BERNARD R. BROOKS     FEBRUARY,1989
  !
  use chm_kinds
  implicit none
  INTEGER IUPT(*),NDIM

  INTEGER I,IPT

  IPT=0
  DO I=1,NDIM
     IUPT(I)=IPT
     IPT=IPT+NDIM-I
  ENDDO

  RETURN
END SUBROUTINE FILUPT

SUBROUTINE SKIPE(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  ! Select the energy terms to be calculated. B R BROOKS - 4/83
  !
  use chm_kinds
  use dimens_fcm
  use energym
  use stream
  use quantm
  use machutil,only:die
  use string
#if KEY_OPENMM==1
  use omm_ctrl, only: omm_system_changed
#endif

  implicit none
  ! Passed variables.
  CHARACTER(len=*) :: COMLYN
  INTEGER   COMLEN
  ! Local variables.
  CHARACTER(len=4) :: CETEMP(LENENT), WORD
  INTEGER   I, NINCL
  LOGICAL   SKIP, SKIPN, DONE, FOUND

  SKIP = .FALSE.
  DONE = .FALSE.

  DO WHILE (.NOT. DONE)
     WORD = NEXTA4(COMLYN, COMLEN)
     IF (WORD .EQ. '    ') THEN
        !---- Procedure FINISH-UP
        IF(PRNLEV.GE.2) WRITE (OUTU,'(A)') &
             ' SKIPE> The following energy terms will be computed :'
        NINCL = 0
        DO I = 1,LENENT
           IF (CETERM(I) .NE. '    ' .AND. QETERM(I)) THEN
              NINCL = NINCL + 1
              CETEMP(NINCL) = CETERM(I)
           ENDIF
        ENDDO
        IF(PRNLEV.GE.2) WRITE(OUTU,'(1X,15A5)') &
             (CETEMP(I), I = 1, NINCL)
#if KEY_OPENMM==1
        call omm_system_changed()
#endif
        RETURN
        !----
     ELSE IF (WORD .EQ. 'INCL') THEN
        SKIP = .FALSE.
     ELSE IF (WORD .EQ. 'EXCL') THEN
        SKIP = .TRUE.
     ELSE IF (WORD .EQ. 'INIT') THEN
        DO I = 1,LENENT
           QETERM(I) = .TRUE.
        ENDDO
#if KEY_QUANTUM==1
        DO I = 1,LENQEQ
           QEQTRM(I) = .TRUE.
        ENDDO
#endif
        RETURN
     ELSE IF (WORD .EQ. 'ALL ') THEN
        DO I = 1,LENENT
           QETERM(I) = SKIP
        ENDDO
#if KEY_QUANTUM==1
        DO I = 1,LENQEQ
           QEQTRM(I) = SKIP
        ENDDO
#endif
     ELSE IF (WORD .EQ. 'NONE') THEN
        SKIPN = .NOT. SKIP
        DO I = 1,LENENT
           QETERM(I) = SKIPN
        ENDDO
#if KEY_QUANTUM==1
        DO I = 1,LENQEQ
           QEQTRM(I) = SKIPN
        ENDDO
#endif
     ELSE
        FOUND = .FALSE.
        DO I = 1,LENENT
           IF (CETERM(I) .EQ. WORD) THEN
              QETERM(I) = SKIP
              FOUND    = .TRUE.
           ENDIF
        ENDDO

#if KEY_QUANTUM==1
        IF(.NOT.FOUND) THEN
           DO I = 1,LENQEQ
              IF(CEQTRM(I) .EQ. WORD) THEN
                 QEQTRM(I) = SKIP
                 FOUND    = .TRUE.
              ENDIF
           ENDDO
        ENDIF
#endif

        IF (.NOT.(FOUND)) THEN
           IF(WRNLEV.GE.2) WRITE (OUTU,'(A,A4,A)') &
                ' SKIPE>  **** ERROR **** Unrecognised energy term = ', &
                WORD, '.'
           CALL DIEWRN(0)
           !---- Procedure FINISH-UP
           IF(PRNLEV.GE.2) WRITE (OUTU,'(A)') &
                ' SKIPE> The following energy terms will be computed :'
           NINCL = 0
           DO I = 1,LENENT
              IF (CETERM(I) .NE. '    ' .AND. QETERM(I)) THEN
                 NINCL = NINCL + 1
                 CETEMP(NINCL) = CETERM(I)
              ENDIF
           ENDDO
           IF(PRNLEV.GE.2) WRITE(OUTU,'(1X,15A5)') &
                (CETEMP(I), I = 1, NINCL)
#if KEY_QUANTUM==1
           IF(NATQM.GT.0) THEN
              IF(PRNLEV.GE.2) WRITE (OUTU,'(A)') &
                   ' SKIPE> The following quantum energy terms will be computed :'
              NINCL = 0
              DO I = 1,LENQEQ
                 IF(CEQTRM(I) .NE. '    ' .AND. QEQTRM(I)) THEN
                    NINCL = NINCL + 1
                    CETEMP(NINCL) = CEQTRM(I)
                 ENDIF
              ENDDO
              IF(PRNLEV.GE.2) WRITE(OUTU,'(1X,15A5)') &
                   (CETEMP(I), I = 1,NINCL)
           ENDIF
#endif
           RETURN
           !----
        ENDIF
     ENDIF

  ENDDO
  CALL DIE

END SUBROUTINE SKIPE

SUBROUTINE FINCYC(NCYCLE,IHTFRQ,IEQFRQ,NTRFRQ,ISVFRQ, &
     INBFRQ,IHBFRQ,ILBFRQ,IMGFRQ,NSAVC,NPHFREQ,NSTEP &
#if KEY_TMD==1
     ,inrt &
#endif
     )
  !-----------------------------------------------------------------------
  ! THIS ROUTINE FINDS THE LOWEST VALUE FOR THE EIGHT UPDATE
  ! FREQUENCY VARIABLES. IF SOME UPDATE VALUES ARE NOT
  ! MULTIPLES OF THE MOST FREQUENT UPDATE CYCLE, THEN A WARNING IS
  ! ISSUED AND THE HIGHER VALUE IS ROUNDED DOWN TO THE NEAREST
  ! MULTIPLE OF THE SMALLEST UPDATE FREQUENCY. ZEROES ARE IGNORED.
  ! B. R. BROOKS  8/11/83
  !
  use chm_kinds
  use stream
  implicit none

  INTEGER NCYCLE,IHTFRQ,IEQFRQ,NTRFRQ,ISVFRQ,INBFRQ
  INTEGER IHBFRQ,ILBFRQ,IMGFRQ,NSAVC,NSTEP,NPHFREQ

  INTEGER NWARN,NSMALL,I,J

#if KEY_TMD==1
  INTEGER,PARAMETER :: NFRQ=12
#else
  INTEGER,PARAMETER :: NFRQ=11
#endif
  integer inrt
  INTEGER IFRQ(NFRQ),JFRQ(NFRQ)
  CHARACTER(len=8) :: AFRQ(NFRQ)
  DATA AFRQ/'NSTEP   ','INBFRQ  ','IHBFRQ  ','IEQFRQ  ','IHTFRQ  ', &
       'NTRFRQ  ','ISVFRQ  ','ILBFRQ  ','IMGFRQ  ','NSAVC   ', 'NPHFREQ ' &
#if KEY_TMD==1
  ,'INRT    '  &
#endif
  /

  IFRQ(1)=NSTEP
  IFRQ(2)=INBFRQ
  IFRQ(3)=IHBFRQ
  IFRQ(4)=IEQFRQ
  IFRQ(5)=IHTFRQ
  IFRQ(6)=NTRFRQ
  IFRQ(7)=ISVFRQ
  IFRQ(8)=ILBFRQ
  IFRQ(9)=IMGFRQ
  IFRQ(10)=NSAVC
  IFRQ(11)=NPHFREQ
#if KEY_TMD==1
  IFRQ(12)=INRT
#endif

  NWARN=0
  NSMALL=0
  DO I=1,NFRQ
     JFRQ(I)=0
     IF(IFRQ(I).GT.0) THEN
        IF(NSMALL.EQ.0 .OR. IFRQ(I).LT.NSMALL) NSMALL=IFRQ(I)
     ENDIF
  ENDDO

  IF(NSMALL.EQ.0) RETURN
  NCYCLE=NSMALL

  DO I=1,NFRQ
     IF(IFRQ(I).GT.0) THEN
        J=IFRQ(I)/NSMALL
        IF (J .EQ. 0) J = 1
        J=J*NSMALL
        IF(J.NE.IFRQ(I)) NWARN=NWARN+1
        JFRQ(I)=J
     ENDIF
  ENDDO

  IF(NWARN.GT.0) THEN
     CALL WRNDIE(1,'<FINCYC>', &
          'SMALLEST UPDATE FREQUENCY IS NOT A COMMON DENOMINATOR')
     IF(WRNLEV.GE.2) WRITE(OUTU,35)
     DO I=1,NFRQ
        IF(IFRQ(I).GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,45) &
             AFRQ(I),IFRQ(I),JFRQ(I)
     ENDDO
  ENDIF
35 FORMAT(12X,'OLD',4X,'NEW')
45 FORMAT(2X,A8,I5,2X,I5)

  IF (NSTEP  .GT. 0) NSTEP  = JFRQ(1)
  IF (INBFRQ .GT. 0) INBFRQ = JFRQ(2)
  IF (IHBFRQ .GT. 0) IHBFRQ = JFRQ(3)
  IF (IEQFRQ .GT. 0) IEQFRQ = JFRQ(4)
  IF (IHTFRQ .GT. 0) IHTFRQ = JFRQ(5)
  IF (NTRFRQ .GT. 0) NTRFRQ = JFRQ(6)
  IF (ISVFRQ .GT. 0) ISVFRQ = JFRQ(7)
  IF (ILBFRQ .GT. 0) ILBFRQ = JFRQ(8)
  IF (IMGFRQ .GT. 0) IMGFRQ = JFRQ(9)
  IF (NSAVC  .GT. 0) NSAVC  = JFRQ(10)
  IF (NPHFREQ .GT. 0) NPHFREQ = JFRQ(11)
#if KEY_TMD==1
!LNI bugfix August 2016.
  IF (INRT   .GT. 0) INRT   = JFRQ(12)
#endif
  !
  ! TEST FOR PROPER RELATIONSHIP FOR IMAGE AND NONBOND UPDATES
  !
  IF(IMGFRQ.GT.0) THEN
     IF (INBFRQ.NE.0) THEN
        IF(IMGFRQ.NE.((IMGFRQ/INBFRQ)*INBFRQ)) THEN
           CALL WRNDIE(-2,'<FINCYC>', &
                'IMGFRQ is not a multiple of INBFRQ')
        ENDIF
     ELSE
        CALL WRNDIE(-2,'<FINCYC>', &
             'INBFRQ is zero when IMGFRQ is not')
     ENDIF
  ENDIF

  RETURN
END SUBROUTINE FINCYC

SUBROUTINE ENERIN
  !-----------------------------------------------------------------------
  ! Initialise the Energy Data Structure.
  !
  use chm_kinds
  use dimens_fcm
  use number

  use econtmod
  use energym
  use epert_mod
  use quantm
#if KEY_SQUANTM==1
  use squantm
#endif
  implicit none
  ! Local variables.
  INTEGER I

  ! Initialize energy analysis flags
  QECONT=.FALSE.
  QATERM=.FALSE.

  ! Initialise the properties arrays and the energy terms arrays.
  DO I = 1,LENENP
     CEPROP(I) = '    '
     QEPROP(I) = .TRUE.
     EPROP(I)  = ZERO
#if KEY_PERT==1
     EPPRTM(I) = ZERO
     EPPRTL(I) = ZERO
     EPPRTD(I) = ZERO
     EPPRTA(I) = ZERO
     EPPRTF(I) = ZERO
     EPPRTT(I) = ZERO
     QEPPRT(I) = .TRUE.
#endif
  ENDDO
  !yw...17_July-2002, Initialize CETERM and ETERM
  DO I=1,LENENT
     CETERM(I) = '    '
     ETERM(I) = ZERO
  ENDDO

  ! Initialise the remaining variables.
  ECALLS = 0
  TOT1ST = 0
  TOT2ND = 0
  EOLD   = ZERO
  !
  ! Initialize the energy property names.
  !
  ! !!! Note: This list must be kept up to date and it also must
  !     exactly match the comments in fcm/energy.f90 !!!
  !
  CEPROP(TOTE)   = 'TOTE'  ! total energy
  CEPROP(TOTKE)  = 'TOTK'  ! total kinetic energy
  CEPROP(EPOT)   = 'ENER'  ! total potential energy
  CEPROP(TEMPS)  = 'TEMP'  ! temperature (from KE)
  CEPROP(GRMS)   = 'GRMS'  ! rms gradient
  CEPROP(BPRESS) = 'BPRE'  ! boundary pressure applied
  CEPROP(HFCTE)  = 'HFCT'  ! total energy with HFC
  CEPROP(HFCKE)  = 'HFCK'  ! total kinetic energy with HFC
  CEPROP(EHFC)   = 'EHFC'  ! high frequency correction energy
  CEPROP(EWORK)  = 'EHYS'  ! slow growth hysteresis energy correction
  CEPROP(VOLUME) = 'VOLU'  ! the volume of the system.
  CEPROP(PRESSE) = 'PRSE'  ! the pressure calculated from the external virial.
  CEPROP(PRESSI) = 'PRSI'  ! the pressure calculated from the internal virial.
  CEPROP(VIRE)   = 'VIRE'  ! the external virial.
  CEPROP(VIRI)   = 'VIRI'  ! the internal virial.
  CEPROP(VIRKE)  = 'VIRK'  ! the virial "kinetic energy".
#if KEY_ACE==1
  CETERM(SELF)   = 'SELF'  ! electrostatic self energy
  CETERM(SCREEN) = 'SCRE'  ! screening of coulombic interactions
  CETERM(COUL)   = 'COUL'  ! coulomb interactions
  CEPROP(SOLV)   = 'SOLV'  ! total solvation energy
  CEPROP(INTER)  = 'INTE'  ! total interaction energy
#endif
  ! SAPATEL
#if KEY_CHEQ==1
  CEPROP(CGKE)   = 'CGKE'  ! Charge kinetuc energy
  CEPROP(CGPOT)  = 'CGEP'  ! Charge potential energy
  CEPROP(ALLE)   = 'ALLE'  ! Total energy with FQ
  CEPROP(ALLP)   = 'ALLP'  ! Total Potential energy with FQ
  CEPROP(ALLK)   = 'ALLK'  ! Total Kinetic energy with FQ
  CEPROP(SDMIN)  = 'SD'
#endif
  ! SAPATEL

#if KEY_FLUCQ==1
  CETERM(FQKIN)  = 'FQKI'  ! fluctuating charge kinetic energy
#endif
#if KEY_PHMD==1
  CETERM(PHKIN)  = 'PHKI'  ! kinetic energy of extended variables in PHMD
#endif
  ! PJ 06/2005
#if KEY_PIPF==1
  CEPROP(DIPK)   = 'DIPK'  ! Dipole kinetic energy
  CEPROP(DIPT)   = 'DIPT'  ! Dipole temperature
#endif
  !
  ! Initialize the energy term names.
  !
  ! !!! Note: This list must be kept up to date and is also must
  !     exactly match the comments in fcm/energy.f90 !!!
  !
  CETERM(BOND)   = 'BOND'  ! bond (1-2) energy
  CETERM(ANGLE)  = 'ANGL'  ! angle (1-3) energy
  CETERM(UREYB)  = 'UREY'  ! additional 1-3 urey bradley energy
  CETERM(DIHE)   = 'DIHE'  ! dihedral 1-4 energy
  CETERM(IMDIHE) = 'IMPR'  ! improper planar of chiral energy
  CETERM(VDW)    = 'VDW '  ! van der waal energy
  CETERM(ELEC)   = 'ELEC'  ! electrostatic energy
  CETERM(HBOND)  = 'HBON'  ! hydrogen bonding energy
  CETERM(USER)   = 'USER'  ! user supplied energy term
  CETERM(CHARM)  = 'HARM'  ! harmonic positional restraint energy
  CETERM(PCHARM) = 'PCHA'  ! principal component harmonic positional restraint energy - clb3 8/21
  CETERM(CDIHE)  = 'CDIH'  ! dihedral restraint energy
  CETERM(CPUCK)  = 'CPUC'  ! puckering restraint energy
  CETERM(CINTCR) = 'CIC '  ! internal coordinate restraint energy
  CETERM(CQRT)   = 'CDRO'  ! droplet restraint energy (approx const press)
#if KEY_HMCOM==1
  CETERM(HMCM)   = 'HMCM'  ! center of masses harmonic constraint
#endif
#if KEY_CPATH==1
  CETERM(PATH)   = 'PATH'  ! PATH restraint
#endif
#if KEY_CMAP==1
  CETERM(CMAP)   = 'CMAP'  ! cross-term maps
#endif
  CETERM(NOE)    = 'NOE'   ! general distance restraint energy (for NOE)
  CETERM(SBNDRY) = 'SBOU'  ! solvent boundary lookup table energy
  CETERM(IMVDW)  = 'IMNB'  ! primary-image van der waal energy
  CETERM(IMELEC) = 'IMEL'  ! primary-image electrostatic energy
  CETERM(IMHBND) = 'IMHB'  ! primary-image hydrogen bond energy
  CETERM(EWKSUM) = 'EWKS'  ! Ewald k-space sum energy term
  CETERM(EWSELF) = 'EWSE'  ! Ewald self energy term
  CETERM(EXTNDE) = 'EXTE'  ! extended electrostatic energy
  CETERM(RXNFLD) = 'RXNF'  ! reaction field electrostatic energy
  CETERM(ST2)    = 'ST2'   ! ST2 water-water energy
  CETERM(IMST2)  = 'IMST'  ! primary-image ST2 water-water energy
  CETERM(TSM)    = 'TSM'   ! TMS free energy term
  CETERM(QMEL)   = 'QMEL'  ! Quantum (QM) energy with QM/MM electrostatics
  CETERM(QMVDW)  = 'QMVD'  ! Quantum (QM/MM) van der Waal term
  CETERM(ASP)    = 'ASP'   ! Atomic solvation parameter (surface) energy
  CETERM(EHARM)  = 'EHAR'  ! Restraint term for Implicit Euler integration
  CETERM(GEO)    = 'GEO '  ! Mean-Field-Potential energy term
  CETERM(MDIP)   = 'MDIP'  ! Dipole Mean-Field-Potential energy term
  CETERM(PINT)   = 'PINT'  ! ???????
#if KEY_RPATH==1
  CETERM(PRMS)   = 'PRMS'  ! Replica/Path RMS deviation energy
#endif
#if KEY_RPATH==1
  CETERM(PANG)   = 'PANG'  ! Replica/Path RMS angle deviation energy
#endif
  CETERM(SSBP)   = 'SSBP'  ! ???????
  CETERM(BK4D)   = 'BK4D'  ! 4-D energy
  CETERM(SHEL)   = 'SHEL'  ! ???????
  CETERM(RESD)   = 'RESD'  ! Restrained Distance energy
  CETERM(SHAP)   = 'SHAP'  ! Shape restraint energy
  CETERM(STRB)   = 'STRB'  ! Strech-Bend coupling energy (MMFF and CFF)
  CETERM(OOPL)   = 'OOPL'  ! Out-off-plane energy (MMFF and CFF)
  CETERM(PULL)   = 'PULL'  ! Pulling force energy
  CETERM(POLAR)  = 'POLA'  ! Polarizable water energy
  CETERM(DMC )   = 'DMC '  ! Distance map restraint energy
  CETERM(RGY )   = 'RGY '  ! Radius of Gyration restraint energy
  CETERM(EWEXCL) = 'EWEX'  ! Ewald exclusion correction energy
  CETERM(EWQCOR) = 'EWQC'  ! Ewald total charge correction energy
  CETERM(EWUTIL) = 'EWUT'  ! Ewald utility energy term (for misc. corrections)
  CETERM(PBELEC) = 'PBEL'  ! Poisson-Boltzmann electrostatic solvation energy
  CETERM(PBNP)   = 'PBNP'  ! Poisson-Boltzmann nonpolar solvation energy
  CETERM(GBEnr)  = 'GBEN'  ! Generalized Born Energy
  CETERM(GSBP)   = 'GSBP'  ! Generalized Solvent Boundary Potential energy
  CETERM(SMBP)   = 'SMBP'  ! Solvent Macormolecule Boundary Potential energy
  CETERM(MBDEFRM)= 'MBDE'  ! Body mode deformation energy
  CETERM(STRSTR) = 'STRS'  ! Stretch-Stretch coupling energy (CFF)
  CETERM(BNDBND) = 'BNDB'  ! Bend-Bend coupling energy (CFF)
  CETERM(BNDTW)  = 'BNDT'  ! Bend-Twist coupling energy (CFF)
  CETERM(EBST)   = 'EBST'  ! End-Bond-Stretch-Twist coupling energy (CFF)
  CETERM(MBST)   = 'MBST'  ! Middle-Bond-Stretch-Twist coupling energy (CFF)
  CETERM(BBT)    = 'BBT '  ! Bend-Bend-Twist coupling energy (CFF)
  CETERM(SST)    = 'SST '  ! Stretch-Stretch-Twist coupling energy (CFF)
  CETERM(VMOD)   = 'VMOD'  ! Normal mode restraint energy
  ! smg/mf MSU 2008/2010
#if KEY_EPMF==1
  CETERM(PMF1D)  = 'PMF1D' ! 1D PMF energy (EPMF)
  CETERM(PMF2D)  = 'PMF2D' ! 2D PMF energy (EPMF)
#endif
#if KEY_PRIMO==1
  CETERM(PRIMO)  = 'PRIMO'  ! PRIMO virtual atom energies (PRIMO)
#endif
#if KEY_EDS==1
  CETERM(EDS)    = 'EDS'    ! Efficient distribution sampling energy
#endif

#if KEY_ADUMB==1
  CETERM(ADUMB)  = 'ADUM'  ! adaptive umbrella potential
#endif
#if KEY_ACE==1
  CETERM(HYDR)   = 'HYDR'  ! non-polar solvation energy
#endif
#if KEY_GRID==1
  Ceterm(GrvdW)  = 'GRVD'  ! Grid-based vdW energy
  Ceterm(GrElec) = 'GREL'  ! Grid-based elec energy
  Ceterm(GrHBON) = 'GRHB'  ! Grid-based hydrogen bond energy !! Yujin
#endif 
#if KEY_FLUCQ==1
  CETERM(FQPOL)  = 'FQPO'  ! fluctuating charge polarisation energy
#endif
#if KEY_PHMD==1
  CETERM(PHENR)  = 'PHEN'  ! Biasing and VDW energy in PHMD
#endif
#if KEY_FACTS==1
  CETERM(IFCTPOL) = 'FCTP'  ! FACTS electrostatic solvation free energy
  CETERM(IFCTNPL) = 'FCTN'  ! FACTS nonpolar solvation free energy
#endif
#if KEY_SASAE==1
  CETERM(SASTRM) = 'SASL'  ! SASA mean solvation energy
#endif
#if KEY_OVERLAP==1
  CETERM(QOVLAP) = 'OLAP'  ! Overlap energy term
#endif
  ! PJ 06/2005
#if KEY_PIPF==1
  CETERM(PIPF)   = 'EPOL'  ! polarization energy term
#endif
#if KEY_LRVDW==1
  CETERM(ELRC)   = 'ELRC'  ! Long range vdw correction
#endif
#if KEY_RXNCOR==1
  CETERM(UMBR)   = 'UMBR'  ! Umbrella potential for RXNCOR
#endif
#if KEY_EMAP==1
      CETERM(EEMAP)   = 'EEMAP'  ! constraint energy from EMAP
#endif
  CETERM(rushRepu) = 'RREP' ! RUSH repulsion energy
  CETERM(rushPhob) = 'RPHO' ! RUSH hydrophobic energy
  CETERM(rushHbnd) = 'RHBN' ! RUSH intra-molecular h-bond energy
  CETERM(rushBdon) = 'RBDO' ! RUSH donor - solvent h-bond energy
  CETERM(rushBacc) = 'RBAC' ! RUSH acceptor - solvent h-bond energy
  CETERM(rushArom) = 'RARO' ! RUSH aromatic - aromatic energy
#if KEY_RDC==1
  CETERM(ERDC)   = 'ERDC' ! residueal dipolar coupling
#endif
#if KEY_SSNMR==1
  CETERM(ECS)   = 'ECS' ! N chemical shift energy term
#endif
#if KEY_RMD==1
  CETERM(CROS) = 'CROS'    ! Surface crossing energy correction
#endif
#if KEY_MRMD==1
  CETERM(MRMD) = 'ERMD'    ! MRMD energy correction
#endif
#if KEY_MMPT==1
      CETERM(MMPT) = 'MMPT'    ! MMPT energy correction
#endif
#if KEY_MSMMPT==1
      CETERM(MSMMPT) = 'MSPT'    ! MSMMPT energy correction
#endif
#if KEY_VALBOND==1
      CETERM(VALB) = 'VALB'    ! VALBOND bending energy
#endif
#if KEY_GAMUS==1
      CETERM(GAMUS)  = 'GAMU'  ! GAMUS potential
#endif
#if KEY_PNM==1
  CETERM(PNME) = 'PNME' ! plastic network model energy
#endif
#if KEY_LARMORD==1
  ceterm(cspres) = 'CSRES'  !chemical shift restraunt force
#endif
#if KEY_CONSHELIX==1
!      CETERM(ECHEL) = 'ECHE' ! Helix-Helix Distance restraint energy
!      CETERM(ECHAN) = 'ECHA' ! Helix-helix Cross angle restraint energy
      CETERM(ECHDL) = 'ECHD' ! Helix-Helix Distance restraint energy
      CETERM(ECHAN) = 'ECHA' ! Helix-helix Cross(Hinge) angle restraint energy
      CETERM(ECHTN) = 'ECHT' ! Helix Tilt angle restraint energy
      CETERM(ECHRN) = 'ECHR' ! Helix Rotation angle restraint energy
#endif
#if KEY_DHDGB==1
      CETERM(DEFE) = 'DEFE' !MEMBRANE DEFORMATION ENERGY
#endif
#if KEY_LJPME==1
      CETERM(LJSEL) = 'LJSE' ! The self term for LJ-PME
      CETERM(LJREC) = 'LJRE' ! The reciprocal space term for LJ-PME
      CETERM(LJEXC) = 'LJEX' ! The excluded term for LJ-PME
#endif

  !
  ! Initialize the pressure/virial terms.
  !
  ! !!! Note: This list must be kept up to date and it also must
  !     exactly match the comments in fcm/energy.f90 !!!
  !
  CEPRSS(VEXX)   = 'VEXX'  ! External virial terms
  CEPRSS(VEXY)   = 'VEXY'
  CEPRSS(VEXZ)   = 'VEXZ'
  CEPRSS(VEYX)   = 'VEYX'
  CEPRSS(VEYY)   = 'VEYY'
  CEPRSS(VEYZ)   = 'VEYZ'
  CEPRSS(VEZX)   = 'VEZX'
  CEPRSS(VEZY)   = 'VEZY'
  CEPRSS(VEZZ)   = 'VEZZ'
  CEPRSS(VIXX)   = 'VIXX'  ! Internal virial terms
  CEPRSS(VIXY)   = 'VIXY'
  CEPRSS(VIXZ)   = 'VIXZ'
  CEPRSS(VIYX)   = 'VIYX'
  CEPRSS(VIYY)   = 'VIYY'
  CEPRSS(VIYZ)   = 'VIYZ'
  CEPRSS(VIZX)   = 'VIZX'
  CEPRSS(VIZY)   = 'VIZY'
  CEPRSS(VIZZ)   = 'VIZZ'
  CEPRSS(PEXX)   = 'PEXX'  ! External pressure terms
  CEPRSS(PEXY)   = 'PEXY'
  CEPRSS(PEXZ)   = 'PEXZ'
  CEPRSS(PEYX)   = 'PEYX'
  CEPRSS(PEYY)   = 'PEYY'
  CEPRSS(PEYZ)   = 'PEYZ'
  CEPRSS(PEZX)   = 'PEZX'
  CEPRSS(PEZY)   = 'PEZY'
  CEPRSS(PEZZ)   = 'PEZZ'
  CEPRSS(PIXX)   = 'PIXX'  ! Internal pressure terms
  CEPRSS(PIXY)   = 'PIXY'
  CEPRSS(PIXZ)   = 'PIXZ'
  CEPRSS(PIYX)   = 'PIYX'
  CEPRSS(PIYY)   = 'PIYY'
  CEPRSS(PIYZ)   = 'PIYZ'
  CEPRSS(PIZX)   = 'PIZX'
  CEPRSS(PIZY)   = 'PIZY'
  CEPRSS(PIZZ)   = 'PIZZ'

#if KEY_QUANTUM==1
  DO I = 1,LENQEQ
     CEQTRM(I) = '    '
     QEQTRM(I) = .TRUE.
     EEQTRM(I) = ZERO
     EQPRA(I) = ZERO
     EQPR2A(I) = ZERO
     EQPRP(I) = ZERO
     EQPR2P(I) = ZERO
  ENDDO
  !
  ! QMPERT = .TRUE.   to perform QM/MM electrostatic perturbation
  ! QDECOM = .TRUE.   to decompose QM/MM energy
  !
  QMPERT = .FALSE.
  QDECOM = .FALSE.
  !
  ! The following correspond to subterms of the quantum mechanical
  ! electronic energy QMEL, which has both purely QM/QM parts and
  ! also QM/MM parts.
  !
  ! QQCC is for QM-QM core interactions
  ! QQEL is the QM electronic energy
  ! QMEE is for MM-charge/QM-valence-electron interactions
  ! QMCH is for MM-charge/QM-core interactions
  ! QATH is for the atomic heats of formation
  !
  ! Note that QQEL and QQEE are not completely separable.  QQEL can
  ! be evaluated without QQEE (which corresponds to the QM system of
  ! atoms in the absence of an external charge), but QQEE cannot be
  ! evaluated without QQEL (which would correspond to the QM valence
  ! electrons in the presence of an external charge but without the
  ! QM core -- the SCF calculations would be very weird).
  !
  ! QPT1 is for QM electrostatic perturbation from L0 -> L1
  ! QPT2 is for QM electrostatic perturbation from L0 -> L2
  ! QVER is the QM/MM vertical interaction energy
  ! QSTA is the QM/MM polarization stabilization energy
  ! QDIS is the QM distortion energy
  ! QPOL is the total QM/MM polarization energy
  ! QGAS is the gas phase QM energy
  !
  CEQTRM(QQCC) = 'QQCC'
  CEQTRM(QQEL) = 'QQEL'
  CEQTRM(QMEE) = 'QMEE'
  CEQTRM(QMCH) = 'QMCH'
  CEQTRM(QATH) = 'QATH'
  CEQTRM(QPT1) = 'QPT1'
  CEQTRM(QPT2) = 'QPT2'
  CEQTRM(QVER) = 'QVER'
  CEQTRM(QSTA) = 'QSTA'
  CEQTRM(QDIS) = 'QDIS'
  CEQTRM(QPOL) = 'QPOL'
  CEQTRM(QGAS) = 'QGAS'
#endif
#if KEY_SQUANTM==1
  !
  ! QPT1 is for QM electrostatic perturbation from L0 -> L1
  ! QPT2 is for QM electrostatic perturbation from L0 -> L2
  !
  DO I = 1,LENQEQ
     EEQTRM(I) = ZERO
     EQPRA(I)  = ZERO
     EQPR2A(I) = ZERO
     EQPRP(I)  = ZERO
     EQPR2P(I) = ZERO
  END DO
#endif
  !=======================================================================
#if KEY_HQBM==1
  ! Added by Emanuele Paci: Initialize HQBM and AFM flag
  ! HQBM.FCM
  HQBM=.FALSE.
#endif
  ! moved to iniall.src: clb3
  ! ##IF AFM
  !! AFM.FCM
  !      LAFM=.FALSE.
  ! ##ENDIF
  !
  RETURN
END SUBROUTINE ENERIN



SUBROUTINE GETE0(OPTION,COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  ! Process the CHARMM ENERGY and GETE commands.
  !

  use new_timer,only:timer_start,timer_stop,T_list
#if KEY_CHEQ==1
  use cheq,only: qcginv,qcgpol,qcginvf,qprneta,cgmodel,qcg,  &
#endif
#if KEY_CHEQ==1
       qdcgn,checketa,checkqnorm, qpartbin, allocate_cheq
#endif

  use chm_kinds
  use dimens_fcm
  use exfunc
  use number

  use code
  use coord
  use coordc
  use energym
  use image
  use psf
  use stream
  use string
  use pert
  use pert_mod
  use fast, only: faster
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec,natoml
  use domdec,only:domdec_com
  use domdec_dr_common,only: q_direct_node
#endif
#if KEY_DOMDEC==1
  use inbnd,only:cutnb
#endif
  use heurist,only:updeci
#if KEY_OPENMM==1
  use omm_ctrl, only: omm_system_changed, omm_requested
  use omm_main, only: omm_energy
#endif /* KEY_OPENMM */

#if KEY_BLADE == 1
  use blade_ctrl_module, only: blade_requested
  use blade_main, only: blade_energy
#endif /* KEY_BLADE */

#if KEY_DHDGB==1
  use dhdgb
#endif

  implicit none
#if KEY_CHEQ==1
  LOGICAL QCGX
  LOGICAL QCHEQRDPRM, QCHEQNORM
  INTEGER NDCG
#endif
  ! Passed variables.
  CHARACTER(len=*) :: COMLYN, OPTION
  INTEGER   COMLEN
  ! Local variables.
  INTEGER   UNIT
  LOGICAL   QCOMP, QPRINT, QUPALL
#if KEY_DOMDEC==1
  integer ddind
#endif
  logical :: want_openmm, want_blade

  ! Get the COMP option.
  QCOMP = INDXA(COMLYN, COMLEN, 'COMP') .GT. 0

#if KEY_DOMDEC==1
  ! Get the DOMDEC option. NOTE: q_domdec is switched on in domdec_com
  ddind = indxa(comlyn, comlen, 'DOMD')
  if (ddind > 0) then
     call domdec_com(comlyn(ddind:comlen), comlen-ddind+1)
  endif
  if (q_domdec) then
     if (faster .eq. -1) then
        CALL WRNDIE(-5,'<GETE>', 'DOMDec cannot be used with FASTer OFF')
     endif
  else
     natoml = natom
  endif
#endif

  want_openmm = .false.
#if KEY_OPENMM==1
  want_openmm = omm_requested(COMLYN, COMLEN, 'GETE0')
#endif /* KEY_OPENMM */

  want_blade = .false.
#if KEY_BLADE == 1
  want_blade = blade_requested(COMLYN, COMLEN, 'GETE0')
#endif /* KEY_BLADE */

#if KEY_CHEQ==1
  QCGX=(INDXA(COMLYN, COMLEN, 'CHEQ') .GT.0)
  IF (QCGX.AND.(PRNLEV.GT.1)) &
       WRITE(OUTU,'(a)')"<GETE>: CHEQ REQUESTED"
  QCGINV = INDXA(COMLYN, COMLEN, 'CGIN') .GT. 0
  QCGPOL = INDXA(COMLYN, COMLEN, 'POLT') .GT. 0
  QCGINVF = INDXA(COMLYN, COMLEN, 'CGFC') .GT. 0
  QPRNETA = INDXA(COMLYN, COMLEN, 'FQPA') .GT. 0
  CGMODEL=GTRMI(COMLYN,COMLEN,'CGMD',CGMODEL)
  IF (QCGX.OR.QCGINV.OR.QCGPOL) THEN
     if(.not.allocated(qpartbin)) then
        call wrndie(-1,'<eutil>','CHEQ not set-up')
     elseif(natim>natom) then
        call allocate_cheq(natim,ngrp)
     endif
     CALL CHECKETA(QCHEQRDPRM)
     IF (.not.QCHEQRDPRM) THEN
        if(prnlev.gt.1)write(outu,'(2a)') &
             'CHEQ ENERGY HAS BEEN REQUESTED BUT', &
             ' CORRECT PARAMETERS HAVE NOT BEEN READ'
        CALL WRNDIE(-1,'<EUTIL>', &
             'CHEQ PARAMETERS HAVE NOT BEEN READ')
     ENDIF
     CALL CHECKQNORM(QCHEQNORM)
     IF (.not.QCHEQNORM) THEN
        if(prnlev.gt.1)write(outu,'(2a)') &
             'CHEQ ENERGY CALL HAS BEEN REQUESTED BUT', &
             ' CORRECT NORMALIZATION SPECIFICATIONS LACKING'
        CALL WRNDIE(-1,'<GET0>', &
             'CHEQ NORMALIZATION ASSIGNMENT INCORRECT')
     ENDIF
     ! ---- SINCE USER WANTS CHEQ, MAKE SURE NORMALIZATION IS SET UP AND CORRECT
     QCG=QCG.OR.QCGX
     QCG=QCG.OR.QCGINV
     IF(PRNLEV.GT.1)THEN
        IF ( QCG ) &
             WRITE(OUTU,'(a)')"<GETE>: CHEQ REQUESTED"
        IF (QCGINV)  &
             WRITE(OUTU,'(a)')"<GETE>: CG INVERSION REQUESTED"
        IF(QCGINVF) &
             WRITE(OUTU,'(a)')"<GETE>: CG FORCE REQUESTED"
        IF (CGMODEL.NE.0) &
             WRITE(OUTU,'(a,I6)')"<GETE>: CGMODEL SET TO",CGMODEL
     ENDIF
     QCG=QCG.OR.QCGX
     IF (QCG.AND.(.NOT.QCGX)) &
          WRITE(OUTU,'(a)')"<GETE>: CHEQ SET"
     IF(QCGINVF.AND.(PRNLEV.GT.1)) &
          WRITE(OUTU,'(a)')"<GETE>: CG FORCE REQUESTED"
     CGMODEL=GTRMI(COMLYN,COMLEN,'CGMD',CGMODEL)
     IF ((CGMODEL.NE.0).AND.(PRNLEV.GT.1)) &
          WRITE(OUTU,'(a,I6)')"<GETE>: CGMODEL SET TO",CGMODEL
     NDCG=GTRMI(COMLYN,COMLEN,'NDCG',1)
     IF (NDCG.EQ.1) THEN
        QDCGN=.TRUE.
     ELSE
        QDCGN=.FALSE.
     ENDIF
  ELSE
  ENDIF
#endif

  ! Do updates if necessary.
  ! print *,"========== GETE0 calling update",qcomp,option
  if (.not. want_blade) then ! Skip all this if using BLaDE
  call timer_start(T_list)
  QUPALL=(OPTION .EQ. 'ENER')
  IF (QCOMP) THEN
     CALL UPDATE(COMLYN, COMLEN, XCOMP,YCOMP,ZCOMP,WCOMP,QUPALL, &
          .TRUE.,QUPALL,QUPALL,QUPALL, &
          0,[zero],[zero],[zero],[zero],[zero],[zero])
  ELSE
     CALL UPDATE(COMLYN, COMLEN, X,Y,Z,WMAIN,QUPALL, &
          .TRUE.,QUPALL,QUPALL,QUPALL, &
          0,[zero],[zero],[zero],[zero],[zero],[zero])
  ENDIF

#if KEY_DOMDEC==1
  if (q_domdec .and. q_direct_node) then
#endif
     IF(OPTION .NE. 'ENER') THEN
        IF (QCOMP) THEN
           CALL UPDECI(1,XCOMP,YCOMP,ZCOMP,WCOMP,0,&
                (/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
        ELSE
           CALL UPDECI(1,X,Y,Z,WMAIN,0,&
                (/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
        ENDIF
     ENDIF
#if KEY_DOMDEC==1
  endif
#endif
  call timer_stop(T_list)
  endif ! (.not. want_blade)

  IF (OPTION .EQ. 'ENER') THEN
     QPRINT = .TRUE.
  ELSE
     QPRINT = INDXA(COMLYN, COMLEN, 'PRIN') .GT. 0
     QPRINT = .NOT.(INDXA(COMLYN, COMLEN, 'NOPR') .GT. 0)
  ENDIF

#if KEY_PERT==1
  IF(QPERT) CALL PERTDF
#endif

  ! Calculate the energy and first derivatives.
  IF (QCOMP) THEN
     IF(NATOMT.GT.MAXA) CALL WRNDIE(-3,'<GETEO>', &
          'Atom limit exceeded for the GETE command using COMP and images')
     CALL GETE(XCOMP, YCOMP, ZCOMP, XCOMP, YCOMP, ZCOMP, 0, &
          want_openmm, want_blade &
#if KEY_DHDGB==1
          ,SDEF &
#endif
)
#if KEY_OPENMM==1
     if (want_openmm) call omm_energy(XCOMP, YCOMP, ZCOMP)
#endif /* KEY_OPENMM */

#if KEY_BLADE==1
     if (want_blade) call blade_energy(XCOMP, YCOMP, ZCOMP)
#endif /* KEY_BLADE */
  ELSE
     CALL GETE(X, Y, Z, X, Y, Z, 0, &
          want_openmm, want_blade &
#if KEY_DHDGB==1
          ,SDEF &
#endif
)
#if KEY_OPENMM==1
     if (want_openmm) call omm_energy(X, Y, Z)
#endif /* KEY_OPENMM */

#if KEY_BLADE==1
     if (want_blade) call blade_energy(X, Y, Z)
#endif /* KEY_BLADE */
  ENDIF

  ! Do some printing.
  IF (QPRINT) THEN
     UNIT = GTRMI(COMLYN, COMLEN, 'UNIT', OUTU)
     IF(UNIT.EQ.OUTU .AND. PRNLEV.LT.2) QPRINT=.FALSE.
     IF(UNIT.NE.OUTU .AND. IOLEV.LT.0) QPRINT=.FALSE.
     IF(QPRINT) CALL PRINTE(UNIT, EPROP, ETERM, 'ENER', 'ENR', &
          .TRUE.,0, ZERO, ZERO, .TRUE.)
  ENDIF

#if KEY_PERT==1
  IF(QPERT) CALL PERTAN(.FALSE.)
#endif
#if KEY_CHEQ==1
  ! ------Turn off all options, leave QCG alone -------
  QCGINV  = .FALSE.
  QCGPOL  = .FALSE.
  QCGINVF = .FALSE.
  QPRNETA = .FALSE.
  CGMODEL = 1

#endif

  RETURN
END SUBROUTINE GETE0

#if KEY_BLOCK==1 /*ldm_routines*/
SUBROUTINE SETFLA(BFLAMB,NBLOCK)
  !-----------------------------------------------------------------------
  ! THIS ROUTINE INITIALIZES LAMBDA FORCES
  ! JAMES KONG
  use chm_kinds
  implicit none
  INTEGER NBLOCK
  real(chm_real) BFLAMB(*)
  ! LOCAL
  INTEGER I

  ! BEGIN
  DO I=1, NBLOCK
     BFLAMB(I)=0.0
  ENDDO
  RETURN
END SUBROUTINE SETFLA

SUBROUTINE COMFLA(BFLAMB,NBLOCK, &
     BXLAMB,BVLAMB,BMLAMB,BELAMB &
     ,LSTRT &
     ,GAMMALD,BIBLAM,ILALDM,FRAND &
     )
  !-----------------------------------------------------------------------
  ! THIS ROUTINE INITIALIZES LAMBDA FORCES - JAMES KONG
  !
  ! Note LAGMUL is not used in this routine, it was used
  ! (commented out already) in case of linear scaling.
  use chm_kinds
  implicit none
  INTEGER NBLOCK
  integer lstrt
  real(chm_real) BFLAMB(*), LAGMUL
  real(chm_real) BXLAMB(*),BVLAMB(*),BMLAMB(*), BELAMB(*)
  real(chm_real) BIBLAM(*),GAMMALD(*)
  LOGICAL ILALDM
  real(chm_real) FRAND(NBLOCK), ERAN
      real(chm_real) E_NUM, EDEN
  ! LOCAL
  INTEGER I

  ! BEGIN
  ! Total number of lambdas = Nblock -1 since first block is the environment
  ! *************************************
  ! linear scaling of energy case
  ! LAGMUL = LAGMUL/FLOAT(NBLOCK - 1)
  ! f(j) = - v(j) + Sum_v(j)/n
  ! DO 10   I=2, NBLOCK
  !    BFLAMB(I) = - BFLAMB(I) + LAGMUL
  ! 10 CONTINUE
  ! *************************************
  ! quadratic scaling: calculating lagrange multiplier
      E_NUM = 0.0
  EDEN = 0.0
  ERAN = 0.0
  IF(ILALDM) THEN
     CALL DLNGV2(GAMMALD,NBLOCK,LSTRT,FRAND)
  ENDIF
  DO I = LSTRT, NBLOCK
     BFLAMB(I) = BFLAMB(I) - BELAMB(I)
  ENDDO
  !
  ! --- main loop ---
  !
  DO I = LSTRT, NBLOCK
     IF(ILALDM) THEN
        ERAN = ERAN - BXLAMB(I)*(FRAND(I)-BIBLAM(I)*BVLAMB(I))
     ENDIF
         E_NUM = E_NUM + 2.0*BXLAMB(I)*BXLAMB(I)*BFLAMB(I) &
          - BMLAMB(I)*BVLAMB(I)*BVLAMB(I)
     EDEN = EDEN + BXLAMB(I)*BXLAMB(I)
  ENDDO
      E_NUM = E_NUM/(2.*EDEN)
  IF(ILALDM) ERAN = ERAN/(2.*EDEN)
  DO I = LSTRT, NBLOCK
         BFLAMB(I) = 2.*BXLAMB(I)*(E_NUM - BFLAMB(I))
     IF(ILALDM)THEN
        BFLAMB(I)=BFLAMB(I)+2.*BXLAMB(I)*ERAN + FRAND(I)
     ENDIF
  ENDDO
  ! --- main loop end ---
  ! *************************************
  RETURN
END SUBROUTINE COMFLA


SUBROUTINE SAVPOTEN(BFLAMB,BPTNLAMB,BELAMB,NBLOCK,LSTRT)
  ! banba shinichi
  use chm_kinds
  implicit none
  INTEGER NBLOCK, LSTRT, I
  real(chm_real) BFLAMB(*), BPTNLAMB(*), BELAMB(*)

  DO I = LSTRT, NBLOCK
     BPTNLAMB(I)=BFLAMB(I)-BELAMB(I)
  ENDDO
  RETURN
END SUBROUTINE SAVPOTEN

SUBROUTINE COMFLA2(BFLAMB,NBLOCK, &
     BXLAMB,BVLAMB,BMLAMB,BELAMB,LMDCOEF &
     ,NRST,BFRST,LSTRT &
     ,GAMMALD,BIBLAM,ILALDM,FRAND         &
     )
  !-----------------------------------------------------------------------
  ! This routine is for restraining potential which call at dynamics loop
  use chm_kinds
  implicit none
  INTEGER NBLOCK, LSTRT
  real(chm_real) BFLAMB(*),LMDCOEF, BFRST(*)
  real(chm_real) BXLAMB(*),BVLAMB(*),BMLAMB(*), BELAMB(*)
  INTEGER NRST
  real(chm_real) BIBLAM(*),GAMMALD(*)
  LOGICAL ILALDM
  real(chm_real) FRAND(NBLOCK), ERAN
      real(chm_real) E_NUM, EDEN
  ! LOCAL
  INTEGER I, J
      E_NUM = 0.0
  EDEN = 0.0
  ERAN = 0.0
  IF(ILALDM) THEN
     CALL DLNGV2(GAMMALD,NBLOCK,LSTRT,FRAND)
  ENDIF

  DO I = LSTRT, NBLOCK
     BFLAMB(I) = BFLAMB(I) - BELAMB(I)
     IF(NRST.EQ.2)  &
          BFRST(I) = BFRST(I) -  BELAMB(I)
  ENDDO
  ! --- main loop start ---
  DO I = LSTRT, NBLOCK
     IF(ILALDM) THEN
        ERAN = ERAN - BXLAMB(I)*(FRAND(I)-BIBLAM(I)*BVLAMB(I))
     ENDIF
     IF(NRST.NE.2)THEN
            E_NUM = E_NUM + 2.0*BXLAMB(I)*BXLAMB(I)*BFLAMB(I) &
             * (1.d0 - LMDCOEF) &
             - BMLAMB(I)*BVLAMB(I)*BVLAMB(I)
     ELSE
            E_NUM = E_NUM + 2.0*BXLAMB(I)*BXLAMB(I)*BFLAMB(I) &
             -LMDCOEF*2.0*BXLAMB(I)*BXLAMB(I)*BFRST(I) &
             -BMLAMB(I)*BVLAMB(I)*BVLAMB(I)
     ENDIF
     EDEN = EDEN + BXLAMB(I)*BXLAMB(I)
  ENDDO
      E_NUM = E_NUM/(2.*EDEN)
  IF(ILALDM) ERAN = ERAN/(2.*EDEN)
  DO I = LSTRT, NBLOCK
     IF(NRST.NE.2)THEN
            BFLAMB(I)=2.*BXLAMB(I)*(E_NUM-BFLAMB(I) + LMDCOEF*BFLAMB(I) )
     ELSE
            BFLAMB(I)=2.*BXLAMB(I)*(E_NUM-BFLAMB(I) + LMDCOEF*BFRST(I) )
     ENDIF
     IF(ILALDM)THEN
        BFLAMB(I)=BFLAMB(I)+2.*BXLAMB(I)*ERAN + FRAND(I)
     ENDIF
  ENDDO
  ! --- main loop end  ---
  RETURN
END SUBROUTINE COMFLA2

SUBROUTINE FVBIAS(NBIASV, NBVIDI, NBVIDJ, NBCLAS, &
     RRREUP, RRRLOW, RKBIAS, NPBIAS, BFLAMB, BXLAMB)

  use chm_kinds
  implicit none
  INTEGER NBIASV, NBVIDI(*), NBVIDJ(*)
  INTEGER NBCLAS(*), NPBIAS(*), K, I, J
  real(chm_real)  RRREUP(*), RRRLOW(*), RKBIAS(*), BFLAMB(*)
  real(chm_real)  FTEMP, BXLAMB(*), BX2
  DO K = 1, NBIASV
     I = NBVIDI(K)
     IF(NBCLAS(K).EQ.1) THEN
        FTEMP = 0.0
        BX2 = BXLAMB(I)*BXLAMB(I)
        IF(BX2.LT.RRREUP(K)) THEN
           FTEMP = (BX2 - RRREUP(K))**( NPBIAS(K) - 1 )
           FTEMP = RKBIAS(K)*NPBIAS(K)*FTEMP*2.*BXLAMB(I)
           BFLAMB(I) = BFLAMB(I) - FTEMP
        ENDIF
     ELSE IF(NBCLAS(K).EQ.2) THEN
        FTEMP = 0.0
        BX2 = BXLAMB(I)*BXLAMB(I)
        IF(BX2.GT.RRRLOW(K)) THEN
           FTEMP = (BX2 - RRRLOW(K))**( NPBIAS(K) - 1 )
           FTEMP = RKBIAS(K)*NPBIAS(K)*FTEMP*2.*BXLAMB(I)
           BFLAMB(I) = BFLAMB(I) - FTEMP
        ENDIF
     ELSE IF(NBCLAS(K).EQ.3) THEN
        FTEMP = 0.0
        J = NBVIDJ(K)
        ! rrreup(k) stores constant for coupling interactions
        BX2 = BXLAMB(I)*BXLAMB(I) - BXLAMB(J)*BXLAMB(J)
        FTEMP = BX2 - RRREUP(K)
        FTEMP = FTEMP**( NPBIAS(K) - 1 )
        FTEMP = RKBIAS(K)*NPBIAS(K)*FTEMP
        BFLAMB(I) = BFLAMB(I) - FTEMP*2.*BXLAMB(I)
        BFLAMB(J) = BFLAMB(J) + FTEMP*2.*BXLAMB(J)
     END IF
  ENDDO
  RETURN
END SUBROUTINE FVBIAS
#endif /* (ldm_routines)*/

end module eutil
