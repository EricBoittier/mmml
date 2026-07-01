module estats_mod
  use chm_kinds
  use dimens_fcm
  implicit none
  !CHARMM Element source/fcm/estats.fcm $Revision: 1.12 $
  !
  ! Description of variables for energy statistics command
  !    
  ! IENCAL  Current no. of calls to ENERGY
  ! NINCMT  Current no. of data points (no. of selected ener values)
  ! ESTMOD  Period of ener selection cycle (selects every ESTMOD calls) 
  ! AVENER  (Array) Avg of energy (components) over NINCMT points.
  ! VARIAN  (Array) Variance of energy (components).
  ! SQRAVG  (Array) Running avg of square of energies (components).
  ! AVENEP  Average potential energy over NINCMT data points.
  ! VARIAP  Variance of PE
  ! SQRAVP  Running avg of square of PE
  ! LESTAT  True when calculating statistics
  ! ESTRCL  Call after which to begin collecting data points 
  ! MXENCL  Call after which no more data points are collected
  ! LPLAST  True when stats to be printed at end of collection
  ! LSTPRI  True when stats are writing to unit 6
  ! PRNMOD  Period for writing to unit 6
  ! LPRNTU  True when writing stats to file
  ! PRNUNT  Fortran unit onto which stats are to be written
  ! IPRNMD  Period for writing to PRNUNT
  ! LPRNTE  True when writing PE to file
  ! EPRUNT  Fortran unit onto which PE is to be written
  ! EPRNMD  Period for writing PE to EPRUNT
  ! NSTTER  No. of terms (e.g. VDW, ELEC) to have stats calculated
  ! STERMX  Max number of component energy terms
  ! TERMNM  (Array) Names of the selected energy terms
  ! TERMNO  CHARMM number of selected energy terms 
  ! LSTVAR  True if variabilizing results
  ! AMRSNM  (Array) Names of variabilized energy averages
  ! VMRSNM  (Array) Names of variabilized energy variances
  ! UPLIMT  Energy above which point will not be included in calc 
  ! LOLIMT  Lower energy limit
  ! ENLIMT  Default energy limit
  ! NOTRNG  Number of energy values not included (above ENLIMT)
  !
#if KEY_ESTATS==1
  integer,PARAMETER :: STERMX=50
  integer,PARAMETER :: ENLIMT=99999999
  INTEGER NINCMT,IENCAL,ESTMOD,MXENCL,ESTRCL,EPRNMD
  INTEGER PRNMOD,NSTTER,PRNUNT,IPRNMD,TERMNO(STERMX)
  INTEGER EPRUNT,NOTRNG
  !
  CHARACTER(len=4) :: TERMNM(STERMX),AMRSNM(STERMX),VMRSNM(STERMX)
  !
  real(chm_real) AVENER(STERMX),VARIAN(STERMX),SQRAVG(STERMX)
  real(chm_real) AVENEP,VARIAP,SQRAVP,UPLIMT,LOLIMT

  LOGICAL LESTAT,LPLAST,LPRNTU,LSTPRI,LSTVAR,LPRNTE
  !
#endif /* ESTATS*/
  !
contains

  SUBROUTINE ANAL(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use exfunc
    !
    use coord
    use econtmod
    use stream
    use fast
    use psf
    use machdep
    use select
    use string
    implicit none
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    !
    CHARACTER(len=4) WRD
    !
    ! if we reach this routine, we'll need the econt_ltm  arrays so we
    ! allocate them. Check if they are allocated first. cb3
    if(.not.allocated(anslct)) call allocate_econt

100 CONTINUE
    !
    WRD=NEXTA4(COMLYN,COMLEN)
    !
    !-----------------------------------------------------------------------
    IF(WRD.EQ.'ON') THEN
       ! Turn on energy partition analysis
       QECONT=.TRUE.
       FASTER=-1
       LFAST =-1
       QFASTNB = .FALSE.
       IF(PRNLEV.GE.2) WRITE(OUTU,255)
255    FORMAT(' CHARMM: Energy partition analysis will performed.', &
            ' Fast routines disabled.')
       !-----------------------------------------------------------------------
    ELSE IF(WRD.EQ.'OFF') THEN
    ! if we reach this here, we'll don't need the econt_ltm arrays so we
    ! deallocate them. Check if they are allocated first. cb3
    if(allocated(anslct)) call deallocate_econt
       QATERM=.FALSE.
       QECONT=.FALSE.
       FASTER=0
       LFAST=0
       QFASTNB = .FALSE.
       IF(PRNLEV.GE.2) WRITE(OUTU,256)
256    FORMAT(' CHARMM: Energy analysis will not be performed.', &
            ' Default fast routines enabled.')
       RETURN
       !-----------------------------------------------------------------------
    ELSE IF(WRD.EQ.'TERM') THEN
       CALL SELCTA(COMLYN,COMLEN,ANSLCT,X,Y,Z,WMAIN,.TRUE.)
       QATERM=.TRUE.
       QAONLY=.TRUE.
       QANBND=.FALSE.
       FASTER=-1
       LFAST =-1
       QFASTNB = .FALSE.
       QAUNIT=-1
       IF(PRNLEV.GE.2) WRITE(OUTU,257)
257    FORMAT(' CHARMM: Energy term analysis will performed.', &
            ' Fast routines disabled.')
       !-----------------------------------------------------------------------
    ELSE IF(QATERM.AND. WRD.EQ.'ALL') THEN
       ! Include only terms where all atoms are specified
       QAONLY=.TRUE.
       !-----------------------------------------------------------------------
    ELSE IF(QATERM.AND. WRD.EQ.'ANY') THEN
       ! Include terms where any atoms are specified
       QAONLY=.FALSE.
       !-----------------------------------------------------------------------
    ELSE IF(QATERM .AND. WRD.EQ.'NONB') THEN
       ! Include nonbond interactions for analysis
       QANBND=.TRUE.
       !-----------------------------------------------------------------------
    ELSE IF(QATERM .AND. WRD.EQ.'NONO') THEN
       ! Do not include nonbond interactions
       QANBND=.FALSE.
       !-----------------------------------------------------------------------
    ELSE IF(QATERM .AND. WRD.EQ.'UNIT') THEN
       ! get unit number for printing energy terms.
       QAUNIT=NEXTI(COMLYN,COMLEN)
       !-----------------------------------------------------------------------
    ELSE IF(WRD.EQ.' ') THEN
       RETURN
    ELSE
       CALL WRNDIE(0,'<CHARMM>','UNRECOGNIZED ANALYSIS OPTION')
    ENDIF
    !-----------------------------------------------------------------------
    !
    GOTO 100
    !
    !
  END SUBROUTINE ANAL

#if KEY_ESTATS==1 /*running energy_statistics*/
  subroutine energy_anal_iniall()
    lestat=.false.
    nincmt=0
    nstter=0
    varian(1:stermx)=0
    avener(1:stermx)=0
    sqravg(1:stermx)=0
    avenep=0
    variap=0
    sqravp=0
    return
  end subroutine energy_anal_iniall


  SUBROUTINE ESTATS(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    use chm_kinds
    !--mfc--   use estats_mod
    use exfunc
    use stream
    use string
    implicit none
    !
    ! This routine sets up the energy statistics calculations  
    !                                RJ Petrella 10.22.00
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN,I
    ! 
    ! reset counters, flags, accumulation arrays
    DO I = 1,STERMX
       AVENER(I) = 0
       VARIAN(I) = 0
       SQRAVG(I) = 0
       TERMNO(I) = 0
    ENDDO
    AVENEP = 0
    VARIAP = 0
    SQRAVP = 0
    IENCAL = 0 
    NSTTER = 0 
    NINCMT = 0
    NOTRNG = 0
    MXENCL = 0
    !
    LESTAT = .TRUE.
    LPRNTU = .TRUE.
    LSTPRI = .TRUE. 
    LPLAST = .FALSE.
    LPRNTE = .TRUE.
    LSTVAR = .FALSE.
    ESTMOD=GTRMI(COMLYN,COMLEN,'IPRF',1)
    MXENCL=GTRMI(COMLYN,COMLEN,'LENG',0)
    ESTRCL=GTRMI(COMLYN,COMLEN,'SKIP',0)
    PRNMOD=GTRMI(COMLYN,COMLEN,'NPRI',-1)
    IPRNMD=GTRMI(COMLYN,COMLEN,'NUPR',-1)
    PRNUNT=GTRMI(COMLYN,COMLEN,'IUNW',-1)
    EPRNMD=GTRMI(COMLYN,COMLEN,'NEPR',-1)
    EPRUNT=GTRMI(COMLYN,COMLEN,'IUPE',-1)
    UPLIMT=GTRMF(COMLYN,COMLEN,'UPLM',ENLIMT*1._chm_real)
    LOLIMT=GTRMF(COMLYN,COMLEN,'LOLM',-ENLIMT*1._chm_real)
    !  check cycle parameters
    IF (ESTMOD.LT.1)  &
         CALL WRNDIE(-1,'<ESTATS>','BAD MODU')
    !
    IF(IPRNMD.EQ.-1) THEN
       LPRNTU = .FALSE.
    ELSE IF (IPRNMD.LT.-1) THEN 
       CALL WRNDIE(-1,'<ESTATS>','BAD NUPR')
    ELSE IF (MOD(IPRNMD,ESTMOD).NE.0) THEN
       CALL WRNDIE(-1,'<ESTATS>', &
            'UNIT WRITE CYC NUPR NOT A MULTIPLE OF COLLECTION CYC')
    ENDIF
    IF(PRNUNT.EQ.-1) LPRNTU = .FALSE.
    !
    IF(PRNMOD.EQ.-1) THEN
       LSTPRI = .FALSE.
    ELSE IF (PRNMOD.LT.-1) THEN 
       CALL WRNDIE(-1,'<ESTATS>','BAD NPRI')
    ELSE IF (MOD(PRNMOD,ESTMOD).NE.0) THEN
       CALL WRNDIE(-1,'<ESTATS>', &
            'PRINT CYCLE NPRI NOT A MULTIPLE OF COLLECTION CYC')
    ENDIF
    !
    IF(EPRNMD.EQ.-1) THEN
       LPRNTE = .FALSE.
    ELSE IF (EPRNMD.LT.-1) THEN 
       CALL WRNDIE(-1,'<ESTATS>','BAD NEPR')
    ELSE IF (MOD(IPRNMD,EPRNMD).NE.0) THEN
       CALL WRNDIE(-1,'<ESTATS>', &
            'ENER WRITE CYC NEPR NOT A MULTIPLE OF COLLECTION CYC')
    ENDIF
    IF(EPRUNT.EQ.-1) LPRNTE = .FALSE.
    !
    IF ((ESTMOD.EQ.0).OR.(PRNMOD.EQ.0).OR. &
         (IPRNMD.EQ.0).OR.(EPRNMD.EQ.0))  &
         CALL WRNDIE(-1,'<ESTATS>','PERIOD(S) CANNOT BE ZERO')
    !
    IF (LOLIMT.GE.UPLIMT)  &
         CALL WRNDIE(-1,'<ESTATS>','BAD ENERGY LIMITS')
    IF(INDXA(COMLYN,COMLEN,'VARI').GT.0) LSTVAR = .TRUE.
    IF(INDXA(COMLYN,COMLEN,'FPRI').GT.0) LPLAST = .TRUE.
    IF(INDXA(COMLYN,COMLEN,'STOP').GT.0) CALL STOPSTAT
    IF (LESTAT) THEN 
       IF(PRNLEV.GE.2) WRITE(OUTU,100)
100    FORMAT('RUNNING ENERGY ANALYSIS HAS BEEN REQUESTED')
       !
       ! energy components to be analyzed 
       IF(INDXA(COMLYN,COMLEN,'BOND').GT.0) THEN 
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='BOND'
          TERMNO(NSTTER)=1 
          AMRSNM(NSTTER)='ABON'
          VMRSNM(NSTTER)='VBON'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'ANGL').GT.0) THEN
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='ANGL'
          TERMNO(NSTTER)=2
          AMRSNM(NSTTER)='AANG'
          VMRSNM(NSTTER)='VANG'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'UREY').GT.0) THEN
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='UREY'
          TERMNO(NSTTER)=3
          AMRSNM(NSTTER)='AURE'
          VMRSNM(NSTTER)='VURE'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'DIHE').GT.0) THEN
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='DIHE'
          TERMNO(NSTTER)=4
          AMRSNM(NSTTER)='ADIH'
          VMRSNM(NSTTER)='VDIH'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'IMPR').GT.0) THEN
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='IMPR'
          TERMNO(NSTTER)=5
          AMRSNM(NSTTER)='AIMP'
          VMRSNM(NSTTER)='VIMP'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'VDW').GT.0) THEN
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='VDW'
          TERMNO(NSTTER)=6
          AMRSNM(NSTTER)='AVDW'
          VMRSNM(NSTTER)='VVDW'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'ELEC').GT.0) THEN
          NSTTER = NSTTER + 1 
          TERMNM(NSTTER)='ELEC'
          TERMNO(NSTTER)=7
          AMRSNM(NSTTER)='AELE'
          VMRSNM(NSTTER)='VELE'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'HBON').GT.0) THEN
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='HBON'
          TERMNO(NSTTER)=8
          AMRSNM(NSTTER)='AHBO'
          VMRSNM(NSTTER)='VHBO'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'USER').GT.0) THEN
          NSTTER = NSTTER + 1 
          TERMNM(NSTTER)='USER'
          TERMNO(NSTTER)=9
          AMRSNM(NSTTER)='AUSE'
          VMRSNM(NSTTER)='VUSE'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'SBOU').GT.0) THEN
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='SBOU'
          TERMNO(NSTTER)=15
          AMRSNM(NSTTER)='ASBO'
          VMRSNM(NSTTER)='VSBO'
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'ASP').GT.0) THEN
          NSTTER = NSTTER + 1
          TERMNM(NSTTER)='ASP'
          TERMNO(NSTTER)=28
          AMRSNM(NSTTER)='AASP'
          VMRSNM(NSTTER)='VASP'
       ENDIF
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)') &
               'STATISTICS FOR THE FOLLOWING TERMS WILL BE CALCULATED:'
          WRITE(OUTU,'(A)') 'EPOT'
          DO I = 1,NSTTER
             WRITE(OUTU,'(A)') TERMNM(I)
          ENDDO
       ENDIF
       LPLAST = .TRUE. 
       WRITE (OUTU,'(A,I14)')  &
            'LENGTH OF COLLECTION = ',MXENCL
       WRITE (OUTU,'(A,I14)') &
            'SKIP LENGTH =          ',ESTRCL
       WRITE (OUTU,'(A,I14)')  &
            'COLLECTION EVERY =     ',ESTMOD
       WRITE (OUTU,'(A,I14)')  &
            'PRINT STATS EVERY =    ',PRNMOD
       WRITE (OUTU,'(A,I14)') &
            'WRITE STATS FILE EVERY ',IPRNMD
       IF (LSTVAR) THEN
          WRITE (OUTU,'(A,A14)') &
               'RESULTS AS VARIABLES   ','YES'
       ELSE
          WRITE (OUTU,'(A,A14)') &
               'RESULTS AS VARIABLES   ','NO'
       ENDIF
       IF (LPLAST) THEN
          WRITE (OUTU,'(A,A14)') &
               'PRINT FINAL RESULTS    ','YES'
       ELSE
          WRITE (OUTU,'(A,A14)') &
               'PRINT FINAL RESULTS    ','NO'
       ENDIF
       IF (LPRNTE) THEN
          WRITE (OUTU,'(A,I14)') &
               'WRITE PE TO FILE EVERY ',EPRNMD
       ENDIF
       IF (MXENCL.LE.0) THEN
          CALL WRNDIE(-1,'<ESTATS>','DATA STREAM LENGTH IS <= ZERO')
       ENDIF !If LESTAT
       DO I = 1,NSTTER
          VARIAN(I) = 0
          AVENER(I) = 0
          SQRAVG(I) = 0
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE ESTATS

  SUBROUTINE ESTACAL
    !-----------------------------------------------------------------------
    use chm_kinds
    !--mfc--   use estats_mod
    use energym
    use param_store, only: set_param

    implicit none
    !
    ! This routine calculates basic statistics (mean and variance)
    ! for a series of energy values during a CHARMM run.
    ! The terms throughout the calculation grow with n^0 or n^(-1)
    ! where n is the number of data points, or energy values 
    ! collected. This avoids accumulating very large terms during 
    ! long data collection runs.
    ! The size of the largest terms are on the order of the variance or 
    ! the square of the energies.
    ! The routine supports the analysis of the potential energy as
    ! well as of the component energy terms, such as VDW, ELEC,etc.
    !                                                RJPetrella 10.22.00
    !
    real(chm_real) ENERCN,ENERSQ,RATIO1,RATIO2
    real(chm_real) AVEINC,NINCP1
    INTEGER I,NUMTER
    LOGICAL INRANG
    ! 
    NINCP1 = (NINCMT+1)
    RATIO2 = (NINCMT-1)/NINCP1
    RATIO1 = NINCMT/NINCP1
    ! check that energy values are in an acceptable range
    INRANG = .TRUE.
    I = 1
    DO WHILE ((I.LE.NSTTER).AND.(INRANG)) 
       NUMTER = TERMNO(I)
       IF ((ETERM(NUMTER).GE.UPLIMT).OR.(ETERM(NUMTER).LE.LOLIMT)) &
            INRANG = .FALSE.
       I = I + 1
    ENDDO
    IF ((EPROP(EPOT).GE.UPLIMT).OR.(EPROP(EPOT).LE.LOLIMT)) &
         INRANG = .FALSE.
    IF (INRANG) THEN
       DO I = 1,NSTTER
          NUMTER = TERMNO(I)
          !
          !  "center" the energy
          ENERCN = ETERM(NUMTER) - AVENER(I)
          !  define some variables 
          ENERSQ = ENERCN*ENERCN 
          ! calculate the variance if data point is not the first one
          IF (NINCMT.GE.1) THEN
             VARIAN(I) = VARIAN(I)*RATIO2 + (ENERSQ + SQRAVG(I)) &
                  /NINCP1
          ENDIF
          ! increment the average energy and update avg sum of squares
          AVEINC = ENERCN/NINCP1
          AVENER(I) = AVENER(I) + AVEINC
          SQRAVG(I) = SQRAVG(I)*RATIO1 + (ENERSQ/NINCP1)  &
               - AVEINC*AVEINC
          ! variabilize if requested
          IF (LSTVAR) THEN
             call set_param(AMRSNM(I),AVENER(I))
             call set_param(VMRSNM(I),VARIAN(I))
          ENDIF
          ! increment data point counter      
       ENDDO
       !    Repeat above for potential energy
       !  "center" the energy
       ENERCN = EPROP(EPOT) - AVENEP
       ENERSQ = ENERCN*ENERCN
       ! calculate the variance if data point is not the first one
       IF (NINCMT.GE.1) THEN
          VARIAP = VARIAP*RATIO2 + (ENERSQ + SQRAVP) &
               /NINCP1
       ENDIF
       ! increment the average energy and update avg sum of squares
       AVEINC = ENERCN/NINCP1
       AVENEP = AVENEP + AVEINC
       SQRAVP = SQRAVP*RATIO1 + (ENERSQ/NINCP1)  &
            - AVEINC*AVEINC
       IF (LSTVAR) THEN
          call set_param('AENE',AVENEP)
          call set_param('VENE',VARIAP)
       ENDIF
       ! increment data point counter      
       NINCMT = NINCMT + 1
    ELSE !not in range
       NOTRNG = NOTRNG + 1
    ENDIF
    RETURN
  END SUBROUTINE ESTACAL

  SUBROUTINE PRNSTAT
    ! ------------------------------------------------------------
    use chm_kinds
    !--mfc--   use estats_mod
    use stream
    implicit none
    !
    real(chm_real) FLUCTU 
    INTEGER I
    !
    IF (NINCMT.GT.0) THEN
       WRITE(OUTU,'(A)') '*****ENERGY STATISTICS*****'
       WRITE(OUTU,'(A9,1X,A14,1X,A17,1X,A17)') 'TERM','NO. POINTS', &
            'AVERAGE','FLUCTUAT'
       FLUCTU = SQRT(VARIAP)
       WRITE(OUTU,'(A11,1X,I14,1X,F17.8,1X,F17.8)') 'ESTAT> EPOT', &
            NINCMT,AVENEP,FLUCTU
       DO I = 1,NSTTER
          FLUCTU = SQRT(VARIAN(I))
          WRITE(OUTU,'(A6,1X,A4,1X,I14,1X,F17.8,1X,F17.8)') 'ESTAT>', &
               TERMNM(I),NINCMT,AVENER(I),FLUCTU
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE PRNSTAT

  SUBROUTINE PRNSTATU
    ! ------------------------------------------------------------
    use chm_kinds
    !--mfc--   use estats_mod
    use stream
    implicit none
    !
    INTEGER I
    real(chm_real) FLUCTU
    !
    IF(NINCMT.EQ.1) THEN
       WRITE(PRNUNT,'(A)') '*****ENERGY STATISTICS*****'
       WRITE(PRNUNT,'(A4,1X,A14,1X,A17,1X,A17)') 'TERM','NO. POINTS', &
            'AVERAGE','FLUCTUAT'
    ENDIF
    IF(NINCMT.GT.0) THEN      
       FLUCTU = SQRT(VARIAP)
       WRITE(PRNUNT,'(A4,1X,I14,1X,F17.8,1X,F17.8)')'EPOT',NINCMT, &
            AVENEP,FLUCTU
       DO I = 1,NSTTER
          FLUCTU = SQRT(VARIAN(I))
          WRITE(PRNUNT,'(A4,1X,I14,1X,F17.8,1X,F17.8)') TERMNM(I),NINCMT, &
               AVENER(I),FLUCTU
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE PRNSTATU

  SUBROUTINE STOPSTAT
    ! ------------------------------------------------------------
    use chm_kinds
    !--mfc--   use estats_mod
    use stream
    use exfunc
    !
    INTEGER I
    IF(PRNLEV.GE.2) WRITE(OUTU,50)
50  FORMAT('STOPPING STATISTICS')
    !
    LESTAT =.FALSE.
    IF (PRNLEV.GE.2) THEN
       IF (NINCMT.EQ.0) THEN
          WRITE(OUTU,'(A)') &
               '<PRNSTAT>: NO DATA POINTS AVAILABLE FOR STATISTICS'
       ELSE IF (LPLAST) THEN
          CALL PRNSTAT
          WRITE(OUTU,'(I14,1X,A)')  &
               NOTRNG,'POINTS WERE OUT OF RANGE AND EXCLUDED'
       ENDIF
    ENDIF
    ! reset accumulation arrays, counters
    DO I = 1,NSTTER
       VARIAN(I) = 0
       AVENER(I) = 0
       SQRAVG(I) = 0
    ENDDO
    NSTTER=0
    NINCMT=0
    AVENEP=0
    VARIAP=0
    SQRAVP=0
    RETURN
  END SUBROUTINE STOPSTAT

  SUBROUTINE PRNENERU
    ! ------------------------------------------------------------
    use chm_kinds
    !--mfc--   use estats_mod
    use energym
    use stream
    implicit none
    !
    IF (NINCMT.GT.0) THEN
       WRITE(EPRUNT,'(A9,1X,I14,1X,F17.8)')  &
            'ESTV> ENER',NINCMT,EPROP(EPOT)
    ENDIF
    RETURN
  END SUBROUTINE PRNENERU
#endif /* (running energy_statistics)*/

end module estats_mod

