module rxcons

contains
#if KEY_RXNCONS==0 /*rxncons_main*/
  SUBROUTINE RXCONSPS(COMLYN,COMLEN)
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN
    CALL WRNDIE(-1,'<CHARMM>','RXNCONS code not compiled.')
  END SUBROUTINE RXCONSPS
#else /* (rxncons_main)*/
  SUBROUTINE RXCONSPS(COMLYN,COMLEN)
    !
    ! This routine parses the reaction coordinate constraint
    ! command
    !
    use number
    use memory
    use exfunc
    use dimens_fcm
    use coord
    use coordc
    use psf
    use stream
    use string
    use replica_mod

    use rxncons
    use rxcons1
    use rxcons2
    use rxnconswt
    use rxncons3
    use rxncons4
    use rxcons3
    use rxcons4
    use parallel
    use repdstr
    use chm_kinds
#if KEY_DOMDEC==1
    use domdec_common,only:domdec_system_changed
#endif
    implicit none

    INTEGER COMLEN
    CHARACTER(len=*) COMLYN

    INTEGER I
    logical :: qwarn

    LCLEAN=.FALSE.
    IF(INDXA(COMLYN,COMLEN,'CLEAN').GT.0) THEN
       IF(LRXCNS) THEN
          if (prnlev >= 2) WRITE(OUTU,11) 'The reaction coordinate constraint' , &
               'will be cleaned'
          LRXCNS=.FALSE.
          LCLEAN=.TRUE.
          if(allocated(lprid)) &
               call chmdealloc('rxcons.src','RXCONSPS','lprid ',maxaim,intg=lprid)
#if KEY_DOMDEC==1
          call domdec_system_changed()
#endif
       ELSE
          CALL WRNDIE(-1,'<RXCONS>', &
               'The reaction coordinate constraint has not been set')
          RETURN
       ENDIF
    ENDIF

    ! we need this here in case of future development might support
    ! more than one call to RCON command in the script
    if(.not.allocated(lprid)) &
         call chmalloc('rxcons.src','RXCONSPS','lprid ',maxaim,intg=lprid)
!    call allocate_rxncons  ! this is not needed anymore

    QPFORCE = (INDXA(COMLYN, COMLEN, 'PPFO') .GT. 0) 
    IF(LRXCNS.AND.(.NOT.LCLEAN).AND.(.NOT.QPFORCE))  &
         CALL WRNDIE(-3,'<RXCONSPS>', &
         'Only one rxn coordinate is allowed to be set at this point')
    ! . read in reaction coordinate type

    PLAGRANM = (INDXA(COMLYN, COMLEN, 'PLAG') .GT. 0)

    VRXNCS = GTRMF(COMLYN, COMLEN, 'RXNC',  ZERO)
    IUNLAG = GTRMI(COMLYN,COMLEN,'IUNL',-1)
    NSAVLG = GTRMI(COMLYN,COMLEN,'NSAV',-1)
    IF(IUNLAG.GT.0.AND.PLAGRANM) THEN
       if (prnlev >= 2) WRITE(OUTU,'(A,I5,A,I5,A)') &
            'Lagrange multiplier will be written to unit: ',IUNLAG, &
            'every ',NSAVLG,' steps'
       WRITE(IUNLAG,11) '      LAGRANGE MULTIPLIER,     Z^-1/2,      G'
    ENDIF

    LRXPRN = (INDXA(COMLYN, COMLEN, 'PRIN') .GT. 0)
    MAXITR = GTRMI(COMLYN,COMLEN,'MAXI',2000)

11  FORMAT(A,X,A)

    IF(INDXA(COMLYN,COMLEN,'BDIS').GT.0) THEN
       IRXCNS=1
       NRXATM=3
       NRXNCS=1
       CALL RX_1_SETUP(COMLYN,COMLEN)
    ELSEIF(INDXA(COMLYN,COMLEN,'BOND').GT.0) THEN
       IRXCNS=2
       NRXATM=2
       NRXNCS=1
       IF(VRXNCS.LE.ZERO.AND..NOT.LCLEAN) &
            CALL WRNDIE(-1,'<RXCONSPS>', &
            'RCONS must be positive for BOND.')
       CALL RX_2_SETUP(COMLYN,COMLEN)
    ELSEIF(INDXA(COMLYN,COMLEN,'PCNS').GT.0) THEN
       IF(.NOT.LCLEAN)THEN
          IRXCNS=3
          NRXNCS=1
          QCNOTRN= (INDXA(COMLYN, COMLEN, 'NOTR') .GT. 0)
          QCNOROT= (INDXA(COMLYN, COMLEN, 'NORO') .GT. 0)
          QCMASS = (INDXA(COMLYN, COMLEN, 'MASS') .GT. 0)
          QCWEIG = (INDXA(COMLYN, COMLEN, 'WEIG') .GT. 0)
          QCWCOMP = (INDXA(COMLYN, COMLEN, 'WCOM') .GT. 0)
          QCWCOMP2 = (INDXA(COMLYN, COMLEN, 'WCM2') .GT. 0)
          NRXATM=0
          IF(QCWEIG)THEN
             DO I=1,NATOM
                IF(WMAIN(I).GT.0) NRXATM=NRXATM+1
             ENDDO
          ELSEIF(QCWCOMP)THEN
             DO I=1,NATOM
                IF(WCOMP(I).GT.0) NRXATM=NRXATM+1
             ENDDO
#if KEY_COMP2==1
          ELSEIF(QCWCOMP2)THEN
             DO I=1,NATOM
                IF(WCOMP2(I).GT.0) NRXATM=NRXATM+1
             ENDDO
#endif 
          ELSE
             NRXATM=NATOM
          ENDIF
       ENDIF
       CALL RX_3_SETUP(COMLYN,COMLEN)
    ELSEIF(INDXA(COMLYN,COMLEN,'PATH').GT.0) THEN
       IRXCNS=4
       NRXATM=0
#if KEY_REPLICA==0 && KEY_REPDSTR==0
       CALL WRNDIE(-1,'<REPLICA>','REPLICA code not compiled.')
#else /**/
       !
#if KEY_REPLICA==1
       qwarn=.not.qrep                       
#endif
#if KEY_REPDSTR==1
       qwarn=.not.qrepdstr                   
#endif
#if KEY_REPLICA==1 && KEY_REPDSTR==1
       qwarn=(.not.qrep).and.(.not.qrepdstr) 
#endif
       IF(qwarn) CALL WRNDIE(-1,'<RXNCONSPS>', &
            'The PATH RMSD constraint needs replicas')
       !
       QCNOTRN = (INDXA(COMLYN, COMLEN, 'NOTR') .GT. 0)
       QCNOROT = (INDXA(COMLYN, COMLEN, 'NORO') .GT. 0)
       QCMASS  = (INDXA(COMLYN, COMLEN, 'MASS') .GT. 0)
       QCWEIG  = (INDXA(COMLYN, COMLEN, 'WEIG') .GT. 0)
       QCWCOMP = (INDXA(COMLYN, COMLEN, 'WCOM') .GT. 0)
       QCWCOMP2= (INDXA(COMLYN, COMLEN, 'WCM2') .GT. 0)
       QPTCSCL = (INDXA(COMLYN, COMLEN, 'CYCL') .GT. 0) ! for cyclic path
       QFXREP  = (INDXA(COMLYN, COMLEN, 'FIXR') .GT. 0) 
       QPTAN   = (INDXA(COMLYN, COMLEN, 'PTAN') .GT. 0) 
       IF(QPTAN)IUTAN = GTRMI(COMLYN,COMLEN,'IUTA',-1)
       IF(QPTAN.AND.(IUTAN.LT.ZERO)) &
            CALL WRNDIE(-1,'<RXCONSPS>', &
            'IUTAN is negative.')
       IF(QFXREP)IFXREP = GTRMI(COMLYN,COMLEN,'IFIX',1)
       CALL RX_4_SETUP(COMLYN,COMLEN)
#endif /*  IF REPLICA*/
    ENDIF
    RETURN
  END subroutine rxconsps
#endif /* (rxncons_main)*/
end module rxcons

