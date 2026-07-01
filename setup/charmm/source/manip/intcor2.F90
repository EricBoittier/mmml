module intcor2
  use chm_kinds
  use dimens_fcm
  implicit none

contains
  SUBROUTINE PURGIC(LENIC,B1IC,B2IC,T1IC,T2IC,PIC, &
       IAR,JAR,KAR,LAR,TAR, &
       LFILL,LALL,LDIHE,LIMPR,LCLEAR, &
       NATOM,IMOVE,IAC)
    !
    !     COMPRESS INTERNAL COORDINATES
    !
    !     By Bernard R. Brooks    1982
    !     Rick Lapp 25-May-1998: added code for CFF
    !
    use consta
    use exfunc
    use number
    use stream
    use param
    !
    implicit none
    !
    INTEGER LENIC
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    LOGICAL LFILL,LALL,LDIHE,LIMPR,LCLEAR
    INTEGER NATOM,IMOVE(*),IAC(*)
    !
    INTEGER II, I, J
    INTEGER II1, JJ1, KK1, LL1
    INTEGER QAT(5)
    INTEGER IX
    INTEGER NPERD
    real(chm_real) EMIN,ANGMIN,EVAL,ANG
    LOGICAL LDROP
    INTEGER :: MARK2=-99
    !
    !
    IF(.NOT.LFILL) THEN
       !
       IF(LCLEAR) THEN
          DO I=1,LENIC
             IF(IAR(I) <= 0) IAR(I)=MARK2
             IF(JAR(I) <= 0) IAR(I)=MARK2
             IF(KAR(I) <= 0) IAR(I)=MARK2
             IF(LAR(I) <= 0) IAR(I)=MARK2
          ENDDO
       ENDIF
       !
       II=0
       DO I=1,LENIC
          LDROP=(IAR(I) == MARK2.OR.JAR(I) == MARK2.OR.KAR(I) == MARK2 &
               .OR.LAR(I) == MARK2)
          IF(.NOT.LDROP) THEN
             II=II+1
             IF(I /= II) THEN
                IAR(II)=IAR(I)
                JAR(II)=JAR(I)
                KAR(II)=KAR(I)
                LAR(II)=LAR(I)
                TAR(II)=TAR(I)
                B1IC(II)=B1IC(I)
                B2IC(II)=B2IC(I)
                T1IC(II)=T1IC(I)
                T2IC(II)=T2IC(I)
                PIC(II)=PIC(I)
             ENDIF
          ENDIF
       ENDDO
       LENIC=II
    ENDIF
    !
    IF(LFILL) THEN
       DO II=1,LENIC
          IF(B1IC(II) <= 0.001 .OR.LALL) THEN
             II1=KAR(II)
             JJ1=LAR(II)
             IF(II1 > 0.AND.JJ1 > 0) THEN
                QAT(1)=II1
                QAT(2)=JJ1
                CALL CODES(QAT(3),[0],[0],[0], &
                     NATOM,IMOVE,IAC,1,QAT(1),QAT(2), &
                     0,[0],[0],[0],0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
                     .FALSE.,0,                    & ! DRUDE
#if KEY_CMAP==1
                     [0],0,[0],[0],[0],[0],[0],[0],[0],[0],          &
#endif
                     .TRUE.,.TRUE.)
                IX=QAT(3)
                IF(IX > 0) THEN
                   B1IC(II)=CBB(IX)
                ELSE
                   IF(WRNLEV >= 2) WRITE(OUTU,'(5A)') &
                        ' %BUILDC-WARNING: no bond parameters for types (', &
                        ATC(IAC(II1))(1:idleng),',', &
                        ATC(IAC(JJ1))(1:idleng),')'
                ENDIF
             ENDIF
          ENDIF
          !
          IF(B2IC(II) <= 0.001 .OR.LALL) THEN
             II1=IAR(II)
             JJ1=JAR(II)
             IF(TAR(II)) JJ1=KAR(II)
             IF(II1 > 0.AND.JJ1 > 0) THEN
                QAT(1)=II1
                QAT(2)=JJ1
                CALL CODES(QAT(3),[0],[0],[0], &
                     NATOM,IMOVE,IAC,1,QAT(1),QAT(2), &
                     0,[0],[0],[0],0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
                     .FALSE.,0,                    & ! DRUDE
#if KEY_CMAP==1
                     [0],0,[0],[0],[0],[0],[0],[0],[0],[0],          &
#endif
                     .TRUE.,.TRUE.)
                IX=QAT(3)
                IF(IX > 0) THEN
                   B2IC(II)=CBB(IX)
                ELSE
                   IF(WRNLEV >= 2) WRITE(OUTU,'(5A)') &
                        ' %BUILDC-WARNING: no bond parameters for types (', &
                        ATC(IAC(II1))(1:idleng),',', &
                        ATC(IAC(JJ1))(1:idleng),')'
                ENDIF
             ENDIF
          ENDIF
          !
          IF(T1IC(II) <= 0.001 .OR.LALL) THEN
             II1=LAR(II)
             JJ1=KAR(II)
             KK1=JAR(II)
             IF(II1 > 0.AND.JJ1 > 0.AND.KK1 > 0) THEN
                QAT(1)=II1
                QAT(2)=JJ1
                QAT(3)=KK1
                CALL CODES([0],QAT(4),[0],[0], &
                     NATOM,IMOVE,IAC,0,[0],[0], &
                     1,QAT(1),QAT(2),QAT(3), &
                     0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
                     .FALSE.,0,                    & ! DRUDE
#if KEY_CMAP==1
                     [0],0,[0],[0],[0],[0],[0],[0],[0],[0],          &
#endif
                     .TRUE.,.TRUE.)
                IX=QAT(4)
                IF(IX > 0) THEN
                   T1IC(II)=CTB(IX)*RADDEG
                ELSE
                   IF(WRNLEV >= 2) WRITE(OUTU,'(7A)') &
                        ' %BUILDC-WARNING: no angle parameters for types (', &
                        ATC(IAC(II1))(1:idleng),',',ATC(IAC(JJ1))(1:idleng), &
                        ',',ATC(IAC(KK1))(1:idleng),')'
                ENDIF
             ENDIF
          ENDIF
          !
          IF(T2IC(II) <= 0.001 .OR.LALL) THEN
             II1=IAR(II)
             JJ1=KAR(II)
             KK1=JAR(II)
             IF(.NOT.TAR(II)) THEN
                JJ1=JAR(II)
                KK1=KAR(II)
             ENDIF
             IF(II1 > 0.AND.JJ1 > 0.AND.KK1 > 0) THEN
                QAT(1)=II1
                QAT(2)=JJ1
                QAT(3)=KK1
                CALL CODES([0],QAT(4),[0],[0], &
                     NATOM,IMOVE,IAC,0,[0],[0], &
                     1,QAT(1),QAT(2),QAT(3), &
                     0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
                     .FALSE.,0,                    & ! DRUDE
#if KEY_CMAP==1
                     [0],0,[0],[0],[0],[0],[0],[0],[0],[0],          &
#endif
                     .TRUE.,.TRUE.)
                IX=QAT(4)
                IF(IX > 0) THEN
                   T2IC(II)=CTB(IX)*RADDEG
                ELSE
                   IF(WRNLEV >= 2) WRITE(OUTU,'(7A)') &
                        ' %BUILDC-WARNING: no angle parameters for types (', &
                        ATC(IAC(II1))(1:idleng),',',ATC(IAC(JJ1))(1:idleng), &
                        ',',ATC(IAC(KK1))(1:idleng),')'
                ENDIF
             ENDIF
          ENDIF
          !
          !           fill dihedrals (if explicitly requested)
          IF(LDIHE .AND. .NOT.TAR(II)) THEN
             IF(PIC(II) <= 0.001 .OR.LALL) THEN
                II1=IAR(II)
                JJ1=JAR(II)
                KK1=KAR(II)
                LL1=LAR(II)
                IF(II1 > 0.AND.JJ1 > 0.AND.KK1 > 0.AND.LL1 > 0) THEN
                   QAT(1)=II1
                   QAT(2)=JJ1
                   QAT(3)=KK1
                   QAT(4)=LL1
                   CALL CODES([0],[0],QAT(5),[0], &
                        NATOM,IMOVE,IAC,0,[0],[0],0,[0],[0],[0],1, &
                        QAT(1),QAT(2),QAT(3),QAT(4), &
                        0,[0],[0],[0],[0], &
                        .FALSE.,0,                    & ! DRUDE
#if KEY_CMAP==1
                        [0],0,[0],[0],[0],[0],[0],[0],[0],[0],          &
#endif
                        .TRUE.,.TRUE.)
                   IX=QAT(5)
                   IF(IX > 0) THEN
                      !                    find the minimum nearest 180.0
                      NPERD=0
122                   CONTINUE
                      NPERD=NPERD+1
                      IF(CPD(IX+NPERD-1) < 0) GOTO 122
                      EMIN=9999.0
                      ANGMIN=180.0
                      DO I=180,-180,-6
                         ANG=I*DEGRAD
                         EVAL=ZERO
                         DO J=IX,IX+NPERD-1
                            ! e = k * ( 1.0 + cos( periodicity * phi - phase ) )
                            EVAL=EVAL+  &
                                 CPC(J)*(ONE+COS(IABS(CPD(J))*ANG-CPB(J)))
                         ENDDO
                         IF(EVAL < EMIN) THEN
                            EMIN=EVAL
                            ANGMIN=I
                         ENDIF
                      ENDDO
                      PIC(II)=ANGMIN
                   ELSE
                      IF(WRNLEV >= 2) WRITE(OUTU,'(7A)') &
                           ' %BUILDC-WARNING: no dihedral parameters for types (', &
                           ATC(IAC(II1))(1:idleng),',',ATC(IAC(JJ1))(1:idleng),',', &
                           ATC(IAC(KK1))(1:idleng),',',ATC(IAC(KK1))(1:idleng),')'
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
          !
          !           fill improper dihedrals (if explicitly requested)
          IF(LIMPR .AND. TAR(II)) THEN
             IF(PIC(II) <= 0.001 .OR.LALL) THEN
                II1=IAR(II)
                JJ1=JAR(II)
                KK1=KAR(II)
                LL1=LAR(II)
                IF(II1 > 0.AND.JJ1 > 0.AND.KK1 > 0.AND.LL1 > 0) THEN
                   IX=0
                   IF(IX == 0) THEN
                      !                   First try CHARMM type of improper where CID is zero
                      !                   Here we assume that the central atom is first and
                      !                   the off axis atom is at the end position.
                      QAT(1)=KK1
                      QAT(2)=II1
                      QAT(3)=JJ1
                      QAT(4)=LL1
                      CALL CODES([0],[0],[0],QAT(5), &
                           NATOM,IMOVE,IAC,&
                           0,[0],[0],0,[0],[0],[0],0,[0],[0],[0],[0], &
                           1,QAT(1),QAT(2),QAT(3),QAT(4), &
                           .FALSE.,0,                       & ! DRUDE
#if KEY_CMAP==1
                           [0],0,[0],[0],[0],[0],[0],[0],[0],[0],             &
#endif
                           .TRUE.,.TRUE.)
                      IX=QAT(5)
                      IF(IX > 0) THEN
                         IF(CID(IX) /= 0) THEN
                            !                       reject since it's not a CHARMM type...
                            IX=0
                         ELSE
                            PIC(II)=180.0 - CTB(IX)*RADDEG*SQRT(THREE)
                            IF(PIC(II) > 180.0) PIC(II)=PIC(II)-360.0
                         ENDIF
                      ENDIF
                   ENDIF
                   IF(IX == 0) THEN
                      !                   Next try another CHARMM type of improper
                      !                   Here we exchange the middle two atoms
                      !                   the off axis atom is at the end position.
                      QAT(1)=KK1
                      QAT(2)=JJ1
                      QAT(3)=II1
                      QAT(4)=LL1
                      CALL CODES([0],[0],[0],QAT(5), &
                           NATOM,IMOVE,IAC, &
                           0,[0],[0],0,[0],[0],[0],0,[0],[0],[0],[0], &
                           1,QAT(1),QAT(2),QAT(3),QAT(4), &
                           .FALSE.,0,                       & ! DRUDE
#if KEY_CMAP==1
                           [0],0,[0],[0],[0],[0],[0],[0],[0],[0],             &
#endif
                           .TRUE.,.TRUE.)
                      IX=QAT(5)
                      IF(IX > 0) THEN
                         IF(CID(IX) /= 0) THEN
                            !                       reject since it's not a CHARMM type...
                            IX=0
                         ELSE
                            PIC(II)=180.0 + CTB(IX)*RADDEG*SQRT(THREE)
                            IF(PIC(II) > 180.0) PIC(II)=PIC(II)-360.0
                         ENDIF
                      ENDIF
                   ENDIF
                   IF(IX == 0) THEN
                      !                   Next try another CHARMM type of improper
                      !                   Now the off axis atom is at the first position.
                      QAT(1)=KK1
                      QAT(2)=JJ1
                      QAT(3)=LL1
                      QAT(4)=II1
                      CALL CODES([0],[0],[0],QAT(5), &
                           NATOM,IMOVE,IAC, &
                           0,[0],[0],0,[0],[0],[0],0,[0],[0],[0],[0], &
                           1,QAT(1),QAT(2),QAT(3),QAT(4), &
                           .FALSE.,0,                       & ! DRUDE
#if KEY_CMAP==1
                           [0],0,[0],[0],[0],[0],[0],[0],[0],[0],             &
#endif
                           .TRUE.,.TRUE.)
                      IX=QAT(5)
                      IF(IX > 0) THEN
                         IF(CID(IX) /= 0) THEN
                            !                       reject since it's not a CHARMM type...
                            IX=0
                         ELSE
                            PIC(II)=180.0 + CTB(IX)*RADDEG*SQRT(THREE)
                            IF(PIC(II) > 180.0) PIC(II)=PIC(II)-360.0
                         ENDIF
                      ENDIF
                   ENDIF
                   IF(IX == 0) THEN
                      !                   Next try another CHARMM type of improper
                      !                   Now the off axis atom is at the first position.
                      QAT(1)=KK1
                      QAT(2)=LL1
                      QAT(3)=JJ1
                      QAT(4)=II1
                      CALL CODES([0],[0],[0],QAT(5), &
                           NATOM,IMOVE,IAC, &
                           0,[0],[0],0,[0],[0],[0],0,[0],[0],[0],[0], &
                           1,QAT(1),QAT(2),QAT(3),QAT(4), &
                           .FALSE.,0,                       & ! DRUDE
#if KEY_CMAP==1
                           [0],0,[0],[0],[0],[0],[0],[0],[0],[0],             &
#endif
                           .TRUE.,.TRUE.)
                      IX=QAT(5)
                      IF(IX > 0) THEN
                         IF(CID(IX) /= 0) THEN
                            !                       reject since it's not a CHARMM type...
                            IX=0
                         ELSE
                            PIC(II)=180.0 - CTB(IX)*RADDEG*SQRT(THREE)
                            IF(PIC(II) > 180.0) PIC(II)=PIC(II)-360.0
                         ENDIF
                      ENDIF
                   ENDIF
                   IF(IX == 0) THEN
                      !                   ok, try AMBER type of improper where CID is nonzero
                      QAT(1)=II1
                      QAT(2)=JJ1
                      QAT(3)=KK1
                      QAT(4)=LL1
                      CALL CODES([0],[0],[0],QAT(5), &
                           NATOM,IMOVE,IAC, &
                           0,[0],[0],0,[0],[0],[0],0,[0],[0],[0],[0], &
                           1,QAT(1),QAT(2),QAT(3),QAT(4), &
                           .FALSE.,0,                       & ! DRUDE
#if KEY_CMAP==1
                           [0],0,[0],[0],[0],[0],[0],[0],[0],[0],             &
#endif
                           .TRUE.,.TRUE.)
                      IX=QAT(5)
                   ENDIF
                   IF(IX > 0) THEN
                      IF(CID(IX) == 0) THEN
                         !                     reject since it's a CHARMM type...
                         IX=0
                      ELSE
                         !                     find the minimum nearest 180.0
                         NPERD=0
132                      CONTINUE
                         NPERD=NPERD+1
                         IF(CID(IX+NPERD-1) < 0) GOTO 132
                         EMIN=9999.0
                         ANGMIN=180.0
                         DO I=180,-180,-6
                            ANG=I*DEGRAD
                            EVAL=ZERO
                            DO J=IX,IX+NPERD-1
                               ! e = k * ( 1.0 + cos( periodicity * phi - phase ) )
                               EVAL=EVAL+  &
                                    CIC(J)*(ONE+COS(IABS(CID(J))*ANG-CIB(J)))
                            ENDDO
                            IF(EVAL < EMIN) THEN
                               EMIN=EVAL
                               ANGMIN=I
                            ENDIF
                         ENDDO
                         PIC(II)=ANGMIN
                      ENDIF
                   ENDIF
                   IF(IX == 0) THEN
                      IF(WRNLEV >= 2) WRITE(OUTU,'(7A)') &
                           ' %BUILDC-WARNING: no dihedral parameters for types (', &
                           ATC(IAC(II1))(1:idleng),',',ATC(IAC(JJ1))(1:idleng),',', &
                           ATC(IAC(KK1))(1:idleng),',',ATC(IAC(KK1))(1:idleng),')'
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
          !
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE PURGIC

  SUBROUTINE REMAPIC2(MAP,MARK,LENIC,B1IC,B2IC,T1IC,T2IC,PIC, &
       IAR,JAR,KAR,LAR,TAR)
    !
    !     REMAPS AND COMPRESS INTERNAL COORDINATES
    !
    !     By Bernard R. Brooks    1998
    !
    use exfunc
    implicit none
    !
    INTEGER MAP(*),MARK,LENIC
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    INTEGER I,II,J
    !
    DO I=1,LENIC
       IF(IAR(I) <= 0) IAR(I)=MARK
       IF(JAR(I) <= 0) JAR(I)=MARK
       IF(KAR(I) <= 0) KAR(I)=MARK
       IF(LAR(I) <= 0) LAR(I)=MARK
    ENDDO
    !
    CALL AVALUE(MAP,IAR,LENIC,MARK)
    CALL AVALUE(MAP,JAR,LENIC,MARK)
    CALL AVALUE(MAP,KAR,LENIC,MARK)
    CALL AVALUE(MAP,LAR,LENIC,MARK)
    !
    ! Remove any worthless IC table entries
    II=0
    DO I=1,LENIC
       J=0
       IF(IAR(I) /= MARK) J=J+6
       IF(JAR(I) /= MARK) J=J+7
       IF(KAR(I) /= MARK) J=J+8
       IF(LAR(I) /= MARK) J=J+4
       IF(J >= 12 .AND. J /= 15) THEN
          II=II+1
          IAR(II)=IAR(I)
          JAR(II)=JAR(I)
          KAR(II)=KAR(I)
          LAR(II)=LAR(I)
          TAR(II)=TAR(I)
          B1IC(II)=B1IC(I)
          B2IC(II)=B2IC(I)
          T1IC(II)=T1IC(I)
          T2IC(II)=T2IC(I)
          PIC(II)=PIC(I)
       ENDIF
    ENDDO
    LENIC=II
    !
    RETURN
  END SUBROUTINE REMAPIC2

  SUBROUTINE PUTICEL(II,JJ,KK,LL,BART,ICB1,ICB2, &
       ICTH1,ICTH2,ICPHI,ICEL, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    ! This routine fills a specific IC table element.  It has been
    ! added to make it easier to deal with the IC table as a
    ! structured data file.
    !
    !     By Bernard R. Brooks    1998
    !
    implicit none
    !
    INTEGER II,JJ,KK,LL
    LOGICAL BART
    real(chm_real) ICB1,ICB2,ICTH1,ICTH2,ICPHI
    INTEGER ICEL
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    ! Fill a specific IC table entry
    !
    IF(ICEL <= 0) RETURN
    !
    B1IC(ICEL)= ICB1
    B2IC(ICEL)= ICB2
    T1IC(ICEL)= ICTH1
    T2IC(ICEL)= ICTH2
    PIC(ICEL) = ICPHI
    IAR(ICEL) = II
    JAR(ICEL) = JJ
    KAR(ICEL) = KK
    LAR(ICEL) = LL
    TAR(ICEL) = BART
    RETURN
  END SUBROUTINE PUTICEL

  SUBROUTINE GETICEL(II,JJ,KK,LL,BART,ICB1,ICB2, &
       ICTH1,ICTH2,ICPHI,ICEL, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    ! This routine retrieves a specific IC table element.  It has been
    ! added to make it easier to deal with the IC table as a
    ! structured data file.
    !
    !     By Bernard R. Brooks    1998
    !
    implicit none
    !
    INTEGER II,JJ,KK,LL
    LOGICAL BART
    real(chm_real) ICB1,ICB2,ICTH1,ICTH2,ICPHI
    INTEGER ICEL
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    ! Retrieve a specific IC table entry
    !
    IF(ICEL <= 0) CALL DIEWRN(-4)
    !
    ICB1  = B1IC(ICEL)
    ICB2  = B2IC(ICEL)
    ICTH1 = T1IC(ICEL)
    ICTH2 = T2IC(ICEL)
    ICPHI = PIC(ICEL)
    II    = IAR(ICEL)
    JJ    = JAR(ICEL)
    KK    = KAR(ICEL)
    LL    = LAR(ICEL)
    BART  = TAR(ICEL)
    !
    RETURN
  END SUBROUTINE GETICEL

  SUBROUTINE BILDC(NSTART,NSTOP,X,Y,Z, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR,NATOM,ICACTV,PNOTEST,cumetime)
    !
    !     THIS ROUTINE CONSTRUCTS CARTESIAN COORDINATES FROM THE INTERNAL
    !     COORDINATES
    !
    !     By Bernard R. Brooks    1982
    !
   use new_timer,only: seconds

    use stream
    use number
#if KEY_ZEROM==1
    use zdata_mod
#endif
    implicit none
    !
    INTEGER NSTART,NSTOP
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    INTEGER NATOM
    integer,intent(in),dimension(:),optional :: ICACTV
    logical,intent(in),optional :: PNOTEST
    !
    real(chm_real) R,THETA,PHI
    INTEGER INC,KK,II,I,J,K,L,ITEMP
    LOGICAL RETRY,JUNK,UNKN,OK
    INTEGER CHANGE
    integer :: JJ
    real(chm_real),intent(out),optional :: cumetime
    real(chm_real) :: etime,ctime,time_01
    integer :: pass=0
    logical :: QTIMING=.false.
    logical :: QNOTEST

#if KEY_ADUMB==1 /*RJP 5.25.02*/
    integer :: OLDLEV
#endif
! end of declarations--------------------------
    !
    if(present(cumetime)) cumetime =0
    QNOTEST = .false.
    if(present(PNOTEST)) QNOTEST=PNOTEST
    pass = pass + 1
    !
#if KEY_ADUMB==1
    OLDLEV = PRNLEV
    PRNLEV = 1
#endif
    IF(NSTOP < NSTART) THEN
#if KEY_ADUMB == 1
       prnlev = oldlev
#endif /* KEY_ADUMB */
       CALL WRNDIE(1, '<BILDC>', 'BILDC called with a null IC table')
       RETURN
    ENDIF
    !
    INC=1
    KK=NSTART-INC
    JJ=NSTART-INC
#if KEY_ZEROM==1 /*zerom*/
    IF(QICBREV) THEN
       KK = NSTOP + INC
       JJ = NSTOP + INC
       INC = -INC
    ENDIF
#endif /* (zerom)*/
    !
    CHANGE=2
    DO WHILE(CHANGE > 0)
       CHANGE=CHANGE-1
       RETRY=.FALSE.
       DO II=NSTART,NSTOP
          if(present(ICACTV)) then
           JJ = JJ + INC
           KK = ICACTV(JJ)
          else
           KK=KK+INC
          endif
          !
          ! Find atom whose coordinates are to be generated (atom l).
          !
          I=IAR(KK)
          J=JAR(KK)
          K=KAR(KK)
          L=LAR(KK)
          PHI=(PIC(KK))
          R=(B1IC(KK))
          THETA=(T1IC(KK))
          !
          IF(INC < 0) THEN
             R=(B2IC(KK))
             THETA=(T2IC(KK))
             ITEMP=I
             I=L
             L=ITEMP
             IF(TAR(KK)) THEN
                PHI=-PHI
             ELSE
                ITEMP=J
                J=K
                K=ITEMP
             ENDIF
          ENDIF
          !
          ! SEE IF COORDINATES ARE ALREADY KNOWN
          !
          IF(I <= 0) THEN
             CONTINUE
          ELSE IF(J <= 0) THEN
             CONTINUE
          ELSE IF(K <= 0) THEN
             CONTINUE
          ELSE IF(L <= 0) THEN
             CONTINUE
          ELSE IF(R < 0.001) THEN
             CONTINUE
          ELSE IF(THETA < 0.001) THEN
             CONTINUE
          ELSE IF(I == K) THEN
             CONTINUE
          ELSE IF(I == J) THEN
             CONTINUE
          ELSE IF(K == J) THEN
             CONTINUE
          ELSE
             !
             ! Check to see if all antecedents are known
             !
             UNKN=X(L) == ANUM
             JUNK=(X(I) == ANUM.OR.X(J) == ANUM.OR.X(K) == ANUM)
             RETRY=RETRY.OR.JUNK.OR.UNKN
             IF (UNKN.AND..NOT.JUNK) THEN
                !
                ! Set geometrical parameters
                !
                if(QTIMING) then
                 call seconds(etime,ctime)
                 time_01 = etime
                endif

                CALL CARTCV(X,Y,Z,I,J,K,L,R,THETA,PHI,OK,pnotest=QNOTEST)

                if(QTIMING) then
                 call seconds(etime,ctime)
                 if(present(cumetime)) then
                  cumetime = cumetime + etime - time_01
                 endif
                endif
                IF(OK) CHANGE=2
             ENDIF
          ENDIF
       ENDDO
       KK=KK+INC
       JJ=JJ+INC
       INC=-INC
       !
       ! Retry will be false if all coordinates are known
    ENDDO
    !
    IF(RETRY) THEN
       CALL WRNDIE(1,'<BUILDC>','SOME COORDINATES NOT BUILT')
    ELSE
#if KEY_ZEROM==1
       CONTINUE
#else /**/
       IF(PRNLEV >= 2) WRITE(OUTU,123)
123    FORMAT(' ALL POSSIBLE COORDINATES HAVE BEEN PLACED')
#endif
    ENDIF
    KK=0
    DO I=1,NATOM
       IF(X(I) == ANUM) KK=KK+1
    ENDDO
    IF (KK /= 0 .AND. WRNLEV >= 2) WRITE(OUTU,124) KK
124 FORMAT(' ****  WARNING  ****',I5, &
         ' COORDINATES ARE STILL UNDEFINED')
    !
#if KEY_ADUMB==1
    PRNLEV = OLDLEV
#endif
    RETURN
  END SUBROUTINE BILDC

  SUBROUTINE CARTCV(X,Y,Z,I,J,K,L,RX,TX,PX,OK,PNOTEST)
    !
    !     THIS ROUTINE FINDS THE POSITION OF L FROM THE POSITIONS
    !     OF I,J,K AND R,THETA,PHI. THE COORDINATE ARRAY IS MODIFIED
    !     ANGLES ARE IN DEGREES.
    !
    !     By Bernard R. Brooks    1982
    !
    use consta
    use psf
    use stream
    !
    implicit none
    INTEGER,intent(in) :: I,J,K,L
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real),intent(in) :: RX,TX,PX
    logical,intent(in),optional :: PNOTEST
    LOGICAL OK
    real(chm_real) :: T,P,CST,SNT,CSP,SNP,XA,YA,ZA,XB,YB,ZB,XBB,YBB,ZBB
    real(chm_real) :: RA,XC,YC,ZC,RC,WA,WB,WC
!  local
    logical :: QNOTEST,QTIMING=.false.
    real(chm_real) :: etime,ctime,cumetime=0,time_01
!    integer :: pass=0

    QNOTEST=.false.
    if(present(PNOTEST)) QNOTEST=PNOTEST
    !RCZ 91/10/24 set trap for unknown atoms
    if(.not.QNOTEST) then
     OK=I <= 0 .OR. J <= 0 .OR. K <= 0 .OR. L <= 0
     OK=OK .OR. I > NATOM .OR. J > NATOM .OR. &
         K > NATOM .OR. L > NATOM
    IF(OK) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,'(A,4(1X,I5))')  &
            ' CARTCV> ERROR: Unknown atoms I,J,K,L=',I,J,K,L
       CALL WRNDIE(-1,'<CARTCV>','*** BAD ADDRESS ***')
     ENDIF
    endif
    !RCZ 91/10/24
    !
    XA=X(J)-X(K)
    YA=Y(J)-Y(K)
    ZA=Z(J)-Z(K)
    XBB=X(I)-X(J)
    YBB=Y(I)-Y(J)
    ZBB=Z(I)-Z(J)
    !
    RA=1/SQRT(XA*XA+YA*YA+ZA*ZA)
!    write(6,*) 'XYZ A ',XA,YA,ZA,' RA ',RA
    !
    XC=YBB*ZA-ZBB*YA
    YC=ZBB*XA-XBB*ZA
    ZC=XBB*YA-YBB*XA
    !
    RC=1/SQRT(XC*XC+YC*YC+ZC*ZC)
    !
    IF (RC > 1.0D+12) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,22) I,J,K,L
22     FORMAT(' Note from CARTCV: I-J-K is linear.',4I5, &
            ', L not built.')
       OK=.FALSE.
       WRITE(6,*) 'RC is ',RC !temporary
       RETURN
    ENDIF
    !
    XC=XC*RC
    YC=YC*RC
    ZC=ZC*RC
    !
    XB=YA*ZC-ZA*YC
    YB=ZA*XC-XA*ZC
    ZB=XA*YC-YA*XC
    !
    T=DEGRAD*TX
    P=DEGRAD*PX
    SNT=SIN(T)
    WA=COS(T)
    WB=SNT*COS(P)
    WC=SNT*SIN(P)
    !
    X(L)=X(K)+RX*(RA*(WA*XA+WB*XB)+WC*XC)
    Y(L)=Y(K)+RX*(RA*(WA*YA+WB*YB)+WC*YC)
    Z(L)=Z(K)+RX*(RA*(WA*ZA+WB*ZB)+WC*ZC)
    !
    OK=.TRUE.
    RETURN
  END SUBROUTINE CARTCV

  SUBROUTINE SEED(I,J,K,X,Y,Z,LENIC, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    !     THIS ROUTINE CREATE THE COORDINATES FOR THE FIRST THREE ATOMS
    !     OR THOSE SPECIFIED IN THE SEED COMMAND LINE.
    !
    !  Note:  PIC is not used in this routine. Called only once. ABl.
    !
    !     By Bernard R. Brooks    1982
    !
    use number
    use consta
    use stream
    implicit none
    !
    INTEGER I,J,K
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER LENIC
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    real(chm_real) RIJ,RJK,THETA
    INTEGER IIC
    !
    RIJ=0.0
    RJK=0.0
    THETA=0.0
    DO IIC=1,LENIC
       IF(B2IC(IIC) > 0.0) THEN
          IF(IAR(IIC) == I.AND.JAR(IIC) == J.AND..NOT.TAR(IIC)) &
               RIJ=B2IC(IIC)
          IF(IAR(IIC) == J.AND.JAR(IIC) == I.AND..NOT.TAR(IIC)) &
               RIJ=B2IC(IIC)
          IF(IAR(IIC) == I.AND.KAR(IIC) == J.AND.TAR(IIC)) &
               RIJ=B2IC(IIC)
          IF(IAR(IIC) == J.AND.KAR(IIC) == I.AND.TAR(IIC)) &
               RIJ=B2IC(IIC)
          IF(IAR(IIC) == K.AND.JAR(IIC) == J.AND..NOT.TAR(IIC)) &
               RJK=B2IC(IIC)
          IF(IAR(IIC) == J.AND.JAR(IIC) == K.AND..NOT.TAR(IIC)) &
               RJK=B2IC(IIC)
          IF(IAR(IIC) == K.AND.KAR(IIC) == J.AND.TAR(IIC)) &
               RJK=B2IC(IIC)
          IF(IAR(IIC) == J.AND.KAR(IIC) == K.AND.TAR(IIC)) &
               RJK=B2IC(IIC)
       ENDIF
       IF(B1IC(IIC) > 0.0) THEN
          IF(LAR(IIC) == I.AND.KAR(IIC) == J) RIJ=B1IC(IIC)
          IF(LAR(IIC) == J.AND.KAR(IIC) == I) RIJ=B1IC(IIC)
          IF(LAR(IIC) == K.AND.KAR(IIC) == J) RJK=B1IC(IIC)
          IF(LAR(IIC) == J.AND.KAR(IIC) == K) RJK=B1IC(IIC)
       ENDIF
       !
       IF(KAR(IIC) == J) THEN
          IF(T1IC(IIC) > 0.0) THEN
             IF(LAR(IIC) == I.AND.JAR(IIC) == K) THETA=T1IC(IIC)
             IF(LAR(IIC) == K.AND.JAR(IIC) == I) THETA=T1IC(IIC)
          ENDIF
          IF(T2IC(IIC) > 0.0) THEN
             IF(IAR(IIC) == I.AND.JAR(IIC) == K.AND.TAR(IIC)) &
                  THETA=T2IC(IIC)
             IF(IAR(IIC) == K.AND.JAR(IIC) == I.AND.TAR(IIC)) &
                  THETA=T2IC(IIC)
          ENDIF
       ELSE
          IF(JAR(IIC) == J.AND..NOT.TAR(IIC).AND.T2IC(IIC) > 0.0)THEN
             IF(IAR(IIC) == I.AND.KAR(IIC) == K) THETA=T2IC(IIC)
             IF(IAR(IIC) == K.AND.KAR(IIC) == I) THETA=T2IC(IIC)
          ENDIF
       ENDIF
    ENDDO
    !
    IF(X(I) < ANUM.OR.X(J) < ANUM.OR.X(K) < ANUM) THEN
       CALL WRNDIE(0,'<SEED>','COORDINATES ALREADY KNOWN.')
       RETURN
    ENDIF
    IF(RIJ == 0.0.OR.RJK == 0.0.OR.THETA == 0.0) THEN
       !RCZ 91/10/24
       IF(PRNLEV >= 2) WRITE(OUTU,'(A,3(1X,I5),3(1X,F7.2))')  &
            ' IC SEED> I,J,K,RIJ,RJK,THETA=',I,J,K,RIJ,RJK,THETA
       !RCZ 91/10/24
       CALL WRNDIE(0,'<SEED>', &
            'SOME IC VALUES ARE ZERO FOR SEED ATOMS.')
       RETURN
    ENDIF
    !
    X(I)=0.0
    Y(I)=0.0
    Z(I)=0.0
    X(J)=RIJ
    Y(J)=0.0
    Z(J)=0.0
    THETA=THETA*(PI/180.0)
    X(K)=RIJ-RJK*COS(THETA)
    Y(K)=RJK*SIN(THETA)
    Z(K)=0.0
    !
    RETURN
  END SUBROUTINE SEED

  SUBROUTINE FILLIC(LENIC,LAPPE,LPRES,X,Y,Z, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    !     THIS ROUTINE FILLS THE INTERNAL COORDINATE ARRAYS USING X,Y,Z
    !
    !     By Bernard R. Brooks    1982
    !
    use stream
#if KEY_ZEROM==1
    use zdata_mod
#endif
    implicit none
    !
    INTEGER LENIC
    LOGICAL LAPPE,LPRES
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    INTEGER IC,I,J,K,L
    real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
    LOGICAL T
    !
    IF(LPRES.AND.LAPPE) THEN
       CALL WRNDIE(0,'<FILLIC>', &
            'APPEnd and PREServe options are not compatable')
       LPRES=.FALSE.
    ENDIF
    !
    IF(LENIC <= 0) THEN
       CALL WRNDIE(1,'<FILLIC>','FILLIC called with a null IC table')
       RETURN
    ENDIF
    !
    DO IC=1,LENIC
       I=IAR(IC)
       J=JAR(IC)
       K=KAR(IC)
       L=LAR(IC)
       T=TAR(IC)
       IF(LPRES) THEN
          RIJ=B2IC(IC)
          TIJK=T2IC(IC)
          PIJKL=PIC(IC)
          TJKL=T1IC(IC)
          RKL=B1IC(IC)
       ELSE
          RIJ=0.0
          TIJK=0.0
          PIJKL=0.0
          TJKL=0.0
          RKL=0.0
       ENDIF
       !
       CALL GETICV(I,J,K,L,T,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
       IF(LAPPE) THEN
          B2IC(IC)=RIJ+B2IC(IC)
          T2IC(IC)=TIJK+T2IC(IC)
          PIC(IC)=PIJKL+PIC(IC)
          T1IC(IC)=TJKL+T1IC(IC)
          B1IC(IC)=RKL+B1IC(IC)
       ELSE
          B2IC(IC)=RIJ
          T2IC(IC)=TIJK
          PIC(IC)=PIJKL
          T1IC(IC)=TJKL
          B1IC(IC)=RKL
       ENDIF
       !
       IF(PIC(IC) > 180.0) PIC(IC)=PIC(IC)-360.0
       IF(PIC(IC) < -180.0) PIC(IC)=PIC(IC)+360.0
    ENDDO
#if KEY_ZEROM==1
    QICFILL = .TRUE.
#endif
    RETURN
  END SUBROUTINE FILLIC

  SUBROUTINE DIFFIC(DELTA,FX,FY,FZ,X,Y,Z,LENIC,LAPPE, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    !     THIS ROUTINE APPROXIMATES THE DERIVATIVES OF THE INTERNAL
    !     COORDINATES BY FINITE DIFFERENCES.
    !      X,Y,Z - REFERENCE COORDINATES
    !      FX,FY,FZ - DIFFERENCE COORDINATES
    !      DELTA - SCALE FACTOR BY WHICH TO MULTIPLY RESULTS
    !
    !     By Bernard R. Brooks    1982
    !
    !
    use stream
    implicit none
    !
    real(chm_real) DELTA
    real(chm_real) FX(*),FY(*),FZ(*),X(*),Y(*),Z(*)
    INTEGER LENIC
    LOGICAL LAPPE
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    INTEGER IC,I,J,K,L
    real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
    real(chm_real) RIJC,TIJKC,PIJKLC,TJKLC,RKLC
    LOGICAL T
    !
    !
    DO IC=1,LENIC
       I=IAR(IC)
       J=JAR(IC)
       K=KAR(IC)
       L=LAR(IC)
       T=TAR(IC)
       !
       RIJ=0.0
       TIJK=0.0
       PIJKL=0.0
       TJKL=0.0
       RKL=0.0
       RIJC=0.0
       TIJKC=0.0
       PIJKLC=0.0
       TJKLC=0.0
       RKLC=0.0
       !
       CALL GETICV(I,J,K,L,T,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
       CALL GETICV(I,J,K,L,T,RIJC,TIJKC,PIJKLC,TJKLC,RKLC,FX,FY,FZ)
       RIJ=(RIJC-RIJ)*DELTA
       TIJK=(TIJKC-TIJK)*DELTA
       PIJKL=(PIJKLC-PIJKL)
       IF(PIJKL > 180.0) PIJKL=PIJKL-360.0
       IF(PIJKL < -180.0) PIJKL=PIJKL+360.0
       PIJKL=PIJKL*DELTA
       TJKL=(TJKLC-TJKL)*DELTA
       RKL=(RKLC-RKL)*DELTA
       IF(LAPPE) THEN
          B2IC(IC)=RIJ+B2IC(IC)
          T2IC(IC)=TIJK+T2IC(IC)
          PIC(IC)=PIJKL+PIC(IC)
          T1IC(IC)=TJKL+T1IC(IC)
          B1IC(IC)=RKL+B1IC(IC)
       ELSE
          B2IC(IC)=RIJ
          T2IC(IC)=TIJK
          PIC(IC)=PIJKL
          T1IC(IC)=TJKL
          B1IC(IC)=RKL
       ENDIF
       IF(PIC(IC) > 180.0) PIC(IC)=PIC(IC)-360.0
       IF(PIC(IC) < -180.0) PIC(IC)=PIC(IC)+360.0
    ENDDO
    RETURN
  END SUBROUTINE DIFFIC

  SUBROUTINE INTDER(DELTA,DX,DY,DZ,X,Y,Z,LENIC,LAPPE, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    !     THIS ROUTINE GENERATES THE DERIVATIVES OF THE INTERNAL
    !     COORDINATES BY ANALYTIC DERIVATIVES
    !      X,Y,Z - REFERENCE COORDINATES
    !      DX,DY,DZ - COORDINATE DERIVATIVES
    !      DELTA - SCALE FACTOR BY WHICH TO MULTIPLY RESULTS
    !
    !     By Bernard R. Brooks    1982
    !
    !
    use stream
    implicit none
    !
    real(chm_real) DELTA
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    INTEGER LENIC
    LOGICAL LAPPE
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    INTEGER IC,I,J,K,L
    real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
    LOGICAL T
    !
    !
    DO IC=1,LENIC
       I=IAR(IC)
       J=JAR(IC)
       K=KAR(IC)
       L=LAR(IC)
       T=TAR(IC)
       CALL GETICD(I,J,K,L,T,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z,DX,DY,DZ)
       IF(LAPPE) THEN
          B2IC(IC)=B2IC(IC)+RIJ*DELTA
          T2IC(IC)=T2IC(IC)+TIJK*DELTA
          PIC(IC)=PIC(IC)+PIJKL*DELTA
          T1IC(IC)=T1IC(IC)+TJKL*DELTA
          B1IC(IC)=B1IC(IC)+RKL*DELTA
       ELSE
          B2IC(IC)=RIJ*DELTA
          T2IC(IC)=TIJK*DELTA
          PIC(IC)=PIJKL*DELTA
          T1IC(IC)=TJKL*DELTA
          B1IC(IC)=RKL*DELTA
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE INTDER

  SUBROUTINE GETICD(I,J,K,L,T,RIJ,TIJK,PIJKL,TJKL,RKL, &
       X,Y,Z,DX,DY,DZ)
    !
    !     THIS ROUTINE COMPUTES ANALYTIC DERIVATIVES OF INTERNAL
    !     COORDINATES FROM CARTESIAN DERIVATIVES. THE FORMAT MATCHES
    !     THE IC TABLES.
    !
    !     By Bernard R. Brooks    1982
    !
    !
    use number
    use consta
    use exfunc
    use stream
    use chutil,only:initia
    implicit none
    !
    INTEGER I,J,K,L
    LOGICAL T
    real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
    real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    !
    real(chm_real) XI,YI,ZI,DXI,DYI,DZI,XJ,YJ,ZJ,DXJ,DYJ,DZJ
    real(chm_real) XK,YK,ZK,DXK,DYK,DZK,XL,YL,ZL,DXL,DYL,DZL
    real(chm_real) DFX,DFY,DFZ,FX,FY,FZ,FR2,FR,DRIJ,DHX,DHY,DHZ
    real(chm_real) HX,HY,HZ,HR2,HR,DRKL,DEX,DEY,DEZ,EX,EY,EZ,ER2,ER
    real(chm_real) DRIK,DTH1,DTH2,DTH3,DPHI,DGX,DGY,DGZ,GX,GY,GZ
    real(chm_real) GR2,GR,GRR,GXR,GYR,GZR,FRR,FXR,FYR,FZR
    real(chm_real) CST,DTXI,DTXJ,DTYI,DTYJ,DTZI,DTZJ
    real(chm_real) HRR,HXR,HYR,HZR,ERR,EXR,EYR,EZR
    real(chm_real) AX,AY,AZ,BX,BY,BZ,DAX,DAY,DAZ,DBX,DBY,DBZ
    real(chm_real) RA2,RB2,RA,RB,RAR,RBR,AXR,AYR,AZR,BXR,BYR,BZR
    real(chm_real) CX,CY,CZ,SNT,DAXR,DAYR,DAZR,DBXR,DBYR,DBZR
    real(chm_real) DTAX,DTAY,DTAZ,DTBX,DTBY,DTBZ
    LOGICAL LI,LJ,LK,LL,OK
    !
    LI=INITIA(I,X,Y,Z)
    LJ=INITIA(J,X,Y,Z)
    LK=INITIA(K,X,Y,Z)
    LL=INITIA(L,X,Y,Z)
    OK=.TRUE.
    !
    IF(LI) THEN
       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
       DXI=DX(I)
       DYI=DY(I)
       DZI=DZ(I)
    ENDIF
    IF(LJ) THEN
       XJ=X(J)
       YJ=Y(J)
       ZJ=Z(J)
       DXJ=DX(J)
       DYJ=DY(J)
       DZJ=DZ(J)
    ENDIF
    IF(LK) THEN
       XK=X(K)
       YK=Y(K)
       ZK=Z(K)
       DXK=DX(K)
       DYK=DY(K)
       DZK=DZ(K)
    ENDIF
    IF(LL) THEN
       XL=X(L)
       YL=Y(L)
       ZL=Z(L)
       DXL=DX(L)
       DYL=DY(L)
       DZL=DZ(L)
    ENDIF
    !
    ! BOND DERIVATIVES
    !
    IF(LI.AND.LJ) THEN
       DFX=DXI-DXJ
       DFY=DYI-DYJ
       DFZ=DZI-DZJ
       FX=XI-XJ
       FY=YI-YJ
       FZ=ZI-ZJ
       FR2=FX*FX + FY*FY + FZ*FZ
       FR=SQRT(FR2)
       DRIJ=(FX*DFX+FY*DFY+FZ*DFZ)/FR
    ELSE
       DRIJ=0.0
    ENDIF
    !
    IF(LL.AND.LK) THEN
       DHX=DXL-DXK
       DHY=DYL-DYK
       DHZ=DZL-DZK
       HX=XL-XK
       HY=YL-YK
       HZ=ZL-ZK
       HR2=HX*HX + HY*HY + HZ*HZ
       HR=SQRT(HR2)
       DRKL=(HX*DHX+HY*DHY+HZ*DHZ)/HR
    ELSE
       DRKL=0.0
    ENDIF
    !
    IF(T.AND.LI.AND.LK) THEN
       DEX=DXI-DXK
       DEY=DYI-DYK
       DEZ=DZI-DZK
       EX=XI-XK
       EY=YI-YK
       EZ=ZI-ZK
       ER2=EX*EX + EY*EY + EZ*EZ
       ER=SQRT(ER2)
       DRIK=(EX*DEX+EY*DEY+EZ*DEZ)/ER
    ELSE
       DRIK=0.0
    ENDIF
    !
    ! ANGLE DERIVATIVES
    !
    DTH1=0.0
    DTH2=0.0
    DTH3=0.0
    DPHI=0.0
    IF(I == J .OR. I == K) LI=.FALSE.
    IF(L == J .OR. L == K) LL=.FALSE.
    !
    IF(LJ.AND.LK) THEN
       DGX=DXJ-DXK
       DGY=DYJ-DYK
       DGZ=DZJ-DZK
       GX=XJ-XK
       GY=YJ-YK
       GZ=ZJ-ZK
       GR2=GX*GX+GY*GY+GZ*GZ
       GR=SQRT(GR2)
       GRR=1.0/GR
       GXR=GX*GRR
       GYR=GY*GRR
       GZR=GZ*GRR
       IF(LI) THEN
          FRR=1.0/FR
          FXR=FX*FRR
          FYR=FY*FRR
          FZR=FZ*FRR
          CST=FXR*GXR+FYR*GYR+FZR*GZR
          !
          IF(ABS(CST) >= COSMAX) THEN
             CST=SIGN(COSMAX,CST)
             IF(WRNLEV >= 2) WRITE(OUTU,54) I,J,K
54           FORMAT(' WARNING FROM INTDER. ANGLE IS ALMOST LINEAR.', &
                  3I5)
             OK=.FALSE.
          ENDIF
          !
          DTXI=FRR*(GXR-CST*FXR)
          DTXJ=GRR*(FXR-CST*GXR)
          DTYI=FRR*(GYR-CST*FYR)
          DTYJ=GRR*(FYR-CST*GYR)
          DTZI=FRR*(GZR-CST*FZR)
          DTZJ=GRR*(FZR-CST*GZR)
          !
          DTH1=DTXI*DFX+DTYI*DFY+DTZI*DFZ+DTXJ*DGX+DTYJ*DGY+DTZJ*DGZ
          DTH1=DTH1*SQRT(ONE/(ONE-CST*CST))
       ENDIF
       !
       !
       IF(LL) THEN
          HRR=1.0/HR
          HXR=HX*HRR
          HYR=HY*HRR
          HZR=HZ*HRR
          CST=HXR*GXR+HYR*GYR+HZR*GZR
          !
          IF(ABS(CST) >= COSMAX) THEN
             CST=SIGN(COSMAX,CST)
             IF(WRNLEV >= 2) WRITE(OUTU,54) J,K,L
             OK=.FALSE.
          ENDIF
          !
          DTXI=HRR*(GXR-CST*HXR)
          DTXJ=GRR*(HXR-CST*GXR)
          DTYI=HRR*(GYR-CST*HYR)
          DTYJ=GRR*(HYR-CST*GYR)
          DTZI=HRR*(GZR-CST*HZR)
          DTZJ=GRR*(HZR-CST*GZR)
          !
          DTH2=DTXI*DHX+DTYI*DHY+DTZI*DHZ+DTXJ*DGX+DTYJ*DGY+DTZJ*DGZ
          DTH2=-DTH2*SQRT(ONE/(ONE-CST*CST))
       ENDIF
       !
       IF(T.AND.LI) THEN
          ERR=1.0/ER
          EXR=EX*ERR
          EYR=EY*ERR
          EZR=EZ*ERR
          CST=EXR*GXR+EYR*GYR+EZR*GZR
          !
          IF(ABS(CST) >= COSMAX) THEN
             CST=SIGN(COSMAX,CST)
             IF(WRNLEV >= 2) WRITE(OUTU,54) I,K,J
             OK=.FALSE.
          ENDIF
          !
          DTXI=ERR*(GXR-CST*EXR)
          DTXJ=GRR*(EXR-CST*GXR)
          DTYI=ERR*(GYR-CST*EYR)
          DTYJ=GRR*(EYR-CST*GYR)
          DTZI=ERR*(GZR-CST*EZR)
          DTZJ=GRR*(EZR-CST*GZR)
          !
          DTH3=DTXI*DEX+DTYI*DEY+DTZI*DEZ+DTXJ*DGX+DTYJ*DGY+DTZJ*DGZ
          DTH3=-DTH3*SQRT(ONE/(ONE-CST*CST))
       ENDIF
       !
       ! DIHEDRAL DERIVATIVES
       !
       DPHI=0.0
       IF(OK) THEN
          !
          IF(LI.AND.LL) THEN
             AX=FY*GZ-FZ*GY
             AY=FZ*GX-FX*GZ
             AZ=FX*GY-FY*GX
             BX=HY*GZ-HZ*GY
             BY=HZ*GX-HX*GZ
             BZ=HX*GY-HY*GX
             DAX=DFY*GZ+FY*DGZ-DFZ*GY-FZ*DGY
             DAY=DFZ*GX+FZ*DGX-DFX*GZ-FX*DGZ
             DAZ=DFX*GY+FX*DGY-DFY*GX-FY*DGX
             DBX=DHY*GZ+HY*DGZ-DHZ*GY-HZ*DGY
             DBY=DHZ*GX+HZ*DGX-DHX*GZ-HX*DGZ
             DBZ=DHX*GY+HX*DGY-DHY*GX-HY*DGX
             !
             RA2=AX*AX+AY*AY+AZ*AZ
             RB2=BX*BX+BY*BY+BZ*BZ
             RA=SQRT(RA2)
             RB=SQRT(RB2)
             IF(RA <= 0.01) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,10) I,J,K,L
10              FORMAT(' WARNING FROM INTDER.  PHI IS ALMOST LINEAR.', &
                     4I4)
                RA=0.01
             ENDIF
             RAR=1.0/RA
             IF(RB <= 0.01) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,10) I,J,K,L
                RB=0.01
             ENDIF
             RBR=1.0/RB
             !
             !
             AXR=AX*RAR
             AYR=AY*RAR
             AZR=AZ*RAR
             BXR=BX*RBR
             BYR=BY*RBR
             BZR=BZ*RBR
             !
             CX=AYR*BZR-AZR*BYR
             CY=AZR*BXR-AXR*BZR
             CZ=AXR*BYR-AYR*BXR
             SNT=SQRT(CX*CX+CY*CY+CZ*CZ)
             IF(GX*CX+GY*CY+GZ*CZ > 0.0) SNT=-SNT
             !
             CST=AXR*BXR+AYR*BYR+AZR*BZR
             IF(ABS(CST) >= 1.0) CST=SIGN(ONE,CST)
             !
             !
             IF(ABS(CST) > 0.6) THEN
                !
                ! USE PART(PHI)/PART(SIN(PHI)) IN CHAIN RULE
                !
                DAXR=RAR*(DAX*(1.0-AXR*AXR)-DAY*AXR*AYR-DAZ*AXR*AZR)
                DAYR=RAR*(DAY*(1.0-AYR*AYR)-DAX*AYR*AXR-DAZ*AYR*AZR)
                DAZR=RAR*(DAZ*(1.0-AZR*AZR)-DAY*AZR*AYR-DAX*AZR*AXR)
                DBXR=RBR*(DBX*(1.0-BXR*BXR)-DBY*BXR*BYR-DBZ*BXR*BZR)
                DBYR=RBR*(DBY*(1.0-BYR*BYR)-DBX*BYR*BXR-DBZ*BYR*BZR)
                DBZR=RBR*(DBZ*(1.0-BZR*BZR)-DBY*BZR*BYR-DBX*BZR*BXR)
                !
                FXR=GYR*AZR-GZR*AYR
                FYR=GZR*AXR-GXR*AZR
                FZR=GXR*AYR-GYR*AXR
                HXR=BYR*GZR-BZR*GYR
                HYR=BZR*GXR-BXR*GZR
                HZR=BXR*GYR-BYR*GXR
                !
                DPHI=DAXR*HXR+DAYR*HYR+DAZR*HZR+DBXR*FXR+DBYR*FYR+ &
                     DBZR*FZR
                DPHI=-DPHI/CST
             ELSE
                !
                ! USE PART(PHI)/PART(COS(PHI)) IN CHAIN RULE
                !
                DTAX=RAR*(BXR-CST*AXR)
                DTAY=RAR*(BYR-CST*AYR)
                DTAZ=RAR*(BZR-CST*AZR)
                DTBX=RBR*(AXR-CST*BXR)
                DTBY=RBR*(AYR-CST*BYR)
                DTBZ=RBR*(AZR-CST*BZR)
                !
                DPHI=DTAX*DAX+DTAY*DAY+DTAZ*DAZ+DTBX*DBX+DTBY*DBY+ &
                     DTBZ*DBZ
                DPHI=-DPHI/SNT
             ENDIF
             !
          ENDIF
       ENDIF
    ENDIF
    !
    IF(T) THEN
       RIJ=DRIK
       TIJK=DTH3*RADDEG
    ELSE
       RIJ=DRIJ
       TIJK=DTH1*RADDEG
    ENDIF
    PIJKL=DPHI*RADDEG
    TJKL=DTH2*RADDEG
    RKL=DRKL
    !
    RETURN
  END SUBROUTINE GETICD

  SUBROUTINE GETICV(I,J,K,L,T,RIJ,TIJK,PIJKL,TJKL,RKL, &
       X,Y,Z)
    !
    !     THIS ROUTINE COMPUTES GEOMETRIC VALUES OF INTERNAL
    !     COORDINATES FROM CARTESIAN COORDINATES. THE FORMAT MATCHES
    !     THE IC TABLES.
    !     Only one entry is done for each call. The variables RIJ,...RKL
    !     must be filled with default values before calling this routine.
    !
    !     By Bernard R. Brooks    1982
    !
    use number
    use consta
    use exfunc
    use stream
    use chutil,only:initia
    implicit none
    !
    INTEGER I,J,K,L
    LOGICAL T
    real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
    real(chm_real) X(*),Y(*),Z(*)
    !
    real(chm_real) XI,YI,ZI,XJ,YJ,ZJ,XK,YK,ZK,XL,YL,ZL
    real(chm_real) FX,FY,FZ,FR2,FR,HX,HY,HZ,HR2,HR,EX,EY,EZ,ER2,ER
    real(chm_real) RIK,TH1,TH2,TH3,PHI,GX,GY,GZ,GR2,GR,GRR,GXR,GYR,GZR
    real(chm_real) FRR,FXR,FYR,FZR,CST,HRR,HXR,HYR,HZR,ERR,EXR,EYR,EZR
    real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA,RB,RAR,RBR,AXR,AYR,AZR
    real(chm_real) BXR,BYR,BZR,CX,CY,CZ
    LOGICAL LI,LJ,LK,LL,OK
    real(chm_real) :: AMARK=-9999.0_chm_real
    !
    LI=INITIA(I,X,Y,Z)
    LJ=INITIA(J,X,Y,Z)
    LK=INITIA(K,X,Y,Z)
    LL=INITIA(L,X,Y,Z)
    OK=.TRUE.
    !
    IF(LI) THEN
       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
    ENDIF
    IF(LJ) THEN
       XJ=X(J)
       YJ=Y(J)
       ZJ=Z(J)
    ENDIF
    IF(LK) THEN
       XK=X(K)
       YK=Y(K)
       ZK=Z(K)
    ENDIF
    IF(LL) THEN
       XL=X(L)
       YL=Y(L)
       ZL=Z(L)
    ENDIF
    !
    RIK=RIJ
    !
    ! BOND DISTANCES
    !
    IF(LI.AND.LJ) THEN
       FX=XI-XJ
       FY=YI-YJ
       FZ=ZI-ZJ
       FR2=FX*FX + FY*FY + FZ*FZ
       FR=SQRT(FR2)
       RIJ=FR
    ENDIF
    !
    IF(LL.AND.LK) THEN
       HX=XL-XK
       HY=YL-YK
       HZ=ZL-ZK
       HR2=HX*HX + HY*HY + HZ*HZ
       HR=SQRT(HR2)
       RKL=HR
    ENDIF
    !
    IF(T.AND.LI.AND.LK) THEN
       EX=XI-XK
       EY=YI-YK
       EZ=ZI-ZK
       ER2=EX*EX + EY*EY + EZ*EZ
       ER=SQRT(ER2)
       RIK=ER
    ENDIF
    !
    ! ANGLE VALUES
    !
    TH1=AMARK
    TH2=AMARK
    TH3=AMARK
    PHI=AMARK
    IF(I == J .OR. I == K) LI=.FALSE.
    IF(L == J .OR. L == K) LL=.FALSE.
    !
    IF(LJ.AND.LK) THEN
       GX=XJ-XK
       GY=YJ-YK
       GZ=ZJ-ZK
       GR2=GX*GX+GY*GY+GZ*GZ
       GR=SQRT(GR2)
       GRR=1.0/GR
       GXR=GX*GRR
       GYR=GY*GRR
       GZR=GZ*GRR
       IF(LI) THEN
          FRR=1.0/FR
          FXR=FX*FRR
          FYR=FY*FRR
          FZR=FZ*FRR
          CST=-FXR*GXR-FYR*GYR-FZR*GZR
          !
          IF(ABS(CST) >= COSMAX) THEN
             CST=SIGN(COSMAX,CST)
             IF(WRNLEV >= 2) WRITE(OUTU,54) I,J,K
54           FORMAT(' WARNING FROM GETICV. ANGLE OF ATOMS',3I5, &
                  ' IS ALMOST LINEAR.')
             OK=.FALSE.
          ENDIF
          TH1=ACOS(CST)
       ENDIF
       !
       !
       IF(LL) THEN
          HRR=1.0/HR
          HXR=HX*HRR
          HYR=HY*HRR
          HZR=HZ*HRR
          CST=HXR*GXR+HYR*GYR+HZR*GZR
          !
          IF(ABS(CST) >= COSMAX) THEN
             CST=SIGN(COSMAX,CST)
             IF(WRNLEV >= 2) WRITE(OUTU,54) J,K,L
             OK=.FALSE.
          ENDIF
          TH2=ACOS(CST)
       ENDIF
       !
       !
       IF(T.AND.LI) THEN
          ERR=1.0/ER
          EXR=EX*ERR
          EYR=EY*ERR
          EZR=EZ*ERR
          CST=EXR*GXR+EYR*GYR+EZR*GZR
          !
          IF(ABS(CST) >= COSMAX) THEN
             CST=SIGN(COSMAX,CST)
             IF(WRNLEV >= 2) WRITE(OUTU,54) I,K,J
             OK=.FALSE.
          ENDIF
          TH3=ACOS(CST)
       ENDIF
       !
       ! DIHEDRAL VALUES
       !
       IF(OK) THEN
          !
          IF(LI.AND.LL) THEN
             AX=FY*GZ-FZ*GY
             AY=FZ*GX-FX*GZ
             AZ=FX*GY-FY*GX
             BX=HY*GZ-HZ*GY
             BY=HZ*GX-HX*GZ
             BZ=HX*GY-HY*GX
             !
             RA2=AX*AX+AY*AY+AZ*AZ
             RB2=BX*BX+BY*BY+BZ*BZ
             RA=SQRT(RA2)
             RB=SQRT(RB2)
             IF(RA <= 0.01) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,10) I,J,K,L
10              FORMAT(' WARNING FROM GETICV.  PHI IS ALMOST LINEAR.', &
                     4I4)
                RA=0.01
             ENDIF
             RAR=1.0/RA
             IF(RB <= 0.01) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,10) I,J,K,L
                RB=0.01
             ENDIF
             RBR=1.0/RB
             !
             !
             AXR=AX*RAR
             AYR=AY*RAR
             AZR=AZ*RAR
             BXR=BX*RBR
             BYR=BY*RBR
             BZR=BZ*RBR
             !
             CST=AXR*BXR+AYR*BYR+AZR*BZR
             IF(ABS(CST) >= 1.0) CST=SIGN(ONE,CST)
             PHI=ACOS(CST)
             CX=AYR*BZR-AZR*BYR
             CY=AZR*BXR-AXR*BZR
             CZ=AXR*BYR-AYR*BXR
             IF(GX*CX+GY*CY+GZ*CZ > 0.0) PHI=-PHI
          ENDIF
       ENDIF
    ENDIF
    !
    !
    IF(T) THEN
       RIJ=RIK
       TH1=TH3
    ENDIF
    IF(TH1 /= AMARK) TIJK=TH1*RADDEG
    IF(TH2 /= AMARK) TJKL=TH2*RADDEG
    IF(PHI /= AMARK) PIJKL=PHI*RADDEG
    !
    RETURN
  END SUBROUTINE GETICV

  SUBROUTINE AUTGENIC(NATOM,NBOND,NATBON,IATBON,ISLCT,IDONE, &
       ICOUNT,AMASS,IB,JB,QTHREE,QRTF,INTLEN, &
       LENIC,B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    !     This routine automatically generates IC table entries for
    !     a set of selected atoms.  There is no checking for duplicate
    !     entries in the IC table.  This routine will add a complete
    !     set sufficient to: IC SEED, IC FILL PARAM, IC BUILD.
    !
    !     By Bernard R. Brooks  Febriary 27, 1998
    !
    use exfunc
    use dimens_fcm
    use number
    use stream
    use machutil,only:die
    implicit none
    !
    INTEGER NATOM,NBOND
    INTEGER NATBON(NATOM),IATBON(IATBMX,NATOM)
    INTEGER ISLCT(NATOM),IDONE(NATOM),ICOUNT(NATOM)
    real(chm_real)  AMASS(NATOM)
    INTEGER IB(NBOND),JB(NBOND)
    LOGICAL QTHREE,QRTF
    INTEGER INTLEN,LENIC
    real(chm_real)  B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    !
    INTEGER I,IBT,JBT,J,K,KBT,LBT,ISTART
    INTEGER IOPEN,NOPEN,NDONE,ICARD
    real(chm_real) WEIGHT(IATBMX),AM,PHI
    LOGICAL MORE
    !
    INTEGER :: MARK=-99
    !
    ISTART=LENIC+1
    ICARD=0
    IF(QRTF) ICARD=3
    !
    ! Construct the cross reference bond list
    !
    DO I=1,NATOM
       NATBON(I)=0
       IDONE(I)=0
       IF(ISLCT(I) /= 1) IDONE(I)=MARK ! flag unselected atoms as done
    ENDDO
    DO I=1,NBOND
       IBT=IB(I)
       JBT=JB(I)
       IF(IBT > 0 .AND. JBT > 0) THEN
          NATBON(IBT)=NATBON(IBT)+1
          IF(NATBON(IBT) > IATBMX) THEN
             IF(WRNLEV >= 2) WRITE(OUTU,335) IBT
335          FORMAT(' <AUTGENIC>: Too many bonds for atom',I5, &
                  ' No IC table entries added')
             CALL DIEWRN(-1)
             RETURN
          ENDIF
          IATBON(NATBON(IBT),IBT)=JBT
          NATBON(JBT)=NATBON(JBT)+1
          IF(NATBON(JBT) > IATBMX) THEN
             IF(WRNLEV >= 2) WRITE(OUTU,335) JBT
             CALL DIEWRN(-1)
             RETURN
          ENDIF
          IATBON(NATBON(JBT),JBT)=IBT
       ENDIF
    ENDDO
    !
#if KEY_DEBUG==1
    write(6,88) 'natom nbond',natom,nbond
    write(6,88) 'natbon',natbon
    write(6,88) 'iatbon',iatbon
87  format('AUTGENIC:----------------------------------')
88  format('AUTGENIC: ',A,(/16I5))
89  format('AUTGENIC: ',A,16I5)
90  format('AUTGENIC: ',A,8F10.2)
#endif
    !
    ! all is setup for the next pass....
    !
100 CONTINUE
#if KEY_DEBUG==1
    write(6,87)
    write(6,89) 'Starting search at 100 entry'
#endif
    !
    !  find first "not done" selected atom which is connected to a done
    !  atom.  If found: do improper add sequence
    !         If not found: do fragment add sequence
    !
    ! search for a "not done" atom bonded to a "done" atom
    IBT=0
    DO I=1,NATOM
       IF(ISLCT(I) == 1) THEN
          IF(IDONE(I) == 0) THEN
             IF(IBT == 0) IBT=I
             JBT=0
             DO J=1,NATBON(I)
                K=IATBON(J,I)
                IF(IDONE(K) > 0) THEN
                   JBT=K
                   GOTO 200
                ENDIF
                IF(IDONE(K) == MARK .AND. JBT == 0) JBT=K
             ENDDO
             IF(JBT > 0) GOTO 200
          ENDIF
       ENDIF
    ENDDO
    !
#if KEY_DEBUG==1
    write(6,89) 'IBT at not found',IBT
#endif
    !
    IF(IBT == 0) THEN  ! all atoms are now done
       IF(ISTART <= LENIC) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,605)
605       FORMAT(/15X,'Generation complete. New IC table entries:')
          CALL WRITIC(ISTART,LENIC,-1,ICARD,OUTU, &
               B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
       ENDIF
       RETURN
    ENDIF
    !
    ! No atom was found that was bonded to a "done" atom.  The first
    ! not done atom is IBT.  Use this atom as a seed (unless finished)...
    !
    JBT=IBT
    IDONE(JBT)=MARK !  atom has no predecessor
    !
    ! Process all of the atoms bonded to atom JBT
200 CONTINUE
    !
#if KEY_DEBUG==1
    write(6,89) 'JBT at 200 enter',JBT
#endif
    !
    ! How many things are hanging off of JBT
    !
    NDONE=0
    NOPEN=0
    DO I=1,NATBON(JBT)
       IF(IDONE(IATBON(I,JBT)) == 0) THEN
          NOPEN=NOPEN+1
       ELSE
          NDONE=NDONE+1
       ENDIF
    ENDDO
    !
#if KEY_DEBUG==1
    write(6,89) 'NOPEN,NDONE',NOPEN,NDONE
#endif
    !
    ! If NOPEN=0 isolated molecule
    !    NOPEN=1 one branch needs extending
    !    NOPEN>1 competing branches (do largest, then do others)
    !
    !    NDONE=0 start a new fragment (no connects to any existing IC)
    !    NDONE=1 one back connection do NORMAL DIHEDRAL
    !    NDONE>1 multiple connections back connection do IMPROPER DIHEDRAL
    !
    !
    IF(NOPEN == 0) GOTO 100 ! Done with this atom (or isolated atom)
    IF(NOPEN == 1) THEN
       DO I=1,NATBON(JBT)
          IF(IDONE(IATBON(I,JBT)) == 0) IBT=IATBON(I,JBT)
       ENDDO
    ELSE
       !       find the heaviest chain
       DO I=1,NATOM
          ICOUNT(I)=0
          IF(IDONE(I) /= 0) ICOUNT(I)=MARK
       ENDDO
       ICOUNT(JBT)=MARK  ! just in case...
       IOPEN=0
       DO I=1,NATBON(JBT)
          IF(IDONE(IATBON(I,JBT)) == 0) THEN
             IOPEN=IOPEN+1
             ICOUNT(IATBON(I,JBT))=IOPEN
          ENDIF
       ENDDO
       !
250    CONTINUE
       MORE=.FALSE.
       ! Do a limited forward sweep (slow, but better results expected)
       DO I=1,NATOM
          IF(ICOUNT(I) == 0) THEN
             DO J=1,NATBON(I)
                IF(ICOUNT(IATBON(J,I)) > 0) THEN
                   ICOUNT(I)=-ICOUNT(IATBON(J,I))
                   MORE=.TRUE.
                ENDIF
             ENDDO
          ENDIF
       ENDDO
       DO I=1,NATOM
          IF(ICOUNT(I) < 0) THEN
             IF(ICOUNT(I) /= MARK) ICOUNT(I)=-ICOUNT(I)
          ENDIF
       ENDDO
       IF(MORE) GOTO 250
       !
       DO I=1,NOPEN
          WEIGHT(I)=ZERO
       ENDDO
       DO I=1,NATOM
          IF(ICOUNT(I) > 0) THEN
             J=ICOUNT(I)
             AM=HALF
             IF(AMASS(I) > HALF) AM=AMASS(I)
             WEIGHT(J)=WEIGHT(J)+AM
          ENDIF
       ENDDO
       !
#if KEY_DEBUG==1
       write(6,90) 'weights:',(weight(i),i=1,nopen)
#endif
       !
       AM=ZERO
       IOPEN=0
       DO I=1,NOPEN
          IF(WEIGHT(I) > AM) THEN
             IOPEN=I
             AM=WEIGHT(I)
          ENDIF
       ENDDO
       !       the heaviest chain is IOPEN
       IF(IOPEN == 0) CALL DIE
       J=0
       DO I=1,NATBON(JBT)
          IF(IDONE(IATBON(I,JBT)) == 0) THEN
             J=J+1
             IF(J == IOPEN) IBT=IATBON(I,JBT)
          ENDIF
       ENDDO
    ENDIF
    !
#if KEY_DEBUG==1
    write(6,89) 'Best IBT for JBT:',IBT,JBT
#endif
    !
    ! OK, Now place IBT relative to JBT
    !
    IF(NDONE == 0) THEN
       !     There is no chain going back from JBT.  This is
       !     part of the seed.  No need to do more...
       IDONE(IBT)=JBT

#if KEY_DEBUG==1
       write(6,89) 'Reject IC:', 0,0,JBT,IBT,NDONE
#endif

       IF(NOPEN > 1) GOTO 200 ! do next chain on JBT
       GOTO 100 ! Done with JBT
    ENDIF

    !     find first predecessor
    KBT=IDONE(JBT)
    IF(KBT == MARK) THEN
       DO I=1,NATBON(JBT)
          IF(IDONE(IATBON(I,JBT)) /= 0 .AND. KBT == MARK) THEN
             KBT=IATBON(I,JBT)
             IDONE(JBT)=KBT
          ENDIF
       ENDDO
    ENDIF
    IF(KBT == MARK) THEN
       !       the path back does not look good, it must be part of the seed.
       IDONE(IBT)=JBT

#if KEY_DEBUG==1
       write(6,89) 'Reject IC:', 0,KBT,JBT,IBT,NDONE
#endif

       IF(NOPEN > 1) GOTO 200 ! do next chain on JBT
       GOTO 100 ! Done with JBT
    ENDIF
    !
    PHI=180.0
    IF(NDONE == 1) THEN
       !       There is only one chain going back from JBT use this with
       !       a normal dihedral type:   x2   x1   jbt  ibt
       LBT=IDONE(KBT)
       IF(LBT == JBT) LBT=MARK
       IF(LBT == MARK) THEN
          DO I=1,NATBON(KBT)
             IF(IDONE(IATBON(I,KBT)) /= 0 .AND. LBT == MARK) THEN
                IF(IATBON(I,KBT) /= JBT) THEN
                   LBT=IATBON(I,KBT)
                   IDONE(KBT)=LBT
                ENDIF
             ENDIF
          ENDDO
       ENDIF
    ELSE
       !       There are multiple chains going back from JBT. Use
       !       improper definition type:  x0  x1   *jbt  ibt
       LBT=MARK
       DO I=1,NATBON(JBT)
          IF(IDONE(IATBON(I,JBT)) /= 0 .AND. LBT == MARK) THEN
             IF(IATBON(I,JBT) /= KBT) LBT=IATBON(I,JBT)
          ENDIF
       ENDDO
       !       make a crude guess at the torsion value
       PHI=-360.0*NOPEN/(NOPEN+NDONE-1)
       IF(PHI <= -180.0) PHI=PHI+360.0
    ENDIF
    IDONE(IBT)=JBT
    IF(LBT /= MARK .OR. QTHREE) THEN
       !       add one element
       IF(LENIC >= INTLEN) THEN
          CALL WRNDIE(-3,'<AUTGENIC>','Overflow of IC table entries')
          RETURN
       ENDIF
       LENIC=LENIC+1
       B1IC(LENIC)=ZERO
       B2IC(LENIC)=ZERO
       T1IC(LENIC)=ZERO
       T2IC(LENIC)=ZERO
       PIC(LENIC)=PHI
       IAR(LENIC)=LBT
       JAR(LENIC)=KBT
       KAR(LENIC)=JBT
       LAR(LENIC)=IBT
       TAR(LENIC)=(NDONE > 1)

#if KEY_DEBUG==1
       write(6,89) 'Adding IC:', LBT,KBT,JBT,IBT,NDONE
    ELSE
       write(6,89) 'Reject IC:', LBT,KBT,JBT,IBT,NDONE
#endif
    ENDIF
    !
    IF(NOPEN > 1) GOTO 200 ! do next chain on JBT
    GOTO 100 ! Done with JBT
    !
  END SUBROUTINE AUTGENIC

  SUBROUTINE FINDIC(NSTART,NSTOP,IAR,JAR,KAR,LAR, &
       ATOMN,FIRSIC,LASTIC)
    !
    !     FOR A GIVEN ATOM, THIS ROUTINE FINDS THE APPROPRIATE
    !     START (FIRSIC) AND STOP (LASTIC) IC TABLE PARAMETERS (i.e.
    !     first and last entries in the table that contain that atom).
    !          -RJ Petrella 2001
    !
    !
    use stream
    implicit none
    !
    INTEGER NSTART,NSTOP
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    INTEGER ATOMN,FIRSIC,LASTIC
    !
    INTEGER KK,II,I,J,K,L
    !
    IF(NSTOP < NSTART) THEN
       CALL WRNDIE(1,'<FINDIC>','FINDIC called with a null IC table')
       RETURN
    ENDIF
    !
    KK=NSTART-1
    FIRSIC=999999999
    LASTIC=-1

    DO II=NSTART,NSTOP
       KK=KK+1
       !
       I=IAR(KK)
       J=JAR(KK)
       K=KAR(KK)
       L=LAR(KK)
       IF((I == ATOMN).OR.(J == ATOMN).OR.(K == ATOMN).OR. &
            (L == ATOMN)) THEN
          IF(KK < FIRSIC) FIRSIC = KK
          IF(KK > LASTIC) LASTIC = KK
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE FINDIC
  SUBROUTINE WRITIC(ISTART,ISTOP,IPRINT,ICARD,IUNIT, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)

    !     THIS ROUTINE WRITES AN INTERNAL COORDINATE FILE MODULE
    !
    !     By Bernard R. Brooks    1982

    use chm_kinds
    use dimens_fcm
    use exfunc
    use ctitla
    use psf
    use stream
    use memory
    use chutil,only:getres,atomid
    !
    implicit none
    !
    INTEGER ISTART,ISTOP,IPRINT,ICARD,IUNIT
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    INTEGER N,I,IIRES,JJRES,KKRES,LLRES
    INTEGER II,JJ,KK,LL
    INTEGER IIX,JJX,KKX,LLX
    INTEGER ICNTRL(20)
    CHARACTER(len=8) AII,AJJ,AKK,ALL,WRD
    CHARACTER(len=8) AIS,AIR,AJS,AJR,AKS,AKR,ALS,ALR
    CHARACTER(len=1) :: STAR='*'
    character(len=40) fmt3,fmt3b
    character(len=60) fmt4
    CHARACTER(len=8) :: QUES='??  ',HDR='IC  '

    IF(IUNIT /= OUTU .AND. IOLEV < 0) RETURN
    IF(IUNIT == OUTU .AND. PRNLEV < 3) RETURN

    IF(IUNIT /= OUTU .AND.PRNLEV >= 2) WRITE(OUTU,822) IUNIT
822 FORMAT(' INTERNAL COORDINATES WRITTEN TO UNIT',I3)

    ! write(0,*)"istart, istop", istart, istop
    N=ISTOP-ISTART+1
    DO I=1,20
       ICNTRL(I)=0
    ENDDO

    ICNTRL(1)=20
    if (qextfmt) then
       icntrl(1)=30           ! yw 29-Jan-2003 indicate extended format
       fmt3='(I9,1X,4(I5,1X,A8),F9.4,3F8.2,F9.4)'
       fmt4='(I10,4(1X,A8,1X,A8,1X,A8,'':''),F12.6,3F12.4,F12.6)'
    else
       fmt3='(I5,1X,4(I3,1X,A4),F9.4,3F8.2,F9.4)'
       fmt4='(I5,4(1X,A4,1X,A4,1X,A4,'':''),F12.6,3F12.4,F12.6)'
    endif
    fmt3b='('' IC    '',4A8,1X,F9.4,3F8.2,F9.4)' ! ic print rtf always need extended format, HJ

    ICNTRL(2)=ICARD
    IF(IPRINT == 0.AND.ICARD == 0) THEN
       ! write-binary-file
       WRITE(IUNIT) HDR,ICNTRL
       CALL WRTITL(TITLEA,NTITLA,IUNIT,-1)
       WRITE(IUNIT) N
       WRITE(IUNIT) (IAR(I),I=1,N)
       WRITE(IUNIT) (JAR(I),I=1,N)
       WRITE(IUNIT) (KAR(I),I=1,N)
       WRITE(IUNIT) (LAR(I),I=1,N)
       WRITE(IUNIT) (TAR(I),I=1,N)
       WRITE(IUNIT) (B1IC(I),I=1,N)
       WRITE(IUNIT) (B2IC(I),I=1,N)
       WRITE(IUNIT) (T1IC(I),I=1,N)
       WRITE(IUNIT) (T2IC(I),I=1,N)
       WRITE(IUNIT) (PIC(I),I=1,N)
    ELSE
       ! write-card-file
       IF(IPRINT /= 0) THEN
          IF(IPRINT >= 0) THEN
             WRITE(IUNIT,1)
1            FORMAT(/7X,'INTERNAL COORDINATES')
             CALL WRTITL(TITLEA,NTITLA,OUTU,1)
             WRITE(IUNIT,24) N
24           FORMAT(/5X,'NUMBER OF INTERNAL COORDINATES : ',I10)
          ENDIF
          if (qextfmt) then
             write(iunit,'(/,3a,/)') &
                  '        N      I             J             K ', &
                  '            L        R(I(J/K))', &
                  ' T(I(JK/KJ)) PHI   T(JKL)   R(KL)'
          else
             write(iunit,'(/,2a,/)') &
                  '   N     I       J       K       L   R(I(J/K))', &
                  ' T(I(JK/KJ)) PHI   T(JKL)   R(KL)'
          endif
       ELSE
          CALL WRTITL(TITLEA,NTITLA,IUNIT,0)
          WRITE(IUNIT,27) ICNTRL
27        FORMAT(20I4)
          WRITE(IUNIT,28) N,ICARD
28        FORMAT(16I5)
       ENDIF
       IF(ICARD /= 2) THEN
          !           process normal IC table information
          DO I=ISTART,ISTOP
             IIRES=-99
             AII=QUES
             JJRES=-99
             AJJ=QUES
             KKRES=-99
             AKK=QUES
             LLRES=-99
             ALL=QUES

             II=IAR(I)
             IF(II > 0) THEN
                IIRES=GETRES(II,IBASE,NREST)
                AII=ATYPE(II)
             ENDIF
             JJ=JAR(I)
             IF(JJ > 0) THEN
                JJRES=GETRES(JJ,IBASE,NREST)
                AJJ=ATYPE(JJ)
             ENDIF
             KK=KAR(I)
             IF(KK > 0) THEN
                KKRES=GETRES(KK,IBASE,NREST)
                AKK=ATYPE(KK)
                IF(TAR(I)) THEN
                   WRD=STAR//AKK
                   AKK=WRD
                ENDIF
             ENDIF
             LL=LAR(I)
             IF(LL > 0) THEN
                LLRES=GETRES(LL,IBASE,NREST)
                ALL=ATYPE(LL)
             ENDIF
             ! is there a difference when ICARD==0? LNI
             IF(ICARD == 1 .OR. ICARD == 0 )THEN
               WRITE(IUNIT,fmt3) I,IIRES, &
                  AII,JJRES,AJJ,KKRES,AKK,LLRES,ALL, &
                  B2IC(I),T2IC(I),PIC(I),T1IC(I),B1IC(I)
             ELSE
             !ICARD==3
               WRITE(IUNIT,fmt3b) AII,AJJ,AKK,ALL, &
                  B2IC(I),T2IC(I),PIC(I),T1IC(I),B1IC(I)
             ENDIF
          ENDDO
       ELSE
          !           write out using resid/segid information
          DO I=ISTART,ISTOP
             AIR=QUES
             AIS=QUES
             AJR=QUES
             AJS=QUES
             AKR=QUES
             AKS=QUES
             ALR=QUES
             ALS=QUES
             AII=QUES
             AJJ=QUES
             AKK=QUES
             ALL=QUES
             !
             IIX=IAR(I)
             IF(IIX > 0) CALL ATOMID(IIX,AIS,AIR,WRD,AII)
             JJX=JAR(I)
             IF(JJX > 0) CALL ATOMID(JJX,AJS,AJR,WRD,AJJ)
             KKX=KAR(I)
             IF(KKX > 0) THEN
                CALL ATOMID(KKX,AKS,AKR,WRD,AKK)
                IF(TAR(I)) THEN
                   WRD=STAR//AKK
                   AKK=WRD
                ENDIF
             ENDIF
             LLX=LAR(I)
             IF(LLX > 0) CALL ATOMID(LLX,ALS,ALR,WRD,ALL)
             WRITE(IUNIT,fmt4) I,AIS,AIR,AII,AJS,AJR,AJJ, &
                  AKS,AKR,AKK,ALS,ALR,ALL, &
                  B2IC(I),T2IC(I),PIC(I),T1IC(I),B1IC(I)
          ENDDO
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE WRITIC

  subroutine MK_ICACTV(NSTART,NSTOP,X,Y,Z, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR,NATOMX,ISLCT,NSLCT, &
    ICACTV,NICACTV)
    use memory
    use parallel,only: MYNODP
    implicit none
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real),intent(in),dimension(:) :: B1IC,B2IC,T1IC,T2IC,PIC
    integer,intent(in),dimension(:) :: IAR,JAR,KAR,LAR,ISLCT
    logical,intent(in),dimension(:) :: TAR
    integer,intent(in) :: NSTART,NSTOP,NATOMX,NSLCT
    integer,intent(out),dimension(:),allocatable :: ICACTV  !active IC table
    integer,intent(out) :: NICACTV !number of lines in active ic table
! local
    integer,dimension(:),allocatable :: WRKATM,WRKIC
    integer :: I,J,K,L,KK,II
    logical :: QVERBOSE=.false.

    call chmalloc('intcor2.src','MK_ICACTIV','WRKATM',NATOMX,intg=WRKATM)
    WRKATM = 0
    call chmalloc('intcor2.src','MK_ICACTIV','WRKIC',NSTOP,intg=WRKIC)
    WRKIC = 0

    do II = 1,NSLCT
      WRKATM(ISLCT(II)) = 1
      if(QVERBOSE) then
       if(MYNODP.eq.1) then
        write(6,*) 'active atom ',II,' atom is ',ISLCT(II)
       endif
      endif
    enddo
    if(QVERBOSE) then
     if(MYNODP.eq.1) then
      do KK = NSTART,NSTOP
       WRITE(6,*) 'TABLE LINE ',KK, IAR(KK),JAR(KK),KAR(KK),LAR(KK)
      enddo
     endif
    endif
    do KK = NSTART,NSTOP
      I = IAR(KK)
      if(I.gt.0) then
       if(WRKATM(I)==1) then
        WRKIC(KK) = 1
        cycle
       endif
      endif
      J = JAR(KK)
      if(J.gt.0) then
       if(WRKATM(J)==1) then
        WRKIC(KK) = 1
        cycle
       endif
      endif
      K = KAR(KK)
      if(K.gt.0) then
       if(WRKATM(K)==1) then
        WRKIC(KK) = 1
        cycle
       endif
      endif
      L = LAR(KK)
      if(L.gt.0) then
       if(WRKATM(L)==1) then
        WRKIC(KK) = 1
        cycle
       endif
      endif
    enddo
    NICACTV = 0
    do KK = NSTART,NSTOP
      if(WRKIC(KK).eq.1) then
       NICACTV = NICACTV + 1
      endif
    enddo
    if(allocated(ICACTV)) call chmdealloc('intcor2.src','MK_ICACTIV','ICACTV',NICACTV,intg=ICACTV)
    call chmalloc('intcor2.src','MK_ICACTIV','ICACTV',NICACTV,intg=ICACTV)
    NICACTV = 0
    do KK = NSTART,NSTOP
      if(WRKIC(KK).eq.1) then
       NICACTV = NICACTV + 1
       ICACTV(NICACTV)=KK
      endif
    enddo

    call chmdealloc('intcor2.src','MK_ICACTIV','WRKATM',NATOMX,intg=WRKATM)
    call chmdealloc('intcor2.src','MK_ICACTIV','WRKIC',NSTOP,intg=WRKIC)
    if(QVERBOSE) then
     if(MYNODP.eq.1) then
      do KK =1,NICACTV
       WRITE(6,*) KK,' LINE IN IC TABLE ACTIVE ',ICACTV(KK)
      enddo
     endif
    endif
  end subroutine MK_ICACTV
end module intcor2
