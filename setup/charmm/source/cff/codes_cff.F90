#if KEY_CFF==1
SUBROUTINE CODES_CFF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  Written by R. Lapp but adapted from routines tpk, ttpk and phpk2
  !  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
!  use mmffm
  implicit none
  INTEGER IBOND,ITHETA,I,J,I1,I2,J2,K2,L2
  INTEGER ITHT1,ITHAT1,IVTXAT,ITHAT3,JTHETA,JTHAT1,JVTXAT, &
       JTHAT3,ITTAT1,ITTAT2,ITTAT3,ITTE1,ITTE2,ITTE3,ITTE4, &
       KND1,KND2,KND3,KND4,INAME,IT1,IT2,IT3
  LOGICAL IREV

  NUMAT3 = NATOM*3
  NTHTH = 0
  IREV = .FALSE.
  loop200: DO I=1,NTHETA
     I2=IT(I)
     J2=JT(I)
     K2=KT(I)
     !
     !     Find the two bonds in this angle and store them in ITBW
     !
     DO IBOND=1,NBOND
        IF(J2 == IB(IBOND)) THEN
           IF(I2 == JB(IBOND)) ITBW(1,I)=IBOND
           IF(K2 == JB(IBOND)) ITBW(2,I)=IBOND
        ELSEIF(J2 == JB(IBOND)) THEN
           IF(I2 == IB(IBOND)) ITBW(1,I)=IBOND
           IF(K2 == IB(IBOND)) ITBW(2,I)=IBOND
        ENDIF
     ENDDO
     !
     !     If this angle has an automatic parameter search to see if there
     !     is a bond-ordered parameter that should be used instead.
     !
     IF (ICT(I) < 0) THEN
        I2=ITE(IAC(IT(I)))
        J2=ITE(IAC(JT(I)))
        K2=ITE(IAC(KT(I)))
        IT1=BONDTYPE_CFF(ITBW(1,I))
        IT2=BONDTYPE_CFF(ITBW(2,I))
        IREV=.FALSE.
        IF(I2 == K2) THEN
           DO J=NCT+1,NCTO
              IF (I2 == TID1(J).AND.J2 == TID2(J).AND.K2.EQ.TID3(J))THEN
                 IF (IT1 == TORD1(J).AND.IT2 == TORD2(J)) THEN
                    ICT(I)=J
                 ELSEIF (IT1 == TORD2(J).AND.IT2 == TORD1(J)) THEN
                    ICT(I)=J
                    IREV=.TRUE.
                 ENDIF
              ENDIF
           ENDDO
        ELSEIF(I2 < K2) THEN
           DO J=NCT+1,NCTO
              IF (I2 == TID1(J).AND.J2 == TID2(J).AND.K2.EQ.TID3(J))THEN
                 IF (IT1 == TORD1(J).AND.IT2 == TORD2(J))ICT(I)=J
              ENDIF
           ENDDO
        ELSE
           DO J=NCT+1,NCTO
              IF (I2 == TID3(J).AND.J2 == TID2(J).AND.K2.EQ.TID1(J))THEN
                 IF (IT1 == TORD2(J).AND.IT2 == TORD1(J))THEN
                    ICT(I)=J
                    IREV=.TRUE.
                 ENDIF
              ENDIF
           ENDDO
        ENDIF
        IF(ICT(I) < 0) ICT(I) = -ICT(I)
     ENDIF
     !
     ! In order for cross terms to work it is necessary for the atoms to
     ! be listed in the same order as the atom types are listed for the
     ! corresponding parameter in the forcefield.
     !
     I1=ICT(I)
     IF (I1 /= 0 .AND. .NOT.IREV) THEN
        IF ((TID1(I1) > 0).AND.(I2 /= TID1(I1))) IREV=.TRUE.
        IF ((TID3(I1) > 0).AND.(K2 /= TID3(I1))) IREV=.TRUE.
     ENDIF
     IF(IREV) THEN
        I1=IT(I)
        IT(I)=KT(I)
        KT(I)=I1
        I1=ITBW(1,I)
        ITBW(1,I)=ITBW(2,I)
        ITBW(2,I)=I1
     ENDIF
     !
     !  For crossterm calculations the bonds are directional.  The flags in
     !  ITFLG indicate whether the use the bond derivatives as calculated or
     !  to negate them while calculating the cross-term derivatives.
     !  
     I2=IT(I)
     J2=JT(I)
     K2=KT(I)
     IF (I2 == IB(ITBW(1,I))) THEN
        ITFLG(1,I) = 1
     ELSE
        ITFLG(1,I) = -1
     ENDIF
     IF (K2 == IB(ITBW(2,I))) THEN
        ITFLG(2,I) = 1
     ELSE
        ITFLG(2,I) = -1
     ENDIF
     ITHT1 = I + 1
     !
     ITHAT1 = IT(I)
     IVTXAT = JT(I)
     ITHAT3 = KT(I)
     !
     !  Find all angles which share common bonds to set up the angle-angle
     !  cross-terms.
     !
     loop110: DO JTHETA=ITHT1,NTHETA
        JTHAT1 = IT(JTHETA)
        JVTXAT = JT(JTHETA)
        JTHAT3 = KT(JTHETA)
        !
        !-----check if theta*theta interaction - only if the 2 vertex (middle)
        !     atoms of the 2 angles are same is it a theta*theta interaction,
        !     i.e. atoms ivtxat and jvtxat (the middle atom of a theta
        !     interaction is the vertex atom of the angle by definition),
        !
        IF (IVTXAT  /=  JVTXAT) cycle loop200 !GOTO 200
        !
        !-----find which bond is common bond of the 2 theta interactions -
        IF (ITHAT1  ==  JTHAT1) THEN
           !
           !     interaction is ithat3--(ivtxat/jvtxat)--jthat3 and
           !     (ivtxat/jvtxat)--(ithat1/jthat1) is common
           ITTAT1 = ITHAT3
           ITTAT2 = JTHAT3
           ITTAT3 = ITHAT1
           !
        ELSEIF (ITHAT1  ==  JTHAT3) THEN
           !
           !     interaction is ithat3--(ivtxat/jvtxat)--jthat1 and
           !     (ivtxat/jvtxat)--(ithat1/jthat3) is common
           ITTAT1 = ITHAT3
           ITTAT2 = JTHAT1
           ITTAT3 = ITHAT1
           !
        ELSEIF (ITHAT3  ==  JTHAT1) THEN
           !
           !-----interaction is ithat1--(ivtxat/jvtxat)--jthat3 and
           !     (ivtxat/jvtxat)--(ithat3/jthat1) is common
           ITTAT1 = ITHAT1
           ITTAT2 = JTHAT3
           ITTAT3 = ITHAT3
           !
        ELSEIF (ITHAT3  ==  JTHAT3) THEN
           !
           !-----interaction is ithat1--(ivtxat/jvtxat)--jthat1 and
           !     (ivtxat/jvtxat)--(ithat3/jthat3) is common
           ITTAT1 = ITHAT1
           ITTAT2 = JTHAT1
           ITTAT3 = ITHAT3
           !
        ELSE
           !
           !-----if no common bond then no theta*theta interaction
           cycle loop110
        ENDIF
        !
        !-----found a theta*theta interaction - save the 4 atom numbers
        !     involved in ittw. the atom indexes are now in the
        !     form ittat1--ivtxat--ittat2 and ivtxat--ittat3 is the commmon
        !     bond
        NTHTH = NTHTH + 1
        !
        ITTW(1,NTHTH) = ITTAT1
        ITTW(2,NTHTH) = ITTAT2
        ITTW(3,NTHTH) = ITTAT3
        ITTW(4,NTHTH) = IVTXAT
        !
        !-----find the atom types of the 4 atoms involved and pass these
        !     through the theta equivalence array to get actual atom type
        !     of the interaction
        ITTE1 = ITE(IAC(ITTAT1))
        ITTE2 = ITE(IAC(ITTAT2))
        ITTE3 = ITE(IAC(ITTAT3))
        ITTE4 = ITE(IAC(IVTXAT))
        !
        !-----order the atoms involved according to atom type AND position -
        !     see note. search the ttid list for this parameter and
        !     save the parameter pointer in ictt. if the parmater is not found
        !     print an error for this theta*theta interaction.
        !     note. in ttid parameters are stored as
        !     ittat1--(ivtxat--ittat3)--ittat2, hence the change in order here.
        KND1 = MIN0(ITTE1,ITTE2)
        KND2 = ITTE4
        KND3 = ITTE3
        KND4 = MAX0(ITTE1,ITTE2)
        !
        ICTT(NTHTH)=NTTP+1 ! preset to next, reset in loop if match is found
        loopmatch: DO INAME=1,NTTP
           IF (KND1  ==  TTID1(INAME)) THEN
              IF (KND2  ==  TTID2(INAME)) THEN
                 IF (KND3  ==  TTID3(INAME)) THEN
                    IF (KND4  ==  TTID4(INAME)) THEN
                       ICTT(NTHTH)=INAME
                       exit loopmatch
                    ENDIF
                 ENDIF
              ENDIF
           ENDIF
        ENDDO loopmatch

        !     add the theta angle index to the
        !     itttw array in itttw(1,) of theta angle ittat1-ivtxat-ittat3.
        ITTTW(1,NTHTH) = I

        !-----the theta angle index for the 2nd theta angle is stored
        !     in itttw(2,) for theta angle ittat2-ivtxat-ittat3.
        ITTTW(2,NTHTH) = JTHETA
        IF (ITTAT1 == IT(I)) THEN
           ITTFLG(1,NTHTH) = -1
        ELSE
           ITTFLG(1,NTHTH) = 1
        ENDIF
        IF (ITTAT2 == IT(JTHETA)) THEN
           ITTFLG(2,NTHTH) = -1
        ELSE
           ITTFLG(2,NTHTH) = 1
        ENDIF
        !
     ENDDO loop110
  ENDDO loop200
  !
  NBB=0
  DO I=1,NPHI
     J2=IPE(IAC(JP(I)))
     K2=IPE(IAC(KP(I)))
     IF(J2 > K2) THEN
        I1=IP(I)
        IP(I)=LP(I)
        LP(I)=I1
        I1=JP(I)
        JP(I)=KP(I)
        KP(I)=I1
     ENDIF
     I2=IP(I)
     J2=JP(I)
     K2=KP(I)
     L2=LP(I)
     !
     !  Find the three bonds in the torsion and store them in IPBW
     !
     DO IBOND=1,NBOND
        IF(J2 == IB(IBOND)) THEN
           IF(I2 == JB(IBOND)) IPBW(1,I)=IBOND
           IF(K2 == JB(IBOND)) IPBW(2,I)=IBOND
        ELSEIF(J2 == JB(IBOND)) THEN
           IF(I2 == IB(IBOND)) IPBW(1,I)=IBOND
           IF(K2 == IB(IBOND)) IPBW(2,I)=IBOND
        ENDIF
        IF(K2 == IB(IBOND)) THEN
           IF(L2 == JB(IBOND)) IPBW(3,I)=IBOND
        ELSEIF(K2 == JB(IBOND)) THEN
           IF(L2 == IB(IBOND)) IPBW(3,I)=IBOND
        ENDIF
     ENDDO
     I2=IPE(IAC(IP(I)))
     J2=IPE(IAC(JP(I)))
     K2=IPE(IAC(KP(I)))
     L2=IPE(IAC(LP(I)))
     IT1=BONDTYPE_CFF(IPBW(1,I))
     IT2=BONDTYPE_CFF(IPBW(2,I))
     IT3=BONDTYPE_CFF(IPBW(3,I))
     IREV=.FALSE.
     !
     !  If the parameter assigned to this torsion is automatic then check
     !  to see if there is a bond-ordered parameter.
     !
     IF (ICP(I) < 0) THEN
        IF(I2 == J2.AND.I2 == K2.AND.I2.EQ.L2) THEN
           DO J=NCP+1,NCPO
              IF (I2 == PID1(J).AND.J2 == PID2(J).AND. &
                   K2 == PID3(J).AND.L2 == PID4(J))THEN
                 IF (IT1 == PORD1(J).AND.IT2 == PORD2(J).AND. &
                      IT3 == PORD3(J)) THEN
                    ICP(I)=J
                 ELSEIF (IT3 == PORD1(J).AND.IT2 == PORD2(J).AND. &
                      IT1 == PORD3(J)) THEN
                    ICP(I)=J
                    IREV=.TRUE.
                 ENDIF
              ENDIF
           ENDDO
        ELSEIF(J2 == K2) THEN
           IF (I2 <= L2) THEN
              DO J=NCP+1,NCPO
                 IF (I2 == PID1(J).AND.J2 == PID2(J).AND. &
                      K2 == PID3(J).AND.L2 == PID4(J))THEN
                    IF (IT1 == PORD1(J).AND.IT2 == PORD2(J).AND. &
                         IT3 == PORD3(J))ICP(I)=J
                 ENDIF
              ENDDO
           ELSE
              DO J=NCP+1,NCPO
                 IF (I2 == PID4(J).AND.J2 == PID3(J).AND. &
                      K2 == PID2(J).AND.L2 == PID1(J))THEN
                    IF (IT1 == PORD3(J).AND.IT2 == PORD2(J).AND. &
                         IT3 == PORD1(J)) THEN
                       ICP(I)=J
                       IREV=.TRUE.
                    ENDIF
                 ENDIF
              ENDDO
           ENDIF
        ELSEIF(J2 < K2) THEN
           DO J=NCP+1,NCPO
              IF (I2 == PID1(J).AND.J2 == PID2(J).AND. &
                   K2 == PID3(J).AND.L2 == PID4(J))THEN
                 IF (IT1 == PORD1(J).AND.IT2 == PORD2(J).AND. &
                      IT3 == PORD3(J))ICP(I)=J
              ENDIF
           ENDDO
        ELSE
           DO J=NCP+1,NCPO
              IF (I2 == PID4(J).AND.J2 == PID3(J).AND. &
                   K2 == PID2(J).AND.L2 == PID1(J))THEN
                 IF (IT1 == PORD3(J).AND.IT2 == PORD2(J).AND. &
                      IT3 == PORD1(J)) THEN
                    ICP(I)=J
                    IREV=.TRUE.
                 ENDIF
              ENDIF
           ENDDO
        ENDIF
        IF(ICP(I) < 0) ICP(I) = -ICP(I)
     ENDIF
     !
     !  In order for all crossterms to work it is necessary to have the atoms
     !  listed in the same order as the corresponding atom types are
     !  listed for the parameter in the forcefield.  Reverse the atoms in
     !  this torsion if necessary.
     !
     I1=ICP(I)
     IF (I1 /= 0 .AND. .NOT.IREV) THEN
        IF ((PID1(I1) > 0).AND.(I2 /= PID1(I1))) IREV=.TRUE.
        IF ((PID2(I1) > 0).AND.(J2 /= PID2(I1))) IREV=.TRUE.
        IF ((PID3(I1) > 0).AND.(K2 /= PID3(I1))) IREV=.TRUE.
        IF ((PID4(I1) > 0).AND.(L2 /= PID4(I1))) IREV=.TRUE.
     ENDIF
     IF(IREV) THEN
        I1=IP(I)
        IP(I)=LP(I)
        LP(I)=I1
        I1=JP(I)
        JP(I)=KP(I)
        KP(I)=I1
        I1=IPBW(1,I)
        IPBW(1,I)=IPBW(3,I)
        IPBW(3,I)=I1
     ENDIF
     I2=IP(I)
     J2=JP(I)
     K2=KP(I)
     L2=LP(I)
     !
     !  Find the two angles in this torsion and save them in IPTW
     !
     DO ITHETA=1,NTHETA
        IF(J2 == JT(ITHETA)) THEN
           IF(I2 == IT(ITHETA).AND.K2 == KT(ITHETA)) IPTW(1,I)=ITHETA
           IF(K2 == IT(ITHETA).AND.I2 == KT(ITHETA)) IPTW(1,I)=ITHETA
        ENDIF
        IF(K2 == JT(ITHETA)) THEN
           IF(J2 == IT(ITHETA).AND.L2 == KT(ITHETA)) IPTW(2,I)=ITHETA
           IF(L2 == IT(ITHETA).AND.J2 == KT(ITHETA)) IPTW(2,I)=ITHETA
        ENDIF
     ENDDO
     !
     !  The flags in IPHFLG indicate whether to use the angle derivatives
     !  as calculated or whether to negate them while calculating the
     !  derivatives of the cross-terms.
     !
     IF (I2 == IT(IPTW(1,I))) THEN
        IPHFLG(1,I) = -1
     ELSE
        IPHFLG(1,I) = 1
     ENDIF
     IF (J2 == IT(IPTW(2,I))) THEN
        IPHFLG(2,I) = -1
     ELSE
        IPHFLG(2,I) = 1
     ENDIF
     !
     !  If this torsion has a non-zero parameter for the bond-bond-13 crossterm
     !  then save this torsion for the bond-bond energy calculation.
     !
     IF(CBB2(ICP(I)) /= ZERO)THEN
        NBB=NBB+1
        IBBW(NBB)=I
     ENDIF
  ENDDO
  !
  return
end SUBROUTINE CODES_CFF

#endif 

SUBROUTINE NULL_codes_CFF
  RETURN
END SUBROUTINE NULL_codes_CFF

