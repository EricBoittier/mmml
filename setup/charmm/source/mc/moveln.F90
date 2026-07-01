module mcmoveln
#if KEY_MC==1
contains
  SUBROUTINE MOVELN(COMLYN,COMLEN)
    !
    !       Top parsing routine for move linking.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use mc
    use stream
    use string
    use psf
    use exfunc
#if KEY_SAMC==1
    use samc, only: lspmv
#endif

    implicit none
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    !
    !       Local Variables
    !
    INTEGER I, J, IM1, IM2, ICDLEN, IRG, JM1, JM2
    type(chm_iptr) :: ITP(MBONDT)
    INTEGER NTP(MBONDT), IROOTG(MMVTYP)
    CHARACTER(len=4) ML
    
#if KEY_SAMC==1
    IF(LSPMV) THEN
        CALL WRNDIE(-5,'<MOVELN>','SAMC USED, BUT NOT COMPATIBLE WITH MOVE LINKING IN ITS CURRENT FORM.')
    ENDIF
#endif

    !       Get move indices
    IM1 = KW2IND(COMLYN,COMLEN,'LAB1','MVI1',MVLABL,NMVTYP)
    IM2 = KW2IND(COMLYN,COMLEN,'LAB2','MVI2',MVLABL,NMVTYP)

    !       Do some checks
    IF (IM1.EQ.0.OR.IM1.GT.NMVTYP.OR.IM2.GT.NMVTYP) THEN
       CALL WRNDIE (-2, '<MOVELN>', 'MOVE INDEX OUT OF BOUNDS')
       RETURN
    ELSE IF (IM1.EQ.IM2) THEN
       CALL WRNDIE (-2, '<MOVELN>', 'CANNOT LINK TO SELF')
       RETURN
    ELSE IF (IM2.EQ.0 .AND. NXTMVG(IM1).EQ.0) THEN
       CALL WRNDIE (-5, '<MOVELN>', 'MOVE GROUP NOT LINKED')
    ELSE IF (MCMINN(IM1).LT.0) THEN
       !         Do not allow hybrid MC
       CALL WRNDIE (-5, '<MOVELN>', 'LINKING HMC NOT ALLOWED')
       RETURN
    ENDIF

    if (IM2 .gt. 0) then
       IF (NXTMVG(IM2) .GT. 0) THEN
          !         Avoid loops
          CALL WRNDIE (-5, '<MOVELN>', 'MUST ADD TO END OF MOVE CHAIN')
          RETURN
       ELSE IF (NMVATM(IM1) .NE. NMVATM(IM2)) THEN
          !         Currently only index matching is allowed.
          !         It is anticipated that this will change in the future
          CALL WRNDIE (-5, '<MOVELN>', 'MISMATCH IN INSTANCES')
          RETURN
       ELSE IF (MCMINN(IM2) .LT. 0) THEN
          !         Do not allow hybrid MC
          CALL WRNDIE (-5, '<MOVELN>', 'LINKING HMC NOT ALLOWED')
          RETURN
       ENDIF
    end if

    !       Determine the primary groups for the existing links
    DO I = 1, NACMVG
       IRG = IACMVG(I)
       J   = IRG
       IROOTG(J) = IRG
10     IF (NXTMVG(J) .GT. 0) THEN
          J = NXTMVG(J)
          IROOTG(J) = IRG
          GOTO 10
       ENDIF
    ENDDO

#if KEY_GCMC==1
    !       ARD 01-05-21 Grand canonical
    IF (INDXA(COMLYN, COMLEN, 'GCMC') .GT. 0) THEN
       IF (MVTYPE(IM1).EQ.8) THEN
          IGCMVG(IM2)         = IM1
          IGCMVG(IROOTG(IM2)) = IM1
       ELSE IF (MVTYPE(IM2).EQ.8) THEN
          IGCMVG(IM1)         = IM2
          IGCMVG(IROOTG(IM1)) = IM2
       ELSE
          CALL WRNDIE (-5, '<MVCOUP>', 'NO GC MOVE GROUP SELECTED')
       ENDIF
       RETURN
    ENDIF
#endif 

    !       If the first group currently points to another group, restore
    !       the latter group to full status
    IF (NXTMVG(IM1).GT.0) THEN

       IF (NXTMVG(NXTMVG(IM1)) .GT. 0) CALL WRNDIE (-5, '<MOVELN>', &
            'CANNOT BREAK CHAIN IN MIDDLE')

       IF (PRNLEV .GE. 2) THEN
          WRITE (OUTU,'(A,I3,1X,A,I3)') &
               ' MOVELN> Unlinking move group ',NXTMVG(IM1),'from ',IM1
       ENDIF

       NACMVG = NACMVG + 1
       IACMVG(NACMVG) = NXTMVG(IM1)

    ENDIF

    !       Point to second group of moves
    NXTMVG(IM1) = IM2

    IF (IM2 .GT. 0) THEN

       IF (PRNLEV .GE. 2) THEN
          WRITE (OUTU,'(A,I3,1X,A,I3)') &
               ' MOVELN> Linking move group ',IM2,'to ',IM1
       ENDIF

       !         Swap in last to active move group list
       IACMVG(IM2) = IACMVG(NACMVG)
       NACMVG = NACMVG - 1
    ENDIF

    IRG = IROOTG(IM1)
    CALL MOVEL2(IM1,IM2,ILNMVP(IRG), &
         ILNBDP(IRG),IROOTG,NMVATM,IMVNGP,IBLSTP, &
         MBONDT,MMVTYP,QBND,QLNBND,ITP,NTP,NATOM)

    !       If IM2 is zero, MOVEL2 will free ILNMVP and ILNBDP for the
    !       whole chain.  If there is still a chain, it is necessary to
    !       rebuild these.
    IF (IM2 .EQ. 0) THEN
       JM1 = IRG
20     JM2 = NXTMVG(JM1)
       IF (JM2 .GT. 0) THEN
          CALL MOVEL2(JM1,JM2,ILNMVP(IRG), &
               ILNBDP(IRG),IROOTG,NMVATM,IMVNGP,IBLSTP, &
               MBONDT,MMVTYP,QBND,QLNBND,ITP,NTP,NATOM)
          JM1 = JM2
          GOTO 20
       ENDIF
    ENDIF

    IF (PRNLEV .GT. 3) THEN
       IRG = IROOTG(IM1)
       WRITE (OUTU,'(1X,A7,A9,A11,A6)') 'MOVELN>', 'Position', &
            'Move Index', 'Label'
       WRITE (OUTU,'(1X,A7,A9,A11,A6)') '-------', '--------', &
            '----------','-----'
       I = 1
30     WRITE (OUTU,'(1X,A7,I9,I11,A6)') 'MOVELN>', I, IRG, &
            MVLABL(IRG)
       IF (NXTMVG(IRG).GT.0) THEN
          I = I + 1
          IRG = NXTMVG(IRG)
          GOTO 30
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE MOVELN

  SUBROUTINE MOVEL2(IM1,IM2,ILNMVP,ILNBDP, &
       IROOTG,NMVATM,IMVNGP,IBLSTP,MBONDT,MMVTYP, &
       QBND,QLNBND,ITP,NTP,NATOM)
    !
    !       This is the routine that actually does the linking.
    !
    !       Aaron R. Dinner
    !
    use chm_types
    use memory
    use mcmvutil, only: frebls
    implicit none
    !
    !       Passed variables
    !
    INTEGER IM1, IM2, NACMVG, IACMVG
    type(iptr_ptr) :: ILNMVP, ILNBDP
    type(iptr_ptr) :: IMVNGP(:), IBLSTP(:)
    INTEGER NMVATM(*), NATOM
    INTEGER MBONDT, IROOTG(*), NTP(MBONDT), MMVTYP
    type(chm_iptr) :: ITP(MBONDT)
    LOGICAL QBND(MBONDT,MMVTYP), QLNBND(MBONDT,MMVTYP)
    !
    !       Local variables
    !
    integer,allocatable,dimension(:) :: ISP, JSP
    INTEGER I, J, IBONDP, NM
    type(chm_iptr) :: LISTP, TEMPP, BONDP
    type(chm_iptr),pointer,dimension(:) :: MNEWP, BNEWP
    INTEGER IRG, IMNEWP, IBNEWP
    LOGICAL LCHAIN, QBD1, QBD2

    !       Check if already a multiple-group chain
    IRG = IROOTG(IM1)
    LCHAIN = associated(ILNMVP%A) .and. .not. associated(ILNMVP%A, IMVNGP(IRG)%A)

    QBD1 = .FALSE.
    QBD2 = .FALSE.
    DO I = 1, MBONDT
       IF (QLNBND(I,IRG)) QBD1 = .TRUE.
       if (im2 .gt. 0) then
          IF (QBND(I,IM2)) then
             QBD2 = .TRUE.
          end if
       end if
    ENDDO

    IF (IM2 > 0) THEN
       allocate(MNEWP(NMVATM(IRG)))
       call chmalloc('moveln.src','MOVEL2','ISP',NATOM,intg=ISP)
       call chmalloc('moveln.src','MOVEL2','JSP',NATOM,intg=JSP)

       IF (QBD1 .OR. QBD2) allocate(BNEWP(NMVATM(IRG)))

    ENDIF

    DO I = 1, NMVATM(IRG)

       IF (IM2 > 0) THEN
          ISP(1:NATOM) = 0
          JSP(1:NATOM) = 0
          CALL LNMVLS(LISTP, ILNMVP%A(I), &
               IMVNGP(IM2)%A(I), &
               ISP, JSP, NATOM)

          IF (QBD1.AND.QBD2) THEN
             CALL LNBDLS(BONDP, ILNBDP%A(I)%A, &
                  IBLSTP(IM2)%A(I)%A, MBONDT, &
                  QLNBND(:,IRG), QBND(:,IM2), ITP, NTP)
          ELSEIF (QBD1) THEN
             CALL LNBLST(BONDP, ILNBDP%A(I)%A, MBONDT, &
                  QLNBND(1,IRG))
          ELSEIF (QBD2) THEN
             CALL LNBLST(BONDP, IBLSTP(IM2)%A(I)%A, &
                  MBONDT,QBND(1,IM2))
          ENDIF
          MNEWP(I)%A => LISTP%A
          IF (QBD1 .OR. QBD2) BNEWP(I)%A => BONDP%A
       ENDIF

       IF (LCHAIN) THEN
          TEMPP%A => ILNMVP%A(I)%A
          NM = TEMPP%A(1)
          NM = TEMPP%A(NM)
          call chmdealloc('moveln.src','MOVEL2','TEMPP',NM,intgp=TEMPP%A)

          IF (QBD1) THEN
             TEMPP%A => ILNBDP%A(I)%A
             CALL FREBLS(TEMPP, MBONDT, QLNBND(:,IRG))
          ENDIF
       ENDIF

    ENDDO

    IF (LCHAIN) THEN
       deallocate(ILNMVP%A)
       IF (QBD1) deallocate(ILNBDP%A)
    ENDIF

    IF (IM2 > 0) THEN
       !         Assign pointer to new arrays
       ILNMVP%A => MNEWP
       IF (QBD1.OR.QBD2) ILNBDP%A => BNEWP
       DO I = 1, MBONDT
          QLNBND(I,IRG) = QLNBND(I,IRG).OR.QBND(I,IM2)
       ENDDO
       call chmdealloc('moveln.src','MOVEL2','JSP',NATOM,intg=JSP)
       call chmdealloc('moveln.src','MOVEL2','ISP',NATOM,intg=ISP)
    ELSE
       !         Restore pointer to old arrays (this destroys the chain)
       ILNMVP%A => IMVNGP(IRG)%A
       ILNBDP%A => IBLSTP(IRG)%A
       DO I = 1, MBONDT
          QLNBND(I,IRG) = QBND(I,IRG)
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE MOVEL2

  SUBROUTINE LNBDLS(RETP,IBLST1,IBLST2,MBONDT,QBND1,QBND2, &
       TERMP,NBNEW)
    !
    !       Combine two bonded term lists for move linking.
    !
    use chm_types
    use memory
    use mcmvcrot, only: unqbnd
    use mcmvrtrn, only: assibl
    implicit none

    type(chm_iptr) :: RETP
    INTEGER IBLST1(:), IBLST2(:)
    type(chm_iptr) :: TERMP(:)
    INTEGER NBNEW(:), MBONDT
    LOGICAL QBND1(MBONDT), QBND2(MBONDT)
    !
    integer,allocatable,dimension(:) :: IBP1, IBP2
    INTEGER I, J, ICURR1, ICURR2, NB1, NB2
    INTEGER NTOT, NE

    ICURR1 = 0
    ICURR2 = 0
    NTOT = 0
    DO I = 1, MBONDT

       IF (QBND1(I)) THEN
          ICURR1 = ICURR1 + 1
          NB1 = IBLST1(ICURR1) - ICURR1
          call chmalloc('moveln.src','LNBDLS','IBP1',NB1+1,intg=IBP1)
          IBP1(1) = NB1
          DO J = 1, NB1
             ICURR1 = ICURR1 + 1
             IBP1(J+1) = IBLST1(ICURR1)
          ENDDO
       ENDIF

       IF (QBND2(I)) THEN
          ICURR2 = ICURR2 + 1
          NB2 = IBLST2(ICURR2) - ICURR2
          call chmalloc('moveln.src','LNBDLS','IBP2',NB2+1,intg=IBP2)
          IBP2(1) = NB2
          DO J = 1, NB2
             ICURR2 = ICURR2 + 1
             IBP2(J+1) = IBLST2(ICURR2)
          ENDDO
       ENDIF

       CALL UNQBND(TERMP(I), NBNEW(I), IBP1, IBP2)
       NTOT = NTOT + NBNEW(I)

       IF (QBND2(I)) call chmdealloc('moveln.src','LNBDLS','IBP2',NB2+1,intg=IBP2)
       IF (QBND1(I)) call chmdealloc('moveln.src','LNBDLS','IBP1',NB1+1,intg=IBP1)
    ENDDO

    call chmalloc('moveln.src','LNBDLS','RETP',NTOT,intgp=RETP%A)

    NE = 0
    DO I = 1, MBONDT
       IF (QBND1(I) .OR. QBND2(I)) THEN
          NE = NE + 1
          CALL ASSIBL(NE, NBNEW(I), TERMP(I)%A, RETP%A)
          call chmdealloc('moveln.src','LNBDLS','TERMP(I)',NBNEW(I),intgp=TERMP(I)%A)
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE LNBDLS

  SUBROUTINE LNMVLS(LISTP,IMVNG1,IMVNG2,IMVAT1,IMVAT2,NATOM)
    !
    !       Combine two moving atom lists.
    !
    !       This routine differs from IMVADD in that duplications are
    !       eliminated.
    !
    use chm_types
    use mcmvutil
    implicit none
    !
    type(chm_iptr) :: LISTP, IMVNG1, IMVNG2
    INTEGER IMVAT1(:), IMVAT2(:)
    INTEGER NATOM
    !
    INTEGER I, J, MG1, MG2, MG1MG2, IFLAG, N
    type(chm_iptr) :: TEMPP

    if (associated(IMVNG1%A)) CALL TAGATM(IMVAT1,IMVNG1%A,.FALSE.,0)
    if (associated(IMVNG2%A)) CALL TAGATM(IMVAT2,IMVNG2%A,.FALSE.,0)

    MG1 =  0
    MG2 =  0
    DO I = 1, NATOM
       IF (IMVAT1(I).GT.MG1) MG1 = IMVAT1(I)
       IF (IMVAT2(I).GT.MG2) MG2 = IMVAT2(I)
    ENDDO
    MG1 = MG1 + 1
    MG2 = MG2 + 1
    MG1MG2 = MG1*MG2

    DO I = 1, NATOM
       IMVAT1(I) = IMVAT1(I)*MG2 + IMVAT2(I)
    ENDDO

    IFLAG = 0
    DO I = 1, MG1MG2
       N = 0
       DO J = 1, NATOM
          IF (IMVAT1(J) .EQ. I) THEN
             IMVAT2(J) = 1
             N = N + 1
          ELSE
             IMVAT2(J) = 0
          ENDIF
       ENDDO
       IF (N .GT. 0) THEN
          IFLAG = IFLAG + 1
          CALL IMVLST(TEMPP, NATOM, IMVAT2)
          CALL IMVADD(LISTP, IFLAG, TEMPP)
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE LNMVLS

  SUBROUTINE LNBLST(NEWP,IOLD,MBONDT,QBND)
    !
    !       Frees a bonded term list for a single move instance.
    !
    !       Aaron R. Dinner
    !
    use chm_types
    use memory
    implicit none

    type(chm_iptr) :: NEWP
    INTEGER IOLD(:), MBONDT
    LOGICAL QBND(MBONDT)
    !
    INTEGER I, NB
    !
    NB = 0
    DO I = 1, MBONDT
       IF (QBND(I)) THEN
          NB = NB + 1
          NB = IOLD(NB)
       ENDIF
    ENDDO

    call chmalloc('moveln.src','LNBLST','NEWP',NB,intgp=NEWP%A)
    NEWP%A(1:NB) = IOLD(1:NB)

    RETURN
  END SUBROUTINE LNBLST

  INTEGER FUNCTION KW2IND(COMLYN,COMLEN,LABKW,INDKW,MVLABL,NMVTYP)
    !
    !       Get a move index
    !
    use chm_kinds
    use exfunc
    use mcmvutil, only: matchl, padlab
    use string
    implicit none
    CHARACTER(len=*) COMLYN, LABKW*(*), INDKW*(*)
    INTEGER COMLEN, NMVTYP
    CHARACTER(len=4) MVLABL(NMVTYP)
    !
    INTEGER ICDLEN
    CHARACTER(len=4) ML

    CALL GTRMWD(COMLYN,COMLEN,LABKW,4,ML,4,ICDLEN)
    IF (ICDLEN .GT. 0) THEN
       !CCC          CALL PADLAB(KW2IND,ICDLEN)
       CALL PADLAB(ML,ICDLEN)
       KW2IND = MATCHL(NMVTYP,MVLABL,ML)
    ELSE
       KW2IND = GTRMI(COMLYN,COMLEN,INDKW, 0)
    ENDIF
    RETURN
  END FUNCTION KW2IND

#endif 
end module mcmoveln

