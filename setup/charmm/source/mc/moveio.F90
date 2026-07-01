module mcmoveio
#if KEY_MC==1
contains

  SUBROUTINE MOVEWR(COMLYN,COMLEN)
    !
    !       Writes out the complete MC move set
    !       The file must be opened for formatted write beforehand.
    !       Formatted is chosen to allow manual modification.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use ctitla
    use exfunc
    use mc
    use stream
    use string
    use version

    implicit none
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN

    integer,pointer,dimension(:) :: TEMPP
    INTEGER IUNIT, IM, I, J, K, NB, NM
    INTEGER NPIV
    LOGICAL ERROR
    !
    TEMPP => null()
    IUNIT = GTRMI(COMLYN,COMLEN,'UNIT', -1)
    IF (IUNIT .EQ. -1) &
         CALL WRNDIE (0, '<MOVEWR>', 'INVALID UNIT NUMBER')

    CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
    CALL WRTITL(TITLEA,NTITLA,IUNIT,0)

    !       ARD 99-11-10 Version dependence added
    WRITE (IUNIT, '(I4)') VERNUM
    WRITE (IUNIT, '(I4)') NMVTYP
    DO IM = 1, NMVTYP

       !         Figure out the number of pivot atoms
       NPIV = NPIVF(MVTYPE(IM))

       WRITE (IUNIT, '(A8,I5)') 'MVINDEX ', IM
       WRITE (IUNIT, '(A5,A8)') 'LABEL', MVLABL(IM)
       WRITE (IUNIT, '(2I4,999L3)') MVTYPE(IM), NPIV, &
            (QBND(I,IM),I=1,MBONDT)
       WRITE (IUNIT, '(I10,F22.16,I4)') NMVATM(IM), WEIGHT(IM), &
            NLIMIT(IM)
       WRITE (IUNIT, '(2L3,5F22.16)') ARMLIM(IM), ANISO(IM), &
            ARMMAX(IM), ARMP(IM),  ARMA(IM),  ARMB(IM), DOMCF(IM)

       !         ARD 99-11-10 Version dependence added
       WRITE (IUNIT, '(F22.16)') TFACT(IM)
       WRITE (IUNIT, '(2I10,4F22.16)') MCMINN(IM),MCMTYP(IM), &
            RMCSTP(IM),RMCMNF(IM),RMCMNG(IM),RMCMNS(IM)

       !         If it is a volume move, save the mode held in ipivtp
       IF (MVTYPE(IM) == 7) &
            WRITE (IUNIT, '(I5)') IPIVTP(IM)%A(1)%A(1)

       WRITE (IUNIT, '(A)') 'INDEX NBONDS NMOVING PIVOTS'
       NM = 0
       DO I = 1, NMVATM(IM)

          !           Get the number of bonded terms
          NB = 0
          IF (associated(IBLSTP(IM)%A)) THEN
             TEMPP => IBLSTP(IM)%A(I)%A
             DO J = 1, MBONDT
                IF (QBND(J,IM)) THEN
                   NB = NB + 1
                   NB = TEMPP(NB)
                ENDIF
             ENDDO
          ENDIF

          !           Get number of moving atoms
          TEMPP => IMVNGP(IM)%A(I)%A
          NM = TEMPP(1)
          NM = TEMPP(NM)

          IF (NPIV .EQ. 0) THEN
             WRITE (IUNIT,'(I9,999I5)') I, NB, NM
          ELSE
             TEMPP => IPIVTP(IM)%A(I)%A
             WRITE (IUNIT,'(I9,999I5)') I, NB, NM, &
                  (TEMPP(J), J = 1, NPIV)
          ENDIF
       ENDDO

       !         Bonded terms
       IF (associated(IBLSTP(IM)%A)) THEN
          WRITE (IUNIT, '(A)') 'BOND TERMS'
          DO I = 1, NMVATM(IM)
             TEMPP => IBLSTP(IM)%A(I)%A
             NB = 0
             DO J = 1, MBONDT
                IF (QBND(J,IM)) THEN
                   NB = NB + 1
                   NB = TEMPP(NB)
                ENDIF
             ENDDO
             WRITE (IUNIT,'(I9,999I6)') I, &
                  (TEMPP(J), J=1,NB)
          ENDDO
       ENDIF

       !         Moving atoms
       WRITE (IUNIT, '(A)') 'MOVING ATOMS'
       DO I = 1, NMVATM(IM)
          TEMPP => IMVNGP(IM)%A(I)%A
          NM = TEMPP(1)
          NM = TEMPP(NM)
          WRITE (IUNIT,'(I9,999I6)') I, &
               (TEMPP(J), J=1,NM)
       ENDDO

       !         limits
       WRITE (IUNIT, '(A)') 'DMAX'
       IF (ANISO(IM)) THEN
          DO I = 1, NMVATM(IM)
             J = (I - 1)*9
             WRITE (IUNIT,'(I9,9F22.16)') I, (MDXP(IM)%A(J+K), K = 1, 9)
          ENDDO
       ELSE
          DO I = 1, NMVATM(IM)
             WRITE (IUNIT,'(I9,F22.16)') I, MDXP(IM)%A(I)
          ENDDO
       ENDIF

    ENDDO

    !       Linking
    WRITE (IUNIT, '(A)') 'LINKING'
    WRITE (IUNIT,'(999I5)') NACMVG, (IACMVG(I), I = 1, NACMVG)
    WRITE (IUNIT,'(999I5)') (NXTMVG(I), I = 1, NMVTYP)

    CALL VCLOSE(IUNIT,'KEEP',ERROR)

    RETURN
  END SUBROUTINE MOVEWR

  INTEGER FUNCTION NPIVF(IM)
    !
    !       Returns the number of pivot atoms as a function of move type.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none

    INTEGER IM

    IF (IM .EQ. 1) THEN
       NPIVF = 0
    ELSE IF (IM .EQ. 2) THEN
       NPIVF = 1
    ELSE IF (IM .EQ. 3) THEN
       NPIVF = 0
    ELSE IF (IM .EQ. 4) THEN
       NPIVF = 2
    ELSE IF (IM .EQ. 5) THEN
       NPIVF = 14
    ELSE IF (IM .EQ. 6) THEN
       NPIVF = 0
    ELSE IF (IM .EQ. 7) THEN
       NPIVF = 0
    ELSE IF (IM .EQ. 8) THEN
       NPIVF = 0
    ELSE
       CALL WRNDIE(-5,'<NPIVF>', &
            'Internal Error --- UNKNOWN MOVE TYPE')
    ENDIF
    RETURN
  END FUNCTION NPIVF

  SUBROUTINE MOVERD(COMLYN,COMLEN)
    !
    !       Reads in the complete MC move set
    !       The file must be opened for formatted read beforehand.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use dimens_fcm
    use ctitla
    use exfunc
    use memory
    use mc
    use mcmoveln, only: movel2
    use mcmvutil, only: padlab
    use number
    use psf
    use string

    implicit none
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    !
    !       Local Variables
    !
    INTEGER IUNIT, JMV, IMV, I, J, K, NB, NM, ILEN
    integer,pointer,dimension(:) :: TEMPP
    INTEGER NPIV, IPIV, IVERWR, IADD
    LOGICAL QBONDS, APPND
    CHARACTER(len=8) STEMP
    !       Move linking
    type(chm_iptr) :: ITP(MBONDT)
    INTEGER JM1, JM2, IROOTG(MMVTYP), NTP(MBONDT)
    INTEGER NACTMP, IADDAC, IRG
    !
    TEMPP => null()
    IUNIT = GTRMI(COMLYN,COMLEN,'UNIT', -1)
    IF (IUNIT .EQ. -1) &
         CALL WRNDIE (0, '<MOVERD>', 'INVALID UNIT NUMBER')

    CALL RDTITL(TITLEB,NTITLB,IUNIT,0)

    !       Check whether we should append the current moves or replace them
    APPND = GTRMI(COMLYN,COMLEN,'APPE', 1) .NE. 0
    IF (APPND) THEN
       IMV = NMVTYP
       IADD = NMVTYP
       IADDAC = NACMVG
    ELSE
       DO JMV = 1, NMVTYP
          CALL FREMVS(IMVNGP(JMV),IBLSTP(JMV),IPIVTP(JMV),ILNMVP(JMV), &
               ILNBDP(JMV),MDXP(JMV),NMVATM(JMV),MVTYPE(JMV), &
               MBONDT, QBND(:,JMV), QLNBND(:,JMV), ANISO(JMV))
       ENDDO
       IMV = 0
       IADD = 0
       IADDAC = 0
       NACMVG = 0
    ENDIF

    !       ARD 99-11-10 Version dependence added
    READ (IUNIT, *) IVERWR
    READ (IUNIT, *) NMVTYP

    DO JMV = 1, NMVTYP
       IMV = IMV + 1
       READ (IUNIT, '(A)')
       READ (IUNIT, '(A5,4X,A4)') STEMP, MVLABL(IMV)
       ILEN = LEN(MVLABL(IMV))
       CALL PADLAB(MVLABL(IMV),ILEN)
       READ (IUNIT, *) MVTYPE(IMV), NPIV, (QBND(I,IMV),I=1,MBONDT)
       READ (IUNIT, *) NMVATM(IMV), WEIGHT(IMV), NLIMIT(IMV)
       READ (IUNIT, *) ARMLIM(IMV), ANISO(IMV), &
            ARMMAX(IMV), ARMP(IMV),  ARMA(IMV),  ARMB(IMV), DOMCF(IMV)

       !         ARD 99-11-10 Version dependence added
       IF (IVERWR .GE. 28) THEN
          READ (IUNIT, *) TFACT(IMV)
          READ (IUNIT, *) MCMINN(IMV),MCMTYP(IMV), &
               RMCSTP(IMV),RMCMNF(IMV),RMCMNG(IMV),RMCMNS(IMV)
       ELSE
          TFACT(IMV)  = ONE
          MCMINN(IMV) = 0
       ENDIF

       !         If it is a volume move, read the mode held in ipivtp
       IF (MVTYPE(IMV) == 7) THEN
          allocate(IPIVTP(IMV)%A(1))
          call chmalloc('moveio.src','MOVERD','IPIVTP(IMV)',1,intgp=IPIVTP(IMV)%A(1)%A)
          READ (IUNIT, *) IPIVTP(IMV)%A(1)%A(1)
       ENDIF

       allocate(IMVNGP(IMV)%A(NMVATM(IMV)))
       IF (NPIV > 0) allocate(IPIVTP(IMV)%A(NMVATM(IMV)))
       QBONDS = .FALSE.
       DO I = 1, MBONDT
          IF (QBND(I,IMV)) QBONDS = .TRUE.
       ENDDO
       IF (QBONDS) THEN
          allocate(IBLSTP(IMV)%A(NMVATM(IMV)))
       ELSE
          IBLSTP(IMV)%A => null()
       ENDIF
       IF (ANISO(IMV)) THEN
          call chmalloc('moveio.src','MOVERD','MDXP(IMV)',9*NMVATM(IMV),crlp=MDXP(IMV)%A)
       ELSE
          call chmalloc('moveio.src','MOVERD','MDXP(IMV)',NMVATM(IMV),crlp=MDXP(IMV)%A)
       ENDIF

       READ (IUNIT, '(A)')
       DO I = 1, NMVATM(IMV)

          IF (NPIV .EQ. 0) THEN
             READ (IUNIT,*) J, NB, NM
          ELSE
             call chmalloc('moveio.src','MOVERD','TEMPP',NPIV,intgp=TEMPP)
             IF (NPIV == 1) THEN
                READ (IUNIT,*) J, NB, NM, TEMPP(1)
             ELSE
                CALL RDPIV(TEMPP,IUNIT,NB,NM,NPIV)
             ENDIF
             IPIVTP(IMV)%A(I)%A => TEMPP
          ENDIF

          IF (QBONDS) THEN
             call chmalloc('moveio.src','MOVERD','TEMPP',NB,intgp=TEMPP)
             TEMPP(1) = NB
             IBLSTP(IMV)%A(I)%A => TEMPP
          ENDIF
          call chmalloc('moveio.src','MOVERD','TEMPP',NM,intgp=TEMPP)
          TEMPP(1) = NM
          IMVNGP(IMV)%A(I)%A => TEMPP

       ENDDO

       !         Bonded terms
       IF (QBONDS) THEN
          READ (IUNIT, '(A)')
          DO I = 1, NMVATM(IMV)
             TEMPP => IBLSTP(IMV)%A(I)%A
             NB = TEMPP(1)
             CALL RDOTH(TEMPP, IUNIT, NB)
          ENDDO
       ENDIF

       !         Moving atoms
       READ (IUNIT, '(A)')
       DO I = 1, NMVATM(IMV)
          TEMPP => IMVNGP(IMV)%A(I)%A
          NM = TEMPP(1)
          CALL RDOTH(TEMPP, IUNIT, NM)
       ENDDO

       !         limits
       READ (IUNIT, '(A)')
       IF (ANISO(IMV)) THEN
          DO I = 1, NMVATM(IMV)
             J = (I - 1)*9
             CALL RDRMDX(MDXP(IMV)%A, IUNIT, J+1, J+9)
          ENDDO
       ELSE
          DO I = 1, NMVATM(IMV)
             CALL RDRMDX(MDXP(IMV)%A, IUNIT, I, I)
          ENDDO
       ENDIF

       !         Set up arrays for move linking.
       !         Each move group starts active and unlinked.
       NXTMVG(IMV) = 0
       ILNMVP(IMV) = IMVNGP(IMV)
       ILNBDP(IMV) = IBLSTP(IMV)
       DO I = 1, MBONDT
          QLNBND(I,IMV) = QBND(I,IMV)
       ENDDO

    ENDDO

    NMVTYP = IMV

    !       Linking
    IF (IVERWR .LT. 29) RETURN

    READ (IUNIT, '(A)')
    READ (IUNIT,*) NACTMP, (IACMVG(I+IADDAC), I = 1, NACTMP)
    READ (IUNIT,*) (NXTMVG(I), I = IADD+1, NMVTYP)

    DO I = IADD+1, NMVTYP
       IF (NXTMVG(I) .GT. 0) NXTMVG(I) = NXTMVG(I) + IADD
    ENDDO
    DO I = 1, NACTMP
       IACMVG(NACMVG+I) = IACMVG(NACMVG+I) + IADD
    ENDDO
    NACMVG = NACMVG + NACTMP

    !       Build the moving and bond term lists for the new linked moves

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

    !       Determine the primary groups for the existing links
    DO I = IADDAC+1, NACMVG
       JM1 = IACMVG(I)
20     JM2 = NXTMVG(JM1)
       IF (JM2 .GT. 0) THEN
          CALL MOVEL2(JM1,JM2,ILNMVP(IACMVG(I)),ILNBDP(IACMVG(I)), &
               IROOTG,NMVATM,IMVNGP,IBLSTP,MBONDT,MMVTYP,QBND, &
               QLNBND,ITP,NTP,NATOM)
          JM1 = JM2
          GOTO 20
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE MOVERD

  SUBROUTINE FREMVS(IMVNGP,IBLSTP,IPIVTP,ILNMVP,ILNBDP,MDXP, &
       NMVATM,MVTYPE,MBONDT,QBND,QLNBND,ANISO)
    !
    !       Frees all the heap arrays associated with a move
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use memory
    use mcmvutil, only: frebls

    implicit none

    type(iptr_ptr) :: IMVNGP, IBLSTP, ILNMVP, ILNBDP, IPIVTP
    type(chm_ptr) :: MDXP
    INTEGER NMVATM, MVTYPE, MBONDT
    LOGICAL QBND(MBONDT), QLNBND(MBONDT), ANISO

    type(chm_iptr) :: TEMPP
    INTEGER NPIV, I, J, NM, NB

    IF (associated(ILNMVP%A) .and. .not. associated(ILNMVP%A, IMVNGP%A)) THEN
       DO I = 1, NMVATM
          TEMPP%A => ILNMVP%A(I)%A
          IF (associated(TEMPP%A)) THEN
             NM = TEMPP%A(1)
             NM = TEMPP%A(NM)
             call chmdealloc('moveio.src','FREMVS','TEMPP',NM,intgp=TEMPP%A)
          ENDIF
       ENDDO
       deallocate(ILNMVP%A)
    ENDIF

    IF (associated(IMVNGP%A)) THEN
       DO I = 1, NMVATM
          TEMPP%A => IMVNGP%A(I)%A
          IF (associated(TEMPP%A)) THEN
             NM = TEMPP%A(1)
             NM = TEMPP%A(NM)
             call chmdealloc('moveio.src','FREMVS','TEMPP',NM,intgp=TEMPP%A)
          ENDIF
       ENDDO
       deallocate(IMVNGP%A)
    ENDIF

    IF (associated(ILNBDP%A) .and. .not. associated(ILNBDP%A, IBLSTP%A)) THEN
       DO I = 1, NMVATM
          TEMPP%A => ILNBDP%A(I)%A
          CALL FREBLS(TEMPP, MBONDT, QLNBND)
       ENDDO
       deallocate(ILNBDP%A)
    ENDIF

    IF (associated(IBLSTP%A)) THEN
       DO I = 1, NMVATM
          TEMPP%A => IBLSTP%A(I)%A
          CALL FREBLS(TEMPP, MBONDT, QBND)
       ENDDO
       deallocate(IBLSTP%A)
    ENDIF

    IF (associated(IPIVTP%A)) THEN
       IF (MVTYPE == 7) THEN
          call chmdealloc('moveio.src','FREMVS','IPIVTP',1,intgp=IPIVTP%A(1)%A)
       ELSE
          NPIV = NPIVF(MVTYPE)
          IF (NPIV > 0) THEN
             DO I = 1, NMVATM
                TEMPP%A => IPIVTP%A(I)%A
                call chmdealloc('moveio.src','FREMVS','TEMPP',NPIV,intgp=TEMPP%A)
             ENDDO
          ENDIF
       ENDIF
       deallocate(IPIVTP%A)
    ENDIF

    IF (ANISO) THEN
       call chmdealloc('moveio.src','FREMVS','MDXP',NMVATM*9,crlp=MDXP%A)
    ELSE
       call chmdealloc('moveio.src','FREMVS','MDXP',NMVATM,crlp=MDXP%A)
    ENDIF
    MDXP%A => null()

    RETURN
  END SUBROUTINE FREMVS

  SUBROUTINE RDPIV(IPIVOT,IUNIT,NB,NM,NPIV)
    !
    !       Reads a line into the pivot array
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IPIVOT(*), IUNIT, NB, NM, NPIV
    INTEGER I, J
    READ (IUNIT,*) I, NB, NM, (IPIVOT(J), J = 1, NPIV)
    RETURN
  END SUBROUTINE RDPIV

  SUBROUTINE RDOTH(IOTH,IUNIT,N)
    !
    !       Reads the bond and moving lists
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IOTH(*), IUNIT, N
    INTEGER I, J
    READ (IUNIT,*) I, (IOTH(J), J = 1, N)
    RETURN
  END SUBROUTINE RDOTH

  SUBROUTINE RDRMDX(RMDX,IUNIT,ISTART,IEND)
    !
    !       Reads a line into the limits array
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IUNIT, ISTART,IEND
    INTEGER I, J
    real(chm_real)  RMDX(:)
    READ (IUNIT,*) I, (RMDX(J), J = ISTART, IEND)
    RETURN
  END SUBROUTINE RDRMDX

#endif 
end module mcmoveio

