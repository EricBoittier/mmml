module mcwl
  use chm_kinds
  implicit none

#if KEY_MC==1
contains

  SUBROUTINE WLINIT(NWLBIN,NRXNCR,NRXSTT,IWLHST,RWLLGF,IUNWL,IWLBN1)
    !
    !       Initializes histogram arrays for Wang-Landau sampling.
    !       ARD and A. Ma 06-06-30
    !
    use number
    !
    integer,dimension(:) :: IWLHST
    real(chm_real),dimension(:) :: RWLLGF
    INTEGER NWLBIN, NRXNCR, NRXSTT(*), IUNWL
    INTEGER IWLBN1
    !
    INTEGER I
    real(chm_real)  R
    LOGICAL QWLINB

    IF (NRXNCR .LE. 0) THEN
       CALL WRNDIE (-5,'<WLINIT>','W-L W/NO COORDINATES')
    ENDIF

    IF (IUNWL .GT. 0) THEN
       !         Read an existing histogram
       CALL RDWL(IUNWL, RWLLGF, IWLHST, NWLBIN)
    ELSE 
       !         Initialize the histogram
       RWLLGF(1:NWLBIN) = ZERO
       IWLHST(1:NWLBIN) = 0
    ENDIF

    !       Call WLUPDT with dummy values to initialize bin
    CALL WLUPDT(QWLINB,1,IWLBN1,RWLLGF,NWLBIN,R,ONE)
    IF (.NOT. QWLINB) THEN
       CALL WRNDIE (-5,'<WLINIT>','Initial structure out of bounds')
    ENDIF

    RETURN
  END SUBROUTINE WLINIT

  SUBROUTINE WLUPDT(QWLINB,IBIN1,IBIN2,RWLLGF,NWLBIN,DEKIN,BETA)
    !
    !       Determines if the new configuration is in bounds and sets up
    !       quantities for the Wang-Landau acceptance criterion.
    !       ARD and A. Ma 06-06-30
    !
    use number
    use rxncom
#if KEY_RXNCOR==1
    use rxenemod,only: ascend  
#endif

    INTEGER IBIN1, IBIN2, NWLBIN
    real(chm_real)  RWLLGF(*), DEKIN, BETA
    LOGICAL QWLINB

    CALL ASCEND
    QWLINB=WLINBF(DELVAL,TREELO,NRXNCR,LODEL,HIDEL)
    IF (QWLINB .AND. (NRXNCR .GT. 0)) THEN
       IBIN2 = IRXNBF(DELVAL,TREELO,NRXNCR,NRXSTT, &
            LODEL,DELDEL,NWLBIN)
       DEKIN = DEKIN + (RWLLGF(IBIN2) - RWLLGF(IBIN1)) / BETA
    ENDIF

    RETURN
  END SUBROUTINE WLUPDT

  SUBROUTINE WLCHK(IUNWL,NRXNCR,NWLBIN,IWLHST,RWLLGF,NRXSTT,FLTNES, &
       RWLADD)
    !
    !       Check if Wang-Landau histogram (IWLHST) is flat
    !       ARD and A. Ma 06-06-30
    !
    use number
    use stream
    !
    INTEGER IUNWL, NRXNCR, NWLBIN, IWLHST(*), NRXSTT(*)
    real(chm_real)  RWLLGF(*), FLTNES, RWLADD
    !
    INTEGER I
    !
    IF (.NOT. FLTHST(IWLHST,NRXNCR,NRXSTT,FLTNES)) RETURN

    RWLADD = HALF*RWLADD
    IF (PRNLEV .GE. 2) WRITE (OUTU,'(A,1X,F20.12)') &
         ' WLCHK> Updating WLINcrement to ', RWLADD

    REWIND(UNIT=IUNWL)
    DO I = 1, NWLBIN
       WRITE (IUNWL,'(I10,F20.12,I10)') I,RWLLGF(I),IWLHST(I)
    ENDDO

    RETURN
  END SUBROUTINE WLCHK

  SUBROUTINE WLADD(IB,IWLHST,RWLLGF,RWLADD)
    !
    !       Add to the histogram for Wang-Landau sampling
    !       ARD and A. Ma 06-06-30
    !
    INTEGER IB, IWLHST(*)
    real(chm_real)  RWLADD, RWLLGF(*)

    RWLLGF(IB) = RWLLGF(IB) + RWLADD
    IWLHST(IB) = IWLHST(IB) + 1

    RETURN
  END SUBROUTINE WLADD

  INTEGER FUNCTION IRXNBF(DELVAL,TREELO,NRXNCR,NRXSTT,LODEL,DELDEL, &
       NWLBIN)
    !       
    !       05-04-14 ARD
    !       Determine histogram bin
    !
    INTEGER NRXNCR, NRXSTT(*), TREELO(*), NWLBIN
    real(chm_real)  DELVAL(5,*), LODEL(*), DELDEL(*)
    !
    INTEGER IR, I, NFACT

    IRXNBF  = 0
    NFACT = 1
    DO IR = 1, NRXNCR

       I = INT((DELVAL(1,TREELO(IR))-LODEL(IR))/DELDEL(IR)) 
       I = MAX(0,MIN(I,NRXSTT(IR)-1))
       IRXNBF = IRXNBF + I*NFACT
       NFACT = NFACT * NRXSTT(IR)

    ENDDO
    IRXNBF = IRXNBF + 1

    RETURN
  END FUNCTION IRXNBF

  LOGICAL FUNCTION WLINBF(DELVAL,TREELO,NRXNCR,LODEL,HIDEL)
    !
    !       Determine if a new configuration is in bounds
    !       Ao Ma 05-04-14 
    !
    INTEGER NRXNCR, IR, TREELO(*)
    real(chm_real)  DELVAL(3,*), LODEL(*), HIDEL(*)

    DO IR = 1, NRXNCR
       IF ((DELVAL(1,TREELO(IR)) .LT. LODEL(IR)) .OR.  &
            (DELVAL(1,TREELO(IR)) .GT. HIDEL(IR))) THEN
          WLINBF = .FALSE.
          RETURN
       ENDIF
    ENDDO

    WLINBF = .TRUE.
    RETURN
  END FUNCTION WLINBF

  LOGICAL FUNCTION FLTHST(HST,NRXNCR,NRXSTT,FLTNES)
    !
    !       Determine if the histogram accumulated so far is flat
    !       Ao Ma 05-04-14 
    !
    use number
    !
    INTEGER NRXNCR, NRXSTT(*), HST(*)
    real(chm_real)  FLTNES 
    !
    INTEGER IR, NWLBIN, IBIN
    real(chm_real)  AVG, VAR

    NWLBIN = 1
    DO IR = 1, NRXNCR
       NWLBIN = NWLBIN*NRXSTT(IR)
    ENDDO

    AVG = ZERO
    VAR = ZERO
    DO IBIN = 1, NWLBIN
       AVG = AVG + REAL(HST(IBIN))
       VAR = VAR + REAL(HST(IBIN))**2
    ENDDO
    AVG = AVG/REAL(NWLBIN)
    VAR = VAR/REAL(NWLBIN)
    VAR = SQRT((VAR - AVG*AVG))/AVG

    FLTHST = (VAR .LT. FLTNES) 

    RETURN
  END FUNCTION FLTHST

  SUBROUTINE RDWL(IUNIT,RWLLGF,IWLHST,NWLBIN)
    !
    !       Read the existing weights for the Wang-Landau method
    !
    !       The file format is:
    !
    !           CHARMM title
    !           Index   Fbin     Hst_bin
    !           i       F_i      H_i
    !                    .       .
    !                    .       .
    !                    .       .
    !           Nbin    F_Nbin   H_Nbin
    !
    !       ARD and Ao Ma 06-06-30
    !
    use ctitla
    use stream
    !
    !       Passed Variables
    !
    INTEGER NWLBIN, IWLHST(*)
    REAL(chm_real)  RWLLGF(*)
    !
    INTEGER IUNIT, I, J
    LOGICAL ERROR

    IF (IUNIT .EQ. -1) CALL WRNDIE (-5, '<RDWL>', 'INVALID UNIT')

    CALL RDTITL(TITLEB,NTITLB,IUNIT,0)

    I = 1
100 READ (IUNIT,*,END=200) J, RWLLGF(I), IWLHST(I)
    I = I + 1 
    GOTO 100
200 CALL VCLOSE(IUNIT,'READ',ERROR)
    IF (I-1 .NE. NWLBIN) &
         CALL WRNDIE (-5, '<RDWL>', 'READ BINS NE NWLBIN')

    RETURN
  END SUBROUTINE RDWL
#endif 

end module mcwl

