module mcio
#if KEY_MC==1
contains

  SUBROUTINE MCWRCD(IOAVES,IUNWRI,IUNCRD,IOENGY,NSAVC,ISVFRQ, &
       TITLEA,NTITLA,IFRATP,NFREAT,ISTEP,NSTEPS, &
       TKELV,ETOT,ISDMC,X,Y,Z &
#if KEY_GCMC==1
       ,GCMCON     & 
#endif
       )
    !
    !       Writes out coordinates to a trajectory file.
    !
    !       Aaron R. Dinner
    !

    use cheq,only:qcg
    use chm_kinds
    use dimens_fcm
    use consta
    use number
    use dynio
    use psf
    use memory
#if KEY_CHEQ==1
    use stream          
#endif
#if KEY_DHDGB==1
    use dhdgb,only:totals
#endif

    implicit none
    !
    !       Passed Variables
    !
    INTEGER IOAVES,IUNWRI,IUNCRD,IOENGY,NSAVC,ISVFRQ
    integer :: IFRATP(:)
    INTEGER NFREAT, ISTEP, NSTEPS, ISDMC
    INTEGER NTITLA
    CHARACTER(len=80) :: TITLEA(:)
    real(chm_real) :: TKELV, ETOT, X(:), Y(:), Z(:)
#if KEY_GCMC==1
    LOGICAL GCMCON(:)     
#endif
    !
    !     Local Variables
    !
    INTEGER IFILE, NFILE, I
    real(chm_real),allocatable,dimension(:) :: IDP
#if KEY_GCMC==1
    real(chm_real),allocatable,dimension(:) :: IXT, IYT, IZT 
#endif
    real(chm_real)  DTAMC, RS
#if KEY_CHEQ==1
    real(chm_real),allocatable,dimension(:) :: ICGOLD, IVCG, ICGTMP
    LOGICAL   QCHEQRDPRM
#endif 
#if KEY_DHDGB==1
    real(chm_real) DUM_FHDGB(TOTALS)
#endif
    !
    !     Time is stored such that 1000 steps appear as 1 ps
    !
    IF (NSAVC .GT. 0) THEN
       IF (MOD(ISTEP,NSAVC) .EQ. 0) THEN
          !           Large steps numbers break QUANTA
          !           Normalize but get info back in DTAMC
          IFILE = ISTEP/NSAVC
          NFILE = NSTEPS/NSAVC
          DTAMC = FLOAT(NSAVC)*0.001/TIMFAC
#if KEY_GCMC==1
          !           Allocate space for an array to save existing atom positions
          !           in order to preserve internal structure of "off" molecules.
          call chmalloc('mcio.src','MCWRCD','IXT',NATOM,crl=IXT)
          call chmalloc('mcio.src','MCWRCD','IYT',NATOM,crl=IYT)
          call chmalloc('mcio.src','MCWRCD','IZT',NATOM,crl=IZT)
          !           Assign "off" atoms in GCMC a dummy value.
          DO I = 1, NATOM
             IF (.NOT. GCMCON(I)) THEN
                IXT(I) = X(I)
                IYT(I) = Y(I)
                IZT(I) = Z(I)
                X(I) = ANUM
                Y(I) = ANUM
                Z(I) = ANUM
             ENDIF
          ENDDO
#endif 
          CALL WRITMC(X,Y,Z,NATOM, IFRATP, NFREAT,1,IFILE, &
               0,DTAMC,1,NFILE,TITLEA,NTITLA,IUNCRD)
#if KEY_GCMC==1
          !           Restore the coordinates of "off" molecules
          DO I = 1, NATOM
             IF (.NOT. GCMCON(I)) THEN
                X(I) = IXT(I)
                Y(I) = IYT(I)
                Z(I) = IZT(I)
             ENDIF
          ENDDO
          call chmdealloc('mcio.src','MCWRCD','IZT',NATOM,crl=IZT)
          call chmdealloc('mcio.src','MCWRCD','IYT',NATOM,crl=IYT)
          call chmdealloc('mcio.src','MCWRCD','IXT',NATOM,crl=IXT)
#endif 
       ENDIF
    ENDIF

    IF (ISVFRQ .GT. 0) THEN
       IF (MOD(ISTEP,ISVFRQ) .EQ. 0) THEN
          !           Dummy array
          call chmalloc('mcio.src','MCWRCD','IDP',NATOM,crl=IDP)
#if KEY_CHEQ==1
          call chmalloc('mcio.src','MCWRCD','ICGTMP',NATOM,crl=ICGTMP)
          call chmalloc('mcio.src','MCWRCD','ICGOLD',NATOM,crl=ICGOLD)
          call chmalloc('mcio.src','MCWRCD','IVCG',NATOM,crl=IVCG)
          IF(QCG)THEN
             IF (PRNLEV.GT.1) WRITE(OUTU,'(3A)') &
                  '<MCIO> CHEQ is turned on but charge information', &
                  '/      will not be written to trajectory.', &
                  '/      CHEQ not yet enabled for MC.'
          ENDIF
#endif 
          IDP(1:NATOM) = ZERO
          RS = FLOAT(ISDMC)
          CALL WRIDYN(IUNWRI,NATOM,X,Y,Z, &
               IDP, IDP, IDP, &
               IDP, IDP, IDP, &
#if KEY_CHEQ==1
               CG, ICGOLD, IVCG, .FALSE.,         & 
#endif
               ! PJ 06/2005
#if KEY_PIPF==1
               uind0, uind0, uind0, .FALSE.,      & 
#endif
#if KEY_PIPF==1
               0, (/ ZERO /), (/ ZERO /),         & 
#endif
#if KEY_DYNVV2==1
               .FALSE., IDP, IDP, IDP,            & 
#endif
               ISTEP,ISTEP,3*NATOM,NSTEPS,NSAVC,0,RS,TKELV,0,0 &
#if KEY_BLOCK==1 /*ldm*/
               , .FALSE., .FALSE. &
               , 0, (/ ZERO /), (/ ZERO /), (/ ZERO /), 0 &
#endif 
#if KEY_FOURD==1
               , (/ ZERO /), (/ ZERO /)            & 
#endif
#if KEY_SCCDFTB==1
               ,.FALSE.,.FALSE.,0,0,ZERO,ZERO,ZERO & 
#endif
#if KEY_DHDGB==1
!AP/MF
        ,.FALSE.,TOTALS,DUM_FHDGB,DUM_FHDGB,DUM_FHDGB &
#endif
               )
#if KEY_CHEQ==1
          call chmdealloc('mcio.src','MCWRCD','IVCG',NATOM,crl=IVCG)
          call chmdealloc('mcio.src','MCWRCD','ICGOLD',NATOM,crl=ICGOLD)
          call chmdealloc('mcio.src','MCWRCD','ICGTMP',NATOM,crl=ICGTMP)
#endif 
          call chmdealloc('mcio.src','MCWRCD','IDP',NATOM,crl=IDP)
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE MCWRCD

  SUBROUTINE MCRDCD(IUNREA,ISDMC,X,Y,Z)
    !
    !       Reads in coordinates from a restart file.
    !
    !       Aaron R. Dinner
    !

#if KEY_CHEQ==1
    use cheq,only: qcg          
#endif

    use chm_kinds
    use dimens_fcm
    use exfunc
    use number
    use stream
    use dynio
    use psf
    use memory
#if KEY_DHDGB==1
    use dhdgb,only:totals
#endif
    implicit none
    !
    INTEGER IUNREA, ISDMC
    real(chm_real)  X(*), Y(*), Z(*)
    !
    INTEGER I
    real(chm_real),allocatable,dimension(:) :: IDP
    real(chm_real)  R, RS
#if KEY_CHEQ==1
    real(chm_real),allocatable,dimension(:) :: ICG, ICGOLD, IVCG
#endif 
#if KEY_DHDGB==1
    real(chm_real) DUM_FHDGB1(TOTALS)
    real(chm_real) DUM_FHDGB2(TOTALS)
    real(chm_real) DUM_FHDGB3(TOTALS)
    LOGICAL Q_DUM
#endif
    !
    !       Dummy array
    call chmalloc('mcio.src','MCRDCD','IDP',NATOM,crl=IDP)
#if KEY_CHEQ==1
    !mfc   --- Temporary arrays to accomodate CHEQ data in a trajectory
    !mfc   --- Discarded after read, until needed in future development
    call chmalloc('mcio.src','MCRDCD','ICG',NATOM,crl=ICG)
    call chmalloc('mcio.src','MCRDCD','ICGOLD',NATOM,crl=ICGOLD)
    call chmalloc('mcio.src','MCRDCD','IVCG',NATOM,crl=IVCG)
    write(outu,*) "--------- QCG ",qcg
#endif 
    CALL READYN(IUNREA,NATOM,X,Y,Z, &
         IDP, IDP, IDP, &
         IDP, IDP, IDP, &
#if KEY_CHEQ==1
         ICG, ICGOLD, IVCG, QCG,                    & 
#endif
#if KEY_PIPF==1
         uind0, uind0, uind0, .FALSE.,              & 
#endif
#if KEY_PIPF==1
         0, (/ ZERO /), (/ ZERO /),                 & 
#endif
#if KEY_DYNVV2==1
         .FALSE., IDP, IDP, IDP,                    & 
#endif
         I,I,I,I,I,I,RS,R,I,I &
#if KEY_BLOCK==1 /*ldm*/
         , .FALSE., .FALSE. &
         , 0, (/ ZERO /),  (/ ZERO /),  (/ ZERO /), I &
#endif 
#if KEY_FOURD==1
         , (/ ZERO /), (/ ZERO /)                    & 
#endif
#if KEY_SCCDFTB==1
         ,.FALSE.,.FALSE.,.FALSE.,0,0,ZERO,ZERO,ZERO & 
#endif
#if KEY_DHDGB==1
          ,Q_DUM,TOTALS,DUM_FHDGB1,DUM_FHDGB2,DUM_FHDGB3 &
#endif
         )

    ISDMC = INT(RS)
#if KEY_CHEQ==1
    call chmdealloc('mcio.src','MCRDCD','IVCG',NATOM,crl=IVCG)
    call chmdealloc('mcio.src','MCRDCD','ICGOLD',NATOM,crl=ICGOLD)
    call chmdealloc('mcio.src','MCRDCD','ICG',NATOM,crl=ICG)
#endif 
    call chmdealloc('mcio.src','MCRDCD','IDP',NATOM,crl=IDP)
    RETURN
  END SUBROUTINE MCRDCD

  SUBROUTINE WRITMC(X,Y,Z,NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF, &
       DELTA,NSAVC,NSTEPS,TITLE,NTITLA,IUNCRD)
    !
    !       Writes a set of coordinates for a single mc step. The format
    !       for the trajectory file varies with whether any atoms are fixed.
    !       ICNTRL(9) stores the number of fixed atoms which will be zero for
    !       all previous trajectory files so compatibility is assured.
    !
    !       Adapted from WRITEC by Sung-Sau So
    !       Modified by Aaron R. Dinner February 1996
    !
    use chm_kinds
    use dimens_fcm
    use stream
    use image
    use version
    implicit none
    !
    !       Passed Variables
    !
    INTEGER NATOM,FREEAT(*),NFREAT,NPRIV,ISTEP,NDEGF
    INTEGER NSAVC,NSTEPS
    INTEGER NTITLA,IUNCRD
    real(chm_real) X(NATOM),Y(NATOM),Z(NATOM), DELTA
    CHARACTER(len=80) :: TITLE(:)
    !
    !       Local Variables
    !
    INTEGER ICNTRL(20),IFILE,NFILE,I
    REAL(chm_real4) DELTA4
    LOGICAL ERROR,QCRYS
    CHARACTER(len=4), PARAMETER :: HDR='CORD'
    !
    !       Save all variables between calls to WRITMC
    !
    SAVE
    !
    !       If writing is not allowed, return.
    !
    IF(IUNCRD.LT.0 .OR. IOLEV.LT.0) RETURN
    !
    !       The calculation of IFILE and NFILE may have to be modified
    !       if they become too large due to QUANTA limits.
    !
    IFILE = ISTEP/NSAVC
    NFILE = NSTEPS/NSAVC
    QCRYS = (XTLTYP.NE.'    ')

    IF (IFILE.EQ.1) THEN
       !
       !         Initiation of Control header
       !
       ICNTRL(1:20) = 0
       ICNTRL(1)=NFILE
       ICNTRL(2)=NPRIV
       ICNTRL(3)=NSAVC
       ICNTRL(4)=NSTEPS
       ICNTRL(8)=NDEGF
       ICNTRL(9)=NATOM-NFREAT
       DELTA4=DELTA
       ! store bytes of delta4 in icntrl(10)
       icntrl(10) = transfer(delta4, icntrl(10))
       IF(QCRYS) ICNTRL(11)=1
       ICNTRL(20)=VERNUM

       WRITE(IUNCRD) HDR,ICNTRL
       CALL WRTITL(TITLE,NTITLA,IUNCRD,-1)
       WRITE(IUNCRD) NATOM
       IF (NFREAT.NE.NATOM) WRITE(IUNCRD) (FREEAT(I),I=1,NFREAT)
       !
       !         Finally, the actual coordinate writes
       !

#if KEY_SINGLE==1
       IF(QCRYS) WRITE(IUNCRD) XTLABC
       WRITE(IUNCRD) X
       WRITE(IUNCRD) Y
       WRITE(IUNCRD) Z
    ELSE IF (NFREAT .EQ. NATOM) THEN
       !
       !         IFILE is not 1.  No initiation.
       !         If the number of free atoms is equal to the total, write all.
       !
       IF(QCRYS) WRITE(IUNCRD) XTLABC
       WRITE(IUNCRD) X
       WRITE(IUNCRD) Y
       WRITE(IUNCRD) Z
    ELSE
       !
       !         Otherwise, write free coordinates only.
       !
       IF(QCRYS) WRITE(IUNCRD) XTLABC
       WRITE(IUNCRD) (X(FREEAT(I)),I=1,NFREAT)
       WRITE(IUNCRD) (Y(FREEAT(I)),I=1,NFREAT)
       WRITE(IUNCRD) (Z(FREEAT(I)),I=1,NFREAT)
#else /**/
       !
       !         Not a Cray single calculation.
       !         Pick up in the code where we left off -- the initial coord write.
       !
       IF(QCRYS) WRITE(IUNCRD) XTLABC
       WRITE(IUNCRD) (SNGL(X(I)),I=1,NATOM)
       WRITE(IUNCRD) (SNGL(Y(I)),I=1,NATOM)
       WRITE(IUNCRD) (SNGL(Z(I)),I=1,NATOM)
    ELSE IF (NFREAT .EQ. NATOM) THEN
       !
       !         IFILE is not 1.  No initiation.
       !         If the number of free atoms is equal to the total, write all.
       !
       IF(QCRYS) WRITE(IUNCRD) XTLABC
       WRITE(IUNCRD) (SNGL(X(I)),I=1,NATOM)
       WRITE(IUNCRD) (SNGL(Y(I)),I=1,NATOM)
       WRITE(IUNCRD) (SNGL(Z(I)),I=1,NATOM)
    ELSE
       !
       !         Otherwise, write free coordinates only.
       !
       IF(QCRYS) WRITE(IUNCRD) XTLABC
       WRITE(IUNCRD) (SNGL(X(FREEAT(I))),I=1,NFREAT)
       WRITE(IUNCRD) (SNGL(Y(FREEAT(I))),I=1,NFREAT)
       WRITE(IUNCRD) (SNGL(Z(FREEAT(I))),I=1,NFREAT)
#endif 
    ENDIF
    !
    !       LN MOD /APR 90
    !       Make sure everything is put on disk (which is needed on some
    !       machines in case of a job crash
    !
    CALL SAVEIT(IUNCRD)
    !
    !       If there are more coordinates to be written, just return.
    !
    IF(IFILE.LT.NFILE) RETURN

    IF(PRNLEV.GE.2) THEN
       WRITE(OUTU,101) (ICNTRL(I),I=1,3),IUNCRD
101    FORMAT(/2X,I5,'   COORDINATE SETS STARTING FROM',/, &
            5X,'STEP NO ',I8,'   FOR EVERY ',I5,'  STEPS',/, &
            5X,'WRITTEN ON UNIT',I5,/)
    ENDIF

    CALL VCLOSE(IUNCRD,'KEEP',ERROR)

    RETURN
  END SUBROUTINE WRITMC

  SUBROUTINE MCSTAT(NMVTYP,NACMVG,IACMVG,MVLABL,NMVTOT,NMVACC)
    !
    !       Prints some simple acceptance statistics for MC
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use number
    use stream
    implicit none
    INTEGER NMVTYP, NACMVG, IACMVG(NACMVG)
    INTEGER NMVTOT(NMVTYP), NMVACC(NMVTYP)
    CHARACTER(len=4) MVLABL(NMVTYP)
    !
    INTEGER I
    real(chm_real)  R

    IF (PRNLEV .GT. 0) THEN
       WRITE (OUTU,'(A13,A7,5X,A12,5X,A12,2X,A5)') 'MCSTAT: Index', &
            'Label','Total', 'Accepted', 'Ratio'
       WRITE (OUTU,'(A13,A7,2X,A15,2X,A15,2X,A5)') '------  -----', &
            '  -----','---------------', '---------------', &
            '-----'
       DO I = 1, NACMVG
          IF (NMVTOT(IACMVG(I)).GT.0) THEN
             R = FLOAT(NMVACC(IACMVG(I)))/NMVTOT(IACMVG(I))
          ELSE
             R = ZERO
          ENDIF
          WRITE (OUTU,'(A7,I6,A7,5X,I12,5X,I12,2X,F5.3)') 'MCSTAT>', &
               IACMVG(I), MVLABL(IACMVG(I)), NMVTOT(IACMVG(I)), &
               NMVACC(IACMVG(I)), R
       ENDDO
       WRITE (OUTU,'(A13,A7,2X,A15,2X,A15,2X,A5)') '------  -----', &
            '  -----','---------------', '---------------', &
            '-----'
    ENDIF

    RETURN
  END SUBROUTINE MCSTAT

  SUBROUTINE RDMULT(IUNIT,MLTLGP,NMULT,EMULTN, EMULTX,DEMULT)
    !
    !       Reads the multicanonical weights
    !
    !       The file format is:
    !
    !           CHARMM title
    !           Emin  Emax   Nbin
    !           i     E_i    ln[n(E_i)]
    !                  .
    !                  .
    !                  .
    !           Nbin  E_Nbin ln[n(E_Nbin)]
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use ctitla
    use memory
    use stream
    implicit none
    !
    !       Passed Variables
    !

    type(chm_ptr) :: MLTLGP
    INTEGER NMULT
    real(chm_real)  EMULTN, EMULTX, DEMULT
    !
    INTEGER IUNIT, I, J
    real(chm_real)  E, ELGMLT
    LOGICAL ERROR

    !
    IF (IUNIT .EQ. -1) &
         CALL WRNDIE (-5, '<RDMULT>', 'INVALID UNIT NUMBER')

    CALL RDTITL(TITLEB,NTITLB,IUNIT,0)

    READ (IUNIT,*) EMULTN, EMULTX, NMULT
    DEMULT = (EMULTX - EMULTN)/NMULT

    IF (PRNLEV .GE. 2) THEN
       WRITE (OUTU,'(a,f12.3)') ' RDMULT> multicanonical EMIN = ', &
            EMULTN
       WRITE (OUTU,'(a,f12.3)') ' RDMULT> multicanonical EMAX = ', &
            EMULTX
       WRITE (OUTU,'(a,f12.3)') ' RDMULT> multicanonical DE   = ', &
            DEMULT
    ENDIF

    call chmalloc('mcio.src','RDMULT','MLTLGP',NMULT,crlp=MLTLGP%A)
    DO I = 1, NMULT
       READ (IUNIT,*) J, E, ELGMLT
       MLTLGP%A(I) = ELGMLT
    ENDDO
    CALL VCLOSE(IUNIT,'READ',ERROR)
    RETURN
  END SUBROUTINE RDMULT

#endif 
end module mcio

