module secondary_structure
  use chm_kinds
  use dimens_fcm
  use number
    implicit none

    integer,PARAMETER :: IO=7,IN=IO+1,IHN=IN+1,ICA=IHN+1,IRSN=ICA+1
    LOGICAL QUIET,QVERB,QSTRICT
    integer,PARAMETER :: IALPH=1,IBET=2,I310=3,IPI=4
    integer,allocatable,dimension(:,:) :: ire,jre
    real(chm_real) CUTH,CUTA
contains

  SUBROUTINE SECSTR(W,ISEL,JSEL,MODE)
    !-----------------------------------------------------------------------
    !     Computes secondary structure of residues in ISEL in the context
    !     of JSEL
    !     Currently using Kabsch&Sander (Biopolymers 22, 1983, 2577)
    !     definition of alpha-helix and beta-strand (ladder).
    ! 
    !     Sets CHARMM variables ?NALPHA and ?NBETA to number of residues
    !     in alpha/beta structure
    !     ?ALPHA and ?BETA  are set to fraction of residues with that type
    !     of structure. The fraction is computed from number of peptide
    !     residues in the first selection. 
    !     On return Calphas have W-array set to 0, 1 (alpha), 2 (beta)
    !
    !     COOR SECStructure [first-selection [second-selection]] - 
    !        [QUIEt | VERBose] [CUTH real] [CUTA real] [STRIct]
    ! 
    !     The default H-bond criterion is CUTH=2.6, slightly longer than
    !     the default 2.4 used in coor hbond (from DeLoof et al JACS 1992);
    !     this is to be slightly more generous in defining secondary structures.
    !     STRIct keyword enforces strict adherence to K&S concering endresidues of
    !     helices (ie, they are typically not included); default is not strict, so 
    !     the ends are included in the calculation of % alpha (?ALPHA).
    !     L. Nilsson, KI, December 2005
    !
  use exfunc
  use comand
  use psf
  use memory
  use string

    real(chm_real) W(*)
    INTEGER ISEL(*),JSEL(*)
    !
    INTEGER NISEL,NJSEL,MODE
    QUIET=.TRUE.
    QVERB=.FALSE.
    IF(MODE > 0) THEN 
      !
      !     QUIEt overrides VERBOse
      QSTRICT=INDXA(COMLYN,COMLEN,'STRI') > 0
      QUIET=INDXA(COMLYN,COMLEN,'QUIE') > 0
      QVERB=INDXA(COMLYN,COMLEN,'VERB') > 0 .AND. .NOT. QUIET
      !     Default to no angle and r(H...A) < 2.6A
      !     ( cf, 2.4A in DeLoof et al JACS 1992))
      CUTA=GTRMF(COMLYN,COMLEN,'CUTA',NINE99)
      CUTH=2.6
!      CUTH=GTRMF(COMLYN,COMLEN,'CUTHB',CUTH)
      CUTH=GTRMF(COMLYN,COMLEN,'CUT',CUTH)
    ENDIF
    call chmalloc('secstr.src','SECSTR','IRE',irsn,NRES,intg=IRE)
    call chmalloc('secstr.src','SECSTR','JRE',irsn,NRES,intg=JRE)
    ! Setup index arrays
    CALL SECSTR1(ISEL,JSEL, &
         ATYPE,NRES,IBASE,NISEL,NJSEL)
    !
    ! Compute secondary structure
    CALL SECSTR2(W, &
         NATOM,NRES,RES,NISEL,NJSEL)

    call chmdealloc('secstr.src','SECSTR','IRE',irsn,NRES,intg=IRE)
    call chmdealloc('secstr.src','SECSTR','JRE',irsn,NRES,intg=JRE)
    RETURN
  END SUBROUTINE SECSTR

  SUBROUTINE SECSTR1(ISEL,JSEL,ATYPE,NRES,IBASE,NISEL,NJSEL)
  use exfunc
  use stream

    INTEGER IDXO,IDXN,IDXHN,IDXCA,IDXC,NISEL,NJSEL
    INTEGER ISEL(*),JSEL(*),NRES,IBASE(*)
    character(len=8) ATYPE(*)

    INTEGER I,J,K,L,IS,IQ
    LOGICAL QPEP,QI,QJ

    ! Setup indexing for ISEL and JSEL
    ! Assume that all residues having one CA atom, one N atom and one C atom
    ! are regular amino acids.
    ! N and C termini may need additional work
    !
#if KEY_DEBUG==1
    WRITE(*,*) 'NRES,IO,IRSN:',NRES,IO,IRSN 
#endif
    I=0
    J=0
    DO K=1,NRES
       IDXCA=0
       IDXC=0
       IDXO=0
       IDXN=0 
       IDXHN=0 
       QI=.FALSE.
       QJ=.FALSE.
       IS=IBASE(K)+1
       IQ=IBASE(K+1)
       DO L=IS,IQ
          IF(ATYPE(L) == 'CA') IDXCA=L
          IF(ATYPE(L) == 'C')  IDXC=L
          IF(ATYPE(L) == 'O')  IDXO=L
          IF(ATYPE(L) == 'N')  IDXN=L
          ! Allow H as name of amide hydrogen 
          IF(ATYPE(L) == 'H') IDXHN=L
          IF(ATYPE(L) == 'HN') IDXHN=L
          ! If any atom in residue is in selection, consider residue as selected
          IF(ISEL(L) == 1) QI=.TRUE.
          IF(JSEL(L) == 1) QJ=.TRUE.
       ENDDO
       ! J-array needs to be zeroed since we will index into it with RES NUMBER 
       JRE(1:irsn,K)=0

       ! assume regular aa if atom names C, CA, N are present
       QPEP=IDXC > 0.AND.IDXCA > 0.AND.IDXN > 0
#if KEY_DEBUG==1
       IF(.NOT.QPEP)THEN
          write(*,*) 'Not aa? I,KIDXCA,IDXC,IDXO,IDXN,IDXHN'
          write(*,*) I,K,IDXCA,IDXC,IDXO,IDXN,IDXHN
       ENDIF
#endif 
       IF(QPEP)THEN
          IF(QI)THEN
             I=I+1
             IRE(ICA,I)=IDXCA
             IRE(IN,I)=IDXN
             IRE(IHN,I)=IDXHN
             IRE(IO,I)=IDXO
             IRE(IRSN,I)=K
             IRE(1:io-1,I)=0
          ENDIF
          IF(QJ)THEN
             J=J+1
             JRE(ICA,K)=IDXCA
             JRE(IN,K)=IDXN
             JRE(IHN,K)=IDXHN
             JRE(IO,K)=IDXO
             JRE(IRSN,K)=K
          ENDIF
       ENDIF
    ENDDO
    NISEL=I
    NJSEL=J
    IF(I > NRES .OR. J >  NRES)  &
         CALL WRNDIE(-3,'<SECSTR>','INTERNAL SELECTION ERROR')
    IF(I == 0) THEN
       CALL WRNDIE(-1,'<SECSTR>','No amino acids in first selection')
       RETURN
    ENDIF
    IF(J  == 0) THEN
       CALL WRNDIE(-1,'<SECSTR>', &
            'No amino acids in second selection')
       RETURN
    ENDIF
    RETURN
  END SUBROUTINE SECSTR1

  SUBROUTINE SECSTR2(W,NATOM,NRES,RES,NISEL,NJSEL)
  use exfunc
  use stream
  use hbanal_mod,only:qdisa2
  use param_store, only: set_param

    implicit none

    INTEGER IDXO,IDXN,IDXHN,IDXCA,IDXC,NISEL,NJSEL,NATOM,NRES
    real(chm_real) W(natom)  !,CUTH,CUTA
    character(len=8) RES(*)
    INTEGER IA,ID,IH,I,J,K,L,NALPHA,NBETA,IR1,JR1,IM1,IRM1,IRP1
    INTEGER IR,I1,I2,JR,JMAX,IRM2,IRP2,NOFFS,N310,NPI
    real(chm_real) ALPHA,BETA,H310,HPI
    character(len=1) TP(0:4)
    DATA TP/' ','H','E','3','P'/

    IF(NISEL <= 0 .OR. NJSEL <= 0) RETURN
    W=zero
    ! Now find out who is hydrogen-bonded to whom
    ! For now use routines from HBANAL, which means MAIN coordinates are used
    !
#if KEY_DEBUG==1 && KEY_DEBUG==1
    WRITE(*,*) 'NISEL,NRES,IRSN:',NISEL,NRES,IRSN    
#endif
    DO I=1,NISEL
       ! I-acceptors
       IA=IRE(IO,I)
       IR=IRE(IRSN,I)
       IF(IA > 0)THEN 
          K=0
          DO J=1,NRES
             ID=JRE(IHN,J)
             IH=JRE(IN,J)
             IF(ID > 0.AND.IR /= JRE(IRSN,J))THEN    
                IF(QDISA2(ID,IA,IH,CUTH,CUTA,.FALSE.)) THEN
                   K=K+1
                   IF(K >= IO)  &
                        CALL WRNDIE(-3,'<SECSTR>','INTERNAL ERROR')
#if KEY_DEBUG==1
                   WRITE(*,*) 'K,I:', K,I   
#endif
                   IRE(K,I)=JRE(IRSN,J)
                ENDIF
             ENDIF
          ENDDO
       ENDIF
    ENDDO

    ! J-acceptors, for SHEET definitions
    DO J=1,NRES
       IA=JRE(IO,J)
       JR=JRE(IRSN,J)
       IF(IA > 0)THEN 
          K=0
          DO I=1,NISEL
             ID=IRE(IHN,I)
             IH=IRE(IN,I)
             IF(ID > 0.AND.JR /= IRE(IRSN,I))THEN    
                IF(QDISA2(ID,IA,IH,CUTH,CUTA,.FALSE.)) THEN
                   K=K+1
                   IF(K >= IO)  &
                        CALL WRNDIE(-3,'<SECSTR>','INTERNAL ERROR')
                   JRE(K,J)=IRE(IRSN,I)
                ENDIF
             ENDIF
          ENDDO
       ENDIF
    ENDDO
#if KEY_DEBUG==1
    !
    ! Let's see what we have
    WRITE(*,*) 'I-selection'
    DO I=1,NISEL
       WRITE(*,*) I,(IRE(J,I),J=1,IRSN)
    ENDDO
    WRITE(*,*) 'J-selection'
    DO I=1,NRES
       WRITE(*,*) I,(JRE(J,I),J=1,IRSN)
    ENDDO
#endif 
    !
    ! Kabsch & Sander criteria for helix or sheet; need also to check that
    ! basic imperfections are handled according to K&S
    ! For now also assume that ISEL is reasonably continuous.
    !
    ! MINIMAL ALPHA HELIX (4 RES, I+1 -> I+4): H-bonds (I,I+4),(I+1,I+5)
    NALPHA=0
    DO I=1,NISEL-4
       IR=IRE(IRSN,I)
       IR1=IRE(IRSN,I+1)
       IF(IR1 /= IR+1) &
            CALL WRNDIE(-3,'<SECSTR>','NON-CONTIGUOUS PEPTIDE')
       I1=0
       I2=0
       DO K=1,IO-1
          IF(IRE(K,I) == IR+4) I1=1
          IF(IRE(K,I+1) == IR1+4) I2=1
       ENDDO
       IF(I1 == 1 .AND. I2 == 1)THEN
          W(IRE(ICA,I+1))=IALPH
          W(IRE(ICA,I+2))=IALPH
          W(IRE(ICA,I+3))=IALPH
          W(IRE(ICA,I+4))=IALPH
       ENDIF
       IF(W(IRE(ICA,I)) == IALPH) NALPHA=NALPHA+1
    ENDDO
    DO I=NISEL-3,NISEL
       IF(W(IRE(ICA,I)) == IALPH) NALPHA=NALPHA+1
    ENDDO
    ! K&S definition "misses" one residue at each end of perfect alpha-helix;
    ! Fraction alpha is corrected to account for this, so we should get 100% for perfect helix.
    ! If last residue in first selection is denoted as alpha only the N-terminal 
    ! "correction" is required.

    IF(QSTRICT)THEN
      NOFFS=0
    ELSE
      NOFFS=2
      IF(W(IRE(ICA,NISEL))==IALPH) NOFFS=1
    ENDIF

    ! 3-10 HELIX
    N310=0
    DO I=1,NISEL-3
       IR=IRE(IRSN,I)
       IR1=IRE(IRSN,I+1)
       IF(IR1 /= IR+1) &
            CALL WRNDIE(-3,'<SECSTR>','NON-CONTIGUOUS PEPTIDE')
       I1=0
       I2=0
       DO K=1,IO-1
          IF(IRE(K,I) == IR+3) I1=1
          IF(IRE(K,I+1) == IR1+3) I2=1
       ENDDO
       IF(I1 == 1 .AND. I2 == 1)THEN
          IF (W(IRE(ICA,I+1))==0) W(IRE(ICA,I+1))=I310
          IF (W(IRE(ICA,I+2))==0) W(IRE(ICA,I+2))=I310
          IF (W(IRE(ICA,I+3))==0) W(IRE(ICA,I+3))=I310
       ENDIF
       IF(W(IRE(ICA,I)) == I310) N310=N310+1
    ENDDO
    DO I=NISEL-2,NISEL
       IF(W(IRE(ICA,I)) == I310) N310=N310+1
    ENDDO

    ! PI HELIX
    NPI=0
    DO I=1,NISEL-5
       IR=IRE(IRSN,I)
       IR1=IRE(IRSN,I+1)
       IF(IR1 /= IR+1) &
            CALL WRNDIE(-3,'<SECSTR>','NON-CONTIGUOUS PEPTIDE')
       I1=0
       I2=0
       DO K=1,IO-1
          IF(IRE(K,I) == IR+5) I1=1
          IF(IRE(K,I+1) == IR1+5) I2=1
       ENDDO
       IF(I1 == 1 .AND. I2 == 1)THEN
          IF (W(IRE(ICA,I+1))==0) W(IRE(ICA,I+1))=IPI
          IF (W(IRE(ICA,I+2))==0) W(IRE(ICA,I+2))=IPI
          IF (W(IRE(ICA,I+3))==0) W(IRE(ICA,I+3))=IPI
          IF (W(IRE(ICA,I+4))==0) W(IRE(ICA,I+4))=IPI
          IF (W(IRE(ICA,I+5))==0) W(IRE(ICA,I+5))=IPI
       ENDIF
       IF(W(IRE(ICA,I)) == IPI) NPI=NPI+1
    ENDDO
    DO I=NISEL-4,NISEL
       IF(W(IRE(ICA,I)) == IPI) NPI=NPI+1
    ENDDO


    !
    ! BETA:  I and J AR PART OF BETA LADDER IF
    !      (I-1,J) AND (J,I+1)   or (J-1,I) AND (I,J+1)
    ! or   (I,J) AND (J,I)  or  (I-1,J+1) AND (J-1,I+1)
    NBETA=0
    DO I=1,NISEL
       ! If RES I has already been classified as ALPHA, no need to check for BETA
       IF(W(IRE(ICA,I)) /= IALPH)THEN   
          IR=IRE(IRSN,I)
          IRM1=IR-1
          IRM2=IR-2
          IRP1=IR+1
          IRP2=IR+2
          I1=0
          ! case 1
          IF(IR > 1.AND.IR < NRES.AND.I > 1)THEN
             DO K=1,IO-1
                JR=IRE(K,I-1)
                IF(JR > 0.AND. (JR < IRM2 .OR.JR > IRP2))THEN
                   DO L=1,IO-1
                      IF(JRE(L,JR) == IRP1)THEN
                         I1=1
#if KEY_DEBUG==1
                         write(*,*) 'Case 1, I,K,JR,L:',I,K,JR,L 
#endif
                      ENDIF
                   ENDDO
                ENDIF
             ENDDO
          ENDIF
          ! case 2
          DO K=1,IO-1
             JR=IRE(K,I)-2
             IF(JR > 0 .AND.((JR+1) < IRM2 .OR. (JR+1) > IRP2))THEN
                DO L=1,IO-1
                   IF(JRE(L,JR) == IR) THEN
                      I1=1
#if KEY_DEBUG==1
                      write(*,*) 'Case 2, I,K,JR,L:',I,K,JR,L 
#endif
                   ENDIF
                ENDDO
             ENDIF
          ENDDO
          ! case 3
          DO K=1,IO-1
             JR=IRE(K,I)
             IF(JR > 0 .AND. (JR < IRM2 .OR. JR > IRP2))THEN
                DO L=1,IO-1
                   IF(JRE(L,JR) == IR) THEN
                      I1=1
#if KEY_DEBUG==1
                      write(*,*) 'Case 3, I,K,JR,L:',I,K,JR,L 
#endif
                   ENDIF
                ENDDO
             ENDIF
          ENDDO
          ! case 4
          IF(IR > 1 .AND. IR < NRES .AND. I > 1)THEN
             DO K=1,IO-1
                JR=IRE(K,I-1)-2
                IF(JR > 0 .AND.((JR+1) < IRM2.OR.(JR+1) > IRP2))THEN
                   DO L=1,IO-1
                      IF(JRE(L,JR) == IRP1)THEN
                         I1=1
#if KEY_DEBUG==1
                         write(*,*) 'Case 4, I,K,JR,L:',I,K,JR,L 
#endif
                      ENDIF
                   ENDDO
                ENDIF
             ENDDO
          ENDIF
          !
          IF(I1 == 1)THEN
             W(IRE(ICA,I))=IBET
             NBETA=NBETA+1
          ENDIF
       ENDIF
    ENDDO
! NOFFS correction should be 1 if the last residue in first selection is NOT alpha helical
    ALPHA=FLOAT(NALPHA+NOFFS)/FLOAT(NISEL)
    BETA=FLOAT(NBETA)/FLOAT(NISEL)
    H310=FLOAT(N310)/FLOAT(NISEL)
    HPI =FLOAT(NPI)/FLOAT(NISEL)
    call set_param('ALPHA',ALPHA)
    call set_param('BETA',BETA)
    call set_param('H310',H310)
    call set_param('HPI',HPI)
    CALL set_param('NALPHA',NALPHA)
    CALL set_param('NBETA',NBETA)
    CALL set_param('N310',N310)
    CALL set_param('NPI',NPI)
    IF(QUIET)  RETURN
    IF(PRNLEV < 2) RETURN
    IF(.NOT. QSTRICT) THEN
        WRITE(OUTU,'(/,A)') & 
       'Secondary structure (Kabsch&Sander) analysis; including end residues in % alpha calculation'
    ELSE
         WRITE(OUTU,'(/,A)') 'Secondary structure (Kabsch&Sander) analysis.'
    ENDIF
    WRITE(OUTU,100) NISEL,NJSEL,NALPHA,INT(ALPHA*100), &
         NBETA,INT(BETA*100),N310,INT(H310*100),NPI,INT(HPI*100)
100 FORMAT(/'Using',I7,' aa in a context of',I7,' aa.', &
         / I7,' aa in alpha-helix (',I3,'%), and', &
         I7,' aa in beta-strands (',I3,'%)', &
         / I7,' aa in 3-10 helix (',I3,'%), and', &
         I7,' aa in pi helix (',I3,'%).')
    IF(.NOT.QVERB) RETURN
    DO I=1,NISEL,15
       JMAX=MIN(NISEL,I+14)
       WRITE(OUTU,110) IRE(IRSN,I),(RES(IRE(IRSN,J)),J=I,JMAX)
       WRITE(OUTU,120) (TP(int(W(IRE(ICA,J)))),J=I,JMAX)
110    FORMAT(I7,': ',14(A3,'-'),A3)
120    FORMAT(9X,15(1X,A1,2X),/)
    ENDDO
    RETURN
  END SUBROUTINE SECSTR2


end module secondary_structure

