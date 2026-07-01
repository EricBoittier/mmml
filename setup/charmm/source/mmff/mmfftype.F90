#if KEY_MMFF==1
! =====================================================================
! SUBROUTINE ARFUSE : Detect aromatic ring fusions
! SUB-RING "NR" IS A CANDIDATE FOR BEING DECLARED AROMATIC.
! IT HAS, WITHIN IT, 4 PI ELECTRONS.  IT WILL BE DECLARED
! AROMATIC IF IT IS FUSED, VIA TWO EXOCYCLIC DOUBLE BONDS, TO
! A SECOND OF THE "NRNGS" SUB-RINGS WHICH ITSELF HAS ALREADY
! BEEN CLASSIFIED AS AROMATIC
! =====================================================================
SUBROUTINE ARFUSE(NR,NRNGS,LIST,MAXLEV,MXSIZE,NRGAT,AROM,AROMRG)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "JM" to "JJM" to avoid
  !                       conflict with CHARMM name.
  !
  !  20 Oct 95 Jay Banks: removed "ITAB" as an argument, since it's now in
  !                       mmff.f90.
  !
  !  22 Nov 95 Jay Banks: changed WRITE(OUTU,*) to explicit format.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use stream
  implicit none
  !
  integer MXSIZE,MAXLEV
  integer LIST(MAXLEV,MXSIZE)
  integer NRGAT(*)
  logical AROM(*)
  LOGICAL AROMRG
  !
  integer i1, i2, ii, ij, ijbnds, inr
  integer jj, jjm, k, m
  integer mndbl
  integer nr, nrngs
  !
  AROMRG=.FALSE.
  DO M=1,NRNGS
     IF(M.EQ.NR) GOTO 100
     IF(.NOT.AROM(M)) GOTO 100
     !   SUB-RING "M" IS AROMATIC;  EXAMINE FOR POSSIBLE FUSION
     MNDBL=0
     DO INR=1,NRGAT(NR)
        II=LIST(NR,INR)
        DO JJM=1,NRGAT(M)
           JJ=LIST(M,JJM)
           IJBNDS=0
           DO K=1,4
              IF(ITAB(K,II).LE.0) GOTO 300
              IF(ITAB(K,II).EQ.JJ) THEN
                 !         WRITE(OUTU,*) ' K,II,JJ',K,II,JJ
                 !   MAKE SURE JJ IS NOT ALSO IN SUBRING NR
                 DO IJ=1,NRGAT(NR)
                    IF(LIST(NR,IJ).EQ.JJ) GOTO 400
                 ENDDO
                 IJBNDS=IJBNDS+1
                 !        WRITE(OUTU,*) ' IJBNDS',IJBNDS
              ENDIF
400           CONTINUE
           ENDDO
           IF(IJBNDS.GE.2) THEN
              if (prnlev .ge. 2) WRITE(OUTU,'(A,2I6)') &
                   ' EXOCYCLIC MULT BOND',II,JJ
              !     IF(IZ.NE.0) WRITE(IZ,*) ' EXOCYCLIC MULT BOND',II,JJ
              MNDBL=MNDBL+1
              IF(MNDBL.EQ.1) I1=II
              IF(MNDBL.EQ.2) I2=II
           ENDIF
300        CONTINUE
        ENDDO
     ENDDO
     IF(MNDBL.EQ.2) THEN
        !   SO FAR, SO GOOD.  TWO EXOCYCLIC MULTIPLE (PRESUMABLY DOUBLE) BONDS
        !   HAVE BEEN FOUND.  ARE THE "NR" RING ATOMS (I1 AND I2) BONDED
        IF(CONN12(I1,I2).EQ.1) THEN
           !         YES, THEY ARE.  DECLARE RING NR AS AROMATIC
           AROMRG=.TRUE.
           AROM(NR)=.TRUE.
        ENDIF
     ENDIF
100  CONTINUE
  ENDDO
  RETURN
END SUBROUTINE ARFUSE

! =====================================================================
! SUBROUTINE AR2FUSE : Detect aromatic ring fusions
! SUB-RING "NR" IS A CANDIDATE FOR BEING DECLARED AROMATIC.
! IT HAS, WITHIN IT, 2 PI ELECTRONS.  IT WILL BE DECLARED
! AROMATIC IF IT IS FUSED, VIA TWO SETS OF TWO EXOCYCLIC DOUBLE BONDS,
! TO TWO OTHER OF THE "NRNGS" SUB-RINGS WHICH THEMSELVES HAVE ALREADY
! BEEN CLASSIFIED AS AROMATIC
! =====================================================================
SUBROUTINE AR2FUSE(NR,NRNGS,LIST,MAXLEV,MXSIZE,NRGAT,AROM,AROMRG)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  Written by T. Halgren but based on the adaptation by R. Czerminski
  !  of Molecular Simulations, Inc. of the similar subroutine ARFUSE,
  !  itself based on code developed at Merck and Co., Inc. by T. Halgren
  !  and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "JM" to "JJM" to avoid
  !                       conflict with CHARMM name.
  !
  !  20 Oct 95 Jay Banks: removed "ITAB" as an argument, since it's now in
  !                       mmff.f90.
  !
  !  22 Nov 95 Jay Banks: changed WRITE(OUTU,*) to explicit format.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  19 May 98 Tom Halgren: added test to make sure a double fusion involves
  !                         two non-overlapping fusion bonds
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use stream
  implicit none
  !
  integer MXSIZE,MAXLEV
  integer LIST(MAXLEV,MXSIZE)
  integer NRGAT(*)
  logical AROM(*)
  LOGICAL AROMRG
  !
  integer i1, i2, ii, ij, ijbnds, inr
  integer jj, jjm, k, m
  integer mndbl
  integer nr, nrngs
  integer NFUSE, NF1, NF2
  INTEGER IFUSE(2,3)
  !
  AROMRG=.FALSE.
  NFUSE=0
  DO NF1=1,3
     DO K=1,2
        IFUSE(K,NF1)=0
     ENDDO
  ENDDO
  DO M=1,NRNGS
     IF(M.EQ.NR) GOTO 100
     IF(.NOT.AROM(M)) GOTO 100
     !   SUB-RING "M" IS AROMATIC;  EXAMINE FOR POSSIBLE FUSION
     MNDBL=0
     DO INR=1,NRGAT(NR)
        II=LIST(NR,INR)
        DO JJM=1,NRGAT(M)
           JJ=LIST(M,JJM)
           IJBNDS=0
           DO K=1,4
              IF(ITAB(K,II).LE.0) GOTO 300
              IF(ITAB(K,II).EQ.JJ) THEN
                 !         WRITE(OUTU,*) ' K,II,JJ',K,II,JJ
                 !   MAKE SURE JJ IS NOT ALSO IN SUBRING NR
                 DO IJ=1,NRGAT(NR)
                    IF(LIST(NR,IJ).EQ.JJ) GOTO 400
                 ENDDO
                 IJBNDS=IJBNDS+1
                 !        WRITE(OUTU,*) ' IJBNDS',IJBNDS
              ENDIF
400           CONTINUE
           ENDDO
           IF(IJBNDS.GE.2) THEN
              if (prnlev .ge. 2) WRITE(OUTU,'(A,2I6)') &
                   ' EXOCYCLIC MULT BOND',II,JJ
              !     IF(IZ.NE.0) WRITE(IZ,*) ' EXOCYCLIC MULT BOND',II,JJ
              MNDBL=MNDBL+1
              IF(MNDBL.EQ.1) I1=II
              IF(MNDBL.EQ.2) I2=II
           ENDIF
300        CONTINUE
        ENDDO
     ENDDO
     IF(MNDBL.EQ.2) THEN
        !   SO FAR, SO GOOD.  TWO EXOCYCLIC MULTIPLE (AT LEAST DOUBLE) BONDS
        !   HAVE BEEN FOUND.  ARE THE "NR" RING ATOMS (I1 AND I2) BONDED?
        IF(CONN12(I1,I2).EQ.1) THEN
           !          YES, THEY ARE.  INCREMENT THE COUNT FOR NFUSE
           NFUSE=NFUSE+1
           IF(NFUSE.GT.3) CALL WRNDIE(-5,'<ar2fuse>', &
                'ring fusion pattern exceeds programmed limit of 3')
           !
           ! RECORD WHICH ATOMS IN THE 5-RING THESE ARE
           !
           IFUSE(1,NFUSE)=I1
           IFUSE(2,NFUSE)=I2
        ENDIF
     ENDIF
100  CONTINUE
  ENDDO
  !
  !  IS THE DOUBLE-FUSION CONDITION MET?
  !
  IF(NFUSE.LE.1) RETURN
  AROMRG=.FALSE.
  !
  !   MAKE SURE THAT FOUR DIFFERENT ATOMS OF THE 5-RING ARE INVOLVED
  !   IN TWO NON-OVERLAPPING EXOCYCLIC DOUBLE-SINGLE-DOUBLE PATTERNS
  !
  DO NF2=2,NFUSE
     DO NF1=1,NF2-1
        IF( IFUSE(1,NF1).NE.IFUSE(1,NF2) .AND. &
             IFUSE(2,NF1).NE.IFUSE(1,NF2) .AND. &
             IFUSE(1,NF1).NE.IFUSE(2,NF2) .AND. &
             IFUSE(2,NF1).NE.IFUSE(2,NF2) ) AROMRG=.TRUE.
     ENDDO
  ENDDO
  IF(AROMRG) AROM(NR)=.TRUE.
  RETURN
END SUBROUTINE AR2FUSE

! =====================================================================
! SUBROUTINE ARRING : Detect aromatic rings
!   THIS 5- OR 6-MEMBERED RING HAS BEEN MARKED AS AROMATIC.
!   ADJUST MM2 ATOM TYPES, IF NECESSARY
!
!   IF A 5-MEMBERED HETEROCYCLIC RING, PERCEIVE WHETHER EACH SUCH
!   POSITION IS THAT OF THE PI-LONE-PAIR HETEROATOM, IS ALPHA TO
!   IT, OR IS BETA TO IT.  PASS THIS INFO TO SUBROUTINE MAKEAR
! =====================================================================
!
SUBROUTINE ARRING(LIST,MAXLEV,MXSIZE,NRING,NSIZE)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use stream
  use mmffm
  implicit none
  !
  integer MXSIZE,MAXLEV
  integer LIST(MAXLEV,MXSIZE)
  integer NRING, NSIZE, NUMNIT, MTN, NTR
  !
  integer k, kpilp, l5
  integer n, npilp
  integer L5TYPE(5)
  LOGICAL IMCAT,N5ANION
  ! T.A. Halgren change
  integer l, ln, kn
  !
  !   THIS 5- OR 6-MEMBERED RING HAS BEEN MARKED AS AROMATIC.
  !   ADJUST MMFF ATOM TYPES, IF NECESSARY, AND MARK ALL BONDS AS
  !   AROMATIC
  !
  !
  DO K = 2,NSIZE
     DO L = 1,K-1
        IF(CONN12(LIST(NRING,K),LIST(NRING,L)).EQ.1) THEN
           LN=LIST(NRING,L)
           KN=LIST(NRING,K)
           IF(N_ARBOND.EQ.0) THEN
              N_ARBOND=N_ARBOND+1
              AR_BOND(1,N_ARBOND)=LN
              AR_BOND(2,N_ARBOND)=KN
           ELSE
              DO N=1,N_ARBOND
                 IF( LN.EQ.AR_BOND(1,N) .AND. KN.EQ.AR_BOND(2,N) ) &
                      GO TO 10
                 IF( LN.EQ.AR_BOND(2,N) .AND. KN.EQ.AR_BOND(1,N) ) &
                      GO TO 10
              ENDDO
           ENDIF
           !
           !  OK - THIS AROMATIC BOND IS NOT YET ON THE LIST -- ADD IT
           !
           N_ARBOND=N_ARBOND+1
           AR_BOND(1,N_ARBOND)=LN
           AR_BOND(2,N_ARBOND)=KN
        ENDIF
10      CONTINUE
     ENDDO
  ENDDO
  !
  !   IF A 5-MEMBERED HETEROCYCLIC RING, PERCEIVE WHETHER EACH SUCH
  !   POSITION IS THAT OF THE PI-LONE-PAIR HETEROATOM, IS ALPHA TO
  !   IT, OR IS BETA TO IT.  PASS THIS INFO TO SUBROUTINE MAKEAR
  !
  ! end T.A. Halgren change
  KPILP=0
  NPILP=0
  IMCAT=.FALSE.
  N5ANION=.FALSE.
  if(prnlev.gt.5) write(outu,'(a,i5)') &
       ' arring> nsize=',nsize
  IF(NSIZE.EQ.5) THEN
     DO K=1,5
        N=LIST(NRING,K)
        MTN=MTYPE(N)
        IF(MCOORD(MTN).EQ.3.AND.MVALMAX(MTN).EQ.3.AND. &
             AtNum(N).EQ.7) THEN
           KPILP=K
           if(prnlev.gt.5) write(outu,'(a,a)') &
                ' arring> pi lone pair: symb = ',symb(n)
           GOTO 50
        ELSE IF(MCOORD(MTN).EQ.2.AND.MVALMAX(MTN).EQ.2.AND. &
             (AtNum(N).EQ.8.OR.AtNum(N).EQ.16)) THEN
           KPILP=K
           if(prnlev.gt.5) write(outu,'(a,a)') &
                ' arring> pi lone pair: symb = ',symb(n)
           GOTO 50
        ENDIF
50      CONTINUE
        IF(KPILP.NE.0) GOTO 100
     ENDDO
100  CONTINUE
     IF(KPILP.NE.0) THEN
        NPILP=LIST(NRING,KPILP)
        if(prnlev.gt.5) write(outu,'(a,3i5)') &
             ' arring> nring,kpilp,npilp=',nring,kpilp,npilp
     ELSE IF(KPILP.EQ.0) THEN
        !
        !   UNIQUE HETEROATOM POSITION NOT FOUND - SEE WHETHER THIS, INSTEAD, IS
        !   AN IMIDAZOLIUM OR IMIDAZOLIUM-DERIVED CATION OR IS A 5-RING ANNION
        !   CONTAINING ONE OR MORE NITROGENS OVER WHICH THE CHARGE CAN BE
        !   DISPERSED
        !
        DO K=1,5
           N=LIST(NRING,K)
           DO NTR=1,NTRAROM
              IF(IRSIZE(NTR).EQ.5.AND.LIMCAT(NTR).EQ.1) THEN
                 IF(SYMB(N).EQ.SYMOLD(NTR)) THEN
                    IMCAT=.TRUE.
                    if(prnlev.gt.5) write(outu,'(a,a)') &
                         ' arring> im cation: symold = ',symold(ntr)
                    GOTO 150
                 ENDIF
              ELSE IF(IRSIZE(NTR).EQ.5.AND.LN5AN(NTR).EQ.1) THEN
                 IF(SYMB(N).EQ.SYMOLD(NTR)) THEN
                    N5ANION=.TRUE.
                    if(prnlev.gt.5) write(outu,'(a,a)') &
                         ' arring> n5 anion: symold = ',symold(ntr)
                    GOTO 150
                 ENDIF
              ENDIF
           ENDDO
        ENDDO
150     CONTINUE
        IF(.NOT.IMCAT.AND..NOT.N5ANION) THEN
           CALL SAYC(' ')
           CALL SAYC(' ***** ERROR: unique pi-lone-pair hetero'// &
                ' atom not found *****')
           CALL WRNDIE(-5,'<arring>','no unique pi-lone-pair')
        ENDIF
     ENDIF
     !
     !   NOW CLASSIFY THE OTHER 5-RING POSITIONS RELATIVE TO THIS ONE
     !
     DO K=1,5
        N=LIST(NRING,K)
        IF(IMCAT.OR.N5ANION) THEN
           L5TYPE(K)=4
        ELSE IF(K.EQ.KPILP) THEN
           L5TYPE(K)=1
        ELSE IF(CONN12(N,NPILP).EQ.1) THEN
           L5TYPE(K)=2
        ELSE IF(CONN13(N,NPILP).EQ.1) THEN
           L5TYPE(K)=3
        ENDIF
     ENDDO
     !D        WRITE(OUTU,*) ' KPILP,NPILP =',KPILP,NPILP
  ENDIF
  DO K=1,NSIZE
     N=LIST(NRING,K)
     IF(NSIZE.EQ.5) THEN
        !D        WRITE(OUTU,'(A,4I5,A)') 'NRING,K,LIST,L5TYPE,SYMB',
        !D    .   NRING,K,LIST(NRING,K),L5TYPE(K),'  '//SYMB(N)
        CALL MAKEAR(N,NSIZE,L5TYPE(K))
     ELSE
        L5=0
        !D        WRITE(OUTU,'(A,3I5,A)') 'NRING,K,LIST,SYMB',
        !D    .   NRING,K,LIST(NRING,K),'  '//SYMB(N)
        CALL MAKEAR(N,NSIZE,L5)
     ENDIF
  ENDDO
  !
  !   IF THIS IS A 5-RING ANION, NOW DISTRIBUTE THE UNIT NEGATIVE
  !   CHARGE OVER ALL THE NITROGENS AND RESET THE ATOM TYPES AND
  !   FORMAL CHARGES
  !
  IF(N5ANION) THEN
     NUMNIT=0
     DO K=1,5
        N=LIST(NRING,K)
        IF(AtNum(N).EQ.7) THEN
           NUMNIT=NUMNIT+1
        ENDIF
     ENDDO
     DO K=1,5
        N=LIST(NRING,K)
        IF(AtNum(N).EQ.7) THEN
           DO NTR=1,NTRAROM
              IF(IRSIZE(NTR).EQ.5.AND.LN5AN(NTR).EQ.1) THEN
                 SYMB(N)=SYMAROM(NTR)
                 if(prnlev.gt.5) WRITE(OUTU,'(A,I4,3X,A)') &
                      ' N5 ANION: N, SYMAROM = ',N, SYMAROM(NTR)
                 GOTO 200
              ENDIF
           ENDDO
200        CONTINUE
           PARTLQ(N)=-1./NUMNIT
           CALL TRSYMB(N)
        ENDIF
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE ARRING

!  ============================================================
SUBROUTINE BINFO(IATOM,NATOM,ATNUM,ITAB,KH,KX,KATTCH,KBONDS, &
     KC,KCB,KN,KNB,KO,KOB,KS,KSB,KKP,KPB)
  !   COLLECT AND RETURN BONDING INFORMATION FOR ATOM IATOM
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "KP" to "KKP" to avoid
  !                       conflict with CHARMM name
  !
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  integer NATOM
  integer ATNUM(*)
  integer ITAB(6,natom)
  !
  integer i, iatom, imax, j, ja, kattch, kbonds, kc
  integer kcb, kh, kn, knb, ko, kob, kkp, kpb, ks, ksb, kx, l
  !
  LOGICAL NEW
  !
  !  INITIALIZE RETURN QUANTITIES
  KH=0
  !                      INITIALIZE AS # OF IMPLICIT H'S, IN CASE NOT
  !                      ALL HYDROGENS ARE PRESENT
  KX=0
  KATTCH=0
  KBONDS=0
  !                       INITIALIZE AND # OF IMPLICIT H'S
  KC=0
  KCB=0
  KN=0
  KNB=0
  KO=0
  KOB=0
  KS=0
  KSB=0
  KKP=0
  KPB=0
  ! LOOP OVER ATOMS ATTACHED TO IATOM AND INCREMENT RETURN QUANTITIES
  IMAX=5
  DO I=1,IMAX
     J=ITAB(I,IATOM)
     IF(J.LE.0) THEN
        !D     WRITE(6,110) IATOM,KATTCH,KH,KC,KN,KO,KS,KKP,KBONDS,
        !D    . KCB,KNB,KOB,KSB,KPB
        !D     WRITE(12,110) IATOM,KATTCH,KH,KC,KN,KO,KS,KKP,KBONDS,
        !D    . KCB,KNB,KOB,KSB,KPB
110     FORMAT(' ATOM',I3,'   NEIGHBORS',7I3,'    BONDS',6I3)
        !D      WRITE(6,*) (ITAB(M,IATOM),M=1,5)
        RETURN
     ENDIF
     !   SEE WHETHER THIS IS A NEW ATOM ON IATOM OR IS LISTED PREVIOUSLY
     !   (INDICATING A MULTIPLE BOND BETWEEN IATOM AND J)
     NEW=.TRUE.
     IF(I.GT.1) THEN
        DO L=1,I-1
           IF(ITAB(L,IATOM).EQ.J) NEW=.FALSE.
        ENDDO
     ENDIF
     JA=ATNUM(J)
     IF(JA.EQ.1) THEN
        KH=KH+1
     ELSE IF(JA.EQ.6) THEN
        KCB=KCB+1
        IF(NEW) KC=KC+1
        !                          MULTIPLY BONDED TO SAME C IF NEW=.FALSE.
     ELSE IF(JA.EQ.7) THEN
        KNB=KNB+1
        IF(NEW) KN=KN+1
     ELSE IF(JA.EQ.8) THEN
        KOB=KOB+1
        IF(NEW) KO=KO+1
     ELSE IF(JA.EQ.15) THEN
        KPB=KPB+1
        IF(NEW) KKP=KKP+1
     ELSE IF(JA.EQ.16) THEN
        KSB=KSB+1
        IF(NEW) KS=KS+1
     ENDIF
     KBONDS=KBONDS+1
     IF(NEW) KATTCH=KATTCH+1
     KX=KATTCH-KH
  ENDDO
  RETURN
  !
END SUBROUTINE BINFO

! ====================================================================
! SUBROUTINE HTYPE : Assign symbolic atom types for hydrogen atoms
! ====================================================================
SUBROUTINE HTYPE
  !     SUBROUTINE HTYPE(NATOM,ATNUM,ITAB)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, to avoid 
  ! conflict with variable in common (mmff.f90).
  !
  ! Jay Banks 08 Nov 95: changed common variable PTYPE (mmff.f90: parent
  ! atom type) to PATYPE to avoid conflict with CHARMM variable.
  !
  ! Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  ! WRITE statement followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use stream
  use mmffm
  implicit none
  !
  !     integer NATOM    ! number of atoms
  !     integer ATNUM(*) ! atomic numbers
  !     integer ITAB(6,NATOM)
  !
  integer iia, ierr, IPIA, IPH
  !
  CHARACTER(len=4) PSYMB
  !
  !  ---- ASSIGN ALL HYDROGEN ATOM TYPES --------------------------------
  IERR=0
  DO IIA=1,NATOM
     IF(ATNUM(IIA).NE.1) GOTO 100
     IF(MTYPE(IIA).NE.0) GOTO 100
     !  ---- SPECIAL CASE FOR HYDROGEN IN H2 -----------------------
     IPIA=ITAB(1,IIA)
     IF(ATNUM(IPIA).EQ.1) THEN
        SYMB(IIA)='HH'
        SYMB(IPIA)='HH'
        GOTO 100
     ENDIF
     SYMB(IIA)='    '
     IPIA=ITAB(1,IIA)
     PSYMB=SYMB(IPIA)
     DO IPH=1,NTRHYD
        IF(PSYMB.EQ.PATYPE(IPH)) THEN
           SYMB(IIA)=HDTYPE(IPH)
           !
           !  NOW TRANSLATE THE ATOM-TYPE SYMBOL SYMB(IIA) INTO THE MMFF NUMERICAL
           !   ATOM TYPE
           CALL TRSYMB(IIA)
           !D               WRITE(OUTU,'(A,2I5,A)')
           !D    .        ' ATOMS',IIA,IPIA, ' SYMB, PSYMB = '//SYMB(IIA)//' '
           !D    .         //SYMB(IPIA)
           GOTO 100
        ENDIF
     ENDDO
     !
     !   ERROR IF WE FELL THROUGH THE LOOP  - CANNOT TYPE THIS H
     !
     if(wrnlev.ge.2) &
          WRITE(OUTU,210) IIA, QNAME(IIA),IPIA,QNAME(IPIA),SYMB(IPIA)
210  FORMAT(/' ***** ERROR - CANNOT ASSIGN TYPE FOR HYDROGEN',I5,1X,A, &
          /'               ON PARENT ATOM',I5,1X,A,' OF TYPE ',A)
     IERR=IERR+1
100  CONTINUE
  ENDDO
  IF(IERR.EQ.0) RETURN
  !  ERROR SECTION - ONE OR MORE HYDROGENS COULD NOT NOT BE TYPED
  if(wrnlev.ge.2) WRITE(OUTU,2100) IERR
  !     IF(IZ.NE.0) WRITE(IZ,2100) IERR
2100 FORMAT(//' ***** FATAL ERROR *****' &
       /I5,' HYDROGENS COULD NOT BE', &
       ' ASSIGNED MMFF ATOM TYPES'//' ***** EXECUTION ENDING *****')
  CALL WRNDIE(-5,'<HTYPE>','HYDROGENS COULD NOT BE TYPED')
  !
  RETURN
END SUBROUTINE HTYPE

! FUNCTION IATTCH : FIND THE INT+1 OCCURRENCE OF ATOM IATTCH
! FIND THE INT+1 OCCURRENCE OF ATOM IATTCH OF ATOMIC NUMBER JATYPE
! ATTACHED TO ATOM IATOM OTHER THAN ATOM NX
! IT IS PRESUMED THAT THE NUMBER OF JA TYPE ATOMS ATTACHED TO IATOM
! IS KNOWN TO BE .GE. INT+1
! =================================================================
INTEGER FUNCTION IATTCH(IATOM,NATOM,ATNUM,ITAB,JA,NX,INT)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  implicit none
  !
  integer IATOM
  integer NATOM
  integer ATNUM(*)
  integer ITAB(6,*)
  integer ja, nx, int
  !
  integer i, j, jx, kount
  LOGICAL NEW
  !
  CHARACTER(len=10) QNAME
  external QNAME
  !
  IATTCH=0
  KOUNT=INT+1
  DO J=1,5
     JX=ITAB(J,IATOM)
     IF(JX.LE.0) GOTO 100
     IF(JX.EQ.NX) GOTO 100
     !                             INSTRUCTIONS ARE TO EXCLUDE NX FROM SEARCH
     NEW=.TRUE.
     IF(J.GT.1) THEN
        DO I=1,J-1
           IF(ITAB(I,IATOM).EQ.JX) NEW=.FALSE.
        ENDDO
     ENDIF
     IF(NEW.AND.ATNUM(JX).EQ.JA) THEN
        IF(KOUNT.EQ.1) THEN
           IATTCH=JX
           RETURN
        ELSE
           KOUNT=KOUNT-1
        ENDIF
     ENDIF
100  CONTINUE
  ENDDO
  !   IF STILL HERE, OCCURRENCE INT+1 NOT FOUND -  ERROR SOMEWHERE
  if(wrnlev.ge.2) WRITE(OUTU,200) INT+1,JA,IATOM,QNAME(IATOM)
  !     IF(IZ.NE.0) WRITE(IZ,200) INT+1,JA,IATOM,QNAME(IATOM)
200 FORMAT(/' ***** ERROR - OCCURRENCE ',I3,' OF AN ATOM WITH', &
       ' ATOMIC NUMBER',I3,' BOUND TO '/ &
       '               ATOM',I5,1X,A,' NOT FOUND')
  !
  RETURN
END FUNCTION IATTCH

! ====================================================================
! SUBROUTINE MAKEAR : MARKS AROMATIC ATOMS
! MARK ATOM N AS AROMATIC IN TERMS OF ITS SYMBOLIC AND NUMERIC
! ATOM TYPES AND IN TERMS OF LOGICAL DESIGNATOR IFAROM(N)=.TRUE.
! RETURN IF ALREADY AROMATIC AND NOT IN A 5-MEMBERED RING
! ====================================================================
SUBROUTINE MAKEAR(N,NSIZE,L5)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statement followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use stream
  use mmffm
  implicit none
  !
  integer l5, n, NTR
  integer nsize
  !
  CHARACTER(len=4) OLD,NEW
  !
  IF(IFAROM(N).AND.NSIZE.NE.5) RETURN
  !   REPLACE SYMBOLIC ATOM TYPE WITH ITS AROMATIC EQUIVALENT
  OLD=SYMB(N)
  NEW=OLD
  IF(NSIZE.EQ.6) THEN
     !D        WRITE(IO,'(A,A)') ' 6-RING: OLD = ',OLD
     !
     !   FIRST TRY FOR AN EXACT MATCH
     !
     DO NTR=1,NTRAROM
        IF(IRSIZE(NTR).EQ.6)THEN
           IF(OLD.EQ.SYMOLD(NTR)) THEN
              !D                 WRITE(IO,'(A,A)') ' 6-RING: SYMOLD = ',SYMOLD(NTR)
              NEW=SYMAROM(NTR)
              GOTO 50
           ENDIF
        ENDIF
     ENDDO
     !
     !   AN EXACT MATCH WAS NOT FOUND; TRY FOR A WILDCARDED MATCH
     !
     DO NTR=1,NTRAROM
        IF(IRSIZE(NTR).EQ.6)THEN
           IF(OLD.EQ.SYMOLD(NTR)) THEN
              !D                 WRITE(IO,'(A,A)') ' 6-RING: SYMOLD = ',SYMOLD(NTR)
              NEW=SYMAROM(NTR)
              GOTO 50
           ENDIF
        ENDIF
     ENDDO
     DO NTR=1,NTRAROM
        IF(IRSIZE(NTR).EQ.6)THEN
           IF(ATNUM(N).EQ.IANUM(NTR).AND. &
                SYMOLD(NTR)(2:2).EQ.'*') THEN
              !D                 WRITE(IO,'(A,A,A,A)') ' 6-RING: NEW = ',
              !D    .            SYMAROM(NTR),'   SYMOLD = ',SYMOLD(NTR)
              NEW=SYMAROM(NTR)
              GOTO 50
           ENDIF
        ENDIF
     ENDDO
50   CONTINUE
     !
  ELSE IF(NSIZE.EQ.5) THEN
     !
     !   NOTE WHETHER THIS IS THE PI-LONE-PAIR HETEROATOM POSITION,
     !   IS ALPHA TO IT, OR IS BETA TO IT, OR WHETHER IS AN IMIDAZOLIUM
     !   OR IMIDAZOLIUM-DERIVED RING
     !
     !D        WRITE(IO,'(A,A,A,I1)') ' 5-RING: OLD = ',
     !D    .   OLD,'  L5 = ',L5
     DO NTR=1,NTRAROM
        IF(IRSIZE(NTR).EQ.5.AND.L5.EQ.L5POS(NTR)) THEN
           !
           !   FIRST TRY FOR AN EXACT MATCH
           !
           IF(OLD.EQ.SYMOLD(NTR)) THEN
              !D                 WRITE(IO,'(A,A,A,I1)') ' 5-RING: NEW = ',
              !D    .            SYMAROM(NTR),'  L5 = ',L5
              NEW=SYMAROM(NTR)
              GOTO 60
           ENDIF
        ENDIF
     ENDDO
     !
     !   AN EXACT MATCH WAS NOT FOUND; TRY FOR A WILD-CARDED MATCH
     !
     DO NTR=1,NTRAROM
        IF(IRSIZE(NTR).EQ.5.AND.L5.EQ.L5POS(NTR)) THEN
           IF(ATNUM(N).EQ.IANUM(NTR).AND. &
                SYMOLD(NTR)(2:2).EQ.'*') THEN
              !D                 WRITE(IO,'(A,A,A,I1,A,A)') ' 5-RING: NEW = ',
              !D    .            SYMAROM(NTR),'  L5 = ',L5,' SYMOLD = ',SYMOLD(NTR)
              NEW=SYMAROM(NTR)
              GOTO 60
           ENDIF
        ENDIF
     ENDDO
60   CONTINUE
  ENDIF
  !
  !   NOW TEST THAT A CHANGE HAS BEEN MADE
  !
  IF(OLD.EQ.NEW.AND..NOT.IFAROM(N)) THEN
     if(wrnlev.ge.2) WRITE(OUTU,100) N,QNAME(N),OLD
     !     IF(IZ.NE.0) WRITE(IZ,100) N,QNAME(N),OLD
100  FORMAT(/' ***  ERROR: ATOM ',I3,1X,A,' OF TYPE ',A, &
          ' COULD NOT BE RETYPED AS AROMATIC'/ &
          ' *** EXECUTION ENDING ***')
     CALL WRNDIE(-5,'<makear>','ERROR MSG')
  ELSE
     !   MARK AS AROMATIC
     !
     IFAROM(N)=.TRUE.
  ENDIF
  !
  !   ASSIGN THE NEW TYPE
  !
  SYMB(N)=NEW
  !   TRANSLATE TO GET THE NUMERIC ATOM TYPE REGISTERED
  CALL TRSYMB(N)
  !
  RETURN
END SUBROUTINE MAKEAR

! ======================================================================
! INTEGER FUNCTION NBND2 : FIND OUT IF THERE ARE ANY DOUBLE BONDS
!                          ON IATOM -----
! ======================================================================
INTEGER FUNCTION NBND2(IATOM,ITAB)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  implicit none
  !
  integer iatom
  integer ITAB(6,*)
  !
  integer i, icur, ik1, ikount, ioccur, ip1, j
  !
  NBND2 = 0
  IKOUNT = 0
  DO I = 1,5
     IF(ITAB(I,IATOM).EQ.-1 ) GOTO 200
     IKOUNT = IKOUNT + 1
  ENDDO
200 CONTINUE
  IF(IKOUNT .LT. 2) RETURN
  !     ----- IKOUNT IS THE NUMBER OF BONDS - SEE IF
  !           ANY OF THEM OCCUR TWICE -----
  IK1 = IKOUNT - 1
  IOCCUR = 0
  DO I = 1,IK1
     IP1 = I + 1
     ICUR = ITAB(I,IATOM)
     DO J = IP1,IKOUNT
        IF(ICUR .EQ. ITAB(J,IATOM)) IOCCUR = IOCCUR + 1
        !      IF(ICUR .EQ. ITAB(J,IATOM)) THEN
        !        WRITE(6,'(A,6I5)') ' IATOM, I, J, ICUR, ITAB(J,IATOM), IOCCUR',
        !     . IATOM, I, J, ICUR, ITAB(J,IATOM), IOCCUR
        !      ENDIF
     ENDDO
     !      IF(IOCCUR .EQ. 1) GOTO 500
     !      IF(IOCCUR .GT. 1) GOTO 600
  ENDDO
  !  500 CONTINUE
  !      IF(IOCCUR .EQ. 1) NBND2 = 1
  !      WRITE(OUTU,*) ' NBND2 = ',NBND2
  !      RETURN
  !  600 CONTINUE
  NBND2=IOCCUR
  !     WRITE(6,*) ' NBND2 = ',NBND2
  !
  RETURN
END FUNCTION NBND2

! ======================================================================
! INTEGER FUNCTION NBND3 : FIND OUT IF THERE ARE ANY TRIPLE BONDS
!                          ON IATOM -----
! ======================================================================
INTEGER FUNCTION NBND3(IATOM,ITAB)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  implicit none
  !
  integer iatom
  integer ITAB(6,*)
  !
  integer i, icur, ik1, ikount, ioccur, ip1, j
  !
  NBND3 = 0
  IKOUNT = 0
  DO I = 1,5
     IF(ITAB(I,IATOM).EQ.-1 ) GOTO 200
     IKOUNT = IKOUNT + 1
  ENDDO
200 CONTINUE
  IF(IKOUNT .LT. 3) GOTO 600
  !     ----- IKOUNT IS THE NUMBER OF BONDS - SEE IF
  !           ANY OF THEM OCCUR THREE TIMES -----
  IK1 = IKOUNT - 1
  DO I = 1,IK1
     IOCCUR = 0
     IP1 = I + 1
     ICUR = ITAB(I,IATOM)
     DO J = IP1,IKOUNT
        IF(ICUR .EQ. ITAB(J,IATOM)) IOCCUR = IOCCUR + 1
     ENDDO
     IF(IOCCUR .EQ. 2) GOTO 500
     IF(IOCCUR .GT. 2) GOTO 600
  ENDDO
500 CONTINUE
  IF(IOCCUR .EQ. 2) NBND3 = 1
600 CONTINUE
  !
  RETURN
END FUNCTION NBND3

! ======================================================================
! INTEGER FUNCTION NBNDX : Return number of attached atoms
! (and bonds thereto) of type 'X'
!     ----- RETURN INOFRMATION ABOUT IATOM BASED ON THE
!           IOP CODE -----
!     IOP VALUES AND FUNCTIONS ARE -
!     IOP = 0   NUMBER OF DISTINCT ATOMS OF TYPE ITYP
!               ATTACHED TO IATOM
!     IOP = 1   SET TO 1 IF ANY ATOMS OF TYPE ITYP ARE
!               ATTACHED BY ANY BONDS
!     IOP = 2   SET TO 1 IF ANY DOUBLE BONDS OF TYPE
!               ITYP ARE PRESENT
! ======================================================================
INTEGER FUNCTION NBNDX(IATOM,NATOM, ITAB,ATNUM,ITYP,IOP)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use stream
  implicit none
  !
  integer iatom, iop, ityp, NATOM
  integer ITAB(6,*),ATNUM(*)
  !
  integer i, j, j1, jatom, k
  !
  NBNDX = 0
  IF(IOP .NE. 0) GOTO 500
  !     ----- COUNT THE NUMBER OF DISTINCT ATOMS OF TYPE
  !           ITYP ATTACHED TO IATOM -----
  DO J = 1,5
     IF( ITAB(J,IATOM).EQ.-1) GOTO 400
     JATOM=ITAB(J,IATOM)
     !D     WRITE(OUTU,'(A,4I5)') ' NBNDX: IATOM,JATOM,ATNUM',
     !D    . IATOM,JATOM,ATNUM(IATOM),ATNUM(JATOM)
     !D     IF(IZ.NE.0) WRITE(IZ,'(A,4I5)') ' NBNDX: IATOM,JATOM,ATNUM',
     !D    . IATOM,JATOM,ATNUM(IATOM),ATNUM(JATOM)
     IF(ATNUM(ITAB(J,IATOM)).NE.ITYP) GOTO 300
     !      IF(IPENT(ATNUM, ITAB(J,IATOM)) .NE. ITYP) GOTO 300
     !     ----- CORRECT FOR MULTIPLE BONDS -----
     J1 = J + 1
     IF(J1 .GT. 5) GOTO 200
     DO K = J1,5
        IF( ITAB(K,IATOM).EQ.-1) GOTO 200
        IF( ITAB(J,IATOM) .EQ.  ITAB(K,IATOM)) GOTO 300
     ENDDO
200  CONTINUE
     NBNDX = NBNDX + 1
300  CONTINUE
  ENDDO
400 CONTINUE
  GOTO 1100
  !     ----- CHECK TO SEE IF ANY ATOMS OF ATOMIC NUMBER ITYP ARE
  !           ATTACHED -----
500 CONTINUE
  IF(IOP .NE. 1) GOTO 700
  DO I = 1,5
     IF( ITAB(I,IATOM).EQ.-1) GOTO 600
     IF(ATNUM(ITAB(I,IATOM)).NE.ITYP) GOTO 600
     !      IF(IPENT(ATNUM, ITAB(I,IATOM)) .NE. ITYP) GOTO 600
     NBNDX = 1
     GOTO 1100
600  CONTINUE
  ENDDO
  GOTO 1100
  !     ----- CHECK TO SEE IF MULTIPLE BONDS OF TYPE ITYP
  !           ARE ATTACHED -----
700 CONTINUE
  IF(IOP .NE. 2) GOTO 1100
  DO I = 1,5
     IF( ITAB(I,IATOM).EQ.-1) GOTO 1000
     IF(ATNUM(ITAB(I,IATOM)).NE.ITYP) GOTO 900
     !      IF(IPENT(ATNUM, ITAB(I,IATOM)) .NE. ITYP) GOTO 900
     !     ----- FOUND ONE OF THIS TYPE - IS THERE ANOTHER -----
     J1 = I + 1
     IF(J1 .GT. 15) GOTO 900
     DO J = J1,5
        IF( ITAB(J,IATOM).EQ.-1) GOTO 900
        IF(ITAB(J,IATOM).NE.ITAB(I,IATOM)) GOTO 800
        IF(ATNUM(ITAB(J,IATOM)).NE.ITYP) GOTO 800
        !      IF( ITAB(J,IATOM) .NE.  ITAB(I,IATOM) .OR.IPENT(ATNUM, ITAB(J,IAT
        !     .OM)) .NE. ITYP) GOTO 800
        NBNDX = 1
        GOTO 1100
800     CONTINUE
     ENDDO
900  CONTINUE
  ENDDO
1000 CONTINUE
1100 CONTINUE
  !
  RETURN
END FUNCTION NBNDX

! ===================================================================
! INTEGER FUNCTION NTERMA : FIND THE NUMBER OF TERMINAL ATOMS
! OF ATOMIC NUMBER JA ATTACHED TO ATOM IATOM
! ===================================================================
INTEGER FUNCTION NTERMA(IATOM,NATOM,ITAB,ATNUM,JA)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  implicit none
  !
  integer ITAB(6,*),ATNUM(*)
  integer iatom, ja, NATOM
  !
  integer k, katom, l
  LOGICAL NEW
  !
  NTERMA=0
  DO K=1,5
     IF(ITAB(K,IATOM).LE.0) GOTO 200
     KATOM=ITAB(K,IATOM)
     NEW=.TRUE.
     !   MAKE SURE THIS IS A NEW ATOM ON ITAOM, NOT JUST AN ATOM MULTIPLY
     !   LISTED TO INDICATE A MULTIPLE BOND
     IF(K.GT.1) THEN
        DO L=1,K-1
           IF(ITAB(L,IATOM).EQ.KATOM) NEW=.FALSE.
        ENDDO
        IF(.NOT.NEW) GOTO 100
     ENDIF
     IF(ITAB(2,KATOM).LE.0) THEN
        IF(ATNUM(KATOM).NE.JA) GOTO 100
        NTERMA=NTERMA+1
        GOTO 100
     ELSE IF(ITAB(3,KATOM).LE.0.AND.ITAB(1,KATOM).EQ.ITAB(2,KATOM)) &
          THEN
        IF(ATNUM(KATOM).NE.JA) GOTO 100
        NTERMA=NTERMA+1
        GOTO 100
     ELSE IF(ITAB(4,KATOM).LE.0.AND.ITAB(1,KATOM).EQ.ITAB(2,KATOM) &
          .AND.ITAB(2,KATOM).EQ.ITAB(3,KATOM)) THEN
        IF(ATNUM(KATOM).NE.JA) GOTO 100
        NTERMA=NTERMA+1
        GOTO 100
     ENDIF
100  CONTINUE
  ENDDO
200 CONTINUE
  !D     WRITE(6,300) IATOM,NTERMA,JA
  !D     WRITE(12,300) IATOM,NTERMA,JA
300 FORMAT(' ATOM',I3,' HAS',I3,' ATTACHED TERMINAL ATOMS', &
       'OF ATOMIC NUMBER',I3)
  !
  RETURN
END FUNCTION NTERMA

! ================================================================
! SUBROUTINE RETYPE : Detect cases having quaternary pyridine N's
! ================================================================
SUBROUTINE RETYPE(NPDP)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Programmed by T. A. Halgren, Merck and Co., Inc.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use psf
  use stream
  use string
  implicit none
  integer iatom, npdp
  integer jattch, jbonds, jc, jcb, jh, jn, jnb, jo
  integer job, JPA, jpb, js, jsb, jx, kattch, kbonds, kc, kcb, kh
  integer kn, knb, ko, kob, KPA, kpb, ks, ksb, kx
  integer lattch, lbonds, lc, lcb, lh, ln, lnb, lo, lob, LPA, lpb
  integer ls, lsb, lx, nit1, nit2, nit3
  NPDP=0
  !
  DO IATOM=1,NATOM
     !
     !   BRANCH IF THIS IS NOT A CARBON ATOM
     !
     IF(ATNUM(IATOM).NE.6) GOTO 1400
     !   GET THEN NUMBERS AND TYPES OF THE ATOMS BONDED TO IATOM
     CALL BINFO(IATOM,NATOM,ATNUM,ITAB, &
          KH,KX,KATTCH,KBONDS,KC,KCB,KN,KNB,KO,KOB,KS,KSB,KPA,KPB)
     !
     !  ----  CARBON  ------------------------------------------------
     !
     IF(KATTCH.EQ.3.AND.KBONDS.EQ.4.AND.KN.GE.2) THEN
        NIT1=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
        NIT2=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,1)
        CALL BINFO(NIT1,NATOM,ATNUM,ITAB, &
             JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
        CALL BINFO(NIT2,NATOM,ATNUM,ITAB, &
             LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
        IF(KN.EQ.2) THEN
           IF(SYMB(NIT1).NE.'NPD+'.AND.SYMB(NIT2).NE.'NPD+') &
                GOTO 1400
           IF(SYMB(NIT1).EQ.'NIM+'.OR.SYMB(NIT1).EQ.'NCN+') THEN
              IFNPDP(NIT2)=.TRUE.
              NPDP=NPDP+1
           ELSE IF(SYMB(NIT2).EQ.'NIM+'.OR.SYMB(NIT2).EQ.'NCN+') &
                THEN
              IFNPDP(NIT1)=.TRUE.
              NPDP=NPDP+1
           ENDIF
        ELSE IF(KN.EQ.3) THEN
           NIT3=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,2)
           IF(SYMB(NIT1).EQ.'NPD+') THEN
              IF(SYMB(NIT2).EQ.'NIM+'.OR.SYMB(NIT2).EQ.'NCN+') &
                   THEN
                 IFNPDP(NIT1)=.TRUE.
                 NPDP=NPDP+1
              ELSE IF(SYMB(NIT3).EQ.'NIM+'.OR.SYMB(NIT3).EQ.'NCN+') &
                   THEN
                 NPDP=NPDP+1
                 IFNPDP(NIT1)=.TRUE.
              ENDIF
           ELSE IF(SYMB(NIT2).EQ.'NPD+') THEN
              IF(SYMB(NIT1).EQ.'NIM+'.OR.SYMB(NIT1).EQ.'NCN+') &
                   THEN
                 IFNPDP(NIT2)=.TRUE.
                 NPDP=NPDP+1
              ELSE IF(SYMB(NIT3).EQ.'NIM+'.OR.SYMB(NIT3).EQ.'NCN+') &
                   THEN
                 IFNPDP(NIT2)=.TRUE.
                 NPDP=NPDP+1
              ENDIF
           ELSE IF(SYMB(NIT3).EQ.'NPD+') THEN
              IF(SYMB(NIT1).EQ.'NIM+'.OR.SYMB(NIT1).EQ.'NCN+') THEN
                 IFNPDP(NIT3)=.TRUE.
                 NPDP=NPDP+1
              ELSE IF(SYMB(NIT2).EQ.'NIM+'.OR.SYMB(NIT2).EQ.'NCN+') &
                   THEN
                 IFNPDP(NIT3)=.TRUE.
                 NPDP=NPDP+1
              ENDIF
           ENDIF
        ENDIF
     ENDIF
1400 CONTINUE
  ENDDO
  !
  RETURN
END SUBROUTINE RETYPE

! ====================================================================
! SUBROUTINE RGTYPE : PERCEIVE RINGS AND RING SIZES
! FIND OCCURENCES OF AROMATIC RINGS AND ADJUST MM2 ATOM TYPES,
! IF NECESSARY
! ====================================================================
SUBROUTINE RGTYPE
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !
  !  9 Jul 96 Tom Halgren: modify ring perception to remove bonds from
  !                        list when necessary to avoid "dead ends". 
  !  20 Oct 95 Jay Banks: removed "ITAB" as an argument from calls to
  !                       arfuse and ar2fuse, which now get it from
  !                       mmff.f90.
  ! Jay Banks 25 Oct 95: add ##INCLUDE mmff_arrays.f90, change array name
  ! XYZM to XYZ.
  !
  ! Jay Banks 09 Nov 95: added argument XYZ (and removed ##INCLUDEs for
  ! both coord.f90 and mmff_arrays.fcm), and wrote above "wrapper" to copy
  ! to it, pass it, and copy from it.
  !
  ! Jay Banks 22 Nov 95: changed several WRITE(OUTU,*) statements to
  ! explicit format.
  !
  ! Jay Banks 27 Nov 95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  ! WRITE statements followed by CALL WRNDIE.
  !
  ! Jay Banks 27 Nov 95: eliminated wrapper routine, changed array XYZ to
  ! X, Y, Z, and added ##INCLUDE coord.f90 to get them from common.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use coord
  use mmffm
  use psf
  use stream
  implicit none
  !
  integer NCBUT,NCPROP
  COMMON/CYCLC/NCBUT,NCPROP
  integer MAT1(NATOM), MAT2(NATOM), NDELBND, I12DEL, IDEL, JDEL
  !
  integer i, iatom
  integer ii, imax, ir
  integer itry, j, jjx, k, l
  integer narom
  integer noat, nobd, nr, nrngs
  integer kk, kn, kold, lk, ln, m, mmax
  integer kl, klat, klm, klmat, lat, lchild, lmat, ml, mlm
  integer i1, i2, k1, k2, ll, mm, mp
  integer ifrst, k1at, k1p, k2at, k2p, last, lend
  integer mx, nminus, nplus
  integer iconn, idiff, ijbnds, il, imatch, nadd, ncand
  integer nrg
  real(chm_real)  dmax, d, factor
  !
  LOGICAL ODD,EVEN,AROMRG,del
  integer IRTEMP(NATOM),KATTCH(NATOM)
  integer, PARAMETER :: MXSIZE=200,MAXLEV=50,MAXPER=20,MXRNGS=50,MAXTRY=10
  integer MNATCH(MAXLEV,MAXPER)
  integer IPAR(MAXLEV,MAXPER)
  integer LEV(MAXLEV,MAXPER)
  integer LIST(MAXLEV,MXSIZE)
  integer NRGAT(MXRNGS)
  logical AROM(MXRNGS)
  integer NPI(MXRNGS)
  integer NPP(MXRNGS)
  integer NPB(MXRNGS)
  logical BRIDGE
  !
  NCPROP=0
  NCBUT=0
  NAROM=0
  NDELBND=0
  ! T.A. Halgren change
  !
  ! ZERO ARBOND ELEMENTS
  !
  DO M=1,MAXB
     AR_BOND(1,M)=0
     AR_BOND(2,M)=0
  ENDDO
  N_ARBOND=0
  ! end T.A. Halgren change
  IF(IRINGS.EQ.0) RETURN
  !   SET UP ITAB ARRAY WITH MULTIPLE ENTRIES FOR MULTIPLE BONDS
  CALL TABINT(2,IB,JB,BondType,NATOM,NBOND,ITAB,LOCAT)
  !   PROCESS RINGS IN TURN -- SEE WHETHER EACH IS A SIMPLE OR A COMPOUND
  !   RING
  DO IR=1,IRINGS
     !  GET NUMBER OF ATOMS AND NUMBER OF BONDS IN THIS RING
     NOAT=0
     NOBD=0
     DO I=1,NATOM
        IRTEMP(I)=0
        IF(INRING(I).EQ.IR) THEN
           NOAT=NOAT+1
           IRTEMP(NOAT)=I
        ENDIF
     ENDDO
     !      if(prnlev.ge.6) WRITE(OUTU,*) (IRTEMP(I),I=1,NOAT)
     DO I=1,NOAT
        KATTCH(I)=0
        DO J=1,NOAT
           IF(I.EQ.J) GOTO 300
           II=IRTEMP(I)
           jjx=IRTEMP(J)
           IF(CONN12(II,jjx).EQ.1) KATTCH(I)=KATTCH(I)+1
300        CONTINUE
        ENDDO
        !      if(prnlev.ge.6) WRITE(OUTU,*) 'I,II,KATTCH',I,II,KATTCH(I)
        NOBD=NOBD+KATTCH(I)
     ENDDO
     !   CORRECT FOR DOUBLE COUNTING
     NOBD=NOBD/2
     !     TEST FOR NUMBER OF RINGS IN "RING" IR
     NRNGS=1+NOBD-NOAT
     if(prnlev.ge.6) WRITE(OUTU,*) 'RINGS',NRNGS,'    ATOMS',NOAT, &
          '     BONDS',NOBD
     !D     WRITE(12,*) 'RINGS',NRNGS,'    ATOMS',NOAT,
     !D    . '     BONDS',NOBD
     IF(NOAT.LE.2) GOTO 6000
     if(prnlev.ge.2) WRITE(OUTU,250) IR,NRNGS
     !     IF(IZ.NE.0) WRITE(IZ,250) IR,NRNGS
250  FORMAT(/' RING',I3,' HAS ',I3,' SUBRINGS')
     DO NR=1,NRNGS
        !   INITIALIZE COUNTER ITRY TO KEEP TRACK OF HOW MANY ATTEMPTS
        !   ARE MADE TO DEFINE SUBRING NR -- WILL ABORT CALCULATION IF
        !   ITRY EXCEEDS LIMIT = MAXTRY
        ITRY=0
260     CONTINUE
        BRIDGE=.FALSE.
        DO I=1,NOAT
           IF(KATTCH(I).LT.3) BRIDGE=.TRUE.
        ENDDO
        if(prnlev.ge.6) WRITE(OUTU,320) NR,(IRTEMP(I),I=1,NOAT)
        !D     WRITE(12,320) NR,(IRTEMP(I),I=1,NOAT)
320     FORMAT(' SUBRING',I3,'    RING ATOMS',/ &
             (10I5))
        if(prnlev.ge.6) WRITE(OUTU,325) (KATTCH(I),I=1,NOAT)
        !D     WRITE(12,325) (KATTCH(I),I=1,NOAT)
325     FORMAT(' NUMBER OF ATTACHMENTS',/(10I5))
        DO I=1,MXSIZE
           LIST(NR,I)=0
        ENDDO
        DO I=1,MAXLEV
           DO J=1,MAXPER
              LEV(I,J)=0
              IPAR(I,J)=0
           ENDDO
        ENDDO
        !   FIND ATOMS AND BONDS IN SUBRING NR
        IMAX=0
        DO I=1,NOAT
           IF(KATTCH(I).GT.0) THEN
              !     MAKE SURE WE HAVE A VALID SEARCH-START ATOM
              IMAX=I
              GOTO 380
           ENDIF
        ENDDO
380     CONTINUE
        IF(IMAX.EQ.0) THEN
           !   NO SUITABLE START ATOM COULD BE FOUND - FATAL ERROR
           if(wrnlev.ge.2) WRITE(OUTU,390)
           !     IF(IZ.NE.0) WRITE(IZ,390)
390        FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-', &
                'TYPING PROCEDURE *****'/ &
                '       (NO SUITABLE START ATOM COULD BE FOUND)'/ &
                '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/ &
                ' ***** EXECUTION ENDING *****')
           CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
        ENDIF
        !
        DO I=1,NOAT
           IF(KATTCH(I).EQ.2) THEN
              IMAX=I
              GO TO 401
           ELSE IF(KATTCH(I).LE.0) THEN
              GO TO 400
           ELSE
              IMAX=I
           ENDIF
400        CONTINUE
        ENDDO
401     CONTINUE
        !   SEARCH WILL START AT ATOM IRTEMP(IMAX)
        II=IRTEMP(IMAX)
        !
990     CONTINUE
        IATOM=II
        if(prnlev.ge.6) WRITE(OUTU,*) ' START ATOM',II
        !D     WRITE(12,*) ' START ATOM',II
        ODD=.FALSE.
        EVEN=.FALSE.
        !   MAKE SURE WE AREN'T JUST SPINNING OUR WHEELS
        ITRY=ITRY+1
        IF(ITRY.GT.MAXTRY) THEN
           if(wrnlev.ge.2) WRITE(OUTU,995) NR,MAXTRY
           !     IF(IZ.NE.0) WRITE(IZ,995) NR,MAXTRY
995        FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-', &
                'TYPING PROCEDURE *****'/ &
                '       (UNABLE TO DEFINE SUBRING ',I3,' IN ',I3,' ATTEMPTS)'/ &
                '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/ &
                ' ***** EXECUTION ENDING *****')
           CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
        ENDIF
        DO L=1,MAXLEV
           MMAX=MAXPER
           IF(L.EQ.1) MMAX=1
           kn=0
           DO M=1,MMAX
              II=IATOM
              IF(L.GT.1) II=LEV(L-1,M)
              IF(II.EQ.0) GOTO 1100
              DO K=1,NOAT
                 IF(IRTEMP(K).LE.0) GOTO 1200
                 KK=IRTEMP(K)
                 !   FIND OUT WHETHER THIS ATOM IS NEW
                 IF(L.GT.1) THEN
                    DO ln=1,L-1
                       DO LK=1,MAXPER
                          KOLD=LEV(ln,LK)
                          IF(KOLD.EQ.KK) GOTO 1200
                          KOLD=IPAR(ln,LK)
                          IF(KOLD.EQ.KK) GOTO 1200
                       ENDDO
                    ENDDO
                 ENDIF
                 IF(CONN12(II,KK).EQ.1.AND.I12DEL(II,KK,MAT1,MAT2,NDELBND).EQ.0)  &
                      THEN
                    kn=kn+1
                    LEV(L,kn)=KK
                    IPAR(L,kn)=II
                    !        IF(L.GT.1 .and. prnlev.ge.6) WRITE(OUTU,*)
                    !    . 'KATTCH(K),M,MNATCH(L-1,M)',
                    !    . KATTCH(K),M,MNATCH(L-1,M)
                    IF(L.EQ.1) MNATCH(L,kn)=MIN(KATTCH(K),KATTCH(IMAX))
                    IF(L.GT.1) MNATCH(L,kn)=MIN(KATTCH(K),MNATCH(L-1,M))
                    if(prnlev.ge.6) WRITE(OUTU,1260) 'L,kn,LEV,MNATCH,IPAR', &
                         L,kn,LEV(L,kn),MNATCH(L,kn),IPAR(L,kn)
1260                FORMAT(A,5I5)
                    !D       WRITE(12,*) 'L,kn,LEV,MNATCH,IPAR',L,kn,LEV(L,kn),MNATCH(L,kn),
                    !D    .IPAR(L,kn)
                 ENDIF
1200             CONTINUE
              ENDDO
1100          CONTINUE
           ENDDO
           !   MAKE SURE THE TRAIL HASN'T GONE COLD - I.E., MAKE SURE THAT AT LEAST
           !   ONE ATOM WAS ADDED (kn.GE.1) AT THIS PERCEPTION LEVEL
           IF(kn.EQ.0) THEN
              if(wrnlev.ge.2) WRITE(OUTU,1340) NR,L
              !     IF(IZ.NE.0) WRITE(IZ,1340) NR,L
1340          FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-', &
                   'TYPING PROCEDURE *****'/ &
                   '       (THE CHAIN TRACING FOR SUBRING ',I3,' HAS RUN INTO', &
                   ' A DEAD END AT STEP ',I3,')'/ &
                   '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/ &
                   ' ***** EXECUTION ENDING *****')
              CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
           ENDIF
           !   MAKE SURE THAT EACH PARENT ATOM AT LEVEL L-1 HAS AT LEAST ONE CHILD
           IF(L.GE.2) THEN
              DO MLM=1,MAXPER
                 LMAT=LEV(L-1,MLM)
                 IF(LMAT.EQ.0) GOTO 1300
                 DO ML=1,MAXPER
                    LAT=LEV(L,ML)
                    IF(LAT.EQ.0) GOTO 1350
                    IF(CONN12(LAT,LMAT).EQ.1.AND.I12DEL(LAT,LMAT,MAT1,MAT2,NDELBND) &
                         .EQ.0) GOTO 1300
1350                CONTINUE
                 ENDDO
                 !   IF HERE, NO CHILD FOR ATOM LMAT AT LEVEL L-1 WAS FOUND
                 if(prnlev.ge.6) WRITE(OUTU,*) ' ATOM ',LMAT,' AT LEVEL',L-1, &
                      ' HAS NO CHILD'
                 !D       WRITE(12,*) ' ATOM ',LMAT,' AT LEVEL',L-1,
                 !D    .  ' HAS NO CHILD'
                 DO K=1,NOAT
                    IF(IRTEMP(K).NE.LMAT) GOTO 1360
                    !   MARK ATOM K FOR DELETION IF IT HAS ONLY ONE ATTACHMENT
                    IF(KATTCH(K).GE.2) GOTO 1300
                    IRTEMP(K)=0
                    KATTCH(K)=0
                    DO KL=1,NOAT
                       IF(IRTEMP(KL).EQ.0) GOTO 1370
                       KLAT=IRTEMP(KL)
                       IF(CONN12(KLAT,LMAT).EQ.1.AND.I12DEL(KLAT,LMAT,MAT1,MAT2, &
                            NDELBND).EQ.0) THEN
                          !         REMOVE ATTACHMENT BETWEEN LMAT AND KLAT
                          KATTCH(KL)=KATTCH(KL)-1
                          KATTCH(KL)=MAX(KATTCH(KL),0)
                          IF(KATTCH(KL).LE.1) THEN
                             !          ALSO DELETE KLAT, AS IT CAN NO LONGER BE IN A
                             !          REMAINING RING
                             IRTEMP(KL)=0
                             KATTCH(KL)=0
                             !          NOW ASK ABOUT KLAT'S PARENT, WHO MAY ALSO NO LONGER
                             !          QUALIFY AS A RING ATOM
                             DO KLM=1,NOAT
                                IF(IRTEMP(KLM).EQ.0) GOTO 1375
                                KLMAT=IRTEMP(KLM)
                                IF(CONN12(KLMAT,KLAT).EQ.1.AND.I12DEL(KLMAT,KLAT,MAT1, &
                                     MAT2,NDELBND).EQ.0) THEN
                                   !          REMOVE ATTACHMENT BETWEEN KLAT AND KLMAT
                                   KATTCH(KLM)=MAX(0,KATTCH(KLM)-1)
                                   IF(KATTCH(KLM).LE.1) THEN
                                      IRTEMP(KLM)=0
                                      KATTCH(KLM)=0
                                   ENDIF
                                ENDIF
1375                            CONTINUE
                             ENDDO
                          ENDIF
                       ENDIF
1370                   CONTINUE
                    ENDDO
1360                CONTINUE
                 ENDDO
                 !     NOW GO BACK AND TRY AGAIN TO DEFINE THIS SUBRING
                 GOTO 260
1300             CONTINUE
              ENDDO
           ELSE IF(L.EQ.1) THEN
              !   MAKE SURE THE CHOSEN START ATOM HAS AT LEAST TWO CHILDREN
              LCHILD=0
              DO M=1,MAXPER
                 IF(LEV(1,M).NE.0) LCHILD=LCHILD+1
              ENDDO
              IF(LCHILD.LT.2) THEN
                 if(prnlev.ge.6) WRITE(OUTU,*) ' START ATOM',IMAX, &
                      ' HAS LESS THAN 2 CHILDREN'
                 !D         WRITE(12,*) ' START ATOM',IMAX,' HAS LESS THAN '
                 !D    .     ,'2 CHILDREN'
                 IRTEMP(IMAX)=0
                 KATTCH(IMAX)=0
                 DO K=1,NOAT
                    KK=IRTEMP(K)
                    IF(KK.EQ.0) GOTO 1400
                    IF(CONN12(IATOM,KK).NE.1) GOTO 1400
                    KATTCH(K)=KATTCH(K)-1
                    KATTCH(K)=MAX(KATTCH(K),0)
                    IF(KATTCH(K).LE.1) THEN
                       IRTEMP(K)=0
                       KATTCH(K)=0
                    ENDIF
1400                CONTINUE
                 ENDDO
                 !   NOW GO BACK AND TRY AGAIN TO DEFINE THIS SUBRING
                 GOTO 260
              ENDIF
           ENDIF
           !   ADDITIONS TO LEVEL L HAVE NOW ALL BEEN MADE.
           !   EVALUATE TO SEE IF A RING CAN BE COMPLETED
           !   FIRST SEE IF TWO JUST-ADDED ATOMS ARE IDENTICAL
           if(prnlev.ge.6) WRITE(OUTU,*) ' kn =', kn,'    IDENTICAL?'
           !D     WRITE(12,*) ' kn =', kn,'    IDENTICAL?'
           DO K1=2,kn
              DO K2=1,K1-1
                 if(prnlev.ge.6) WRITE(OUTU,1510) 'L,K1,K2,LEV(L,K1),LEV(L,K2)', &
                      L,K1,K2,LEV(L,K1),LEV(L,K2)
1510             FORMAT(1X,A,5I5)
                 !   MAKE SURE THIS RING WOULD HAVE A NON-JUNCTION SECTION
                 IF(BRIDGE.AND.MNATCH(L,K1).GE.3.AND.MNATCH(L,K2).GE.3) GOTO 1520
                 IF(LEV(L,K1).EQ.LEV(L,K2).AND.IPAR(L,K1).NE.IPAR(L,K2)) THEN
                    !       K1 = K2.  AN EVEN-NUMBERED RING HAS BEEN CLOSED.
                    !       SET SOME INDICES AND BRANCH TO GET-RING-LIST SECTION
                    LL=L
                    MP=LL
                    MM=LL+2
                    EVEN=.TRUE.
                    ODD=.FALSE.
                    GOTO 2000
                 ENDIF
1520             CONTINUE
              ENDDO
           ENDDO
           !   NEXT SEE IF THE JUST-ADDED ATOMS BONDED ARE TO EACH OTHER
           if(prnlev.ge.6) WRITE(OUTU,*) ' BONDED?'
           !D     WRITE(12,*) ' BONDED?'
           DO K1=2,kn
              DO K2=1,K1-1
                 I1=LEV(L,K1)
                 I2=LEV(L,K2)
                 !   MAKE SURE THIS RING WOULD HAVE A NON-JUNCTION SECTION
                 IF(BRIDGE.AND.MNATCH(L,K1).GE.3.AND.MNATCH(L,K2).GE.3) GOTO 1800
                 IF(CONN12(I1,I2).EQ.1.AND.I12DEL(I1,I2,MAT1,MAT2,NDELBND).EQ.0)  &
                      THEN
                    !      YES, THEY ARE; AN ODD-NUMBERED RING HAS BEEN DETECTED.
                    !      SET SOME INDICES AND BRANCH
                    LL=L
                    MP=LL+1
                    MM=LL+2
                    ODD=.TRUE.
                    EVEN=.FALSE.
                    GOTO 2000
                 ENDIF
1800             CONTINUE
              ENDDO
           ENDDO
           !   NEITHER TERMINATION CONDITION HAS OCCURRED.  EXTEND SEARCH
        ENDDO
        !   MAX PERCEPTION LIMIT REACHED - RING TOO LARGE -- FATAL ERROR
        if(wrnlev.ge.2) WRITE(OUTU,1900)
        !     IF(IZ.NE.0) WRITE(IZ,1900)
1900    FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-', &
             'TYPING PROCEDURE *****'/ &
             '       (MAXIMUM RING-SIZE LIMIT REACHED)'/ &
             '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/ &
             ' ***** EXECUTION ENDING *****')
        CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
        !   GET ORDERED LIST OF ATOMS FOR THIS SUB RING
2000    CONTINUE
        if(prnlev.ge.6) WRITE(OUTU,*) ' RING FOUND AT LEVEL',LL
        !D     WRITE(12,*) ' RING FOUND AT LEVEL',LL
        IF(EVEN) NRGAT(NR)=2*LL
        IF(ODD) NRGAT(NR)=2*LL+1
        if(prnlev.ge.6) WRITE(OUTU,*) ' RING SIZE =',NRGAT(NR)
        !D      WRITE(12,*) ' RING SIZE =',NRGAT(NR)
        !   PLACE ATOMS STARTING FROM MIDDLE OF LIST
        !   FIRST ONES ARE THOSE IN SLOTS K1 AND K2
        DO ML=LL,1,-1
           !   IDENTIFY THE NEW ATOMS
           NPLUS=LEV(ML,K1)
           NMINUS=LEV(ML,K2)
           !   CHECK FOR UNIQUENESS TO MAKE SURE THE CLOSURE APPLIES TO THE
           !   WHOLE RING BEING EXPLORED, AND NOT TO A SUBRING
           IF(NMINUS.EQ.NPLUS.AND.ML.LT.LL) THEN
              if(prnlev.ge.6) WRITE(OUTU,2125) NMINUS
              !D        WRITE(12,2125) CNAME(NMINUS),NMIMUS
2125          FORMAT(' *** SUBRING FOUND CONTAINING ATOM ', &
                   I6,' ***')
              !   TRY AGAIN USING  SUB-RING ATOM LL+1 AS THE STARTING POINT
              II=LIST(NR,LL+1)
              !   FIRST ZERO PERCEPTION ARRAYS
              DO I=1,MXSIZE
                 LIST(NR,I)=0
              ENDDO
              DO I=1,MAXLEV
                 DO J=1,MAXPER
                    LEV(I,J)=0
                    IPAR(I,J)=0
                 ENDDO
              ENDDO
              !   RECORD WHICH SLOT II IS IN
              DO I=1,NOAT
                 IF(IRTEMP(I).EQ.II) THEN
                    IMAX=I
                    GOTO 990
                 ENDIF
              ENDDO
              !   IF HERE, ERROR.  WRITE MSG
              if(wrnlev.ge.2) WRITE(OUTU,2145) II
              !     IF(IZ.NE.0) WRITE(IZ,2145) II
2145          FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-', &
                   'TYPING PROCEDURE *****'/ &
                   '       (NEW START ATOM ',I6,' NOT IDENTIFIED)'/ &
                   '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/ &
                   ' ***** EXECUTION ENDING *****')
              CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
           ENDIF
           DO MX=MM,MP
              IF(NMINUS.EQ.LIST(NR,MP).OR.NMINUS.EQ.LIST(NR,MM))  THEN
                 if(prnlev.ge.6) WRITE(OUTU,2160) NMINUS
                 !D        WRITE(12,2160) CNAME(NMINUS),NMIMUS
2160             FORMAT(' *** ATOM ',I6, ' WAS ALREADY ON THE LIST ***')
                 !   TRY AGAIN USING  SUB-RING ATOM LL+1 AS THE STARTING POINT
                 II=LIST(NR,LL+1)
                 !   FIRST ZERO PERCEPTION ARRAYS
                 DO  I=1,MXSIZE
                    LIST(NR,I)=0
                 ENDDO
                 DO I=1,MAXLEV
                    DO J=1,MAXPER
                       LEV(I,J)=0
                       IPAR(I,J)=0
                    ENDDO
                 ENDDO
                 !   RECORD WHICH SLOT II IS IN
                 DO I=1,NOAT
                    IF(IRTEMP(I).EQ.II) THEN
                       IMAX=I
                       GOTO 990
                    ENDIF
                 ENDDO
                 !   IF HERE, ERROR.  WRITE MSG
                 if(wrnlev.ge.2) WRITE(OUTU,2145) II
                 !     IF(IZ.NE.0) WRITE(IZ,2145) II
                 ! 2145    FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-',
                 !     .   'TYPING PROCEDURE *****'/
                 !     .   '       (NEW START ATOM ',I6,' NOT IDENTIFIED)'/
                 !     .  '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/
                 !     .  ' ***** EXECUTION ENDING *****')
                 CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
              ENDIF
              IF(NPLUS.EQ.LIST(NR,MP).OR.NPLUS.EQ.LIST(NR,MM))  THEN
                 if(prnlev.ge.6) WRITE(OUTU,2160) NPLUS
                 !D        WRITE(12,2160) CNAME(NPLUS),NMIMUS
                 !   TRY AGAIN USING  SUB-RING ATOM LL+1 AS THE STARTING POINT
                 II=LIST(NR,LL+1)
                 !  FIRST ZERO PERCEPTION ARRAYS
                 DO I=1,MXSIZE
                    LIST(NR,I)=0
                 ENDDO
                 DO I=1,MAXLEV
                    DO J=1,MAXPER
                       LEV(I,J)=0
                       IPAR(I,J)=0
                    ENDDO
                 ENDDO
                 !   RECORD WHICH SLOT II IS IN
                 DO I=1,NOAT
                    IF(IRTEMP(I).EQ.II) THEN
                       IMAX=I
                       GOTO 990
                    ENDIF
                 ENDDO
                 !   IF HERE, ERROR.  WRITE MSG
                 if(wrnlev.ge.2) WRITE(OUTU,2145) II
                 !     IF(IZ.NE.0) WRITE(IZ,2145) II
                 ! 2145    FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-',
                 !     .   'TYPING PROCEDURE *****'/
                 !     .   '       (NEW START ATOM ',I6,' NOT IDENTIFIED)'/
                 !     .  '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/
                 !     .  ' ***** EXECUTION ENDING *****')
                 CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
              ENDIF
           ENDDO
           !   PLACE ON LIST
           MP=MP+1
           MM=MM-1
           LIST(NR,MP)=NPLUS
           LIST(NR,MM)=NMINUS
           if(prnlev.ge.6) WRITE(OUTU,2120) (LIST(NR,I),I=1,NRGAT(NR))
           !D     WRITE(12,2120) (LIST(NR,I),I=1,NRGAT(NR))
2120       FORMAT(10I5)
           !   NOTE FOR ML=LL THAT THESE PLACEMENTS ARE REDUNDANT (MP=MM) IF AN
           !   EVEN-MEMBERED RING HAS BEEN FOUND
           !
           !   NOW IDENTIFY THE PARENTS OF THE JUST-PLACED ATOMS
           K1P=IPAR(ML,K1)
           K2P=IPAR(ML,K2)
           IF(ML.GT.1) THEN
              !        FIND THE "SLOTS" THE PARENTS WERE IN
              DO K=1,MAXPER
                 IF(LEV(ML-1,K).EQ.K1P) K1=K
                 IF(LEV(ML-1,K).EQ.K2P) K2=K
              ENDDO
           ELSE
              !   HAVE WORKED BACK TO THE ROOT ATOM (K1P=K2P=IATOM)
              LIST(NR,1)=K1P
           ENDIF
        ENDDO
        if(prnlev.ge.6) WRITE(OUTU,2110) NR,NRGAT(NR), &
             (LIST(NR,I),I=1,NRGAT(NR))
        !D     WRITE(12,2110) NR,NRGAT(NR),(LIST(NR,I),I=1,NRGAT(NR))
2110    FORMAT(' NUMBER OF SUB-RING ',I2, ' ATOMS:',I3/' MEMBERS --'/ &
             (10I5))
        !   IF THIS RING HAS AT LEAST 4 MEMBERS,
        !   MAKE ADDITIONAL CHECK FOR UNIQUENESS BY SEEING WHETHER ANY TWO
        !   ATOMS OTHER THAT THE FIRST AND LAST ARE BONDED TO ONE ANOTHER
        IF(NRGAT(NR).LE.3) GOTO 2325
        DO I1=1,NRGAT(NR)-2
           K1=LIST(NR,I1)
           II=NRGAT(NR)
           IF(I1.EQ.1) II=NRGAT(NR)-1
           DO I2=I1+2,II
              K2=LIST(NR,I2)
              IF(CONN12(K1,K2).EQ.1.AND.I12DEL(K1,K2,MAT1,MAT2,NDELBND).EQ.0)  &
                   THEN
                 !   IF HERE, A SMALLER RING IS CONTAINED IN THE RING JUST FOUND
                 if(prnlev.ge.6) WRITE(OUTU,2290) K1,K2
                 !D        WRITE(IZ,2290) K1,K2
2290             FORMAT(' *** A SMALLER RING CONTAINING ATOMS ',2I6, &
                      '  IS PRESENT ***')
                 !
                 !       FIND SLOTS FOR K1 AND K2
                 DO I=1,NOAT
                    IF(IRTEMP(I).EQ.K1) K1AT=I
                    IF(IRTEMP(I).EQ.K2) K2AT=I
                 ENDDO
                 !   CHOOSE THE ATOM WITH THE MAXIMUM NUMBER OF ATTACHMENTS AS
                 !   THE NEW START ATOM, THEN REPEAT THE RING FINDING
                 II=K1
                 IF(KATTCH(K2AT).GT.KATTCH(K1AT)) II=K2
                 !     FIRST ZERO THE PERCEPTION ARRAYS
                 DO I=1,MXSIZE
                    LIST(NR,I)=0
                 ENDDO
                 DO I=1,MAXLEV
                    DO J=1,MAXPER
                       LEV(I,J)=0
                       IPAR(I,J)=0
                    ENDDO
                 ENDDO
                 !   RECORD WHICH SLOT II IS IN
                 IMAX=K1AT
                 IF(II.EQ.K2) IMAX=K2AT
                 GOTO 990
              ENDIF
           ENDDO
        ENDDO
2325    CONTINUE
        !   DELETE THIS SUBRING BY "REMOVING" A SECTION OF NON-RING-
        !   JUNCTION ATOMS
        DEL=.FALSE.
        LAST=0
        ifrst=0
        LEND=0
        DO K=1,NRGAT(NR)
           kn=K
           DO I=1,NOAT
              KK=I
              IF(IRTEMP(I).EQ.LIST(NR,K)) GOTO 2360
           ENDDO
           !   NO MATCH COULD BE BE FOUND FOR ATOM LIST(NR,K) - FATAL ERROR
           if(wrnlev.ge.2) WRITE(OUTU,2355) QNAME(LIST(NR,K))
           !     IF(IZ.NE.0) WRITE(IZ,2355) QNAME(LIST(NR,K))
2355       FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-', &
                'TYPING PROCEDURE *****'/ &
                '       (NO MATCH FOR ATOM ',A,' COULD BE FOUND)'/ &
                '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/ &
                ' ***** EXECUTION ENDING *****')
           CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
2360       CONTINUE
           IF(kn.EQ.1) ifrst=KK
           IF(KATTCH(KK).LE.2) THEN
              !   TEST WHETHER THIS IS THE FIRST ATOM IN THE NON-JUNCTION STRIP
              IF(.NOT.DEL) THEN
                 !   REMOVE AN ATTACHMENT TO THE PREVIOUS ATOM ("LAST")
                 IF(LAST.NE.0) KATTCH(LAST)=KATTCH(LAST)-1
                 IF(LAST.NE.0) LEND=LEND+1
              ENDIF
              !       THIS ATOM IS PART OF A NON-JUNCTION SECTION: DELETE IT
              DEL=.TRUE.
              IRTEMP(KK)=0
              KATTCH(KK)=0
              !   IF KK IS THE LAST ATOM ON THE LIST, WE MUST STILL SEVER THE
              !   CONNECTION TO THE FIRST ATOM ("ifrst")
              IF(kn.EQ.NRGAT(NR)) THEN
                 KATTCH(ifrst)=KATTCH(ifrst)-1
                 LEND=LEND+1
                 !        NOW WE ARE DONE
                 GOTO 2600
              ENDIF
           ELSE
              !   TEST ON "DEL" TO SEE WHETHER WE HAVE GOTTEN TO A NON-JUNCTION
              !   STRIP AS YET; TAKE NO ACTION IF WE HAVE NOT.  JUST RECORD THE
              !   IDENTITY OF THIS ATOM AS "LAST"
              LAST=KK
              IF(.NOT.DEL) GOTO 2340
              !       HAVE BEEN DELETING A STRIP OF NON-JUNCTION ATOMS (SINCE
              !       DEL=.TRUE.) AND HAVE NOW RUN INTO A JUNCTION ATOM:
              !       REDUCE ITS NUMBER OF ATTACHEMNTS
              !       BY ONE.  THEN GO TO OTHER END OF LIST AND WORK BACKWARDS
              KATTCH(KK)=KATTCH(KK)-1
              LEND=LEND+1
              GOTO 2400
           ENDIF
2340       CONTINUE
        ENDDO
2400    CONTINUE
        IF(LEND.EQ.0.OR.LEND.GT.2) THEN
           !  POSSIBLE ERROR - NO ENDS OF THE NON-JUNCTION STRIP HAVE BEEN DISCONNECTED
           IF(BRIDGE) THEN
              if(wrnlev.ge.2) WRITE(OUTU,2410) 
              !            IF(IZ.NE.0) WRITE(IZ,2410) 
2410          FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-', &
                   'TYPING PROCEDURE *****'/ &
                   '       (NO NON-JUNCTION ATOMS WERE REMOVED)'/ &
                   '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/ &
                   ' ***** EXECUTION ENDING *****')
              CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
           ELSE
              !   JUST MARK ONE BOND FOR DELETION, AND CONTINUE
              IDEL=LIST(NR,1)
              JDEL=LIST(NR,2)
              NDELBND=NDELBND+1
              MAT1(NDELBND)=IDEL
              MAT2(NDELBND)=JDEL
              DO I=1,NOAT
                 IF(IRTEMP(I).EQ.LIST(NR,1)) KATTCH(I)=KATTCH(I)-1
                 IF(IRTEMP(I).EQ.LIST(NR,2)) KATTCH(I)=KATTCH(I)-1
              ENDDO
              GO TO 2600
           ENDIF
        ENDIF
        IF(LEND.EQ.2) THEN
           !
           !   BOTH ENDS HAVE BEEN SEVERED, SO WE ARE THROUGH
           GOTO 2600
        ENDIF
        IF(kn.GE.NRGAT(NR)) THEN
           !   WE HAVE REACHED THE OTHER END OF THE RING LIST, SO WE ARE THROUGH
           GOTO 2600
        ENDIF
        DO K=NRGAT(NR),kn+1,-1
           DO I=1,NOAT
              KK=I
              IF(IRTEMP(I).EQ.LIST(NR,K)) GOTO 2560
           ENDDO
           !   NO MATCH COULD BE BE FOUND FOR ATOM LIST(NR,K) - FATAL ERROR
           if(wrnlev.ge.2) WRITE(OUTU,2355) QNAME(LIST(NR,K))
           !     IF(IZ.NE.0) WRITE(IZ,2355) QNAME(LIST(NR,K))
           ! 2355   FORMAT(//' ***** A FATAL ERROR HAS OCCURED IN THE RING-',
           !     .  'TYPING PROCEDURE *****'/
           !     .  '       (NO MATCH FOR ATOM ',A,' COULD BE FOUND)'/
           !     .  '       PLEASE REPORT PROBLEM TO TOM HALGREN (908-594-7735)'/
           !     .  ' ***** EXECUTION ENDING *****')
           CALL WRNDIE(-5,'<rgtype>','ERROR MSG')
2560       CONTINUE
           IF(KATTCH(KK).LE.2) THEN
              IRTEMP(KK)=0
              KATTCH(KK)=0
           ELSE
              KATTCH(KK)=KATTCH(KK)-1
              GOTO 2600
           ENDIF
        ENDDO
2600    CONTINUE
     ENDDO
     if(prnlev.ge.6) WRITE(OUTU,5100) (IRTEMP(I),I=1,NOAT)
     !     WRITE(12,5100) (IRTEMP(I),I=1,NOAT)
5100 FORMAT(' REMAINING RING ATOMS:'/(10I5))
     if(prnlev.ge.6) WRITE(OUTU,5110) (KATTCH(I),I=1,NOAT)
5110 FORMAT(' ATTACHMENTS:'/(10I5))
     !   MAKE USE OF THE INFORMATION ON RING CONSTITUTION
     !   BEGIN BY RECORDING CYCLOPROPYL RINGS AND ADJUSTING ATOM TYPES
     DO NR=1,NRNGS
        IF(NRGAT(NR).EQ.3) THEN
           !  INCREMENT COUNT FOR NUMBER OF CYCLOPROPANE RINGS
           NCPROP=NCPROP+1
           if(prnlev.ge.2) WRITE(OUTU,'(A,I4,A)') &
                ' SUBRING',NR,' IS A 3-MEMBERED RING'
           !       IF(IZ.NE.0) WRITE(IZ,*) ' SUBRING',NR,' IS A 3-MEMBERED RING'
           DO IL=1,3
              II=LIST(NR,IL)
              !   RECLASSIFY SP3 CARBON IN 3-MEMBERED RING AS CYCLOPROPYL-C
              IF(SYMB(II)(1:2).EQ.'CR') THEN
                 SYMB(II)='CR3R'
                 !  TRANSLATE THE SYMBOLIC INTO THE NUMERIC ATOM TYPE
                 CALL TRSYMB(II)
              ENDIF
           ENDDO
        ENDIF
        !   RECORD ANY FOUR-MEMBERED RINGS
        IF(NRGAT(NR).EQ.4) THEN
           NCBUT=NCBUT+1
           if(prnlev.ge.2) &
                WRITE(OUTU,'(A,I4,A)') ' SUBRING',NR,' IS A 4-MEMBERED RING'
           !       IF(IZ.NE.0) WRITE(IZ,*) ' SUBRING',NR,' IS A 4-MEMBERED RING'
           DO IL=1,4
              II=LIST(NR,IL)
              !   RECLASSIFY SP3 CARBON IN 4-MEMBERED RING AS CYCLOBUTYL-C
              IF(SYMB(II).EQ.'CR3R') GOTO 7150
              IF(SYMB(II).EQ.'CR') THEN
                 SYMB(II)='CR4R'
                 !  TRANSLATE THE SYMBOLIC INTO THE NUMERIC ATOM TYPE
                 CALL TRSYMB(II)
              ELSE IF(SYMB(II).EQ.'C=C') THEN
                 SYMB(II)='CE4R'
                 !  TRANSLATE THE SYMBOLIC INTO THE NUMERIC ATOM TYPE
                 CALL TRSYMB(II)
              ENDIF
7150          CONTINUE
           ENDDO
        ENDIF
     ENDDO
     !   NOW PERCIEVE AND ASSIGN MMFF ATOM TYPES FOR ANY AROMATIC RINGS.
     !   THIS  MAY REQUIRE SEVERAL PASSES THROUGH THE RING LIST
     !   BEGIN BY ASSIGNING THE NUMBER OF PI ELECTRONS CONTAINED WITHIN
     !   ANY FIVE- OR SIX-MEMBERED (SUB)RINGS
     NCAND=0
     DO NR=1,NRNGS
        AROM(NR)=.FALSE.
        NPI(NR)=0
        NPB(NR)=0
        NPP(NR)=0
        if(prnlev.ge.6) WRITE(OUTU,*) ' SUBRING',NR,NRGAT(NR),' ATOMS'
        IF(NRGAT(NR).EQ.5.OR.NRGAT(NR).EQ.6) THEN
           NRG=NRGAT(NR)
           NCAND=NCAND+1
           !         WRITE(12,*) ' SUBRING',NR,NRG,' ATOMS'
           !   FIND NUMBER OF PI BONDS WITHIN RING IN THE KEKULE STRUCTURE BEING US
           DO I=1,NRG
              J=I+1
              IF(I.EQ.NRG) J=1
              II=LIST(NR,I)
              jjx=LIST(NR,J)
              IJBNDS=0
              if(prnlev.ge.6) WRITE(OUTU,*)  '  II,ITAB',II, &
                   (ITAB(K,II),K=1,4)
              !D        IF(IZ.NE.0) WRITE(OUTU,*)  '  II,ITAB',II,(ITAB(K,II),K=1,4)
              DO K=1,4
                 IF(ITAB(K,II).EQ.jjx) IJBNDS=IJBNDS+1
              ENDDO
              IF(IJBNDS.GE.2) THEN
                 NPB(NR)=NPB(NR)+1
                 if(prnlev.ge.6) WRITE(OUTU,*) '     PI BOND',II,jjx
                 !D          IF(IZ.NE.0) WRITE(IZ,*) '     PI BOND',II,jjx
              ENDIF
              !   ADD TWO ELECTRONS FOR EACH "PI" UNSHARED PAIR ON A RING N, O, OR S
              IF(ATNUM(II).EQ.7) THEN
                 IMATCH=0
                 DO K=2,4
                    IF(ITAB(K,II).LT.1) GOTO 8300
                    ICONN=K
                    DO kn=1,K-1
                       IF(ITAB(K,II).EQ.ITAB(kn,II)) IMATCH=IMATCH+1
                    ENDDO
8300                CONTINUE
                 ENDDO
                 IDIFF=ICONN-IMATCH
                 IF(IDIFF.EQ.3.AND.ICONN.EQ.3) THEN
                    !           THIS N BONDED TO 3 DIFFERENT NEIGBORS, SO HAS A PI PAIR
                    !           IF IT ALSO MAKES ONLY THREE (NOT FOUR) BONDS
                    NPP(NR)=NPP(NR)+1
                    if(prnlev.ge.2) WRITE(OUTU,'(A,I6)') &
                         '     PI PAIR ON SP2-N',II
                    !     IF(IZ.NE.0) WRITE(IZ,*) '     PI PAIR ON SP2-N',II
                 ELSE IF(IDIFF.EQ.2.AND.ICONN.EQ.2) THEN
                    !           THIS N BONDED TO ONLY TWO NEIGHBORS WITH SINGLE BONDS,
                    !           SO HAS A PI (AND A SIGMA) PAIR
                    NPP(NR)=NPP(NR)+1
                    if(prnlev.ge.2) WRITE(OUTU,'(A,I6)') &
                         '     PI PAIR ON DICOORD N',II
                    !            IF(IZ.NE.0) WRITE(IZ,*) '     PI PAIR ON DICOORD N',II
                 ENDIF
              ENDIF
              IF(ATNUM(II).EQ.8.OR.(ATNUM(II).EQ.16.AND. &
                   ITAB(3,II).LE.0)) THEN
                 !   ANY OXYGEN OR DIVALENT SULFUR IN  A RING MUST HAVE A PI PAIR
                 !
                 NPP(NR)=NPP(NR)+1
                 if(prnlev.ge.2) WRITE(OUTU,'(A,I6)') &
                      '      PI PAIR ON O OR S',II
                 !         IF(IZ.NE.0) WRITE(IZ,*) '      PI PAIR ON O OR S',II
              ENDIF
           ENDDO
           NPI(NR)=2*(NPB(NR)+NPP(NR))
           IF(NPI(NR).GT.0 .and. prnlev.ge.2) WRITE(OUTU,8090) NR,NPI(NR)
           !     IF(IZ.NE.0) WRITE(IZ,8090) NR,NPI(NR)
8090       FORMAT(' SUBRING',I3,' has',I3,' PI electrons')
        ENDIF
     ENDDO
     !   NOW TEST ANY 6-PI RINGS FOR AROMATICITY
     NAROM=0
     DO NR=1,NRNGS
        IF(NPI(NR).EQ.6) THEN
           IF(NRGAT(NR).EQ.6.AND.NPB(NR).EQ.3) AROM(NR)=.TRUE.
           IF(NRGAT(NR).EQ.5.AND.NPB(NR).EQ.2.AND.NPP(NR).EQ.1) &
                AROM(NR)=.TRUE.
           IF(AROM(NR)) THEN
              CALL ARRING(LIST,MAXLEV,MXSIZE,NR,NRGAT(NR))
              NAROM=NAROM+1
              if(prnlev.ge.2) &
                   WRITE(OUTU,'(A,I3,A)') ' SUBRING',NR,' IS AROMATIC'
              !           IF(IZ.NE.0)
              !    .      WRITE(IZ,'(A,I3,A)') ' SUBRING',NR,' IS AROMATIC'
           ENDIF
        ENDIF
     ENDDO
     IF(NAROM.EQ.0) GOTO 6000
     !   FINALLY, TYPE APPROPRIATE 4-PI RINGS AS AROMATIC IF FUSED TO AN
     !   AROMATIC RING (THUS ALSO BECOMING AROMATIC)
     !   ALSO TYPE AS AROMATIC 2-PI 5- AND 6-MEMMBERED RINGS IF PROPERLY
     !   FUSED TO TWO ALREADY-AROMATIC RINGS
     !   MUST DO THIS ITERATIVELY UNTIL NO MORE AROMATIC SUBRINGS
     !   ARE DECLARED
8900 CONTINUE
     NADD=0
     DO NR=1,NRNGS
        IF(AROM(NR).OR.NRGAT(NR).LT.5.OR.NRGAT(NR).GT.6) GOTO 9000
        IF(NPI(NR).NE.4) GOTO 9000
        IF(NRGAT(NR).EQ.6.AND.NPB(NR).NE.2) GOTO 9000
        IF(NRGAT(NR).EQ.5.AND.NPB(NR).NE.1) GOTO 9000
        !   THIS RING HAS 4 PI ELECTRONS.  IF THE RING-SIZE IS 6,
        !   IT HAS 2 PI BONDS AND NO PI UNSHARED PAIRS.  IF THE
        !   RING SIZE IS 5, IT HAS ONE PI BOND AND ONE PI PAIR.
        !   THE NECESSARY PRECONDITIONS HAVE THUS BEEN MET.
        !   DETERMINE WHETHER IT REALLY QUALIFIES.
        CALL ARFUSE(NR,NRNGS,LIST,MAXLEV,MXSIZE,NRGAT,AROM,AROMRG)
        IF(AROMRG) THEN
           NAROM=NAROM+1
           NADD=NADD+1
           if(prnlev.ge.2) WRITE(OUTU,'(A,I3,A)') &
                ' SUBRING',NR,' ALSO RECOGNIZED AS AROMATIC'
           !     IF(IZ.NE.0) WRITE(IZ,'(A,I3,A)')
           !    .       ' SUBRING',NR,' ALSO RECOGNIZED AS AROMATIC'
           CALL ARRING(LIST,MAXLEV,MXSIZE,NR,NRGAT(NR))
           AROM(NR)=.TRUE.
        ENDIF
9000    CONTINUE
     ENDDO
     DO NR=1,NRNGS
        IF(AROM(NR).OR.NRGAT(NR).LT.5.OR.NRGAT(NR).GT.6) GO TO 9100
        IF(NRGAT(NR).EQ.6.AND.NPB(NR).NE.1) GO TO 9100
        IF(NPI(NR).NE.2) GOTO 9100
        IF(NRGAT(NR).EQ.5.AND.NPB(NR).NE.0) GOTO 9100
        !   THIS 5-MEMBERED RING HAS 2 PI ELECTRONS;  IT HAS NO PI BONDS
        !   AND ONE PI PAIR, OR THIS 6-MEMBERED RING HAS ONE PI BOND
        !   THE NECESSARY PRECONDITIONS HAVE THUS BEEN MET.
        !   DETERMINE WHETHER IT REALLY QUALIFIES.
        CALL AR2FUSE(NR,NRNGS,LIST,MAXLEV,MXSIZE,NRGAT,AROM,AROMRG)
        IF(AROMRG) THEN
           NAROM=NAROM+1
           NADD=NADD+1
           if(prnlev.ge.2) WRITE(OUTU,'(A,I3,A)') &
                ' SUBRING',NR,' ALSO RECOGNIZED AS AROMATIC'
           !           IF(IZ.NE.0) WRITE(IZ,'(A,I3,A)')
           !     .       ' SUBRING',NR,' ALSO RECOGNIZED AS AROMATIC'
           CALL ARRING(LIST,MAXLEV,MXSIZE,NR,NRGAT(NR))
           AROM(NR)=.TRUE.
        ENDIF
9100    CONTINUE
     ENDDO
     IF(NADD.NE.0) GOTO 8900
6000 CONTINUE
  ENDDO
  !   RESTORE ITAB ARRAY WITH SINGLE ENTRIES FOR MULTIPLE BONDS
  CALL TABINT(1,IB,JB,BondType,NATOM,NBOND,ITAB,LOCAT)
  RETURN
END SUBROUTINE RGTYPE

! ==========================================================
! SUBROUTINE XTYPE : Assign symbolic atom types for non-hydrogen atoms
! ==========================================================
SUBROUTINE XTYPE
  !CCCC     SUBROUTINE XTYPE(NATOM,ATNUM,ITAB)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use psf
  use stream
  use string
  implicit none
  !
  !     integer NATOM
  !     integer ATNUM(*)
  !     integer ITAB(6,*)
  !
  integer iatom, ierr
  !
  integer, PARAMETER :: MAXKN=53 ! maximum allowed atomic number
  !
  integer icarb, icarb1, icarb2, icarb3, initr1, initr2, initrg
  integer iox, iox1, iox2, iphos, iphos1, iphos2, iphos3
  integer isulf, isulf3, iterm
  integer jattch, jbonds, jc, jcb, jh, jn, jnb, jo
  integer job, JPA, jpb, js, jsb, jx, kattch, kbonds, kc, kcb, kh
  integer kn, knb, ko, kob, KPA, kpb, ks, ksb, kx
  integer lattch, lbonds, lc, lcb, lh, ln, lnb, lo, lob, LPA, lpb
  integer ls, lsb, lx, mattch, mbonds, mc, mcb, mh, mn, mnb, mo
  integer mob, mpA, mpb, ms, msb, mx, nit1, nit2, nit3, ntox
  integer ntox1, ntox2
  integer initr, ichlor, isulf1, isulf2, nts
  integer nno, ntsf, n1, n2, ncnord
  real(chm_real)  xntox, xnts
  !
  integer i
  integer :: KNOWN(MAXKN)=(/1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,0,0,0,0,0,1, &
       0,0,1,1,0,0,0,0,1,(0,i=1,17),1/)
  IERR=0
  DO IATOM=1,NATOM
     !   INITIALIZE FORMAL CHARGES TO ZERO AND ATOM TYPE SYMBOLS TO BLANK
     PartlQ(IATOM)=0.
     SYMB(IATOM)='    '
     IFAROM(IATOM)=.FALSE.
  ENDDO
  DO IATOM=1,NATOM
     !
     !   DO NOT ASSIGN MMFF TYPES IF THEY ARE ALREADY ASSIGNED IN THE INPUT D
     !   AS MTYPE IS NONZERO, THIS IS EITHER AN ATOM WHOSE TYPE WAS
     !   DEFINED ON INPUT, OR IS AN ATOM WHICH IS PART OF A GROUP
     !   (E.G., GUANIDINIUM) AND WHOSE TYPE WAS ASSIGNED WHEN
     !   A LOWER NUMBERED ATOM IN THAT GROUP WAS TYPED
     !   IN EITHER CASE, BRANCH: DO NOT PROCESS THIS ATOM
     !
     IF(MTYPE(IATOM).NE.0) GOTO 1400
     !
     !RCZ 931130 - attempt to introduce atoms with negative atomic
     !             numbers as dummy atoms
     !
     !      IF(ATNUM(IATOM).LT.1) THEN
     !        if(wrnlev.ge.2) WRITE(OUTU,10) QNAME(IATOM),ATNUM(IATOM)
     !C     IF(IZ.NE.0) WRITE(IZ,10) QNAME(IATOM),ATNUM(IATOM)
     !   10   FORMAT(' ATOM',1X,A,' HAS AN INVALID ATOMIC NUMBER OF',
     !     .  I10//' SORRY, BUT MMFF CANNOT HANDLE THIS CASE')
     !        GOTO 1399
     !      ENDIF
     !
     IF(ABS(ATNUM(IATOM)).GT.MAXKN) THEN
        if(wrnlev.ge.2) WRITE(OUTU,20) ATNUM(IATOM), QNAME(IATOM),MAXKN
        !     IF(IZ.NE.0) WRITE(IZ,20) ATNUM(IATOM), QNAME(IATOM),MAXKN
20      FORMAT(' ATOMIC NUMBER OF',I5,' FOR ATOM',1X,A, &
             ' IS TOO HIGH'//' MAXIMUM ATOMIC NUMBER RECOGNIZED BY', &
             ' MMFF IS',I3)
        GOTO 1399
     ENDIF
     IF(KNOWN(ATNUM(IATOM)).NE.1) THEN
        if(wrnlev.ge.2) WRITE(OUTU,30) QNAME(IATOM),ATNUM(IATOM)
        !     IF(IZ.NE.0) WRITE(IZ,30) QNAME(IATOM),ATNUM(IATOM)
30      FORMAT(' SORRY, BUT MMFF DOES NOT RECOGNIZE ATOM',1X,A, &
             '  WITH ATOMIC NUMBER',I3)
        GOTO 1399
     ENDIF
     !   BRANCH IF THIS IS A HYDROGEN ATOM
     !
     IF(ATNUM(IATOM).EQ.1) GOTO 1400
     !   GET THEN NUMBERS AND TYPES OF THE ATOMS BONDED TO IATOM
     CALL BINFO(IATOM,NATOM,ATNUM,ITAB, &
          KH,KX,KATTCH,KBONDS,KC,KCB,KN,KNB,KO,KOB,KS,KSB,KPA,KPB)
     SYMB(IATOM)='    '
     !  ----  OXYGEN  -------------------------------------------------
     IF(ATNUM(IATOM).NE.8) GOTO 200
     IF(KATTCH.EQ.2.AND.KBONDS.EQ.2) THEN
        NNO=0
        IF(KN.EQ.1) THEN
           INITR=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
           NNO=NTERMA(INITR,NATOM,ITAB,ATNUM,8)
        ENDIF
        IF(KC.EQ.1) SYMB(IATOM)='OR'
        IF(KC.EQ.1.AND.NNO.EQ.0.AND.KS+KPA.EQ.0) THEN
           !         THIS IS AN ALCOHOL OR  ENOL OR CARBOXYLIC ACID OXYGEN
           !   FIND AND CHARACTERIZE THE ATTACHED CARBON
           ICARB=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,0)
           CALL BINFO(ICARB,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LATTCH.EQ.4) THEN
              !                                   ALCOHOL - LEAVE "OR" IN PLACE
              GOTO 1200
           ELSE IF(LATTCH.EQ.3) THEN
              IF(LOB.EQ.LO+1) THEN
                 SYMB(IATOM)='OC=O'
                 !                                    ESTER OR ACID -O-
                 GOTO 1200
              ELSE IF(LCB.EQ.LC+1) THEN
                 SYMB(IATOM)='OC=C'
                 !                                         ENOL OR ENOL ETHER -O-
                 GOTO 1200
              ELSE IF(LNB.EQ.LN+1) THEN
                 SYMB(IATOM)='OC=N'
                 GOTO 1200
              ELSE IF(LSB.EQ.LS+1) THEN
                 SYMB(IATOM)='OC=S'
                 !                GOTO 1200
              ELSE
                 !                  DEFAULT CASE - LEAVE 'OR' IN PLACE
                 GOTO 1200
              ENDIF
           ENDIF
        ELSE IF(KH.EQ.2) THEN
           !         THIS IS A WATER OXYGEN
           SYMB(IATOM)='OH2'
           GOTO 1200
        ELSE IF(KH.EQ.0.AND.KC.EQ.2.AND.KCB.EQ.2) THEN
           !         ESTER OR ETHER OR ENOL ETHER OXYGEN
           SYMB(IATOM)='OR'
           !   FIND AND CHARACTERIZE THE TWO ATTACHED CARBONS
           ICARB1=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,0)
           CALL BINFO(ICARB1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           ICARB2=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,1)
           CALL BINFO(ICARB2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(LATTCH.EQ.3.OR.JATTCH.EQ.3) THEN
              IF(LOB.EQ.LO+1.OR.JOB.EQ.JO+1) THEN
                 SYMB(IATOM)='OC=O'
                 !                                 ESTER -O- SINCE C=O ATTACHED
                 GOTO 1200
              ELSE IF(LCB.EQ.LC+1.OR.JCB.EQ.JC+1) THEN
                 SYMB(IATOM)='OC=C'
                 !                                        ENOL ETHER SINCE ONE C IS
                 GOTO 1200
              ELSE IF(LNB.EQ.LN+1.OR.JNB.EQ.JN+1) THEN
                 SYMB(IATOM)='OC=N'
                 GOTO 1200
              ELSE IF(LSB.EQ.LS+1.OR.JSB.EQ.JS+1) THEN
                 SYMB(IATOM)='OC=S'
                 GOTO 1200
              ELSE
                 !                LEAVE 'OR' IN PLACE
                 GOTO 1200
              ENDIF
           ENDIF
           GOTO 1200
        ENDIF
        IF(KN.EQ.1) THEN
           !          NON-TERMINAL OXYGEN ATTACHED TO NITROGEN
           !   FIND AND CHARACTERIZE THE ATTACHED NITROGEN
           INITR=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
           CALL BINFO(INITR,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LO.EQ.3.AND.LOB.EQ.4) THEN
              SYMB(IATOM)='ONO2'
              !                                NITRATE OXYGEN
              GOTO 1200
           ELSE IF(LO.EQ.2.AND.LOB.EQ.3) THEN
              SYMB(IATOM)='ON=O'
              !                                -O-N=O OXYGEN
              GOTO 1200
           ENDIF
        ENDIF
        !        IF(KS.EQ.1.AND.(KH.EQ.1.OR.KC.EQ.1)) THEN
        IF(KS.EQ.1) THEN
           !         SULF ESTER OR ACID SUCH AS R-O-S OR H-O-S
           ISULF=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           CALL BINFO(ISULF,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LO.EQ.4) SYMB(IATOM)='OSO3'
           IF(LO.EQ.3) SYMB(IATOM)='OSO2'
           IF(LO.EQ.2.AND.LOB.EQ.2) SYMB(IATOM)='OSO'
           IF(LO.EQ.2.AND.LOB.EQ.3) SYMB(IATOM)='OS=O'
           IF(LO.EQ.3.AND.LOB.EQ.4) SYMB(IATOM)='OS=O'
           IF(LO.EQ.1) SYMB(IATOM)='-OS'
           GOTO 1200
        ENDIF
        IF(KS.EQ.2) THEN
           !         SULF ESTER OR ACID S-O-S
           ISULF1=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           ISULF2=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,1)
           CALL BINFO(ISULF1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           CALL BINFO(ISULF2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(LO.GE.JO) THEN
              IF(LO.EQ.4) SYMB(IATOM)='OSO3'
              IF(LO.EQ.3) SYMB(IATOM)='OSO2'
              IF(LO.EQ.2.AND.LOB.EQ.2) SYMB(IATOM)='OSO'
              IF(LO.EQ.2.AND.LOB.EQ.3) SYMB(IATOM)='OS=O'
              IF(LO.EQ.3.AND.LOB.EQ.4) SYMB(IATOM)='OS=O'
              IF(LO.EQ.1) SYMB(IATOM)='-OS'
              GOTO 1200
           ELSE
              IF(JO.EQ.4) SYMB(IATOM)='OSO3'
              IF(JO.EQ.3) SYMB(IATOM)='OSO2'
              IF(JO.EQ.2.AND.JOB.EQ.2) SYMB(IATOM)='OSO'
              IF(JO.EQ.2.AND.JOB.EQ.3) SYMB(IATOM)='OS=O'
              IF(JO.EQ.3.AND.JOB.EQ.4) SYMB(IATOM)='OS=O'
              IF(JO.EQ.1) SYMB(IATOM)='-OS'
              GOTO 1200
           ENDIF
        ENDIF
        !        IF(KPA.EQ.1.AND.(KH.EQ.1.OR.KC.EQ.1)) THEN
        IF(KPA.EQ.1) THEN
           !         PHOS ESTER OR ACID SUCH AS R-O-P OR H-O-P
           IPHOS=IATTCH(IATOM,NATOM,ATNUM,ITAB,15,0,0)
           CALL BINFO(IPHOS,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LO.EQ.4) SYMB(IATOM)='OPO3'
           IF(LO.EQ.3) SYMB(IATOM)='OPO2'
           IF(LO.EQ.2) SYMB(IATOM)='OPO'
           IF(LO.EQ.1) SYMB(IATOM)='-OP'
           GOTO 1200
        ENDIF
        IF(KPA.EQ.2) THEN
           !         PHOS ESTER OR ACID P-O-P
           IPHOS1=IATTCH(IATOM,NATOM,ATNUM,ITAB,15,0,0)
           CALL BINFO(IPHOS1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IPHOS2=IATTCH(IATOM,NATOM,ATNUM,ITAB,15,0,1)
           CALL BINFO(IPHOS2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(LO.GE.JO) THEN
              IF(LO.EQ.4) SYMB(IATOM)='OPO3'
              IF(LO.EQ.3) SYMB(IATOM)='OPO2'
              IF(LO.EQ.2) SYMB(IATOM)='OPO'
              IF(LO.EQ.1) SYMB(IATOM)='-OP'
              GOTO 1200
           ELSE
              IF(JO.EQ.4) SYMB(IATOM)='OPO3'
              IF(JO.EQ.3) SYMB(IATOM)='OPO2'
              IF(JO.EQ.2) SYMB(IATOM)='OPO'
              IF(JO.EQ.1) SYMB(IATOM)='-OP'
              GOTO 1200
           ENDIF
        ENDIF
        !   IF STILL HERE, THIS OXYGEN COULD NOT BE SPECIFICALLY TYPED -
        !   ASSIGN -O- AS THE DEFAULT TYPE UNLESS 'OR' HAS ALREADY
        !   BEEN ASSIGNED
        IF(SYMB(IATOM).NE.'OR') SYMB(IATOM)='-O-'
        GOTO 1200
     ENDIF
     !     TERMINAL OXYGENS
     IF(KATTCH.EQ.1) THEN
        !                        THIS IS A TERMINAL OXYGEN
        IF(KH.EQ.1) THEN
           !           OXIDE OXYGEN IN HYDROXIDE ANION
           SYMB(IATOM)='OM'
           PartlQ(IATOM)=-1.
           GOTO 1200
        ELSE IF(KC.EQ.1) THEN
           !                        TERMINAL OXYGEN ATTACHED TO CARBON
           ICARB=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,0)
           CALL BINFO(ICARB,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LATTCH.EQ.4) THEN
              !         THIS IS AN OXIDE OXYGEN, AS IT IS ON A TETRAHEDRAL CARBON
              PartlQ(IATOM)=-1.
              SYMB(IATOM)='OM'
              GOTO 1200
           ENDIF
           IF(LO.EQ.1.AND.LOB.EQ.2) THEN
              !         THIS IS A CARBONYL OXYGEN IN A KETONE, ALDEHYDE, OR AMIDE
              !           OR THIOESTER, ETC.
              SYMB(IATOM)='O=C'
              IF(LN.EQ.0.AND.LC+LH.EQ.2) SYMB(IATOM)='O=CR'
              IF(LN.NE.0) THEN
                 IF(LATTCH.EQ.3) THEN
                    SYMB(IATOM)='O=CN'
                 ENDIF
              ENDIF
              GOTO 1200
           ELSE IF(LATTCH.EQ.3.AND.LO.EQ.LOB) THEN
              !         THIS IS AN OXIDE OXYGEN, BUT ON A SP2-HYBRIDIZED CARBON
              PartlQ(IATOM)=-1.
              SYMB(IATOM)='OM2'
              GOTO 1200
           ENDIF
           IF(LO.EQ.2) THEN
              !                          TERMINAL OXYGEN IN -CO2(-) OR -CO2(H,R)
              !                          FIND NUMBER OF TERMINAL O'S ON ICARB
              ITERM=NTERMA(ICARB,NATOM,ITAB,ATNUM,8)
              IF(ITERM.EQ.2) THEN
                 !                               SECOND OXYGEN IS ALSO TERMINAL
                 SYMB(IATOM)='O2CM'
                 PartlQ(IATOM)=-0.5
                 GOTO 1200
              ELSE IF(ITERM.EQ.1) THEN
                 !                               SECOND OX IS NONTERMINAL
                 IF(LOB.EQ.3) THEN
                    SYMB(IATOM)='O=CO'
                    IF(LN.EQ.1) SYMB(IATOM)='O=CN'
                    !                               AMIDE TAKES PRECEDENCE
                    GOTO 1200
                 ELSE IF(LOB.EQ.2) THEN
                    !                               THIS IS ALSO AN OXIDE OXYGEN
                    SYMB(IATOM)='OM'
                    PartlQ(IATOM)=-1.
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
           IF(LO.EQ.3) THEN
              !                          MAY BE A CARBONIC ACID O=C OXYGEN
              !                          FIND NUMBER OF TERMINAL O'S ON ICARB
              ITERM=NTERMA(ICARB,NATOM,ITAB,ATNUM,8)
              IF(ITERM.EQ.1) THEN
                 !                               THIS ONE IS THE ONLY TERMINAL OXYGEN
                 SYMB(IATOM)='O=CO'
                 GOTO 1200
              ENDIF
           ENDIF
        ENDIF
        IF(KN.EQ.1) THEN
           !                        TERMINAL OXYGEN ON NITROGEN
           INITRG=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
           !                        THE TERMINAL NITROGEN IS ATOM INITRG; GET
           !                        ITS BONDING PROFILE
           CALL BINFO(INITRG,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           !            COUNT THE NUMBER OF TERMINAL OXYGENS
           NTOX=NTERMA(INITRG,NATOM,ITAB,ATNUM,8)
           IF(NTOX.EQ.1.AND.KNB.EQ.2) THEN
              !                           NITROSO OXYGEN
              SYMB(IATOM)='O=N'
              GOTO 1200
           ELSE IF(NTOX.EQ.1.AND.KNB.EQ.1) THEN
              IF(LATTCH.EQ.4.AND.LBONDS.EQ.4) THEN
                 !                           N-OXIDE OXYGEN
                 SYMB(IATOM)='OXN'
                 PartlQ(IATOM)=0.
                 GOTO 1200
              ELSE IF(LATTCH.EQ.3.AND.LBONDS.EQ.4) THEN
                 !                           N-OXIDE OXYGEN ON SP2 NITROGEN
                 SYMB(IATOM)='OXN'
                 PartlQ(IATOM)=0.
                 GOTO 1200
              ELSE IF(LATTCH.EQ.3.AND.LBONDS.EQ.3) THEN
                 !                           NEGATIVELY CHARGED TERMINAL OXYGEN ON NITROG
                 SYMB(IATOM)='OM'
                 PartlQ(IATOM)=-1.
                 GOTO 1200
              ELSE IF(LATTCH.EQ.2.AND.LBONDS.EQ.3) THEN
                 !                           NEGATIVELY CHARGED TERMINAL OXYGEN ON NITROG
                 SYMB(IATOM)='OM2'
                 PartlQ(IATOM)=-1.
                 GOTO 1200
              ELSE
                 !
                 !   THIS ENVIRONMENT IS NOT ONE RECOGNIZED BY MMFF - WRITE
                 !   WARNING MESSAGE AND BRANCH
                 !
                 if(wrnlev.ge.2) WRITE(OUTU,215) QNAME(IATOM),QNAME(INITRG)
                 !              IF(IZ.NE.0) if(wrnlev.ge.2) WRITE(OUTU,215) QNAME(IATOM),
                 GOTO 1399
              ENDIF
           ELSE IF(NTOX.EQ.2.AND.LO.EQ.2) THEN
              !                                TWO TERMINAL OXYGENS, E.G.,
              !                                NITRO-GROUP OXYGEN
              SYMB(IATOM)='O2N'
              GOTO 1200
           ELSE IF(NTOX.EQ.2.AND.LO.EQ.3) THEN
              !                                TWO TERMINAL OXYGENS, E.G.,
              !                                NITRO-GROUP OXYGEN
              SYMB(IATOM)='O2NO'
              GOTO 1200
           ELSE IF(NTOX.EQ.3) THEN
              !                                NITRATE OXYGEN, NO3(-)
              SYMB(IATOM)='O3N'
              PartlQ(IATOM)=-1./3.
              GOTO 1200
           ELSE
              !   IF STILL HERE, ERROR
              !   IF STILL HERE, TYPE NOT RECOGNIZED - WRITE MSG
              if(wrnlev.ge.2) WRITE(OUTU,215) QNAME(IATOM),QNAME(INITRG)
              !           IF(IZ.NE.0)
              !    .      WRITE(IZ,215) QNAME(IATOM),QNAME(INITRG)
              !           IF(IZ.NE.0) WRITE(IZ,220) QNAME(IATOM)
215           FORMAT(' OXYGEN',1X,A,' IS ATTACHED TO NITROGEN', &
                   1X,A,' BUT COULD NOT BE SUCCESSFULLY TYPED.', &
                   '  SORRY')
              GOTO 1399
           ENDIF
        ENDIF
        IF(KS.EQ.1) THEN
           !                        TERMINAL OXYGEN ON SULFUR
           ISULF=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           CALL BINFO(ISULF,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           !            COUNT THE NUMBER OF TERMINAL OXYGENS
           NTOX=NTERMA(ISULF,NATOM,ITAB,ATNUM,8)
           XNTOX=NTOX
           IF(NTOX.EQ.1.AND.LATTCH.EQ.3.AND.LBONDS.EQ.4) THEN
              !                           SULFOXIDE OXYGEN
              SYMB(IATOM)='O=S'
              GOTO 1200
           ELSE IF(NTOX.EQ.1.AND.LATTCH.EQ.3.AND.LBONDS.EQ.3 &
                .AND.LSB.EQ.1) THEN
              !                           THIOSULFINYL OXYGEN - IT'S PARTNER,
              !                           SYMB='SSMO', ALSO HAS FORMAL CHARGE = -1/2
              SYMB(IATOM)='OSMS'
              PartlQ(IATOM)=-0.5
              GOTO 1200
           ELSE IF(NTOX.EQ.1.AND.LATTCH.EQ.2.AND.LCB.EQ.2) THEN
              !                           SYLFINYL SULFUR (IN O=S=C<)
              SYMB(IATOM)='O=S='
              GOTO 1200
           ELSE IF(NTOX.EQ.1.AND.LATTCH.EQ.4) THEN
              !                           SINGLE TERMINAL OXYGEN ON TETRACOORD S
              SYMB(IATOM)='O-S'
              GOTO 1200
           ELSE IF(NTOX.EQ.2) THEN
              !                                TWO TERMINAL OXYGENS, E.G.,
              !                                SULFONE OR SULFONAMIDE OXYGEN
              SYMB(IATOM)='O2S'
              PartlQ(IATOM)=-(NTOX-LBONDS+2)/XNTOX
              !                PartlQ=0 IF FOUR ATTACHMENTS AND TWO TERMINAL OX
              !                PartlQ=-1/2 IF THREE ATTACHMENTS AND TWO TERM OX
              GOTO 1200
           ELSE IF(NTOX.EQ.3) THEN
              !                                THREE TERMINAL OX, E.G., ROSO3(-)
              SYMB(IATOM)='O3S'
              PartlQ(IATOM)=-(NTOX-LBONDS+2)/XNTOX
              !                 PartlQ=-1/3 IF FOUR ATTACHMENTS AND THREE TERM OX
              !                 PartlQ=-2/3 IF THREE ATTACHMENTS AND THREE TERM OX
              !                             (I.E., SO3(-2))
              GOTO 1200
           ELSE IF(NTOX.EQ.4) THEN
              !                                SULFATE OXYGEN
              SYMB(IATOM)='O4S'
              PartlQ(IATOM)=-(NTOX-LBONDS+2)/XNTOX
              !                PartlQ(IATOM)=-0.5=-2/4 IF FOUR ATTCH & FOUR TERM OX
              GOTO 1200
           ENDIF
        ENDIF
        IF(KPA.EQ.1) THEN
           !                        TERMINAL OXYGEN ATTACHED TO PHOSPHORUS
           IPHOS=IATTCH(IATOM,NATOM,ATNUM,ITAB,15,0,0)
           !         CHARACTERIZE BONDING TO IPHOS
           CALL BINFO(IPHOS,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           NTOX=NTERMA(IPHOS,NATOM,ITAB,ATNUM,8)
           XNTOX=NTOX
           IF(NTOX.EQ.1) THEN
              !                          PHOSPHOXIDE OXYGEN
              SYMB(IATOM)='OP'
              !            PartlQ(IATOM)=-(NTOX-LATTCH+3)/XNTOX
              PartlQ(IATOM)=0.
              GOTO 1200
           ELSE IF(NTOX.EQ.2) THEN
              SYMB(IATOM)='O2P'
              PartlQ(IATOM)=-(NTOX-LATTCH+3)/XNTOX
              !              PartlQ=-1/2 IF FOUR ATTACHMENTS AND TWO TERMINAL OX
              GOTO 1200
           ELSE IF(NTOX.EQ.3) THEN
              !                                 PHOSPHONATE OXYGEN
              SYMB(IATOM)='O3P'
              PartlQ(IATOM)=-(NTOX-LATTCH+3)/XNTOX
              !              PartlQ=-2/3 IF FOUR ATTACHMENTS AND THREE TERM OX
              GOTO 1200
           ELSE IF(NTOX.EQ.4) THEN
              !                                PHOSPHATE OXYGEN
              SYMB(IATOM)='O4P'
              PartlQ(IATOM)=-(NTOX-LATTCH+3)/XNTOX
              !              PartlQ=-3/4 IF FOUR ATTACHMENTS AND FOUR TERMINAL OX
              !                          (I.E., PO4(-3) ION)
              GOTO 1200
           ENDIF
        ENDIF
        !   IS THIS OXYGEN ATTACHED TO CHLORINE IN CLO4(-)?
        ICHLOR=0
        ICHLOR=IATTCH(IATOM,NATOM,ATNUM,ITAB,17,0,0)
        IF(ICHLOR.NE.0) THEN
           ITERM=NTERMA(ICHLOR,NATOM,ITAB,ATNUM,8)
           IF(ITERM.EQ.4) THEN
              !     YES - THIS O IS ONE OF FOUR TERMINAL OXYGENS ATTACHED TO CL
              SYMB(IATOM)='O4CL'
              PartlQ(IATOM)=-0.25
              GOTO 1200
           ENDIF
        ENDIF
     ENDIF
     IF(KBONDS.GE.3) THEN
        !        AN ONIUM ION OF SOME SORT
        IF(KATTCH.EQ.3) THEN
           SYMB(IATOM)='O+'
           PartlQ(IATOM)=1.
           GOTO 1200
        ELSE IF(KATTCH.EQ.2) THEN
           SYMB(IATOM)='O=+'
           PartlQ(IATOM)=1.
           GOTO 1200
        ELSE IF(KATTCH.EQ.1) THEN
           SYMB(IATOM)='O%+'
           PartlQ(IATOM)=1.
           GOTO 1200
        ENDIF
     ENDIF
200  CONTINUE
     IF(ATNUM(IATOM).NE.7) GOTO 600
     ! ----  NITROGEN  ----------------------------------------------------
     IF(KATTCH.EQ.4) THEN
        IF(KBONDS.EQ.4) THEN
           IF(KO.EQ.KOB.AND.KO.EQ.1) THEN
              !                            IS THIS IS A NEUTRAL N-OXIDE NITROGEN?
              IOX=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,0,0)
              CALL BINFO(IOX,NATOM,ATNUM,ITAB, &
                   LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
              IF(LATTCH.EQ.1) THEN
                 !                            YES, THIS IS A NEUTRAL N-OXIDE NITROGEN
                 SYMB(IATOM)='N3OX'
                 PartlQ(IATOM)=0.
                 GOTO 1200
              ELSE
                 !                            THIS IS A POSITIVELY CHARGED QUATERNARY N
                 SYMB(IATOM)='NR+'
                 PartlQ(IATOM)=1.
                 GOTO 1200
              ENDIF
           ELSE
              !                            THIS IS A POSITIVELY CHARGED QUATERNARY N
              SYMB(IATOM)='NR+'
              PartlQ(IATOM)=1.
              GOTO 1200
           ENDIF
        ELSE
           if(wrnlev.ge.2) WRITE(OUTU,210) QNAME(IATOM)
           !         IF(IZ.NE.0) WRITE(IZ,210) QNAME(IATOM)
210        FORMAT(' NITROGEN',1X,A,' IS BONDED TO FOUR OTHER ATOMS' &
                ,', BUT'/' NOT IN A WAY THAT MMFF CAN RECOGNIZE.  SORRY')
           GOTO 1399
        ENDIF
     ENDIF
     IF(KATTCH.EQ.3.AND.KBONDS.EQ.4) THEN
        !   THIS NITROGEN HAS THREE NEIGHBORS, ONE DOUBLY BONDED
        !   IS THIS A NITRO GROUP?
        IF(KO.EQ.2.AND.KOB.EQ.3) THEN
           SYMB(IATOM)='NO2'
           GOTO 1200
        ELSE IF(KO.EQ.3.AND.KOB.EQ.4) THEN
           SYMB(IATOM)='NO3'
           GOTO 1200
        ELSE IF(KO.EQ.KOB.AND.KO.EQ.1.AND.KCB.EQ.KC+1) THEN
           !                            IS THIS IS A NEUTRAL N-OXIDE NITROGEN
           !                            WITH N DOUBLY BONDED TO C?
           IOX=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,0,0)
           CALL BINFO(IOX,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LATTCH.EQ.1) THEN
              !                            YES, THIS IS A NEUTRAL N-OXIDE NITROGEN
              SYMB(IATOM)='N2OX'
              PartlQ(IATOM)=0.
              GOTO 1200
           ELSE IF(LATTCH.EQ.2) THEN
              !                             NO, THIS IS A FORMALLY CHARGED NITROGEN
              SYMB(IATOM)='N+=C'
              PartlQ(IATOM)=1.
              GOTO 1200
           ENDIF
        ELSE IF(KO.EQ.KOB.AND.KOB.EQ.1.AND.KNB.EQ.KN+1) THEN
           !                             NEUTRAL AMINE OXIDE WITH N DOUBLE BONDED T
           SYMB(IATOM)='N2OX'
           GOTO 1200
        ELSE IF(KO.EQ.KOB.AND.KO.EQ.2) THEN
           !                            IS THIS IS ALSO A NEUTRAL N-OXIDE NITROGEN?
           !         FIND THE TWO OXYGENS
           IOX1=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,0,0)
           IOX2=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,IOX1,0)
           !                        CHARACTERIZE THEN
           CALL BINFO(IOX1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           CALL BINFO(IOX2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(LATTCH+JATTCH.EQ.3) THEN
              !                            YES, THIS IS ASLO A NEUTRAL N-OXIDE NITROGE
              SYMB(IATOM)='N2OX'
              PartlQ(IATOM)=0.
              GOTO 1200
           ENDIF
        ELSE IF(KCB.EQ.KC+1) THEN
           !                             DOUBLY BONDED TO C: IMINIUM NITROGEN
           SYMB(IATOM)='N+=C'
           PartlQ(IATOM)=1.
           GOTO 1200
        ELSE IF(KNB.EQ.KN+1) THEN
           !                             DOUBLY BONDED TO N:
           SYMB(IATOM)='N+=N'
           PartlQ(IATOM)=1.
           GOTO 1200
        ENDIF
        !   IF STILL HERE, TYPE NOT RECOGNIZED - WRITE MSG
        if(wrnlev.ge.2) WRITE(OUTU,220) QNAME(IATOM)
        !       IF(IZ.NE.0) WRITE(IZ,220) QNAME(IATOM)
220     FORMAT(' NITROGEN ATOM',1X,A,' MAKES ', &
             'FOUR BONDS TO THREE OTHER ATOMS' &
             /' BUT COULD NOT BE SUCCESSFULLY TYPED.  SORRY')
        GOTO 1399
     ENDIF
     IF(KATTCH.EQ.3.AND.KBONDS.EQ.3) THEN
        !   THIS NITROGEN HAS THREE SINGLY BONDED NEIGHBORS
        !   FIRST SET DEFAULT TYPE (= AMINE)
        SYMB(IATOM)='NR'
        !
        IF(KS.GE.1) THEN
           !                        SULFONAMIDE?
           !        FIND AND ASSESS THE FIRST (OR ONLY) SULFUR
           !
           ISULF1=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           CALL BINFO(ISULF1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           !         WRITE(6,*) ' LO =',LO
           IF(LO.GE.2) THEN
              !
              !   FIND OUT HOW MANY OXYGENS ARE TERMINAL
              !
              ITERM=NTERMA(ISULF1,NATOM,ITAB,ATNUM,8)
              !      WRITE(6,*) ' ITERM = ',ITERM
              IF(ITERM.GE.2) THEN
                 IF(LO.EQ.3) THEN
                    SYMB(IATOM)='NSO3'
                    GOTO 1200
                 ELSE IF(LO.EQ.2) THEN
                    SYMB(IATOM)='NSO2'
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        IF(KS.GE.2) THEN
           !   FIND AND PROCESS THE SECOND SULFUR
           ISULF2=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,1)
           CALL BINFO(ISULF2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(JO.GE.2) THEN
              !
              !   FIND OUT HOW MANY OXYGENS ARE TERMINAL
              !
              ITERM=NTERMA(ISULF2,NATOM,ITAB,ATNUM,8)
              IF(ITERM.GE.2) THEN
                 IF(JO.EQ.3) THEN
                    SYMB(IATOM)='NSO3'
                    GOTO 1200
                 ELSE IF(JO.EQ.2) THEN
                    SYMB(IATOM)='NSO2'
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        IF(KS.GE.3) THEN
           !   FIND AND PROCESS THE THIRD SULFUR
           ISULF3=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,2)
           CALL BINFO(ISULF3,NATOM,ATNUM,ITAB, &
                MH,MX,MATTCH,MBONDS,MC,MCB,MN,MNB,MO,MOB,MS,MSB,MPA,MPB)
           IF(MO.GE.2) THEN
              !
              !   FIND OUT HOW MANY OXYGENS ARE TERMINAL
              !
              ITERM=NTERMA(ISULF3,NATOM,ITAB,ATNUM,8)
              IF(ITERM.GE.2) THEN
                 IF(MO.EQ.3) THEN
                    SYMB(IATOM)='NSO3'
                    GOTO 1200
                 ELSE IF(MO.EQ.2) THEN
                    SYMB(IATOM)='NSO2'
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        !
        IF(KPA.GE.1) THEN
           !                        PHOSPHONAMIDE?
           !        FIND AND ASSESS THE FIRST (OR ONLY) PHOSPHORUS
           !
           IPHOS1=IATTCH(IATOM,NATOM,ATNUM,ITAB,15,0,0)
           CALL BINFO(IPHOS1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LO.GE.2) THEN
              !
              !   FIND OUT HOW MANY OXYGENS ARE TERMINAL
              !
              ITERM=NTERMA(IPHOS1,NATOM,ITAB,ATNUM,8)
              IF(ITERM.GE.2) THEN
                 IF(LO.EQ.3) THEN
                    SYMB(IATOM)='NPO3'
                    GOTO 1200
                 ELSE IF(LO.EQ.2) THEN
                    SYMB(IATOM)='NPO2'
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        IF(KPA.GE.2) THEN
           !   FIND AND PROCESS TRHE SECOND PHOSPHORUS
           IPHOS2=IATTCH(IATOM,NATOM,ATNUM,ITAB,15,0,1)
           CALL BINFO(IPHOS2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(JO.GE.2) THEN
              !
              !   FIND OUT HOW MANY OXYGENS ARE TERMINAL
              !
              ITERM=NTERMA(IPHOS2,NATOM,ITAB,ATNUM,8)
              IF(ITERM.GE.2) THEN
                 IF(JO.EQ.3) THEN
                    SYMB(IATOM)='NPO3'
                    GOTO 1200
                 ELSE IF(JO.EQ.2) THEN
                    SYMB(IATOM)='NPO2'
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        IF(KPA.GE.3) THEN
           !   FIND AND PROCESS THE THIRD PHOSPHORUS
           IPHOS3=IATTCH(IATOM,NATOM,ATNUM,ITAB,15,0,2)
           CALL BINFO(IPHOS3,NATOM,ATNUM,ITAB, &
                MH,MX,MATTCH,MBONDS,MC,MCB,MN,MNB,MO,MOB,MS,MSB,MPA,MPB)
           IF(MO.GE.2) THEN
              !
              !   FIND OUT HOW MANY OXYGENS ARE TERMINAL
              !
              ITERM=NTERMA(IPHOS3,NATOM,ITAB,ATNUM,8)
              IF(ITERM.GE.2) THEN
                 IF(MO.EQ.3) THEN
                    SYMB(IATOM)='NPO3'
                    GOTO 1200
                 ELSE IF(MO.EQ.2) THEN
                    SYMB(IATOM)='NPO2'
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        !   FIND AND CHARACTERIZE THE NEIGHBORING CARBON ATOMS
        IF(KC.GE.1) THEN
           !                        CHARACTERIZE THE FIRST CARBON
           ICARB1=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,0)
           CALL BINFO(ICARB1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
        ENDIF
        IF(KC.EQ.1) THEN
           IF(LATTCH.EQ.3) THEN
              IF(LOB.EQ.LO+1) THEN
                 !                                 ATTACHED CARBON IS C=O, SO AMIDE
                 SYMB(IATOM)='NC=O'
                 GOTO 1200
              ELSE IF(LSB.EQ.LS+1) THEN
                 !                                 ATTACHED CARBON IS C=S, SO THIOAMIDE
                 SYMB(IATOM)='NC=S'
                 GOTO 1200
              ELSE IF(LNB.EQ.LN+1) THEN
                 !                                 ATTACHED CARBON IS C=N
                 SYMB(IATOM)='NC=N'
                 GOTO 1200
              ELSE IF(LCB.EQ.LC+1) THEN
                 !                                 ATTACHED TO C=C
                 SYMB(IATOM)='NC=C'
                 GOTO 1200
              ELSE IF(LPB.EQ.LPA+1) THEN
                 !                                 ATTACHED TO C=P
                 SYMB(IATOM)='NC=P'
                 GOTO 1200
              ELSE
                 !                     SOME OTHER SITUATION - LEAVE AS 'NR'
                 GOTO 1200
              ENDIF
           ELSE IF(LATTCH.EQ.2) THEN
              IF(LNB.EQ.LN+2) THEN
                 !                                 ATTACHED CARBON IS C%N
                 SYMB(IATOM)='NC%N'
                 GOTO 1200
              ELSE IF(LCB.EQ.LC+2) THEN
                 !                                 ATTACHED TO C%C
                 SYMB(IATOM)='NC%C'
                 GOTO 1200
              ELSE
                 !                SOME OTHER SITUATION - LEAVE AS 'NR' IN MOST CASES
                 IF(KN.EQ.0) GOTO 1200
              ENDIF
           ENDIF
        ENDIF
        IF(KC.GE.2) THEN
           !                        CHARACTERIZE THE SECOND CARBON
           ICARB2=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,1)
           CALL BINFO(ICARB2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(KC.EQ.2) THEN
              IF(LATTCH.EQ.4.AND.JATTCH.EQ.4.AND.KN.EQ.0) GOTO 1200
              IF(LOB+JOB.GE.LO+JO+1) THEN
                 !                                  AT LEAST ONE -C=O ATTACHED
                 SYMB(IATOM)='NC=O'
                 GOTO 1200
              ELSE IF(LSB+JSB.GE.LS+JS+1) THEN
                 !                                 AT LEAST ONE -C=S ATTACHED
                 SYMB(IATOM)='NC=S'
                 GOTO 1200
              ELSE IF(LNB.EQ.LN+2.OR.JNB.EQ.JN+2) THEN
                 !                                  AT LEAST ONE -C%N ATTACHED
                 SYMB(IATOM)='NC%N'
                 GOTO 1200
              ELSE IF(LNB.EQ.LN+1.OR.JNB.EQ.JN+1) THEN
                 !                                  AT LEAST ONE -C=N ATTACHED
                 SYMB(IATOM)='NC=N'
                 GOTO 1200
              ELSE IF(LCB+JCB.GE.LC+JC+1) THEN
                 !                                  AT LEAST ONE -C=C ATTACHED
                 SYMB(IATOM)='NC=C'
                 GOTO 1200
              ELSE IF(LPB+JPB.GE.LPA+JPA+1) THEN
                 !                                 AT LEAST ONE -C=P ATTACHED
                 SYMB(IATOM)='NC=P'
                 GOTO 1200
              ELSE
                 !                SOME OTHER CASE - LEAVE AS 'NR' IN MOST CASES
                 IF(KN.EQ.0) GOTO 1200
              ENDIF
           ENDIF
        ENDIF
        IF(KC.EQ.3) THEN
           !                        CHARACTERIZE THE THIRD CARBON
           ICARB3=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,2)
           CALL BINFO(ICARB3,NATOM,ATNUM,ITAB, &
                MH,MX,MATTCH,MBONDS,MC,MCB,MN,MNB,MO,MOB,MS,MSB,MPA,MPB)
           IF(LATTCH.EQ.4.AND.JATTCH.EQ.4.AND.MATTCH.EQ.4) GOTO 1200
           IF(LOB+JOB+MOB.GE.LO+JO+MO+1) THEN
              !                                  AT LEAST ONE -C=O ATTACHED
              SYMB(IATOM)='NC=O'
              GOTO 1200
           ELSE IF(LSB+JSB+MSB.GE.LS+JS+MS+1) THEN
              !                                 AT LEAST ONE -C=S ATTACHED
              SYMB(IATOM)='NC=S'
              GOTO 1200
           ELSE IF(LNB+JNB+MNB.GE.LN+JN+MN+1) THEN
              !                                  AT LEAST ONE -C=N ATTACHED
              SYMB(IATOM)='NC=N'
              GOTO 1200
           ELSE IF(LCB+JCB+MCB.GE.LC+JC+MC+1) THEN
              !                                  AT LEAST ONE -C=C ATTACHED
              SYMB(IATOM)='NC=C'
              GOTO 1200
           ELSE IF(LPB+JPB+MPB.GE.LPA+JPA+MPA+1) THEN
              !                                 AT LEAST ONE -C=P ATTACHED
              SYMB(IATOM)='NC=P'
              GOTO 1200
           ELSE
              !                SOME OTHER CASE - LEAVE AS 'NR'
              GOTO 1200
           ENDIF
        ENDIF
        IF(KN.GE.1) THEN
           !                        CHARACTERIZE THE FIRST NITROGEN
           INITR1=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
           CALL BINFO(INITR1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
        ENDIF
        IF(KN.GE.2) THEN
           !                        CHARACTERIZE THE SECOND NITROGEN
           INITR2=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,1)
           CALL BINFO(INITR2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
        ENDIF
        IF(KN.EQ.1) THEN
           IF(LATTCH.EQ.4) GOTO 1200
           IF(LNB.EQ.LN+1) THEN
              !                                 ATTACHED NITROGEN IS N=N
              SYMB(IATOM)='NN=N'
              GOTO 1200
           ENDIF
        ELSE IF(KN.GE.2) THEN
           IF(LATTCH.EQ.4.AND.JATTCH.EQ.4) GOTO 1200
           IF(LNB+JNB.GE.LN+JN+1) THEN
              !                                  AT LEAST ONE -N=N ATTACHED
              SYMB(IATOM)='NN=N'
              GOTO 1200
           ELSE IF(LCB+JCB.GE.LC+JC+1) THEN
              !                                  AT LEAST ONE -N=C ATTACHED
              SYMB(IATOM)='NN=C'
              GOTO 1200
           ENDIF
        ENDIF
     ENDIF
     !   TWO NEIGHBORS MAKING THREE BONDS
     IF(KATTCH.EQ.2.AND.KBONDS.EQ.3) THEN
        !                THIS N HAS TWO NEIGHBORS AND MAKES THREE BONDS.  TYPE I
        IF(KC.EQ.2.AND.KCB.EQ.3) THEN
           !                                    IMINE N IN C-N=C
           SYMB(IATOM)='N=C'
           GOTO 1200
        ENDIF
        IF(KH.EQ.1.AND.KC.EQ.1.AND.KCB.EQ.2) THEN
           !                                    IMINE N IN H-N=C
           SYMB(IATOM)='N=C'
           GOTO 1200
        ENDIF
        IF(KC.EQ.1.AND.KCB.EQ.2) THEN
           !                                 SOME OTHER -N=C
           SYMB(IATOM)='N=C'
           GOTO 1200
        ENDIF
        IF(KNB.EQ.KN+1) THEN
           !                          N=N IN C-N=N OR X-N=N (AZO)
           SYMB(IATOM)='N=N'
           GOTO 1200
        ENDIF
        IF(KO.EQ.1.AND.KOB.EQ.2) THEN
           !                                    NITROSO N IN O=N-
           SYMB(IATOM)='N=O'
           GOTO 1200
        ENDIF
        IF(KS.NE.0) THEN
           !                        SP2 NITROGEN ATTACHED TO SO2?
           IF(KS.EQ.1) THEN
              ISULF=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
              CALL BINFO(ISULF,NATOM,ATNUM,ITAB, &
                   LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
              IF(LO.EQ.2) THEN
                 SYMB(IATOM)='NSO2'
                 GOTO 1200
              ELSE IF(LATTCH.EQ.4.AND.LOB.EQ.LO.AND.LNB.EQ.LN) THEN
                 !                         NITROGEN SINGLE BONDED TO S AS IN >S(-NX)(-O)
                 SYMB(IATOM)='NSO'
                 GOTO 1200
              ENDIF
           ELSE IF(KS.EQ.2) THEN
              !   MAY NEED TO FIND AND CHARACTERIZE BOTH SULFURS
              ISULF1=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
              CALL BINFO(ISULF1,NATOM,ATNUM,ITAB, &
                   LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
              IF(LATTCH.EQ.4.AND.LBONDS.EQ.4) THEN
                 IF(LO.EQ.2) THEN
                    SYMB(IATOM)='NSO2'
                    GOTO 1200
                 ENDIF
              ENDIF
              !  FIND AND CHARACTERIZE THE SECOND SULFUR
              ISULF2=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,1)
              CALL BINFO(ISULF2,NATOM,ATNUM,ITAB, &
                   LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
              IF(LATTCH.EQ.4.AND.LBONDS.EQ.4) THEN
                 IF(LO.EQ.2) THEN
                    SYMB(IATOM)='NSO2'
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        !   TWO NEIGHBORS MAKING FOUR BONDS
     ELSE IF(KATTCH.EQ.2.AND.KBONDS.EQ.4) THEN
        IF(KN.EQ.2.AND.KNB.EQ.4) THEN
           !           CENTRAL NITROGEN IN AN AZID0 GROUP
           SYMB(IATOM)='=N='
           GOTO 1200
        ELSE IF(KN.EQ.1.AND.KNB.EQ.2.AND.KC.EQ.1.AND.KCB.EQ.2) THEN
           !           CENTRAL NITROGEN IN A DIAZO GROUP GROUP
           SYMB(IATOM)='=N='
           GOTO 1200
        ENDIF
        IF(NBND3(IATOM,ITAB).EQ.1) THEN
           !           FOUR BONDS, ONE TRIPLE
           IF(KNB.EQ.KN+2) THEN
              !               DIAZO NITROGEN
              SYMB(IATOM)='NR%'
              PartlQ(IATOM)=1.
              GOTO 1200
           ELSE IF(KCB.EQ.KC+2) THEN
              !               ISONIRTILE NITROGEN
              SYMB(IATOM)='NR%'
              GOTO 1200
           ENDIF
           !     ERROR IF STILL HERE
           if(wrnlev.ge.2) WRITE(OUTU,226) QNAME(IATOM)
           !            IF(IZ.NE.0) WRITE(IZ,226) QNAME(IATOM)
226        FORMAT(' NITROGEN',1X,A,' MAKES A TRIPLE BOND TO' &
                ,' A NEIGBORS, BUT '/ &
                ' MMFF CANNOT ASSIGN THE ATOM TYPE.  SORRY'/)
           GOTO 1399
        ENDIF
     ELSE IF(KATTCH.EQ.2.AND.KBONDS.EQ.2) THEN
        !   TWO NEIGHBORS MAKING TWO BONDS -- MAY BE (C,H)-N(-)-SO2-X
        !   OR SOME OTHER NEGATIVELY CHARGED NITROGEN
        IF(KS.EQ.1.AND.(KC.EQ.1.OR.KH.EQ.1.OR.KN.EQ.1)) THEN
           !                        CHARACTERIZE THE SULFUR ATOM
           ISULF=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           CALL BINFO(ISULF,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB, &
                LPA,LPB)
           !         FIND OUT HOW MANY TERMINAL OXYGENS ARE ON THE SULFUR
           NTOX=NTERMA(ISULF,NATOM,ITAB,ATNUM,8)
           IF(LATTCH.EQ.4.AND.NTOX.GE.2) THEN
              SYMB(IATOM)='NM'
              PartlQ(IATOM)=-1.
              GOTO 1200
           ENDIF
           IF(LATTCH.EQ.4.AND.NTOX.EQ.1) THEN
              SYMB(IATOM)='NSO'
              GOTO 1200
           ENDIF
        ELSE IF(KC+KH+KN+KO.EQ.2) THEN
           !                      SINCE NEITHER ATTACHED NEIGHBOR IS CABAPLE OF
           !                      'HYPERVALENT BONDING', THIS MUST BE A REAL N ANIO
           SYMB(IATOM)='NM'
           PartlQ(IATOM)=-1.
           GOTO 1200
        ELSE
           !     ERROR IF STILL HERE
           if(wrnlev.ge.2) WRITE(OUTU,225) QNAME(IATOM)
           !            IF(IZ.NE.0) WRITE(IZ,225) QNAME(IATOM)
225        FORMAT(' NITROGEN',1X,A,' MAKES SINGLE BONDS TO' &
                ,' TWO NEIGBORS, BUT '/ &
                ' MMFF CANNOT ASSIGN THE ATOM TYPE.  SORRY'/)
           GOTO 1399
        ENDIF
     ENDIF
     !   TERMINAL NITROGENS
     IF(KATTCH.EQ.1) THEN
        IF(KBONDS.EQ.3) THEN
           !                             NITRILE NITROGEN
           SYMB(IATOM)='NSP'
           GOTO 1200
        ENDIF
        IF(KATTCH.EQ.1.AND.KN.EQ.1.AND.KNB.EQ.2) THEN
           !         MAY BE A TERMINAL AZIDO OR DIAZO NITROGEN
           !         FIND AND CHARACTERIZE THE ATTACHED N
           INITR=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
           CALL BINFO(INITR,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LN.EQ.2.AND.LNB.EQ.4) THEN
              SYMB(IATOM)='NAZT'
              GOTO 1200
           ELSE IF(LN.EQ.1.AND.LNB.EQ.2.AND.LC.EQ.1.AND.LCB.EQ.2) THEN
              SYMB(IATOM)='NAZT'
              GOTO 1200
           ENDIF
        ENDIF
        !     ERROR IF STILL HERE
        if(wrnlev.ge.2) WRITE(OUTU,230) QNAME(IATOM)
        !       IF(IZ.NE.0) WRITE(IZ,230) QNAME(IATOM)
230     FORMAT(' NITROGEN',1X,A,' IS BONDED TO ONLY ONE' &
             ,' NEIGHBOR, BUT NOT VIA A TRIPLE BOND'/' SORRY, BUT' &
             ,' MMFF CANNOT HANDLE THIS SITUATION')
        GOTO 1399
     ENDIF
600  CONTINUE
     !  ----  CARBON  ------------------------------------------------
     IF(ATNUM(IATOM).NE.6) GOTO 800
     IF(KATTCH.EQ.1) THEN
        IF(NBND3(IATOM,ITAB).EQ.1) THEN
           IF(KN.EQ.1) THEN
              !              ONE TRIPLY-BONDED N ATTACHED: CLASSIFY AS CSP IN
              !              AN ISONITRILE
              SYMB(IATOM)='C%'
              GOTO 1200
           ENDIF
        ENDIF
        if(wrnlev.ge.2) WRITE(OUTU,610) QNAME(IATOM)
        !       IF(IZ.NE.0) WRITE(IZ,610) QNAME(IATOM)
610     FORMAT(' CARBON',1X,A,' IS BONDED TO ONLY ONE OTHER' &
             ,' ATOM'/' SORRY, BUT MMFF CANNOT HANDLE THIS ' &
             ,' SITUATION')
        GOTO 1399
     ENDIF
     IF(KATTCH.EQ.2) THEN
        !                        MUST BE AN ACETYLENIC OR ALLENIC CARBON
        IF(NBND3(IATOM,ITAB).EQ.1) THEN
           !                                       ACETYLENIC
           SYMB(IATOM)='CSP'
           GOTO 1200
        ENDIF
        IF(NBND2(IATOM,ITAB).EQ.2) THEN
           !                                       ALLENIC
           SYMB(IATOM)='=C='
           GOTO 1200
        ENDIF
        !     IF STILL HERE, THIS CARBON COULD NOT BE TYPED
        if(wrnlev.ge.2) WRITE(OUTU,630) QNAME(IATOM)
        !       IF(IZ.NE.0) WRITE(IZ,630) QNAME(IATOM)
630     FORMAT(' CARBON',1X,A,' HAS TWO NEIGHBORS, BUT' &
             ,' MMFF COULD NOT '/' SUCCESSFULLY ASSIGN ITS' &
             ,' ATOM TYPE.   SORRY.')
        GOTO 1399
     ENDIF
     IF(KATTCH.EQ.3.AND.KBONDS.EQ.4) THEN
        !                        AN SP2 CARBON OF SOME TYPE
        IF(NBNDX(IATOM,NATOM,ITAB,ATNUM,6,2).NE.0) THEN
           !                        ASSIGN DEFAULT TYPE OF C=C IF DOUBLY
           !                        BONDED TO ANOTHER CARBON
           SYMB(IATOM)='C=C'
           !
        ELSE IF(NBNDX(IATOM,NATOM,ITAB,ATNUM,7,2).NE.0) THEN
           !                        ASSIGN SEMIGENERIC TYPE OF C=N
           SYMB(IATOM)='C=N'
           !
        ELSE IF(NBNDX(IATOM,NATOM,ITAB,ATNUM,8,2).NE.0) THEN
           !                        ASSIGN SEMIGENERIC TYPE OF C=O
           SYMB(IATOM)='C=O'
           !
        ELSE IF(NBNDX(IATOM,NATOM,ITAB,ATNUM,15,2).NE.0) THEN
           !                        ASSIGN SEMIGENERIC TYPE OF C=P
           SYMB(IATOM)='C=P'
           !
        ELSE IF(NBNDX(IATOM,NATOM,ITAB,ATNUM,16,2).NE.0) THEN
           !                        ASSIGN SEMIGENERIC TYPE OF C=S
           SYMB(IATOM)='C=S'
        ELSE
           !                        OTHERWISE ASSIGN GENERIC DEFAULT TYPE OF CSP2
           SYMB(IATOM)='CSP2'
        ENDIF
        IF(KPB.EQ.KPA+1) THEN
           !                        CARBON DOUBLY BONDED TO PHOSPHORUS
           SYMB(IATOM)='C=P'
           GOTO 1200
        ENDIF
        IF(KS.EQ.1.AND.KSB.EQ.2) THEN
           !     FIND AND CHARACERIZE THE ATTACHED SULFUR
           ISULF=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           CALL BINFO(ISULF,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LATTCH.EQ.3.AND.LBONDS.EQ.4.AND.LO.EQ.2) THEN
              !                   CARBON DOUBLY BONDED TO SULFUR WHICH IN TURN
              !                   IS DI-OXYGENATED
              SYMB(IATOM)='CSO2'
              GOTO 1200
           ELSE IF(LATTCH.EQ.2.AND.LBONDS.EQ.4.AND.LOB.EQ.2) THEN
              !                    THE CARBON IS DOUBLY BONDED TO SULFUR, WHICH IN
              !                    TURN INS DOUBLY BONDED TO OXYGEN
              SYMB(IATOM)='CS=O'
              GOTO 1200
           ENDIF
        ENDIF
        IF(KS.EQ.2) THEN
           !                        THIOACID, ESTER OR ANION
           !         FIND THE TWO SULFURS
           ISULF1=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           ISULF2=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,ISULF1,0)
           !                        CHARACTERIZE THEN
           CALL BINFO(ISULF1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           CALL BINFO(ISULF2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(LATTCH.EQ.1.AND.JATTCH.EQ.1) THEN
              !                              ISULF1 AND ISULF2 ARE TERMINAL SULFURS
              !                              SO THIS IS A THIOCARBOXYLATE ANION
              SYMB(IATOM)='CS2M'
              !
              !   ALSO TYPE THE ATTACHED SULFURS - STRICTLY SPEAKING, THE SULFUR-
              !   TYPING CODE SHOULD HANDLE THEM, BUT THIS APPROACH "PLAYS IT SAFE"
              !
              SYMB(ISULF1)='S2CM'
              PartlQ(ISULF1)=-0.5
              CALL TRSYMB(ISULF1)
              SYMB(ISULF2)='S2CM'
              PartlQ(ISULF2)=-0.5
              CALL TRSYMB(ISULF2)
              GOTO 1200
           ELSE IF(LATTCH+JATTCH.EQ.3.AND.LBONDS+JBONDS.EQ.4) THEN
              !                             ONE IS S=C, THE OTHER -S-C SO THIS
              !                             IS A THIOACID OR ESTER OR AMIDE
              SYMB(IATOM)='CSS'
              IF(KN.EQ.1.AND.KNB.EQ.1) THEN
                 !
                 !                MAKE SURE THIS IS A SATURATED NITROGEN
                 !
                 INITR=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
                 CALL BINFO(INITR,NATOM,ATNUM,ITAB, &
                      LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
                 IF(LATTCH.EQ.LBONDS.AND.LATTCH.EQ.3) THEN
                    SYMB(IATOM)='C=SN'
                 ENDIF
              ENDIF
              GOTO 1200
           ELSE IF(LATTCH+JATTCH.EQ.4.AND.LBONDS+JBONDS.EQ.6) THEN
              IF((LO.EQ.1.AND.LOB.EQ.2) .OR. (JO.EQ.1.AND.JOB.EQ.2)) THEN
                 !
                 !      THIS CARBON IS ALSO DOUBLY BONDED TO SULFUR WHICH IN TURN IS
                 !      DOUBLY BONDED TO OXYGEN (IT'S ALSO SINGLY BONDED TO A SECOND,
                 !      DIVALENT, SULFUR; THE TWO TOGETHER MAKE 6 BONDS TO 4 ATOMS)
                 !
                 SYMB(IATOM)='CS=O'
                 GOTO 1200
              ENDIF
           ENDIF
        ENDIF
        IF(KS.EQ.3) THEN
           !                        THIOACID, ESTER OR ANION
           NTSF=NTERMA(IATOM,NATOM,ITAB,ATNUM,16)
           IF(NTSF.EQ.2) THEN
              !                           THERE ARE TWO ARE TERMINAL SULFURS
              !                              SO THIS IS A THIOCARBOXYLATE ANION
              SYMB(IATOM)='CS2M'
           ELSE IF(NTSF.EQ.1) THEN
              SYMB(IATOM)='CSS'
           ENDIF
           GOTO 1200
        ENDIF
        IF(KO.EQ.0) THEN
           !                        NO ATTACHED OXYGEN(S)
           IF(KN.EQ.0) THEN
              !                          ALSO NO NITROGEN(S)
              IF(NBNDX(IATOM,NATOM,ITAB,ATNUM,6,2).NE.0) THEN
                 !                          DOUBLE BONDED TO ANOTHER CARBON
                 SYMB(IATOM)='C=C'
                 GOTO 1200
              ENDIF
           ELSE IF(KN.EQ.1) THEN
              !                               ATTACHED TO ONE NITROGEN
              IF(KNB.EQ.2) THEN
                 !                          AND DOUBLY-BONDED TO IT: IMINE
                 SYMB(IATOM)='C=N'
                 GOTO 1200
              ELSE IF(KCB.EQ.KC+1) THEN
                 !                          BUT DOUBLY-BONDED TO CARBON: ENAMINE
                 SYMB(IATOM)='C=C'
                 GOTO 1200
              ELSE IF(KSB.EQ.KS+1) THEN
                 !                          BUT DOUBLY BONDED TO S: THIOAMIDE
                 !
                 !                MAKE SURE THIS IS A SATURATED NITROGEN
                 !
                 INITR=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
                 CALL BINFO(INITR,NATOM,ATNUM,ITAB, &
                      LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
                 IF(LATTCH.EQ.LBONDS.AND.LATTCH.EQ.3) THEN
                    SYMB(IATOM)='C=SN'
                 ENDIF
                 GOTO 1200
              ELSE
                 !
                 !      IF STILL HERE, USE THE PREVIOUSLY "SEMIGENERIC" TYPE OF
                 !      C=S, C=N, C=O, C=C OR C=P, OR ASSIGN GENERIC TYPE OF CSP2
                 !
                 IF(SYMB(IATOM).EQ.' ') SYMB(IATOM)='CSP2'
                 GOTO 1200
              ENDIF
           ELSE IF(KN.EQ.2.AND.KNB.EQ.2) THEN
              !                               ATTACHED TO TWO NITROGENS
              IF(KSB.EQ.KS+1) THEN
                 !                          BUT DOUBLY BONDED TO S: THIOAMIDE
                 SYMB(IATOM)='C=SN'
                 GOTO 1200
              ELSE
                 !
                 !      IF STILL HERE, USE THE PREVIOUSLY "SEMIGENERIC" TYPE OF
                 !      C=S, C=N, C=O, C=C OR C=P, OR ASSIGN GENERIC TYPE OF CSP2
                 !
                 IF(SYMB(IATOM).EQ.' ') SYMB(IATOM)='CSP2'
                 GOTO 1200
              ENDIF
           ELSE IF(KN.EQ.2.AND.KNB.EQ.3) THEN
              !                                            WE HAVE N-C=N OR N-C=N+
              !                                            TYPE THE NITROGENS
              NIT1=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
              NIT2=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,1)
              CALL BINFO(NIT1,NATOM,ATNUM,ITAB, &
                   JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
              CALL BINFO(NIT2,NATOM,ATNUM,ITAB, &
                   LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
              IF(JATTCH.EQ.3.AND.LATTCH.EQ.3) THEN
                 !                                    ALL N'S HAVE THREE NEIGHBORS,
                 !                                    SO THIS MAY BE NCN+
                 NTOX1=NTERMA(NIT1,NATOM,ITAB,ATNUM,8)
                 NTOX2=NTERMA(NIT2,NATOM,ITAB,ATNUM,8)
                 IF(NTOX1.EQ.0.AND.NTOX2.EQ.0.AND. &
                      .NOT.IFNPDP(NIT1).AND..NOT.IFNPDP(NIT2)) THEN
                    SYMB(IATOM)='CNN+'
                    !          TYPE THE NITROGENS, TOO
                    PartlQ(NIT1)=1./2.
                    PartlQ(NIT2)=1./2.
                    SYMB(NIT1)='NCN+'
                    SYMB(NIT2)='NCN+'
                    CALL TRSYMB(NIT1)
                    CALL TRSYMB(NIT2)
                    GOTO 1200
                 ELSE
                    SYMB(IATOM)='C=N'
                    GOTO 1200
                 ENDIF
              ELSE
                 SYMB(IATOM)='C=N'
                 GOTO 1200
              ENDIF
           ELSE IF(KN.EQ.3.AND.KNB.EQ.4) THEN
              !                                            GUANIDINE OR GUANIDINIUM
              !                                            CHARACTERIZE THE NITROGENS
              NIT1=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
              NIT2=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,1)
              NIT3=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,2)
              CALL BINFO(NIT1,NATOM,ATNUM,ITAB, &
                   JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
              CALL BINFO(NIT2,NATOM,ATNUM,ITAB, &
                   LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
              CALL BINFO(NIT3,NATOM,ATNUM,ITAB, &
                   MH,MX,MATTCH,MBONDS,MC,MCB,MN,MNB,MO,MOB,MS,MSB,MPA,MPB)
              IF(JATTCH.EQ.3.AND.LATTCH.EQ.3.AND.MATTCH.EQ.3) THEN
                 !                                    ALL N'S HAVE THREE NEIGHBORS,
                 !                                    SO THIS IS GUANIDINIUM
                 SYMB(IATOM)='CGD+'
                 !          TYPE THE NITROGENS, TOO
                 PartlQ(NIT1)=1./3.
                 PartlQ(NIT2)=1./3.
                 PartlQ(NIT3)=1./3.
                 SYMB(NIT1)='NGD+'
                 SYMB(NIT2)='NGD+'
                 SYMB(NIT3)='NGD+'
                 CALL TRSYMB(NIT1)
                 CALL TRSYMB(NIT2)
                 CALL TRSYMB(NIT3)
                 GOTO 1200
              ELSE IF(   JATTCH+LATTCH+MATTCH == 8 &
                   .AND. JBONDS+LBONDS+MBONDS == 9) THEN
                 !                ONE N HAS ONLY TWO NEIGHBORS BUT IS TRIVALENT,
                 !                SO THIS IS GUANIDINE
                 SYMB(IATOM)='CGD'
                 GOTO 1200
                 !
              ELSE IF(   JATTCH+LATTCH+MATTCH == 8 &
                   .AND. JBONDS+LBONDS+MBONDS == 10) THEN
                 !
                 !   THIS IS REALLY A NCN+ SITUATION WITH A THIRD ATTACHED NITROGEN WHICH
                 !   MAKES A DOUBLE BOND TO SOME OTHR ATOM.  WHICH NITROGEN IS THAT?
                 IF(JATTCH.EQ.2.AND.JBONDS.EQ.3) THEN
                    N1=NIT2
                    N2=NIT3
                 ELSE IF(LATTCH.EQ.2.AND.LBONDS.EQ.3) THEN
                    N1=NIT1
                    N2=NIT3
                 ELSE IF(MATTCH.EQ.2.AND.MBONDS.EQ.3) THEN
                    N1=NIT1
                    N2=NIT2
                 ENDIF
                 NTOX1=NTERMA(N1,NATOM,ITAB,ATNUM,8)
                 NTOX2=NTERMA(N2,NATOM,ITAB,ATNUM,8)
                 NCNORD=IBORDR(IATOM,N1)+IBORDR(IATOM,N2)
                 IF(NTOX1.EQ.0.AND.NTOX2.EQ.0.AND.NCNORD.EQ.3.AND. &
                      .NOT.IFNPDP(N1).AND..NOT.IFNPDP(N2)) THEN
                    SYMB(IATOM)='CNN+'
                    !          TYPE THE NITROGENS, TOO
                    PartlQ(N1)=1./2.
                    PartlQ(N2)=1./2.
                    SYMB(N1)='NCN+'
                    SYMB(N2)='NCN+'
                    CALL TRSYMB(N1)
                    CALL TRSYMB(N2)
                    GOTO 1200
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        IF(KO.EQ.1.AND.KOB.EQ.1.AND.KN.EQ.2.AND.KNB.EQ.3) THEN
           !                                            WE HAVE N-C=N OR N-C=N+
           !                                            TYPE THE NITROGENS
           NIT1=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
           NIT2=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,1)
           CALL BINFO(NIT1,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           CALL BINFO(NIT2,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(JATTCH.EQ.3.AND.LATTCH.EQ.3) THEN
              !                                    ALL N'S HAVE THREE NEIGHBORS,
              !                                    SO THIS MAY BE NCN+
              NTOX1=NTERMA(NIT1,NATOM,ITAB,ATNUM,8)
              NTOX2=NTERMA(NIT2,NATOM,ITAB,ATNUM,8)
              IF(NTOX1.EQ.0.AND.NTOX2.EQ.0.AND. &
                   .NOT.IFNPDP(NIT1).AND..NOT.IFNPDP(NIT2)) THEN
                 SYMB(IATOM)='CNN+'
                 !          TYPE THE NITROGENS, TOO
                 PartlQ(NIT1)=1./2.
                 PartlQ(NIT2)=1./2.
                 SYMB(NIT1)='NCN+'
                 SYMB(NIT2)='NCN+'
                 CALL TRSYMB(NIT1)
                 CALL TRSYMB(NIT2)
                 GOTO 1200
              ENDIF
           ENDIF
        ELSE IF(KO.EQ.1.AND.KOB.EQ.1) THEN
           !                                CARBON IN ENOL OR ENOL ETHER?
           IF(KNB.EQ.KN.AND.KCB.EQ.KC+1) THEN
              !                                C=C PRESENT AND ANY C-N BONDS ARE SINGL
              SYMB(IATOM)='C=C'
              GOTO 1200
           ELSE IF(KNB.EQ.KN+1) THEN
              !                                   DOUBLY BONDED TO N
              SYMB(IATOM)='C=N'
              GOTO 1200
           ELSE IF(KSB.EQ.KS+1) THEN
              !                                  DOUBLY BONDED TO SULFUR
              SYMB(IATOM)='C=S'
              IF(KN.EQ.1.AND.KNB.EQ.1) THEN
                 !                MAKE SURE THIS IS A SATURATED NITROGEN
                 !
                 INITR=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
                 CALL BINFO(INITR,NATOM,ATNUM,ITAB, &
                      LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
                 IF(LATTCH.EQ.LBONDS.AND.LATTCH.EQ.3) THEN
                    SYMB(IATOM)='C=SN'
                 ENDIF
              ENDIF
              GOTO 1200
           ENDIF
        ENDIF
        IF(KO.EQ.1.AND.KOB.EQ.2) THEN
           IF(KH+KC.EQ.2) THEN
              !                               ALDEHYDE OR KETONE
              SYMB(IATOM)='C=OR'
              GOTO 1200
           ENDIF
           IF(KN.NE.0) THEN
              IF(KN.EQ.1) SYMB(IATOM)='C=ON'
              !                                 AMIDE
              IF(KN.EQ.2) SYMB(IATOM)='CONN'
              !                                 UREA
              GOTO 1200
           ENDIF
           IF(KH+KC.EQ.1) THEN
              IF(KS.EQ.1) THEN
                 !                            THIOESTER
                 SYMB(IATOM)='C=OS'
                 GOTO 1200
              ENDIF
              !     IF STILL HERE, WE HAVE SOME OTHER CARBONYL - TYPE AS CO
              SYMB(IATOM)='C=O'
              GOTO 1200
           ENDIF
        ENDIF
        IF(KS.EQ.2) THEN
           !
           !      -S-C(=O)-S- FRAGMENT
           !
           SYMB(IATOM)='C=OS'
           GOTO 1200
        ENDIF
        IF(KO.EQ.2) THEN
           !                        ACID, ESTER, OR CARBOXYLATE ANION
           !         FIND THE TWO OXYGENS
           IOX1=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,0,0)
           IOX2=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,IOX1,0)
           !                        CHARACTERIZE THEN
           CALL BINFO(IOX1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           CALL BINFO(IOX2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(LATTCH.EQ.1.AND.JATTCH.EQ.1) THEN
              !                              IOX1 AND IOX2 ARE TERMINAL OXYGENS,
              !                              SO THIS IS A CARBOXYLATE ANION
              SYMB(IATOM)='CO2M'
              GOTO 1200
           ENDIF
           IF(LATTCH+JATTCH.EQ.3.AND.LBONDS+JBONDS.EQ.4) THEN
              !                             ONE IS O=C, THE OTHER -O-C
              !                             SO THIS IS AN ACID OR AN ESTER
              SYMB(IATOM)='COO'
              IF(KN.EQ.1) SYMB(IATOM)='C=ON'
              !                                 AMIDE
              GOTO 1200
           ENDIF
           IF(SYMB(IATOM).NE.' ') GOTO 1200
           !   IF STILL HERE, SOMETHING IS WRONG
           if(wrnlev.ge.2) WRITE(OUTU,640) QNAME(IATOM)
           !         IF(IZ.NE.0) WRITE(IZ,640) QNAME(IATOM)
640        FORMAT(' CARBON',1X,A,' HAS TWO ATTACHED OXYGENS', &
                /' BUT COULD NOT BE SUCCESSFULLY TYPED BY MMFF.  SORRY')
           GOTO 1399
        ENDIF
        IF(KN.EQ.1.AND.KOB.EQ.KO+1) THEN
           !                        CARBAMATE -N-CO-O-
           SYMB(IATOM)='COON'
           GOTO 1200
        ENDIF
        IF(KO.EQ.3) THEN
           !                        CARBONIC ACID OR ESTER
           SYMB(IATOM)='COOO'
           GOTO 1200
        ENDIF
        !
        !      IF STILL HERE, USE THE PREVIOUSLY "SEMIGENERIC" TYPE OF
        !      C=S, C=N, C=O, C=C OR C=P, OR ASSIGN GENERIC TYPE OF CSP2
        !
        IF(SYMB(IATOM).EQ.' ') SYMB(IATOM)='CSP2'
        GOTO 1200
     ENDIF
     IF(KATTCH.EQ.4) THEN
        !                        A SATURATED CARBON - TETRAHEDRAL
        SYMB(IATOM)='CR'
        !                  CYCLOPROPYL OR CYCLOBUTYL ASSIGNMENT, IF RELEVANT,
        !                  WILL BE MADE DURING THE RING PERCEPTION PHASE
        GOTO 1200
     ENDIF
800  CONTINUE
     IF(ATNUM(IATOM).NE.16) GOTO 900
     ! ----  SULFUR ---------------------------------------------------
     IF(KATTCH.EQ.1) THEN
        !                            A TERMINAL SULFUR
        IF(KC.EQ.1.AND.KCB.EQ.1) THEN
           !                        TERMINAL SULFUR ATTACHED TO CARBON
           ICARB=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,0)
           CALL BINFO(ICARB,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LS.EQ.1) THEN
              !                          TERMINAL SULFUR, NEGATIVELY CHARGED
              SYMB(IATOM)='SM'
              PartlQ(IATOM)=-1.
              GOTO 1200
           ELSE IF(LS.EQ.2) THEN
              !                          TERMINAL SULFUR IN -CS2(-) OR -CS2(H,R)
              !                          FIND NUMBER OF TERMINAL S'S ON ICARB
              ITERM=NTERMA(ICARB,NATOM,ITAB,ATNUM,16)
              IF(ITERM.EQ.2) THEN
                 !                               SECOND SULFUR IS ALSO TERMINAL
                 SYMB(IATOM)='S2CM'
                 PartlQ(IATOM)=-0.5
                 GOTO 1200
              ELSE
                 !                               ONLY IATOM IS A TERMINAL SULFUR
                 SYMB(IATOM)='SM'
                 PartlQ(IATOM)=-1.
                 GOTO 1200
              ENDIF
           ENDIF
        ENDIF
        IF(KC.EQ.1.AND.KCB.EQ.2) THEN
           ICARB=IATTCH(IATOM,NATOM,ATNUM,ITAB,6,0,0)
           ITERM=NTERMA(ICARB,NATOM,ITAB,ATNUM,16)
           IF(ITERM.EQ.1) THEN
              SYMB(IATOM)='S=C'
              GOTO 1200
           ELSE IF(ITERM.EQ.2) THEN
              SYMB(IATOM)='S2CM'
              PartlQ(IATOM)=-0.5
              GOTO 1200
           ENDIF
        ENDIF
        IF(KN.EQ.1.AND.KNB.EQ.2) THEN
           !                        TERMINAL SULFUR DOUBLY BONDED TO NITROGEN
           SYMB(IATOM)='S=N'
           GOTO 1200
        ENDIF
        IF(KPA.EQ.1) THEN
           !                        TERMINAL SULFUR ATTACHED TO PHOSPHORUS
           IPHOS=IATTCH(IATOM,NATOM,ATNUM,ITAB,15,0,0)
           !         CHARACTERIZE BONDING TO IPHOS
           CALL BINFO(IPHOS,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           NTS=NTERMA(IPHOS,NATOM,ITAB,ATNUM,16)
           XNTS=NTS
           IF(NTS.EQ.1) THEN
              !                          "PHOSPHOXID" SULFUR
              SYMB(IATOM)='S-P'
              !            PartlQ(IATOM)=-(NTS-LATTCH+3)/XNTS
              PartlQ(IATOM)=0.
              GOTO 1200
           ENDIF
        ENDIF
        IF(KS.EQ.1) THEN
           !                                      TERMINAL SULFUR ATTACHED TO SULFU
           !   FIND AND CHARACTERIZE THE ATTACHED SULFUR
           ISULF=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           !                        CHARACTERIZE THEN
           CALL BINFO(ISULF,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LATTCH.EQ.3.AND.LO.EQ.1.AND.LOB.EQ.1) THEN
              SYMB(IATOM)='SSMO'
              PartlQ(IATOM)=-0.5
              GOTO 1200
           ENDIF
        ENDIF
     ENDIF
     IF(KATTCH.EQ.2.AND.KBONDS.EQ.2) THEN
        !                        THIOL OR SULFIDE
        SYMB(IATOM)='S'
        GOTO 1200
     ENDIF
     IF(KATTCH.EQ.2.AND.KBONDS.EQ.4) THEN
        IF(KC.EQ.1.AND.KCB.EQ.2.AND.KO.EQ.1.AND.KOB.EQ.2) THEN
           !                    SULFINYL SULFUR (O=S=C<)
           SYMB(IATOM)='=S=O'
           GOTO 1200
        ENDIF
     ENDIF
     IF(KATTCH.EQ.3.AND.KOB.EQ.KO+1) THEN
        !                                    SULFOXIDE
        SYMB(IATOM)='S=O'
        GOTO 1200
     ENDIF
     IF(KATTCH.EQ.3) THEN
        IF(KBONDS.EQ.4.AND.KC.EQ.1.AND.KCB.EQ.2.AND.KO.EQ.2) THEN
           !                                   C=SO2 TYPE FOR SULFUR
           SYMB(IATOM)='=SO2'
           GOTO 1200
        ELSE IF(KBONDS.EQ.4.AND.KNB.EQ.KN+1) THEN
           !                                   SULFUR IN >S=N-
           SYMB(IATOM)='>S=N'
           GOTO 1200
        ELSE IF(KBONDS.EQ.3.AND.KO.EQ.2) THEN
           !   FIND AND CHARACTERIZE THE ATTACHED OXYGENS
           IOX1=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,0,0)
           IOX2=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,IOX1,0)
           !                        CHARACTERIZE THEN
           CALL BINFO(IOX1,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           CALL BINFO(IOX2,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(LATTCH.EQ.1.AND.JATTCH.EQ.1) THEN
              SYMB(IATOM)='SO2M'
              !                      SULFINATE SULFUR IN R-SO2(-1)
              !                      OXYGENS OF TYPE 'O2S' CARRY FORMAL Q'S OF -0.5
              GOTO 1200
           ENDIF
        ELSE IF(KBONDS.EQ.3.AND.KOB.EQ.1.AND.KSB.EQ.1) THEN
           !   FIND AND CHARACTERIZE THE ATTACHED OXYGEN AND SULFUR
           IOX=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,0,0)
           ISULF=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           !                        CHARACTERIZE THEN
           CALL BINFO(IOX,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           CALL BINFO(ISULF,NATOM,ATNUM,ITAB, &
                JH,JX,JATTCH,JBONDS,JC,JCB,JN,JNB,JO,JOB,JS,JSB,JPA,JPB)
           IF(LATTCH.EQ.1.AND.JATTCH.EQ.1) THEN
              SYMB(IATOM)='SSOM'
              !                      TRICOORD THIOSULFINATE SULFUR IN R-SSO(-1)
              !                      THE TERMINAL OXYGEN AND SULFUR SHOULD CARRY
              !                      FORMAL Q'S OF -0.5; THEIR SYMBOLIC ATOM TYPES
              !                      ARE 'OSMS' AND 'SSMO', RESPECTIVELY
              GOTO 1200
           ENDIF
        ENDIF
     ENDIF
     IF(KATTCH.EQ.4) THEN
        !                        SULFATE, SULFONAMIDE, SULFONATE, ETC
        IF(KN.EQ.1.AND.KO.EQ.1) THEN
           !         THIS IS AN ALCOHOL OR  ENOL OR CARBOXYLIC ACID OXYGEN
           !   FIND AND CHARACTERIZE THE ATTACHED NITROGEN
           INITR=IATTCH(IATOM,NATOM,ATNUM,ITAB,7,0,0)
           CALL BINFO(INITR,NATOM,ATNUM,ITAB, &
                LH,LX,LATTCH,LBONDS,LC,LCB,LN,LNB,LO,LOB,LS,LSB,LPA,LPB)
           IF(LATTCH.EQ.2) THEN
              !                NITROGEN ANALOG OF A SULFONE
              SYMB(IATOM)='SNO'
              GOTO 1200
           ENDIF
        ENDIF
        IF(KN.GE.1.AND.KO.EQ.2) THEN
           !                                    SULFONAMIDE
           SYMB(IATOM)='SO2N'
           GOTO 1200
        ENDIF
        IF(KO.EQ.2) THEN
           !                         SULFONE
           SYMB(IATOM)='SO2'
           GOTO 1200
        ENDIF
        IF(KO.EQ.3) THEN
           !                         SULFONATE
           SYMB(IATOM)='SO3'
           GOTO 1200
        ENDIF
        IF(KO.EQ.4) THEN
           !                        SULFATE
           SYMB(IATOM)='SO4'
           GOTO 1200
        ENDIF
     ENDIF
     !   TYPING HAS FAILED IF STILL HERE
     !
     if(wrnlev.ge.2) WRITE(OUTU,810) QNAME(IATOM),KATTCH
     !     IF(IZ.NE.0) WRITE(IZ,810) QNAME(IATOM),KATTCH
810  FORMAT(' SULFUR',1X,A,' HAS',I3,' NEIGHBORS, BUT' &
          ,' COULD NOT'/' BE SUCCESSFULLY TYPED BY MMFF.', &
          '   SORRY.')
     GOTO 1399
900  CONTINUE
     IF(ATNUM(IATOM).NE.15) GOTO 1000
     !  ----  PHOSPHOROUS  ---------------------------------------------
     IF(KATTCH.EQ.4) SYMB(IATOM)='PTET'
     IF(KO.EQ.4.AND.KATTCH.EQ.4) THEN
        !                      PHOSPHATE OR PHOSPHORIC ESTER
        SYMB(IATOM)='PO4'
        GOTO 1200
     ENDIF
     IF(KO.EQ.3.AND.KATTCH.EQ.4) THEN
        !                      PHOSPHONATE OR PHOSPHONIC ESTER
        SYMB(IATOM)='PO3'
        GOTO 1200
     ENDIF
     IF(KO.EQ.2.AND.KATTCH.EQ.4) THEN
        !                      PHOSPHINIC ACID OR ESTER
        SYMB(IATOM)='PO2'
        GOTO 1200
     ENDIF
     ! T.A Halgren change
     IF(KO.EQ.1.AND.KS.EQ.1.AND.KATTCH.EQ.4) THEN
        !                       MIXED THIOPHOSPHATE?
        NTOX=NTERMA(IATOM,NATOM,ITAB,ATNUM,8)
        NTSF=NTERMA(IATOM,NATOM,ITAB,ATNUM,16)
        IF(NTOX.EQ.1.AND.NTSF.EQ.1) THEN
           IOX=IATTCH(IATOM,NATOM,ATNUM,ITAB,8,0,0)
           ISULF=IATTCH(IATOM,NATOM,ATNUM,ITAB,16,0,0)
           SYMB(IATOM)='PO2'
           SYMB(IOX)='O2P'
           SYMB(ISULF)='SM'
           PARTLQ(IOX)=-0.5
           PARTLQ(ISULF)=-0.5
           CALL TRSYMB(IOX)
           CALL TRSYMB(ISULF)
           GO TO 1200
        ENDIF
     ENDIF
     ! End T.A. Halgren charge
     IF(KO.EQ.1.AND.KATTCH.EQ.4) THEN
        !                                     PHOSPHOXIDE
        SYMB(IATOM)='PO'
        GOTO 1200
     ENDIF
     IF(SYMB(IATOM).EQ.'PTET') GOTO 1200
     IF(KATTCH.EQ.3) THEN
        !                             PHOSPHINE
        SYMB(IATOM)='P'
        GOTO 1200
     ENDIF
     IF(KATTCH.EQ.2.AND.KBONDS.EQ.3) THEN
        IF(KC.EQ.2.AND.KCB.EQ.3) THEN
           !                             PHOSPHOROUS IN C-P=C
           SYMB(IATOM)='-P=C'
           GOTO 1200
        ELSE IF(KC.EQ.1.AND.KCB.EQ.2) THEN
           !                             NOT C-P=C, BUT P IS DOUBLE BONDED TO C
           SYMB(IATOM)='-P=C'
           GOTO 1200
        ENDIF
     ENDIF
     !  IF STILL HERE, TYPING FOR PHOSPHOROUS HAS FAILED
     if(wrnlev.ge.2) WRITE(OUTU,910) QNAME(IATOM),KATTCH
     !     IF(IZ.NE.0) WRITE(IZ,910) QNAME(IATOM),KATTCH
910  FORMAT(' PHOSPHOROUS',1X,A,' HAS', I3,' NEIGHBORS,' &
          ,' BUT COULD NOT BE'/' SUCCESSFULLY TYPED BY MMFF. ' &
          ,'    SORRY.')
     GOTO 1399
1000 CONTINUE
     !  ---- OTHER ATOM TYPES (SILICON, HALOGEN) ---------------------------
     IF(ATNUM(IATOM).EQ.9) THEN
        IF(KATTCH.EQ.1) THEN
           SYMB(IATOM)='F'
           GOTO 1200
        ELSE IF(KATTCH.EQ.0) THEN
           SYMB(IATOM)='F-'
           PartlQ(IATOM)=-1.
           GOTO 1200
        ENDIF
     ENDIF
     IF(ATNUM(IATOM).EQ.14) SYMB(IATOM)='SI'
     IF(ATNUM(IATOM).EQ.17) THEN
        IF(KATTCH.EQ.1) THEN
           SYMB(IATOM)='CL'
           GOTO 1200
        ELSE IF(KATTCH.EQ.0) THEN
           SYMB(IATOM)='CL-'
           PartlQ(IATOM)=-1.
        ELSE IF(KO.EQ.4) THEN
           SYMB(IATOM)='CLO4'
           GOTO 1200
        ENDIF
     ENDIF
     IF(ATNUM(IATOM).EQ.32) SYMB(IATOM)='GE'
     IF(ATNUM(IATOM).EQ.33) SYMB(IATOM)='AS'
     IF(ATNUM(IATOM).EQ.34) SYMB(IATOM)='SE'
     IF(ATNUM(IATOM).EQ.35) THEN
        IF(KATTCH.EQ.1) THEN
           SYMB(IATOM)='BR'
           GOTO 1200
        ELSE IF(KATTCH.EQ.0) THEN
           SYMB(IATOM)='BR-'
           PartlQ(IATOM)=-1.
           GOTO 1200
        ENDIF
     ENDIF
     IF(ATNUM(IATOM).EQ.50) SYMB(IATOM)='SN'
     IF(ATNUM(IATOM).EQ.51) SYMB(IATOM)='SB'
     IF(ATNUM(IATOM).EQ.52) SYMB(IATOM)='TE'
     IF(ATNUM(IATOM).EQ.53) SYMB(IATOM)='I'
     !
     !    CATIONS
     !
     IF(ATNUM(IATOM).EQ.3) THEN
        SYMB(IATOM)='LI+'
        PartlQ(IATOM)=1.
        GOTO 1200
     ELSE IF(ATNUM(IATOM).EQ.11) THEN
        SYMB(IATOM)='NA+'
        PartlQ(IATOM)=1.
        GOTO 1200
     ELSE IF(ATNUM(IATOM).EQ.12) THEN
        SYMB(IATOM)='MG+2'
        PartlQ(IATOM)=2.
        GOTO 1200
     ELSE IF(ATNUM(IATOM).EQ.19) THEN
        SYMB(IATOM)='K+'
        PartlQ(IATOM)=1.
        GOTO 1200
     ELSE IF(ATNUM(IATOM).EQ.20) THEN
        SYMB(IATOM)='CA+2'
        PartlQ(IATOM)=2.
        GOTO 1200
     ELSE IF(ATNUM(IATOM).EQ.26) THEN
        IF(NICHG(IATOM).EQ.4) THEN
           SYMB(IATOM)='FE+2'
           PartlQ(IATOM)=2.
           GOTO 1200
        ELSE
           SYMB(IATOM)='FE+3'
           PartlQ(IATOM)=3.
           GOTO 1200
        ENDIF
     ELSE IF(ATNUM(IATOM).EQ.29) THEN
        IF(NICHG(IATOM).EQ.1) THEN
           SYMB(IATOM)='CU+1'
           PartlQ(IATOM)=1.
           GOTO 1200
        ELSE
           SYMB(IATOM)='CU+2'
           PartlQ(IATOM)=2.
           GOTO 1200
        ENDIF
     ELSE IF(ATNUM(IATOM).EQ.30) THEN
        SYMB(IATOM)='ZN+2'
        PartlQ(IATOM)=2.
        GOTO 1200
     ENDIF
1200 CONTINUE
     IF(SYMB(IATOM).EQ.'    ') THEN
        if(wrnlev.ge.2) WRITE(OUTU,1210) QNAME(IATOM)
        !       IF(IZ.NE.0) WRITE(IZ,1210) QNAME(IATOM)
1210    FORMAT(' ATOM TYPING FOR ATOM',1X,A,' HAS FAILED.' &
             ,'   SORRY.')
        GOTO 1399
     ENDIF
     !   NOW TRANSLATE THE ATOM SYMB(IATOM) INTO THE MMFF NUMERICAL ATOM TYPE
     !
     CALL TRSYMB(IATOM)
     GOTO 1400
1399 CONTINUE
     IERR=IERR+1
1400 CONTINUE
  ENDDO
  IF(IERR.EQ.0) RETURN
  !  ERROR SECTION - ONE OR MORE NON-HYDROGENS COULD NOT NOT BE TYPED
  WRITE(SCRTCH,'(I5,A)') IERR, &
       ' NON-HYDROGENS COULD NOT BE ASSIGNED MMFF ATOM TYPES'
  !     IF(IZ.NE.0) WRITE(IZ,2100) IERR
  !2100 FORMAT(//' ***** FATAL ERROR *****'
  !    . /I5,' NON-HYDROGENS COULD NOT BE',
  !    . ' ASSIGNED MMFF ATOM TYPES'//' ***** EXECUTION ENDING *****')
  CALL WRNDIE(-5,'<xtype>',SCRTCH(:LEN_TRIM(SCRTCH)))
  !
  RETURN
END SUBROUTINE XTYPE

! ====================================================================
! INTEGER FUNCTION I12DEL: PRUNE RING SYSTEM BY MARKING "BRIDGE" BONDS
! FOR DELETION FROM LISTING USED BY SUBROUTINE RGTYPE
! ====================================================================
INTEGER FUNCTION I12DEL(I,J,MAT1,MAT2,NDELBND)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Written by T. A. Halgren, Merck Research Labs, July 1996
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  use chm_kinds
  implicit none
  integer MAT1(*), MAT2(*), NDELBND
  !
  integer M, I, J
  !
  I12DEL=0
  IF(NDELBND.EQ.0) RETURN
  DO M=1,NDELBND
     IF(I.EQ.MAT1(M).AND.J.EQ.MAT2(M)) THEN
        I12DEL=1
        RETURN
     ELSE IF(I.EQ.MAT2(M).AND.J.EQ.MAT1(M)) THEN
        I12DEL=1
        RETURN
     ENDIF
  ENDDO
  RETURN
END FUNCTION I12DEL
#else /**/
SUBROUTINE NULL_mmfftype
  RETURN
END SUBROUTINE NULL_mmfftype
#endif 


