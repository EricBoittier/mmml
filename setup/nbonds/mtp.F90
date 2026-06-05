module mtp_fcm
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="mtp.src"

!
!   Nuria Plattner october 2008: Atomic Multipole Arrays
!   Arrays for spherical tensor notation
!
!
!    Variable  Index    Purpose
!
!     MOLN              maximal number of atomic multipole molecules
!     MOLS              number of molecules
!     NMOL              number of primary molecules/fragments
!     IMOL              number of image molecules/fragments
!     SITE     mols     number of atoms in the molecule
!     RFNR     mols     reference atoms for the molecule
!     RANK     atom     rank of atomic multipole expansion
!     REFM     atom     reference molecule for each atom
!
!     QMTP              flag for module use
!     QANH     mols     flag for anharmonic bond potential
!     QFLC     mols     flag for fluctuating moments
!     QDUM     mols     flag for dummy atoms
!     LINT     mols     flag for linear triatomic
!
!  parameters for fluctuating moments; only of linear geometry
!
!     CBBEQDI  mols     equilibrium distance (in angstrom)
!     Q1       atom     equilibrium value for fluctuating charge
!     Q2       atom     linear change of fluctuating charge
!     Q1Z1     atom     equilibrium value for fluctuating dipol
!     Q2Z2     atom     linear change of fluctuating dipol
!     Q201     atom     equilibrium value for fluctuating quadrupol
!     Q202     atom     linear change fluctuating quadrupol
!     Q22C1    atom     equilibrium value for fluctuating quadrupol
!     Q22C2    atom     linear change fluctuating quadrupol
!     Q301     atom     equilibrium value for fluctuating octopole
!     Q302     atom     linear change fluctuating octopole
!
!  parameters for atomic multipole moments
!
!     Q1Z      atom     z-dipol component
!     Q1X      atom     x-dipol component
!     Q1Y      atom     y-dipol component
!     Q20      atom     zz quadrupole component
!     Q21c     atom     xz quadrupole component
!     Q21s     atom     yz quadrupole component
!     Q22c     atom     x**2-y**2 quadrupole component
!     Q22s     atom     xy quadrupole component
!     Q30      atom     octopole z-component
!
!  global MTP variables
!
!
!     RAX               X-vector component  = RAXYZ(1)
!     RAY               Y-vector component  = RAXYZ(2)
!     RAZ               Z-vector component  = RAXYZ(3)
!
!     RAX3              3*RAX
!     RAY3              3*RAY
!     RAZ3              3*RAZ
!     RAX5              5*RAX
!     RAY5              5*RAY
!     RAZ5              5*RAZ
!     RAX7              7*RAX
!     RAY7              7*RAY
!     RAZ7              7*RAZ
!     RAX9              9*RAX
!     RAY9              9*RAY
!     RAZ9              9*RAZ
!     RM1,RM2,RM3,RM4,RM5,RM6    1/R,R**-2,R**-3,R**-4,R**-5,R**-6
!
!     DEL0    moln, 3   reference distance for correction term
!     ORTH    2, terms  storage array for rotational correction calculation
!     FORT    2, 9      prefactors for correction term
!
!     TOKB              TOKCAL/BOHRR
!     DEG2RAD           degree -> radian (pi/180)
!

      LOGICAL, SAVE :: QMTP
      LOGICAL, SAVE, ALLOCATABLE :: QANH(:),QFLC(:),QDUM(:),LINT(:)
      INTEGER, SAVE :: MOLN, MOLS,NMOL,IMOL, NUMDA, IMTPIMG, NASTP
      INTEGER, SAVE, ALLOCATABLE :: SITE(:),RFNR(:,:)
      INTEGER, SAVE, ALLOCATABLE :: REFM(:),RANK(:)
      INTEGER, SAVE, ALLOCATABLE :: KDA(:), IPDA(:), IZDA(:,:)
      real(chm_real), save, allocatable :: Q1(:),Q2(:),   &
                      Q1Z1(:),Q1Z2(:), Q201(:), Q202(:),  &
                      Q22C1(:),Q22C2(:), Q301(:), Q302(:)
      real(chm_real), save, allocatable :: Q1Z(:),Q1X(:),Q1Y(:),   &
                      Q20(:),Q21C(:),Q21S(:), Q22C(:),Q22S(:),Q30(:)
      real(chm_real), save :: RAX,RAY,RAZ,RM1,RM2,RM3,RM4,RM5,RM6, &
                      RAX3,RAY3,RAZ3,RAX5,RAY5,RAZ5,               &
                      RAX7,RAY7,RAZ7,RAX9,RAY9,RAZ9,               &
                      ORTH(2,24),FORT(2,9), TOKB, DEG2RAD
      real(chm_real), save, allocatable :: CBBEQDI(:),CBBVE(:),CBBS(:)
      real(chm_real), save, allocatable :: DEL0(:,:), RZDA(:,:), RCDA(:,:)
      real(chm_real),save, allocatable :: TRFX(:,:),TRFY(:,:),TRFZ(:,:)
      real(chm_real),save, allocatable :: ADX(:,:),ADY(:,:),ADZ(:,:)
      real(chm_real),save, allocatable :: ADXS(:,:),ADYS(:,:),ADZS(:,:)

contains

!c calculates the electrostatic energy of atomic multipole expansions
!c
      SUBROUTINE MTP
!C --------------------------------------------------------------------
!c This routine is called if the keyword MTP is provided;
!c The specific options are processed here.
!C --------------------------------------------------------------------
!C
      use chm_kinds
      use stream
      use dimens_fcm
      implicit none
!C
      QMTP = .TRUE.
      IF(PRNLEV.GT.2) WRITE(OUTU,*)' <MTP> ENERGY ROUTINE MTP ACTIVATED'

      CALL MTPINIT()
!c
      RETURN
      END SUBROUTINE MTP
!C
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE MTPINIT
!C reads the atomic multipole parameters 
!C initializes the arrays as needed
!C
      use chm_kinds
      use stream
      use exfunc
      use dimens_fcm
      use comand
      use psf
      use param
      use consta
      use coord
      use string
      use memory
      use parallel,only: psnd4,psnd8
      implicit none
!C
!C local variables
      INTEGER INUNI,REFS,I,J,K,N,MOLCNT,RFRNG(2),REFAT,CNT,IDX,IDY
      INTEGER L,NUMDA2, NATINCM
      real(chm_real) DI, AN
      CHARACTER*4 WRD

      DEG2RAD = datan(1.d0) * 4.d0 / 180.d0
      IMTPIMG = 0   ! becomes 1 if images are used

      ! Dynamic allocation of memory
      call chmalloc('mtp.src','MTPINIT','REFM',MAXAIM,intg=REFM)
      call chmalloc('mtp.src','MTPINIT','RANK',MAXAIM,intg=RANK)
      call chmalloc('mtp.src','MTPINIT','Q1',MAXAIM,crl=Q1)
      call chmalloc('mtp.src','MTPINIT','Q2',MAXAIM,crl=Q2)
      call chmalloc('mtp.src','MTPINIT','Q1Z1',MAXAIM,crl=Q1Z1)
      call chmalloc('mtp.src','MTPINIT','Q1Z2',MAXAIM,crl=Q1Z2)
      call chmalloc('mtp.src','MTPINIT','Q201',MAXAIM,crl=Q201)
      call chmalloc('mtp.src','MTPINIT','Q202',MAXAIM,crl=Q202)
      call chmalloc('mtp.src','MTPINIT','Q22C1',MAXAIM,crl=Q22C1)
      call chmalloc('mtp.src','MTPINIT','Q22C2',MAXAIM,crl=Q22C2)
      call chmalloc('mtp.src','MTPINIT','Q301',MAXAIM,crl=Q301)
      call chmalloc('mtp.src','MTPINIT','Q302',MAXAIM,crl=Q302)
      call chmalloc('mtp.src','MTPINIT','Q1Z',MAXAIM,crl=Q1Z)
      call chmalloc('mtp.src','MTPINIT','Q1X',MAXAIM,crl=Q1X)
      call chmalloc('mtp.src','MTPINIT','Q1Y',MAXAIM,crl=Q1Y)
      call chmalloc('mtp.src','MTPINIT','Q20',MAXAIM,crl=Q20)
      call chmalloc('mtp.src','MTPINIT','Q21C',MAXAIM,crl=Q21C)
      call chmalloc('mtp.src','MTPINIT','Q21S',MAXAIM,crl=Q21S)
      call chmalloc('mtp.src','MTPINIT','Q22C',MAXAIM,crl=Q22C)
      call chmalloc('mtp.src','MTPINIT','Q22S',MAXAIM,crl=Q22S)
      call chmalloc('mtp.src','MTPINIT','Q30',MAXAIM,crl=Q30)

      ! Initialize MTP parameters
      DO I = 1, MAXAIM
         Q1(I) = 0.D0
         Q2(I) = 0.D0
         Q1Z1(I) = 0.D0
         Q1Z2(I) = 0.D0
         Q201(I) = 0.D0
         Q202(I) = 0.D0
         Q301(I) = 0.D0
         Q302(I) = 0.D0
         Q22C1(I) = 0.D0
         Q22C2(I) = 0.D0
         Q1Z(I) = 0.D0
         Q1X(I) = 0.D0
         Q1Y(I) = 0.D0
         Q20(I) = 0.D0
         Q21C(I) = 0.D0
         Q21S(I) = 0.D0
         Q22C(I) = 0.D0
         Q22S(I) = 0.D0
         Q30(I) = 0.D0
         REFM(I) = 0
         RANK(I) = 0
      ENDDO


      IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP> MTPINIT'   
      IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP> MTP Spherical Tensor Module'     
      INUNI = GTRMI(COMLYN,COMLEN,'MTPUNIT',-1)
      IF (INUNI.EQ.-1) THEN
        IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP> ERROR. NO INPUT PARAMETERS'
      ELSE
        IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP> READING INPUT PARAMETERS'

        IF(IOLEV.GT.0) READ(INUNI,*) MOLS
        IF(IOLEV.GT.0) READ(INUNI,*) REFS
#if KEY_PARALLEL==1
        CALL PSND4(MOLS,1)
        CALL PSND4(REFS,1)
#endif 
        IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP>', MOLS, 'MOLECULES'
        IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP>', REFS, 'GEOMETRY TYPES'

        ! Dynamic allocation of memory
        MOLN = MOLS + 5   ! size of array (5 extra components)
        call chmalloc('mtp.src','MTPINIT','QANH',MOLN,log=QANH)
        call chmalloc('mtp.src','MTPINIT','QFLC',MOLN,log=QFLC)
        call chmalloc('mtp.src','MTPINIT','QDUM',MOLN,log=QDUM)
        call chmalloc('mtp.src','MTPINIT','LINT',MOLN,log=LINT)
        call chmalloc('mtp.src','MTPINIT','SITE',MOLN,intg=SITE)
        call chmalloc('mtp.src','MTPINIT','RFNR',MOLN,5,intg=RFNR)
        call chmalloc('mtp.src','MTPINIT','KDA', MOLN,intg=KDA)
        call chmalloc('mtp.src','MTPINIT','IPDA',2*MOLN,intg=IPDA)
        call chmalloc('mtp.src','MTPINIT','IZDA',3,2*MOLN,intg=IZDA)
        call chmalloc('mtp.src','MTPINIT','RZDA',3,2*MOLN,crl=RZDA)
        call chmalloc('mtp.src','MTPINIT','RCDA',3,2*MOLN,crl=RCDA)
        call chmalloc('mtp.src','MTPINIT','DEL0',MOLN,3,crl=DEL0)
        call chmalloc('mtp.src','MTPINIT','CBBEQDI',MOLN,crl=CBBEQDI)
        call chmalloc('mtp.src','MTPINIT','CBBVE',  MOLN,crl=CBBVE)
        call chmalloc('mtp.src','MTPINIT','CBBS',   MOLN,crl=CBBS)

        call chmalloc('mtp.src','MTPINIT','TRFX',MOLN,3,crl=TRFX)
        call chmalloc('mtp.src','MTPINIT','TRFY',MOLN,3,crl=TRFY)
        call chmalloc('mtp.src','MTPINIT','TRFZ',MOLN,3,crl=TRFZ)
        call chmalloc('mtp.src','MTPINIT','ADX',MOLN,3,crl=ADX)
        call chmalloc('mtp.src','MTPINIT','ADY',MOLN,3,crl=ADY)
        call chmalloc('mtp.src','MTPINIT','ADZ',MOLN,3,crl=ADZ)
        call chmalloc('mtp.src','MTPINIT','ADXS',MOLN,3,crl=ADXS)
        call chmalloc('mtp.src','MTPINIT','ADYS',MOLN,3,crl=ADYS)
        call chmalloc('mtp.src','MTPINIT','ADZS',MOLN,3,crl=ADZS)

        MOLCNT = 0
        NUMDA = 0   ! total number of dummy atoms
        DO I=1, REFS   ! REFS seems to be total number of unique MTP molecules
          MOLCNT = MOLCNT + 1
          !C C C read parameters for prototype molecules C C 
          N = MOLCNT
          IF(IOLEV.GT.0) READ(INUNI,*) SITE(N)
          IF(IOLEV.GT.0) READ(INUNI,*) RFNR(N,1),RFNR(N,2),RFNR(N,3),RFNR(N,4),RFNR(N,5)
          IF(IOLEV.GT.0) READ(INUNI,*) WRD
          IF ( WRD .EQ. 'QFLC') then
            QFLC(N) = .TRUE.
            IF(IOLEV.GT.0) READ(INUNI,*) CBBEQDI(N)
            IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP>  Fluctuating Moments Used'
          ELSE 
            QFLC(N) = .FALSE.
          ENDIF

          QDUM(N) =  .FALSE.
          QANH(N) = .FALSE. 

          IF(IOLEV.GT.0) READ(INUNI,*) WRD
          IF ( WRD .EQ. 'LINT') then
            LINT(N) = .TRUE.
          ELSE 
            LINT(N) = .FALSE. 
          ENDIF

          IF(IOLEV.GT.0) READ(INUNI,*) RFRNG(1), RFRNG(2)
#if KEY_PARALLEL==1
          CALL PSND4(SITE(N),1)
          DO J = 1, 5
             CALL PSND4(RFNR(N,J),1)
          ENDDO
          CALL PSND4(QFLC(N),1)
          CALL PSND4(QANH(N),1)
          CALL PSND4(LINT(N),1)
          CALL PSND4(RFRNG,2)
          CALL PSND8(CBBEQDI(N),1)
          CALL PSND8(CBBVE(N),1)
          CALL PSND8(CBBS(N),1)
#endif 
          !C C C write parameter setup
          IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP> REFERENCE NUMBERS '
          IF(PRNLEV.GT.2) WRITE(OUTU,*) RFNR(N,1),RFNR(N,2),RFNR(N,3),RFNR(N,4),RFNR(N,5)
          IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP>', SITE(N), 'SITES,  RANGE=',(RFRNG(L),L=1,2)
          IF ( RFNR(N,3) .EQ. 0 ) THEN
             IF(PRNLEV.GT.2) WRITE(OUTU,*) '               Diatomic'
             NATINCM = 2
          ELSE IF ( RFNR(N,4) .EQ. 0 ) THEN
             IF (LINT(N)) THEN
                IF(PRNLEV.GT.2) WRITE(OUTU,*) '               Linear Triatomic'
             ELSE
                IF(PRNLEV.GT.2) WRITE(OUTU,*) '               C2v'
             ENDIF
             NATINCM = 3
          ELSE 
             IF(PRNLEV.GT.2) WRITE(OUTU,*) 'ERROR: Symmetry type not available'
          ENDIF
          KDA(N) = 0    ! number of dummy atoms in the current molecule
          DO J=1, SITE(N)
             IF(IOLEV.GT.0) THEN
               READ(INUNI,*) REFAT
               IF (REFAT.LT.0) THEN     ! DUMMY ATOM
                 QDUM(N) = .TRUE.
                 IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP>  Dummy Atom Used'
                 KDA(N) = KDA(N) + 1   ! Number of dummy atoms in the current molecule
                 NUMDA = NUMDA + 1     ! Total number of dummy atoms
                 IZDA(1,NUMDA) = -REFAT
                 REFAT = NATOM + NUMDA
                 IPDA(NUMDA) = REFAT
                 READ(INUNI,*) IZDA(2,NUMDA),(RZDA(L,NUMDA),L=1,2),NASTP ! r, theta (no phi)
                 IF (REFAT.GT.MAXAIM) GOTO 95
               ELSE
                 IF (KDA(N).NE.0) THEN
                   IF(PRNLEV.GT.2) WRITE(OUTU,*) 'Place dummy atom(s) at the end.'
                   STOP
                 ENDIF
               ENDIF
               READ(INUNI,*) RANK(REFAT)
             ENDIF
#if KEY_PARALLEL==1
             CALL PSND4(QDUM(N),1)
             CALL PSND4(REFAT,1)
             CALL PSND4(RANK(REFAT),1)
#endif 
             REFM(REFAT) = MOLCNT
             IF (QFLC(N)) THEN
             !C C C read parameters for fluctuating moments
                IF(IOLEV.GT.0) READ(INUNI,*) Q1(REFAT)
                IF(IOLEV.GT.0) READ(INUNI,*) Q2(REFAT)
                IF (RANK(REFAT) .GT. 0 ) THEN
                   IF(IOLEV.GT.0) READ(INUNI,*) Q1Z1(REFAT)
                   IF(IOLEV.GT.0) READ(INUNI,*) Q1Z2(REFAT)
                   IF (RANK(REFAT) .GT. 1 ) THEN
                      IF(IOLEV.GT.0) READ(INUNI,*) Q201(REFAT)
                      IF(IOLEV.GT.0) READ(INUNI,*) Q202(REFAT)
                      IF (SITE(N) .GT. NATINCM) THEN
                         IF(IOLEV.GT.0) READ(INUNI,*) Q22C1(REFAT)
                         IF(IOLEV.GT.0) READ(INUNI,*) Q22C2(REFAT)
                      ENDIF
                      IF (RANK(REFAT) .GT. 2) THEN
                         IF(IOLEV.GT.0) READ(INUNI,*) Q301(REFAT)
                         IF(IOLEV.GT.0) READ(INUNI,*) Q302(REFAT)
                      ENDIF
                   ENDIF
                ENDIF
#if KEY_PARALLEL==1
                CALL PSND8(Q1(REFAT),1)
                CALL PSND8(Q2(REFAT),1)
                CALL PSND8(Q1Z1(REFAT),1)
                CALL PSND8(Q1Z2(REFAT),1)
                CALL PSND8(Q201(REFAT),1)
                CALL PSND8(Q202(REFAT),1)
                CALL PSND8(Q22C1(REFAT),1)
                CALL PSND8(Q22C2(REFAT),1)
                CALL PSND8(Q301(REFAT),1)
                CALL PSND8(Q302(REFAT),1)
#endif 
             ELSE
             !C C C read parameters for static moments
                IF (RANK(REFAT) .GT. 0 ) THEN
                   IF(IOLEV.GT.0) READ(INUNI,*) Q1Z(REFAT)
                   IF(IOLEV.GT.0) READ(INUNI,*) Q1X(REFAT)
                   IF(IOLEV.GT.0) READ(INUNI,*) Q1Y(REFAT)
                   IF (RANK(REFAT) .GT. 1 ) THEN
                      IF(IOLEV.GT.0) READ(INUNI,*) Q20(REFAT)
                      IF(IOLEV.GT.0) READ(INUNI,*) Q21C(REFAT)
                      IF(IOLEV.GT.0) READ(INUNI,*) Q21S(REFAT)
                      IF(IOLEV.GT.0) READ(INUNI,*) Q22C(REFAT)
                      IF(IOLEV.GT.0) READ(INUNI,*) Q22S(REFAT)
                   ENDIF
                ENDIF
#if KEY_PARALLEL==1
                CALL PSND8(Q1X(REFAT),1)
                CALL PSND8(Q1Y(REFAT),1)
                CALL PSND8(Q1Z(REFAT),1)
                CALL PSND8(Q20(REFAT),1)
                CALL PSND8(Q21C(REFAT),1)
                CALL PSND8(Q21S(REFAT),1)
                CALL PSND8(Q22C(REFAT),1)
                CALL PSND8(Q22S(REFAT),1)
#endif 
             ENDIF
          ENDDO
          IF ( RFRNG(2) .GT. RFRNG(1) ) THEN       
          !C C C apply parameters to all molecules of the same type C C
             CNT = (RFRNG(2) - RFRNG(1) + 1) / (SITE(N) - KDA(N))
             DO J= N+1, N+CNT
                MOLCNT = MOLCNT + 1
                QFLC(J) = QFLC(N)
                CBBEQDI(J) = CBBEQDI(N)
                CBBVE(J) = CBBVE(N)
                CBBS(J) = CBBS(N)
                QDUM(J) = QDUM(N)
                QANH(J) = QANH(N)
                LINT(J) = LINT(N)
                SITE(J) = SITE(N)
                KDA(J) = KDA(N)
                DO K=1, 5
                   RFNR(J,K) = RFNR(N,K)
                   IF (RFNR(J,K).NE.0) THEN
                      RFNR(J,K) = RFNR(J,K) + (SITE(N) - KDA(N))*(J-N)
                   ENDIF
                ENDDO
                DO K=1, SITE(J)
                   IF (K.GT.SITE(J)-KDA(N)) THEN   ! DUMMY ATOM
                      NUMDA = NUMDA + 1   ! Total number of dummy atoms
                      DO L=1, 3
                         IZDA(L,NUMDA) = IZDA(L,NUMDA-KDA(N))
                         IF (IZDA(L,NUMDA).NE.0) THEN
                            IZDA(L,NUMDA)=IZDA(L,NUMDA)+SITE(N)-KDA(N)
                         ENDIF
                         RZDA(L,NUMDA) = RZDA(L,NUMDA-KDA(N))
                      ENDDO
                      IDX = NATOM + NUMDA
                      REFAT = IDX - KDA(N)
                      IPDA(NUMDA) = REFAT
                      IF (REFAT.GT.MAXAIM) GOTO 95
                   ELSE
                      IDX = RFNR(J,K)   ! Destination site
                      REFAT = RFNR(N,K) ! Source site
                   ENDIF
                   RANK(IDX) = RANK(REFAT)
                   REFM(IDX) = MOLCNT
                   IF (QFLC(J)) THEN
                   !C C C copy parameters for fluctuating moments
                      Q1(IDX) = Q1(REFAT)
                      Q2(IDX) = Q2(REFAT)
                      IF (RANK(IDX) .GT. 0 ) THEN
                         Q1Z1(IDX) = Q1Z1(REFAT)
                         Q1Z2(IDX) = Q1Z2(REFAT)
                         IF (RANK(IDX) .GT. 1 ) THEN
                            Q201(IDX) = Q201(REFAT)
                            Q202(IDX) = Q202(REFAT)
                            Q22C1(IDX) = Q22C1(REFAT)
                            Q22C2(IDX) = Q22C2(REFAT)
                            IF (RANK(IDX) .GT. 2 ) THEN
                               Q301(IDX) = Q301(REFAT)
                               Q302(IDX) = Q302(REFAT)
                            ENDIF
                         ENDIF
                      ENDIF
                   ELSE
                   !C C C copy parameters for static moments
                      IF (RANK(IDX) .GT. 0 ) THEN
                         Q1Z(IDX) = Q1Z(REFAT)
                         Q1X(IDX) = Q1X(REFAT)
                         Q1Y(IDX) = Q1Y(REFAT)
                         IF (RANK(IDX) .GT. 1 ) THEN
                            Q20(IDX) = Q20(REFAT)
                            Q21C(IDX) = Q21C(REFAT)
                            Q21S(IDX) = Q21S(REFAT)
                            Q22C(IDX) = Q22C(REFAT)
                            Q22S(IDX) = Q22S(REFAT)
                         ENDIF
                      ENDIF
                   ENDIF 
                ENDDO   ! K
                IF (SITE(J) .LT. 5 ) THEN
                   DO K=(SITE(J)+1), 5
                      RFNR(J,K) = 0
                   ENDDO
                ENDIF
             ENDDO   ! J
          ENDIF
        ENDDO   ! I

        TOKB = TOKCAL/BOHRR   ! moved from MTPX

      ENDIF

      DO I = 1, NUMDA   ! initialize dummy atom coord.
         J = IZDA(1,I)
         K = IZDA(2,I)
         DI = RZDA(1,I)
         AN = RZDA(2,I)
         CALL INITDA(X(J),Y(J),Z(J),X(K),Y(K),Z(K),DI,AN,RCDA(1,I))
         ! Convert to Bohr
         RCDA(1,I) = RCDA(1,I) / BOHRR
         RCDA(2,I) = RCDA(2,I) / BOHRR
         RCDA(3,I) = RCDA(3,I) / BOHRR
      ENDDO

!C C C Check Setup by Comparison
      IF (MOLS .ne. MOLCNT ) THEN
         IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP> ERROR: wrong number of sites!', &
              MOLS, MOLCNT
      ENDIF
!c
!c    Initialize image counting parameters
      NMOL = MOLS
      IMOL = MOLS
!c
      RETURN


95    IF(PRNLEV.GT.2) WRITE(OUTU,*) "TOO MANY DUMMY ATOMS."
      STOP

      END SUBROUTINE MTPINIT

!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
!c
!c     Initialize dummy atom position of NO
!c

      subroutine initda(xn,yn,zn,xo,yo,zo,di,an, rda)

      use chm_kinds
      implicit none
      real(chm_real) xn,yn,zn, xo,yo,zo, di,an   ! INPUT
      real(chm_real) pi, d2r, bondlen, bondang
      real(chm_real) dx,dy,dz, dist, phi, theta
      real(chm_real) x0,y0,z0, x1,y1,z1, x2,y2,z2, rda(3)

!     pi = datan(1.d0) * 4.d0
!     d2r = pi / 180.d0
      d2r = DEG2RAD

      bondlen = di         ! Angstrom
      bondang = an * d2r   ! deg -> rad

      ! Vector N -> O
      dx = xo - xn
      dy = yo - yn
      dz = zo - zn
      dist = dsqrt( dx*dx + dy*dy + dz*dz )

      x0 = 0.d0
      y0 = bondlen * dsin( bondang )
      z0 = bondlen * dcos( bondang )

      theta = dacos(dz/dist)   ! in radian
      x1 = x0
      y1 =  dcos(theta)*y0 + dsin(theta)*z0
      z1 = -dsin(theta)*y0 + dcos(theta)*z0

      phi = datan2(dx,dy)      ! in radian
      x2 =  dcos(phi)*x1 + dsin(phi)*y1
      y2 = -dsin(phi)*x1 + dcos(phi)*y1
      z2 = z1

      rda(1) = x2 + xn
      rda(2) = y2 + yn
      rda(3) = z2 + zn

!     print *, "theta = ", theta/d2r, "   phi = ", phi/d2r

      return
      end subroutine initda
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE MTPX(NATOMX,JNBL,INBL)
! Main MTP Routine; Calculates Energies and Forces 
!
      use chm_kinds
      use stream
      use dimens_fcm
      use consta
      use coord
      use energym
      use deriv
      use psf
      use image
      use number
      use inbnd
      implicit none
!
! C C C C C local variables 
      INTEGER NATOMX,JNBL(*),INBL(*)
      INTEGER N1,N2,N3,N4,N5,N6,N0,NN,ITEMP   !,STRT,STP
      INTEGER M1,M2,M3,M4,M5,ITMP2
      INTEGER NC,ND,NM,NP, MAXND !, NASTP
      PARAMETER (MAXND=180)   ! Max # of dihedral angle
!     PARAMETER (NASTP=5)    ! Interval of angle scanning in deg.
      real(chm_real) R, RAXYZ(3)
      real(chm_real) EMTP,TE       !, L1E,L2E,L3E,L4E,L5E,L6E
      real(chm_real) XGR,YGR,ZGR   !, XGR2,YGR2,ZGR2, GRDT(3)
      real(chm_real) SMCT   !, CTSQ
      real(chm_real) UVDA(3,4), ODA(3)
      real(chm_real) EMTP2(MAXND), EMTPM, EMAXI   !, TE2(MAXND)
      real(chm_real) TMPTH, TMPRD, RIJNO, TMPBS, TMPBC
!
! define number of molecules for primary or image atoms
      IF (NATOMX .EQ. NATIM) THEN
         NMOL = IMOL
      ELSE
         NMOL = MOLS
      ENDIF
!
! all calculations in this module are in units of bohrr
      DO NN=1, NATOMX
         X(NN) = X(NN)/BOHRR
         Y(NN) = Y(NN)/BOHRR
         Z(NN) = Z(NN)/BOHRR
      ENDDO

!
! Calculate orienation parameters for each molecule 
      CALL REFAX(NATOMX,X,Y,Z)


!--------------------------------------------------------------------- 
      IF ( NUMDA .EQ. 0 ) GOTO 77   !

!     For dummy atom on NO
      IF ( NATOMX .EQ. NATIM ) THEN
        DO NN=1, NUMDA     ! over dummy atoms -> should be changed to MTP molecules
          N2 = IZDA(1,NN)
          N1 = REFM(N2)
          CALL REFAXD(N1,NATOMX,X,Y,Z,RCDA(1,NN))
        ENDDO   ! This loop seems redundant.
      ELSE
        DO NN=1, NUMDA     ! over dummy atoms -> should be changed to MTP molecules
          NP = NN + NATOM
          N0 = IZDA(2,NN)
          N2 = IZDA(1,NN)
          N1 = REFM(N2)
          IF ( N2.EQ.1 ) THEN
            ITEMP = 0
          ELSE
            ITEMP = INBL( N2 - 1 )
          ENDIF

          TMPRD = RZDA(1,NN) / BOHRR     ! r in bohr
          TMPTH = DEG2RAD * RZDA(2,NN)   ! theta in radian

          ODA(1) = X(N2)   ! Local origin for dummy atom
          ODA(2) = Y(N2)
          ODA(3) = Z(N2)

          ! set up local unit vectors for dummy atom (e1,e2,e3)
          ! e3 = r(N->O) / |r(N->O)|
          ! e4 = r(N->X) / |r(N->X)|   where X is the previous dummy atom position
          ! e2 = e3 X e4 / |e3 X e4|
          ! e1 = e2 X e3
          CALL UNITVEC(X(N2),X(N0),Y(N2),Y(N0),Z(N2),Z(N0),R,UVDA(1,3)) ! N2 -> N0
          RIJNO = R
          CALL UNITVEC2(X(N2),Y(N2),Z(N2),RCDA(1,NN),R,UVDA(1,4))       ! N2 -> DA
          CALL VECPROD(UVDA(1,4),UVDA(1,3),UVDA(1,1))    ! e1 = e4 X e3 / |e4 X e3|
          CALL VECPROD2(UVDA(1,3),UVDA(1,1),UVDA(1,2))   ! e2 = e3 X e1   (e3 T e1)

          NC = 0
          DO ND = 0, 179, NASTP   ! Rotation angle around z axis
            ! As NO is symmetric wrt 180 rotation, scanning from 0 to 180 is enough.
            ! -> will be changed to include all the dummy atoms in the current MTP molecule
            NC = NC + 1
            EMTP2(NC) = 0.D0

            CALL GETDAXYZ(ND,TMPTH,TMPRD,UVDA,ODA,RCDA(1,NN))
            CALL REFAXD(N1,NATOMX,X,Y,Z,RCDA(1,NN))

            ! Energy of real atoms should also be obtained along the angle 
            ITMP2 = 0
            DO M2 = 1, NATOMX   ! should be changed to atoms in the current MTP molecule
              M1 = REFM(M2)
              M4 = IABS(JNBL(ITMP2 + 1))
              M3 = REFM(M4)
      
              DO M5 = ITMP2 + 1, INBL(M2)
                M4 = IABS(JNBL(M5))
                M3 = REFM(M4)
                IF ( M1 .NE. 0 .OR. M3 .NE. 0 ) THEN
                  CALL UNITVEC(X(M2),X(M4),Y(M2),Y(M4),Z(M2),Z(M4),R,RAXYZ) ! M2 -> M4
                  CALL MTPXCORE(M1,M2,M3,M4, R,RAXYZ, &
                                TE,XGR,YGR,ZGR )
                  ! MAYBE GOOD TO HAVE SPECIAL SUBROUTINES ONLY FOR ENERGIES
                  ! ADD SMOOTH CUTOFF HERE
                  !
                  EMTP2(NC) = EMTP2(NC) + TE
                ENDIF
              ENDDO
              ITMP2 = INBL(M2)
            ENDDO

          ENDDO   ! ND (dihedral angle)

          ! Select dihedral angle giving lowest energy
          NM = 1
          M5 = 1
          EMTPM = EMTP2(1)   ! EMTPM : EMTP minimum
          EMAXI = EMTPM
          DO ND = 1, NC
            IF (EMTP2(ND).LT.EMTPM) THEN
              EMTPM = EMTP2(ND)
              NM = ND
            ENDIF
            IF (EMTP2(ND).GT.EMAXI) THEN
              EMAXI = EMTP2(ND)
              M5 = ND
            ENDIF
            M3 = (ND-1)*NASTP
          ENDDO
          ND = (NM-1) * NASTP   ! NASTP : step in the DO loop of ND over dihedral angle
          M4 = (M5-1) * NASTP
          IF(PRNLEV.GT.2) WRITE(OUTU,'(A,2I5,5X,A,F12.8,A)')   &
            " Angles of Emin and Emax : ", ND, M4,             &
            " Ediff = ", (EMAXI - EMTPM)*TOKCAL, " kcal/mol"

          CALL GETDAXYZ(ND,TMPTH,TMPRD,UVDA,ODA,RCDA(1,NN))
          CALL REFAXD(N1,NATOMX,X,Y,Z,RCDA(1,NN))

          ! Print coordinates of NO with dummy atom
          M1 = RFNR(N1,1)
          M2 = RFNR(N1,2)
          IF(PRNLEV.GT.2) THEN
            WRITE(OUTU,'(A,3F13.8)') " Coord. of N : ", X(M1),Y(M1),Z(M1)
            WRITE(OUTU,'(A,3F13.8)') " Coord. of O : ", X(M2),Y(M2),Z(M2)
            WRITE(OUTU,'(A,3F13.8)') " Coord. of X : ", (RCDA(M3,NN),M3=1,3)
          ENDIF

        ENDDO   ! NN (number of dummy atoms)
      ENDIF

77    CONTINUE
!--------------------------------------------------------------------- 

!
! INITIALIZE ATOMIC MULTIPOLE INTERACTION ENERGIES AND GRADIENTS
      EMTP = 0.D0
      DO N0 = 1, NMOL
         DO NN = 1, 3
            ADX(N0,NN) = 0.D0
            ADY(N0,NN) = 0.D0
            ADZ(N0,NN) = 0.D0
         ENDDO
      ENDDO


! CALCULATE ATOMIC MULTIPOLE INTERACTION ENERGIES AND GRADIENTS
      ITEMP = 0
      DO N2 = 1, NATOMX
        N1 = REFM(N2)
        DO N5 = ITEMP + 1, INBL(N2)
          N4 = IABS(JNBL(N5))
          N3 = REFM(N4)
          IF ( N1 .NE. 0 .OR. N3 .NE. 0 ) THEN
            CALL UNITVEC(X(N2),X(N4),Y(N2),Y(N4),Z(N2),Z(N4),R,RAXYZ) ! N2 -> N4
            IF (R*BOHRR .LT. CTOFNB) THEN   !!!!!
              IF (N1 .NE. 0) THEN
                DO N6 = 1, 3   ! save ADX,ADY,ADZ for smooth cutoff
                  ADXS(N1,N6) = ADX(N1,N6)
                  ADYS(N1,N6) = ADY(N1,N6)
                  ADZS(N1,N6) = ADZ(N1,N6)
                ENDDO
              ENDIF
              IF (N3 .NE. 0) THEN
                DO N6 = 1, 3   ! save ADX,ADY,ADZ for smooth cutoff
                  ADXS(N3,N6) = ADX(N3,N6)
                  ADYS(N3,N6) = ADY(N3,N6)
                  ADZS(N3,N6) = ADZ(N3,N6)
                ENDDO
              ENDIF
              CALL MTPXCORE(N1,N2,N3,N4, R,RAXYZ, &
                            TE,XGR,YGR,ZGR )
              ! put smooth cutoff here
              IF (R*BOHRR .GT. CTONNB) THEN
                SMCT = ((CTOFNB - R*BOHRR)/(CTOFNB - CTONNB))
                XGR = XGR*SMCT
                YGR = YGR*SMCT
                ZGR = ZGR*SMCT
                TE = TE*SMCT

                SMCT = ( 1.D0 - SMCT )
                IF (N1 .NE. 0) THEN
                  DO N6 = 1, 3
                    ADX(N1,N6)=(ADXS(N1,N6)-ADX(N1,N6))*SMCT+ADX(N1,N6)
                    ADY(N1,N6)=(ADYS(N1,N6)-ADY(N1,N6))*SMCT+ADY(N1,N6)
                    ADZ(N1,N6)=(ADZS(N1,N6)-ADZ(N1,N6))*SMCT+ADZ(N1,N6)
                  ENDDO
                ENDIF
                IF (N3 .NE. 0) THEN
                  DO N6 = 1, 3
                    ADX(N3,N6)=(ADXS(N3,N6)-ADX(N3,N6))*SMCT+ADX(N3,N6)
                    ADY(N3,N6)=(ADYS(N3,N6)-ADY(N3,N6))*SMCT+ADY(N3,N6)
                    ADZ(N3,N6)=(ADZS(N3,N6)-ADZ(N3,N6))*SMCT+ADZ(N3,N6)
                  ENDDO
                ENDIF

              ENDIF
              DX(N4) = DX(N4) + XGR/R*TOKB
              DY(N4) = DY(N4) + YGR/R*TOKB
              DZ(N4) = DZ(N4) + ZGR/R*TOKB
              DX(N2) = DX(N2) - XGR/R*TOKB
              DY(N2) = DY(N2) - YGR/R*TOKB
              DZ(N2) = DZ(N2) - ZGR/R*TOKB
              EMTP = EMTP + TE
            ENDIF   !!!!!
          ENDIF
        ENDDO   ! N5
        ITEMP = INBL(N2)
      ENDDO   ! N2
!     WRITE(*,'(A,F12.8,A)') " Ener = ", EMTP*TOKCAL, " kcal/mol"


!
! add MTP energy to electrostatics
      IF (NATOMX .EQ. NATIM) THEN
         ETERM(IMELEC) = ETERM(IMELEC) + EMTP*TOKCAL 
      ELSE
         ETERM(ELEC) = ETERM(ELEC) + EMTP*TOKCAL 
      ENDIF
!
! add correction terms to each axis system
      DO N0 = 1, NMOL
        IF ( RFNR(N0,3) .EQ. 0 ) THEN
          DX(RFNR(N0,1)) = DX(RFNR(N0,1)) - ADZ(N0,1)/DEL0(N0,3)*TOKB
          DY(RFNR(N0,1)) = DY(RFNR(N0,1)) - ADZ(N0,2)/DEL0(N0,3)*TOKB
          DZ(RFNR(N0,1)) = DZ(RFNR(N0,1)) - ADZ(N0,3)/DEL0(N0,3)*TOKB
          DX(RFNR(N0,2)) = DX(RFNR(N0,2)) + ADZ(N0,1)/DEL0(N0,3)*TOKB
          DY(RFNR(N0,2)) = DY(RFNR(N0,2)) + ADZ(N0,2)/DEL0(N0,3)*TOKB
          DZ(RFNR(N0,2)) = DZ(RFNR(N0,2)) + ADZ(N0,3)/DEL0(N0,3)*TOKB
        ELSE IF ( RFNR(N0,4) .EQ. 0 ) THEN
          IF (LINT(N0)) THEN
            DX(RFNR(N0,2)) = DX(RFNR(N0,2)) + ADZ(N0,1)/DEL0(N0,3)*TOKB
            DY(RFNR(N0,2)) = DY(RFNR(N0,2)) + ADZ(N0,2)/DEL0(N0,3)*TOKB
            DZ(RFNR(N0,2)) = DZ(RFNR(N0,2)) + ADZ(N0,3)/DEL0(N0,3)*TOKB
            DX(RFNR(N0,3)) = DX(RFNR(N0,3)) - ADZ(N0,1)/DEL0(N0,3)*TOKB
            DY(RFNR(N0,3)) = DY(RFNR(N0,3)) - ADZ(N0,2)/DEL0(N0,3)*TOKB
            DZ(RFNR(N0,3)) = DZ(RFNR(N0,3)) - ADZ(N0,3)/DEL0(N0,3)*TOKB
          ELSE
            DX(RFNR(N0,1)) = DX(RFNR(N0,1)) + ADZ(N0,1)/DEL0(N0,3)*TOKB
            DY(RFNR(N0,1)) = DY(RFNR(N0,1)) + ADZ(N0,2)/DEL0(N0,3)*TOKB
            DZ(RFNR(N0,1)) = DZ(RFNR(N0,1)) + ADZ(N0,3)/DEL0(N0,3)*TOKB
            DX(RFNR(N0,2)) = DX(RFNR(N0,2)) - ADZ(N0,1)/DEL0(N0,3)*HALF*TOKB
            DY(RFNR(N0,2)) = DY(RFNR(N0,2)) - ADZ(N0,2)/DEL0(N0,3)*HALF*TOKB
            DZ(RFNR(N0,2)) = DZ(RFNR(N0,2)) - ADZ(N0,3)/DEL0(N0,3)*HALF*TOKB
            DX(RFNR(N0,3)) = DX(RFNR(N0,3)) - ADZ(N0,1)/DEL0(N0,3)*HALF*TOKB
            DY(RFNR(N0,3)) = DY(RFNR(N0,3)) - ADZ(N0,2)/DEL0(N0,3)*HALF*TOKB
            DZ(RFNR(N0,3)) = DZ(RFNR(N0,3)) - ADZ(N0,3)/DEL0(N0,3)*HALF*TOKB
!
            DX(RFNR(N0,2)) = DX(RFNR(N0,2)) + ADY(N0,1)/DEL0(N0,2)*TOKB
            DY(RFNR(N0,2)) = DY(RFNR(N0,2)) + ADY(N0,2)/DEL0(N0,2)*TOKB
            DZ(RFNR(N0,2)) = DZ(RFNR(N0,2)) + ADY(N0,3)/DEL0(N0,2)*TOKB
            DX(RFNR(N0,3)) = DX(RFNR(N0,3)) - ADY(N0,1)/DEL0(N0,2)*TOKB
            DY(RFNR(N0,3)) = DY(RFNR(N0,3)) - ADY(N0,2)/DEL0(N0,2)*TOKB
            DZ(RFNR(N0,3)) = DZ(RFNR(N0,3)) - ADY(N0,3)/DEL0(N0,2)*TOKB
!
            DX(RFNR(N0,1)) = DX(RFNR(N0,1)) + ADX(N0,1)/DEL0(N0,3)*TOKB
            DY(RFNR(N0,1)) = DY(RFNR(N0,1)) + ADX(N0,2)/DEL0(N0,3)*TOKB
            DZ(RFNR(N0,1)) = DZ(RFNR(N0,1)) + ADX(N0,3)/DEL0(N0,3)*TOKB
            DX(RFNR(N0,2)) = DX(RFNR(N0,2)) - ADX(N0,1)/DEL0(N0,3)*HALF*TOKB
            DY(RFNR(N0,2)) = DY(RFNR(N0,2)) - ADX(N0,2)/DEL0(N0,3)*HALF*TOKB
            DZ(RFNR(N0,2)) = DZ(RFNR(N0,2)) - ADX(N0,3)/DEL0(N0,3)*HALF*TOKB
            DX(RFNR(N0,3)) = DX(RFNR(N0,3)) - ADX(N0,1)/DEL0(N0,3)*HALF*TOKB
            DY(RFNR(N0,3)) = DY(RFNR(N0,3)) - ADX(N0,2)/DEL0(N0,3)*HALF*TOKB
            DZ(RFNR(N0,3)) = DZ(RFNR(N0,3)) - ADX(N0,3)/DEL0(N0,3)*HALF*TOKB
          ENDIF
        ENDIF
      ENDDO   ! N0
!
! return coordinates in angstrom
      DO NN = 1, NATOMX
         X(NN) = X(NN)*BOHRR
         Y(NN) = Y(NN)*BOHRR
         Z(NN) = Z(NN)*BOHRR
      ENDDO
!
      RETURN
      END SUBROUTINE MTPX
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE GETDAXYZ(ND,TMPTH,R,UVDA,ODA,XYZDA)
!   Output : XYZDA (X,Y,Z)
!            X,Y,Z coord. of dummy atom in global Cartesian coord. system

      use chm_kinds
      IMPLICIT NONE
      INTEGER ND
!     real(chm_real) DEG2RAD   ! pi / 180
      real(chm_real) TMPTH,R,UVDA(3,4),ODA(3),XYZDA(3)
      real(chm_real) XDA,YDA,ZDA, TMPPH, XYDA

      TMPPH = DEG2RAD * DBLE(ND)   ! N.B.
      XYDA = R * DSIN(TMPTH)

      XDA = XYDA * DCOS(TMPPH)   ! e1 component
      YDA = XYDA * DSIN(TMPPH)   ! e2 component
      ZDA = R * DCOS(TMPTH)      ! e3 component

      XYZDA(1) = XDA*UVDA(1,1)+YDA*UVDA(1,2)+ZDA*UVDA(1,3) + ODA(1) ! global x
      XYZDA(2) = XDA*UVDA(2,1)+YDA*UVDA(2,2)+ZDA*UVDA(2,3) + ODA(2) ! global y
      XYZDA(3) = XDA*UVDA(3,1)+YDA*UVDA(3,2)+ZDA*UVDA(3,3) + ODA(3) ! global z

      RETURN
      END SUBROUTINE GETDAXYZ
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE UNITVEC(X1,X2,Y1,Y2,Z1,Z2,R,UVEC)   ! 1 -> 2
!   Calculates unit vector (UVEC) and distance (R)
!   from point 1 to point 2

      use chm_kinds
      implicit none
      real(chm_real) X1,X2,Y1,Y2,Z1,Z2   ! INPUT
      real(chm_real) R,UVEC(3)           ! OUTPUT
      real(chm_real) DFX,DFY,DFZ         ! TEMP

      DFX = X2 - X1
      DFY = Y2 - Y1
      DFZ = Z2 - Z1

      R = DSQRT( DFX*DFX + DFY*DFY + DFZ*DFZ )

      UVEC(1) = DFX / R
      UVEC(2) = DFY / R
      UVEC(3) = DFZ / R

      RETURN
      END SUBROUTINE UNITVEC
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE UNITVEC2(X1,Y1,Z1,V2,R,UVEC)   ! 1 -> 2

      use chm_kinds
      IMPLICIT NONE
      real(chm_real) X1,Y1,Z1,V2(3)   ! INPUT
      real(chm_real) R,UVEC(3)        ! OUTPUT
      real(chm_real) DFX,DFY,DFZ      ! TEMP

      DFX = V2(1) - X1
      DFY = V2(2) - Y1
      DFZ = V2(3) - Z1

      R = DSQRT( DFX*DFX + DFY*DFY + DFZ*DFZ )

      UVEC(1) = DFX / R
      UVEC(2) = DFY / R
      UVEC(3) = DFZ / R

      RETURN
      END SUBROUTINE UNITVEC2
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE VECPROD(V1,V2,VP)   ! VP = V1 X V2 / | V1 X V2 |

      use chm_kinds
      IMPLICIT NONE
      real(chm_real) V1(3),V2(3)   ! INPUT
      real(chm_real) VP(3)         ! OUTPUT
      real(chm_real) R             ! TEMP

      VP(1) = V1(2) * V2(3) - V1(3) * V2(2)
      VP(2) = V1(3) * V2(1) - V1(1) * V2(3)
      VP(3) = V1(1) * V2(2) - V1(2) * V2(1)

      R = DSQRT( VP(1)**2 + VP(2)**2 + VP(3)**2 )

      VP(1) = VP(1) / R
      VP(2) = VP(2) / R
      VP(3) = VP(3) / R

      RETURN
      END SUBROUTINE VECPROD
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE VECPROD2(V1,V2,VP)   ! VP = V1 X V2

      use chm_kinds
      IMPLICIT NONE
      real(chm_real) V1(3),V2(3)   ! INPUT
      real(chm_real) VP(3)         ! OUTPUT

      VP(1) = V1(2) * V2(3) - V1(3) * V2(2)
      VP(2) = V1(3) * V2(1) - V1(1) * V2(3)
      VP(3) = V1(1) * V2(2) - V1(2) * V2(1)

      RETURN
      END SUBROUTINE VECPROD2
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE VECNORM(V1,VP)   ! VP = V1 / |V1|

      use chm_kinds
      IMPLICIT NONE
      real(chm_real) V1(3)   ! INPUT
      real(chm_real) VP(3)   ! OUTPUT
      real(chm_real) R

      R = DSQRT( V1(1)**2 + V1(2)**2 + V1(3)**2 )

      VP(1) = V1(1) / R
      VP(2) = V1(2) / R
      VP(3) = V1(3) / R

      RETURN
      END SUBROUTINE VECNORM
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE VECSUM(C1,C2,V1,V2,VP)   ! VP = C1 . V1 + C2 . V2

      use chm_kinds
      IMPLICIT NONE
      real(chm_real) C1,C2, V1(3),V2(3)   ! INPUT
      real(chm_real) VP(3)                ! OUTPUT

      VP(1) = C1*V1(1) + C2*V2(1)
      VP(2) = C1*V1(2) + C2*V2(2)
      VP(3) = C1*V1(3) + C2*V2(3)

      RETURN
      END SUBROUTINE VECSUM
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE VECSUM2(C2,V1,V2,VP)   ! VP = V1 - C2 . V2

      use chm_kinds
      IMPLICIT NONE
      real(chm_real) C2, V1(3),V2(3)   ! INPUT
      real(chm_real) VP(3)             ! OUTPUT

      VP(1) = V1(1) - C2*V2(1)
      VP(2) = V1(2) - C2*V2(2)
      VP(3) = V1(3) - C2*V2(3)

      RETURN
      END SUBROUTINE VECSUM2
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE MTPXCORE(N1,N2,N3,N4, R,RAXYZ, &
                          TE,XGR,YGR,ZGR)
! Split from MTPX
! Calculates Energies and Forces 
!
      use chm_kinds
      use dimens_fcm
      use consta
      use coord
      use energym
      use deriv
      use psf
      use image
      use number
      use inbnd
      implicit none
!
! C C C C C local variables
      INTEGER N1,N2,N3,N4   !,N5,ITEMP,STRT,STP
      real(chm_real) R, RAXYZ(3), TRAX,TRAY,TRAZ,TRBX,TRBY,TRBZ
      real(chm_real) CXX,CXY,CXZ,CYX,CYY,CYZ,CZX,CZY,CZZ
      real(chm_real) TE,XGR,YGR,ZGR,L2E,L3E,L4E,L5E,L6E
      real(chm_real) L2GR(3),L3GR(3),L4GR(3),L5GR(3),L6GR(3)

      RAX = RAXYZ(1)
      RAY = RAXYZ(2)
      RAZ = RAXYZ(3)

      ! rotate the interaction vectors according to the molecular orientation
      IF (N1 .NE. 0 ) THEN
        TRAX = TRFX(N1,1)*RAX+TRFX(N1,2)*RAY+TRFX(N1,3)*RAZ
        TRAY = TRFY(N1,1)*RAX+TRFY(N1,2)*RAY+TRFY(N1,3)*RAZ  
        TRAZ = TRFZ(N1,1)*RAX+TRFZ(N1,2)*RAY+TRFZ(N1,3)*RAZ
      ENDIF
      IF (N3 .NE. 0 ) THEN
        TRBX =-TRFX(N3,1)*RAX-TRFX(N3,2)*RAY-TRFX(N3,3)*RAZ
        TRBY =-TRFY(N3,1)*RAX-TRFY(N3,2)*RAY-TRFY(N3,3)*RAZ  
        TRBZ =-TRFZ(N3,1)*RAX-TRFZ(N3,2)*RAY-TRFZ(N3,3)*RAZ
        ! calculate interaction coefficient for multipole-multipole interactions
        IF (N1 .NE. 0 ) THEN
           CXX = TRFX(N1,1)*TRFX(N3,1) + TRFX(N1,2)*TRFX(N3,2) &
                + TRFX(N1,3)*TRFX(N3,3)
           CXY = TRFX(N1,1)*TRFY(N3,1) + TRFX(N1,2)*TRFY(N3,2) &
                + TRFX(N1,3)*TRFY(N3,3)
           CXZ = TRFX(N1,1)*TRFZ(N3,1) + TRFX(N1,2)*TRFZ(N3,2) &
                + TRFX(N1,3)*TRFZ(N3,3)
           CYX = TRFY(N1,1)*TRFX(N3,1) + TRFY(N1,2)*TRFX(N3,2) &
                + TRFY(N1,3)*TRFX(N3,3)
           CYY = TRFY(N1,1)*TRFY(N3,1) + TRFY(N1,2)*TRFY(N3,2) &
                + TRFY(N1,3)*TRFY(N3,3)
           CYZ = TRFY(N1,1)*TRFZ(N3,1) + TRFY(N1,2)*TRFZ(N3,2) &
                + TRFY(N1,3)*TRFZ(N3,3)
           CZX = TRFZ(N1,1)*TRFX(N3,1) + TRFZ(N1,2)*TRFX(N3,2) &
                + TRFZ(N1,3)*TRFX(N3,3)
           CZY = TRFZ(N1,1)*TRFY(N3,1) + TRFZ(N1,2)*TRFY(N3,2) &
                + TRFZ(N1,3)*TRFY(N3,3)
           CZZ = TRFZ(N1,1)*TRFZ(N3,1) + TRFZ(N1,2)*TRFZ(N3,2) &
                + TRFZ(N1,3)*TRFZ(N3,3)
        ENDIF   ! N1
        !c change sign for use in derivatives
        TRFX(N3,1) = -TRFX(N3,1)
        TRFX(N3,2) = -TRFX(N3,2)
        TRFX(N3,3) = -TRFX(N3,3)
        TRFY(N3,1) = -TRFY(N3,1)
        TRFY(N3,2) = -TRFY(N3,2)
        TRFY(N3,3) = -TRFY(N3,3)
        TRFZ(N3,1) = -TRFZ(N3,1)
        TRFZ(N3,2) = -TRFZ(N3,2)
        TRFZ(N3,3) = -TRFZ(N3,3)
        ADX(N3,1) = -ADX(N3,1)
        ADX(N3,2) = -ADX(N3,2)
        ADX(N3,3) = -ADX(N3,3)
        ADY(N3,1) = -ADY(N3,1)
        ADY(N3,2) = -ADY(N3,2)
        ADY(N3,3) = -ADY(N3,3)
        ADZ(N3,1) = -ADZ(N3,1)
        ADZ(N3,2) = -ADZ(N3,2)
        ADZ(N3,3) = -ADZ(N3,3)
      ENDIF   ! N3

      TE = 0.D0
      XGR = 0.D0
      YGR = 0.D0
      ZGR = 0.D0
!C    L2 ----- L2 ----- L2 ----- L2 ----- L2 ----- L2 ----- L2 -----
      IF((RANK(N4) + RANK(N2)) .GT. 0)THEN
        L2E = 0.D0
        L2GR(1) = 0.D0
        L2GR(2) = 0.D0
        L2GR(3) = 0.D0
        RM1 = 1.0d0 / R
        RM2 = RM1*RM1
        RAX3 = THREE*RAX
        RAY3 = THREE*RAY
        RAZ3 = THREE*RAZ
        IF (Q1Z(N4).NE.0.D0) THEN
          CALL R0R1(L2E,L2GR(1),L2GR(2),L2GR(3),      &
                    ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),    &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                    TRBZ,CG(N2),Q1Z(N4))
        ENDIF
        IF (Q1X(N4).NE.0.D0) THEN
          CALL R0R1(L2E,L2GR(1),L2GR(2),L2GR(3),      &
                    ADX(N3,1),ADX(N3,2),ADX(N3,3),    &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                    TRBX,CG(N2),Q1X(N4))
        ENDIF
        IF (Q1Y(N4).NE.0.D0) THEN
          CALL R0R1(L2E,L2GR(1),L2GR(2),L2GR(3),      &
                    ADY(N3,1),ADY(N3,2),ADY(N3,3),    &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3), &
                    TRBY,CG(N2),Q1Y(N4))
        ENDIF
        IF (Q1Z(N2).NE.0.D0) THEN
          CALL R0R1(L2E,L2GR(1),L2GR(2),L2GR(3),      &
                    ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),    &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRAZ,CG(N4),Q1Z(N2))
        ENDIF
        IF (Q1X(N2).NE.0.D0) THEN
          CALL R0R1(L2E,L2GR(1),L2GR(2),L2GR(3),      &
                    ADX(N1,1),ADX(N1,2),ADX(N1,3),    &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                    TRAX,CG(N4),Q1X(N2))
        ENDIF
        IF (Q1Y(N2).NE.0.D0) THEN
          CALL R0R1(L2E,L2GR(1),L2GR(2),L2GR(3),      &
                    ADY(N1,1),ADY(N1,2),ADY(N1,3),    &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3), &
                    TRAY,CG(N4),Q1Y(N2))
        ENDIF
        TE = TE + L2E
        XGR = XGR + L2GR(1)
        YGR = YGR + L2GR(2)
        ZGR = ZGR + L2GR(3)
!C      L3 ----- L3 ----- L3 ----- L3 ----- L3 ----- L3 ----- L3 -----
        IF((RANK(N4) + RANK(N2)) .GT. 1)THEN
          L3E = 0.D0
          L3GR(1) = 0.D0
          L3GR(2) = 0.D0
          L3GR(3) = 0.D0
          RM3 = RM1*RM2
          RAX5 = FIVE*RAX
          RAY5 = FIVE*RAY
          RAZ5 = FIVE*RAZ
!c        r1r1 -- r1r1 -- r1r1 -- r1r1 -- r1r1 -- r1r1 -- r1r1 -- r1r1 --
          IF (Q1Z(N2).NE.0.D0) THEN
            IF (Q1Z(N4).NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CZZ,     &
                ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),               &
                ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),               &
                TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),TRFZ(N3,1), &
                TRFZ(N3,2),TRFZ(N3,3),TRAZ,TRBZ,Q1Z(N2),Q1Z(N4))
            ENDIF
            IF (Q1X(N4).NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CZX,     &
                ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),               &
                ADX(N3,1),ADX(N3,2),ADX(N3,3),               &
                TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),TRFX(N3,1), &
                TRFX(N3,2),TRFX(N3,3),TRAZ,TRBX,Q1Z(N2),Q1X(N4))
            ENDIF
            IF (Q1Y(N4).NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CZY,     &
                ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),               &
                ADY(N3,1),ADY(N3,2),ADY(N3,3),               &
                TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),TRFY(N3,1), &
                TRFY(N3,2),TRFY(N3,3),TRAZ,TRBY,Q1Z(N2),Q1Y(N4))
            ENDIF
          ENDIF
          IF (Q1X(N2).NE.0.D0) THEN
            IF (Q1Z(N4).NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CXZ,     &
                ADX(N1,1),ADX(N1,2),ADX(N1,3),               &
                ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),               &
                TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFZ(N3,1), &
                TRFZ(N3,2),TRFZ(N3,3),TRAX,TRBZ,Q1X(N2),Q1Z(N4))
            ENDIF
            IF (Q1X(N4) .NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CXX,     &
                ADX(N1,1),ADX(N1,2),ADX(N1,3),               &
                ADX(N3,1),ADX(N3,2),ADX(N3,3),               &
                TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFX(N3,1), &
                TRFX(N3,2),TRFX(N3,3),TRAX,TRBX,Q1X(N2),Q1X(N4))
            ENDIF
            IF (Q1Y(N4).NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CXY,     &
                ADX(N1,1),ADX(N1,2),ADX(N1,3),               &
                ADY(N3,1),ADY(N3,2),ADY(N3,3),               &
                TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFY(N3,1), &
                TRFY(N3,2),TRFY(N3,3),TRAX,TRBY,Q1X(N2),Q1Y(N4))
            ENDIF
          ENDIF
          IF (Q1Y(N2).NE.0.D0) THEN
            IF (Q1Z(N4).NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CYZ,     &
                ADY(N1,1),ADY(N1,2),ADY(N1,3),               &
                ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),               &
                TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),TRFZ(N3,1), &
                TRFZ(N3,2),TRFZ(N3,3),TRAY,TRBZ,Q1Y(N2),Q1Z(N4))
            ENDIF
            IF (Q1X(N4).NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CYX,     &
                ADY(N1,1),ADY(N1,2),ADY(N1,3),               &
                ADX(N3,1),ADX(N3,2),ADX(N3,3),               &
                TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),TRFX(N3,1), &
                TRFX(N3,2),TRFX(N3,3),TRAY,TRBX,Q1Y(N2),Q1X(N4))
            ENDIF
            IF (Q1Y(N4).NE.0.D0) THEN
              CALL R1R1(L3E,L3GR(1),L3GR(2),L3GR(3),CYY,     &
                ADY(N1,1),ADY(N1,2),ADY(N1,3),               &
                ADY(N3,1),ADY(N3,2),ADY(N3,3),               &
                TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),TRFY(N3,1), &
                TRFY(N3,2),TRFY(N3,3),TRAY,TRBY,Q1Y(N2),Q1Y(N4))
            ENDIF
          ENDIF
!c        r0r2 -- r0r2 -- r0r2 -- r0r2 -- r0r2 -- r0r2 -- r0r2 -- r0r2 --
          IF (RANK(N4).GT.1) THEN
            IF (Q20(N4) .NE.0.D0) THEN
              CALL R0R20(L3E,L3GR(1),L3GR(2),L3GR(3),  &
                ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),         &
                TRBZ,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                CG(N2),Q20(N4))
            ENDIF
            IF (Q21C(N4).NE.0.D0) THEN
              CALL R0R2I(L3E,L3GR(1),L3GR(2),L3GR(3),       &
                TRBX,TRBZ,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CG(N2),Q21C(N4))
            ENDIF
            IF (Q21S(N4).NE.0.D0) THEN
              CALL R0R2I(L3E,L3GR(1),L3GR(2),L3GR(3),       &
                TRBY,TRBZ,TRFY(N3,1),TRFY(N3,2),TRFY(N3,3), &
                TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CG(N2),Q21S(N4))
            ENDIF
            IF (Q22C(N4).NE.0.D0) THEN
              CALL R0R22C(L3E,L3GR(1),L3GR(2),L3GR(3),      &
                ADX(N3,1),ADX(N3,2),ADX(N3,3),              &
                ADY(N3,1),ADY(N3,2),ADY(N3,3),              &
                TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),MINONE,2,  &
                TRBX,TRBY,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),CG(N2),Q22C(N4))
            ENDIF
            IF (Q22S(N4).NE.0.D0) THEN
              CALL R0R2I(L3E,L3GR(1),L3GR(2),L3GR(3),       &
                TRBX,TRBY,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),CG(N2),Q22S(N4))
            ENDIF
          ENDIF
!c        r2r0 -- r2r0 -- r2r0 -- r2r0 -- r2r0 -- r2r0 -- r2r0 -- r2r0 --
          IF (RANK(N2).GT.1) THEN
            IF (Q20(N2).NE.0.D0) THEN
              CALL R0R20(L3E,L3GR(1),L3GR(2),L3GR(3),  &
                ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),         &
                TRAZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                CG(N4),Q20(N2))
            ENDIF
            IF (Q21C(N2).NE.0.D0) THEN
              CALL R0R2I(L3E,L3GR(1),L3GR(2),L3GR(3),       &
                TRAX,TRAZ,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CG(N4),Q21C(N2))
            ENDIF
            IF (Q21S(N2).NE.0.D0) THEN
              CALL R0R2I(L3E,L3GR(1),L3GR(2),L3GR(3),       &
                TRAY,TRAZ,TRFY(N1,1),TRFY(N1,2),TRFY(N1,3), &
                TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CG(N4),Q21S(N2))
            ENDIF
            IF (Q22C(N2).NE.0.D0) THEN
              CALL R0R22C(L3E,L3GR(1),L3GR(2),L3GR(3),      &
                ADX(N1,1),ADX(N1,2),ADX(N1,3),              &
                ADY(N1,1),ADY(N1,2),ADY(N1,3),              &
                TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),ONE,1,     &
                TRAX,TRAY,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),CG(N4),Q22C(N2))
            ENDIF
            IF (Q22S(N2).NE.0.D0) THEN
              CALL R0R2I(L3E,L3GR(1),L3GR(2),L3GR(3),       &
                TRAX,TRAY,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),CG(N4),Q22S(N2))
            ENDIF
          ENDIF
          TE = TE + L3E
          XGR = XGR + L3GR(1)
          YGR = YGR + L3GR(2)
          ZGR = ZGR + L3GR(3)
!c        L4 ----- L4 ----- L4 ----- L4 ----- L4 ----- L4 ----- L4 -----
          IF((RANK(N4) + RANK(N2)) .GT. 2)THEN
            L4E = 0.D0
            L4GR(1) = 0.D0
            L4GR(2) = 0.D0
            L4GR(3) = 0.D0
            RM4 = RM2*RM2
            RAX7 = SEVEN*RAX
            RAY7 = SEVEN*RAY
            RAZ7 = SEVEN*RAZ
!c          r0r3 -- r0r3 -- r0r3 -- r0r3 -- r0r3 -- r0r3 -- r0r3 -- r0r3 --
            IF (RANK(N4).GT.2) THEN
              CALL R0R3(L4E,L4GR(1),L4GR(2),L4GR(3),   &
                ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),         &
                TRBZ,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                CG(N2),Q30(N4))
            ENDIF
!c          r3r0 -- r3r0 -- r3r0 -- r3r0 -- r3r0 -- r3r0 -- r3r0 -- r3r0 --
            IF (RANK(N2).GT.2) THEN
              CALL R0R3(L4E,L4GR(1),L4GR(2),L4GR(3),   &
                ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),         &
                TRAZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                CG(N4),Q30(N2))
            ENDIF
!c          r1r2 -- r1r2 -- r1r2 -- r1r2 -- r1r2 -- r1r2 -- r1r2 -- r1r2 --
            IF (RANK(N4).GT.1) THEN
              IF (Q20(N4).NE.0.D0) THEN
                IF (Q1Z(N2).NE.0.D0) THEN
                  CALL R1R20(L4E,L4GR(1),L4GR(2),L4GR(3),       &
                    ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),              &
                    ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),              &
                    TRAZ,TRBZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CZZ,Q1Z(N2),Q20(N4))
                ENDIF
                IF (Q1X(N2).NE.0.D0) THEN
                  CALL R1R20(L4E,L4GR(1),L4GR(2),L4GR(3),       &
                    ADX(N1,1),ADX(N1,2),ADX(N1,3),              &
                    ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),              &
                    TRAX,TRBZ,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CXZ,Q1X(N2),Q20(N4))
                 ENDIF
                IF (Q1Y(N2).NE.0.D0) THEN
                  CALL R1R20(L4E,L4GR(1),L4GR(2),L4GR(3),       &
                    ADY(N1,1),ADY(N1,2),ADY(N1,3),              &
                    ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),              &
                    TRAY,TRBZ,TRFY(N1,1),TRFY(N1,2),TRFY(N1,3), &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CYZ,Q1Y(N2),Q20(N4))
                ENDIF
              ENDIF
              IF (Q21C(N4).NE.0.D0) THEN
                IF (Q1Z(N2).NE.0.D0)THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAZ,TRBX,TRBZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFZ(N3,1),     &
                    TRFZ(N3,2),TRFZ(N3,3),CZX,CZZ,Q1Z(N2),Q21C(N4))
                ENDIF
                IF (Q1X(N2).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAX,TRBX,TRBZ,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFZ(N3,1),     &
                    TRFZ(N3,2),TRFZ(N3,3),CXX,CXZ,Q1X(N2),Q21C(N4))
                ENDIF
                IF (Q1Y(N2).NE.0.D0)THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAY,TRBX,TRBZ,TRFY(N1,1),TRFY(N1,2),TRFY(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFZ(N3,1),     &
                    TRFZ(N3,2),TRFZ(N3,3),CYX,CYZ,Q1Y(N2),Q21C(N4))
                ENDIF
              ENDIF
              IF (Q21S(N4).NE.0.D0) THEN
                IF (Q1Z(N2).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAZ,TRBY,TRBZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),TRFZ(N3,1),     &
                    TRFZ(N3,2),TRFZ(N3,3),CZY,CZZ,Q1Z(N2),Q21S(N4))
                ENDIF
                IF (Q1X(N2).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAX,TRBY,TRBZ,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),TRFZ(N3,1),     &
                    TRFZ(N3,2),TRFZ(N3,3),CXY,CXZ,Q1X(N2),Q21S(N4))
                ENDIF
                IF (Q1Y(N2).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAY,TRBY,TRBZ,TRFY(N1,1),TRFY(N1,2),TRFY(N1,3), &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),TRFZ(N3,1),     &
                    TRFZ(N3,2),TRFZ(N3,3),CYY,CYZ,Q1Y(N2),Q21S(N4))
                ENDIF
              ENDIF
              IF (Q22C(N4).NE.0.D0) THEN
                IF (Q1X(N2).NE.0.D0) THEN
                  CALL R1R22C(L4E,L4GR(1),L4GR(2),L4GR(3),           &
                    ADX(N1,1),ADX(N1,2),ADX(N1,3),                   &
                    ADX(N3,1),ADX(N3,2),ADX(N3,3),                   &
                    ADY(N3,1),ADY(N3,2),ADY(N3,3),                   &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),MINONE,2,       &
                    TRAX,TRBX,TRBY,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFY(N3,1),     &
                    TRFY(N3,2),TRFY(N3,3),CXX,CXY,Q1X(N2),Q22C(N4))
                ENDIF
                IF (Q1Y(N2).NE.0.D0) THEN
                  CALL R1R22C(L4E,L4GR(1),L4GR(2),L4GR(3),           &
                    ADY(N1,1),ADY(N1,2),ADY(N1,3),                   &
                    ADX(N3,1),ADX(N3,2),ADX(N3,3),                   &
                    ADY(N3,1),ADY(N3,2),ADY(N3,3),                   &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),MINONE,2,       &
                    TRAY,TRBX,TRBY,TRFY(N1,1),TRFY(N1,2),TRFY(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFY(N3,1),     &
                    TRFY(N3,2),TRFY(N3,3),CYX,CYY,Q1Y(N2),Q22C(N4))
                ENDIF
!!              IF (Q1Z(N2).NE.0.D0) THEN
                  CALL R1R22C(L4E,L4GR(1),L4GR(2),L4GR(3),           &
                    ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),                   &
                    ADX(N3,1),ADX(N3,2),ADX(N3,3),                   &
                    ADY(N3,1),ADY(N3,2),ADY(N3,3),                   &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),MINONE,2,       &
                    TRAZ,TRBX,TRBY,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFY(N3,1),     &
                    TRFY(N3,2),TRFY(N3,3),CZX,CZY,Q1Z(N2),Q22C(N4))
!!              ENDIF
              ENDIF
              IF (Q22S(N4).NE.0.D0) THEN
                IF (Q1Z(N2).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAZ,TRBX,TRBY,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFY(N3,1),     &
                    TRFY(N3,2),TRFY(N3,3),CZX,CZY,Q1X(N2),Q22S(N4))
                ENDIF
                IF (Q1X(N2).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAX,TRBX,TRBY,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFY(N3,1),     &
                    TRFY(N3,2),TRFY(N3,3),CXX,CXY,Q1Y(N2),Q22S(N4))
                ENDIF
                IF (Q1Y(N2).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRAY,TRBX,TRBY,TRFY(N1,1),TRFY(N1,2),TRFY(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFY(N3,1),     &
                    TRFY(N3,2),TRFY(N3,3),CYX,CYY,Q1Y(N2),Q22S(N4))
                ENDIF
              ENDIF
            ENDIF   ! RANK(N4)
            IF (RANK(N2).GT.1) THEN
              IF (Q20(N2).NE.0.D0) THEN
                IF (Q1Z(N4).NE.0.D0) THEN
                  CALL R1R20(L4E,L4GR(1),L4GR(2),L4GR(3),       &
                    ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),              &
                    ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),              &
                    TRBZ,TRAZ,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CZZ,Q1Z(N4),Q20(N2))
                ENDIF
                IF (Q1X(N4).NE.0.D0) THEN
                  CALL R1R20(L4E,L4GR(1),L4GR(2),L4GR(3),       &
                    ADX(N3,1),ADX(N3,2),ADX(N3,3),              &
                    ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),              &
                    TRBX,TRAZ,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CZX,Q1X(N4),Q20(N2))
                ENDIF
                IF (Q1Y(N4).NE.0.D0) THEN
                  CALL R1R20(L4E,L4GR(1),L4GR(2),L4GR(3),       &
                    ADY(N3,1),ADY(N3,2),ADY(N3,3),              &
                    ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),              &
                    TRBY,TRAZ,TRFY(N3,1),TRFY(N3,2),TRFY(N3,3), &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CZY,Q1Y(N4),Q20(N2))
                ENDIF
              ENDIF
              IF (Q21C(N2).NE.0.D0) THEN
                IF (Q1Z(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBZ,TRAX,TRAZ,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFZ(N1,1),     &
                    TRFZ(N1,2),TRFZ(N1,3),CXZ,CZZ,Q1Z(N4),Q21C(N2))
                ENDIF
                IF (Q1X(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBX,TRAX,TRAZ,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFZ(N1,1),     &
                    TRFZ(N1,2),TRFZ(N1,3),CXX,CZX,Q1X(N4),Q21C(N2))
                ENDIF
                IF (Q1Y(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBY,TRAX,TRAZ,TRFY(N3,1),TRFY(N3,2),TRFY(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFZ(N1,1),     &
                    TRFZ(N1,2),TRFZ(N1,3),CXY,CZY,Q1Y(N4),Q21C(N2))
                ENDIF
              ENDIF
              IF (Q21S(N2).NE.0.D0) THEN
                IF (Q1Z(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBZ,TRAY,TRAZ,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),TRFZ(N1,1),     &
                    TRFZ(N1,2),TRFZ(N1,3),CYZ,CZZ,Q1Z(N4),Q21S(N2))
                ENDIF
                IF (Q1X(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBX,TRAY,TRAZ,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),TRFZ(N1,1),     &
                    TRFZ(N1,2),TRFZ(N1,3),CYX,CZX,Q1X(N4),Q21S(N2))
                ENDIF
                IF (Q1Y(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBY,TRAY,TRAZ,TRFY(N3,1),TRFY(N3,2),TRFY(N3,3), &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),TRFZ(N1,1),     &
                    TRFZ(N1,2),TRFZ(N1,3),CYY,CZY,Q1Y(N4),Q21S(N2))
                ENDIF
              ENDIF
              IF (Q22C(N2).NE.0.D0) THEN
                IF (Q1X(N4).NE.0.D0) THEN
                  CALL R1R22C(L4E,L4GR(1),L4GR(2),L4GR(3),           &
                    ADX(N3,1),ADX(N3,2),ADX(N3,3),                   &
                    ADX(N1,1),ADX(N1,2),ADX(N1,3),                   &
                    ADY(N1,1),ADY(N1,2),ADY(N1,3),                   &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),ONE,1,          &
                    TRBX,TRAX,TRAY,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFY(N1,1),     &
                    TRFY(N1,2),TRFY(N1,3),CXX,CYX,Q1X(N4),Q22C(N2))
                ENDIF
                IF (Q1Y(N4).NE.0.D0) THEN
                  CALL R1R22C(L4E,L4GR(1),L4GR(2),L4GR(3),           &
                    ADY(N3,1),ADY(N3,2),ADY(N3,3),                   &
                    ADX(N1,1),ADX(N1,2),ADX(N1,3),                   &
                    ADY(N1,1),ADY(N1,2),ADY(N1,3),                   &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),ONE,1,          &
                    TRBY,TRAX,TRAY,TRFY(N3,1),TRFY(N3,2),TRFY(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFY(N1,1),     &
                    TRFY(N1,2),TRFY(N1,3),CXY,CYY,Q1Y(N4),Q22C(N2))
                ENDIF
!!              IF (Q1Z(N4).NE.0.D0) THEN
                  CALL R1R22C(L4E,L4GR(1),L4GR(2),L4GR(3),           &
                    ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),                   &
                    ADX(N1,1),ADX(N1,2),ADX(N1,3),                   &
                    ADY(N1,1),ADY(N1,2),ADY(N1,3),                   &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),ONE,1,          &
                    TRBZ,TRAX,TRAY,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFY(N1,1),     &
                    TRFY(N1,2),TRFY(N1,3),CXZ,CYZ,Q1Z(N4),Q22C(N2))
!!              ENDIF
              ENDIF
              IF (Q22S(N2).NE.0.D0) THEN
                IF (Q1Z(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBZ,TRAX,TRAY,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFY(N1,1),     &
                    TRFY(N1,2),TRFY(N1,3),CXZ,CYZ,Q1Z(N4),Q22S(N2))
                ENDIF
                IF (Q1X(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBX,TRAX,TRAY,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFY(N1,1),     &
                    TRFY(N1,2),TRFY(N1,3),CXX,CYX,Q1X(N4),Q22S(N2))
                ENDIF
                IF (Q1Y(N4).NE.0.D0) THEN
                  CALL R1R2I(L4E,L4GR(1),L4GR(2),L4GR(3),            &
                    TRBY,TRAX,TRAY,TRFY(N3,1),TRFY(N3,2),TRFY(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFY(N1,1),     &
                    TRFY(N1,2),TRFY(N1,3),CXY,CYY,Q1Y(N4),Q22S(N2))
                ENDIF
              ENDIF
            ENDIF
            TE = TE + L4E
            XGR = XGR + L4GR(1)
            YGR = YGR + L4GR(2)
            ZGR = ZGR + L4GR(3)
!c          L5 ----- L5 ----- L5 ----- L5 ----- L5 ----- L5 ----- L5 -----
            IF((RANK(N4) + RANK(N2)) .GT. 3)THEN
              L5E = 0.D0
              L5GR(1) = 0.D0
              L5GR(2) = 0.D0
              L5GR(3) = 0.D0
              RM5 = RM2*RM3
              RAX9 = NINE*RAX
              RAY9 = NINE*RAY
              RAZ9 = NINE*RAZ
!c            r1r3 -- r1r3 -- r1r3 -- r1r3 -- r1r3 -- r1r3 -- r1r3 -- r1r3 --
              IF (RANK(N4).GT.2) THEN
                IF (Q1Z(N2).NE.0.D0) THEN
                  CALL R1R3(L5E,L5GR(1),L5GR(2),L5GR(3),        &
                    TRAZ,TRBZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CZZ,Q1Z(N2),Q30(N4))
                ENDIF
                IF (Q1X(N2).NE.0.D0) THEN
                  CALL R1R3(L5E,L5GR(1),L5GR(2),L5GR(3),        &
                    TRAX,TRBZ,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CXZ,Q1X(N2),Q30(N4))
                ENDIF
                IF (Q1Y(N2).NE.0.D0) THEN
                  CALL R1R3(L5E,L5GR(1),L5GR(2),L5GR(3),        &
                    TRAY,TRBZ,TRFY(N1,1),TRFY(N1,2),TRFY(N1,3), &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CYZ,Q1Y(N2),Q30(N4))
                ENDIF
              ENDIF
!c            r3r1 -- r3r1 -- r3r1 -- r3r1 -- r3r1 -- r3r1 -- r3r1 -- r3r1 --
              IF (RANK(N2).GT.2) THEN
                IF (Q1Z(N4).NE.0.D0) THEN
                  CALL R1R3(L5E,L5GR(1),L5GR(2),L5GR(3),        &
                    TRBZ,TRAZ,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CZZ,Q1Z(N4),Q30(N2))
                ENDIF
                IF (Q1X(N4).NE.0.D0) THEN
                  CALL R1R3(L5E,L5GR(1),L5GR(2),L5GR(3),        &
                    TRBX,TRAZ,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CZX,Q1X(N4),Q30(N2))
                ENDIF
                IF (Q1Y(N4).NE.0.D0) THEN
                  CALL R1R3(L5E,L5GR(1),L5GR(2),L5GR(3),        &
                    TRBY,TRAZ,TRFY(N3,1),TRFY(N3,2),TRFY(N3,3), &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CZY,Q1Y(N4),Q30(N2))
                ENDIF
              ENDIF
!c            r2r2 -- r2r2 -- r2r2 -- r2r2 -- r2r2 -- r2r2 -- r2r2 -- r2r2 --
              IF (Q20(N2).NE.0.D0) THEN
                IF (Q20(N4).NE.0.D0) THEN
                  CALL R20R20(L5E,L5GR(1),L5GR(2),L5GR(3),      &
                    ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),              &
                    ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),              &
                    TRAZ,TRBZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CZZ,Q20(N2),Q20(N4))
                ENDIF
                IF (Q21C(N4).NE.0.D0) THEN
                  CALL R20R2I(L5E,L5GR(1),L5GR(2),L5GR(3),           &
                    TRAZ,TRBX,TRBZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFZ(N3,1),     &
                    TRFZ(N3,2),TRFZ(N3,3),CZX,CZZ,Q20(N2),Q21C(N4))
                ENDIF
                IF (Q21S(N4).NE.0.D0) THEN
                  CALL R20R2I(L5E,L5GR(1),L5GR(2),L5GR(3),           &
                    TRAZ,TRBY,TRBZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),TRFZ(N3,1),     &
                    TRFZ(N3,2),TRFZ(N3,3),CZY,CZZ,Q20(N2),Q21S(N4))
                ENDIF
                IF (Q22C(N4).NE.0.D0) THEN
                  CALL R20R22C(L5E,L5GR(1),L5GR(2),L5GR(3),          &
                    ADZ(N1,1),ADZ(N1,2),ADZ(N1,3),                   &
                    ADX(N3,1),ADX(N3,2),ADX(N3,3),                   &
                    ADY(N3,1),ADY(N3,2),ADY(N3,3),2,                 &
                    TRAZ,TRBX,TRBY,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFY(N3,1),     &
                    TRFY(N3,2),TRFY(N3,3),CZX,CZY,Q20(N2),Q22C(N4))
                ENDIF
                IF (Q22S(N4).NE.0.D0) THEN
                  CALL R20R2I(L5E,L5GR(1),L5GR(2),L5GR(3),           &
                    TRAZ,TRBX,TRBY,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),TRFY(N3,1),     &
                    TRFY(N3,2),TRFY(N3,3),CZX,CZY,Q20(N2),Q22S(N4))
                ENDIF
              ENDIF
              IF (Q21C(N2).NE.0.D0) THEN
                IF (Q20(N4).NE.0.D0) THEN
                  CALL R20R2I(L5E,L5GR(1),L5GR(2),L5GR(3),           &
                    TRBZ,TRAX,TRAZ,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),TRFZ(N1,1),     &
                    TRFZ(N1,2),TRFZ(N1,3),CXZ,CZZ,Q20(N4),Q21C(N2))
                ENDIF
                IF (Q21C(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAX,TRAZ,TRBX,TRBZ,                   &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),      &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),      &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),      &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),      &
                    CXX,CXZ,CZX,CZZ,Q21C(N2),Q21C(N4))
                ENDIF
                IF (Q21S(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAX,TRAZ,TRBY,TRBZ,                   &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),      &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),      &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),      &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),      &
                    CXY,CXZ,CYX,CZZ,Q21C(N2),Q21S(N4))
                ENDIF
                IF (Q22C(N4).NE.0.D0) THEN
                  CALL R2IR22C(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAX,TRAZ,TRBX,TRBY,                    &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),       &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),       &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),       &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),       &
                    CXX,CXY,CZX,CZY,Q21C(N2),Q22C(N4))
                ENDIF
                IF (Q22S(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAX,TRAZ,TRBX,TRBY,                   &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),      &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),      &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),      &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),      &
                    CXX,CXY,CZX,CZY,Q21C(N2),Q22S(N4))
                ENDIF
              ENDIF
              IF (Q21S(N2).NE.0.D0) THEN
                IF (Q20(N4).NE.0.D0) THEN
                  CALL R20R2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRBZ,TRAY,TRAZ,                        &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),      &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),      &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),      &
                    CYZ,CZZ,Q20(N4),Q21S(N2))
                ENDIF
                IF (Q21C(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAY,TRAZ,TRBX,TRBZ,                   &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),      &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),      &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),      &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),      &
                    CYX,CYZ,CZX,CZZ,Q21S(N2),Q21C(N4))
                ENDIF
                IF (Q21S(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAY,TRAZ,TRBY,TRBZ,                   &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),      &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),      &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),      &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),      &
                    CYY,CYZ,CZY,CZZ,Q21S(N2),Q21S(N4))
                ENDIF
                IF (Q22C(N4).NE.0.D0) THEN
                  CALL R2IR22C(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAY,TRAZ,TRBX,TRBY,                    &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),       &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),       &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),       &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),       &
                    CYX,CYY,CZX,CZY,Q21S(N2),Q22C(N4))
                ENDIF
                IF (Q22S(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAY,TRAZ,TRBX,TRBY,                   &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),      &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),      &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),      &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),      &
                    CYX,CYY,CZX,CZY,Q21S(N2),Q22S(N4))
                ENDIF
              ENDIF
              IF (Q22C(N2).NE.0.D0) THEN
                IF (Q20(N4).NE.0.D0) THEN
                  CALL R20R22C(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    ADZ(N3,1),ADZ(N3,2),ADZ(N3,3),          &
                    ADX(N1,1),ADX(N1,2),ADX(N1,3),          &
                    ADY(N1,1),ADY(N1,2),ADY(N1,3),          &
                    1,TRBZ,TRAX,TRAY,                       &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),       &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),       &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),       &
                    CXZ,CYZ,Q20(N4),Q22C(N2))
                ENDIF
                IF (Q21C(N4).NE.0.D0) THEN
                  CALL R2IR22C(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRBX,TRBZ,TRAX,TRAY,                    &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),       &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),       &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),       &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),       &
                    CXX,CYX,CXZ,CYZ,Q21C(N4),Q22C(N2))
                ENDIF
                IF (Q21S(N4).NE.0.D0) THEN
                  CALL R2IR22C(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRBY,TRBZ,TRAX,TRAY,                    &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),       &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),       &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),       &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),       &
                    CXY,CYY,CXZ,CYZ,Q21S(N4),Q22C(N2))
                ENDIF
                IF (Q22C(N4).NE.0.D0) THEN
                  CALL R22CR22C(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    ADX(N1,1),ADX(N1,2),ADX(N1,3),           &
                    ADY(N1,1),ADY(N1,2),ADY(N1,3),           &
                    TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),ONE,    &
                    ADX(N3,1),ADX(N3,2),ADX(N3,3),           &
                    ADY(N3,1),ADY(N3,2),ADY(N3,3),           &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),        &
                    MINONE,TRAX,TRAY,TRBX,TRBY,              &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),        &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),        &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),        &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),        &
                    CXX,CXY,CYX,CYY,Q22C(N4),Q22C(N2))
                ENDIF
                IF (Q22S(N4).NE.0.D0) THEN
                  CALL R2IR22C(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRBX,TRBY,TRAX,TRAY,                    &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),       &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),       &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),       &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),       &
                    CXX,CXY,CYX,CYY,Q22S(N4),Q22C(N2))
                ENDIF
              ENDIF
              IF (Q22S(N2).NE.0.D0) THEN
                IF (Q20(N4).NE.0.D0) THEN
                  CALL R20R2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRBZ,TRAX,TRAY,                        &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),      &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),      &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),      &
                    CXZ,CYZ,Q20(N4),Q22S(N2))
                ENDIF
                IF (Q21C(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAX,TRAY,TRBX,TRBZ,                   &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),      &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),      &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),      &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),      &
                    CXX,CXZ,CYX,CYZ,Q22S(N2),Q21C(N4))
                ENDIF
                IF (Q21S(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAX,TRAY,TRBY,TRBZ,                   &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),      &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),      &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),      &
                    TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),      &
                    CXY,CYZ,CYY,CXZ,Q22S(N2),Q21S(N4))
                ENDIF
                IF (Q22C(N4).NE.0.D0) THEN
                  CALL R2IR22C(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAX,TRAY,TRBX,TRBY,                    &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),       &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),       &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),       &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),       &
                    CXX,CXY,CYX,CYY,Q22S(N2),Q22C(N4))
                ENDIF
                IF (Q22S(N4).NE.0.D0) THEN
                  CALL R2IR2I(L5E,L5GR(1),L5GR(2),L5GR(3), &
                    TRAX,TRAY,TRBX,TRBY,                   &
                    TRFX(N1,1),TRFX(N1,2),TRFX(N1,3),      &
                    TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),      &
                    TRFX(N3,1),TRFX(N3,2),TRFX(N3,3),      &
                    TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),      &
                    CXX,CXY,CYX,CYY,Q22S(N2),Q22S(N4))
                ENDIF
              ENDIF
              TE = TE + L5E
              XGR = XGR + L5GR(1)
              YGR = YGR + L5GR(2)
              ZGR = ZGR + L5GR(3)
!c            L6 ----- L6 ----- L6 ----- L6 ----- L6 ----- L6 ----- L6 -----
              IF((RANK(N4) + RANK(N2)) .GT. 4)THEN
                L6E = 0.D0
                L6GR(1) = 0.D0
                L6GR(2) = 0.D0
                L6GR(3) = 0.D0
                RM6 = RM3*RM3
!c              r2r3 -- r2r3 -- r2r3 -- r2r3 -- r2r3 -- r2r3 -- r2r3 -- r2r3 --
                IF (Q30(N4).NE.0.D0) THEN
                  IF (Q20(N2).NE.0.D0) THEN
                    CALL R20R30(L6E,L6GR(1),L6GR(2),L6GR(3),      &
                      TRAZ,TRBZ,TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3), &
                      TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3),CZZ,Q20(N2),Q30(N4))
                  ENDIF
                  IF (Q22C(N4).NE.0.D0) THEN
                    CALL R22CR30(L6E,L6GR(1),L6GR(2),L6GR(3),          &
                      TRAX,TRAY,TRBZ,TRFX(N1,1),TRFX(N1,2),TRFX(N1,3), &
                      TRFY(N1,1),TRFY(N1,2),TRFY(N1,3),TRFZ(N3,1),     &
                      TRFZ(N3,2),TRFZ(N3,3),CXZ,CYZ,Q22C(N2),Q30(N4))
                  ENDIF
                ENDIF
!c              r3r2 -- r3r2 -- r3r2 -- r3r2 -- r3r2 -- r3r2 -- r3r2 -- r3r2 --
                IF (Q30(N2).NE.0.D0) THEN
                  IF (Q20(N4).NE.0.D0) THEN
                    CALL R20R30(L6E,L6GR(1),L6GR(2),L6GR(3),      &
                      TRBZ,TRAZ,TRFZ(N3,1),TRFZ(N3,2),TRFZ(N3,3), &
                      TRFZ(N1,1),TRFZ(N1,2),TRFZ(N1,3),CZZ,Q20(N4),Q30(N2))
                  ENDIF
                  IF (Q22c(N4).NE.0.D0) THEN
                    CALL R22CR30(L6E,L6GR(1),L6GR(2),L6GR(3),          &
                      TRBX,TRBY,TRAZ,TRFX(N3,1),TRFX(N3,2),TRFX(N3,3), &
                      TRFY(N3,1),TRFY(N3,2),TRFY(N3,3),TRFZ(N1,1),     &
                      TRFZ(N1,2),TRFZ(N1,3),CZX,CZY,Q22C(N4),Q30(N2))
                  ENDIF
                ENDIF
                TE = TE + L6E
                XGR = XGR + L6GR(1)
                YGR = YGR + L6GR(2)
                ZGR = ZGR + L6GR(3)
              ENDIF   ! L6
            ENDIF   ! L5
          ENDIF   ! L4
        ENDIF   ! L3
      ENDIF   ! L2

      IF (N3 .NE. 0 ) THEN
      ! change signs back (needed for other interactions)
         TRFX(N3,1) = -TRFX(N3,1)
         TRFX(N3,2) = -TRFX(N3,2)
         TRFX(N3,3) = -TRFX(N3,3)
         TRFY(N3,1) = -TRFY(N3,1)
         TRFY(N3,2) = -TRFY(N3,2)
         TRFY(N3,3) = -TRFY(N3,3)
         TRFZ(N3,1) = -TRFZ(N3,1)
         TRFZ(N3,2) = -TRFZ(N3,2)
         TRFZ(N3,3) = -TRFZ(N3,3)
         ADX(N3,1) = -ADX(N3,1)
         ADX(N3,2) = -ADX(N3,2)
         ADX(N3,3) = -ADX(N3,3)
         ADY(N3,1) = -ADY(N3,1)
         ADY(N3,2) = -ADY(N3,2)
         ADY(N3,3) = -ADY(N3,3)
         ADZ(N3,1) = -ADZ(N3,1)
         ADZ(N3,2) = -ADZ(N3,2)
         ADZ(N3,3) = -ADZ(N3,3)
      ENDIF

      RETURN
      END SUBROUTINE MTPXCORE
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE REFAX(NATOMX,X,Y,Z)
! Calculates orienation parameters for each molecule 
! Fluctuating charges and moments are updated
!
      use chm_kinds
      use stream
      use dimens_fcm
      use consta
      use param
      use psf
      use number
      use image
      implicit none
!
!     C C C C C local and passed variables
      INTEGER I,J,NATOMX
      real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
      real(chm_real) DRX(5,5),DRY(5,5),DRZ(5,5),DIST,DDF,VAR(10)
!
      DO I=1, NMOL
        !  -----------------------------------------------------
        IF ( RFNR(I,3) .EQ. 0 ) THEN
        !  linear linear linear linear linear linear linear
        !  definition of signs has to be adapted to parameters
          DRX(1,2) = X(RFNR(I,2)) - X(RFNR(I,1))
          DRY(1,2) = Y(RFNR(I,2)) - Y(RFNR(I,1))
          DRZ(1,2) = Z(RFNR(I,2)) - Z(RFNR(I,1))
          DIST = SQRT((DRX(1,2))**2 + (DRY(1,2))**2 + (DRZ(1,2))**2)
          DEL0(I,3) = DIST
          TRFZ(I,1) = DRX(1,2)/DIST
          TRFZ(I,2) = DRY(1,2)/DIST
          TRFZ(I,3) = DRZ(1,2)/DIST
          IF (QFLC(I)) THEN
          ! update fluctuating charges and moments
            DDF = DIST - CBBEQDI(I)/BOHRR   ! in bohr
            DDF = DDF*BOHRR                 ! convert to angstrom
            DO J=(RFNR(I,1)), (RFNR(I,2))
              CG(J) = Q1(J) + Q2(J)*DDF
              Q1Z(J) = Q1Z1(J) + Q1Z2(J)*DDF
              IF (RANK(J) .GT. 1) then
                Q20(J) = Q201(J) + Q202(J)*DDF
                Q22C(J) = Q22C1(J) + Q22C2(J)*DDF
                IF (RANK(J) .GT. 2) then
                  Q30(J) = Q301(J) + Q302(J)*DDF
                ENDIF
              ENDIF
            ENDDO   ! J
          ENDIF   ! QFLC(I)
! BEGIN RRKR
! END RRKR
        !  -----------------------------------------------------
        ELSE IF ( RFNR(I,4) .EQ. 0 ) THEN
          IF (LINT(I)) THEN
          !   Linear triatomic linear triatomic linear triatomic
            DRX(2,3) = X(RFNR(I,2)) - X(RFNR(I,3))
            DRY(2,3) = Y(RFNR(I,2)) - Y(RFNR(I,3))
            DRZ(2,3) = Z(RFNR(I,2)) - Z(RFNR(I,3))
            DIST = SQRT((DRX(2,3))**2 + (DRY(2,3))**2 + (DRZ(2,3))**2)
            TRFZ(I,1) = DRX(2,3)/DIST
            TRFZ(I,2) = DRY(2,3)/DIST
            TRFZ(I,3) = DRZ(2,3)/DIST
            DEL0(I,3) = DIST
          ELSE
          !    C2v  C2v  C2v  C2v  C2v  C2v  C2v  C2v  C2v  C2v 
            DRX(1,2) = X(RFNR(I,1)) - X(RFNR(I,2))
            DRX(1,3) = X(RFNR(I,1)) - X(RFNR(I,3))
            DRX(2,3) = X(RFNR(I,2)) - X(RFNR(I,3))
            DRY(1,2) = Y(RFNR(I,1)) - Y(RFNR(I,2))
            DRY(1,3) = Y(RFNR(I,1)) - Y(RFNR(I,3))
            DRY(2,3) = Y(RFNR(I,2)) - Y(RFNR(I,3))
            DRZ(1,2) = Z(RFNR(I,1)) - Z(RFNR(I,2))
            DRZ(1,3) = Z(RFNR(I,1)) - Z(RFNR(I,3))
            DRZ(2,3) = Z(RFNR(I,2)) - Z(RFNR(I,3))
!
            VAR(1) = HALF*(DRX(1,2) + DRX(1,3))
            VAR(2) = HALF*(DRY(1,2) + DRY(1,3))
            VAR(3) = HALF*(DRZ(1,2) + DRZ(1,3))
            DIST = SQRT(VAR(1)**2 + VAR(2)**2 + VAR(3)**2)
            DEL0(I,3) = DIST   ! O-c(H-H) distance (c: center)
            VAR(1) = VAR(1)/DIST
            VAR(2) = VAR(2)/DIST
            VAR(3) = VAR(3)/DIST
            TRFZ(I,1) = VAR(1)
            TRFZ(I,2) = VAR(2)
            TRFZ(I,3) = VAR(3)
!
            DIST = SQRT((DRX(2,3))**2 + (DRY(2,3))**2 + (DRZ(2,3))**2)
            DEL0(I,2) = DIST   ! H-H distance
            VAR(4) = DRX(2,3)/DIST
            VAR(5) = DRY(2,3)/DIST
            VAR(6) = DRZ(2,3)/DIST
            TRFY(I,1) = VAR(4)
            TRFY(I,2) = VAR(5)
            TRFY(I,3) = VAR(6)
!
            DEL0(I,1) = DEL0(I,2)   ! ???
            TRFX(I,1) = TRFY(I,1)*(TRFZ(I,1)**2)                    &
                      + TRFY(I,2)*(TRFZ(I,2)*TRFZ(I,1) + TRFZ(I,3)) &
                      + TRFY(I,3)*(TRFZ(I,3)*TRFZ(I,1) - TRFZ(I,2))
            TRFX(I,2) = TRFY(I,1)*(TRFZ(I,1)*TRFZ(I,2) - TRFZ(I,3)) &
                      + TRFY(I,2)*(TRFZ(I,2)**2)                    &
                      + TRFY(I,3)*(TRFZ(I,3)*TRFZ(I,2) + TRFZ(I,1))
            TRFX(I,3) = TRFY(I,1)*(TRFZ(I,1)*TRFZ(I,3) + TRFZ(I,2)) &
                      + TRFY(I,2)*(TRFZ(I,2)*TRFZ(I,3) - TRFZ(I,1)) &
                      + TRFY(I,3)*(TRFZ(I,3)**2)
!BEGIN KKY
!END KKY
          ENDIF
!       
        ENDIF
!
      ENDDO
!
      RETURN
      END SUBROUTINE REFAX
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE REFAXD(I,NATOMX,X,Y,Z,XYZDA)
! Calculates orienation parameters for one molecule with a dummy atom
!
      use chm_kinds
      use dimens_fcm
      use consta
      use param
      use psf
      use number
      use image
      implicit none
!
! C C C C C local and passed variables
      INTEGER I,NATOMX
      real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX), XYZDA(3)
!     real(chm_real) TRFX(MOLN,3),TRFY(MOLN,3),TRFZ(MOLN,3)
      real(chm_real) TRFXD(3),TRFYD(3),TRFZD(3)
      real(chm_real) VT12(3), VT13(3), VTEM(3)
      real(chm_real) DPROD

      VT12(1) = X(RFNR(I,2)) - X(RFNR(I,1))
      VT12(2) = Y(RFNR(I,2)) - Y(RFNR(I,1))
      VT12(3) = Z(RFNR(I,2)) - Z(RFNR(I,1))
      VT13(1) = XYZDA(1)     - X(RFNR(I,1))
      VT13(2) = XYZDA(2)     - Y(RFNR(I,1))
      VT13(3) = XYZDA(3)     - Z(RFNR(I,1))

      CALL VECNORM(VT12, TRFZD)   ! unit vector e(N->O)

      DPROD = VT13(1)*TRFZD(1) + VT13(2)*TRFZD(2) + VT13(3)*TRFZD(3)
!     CALL VECSUM2(DOTPROD(VT13,TRFZD), VT13,TRFZD, VTEM)
      CALL VECSUM2(DPROD, VT13,TRFZD, VTEM)
      CALL VECNORM(VTEM, TRFYD)   ! unit vector perpendicular to e(N->O)

      CALL VECPROD2(TRFYD,TRFZD,TRFXD)

      TRFX(I,1) = TRFXD(1)
      TRFX(I,2) = TRFXD(2)
      TRFX(I,3) = TRFXD(3)
      TRFY(I,1) = TRFYD(1)
      TRFY(I,2) = TRFYD(2)
      TRFY(I,3) = TRFYD(3)
      TRFZ(I,1) = TRFZD(1)
      TRFZ(I,2) = TRFZD(2)
      TRFZ(I,3) = TRFZD(3)

      RETURN
      END SUBROUTINE REFAXD
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R0R1(EN,XGRD,YGRD,ZGRD,AD1,AD2,AD3, &
                TRF1,TRF2,TRF3,TRA,KU,KU1)
!     calculates charge-dipole interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,ENER,XGRD,YGRD,ZGRD,AD1,AD2,AD3
      real(chm_real) TRF1,TRF2,TRF3,TRA,KU,KU1
!
      ENER = KU*KU1*RM2
      XGRD = ENER*(TRF1 - RAX3*TRA) + XGRD
      YGRD = ENER*(TRF2 - RAY3*TRA) + YGRD
      ZGRD = ENER*(TRF3 - RAZ3*TRA) + ZGRD
      EN = ENER*TRA + EN
!
      AD1 = AD1 + ENER*(RAX - TRF1*TRA)
      AD2 = AD2 + ENER*(RAY - TRF2*TRA)
      AD3 = AD3 + ENER*(RAZ - TRF3*TRA)
!
      RETURN
      END SUBROUTINE R0R1
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R1R1(EN,XGRD,YGRD,ZGRD,CAB, &
                AD1,AD2,AD3,BD1,BD2,BD3,     &
                TRA1,TRA2,TRA3,TRB1,TRB2,TRB3,TRA,TRB,K1A,K1B)
!     calculates dipole-dipole interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,CAB,ENER(2)
      real(chm_real) TRA1,TRA2,TRA3,TRB1,TRB2,TRB3,TRA,TRB,K1A,K1B
      real(chm_real) AD1,AD2,AD3,BD1,BD2,BD3
!
      ENER(1) = K1A*K1B*THREE*RM3
      ENER(2) = K1A*K1B*RM3
      EN = ENER(1)*TRA*TRB + ENER(2)*CAB + EN
      XGRD = ENER(1)*(TRA1*TRB + TRB1*TRA - RAX5*TRA*TRB) &
           - ENER(2)*RAX3*CAB + XGRD
      YGRD = ENER(1)*(TRA2*TRB + TRB2*TRA - RAY5*TRA*TRB) &
           - ENER(2)*RAY3*CAB + YGRD
      ZGRD = ENER(1)*(TRA3*TRB + TRB3*TRA - RAZ5*TRA*TRB) &
           - ENER(2)*RAZ3*CAB + ZGRD
!
      AD1 = AD1 + ENER(1)*(RAX*TRB - TRA1*TRA*TRB) &
                - ENER(2)*(TRB1 + TRA1*CAB)
      AD2 = AD2 + ENER(1)*(RAY*TRB - TRA2*TRA*TRB) &
                - ENER(2)*(TRB2 + TRA2*CAB)
      AD3 = AD3 + ENER(1)*(RAZ*TRB - TRA3*TRA*TRB) &
                - ENER(2)*(TRB3 + TRA3*CAB)
      BD1 = BD1 + ENER(1)*(RAX*TRA - TRB1*TRA*TRB) &
                - ENER(2)*(TRA1 + TRB1*CAB)
      BD2 = BD2 + ENER(1)*(RAY*TRA - TRB2*TRA*TRB) &
                - ENER(2)*(TRA2 + TRB2*CAB)
      BD3 = BD3 + ENER(1)*(RAZ*TRA - TRB3*TRA*TRB) &
                - ENER(2)*(TRA3 + TRB3*CAB)
!
      RETURN
      END SUBROUTINE R1R1
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R0R20(EN,XGRD,YGRD,ZGRD,AD1,AD2,AD3, &
                TRBZ,TRB1,TRB2,TRB3,KU,KU20)
!     calculates charge-20 inteRaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(2),AD1,AD2,AD3
      real(chm_real) TRBZ,TRB1,TRB2,TRB3,KU,KU20
!
      ENER(1) = KU*KU20*RM3*HALF*THREE*TRBZ
      ENER(2) = -KU*KU20*RM3*HALF
      EN = ENER(1)*TRBZ + ENER(2) + EN
      XGRD = ENER(1)*(TWO*TRB1 - RAX5*TRBZ) &
           - ENER(2)*RAX3 + XGRD
      YGRD = ENER(1)*(TWO*TRB2 - RAY5*TRBZ) &
           - ENER(2)*RAY3 + YGRD
      ZGRD = ENER(1)*(TWO*TRB3 - RAZ5*TRBZ) &
           - ENER(2)*RAZ3 + ZGRD
!
      AD1 = AD1 + ENER(1)*TWO*(RAX - TRB1*TRBZ)
      AD2 = AD2 + ENER(1)*TWO*(RAY - TRB2*TRBZ)
      AD3 = AD3 + ENER(1)*TWO*(RAZ - TRB3*TRBZ)
!
      RETURN
      END SUBROUTINE R0R20
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R0R2I(EN,XGRD,YGRD,ZGRD, &
                TRBI,TRBJ,TRI1,TRI2,TRI3,TRJ1,TRJ2,TRJ3,KU,KU2IJ)
!     calculates charge-(21c,21s,22s) interaction
!C
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER
      real(chm_real) TRBI,TRBJ,TRI1,TRI2,TRI3,TRJ1,TRJ2,TRJ3,KU,KU2IJ
!
      ENER = KU*KU2IJ*RM3*1.732050808D0
      XGRD = ENER*(TRI1*TRBJ + TRJ1*TRBI - RAX5*TRBI*TRBJ) + XGRD
      YGRD = ENER*(TRI2*TRBJ + TRJ2*TRBI - RAY5*TRBI*TRBJ) + YGRD
      ZGRD = ENER*(TRI3*TRBJ + TRJ3*TRBI - RAZ5*TRBI*TRBJ) + ZGRD
      EN = ENER*TRBI*TRBJ + EN
!
      RETURN
      END SUBROUTINE R0R2I
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R0R22C(EN,XGRD,YGRD,ZGRD, &
                AX1,AX2,AX3,AY1,AY2,AY3,   &
                TZ1,TZ2,TZ3,S,N,           &
                TRBX,TRBY,TRX1,TRX2,TRX3,TRY1,TRY2,TRY3,KU,KU22C)
!     calculates charge-22c interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      INTEGER N
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(2),CONST
      real(chm_real) AX1,AX2,AX3,AY1,AY2,AY3
      real(chm_real) TZ1,TZ2,TZ3,S
      real(chm_real) TRBX,TRBY,TRX1,TRX2,TRX3,TRY1,TRY2,TRY3,KU,KU22C
!
      CONST = KU*KU22C*RM3*0.866025404D0
      ENER(1) = CONST*TRBX
      ENER(2) = -CONST*TRBY
      EN = ENER(1)*TRBX + ENER(2)*TRBY + EN
      XGRD = ENER(1)*(TWO*TRX1 - RAX5*TRBX) + &
             ENER(2)*(TWO*TRY1 - RAX5*TRBY) + XGRD
      YGRD = ENER(1)*(TWO*TRX2 - RAY5*TRBX) + &
             ENER(2)*(TWO*TRY2 - RAY5*TRBY) + YGRD
      ZGRD = ENER(1)*(TWO*TRX3 - RAZ5*TRBX) + &
             ENER(2)*(TWO*TRY3 - RAZ5*TRBY) + ZGRD
!
      AY1 = AY1 + ENER(2)*TWO*(RAX - TRY1*TRBY)
      AY2 = AY2 + ENER(2)*TWO*(RAY - TRY2*TRBY)
      AY3 = AY3 + ENER(2)*TWO*(RAZ - TRY3*TRBY)

      ! only for x-axis defined by orthogonality (C2v)
      ORTH(N,1) = (TZ1**2*RAX + RAY*(TZ1*TZ2 - S*TZ3) &
                 + RAZ*(TZ1*TZ3 + S*TZ2)) - TRY1*TRBX
      ORTH(N,2) = (RAX*(TZ2*TZ1 + S*TZ3)+TZ2**2*RAY   &
                 + RAZ*(TZ2*TZ3 - S*TZ1)) - TRY2*TRBX
      ORTH(N,3) = (RAX*(TZ3*TZ1 - S*TZ2) + RAY*(TZ3*TZ2 + S*TZ1) &
                 + RAZ*TZ3**2) - TRY3*TRBX
      FORT(N,1) = TRY1*TWO*(TZ1 - TZ1**3) + TRY2*TZ2                  &
               - TWO*TRY2*TZ2*TZ1**2 + TRY3*TZ3 - TWO*TRY3*TZ3*TZ1**2 &
                 - S*TRY2*TZ1*TZ3 + S*TRY3*TZ2*TZ1
      FORT(N,2) = TRY1*TZ2 - TWO*TRY1*TZ1**2*TZ2                      &
                 - TWO*TRY2*TZ2**2*TZ1 - TWO*TRY3*TZ3*TZ1*TZ2         &
                 + S*TRY1*TZ3*TZ1 + S*TRY3 - S*TRY3*TZ1**2
      FORT(N,3) = TRY1*TZ3 - TWO*TRY3*TZ1*TZ3**2                      &
                 - TWO*TRY1*TZ1**2*TZ3 - TWO*TRY2*TZ2*TZ1*TZ3         &
                 - S*TRY1*TZ1*TZ2 - S*TRY2 + S*TRY2*TZ1**2
      ORTH(N,4) = RAX*FORT(N,1) + RAY*FORT(N,2) + RAZ*FORT(N,3)
      FORT(N,4) = TRY2*TZ1 - TWO*TRY1*TZ1**2*TZ2              &
                 - TWO*TRY2*TZ2**2*TZ1 - TWO*TRY3*TZ3*TZ1*TZ2 &
                 - S*TRY2*TZ3*TZ2 - S*TRY3 + S*TRY3*TZ2**2
      FORT(N,5) = TWO*TRY2*TZ2- TWO*TRY2*TZ2**3 + TRY1*TZ1            &
               - TWO*TRY1*TZ2**2*TZ1 + TRY3*TZ3 - TWO*TRY3*TZ3*TZ2**2 &
                 + S*TRY1*TZ3*TZ2 - S*TRY3*TZ1*TZ2
      FORT(N,6) = TRY2*TZ3 - TWO*TRY3*TZ3**2*TZ2              &
                 - TWO*TRY1*TZ1*TZ2*TZ3 - TWO*TRY2*TZ2**2*TZ3 &
                 + S*TRY1 - S*TRY1*TZ2**2 + S*TRY2*TZ1*TZ2
      ORTH(N,5) = RAX*FORT(N,4) + RAY*FORT(N,5) + RAZ*FORT(N,6)
      FORT(N,7) = TRY3*TZ1 - TWO*TRY1*TZ1**2*TZ3              &
                 - TWO*TZ3*TRY2*TZ2*TZ1 - TWO*TRY3*TZ3**2*TZ1 &
                 + S*TRY2 - S*TRY2*TZ3**2 + S*TRY3*TZ3*TZ2
      FORT(N,8) = TRY3*TZ2 - TWO*TRY3*TZ3**2*TZ2              &
                 - TWO*TRY2*TZ2**2*TZ3 - TWO*TRY1*TZ2*TZ1*TZ3 &
                 - S*TRY1 + S*TRY1*TZ3**2 - S*TRY3*TZ1*TZ3
      FORT(N,9) = TWO*TRY3*TZ3- TWO*TRY3*TZ3**3 + TRY1*TZ1            &
               - TWO*TRY1*TZ1*TZ3**2 + TRY2*TZ2 - TWO*TRY2*TZ2*TZ3**2 &
                 - S*TRY1*TZ2*TZ3 + S*TRY2*TZ1*TZ3
      ORTH(N,6) = RAX*FORT(N,7) + RAY*FORT(N,8) + RAZ*FORT(N,9)
      AY1 = AY1 + ENER(1)*TWO*ORTH(N,1)
      AY2 = AY2 + ENER(1)*TWO*ORTH(N,2)
      AY3 = AY3 + ENER(1)*TWO*ORTH(N,3)
      AX1 = AX1 + ENER(1)*TWO*ORTH(N,4)
      AX2 = AX2 + ENER(1)*TWO*ORTH(N,5)
      AX3 = AX3 + ENER(1)*TWO*ORTH(N,6)
!
      RETURN
      END SUBROUTINE R0R22C
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R0R3(EN,XGRD,YGRD,ZGRD,AD1,AD2,AD3, &
                TRBZ,TRZ1,TRZ2,TRZ3,KU,KU30)
!     calculates charge-30 interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(2),AD1,AD2,AD3
      real(chm_real) TRBZ,TRZ1,TRZ2,TRZ3,KU,KU30
!
      ENER(1) = KU*KU30*RM4*HALF*FIVE*TRBZ**2
      ENER(2) = -KU*KU30*RM4*HALF*THREE
      EN = (ENER(1) + ENER(2))*TRBZ + EN
      XGRD = ENER(1)*(THREE*TRZ1 - RAX7*TRBZ) &
           + ENER(2)*(TRZ1 - RAX5*TRBZ) + XGRD
      YGRD = ENER(1)*(THREE*TRZ2 - RAY7*TRBZ) &
           + ENER(2)*(TRZ2 - RAY5*TRBZ) + YGRD
      ZGRD = ENER(1)*(THREE*TRZ3 - RAZ7*TRBZ) &
           + ENER(2)*(TRZ3 - RAZ5*TRBZ) + ZGRD
!
      AD1 = AD1 + ENER(1)*THREE*(RAX - TRZ1*TRBZ) &
                + ENER(2)*(RAX - TRZ1*TRBZ)
      AD2 = AD2 + ENER(1)*THREE*(RAY - TRZ2*TRBZ) &
                + ENER(2)*(RAY - TRZ2*TRBZ)
      AD3 = AD3 + ENER(1)*THREE*(RAZ - TRZ3*TRBZ) &
                + ENER(2)*(RAZ - TRZ3*TRBZ)
!
      RETURN
      END SUBROUTINE R0R3
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R1R20(EN,XGRD,YGRD,ZGRD, &
                AD1,AD2,AD3,BD1,BD2,BD3,  &
                TRAI,TRBZ,TRI1,TRI2,TRI3,TRZ1,TRZ2,TRZ3,CIZ,KU1I,KU20)
!     calculates dipole-20 interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(3),CONST
      real(chm_real) TRAI,TRBZ,TRI1,TRI2,TRI3,TRZ1,TRZ2,TRZ3, &
                     CIZ,KU1I,KU20
      real(chm_real) AD1,AD2,AD3,BD1,BD2,BD3
!
      CONST = KU1I*KU20*HALF*RM4
      ENER(1) = CONST*FIFTN*(TRBZ)
      ENER(2) = CONST*SIX
      ENER(3) = -CONST*THREE
      EN = (ENER(1)*TRAI + ENER(2)*CIZ)*TRBZ + ENER(3)*TRAI + EN
      XGRD = ENER(1)*(TWO*TRZ1*TRAI + TRI1*TRBZ - RAX7*TRBZ*TRAI) &
           + ENER(2)*CIZ*(TRZ1 - RAX5*TRBZ)                       &
           + ENER(3)*(TRI1 - RAX5*TRAI) + XGRD
      YGRD = ENER(1)*(TWO*TRZ2*TRAI + TRI2*TRBZ - RAY7*TRBZ*TRAI) &
           + ENER(2)*CIZ*(TRZ2 - RAY5*TRBZ)                       &
           + ENER(3)*(TRI2 - RAY5*TRAI) + YGRD
      ZGRD = ENER(1)*(TWO*TRZ3*TRAI + TRI3*TRBZ - RAZ7*TRBZ*TRAI) &
           + ENER(2)*CIZ*(TRZ3 - RAZ5*TRBZ)                       &
           + ENER(3)*(TRI3 - RAZ5*TRAI) + ZGRD
!
      AD1 = AD1 + (ENER(1)*TRBZ + ENER(3))*(RAX - TRI1*TRAI) &
                 - ENER(2)*TRBZ*(TRZ1 + TRI1*CIZ)
      AD2 = AD2 + (ENER(1)*TRBZ + ENER(3))*(RAY - TRI2*TRAI) &
                 - ENER(2)*TRBZ*(TRZ2 + TRI2*CIZ)
      AD3 = AD3 + (ENER(1)*TRBZ + ENER(3))*(RAZ - TRI3*TRAI) &
                 - ENER(2)*TRBZ*(TRZ3 + TRI3*CIZ)
      BD1 = BD1 + ENER(1)*TWO*(RAX - TRZ1*TRBZ)*TRAI &
            - ENER(2)*(TRI1*TRBZ - RAX*CIZ + TWO*TRZ1*TRBZ*CIZ)
      BD2 = BD2 + ENER(1)*TWO*(RAY - TRZ2*TRBZ)*TRAI &
            - ENER(2)*(TRI2*TRBZ - RAY*CIZ + TWO*TRZ2*TRBZ*CIZ)
      BD3 = BD3 + ENER(1)*TWO*(RAZ - TRZ3*TRBZ)*TRAI &
            - ENER(2)*(TRI3*TRBZ - RAZ*CIZ + TWO*TRZ3*TRBZ*CIZ)
!
      RETURN
      END SUBROUTINE R1R20
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R1R2I(EN,XGRD,YGRD,ZGRD,                    &
                TRA,TRBI,TRBJ,TRA1,TRA2,TRA3,TRI1,TRI2,TRI3, &
                TRJ1,TRJ2,TRJ3,CAI,CAJ,KU1A,KU2IJ)
!     calculates dipole-(21c,21s,22s) interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(3),CONST
      real(chm_real) TRA,TRBI,TRBJ,TRA1,TRA2,TRA3,TRI1,TRI2,TRI3
      real(chm_real) TRJ1,TRJ2,TRJ3,CAI,CAJ,KU1A,KU2IJ
!
      CONST = KU1A*KU2IJ*1.732050808D0*RM4
      ENER(1) = CONST*CAI
      ENER(2) = CONST*CAJ
      ENER(3) = CONST*FIVE
      EN = ENER(1)*TRBI + ENER(2)*TRBJ + ENER(3)*TRBI*TRBJ*TRA + EN
      XGRD = ENER(1)*(TRI1 - RAX5*TRBI)                              &
           + ENER(2)*(TRJ1 - RAX5*TRBJ)                              &
           + ENER(3)*(TRI1*TRBJ*TRA + TRJ1*TRBI*TRA + TRA1*TRBI*TRBJ &
                    - RAX7*TRBI*TRBJ*TRA) + XGRD
      YGRD = ENER(1)*(TRI2 - RAY5*TRBI)                              &
           + ENER(2)*(TRJ2 - RAY5*TRBJ)                              &
           + ENER(3)*(TRI2*TRBJ*TRA + TRJ2*TRBI*TRA + TRA2*TRBI*TRBJ &
                    - RAY7*TRBI*TRBJ*TRA) + YGRD
      ZGRD = ENER(1)*(TRI3 - RAZ5*TRBI)                              &
           + ENER(2)*(TRJ3 - RAZ5*TRBJ)                              &
           + ENER(3)*(TRI3*TRBJ*TRA + TRJ3*TRBI*TRA + TRA3*TRBI*TRBJ &
                    - RAZ7*TRBI*TRBJ*TRA) + ZGRD
!
      RETURN
      END SUBROUTINE R1R2I
!--------------------------------------------------------------------- 
!--------------------------------------------------------------------- 
      SUBROUTINE R1R22C(EN,XGRD,YGRD,ZGRD,                   &
                BI1,BI2,BI3,AX1,AX2,AX3,AY1,AY2,AY3,         &
                TRZ1,TRZ2,TRZ3,S,N,                          &
                TRA,TRBX,TRBY,TRA1,TRA2,TRA3,TRX1,TRX2,TRX3, &
                TRY1,TRY2,TRY3,CAX,CAY,KU1A,KU22C)
!     calculates dipole-22c interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      INTEGER N
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(4),CONST
      real(chm_real) TRA,TRBX,TRBY,TRA1,TRA2,TRA3,TRX1,TRX2,TRX3
      real(chm_real) TRY1,TRY2,TRY3,CAX,CAY,KU1A,KU22C
      real(chm_real) BI1,BI2,BI3,AX1,AX2,AX3,AY1,AY2,AY3
      real(chm_real) TRZ1,TRZ2,TRZ3,S
!
      CONST = KU1A*KU22C*0.866025404D0*RM4
      ENER(1) = CONST*FIVE*TRBX
      ENER(2) =-CONST*FIVE*TRBY
      ENER(3) = CONST*TWO
      ENER(4) =-CONST*TWO
      EN = (ENER(1)*TRBX + ENER(2)*TRBY)*TRA + ENER(3)*TRBX*CAX &
          + ENER(4)*TRBY*CAY + EN
      XGRD = ENER(1)*(TWO*TRX1*TRA + TRA1*TRBX - RAX7*TRBX*TRA) &
            +ENER(2)*(TWO*TRY1*TRA + TRA1*TRBY - RAX7*TRBY*TRA) &
            +ENER(3)*(TRX1 - RAX5*TRBX)*CAX                     &
            +ENER(4)*(TRY1 - RAX5*TRBY)*CAY + XGRD
      YGRD = ENER(1)*(TWO*TRX2*TRA + TRA2*TRBX - RAY7*TRBX*TRA) &
            +ENER(2)*(TWO*TRY2*TRA + TRA2*TRBY - RAY7*TRBY*TRA) &
            +ENER(3)*(TRX2 - RAY5*TRBX)*CAX                     &
            +ENER(4)*(TRY2 - RAY5*TRBY)*CAY + YGRD
      ZGRD = ENER(1)*(TWO*TRX3*TRA + TRA3*TRBX - RAZ7*TRBX*TRA) &
            +ENER(2)*(TWO*TRY3*TRA + TRA3*TRBY - RAZ7*TRBY*TRA) &
            +ENER(3)*(TRX3 - RAZ5*TRBX)*CAX                     &
            +ENER(4)*(TRY3 - RAZ5*TRBY)*CAY + ZGRD  
!
! change values for cax
      ORTH(N,7) = (TRZ1**2*TRA1 + TRA2*(TRZ1*TRZ2 - S*TRZ3) &
               + TRA3*(TRZ1*TRZ3 + S*TRZ2)) + TRY1 * CAX
      ORTH(N,8) = (TRA1*(TRZ2*TRZ1 + S*TRZ3)+TRZ2**2*TRA2   &
               + TRA3*(TRZ2*TRZ3 - S*TRZ1)) + TRY2 * CAX
      ORTH(N,9) = (TRA1*(TRZ3*TRZ1 - S*TRZ2)                &
               + TRA2*(TRZ3*TRZ2+S*TRZ1) + TRA3*TRZ3**2) + TRY3 * CAX
      ORTH(N,10) = (TRA1*FORT(N,1) +TRA2*FORT(N,2) +TRA3*FORT(N,3))
      ORTH(N,11) = (TRA1*FORT(N,4) +TRA2*FORT(N,5) +TRA3*FORT(N,6))
      ORTH(N,12) = (TRA1*FORT(N,7) +TRA2*FORT(N,8) +TRA3*FORT(N,9))
!
      BI1 = BI1 + ((ENER(1)*TRBX + ENER(2)*TRBY)*(RAX - TRA1*TRA) &
           - ENER(3)*TRBX*(TRX1 + TRA1*CAX)                  &
           - ENER(4)*TRBY*(TRY1 + TRA1*CAY))
      BI2 = BI2 + ((ENER(1)*TRBX + ENER(2)*TRBY)*(RAY - TRA2*TRA) &
           - ENER(3)*TRBX*(TRX2 + TRA2*CAX)                  &
           - ENER(4)*TRBY*(TRY2 + TRA2*CAY))
      BI3 = BI3 + ((ENER(1)*TRBX + ENER(2)*TRBY)*(RAZ - TRA3*TRA) &
           - ENER(3)*TRBX*(TRX3 + TRA3*CAX)                  &
           - ENER(4)*TRBY*(TRY3 + TRA3*CAY))
      AY1 = AY1 + ENER(2)*TWO*TRA*(RAX - TRY1*TRBY) &
           - ENER(4)*(TRA1*TRBY - RAX*CAY + TWO*TRY1*TRBY*CAY)
      AY2 = AY2 + ENER(2)*TWO*TRA*(RAY - TRY2*TRBY) &
           - ENER(4)*(TRA2*TRBY - RAY*CAY + TWO*TRY2*TRBY*CAY)
      AY3 = AY3 + ENER(2)*TWO*TRA*(RAZ - TRY3*TRBY) &
           - ENER(4)*(TRA3*TRBY - RAZ*CAY + TWO*TRY3*TRBY*CAY)
      AY1 = AY1 + ENER(1)*TRA*TWO*ORTH(N,1) +ENER(3)*(ORTH(N,1)*CAX-ORTH(N,7)*TRBX)
      AY2 = AY2 + ENER(1)*TRA*TWO*ORTH(N,2) +ENER(3)*(ORTH(N,2)*CAX-ORTH(N,8)*TRBX)
      AY3 = AY3 + ENER(1)*TRA*TWO*ORTH(N,3) +ENER(3)*(ORTH(N,3)*CAX-ORTH(N,9)*TRBX)
      AX1 = AX1 + ENER(1)*TRA*TWO*ORTH(N,4) +ENER(3)*(ORTH(N,4)*CAX-ORTH(N,10)*TRBX)
      AX2 = AX2 + ENER(1)*TRA*TWO*ORTH(N,5) +ENER(3)*(ORTH(N,5)*CAX-ORTH(N,11)*TRBX)
      AX3 = AX3 + ENER(1)*TRA*TWO*ORTH(N,6) +ENER(3)*(ORTH(N,6)*CAX-ORTH(N,12)*TRBX)
!
      RETURN
      END SUBROUTINE R1R22C
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R1R3(EN,XGRD,YGRD,ZGRD, &
                TRA,TRBZ,TRA1,TRA2,TRA3,TRZ1,TRZ2,TRZ3,CAZ,KU1A,KU30)
!     calculates dipole-30 interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(4),CONST
      real(chm_real) TRA,TRBZ,TRA1,TRA2,TRA3,TRZ1,TRZ2,TRZ3
      real(chm_real) CAZ,KU1A,KU30
!
      CONST = KU1A*KU30*HALF*RM5
      ENER(1) = CONST*35D0*TRBZ**2
      ENER(2) = CONST*FIFTN*TRBZ*CAZ
      ENER(3) = -CONST*FIFTN
      ENER(4) = -CONST*THREE*CAZ
      EN = (ENER(1)*TRA + ENER(2) + ENER(3)*TRA)*TRBZ + ENER(4) + EN
      XGRD = ENER(2)*(TWO*TRZ1 - RAX7*TRBZ)                 &
            +ENER(3)*(TRZ1*TRA + TRA1*TRBZ - RAX7*TRBZ*TRA) &
            -ENER(4)*RAX5 + XGRD                            &
            +ENER(1)*(TRA1*TRBZ + THREE*TRZ1*TRA - EIGHT*RAX*TRBZ*TRA)
      YGRD = ENER(2)*(TWO*TRZ2 - RAY7*TRBZ)                 &
            +ENER(3)*(TRZ2*TRA + TRA2*TRBZ - RAY7*TRBZ*TRA) &
            -ENER(4)*RAY5 + YGRD                            &
            +ENER(1)*(TRA2*TRBZ + THREE*TRZ2*TRA - EIGHT*RAY*TRBZ*TRA)
      ZGRD = ENER(2)*(TWO*TRZ3 - RAZ7*TRBZ)                 &
            +ENER(3)*(TRZ3*TRA + TRA3*TRBZ - RAZ7*TRBZ*TRA) &
            -ENER(4)*RAZ5 + ZGRD                            &
            +ENER(1)*(TRA3*TRBZ + THREE*TRZ3*TRA - EIGHT*RAZ*TRBZ*TRA)
!
      RETURN
      END SUBROUTINE R1R3
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R20R20(EN,XGRD,YGRD,ZGRD,               &
                AD1,AD2,AD3,BD1,BD2,BD3,                 &
                TRAZ,TRBZ,TAZ1,TAZ2,TAZ3,TBZ1,TBZ2,TBZ3, &
                CZZ,KU20A,KU20B)
!     calculates 20-20 interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(5),CONST
      real(chm_real) TRAZ,TRBZ,TAZ1,TAZ2,TAZ3,TBZ1,TBZ2,TBZ3
      real(chm_real) CZZ,KU20A,KU20B
      real(chm_real) AD1,AD2,AD3,BD1,BD2,BD3
!
      CONST = KU20A*KU20B*PT75*RM5
      ENER(1) = CONST*35D0*TRAZ*TRBZ
      ENER(2) =-CONST*FIVE*TRAZ
      ENER(3) =-CONST*FIVE*TRBZ
      ENER(4) = CONST*TWENTY
      ENER(5) = CONST*TWO*(CZZ**2 + ONE)
      EN = (ENER(1)+ENER(4)*CZZ)*TRAZ*TRBZ + ENER(2)*TRAZ &
         + ENER(3)*TRBZ + ENER(5) + EN
      XGRD = ENER(1)*(TWO*TAZ1*TRBZ + TWO*TBZ1*TRAZ - RAX9*TRAZ*TRBZ) &
           + ENER(2)*(TWO*TAZ1 - RAX7*TRAZ)                           &
           + ENER(3)*(TWO*TBZ1 - RAX7*TRBZ)                           &
           + ENER(4)*(TAZ1*TRBZ + TBZ1*TRAZ - RAX7*TRAZ*TRBZ)*CZZ     &
           - ENER(5)*RAX5 + XGRD
      YGRD = ENER(1)*(TWO*TAZ2*TRBZ + TWO*TBZ2*TRAZ - RAY9*TRAZ*TRBZ) &
           + ENER(2)*(TWO*TAZ2 - RAY7*TRAZ)                           &
           + ENER(3)*(TWO*TBZ2 - RAY7*TRBZ)                           &
           + ENER(4)*(TAZ2*TRBZ + TBZ2*TRAZ - RAY7*TRAZ*TRBZ)*CZZ     &
           - ENER(5)*RAY5 + YGRD
      ZGRD = ENER(1)*(TWO*TAZ3*TRBZ + TWO*TBZ3*TRAZ - RAZ9*TRAZ*TRBZ) &
           + ENER(2)*(TWO*TAZ3 - RAZ7*TRAZ)                           &
           + ENER(3)*(TWO*TBZ3 - RAZ7*TRBZ)                           &
           + ENER(4)*(TAZ3*TRBZ + TBZ3*TRAZ - RAZ7*TRAZ*TRBZ)*CZZ     &
           - ENER(5)*RAZ5 + ZGRD
!
      AD1 = AD1 + (ENER(1)*TRBZ + ENER(2))*TWO*(RAX - TAZ1*TRAZ) &
            - ENER(4)*(TBZ1*TRAZ - RAX*CZZ + TAZ1*TRAZ*CZZ)*TRBZ &
            - CONST*FOUR*CZZ*(TBZ1 + TAZ1*CZZ)
      AD2 = AD2 + (ENER(1)*TRBZ + ENER(2))*TWO*(RAY - TAZ2*TRAZ) &
            - ENER(4)*(TBZ2*TRAZ - RAY*CZZ + TAZ2*TRAZ*CZZ)*TRBZ &
            - CONST*FOUR*CZZ*(TBZ2 + TAZ2*CZZ)
      AD3 = AD3 + (ENER(1)*TRBZ + ENER(2))*TWO*(RAZ - TAZ3*TRAZ) &
            - ENER(4)*(TBZ3*TRAZ - RAZ*CZZ + TAZ3*TRAZ*CZZ)*TRBZ &
            - CONST*FOUR*CZZ*(TBZ3 + TAZ3*CZZ)
      BD1 = BD1 + (ENER(1)*TRAZ + ENER(3))*TWO*(RAX - TBZ1*TRBZ) &
            - ENER(4)*(TAZ1*TRBZ - RAX*CZZ + TBZ1*TRBZ*CZZ)*TRAZ &
            - CONST*FOUR*CZZ*(TAZ1 + TBZ1*CZZ)
      BD2 = BD2 + (ENER(1)*TRAZ + ENER(3))*TWO*(RAY - TBZ2*TRBZ) &
            - ENER(4)*(TAZ2*TRBZ - RAY*CZZ + TBZ2*TRBZ*CZZ)*TRAZ &
            - CONST*FOUR*CZZ*(TAZ2 + TBZ2*CZZ)
      BD3 = BD3 + (ENER(1)*TRAZ + ENER(3))*TWO*(RAZ - TBZ3*TRBZ) &
            - ENER(4)*(TAZ3*TRBZ - RAZ*CZZ + TBZ3*TRBZ*CZZ)*TRAZ &
            - CONST*FOUR*CZZ*(TAZ3 + TBZ3*CZZ)
!
      RETURN
      END SUBROUTINE R20R20
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R20R2I(EN,XGRD,YGRD,ZGRD,                   &
               TRAZ,TRBI,TRBJ,TRZ1,TRZ2,TRZ3,TRI1,TRI2,TRI3, &
               TRJ1,TRJ2,TRJ3,CZI,CZJ,KU20,KU2IJ)
!     calculates 20-(21c,21s,22s) interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(5),CONST
      real(chm_real) TRAZ,TRBI,TRBJ,TRZ1,TRZ2,TRZ3,TRI1,TRI2,TRI3
      real(chm_real) TRJ1,TRJ2,TRJ3,CZI,CZJ,KU20,KU2IJ
!
      CONST = KU20*KU2IJ*0.866025404D0*RM5
      ENER(1) = CONST*35D0*TRAZ
      ENER(2) =-CONST*FIVE
      ENER(3) = CONST*TEN*CZJ
      ENER(4) = CONST*TEN*CZI
      ENER(5) = CONST*TWO*CZI*CZJ
      EN = (ENER(1)*TRAZ+ENER(2))*TRBI*TRBJ + (ENER(3)*TRBI + ENER(4)*TRBJ)*TRAZ &
          + ENER(5) + EN
      XGRD = ENER(1)*( (TWO*TRZ1 - RAX9*TRAZ)*TRBI*TRBJ &
                     + (TRI1*TRBJ + TRJ1*TRBI)*TRAZ )   &
           + ENER(2)*(TRI1*TRBJ + TRJ1*TRBI - RAX7*TRBI*TRBJ)  &
           + ENER(3)*(TRZ1*TRBI + TRI1*TRAZ - RAX7*TRAZ*TRBI)  &
           + ENER(4)*(TRZ1*TRBJ + TRJ1*TRAZ - RAX7*TRAZ*TRBJ)  &
           - ENER(5)*RAX5 + XGRD
      YGRD = ENER(1)*( (TWO*TRZ2 - RAY9*TRAZ)*TRBI*TRBJ &
                     + (TRI2*TRBJ + TRJ2*TRBI)*TRAZ )   &
           + ENER(2)*(TRI2*TRBJ + TRJ2*TRBI - RAY7*TRBI*TRBJ)  &
           + ENER(3)*(TRZ2*TRBI + TRI2*TRAZ - RAY7*TRAZ*TRBI)  &
           + ENER(4)*(TRZ2*TRBJ + TRJ2*TRAZ - RAY7*TRAZ*TRBJ)  &
           - ENER(5)*RAY5 + YGRD
      ZGRD = ENER(1)*( (TWO*TRZ3 - RAZ9*TRAZ)*TRBI*TRBJ &
                     + (TRI3*TRBJ + TRJ3*TRBI)*TRAZ )   &
           + ENER(2)*(TRI3*TRBJ + TRJ3*TRBI - RAZ7*TRBI*TRBJ)  &
           + ENER(3)*(TRZ3*TRBI + TRI3*TRAZ - RAZ7*TRAZ*TRBI)  &
           + ENER(4)*(TRZ3*TRBJ + TRJ3*TRAZ - RAZ7*TRAZ*TRBJ)  &
           - ENER(5)*RAZ5 + ZGRD
!
      RETURN
      END SUBROUTINE R20R2I
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R20R22C(EN,XGRD,YGRD,ZGRD,                   &
                BI1,BI2,BI3,AX1,AX2,AX3,AY1,AY2,AY3,N,        &
                TRAZ,TRBX,TRBY,TRZ1,TRZ2,TRZ3,TBX1,TBX2,TBX3, &
                TBY1,TBY2,TBY3,CZX,CZY,KU20,KU22C)
!     calculates 20-22c interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      INTEGER N
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(7),CONST
      real(chm_real) TRAZ,TRBX,TRBY,TRZ1,TRZ2,TRZ3,TBX1,TBX2,TBX3
      real(chm_real) TBY1,TBY2,TBY3,CZX,CZY,KU20,KU22C
      real(chm_real) BI1,BI2,BI3,AX1,AX2,AX3,AY1,AY2,AY3
!
      CONST = KU20*KU22C*0.433012702D0*RM5
      ENER(1) = CONST*35D0*TRAZ*TRBX
      ENER(2) =-CONST*35D0*TRAZ*TRBY
      ENER(3) =-CONST*FIVE*TRBX
      ENER(4) = CONST*FIVE*TRBY
      ENER(5) = CONST*TWENTY
      ENER(6) =-CONST*TWENTY
      ENER(7) = CONST*TWO*(CZX**2 - CZY**2)
      EN = (ENER(1)*TRBX + ENER(2)*TRBY)*TRAZ + ENER(3)*TRBX + ENER(4)*TRBY &
          +(ENER(5)*TRBX*CZX + ENER(6)*TRBY*CZY)*TRAZ + ENER(7) + EN
      XGRD = ENER(1)*(TWO*TRZ1*TRBX + TWO*TBX1*TRAZ - RAX9*TRAZ*TRBX) &
            +ENER(2)*(TWO*TRZ1*TRBY + TWO*TBY1*TRAZ - RAX9*TRAZ*TRBY) &
            +ENER(3)*(TWO*TBX1 - RAX7*TRBX)                           &
            +ENER(4)*(TWO*TBY1 - RAX7*TRBY)                           &
            +ENER(5)*(TRZ1*TRBX + TBX1*TRAZ - RAX7*TRAZ*TRBX)*CZX     &
            +ENER(6)*(TRZ1*TRBY + TBY1*TRAZ - RAX7*TRAZ*TRBY)*CZY     &
            -ENER(7)*RAX5 + XGRD
      YGRD = ENER(1)*(TWO*TRZ2*TRBX + TWO*TBX2*TRAZ - RAY9*TRAZ*TRBX) &
            +ENER(2)*(TWO*TRZ2*TRBY + TWO*TBY2*TRAZ - RAY9*TRAZ*TRBY) &
            +ENER(3)*(TWO*TBX2 - RAY7*TRBX)                           &
            +ENER(4)*(TWO*TBY2 - RAY7*TRBY)                           &
            +ENER(5)*(TRZ2*TRBX + TBX2*TRAZ - RAY7*TRAZ*TRBX)*CZX     &
            +ENER(6)*(TRZ2*TRBY + TBY2*TRAZ - RAY7*TRAZ*TRBY)*CZY     &
            -ENER(7)*RAY5 + YGRD
      ZGRD = ENER(1)*(TWO*TRZ3*TRBX + TWO*TBX3*TRAZ - RAZ9*TRAZ*TRBX) &
            +ENER(2)*(TWO*TRZ3*TRBY + TWO*TBY3*TRAZ - RAZ9*TRAZ*TRBY) &
            +ENER(3)*(TWO*TBX3 - RAZ7*TRBX)                           &
            +ENER(4)*(TWO*TBY3 - RAZ7*TRBY)                           &
            +ENER(5)*(TRZ3*TRBX + TBX3*TRAZ - RAZ7*TRAZ*TRBX)*CZX     &
            +ENER(6)*(TRZ3*TRBY + TBY3*TRAZ - RAZ7*TRAZ*TRBY)*CZY     &
            -ENER(7)*RAZ5 + ZGRD
!
      BI1 = BI1 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*(RAX - TRZ1*TRAZ) &
           - ENER(5)*TRBX*(TBX1*TRAZ - RAX*CZX + TWO*TRZ1*TRAZ*CZX)   &
           - ENER(6)*TRBY*(TBY1*TRAZ - RAX*CZY + TWO*TRZ1*TRAZ*CZY)   &
           - CONST*TWO*CZY*(-TWO*TBY1 + TRZ1*CZY)                     &
           - CONST*TWO*CZX*( TWO*TBX1 - TRZ1*CZX)
      BI2 = BI2 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*(RAY - TRZ2*TRAZ) &
           - ENER(5)*TRBX*(TBX2*TRAZ - RAY*CZX + TWO*TRZ2*TRAZ*CZX)   &
           - ENER(6)*TRBY*(TBY2*TRAZ - RAY*CZY + TWO*TRZ2*TRAZ*CZY)   &
           - CONST*TWO*CZY*(-TWO*TBY2 + TRZ2*CZY)                     &
           - CONST*TWO*CZX*( TWO*TBX2 - TRZ2*CZX)
      BI3 = BI3 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*(RAZ - TRZ3*TRAZ) &
           - ENER(5)*TRBX*(TBX3*TRAZ - RAZ*CZX + TWO*TRZ3*TRAZ*CZX)   &
           - ENER(6)*TRBY*(TBY3*TRAZ - RAZ*CZY + TWO*TRZ3*TRAZ*CZY)   &
           - CONST*TWO*CZY*(-TWO*TBY3 + TRZ3*CZY)                     &
           - CONST*TWO*CZX*( TWO*TBX3 - TRZ3*CZX)
      AY1 = AY1 + (ENER(2)*TRAZ+ENER(4))*TWO*(RAX - TBY1*TRBY)        &
           - ENER(6)*(TRZ1*TRBY - RAX*CZY + TWO*TBY1*TRBY*CZY)*TRAZ   &
           + CONST*TWO*CZY*(TWO*TRZ1 - TBY1*CZY)
      AY2 = AY2 + (ENER(2)*TRAZ+ENER(4))*TWO*(RAY - TBY2*TRBY)        &
           - ENER(6)*(TRZ2*TRBY - RAY*CZY + TWO*TBY2*TRBY*CZY)*TRAZ   &
           + CONST*TWO*CZY*(TWO*TRZ2 - TBY2*CZY)
      AY3 = AY3 + (ENER(2)*TRAZ+ENER(4))*TWO*(RAZ - TBY3*TRBY)        &
           - ENER(6)*(TRZ3*TRBY - RAZ*CZY + TWO*TBY3*TRBY*CZY)*TRAZ   &
           + CONST*TWO*CZY*(TWO*TRZ3 - TBY3*CZY)
      AY1 = AY1 + (ENER(1)*TRAZ + ENER(3))*TWO*ORTH(N,1)  &
            + ENER(5)*(ORTH(N,1)*CZX-ORTH(N,7)*TRBX)*TRAZ &
            - CONST*FOUR*CZX*ORTH(N,7)
      AY2 = AY2 + (ENER(1)*TRAZ + ENER(3))*TWO*ORTH(N,2)  &
            + ENER(5)*(ORTH(N,2)*CZX-ORTH(N,8)*TRBX)*TRAZ &
            - CONST*FOUR*CZX*ORTH(N,8)
      AY3 = AY3 + (ENER(1)*TRAZ + ENER(3))*TWO*ORTH(N,3)  &
            + ENER(5)*(ORTH(N,3)*CZX-ORTH(N,9)*TRBX)*TRAZ &
            - CONST*FOUR*CZX*ORTH(N,9)
      AX1 = AX1 + (ENER(1)*TRAZ + ENER(3))*TWO*ORTH(N,4)   &
            + ENER(5)*(ORTH(N,4)*CZX-ORTH(N,10)*TRBX)*TRAZ &
            - CONST*FOUR*CZX*ORTH(N,10)
      AX2 = AX2 + (ENER(1)*TRAZ + ENER(3))*TWO*ORTH(N,5)   &
            + ENER(5)*(ORTH(N,5)*CZX-ORTH(N,11)*TRBX)*TRAZ &
            - CONST*FOUR*CZX*ORTH(N,11)
      AX3 = AX3 + (ENER(1)*TRAZ + ENER(3))*TWO*ORTH(N,6)   &
            + ENER(5)*(ORTH(N,6)*CZX-ORTH(N,12)*TRBX)*TRAZ &
            - CONST*FOUR*CZX*ORTH(N,12)
!
      RETURN
      END SUBROUTINE R20R22C
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R2IR2I(EN,XGRD,YGRD,ZGRD,                         &
                TRAI,TRAJ,TRBI,TRBJ,TAI1,TAI2,TAI3,TAJ1,TAJ2,TAJ3, &
                TBI1,TBI2,TBI3,TBJ1,TBJ2,TBJ3,                     &
                CAIBI,CAIBJ,CAJBI,CAJBJ,KUA2I,KUB2I)
!     calculates (21c,21s,22s)-(21c,21s,22s) interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(6),CONST
      real(chm_real) TRAI,TRAJ,TRBI,TRBJ,TAI1,TAI2,TAI3,TAJ1,TAJ2,TAJ3
      real(chm_real) TBI1,TBI2,TBI3,TBJ1,TBJ2,TBJ3
      real(chm_real) CAIBI,CAIBJ,CAJBI,CAJBJ,KUA2I,KUB2I
!
      CONST = KUA2I*KUB2I*RM5
      ENER(1) = CONST*35D0
      ENER(2) = CONST*FIVE*CAJBJ
      ENER(3) = CONST*FIVE*CAJBI
      ENER(4) = CONST*FIVE*CAIBJ
      ENER(5) = CONST*FIVE*CAIBI
      ENER(6) = CONST*(CAIBI*CAIBJ + CAJBJ*CAJBI)
      EN = ENER(1)*TRAI*TRAJ*TRBI*TRBJ + (ENER(2)*TRBI+ENER(3)*TRBJ)*TRAI &
         + (ENER(4)*TRBI+ENER(5)*TRBJ)*TRAJ + ENER(6) + EN
      XGRD = ENER(1)*( (TAI1*TRAJ + TAJ1*TRAI)*TRBI*TRBJ            &
             + (TBI1*TRBJ + TBJ1*TRBI - RAX9*TRBI*TRBJ)*TRAI*TRAJ ) &
           + ENER(2)*(TAI1*TRBI + TBI1*TRAI - RAX7*TRAI*TRBI) &
           + ENER(3)*(TAI1*TRBJ + TBJ1*TRAI - RAX7*TRAI*TRBJ) &
           + ENER(4)*(TAJ1*TRBI + TBI1*TRAJ - RAX7*TRAJ*TRBI) &
           + ENER(5)*(TAJ1*TRBJ + TBJ1*TRAJ - RAX7*TRAJ*TRBJ) &
           - ENER(6)*RAX5 + XGRD
      YGRD = ENER(1)*( (TAI2*TRAJ + TAJ2*TRAI)*TRBI*TRBJ            &
             + (TBI2*TRBJ + TBJ2*TRBI - RAY9*TRBI*TRBJ)*TRAI*TRAJ ) &
           + ENER(2)*(TAI2*TRBI + TBI2*TRAI - RAY7*TRAI*TRBI) &
           + ENER(3)*(TAI2*TRBJ + TBJ2*TRAI - RAY7*TRAI*TRBJ) &
           + ENER(4)*(TAJ2*TRBI + TBI2*TRAJ - RAY7*TRAJ*TRBI) &
           + ENER(5)*(TAJ2*TRBJ + TBJ2*TRAJ - RAY7*TRAJ*TRBJ) &
           - ENER(6)*RAY5 + YGRD
      ZGRD = ENER(1)*( (TAI3*TRAJ + TAJ3*TRAI)*TRBI*TRBJ            &
             + (TBI3*TRBJ + TBJ3*TRBI - RAZ9*TRBI*TRBJ)*TRAI*TRAJ ) &
           + ENER(2)*(TAI3*TRBI + TBI3*TRAI - RAZ7*TRAI*TRBI) &
           + ENER(3)*(TAI3*TRBJ + TBJ3*TRAI - RAZ7*TRAI*TRBJ) &
           + ENER(4)*(TAJ3*TRBI + TBI3*TRAJ - RAZ7*TRAJ*TRBI) &
           + ENER(5)*(TAJ3*TRBJ + TBJ3*TRAJ - RAZ7*TRAJ*TRBJ) &
           - ENER(6)*RAZ5 + ZGRD
!
      RETURN
      END SUBROUTINE R2IR2I
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R2IR22C(EN,XGRD,YGRD,ZGRD,                        &
                TRAI,TRAJ,TRBX,TRBY,TAI1,TAI2,TAI3,TAJ1,TAJ2,TAJ3, &
                TBX1,TBX2,TBX3,TBY1,TBY2,TBY3,                     &
                CIX,CIY,CJX,CJY,KU2I,KU22C)
!     calculates (21c,21s,22s)-22c interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(7),CONST
      real(chm_real) TRAI,TRAJ,TRBX,TRBY,TAI1,TAI2,TAI3,TAJ1,TAJ2,TAJ3
      real(chm_real) TBX1,TBX2,TBX3,TBY1,TBY2,TBY3
      real(chm_real) CIX,CIY,CJX,CJY,KU2I,KU22C
!
      CONST = KU2I*KU22C*HALF*RM5
      ENER(1) = CONST*35D0*TRBX
      ENER(2) =-CONST*35D0*TRBY
      ENER(3) = CONST*TEN*CJX
      ENER(4) =-CONST*TEN*CJY
      ENER(5) = CONST*TEN*CIX
      ENER(6) =-CONST*TEN*CIY
      ENER(7) = CONST*TWO*(CIX*CJX - CIY*CJY)
      EN = (ENER(1)*TRBX + ENER(2)*TRBY)*TRAI*TRAJ + (ENER(3)*TRBX  &
          + ENER(4)*TRBY)*TRAI + (ENER(5)*TRBX + ENER(6)*TRBY)*TRAJ + ENER(7) + EN
      XGRD = ENER(1)*( (TWO*TBX1 - RAX9*TRBX)*TRAI*TRAJ + (TAI1*TRAJ + TAJ1*TRAI)*TRBX ) &
           + ENER(2)*( (TWO*TBY1 - RAX9*TRBY)*TRAI*TRAJ + (TAI1*TRAJ + TAJ1*TRAI)*TRBY ) &
           + ENER(3)*(TAI1*TRBX + TBX1*TRAI - RAX7*TRAI*TRBX)   &
           + ENER(4)*(TAI1*TRBY + TBY1*TRAI - RAX7*TRAI*TRBY)   &
           + ENER(5)*(TAJ1*TRBX + TBX1*TRAJ - RAX7*TRAJ*TRBX)   &
           + ENER(6)*(TAJ1*TRBY + TBY1*TRAJ - RAX7*TRAJ*TRBY)   &
           - ENER(7)*RAX5 + XGRD
      YGRD = ENER(1)*( (TWO*TBX2 - RAY9*TRBX)*TRAI*TRAJ + (TAI2*TRAJ + TAJ2*TRAI)*TRBX ) &
           + ENER(2)*( (TWO*TBY2 - RAY9*TRBY)*TRAI*TRAJ + (TAI2*TRAJ + TAJ2*TRAI)*TRBY ) &
           + ENER(3)*(TAI2*TRBX + TBX2*TRAI - RAY7*TRAI*TRBX)   &
           + ENER(4)*(TAI2*TRBY + TBY2*TRAI - RAY7*TRAI*TRBY)   &
           + ENER(5)*(TAJ2*TRBX + TBX2*TRAJ - RAY7*TRAJ*TRBX)   &
           + ENER(6)*(TAJ2*TRBY + TBY2*TRAJ - RAY7*TRAJ*TRBY)   &
           - ENER(7)*RAY5 + YGRD
      ZGRD = ENER(1)*( (TWO*TBX3 - RAZ9*TRBX)*TRAI*TRAJ + (TAI3*TRAJ + TAJ3*TRAI)*TRBX ) &
           + ENER(2)*( (TWO*TBY3 - RAZ9*TRBY)*TRAI*TRAJ + (TAI3*TRAJ + TAJ3*TRAI)*TRBY ) &
           + ENER(3)*(TAI3*TRBX + TBX3*TRAI - RAZ7*TRAI*TRBX)   &
           + ENER(4)*(TAI3*TRBY + TBY3*TRAI - RAZ7*TRAI*TRBY)   &
           + ENER(5)*(TAJ3*TRBX + TBX3*TRAJ - RAZ7*TRAJ*TRBX)   &
           + ENER(6)*(TAJ3*TRBY + TBY3*TRAJ - RAZ7*TRAJ*TRBY)   &
           - ENER(7)*RAZ5 + ZGRD
! 
      RETURN
      END SUBROUTINE R2IR22C
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R22CR22C(EN,XGRD,YGRD,ZGRD,                       &
                AX1,AX2,AX3,AY1,AY2,AY3,TAZ1,TAZ2,TAZ3,SA,         &
                BX1,BX2,BX3,BY1,BY2,BY3,TBZ1,TBZ2,TBZ3,SB,         &
                TRAX,TRAY,TRBX,TRBY,TAX1,TAX2,TAX3,TAY1,TAY2,TAY3, &
                TBX1,TBX2,TBX3,TBY1,TBY2,TBY3,                     &
                CXX,CXY,CYX,CYY,KUA22C,KUB22C)
!     calculates 22c-22c interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(9),CONST
      real(chm_real) TRAX,TRAY,TRBX,TRBY,TAX1,TAX2,TAX3,TAY1,TAY2,TAY3
      real(chm_real) TBX1,TBX2,TBX3,TBY1,TBY2,TBY3
      real(chm_real) CXX,CXY,CYX,CYY,KUA22C,KUB22C
      real(chm_real) AX1,AX2,AX3,AY1,AY2,AY3,TAZ1,TAZ2,TAZ3,SA
      real(chm_real) BX1,BX2,BX3,BY1,BY2,BY3,TBZ1,TBZ2,TBZ3,SB
!
      CONST = KUA22C*KUB22C*PT25*RM5
      ENER(1) = CONST*35D0*TRAX*TRBX
      ENER(2) =-CONST*35D0*TRAX*TRBY
      ENER(3) =-CONST*35D0*TRAY*TRBX
      ENER(4) = CONST*35D0*TRAY*TRBY
      ENER(5) = CONST*TWENTY
      ENER(6) =-CONST*TWENTY
      ENER(7) =-CONST*TWENTY
      ENER(8) = CONST*TWENTY
      ENER(9) = CONST*TWO*(CXX**2 - CXY**2 - CYX**2 + CYY**2)
      EN = (ENER(1)*TRBX + ENER(2)*TRBY)*TRAX + (ENER(3)*TRBX + ENER(4)*TRBY)*TRAY &
         + (ENER(5)*TRBX*CXX + ENER(6)*TRBY*CXY)*TRAX &
         + (ENER(7)*TRBX*CYX + ENER(8)*TRBY*CYY)*TRAY + ENER(9) + EN
      XGRD = ENER(1)*(TWO*TAX1*TRBX + TWO*TBX1*TRAX - RAX9*TRAX*TRBX) &
           + ENER(2)*(TWO*TAX1*TRBY + TWO*TBY1*TRAX - RAX9*TRAX*TRBY) &
           + ENER(3)*(TWO*TAY1*TRBX + TWO*TBX1*TRAY - RAX9*TRAY*TRBX) &
           + ENER(4)*(TWO*TAY1*TRBY + TWO*TBY1*TRAY - RAX9*TRAY*TRBY) &
           + ENER(5)*(TAX1*TRBX + TBX1*TRAX - RAX7*TRAX*TRBX)*CXX     &
           + ENER(6)*(TAX1*TRBY + TBY1*TRAX - RAX7*TRAX*TRBY)*CXY     &
           + ENER(7)*(TAY1*TRBX + TBX1*TRAY - RAX7*TRAY*TRBX)*CYX     &
           + ENER(8)*(TAY1*TRBY + TBY1*TRAY - RAX7*TRAY*TRBY)*CYY     &
           - ENER(9)*RAX5 + XGRD
      YGRD = ENER(1)*(TWO*TAX2*TRBX + TWO*TBX2*TRAX - RAY9*TRAX*TRBX) &
           + ENER(2)*(TWO*TAX2*TRBY + TWO*TBY2*TRAX - RAY9*TRAX*TRBY) &
           + ENER(3)*(TWO*TAY2*TRBX + TWO*TBX2*TRAY - RAY9*TRAY*TRBX) &
           + ENER(4)*(TWO*TAY2*TRBY + TWO*TBY2*TRAY - RAY9*TRAY*TRBY) &
           + ENER(5)*(TAX2*TRBX + TBX2*TRAX - RAY7*TRAX*TRBX)*CXX     &
           + ENER(6)*(TAX2*TRBY + TBY2*TRAX - RAY7*TRAX*TRBY)*CXY     &
           + ENER(7)*(TAY2*TRBX + TBX2*TRAY - RAY7*TRAY*TRBX)*CYX     &
           + ENER(8)*(TAY2*TRBY + TBY2*TRAY - RAY7*TRAY*TRBY)*CYY     &
           - ENER(9)*RAY5 + YGRD
      ZGRD = ENER(1)*(TWO*TAX3*TRBX + TWO*TBX3*TRAX - RAZ9*TRAX*TRBX) &
           + ENER(2)*(TWO*TAX3*TRBY + TWO*TBY3*TRAX - RAZ9*TRAX*TRBY) &
           + ENER(3)*(TWO*TAY3*TRBX + TWO*TBX3*TRAY - RAZ9*TRAY*TRBX) &
           + ENER(4)*(TWO*TAY3*TRBY + TWO*TBY3*TRAY - RAZ9*TRAY*TRBY) &
           + ENER(5)*(TAX3*TRBX + TBX3*TRAX - RAZ7*TRAX*TRBX)*CXX     &
           + ENER(6)*(TAX3*TRBY + TBY3*TRAX - RAZ7*TRAX*TRBY)*CXY     &
           + ENER(7)*(TAY3*TRBX + TBX3*TRAY - RAZ7*TRAY*TRBX)*CYX     &
           + ENER(8)*(TAY3*TRBY + TBY3*TRAY - RAZ7*TRAY*TRBY)*CYY     &
           - ENER(9)*RAZ5 + ZGRD
!
      ORTH(1,13) = (TAZ1**2*TBX1 + TBX2*(TAZ1*TAZ2 - SA*TAZ3) &
                  + TBX3*(TAZ1*TAZ3 + SA*TAZ2)) + TAY1*CXX
      ORTH(1,14) = (TBX1*(TAZ2*TAZ1 + SA*TAZ3)+TAZ2**2*TBX2   &
                  + TBX3*(TAZ2*TAZ3 - SA*TAZ1)) + TAY2*CXX
      ORTH(1,15) = (TBX1*(TAZ3*TAZ1-SA*TAZ2)+TBX2*(TAZ3*TAZ2+SA*TAZ1) &
                  + TBX3*TAZ3**2) + TAY3*CXX
      ORTH(1,16) = (TBX1*FORT(1,1) +TBX2*FORT(1,2) +TBX3*FORT(1,3))
      ORTH(1,17) = (TBX1*FORT(1,4) +TBX2*FORT(1,5) +TBX3*FORT(1,6))
      ORTH(1,18) = (TBX1*FORT(1,7) +TBX2*FORT(1,8) +TBX3*FORT(1,9))
!
      ORTH(1,19) = (TAZ1**2*TBY1 + TBY2*(TAZ1*TAZ2 - SA*TAZ3) &
               + TBY3*(TAZ1*TAZ3 + SA*TAZ2)) + TAY1*CXY
      ORTH(1,20) = (TBY1*(TAZ2*TAZ1 + SA*TAZ3)+TAZ2**2*TBY2   &
               + TBY3*(TAZ2*TAZ3 - SA*TAZ1)) + TAY2*CXY
      ORTH(1,21) = (TBY1*(TAZ3*TAZ1-SA*TAZ2)+TBY2*(TAZ3*TAZ2+SA*TAZ1) &
               + TBY3*TAZ3**2) + TAY3*CXY
      ORTH(1,22) = (TBY1*FORT(1,1) +TBY2*FORT(1,2) +TBY3*FORT(1,3))
      ORTH(1,23) = (TBY1*FORT(1,4) +TBY2*FORT(1,5) +TBY3*FORT(1,6))
      ORTH(1,24) = (TBY1*FORT(1,7) +TBY2*FORT(1,8) +TBY3*FORT(1,9))
!
      ORTH(2,19) = (TBZ1**2*TAY1 + TAY2*(TBZ1*TBZ2 - SB*TBZ3) &
               + TAY3*(TBZ1*TBZ3 + SB*TBZ2)) + TBY1*CYX
      ORTH(2,20) = (TAY1*(TBZ2*TBZ1 + SB*TBZ3)+TBZ2**2*TAY2   &
               + TAY3*(TBZ2*TBZ3 - SB*TBZ1)) + TBY2*CYX
      ORTH(2,21) = (TAY1*(TBZ3*TBZ1-SB*TBZ2)+TAY2*(TBZ3*TBZ2+SB*TBZ1) &
               + TAY3*TBZ3**2) + TBY3*CYX
      ORTH(2,22) = (TAY1*FORT(2,1) +TAY2*FORT(2,2) +TAY3*FORT(2,3))
      ORTH(2,23) = (TAY1*FORT(2,4) +TAY2*FORT(2,5) +TAY3*FORT(2,6))
      ORTH(2,24) = (TAY1*FORT(2,7) +TAY2*FORT(2,8) +TAY3*FORT(2,9))
!
      ORTH(2,13) = (TBZ1**2*TAX1 + TAX2*(TBZ1*TBZ2 - SB*TBZ3) &
               + TAX3*(TBZ1*TBZ3 + SB*TBZ2)) + TBY1*CXX
      ORTH(2,14) = (TAX1*(TBZ2*TBZ1 + SB*TBZ3)+TBZ2**2*TAX2   &
               + TAX3*(TBZ2*TBZ3 - SB*TBZ1)) + TBY2*CXX
      ORTH(2,15) = (TAX1*(TBZ3*TBZ1-SB*TBZ2)+TAX2*(TBZ3*TBZ2+SB*TBZ1) &
               + TAX3*TBZ3**2) + TBY3*CXX
      ORTH(2,16) = (TAX1*FORT(2,1) +TAX2*FORT(2,2) +TAX3*FORT(2,3))
      ORTH(2,17) = (TAX1*FORT(2,4) +TAX2*FORT(2,5) +TAX3*FORT(2,6))
      ORTH(2,18) = (TAX1*FORT(2,7) +TAX2*FORT(2,8) +TAX3*FORT(2,9))
!
      AY1 = AY1 + ((ENER(3)*TRBX+ENER(4)*TRBY)*TWO*(RAX - TAY1*TRAY)     &
                - ENER(7)*(TBX1*TRAY - RAX*CYX + TWO*TAY1*TRAY*CYX)*TRBX &
                - ENER(8)*(TBY1*TRAY - RAX*CYY + TWO*TAY1*TRAY*CYY)*TRBY &
                + CONST*FOUR*(CYX*(TBX1 + TAY1*CYX)                      &
                - CYY*(TBY1 + TAY1*CYY)))
      AY2 = AY2 + ((ENER(3)*TRBX+ENER(4)*TRBY)*TWO*(RAY - TAY2*TRAY)     &
                - ENER(7)*(TBX2*TRAY - RAY*CYX + TWO*TAY2*TRAY*CYX)*TRBX &
                - ENER(8)*(TBY2*TRAY - RAY*CYY + TWO*TAY2*TRAY*CYY)*TRBY &
                + CONST*FOUR*(CYX*(TBX2 + TAY2*CYX)                      &
                - CYY*(TBY2 + TAY2*CYY)))
      AY3 = AY3 + ((ENER(3)*TRBX+ENER(4)*TRBY)*TWO*(RAZ - TAY3*TRAY)     &
                - ENER(7)*(TBX3*TRAY - RAZ*CYX + TWO*TAY3*TRAY*CYX)*TRBX &
                - ENER(8)*(TBY3*TRAY - RAZ*CYY + TWO*TAY3*TRAY*CYY)*TRBY &
                + CONST*FOUR*(CYX*(TBX3 + TAY3*CYX)                      &
                - CYY*(TBY3 + TAY3*CYY)))
      BY1 = BY1 + ((ENER(2)*TRAX+ENER(4)*TRAY)*TWO*(RAX - TBY1*TRBY)     &
                - ENER(6)*(TAX1*TRBY - RAX*CXY + TWO*TBY1*TRBY*CXY)*TRAX &
                - ENER(8)*(TAY1*TRBY - RAX*CYY + TWO*TBY1*TRBY*CYY)*TRAY &
                + CONST*FOUR*(CXY*(TAX1 + TBY1*CXY)                      &
                - CYY*(TAY1 + TBY1*CYY)))
      BY2 = BY2 + ((ENER(2)*TRAX+ENER(4)*TRAY)*TWO*(RAY - TBY2*TRBY)     &
                - ENER(6)*(TAX2*TRBY - RAY*CXY + TWO*TBY2*TRBY*CXY)*TRAX &
                - ENER(8)*(TAY2*TRBY - RAY*CYY + TWO*TBY2*TRBY*CYY)*TRAY &
                + CONST*FOUR*(CXY*(TAX2 + TBY2*CXY)                      &
                - CYY*(TAY2 + TBY2*CYY)))
      BY3 = BY3 + ((ENER(2)*TRAX+ENER(4)*TRAY)*TWO*(RAZ - TBY3*TRBY)     &
                - ENER(6)*(TAX3*TRBY - RAZ*CXY + TWO*TBY3*TRBY*CXY)*TRAX &
                - ENER(8)*(TAY3*TRBY - RAZ*CYY + TWO*TBY3*TRBY*CYY)*TRAY &
                + CONST*FOUR*(CXY*(TAX3 + TBY3*CXY)                      &
                - CYY*(TAY3 + TBY3*CYY)))
      AY1 = AY1 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*ORTH(1,1)     &
                +  ENER(5)*(ORTH(1,1)*CXX - ORTH(1,13)*TRAX)*TRBX &
                +  ENER(6)*(ORTH(1,1)*CXY - ORTH(1,19)*TRAX)*TRBY &
                -  CONST*FOUR*(CXX*ORTH(1,13) - CXY*ORTH(1,19))
      AY2 = AY2 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*ORTH(1,2)     &
                +  ENER(5)*(ORTH(1,2)*CXX - ORTH(1,14)*TRAX)*TRBX &
                +  ENER(6)*(ORTH(1,2)*CXY - ORTH(1,20)*TRAX)*TRBY &
                -  CONST*FOUR*(CXX*ORTH(1,14) - CXY*ORTH(1,20))
      AY3 = AY3 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*ORTH(1,3)     &
                +  ENER(5)*(ORTH(1,3)*CXX - ORTH(1,15)*TRAX)*TRBX &
                +  ENER(6)*(ORTH(1,3)*CXY - ORTH(1,21)*TRAX)*TRBY &
                -  CONST*FOUR*(CXX*ORTH(1,15) - CXY*ORTH(1,21))
      AX1 = AX1 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*ORTH(1,4)     &
                +  ENER(5)*(ORTH(1,4)*CXX - ORTH(1,16)*TRAX)*TRBX &
                +  ENER(6)*(ORTH(1,4)*CXY - ORTH(1,22)*TRAX)*TRBY &
                -  CONST*FOUR*(CXX*ORTH(1,16) - CXY*ORTH(1,22))
      AX2 = AX2 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*ORTH(1,5)     &
                +  ENER(5)*(ORTH(1,5)*CXX - ORTH(1,17)*TRAX)*TRBX &
                +  ENER(6)*(ORTH(1,5)*CXY - ORTH(1,23)*TRAX)*TRBY &
                -  CONST*FOUR*(CXX*ORTH(1,17) - CXY*ORTH(1,23))
      AX3 = AX3 + (ENER(1)*TRBX + ENER(2)*TRBY)*TWO*ORTH(1,6)     &
                +  ENER(5)*(ORTH(1,6)*CXX - ORTH(1,18)*TRAX)*TRBX &
                +  ENER(6)*(ORTH(1,6)*CXY - ORTH(1,24)*TRAX)*TRBY &
                -  CONST*FOUR*(CXX*ORTH(1,18) - CXY*ORTH(1,24))
      BY1 = BY1 + (ENER(1)*TRAX + ENER(3)*TRAY)*TWO*ORTH(2,1)     &
                +  ENER(5)*(ORTH(2,1)*CXX - ORTH(2,13)*TRBX)*TRAX &
                +  ENER(7)*(ORTH(2,1)*CYX - ORTH(2,19)*TRBX)*TRAY &
                -  CONST*FOUR*(CXX*ORTH(2,13) - CYX*ORTH(2,19))
      BY2 = BY2 + (ENER(1)*TRAX + ENER(3)*TRAY)*TWO*ORTH(2,2)     &
                +  ENER(5)*(ORTH(2,2)*CXX - ORTH(2,14)*TRBX)*TRAX &
                +  ENER(7)*(ORTH(2,2)*CYX - ORTH(2,20)*TRBX)*TRAY &
                -  CONST*FOUR*(CXX*ORTH(2,14) - CYX*ORTH(2,20))
      BY3 = BY3 + (ENER(1)*TRAX + ENER(3)*TRAY)*TWO*ORTH(2,3)     &
                +  ENER(5)*(ORTH(2,3)*CXX - ORTH(2,15)*TRBX)*TRAX &
                +  ENER(7)*(ORTH(2,3)*CYX - ORTH(2,21)*TRBX)*TRAY &
                -  CONST*FOUR*(CXX*ORTH(2,15) - CYX*ORTH(2,21))
      BX1 = BX1 + (ENER(1)*TRAX + ENER(3)*TRAY)*TWO*ORTH(2,4)     &
                +  ENER(5)*(ORTH(2,4)*CXX - ORTH(2,16)*TRBX)*TRAX &
                +  ENER(7)*(ORTH(2,4)*CYX - ORTH(2,22)*TRBX)*TRAY &
                -  CONST*FOUR*(CXX*ORTH(2,16) - CYX*ORTH(2,22))
      BX2 = BX2 + (ENER(1)*TRAX + ENER(3)*TRAY)*TWO*ORTH(2,5)     &
                +  ENER(5)*(ORTH(2,5)*CXX - ORTH(2,17)*TRBX)*TRAX &
                +  ENER(7)*(ORTH(2,5)*CYX - ORTH(2,23)*TRBX)*TRAY &
                -  CONST*FOUR*(CXX*ORTH(2,17) - CYX*ORTH(2,23))
      BX3 = BX3 + (ENER(1)*TRAX + ENER(3)*TRAY)*TWO*ORTH(2,6)     &
                +  ENER(5)*(ORTH(2,6)*CXX - ORTH(2,18)*TRBX)*TRAX &
                +  ENER(7)*(ORTH(2,6)*CYX - ORTH(2,24)*TRBX)*TRAY &
                -  CONST*FOUR*(CXX*ORTH(2,18) - CYX*ORTH(2,24))
!
      RETURN
      END SUBROUTINE R22CR22C
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R20R30(EN,XGRD,YGRD,ZGRD,               &
                TRAZ,TRBZ,TAZ1,TAZ2,TAZ3,TBZ1,TBZ2,TBZ3, &
                CZZ,KU20,KU30)
!     calculates 20-30 interaction
!
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(7),CONST
      real(chm_real) TRAZ,TRBZ,TAZ1,TAZ2,TAZ3,TBZ1,TBZ2,TBZ3
      real(chm_real) CZZ,KU20,KU30
!
      CONST = KU20*KU30*1.25D0*RM6
      ENER(1) = CONST*63D0*TRAZ*TRBZ**2
      ENER(2) =-CONST*SEVEN*TRBZ**2
      ENER(3) =-CONST*21D0*TRAZ
      ENER(4) = CONST*42D0*TRBZ*CZZ
      ENER(5) = CONST*THREE
      ENER(6) =-CONST*SIX*CZZ
      ENER(7) = CONST*SIX*CZZ**2
      EN = ( (ENER(1) + ENER(3) + ENER(4))*TRAZ + ENER(2) + &
                   ENER(5) + ENER(7))*TRBZ + ENER(6)*TRAZ + EN
      XGRD = ENER(1)*(TWO*TAZ1*TRBZ + THREE*TBZ1*TRAZ - ELEVEN*RAX*TRAZ*TRBZ) &
           + ENER(2)*(THREE*TBZ1 - RAX9*TRBZ)                       &
           + ENER(3)*(TBZ1*TRAZ + TWO*TAZ1*TRBZ - RAX9*TRAZ*TRBZ)   &
           + ENER(4)*(TWO*TBZ1*TRAZ + TAZ1*TRBZ - RAX9*TRAZ*TRBZ)   &
           + ENER(5)*(TBZ1 - RAX7*TRBZ)                             &
           + ENER(6)*(TAZ1 - RAX7*TRAZ)                             &
           + ENER(7)*(TBZ1 - RAX7*TRBZ) + XGRD
      YGRD = ENER(1)*(TWO*TAZ2*TRBZ + THREE*TBZ2*TRAZ - ELEVEN*RAY*TRAZ*TRBZ) &
           + ENER(2)*(THREE*TBZ2 - RAY9*TRBZ)                       &
           + ENER(3)*(TBZ2*TRAZ + TWO*TAZ2*TRBZ - RAY9*TRAZ*TRBZ)   &
           + ENER(4)*(TWO*TBZ2*TRAZ + TAZ2*TRBZ - RAY9*TRAZ*TRBZ)   &
           + ENER(5)*(TBZ2 - RAY7*TRBZ)                             &
           + ENER(6)*(TAZ2 - RAY7*TRAZ)                             &
           + ENER(7)*(TBZ2 - RAY7*TRBZ) + YGRD
      ZGRD = ENER(1)*(TWO*TAZ3*TRBZ + THREE*TBZ3*TRAZ - ELEVEN*RAZ*TRAZ*TRBZ) &
           + ENER(2)*(THREE*TBZ3 - RAZ9*TRBZ)                       &
           + ENER(3)*(TBZ3*TRAZ + TWO*TAZ3*TRBZ - RAZ9*TRAZ*TRBZ)   &
           + ENER(4)*(TWO*TBZ3*TRAZ + TAZ3*TRBZ - RAZ9*TRAZ*TRBZ)   &
           + ENER(5)*(TBZ3 - RAZ7*TRBZ)                             &
           + ENER(6)*(TAZ3 - RAZ7*TRAZ)                             &
           + ENER(7)*(TBZ3 - RAZ7*TRBZ) + ZGRD
!
      RETURN
      END SUBROUTINE R20R30
!--------------------------------------------------------------------- 
!---------------------------------------------------------------------
      SUBROUTINE R22CR30(EN,XGRD,YGRD,ZGRD,                &
             TRAX,TRAY,TRBZ,TAX1,TAX2,TAX3,TAY1,TAY2,TAY3, &
             TBZ1,TBZ2,TBZ3,CXZ,CYZ,KU22C,KU30)
!     calculates 22c-30 interaction
!C
      use chm_kinds
      use dimens_fcm
      use number
      implicit none
      real(chm_real) EN,XGRD,YGRD,ZGRD,ENER(10),CONST
      real(chm_real) TRAX,TRAY,TRBZ,TAX1,TAX2,TAX3,TAY1,TAY2,TAY3
      real(chm_real) TBZ1,TBZ2,TBZ3,CXZ,CYZ,KU22C,KU30
!
      CONST = KU22C*KU30*1.25D0*1.732050808D0*RM6
      ENER(1) = CONST*21D0*TRBZ**2*TRAX
      ENER(2) =-CONST*21D0*TRBZ**2*TRAY
      ENER(3) =-CONST*SEVEN*TRAX
      ENER(4) = CONST*SEVEN*TRAY
      ENER(5) = CONST*14D0*TRBZ*CXZ
      ENER(6) =-CONST*14D0*TRBZ*CYZ
      ENER(7) =-CONST*TWO*CXZ
      ENER(8) = CONST*TWO*CYZ
      ENER(9) = CONST*TWO*CXZ**2
      ENER(10) =-CONST*TWO*CYZ**2
      EN = ( (ENER(1) + ENER(3) + ENER(5))*TRAX + (ENER(2) + ENER(4) &
         + ENER(6))*TRAY + ENER(9) + ENER(10))*TRBZ + ENER(7)*TRAX + ENER(8)*TRAY + EN
      XGRD = ENER(1)*(THREE*TBZ1*TRAX + TWO*TAX1*TRBZ - ELEVEN*RAX*TRBZ*TRAX) &
           + ENER(2)*(THREE*TBZ1*TRAY + TWO*TAY1*TRBZ - ELEVEN*RAX*TRBZ*TRAY) &
           + ENER(3)*(TBZ1*TRAX + TWO*TAX1*TRBZ - RAX9*TRBZ*TRAX)  &
           + ENER(4)*(TBZ1*TRAY + TWO*TAY1*TRBZ - RAX9*TRBZ*TRAY)  &
           + ENER(5)*(TWO*TBZ1*TRAX + TAX1*TRBZ - RAX9*TRBZ*TRAX)  &
           + ENER(6)*(TWO*TBZ1*TRAY + TAY1*TRBZ - RAX9*TRBZ*TRAY)  &
           + ENER(7)*(TAX1 - RAX7*TRAX)                            &
           + ENER(8)*(TAY1 - RAX7*TRAY)                            &
           + ENER(9)*(TBZ1 - RAX7*TRBZ)                            &
           + ENER(10)*(TBZ1 - RAX7*TRBZ) + XGRD
      YGRD = ENER(1)*(THREE*TBZ2*TRAX + TWO*TAX2*TRBZ - ELEVEN*RAY*TRBZ*TRAX) &
           + ENER(2)*(THREE*TBZ2*TRAY + TWO*TAY2*TRBZ - ELEVEN*RAY*TRBZ*TRAY) &
           + ENER(3)*(TBZ2*TRAX + TWO*TAX2*TRBZ - RAY9*TRBZ*TRAX)  &
           + ENER(4)*(TBZ2*TRAY + TWO*TAY2*TRBZ - RAY9*TRBZ*TRAY)  &
           + ENER(5)*(TWO*TBZ2*TRAX + TAX2*TRBZ - RAY9*TRBZ*TRAX)  &
           + ENER(6)*(TWO*TBZ2*TRAY + TAY2*TRBZ - RAY9*TRBZ*TRAY)  &
           + ENER(7)*(TAX2 - RAY7*TRAX)                            &
           + ENER(8)*(TAY2 - RAY7*TRAY)                            &
           + ENER(9)*(TBZ2 - RAY7*TRBZ)                            &
           + ENER(10)*(TBZ2 - RAY7*TRBZ) + YGRD
      ZGRD = ENER(1)*(THREE*TBZ3*TRAX + TWO*TAX3*TRBZ - ELEVEN*RAZ*TRBZ*TRAX) &
           + ENER(2)*(THREE*TBZ3*TRAY + TWO*TAY3*TRBZ - ELEVEN*RAZ*TRBZ*TRAY) &
           + ENER(3)*(TBZ3*TRAX + TWO*TAX3*TRBZ - RAZ9*TRBZ*TRAX)  &
           + ENER(4)*(TBZ3*TRAY + TWO*TAY3*TRBZ - RAZ9*TRBZ*TRAY)  &
           + ENER(5)*(TWO*TBZ3*TRAX + TAX3*TRBZ - RAZ9*TRBZ*TRAX)  &
           + ENER(6)*(TWO*TBZ3*TRAY + TAY3*TRBZ - RAZ9*TRBZ*TRAY)  &
           + ENER(7)*(TAX3 - RAZ7*TRAX)                            &
           + ENER(8)*(TAY3 - RAZ7*TRAY)                            &
           + ENER(9)*(TBZ3 - RAZ7*TRBZ)                            &
           + ENER(10)*(TBZ3 - RAZ7*TRBZ) + ZGRD
!
      RETURN
      END SUBROUTINE R22CR30
!--------------------------------------------------------------------- 

!---------------------------------------------------------------------
!
      SUBROUTINE IMTP(NATM,IS,IQ)
!    This routine is called by the image update routine.
!     --> upimag.src
!     The MTP parameter arrays are extended for images
!
      use chm_kinds
      use dimens_fcm
      use psf
      implicit none
      INTEGER NATM, IS, IQ, K, I, L, SCNT
!
      IF (IMTPIMG .EQ. 0) THEN
         CALL IMTPREALLOC(1)   ! 1: initial call
         IMTPIMG = 1
      ENDIF

      SCNT = 0
      K = NATM
      DO I=IS, IQ
         K = K + 1
         IF (K .GT. NATOM ) THEN
            CG(K) = CG(I)
            RANK(K) = RANK(I)
            RANK(K) = RANK(I)
            IF (REFM(I) .NE. 0 ) THEN
               SCNT = SCNT + 1

               IF (SCNT .EQ. 1) THEN
                  IMOL = IMOL + 1
                  IF (IMOL .GT. MOLN) THEN   !
                     CALL IMTPREALLOC(2)   ! 2: 2nd or later calls
                  ENDIF
                  DO L=1, 5
                     IF (L .LE.SITE(REFM(I))) THEN
                        RFNR(IMOL,L) = K + L - 1   !!
                     ELSE
                        RFNR(IMOL,L) = 0
                     ENDIF
                  ENDDO
               ENDIF

               IF (SCNT .EQ. SITE(REFM(I))) THEN
                  SCNT = 0
               ENDIF

               REFM(K) = IMOL
               SITE(REFM(K)) = SITE(REFM(I))

               IF ( QANH(REFM(I))) then
                  QANH(IMOL) = .TRUE.
               ELSE
                  QANH(IMOL) = .FALSE.
               ENDIF

               IF ( QFLC(REFM(I))) then
                  QFLC(IMOL) = .TRUE.
               ELSE
                  QFLC(IMOL) = .FALSE.
               ENDIF

               IF ( LINT(REFM(I))) then !!
                  LINT(IMOL) = .TRUE.   !!
               ELSE                     !!
                  LINT(IMOL) = .FALSE.  !!
               ENDIF                    !!

!              linear molecules
               IF ( RFNR(REFM(K),3) .eq. 0 ) THEN   !!
                  Q1(K) = Q1(I)
                  Q2(K) = Q2(I)
                  IF ( RANK(I) .GT. 0 ) THEN
                     Q1Z1(K) = Q1Z1(I)
                     Q1Z2(K) = Q1Z2(I)
                     IF ( RANK(I) .GT. 1 ) THEN
                        Q201(K) = Q201(I)
                        Q202(K) = Q202(I)
                        IF ( RANK(I) .GT. 2 ) THEN
                           Q301(K) = Q301(I)
                           Q302(K) = Q302(I)
                        ENDIF
                     ENDIF
                  ENDIF
               ENDIF

!              other geometries
               Q1Z(K) = Q1Z(I)
               Q1X(K) = Q1X(I)
               Q1Y(K) = Q1Y(I)
               Q20(K) = Q20(I)
               Q21C(K) = Q21C(I)
               Q21S(K) = Q21S(I)
               Q22C(K) = Q22C(I)
               Q22S(K) = Q22S(I)
               Q30(K) = 0   !!
            ELSE
               REFM(K) = 0
            ENDIF
         ENDIF
      ENDDO
!     write(6,*) 'IMTP> MTP image fragments', IMOL,K
!
      RETURN
      END SUBROUTINE IMTP
!---------------------------------------------------------------------
      SUBROUTINE IMTPREALLOC(ISTAT)
!C
      use chm_kinds
      use dimens_fcm
      use image
      use number
      use consta
      use stream
      use memory
      implicit none

      INTEGER ISTAT, NEWMOLN, N1,N2
      real(chm_real) VOL00,VOL01,VOL06,VOL08,VOL12,VRATIO   ! temp variables

      INTEGER, ALLOCATABLE :: ORFNR(:,:), OIZDA(:,:)          ! temp arrays
      real(chm_real), allocatable :: ODEL0(:,:), ORZDA(:,:), ORCDA(:,:)   ! temp arrays


      IF (ISTAT .EQ. 1) THEN   ! initial call
         VOL08 = FOUR/THREE*PI*CUTIM**3
         VOL12 = PI*CUTIM**2*(XUCELL(1)+XUCELL(2)+XUCELL(3))
         VOL06 = TWO*CUTIM*(XUCELL(1)*XUCELL(2)+XUCELL(2)*XUCELL(3) &
                           +XUCELL(3)*XUCELL(1))
         VOL01 = XUCELL(1)*XUCELL(2)*XUCELL(3)
         VRATIO = (VOL08+VOL12+VOL06+VOL01) / VOL01
         NEWMOLN = MOLS * INT(VRATIO) + 1
      ELSE
         NEWMOLN = MOLN * 110 / 100 + 1   ! increase by 10 percent
      ENDIF

      IF(PRNLEV.GT.2) WRITE(OUTU,*) 'MTP> MEMORY REALLOCATION FOR IMAGES'

      call chmrealloc('mtp.src','MTPINIT','QANH',NEWMOLN,log=QANH)
      call chmrealloc('mtp.src','MTPINIT','QFLC',NEWMOLN,log=QFLC)
      call chmrealloc('mtp.src','MTPINIT','QDUM',NEWMOLN,log=QDUM)
      call chmrealloc('mtp.src','MTPINIT','LINT',NEWMOLN,log=LINT)
      call chmrealloc('mtp.src','MTPINIT','SITE',NEWMOLN,intg=SITE)

      call chmrealloc('mtp.src','MTPINIT','KDA', NEWMOLN,intg=KDA)
      call chmrealloc('mtp.src','MTPINIT','CBBEQDI',NEWMOLN,crl=CBBEQDI)
      call chmrealloc('mtp.src','MTPINIT','IPDA',2*NEWMOLN,intg=IPDA)


      ! chmrealloc allows only 1-D arrays in c36a6.
      ! The code below may be simplified using chmrealloc in the future
      ! when chmrealloc allows 2-D arrays.

      call chmalloc('mtp.src','MTPINIT','ORFNR',MOLN,5,intg=ORFNR)
      call chmalloc('mtp.src','MTPINIT','OIZDA',3,2*MOLN,intg=OIZDA)
      call chmalloc('mtp.src','MTPINIT','ORZDA',3,2*MOLN,crl=ORZDA)
      call chmalloc('mtp.src','MTPINIT','ORCDA',3,2*MOLN,crl=ORCDA)
      call chmalloc('mtp.src','MTPINIT','ODEL0',MOLN,3,crl=ODEL0)

      DO N1 = 1, MOLN   ! store some values
         DO N2 = 1, 3
            ORFNR(N1,N2) = RFNR(N1,N2)
            ORFNR(N1,N2+2) = RFNR(N1,N2+2)
            OIZDA(N2,N1) = IZDA(N2,N1)
            OIZDA(N2,N1+MOLN) = IZDA(N2,N1+MOLN)
            ORZDA(N2,N1) = RZDA(N2,N1)
            ORZDA(N2,N1+MOLN) = RZDA(N2,N1+MOLN)
            ORCDA(N2,N1) = RCDA(N2,N1)
            ORCDA(N2,N1+MOLN) = RCDA(N2,N1+MOLN)
            ODEL0(N1,N2) = DEL0(N1,N2)
         ENDDO
      ENDDO

      call chmdealloc('mtp.src','MTPINIT','RFNR',MOLN,5,intg=RFNR)
      call chmdealloc('mtp.src','MTPINIT','IZDA',3,2*MOLN,intg=IZDA)
      call chmdealloc('mtp.src','MTPINIT','RZDA',3,2*MOLN,crl=RZDA)
      call chmdealloc('mtp.src','MTPINIT','RCDA',3,2*MOLN,crl=RCDA)
      call chmdealloc('mtp.src','MTPINIT','DEL0',MOLN,3,crl=DEL0)

      call chmalloc('mtp.src','MTPINIT','RFNR',NEWMOLN,5,intg=RFNR)
      call chmalloc('mtp.src','MTPINIT','IZDA',3,2*NEWMOLN,intg=IZDA)
      call chmalloc('mtp.src','MTPINIT','RZDA',3,2*NEWMOLN,crl=RZDA)
      call chmalloc('mtp.src','MTPINIT','RCDA',3,2*NEWMOLN,crl=RCDA)
      call chmalloc('mtp.src','MTPINIT','DEL0',NEWMOLN,3,crl=DEL0)

      DO N1 = 1, MOLN   ! restore values
         DO N2 = 1, 3
            RFNR(N1,N2) = ORFNR(N1,N2)
            RFNR(N1,N2+2) = ORFNR(N1,N2+2)
            IZDA(N2,N1) = OIZDA(N2,N1)
            IZDA(N2,N1+MOLN) = OIZDA(N2,N1+MOLN)
            RZDA(N2,N1) = ORZDA(N2,N1)
            RZDA(N2,N1+MOLN) = ORZDA(N2,N1+MOLN)
            RCDA(N2,N1) = ORCDA(N2,N1)
            RCDA(N2,N1+MOLN) = ORCDA(N2,N1+MOLN)
            DEL0(N1,N2) = ODEL0(N1,N2)
         ENDDO
      ENDDO

      call chmdealloc('mtp.src','MTPINIT','ORFNR',MOLN,5,intg=ORFNR)
      call chmdealloc('mtp.src','MTPINIT','OIZDA',3,2*MOLN,intg=OIZDA)
      call chmdealloc('mtp.src','MTPINIT','ORZDA',3,2*MOLN,crl=ORZDA)
      call chmdealloc('mtp.src','MTPINIT','ORCDA',3,2*MOLN,crl=ORCDA)
      call chmdealloc('mtp.src','MTPINIT','ODEL0',MOLN,3,crl=ODEL0)


      ! In the case of arrays below, there is no need to keep old values.
      ! So we deallocate first, and then allocate again with larger sizes.

      call chmdealloc('mtp.src','MTPINIT','TRFX',MOLN,3,crl=TRFX)
      call chmdealloc('mtp.src','MTPINIT','TRFY',MOLN,3,crl=TRFY)
      call chmdealloc('mtp.src','MTPINIT','TRFZ',MOLN,3,crl=TRFZ)
      call chmdealloc('mtp.src','MTPINIT','ADX',MOLN,3,crl=ADX)
      call chmdealloc('mtp.src','MTPINIT','ADY',MOLN,3,crl=ADY)
      call chmdealloc('mtp.src','MTPINIT','ADZ',MOLN,3,crl=ADZ)
      call chmdealloc('mtp.src','MTPINIT','ADXS',MOLN,3,crl=ADXS)
      call chmdealloc('mtp.src','MTPINIT','ADYS',MOLN,3,crl=ADYS)
      call chmdealloc('mtp.src','MTPINIT','ADZS',MOLN,3,crl=ADZS)

      call chmalloc('mtp.src','MTPINIT','TRFX',NEWMOLN,3,crl=TRFX)
      call chmalloc('mtp.src','MTPINIT','TRFY',NEWMOLN,3,crl=TRFY)
      call chmalloc('mtp.src','MTPINIT','TRFZ',NEWMOLN,3,crl=TRFZ)
      call chmalloc('mtp.src','MTPINIT','ADX',NEWMOLN,3,crl=ADX)
      call chmalloc('mtp.src','MTPINIT','ADY',NEWMOLN,3,crl=ADY)
      call chmalloc('mtp.src','MTPINIT','ADZ',NEWMOLN,3,crl=ADZ)
      call chmalloc('mtp.src','MTPINIT','ADXS',NEWMOLN,3,crl=ADXS)
      call chmalloc('mtp.src','MTPINIT','ADYS',NEWMOLN,3,crl=ADYS)
      call chmalloc('mtp.src','MTPINIT','ADZS',NEWMOLN,3,crl=ADZS)

      MOLN = NEWMOLN
!
      RETURN
      END SUBROUTINE IMTPREALLOC
!--------------------------------------------------------------------- 

!=====================================================================
! End of module MTP
!=====================================================================

end module mtp_fcm


