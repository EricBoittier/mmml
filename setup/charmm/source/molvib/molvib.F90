#if KEY_MOLVIB==1
SUBROUTINE MOLVIB(NQ,NIC,NIC0,IZMAX,MAXSYM,NGMAX,NBLMAX, &
     NAT,NAT3,NQM,ISTRM,OUTU,X,AMASS,FX, &
     B,G,FS,LS,LX,U1,U2,W1,W2,DD,EXPFRQ, &
     CALFRQ,V1,V2,V3,V4,V5,V6,V7,PRINCI,TOMM,ICTYP, &
     IATI,IATJ,IATK,IATL,INDX,MNAT,LGRUP,IGRUP,KSYMB,IPTA, &
     IPTB,IBLOCK,QSEL,ATYP,SBLOCK,SYMB,SPED)
  !----------------------------------------------------------------
  !     The following is a version of the program MOLVIB          |
  !         which also exists separately from CHARMM              |
  !                                                               |
  !     Joanna and Krzysztof Kuczera, Cambridege, MA 22-Dec-1988  |
  !                                          updated 23-Apr-1991  |
  !----------------------------------------------------------------
  !  General program for molecular vibrations.
  !  Performs the following types of calculations:
  !   a. 'G   ' - diagonalization of G to determine redundancies
  !               in internal coordinates
  !   b. 'GF  ' - diagonalize GF , this is preceded by a
  !               transformation to independent coordinates by a U matrix
  !   c. 'GAUS' - read in cartesian vibrational eigenvectors
  !               and calculate PED; interfaces with CHARMM and GAUSSIAN
  !   d. 'GFX'  - solution of vibrational problem in cartesian coordinates
  !               interfaces with CHARMM and GAUSSIAN
  !   e. 'KANO' - generation of canonical force field
  !   f. 'CRYS' - crystal vibrations (k=0 and whole unit cell only)
  !
  !   Input description: see SUBROUTINE MOLINP and file molinp.dsc
  !
  !  NAT - no. of atoms
  !  NQ  - dimension of the G matrix for diagonalization
  !  NIC0 - initial no. of internal coords 
  !  NIC - no. of internal coordinates after presymmetrization
  !        of G by matrix U1
  !
  !  ISTRM is the number of the FORTRAN unit for input  (from stream.f90)
  !  OUTU  is the number of the FORTRAN unit for output (from stream.f90)
  !----------------------------------------------------------------------
  !
  use chm_kinds
  implicit none
  !
  ! ...Dimensions
  INTEGER NQM,NAT3,NATM,NAT3M,IZMAX,NGMAX,MAXSYM,NBLMAX
  ! ...Basic vibrational arrays
  real(chm_real) B(NQM,NQM),G(NQM,NQM),FS(NQM,NQM),LS(NQM,NQM)
  real(chm_real) LX(NQM,NQM),U1(NQM,NQM),U2(NQM,NQM),DD(NQM)
  real(chm_real) W1(NQM,NQM),W2(NQM,NQM),FX(NQM,NQM)
  real(chm_real) AMASS(NAT3),X(NAT3),EXPFRQ(NQM),CALFRQ(NQM)
  real(chm_real) V1(NQM),V2(NQM),V3(NQM),V4(NQM),V5(NQM)
  real(chm_real) V6(NQM),V7(NQM),PRINCI(3*IZMAX),TOMM(IZMAX)
  INTEGER ICTYP(NQM),IATI(NQM),IATJ(NQM),IATK(NQM)
  INTEGER IATL(NQM),INDX(NQM),MNAT(IZMAX)
  CHARACTER(len=4) ATYP(NAT3)
  !
  ! ... These are arrays used in PED analysis
  ! ... MAXSYM - the max no. of coordinate group members
  ! ...           and max no. of IC symbols; should be
  ! ...           not smaller than NQM 
  ! ... NGMAX - max no. of coordinate groups
  ! ... NBLMAX - should always be equal to NQM
  !
  INTEGER LGRUP(NGMAX),IGRUP(NGMAX,MAXSYM),KSYMB(MAXSYM)
  INTEGER NGRUP,NSYMB
  CHARACTER(len=8) SYMB(MAXSYM)
  CHARACTER(len=8) SPED(NBLMAX)
  CHARACTER(len=4) SBLOCK(NBLMAX)
  INTEGER IPTA(NQM),IBLOCK(NBLMAX),NBLOCK
  INTEGER IPTB(NBLMAX)
  LOGICAL QSEL(NBLMAX)
  real(chm_real) CUTPED
  !
  ! ...Local and passed variables
  CHARACTER(len=4) JCONT
  real(chm_real) S,STPSIZ
  INTEGER NUMAT,NQ,NIC,NIC0,IPRNT,NULL,NSTRT,NAT,NQ2
  INTEGER IGLEV,ICANO,IFPED,IFTRRM,IZMOL,IFCRYS,IFSTEP
  INTEGER ISTCOR,IFFMAT,IFLMAT,IFEXPF,IFSYMM,IFTRAN
  INTEGER I,J,K,II,NT,ISTER,JN,IZMOL3,MUMAT,NIC02,IPRNTK
  INTEGER ISTRM,OUTU

  ! ...Initialize 09-19-2009 YW
  ICANO = 0
  !
  ! ...In CHARMM versiononly: actual = max dimension
  NATM=NAT
  NAT3M=NAT3
  !
  ! ... Perform all input operations and initialization first
  !
  CALL MOLINP(JCONT,NUMAT,NQ,NIC,NIC0,IPRNT,NULL,NSTRT, &
       NAT,NAT3,NQ2,IGLEV,ICANO,IFPED,IFTRRM, &
       IZMOL,IFCRYS,IFSTEP,ISTCOR,STPSIZ, &
       IFFMAT,IFLMAT,IFEXPF,IFSYMM,IFTRAN, &
       NQM,NATM,NAT3M,IZMAX,U1,U2,FS,FX, &
       LX,W2,V1,V2,V3,ATYP,CUTPED,CALFRQ,W1, &
       ISTRM,OUTU,NGMAX,MAXSYM,NBLMAX, &
       LGRUP,IGRUP,KSYMB,NGRUP,NSYMB, &
       SYMB,SPED,SBLOCK,IPTA,IBLOCK,NBLOCK,IPTB,QSEL, &
       MNAT,X,AMASS,EXPFRQ,ICTYP,IATI,IATJ,IATK,IATL)
  !
  ! ... Calculate B and G matrices and symmetrize
  !
  IF(JCONT.NE.'CRYS' .AND. JCONT.NE.'STEP' .AND. JCONT.NE.'KANO') &
       THEN
     CALL GMAT(NIC0,NIC,NQ,NAT,NAT3,IPRNT,NUMAT,JCONT, &
          NQM,ICTYP,IATI,IATJ,IATK,IATL, &
          U1,U2,W1,AMASS,B,G,X,OUTU)
  END IF
  !
1001 FORMAT(1X,I3,8F9.6,12(/4X,8F9.6))
  !
  ! ... Branch point for the different options
  !
  !
  !
  ! ... Redundancy analysis, in this section NQ=NIC
  !
  IF(JCONT.EQ.'G   ') THEN
     !
     ! ... Diagonalize G to determine null space
     ISTER=0
     CALL GFDIAG(NAT,NAT3,NQ,NQ2,NQ2,IPRNT,ISTER,JCONT, &
          NQM,G,W1,W2,LS,FS,DD,V1,CALFRQ,OUTU)
     !
     IF(IGLEV.EQ.1) THEN
        WRITE(OUTU,*) ' G option finished : level 1'
        RETURN
     END IF
     !
     ! ... Check dimension of null space; if not all null basis vectors
     ! ... supplied in U2, then generate the missing ones in ORTNUL
     !
     II=0
     DO I=1,NQ
        IF(DD(I).LT.0.0001) II=II+1
     ENDDO
     IF(II.LT.NULL) THEN
        WRITE(OUTU,*) ' Warning : II.LT.NULL; proceeding'
     END IF
     !
     IF(II.GT.NULL) CALL ORTNUL(U2,W1,W2,V1,NULL,II,NIC,NQM,OUTU)
     !      
     ! ... Use Gram-Schmidt procedure to find independent coordinates
     ! ...    orthogonal to null space basis
     !
     IF(NSTRT.GT.0)  CALL ORTHOP(U2,V1,II,NSTRT,NQ,NQM,OUTU)
     !
     ! ... Transform back to primitive coordinates
     !
     IF(NUMAT.LE.1) THEN
        WRITE(OUTU,*) ' G option finished '
        RETURN
     END IF
     !
     NT=II+NSTRT
     DO I=1,NT
        DO J=1,NIC0
           S=0.0
           DO K=1,NIC
              S = S + U2(I,K)*U1(K,J)
           END DO
           W2(I,J)=S
        END DO
     END DO
     !
     WRITE(OUTU,*) 
     WRITE(OUTU,*) ' The back-transformed orthonormal coordinates '
     WRITE(OUTU,*) '                 null space: '
     WRITE(OUTU,*) 
     DO I=1,II
        WRITE(OUTU,1001) I,(W2(I,J),J=1,NIC0)
     END DO
     !
     IF(NULL.EQ.II) THEN
        WRITE(OUTU,*) 
        WRITE(OUTU,*) '    independent coordinates: '
        WRITE(OUTU,*) 
        DO I=II+1,NT
           WRITE(OUTU,1001) I,(W2(I,J),J=1,NIC0)
        END DO
     END IF
     !
     WRITE(OUTU,*) ' G option finished '
     RETURN
  END IF
  !
  !
  !
  ! ... GF problem in internal coordinates
  !
  IF(JCONT.EQ.'GF  ') THEN

     IF(NUMAT.GE.1) THEN 
        CALL FRGET(NIC0,NIC,NQ,IPRNT,NUMAT,ICANO, &
             NQM,U1,U2,W1,W2,FS,FX,OUTU)
     END IF
     ! ... Diagonalize GF
     ISTER=1
     CALL GFDIAG(NAT,NAT3,NQ,NQ2,NQ2,IPRNT,ISTER,JCONT, &
          NQM,G,W1,W2,LS,FS,DD,V1,CALFRQ,OUTU)

     ! ... Find potential energy distribution for normal modes
     ISTER=1
     CALL PED0(NAT,NAT3,NQ,NIC,NIC0,IPRNT,ISTER,NUMAT,CUTPED, &
          NQM,B,LX,LS,W2,CALFRQ,V1,V2,IPTA,INDX,NBLMAX,OUTU)
     IF(IFPED.GT.0) THEN
        CALL PED3(W2,CALFRQ,V3,V4,NQ,NQM,IFSYMM,IFEXPF, &
             EXPFRQ,V5, &
             MAXSYM,NGMAX,NBLMAX,LGRUP,IGRUP,KSYMB, &
             NGRUP,NSYMB,SYMB,SPED,SBLOCK,IPTA,IPTB, &
             IBLOCK,NBLOCK,QSEL,CUTPED,OUTU)
     END IF
     !
     WRITE(OUTU,*) 'GF option finished'
     RETURN
  END IF
  !
  !
  !
  ! ... Read in GAUSSIAN LX, transform to LS and find PED
  !
  IF(JCONT.EQ.'GAUS') THEN
     !
     ! ... Remove translation-rotation contribution
     ! 
     IF(IFTRRM.GT.0) THEN
        CALL RMVTR0(X,LX,W2,NQM,NQ,NAT,NAT3,AMASS,CALFRQ,IPRNT)
     END IF
     !
     ! ... Transform eigenvectors to independent internal coordinates
     ! ...  S  and generate PED
     ISTER=0
     CALL PED0(NAT,NAT3,NQ,NIC,NIC0,IPRNT,ISTER,NUMAT,CUTPED, &
          NQM,B,LX,LS,W2,CALFRQ,V1,V2,IPTA,INDX,NBLMAX,OUTU)
     !
     IF(IFPED.GT.0) THEN
        CALL PED3(W2,CALFRQ,V3,V4,NQ,NQM,IFSYMM,IFEXPF, &
             EXPFRQ,V5, &
             MAXSYM,NGMAX,NBLMAX,LGRUP,IGRUP,KSYMB, &
             NGRUP,NSYMB,SYMB,SPED,SBLOCK,IPTA,IPTB, &
             IBLOCK,NBLOCK,QSEL,CUTPED,OUTU)
     END IF
     !
     WRITE(OUTU,*) 'GAUSS option finished'
     RETURN
  END IF
  !
  !
  !
  ! ... Vibrational problem in cartesian coordinates
  !
  IF(JCONT.EQ.'GFX ') THEN
     !
     CALL GFX(NAT,NAT3,NQ,NQ2,NQ2,IPRNT,ISTER,JCONT, &
          NQM,FX,LX,W1,LS,DD,V1,AMASS,CALFRQ,OUTU)
     !
     !     Get rid of translational and rotational eigenvectors
     !     and eigenvalues
     ! 
     DO I=1,NAT3
        JN=0
        DO J=7,NAT3
           JN=JN+1
           LX(I,JN)=LX(I,J)
        END DO
     END DO
     JN=0
     DO J=7,NAT3
        JN=JN+1
        DD(JN)=DD(J)
        CALFRQ(JN)=CALFRQ(J)
     END DO
     !
     !     Transform LX to LS and calculate PED
     !
     ISTER=0
     CALL PED0(NAT,NAT3,NQ,NIC,NIC0,IPRNT,ISTER,NUMAT,CUTPED, &
          NQM,B,LX,LS,W2,CALFRQ,V1,V2,IPTA,INDX,NBLMAX,OUTU)
     IF(IFPED.GT.0) THEN
        CALL PED3(W2,CALFRQ,V3,V4,NQ,NQM,IFSYMM,IFEXPF, &
             EXPFRQ,V5, &
             MAXSYM,NGMAX,NBLMAX,LGRUP,IGRUP,KSYMB, &
             NGRUP,NSYMB,SYMB,SPED,SBLOCK,IPTA,IPTB, &
             IBLOCK,NBLOCK,QSEL,CUTPED,OUTU)
     END IF
     !
     ! ...  Repeat calculations in S coordinates
     ! ...  This is done for test purposes only for IPRNT.GE.6
     !
     IF(IPRNT.GE.6) THEN
        ! ...  Generate the FS matrix (F matrix in S coordinates)
        CALL FSGL(NAT,NAT3,NQ,IPRNT,NQM,LX,FS,DD,OUTU)
        IF(NUMAT.GE.1) THEN 
           CALL FRGET(NIC0,NIC,NQ,IPRNT,NUMAT,ICANO, &
                NQM,U1,U2,W1,W2,FS,FX,OUTU)
        END IF
        !
        ! ... Diagonalize GF
        ISTER=1
        CALL GFDIAG(NAT,NAT3,NQ,NQ2,NQ2,IPRNT,ISTER,JCONT, &
             NQM,G,W1,W2,LS,FS,DD,V1,CALFRQ,OUTU)
        !
        ! ... Find potential energy distribution for normal modes
        ISTER=1
        CALL PED0(NAT,NAT3,NQ,NIC,NIC0,IPRNT,ISTER,NUMAT,CUTPED, &
             NQM,B,LX,LS,W2,CALFRQ,V1,V2,IPTA,INDX,NBLMAX,OUTU)
        IF(IFPED.GT.0) THEN
           CALL PED3(W2,CALFRQ,V3,V4,NQ,NQM,IFSYMM,IFEXPF, &
                EXPFRQ,V5, &
                MAXSYM,NGMAX,NBLMAX,LGRUP,IGRUP,KSYMB, &
                NGRUP,NSYMB,SYMB,SPED,SBLOCK,IPTA,IPTB, &
                IBLOCK,NBLOCK,QSEL,CUTPED,OUTU)
        END IF
        !
     END IF ! IPRNT.GE.6
     !
     WRITE(OUTU,*) 'GFX option finished'
     RETURN
  END IF
  !
  !
  !
  ! ... Process 'KANO' keyword: diagonalize G and find FR*
  !
  IF(JCONT.EQ.'KANO') THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' Canonical force field evaluation begins'
     WRITE(OUTU,*)
     !
     ! ... Calculate G matrix in primitive IC's (R-coords)
     ! ... set NUMAT=0 to avoid G transformations at this stage
     !
     MUMAT=0
     CALL GMAT(NIC0,NIC,NQ,NAT,NAT3,IPRNT,MUMAT,JCONT, &
          NQM,ICTYP,IATI,IATJ,IATK,IATL, &
          U1,U2,W1,AMASS,B,G,X,OUTU)
     !
     ! ... If necessary, transform F
     ! ... NUMAT.GE.1 means that at least one U matrix has been supplied
     ! ... IFTRAN.GE.1 means that the input F matrix is not in primitive IC's
     !
     IF(NUMAT.GE.1 .AND. IFTRAN.GE.1) THEN 
        IF(NUMAT.NE.IFTRAN) THEN
           WRITE(OUTU,*) ' ****Warning: NUMAT.NE.IFTRAN; proceeding'
           WRITE(OUTU,*)
        END IF
        CALL FRGET(NIC0,NIC,NQ,IPRNT,NUMAT,ICANO, &
             NQM,U1,U2,W1,W2,FS,FX,OUTU)
        DO I=1,NIC0
           DO J=1,NIC0
              FS(I,J)=FX(I,J)
           END DO
        END DO
     END IF
     !
     ! ... Diagonalize G - eigenvectors returned in columns of W1
     ! ... NB. the G dimensions are NIC0xNIC0
     !
     ISTER=0
     NIC02=NIC0*NIC0
     CALL GFDIAK(NAT,NAT3,NIC0,NIC02,NIC02,IPRNT,ISTER,JCONT, &
          NQM,G,FS,LS,W1,W2,V1,DD,CALFRQ,OUTU)
     !
     ! ... Calculate the projection operator WT*V and transform FR -> FR*
     ! ... On exit, KANONR places the canonic ff FR* in the FS array
     ! ...     G eigenvectors in primitive IC's in  LX
     ! ...     projection operator WT*W in FX
     !
     IPRNTK=4
     CALL KANONR(NIC0,NQ,IPRNTK,NUMAT,NQM,W1,FX,FS,LX,OUTU)
     !
     ! ... Diagonalize GF in the R coords to check transformation
     !
     ISTER=1
     CALL GFDIAK(NAT,NAT3,NIC0,NIC02,NIC02,IPRNT,ISTER,JCONT, &
          NQM,G,FS,LS,W1,W2,V1,DD,CALFRQ,OUTU)
     !
     ! ... Final verification
     ! ... Transform G to independent coords
     !
     IF(NUMAT.GT.0) CALL GTRAN(G,U1,W1,NIC0,NIC,NQM,IPRNT,OUTU)
     IF(NUMAT.GT.1) CALL GTRAN(G,U2,W1,NIC,NQ,NQM,IPRNT,OUTU)
     IF(NUMAT.GT.0) THEN
        CALL FTRAK(NIC0,NIC,NQ,IPRNT,NUMAT,NQM,W1,U1,U2,FS,LX, &
             V1,W2,INDX,OUTU)
     END IF
     !
     ! ... Diagonalize GF in independent coordinates, with PED
     ISTER=1
     CALL GFDIAG(NAT,NAT3,NQ,NQ2,NQ2,IPRNT,ISTER,JCONT, &
          NQM,G,W1,W2,LS,FS,DD,V1,CALFRQ,OUTU)

     ! ... Find potential energy distribution for normal modes
     ISTER=1
     CALL PED0(NAT,NAT3,NQ,NIC,NIC0,IPRNT,ISTER,NUMAT,CUTPED, &
          NQM,B,LX,LS,W2,CALFRQ,V1,V2,IPTA,INDX,NBLMAX,OUTU)
     IF(IFPED.GT.0) THEN
        CALL PED3(W2,CALFRQ,V3,V4,NQ,NQM,IFSYMM,IFEXPF, &
             EXPFRQ,V5, &
             MAXSYM,NGMAX,NBLMAX,LGRUP,IGRUP,KSYMB, &
             NGRUP,NSYMB,SYMB,SPED,SBLOCK,IPTA,IPTB, &
             IBLOCK,NBLOCK,QSEL,CUTPED,OUTU)
     END IF
     !
     WRITE(OUTU,*) 'KANO option finished'
     RETURN
  END IF
  !
  !
  !
  ! ... Crystal vibrations
  !
  IF(JCONT.EQ.'CRYS') THEN

     ! ...  First step : vibrational analysis in cartesian coords
     ! ...  Diagonalize FX
     CALL GFX(NAT,NAT3,NQ,NQ2,NQ2,IPRNT,ISTER,JCONT, &
          NQM,FX,LX,W1,LS,DD,V1,AMASS,CALFRQ,OUTU)
     !
     ! ... Orient the individual molecules in the unit cell
     ! ... From here on, X are the new oriented set;
     ! ... The tranformation coefficients are given in W2: Xnew=W2*Xold
     IZMOL3=3*IZMOL
     CALL MOLORT(W2,V1,X,AMASS,NAT,NAT3,NQM,IZMOL,IZMOL3,IPRNT, &
          PRINCI,TOMM,MNAT,OUTU)
     !
     ! ... Calculate the G matrix in Molecular Coordinates
     ! ... G is symmetrized by U1 and U2, if they are supplied
     CALL GMATC(NIC0,NIC,NQ,NAT,NAT3,IPRNT,NUMAT,JCONT, &
          IZMOL,IZMOL3,ICTYP,IATI,IATJ,IATK,IATL, &
          NQM,B,G,U1,U2,W1,X,AMASS,PRINCI,TOMM,OUTU)
     !
     ! ... Transform FX to final molecular coordinates
     CALL CRYSTF(NQ,IPRNT,NQM,FX,FS,B,LS,W1,W2,V1,INDX,OUTU)
     !
     ! ... Diagonalize GF in molecular coordinates
     ISTER=1
     CALL GFDIAG(NAT,NAT3,NQ,NQ2,NQ2,IPRNT,ISTER,JCONT, &
          NQM,G,W1,W2,LS,FS,DD,V1,CALFRQ,OUTU)
     !
     ! ... Find potential energy distribution for normal modes
     ISTER=1
     CALL PED0(NAT,NAT3,NQ,NIC,NIC0,IPRNT,ISTER,NUMAT,CUTPED, &
          NQM,B,LX,LS,W2,CALFRQ,V1,V2,IPTA,INDX,NBLMAX,OUTU)
     IF(IFPED.GT.0) THEN
        CALL PED3(W2,CALFRQ,V3,V4,NQ,NQM,IFSYMM,IFEXPF, &
             EXPFRQ,V5, &
             MAXSYM,NGMAX,NBLMAX,LGRUP,IGRUP,KSYMB, &
             NGRUP,NSYMB,SYMB,SPED,SBLOCK,IPTA,IPTB, &
             IBLOCK,NBLOCK,QSEL,CUTPED,OUTU)
     END IF
     !
     WRITE(OUTU,*) ' CRYS option finished'
     RETURN
  END IF
  !
  !
  !
  !
  ! ... STEP option: deform cartesian coordinates X by
  ! ...   IFSTEP=1 - stepping along cartesian eigenvector #ISTCOR
  ! ...   IFSTEP=2 - stepping along internal eigenvector #ISTCOR
  ! ...   IFSTEP=3 - stepping along internal coordinate #ISTCOR
  !
  IF(JCONT.EQ.'STEP') THEN
     !
     ! ... 
     ! 
     CALL STEPUP(NUMAT,NQ,NIC,NIC0,IPRNT,NAT,NAT3, &
          IFSTEP,ISTCOR,STPSIZ,IFFMAT,IFLMAT, &
          NQM,LX,X,V7,OUTU)
     !
     WRITE(OUTU,*) ' STEP option finished'
     RETURN
  END IF
  !
  !
  !
  !
  ! ... Process unrecognized option
  !
  WRITE(OUTU,*) ' **** Error : unrecognized option ',JCONT
  WRITE(OUTU,*) ' Check option card '
  !
  RETURN
END SUBROUTINE MOLVIB
#else /**/
SUBROUTINE NULL_MV
  RETURN
END SUBROUTINE NULL_MV
#endif /*  MOLVIB*/

