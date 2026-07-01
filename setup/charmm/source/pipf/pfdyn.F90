#if KEY_PIPF==1 /* pipf_main */
SUBROUTINE PFPREP(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     allocate memory for define PIPF options, the actual define
  !     routine is PFINT
  use chm_kinds
  use dimens_fcm
  use number
  use coord
  use stream
  use psf
  use pipfm
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  !
  ! set the PIPF flag
  QPIPF=.TRUE.

  ! get PIPF options
  CALL PFINT(COMLYN,COMLEN)
  !
  RETURN
END SUBROUTINE PFPREP

SUBROUTINE PFINT(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     define PIPF related variables, for example, converge
  !     criteria, maximum iteration, etc.
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use string
  use coord
  use stream
  use psf
  use image
  use pipfm
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  INTEGER   ITDEF
  real(chm_real)    TRDEF,UMDEF,TUDEF,AFDEF
  real(chm_real), PARAMETER :: TENM5 = 1.D-5

#if KEY_PIPF==0  /* KEY_PIPF == 0 */
  CALL WRNDIE(-1,'<PFINT>','PIPF code not compiled.')
#else  /* KEY_PIPF == 0 */

  ! DEFUALT
  TRDEF = TENM5
  ITDEF = 25
  UMDEF = 0.5
  TUDEF = 1.0D0

  ! DEFAULT DAMP FACTOR IS FROM (Thole, ChemPhys 59, 341, 1981).
  AFDEF = 0.572D0
  ! MINV (matrix inverse)
  QMINV = INDXA(COMLYN, COMLEN, 'MINV') .GT. 0
  QMPOL = INDXA(COMLYN, COMLEN, 'MPOL') .GT. 0
  QVPOL = INDXA(COMLYN, COMLEN, 'VPOL') .GT. 0
  ! DYNA (ext Lagrangian)
  QPFDYN = INDXA(COMLYN, COMLEN, 'DYNA') .GT. 0
  IF (PRNLEV.GE.2) THEN
     IF (QPFDYN) THEN
        WRITE(OUTU,*) 'PFINT> Induced dipoles in PIPF ', &
             'are computed dynamically '
        WRITE(OUTU,*) '       with an extended Lagrangian '

        ! UMAS (fictious mass associated with dipole)
        UMAS=GTRMF(COMLYN,COMLEN,'UMAS',UMDEF)
        IF (PRNLEV.GE.2) WRITE(OUTU,10) UMAS
        call chmalloc('pfdyn.src','PFINT','IUMAS',NATOM,crl=IUMAS)
        CALL ASGUMAS(UMAS,IUMAS)
10      FORMAT(1X,'PFINT> Fictitious mass of dipole: ', &
             'UMAS [(ps/e*A)^2*kcal/mol] = ', F12.5)

        ! TSTA (initial dipole temperature)
        TSTAU=GTRMF(COMLYN,COMLEN,'TSTA',TUDEF)
        IF (PRNLEV.GE.2) WRITE(OUTU,20) TSTAU
20      FORMAT(1X,'PFINT> Initial dipole temperature: ', &
             'TSTA (K) = ', F12.5)

        ! UINT (nose-hoover heat bath)
        NHFLAG=GTRMI(COMLYN,COMLEN,'UINT',5)
        IF (NHFLAG.EQ.1) THEN
           IF (PFBASETUP) THEN
              IF(PRNLEV.GE.2) THEN
                 WRITE(OUTU,*)  &
                      ' Dipole Hoover Baths Are Set Up Correctly'
              ENDIF
           ELSE
              IF (QPFBA) THEN
                 CALL WRNDIE(-1,'<PFINT>', &
                      'CHECK "PFBA" SYNTAX for THERMOSTATS CHANGE')
              ENDIF
           ENDIF
        ENDIF
        IF(PRNLEV.GE.2)THEN
           WRITE(OUTU,*) 'PFINT> Dipole integrator: UINT =',  &
                NHFLAG
        ENDIF

        ! UFRS (dipole moment for first dynamics)
        NUFRS=GTRMI(COMLYN,COMLEN,'UFRS',1)
        IF (NUFRS.EQ.1) THEN
           IF(PRNLEV.GE.2) THEN
              WRITE(OUTU,*) &
                   'PFINT> UFRS=1: Initial induced dipole moment are ',    &
                   'set to zeros.'
           ENDIF
        ELSE IF (NUFRS .EQ. 2) THEN
           IF(PRNLEV.GE.2) THEN
              WRITE(OUTU,*) &
                   'PFINT> UFRS=2: Zero order induced dipoles are ', &
                   'used for first step dynamics. '
           ENDIF
        ELSE IF (NUFRS .GE. 3) THEN
           IF(PRNLEV.GE.2) THEN
              WRITE(OUTU,*) &
                   'PFINT> UFRS>2: fully converged induced dipoles are',      &
                   ' used for first step dynamics. '
           ENDIF
           !               ...CONV (dipole convergence criteria)
           DTHRES = GTRMF(COMLYN,COMLEN,'CONV',TRDEF)
           WRITE(OUTU,30) DTHRES
           !               ...ITER (max. iteration)
           ITRMX = GTRMI(COMLYN,COMLEN,'ITER',ITDEF)
           WRITE(OUTU,*)'PFINT> Max. ITER = ', ITRMX
        ENDIF

        ! ANGL (calcualte the average angle of dynamical dipole with electric field)
        QUEANG = INDXA(COMLYN, COMLEN, 'ANGL') .GT. 0
        IF (QUEANG) THEN
           IF(PRNLEV.GE.2) THEN
              WRITE(OUTU,*) &
                   'PFINT> ANGL: angle between dipoles and electric ',   &
                   'field will be calculated. '
           ENDIF
        END IF
        ! induced dipole calculated by matrix inverse
     ELSEIF(QMINV) THEN
        WRITE(OUTU,*) 'PFINT> Induced dipoles are ', &
             'determined by matrix inversion in PIPF'
        IF (QMPOL) THEN
           WRITE(OUTU,*) 'PFINT> Molecular polarizability is ', &
                'calculated by matrix inversion in PIPF'
        ENDIF
        IF (QVPOL) THEN
           WRITE(OUTU,*) 'PFINT> Vibrational analysis with ', &
                'polarization'
        ENDIF
        !
        ! Otherwise, use a classical iterative process
        !
     ELSE
        WRITE(OUTU,*) 'PFINT> Induced dipoles are ', &
             'determined iteratively in PIPF'

        ! CONV (dipole convergence criteria)
        DTHRES = GTRMF(COMLYN,COMLEN,'CONV',TRDEF)
        WRITE(OUTU,30) DTHRES
30      FORMAT(1X,'PFINT> Dipole CONV (Deby/center) = ', F12.5)

        ! ITER (max. iteration)
        ITRMX = GTRMI(COMLYN,COMLEN,'ITER',ITDEF)
        WRITE(OUTU,*)'PFINT> Max. ITER = ', ITRMX

     ENDIF

     ! CTOF (energy cut-off)
     PFCTOF = GTRMF(COMLYN,COMLEN,'CTOF',-1.0D0)
     IF (PFCTOF .LE. 0.0D0) THEN
        WRITE(OUTU,*)'PFINT> Dipole CTOF: use non-bonded cutoff'
     ELSE
        WRITE(OUTU,40) PFCTOF
     ENDIF
40   FORMAT(1X,'PFINT> Dipole CTOF = ', F12.5)

     ! DAMP (damping)
     NPDAMP=GTRMI(COMLYN,COMLEN,'DAMP',0)
     IF (NPDAMP .EQ. 0) THEN
        WRITE(OUTU,*)'PFINT> DAMP = ', NPDAMP, '(no damping)'
     ELSE IF (NPDAMP .EQ. 1) THEN
        DPFAC = GTRMF(COMLYN,COMLEN,'AFAC',AFDEF)
        WRITE(OUTU,*)'PFINT> DAMP = ', NPDAMP, '(Thole roh2)'
        WRITE(OUTU,50) DPFAC
     ELSE IF (NPDAMP .EQ. 2) THEN
        DPFAC = GTRMF(COMLYN,COMLEN,'AFAC',AFDEF)
        WRITE(OUTU,*)'PFINT> DAMP = ', NPDAMP, '(Thole roh4)'
        WRITE(OUTU,50) DPFAC
     ELSE
        CALL WRNDIE(-1,'<PFINT>', 'DAMP>2 not available')
        NPDAMP = 0
        WRITE(OUTU,*)'PFINT> Set DAMP = 0 and continue'
     ENDIF
50   FORMAT(1X,'PFINT> Gaussian width AFAC = ', F12.5)

     ! PFMODE (start point for iteration)
     PFMODE = GTRMI(COMLYN,COMLEN,'PFMD',0)
     IF(PFMODE .EQ. 0) THEN
        WRITE(OUTU,*) 'PFINT> PFMD = ',PFMODE, 'iteration start ', &
             'from 0 dipole'
     ELSE IF(PFMODE .EQ. 1) THEN
        WRITE(OUTU,*) 'PFINT> PFMD = ',PFMODE, 'iteration start ', &
             'from last dynamics step'
     ELSE
        WRITE(OUTU,*) 'PFINT> PFMD = ',PFMODE, 'invalid mode'
     ENDIF

     ! AVDP (print dipole info)
     NAVDIP=GTRMI(COMLYN,COMLEN,'AVDP',0)
     IF (NAVDIP .GT. 0) THEN
        WRITE(OUTU,*)'PFINT> Ave. dips. are calculated ',    &
             'based on units of ', NAVDIP, 'atoms'
     ENDIF

     ! EXCL (exlusion of 1-4 polarization)
     QPFEX = INDXA(COMLYN, COMLEN, 'EXCL') .GT. 0
     IF (QPFEX) THEN
        WRITE(OUTU,*)'PFINT> EXCL: exclude 1-4 polarization'
     ENDIF
     !
     ! GET THE NUMBER OF PRIMARY CELL ATOMS AND TOTAL NUMBER OF
     ! ATOMS FOR SPECIAL TREATMENT IF ANY IMAGE ATOMS ARE INCLUDED
     !
     NPFPR = NATOM
     NPFIM = NATIM

  ENDIF

#endif  /* KEY_PIPF == 0 */

  RETURN
END SUBROUTINE PFINT

SUBROUTINE EPFDY(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     ALP &
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE ALLOCATES MEMORY FOR COMPUTING POLARIZATION
  !     ENERGIES AND FORCES IN THE DYNAMICAL DIPOLE APPROACHE. THE
  !     ACTUAL CACLULATION IS DONE BY SUBROUTINE EPFDY2
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     ALP    - polarizability (passed in param.f90)
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use memory
  use energym
  use pipfm
  implicit none
  !
  real(chm_real) EPOL
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*)
  !
  INTEGER N3

  real(chm_real),allocatable,dimension(:) :: IEFIELD
  real(chm_real),allocatable,dimension(:) :: IEIND
  integer,allocatable,dimension(:) :: IIPOL
  !
  ! SKIP EPOL IF SPECIFIED.
  ! --- Better to move to the ENBOND argument list
  !
  IF (.NOT. QETERM(PIPF)) RETURN

  ! SKIP EPOL IF PIPF IS NOT SPECIFIED EXPLICITLY
  IF (.NOT. QPIPF) RETURN
  !
  !     ALLOCATE MEMORY
  !
  N3 = 3*NATOM
  call chmalloc('pfdyn.src','EPFDY','IEFIELD',N3,crl=IEFIELD)
  call chmalloc('pfdyn.src','EPFDY','IEIND',N3,crl=IEIND)
  call chmalloc('pfdyn.src','EPFDY','IIPOL',NATOM,intg=IIPOL)
  !
  !     GET POINTER TO DIPOLE MOMENT FROM COMMON BLOCK, WHICH
  !     HAS BEEN ALLOCATED BY DYNAMICS ROUTINE
  !

  CALL EPFDY2(IFRSTA,NATOM,JNB,INBLO,CG, &
       MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
       LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
       EPOL,ALP,IEFIELD,IEIND,IIPOL,IESAV &
       )
  !
  !     FREE MEMORY
  !
  call chmdealloc('pfdyn.src','EPFDY','IEFIELD',N3,crl=IEFIELD)
  call chmdealloc('pfdyn.src','EPFDY','IEIND',N3,crl=IEIND)
  call chmdealloc('pfdyn.src','EPFDY','IIPOL',NATOM,intg=IIPOL)

  RETURN
END SUBROUTINE EPFDY
!
SUBROUTINE EPFDY2(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     EPOL,ALP,EFIELD,EIND,IPOL,ESAV &
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES POLARIZATION ENERGIES AND FORCES
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability (passed in param.f90)
  !
  !     DMUIND - induced dipole for current iteration (UIND)
  !     EFIELD - permenent electric field
  !     EIND - induced electric field
  !     IPOL - tag to mark an atom as a dipole center
  !     DDMUIND - induction force on induced dipole (DUIND)
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use energym
  use stream
  use pipfm
#if KEY_PBOUND==1
  use pbound  
#endif
  implicit none
#if KEY_PBOUND==1
  real(chm_real) CORR  
#endif
  !
  real(chm_real) EPOL
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*)
  !
  INTEGER NITER, NCOUNT
  INTEGER I,J,JPR,NB,NPR,ITEMP,I1,J1
  real(chm_real) EFIELD(3,*),EIND(3,*)
  real(chm_real) E14M1,E14F,CGT2,CGT
  INTEGER IPOL(*)
  real(chm_real) XX,YY,ZZ,R1,R2,R3,QR3I,QR3J, &
       RR3,RR3XX,RR3YY,RR3ZZ
  real(chm_real) TXWW(6)
  real(chm_real) CGI,CGJ,DPR1,DPR2,DPR3,R5
  !
  INTEGER IANG
  real(chm_real) ESAV(3,*),UEANG,UXI,UYI,UZI,EXI,EYI,EZI
  !
  real(chm_real) U3,AU3,FLMD3,FLMD5,FLMD7
  !
  real(chm_real) RIJ2,PFCTOF2
  PFCTOF2=PFCTOF*PFCTOF
  !
  !     SET DIPOLE CENTER TAG TO ZERO
  !
  DO I = 1, NATOM
     IPOL(I) = 0
  END DO
  !
  CALL GZERO(EFIELD,3,NATOM)
  CALL GZERO(EIND,3,NATOM)
  !
  NB = 0
  ITEMP = 0

  ! DO BLOCK EXPANSION OF CODE TO IMPROVE EFFICIENCY
  IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.0) THEN
#undef PIPF_CTOF
#undef PIPF_DAMP
#include "pfdyn.inc"
  ELSE IF(PFCTOF.GT.0.0D0 .AND. NPDAMP.EQ.0) THEN
#define PIPF_CTOF 1
#include "pfdyn.inc"
#undef PIPF_CTOF
  ELSE IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.1) THEN
#define PIPF_DAMP 1
#include "pfdyn.inc"
  ELSE
#define PIPF_CTOF 1
#include "pfdyn.inc"
#undef PIPF_CTOF
#undef PIPF_DAMP
  ENDIF

  !     POLARIZATION ENERGY AND FORCE (CONVERT TO KMA UNIT)
  !
  !     NOTE: IN ENERGY, THE INDUCED FIELD HERE HAS A PREFACTOR
  !           OF 0.5. THIS IS THE RESULT OF THE NET
  !           DIPOLE-DIPOLE ITERACTION (-) WHERE A 1/2 IS INTRODUCED
  !           FOR DOUBLE COUNTING SINCE FOR ONE PAIR OF DIPOLE THERE
  !           IS ONLY ONE INTERACTION ENERGY BETWEEN THEM. WE CAN
  !           CALCULATE THE TOTAL ENERGY AS IF THE DIPOLE IS PRESENT
  !           AN EFFECTIVE FIELD OF THE PERMENT FIELD PLUS 1/2 OF
  !           THE INDUCED FIELD. NOTE THIS IS ONLY FOR CONVINIENCE IN
  !           ENERGY CALCULATIONS; THE ACTUAL TOTAL ELECTRIC FIELD
  !           SHOULD HAVE A PREFACTOR OF ONE.
  !
  !           ALSO, IN DIPOLE FORCE, THE PREFACTOR OF THE INDUCED
  !           DIPOLE BECOMES ONE, SINCE THE DIPOLE IS A SQUARE
  !           TERM, WHICH CANCELS THE 1/2 WHEN A DERIVATIVE IS MADE.
  !

  !
  ! Primary-Primary interaction
  !
  IF (NATOM .LE. NPFPR) THEN
     EPOL = 0.0D+00
     DO I = IFRSTA,NATOM
        IF(IPOL(I).EQ.1) THEN

           I1=ITC(IAC(I))

           EPOL = EPOL-CCELEC * ( UIND(1,I)*EFIELD(1,I)+    &
                UIND(2,I)*EFIELD(2,I)+ &
                UIND(3,I)*EFIELD(3,I)+     &
                                ! dipole-dipole
                0.5D0*(UIND(1,I)*EIND(1,I)+ &
                UIND(2,I)*EIND(2,I)+ &
                UIND(3,I)*EIND(3,I)) - &
                                ! self polarization
                (UIND(1,I)*UIND(1,I) +  &
                UIND(2,I)*UIND(2,I) + &
                UIND(3,I)*UIND(3,I))/     &
                (2.0D0*ALP(I1)) )
           !
           ! INDUCTION ENERGY FORCE ON DIPOLE (SHOULD BE ZERO BEFORE THE RPIMARY
           ! DIPOLE CALCULTION, SIMILIAR TO THE NUCLEAR FORCE, BUT ACCUMULATE
           ! FOR THE IMAGE DIPOLE, NOT PROPERLY ZERO YET).
           !
           DUIND(1,I) = DUIND(1,I) - CCELEC* &
                (EFIELD(1,I)+EIND(1,I)-UIND(1,I)/ALP(I1))
           DUIND(2,I) = DUIND(2,I) - CCELEC* &
                (EFIELD(2,I)+EIND(2,I)-UIND(2,I)/ALP(I1))
           DUIND(3,I) = DUIND(3,I) - CCELEC* &
                (EFIELD(3,I)+EIND(3,I)-UIND(3,I)/ALP(I1))

           !
           ! debug
           !              write(*,*) 'du_pp(1,',i,')=', duind(1,i)
           !              write(*,*) 'du_pp(2,',i,')=', duind(2,i)
           !              write(*,*) 'du_pp(3,',i,')=', duind(3,i)
           !

        ENDIF
     ENDDO
  ENDIF

  !
  ! IF IT IS AN IMAGE-PRIMARY INTERACTION, THEN WE DO NOT RECOUNT THE
  ! SELF-ENERGY FOR PRIMARY DIPOLE AGAIN, SUCH PENALTY ENERGY HAS BEEN
  ! INCLUDED WHEN PRIMARY-PRIMARY NONBONDED LIST IS PROCESSED. THE FORCE
  ! ON IMAGE DIPOLE WILL BE TRANSFERRED TO PRIMARY ATOMS AT THE BEGINING
  ! THE ENERGY CALCULATION.
  !
  IF (NATOM .GT. NPFPR) THEN
     EPOL = 0.0D+00
     DO I = IFRSTA,NATOM
        IF(IPOL(I).EQ.1) THEN
           I1=ITC(IAC(I))
           EPOL = EPOL-CCELEC * ( UIND(1,I)*EFIELD(1,I)+    &
                UIND(2,I)*EFIELD(2,I)+ &
                UIND(3,I)*EFIELD(3,I)+ &
                0.5D0 * (UIND(1,I)*EIND(1,I)+    &
                UIND(2,I)*EIND(2,I)+ &
                UIND(3,I)*EIND(3,I)) )
           !
           ! INDUCTION ENERGY FORCE ON DIPOLE (HAS BE ZERO BEFORE EACH
           ! ENERGY RUN, FORCE ON IMAGE DIPOLE HAS BEEN TRANSFERED TO
           ! PRIMARY DIPOLE CENTER AS WELL)
           !
           DUIND(1,I) = DUIND(1,I) - CCELEC* &
                (EFIELD(1,I)+EIND(1,I))
           DUIND(2,I) = DUIND(2,I) - CCELEC* &
                (EFIELD(2,I)+EIND(2,I))
           DUIND(3,I) = DUIND(3,I) - CCELEC* &
                (EFIELD(3,I)+EIND(3,I))

           !
           ! debug
           !
           !              write(*,*) 'du_pi(1,',i,')=', duind(1,i)
           !              write(*,*) 'du_pi(2,',i,')=', duind(2,i)
           !              write(*,*) 'du_pi(3,',i,')=', duind(3,i)
           !

        ENDIF
     ENDDO
  ENDIF

  IF (PRNLEV .GE. 7) THEN
     WRITE(OUTU,*) 'Polarization Energy (EPOL):', EPOL
  END IF
  !
  ! STORE EPOL TO ENERGY TERM COMMON BLOCK
  ! --- BETTER TO MOVE TO THE ENBOND ARGUMENT LIST
  !     MODIFIED TO CUMULATIVE ADDITION TO HANDEL BOTH
  !     PRIMARY AND IMAGE POLARIZATION ENERGY
  !         ETERM(PIPF) = EPOL
  !
  ETERM(PIPF) = ETERM(PIPF)+EPOL

  !
  ! CALCULATE THE AVERAGE ANGLE BETWEEN THE DYNAMICAL INDUCED
  ! DIPOLE WITH THE TOTAL ELECTRIC FIELD
  !
  IF (QUEANG) THEN
     IF (NPFIM .LE. NPFPR) THEN

        IANG = 0
        UEANG = 0.0D0
        DO I = IFRSTA,NATOM
           IF(IPOL(I).EQ.1) THEN
              IANG = IANG + 1
              UXI = UIND(1,I)
              UYI = UIND(2,I)
              UZI = UIND(3,I)
              EXI = EFIELD(1,I) + EIND(1,I)
              EYI = EFIELD(2,I) + EIND(2,I)
              EZI = EFIELD(3,I) + EIND(3,I)
              UEANG = UEANG + ACOS((UXI*EXI+UYI*EYI+UZI*EZI)/ &
                   SQRT((UXI*UXI+UYI*UYI+UZI*UZI)* &
                   (EXI*EXI+EYI*EYI+EZI*EZI)))
           END IF
        END DO
        UEANG = (UEANG / IANG)/PI * 180.0D0
        IF (PRNLEV .GE. 2) WRITE(OUTU,10) UEANG
10      FORMAT(1X,'Ave. U-E angles: ', F8.3)

     ELSE IF (NPFIM .GT. NPFPR) THEN
        IF (NATOM .LE. NPFPR) THEN
           CALL GZERO(ESAV,3,NPFIM)
           DO I = IFRSTA,NATOM
              IF(IPOL(I).EQ.1) THEN
                 ESAV(1,I) = ESAV(1,I) + EFIELD(1,I) + EIND(1,I)
                 ESAV(2,I) = ESAV(2,I) + EFIELD(2,I) + EIND(2,I)
                 ESAV(3,I) = ESAV(3,I) + EFIELD(3,I) + EIND(3,I)
              ENDIF
           ENDDO
        ELSE IF (NATOM .GT. NPFPR) THEN
           DO I = IFRSTA,NATOM
              IF(IPOL(I).EQ.1) THEN
                 ESAV(1,I) = ESAV(1,I) + EFIELD(1,I) + EIND(1,I)
                 ESAV(2,I) = ESAV(2,I) + EFIELD(2,I) + EIND(2,I)
                 ESAV(3,I) = ESAV(3,I) + EFIELD(3,I) + EIND(3,I)
              ENDIF
           ENDDO
           !
           CALL DPTRANSA(ESAV,NPFPR,NPFIM)
           !
           IANG = 0
           UEANG = 0.0D0
           DO I = 1,NPFPR
              IANG = IANG + 1
              UXI = UIND(1,I)
              UYI = UIND(2,I)
              UZI = UIND(3,I)
              EXI = ESAV(1,I)
              EYI = ESAV(2,I)
              EZI = ESAV(3,I)
              UEANG = UEANG + ACOS((UXI*EXI+UYI*EYI+UZI*EZI)/ &
                   SQRT((UXI*UXI+UYI*UYI+UZI*UZI)* &
                   (EXI*EXI+EYI*EYI+EZI*EZI)))
           ENDDO
           UEANG = (UEANG / IANG)/PI * 180.0D0
           IF (PRNLEV .GE. 2) WRITE(OUTU,10) UEANG
        END IF
     END IF
  END IF

  !
  ! PRINT OUT DIPOLE INFOR
  !
  IF (NAVDIP .GT. 0) THEN
     IF (NATOM .GT. NPFPR) THEN
        CALL PRDIP(NPFPR,NAVDIP,X,Y,Z,CG,UIND)
     END IF
  END IF

  RETURN
END SUBROUTINE EPFDY2

SUBROUTINE NOSEPF(COMLYN,COMLEN)
  !-----------------------------------------
  !   this routine sets up multiple heat baths
  !   for induced dipole kinetics in PIPF
  !
  !   adopted from algorithm by M. Watanabe
  !   Sandeep Patel
  !
  !-------------------------------------
  use chm_kinds
  !
  use dimens_fcm
  use memory
  use number
  !
  use psf
  use stream
  use string
  use nose_mod
  use pipfm

  implicit none

  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  !
  LOGICAL     UEND,UCHECK,EOF,LUSED
  INTEGER     NINT,JINT,I,J,K,JTEMP,UDEGF

  integer,allocatable,dimension(:) :: ITEMP

  CHARACTER(len=4) WRD
  real(chm_real), PARAMETER :: ROOMT=298.0D0

  PFBASETUP = .FALSE.
  EOF = .FALSE.
  UCHECK = .FALSE.
  UEND = .FALSE.

  !     PFBA keyword has been found
  QPFBA = .TRUE.

  !     get number of baths
  NPFBATHS = NEXTI(COMLYN,COMLEN)

  if (PRNLEV.GE.2) WRITE(OUTU,*) " NUMBER OF BATHS = ", NPFBATHS

  !     initialize some variables for each bath
  DO K = 1, NPFBATHS
     PFNHSBATH(K)=1.0
     PFNHSOBATH(K)=1.0
     PFNHMBATH(K)=0.005
     PFTEMPBATH(K)=1.0
  ENDDO


10 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
       EOF,.TRUE.,.TRUE.,'PFBA> ')
  CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
  WRD=NEXTA4(COMLYN,COMLEN)
  IF (WRD .EQ. '    ') LUSED=.TRUE.
  IF (LUSED) GOTO 10
  !
20 CONTINUE
  IF (WRD .EQ. 'CALL') THEN
     if (prnlev.ge.2) write(outu,*) " CALLING "
     !     Procedure ASSIGN-A-BATH
     I=NEXTI(COMLYN,COMLEN)
     IF (I .LE. NPFBATHS) THEN
        call chmalloc('pfdyn.src','NOSEPF','ITEMP',NATOM,intg=ITEMP)
        if (prnlev.ge.2) write(outu,*) " CALL ASSIGN BATH "
        call PFBATHASGN(COMLYN,COMLEN,ITEMP,NATOM,INLCKP, &
             I,NRES,IBASE,UDEGF)

        NDGFBPF(I) = UDEGF

        if (prnlev.ge.2) write(outu, '(A, 2I8)') &
             "NUMBER OF DEG. OF U FREEDOM for BAth =",I,UDEGF
        call chmdealloc('pfdyn.src','NOSEPF','ITEMP',NATOM,intg=ITEMP)

        if (prnlev.ge.2) write(outu,*) " DONE ASSIGN U's"
        UCHECK=.TRUE.
        IF(PRNLEV.GE.2) WRITE(OUTU,30) I
30      FORMAT(' The selected atoms have been reassigned to block',I4)
     ELSE
        CALL WRNDIE(-3,'<NOSE>', &
             'Failed attempt to reassign atoms.  Block number too high.')
     ENDIF

     !
  ELSE IF (WRD .EQ. 'COEF') THEN
     I=NEXTI(COMLYN,COMLEN)
     PFNHMBATH(I)=GTRMF(COMLYN,COMLEN,'UREF',ZERO)
     PFTEMPBATH(I)=GTRMF(COMLYN,COMLEN,'TREF',ROOMT)


  ELSE IF (WRD .EQ. 'END ') THEN
     !     Procedure FINISH-UP-AND-END
     CALL XTRANE(COMLYN,COMLEN,'PFBA')
     GOTO 700

  ELSE
  ENDIF

  CALL XTRANE(COMLYN,COMLEN,'PFBA')
  IF (UEND) THEN
     WRD='END '
  ELSE
690  CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
          EOF,.TRUE.,.TRUE.,'PFBA> ')
     CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
     WRD=NEXTA4(COMLYN,COMLEN)
     IF (WRD .EQ. '    ') LUSED=.TRUE.
     IF (LUSED) GOTO 690
  ENDIF
  GOTO 20


700 CONTINUE


  DO K = 1,NPFBATHS
     IF (PRNLEV.GE.2) WRITE(OUTU,*) &
          K,PFNHMBATH(K),PFTEMPBATH(K),PFNHSBATH(K), &
          PFNHSOBATH(K), IFSTBPF(K),ILSTBPF(K), NDGFBPF(K)
  ENDDO

  PFBASETUP = .TRUE.

  RETURN
END SUBROUTINE NOSEPF


SUBROUTINE PFBATHASGN(COMLYN,COMLEN,ITEMP,NATOM,IBLOCK, &
     INDEX,NRES,IBASE,UDEGF)
  !-------------------------------------------------------
  !    this subroutine handles assignment of induced dipole
  !    to specific heat baths
  !
  !     adopted algorithm from M. Watanabe
  !   Sandeep Patel
  !-----------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use coord
  use select
  use stream
  use pipfm
  implicit none
  !
  INTEGER ITEMP(*), IBLOCK(*),IBASE(*)
  INTEGER NATOM, COMLEN, INDEX, ICOUNT,ILASTATM
  CHARACTER(len=*) COMLYN
  !     LOCAL
  INTEGER I
  INTEGER NPF,NPFR,UDEGF,NRES,J
  LOGICAL  INRES
  !
  !     BEGIN
  !
  CALL SELCTA(COMLYN,COMLEN,ITEMP,X,Y,Z,WMAIN,.TRUE.)
  ICOUNT=0
  DO I=1, NATOM
     IF (ITEMP(I) .EQ. 1) THEN
        ICOUNT=ICOUNT+1
        ILASTATM=I
     ENDIF
  ENDDO

  ILSTBPF(INDEX) = ILASTATM
  IFSTBPF(INDEX) = ILASTATM - ICOUNT + 1
  !

  !   get number of degrees of freedom

  !C degrees of freedom=3 x no. of PIPF atoms-no. of restraints (residues)
  !  Q: do we have restraint degree of freedom in dipole? It seems
  !     CHEQ excluded the numer of residue as restraints, do we do
  !     that? Does it mean in CHEQ, the charge for each residue
  !     is conserved?
  !
  UDEGF = 0
  NPF=0
  NPFR=0
  DO I=1,NRES
     INRES=.FALSE.
     DO J=IBASE(I)+1,IBASE(I+1)
        IF (ITEMP(J).NE.0) THEN
           NPF=NPF+1
           INRES=.TRUE.
        ENDIF
     ENDDO
     IF (INRES) NPFR=NPFR+1
  ENDDO
  IF ((NPF-NPFR).LE.0) THEN
     CALL WRNDIE(-4,'<PFDEGF>', &
          'Number of dipole degrees of freedom zero or negative!')
     UDEGF=0
  ELSE
     UDEGF=NPF-NPFR
     !        UDEGF = NPF
     IF (PRNLEV.GE.5) WRITE(OUTU,10) NPF-NPFR
10   FORMAT(' PFDEGF> System has ',I8, &
          ' degrees of dipole freedom')
  ENDIF

  RETURN
END SUBROUTINE PFBATHASGN

!---------------------------------------------------------
SUBROUTINE ASGUMAS(UMA,PMASSU)
  !---------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use coord
  use psf
  use consta
  implicit none

  INTEGER I
  real(chm_real)  UMA,PMASSU(NATOM)

  DO I=1,NATOM
     PMASSU(I)=UMA/(TIMFAC*TIMFAC)
  ENDDO
  RETURN
END SUBROUTINE ASGUMAS


!----------------------------------------------------------------------
SUBROUTINE PFDYN(UINDO,UINDN,VUIND,PMASSU,DELTA)
  !----------------------------------------------------------------------
  ! This routine carries out MD verlet step for the induced dipole in PIPF
  !
  use chm_kinds
  use consta
  use dimens_fcm
  use energym
  use number
  use psf
  use stream
  use nose_mod
  use pipfm
  implicit none
  !
  real(chm_real) UINDO(3,*),UINDN(3,*),VUIND(3,*),     &
       PMASSU(*),DELTA
  !
  INTEGER I,J,K,L,M,NHITR,PFNHMX,IMAX
  real(chm_real) PFNHTL,PFTEMP,PFCDGF,UMASS,KEU,     &
       PFNHABATH(10),PFNHSNBATH(10),NHSDIFBATH(10), &
       MAXERROR,MAXEOLD,MV2TMPBATH(10),ERROR
  !
  !
  ! SKIP EPOL IF SPECIFIED.
  ! --- Better to move to the ENBOND argument list
  !
  IF (.NOT. QETERM(PIPF)) RETURN
  !
  IF (.NOT. QPIPF .OR. .NOT. QPFDYN) RETURN
  !
  IF (NHFLAG.EQ.1) THEN

     PFNHMX  = 1000
     PFNHTL = 0.000001
     DO K = 1, NPFBATHS
        KEUBATH(K) = 0.0
        DO L = IFSTBPF(K), ILSTBPF(K)
           DO M = 1, 3
              KEUBATH(K)=KEUBATH(K)+PMASSU(L)*VUIND(M,L)*VUIND(M,L)
           END DO
        ENDDO
        !
        ! Degree of freedom of dynamical dipole = 3 x no. of dipole centers+1
        ! (Here we do not apply any constraints like conserved residue charge
        ! in CHEQ)
        !
        ! Rigourously, we should only count those atoms within the non-bonded
        ! list, which have a dipole center, I am not sure treat all atoms in an
        ! equal footing will be correct, since some of atoms do not participate
        ! the dipole dynamics)
        !
        NDGFBPF(K) =( ILSTBPF(K)-IFSTBPF(K) + 1 ) * 3 + 1

     ENDDO

     !
     ! Do Nose-Hoover iterations if necessary
     !
     IF (PFNHMBATH(1).NE.ZERO) THEN
        DO NHITR=1,PFNHMX
           DO K = 1,NPFBATHS
              PFTEMP = PFTEMPBATH(K)
              PFCDGF = NDGFBPF(K)
              PFNHABATH(K) = (KEUBATH(K)-(PFCDGF)*KBOLTZ*PFTEMP)/     &
                   PFNHMBATH(K)
              PFNHSNBATH(K)= TWO*PFNHSBATH(K)-PFNHSOBATH(K)+ &
                   DELTA**2*PFNHABATH(K)
              NHSDIFBATH(K)= PFNHSNBATH(K)-PFNHSOBATH(K)
           ENDDO

           ! Propagate the dipoles by standard Verlet, including the calculated
           ! scaling constant, and get a new estimate for mv**2

           MAXERROR = -1000.0
           MAXEOLD = MAXERROR
           IMAX=0
           DO K = 1, NPFBATHS
              MV2TMPBATH(K)=0.0
              DO L = IFSTBPF(K),ILSTBPF(K)
                 UMASS=PMASSU(L)
                 DO M = 1, 3
                    UINDN(M,L)=(TWO*UIND(M,L)-UINDO(M,L)* &
                         (ONE-0.25D0*NHSDIFBATH(K))-    &
                         DELTA**2*DUIND(M,L)/UMASS)/ &
                         (ONE+0.25D0*NHSDIFBATH(K))

                    VUIND(M,L)=(UINDN(M,L)-UINDO(M,L))/(TWO*DELTA)

                    MV2TMPBATH(K)=MV2TMPBATH(K)+ &
                         UMASS*VUIND(M,L)*VUIND(M,L)
                 ENDDO
              ENDDO
              ERROR = ABS(MV2TMPBATH(K)-KEUBATH(K))
              MAXERROR = MAX(ERROR,MAXERROR)
              IF(MAXERROR.NE.MAXEOLD) IMAX=K
              MAXEOLD = MAXERROR
           ENDDO

           ! If the new value of mv**2 is close enough to the old, we have reached
           ! self-consistency and can continue. Otherwise, do another Nose iteration

           IF (MAXERROR.LT.PFNHTL) THEN

              ! Add the Nose-Hoover contributions to the energies
              !                 IF (MOD(ISTEP,100).EQ.0) THEN
              !                    WRITE(69,10) NHITR,PFNHS,NHSDIF/(TWO*DELTA),
              !     &                           PFNHA,PFCMV2/TWO
              !19                  FORMAT(' PFCHP2> ',I4,' iterations; S= ',F9.3,
              !     &                     ' dS/dt= ',F9.3,' d2S/dt2= ',F9.3,' KE=',F9.3)
              !                 ENDIF

              KEU=ZERO
              DO K=1,NPFBATHS
                 KEUBATH(K) = MV2TMPBATH(K)
                 PFNHSOBATH(K) = PFNHSBATH(K)
                 PFNHSBATH(K) = PFNHSNBATH(K)
                 KEU = KEU + KEUBATH(K)
              ENDDO
              GOTO 1039
           ENDIF

           DO K = 1,NPFBATHS
              KEUBATH(K) = MV2TMPBATH(K)
           ENDDO

        ENDDO
        CALL WRNDIE(-3,'<PFCNW2>', &
             'Maximum Nose-Hoover iterations exceeded')
        !    &        KEU, KESOLUTE, KEWATER=',KEU,KEUSOL,MV2TMPWAT)

     ENDIF

     !
     !  UPDATE DIPOLES
     !

1039 CONTINUE

     DO I = 1,NATOM
        DO M = 1, 3
           UINDO(M,I)=UIND(M,I)
           UIND(M,I) = UINDN(M,I)
        END DO
     ENDDO

  ELSEIF (NHFLAG.EQ.2) THEN

     KEU=0.0
     DO I=1,NATOM
        DO M = 1, 3
           UINDN(M,I) = 2.0D0*UIND(M,I) - UINDO(M,I) -      &
                (DELTA**2)*DUIND(M,I)/PMASSU(I)
           VUIND(M,I)=(UINDN(M,I)-UINDO(M,I))/(2.0D0*DELTA)
           KEU=KEU+PMASSU(I)*VUIND(M,I)*VUIND(M,I)
           UINDO(M,I)= UIND(M,I)
           UIND(M,I)= UINDN(M,I)
        END DO
     ENDDO

  ENDIF

  !
  ! GET DIPOLE KINETIC ENERGY AND TEMPERATURE (For Dipole temperature, this
  ! is not exactly true, we should only count those atoms within the non-bonded
  ! list, which have a dipole center, see the disscussion above about the
  ! dipole degrees of freedom)
  !
  EPROP(DIPK) =  KEU
  EPROP(DIPT) =  KEU/(KBOLTZ*NATOM*3)

  RETURN
END SUBROUTINE PFDYN


!-----------------------------------------------------------------------------
SUBROUTINE UINDASG(K,I,UIND)
  !-----------------------------------------------------------------------------
  ! Assign dynammical dipole moment to image atoms
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  INTEGER K, I, J
  real(chm_real) UIND(3,*)
  !
  DO J = 1, 3
     UIND(J,K) = UIND(J,I)
  ENDDO
  !
  RETURN
END SUBROUTINE UINDASG

!-----------------------------------------------------------------------------
SUBROUTINE DUZERO(I,DUIND)
  !-----------------------------------------------------------------------------
  ! Zero dipole gradient before energy calculations in dynamical dipole scheme
  use chm_kinds
  use consta
  use dimens_fcm
  implicit none
  !
  INTEGER I, J
  real(chm_real) DUIND(3,*)
  !
  DO J = 1, 3
     DUIND(J,I) = 0.0D0
  ENDDO
  !
  RETURN
END SUBROUTINE DUZERO


SUBROUTINE DPFST(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     ALP &
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATE THE ZERO ORDER INDUCED DIPOLE
  !     FOR DYNAMICS FIRST STEP (OR CALCULATE THE FULLY CONVERGED
  !     DIPOLE FOR THE FIRST STEP)
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     ALP    - polarizability (passed in param.f90)
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use memory
  use energym
  use pipfm
  implicit none
  !
  real(chm_real) EPOL
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*)
  !
  INTEGER N3,NB,ITEMP,NPR,NB6,I

  real(chm_real),allocatable,dimension(:) :: IEFIELD
  real(chm_real),allocatable,dimension(:) :: IDDT
  real(chm_real),allocatable,dimension(:) :: IEIND
  integer,allocatable,dimension(:) :: IIPOL
  !
  ! SKIP EPOL IF SPECIFIED.
  ! --- Better to move to the ENBOND argument list
  !
  IF (.NOT. QETERM(PIPF)) RETURN

  ! SKIP EPOL IF PIPF IS NOT SPECIFIED EXPLICITLY
  IF (.NOT. QPIPF) RETURN

  !
  ! DOING NOTHING IF USE ZERO-VALUED INITIAL DIPOLE
  !
  IF (NUFRS .EQ. 1) THEN

     !
     ! USE A ZERO-ORDER INDUCED DIPOLE: u = alp.E^0
     !
  ELSE IF (NUFRS .EQ. 2) THEN
     !
     !     ALLOCATE MEMORY
     !
     N3 = 3*NATOM
     call chmalloc('pfdyn.src','DPFST','IEFIELD',N3,crl=IEFIELD)
     call chmalloc('pfdyn.src','DPFST','IIPOL',NATOM,intg=IIPOL)
     !
     !     GET POINTER TO DIPOLE MOMENT FROM COMMON BLOCK, WHICH
     !     HAS BEEN ALLOCATED BY DYNAMICS ROUTINE
     !

     CALL DPFST2(IFRSTA,NATOM,JNB,INBLO,CG, &
          MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
          LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
          EPOL,ALP,IEFIELD,IIPOL &
          )
     !
     !     FREE MEMORY
     !
     call chmdealloc('pfdyn.src','DPFST','IEFIELD',N3,crl=IEFIELD)
     call chmdealloc('pfdyn.src','DPFST','IIPOL',NATOM,intg=IIPOL)

     !
     ! OR USE A FULLY-CONVERGED DIPOLE AS THE FIRST STEP
     ! WE DO NOT SAVE DIPOLE TENSOR FOR MEMORY EFFICIENCY, SINCE
     ! THIS IS ONLY AN ONE-DYNAMICS STEP CALCULAITON
     !
  ELSE IF (NUFRS .GE. 3) THEN
     NB = 0
     ITEMP = 0
     DO I=IFRSTA,NATOM
        NPR=INBLO(I)-ITEMP
        ITEMP=INBLO(I)
        IF(NPR.GT.0) NB = NB + NPR
     END DO
     !
     !     ALLOCATE MEMORY
     !
     N3 = 3*NATOM
     NB6 = 6*NB
     call chmalloc('pfdyn.src','DPFST','IDDT',N3,crl=IDDT)
     call chmalloc('pfdyn.src','DPFST','IEFIELD',N3,crl=IEFIELD)
     call chmalloc('pfdyn.src','DPFST','IEIND',N3,crl=IEIND)
     call chmalloc('pfdyn.src','DPFST','IIPOL',NATOM,intg=IIPOL)

     CALL DPFST3(IFRSTA,NATOM,JNB,INBLO,CG, &
          MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
          LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
          EPOL,ALP,IDDT,IEFIELD,IEIND,IIPOL)
     !
     !     FREE MEMORY
     !
     call chmdealloc('pfdyn.src','DPFST','IDDT',N3,crl=IDDT)
     call chmdealloc('pfdyn.src','DPFST','IEFIELD',N3,crl=IEFIELD)
     call chmdealloc('pfdyn.src','DPFST','IEIND',N3,crl=IEIND)
     call chmdealloc('pfdyn.src','DPFST','IIPOL',NATOM,intg=IIPOL)

  END IF

  RETURN
END SUBROUTINE DPFST

SUBROUTINE DPFST2(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     EPOL,ALP,EFIELD,IPOL &
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES ZERO ORDER DIPOLE MOMENT
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability (passed in param.f90)
  !
  !     UIND   - induced dipole for current iteration (ltm_pipf)
  !     EFIELD - total electric field
  !     IPOL   - tag to mark an atom as a dipole center
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use energym
  use pipfm
#if KEY_PBOUND==1
  use pbound  
#endif
  implicit none
#if KEY_PBOUND==1
  real(chm_real) CORR  
#endif
  !
  real(chm_real) EPOL
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*)

  !
  INTEGER NITER, NCOUNT
  INTEGER I,J,JPR,NB,NPR,ITEMP,I1,J1
  real(chm_real) EFIELD(3,*)
  real(chm_real) E14M1,E14F,CGT2,CGT
  INTEGER IPOL(*)
  real(chm_real) XX,YY,ZZ,R1,R2,R3,QR3I,QR3J, &
       RR3,RR3XX,RR3YY,RR3ZZ
  real(chm_real) TXWW(6)
  !
  real(chm_real) U3,AU3,FLMD3,FLMD5
  !
  real(chm_real) RIJ2,PFCTOF2
  PFCTOF2=PFCTOF*PFCTOF
  !
  !     SET DIPOLE CENTER TAG TO ZERO
  !
  DO I = 1, NATOM
     IPOL(I) = 0
  END DO
  !
  CALL GZERO(EFIELD,3,NATOM)
  !
  NB = 0
  ITEMP = 0
  !
  !=======================================================================
  !   MAIN LOOP BEGIN
  !=======================================================================
  DO I=IFRSTA,NATOM
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR.GT.0) THEN

        I1=ITC(IAC(I))

        DO JPR=1,NPR
           NB=NB+1
           IF (JNB(NB).LT.0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNB(NB)
           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNB(NB)
           ENDIF

           J1=ITC(IAC(J))

           XX = X(I)-X(J)
           YY = Y(I)-Y(J)
           ZZ = Z(I)-Z(J)

#if KEY_PBOUND==1 /*pbound*/
           If(qBoun) then
              If(qCUBoun.or.qTOBoun) then
                 XX = BOXINV * XX
                 YY = BOYINV * YY
                 ZZ = BOZINV * ZZ
                 xx = xx - nint(xx)
                 yy = yy - nint(yy)
                 zz = zz - nint(zz)
!!$                 IF(XX.GT.  HALF) XX = XX - ONE
!!$                 IF(XX.LT. -HALF) XX = XX + ONE
!!$                 IF(YY.GT.  HALF) YY = YY - ONE
!!$                 IF(YY.LT. -HALF) YY = YY + ONE
!!$                 IF(ZZ.GT.  HALF) ZZ = ZZ - ONE
!!$                 IF(ZZ.LT. -HALF) ZZ = ZZ + ONE
                 If (qTOBoun) Then
                    CORR = HALF * AINT ( R75 * (ABS(XX) + &
                         ABS(YY) + &
                         ABS(ZZ)))
                    XX = XX - SIGN( CORR,  XX  )
                    YY = YY - SIGN( CORR,  YY  )
                    ZZ = ZZ - SIGN( CORR,  ZZ  )
                 Endif
                 XX = XSIZE * XX
                 YY = YSIZE * YY
                 ZZ = ZSIZE * ZZ
              Else
                 Call PBMove(XX, YY, ZZ)
              Endif
           Endif
#endif /*      (pbound)*/
           !
           !      APPLY AN ENERGY CUT-OFF
           !
           RIJ2 = XX*XX+YY*YY+ZZ*ZZ
           IF (RIJ2 .LT. PFCTOF2) THEN
              !
              !           R2 = 1.D+00/(XX*XX+YY*YY+ZZ*ZZ)
              !
              R2 = 1.D+00/RIJ2
              R1 = SQRT(R2)
              R3 = R1*R2
              !
              !      PERMENENT ELECTRIC FIELD
              !      E0 FOLLOW EQ. (5) in CHEM. PHYS. LETT. 1990, 166, 180.
              !
              !               E^i = - sum(T^iA Q^A) and T^iA = - a/R^3 (a denotes a^i-a^j)
              !                a       A   a             a
              !
              !
              !       NO DAMPING (DEFAULT)
              !
              IF (NPDAMP .EQ. 0) THEN
                 QR3I = CG(I)*R3
                 QR3J = CG(J)*R3
                 !
                 !       OR APPLY A DAMPING FACTOR
                 !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 !
              ELSE IF (NPDAMP .EQ. 1) THEN

                 U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
                 AU3 = DPFAC*U3
                 FLMD3 = 1.0D0 - EXP(-AU3)
                 FLMD5 = 1.0D0 - (1.0D0+AU3)*EXP(-AU3)
                 QR3I = CG(I)*R3*FLMD3
                 QR3J = CG(J)*R3*FLMD3

              END IF

              IF (QPFEX .AND. JNB(NB).LT.0) THEN
                 ! OPTIONALLY EXCLUDE 1-4 POLARIZATION, DO NOTHING
              ELSE
                 EFIELD(1,I) = EFIELD(1,I)+XX*QR3J
                 EFIELD(2,I) = EFIELD(2,I)+YY*QR3J
                 EFIELD(3,I) = EFIELD(3,I)+ZZ*QR3J
                 EFIELD(1,J) = EFIELD(1,J)-XX*QR3I
                 EFIELD(2,J) = EFIELD(2,J)-YY*QR3I
                 EFIELD(3,J) = EFIELD(3,J)-ZZ*QR3I
              ENDIF

              !       MARK J AS A DIPOLE CENTER
              IPOL(J) = 1


              ! END THE ENERGY CUT-OFF
           END IF

           !
           ! END JPR LOOP
           !
        ENDDO

        !       MARK I AS A DIPOLE CENTER
        IPOL(I) = 1

     ENDIF
     !
     ! END MAIN LOOP
     !
  ENDDO

  !
  !     INDUCED DIPOLE MOMENT
  !
  DO I = IFRSTA,NATOM
     IF(IPOL(I).EQ.1) THEN
        I1=ITC(IAC(I))
        UIND(1,I) = UIND(1,I) + ALP(I1)*EFIELD(1,I)
        UIND(2,I) = UIND(1,I) + ALP(I1)*EFIELD(2,I)
        UIND(3,I) = UIND(1,I) + ALP(I1)*EFIELD(3,I)
     ENDIF
  ENDDO

  RETURN
END SUBROUTINE DPFST2


!-----------------------------------------------------------------------
SUBROUTINE DPFST3(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     EPOL,ALP,DDT,EFIELD,EIND,IPOL &
                                !    &  ,IOFF
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES CONVERGED DIPOLE VIA THE SELF-CONSISTENT
  !     ITERATIVE PROCESS FOR THE FIRST DYNAMICS STEP
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability (passed in param.f90)
  !
  !     UIND - induced dipole for current iteration (ltm_pipf)
  !     DDT - induced dipole for previous iteration
  !     EFIELD - permenent electric field
  !     EIND - induced electric field
  !     TXWW - rank 2 polar tensor
  !     IPOL - tag to mark an atom as a dipole center
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use energym
  use stream
  use pipfm
#if KEY_PBOUND==1
  use pbound  
#endif
  implicit none
#if KEY_PBOUND==1
  real(chm_real) CORR  
#endif
  !
  real(chm_real) EPOL
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*)
  !
  INTEGER NITER, NCOUNT
  INTEGER I,J,JPR,NB,NPR,ITEMP,I1,J1
  real(chm_real) EFIELD(3,*)
  real(chm_real) DDT(3,*),EIND(3,*)
  real(chm_real) DMUNEW,DDIFF
  real(chm_real) E14M1,E14F,CGT2,CGT
  INTEGER IPOL(*)
  real(chm_real) XX,YY,ZZ,R1,R2,R3,QR3I,QR3J, &
       RR3,RR3XX,RR3YY,RR3ZZ
  real(chm_real) TXWW(6)
  !
  real(chm_real) U3,AU3,FLMD3,FLMD5
  !
  real(chm_real) RIJ2,PFCTOF2
  PFCTOF2=PFCTOF*PFCTOF
  !
  !     SET DIPOLE CENTER TAG TO ZERO
  !
  DO I = 1, NATOM
     IPOL(I) = 0
  END DO
  !
  !     ZERO INITIAL INDUCED DIPOLE
  !
  CALL GZERO(UIND,3,NATOM)

  NITER = 0
100 NITER = NITER+1
  CALL COORTR(DDT,UIND,3*NATOM)
  CALL GZERO(EFIELD,3,NATOM)
  CALL GZERO(EIND,3,NATOM)
  !
  NB = 0
  ITEMP = 0

  !
  !=======================================================================
  !   MAIN LOOP BEGIN
  !=======================================================================
  DO I=IFRSTA,NATOM
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR.GT.0) THEN

        I1=ITC(IAC(I))

        DO JPR=1,NPR
           NB=NB+1
           IF (JNB(NB).LT.0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNB(NB)
           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNB(NB)
           ENDIF

           J1=ITC(IAC(J))

           XX = X(I)-X(J)
           YY = Y(I)-Y(J)
           ZZ = Z(I)-Z(J)

#if KEY_PBOUND==1 /*pbound*/
           If(qBoun) then
              If(qCUBoun.or.qTOBoun) then
                 XX = BOXINV * XX
                 YY = BOYINV * YY
                 ZZ = BOZINV * ZZ
                 xx = xx - nint(xx)
                 yy = yy - nint(yy)
                 zz = zz - nint(zz)
!!$                 IF(XX.GT.  HALF) XX = XX - ONE
!!$                 IF(XX.LT. -HALF) XX = XX + ONE
!!$                 IF(YY.GT.  HALF) YY = YY - ONE
!!$                 IF(YY.LT. -HALF) YY = YY + ONE
!!$                 IF(ZZ.GT.  HALF) ZZ = ZZ - ONE
!!$                 IF(ZZ.LT. -HALF) ZZ = ZZ + ONE
                 If (qTOBoun) Then
                    CORR = HALF * AINT ( R75 * (ABS(XX) + &
                         ABS(YY) + &
                         ABS(ZZ)))
                    XX = XX - SIGN( CORR,  XX  )
                    YY = YY - SIGN( CORR,  YY  )
                    ZZ = ZZ - SIGN( CORR,  ZZ  )
                 Endif
                 XX = XSIZE * XX
                 YY = YSIZE * YY
                 ZZ = ZSIZE * ZZ
              Else
                 Call PBMove(XX, YY, ZZ)
              Endif
           Endif
#endif /*      (pbound)*/

           !
           !      APPLY AN ENERGY CUT-OFF
           !
           RIJ2 = XX*XX+YY*YY+ZZ*ZZ
           IF (RIJ2 .LT. PFCTOF2) THEN

              !
              !           R2 = 1.D+00/(XX*XX+YY*YY+ZZ*ZZ)
              !
              R2 = 1.D+00/RIJ2
              R1 = SQRT(R2)
              R3 = R1*R2

              !
              !      0TH-ORDER ELECTRIC FIELD
              !      FOLLOW EQ. (5) in CHEM. PHYS. LETT. 1990, 166, 180.
              !
              !               E^i = - sum(T^iA Q^A) and T^iA = - a/R^3 (a denotes a^i-a^j)
              !                a       A   a             a
              !

              !
              !       NO DAMPING (DEFAULT)
              !
              IF (NPDAMP .EQ. 0) THEN
                 QR3I = CG(I)*R3
                 QR3J = CG(J)*R3
                 !
                 !       OR APPLY A DAMPING FACTOR
                 !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 !
              ELSE IF (NPDAMP .EQ. 1) THEN

                 U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
                 AU3 = DPFAC*U3
                 FLMD3 = 1.0D0 - EXP(-AU3)
                 FLMD5 = 1.0D0 - (1.0D0+AU3)*EXP(-AU3)
                 QR3I = CG(I)*R3*FLMD3
                 QR3J = CG(J)*R3*FLMD3

              END IF

              IF (QPFEX .AND. JNB(NB).LT.0) THEN
                 ! OPTIONALLY EXCLUDE 1-4 POLARIZATION, DO NOTHING
              ELSE
                 EFIELD(1,I) = EFIELD(1,I)+XX*QR3J
                 EFIELD(2,I) = EFIELD(2,I)+YY*QR3J
                 EFIELD(3,I) = EFIELD(3,I)+ZZ*QR3J
                 EFIELD(1,J) = EFIELD(1,J)-XX*QR3I
                 EFIELD(2,J) = EFIELD(2,J)-YY*QR3I
                 EFIELD(3,J) = EFIELD(3,J)-ZZ*QR3I
              ENDIF
              !
              !      DIPOLE TENSOR
              !
              !         SIGN CONVENTION FLOLLOWS THE DEFINITION AS EQ. (6)
              !         DESCRIBED IN CHEM. PHYS. LETT. 1990, 166, 180.
              !
              !              T^ij  = d^i d^i (|Ri-Rj|^-1)
              !               ab      a   b
              !

              !
              !        NO DAMPING (DEFAULT)
              !
              IF (NPDAMP .EQ. 0) THEN
                 RR3 = 3.0D+00*R3*R2
                 RR3XX = RR3*XX
                 RR3YY = RR3*YY
                 RR3ZZ = RR3*ZZ
                 !
                 !        OR APPLY A DAMPING FACTOR
                 !        (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 !
              ELSE IF (NPDAMP .EQ. 1) THEN
                 RR3 = 3.0D+00*R3*R2*FLMD5
                 RR3XX = RR3*XX
                 RR3YY = RR3*YY
                 RR3ZZ = RR3*ZZ
                 R3 = R3 * FLMD3
              END IF


              !         ACCORDING TO THIS CONVENTION, WE HAVE:
              !
              !              T^ij  = - T^ji,   T^ij = T^ji , and T^ij = -T^ji
              !               a         a       ab     ab         abc     abc
              !
              !         AND SO ON (OPPSITE SIGN FOR ODD ORDER OF DIFFERENTIATION)
              !         WE MAP TXWWT TO A 2-D ARRAY, ONLY HAVE 6xNB TERMS.
              !         NOTE:  TXWW(1:6,NB) = TXWW(1:6,I,J) = TXWW(1:6,J,I)
              !
              IF (QPFEX .AND. JNB(NB).LT.0) THEN
                 ! OPTIONALLY EXCLUDE 1-4 POLARIZATION
                 TXWW(1) = 0.0D0
                 TXWW(2) = 0.0D0
                 TXWW(3) = 0.0D0
                 TXWW(4) = 0.0D0
                 TXWW(5) = 0.0D0
                 TXWW(6) = 0.0D0
              ELSE
                 TXWW(1) = RR3XX*XX-R3
                 TXWW(2) = RR3XX*YY
                 TXWW(3) = RR3YY*YY-R3
                 TXWW(4) = RR3XX*ZZ
                 TXWW(5) = RR3YY*ZZ
                 TXWW(6) = RR3ZZ*ZZ-R3
              ENDIF
              !
              !      COMPUTE INDUCED FIELD
              !      SINCE WE USE NONBONDED PAIR (I<J), WE NEED TO INCLUDE
              !      CONTRIBUTION TO J FOR I>J PAIR EXPLICITLY
              !
              !      NOTE:  TXWW(1:6,NB) = TXWW(1:6,I,J) = TXWW(1:6,J,I)
              !
              EIND(1,I) = EIND(1,I)+TXWW(1)*UIND(1,J) &
                   +TXWW(2)*UIND(2,J)+TXWW(4)*UIND(3,J)
              EIND(2,I) = EIND(2,I)+TXWW(2)*UIND(1,J) &
                   +TXWW(3)*UIND(2,J)+TXWW(5)*UIND(3,J)
              EIND(3,I) = EIND(3,I)+TXWW(4)*UIND(1,J) &
                   +TXWW(5)*UIND(2,J)+TXWW(6)*UIND(3,J)

              EIND(1,J) = EIND(1,J)+TXWW(1)*UIND(1,I) &
                   +TXWW(2)*UIND(2,I)+TXWW(4)*UIND(3,I)
              EIND(2,J) = EIND(2,J)+TXWW(2)*UIND(1,I) &
                   +TXWW(3)*UIND(2,I)+TXWW(5)*UIND(3,I)
              EIND(3,J) = EIND(3,J)+TXWW(4)*UIND(1,I) &
                   +TXWW(5)*UIND(2,I)+TXWW(6)*UIND(3,I)

              !       MARK J AS A DIPOLE CENTER
              IF (NITER .EQ. 1) IPOL(J) = 1

              ! END THE ENERGY CUT-OFF
           END IF

           !
           ! END JPR LOOP
           !
        ENDDO

        !       MARK I AS A DIPOLE CENTER
        IF (NITER .EQ. 1) IPOL(I) = 1

     ENDIF
     !
     ! END MAIN LOOP
     !
  ENDDO

  !
  !      NEW INDUCED DIPOLES
  !      SIGN FOLLOWS EQ. (4) in CHEM. PHYS. LETT. 1990, 166, 180.
  !
  NCOUNT = 0
  DMUNEW = 0.0D+00
  DO I=IFRSTA,NATOM
     IF(IPOL(I).EQ.1) THEN
        NCOUNT = NCOUNT + 1
        I1=ITC(IAC(I))
        UIND(1,I) = ALP(I1)*(EFIELD(1,I)+EIND(1,I))
        UIND(2,I) = ALP(I1)*(EFIELD(2,I)+EIND(2,I))
        UIND(3,I) = ALP(I1)*(EFIELD(3,I)+EIND(3,I))
        DMUNEW = DMUNEW+(UIND(1,I)-DDT(1,I))**2+ &
             (UIND(2,I)-DDT(2,I))**2+ &
             (UIND(3,I)-DDT(3,I))**2
     ENDIF
  ENDDO


  ! DIPOLE MOMENT UNIT, CONVERT ATOMIC CHARGE AND aNGSTROM TO DEBYE
  !
  DDIFF= 4.8028132D+00*DSQRT(DMUNEW/DBLE(NCOUNT))

  IF(NITER.GT.15) THEN
     WRITE(OUTU,*) 'CONVERGENCE PROBLEMS in EPIPF:',NITER
     WRITE(OUTU,10) NITER, DDIFF, DTHRES
10   FORMAT(1X,'DPFST3>  ITER', I3, ':', E15.5, E15.5)
     if(niter.gt.ITRMX.AND.DDIFF.GT.10.D+00*DTHRES) THEN
        EPOL = 99.99D+00
        WRITE(OUTU,*) 'CONVERGENCE FAILED - GIVE A LARGE ENERGY'
        RETURN
     ENDIF
  ENDIF
  !
  ! CONVERGE BOTH DIPOLE AND ENERGY
  !
  IF(DDIFF.GT.DTHRES) GOTO 100

  !
  !     FINAL POLARIZATION ENERGY
  !
  EPOL = 0.0D+00
  DO I = IFRSTA,NATOM
     IF(IPOL(I).EQ.1) THEN
        EPOL = EPOL+UIND(1,I)*EFIELD(1,I)+UIND(2,I) &
             *EFIELD(2,I)+UIND(3,I)*EFIELD(3,I)
     ENDIF
  ENDDO
  !
  ! ENERGY, CONVERT SI TO KCAL/MOL
  !
  EPOL = -0.5D0*CCELEC*EPOL
  !
  IF (PRNLEV .GE. 7) THEN
     WRITE(OUTU,*) 'DPFST3>   Polarization Energy (EPOL):', EPOL
  ENDIF

  RETURN
END SUBROUTINE DPFST3

!----------------------------------------------------------------------
SUBROUTINE CPDPIMG(IMATTRX,UIND)
  !----------------------------------------------------------------------
  ! THIS ROUTINE COPIES DIPOLE FROM MAIN SET TO IMAGE ATOMS
  ! FOR USE IN CHARGE DYNAMICS
  !
  use chm_kinds
  use dimens_fcm
  use image
  use psf
  implicit none
  !
  INTEGER IMATTRX(*)
  real(chm_real) UIND(3,*)
  !
  INTEGER I,J
  IF (NATIM.GT.NATOM) THEN
     DO I=NATOM+1,NATIM
        J=IMATTRX(I)
        ! - debug
        !          write(*,*) '--- P-I mapping: ', J, I

        UIND(1,I)=UIND(1,J)
        UIND(2,I)=UIND(2,J)
        UIND(3,I)=UIND(3,J)
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE CPDPIMG

!----------------------------------------------------------------------
SUBROUTINE DPTRANSI(DUIND,NATOM,NATIM,IMATTR)
  !----------------------------------------------------------------------
  ! THIS ROUTINE TRANSFER THE DIPOLE FORCE ON IMAGE ATOMS TO
  ! THE REAL ATOMS
  use chm_kinds
  use number
  implicit none
  real(chm_real) DUIND(3,*)
  INTEGER NATOM,NATIM,IMATTR(*)
  INTEGER I,J

  DO I=NATOM+1,NATIM
     J=IMATTR(I)
     DUIND(1,J) = DUIND(1,J) + DUIND(1,I)
     DUIND(2,J) = DUIND(2,J) + DUIND(2,I)
     DUIND(3,J) = DUIND(3,J) + DUIND(3,I)
     DUIND(1,I)=ZERO
     DUIND(2,I)=ZERO
     DUIND(3,I)=ZERO
  ENDDO

  RETURN
END SUBROUTINE DPTRANSI

!----------------------------------------------------------------------
SUBROUTINE DPTRANSA(ESAV,NPFPR,NPFIM)
  !----------------------------------------------------------------------
  ! THIS ROUTINE TRANSFER THE ELECTRIC FIELD ON IMAGE ATOMS TO
  ! THE REAL ATOMS
  use chm_kinds
  use chm_types
  !--yw--!
  use bases_fcm
  use dimens_fcm
  use number
  use inbnd
  use image
  implicit none
  real(chm_real) ESAV(3,*)
  INTEGER NPFPR, NPFIM

  CALL DPTRANSI(ESAV,NPFPR,NPFIM,BIMAG%IMATTR)
  RETURN
END SUBROUTINE DPTRANSA
#endif /* (pipf_main) */
subroutine pfdyn_dummy()
  return
end subroutine pfdyn_dummy
