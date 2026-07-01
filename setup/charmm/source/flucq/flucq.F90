module flucqm
  use chm_kinds
  use dimens_fcm
  implicit none
  ! The files in this directory implement a fluctuating charge force field
  ! within CHARMM. This force field is based on the method developed by
  ! Rick, Stuart and Berne (Rick et. al., J. Chem. Phys. 101 (7) 1994 p6141)
  ! for molecular dynamics, and extended for hybrid QM/MM simulations
  ! (Bryce et. al., Chem. Phys. Lett. 279 1997, p367)
  !
  ! Fluctuating charge code is compiled into CHARMM by specifying the
  ! FLUCQ keyword in pref.dat, and simulation is controlled using the
  ! FLUCQ keyword in a CHARMM input file

  integer,allocatable,dimension(:),save,public :: fqseln
  real(chm_real),allocatable,dimension(:),save,public :: fqjr,fqcfor

#if KEY_FLUCQ==1 /*flucq*/
contains

  SUBROUTINE FQINIT(COMLYN,COMLEN)
    !
    !     Processes the FLUCQ command from a CHARMM input file
    !     Author: Ben Webb, 2000
    !
    use exfunc
    use stream
    use string
    use number
    use psf
    use flucq
    use parallel
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    real(chm_real) DT,ZDT,TQDES,REFE
    INTEGER I
    LOGICAL PRIRES

    IF (INDXA(COMLYN,COMLEN,'ON').GT.0) THEN
       IF (QFLUC) THEN
          IF (PRNLEV.GE.5) WRITE(OUTU,*) &
               'FQINIT> Restarting fluctuating charge code'
          CALL FQEND
          QFLUC=.FALSE.
       ELSE
          IF (PRNLEV.GE.5) WRITE(OUTU,*) &
               'FQINIT> Initialising fluctuating charge code'
       ENDIF
       CALL FQSTRT(COMLYN,COMLEN)
       QFLUC = .TRUE.
    ELSE IF (INDXA(COMLYN,COMLEN,'OFF').GT.0) THEN
       IF (PRNLEV.GE.5) WRITE(OUTU,*) 'FQINIT> Fluctuating charge ', &
            'code will now NOT be called'
       IF (QFLUC) CALL FQEND
       QFLUC = .FALSE.
    ELSE IF (QFLUC) THEN
       IF (INDXA(COMLYN,COMLEN,'EXAC').GT.0) THEN
          CALL FQEXAC(COMLYN,COMLEN,.TRUE.)
       ELSE IF (INDXA(COMLYN,COMLEN,'PRIN').GT.0) THEN
#if KEY_PARALLEL==1
          IF (MYNOD.EQ.0) THEN
#endif 
             DO I=1,NATOM
                WRITE(OUTU,30) I,CG(I),FQCFOR(I)
             ENDDO
30           FORMAT(' FQINIT> Atom ',I6,' charge ', &
                  F9.3,' force ',F12.3)
#if KEY_PARALLEL==1
          ENDIF
          CALL PSYNC()
#endif 
       ELSE IF (INDXA(COMLYN,COMLEN,'REFE').GT.0) THEN
          CALL FQSETR(COMLYN,COMLEN)
       ELSE
          CALL WRNDIE(0,'<FQINIT>', &
               'Unrecognised FLUCQ sub-command')
       ENDIF
    ELSE
       CALL WRNDIE(0,'<FQINIT>', &
            'FLUCQ subcommand invalid - code not yet initialised')
    ENDIF

    RETURN
  END SUBROUTINE FQINIT

  SUBROUTINE FQSETR(COMLYN,COMLEN)
    !
    !     Processes the FLUCQ REFER subcommand, to set
    !     reference polarisation energy
    !     Author: Ben Webb, 2000
    !
    use exfunc
    use stream
    use string
    use number
    use psf
    use energym
    use flucq
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER I
    real(chm_real) OLDREF
    LOGICAL QETTMP(LENENT)
    OLDREF=FQREFE
    ! Back up arrays of energy terms
    DO I=1,LENENT
       QETTMP(I)=QETERM(I)
    ENDDO
    IF (INDXA(COMLYN,COMLEN,'GAS').GT.0) THEN
       IF (PRNLEV.GE.5) WRITE(OUTU,*) &
            'FQSETR> Using gas phase polarisation ', &
            'as reference energy'
       FQREFE=ZERO
       DO I=1,LENENT
          QETERM(I)=.FALSE.
       ENDDO
       QETERM(BOND)=.TRUE.
       QETERM(ANGLE)=.TRUE.
       QETERM(UREYB)=.TRUE.
       QETERM(DIHE)=.TRUE.
       QETERM(IMDIHE)=.TRUE.
#if KEY_CMAP==1
       QETERM(CMAP)=.TRUE.        
#endif
       QETERM(FQPOL)=.TRUE.
       CALL FQEXAC(COMLYN,COMLEN,.TRUE.)
       FQREFE=ETERM(FQPOL)
       IF (PRNLEV.GE.5) WRITE(OUTU,10) FQREFE
10     FORMAT(' FQSETR> Gas phase polarisation energy ',F15.5)
    ELSE IF (INDXA(COMLYN,COMLEN,'SOLV').GT.0) THEN
       FQREFE=ZERO
       DO I=1,LENENT
          QETERM(I)=.TRUE.
       ENDDO
       QETERM(QMEL)=.FALSE.
       QETERM(QMVDW)=.FALSE.
       CALL FQEXAC(COMLYN,COMLEN,.TRUE.)
       FQREFE=ETERM(FQPOL)
       IF (PRNLEV.GE.5) WRITE(OUTU,20) FQREFE
20     FORMAT(' FQSETR> Solvent polarisation energy ',F15.5)
    ELSE IF (INDXA(COMLYN,COMLEN,'CURR').GT.0) THEN
       FQREFE=ETERM(FQPOL)+FQREFE
       IF (PRNLEV.GE.5) WRITE(OUTU,30) FQREFE
    ELSE
       FQREFE=GTRMF(COMLYN,COMLEN,'ENER',FQREFE)
       IF (PRNLEV.GE.5) WRITE(OUTU,30) FQREFE
    ENDIF
    IF (FQREFE.EQ.OLDREF) THEN
       CALL WRNDIE(0,'<FQSETR>', &
            'Reference energy unchanged by FLUCQ REFE command')
    ENDIF
30  FORMAT(' FQSETR> Reference polarisation ', &
         'energy set to ',F15.5)
    ! Restore arrays of energy terms
    DO I=1,LENENT
       QETERM(I)=QETTMP(I)
    ENDDO
    RETURN
  END SUBROUTINE FQSETR

  SUBROUTINE FQDPAR(COMLYN,COMLEN)
    !
    !     Parses a dynamics command for FlucQ options
    !     Author: Ben Webb, 2000
    !
    use exfunc
    use number
    use stream
    use string
    use flucq
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    real(chm_real) TENM7
    TENM7=TENM8*TEN

    FQSCAL=GTRMI(COMLYN,COMLEN,'FQSC',0)
    FQTEMP=GTRMF(COMLYN,COMLEN,'FQTE',ZERO)
    FQTCOU=GTRMF(COMLYN,COMLEN,'FQTC',ZERO)
    FQNHM=GTRMF(COMLYN,COMLEN,'FQMA',ZERO)
    FQNHTL=GTRMF(COMLYN,COMLEN,'FQTO',TENM7)
    FQNHMX=GTRMI(COMLYN,COMLEN,'FQIT',100)
    FQUNIT=GTRMI(COMLYN,COMLEN,'FQUN',-1)

    RETURN
  END SUBROUTINE FQDPAR

  SUBROUTINE FQSTRT(COMLYN,COMLEN)
    !
    !     Allocates and initialises all FlucQ arrays and data
    !     Author: Ben Webb, 2000
    !

#if KEY_CADPAC == 1
    use cadpac_data, only: ncadpc
#endif
    
    use memory
    use exfunc
    use select
    use stream
    use string
    use coord
    use psf
    use image
    use param
#if KEY_GAMESS==1 || KEY_CADPAC==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
    use gamess_fcm
#endif 
#if KEY_MNDO97==1
    use mndo97
#endif 
#if KEY_QUANTUM==1
    use quantm
#endif 
#if KEY_SQUANTM==1
    use squantm
#endif 
    use code
    use number
    use flucq

    implicit none
    
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN

    INTEGER I
    !
    call chmalloc('flucq.src','FQSTRT','FQSELN',NATOM,intg=FQSELN)
    IF (NTRANS.GT.0) THEN
       ! Yuck! Would be nice to just dimension this NATIM, but unfortunately
       ! NATIM can change during a dynamics run
       call chmalloc('flucq.src','FQSTRT','FQCFOR',MAXAIM,crl=FQCFOR)
    ELSE
       call chmalloc('flucq.src','FQSTRT','FQCFOR',NATOM,crl=FQCFOR)
    ENDIF
    call chmalloc('flucq.src','FQSTRT','FQOLDQ',NATOM,crl=FQOLDQ)
    call chmalloc('flucq.src','FQSTRT','FQNEWQ',NATOM,crl=FQNEWQ)
    call chmalloc('flucq.src','FQSTRT','FQJR',NCB,crl=FQJR)
    FQREFE=ZERO
    CALL FQZERO
    CALL FQSETT(ZERO)
    CALL SELCTA(COMLYN,COMLEN,FQSELN,X,Y,Z,WMAIN,.FALSE.)
    CALL FQPRSE(FQSELN)
    ! Update the CODES lists if necessary
    IF (MUSTUP) CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE.,.TRUE., &
         .TRUE.,.TRUE.,.TRUE., &
         0,[zero],[zero],[zero],[zero],[zero],[zero])
    CALL FQJCA(FQJR)
    FQGRUP=(INDXA(COMLYN,COMLEN,'GROU').GT.0)
    FQFIXD=(INDXA(COMLYN,COMLEN,'NOFI').EQ.0)
    IF (PRNLEV.GE.5) THEN
       IF (FQFIXD) THEN
          WRITE(OUTU,*) &
               'FQSTRT> Assuming that all FQ bond lengths are fixed'
       ELSE
          WRITE(OUTU,*) &
               'FQSTRT> Intramolecular interactions recalculation on'
       ENDIF
    ENDIF

    IF (FQGRUP) THEN
       CALL FQDEGF(FQSELN,NGRP,IGPBS,NATOM,FQCDGF)
    ELSE
       CALL FQDEGF(FQSELN,NRES,IBASE,NATOM,FQCDGF)
    ENDIF

#if KEY_GAMESS==1 || KEY_CADPAC==1 || KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
#if KEY_QUANTUM==1
    IF (NATQM.GT.0) THEN
       IF (PRNLEV.GE.5) WRITE(OUTU,20) NATQM,'QUANTUM'
    ENDIF
#elif KEY_GAMESS==1
    IF (NGAMES.GT.0) THEN
       IF (PRNLEV.GE.5) WRITE(OUTU,20) NGAMES,'GAMESS'
    ENDIF
#elif KEY_CADPAC==1
    IF (NCADPC.GT.0) THEN
       IF (PRNLEV.GE.5) WRITE(OUTU,20) NCADPC,'CADPAC'
    ENDIF
#elif KEY_MNDO97==1
    IF (NUMAT.GT.0) THEN
       IF (PRNLEV.GE.5) WRITE(OUTU,20) NUMAT,'MNDO97'
    ENDIF
#elif KEY_SQUANTM==1
    IF (NATQM(1).GT.0) THEN
       IF (PRNLEV.GE.5) WRITE(OUTU,20) NATQM(1),'SQUANTM'
    ENDIF
#endif 
20  FORMAT(' FQSTRT> FQ selection will interact with ',I6, &
         ' QM (',A,') atoms')
#endif 
    RETURN
  END SUBROUTINE FQSTRT

  SUBROUTINE FQDEGF(FQSELN,NRES,IBASE,NATOM,FQCDGF)
    !
    !     Calculates number of charge degrees of freedom
    !
    !     FQSELN: selection of FQ atoms
    !     NRES:   number of residues (or groups)
    !     IBASE:  index array over residues (or groups)
    !     NATOM:  number of atoms
    !     FQCDGF: number of charge degrees of freedom, to be set on output
    !
    !     Author: Ben Webb, 2000
    !
    use exfunc
    use stream
    INTEGER FQSELN(*),NRES,IBASE(*),NATOM
    INTEGER FQCDGF
    INTEGER I,J,NFQ,NFQR
    LOGICAL INRES
    ! degrees of freedom=no. of FQ atoms-no. of restraints (residues)
    NFQ=0
    NFQR=0
    DO I=1,NRES
       INRES=.FALSE.
       DO J=IBASE(I)+1,IBASE(I+1)
          IF (FQSELN(J).NE.0) THEN
             NFQ=NFQ+1
             INRES=.TRUE.
          ENDIF
       ENDDO
       IF (INRES) NFQR=NFQR+1
    ENDDO
    IF ((NFQ-NFQR).LE.0) THEN
       CALL WRNDIE(-4,'<FQDEGF>', &
            'Number of charge degrees of freedom zero or negative!')
       FQCDGF=0
    ELSE
       FQCDGF=NFQ-NFQR
       IF (PRNLEV.GE.5) WRITE(OUTU,10) NFQ-NFQR
10     FORMAT(' FQDEGF> System has ',I8, &
            ' degrees of charge freedom')
    ENDIF
    RETURN
  END SUBROUTINE FQDEGF


  SUBROUTINE FQPRSE(FQSELN)
    !
    !     Removes from the FQ selection (FQSELN) atoms
    !     that are QM atoms, or are without defined FQ parameters
    !
    !     Author: Ben Webb, 2000
    !
    use exfunc
    use param
    use stream
    use psf
#if KEY_QUANTUM==1
    use quantm
#endif 
#if KEY_GAMESS==1 || KEY_CADPAC==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
    use gamess_fcm
#endif 
#if KEY_MNDO97==1
    use mndo97
#endif 
#if KEY_SQUANTM==1
    use squantm
#endif 
    INTEGER FQSELN(*)
    INTEGER I,NUMPRU,NUMFQ
    NUMFQ=0
    DO I=1,NATOM
       IF (FQSELN(I).NE.0) NUMFQ=NUMFQ+1
    ENDDO
    NUMPRU=0
    DO I=1,NATOM
       IF (FQSELN(I).NE.0.AND.FQZETA(IAC(I)).EQ.0.0d0) THEN
          NUMPRU=NUMPRU+1
          FQSELN(I)=0
       ENDIF
    ENDDO
    IF (NUMPRU.GT.0.AND.PRNLEV.GE.5) WRITE(OUTU,10) NUMPRU
10  FORMAT(' FQPRSE> ',I8,' atoms have been removed from FQ ', &
         'selection ',/, &
         '         (no FQ parameters defined for these atoms)')
    NUMFQ=NUMFQ-NUMPRU
#if KEY_QUANTUM==1
    NUMPRU=0
    IF (NATQM.GT.0) THEN
       DO I=1,NATOM
          IF (FQSELN(I).NE.0.AND.QATLAB(I).GE.0) THEN
             NUMPRU=NUMPRU+1
             FQSELN(I)=0
          ENDIF
       ENDDO
    ENDIF
    IF (NUMPRU.GT.0.AND.PRNLEV.GE.5) WRITE(OUTU,21) NUMPRU
21  FORMAT(' FQPRSE> ',I8,' atoms have been removed from FQ ', &
         'selection ',/, &
         '         (these are QUANTUM QM or link atoms)')
    NUMFQ=NUMFQ-NUMPRU
#endif 
#if KEY_GAMESS==1 || KEY_CADPAC==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
    NUMPRU=0
    DO I=1,NATOM
       IF (FQSELN(I).NE.0.AND.((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))) &
            THEN
          NUMPRU=NUMPRU+1
          FQSELN(I)=0
       ENDIF
    ENDDO
    IF (NUMPRU.GT.0.AND.PRNLEV.GE.5) WRITE(OUTU,20) NUMPRU
20  FORMAT(' FQPRSE> ',I8,' atoms have been removed from FQ ', &
         'selection ',/, &
         ' (these are GAMESS/CADPAC/MNDO97/SQUANTM QM or link atoms)')
    NUMFQ=NUMFQ-NUMPRU
#endif 
    IF (NUMFQ.LE.0) THEN
       CALL WRNDIE(-4,'<FQPRSE>','FLUCQ selection is empty!')
    ELSE
       IF (PRNLEV.GE.5) WRITE(OUTU,30) NUMFQ
30     FORMAT(' FQPRSE> A total of ',I8, &
            ' atoms will be modelled by FQ')
    ENDIF
    RETURN
  END SUBROUTINE FQPRSE

  SUBROUTINE FQJCA(FQJR)
    !
    !     Fills the FQJR and FQJZ arrays of Slater overlap interactions
    !
    !     FQJR: array of atom-atom Slater overlap bond terms
    !           (filled on output with the values at equilibrium bond lengths)
    !
    !     Author: Ben Webb, 2000
    !
    use exfunc
    use stream
    use number
    use consta
    use psf
    use code
    use param
    real(chm_real) FQJR(*)

    INTEGER I,IND,ITYPE,JTYPE
    !      real(chm_real) FQINT
    FQJZ(1:MAXATC) = ZERO

    DO I=1,NATC
       IF (FQPRIN(I).EQ.1) THEN
          FQJZ(I) = 5.0d0*FQZETA(I)/8.0d0/4184.0*8.31451* &
               (408.69)**2/BOHRR
          IF (PRNLEV.GE.6) THEN
             WRITE(OUTU,10) 'FQJZ',I,FQJZ(I)
          ENDIF
       ELSE IF (FQPRIN(I).EQ.2) THEN
          FQJZ(I) = 93.0d0*FQZETA(I)/256.0d0/4184.0* &
               8.31451*(408.69)**2/BOHRR
          IF (PRNLEV.GE.6) THEN
             WRITE(OUTU,10) 'FQJZ',I,FQJZ(I)
          ENDIF
       ENDIF
    ENDDO
    FQJR(1:NCB) = ZERO
    DO I=1,NBOND
       IND=ICB(I)
       IF (FQJR(IND).EQ.ZERO) THEN
          ITYPE=IAC(IB(I))
          JTYPE=IAC(JB(I))
          FQJR(IND)=FQINT(CBB(IND),ITYPE,JTYPE)
          IF (PRNLEV.GE.6) THEN
             WRITE(OUTU,10) 'FQJR',I,FQJR(I)
          ENDIF
       ENDIF
    ENDDO
10  FORMAT(' FQJCA> ',A4,'(',I4,') = ',F12.4)
    RETURN
  END SUBROUTINE FQJCA

  real(chm_real) FUNCTION FQINT(R,ITYPE,JTYPE)
    !
    !     Calculates the Slater orbital overlap term FQJR (returned) for
    !     atom types ITYPE and JTYPE, of separation R
    !     Author: Ben Webb, 2000
    use exfunc
    use stream
    use consta
    use number
    use param
    real(chm_real) R
    INTEGER ITYPE,JTYPE
    real(chm_real) INTGRND,SUM,RK,TZE1,TZE2,DK,AA1,AA2
    real(chm_real) FK1,FK2,FA,FB
    INTEGER K,KMAX,NINT
    IF (FQZETA(ITYPE).EQ.ZERO.OR.FQZETA(JTYPE).EQ.ZERO) THEN
       FQINT=ZERO
       RETURN
    ENDIF
    TZE1=FQZETA(ITYPE)/BOHRR
    TZE2=FQZETA(JTYPE)/BOHRR
    KMAX=8
    NINT=2000
    DK=FLOAT(KMAX)/FLOAT(NINT)
    SUM=HALF*DK
    DO K=1,NINT
       RK=K*DK
       AA1=ONE+(RK/(TWO*TZE1))**2
       AA2=ONE+(RK/(TWO*TZE2))**2
       FK1=ONE/AA1**2
       FK2=ONE/AA2**2
       IF (FQPRIN(ITYPE).EQ.1) THEN
          FA=FK1
       ELSE IF (FQPRIN(ITYPE).EQ.2) THEN
          FA=FK1-THREE*RK**2*FK1**1.5/(FOUR*TZE1**2) &
               +RK**4*FK1**2/(8.d0*TZE1**4)
       ENDIF
       IF (FQPRIN(JTYPE).EQ.1) THEN
          FB=FK2
       ELSE IF (FQPRIN(JTYPE).EQ.2) THEN
          FB=FK2-THREE*RK**2*FK2**1.5/(FOUR*TZE2**2) &
               +RK**4*FK2**2/(8.d0*TZE2**4)
       ENDIF
       INTGRND=SIN(RK*R)/(RK*R)*FA*FB
       SUM=SUM+INTGRND*DK
    ENDDO
    FQINT=SUM*TWO/PI/4184.0*8.31451*(408.69)**2
    RETURN
  END FUNCTION FQINT

  SUBROUTINE FQEND
    !
    !     Cleans up after FlucQ; frees all arrays
    !
    !     Author: Ben Webb, 2000
    !
    use memory
    use exfunc
    use stream
    use psf
    use image
    use param
    use flucq

    IF (PRNLEV.GE.5) WRITE(OUTU,*) 'FQEND> FlucQ ending...'
    call chmdealloc('flucq.src','FQEND','FQSELN',NATOM,intg=FQSELN)
    IF (NTRANS.GT.0) THEN
       call chmdealloc('flucq.src','FQEND','FQCFOR',MAXAIM,crl=FQCFOR)
    ELSE
       call chmdealloc('flucq.src','FQEND','FQCFOR',NATOM,crl=FQCFOR)
    ENDIF
    call chmdealloc('flucq.src','FQEND','FQOLDQ',NATOM,crl=FQOLDQ)
    call chmdealloc('flucq.src','FQEND','FQNEWQ',NATOM,crl=FQNEWQ)
    call chmdealloc('flucq.src','FQEND','FQJR',NCB,crl=FQJR)
    QFLUC=.FALSE.
    RETURN
  END SUBROUTINE FQEND
  !
#endif /* (flucq)*/
END MODULE FLUCQM

