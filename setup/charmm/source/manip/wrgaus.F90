module wrgaus_mod
  use chm_kinds
  use dimens_fcm
  implicit none

contains

  SUBROUTINE WRGAUS(IUNIT,IIN,JIN,KIN,X,Y,Z,LENIC, &
       IAR,JAR,KAR,LAR,TAR,RVAL,AVAL,DVAL,IUSED,NEWZMT)
    !
    !     THIS ROUTINE CREATE THE COORDINATES FOR THE FIRST THREE ATOMS
    !     OR THOSE SPECIFIED IN THE SEED COMMAND LINE.
    !
    !     By Bernard R. Brooks    1982
    !
    !     THE ROUTINE IS EXTENDED TO CREATE MIXED CARTESIAN AND Z-MATRIX
    !     GAUSSIAN INPUT FOR THE PURPOSE OF PARTIAL GEOMETRY OPTIMIZATION
    !     OF TWO INTERACTING MOIETIES. CARTESIAN COORDINATE REPRESENT FIRST
    !     FIXED FRAGMENT. ITS COORDINATES ARE DEFINED IN CHARMM AS HAVING 
    !     NO Z-MATRIX COORDINATES, E.G. CYTOSINE MOLECULE. SECOND FIXED 
    !     FRAGMENT HAS TO BE DEFINED IN CHARMM VIA Z-MATRIX COORDINATES 
    !     IN RESPECT TO THE FIRST FRAGMENT, E.G. WATER MOLECULE. THIS 
    !     DEFINITION ALLOWS PARTIAL OPTIMIZATION OF MUTUAL ORIENTATION OF 
    !     THE INTERACTING FRAGMENTS. THIS ROUTINE GENERATES GAUSSIAN 
    !     INPUT FILE READY FOR QM CALCULATION AFTER MINIMUM MANUAL 
    !     CORRECTIONS BY SELECTING WHICH INTERNAL COORDINATES SHOULD BE
    !     OPTIMIZED AND WHICH FIXED.
    !
    !     SELECTION BETWEEN PLAIN Z-MATRIX (ORIGINAL MODE) AND MIXED 
    !     COORDINATE (NEW MODE) OUTPUT FORMATS IS CONTROLLED BY KEYWORD
    !     "ZMIX". PRESENCE OF THIS KEYWORD INVOKES NEW MODE, OTHERWISE 
    !     ORIGINAL MODE WILL BE EXECUTED.
    !
    !     Victor M. Anisimov      2003
    !
    use chm_kinds
    use number
    use dimens_fcm
    use consta
    use psf
    use param
    use stream
    use intcor2,only:geticv
    implicit none
    !
    INTEGER IUNIT,IIN,JIN,KIN
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER LENIC
    INTEGER IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    real(chm_real) RVAL(*),AVAL(*),DVAL(*)
    INTEGER IUSED(*)
    LOGICAL NEWZMT
    !
    real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
    INTEGER INC,KK,II,I,J,K,L,ITEMP
    INTEGER NRVAL,NAVAL,NDVAL,NUSED
    LOGICAL RETRY,JUNK,UNKN,OK
    INTEGER CHANGE
    CHARACTER(len=80) FMT
    CHARACTER(len=4) LETTER
    LOGICAL WATER
    !
    IF(OUTU == IUNIT .AND. PRNLEV < 3) RETURN
    IF(OUTU /= IUNIT .AND. IOLEV < 0) RETURN
    !
    OK=.TRUE.
    DO I=1,NATOM
       IUSED(I)=0
       IF(ABS(X(I)) >= ANUM) OK=.FALSE.
       IF(ABS(Y(I)) >= ANUM) OK=.FALSE.
       IF(ABS(Z(I)) >= ANUM) OK=.FALSE.
    ENDDO
    !
    ! Print Gaussian header
    !
    WRITE(IUNIT,"(A//A//A)")  &
         '# MP2/6-31G* Opt=Z-matrix', &
         'Partial geometry optimization', &
         '  0  1'
    !
    IF(NEWZMT) THEN
       !
       ! Check to see the last atoms are swm4 water and two dummy atoms.
       !   This situation is handled as a special case and as a part of
       !   the parameter development procedure. Ref. to parameter development
       !   guidelines for details about this procedure.
       !
       IF(NATOM > 6.AND. &
            ATYPE(NATOM-5) == 'OH2' .AND. ATYPE(NATOM-4) == 'OM' .AND. &
            ATYPE(NATOM-3) == 'H1' .AND. ATYPE(NATOM-2) == 'H2' .AND. &
            ATYPE(NATOM-1) == 'DUM' .AND. ATYPE(NATOM) == 'DUM') THEN
          WATER=.TRUE.
       ELSE
          WATER=.FALSE.
       ENDIF
       !
       ! Print Cartesian coordinates of atoms not defined in IC records (by L index)
       !
       DO I=1,NATOM-LENIC
          WRITE(IUNIT,"(A5,3X,3F12.6)") ATYPE(I),X(I),Y(I),Z(I)
          IUSED(I)=I
       ENDDO
       NUSED=NATOM-LENIC
    ELSE
       !
       ! Label IIN,JIN,KIN atomic indexes as "used"
       !
       IUSED(IIN)=1
       IUSED(JIN)=2
       IUSED(KIN)=3
       NUSED=3
       !
       ! Print First atom of canonical Z-matrix
       !
       FMT='(A5)'
       WRITE(IUNIT,FMT) ATYPE(IIN)
       !
       ! Print Second atom of canonical Z-matrix
       !
       FMT='(A5,I5,''  R'',I1)'
       WRITE(IUNIT,FMT) ATYPE(JIN),1,1
       !
       ! Print Third atom of canonical Z-matrix
       !
       FMT='(A5,I5,''  R'',I1,I5,''  A'',I1)'
       WRITE(IUNIT,FMT) ATYPE(KIN),2,2,1,1
    ENDIF
    !
    CALL GETICV(IIN,JIN,KIN,0,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
    NRVAL=2
    RVAL(1)=RIJ
    NAVAL=1
    AVAL(1)=TIJK
    CALL GETICV(JIN,KIN,0,0,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
    RVAL(2)=RIJ
    NDVAL=0
    !
    !     THIS ROUTINE CONSTRUCTS CARTESIAN COORDINATES FROM THE INTERNAL
    !     COORDINATES
    !
    INC=1
    KK=0
    !
    CHANGE=2
    RETRY=.TRUE.
    DO WHILE(CHANGE > 0.AND.RETRY)
       CHANGE=CHANGE-1
       RETRY=.FALSE.
       DO II=1,LENIC
          KK=KK+INC
          !
          ! Find atom whose coordinates are to be generated (atom l).
          !
          I=IAR(KK)
          J=JAR(KK)
          K=KAR(KK)
          L=LAR(KK)
          !
          IF(INC < 0) THEN
             ITEMP=I
             I=L
             L=ITEMP
             IF(.NOT.TAR(KK)) THEN
                ITEMP=J
                J=K
                K=ITEMP
             ENDIF
          ENDIF
          !
          ! See if coordinates are already known
          !
          IF(I <= 0) THEN
             CONTINUE
          ELSE IF(J <= 0) THEN
             CONTINUE
          ELSE IF(K <= 0) THEN
             CONTINUE
          ELSE IF(L <= 0) THEN
             CONTINUE
          ELSE IF(I == K) THEN
             CONTINUE
          ELSE IF(I == J) THEN
             CONTINUE
          ELSE IF(K == J) THEN
             CONTINUE
          ELSE
             !
             ! Check to see if all antecedents are known
             !
             UNKN=IUSED(L) == 0
             JUNK=(IUSED(I) == 0.OR.IUSED(J) == 0.OR.IUSED(K) == 0)
             RETRY=RETRY.OR.JUNK.OR.UNKN
             !
             ! Skip Lone Pair type atoms
             !
             IF (UNKN.AND..NOT.JUNK.AND.ATC(IAC(L)) /= 'LP') THEN
                !
                ! Set geometrical parameters
                !
                CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL, &
                     X,Y,Z)
                !
                NRVAL=NRVAL+1
                NAVAL=NAVAL+1
                NDVAL=NDVAL+1
                NUSED=NUSED+1
                RVAL(NRVAL)=RKL
                AVAL(NAVAL)=TJKL
                DVAL(NDVAL)=PIJKL
                IUSED(L)=NUSED
                !
                FMT='(A5,I5,''  R'',I1,I5,''  A'',I1,I5,''  D'',I1)'
                !
                IF(NRVAL >= 10)  FMT(15:18)='2,I4'
                IF(NRVAL >= 100) FMT(15:18)='3,I3'
                IF(NAVAL >= 10)  FMT(27:30)='2,I4'
                IF(NAVAL >= 100) FMT(27:30)='3,I3'
                IF(NDVAL >= 10)  FMT(39:39)='2'
                IF(NDVAL >= 100) FMT(39:39)='3'
                !
                ! Replace Gaussian unsupported labels by appropriate ones
                !
                LETTER = ATYPE(L)
                IF(LETTER == 'DUM') THEN
                   !                    Dummy atom
                   LETTER='X'
                ELSEIF(ATC(IAC(L)) == 'OW') THEN
                   !                    Oxygen; water molecule
                   LETTER='O'
                ELSEIF(ATC(IAC(L)) == 'HW') THEN
                   !                    Hydrogen; water molecule
                   LETTER='H'
                ELSEIF(INDEX(ATC(IAC(L)),'HN') > 0) THEN
                   LETTER='H'
                ELSEIF(INDEX(ATC(IAC(L)),'HO') > 0) THEN
                   LETTER='H'
                ELSEIF(INDEX(ATC(IAC(L)),'NN') > 0) THEN
                   LETTER='N'
                ELSEIF(INDEX(ATC(IAC(L)),'ON') > 0) THEN
                   LETTER='O'
                ENDIF
                !
                ! Write Z-matrix line
                !
                WRITE(IUNIT,FMT) LETTER,IUSED(K),NRVAL, &
                     IUSED(J),NAVAL,IUSED(I),NDVAL
                CHANGE=2
                !
             ENDIF
          ENDIF
       ENDDO
       KK=KK+INC
       INC=-INC
       !
       ! Retry will be false if all coordinates are known
    ENDDO
    !
    IF(RETRY) THEN
       CALL WRNDIE(1,'<WRGAUS>','SOME COORDINATES FOUND')
    ELSE
       IF(PRNLEV >= 2) WRITE(OUTU,123)
    ENDIF
123 FORMAT(' ALL POSSIBLE COORDINATES HAVE BEEN PLACED')
    KK=0
    DO I=1,NATOM
       IF (IUSED(I) == 0) KK=KK+1
    ENDDO
    IF (KK /= 0 .AND. WRNLEV >= 2) WRITE(OUTU,124) KK
124 FORMAT(' ****  WARNING  ****',I5, &
         ' ATOMS HAVE NOT BEEN WRITTEN.')
    !
    WRITE(IUNIT,55)
55  FORMAT(A)
    !
    ! Printing of Z-matrix parameters starts here. 
    ! In reality it is difficult to separate variable and fixed parameters 
    !   automatically, since CHARMM is unaware which internal parameters 
    !   are driven and which are fixed. Hence, manual correction to the 
    !   gaussian file produced will be necessary in order to split
    !   variable and fixed Z-matrix parameter blocks.
    !
    IF(NEWZMT) THEN
       !
       !     Do not print first two distances because these are defined by
       !     Cartesian coordinates 
       !
       IF(WATER) THEN
          ! Print variable parameters
          FMT='(A1,I1,F13.5)'
          WRITE(IUNIT,FMT) 'R',4,RVAL(4)
          WRITE(IUNIT,*) ' '
          ! Print fixed parameters
          WRITE(IUNIT,FMT) 'R',3,RVAL(3)
          DO I=5,NRVAL
             FMT='(A1,I1,F13.5)'
             IF(I >= 10)  FMT(6:10)='2,F12'
             IF(I >= 100) FMT(6:10)='3,F11'
             WRITE(IUNIT,FMT) 'R',I,RVAL(I)
          ENDDO
       ELSE
          ! Print all parameters as one block
          DO I=3,NRVAL
             FMT='(A1,I1,F13.5)'
             IF(I >= 10)  FMT(6:10)='2,F12'
             IF(I >= 100) FMT(6:10)='3,F11'
             WRITE(IUNIT,FMT) 'R',I,RVAL(I)
          ENDDO
       ENDIF
    ELSE
       !
       ! Print all distances
       !
       DO I=1,NRVAL
          FMT='(A1,I1,F13.5)'
          IF(I >= 10)  FMT(6:10)='2,F12'
          IF(I >= 100) FMT(6:10)='3,F11'
          WRITE(IUNIT,FMT) 'R',I,RVAL(I)
       ENDDO
    ENDIF
    !
    !
    IF(NEWZMT) THEN
       !
       !     Do not print first valence angle because it is defined by
       !     Cartesian coordinates
       !
       DO I=2,NAVAL
          FMT='(A1,I1,F13.5)'
          IF(I >= 10)  FMT(6:10)='2,F12'
          IF(I >= 100) FMT(6:10)='3,F11'
          WRITE(IUNIT,FMT) 'A',I,AVAL(I)
       ENDDO
    ELSE
       !
       ! Print all valence angles
       !
       DO I=1,NAVAL
          FMT='(A1,I1,F13.5)'
          IF(I >= 10)  FMT(6:10)='2,F12'
          IF(I >= 100) FMT(6:10)='3,F11'
          WRITE(IUNIT,FMT) 'A',I,AVAL(I)
       ENDDO
    ENDIF
    !
    ! Print all torsion angles
    !
    DO I=1,NDVAL
       FMT='(A1,I1,F13.5)'
       IF(I >= 10)  FMT(6:10)='2,F12'
       IF(I >= 100) FMT(6:10)='3,F11'
       WRITE(IUNIT,FMT) 'D',I,DVAL(I)
    ENDDO
    !
    WRITE(IUNIT,"(/)") 
    !
    RETURN
  END SUBROUTINE WRGAUS
end module wrgaus_mod

