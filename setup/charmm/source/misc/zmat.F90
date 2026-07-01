#if KEY_NOMISC==0
SUBROUTINE ZMTRX
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use psf
  use stream
  use string
  use coord
  use memory
  use number,only:zero
  use nmrm, only:sel1at 
  implicit none
  !-----------------------------------------------------------------------
  ! General PSF and COOR information
  ! Selection of atoms
  INTEGER,allocatable,dimension(:) :: ISLCT

  ! Zmatrix arrays
  real(chm_real),allocatable,dimension(:,:) :: ZMAT
  INTEGER,allocatable,dimension(:,:) :: IZMAT
  INTEGER       NZMAT
  !
  !  Miscelaneous Local variables
  LOGICAL       EOF, OK
  INTEGER       I, N, IATOM
  !
  !
  ! Allocate
  call chmalloc('zmat.src','ZMTRX','ISLCT',natom,intg=ISLCT)
  call chmalloc('zmat.src','ZMTRX','ZMAT' ,3,natom,crl=ZMAT)
  call chmalloc('zmat.src','ZMTRX','IZMAT',4,natom,intg=IZMAT)

  NZMAT=0
  OK=.TRUE.
  EOF=.FALSE.

  if (prnlev >= 2) then
     WRITE(OUTU,100) 'ZMATRIX'
     WRITE(OUTU,100) 'One entry is expected per atom'
     WRITE(OUTU,100) '(no blank lines allowed)'
     WRITE(OUTU,100) '<at1>'
     WRITE(OUTU,100) '<at2> <at1> bond '
     WRITE(OUTU,100) '<at3> <at2> bond <at1> theta'
     WRITE(OUTU,100) '<at4> <at3> bond <at2> theta <at1> dihe'
     WRITE(OUTU,100) '...'
     WRITE(OUTU,*)
100  FORMAT(6X,A)
  endif

  !-----------------------------------------------------------------------
  mainloop: do while(.true.)
     CALL XTRANE(COMLYN,COMLEN,'ZMAT ')
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
          '  ZMAT> ')
     IF(EOF)THEN
        CALL PPSTRM(OK)
        IF(.NOT.OK)  then
           call chmdealloc('zmat.src','ZMTRX','ISLCT',natom,intg=ISLCT)
           call chmdealloc('zmat.src','ZMTRX','ZMAT' ,3,natom,crl=ZMAT)
           call chmdealloc('zmat.src','ZMTRX','IZMAT',4,natom,intg=IZMAT)
           RETURN
        endif
     ENDIF

     IF(INDXA(COMLYN,COMLEN,'ZEND') .GT. 0)THEN
        if (prnlev >= 2) then
           WRITE(OUTU,*)   'NZMAT = ',NZMAT
           WRITE(OUTU,101) IZMAT(1,1)
           WRITE(OUTU,101) IZMAT(2,2),IZMAT(1,2),ZMAT(1,2)
           WRITE(OUTU,101) IZMAT(3,3),IZMAT(2,3),ZMAT(1,3), &
                IZMAT(1,3),ZMAT(2,3)
           DO N=4,NZMAT
              WRITE(OUTU,101) IZMAT(4,N),IZMAT(3,N),ZMAT(1,N), &
                   IZMAT(2,N),ZMAT(2,N), &
                   IZMAT(1,N),ZMAT(3,N)
101           FORMAT(1x,I4,I4,F12.4,I4,F12.4,I4,F12.4)
           enddo
        endif
        CALL ZCONSTR(.TRUE.,4,NZMAT,IZMAT,ZMAT,X,Y,Z)
        call chmdealloc('zmat.src','ZMTRX','ISLCT',natom,intg=ISLCT)
        call chmdealloc('zmat.src','ZMTRX','ZMAT' ,3,natom,crl=ZMAT)
        call chmdealloc('zmat.src','ZMTRX','IZMAT',4,natom,intg=IZMAT)
        RETURN
     ELSE
        NZMAT=NZMAT+1
        IF(NZMAT.EQ.1)THEN
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(1,NZMAT)=IATOM

        ELSEIF(NZMAT.EQ.2)THEN
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(2,NZMAT)=IATOM
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(1,NZMAT)=IATOM
           ZMAT(1,NZMAT)=GTRMF(COMLYN,COMLEN,'DIST',ZERO)

        ELSEIF(NZMAT.EQ.3)THEN
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(3,NZMAT)=IATOM
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(2,NZMAT)=IATOM
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(1,NZMAT)=IATOM
           ZMAT(1,NZMAT)=GTRMF(COMLYN,COMLEN,'DIST',ZERO)
           ZMAT(2,NZMAT)=GTRMF(COMLYN,COMLEN,'THET',ZERO)

        ELSEIF(NZMAT.GE.4)THEN
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(4,NZMAT)=IATOM
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(3,NZMAT)=IATOM
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(2,NZMAT)=IATOM
           CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT, &
                NATOM,IBASE,NICTOT,NSEG,RESID,RES,SEGID,X,Y,Z,WMAIN)
           IZMAT(1,NZMAT)=IATOM
           ZMAT(1,NZMAT)=GTRMF(COMLYN,COMLEN,'DIST',ZERO)
           ZMAT(2,NZMAT)=GTRMF(COMLYN,COMLEN,'THET',ZERO)
           ZMAT(3,NZMAT)=GTRMF(COMLYN,COMLEN,'DIHE',ZERO)
        ENDIF
     ENDIF
  enddo mainloop
END SUBROUTINE ZMTRX

SUBROUTINE ZCONSTR(QINIT,FIRST,NZMAT,IZMAT,ZMAT,X,Y,Z)
  !-----------------------------------------------------------------------
  ! Construct atoms from a zmatrix
  !     1         4         bond  3-4
  !      \       /          theta 2-3-4   (theta=0 gives 4 between 2-3)
  !       2 --- 3           phi   1-2-3-4 (phi=0 gives a cis)
  !-----------------------------------------------------------------------
  use chm_kinds
  use number
  use consta
  use vector
  implicit none
  LOGICAL QINIT
  INTEGER FIRST,NZMAT,IZMAT(4,*)
  real(chm_real)  ZMAT(3,*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) V12(3),V32(3),V34(3),VPL(3),THETA,BOND,PHI
  INTEGER I,IZ1,IZ2,IZ3,IZ4
  !
  IF(QINIT)THEN
     ! Put the first atom
     IZ1=IZMAT(1,1)
     X(IZ1)=ZERO
     Y(IZ1)=ZERO
     Z(IZ1)=ZERO
     IF(NZMAT.EQ.1) RETURN

     ! Put the second atom
     IZ2=IZMAT(2,2)
     BOND=ZMAT(1,2)
     X(IZ2)=BOND
     Y(IZ2)=ZERO
     Z(IZ2)=ZERO
     IF(NZMAT.EQ.2) RETURN

     ! Put the third atom
     IZ2=IZMAT(2,3)
     IZ3=IZMAT(3,3)
     BOND=ZMAT(1,3)
     THETA=ZMAT(2,3)*DEGRAD
     IF(IZMAT(1,2).EQ.IZMAT(1,3))THEN
        X(IZ3)=X(IZ2)-BOND*COS(THETA)
     ELSE
        X(IZ3)=X(IZ2)+BOND*COS(THETA)
     ENDIF
     Y(IZ3)=BOND*SIN(THETA)
     Z(IZ3)=ZERO
     IF(NZMAT.EQ.3) RETURN

  ENDIF

  ! Put the remaining atoms
  DO I=FIRST,NZMAT
     BOND=ZMAT(1,I)
     THETA=ZMAT(2,I)*DEGRAD
     PHI=ZMAT(3,I)*DEGRAD
     IZ1=IZMAT(1,I)
     IZ2=IZMAT(2,I)
     IZ3=IZMAT(3,I)
     IZ4=IZMAT(4,I)
     !     make the unit vector R2-R3
     V32(1)=X(IZ2)-X(IZ3)
     V32(2)=Y(IZ2)-Y(IZ3)
     V32(3)=Z(IZ2)-Z(IZ3)
     CALL NORMALL(V32,3)
     V34(1)=V32(1)
     V34(2)=V32(2)
     V34(3)=V32(3)

     !     make the unit vector R2-R1
     V12(1)=X(IZ2)-X(IZ1)
     V12(2)=Y(IZ2)-Y(IZ1)
     V12(3)=Z(IZ2)-Z(IZ1)
     CALL NORMALL(V12,3)

     !     Make the vector normal to the plane Vpl = (V12 x V32)
     !     check if co-linear with the 2-3 bond
     IF(ABS(ZMAT(2,I)-180.0).GE.RSMALL)THEN
        CALL CROSS3(V12,V32,VPL)
        CALL ROTATX(V34,VPL,V34,-THETA)
        CALL ROTATX(V34,V32,V34,PHI)
     ELSE
        V34(1)=-V34(1)
        V34(2)=-V34(2)
        V34(3)=-V34(3)
     ENDIF

     X(IZ4)=X(IZ3)+BOND*V34(1)
     Y(IZ4)=Y(IZ3)+BOND*V34(2)
     Z(IZ4)=Z(IZ3)+BOND*V34(3)
  enddo
  RETURN
END SUBROUTINE ZCONSTR

SUBROUTINE ROTATX(R1,N,R2,THETA)
  !-----------------------------------------------------------------------
  use chm_kinds
  use vector
  implicit none
  real(chm_real) R1(3),N(3),R2(3),THETA,ROT(3)
  real(chm_real) R1XN(3),NR1
  real(chm_real) COST,SINT,FACT
  INTEGER J

  !     From Goldstein page 165
  !     r2 = r1 cos(theta) + n (n.r1)(1-cos(theta)) + (r1 x n)sin(theta)
  COST=COS(THETA)
  SINT=SIN(THETA)
  CALL NORMALL(N,3)
  CALL CROSS3(R1,N,R1XN)
  CALL DOTPR(N,R1,3,NR1)
  FACT=NR1*(1.0D0-COST)
  DO  J=1,3
     ROT(J)=R1(J)*COST+N(J)*FACT+R1XN(J)*SINT
  enddo

  R2(1:3)=ROT(1:3)
  RETURN
END SUBROUTINE ROTATX

SUBROUTINE G94INI
  !-----------------------------------------------------------------------
  !
  !     These soubroutines prepare input for GAUSSIAN program
  !     Selected atoms are treated quantum mechanically,
  !     while the rest of system is written at the end of file
  !     in a format to be processed by the GAUSSIAN CHARGE command
  !     (coordinates of point charges are in atomic units!)
  !     Suitable for potential energy surface scans of small systems
  !     in a protein, solvent or solid state environment
  !
  use chm_kinds
  use dimens_fcm
  use comand
  implicit none
  !
  CALL G94SEL(COMLYN,COMLEN)
  RETURN
END SUBROUTINE G94INI

SUBROUTINE G94SEL(COMLYN, COMLEN)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use coord
  use psf
  use select
  use stream
  use string

  implicit none
  !
  integer,allocatable,dimension(:) :: ISLCT
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  INTEGER IG94, NCHAR
  LOGICAL QCONT
  !
  call chmalloc('zmat.src','G94SEL','ISLCT',NATOM,intg=ISLCT)
  QCONT=(INDXA(COMLYN,COMLEN,'CNTL').GT.0)
  call SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
  !
  IG94 = GTRMI(COMLYN,COMLEN,'UNIT',6)
  !
  ! Number of characters from atom name to print
  NCHAR = GTRMI(COMLYN,COMLEN,'NCHA',1)
  IF(NCHAR.LT.1 .OR. NCHAR.GT.4) NCHAR=1
  !
  call G94ATO(IG94,NCHAR,ISLCT,QCONT)
  !
  call chmdealloc('zmat.src','G94SEL','ISLCT',NATOM,intg=ISLCT)
  COMLEN = 0
  !
  RETURN
END SUBROUTINE G94SEL

SUBROUTINE G94ATO(IG94,NCHAR,ISLCT,QCONT)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use consta
  use number
  use coord
  use stream
  use psf
  use gamess_fcm, only: qmused_qchem,qmused_g09,qmused_turbo
  implicit none
  !
  INTEGER ISLCT(*), IG94, NCHAR
  CHARACTER(len=16) SFORMAT
  !
  !
  INTEGER I, IPT
  CHARACTER(len=100) COMLYN, WD, LIN
  INTEGER COMLEN,WDLEN
  LOGICAL EOF,QCONT,QEXTCHRG
  !
  !     Get headers for G94 run
  CALL G94INP(IG94,QCONT)
  !
  IF (.not. QCONT) THEN
  DO I = 1, NATOM
     IF (ISLCT(I).EQ.1) THEN
        if(qmused_qchem.or.qmused_g09.or.qmused_turbo) then
           WRITE(IG94,'(1X,A1,3F12.5)') ATYPE(I), X(I), Y(I), Z(I)
        else
           WRITE(SFORMAT,'(A,I1,A)') "(1X,A",NCHAR,",3F12.5)"
           WRITE(IG94,SFORMAT) ATYPE(I), X(I), Y(I), Z(I)
        endif
     ENDIF
  ENDDO

  qmflags: if(qmused_qchem.or.qmused_g09) then
  !
  !     Get basis set
  !
  CALL G94INP(IG94,QCONT)
  !
  !write(*,*)'CONTROL = ',QCONT, .not. QCONT 
  IF (.not. QCONT) THEN 
     !WRITE(*,*)'IM IN THE LOOP TO WRITE STUFF...'
     !write(*,*)'ISLCT = ', I, ISLCT(1)
     DO I=1,NATOM 
         IF (ISLCT(I) .EQ. 0) QEXTCHRG=.TRUE. 
     ENDDO 
     IF (QEXTCHRG) THEN
        WRITE(IG94,'(A17)')'$external_charges'
     ENDIF
     
     IPT=1
     DO I = 1, NATOM
        IF (ISLCT(I).EQ.0) THEN
           IF(ABS(CG(I)).GT.RSMALL) THEN
              WRITE(IG94,'(3F12.5,F8.3)') &
                   X(I),Y(I),Z(I),CG(I)
              IPT = IPT+1
           ENDIF
        ENDIF
     ENDDO

     !IF (ISLCT(1).EQ.0) THEN
     IF (QEXTCHRG) THEN
        WRITE(IG94,'(A4)')'$end'
        WRITE(IG94,*)
     ENDIF
  ENDIF

  else

  WRITE(IG94,*)
  !
  !     Get basis set
  !
  CALL G94INP(IG94,QCONT)
  !
  IPT=1
  DO I = 1, NATOM
     IF (ISLCT(I).EQ.0) THEN
        IF(ABS(CG(I)).GT.RSMALL) THEN
           WRITE(IG94,'(3F12.5,F8.3)') &
                X(I)/BOHRR,Y(I)/BOHRR,Z(I)/BOHRR,CG(I)
           IPT = IPT+1
        ENDIF
     ENDIF
  ENDDO

  endif qmflags

  ENDIF 
  !
  RETURN
END SUBROUTINE G94ATO

SUBROUTINE G94INP(IG94,QCONT)
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  use dimens_fcm
  !
  use stream
  use string
  use gamess_fcm, only: qmused_qchem,qmused_g09

  implicit none
  !
  INTEGER IPT, IG94
  INTEGER, PARAMETER :: MLN=100
  CHARACTER(len=MLN) COMLYN, WD, LIN
  INTEGER COMLEN,WDLEN
  LOGICAL EOF,QCONT
  !
  loop102: do while(.true.)
     LIN=' '
     IPT=1
     EOF=.FALSE.
     if(qmused_qchem.or.qmused_g09) then
        CALL RDCMND(COMLYN,MLN,COMLEN,ISTRM,EOF,.TRUE.,.TRUE.,'QCINP>')
        IF(EOF)CALL WRNDIE(-5,'<QCINP>','NO INPUT FOR QCHEM.')
     else
        CALL RDCMND(COMLYN,MLN,COMLEN,ISTRM,EOF,.TRUE.,.TRUE.,'G94INP>')
        IF(EOF)CALL WRNDIE(-5,'<G94INP>','NO INPUT FOR G94.')
     endif
     loop103: do while(comlen /= 0)
        CALL NEXTWD(COMLYN,COMLEN,WD,MLN,WDLEN)
        if(qmused_qchem.or.qmused_g09) then
           IF((WD.EQ.'END').OR. (WD.EQ.'QCHEM_MOLECULE')) exit loop102
           !       IF((WD.EQ.'END') .OR.  &
           !            (WD.EQ.'QCHEM_MOLECULE')) exit loop102
           IF(WD.EQ.'QCHEM_MISC') cycle loop102
        else
           IF(WD.EQ.'END') exit loop102
           IF((WD.EQ.'GAUSSIAN_HEADERS').OR. &
                (WD.EQ.'GAUSSIAN_BASIS')) cycle loop102
        endif
        LIN(IPT:IPT+WDLEN+1) = WD//' '
        IPT = IPT+WDLEN+1
        IF(COMLEN.EQ.0) exit loop103
     enddo loop103

     CALL TRIMA(LIN,IPT)

     IF (LIN(1:IPT) .eq. '$') THEN 
        WRITE(IG94,'(A)')LIN(1:IPT-1)
        !write(*,*)'------------> BLANK LINE!!!!'
     ELSE 
        WRITE(IG94,'(A)')LIN(1:IPT)
     ENDIF 
     !WRITE(IG94,'(A)')LIN(1:IPT)
  enddo loop102
  !
  RETURN
END SUBROUTINE G94INP

#else /**/
SUBROUTINE ZMTRX
  CALL WRNDIE(-1,'<ZMTRX>','Z-Matrix code is not compiled.')
  RETURN
end SUBROUTINE ZMTRX
#endif 

