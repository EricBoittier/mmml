#if KEY_DIMB==1 /*dimb_main*/
SUBROUTINE INDDD1(II,INBCMP,INII)
  !-----------------------------------------------------------------------
  !     15-Dec-1992 David Perahia
  !     15-Dec-1994 Herman van Vlijmen
  !
  !     This routine calculates indices pointing into DD1
  !     for diagonal terms
  !     INBCMP  : pair list indices
  !     INII : index pointing into DD1
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  ! Passed variables
  INTEGER II,INBCMP(*),INII
  
  ! Begin
  IF(II == 1) THEN
     INII=(II-1)*6
  ELSE
     INII=(II-1)*6+INBCMP(II-1)*9
  ENDIF
  
  RETURN
END SUBROUTINE INDDD1

!
SUBROUTINE OFFDD1(II,JJ,INBCMP,JNBCMP,INIJ)
  !-----------------------------------------------------------------------
  !     15-Dec-1992 David Perahia
  !     15-Dec-1994 Herman van Vlijmen
  !
  !     This routine calculates indices pointing into DD1
  !     INBCMP and JNBCMP : pair list indices
  !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  implicit none
  ! Passed variables
  INTEGER II,JJ,INBCMP(*),JNBCMP(*),INIJ
  ! Local variables
  INTEGER NBR1,NBR2,JNBR
  
  ! Begin
  IF(II == 1) THEN
     NBR1=1
  ELSE
     NBR1=INBCMP(II-1)+1
  ENDIF
  NBR2=INBCMP(II)
  DO JNBR=NBR1,NBR2
     IF(JNBCMP(JNBR) == JJ) GOTO 900
  ENDDO
  WRITE(OUTU,103) II,JJ
  CALL WRNDIE(-3,'<OFFDD1>','Pair list not found')
103 FORMAT(/' Warning from OFFDD1: Atom pair ',2I6,' not found in', &
         'pair list')
900 INIJ=II*6+(JNBR-1)*9
  
  RETURN
END SUBROUTINE OFFDD1

!
SUBROUTINE MAKDDU(DD1,DD1CMP,INBCMP,JNBCMP,IUPT,IS1,IS2,NATOM)
  !-----------------------------------------------------------------------
  !     22-Jan-1993 David Perahia
  !     15-Dec-1994 Herman van Vlijmen
  !
  !     This routine generates the second derivative array DD1 from the
  !     compact second derivative array DD1CMP (upper triangle)
  !     Only the submatrix spanning from atom IS1 through atom IS2
  !     is generated.
  !     INBCMP and JNBCMP : pair list indices
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimb
  implicit none
  ! Passed variables
  real(chm_real) DD1(*),DD1CMP(*)
  INTEGER IS1,IS2,NATOM,INBCMP(*),JNBCMP(*),IUPT(*)
  ! Local variables
  INTEGER I,I2,J,K,II,JJ,NLINE,NELEM,NB,NBJ,IS1M1,NBR,IADD
  
  ! 
  NLINE=(IS2-IS1+1)*3
  NELEM=(NLINE*(NLINE+1))/2
  DO I=1,NELEM
     DD1(I)=0.D0
  ENDDO
  NB=0
  NBJ=0
  IS1M1=IS1-1
  IF(IS1 > 1) THEN
     NBJ=INBCMP(IS1M1)
     NB=IS1M1*6+NBJ*9
  ENDIF
  IF(IS2 == NATOM) THEN
     I2=NATOM-1
  ELSE
     I2=IS2
  ENDIF
  
  loop100: DO I=IS1,I2
     NBR=INBCMP(I)
     II=3*(I-IS1+1)-2
     !
     ! diagonal terms 
     !
     IADD=IUPT(II)+II
     DD1(IADD)=DD1CMP(NB+IIXXCM)
     IADD=IUPT(II+1)+II+1
     DD1(IADD)=DD1CMP(NB+IIYYCM)
     IADD=IUPT(II+2)+II+2
     DD1(IADD)=DD1CMP(NB+IIZZCM)
     IADD=IUPT(II)+II+1
     DD1(IADD)=DD1CMP(NB+IIXYCM)
     IADD=IUPT(II)+II+2
     DD1(IADD)=DD1CMP(NB+IIXZCM)
     IADD=IUPT(II+1)+II+2
     DD1(IADD)=DD1CMP(NB+IIYZCM)
     
     NB=NB+6   
     IF(NBJ == NBR) cycle loop100
     
     DO K=NBJ+1,NBR
        J=JNBCMP(K)
        IF(J <= IS2) then !GOTO 199
           JJ=3*(J-IS1+1)-2
           !
           ! off-diagonal terms 
           !
           IADD=IUPT(II)+JJ
           DD1(IADD)=DD1CMP(NB+IJXXCM)
           IADD=IUPT(II+1)+JJ+1
           DD1(IADD)=DD1CMP(NB+IJYYCM)
           IADD=IUPT(II+2)+JJ+2
           DD1(IADD)=DD1CMP(NB+IJZZCM)
           IADD=IUPT(II+1)+JJ
           DD1(IADD)=DD1CMP(NB+IJYXCM)
           IADD=IUPT(II)+JJ+1
           DD1(IADD)=DD1CMP(NB+IJXYCM)
           IADD=IUPT(II+2)+JJ
           DD1(IADD)=DD1CMP(NB+IJZXCM)
           IADD=IUPT(II)+JJ+2
           DD1(IADD)=DD1CMP(NB+IJXZCM)
           IADD=IUPT(II+2)+JJ+1
           DD1(IADD)=DD1CMP(NB+IJZYCM)
           IADD=IUPT(II+1)+JJ+2
           DD1(IADD)=DD1CMP(NB+IJYZCM)
        endif
        NB=NB+9   
     enddo
     NBJ=NBR
  enddo loop100
        
  !     
  ! diagonal terms of the last atom 
  !     
  IF (IS2 == NATOM) THEN
     II=3*(NATOM-IS1+1)-2
     IADD=IUPT(II)+II
     DD1(IADD)=DD1CMP(NB+IIXXCM)
     IADD=IUPT(II+1)+II+1
     DD1(IADD)=DD1CMP(NB+IIYYCM)
     IADD=IUPT(II+2)+II+2
     DD1(IADD)=DD1CMP(NB+IIZZCM)
     IADD=IUPT(II)+II+1
     DD1(IADD)=DD1CMP(NB+IIXYCM)
     IADD=IUPT(II)+II+2
     DD1(IADD)=DD1CMP(NB+IIXZCM)
     IADD=IUPT(II+1)+II+2
     DD1(IADD)=DD1CMP(NB+IIYZCM)
  ENDIF
  
  RETURN
END SUBROUTINE MAKDDU

!
SUBROUTINE MAKDWU(DD1,DD1CMP,INBCMP,JNBCMP,IUPT,IS1,IS2,IS3,IS4, &
     NATOM)
  !-----------------------------------------------------------------------
  !     22-Jan-1993 David Perahia
  !     15-Dec-1994 Herman van Vlijmen
  !
  !     This routine generates the second derivative array DD1 from the
  !     compact second derivative array DD1CMP (upper triangle)
  !     Only the submatrix spanning from atom IS1 through atom IS2
  !     and from atom IS3 through atom IS4 is generated.
  !     INBCMP and JNBCMP : pair list indices
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimb
  use number
  implicit none
  ! Passed variables
  real(chm_real) DD1(*),DD1CMP(*)
  INTEGER IS1,IS2,IS3,IS4,NATOM,INBCMP(*),JNBCMP(*),IUPT(*)
  ! Local variables
  INTEGER I,J,K,II,JJ,IS1M1,IADD
  INTEGER NLINE,NLINE1,NLINE2,NELEM,NB,NBJ,NBR
  
  ! Begin
  NLINE1=(IS2-IS1+1)*3
  NLINE2=(IS4-IS3+1)*3
  NLINE=NLINE1+NLINE2
  NELEM=(NLINE*(NLINE+1))/2
  DD1(1:nelem)=zero
  NB=0
  NBJ=0
  IS1M1=IS1-1
  IF(IS1 > 1) THEN
     NBJ=INBCMP(IS1M1)
     NB=IS1M1*6+NBJ*9
  ENDIF
  
  loop100:DO I=IS1,IS2
     NBR=INBCMP(I)
     II=3*(I-IS1+1)-2
     !
     ! diagonal terms 
     !
     IADD=IUPT(II)+II
     DD1(IADD)=DD1CMP(NB+IIXXCM)
     IADD=IUPT(II+1)+II+1
     DD1(IADD)=DD1CMP(NB+IIYYCM)
     IADD=IUPT(II+2)+II+2
     DD1(IADD)=DD1CMP(NB+IIZZCM)
     IADD=IUPT(II)+II+1
     DD1(IADD)=DD1CMP(NB+IIXYCM)
     IADD=IUPT(II)+II+2
     DD1(IADD)=DD1CMP(NB+IIXZCM)
     IADD=IUPT(II+1)+II+2
     DD1(IADD)=DD1CMP(NB+IIYZCM)
     NB=NB+6   
     IF(NBJ == NBR) cycle loop100
     DO K=NBJ+1,NBR
        J=JNBCMP(K)
        IF((J <= IS2) .or. .not.(J < IS3.OR.J.GT.IS4)) then  ! GOTO 199
           IF(J <= IS2) JJ=3*(J-IS1+1)-2
           IF(J >= IS3) JJ=3*(J-IS3+1)-2+NLINE1
           !
           ! off-diagonal terms 
           !
           IADD=IUPT(II)+JJ
           DD1(IADD)=DD1CMP(NB+IJXXCM)
           IADD=IUPT(II+1)+JJ+1
           DD1(IADD)=DD1CMP(NB+IJYYCM)
           IADD=IUPT(II+2)+JJ+2
           DD1(IADD)=DD1CMP(NB+IJZZCM)
           IADD=IUPT(II+1)+JJ
           DD1(IADD)=DD1CMP(NB+IJYXCM)
           IADD=IUPT(II)+JJ+1
           DD1(IADD)=DD1CMP(NB+IJXYCM)
           IADD=IUPT(II+2)+JJ
           DD1(IADD)=DD1CMP(NB+IJZXCM)
           IADD=IUPT(II)+JJ+2
           DD1(IADD)=DD1CMP(NB+IJXZCM)
           IADD=IUPT(II+2)+JJ+1
           DD1(IADD)=DD1CMP(NB+IJZYCM)
           IADD=IUPT(II+1)+JJ+2
           DD1(IADD)=DD1CMP(NB+IJYZCM)
        endif
        NB=NB+9   
     enddo
     NBJ=NBR
  enddo loop100
  
  RETURN
END SUBROUTINE MAKDWU

!
SUBROUTINE MASSDD(DD1CMP,DDM,INBCMP,JNBCMP,NATOM)
  !-----------------------------------------------------------------------
  !     09-Feb-1993 David Perahia
  !     15-Dec-1994 Herman van Vlijmen
  !
  !     Mass weight the compact second derivative matrix
  !     DD1CMP : compact second derivative matrix
  !     INBCMP and JNBCMP : pair list indices
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimb
  implicit none
  ! Passed variables
  real(chm_real) DDM(*),DD1CMP(*)
  INTEGER INBCMP(*),JNBCMP(*)
  ! Local variables
  real(chm_real) DDMM
  INTEGER I,J,K,NATOM,NB,NBJ,NBR
  
  ! Begin
  NB=0 
  NBJ=0
  
  loop100: DO I=1,NATOM-1
     NBR=INBCMP(I)
     DDMM=DDM(I)*DDM(I)
     !
     ! diagonal terms 
     !
     DD1CMP(NB+IIXXCM)=DD1CMP(NB+IIXXCM)*DDMM
     DD1CMP(NB+IIYYCM)=DD1CMP(NB+IIYYCM)*DDMM
     DD1CMP(NB+IIZZCM)=DD1CMP(NB+IIZZCM)*DDMM
     DD1CMP(NB+IIXYCM)=DD1CMP(NB+IIXYCM)*DDMM
     DD1CMP(NB+IIXZCM)=DD1CMP(NB+IIXZCM)*DDMM
     DD1CMP(NB+IIYZCM)=DD1CMP(NB+IIYZCM)*DDMM
     NB=NB+6   
     IF(NBJ  ==  NBR) cycle loop100
     DO K=NBJ+1,NBR
        J=JNBCMP(K)
        DDMM=DDM(I)*DDM(J)
        !
        ! off-diagonal terms 
        !
        DD1CMP(NB+IJXXCM)=DD1CMP(NB+IJXXCM)*DDMM
        DD1CMP(NB+IJYYCM)=DD1CMP(NB+IJYYCM)*DDMM
        DD1CMP(NB+IJZZCM)=DD1CMP(NB+IJZZCM)*DDMM
        DD1CMP(NB+IJYXCM)=DD1CMP(NB+IJYXCM)*DDMM
        DD1CMP(NB+IJXYCM)=DD1CMP(NB+IJXYCM)*DDMM
        DD1CMP(NB+IJZXCM)=DD1CMP(NB+IJZXCM)*DDMM
        DD1CMP(NB+IJXZCM)=DD1CMP(NB+IJXZCM)*DDMM
        DD1CMP(NB+IJZYCM)=DD1CMP(NB+IJZYCM)*DDMM
        DD1CMP(NB+IJYZCM)=DD1CMP(NB+IJYZCM)*DDMM
        NB=NB+9   
     enddo
     NBJ=NBR
  enddo loop100
  
  !     
  ! diagonal terms of the last atom 
  !     
  DDMM=DDM(NATOM)*DDM(NATOM)
  DD1CMP(NB+IIXXCM)=DD1CMP(NB+IIXXCM)*DDMM
  DD1CMP(NB+IIYYCM)=DD1CMP(NB+IIYYCM)*DDMM
  DD1CMP(NB+IIZZCM)=DD1CMP(NB+IIZZCM)*DDMM
  DD1CMP(NB+IIXYCM)=DD1CMP(NB+IIXYCM)*DDMM
  DD1CMP(NB+IIXZCM)=DD1CMP(NB+IIXZCM)*DDMM
  DD1CMP(NB+IIYZCM)=DD1CMP(NB+IIYZCM)*DDMM
  return
end SUBROUTINE MASSDD

#endif /* (dimb_main)*/
SUBROUTINE NULL_DIMBC
  RETURN
END SUBROUTINE NULL_DIMBC

