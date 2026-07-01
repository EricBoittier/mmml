module dimbutils

contains

#if KEY_DIMB==1 /*dimb_main*/
  SUBROUTINE ADJNME(DDV,PARDDV,NAT3,NDIM,NFRE,NFRET,IS1,IS2)
    !-----------------------------------------------------------------------
    !     11-Jan-1994 David Perahia
    !     16-Dec-1994 Herman van Vlijmen
    !
    !     Put the calculated eigenvectors (PARDDV) of the block of atoms
    !     IS1 to IS2 into DDV. Degrees of freedom that do not belong to
    !     this block are set to zero. This is used in generating the initial
    !     basis for the calculations.
    !-----------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    INTEGER NAT3,NFRE,NFRET,IS1,IS2,NDIM
    real(chm_real) DDV(NAT3,*),PARDDV(NDIM,*)
    ! Local variables
    INTEGER I,J,K,NELEM,IC1,IC2,IFRE

    ! Begin
    IC1=(IS1-1)*3
    IC2=IS2*3
    NELEM = NFRE*NAT3

    IF(IS1 > 1) THEN
       DO J=NFRET+1,NFRET+NFRE
          DO I=1,IC1
             DDV(I,J)=0.D0
          ENDDO
       ENDDO
    ENDIF

    IFRE=0
    DO J=NFRET+1,NFRET+NFRE
       IFRE=IFRE+1
       DO I=1,NDIM
          K=IC1+I
          DDV(K,J)=PARDDV(I,IFRE)
       ENDDO
    ENDDO

    IF(IC2 < NAT3) THEN
       DO J=NFRET+1,NFRET+NFRE
          DO I=IC2+1,NAT3
             DDV(I,J)=0.D0
          ENDDO
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE ADJNME

  !
  SUBROUTINE ADZERD(DDV,NF1,NF2,NDIM,IS1,IS2,IS3,IS4)
    !-----------------------------------------------------------------------
    !     01-Mar-1994 David Perahia
    !     16-Dec-1994 Herman van Vlijmen
    !
    !     Zero the elements corresponding to atoms IS1 to IS2,
    !     and IS3 to IS4, in DDV. Only vectors from NF1 to NF2
    !     are considered. This routine is used in setting up the
    !     basis for the mixed basis diagonalization.
    !-----------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    INTEGER NDIM,NF1,NF2,IS1,IS2,IS3,IS4
    real(chm_real) DDV(NDIM,*)
    ! Local variables
    INTEGER I,J,IC1,IC2,IC3,IC4

    ! Begin
    IC1=(IS1-1)*3+1
    IC2=IS2*3
    IC3=(IS3-1)*3+1
    IC4=IS4*3
    DO J=NF1,NF2
       DO I=IC1,IC2
          DDV(I,J)=0.D0
       ENDDO
       DO I=IC3,IC4
          DDV(I,J)=0.D0
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE ADZERD

  !
  SUBROUTINE ADZER(DDV,NF1,NF2,NDIM,IS1,IS2)
    !-----------------------------------------------------------------------
    !     01-Feb-1993 David Perahia
    !     16-Dec-1994 Herman van Vlijmen
    !
    !     Zero the elements corresponding to atoms IS1 to IS2 in DDV.
    !     Only vectors from NF1 to NF2 are considered.
    !     This routine is used in setting up the
    !     basis for the mixed basis diagonalization.
    !-----------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    INTEGER NDIM,NF1,NF2,IS1,IS2
    real(chm_real) DDV(NDIM,*)
    ! Local variables
    INTEGER I,J,IC1,IC2

    ! Begin
    IC1=(IS1-1)*3+1
    IC2=IS2*3
    DO J=NF1,NF2
       DO I=IC1,IC2
          DDV(I,J)=0.D0
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE ADZER

  !
  SUBROUTINE CLETR(DDV,DDVTR,NAT3,ISTRT,ISTOP,NFCUT,DDEV,DDF)
    !-----------------------------------------------------------------------
    !     01-Feb-1993 David Perahia
    !     16-Dec-1994 Herman van Vlijmen
    !
    !     This routine deletes the translation-rotation vectors
    !     from DDV, and replaces them by a new (exact) set. All
    !     vectors are subsequently orthonormalized.
    !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  use number
  use vector
    implicit none
    ! Passed variables
    INTEGER ISTRT,ISTOP,NAT3,NFCUT
    real(chm_real) DDV(NAT3,*),DDVTR(NAT3,*)
    real(chm_real) DDEV(*),DDF(*)
    ! Local variables
    INTEGER I,J,ITR,OLDPRN
    real(chm_real) PDR,SUMTR,TOLER
    LOGICAL QTR,LPURG

    ! Begin
    TOLER=TENM5
    ITR=0
    DO I=ISTRT,ISTOP
       QTR=.FALSE.
       IF(ITR == 6) GOTO 10
       SUMTR=ZERO
       DO J=1,6
          CALL DOTPR(DDVTR(1,J),DDV(1,I),NAT3,PDR)
          SUMTR=SUMTR+ABS(PDR)
       ENDDO
       IF(SUMTR > 0.7) QTR=.TRUE.
       IF(QTR) THEN
          ddv(1:nat3,i)=zero
          ITR=ITR+1
       ENDIF
    ENDDO

10  CONTINUE
    IF(ITR < 6) CALL WRNDIE(-3,'<CLETR>', &
         'There are less than six Trans-Rot vectors')

    NFCUT=ISTOP
    LPURG=.TRUE.
    OLDPRN=PRNLEV
    PRNLEV=1
    CALL ORTHNM(ISTRT,ISTOP,NFCUT,DDV,NAT3,LPURG,TOLER)
    PRNLEV=OLDPRN
    DO I=NFCUT,1,-1
       J=I+6
       ddv(1:nat3,j) = ddv(1:nat3,i)
    ENDDO
    NFCUT=NFCUT+6
    ddv(1:nat3,1:6)=ddvtr(1:nat3,1:6)
    OLDPRN=PRNLEV
    PRNLEV=1
    CALL ORTHNM(ISTRT,ISTOP,NFCUT,DDV,NAT3,LPURG,TOLER)
    PRNLEV=OLDPRN

    RETURN
  END SUBROUTINE CLETR

  !
  SUBROUTINE RLEG2(V,W,NATOM,DD1,INBCMP,JNBCMP,NFR)
    !-----------------------------------------------------------------------
    !     10-Mar-1984 Bernard R. Brooks (original RALEG2 routine)
    !     01-Mar-1993 David Perahia (adapted from RALEG2 routine)
    !     16-Dec-1994 Herman van Vlijmen
    !
    !     This routine computes the Raleigh quotient for multiple vectors
    !     using the compressed second derivative matrix.
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimb
  use parallel
  use machutil, only: eclock
    implicit none
    ! Passed variables
    real(chm_real) V(*),W(*),DD1(*)
    INTEGER NATOM,NFR
    INTEGER INBCMP(*),JNBCMP(*)
    ! Local variables
    INTEGER NAT3,IJADD,NBJ,LIM,NBR,JPR,ITEMP
    INTEGER IPT0,IPT1,IPT2,JPT0,JPT1,JPT2,I3,J3,I,J,IFR
    INTEGER IIXXC,IIXYC,IIXZC,IIYYC,IIYZC,IIZZC
    INTEGER IJXXC,IJXYC,IJXZC,IJYXC,IJYYC,IJYZC,IJZXC,IJZYC,IJZZC
#if KEY_PARALLEL==1
    real(chm_real) TIMMER
    ! Begin
    TIMMER = ECLOCK()
#endif 
    NAT3=NATOM*3
    DO I=1,NAT3*NFR
       W(I)=0.0
    ENDDO
    LIM=NATOM-1
    DO I=MYNODP,LIM,NUMNOD
       !     DO I=1,LIM
       NBR=INBCMP(I)
       IF(I /= 1) THEN
          NBJ=INBCMP(I-1)
       ELSE
          NBJ=0
       ENDIF
       IJADD=6*(I-1)+9*NBJ
       I3=I*3-2
       IPT0=I3
       IIXXC=IJADD+1
       IIXYC=IJADD+2
       IIXZC=IJADD+3
       IIYYC=IJADD+4
       IIYZC=IJADD+5
       IIZZC=IJADD+6
       DO IFR=1,NFR
          IPT1=IPT0+1
          IPT2=IPT0+2
          W(IPT0)=W(IPT0)+ &
               (V(IPT0)*DD1(IIXXC)+V(IPT1)*DD1(IIXYC)+V(IPT2)*DD1(IIXZC))
          W(IPT1)=W(IPT1)+ &
               (V(IPT0)*DD1(IIXYC)+V(IPT1)*DD1(IIYYC)+V(IPT2)*DD1(IIYZC))
          W(IPT2)=W(IPT2)+ &
               (V(IPT0)*DD1(IIXZC)+V(IPT1)*DD1(IIYZC)+V(IPT2)*DD1(IIZZC))
          IPT0=IPT0+NAT3
       ENDDO

       IJADD=IJADD+6
       IF(NBR > NBJ) THEN
          DO JPR=NBJ+1,NBR
             J=JNBCMP(JPR)
             J3=J*3-2
             IPT0=I3
             JPT0=J3
             IJXXC=IJADD+1
             IJXYC=IJADD+2
             IJXZC=IJADD+3
             IJYXC=IJADD+4
             IJYYC=IJADD+5
             IJYZC=IJADD+6
             IJZXC=IJADD+7
             IJZYC=IJADD+8
             IJZZC=IJADD+9
             DO IFR=1,NFR
                IPT1=IPT0+1
                IPT2=IPT0+2
                JPT1=JPT0+1
                JPT2=JPT0+2
                W(JPT0)=W(JPT0)+(V(IPT0)*DD1(IJXXC)+ &
                     V(IPT1)*DD1(IJYXC)+V(IPT2)*DD1(IJZXC))
                W(JPT1)=W(JPT1)+(V(IPT0)*DD1(IJXYC)+ &
                     V(IPT1)*DD1(IJYYC)+V(IPT2)*DD1(IJZYC))
                W(JPT2)=W(JPT2)+(V(IPT0)*DD1(IJXZC)+ &
                     V(IPT1)*DD1(IJYZC)+V(IPT2)*DD1(IJZZC))
                IPT0=IPT0+NAT3
                JPT0=JPT0+NAT3
             ENDDO

             IPT0=I3
             JPT0=J3
             DO IFR=1,NFR
                IPT1=IPT0+1
                IPT2=IPT0+2
                JPT1=JPT0+1
                JPT2=JPT0+2
                W(IPT0)=W(IPT0)+(V(JPT0)*DD1(IJXXC)+ &
                     V(JPT1)*DD1(IJXYC)+V(JPT2)*DD1(IJXZC))
                W(IPT1)=W(IPT1)+(V(JPT0)*DD1(IJYXC)+ &
                     V(JPT1)*DD1(IJYYC)+V(JPT2)*DD1(IJYZC))
                W(IPT2)=W(IPT2)+(V(JPT0)*DD1(IJZXC)+ &
                     V(JPT1)*DD1(IJZYC)+V(JPT2)*DD1(IJZZC))
                IPT0=IPT0+NAT3
                JPT0=JPT0+NAT3
             ENDDO
             IJADD=IJADD+9

          ENDDO  !  JPR=NBJ+1,NBR
          NBJ=NBR

       ENDIF  !  IF(NBJ > NBR)
    ENDDO     !  I=1,LIM

    ! last atom
#if KEY_PARALLEL==1
    IF(MYNOD == 0) THEN
#endif 
       I3=NATOM*3-2
       IPT0=I3
       IJADD=6*LIM+9*INBCMP(LIM)
       IIXXC=IJADD+1
       IIXYC=IJADD+2
       IIXZC=IJADD+3
       IIYYC=IJADD+4
       IIYZC=IJADD+5
       IIZZC=IJADD+6

       DO IFR=1,NFR
          IPT1=IPT0+1
          IPT2=IPT0+2
          W(IPT0)=W(IPT0)+ &
               (V(IPT0)*DD1(IIXXC)+V(IPT1)*DD1(IIXYC)+V(IPT2)*DD1(IIXZC))
          W(IPT1)=W(IPT1)+ &
               (V(IPT0)*DD1(IIXYC)+V(IPT1)*DD1(IIYYC)+V(IPT2)*DD1(IIYZC))
          W(IPT2)=W(IPT2)+ &
               (V(IPT0)*DD1(IIXZC)+V(IPT1)*DD1(IIYZC)+V(IPT2)*DD1(IIZZC))
          IPT0=IPT0+NAT3
       ENDDO
#if KEY_PARALLEL==1
    ENDIF
    TMERI(6)=TMERI(6)+ECLOCK()-TIMMER
#endif 
    RETURN
  END SUBROUTINE RLEG2

  !
  SUBROUTINE TRROT(X,Y,Z,DDV,NAT3,NFRE,DDM)
    !-----------------------------------------------------------------------
    !     01-Feb-1993 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Add normalized translation and rotation vectors into DDV array
    !     beginning at NFRE.
    !-----------------------------------------------------------------------
  use chm_kinds
  use number
  use vector
    implicit none
    ! Passed variables
    INTEGER NAT3,NFRE
    real(chm_real) DDV(NAT3,*),DDM(*)
    real(chm_real) X(*),Y(*),Z(*)
    ! Local variables
    real(chm_real) TETA,CTETA,STETA
    PARAMETER (TETA=PT0001)
    INTEGER I,NATOM,NF,IPT

    ! Begin
    NATOM=NAT3/3
    NF=NFRE-1
    CTETA=COS(TETA)
    STETA=SIN(TETA)
    ! translations
    ! X
    IPT=0
    DO I=1,NATOM
       DDV(IPT+1,NF+1)=ONE/DDM(I)
       DDV(IPT+2,NF+1)=ZERO
       DDV(IPT+3,NF+1)=ZERO
       IPT=IPT+3
    ENDDO
    ! Y
    IPT=0
    DO I=1,NATOM
       DDV(IPT+1,NF+2)=ZERO
       DDV(IPT+2,NF+2)=ONE/DDM(I)
       DDV(IPT+3,NF+2)=ZERO
       IPT=IPT+3
    ENDDO
    ! Z
    IPT=0
    DO I=1,NATOM
       DDV(IPT+1,NF+3)=ZERO
       DDV(IPT+2,NF+3)=ZERO
       DDV(IPT+3,NF+3)=ONE/DDM(I)
       IPT=IPT+3
    ENDDO

    ! rotations
    ! X
    IPT=0
    DO I=1,NATOM
       DDV(IPT+1,NF+4)=ZERO
       DDV(IPT+2,NF+4)=(Y(I)-(CTETA*Y(I)-STETA*Z(I)))/DDM(I)
       DDV(IPT+3,NF+4)=(Z(I)-(STETA*Y(I)+CTETA*Z(I)))/DDM(I)
       IPT=IPT+3
    ENDDO
    ! Y
    IPT=0
    DO I=1,NATOM
       DDV(IPT+1,NF+5)=(X(I)-(CTETA*X(I)+STETA*Z(I)))/DDM(I)
       DDV(IPT+2,NF+5)=ZERO
       DDV(IPT+3,NF+5)=(Z(I)-(-STETA*X(I)+CTETA*Z(I)))/DDM(I)
       IPT=IPT+3
    ENDDO
    ! Z
    IPT=0
    DO I=1,NATOM
       DDV(IPT+1,NF+6)=(X(I)-(CTETA*X(I)-STETA*Y(I)))/DDM(I)
       DDV(IPT+2,NF+6)=(Y(I)-(STETA*X(I)+CTETA*Y(I)))/DDM(I)
       DDV(IPT+3,NF+6)=ZERO
       IPT=IPT+3
    ENDDO

    ! normalize the vectors
    DO I=NF+1,NF+6
       CALL NORMALL(DDV(1,I),NAT3)
    ENDDO
    !
    RETURN
  END SUBROUTINE TRROT

  !

  SUBROUTINE ADTRRT(X,Y,Z,PARDDV,DDM,NDIM,IS1,IS2,NFRE)
    !-------------------------------------------------------------------
    !     06-Feb-1996 Herman van Vlijmen
    !
    !     Add translation and rotation vectors to the eigenvectors
    !     of the atom block IS1-IS2.
    !-------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    INTEGER NDIM,IS1,IS2,NFRE
    real(chm_real) X(*),Y(*),Z(*),PARDDV(NDIM,*),DDM(*)
    ! Local variables
    INTEGER I,J

    ! Begin
    DO I=NFRE,7,-1
       DO J=1,NDIM
          PARDDV(J,I) = PARDDV(J,I-6)
       ENDDO
    ENDDO

    CALL TRROT(X(IS1),Y(IS1),Z(IS1),PARDDV,NDIM,1,DDM(IS1))

    RETURN
  END SUBROUTINE ADTRRT


  SUBROUTINE MAPMOD(DDV,DDSCR2,NAT3,NAT3P,NDIM,IBOUND,IUNRMD)
    !-----------------------------------------------------------------------
    !     17-May-1996 Herman van Vlijmen
    !
    !     Maps elements IBOUND(1,1) to IBOUND(1,2) of a normal mode
    !     file contained in unit IUNRMD into elements IBOUND(2,1) to
    !     IBOUND(2,2) of DDV. This is useful if modes are known for a very
    !     similar structure.
    !-----------------------------------------------------------------------
  use chm_kinds
  use number,only:zero
    implicit none
    ! Passed variables
    INTEGER NAT3,NAT3P,NDIM,IBOUND(2,2),IUNRMD
    real(chm_real)  DDV(NAT3,*),DDSCR2(NAT3P)
    ! Local variables
    INTEGER J1,J2,K1,I,J
    !-----------------------------------------------------------------------
    ! Begin
    J1=(IBOUND(1,1)-1)*3+1
    J2=IBOUND(1,2)*3
    K1=(IBOUND(2,1)-1)*3
    DO I=1,NDIM
       READ(IUNRMD) DDSCR2
       ddv(1:nat3,i)=zero
       DDV(K1+J1:k1+j2,I) = DDSCR2(J1:j2)
    ENDDO
    RETURN
  END SUBROUTINE MAPMOD
#endif /* (dimb_main)*/

  subroutine dimbutils_dummy()
    return
  end subroutine dimbutils_dummy

end module dimbutils

