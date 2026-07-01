#if KEY_FOURD==0 /*4defour*/
SUBROUTINE PARSE4D(COMLYN,COMLEN)
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  CALL WRNDIE(-1,'<PARSE4D>','No FOURD code compiled')
  RETURN
END SUBROUTINE PARSE4D
#else /* (4defour)*/
SUBROUTINE PARSE4D(COMLYN,COMLEN)
  !
  ! This routine parses all of the 4D options for minimization
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use energym
  use fourdm
  use stream
  use string
  !
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  !
  INTEGER I
  !
  IF(INDXA(COMLYN,COMLEN,'OFF').GT.0) THEN
     DIM4=.FALSE.
     QSHAK4=.FALSE.
     CM4D=.FALSE.
     call deallocate_fourd_ltm()
     RETURN
  ENDIF
  !
  if(.not.allocated(fdim)) then
     call allocate_fourd_ltm()
  endif
  DIM4=.TRUE.
  FCOUNT=0
  INC4D=99999
  DEC4D=99999
  K4DI= GTRMF(COMLYN,COMLEN,'K4DI',K4DI)
  MULTK4 = GTRMF(COMLYN,COMLEN,'MULT',MULTK4)
  DO I = 1,LENENT
     DIM4ON(I) = 1
  ENDDO
  IF (INDXA(COMLYN, COMLEN, 'SKBO') .GT. 0) DIM4ON(1)=0
  IF (INDXA(COMLYN, COMLEN, 'SKAN') .GT. 0) DIM4ON(2)=0
  IF (INDXA(COMLYN, COMLEN, 'SKDI') .GT. 0) DIM4ON(4)=0
  IF (INDXA(COMLYN, COMLEN, 'SKVD') .GT. 0) DIM4ON(6)=0
  IF (INDXA(COMLYN, COMLEN, 'SKEL') .GT. 0) DIM4ON(7)=0
  IF (INDXA(COMLYN, COMLEN, 'SKCO') .GT. 0) DIM4ON(11)=0

  ! 4D shake
  IF(INDXA(COMLYN,COMLEN,'SHAK').GT.0) THEN
     QSHAK4=.TRUE.
     CM4D=.TRUE.
  ENDIF
  !
#if KEY_DEBUG==1
  write(outu,*)'in charmm*.src'
  write(outu,*)'dim4on(bond),dim4on(vdw),dim4on(elec)'
  write(outu,*)dim4on(1),dim4on(6),dim4on(7)
#endif 
  RETURN
  !
END SUBROUTINE PARSE4D

SUBROUTINE SHAKA4(FDIM,AMASS,NATOM,IMOVE4)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  implicit none
  real(chm_real) FDIM(*)
  real(chm_real) AMASS(*)
  INTEGER NATOM
  INTEGER IMOVE4(*)
  !
  real(chm_real) FDC,TMASS
  INTEGER I
  !
  !    Let's figure out the center of mass for the 4th D Solute
  !
  FDC=ZERO
  TMASS=ZERO
  DO I = 1,NATOM
     IF (IMOVE4(I).EQ.0) THEN
        FDC=FDC + FDIM(I)*AMASS(I)
        TMASS=TMASS + AMASS(I)
     ENDIF
  ENDDO
  FDC=FDC/TMASS
  !
  !    Set the 4th D values all to the Center of Mass 4th D coordinate.
  DO I = 1,NATOM
     IF (IMOVE4(I).EQ.0) FDIM(I)=FDC
  ENDDO

  RETURN
END SUBROUTINE SHAKA4

SUBROUTINE SHAKF4(DFDIM,AMASS,NATOM,IMOVE4)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  implicit none
  real(chm_real) DFDIM(*)
  real(chm_real) AMASS(*)
  INTEGER NATOM
  INTEGER IMOVE4(*)
  !
  real(chm_real) FDC,TMASS
  INTEGER I
  !
  !    Let's figure out the center of mass force for the 4th D Solute
  !
  FDC=ZERO
  TMASS=ZERO
  DO I = 1,NATOM
     IF (IMOVE4(I).EQ.0) THEN
        FDC=FDC + DFDIM(I)*AMASS(I)
        TMASS=TMASS + AMASS(I)
     ENDIF
  ENDDO
  FDC=FDC/TMASS
  !
  !    Set the 4th D forces so all atoms have same 4D acceleration.
  !    Preserve the total force.
  DO I = 1,NATOM
     IF (IMOVE4(I).EQ.0) DFDIM(I)=FDC*AMASS(I)
  ENDDO
  !
  RETURN
END SUBROUTINE SHAKF4

SUBROUTINE EFOUR(E4D,NATOM,AMASS)
  !-----------------------------------------------------------------------
  !     CALCULATES PROJECTION TERM FROM 4D-->3D AS GIVEN BY
  !          (1/2)*K4d*W**2
  !           where W=FDIM-FDEQ
  !
  !     K4D  ... force constant
  !     K4DI ... force constant initially
  !     MULTK4.. the final factor K4DI is multiplied by at the end
  !              of the back projection.
  !
  !     Written by Elan Eisenmesser
  use chm_kinds
  use dimens_fcm
  use fourdm
  implicit none
  real(chm_real) E4D, AMASS(*)
  real(chm_real) K4D,KTEMP
  INTEGER I, NATOM
  !     SAVE K4D
  real(chm_real) FDC,TMASS
  INTEGER DIVE
  !
  !   K4D default set arbitrarily to
  !       100KJ/mol*nm**2=.2390Kcal/mol*Ang**2
  !   and after (DEC4D-INC4D) steps it's increased to MULTK4*K4DI
  E4D=0.
  IF(FCOUNT.LE.INC4D) K4D=K4DI
  IF((FCOUNT.GT.INC4D).AND.(FCOUNT.LT.DEC4D)) THEN
     KTEMP=DEC4D-INC4D
     K4D=K4DI+(MULTK4-1)*K4DI*(FCOUNT-INC4D)/KTEMP
  ENDIF
  IF(FCOUNT.GE.DEC4D) K4D=MULTK4*K4DI

  IF (CM4D) THEN
     !       Let's figure out the center of mass for the 4th D Solute
     TMASS=0.0
     DIVE=0
     DO I = 1,NATOM
        IF (IMOVE4(I).EQ.0) THEN
           TMASS=TMASS + AMASS(I)
        ENDIF
     ENDDO
     DO I=1,NATOM
        IF(IMOVE4(I).EQ.0) THEN
           E4D = E4D + (0.5*K4D*((FDIM(I)-FDEQ(I))**2))
           DFDIM(I)=DFDIM(I)+(K4D*(FDIM(I)-FDEQ(I))*AMASS(I)/TMASS)
           DIVE=DIVE + 1
        ENDIF
     ENDDO
     E4D = E4D/FLOAT(DIVE)
  ELSE
     DO I=1,NATOM
        IF(IMOVE4(I).EQ.0) THEN
           E4D = E4D + (0.5*K4D*((FDIM(I)-FDEQ(I))**2))
           DFDIM(I)=DFDIM(I)+(K4D*(FDIM(I)-FDEQ(I)))
        ENDIF
     ENDDO
  ENDIF

  RETURN
END SUBROUTINE EFOUR

SUBROUTINE EPHI4(EP,IP,JP,KP,LP,ICP,NPHI,CPC,CPD,CPB, &
     DX,DY,DZ,X,Y,Z,QECONT,ECONT,ICONHP,ISKP, &
     DD1,IUPT,QSECD)
  !-----------------------------------------------------------------------
  !    This subroutine replaces only the proper dihedral energy and
  !    first derivative calculations.
  !
  !    Written by Elan Eisenmesser
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use stream
  use fourdm
#if KEY_BLOCK==1
  use block_fcm
#endif /* BLOCK*/
  use consta
  implicit none
  real(chm_real) EP
  INTEGER IP(*),JP(*),KP(*),LP(*),ICP(*)
  INTEGER NPHI
  real(chm_real) CPC(*),CPB(*), DI
  INTEGER CPD(*)
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  INTEGER ICONHP
  INTEGER ISKP(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  !
  real(chm_real) FX,FY,FZ,FFDIM,GX,GY,GZ,GFDIM,HX,HY,HZ,HFDIM
  real(chm_real) E1,DF1,DDF1,E,DF,DDF
  INTEGER I,J,K,L
  real(chm_real) CT,CT2,CPBIC
  real(chm_real) AX,AY,AZ,AFDIM,BX,BY,BZ,BFDIM,FG,HG
  INTEGER IC,IPER
  LOGICAL LREP,NOCONS
  real(chm_real) RMJX,RMJY,RMJZ,RNKX,RNKY,RNKZ,RMJNOR,RNKNOR,MJDNK
  real(chm_real) RNK2,RMJ2,ARG
  real(chm_real) AP,AP2,GNK2,GMJ2,EXTRAD
  real(chm_real) HGGGJ,FGGGJ,HGGGK,FGGGK
  !
  real(chm_real) ADOTB,ANORM,BNORM,FDOTG,GDOTG,HDOTG
  real(chm_real) RANORM,RBNORM
  real(chm_real) DFXI,DFYI,DFZI,DFFDI,DFXL,DFYL,DFZL,DFFDL
  real(chm_real) DFXJ,DFYJ,DFZJ,DFFDJ,DFXK,DFYK,DFZK,DFFDK
  INTEGER IPHI
  EP=ZERO
  NOCONS= ICONHP.EQ.0
  IF (NPHI.EQ.0) RETURN
  IPHI=0
  IF (.NOT.(IPHI.GE.NPHI)) THEN
10   CONTINUE
     IPHI=IPHI+1
     I=IP(IPHI)
     IF (.NOT.(NOCONS)) THEN
        IF(ISKP(IPHI).NE.0) GOTO 160
     ENDIF
     J=JP(IPHI)
     K=KP(IPHI)
     L=LP(IPHI)
     IC=ICP(IPHI)
     IF (IC.EQ.0) GOTO 160
     FX=X(I)-X(J)
     FY=Y(I)-Y(J)
     FZ=Z(I)-Z(J)
     FFDIM=FDIM(I)-FDIM(J)
     GX=X(J)-X(K)
     GY=Y(J)-Y(K)
     GZ=Z(J)-Z(K)
     GFDIM=FDIM(J)-FDIM(K)
     HX=X(L)-X(K)
     HY=Y(L)-Y(K)
     HZ=Z(L)-Z(K)
     HFDIM=FDIM(L)-FDIM(K)
     ! FROM GUNSTEREN'S PAPER:
     ! THE PHI ANGLE IS NOW BETWEEN PLANES i-j-k AND j-k-l
     ! PHI=ARCCOS((AdotB)/AB) WHERE A=F-(FdotG)G/G**2 AND
     !                              B=H+(HdotG)G/G**2
     ! NOTE: FG=-(FdotG)/G**2 AND HG=(HdotG)/G**2
     FDOTG=FX*GX+FY*GY+FZ*GZ+FFDIM*GFDIM
     GDOTG=GX*GX+GY*GY+GZ*GZ+GFDIM*GFDIM
     FG=FDOTG/GDOTG
     HDOTG=HX*GX+HY*GY+HZ*GZ+HFDIM*GFDIM
     HG=HDOTG/GDOTG
     AX=FX-FG*GX
     AY=FY-FG*GY
     AZ=FZ-FG*GZ
     AFDIM=FFDIM-FG*GFDIM
     BX=HX-HG*GX
     BY=HY-HG*GY
     BZ=HZ-HG*GZ
     BFDIM=HFDIM-HG*GFDIM
     !        BFDIM=HFDIM-FG*GFDIM
     ! NOW TO COMPUTE THE ANGLE BETWEEN THE PLANES CALLED
     ! OR RATHER THE COSINE OF IT:CT
     ADOTB=AX*BX+AY*BY+AZ*BZ+AFDIM*BFDIM
     ANORM=SQRT(AX*AX+AY*AY+AZ*AZ+AFDIM*AFDIM)
     BNORM=SQRT(BX*BX+BY*BY+BZ*BZ+BFDIM*BFDIM)
     CT=(ADOTB/(ANORM*BNORM))
     ! Energy and derivative contributions
     IF (CPD(IC).NE.0) THEN
        ! SET UP FOR THE PROPER DIHEDRALS
        !
        E=ZERO
        DF=ZERO
        DDF=ZERO
30      CONTINUE
        CPBIC=CPB(IC)
        IPER=CPD(IC)
        IF(IPER.GE.0)THEN
           LREP=.FALSE.
        ELSE
           LREP=.TRUE.
           IPER=-IPER
        ENDIF
        !
        IF (IPER.EQ.1) THEN
           E1=CT
           DF1=ONE
           DDF1=ZERO
        ELSE IF (IPER.EQ.2) THEN
           E1=TWO*CT*CT-ONE
           DF1=FOUR*CT
           DDF1=FOUR
        ELSE IF (IPER.EQ.3) THEN
           CT2=CT*CT
           E1=CT*(FOUR*CT2-THREE)
           DF1=TWELVE*CT2-THREE
           DDF1=24.D0*CT
        ELSE IF (IPER.EQ.4) THEN
           CT2=CT*CT
           E1=ONE+CT2*EIGHT*(CT2-ONE)
           DF1=16.D0*CT*(CT2+CT2-ONE)
           DDF1=16.D0*(SIX*CT2-ONE)
        ELSE IF (IPER.EQ.6) THEN
           CT2=CT*CT
           E1=CT2*(CT2*(CT2*32.D0-48.D0)+18.D0)-ONE
           DF1=CT*(CT2*(CT2*192.D0-192.D0)+36.D0)
           DDF1=CT2*(CT2*960.D0-576.D0)+36.D0
        ELSE IF (IPER.EQ.5) THEN
           CT2=CT*CT
           E1=CT*(CT2*(CT2*16.D0-20.D0)+FIVE)
           DF1=CT2*(CT2*80.D0-60.D0)+FIVE
           DDF1=CT*(CT2*320.D0-120.D0)
        ELSE IF (IPER.EQ.0) THEN
           E1=ZERO
           DF1=ZERO
           DDF1=ZERO
        ELSE
           IF(WRNLEV.GE.2) WRITE(OUTU,40) I,J,K,L,IC,CPD(IC)
40         FORMAT(' BAD PERIOD: (I,J,K,L,IC,IPER)',6I5)
           CALL WRNDIE(-3,'<EPHI>  ', &
                'BAD PERIODICITY IN LIST FOR DIHEDRAL ANGLES')
        ENDIF
        !
        ARG=CPC(IC)
        IF (CPBIC.NE.ZERO) THEN
           ARG=-ARG
           IF (ABS(PI-CPBIC).GT.PT01) THEN
              CALL WRNDIE(-3,'<EPHI4>  ', &
                   'BAD PHASE IN LIST FOR DIHEDRAL ANGLES')
           ENDIF
        ENDIF
        E=E+CPC(IC)+ARG*E1
        DF=DF+DF1*ARG
        DDF=DDF+DDF1*ARG
        !
        !  First derivatives:
        RANORM=1./ANORM
        RBNORM=1./BNORM
        DF=-ARG*DF1
        !     for atom I
        DFXI=-DF*RANORM*((BX/BNORM)-CT*(AX/ANORM))
        DFYI=-DF*RANORM*((BY/BNORM)-CT*(AY/ANORM))
        DFZI=-DF*RANORM*((BZ/BNORM)-CT*(AZ/ANORM))
        DFFDI=-DF*RANORM*((BFDIM/BNORM)-CT*(AFDIM/ANORM))
        DX(I)=DX(I)+DFXI
        DY(I)=DY(I)+DFYI
        DZ(I)=DZ(I)+DFZI
        DFDIM(I)=DFDIM(I)+DFFDI
        !     for atom L
        DFXL=-DF*RBNORM*((AX/ANORM)-CT*(BX/BNORM))
        DFYL=-DF*RBNORM*((AY/ANORM)-CT*(BY/BNORM))
        DFZL=-DF*RBNORM*((AZ/ANORM)-CT*(BZ/BNORM))
        DFFDL=-DF*RBNORM*((AFDIM/ANORM)-CT*(BFDIM/BNORM))
        DX(L)=DX(L)+DFXL
        DY(L)=DY(L)+DFYL
        DZ(L)=DZ(L)+DFZL
        DFDIM(L)=DFDIM(L)+DFFDL
        !     for atom J
        DFXJ=((-FG-1)*DFXI)-(HG*DFXL)
        DFYJ=((-FG-1)*DFYI)-(HG*DFYL)
        DFZJ=((-FG-1)*DFZI)-(HG*DFZL)
        DFFDJ=((-FG-1)*DFFDI)-(HG*DFFDL)
        DX(J)=DX(J)+DFXJ
        DY(J)=DY(J)+DFYJ
        DZ(J)=DZ(J)+DFZJ
        DFDIM(J)=DFDIM(J)+DFFDJ
        !     for atom k
        DFXK=((HG-1)*DFXL)-(-FG*DFXI)
        DFYK=((HG-1)*DFYL)-(-FG*DFYI)
        DFZK=((HG-1)*DFZL)-(-FG*DFZI)
        DFFDK=((HG-1)*DFFDL)-(-FG*DFFDI)
        DX(K)=DX(K)+DFXK
        DY(K)=DY(K)+DFYK
        DZ(K)=DZ(K)+DFZK
        DFDIM(K)=DFDIM(K)+DFFDK
        IF(LREP) THEN
           IC=IC+1
           GOTO 30
        ENDIF
     ELSE
        !
        ! Set up for the improper dihedrals given by Gunsteren's Paper:
        !        Rmj=Rij x Rkj       and    Zeta=arccos(RmjdotRnk/RmjRnk)
        !        Rnk=Rkj x Rkl                 is the angle between
        !                                     planes i-j-k and j-k-l
        ! Same as normal charmm so not done here
     ENDIF
     EP = EP + E
160  CONTINUE
     IF (.NOT.(IPHI.GE.NPHI)) GOTO 10
  ENDIF
  RETURN
END SUBROUTINE EPHI4

SUBROUTINE FILL4(X,Y,Z,NATOM,E4FILL)
  !-----------------------------------------------------------------------
  !     Fill the fourth dimension coordinates.
  !     The radius of gyration is calculated and
  !     from this a random set of 4th dimensional
  !     coordinates is determined and returned in
  !     the array FDIM.  E4FILL dictates how large
  !     their value will be.
  !
  !     Written by Elan Eisenmesser
  use chm_kinds
  use exfunc
  use clcg_mod,only:random
  use dimens_fcm
  use fourdm
  use number
  use consta
  implicit none
  real(chm_real) SD,A,B
  real(chm_real) XCM,YCM,ZCM,RCM
  real(chm_real) E4FILL
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER SEED,NATOM
  INTEGER I
  !
  DO I=1,NATOM
     FDIM(I)=0.0
     DFDIM(I)=0.0
  ENDDO
  !
  !     Now the filling of the fourth coordinates
  !     Determine standard deviation(SD) from
  !       radius of gyration(RG)
  !        Rg**2=sum(ri-rcm)**2/N
  XCM=0.
  YCM=0.
  ZCM=0.
  DO I=1,NATOM
     XCM=XCM + X(I)
     YCM=YCM + Y(I)
     ZCM=ZCM + Z(I)
  ENDDO
  XCM=XCM/NATOM
  YCM=YCM/NATOM
  ZCM=ZCM/NATOM
  DO I=1,NATOM
     X(I)=X(I)-XCM
     Y(I)=Y(I)-YCM
     Z(I)=Z(I)-ZCM
  ENDDO
  !
  !     Now the filling of the fourth coordinates
  !
  SD=SQRT(TWO*E4FILL/(K4DI*NATOM))
  SEED=314159
  DO I=1,NATOM
     A=SQRT(MINTWO*LOG(RANDOM(SEED)))
     B=TWO*PI*RANDOM(SEED)
     FDIM(I)=SD*A*COS(B)
  ENDDO
  RETURN
END SUBROUTINE FILL4
#endif /* (4defour)*/

