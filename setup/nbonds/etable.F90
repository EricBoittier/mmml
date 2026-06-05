module etablem
  use chm_kinds
  use dimens_fcm

#if KEY_CHEQ==1
  use cheq,only: qcg,cgmodel,   &                  
       DCH,SUMDCH,DDCH                         
#endif

  !CHARMM Element source/fcm/etable.fcm 1.1
#if KEY_NOMISC==0 /*etable_fcm*/
  !
  !      ETABLE.FCM  - Nonbond energy lookup tables
  !
  !     NTABTP       - Number of atom types for table usage
  !     NTABSQ       - Number of columns in the table (N*(N+1)/2)
  !     NTABLN       - Number of rows in the table
  !     NTABST       - Number of table sets (2 normal,3 hessian)
  !
  !     TABDEN       - Denominator for table definition
  !     TABRHO       - Exponent for table lookup
  !     TABHOM       - Offset for table lookup
  !
  !            R(i) = EXP ( RHO*(i-1) + HOM ) / DEN
  !
  !     IPTTB1      - Pointer for energy table
  !     IPTTB2      - Pointer for force table
  !     IPTTB3      - Pointer for second derivative table
  !     IPTRT       - Pointer for distance array
  !
  ! integers
  real(chm_real),allocatable,dimension(:,:) :: IPTTB1,ipttb2,ipttb3
  real(chm_real),allocatable,dimension(:) :: iptrt
  INTEGER NTABTP, NTABSQ, NTABLN, NTABST, ITBITC(MAXATB)
  ! reals
  real(chm_real)  TABDEN, TABHOM, TABRHO

  contains
    subroutine etable_init()
      ntabtp=0
      return
    end subroutine etable_init
#endif /* (etable_fcm)*/
    !


  end module etablem





#if KEY_NOMISC==0
SUBROUTINE REATBL(IUNIT,NATC)
  !
  !     THIS ROUTINE READS NONBOND LOOKUP TABLES
  !
  !     BERNARD R. BROOKS  9/11/83
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use etablem
  use ctitla
  use stream
  use memory
  use machutil,only:die
  use parallel,only: psnd4,psnd8
  implicit none
  !
  INTEGER IUNIT,NATC
  !
  !
  INTEGER NATX,J
  INTEGER ICNTRL(20)
  CHARACTER(len=4) :: HDRN='ETAB',HDR
  !
  IF(IOLEV > 0) THEN
     READ(IUNIT) HDR,ICNTRL
     IF(HDR /= HDRN) CALL WRNDIE(-2,'<REATBL>','HEADERS DONT MATCH')
     CALL RDTITL(TITLEB,NTITLB,IUNIT,-1)
     CALL WRTITL(TITLEB,NTITLB,OUTU,1)
  ENDIF
  !
#if KEY_PARALLEL==1
  CALL PSND4(ICNTRL,20)
#endif 
  !
  NTABTP=ICNTRL(1)
  NTABSQ=ICNTRL(2)
  IF(NTABSQ /= (NTABTP*(NTABTP+1))/2) CALL DIE
  NTABLN=ICNTRL(3)
  NTABST=ICNTRL(4)
  NATX=ICNTRL(5)
  IF(NATC /= NATX) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,33) NATC,NATX
33   FORMAT(' NATC=',I5,'  FROM FILE=',I5)
     CALL WRNDIE(-1,'<REATBL>','NUMBER OF ATOMS TYPES DONT MATCH')
  ENDIF
  !
  IF(IOLEV > 0) THEN
     READ(IUNIT) TABDEN,TABHOM,TABRHO
     DO J=1,MAXATB
        ITBITC(J)=0
     ENDDO
     READ(IUNIT) (ITBITC(J),J=1,NATX)

     call chmalloc('etable.src','REATBL','IPTTB1',NTABSQ,NTABLN,crl=IPTTB1)
     call chmalloc('etable.src','REATBL','IPTTB2',NTABSQ,NTABLN,crl=IPTTB2)
     call chmalloc('etable.src','REATBL','IPTTB3',NTABSQ,NTABLN,crl=IPTTB3)
     call chmalloc('etable.src','REATBL','IPTRT',NTABLN,crl=IPTRT)

     CALL REATB2(IUNIT,NTABSQ,NTABLN,NTABST, &
          IPTRT,IPTTB1,IPTTB2,IPTTB3, &
          TABDEN,TABHOM,TABRHO)
  ENDIF
  !
#if KEY_PARALLEL==1
  CALL PSND8(TABDEN,1)
  CALL PSND8(TABHOM,1)
  CALL PSND8(TABRHO,1)
  CALL PSND4(ITBITC,MAXATB)
  CALL PSND8m(IPTTB1,NTABSQ*NTABLN)
  CALL PSND8m(IPTTB2,NTABSQ*NTABLN)
  CALL PSND8m(IPTTB3,NTABSQ*NTABLN)
  CALL PSND8(IPTRT ,NTABLN)
#endif 
  !
  RETURN
END SUBROUTINE REATBL

SUBROUTINE REATB2(IUNIT,NPAIR,NLEN,NTAB,RTAB,TAB1,TAB2,TAB3, &
     DEN,HOM,RHO)
  !
  !     THIS ROUTINE READS THE ACTUAL TABLE ARRAYS  - BRB 9/10/83
  !
  use chm_kinds
  implicit none
  !
  INTEGER IUNIT,NPAIR,NLEN,NTAB
  real(chm_real) RTAB(NLEN)
  real(chm_real) TAB1(NPAIR,NLEN),TAB2(NPAIR,NLEN),TAB3(NPAIR,NLEN)
  real(chm_real) DEN,HOM,RHO
  !
  INTEGER I
  !
  IF(NTAB >= 1) READ(IUNIT) TAB1
  IF(NTAB >= 2) READ(IUNIT) TAB2
  IF(NTAB >= 3) READ(IUNIT) TAB3
  DO I=1,NLEN
     RTAB(I)=EXP(RHO*(I-1)+HOM)/DEN
  ENDDO
  RETURN
END SUBROUTINE REATB2

SUBROUTINE WRITBL(IUNIT,IPRINT,NATC)
  !
  !     THIS ROUTINE WRITES NONBOND LOOKUP TABLES
  !
  !     BERNARD R. BROOKS  9/11/83
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use etablem
  use ctitla
  use stream
  use memory
  !
  implicit none
  !
  INTEGER IUNIT,IPRINT,NATC
  !
  INTEGER J
  INTEGER ICNTRL(20)
  CHARACTER(len=4) :: HDRN='ETAB'
  !
  IF(IOLEV < 0) RETURN
  IF(IPRINT == 0) THEN
     !
     ! WRITE OUT A BINARY FILE
     !
     ICNTRL(1)=NTABTP
     ICNTRL(2)=NTABSQ
     ICNTRL(3)=NTABLN
     ICNTRL(4)=NTABST
     ICNTRL(5)=NATC
     ICNTRL(20)=18
     WRITE(IUNIT) HDRN,ICNTRL
     CALL WRTITL(TITLEA,NTITLA,IUNIT,-1)
     !
     WRITE(IUNIT) TABDEN,TABHOM,TABRHO
     WRITE(IUNIT) (ITBITC(J),J=1,NATC)

     CALL WRITB2(IUNIT,NTABSQ,NTABLN,NTABST, &
          IPTTB1,IPTTB2,IPTTB3)
     !
  ELSE
     !
     ! Write card file data
     CALL WRNDIE(0,'<WRITBL>','NO card format')
     !
  ENDIF
  RETURN
END SUBROUTINE WRITBL

SUBROUTINE WRITB2(IUNIT,NPAIR,NLEN,NTAB,TAB1,TAB2,TAB3)
  !
  !     THIS ROUTINE WRITES THE ACTUAL TABLE ARRAYS  - BRB 9/10/83
  !
  use chm_kinds
  implicit none
  INTEGER IUNIT,NPAIR,NLEN,NTAB
  real(chm_real) TAB1(NPAIR,NLEN),TAB2(NPAIR,NLEN),TAB3(NPAIR,NLEN)
  !
  IF(NTAB >= 1) WRITE(IUNIT) TAB1
  IF(NTAB >= 2) WRITE(IUNIT) TAB2
  IF(NTAB >= 3) WRITE(IUNIT) TAB3
  RETURN
END SUBROUTINE WRITB2

SUBROUTINE ETABLE(ENB,EEL,NATOM,X,Y,Z,DX,DY,DZ,EPS,CTOFNB, &
     INBLO,JNB,IAC,CG,NTABTP,NTABSQ,NTABLN,NTABST,ITBITC, &
     DENO,RHOM,DRHO,VEE,FT,DFT,R,DD1,IUPT,QSECD &
#if KEY_IMCUBES==1
     ,lbycbim  &    
#endif
     )
  !
  !   *********T A B L E   L O O K - U P   V E R S I O N ***********
  !
  !                                                M. PETTITT JUN 83
  !  THIS ROUTINE CALCULATES NONBONDED POTENTIALS AND FORCES
  !  WITH A TABLE LOOKUP. THE POINTS ARE GIVEN AT THE LOG FFT MESH
  !  AND THEN A LINEAR INTERPOLATION IS DONE TO GET V OR F AT
  !  THE DESIRED RIJ.
  !                     CONVERTED TO CHARMM - BRB 9/11/83
  !
  !     R(NTABLN)=THE LNFFT GRID
  !     VEE(NTABSQ,NTABLN)=THE POTENTIAL (OF MEAN FORCE)
  !     FT (NTABSQ,NTABLN)=THE DERIVATIVE OF THE POTENTIAL (OF MEAN FORCE)
  !     DFT(NTABSQ,NTABLN)=THE SECOND DERIVATIVE OF THE POTENTIAL
  !                                                                      C
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !

#if KEY_CHEQ==1
  use cheq,only: qcg,cgmodel,   &                  
       DCH,SUMDCH,DDCH                         
#endif

  use chm_kinds
  use dimens_fcm
  use stream,only: OUTU
  use consta
#if KEY_DIMB==1
  use dimb  
#endif
  implicit none
  !
  real(chm_real) ENB,EEL
  INTEGER NATOM
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) EPS,CTOFNB,CG(*)
  INTEGER INBLO(*)
  INTEGER JNB(*),IAC(*)
  INTEGER NTABTP,NTABSQ,NTABLN,NTABST,ITBITC(*)
  real(chm_real) DENO,RHOM,DRHO
  real(chm_real) VEE(NTABSQ,NTABLN),FT(NTABSQ,NTABLN), &
       DFT(NTABSQ,NTABLN)
  real(chm_real) R(*),DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
#if KEY_IMCUBES==1
  logical lbycbim
#endif 
  !
  ! SAPATEL
#if KEY_CHEQ==1
  real(chm_real) HIJ
#endif 
  ! SAPATEL
  INTEGER IOFF(MAXATC)
  real(chm_real) CTNBSQ,CGF,CGI,DXI,DYI,DZI,RIJ2,RIJ,VEL,DENOM,VM,VB
  real(chm_real) E12,FM,FB,DF,DDF,AXX,AYY,AZZ,AXY,AXZ,AYZ
  INTEGER I,ITEMP,NB,IMAX,NPAIR,IACI,ICNA,K,J
  INTEGER IACJ,IPC,IJ,JJ,II,IADD
  LOGICAL LSECD,IJTEST
  !
  !     DATA RHOM,DRHO/-5.12D0,.02D0/
  !
  LSECD=QSECD
  CTNBSQ=CTOFNB**2
  DO I=1,NTABTP
     IOFF(I)=(I*(I-1))/2
  ENDDO
  !
  IF(LSECD.AND.NTABST < 3) CALL WRNDIE(-3,'<ETABLE>', &
       'THREE TABLES NEEDED FOR SECOND DER.')
  !
  ITEMP=0
  NB=0
  EEL=0.0
  ENB=0.0
  IF(EPS <= 0) THEN
     CGF=0.0
  ELSE
     CGF=CCELEC/EPS
  ENDIF
  !
  IMAX=NATOM !-1
!  write (*,*) 'IMAX=',IMAX
  loop300: DO I=1,IMAX
#if KEY_IMCUBES==1
     if(lbycbim)itemp=inblo(I+natom)
#endif 
     NPAIR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF (NPAIR == 0) cycle loop300
     IACI=ITBITC(IAC(I))
     IF(IACI > NTABTP .OR. IACI <= 0) CALL WRNDIE(-3,'<ETABLE>', &
          'ATOM WITH UNDEFINED CODE FOUND')
     CGI=CG(I)*CGF
     ICNA=IOFF(IACI)
 !    write (*,*) 'i,npair=',i,npair
     !
     loop200: DO K=1,NPAIR
        NB=NB+1
        IF(JNB(NB) < 0) THEN
           J=-JNB(NB)
        ELSE
           J=JNB(NB)
        ENDIF
        DXI=X(I)-X(J)
        DYI=Y(I)-Y(J)
        DZI=Z(I)-Z(J)
        RIJ2=DXI*DXI+DYI*DYI+DZI*DZI
!           write (*,'(a,i4,i4,a,f10.4)') 'i,j=',i,j,' rij=',sqrt(rij2)
!           write (*,'(a,3f10.4)') 'xyz(i)=',x(i),y(i),z(i)
!           write (*,'(a,3f10.4)') 'xyz(j)=',x(j),y(j),z(j)
        IF (RIJ2 > CTNBSQ) cycle loop200
        RIJ=SQRT(RIJ2)
        ! SAPATEL
#if KEY_CHEQ==1
        IF (QCG.AND.(CGMODEL == 1)) THEN
           HIJ=1.0/RIJ
           VEL=CGI*CG(J)*HIJ
           DCH(I)=DCH(I)+HIJ*CGF*CG(J)
           DCH(J)=DCH(J)+HIJ*CGI
        ELSE
#endif 
           ! SAPATEL
           VEL=CGI*CG(J)/RIJ
           ! SAPATEL
#if KEY_CHEQ==1
        ENDIF
        IF (.NOT.QCG.OR.(CGMODEL == 1)) THEN
#endif 
           ! SAPATEL
           EEL=EEL+VEL
           ! SAPATEL
#if KEY_CHEQ==1
        ENDIF
#endif 
        ! SAPATEL

        IACJ=ITBITC(IAC(J))
        !
        !   NOW FOR USING THE TABLE
        !
        IF(IACI > IACJ) THEN
           IPC=ICNA+IACJ
        ELSE
           IPC=IOFF(IACJ)+IACI
        ENDIF
        !
        !  GET THE RELAVENT LOWER INDEX FROM THE GRID
        !
        IJ=INT((LOG(DENO*RIJ)-RHOM)/DRHO+1.)
        IF(IJ <= 0 .OR. IJ >= NTABLN) then
           write (OUTU,'(a,2i5,a,f8.3)') 'i,j=',i,j,' rij=',rij
           write (OUTU,'(a,3f8.3)') '(i) x,y,z=',x(i),y(i),z(i)
           write (OUTU,'(a,3f8.3)') '(j) x,y,z=',x(j),y(j),z(j)
           CALL WRNDIE(-3,'<ETABLE>', &
             'LOOKUP ELEMENT OUT OF RANGE')
        endif
        !  WELL NEED THIS DENOMINATOR
        DENOM=R(IJ)-R(IJ+1)
        !  SET UP LINEAR INTERPOLATER
        VM=(VEE(IPC,IJ)-VEE(IPC,IJ+1))/DENOM
        VB=VEE(IPC,IJ)-VM*R(IJ)
        !  NOW FOR THE TOTAL POTENTAL
        E12=VM*RIJ+VB-VEL
!        write (*,*) 'e12=',e12
        ENB=ENB+E12
        !
        !   NOW FOR THE DERIVATIVES OF VEE
        !
        FM=(FT(IPC,IJ)-FT(IPC,IJ+1))/DENOM
        FB=FT(IPC,IJ)-FM*R(IJ)
        DF=FM+FB/RIJ
        DX(J)=DX(J)-DXI*DF
        DX(I)=DX(I)+DXI*DF
        DY(J)=DY(J)-DYI*DF
        DY(I)=DY(I)+DYI*DF
        DZ(J)=DZ(J)-DZI*DF
        DZ(I)=DZ(I)+DZI*DF
        !
        IF(LSECD) THEN
           FM=(DFT(IPC,IJ)-DFT(IPC,IJ+1))/DENOM
           FB=DFT(IPC,IJ)-FM*R(IJ)
           DDF=FM*RIJ+FB
           !
           DDF=(DDF-DF)/RIJ2
           AXX=DXI*DXI*DDF+DF
           AYY=DYI*DYI*DDF+DF
           AZZ=DZI*DZI*DDF+DF
           AXY=DXI*DYI*DDF
           AXZ=DXI*DZI*DDF
           AYZ=DYI*DZI*DDF
           !
#if KEY_DIMB==1
           IF(QCMPCT) THEN
              CALL ETACMP(I,J,AXX,AXY,AXZ,AYY,AYZ,AZZ,DD1, &
                   PINBCM,PJNBCM)
           ELSE
#endif /*  DIMB*/

              II=3*I-2
              JJ=3*J-2
              IJTEST=(J < I)
              !
              IADD=IUPT(II)+II
              DD1(IADD)=DD1(IADD)+AXX
              IADD=IUPT(II+1)+II+1
              DD1(IADD)=DD1(IADD)+AYY
              IADD=IUPT(II+2)+II+2
              DD1(IADD)=DD1(IADD)+AZZ
              IADD=IUPT(II)+II+1
              DD1(IADD)=DD1(IADD)+AXY
              IADD=IUPT(II)+II+2
              DD1(IADD)=DD1(IADD)+AXZ
              IADD=IUPT(II+1)+II+2
              DD1(IADD)=DD1(IADD)+AYZ
              !
              IADD=IUPT(JJ)+JJ
              DD1(IADD)=DD1(IADD)+AXX
              IADD=IUPT(JJ+1)+JJ+1
              DD1(IADD)=DD1(IADD)+AYY
              IADD=IUPT(JJ+2)+JJ+2
              DD1(IADD)=DD1(IADD)+AZZ
              IADD=IUPT(JJ)+JJ+1
              DD1(IADD)=DD1(IADD)+AXY
              IADD=IUPT(JJ)+JJ+2
              DD1(IADD)=DD1(IADD)+AXZ
              IADD=IUPT(JJ+1)+JJ+2
              DD1(IADD)=DD1(IADD)+AYZ
              !
              IF(IJTEST) THEN
                 IADD=IUPT(JJ)+II
                 DD1(IADD)=DD1(IADD)-AXX
                 IADD=IUPT(JJ+1)+II+1
                 DD1(IADD)=DD1(IADD)-AYY
                 IADD=IUPT(JJ+2)+II+2
                 DD1(IADD)=DD1(IADD)-AZZ
                 IADD=IUPT(JJ)+II+1
                 DD1(IADD)=DD1(IADD)-AXY
                 IADD=IUPT(JJ+1)+II
                 DD1(IADD)=DD1(IADD)-AXY
                 IADD=IUPT(JJ)+II+2
                 DD1(IADD)=DD1(IADD)-AXZ
                 IADD=IUPT(JJ+2)+II
                 DD1(IADD)=DD1(IADD)-AXZ
                 IADD=IUPT(JJ+1)+II+2
                 DD1(IADD)=DD1(IADD)-AYZ
                 IADD=IUPT(JJ+2)+II+1
                 DD1(IADD)=DD1(IADD)-AYZ
              ELSE
                 IADD=IUPT(II)+JJ
                 DD1(IADD)=DD1(IADD)-AXX
                 IADD=IUPT(II+1)+JJ+1
                 DD1(IADD)=DD1(IADD)-AYY
                 IADD=IUPT(II+2)+JJ+2
                 DD1(IADD)=DD1(IADD)-AZZ
                 IADD=IUPT(II+1)+JJ
                 DD1(IADD)=DD1(IADD)-AXY
                 IADD=IUPT(II)+JJ+1
                 DD1(IADD)=DD1(IADD)-AXY
                 IADD=IUPT(II+2)+JJ
                 DD1(IADD)=DD1(IADD)-AXZ
                 IADD=IUPT(II)+JJ+2
                 DD1(IADD)=DD1(IADD)-AXZ
                 IADD=IUPT(II+2)+JJ+1
                 DD1(IADD)=DD1(IADD)-AYZ
                 IADD=IUPT(II+1)+JJ+2
                 DD1(IADD)=DD1(IADD)-AYZ
              ENDIF

#if KEY_DIMB==1
           ENDIF
#endif /*  DIMB*/

        ENDIF
     enddo loop200
  enddo loop300
  !
  RETURN
end SUBROUTINE ETABLE
#endif 

SUBROUTINE NULL_TB
  return
END SUBROUTINE NULL_TB

