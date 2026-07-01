module cmapm
#if KEY_CMAP==1 /*cmap_main*/
  use chm_kinds
  use chm_types
  implicit none

  type cmap_t
     type(chm_ptr_2d) :: grid(4)
  end type cmap_t

  ! MAXCTP - Maximum number of cross-term maps
  ! NCTP             Number of cross-term maps
  ! KCTP      Cmap   Key for searching for double torsion
  ! GSCTP     Cmap   Grid size
  ! MCTP      Cmap   Grid map
  ! ICTPCNT   Cmap   Count of cross-term use
  ! CTPA     8,Cmap  Atom types for each double torsion
  integer,parameter :: MAXCTP = 100
  type(cmap_t),save :: MCTP(MAXCTP)
  integer(chm_int8) :: kctp(maxctp)
  integer :: NCTP, GSCTP(MAXCTP), ICTPCNT(MAXCTP)
#if KEY_FLEXPARM==1
  integer :: CTPA(MAXCTP,8)  
#endif

contains
  !
  ! These routines support cross-term maps for two dihedral angles.
  !

  subroutine cmap_allocate_table(map,num)
    use memory
    implicit none
    integer,intent(in) :: num
    type(cmap_t) :: map
    integer :: i

    do i = 1, 4
       call chmalloc("ecmap.src", "cmap_allocate_table", "map%grid%a", &
            num, num, crlp=map%grid(i)%a)
       map%grid(i)%len1 = num
       map%grid(i)%len2 = num
    enddo
    return
  end subroutine cmap_allocate_table

  SUBROUTINE AINDX_CMAP(invmap, array, n)
    implicit none
    integer :: n, invmap(*)
    type(cmap_t) :: array(*)
    type(cmap_t) :: work(n)
    integer :: i

    do i = 1, n
       work(i) = array(invmap(i))
    enddo
    array(1:n) = work(1:n)
  END SUBROUTINE AINDX_CMAP

  SUBROUTINE RDCMAP(COMLYN,MXCMSZ,COMLEN,QPRINT,IUNIT,NUM,GMAP)
    !
    !     This routine reads a cross-term map from a
    !     parameter input stream
    !
    !     The arguments are as follows:
    !
    !     UNIT   input unit (open)
    !     NUM    number of elements to be read
    !     GMAP   data array where elements are stored

    use number
    use stream
    use string, only:nextf,trime
    implicit none

    CHARACTER(len=*) COMLYN
    INTEGER COMLEN,MXCMSZ
    LOGICAL QPRINT
    LOGICAL EOF

    INTEGER IUNIT,NUM
    real(chm_real) GMAP(num,num)

    INTEGER NREAD1,nread2
    INTEGER i,count

    NREAD1=1
    NREAD2=1
    EOF=.FALSE.
    count=0

9111 CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.FALSE., &
         QPRINT,'PARRDR> ')

    IF (EOF) GOTO 9113

    CALL TRIME(COMLYN,COMLEN)

    IF (COMLEN.LE.0) GOTO 9111

9112 GMAP(NREAD1,nread2)=NEXTF(COMLYN,COMLEN)
    NREAD1=NREAD1+1
    count=count+1
    if(nread1 > num)then
       nread2=nread2+1
       nread1=1
    endif
    IF (COMLEN.GT.0 .AND. count.LE.NUM*num) GOTO 9112
    IF (count.lt.NUM*num) GOTO 9111

9113 CONTINUE

    IF (count.lt.NUM*num) THEN
       IF (WRNLEV.GE.-1) THEN
          WRITE(OUTU,9211)
9211      FORMAT('****** Warning from PARRDR ******',/, &
               ' Premature end of parameter file while reading CMAP')
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE RDCMAP

  SUBROUTINE CMAPSPL(dx,y,n,u,y2)
    ! spline routines for CMAP

    use number
    implicit none

    real(chm_real) dx
    real(chm_real) y(*)
    INTEGER n
    real(chm_real) u(*)
    real(chm_real) y2(*)

    INTEGER i
    real(chm_real) pinv,p
    real(chm_real) v,dxinv

    Y2(1)=ZERO
    U(1)=ZERO

    dxinv=1.0/dx
    DO i=2,N-1
       pinv=1.0/(y2(i-1)+FOUR)
       y2(i)=-pinv
       u(i)=((SIX*y(i+1)-TWELVE*y(i)+SIX*y(i-1))*dxinv*dxinv- &
            u(i-1))*pinv
    ENDDO

    y2(n)=ZERO
    DO i=n-1,1,-1
       y2(i)=y2(i)*y2(i+1)+u(i)
    ENDDO

    RETURN
  END SUBROUTINE CMAPSPL

  SUBROUTINE CMAPSPI(XMIN,DX,YA,Y2A,X,Y,Y1)

    use number
    implicit none

    real(chm_real) xmin,dx,ya(*),y2a(*),x,y,y1
    INTEGER inx
    real(chm_real) x1,x2,a,b

    inx=NINT((x-xmin)/dx-HALF)+1

    a=(xmin+DBLE(inx)*dx-x)/dx
    b=(x-xmin-DBLE(inx-1)*dx)/dx

    y=a*ya(inx)+b*ya(inx+1)+ &
         ((a*a*a-a)*y2a(inx)+(b*b*b-b)*y2a(inx+1))*(dx*dx)/SIX
    y1=(ya(inx+1)-ya(inx))/dx- &
         (THREE*a*a-ONE)/SIX*dx*y2a(inx)+ &
         (THREE*b*b-ONE)/SIX*dx*y2a(inx+1)

    RETURN
  END SUBROUTINE CMAPSPI

  SUBROUTINE SETCMAP(NUM,XM,GMAP,DX)
    !     set derivatives from bicubic spline fit to ensure C2 continuity

    use number
    implicit none

    INTEGER NUM,XM
    type(cmap_t) :: gmap
    real(chm_real) DX
    !local
    real(chm_real) TGMAP(NUM+XM+XM,NUM+XM+XM)
    real(chm_real) T2(NUM+XM+XM,NUM+XM+XM)
    real(chm_real) U(NUM+XM+XM),U2(NUM+XM+XM)
    real(chm_real) YYTMP(NUM+XM+XM),Y1TMP(NUM+XM+XM)
    real(chm_real) PHI,PSI
    real(chm_real) xmin
    real(chm_real) v,v1,v2,v12
    INTEGER i,j,k,ii,jj
    !
    xmin=-ONE8TY-xm*dx
    DO i=1,num+xm+xm
       ii=MOD(i+NUM-xm-1,NUM)+1
       DO j=1,num+xm+xm
          jj=MOD(j+NUM-xm-1,NUM)+1
          TGMAP(i,j) = GMAP%grid(1)%a(ii,jj)
       ENDDO
    ENDDO

    DO i=1,num+xm+xm
       CALL CMAPSPL(dx,tgmap(1,i),num+xm+xm,u(1),t2(1,i))
    ENDDO

    DO i=1+xm,num+xm
       phi=(i-xm-1)*dx-ONE8TY
       DO j=1+xm,num+xm
          psi=(j-xm-1)*dx-ONE8TY
          DO k=1,num+xm+xm
             CALL CMAPSPI(xmin,dx,tgmap(1,k), &
                  t2(1,k),psi,yytmp(k),y1tmp(k))
          ENDDO

          CALL CMAPSPL(dx,yytmp(1),num+xm+xm,u(1),u2(1))
          CALL CMAPSPI(xmin,dx,yytmp(1),u2(1),phi,v,v1)
          CALL CMAPSPL(dx,y1tmp(1),num+xm+xm,u(1),u2(1))
          CALL CMAPSPI(xmin,dx,y1tmp(1),u2(1),phi,v2,v12)
          GMAP%grid(2)%a(j-xm,i-xm) = v1
          GMAP%grid(3)%a(j-xm,i-xm) = v2
          GMAP%grid(4)%a(j-xm,i-xm) = v12
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE SETCMAP

  SUBROUTINE GCSTUP2(NGRD,GGRD,TY,TY1,TY2,TY12,IP1,IP2)

    use number
    implicit none

    integer, intent(in) :: NGRD
    type(cmap_t), intent(in) :: GGRD
    real(chm_real), dimension(4), intent(out) :: TY, TY1, TY2, TY12
    integer, intent(in) :: IP1, IP2

    INTEGER IP1P1,IP2P1

    IP1P1 = mod(IP1, NGRD) + 1
    IP2P1 = mod(IP2, NGRD) + 1

    ! see also epmf:HBM2TY
    TY(1)=GGRD%grid(1)%a(IP2,IP1)
    TY(2)=GGRD%grid(1)%a(IP2,IP1P1)
    TY(3)=GGRD%grid(1)%a(IP2P1,IP1P1)
    TY(4)=GGRD%grid(1)%a(IP2P1,IP1)

    TY1(1)=GGRD%grid(2)%a(IP2,IP1)
    TY1(2)=GGRD%grid(2)%a(IP2,IP1P1)
    TY1(3)=GGRD%grid(2)%a(IP2P1,IP1P1)
    TY1(4)=GGRD%grid(2)%a(IP2P1,IP1)

    TY2(1)=GGRD%grid(3)%a(IP2,IP1)
    TY2(2)=GGRD%grid(3)%a(IP2,IP1P1)
    TY2(3)=GGRD%grid(3)%a(IP2P1,IP1P1)
    TY2(4)=GGRD%grid(3)%a(IP2P1,IP1)

    TY12(1)=GGRD%grid(4)%a(IP2,IP1)
    TY12(2)=GGRD%grid(4)%a(IP2,IP1P1)
    TY12(3)=GGRD%grid(4)%a(IP2P1,IP1P1)
    TY12(4)=GGRD%grid(4)%a(IP2P1,IP1)

    RETURN
  END SUBROUTINE GCSTUP2

SUBROUTINE GCSCF(TY,TY1,TY2,TY12,GRES1,GRES2,TC)

  use chm_kinds
  use number
  implicit none

  real(chm_real) TY(4)
  real(chm_real) TY1(4)
  real(chm_real) TY2(4)
  real(chm_real) TY12(4)
  real(chm_real) GRES1,GRES2
  real(chm_real) TC(4,4)

  INTEGER i,j,k,in
  real(chm_real) XX,TX(16)

  INTEGER WT(16,16)

  SAVE WT
  DATA WT /1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, &
          -3, 0, 0, 3, 0, 0, 0, 0,-2, 0, 0,-1, 0, 0, 0, 0, &
           2, 0, 0,-2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, &
           0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, &
           0, 0, 0, 0,-3, 0, 0, 3, 0, 0, 0, 0,-2, 0, 0,-1, &
           0, 0, 0, 0, 2, 0, 0,-2, 0, 0, 0, 0, 1, 0, 0, 1, &
          -3, 3, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
           0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0, &
           9,-9, 9,-9, 6, 3,-3,-6, 6,-6,-3, 3, 4, 2, 1, 2, &
          -6, 6,-6, 6,-4,-2, 2, 4,-3, 3, 3,-3,-2,-1,-1,-2, &
           2,-2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
           0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0, &
          -6, 6,-6, 6,-3,-3, 3, 3,-4, 4, 2,-2,-2,-2,-1,-1, &
           4,-4, 4,-4, 2, 2,-2,-2, 2,-2,-2, 2, 1, 1, 1, 1/

  DO i=1,4
     TX(i)=TY(i)
     TX(i+4)=TY1(i)*GRES1
     TX(i+8)=TY2(i)*GRES2
     TX(i+12)=TY12(i)*GRES1*GRES2
  ENDDO

  IN=1
  DO i=1,4
     DO j=1,4
        XX=ZERO
        DO k=1,16
           XX=XX+DBLE(WT(k,in))*TX(k)
        ENDDO
        IN=IN+1
        TC(i,j)=XX
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE GCSCF

!!!!!!!!------------------------------------------------------------------

SUBROUTINE APPEND_BMAP_TO_CMAP( )        !( BMAP_DIM,BMAP_UNT, BMAP_IND,BMAP_LAM)
    use chm_kinds
    use dimens_fcm
    use code      !!ICCT
    use psf
    use number
    use stream
    use string
    use comand
    use bpcmap_mod
    implicit none

    integer         ::  I,J,K
    INTEGER         ::  I1,J1,K1,L1, I2,J2,K2,L2
    INTEGER         ::  XM
   
300 FORMAT(/,6X,A,I6)                              
    WRITE(OUTU,300) 'Maximum number of cross-term maps is : ', MAXCTP
    WRITE(OUTU,300) 'Number of cross-term maps (CMAP) is : ',nctp
    WRITE(OUTU,300) 'Number of BPC-term maps is : ',nbtp

    DO I=2,MAXCTP
       IF(I1CT(I-1).NE.0 .AND. I1CT(I).EQ.0)  I1=I
       IF(J1CT(I-1).NE.0 .AND. J1CT(I).EQ.0)  J1=I
       IF(K1CT(I-1).NE.0 .AND. K1CT(I).EQ.0)  K1=I
       IF(L1CT(I-1).NE.0 .AND. L1CT(I).EQ.0)  L1=I
             
       IF(I2CT(I-1).NE.0 .AND. I2CT(I).EQ.0)  I2=I
       IF(J2CT(I-1).NE.0 .AND. J2CT(I).EQ.0)  J2=I
       IF(K2CT(I-1).NE.0 .AND. K2CT(I).EQ.0)  K2=I
       IF(L2CT(I-1).NE.0 .AND. L2CT(I).EQ.0)  L2=I
    END DO

200 FORMAT(/,6X,A)
    IF( (I1.NE.J1) .OR. (J1.NE.K1) .OR.(K1.NE.L1) & 
         .OR. (L1.NE.I2) .OR. & 
        (I2.NE.J2) .OR. (J2.NE.K2) .OR.(K2.NE.L2) ) THEN
         WRITE(OUTU,210) 'ERROR IN CMAP CROSS ATOM INDEX: ENERGY/ECMAP.SRC:APPEND_BMAP_TO_CMAP'
         CALL WRNDIE(-1,'<GEO>','ERROR IN CMAP CROSS ATOM INDEX')
    END IF

210 FORMAT(/,6X,A,I4,A, 8(I4))   
   
    DO I=1, I1-1
        WRITE(OUTU,210) 'THE CMAP DIHIDRAL INDEX ',K+I, ' : ',&
              I1CT(K+I), J1CT(K+I), K1CT(K+I), L1CT(K+I), &
              I2CT(K+I), J2CT(K+I), K2CT(K+I), L2CT(K+I)
    END DO

    K=I1-1
    DO I=1,NBTP
       I1CT(K+I) = BMAP_IND(I,1)
       J1CT(K+I) = BMAP_IND(I,2)
       K1CT(K+I) = BMAP_IND(I,3)
       L1CT(K+I) = BMAP_IND(I,4)
   
       I2CT(K+I) = BMAP_IND(I,5)
       J2CT(K+I) = BMAP_IND(I,6)
       K2CT(K+I) = BMAP_IND(I,7)
       L2CT(K+I) = BMAP_IND(I,8)
    END DO


    DO I=1, NBTP
       WRITE(OUTU,210) 'THE BPCMAP DIHIDRAL INDEX ',K+I, ' : ',&
             I1CT(K+I), J1CT(K+I), K1CT(K+I), L1CT(K+I), &
             I2CT(K+I), J2CT(K+I), K2CT(K+I), L2CT(K+I)
    END DO
        
    DO I=1, NBTP
       GSCTP(NCTP+I)=BMAP_DIM(I)
       CALL  cmap_allocate_table(mctp(NCTP+I),BMAP_DIM(I) )
       CALL  RDCMAP(COMLYN,MXCMSZ,COMLEN,.FALSE.,BMAP_UNT(I), & 
                    BMAP_DIM(I), mctp(NCTP+I)%grid(1)%A)
      
       DO J=1, BMAP_DIM(I)
          DO K=1, BMAP_DIM(I)
             mctp(nctp+I)%grid(1)%A(J,K)=mctp(nctp+I)%grid(1)%A(J,K)*BMAP_LAM(I)
          END DO
       END DO
            
       XM= BMAP_DIM(I)/2
       CALL SETCMAP( BMAP_DIM(I), XM, mctp(nctp+I),THR6TY/BMAP_DIM(I) )
       
       REWIND( BMAP_UNT(I) ) 
   
    END DO
     
    RETURN
END SUBROUTINE APPEND_BMAP_TO_CMAP

!!!==========================================================

SUBROUTINE ECMAP(EC,I1P,J1P,K1P,L1P,I2P,J2P,K2P,L2P, &
     ICPT,NCT, &
     DX,DY,DZ,X,Y,Z, &
     QECONTX,ECONTX,ICONHP,ISKP,DD1,IUPT,QSECD &
     )

  !-----------------------------------------------------------------------
  !     CALCULATES DIHEDRAL CROSS-TERMS BY BICUBIC INTERPOLATION
  !     FROM GRID-BASED DISCRETE MAP
  !
  !     FIRST DERIVATIVES ARE ADDED TO DX, DY, DZ AND IF
  !     QSECD SECOND DERIVATIVES TO DD1.
  !
  !     The parameters of the routine are:
  !
  !     EC          <- Cross-Term Energy
  !     I1P,J1P,K1P,L1P(phi) -> atom number for first dihedral
  !     I2P,J2P,K2P,L2P(phi) -> atom number for second dihedral
  !     ICPT(phi)   -> parameter number associated with cross-term type
  !     NCT         -> Number of cross-terms
  !     DX,DY,DZ(atom) <-> Force matrices
  !     X,Y,Z(atom) -> Coordinate matrices
  !     QECONT      -> Flag for energy/atom statistics
  !     ECONT(atom) <- matrix of energy/atom
  !     ICONHP      -> Flag to activate the skipping procedure
  !     ISKP(phi)   -> matrix of flag to skip dihedral. Skip if ISKIP.ne.0
  !     DD1        <-> Second derivative matrix (upper triangle)
  !     IUPT(atom)  -> Index function for DD1
  !     QSECD       -> Second derivative flag.
  !
  !
  !     By Michael Feig 2002
  !
  !     Milan Hodoscek, June 2007:
  !       - Fixed/Added some basic functionality for BLOCK.
  !       - No BLOCK&CMAP for: LDM, energy/genborn.src, ##DOCK yet
  !         Someone (MF?) please fix the rest - see ##NOTDEF
  !
  !
  !   Hiqmet Kamberaj, November 2007:
  !   Tsallis scaling of the CMAP term added
  !
  !     Ryan Hayes, July 2018:
  !       - Added some msld functionality
#if KEY_TSALLIS==1
  use tsallis_module,only : tdx,tdy,tdz,ebtors, &
       tsndf,qtalpha,tsalpha,qttsall,iascale
#endif 
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use parallel
  use block_fcm
  use lambdam
  use dimb
  use consta
  use econtmod
  use stream
  use bpcmap_mod
#if KEY_GENETIC==1
  use galgor,only: qga_ener    
#endif
  use chutil,only:atomid
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
  use domdec_bonded,only:ncmaptbl,cmaptbl
#endif
#if KEY_GRAPE==1
    use grape,only:lfmm
#endif
  !
  implicit none
  !
  real(chm_real) EC
  INTEGER I1P(*),J1P(*),K1P(*),L1P(*), &
       I2P(*),J2P(*),K2P(*),L2P(*), &
       ICPT(*)
  INTEGER NCT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  INTEGER ICONHP
  INTEGER ISKP(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  !
#if KEY_TSALLIS==1
  ! local variables
  INTEGER       :: FI1,FJ1,FK1,FL1, sum1
  INTEGER       :: FI2,FJ2,FK2,FL2, sum2
  LOGICAL       :: TSALLISMD
#endif 
  !
  real(chm_real) MRES,MRES1
  real(chm_real) FX1,FY1,FZ1,GX1,GY1,GZ1,HX1,HY1,HZ1
  real(chm_real) FX2,FY2,FZ2,GX2,GY2,GZ2,HX2,HY2,HZ2
  real(chm_real) AX1,AY1,AZ1,BX1,BY1,BZ1
  real(chm_real) AX2,AY2,AZ2,BX2,BY2,BZ2
  real(chm_real) RA21,RB21,RA2R1,RB2R1,RG21,RG1,RGR1,RGR21
  real(chm_real) RA22,RB22,RA2R2,RB2R2,RG22,RG2,RGR2,RGR22
  real(chm_real) RABR1,CP1,SP1,RABR2,CP2,SP2
  real(chm_real) PHI1,PHI2,XPHI1,XPHI2
  real(chm_real) TY(4),TY1(4),TY2(4),TY12(4)
  real(chm_real) TC(4,4),TT,TU
  real(chm_real) E,DF1,DF2

  INTEGER I1,J1,K1,L1,IC,I2,J2,K2,L2,ICT
  LOGICAL NOCONS,QAFIRST
  INTEGER NWARN,NWARNX
  INTEGER IPHI1,IPHI2
  INTEGER KK1,KK2
  CHARACTER(len=8) SIDDNI1,RIDDNI1,RESDNI1,ACDNI1
  CHARACTER(len=8) SIDDNJ1,RIDDNJ1,RESDNJ1,ACDNJ1
  CHARACTER(len=8) SIDDNI2,RIDDNI2,RESDNI2,ACDNI2
  CHARACTER(len=8) SIDDNJ2,RIDDNJ2,RESDNJ2,ACDNJ2
  CHARACTER(len=8) SIDDNK1,RIDDNK1,RESDNK1,ACDNK1
  CHARACTER(len=8) SIDDNL1,RIDDNL1,RESDNL1,ACDNL1
  CHARACTER(len=8) SIDDNK2,RIDDNK2,RESDNK2,ACDNK2
  CHARACTER(len=8) SIDDNL2,RIDDNL2,RESDNL2,ACDNL2

  real(chm_real) GAA1,GBB1,FG1,HG1,FGA1,HGB1
  real(chm_real) FGRG21,HGRG21,DFRG31
  real(chm_real) DFX1,DFY1,DFZ1,DHX1,DHY1,DHZ1,DGX1,DGY1,DGZ1
  real(chm_real) DTFX1,DTFY1,DTFZ1
  real(chm_real) DTHX1,DTHY1,DTHZ1,DTGX1,DTGY1,DTGZ1
  real(chm_real) GAFX1,GAFY1,GAFZ1,GBHX1,GBHY1,GBHZ1
  real(chm_real) FAGX1,FAGY1,FAGZ1,HBGX1,HBGY1,HBGZ1
  real(chm_real) GAA2,GBB2,FG2,HG2,FGA2,HGB2
  real(chm_real) FGRG22,HGRG22,DFRG32
  real(chm_real) DFX2,DFY2,DFZ2,DHX2,DHY2,DHZ2,DGX2,DGY2,DGZ2
  real(chm_real) DTFX2,DTFY2,DTFZ2
  real(chm_real) DTHX2,DTHY2,DTHZ2,DTGX2,DTGY2,DTGZ2
  real(chm_real) GAFX2,GAFY2,GAFZ2,GBHX2,GBHY2,GBHZ2
  real(chm_real) FAGX2,FAGY2,FAGZ2,HBGX2,HBGY2,HBGZ2
  real(chm_real) DDFGH(45)
  INTEGER II,JJ,KK,LL,IADD
  LOGICAL IJTEST,IKTEST,ILTEST,JKTEST,JLTEST,KLTEST
  real(chm_real) DDF1,DDF2,DDF12
  real(chm_real) FAC

  real(chm_real) TP,TPHI1,TPHI2,CPHI1,CPHI2

  INTEGER INX1(4),INX2(4), III1, III2, II2, II1
  real(chm_real) DTT1MX(4),DTT1MY(4),DTT1MZ(4)
  real(chm_real) DTT2MX(4),DTT2MY(4),DTT2MZ(4)
  real(chm_real) DTT1PX(4),DTT1PY(4),DTT1PZ(4)
  real(chm_real) DTT2PX(4),DTT2PY(4),DTT2PZ(4)

  !      real(chm_real) RABR,CP,AP,SP,E,DF,DDF,CA,SA,ARG,APR
  !      INTEGER NWARN,NWARNX,ICT,I,J,K,L,IC,IPER,NPER
  !      LOGICAL LREP,NOCONS,QAFIRST

  !
#if KEY_BLOCK==1
  INTEGER IBL1, JBL1, KKK1, LLL1, KDOC
  INTEGER IBL2, JBL2, KKK2, LLL2
  real(chm_real)  coef
  real(chm_real)  COEF1,DOCFI1,DOCFJ1,DOCFK1,DOCFJ11,DOCFK11,DOCFL1
  real(chm_real)  COEF2,DOCFI2,DOCFJ2,DOCFK2,DOCFJ12,DOCFK12,DOCFL2
!ldm
  real(chm_real) UNSCALE,FALPHA
  real(chm_real) DFORG
!ldm
#endif /*  BLOCK*/
  !
  real(chm_real), parameter :: RXMIN=0.005D0, RXMIN2=0.000025D0
  !
  INTEGER loopFirst,loopLast,loopinc
  integer icta

  loopFirst = 1
  loopinc = 1
  loopLast = nct
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1 /*domdec*/
  if (q_domdec) then
     loopFirst = 1
     loopinc = 1
     loopLast = ncmaptbl
  else
#endif /* (domdec)*/
     loopfirst=MYNODP
     loopinc = NUMNOD
#if KEY_DOMDEC==1
  endif  
#endif
#else /**/
#if KEY_GENETIC==1
  If(qGA_Ener) loopFirst = Int(EC)
#endif 
#endif 
  EC=ZERO
  NOCONS=(ICONHP.GT.0)
  if (loopLast <= 0) return
  NWARN=0
  NWARNX=0
  QAFIRST=.TRUE.
  !
#if KEY_TSALLIS==1
  EBTORS(2)=ZERO   
#endif
  !
#if KEY_GRAPE==1
  if(lfmm)then
     loopFirst = 1
     loopinc = 1
     loopLast = nct
  endif
#endif
  !
   IF(QBPC) THEN
        QBPC = .FALSE. 
        CALL APPEND_BMAP_TO_CMAP( )
        do I1=1,MAXCTP-1
           if(ICPT(I1).ne.0 .and. ICPT(I1+1).eq.0)then
              do J1=1, NBTP
                 ICPT(NCT+J1)=NCTP+J1
              end do
              EXIT
           end if
        end do
   END IF

  loop10: do icta=loopFirst,loopLast + NBTP,loopinc
  
#if KEY_DOMDEC==1
     if (q_domdec) then
        ict = cmaptbl(icta)
     else
#endif 
        ict = icta
#if KEY_DOMDEC==1
     endif  
#endif
     !
     I1=I1P(ICT)

     IF(NOCONS) THEN
        IF(ISKP(ICT).NE.0) cycle loop10
     ENDIF

     IC=ICPT(ICT)
     IF(IC.EQ.0) cycle loop10

     J1=J1P(ICT)
     K1=K1P(ICT)
     L1=L1P(ICT)

     I2=I2P(ICT)
     J2=J2P(ICT)
     K2=K2P(ICT)
     L2=L2P(ICT)
     !
#if KEY_GRAPE==1
     if(lfmm) then
        if((fmmcpu(i1)/=1).and.(fmmcpu(j1)/=1).and.(fmmcpu(k1)/=1) &
             .and.(fmmcpu(l1)/=1).and.(fmmcpu(i2)/=1).and.(fmmcpu(j2)/=1) &
             .and.(fmmcpu(k2)/=1).and.(fmmcpu(l2)/=1)) cycle loop10
     endif
#endif

     ! H Kamberaj (Torsion scaling) November 2007
#if KEY_TSALLIS==1
     if (qttsall .or. qtalpha) then
        FI1 = IASCALE(I1)
        FJ1 = IASCALE(J1)
        FK1 = IASCALE(K1)
        FL1 = IASCALE(L1)

        FI2 = IASCALE(I2)
        FJ2 = IASCALE(J2)
        FK2 = IASCALE(K2)
        FL2 = IASCALE(L2)

        sum1 = fi1+fj1+fk1+fl1
        sum2 = fi2+fj2+fk2+fl2
        tsallismd = (sum1 == 4 .or. sum2 == 4)
!        TSALLISMD =  ( &
!             ( (FI1 .EQ. 1) .AND. (FJ1 .EQ. 1) .AND. &
!             (FK1 .EQ. 1) .AND. (FL1 .EQ. 1) ) .OR. &
!             ( (FI2 .EQ. 1) .AND. (FJ2 .EQ. 1) .AND. &
!             (FK2 .EQ. 1) .AND. (FL2 .EQ. 1) ) &
!             )
        IF (QTTSALL.AND.TSALLISMD) TSNDF=TSNDF+1
        IF (QTALPHA) THEN
           IF (TSALLISMD) IC=IC*TSALPHA
        ENDIF
     endif
#endif 
     !
     ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
     FX1=X(I1)-X(J1)
     FY1=Y(I1)-Y(J1)
     FZ1=Z(I1)-Z(J1)
     GX1=X(J1)-X(K1)
     GY1=Y(J1)-Y(K1)
     GZ1=Z(J1)-Z(K1)
     HX1=X(L1)-X(K1)
     HY1=Y(L1)-Y(K1)
     HZ1=Z(L1)-Z(K1)
     ! A=F^G, B=H^G
     AX1=FY1*GZ1-FZ1*GY1
     AY1=FZ1*GX1-FX1*GZ1
     AZ1=FX1*GY1-FY1*GX1
     BX1=HY1*GZ1-HZ1*GY1
     BY1=HZ1*GX1-HX1*GZ1
     BZ1=HX1*GY1-HY1*GX1
     ! RG=|G|, RGR=1/|G|
     RA21=AX1*AX1+AY1*AY1+AZ1*AZ1
     RB21=BX1*BX1+BY1*BY1+BZ1*BZ1
     RG21=GX1*GX1+GY1*GY1+GZ1*GZ1
     RG1=SQRT(RG21)
     ! Warnings have been simplified.
     IF((RA21.LE.RXMIN2).OR.(RB21.LE.RXMIN2) &
          .OR.(RG1.LE.RXMIN)) THEN
        NWARN=NWARN+1
        IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
           WRITE(OUTU,20) ICT,I1,J1,K1,L1
20         FORMAT(' ECMAP: WARNING.  dihedral',I5,' is almost linear.'/ &
                ' derivatives may be affected for atoms:',4I5)
        ENDIF
        cycle loop10
     ENDIF
     !
     RGR1=ONE/RG1
     RA2R1=ONE/RA21
     RB2R1=ONE/RB21
     RABR1=SQRT(RA2R1*RB2R1)
     ! CP=cos(phi)
     CP1=(AX1*BX1+AY1*BY1+AZ1*BZ1)*RABR1
     ! SP=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
     ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
     SP1=RG1*RABR1*(AX1*HX1+AY1*HY1+AZ1*HZ1)

     IF (CP1.LT.-HALF .OR. CP1.GT.HALF) THEN
        PHI1=ASIN(SP1)*RADDEG

        IF(CP1.LT.ZERO) THEN
           IF(PHI1.GT.ZERO) THEN
              PHI1=ONE8TY-PHI1
           ELSE
              PHI1=-ONE8TY-PHI1
           ENDIF
        ENDIF
     ELSE
        PHI1=ACOS(CP1)*RADDEG
        IF (SP1.LT.ZERO) THEN
           PHI1=-PHI1
        ENDIF
     ENDIF

     ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
     FX2=X(I2)-X(J2)
     FY2=Y(I2)-Y(J2)
     FZ2=Z(I2)-Z(J2)
     GX2=X(J2)-X(K2)
     GY2=Y(J2)-Y(K2)
     GZ2=Z(J2)-Z(K2)
     HX2=X(L2)-X(K2)
     HY2=Y(L2)-Y(K2)
     HZ2=Z(L2)-Z(K2)
     ! A=F^G, B=H^G
     AX2=FY2*GZ2-FZ2*GY2
     AY2=FZ2*GX2-FX2*GZ2
     AZ2=FX2*GY2-FY2*GX2
     BX2=HY2*GZ2-HZ2*GY2
     BY2=HZ2*GX2-HX2*GZ2
     BZ2=HX2*GY2-HY2*GX2
     ! RG=|G|, RGR=1/|G|
     RA22=AX2*AX2+AY2*AY2+AZ2*AZ2
     RB22=BX2*BX2+BY2*BY2+BZ2*BZ2
     RG22=GX2*GX2+GY2*GY2+GZ2*GZ2
     RG2=SQRT(RG22)
     ! Warnings have been simplified.
     IF((RA22.LE.RXMIN2).OR.(RB22.LE.RXMIN2) &
          .OR.(RG2.LE.RXMIN)) THEN
        NWARN=NWARN+1
        IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
           WRITE(OUTU,20) ICT,I2,J2,K2,L2
        ENDIF
        cycle loop10
     ENDIF
     !
     RGR2=ONE/RG2
     RA2R2=ONE/RA22
     RB2R2=ONE/RB22
     RABR2=SQRT(RA2R2*RB2R2)
     ! CP=cos(phi)
     CP2=(AX2*BX2+AY2*BY2+AZ2*BZ2)*RABR2
     ! SP=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
     ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
     SP2=RG2*RABR2*(AX2*HX2+AY2*HY2+AZ2*HZ2)

     IF (CP2.LT.-HALF .OR. CP2.GT.HALF) THEN
        PHI2=ASIN(SP2)*RADDEG

        IF(CP2.LT.ZERO) THEN
           IF(PHI2.GT.ZERO) THEN
              PHI2=ONE8TY-PHI2
           ELSE
              PHI2=-ONE8TY-PHI2
           ENDIF
        ENDIF
     ELSE
        PHI2=ACOS(CP2)*RADDEG
        IF (SP2.LT.ZERO) THEN
           PHI2=-PHI2
        ENDIF
     ENDIF

     XPHI1=PHI1+ONE8TY
     XPHI2=PHI2+ONE8TY

     IF (XPHI1.LT.ZERO) THEN
        XPHI1=XPHI1+THR6TY
     ELSE IF (XPHI1.GE.THR6TY) THEN
        XPHI1=XPHI1-THR6TY
     ENDIF

     IF (XPHI2.LT.ZERO) THEN
        XPHI2=XPHI2+THR6TY
     ELSE IF (XPHI2.GE.THR6TY) THEN
        XPHI2=XPHI2-THR6TY
     ENDIF

     MRES=THR6TY/DBLE(GSCTP(IC))

     IPHI1=INT(XPHI1/MRES)+1
     IPHI2=INT(XPHI2/MRES)+1

     CALL GCSTUP2(GSCTP(IC),MCTP(IC), &
          TY,TY1,TY2,TY12,IPHI1,IPHI2)

     MRES1=MRES                           ! to fix mf_080724
     CALL GCSCF(TY,TY1,TY2,TY12,MRES1,MRES,TC)

     TT=(XPHI1-DBLE(IPHI1-1)*MRES)/MRES
     TU=(XPHI2-DBLE(IPHI2-1)*MRES)/MRES

     E=ZERO
     DF1=ZERO
     DF2=ZERO
     DDF1=ZERO
     DDF2=ZERO
     DDF12=ZERO

     DO II=4,1,-1
        E=TT*E+((TC(II,4)*TU+TC(II,3))*TU+ &
             TC(II,2))*TU+TC(II,1)
        DF1=TU*DF1+(THREE*TC(4,II)*TT+TWO*TC(3,II))*TT+ &
             TC(2,II)
        DF2=TT*DF2+(THREE*TC(II,4)*TU+TWO*TC(II,3))*TU+ &
             TC(II,2)
        DDF1=TU*DDF1+TWO*THREE*TC(4,II)*TT+TWO*TC(3,II)
        DDF2=TT*DDF2+TWO*THREE*TC(II,4)*TU+TWO*TC(II,3)
     ENDDO


     DDF12=TC(2,2)+TWO*TC(3,2)*TT+THREE*TC(4,2)*TT*TT+ &
          TWO*TU*(TC(2,3)+TWO*TC(3,3)*TT+THREE*TC(4,3)*TT*TT)+ &
          THREE*TU*TU*(TC(2,4)+TWO*TC(3,4)*TT+THREE*TC(4,4)*TT*TT)

     FAC=RADDEG/MRES

     DF1=DF1*FAC
     DF2=DF2*FAC

     DDF1=DDF1*FAC*FAC
     DDF2=DDF2*FAC*FAC
     DDF12=DDF12*FAC*FAC

     !

#if KEY_BLOCK==1 /*big_block*/
     IF (QBLOCK) THEN
        IBL1 = IBLCKP(I1)
        JBL1 = IBLCKP(J1)
        KKK1 = IBLCKP(K1)
        LLL1 = IBLCKP(L1)

        IBL2 = IBLCKP(I2)
        JBL2 = IBLCKP(J2)
        KKK2 = IBLCKP(K2)
        LLL2 = IBLCKP(L2)

        ! mf, still todo
#if KEY_DOCK==1
        IF(QDOCK)CALL WRNDIE(-5,'<ECMAP>', &
             'DOCK and BLOCK doesnt work yet.')
        !         three pairs (i,j), (k,j) and (k,l)
        DOCFI1 = 1.0
        DOCFJ1 = 1.0
        DOCFK1 = 1.0
        DOCFJ11 = 1.0
        DOCFL1 = 1.0
        DOCFK11 = 1.0
        IF(QDOCK) THEN
           KDOC  = (IBL1 - 1)*NBLOCK + JBL1
           DOCFI1 = BLDOCP(KDOC)
           KDOC  = (JBL1 - 1)*NBLOCK + IBL1
           DOCFJ1 = BLDOCP(KDOC)
           KDOC  = (KKK1 - 1)*NBLOCK + JBL1
           DOCFK1 = BLDOCP(KDOC)
           KDOC  = (JBL1 - 1)*NBLOCK + KKK1
           DOCFJ1 = BLDOCP(KDOC)
           KDOC  = (KKK1 - 1)*NBLOCK + LLL1
           DOCFK1 = BLDOCP(KDOC)
           KDOC  = (LLL1 - 1)*NBLOCK + KKK1
           DOCFL1 = BLDOCP(KDOC)

           KDOC  = (IBL2 - 1)*NBLOCK + JBL2
           DOCFI2 = BLDOCP(KDOC)
           KDOC  = (JBL2 - 1)*NBLOCK + IBL2
           DOCFJ2 = BLDOCP(KDOC)
           KDOC  = (KKK2 - 1)*NBLOCK + JBL2
           DOCFK2 = BLDOCP(KDOC)
           KDOC  = (JBL2 - 1)*NBLOCK + KKK2
           DOCFJ2 = BLDOCP(KDOC)
           KDOC  = (KKK2 - 1)*NBLOCK + LLL2
           DOCFK2 = BLDOCP(KDOC)
           KDOC  = (LLL2 - 1)*NBLOCK + KKK2
           DOCFL2 = BLDOCP(KDOC)
        ENDIF
#endif /*  DOCK*/
        !     In case there are more than 2 blocks
        !     it would be better if we take the middle ones
        !     but not just yet...
        IF (IBL1 .EQ. JBL1) JBL1=KKK1
        IF (IBL1 .EQ. JBL1) JBL1=LLL1
        IF (IBL1 .EQ. JBL1) JBL1=IBL2
        IF (IBL1 .EQ. JBL1) JBL1=JBL2
        IF (IBL1 .EQ. JBL1) JBL1=KKK2
        IF (IBL1 .EQ. JBL1) JBL1=LLL2
        IF (JBL1 .LT. IBL1) THEN
           KKK1=JBL1
           JBL1=IBL1
           IBL1=KKK1
        ENDIF
        !          KKK1=IBL1+JBL1*(JBL1-1)/2
        !          COEF1 = BLCOEC(KKK1)
        !     COEF2 makes no sense!
        ! +++++++++++++++++++++++++++++++++++++++++++++++++++
        ! MH-2010: I am going to uncomment above lines
        ! the code below (from H.K.)is for coef not COEF1.
        ! Anyway this needs to be checked further:
        ! see the new testcase: c36test/cmapblock.inp
        ! +++++++++++++++++++++++++++++++++++++++++++++++++++
        KKK1=IBL1+JBL1*(JBL1-1)/2
        COEF1 = BLCOEC(KKK1)
        !
        ! Hiqmet Kamberaj (Block Code), November 2007
        if ( (ibl1 .eq. jbl1) .and. &
             (ibl2 .eq. jbl2) ) then
           if (ibl2 .lt. ibl1) then
              kkk1 = ibl2
              ibl2 = ibl1
              ibl1 = ibl2
           endif
           kkk1 = ibl1 + ibl2*(ibl2-1)/2
           coef = BLCOEC(kkk1)
        else
           coef = one
        endif

#if KEY_NOTDEF==1

        ! mf?
        IF (QPRNTV) THEN
           IF (IBL1 == 1 .OR. JBL1 == 1 .OR. IBL1 == JBL1 .OR. &
                IBL2 == 1 .OR. JBL2 == 1 .OR. IBL2 == JBL2) THEN
              IF(QNOPH) THEN
                 VBCMAP(JBL1) = VBCMAP(JBL1) + E
              ENDIF
           ENDIF
        ENDIF
#if KEY_BLOCK==1 /*ldm*/
        IF (QLDM .or. QLMC) THEN
           !           first row or diagonal elements exclude (1,1).
           UNSCALE = 0.0
           IF ((IBL1 == 1 .AND. JBL1 >= LSTRT) .or. &
                (IBL1 >= LSTRT .AND. IBL1 == JBL1)) UNSCALE = E
        ENDIF
#endif /*     LDM*/
        !?          IF ( QNOIM .AND. (CPD(IC).EQ.0) ) COEF=1.0
        !?          IF ( QNOPH .AND. (CPD(IC).NE.0) ) COEF=1.0
#if KEY_DOCK==1
        ! H Kamberaj (Mar 2007)
        IF(QDOCK) THEN
           E=E*COEF*(DOCFI1+DOCFJ1+DOCFK1+DOCFJ11+ &
                DOCFK11+DOCFL1)/6.0

           !?            E=E*COEF1*(DOCFI1+DOCFJ1+DOCFK1+DOCFJ11+
           !?     &                 DOCFK11+DOCFL1)/6.0
        ELSE
#endif 
           !?            E=E*COEF
#if KEY_DOCK==1
        ENDIF
#endif 

#endif /*   NOTDEF*/
        ! RLH 2018-07-24
        if (QMLD) then
           ! Use MSLD scaling
           if (QNOCT) then
              COEF1=1
           else
              ! ! Two-block lambda forces
              ! COEF1=BLCOEP(MLDCMAPC(ICT))
              ! CALL MSLD_LAMBDAFORCE(MLDCMAPI(ICT),MLDCMAPJ(ICT),E)
              
              ! Three-block lambda forces
              CALL MSLD_LAMBDAFORCE3(MLDCMAPI(ICT),MLDCMAPJ(ICT),MLDCMAPK(ICT),E,COEF1)
           endif
        endif
        !
        !     Scale them now:
        !
        E=E*COEF1
        DF1=DF1*COEF1
        DF2=DF2*COEF1
        DDF1=DDF1*COEF1
        DDF2=DDF2*COEF1
        DDF12=DDF12*COEF1
        !
     ENDIF ! QBLOCK
     !
#if KEY_NOTDEF==1
     ! Set the energy.
#endif /* notdef*/

#endif /* (big_block)*/

     !
     ! H KAMBERAJ NOVEMBER 2007 (SCALING TORSIONS, TSALLIS MD)
#if KEY_TSALLIS==1
     IF (QTTSALL .AND. TSALLISMD) THEN
        EBTORS(2)=EBTORS(2) + E
     ELSE
        EC=EC+E
     ENDIF
#else /**/
     EC=EC+E
#endif 
     !
     !brb...19-Jul-94 New ANAL terms
     IF(QATERM) THEN
        KK1=ANSLCT(I1)+ANSLCT(J1)+ANSLCT(K1)+ANSLCT(L1)
        KK2=ANSLCT(I2)+ANSLCT(J2)+ANSLCT(K2)+ANSLCT(L2)
        IF(KK1+KK2.EQ.8 .OR. &
             (KK1+KK2.GE.1 .AND. .NOT.QAONLY)) THEN
           IF(QAUNIT.LT.0) THEN
              II=OUTU
           ELSE
              II=QAUNIT
           ENDIF
           !
           IF(PRNLEV.GE.5) THEN
              IF(QAFIRST) THEN
                 IF(QLONGL) THEN
                    WRITE(II,253)
                 ELSE
                    WRITE(II,254)
                 ENDIF
253              FORMAT('ANAL: CMAP: Index        Atom-I              ', &
                      '     Atom-J                   Atom-K         ', &
                      '          Atom-L          ', &
                      '        Dihedral       Energy   ', &
                      '      Force           Parameters')
254              FORMAT('ANAL: CMAP: Index        Atom-I              ', &
                      '     Atom-J',/ &
                      '                         Atom-K         ', &
                      '          Atom-L          ',/ &
                      '        Dihedral       Energy   ', &
                      '      Force           Parameters')
                 QAFIRST=.FALSE.
              ENDIF
              CALL ATOMID(I1,SIDDNI1,RIDDNI1,RESDNI1,ACDNI1)
              CALL ATOMID(J1,SIDDNJ1,RIDDNJ1,RESDNJ1,ACDNJ1)
              CALL ATOMID(K1,SIDDNK1,RIDDNK1,RESDNK1,ACDNK1)
              CALL ATOMID(L1,SIDDNL1,RIDDNL1,RESDNL1,ACDNL1)
              CALL ATOMID(I2,SIDDNI2,RIDDNI2,RESDNI2,ACDNI2)
              CALL ATOMID(J2,SIDDNJ2,RIDDNJ2,RESDNJ2,ACDNJ2)
              CALL ATOMID(K2,SIDDNK2,RIDDNK2,RESDNK2,ACDNK2)
              CALL ATOMID(L2,SIDDNL2,RIDDNL2,RESDNL2,ACDNL2)
              IF(QLONGL) THEN
                 WRITE(II,255) ICT, &
                      I1,SIDDNI1(1:idleng),RIDDNI1(1:idleng), &
                      RESDNI1(1:idleng),ACDNI1(1:idleng), &
                      J1,SIDDNJ1(1:idleng),RIDDNJ1(1:idleng), &
                      RESDNJ1(1:idleng),ACDNJ1(1:idleng), &
                      K1,SIDDNK1(1:idleng),RIDDNK1(1:idleng), &
                      RESDNK1(1:idleng),ACDNK1(1:idleng), &
                      L1,SIDDNL1(1:idleng),RIDDNL1(1:idleng), &
                      RESDNL1(1:idleng),ACDNL1(1:idleng), &
                      I2,SIDDNI2(1:idleng),RIDDNI2(1:idleng), &
                      RESDNI2(1:idleng),ACDNI2(1:idleng), &
                      J2,SIDDNJ2(1:idleng),RIDDNJ2(1:idleng), &
                      RESDNJ2(1:idleng),ACDNJ2(1:idleng), &
                      K2,SIDDNK2(1:idleng),RIDDNK2(1:idleng), &
                      RESDNK2(1:idleng),ACDNK2(1:idleng), &
                      L2,SIDDNL2(1:idleng),RIDDNL2(1:idleng), &
                      RESDNL2(1:idleng),ACDNL2(1:idleng), &
                      PHI1,PHI2,E,DF1,DF2,IC
              ELSE
                 WRITE(II,256) ICT, &
                      I1,SIDDNI1(1:idleng),RIDDNI1(1:idleng), &
                      RESDNI1(1:idleng),ACDNI1(1:idleng), &
                      J1,SIDDNJ1(1:idleng),RIDDNJ1(1:idleng), &
                      RESDNJ1(1:idleng),ACDNJ1(1:idleng), &
                      K1,SIDDNK1(1:idleng),RIDDNK1(1:idleng), &
                      RESDNK1(1:idleng),ACDNK1(1:idleng), &
                      L1,SIDDNL1(1:idleng),RIDDNL1(1:idleng), &
                      RESDNL1(1:idleng),ACDNL1(1:idleng), &
                      I2,SIDDNI2(1:idleng),RIDDNI2(1:idleng), &
                      RESDNI2(1:idleng),ACDNI2(1:idleng), &
                      J2,SIDDNJ2(1:idleng),RIDDNJ2(1:idleng), &
                      RESDNJ2(1:idleng),ACDNJ2(1:idleng), &
                      K2,SIDDNK2(1:idleng),RIDDNK2(1:idleng), &
                      RESDNK2(1:idleng),ACDNK2(1:idleng), &
                      L2,SIDDNL2(1:idleng),RIDDNL2(1:idleng), &
                      RESDNL2(1:idleng),ACDNL2(1:idleng), &
                      PHI1,PHI2,E,DF1,DF2,IC
              ENDIF
255           FORMAT('ANAL: CMAP>',2I5,4(1X,A),I5,4(1X,A), &
                   I5,4(1X,A),I5,4(1X,A),/, &
                   I5,4(1X,A),I5,4(1X,A),I5,4(1X,A),I5,4(1X,A), &
                   5F15.6,I5)
256           FORMAT('ANAL: CMAP>',2I5,4(1X,A),I5,4(1X,A),/ &
                   I5,4(1X,A),I5,4(1X,A),/, &
                   I5,4(1X,A),I5,4(1X,A),/, &
                   I5,4(1X,A),I5,4(1X,A),/, &
                   5F15.6,I5)
           ENDIF
        ENDIF
     ENDIF
     !
     ! Contribution on atoms.
     IF(QECONTX) THEN
        E=E*PT25*PT25
        ECONTX(I1)=ECONTX(I1)+E
        ECONTX(J1)=ECONTX(J1)+E
        ECONTX(K1)=ECONTX(K1)+E
        ECONTX(L1)=ECONTX(L1)+E
        ECONTX(I2)=ECONTX(I2)+E
        ECONTX(J2)=ECONTX(J2)+E
        ECONTX(K2)=ECONTX(K2)+E
        ECONTX(L2)=ECONTX(L2)+E
     ENDIF
     !
     ! Compute derivatives wrt catesian coordinates.
     ! this section is for first derivatives only.
     !
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
        ! GAA=-|G|/A^2, GBB=|G|/B^2, FG=F.G, HG=H.G
        !  FGA=F.G/(|G|A^2), HGB=H.G/(|G|B^2)
        FG1=FX1*GX1+FY1*GY1+FZ1*GZ1
        HG1=HX1*GX1+HY1*GY1+HZ1*GZ1
        FGA1=FG1*RA2R1*RGR1
        HGB1=HG1*RB2R1*RGR1
        GAA1=-RA2R1*RG1
        GBB1=RB2R1*RG1
        ! DTFi=d(phi)/dFi, DTGi=d(phi)/dGi, DTHi=d(phi)/dHi. (used in 2nd deriv)
        DTFX1=GAA1*AX1
        DTFY1=GAA1*AY1
        DTFZ1=GAA1*AZ1
        DTGX1=FGA1*AX1-HGB1*BX1
        DTGY1=FGA1*AY1-HGB1*BY1
        DTGZ1=FGA1*AZ1-HGB1*BZ1
        DTHX1=GBB1*BX1
        DTHY1=GBB1*BY1
        DTHZ1=GBB1*BZ1
        ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
        DFX1=DF1*DTFX1
        DFY1=DF1*DTFY1
        DFZ1=DF1*DTFZ1
        DGX1=DF1*DTGX1
        DGY1=DF1*DTGY1
        DGZ1=DF1*DTGZ1
        DHX1=DF1*DTHX1
        DHY1=DF1*DTHY1
        DHZ1=DF1*DTHZ1
        ! Distribute over Ri.

        ! todo
#if KEY_NOTDEF==1
#if KEY_BLOCK==1
#if KEY_DOCK==1
        IF(QDOCK) THEN
           ! H KAMBERAJ (TSALLISMD) november 2007
#if KEY_TSALLIS==1
           IF (QTTSALL .AND. TSALLISMD) THEN
              TDX(I1)=TDX(I1)+DFX1*DOCFI1
              TDY(I1)=TDY(I1)+DFY1*DOCFI1
              TDZ(I1)=TDZ(I1)+DFZ1*DOCFI1
              TDX(J1)=TDX(J1)-DFX1*DOCFJ1+DGX1*DOCFJ11
              TDY(J1)=TDY(J1)-DFY1*DOCFJ1+DGY1*DOCFJ11
              TDZ(J1)=TDZ(J1)-DFZ1*DOCFJ1+DGZ1*DOCFJ11
              TDX(K1)=TDX(K1)-DHX1*DOCFK1-DGX1*DOCFK1
              TDY(K1)=TDY(K1)-DHY1*DOCFK1-DGY1*DOCFK1
              TDZ(K1)=TDZ(K1)-DHZ1*DOCFK1-DGZ1*DOCFK1
              TDX(L1)=TDX(L1)+DHX1*DOCFL1
              TDY(L1)=TDY(L1)+DHY1*DOCFL1
              TDZ(L1)=TDZ(L1)+DHZ1*DOCFL1
           ELSE
              DX(I1)=DX(I1)+DFX1*DOCFI1
              DY(I1)=DY(I1)+DFY1*DOCFI1
              DZ(I1)=DZ(I1)+DFZ1*DOCFI1
              DX(J1)=DX(J1)-DFX1*DOCFJ1+DGX1*DOCFJ11
              DY(J1)=DY(J1)-DFY1*DOCFJ1+DGY1*DOCFJ11
              DZ(J1)=DZ(J1)-DFZ1*DOCFJ1+DGZ1*DOCFJ11
              DX(K1)=DX(K1)-DHX1*DOCFK1-DGX1*DOCFK1
              DY(K1)=DY(K1)-DHY1*DOCFK1-DGY1*DOCFK1
              DZ(K1)=DZ(K1)-DHZ1*DOCFK1-DGZ1*DOCFK1
              DX(L1)=DX(L1)+DHX1*DOCFL1
              DY(L1)=DY(L1)+DHY1*DOCFL1
              DZ(L1)=DZ(L1)+DHZ1*DOCFL1
           ENDIF
#else /**/
           DX(I1)=DX(I1)+DFX1*DOCFI1
           DY(I1)=DY(I1)+DFY1*DOCFI1
           DZ(I1)=DZ(I1)+DFZ1*DOCFI1
           DX(J1)=DX(J1)-DFX1*DOCFJ1+DGX1*DOCFJ11
           DY(J1)=DY(J1)-DFY1*DOCFJ1+DGY1*DOCFJ11
           DZ(J1)=DZ(J1)-DFZ1*DOCFJ1+DGZ1*DOCFJ11
           DX(K1)=DX(K1)-DHX1*DOCFK1-DGX1*DOCFK1
           DY(K1)=DY(K1)-DHY1*DOCFK1-DGY1*DOCFK1
           DZ(K1)=DZ(K1)-DHZ1*DOCFK1-DGZ1*DOCFK1
           DX(L1)=DX(L1)+DHX1*DOCFL1
           DY(L1)=DY(L1)+DHY1*DOCFL1
           DZ(L1)=DZ(L1)+DHZ1*DOCFL1
#endif 
           !
        ELSE
#endif 
#endif 
#endif /* notdef*/
           ! H KAMBERAJ November 2007 (TSALLIS MD)
#if KEY_TSALLIS==1
           IF (QTTSALL .AND. TSALLISMD) THEN
              TDX(I1)=TDX(I1)+DFX1
              TDY(I1)=TDY(I1)+DFY1
              TDZ(I1)=TDZ(I1)+DFZ1
              TDX(J1)=TDX(J1)-DFX1+DGX1
              TDY(J1)=TDY(J1)-DFY1+DGY1
              TDZ(J1)=TDZ(J1)-DFZ1+DGZ1
              TDX(K1)=TDX(K1)-DHX1-DGX1
              TDY(K1)=TDY(K1)-DHY1-DGY1
              TDZ(K1)=TDZ(K1)-DHZ1-DGZ1
              TDX(L1)=TDX(L1)+DHX1
              TDY(L1)=TDY(L1)+DHY1
              TDZ(L1)=TDZ(L1)+DHZ1
           ELSE
              DX(I1)=DX(I1)+DFX1
              DY(I1)=DY(I1)+DFY1
              DZ(I1)=DZ(I1)+DFZ1
              DX(J1)=DX(J1)-DFX1+DGX1
              DY(J1)=DY(J1)-DFY1+DGY1
              DZ(J1)=DZ(J1)-DFZ1+DGZ1
              DX(K1)=DX(K1)-DHX1-DGX1
              DY(K1)=DY(K1)-DHY1-DGY1
              DZ(K1)=DZ(K1)-DHZ1-DGZ1
              DX(L1)=DX(L1)+DHX1
              DY(L1)=DY(L1)+DHY1
              DZ(L1)=DZ(L1)+DHZ1
           ENDIF
#else /**/
           DX(I1)=DX(I1)+DFX1
           DY(I1)=DY(I1)+DFY1
           DZ(I1)=DZ(I1)+DFZ1
           DX(J1)=DX(J1)-DFX1+DGX1
           DY(J1)=DY(J1)-DFY1+DGY1
           DZ(J1)=DZ(J1)-DFZ1+DGZ1
           DX(K1)=DX(K1)-DHX1-DGX1
           DY(K1)=DY(K1)-DHY1-DGY1
           DZ(K1)=DZ(K1)-DHZ1-DGZ1
           DX(L1)=DX(L1)+DHX1
           DY(L1)=DY(L1)+DHY1
           DZ(L1)=DZ(L1)+DHZ1
#endif 
           !
#if KEY_NOTDEF==1
#if KEY_BLOCK==1
#if KEY_DOCK==1
        ENDIF
#endif 
#endif 
#endif /* notdef*/

#if KEY_NOTDEF==1
#if KEY_BLOCK==1 /*ldm*/
        IF (RSTP) THEN
           IF ((.NOT. QNOIM .AND. CPD(IC) == 0) .OR. &
                (.NOT. QNOPH .AND. CPD(IC) /= 0)) THEN
              IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                   (IBL >= LSTRT .AND. IBL == JBL)) THEN
                 DFX=DFORG*DTFX
                 DFY=DFORG*DTFY
                 DFZ=DFORG*DTFZ
                 DGX=DFORG*DTGX
                 DGY=DFORG*DTGY
                 DGZ=DFORG*DTGZ
                 DHX=DFORG*DTHX
                 DHY=DFORG*DTHY
                 DHZ=DFORG*DTHZ
                 ENVDX(I) = ENVDX(I) + DFX
                 ENVDY(I) = ENVDY(I) + DFY
                 ENVDZ(I) = ENVDZ(I) + DFZ
                 ENVDX(J) = ENVDX(J) - DFX + DGX
                 ENVDY(J) = ENVDY(J) - DFY + DGY
                 ENVDZ(J) = ENVDZ(J) - DFZ + DGZ
                 ENVDX(K) = ENVDX(K) - DHX - DGX
                 ENVDY(K) = ENVDY(K) - DHY - DGY
                 ENVDZ(K) = ENVDZ(K) - DHZ - DGZ
                 ENVDX(L) = ENVDX(L) + DHX
                 ENVDY(L) = ENVDY(L) + DHY
                 ENVDZ(L) = ENVDZ(L) + DHZ
              ENDIF
           ENDIF
        ENDIF
#endif /*  LDM*/

#endif /* notdef*/
        !
#if KEY_IPRESS==1
        IF(QIPRSS) THEN
           PVIR(I1)=PVIR(I1)+DFX1*FX1+DFY1*FY1+DFZ1*FZ1
           PVIR(J1)=PVIR(J1)+DFX1*FX1+DFY1*FY1+DFZ1*FZ1+ &
                DGX1*GX1+DGY1*GY1+DGZ1*GZ1
           PVIR(K1)=PVIR(K1)+DGX1*GX1+DGY1*GY1+DGZ1*GZ1+ &
                DHX1*HX1+DHY1*HY1+DHZ1*HZ1
           PVIR(L1)=PVIR(L1)+DHX1*HX1+DHY1*HY1+DHZ1*HZ1
        ENDIF
#endif 


        ! second torsion

        ! GAA=-|G|/A^2, GBB=|G|/B^2, FG=F.G, HG=H.G
        !  FGA=F.G/(|G|A^2), HGB=H.G/(|G|B^2)
        FG2=FX2*GX2+FY2*GY2+FZ2*GZ2
        HG2=HX2*GX2+HY2*GY2+HZ2*GZ2
        FGA2=FG2*RA2R2*RGR2
        HGB2=HG2*RB2R2*RGR2
        GAA2=-RA2R2*RG2
        GBB2=RB2R2*RG2
        ! DTFi=d(phi)/dFi, DTGi=d(phi)/dGi, DTHi=d(phi)/dHi. (used in 2nd deriv)
        DTFX2=GAA2*AX2
        DTFY2=GAA2*AY2
        DTFZ2=GAA2*AZ2
        DTGX2=FGA2*AX2-HGB2*BX2
        DTGY2=FGA2*AY2-HGB2*BY2
        DTGZ2=FGA2*AZ2-HGB2*BZ2
        DTHX2=GBB2*BX2
        DTHY2=GBB2*BY2
        DTHZ2=GBB2*BZ2
        ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
        DFX2=DF2*DTFX2
        DFY2=DF2*DTFY2
        DFZ2=DF2*DTFZ2
        DGX2=DF2*DTGX2
        DGY2=DF2*DTGY2
        DGZ2=DF2*DTGZ2
        DHX2=DF2*DTHX2
        DHY2=DF2*DTHY2
        DHZ2=DF2*DTHZ2
        ! Distribute over Ri.

        ! todo
#if KEY_NOTDEF==1
#if KEY_BLOCK==1
#if KEY_DOCK==1
        IF(QDOCK) THEN
           ! H KAMBERAJ November 2007 (TSALLIS MD)
#if KEY_TSALLIS==1
           IF (QTTSALL .AND. TSALLISMD) THEN
              TDX(I2)=TDX(I2)+DFX2*DOCFI2
              TDY(I2)=TDY(I2)+DFY2*DOCFI2
              TDZ(I2)=TDZ(I2)+DFZ2*DOCFI2
              TDX(J2)=TDX(J2)-DFX2*DOCFJ2+DGX2*DOCFJ12
              TDY(J2)=TDY(J2)-DFY2*DOCFJ2+DGY2*DOCFJ12
              TDZ(J2)=TDZ(J2)-DFZ2*DOCFJ2+DGZ2*DOCFJ12
              TDX(K2)=TDX(K2)-DHX2*DOCFK2-DGX2*DOCFK2
              TDY(K2)=TDY(K2)-DHY2*DOCFK2-DGY2*DOCFK2
              TDZ(K2)=TDZ(K2)-DHZ2*DOCFK2-DGZ2*DOCFK2
              TDX(L2)=TDX(L2)+DHX2*DOCFL2
              TDY(L2)=TDY(L2)+DHY2*DOCFL2
              TDZ(L2)=TDZ(L2)+DHZ2*DOCFL2
           ELSE
              DX(I2)=DX(I2)+DFX2*DOCFI2
              DY(I2)=DY(I2)+DFY2*DOCFI2
              DZ(I2)=DZ(I2)+DFZ2*DOCFI2
              DX(J2)=DX(J2)-DFX2*DOCFJ2+DGX2*DOCFJ12
              DY(J2)=DY(J2)-DFY2*DOCFJ2+DGY2*DOCFJ12
              DZ(J2)=DZ(J2)-DFZ2*DOCFJ2+DGZ2*DOCFJ12
              DX(K2)=DX(K2)-DHX2*DOCFK2-DGX2*DOCFK2
              DY(K2)=DY(K2)-DHY2*DOCFK2-DGY2*DOCFK2
              DZ(K2)=DZ(K2)-DHZ2*DOCFK2-DGZ2*DOCFK2
              DX(L2)=DX(L2)+DHX2*DOCFL2
              DY(L2)=DY(L2)+DHY2*DOCFL2
              DZ(L2)=DZ(L2)+DHZ2*DOCFL2
           ENDIF
#else /**/
           DX(I2)=DX(I2)+DFX2*DOCFI2
           DY(I2)=DY(I2)+DFY2*DOCFI2
           DZ(I2)=DZ(I2)+DFZ2*DOCFI2
           DX(J2)=DX(J2)-DFX2*DOCFJ2+DGX2*DOCFJ12
           DY(J2)=DY(J2)-DFY2*DOCFJ2+DGY2*DOCFJ12
           DZ(J2)=DZ(J2)-DFZ2*DOCFJ2+DGZ2*DOCFJ12
           DX(K2)=DX(K2)-DHX2*DOCFK2-DGX2*DOCFK2
           DY(K2)=DY(K2)-DHY2*DOCFK2-DGY2*DOCFK2
           DZ(K2)=DZ(K2)-DHZ2*DOCFK2-DGZ2*DOCFK2
           DX(L2)=DX(L2)+DHX2*DOCFL2
           DY(L2)=DY(L2)+DHY2*DOCFL2
           DZ(L2)=DZ(L2)+DHZ2*DOCFL2
#endif 
           !
        ELSE
#endif 
#endif 
#endif /* notdef*/
           ! H KAMBERAJ November 2007 (TSALLISMD)
#if KEY_TSALLIS==1
           IF (QTTSALL .AND. TSALLISMD) THEN
              TDX(I2)=TDX(I2)+DFX2
              TDY(I2)=TDY(I2)+DFY2
              TDZ(I2)=TDZ(I2)+DFZ2
              TDX(J2)=TDX(J2)-DFX2+DGX2
              TDY(J2)=TDY(J2)-DFY2+DGY2
              TDZ(J2)=TDZ(J2)-DFZ2+DGZ2
              TDX(K2)=TDX(K2)-DHX2-DGX2
              TDY(K2)=TDY(K2)-DHY2-DGY2
              TDZ(K2)=TDZ(K2)-DHZ2-DGZ2
              TDX(L2)=TDX(L2)+DHX2
              TDY(L2)=TDY(L2)+DHY2
              TDZ(L2)=TDZ(L2)+DHZ2
           ELSE
              DX(I2)=DX(I2)+DFX2
              DY(I2)=DY(I2)+DFY2
              DZ(I2)=DZ(I2)+DFZ2
              DX(J2)=DX(J2)-DFX2+DGX2
              DY(J2)=DY(J2)-DFY2+DGY2
              DZ(J2)=DZ(J2)-DFZ2+DGZ2
              DX(K2)=DX(K2)-DHX2-DGX2
              DY(K2)=DY(K2)-DHY2-DGY2
              DZ(K2)=DZ(K2)-DHZ2-DGZ2
              DX(L2)=DX(L2)+DHX2
              DY(L2)=DY(L2)+DHY2
              DZ(L2)=DZ(L2)+DHZ2
           ENDIF
#else /**/
           DX(I2)=DX(I2)+DFX2
           DY(I2)=DY(I2)+DFY2
           DZ(I2)=DZ(I2)+DFZ2
           DX(J2)=DX(J2)-DFX2+DGX2
           DY(J2)=DY(J2)-DFY2+DGY2
           DZ(J2)=DZ(J2)-DFZ2+DGZ2
           DX(K2)=DX(K2)-DHX2-DGX2
           DY(K2)=DY(K2)-DHY2-DGY2
           DZ(K2)=DZ(K2)-DHZ2-DGZ2
           DX(L2)=DX(L2)+DHX2
           DY(L2)=DY(L2)+DHY2
           DZ(L2)=DZ(L2)+DHZ2
#endif 
           !
#if KEY_NOTDEF==1
#if KEY_BLOCK==1
#if KEY_DOCK==1
        ENDIF
#endif 
#endif 
#endif /* notdef*/

#if KEY_NOTDEF==1
#if KEY_BLOCK==1 /*ldm*/
        IF (RSTP) THEN
           IF ((.NOT. QNOIM .AND. CPD(IC) == 0) .OR. &
                (.NOT. QNOPH .AND. CPD(IC) /= 0)) THEN
              IF ((IBL == 1 .AND. JBL >= LSTRT) .or. &
                   (IBL >= LSTRT .AND. IBL == JBL)) THEN
                 DFX=DFORG*DTFX
                 DFY=DFORG*DTFY
                 DFZ=DFORG*DTFZ
                 DGX=DFORG*DTGX
                 DGY=DFORG*DTGY
                 DGZ=DFORG*DTGZ
                 DHX=DFORG*DTHX
                 DHY=DFORG*DTHY
                 DHZ=DFORG*DTHZ
                 ENVDX(I) = ENVDX(I) + DFX
                 ENVDY(I) = ENVDY(I) + DFY
                 ENVDZ(I) = ENVDZ(I) + DFZ
                 ENVDX(J) = ENVDX(J) - DFX + DGX
                 ENVDY(J) = ENVDY(J) - DFY + DGY
                 ENVDZ(J) = ENVDZ(J) - DFZ + DGZ
                 ENVDX(K) = ENVDX(K) - DHX - DGX
                 ENVDY(K) = ENVDY(K) - DHY - DGY
                 ENVDZ(K) = ENVDZ(K) - DHZ - DGZ
                 ENVDX(L) = ENVDX(L) + DHX
                 ENVDY(L) = ENVDY(L) + DHY
                 ENVDZ(L) = ENVDZ(L) + DHZ
              ENDIF
           ENDIF
        ENDIF
#endif /*  LDM*/

#endif /* notdef*/
        !
#if KEY_IPRESS==1
        IF(QIPRSS) THEN
           PVIR(I2)=PVIR(I2)+DFX2*FX2+DFY2*FY2+DFZ2*FZ2
           PVIR(J2)=PVIR(J2)+DFX2*FX2+DFY2*FY2+DFZ2*FZ2+ &
                DGX2*GX2+DGY2*GY2+DGZ2*GZ2
           PVIR(K2)=PVIR(K2)+DGX2*GX2+DGY2*GY2+DGZ2*GZ2+ &
                DHX2*HX2+DHY2*HY2+DHZ2*HZ2
           PVIR(L2)=PVIR(L2)+DHX2*HX2+DHY2*HY2+DHZ2*HZ2
        ENDIF
#endif 

        !
        ! Second derivative part, first torsion
        !
        IF (QSECD) THEN
           !
           ! RGR2=1/G.G,FGRG2=(F.G)/(G.G),HGRG2=(H.G)/(G.G),DFRG3=(dE/dPhi)/|G|^3
           RGR21=RGR1*RGR1
           FGRG21=FG1*RGR21
           HGRG21=HG1*RGR21
           DFRG31=DF1*RGR21*RGR1
           ! GAF=-G^A/A.A, GBH=-G^B/B.B, FAG=F^A/A.A, HBG=-H^B/B.B
           GAFX1=RA2R1*(AY1*GZ1-AZ1*GY1)
           GAFY1=RA2R1*(AZ1*GX1-AX1*GZ1)
           GAFZ1=RA2R1*(AX1*GY1-AY1*GX1)
           GBHX1=RB2R1*(BY1*GZ1-BZ1*GY1)
           GBHY1=RB2R1*(BZ1*GX1-BX1*GZ1)
           GBHZ1=RB2R1*(BX1*GY1-BY1*GX1)
           FAGX1=RA2R1*(FY1*AZ1-FZ1*AY1)
           FAGY1=RA2R1*(FZ1*AX1-FX1*AZ1)
           FAGZ1=RA2R1*(FX1*AY1-FY1*AX1)
           HBGX1=RB2R1*(BY1*HZ1-BZ1*HY1)
           HBGY1=RB2R1*(BZ1*HX1-BX1*HZ1)
           HBGZ1=RB2R1*(BX1*HY1-BY1*HX1)
           ! What are the indexes ?
           ! ddE/dX.dY= DDFGH(n)      Fx, Fy, Fz,|Gx, Gy, Gz,|Hx, Hy, Hz. X/
           !                          ----------------------------------- / Y
           !               n=          1   2   4 | 7  11  16 |22  29  37   Fx
           !                               3   5 | 8  12  17 |23  30  38   Fy
           !                                   6 | 9  13  18 |24  31  39   Fz
           !                                     ------------------------
           !                                      10  14  19 |25  32  40   Gx
           !                                          15  20 |26  33  41   Gy
           !                                              21 |27  34  42   Gz
           !                                                 ------------
           !                                                  28  35  43   Hx
           !                                                      36  44   Hy
           !                                                          45   Hz
           !
           ! ddE/dF.dF
           DDFGH(1) =DDF1*DTFX1*DTFX1+TWO*DFX1*GAFX1
           DDFGH(2) =DDF1*DTFX1*DTFY1+DFX1*GAFY1+DFY1*GAFX1
           DDFGH(3) =DDF1*DTFY1*DTFY1+TWO*DFY1*GAFY1
           DDFGH(4) =DDF1*DTFX1*DTFZ1+DFX1*GAFZ1+DFZ1*GAFX1
           DDFGH(5) =DDF1*DTFY1*DTFZ1+DFY1*GAFZ1+DFZ1*GAFY1
           DDFGH(6) =DDF1*DTFZ1*DTFZ1+TWO*DFZ1*GAFZ1
           ! ddE/dF.dG
           DDFGH(7) =DDF1*DTFX1*DTGX1+FAGX1*DFX1-FGRG21*DFX1*GAFX1
           DDFGH(8) =DDF1*DTFY1*DTGX1+FAGY1*DFX1-FGRG21*DFY1*GAFX1
           DDFGH(9) =DDF1*DTFZ1*DTGX1+FAGZ1*DFX1-FGRG21*DFZ1*GAFX1
           DDFGH(11)=DDF1*DTFX1*DTGY1+FAGX1*DFY1-FGRG21*DFX1*GAFY1
           DDFGH(12)=DDF1*DTFY1*DTGY1+FAGY1*DFY1-FGRG21*DFY1*GAFY1
           DDFGH(13)=DDF1*DTFZ1*DTGY1+FAGZ1*DFY1-FGRG21*DFZ1*GAFY1
           DDFGH(16)=DDF1*DTFX1*DTGZ1+FAGX1*DFZ1-FGRG21*DFX1*GAFZ1
           DDFGH(17)=DDF1*DTFY1*DTGZ1+FAGY1*DFZ1-FGRG21*DFY1*GAFZ1
           DDFGH(18)=DDF1*DTFZ1*DTGZ1+FAGZ1*DFZ1-FGRG21*DFZ1*GAFZ1
           ! ddE/dF.dH
           DDFGH(22)=DDF1*DTFX1*DTHX1
           DDFGH(23)=DDF1*DTFY1*DTHX1
           DDFGH(24)=DDF1*DTFZ1*DTHX1
           DDFGH(29)=DDF1*DTFX1*DTHY1
           DDFGH(30)=DDF1*DTFY1*DTHY1
           DDFGH(31)=DDF1*DTFZ1*DTHY1
           DDFGH(37)=DDF1*DTFX1*DTHZ1
           DDFGH(38)=DDF1*DTFY1*DTHZ1
           DDFGH(39)=DDF1*DTFZ1*DTHZ1
           ! ddE/dG.dG
           DDFGH(10)=DDF1*DTGX1*DTGX1-DFRG31*(GAFX1*AX1-GBHX1*BX1) &
                -TWO*FGRG21*DFX1*FAGX1+TWO*HGRG21*DHX1*HBGX1
           DDFGH(14)=DDF1*DTGX1*DTGY1 &
                -HALF*DFRG31*(GAFX1*AY1+GAFY1*AX1-GBHX1*BY1-GBHY1*BX1) &
                -FGRG21*(DFX1*FAGY1+DFY1*FAGX1)+HGRG21*(DHX1*HBGY1+DHY1*HBGX1)
           DDFGH(15)=DDF1*DTGY1*DTGY1-DFRG31*(GAFY1*AY1-GBHY1*BY1) &
                -TWO*FGRG21*DFY1*FAGY1+TWO*HGRG21*DHY1*HBGY1
           DDFGH(19)=DDF1*DTGX1*DTGZ1 &
                -HALF*DFRG31*(GAFX1*AZ1+GAFZ1*AX1-GBHX1*BZ1-GBHZ1*BX1) &
                -FGRG21*(DFX1*FAGZ1+DFZ1*FAGX1)+HGRG21*(DHX1*HBGZ1+DHZ1*HBGX1)
           DDFGH(20)=DDF1*DTGY1*DTGZ1 &
                -HALF*DFRG31*(GAFY1*AZ1+GAFZ1*AY1-GBHY1*BZ1-GBHZ1*BY1) &
                -FGRG21*(DFY1*FAGZ1+DFZ1*FAGY1)+HGRG21*(DHY1*HBGZ1+DHZ1*HBGY1)
           DDFGH(21)=DDF1*DTGZ1*DTGZ1-DFRG31*(GAFZ1*AZ1-GBHZ1*BZ1) &
                -TWO*FGRG21*DFZ1*FAGZ1+TWO*HGRG21*DHZ1*HBGZ1
           ! ddE/dG.dH
           DDFGH(25)=DDF1*DTGX1*DTHX1-DHX1*HBGX1-HGRG21*GBHX1*DHX1
           DDFGH(26)=DDF1*DTGY1*DTHX1-DHY1*HBGX1-HGRG21*GBHY1*DHX1
           DDFGH(27)=DDF1*DTGZ1*DTHX1-DHZ1*HBGX1-HGRG21*GBHZ1*DHX1
           DDFGH(32)=DDF1*DTGX1*DTHY1-DHX1*HBGY1-HGRG21*GBHX1*DHY1
           DDFGH(33)=DDF1*DTGY1*DTHY1-DHY1*HBGY1-HGRG21*GBHY1*DHY1
           DDFGH(34)=DDF1*DTGZ1*DTHY1-DHZ1*HBGY1-HGRG21*GBHZ1*DHY1
           DDFGH(40)=DDF1*DTGX1*DTHZ1-DHX1*HBGZ1-HGRG21*GBHX1*DHZ1
           DDFGH(41)=DDF1*DTGY1*DTHZ1-DHY1*HBGZ1-HGRG21*GBHY1*DHZ1
           DDFGH(42)=DDF1*DTGZ1*DTHZ1-DHZ1*HBGZ1-HGRG21*GBHZ1*DHZ1
           ! ddE/dH.dH
           DDFGH(28)=DDF1*DTHX1*DTHX1+TWO*DHX1*GBHX1
           DDFGH(35)=DDF1*DTHX1*DTHY1+DHX1*GBHY1+DHY1*GBHX1
           DDFGH(36)=DDF1*DTHY1*DTHY1+TWO*DHY1*GBHY1
           DDFGH(43)=DDF1*DTHX1*DTHZ1+DHX1*GBHZ1+DHZ1*GBHX1
           DDFGH(44)=DDF1*DTHY1*DTHZ1+DHY1*GBHZ1+DHZ1*GBHY1
           DDFGH(45)=DDF1*DTHZ1*DTHZ1+TWO*DHZ1*GBHZ1

           II=3*I1-2
           JJ=3*J1-2
           KK=3*K1-2
           LL=3*L1-2
           IJTEST=(J1.LT.I1)
           IKTEST=(K1.LT.I1)
           JKTEST=(K1.LT.J1)
           ILTEST=(L1.LT.I1)
           JLTEST=(L1.LT.J1)
           KLTEST=(L1.LT.K1)
           !
           IADD=IUPT(II)+II
           DD1(IADD)=DD1(IADD)+DDFGH(1)
           IADD=IUPT(II+1)+II+1
           DD1(IADD)=DD1(IADD)+DDFGH(3)
           IADD=IUPT(II+2)+II+2
           DD1(IADD)=DD1(IADD)+DDFGH(6)
           IADD=IUPT(II)+II+1
           DD1(IADD)=DD1(IADD)+DDFGH(2)
           IADD=IUPT(II)+II+2
           DD1(IADD)=DD1(IADD)+DDFGH(4)
           IADD=IUPT(II+1)+II+2
           DD1(IADD)=DD1(IADD)+DDFGH(5)
           !
           IADD=IUPT(LL)+LL
           DD1(IADD)=DD1(IADD)+DDFGH(28)
           IADD=IUPT(LL+1)+LL+1
           DD1(IADD)=DD1(IADD)+DDFGH(36)
           IADD=IUPT(LL+2)+LL+2
           DD1(IADD)=DD1(IADD)+DDFGH(45)
           IADD=IUPT(LL)+LL+1
           DD1(IADD)=DD1(IADD)+DDFGH(35)
           IADD=IUPT(LL)+LL+2
           DD1(IADD)=DD1(IADD)+DDFGH(43)
           IADD=IUPT(LL+1)+LL+2
           DD1(IADD)=DD1(IADD)+DDFGH(44)
           !
           IADD=IUPT(JJ)+JJ
           DD1(IADD)=DD1(IADD)+DDFGH(1)+DDFGH(10)-DDFGH(7)-DDFGH(7)
           IADD=IUPT(JJ+1)+JJ+1
           DD1(IADD)=DD1(IADD)+DDFGH(3)+DDFGH(15)-DDFGH(12)-DDFGH(12)
           IADD=IUPT(JJ+2)+JJ+2
           DD1(IADD)=DD1(IADD)+DDFGH(6)+DDFGH(21)-DDFGH(18)-DDFGH(18)
           IADD=IUPT(JJ)+JJ+1
           DD1(IADD)=DD1(IADD)+DDFGH(2)+DDFGH(14)-DDFGH(11)-DDFGH(8)
           IADD=IUPT(JJ)+JJ+2
           DD1(IADD)=DD1(IADD)+DDFGH(4)+DDFGH(19)-DDFGH(16)-DDFGH(9)
           IADD=IUPT(JJ+1)+JJ+2
           DD1(IADD)=DD1(IADD)+DDFGH(5)+DDFGH(20)-DDFGH(17)-DDFGH(13)
           !
           IADD=IUPT(KK)+KK
           DD1(IADD)=DD1(IADD)+DDFGH(28)+DDFGH(10)+DDFGH(25)+DDFGH(25)
           IADD=IUPT(KK+1)+KK+1
           DD1(IADD)=DD1(IADD)+DDFGH(36)+DDFGH(15)+DDFGH(33)+DDFGH(33)
           IADD=IUPT(KK+2)+KK+2
           DD1(IADD)=DD1(IADD)+DDFGH(45)+DDFGH(21)+DDFGH(42)+DDFGH(42)
           IADD=IUPT(KK)+KK+1
           DD1(IADD)=DD1(IADD)+DDFGH(35)+DDFGH(14)+DDFGH(32)+DDFGH(26)
           IADD=IUPT(KK)+KK+2
           DD1(IADD)=DD1(IADD)+DDFGH(43)+DDFGH(19)+DDFGH(40)+DDFGH(27)
           IADD=IUPT(KK+1)+KK+2
           DD1(IADD)=DD1(IADD)+DDFGH(44)+DDFGH(20)+DDFGH(41)+DDFGH(34)
           !
           IF (IJTEST) THEN
              IADD=IUPT(JJ)+II
              DD1(IADD)=DD1(IADD)+DDFGH(7)-DDFGH(1)
              IADD=IUPT(JJ+1)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(12)-DDFGH(3)
              IADD=IUPT(JJ+2)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(18)-DDFGH(6)
              IADD=IUPT(JJ)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(8)-DDFGH(2)
              IADD=IUPT(JJ+1)+II
              DD1(IADD)=DD1(IADD)+DDFGH(11)-DDFGH(2)
              IADD=IUPT(JJ)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(9)-DDFGH(4)
              IADD=IUPT(JJ+2)+II
              DD1(IADD)=DD1(IADD)+DDFGH(16)-DDFGH(4)
              IADD=IUPT(JJ+1)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(13)-DDFGH(5)
              IADD=IUPT(JJ+2)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(17)-DDFGH(5)
           ELSE
              IADD=IUPT(II)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(7)-DDFGH(1)
              IADD=IUPT(II+1)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(12)-DDFGH(3)
              IADD=IUPT(II+2)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(18)-DDFGH(6)
              IADD=IUPT(II+1)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(8)-DDFGH(2)
              IADD=IUPT(II)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(11)-DDFGH(2)
              IADD=IUPT(II+2)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(9)-DDFGH(4)
              IADD=IUPT(II)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(16)-DDFGH(4)
              IADD=IUPT(II+2)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(13)-DDFGH(5)
              IADD=IUPT(II+1)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(17)-DDFGH(5)
           ENDIF
           !
           IF (IKTEST) THEN
              IADD=IUPT(KK)+II
              DD1(IADD)=DD1(IADD)-DDFGH(7)-DDFGH(22)
              IADD=IUPT(KK+1)+II+1
              DD1(IADD)=DD1(IADD)-DDFGH(12)-DDFGH(30)
              IADD=IUPT(KK+2)+II+2
              DD1(IADD)=DD1(IADD)-DDFGH(18)-DDFGH(39)
              IADD=IUPT(KK)+II+1
              DD1(IADD)=DD1(IADD)-DDFGH(8)-DDFGH(23)
              IADD=IUPT(KK+1)+II
              DD1(IADD)=DD1(IADD)-DDFGH(11)-DDFGH(29)
              IADD=IUPT(KK)+II+2
              DD1(IADD)=DD1(IADD)-DDFGH(9)-DDFGH(24)
              IADD=IUPT(KK+2)+II
              DD1(IADD)=DD1(IADD)-DDFGH(16)-DDFGH(37)
              IADD=IUPT(KK+1)+II+2
              DD1(IADD)=DD1(IADD)-DDFGH(13)-DDFGH(31)
              IADD=IUPT(KK+2)+II+1
              DD1(IADD)=DD1(IADD)-DDFGH(17)-DDFGH(38)
           ELSE
              IADD=IUPT(II)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(7)-DDFGH(22)
              IADD=IUPT(II+1)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(12)-DDFGH(30)
              IADD=IUPT(II+2)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(18)-DDFGH(39)
              IADD=IUPT(II+1)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(8)-DDFGH(23)
              IADD=IUPT(II)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(11)-DDFGH(29)
              IADD=IUPT(II+2)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(9)-DDFGH(24)
              IADD=IUPT(II)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(16)-DDFGH(37)
              IADD=IUPT(II+2)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(13)-DDFGH(31)
              IADD=IUPT(II+1)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(17)-DDFGH(38)
           ENDIF
           !
           IF (ILTEST) THEN
              IADD=IUPT(LL)+II
              DD1(IADD)=DD1(IADD)+DDFGH(22)
              IADD=IUPT(LL+1)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(30)
              IADD=IUPT(LL+2)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(39)
              IADD=IUPT(LL)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(23)
              IADD=IUPT(LL+1)+II
              DD1(IADD)=DD1(IADD)+DDFGH(29)
              IADD=IUPT(LL)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(24)
              IADD=IUPT(LL+2)+II
              DD1(IADD)=DD1(IADD)+DDFGH(37)
              IADD=IUPT(LL+1)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(31)
              IADD=IUPT(LL+2)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(38)
           ELSE
              IADD=IUPT(II)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(22)
              IADD=IUPT(II+1)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(30)
              IADD=IUPT(II+2)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(39)
              IADD=IUPT(II+1)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(23)
              IADD=IUPT(II)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(29)
              IADD=IUPT(II+2)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(24)
              IADD=IUPT(II)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(37)
              IADD=IUPT(II+2)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(31)
              IADD=IUPT(II+1)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(38)
           ENDIF
           !
           IF (JKTEST) THEN
              IADD=IUPT(KK)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(7)+DDFGH(22)-DDFGH(10)-DDFGH(25)
              IADD=IUPT(KK+1)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(12)+DDFGH(30)-DDFGH(15)-DDFGH(33)
              IADD=IUPT(KK+2)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(18)+DDFGH(39)-DDFGH(21)-DDFGH(42)
              IADD=IUPT(KK)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(8)+DDFGH(23)-DDFGH(14)-DDFGH(26)
              IADD=IUPT(KK+1)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(11)+DDFGH(29)-DDFGH(14)-DDFGH(32)
              IADD=IUPT(KK)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(9)+DDFGH(24)-DDFGH(19)-DDFGH(27)
              IADD=IUPT(KK+2)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(16)+DDFGH(37)-DDFGH(19)-DDFGH(40)
              IADD=IUPT(KK+1)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(13)+DDFGH(31)-DDFGH(20)-DDFGH(34)
              IADD=IUPT(KK+2)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(17)+DDFGH(38)-DDFGH(20)-DDFGH(41)
           ELSE
              IADD=IUPT(JJ)+KK
              DD1(IADD)=DD1(IADD)+DDFGH(7)+DDFGH(22)-DDFGH(10)-DDFGH(25)
              IADD=IUPT(JJ+1)+KK+1
              DD1(IADD)=DD1(IADD)+DDFGH(12)+DDFGH(30)-DDFGH(15)-DDFGH(33)
              IADD=IUPT(JJ+2)+KK+2
              DD1(IADD)=DD1(IADD)+DDFGH(18)+DDFGH(39)-DDFGH(21)-DDFGH(42)
              IADD=IUPT(JJ+1)+KK
              DD1(IADD)=DD1(IADD)+DDFGH(8)+DDFGH(23)-DDFGH(14)-DDFGH(26)
              IADD=IUPT(JJ)+KK+1
              DD1(IADD)=DD1(IADD)+DDFGH(11)+DDFGH(29)-DDFGH(14)-DDFGH(32)
              IADD=IUPT(JJ+2)+KK
              DD1(IADD)=DD1(IADD)+DDFGH(9)+DDFGH(24)-DDFGH(19)-DDFGH(27)
              IADD=IUPT(JJ)+KK+2
              DD1(IADD)=DD1(IADD)+DDFGH(16)+DDFGH(37)-DDFGH(19)-DDFGH(40)
              IADD=IUPT(JJ+2)+KK+1
              DD1(IADD)=DD1(IADD)+DDFGH(13)+DDFGH(31)-DDFGH(20)-DDFGH(34)
              IADD=IUPT(JJ+1)+KK+2
              DD1(IADD)=DD1(IADD)+DDFGH(17)+DDFGH(38)-DDFGH(20)-DDFGH(41)
           ENDIF
           !
           IF (JLTEST) THEN
              IADD=IUPT(LL)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(25)-DDFGH(22)
              IADD=IUPT(LL+1)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(33)-DDFGH(30)
              IADD=IUPT(LL+2)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(42)-DDFGH(39)
              IADD=IUPT(LL)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(26)-DDFGH(23)
              IADD=IUPT(LL+1)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(32)-DDFGH(29)
              IADD=IUPT(LL)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(27)-DDFGH(24)
              IADD=IUPT(LL+2)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(40)-DDFGH(37)
              IADD=IUPT(LL+1)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(34)-DDFGH(31)
              IADD=IUPT(LL+2)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(41)-DDFGH(38)
           ELSE
              IADD=IUPT(JJ)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(25)-DDFGH(22)
              IADD=IUPT(JJ+1)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(33)-DDFGH(30)
              IADD=IUPT(JJ+2)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(42)-DDFGH(39)
              IADD=IUPT(JJ+1)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(26)-DDFGH(23)
              IADD=IUPT(JJ)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(32)-DDFGH(29)
              IADD=IUPT(JJ+2)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(27)-DDFGH(24)
              IADD=IUPT(JJ)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(40)-DDFGH(37)
              IADD=IUPT(JJ+2)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(34)-DDFGH(31)
              IADD=IUPT(JJ+1)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(41)-DDFGH(38)
           ENDIF
           !
           IF (KLTEST) THEN
              IADD=IUPT(LL)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(25)-DDFGH(28)
              IADD=IUPT(LL+1)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(33)-DDFGH(36)
              IADD=IUPT(LL+2)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(42)-DDFGH(45)
              IADD=IUPT(LL)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(26)-DDFGH(35)
              IADD=IUPT(LL+1)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(32)-DDFGH(35)
              IADD=IUPT(LL)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(27)-DDFGH(43)
              IADD=IUPT(LL+2)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(40)-DDFGH(43)
              IADD=IUPT(LL+1)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(34)-DDFGH(44)
              IADD=IUPT(LL+2)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(41)-DDFGH(44)
           ELSE
              IADD=IUPT(KK)+LL
              DD1(IADD)=DD1(IADD)-DDFGH(25)-DDFGH(28)
              IADD=IUPT(KK+1)+LL+1
              DD1(IADD)=DD1(IADD)-DDFGH(33)-DDFGH(36)
              IADD=IUPT(KK+2)+LL+2
              DD1(IADD)=DD1(IADD)-DDFGH(42)-DDFGH(45)
              IADD=IUPT(KK+1)+LL
              DD1(IADD)=DD1(IADD)-DDFGH(26)-DDFGH(35)
              IADD=IUPT(KK)+LL+1
              DD1(IADD)=DD1(IADD)-DDFGH(32)-DDFGH(35)
              IADD=IUPT(KK+2)+LL
              DD1(IADD)=DD1(IADD)-DDFGH(27)-DDFGH(43)
              IADD=IUPT(KK)+LL+2
              DD1(IADD)=DD1(IADD)-DDFGH(40)-DDFGH(43)
              IADD=IUPT(KK+2)+LL+1
              DD1(IADD)=DD1(IADD)-DDFGH(34)-DDFGH(44)
              IADD=IUPT(KK+1)+LL+2
              DD1(IADD)=DD1(IADD)-DDFGH(41)-DDFGH(44)
           ENDIF


           !
           ! Second derivative part, second torsion
           !
           !
           ! RGR2=1/G.G,FGRG2=(F.G)/(G.G),HGRG2=(H.G)/(G.G),DFRG3=(dE/dPhi)/|G|^3
           RGR22=RGR2*RGR2
           FGRG22=FG2*RGR22
           HGRG22=HG2*RGR22
           DFRG32=DF2*RGR22*RGR2
           ! GAF=-G^A/A.A, GBH=-G^B/B.B, FAG=F^A/A.A, HBG=-H^B/B.B
           GAFX2=RA2R2*(AY2*GZ2-AZ2*GY2)
           GAFY2=RA2R2*(AZ2*GX2-AX2*GZ2)
           GAFZ2=RA2R2*(AX2*GY2-AY2*GX2)
           GBHX2=RB2R2*(BY2*GZ2-BZ2*GY2)
           GBHY2=RB2R2*(BZ2*GX2-BX2*GZ2)
           GBHZ2=RB2R2*(BX2*GY2-BY2*GX2)
           FAGX2=RA2R2*(FY2*AZ2-FZ2*AY2)
           FAGY2=RA2R2*(FZ2*AX2-FX2*AZ2)
           FAGZ2=RA2R2*(FX2*AY2-FY2*AX2)
           HBGX2=RB2R2*(BY2*HZ2-BZ2*HY2)
           HBGY2=RB2R2*(BZ2*HX2-BX2*HZ2)
           HBGZ2=RB2R2*(BX2*HY2-BY2*HX2)
           ! What are the indexes ?
           ! ddE/dX.dY= DDFGH(n)      Fx, Fy, Fz,|Gx, Gy, Gz,|Hx, Hy, Hz. X/
           !                          ----------------------------------- / Y
           !               n=          1   2   4 | 7  11  16 |22  29  37   Fx
           !                               3   5 | 8  12  17 |23  30  38   Fy
           !                                   6 | 9  13  18 |24  31  39   Fz
           !                                     ------------------------
           !                                      10  14  19 |25  32  40   Gx
           !                                          15  20 |26  33  41   Gy
           !                                              21 |27  34  42   Gz
           !                                                 ------------
           !                                                  28  35  43   Hx
           !                                                      36  44   Hy
           !                                                          45   Hz
           !
           ! ddE/dF.dF
           DDFGH(1) =DDF2*DTFX2*DTFX2+TWO*DFX2*GAFX2
           DDFGH(2) =DDF2*DTFX2*DTFY2+DFX2*GAFY2+DFY2*GAFX2
           DDFGH(3) =DDF2*DTFY2*DTFY2+TWO*DFY2*GAFY2
           DDFGH(4) =DDF2*DTFX2*DTFZ2+DFX2*GAFZ2+DFZ2*GAFX2
           DDFGH(5) =DDF2*DTFY2*DTFZ2+DFY2*GAFZ2+DFZ2*GAFY2
           DDFGH(6) =DDF2*DTFZ2*DTFZ2+TWO*DFZ2*GAFZ2
           ! ddE/dF.dG
           DDFGH(7) =DDF2*DTFX2*DTGX2+FAGX2*DFX2-FGRG22*DFX2*GAFX2
           DDFGH(8) =DDF2*DTFY2*DTGX2+FAGY2*DFX2-FGRG22*DFY2*GAFX2
           DDFGH(9) =DDF2*DTFZ2*DTGX2+FAGZ2*DFX2-FGRG22*DFZ2*GAFX2
           DDFGH(11)=DDF2*DTFX2*DTGY2+FAGX2*DFY2-FGRG22*DFX2*GAFY2
           DDFGH(12)=DDF2*DTFY2*DTGY2+FAGY2*DFY2-FGRG22*DFY2*GAFY2
           DDFGH(13)=DDF2*DTFZ2*DTGY2+FAGZ2*DFY2-FGRG22*DFZ2*GAFY2
           DDFGH(16)=DDF2*DTFX2*DTGZ2+FAGX2*DFZ2-FGRG22*DFX2*GAFZ2
           DDFGH(17)=DDF2*DTFY2*DTGZ2+FAGY2*DFZ2-FGRG22*DFY2*GAFZ2
           DDFGH(18)=DDF2*DTFZ2*DTGZ2+FAGZ2*DFZ2-FGRG22*DFZ2*GAFZ2
           ! ddE/dF.dH
           DDFGH(22)=DDF2*DTFX2*DTHX2
           DDFGH(23)=DDF2*DTFY2*DTHX2
           DDFGH(24)=DDF2*DTFZ2*DTHX2
           DDFGH(29)=DDF2*DTFX2*DTHY2
           DDFGH(30)=DDF2*DTFY2*DTHY2
           DDFGH(31)=DDF2*DTFZ2*DTHY2
           DDFGH(37)=DDF2*DTFX2*DTHZ2
           DDFGH(38)=DDF2*DTFY2*DTHZ2
           DDFGH(39)=DDF2*DTFZ2*DTHZ2
           ! ddE/dG.dG
           DDFGH(10)=DDF2*DTGX2*DTGX2-DFRG32*(GAFX2*AX2-GBHX2*BX2) &
                -TWO*FGRG22*DFX2*FAGX2+TWO*HGRG22*DHX2*HBGX2
           DDFGH(14)=DDF2*DTGX2*DTGY2 &
                -HALF*DFRG32*(GAFX2*AY2+GAFY2*AX2-GBHX2*BY2-GBHY2*BX2) &
                -FGRG22*(DFX2*FAGY2+DFY2*FAGX2)+HGRG22*(DHX2*HBGY2+DHY2*HBGX2)
           DDFGH(15)=DDF2*DTGY2*DTGY2-DFRG32*(GAFY2*AY2-GBHY2*BY2) &
                -TWO*FGRG22*DFY2*FAGY2+TWO*HGRG22*DHY2*HBGY2
           DDFGH(19)=DDF2*DTGX2*DTGZ2 &
                -HALF*DFRG32*(GAFX2*AZ2+GAFZ2*AX2-GBHX2*BZ2-GBHZ2*BX2) &
                -FGRG22*(DFX2*FAGZ2+DFZ2*FAGX2)+HGRG22*(DHX2*HBGZ2+DHZ2*HBGX2)
           DDFGH(20)=DDF2*DTGY2*DTGZ2 &
                -HALF*DFRG32*(GAFY2*AZ2+GAFZ2*AY2-GBHY2*BZ2-GBHZ2*BY2) &
                -FGRG22*(DFY2*FAGZ2+DFZ2*FAGY2)+HGRG22*(DHY2*HBGZ2+DHZ2*HBGY2)
           DDFGH(21)=DDF2*DTGZ2*DTGZ2-DFRG32*(GAFZ2*AZ2-GBHZ2*BZ2) &
                -TWO*FGRG22*DFZ2*FAGZ2+TWO*HGRG22*DHZ2*HBGZ2
           ! ddE/dG.dH
           DDFGH(25)=DDF2*DTGX2*DTHX2-DHX2*HBGX2-HGRG22*GBHX2*DHX2
           DDFGH(26)=DDF2*DTGY2*DTHX2-DHY2*HBGX2-HGRG22*GBHY2*DHX2
           DDFGH(27)=DDF2*DTGZ2*DTHX2-DHZ2*HBGX2-HGRG22*GBHZ2*DHX2
           DDFGH(32)=DDF2*DTGX2*DTHY2-DHX2*HBGY2-HGRG22*GBHX2*DHY2
           DDFGH(33)=DDF2*DTGY2*DTHY2-DHY2*HBGY2-HGRG22*GBHY2*DHY2
           DDFGH(34)=DDF2*DTGZ2*DTHY2-DHZ2*HBGY2-HGRG22*GBHZ2*DHY2
           DDFGH(40)=DDF2*DTGX2*DTHZ2-DHX2*HBGZ2-HGRG22*GBHX2*DHZ2
           DDFGH(41)=DDF2*DTGY2*DTHZ2-DHY2*HBGZ2-HGRG22*GBHY2*DHZ2
           DDFGH(42)=DDF2*DTGZ2*DTHZ2-DHZ2*HBGZ2-HGRG22*GBHZ2*DHZ2
           ! ddE/dH.dH
           DDFGH(28)=DDF2*DTHX2*DTHX2+TWO*DHX2*GBHX2
           DDFGH(35)=DDF2*DTHX2*DTHY2+DHX2*GBHY2+DHY2*GBHX2
           DDFGH(36)=DDF2*DTHY2*DTHY2+TWO*DHY2*GBHY2
           DDFGH(43)=DDF2*DTHX2*DTHZ2+DHX2*GBHZ2+DHZ2*GBHX2
           DDFGH(44)=DDF2*DTHY2*DTHZ2+DHY2*GBHZ2+DHZ2*GBHY2
           DDFGH(45)=DDF2*DTHZ2*DTHZ2+TWO*DHZ2*GBHZ2

           II=3*I2-2
           JJ=3*J2-2
           KK=3*K2-2
           LL=3*L2-2
           IJTEST=(J2.LT.I2)
           IKTEST=(K2.LT.I2)
           JKTEST=(K2.LT.J2)
           ILTEST=(L2.LT.I2)
           JLTEST=(L2.LT.J2)
           KLTEST=(L2.LT.K2)
           !
           IADD=IUPT(II)+II
           DD1(IADD)=DD1(IADD)+DDFGH(1)
           IADD=IUPT(II+1)+II+1
           DD1(IADD)=DD1(IADD)+DDFGH(3)
           IADD=IUPT(II+2)+II+2
           DD1(IADD)=DD1(IADD)+DDFGH(6)
           IADD=IUPT(II)+II+1
           DD1(IADD)=DD1(IADD)+DDFGH(2)
           IADD=IUPT(II)+II+2
           DD1(IADD)=DD1(IADD)+DDFGH(4)
           IADD=IUPT(II+1)+II+2
           DD1(IADD)=DD1(IADD)+DDFGH(5)
           !
           IADD=IUPT(LL)+LL
           DD1(IADD)=DD1(IADD)+DDFGH(28)
           IADD=IUPT(LL+1)+LL+1
           DD1(IADD)=DD1(IADD)+DDFGH(36)
           IADD=IUPT(LL+2)+LL+2
           DD1(IADD)=DD1(IADD)+DDFGH(45)
           IADD=IUPT(LL)+LL+1
           DD1(IADD)=DD1(IADD)+DDFGH(35)
           IADD=IUPT(LL)+LL+2
           DD1(IADD)=DD1(IADD)+DDFGH(43)
           IADD=IUPT(LL+1)+LL+2
           DD1(IADD)=DD1(IADD)+DDFGH(44)
           !
           IADD=IUPT(JJ)+JJ
           DD1(IADD)=DD1(IADD)+DDFGH(1)+DDFGH(10)-DDFGH(7)-DDFGH(7)
           IADD=IUPT(JJ+1)+JJ+1
           DD1(IADD)=DD1(IADD)+DDFGH(3)+DDFGH(15)-DDFGH(12)-DDFGH(12)
           IADD=IUPT(JJ+2)+JJ+2
           DD1(IADD)=DD1(IADD)+DDFGH(6)+DDFGH(21)-DDFGH(18)-DDFGH(18)
           IADD=IUPT(JJ)+JJ+1
           DD1(IADD)=DD1(IADD)+DDFGH(2)+DDFGH(14)-DDFGH(11)-DDFGH(8)
           IADD=IUPT(JJ)+JJ+2
           DD1(IADD)=DD1(IADD)+DDFGH(4)+DDFGH(19)-DDFGH(16)-DDFGH(9)
           IADD=IUPT(JJ+1)+JJ+2
           DD1(IADD)=DD1(IADD)+DDFGH(5)+DDFGH(20)-DDFGH(17)-DDFGH(13)
           !
           IADD=IUPT(KK)+KK
           DD1(IADD)=DD1(IADD)+DDFGH(28)+DDFGH(10)+DDFGH(25)+DDFGH(25)
           IADD=IUPT(KK+1)+KK+1
           DD1(IADD)=DD1(IADD)+DDFGH(36)+DDFGH(15)+DDFGH(33)+DDFGH(33)
           IADD=IUPT(KK+2)+KK+2
           DD1(IADD)=DD1(IADD)+DDFGH(45)+DDFGH(21)+DDFGH(42)+DDFGH(42)
           IADD=IUPT(KK)+KK+1
           DD1(IADD)=DD1(IADD)+DDFGH(35)+DDFGH(14)+DDFGH(32)+DDFGH(26)
           IADD=IUPT(KK)+KK+2
           DD1(IADD)=DD1(IADD)+DDFGH(43)+DDFGH(19)+DDFGH(40)+DDFGH(27)
           IADD=IUPT(KK+1)+KK+2
           DD1(IADD)=DD1(IADD)+DDFGH(44)+DDFGH(20)+DDFGH(41)+DDFGH(34)
           !
           IF (IJTEST) THEN
              IADD=IUPT(JJ)+II
              DD1(IADD)=DD1(IADD)+DDFGH(7)-DDFGH(1)
              IADD=IUPT(JJ+1)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(12)-DDFGH(3)
              IADD=IUPT(JJ+2)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(18)-DDFGH(6)
              IADD=IUPT(JJ)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(8)-DDFGH(2)
              IADD=IUPT(JJ+1)+II
              DD1(IADD)=DD1(IADD)+DDFGH(11)-DDFGH(2)
              IADD=IUPT(JJ)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(9)-DDFGH(4)
              IADD=IUPT(JJ+2)+II
              DD1(IADD)=DD1(IADD)+DDFGH(16)-DDFGH(4)
              IADD=IUPT(JJ+1)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(13)-DDFGH(5)
              IADD=IUPT(JJ+2)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(17)-DDFGH(5)
           ELSE
              IADD=IUPT(II)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(7)-DDFGH(1)
              IADD=IUPT(II+1)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(12)-DDFGH(3)
              IADD=IUPT(II+2)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(18)-DDFGH(6)
              IADD=IUPT(II+1)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(8)-DDFGH(2)
              IADD=IUPT(II)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(11)-DDFGH(2)
              IADD=IUPT(II+2)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(9)-DDFGH(4)
              IADD=IUPT(II)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(16)-DDFGH(4)
              IADD=IUPT(II+2)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(13)-DDFGH(5)
              IADD=IUPT(II+1)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(17)-DDFGH(5)
           ENDIF
           !
           IF (IKTEST) THEN
              IADD=IUPT(KK)+II
              DD1(IADD)=DD1(IADD)-DDFGH(7)-DDFGH(22)
              IADD=IUPT(KK+1)+II+1
              DD1(IADD)=DD1(IADD)-DDFGH(12)-DDFGH(30)
              IADD=IUPT(KK+2)+II+2
              DD1(IADD)=DD1(IADD)-DDFGH(18)-DDFGH(39)
              IADD=IUPT(KK)+II+1
              DD1(IADD)=DD1(IADD)-DDFGH(8)-DDFGH(23)
              IADD=IUPT(KK+1)+II
              DD1(IADD)=DD1(IADD)-DDFGH(11)-DDFGH(29)
              IADD=IUPT(KK)+II+2
              DD1(IADD)=DD1(IADD)-DDFGH(9)-DDFGH(24)
              IADD=IUPT(KK+2)+II
              DD1(IADD)=DD1(IADD)-DDFGH(16)-DDFGH(37)
              IADD=IUPT(KK+1)+II+2
              DD1(IADD)=DD1(IADD)-DDFGH(13)-DDFGH(31)
              IADD=IUPT(KK+2)+II+1
              DD1(IADD)=DD1(IADD)-DDFGH(17)-DDFGH(38)
           ELSE
              IADD=IUPT(II)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(7)-DDFGH(22)
              IADD=IUPT(II+1)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(12)-DDFGH(30)
              IADD=IUPT(II+2)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(18)-DDFGH(39)
              IADD=IUPT(II+1)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(8)-DDFGH(23)
              IADD=IUPT(II)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(11)-DDFGH(29)
              IADD=IUPT(II+2)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(9)-DDFGH(24)
              IADD=IUPT(II)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(16)-DDFGH(37)
              IADD=IUPT(II+2)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(13)-DDFGH(31)
              IADD=IUPT(II+1)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(17)-DDFGH(38)
           ENDIF
           !
           IF (ILTEST) THEN
              IADD=IUPT(LL)+II
              DD1(IADD)=DD1(IADD)+DDFGH(22)
              IADD=IUPT(LL+1)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(30)
              IADD=IUPT(LL+2)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(39)
              IADD=IUPT(LL)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(23)
              IADD=IUPT(LL+1)+II
              DD1(IADD)=DD1(IADD)+DDFGH(29)
              IADD=IUPT(LL)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(24)
              IADD=IUPT(LL+2)+II
              DD1(IADD)=DD1(IADD)+DDFGH(37)
              IADD=IUPT(LL+1)+II+2
              DD1(IADD)=DD1(IADD)+DDFGH(31)
              IADD=IUPT(LL+2)+II+1
              DD1(IADD)=DD1(IADD)+DDFGH(38)
           ELSE
              IADD=IUPT(II)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(22)
              IADD=IUPT(II+1)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(30)
              IADD=IUPT(II+2)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(39)
              IADD=IUPT(II+1)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(23)
              IADD=IUPT(II)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(29)
              IADD=IUPT(II+2)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(24)
              IADD=IUPT(II)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(37)
              IADD=IUPT(II+2)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(31)
              IADD=IUPT(II+1)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(38)
           ENDIF
           !
           IF (JKTEST) THEN
              IADD=IUPT(KK)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(7)+DDFGH(22)-DDFGH(10)-DDFGH(25)
              IADD=IUPT(KK+1)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(12)+DDFGH(30)-DDFGH(15)-DDFGH(33)
              IADD=IUPT(KK+2)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(18)+DDFGH(39)-DDFGH(21)-DDFGH(42)
              IADD=IUPT(KK)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(8)+DDFGH(23)-DDFGH(14)-DDFGH(26)
              IADD=IUPT(KK+1)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(11)+DDFGH(29)-DDFGH(14)-DDFGH(32)
              IADD=IUPT(KK)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(9)+DDFGH(24)-DDFGH(19)-DDFGH(27)
              IADD=IUPT(KK+2)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(16)+DDFGH(37)-DDFGH(19)-DDFGH(40)
              IADD=IUPT(KK+1)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(13)+DDFGH(31)-DDFGH(20)-DDFGH(34)
              IADD=IUPT(KK+2)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(17)+DDFGH(38)-DDFGH(20)-DDFGH(41)
           ELSE
              IADD=IUPT(JJ)+KK
              DD1(IADD)=DD1(IADD)+DDFGH(7)+DDFGH(22)-DDFGH(10)-DDFGH(25)
              IADD=IUPT(JJ+1)+KK+1
              DD1(IADD)=DD1(IADD)+DDFGH(12)+DDFGH(30)-DDFGH(15)-DDFGH(33)
              IADD=IUPT(JJ+2)+KK+2
              DD1(IADD)=DD1(IADD)+DDFGH(18)+DDFGH(39)-DDFGH(21)-DDFGH(42)
              IADD=IUPT(JJ+1)+KK
              DD1(IADD)=DD1(IADD)+DDFGH(8)+DDFGH(23)-DDFGH(14)-DDFGH(26)
              IADD=IUPT(JJ)+KK+1
              DD1(IADD)=DD1(IADD)+DDFGH(11)+DDFGH(29)-DDFGH(14)-DDFGH(32)
              IADD=IUPT(JJ+2)+KK
              DD1(IADD)=DD1(IADD)+DDFGH(9)+DDFGH(24)-DDFGH(19)-DDFGH(27)
              IADD=IUPT(JJ)+KK+2
              DD1(IADD)=DD1(IADD)+DDFGH(16)+DDFGH(37)-DDFGH(19)-DDFGH(40)
              IADD=IUPT(JJ+2)+KK+1
              DD1(IADD)=DD1(IADD)+DDFGH(13)+DDFGH(31)-DDFGH(20)-DDFGH(34)
              IADD=IUPT(JJ+1)+KK+2
              DD1(IADD)=DD1(IADD)+DDFGH(17)+DDFGH(38)-DDFGH(20)-DDFGH(41)
           ENDIF
           !
           IF (JLTEST) THEN
              IADD=IUPT(LL)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(25)-DDFGH(22)
              IADD=IUPT(LL+1)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(33)-DDFGH(30)
              IADD=IUPT(LL+2)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(42)-DDFGH(39)
              IADD=IUPT(LL)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(26)-DDFGH(23)
              IADD=IUPT(LL+1)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(32)-DDFGH(29)
              IADD=IUPT(LL)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(27)-DDFGH(24)
              IADD=IUPT(LL+2)+JJ
              DD1(IADD)=DD1(IADD)+DDFGH(40)-DDFGH(37)
              IADD=IUPT(LL+1)+JJ+2
              DD1(IADD)=DD1(IADD)+DDFGH(34)-DDFGH(31)
              IADD=IUPT(LL+2)+JJ+1
              DD1(IADD)=DD1(IADD)+DDFGH(41)-DDFGH(38)
           ELSE
              IADD=IUPT(JJ)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(25)-DDFGH(22)
              IADD=IUPT(JJ+1)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(33)-DDFGH(30)
              IADD=IUPT(JJ+2)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(42)-DDFGH(39)
              IADD=IUPT(JJ+1)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(26)-DDFGH(23)
              IADD=IUPT(JJ)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(32)-DDFGH(29)
              IADD=IUPT(JJ+2)+LL
              DD1(IADD)=DD1(IADD)+DDFGH(27)-DDFGH(24)
              IADD=IUPT(JJ)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(40)-DDFGH(37)
              IADD=IUPT(JJ+2)+LL+1
              DD1(IADD)=DD1(IADD)+DDFGH(34)-DDFGH(31)
              IADD=IUPT(JJ+1)+LL+2
              DD1(IADD)=DD1(IADD)+DDFGH(41)-DDFGH(38)
           ENDIF
           !
           IF (KLTEST) THEN
              IADD=IUPT(LL)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(25)-DDFGH(28)
              IADD=IUPT(LL+1)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(33)-DDFGH(36)
              IADD=IUPT(LL+2)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(42)-DDFGH(45)
              IADD=IUPT(LL)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(26)-DDFGH(35)
              IADD=IUPT(LL+1)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(32)-DDFGH(35)
              IADD=IUPT(LL)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(27)-DDFGH(43)
              IADD=IUPT(LL+2)+KK
              DD1(IADD)=DD1(IADD)-DDFGH(40)-DDFGH(43)
              IADD=IUPT(LL+1)+KK+2
              DD1(IADD)=DD1(IADD)-DDFGH(34)-DDFGH(44)
              IADD=IUPT(LL+2)+KK+1
              DD1(IADD)=DD1(IADD)-DDFGH(41)-DDFGH(44)
           ELSE
              IADD=IUPT(KK)+LL
              DD1(IADD)=DD1(IADD)-DDFGH(25)-DDFGH(28)
              IADD=IUPT(KK+1)+LL+1
              DD1(IADD)=DD1(IADD)-DDFGH(33)-DDFGH(36)
              IADD=IUPT(KK+2)+LL+2
              DD1(IADD)=DD1(IADD)-DDFGH(42)-DDFGH(45)
              IADD=IUPT(KK+1)+LL
              DD1(IADD)=DD1(IADD)-DDFGH(26)-DDFGH(35)
              IADD=IUPT(KK)+LL+1
              DD1(IADD)=DD1(IADD)-DDFGH(32)-DDFGH(35)
              IADD=IUPT(KK+2)+LL
              DD1(IADD)=DD1(IADD)-DDFGH(27)-DDFGH(43)
              IADD=IUPT(KK)+LL+2
              DD1(IADD)=DD1(IADD)-DDFGH(40)-DDFGH(43)
              IADD=IUPT(KK+2)+LL+1
              DD1(IADD)=DD1(IADD)-DDFGH(34)-DDFGH(44)
              IADD=IUPT(KK+1)+LL+2
              DD1(IADD)=DD1(IADD)-DDFGH(41)-DDFGH(44)
           ENDIF



           ! second derivative cross-terms

           INX1(1)=I1
           INX1(2)=J1
           INX1(3)=K1
           INX1(4)=L1

           INX2(1)=I2
           INX2(2)=J2
           INX2(3)=K2
           INX2(4)=L2

           DTT1MX(1)=ZERO
           DTT1MY(1)=ZERO
           DTT1MZ(1)=ZERO
           DTT1PX(1)=DTFX1
           DTT1PY(1)=DTFY1
           DTT1PZ(1)=DTFZ1

           DTT1MX(2)=-DTFX1
           DTT1MY(2)=-DTFY1
           DTT1MZ(2)=-DTFZ1
           DTT1PX(2)=DTGX1
           DTT1PY(2)=DTGY1
           DTT1PZ(2)=DTGZ1

           DTT1MX(3)=-DTGX1
           DTT1MY(3)=-DTGY1
           DTT1MZ(3)=-DTGZ1
           DTT1PX(3)=-DTHX1
           DTT1PY(3)=-DTHY1
           DTT1PZ(3)=-DTHZ1

           DTT1MX(4)=DTHX1
           DTT1MY(4)=DTHY1
           DTT1MZ(4)=DTHZ1
           DTT1PX(4)=ZERO
           DTT1PY(4)=ZERO
           DTT1PZ(4)=ZERO

           DTT2MX(1)=ZERO
           DTT2MY(1)=ZERO
           DTT2MZ(1)=ZERO
           DTT2PX(1)=DTFX2
           DTT2PY(1)=DTFY2
           DTT2PZ(1)=DTFZ2

           DTT2MX(2)=-DTFX2
           DTT2MY(2)=-DTFY2
           DTT2MZ(2)=-DTFZ2
           DTT2PX(2)=DTGX2
           DTT2PY(2)=DTGY2
           DTT2PZ(2)=DTGZ2

           DTT2MX(3)=-DTGX2
           DTT2MY(3)=-DTGY2
           DTT2MZ(3)=-DTGZ2
           DTT2PX(3)=-DTHX2
           DTT2PY(3)=-DTHY2
           DTT2PZ(3)=-DTHZ2

           DTT2MX(4)=DTHX2
           DTT2MY(4)=DTHY2
           DTT2MZ(4)=DTHZ2
           DTT2PX(4)=ZERO
           DTT2PY(4)=ZERO
           DTT2PZ(4)=ZERO

           DO II1=1,4
              III1=INX1(II1)
              DO II2=1,4
                 III2=INX2(II2)
                 II=3*III1-2
                 JJ=3*III2-2
                 IF (III1.EQ.III2) THEN
                    IADD=IUPT(II)+II
                    DD1(IADD)=DD1(IADD)+ &
                         TWO*DDF12*(DTT1MX(II1)*DTT2MX(II2)+DTT1MX(II1)*DTT2PX(II2)+ &
                         DTT1PX(II1)*DTT2MX(II2)+DTT1PX(II1)*DTT2PX(II2))
                    IADD=IUPT(II+1)+II+1
                    DD1(IADD)=DD1(IADD)+ &
                         TWO*DDF12*(DTT1MY(II1)*DTT2MY(II2)+DTT1MY(II1)*DTT2PY(II2)+ &
                         DTT1PY(II1)*DTT2MY(II2)+DTT1PY(II1)*DTT2PY(II2))
                    IADD=IUPT(II+2)+II+2
                    DD1(IADD)=DD1(IADD)+ &
                         TWO*DDF12*(DTT1MZ(II1)*DTT2MZ(II2)+DTT1MZ(II1)*DTT2PZ(II2)+ &
                         DTT1PZ(II1)*DTT2MZ(II2)+DTT1PZ(II1)*DTT2PZ(II2))
                    IADD=IUPT(II)+II+1
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MX(II1)*DTT2MY(II2)+DTT1MX(II1)*DTT2PY(II2)+ &
                         DTT1PX(II1)*DTT2MY(II2)+DTT1PX(II1)*DTT2PY(II2)+ &
                         DTT1MY(II1)*DTT2MX(II2)+DTT1MY(II1)*DTT2PX(II2)+ &
                         DTT1PY(II1)*DTT2MX(II2)+DTT1PY(II1)*DTT2PX(II2))
                    IADD=IUPT(II)+II+2
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MX(II1)*DTT2MZ(II2)+DTT1MX(II1)*DTT2PZ(II2)+ &
                         DTT1PX(II1)*DTT2MZ(II2)+DTT1PX(II1)*DTT2PZ(II2)+ &
                         DTT1MZ(II1)*DTT2MX(II2)+DTT1MZ(II1)*DTT2PX(II2)+ &
                         DTT1PZ(II1)*DTT2MX(II2)+DTT1PZ(II1)*DTT2PX(II2))
                    IADD=IUPT(II+1)+II+2
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MZ(II1)*DTT2MY(II2)+DTT1MZ(II1)*DTT2PY(II2)+ &
                         DTT1PZ(II1)*DTT2MY(II2)+DTT1PZ(II1)*DTT2PY(II2)+ &
                         DTT1MY(II1)*DTT2MZ(II2)+DTT1MY(II1)*DTT2PZ(II2)+ &
                         DTT1PY(II1)*DTT2MZ(II2)+DTT1PY(II1)*DTT2PZ(II2))
                 ELSE IF (III1.LT.III2) THEN
                    IADD=IUPT(II)+JJ
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MX(II1)*DTT2MX(II2)+DTT1MX(II1)*DTT2PX(II2)+ &
                         DTT1PX(II1)*DTT2MX(II2)+DTT1PX(II1)*DTT2PX(II2))
                    IADD=IUPT(II+1)+JJ
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MY(II1)*DTT2MX(II2)+DTT1MY(II1)*DTT2PX(II2)+ &
                         DTT1PY(II1)*DTT2MX(II2)+DTT1PY(II1)*DTT2PX(II2))
                    IADD=IUPT(II+2)+JJ
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MZ(II1)*DTT2MX(II2)+DTT1MZ(II1)*DTT2PX(II2)+ &
                         DTT1PZ(II1)*DTT2MX(II2)+DTT1PZ(II1)*DTT2PX(II2))
                    IADD=IUPT(II)+JJ+1
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MX(II1)*DTT2MY(II2)+DTT1MX(II1)*DTT2PY(II2)+ &
                         DTT1PX(II1)*DTT2MY(II2)+DTT1PX(II1)*DTT2PY(II2))
                    IADD=IUPT(II+1)+JJ+1
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MY(II1)*DTT2MY(II2)+DTT1MY(II1)*DTT2PY(II2)+ &
                         DTT1PY(II1)*DTT2MY(II2)+DTT1PY(II1)*DTT2PY(II2))
                    IADD=IUPT(II+2)+JJ+1
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MZ(II1)*DTT2MY(II2)+DTT1MZ(II1)*DTT2PY(II2)+ &
                         DTT1PZ(II1)*DTT2MY(II2)+DTT1PZ(II1)*DTT2PY(II2))
                    IADD=IUPT(II)+JJ+2
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MX(II1)*DTT2MZ(II2)+DTT1MX(II1)*DTT2PZ(II2)+ &
                         DTT1PX(II1)*DTT2MZ(II2)+DTT1PX(II1)*DTT2PZ(II2))
                    IADD=IUPT(II+1)+JJ+2
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MY(II1)*DTT2MZ(II2)+DTT1MY(II1)*DTT2PZ(II2)+ &
                         DTT1PY(II1)*DTT2MZ(II2)+DTT1PY(II1)*DTT2PZ(II2))
                    IADD=IUPT(II+2)+JJ+2
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MZ(II1)*DTT2MZ(II2)+DTT1MZ(II1)*DTT2PZ(II2)+ &
                         DTT1PZ(II1)*DTT2MZ(II2)+DTT1PZ(II1)*DTT2PZ(II2))
                 ELSE
                    IADD=IUPT(JJ)+II
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MX(II1)*DTT2MX(II2)+DTT1MX(II1)*DTT2PX(II2)+ &
                         DTT1PX(II1)*DTT2MX(II2)+DTT1PX(II1)*DTT2PX(II2))
                    IADD=IUPT(JJ+1)+II
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MX(II1)*DTT2MY(II2)+DTT1MX(II1)*DTT2PY(II2)+ &
                         DTT1PX(II1)*DTT2MY(II2)+DTT1PX(II1)*DTT2PY(II2))
                    IADD=IUPT(JJ+2)+II
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MX(II1)*DTT2MZ(II2)+DTT1MX(II1)*DTT2PZ(II2)+ &
                         DTT1PX(II1)*DTT2MZ(II2)+DTT1PX(II1)*DTT2PZ(II2))
                    IADD=IUPT(JJ)+II+1
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MY(II1)*DTT2MX(II2)+DTT1MY(II1)*DTT2PX(II2)+ &
                         DTT1PY(II1)*DTT2MX(II2)+DTT1PY(II1)*DTT2PX(II2))
                    IADD=IUPT(JJ+1)+II+1
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MY(II1)*DTT2MY(II2)+DTT1MY(II1)*DTT2PY(II2)+ &
                         DTT1PY(II1)*DTT2MY(II2)+DTT1PY(II1)*DTT2PY(II2))
                    IADD=IUPT(JJ+2)+II+1
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MY(II1)*DTT2MZ(II2)+DTT1MY(II1)*DTT2PZ(II2)+ &
                         DTT1PY(II1)*DTT2MZ(II2)+DTT1PY(II1)*DTT2PZ(II2))
                    IADD=IUPT(JJ)+II+2
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MZ(II1)*DTT2MX(II2)+DTT1MZ(II1)*DTT2PX(II2)+ &
                         DTT1PZ(II1)*DTT2MX(II2)+DTT1PZ(II1)*DTT2PX(II2))
                    IADD=IUPT(JJ+1)+II+2
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MZ(II1)*DTT2MY(II2)+DTT1MZ(II1)*DTT2PY(II2)+ &
                         DTT1PZ(II1)*DTT2MY(II2)+DTT1PZ(II1)*DTT2PY(II2))
                    IADD=IUPT(JJ+2)+II+2
                    DD1(IADD)=DD1(IADD)+ &
                         DDF12*(DTT1MZ(II1)*DTT2MZ(II2)+DTT1MZ(II1)*DTT2PZ(II2)+ &
                         DTT1PZ(II1)*DTT2MZ(II2)+DTT1PZ(II1)*DTT2PZ(II2))
                 ENDIF
              ENDDO
           ENDDO
        ENDIF

#if KEY_BLOCK==1
     ENDIF
#endif /*  BLOCK*/
     !
  enddo loop10
  !
  NWARN=NWARN+NWARNX
  IF(NWARN.GT.5 .AND. WRNLEV.GE.2) WRITE(OUTU,175) NWARN
175 FORMAT(' TOTAL OF',I6,' WARNINGS FROM ECMAP')
  !
  return
end SUBROUTINE ECMAP

#endif /* (cmap_main)*/
end module cmapm

