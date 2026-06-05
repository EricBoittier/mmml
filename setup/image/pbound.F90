!
!  This code was developed by William A. Shirley and Charles L. Brooks, III,
!  Department of Molecular Biology, The Scripps Research Institute, 
!  during the spring of 1995.
!
!  Its purpose is to speed up calculations which use the periodic boundry 
!  conditions for A TRUNCATED OCTAHEDRAL BOX; A RHOMBIC DODECAHEDRAL BOX;
!  A TWO AND THREE DIMENSIONAL RHOMBOIDAL BOX.
!
!  Reference:
!  M. P. Allen and D. J. Tildesley, Computer Simulation of Liquids, Ch. 1.
!
!  The basic code is implemented following the outline:
!  
!  1. Parse the control variables based upon the keyword BOUNd.
!     These include:
!
!     To initiate PBound routines
!
!     BOUNd  {TOBOUN}  BOXL <real>  CUTNB <real>
!            {RDBOUN}
!            {RHBOUN}
!            {CUBOUN}  {XSIZe <real> [YSIZe <real>][ ZSIZe <real>}]
!            {OFF}
!
!     where
!
!             TOBOUN =        TruncatedOctahedralBOUNd
!             RDBOUN =        RhombicDodecahedralBOUNd
!             RHBOUN = Two dimensional RHomboidalBOUNd
!             CUBOUN =                      CUbicBOUNd
!  
!     and BOXL <real> and CUTNB <real> are required.
! Modification by LNI, nov 1995.
!     use of XSIZe, YSIZe, and ZSIZe instead of BOXL just gives a box with
!     different X, Y and Z sidelengths. Specifying just XSIZe is the same
!     as BOXL. specifying XSIZe and YSIZE gives ZSIZe=YSIZe

!
!     It sets up the control flags.  These flags turn off the existing
!     image generation code during run time.
!
!  2. Do not generate image atoms to add to the PSF.  When UPIMAGE 
!     is called, skip MKIMAT.  Set the completion flag in NBONDM.
!  
!  3. During the generation of the nonbonded list use the MI (minmum
!     image) in NBONDG to get correct pairs onto the list.
!
!  4. During the calculation of the nonbonded energies in 
!     ENBFAST, adjust the distances using MI to get the correct
!     VDW and Elec energies.
!
!
!     Files Modified:
!     charmm/charmm_main.src      to recognize bound commands
!     image/upimag.src            to skip mkimat
!           nbondm.src            to fake a complete image update
!                                 also to implement MI
!     nbonds/enbfast.src          to implement MI
!            nbondg.src           to implement MI:
!                                 1. make sure groups are not excluded
!                                 2. do MI to atoms
!
!     File added:
!     nbonds/pbound.src
!     fcm/pbound.f90              hardwired periodic boundary variables
!
#if KEY_PBOUND==0 /*not_pbound*/
Subroutine Bound(COMLYN,COMLEN)
  Character(len=*) COMLYN
  Integer       COMLEN
  CALL WRNDIE(-1,'<PBOUND>', &
       'Hardwired Periodic Boundry code not compiled.')
  RETURN
End Subroutine Bound
#else /*       (not_pbound)*/
Subroutine Bound(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This is the main calling routine for the faster hardwired periodic
  !     boundary procedures.
  !  
  use chm_kinds
  use number
  use pbound
  use stream
  use string
  use inbnd,only:cutnb

  implicit none

  Character(len=*) COMLYN
  Integer       COMLEN

  !     parse the boundary conditions
  qBoun   = .true.
  qTOBoun = .false.
  qRDBoun = .false.
  qRHBoun = .false.
  qCUBoun = .false.
  If (IndxA(comLyn,comLen,'TOBOUN').GT.0) qTOBoun = .true.
  If (IndxA(comLyn,comLen,'RDBOUN').GT.0) qRDBoun = .true.
  If (IndxA(comLyn,comLen,'RHBOUN').GT.0) qRHBoun = .true.
  If (IndxA(comLyn,comLen,'CUBOUN').GT.0) qCUBoun = .true.
  If (IndxA(comLyn,comLen,'OFF').GT.0)    qBoun   = .false.
  !
  RCUTB =GTRMF(COMLYN,COMLEN,'CUTNB',ZERO)
  XSIZE =GTRMF(COMLYN,COMLEN, 'BOXL',ZERO) 
  XSIZE =GTRMF(COMLYN,COMLEN, 'XSIZ',XSIZE)
  YSIZE =GTRMF(COMLYN,COMLEN, 'YSIZ',XSIZE)
  ZSIZE =GTRMF(COMLYN,COMLEN, 'ZSIZ',XSIZE)

  !     write info. of the current boundary
  IF (PRNLEV.GE.2) THEN
     Write(OutU,'(A)') ' Hardwired Periodic Boundaries: '
     If(qTOBoun) Write(OutU,'(/,A)') '   A TRUNCATED OCTAHEDRAL BOX'
     If(qRDBoun) Write(OutU,'(/,A)') '   A RHOMBIC DODECAHEDRAL BOX'
     If(qRHBoun) Write(OutU,'(/,A)') &
          '   A TWO DIMENSIONAL RHOMBOIDAL BOX'
     If(qCUBoun) Write(OutU,'(/,A)') &
          '   A THREE DIMENSIONAL RHOMBOIDAL BOX'
     If(.not.qBoun) Write(OutU,'(A)') '              Turned Off'
  ENDIF
  IF(.NOT. QBOUN) RETURN  
  If (XSIZE*YSIZE*ZSIZE .LE. ZERO) Then
     CALL WRNDIE(-1,'<PBOUND>', &
          'BOXL(X/Y/Zsize) must be provided with the BOUND command.')
  Else
     BOXINV=ONE/XSIZE
     BOYINV=ONE/YSIZE
     BOZINV=ONE/ZSIZE
     IF (PRNLEV.GE.2)THEN 
        Write(OutU,'(A,3F9.4)') ' X,Y,Z-sides=',XSIZE,YSIZE,ZSIZE
     ENDIF
  EndIf
  If (RCutB .eq. ZERO) Then
     CALL WRNDIE(-1,'<PBOUND>', &
          'CUTNB must be provided with the BOUND command.')
  Else
     IF (PRNLEV.GE.2) Write(OutU,'(A9,F9.4)') '  CUTNB = ', RCutB
  EndIf

  cutnb = rcutb

  RCutSQB = RCutB * RCutB
  OutUnit = OutU

  Return
End Subroutine Bound

Subroutine PBMove(XD, YD, ZD)
  !-----------------------------------------------------------------------
  use chm_kinds
  use pbound
  use number
  implicit none
  !
  !     Vector between two points in space:
  real(chm_real) XD, YD, ZD

  !     Local variables
  real(chm_real) CORR
  real(chm_real) TMPVAR1,TMPVAR2

  XD = XD * BOXINV
  YD = YD * BOYINV
  ZD = ZD * BOZINV

  If  (qRDBoun) Then
     XD      = XD    - INT(XD + SIGN(HALF,XD))
     YD      = YD    - INT(YD + SIGN(HALF,YD))
     ZD      = ZD    - RT2*INT(RRT2*ZD + SIGN(HALF,RRT2*ZD))
     !NX        XD      =  XD   - DNINT (  XD  )
     !NX        YD      =  YD   - DNINT (  YD  )
     !NX        ZD      =  ZD   - RT2 * DNINT ( RRT2 *  ZD  )
     CORR = HALF * INT ( ( ABS (  XD   ) + &
          ABS (  YD   ) + &
          RT2 * ABS (  ZD   ) ) )
     XD      =  XD     - SIGN ( CORR,  XD   )
     YD      =  YD     - SIGN ( CORR,  YD   )
     ZD      =  ZD     - SIGN ( CORR,  ZD   ) * RT2
  Else If  (qRHBoun) Then
     TMPVAR1 = XD   - RRT3 *  YD
     TMPVAR2 = RRT32 *  YD
     TMPVAR2 =INT(TMPVAR2 + SIGN(HALF,TMPVAR2))
     XD      = XD    - INT(TMPVAR1 + SIGN(HALF,TMPVAR1)) &
          - TMPVAR2*HALF
     YD      = YD    - TMPVAR2*RT32
     !NX        XD      = XD    - DNINT ( XD   - RRT3 *  YD  )
     !NX     $                  - DNINT ( RRT32 *  YD   ) * HALF
     !NX        YD      = YD    - DNINT ( RRT32 *  YD   ) * RT32
  Else 
     CALL WRNDIE(-1,'<PBOUND>', &
          ' Periodic boundaries not defined.')
  EndIf
  XD = XD * XSIZE
  YD = YD * YSIZE
  ZD = ZD * ZSIZE
  Return
END Subroutine PBMove

#endif /*  (not_pbound)*/

SUBROUTINE CRDMOV(NATOM,X,Y,Z,XOLD,YOLD,ZOLD,NRES,IBASE)
  !-----------------------------------------------------------------------
  use chm_kinds
  use number
  !
  use pbound
  use contrl
  implicit none
  !
  ! JG 5/02
  INTEGER NATOM,NRES,IBASE(*)
  real(chm_real) X(*),Y(*),Z(*),XOLD(*),YOLD(*),ZOLD(*)
  !
  ! local
  INTEGER I,J,I1,I2
  real(chm_real) XTRANS,YTRANS,ZTRANS,XEDGE,YEDGE,ZEDGE, &
       XEDG2,YEDG2,ZEDG2
  real(chm_real) XCM,YCM,ZCM,XDEL,YDEL,ZDEL
  !
#if KEY_PBOUND==1 /*pbound*/
  !
  ! First Locate Center of entire system
  XCM=ZERO
  YCM=ZERO
  ZCM=ZERO
  DO I = 1,NCRES
     I1 = IBASE(I)+1
     I2 = IBASE(I+1)
     DO J = I1,I2
        XCM = XCM + X(J)
        YCM = YCM + Y(J)
        ZCM = ZCM + Z(J)
     ENDDO
  ENDDO
  XCM = XCM/DBLE(IBASE(NCRES+1)-IBASE(1))
  YCM = YCM/DBLE(IBASE(NCRES+1)-IBASE(1))
  ZCM = ZCM/DBLE(IBASE(NCRES+1)-IBASE(1))
  XTRANS = -XCM
  YTRANS = -YCM
  ZTRANS = -ZCM
  !
  XEDGE = XSIZE
  YEDGE = YSIZE
  ZEDGE = ZSIZE
  !
  XEDG2 = 0.5*XEDGE
  YEDG2 = 0.5*YEDGE
  ZEDG2 = 0.5*ZEDGE
  !  Translate everything
  DO I = 1,NATOM
     X(I) = X(I)+XTRANS
     Y(I) = Y(I)+YTRANS
     Z(I) = Z(I)+ZTRANS
     XOLD(I) = XOLD(I)+XTRANS
     YOLD(I) = YOLD(I)+YTRANS
     ZOLD(I) = ZOLD(I)+ZTRANS
  ENDDO
  !         write(6,*)' XTRANS =',XTRANS,' YTRANS =',YTRANS
  !         write(6,*)' ZTRANS =',ZTRANS
  !  Check if molecule is out side of the unit cell. If outside,
  !  move back into the unit cell
  !
  DO I = NCRES+1,NRES
     I1 = IBASE(I)+1
     I2 = IBASE(I+1)
     XCM=ZERO
     YCM=ZERO
     ZCM=ZERO
     DO J = I1,I2
        XCM = XCM + X(J)
        YCM = YCM + Y(J)
        ZCM = ZCM + Z(J)
     ENDDO
     XCM = XCM/DBLE(I2-I1+1)
     YCM = YCM/DBLE(I2-I1+1)
     ZCM = ZCM/DBLE(I2-I1+1)
     !
     IF(DABS(XCM) > XEDG2) THEN
        XDEL = DSIGN(XEDGE,XCM)
        DO J = I1,I2
           X(J) = X(J) - XDEL
           XOLD(J) = XOLD(J) - XDEL
        ENDDO
     END IF
     IF (DABS(YCM) > YEDG2) THEN
        YDEL = DSIGN(YEDGE,YCM)
        DO J = I1, I2
           Y(J) = Y(J) - YDEL
           YOLD(J) = YOLD(J) - YDEL
        ENDDO
     ENDIF
     IF (DABS(ZCM) > ZEDG2) THEN
        ZDEL = DSIGN(ZEDGE,ZCM)
        DO J = I1, I2
           Z(J) = Z(J) - ZDEL
           ZOLD(J) = ZOLD(J) - ZDEL
        ENDDO
     ENDIF
  ENDDO
#endif /* (pbound)*/
  !
  RETURN
END SUBROUTINE CRDMOV

