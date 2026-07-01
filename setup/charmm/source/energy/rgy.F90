module rgym
  use chm_kinds
  use dimens_fcm
  implicit none
!  FORCERG  Force constant for Rgy potential
!  RGY0     Reference Rgy for Rgy potential
!  DUNITRG  Output unit for Rgy trajectory information
!  NRGY     Number of atoms selected for Rgy potential
!  NSVRGY   save frequency for RGY writes
!  NCOUNTRG counter for dynamics steps in Rgy rouutines
!  DSTEPRG  current step number from dynamics
!  RGYSLCT  pointer for Rgy selection
!  QRGY     Logical flag for presence of Rgy potential
!  QSAVRG   Logical flag for whether Rgy info written to output
!  RmsdRef  Reference coordinates for rmsd restraint
!  Qrmsd    Logical flag indicating RMSD restraint instead of Rgy restraint
!  LOrie    Logical flag indicating RMSD restraint to minimum RMSD oriented
!           structure.
#if KEY_RGYCONS==1
   real(chm_real) FORCERG, RGY0
   INTEGER DUNITRG, NRGY, NSVRGY, NCOUNTRG
   Integer DSTEPRG
   integer,allocatable,dimension(:) :: RGYSLCT
   real(chm_real),allocatable,dimension(:,:) :: RmsdRef
   LOGICAL QRGY, QSAVRG, QRmsd, LOrie

contains

  subroutine rgy_iniall()
    qsavrg=.false.
    qrgy=.false.
    nrgy = 0
    return
  end subroutine rgy_iniall


   SUBROUTINE RGYSET
!
! Sets up Radius of GYration constraint force field.
!
! Author: Charles L. Brooks III
! Overhauled for fortran by Erik Boczko (Were you expecting a BRB ?)
!
!   SYNTAX:  RGYRation FORCe real REFErence real RMSD COMP ORIE OUTPut_unit integer
!            NSAVe_output integer SELE { atom selection } END
!     or     RGYR RESET
!
  use psf
  use comand
  use coord
  use coordc
  use select
  use stream
  use string
  use number
  use exfunc
  use memory
  use parallel
      implicit none
!
! local
      INTEGER I
      integer,allocatable,dimension(:) :: SLCT
      LOGICAL LRESET, QComp
!
! parse the command line info
!
      LRESET = ( INDXA(COMLYN, COMLEN, 'RESE') .GT. 0 )
      IF( QRGY .AND. LRESET ) THEN
         QSAVRG = .FALSE.
         QRGY = .FALSE.
         If (Qrmsd) Then
            call chmdealloc('rgy.src','RGYSET','RmsdRef',3,NRgy,crl=RmsdRef)
         Else
            call chmdealloc('rgy.src','RGYSET','RmsdRef',3,1,crl=RmsdRef)
         Endif
         call chmdealloc('rgy.src','RGYSET','RGYSLCT',NRGY,intg=RGYSLCT)
         NRGY = 0
      ELSE
         FORCERG = GTRMF(COMLYN,COMLEN,'FORC',ZERO)
         RGY0 = GTRMF(COMLYN,COMLEN,'REFE',ZERO)
         DUNITRG = GTRMI(COMLYN,COMLEN,'OUTP',6)
         NSVRGY = GTRMI(COMLYN,COMLEN,'NSAV',0)
         Qrmsd = .false.
         QComp = .false.
         QRMSD = (IndxA(Comlyn, Comlen,'RMSD').gt.0)
         QCOMP = (IndxA(Comlyn, Comlen,'COMP').gt.0)
         LORIE = (IndxA(Comlyn, Comlen,'ORIE').gt.0)
!  Select atoms for RGY restraint
         call chmalloc('rgy.src','RGYSET','SLCT',NATOM,intg=SLCT)
         CALL SELCTA(COMLYN,COMLEN,SLCT,X,Y,Z,WMAIN,.TRUE.)
         CALL CNTRG0(SLCT, NATOM)
         call chmalloc('rgy.src','RGYSET','RGYSLCT',NRGY,intg=RGYSLCT)
         If (QRmsd) Then
            call chmalloc('rgy.src','RGYSET','RmsdRef',3,NRGY,crl=RmsdRef)
         Else
            call chmalloc('rgy.src','RGYSET','RmsdRef',3,1,crl=RmsdRef)
         Endif
         If (QComp) Then
            CALL SETSLRG(SLCT, NATOM, XComp, YComp, ZComp)
         Else
            CALL SETSLRG(SLCT, NATOM, X, Y, Z)
         Endif
         call chmdealloc('rgy.src','RGYSET','SLCT',NATOM,intg=SLCT)

         IF((NSVRGY.GT.0).AND.(DUNITRG.GT.0)) QSAVRG = .TRUE.
         IF((QSAVRG.OR.(FORCERG.GE.ZERO)).AND.(NRGY.GT.0)) QRGY = .TRUE.
      ENDIF
!
! Print the info
!
      IF(PRNLEV.GE.5.AND.MYNOD.EQ.0) THEN
        IF(QRGY) THEN
          If (QRmsd) Then
             Write(OutU,'(a)') &
                  'RGYSET : An RMSD restraint has been set'
             WRITE(OUTU,1) FORCERG,RGY0,NSVRGY,DUNITRG,NRGY
             If (QCOMP) Write(Outu,'(a)') &
       ' RGYSET: Restraint reference coordinates from comparison set'
             If (LORIE) Write(Outu,'(a)') &
       ' RGYSET:  Restraint applied to RMSD minimum oriented reference'
          Else
             WRITE(OUTU,'(a)') &
                  'RGYSET : An rgy restraint has been set'
             WRITE(OUTU,1) FORCERG,RGY0,NSVRGY,DUNITRG,NRGY
          Endif
        ELSE
           WRITE(OUTU,'(a)') &
                ' RGYSET : No constraint set nothing done '
        ENDIF
      ENDIF

 1    FORMAT('RGYSET: FORCe = ',F10.6,' REFE = ',F10.6,' NSAVe = ',I4, &
           ' ON UNIT = ',I4,' # ATOMS RESTRAINED = ',I8)

      RETURN
   END SUBROUTINE RGYSET

   SUBROUTINE CNTRG0(SLCT, NATOM)
      implicit none
      INTEGER SLCT(*), NATOM, I
      DO I=1,NATOM
         IF(SLCT(I).EQ.1) NRGY = NRGY + 1
      ENDDO
      RETURN
   END SUBROUTINE CNTRG0

   SUBROUTINE SETSLRG(SLCT, NATOM, X, Y, Z)
  use number
      implicit none
      INTEGER SLCT(*), NATOM, I, NG
      real(chm_real) X(*), Y(*), Z(*)
      real(chm_real) Xcg, Ycg, Zcg
      NG = 0
      Xcg = Zero
      Ycg = Zero
      Zcg = Zero
      DO I=1,NATOM
         IF(SLCT(I).EQ.1) THEN
            NG = NG + 1
            RGYSLCT(NG) = I
            If (QRmsd) Then
               RmsdRef(1,Ng) = X(i)
               RmsdRef(2,Ng) = Y(i)
               RmsdRef(3,Ng) = Z(i)
               Xcg = Xcg + X(i)
               Ycg = Ycg + Y(i)
               Zcg = Zcg + Z(i)
!        write(6,*)' Defining for atom ', i,
!     &  ' Xcg= ', Ref(1,Ng), ' Ycg= ', Ref(2,Ng), ' Zcg= ', Ref(3,Ng)
            Endif
         ENDIF
      ENDDO

      Xcg = Xcg / Ng
      Ycg = Ycg / Ng
      Zcg = Zcg / Ng
      If ( LOrie ) Then
         Do I = 1, Ng
            RmsdRef(1,I) = RmsdRef(1,I) - Xcg
            RmsdRef(2,I) = RmsdRef(2,I) - Ycg
            RmsdRef(3,I) = RmsdRef(3,I) - Zcg
         Enddo
      Endif
      IF( ( NG - NRGY ) .NE. 0 ) THEN
        CALL WRNDIE(0,'<SETSLRG>','SELECTION ERROR')
      ENDIF
      RETURN
   END SUBROUTINE SETSLRG

   SUBROUTINE ERGY(ECRGY,X,Y,Z,DX,DY,DZ)
!
!      THIS ROUTINE ADDS A Quadratic POTENTIAL to restrain the
!      Radius of GYration or the RMSD wrt a target.
!      The Radius of GYration is defined
!      with respect to the center of geometry (CG) of selected atoms.
!
!                               0  2
!      E= 1/2 * CONST * (R   - R  )
!                         GY    GY
!
!  where
!      2                      2
!     R  = 1/N SUM ( r  - R  )
!      GY       i     i    CG
!
!  and
!
!     R  = 1/N SUM ( r )
!      CG       i     i
!
!     The RMSD restraint is identical in form except the
!     center of geometry is replaced by a set of reference coorinates.

!     By Charles L. Brooks III, March, 1990
!
  use stream
  use number
  use contrl
  use corsubs,only:frotu
      implicit none

      real(chm_real) ECRGY
      real(chm_real) X(*),Y(*),Z(*)
      real(chm_real) DX(*),DY(*),DZ(*)

      INTEGER I, K, KB
      real(chm_real) XCG, YCG, ZCG, CON, RGY, DEN, EVA(3),DEVA(3,3)
      real(chm_real) U(9), R(9), Cmxc, Cmyc, Xr, Yr, Zr
      real(chm_real) Xi, Yi, Zi, Xj, Yj, Zj
      LOGICAL WFLAG, QEVW

      DEN = NRGY
      ECRGY = ZERO
!
!     FIND THE CENTER OF GEOMETRY
!
      If ( ( .not. Qrmsd ) .or. ( QRmsd .and. LOrie ) ) Then
         XCG=Zero
         YCG=Zero
         ZCG=Zero
         DO I=1,NRGY
            Kb = RgySLct(I)
            XCG=XCG+X(Kb)
            YCG=YCG+Y(Kb)
            ZCG=ZCG+Z(Kb)
         ENDDO
         XCG=XCG/DEN
         YCG=YCG/DEN
         ZCG=ZCG/DEN

         If ( LOrie ) Then
            Xr = Zero
            Yr = Zero
            Zr = Zero
            DO I=1,NRGY
               Xr = Xr + RmsdRef(1,I)
               Yr = Yr + RmsdRef(2,I)
               Zr = Zr + RmsdRef(3,I)
            ENDDO
            Xr = Xr / Den
            Yr = Yr / Den
            Zr = Zr / Den

!
!       COMPUTE ROTATION MATRIX FROM LAGRANGIAN
!
            DO I=1,9
               R(I)=Zero
            ENDDO
            DO K=1,NRgy
               KB = RgySLct(K)
               RmsdRef(1,K) = RmsdRef(1,K) - Xr
               RmsdRef(2,K) = RmsdRef(2,K) - Yr
               RmsdRef(3,K) = RmsdRef(3,K) - Zr
               XI = RmsdRef(1,K)
               YI = RmsdRef(2,K)
               ZI = RmsdRef(3,K)
               XJ = X(KB) - Xcg
               YJ = Y(KB) - Ycg
               ZJ = Z(KB) - Zcg
               R(1)=R(1)+XI*XJ
               R(2)=R(2)+XI*YJ
               R(3)=R(3)+XI*ZJ
               R(4)=R(4)+YI*XJ
               R(5)=R(5)+YI*YJ
               R(6)=R(6)+YI*ZJ
               R(7)=R(7)+ZI*XJ
               R(8)=R(8)+ZI*YJ
               R(9)=R(9)+ZI*ZJ
            ENDDO

            CALL FROTU(R, EVA, DEVA, U , ZERO, QEVW, .False.)

            DO K=1,NRgy
               CMXC=U(1)*RmsdRef(1,K)+U(4)*RmsdRef(2,K)+U(7)*RmsdRef(3,K)+Xcg
               CMYC=U(2)*RmsdRef(1,K)+U(5)*RmsdRef(2,K)+U(8)*RmsdRef(3,K)+Ycg
               RmsdRef(3,K)=U(3)*RmsdRef(1,K)+U(6)*RmsdRef(2,K)+U(9)*RmsdRef(3,K)+Zcg
               RmsdRef(1,K)=CMXC
               RmsdRef(2,K)=CMYC
            ENDDO
         Endif
      Endif
!
!     FIND RGY
!
      RGY = Zero
      DO I=1,NRGY
         If (Qrmsd) Then
            Xcg = RmsdRef(1,I)
            Ycg = RmsdRef(2,I)
            Zcg = RmsdRef(3,I)
!            write(6,*)' For atom ', i,
!     &      ' Xcg= ', xcg, ' Ycg= ', ycg, ' Zcg= ', zcg
         Endif
         RGY = RGY + (X(RGYSLCT(I)) - XCG)*(X(RGYSLCT(I)) - XCG)
         RGY = RGY + (Y(RGYSLCT(I)) - YCG)*(Y(RGYSLCT(I)) - YCG)
         RGY = RGY + (Z(RGYSLCT(I)) - ZCG)*(Z(RGYSLCT(I)) - ZCG)
      ENDDO

      IF(RGY .GT. PT0001) Then

         RGY = DSQRT(RGY/DEN)
!         Write(6,*)' RMSD = ', Rgy
!
!    COMPUTE ENERGY AND FORCES FOR THIS CONSTRAINT
!
         ECRGY = HALF * FORCERG * ( RGY - RGY0 ) * ( RGY - RGY0 )

!CC   MH01:
!CC   TEST FIRSt disagrees with the below, so I commented it out
!CC   (same formula=same deriv!)
!CC         If(Qrmsd) Then
            Con = ForceRG * ( One - Rgy0 / Rgy ) / Den
!CC         Else
!CC            CON = FORCE * ( ONE - RGY0 / RGY )
!CC     &                  * ( ONE - ONE/DEN ) / DEN
!CC         Endif
!
         DO I = 1,NRGY
            If (Qrmsd) Then
               Xcg = RmsdRef(1,I)
               Ycg = RmsdRef(2,I)
               Zcg = RmsdRef(3,I)
            Endif
            DX(RGYSLCT(I)) = DX(RGYSLCT(I)) &
                           + CON * ( X(RGYSLCT(I)) - XCG )
            DY(RGYSLCT(I)) = DY(RGYSLCT(I)) &
                           + CON * ( Y(RGYSLCT(I)) - YCG )
            DZ(RGYSLCT(I)) = DZ(RGYSLCT(I)) &
                           + CON * ( Z(RGYSLCT(I)) - ZCG )
         ENDDO
      Endif
!      WRITE(DUNIT,'(7(1X,F15.8),1X,I5)')
!     &           RGY, ECRGY, FORCE, RGY0, XCG, YCG, ZCG, NCOUNT
!
!   NOW WRITE OUT FILE IF APPROPRIATE
!
      IF(QSAVRG) THEN
         IF ((IREST == 1).AND.(NCOUNTRG == 0)) THEN
            NCOUNTRG = DSTEPRG - 1  ! for restart dynamics
            WRITE(OUTU,*) 'ECRGY: NCOUNT updated with restart'
         ENDIF                  ! restart
         IF (NCOUNTRG < DSTEPRG) THEN
            NCOUNTRG = NCOUNTRG + 1
            IF (MOD(NCOUNTRG,NSVRGY) == 0 .AND. NCOUNTRG /= 0) THEN
               WRITE(DUNITRG, '(7(1X,F15.8),1X,I5)') &
                     RGY, ECRGY, FORCERG, RGY0, XCG, YCG, ZCG, NCOUNTRG
            ENDIF
         ENDIF                  ! increment ncount and write outfile
      ENDIF                     !IF(QSAVRG)

      RETURN
   END SUBROUTINE ERGY
#endif 
end module rgym

