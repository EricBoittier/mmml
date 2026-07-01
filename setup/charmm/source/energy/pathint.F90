module mpathint
  use chm_kinds
  use dimens_fcm

  implicit none
  character(len=*),private,parameter :: file_name   ="pathint.src"
#if KEY_PATHINT==1 /*pathint_main*/
   INTEGER,PARAMETER :: NMAXAT = 100
   INTEGER,allocatable,dimension(:) :: IPI, IPF
   integer ::NBEADS, NPIAT
   integer,allocatable,dimension(:) :: NATIDX
   logical,allocatable,dimension(:) :: NFIXED
   real(chm_real) PITEMP
   real(chm_real),allocatable,dimension(:) :: FMASS
   LOGICAL QPINT

#if KEY_MC==1
!  ARD 00-11-28
!  Added these for faster Monte Carlo calculations
   integer,allocatable,dimension(:) :: IBPIMC, JBPIMC
#endif 

contains

  subroutine allocate_pathint(natom)
    use memory
    integer :: natom
    character(len=*),parameter :: routine_name="allocate_pathint"
    call chmalloc(file_name,routine_name,'ipi ',natom,intg=ipi)
    call chmalloc(file_name,routine_name,'ipf ',natom,intg=ipf)
    call chmalloc(file_name,routine_name,'fmass ',natom,crl=fmass)

    return
  end subroutine allocate_pathint


   SUBROUTINE PINT_INIT(COMLYN,COMLEN)
!-----------------------------------------------------------------------
! THIS ROUTINE DEFINES SPRINGS FOR PATH INTEGRAL CALCULATIONS
!
  use exfunc
  use number
  use psf
  use coord
  use memory
  use select
  use string

      implicit none

      INTEGER COMLEN
      CHARACTER(len=*) COMLYN

      INTEGER I
!
!mh..09-FEB-98 in pathint.f90      DATA QPINT /.FALSE./
      QPINT = .FALSE.

      NPIAT = 0
      ! It appears we don't need this until here, and only need natom
      ! elments. cb3
      if(.not.allocated(ipi)) call allocate_pathint(natom)

      CALL SELCTA(COMLYN,COMLEN,IPI,X,Y,Z,WMAIN,.TRUE.)
      CALL SELCTA(COMLYN,COMLEN,IPF,X,Y,Z,WMAIN,.TRUE.)

      DO I = 1,NATOM
         IF (IPI(I) .EQ. 1) NPIAT = NPIAT + 1
      ENDDO

      call chmalloc('pathint.src','PINT','NATIDX',NPIAT,intg=NATIDX)
      call chmalloc('pathint.src','PINT','NFIXED',NPIAT,log=NFIXED)

      CALL PINTHP()

      PITEMP=GTRMF(COMLYN,COMLEN,'TEMP',THRHUN)
      NBEADS=GTRMI(COMLYN,COMLEN,'BEADS',16)

      QPINT = .TRUE.

!     ARD 00-11-28
!     Setup additional arrays for faster MC calculations
#if KEY_MC==1 /*pimc1*/
      IF (INDXA(COMLYN, COMLEN, 'MC') .GT. 0) THEN
!     Number of springs is NBEADS*NPIAT
        call chmalloc('pathint.src','PINT','IBPIMC',NBEADS*NPIAT,intg=IBPIMC)
        call chmalloc('pathint.src','PINT','JBPIMC',NBEADS*NPIAT,intg=JBPIMC)
        CALL STPIMC()
      ENDIF
#endif /* (pimc1)*/

!     Copy mass array for the force constant
      FMASS(1:NATOM) = AMASS(1:NATOM)

   END SUBROUTINE PINT_INIT

   SUBROUTINE PINTHP()
!-----------------------------------------------------------------------
!
  use psf
      implicit none

      INTEGER I, FIRST, LAST

      FIRST = -1
      LAST = -1
      NPIAT = 0
      DO I = 1,NATOM
         IF (IPI(I) .EQ. 1) THEN
            NPIAT = NPIAT + 1
            NATIDX(NPIAT) = I
            NFIXED(NPIAT) = IPF(I) .EQ. 1
            IF (FIRST .GE. 0 .AND. LAST .GE. 0) &
                 CALL WRNDIE(-1,'<PINT>', &
                 'Path integral atoms not in one block')
            IF (FIRST .LT. 0) FIRST = I
         ELSE
            IF (FIRST .GE. 0 .AND. LAST .LT. 0) &
                 LAST = I-1
         ENDIF
      ENDDO
      IF (LAST .LT. 0) LAST = NATOM

   END SUBROUTINE PINTHP

   SUBROUTINE EPINT(X,Y,Z,DX,DY,DZ)
!-----------------------------------------------------------------------
! THIS ROUTINE CALCULATES SPRING POTENTIALS FOR PATH INTEGRALS
!
  use psf
  use consta
  use exfunc
  use energym
      implicit none

      real(chm_real) X(*), Y(*), Z(*), DX(*), DY(*), DZ(*)

      INTEGER I,J,IDX1,IDX2
      real(chm_real) LAMBDA,KBT,FCONST
      real(chm_real) XD,YD,ZD,R,R2
      real(chm_real) SX,SY,SZ

      IF (NPIAT .EQ. 0 .OR. NBEADS .EQ. 1) RETURN
!
! ARD 00-11-28
! Moved these constants up here and removed the gratuitous SQRT
      LAMBDA = ANGSTROM*JKBOLTZ*PITEMP/HBAR
      KBT    = 0.5*NBEADS*AMU*LAMBDA*LAMBDA/KCALMOL

      DO I = 1,NPIAT
! This part was primarily included to treat some
! atoms of the segment classically since replica does not accept a subselection
! of an entire segment (i.e. pint applies to the whole segment).
! It is not compatible with the new version of verlet and then is commented.
!         IF (FIXED(I)) THEN
!            SX = 0.
!            SY = 0.
!            SZ = 0.
!            IDX1 = ATIDX(I)
!            DO J = 1,NBEADS
!               SX = SX + X(IDX1)
!               SY = SY + Y(IDX1)
!               SZ = SZ + Z(IDX1)
!               IDX1 = IDX1 + NPIAT
!            ENDDO
!            SX = SX/NBEADS
!            SY = SY/NBEADS
!            SZ = SZ/NBEADS
!            IDX1 = ATIDX(I)
!            DO J = 1,NBEADS
!               X(IDX1) = SX
!               Y(IDX1) = SY
!               Z(IDX1) = SZ
!               IDX1 = IDX1 + NPIAT
!            ENDDO
!c$$$            IDX1 = ATIDX(I)
!c$$$            IDX2 = IDX1 + NPIAT
!c$$$            DO J = 2,NBEADS
!c$$$               X(IDX2) = X(IDX1)
!c$$$               Y(IDX2) = Y(IDX1)
!c$$$               Z(IDX2) = Z(IDX1)
!c$$$               IDX2 = IDX2 + NPIAT
!c$$$            ENDDO
!         ELSE
            IDX1 = NATIDX(I)
            FCONST = KBT*FMASS(IDX1)
            DO J = 1,NBEADS
               IF (J .EQ. NBEADS) THEN
                  IDX2 = NATIDX(I)
               ELSE
                  IDX2 = IDX1 + NPIAT
               ENDIF
               XD = X(IDX1)-X(IDX2)
               YD = Y(IDX1)-Y(IDX2)
               ZD = Z(IDX1)-Z(IDX2)
               R2 = XD*XD + YD*YD + ZD*ZD
               ETERM(PINT) = ETERM(PINT) + FCONST*R2
               DX(IDX1) = DX(IDX1) + 2.*FCONST*XD
               DY(IDX1) = DY(IDX1) + 2.*FCONST*YD
               DZ(IDX1) = DZ(IDX1) + 2.*FCONST*ZD
               DX(IDX2) = DX(IDX2) - 2.*FCONST*XD
               DY(IDX2) = DY(IDX2) - 2.*FCONST*YD
               DZ(IDX2) = DZ(IDX2) - 2.*FCONST*ZD
               IDX1 = IDX2
            ENDDO
!         ENDIF
      ENDDO

      RETURN
   END SUBROUTINE EPINT

#if KEY_MC==1 /*pimc2*/
   SUBROUTINE STPIMC()
!
!       Setup arrays of the same form as the IB and JB bond arrays to
!       keep track of the springs for faster Monte Carlo calculations.
!
!       Aaron R. Dinner 00-11-28
!
      implicit none
!     Local  variables
      INTEGER I, J, IDX1, IDX2, NB

      NB = 0
      DO I = 1, NPIAT
         IDX1 = NATIDX(I)
         DO J = 1, NBEADS
            IF (J .EQ. NBEADS) THEN
               IDX2 = NATIDX(I)
            ELSE
               IDX2 = IDX1 + NPIAT
            ENDIF
            NB = NB + 1
            IBPIMC(NB) = IDX1
            JBPIMC(NB) = IDX2
            IDX1 = IDX2
         ENDDO
      ENDDO

      RETURN
   END SUBROUTINE STPIMC

   SUBROUTINE EPIMC(EP,AMASS,NB,IBLST,X,Y,Z)
!
!       Path integral energy for Monte Carlo
!
!       Aaron R. Dinner 00-11-28
!
  use consta
  use number
      implicit none
!     Passed variables
      INTEGER NB, IBLST(*)
      real(chm_real)  EP, X(*), Y(*), Z(*), AMASS(*)
!     Local  variables
      INTEGER I, IDX1, IDX2
      real(chm_real)  XD, YD, ZD, R2, KBT, KT, H2, A2, FCONST

      EP = ZERO

      H2  = HBAR*HBAR
      A2  = ANGSTROM*ANGSTROM
      KT  = JKBOLTZ*PITEMP
      KBT = 0.5*NBEADS*AMU*A2*KT*KT/(KCALMOL*H2)
      DO I = 1, NB
         IDX1 = IBPIMC(IBLST(I))
         IDX2 = JBPIMC(IBLST(I))

         FCONST = KBT*AMASS(IDX1)

         XD = X(IDX1)-X(IDX2)
         YD = Y(IDX1)-Y(IDX2)
         ZD = Z(IDX1)-Z(IDX2)
         R2 = XD*XD + YD*YD + ZD*ZD
         EP = EP + FCONST*R2
      ENDDO

      RETURN
   END SUBROUTINE EPIMC
#endif /* (pimc2)*/

#endif /* (pathint_main)*/

end module mpathint

