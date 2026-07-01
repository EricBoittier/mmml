module afm_module

  use chm_kinds
  implicit none

  !     Emanuele Paci 20-Jun-2003
  !     This common file contains hqbm defaults which are read
  !     from the parameter file. They are processed in the routines EMBIAS
  !   
  !-----------------------------------------------------------------------
  !     Flags or values not present in this file may not be specified as
  !     defaults when reading a card parameter file is read.  See PARRDR
  !     for more information.  All variables here MUST also be parsed
  !     in PARRDR.
  ! 
  !     MAXSEL must be equal or larger to the number of degrees of freedom
  !     used in the definition of the reaction coordinate in HQBM
  INTEGER, parameter :: MAXSEL=2

  !     Converts forces from (kcal/mol)/A to pN
  real(chm_real),parameter :: fconv=69.478508

  INTEGER,save :: ISEL(maxsel)
  INTEGER,save :: NSEL,IUNJUJ,ISTPAFM
  real(chm_real),save ::  DFALPHA,XIMAX,BETA
  LOGICAL,save :: AFMCF,AFMSMD,AFMBMD,lafm

  !
  !===============================================================
contains
  !---------------------------------------------------------------

#if KEY_AFM==1 /*afm_outer*/
  subroutine afm_init()
#if KEY_AFM==1
    lafm = .FALSE.         
#endif
    return
  end subroutine afm_init

  SUBROUTINE AFMINI
    !-----------------------------------------------------------------------
    !     Author: Emanuele Paci <24-june-02>
    !     This routine is invoked at the beginning of each CHARMM run.
    !     if the option AFMINI is called
    !
    use memory
    use dimens_fcm
    use energym
    use exfunc
    use comand
    use coord
    use number
    use psf
    use select
    use stream
    use string

    real(chm_real) A
    INTEGER I
    integer,allocatable,dimension(:) :: emslct

    LAFM = .TRUE.

    ISTPAFM = 0

    IF (INDXA(COMLYN, COMLEN, 'RESE') .GT. 0) THEN
       WRITE(OUTU,*)' AFM Module turned off'
       AFMCF =  .false.
       AFMBMD = .false.
       AFMSMD = .false.
       LAFM = .false.
       RETURN
    ENDIF

    IUNJUJ=GTRMI(COMLYN,COMLEN,'IUNJ',96)

    !     Different reaction coordinates are available
    IF (INDXA(COMLYN, COMLEN, 'CF' ) .GT. 0) AFMCF =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'BMD') .GT. 0) AFMBMD =.TRUE.
    IF (INDXA(COMLYN, COMLEN, 'SMD') .GT. 0) AFMSMD =.TRUE.


    A=GTRMF(COMLYN,COMLEN,'ALPHA',FMARK)
    IF(A.GE.0.0) THEN
       DFALPHA=A
    ELSE
       DFALPHA=0.0D0
    ENDIF

    !     Convert the Bias from pN to charmm units: (kcal/mol)/A
    DFALPHA=DFALPHA/FCONV

    !         to select atoms.
    call chmalloc("afm.src","afmini","emslct",natom,intg=emslct)
    CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)

    NSEL = 0
    DO I=1,natom
       IF (EMSLCT(I) .EQ. 1) THEN
          NSEL = NSEL + 1
          ISEL(NSEL) = I
       END IF
    END DO
    call chmdealloc("afm.src","afmini","emslct",natom,intg=emslct)

    !        Check that only a pair of atoms is selected

    IF (NSEL .GT. 2) THEN
       WRITE(OUTU,'(2(A,I5))')' AFMINI> NSEL',NSEL &
            ,' IS LARGER THAN 2!!!!'
       CALL WRNDIE(-1,'<AFMINI>', &
            'YOU CANNOT PULL MORE THAN 2 ATOMS')
    END IF
    
    ! BIOVIA Code Start : Bug fix
    WRITE(OUTU,*, ERR=10)' pulling method CF/BMD/SMD ?',AFMCF,AFMBMD,AFMSMD
10    IF ((AFMCF .AND. AFMBMD) .OR. (AFMBMD .AND. AFMSMD) .OR. (AFMCF &
         .AND. AFMSMD)) THEN 
       CALL WRNDIE(-1,'<AFMINI>' &
            ,'Please define one pulling method only')
    END IF
    ! BIOVIA Code End
    
    IF (AFMSMD) THEN 
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A)')
       !        BETA is the SMD pulling speed
       A=GTRMF(COMLYN,COMLEN,'BETA',FMARK)
       IF(A.GE.0.0) THEN
          BETA=A
       ELSE
          BETA=0.0D0
       ENDIF

    END IF

    A=GTRMF(COMLYN,COMLEN,'XIMAX',FMARK)
    IF(A.GE.0.0) THEN
       XIMAX=A
    ELSE
       XIMAX=999.9D0
    ENDIF

    IF (.NOT.(AFMCF) .AND. .NOT.(AFMBMD) .AND. .NOT.(AFMSMD)) THEN
       CALL WRNDIE(-1,'<AFMINI>' &
            ,'You have to define a pulling method (CF/SMD/BMD)')
    END IF

    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I5)')'AFMINI> nsel',nsel
    RETURN
  END SUBROUTINE AFMINI

  SUBROUTINE AFM(EU,X,Y,Z,DX,DY,DZ,NATOMX)
    !-----------------------------------------------------------------------
    !     EU   - User energy to be returned
    !     X,Y,Z - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOMX - Number of atoms
    !
    !     Author: Emanuele Paci <07-02-97>
    !
    use chm_kinds
    use dimens_fcm
    use coordc
    use number
    use contrl
    !  use afm
    use stream
    use psf
    use reawri
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none
    real(chm_real) EU
    INTEGER NATOMX,SIZEIJ
    real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
    real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)

    !     MAXSEL is defined in afm.f90
    !
    PARAMETER (SIZEIJ = MAXSEL*(MAXSEL-1)/2)
    !
    INTEGER I,J,L,L1,L2,K,II,JJ,LL,NSEL2,IL(SIZEIJ),JL(SIZEIJ)
    real(chm_real) XIJ,YIJ,ZIJ,R     ! ,FCONV
    real(chm_real) RHO,RHO0,XI,XIA,DXI,ALPHA,ALPHA2,AUX,FORCE
    SAVE XI,XIA,RHO0,IL,JL,NSEL2
    !
    !
    !     Converts forces from (kcal/mol)/A to pN
    !      FCONV=69.478508
    !
    ALPHA=DFALPHA
    ALPHA2=ALPHA/TWO
    EU=0.0

    !     The first time AFM is called, compute the distance RIJ at t=0
    !
    !     Do the first time that this routine is called

    IF (ISTPAFM .EQ. 0) THEN
       IF (PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A)')' AFM> CALLING AFM BIAS'
          WRITE(OUTU,'(A,F19.8)')' AFM> ALPHA=',ALPHA
          WRITE(OUTU,'(A)') &
               ' AFM> THE EXTERNAL PERTURBATION PULLS TWO ATOMS APART'
       END IF
       !
       IF (PRNLEV.GE.6) WRITE(OUTU,'(A)') &
            ' AFM> COORDINATES OF THE INITIAL CONFIG.:'
       DO II=1,NSEL
          I = ISEL(II) 
          IF (PRNLEV.GE.6) WRITE(OUTU,'(A9,I5,3F12.4)') 'AFM> ' &
               ,I,X(I),Y(I),Z(I)
       END DO

       L=1
       II=1
       JJ=2
       IL(L)=ISEL(II)
       JL(L)=ISEL(JJ)

       NSEL2=1

       I=IL(L)
       J=JL(L)
       XIJ=X(I)-X(J)
       YIJ=Y(I)-Y(J)
       ZIJ=Z(I)-Z(J)
       RHO = SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
       RHO0 = RHO

       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)') &
            ' AFM> INITIAL DISTANCE = ',RHO0
       IF (PRNLEV.GE.2) WRITE(OUTU,'(A,I7)') &
            ' AFM> NUMBER OF DISTANCES = ',NSEL2

       XI = RHO0
       XIA = XI

       !     Do everytime that this routine is called except the first
    ELSE  

       L=1
       I=IL(L)
       J=JL(L)            
       XIJ=X(I)-X(J)
       YIJ=Y(I)-Y(J)
       ZIJ=Z(I)-Z(J)
       RHO = SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)

       XI=RHO

    END IF

    !     Computes energy and forces associated to the external perturbation
    !     for each reaction coordinate
    !     react_coordinate = XI
    !     energy = EU 
    !     -force = DX = d(EU)/d(XI) x d(XI)/d(r_i) x d(r_i)/d(X(I))
    !                 =   AUX       x ....         x ...

    IF (ALPHA .NE. 0.0) THEN

       IF (AFMBMD) THEN

          IF (XI .LT. XIA) THEN
             DXI=XI-XIA
             EU=ALPHA2*DXI**2
             AUX=ALPHA*DXI/XI
             FORCE=-ALPHA*DXI*FCONV

             !        Compute forces
             I=IL(L)
             J=JL(L)
             XIJ=X(I)-X(J)
             YIJ=Y(I)-Y(J)
             ZIJ=Z(I)-Z(J)
             DX(I)=DX(I)+AUX*XIJ
             DY(I)=DY(I)+AUX*YIJ
             DZ(I)=DZ(I)+AUX*ZIJ
             DX(J)=DX(J)-AUX*XIJ
             DY(J)=DY(J)-AUX*YIJ
             DZ(J)=DZ(J)-AUX*ZIJ
          ELSE
             EU=ZERO
             FORCE=ZERO
             XIA=XI
          END IF

       ELSE IF (AFMSMD) THEN

          DXI=XI-XIA
          EU=ALPHA2*DXI**2
          AUX=ALPHA*DXI/XI
          FORCE=-ALPHA*DXI*FCONV

          !        Compute forces
          I=IL(L)
          J=JL(L)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          DX(I)=DX(I)+AUX*XIJ
          DY(I)=DY(I)+AUX*YIJ
          DZ(I)=DZ(I)+AUX*ZIJ
          DX(J)=DX(J)-AUX*XIJ
          DY(J)=DY(J)-AUX*YIJ
          DZ(J)=DZ(J)-AUX*ZIJ

          XIA=XIA+BETA*TIMEST

       ELSE IF (AFMCF) THEN

          EU=-ALPHA*XI
          AUX=-ALPHA/XI
          FORCE=ALPHA*FCONV

          !        Compute forces
          I=IL(L)
          J=JL(L)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          DX(I)=DX(I)+AUX*XIJ
          DY(I)=DY(I)+AUX*YIJ
          DZ(I)=DZ(I)+AUX*ZIJ
          DX(J)=DX(J)-AUX*XIJ
          DY(J)=DY(J)-AUX*YIJ
          DZ(J)=DZ(J)-AUX*ZIJ

       END IF

    END IF

    !     Prints length in A, force in pN, energy in kcal/mol
    IF(MOD(ISTPAFM,NPRINT).EQ.0)  &
         WRITE(IUNJUJ,'(I9,4(1X,F14.6))')ISTPAFM &
         ,XI,XIA,FORCE,EU
    ISTPAFM=ISTPAFM+1

    IF (XI .GT. XIMAX) CALL WRNDIE(-1,'<AFM>', &
         'Reaction coordinate exceeded maximum value.')

123 CONTINUE
    RETURN
  end SUBROUTINE AFM

#endif /*       (afm_outer)  */

  SUBROUTINE AFMDUMMY

    RETURN
  END subroutine afmdummy
end module afm_module

