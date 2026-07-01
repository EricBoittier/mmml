module usermod
  logical :: printe_flag = .false., printe_off = .false.
  integer :: iuene

contains

SUBROUTINE USERSB
  !
  !     THIS DOES NOTHING SAVE TELL USER THAT HE CALLED A NULL PROGRAM.
  !
  !     Author: Robert Bruccoleri
  !
  use chm_kinds
  use stream
  use comand
  use string
  implicit none
  !
  CALL WRNDIE(1,'<USERSB>','USERSB CALLED. NO ACTION TAKEN.')
  RETURN
END SUBROUTINE USERSB

SUBROUTINE USERE(EU,X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOMX)
  !
  !     THE USER ENERGY ROUTINE. DOES NOTHING IN THE NORMAL SYSTEM
  !     EXCEPT SET EU TO ZERO. QECONT IS A FLAG WHICH WILL BE SET TO 1
  !     WHEN THE ROUTINE IS CALLED IN THE QECONTIS SECTION. INDIVIDUAL
  !     ATOM USER ENERGIES ARE THEN RETURNED IN ECONT, AND THE DERIVATIVE
  !     ARRAYS SHOULD NOT BE ACCESSED.
  !
  !     EU   - User energy to be returned
  !     X,Y,Z - Coordinates for energy evaluation
  !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
  !     QECONT - Flag for analysis (0-no analysis,>0=analysis)
  !     ECONT(NATOMX) - Analysis array to be filled if QECONT>0.
  !     NATOMX - Number of atoms
  !
  !     Author: Robert Bruccoleri
  !
  use chm_kinds
  implicit none
  real(chm_real) EU
  INTEGER NATOMX
  real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
  real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
  LOGICAL QECONT
  real(chm_real) ECONT(NATOMX)
  !
  INTEGER I
  !
  EU=0.0
  IF(QECONT) THEN
     DO I=1,NATOMX
        ECONT(I)=0.0
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE USERE

SUBROUTINE USRACM(NATOMX,X,Y,Z,DX,DY,DZ,QECONT,ECONT,ICALL)
  !
  !     THE USER ACCUMULATION ROUTINE. DOES NOTHING IN THE NORMAL SYSTEM.
  !     This routine is called every time the energy is evaluated.
  !     The ICALL integer tells whether this is a "real" step for
  !     minimization or dynamics.
  !
  !     X,Y,Z    - Coordinates for energy evaluation
  !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
  !     ICALL    - ECALLS increment 
  !
  !     Author: B. Brooks
  !
  use chm_kinds
  implicit none
  INTEGER NATOMX
  real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
  real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  INTEGER ICALL
  !
  RETURN
END SUBROUTINE USRACM

SUBROUTINE USERNM(IUSER,X,Y,Z,XNORMA,YNORMA,ZNORMA,NATOMX)
  !
  !     THIS ROUTINE IS USED TO SETUP A GUESS NORMAL MODE, OR A MOTION
  !     OF INTEREST TO BE ANALYSED, OR APPENDED TO THE COORDINATE SET.
  !     THE CALLING SEQUENCE IS 3 ARRAYS WHICH ARE TO BE FILLED
  !     WITH COORDINATE DISPLACEMENTS. (I.E. DONT MASS WEIGHT THEM)
  !     SEE THE DOCUMENTATION OF VIBRAN FOR FURTHER DETAILS
  !
  !     IUSER  - Code for which function to use. This is only important
  !               if more than one user mode is specified.
  !     X,Y,Z   - Reference coordinates
  !     XNORMA,YNORMA,ZNORMA - Mode vector to be filled
  !     NATOMX  - Number of atoms
  !
  !     Author: Bernie Brooks
  !
  use chm_kinds
  implicit none
  INTEGER IUSER
  INTEGER NATOMX
  real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
  real(chm_real) XNORMA(NATOMX),YNORMA(NATOMX),ZNORMA(NATOMX)
  !
  INTEGER I
  !
  !     The default user mode is the radius of gyration motion about
  !     the origin.
  !
  DO I=1,NATOMX
     XNORMA(I)=X(I)
     YNORMA(I)=Y(I)
     ZNORMA(I)=Z(I)
  ENDDO
  !
  CALL WRNDIE(1,'<USERNM>', &
       'NO ROUTINE PROVIDED, RGYR MODE RETURNED')
  RETURN
END SUBROUTINE USERNM

SUBROUTINE USRSEL(NTAGS,FLAGS,X,Y,Z,QCOOR)
  !
  !     THIS ROUTINE ALLOWS A USER TO SELECT ATOMS.
  !
  !     NTAGS - number of entries (atoms)
  !     FLAGS(NTAGS) - array to hold selection values
  !     X,Y,Z  - coordinates
  !     QCOOR  - logical flag specifying that coordinates are present
  !              If .FALSE., then the coordinates should not be accessed.
  !
  !     If an atom is selected, set FLAGS value to be nonzero.
  !
  !
  use chm_kinds
  implicit none
  INTEGER NTAGS
  INTEGER FLAGS(NTAGS)
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL QCOOR
  !
  INTEGER I
  !
  CALL WRNDIE(1,'<USRSEL>','NO USER SELECTION SPECIFIED')
  !
  !     THE DEFAULT USER SELECTION IS TO INCLUDE EVERYTHING
  DO I=1,NTAGS
     FLAGS(I)=1
  ENDDO
  RETURN
END SUBROUTINE USRSEL


SUBROUTINE USRTIM(SERVAL,QAT,NQ,ITIME, &
     NATOMX,X,Y,Z,XREF,YREF,ZREF,NSKIP,DELTA,TVAL,ISLCT)
  !
  !     This routine allows the user to specify a time series
  !     for use in CORREL.
  !
  !     SERVAL  - Value for time series specified in the ENTER command.
  !                May be used for distinguishing time series, or for any
  !                other purpose.
  !     QAT(NQ)- List of atoms that were specified in the ENTER command
  !     NQ     - Number of atoms specified in the ENTER command
  !     ITIME  - Which time step is currenlty being processed
  !     NATOMX - Number of atoms
  !     X,Y,Z  - Coordinates for current time step
  !     XREF,YREF,ZREF - Reference coordinates for time series
  !     DELTA  - Time interval between steps in picoseconds.
  !     TVAL   - Time series value returned
  !     ISLCT  - The first atom selection array from the TRAJ command
  !
  use chm_kinds
  implicit none
  !
  real(chm_real) SERVAL
  INTEGER ITIME,NATOMX,NQ
  INTEGER QAT(NQ)
  real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
  real(chm_real) XREF(NATOMX),YREF(NATOMX),ZREF(NATOMX)
  INTEGER NSKIP
  real(chm_real) DELTA,TVAL
  INTEGER ISLCT(*)
  !
  CALL WRNDIE(1,'<USECOR>','NO USER TIME SERIES SPECIFIED')
  !
  !     THE DEFAULT USER TIME SERIES IS SIMPLY THE TIME
  TVAL=ITIME/DELTA
  RETURN
END SUBROUTINE USRTIM

SUBROUTINE USRINI
  !
  !     This routine allows the user to specify a startup procedure.
  !     This routine is invoked at the beginning of each CHARMM run.
  !
  use chm_kinds
  implicit none
  !
  RETURN
END SUBROUTINE USRINI

SUBROUTINE USPAFL(I,XX,YY,ZZ,XY,XZ,YZ, &
     XNORM,YNORM,ZNORM,NATOM,ISLCT)
  !
  !  This routine processes the user specific principal axis
  !  fluctuation code.  See documentation for details.
  !
  use chm_kinds
  implicit none
  INTEGER I,NATOM
  real(chm_real) XX(*),YY(*),ZZ(*),XY(*),XZ(*),YZ(*)
  real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
  INTEGER ISLCT(*)
  !
  RETURN
END SUBROUTINE USPAFL

SUBROUTINE USER_RNGSPEC
  ! specification for USER random number generator

  IMPLICIT NONE

  RETURN
END SUBROUTINE USER_RNGSPEC


end module usermod

