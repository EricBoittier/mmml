module coord
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name="coord_ltm.src"
!
!     The Coordinates
!
!     Purpose:
!     Holding the Cartesian coordinates of all the atoms in the system
!
!     I/O: COORIO.FLX
!
!     Variable  Purpose
!
!     X         X component
!     Y         Y component
!     Z         Z component
!     WMAIN     Weight or temp factor specification
!
  real(chm_real),save,allocatable,dimension(:) :: X,Y,Z,WMAIN
  
contains

  subroutine allocate_coord_ltm()
    use memory
    character(len=*),parameter :: routine_name="allocate_coord_ltm"
#ifdef KEY_RESIZE    
    integer:: i0
    i0=1
    call chmalloc(file_name,routine_name,'x ',2*i0,crl=x)
    call chmalloc(file_name,routine_name,'y ',2*i0,crl=y)
    call chmalloc(file_name,routine_name,'z ',2*i0,crl=z)
    call chmalloc(file_name,routine_name,'wmain ',2*i0,crl=wmain)
#else
    call chmalloc(file_name,routine_name,'x ',maxaim,crl=x)
    call chmalloc(file_name,routine_name,'y ',maxaim,crl=y)
    call chmalloc(file_name,routine_name,'z ',maxaim,crl=z)
    call chmalloc(file_name,routine_name,'wmain ',maxaim,crl=wmain)
#endif    
    return
  end subroutine allocate_coord_ltm

  SUBROUTINE TSTCRD(LOK,X,Y,Z,N)
    !
    !     THIS ROUTINE TESTS THE COORDINATE ARRAY TO INSURE THAT ALL
    !     COORDINATES ARE DEFINED AND WITHIN RANGE (-9990.0 TO 9990.0)
    !
    !      By Bernard R. Brooks    1983
    !
    use chm_kinds
    implicit none
    real(chm_real) X(*),Y(*),Z(*)
    LOGICAL LOK
    real(chm_real) :: RANGE=9990.0_chm_real
    INTEGER I,N
    !
    LOK=.TRUE.
    DO I=1,N
       LOK=(X(I).GT.-RANGE .AND. X(I).LT.RANGE) .AND. &
            (Y(I).GT.-RANGE .AND. Y(I).LT.RANGE) .AND. &
            (Z(I).GT.-RANGE .AND. Z(I).LT.RANGE)
       IF(.NOT.LOK) THEN
          CALL WRNDIE(-1,'<TSTCRD>', &
               'SOME ATOM COORDINATES UNDEFINED OR OUT OF RANGE')
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE TSTCRD

end module coord

