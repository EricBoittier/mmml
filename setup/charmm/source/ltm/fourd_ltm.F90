module fourdm
  use chm_kinds
  use dimens_fcm
  use energym,only:lenent
  character(len=*),private,parameter :: file_name   ="fourd_ltm.src"
!CHARMM Element source/fcm/fourd.fcm 1.1
!
#if KEY_FOURD==1 /*4dfcm*/
!     The fourth dimension coordinates, first derivative,
!     and fourth dimension flag.
!
!     DIM4.....Main flag indicating 4-d restraints in use.
!
!     FDIM.....The coordinate (in analogy to X,Y, & Z) of the 4th D.
!     DFDIM....The forces (in analogy to DX,DY, & DZ) of the 4th D.
!     FDEQ.....The equilibrium value(s) that the 4th D function will use as
!                 the center of the harmonic.  Used for restraining the
!                 4th D to non zero values (i.e. forcing a system into
!                 the 4th D).
!     FDCOMP(N)..4th-d reference positions
!     K4DI.....Initial 4th D force constant.
!     INCD4....The point at which to increase K4D.   
!     DECD4....The point at which to stop increasing K4D.   
!     REST4....Flag to restart 4d dynamics
!     MULTK4...The factor by which K4DI will increase linearly from
!              INC4D to DEC4D.
!     CM4D.....Flag for applying mass weighted 4D forces.  Only set if
!                 one is using SHA4, but maybe it should be a 4D option 
!                 in the future.
!     DIM4ON(10)..
!     FCOUNT...
!
!     Purpose:
!     Holding the fourth D coordinates, 1st derivatives, velocities,etc. 
!
!     Written by Elan Eisenmesser 
!
!
      real(chm_real),allocatable,dimension(:) ::   FDIM,DFDIM,FDCOMP,FDEQ
!
      real(chm_real)   K4DI,MULTK4
!
      INTEGER  DIM4ON(LENENT),INC4D,DEC4D,FCOUNT,REST4,FDRGEO,FDDGEO
      INTEGER,allocatable,dimension(:) ::  IMOVE4
!
      LOGICAL  DIM4,CM4D,QSHAK4
!
!
contains

  subroutine fourd_init()
    use number,only:fifty,one
    implicit none
    k4di= fifty
    multk4 = one
    dim4=.false.
    cm4d=.false.
    qshak4=.false.
    return
  end subroutine fourd_init


  subroutine allocate_fourd_ltm()
    use memory
    implicit none
    character(len=*),parameter :: routine_name="allocate_fourd_ltm"
    call chmalloc(file_name,routine_name,'fdim  ',maxaim,crl=fdim)
    call chmalloc(file_name,routine_name,'dfdim ',maxaim,crl=dfdim)
    call chmalloc(file_name,routine_name,'fdcomp',maxaim,crl=fdcomp)
    call chmalloc(file_name,routine_name,'fdeq  ',maxaim,crl=fdeq)
    call chmalloc(file_name,routine_name,'imove4',maxaim,intg=imove4)
    imove4 = 0
    return
  end subroutine allocate_fourd_ltm

  subroutine deallocate_fourd_ltm()
    use memory
    implicit none
    character(len=*),parameter :: routine_name="deallocate_ford_ltm"

    call chmdealloc(file_name,routine_name,'fdim  ',maxaim,crl=fdim)
    call chmdealloc(file_name,routine_name,'dfdim ',maxaim,crl=dfdim)
    call chmdealloc(file_name,routine_name,'fdcomp',maxaim,crl=fdcomp)
    call chmdealloc(file_name,routine_name,'fdeq  ',maxaim,crl=fdeq)
    call chmdealloc(file_name,routine_name,'imove4',maxaim,intg=imove4)
  end subroutine deallocate_fourd_ltm

#endif /* (4dfcm)*/

end module fourdm

