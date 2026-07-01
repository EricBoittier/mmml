module code
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name="code_ltm.src"
!
!             The CODES list
!
!     Purpose:
!
!     Provides a pointer from each bond, angle, torsion, improper
!     torsion, into the parameters needed to calculate a
!     particular energy
!
!     I/O:    Never read or written
!
!     Variable   Purpose
!
!     ICB        Pointer for bond parameters
!     ICI        Pointer for improper torsion parameters
!     ICP        Pointer for torsion parameters
!     ICT        Pointer for bond angle parameters
#if KEY_CMAP==1
!     ICCT       Pointer for cross-term parameters
#endif 
!     The dimensions of each array must match the dimensions used
!     in the PSF.
!
!     MUSTUP     Logical flag that says that the PSF has been modified
!                since the last codes update, and the CODES arrays need
!                to be updated before the next energy call.
#if KEY_ACE==1
!     ACEUP      Logical flag that tells ACE that PSF has been modified
!                and that the nonbonded exclusion list contribution
!                to self energies needs to be recalculated (IESFIX).
#endif 
!
  integer,save,allocatable,dimension(:) :: ICB, ICT, ICP, ICI
#if KEY_CMAP==1
  integer,save,allocatable,dimension(:) :: ICCT 
#endif


  LOGICAL,save ::  MUSTUP 
#if KEY_ACE==1
  logical,save :: ACEUP  
#endif

contains
  subroutine code_init()
    mustup=.false.
    return
  end subroutine code_init

  subroutine allocate_code()
    use memory
    character(len=*),parameter :: routine_name="allocate_code"

    call chmalloc(file_name,routine_name,'icb ',maxb  ,intg=icb)
    call chmalloc(file_name,routine_name,'ict ',maxt  ,intg=ict)
    call chmalloc(file_name,routine_name,'icp ',maxp  ,intg=icp)
    call chmalloc(file_name,routine_name,'ici ',maximp,intg=ici)
#if KEY_CMAP==1
    call chmalloc(file_name,routine_name,'icct',maxcrt,intg=icct)    
#endif
    ici = 0

    return
  end subroutine allocate_code

end module code

