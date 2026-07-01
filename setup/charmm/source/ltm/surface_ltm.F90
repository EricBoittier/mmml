module surface
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="surface_ltm.src"
!CHARMM Element source/fcm/surface.fcm 1.1
! File: surface.fcm
!
!           Common block for solvent free energy.
!
#if KEY_ASPENER==1 /*surface_fcm*/
!
!  radius of a water molecule
      real(chm_real) PROBE_RADIUS
!
!  atomic solvation parameter, indexed on atom 
      real(chm_real),allocatable,dimension(:) :: ASPV
!
!  indexed on atom, tells if atom is a hydrogen atom or lone pair.
!  These will be ignore completely by usere.
      integer,allocatable,dimension(:) :: IGNORE
!
!  flag which tells usere whether to read the solvent info.
      logical SOLVENT_ENTERED
!
!  vdw radius used for the solvent energy.  Includes probe radius.
!  Indexed on atom.
      real(chm_real),allocatable,dimension(:) :: VDW_SURF
!
!  reference solvent free energy, for the whole system
      real(chm_real) REFER_SOLVENT_ENER
!
!
contains
  subroutine surface_iniall()
#if KEY_ASPENER==1
    solvent_entered=.false.   
#endif
    return
  end subroutine surface_iniall

  subroutine allocate_surface(natom)
    use memory
    integer natom
    character(len=*),parameter :: routine_name="allocate_surface"
    if(allocated(aspv).and.size(aspv)>natom) then
       return
    elseif(allocated(aspv)) then
       call chmdealloc(file_name,routine_name,'aspv  ',size(aspv),crl=aspv)
       call chmdealloc(file_name,routine_name,'ignore',size(ignore),intg=ignore)
       call chmdealloc(file_name,routine_name,'vdw_surf', &
            size(vdw_surf),crl=vdw_surf)
    else
       call chmalloc(file_name,routine_name,'aspv  ',natom,crl=aspv)
       call chmalloc(file_name,routine_name,'ignore',natom,intg=ignore)
       call chmalloc(file_name,routine_name,'vdw_surf',natom,crl=vdw_surf)
    endif
    return
  end subroutine allocate_surface
#endif /* (surface_fcm)*/

end module surface

