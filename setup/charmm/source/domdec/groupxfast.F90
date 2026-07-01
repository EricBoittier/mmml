module groupxfast

#if KEY_DOMDEC==1 /*domdec_main*/

  ! Module for definition of groups using DOMDEC non-bonded routines
  use chm_kinds
  use dimens_fcm
  implicit none
  private

  ! Maximum group size is 7 atoms
  integer, parameter :: max_group_size = 7

  ! Atom group division: Solutes are single atom groups, solvents are 3 or 4 (TIP4) atom groups
  integer ngroup
  integer, allocatable, dimension(:) :: group
  
  ! "Inverse" group list: for each atom gives the group index
  integer, allocatable, dimension(:) :: invgroup

  ! Group center and box size
  real(chm_real), allocatable, dimension(:,:) :: groupcenter, groupbox

  ! Group squared radius
  real(chm_real), allocatable, dimension(:) :: grouprad

  ! Group displacement (done in domdec_comm.src/recenter_coord())
  real(chm_real), allocatable, dimension(:,:) :: groupsh

  ! Group location in relation to homebox: 1 if in home box, 0 otherwise
  ! bit 0: X
  ! bit 1: Y
  ! bit 2: Z
  integer, allocatable, dimension(:) :: grouploc

  ! Maximum group radius
  real(chm_real) maxgrp_rad, maxwater_rad

  ! max(groupbox)
  real(chm_real) maxbox(3)

  ! Public subroutines
  public calc_groupbox, make_groups, group_in, group_out

  ! Public variables
  public ngroup, group, groupcenter, maxgrp_rad, invgroup, &
       maxwater_rad, groupbox, maxbox, groupsh, max_group_size, grouprad
  public grouploc

  interface group_out
     module procedure group_out_notype
     module procedure group_out_type
     module procedure group_out_onlytype
  end interface

contains

  ! *
  ! * Packs in group definitions:
  ! * Bits: 0-27  = atom position (max. 2^28-1 = 256M)
  ! * Bits: 28-30 = group size - 1 (max. size is 2^3 - 1 = 7)
  ! * Bits: 31-31 = group type: 0 = solute, 1 = solvent
  ! *
  integer function group_in(is, iq, type)
    implicit none
    integer, intent(in) :: is, iq, type

    group_in = ior(ior(ishft(iq-is,28),ishft(type,31)), is)

    return
  end function group_in

  ! *
  ! * Reverse of group_in, unpacks group, ignores group_type
  ! * mask: FFFFFFF = 2^28 - 1
  ! *
  subroutine group_out_notype(g, is, iq)
    implicit none
    integer, intent(in) :: g
    integer, intent(out) :: is, iq

    is = iand(g, Z'FFFFFFF')
    iq = is + iand(ishft(g, -28),7)

    return
  end subroutine group_out_notype

  ! *
  ! * Reverse of group_in, unpacks group
  ! * mask: FFFFFFF = 2^28 - 1
  ! *
  subroutine group_out_type(g, is, iq, type)
    implicit none
    integer, intent(in) :: g
    integer, intent(out) :: is, iq, type

    is = iand(g, Z'FFFFFFF')
    iq = is + iand(ishft(g, -28),7)
    type = iand(ishft(g, -31),1)

    return
  end subroutine group_out_type

  ! *
  ! * Reverse of group_in, unpacks group
  ! * mask: FFFFFFF = 2^28 - 1
  ! *
  subroutine group_out_onlytype(g, type)
    implicit none
    integer, intent(in) :: g
    integer, intent(out) :: type

    type = iand(ishft(g, -31),1)

    return
  end subroutine group_out_onlytype

  ! *
  ! * Calculates group i center and box size
  ! *
  subroutine calc_groupbox(i, x, y, z)
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: i
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    ! Variables
    integer is, iq
    real(chm_real) xmin, ymin, zmin, xmax, ymax, zmax

    call group_out(group(i), is, iq)

    xmin = minval(x(is:iq))
    ymin = minval(y(is:iq))
    zmin = minval(z(is:iq))
    xmax = maxval(x(is:iq))
    ymax = maxval(y(is:iq))
    zmax = maxval(z(is:iq))

    groupcenter(1,i) = half*(xmin + xmax)
    groupcenter(2,i) = half*(ymin + ymax)
    groupcenter(3,i) = half*(zmin + zmax)

    groupbox(1,i) = xmax - groupcenter(1,i)
    groupbox(2,i) = ymax - groupcenter(2,i)
    groupbox(3,i) = zmax - groupcenter(3,i)

    return
  end subroutine calc_groupbox

  ! *
  ! * Calculates the maximum radius of all groups
  ! *
  subroutine calc_group_maxrad(x, y, z, maxrad, maxrad_solvent)
    use number
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(out) :: maxrad, maxrad_solvent
    ! Variables
    integer i, type

    maxrad = zero
    maxrad_solvent = zero

    do i=1,ngroup
       call calc_groupbox(i, x, y, z)
       grouprad(i) = groupbox(1,i)**2 + groupbox(2,i)**2 + groupbox(3,i)**2
       maxrad = max(grouprad(i), maxrad)
       call group_out(group(i), type)
       if (type == 1) maxrad_solvent = max(grouprad(i),maxrad_solvent)
    enddo

    maxrad = sqrt(maxrad)
    maxrad_solvent = sqrt(maxrad_solvent)

    return
  end subroutine calc_group_maxrad

  ! *
  ! * Check and reorder psf such that hydrogens are after heavy atoms they are bonded to
  ! * NOTE: the reordering is done only within groups
  ! *
  subroutine reorder_psf()
    use memory
    use psf
    use chutil,only:atomid,hydrog
    implicit none
    ! Variables
    integer, allocatable, dimension(:,:) :: hydroglist
    integer, allocatable, dimension(:) :: heavy, reorder
    integer nheavy, nheavy_max, next_heavy, reorder_len
    character(len=5) sid, rid, ren, ac
    integer igrp, is, iq, i, t, h, j, k
    logical q_reorder

    ! Build list of all hydrogen bonds for each atom
    call chmalloc('groupxfast.src','reorder_psf','hydroglist',3,natom,intg=hydroglist)
    hydroglist(1:3,1:natom) = 0
    do i=1,nbond
       if (ib(i) <= 0 .or. jb(i) <= 0) cycle
       if (hydrog(ib(i))) then
          ! ib(i) is hydrogen
          t = jb(i)
          h = ib(i)
       elseif (hydrog(jb(i))) then
          ! jb(i) is hydrogen
          t = ib(i)
          h = jb(i)
       else
          ! Not a hydrogen bond, don't include
          cycle
       endif
       if (hydroglist(1,t) == 0) then
          hydroglist(1,t) = h
       elseif (hydroglist(2,t) == 0) then
          hydroglist(2,t) = h
       elseif (hydroglist(3,t) == 0) then
          hydroglist(3,t) = h
       else
          call wrndie(-5,'<groupxfast>','More than 3 hydrogen bonds not possible!')
       endif
    enddo
    
    ! Calculate the max number of heavy atoms and maximum group size
    nheavy_max = 0
    reorder_len = 0
    do igrp=1,ngrp
       is = igpbs(igrp) + 1
       iq = igpbs(igrp+1)
       reorder_len = max(reorder_len, iq-is+1)
       call atomid(is, sid, rid, ren, ac)
       if ((trim(ren) == 'TIP3' .and. iq-is+1 == 3) .or. &
            (trim(ren) == 'SPC'  .and. iq-is+1 == 3) .or. &
            (trim(ren) == 'TIP4' .and. iq-is+1 == 4)) then
          ! group is solvent
       else
          ! group is solute
          nheavy = 0
          do i=is,iq
             if (.not.hydrog(i)) nheavy = nheavy + 1
          enddo
          nheavy_max = max(nheavy_max, nheavy)
       endif
    enddo

    call chmalloc('groupxfast.src','reorder_psf','heavy',nheavy_max,intg=heavy)
    call chmalloc('groupxfast.src','reorder_psf','reorder',reorder_len,intg=reorder)

    do igrp=1,ngrp
       is = igpbs(igrp) + 1
       iq = igpbs(igrp+1)
       call atomid(is, sid, rid, ren, ac)
       if ((trim(ren) == 'TIP3' .and. iq-is+1 == 3) .or. &
            (trim(ren) == 'SPC'  .and. iq-is+1 == 3) .or. &
            (trim(ren) == 'TIP4' .and. iq-is+1 == 4)) then
          ! group is solvent
       else
          ! group is solute
          !
          ! Find all heavy atoms and put their indices into heavy(1:nheavy)
          ! Hydrogens (max 3) connected to each heavy atom are in:
          ! hydroglist(1:3, heavy(i)), where i=1:nheavy
          nheavy = 0
          do i=is,iq
             if (.not.hydrog(i)) then
                nheavy = nheavy + 1
                heavy(nheavy) = i
             endif
          enddo
          
          ! Go through heavy atoms and check that all hydrogens are in correct order
          q_reorder = .false.
          heavy_loop: do i=1,nheavy
             if (i < nheavy) then
                next_heavy = heavy(i+1) - 1
             else
                next_heavy = iq
             endif
             do j=1,3
                h = hydroglist(j,heavy(i))
                if (h == 0) exit        ! No more hydrogens, exit loop
                ! Hydrogen index h must be within [heavy(i)+1, next_heavy]
                if (h < is .or. h > iq) then
                   ! Hydrogen is outside group => Fatal error
                   call wrndie(-5,'<groupxfast>','Hydrogen is outside group')
!!$                elseif (h < heavy(i)+1) then
!!$                   ! hydrogen is before the heavy atom
!!$                   call wrndie(-5,'<groupxfast>','Hydrogen is before heavy atom')
!!$                   q_reorder = .true.
!!$                   exit heavy_loop
                elseif (h > next_heavy) then
                   ! hydrogen is too far after the heavy atom
                   call wrndie(-5,'<groupxfast>','Hydrogen is too far after heavy atom')
                   q_reorder = .true.
                   exit heavy_loop
                endif
             enddo
          enddo heavy_loop

!!$          if (q_reorder) then
!!$             ! Re-order atoms!
!!$             k = 0
!!$             do i=1,nheavy
!!$                ! Insert heavy atom to the reorder list:
!!$                k = k + 1
!!$                reorder(k) = heavy(i)
!!$                ! Insert hydrogens to the reorder list after the heavy atom:
!!$                do j=1,3
!!$                   h = hydroglist(j,heavy(i))
!!$                   if (h == 0) exit        ! No more hydrogens, exit loop
!!$                   k = k + 1
!!$                   reorder(k) = h
!!$                enddo
!!$             enddo
!!$             ! Old order: is:iq
!!$             ! New order: reorder(is:iq)
!!$             do i=is,iq
!!$                if (reorder(i) /= i) then
!!$                   ! Atom i has been moved
!!$                endif
!!$             enddo
!!$          endif

       endif
    enddo

    call chmdealloc('groupxfast.src','reorder_psf','heavy',nheavy_max,intg=heavy)
    call chmdealloc('groupxfast.src','reorder_psf','reorder',reorder_len,intg=reorder)
    call chmdealloc('groupxfast.src','reorder_psf','hydroglist',3,natom,intg=hydroglist)

    return
  end subroutine reorder_psf

  ! *
  ! * Find groups: solute = heavy atom + hydrogens, solvent = oxygen + hydrogens
  ! * NOTE: Assumes things are arranged such that the hydrogens follow the heavy atom in the list
  ! *
  ! * if q_set_group = .true., sets group(:)
  ! * if q_set_group = .false, just calculates ngroup
  ! *
  subroutine find_groups(q_set_group, nhydrog)
    use stream,only:outu, prnlev
    use psf,only:natom,ngrp,igpbs, qdrude
    use chutil,only:atomid,hydrog
    implicit none
    ! Input / Output
    logical, intent(in) :: q_set_group
    integer, intent(in) :: nhydrog(:)
    ! Variables
    character(len=5) sid, rid, ren, ac
    integer i, j, is, iq, igrp, i_heavy, max_num_light
    logical found_heavy

    if (q_set_group) then
       if (.not.allocated(group) .or. .not.allocated(invgroup)) then
          call wrndie(-5,'<groupxfast>','Group arrays not allocated')
       elseif (size(group) < ngroup+1 .or. size(invgroup) < natom) then
          call wrndie(-5,'<groupxfast>','Group arrays too small')
       endif
    endif

    ! Set the maximum number of "light" (i.e. non-heavy atoms) that are with the heavy atom
    if (qdrude) then
       ! Four hydrogens + one drude
       max_num_light = 5
    else
       ! Four hydrogens
       max_num_light = 4
    endif

    ngroup = 0
    do igrp=1,ngrp
       is = igpbs(igrp) + 1
       iq = igpbs(igrp+1)
       call atomid(is, sid, rid, ren, ac)
       if ((trim(ren) == 'TIP3' .and. iq-is+1 == 3) .or. &
            (trim(ren) == 'SPC' .and. iq-is+1 == 3)) then
          ngroup = ngroup + 1
          if (q_set_group) then
             group(ngroup) = group_in(is, iq, 1)
          endif
       elseif ((trim(ren) == 'TIP4') .or. (trim(ren) == 'SWM4') .or. (trim(ren) == 'SWM6')) then
          ! NOTE: We classify these solvents as solutes
          ngroup = ngroup + 1
          if (q_set_group) then
             group(ngroup) = group_in(is, iq, 0)
          endif
       else
          i = is
          if (iq-is == 0) then
             ! Single atom group
             if (nhydrog(i) > 0) then
                if (prnlev > 2) write (outu,'(a,2i6)') 'group,atom=',igrp,i
                call wrndie(-5,'<groupxfast>',&
                     'Structure is broken: hydrogen bond between two separate groups')
             endif
             ngroup = ngroup + 1
             if (q_set_group) then
                group(ngroup) = group_in(i, i, 0)
             endif
             i = i + 1
          else
             ! Multi atom group
             do while (i <= iq)
                if (hydrog(i)) then
                   ! Group starts with a hydrogen i
                   ! => add group and look for the heavy atom in the range i:i+max_num_light
                   ngroup = ngroup + 1
                   found_heavy = .false.
                   do i_heavy=i,min(i+max_num_light, iq)
                      if (.not.hydrog(i_heavy)) then
                         ! Found heavy atom at i_heavy
                         found_heavy = .true.
                         exit
                      endif
                   enddo
                   if (.not.found_heavy) then
                      write (outu,'(a,i8,/,4a)') &
                           '**** ERROR ***** Check group starting at atom',&
                           is,' with segid,resid,resname=',sid,rid,ren
                      call wrndie(-5,'<groupxfast>',&
                           'In RTF, hydrogens must be adjacent to heavy atoms they are bonded to')
                   endif
                   if (q_set_group) then
                      group(ngroup) = group_in(i, i + nhydrog(i_heavy), 0)
                   endif
                   ! Skip to the next group
                   i = i + nhydrog(i_heavy) + 1
                else
                   ! Check to make sure what we are adding are actually hydrogens
                   do j=i+1,i+nhydrog(i)
                      if (.not.hydrog(j)) then
                         write (outu,'(a,i8,/,4a)') &
                              '**** ERROR ***** Check group starting at atom',&
                              is,' with segid,resid,resname=',sid,rid,ren
                         call wrndie(-5,'<groupxfast>',&
                              'In RTF, hydrogens must be adjacent to heavy atoms that they are bonded to')
                      endif
                   enddo
                   ! Group starts with a heavy atom i
                   ! => add group and skip to the end
                   ngroup = ngroup + 1
                   if (q_set_group) then
                      group(ngroup) = group_in(i, i+nhydrog(i), 0)
                   endif
                   i = i + nhydrog(i) + 1
                endif
             enddo
          endif
       endif
    enddo

    if (q_set_group) then
       ! Check groups, and make invgroup
       i = 0
       do igrp=1,ngroup
          call group_out(group(igrp), is, iq)
          if (iq - is + 1 > max_group_size) then
             write (*,'(a,i5)') 'group size=',iq - is + 1
             call wrndie(-5,'<groupxfast>','Group has over 7 atoms')
          endif
          invgroup(is:iq) = igrp
          i = i + iq-is+1
          if (iq /= i) then
             write (outu,'(a,4i8)') 'igrp,is,iq,i=',igrp,is,iq,i
             call wrndie(-5,'<groupxfast>','Problem in group definitions')
          endif
       enddo
       ! Check the total number of atoms in groups
       if (i /= natom) then
          write (outu,'(a,2i6)') 'i,natom=',i,natom
          call wrndie(-5,'<groupxfast>',&
               'Number of atoms in groups does not match the total number of atoms')
       endif
    endif

    return
  end subroutine find_groups

  ! *
  ! * Prints out groups on screen. For DEBUGGING
  ! *
  subroutine print_groups()
    use stream,only:outu,prnlev
    implicit none
    ! Variables
    integer igrp, is, iq

    if (prnlev > 2) then
       do igrp=1,ngroup
          call group_out(group(igrp), is, iq)
          write (outu,'(a,i5,2i8)') 'igrp,is,iq=',igrp,is,iq
       enddo
    endif

    return
  end subroutine print_groups

  ! *
  ! * Builds nhydrog -list which tells the number of hydrogens (0 to 3) bonded to each
  ! * heavy atom
  ! *
  subroutine build_nhydrog(nhydrog)
    use psf,only:nbond, ib, jb ,natom, qdrude, isdrude
    use chutil,only:hydrog
    implicit none
    ! Input / Output
    integer nhydrog(*)
    ! Variables
    integer max_nhydrog
    integer i, t, h

    ! Count the number of hydrogens to the lower 16 bits and the number of drudes
    ! into the higher 16 bits. Combine at then end to a single count
    nhydrog(1:natom) = 0

    do i=1,nbond
       if (ib(i) <= 0 .or. jb(i) <= 0) cycle
       if (hydrog(ib(i))) then
          ! ib(i) is hydrogen
          t = jb(i)
          h = ib(i)
       elseif (hydrog(jb(i))) then
          ! jb(i) is hydrogen
          t = ib(i)
          h = jb(i)
       else
          ! Not a hydrogen bond, don't include
          cycle
       endif
       
       if (qdrude .and. isdrude(h)) then
          nhydrog(t) = nhydrog(t) + 65536
       else
          nhydrog(t) = nhydrog(t) + 1
       endif

       if (ishft(nhydrog(t),-16) > 1) then
          call wrndie(-5,'<groupxfast>',&
               'build_nhydrog, More than 1 drude per heavy atom not allowed!')
       endif

       if (iand(nhydrog(t),65535) > 4) then
          call wrndie(-5,'<groupxfast>','build_nhydrog, More than 4 hydrogen bonds not allowed!')
       endif

    enddo

    do i=1,natom
       nhydrog(i) = ishft(nhydrog(i),-16) + iand(nhydrog(i),65535)
    enddo

    return
  end subroutine build_nhydrog

  ! *
  ! * Divides atoms into groups: Solutes as single atom groups, solvents as 3 or 4 (TIP4) atom groups
  ! *
  subroutine make_groups(x, y, z)
    use psf,only:natom
    use number
    use memory
    use inbnd,only:cutnb
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    ! Variables
    integer, allocatable, dimension(:) :: nhydrog

    ! Check and reorder psf if needed
!    call reorder_psf()

    ! Calculate number of hydrogen bonds for each atom
    call chmalloc('groupxfast.src','make_groups','nhydrog',natom,intg=nhydrog)
    call build_nhydrog(nhydrog)

    ! Calculate number of groups
    call find_groups(.false., nhydrog)

    if (.not.allocated(group)) then
       call chmalloc('groupxfast.src','make_groups','group',ngroup+1,intg=group)
    elseif (size(group) < ngroup+1) then
       call chmdealloc('groupxfast.src','make_groups','group',size(group),intg=group)
       call chmalloc('groupxfast.src','make_groups','group',ngroup+1,intg=group)
    endif

    if (.not.allocated(groupcenter)) then
       call chmalloc('groupxfast.src','make_groups','groupcenter',3,ngroup,crl=groupcenter)
       call chmalloc('groupxfast.src','make_groups','groupbox',3,ngroup,crl=groupbox)
       call chmalloc('groupxfast.src','make_groups','groupsh',3,ngroup,crl=groupsh)
    elseif (size(groupcenter,2) < ngroup) then
       call chmdealloc('groupxfast.src','make_groups','groupcenter',3,size(groupcenter,2),&
            crl=groupcenter)
       call chmalloc('groupxfast.src','make_groups','groupcenter',3,ngroup,crl=groupcenter)
       call chmdealloc('groupxfast.src','make_groups','groupbox',3,size(groupbox,2),&
            crl=groupbox)
       call chmalloc('groupxfast.src','make_groups','groupbox',3,ngroup,crl=groupbox)
       call chmdealloc('groupxfast.src','make_groups','groupsh',3,size(groupsh,2),&
            crl=groupsh)
       call chmalloc('groupxfast.src','make_groups','groupsh',3,ngroup,crl=groupsh)
    endif

    if (allocated(grouploc)) then
       if (size(grouploc) < ngroup) then
          call chmdealloc('groupxfast.src','make_groups','grouploc',size(grouploc),intg=grouploc)
       endif
    endif
    if (.not.allocated(grouploc)) then
       call chmalloc('groupxfast.src','make_groups','grouploc',ngroup,intg=grouploc)
    endif

    if (.not.allocated(grouprad)) then
       call chmalloc('groupxfast.src','make_groups','grouprad',ngroup,crl=grouprad)
    elseif (size(grouprad) < ngroup) then
       call chmdealloc('groupxfast.src','make_groups','grouprad',size(grouprad),crl=grouprad)
       call chmalloc('groupxfast.src','make_groups','grouprad',ngroup,crl=grouprad)
    endif

    if (.not.allocated(invgroup)) then
       call chmalloc('groupxfast.src','make_groups','invgroup',natom,intg=invgroup)
    elseif (size(invgroup) < natom) then
       call chmdealloc('groupxfast.src','make_groups','invgroup',size(invgroup),intg=invgroup)
       call chmalloc('groupxfast.src','make_groups','invgroup',natom,intg=invgroup)
    endif

    ! Set groups
    call find_groups(.true., nhydrog)
    !call print_groups()

    call chmdealloc('groupxfast.src','make_groups','nhydrog',natom,intg=nhydrog)

    ! Calculate maximum group radius
    call calc_group_maxrad(x, y, z, maxgrp_rad, maxwater_rad)
    ! We assume the groups are very stiff, maximum 10% variation in group radius allowed
    maxgrp_rad = maxgrp_rad*1.10d0
    ! Assume water radius can vary by maximum of 3%
    maxwater_rad = maxwater_rad*1.03d0
    ! cut-off = cutnb + 2 x max group radius

    return
  end subroutine make_groups

#endif /* (domdec_main)*/

end module groupxfast

