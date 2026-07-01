
#ifdef BLOCK_ON
  subroutine KERNEL_NAME(ig, xyzq, top_excl_pos, top_excl, is, iq, itype, is_unpack, ish,&
       ntmpl, tmpl, nblist, group, group_unpack)
!    use groupxfast,only:group_out
!    use nblist_types,only:nblist_pair_t
!    use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
    ! Input / Output
    integer, intent(in) :: ig, is, iq, itype, is_unpack, ish, tmpl(2,*), ntmpl
    integer, intent(in) :: top_excl_pos(:), top_excl(:), group_unpack(:), group(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(nblist_pair_t), intent(inout) :: nblist(:)
    ! Variables
    integer ii, jj, js, jq, j, jg, jtype, ii_unpack, jj_unpack, dummy
    integer excl_start, excl_end, min_excl_j, max_excl_j

    if (q_block_in_use_ij(is, iq, is_unpack, ntmpl, tmpl, group, group_unpack)) then
       ! BLOCK in use. We will skip 1-4 interactions and exclusions since these are taken care
       ! by ebonded_domdec.src and enbfix14.src
       ii_unpack = is_unpack
       do ii=is,iq
          ! Start neighbor lists
          call add_i_nblist(nblist(iuuc_block)%ni, nblist(iuuc_block)%nj, &
               nblist(iuuc_block)%ni_max, nblist(iuuc_block)%indi, nblist(iuuc_block)%startj, &
               nblist(iuuc_block)%iscoord, ish, ii)
          !-------------------------------------------------------------------------------
          excl_start = top_excl_pos(ii_unpack)
          excl_end = top_excl_pos(ii_unpack+1) - 1
          min_excl_j = top_excl(excl_start)
          max_excl_j = top_excl(excl_end)

          ! Loop over charge groups j
          do j=1,ntmpl
             jg = tmpl(1,j)
             call group_out(group(jg), js, jq, jtype)
             if (jtype == 1) then                
                ! Solute (ig) - Solvent (jg)
                ! Note: this interaction cannot have 1-4 interactions or exclusions
                do jj=js,jq
                   call add_j_nblist(nblist(iuuc_block)%nj, nblist(iuuc_block)%nj_max, &
                        nblist(iuuc_block)%indj, jj)
                enddo
                nblist(iuuc_block)%np = nblist(iuuc_block)%np + (jq-js+1)
             else
                ! Solute (ig) - Solute (jg)
                jg = tmpl(2,j)
                call group_out(group_unpack(jg), jj_unpack, dummy)

                ! j atom range is jj_unpack:jj_unpack+jq-js
                ! exclusion range is min_excl_j:max_excl_j
                if (jj_unpack+jq-js < min_excl_j .or. jj_unpack > max_excl_j) then
                   ! No exclusions
                   do jj=js,jq
                      nblist(iuuc_block)%np = nblist(iuuc_block)%np + 1
                      call add_j_nblist(nblist(iuuc_block)%nj, nblist(iuuc_block)%nj_max, &
                           nblist(iuuc_block)%indj, jj)
                   enddo
                else
                   ! Possible exclusions
                   do jj=js,jq
                      if (jj_unpack < min_excl_j .or. jj_unpack > max_excl_j .or. &
                           .not.any(top_excl(excl_start:excl_end) == jj_unpack)) then
                         ! No exclusion
                         nblist(iuuc_block)%np = nblist(iuuc_block)%np + 1
                         call add_j_nblist(nblist(iuuc_block)%nj, nblist(iuuc_block)%nj_max, &
                              nblist(iuuc_block)%indj, jj)
                      endif
                      jj_unpack = jj_unpack + 1
                   enddo
                endif
             endif
          enddo  ! do j=1,ntmpl
          !-------------------------------------------------------------------------------
          call stop_i_nblist(nblist(iuuc_block)%ni, nblist(iuuc_block)%nj, &
               nblist(iuuc_block)%startj)
          ii_unpack = ii_unpack + 1
       enddo  ! do ii=is,iq
    else
#ifdef DOUBLE_PREC
       call flush_pair_pd&
#else
       call flush_pair_ps&
#endif
       (ig, xyzq, top_excl_pos, top_excl, is, iq, itype, is_unpack, ish, &
       ntmpl, tmpl, nblist, group, group_unpack)
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef GROUP_ON
  ! *
  ! * Flush the temporary list by adding entries to the actual Pair neighbor list
  ! *
  subroutine KERNEL_NAME(ig, xyzq, top_excl_pos, top_excl, is, iq, itype, is_unpack, ish, &
       ntmpl, tmpl, nblist, group, group_unpack)
!    use groupxfast,only:group_out
!    use nblist_types,only:nblist_pair_t
!    use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: ig, is, iq, itype, is_unpack, ish, tmpl(2,*), ntmpl
    integer, intent(in) :: top_excl_pos(:), top_excl(:), group_unpack(:), group(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), parameter :: zero = 0.0_chm_real
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), parameter :: zero = 0.0_chm_real4
#endif
    type(nblist_pair_t), intent(inout) :: nblist(:)
    ! Variables
    integer j, jg, js, jq, jtype, ii, jj, ii_unpack, jj_unpack, dummy
    integer excl_start, excl_end, min_excl_j, max_excl_j

    if (itype == 1) then
       ! Group ig is solvent

       ! Start neighbor lists
       call add_i_nblist(nblist(iuvc)%ni, nblist(iuvc)%nj, nblist(iuvc)%ni_max, &
            nblist(iuvc)%indi, nblist(iuvc)%startj, nblist(iuvc)%iscoord, ish, is)
       call add_i_nblist(nblist(iuv)%ni, nblist(iuv)%nj, nblist(iuv)%ni_max, &
            nblist(iuv)%indi, nblist(iuv)%startj, nblist(iuv)%iscoord, ish, is)
       call add_i_nblist(nblist(ivv)%ni, nblist(ivv)%nj, nblist(ivv)%ni_max, &
            nblist(ivv)%indi, nblist(ivv)%startj, nblist(ivv)%iscoord, ish, is)

       ! Loop over charge groups j
       do j=1,ntmpl
          jg = tmpl(1,j)
          call group_out(group(jg), js, jq, jtype)
          if (jtype == 1) then
             ! Solvent (ig) - Solvent (jg)
             nblist(ivv)%np = nblist(ivv)%np + (iq-is+1)*(iq-is+1)
             call add_j_nblist(nblist(ivv)%nj, nblist(ivv)%nj_max, nblist(ivv)%indj, js)
          else
             ! Solvent (ig) - Solute (jg)
             do jj=js,jq
                if (xyzq(jj)%q /= zero) then
                   nblist(iuvc)%np = nblist(iuvc)%np + (iq-is+1)
                   call add_j_nblist(nblist(iuvc)%nj, nblist(iuvc)%nj_max, nblist(iuvc)%indj, jj)
                else
                   nblist(iuv)%np = nblist(iuv)%np + (iq-is+1)
                   call add_j_nblist(nblist(iuv)%nj, nblist(iuv)%nj_max, nblist(iuv)%indj, jj)
                endif
             enddo
          endif
       enddo
       
       call stop_i_nblist(nblist(iuvc)%ni, nblist(iuvc)%nj, nblist(iuvc)%startj)
       call stop_i_nblist(nblist(iuv)%ni, nblist(iuv)%nj, nblist(iuv)%startj)
       call stop_i_nblist(nblist(ivv)%ni, nblist(ivv)%nj, nblist(ivv)%startj)

    else
       ! Group ig is solute
       ii_unpack = is_unpack
       do ii=is,iq
          ! Start neighbor lists
          call add_i_nblist(nblist(iuuc)%ni, nblist(iuuc)%nj, nblist(iuuc)%ni_max, &
               nblist(iuuc)%indi, nblist(iuuc)%startj, nblist(iuuc)%iscoord, ish, ii)
          call add_i_nblist(nblist(iuu)%ni, nblist(iuu)%nj, nblist(iuu)%ni_max, &
               nblist(iuu)%indi, nblist(iuu)%startj, nblist(iuu)%iscoord, ish, ii)

          excl_start = top_excl_pos(ii_unpack)
          excl_end = top_excl_pos(ii_unpack+1) - 1
          min_excl_j = top_excl(excl_start)
          max_excl_j = top_excl(excl_end)

          ! Loop over charge groups j
          do j=1,ntmpl
             jg = tmpl(1,j)
             call group_out(group(jg), js, jq, jtype)
             if (jtype == 1) then
                ! Solute (ig) - Solvent (jg)
                ! Note: this interaction cannot have 1-4 interactions or exclusions
                if (xyzq(ii)%q /= zero) then
                   do jj=js,jq
                      call add_j_nblist(nblist(iuuc)%nj, nblist(iuuc)%nj_max, &
                           nblist(iuuc)%indj, jj)
                   enddo
                   nblist(iuuc)%np = nblist(iuuc)%np + (jq-js+1)
                else
                   do jj=js,jq
                      call add_j_nblist(nblist(iuu)%nj, nblist(iuu)%nj_max, &
                           nblist(iuu)%indj, jj)
                   enddo
                   nblist(iuu)%np = nblist(iuu)%np + (jq-js+1)
                endif
             else
                ! Solute (ig) - Solute (jg)
                jg = tmpl(2,j)
                call group_out(group_unpack(jg), jj_unpack, dummy)
                ! j atom range is jj_unpack:jj_unpack+jq-js
                ! exclusion range is min_excl_j:max_excl_j
                if (jj_unpack+jq-js < min_excl_j .or. jj_unpack > max_excl_j) then
                   ! No exclusions
                   do jj=js,jq
                      if (xyzq(ii)%q /= zero .and. xyzq(jj)%q /= zero) then
                         nblist(iuuc)%np = nblist(iuuc)%np + 1
                         call add_j_nblist(nblist(iuuc)%nj, nblist(iuuc)%nj_max, &
                              nblist(iuuc)%indj, jj)
                      else
                         nblist(iuu)%np = nblist(iuu)%np + 1
                         call add_j_nblist(nblist(iuu)%nj, nblist(iuu)%nj_max, &
                              nblist(iuu)%indj, jj)
                      endif
                   enddo
                else
                   ! Possible exclusions
                   do jj=js,jq
                      if (jj_unpack < min_excl_j .or. jj_unpack > max_excl_j .or. &
                           .not.any(top_excl(excl_start:excl_end) == jj_unpack)) then
                         ! No exclusion
                         if (xyzq(ii)%q /= zero .and. xyzq(jj)%q /= zero) then
                            nblist(iuuc)%np = nblist(iuuc)%np + 1
                            call add_j_nblist(nblist(iuuc)%nj, nblist(iuuc)%nj_max, &
                                 nblist(iuuc)%indj, jj)
                         else
                            nblist(iuu)%np = nblist(iuu)%np + 1
                            call add_j_nblist(nblist(iuu)%nj, nblist(iuu)%nj_max, &
                                 nblist(iuu)%indj, jj)
                         endif
                      endif
                      jj_unpack = jj_unpack + 1
                   enddo
                endif
             endif
          enddo  ! do j=1,ntmpl
          call stop_i_nblist(nblist(iuuc)%ni, nblist(iuuc)%nj, nblist(iuuc)%startj)
          call stop_i_nblist(nblist(iuu)%ni, nblist(iuu)%nj, nblist(iuu)%startj)
          ii_unpack = ii_unpack + 1
       enddo  ! do ii=is,iq
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef ATOM_ON
  ! *
  ! * Flush the temporary list by adding entries to the actual Pair neighbor list
  ! *
  subroutine KERNEL_NAME(ig, xyzq, is, iq, itype, is_unpack, ish, &
       ntmpl, tmpl, nblist)
!    use groupxfast,only:group_out
!    use nblist_types,only:nblist_pair_t
!    use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: ig, is, iq, itype, is_unpack, ish, tmpl(2,*), ntmpl
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), parameter :: zero = 0.0_chm_real
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), parameter :: zero = 0.0_chm_real4
#endif
    type(nblist_pair_t), intent(inout) :: nblist(:)
    ! Variables
    integer j, jg, js, jq, jtype, ii, jj, ii_unpack, jj_unpack, dummy
    integer excl_start, excl_end, min_excl_j, max_excl_j

    ! Group ig is solute
    ii_unpack = is_unpack
    do ii=is,iq
       ! Start neighbor lists
       call add_i_nblist(nblist(iuuc)%ni, nblist(iuuc)%nj, nblist(iuuc)%ni_max, &
            nblist(iuuc)%indi, nblist(iuuc)%startj, nblist(iuuc)%iscoord, ish, ii)
       call add_i_nblist(nblist(iuu)%ni, nblist(iuu)%nj, nblist(iuu)%ni_max, &
            nblist(iuu)%indi, nblist(iuu)%startj, nblist(iuu)%iscoord, ish, ii)
       
       excl_start = top_excl_pos(ii_unpack)
       excl_end = top_excl_pos(ii_unpack+1) - 1
       min_excl_j = top_excl(excl_start)
       max_excl_j = top_excl(excl_end)
       
       ! Loop over charge groups j
       do j=1,ntmpl
          jg = tmpl(1,j)
          call group_out(group(jg), js, jq, jtype)
          if (jtype == 1) then                
             ! Solute (ig) - Solvent (jg)
             ! Note: this interaction cannot have 1-4 interactions or exclusions
             if (xyzq(ii)%q /= zero) then
                do jj=js,jq
                   call add_j_nblist(nblist(iuuc)%nj, nblist(iuuc)%nj_max, &
                        nblist(iuuc)%indj, jj)
                enddo
                nblist(iuuc)%np = nblist(iuuc)%np + (jq-js+1)
             else
                do jj=js,jq
                   call add_j_nblist(nblist(iuu)%nj, nblist(iuu)%nj_max, &
                        nblist(iuu)%indj, jj)
                enddo
                nblist(iuu)%np = nblist(iuu)%np + (jq-js+1)
             endif
          else
             ! Solute (ig) - Solute (jg)
             jg = tmpl(2,j)
             call group_out(group_unpack(jg), jj_unpack, dummy)
             
             ! j atom range is jj_unpack:jj_unpack+jq-js
             ! exclusion range is min_excl_j:max_excl_j
             if (jj_unpack+jq-js < min_excl_j .or. jj_unpack > max_excl_j) then
                ! No exclusions
                do jj=js,jq
                   if (xyzq(ii)%q /= zero .and. xyzq(jj)%q /= zero) then
                      nblist(iuuc)%np = nblist(iuuc)%np + 1
                      call add_j_nblist(nblist(iuuc)%nj, nblist(iuuc)%nj_max, &
                           nblist(iuuc)%indj, jj)
                   else
                      nblist(iuu)%np = nblist(iuu)%np + 1
                      call add_j_nblist(nblist(iuu)%nj, nblist(iuu)%nj_max, &
                           nblist(iuu)%indj, jj)
                   endif
                enddo
             else
                ! Possible exclusions
                do jj=js,jq
                   if (jj_unpack < min_excl_j .or. jj_unpack > max_excl_j .or. &
                        .not.any(top_excl(excl_start:excl_end) == jj_unpack)) then
                      ! No exclusion
                      if (xyzq(ii)%q /= zero .and. xyzq(jj)%q /= zero) then
                         nblist(iuuc)%np = nblist(iuuc)%np + 1
                         call add_j_nblist(nblist(iuuc)%nj, nblist(iuuc)%nj_max, &
                              nblist(iuuc)%indj, jj)
                      else
                         nblist(iuu)%np = nblist(iuu)%np + 1
                         call add_j_nblist(nblist(iuu)%nj, nblist(iuu)%nj_max, &
                              nblist(iuu)%indj, jj)
                      endif
                   endif
                   jj_unpack = jj_unpack + 1
                enddo
             endif
          endif
       enddo  ! do j=1,ntmpl
       call stop_i_nblist(nblist(iuuc)%ni, nblist(iuuc)%nj, nblist(iuuc)%startj)
       call stop_i_nblist(nblist(iuu)%ni, nblist(iuu)%nj, nblist(iuu)%startj)
       ii_unpack = ii_unpack + 1
    enddo  ! do ii=is,iq

    return
  end subroutine KERNEL_NAME
#endif
