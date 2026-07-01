module enb_core_wrapper_mod

#if KEY_DOMDEC==1
  use chm_kinds
  use, intrinsic :: iso_c_binding
  use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
  ! Simple wrapper module for the nonbonded energy core routines
  private

  public invsqrt_approx

  public enb_vv_charmm_dp_sse, enb_uv_coul_vdw_dp_sse, enb_uv_vdw_dp_sse, enb_uu_coul_vdw_dp_sse
  public enb_uu_vdw_dp_sse

  public enb_vv_charmm_sp_sse, enb_uv_coul_vdw_sp_sse, enb_uv_vdw_sp_sse, enb_uu_coul_vdw_sp_sse
  public enb_uu_vdw_sp_sse

  public enb_vv_charmm_dp_avx, enb_uv_coul_vdw_dp_avx, enb_uv_vdw_dp_avx, enb_uu_coul_vdw_dp_avx
  public enb_uu_vdw_dp_avx

  public enb_vv_charmm_sp_avx, enb_uv_coul_vdw_sp_avx, enb_uv_vdw_sp_avx, enb_uu_coul_vdw_sp_avx
  public enb_uu_vdw_sp_avx

  interface

    subroutine invsqrt_approx(x, y) bind(C)
      import
      real(c_float), intent(in) :: x
      real(c_float), intent(out) :: y
    end subroutine invsqrt_approx

    subroutine enb_vv_charmm_dp_sse(ni, indi, indj, startj, pftable, hinv, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
         c6multOO, c6multOH, c6multHH, &
#endif
         qOO, qOH, qHH, roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
#if KEY_LJPME==1
      real(c_double), intent(in) :: c6multOO, c6multOH, c6multHH
#endif
      real(c_double), intent(in) :: qOO, qOH, qHH
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_vv_charmm_dp_sse

    subroutine enb_uv_coul_vdw_dp_sse(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_coul_vdw_dp_sse

    subroutine enb_uv_vdw_dp_sse(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_vdw_dp_sse

    subroutine enb_uu_coul_vdw_dp_sse(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_coul_vdw_dp_sse

    subroutine enb_uu_vdw_dp_sse(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_vdw_dp_sse

    subroutine enb_vv_charmm_sp_sse(ni, indi, indj, startj, pftable, hinv, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
         c6multOO, c6multOH, c6multHH, &
#endif
         qOO, qOH, qHH, roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
#if KEY_LJPME==1
      real(c_double), intent(in) :: c6multOO, c6multOH, c6multHH
#endif
      real(c_double), intent(in) :: qOO, qOH, qHH
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_vv_charmm_sp_sse

    subroutine enb_uv_coul_vdw_sp_sse(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_float), intent(in) :: vdwparam(*)
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_coul_vdw_sp_sse

    subroutine enb_uv_vdw_sp_sse(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_float), intent(in) :: vdwparam(*)
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_vdw_sp_sse

    subroutine enb_uu_coul_vdw_sp_sse(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_float), intent(in) :: vdwparam(*)
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_coul_vdw_sp_sse

    subroutine enb_uu_vdw_sp_sse(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_float), intent(in) :: vdwparam(*)
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_vdw_sp_sse


    !-----------------------------------------------------------------
    ! --------------------------- AVX --------------------------------
    !-----------------------------------------------------------------

    subroutine enb_vv_charmm_dp_avx(ni, indi, indj, startj, pftable, hinv, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
         c6multOO, c6multOH, c6multHH, &
#endif
         qOO, qOH, qHH, roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
#if KEY_LJPME==1
      real(c_double), intent(in) :: c6multOO, c6multOH, c6multHH
#endif
      real(c_double), intent(in) :: qOO, qOH, qHH
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_vv_charmm_dp_avx

    subroutine enb_uv_coul_vdw_dp_avx(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_coul_vdw_dp_avx

    subroutine enb_uv_vdw_dp_avx(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_vdw_dp_avx

    subroutine enb_uu_coul_vdw_dp_avx(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_coul_vdw_dp_avx

    subroutine enb_uu_vdw_dp_avx(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_vdw_dp_avx

    subroutine enb_vv_charmm_sp_avx(ni, indi, indj, startj, pftable, hinv, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
         c6multOO, c6multOH, c6multHH, &
#endif
         qOO, qOH, qHH, roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
#if KEY_LJPME==1
      real(c_double), intent(in) :: c6multOO, c6multOH, c6multHH
#endif
      real(c_double), intent(in) :: qOO, qOH, qHH
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_vv_charmm_sp_avx

    subroutine enb_uv_coul_vdw_sp_avx(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_float), intent(in) :: vdwparam(*)
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_coul_vdw_sp_avx

    subroutine enb_uv_vdw_sp_avx(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_float), intent(in) :: vdwparam(*)
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_vdw_sp_avx

    subroutine enb_uu_coul_vdw_sp_avx(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_float), intent(in) :: vdwparam(*)
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_coul_vdw_sp_avx

    subroutine enb_uu_vdw_sp_avx(ni, indi, indj, startj,&
         pftable, hinv, vdwtype, vdwparam, xyzq, &
         force, iscoord, sforce, scoordtab, vdwpot, &
         roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_float), intent(in) :: pftable(*)
      real(c_double), intent(in) :: hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_float), intent(in) :: vdwparam(*)
      type(xyzq_sp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_float), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_vdw_sp_avx

  end interface

#endif

end module enb_core_wrapper_mod
