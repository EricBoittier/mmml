!> Averages and fluctuations in unit cell dimensions
!> for constant pressure dynamics.
!> Caller is responsible for checking PRNLEV.
module avfl_ucell
   use chm_kinds
   use image, only: XTLTYP
   use number, only: ZERO
   use stream, only: OUTU
   implicit none

   !> Accumulation arrays for the unit cell.
   real(chm_real) :: uc1a(6), uc1b(6), uc2a(6), uc2b(6)
   real(chm_real) :: grad1a, grad1b, grad2a, grad2b

   private :: aver_msg, fluc_msg

contains

   subroutine avfl_ucell_reset
      uc1a = ZERO
      uc2a = ZERO
      grad1a = ZERO
      grad2a = ZERO
   end subroutine avfl_ucell_reset

   subroutine avfl_ucell_reset_lt
      uc1b = ZERO
      uc2b = ZERO
      grad1b = ZERO
      grad2b = ZERO
   end subroutine avfl_ucell_reset_lt

   subroutine avfl_ucell_update()
      use image, only: XUCELL, DXTL, XDIM
      use vector, only: dotvec
      real(chm_real) :: rval, gnorm

      uc1a = uc1a + XUCELL
      uc2a = uc2a + XUCELL ** 2
      rval = dotvec(DXTL, DXTL, XDIM)
      if (rval > ZERO) then
         gnorm = sqrt(rval / XDIM)
         grad1a = grad1a + gnorm
         grad2a = grad2a + gnorm ** 2
      endif
   end subroutine avfl_ucell_update

   subroutine avfl_ucell_update_lt()
      uc1b = uc1b + uc1a
      uc2b = uc2b + uc2a
      grad1b = grad1b + grad1a
      grad2b = grad2b + grad2a
   end subroutine avfl_ucell_update_lt

   subroutine avfl_ucell_compute(dnum)
      real(chm_real), intent(in) :: dnum
      real(chm_real) :: fluctc(6), fluctd

      uc1a = uc1a / dnum
      fluctc = uc2a / dnum - uc1a ** 2
      uc2a = ZERO
      where (fluctc > ZERO) uc2a = sqrt(fluctc)

      grad1a = grad1a / dnum
      fluctd = grad2a / dnum - grad1a ** 2
      grad2a = ZERO
      if (fluctd > ZERO) grad2a = sqrt(fluctd)
   end subroutine avfl_ucell_compute

   subroutine avfl_ucell_compute_lt(dnum)
      real(chm_real), intent(in) :: dnum
      real(chm_real) :: fluctc(6), fluctd

      uc1a = uc1b / dnum
      fluctc = uc2b / dnum - uc1a ** 2
      uc2a = ZERO
      where (fluctc > ZERO) uc2a = sqrt(fluctc)

      grad1a = grad1b / dnum
      fluctd = grad2b / dnum - grad1a ** 2
      grad2a = ZERO
      if (fluctd > ZERO) grad2a = sqrt(fluctd)
   end subroutine avfl_ucell_compute_lt

   subroutine avfl_ucell_print_aver(numstp)
      use averfluc, only: EPRSA
      integer, intent(in) :: numstp

      call aver_msg(numstp)
      call PRNXTLD(OUTU, 'AVER', XTLTYP, uc1a, .true., grad1a, &
            .true., EPRSA)
   end subroutine avfl_ucell_print_aver

   subroutine avfl_ucell_print_fluc(numstp)
      use averfluc, only: EPRS2A
      integer, intent(in) :: numstp

      call fluc_msg(numstp)
      CALL PRNXTLD(OUTU, 'FLUC', XTLTYP, uc2a, .TRUE., grad2a, &
            .true., EPRS2A)
   end subroutine avfl_ucell_print_fluc

   subroutine avfl_ucell_print_aver_lt(numstp)
      use averfluc, only: EPRSA
      integer, intent(in) :: numstp

      call aver_msg(numstp)
      call PRNXTLD(OUTU, 'LAVE', XTLTYP, uc1a, .TRUE., grad1a, &
            .true., EPRSA)
   end subroutine avfl_ucell_print_aver_lt

   subroutine avfl_ucell_print_fluc_lt(numstp)
      use averfluc, only: EPRS2A
      integer, intent(in) :: numstp

      call fluc_msg(numstp)
      call PRNXTLD(OUTU, 'LFLC', XTLTYP, uc2a, .TRUE., grad2a, &
            .true., EPRS2A)
   end subroutine avfl_ucell_print_fluc_lt

   subroutine aver_msg(numstp)
      integer, intent(in) :: numstp

      write (OUTU, '(A,I0,A)') &
            ' Lattice Parameters> Averages for the last ', numstp, ' steps:'
   end subroutine aver_msg

   subroutine fluc_msg(numstp)
      integer, intent(in) :: numstp

      write (OUTU, '(A,I0,A)') &
            ' Lattice Parameters> RMS fluctuations for the last ', numstp, ' steps:'
   end subroutine fluc_msg

end module avfl_ucell

