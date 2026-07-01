!> Averages and fluctuations in energy terms for dynamics.
!> Caller is responsible for checking PRNLEV.
module averfluc
   use chm_kinds
   use energym, only: LENENP, LENENT, LENENV
   use number, only: ZERO
   use stream, only: OUTU
   implicit none

   ! dynio knows too much to make these private yet

   !> Short-term accumulation arrays.
   real(chm_real) :: &
         eprpa(LENENP), eprp2a(LENENP), &
         etrma(LENENT), etrm2a(LENENT), &
         eprsa(LENENV), eprs2a(LENENV)

   !> Long-term accumulation arrays.
   real(chm_real) :: &
         eprpp(LENENP), eprp2p(LENENP), &
         etrmp(LENENT), etrm2p(LENENT), &
         eprsp(LENENV), eprs2p(LENENV)

   private :: aver_msg, fluc_msg

contains

   !> Resets the short-term accumulation arrays.
   subroutine avfl_reset()
      eprpa = ZERO
      eprp2a = ZERO
      etrma = ZERO
      etrm2a = ZERO
      eprsa = ZERO
      eprs2a = ZERO
   end subroutine avfl_reset

   !> Resets the long-term accumulation arrays.
   subroutine avfl_reset_lt()
      eprpp = ZERO
      eprp2p = ZERO
      etrmp = ZERO
      etrm2p = ZERO
      eprsp = ZERO
      eprs2p = ZERO
   end subroutine avfl_reset_lt

   !> Adds current energy values to the short-term accumulation arrays.
   subroutine avfl_update(eprop, eterm, epress)
      real(chm_real), intent(in) :: eprop(LENENP), eterm(LENENT), epress(LENENV)
      eprpa = eprpa + eprop
      eprp2a = eprp2a + eprop ** 2
      etrma = etrma + eterm
      etrm2a = etrm2a + eterm ** 2
      eprsa = eprsa + epress
      eprs2a = eprs2a + epress ** 2
   end subroutine avfl_update

   !> Adds short-term accumulation arrays to long-term accumulation arrays.
   subroutine avfl_update_lt()
      eprpp = eprpp + eprpa
      eprp2p = eprp2p + eprp2a
      etrmp = etrmp + etrma
      etrm2p = etrm2p + etrm2a
      eprsp = eprsp + eprsa
      eprs2p = eprs2p + eprs2a
   end subroutine avfl_update_lt

   !> Computes short-term averages and fluctuations.
   subroutine avfl_compute(dnum)
      real(chm_real), intent(in) :: dnum
      real(chm_real) :: fluctp(LENENP), fluctt(LENENT), fluctv(LENENV)

      eprpa = eprpa / dnum
      fluctp = eprp2a / dnum - eprpa ** 2
      eprp2a = ZERO
      where (fluctp > ZERO) eprp2a = sqrt(fluctp)

      etrma = etrma / dnum
      fluctt = etrm2a / dnum - etrma ** 2
      etrm2a = ZERO
      where (fluctt > ZERO) etrm2a = sqrt(fluctt)

      eprsa = eprsa / dnum
      fluctv = eprs2a / dnum - eprsa ** 2
      eprs2a = ZERO
      where (fluctv > ZERO) eprs2a = sqrt(fluctv)
   end subroutine avfl_compute

   !> Computes long-term averages and fluctuations.
   subroutine avfl_compute_lt(dnum)
      real(chm_real), intent(in) :: dnum
      real(chm_real) :: fluctp(LENENP), fluctt(LENENT), fluctv(LENENV)

      eprpa = eprpp / dnum
      fluctp = eprp2p / dnum - eprpa ** 2
      eprp2a = ZERO
      where (fluctp > ZERO) eprp2a = sqrt(fluctp)

      etrma = etrmp / dnum
      fluctt = etrm2p / dnum - etrma ** 2
      etrm2a = ZERO
      where (fluctt > ZERO) etrm2a = sqrt(fluctt)

      eprsa = eprsp / dnum
      fluctv = eprs2p / dnum - eprsa ** 2
      eprs2a = ZERO
      where (fluctv > ZERO) eprs2a = sqrt(fluctv)
   end subroutine avfl_compute_lt

   subroutine avfl_print_aver(numstp, time, tag)
      integer, intent(in) :: numstp
      real(chm_real), intent(in) :: time
      character(len=*), intent(in), optional :: tag

      call aver_msg(numstp, tag)
      call PRINTE(OUTU, eprpa, etrma, 'AVER', 'DYN', .true., &
            numstp, time, ZERO, .true.)
   end subroutine avfl_print_aver

   subroutine avfl_print_fluc(numstp, time, tag)
      integer, intent(in) :: numstp
      real(chm_real), intent(in) :: time
      character(len=*), intent(in), optional :: tag

      call fluc_msg(numstp, tag)
      call PRINTE(OUTU, eprp2a, etrm2a, 'FLUC', 'DYN', .false., &
            numstp, time, ZERO, .true.)
   end subroutine avfl_print_fluc

   subroutine avfl_print_aver_lt(numstp, time, tag)
      integer, intent(in) :: numstp
      real(chm_real), intent(in) :: time
      character(len=*), intent(in), optional :: tag

      call aver_msg(numstp, tag)
      call PRINTE(OUTU, eprpa, etrma, 'LAVE', 'DYN', .false., &
            numstp, time, ZERO, .true.)
   end subroutine avfl_print_aver_lt

   subroutine avfl_print_fluc_lt(numstp, time, tag)
      integer, intent(in) :: numstp
      real(chm_real), intent(in) :: time
      character(len=*), intent(in), optional :: tag

      call fluc_msg(numstp, tag)
      call PRINTE(OUTU, eprp2a, etrm2a, 'LFLC', 'DYN', .false., &
            numstp, time, ZERO, .true.)
   end subroutine avfl_print_fluc_lt

   subroutine aver_msg(numstp, tag)
      integer, intent(in) :: numstp
      character(len=*), intent(in), optional :: tag

      if (present(tag)) then
         write (OUTU, '(X,2A,I0,A)') &
               tag, ' Averages for the last ', numstp, ' steps:'
      else
         ! match legacy output
         write (OUTU, '(A,I8,A)') &
               ' * * * AVERAGES FOR THE LAST ', numstp, ' STEPS'
      endif
   end subroutine aver_msg

   subroutine fluc_msg(numstp, tag)
      integer, intent(in) :: numstp
      character(len=*), intent(in), optional :: tag

      if (present(tag)) then
         write (OUTU, '(X,2A,I0,A)') &
               tag, ' RMS fluctuations for the last ', numstp,' steps:'
      else
         ! match legacy output
         write (OUTU, '(A,I8,A)') &
               ' * * * RMS FLUCTUATIONS FOR ', numstp,' STEPS'
      endif
   end subroutine fluc_msg

end module averfluc

