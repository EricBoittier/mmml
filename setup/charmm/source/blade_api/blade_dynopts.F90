module blade_dynopts
   use chm_kinds
   use stream, only: OUTU, PRNLEV
   implicit none

   type blade_dynopts_t
      real(chm_real) :: temperatureReference, pressureReference
      real(chm_real) :: volumeFluctuation
      integer :: pressureFrequency
      logical :: blade_qrexchg
   end type blade_dynopts_t

contains

   !> Parses BLaDE temperature and pressure control options
   !> from a DYNA command.
   function blade_parse_options(comlyn, comlen, temnew, &
        qrexchg, caller) result(opts)
      use reawri, only: TREF, FINALT
      use string
      use number
      ! use blade_main, only: blade_initialized

      type(blade_dynopts_t) :: opts
      character(len=*), intent(inout) :: comlyn
      integer, intent(inout) :: comlen
      real(chm_real), intent(in) :: temnew
      logical, intent(in) :: qrexchg
      character(len=*), intent(in) :: caller

      opts%blade_qrexchg = qrexchg

      opts%pressureFrequency = 0
      if (indxa(comlyn,comlen,'PRMC') > 0) then
         opts%pressureFrequency = 50
      endif

      opts%pressureFrequency = gtrmi(comlyn,comlen,'IPRS',opts%pressureFrequency)
      opts%pressureReference = gtrmf(comlyn,comlen,'PREF',one)
      opts%volumeFluctuation = gtrmf(comlyn,comlen,'PRDV',100.0)

      opts%temperatureReference = FINALT

      ! this will cause a module dependency cycle
      ! this subroutine could be called from blade_ctrl
      ! or some other module where after call returns,
      ! blade_initialized can be set in the calling routine
      ! blade_initialized = .false. ! Need to tear down and rebuild if any options changed
   end function blade_parse_options

   !> Outputs a description of temperature and pressure control options.
   subroutine blade_report_options(opts, caller)
      use reawri, only: FINALT, ISEED

      type(blade_dynopts_t), intent(in) :: opts
      character(len=*), intent(in) :: caller
      character(len=10) :: tag

      if (PRNLEV < 2) return
      write (tag, '(3a)') ' ', caller, '> '
      write (OUTU,'(2a)') tag, 'BLaDE interface requested for energy and dynamics calculations.'
      if(opts%blade_qrexchg) write (OUTU,'(2a)') tag, 'BLaDE T-replica exchange dynamics being run.'

      if (opts%pressureFrequency.gt.0) then
         write (OUTU,'(2a)') tag, &
               'CPT dynamics through BLaDE interface requested using Langevin heatbath.'
         write (OUTU,'(2a,2x,f6.2,2x,a)') tag, &
               'MC barostat coupled to reference pressure', opts%pressureReference, 'atmospheres.'
         write (OUTU,'(2a,2x,f6.2,2x,a)') tag, &
               'MC barostat using reference temperature', opts%temperatureReference, 'K.'
         write (OUTU,'(2a,2x,i5,2x,a)') tag, &
               'MC barostat volume move attempted every', opts%pressureFrequency, 'timesteps.'
         write (OUTU,'(2a,2x,f6.2,2x,a)') tag, &
               'MC barostat volume move size', opts%volumeFluctuation, 'A^3'
      else
         write (OUTU,'(2a)') tag, &
               'NVT dynamics through BLaDE interface requested using Langevin heatbath.'
         write (OUTU,'(2a,2x,f6.2,2x,a)') tag, &
               'thermostate using reference temperature', opts%temperatureReference, 'K.'
      endif
   end subroutine blade_report_options
end module blade_dynopts

