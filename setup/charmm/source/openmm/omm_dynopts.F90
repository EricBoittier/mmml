module omm_dynopts
   use chm_kinds
   use stream, only: OUTU, PRNLEV
   implicit none

   type omm_dynopts_t
      real(chm_real) :: collisionFrequency, frictionCoefficient
      real(chm_real) :: temperatureReference, pressureReference, variableTS_tol
      real(chm_real) :: surfaceTension, numberInterfaces
      real(chm_real) :: pressureIn3Dimensions(3)
      integer :: pressureFrequency
      logical :: qAndersen, qPressure, qVariable, qLangevin, qPressure2
      logical :: omm_qrexchg, omm_updateT = .false.
      character(len=10) :: frictionSource
   end type omm_dynopts_t

contains

   !> Parses OpenMM temperature and pressure control options
   !> from a DYNA command.
   function omm_parse_options(comlyn, comlen, temnew, &
        qrexchg, caller) result(opts)
      use reawri, only: TREF, FINALT
      use contrl, only: ILANG
      use cnst_fcm, only: FBETA
      use string
      use number

      type(omm_dynopts_t) :: opts
      character(len=*), intent(inout) :: comlyn
      integer, intent(inout) :: comlen
      real(chm_real), intent(in) :: temnew
      logical, intent(in) :: qrexchg
      character(len=*), intent(in) :: caller

      opts%omm_qrexchg = qrexchg
      opts%qAndersen = indxa(comlyn,comlen,'ANDE') > 0
      if (opts%qAndersen) then
         opts%collisionFrequency = GTRMF(COMLYN,COMLEN,'COLF',thosnd)
      endif

      opts%qLangevin = ILANG == 1
      if (opts%qLangevin) then
         opts%frictionCoefficient = GTRMF(COMLYN,COMLEN,'GAMM',zero)
         if (opts%frictionCoefficient > ZERO) then
            opts%frictionSource = 'specified'
         else
            if (allocated(FBETA)) opts%frictionCoefficient = maxval(FBETA)
            if (opts%frictionCoefficient > ZERO) then
               opts%frictionSource = 'max(FBETA)'
            else
               opts%frictionCoefficient = FIVE
               opts%frictionSource = 'default'
            endif
         endif
      endif

      opts%qPressure = indxa(comlyn,comlen,'PRMC') > 0
      if (opts%qPressure) then
         opts%pressureFrequency = gtrmi(comlyn,comlen,'IPRS',25)
         opts%pressureReference = gtrmf(comlyn,comlen,'PREF',one)
         opts%pressureIn3Dimensions(1) = gtrmf(comlyn,comlen,'PRXX',zero)
         opts%pressureIn3Dimensions(2) = gtrmf(comlyn,comlen,'PRYY',zero)
         opts%pressureIn3Dimensions(3) = gtrmf(comlyn,comlen,'PRZZ',zero)
         opts%surfaceTension = gtrmf(comlyn,comlen,'TENS',zero)
         ! Look for variable for constant surface tension specifying
         ! how many interfaces present. Usual set-up yields two
         opts%numberInterfaces = gtrmf(comlyn,comlen,'NINT',two)
         opts%surfaceTension = opts%surfaceTension * opts%numberInterfaces
         opts%qPressure2 = any(opts%pressureIn3Dimensions /= ZERO) &
               .or. opts%surfaceTension /= ZERO
      endif

      opts%qVariable = indxa(comlyn,comlen,'VARI') > 0
      if (opts%qVariable) then
         opts%variableTS_tol = GTRMF(COMLYN,COMLEN,'VTOL',tenm3)
      endif

      if (opts%qAndersen .or. opts%qLangevin) then
         opts%temperatureReference = FINALT
      else if (opts%qPressure) then
         call wrndie(-1, caller, &
               'Andersen or Langevin heatbath required with MC barostat in OpenMM.')
      endif
#if KEY_OPENMM==0
      if (opts%qPressure .and. opts%qPressure2) then
         call wrndie(-5, caller, &
               'PRMC options PRXX, PRYY, PRZZ, TENS require OpenMM with CHARMM plugin.')
      endif
#endif 
   end function omm_parse_options

   !> Outputs a description of temperature and pressure control options.
   subroutine omm_report_options(opts, caller)
      use reawri, only: FINALT, ISEED

      type(omm_dynopts_t), intent(in) :: opts
      character(len=*), intent(in) :: caller
      character(len=10) :: tag

      if (PRNLEV < 2) return
      write (tag, '(3a)') ' ', caller, '> '
      write (OUTU,'(2a)') tag, 'OpenMM interface requested for energy and dynamics calculations.'
      if(opts%omm_qrexchg) write (OUTU,'(2a)') tag, 'OpenMM T-replica exchange dynamics being run.'

      if (opts%qAndersen) then
         if(opts%omm_qrexchg) call wrndie(-1, caller, &
              'Andersen heatbath not compatible with T-rex in OpenMM, use Langevin.')
         WRITE (OUTU,'(2a)') tag, 'Constant temperature w/ OpenMM using Andersen heatbath requested.'
         WRITE (OUTU,'(2(8X,A,F12.7,A,/))') &
               ' Reference temperature         = ', opts%temperatureReference, ' K.', &
               ' Andersen collision frequency = ', opts%collisionFrequency, ' ps^-1.'
      elseif (opts%qLangevin) then
         WRITE (OUTU,'(2a)') tag, 'Constant temperature w/ OpenMM using Langevin heatbath requested.'
         WRITE (OUTU,'(2a,2x,f8.3,a)') tag, 'Reference heatbath temperature is', opts%temperatureReference, 'K'
         if (opts%frictionSource == 'specified') then
            write (OUTU,'(2a,2x,f8.3,2x,a)') tag, &
                  'Using friction coefficient value of', &
                  opts%frictionCoefficient, 'ps^-1'
         else
            write (OUTU,'(4a,2x,f8.3,2x,a)') tag, &
                  'Friction coefficient not specified, using ', &
                  opts%frictionSource, ' value', opts%frictionCoefficient, 'ps^-1'
         endif
      endif

      if (opts%qPressure) then
         if (opts%qAndersen) then
            write (OUTU,'(2a)') tag, &
                  'CPT dynamics through OpenMM interface requested using Andersen heatbath.'
         elseif (opts%qLangevin) then
            write (OUTU,'(2a)') tag, &
                  'CPT dynamics through OpenMM interface requested using Langevin heatbath.'
         endif
         write (OUTU,'(2a,2x,f6.2,2x,a)') tag, &
               'MC barostat coupled to reference pressure', opts%pressureReference, 'atmospheres.'
         write (OUTU,'(2a,2x,f6.2,2x,a)') tag, &
               'MC barostat using reference temperature', opts%temperatureReference, 'K.'
         write (OUTU,'(2a,2x,i5,2x,a)') tag, &
               'MC barostat volume move attempted every', opts%pressureFrequency, 'timesteps.'
      endif

      if (.not. (opts%qPressure .or. opts%qAndersen .or. opts%qLangevin)) then
         write (OUTU,'(2a,/,a,f7.3,a,i7)') tag, &
               'Leapfrog Verlet requested on OpenMM interface with', &
               '         initial velocities chosen at ',FINALT,' K, using random seed', ISEED
      endif

      if (opts%qVariable) write(OUTU,'(a,8x,a,e12.4)') tag, &
            'Using a variable timestep integrator with tolerance ', opts%variableTS_tol

   end subroutine omm_report_options

   logical function same_dynopts(a, b) result(same)
      type(omm_dynopts_t), intent(in) :: a, b
      same = (a%qLangevin .eqv. b%qLangevin) &
            .and. (a%qVariable .eqv. b%qVariable) &
            .and. (a%qAndersen .eqv. b%qAndersen) &
            .and. (a%qPressure .eqv. b%qPressure) &
            .and. (a%omm_qrexchg .eqv. b%omm_qrexchg) 
      if (.not. same) return
      if (a%qVariable) then
         same = same .and. a%variableTS_tol == b%variableTS_tol
      endif
      if (a%qAndersen .or. a%qLangevin) then
        !ignore the temperature change, because omm_updateT will be true if
        !changed, and the tempearture will be updated using the subroutine new_Temperature 
        !if(.not.a%omm_qrexchg) &
        !     same = same .and. a%temperatureReference == b%temperatureReference
         if (a%qAndersen) same = same .and. a%collisionFrequency == b%collisionFrequency
         if (a%qLangevin) same = same .and. a%frictionCoefficient == b%frictionCoefficient
         if (a%qPressure) then
            same = same .and. all(a%pressureIn3Dimensions == b%pressureIn3Dimensions)
            same = same .and. a%surfaceTension == b%surfaceTension
            same = same .and. a%numberInterfaces == b%numberInterfaces
            same = same .and. a%pressureReference == b%pressureReference
            same = same .and. a%pressureFrequency == b%pressureFrequency
         endif
      endif
   end function same_dynopts

end module omm_dynopts

