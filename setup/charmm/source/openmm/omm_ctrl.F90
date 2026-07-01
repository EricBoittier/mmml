module omm_ctrl
   use chm_kinds
   use number
   use stream, only: OUTU, PRNLEV
   implicit none

   logical, save, private :: omm_active = .false.

contains

   !> True if the present command includes the OMM option
   !> or was preceded by an OMM ON command, otherwise false.
   logical function omm_requested(comlyn, comlen, caller)
      use string, only: INDXA

      character(len=*), intent(in) :: comlyn
      integer, intent(in) :: comlen
      character(len=*), intent(in) :: caller

      omm_requested = omm_active
      if (INDXA(comlyn, comlen, 'OMM') <= 0) return
#if KEY_OPENMM==1
      call omm_ok(caller)
      omm_requested = .true.
#else /**/
      call wrndie(-1, caller, 'OPENMM code is not compiled.')
#endif
   end function omm_requested

   !> Interprets a top-level command to enable or disable OpenMM.
   !> OMM ON - Sets omm_active. Context may be created later as needed.
   !> OMM OFF - Clears omm_active but retains Context.
   !> OMM CLEAR - Clears omm_active and destroys Context.

   subroutine omm_command(comlyn, comlen)
      use string, only: INDXA, nexti, gtrmf, gtrmi, nexta4, nexta8
      use inbnd, only: lommgb, qgbsw_omm, qgbmv_omm, qmses_omm ! AN EXAMPLE OF OPENMM PLUGIN
#if KEY_OPENMM==1
      use omm_main, only: teardown_openmm, serialize, nbopts, check_nbopts, &
           omm_change_lambda
      use omm_gbsa, only : qgbsa, soluteEPS, solventEPS
      use gbsw, only: qgbsw
      use omm_gbsw, only : qgbsw_import_settings, qphmd_omm, qphmd_initialized, &
            soluteEPSgbsw, solventEPSgbsw
      use gbmv, only: qgbmv
      use omm_gbmv, only : qgbmv_import_settings, &
            soluteEPSgbmv, solventEPSgbmv
      use omm_mses, only : qmses ! AN EXAMPLE OF OPENMM PLUGIN
#if KEY_PHMD==1
      use phmd, only: qphmd
#endif
      use omm_glblopts
      use omm_nbopts
#endif
      implicit none
#if KEY_OPENMM==1
      type(omm_nbopts_t) :: nbcurr
#endif
      character(len=*), intent(inout) :: comlyn
      integer, intent(inout) :: comlen

      integer :: xml_unit
      logical :: omm_state_cmd
      character(len=20) :: blank
      character(len=4) :: wrd, wrd1

      omm_state_cmd = .false.
#if KEY_OPENMM==1
      call init_platformInfo()
      do while (comlen > 0)
         wrd = nexta4(comlyn,comlen)
         qgbsa = .false.

         if (qgbsw) then
            qgbsw_omm = .true.
            qgbsw_import_settings = .false.
            soluteEPSgbsw = GTRMF(comlyn,comlen,'UUEPS',ONE)
            solventEPSgbsw = GTRMF(comlyn,comlen,'VVEPS',80.0)
#if KEY_PHMD==1
            if (qphmd) then
                qphmd_omm = .true.
                qphmd_initialized = .false.
            endif
#endif
         endif

         ! qgbmv: T -> GBMV command is before OMM
         !        F -> GBMV command does not exist in the input file
         if (qgbmv) then
            qgbmv_omm = .true.
            qgbmv_import_settings = .false.
            soluteEPSgbmv = GTRMF(comlyn,comlen,'UUEPS',ONE)
            solventEPSgbmv = GTRMF(comlyn,comlen,'VVEPS',80.0)
         endif

         ! qmses: T -> MSES command is before OMM
         !        F -> MSES command does not exist in the input file
         ! default value: qmses will call qmses_omm
         if (qmses) then           ! AN EXAMPLE OF OPENMM PLUGIN 
            qmses_omm = .true.     ! AN EXAMPLE OF OPENMM PLUGIN
         endif                     ! AN EXAMPLE OF OPENMM PLUGIN
         cmds: select case(wrd)

         case('ON  ') cmds
            omm_state_cmd = .true.
            omm_active = .true.

         case('OFF ') cmds
            omm_state_cmd = .true.
            omm_active = .false.
            qgbsw_omm = .false.
            qgbmv_omm = .false.
            qmses_omm = .false.    ! AN EXAMPLE OF OPENMM PLUGIN
            qnocpu = .false.

         case('CLEA') cmds
            omm_state_cmd = .true.
            omm_active = .false.
            qgbsw_omm = .false.
            qgbmv_omm = .false.
            qmses_omm = .false.    ! AN EXAMPLE OF OPENMM PLUGIN
            qnocpu = .false.
            call teardown_openmm()
            call init_glblopts()

         case('SERI') cmds
            ! default to serializing system
            wrd1 = 'SYST'

            ! otherwise serialize given object: state or integrator
            if (INDXA(comlyn, comlen, 'STAT') > 0) wrd1 = 'STAT'
            if (INDXA(comlyn, comlen, 'INTE') > 0) wrd1 = 'INTE'

            ! get the output unit for the resulting xml text
            xml_unit = GTRMI(comlyn,comlen,'UNIT', 0)
            if(xml_unit .eq. 0) xml_unit = outu

            call serialize(wrd1, xml_unit)

         case('GBSA') cmds
            qgbsa = .true.
            soluteEPS = GTRMF(comlyn,comlen,'UUEPS',ONE)
            solventEPS = GTRMF(comlyn,comlen,'VVEPS',78.85)
            if (INDXA(comlyn, comlen, 'GBOFF') > 0) qgbsa = .false.
            if (INDXA(comlyn, comlen, 'GBON') > 0) qgbsa = .true.
            lommgb = qgbsa

         case('GBSW') cmds
            qgbsw_import_settings = .true.
            soluteEPSgbsw = GTRMF(comlyn,comlen,'UUEPS',ONE)
            solventEPSgbsw = GTRMF(comlyn,comlen,'VVEPS',80.0)
            if (INDXA(comlyn, comlen, 'GBOFF') > 0) qgbsw_import_settings = .false.
            if (INDXA(comlyn, comlen, 'GBON') > 0) qgbsw_import_settings = .true.
            qgbsw_omm = qgbsw_import_settings
            if (qphmd .and. qgbsw_import_settings) then
                qphmd_omm = .true.
                qphmd_initialized = .false.
            endif

         ! OMM GBMV is not often used
         case('GBMV') cmds
            qgbmv_import_settings = .true.
            soluteEPSgbmv = GTRMF(comlyn,comlen,'UUEPS',ONE)
            solventEPSgbmv = GTRMF(comlyn,comlen,'VVEPS',80.0)
            if (INDXA(comlyn, comlen, 'GBOFF') > 0) qgbmv_import_settings = .false.
            if (INDXA(comlyn, comlen, 'GBON') > 0) qgbmv_import_settings = .true.
            qgbmv_omm = qgbmv_import_settings

         ! OMM MSES 
         case('MSES') cmds       ! AN EXAMPLE OF OPENMM PLUGIN
            qmses_omm = .true.   ! AN EXAMPLE OF OPENMM PLUGIN

         case('PLAT') cmds
            wrd1 = nexta4(comlyn,comlen)
            if(wrd1 == 'CUDA') omm_platform = 'CUDA'
            if(wrd1 == 'OPEN') omm_platform = 'OpenCL'
            if(wrd1 == 'REFE') omm_platform = 'Reference'
            if(wrd1 == 'CPU ') omm_platform = 'CPU'

         case('NOCP') cmds
            qnocpu = .true.

         case('PREC') cmds
            wrd1 = nexta4(comlyn,comlen)
            if(wrd1 == 'SING') omm_precision = 'single'
            if(wrd1 == 'MIXE') omm_precision = 'mixed'
            if(wrd1 == 'DOUB') omm_precision = 'double'

         case('DEVI') cmds
            omm_deviceid = nexta8(comlyn,comlen)

         case('TORS') cmds

            if (INDXA(comlyn, comlen, 'TOFF') > 0) qtor_repex = .false.
            if (INDXA(comlyn, comlen, 'TON') > 0) qtor_repex = .true.
            torsion_lambda = GTRMF(comlyn,comlen,'TLAM',-ONE)
            if(qtor_repex) then
               if(torsion_lambda >= zero) then
                  call omm_change_lambda(torsion_lambda)
               else
                  call wrndie(-1, 'omm_command', &
                       'Need to specify lambda >= 0.')
               endif
            endif
            nbopts%qtor_repex = qtor_repex
            nbopts%torsion_lambda = torsion_lambda
            nbcurr = current_nbopts()
            nbopts = nbcurr

         end select cmds
      enddo

      if (omm_state_cmd .and. PRNLEV >= 2) then
         if (omm_active) then
            call omm_ok('<CHARMM>')
            write (OUTU, '(a)') &
                 'Energy and dynamics calculations will use OpenMM.'
         else
            write (OUTU, '(a)') &
                 'Energy and dynamics calculations will not use OpenMM unless requested.'
         endif
      endif

      if ( qgbsw_omm .and. omm_platform == 'Reference') then
          call wrndie(-1,'<CHARMM>', &
          'GBSW not on OpenMM Reference platform: Use GBSW w/o OpenMM')
      endif
      if ( qgbmv_omm ) then
          write (OUTU, '(a)') 'GBMV only works on the OpenMM CUDA platform'
      endif

#else /**/
      call wrndie(-1, '<CHARMM>', 'OPENMM code is not compiled.')
#endif
   end subroutine omm_command

#if KEY_OPENMM==1
   subroutine init_platformInfo()
     use omm_glblopts

     qnocpu = .false.

     if(trim(omm_platform) /= 'CUDA' &
          .and. trim(omm_platform) /= 'OpenCL' &
          .and. trim(omm_platform) /= 'Reference' &
          .and. trim(omm_platform) /= 'CPU' ) &
          omm_platform = ''

     if(trim(omm_precision) /= 'single' &
          .and. trim(omm_precision) /= 'mixed' &
          .and. trim(omm_precision) /= 'double' ) &
          omm_precision = ''

     if(        index(omm_deviceid,'0') == 0 &
          .and. index(omm_deviceid,'1') == 0 &
          .and. index(omm_deviceid,'2') == 0 &
          .and. index(omm_deviceid,'3') == 0 &
          .and. index(omm_deviceid,'4') == 0 &
          .and. index(omm_deviceid,'5') == 0 &
          .and. index(omm_deviceid,'6') == 0 &
          .and. index(omm_deviceid,'7') == 0 &
          .and. index(omm_deviceid,'8') == 0 &
          .and. index(omm_deviceid,'9') == 0 ) &
          omm_deviceid = ''

     return
   end subroutine init_platformInfo
#endif

   !> Check whether unsupported features are turned on
   subroutine omm_ok(caller)
      use replica_ltm, only: qRep
      character(len=*), intent(in) :: caller

      if (qRep) then
         call wrndie(-1, caller, 'Replicas not supported with OpenMM')
      endif
   end subroutine omm_ok

   !> To be called when atoms are added or deleted from the PSF.
   subroutine omm_system_changed()
      use omm_main
#if KEY_OPENMM==1
      call omm_invalidate()
#endif
   end subroutine omm_system_changed
end module omm_ctrl
