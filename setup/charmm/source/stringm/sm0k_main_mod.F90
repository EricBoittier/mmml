module sm0k_main_mod ! string-method-at-0-K
#if KEY_STRINGM == 1

      contains

      SUBROUTINE sm0k_main(COMLYN,COMLEN)
!----------------------------------------------------------------------
! command parser for the 0K string
!----------------------------------------------------------------------
      use sm_config, only : stat_on, stat_freq, repa_on, repa_freq, &
& confcons_on, confcons_freq, chirality_on, chirality_freq, &
& string_noprint
!
      use coord; use coordc
      use dimens_fcm
      use minmiz_module, only: minmiz
      use sm0k
      use stream
      use string
!
      implicit none
!
      CHARACTER(LEN=*) :: COMLYN
      integer :: COMLEN
!
! local variables
      character(len=8) :: keyword
      integer :: i

      integer :: isd, iconj, icgsd, iabnr, inrap
      integer, pointer :: ifixed(:) ! pointer to fixed atoms

!
      character(len=len("SM0K_MAIN>") ),parameter::whoami="SM0K_MAIN>";!macro
!
      keyword=nexta8(comlyn,comlen)
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (( keyword(1:4).eq.'INIT'(1:4) )) then
        call sm0k_init()
        return
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (.not.sm0k_initialized) then
        call sm0k_init()
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (( keyword(1:4).eq.'DONE'(1:4) )) then
        call sm0k_done()
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! interpolate path
      elseif (( keyword(1:4).eq.'INTE'(1:4) )) then
        call sm0k_interpolate(comlyn, comlen)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! conformational consistency
      elseif (( keyword(1:4).eq.'CONF'(1:4) )) then
        call sm0k_confcons()
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! chirality
      elseif (( keyword(1:4).eq.'CHIR'(1:4) )) then
        call sm0k_chirality()
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! reparametrization setup/invocation
      elseif (( keyword(1:4).eq.'REPA'(1:4) )) then
       if (comlen.gt.0) then ! this is an initialization call!
        call sm0k_repa_init(comlyn, comlen)
       else
        i=0; call sm0k_repa(i)! repa routine will reparametrize main coords
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! statistics setup/invocation
      elseif (( keyword(1:4).eq.'STAT'(1:4) )) then
       if (comlen.gt.0) then ! this is an initialization call!
        call sm0k_stat_init(comlyn, comlen)
       else
        i=0; call sm0k_stat(i) ! compute statistics from main coordinates
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'MINI'(1:4) )) then
! string minimization
! SD routine will be called;
! SD is the only minimizer allowed, other minimizers removed below
! setting "repa_on" to true so that
! SD knows to call reparametrization
! other options are let through -- use at your risk!
! delete ABNR, POWE, CONJ, CGSD

       isd=indxa(comlyn, comlen, 'SD')
       iconj=indxa(comlyn, comlen, 'CONJ')
       icgsd=indxa(comlyn, comlen, 'CGSD')
       iabnr=indxa(comlyn, comlen, 'ABNR')
       inrap=indxa(comlyn, comlen, 'NRAP')
       if ((iconj+icgsd+iabnr+inrap).gt.0) then
        call wrndie(0,whoami,trim(' ONLY SD MINIMIZATION IS SUPPORTED. NOTHING DONE'))
        return
       endif
! force SD minimization
       call joinwd(comlyn, mxcmsz, comlen, 'SD ', 3)

!cccccccccccccccccc reparametrization option cccccccccccccccccccccc
       repa_freq=gtrmi(comlyn, comlen, 'REPF', -1)
       if (repa_freq.le.0) then
        repa_on=.false.
        WRITE (sm0k_info,'(/,2A,/,2A,/,2A/)') &
     & whoami,' STRING METHOD ENABLED, BUT', &
     & whoami,' REPARAMETRIZATION FREQUENCY ZERO OR UNSPECIFIED.', &
     & whoami,' REPARAMETRIZATION WILL NOT BE DONE.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
       else
        repa_on=.true.
        WRITE (sm0k_info,'(/,2A,/,2A,I7,A/)') &
     & whoami,' STRING METHOD ENABLED.', &
     & whoami,' WILL REPARAMETRIZE AFTER EVERY ', &
     & repa_freq,' MINIMIZATION ITERATIONS' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
       endif ! repa_freq
!cccccccccccccccc conformational consistency options ccccccccccccccc
       confcons_freq=gtrmi(comlyn, comlen, 'CONFF', -999)
       if (confcons_freq.ne.-999) then ! no message if -999 given
        if (confcons_freq.le.0) then
         confcons_on=.false.
         WRITE (sm0k_info,'(2A)') &
     & whoami,' CONF/CONS FREQUENCY ZERO OR UNSPECIFIED.', &
     & whoami,' CONF/CONS CHECKING WILL NOT BE DONE.';
             if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        else
         confcons_on=.true.
         write(sm0k_info,'(/,2A,I7,A/)') &
     & whoami,' WILL CHECK PATH FOR CONF/CONS AFTER EVERY ', &
     & confcons_freq,' MINIMIZATION ITERATIONS'
         if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';

! currently, confcons checking does not support fixed atoms
         ifixed=>sm0k_fixed_atoms()
         if (size(ifixed).gt.0) then
          call wrndie(0,whoami,trim(' FIXED ATOMS ARE CURRENTLY NOT SUPPORTED WITH CONF/CONS'))
          confcons_on=.false. ; confcons_freq=0
         endif
         deallocate(ifixed)

        endif ! confcons_freq
       endif ! confcons_freq
!cccccccccccccccc chirality optionsccccccccccccccc ccccccccccccccc
       chirality_freq=gtrmi(comlyn, comlen, 'CHIRF', -999)
       if (chirality_freq.ne.-999) then
        if (chirality_freq.le.0) then
         chirality_on=.false.
         WRITE (sm0k_info,'(2A)') &
     & whoami,' CHIRALITY FREQUENCY ZERO OR UNSPECIFIED.', &
     & whoami,' CHIRALITY CHECKING WILL NOT BE DONE.';
             if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        else
         chirality_on=.true.
         write(sm0k_info,'(/,2A,I7,A/)') &
     & whoami,' WILL CHECK PATH FOR CHIRALITY ERRORS AFTER EVERY ', &
     & chirality_freq,' MINIMIZATION ITERATIONS'
         if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';

         ifixed=>sm0k_fixed_atoms()
         if (size(ifixed).gt.0) then
          call wrndie(0,whoami,trim(' CHIRALITY OPTION DOES NOT CURRENTLY SUPPORT FIXED ATOMS'))
          chirality_on=.false. ; chirality_freq=0
         endif
         deallocate(ifixed)

        endif ! chirality_freq
       endif ! chirality_freq
!cccccccccccccccccc statistics output option cccccccccccccccccccccc
       if (repa_on) then ! it makes sense to output string statistics only when reparametrization is enabled
! if you want to follow the unparametrized dynamics, just set maxiter to 0 in the repa setup call
        stat_freq=gtrmi(comlyn, comlen, 'STAF', -1)
        if (stat_freq.le.0) then
        stat_on=.false.
        WRITE (sm0k_info,'(/,2A,/,2A/)') &
     & whoami,' STATISTICS OUTPUT FREQUENCY NOT SPECIFIED.', &
     & whoami,' STATISTICS WILL NOT BE OUTPUT.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        else
         stat_on=.true.
         WRITE (sm0k_info,'(/,2A,I6,A/)') &
     & whoami,' WILL OUTPUT STATISTICS AFTER EVERY ', &
     & stat_freq,' REPARAMETRIZATION ITERATIONS' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        stat_freq=stat_freq*repa_freq
        endif ! stat_freq
       else ! repa_on
        WRITE (sm0k_info,'(/,2A,/,2A/)') &
     & whoami,' STATISTICS OUTPUT REQUIRES REPARAMETRIZATION', &
     & whoami,' (DISABLED). STATISTICS WILL NOT BE OUTPUT.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        stat_on=.false.
       endif
!
       string_noprint=(indxa(comlyn, comlen, 'NOPR').gt.0)
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       call minmiz(comlyn, comlen)
       repa_on=.false. ! turn off reparametrization for regular SD
       stat_on=.false. ! turn off statistics output for regular SD
       string_noprint=.false.

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      else
         write(sm0k_info(1),*) 'UNRECOGNIZED SUBCOMMAND: ', keyword
         call wrndie(0, whoami, trim(sm0k_info(1)))
      endif
!
    end subroutine sm0k_main
#endif /* KEY_STRINGM */
end module sm0k_main_mod
