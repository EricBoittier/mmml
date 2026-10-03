!---------------------------------------------------------------------
SUBROUTINE EABF_CYCLE(ICALL)
!---------------------------------------------------------------------
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
use energym
use corsubs
implicit none
integer :: icall,i,ii,k,l,istat
real(chm_real) :: v,prev_value,CMXC,CMYC,xxx,yyy,zzz

    if(ICALL .EQ. 1)then
    if(mynod==0)then
      if(MDSTEP .EQ. 0 .OR. eabf_rst .EQ. 1)then
         !Open eABF output files     
         open(unit=777,file='eabf_dynamics.out')
         !open(unit=323,file='eabf_fl_hist.dat')
         !open(unit=423,file='eabf_me_fl_hist.dat')
         !Calculate CV Values
         rmsd_ref_count=0
         do i=0,eabf_cv_count-1
            call calc_xi(i)
         enddo
         !Propagate Particles & Apply Forces
         rmsd_ref_count=0
         call IMPULSE_INTEGRATOR()
         !Write eABF Dynamics Information
         call WRITE_EABF()
         eabf_rst=0
      endif
      if(MDSTEP .GT. 0)then         
         !Calculate CV Values
         rmsd_ref_count=0
         do i=0,eabf_cv_count-1
            call calc_xi(i)
         enddo
         !Propagate Particles & Apply Forces
         rmsd_ref_count=0
         call IMPULSE_INTEGRATOR()
         !Write eABF Dynamics Information
         call WRITE_EABF()
         !if(mod(MDSTEP+1,restart_frq) .EQ. 0)then
         !   !Backup restart in simulation directory using local bash script
         !    call system("./backup_restart.sh")
         !endif
         if(mod(MDSTEP,restart_frq) .EQ. 0)then
            call WRITE_EABF_RESTART()
         endif
      endif
    endif
    endif
END SUBROUTINE
