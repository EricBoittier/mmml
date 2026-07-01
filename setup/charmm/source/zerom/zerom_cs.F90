module zcs
use chm_types
implicit none

 contains

 subroutine cs_alloc(CSI,NSUBS,NCONF,NLINE,FILENM,SUBNM,QDEALL)
! allocates memory for conformer set
! if QDEALL arguement is present and true, deallocates instead
  use ztypes
  use memory
  implicit none 
  type(confset) :: CSI
  integer,intent(in) :: NSUBS,NCONF,NLINE
  character(len=*),intent(in) :: FILENM,SUBNM
  logical,intent(in),optional :: QDEALL
! local
  logical :: LDEALL

  LDEALL = .false.
  if(present(QDEALL)) LDEALL = QDEALL

  if(allocated(CSI%HICNF)) call chmdealloc(FILENM,SUBNM,'CSI%HICNF',NSUBS,intg=CSI%HICNF)
  if(allocated(CSI%LOCNF)) call chmdealloc(FILENM,SUBNM,'CSI%LOCNF',NSUBS,intg=CSI%LOCNF)
  if(allocated(CSI%LODOF)) call chmdealloc(FILENM,SUBNM,'CSI%LODOF',NCONF,intg=CSI%LODOF)
  if(allocated(CSI%HIDOF)) call chmdealloc(FILENM,SUBNM,'CSI%HIDOF',NCONF,intg=CSI%HIDOF)
  if(allocated(CSI%MSTDF)) call chmdealloc(FILENM,SUBNM,'CSI%MSTDF',NLINE,intg=CSI%MSTDF)
  if(allocated(CSI%MSTDV)) call chmdealloc(FILENM,SUBNM,'CSI%MSTDV',NLINE,crl=CSI%MSTDV)
  if(allocated(CSI%ENERG)) call chmdealloc(FILENM,SUBNM,'CSI%ENERG',NCONF,crl=CSI%ENERG)

  if(.not.LDEALL) then
   call chmalloc(FILENM,SUBNM,'CSI%HICNF',NSUBS,intg=CSI%HICNF)
   call chmalloc(FILENM,SUBNM,'CSI%LOCNF',NSUBS,intg=CSI%LOCNF)
   call chmalloc(FILENM,SUBNM,'CSI%LODOF',NCONF,intg=CSI%LODOF)
   call chmalloc(FILENM,SUBNM,'CSI%HIDOF',NCONF,intg=CSI%HIDOF)
   call chmalloc(FILENM,SUBNM,'CSI%MSTDF',NLINE,intg=CSI%MSTDF)
   call chmalloc(FILENM,SUBNM,'CSI%MSTDV',NLINE,crl=CSI%MSTDV)
   call chmalloc(FILENM,SUBNM,'CSI%ENERG',NCONF,crl=CSI%ENERG)
  endif

  end subroutine cs_alloc

 subroutine cs_init(CSI,WHICH)
  use ztypes
  implicit none
  type(CONFSET),intent(inout) :: CSI
  character(len=*),optional,intent(in) :: WHICH
!local
  logical :: QINT,QREAL

  QINT = .true.
  QREAL = .true. 
  if(present(WHICH)) then
   QINT = .false.
   QREAL = .false.
   if(WHICH.eq.'INTEGER') then
     QINT = .true.
   else if(WHICH.eq.'REAL') then
     QREAL = .true. 
   endif
  endif
  
  if(QINT) then
   CSI%HICNF = 0
   CSI%LOCNF = 0
   CSI%LODOF = 0
   CSI%HIDOF = 0
   CSI%MSTDF = 0
  endif
  if(QREAL) then
   CSI%MSTDV = 0
   CSI%ENERG = 0 
  endif
 end subroutine cs_init
!----------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------
 subroutine cs_write(CSI,NSUBS,WUNIT)
 use ztypes,only: CONFSET
  
 type(CONFSET),intent(in) :: CSI
 integer,intent(in) :: NSUBS,WUNIT
 integer :: SS,CNF,JJ
 
10  FORMAT(I14,I14,I14,F14.7,F14.7)
  
 do SS = 1,NSUBS
   do CNF = CSI%LOCNF(SS),CSI%HICNF(SS)
      do JJ = CSI%LODOF(CNF),CSI%HIDOF(CNF)
         WRITE(WUNIT,10) SS,CNF,CSI%MSTDF(JJ),CSI%MSTDV(JJ),CSI%ENERG(CNF)
!         WRITE(MYNODP+300,10) SS,CNF,CSI%MSTDF(JJ),CSI%MSTDV(JJ),CSI%ENERG(CNF)
      enddo
   enddo
 enddo

 end subroutine cs_write
!----------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------

 subroutine cs_trim(CSI,NSUBS,ECUT,VCUT,LVCUT,ECUTAR,CSO)
! trims the conformer set by energy cutoff; ECUT is the fixed cutoff; VCUT is width
! of band above minimum.  RETURNS
 use ztypes,only: CONFSET
#if KEY_PARALLEL==1
 use parallel,only: MYNODP 
#endif
 use nbndcc_utilb,only: parstoperr
 use memory
 implicit none

 type(CONFSET),intent(inout) :: CSI
 type(CONFSET),intent(out),optional :: CSO
 integer,intent(in) :: NSUBS
 real(chm_real),optional :: ECUT,VCUT,LVCUT
 real(chm_real),dimension(:),optional :: ECUTAR
! local
 type(CONFSET),target :: CST
 integer :: NLINE,NLOC,SS,CNF,NCNF,OLDCNF,TOTSIZ,JJ,SIZCNF
 real(chm_real) :: ENER,CUTOFF,MINIME,GCUTOFF
 logical :: QECUT=.false.,QVCUT=.false.,QLVCUT=.false.,QECUTAR=.false.
 integer :: NCONFSS,NLINELOC 
 real(chm_real),dimension(:),allocatable :: LOCMIN

 QECUT = .false.   !flg for global energy cutoff
 if(present(ECUT)) QECUT = .true.
 QVCUT = .false.   !flg for "band" cutoff above global minimum
 if(present(VCUT)) QVCUT = .true.
 QLVCUT = .false.   !flg for "band" cutoff applied to the local minimum for each subspace
 if(present(LVCUT)) QLVCUT = .true.
 QECUTAR = .false.  !flg for local energy cutoff for each subspace
 if(present(ECUTAR)) QECUTAR = .true.

!  WRITE(6,*) 'MYNODP ',MYNODP,' QECUT ',QECUT,' QVCUT ',QVCUT,' QLVCUT ',QLVCUT,' QECUTAR ',QECUTAR

 if((.not.QECUT.and..not.QVCUT.and..not.QLVCUT.and..not.QECUTAR)) then
  call parstoperr('<confset_trimen>','NO ENERGY CUTOFF SPECIFIED') 
 endif

 if(QLVCUT) then
  if(allocated(LOCMIN)) deallocate(LOCMIN)
  call chmalloc('zerom_cs','cs_trim','LOCMIN',NSUBS,crl=LOCMIN)
 endif
! write(6,*) 'top: size of CSI%LOCNF ',size(CSI%LOCNF), 'size of CSI%HICNF ',size(CSI%HICNF) 
 if((QVCUT).or.(QLVCUT)) then
  MINIME = 9.999D99
  do SS = 1,NSUBS
   if(QLVCUT) LOCMIN(SS) = 9.999D99 
   do CNF = CSI%LOCNF(SS),CSI%HICNF(SS)
     ENER = CSI%ENERG(CNF) 
     if(ENER.LT.MINIME) MINIME = ENER
     if(QLVCUT) then
      if(ENER.LT.LOCMIN(SS)) LOCMIN(SS) = ENER
     endif
   enddo
  enddo
 endif

!global cutoffs
 GCUTOFF = 9.9999D99
 if(QECUT) GCUTOFF = min(GCUTOFF,ECUT)
 if(QVCUT) GCUTOFF = min(GCUTOFF,VCUT+MINIME)

#if KEY_PARALLEL==1
    if(MYNODP.eq.1) then 
#endif
     WRITE(6,*) 'QECUT is ',QECUT,' QVCUT is ',QVCUT
     if(QECUT) WRITE(6,*) 'ECUT ',ECUT,' MINIME ',MINIME,' GCUTOFF ',GCUTOFF
     if(QVCUT) WRITE(6,*) 'VCUT ',VCUT,' MINIME ',MINIME,' GCUTOFF ',GCUTOFF
#if KEY_PARALLEL==1
    endif
#endif
     
!  WRITE(6,*) 'QECUT ',QECUT,' QVCUT ',QVCUT,' QLVCUT ',QLVCUT,' QECUTAR ',QECUTAR
 NCNF = 0
!  WRITE(6,*) 'serial: SUBSPACEs total ',NSUBS
 do SS = 1,NSUBS
!  write(6,*) 'size of CSI%LOCNF ',size(CSI%LOCNF), 'size of CSI%HICNF ',size(CSI%HICNF) 
!  WRITE(6,*) 'serial: SUBSPACE ',SS,' LOCONF ',CSI%LOCNF(SS),' HICONF ',CSI%HICNF(SS)
  CUTOFF = GCUTOFF
  if(QECUTAR) CUTOFF = min(CUTOFF,ECUTAR(SS))
  if(QLVCUT) CUTOFF = min(CUTOFF,LOCMIN(SS)+LVCUT)
!  WRITE(6,*) 'SS ',SS,' ECUTAR ',ECUTAR(SS),' LOCMIN ',LOCMIN(SS),' CUTOFF ',CUTOFF
!  if(MYNODP.eq.1) then !PARALLEL
!   if(QLVCUT) then
!    WRITE(6,*) ' LVCUT ',LVCUT,' CUTOFF ',CUTOFF
!   endif
!  endif !PARALLEL
  NCONFSS = 0
  do CNF = CSI%LOCNF(SS),CSI%HICNF(SS)
!     WRITE(6,*) 'MYNODP ',MYNODP,' SUBSPACE ',SS,' LOCONF ',CSI%LOCNF(SS),' HICONF ',CSI%HICNF(SS)
     ENER = CSI%ENERG(CNF)
     if(ENER.LT.CUTOFF) then
!       WRITE(200+MYNODP,*) 'SUBS ',SS,' CNF ',CNF,' ENER ',ENER
       NCNF = NCNF + 1
       NCONFSS = NCONFSS + 1
     endif
  enddo
#if KEY_PARALLEL==1
  if(MYNODP.eq.1) then 
#endif
   if(NCONFSS.eq.0) then
    WRITE(6,*) 'WARNING: SUBSPACE ',SS,' CONTAINS NO CONFORMERS WITHIN ENERGY CUTOFFS '  
   endif
#if KEY_PARALLEL==1
  endif 
#endif
 enddo
 if(NCNF.eq.0) then
   call parstoperr('<cs_trim>','NO CONFORMERS FALL WITHIN ENERGY CUTOFFS')
 endif

 SIZCNF = CSI%HIDOF(1) - CSI%LODOF(1) + 1 !should always work, since first conf is complete
 
 TOTSIZ = SIZCNF*NCNF

! write(6,*) 'NSUBS ',NSUBS,' NCNF ',NCNF,' TOTSIZ ',TOTSIZ
 call cs_alloc(CST,NSUBS,NCNF,TOTSIZ,'zerom_util','cs_trim')
 call cs_init(CST)
 
 NCNF = 0
 NLINE = 0
 do SS = 1,NSUBS
   CUTOFF = GCUTOFF
   if(QECUTAR) CUTOFF = min(CUTOFF,ECUTAR(SS))
   if(QLVCUT) CUTOFF = min(CUTOFF,LOCMIN(SS)+LVCUT)
   NCONFSS = 0
   do OLDCNF = CSI%LOCNF(SS),CSI%HICNF(SS)
     ENER = CSI%ENERG(OLDCNF)
     if(ENER.LT.CUTOFF) then
       NCNF = NCNF + 1
       NCONFSS = NCONFSS + 1
       CST%ENERG(NCNF) = ENER
       NLINELOC = 0
       do JJ = CSI%LODOF(OLDCNF),CSI%HIDOF(OLDCNF)
        NLINE = NLINE + 1
        NLINELOC = NLINELOC + 1
        CST%MSTDF(NLINE) = CSI%MSTDF(JJ)
        CST%MSTDV(NLINE) = CSI%MSTDV(JJ)
       enddo
       CST%HIDOF(NCNF) = NLINE
       CST%LODOF(NCNF) = NLINE - NLINELOC + 1  !NLINELOC should always be size of conf
       if(NLINELOC.ne.SIZCNF) then
        WRITE(6,*) 'ERROR:  conformer not of correct size'
        WRITE(6,*) 'NLINELOC ',NLINELOC,' SIZCNF ',SIZCNF
        STOP
       endif
     endif
   enddo
   CST%LOCNF(SS) = NCNF - NCONFSS + 1
   CST%HICNF(SS) = NCNF
 enddo
! call cs_write(CST,NSUBS,350)

 if(present(CSO)) then
  call cs_copy(CST,CSO)
 else
  call cs_copy(CST,CSI)
 endif

! write(6,*) 'present(CSO)3 is ',present(CSO)
 end subroutine cs_trim

!----------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------

 subroutine cs_copy(CSI,CSO)
 use ztypes,only: CONFSET
 implicit none
 type(CONFSET),intent(in) :: CSI
 type(CONFSET),intent(out) :: CSO
!local
 integer :: NSUB,NCNF,NLIN

 NSUB = size(CSI%LOCNF)
 NCNF = size(CSI%LODOF)
 NLIN = size(CSI%MSTDF)

 call cs_alloc(CSO,NSUB,NCNF,NLIN,'zerom_util','cs_copy')

 CSO%LODOF=CSI%LODOF
 CSO%HIDOF=CSI%HIDOF
 CSO%LOCNF=CSI%LOCNF
 CSO%HICNF=CSI%HICNF
 CSO%MSTDF=CSI%MSTDF
 CSO%MSTDV=CSI%MSTDV
 CSO%ENERG=CSI%ENERG

 end subroutine cs_copy
 
!----------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------
 end module zcs
