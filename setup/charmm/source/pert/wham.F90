SUBROUTINE WHAM0
  !
  ! WHAM READS THE PERTURBATION FILE WRITEN BY EPERT TO UNIT IWHAM
  ! TO GET THE FREE ENERGY PERTURBATIONS
  !
  ! Original version: Benoit Roux, July 1995.
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use comand
  use exfunc
#if KEY_PARALLEL==1
  use parallel, only : mynod  
#endif
  use stream
  use string

  implicit none
  !
  integer,allocatable,dimension(:) :: Ntime
  real(chm_real),allocatable,dimension(:) :: epprtd
  real(chm_real),allocatable,dimension(:) :: lambda
  real(chm_real),allocatable,dimension(:) :: F
  real(chm_real),allocatable,dimension(:) :: F2

  INTEGER maxtime, maxwind
  INTEGER nb_wind
  !
  IF(PRNLEV >= 3) THEN
     write(outu,100)
     write(outu,100)  '----------------------------------'
     write(outu,100)  'Weigthed Histogram Analysis Method'
     write(outu,100)  'Author:  Benoit Roux (1995).'
     write(outu,100)
100  FORMAT(6X,2A)
  ENDIF

  maxtime = GTRMI(COMLYN,COMLEN,'MAXT',-1)
  maxwind = GTRMI(COMLYN,COMLEN,'MAXW',-1)

  if(maxwind < 0)then
     CALL WRNDIE(-1,'<WHAM>','Must specify the number of windows')
  endif

  if(maxtime < 0)then
     CALL WRNDIE(-1,'<WHAM>','Must specify the number of DATA')
  endif
  !
  call chmalloc('wham.src','WHAM0','Ntime',maxwind,intg=Ntime)
  call chmalloc('wham.src','WHAM0','epprtd',maxwind*maxtime,crl=epprtd)
  call chmalloc('wham.src','WHAM0','lambda',maxwind,crl=lambda)
  call chmalloc('wham.src','WHAM0','F',maxwind,crl=F)
  call chmalloc('wham.src','WHAM0','F2',maxwind,crl=F2)
  !
#if KEY_PARALLEL==1
if(mynod == 0) then 
#endif
  call WHAM1(maxwind, maxtime, nb_wind, Ntime, epprtd, &
       Lambda, F, F2)
#if KEY_PARALLEL==1
endif               
#endif
  !
  call chmdealloc('wham.src','WHAM0','Ntime',maxwind,intg=Ntime)
  call chmdealloc('wham.src','WHAM0','epprtd',maxwind*maxtime,crl=epprtd)
  call chmdealloc('wham.src','WHAM0','lambda',maxwind,crl=lambda)
  call chmdealloc('wham.src','WHAM0','F',maxwind,crl=F)
  call chmdealloc('wham.src','WHAM0','F2',maxwind,crl=F2)

  RETURN
END SUBROUTINE WHAM0

SUBROUTINE WHAM1(maxwind, maxtime, nb_wind, Ntime, &
     epprtd, lambda, F, F2)
  !
  use chm_kinds
  use dimens_fcm
  use comand
  use consta
  use ctitla
  use exfunc
  use stream
  use string
  use number
  use parallel
  use param_store, only: set_param

  implicit none
  !
  INTEGER maxwind, maxtime
  INTEGER nb_wind, Ntime(*), Nskip
  real(chm_real)  epprtd(maxwind,maxtime)
  real(chm_real)   lambda(*), F(*), F2(*)
  !
  INTEGER icycle, Ncycle
  real(chm_real)  Tol
  LOGICAL Converged
  INTEGER i, iwind, iunwham
  real(chm_real) kBT, Temp, lambx,eppx
  INTEGER numbx, Ntot1, Ntot2
  INTEGER IOFFSET
  LOGICAL EOF,OK,DONE
  EOF   = .FALSE.
  Ntot1 = 0
  Ntot2 = 0

  ! Boltzmann factor
  Temp = 300.0
  Temp = GTRMF(COMLYN,COMLEN,'TEMP',Temp)
  kBT=kBoltz*Temp

  ! Get wham control parameters
  Tol = GTRMF(COMLYN,COMLEN,'TOL',zero)
  Ncycle = GTRMI(COMLYN,COMLEN,'NSTE',100)
  if(prnlev >= 2) write(outu,98)  &
       'Maximum number of iteration ',ncycle, &
       '    Tolerance ',Tol
98 format(a,i5,a,f6.4)

  IOFFSET = GTRMI(COMLYN,COMLEN,'IOFF',1)
  if(prnlev >= 2) write(outu,98)  &
       'Reference energy level at window ',ioffset
  Nskip   = GTRMI(COMLYN,COMLEN,'NSKI',1)
  if(prnlev >= 2) write(outu,98) 'Nskip ',Nskip

  ! Read the data
  iunwham = GTRMI(COMLYN,COMLEN,'UNIT',-1)
  if(prnlev >= 2) write(outu,98) 'Reading data from unit: ',iunwham
  call rdtitl(titleb,ntitlb,iunwham,0)
  if(prnlev >= 2) write(outu,*)
  !
  ! Add a first window with lambda=0.0
  nb_wind   = 1
  Ntime(1)  = 0
  lambda(1) = zero
1000 read(iunwham,*,end=2000) lambx, eppx

  if(lambx /= lambda(nb_wind))then
     nb_wind = nb_wind + 1
     if(nb_wind > maxwind)then
        if(prnlev >= 2) write(outu,98)  &
             'Number of windows exceeds limit ',maxwind
        call wrndie(-1,'<WHAM>','Try again...')
     endif
     Ntime(nb_wind) = 0
     if(Ntot1 > Ntot2) Ntot2=Ntot1
     Ntot1 = 0
     lambda(nb_wind) = lambx
  endif

  Ntot1 = Ntot1 + 1
  if(Ntime(nb_wind) < maxtime)then
     Ntime(nb_wind) = Ntime(nb_wind) + 1
     epprtd(nb_wind,Ntime(nb_wind)) = eppx
     if(prnlev > 6)then
        write(outu,*) nb_wind,Ntime(nb_wind), &
             lambda(nb_wind),epprtd(nb_wind,Ntime(nb_wind))
     endif
  endif

  goto 1000

2000 nb_wind = nb_wind + 1
  if(nb_wind > maxwind)then
     if(prnlev >= 2) write(outu,98)  &
          'Number of windows exceeds limit ',maxwind
     call wrndie(-1,'<WHAM>','Try again...')
  endif
  ! Add a last window with lambda=1.0
  Ntime(nb_wind) = 0
  lambda(nb_wind) = 1.0

  ! Add supplementary windows with no data point for perturbations
  DONE = .true.
  do while (DONE)
     lambx = gtrmf(comlyn,comlen,'LAMB',MEGA)
     if(lambx == MEGA)then
        DONE = .false.
     else
        nb_wind = nb_wind + 1
        Ntime(nb_wind) = 0
        lambda(nb_wind) = lambx
     endif
  enddo

  if(prnlev >= 2) write(outu,98)   &
       'Total number of windows ', nb_wind
  if(Ntot2 > maxtime)then
     if(prnlev >= 2) write(outu,98)   &
          'Some data points have been left out'
     if(prnlev >= 2) write(outu,99)   &
          'Ntotal ', Ntot2, ' > MaxTime ',maxtime
  else
     if(prnlev >= 2) write(outu,98)  'All data points have been read'
  endif

  if(prnlev >= 2) write(outu,'(a)') '    Window     Ntime    Lambda'
  do i=1,nb_wind
     if(prnlev >= 2) write(outu,'(5x,i5,5x,i5,f10.5)')  &
          i, Ntime(i), Lambda(i)
  enddo

  ! Initialize F2 by reading initial guess or assigning 0
  if(indxa(comlyn, comlen, 'GUES')  >  0)then
     do i=1,nb_wind
    
!        CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE.,'  WHAM> ')
        read(istrm,'(A)') comlyn
         comlen=80   
         call cnvtuc(comlyn,comlen)
!        if(eof)then
!           call ppstrm(ok)
!           if(.not.ok)  return
!        endif
        iwind     = gtrmi(comlyn,comlen,'WIND',1)
        F2(iwind) = gtrmf(comlyn,comlen,'F()',zero)
        lambx     = gtrmf(comlyn,comlen,'LAMB',MEGA)
        numbx     = gtrmi(comlyn,comlen,'NUMB',1)
        if(lambx /= MEGA)then
           if(lambx /= lambda(iwind))then
              if(prnlev >= 2) write(outu,*) lambx, lambda(iwind)
              CALL WRNDIE(-1,'<WHAM>','The values of Lambda do not match')
           endif
        endif
     enddo
  else
     do i=1,nb_wind
        F2(i) = ZERO
     enddo
  endif

  !---- MFC/CLB avoid parallel execution file IO problems by 
  !     not calling WHAM except for master node
#if KEY_PARALLEL==1
  if(mynod == 0 ) then      
#endif

     CALL WHAM(maxwind, maxtime, nb_wind, Ntime, Nskip, epprtd,  &
          Lambda, F, F2, kBT, icycle, Ncycle, Tol, Converged)

#if KEY_PARALLEL==1
  endif                     
#endif

  !  Write out the results
  if(prnlev >= 2) write(outu,*)
  if(Converged)then
     if(prnlev >= 2) write(outu,99)  &
          'WHAM is converged after ',icycle,' iterations'
  else
     if(prnlev >= 2) write(outu,99)  &
          'WHAM is NOT converged after ',icycle,' iterations'
  endif

  do i=1,nb_wind
     if(prnlev >= 2) write(outu,99)  &
          'Window ', i, '   Number_of_data ', Ntime(i), &
          '   Lambda ', lambda(i), '    F() ',F2(i)-F2(IOFFSET)
99   format(a,i5,a,i5,a,f6.3,a,f12.5)
  enddo
  call set_param('WHAMFE', F2(nb_wind)-F2(IOFFSET))

  if(.not.Converged)then
     CALL WRNDIE(-1,'<WHAM>','The iteration is NOT converged')
  endif

  RETURN
END SUBROUTINE WHAM1

SUBROUTINE WHAM(maxwind, maxtime, nb_wind, Ntime, Nskip, epprtd, &
     Lambda, F, F2, kBT, icycle, Ncycle, Tol, Converged)
  !
  !  This subroutine performs a Wheigthed Histogram Analysis Method to
  !  unbias the windows of a free energy calculation.
  !  For reference, see Kumar et al. J. Comp. Chem. 13:1011-1021 (1992).
  !
  ! The free energy constants F(k) are given by
  ! exp(-F(k)/kBT) = Sum_i { Sum_t ( Top_i(t) / Bot_i(t) ) },
  ! where
  ! Top_i(t) = Exp(-W_k[X_i(t)]/kBT)
  ! and
  ! Bot_i(t) = Sum_j  { Ntime(j) * Exp(+F(j)/kBT-W_j[X_i(t)]/kBT) }
  !
  use chm_kinds
  use stream
  use number
  implicit none
  INTEGER maxwind, maxtime
  INTEGER nb_wind, Ntime(*), Nskip
  real(chm_real)  lambda(*)
  real(chm_real)  epprtd(maxwind,maxtime)
  real(chm_real)  arg1, arg2, argmax

  real(chm_real)  F(*), F2(*), kBT
  INTEGER ncycle
  real(chm_real)  Tol
  LOGICAL Converged

  integer icycle
  real(chm_real)  Bot_i, Top_i, diff
  integer i, j, k, t

  ! Iterate the WHAM equation to unbias and recombine the histograms
  if(prnlev >= 2) write(outu,*)
  if(prnlev >= 2) write(outu,'(a)') &
       'Iterate the WHAM equations to get the free energies'


  do icycle=1,ncycle
     if(prnlev > 6) write(outu,*) 'Cycle=',icycle

     ! Calculate the free energy constants F(k)
     do k=1,nb_wind

        F(k)=zero

        do i=1,nb_wind
           do t=1,Ntime(i),Nskip

              ! Calculate Top_i(t)
              arg1   = (-lambda(k)*epprtd(i,t))/kbt

              ! Find maximum argument of exponentials to factor out
              argmax = arg1
              do j=1,nb_wind
                 arg2  = (F2(j)-lambda(j)*epprtd(i,t))/kbt
                 if(arg2 > argmax)then
                    argmax = arg2
                 endif
              enddo
              Top_i = exp(arg1-argmax)

              ! Calculate Bot_i(t)
              Bot_i  = zero
              do j=1,nb_wind
                 arg2  = (F2(j)-lambda(j)*epprtd(i,t))/kbt-argmax
                 Bot_i  = Bot_i  + (Ntime(j)/Nskip) * exp(arg2)
              enddo

              ! Calculate Sum_i Sum_t { Top_i(t) / Bot_i(t)  }

              F(k) = F(k) + Top_i / Bot_i

           enddo
        enddo
     enddo
     !
     ! Test for convergence
     !
     Converged = .True.
     do k=1,nb_wind
        F(k) = -kBT*Log( F(k) )
        if(prnlev > 6) write(outu,*) 'window ',k,' F(k) ',F(k)
        diff = Abs( F2(k) - F(k) )
        F2(k) = F(k) - F(1)
        if(Diff > Tol) Converged = .False.
        if(prnlev > 8) write(outu,*) 'Converged ',Converged
     enddo
     if(prnlev > 6) write(outu,*) 'Converged ',Converged
     if(Converged) RETURN
  enddo
  RETURN
END SUBROUTINE WHAM

