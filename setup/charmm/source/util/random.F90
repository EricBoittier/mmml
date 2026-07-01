module rndnum
  use chm_kinds
  implicit none
  !
  !     RNDNUM.FCM
  !      declarations associated with ?RAND substitution
  !             rm venable  <*>  7 dec 1989
  !
  !  IRNDSD   separate random seed; default 1380662
  !  IRNTYP   distribution type; default = 0 = uniform, 1 = gaussian
  !  RNDSGMA  std dev (sigma) for Gaussian; no default, req'd arg
  !  RNDSCAL  scale factor; default 1.0
  !  RNDOFFS  offset; mean for Gaussian; default 0.0
  !  QOLDRNG  Use "old" random number generator instead of CLCG
  !
  INTEGER   IRNDSD,IRNTYP
  real(chm_real)    RNSGMA,RNSCAL,RNOFFS
  LOGICAL   QOLDRNG, QRANDPAR
  !   varibles for the new random generators
  ! some common variables:
  !
  !  rngseeds           storage for array of seeds
  !                     (we don't know how many, so it is allocatable)
  !
  !  The following two have to be initialized in the:
  !  where we parse the command line and not in iniall routines!!!
  !
  !  qoldrandom         flag to assign initial defaults
  !            .true.   assign oldrandom as default because of
  !                     charmm command line flag: -oldrandom
  !            .false.  set clcg with seeds assigned from system time
  !                     or any other than oldrandom
  !  qbrokenclcg        flag to assign initial defaults
  !            .true.   assign broken CLCG as default because of
  !                     charmm command line flag: -brokenclcg
  !            .false.  set clcg with seeds assigned from system time
  !                     or any other than old broken clcg
  !
  !  rngchoice    0     old random number generator
  !  rngchoice    1     the default fixed RNG (clcg)
  !  rngchoice    2     get RNG from new fortran standard:
  !                     call random_number()
  !  rngchoice    3     user programmed RNG in charmm/usersb.src
  !
  !  rngdistrchoice 0   uniform
  !                 1   gaussian
  !
  !  rngseries   1-100  which CLCG series we want to use (default 1, or is it 2?)
  !
  !  nrand              number of seeds (1,4,8,...)
  !
  logical qoldrandom,qbrokenclcg,qrngcos,qrngsin

  integer,allocatable,dimension(:) :: rngseeds
  integer :: nrand, rngseries
  integer :: rngchoice,seedchoice,rngdistrchoice

#if KEY_LIBRARY == 1
  logical :: qapi_seed_set
#endif
  
contains

  subroutine rndnum_iniall
    use number
    use memory
    character(len=12) :: filnam="random.src  ", routine="rndnum_iniall"
    integer iseed,mynod
    integer(chm_int8) :: count,rate,maxcount
    integer(chm_int4) :: count4,rate4,maxcount4

#if KEY_LIBRARY == 1
    qapi_seed_set = .false. ! fortran std random_number
    rngchoice = 2
#else
    rngchoice = 1 ! CLCG default now  
#endif
    
    irndsd=1380662
    irntyp=0
    rnscal = ONE
    rnoffs = ZERO
    rnsgma = ONE
    qoldrng= .false.  !yw 05-aug-2008
    if (qoldrandom) qoldrng=.true.

    rngdistrchoice = 0 ! uniform distribution
    nrand = 4
    rngseries = 1      ! which series we take from CLCG
    qrandpar = .false. ! false by default because it hurts the parallel performance
   return
  end subroutine rndnum_iniall

end module rndnum


module clcg_mod
  use chm_kinds
  implicit none
  !------------------------------------------------------------------CC
  !  Header file for RNG: CLCG (source/util/clcg.src) X. Qian 12/00   C
  !  The variables are:                                               C
  !      Maxgen     512 (the same as MAXNODE in parallel.fcm)         C
  !                 maximum number of independent streams of random   C
  !                 number sequences  (This value can be increased    C
  !                 as needed.)                                       C
  !      IniSD      1                                                 C
  !      LstSD      2                                                 C
  !      NewSD      3                                                 C
  !                 These three option flags are used to initialize   C
  !                 the RNG with different initial conditions.        C
  !                 By default, initial seeds lcgIg{(i=1, ..., 4),g}  C
  !                 and last seeds  lcgLg{(i=1, ..., 4),g}            C
  !                 (for g=1...Maxgen) are set to the original        C
  !                 seeds, and previous seeds, respectively.          C
  !                 Calls to IniGen(g,stype)                          C
  !                 (where stype is IniSD or LstSD or NewSD)          C
  !                 can reset the seeds for stream g to the initial   C
  !                 values (stype = IniSD), previous values           C
  !                 (stype = LstSD) or new values (stype = NewSD).    C
  !      lcgIg      Initial seed values, dimension 4 by Maxgen        C
  !                 for the four LCGs.                                C
  !      lcgLg      Last seed values, dimension 4 by Maxgen           C
  !      lcgCg      Current seed values, dimension 4 by Maxgen        C
  !                                                                   C
  !     lcgmul(4)   The multipliers for the 4 LCGs                    C
  !     lcgmod(4)   The moduli for the 4 LCGs                         C
  !                 THE MULTIPLIER AND MODULI VALUES                  C
  !                     MUST NOT BE CHANGED                           C
  !                                                                   C
  !    lcgaw(4)     lcgmul{j}^{2^w}      w=41, j=1, ..., 4.           C
  !    lcgavw(4)    lcgmul{j}^{2^(v+w)}, v=31, j=1, ..., 4.           C
  !                 These two arrays are used to generate initial     C
  !                 seeds for the specified Maxgen number of the      C
  !                 streams  with the  initial seeds given by user    C
  !                 or from default values.                           C
  !                                                                   C

  integer,parameter ::Maxgen = 512, IniSD = 1, LstSD = 2, NewSD = 3
  integer lcgmul(4),lcgmod(4),lcgaw(4),lcgavw(4)
  !MH13: Maybe this should be here: save ???
  !integer,save :: lcgIg(4,Maxgen),lcgLg(4,Maxgen),lcgCg(4,Maxgen)
  integer :: lcgIg(4,Maxgen),lcgLg(4,Maxgen),lcgCg(4,Maxgen)
  data lcgmul/45991,207707,138556,49689/
  data lcgmod/2147483647,2147483543,2147483423,2147483323/ 


  !=======================================================================
contains
  
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !               Combined Linear Congruential Generator (CLCG)        C
  !  Adapted from Pierre L'Ecuyer & Terry H Andres' C version code.    C
  !  References                                                        C
  ! [1] P. L'Ecuyer and T. H. Andres,                                  C
  !     ``A Random Number Generator Based on the Combination           C
  !     of Four LCGs'', Mathematics and Computers in Simulation,       C
  !     44 (1997), 99--107.                                            C
  ! [2] http://www.iro.umontreal.ca/~lecuyer/                          C
  !                                                                    C
  ! For further information, please contact                            C
  !              Tamar Schlick                                         C  
  !              schlick@nyu.edu                                       C
  ! Converted to FORTRAN by                                            C
  !               Xiaoliang Qian  10/7/99                              C
  !                                                                    C
  !     Fixed the usage of this code by                                C
  !               Milan Hodoscek  6/6/04                               C
  !                                                                    C
  !     NOTE: The code needs to be improved for parallel               C
  !                                                                    C
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !  This pseudorandom number generator combines 4 linear              C
  !  congruential generators (LCGs) to get the long period             C
  !  of about 2^121 for the resulting sequence. This sequence has      C
  !  passed many statistical tests of `randomness'.                    C
  !  Essentially, the four LCGs defined as                             C
  !                                                                    C
  !      X{j,n} = lcgmul{j}X{j,n-1} mod lcgmod{j}                (1)   C
  !                            j=1,2,3,4                               C
  !                            n=1,2,3, ...                            C
  !                                                                    C
  !  with lcgmod{1}= 2147483647, lcgmod{2}=2147483543,                 C
  !       lcgmod{3}=2147483423, lcgmod{4}=2147483323                   C
  !  and lcgmul{1}=45991, lcgmul{2}=207707, lcgmul{3}=138556,          C
  !  lcgmul{4}=49689.                                                  C
  !                                                                    C
  !  The construct                                                     C
  !      Z{n} = (Sum [(-1)^{j+1}*X{j,n}/lcgmod{j}]) mod 1   (2)        C
  !                                                                    C
  ! for n=1,2,3, ... is then a uniformly distributed random sequence   C
  ! in (0,1). It can be proved that the LCG corresponding to the       C
  ! combined generator has modulus, multiplier, and period length of   C
  !      21267641435849934371830464348413044909,                       C
  !      5494569482908719143153333426731027229,                        C
  !      (2^{31}-2)(2^{31}-106)(2^{31}-226)(2^{31}-326) ~ 2^{121},     C
  !  respectively.                                                     C
  !                                                                    C
  !  The default initial seed is the vector {11111111, 22222222,       C
  !  33333333, 44444444} and can be changed by calling SetIniSD        C
  !  after calling CLCGInit.                                           C
  !                                                                    C
  !  This RNG can be used under parallel conditions to give            C 
  !  independent random number sequence when each processor            C
  !  calls with different stream number g (e.g., RANDOM(g)).           C
  !                                                                    C
  !  To use these RNG routines, the user should proceed as follows:    C
  !                                                                    C
  !  1. Call routine CLCGInit() to initialize all Maxgen (100) streams C
  !     using four default initial seeds.                              C
  !  2. [Optional] Call SetiniSD(sd) with desired seed array           C
  !     (4 values) to override the default values specified in Init(). C
  !  3. Call function RANDOM(k), where k is an integer from 1 to 100   C
  !     specifying the stream number. For parallel codes, k can be set C
  !     to a processor id number.                                      C
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC 
  subroutine CLCGInit (ISEED)
    !------------------------------------------------------------------------C
    !   Initialize the RNG with seed values in vector sd of dimension 4.     C
    !   Each such initial vector generates one stream of random variates     C
    !   combining the 4 LCG sequences with resulting sequence length         C
    !   2^{121}.                                                             C
    !   The lcgaw and lcgavw arrays of dimension 4 have default values       C
    !       lcgaw{j} = lcgmul{j}^{2^31} mod lcgmod{j}                        C
    !   and lcgavw{j} = lcgmul{j}^{2^41} mod lcgmod{j},                      C
    !   for j=1, ..., 4 corresponding to the 4 LCGs.                         C
    ! Converted to FORTRAN by                                                C
    !               Xiaoliang Qian  10/7/99                                  C
    
  use chm_kinds
  use rndnum
!  use exfunc
    implicit none
    !
    INTEGER ISEED
    !
    integer sd(4)
    integer v,w,j,i
    parameter (v = 31, w = 41)
    data sd/11111111,22222222,33333333,44444444/
    !
    !     In many places CHARMM controls the initialization
    !     of SEED. This is one way to do it, because we only get
    !     one number from the user.
    !
    if (qbrokenclcg.or.qoldrandom) then
       sd(1)=iseed
    else
       sd(1:4) = rngseeds(1:4)
    endif
    !
    do j = 1,4
       lcgaw(j) = lcgmul(j)
       do  i = 1,w
          lcgaw(j) = MulMod (lcgaw(j),lcgaw(j),lcgmod(j))
       enddo
       lcgavw(j) = lcgaw(j)
       do i = 1,v
          lcgavw(j) = MulMod (lcgavw(j),lcgavw(j),lcgmod(j))
       enddo
    enddo

    call SetiniSD (sd)

  end subroutine CLCGInit
  
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  
  subroutine SetiniSD (s)
    !------------------------------------------------------------------------C
    !  Set initial seed values for all 100 (= Maxgen, defined in clcg.f90)   C
    !  streams using the initial seeds from the first stream.                C
    ! Converted to FORTRAN by                                                C
    !               Xiaoliang Qian  10/7/99                                  C
    
  use chm_kinds
!  use exfunc
    implicit none
    !
    integer g,s(4),j 
    
    do j = 1,4
       lcgIg(j,1) = s(j)
    enddo
    call IniGen (1,IniSD)
    do g = 2, Maxgen
       do  j = 1,4
          lcgIg(j,g) = MulMod (lcgavw(j),lcgIg(j,g-1),lcgmod(j))
       enddo
       call IniGen (g,IniSD)
    enddo
  end subroutine SetiniSD
  
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  
  function RANDOM (g,label) result(random_rtn)
    !------------------------------------------------------------------------C
    !   Return a double precision uniformly distributed random number in     C
    !   (0,1) from the gth stream and reset the current seed Cg accordingly  C
    !   (i.e., using one of the 100 initial seed vectors generated in the    C
    !   SetiniSD routine).                                                   C
    ! Converted to FORTRAN by                                                C
    !               Xiaoliang Qian  10/7/99                                  C
    !
    !
  use chm_kinds
  use number
!  use parallel
  use rndnum
    implicit none
    
    integer g,k,s,j
    real(chm_real) u(4), random_rtn,rng1
    integer  dv(4),mv(4)
    character(len=*),optional :: label
    data dv/46693,10339,15499,43218/ 
    data mv/25884, 870,3979,24121/ 
    data u/4.65661287524579692d-10,-4.65661310075985993d-10, &
         4.65661336096842131d-10,-4.65661357780891134d-10/
    
    !
    if (qoldrng.or.(rngchoice==0)) then
       random_rtn = oldrandom(g)
       return
    endif

    ! system provided RNG
    if(rngchoice == 2) then
       call random_number(rng1)
       random_rtn = rng1
       return
    endif

    if (g  <=  1)  g = 1
    g= mod(g-1,Maxgen) + 1

    if(rngchoice == 1) g = 1  ! Always use stream number 1 in this case!

    RANDOM_rtn = zero

    do j = 1,4
       s = lcgCg(j,g)
       k = s/dv(j)
       s = lcgmul(j) * (s - k * dv(j)) - k * mv(j)
       if (s  <  0) s = s + lcgmod(j)
       lcgCg(j,g) = s
       RANDOM_rtn = RANDOM_RTN + u(j) * s
       if (RANDOM_RTN  <  zero)  RANDOM_RTN = RANDOM_RTN + one
       if (RANDOM_RTN  >=  one)   RANDOM_RTN = RANDOM_RTN - one
    enddo
!    if(present(label)) then
!       write(150+mynod,'(f20.15,3x,a)')random_rtn,label
!    else
!       write(150+mynod,'(f20.15)')random_rtn
!    endif
    return
  end function RANDOM
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !
  subroutine  SetSeed (g,s)
    !------------------------------------------------------------------------C
    !  This optional routine uses the input seed value s for stream g        C
    !  instead of the default settings (routine SetiniSD).                   C
    ! Converted to FORTRAN by                                                C
    !               Xiaoliang Qian  10/7/99                                  C
    
    
    integer g,s(4),j
    
    if (g  <=  1)  g = 1
    g= mod(g-1,Maxgen) + 1
    
    do j = 1,4          
       lcgIg(j,g) = s(j)
    enddo
    call IniGen (g,IniSD)               
  end subroutine SetSeed
  
  subroutine  GetSeed (g,s)
    !------------------------------------------------------------------------C
    !  This optional routine returns current seed value s for stream g       C
    ! Converted to FORTRAN by                                                C
    !               Xiaoliang Qian  10/7/99                                  C
    
    
    integer g,s(4),j
    
    if (g  <=  1)  g = 1
    g= mod(g-1,Maxgen) + 1
    
    do  j = 1,4          
       s(j)= lcgCg(j,g)
    enddo
    return
  end subroutine GetSeed
  
  
  
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  
  subroutine IniGen (g,stype)
    !------------------------------------------------------------------------
    !  This optional routine resets the gth stream so that the initial seed
    !  is either the original initial seed (if stype = IniSD) or the last
    !  seed (if stype = 3).
    ! Converted to FORTRAN by
    !               Xiaoliang Qian  10/7/99
    
    integer g,stype,j,ig
    
    ig=g
    if (ig  <=  1)  ig = 1
    ig= mod(ig-1,Maxgen) + 1
    do j = 1,4
       if (stype == IniSD) then
          lcgLg(j,ig) = lcgIg(j,ig)
       else
          if (stype == NewSD)  &
               lcgLg(j,ig) = MulMod (lcgaw(j),lcgLg(j,ig),lcgmod(j)) 
       endif
       lcgCg(j,ig) = lcgLg(j,ig)
    enddo
    return
  end subroutine IniGen
  
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  
  
  integer function MulMod (s,t,M)
    !--------------------------------------------------------------------C
    !  Return s*t mod M. All numbers out of range are truncated          C
    !  before the mod operation to avoid overflow. See Park & Miller,    C
    !  Comm. ACM 31, 1192 (1988) for this multiplication procedure.      C
    ! Converted to FORTRAN by                                            C
    !               Xiaoliang Qian  10/7/99                              C

    integer s,t,M,H
    parameter (H = 32768) 
    integer   S0,S1,q,qh,rh,k 

    if (s  <  0) s = s + M
    if (t  <  0) t = t + M
    if (s  <  H) then
       S0 = s
       MulMod = 0
    else
       S1 = s / H
       S0 = s - H * S1
       qh = M / H
       rh = M - H * qh

       if (S1  >=  H) then
          S1 = S1 - H
          k = t / qh
          MulMod = H * (t - k * qh) - k * rh

10        if (MulMod  <  0) then
             MulMod = MulMod + M      
             goto 10
          endif

       else
          MulMod = 0
       endif

       if (S1  /=  0) then
          q = M / S1
          k = t / q
          MulMod = MulMod - k * (M - S1 * q)
          if (MulMod  >  0) MulMod = MulMod - M
          MulMod = MulMod + S1 * (t - k * q) 

20        if (MulMod  <  0) then
             MulMod = MulMod + M      
             goto 20
          endif
       endif

       k = MulMod / qh
       MulMod = H * (MulMod - k * qh) - k * rh

30     if (MulMod  <  0) then
          MulMod = MulMod + M      
          goto 30
       endif
    endif

    if (S0  /=  0) then
       Q = M / S0
       k = t / q                         
       MulMod = MulMod - k * (M - S0 * q)
       if (MulMod  >  0) MulMod = MulMod - M
       MulMod = MulMod + S0 * (t - k * q) 
40     if (MulMod  <  0)  then
          MulMod = MulMod + M      
          goto 40
       endif
    endif
    return
  end function MulMod


  FUNCTION OLDRANDOM(ISEED) result(random)
    !-----------------------------------------------------------------------
    !     RANDOM NUMBER GENERATOR: UNIFORM DISTRIBUTION (0,1)
    !     ISEED: SEED FOR GENERATOR. ON THE FIRST CALL THIS HAS TO
    !     HAVE A VALUE IN THE EXCLUSIVE RANGE (1, 2147483647)
    !     AND WILL BE REPLACED BY A NEW VALUE TO BE USED IN
    !     FOLLOWING CALL.
    !
    !     REF: Lewis, P.A.W., Goodman, A.S. & Miller, J.M. (1969)
    !     "Pseudo-random number generator for the System/360", IBM
    !     Systems Journal 8, 136.
    !
    !     This is a "high-quality" machine independent generator.
    !     INTEGERS are supposed to be 32 bits or more.
    !     The same algorithm is used as the basic IMSL generator.
    !
    !      Author: Lennart Nilsson
    !
    !     As the CLCG becomes the default part of the random number code,
    !     this function is renamed to OLDRANDOM. yw 08-Aug-2008
    !
    use chm_kinds
    implicit none
    INTEGER ISEED
    real(chm_real) DSEED,DIVIS,DENOM,MULTIP,random
    DATA  DIVIS/2147483647.D0/
    DATA  DENOM /2147483711.D0/
    DATA  MULTIP/16807.D0/
    !
    IF(ISEED <= 1) ISEED=314159
    DSEED=MULTIP*ISEED
    DSEED=MOD(DSEED,DIVIS)
    RANDOM=DSEED/DENOM
    ISEED=DSEED
    !
    RETURN
  END FUNCTION OLDRANDOM

  !
  !     <*> the following code added by rm venable, FDA Biophysics Lab <*>
  !         Modified by SRD 1/27/91.
  !
  FUNCTION BMGAUS(RF,IG) result(bmgaus_rtn)
    !-----------------------------------------------------------------------
    !     This function routine generates a Gaussian random
    !     deviate of 0.0 mean and standard deviation RF.
    !     The algorithm from Box and Muller.
    !
    !      Bernard R. Brooks   January, 1988
    !
    !     copied from dynlng and renamed <*>  rm venable  <*>  7 dec 1989
    !
    use chm_kinds
    use consta
    use number
    use rndnum
    implicit none
    real(chm_real)  RF,bmgaus_rtn,rng1,rng2,rng
    INTEGER IG
    !
    !OLD routine
    if(qoldrandom.or.qbrokenclcg)then
       BMGAUS_RTN=RF*(MINTWO*LOG(OLDRANDOM(IG)))**HALF &
            *COS(TWO*PI*OLDRANDOM(IG))
    else

       ! NEW routine
       if ((rngchoice == 0) .or. (rngchoice == 4)) then
          rng1 = oldrandom(ig) ! unfortunately the order here matters
          rng2 = oldrandom(ig) ! in comparison to the above!
       elseif (rngchoice == 1) then
          rng1 = random(rngseries)
          rng2 = random(rngseries)
       elseif (rngchoice == 2) then
          call random_number(rng1)
          call random_number(rng2)
       else
          rng1=one
          rng2=half  ! to make it zero :-)
       endif

       bmgaus_rtn=rf*cos(two*pi*rng2)*(mintwo*log(rng1))**half

       ! END NEW routine

    endif

    RETURN
  END FUNCTION BMGAUS

  SUBROUTINE RANDSPEC()
    !-----------------------------------------------------------------------
    !     called by MISCOM to set random number specifications
    !
    use chm_kinds
    use exfunc
    use dimens_fcm
    use number
    !
    use stream
    use string
    use comand
    use rndnum
    use memory
    use parallel ! used also when not parallel
    use usermod,only: user_rngspec
    use param_store, only: set_param

    implicit none
    !
    character(len=4)   WRD
    integer(chm_int8) :: count, rate, maxcount
    integer(chm_int4) :: count4, rate4, maxcount4
    integer i
    integer,allocatable,dimension(:) :: lrngseeds
    logical qpresent
    !

! FOR COMPATIBILITY
!@!    if(qbrokenclcg .or. qoldrandom) then
!@!       !     Use old code here
!@!       !     second word must be distribution type
!@!       !
!@!       WRD=NEXTA4(COMLYN,COMLEN)
!@!       !
!@!       IF (WRD == 'UNIF') THEN
!@!          IRNTYP=0
!@!       ELSE IF (WRD == 'GAUS') THEN
!@!          IRNTYP=1
!@!       ELSE IF (WRD == 'OLDR') THEN
!@!          QOLDRNG = .TRUE.
!@!          if (prnlev > 2) write(outu,'(a)') &
!@!               'RANDSPEC> OLD Pseudo-Random number generator in use.'
!@!          return
!@!       ELSE IF (WRD == 'CLCG') THEN
!@!          QOLDRNG = .FALSE.
!@!          if (prnlev > 2) write(outu,'(a)') &
!@!               'RANDSPEC> CLCG Random number generator in use.'
!@!          !        write(*,*)'randspec>mynod,numnod=',mynod,numnod
!@!          return
!@!       ELSE
!@!          IF(WRNLEV >= 2) WRITE(OUTU,1001) WRD
!@!          RETURN
!@!       ENDIF
!@!1001   FORMAT ('<RANDSPEC> NOTHING DONE; ',A4, &
!@!            ' IS AN INVALID DISTRIBUTION TYPE')
!@!       !
!@!       !     if Gaussian, get sigma
!@!       !
!@!       IF(IRNTYP == 1) RNSGMA=NEXTF(COMLYN,COMLEN)
!@!       !
!@!       !     check for other options
!@!       !
!@!       IF(0 /= INDXA(COMLYN,COMLEN,'ACOS')) IRNTYP=IRNTYP+4
!@!       IF(0 /= INDXA(COMLYN,COMLEN,'ASIN')) IRNTYP=IRNTYP+8
!@!       IRNDSD=GTRMI(COMLYN,COMLEN,'ISEE',1380662)
!@!       RNSCAL=GTRMF(COMLYN,COMLEN,'SCAL',ONE)
!@!       RNOFFS=GTRMF(COMLYN,COMLEN,'OFFS',ZERO)
!@!       !
!@!       !     verify settings
!@!       !
!@!       IF(PRNLEV >= 2) WRITE(OUTU,1002) WRD,IRNDSD,RNSGMA,RNSCAL,RNOFFS
!@!1002   FORMAT('DIST= ',A4,'     SEED= ',I10,'     SIGMA= ',G12.4, &
!@!            /,'    SCALE= ',G12.4,'     OFFSET= ',G12.4)
!@!       !
!@!       !
!@!       ! Now we use a NEW syntax from doc/random.doc
!@!       !
!@!    else
! END COMPATIBILITY



! NEW CODE HERE:

    ! scan all the words in rand command so the order is not important

! there might be a need for this : TEST IT!!!
!@!    if(qbrokenclcg .or. qoldrandom) then
!@!       !     Use old code here

!    write(outu,*)' RANDSPEC> begin: rngchoice, qoldrandom, qbrokenclcg, qoldrng',&
!         rngchoice, qoldrandom, qbrokenclcg, qoldrng

    if (qbrokenclcg) then
       ! in this case the default should be the same as 'OLDC'
       rngchoice = 4
       nrand=1          ! See the comments at 'OLDC'
       qoldrng = .false.
       qoldrandom=.false.
    else
       rngchoice = 1 ! the default
       !maybe:
       qoldrandom=.false.
       qbrokenclcg=.false.
    endif

    if (indxa(comlyn,comlen,'CLCG') /= 0) then
       rngchoice = 1
       nrand=4
    endif

    if (indxa(comlyn,comlen,'OLDR') /= 0) then
       rngchoice = 0
       nrand=1
       qoldrandom = .true.
       qoldrng = .true.
    endif

    if (indxa(comlyn,comlen,'SYST') /= 0) then
       rngchoice = 2
       call random_seed(nrand)
    endif

    if (indxa(comlyn,comlen,'USER') /= 0) then
       rngchoice = 3
       nrand = 1   ! it has at least one, user routine may change it
       call user_rngspec
    endif

    if (indxa(comlyn,comlen,'OLDC') /= 0) then
       rngchoice = 4  ! OLDCLCG also available here... not only with -prevclcg
       nrand=1        ! remember, it is broken, so it has to be 1
       qbrokenclcg = .true.
       qoldrng = .false.
    endif

    ! export nrand for scripts to refer as ?NRAND
    call set_param('NRAND',nrand)

    ! in case we change the RNG in the same script we need to deallocate here
    if(allocated(rngseeds)) deallocate (rngseeds)
    call chmalloc('random.src','RANDSPEC','RNGSEEDS',NRAND,intg=rngseeds)

    ! Fill the seeds with the time values.
    ! For parallel make sure each process has distinct seed(s)
    ! we just take the small integers here:
    call system_clock(count4,rate4,maxcount4)
    ! previous 50000*mynod - not OK for integer overflow:
    rngseeds(1:nrand) = count4 + mynod
    if(rngchoice == 0) irndsd = count4 + mynod

    ! TIME is the default if no iseed is specified
    ! But maybe for oldrng we keep the old stuff, but fixed for parallel
    if (indxa(comlyn,comlen,'TIME') == 0) irndsd = 1380662 + mynod

    ! process ISEEd:
    call chmalloc('random.src','RANDSPEC','LRNGSEEDS',NRAND,intg=lrngseeds)
    lrngseeds(1:nrand) = rngseeds(1:nrand)  ! we want previous ones as default
    call gtrmim(nrand,comlyn,comlen,'ISEE',lrngseeds,rngseeds,qpresent)
    if(qpresent)  then
       if((rngchoice == 0).or.(rngchoice == 4)) irndsd = rngseeds(1)
       if(rngchoice == 1) call clcginit(irndsd)
       if(rngchoice == 4) call clcginit(irndsd)
       if(rngchoice == 2) call random_seed(put=rngseeds)
    endif
    call chmdealloc('random.src','RANDSPEC','LRNGSEEDS',NRAND,intg=lrngseeds)

    ! With this flag ON parallel LD simulations are the same for any NUMNOD:
    QRANDPAR = (INDXA(COMLYN,COMLEN,'PARA') /= 0)

    IF(INDXA(COMLYN,COMLEN,'UNIF') /= 0) THEN 
       rngdistrchoice = 0 ! default from iniall.src
       IRNTYP=0
    ENDIF

    IF(INDEX(COMLYN,'GAUS') /= 0) THEN 
       rngdistrchoice = 1
       IRNTYP=1
       rnsgma = gtrmf(comlyn,comlen,'GAUS',one)
    ENDIF

    qrngsin = (indxa(comlyn,comlen,'ASIN') /= 0)
    if(qrngsin) irntyp=irntyp+4
    qrngcos = (indxa(comlyn,comlen,'ACOS') /= 0)
    if(qrngcos) irntyp=irntyp+8


    rnscal = gtrmf(comlyn,comlen,'SCAL',one)
    rnoffs = gtrmf(comlyn,comlen,'OFFS',zero)

    if (indxa(comlyn,comlen,'TEST') /= 0) then
       ! this is not implemented yet...
    endif

    if(prnlev>2) then
       if(rngchoice==0)write(outu,'(a)') &
            'RANDSPEC> OLDRANDOM pseudo-random number generator in use'
       if(rngchoice==1)write(outu,'(a)') &
            'RANDSPEC> CLCG random number generator in use:'
       if(rngchoice==2)write(outu,'(a)') &
            'RANDSPEC> System supplied random number generator in use'
       if(rngchoice==3)write(outu,'(a)') &
            'RANDSPEC> User supplied random number generator in use'
       if(rngchoice==4)write(outu,'(a)') &
            'RANDSPEC> OLD CLCG (limited control on seeds) pseudo-random number generator in use'
       if (rngdistrchoice == 0) &
            write(outu,'(''RANDSPEC> UNIFORM distribution will be used'')')
       if (rngdistrchoice == 1) &
            write(outu, &
            '(''RANDSPEC> GAUSSIAN distribution will be used, with sigma = '',f0.5)') &
            rnsgma
       if ( rngchoice > 0 )write(outu,'(''RANDSPEC> seeds  = '',16(i0,1x))') &
            (rngseeds(i),i=1,nrand)
       if ( rngchoice == 0 )write(outu,'(''RANDSPEC> seed   = '',i0)')irndsd
       write(outu,'(''RANDSPEC> SCALE  = '',f0.8)') rnscal
       write(outu,'(''RANDSPEC> OFFSET = '',f0.8)') rnoffs
       if (qrngsin)write(outu,'(''RANDSPEC> ASIN option specified'')')
       if (qrngcos)write(outu,'(''RANDSPEC> ACOS option specified'')')
       if (qrngcos.and.qrngsin)call wrndie(-3,'<RANDSPEC>', &
            'ASIN and ACOS cannot be specified at the same time')
       if (qrandpar)write(outu,'(''RANDSPEC> PARAllel option specified'')')
    endif

! DO TU !!!!


!@!! OLD CODE - use parts of it, but it is non-functional
!@!    goto 9977
!@!
!@!       wrd=nexta4(comlyn,comlen)
!@!       rngchoice = 1
!@!       if (wrd == 'OLDR') then
!@!
!@!          nrand = 1
!@!          rngchoice = 0
!@!          call system_clock(count4,rate4,maxcount4)
!@!          irndsd = count4
!@!
!@!       else if (wrd == 'SYST') then ! use random_number()
!@!
!@!          rngchoice = 2
!@!          ! first we may need to reallocate the seeds array
!@!          if(allocated(rngseeds))then
!@!             call chmdealloc('random.src','RANDSPEC','RNGSEEDS',NRAND,intg=rngseeds)
!@!          endif
!@!          call random_seed(nrand)
!@!          call chmalloc('random.src','RANDSPEC','RNGSEEDS',NRAND,intg=rngseeds)
!@!          ! we assume TIME is the default for seeds
!@!          call system_clock(count,rate,maxcount)
!@!          rngseeds(1:nrand) = count + 50000*mynod
!@!          call random_seed(put=rngseeds)
!@!
!@!       else if (wrd == 'USER') then ! use user_rngspec in charmm/usersb.src
!@!
!@!          rngchoice = 3
!@!          call user_rngspec
!@!
!@!       else   ! this is a default & fixed CLCG
!@!
!@!          if(allocated(rngseeds))then
!@!             call chmdealloc('random.src','RANDSPEC','RNGSEEDS',NRAND,intg=rngseeds)
!@!          endif
!@!          nrand = 4  ! always 4
!@!          call chmalloc('random.src','RANDSPEC','RNGSEEDS',NRAND,intg=rngseeds)
!@!          ! NOTE: No big numbers for CLCG seeds
!@!          call system_clock(count4,rate4,maxcount4)
!@!          rngseeds(1:nrand) = count4 + 50000*mynod
!@!          irndsd = 1380662
!@!          call clcginit(irndsd)
!@!
!@!       endif
!@!       ! save the number of seeds in to the ?NRAND parameter
!@!       ! this allows users to query for numebr of seeds in case
!@!       ! of random_number() calls.
!@!       call setmsi('NRAND',nrand)
!@!       !
!@!       ! parse the rest of the command
!@!
!@!       ! parallel LD simulations are the same for any NUMNOD:
!@!       QRANDPAR = (INDXA(COMLYN,COMLEN,'PARA') /= 0)
!@!
!@!       IF(INDXA(COMLYN,COMLEN,'UNIF') /= 0) rngdistrchoice = 0 ! default from iniall.src
!@!
!@!       IF((INDXA(COMLYN,COMLEN,'GAUS') /= 0) .or. (wrd == 'GAUSS')) then
!@!          rngdistrchoice = 1
!@!          rnsgma = nextf(comlyn,comlen)
!@!       endif
!@!
!@!       qrngsin = (indxa(comlyn,comlen,'ASIN') /= 0)
!@!       qrngcos = (indxa(comlyn,comlen,'ACOS') /= 0)
!@!
!@!       rnscal = gtrmf(comlyn,comlen,'SCAL',one)
!@!       rnoffs = gtrmf(comlyn,comlen,'OFFS',zero)
!@!
!@!       irndsd = 1380662
!@!       call chmalloc('random.src','RANDSPEC','LRNGSEEDS',NRAND,intg=lrngseeds)
!@!       lrngseeds(1:nrand) = rngseeds(1:nrand)  ! we want previous ones as default
!@!       call gtrmim(nrand,comlyn,comlen,'ISEE',lrngseeds,rngseeds,qpresent)
!@!       if(qpresent)  then
!@!          if(rngchoice == 1) call clcginit(irndsd)
!@!          if(rngchoice == 2) call random_seed(put=rngseeds)
!@!       endif
!@!       call chmdealloc('random.src','RANDSPEC','LRNGSEEDS',NRAND,intg=lrngseeds)
!@!
!@!       ! printout the command specifictions:
!@!
!@!       if(prnlev>2) then
!@!          if(rngchoice==0)write(outu,'(a)') &
!@!               'RANDSPEC> OLDRANDOM pseudo-random number generator in use'
!@!          if(rngchoice==1)write(outu,'(a)') &
!@!               'RANDSPEC> CLCG random number generator in use:'
!@!          if(rngchoice==2)write(outu,'(a)') &
!@!               'RANDSPEC> System supplied random number generator in use'
!@!          if(rngchoice==3)write(outu,'(a)') &
!@!               'RANDSPEC> User supplied random number generator in use'
!@!          if (rngdistrchoice == 0) &
!@!               write(outu,'(''RANDSPEC> UNIFORM distribution will be used'')')
!@!          if (rngdistrchoice == 1) &
!@!               write(outu, &
!@!               '(''RANDSPEC> GAUSSIAN distribution will be used, with sigma = '',f0.5)') &
!@!               rnsgma
!@!          if ( rngchoice > 0 )write(outu,'(''RANDSPEC> seeds  = '',4(i0,1x))') &
!@!               (rngseeds(i),i=1,nrand)
!@!          if ( rngchoice == 0 )write(outu,'(''RANDSPEC> seed   = '',i0)')irndsd
!@!          write(outu,'(''RANDSPEC> SCALE  = '',f0.8)') rnscal
!@!          write(outu,'(''RANDSPEC> OFFSET = '',f0.8)') rnoffs
!@!          if (qrngsin)write(outu,'(''RANDSPEC> ASIN option specified'')')
!@!          if (qrngcos)write(outu,'(''RANDSPEC> ACOS option specified'')')
!@!          if (qrngcos.and.qrngsin)call wrndie(-3,'<RANDSPEC>', &
!@!               'ASIN and ACOS cannot be specified at the same time')
!@!          if (qrandpar)write(outu,'(''RANDSPEC> PARAllel option specified'')')
!@!       endif
!@!!    endif
!@!9977 continue
    !
    RETURN
  END SUBROUTINE RANDSPEC
  !
  subroutine rngmodseeds(qpresent,iseed)
    use chm_kinds
    use rndnum
    use string
    !
    ! This routine is used when iseed may have been specified in some of the
    ! charmm commands in addition to random command above.  it takes
    ! care of all the initialization process for different
    ! compatibility modes
    !
    implicit none

    integer iseed,ierr
    logical qpresent

    ! always perform this for compatibility
    if(qoldrandom.or.qbrokenclcg) then
       if (.not.qoldrng) then     !yw 05-Aug-2008
          if(qpresent) iseed=rngseeds(1)
          CALL CLCGINIT(ISEED)
          ISEED=1
       endif
    else
       !
       ! if iseed keyword was not specified then in the new
       ! random nothing needs to be changed. We only
       ! set RNGs if new seeds were read
       !
       if (qpresent) then
          ! Write a message if using only one integer after ISEED keyword
          if(gtrmim_err == -100) then
          ! Set all seed elements to given seed value to avoid spurious
          ! numbers taken from the command line. LNilsson October 2010.
             rngseeds(2:nrand)=rngseeds(1)
             call wrndie(5,'<RNGMODSEEDS>', &
                  'Only one seed number specified! See random.doc for syntax.')
          endif
          if(rngchoice > 0) then
             if(rngchoice == 1) call clcginit(iseed)
             if(rngchoice == 2) call random_seed(put=rngseeds)
             !               write(*,*)'rngmodseeds> are we here in fixed?'
             !               write(*,*)'rngmodseeds> rngchoice,seed=',rngchoice,rngseeds(1:nrand)
             !            else
             !               irndsd = iseed
          endif

       endif

    endif
    !
    return
  end subroutine rngmodseeds
  !
  subroutine IRANDOM()
    use chm_kinds
    use exfunc
    use dimens_fcm
    use number 
    use stream
    use string
    use comand
    use param_store, only: set_param

    implicit none

    !
    !  returns a random integer in the range specified
    !
    real(chm_real8) REALNB,AB
    INTEGER MAXSER
    PARAMETER(MAXSER=10000)
    INTEGER BEGNUM(MAXSER),ENDNUM(MAXSER),ISEEDN(MAXSER)
    INTEGER FAC1,FAC2
    DATA FAC1/10000000/
    DATA FAC2/1000000/
    INTEGER TOTAL,RANDM1,FACTOR,PLACES
    INTEGER RANGE,RGCP,I,SERIES
    LOGICAL DONE,QRSETUP(MAXSER)
    real(chm_real) RR,SHORT,TEMP1
    SAVE BEGNUM,ENDNUM,ISEEDN
    SAVE QRSETUP
    INTEGER RANDCNT
    DATA RANDCNT/0/
    !   Given a seed integer, this subroutine 
    !   produces a random integer in the range between
    !   begnum and endnum and returns it in TOTAL.
    !   It can produce many series of random integers
    !   in a single run.  The period is not less than
    !   10^14.  
    !                            --R.J. Petrella 
    !
    SERIES = GTRMI(COMLYN,COMLEN,'SERI',-1)
    IF(INDXA(COMLYN,COMLEN,'SETU') > 0) THEN
       IF(SERIES == -1) THEN
          WRITE(OUTU,*) &
               '  WARNING: NO SERIES SPECIFIED.  ASSUMING 1'       
          SERIES=1
       ENDIF
       WRITE(OUTU,*) '  SETTING UP RANDOM INTEGER GENERATION'
       WRITE(OUTU,*) '  FROM UNIFORM DISTRIBUTION' 
       ISEEDN(SERIES) = -1
       BEGNUM(SERIES) = -1
       ENDNUM(SERIES) = -1
       ISEEDN(SERIES) = GTRMI(COMLYN,COMLEN,'SEED',-1)
       BEGNUM(SERIES) = GTRMI(COMLYN,COMLEN,'BEGI',-1)
       ENDNUM(SERIES) = GTRMI(COMLYN,COMLEN,'ENDI',-1)
       IF(ISEEDN(SERIES) == -1) THEN
          CALL WRNDIE(-5,'<IRANDOM>', &
               ' NO SEED SPECIFIED ')
       ENDIF
       IF(BEGNUM(SERIES) == -1) THEN
          CALL WRNDIE(-5,'<IRANDOM>', &
               ' NO LOWER LIMIT SPECIFIED ')
       ENDIF
       IF(ENDNUM(SERIES) == -1) THEN
          CALL WRNDIE(-5,'<IRANDOM>', &
               ' NO UPPER LIMIT SPECIFIED ')
       ENDIF
       IF(QRSETUP(SERIES)) THEN
          WRITE(OUTU,*)  &
               '  WARNING: OVERWRITING PREVIOUS RANDOM INTEGER '
          WRITE(OUTU,*) &
               '  ******  GENERATION PARAMETERS ************** '
       ENDIF
       WRITE(OUTU,'(A10,1X,I10,1X,A1)') &
            'FOR SERIES',SERIES,':'
       WRITE(OUTU,'(2X,A7,1X,I12,1X,A7,1X,I12,1X,A7,1X,I12)')  &
            'SEED = ',ISEEDN(SERIES),'BEGI = ',BEGNUM(SERIES), &
            'ENDI = ',ENDNUM(SERIES)
       QRSETUP(SERIES)=.TRUE.
       RETURN
    ELSE
       IF(SERIES == -1) SERIES = 1
       IF(.NOT.QRSETUP(SERIES)) THEN
          CALL WRNDIE(-5,'<IRANDOM>', &
               'RANDOM INTEGER GENERATION PARAMETERS NOT SET UP')
       ENDIF
    ENDIF
    !      
    RANGE = ENDNUM(SERIES)-BEGNUM(SERIES)
    IF(RANGE <= 0) THEN
       CALL WRNDIE(-5,'<IRANDOM>', &
            'DISTRIBUTION RANGE MUST BE GREATER THAN ZERO')
    ENDIF
    RR = RANGE
    !      WRITE(6,*) 'BEGI is ',BEGNUM
    !      WRITE(6,*) 'ENDI is ',ENDNUM
    !      WRITE(6,*) 'RANGE is ',RANGE
    !      WRITE(6,*) 'SEED is ',ISEEDN
    PLACES = 0
    DO WHILE (RR >= 1)
       RR = RR/10
       PLACES = PLACES + 1
    ENDDO
    !
    !      WRITE(6,*) 'PLACE  IS ', PLACES
    ! find a random integer between 0 and range
    DONE = .FALSE.
    DO WHILE(.NOT.DONE)
       TOTAL = 0
       FACTOR = 1
       DO I = 1,PLACES
          REALNB = ISEEDN(SERIES)
          !        WRITE(6,*) 'REALNB is ',REALNB
          AB = COS(REALNB)
          IF (AB < 0) AB = -AB
          !        WRITE(6,'(A10,F8.6,A7,F18.15)') 'REALNB IS ',REALNB,'COS IS ',AB
          TEMP1 = AB*FAC2
          SHORT = TEMP1 - INT(TEMP1)
          !        WRITE(6,'(A9,F18.15)') 'SHORT IS ',SHORT
          !        WRITE(6,*) 'INT(AB*FAC1)',INT(AB*FAC1),'INT(AB*FAC2)*10',
          !     & INT(AB*FAC2)*10
          RANDM1 = (INT(SHORT*FAC1))-(INT(SHORT*FAC2)*10)
          !        WRITE(6,*) 'RANDM1 is ',RANDM1 
          TOTAL = TOTAL + FACTOR*RANDM1
          FACTOR = FACTOR*10
          ISEEDN(SERIES) = ISEEDN(SERIES) + 1
       ENDDO
       !       WRITE(6,*) 'TOTAL BEFORE SCREEN is ',TOTAL
       IF(TOTAL <= RANGE) DONE = .TRUE.
       TOTAL = TOTAL + BEGNUM(SERIES)
    ENDDO
    CALL set_param('IRAN',TOTAL)
    WRITE(6,*) 'SERIES ',SERIES,'RANDOM NUMBER IS ',TOTAL
    !
    RETURN
  END SUBROUTINE IRANDOM

  FUNCTION RANUMB() result(ranumb_rtn)
    !-----------------------------------------------------------------------
    !     returns a random number according to specifications
    !     set in RANDSPEC
    !
    use chm_kinds
    use consta
    use rndnum
    use number
    implicit none
    !
    real(chm_real) FAC,ranumb_rtn,rng
    !
    if(qbrokenclcg.or.qoldrandom)then

       !OLD stuff

       IF (IRNTYP == 0) THEN
          RANUMB_RTN=RNOFFS+RNSCAL*OLDRANDOM(IRNDSD)
       ELSE IF (IRNTYP == 1) THEN
          RANUMB_RTN=RNOFFS+RNSCAL*BMGAUS(RNSGMA,IRNDSD)
       ELSE IF (IRNTYP == 4) THEN
          RANUMB_RTN=RADDEG*ACOS(RNOFFS+RNSCAL*OLDRANDOM(IRNDSD))
       ELSE IF (IRNTYP == 5) THEN
          RANUMB_RTN=RADDEG*ACOS(RNOFFS+RNSCAL*BMGAUS(RNSGMA,IRNDSD))
       ELSE IF (IRNTYP == 8) THEN
          RANUMB_RTN=RADDEG*ASIN(RNOFFS+RNSCAL*OLDRANDOM(IRNDSD))
       ELSE IF (IRNTYP == 9) THEN
          RANUMB_RTN=RADDEG*ASIN(RNOFFS+RNSCAL*BMGAUS(RNSGMA,IRNDSD))
       ENDIF

    else

       !NEW stuff

       if (rngdistrchoice == 1) then
          rng = bmgaus(rnsgma,irndsd)    ! we chose RNG inside this routine
       else
          if(rngchoice == 0) then
             rng = oldrandom(irndsd)
          elseif(rngchoice == 1) then
             rng = random(rngseries)
             !               write(*,*)'ranumb>choice=1,rng,ser=',rng,rngseries
          elseif(rngchoice == 2) then
             call random_number(rng)
          else
             rng = zero
          endif
       endif

       rng = rnoffs+rnscal*rng

       if(qrngsin)rng = raddeg*asin(rng)
       if(qrngcos)rng = raddeg*acos(rng)

       ranumb_rtn = rng

       !END new stuff
    endif
    !
    RETURN
  END FUNCTION RANUMB

  subroutine clcg_iniall(iseed, mynod)
    use rndnum
    use memory
    use param_store, only: set_param

    integer :: iseed, mynod
    integer(chm_int8) :: count, rate, maxcount
    integer(chm_int4) :: count4, rate4, maxcount4

    call rndnum_iniall
    if (qbrokenclcg.or.qoldrandom) then
       nrand = 1
       call chmalloc('iniall.src','INIALL','RNGSEEDS',NRAND,intg=rngseeds)
       rngseeds(1)=iseed
       if (.not.qoldrng) then
          call CLCGInit(ISEED)
          ISEED=1
       endif
    else

       if(rngchoice == 1) then
          nrand = 4
          call chmalloc('iniall.src','INIALL','RNGSEEDS',NRAND,intg=rngseeds)
          ! get the seeds from system time by default
          ! NOTE: no big numbers for CLCG!!
          call system_clock(count4,rate4,maxcount4)
          rngseeds(1:nrand) = count4 + 50000*mynod
          call CLCGInit(iseed)
       endif
       if(rngchoice == 2) then ! this is not needed here for now
          call random_seed(nrand)
          call chmalloc('iniall.src','INIALL','RNGSEEDS',NRAND,intg=rngseeds)
          ! get the seeds from system time by default
          call system_clock(count,rate,maxcount)
          rngseeds(1:nrand) = count + 50000*mynod
          call random_seed(put=rngseeds)
       endif

    endif
    ! save the number of seeds in to the ?NRAND parameter
    ! this allows users to query for numebr of seeds in case
    ! of random_number() calls.
    call set_param('NRAND',nrand)
    return
  end subroutine clcg_iniall

end module clcg_mod
