! contains parameters.
!
! One thing I am missing here is when we add new parameters. For now, it requires 
! changes at several places.
! 1) IMPAR:  
! 2) IM1D : either 1 or 3. (1: am1 parameters; 3: new parameters.)
! 3) ???? :
! 4) also, make sure routine fill_qm_parameters.
!
! Entire process is complicated, and should check the results with old MNDO97 code.
! For much smoother inputing method will be developed in near future.
! -K.N. (2012-10-08)
!
module qm1_parameters
  use chm_kinds
  use dimens_fcm
  !
  use number 
  use qm1_constant
  !

#if KEY_MNDO97==1
  ! correspond to LMZ
  INTEGER, PARAMETER :: nelements = 86
  !
  logical, save :: q_parm_loaded=.false.        !Q to set .true. after loading parameters.

  ! PARAVL
  ! IMPAR: > 0 for atoms with parameters
  !            also, for AM1,PM3,AM1/d, it is the number of gaussian core terms.
  ! IM1D : > 0 for atoms with sp,spd orbitals for AM1/d
  integer,DIMENSION(1:nelements),save :: IMPAR,IM1D
  ! DPARM
  integer,DIMENSION(1:nelements),save :: LORBS
  ! DNBND
  integer,DIMENSION(1:nelements),save :: III,IIID
  ! ATORB, ground state occupation numbers of IOS,IOP,IOD for s,p,d orbitals
  integer,DIMENSION(1:nelements),save :: IOS,IOP,IOD

  ! PARDER
  real(chm_real),DIMENSION(1:nelements),save :: CORE,EHEAT,EISOL

  ! do we need the mass? maybe not. anyway, we can check later.
  !! MASS
  !real(chm_real),DIMENSION(1:nelements),save :: BMS

  ! parameters in MULTIP, note it starts from 0 to nelements
  real(chm_real),DIMENSION(6,0:nelements),save :: DD
  real(chm_real),DIMENSION(9,0:nelements),save :: PO
  real(chm_real),DIMENSION(1:nelements),save   :: DELTA , &
                                                  OMEGA

  ! PAROPT
  real(chm_real),DIMENSION(1:nelements),save :: USS,UPP, &
                                                ZS ,ZP , &
                                                BETAS,BETAP, &
                                                ALP
  ! REP
  ! Note1: HPP are not independe parameter and computed from GPP and GP2.
  !
  ! Note2: in AM1/d and MNDO/d, GSS,GPP,GP2,HPP are internally computed (see
  ! INIGHD routine). So, they are not independent parameters. But,
  ! GSP and HSP are independent parameters, and need input.
  ! 
  real(chm_real),DIMENSION(1:nelements),save :: GSS,GPP,GSP,GP2, &
                                                HSP,HPP

  ! AM,AD,AQ: used for DD,PO.. see below.
  real(chm_real),DIMENSION(1:nelements),save :: QQ,AM,AD,AQ
 
  ! D-orbital specific ones
  ! DPARM1
  real(chm_real),DIMENSION(1:nelements),save :: UDD,ZD,BETAD
  ! DPARM2
  real(chm_real),DIMENSION(1:nelements),save :: ZSN,ZPN,ZDN
  ! DPARM3
  real(chm_real),DIMENSION(1:nelements),save :: F0DD,F2DD,F4DD,F0SD,G2SD, &
                                                F0PD,F2PD,G1PD,G3PD
  ! DPARM4
  real(chm_real),DIMENSION(52,1:nelements),save :: REPD
  ! DPARM6 (initialized in initialize_r2cent and used in ROTD)
  logical,DIMENSION(45,45) :: R2CENT
  ! DPARM7
  integer,DIMENSION(1:nelements),save :: IF0SD,IG2SD

  ! AM1, PM3, AM1/d specfic parameters.
  logical, save :: q_do_am1_pm3=.false.  ! set for .true. if using am1, pm3, etc.
  ! AMPGAU
  real(chm_real),DIMENSION(1:4,1:nelements),save :: GUESS1,GUESS2,GUESS3
  integer,DIMENSION(1:nelements),save :: IMP
  ! DPARXL
  real(chm_real),DIMENSION(1:nelements),save :: GNN

  ! MNDO/d specifie one: probably, atom-atom pair specific parameters.
  ! ABOND
  real(chm_real),DIMENSION(1:nelements,1:nelements),save :: ALPB
  integer,DIMENSION(1:nelements),save :: MALPB


  ! INITIALIZATION FOR D ORBITALS.
  !
  ! PSC
  integer,parameter      :: i_size_matrix=30
  real(chm_real),save :: Fbin(i_size_matrix), &
                         Bbin(i_size_matrix,i_size_matrix), &
                         Bbin_inv(i_size_matrix,i_size_matrix)


  ! it should include qm1_block2 for some parameters.
  ! ATORB
  ! also, updated for GHO atom (85).
  !integer,DIMENSION(1:nelements), save :: IOS,IOP,IOD
  !integer,save :: IOS(1:nelements)=(/                   &
  !         1,2,1,7*2,1,7*2,1,2, 2,2,2,1,2,2,2,2,1,2,6*2,1,3*2,1,1,&
  !         2,1,1,0,1,7*2,1,22*2,1,1,6*2,1/)
  !integer,save :: IOP(1:nelements)=(/                   &
  !         4*0,1,2,3,4,5,6, 0,0,1,2,3,4,5,6,12*0,1,2,3,4,5,6,12*0,&
  !             1,2,3,4,5,6,26*0,1,2,3,4,2,0/)
  !integer,save :: IOD(1:nelements)=(/                   &
  !         20*0,1,2,3,5,5,6,7,8,10, 0,6*0,2*0,1,2,4,5,5,7,8,3*10, &
  !          6*0,0,0,1,14*0,2,3,4,5,6,7,9,10,0,6*0/)
  !
  ! DNBND: 
  ! III -ARRAY -  PRINCIPAL QUANTUM NUMBERS OF SP-AO
  ! IIID-ARRAY -  PRINCIPAL QUANTUM NUMBERS OF  D-AO
  ! also, updated for GHO atom (85)
  !integer,DIMENSION(1:nelements), save :: III,IIID
  !integer,save :: III(1:nelements)=(/2*1,8*2,8*3,18*4,18*5,30*6,2,2/)
  !integer,save :: IIID(1:nelements)=(/30*3,18*4,32*5,4*6,3,3/)
  

  !
  ! DPARM5, used in subroutine WSTORE
  !
  ! indices for one-center two-electron integrals.
  ! W(intij(i),intkl(i))=REP(intrep(i))
  !
  ! in the Raffenetti scheme, the integrals include an extra term
  ! -0.25*(REP(intrf1)+REP(intrf2))
  !
  integer,parameter      :: intij(1:243)=(/                       &
      1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, &
      4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, &
      9, 9, 9,10,10,10,10,10,10,11,11,11,11,11,11,12,12,12,12,12, &
     13,13,13,13,13,14,14,14,15,15,15,15,15,15,15,15,15,15,16,16, &
     16,16,16,17,17,17,17,17,18,18,18,19,19,19,19,19,20,20,20,20, &
     20,21,21,21,21,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22, &
     22,22,23,23,23,23,23,24,24,24,24,24,25,25,25,25,26,26,26,26, &
     26,26,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,29,29,29, &
     29,29,30,30,30,31,31,31,31,31,32,32,32,32,32,33,33,33,33,33, &
     34,34,34,34,35,35,35,35,35,36,36,36,36,36,36,36,36,36,36,36, &
     36,37,37,37,37,38,38,38,38,38,39,39,39,39,39,40,40,40,41,42, &
     42,42,42,42,43,43,43,43,44,44,44,44,44,45,45,45,45,45,45,45, &
     45,45,45/)
  integer,parameter      :: intkl(1:243)=(/                       &
     15,21,28,36,45,12,19,23,39,11,15,21,22,26,28,36,45,13,24,32, &
     38,34,37,43,11,15,21,22,26,28,36,45,17,25,31,16,20,27,44,29, &
     33,35,42,15,21,22,28,36,45, 3, 6,11,21,26,36, 2,12,19,23,39, &
      4,13,24,32,38,14,17,31, 1, 3, 6,10,15,21,22,28,36,45, 8,16, &
     20,27,44, 7,14,17,25,31,18,30,40, 2,12,19,23,39, 8,16,20,27, &
     44, 1, 3, 6,10,11,15,21,22,26,28,36,45, 3, 6,10,15,21,22,28, &
     36,45, 2,12,19,23,39, 4,13,24,32,38, 7,17,25,31, 3, 6,11,21, &
     26,36, 8,16,20,27,44, 1, 3, 6,10,15,21,22,28,36,45, 9,29,33, &
     35,42,18,30,40, 7,14,17,25,31, 4,13,24,32,38, 9,29,33,35,42, &
      5,34,37,43, 9,29,33,35,42, 1, 3, 6,10,11,15,21,22,26,28,36, &
     45, 5,34,37,43, 4,13,24,32,38, 2,12,19,23,39,18,30,40,41, 9, &
     29,33,35,42, 5,34,37,43, 8,16,20,27,44, 1, 3, 6,10,15,21,22, &
     28,36,45/)
  integer,parameter      :: intrep(1:243)=(/                      &
      1, 1, 1, 1, 1, 3, 3, 8, 3, 9, 6, 6,12,14,13, 7, 6,15, 8, 3, &
      3,11, 9,14,17, 6, 7,12,18,13, 6, 6, 3, 2, 3, 9,11,10,11, 9, &
     16,10,11, 7, 6, 4, 5, 6, 7, 9,17,19,32,22,40, 3,33,34,27,46, &
     15,33,28,41,47,35,35,42, 1, 6, 6, 7,29,38,22,31,38,51, 9,19, &
     32,21,32, 3,35,33,24,34,35,35,35, 3,34,33,26,34,11,32,44,37, &
     49, 1, 6, 7, 6,32,38,29,21,39,30,38,38,12,12, 4,22,21,19,20, &
     21,22, 8,27,26,25,27, 8,28,25,26,27, 2,24,23,24,14,18,22,39, &
     48,45,10,21,37,36,37, 1,13,13, 5,31,30,20,29,30,31, 9,19,40, &
     21,32,35,35,35, 3,42,34,24,33, 3,41,26,33,34,16,40,44,43,50, &
     11,44,32,39,10,21,43,36,37, 1, 7, 6, 6,40,38,38,21,45,30,29, &
     38, 9,32,19,22, 3,47,27,34,33, 3,46,34,27,33,35,35,35,52,11, &
     32,50,37,44,14,39,22,48,11,32,49,37,44, 1, 6, 6, 7,51,38,22, &
     31,38,29/)
  integer,parameter      :: intrf1(1:243)=(/                      &
     19,19,19,19,19, 3, 3, 8, 3, 3,33,33, 8,27,25,35,33,15, 8, 3, &
      3,34, 3,27,15,33,35, 8,28,25,33,33, 3, 2, 3, 3,34,24,35, 3, &
     41,26,35,35,33, 2,23,33,35, 3,15, 1,32,22,40, 3, 6,11,14, 0, &
     15, 6,18,16, 0, 7,11,16,19,33,33,35,29,44,22,48,44,52, 3, 1, &
     32,21,32, 3,11, 6,10,11, 7,11,11, 3,11, 6,10,11,34,32,38,37, &
     50,19,33,35,33,32,44,29,21,37,36,44,44, 8, 8, 2,22,21, 1,20, &
     21,22, 8,14,10,13,14, 8,18,13,10,14, 2,10, 5,10,27,28,22,37, &
     31,43,24,21,37,30,39,19,25,25,23,48,36,20,29,36,48, 3, 1,40, &
     21,32,11, 7,11, 3,16,11,10, 6, 3,16,10, 6,11,41,40,38,45,49, &
     34,38,32,37,26,21,45,30,37,19,35,33,33,40,44,44,21,43,36,29, &
     44, 3,32, 1,22, 3, 0,14,11, 6, 3, 0,11,14, 6,11,11, 7,51,35, &
     32,49,37,38,27,37,22,31,35,32,50,39,38,19,33,33,35,52,44,22, &
     48,44,29/)
  integer,parameter      :: intrf2(1:243)=(/                      &
     19,19,19,19,19, 9, 9,12, 9, 3,33,33, 8,27,25,35,33,17,12, 9, &
      9,35, 3,27,15,33,35, 8,28,25,33,33, 9, 4, 9, 3,35,26,34, 3, &
     42,24,34,35,33, 2,23,33,35, 3,15,19,32,22,40, 9,33,35,27,47, &
     17,33,28,42,46,35,34,41,19,33,33,35,29,44,22,48,44,52, 3,19, &
     32,21,32, 9,34,33,26,35,35,34,34, 9,35,33,24,35,35,32,44,39, &
      0,19,33,35,33,32,44,29,21,37,36,44,44, 8, 8, 2,22,21,19,20, &
     21,22,12,27,24,25,27,12,28,25,24,27, 4,26,23,26,27,28,22,37, &
     48,43,26,21,39,36,37,19,25,25,23,48,36,20,29,36,48, 3,19,40, &
     21,32,34,35,34, 9,41,35,26,33, 9,42,24,33,35,42,40,44,43, 0, &
     35,44,32,37,24,21,43,36,39,19,35,33,33,40,44,44,21,43,36,29, &
     44, 3,32,19,22, 9,46,27,35,33, 9,47,35,27,33,34,34,35,52,34, &
     32, 0,39,44,27,37,22,48,34,32, 0,37,44,19,33,33,35,52,44,22, &
     48,44,29/)

  ! special terms for boron, in AM1 or AM1/d
  real(chm_real),parameter      :: BORON1(3,4)=reshape( (/         &
                    0.182613D0,  0.118587D0, -0.073280D0, &
                    0.412253D0, -0.149917D0,  0.000000D0, &
                    0.261751D0,  0.050275D0,  0.000000D0, &
                    0.359244D0,  0.074729D0,  0.000000D0/), (/3,4/))
  real(chm_real),parameter      :: BORON2(3,4)=reshape( (/         &
                    6.0D0,       6.0D0,       5.0D0,      &
                   10.0D0,       6.0D0,       0.0D0,      &
                    8.0D0,       5.0D0,       0.0D0,      &
                    9.0D0,       9.0D0,       0.0D0/), (/3,4/))
  real(chm_real),parameter      :: BORON3(3,4)=reshape( (/         &
                    0.727592D0,  1.466639D0,  1.570975D0, &
                    0.832586D0,  1.186220D0,  0.000000D0, &
                    1.063995D0,  1.936492D0,  0.000000D0, &
                    0.819351D0,  1.574414D0,  0.000000D0/), (/3,4/))


contains

  subroutine initialize_elements_and_params(iqm_mode,QSRP_PhoT)
  ! load parameters
  ! qm_model: am1,pm3,mndo,am1d,mndod
  
  implicit none
  integer :: iqm_mode,i
  logical :: QSRP_PhoT

  if(.not. q_parm_loaded) then
     !
     q_parm_loaded=.true.

     ! we can do here.
     impar(1:nelements) = 0

     ! INITIALIZATION FOR EXTERNAL POINT CHARGES.
     DD(1:6,0) = zero  ! it is also initialzed in qmmm_setup_init
     PO(1:9,0) = zero  ! or 

     ! DEFAULT NUMBER OF ORBITALS PER ELEMENT.
     LORBS(1:2)        = 1
     LORBS(3:nelements)= 4
     LORBS(86)         = 1  ! H-link atom

     ! define factorials and binomial coefficients for SPD integrals.
     if(iqm_mode.eq.4 .or. iqm_mode.eq.5) call FBINOM(i_size_matrix)

     ! initialise IOS,IOP,IOD,III,IIID arrays
     ! also, already updated for GHO atom (85).
     IOS(1:nelements) =(/                   &
         1,2,1,(2,i=1,7),1,(2,i=1,7),1,2, 2,2,2,1,2,2,2,2,1,2,(2,i=1,6),1,(2,i=1,3),1,1,&
         2,1,1,0,1,(2,i=1,7),1,(2,i=1,22),1,1,(2,i=1,6),1/)
     IOP(1:nelements) =(/                   &
         (0,i=1,4),1,2,3,4,5,6, 0,0,1,2,3,4,5,6,(0,i=1,12),1,2,3,4,5,6,(0,i=1,12),&
             1,2,3,4,5,6,(0,i=1,26),1,2,3,4,2,0/)
     IOD(1:nelements) =(/                   &
         (0,i=1,20),1,2,3,5,5,6,7,8,10, 0,(0,i=1,6),0,0,1,2,4,5,5,7,8,10,10,10, &
         (0,i=1,6),0,0,1,(0,i=1,14),2,3,4,5,6,7,9,10,0,(0,i=1,6)/)
     III(1:nelements) =(/1,1,(2,i=1,8),(3,i=1,8),(4,i=1,18),(5,i=1,18),(6,i=1,30),2,2/)
     IIID(1:nelements)=(/(3,i=1,30),(4,i=1,18),(5,i=1,32),(6,i=1,4),3,3/)


     ! fill all original parameters and some arrays.
     call fill_qm_parameters(iqm_mode,QSRP_PhoT)

     ! for two-center integrals involving d orbitals.
     if(iqm_mode.eq.4 .or. iqm_mode.eq.5) call initialize_r2cent

     if(iqm_mode.eq.5) then
        ! MNDO/d bond parameters (mndo/d-specific parameters)
        ! Updated in subroutine MLIG. (call from routine PARDEF.)
        MALPB(1:nelements)            =0
        ALPB(1:nelements,1:nelements) =0.0d0

        call MLIG
     end if

     ! fill QM/MM interaction parameters
     call fill_qmmm_parameters
  end if        ! (.not. q_parm_loaded)

  return
  end subroutine initialize_elements_and_params


  subroutine fill_qm_parameters(iqm_mode,QSRP_PhoT)
  !
  ! computer some variables used/setup based on the parameters. 
  !
  implicit none
  integer :: iqm_mode
  logical :: QSRP_PhoT

  ! local variables
  integer :: i,j,ii
  real(chm_real), parameter :: SMALL=1.0D-10

  ! load parameters for MNDO,AM1,PM3,AM1/d,MNDO/d. 
  ! the parameters will only loaded to the main param arrays.
  call load_qm_parameters(iqm_mode,QSRP_PhoT)


  ! Now, fill the rest of parameters, including DD and PO 
  ! for MNDO,AM1,and PM3
  if(iqm_mode.le.3) then
     do i=1,nelements
        j=IMPAR(i)
        ! for atoms with parameters, IMPAR should be a positive integer. 
        ! for some, which not exist right now, it has a negative integer.
        ! for atoms, which has no parameters, IMPAR is 0.
        if(j.gt.0) then
           HPP(i)=0.5d0*(GPP(i)-GP2(i))
           ! DD(2,i) already copied.
           DD(3,i)=QQ(i)
           PO(1,i)  = PT5/AM(i)
           PO(2,i)  = PT5/AD(i)
           PO(3,i)  = PT5/AQ(i)
           PO(7,i)  = PO(1,i)
           PO(9,i)  = PO(1,i)
        else if(j.lt.0) then
           HPP(i)=0.5d0*(GPP(i)-GP2(i))
           call DDPOHY(i)
           PO(9,i)  = PO(1,i)
           EISOL(I) = EATOM(I,0,0,0)
        end if
     end do
     ! for AM1/d-PhoT: H and O atoms
     if(iqm_mode.eq.2 .and. QSRP_PhoT) then
        i=1
        HPP(i)=0.5d0*(GPP(i)-GP2(i))
        call DDPOHY(i)
        PO(9,i)  = PO(1,i)
        EISOL(I) = EATOM(I,0,0,0)
 
        i=8
        HPP(i)=0.5d0*(GPP(i)-GP2(i))
        call DDPOHY(i)
        PO(9,i)  = PO(1,i)
        EISOL(I) = EATOM(I,0,0,0)
     end if
  ! for AM1/d
  else if(iqm_mode.eq.4) then
     do i=1,nelements
        j =IMPAR(i)    ! number of gaussian core terms.
        ii=IM1D(i)     ! either using original AM1 or others.
        if(ii.eq.1) then      ! original sp-orbitals
           if(j.gt.0) then
              HPP(i)=0.5d0*(GPP(i)-GP2(i))
              ! DD(2,i) already copied.
              DD(3,i)=QQ(i)
              PO(1,i)  = PT5/AM(i)
              PO(2,i)  = PT5/AD(i)
              PO(3,i)  = PT5/AQ(i)
              PO(7,i)  = PO(1,i)
              PO(9,i)  = PO(1,i)
           else if(j.lt.0) then
              HPP(i)=0.5d0*(GPP(i)-GP2(i))
              call DDPOHY(i)
              PO(9,i)  = PO(1,i)
              EISOL(I) = EATOM(I,0,0,0)
           end if
        else if(ii.eq.2) then ! new sp-orbitals.
           if(j.ne.0) then
              HPP(i)=0.5d0*(GPP(i)-GP2(i)) 
              call DDPOHY(i)
              !PO(9,I)  = POCORD; already copied, see load_qm_parameters
              EISOL(I) = EATOM(I,0,0,0)
           end if
        else if(ii.eq.3) then ! spd-orbitals
           if(j.ne.0) then
              HPP(i)   = 0.5d0*(GPP(i)-GP2(i))
              if(F0SD(i).GT.SMALL) then
                  IF0SD(I) = 1
              else
                 IF0SD(I) = 0
              end if
              if(G2SD(I).GT.SMALL) then
                 IG2SD(I) = 1
              else
                 IG2SD(I) = 0
              end if
              LORBS(I) = 9
              CALL INIGHD(I)
              !
              CALL DDPOHY(I)
              !PO(9,I)  = POCORD; already copied, see load_qm_parameters
              EISOL(I) = EATOM(I,0,0,0)
           end if
        end if
     end do
     ! for AM1/d-PhoT: H, O, and P atoms
     if(QSRP_PhoT) then
        i=1
        HPP(i)=0.5d0*(GPP(i)-GP2(i))
        call DDPOHY(i)
        PO(9,i)  = PO(1,i)
        EISOL(I) = EATOM(I,0,0,0)

        i=8
        HPP(i)=0.5d0*(GPP(i)-GP2(i))
        call DDPOHY(i)
        PO(9,i)  = PO(1,i)
        EISOL(I) = EATOM(I,0,0,0)

        i=15
        HPP(i)   = 0.5d0*(GPP(i)-GP2(i))
        if(F0SD(i).GT.SMALL) then
            IF0SD(I) = 1
        else
           IF0SD(I) = 0
        end if
        if(G2SD(I).GT.SMALL) then
           IG2SD(I) = 1
        else
           IG2SD(I) = 0
        end if
        LORBS(I) = 9
        CALL INIGHD(I)
        !
        CALL DDPOHY(I)
        EISOL(I) = EATOM(I,0,0,0)
     end if
  ! for MNDO/d
  else if(iqm_mode.eq.5) then
     do i=1,nelements
        j =IMPAR(i)    ! 0 (no param), 1(original MNDO), 
                       ! 2 (new MNDO), 3(MNDO/d param).
        if(j.eq.1) then
           HPP(i)   = 0.5d0*(GPP(i)-GP2(i))
           ! DD(2,i) already copied.
           DD(3,i)  = QQ(i)
           PO(1,i)  = PT5/AM(i)
           PO(2,i)  = PT5/AD(i)
           PO(3,i)  = PT5/AQ(i)
           PO(7,i)  = PO(1,i)
           PO(9,i)  = PO(1,i)
        else if(j.eq.2) then
           HPP(i)   = 0.5d0*(GPP(i)-GP2(i))
           call DDPOHY(i)
           !PO(9,I)  = POCORD; already copied, see load_qm_parameters
           EISOL(I) = EATOM(I,0,0,0)
        else if(j.eq.3) then
           HPP(i)   = 0.5d0*(GPP(i)-GP2(i))
           if(F0SD(i).GT.SMALL) then
              IF0SD(I) = 1
           else
              IF0SD(I) = 0
           end if
           if(G2SD(I).GT.SMALL) then
              IG2SD(I) = 1
           else
              IG2SD(I) = 0
           end if
           LORBS(I) = 9
           CALL INIGHD(I)
           !
           CALL DDPOHY(I)
           !PO(9,I)  = POCORD; already copied, see load_qm_parameters
           EISOL(I) = EATOM(I,0,0,0)
        end if
     end do
  end if

  return
  end subroutine fill_qm_parameters


  subroutine fill_qmmm_parameters
  !
  ! new version of subroutine INQMMM. So, majority will be set here.
  ! Called from subroutine START, which is called from MNDO97.
  ! 
  use qm1_info, only : qm_main_r
  implicit none

  ! local variables
  integer :: nat_qm
  integer :: i,ni

  ! if these are first call, no need to get coords and charges.
  do i=1,qm_main_r%numat
     ni = qm_main_r%nat(i)
     DELTA(ni)=zero
     OMEGA(ni)=ALP(ni)
  end do
  ! charge separations and additive terms for test charge,
  ! where test charge treated as classical monopole.
  DD(1:6,0) = zero
  PO(1:9,0) = zero

  return
  end subroutine fill_qmmm_parameters


      subroutine load_qm_parameters(iqm_mode,QSRP_PhoT)
      implicit none
      integer :: iqm_mode,i
      logical :: QSRP_PhoT

      ! Initialization for all parameters in MNDO, AM1, PM3, AM1/d, and MNDO/d.
      ! Some of the data statements were copied from MOPAC(6.0) written by 
      ! JJP Steward.
      !
      ! Special Convention:
      ! Element 85 is reserved from GHO-atoms.
      ! Element 86 is reserved for Pseudo-link atom for QM/MM treatments.
      !            This has core charge 1, i.e., one electron in a 2S orbital.
      !            Experimental Heat of Formation and Mass from Methyl (CH3).

      ! *** CORE CHARGES.
      !     LINES      ATOMIC NUMBERS
      !     1 - 3          1 - 18
      !     4 - 5         19 - 36
      !     6 - 7         37 - 54
      !     8 - B         55 - 86

      ! ENTHALPIES OF FORMATION OF GASEOUS ATOMS 
      ! ANNUAL REPORTS,1974,71B,P 117.  
      ! THERE ARE SOME SIGNIFICANT DIFFERENCES BETWEEN THE VALUES REPORTED 
      ! THERE AND THE VALUES PREVIOUSLY IN THE BLOCK OF THIS PROGRAM.  
      ! ONLY THE THIRD  ROW ELEMENTS HAVE BEEN UPDATED.
      ! 
      ! ALL THE OTHER ELEMENTS ARE TAKEN FROM CRC HANDBOOK 1981-1982.
      !     LINES      ATOMIC NUMBERS
      !     1 - 3          1 - 18
      !     4 - 5         19 - 36
      !     6 - 7         37 - 54
      !     8 - B         55 - 86
      GNN(1:nelements) = 1.0d0
      ! special section for AM1/d-PhoT.
      if(QSRP_PhoT .and. (iqm_mode.eq.2 .or. iqm_mode.eq.4)) then
          !GNN(1) =1.0d0
          !GNN(8) =1.0d0
          GNN(15)=0.3537220866D0
      end if
      core( 1)= 1.D0; EHEAT( 1)= 52.102D0 ! H
      core( 2)= 2.D0; EHEAT( 2)=  0.000D0 ! He

      core( 3)= 1.D0; EHEAT( 3)= 38.410D0 ! Li
      core( 4)= 2.D0; EHEAT( 4)= 76.960D0 ! Be
      core( 5)= 3.D0; EHEAT( 5)=135.700D0 ! B
      core( 6)= 4.D0; EHEAT( 6)=170.890D0 ! C
      core( 7)= 5.D0; EHEAT( 7)=113.000D0 ! N
      core( 8)= 6.D0; EHEAT( 8)= 59.559D0 ! O
      core( 9)= 7.D0; EHEAT( 9)= 18.890D0 ! F
      core(10)= 8.D0; EHEAT(10)=  0.000D0 ! Ne

      core(11)= 1.D0; EHEAT(11)= 25.650D0 ! Na
      core(12)= 2.D0; EHEAT(12)= 35.000D0 ! Mg
      core(13)= 3.D0; EHEAT(13)= 79.490D0 ! Al
      core(14)= 4.D0; EHEAT(14)=108.390D0 ! Si
      core(15)= 5.D0; EHEAT(15)= 75.570D0 ! P
      core(16)= 6.D0; EHEAT(16)= 66.400D0 ! S
      core(17)= 7.D0; EHEAT(17)= 28.990D0 ! Cl
      core(18)= 8.D0; EHEAT(18)=  0.000D0 ! Ar

      core(19)= 1.D0; EHEAT(19)= 21.420D0 ! K
      core(20)= 2.D0; EHEAT(20)= 42.600D0 ! Ca
      core(21)= 3.D0; EHEAT(21)= 90.300D0 ! Sc
      core(22)= 4.D0; EHEAT(22)=112.300D0 ! Ti
      core(23)= 5.D0; EHEAT(23)=122.900D0 ! V
      core(24)= 6.D0; EHEAT(24)= 95.000D0 ! Cr
      core(25)= 7.D0; EHEAT(25)= 67.700D0 ! Mn
      core(26)= 8.D0; EHEAT(26)= 99.300D0 ! Fe
      core(27)= 9.D0; EHEAT(27)=102.400D0 ! Co
      core(28)=10.D0; EHEAT(28)=102.800D0 ! Ni
      core(29)=11.D0; EHEAT(29)= 80.700D0 ! Cu
      core(30)= 2.D0; EHEAT(30)= 31.170D0 ! Zn
      core(31)= 3.D0; EHEAT(31)= 65.400D0 ! Ga
      core(32)= 4.D0; EHEAT(32)= 89.500D0 ! Ge
      core(33)= 5.D0; EHEAT(33)= 72.300D0 ! As
      core(34)= 6.D0; EHEAT(34)= 54.300D0 ! Se
      core(35)= 7.D0; EHEAT(35)= 26.740D0 ! Br
      core(36)= 8.D0; EHEAT(36)=  0.000D0 ! Kr

      core(37)= 1.D0; EHEAT(37)= 19.600D0 ! Rb
      core(38)= 2.D0; EHEAT(38)= 39.100D0 ! Sr
      core(39)= 3.D0; EHEAT(39)=101.500D0 ! Y
      core(40)= 4.D0; EHEAT(40)=145.500D0 ! Zr
      core(41)= 5.D0; EHEAT(41)=172.400D0 ! Nb
      core(42)= 6.D0; EHEAT(42)=157.300D0 ! Mo
      core(43)= 7.D0; EHEAT(43)=  0.000D0 ! Tc
      core(44)= 8.D0; EHEAT(44)=155.500D0 ! Ru
      core(45)= 9.D0; EHEAT(45)=133.000D0 ! Rh
      core(46)=10.D0; EHEAT(46)= 90.000D0 ! Pd
      core(47)=11.D0; EHEAT(47)= 68.100D0 ! Ag
      core(48)= 2.D0; EHEAT(48)= 26.720D0 ! Cd
      core(49)= 3.D0; EHEAT(49)= 58.000D0 ! In
      core(50)= 4.D0; EHEAT(50)= 72.200D0 ! Sn
      core(51)= 5.D0; EHEAT(51)= 63.200D0 ! Sb
      core(52)= 6.D0; EHEAT(52)= 47.000D0 ! Te
      core(53)= 7.D0; EHEAT(53)= 25.517D0 ! I
      core(54)= 8.D0; EHEAT(54)=  0.000D0 ! Xe

      core(55)= 1.D0; EHEAT(55)= 18.700D0 ! Cs
      core(56)= 2.D0; EHEAT(56)= 42.500D0 ! Ba
      core(57)= 3.D0; EHEAT(57)=  0.000D0 ! La
      core(58)= 3.D0; EHEAT(58)=101.300D0 ! Ce
      core(59)= 3.D0; EHEAT(59)=  0.000D0 ! Pr
      core(60)= 3.D0; EHEAT(60)=  0.000D0 ! Nd
      core(61)= 3.D0; EHEAT(61)=  0.000D0 ! Pm
      core(62)= 3.D0; EHEAT(62)= 49.400D0 ! Sm
      core(63)= 3.D0; EHEAT(63)=  0.000D0 ! Eu
      core(64)= 3.D0; EHEAT(64)=  0.000D0 ! Gd
      core(65)= 3.D0; EHEAT(65)=  0.000D0 ! Tb
      core(66)= 3.D0; EHEAT(66)=  0.000D0 ! Dy
      core(67)= 3.D0; EHEAT(67)=  0.000D0 ! Ho
      core(68)= 3.D0; EHEAT(68)= 75.800D0 ! Er
      core(59)= 3.D0; EHEAT(69)=  0.000D0 ! Tm
      core(70)= 3.D0; EHEAT(70)= 36.350D0 ! Yb
      core(71)= 3.D0; EHEAT(71)=  0.000D0 ! Lu
      core(72)= 4.D0; EHEAT(72)=148.000D0 ! Hf
      core(73)= 5.D0; EHEAT(73)=186.900D0 ! Ta
      core(74)= 6.D0; EHEAT(74)=203.100D0 ! W
      core(75)= 7.D0; EHEAT(75)=185.000D0 ! Re
      core(76)= 8.D0; EHEAT(76)=188.000D0 ! Os
      core(77)= 9.D0; EHEAT(77)=160.000D0 ! Ir
      core(78)=10.D0; EHEAT(78)=135.200D0 ! Pt
      core(79)=11.D0; EHEAT(79)= 88.000D0 ! Au
      core(80)= 2.D0; EHEAT(80)= 14.690D0 ! Hg
      core(81)= 3.D0; EHEAT(81)= 43.550D0 ! Tl
      core(82)= 4.D0; EHEAT(82)= 46.620D0 ! Pb
      core(83)= 5.D0; EHEAT(83)= 50.100D0 ! Bi
      core(84)= 6.D0; EHEAT(84)=  0.000D0 ! Po
!      core(85)= 7.D0; EHEAT(85)=  0.000D0! At 
      ! GHO atoms
      core(85)= 4.D0; EHEAT(85)=170.890D0 ! Sp3-hybrid GHO-C atom
      core(86)= 1.D0; EHEAT(86)= 34.800D0 ! Rn, but used as the Pseudo-atom
                                           ! used for link atom in QM/MM

      ! list of elements with corresponding parameters for each method
      ! for am1, pm3, and am1d, the value is also used for the number
      ! of gaussian core-core interaction parameters.
      if(iqm_mode.eq.1) then   
         ! MNDO parameters
         IMPAR(1:nelements)=(/                         &
                  1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0, &
                  0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0, &
                  0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0, &
                 (0,i=1,25),            1,0,1,0,0,1,1/)
      else if(iqm_mode.eq.2) then
         ! AM1 parameters
         IMPAR(1:nelements)=(/                         &
                  3,0,0,0,3,4,3,2,2,0,0,3,1,3,3,3,2,0, &
                  0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,2,0, &
                  0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,2,0, &
                 (0,i=1,25),            1,0,0,0,0,4,0/)
      else if(iqm_mode.eq.3) then
         ! PM3 parameters
         IMPAR(1:nelements)=(/                         &
                  2,0,2,2,0,2,2,2,2,0,0,2,2,2,2,2,2,0, &
                  0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0, &
                  0,0,0,0,0,0,0,0,0,0,0,1,2,2,2,2,2,0, &
                 (0,i=1,25),            2,2,2,2,0,2,0/)
      else if(iqm_mode.eq.4) then
         ! AM1/d parameters, also for number of gaussian core terms.
         !                   same as AM1 till atomic number 12 (upto Mg), Zn (30), or 85.
         IMPAR(1:nelements)=(/                         &
                  3,0,0,0,3,4,3,2,2,0,0,3,0,0,3,2,2,0, &   ! temporarily, Mg: use AM1
                  0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0, &   !              Zn: use AM1
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0, &
                 (0,i=1,25),            0,0,0,0,0,4,0/)
         ! Values IM1D=1:  Original AM1 parameters (sp).
         !             2:  New AM1/d parametrs (sp).  <=So, far none.
         !             3:  New AM1/d parameters (spd).
         IM1D(1:nelements)=(/                          &
                  1,1,1,1,1,1,1,1,1,0,1,1,0,0,3,3,3,0, &   ! temporarily, Mg: use AM1
                  0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,3,0, &   !              Zn: use AM1
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0, &
                 (0,i=1,25),            0,0,0,0,0,1,0/)
      else if(iqm_mode.eq.5) then
         ! MNDO/d parameters
         ! Values IMPAR=1:  Original MNDO parameters (sp).
         !              2:  New MNDO/d parametrs (sp).
         !              3:  New MNDO/d parameters (spd).
         IMPAR(1:nelements)=(/                         &
                  1,1,2,1,1,1,1,1,1,0,2,2,3,3,3,3,3,0, &
                  0,0,0,3,0,0,0,3,0,3,3,2,3,3,3,3,3,0, &
                  0,0,0,3,0,0,0,0,0,3,3,2,3,3,3,3,3,0, &
             (0,i=1,17),3,0,0,0,0,0,0,0,2,3,0,0,0,1,1/)
         ! MNDO/d bond parameters:
         ! Updated in subroutine MLIG. (call from routine PARDEF.)
         !
         ! Done above.
         !
         ! MALPB(1:nelements)=0
         ! ALPB(1:nelements) =0.0d0
      end if

      ! H : HYDROGEN (1)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 99, 4899 (1977)
         USS( 1)  =   -11.9062760D0
         UPP( 1)  =     0.0000000D0
         BETAS( 1)=    -6.9890640D0
         BETAP( 1)=    -6.9890640D0
         ZS( 1)   =     1.3319670D0
         ZP( 1)   =     1.3319670D0
         ALP( 1)  =     2.5441341D0
         EISOL( 1)=   -11.9062760D0
         GSS( 1)  =    12.8480000D0
         GSP( 1)  =     0.0000000D0
         GPP( 1)  =     0.0000000D0
         GP2( 1)  =     0.0000000D0
         HSP( 1)  =     0.0000000D0
         DD(2, 1) =     0.0000000D0
         QQ( 1)   =     0.0000000D0
         AM( 1)   =     0.4721793D0
         AD( 1)   =     0.4721793D0
         AQ( 1)   =     0.4721793D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909 (1985)
         if(QSRP_PhoT) then
            ! AM1/d-PhoT parameters: Nam et al., JTCT (2007).
            USS( 1)  =   -10.9346095787D0
            UPP( 1)  =     0.0000000D0
            BETAS( 1)=    -5.9111081432D0
            BETAP( 1)=    -5.9111081432D0 ! probably, not used.
            ZS( 1)   =     1.1438456153D0
            ZP( 1)   =     1.1438456153D0 ! probably, not used.
            ALP( 1)  =     2.8849150878D0
            EISOL( 1)=   -11.3964270D0    ! should be recomputed.
            GSS( 1)  =    13.7374528871D0
            GSP( 1)  =     0.0000000D0
            GPP( 1)  =     0.0000000D0
            GP2( 1)  =     0.0000000D0
            HSP( 1)  =     0.0000000D0
            DD(2, 1) =     0.0000000D0
            QQ( 1)   =     0.0000000D0
            AM( 1)   =     0.4721793D0
            AD( 1)   =     0.4721793D0
            AQ( 1)   =     0.4721793D0
            GUESS1(1, 1)=  0.1062378643D0
            GUESS2(1, 1)=  5.7352899834D0
            GUESS3(1, 1)=  1.2614300828D0
            GUESS1(2, 1)=  0.0040431670D0
            GUESS2(2, 1)=  7.0801221948D0
            GUESS3(2, 1)=  2.0840953766D0
            GUESS1(3, 1)= -0.0027998104D0
            GUESS2(3, 1)=  0.7399134799D0
            GUESS3(3, 1)=  3.6494739656D0
         else
            ! original AM1 parameters
            USS( 1)  =   -11.3964270D0
            UPP( 1)  =     0.0000000D0
            BETAS( 1)=    -6.1737870D0
            BETAP( 1)=    -6.1737870D0
            ZS( 1)   =     1.1880780D0
            ZP( 1)   =     1.1880780D0
            ALP( 1)  =     2.8823240D0
            EISOL( 1)=   -11.3964270D0
            GSS( 1)  =    12.8480000D0
            GSP( 1)  =     0.0000000D0
            GPP( 1)  =     0.0000000D0
            GP2( 1)  =     0.0000000D0
            HSP( 1)  =     0.0000000D0
            DD(2, 1) =     0.0000000D0
            QQ( 1)   =     0.0000000D0
            AM( 1)   =     0.4721793D0
            AD( 1)   =     0.4721793D0
            AQ( 1)   =     0.4721793D0
            GUESS1(1, 1)=  0.1227960D0
            GUESS2(1, 1)=  5.0000000D0
            GUESS3(1, 1)=  1.2000000D0
            GUESS1(2, 1)=  0.0050900D0
            GUESS2(2, 1)=  5.0000000D0
            GUESS3(2, 1)=  1.8000000D0
            GUESS1(3, 1)= -0.0183360D0
            GUESS2(3, 1)=  2.0000000D0
            GUESS3(3, 1)=  2.1000000D0
         end if
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS( 1)  =   -13.0733210D0
         UPP( 1)  =     0.0000000D0
         BETAS( 1)=    -5.6265120D0
         BETAP( 1)=    -5.6265120D0
         ZS( 1)   =     0.9678070D0
         ZP( 1)   =     0.9678070D0
         ALP( 1)  =     3.3563860D0
         EISOL( 1)=   -13.0733210D0
         GSS( 1)  =    14.7942080D0
         GSP( 1)  =     0.0000000D0
         GPP( 1)  =     0.0000000D0
         GP2( 1)  =     0.0000000D0
         HSP( 1)  =     0.0000000D0
         DD(2, 1) =     0.0000000D0
         QQ( 1)   =     0.0000000D0
         AM( 1)   =     0.5437048D0
         AD( 1)   =     0.5437048D0
         AQ( 1)   =     0.5437048D0
         GUESS1(1, 1)=  1.1287500D0
         GUESS2(1, 1)=  5.0962820D0
         GUESS3(1, 1)=  1.5374650D0
         GUESS1(2, 1)= -1.0603290D0
         GUESS2(2, 1)=  6.0037880D0
         GUESS3(2, 1)=  1.5701890D0
      end if  ! H

      ! He: HELIUM (2)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: M. KOLB, W. THIEL, J. COMP. CHEM., 14, 37, (1993)
         USS( 2)  =   -54.4030000D0
         UPP( 2)  =     0.0000000D0
         BETAS( 2)=   -28.6938694D0
         BETAP( 2)=   -28.6938694D0
         ZS( 2)   =     2.0179064D0
         ZP( 2)   =     2.0179064D0
         ALP( 2)  =     3.2620769D0
         EISOL( 2)=   -78.9830000D0
         GSS( 2)  =    29.8230000D0
         GSP( 2)  =     0.0000000D0
         GPP( 2)  =     0.0000000D0
         GP2( 2)  =     0.0000000D0
         HSP( 2)  =     0.0000000D0
         DD(2, 2) =     0.0000000D0
         QQ( 2)   =     0.0000000D0
         AM( 2)   =     1.0960309D0
         AD( 2)   =     1.0960309D0
         AQ( 2)   =     1.0960309D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      end if  ! He

      ! Li: LITHIUM (3)
      if(iqm_mode.eq.1) then
      ! MNDO: TAKEN FROM MNDOC BY W.THIEL,QCPE NO.438, V. 2, P.63, (1982).
         USS( 3)  =    -5.1280000D0
         UPP( 3)  =    -2.7212000D0
         BETAS( 3)=    -1.3500400D0
         BETAP( 3)=    -1.3500400D0
         ZS( 3)   =     0.7023800D0
         ZP( 3)   =     0.7023800D0
         ALP( 3)  =     1.2501400D0
         EISOL( 3)=    -5.1280000D0
         GSS( 3)  =     7.3000000D0
         GSP( 3)  =     5.4200000D0
         GPP( 3)  =     5.0000000D0
         GP2( 3)  =     4.5200000D0
         HSP( 3)  =     0.8300000D0
         DD(2, 3) =     2.0549783D0
         QQ( 3)   =     1.7437069D0
         AM( 3)   =     0.2682837D0
         AD( 3)   =     0.2269793D0
         AQ( 3)   =     0.2614581D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1
      else if(iqm_mode.eq.3) then
      ! PM3: E. ANDERS, R. KOCH, P. FREUNSCHT, J. COMP. CHEM., IN PRESS. 
         USS( 3)  =    -5.3000000D0
         UPP( 3)  =    -3.4000000D0
         BETAS( 3)=    -0.5500000D0
         BETAP( 3)=    -1.5000000D0
         ZS( 3)   =     0.6500000D0
         ZP( 3)   =     0.7500000D0
         ALP( 3)  =     1.2550000D0
         EISOL( 3)=    -5.3000000D0
         GSS( 3)  =     4.5000000D0
         GSP( 3)  =     3.0000000D0
         GPP( 3)  =     5.2500000D0
         GP2( 3)  =     4.5000000D0
         HSP( 3)  =     0.1500000D0
         DD(2, 3) =     2.0357652D0
         QQ( 3)   =     1.6329932D0
         AM( 3)   =     0.1653804D0
         AD( 3)   =     0.1156738D0
         AQ( 3)   =     0.3166080D0
         GUESS1(1, 3)= -0.4500000D0
         GUESS2(1, 3)=  5.0000000D0
         GUESS3(1, 3)=  1.0000000D0
         GUESS1(2, 3)=  0.8000000D0
         GUESS2(2, 3)=  6.5000000D0
         GUESS3(2, 3)=  1.0000000D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d (sp): W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS. 
      !              - TENTATIVE
         USS( 3)  =    -5.40000000D0
         UPP( 3)  =    -3.55577440D0
         ZS( 3)   =     3.71441780D0
         ZP( 3)   =     0.73038482D0
         BETAS( 3)=    -0.94654058D0
         BETAP( 3)=    -2.61631670D0
         ALP( 3)  =     1.25014000D0
         GSS( 3)  =     4.71508920D0
         GPP( 3)  =     4.06731370D0
         GSP( 3)  =     4.92823190D0
         GP2( 3)  =     3.62892060D0
         HSP( 3)  =     0.65437903D0
         ZSN( 3)  =     0.47700000D0
         ZPN( 3)  =     0.38190150D0
         PO(9, 3) =     2.57172660D0
         F0SD( 3) =     0.00000000D0
         G2SD( 3) =     0.00000000D0
      end if  ! Li

      ! Be: BERYLLIUM (4)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: M.J.S. DEWAR, H.S. RZEPA, J. AM. CHEM. SOC., 100, 777, (1978)
         USS( 4)  =   -16.6023780D0
         UPP( 4)  =   -10.7037710D0
         BETAS( 4)=    -4.0170960D0
         BETAP( 4)=    -4.0170960D0
         ZS( 4)   =     1.0042100D0
         ZP( 4)   =     1.0042100D0
         ALP( 4)  =     1.6694340D0
         EISOL( 4)=   -24.2047560D0
         GSS( 4)  =     9.0000000D0
         GSP( 4)  =     7.4300000D0
         GPP( 4)  =     6.9700000D0
         GP2( 4)  =     6.2200000D0
         HSP( 4)  =     1.2800000D0
         DD(2, 4) =     1.4373245D0
         QQ( 4)   =     1.2196103D0
         AM( 4)   =     0.3307607D0
         AD( 4)   =     0.3356142D0
         AQ( 4)   =     0.3846373D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS( 4)  =   -17.2647520D0
         UPP( 4)  =   -11.3042430D0
         BETAS( 4)=    -3.9620530D0
         BETAP( 4)=    -2.7806840D0
         ZS( 4)   =     0.8774390D0
         ZP( 4)   =     1.5087550D0
         ALP( 4)  =     1.5935360D0
         EISOL( 4)=   -25.5166530D0
         GSS( 4)  =     9.0128510D0
         GSP( 4)  =     6.5761990D0
         GPP( 4)  =     6.0571820D0
         GP2( 4)  =     9.0052190D0
         HSP( 4)  =     0.5446790D0
         DD(2, 4) =     1.0090531D0
         QQ( 4)   =     0.8117586D0
         AM( 4)   =     0.3312330D0
         AD( 4)   =     0.2908996D0
         AQ( 4)   =     0.3530008D0
         GUESS1(1, 4)=  1.6315720D0
         GUESS2(1, 4)=  2.6729620D0
         GUESS3(1, 4)=  1.7916860D0
         GUESS1(2, 4)= -2.1109590D0
         GUESS2(2, 4)=  1.9685940D0
         GUESS3(2, 4)=  1.7558710D0
      end if  ! Be

      ! B : BORON (5)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: M.J.S. DEWAR, M.L. MCKEE, J. AM. CHEM. SOC., 99, 5231 (1977)
         USS( 5)  =   -34.5471300D0
         UPP( 5)  =   -23.1216900D0
         BETAS( 5)=    -8.2520540D0
         BETAP( 5)=    -8.2520540D0
         ZS( 5)   =     1.5068010D0
         ZP( 5)   =     1.5068010D0
         ALP( 5)  =     2.1349930D0
         EISOL( 5)=   -64.3159500D0
         GSS( 5)  =    10.5900000D0
         GSP( 5)  =     9.5600000D0
         GPP( 5)  =     8.8600000D0
         GP2( 5)  =     7.8600000D0
         HSP( 5)  =     1.8100000D0
         DD(2, 5) =     0.9579073D0
         QQ( 5)   =     0.8128113D0
         AM( 5)   =     0.3891951D0
         AD( 5)   =     0.4904730D0
         AQ( 5)   =     0.5556979D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: M.J.S. DEWAR, C. JIE, E. G. ZOEBISCH ORGANOMETALLICS 7, 513 (1988)
         USS( 5)  =   -34.4928700D0
         UPP( 5)  =   -22.6315250D0
         BETAS( 5)=    -9.5991140D0
         BETAP( 5)=    -6.2737570D0
         ZS( 5)   =     1.6117090D0
         ZP( 5)   =     1.5553850D0
         ALP( 5)  =     2.4469090D0
         EISOL( 5)=   -63.7172650D0
         GSS( 5)  =    10.5900000D0
         GSP( 5)  =     9.5600000D0
         GPP( 5)  =     8.8600000D0
         GP2( 5)  =     7.8600000D0
         HSP( 5)  =     1.8100000D0
         DD(2, 5) =     0.9107622D0
         QQ( 5)   =     0.7874223D0
         AM( 5)   =     0.3891951D0
         AD( 5)   =     0.5045152D0
         AQ( 5)   =     0.5678856D0
      else if(iqm_mode.eq.3) then
      ! PM3:
      end if  ! B

      ! C : CARBON (6)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 99, 4899 (1977)
         USS( 6)  =   -52.2797450D0
         UPP( 6)  =   -39.2055580D0
         BETAS( 6)=   -18.9850440D0
         BETAP( 6)=    -7.9341220D0
         ZS( 6)   =     1.7875370D0
         ZP( 6)   =     1.7875370D0
         ALP( 6)  =     2.5463800D0
         EISOL( 6)=  -120.5006060D0
         GSS( 6)  =    12.2300000D0
         GSP( 6)  =    11.4700000D0
         GPP( 6)  =    11.0800000D0
         GP2( 6)  =     9.8400000D0
         HSP( 6)  =     2.4300000D0
         DD(2, 6) =     0.8074662D0
         QQ( 6)   =     0.6851578D0
         AM( 6)   =     0.4494671D0
         AD( 6)   =     0.6149474D0
         AQ( 6)   =     0.6685897D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909 (1985)
         USS( 6)  =   -52.0286580D0
         UPP( 6)  =   -39.6142390D0
         BETAS( 6)=   -15.7157830D0
         BETAP( 6)=    -7.7192830D0
         ZS( 6)   =     1.8086650D0
         ZP( 6)   =     1.6851160D0
         ALP( 6)  =     2.6482740D0
         EISOL( 6)=  -120.8157940D0
         GSS( 6)  =    12.2300000D0
         GSP( 6)  =    11.4700000D0
         GPP( 6)  =    11.0800000D0
         GP2( 6)  =     9.8400000D0
         HSP( 6)  =     2.4300000D0
         DD(2, 6) =     0.8236736D0
         QQ( 6)   =     0.7268015D0
         AM( 6)   =     0.4494671D0
         AD( 6)   =     0.6082946D0
         AQ( 6)   =     0.6423492D0
         GUESS1(1, 6)=  0.0113550D0
         GUESS2(1, 6)=  5.0000000D0
         GUESS3(1, 6)=  1.6000000D0
         GUESS1(2, 6)=  0.0459240D0
         GUESS2(2, 6)=  5.0000000D0
         GUESS3(2, 6)=  1.8500000D0
         GUESS1(3, 6)= -0.0200610D0
         GUESS2(3, 6)=  5.0000000D0
         GUESS3(3, 6)=  2.0500000D0
         GUESS1(4, 6)= -0.0012600D0
         GUESS2(4, 6)=  5.0000000D0
         GUESS3(4, 6)=  2.6500000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM.  10, 209 (1989)
         USS( 6)  =   -47.2703200D0
         UPP( 6)  =   -36.2669180D0
         BETAS( 6)=   -11.9100150D0
         BETAP( 6)=    -9.8027550D0
         ZS( 6)   =     1.5650850D0
         ZP( 6)   =     1.8423450D0
         ALP( 6)  =     2.7078070D0
         EISOL( 6)=  -111.2299170D0
         GSS( 6)  =    11.2007080D0
         GSP( 6)  =    10.2650270D0
         GPP( 6)  =    10.7962920D0
         GP2( 6)  =     9.0425660D0
         HSP( 6)  =     2.2909800D0
         DD(2, 6) =     0.8332396D0
         QQ( 6)   =     0.6647750D0
         AM( 6)   =     0.4116394D0
         AD( 6)   =     0.5885862D0
         AQ( 6)   =     0.7647667D0
         GUESS1(1, 6)=  0.0501070D0
         GUESS2(1, 6)=  6.0031650D0
         GUESS3(1, 6)=  1.6422140D0
         GUESS1(2, 6)=  0.0507330D0
         GUESS2(2, 6)=  6.0029790D0
         GUESS3(2, 6)=  0.8924880D0
      end if  ! C

      ! N : NITROGEN (7)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 99, 4899 (1977)
      !       correction of Eisol and AQ values according to 
      !       J.J.P. STEWART, QCPE BULL. 9, 80 (1989).
         USS( 7)  =   -71.9321220D0
         UPP( 7)  =   -57.1723190D0
         BETAS( 7)=   -20.4957580D0
         BETAP( 7)=   -20.4957580D0
         ZS( 7)   =     2.2556140D0
         ZP( 7)   =     2.2556140D0
         ALP( 7)  =     2.8613420D0
         EISOL( 7)=  -202.5662010D0 ! note correction
         GSS( 7)  =    13.5900000D0
         GSP( 7)  =    12.6600000D0
         GPP( 7)  =    12.9800000D0
         GP2( 7)  =    11.5900000D0
         HSP( 7)  =     3.1400000D0
         DD(2, 7) =     0.6399037D0
         QQ( 7)   =     0.5429763D0
         AM( 7)   =     0.4994487D0
         AD( 7)   =     0.7843643D0
         AQ( 7)   =     0.8126445D0 ! note correction
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909 (1985)
         USS( 7)  =   -71.8600000D0
         UPP( 7)  =   -57.1675810D0
         BETAS( 7)=   -20.2991100D0
         BETAP( 7)=   -18.2386660D0
         ZS( 7)   =     2.3154100D0
         ZP( 7)   =     2.1579400D0
         ALP( 7)  =     2.9472860D0
         EISOL( 7)=  -202.4077430D0
         GSS( 7)  =    13.5900000D0
         GSP( 7)  =    12.6600000D0
         GPP( 7)  =    12.9800000D0
         GP2( 7)  =    11.5900000D0
         HSP( 7)  =     3.1400000D0
         DD(2, 7) =     0.6433247D0
         QQ( 7)   =     0.5675528D0
         AM( 7)   =     0.4994487D0
         AD( 7)   =     0.7820840D0
         AQ( 7)   =     0.7883498D0
         GUESS1(1, 7)=  0.0252510D0
         GUESS2(1, 7)=  5.0000000D0
         GUESS3(1, 7)=  1.5000000D0
         GUESS1(2, 7)=  0.0289530D0
         GUESS2(2, 7)=  5.0000000D0
         GUESS3(2, 7)=  2.1000000D0
         GUESS1(3, 7)= -0.0058060D0
         GUESS2(3, 7)=  2.0000000D0
         GUESS3(3, 7)=  2.4000000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS( 7)  =   -49.3356720D0
         UPP( 7)  =   -47.5097360D0
         BETAS( 7)=   -14.0625210D0
         BETAP( 7)=   -20.0438480D0
         ZS( 7)   =     2.0280940D0
         ZP( 7)   =     2.3137280D0
         ALP( 7)  =     2.8305450D0
         EISOL( 7)=  -157.6137755D0
         GSS( 7)  =    11.9047870D0
         GSP( 7)  =     7.3485650D0
         GPP( 7)  =    11.7546720D0
         GP2( 7)  =    10.8072770D0
         HSP( 7)  =     1.1367130D0
         DD(2, 7) =     0.6577006D0
         QQ( 7)   =     0.5293383D0
         AM( 7)   =     0.4375151D0
         AD( 7)   =     0.5030995D0
         AQ( 7)   =     0.7364933D0
         GUESS1(1, 7)=  1.5016740D0
         GUESS2(1, 7)=  5.9011480D0
         GUESS3(1, 7)=  1.7107400D0
         GUESS1(2, 7)= -1.5057720D0
         GUESS2(2, 7)=  6.0046580D0
         GUESS3(2, 7)=  1.7161490D0
      end if  ! N

      ! O : OXYGEN (8)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 99, 4899 (1977)
         USS( 8)  =   -99.6443090D0
         UPP( 8)  =   -77.7974720D0
         BETAS( 8)=   -32.6880820D0
         BETAP( 8)=   -32.6880820D0
         ZS( 8)   =     2.6999050D0
         ZP( 8)   =     2.6999050D0
         ALP( 8)  =     3.1606040D0
         EISOL( 8)=  -317.8685060D0
         GSS( 8)  =    15.4200000D0
         GSP( 8)  =    14.4800000D0
         GPP( 8)  =    14.5200000D0
         GP2( 8)  =    12.9800000D0
         HSP( 8)  =     3.9400000D0
         DD(2, 8) =     0.5346024D0
         QQ( 8)   =     0.4536252D0
         AM( 8)   =     0.5667034D0
         AD( 8)   =     0.9592562D0
         AQ( 8)   =     0.9495934D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909 (1985)
         if(QSRP_PhoT) then
            ! AM1/d-PhoT parameters: Nam et al., JTCT (2007).
            USS( 8)  =   -96.7606758139D0
            UPP( 8)  =   -78.7762028030D0
            BETAS( 8)=   -29.4723064159D0
            BETAP( 8)=   -28.5157850071D0
            ZS( 8)   =     3.0579653594D0
            ZP( 8)   =     2.5153323616D0
            ALP( 8)  =     4.4044172319D0
            EISOL( 8)=  -316.0995200D0   ! should be recomputed.
            GSS( 8)  =    14.2347142713D0
            GSP( 8)  =    14.5394514107D0
            GPP( 8)  =    14.4545302760D0
            GP2( 8)  =    12.9422587063D0
            HSP( 8)  =     4.3397050455D0
            DD(2, 8) =     0.4988896D0
            QQ( 8)   =     0.4852322D0
            AM( 8)   =     0.5667034D0
            AD( 8)   =     0.9961066D0
            AQ( 8)   =     0.9065223D0
            GUESS1(1, 8)=  0.2885257249D0
            GUESS2(1, 8)=  4.8832649364D0
            GUESS3(1, 8)=  0.8509098899D0
            GUESS1(2, 8)=  0.0615855240D0
            GUESS2(2, 8)=  4.4357914865D0
            GUESS3(2, 8)=  1.3536807741D0
         else
            ! original AM1 parameters.
            USS( 8)  =   -97.8300000D0
            UPP( 8)  =   -78.2623800D0
            BETAS( 8)=   -29.2727730D0
            BETAP( 8)=   -29.2727730D0
            ZS( 8)   =     3.1080320D0
            ZP( 8)   =     2.5240390D0
            ALP( 8)  =     4.4553710D0
            EISOL( 8)=  -316.0995200D0
            GSS( 8)  =    15.4200000D0
            GSP( 8)  =    14.4800000D0
            GPP( 8)  =    14.5200000D0
            GP2( 8)  =    12.9800000D0
            HSP( 8)  =     3.9400000D0
            DD(2, 8) =     0.4988896D0
            QQ( 8)   =     0.4852322D0
            AM( 8)   =     0.5667034D0
            AD( 8)   =     0.9961066D0
            AQ( 8)   =     0.9065223D0
            GUESS1(1, 8)=  0.2809620D0
            GUESS2(1, 8)=  5.0000000D0
            GUESS3(1, 8)=  0.8479180D0
            GUESS1(2, 8)=  0.0814300D0
            GUESS2(2, 8)=  7.0000000D0
            GUESS3(2, 8)=  1.4450710D0
         end if
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS( 8)  =   -86.9930020D0
         UPP( 8)  =   -71.8795800D0
         BETAS( 8)=   -45.2026510D0
         BETAP( 8)=   -24.7525150D0
         ZS( 8)   =     3.7965440D0
         ZP( 8)   =     2.3894020D0
         ALP( 8)  =     3.2171020D0
         EISOL( 8)=  -289.3422065D0
         GSS( 8)  =    15.7557600D0
         GSP( 8)  =    10.6211600D0
         GPP( 8)  =    13.6540160D0
         GP2( 8)  =    12.4060950D0
         HSP( 8)  =     0.5938830D0
         DD(2, 8) =     0.4086173D0
         QQ( 8)   =     0.5125738D0
         AM( 8)   =     0.5790430D0
         AD( 8)   =     0.5299630D0
         AQ( 8)   =     0.8179630D0
         GUESS1(1, 8)= -1.1311280D0
         GUESS2(1, 8)=  6.0024770D0
         GUESS3(1, 8)=  1.6073110D0
         GUESS1(2, 8)=  1.1378910D0
         GUESS2(2, 8)=  5.9505120D0
         GUESS3(2, 8)=  1.5983950D0
      end if  ! O

      ! F : FLUORINE (9)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: M.J.S. DEWAR, H.S. RZEPA, J. AM. CHEM. SOC., 100, 777 (1978)
      ! 
      ! parameters changed to those of our alpha carbon!! (JJ 4-6-2000)
         USS( 9)  =   -90.6472587D0
         UPP( 9)  =   -81.0830636D0
         BETAS( 9)=   -21.5843170D0
         BETAP( 9)=   -19.6188751D0
         ZS( 9)   =     1.7394708D0
         ZP( 9)   =     1.8229945D0
         ALP( 9)  =     3.3242151D0
         EISOL( 9)=  -476.6837810D0
         GSS( 9)  =    13.1484341D0
         GSP( 9)  =     7.1970960D0
         GPP( 9)  =    14.0091748D0
         GP2( 9)  =    13.8771665D0
         HSP( 9)  =     2.8362719D0
         DD(2, 9) =     0.5067166D0
         QQ( 9)   =     0.4299633D0
         AM( 9)   =     0.6218302D0
         AD( 9)   =     1.0850301D0
         AQ( 9)   =     1.0343643D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: also changed for our alpha carbon.
         USS( 9)  =  -129.3183911D0
         UPP( 9)  =   -67.7809091D0
         BETAS( 9)=   -21.2269728D0
         BETAP( 9)=   -21.6202115D0
         ZS( 9)   =     7.7159098D0
         ZP( 9)   =     2.0573002D0
         ALP( 9)  =     3.0489458D0
         EISOL( 9)=  -482.2905830D0
         GSS( 9)  =     8.2127799D0
         GSP( 9)  =    13.7079875D0
         GPP( 9)  =     8.0439374D0
         GP2( 9)  =     7.9765951D0
         HSP( 9)  =     2.6691495D0
         DD(2, 9) =     0.4145203D0
         QQ( 9)   =     0.4909446D0
         AM( 9)   =     0.6218302D0
         AD( 9)   =     1.2088792D0
         AQ( 9)   =     0.9449355D0
         GUESS1(1, 9)= -0.0773926D0
         GUESS2(1, 9)=  6.4977382D0
         GUESS3(1, 9)=  1.3219710D0
         GUESS1(2, 9)= -0.1149952D0
         GUESS2(2, 9)=  2.8844458D0
         GUESS3(2, 9)=  2.7051403D0
      else if(iqm_mode.eq.3) then
      ! PM3: also changed for our alpha carbon.
         USS( 9)  =  -129.5313505D0
         UPP( 9)  =   -77.4770376D0
         BETAS( 9)=   -20.5751796D0
         BETAP( 9)=   -21.5275609D0
         ZS( 9)   =     7.2862039D0
         ZP( 9)   =     1.8885075D0
         ALP( 9)  =     2.3182222D0
         EISOL( 9)=  -437.5171690D0
         GSS( 9)  =     7.5858200D0
         GSP( 9)  =    12.0814785D0
         GPP( 9)  =    10.5105306D0
         GP2( 9)  =    10.4878482D0
         HSP( 9)  =     1.1988296D0
         DD(2, 9) =     0.3125302D0
         QQ( 9)   =     0.4916328D0
         AM( 9)   =     0.3857650D0
         AD( 9)   =     0.6768503D0
         AQ( 9)   =     0.6120047D0
         GUESS1(1, 9)= -0.0699312D0
         GUESS2(1, 9)=  1.3241576D0
         GUESS3(1, 9)=  2.7556088D0
         GUESS1(2, 9)= -0.2455150D0
         GUESS2(2, 9)=  8.0997588D0
         GUESS3(2, 9)=  1.2999394D0
      end if  ! F

      ! original MNDO F parameters
      !
      ! USS( 9)  =   -131.0715480D0
      ! UPP( 9)  =   -105.7821370D0
      ! BETAS( 9)=    -48.2904660D0
      ! BETAP( 9)=    -36.5085400D0
      ! ZS( 9)   =      2.8484870D0
      ! ZP( 9)   =      2.8484870D0
      ! ALP( 9)  =      3.4196606D0
      ! EISOL( 9)=   -476.6837810D0
      ! GSS( 9)  =     16.9200000D0
      ! GSP( 9)  =     17.2500000D0
      ! GPP( 9)  =     16.7100000D0
      ! GP2( 9)  =     14.9100000D0
      ! HSP( 9)  =      4.8300000D0
      ! DD(2, 9) =      0.5067166D0
      ! QQ( 9)   =      0.4299633D0
      ! AM( 9)   =      0.6218302D0
      ! AD( 9)   =      1.0850301D0
      ! AQ( 9)   =      1.0343643D0

      ! original AM1 F parameters
      ! M.J.S.DEWAR AND E.G.ZOEBISCH, J.MOL.STRUCT. 180, 1 (1988)
      ! USS( 9)  =   -136.1055790D0
      ! UPP( 9)  =   -104.8898850D0
      ! BETAS( 9)=    -69.5902770D0
      ! BETAP( 9)=    -27.9223600D0
      ! ZS( 9)   =      3.7700820D0
      ! ZP( 9)   =      2.4946700D0
      ! ALP( 9)  =      5.5178000D0
      ! EISOL( 9)=   -482.2905830D0
      ! GSS( 9)  =     16.9200000D0
      ! GSP( 9)  =     17.2500000D0
      ! GPP( 9)  =     16.7100000D0
      ! GP2( 9)  =     14.9100000D0
      ! HSP( 9)  =      4.8300000D0
      ! DD(2, 9) =      0.4145203D0
      ! QQ( 9)   =      0.4909446D0
      ! AM( 9)   =      0.6218302D0
      ! AD( 9)   =      1.2088792D0
      ! AQ( 9)   =      0.9449355D0
      ! GUESS1(1, 9)=   0.2420790D0
      ! GUESS2(1, 9)=   4.8000000D0
      ! GUESS3(1, 9)=   0.9300000D0
      ! GUESS1(2, 9)=   0.0036070D0
      ! GUESS2(2, 9)=   4.6000000D0
      ! GUESS3(2, 9)=   1.6600000D0

      ! original PM3 F parameters
      ! J. J. P. STEWART, J. COMP. CHEM.  10, 209 (1989)
      ! USS( 9)  =   -110.4353030D0
      ! 
      ! BETAS( 9)=    -48.4059390D0
      ! BETAP( 9)=    -27.7446600D0
      ! ZS( 9)   =      4.7085550D0
      ! ZP( 9)   =      2.4911780D0
      ! ALP( 9)  =      3.3589210D0
      ! EISOL( 9)=   -437.5171690D0
      ! GSS( 9)  =     10.4966670D0
      ! GSP( 9)  =     16.0736890D0
      ! GPP( 9)  =     14.8172560D0
      ! GP2( 9)  =     14.4183930D0
      ! HSP( 9)  =      0.7277630D0
      ! DD(2, 9) =      0.3125302D0
      ! QQ( 9)   =      0.4916328D0
      ! AM( 9)   =      0.3857650D0
      ! AD( 9)   =      0.6768503D0
      ! AQ( 9)   =      0.6120047D0
      ! GUESS1(1, 9)=  -0.0121660D0
      ! GUESS2(1, 9)=   6.0235740D0
      ! GUESS3(1, 9)=   1.8568590D0
      ! GUESS1(2, 9)=  -0.0028520D0
      ! GUESS2(2, 9)=   6.0037170D0
      ! GUESS3(2, 9)=   2.6361580D0

      ! Na: SODIUM (11)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d (sp):  W.THIEL AND A.A.VOITYUK, J. PHYS. CHEM., 100, 616 (1996)
         USS(11)  =    -5.20100000D0
         UPP(11)  =    -2.71257317D0
         ZS(11)   =     0.98750834D0
         ZP(11)   =     0.89334983D0
         BETAS(11)=    -1.08738166D0
         BETAP(11)=    -0.48623935D0
         ALP(11)  =     1.17010202D0
         GSS(11)  =     4.59444476D0
         GPP(11)  =     4.29919761D0
         GSP(11)  =     4.14757400D0
         GP2(11)  =     3.79695732D0
         HSP(11)  =     0.53440874D0
         ZSN(11)  =     0.65411258D0
         ZPN(11)  =     0.56440874D0
         PO(9,11) =     1.53055325D0
         F0SD(11) =     0.00000000D0
         G2SD(11) =     0.00000000D0
      end if  ! Na

      ! Mg: MAGNESIUM (12)
      if(iqm_mode.eq.1) then
      ! MNDO: A.A.VOITYUK, ZHURNAL STRUKTURNOI KHIMII 28, 128 (1987)
         USS(12)  =   -15.0400000D0
         UPP(12)  =    -9.2640000D0
         BETAS(12)=    -2.5860000D0
         BETAP(12)=    -2.8420000D0
         ZS(12)   =     1.0490000D0
         ZP(12)   =     0.8890000D0
         ALP(12)  =     1.8130000D0
         EISOL(12)=   -22.6900000D0
         GSS(12)  =     7.3900000D0
         GSP(12)  =     6.5700000D0
         GPP(12)  =     6.6800000D0
         GP2(12)  =     5.9000000D0
         HSP(12)  =     0.8200000D0
         DD(2,12) =     2.0360000D0
         QQ(12)   =     1.8820000D0
         AM(12)   =     0.2715915D0
         AD(12)   =     0.2269632D0
         AQ(12)   =     0.2925688D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: M.C. HUTTER, J.R. REIMERS, N.S. HUSH, J.PHYS.CHEM. (1998)
         USS(12)  =   -14.96959313D0
         UPP(12)  =   -11.56229248D0
         BETAS(12)=    -1.25974355D0
         BETAP(12)=    -0.77836604D0
         ZS(12)   =     1.22339270D0
         ZP(12)   =     1.02030798D0
         ALP(12)  =     1.67049799D0
         EISOL(12)=   -22.43786349D0
         GSS(12)  =     7.50132277D0
         GSP(12)  =     6.34591536D0
         GPP(12)  =     4.77534467D0
         GP2(12)  =     4.34017279D0
         HSP(12)  =     0.48930466D0
         DD(2,12) =     1.75012115D0
         QQ(12)   =     1.64001467D0
         AM(12)   =     0.27568257D0
         AD(12)   =     0.19957236D0
         AQ(12)   =     0.26440152D0
         GUESS1(1,12)=  2.55017735D0
         GUESS2(1,12)=  4.29397225D0
         GUESS3(1,12)=  0.79989601D0
         GUESS1(2,12)= -0.00565806D0
         GUESS2(2,12)=  2.96053910D0
         GUESS3(2,12)=  1.47499983D0
         GUESS1(3,12)= -0.00610286D0
         GUESS2(3,12)=  2.61416919D0
         GUESS3(3,12)=  2.42604040D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(12)  =   -14.6236880D0
         UPP(12)  =   -14.1734600D0
         BETAS(12)=    -2.0716910D0
         BETAP(12)=    -0.5695810D0
         ZS(12)   =     0.6985520D0
         ZP(12)   =     1.4834530D0
         ALP(12)  =     1.3291470D0
         EISOL(12)=   -22.5530760D0
         GSS(12)  =     6.6943000D0
         GSP(12)  =     6.7939950D0
         GPP(12)  =     6.9104460D0
         GP2(12)  =     7.0908230D0
         HSP(12)  =     0.5433000D0
         DD(2,12) =     1.1403950D0
         QQ(12)   =     1.1279899D0
         AM(12)   =     0.2460235D0
         AD(12)   =     0.2695751D0
         AQ(12)   =     0.2767522D0
         GUESS1(1,12)=  2.1170500D0
         GUESS2(1,12)=  6.0094770D0
         GUESS3(1,12)=  2.0844060D0
         GUESS1(2,12)= -2.5477670D0
         GUESS2(2,12)=  4.3953700D0
         GUESS3(2,12)=  2.0636740D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d (sp): W.THIEL AND A.A.VOITYUK, J. PHYS. CHEM., 100, 616 (1996)
         USS(12)  =   -15.09700000D0
         UPP(12)  =   -10.65000000D0
         ZS(12)   =     1.44890446D0
         ZP(12)   =     0.95293002D0
         BETAS(12)=    -1.89588355D0
         BETAP(12)=    -2.14108943D0
         ALP(12)  =     1.62146984D0
         GSS(12)  =     7.37513258D0
         GPP(12)  =     7.04795383D0
         GSP(12)  =     6.88890741D0
         GP2(12)  =     6.22459871D0
         HSP(12)  =     0.72673390D0
         ZSN(12)  =     1.05000000D0
         ZPN(12)  =     0.92527190D0
         PO(9,12) =     1.35077620D0
         F0SD(12) =     0.00000000D0
         G2SD(12) =     0.00000000D0
      end if  ! Mg

      ! Al: ALUMINUM (13)
      if(iqm_mode.eq.1) then
      ! MNDO: L.P. DAVIS, ET.AL.  J. COMP. CHEM., 2, 433 (1981) SEE MANUAL.
      !
      !       The monocentric integrals HSP and GSP for Al are only estimates.
      !       A value of G1 for Al is needed to resolve Olearis integrals.
         USS(13)  =   -23.8070970D0
         UPP(13)  =   -17.5198780D0
         BETAS(13)=    -2.6702840D0
         BETAP(13)=    -2.6702840D0
         ZS(13)   =     1.4441610D0
         ZP(13)   =     1.4441610D0
         ALP(13)  =     1.8688394D0
         EISOL(13)=   -44.4840720D0
         GSS(13)  =     8.0900000D0
         GSP(13)  =     6.6300000D0
         GPP(13)  =     5.9800000D0
         GP2(13)  =     5.4000000D0
         HSP(13)  =     0.7000000D0
         DD(2,13) =     1.3992387D0
         QQ(13)   =     1.1586797D0
         AM(13)   =     0.2973172D0
         AD(13)   =     0.2635574D0
         AQ(13)   =     0.3673560D0
      else if(iqm_mode.eq.2) then
      ! AM1: M. J. S. Dewar, A. J. Holder, Organometallics, 9, 508-511 (1990)
         USS(13)  =   -24.3535850D0
         UPP(13)  =   -18.3636450D0
         BETAS(13)=    -3.8668220D0
         BETAP(13)=    -2.3171460D0
         ZS(13)   =     1.5165930D0
         ZP(13)   =     1.3063470D0
         ALP(13)  =     1.9765860D0
         EISOL(13)=   -46.4208150D0
         GSS(13)  =     8.0900000D0
         GSP(13)  =     6.6300000D0
         GPP(13)  =     5.9800000D0
         GP2(13)  =     5.4000000D0
         HSP(13)  =     0.7000000D0
         DD(2,13) =     1.4040443D0
         QQ(13)   =     1.2809154D0
         AM(13)   =     0.2973172D0
         AD(13)   =     0.2630229D0
         AQ(13)   =     0.3427832D0
         GUESS1(1,13)=  0.0900000D0
         GUESS2(1,13)= 12.3924430D0
         GUESS3(1,13)=  2.0503940D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS(13)  =   -24.8454040D0
         UPP(13)  =   -22.2641590D0
         BETAS(13)=    -0.5943010D0
         BETAP(13)=    -0.9565500D0
         ZS(13)   =     1.7028880D0
         ZP(13)   =     1.0736290D0
         ALP(13)  =     1.5217030D0
         EISOL(13)=   -46.8647630D0
         GSS(13)  =     5.7767370D0
         GSP(13)  =    11.6598560D0
         GPP(13)  =     6.3477900D0
         GP2(13)  =     6.1210770D0
         HSP(13)  =     4.0062450D0
         DD(2,13) =     1.2102799D0
         QQ(13)   =     1.5585645D0
         AM(13)   =     0.2123020D0
         AD(13)   =     0.6418584D0
         AQ(13)   =     0.2262838D0
         GUESS1(1,13)= -0.4730900D0
         GUESS2(1,13)=  1.9158250D0
         GUESS3(1,13)=  1.4517280D0
         GUESS1(2,13)= -0.1540510D0
         GUESS2(2,13)=  6.0050860D0
         GUESS3(2,13)=  2.5199970D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, J. PHYS. CHEM., 100, 616 (1996)
         USS(13)  =   -24.01792910D0
         UPP(13)  =   -20.79597967D0
         ZS(13)   =     1.79402273D0
         ZP(13)   =     1.37130919D0
         BETAS(13)=    -7.10185851D0
         BETAP(13)=    -2.31809618D0
         ALP(13)  =     1.44301676D0
         GSS(13)  =     8.58671016D0
         GPP(13)  =     8.32495723D0
         GSP(13)  =     7.66469306D0
         GP2(13)  =     7.35242020D0
         HSP(13)  =     0.54401293D0
         UDD(13)  =    -5.22082737D0
         ZD(13)   =     0.80591133D0
         BETAD(13)=    -3.35638545D0
         ZSN(13)  =     1.22249269D0
         ZPN(13)  =     1.09291990D0
         ZDN(13)  =     0.80038285D0
         PO(9,13) =     1.58442520D0
         F0SD(13) =     0.00000000D0
         G2SD(13) =     0.00000000D0
      end if  ! Al

      ! Si: SILICON (14)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR, ET. AL. ORGANOMETALLICS  5, 375 (1986)
         USS(14)  =   -37.0375330D0
         UPP(14)  =   -27.7696780D0
         BETAS(14)=    -9.0868040D0
         BETAP(14)=    -1.0758270D0
         ZS(14)   =     1.3159860D0
         ZP(14)   =     1.7099430D0
         ALP(14)  =     2.2053160D0
         EISOL(14)=   -82.8394220D0
         GSS(14)  =     9.8200000D0
         GSP(14)  =     8.3600000D0
         GPP(14)  =     7.3100000D0
         GP2(14)  =     6.5400000D0
         HSP(14)  =     1.3200000D0
         DD(2,14) =     1.2580349D0
         QQ(14)   =     0.9785824D0
         AM(14)   =     0.3608967D0
         AD(14)   =     0.3664244D0
         AQ(14)   =     0.4506740D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S.DEWAR, C. JIE, ORGANOMETALLICS, 6, 1486-1490 (1987)
         USS(14)  =   -33.9536220D0
         UPP(14)  =   -28.9347490D0
         BETAS(14)=    -3.784852D0
         BETAP(14)=    -1.968123D0
         ZS(14)   =     1.830697D0
         ZP(14)   =     1.2849530D0
         ALP(14)  =     2.257816D0
         EISOL(14)=   -79.0017420D0
         GSS(14)  =     9.8200000D0
         GSP(14)  =     8.3600000D0
         GPP(14)  =     7.3100000D0
         GP2(14)  =     6.5400000D0
         HSP(14)  =     1.3200000D0
         DD(2,14) =     1.1631107D0
         QQ(14)   =     1.3022422D0
         AM(14)   =     0.3608967D0
         AD(14)   =     0.3829813D0
         AQ(14)   =     0.3712106D0
         GUESS1(1,14)=  0.25D0
         GUESS2(1,14)=  9.000D0
         GUESS3(1,14)=  0.911453D0
         GUESS1(2,14)=  0.061513D0
         GUESS2(2,14)=  5.00D0
         GUESS3(2,14)=  1.995569D0
         GUESS1(3,14)=  0.0207890D0
         GUESS2(3,14)=  5.00D0
         GUESS3(3,14)=  2.990610D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
      !      Correction, seel QCPE bulletin 10, 34 (1990)
         USS(14)  =   -26.7634830D0
         UPP(14)  =   -22.8136350D0
         BETAS(14)=    -2.8621450D0
         BETAP(14)=    -3.9331480D0
         ZS(14)   =     1.6350750D0
         ZP(14)   =     1.3130880D0
         ALP(14)  =     2.1358090D0
         EISOL(14)=   -67.7882140D0
         GSS(14)  =     5.0471960D0
         GSP(14)  =     5.9490570D0
         GPP(14)  =     6.7593670D0
         GP2(14)  =     5.1612970D0
         HSP(14)  =     0.9198320D0
         DD(2,14) =     1.3144550D0
         QQ(14)   =     1.2743396D0
         AM(14)   =     0.1854905D0
         AD(14)   =     0.3060715D0
         AQ(14)   =     0.4877432D0
         GUESS1(1,14)= -0.3906000D0
         GUESS2(1,14)=  6.0000540D0
         GUESS3(1,14)=  0.6322620D0
         GUESS1(2,14)=  0.0572590D0
         GUESS2(2,14)=  6.0071830D0
         GUESS3(2,14)=  2.0199870D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, J. MOL. STRUCT., 313, 141 (1994)
         USS(14)  =   -36.05153000D0
         UPP(14)  =   -27.53569100D0
         ZS(14)   =     1.91565460D0
         ZP(14)   =     1.68161130D0
         BETAS(14)=    -8.21073420D0
         BETAP(14)=    -4.88462030D0
         ALP(14)  =     1.66006930D0
         GSS(14)  =    10.74164700D0
         GPP(14)  =     7.43649690D0
         GSP(14)  =     7.56066640D0
         GP2(14)  =     6.56775150D0
         HSP(14)  =     0.87753880D0
         UDD(14)  =   -14.67743900D0
         ZD(14)   =     0.96677166D0
         BETAD(14)=    -2.60801150D0
         ZSN(14)  =     1.52929180D0
         ZPN(14)  =     0.97628075D0
         ZDN(14)  =     0.93816441D0
         PO(9,14) =     1.26656550D0
         F0SD(14) =     0.00000000D0
         G2SD(14) =     0.00000000D0
      end if  ! Si

      ! P : PHOSPHORUS (15)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR, M.L.MCKEE, H.S.RZEPA,J. AM. CHEM. SOC., 100 3607 1978
         USS(15)  =   -56.1433600D0
         UPP(15)  =   -42.8510800D0
         BETAS(15)=    -6.7916000D0
         BETAP(15)=    -6.7916000D0
         ZS(15)   =     2.1087200D0
         ZP(15)   =     1.7858100D0
         ALP(15)  =     2.4152800D0
         EISOL(15)=  -152.9599600D0
         GSS(15)  =    11.5600000D0
         GSP(15)  =    10.0800000D0
         GPP(15)  =     8.6400000D0
         GP2(15)  =     7.6800000D0
         HSP(15)  =     1.9200000D0
         DD(2,15) =     1.0129699D0
         QQ(15)   =     0.9370090D0
         AM(15)   =     0.4248438D0
         AD(15)   =     0.4882420D0
         AQ(15)   =     0.4979406D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S.DEWAR AND C.JIE, J.MOL.STRUCT. 187, 1 (1989)
         USS(15)  =   -42.0298630D0
         UPP(15)  =   -34.0307090D0
         BETAS(15)=   - 6.3537640D0
         BETAP(15)=    -6.5907090D0
         ZS(15)   =     1.9812800D0
         ZP(15)   =     1.8751500D0
         ALP(15)  =     2.4553220D0
         EISOL(15)=  -124.4368355D0
         GSS(15)  =    11.5600050D0
         GSP(15)  =     5.2374490D0
         GPP(15)  =     7.8775890D0
         GP2(15)  =     7.3076480D0
         HSP(15)  =     0.7792380D0
         DD(2,15) =     1.0452022D0
         QQ(15)   =     0.8923660D0
         AM(15)   =     0.4248440D0
         AD(15)   =     0.3275319D0
         AQ(15)   =     0.4386854D0
         GUESS1(1,15)= -0.0318270D0
         GUESS2(1,15)=  6.0000000D0
         GUESS3(1,15)=  1.4743230D0
         GUESS1(2,15)=  0.0184700D0
         GUESS2(2,15)=  7.0000000D0
         GUESS3(2,15)=  1.7793540D0
         GUESS1(3,15)=  0.0332900D0
         GUESS2(3,15)=  9.0000000D0
         GUESS3(3,15)=  3.0065760D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS(15)  =   -40.4130960D0
         UPP(15)  =   -29.5930520D0
         BETAS(15)=   -12.6158790D0
         BETAP(15)=    -4.1600400D0
         ZS(15)   =     2.0175630D0
         ZP(15)   =     1.5047320D0
         ALP(15)  =     1.9405340D0
         EISOL(15)=  -117.9591740D0
         GSS(15)  =     7.8016150D0
         GSP(15)  =     5.1869490D0
         GPP(15)  =     6.6184780D0
         GP2(15)  =     6.0620020D0
         HSP(15)  =     1.5428090D0
         DD(2,15) =     1.0644947D0
         QQ(15)   =     1.1120386D0
         AM(15)   =     0.2867187D0
         AD(15)   =     0.4309446D0
         AQ(15)   =     0.3732517D0
         GUESS1(1,15)= -0.6114210D0
         GUESS2(1,15)=  1.9972720D0
         GUESS3(1,15)=  0.7946240D0
         GUESS1(2,15)= -0.0939350D0
         GUESS2(2,15)=  1.9983600D0
         GUESS3(2,15)=  1.9106770D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, J. PHYS. CHEM., 100, 616 (1996)
         USS(15)  =   -47.05552900D0
         UPP(15)  =   -38.06705900D0
         ZS(15)   =     2.26646290D0
         ZP(15)   =     1.94001490D0
         BETAS(15)=    -8.90210430D0
         BETAP(15)=    -9.38611080D0
         ALP(15)  =     1.85255120D0
         GSS(15)  =    11.47975300D0
         GPP(15)  =     8.24872280D0
         GSP(15)  =     8.55756910D0
         GP2(15)  =     7.28509170D0
         HSP(15)  =     2.10780440D0
         UDD(15)  =   -23.69159700D0
         ZD(15)   =     1.10010900D0
         BETAD(15)=    -2.09170080D0
         ZSN(15)  =     1.63437610D0
         ZPN(15)  =     1.08291170D0
         ZDN(15)  =     1.00651470D0
         PO(9,15) =     1.18513000D0
         F0SD(15) =     0.00000000D0
         G2SD(15) =     0.00000000D0
      else if(iqm_mode.eq.4) then
      ! AM1/d : XL, MNDO/d parameters
      !         Need to update later?
         if(QSRP_PhoT) then
            ! AM1/d-PhoT parameters: Nam et al., JTCT (2007).
            USS(15)  =   -46.2508099256D0
            UPP(15)  =   -40.7129183620D0
            ZS(15)   =     1.9091683752D0
            ZP(15)   =     2.0084656144D0
            BETAS(15)=   -11.1947908343D0
            BETAP(15)=   -11.9856214630D0
            ALP(15)  =     1.8832372396D0
            GSS(15)  =    14.6457470000D0
            GPP(15)  =    11.6949180000D0
            GSP(15)  =     5.6896540860D0
            GP2(15)  =    10.3286960000D0
            HSP(15)  =     1.1751150815D0
            UDD(15)  =   -24.5041611320D0
            ZD(15)   =     0.8406673931D0
            BETAD(15)=    -2.3600953876D0
            ZSN(15)  =     2.0851197691D0
            ZPN(15)  =     1.5353362726D0
            ZDN(15)  =     1.2362658508D0
            PO(9,15) =     1.18513000D0
            F0SD(15) =     0.00000000D0
            G2SD(15) =     0.00000000D0
            GUESS1(1,15)= -0.3445287142D0
            GUESS2(1,15)=  3.0349333865D0
            GUESS3(1,15)=  1.1342748299D0
            GUESS1(2,15)= -0.0218468510D0
            GUESS2(2,15)=  1.6845146579D0
            GUESS3(2,15)=  2.7166835655D0
            GUESS1(3,15)= -0.0360032056D0
            GUESS2(3,15)=  5.2433569450D0
            GUESS3(3,15)=  1.9241748889D0
         else 
            ! XL, MNDO/d paramters? No, it is from MNDO/d P parameters.
            USS(15)  =   -47.05552900D0
            UPP(15)  =   -38.06705900D0
            ZS(15)   =     2.26646290D0
            ZP(15)   =     1.94001490D0
            BETAS(15)=    -8.90210430D0
            BETAP(15)=    -9.38611080D0
            ALP(15)  =     1.85255120D0
            GSS(15)  =    11.47975300D0
            GPP(15)  =     8.24872280D0      
            GSP(15)  =     8.55756910D0
            GP2(15)  =     7.28509170D0
            HSP(15)  =     2.10780440D0
            UDD(15)  =   -23.69159700D0
            ZD(15)   =     1.10010900D0
            BETAD(15)=    -2.09170080D0
            ZSN(15)  =     1.63437610D0
            ZPN(15)  =     1.08291170D0
            ZDN(15)  =     1.00651470D0
            PO(9,15) =     1.18513000D0
            F0SD(15) =     0.00000000D0
            G2SD(15) =     0.00000000D0
            GUESS1(1,15)=  0.00000000D0
            GUESS2(1,15)=  0.00000000D0
            GUESS3(1,15)=  0.00000000D0
            GUESS1(2,15)=  0.00000000D0
            GUESS2(2,15)=  0.00000000D0
            GUESS3(2,15)=  0.00000000D0
            GUESS1(3,15)=  0.00000000D0
            GUESS2(3,15)=  0.00000000D0
            GUESS3(3,15)=  0.00000000D0
         end if
      end if  ! P

      ! S : SULFUR (16)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR, C.H. REYNOLDS, J. COMP. CHEM. 7, 140-143 (1986)
         USS(16)  =   -72.2422810D0
         UPP(16)  =   -56.9732070D0
         BETAS(16)=   -10.7616700D0
         BETAP(16)=   -10.1084330D0
         ZS(16)   =     2.3129620D0
         ZP(16)   =     2.0091460D0
         ALP(16)  =     2.4780260D0
         EISOL(16)=  -226.0123900D0
         GSS(16)  =    12.8800000D0
         GSP(16)  =    11.2600000D0
         GPP(16)  =     9.9000000D0
         GP2(16)  =     8.8300000D0
         HSP(16)  =     2.2600000D0
         DD(2,16) =     0.9189935D0
         QQ(16)   =     0.8328514D0
         AM(16)   =     0.4733554D0
         AD(16)   =     0.5544502D0
         AQ(16)   =     0.5585244D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S.DEWAR, Y-C YUAN, THEOCHEM, IN PRESS
         USS(16)  =   -56.6940560D0
         UPP(16)  =   -48.7170490D0
         BETAS(16)=    -3.9205660D0
         BETAP(16)=    -7.9052780D0
         ZS(16)   =     2.3665150D0
         ZP(16)   =     1.6672630D0
         ALP(16)  =     2.4616480D0
         EISOL(16)=  -191.7321930D0
         GSS(16)  =    11.7863290D0
         GSP(16)  =     8.6631270D0
         GPP(16)  =    10.0393080D0
         GP2(16)  =     7.7816880D0
         HSP(16)  =     2.5321370D0
         DD(2,16) =     0.9004265D0
         QQ(16)   =     1.0036329D0
         AM(16)   =     0.4331617D0
         AD(16)   =     0.5907115D0
         AQ(16)   =     0.6454943D0
         GUESS1(1,16)= -0.5091950D0
         GUESS2(1,16)=  4.5936910D0
         GUESS3(1,16)=  0.7706650D0
         GUESS1(2,16)= -0.0118630D0
         GUESS2(2,16)=  5.8657310D0
         GUESS3(2,16)=  1.5033130D0
         GUESS1(3,16)=  0.0123340D0
         GUESS2(3,16)= 13.5573360D0
         GUESS3(3,16)=  2.0091730D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS(16)  =   -49.8953710D0
         UPP(16)  =   -44.3925830D0
         BETAS(16)=    -8.8274650D0
         BETAP(16)=    -8.0914150D0
         ZS(16)   =     1.8911850D0
         ZP(16)   =     1.6589720D0
         ALP(16)  =     2.2697060D0
         EISOL(16)=  -183.4537395D0
         GSS(16)  =     8.9646670D0
         GSP(16)  =     6.7859360D0
         GPP(16)  =     9.9681640D0
         GP2(16)  =     7.9702470D0
         HSP(16)  =     4.0418360D0
         DD(2,16) =     1.1214313D0
         QQ(16)   =     1.0086488D0
         AM(16)   =     0.3294622D0
         AD(16)   =     0.6679118D0
         AQ(16)   =     0.6137472D0
         GUESS1(1,16)= -0.3991910D0
         GUESS2(1,16)=  6.0006690D0
         GUESS3(1,16)=  0.9621230D0
         GUESS1(2,16)= -0.0548990D0
         GUESS2(2,16)=  6.0018450D0
         GUESS3(2,16)=  1.5799440D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, J. PHYS. CHEM., 100, 616 (1996)
         USS(16)  =   -56.88912800D0
         UPP(16)  =   -47.27474500D0
         ZS(16)   =     2.22585050D0
         ZP(16)   =     2.09970560D0
         BETAS(16)=   -10.99954500D0
         BETAP(16)=   -12.21543700D0
         ALP(16)  =     2.02305950D0
         GSS(16)  =    12.19630200D0
         GPP(16)  =     8.54023240D0
         GSP(16)  =     8.85390090D0
         GP2(16)  =     7.54254650D0
         HSP(16)  =     2.64635230D0
         UDD(16)  =   -25.09511800D0
         ZD(16)   =     1.23147250D0
         BETAD(16)=    -1.88066950D0
         ZSN(16)  =     1.73639140D0
         ZPN(16)  =     1.12118170D0
         ZDN(16)  =     1.05084670D0
         PO(9,16) =     1.11550210D0
         F0SD(16) =     0.00000000D0
         G2SD(16) =     0.00000000D0
      else if(iqm_mode.eq.4) then
      ! AM1/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !         TENTATIVE.
         USS(16)  =   -55.44716118D0
         UPP(16)  =   -46.03497016D0
         ZS(16)   =     2.17793120D0
         ZP(16)   =     2.04976942D0
         BETAS(16)=   -10.39146625D0
         BETAP(16)=   -10.61879022D0
         ALP(16)  =     2.14244515D0
         GSS(16)  =    10.45312745D0
         GPP(16)  =     8.10064777D0
         GSP(16)  =     8.22168678D0
         GP2(16)  =     7.15431500D0
         HSP(16)  =     1.96427866D0
         UDD(16)  =   -27.87023731D0
         ZD(16)   =     1.95658462D0
         BETAD(16)=    -1.44954864D0
         ZSN(16)  =     1.48821512D0
         ZPN(16)  =     1.06347203D0
         ZDN(16)  =     0.98524102D0
         PO(9,16) =     1.19319348D0
         F0SD(16) =     0.00000000D0
         G2SD(16) =     0.00000000D0
         GUESS1(1,16)= -0.02440893D0
         GUESS2(1,16)=  4.00000000D0
         GUESS3(1,16)=  1.60472556D0
         GUESS1(2,16)= -0.01886801D0
         GUESS2(2,16)=  4.00000000D0
         GUESS3(2,16)=  2.74477716D0
      end if  ! S

      ! Cl: CHLORINE (17)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR, H.S.RZEPA, J. COMP. CHEM., 4, 158 (1983)
         USS(17)  =  -100.2271660D0
         UPP(17)  =   -77.3786670D0
         BETAS(17)=   -14.2623200D0
         BETAP(17)=   -14.2623200D0
         ZS(17)   =     3.7846450D0
         ZP(17)   =     2.0362630D0
         ALP(17)  =     2.5422010D0
         EISOL(17)=  -353.1176670D0
         GSS(17)  =    15.0300000D0
         GSP(17)  =    13.1600000D0
         GPP(17)  =    11.3000000D0
         GP2(17)  =     9.9700000D0
         HSP(17)  =     2.4200000D0
         DD(2,17) =     0.4986870D0
         QQ(17)   =     0.8217603D0
         AM(17)   =     0.5523705D0
         AD(17)   =     0.8061220D0
         AQ(17)   =     0.6053435D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S.DEWAR AND E.G.ZOEBISCH, J.MOL.STRUCT. 180, 1 (1988)
         USS(17)  =  -111.6139480D0
         UPP(17)  =   -76.6401070D0
         BETAS(17)=   -24.5946700D0
         BETAP(17)=   -14.6372160D0
         ZS(17)   =     3.6313760D0
         ZP(17)   =     2.0767990D0
         ALP(17)  =     2.9193680D0
         EISOL(17)=  -372.1984310D0
         GSS(17)  =    15.0300000D0
         GSP(17)  =    13.1600000D0
         GPP(17)  =    11.3000000D0
         GP2(17)  =     9.9700000D0
         HSP(17)  =     2.4200000D0
         DD(2,17) =     0.5406286D0
         QQ(17)   =     0.8057208D0
         AM(17)   =     0.5523705D0
         AD(17)   =     0.7693200D0
         AQ(17)   =     0.6133369D0
         GUESS1(1,17)=  0.0942430D0
         GUESS2(1,17)=  4.0000000D0
         GUESS3(1,17)=  1.3000000D0
         GUESS1(2,17)=  0.0271680D0
         GUESS2(2,17)=  4.0000000D0
         GUESS3(2,17)=  2.1000000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS(17)  =  -100.6267470D0
         UPP(17)  =   -53.6143960D0
         BETAS(17)=   -27.5285600D0
         BETAP(17)=   -11.5939220D0
         ZS(17)   =     2.2462100D0
         ZP(17)   =     2.1510100D0
         ALP(17)  =     2.5172960D0
         EISOL(17)=  -315.1949480D0
         GSS(17)  =    16.0136010D0
         GSP(17)  =     8.0481150D0
         GPP(17)  =     7.5222150D0
         GP2(17)  =     7.5041540D0
         HSP(17)  =     3.4811530D0
         DD(2,17) =     0.9175856D0
         QQ(17)   =     0.7779230D0
         AM(17)   =     0.5885190D0
         AD(17)   =     0.6814522D0
         AQ(17)   =     0.3643694D0
         GUESS1(1,17)= -0.1715910D0
         GUESS2(1,17)=  6.0008020D0
         GUESS3(1,17)=  1.0875020D0
         GUESS1(2,17)= -0.0134580D0
         GUESS2(2,17)=  1.9666180D0
         GUESS3(2,17)=  2.2928910D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, INTERN. J. QUANT. CHEM., 44, 807 (1992)
         USS(17)  =   -69.62297275D0
         UPP(17)  =   -59.10072899D0
         ZS(17)   =     2.56161065D0
         ZP(17)   =     2.38933800D0
         BETAS(17)=    -6.03729165D0
         BETAP(17)=   -19.18338497D0
         ALP(17)  =     2.18030019D0
         GSS(17)  =    13.21114854D0
         GPP(17)  =     8.99620033D0
         GSP(17)  =     9.41949513D0
         GP2(17)  =     7.94524745D0
         HSP(17)  =     3.08149862D0
         UDD(17)  =   -36.67457320D0
         ZD(17)   =     1.25139777D0
         BETAD(17)=    -1.87778198D0
         ZSN(17)  =     1.88087547D0
         ZPN(17)  =     1.18104227D0
         ZDN(17)  =     1.14061555D0
         PO(9,17) =     1.02981205D0
         F0SD(17) =     0.00000000D0
         G2SD(17) =     0.00000000D0
      else if(iqm_mode.eq.4) then
      ! AM1/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !        - TENTATIVE.
         USS(17)  =   -78.06194644D0
         UPP(17)  =   -62.88144498D0
         ZS(17)   =     2.43588300D0
         ZP(17)   =     2.28114805D0
         BETAS(17)=    -7.21967299D0
         BETAP(17)=   -16.67056778D0
         ALP(17)  =     2.33169279D0
         GSS(17)  =    15.61616551D0
         GPP(17)  =     9.24562290D0
         GSP(17)  =    10.32542638D0
         GP2(17)  =     8.16553201D0
         HSP(17)  =     2.20876789D0
         UDD(17)  =   -37.02056085D0
         ZD(17)   =     1.92536920D0
         BETAD(17)=    -0.62603530D0
         ZSN(17)  =     2.22327851D0
         ZPN(17)  =     1.21378705D0
         ZDN(17)  =     0.93109041D0
         PO(9,17) =     0.83806018D0
         F0SD(17) =     0.00000000D0
         G2SD(17) =     0.00000000D0
         GUESS1(1,17)=  0.01969202D0
         GUESS2(1,17)=  4.00000000D0
         GUESS3(1,17)=  2.07239470D0
         GUESS1(2,17)= -0.02228517D0
         GUESS2(2,17)=  4.00000000D0
         GUESS3(2,17)=  2.47144384D0
      end if  ! Cl

      ! Ti: TITANIUM (22)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d: V. PAGET AND W.THIEL PRELIMINARY UNPUBLISHED PARAMETERS (10/97)
      !         - TENTATIVE.
         USS(22)  =   -28.40643395D0
         UPP(22)  =    -6.80174285D0
         ZS(22)   =     1.21819921D0
         ZP(22)   =     1.19202895D0
         BETAS(22)=   -11.14541839D0
         BETAP(22)=    -8.85425992D0
         ALP(22)  =     1.72500000D0
         GSS(22)  =     6.31685637D0
         GPP(22)  =     5.56521541D0
         GSP(22)  =     9.83852854D0
         GP2(22)  =     4.87845415D0
         HSP(22)  =     3.09763345D0
         UDD(22)  =   -30.00000000D0
         ZD(22)   =     1.90010421D0
         BETAD(22)=    -1.21038210D0
         ZSN(22)  =     1.15553216D0
         ZPN(22)  =     0.93428364D0
         ZDN(22)  =     1.25367709D0
         PO(9,22) =     1.70721098D0
         F0SD(22) =     7.60000000D0
         G2SD(22) =     1.00000000D0
      end if  ! Ti

      ! Fe: IRON (26)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !         - TENTATIVE.
         USS(26)  =   -69.6255370D0
         UPP(26)  =   -32.1677030D0
         ZS(26)   =     1.2000000D0
         ZP(26)   =     1.5107948D0
         BETAS(26)=    -4.3143426D0
         BETAP(26)=    -2.7040735D0
         ALP(26)  =     2.7000000D0
         GSS(26)  =     7.6095363D0
         GPP(26)  =     5.5516124D0
         GSP(26)  =     5.8869148D0
         GP2(26)  =     4.8665298D0
         HSP(26)  =     1.0745243D0
         UDD(26)  =   -86.3710680D0
         ZD(26)   =     2.8596222D0
         BETAD(26)=   -14.6844620D0
         ZSN(26)  =     1.3920000D0
         ZPN(26)  =     0.9320000D0
         ZDN(26)  =     1.6944299D0
         PO(9,26) =     1.2704832D0
         F0SD(26) =     9.0540000D0
         G2SD(26) =     1.0000000D0
      end if  ! Fe

      ! Ni: NICKEL (28)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !         - TENTATIVE.
         USS(28)  =    -92.179314D0
         UPP(28)  =    -56.262348D0
         ZS(28)   =     2.3720709D0
         ZP(28)   =     1.0543835D0
         BETAS(28)=    -12.197514D0
         BETAP(28)=    -0.97677325D0
         ALP(28)  =     2.8719575D0
         GSS(28)  =     7.9830553D0
         GPP(28)  =     6.7631310D0
         GSP(28)  =     6.8814412D0
         GP2(28)  =     5.9285440D0
         HSP(28)  =     1.4918151D0
         UDD(28)  =    -132.45919D0
         ZD(28)   =     2.5838212D0
         BETAD(28)=    -9.8596091D0
         ZSN(28)  =     1.4603272D0
         ZPN(28)  =     1.1353887D0
         ZDN(28)  =     2.0729321D0
         PO(9,28) =     0.81235352D0
         F0SD(28) =     9.5030324D0
         G2SD(28) =     1.0000000D0
      end if  ! Ni

      ! Cu: COPPER (29)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK,PRELIMINARY UNPUBLISHED PARAMETERS (6/95)
      !         - TENTATIVE.
      !         Adpoted by V. Paget (10/97) who added an additional Alpha bond 
      !         parameters for Cu-S (See MLIG routine).
         USS(29)  =  -103.4203200D0
         UPP(29)  =   -77.2073860D0
         ZS(29)   =     2.5000000D0
         ZP(29)   =     1.5577284D0
         BETAS(29)=   -14.9997590D0
         BETAP(29)=    -3.3874786D0
         ALP(29)  =     2.5338080D0
         GSS(29)  =     8.2017202D0
         GPP(29)  =     6.7631310D0
         GSP(29)  =     6.9401293D0
         GP2(29)  =     5.9285440D0
         HSP(29)  =     1.4670091D0
         UDD(29)  =  -196.5362800D0
         ZD(29)   =     2.4486511D0
         BETAD(29)=    -8.8868118D0
         ZSN(29)  =     1.5003272D0
         ZPN(29)  =     1.1353887D0
         ZDN(29)  =     2.8678549D0
         PO(9,29) =     0.75720374D0
         F0SD(29) =     9.6630324D0
         G2SD(29) =     1.0000000D0
      end if  ! Cu

      ! Zn: ZINC (30)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S. DEWAR, K.M. MERZ, ORGANOMETALLICS, 5, 1494-1496 (1986)
         USS(30)  =   -20.8397160D0
         UPP(30)  =   -19.6252240D0
         BETAS(30)=    -1.0000000D0
         BETAP(30)=    -2.0000000D0
         ZS(30)   =     2.0473590D0
         ZP(30)   =     1.4609460D0
         ALP(30)  =     1.5064570D0
         EISOL(30)=   -29.8794320D0
         GSS(30)  =    11.8000000D0
         GSP(30)  =    11.1820180D0
         GPP(30)  =    13.3000000D0
         GP2(30)  =    12.9305200D0
         HSP(30)  =     0.4846060D0
         DD(2,30) =     1.3037826D0
         QQ(30)   =     1.4520183D0
         AM(30)   =     0.4336641D0
         AD(30)   =     0.2375912D0
         AQ(30)   =     0.2738858D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: M.J.S. DEWAR, K.M. MERZ, ORGANOMETALLICS, 7, 522-524 (1988)
         USS(30)  =   -21.0400080D0
         UPP(30)  =   -17.6555740D0
         BETAS(30)=    -1.9974290D0
         BETAP(30)=    -4.7581190D0
         ZS(30)   =     1.9542990D0
         ZP(30)   =     1.3723650D0
         ALP(30)  =     1.4845630D0
         EISOL(30)=   -30.2800160D0
         GSS(30)  =    11.8000000D0
         GSP(30)  =    11.1820180D0
         GPP(30)  =    13.3000000D0
         GP2(30)  =    12.9305200D0
         HSP(30)  =     0.4846060D0
         DD(2,30) =     1.3581113D0
         QQ(30)   =     1.5457406D0
         AM(30)   =     0.4336641D0
         AD(30)   =     0.2317423D0
         AQ(30)   =     0.2621165D0
         GUESS1(1,30)=  0.0000000D0
         GUESS2(1,30)=  0.0000000D0
         GUESS3(1,30)=  0.0000000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(30)  =   -18.5321980D0
         UPP(30)  =   -11.0474090D0
         BETAS(30)=    -0.7155780D0
         BETAP(30)=    -6.3518640D0
         ZS(30)   =     1.8199890D0
         ZP(30)   =     1.5069220D0
         ALP(30)  =     1.3501260D0
         EISOL(30)=   -27.3872000D0
         GSS(30)  =     9.6771960D0
         GSP(30)  =     7.7362040D0
         GPP(30)  =     4.9801740D0
         GP2(30)  =     4.6696560D0
         HSP(30)  =     0.6004130D0
         DD(2,30) =     1.5005758D0
         QQ(30)   =     1.4077174D0
         AM(30)   =     0.3556485D0
         AD(30)   =     0.2375689D0
         AQ(30)   =     0.2661069D0
         GUESS1(1,30)= -0.1112340D0
         GUESS2(1,30)=  6.0014780D0
         GUESS3(1,30)=  1.5160320D0
         GUESS1(2,30)= -0.1323700D0
         GUESS2(2,30)=  1.9958390D0
         GUESS3(2,30)=  2.5196420D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d (sp):  W.THIEL AND A.A.VOITYUK, J. PHYS. CHEM., 100, 616 (1996)
         USS(30)  =   -18.02300000D0
         UPP(30)  =   -12.24216585D0
         ZS(30)   =     1.73150352D0
         ZP(30)   =     1.39358305D0
         BETAS(30)=    -5.01726076D0
         BETAP(30)=    -0.71205972D0
         ALP(30)  =     1.51763697D0
         GSS(30)  =     8.56072836D0
         GPP(30)  =     5.13964830D0
         GSP(30)  =     7.49003598D0
         GP2(30)  =     4.50540309D0
         HSP(30)  =     0.53294610D0
         ZSN(30)  =     1.56600000D0
         ZPN(30)  =     0.86283981D0
         PO(9,30) =     1.58923393D0
         F0SD(30) =     0.00000000D0
         G2SD(30) =     0.00000000D0
      end if  ! Zn

      ! Ga: GALLIUM (31)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(31)  =   -29.8555930D0
         UPP(31)  =   -21.8753710D0
         BETAS(31)=    -4.9456180D0
         BETAP(31)=    -0.4070530D0
         ZS(31)   =     1.8470400D0
         ZP(31)   =     0.8394110D0
         ALP(31)  =     1.6051150D0
         EISOL(31)=   -57.3280250D0
         GSS(31)  =     8.4585540D0
         GSP(31)  =     8.9256190D0
         GPP(31)  =     5.0868550D0
         GP2(31)  =     4.9830450D0
         HSP(31)  =     2.0512600D0
         DD(2,31) =     0.9776692D0
         QQ(31)   =     2.5271534D0
         AM(31)   =     0.3108620D0
         AD(31)   =     0.5129360D0
         AQ(31)   =     0.1546208D0
         GUESS1(1,31)= -0.5601790D0
         GUESS2(1,31)=  5.6232730D0
         GUESS3(1,31)=  1.5317800D0
         GUESS1(2,31)= -0.2727310D0
         GUESS2(2,31)=  1.9918430D0
         GUESS3(2,31)=  2.1838640D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS 
      !         - TENTATIVE.
         USS(31)  =    -21.161180D0
         UPP(31)  =    -17.215830D0
         ZS(31)   =     3.0167789D0
         ZP(31)   =     1.3296269D0
         BETAS(31)=    -2.8235297D0
         BETAP(31)=    -3.7434912D0
         ALP(31)  =     1.4653088D0
         GSS(31)  =     8.4346805D0
         GPP(31)  =     6.6345632D0
         GSP(31)  =     6.4475816D0
         GP2(31)  =     5.8158418D0
         HSP(31)  =     1.6719689D0
         UDD(31)  =    -5.6371679D0
         ZD(31)   =     1.7334894D0
         BETAD(31)=    -0.22936193D0
         ZSN(31)  =     1.5429423D0
         ZPN(31)  =     1.1138049D0
         ZDN(31)  =     0.91638282D0
         PO(9,31) =     1.6129834D0
         F0SD(31) =     0.00000000D0
         G2SD(31) =     0.00000000D0
      end if  ! Ga

      ! Ge: GERMANIUM (32)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR, G.L.GRADY, E.F.HEALY,ORGANOMETALLICS 6, 186-189 (1987)
         USS(32)  =   -33.9493670D0
         UPP(32)  =   -27.4251050D0
         BETAS(32)=    -4.5164790D0
         BETAP(32)=    -1.7555170D0
         ZS(32)   =     1.2931800D0
         ZP(32)   =     2.0205640D0
         ALP(32)  =     1.9784980D0
         EISOL(32)=   -76.2489440D0
         GSS(32)  =     9.8000000D0
         GSP(32)  =     8.3000000D0
         GPP(32)  =     7.3000000D0
         GP2(32)  =     6.5000000D0
         HSP(32)  =     1.3000000D0
         DD(2,32) =     1.2556091D0
         QQ(32)   =     1.0498655D0
         AM(32)   =     0.3601617D0
         AD(32)   =     0.3643722D0
         AQ(32)   =     0.4347337D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S.Dewar and C.Jie, Organometallics, 8, 1544, (1989)
         USS(32)  =   -34.1838890D0
         UPP(32)  =   -28.6408110D0
         BETAS(32)=    -4.3566070D0
         BETAP(32)=    -0.9910910D0
         ZS(32)   =     1.2196310D0
         ZP(32)   =     1.9827940D0
         ALP(32)  =     2.1364050D0
         EISOL(32)=   -78.7084810D0
         GSS(32)  =    10.1686050D0
         GSP(32)  =     8.1444730D0
         GPP(32)  =     6.6719020D0
         GP2(32)  =     6.2697060D0
         HSP(32)  =     0.9370930D0
         DD(2,32) =     1.2472095D0
         QQ(32)   =     1.0698642D0
         AM(32)   =     0.3737084D0
         AD(32)   =     0.3180309D0
         AQ(32)   =     0.3485612D0
         GUESS1(1,32)=  0.0000000D0
         GUESS2(1,32)=  0.0000000D0
         GUESS3(1,32)=  0.0000000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(32)  =   -35.4671955D0
         UPP(32)  =   -31.5863583D0
         BETAS(32)=    -5.3250024D0
         BETAP(32)=    -2.2501567D0
         ZS(32)   =     2.2373526D0
         ZP(32)   =     1.5924319D0
         ALP(32)  =     1.9723370D0
         EISOL(32)=   -84.0156006D0
         GSS(32)  =     5.3769635D0
         GSP(32)  =    10.2095293D0
         GPP(32)  =     7.6718647D0
         GP2(32)  =     6.9242663D0
         HSP(32)  =     1.3370204D0
         DD(2,32) =     1.1920304D0
         QQ(32)   =     1.3321263D0
         AM(32)   =     0.1976098D0
         AD(32)   =     0.3798182D0
         AQ(32)   =     0.3620669D0
         GUESS1(1,32)=  0.9631726D0
         GUESS2(1,32)=  6.0120134D0
         GUESS3(1,32)=  2.1633655D0
         GUESS1(2,32)= -0.9593891D0
         GUESS2(2,32)=  5.7491802D0
         GUESS3(2,32)=  2.1693724D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK,PRELIMINARY UNPUBLISHED PARAMETERS
      !         - TENTATIVE.
         USS(32)  =   -29.36038612D0
         UPP(32)  =   -26.02069403D0
         ZS(32)   =     2.96419627D0
         ZP(32)   =     1.98214542D0
         BETAS(32)=    -1.36399120D0
         BETAP(32)=    -6.15689621D0
         ALP(32)  =     1.53511460D0
         GSS(32)  =     8.75477837D0
         GPP(32)  =     7.18043818D0
         GSP(32)  =     7.12778181D0
         GP2(32)  =     6.29435449D0
         HSP(32)  =     1.22740226D0
         UDD(32)  =   -13.11724149D0
         ZD(32)   =     1.12978634D0
         BETAD(32)=    -3.02169012D0
         ZSN(32)  =     1.60149725D0
         ZPN(32)  =     1.20544589D0
         ZDN(32)  =     1.15713414D0
         PO(9,32) =     1.55400850D0
         F0SD(32) =     0.00000000D0
         G2SD(32) =     0.00000000D0
      end if  ! Ge

      ! As: ARSENIC (33)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:  J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(33)  =   -38.5074240D0
         UPP(33)  =   -35.1524150D0
         BETAS(33)=    -8.2321650D0
         BETAP(33)=    -5.0173860D0
         ZS(33)   =     2.6361770D0
         ZP(33)   =     1.7038890D0
         ALP(33)  =     1.7944770D0
         EISOL(33)=  -122.6326140D0
         GSS(33)  =     8.7890010D0
         GSP(33)  =     5.3979830D0
         GPP(33)  =     8.2872500D0
         GP2(33)  =     8.2103460D0
         HSP(33)  =     1.9510340D0
         DD(2,33) =     0.9679655D0
         QQ(33)   =     1.2449874D0
         AM(33)   =     0.3230063D0
         AD(33)   =     0.5042239D0
         AQ(33)   =     0.2574219D0
         GUESS1(1,33)= -0.4600950D0
         GUESS2(1,33)=  1.9831150D0
         GUESS3(1,33)=  1.0867930D0
         GUESS1(2,33)= -0.0889960D0
         GUESS2(2,33)=  1.9929440D0
         GUESS3(2,33)=  2.1400580D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: T.HANSEN, W.THIEL AND A.VOITYUK, PRELIMINARY UNPUBLISHED
      !         PARAMETERS - TENTATIVE.
         USS(33)  =   -45.35231457D0
         UPP(33)  =   -36.81556942D0
         ZS(33)   =     2.49038782D0
         ZP(33)   =     1.78603525D0
         BETAS(33)=    -1.90452598D0
         BETAP(33)=    -6.28997469D0
         ALP(33)  =     1.78527801D0
         GSS(33)  =     9.90741561D0
         GPP(33)  =     7.21687665D0
         GSP(33)  =     8.01677939D0
         GP2(33)  =     6.32629637D0
         HSP(33)  =     1.56966576D0
         UDD(33)  =    -7.87632387D0
         ZD(33)   =     1.72556723D0
         BETAD(33)=    -3.61171688D0
         ZSN(33)  =     1.81234729D0
         ZPN(33)  =     1.21156315D0
         ZDN(33)  =     1.20184975D0
         PO(9,33) =     1.13365978D0
         F0SD(33) =     0.00000000D0
         G2SD(33) =     0.00000000D0
      end if  ! As

      ! Se: SELENIUM (34)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(34)  =   -55.3781350D0
         UPP(34)  =   -49.8230760D0
         BETAS(34)=    -6.1578220D0
         BETAP(34)=    -5.4930390D0
         ZS(34)   =     2.8280510D0
         ZP(34)   =     1.7325360D0
         ALP(34)  =     3.0439570D0
         EISOL(34)=  -192.7748115D0
         GSS(34)  =     7.4325910D0
         GSP(34)  =    10.0604610D0
         GPP(34)  =     9.5683260D0
         GP2(34)  =     7.7242890D0
         HSP(34)  =     4.0165580D0
         DD(2,34) =     0.8719813D0
         QQ(34)   =     1.2244019D0
         AM(34)   =     0.2731566D0
         AD(34)   =     0.7509697D0
         AQ(34)   =     0.5283737D0
         GUESS1(1,34)=  0.0478730D0
         GUESS2(1,34)=  6.0074000D0
         GUESS3(1,34)=  2.0817170D0
         GUESS1(2,34)=  0.1147200D0
         GUESS2(2,34)=  6.0086720D0
         GUESS3(2,34)=  1.5164230D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: T.HANSEN, W.THIEL AND A.VOITYUK, PRELIMINARY UNPUBLISHED
      !         PARAMETERS - TENTATIVE.
         USS(34)  =   -53.73524614D0
         UPP(34)  =   -42.60566657D0
         ZS(34)   =     2.36196797D0
         ZP(34)   =     2.22540353D0
         BETAS(34)=    -0.87799750D0
         BETAP(34)=    -9.73264791D0
         ALP(34)  =     1.96797282D0
         GSS(34)  =     9.80312440D0
         GPP(34)  =     7.53932187D0
         GSP(34)  =     7.42339492D0
         GP2(34)  =     6.60895105D0
         HSP(34)  =     1.54425360D0
         UDD(34)  =    -8.85993261D0
         ZD(34)   =     1.42825025D0
         BETAD(34)=    -0.25380161D0
         ZSN(34)  =     1.79326947D0
         ZPN(34)  =     1.26569498D0
         ZDN(34)  =     1.25000000D0
         PO(9,34) =     1.36855209D0
         F0SD(34) =     0.00000000D0
         G2SD(34) =     0.00000000D0
      end if  ! Se

      ! Br: BROMINE (35)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR, E.F. HEALY, J. COMP. CHEM., 4, 542 (1983)
         USS(35)  =   -99.9864405D0
         UPP(35)  =   -75.6713075D0
         BETAS(35)=    -8.9171070D0
         BETAP(35)=    -9.9437400D0
         ZS(35)   =     3.8543019D0
         ZP(35)   =     2.1992091D0
         ALP(35)  =     2.4457051D0
         EISOL(35)=  -346.6812500D0
         GSS(35)  =    15.03643948D0
         GSP(35)  =    13.03468242D0
         GPP(35)  =    11.27632539D0
         GP2(35)  =     9.85442552D0
         HSP(35)  =     2.45586832D0
         DD(2,35) =     0.6051074D0
         QQ(35)   =     0.9645873D0
         AM(35)   =     0.5526068D0
         AD(35)   =     0.7258330D0
         AQ(35)   =     0.5574589D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S.DEWAR AND E.G.ZOEBISCH, J.MOL.STRUCT. 180, 1 (1988) 
         USS(35)  =  -104.6560630D0
         UPP(35)  =   -74.9300520D0
         BETAS(35)=   -19.3998800D0
         BETAP(35)=    -8.9571950D0
         ZS(35)   =     3.0641330D0
         ZP(35)   =     2.0383330D0
         ALP(35)  =     2.5765460D0
         EISOL(35)=  -352.3142087D0
         GSS(35)  =    15.0364395D0
         GSP(35)  =    13.0346824D0
         GPP(35)  =    11.2763254D0
         GP2(35)  =     9.8544255D0
         HSP(35)  =     2.4558683D0
         DD(2,35) =     0.8458104D0
         QQ(35)   =     1.0407133D0
         AM(35)   =     0.5526071D0
         AD(35)   =     0.6024598D0
         AQ(35)   =     0.5307555D0
         GUESS1(1,35)=  0.0666850D0
         GUESS2(1,35)=  4.0000000D0
         GUESS3(1,35)=  1.5000000D0
         GUESS1(2,35)=  0.0255680D0
         GUESS2(2,35)=  4.0000000D0
         GUESS3(2,35)=  2.3000000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS(35)  =  -116.6193110D0
         UPP(35)  =   -74.2271290D0
         BETAS(35)=   -31.1713420D0
         BETAP(35)=    -6.8140130D0
         ZS(35)   =     5.3484570D0
         ZP(35)   =     2.1275900D0
         ALP(35)  =     2.5118420D0
         EISOL(35)=  -352.5398970D0
         GSS(35)  =    15.9434250D0
         GSP(35)  =    16.0616800D0
         GPP(35)  =     8.2827630D0
         GP2(35)  =     7.8168490D0
         HSP(35)  =     0.5788690D0
         DD(2,35) =     0.2759025D0
         QQ(35)   =     0.9970532D0
         AM(35)   =     0.5859399D0
         AD(35)   =     0.6755383D0
         AQ(35)   =     0.3823719D0
         GUESS1(1,35)=  0.9604580D0
         GUESS2(1,35)=  5.9765080D0
         GUESS3(1,35)=  2.3216540D0
         GUESS1(2,35)= -0.9549160D0
         GUESS2(2,35)=  5.9447030D0
         GUESS3(2,35)=  2.3281420D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, INTERN. J. QUANT. CHEM., 44, 807 (1992)
         USS(35)  =   -65.40277790D0
         UPP(35)  =   -54.55375352D0
         ZS(35)   =     2.59054101D0
         ZP(35)   =     2.33085649D0
         BETAS(35)=    -8.31497607D0
         BETAP(35)=   -10.50704145D0
         ALP(35)  =     2.09105000D0
         GSS(35)  =    12.22235546D0
         GPP(35)  =     8.53546437D0
         GSP(35)  =     8.26372010D0
         GP2(35)  =     7.48216712D0
         HSP(35)  =     2.74952230D0
         UDD(35)  =   -13.72809929D0
         ZD(35)   =     1.35736115D0
         BETAD(35)=    -0.96259930D0
         ZSN(35)  =     2.23581544D0
         ZPN(35)  =     1.43292654D0
         ZDN(35)  =     1.24257826D0
         PO(9,35) =     1.11312423D0
         F0SD(35) =     0.00000000D0
         G2SD(35) =     0.00000000D0
      else if(iqm_mode.eq.4) then
      ! AM1/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !        - TENTATIVE.
         USS(35)  =   -67.10473939D0
         UPP(35)  =   -57.68160345D0
         ZS(35)   =     3.47395752D0
         ZP(35)   =     2.29854121D0
         BETAS(35)=   -12.20140135D0
         BETAP(35)=    -9.59718493D0
         ALP(35)  =     2.57654595D0
         GSS(35)  =    12.76901805D0
         GPP(35)  =     8.53546437D0
         GSP(35)  =     9.47149397D0
         GP2(35)  =     7.48216712D0
         HSP(35)  =     2.10122000D0
         UDD(35)  =   -36.00870596D0
         ZD(35)   =     2.18336919D0
         BETAD(35)=    -0.56787961D0
         ZSN(35)  =     2.33581553D0
         ZPN(35)  =     1.43292654D0
         ZDN(35)  =     1.14257827D0
         PO(9,35) =     1.06546956D0
         F0SD(35) =     0.00000000D0
         G2SD(35) =     0.00000000D0
         GUESS1(1,35)=  0.53539823D0
         GUESS2(1,35)=  4.00000000D0
         GUESS3(1,35)=  1.08638700D0
         GUESS1(2,35)=  0.08560224D0
         GUESS2(2,35)=  4.00000000D0
         GUESS3(2,35)=  1.95230582D0
      end if  ! Br

      ! Zr: ZIRCONIUM (40)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK,PRELIMINARY UNPUBLISHED PARAMETERS (6/95)
      !         - TENTATIVE.
      !         Adopted by V. Paget (10/97)
         USS(40)  =   -26.6022000D0
         UPP(40)  =   -13.3751010D0
         ZS(40)   =     1.3758868D0
         ZP(40)   =     1.2899575D0
         BETAS(40)=    -7.7496044D0
         BETAP(40)=    -2.7537267D0
         ALP(40)  =     1.4644184D0
         GSS(40)  =     6.0612001D0
         GPP(40)  =     5.0111208D0
         GSP(40)  =     5.1077389D0
         GP2(40)  =     4.3670874D0
         HSP(40)  =     1.0689996D0
         UDD(40)  =   -34.1591730D0
         ZD(40)   =     1.6580218D0
         BETAD(40)=    -6.5469651D0
         ZSN(40)  =     1.3520000D0
         ZPN(40)  =     1.0220000D0
         ZDN(40)  =     2.0926610D0
         PO(9,40) =     1.7917452D0
         F0SD(40) =     6.9400000D0
         G2SD(40) =     1.0200000D0
      end if  ! Zr

      ! Pd: PALLADIUM (46)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d: V. PAGET AND W. THIEL,PRELIMINARY UNPUBLISHED PARAMETERS (10/97)
      !         - TENTATIVE.
         USS(46)  =   -85.80500000D0
         UPP(46)  =   -54.22986390D0
         ZS(46)   =     1.90123551D0
         ZP(46)   =     1.53828330D0
         BETAS(46)=    -4.15072695D0
         BETAP(46)=    -4.86175662D0
         ALP(46)  =     2.25000000D0
         GSS(46)  =     7.44200576D0
         GPP(46)  =     7.45293866D0
         GSP(46)  =     6.21779500D0
         GP2(46)  =     6.49508075D0
         HSP(46)  =     1.28185420D0
         UDD(46)  =  -115.81361000D0
         ZD(46)   =     2.87456910D0
         BETAD(46)=   -24.07391049D0
         ZSN(46)  =     1.66000000D0
         ZPN(46)  =     1.52000000D0
         ZDN(46)  =     2.25000000D0
         PO(9,46) =     0.93798203D0
         F0SD(46) =     8.84000000D0
         G2SD(46) =     1.20000000D0
      end if  ! Pd

      ! Ag:  SILVER (47)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !         - TENTATIVE.
         USS(47)  =   -92.8606090D0
         UPP(47)  =   -66.9447520D0
         ZS(47)   =     2.4620900D0
         ZP(47)   =     1.7052370D0
         BETAS(47)=    -0.7800000D0
         BETAP(47)=    -4.4218983D0
         ALP(47)  =     2.2705234D0
         GSS(47)  =     6.7149932D0
         GPP(47)  =     5.6877692D0
         GSP(47)  =     5.7567018D0
         GP2(47)  =     4.9567724D0
         HSP(47)  =     1.2400026D0
         UDD(47)  =  -137.3694200D0
         ZD(47)   =     2.9938876D0
         BETAD(47)=    -9.1093713D0
         ZSN(47)  =     1.4978339D0
         ZPN(47)  =     1.1600000D0
         ZDN(47)  =     2.4297304D0
         PO(9,47) =     1.0467261D0
         F0SD(47) =     8.6945173D0
         G2SD(47) =     1.7245646D0
      end if  ! Ag

      ! Cd: CADMIUM (48)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(48)  =   -15.8285840D0
         UPP(48)  =     8.7497950D0
         BETAS(48)=    -8.5819440D0
         BETAP(48)=    -0.6010340D0
         ZS(48)   =     1.6793510D0
         ZP(48)   =     2.0664120D0
         ALP(48)  =     1.5253820D0
         EISOL(48)=   -22.4502080D0
         GSS(48)  =     9.2069600D0
         GSP(48)  =     8.2315390D0
         GPP(48)  =     4.9481040D0
         GP2(48)  =     4.6696560D0
         HSP(48)  =     1.6562340D0
         DD(2,48) =     1.5982681D0
         QQ(48)   =     1.2432402D0
         AM(48)   =     0.3383668D0
         AD(48)   =     0.3570290D0
         AQ(48)   =     0.2820582D0
         GUESS1(1,48)=  0.0000000D0
         GUESS2(1,48)=  0.0000000D0
         GUESS3(1,48)=  0.0000000D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d (sp): W.THIEL AND A.A.VOITYUK, J. PHYS. CHEM., 100, 616 (1996)
         USS(48)  =   -16.96970000D0
         UPP(48)  =   -12.40096476D0
         ZS(48)   =     1.74880559D0
         ZP(48)   =     1.56321473D0
         BETAS(48)=    -2.77154379D0
         BETAP(48)=    -1.80565019D0
         ALP(48)  =     1.42461329D0
         GSS(48)  =     7.90443438D0
         GPP(48)  =     7.47999993D0
         GSP(48)  =     7.51570687D0
         GP2(48)  =     6.51866416D0
         HSP(48)  =     0.63674441D0
         ZSN(48)  =     1.76314840D0
         ZPN(48)  =     1.52551900D0
         PO(9,48) =     1.72118577D0
         F0SD(48) =     0.00000000D0
         G2SD(48) =     0.00000000D0
      end if  ! Cd

      ! In: INDIUM (49)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(49)  =   -26.1762050D0
         UPP(49)  =   -20.0058220D0
         BETAS(49)=    -2.9933190D0
         BETAP(49)=    -1.8289080D0
         ZS(49)   =     2.0161160D0
         ZP(49)   =     1.4453500D0
         ALP(49)  =     1.4183850D0
         EISOL(49)=   -51.9750470D0
         GSS(49)  =     6.5549000D0
         GSP(49)  =     8.2298730D0
         GPP(49)  =     6.2992690D0
         GP2(49)  =     4.9842110D0
         HSP(49)  =     2.6314610D0
         DD(2,49) =     1.5766241D0
         QQ(49)   =     1.7774563D0
         AM(49)   =     0.2409004D0
         AD(49)   =     0.4532655D0
         AQ(49)   =     0.3689812D0
         GUESS1(1,49)= -0.3431380D0
         GUESS2(1,49)=  1.9940340D0
         GUESS3(1,49)=  1.6255160D0
         GUESS1(2,49)= -0.1095320D0
         GUESS2(2,49)=  5.6832170D0
         GUESS3(2,49)=  2.8670090D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !         - TENTATIVE.
         USS(49)  =   -19.3122030D0
         UPP(49)  =   -16.3900060D0
         ZS(49)   =     4.8520927D0
         ZP(49)   =     1.3669015D0
         BETAS(49)=    -1.8566979D0
         BETAP(49)=    -3.4871655D0
         ALP(49)  =     1.4049941D0
         GSS(49)  =     6.9419500D0
         GPP(49)  =     6.4040931D0
         GSP(49)  =     6.3010636D0
         GP2(49)  =     5.5810337D0
         HSP(49)  =     1.4712393D0
         UDD(49)  =    -4.1204789D0
         ZD(49)   =     1.9701543D0
         BETAD(49)=    -1.6031897D0
         ZSN(49)  =     1.5484584D0
         ZPN(49)  =     1.3060917D0
         ZDN(49)  =     1.0163828D0
         PO(9,49) =     1.9598240D0
         F0SD(49) =     0.00000000D0
         G2SD(49) =     0.00000000D0
      end if  ! In

      ! Sn: TIN (50)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR,G.L.GRADY,J.J.P.STEWART, J.AM.CHEM.SOC.,106,6771 (1984)
         USS(50)  =   -40.8518020D0
         UPP(50)  =   -28.5602490D0
         BETAS(50)=    -3.2351470D0
         BETAP(50)=    -4.2904160D0
         ZS(50)   =     2.0803800D0
         ZP(50)   =     1.9371060D0
         ALP(50)  =     1.8008140D0
         EISOL(50)=   -92.3241020D0
         GSS(50)  =     9.8000000D0
         GSP(50)  =     8.3000000D0
         GPP(50)  =     7.3000000D0
         GP2(50)  =     6.5000000D0
         HSP(50)  =     1.3000000D0
         DD(2,50) =     1.5697766D0
         QQ(50)   =     1.3262292D0
         AM(50)   =     0.3601617D0
         AD(50)   =     0.3219998D0
         AQ(50)   =     0.3713827D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S. DEWAR ET AL, ORGANOMETALLICS, 10, 431, (1991)
         USS(50)  =   -35.4967410D0
         UPP(50)  =   -28.0976360D0
         ZS(50)   =     2.5993760D0
         ZP(50)   =     1.6959620D0
         BETAS(50)=    -3.2350000D0
         BETAP(50)=    -2.5778900D0
         ALP(50)  =     1.8369360D0
         EISOL(50)=   -80.6887540D0
         GSS(50)  =     9.8000000D0
         GSP(50)  =     8.3000000D0
         GPP(50)  =     7.3000000D0
         GP2(50)  =     6.5000000D0
         HSP(50)  =     1.3000000D0
         DD(2,50) =     1.1528230D0
         QQ(50)   =     1.5148020D0
         AM(50)   =     0.3601620D0
         AD(50)   =     0.3823780D0
         AQ(50)   =     0.3400760D0
         GUESS1(1,50)=  0.0000000D0
         GUESS2(1,50)=  0.0000000D0
         GUESS3(1,50)=  0.0000000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(50)  =   -34.5501920D0
         UPP(50)  =   -25.8944190D0
         BETAS(50)=    -2.7858020D0
         BETAP(50)=    -2.0059990D0
         ZS(50)   =     2.3733280D0
         ZP(50)   =     1.6382330D0
         ALP(50)  =     1.6996500D0
         EISOL(50)=   -78.8877790D0
         GSS(50)  =    10.1900330D0
         GSP(50)  =     7.2353270D0
         GPP(50)  =     5.6738100D0
         GP2(50)  =     5.1822140D0
         HSP(50)  =     1.0331570D0
         DD(2,50) =     1.3120038D0
         QQ(50)   =     1.5681814D0
         AM(50)   =     0.3744959D0
         AD(50)   =     0.3218163D0
         AQ(50)   =     0.2832529D0
         GUESS1(1,50)= -0.1503530D0
         GUESS2(1,50)=  6.0056940D0
         GUESS3(1,50)=  1.7046420D0
         GUESS1(2,50)= -0.0444170D0
         GUESS2(2,50)=  2.2573810D0
         GUESS3(2,50)=  2.4698690D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !         - TENTATIVE.
         USS(50)  =   -30.19670332D0
         UPP(50)  =   -24.17283167D0
         ZS(50)   =     2.56428352D0
         ZP(50)   =     1.99274752D0
         BETAS(50)=    -1.76967330D0
         BETAP(50)=    -4.11434709D0
         ALP(50)  =     1.46041372D0
         GSS(50)  =     8.02366621D0
         GPP(50)  =     6.14093726D0
         GSP(50)  =     6.85242200D0
         GP2(50)  =     5.35169893D0
         HSP(50)  =     0.92242200D0
         UDD(50)  =    -7.47357211D0
         ZD(50)   =     1.20971476D0
         BETAD(50)=    -1.88087750D0
         ZSN(50)  =     1.78974403D0
         ZPN(50)  =     1.25242200D0
         ZDN(50)  =     1.09155000D0
         PO(9,50) =     1.69560892D0
         F0SD(50) =     0.00000000D0
         G2SD(50) =     0.00000000D0
      end if  ! Sn

      ! Sb: ANTIMONY (51)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(51)  =   -56.4321960D0
         UPP(51)  =   -29.4349540D0
         BETAS(51)=   -14.7942170D0
         BETAP(51)=    -2.8179480D0
         ZS(51)   =     2.3430390D0
         ZP(51)   =     1.8999920D0
         ALP(51)  =     2.0343010D0
         EISOL(51)=  -148.9382890D0
         GSS(51)  =     9.2382770D0
         GSP(51)  =     5.2776800D0
         GPP(51)  =     6.3500000D0
         GP2(51)  =     6.2500000D0
         HSP(51)  =     2.4244640D0
         DD(2,51) =     1.4091903D0
         QQ(51)   =     1.3521354D0
         AM(51)   =     0.3395177D0
         AD(51)   =     0.4589010D0
         AQ(51)   =     0.2423472D0
         GUESS1(1,51)=  3.0020280D0
         GUESS2(1,51)=  6.0053420D0
         GUESS3(1,51)=  0.8530600D0
         GUESS1(2,51)= -0.0188920D0
         GUESS2(2,51)=  6.0114780D0
         GUESS3(2,51)=  2.7933110D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: T.HANSEN, W.THIEL AND A.VOITYUK, PRELIMINARY UNPUBLISHED
      !         PARAMETERS - TENTATIVE.
         USS(51)  =   -40.37506195D0
         UPP(51)  =   -33.91714049D0
         ZS(51)   =     2.10000000D0
         ZP(51)   =     1.86606261D0
         BETAS(51)=    -2.86454599D0
         BETAP(51)=    -3.56227358D0
         ALP(51)  =     1.59693495D0
         GSS(51)  =    10.48051795D0
         GPP(51)  =     7.91700506D0
         GSP(51)  =     7.13401188D0
         GP2(51)  =     6.89950503D0
         HSP(51)  =     1.21126651D0
         UDD(51)  =    -6.48447385D0
         ZD(51)   =     1.42560000D0
         BETAD(51)=    -0.39751993D0
         ZSN(51)  =     2.33776480D0
         ZPN(51)  =     1.61464462D0
         ZDN(51)  =     0.90240000D0
         PO(9,51) =     1.17883481D0
         F0SD(51) =     0.00000000D0
         G2SD(51) =     0.00000000D0
      end if  ! Sb

      ! Te: TELLURIUM (52)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(52)  =   -44.9380360D0
         UPP(52)  =   -46.3140990D0
         BETAS(52)=    -2.6651460D0
         BETAP(52)=    -3.8954300D0
         ZS(52)   =     4.1654920D0
         ZP(52)   =     1.6475550D0
         ALP(52)  =     2.4850190D0
         EISOL(52)=  -168.0945925D0
         GSS(52)  =    10.2550730D0
         GSP(52)  =     8.1691450D0
         GPP(52)  =     7.7775920D0
         GP2(52)  =     7.7551210D0
         HSP(52)  =     3.7724620D0
         DD(2,52) =     0.3484177D0
         QQ(52)   =     1.5593085D0
         AM(52)   =     0.3768862D0
         AD(52)   =     1.1960743D0
         AQ(52)   =     0.2184786D0
         GUESS1(1,52)=  0.0333910D0
         GUESS2(1,52)=  5.9563790D0
         GUESS3(1,52)=  2.2775750D0
         GUESS1(2,52)= -1.9218670D0
         GUESS2(2,52)=  4.9732190D0
         GUESS3(2,52)=  0.5242430D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: T.HANSEN, W.THIEL AND A.VOITYUK, PRELIMINARY UNPUBLISHED
      !         PARAMETERS - TENTATIVE.
         USS(52)  =   -46.79068576D0
         UPP(52)  =   -37.04695193D0
         ZS(52)   =     2.54600517D0
         ZP(52)   =     2.38876168D0
         BETAS(52)=    -4.43129829D0
         BETAP(52)=    -6.79821994D0
         ALP(52)  =     1.75551132D0
         GSS(52)  =     8.56521032D0
         GPP(52)  =     6.44891667D0
         GSP(52)  =     6.88545643D0
         GP2(52)  =     5.62009658D0
         HSP(52)  =     1.84925944D0
         UDD(52)  =    -8.45439649D0
         ZD(52)   =     1.54864147D0
         BETAD(52)=    -2.94579824D0
         ZSN(52)  =     1.91053985D0
         ZPN(52)  =     1.31523329D0
         ZDN(52)  =     1.13440000D0
         PO(9,52) =     1.58840233D0
         F0SD(52) =     0.00000000D0
         G2SD(52) =     0.00000000D0
      end if  ! Te

      ! I : IODINE (53)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR, E.F. HEALY, J.J.P. STEWART, J.COMP.CHEM., 5,358 (1984)
         USS(53)  =  -100.0030538D0
         UPP(53)  =   -74.6114692D0
         BETAS(53)=    -7.4144510D0
         BETAP(53)=    -6.1967810D0
         ZS(53)   =     2.2729610D0
         ZP(53)   =     2.1694980D0
         ALP(53)  =     2.2073200D0
         EISOL(53)=  -340.5983600D0
         GSS(53)  =    15.04044855D0
         GSP(53)  =    13.05655798D0
         GPP(53)  =    11.14778369D0
         GP2(53)  =     9.91409071D0
         HSP(53)  =     2.45638202D0
         DD(2,53) =     1.4253233D0
         QQ(53)   =     1.1841707D0
         AM(53)   =     0.5527541D0
         AD(53)   =     0.4593451D0
         AQ(53)   =     0.4585376D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S.DEWAR AND E.G.ZOEBISCH, J.MOL.STRUCT. 180, 1 (1988)
         USS(53)  =  -103.5896630D0
         UPP(53)  =   -74.4299970D0
         BETAS(53)=    -8.4433270D0
         BETAP(53)=    -6.3234050D0
         ZS(53)   =     2.1028580D0
         ZP(53)   =     2.1611530D0
         ALP(53)  =     2.2994240D0
         EISOL(53)=  -346.8642857D0
         GSS(53)  =    15.0404486D0
         GSP(53)  =    13.0565580D0
         GPP(53)  =    11.1477837D0
         GP2(53)  =     9.9140907D0
         HSP(53)  =     2.4563820D0
         DD(2,53) =     1.4878778D0
         QQ(53)   =     1.1887388D0
         AM(53)   =     0.5527544D0
         AD(53)   =     0.4497523D0
         AQ(53)   =     0.4631775D0
         GUESS1(1,53)=  0.0043610D0
         GUESS2(1,53)=  2.3000000D0
         GUESS3(1,53)=  1.8000000D0
         GUESS1(2,53)=  0.0157060D0
         GUESS2(2,53)=  3.0000000D0
         GUESS3(2,53)=  2.2400000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989)
         USS(53)  =   -96.4540370D0
         UPP(53)  =   -61.0915820D0
         BETAS(53)=   -14.4942340D0
         BETAP(53)=    -5.8947030D0
         ZS(53)   =     7.0010130D0
         ZP(53)   =     2.4543540D0
         ALP(53)  =     1.9901850D0
         EISOL(53)=  -288.3160860D0
         GSS(53)  =    13.6319430D0
         GSP(53)  =    14.9904060D0
         GPP(53)  =     7.2883300D0
         GP2(53)  =     5.9664070D0
         HSP(53)  =     2.6300350D0
         DD(2,53) =     0.1581469D0
         QQ(53)   =     1.0467302D0
         AM(53)   =     0.5009902D0
         AD(53)   =     1.6699104D0
         AQ(53)   =     0.5153082D0
         GUESS1(1,53)= -0.1314810D0
         GUESS2(1,53)=  5.2064170D0
         GUESS3(1,53)=  1.7488240D0
         GUESS1(2,53)= -0.0368970D0
         GUESS2(2,53)=  6.0101170D0
         GUESS3(2,53)=  2.7103730D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, INTERN. J. QUANT. CHEM., 44, 807 (1992)
         USS(53)  =   -62.76535256D0
         UPP(53)  =   -50.29211568D0
         ZS(53)   =     2.75654324D0
         ZP(53)   =     2.25307954D0
         BETAS(53)=   -10.69948666D0
         BETAP(53)=    -4.94117776D0
         ALP(53)  =     1.90617441D0
         GSS(53)  =    11.98078196D0
         GPP(53)  =     7.70937227D0
         GSP(53)  =     7.85590192D0
         GP2(53)  =     6.71855729D0
         HSP(53)  =     2.07147462D0
         UDD(53)  =   -12.24830501D0
         ZD(53)   =     1.50233509D0
         BETAD(53)=    -2.35046098D0
         ZSN(53)  =     2.67241100D0
         ZPN(53)  =     1.57229871D0
         ZDN(53)  =     1.25884802D0
         PO(9,53) =     1.13556862D0
         F0SD(53) =     0.00000000D0
         G2SD(53) =     0.00000000D0
      else if(iqm_mode.eq.4) then
      ! AM1/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !        - TENTATIVE.
         USS(53)  =   -64.34769319D0
         UPP(53)  =   -54.08191994D0
         ZS(53)   =     3.21684013D0
         ZP(53)   =     2.18531809D0
         BETAS(53)=    -7.64964060D0
         BETAP(53)=    -5.28948501D0
         ALP(53)  =     1.95761241D0
         GSS(53)  =    11.46290076D0
         GPP(53)  =     7.98713496D0
         GSP(53)  =     9.02588645D0
         GP2(53)  =     6.96062178D0
         HSP(53)  =     1.92196179D0
         UDD(53)  =   -25.51071124D0
         ZD(53)   =     2.03481500D0
         BETAD(53)=    -0.33745830D0
         ZSN(53)  =     2.55689328D0
         ZPN(53)  =     1.62894736D0
         ZDN(53)  =     1.34959843D0
         PO(9,53) =     1.18687235D0
         F0SD(53) =     0.00000000D0
         G2SD(53) =     0.00000000D0
         GUESS1(1,53)= -0.02297018D0
         GUESS2(1,53)=  4.00000000D0
         GUESS3(1,53)=  2.92978474D0
      end if  ! I

      ! Hf: HAFNIUM (72)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !         - TENTATIVE.
         USS(72)  =   -29.5317990D0
         UPP(72)  =   -16.7352620D0
         ZS(72)   =     2.0284073D0
         ZP(72)   =     1.0004882D0
         BETAS(72)=    -8.4847808D0
         BETAP(72)=    -1.3805105D0
         ALP(72)  =     1.5288204D0
         GSS(72)  =     7.0831598D0
         GPP(72)  =     5.6786994D0
         GSP(72)  =     5.8023460D0
         GP2(72)  =     4.9252843D0
         HSP(72)  =     1.1333381D0
         UDD(72)  =   -30.4306880D0
         ZD(72)   =     1.8732565D0
         BETAD(72)=    -5.0552356D0
         ZSN(72)  =     1.8620059D0
         ZPN(72)  =     1.3607669D0
         ZDN(72)  =     2.1100957D0
         PO(9,72) =     1.8347669D0
         F0SD(72) =     7.5906800D0
         G2SD(72) =     1.0000000D0
      end if  ! Hf

      ! Hg: MERCURY (80)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR,  ET. AL. ORGANOMETALLICS 4, 1964 (1985) SEE MANUAL
         USS(80)  =   -19.8095740D0
         UPP(80)  =   -13.1025300D0
         BETAS(80)=    -0.4045250D0
         BETAP(80)=    -6.2066830D0
         ZS(80)   =     2.2181840D0
         ZP(80)   =     2.0650380D0
         ALP(80)  =     1.3356410D0
         EISOL(80)=   -28.8191480D0
         GSS(80)  =    10.8000000D0
         GSP(80)  =     9.3000000D0
         GPP(80)  =    14.3000000D0
         GP2(80)  =    13.5000000D0
         HSP(80)  =     1.3000000D0
         DD(2,80) =     1.7378048D0
         QQ(80)   =     1.4608064D0
         AM(80)   =     0.3969129D0
         AD(80)   =     0.3047694D0
         AQ(80)   =     0.3483102D0
      else if(iqm_mode.eq.2) then
      ! AM1: M.J.S.Dewar and C.Jie, Organometallics 8, 1547, (1989)
         USS(80)  =   -19.9415780D0
         UPP(80)  =   -11.1108700D0
         BETAS(80)=    -0.9086570D0
         BETAP(80)=    -4.9093840D0
         ZS(80)   =     2.0364130D0
         ZP(80)   =     1.9557660D0
         ALP(80)  =     1.4847340D0
         EISOL(80)=   -29.0831560D0
         GSS(80)  =    10.8000000D0
         GSP(80)  =     9.3000000D0
         GPP(80)  =    14.3000000D0
         GP2(80)  =    13.5000000D0
         HSP(80)  =     1.3000000D0
         DD(2,80) =     1.8750829D0
         QQ(80)   =     1.5424241D0
         AM(80)   =     0.3969129D0
         AD(80)   =     0.2926605D0
         AQ(80)   =     0.3360599D0
         GUESS1(1,80)=  0.0000000D0
         GUESS2(1,80)=  0.0000000D0
         GUESS3(1,80)=  0.0000000D0
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(80)  =   -17.7622290D0
         UPP(80)  =   -18.3307510D0
         BETAS(80)=    -3.1013650D0
         BETAP(80)=    -3.4640310D0
         ZS(80)   =     1.4768850D0
         ZP(80)   =     2.4799510D0
         ALP(80)  =     1.5293770D0
         EISOL(80)=   -28.8997380D0
         GSS(80)  =     6.6247200D0
         GSP(80)  =    10.6392970D0
         GPP(80)  =    14.7092830D0
         GP2(80)  =    16.0007400D0
         HSP(80)  =     2.0363110D0
         DD(2,80) =     1.2317811D0
         QQ(80)   =     1.2164033D0
         AM(80)   =     0.2434664D0
         AD(80)   =     0.4515472D0
         AQ(80)   =     0.2618394D0
         GUESS1(1,80)=  1.0827200D0
         GUESS2(1,80)=  6.4965980D0
         GUESS3(1,80)=  1.1951460D0
         GUESS1(2,80)= -0.0965530D0
         GUESS2(2,80)=  3.9262810D0
         GUESS3(2,80)=  2.6271600D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d (sp):  W.THIEL AND A.A.VOITYUK, J. PHYS. CHEM., 100, 616 (1996)
         USS(80)  =   -18.81564903D0
         UPP(80)  =   -13.39711352D0
         ZS(80)   =     2.33310757D0
         ZP(80)   =     1.70831069D0
         BETAS(80)=    -2.21872239D0
         BETAP(80)=    -2.90978573D0
         ALP(80)  =     1.38224172D0
         GSS(80)  =     8.31564948D0
         GPP(80)  =     7.11525878D0
         GSP(80)  =     8.21217300D0
         GP2(80)  =     6.17124983D0
         HSP(80)  =     0.83594100D0
         ZSN(80)  =     2.18600011D0
         ZPN(80)  =     1.70500461D0
         PO(9,80) =     1.63607185D0
         F0SD(80) =     0.00000000D0
         G2SD(80) =     0.00000000D0
      end if  ! Hg

      ! Tl: THALLIUM (81)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(81)  =   -30.0531700D0
         UPP(81)  =   -26.9206370D0
         BETAS(81)=    -1.0844950D0
         BETAP(81)=    -7.9467990D0
         ZS(81)   =     6.8679210D0
         ZP(81)   =     1.9694450D0
         ALP(81)  =     1.3409510D0
         EISOL(81)=   -56.6492050D0
         GSS(81)  =    10.4604120D0
         GSP(81)  =    11.2238830D0
         GPP(81)  =     4.9927850D0
         GP2(81)  =     8.9627270D0
         HSP(81)  =     2.5304060D0
         DD(2,81) =     0.0781362D0
         QQ(81)   =     1.5317110D0
         AM(81)   =     0.3844326D0
         AD(81)   =     2.5741815D0
         AQ(81)   =     0.2213264D0
         GUESS1(1,81)= -1.3613990D0
         GUESS2(1,81)=  3.5572260D0
         GUESS3(1,81)=  1.0928020D0
         GUESS1(2,81)= -0.0454010D0
         GUESS2(2,81)=  2.3069950D0
         GUESS3(2,81)=  2.9650290D0
      else if(iqm_mode.eq.5) then
      ! MNDO/d: W.THIEL AND A.A.VOITYUK, PRELIMINARY UNPUBLISHED PARAMETERS
      !         - TENTATIVE.
         USS(81)  =   -20.05259600D0
         UPP(81)  =   -17.67707800D0
         ZS(81)   =     3.60000000D0
         ZP(81)   =     2.11263370D0
         BETAS(81)=    -0.99067729D0
         BETAP(81)=    -0.81245270D0
         ALP(81)  =     1.41749960D0
         GSS(81)  =     7.03162470D0
         GPP(81)  =     6.80007960D0
         GSP(81)  =     6.55508850D0
         GP2(81)  =     5.89788660D0
         HSP(81)  =     1.59689140D0
         UDD(81)  =    -7.12175200D0
         ZD(81)   =     1.03222310D0
         BETAD(81)=    -0.28812915D0
         ZSN(81)  =     1.84845840D0
         ZPN(81)  =     1.62947930D0
         ZDN(81)  =     1.45638280D0
         PO(9,81) =     1.93483020D0
         F0SD(81) =     0.00000000D0
         G2SD(81) =     0.00000000D0
      end if  ! Tl

      ! Pb: LEAD (82)
      if(iqm_mode.eq.1) then
      ! MNDO: M.J.S.DEWAR, ET.AL ORGANOMETALLICS 4, 1973-1980 (1985)
         USS(82)  =   -47.3196920D0
         UPP(82)  =   -28.8475600D0
         BETAS(82)=    -8.0423870D0
         BETAP(82)=    -3.0000000D0
         ZS(82)   =     2.4982860D0
         ZP(82)   =     2.0820710D0
         ALP(82)  =     1.7283330D0
         EISOL(82)=  -105.8345040D0
         GSS(82)  =     9.8000000D0
         GSP(82)  =     8.3000000D0
         GPP(82)  =     7.3000000D0
         GP2(82)  =     6.5000000D0
         HSP(82)  =     1.3000000D0
         DD(2,82) =     1.5526624D0
         QQ(82)   =     1.4488558D0
         AM(82)   =     0.3601617D0
         AD(82)   =     0.3239309D0
         AQ(82)   =     0.3502057D0
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(82)  =   -30.3227560D0
         UPP(82)  =   -24.4258340D0
         BETAS(82)=    -6.1260240D0
         BETAP(82)=    -1.3954300D0
         ZS(82)   =     3.1412890D0
         ZP(82)   =     1.8924180D0
         ALP(82)  =     1.6200450D0
         EISOL(82)=   -73.4660775D0
         GSS(82)  =     7.0119920D0
         GSP(82)  =     6.7937820D0
         GPP(82)  =     5.1837800D0
         GP2(82)  =     5.0456510D0
         HSP(82)  =     1.5663020D0
         DD(2,82) =     0.9866290D0
         QQ(82)   =     1.5940562D0
         AM(82)   =     0.2576991D0
         AD(82)   =     0.4527678D0
         AQ(82)   =     0.2150175D0
         GUESS1(1,82)= -0.1225760D0
         GUESS2(1,82)=  6.0030620D0
         GUESS3(1,82)=  1.9015970D0
         GUESS1(2,82)= -0.0566480D0
         GUESS2(2,82)=  4.7437050D0
         GUESS3(2,82)=  2.8618790D0
      end if  ! Pb

      ! Bi: BISMUTH (83)
      if(iqm_mode.eq.1) then
      ! MNDO:
      else if(iqm_mode.eq.2) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3: J. J. P. STEWART, J. COMP. CHEM. 12, 320 (1991)
         USS(83)  =   -33.4959380D0
         UPP(83)  =   -35.5210260D0
         BETAS(83)=    -5.6072830D0
         BETAP(83)=    -5.8001520D0
         ZS(83)   =     4.9164510D0
         ZP(83)   =     1.9349350D0
         ALP(83)  =     1.8574310D0
         EISOL(83)=  -109.2774910D0
         GSS(83)  =     4.9894800D0
         GSP(83)  =     6.1033080D0
         GPP(83)  =     8.6960070D0
         GP2(83)  =     8.3354470D0
         HSP(83)  =     0.5991220D0
         DD(2,83) =     0.2798609D0
         QQ(83)   =     1.5590294D0
         AM(83)   =     0.1833693D0
         AD(83)   =     0.6776013D0
         AQ(83)   =     0.2586520D0
         GUESS1(1,83)=  2.5816930D0
         GUESS2(1,83)=  5.0940220D0
         GUESS3(1,83)=  0.4997870D0
         GUESS1(2,83)=  0.0603200D0
         GUESS2(2,83)=  6.0015380D0
         GUESS3(2,83)=  2.4279700D0
      end if  ! Bi

      ! GHO QM-link boundary atom (85)
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! MNDO: Taken from GHO parameters for AM1 for USS, UPP,BETAS, BETAP
      !       Others from M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 99, 4899 (1977)
      !       for Carbon (6)
         USS(85)  =   -52.0286580D0  ! from GHO-AM1 parameters
         UPP(85)  =   -38.7031115D0  !
         BETAS(85)=    -5.5005241D0  !
         BETAP(85)=   -14.6666377D0  !
         ZS(85)   =     1.7875370D0
         ZP(85)   =     1.7875370D0
         ALP(85)  =     2.5463800D0
         EISOL(85)=  -120.5006060D0
         GSS(85)  =    12.2300000D0
         GSP(85)  =    11.4700000D0
         GPP(85)  =    11.0800000D0
         GP2(85)  =     9.8400000D0
         HSP(85)  =     2.4300000D0
         DD(2,85) =     0.8074662D0
         QQ(85)   =     0.6851578D0
         AM(85)   =     0.4494671D0
         AD(85)   =     0.6149474D0
         AQ(85)   =     0.6685897D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1: 
         USS(85)  =   -52.0286580D0  ! from GHO-AT1 parameterization
         UPP(85)  =   -38.7031115D0  !
         BETAS(85)=    -5.5005241D0  !
         BETAP(85)=   -14.6666377D0  !
         ZS(85)   =     1.8086650D0
         ZP(85)   =     1.6851160D0
         ALP(85)  =     2.6482740D0
         EISOL(85)=  -120.8157940D0
         GSS(85)  =    12.2300000D0
         GSP(85)  =    11.4700000D0
         GPP(85)  =    11.0800000D0
         GP2(85)  =     9.8400000D0
         HSP(85)  =     2.4300000D0
         DD(2,85) =     0.8236736D0
         QQ(85)   =     0.7268015D0
         AM(85)   =     0.4494671D0
         AD(85)   =     0.6082946D0
         AQ(85)   =     0.6423492D0
         GUESS1(1,85)=  0.0113550D0
         GUESS2(1,85)=  5.0000000D0
         GUESS3(1,85)=  1.6000000D0
         GUESS1(2,85)=  0.0459240D0
         GUESS2(2,85)=  5.0000000D0
         GUESS3(2,85)=  1.8500000D0
         GUESS1(3,85)= -0.0200610D0
         GUESS2(3,85)=  5.0000000D0
         GUESS3(3,85)=  2.0500000D0
         GUESS1(4,85)= -0.0012600D0
         GUESS2(4,85)=  5.0000000D0
         GUESS3(4,85)=  2.6500000D0
      else if(iqm_mode.eq.3) then
      ! PM3:
         USS(85)  =   -47.2703200D0  ! from GHO-PM3 parameterization
         UPP(85)  =   -35.2877112D0  !
         BETAS(85)=    -2.3820030D0  !
         BETAP(85)=   -14.7041325D0  !
         ZS(85)   =     1.5650850D0
         ZP(85)   =     1.8423450D0
         ALP(85)  =     2.7078070D0
         EISOL(85)=  -111.2299170D0
         GSS(85)  =    11.2007080D0
         GSP(85)  =    10.2650270D0
         GPP(85)  =    10.7962920D0
         GP2(85)  =     9.0425660D0
         HSP(85)  =     2.2909800D0
         DD(2,85) =     0.8332396D0
         QQ(85)   =     0.6647750D0
         AM(85)   =     0.4116394D0
         AD(85)   =     0.5885862D0
         AQ(85)   =     0.7647667D0
         GUESS1(1,85)=  0.0501070D0
         GUESS2(1,85)=  6.0031650D0
         GUESS3(1,85)=  1.6422140D0
         GUESS1(2,85)=  0.0507330D0
         GUESS2(2,85)=  6.0029790D0
         GUESS3(2,85)=  0.8924880D0
      end if ! GHO-atom

      ! Atom 86:
      if(iqm_mode.eq.1 .or. iqm_mode.eq.5) then
      ! Pseudo-atom, used for Link atom for QM/MM treatment.
      ! MNDO: I.ANTES, W.THIEL, TO BE PUBLISHED.
         USS(86)  =   -12.75916989D0
         UPP(86)  =     0.00000000D0
         BETAS(86)=   -11.99647103D0
         BETAP(86)=   -11.99647103D0
         ZS(86)   =     0.93559232D0
         ZP(86)   =     0.93559232D0
         ALP(86)  =     1.65897919D0
         EISOL(86)=   -12.75916989D0
         GSS(86)  =    10.85347036D0
         GSP(86)  =     0.00000000D0
         GPP(86)  =     0.00000000D0
         GP2(86)  =     0.00000000D0
         HSP(86)  =     0.00000000D0
         DD(2,86) =     0.00000000D0
         QQ(86)   =     0.00000000D0
         AM(86)   =     0.39887800D0
         AD(86)   =     0.39887800D0
         AQ(86)   =     0.39887800D0
      else if(iqm_mode.eq.2 .or. iqm_mode.eq.4) then
      ! AM1:
      else if(iqm_mode.eq.3) then
      ! PM3:
      end if  ! Pseudo-link atom

      return
      end subroutine load_qm_parameters


  subroutine initialize_r2cent
  ! 
  ! (originally, routine BDATA4)
  ! definition of logical variables for two-center integrals.
  !
  ! R2CENT(ij,kl)=.false.: two-center integral (ij,kl) is zero.
  ! R2CENT(ij,kl)=.true. : two-center integral (ij,kl) is not zero.
  ! R2CENT(ij,kl)=.true. : is commented out if the integral is zero
  !                        in MNDO/d or AM1/d, but nonzero in general.
  !
  implicit none
  ! local variable
  integer,parameter :: ind_1(14)=(/1,2,3,6,10,11,12,15,18,21,25,28,36,45/),  &
                       ind_2(10)=(/4,5,13,16,17,20,31,34,40,43/),            &
                       ind_3(10)=(/7,8,14,22,23,26,32,35,39,42/),            &
                       ind_4(4) =(/9,27,37,41/),                             &
                       ind_5(8) =(/6,10,18,21,25,28,29,33/)

  ! initialize
  R2CENT(1:45,1:45)=.false.

  ! 1,2,3,6,10,11,12,15,18,21,25,28,36,45
  ! column: 1,2,3,11,12,15,36,45
  R2CENT(ind_1(1:14),1:3)=.true.
  R2CENT(ind_1(1:14),11:12)=.true.
  R2CENT(ind_1(1:14),15)=.true.
  R2CENT(ind_1(1:14),36)=.true.
  R2CENT(ind_1(1:14),45)=.true.

  ! 1,2,3,6,10,11,12,15,18,21,25,28,29,33,36,45
  ! ind_1(1:12) + 29, 33
  ! column: 6,10,18,21,25,28
  R2CENT(ind_1(1:14),6) =.true.
  R2CENT(29,6)=.true.
  R2CENT(33,6)=.true.
  R2CENT(ind_1(1:14),10)=.true.
  R2CENT(29,10)=.true.
  R2CENT(33,10)=.true.
  R2CENT(ind_1(1:14),18)=.true.
  R2CENT(29,18)=.true.
  R2CENT(33,18)=.true.
  R2CENT(ind_1(1:14),21)=.true.
  R2CENT(29,21)=.true.
  R2CENT(33,21)=.true.
  R2CENT(ind_1(1:14),25)=.true.
  R2CENT(29,25)=.true.
  R2CENT(33,25)=.true.
  R2CENT(ind_1(1:14),28)=.true.
  R2CENT(29,28)=.true.
  R2CENT(33,28)=.true.
  ! R2CENT(30, 6) = .TRUE.
  ! R2CENT(30,10) = .TRUE.
  ! R2CENT(30,18) = .TRUE.
  ! R2CENT(30,21) = .TRUE.
  ! R2CENT(30,25) = .TRUE.
  ! R2CENT(30,28) = .TRUE.
  ! R2CENT(30,29) = .TRUE.
  ! R2CENT(30,33) = .TRUE.

  ! 4,5,13,16,17,20,31,34,40,43
  ! column: 4,5,13,16,17,20,31,34,40,43
  R2CENT(ind_2(1:10),4:5)=.true.
  R2CENT(ind_2(1:10),13) =.true.
  R2CENT(ind_2(1:10),16:17)=.true.
  R2CENT(ind_2(1:10),20)=.true.
  R2CENT(ind_2(1:10),31)=.true.
  R2CENT(ind_2(1:10),34)=.true.
  R2CENT(ind_2(1:10),40)=.true.
  R2CENT(ind_2(1:10),43)=.true.

  ! 7,8,14,22,23,26,32,35,39,42
  ! column: 7,8,14,22,23,26,32,35,39,42
  R2CENT(ind_3(1:10),7:8)=.true.
  R2CENT(ind_3(1:10),14)=.true.
  R2CENT(ind_3(1:10),22:23)=.true.
  R2CENT(ind_3(1:10),26)=.true.
  R2CENT(ind_3(1:10),32)=.true.
  R2CENT(ind_3(1:10),35)=.true.
  R2CENT(ind_3(1:10),39)=.true.
  R2CENT(ind_3(1:10),42)=.true.

  ! 9,27,37,41
  ! coulmn: 9,27,37,41
  R2CENT(ind_4(1:4),9)=.true.
  R2CENT(ind_4(1:4),27)=.true.
  R2CENT(ind_4(1:4),37)=.true.
  R2CENT(ind_4(1:4),41)=.true.
  ! R2CENT(19, 9) = .TRUE.
  ! R2CENT(24, 9) = .TRUE.
  ! R2CENT(38, 9) = .TRUE.
  ! R2CENT(19,27) = .TRUE.
  ! R2CENT(24,27) = .TRUE.
  ! R2CENT(38,27) = .TRUE.
  ! R2CENT(19,37) = .TRUE.
  ! R2CENT(24,37) = .TRUE.
  ! R2CENT(38,37) = .TRUE.
  ! R2CENT(19,41) = .TRUE.
  ! R2CENT(24,41) = .TRUE.
  ! R2CENT(38,41) = .TRUE.

  ! 6,10,18,21,25,28,29,33
  ! column: 29,33
  R2CENT(ind_5(1:8),29)=.true.
  R2CENT(ind_5(1:8),33)=.true.

  !     R2CENT( 9,19) = .TRUE.
  !     R2CENT(19,19) = .TRUE.
  !     R2CENT(24,19) = .TRUE.
  !     R2CENT(27,19) = .TRUE.
  !     R2CENT(37,19) = .TRUE.
  !     R2CENT(38,19) = .TRUE.
  !     R2CENT(41,19) = .TRUE.

  !     R2CENT( 9,24) = .TRUE.
  !     R2CENT(19,24) = .TRUE.
  !     R2CENT(24,24) = .TRUE.
  !     R2CENT(27,24) = .TRUE.
  !     R2CENT(37,24) = .TRUE.
  !     R2CENT(38,24) = .TRUE.
  !     R2CENT(41,24) = .TRUE.

  !     R2CENT( 6,30) = .TRUE.
  !     R2CENT(10,30) = .TRUE.
  !     R2CENT(18,30) = .TRUE.
  !     R2CENT(21,30) = .TRUE.
  !     R2CENT(25,30) = .TRUE.
  !     R2CENT(28,30) = .TRUE.
  !     R2CENT(29,30) = .TRUE.
  !     R2CENT(30,30) = .TRUE.
  !     R2CENT(33,30) = .TRUE.

  !     R2CENT( 9,38) = .TRUE.
  !     R2CENT(19,38) = .TRUE.
  !     R2CENT(24,38) = .TRUE.
  !     R2CENT(27,38) = .TRUE.
  !     R2CENT(37,38) = .TRUE.
  !     R2CENT(38,38) = .TRUE.
  !     R2CENT(41,38) = .TRUE.

  !     R2CENT(44,44) = .TRUE.

  return
  end subroutine initialize_r2cent


  subroutine FBINOM(k)
  ! 
  ! Define Factorials and Binomial coefficients for SPD integrals.
  !
  ! F(30)         LOGARITHM OF FACTORIALS.
  ! B(30,30)      BINOMIAL COEFFICIENTS.
  ! B_inv(30,30)  Inverse of B matrix
  !
  !
  implicit none
  integer :: k,i,j
  real(chm_real) :: I_real,b_tmp(k)

  Fbin(1) = zero
  do i=1,K-1
     Fbin(i+1)=Fbin(i)+LOG(zero+i)
  end do

  do i=1,k
     Bbin(i,1) = one
     Bbin_inv(i,1)=one  ! inverse
     do j=2,k
        Bbin(i,j)=zero
        Bbin_inv(i,j)=zero
     end do
  end do
  do i=2,k
     do j=2,i
        Bbin(i,j)=Bbin(i-1,j-1)+Bbin(i-1,j)
     end do
  end do
  !
  ! inverse it, for 
  do i=1,k
     do j=1,k
        if(Bbin(j,i).ne.zero) Bbin_inv(j,i)=one/Bbin(j,i)
     end do
  end do

  return
  end subroutine FBINOM

  subroutine MLIG
  !
  ! Bond parameters for core-core repulsion function in MNDO/d.
  ! There parameters are preliminary until published.
  !
  ! Only for MNDO/d method.
  !
  implicit none
  integer :: NI

  ! *** 10 PARAMETERS FOR Li
  NI          = 3
  MALPB(NI)   = 53
  ALPB( 1,NI) = 1.18506450D0    
  ALPB( 3,NI) = 1.17597499D0    
  ALPB( 6,NI) = 1.14745490D0    
  ALPB( 7,NI) = 1.38231481D0    
  ALPB( 8,NI) = 1.33692461D0    
  ALPB( 9,NI) = 1.26929352D0    
  ALPB(16,NI) = 1.15000000D0    
  ALPB(17,NI) = 1.44205739D0    
  ALPB(35,NI) = 1.31375269D0    
  ALPB(53,NI) = 1.15775309D0    
  ! *** 2 PARAMETERS FOR Na (1 COMMON VALUE)
  NI          = 11
  MALPB(NI)   = 6
  ALPB( 1,NI) = 1.05225212D0    
  ALPB( 6,NI) = 1.05225212D0    
  ! *** 3 PARAMETERS FOR Mg (2 VALUES)
  NI          = 12
  MALPB(NI)   = 16
  ALPB( 1,NI) = 1.35052992D0    
  ALPB( 6,NI) = 1.48172071D0    
  ALPB(16,NI) = 1.48172071D0    
  ! *** 3 PARAMETERS FOR Al (1 COMMON VALUE)
  NI          = 13
  MALPB(NI)   = 13
  ALPB( 1,NI) = 1.38788000D0    
  ALPB( 6,NI) = 1.38788000D0    
  ALPB(13,NI) = 1.38788000D0    
  ! *** 5 PARAMETERS FOR Ti (A. VOITYUK, 6/95)
  !     NI          = 22
  !     MALPB(NI)   = 53
  !     ALPB( 1,NI) = 1.14083935D0
  !     ALPB( 6,NI) = 1.56196589D0
  !     ALPB(17,NI) = 1.70742699D0
  !     ALPB(35,NI) = 1.59201683D0
  !     ALPB(53,NI) = 1.45843102D0
  ! *** 8 PARAMETERS FOR Ti (V. PAGET, 10/97)
  NI          = 22
  MALPB(NI)   = 53
  ALPB( 1,NI) = 1.21817502D0
  ALPB( 6,NI) = 1.63187307D0
  ALPB( 7,NI) = 1.84420350D0
  ALPB( 8,NI) = 1.69737169D0
  ALPB( 9,NI) = 1.67003586D0
  ALPB(17,NI) = 1.75153006D0
  ALPB(35,NI) = 1.68747685D0
  ALPB(53,NI) = 1.57671196D0
  ! *** 2 PARAMETERS FOR Fe (A. VOITYUK)
  NI          = 26
  MALPB(NI)   = 6
  ALPB( 1,NI) = 3.62204080D0    
  ALPB( 6,NI) = 3.22036623D0    
  ! *** 3 PARAMETERS FOR Ni (A. VOITYUK)
  NI          = 28
  MALPB(NI)   = 8
  ALPB( 1,NI) = 2.69496256D0
  ALPB( 6,NI) = 2.85824040D0
  ALPB( 8,NI) = 2.73667307D0
  ! *** 6 PARAMETERS FOR Cu (A. VOITYUK, 6/95)
  ! *** 7 PARAMETERS FOR Cu (V. PAGET, 10/97)
  !     SAME VALUES, ALPB(16,NI) ADDED BY V. PAGET
  NI          = 29
  MALPB(NI)   = 53
  ALPB( 1,NI) = 2.60239308D0
  ALPB( 6,NI) = 2.46276019D0
  ALPB( 9,NI) = 2.55914405D0
  ALPB(16,NI) = 4.00000000D0
  ALPB(17,NI) = 3.40000000D0
  ALPB(35,NI) = 3.10000000D0
  ALPB(53,NI) = 4.00000000D0
  ! *** 7 PARAMETERS FOR Zr (A. VOITYUK, 6/95)
  NI          = 40
  MALPB(NI)   = 53
  ALPB( 1,NI) = 1.18286776D0
  ALPB( 6,NI) = 1.46000000D0
  ALPB( 8,NI) = 1.51086682D0
  ALPB( 9,NI) = 1.44100304D0
  ALPB(17,NI) = 1.47691814D0
  ALPB(35,NI) = 1.44540870D0
  ALPB(53,NI) = 1.40027586D0
  ! *** 6 PARAMETERS FOR Pd (A. VOITYUK, 6/95)
  !     NI          = 46
  !     MALPB(NI)   = 46
  !     ALPB( 1,NI) = 2.23402481D0
  !     ALPB( 6,NI) = 2.24851638D0
  !     ALPB( 8,NI) = 2.25324052D0
  !     ALPB( 9,NI) = 2.19232373D0
  !     ALPB(17,NI) = 2.37707417D0
  !     ALPB(46,NI) = 2.40000000D0
  ! *** 8 PARAMETERS FOR Pd (V. PAGET, 10/97)
  NI          = 46
  MALPB(NI)   = 46
  ALPB( 1,NI) = 2.25000000D0
  ALPB( 6,NI) = 2.25000000D0
  ALPB( 7,NI) = 2.42323000D0
  ALPB( 8,NI) = 2.37312816D0
  ALPB( 9,NI) = 2.25000000D0
  ALPB(15,NI) = 3.00000000D0
  ALPB(17,NI) = 2.54161431D0
  ALPB(46,NI) = 2.40000000D0
  ! *** 4 PARAMETERS FOR Ag (A. VOITYUK)
  NI          = 47
  MALPB(NI)   = 47
  ALPB( 1,NI) = 2.96977640D0
  ALPB( 6,NI) = 2.22156299D0
  ALPB(17,NI) = 2.54475403D0
  ALPB(47,NI) = 5.00000000D0

  return
  end subroutine MLIG

  subroutine inighd(ni)
  !
  ! one-center two-electron integrals for SPD-basis.
  ! NI is the atomic number of the current element.
  !
  implicit none
  integer :: ni
 
  ! local variables
  integer :: i,j,ii

  ! local variables
  integer :: NS,NP,ND
  real(chm_real):: ES,EP,ED,R011,R013,R033,R233,R122,R016,R036,R066, &
                   R155,R125,R244,R236,R266,R234,R246,R355,R466
  real(chm_real),parameter :: S3 =0.17320508075689D+01,  &
                              S5 =0.22360679774998D+01,  &
                              S15=0.38729833462074D+01
  real(chm_real),parameter :: r_3    = one/three,  &
                              r_5    = PT2,        & ! one/five
                              r_7    = one/seven,  &
                              r_15   = one/15.0d0, & ! ... careful with 
                              r_25   = one/25.0d0, & !     other names
                              r_35   = one/35.0d0, & !     defined above. 
                              r_49   = one/49.0d0, &
                              r_125  = one/125.0d0,&
                              r_147  = one/147.0d0,&
                              r_245  = one/245.0d0,&
                              r_441  = one/441.0d0,&
                              r_s5   = one/S5,     &
                              r_s15  = one/S15

  ! check.
  if(LORBS(ni).lt.9) return

  ! III =(/2*1,8*2,8*3,18*4,18*5,30*6,2,2/) ; 1,2,3,4,5,6
  ! IIID=(/30*3,18*4,32*5,4*6,3,3/)         ; 3,4,5,6
  NS     = III(NI)
  NP     = III(NI)
  ND     = IIID(NI)
  ES     = ZSN(NI)
  EP     = ZPN(NI)
  ED     = ZDN(NI)

  ! *** SLATER-CONDON PARAMETERS (Rlij).
  !     FIRST  DIGIT (l)  L QUANTUM NUMBER OF SLATER-CONDON PARAMETER.
  !     SECOND DIGIT (i)  SS 1, SP 2, PP 3, SD 4, PD 5, DD 6 ELECTRON 1.
  !     SECOND DIGIT (j)  SS 1, SP 2, PP 3, SD 4, PD 5, DD 6 ELECTRON 2.
  R011   = RSC(0,NS,ES,NS,ES,NS,ES,NS,ES)
  R013   = RSC(0,NS,ES,NS,ES,NP,EP,NP,EP)
  R033   = RSC(0,NP,EP,NP,EP,NP,EP,NP,EP)
  R233   = RSC(2,NP,EP,NP,EP,NP,EP,NP,EP)
  R122   = RSC(1,NS,ES,NP,EP,NS,ES,NP,EP)
  R016   = RSC(0,NS,ES,NS,ES,ND,ED,ND,ED)
  R036   = RSC(0,NS,EP,NS,EP,ND,ED,ND,ED)
  R066   = RSC(0,ND,ED,ND,ED,ND,ED,ND,ED)
  R155   = RSC(1,NS,EP,ND,ED,NS,EP,ND,ED)
  R125   = RSC(1,NS,ES,NS,EP,NS,EP,ND,ED)
  R244   = RSC(2,NS,ES,ND,ED,NS,ES,ND,ED)
  R236   = RSC(2,NS,EP,NS,EP,ND,ED,ND,ED)
  R266   = RSC(2,ND,ED,ND,ED,ND,ED,ND,ED)
  R234   = RSC(2,NS,EP,NS,EP,NS,ES,ND,ED)
  R246   = RSC(2,NS,ES,ND,ED,ND,ED,ND,ED)
  R355   = RSC(3,NS,EP,ND,ED,NS,EP,ND,ED)
  R466   = RSC(4,ND,ED,ND,ED,ND,ED,ND,ED)

  ! save slater-condon parameters.
  F0DD(NI) = R066
  F2DD(NI) = R266
  F4DD(NI) = R466
  F0PD(NI) = R036
  F2PD(NI) = R236
  G1PD(NI) = R155
  G3PD(NI) = R355
  ! KEEP PREDEFINED SLATER-CONDON PARAMETERS (IF REQUESTED).
  if(IF0SD(ni).gt.0) then
     R016  = F0SD(ni)
  else
     F0SD(ni) = R016
  end if
  if(IG2SD(ni).gt.0) then
     R244  = G2SD(ni)
  else
     G2SD(ni) = R244
  end if
  ! compute one-center two-electron integrals from the slater-condon
  ! parmaters.

  ! Integals involing sp-orbitals.
  GSS(NI) = R011
  GPP(NI) = R033 + FOUR*r_25*R233        ! /25.0d0
  GP2(NI) = R033 - TWO *r_25*R233
  HPP(NI) = THREE*r_25*R233
  ! in mndod/d and am1/d, these two parameters are left as independent variables
  ! to allow fine tuning.
  ! GSP(NI) = R013       
  ! HSP(NI) = R122/THREE 

  ! Integrals involving also d-orbitals.
  REPD( 1,NI) =  R016
  REPD( 2,NI) =  2.0d0*r_3*r_s5*R125      ! /(THREE*S5)
  REPD( 3,NI) =  r_S15*R125               ! one/S15
  REPD( 4,NI) =  2.0d0*r_5*r_S5*R234      ! /(five*S5)
  REPD( 5,NI) =  R036 + 4.0d0*r_35*R236   ! /35.0d0
  REPD( 6,NI) =  R036 + 2.0d0*r_35*R236
  REPD( 7,NI) =  R036 - 4.0d0*r_35*R236
  REPD( 8,NI) = -r_3*r_S5*R125
  REPD( 9,NI) =  SQRT(3.0d0*r_125)*R234    ! /125.0d0
  REPD(10,NI) =  S3   *r_35*R236           ! /35.0d0
  REPD(11,NI) =  3.0d0*r_35*R236
  REPD(12,NI) = -0.2d0*r_S5*R234
  REPD(13,NI) =  R036 - 2.0d0*r_35*R236
  REPD(14,NI) = -2.0d0*S3*r_35*R236
  REPD(15,NI) = -REPD( 3,NI)
  REPD(16,NI) = -REPD(11,NI)
  REPD(17,NI) = -REPD( 9,NI)
  REPD(18,NI) = -REPD(14,NI)
  REPD(19,NI) =  0.2d0*R244
  REPD(20,NI) =  2.0d0*r_7*r_s5*R246
  REPD(21,NI) =  REPD(20,NI)*0.5d0
  REPD(22,NI) = -REPD(20,NI)
  REPD(23,NI) =  4.0d0   *r_15*R155 + 27.0D0   *r_245*R355
  REPD(24,NI) =  2.0d0*S3*r_15*R155 -  9.0d0*S3*r_245*R355
  REPD(25,NI) =  r_15*R155        + 18.D0  *r_245*R355
  REPD(26,NI) = -S3*r_15*R155 + 12.0d0*S3*r_245*R355
  REPD(27,NI) = -S3*r_15*R155 -  3.0d0*S3*r_245*R355
  REPD(28,NI) = -REPD(27,NI)
  REPD(29,NI) =  R066 + 4.0d0*r_49*R266 + 4.0d0 * r_49*R466
  REPD(30,NI) =  R066 + 2.0d0*r_49*R266 - 24.0D0*r_441*R466
  REPD(31,NI) =  R066 - 4.0d0*r_49*R266 + 6.0d0 *r_441*R466
  REPD(32,NI) =  SQRT(3.0d0*r_245)*R246
  REPD(33,NI) =  0.2d0*R155 + 24.0D0*r_245*R355
  REPD(34,NI) =  0.2d0*R155 -  6.0d0*r_245*R355
  REPD(35,NI) =  3.0d0*r_49*R355
  REPD(36,NI) =  r_49*R266     + 30.0d0*r_441*R466
  REPD(37,NI) =  S3 *r_49*R266 -  5.0d0*S3*r_441*R466
  REPD(38,NI) =  R066 - 2.0d0*r_49*R266 -  4.0d0*r_441*R466
  REPD(39,NI) = -2.0d0*S3*r_49*R266 + 10.0d0*S3*r_441*R466
  REPD(40,NI) = -REPD(32,NI)
  REPD(41,NI) = -REPD(34,NI)
  REPD(42,NI) = -REPD(35,NI)
  REPD(43,NI) = -REPD(37,NI)
  REPD(44,NI) =  3.0d0*r_49*R266 + 20.0d0*r_441*R466
  REPD(45,NI) = -REPD(39,NI)
  REPD(46,NI) =  0.2d0*R155 - 3.0d0*r_35*R355
  REPD(47,NI) = -REPD(46,NI)
  REPD(48,NI) =  4.0d0*r_49*R266 + 15.0D0*r_441*R466
  REPD(49,NI) =  3.0d0*r_49*R266 -  5.0d0*r_147*R466
  REPD(50,NI) = -REPD(49,NI)
  REPD(51,NI) =  R066 + 4.0d0*r_49*R266 - 34.D0*r_441*R466
  REPD(52,NI) =  35.D0*r_441*R466

  return

  ! statement function (inline function)
  contains
     !
     function rsc(K,NA,EA,NB,EB,NC,EC,ND,ED)
     !
     ! calculate the radial part of one-center two-electron integrals (Slater-Condon parameter).
     !
     ! K     - type of integral, can be equal to 0,1,2,3,4 in SPD-basis
     ! NA,NB - principle quantum number of AO, electron 1
     ! EA,EB - exponents of AO, electron 1
     ! NC,ND - principle quantum number of AO, electron 2
     ! EC,ED - exponents of AO, electron 2
     !
     implicit none
     integer :: K,NA,NB,NC,ND
     real(chm_real):: EA,EB,EC,ED

     ! local variables
     real(chm_real)::RSC

     integer :: N,NAB,NCD,M,I,M1,M2
     real(chm_real):: AEA,AEB,AEC,AED,ECD,EAB,E,AE,A2,ACD,AAB, &
                      C,S0,S1,S2,S3,E_rECD

     AEA    = LOG(EA)
     AEB    = LOG(EB)
     AEC    = LOG(EC)
     AED    = LOG(ED)
     NAB    = na+nb
     NCD    = nc+nd
     ECD    = EC+ED
     EAB    = EA+EB
     E      = ECD+EAB
     N      = NAB+NCD
     AE     = LOG(E)
     A2     = LOG(two)
     ACD    = LOG(ECD)
     AAB    = LOG(EAB)
     C      = EXP(Fbin(n)+na*AEA+NB*AEB+nc*AEC+nd*AED+PT5*(AEA+AEB+AEC+AED)+A2*(n+2) &
                         -PT5*(Fbin(2*na+1)+Fbin(2*nb+1)+Fbin(2*nc+1)+Fbin(2*nd+1))  &
                         -AE*n)
     C      = C*EV
     S0     = one/E
     S1     = zero
     S2     = zero
     M      = NCD-k
     E_rECD  = E/ECD
     do I=1,M
        S0     = S0*E_rECD   ! E/ECD
        S1     = S1+S0*(Bbin(NCD-k,i)-Bbin(NCD+k+1,i))*Bbin_inv(n,i) ! /Bbin(N,I)
     end do
     m1     = m+1
     m2     = NCD+k+1
     do i=m1,m2
        S0     = S0*E_rECD   ! E/ECD
        S2     = S2+S0*Bbin(m2,i)*Bbin_inv(n,i)  ! /Bbin(N,I)
     end do
     S3     = EXP(AE*n-ACD*m2-AAB*(NAB-k))*Bbin_inv(n,m2)  ! /Bbin(N,M2)
     RSC    = C*(S1-S2+S3)

     return
     end function rsc
  end subroutine inighd

  subroutine ddpohy(ni)
  !
  ! calculate charge separations and additive terms used to compute
  ! the two-center two-electron integrals in MNDO/d and AM1/d.
  ! NI is the atomic number of the current element.
  !
  ! in the case of an SP basis, the charge separations and additive terms
  ! are equivalent to MNDO, AM1, and PM3. so, it can be used for those atoms.
  !
  ! DD(6,*) : charge separation from function AIJL.
  ! PO(9,*) : additive terms from function POIJ.
  ! refer eq. 12-16 of TCA paper for DD, and eq. 19-26 for PO.
  !
  ! index of DD and PO : SS 1, SP 2, PP 3, SD 4, PD 5, DD 6.
  ! multipole          :  L=0,  L=1,  L=2,  L=2,  L=1,  L=2.
  ! special index of PO: PP 7, DD 8.
  ! multipole          :  L=0,  L=0.
  ! atomic core        : ADDITIVE TERM PO(9,NI)
  !
  implicit none
  integer :: ni

  ! local variables
  integer :: n,np,nd
  real(chm_real):: AIJ22, AIJ52,AIJ43,AIJ63, D, DA, FG, Z1, Z2, Z3

  ! AIJ-values are computed as defined in eq. (7) of TCA paper and stored
  ! as variables AIJ. with two digits at the end.
  ! first digit:  1 SS, 2 SP, 3 PP, 4 SD, 5 PD, 6 DD.
  ! 2nd   digit:  L+1 from definition of multiple.

  ! S basis: there is only one additive term.
  PO(1,ni) = PT5*EV/GSS(ni)
  if(LORBS(ni).eq.1) return

  ! SP basis: charge separations and additive terms must be computed.
  N        = III(NI)
  NP       = MAX(N,2)
  Z1       = ZS(NI)
  Z2       = ZP(NI)
  AIJ22    = AIJL(Z1,Z2,N,NP,1)
  DD(2,NI) = AIJ22/SQRT(THREE)
  D        = DD(2,NI)
  FG       = HSP(NI)
  PO(2,NI) = POIJ(1,D,FG)
  DD(3,NI) = SQRT((2*NP+1)*(2*NP+2)/20.0D0) / ZP(NI)
  D        = DD(3,NI)*SQRT(TWO)
  FG       = HPP(NI)
  PO(3,NI) = POIJ(2,D,FG)
  PO(7,NI) = PO(1,NI)
  IF(LORBS(NI).EQ.4) RETURN

  ! SPD basis: additive terms involving D orbitals.
  !            note extra factor of SQRT2 for the charge separations.
  !            DD(i,ni) with i=4,6, which refer to square quadrupoles, for
  !            simplification of the code in REPPD.
  Z3       = ZD(NI)
  ND       = IIID(NI)
  AIJ52    = AIJL(Z2,Z3,NP,ND,1)
  AIJ43    = AIJL(Z1,Z3,N ,ND,2)
  AIJ63    = AIJL(Z3,Z3,ND,ND,2)
  !     SD
  DA       = one/SQRT(15.0D0)
  D        = SQRT(TWO*AIJ43*DA)   ! SQRT(AIJ43*DA)*SQRT(TWO)
  FG       = REPD(19,NI)
  DD(4,NI) = D
  PO(4,NI) = POIJ(2,D,FG)
  !     PD
  D        = AIJ52*SQRT(PT2)    ! one/sqrt(five)=sqrt(one/five)=sqrt(0.2)
  FG       = REPD(23,NI)-1.8D0*REPD(35,NI)
  !     PREVIOUS STATEMENT AS IN THE TCA PAPER,
  !     NEXT STATEMENT AS A POSSIBLE ALTERNATIVE.
  !     FG       = REPD(33,NI)-1.6D0*REPD(35,NI) 
  DD(5,NI) = D
  PO(5,NI) = POIJ(1,D,FG)
  !     DD
  FG       = PT2*(REPD(29,NI)+TWO*REPD(30,NI)+TWO*REPD(31,NI))
  PO(8,NI) = POIJ(0,ONE,FG)
  D        = SQRT(TWO*AIJ63/seven)
  FG       = REPD(44,NI)-(20.0D0/35.0D0)*REPD(52,NI)
  DD(6,NI) = D
  PO(6,NI) = POIJ(2,D,FG)

  return

  ! this is in-lined function.
  contains
     ! general formula for AIJ-values (see above) : see eq. (7) of TCA paper.
     real(chm_real) function AIJL(Z1,Z2,N1,N2,L)
     !
     implicit none
     integer       :: N1,N2,L
     real(chm_real):: Z1,Z2, r_z1_z2
     !
     ! FC(i) contains the factorials of (i-1).
     real(chm_real),parameter :: FC(17)=(/  &
              1.0D0,          1.0D0,       2.0D0,        6.0D0,        24.0D0, &
            120.0D0,        720.0D0,    5040.0D0,    40320.0D0,    362880.0D0, &
        3628800.0D0,   39916800.0D0,4.790016D+08,6.2270208D+09,8.71782912D+10, &
    1.307674368D+12,2.092278989D+13/)
     !
     r_z1_z2=one/(Z1+Z2)
     AIJL=FC(N1+N2+L+1) / SQRT( FC(2*N1+1)*FC(2*N2+1) )              &
                        *(two*Z1*r_z1_z2)**N1 * SQRT(two*Z1*r_z1_z2) &
                        *(two*Z2*r_z1_z2)**N2 * SQRT(two*Z2*r_z1_z2) &
                        /(Z1+Z2)**L
     return
     end function AIJL

     ! function POIJ
     real(chm_real) function POIJ(L,D,FG)
     !
     ! determine additive terms RHO=POIJ for two-center two-electron
     ! integrals
     ! from the requirement that the appropriate one-center two-electron
     ! integrals are reproduced.
     !
     ! see eq. (19)-(26) in TCA paper.
     ! L     L quantum number of additive term.
     ! D     Charge separation
     ! FG    reference one-center integral. (or slater-condon parameter).
     !
     ! special convention in case L=2:
     ! the input value of D (as defined in the calling routine) equals D2*SQRT(two),
     ! with D2 defined in eq. (14)-(16) of the TCA paper. this special convention
     ! for L=2 should be kept in mind when comparing the code below with eq.(24)-(26) 
     ! of the TCA paper. the convention is motivated by code simplifications in
     ! subroutine REPPD which arise when values of D2*SQRT(two) are stored in 
     ! DD(4,NI) and DD(6,NI).
     !
     implicit none
     integer :: L
     real(chm_real):: D,FG

     ! local variables
     integer :: i
     real(chm_real):: DSQ,EV4,EV8,A1,A2,DELTA,Y1,Y2,F1,F2
     real(chm_real),parameter :: EPSIL=1.0D-08,G1=0.382D0,G2=0.618D0, &
                                 r_8=0.125d0  ! 1/8
     integer, parameter :: NITER=100

     ! terms for SS
     if(L.eq.0) then
        POIJ = PT5*EV/FG
     else
     ! higher terms.
        DSQ    = D*D
        EV4    = EV*PT25
        EV8    = EV*r_8
        A1     = PT1
        A2     = five
        do I=1,NITER
           DELTA  = A2-A1
           if(DELTA.lt.EPSIL) exit
           Y1     = A1 + DELTA*G1
           Y2     = A1 + DELTA*G2
           if(L.eq.1) then
              F1= (EV4*(ONE/Y1-ONE/SQRT(Y1**2+DSQ)) - FG)**2
              F2= (EV4*(ONE/Y2-ONE/SQRT(Y2**2+DSQ)) - FG)**2
           else if(L.eq.2) then
              F1= (EV8*( ONE/Y1-TWO/SQRT(Y1**2+DSQ*PT5) +ONE/SQRT(Y1**2+DSQ)) - FG)**2
              F2= (EV8*( ONE/Y2-TWO/SQRT(Y2**2+DSQ*PT5) +ONE/SQRT(Y2**2+DSQ)) - FG)**2
           end if
           if(F1.lt.F2) then
              A2  = Y2
           else
              A1  = Y1
           end if
        end do
        ! define additive terms after convergence of iteractions.
        if(F1.ge.F2) then
           POIJ = A2
        else
           POIJ = A1
        end if
     end if
     return
     end function POIJ
  end subroutine ddpohy

  function eatom(ni,iss,ipp,idd)
  ! 
  ! total energy for an atom with the atomic number NI
  ! which is in a configuration with the occupation numbers
  ! iss,ipp,idd for the s,p,d orbitals.
  !
  ! Note: EATOM is only called with iss=0,ipp=0, idd=0.
  !       so, only need to consider them with 0 values.
  ! 
  implicit none
  integer :: ni,iss,ipp,idd
  real(chm_real):: EATOM

  ! prefactors for exchange terms.
  real(chm_real),parameter:: cp(6)  =(/ 0.D0,  -1.D0,  -3.D0,  &
                                       -1.D0,   0.D0,   0.D0/)
  real(chm_real),parameter:: cd2(10)=(/ 0.D0, -58.D0, -93.D0,  &
                                     -105.D0,-175.D0,-105.D0,  &
                                      -93.D0, -58.D0,   0.D0,  &
                                        0.D0/)
  real(chm_real),parameter:: cd4(10)=(/ 0.D0,   5.D0, -30.D0,  &
                                     -105.D0,-175.D0,-105.D0,  &
                                      -30.D0,   5.D0,   0.D0,  &
                                        0.D0/)
!  real(chm_real),parameter:: occup(-1:10)=(/    &
!                0.0d0,0.0d0,                    &  ! -1 introduce for zero
!                1.0d0,2.0d0,3.0d0,4.0d0, 5.0d0, &  !
!                6.0d0,7.0d0,8.0d0,9.0d0,10.0d0/)

  integer:: is,ip,idL
  real(chm_real):: E,add,ris,rip,rid
  real(chm_real),parameter :: r_5=0.2d0,          & ! 1/5
                              r_15=1.0d0/15.0d0,  & ! 1/15
                              r_70=1.0d0/70.0d0,  & ! 1/70
                              r_441=1.0d0/441.0d0   ! 1/441

  ! find occupation number.
  ! if, iss=ipp=idd=0, find the number based on the atomic number.
  if((ISS+IPP+IDD).eq.0) then  ! this case. use default occupation.
     is  = IOS(ni)             ! 0~2
     ip  = IOP(ni)             ! 0~6
     idL = IOD(ni)             ! 0~10
  else
     is  = iss
     ip  = ipp
     idL = idd
  end if
  ! contribution from S electrons
  E = zero
  if(is.eq.1) then
     E   = E + USS(ni)
  else if(is.eq.2) then
     E   = E + TWO*USS(ni) + GSS(ni)
  end if
  ! contribution from P electrons.
  if(ip.ge.1) then
     E   = E + IP*UPP(ni) +IP*(IP-1)*PT5*GP2(ni) + CP(IP)*HPP(ni)
     if(is.eq.1) then
       E = E + IP*GSP(ni) - MIN(IP,3)*HSP(ni)
     else if(is.eq.2) then
       E = E + IP*(TWO*GSP(ni)-HSP(ni))
     end if
  end if
  ! contribution from D electrons.
  if(LORBS(ni).ge.9 .and. idL.ge.1) then
     ADD = F0DD(ni)-(14.D0*r_441)*(F2DD(ni)+F4DD(ni))
     E   = E + IDL*UDD(ni) + IDL*(IDL-1)*PT5*ADD               &
             +(CD2(IDL)*F2DD(ni)+CD4(IDL)*F4DD(ni))*r_441
     if(is.eq.1) then
       E = E + IDL*F0SD(ni) - MIN(IDL,5)*G2SD(ni)*r_5
     else if(is.eq.2) then
       E = E + IDL*(TWO*F0SD(ni)-G2SD(ni)*r_5)
     end if
     if(ip.ge.1) then
        E   = E + IP*IDL*(F0PD(ni)-r_15*G1PD(ni)-THREE*r_70*G3PD(ni))
     end if
  end if
  EATOM  = E

!  ! real values of occupation number.
!  ris = occup(is); rip = occup(ip); rid = occup(id)
!
!  ! contribution from S electrons
!  ! if is==1, E=USS ; if is==2, E=two*USS+GSS
!  if(is.eq.1) then
!     E = ris*USS(ni)
!  else
!     E = ris*USS(ni)+two*GSS(ni)
!  end if
!
!  ! contribution from P electrons.
!  if(ip.ge.1) then
!     E = E + rip*UPP(ni)+rip*(rip-one)*PT5*GP2(ni)+cp(ip)*HPP(ni)
!     if(is.eq.1) then
!        E = E + rip*GSP(ni) - occup(MIN(ip,3))*HSP(ni)
!     else if(is.eq.2) then
!        E = E + rip*(two*GSP(ni) - HSP(ni))
!     end if
!  end if
!  ! contribution from D electrons.
!  if(LORBS(ni).ge.9 .and. id.ge.1) then
!     add= F0DD(ni)-(14.D0*r_441)*(F2DD(ni)+F4DD(ni))
!     E  = E + rid*UDD(ni) + rid*(rid-one)*PT5*add+ &
!             (cd2(id)*F2DD(ni)+cd4(id)*F4DD(ni))*r_441
!     if(is.eq.1) then
!        E = E + rid*F0SD(ni) - occup(MIN(id,5))*G2SD(ni)*r_5
!     else if(is.eq.2) then
!        E = E + rid*(two*F0SD(ni)-G2SD(ni)*r_5)
!     end if
!     if(ip.ge.1) E=E + rip*rid*(F0PD(ni)-r_15*G1PD(ni)-three*r_70*G3PD(ni))
!  end if
!  EATOM  = E

  return
  end function eatom

#endif
end module qm1_parameters
! end
