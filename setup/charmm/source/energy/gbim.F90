module gbim
  use chm_kinds
  use dimens_fcm
  implicit none
  ! This common block file contains the additional control parameters
  ! for GBIM calculations. 
  ! 
  ! GBIM (Generalized Born model with  Implicit Membrane) is
  ! a modification of CHARMM GENBORN module (B.Dominy & Brooks C.L.III)
  ! by including an implicit membrane in the linearized variant of
  ! Still pair-wise approach.
  ! In additon GBIM allows  optional choice of the intramolecular
  ! dielectric constant.   
  ! 
  ! V. Spassov, L.Yan, S.Szalma (2002) J.Phys. Chem, B, 8726 - 8738.
  ! 
  ! The variables include:
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  ! Note, that  the polarization energy Gpol is given by:
  !
  !                                                      q q
  !                                 N   N                 i j
  ! G   = -C  (1/epsmol-1/eps){1/2 sum sum ----------------------------------}
  !  pol    el                     i=1 j=1 [r^2+alpha *alpha exp(-D  )]^(0.5)
  !                                          ij      i      j      ij
  !
  !+++++ membrane part +++++++++++++++++++++++++++++++++++++++++++++++++++
  ! This part contains the additional membrane parameters for GBIM
  ! calculations.  See also genborn module (genborn.src)
  !=======================================================================
  !
  ! EPS_MOL  =   Value of internal dielectric constant for molecular
  !              (non-solvent) region in Generalized Born  model. 
  !  
  ! memb_IM   =  logical flag: 1 - GB with implicit membrane,
  !                            0 - without membrane
  !  
  ! MembDir_IM =  the  membrane normal is colinear to:
  !               1 - X axis, 2 - Y or  3 - Z [default]
  !    
  ! OUTM_IM   =   logical array: 0 - if the atom is inside,
  !                              1 - outside of the dielectric slab
  ! 
  ! Lmemb_IM  =   membrane thickness in Angstr.
  !
  ! Gamma_IM  =   empirical parameter (see the  GBIM method) 
  !  
  ! Memb_Center = membrane position along the axis (X or Y or Z) 
  !    
  !========================================================================

  real(chm_real)  Eps_MOL, Lmemb_IM, Gamma_IM, MembCenter_IM
  logical,allocatable,dimension(:) :: OUTM_IM
  Integer MembDir_IM
  Logical QGBIMb, memb_IM

contains
  subroutine gbim_init
    qgbimb = .false.
    return
  end subroutine gbim_init

!
! GBIM (Generalized Born model with  Implicit Membrane) is
! a modification of CHARMM GENBORN module (B.Dominy & Brooks C.L.III)
! by including an implicit membrane in the linearized variant of
! Still pair-wise approach.
! 
! V. Spassov, L.Yan, S.Szalma (2002) J.Phys. Chem, B, 8726 - 8738.
!
Subroutine SetGBIM(comlyn, comlen)
  !-----------------------------------------------------------------------
  ! This routine reads in and sets up the data structures for GBIM
  ! calculations.
  !
  use gb_common, only:p,r_gb,vol_gb,alph_gb,sigx_gb,sigy_gb, &
     sigz_gb,t_gb,qanalys,eps_gb,cutalph, gb_atm
  use chm_kinds
  use exfunc
  use stream
  use string
  use dimens_fcm
  use psf
  use memory
  use number
  use coord
  ! use gbim

  use inbnd
  implicit none

  Character(len=*) :: Comlyn
  Integer Comlen
  Logical QWeight

  Integer i
  Character(len=2) :: C(6)
  Data C /'P1','P2','P3','P4','P5','P6'/

  If (IndxA(Comlyn, Comlen,'CLEA') > 0) then
     If ( .not. QGBIMb ) Then
        Call WrnDie(-1,'<SetGBIM>',  &
             'Called Clear w/o GBorn being active')
        Return
     Else
        QGBIMb = .false.
     Endif
     ! Clear up heap space and return

     If ( Prnlev  >=  2 ) then
        Write(Outu,'(A)')' Clearing Generalized Born Arrays'
     Endif

     call chmdealloc('gbim.src','SetGBIM','r_gb',Natom,crl=r_gb)
     call chmdealloc('gbim.src','SetGBIM','vol_gb',Natom,crl=vol_gb)
     call chmdealloc('gbim.src','SetGBIM','alph_gb',Natom,crl=alph_gb)

     call chmdealloc('gbim.src','SetGBIM','sigx_gb',Natom,crl=sigx_gb)
     call chmdealloc('gbim.src','SetGBIM','sigy_gb',Natom,crl=sigy_gb)
     call chmdealloc('gbim.src','SetGBIM','sigz_gb',Natom,crl=sigz_gb)
     call chmdealloc('gbim.src','SetGBIM','t_gb',Natom,crl=t_gb)

     call chmdealloc('gbim.src','SetGBIM','OUTM_IM',Natom,log=OUTM_IM)

     if (QAnalys) then
        call chmdealloc('gbim.src','SetGBIM','gb_atm',Natom,crl=gb_atm)
     else
        call chmdealloc('gbim.src','SetGBIM','gb_atm',1,crl=gb_atm)
     endif

     QAnalys = .false.
     Return

  Endif

  QGBIMb = .true.

  Eps_GB = GtrmF(Comlyn, Comlen, 'EPSILON', Eighty)
  Eps_MOL = GtrmF(Comlyn, Comlen, 'EPSMOL', One)
  Cutalph = GtrmF(Comlyn, Comlen, 'CUTA', Mega)
  QWeight = (IndxA(Comlyn, Comlen,'WEIG') > 0)
  QAnalys = (IndxA(Comlyn, Comlen,'ANAL') > 0)
  Do i = 1, 5
     P(i) = GtrmF(Comlyn, Comlen, C(i), Zero)
  Enddo
  P(6) = GtrmF(Comlyn, Comlen, 'LAMBDA', One)
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  MembDir_IM = 0
  MembCenter_IM = 0.
  memb_IM = .false.
  Lmemb_IM =  GtrmF(Comlyn, Comlen, 'TMEMB', Zero)
  MembCenter_IM = GtrmF(Comlyn, Comlen, 'CENTER', Zero) 

  If ( Prnlev  >=  2 ) write(outu,*) 'L_memb=', Lmemb_IM

  IF(Lmemb_IM  >  Zero) Then

     memb_IM = .true.
     !................  The default value for  Gamma_IM is assumed 0.5  

     Gamma_IM =  GtrmF(Comlyn, Comlen, 'GAMMA', HALF)

     If ( Prnlev  >=  2 ) write(outu,*) 'Gamma=',Gamma_IM

     !..............Should be better  without default value for Gamma_IM??

     MembDir_IM = 0
     If(IndxA(Comlyn, Comlen,'XMDIR') > 0)   then
        MembDir_IM = 1
     Endif

     If(IndxA(Comlyn, Comlen,'YMDIR') > 0)  then
        MembDir_IM = 2
     Endif

     If(IndxA(Comlyn, Comlen,'ZMDIR') > 0)  then
        MembDir_IM = 3
     Endif

     IF ( Prnlev  >=  2 ) then
        If(membDir_IM  ==  3) then
           Write(Outu,'(A,f12.3,A )')  &
                'Membrane  thickness along Z (TMEMB) =', &
                Lmemb_IM , ' Angstr.' 
        EndIf
        If(membDir_IM  ==  2) then
           Write(Outu,'(A,f12.3,A )')  &
                'Membrane  thickness along Y (TMEMB) =', &
                Lmemb_IM , ' Angstr.' 
        EndIf

        If(membDir_IM  ==  1) then
           Write(Outu,'(A,f12.3,A )')  &
                'Membrane  thickness along X (TMEMB) =', &
                Lmemb_IM , ' Angstr.' 
        EndIf

        Write(Outu,'(A,f12.3,A )')  &
             'Centred at =', &
             MembCenter_IM , ' Angstr.' 

     ENDIF

     If((membDir_IM  ==  0).and. memb_IM)  then
        Write(Outu,'(A)')  &
             'WARNING: NO XMDIR, YMDIR or ZMDIR READ'
        Write(Outu,'(A)')  &
             'WARNING: ZMDIR [default] accepted'

        MembDir_IM = 3
     EndIf
     Lmemb_IM = Half*Lmemb_IM
     !........  the half membrane thickness will be used after this point
  EndIF
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  If ( Prnlev  >=  2 ) then
     Write(Outu,'(A)')  &
          ' Generalized Born energy and force terms  will be used'
     Write(Outu,'(A)')  &
          ' The parameters are (P(6) = Lambda)'
     Write(Outu,'(3(A,A,f12.5,2x))') &
          (C(i),' =',P(i),i=1,6)

     Write(Outu,'(A, f12.5)')  &
          ' The molecular  dielectric constant is eps = ', Eps_mol
     Write(Outu,'(A, f12.5)')  &
          ' The  solvent   dielectric constant is eps = ', Eps_GB
     Write(Outu,'(A, f20.5)') &
          ' The maximum allowed GB radius, alpha = ', CutAlph
     If (Qweight) then
        Write(Outu,'(A)')  &
             ' Radii for taken from WMAIN'
     Endif
     If (QAnalys) then
        Write(Outu,'(A)')  &
             'Atomic contributions available in GBAtm through scalar commands'
     Endif
  Endif
  !-----------------------------------------------------------------------
  ! Allocate heap space for needed arrays

  if(allocated(r_gb) .or.  &
       (allocated(vol_gb) .or. allocated(alph_gb))) then
     write(outu,'("ALPH_GB,r_gb.vol_gb allready allocated")')
     call wrndie(-4,"<gbim.src>SETGBIM", &
          "gbim already active when gbim command given")
  endif
  call chmalloc('gbim.src','SetGBIM','r_gb',natom,crl=r_gb)
  call chmalloc('gbim.src','SetGBIM','vol_gb',natom,crl=vol_gb)
  call chmalloc('gbim.src','SetGBIM','alph_gb',natom,crl=alph_gb)

  call chmalloc('gbim.src','SetGBIM','sigx_gb',natom,crl=sigx_gb)
  call chmalloc('gbim.src','SetGBIM','sigy_gb',natom,crl=sigy_gb)
  call chmalloc('gbim.src','SetGBIM','sigz_gb',natom,crl=sigz_gb)
  call chmalloc('gbim.src','SetGBIM','t_gb',natom,crl=t_gb)

  call chmalloc('gbim.src','SetGBIM','OUTM_IM',natom,log=OUTM_IM)

  if (QAnalys) then
     call chmalloc('gbim.src','SetGBIM','gb_atm',natom,crl=gb_atm)
  else
     call chmalloc('gbim.src','SetGBIM','gb_atm',1,crl=gb_atm)
  endif

  ! Fill static arrays

  Call Fill_GBIM(R_GB, Vol_GB, WMain, QWeight)

  Return
End Subroutine SetGBIM

!-----------------------------------------------------------------------
Subroutine Fill_GBIM(R, V, W, QWei)
  use chm_kinds
  use number
  use consta
  use dimens_fcm
  use ffieldm
  use mmffm
  use cff_fcm
  use param
  use psf
  use stream
  implicit none

  real(chm_real) R(*), V(*), W(*)
  Integer I
  Logical Qwei

  Do I=1,Natom
     If (Qwei) then
        R(I) = W(I)
#if KEY_MMFF==1
     ElseIf (FFIELD  ==  mmff ) Then
        R(I) = RSTAR(MTYPE(I)*(MTYPE(I)+1)/2) / Two
        W(I) = R(I)
#endif 
#if KEY_CFF==1
     ElseIf (FFIELD  ==  cff ) Then
        R(I) = (NINE * CNBA(MNO(ITC(IAC(I)),ITC(IAC(I))))  &
             / (SIX * CNBB(MNO(ITC(IAC(I)),ITC(IAC(I)))) ) )**(Third) &
             / Two
        W(I) = R(I)
#endif 
     Else
        R(I) = VDWR(ITC(IAC(I)))
     Endif
     V(I) = FOUR*PI*R(I)**3/THREE
  Enddo

#if KEY_MMFF==1 || KEY_CFF==1
  If ( Prnlev  >=  2 ) then
     If ( FFIELD  ==  mmff ) Write(Outu,'(A)')  &
          ' FFIELD=MMFF active, Atom vdW radii put in wmain'
     If ( FFIELD  ==  cff ) Write(Outu,'(A)')  &
          ' FFIELD=CFF active, Atom vdW radii put in wmain'
  Endif
#endif 
  Return
End Subroutine Fill_GBIM

Function AlphijIM(r2, Ri, Rj, P, P5) result(alphijim_1)

  use chm_kinds
  use consta
  use number
  implicit none

  real(chm_real) r2, Ri, Rj, P, P5, r4, C, Rcomp,alphijim_1

  C = ONE
  If ( P5  >  ZERO ) then
     Rcomp = r2 / ( (Ri+Rj)*(Ri+Rj) )
     If ( Rcomp  <=  (ONE/P5) ) then
        C = - Cos(Rcomp*P5*PI)
        C = C + ONE
        C = HALF * C
        C = C * C
     Endif
  Endif

  r4 = r2 * r2
  AlphijIM_1 = P * C / r4

  Return
End Function AlphijIM

Function Sigij1(r, Ri, Rj, P, P5) result(sigij1_1)

  use chm_kinds
  use consta
  use number
  implicit none

  real(chm_real) :: sigij1_1
  real(chm_real) r, r2, r4, Ri, Rj, P, P5, C, Rcomp, dC

  r2 = r * r
  C = ONE
  dC = Zero
  If ( P5  >  ZERO ) then
     Rcomp = r2 / ( (Ri+Rj)*(Ri+Rj) )
     If ( Rcomp  <=  (ONE/P5) ) then
        C = - Cos(Rcomp*P5*PI)
        C = C + ONE
        dC = C*P5*Pi*Rcomp*Sin(Rcomp*P5*Pi)/r
        C = HALF * C
        C = C * C
     Endif
  Endif

  r4 = r2 * r2
  Sigij1_1 = P * dC / ( r4 * r )  &
       - Four * P * C / ( r4 * r2 )

  Return
End Function Sigij1

!-----------------------------------------------------------------------
! Eps_mol -> Eps_mol
! L_memb -> Lmemb_IM
! memb_dir -> MembDir_IM
! membrane -> memb_IM
! mb_center -> MembCenter_IM
! OUTM -> OUTM_IM
!
Subroutine GBSolvIM(GBEnr, X, Y, Z, DX, DY, DZ, &
     JNbl, INbl, Inbl14, INblo14, &
     QInte, Islct, Jslct)
  use gb_common,only:sigx_gb,sigy_gb,sigz_gb, &
     p,eps_gb,gb_atm,t_gb, &
     r_gb,vol_gb, alph_gb,qanalys
  use chm_kinds
  use number
  use consta
  use dimens_fcm
  use stream
  use psf
#if KEY_IMCUBES==1
  use inbnd, only: lbycbim
#endif
  use gbswit, only: qgbswit, louter, c2ofnb, gbswit_setup, es_switch
#if KEY_PARALLEL==1
  use parallel              
#endif
  implicit none
  Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)

  real(chm_real) GBEnr, X(*), Y(*), Z(*), DX(*), DY(*), DZ(*)
  integer, optional ::  ISlct(*), JSlct(*)
  Logical QInte

  ! Local variables
  Integer i, ic, j, k
  real(chm_real) rij, xij, yij, zij, Sqrtrij
  real(chm_real) Dij, Sij, SqrtSij, Sij3half, expDij
  real(chm_real) Aij, Aji, Bij, Bji, Sigij1Dif
  real(chm_real) Factor, Dxi, Dyi, Dzi, GBtmp     
  Logical LOK
  Integer ITemp, NPr, jpr

#if KEY_GBINLINE==1
  real(chm_real) dAij, Ratio 
#endif

  ! Function declarations
  ! real(chm_real) AlphijIM, Sigij1

  !++++++++++++++++++++++++++++++++++++++++++++++++
  real(chm_real) DD_V, DD_Vji,ZZ_V, V_alph, temp
  real(chm_real) DD_i, DD_j, fsw, dfsw

  !...........................................................

  GBEnr = Zero
  Factor = CCELEC * ( One / Eps_mol - One / Eps_gb )
  call gbswit_setup()

  ! Loop over bonded atoms first:

#if KEY_PARALLEL==1 /*parabond*/
  Do ic = MynodP, Nbond, Numnod
#else /* (parabond)*/
  Do ic = 1, Nbond
#endif /* (parabond)*/
     LOK = .true.       
     i = ib(ic)
     j = jb(ic)
     !+++++++++++++++++++++++++++++++++++++++++
     IF(OUTM_IM(i) .or. OUTM_IM(J) ) then
        !+++++++++++++++++++++++++++++++++++++++++

        If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1)
        If (LOK) Then
           ! Check if both in pair are fixed!
           If ( .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then
              xij = x(i) - x(j)
              yij = y(i) - y(j)
              zij = z(i) - z(j)
              rij = xij * xij + yij * yij + zij * zij
              Sqrtrij = Sqrt(rij)

              !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
              DD_i = Zero
              DD_j = Zero
              IF(memb_IM) then 
                 if( OUTM_IM(i) .or. OUTM_IM(j)) then            
                    V_Alph =   AlphijIM(rij, R_GB(i),R_GB(j),P(2),Zero)
                    V_Alph =   V_Alph  -  AlphijIM(rij, R_GB(i),R_GB(j),P(4),P(5))
                 endif

                 If(OUTM_IM(i)) then
                    if(MembDir_IM == 1)   &
                         DD_i = V_Alph * Diff_V(MembCenter_IM,x(i),R_GB(i),Lmemb_IM)

                    if(MembDir_IM == 2) &
                         DD_i = V_Alph * Diff_V(MembCenter_IM,y(i),R_GB(i),Lmemb_IM)

                    if(MembDir_IM == 3)  &
                         DD_i = V_Alph * Diff_V(MembCenter_IM,z(i),R_GB(i),Lmemb_IM)

                 EndIf

                 If(OUTM_IM(J)) then

                    if(MembDir_IM == 1)   &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,x(j),R_GB(j),Lmemb_IM)

                    if(MembDir_IM == 2) &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,y(j),R_GB(j),Lmemb_IM)

                    if(MembDir_IM == 3)  &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,z(j),R_GB(j),Lmemb_IM)
                 EndIf

              ENDIF
              !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

              Sigij1Dif = Sigij1(Sqrtrij,R_GB(i),R_GB(j),P(2),Zero)

              ! Subtract contributions from non-bonded terms to correct for over counting
              Sigij1Dif = Sigij1Dif -  &
                   Sigij1(Sqrtrij,R_GB(i),R_GB(j),P(4),P(5))

              If ( imove(i)  ==  0 ) then
                 If(OUTM_IM(i)) then
                    temp =  Factor * ( Cg(j) * Cg(j) + T_GB(j) )/ CCELEC
                    Aij = temp*Sigij1Dif * VOL_GB(i)               

                    DX(i) = DX(i) + xij * Aij
                    if(MembDir_IM == 1)  DX(i) = DX(i) + temp*DD_i  

                    DY(i) = DY(i) + yij * Aij
                    if(MembDir_IM == 2)  DY(i) = DY(i) + temp*DD_i   

                    DZ(i) = DZ(i) + zij * Aij
                    if(MembDir_IM == 3)  DZ(i) = DZ(i) + temp*DD_i
                 End If
              Endif

              If ( imove(j)  ==  0 ) then
                 If(OUTM_IM(j)) then
                    temp =  Factor * ( Cg(i) * Cg(i) + T_GB(i) )/ CCELEC 
                    Aji = temp*Sigij1Dif * VOL_GB(j) 

                    DX(j) = DX(j) - xij * Aji
                    if(MembDir_IM == 1)  DX(j) = DX(j) + temp*DD_j
                    DY(j) = DY(j) - yij * Aji
                    if(MembDir_IM == 2)  DY(j) = DY(j) + temp*DD_j
                    DZ(j) = DZ(j) - zij * Aji 
                    if(MembDir_IM == 3)  DZ(j) = DZ(j) + temp*DD_j

                 End If
              Endif

           Endif
        Endif
     ENDIF
  Enddo

  ! Next atoms connected through angles

#if KEY_PARALLEL==1 /*paraangle*/
  Do ic = MynodP, NTheta, NumNod
#else /* (paraangle)*/
  Do ic = 1, NTheta
#endif /* (paraangle)*/

     LOK = .true.       
     i = it(ic)
     j = kt(ic)
     !-----------------------------------------------------------------------
     !+++++++++++++++++++++++++++++++++++++++++
     IF(OUTM_IM(i) .or. OUTM_IM(J) ) then
        !+++++++++++++++++++++++++++++++++++++++++

        If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
        If (LOK) Then

           ! Check if both in pair are fixed!
           If ( .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then

              xij = x(i) - x(j)
              yij = y(i) - y(j)
              zij = z(i) - z(j)

              rij = xij * xij + yij * yij + zij * zij
              !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
              DD_i = Zero
              DD_j = Zero
              IF(memb_IM) then  
                 if( OUTM_IM(i) .or. OUTM_IM(j)) then     
                    V_Alph =   AlphijIM(rij, R_GB(i),R_GB(j),P(3),Zero)
                    V_Alph = V_Alph  -  AlphijIM(rij, R_GB(i),R_GB(j),P(4),P(5))
                 endif

                 If(OUTM_IM(i)) then
                    if(MembDir_IM == 1)   &
                         DD_i = V_Alph * Diff_V(MembCenter_IM,x(i),R_GB(i),Lmemb_IM)

                    if(MembDir_IM == 2) &
                         DD_i = V_Alph * Diff_V(MembCenter_IM,y(i),R_GB(i),Lmemb_IM)

                    if(MembDir_IM == 3)  &
                         DD_i = V_Alph * Diff_V(MembCenter_IM,z(i),R_GB(i),Lmemb_IM)

                 EndIf

                 If(OUTM_IM(J)) then

                    if(MembDir_IM == 1)   &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,x(j),R_GB(j),Lmemb_IM)

                    if(MembDir_IM == 2) &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,y(j),R_GB(j),Lmemb_IM)

                    if(MembDir_IM == 3)  &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,z(j),R_GB(j),Lmemb_IM)

                 EndIf

              ENDIF

              !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

              Sqrtrij = Sqrt(rij)

              Sigij1Dif = Sigij1(Sqrtrij,R_GB(i),R_GB(j),P(3),Zero) 
              ! Subtract contributions from non-bonded terms to correct for over counting
              Sigij1Dif = Sigij1Dif -  &
                   Sigij1(Sqrtrij,R_GB(i),R_GB(j),P(4),P(5))

              If ( imove(i)  ==  0 ) then
                 If(OUTM_IM(i)) then
                    temp =  Factor * ( Cg(j) * Cg(j) + T_GB(j) )/ CCELEC
                    Aij = temp*Sigij1Dif * VOL_GB(i) 

                    DX(i) = DX(i) + xij * Aij 
                    if(MembDir_IM == 1) DX(i) = DX(i) +  temp*DD_i 
                    DY(i) = DY(i) + yij * Aij
                    if(MembDir_IM == 2)  DY(i) = DY(i) +  temp*DD_i  
                    DZ(i) = DZ(i) + zij * Aij  
                    if(MembDir_IM == 3)  DZ(i) = DZ(i) +  temp*DD_i 
                 End If
              Endif

              If ( imove(j)  ==  0 ) then
                 If(OUTM_IM(j)) then
                    temp =  Factor * ( Cg(i) * Cg(i) + T_GB(i) )/ CCELEC
                    Aji = temp*Sigij1Dif * VOL_GB(j)                   

                    ! 1                + Factor * ( Cg(i) * Cg(i) + T_GB(i) ) 
                    ! 2                * Sigij1Dif * VOL_GB(j) / CCELEC

                    DX(j) = DX(j) - xij * Aji
                    if(MembDir_IM == 1)  DX(j) = DX(j) +  temp*DD_j 
                    DY(j) = DY(j) - yij * Aji
                    if(MembDir_IM == 2)  DY(j) = DY(j) +  temp*DD_j
                    DZ(j) = DZ(j) - zij * Aji
                    if(MembDir_IM == 3)  DZ(j) = DZ(j) +  temp*DD_j
                 End If
              Endif

           Endif
        Endif
        !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     ENDIF
     !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Enddo

  ! Now we need to get the atoms in the exclusion list
  Itemp = 0
#if KEY_PARALLEL==1 /*paraexcl*/
  ! Define the atom bounds for this processor.
  Do i = MyNodP, Natom, NumNod
#else /* (paraexcl)*/
  Do i = 1, Natom
#endif /* (paraexcl)  PARALLEL*/

     If (i  /=  1 ) ITemp = INblo14(i - 1)
     Dxi = Zero
     Dyi = Zero
     Dzi = Zero
     GBtmp = Zero

     NPr = INblo14(i) - ITemp
     Do jpr = 1, NPr
        LOK = .true.
        k = Inbl14(Itemp+jpr)
        j = Abs(k)

        !-----------------------------------------------------------------------
        If ( k  >  0 .and.  &
             .not. ( imove(i)  ==  1  &
             .and. imove(j)  ==  1 ) ) then
           If ( QInte ) LOK =  &
                ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
           If (LOK) Then

              xij = x(i) - x(j)
              yij = y(i) - y(j)
              zij = z(i) - z(j)

              rij = xij * xij + yij * yij + zij * zij

              !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
              DD_i = Zero
              DD_j = Zero
              IF(memb_IM) then 
                 if( OUTM_IM(i) .or. OUTM_IM(j)) then               
                    V_Alph =   AlphijIM(rij, R_GB(i),R_GB(j),P(4),P(5))
                 endif

                 If(OUTM_IM(i)) then
                    if(MembDir_IM == 1)   &
                         DD_i = V_Alph *  &
                         Diff_V(MembCenter_IM,x(i),R_GB(i),Lmemb_IM)

                    if(MembDir_IM == 2) &
                         DD_i = V_Alph *  &
                         Diff_V(MembCenter_IM,y(i),R_GB(i),Lmemb_IM)

                    if(MembDir_IM == 3)  &
                         DD_i = V_Alph *  &
                         Diff_V(MembCenter_IM,z(i),R_GB(i),Lmemb_IM)

                 EndIf

                 If(OUTM_IM(J)) then

                    if(MembDir_IM == 1)   &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,x(j),R_GB(j),Lmemb_IM)

                    if(MembDir_IM == 2) &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,y(j),R_GB(j),Lmemb_IM)

                    if(MembDir_IM == 3)  &
                         DD_j = V_Alph * Diff_V(MembCenter_IM,z(j),R_GB(j),Lmemb_IM)

                 EndIf

              ENDIF
              !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

              Sqrtrij = Sqrt(rij)
              Sigij1Dif = Sigij1(Sqrtrij,R_GB(i),R_GB(j),P(4),P(5))
              !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

              If(OUTM_IM(i)) then
                 temp =  Factor * ( Cg(j) * Cg(j) + T_GB(j) )/ CCELEC
                 Aij = temp* Sigij1Dif * VOL_GB(i)
                 DD_i =  temp*DD_i
              Else
                 Aij = Zero
                 DD_i = Zero
              End If

              If(OUTM_IM(j)) then
                 temp =  Factor * ( Cg(i) * Cg(i) + T_GB(i) )/ CCELEC
                 Aji = temp*Sigij1Dif * VOL_GB(j) 
                 DD_j = temp*DD_j
              Else
                 Aji = Zero
                 DD_j = Zero
              End If

              Bij = Zero
              Bji = Zero

              If ( CG(j) * CG(i)  /=  Zero ) then

                 Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                 expDij = dexp(-Dij)

                 Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                 SqrtSij = Sqrt(Sij)
                 Sij3half = Sij * SqrtSij

                 Sij = CG(i) * CG(j) * Factor / Sij3half

                 Aij = Aij + Sij - Sij * PT25 * expDij 

                 !-----------------------------------------------------------------------
                 Bij = Sij * Alph_Gb(i) *  &
                      expDij * ( Alph_Gb(i) * Alph_Gb(j)  &
                      + PT25 * rij ) / CCELEC

                 Aji = Sij - Sij * PT25 * expDij 

                 If(OUTM_IM(j)) then
                    temp =  Factor*(Cg(i) * Cg(i) + T_GB(i))/CCELEC
                    Aji =  Aji   +  temp*Sigij1Dif * VOL_GB(j) 

                 End If

                 Bji = Sij * Alph_Gb(j) *  &
                      expDij * ( Alph_Gb(j) * Alph_Gb(i)  &
                      + PT25 * rij ) / CCELEC

                 GBtmp = GBtmp - Sij * SqrtSij * SqrtSij

                 If (Qanalys) Then
                    GB_Atm(i) = GB_Atm(i) -  &
                         Half * Sij * SqrtSij * SqrtSij
                    GB_Atm(j) = GB_Atm(j) -  &
                         Half * Sij * SqrtSij * SqrtSij
                 Endif

              Endif

              If ( imove(i)  ==  0 ) then
                 Dxi = Dxi + xij * Aij + Sigx_Gb(i) * Bij
                 if(MembDir_IM == 1)  DXi = Dxi  + DD_i
                 Dyi = Dyi + yij * Aij + Sigy_Gb(i) * Bij
                 if(MembDir_IM == 2)  Dyi = Dyi  + DD_i
                 Dzi = Dzi + zij * Aij + Sigz_Gb(i) * Bij
                 if(MembDir_IM == 3)  Dzi = Dzi  + DD_i
              Endif

              If ( imove(j)  ==  0 ) then
                 DX(j) = DX(j) - xij * Aji + Sigx_Gb(j) * Bji
                 if(MembDir_IM == 1)  DX(j) = DX(j) + DD_j

                 DY(j) = DY(j) - yij * Aji + Sigy_Gb(j) * Bji
                 if(MembDir_IM == 2)  DY(j) = DY(j) + DD_j

                 DZ(j) = DZ(j) - zij * Aji + Sigz_Gb(j) * Bji
                 if(MembDir_IM == 3)  DZ(j) = DZ(j) + DD_j
              Endif

           Endif
        Endif
     Enddo

     If (NPr > 0) then
        GBEnr = GBEnr + GBtmp

        DX(i) = DX(i) + Dxi
        DY(i) = DY(i) + Dyi
        DZ(i) = DZ(i) + Dzi
     Endif

  Enddo

  ! Finally do pairs of nonbonded terms
  ! For lists use different construct here
  Itemp = 0
#if KEY_IMCUBES==1 /*imcubes*/
  if(lbycbim)then
     if( (INBL(Natom)-INBL(Natom+natom))  > 0)then
        write(outu,*) INBL(Natom),INBL(Natom-1)
        Call WrnDie(-3,'<gbim.src>GBSOLVIM', &
             'GBsolvIM (gbim.src), last atom has pairs')
     endif
  endif
#endif /* (imcubes)*/
  Do i = 1, Natom-1
#if KEY_IMCUBES==1
     if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif

     Dxi = Zero
     Dyi = Zero
     Dzi = Zero
     GBtmp = Zero

     NPr = INbl(i) - ITemp
     Do jpr = 1, NPr
        k = JNbl(Itemp+jpr)
        j = Abs(k)
        xij = x(i) - x(j)
        yij = y(i) - y(j)
        zij = z(i) - z(j)

        rij = xij * xij + yij * yij + zij * zij
        !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        DD_i = Zero
        DD_j = Zero
        IF(memb_IM) then
           if( OUTM_IM(i) .or. OUTM_IM(j)) then    
              V_Alph =   AlphijIM(rij, R_GB(i),R_GB(j),P(4),P(5))
           endif

           If(OUTM_IM(i)) then
              if(MembDir_IM == 1)   &
                   DD_i = V_Alph * Diff_V(MembCenter_IM,x(i),R_GB(i),Lmemb_IM)

              if(MembDir_IM == 2) &
                   DD_i = V_Alph * Diff_V(MembCenter_IM,y(i),R_GB(i),Lmemb_IM)

              if(MembDir_IM == 3)  &
                   DD_i = V_Alph * Diff_V(MembCenter_IM,z(i),R_GB(i),Lmemb_IM)

           EndIf

           If(OUTM_IM(J)) then

              if(MembDir_IM == 1)   &
                   DD_j = V_Alph * Diff_V(MembCenter_IM,x(j),R_GB(j),Lmemb_IM)

              if(MembDir_IM == 2) &
                   DD_j = V_Alph * Diff_V(MembCenter_IM,y(j),R_GB(j),Lmemb_IM)

              if(MembDir_IM == 3)  &
                   DD_j = V_Alph * Diff_V(MembCenter_IM,z(j),R_GB(j),Lmemb_IM)

           EndIf

        ENDIF
        !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        If (rij  <  C2OfNb) Then
           call es_switch(rij, fsw, dfsw) ! electrostatic switch func
           Sqrtrij = Sqrt(rij)
#if KEY_GBINLINE==0
           Sigij1Dif = Sigij1(Sqrtrij,R_GB(i),R_GB(j),P(4),P(5))
#else /**/
           dAij = Zero
           Aij = One
           Ratio = rij /  &
                (( R_GB(i) + R_GB(j) ) * ( R_GB(i) + R_GB(j) ))
           If ( Ratio  <=  One/P(5) ) then
              Aij = Aij - Cos( Ratio * P(5) * Pi )
              dAij = dAij + Aij &
                   * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi )  &
                   / Sqrtrij
              Aij = Half * Aij
              Aij = Aij * Aij
           Endif
           Aij = P(4) * Aij / ( rij * rij )
           Sigij1Dif = P(4) * dAij / ( rij * rij * Sqrtrij ) &
                - Four * Aij /  rij
#endif 
           If(OUTM_IM(i)) then
              temp =  Factor * ( Cg(j) * Cg(j) + T_GB(j) )/ CCELEC
              Aij = temp* Sigij1Dif * VOL_GB(i)
              DD_i =  temp*DD_i
           Else
              Aij = Zero
              DD_i = Zero
           End If
           !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           If(OUTM_IM(j)) then
              temp =  Factor * ( Cg(i) * Cg(i) + T_GB(i) )/ CCELEC
              Aji = temp*Sigij1Dif * VOL_GB(j) 
              DD_j = temp*DD_j
           Else
              Aji = Zero
              DD_j = Zero
           End If
           !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           Bij = Zero
           Bji = Zero
           Sij = Zero

           If ( CG(j) * CG(i)  /=  Zero ) then

              Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
              expDij = dexp(-Dij)

              Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
              SqrtSij = Sqrt(Sij)
              Sij3half = Sij * SqrtSij

              Sij = CG(i) * CG(j) * Factor / Sij3half * FSw

              Aij = Aij + Sij - Sij * PT25 * expDij 

              Bij = Sij * Alph_Gb(i) *  &
                   expDij * ( Alph_Gb(i) * Alph_Gb(j)  &
                   + PT25 * rij ) / CCELEC

              Aji = Sij - Sij * PT25 * expDij
              If(OUTM_IM(j)) then
                 Aji = Aji + Factor * ( Cg(i) * Cg(i) + T_GB(i) ) &
                      * Sigij1Dif * VOL_GB(j) / CCELEC
              End If
              Bji = Sij * Alph_Gb(j) *  &
                   expDij * ( Alph_Gb(j) * Alph_Gb(i)  &
                   + PT25 * rij ) / CCELEC

              Sij = Sij * SqrtSij * SqrtSij

           Endif

           If ( qgbswit .and. LOuter ) Then
              Aij = Aij  - Sij * DFSw
              Aji = Aji  - Sij * DFSw
           Endif

           GBtmp = GBtmp - Sij

           If (Qanalys) Then
              GB_Atm(i) = GB_Atm(i) - Half * Sij
              GB_Atm(j) = GB_Atm(j) - Half * Sij
           Endif

           If ( imove(i)  ==  0 ) then
              Dxi = Dxi + xij * Aij + Sigx_Gb(i) * Bij
              if(MembDir_IM == 1)  DXi = Dxi  + DD_i

              Dyi = Dyi + yij * Aij + Sigy_Gb(i) * Bij
              if(MembDir_IM == 2)  Dyi = Dyi  + DD_i

              Dzi = Dzi + zij * Aij + Sigz_Gb(i) * Bij
              if(MembDir_IM == 3)  Dzi = Dzi  + DD_i
           Endif

           If ( imove(j)  ==  0 ) then


              DX(j) = DX(j) - xij * Aji + Sigx_Gb(j) * Bji
              if(MembDir_IM == 1)  DX(j) = DX(j) + DD_j

              DY(j) = DY(j) - yij * Aji + Sigy_Gb(j) * Bji
              if(MembDir_IM == 2)  DY(j) = DY(j) + DD_j

              DZ(j) = DZ(j) - zij * Aji + Sigz_Gb(j) * Bji
              if(MembDir_IM == 3)  DZ(j) = DZ(j) +  DD_j

           Endif

        Endif
     Enddo

     If (NPr > 0) then
        GBEnr = GBEnr + GBtmp

        DX(i) = DX(i) + Dxi
        DY(i) = DY(i) + Dyi
        DZ(i) = DZ(i) + Dzi
     Endif

     ITemp = INbl(i)
  Enddo

  ! Finally add in self-terms

#if KEY_PARALLEL==1 /*paraself*/
  ! Define the atom bounds for this processor.
  Do i = MynodP, Natom, NumNod
#else /* (paraself)*/
  Do i = 1, Natom
#endif /* (paraself)  PARALLEL*/

     LOK = .true.
     If ( Qinte ) LOK = Islct(i)  ==  1
     If ( LOK ) Then
        If ( Cg(i)  /=  Zero .and. Imove(i) .ne. 1 ) then

           GBEnr = GBEnr - CG(i) * CG(i) * Factor*Half/Alph_Gb(i)

           If (Qanalys) Then
              GB_Atm(i) = GB_Atm(i)  &
                   - CG(i) * CG(i) * Factor * Half / Alph_Gb(i)
           Endif

           DX(i) = DX(i)  &
                + Factor * CG(i) * CG(i) * Sigx_Gb(i) / CCELEC
           DY(i) = DY(i)  &
                + Factor * CG(i) * CG(i) * Sigy_Gb(i) / CCELEC
           DZ(i) = DZ(i)  &
                + Factor * CG(i) * CG(i) * Sigz_Gb(i) / CCELEC

        Endif
     Endif

  Enddo

  ! Note need to GCOMB GB_Atm(i) if QAnalys true.
#if KEY_PARALLEL==1 /*paraend*/
  ! Combine GB_Atm
  If(Qanalys) Call GComb(GB_atm, Natom)
#endif /* (paraend)*/
  Return
End Subroutine GBSolvIM

!
! L_memb -> Lmemb_IM
! Gamma -> Gamma_IM
! memb_dir -> MembDir_IM
! membrane -> memb_IM
! mb_center -> MembCenter_IM
! OUTM -> OUTM_IM
!
Subroutine FillAlpGBIM( X, Y, Z, &
     JNbl, INbl, Inbl14, INblo14,  &
     QInte, Islct, JSlct)
  use gb_common,only:sigx_gb,sigy_gb,sigz_gb,cutalph,p,qanalys, &
     gb_atm,r_gb,vol_gb, alph_gb,t_gb

  use chm_kinds
  use number
  use consta
  use dimens_fcm
  use stream
  use psf
  use gbswit, only: c2ofnb, gbswit_setup, es_switch
#if KEY_IMCUBES==1
  use inbnd, only: lbycbim
#endif
#if KEY_PARALLEL==1
  use parallel         
#endif
  implicit none
  Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)
  integer, optional :: Islct(*), Jslct(*)
#if KEY_PARALLEL==1
  Integer AtFrst, AtLast
  real(chm_real),allocatable,dimension(:) :: temp
#endif 
  Real(chm_real) X(*), Y(*), Z(*)
  Logical Qinte

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! Local variables
  Integer i, ic, j, k, NFixed
  real(chm_real) rij, xij, yij, zij, Aij, Tij
  real(chm_real) Dij, Sij, SqrtSij, Sij3half, expDij
  real(chm_real) tmp
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++
  real(chm_real) abs_dist , V_i, H_i
  real(chm_real) DD_i, DD_j
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Logical LOK

  Integer ITemp, NPr, jpr
#if KEY_GBINLINE==1
  real(chm_real) dAij, Ratio 
#endif

  ! Function declarations
  ! real(chm_real) AlphijIM, Sigij1, Diff_V

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  real(chm_real) gmb_i,gmb_Sig     
  real(chm_real) g_cntr, g_slv, fsw

  If(memb_IM) then
     g_cntr = Two*Lmemb_IM*(one - one/80.)
     ! g_cntr = - q**2 ln(2)/Lmemb_IM where Lmemb_IM == half thickness

     g_cntr = -CCELEC*0.6931472/g_cntr

  Endif
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  NFixed = 0
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! Do i = 1, Natom
  !    write(outu,'(5f10.3)') X(i),Y(i),Z(i), R_GB(i),VOL_GB(i)
  ! enddo
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Do i = 1, Natom
     !.......................................................................
     ! Will be made more effective
     V_i = FOUR*PI*R_GB(I)**3/THREE
     !.......................................................................
     Alph_Gb(i) = Zero
     Sigx_Gb(i)  = Zero
     Sigy_Gb(i)  = Zero
     Sigz_Gb(i)  = Zero
     T_GB(i)     = Zero
     If ( QAnalys ) GB_Atm(i) = Zero
     If ( imove(i)  ==  1 ) NFixed = NFixed + 1
     !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     OUTM_IM(i) = .true.

     IF(memb_IM) then
        !.......OUTM_IM is recalculated: 
        if(MembDir_IM  ==  1) then
           abs_dist = abs(X(i) - MembCenter_IM)
        elseif(MembDir_IM  ==  2) then
           abs_dist = abs(Y(i) - MembCenter_IM)
        else
           abs_dist = abs(Z(i)- MembCenter_IM) 
        endif

        if((Lmemb_IM - R_GB(i)- abs_dist)  >=  Zero) then 
           OUTM_IM(i) = .false.
           V_i = Zero
        elseif((Lmemb_IM +R_GB(i) - abs_dist)  >  Zero) then
           H_i = abs_dist + R_GB(i) -  Lmemb_IM 
           V_i = PI*H_i*H_i*(R_GB(i) - H_i/THREE)
        endif
     ENDIF
     VOL_GB(i) = V_i

  Enddo

  If ( NFixed  >  0 ) Then
     Call WrnDie(-3,'<FillAlpGBIM>', &
          'GBIM not implemented w/ fixed atoms')
  Endif

  call gbswit_setup()

  ! Loop over bonded atoms first:

#if KEY_PARALLEL==1 /*parabond*/
  Do ic = MynodP, Nbond, Numnod
#else /* (parabond)*/
  Do ic = 1, Nbond
#endif /* (parabond)*/

     LOK = .true.       
     i = ib(ic)
     j = jb(ic)

     If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
     If (LOK) Then

        !+++++++++++++++++++++++++++++++++++++++++
        IF(OUTM_IM(i) .or. OUTM_IM(J) ) then
           !+++++++++++++++++++++++++++++++++++++++++
           xij = x(i) - x(j)
           yij = y(i) - y(j)
           zij = z(i) - z(j)

           rij = xij * xij + yij * yij + zij * zij

           Aij = AlphijIM(rij, R_GB(i), R_GB(j), P(2), Zero)
           ! Subtract contributions from non-bonded terms to correct for over counting
           Aij = Aij - AlphijIM(rij, R_GB(i), R_GB(j), P(4), P(5))

           If(OUTM_IM(j)) then
              Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)

           End If
           If(OUTM_IM(i)) then
              Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i) 
           End If
           rij = sqrt ( rij )

           ! Now do sigma contributions

           Aij = Sigij1(rij, R_GB(i), R_GB(j), P(2), Zero)
           ! Subtract contributions from non-bonded terms to correct for over counting
           Aij = Aij - Sigij1(rij, R_GB(i), R_GB(j), P(4), P(5))

           If ( imove(i)  ==  0 ) then
              If(OUTM_IM(j)) then
                 Sigx_Gb(i) = Sigx_Gb(i) + Aij * VOL_GB(j) * xij      
                 Sigy_Gb(i) = Sigy_Gb(i) + Aij * VOL_GB(j) * yij
                 Sigz_Gb(i) = Sigz_Gb(i) + Aij * VOL_GB(j) * zij

              End If
           Endif

           If ( imove(j)  ==  0 ) then

              If(OUTM_IM(i)) then 

                 Sigx_Gb(j) = Sigx_Gb(j) - Aij * VOL_GB(i) * xij       
                 Sigy_Gb(j) = Sigy_Gb(j) - Aij * VOL_GB(i) * yij  
                 Sigz_Gb(j) = Sigz_Gb(j) - Aij * VOL_GB(i) * zij

              End If
           Endif

        Endif
        !+++++++++++++++++++++++
     ENDIF
     !+++++++++++++++++++++++
  Enddo

  ! Next atoms connected through angles

#if KEY_PARALLEL==1 /*paraangle*/
  Do ic = MynodP, NTheta, NumNod
#else /* (paraangle)*/
  Do ic = 1, NTheta
#endif /* (paraangle)*/

     LOK = .true.       
     i = it(ic)
     j = kt(ic)

     If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
     If (LOK) Then
        !+++++++++++++++++++++++++++++++++++++++++
        IF(OUTM_IM(i) .or. OUTM_IM(J) ) then
           !+++++++++++++++++++++++++++++++++++++++++
           xij = x(i) - x(j)
           yij = y(i) - y(j)
           zij = z(i) - z(j)

           rij = xij * xij + yij * yij + zij * zij

           Aij = AlphijIM(rij, R_GB(i), R_GB(j), P(3), Zero)

           ! Subtract contributions from non-bonded terms to correct for over counting
           Aij = Aij - AlphijIM(rij, R_GB(i), R_GB(j), P(4), P(5))

           If(OUTM_IM(j)) then
              Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)

           End If

           If(OUTM_IM(i)) then
              Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i) 
           End If

           rij = sqrt ( rij )

           ! Now do sigma contributions
           Aij = Sigij1(rij, R_GB(i), R_GB(j), P(3), Zero)

           ! Subtract contributions from non-bonded terms to correct for over counting
           Aij = Aij - Sigij1(rij, R_GB(i), R_GB(j), P(4), P(5))

           If ( imove(i)  ==  0 ) then
              If(OUTM_IM(j)) then
                 Sigx_Gb(i) = Sigx_Gb(i) + Aij * VOL_GB(j) * xij
                 Sigy_Gb(i) = Sigy_Gb(i) + Aij * VOL_GB(j) * yij
                 Sigz_Gb(i) = Sigz_Gb(i) + Aij * VOL_GB(j) * zij
              End If
           Endif

           If ( imove(j)  ==  0 ) then
              If(OUTM_IM(i)) then
                 Sigx_Gb(j) = Sigx_Gb(j) - Aij * VOL_GB(i) * xij
                 Sigy_Gb(j) = Sigy_Gb(j) - Aij * VOL_GB(i) * yij
                 Sigz_Gb(j) = Sigz_Gb(j) - Aij * VOL_GB(i) * zij
              End If
           Endif

        Endif
        !+++++++++++++++++++++++++++++++++++++++++
     ENDIF
     !+++++++++++++++++++++++++++++++++++++++++
  Enddo

  ! Do atoms on exclusion list
  Itemp = 0
#if KEY_PARALLEL==1 /*paraexcl*/
  ! Define the atom bounds for this processor.
  Do i = MyNodP, Natom, NumNod
#else /* (paraexcl)*/
  Do i = 1, Natom
#endif /* (paraexcl)  PARALLEL*/

     If (i  /=  1 ) ITemp = INblo14(i - 1)
     NPr = INblo14(i) - ITemp
     Do jpr = 1, NPr

        LOK = .true.
        k = Inbl14(Itemp+jpr)
        j = Abs(k)

        ! Don't do fixed atoms from exclusion list here, get them below.
        If ( k  >  0  &
             .and. .not. (imove(i)  ==  1  &
             .and. imove(j)  ==  1) ) then
           If ( QInte ) LOK =  &
                ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
           If (LOK) Then

              !+++++++++++++++++++++++++++++++++++++++++
              IF(OUTM_IM(i) .or. OUTM_IM(J) ) then
                 !+++++++++++++++++++++++++++++++++++++++++

                 xij = x(i) - x(j)
                 yij = y(i) - y(j)
                 zij = z(i) - z(j)

                 rij = xij * xij + yij * yij + zij * zij


                 Aij =  AlphijIM(rij, R_GB(i), R_GB(j), P(4), P(5))
                 If(OUTM_IM(j)) then
                    Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)

                 End If

                 If(OUTM_IM(i)) then
                    Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i)  

                 End If
                 rij = sqrt ( rij )

                 ! Now do sigma contributions
                 Aij =  Sigij1(rij, R_GB(i), R_GB(j), P(4), P(5))

                 If ( imove(i)  ==  0 ) then
                    If(OUTM_IM(j)) then

                       Sigx_Gb(i) = Sigx_Gb(i) + Aij * VOL_GB(j) * xij

                       Sigy_Gb(i) = Sigy_Gb(i) + Aij * VOL_GB(j) * yij

                       Sigz_Gb(i) = Sigz_Gb(i) + Aij * VOL_GB(j) * zij

                    End If
                 Endif

                 If ( imove(j)  ==  0 ) then

                    If(OUTM_IM(i)) then
                       Sigx_Gb(j) = Sigx_Gb(j) - Aij * VOL_GB(i) * xij
                       Sigy_Gb(j) = Sigy_Gb(j) - Aij * VOL_GB(i) * yij
                       Sigz_Gb(j) = Sigz_Gb(j) - Aij * VOL_GB(i) * zij
                    End If
                 Endif

              Endif
           Endif
           !+++++++++++++++++++++++++++++++++++++++++
        ENDIF
        !+++++++++++++++++++++++++++++++++++++++++

     Enddo

  Enddo

  call gbswit_setup()
  ! For lists us different construct here
  Itemp = 0
  Do i = 1, Natom-1
#if KEY_IMCUBES==1
     if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif

     NPr = INbl(i) - ITemp
     Do jpr = 1, NPr
        k = JNbl(Itemp+jpr)
        j = Abs(k)

        !+++++++++++++++++++++++++++++++++++++++++
        IF(OUTM_IM(i) .or. OUTM_IM(J) ) then
           !+++++++++++++++++++++++++++++++++++++++++

           xij = x(i) - x(j)
           yij = y(i) - y(j)
           zij = z(i) - z(j)

           rij = xij * xij + yij * yij + zij * zij

           If (rij  <  c2ofnb) Then

#if KEY_GBINLINE==0
              Aij =  AlphijIM(rij, R_GB(i), R_GB(j), P(4), P(5))
#else /**/
              Aij = One
              Ratio = rij /  &
                   (( R_GB(i) + R_GB(j) ) * ( R_GB(i) + R_GB(j) ))
              If ( Ratio  <=  One/P(5) ) then
                 Aij = Aij - Cos( Ratio * P(5) * Pi )
                 Aij = Half * Aij
                 Aij = Aij * Aij
              Endif
              Aij = P(4) * Aij / ( rij * rij )
#endif 

              If(OUTM_IM(j)) then
                 Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)

              End If

              If(OUTM_IM(i)) then
                 Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i)
              End If
              rij = sqrt ( rij )

              ! Now do sigma contributions
#if KEY_GBINLINE==0
              Aij =  Sigij1(rij, R_GB(i), R_GB(j), P(4), P(5))
#else /**/
              !-----------------------------------------------------------------------
              dAij = Zero
              If ( Ratio  <=  One/P(5) ) then
                 dAij = dAij + (  One - Cos( Ratio * P(5) * Pi ) ) &
                      * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi ) / rij
              Endif
              Aij = P(4) * dAij / ( rij * rij * rij * rij * rij ) &
                   - Four * Aij / ( rij * rij )
#endif 

              If ( imove(i)  ==  0 ) then
                 If(OUTM_IM(j)) then

                    Sigx_Gb(i) = Sigx_Gb(i) + Aij * VOL_GB(j) * xij
                    Sigy_Gb(i) = Sigy_Gb(i) + Aij * VOL_GB(j) * yij
                    Sigz_Gb(i) = Sigz_Gb(i) + Aij * VOL_GB(j) * zij

                 End If
              Endif

              If ( imove(j)  ==  0 ) then
                 If(OUTM_IM(i)) then
                    Sigx_Gb(j) = Sigx_Gb(j) - Aij * VOL_GB(i) * xij
                    Sigy_Gb(j) = Sigy_Gb(j) - Aij * VOL_GB(i) * yij
                    Sigz_Gb(j) = Sigz_Gb(j) - Aij * VOL_GB(i) * zij
                 End If
              Endif

           Endif
           !+++++++++++++++++++++++++++++++++++++++++
        ENDIF
        !+++++++++++++++++++++++++++++++++++++++++
     Enddo

     ITemp = INbl(i)
  Enddo

  ! If we use lists and there are fixed atoms, add contributions from
  ! fixed atom lists to Alpha, Sigx_Gb(Y,Z).
  If ( NFixed  >  0 ) Then
     call gbswit_setup()
     Do i = 1, Natom-1
        Do j = i+1, Natom

           If ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) then
              !+++++++++++++++++++++++++++++++++++++++++
              IF(OUTM_IM(i) .or. OUTM_IM(J) ) then
                 !+++++++++++++++++++++++++++++++++++++++++
                 xij = x(i) - x(j)
                 yij = y(i) - y(j)
                 zij = z(i) - z(j)

                 rij = xij * xij + yij * yij + zij * zij

                 If (rij  <  c2ofnb) Then
#if KEY_GBINLINE==0
                    Aij =  AlphijIM(rij, R_GB(i), R_GB(j), P(4), P(5))
#else /**/
                    Aij = One
                    Ratio = rij /  &
                         (( R_GB(i) + R_GB(j) ) * ( R_GB(i) + R_GB(j) ))
                    If ( Ratio  <=  One/P(5) ) then
                       Aij = Aij - Cos( Ratio * P(5) * Pi )
                       Aij = Half * Aij
                       Aij = Aij * Aij
                    Endif
                    Aij = P(4) * Aij / ( rij * rij )
#endif 

                    If(OUTM_IM(j)) then
                       Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)
                    End If

                    If(OUTM_IM(i)) then
                       Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i)
                    End If
                 Endif
              Endif
              !+++++++++++++++++++++++++++++++++++++++++
           ENDIF
           !+++++++++++++++++++++++++++++++++++++++++
        Enddo
     Enddo

  Endif

#if KEY_PARALLEL==1 /*paralpha*/
  ! Get Alpha's updated on all processors
  Call GComb(Alph_gb, Natom)
#endif /* (paralpha)*/
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  IF(memb_IM) then
     Do i = 1, Natom
        g_slv =   CCELEC*P(1)/(Two*R_GB(i)*R_GB(i))  &
             - CCELEC/(Two*R_GB(i)*P(6)) 

        if(MembDir_IM  ==  1) then

           Call Gmbb(MembCenter_IM,X(i),Lmemb_IM,R_GB(i),g_cntr,g_slv, &
                Gamma_IM,gmb_i,gmb_Sig)
           Sigx_Gb(i) = Sigx_Gb(i) + gmb_Sig

        elseif(MembDir_IM  ==  2) then
           Call Gmbb(MembCenter_IM,Y(i),Lmemb_IM,R_GB(i),g_cntr,g_slv, &
                Gamma_IM,gmb_i,gmb_Sig) 
           Sigy_Gb(i) = Sigy_Gb(i) + gmb_Sig        

        else
           Call Gmbb(MembCenter_IM,Z(i),Lmemb_IM,R_GB(i),g_cntr,g_slv, &
                Gamma_IM,gmb_i,gmb_Sig)
           Sigz_Gb(i) = Sigz_Gb(i) + gmb_Sig        
        endif

        Alph_Gb(i) = Alph_Gb(i) + gmb_i 
        Alph_Gb(i) = - CCELEC / ( Alph_Gb(i) * Two )

        If ( Alph_Gb(i)  <=  Zero )  then
           write(outu,*) 'WARNING: Negative alph_gb( ',i,')=', Alph_Gb(i)
           Alph_Gb(i) = CutAlph
        endif
        If ( Alph_Gb(i)  >  CutAlph ) then
           write(outu,*) 'WARNING alph_gb( ',i,')=', Alph_Gb(i)
           Alph_Gb(i) = CutAlph
        endif
     ENDDO
     !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ELSE
     Do i = 1, Natom

        Alph_Gb(i) = Alph_Gb(i) + CCELEC*P(1)/(Two*R_GB(i)*R_GB(i))  &
             - CCELEC/(Two*R_GB(i)*P(6))

        Alph_Gb(i) = - CCELEC / ( Alph_Gb(i) * Two )

        If ( Alph_Gb(i)  <=  Zero )  Alph_Gb(i) = CutAlph
        If ( Alph_Gb(i)  >  CutAlph )  Alph_Gb(i) = CutAlph

     Enddo
  ENDIF

  ! ***Calculation of T***
  ! First get the atoms on exclusion lists
  Itemp = 0
#if KEY_PARALLEL==1 /*parat*/
  ! Define the atom bounds for this processor.
  Do i = MynodP, Natom, NumNod
#else /* (parat)*/
  Do i = 1, Natom
#endif /* (parat)  PARALLEL*/
     If (i  /=  1 ) ITemp = INblo14(i - 1)
     NPr = INblo14(i) - Itemp
     If ( Cg(i)  /=  Zero .and. Npr  >  0 ) then

        Tij = Zero
        Do jpr = 1, NPr
           LOK = .true.
           k = Inbl14(Itemp+jpr)
           j = Abs(k)
           !-----------------------------------------------------------------------
           If ( ( Cg(j)  /=  Zero .and. k  >  0 ) .and. &
                .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then

              If ( QInte ) LOK =  &
                   ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
              If (LOK) Then
                 xij = x(i) - x(j)
                 yij = y(i) - y(j)
                 zij = z(i) - z(j)

                 rij = xij * xij + yij * yij + zij * zij
                 Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                 expDij = dexp(-Dij)

                 Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                 SqrtSij = Sqrt(Sij)
                 Sij3half = Sij * SqrtSij

                 Sij = CG(i) * CG(j) / Sij3half

                 If ( imove(i)  ==  0 ) Tij = Tij  &
                      + Sij * Alph_Gb(i) *  &
                      expDij* ( Alph_Gb(i) * Alph_Gb(j) &
                      + PT25 * rij )

                 If (imove(j)  ==  0 ) T_GB(j) = T_GB(j)  &
                      + Sij * Alph_Gb(j) *  &
                      expDij* ( Alph_Gb(j) * Alph_Gb(i) &
                      + PT25 * rij )

              Endif
           Endif
        Enddo

        T_GB(i) = T_GB(i) + Tij
     Endif
  Enddo
  ! Now non-bonded contributions
  Itemp = 0
#if KEY_IMCUBES==1
  Do i = 1, Natom-1
     if(lbycbim) ITEMP=INBL(I+NATOM)
#else /**/
  Do i = 1, Natom-1
#endif 
     NPr = INbl(i) - ITemp
     If ( Cg(i)  /=  Zero .and. Npr  >  0 ) then

        Tij = Zero
        Do jpr = 1, NPr
           k = JNbl(Itemp+jpr)
           j = Abs(k)

           If ( Cg(j)  /=  Zero ) then
              xij = x(i) - x(j)
              yij = y(i) - y(j)
              zij = z(i) - z(j)

              rij = xij * xij + yij * yij + zij * zij
              If (rij  <  c2ofnb) Then
                 call es_switch(rij, fsw) ! electrostatic switch func
                 Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                 expDij = dexp(-Dij)

                 Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                 SqrtSij = Sqrt(Sij)
                 Sij3half = Sij * SqrtSij

                 Sij = CG(i) * CG(j) / Sij3half * FSw

                 If ( imove(i)  ==  0 ) Tij = Tij  &
                      + Sij * Alph_Gb(i) *  &
                      expDij* ( Alph_Gb(i) * Alph_Gb(j) &
                      + PT25 * rij )

                 If ( imove(j)  ==  0 ) T_GB(j) = T_GB(j)  &
                      + Sij * Alph_Gb(j) *  &
                      expDij* ( Alph_Gb(j) * Alph_Gb(i) &
                      + PT25 * rij )

              Endif
           Endif
        Enddo
        T_GB(i) = T_GB(i) + Tij
     Endif
     ITemp = INbl(i)
  Enddo

#if KEY_PARALLEL==1 /*paraend*/
  allocate(temp(4*natom))
  Do i = 1, Natom
     Temp(i) = T_GB(i)
     Temp(Natom + i) = Sigx_Gb(i)
     Temp(2*Natom + i) = Sigy_Gb(i)
     Temp(3*Natom + i) = Sigz_Gb(i)
  Enddo
  ! Combine T and Sigmas
  Call GComb(Temp, 4*Natom)
  Do i = 1, Natom
     T_GB(i) = Temp(i)
     Sigx_Gb(i) = Temp(Natom + i)
     Sigy_Gb(i) = Temp(2*Natom + i)
     Sigz_Gb(i) = Temp(3*Natom + i)
  Enddo
  deallocate(temp)
#endif /* (paraend)*/

  Return
End Subroutine FillAlpGBIM

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Subroutine Gmbb(mb_center,XYZ,L_memb,Rvdw,g_cntr, &
     g_solv,Gamma,gmb_i,gmb_Sig)
  !......... XYZ - position along X or Y or Z
  use chm_kinds
  use number  
  use stream  
  implicit none

  real(chm_real) Z,XYZ,L_memb,Rvdw,g_cntr,g_solv, &
       Gamma,gmb_i,gmb_Sig
  real(chm_real) D_gmb,L0,zz,denom,exp_z, mb_center

  Z = XYZ - mb_center 
  D_gmb = (g_cntr - g_solv)
  L0 = L_memb - Rvdw
  if(L0  <=  0 )  CALL WRNDIE(-5,'<Gmbb>', &
       'Rvdw > Lmembrane')
  zz = Gamma*(abs(Z) - L0)
  exp_z = dexp(zz)
  denom =  One/(One + exp_z)              
  gmb_i = g_solv + D_gmb*denom

  If(Z  >  ZERO) then
     gmb_Sig = -Gamma*D_gmb*exp_z*denom*denom               

  elseif(Z  <  ZERO) then
     gmb_Sig = Gamma*D_gmb*exp_z*denom*denom               

  else
     gmb_Sig = Zero
  EndIf
  Return
End Subroutine Gmbb

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Function Diff_V(mb_center,XYZ,R,L) result(diff_v_1)
  !..... firs derivative from the volume
  !...      V = pi*(H**2)*(R-H/3), where H = R + abs(z) - L      
  use chm_kinds
  use number
  use consta
  implicit none
  real(chm_real) Diff_V_1
  real(chm_real) Z, ZZ, R, L, H, R2, mb_center,XYZ
  R2 = TWO*R
  Diff_V_1 = Zero
  Z = XYZ -  mb_center
  H = abs(Z)+R-L

  IF(H <= R2) THEN
     if(Z >= Zero) then
        Diff_V_1 = PI*H*(R2 - H)
     else
        Diff_V_1 = PI*H*(H - R2)
     EndIf
  ENDIF
  Return
End Function Diff_V
end module gbim
