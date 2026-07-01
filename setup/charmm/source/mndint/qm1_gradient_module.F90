module qm1_gradient_module
  use chm_kinds
  use number
  use qm1_constant

  contains

#if KEY_MNDO97==1 /*mndo97*/

  subroutine qmqm_gradient(PA,PB,dim_linear_norbs)
  ! 
  ! Fast gradients in cartesian coordinates by finite difference.
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! PA(dim_linear_norbs)   rhf or uhf-alpha density matrix (I).
  ! PB(dim_linear_norbs)   uhf-beta density matrix (I).
  !
  ! DHCORE    computes the two-center integrals required for 
  !           the gradient calculation. (extenal subroutine)
  ! 
  ! Note: in the routine, subroutine psort and fock2d are included below
  !       as contained subroutines.
  !
  use qm1_info, only : qm_control_r,qm_main_r,qm_scf_main_r,mm_main_r
  use qm1_energy_module, only : dr_width_beta,r_dr_width_beta
  !use number
  !use qm1_constant
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  !
  integer :: dim_linear_norbs
  real(chm_real):: PA(dim_linear_norbs),PB(dim_linear_norbs)

  ! local variables
  integer,parameter :: LMP_local=171,LMW_local=2025
  integer :: I,J,IJ,IA,IB,JA,JB,K,M
  integer :: IJ_atom_pair,NORBS,LIN
  real(chm_real):: XSTORE,EH_diff,DER,enuclr(2),EE_diff,EF_diff
  integer :: NA(2),NF(2),NL(2)
  real(chm_real):: h_1(LMP_local),h_2(LMP_local),PA2(LMP_local),PB2(LMP_local), &
                   PAB2_sum(LMP_local),w_1(LMW_local),w_2(LMW_local),X(3,2),    &
                   dgrad_i(3)
  integer :: nnumnod,mmynod,icnt,iicnt

#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
  icnt    = 0
#endif
      
  if(mm_main_r%q_lookup_beta) r_dr_width_beta = one/dr_width_beta
  iicnt = 0

  ! initialization:
  !IJ_atom_pair = qm_scf_main_r%INDX(qm_main_r%numat)+qm_main_r%numat
  !qm_main_r%qm_grads(1:3,1:qm_main_r%numat) = zero
  !
  ! loop over atom pairs ij:
  !     atoms i and j are identified at the beginning of the loop.
  !     atom i gets number 2 in the pair i-j since i.gt.j.
  !     atom j gets number 1 in the pair i-j since i.gt.j.
  loopII: do i=2,qm_main_r%numat
     ia          = qm_main_r%nfirst(i)
     ib          = qm_main_r%nlast(i)
     na(2)       = qm_main_r%nat(i)
     x(1:3,2)    = qm_main_r%qm_coord(1:3,i)
     dgrad_i(1:3)= zero

     loopJJ: do j=1,i-1
        if(na(2) > 1 .or. qm_main_r%nat(j) > 1) iicnt = iicnt + 1
#if KEY_PARALLEL==1
        icnt     = icnt + 1
        if(mmynod .ne. mod(icnt-1,nnumnod)) cycle loopJJ
#endif
        ja       = qm_main_r%nfirst(j)
        jb       = qm_main_r%nlast(j)
        na(1)    = qm_main_r%nat(j)
        x(1:3,1) = qm_main_r%qm_coord(1:3,j)

        NF(1)  = 1
        NL(1)  = JB-JA+1  ! norbs of atom j
        NF(2)  = NL(1)+1  ! 
        NL(2)  = NF(2)+IB-IA
        norbs  = NL(2)
        lin    = qm_scf_main_r%INDX(norbs)+norbs

        ! from density matrix for atom pair i-j
        call psort(JA,JB,IA,IB,PA2,PB2,PAB2_sum,PA,PB,dim_linear_norbs,LMP_local, &
                   qm_scf_main_r%indx)

        ! loop over x,y,z-cart coords by finite difference calculation 
        ! by varying the coords of atom i.
        loopKK: do k=1,3
           xstore = x(k,2)

           ! for the positive direction
           x(k,2) = xstore+qm_control_r%del
           ! compute two-center integrals
           call dhcore(h_1,w_1,lin,LMW_local,i,j,na,nf,nl,x,enuclr(1),iicnt)

           ! for the negative direction
           x(k,2) = xstore-qm_control_r%del
           call dhcore(h_2,w_2,lin,LMW_local,i,j,na,nf,nl,x,enuclr(2),iicnt)

           ! two-center one-electron energy; already take the difference as 1-2.
           EH_diff = zero
           do m=1,lin
              EH_diff = EH_diff + (h_1(m)-h_2(m))*pab2_sum(m)   ! =(pa2(m)+pb2(m))
           end do

           ! two-center two-electron energy: call the routine once, it will return 
           ! the difference of energy components for the negative and positive dirs.
           EF_diff = zero
           call fock2d(EF_diff,h_1,h_2,pa2,pb2,pab2_sum,w_1,w_2,lin,LMW_local,nf,nl,qm_scf_main_r%INDX)
           if(qm_main_r%uhf) then
              call fock2d(EF_diff,h_1,h_2,pb2,pa2,pab2_sum,w_1,w_2,lin,LMW_local,nf,nl,qm_scf_main_r%INDX)
              EF_diff=PT5*EF_diff
           end if

           x(k,2) = xstore
           EE_diff= EH_diff+EF_diff+(enuclr(1)-enuclr(2))
           DER    = EE_diff*EVCAL*qm_control_r%rdel

           qm_main_r%qm_grads(k,j) = qm_main_r%qm_grads(k,j)-DER
           dgrad_i(k)              = dgrad_i(k)+DER  ! for atom i
        end do loopKK
     end do loopJJ

     ! add grad. componet to atom i.
     qm_main_r%qm_grads(1:3,i)=qm_main_r%qm_grads(1:3,i)+dgrad_i(1:3)
  end do loopII

  return

  contains
     ! contains two subroutines: psort & fock2d

     subroutine psort(JA,JB,IA,IB,PA2,PB2,Pab2_sum,PA,PB,LM4,LMP,indx)
     !
     ! extract density matrix for atom pair i-j, in which offdiagonal
     ! elements are multiplied by 2.
     !
     !use number, only : two

     implicit none
     !
     integer :: JA,JB,IA,IB,LM4,LMP,indx(*)
     real(chm_real):: PA2(LMP),PB2(LMP),Pab2_sum(LMP),PA(LM4),PB(LM4)
     ! local variables
     integer :: IJ,K,L,KK,KA

     ! one-center elements from atom j.
     ij     = 0
     do k=ja,jb
        kk     = indx(k)+ja-1
        do l=ja,k-1
           ij     = ij+1
           kk     = kk+1
           pa2(ij)= two*pa(kk)
           pb2(ij)= two*pb(kk)
           pab2_sum(ij)=pa2(ij)+pb2(ij)
        end do
        ij = ij+1 ! for l=k, diagonal term.
        kk = kk+1
        pa2(ij)= pa(kk)
        pb2(ij)= pb(kk)
        pab2_sum(ij)=pa2(ij)+pb2(ij)
     end do
     ! loop over remaining elements.
     do k=ia,ib
        ka     = indx(k)  ! k*(k-1)/2
        ! two-center elements.
        kk     = ka+ja-1
        do l=ja,jb
           ij     = ij+1
           kk     = kk+1
           pa2(ij)= two*pa(kk)
           pb2(ij)= two*pb(kk)
           pab2_sum(ij)=pa2(ij)+pb2(ij)
        end do
        ! one-center elements for atom i.
        kk     = ka+ia-1
        do l=ia,k-1
           ij     = ij+1
           kk     = kk+1
           pa2(ij)= two*pa(kk)
           pb2(ij)= two*pb(kk)
           pab2_sum(ij)=pa2(ij)+pb2(ij)
        end do
        ij = ij+1 ! for l=k, diagonal term
        kk = kk+1
        pa2(ij)= pa(kk)
        pb2(ij)= pb(kk)
        pab2_sum(ij)=pa2(ij)+pb2(ij)
     end do
     return
     end subroutine psort
     !===================================================================


     subroutine fock2d(EF_diff,F1,F2,PA,PB,PAB_sum,W1,W2,LM4,LM9,NFIRST,NLAST,indx)
     !
     ! Two-center two-electron contributions to the Fock matrix: A special version 
     ! for computing the two-electron contribtion of a given atom pair to the 
     ! gradient. (for MNDO-type orbital)
     !
     ! On return, EF_diff is already the difference between two sets (negative
     !                    and positive displacements).
     !
     ! NOTATION. I=INPUT, O=OUTPUT, S=SCRATCH.
     ! EF_diff   difference of energy contribution as
     !           EF_negative - EF_positive displacement (O).
     ! F1(LM4)   Fock matrix contribution from negative displacement (S).
     ! F2(LM4)                                 positive                 .
     ! PA(LM4)   rhf or uhf-alpha density matrix (I).
     ! PB(LM4)   uhf-beta density matrix (I).
     ! W1(LM9)   two-electron integrals from negative displacement (I).
     ! W2(LM9)                               positive
     ! nfirst()  index of first orbital at a given atom (I): nfirst(1)=1
     !                                                       nfirst(2)=nlast(1)+1
     ! nlast()            last                          (I). nlast (1)=norbs(j)
     !                                                   nlast (2)=norbs(j)+norbs(i)
     !

     !use qm1_info, only : qm_scf_main_r
     !use number
     !use qm1_constant

     implicit none
     !
     integer :: LM4,LM9,NFIRST(2),NLAST(2),indx(*)
     real(chm_real):: EF_diff,F1(LM4),F2(LM4),PA(LM4),PB(LM4),PAB_sum(LM4), &
                      W1(LM9),W2(LM9)

     ! local variables
     integer :: I,J,K,L,IA,IB,ID,IJ,IK,IL,JA,JB,JK,JL,KA,KB,KC,KD,KK,KL,ID1,ID2
     real(chm_real):: PSS,paa(3),PA_local(14),PB_local(14), &
                      F15(2),fi1,fi2,fj1,fj2,pij,A(2)
     logical :: IEQJ,KEQL
     real(chm_real):: pp(32)
     integer, parameter :: IWW(4,4)=reshape((/1,2,4,7,2,3,5,8,4,5,6,9,7,8,9,10/),(/4,4/)), &
                           IXH(10) =(/3,5,6,8,9,10,12,13,14,15/), &
                           IXX(10) =(/15,20,21,26,27,28,33,34,35,36/)

     ! initialization
     IA     = nfirst(2) ! nlast(1)+1 (starting of atom i)
     IB     = nlast(2)  ! total number of orbtials of i & j atoms
     JB     = nlast(1)  ! norbs of atom j

     if(ib.eq.2) then               ! H-H pair
        paa(1)= pab_sum(3)   ! =PA(3)+PB(3) 
        paa(2)=-PA(2)*PT5
        paa(3)= pab_sum(1)   ! =PA(1)+PB(1)

        ! negative direction
        f1(1:3)= w1(1)*paa(1:3)
        ! positive direction
        f2(1:3)= w2(1)*paa(1:3)

        EF_diff=EF_diff+ (f1(1)-f2(1))*PA(1)+ (f1(2)-f2(2))*PA(2)+ (f1(3)-f2(3))*PA(3)
     else
        f1(1:LM4)   = zero
        f2(1:LM4)   = zero
        if(ib.eq.5 .and. jb.eq.1) then  ! heavy atom-H pair
           pss = pab_sum(1)
           do i=1,10
              f1(1) = f1(1)+w1(i)*pab_sum(IXH(i)) ! = (PA(IXH(i))+PB(IXH(i)))
              f2(1) = f2(1)+w2(i)*pab_sum(IXH(i))
              !
              f1(IXH(i))=f1(IXH(i))+w1(i)*pss
              f2(IXH(i))=f2(IXH(i))+w2(i)*pss
           end do
           PA_local(1)  =-PA(2)*PT5
           PA_local(2)  =-PA(4)*PT5
           PA_local(3)  =-PA(7)*PT5
           PA_local(4)  =-PA(11)*PT5
           ! 
           !f1(IXH(1:10))= f1(IXH(1:10))+w1(1:10)*pss
           f1(2) = w1(1)*PA_local(1)+w1(2)*PA_local(2)+w1(4)*PA_local(3)+w1( 7)*PA_local(4)
           f1(4) = w1(2)*PA_local(1)+w1(3)*PA_local(2)+w1(5)*PA_local(3)+w1( 8)*PA_local(4)
           f1(7) = w1(4)*PA_local(1)+w1(5)*PA_local(2)+w1(6)*PA_local(3)+w1( 9)*PA_local(4)
           f1(11)= w1(7)*PA_local(1)+w1(8)*PA_local(2)+w1(9)*PA_local(3)+w1(10)*PA_local(4)
           !
           !f2(IXH(1:10))= f2(IXH(1:10))+w2(1:10)*pss
           f2(2) = w2(1)*PA_local(1)+w2(2)*PA_local(2)+w2(4)*PA_local(3)+w2( 7)*PA_local(4)
           f2(4) = w2(2)*PA_local(1)+w2(3)*PA_local(2)+w2(5)*PA_local(3)+w2( 8)*PA_local(4)
           f2(7) = w2(4)*PA_local(1)+w2(5)*PA_local(2)+w2(6)*PA_local(3)+w2( 9)*PA_local(4)
           f2(11)= w2(7)*PA_local(1)+w2(8)*PA_local(2)+w2(9)*PA_local(3)+w2(10)*PA_local(4)
        else if(ib.eq.5 .and. jb.eq.4) then ! H-heavy atom pair
           f15(1:2) = zero
           pss      = pab_sum(15)  
           do i=1,10
              f15(1)= f15(1)+w1(i)*pab_sum(i)  ! =(PA(I)+PB(I))
              f15(2)= f15(2)+w2(i)*pab_sum(i)

              f1(i) = f1(i) +w1(i)*pss
              f2(i) = f2(i) +w2(i)*pss
           end do
           PA_local(1:4) = -PA(11:14)*PT5
           !
           !f1(15) = f15(1)
           f1(11) = w1(1)*PA_local(1)+w1(2)*PA_local(2)+w1(4)*PA_local(3)+w1( 7)*PA_local(4)
           f1(12) = w1(2)*PA_local(1)+w1(3)*PA_local(2)+w1(5)*PA_local(3)+w1( 8)*PA_local(4)
           f1(13) = w1(4)*PA_local(1)+w1(5)*PA_local(2)+w1(6)*PA_local(3)+w1( 9)*PA_local(4)
           f1(14) = w1(7)*PA_local(1)+w1(8)*PA_local(2)+w1(9)*PA_local(3)+w1(10)*PA_local(4)
           f1(15) = f15(1)
           !
           !f2(15) = f15(2)
           f2(11) = w2(1)*PA_local(1)+w2(2)*PA_local(2)+w2(4)*PA_local(3)+w2( 7)*PA_local(4)
           f2(12) = w2(2)*PA_local(1)+w2(3)*PA_local(2)+w2(5)*PA_local(3)+w2( 8)*PA_local(4)
           f2(13) = w2(4)*PA_local(1)+w2(5)*PA_local(2)+w2(6)*PA_local(3)+w2( 9)*PA_local(4)
           f2(14) = w2(7)*PA_local(1)+w2(8)*PA_local(2)+w2(9)*PA_local(3)+w2(10)*PA_local(4)
           f2(15) = f15(2)
        else if(ib.eq.8 .and. jb.eq.4) then  ! heavy atom-heavy atom pair.
           pp(1:10)  = pab_sum(IXX(1:10))  ! =PA(IXX(1:10))+PB(IXX(1:10))
           do i=1,10
              id1    = 10*(i-1)
              id2    = i-10
              fi1    = zero
              fi2    = zero
              fj1    = zero
              fj2    = zero
              do j=1,10
                 fi1 = fi1+w1(id2+10*j)*pp(j)
                 fi2 = fi2+w1(id1+j)*pab_sum(j)
                 ! for positive dir.
                 fj1 = fj1+w2(id2+10*j)*pp(j)
                 fj2 = fj2+w2(id1+j)*pab_sum(j)
              end do
              f1(i)     = fi1
              f1(IXX(i))= fi2

              f2(i)     = fj1
              f2(IXX(i))= fj2
           end do
           pp(11:32) = -PA(11:32)*PT5
           do i=1,4
              id     = INDX(i+4)
              do k=1,4
                 kd     = INDX(k+4)
                 ik     = 10*(IWW(i,k)-1)
                 f1(id+1)= f1(id+1)+pp(kd+1)*w1(ik+1)+pp(kd+2)*w1(ik+2)+pp(kd+3)*w1(ik+4)+pp(kd+4)*w1(ik+7)
                 f1(id+2)= f1(id+2)+pp(kd+1)*w1(ik+2)+pp(kd+2)*w1(ik+3)+pp(kd+3)*w1(ik+5)+pp(kd+4)*w1(ik+8)
                 f1(id+3)= f1(id+3)+pp(kd+1)*w1(ik+4)+pp(kd+2)*w1(ik+5)+pp(kd+3)*w1(ik+6)+pp(kd+4)*w1(ik+9)
                 f1(id+4)= f1(id+4)+pp(kd+1)*w1(ik+7)+pp(kd+2)*w1(ik+8)+pp(kd+3)*w1(ik+9)+pp(kd+4)*w1(ik+10)
                 !
                 f2(id+1)= f2(id+1)+pp(kd+1)*w2(ik+1)+pp(kd+2)*w2(ik+2)+pp(kd+3)*w2(ik+4)+pp(kd+4)*w2(ik+7)
                 f2(id+2)= f2(id+2)+pp(kd+1)*w2(ik+2)+pp(kd+2)*w2(ik+3)+pp(kd+3)*w2(ik+5)+pp(kd+4)*w2(ik+8)
                 f2(id+3)= f2(id+3)+pp(kd+1)*w2(ik+4)+pp(kd+2)*w2(ik+5)+pp(kd+3)*w2(ik+6)+pp(kd+4)*w2(ik+9)
                 f2(id+4)= f2(id+4)+pp(kd+1)*w2(ik+7)+pp(kd+2)*w2(ik+8)+pp(kd+3)*w2(ik+9)+pp(kd+4)*w2(ik+10) 
              end do
           end do
        else                               ! if it has d-orbitals.
           ! general code - also valid for d-orbitals.
           kk     = 0
           do i=ia,ib
              ka     = INDX(I)
              do j=ia,i
                 kb     = INDX(J)
                 ij     = ka+j
                 ieqj   = i.eq.j
                 pij    = pab_sum(ij)  ! =PA(IJ)+PB(IJ)
                 do k=1,jb
                    kc     = INDX(k)
                    ik     = ka+k
                    jk     = kb+k
                    do l=1,k
                       il     = ka+l
                       jl     = kb+l
                       kl     = kc+l
                       keql   = k.eq.l
                       kk     = kk+1
                       a(1)   = w1(kk)
                       a(2)   = w2(kk)
                       f1(ij) = f1(ij) + a(1)*pab_sum(kl)  ! (PA(KL)+PB(KL))
                       f1(kl) = f1(kl) + a(1)*pij
                       !
                       f2(ij) = f2(ij) + a(2)*pab_sum(kl)  ! (PA(KL)+PB(KL))
                       f2(kl) = f2(kl) + a(2)*pij

                       a(1:2) = a(1:2)*PT5
                       if(ieqj.and.keql) then
                          f1(ik) = f1(ik) - a(1)*PA(jl)
                          !
                          f2(ik) = f2(ik) - a(2)*PA(jl)
                       else if(ieqj) then
                          f1(ik) = f1(ik) - a(1)*PA(jl)
                          f1(il) = f1(il) - a(1)*PA(jk)
                          ! 
                          f2(ik) = f2(ik) - a(2)*PA(jl)
                          f2(il) = f2(il) - a(2)*PA(jk)
                       else if(keql) then
                          f1(ik) = f1(ik) - a(1)*PA(jl)
                          f1(jk) = f1(jk) - a(1)*PA(il)
                          !
                          f2(ik) = f2(ik) - a(2)*PA(jl)
                          f2(jk) = f2(jk) - a(2)*PA(il)
                       else
                          f1(ik) = f1(ik) - a(1)*PA(jl)
                          f1(jk) = f1(jk) - a(1)*PA(il)
                          f1(il) = f1(il) - a(1)*PA(jk)
                          f1(jl) = f1(jl) - a(1)*PA(ik)
                          !
                          f2(ik) = f2(ik) - a(2)*PA(jl)
                          f2(jk) = f2(jk) - a(2)*PA(il)
                          f2(il) = f2(il) - a(2)*PA(jk)
                          f2(jl) = f2(jl) - a(2)*PA(ik)    
                       end if
                    end do
                 end do
              end do
           end do
        end if
        ! compute EF_diff (difference of the energy component 
        ! for the negative and positive displacement).
        do i=1,LM4
          EF_diff = EF_diff + (f1(i)-f2(i))*PA(i)
        end do
     end if  ! (ib.eq.2)
     return
     end subroutine fock2d
  ! end of contains.
  !=====================================================================

  end subroutine qmqm_gradient


  subroutine qmmm_gradient(PA,PB,NATOM,XI,YI,ZI,dim_linear_norbs, &
                           numat,numatm,nat,nfirst,nlast,num_orbs,indx, &
                           qm_coord,mm_coord,mm_chrgs,                  &
                           CORE_mat,WW,RI,YY,                           &
                           qm_grads,mm_grads,q_am1_pm3,istart,iend)
  !
  ! Fast gradients in cartesian coords by finite difference based on
  ! the method by M.J. Field et al, J.Comp.Chem. 11, 700 (1990).
  !
  !use chm_kinds
  use qm1_info, only : qm_control_r,mm_main_r ! ,qm_main_r,qm_scf_main_r
  use nbndqm_mod, only: map_qmatom_to_group,map_mmatom_to_group
  use qm1_energy_module, only : rotmat_qmmm,repp_qmmm,rotcora_qmmm,repam1_qmmm, &
                                i_index_look_up,r_core_val,coef_core_val,core_val_shift, &
                                core_cut_val,dr_width,r_dr_width,rval_min,rval_max
  use qm1_mndod, only : reppd_qmmm
  use qm1_parameters, only : CORE,DELTA,OMEGA
  !use number
  !use qm1_constant
#if KEY_PARALLEL==1
  use parallel 
#endif

  implicit none
  !
  integer       :: natom,dim_linear_norbs,numat,numatm, &
                   nat(*),nfirst(*),nlast(*),num_orbs(*),indx(*)
  real(chm_real):: PA(dim_linear_norbs),PB(dim_linear_norbs)
  real(chm_real):: XI(natom), YI(natom), ZI(natom)
  real(chm_real):: qm_coord(3,numat),mm_coord(3,natom),mm_chrgs(natom), &
                   qm_grads(3,numat),mm_grads(3,natom),                 &
                   CORE_mat(10,2),WW(2025),RI(22),YY(675)
  logical       :: q_am1_pm3

  ! local variables
  integer :: I,J,K,L,M,N,NI,IA,IB,IJ,IW,KK,LL,jj,irs_qm,irs_mm,irs_qm2
  integer :: jmax,iorbs,LIN,i_do_switching,i_do_switching_2
  integer,parameter :: LMH=45
  real(chm_real):: DERQ,DERQE,PTCHG,EH_diff,SCALE,ENUC(2),Etot_diff,Etot_ref,Enuc_ref
  real(chm_real):: H_1(LMH),H_2(LMH),Q(LMH)
  real(chm_real):: dgrad(3,natom),CG_I(3),CG_m(3),dxyz_sw(3)
  real(chm_real):: ETOT(3),XCOORD(3,2),PTCHG_SIGN,r_1,r_2,rr,delr,r_sq
  real(chm_real),parameter :: a0_to_kcal=EVCAL*EV*A0
  logical :: q_do_sw_pair
  !
  integer :: istart,iend
  real(chm_real):: ddot_mn ! external function
!#if KEY_PARALLEL==1
!  integer :: ISTRT_CHECK            ! external function
!#endif

!  istart = 1
!  iend   = numatm
!#if KEY_PARALLEL==1
!  if(numnod>1) istart = ISTRT_CHECK(iend,numatm)
!#endif

  ! initialization
  if(mm_main_r%q_switch) mm_main_r%dxyz_sw(1:6,1:mm_main_r%isize_swt_array) =zero

  if(mm_main_r%q_lookup) then
     r_dr_width = one/dr_width
     !
     ! loop over qm atoms (i).
     loopI2: do i=1,NUMAT
        ij = i_index_look_up(i)  ! matching atom id.
        ni     = NAT(i)
        ia     = NFIRST(i)
        ib     = NLAST(i)
        iorbs  = num_orbs(i)
        LIN    = INDX(iorbs)+iorbs
        if(iorbs.ge.9) then
           iw=LIN  ! = INDX(iorbs)+iorbs
           Jmax=10
        else
           Jmax=iorbs
        end if
        XCOORD(1:3,2) = qm_coord(1:3,i)
        ! extract relevant one-center density matrix elements.
        do k=1,iorbs
           kk = INDX(ia-1+k)+ia-1
           j  = INDX(k)
           q(j+1:j+k-1)=two*(PA(kk+1:kk+k-1)+PB(kk+1:kk+k-1))
           q(j+k)      =PA(kk+k)+PB(kk+k)
        end do

        ! local gradient componet for qm atom i.
        CG_i(1:3) = zero

        !
        if(mm_main_r%q_cut_by_group .or. mm_main_r%q_switch) irs_qm = map_qmatom_to_group(i)

        ! loop over mm atoms.
        loopM2: do m=istart,iend           ! 1,numatm
           PTCHG       = mm_chrgs(m)
           if(PTCHG.ge.zero) then
              PTCHG_SIGN = one
           else
              PTCHG_SIGN =-one
           end if
           XCOORD(1:3,1) = mm_coord(1:3,m)
           CG_m(1:3)     = zero  ! local gradient for mm atom.

           ! group-by-group-based cutoff case, skip the pair if its distance is longer than cutoff.
           ! otherwise (default group-based case), include all mm atoms (default).
           q_do_sw_pair =.false.
           if(mm_main_r%q_cut_by_group) then
              irs_mm = map_mmatom_to_group(m)
              if(.not.mm_main_r%q_mmgrp_qmgrp_cut(irs_mm,irs_qm)) cycle
           end if
           !
           if(mm_main_r%q_switch) then
              irs_mm         = map_mmatom_to_group(m)
              i_do_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
              if(i_do_switching > 0) q_do_sw_pair =.true.
           end if

           if(q_do_sw_pair) then
              ! energy for the unperturbed case.

              ! distance R (au) and rotation matrix
              call rotmat_qmmm(1,2,1,iorbs,2,XCOORD,r_1,YY)

              ! local charge-electron attaraction integrals.
              call repp_qmmm(ni,0,i,r_1,RI,CORE_mat)
              if(iorbs.ge.9) call reppd_qmmm(ni,0,r_1,RI,CORE_mat,WW,iw,1)  ! iw=LIN see above

              ! multiplication by point charge
              CORE_mat(1:Jmax,1) = CORE_mat(1:Jmax,1)*PTCHG

              ! contributions to the CORE_mat hamiltonian
              H_1(1:LIN)   = ZERO
              call rotcora_qmmm(1,0,iorbs,0,1,0,CORE_mat,YY,H_1,LMH,mm_main_r%q_diag_coulomb)
              !Etot_ref = ddot_mn(LIN,h_1(1:LIN),1,q(1:LIN),1)  ! DOT_PRODUCT(h_1(1:LIN),q(1:LIN))
              Etot_ref = zero
              do k=1,LIN
                 Etot_ref = Etot_ref + h_1(i)*q(i)
              end do

              ! contributions to the core-core-repulsions
              if(.not.mm_main_r%q_diag_coulomb) then
                 rr = r_1*A0
                 if(rr >= rval_min .and. rr <= rval_max) then
                    jj=  int((rr - rval_min)*r_dr_width) + 1
                    delr= rr - r_core_val(jj)
                    scale=((coef_core_val(1,jj,1,ij)*delr +coef_core_val(2,jj,1,ij))*delr + &
                            coef_core_val(3,jj,1,ij))*delr+coef_core_val(4,jj,1,ij) + &
                            core_val_shift(1,ij)
                    Enuc_ref = CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                    if(q_am1_pm3 .and. rr <= core_cut_val(ij)) then
                       Enuc_ref = Enuc_ref + ((coef_core_val(1,jj,2,ij)*delr +coef_core_val(2,jj,2,ij))*delr + &
                                               coef_core_val(3,jj,2,ij))*delr+coef_core_val(4,jj,2,ij) + &
                                               core_val_shift(2,ij)
                    end if
                 else
                    scale= EXP(-OMEGA(ni)*(rr-DELTA(ni)))+EXP(-five*rr)
                    Enuc_ref= CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                    if(q_am1_pm3) call repam1_qmmm(i,ni,0,rr,Enuc_ref)
                 end if
                 Etot_ref = (Etot_ref + PTCHG*Enuc_ref)*EVCAL
              else
                 Etot_ref = Etot_ref*EVCAL
              end if

              ! scale charge by switching function.
              PTCHG = PTCHG*mm_main_r%sw_val(i_do_switching)
              dxyz_sw(1:3) = Etot_ref*mm_main_r%dsw_val(1:3,i_do_switching)
              if(mm_main_r%q_cut_by_group) then
                 ! for qm atoms.
                 mm_main_r%dxyz_sw(1:3,i_do_switching) = mm_main_r%dxyz_sw(1:3,i_do_switching)  &
                                                        +dxyz_sw(1:3)*mm_main_r%r_num_atom_qm_grp(irs_qm)
                 ! for mm atoms.
                 mm_main_r%dxyz_sw(4:6,i_do_switching) = mm_main_r%dxyz_sw(4:6,i_do_switching)  &
                                                        -dxyz_sw(1:3)*mm_main_r%r_num_atom_mm_grp(irs_mm)
              else
                 ! for qm atoms.
                 irs_qm2          = mm_main_r%q_mmgrp_point_swt(irs_mm)
                 i_do_switching_2 = mm_main_r%q_backmap_dxyz_sw(i_do_switching)
                 mm_main_r%dxyz_sw(1:3,i_do_switching_2) = mm_main_r%dxyz_sw(1:3,i_do_switching_2)  &
                                                          +dxyz_sw(1:3)*mm_main_r%r_num_atom_qm_grp(irs_qm2)
                 ! for mm atoms.
                 mm_main_r%dxyz_sw(4:6,i_do_switching_2) = mm_main_r%dxyz_sw(4:6,i_do_switching_2)  &
                                                          -dxyz_sw(1:3)*mm_main_r%r_num_atom_mm_grp(irs_mm)
              end if
           end if

           ! loop over three cartesian coordinates (k) of qm atom i
           do k=1,3

              !(1) : for the positive displacement.
              XCOORD(k,2) = qm_coord(k,i)+qm_control_r%DEL

              ! distance R (au) and rotation matrix
              call rotmat_qmmm(1,2,1,iorbs,2,XCOORD,r_1,YY)

              ! local charge-electron attaraction integrals.
              call repp_qmmm(ni,0,i,r_1,RI,CORE_mat)
              if(iorbs.ge.9) call reppd_qmmm(ni,0,r_1,RI,CORE_mat,WW,iw,1)  ! iw=LIN see above

              ! multiplication by point charge
              CORE_mat(1:Jmax,1) = CORE_mat(1:Jmax,1)*PTCHG

              ! contributions to the CORE_mat hamiltonian
              H_1(1:LIN)   = ZERO
              call rotcora_qmmm(1,0,iorbs,0,1,0,CORE_mat,YY,H_1,LMH,mm_main_r%q_diag_coulomb)
              ! EH = ddot_mn(LIN,H_1(1:LIN),1,Q(1:LIN),1) ! DOT_PRODUCT(H_1(1:LIN),Q(1:LIN))

              ! contributions to the core-core-repulsions
              if(.not.mm_main_r%q_diag_coulomb) then   
                 rr = r_1*A0
                 if(rr >= rval_min .and. rr <= rval_max) then
                    jj=  int((rr - rval_min)*r_dr_width) + 1
                    delr= rr - r_core_val(jj)
                    scale=((coef_core_val(1,jj,1,ij)*delr +coef_core_val(2,jj,1,ij))*delr + &
                            coef_core_val(3,jj,1,ij))*delr+coef_core_val(4,jj,1,ij) + &
                            core_val_shift(1,ij)
                    enuc(1) = CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                    if(q_am1_pm3 .and. rr <= core_cut_val(ij)) then
                       enuc(1) = enuc(1) + ((coef_core_val(1,jj,2,ij)*delr +coef_core_val(2,jj,2,ij))*delr + &
                                            coef_core_val(3,jj,2,ij))*delr+coef_core_val(4,jj,2,ij) + &
                                            core_val_shift(2,ij)
                    end if
                 else
                    scale= EXP(-OMEGA(ni)*(rr-DELTA(ni)))+EXP(-five*rr)
                    ENUC(1)= CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                    if(q_am1_pm3) call repam1_qmmm(i,ni,0,rr,ENUC(1))
                 end if
              else
                 ENUC(1)= zero
              end if

              !(2) : for the negative displacement.
              !      below is a mere repeat of above sequence of subroutine calls.
              XCOORD(k,2) = qm_coord(k,i)-qm_control_r%DEL
              call rotmat_qmmm(1,2,1,iorbs,2,XCOORD,r_2,YY)
              call repp_qmmm(ni,0,i,r_2,RI,CORE_mat)
              if(iorbs.ge.9) call reppd_qmmm(ni,0,r_2,RI,CORE_mat,WW,iw,1)  ! iw=LIN see above
              CORE_mat(1:Jmax,1) = CORE_mat(1:Jmax,1)*PTCHG
              H_2(1:LIN)   = ZERO
              call rotcora_qmmm(1,0,iorbs,0,1,0,CORE_mat,YY,H_2,LMH,mm_main_r%q_diag_coulomb)
              !EH = ddot_mn(LIN,H_2(1:LIN),1,Q(1:LIN),1)  ! DOT_PRODUCT(H_2(1:LIN),Q(1:LIN))
              if(.not.mm_main_r%q_diag_coulomb) then
                 rr = r_2*A0
                 if(rr >= rval_min .and. rr <= rval_max) then
                    jj=  int((rr - rval_min)*r_dr_width) + 1
                    delr= rr - r_core_val(jj)
                    scale=((coef_core_val(1,jj,1,ij)*delr +coef_core_val(2,jj,1,ij))*delr + &
                            coef_core_val(3,jj,1,ij))*delr+coef_core_val(4,jj,1,ij) + &
                            core_val_shift(1,ij)
                    enuc(2) = CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                    if(q_am1_pm3 .and. rr <= core_cut_val(ij)) then
                       enuc(2) = enuc(2) + ((coef_core_val(1,jj,2,ij)*delr +coef_core_val(2,jj,2,ij))*delr + &
                                            coef_core_val(3,jj,2,ij))*delr+coef_core_val(4,jj,2,ij) + &
                                            core_val_shift(2,ij)
                    end if
                 else
                    scale= EXP(-OMEGA(ni)*(rr-DELTA(ni)))+EXP(-five*rr)
                    ENUC(2)= CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                    if(q_am1_pm3) call repam1_qmmm(i,ni,0,rr,ENUC(2))
                 end if
              else
                 ENUC(2)= zero
              end if

              ! recover XCOORD
              XCOORD(k,2) = qm_coord(k,i)

              ! now, take the difference of energy Etot(1)-Etot(2)
              ! EH_diff = EH(1)-EH(2)
              EH_diff=zero
              do j=1,LIN
                 EH_diff=EH_diff+(h_1(j)-h_2(j))*q(j)
              end do
              Etot_diff= EH_diff+PTCHG*(ENUC(1)-ENUC(2))
              derQ     = Etot_diff*EVCAL*qm_control_r%RDEL
              !
              CG_i(k)  = CG_i(k) + derQ
              CG_m(k)  = CG_m(k) - derQ
           end do
           mm_grads(1:3,m)=mm_grads(1:3,m)+CG_m(1:3)
        end do loopM2    ! M=1,NUMATM

        ! for QM atom
        qm_grads(1:3,i) = qm_grads(1:3,i)+CG_i(1:3)
     end do loopI2       ! I=1,NUMAT
  else
     !
     ! loop over qm atoms (i).
     loopII: do i=1,NUMAT
        ni     = NAT(i)
        ia     = NFIRST(i)
        ib     = NLAST(i)
        iorbs  = num_orbs(i)
        LIN    = INDX(iorbs)+iorbs
        if(iorbs.ge.9) then
           iw=LIN  ! = INDX(iorbs)+iorbs
           Jmax=10
        else
           Jmax=iorbs
        end if
        XCOORD(1:3,2) = qm_coord(1:3,i)
        ! extract relevant one-center density matrix elements.
        do k=1,iorbs
           kk = INDX(ia-1+k)+ia-1
           j  = INDX(k)
           q(j+1:j+k-1)=two*(PA(kk+1:kk+k-1)+PB(kk+1:kk+k-1))
           q(j+k)      =PA(kk+k)+PB(kk+k)
        end do

        ! local gradient componet for qm atom i.
        CG_i(1:3) = zero

        !
        if(mm_main_r%q_cut_by_group .or. mm_main_r%q_switch) irs_qm = map_qmatom_to_group(i)

        ! loop over mm atoms.
        loopMM: do m=istart,iend           ! 1,numatm
           PTCHG       = mm_chrgs(m)
           if(PTCHG.ge.zero) then
              PTCHG_SIGN = one
           else
              PTCHG_SIGN =-one
           end if
           XCOORD(1:3,1) = mm_coord(1:3,m)
           CG_m(1:3)     = zero  ! local gradient for mm atom.

           ! group-by-group-based cutoff case, skip the pair if its distance is longer than cutoff.
           ! otherwise (default group-based case), include all mm atoms (default).
           q_do_sw_pair =.false.
           if(mm_main_r%q_cut_by_group) then
              irs_mm = map_mmatom_to_group(m)
              if(.not.mm_main_r%q_mmgrp_qmgrp_cut(irs_mm,irs_qm)) cycle
           end if
           !
           if(mm_main_r%q_switch) then
              irs_mm         = map_mmatom_to_group(m)
              i_do_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
              if(i_do_switching > 0) q_do_sw_pair =.true.
           end if

           if(q_do_sw_pair) then
              ! energy for the unperturbed case.

              ! distance R (au) and rotation matrix
              call rotmat_qmmm(1,2,1,iorbs,2,XCOORD,r_1,YY)

              ! local charge-electron attaraction integrals.
              call repp_qmmm(ni,0,i,r_1,RI,CORE_mat)
              if(iorbs.ge.9) call reppd_qmmm(ni,0,r_1,RI,CORE_mat,WW,iw,1)  ! iw=LIN see above

              ! multiplication by point charge
              CORE_mat(1:Jmax,1) = CORE_mat(1:Jmax,1)*PTCHG

              ! contributions to the CORE_mat hamiltonian
              H_1(1:LIN)   = ZERO
              call rotcora_qmmm(1,0,iorbs,0,1,0,CORE_mat,YY,H_1,LMH,mm_main_r%q_diag_coulomb)
              !Etot_ref = ddot_mn(LIN,h_1(1:LIN),1,q(1:LIN),1) ! DOT_PRODUCT(h_1(1:LIN),q(1:LIN))
              Etot_ref = zero
              do k=1,LIN
                 Etot_ref = Etot_ref + h_1(k)*q(k)
              end do

              ! contributions to the core-core-repulsions
              if(.not.mm_main_r%q_diag_coulomb) then
                 rr = r_1*A0
                 scale= EXP(-OMEGA(ni)*(rr-DELTA(ni)))+EXP(-five*rr)
                 Enuc_ref= CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                 if(q_am1_pm3) call repam1_qmmm(i,ni,0,rr,Enuc_ref)

                 Etot_ref = (Etot_ref + PTCHG*Enuc_ref)*EVCAL
              else
                 Etot_ref = Etot_ref*EVCAL
              end if

              ! scale charge by swithing function.
              PTCHG = PTCHG*mm_main_r%sw_val(i_do_switching)
              dxyz_sw(1:3) = Etot_ref*mm_main_r%dsw_val(1:3,i_do_switching)
              if(mm_main_r%q_cut_by_group) then
                 ! for qm atoms.
                 mm_main_r%dxyz_sw(1:3,i_do_switching) = mm_main_r%dxyz_sw(1:3,i_do_switching)  &
                                                        +dxyz_sw(1:3)*mm_main_r%r_num_atom_qm_grp(irs_qm)
                 ! for mm atoms.
                 mm_main_r%dxyz_sw(4:6,i_do_switching) = mm_main_r%dxyz_sw(4:6,i_do_switching)  &
                                                        -dxyz_sw(1:3)*mm_main_r%r_num_atom_mm_grp(irs_mm)
              else
                 irs_qm2 = mm_main_r%q_mmgrp_point_swt(irs_mm)
                 i_do_switching_2 = mm_main_r%q_backmap_dxyz_sw(i_do_switching)
                 ! for qm atoms.
                 mm_main_r%dxyz_sw(1:3,i_do_switching_2) = mm_main_r%dxyz_sw(1:3,i_do_switching_2)  &
                                                        +dxyz_sw(1:3)*mm_main_r%r_num_atom_qm_grp(irs_qm2)
                 ! for mm atoms.
                 mm_main_r%dxyz_sw(4:6,i_do_switching_2) = mm_main_r%dxyz_sw(4:6,i_do_switching_2)  &
                                                        -dxyz_sw(1:3)*mm_main_r%r_num_atom_mm_grp(irs_mm)
              end if
           end if

           ! loop over three cartesian coordinates (k) of qm atom i
           do k=1,3

              !(1) : for the positive displacement.
              XCOORD(k,2) = qm_coord(k,i)+qm_control_r%DEL

              ! distance R (au) and rotation matrix
              call rotmat_qmmm(1,2,1,iorbs,2,XCOORD,r_1,YY)

              ! local charge-electron attaraction integrals.
              call repp_qmmm(ni,0,i,r_1,RI,CORE_mat)
              if(iorbs.ge.9) call reppd_qmmm(ni,0,r_1,RI,CORE_mat,WW,iw,1)  ! iw=LIN see above
             
              ! multiplication by point charge
              CORE_mat(1:Jmax,1) = CORE_mat(1:Jmax,1)*PTCHG

              ! contributions to the CORE_mat hamiltonian
              H_1(1:LIN)   = ZERO
              call rotcora_qmmm(1,0,iorbs,0,1,0,CORE_mat,YY,H_1,LMH,mm_main_r%q_diag_coulomb)
              ! EH = ddot_mn(LIN,H_1(1:LIN),1,Q(1:LIN),1)  ! DOT_PRODUCT(H_1(1:LIN),Q(1:LIN))

              ! contributions to the core-core-repulsions
              if(.not.mm_main_r%q_diag_coulomb) then
                 scale= EXP(-OMEGA(ni)*(r_1*A0-DELTA(ni)))+EXP(-five*r_1*A0)
                 ENUC(1)= CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                 if(q_am1_pm3) call repam1_qmmm(i,ni,0,r_1*A0,ENUC(1))
              else
                 ENUC(1)= zero
              end if


              !(2) : for the negative displacement.
              !      below is a mere repeat of above sequence of subroutine calls.
              XCOORD(k,2) = qm_coord(k,i)-qm_control_r%DEL
              call rotmat_qmmm(1,2,1,iorbs,2,XCOORD,r_2,YY)
              call repp_qmmm(ni,0,i,r_2,RI,CORE_mat)
              if(iorbs.ge.9) call reppd_qmmm(ni,0,r_2,RI,CORE_mat,WW,iw,1)  ! iw=LIN see above
              CORE_mat(1:Jmax,1) = CORE_mat(1:Jmax,1)*PTCHG
              H_2(1:LIN)   = ZERO
              call rotcora_qmmm(1,0,iorbs,0,1,0,CORE_mat,YY,H_2,LMH,mm_main_r%q_diag_coulomb)
              !EH = ddot_mn(LIN,H_2(1:LIN),1,Q(1:LIN),1)  ! DOT_PRODUCT(H_2(1:LIN),Q(1:LIN))
              if(.not.mm_main_r%q_diag_coulomb) then
                 scale= EXP(-OMEGA(ni)*(r_2*A0-DELTA(ni)))+EXP(-five*r_2*A0)
                 ENUC(2)= CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
                 if(q_am1_pm3) call repam1_qmmm(i,ni,0,r_2*A0,ENUC(2))
              else
                 ENUC(2)= zero
              end if

              ! recover XCOORD
              XCOORD(k,2) = qm_coord(k,i)

              ! now, take the difference of energy Etot(1)-Etot(2)
              ! EH_diff = EH(1)-EH(2)
              EH_diff=zero
              do j=1,LIN
                 EH_diff=EH_diff+(h_1(j)-h_2(j))*q(j)
              end do
              Etot_diff= EH_diff+PTCHG*(ENUC(1)-ENUC(2))
              derQ     = Etot_diff*EVCAL*qm_control_r%RDEL
              !
              CG_i(k)  = CG_i(k) + derQ
              CG_m(k)  = CG_m(k) - derQ
           end do
           mm_grads(1:3,m)=mm_grads(1:3,m)+CG_m(1:3)
        end do loopMM    ! M=1,NUMATM

        ! for QM atom
        qm_grads(1:3,i) = qm_grads(1:3,i)+CG_i(1:3)
     end do loopII       ! I=1,NUMAT
  end if
  !
  return
  end subroutine qmmm_gradient


  subroutine dhcore(H,W,LM4,LM9,I,J,nat,nfirst,nlast,COORD,ENUCLR,iicnt)
  !
  ! partial core hamiltonian for gradient calcualtions. 
  ! general version for atom pair i-j.
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! H(LM4)    core hamiltonian matrix for atom pair i-j (O).
  ! W(LM9)    two-electron integrals for atom pair i-j (O).
  ! I,J       atom numbers for pair i-j in master list (I).
  ! nat(2)    atomic number of atoms i and j (I).
  ! nfirst(2) number of first orbital at atoms i and j (I).
  ! nlast(2)  number of last  orbital at atoms i and j (I).
  ! coord()   cartesian coordinates for atoms i and j (I).
  ! ENUCLR    core-core repulsion energy for pair i-j (O).
  !
  !use chm_kinds
  !use number
  !use qm1_constant
  use qm1_info, only : qm_control_r,qm_scf_main_r,mm_main_r
  use qm1_energy_module, only : hhpair,rotmat,repp_qmqm, &
                                rotate,betaij,rotbet,rotcora_qmqm,core_repul, &
                                dr_width_beta,r_dr_width_beta,rval_min_beta,rval_max_beta, &
                                ij_pair_look_up,look_up_beta_r,r_beta_val
  use qm1_mndod, only : reppd_qmqm,rotd

  implicit none
  !
  integer :: LM4,LM9,I,J,iicnt,ii,jj,k
  integer :: nat(2),nfirst(2),nlast(2)
  real(chm_real):: ENUCLR,H(LM4),W(LM9),coord(3,2)

  ! local variables
  integer :: KR,NI,NJ,IA,IB,IP,IW,JA,JB,JP,JW,IORBS,JORBS,ipnt
  real(chm_real):: HIJ,WIJ,R,delr


  ! check for special case of H-H pair.
  if(nat(1).eq.1 .and. nat(2).eq.1) then
     call hhpair(j,i,1,2,coord,Hij,Wij,Enuclr)
     H(1) =-wij
     H(2) = Hij
     H(3) =-Wij
     W(1) = Wij        ! always iop.le.0
  else
     ! initialize some variables.
     kr     = 0
     ni     = nat(2)    ! atom i
     ia     = nfirst(2)
     ib     = nlast(2)

     nj     = nat(1)    ! atom j
     ja     = nfirst(1)
     jb     = nlast(1)

     iorbs  = ib-ia+1
     jorbs  = jb-ja+1
     H(1:LM4)=zero

     ! rotation matrix and interatomic distance R(au).
     call rotmat(1,2,jorbs,iorbs,2,coord,R,qm_scf_main_r%YY)

     ! two-electron integrals:
     iw     = (iorbs*(iorbs+1))/2
     jw     = (jorbs*(jorbs+1))/2
     ! two-electron integrals in local coordinate: compute & store the
     ! semiemprical integrals.
     call repp_qmqm(i,j,ni,nj,R,qm_scf_main_r%RI,qm_scf_main_r%CORE_mat)
     if(iorbs.ge.9 .or. jorbs.ge.9)  &
        call reppd_qmqm(ni,nj,R,qm_scf_main_r%RI,qm_scf_main_r%CORE_mat,W,iw,jw)

     ! transform two-electron integrals to molecular coordinates.
     ip     = (ia*(ia+1))/2
     jp     = (ja*(ja+1))/2
     if(iorbs.le.4 .and. jorbs.le.4) then
        call rotate(iw,jw,ip,jp,kr,qm_scf_main_r%RI,qm_scf_main_r%YY, &
                    W,W,iw+jw,LM9,1)
     else
        call rotd(W,qm_scf_main_r%YY,iw,jw)
     end if

     ! resonance integrals.
     if(mm_main_r%q_lookup_beta) then
        if(r >= rval_min_beta .and. r <= rval_max_beta) then
           ipnt = ij_pair_look_up(iicnt)  ! the array counter.
           select case(iorbs+jorbs)
              case (2)
                 ii=1
              case (5)
                 ii=3
              case (8)
                 ii=5
              case (10)
                 ii=7
              case (13)
                 ii=11
              case (18)
                 ii=15
           end select
           jj  = int((r - rval_min_beta)*r_dr_width_beta) + 1
           delr= r - r_beta_val(jj)
           do k=1,ii
              qm_scf_main_r%T(k)=((look_up_beta_r(ipnt)%coef_val(1,k,jj) *delr + &
                                   look_up_beta_r(ipnt)%coef_val(2,k,jj))*delr + &
                                   look_up_beta_r(ipnt)%coef_val(3,k,jj))*delr + &
                                   look_up_beta_r(ipnt)%coef_val(4,k,jj)       + &
                                   look_up_beta_r(ipnt)%val_shift(k)
           end do
        else
           call betaij(i,j,ni,nj,iorbs,jorbs,R,qm_scf_main_r%T)
        end if
     else
        call betaij(i,j,ni,nj,iorbs,jorbs,R,qm_scf_main_r%T)
     end if
     call rotbet(ia,ja,iorbs,jorbs,qm_scf_main_r%T,qm_scf_main_r%YY,H,LM4, &
                 qm_scf_main_r%indx)

     ! core-electron attractions.
     call rotcora_qmqm(ia,ja,iorbs,jorbs,ip,jp,qm_scf_main_r%CORE_mat, &
                       qm_scf_main_r%YY,H,LM4)

     ! core-core repulsions.
     Wij = qm_scf_main_r%RI(1)
     call core_repul(i,j,ni,nj,R,Wij,ENUCLR)
  end if

  return
  end subroutine dhcore

#endif /*mndo97*/
end module qm1_gradient_module
