#if KEY_SQUANTM==1 || KEY_QTURBO==1 /*mainsquatn*/
      SUBROUTINE qm_ewald_setup(kmaxqx,kmaxqy,kmaxqz,ksqmaxq,totkq, &
                                Qcheck)
!
! Initial setup for qm_ewald, - should be called only once per run.
! This routine calculate the total K space vectors and other related
! values. These values should be constant during a QM/MM run, unless
! re-do the setup.
!
  use qmmm_module, only : qmmm_ewald

  use chm_kinds
  use qm2_double

      implicit none

! Passed in
      integer, intent(in)    :: kmaxqx,kmaxqy,kmaxqz,ksqmaxq
      integer, intent(inout) :: totkq
      logical, intent(inout) :: Qcheck

! Local variables
      integer                :: kx,ky,kz,ksy,ksz,ksq

! local preparation
      totkq  = 0

! Calculate the total number of K space vectors
      Do kx = 0, kmaxqx
        if (kx .EQ. 0) then
            ksy = 0
        else
            ksy = -kmaxqy
        end if
        do ky = ksy, kmaxqy
           if (kx .EQ. 0 .and. ky .EQ. 0) then
              ksz = 1
           else
              ksz = -kmaxqz
           end if
           do kz = ksz, kmaxqz
              ksq = kx*kx + ky*ky + kz*kz
              if (ksq .LE. ksqmaxq .and. ksq .NE. 0) totkq = totkq + 1
           end do
        end do
      End do

!
      If (totkq .LE. 0) then
         Write(6,*)'qm_ewald_setup> ',  &
                   'Invalie number of Total Kspace vectors.'
         Write(6,*)'TOTKQ needs to be greater than 0, but now is ',totkq

         Qcheck=.FALSE.
         return
      End if

      return
      END SUBROUTINE qm_ewald_setup


      SUBROUTINE qmmm_ewald_allocate(Nquant,Qcheck)
!
! Allocate required memories for Kvector and Ktables
!
  use qmmm_module, only : qmmm_ewald

  use chm_kinds
  use qm2_double

      implicit none

! Passed in
      integer, intent(in)    :: Nquant
      logical, intent(inout) :: Qcheck

! Local variables
      integer                :: array_size
      integer                :: ier=0

! Deallocate if allocated
      If(associated(qmmm_ewald%scf_mchg))     &
                   Deallocate(qmmm_ewald%scf_mchg)
      If(associated(qmmm_ewald%scf_mchg_2)) &
                   Deallocate(qmmm_ewald%scf_mchg_2)
      If(associated(qmmm_ewald%Kvec))    &
                   Deallocate(qmmm_ewald%Kvec)
      If(associated(qmmm_ewald%Ktable))    &
                   Deallocate(qmmm_ewald%Ktable)
      If(associated(qmmm_ewald%qmktable))    &
                   Deallocate(qmmm_ewald%qmktable)
      If(associated(qmmm_ewald%structfac_mm)) &
                   Deallocate(qmmm_ewald%structfac_mm)
      If(associated(qmmm_ewald%empot))     &
                   Deallocate(qmmm_ewald%empot)
      If(associated(qmmm_ewald%eslf))    &
                   Deallocate(qmmm_ewald%eslf)
      If(associated(qmmm_ewald%d_ewald_mm))    &
                   Deallocate(qmmm_ewald%d_ewald_mm)
      If(associated(qmmm_ewald%qmqmerfcx_data))    &
                   Deallocate(qmmm_ewald%qmqmerfcx_data)
      If(associated(qmmm_ewald%exl_xyz))    &
                   Deallocate(qmmm_ewald%exl_xyz)
      If(associated(qmmm_ewald%dexl_xyz))    &
                   Deallocate(qmmm_ewald%dexl_xyz)
      If(associated(qmmm_ewald%mm_chg_fix_qm)) &
                   Deallocate(qmmm_ewald%mm_chg_fix_qm)

! Now allocate
      Allocate(qmmm_ewald%scf_mchg(Nquant), stat=ier)
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','scf_mchg')

      Allocate(qmmm_ewald%scf_mchg_2(Nquant), stat=ier)
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','scf_mchg_2')

      Allocate(qmmm_ewald%Kvec(qmmm_ewald%totkq), stat=ier)
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','Kvec')

      if(qmmm_ewald%QNoPMEwald) then
         Allocate(qmmm_ewald%Ktable(6,qmmm_ewald%iatotl,qmmm_ewald%totkq),   &
                 stat=ier)                             ! iatotl=natom
      else
         Allocate(qmmm_ewald%Ktable(6,1,1),stat=ier)    ! for dummy allocation
      end if
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','Ktable')

      Allocate(qmmm_ewald%qmktable(6,Nquant,qmmm_ewald%totkq), stat=ier)
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','qmktable')

      if(qmmm_ewald%QNoPMEwald) then
         Allocate(qmmm_ewald%structfac_mm(2,qmmm_ewald%totkq),stat=ier)
      else
         Allocate(qmmm_ewald%structfac_mm(2,1),stat=ier)
      end if
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','structfac_mm')

      Allocate(qmmm_ewald%empot(Nquant), stat=ier)
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','empot')

      Allocate(qmmm_ewald%eslf(Nquant,Nquant), stat=ier)
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','eslf')

      Allocate(qmmm_ewald%d_ewald_mm(3,qmmm_ewald%natom), stat=ier) 
      if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','d_ewald_mm')

! for exclude MM atoms from QM-MM interactions (IGMSEL(i)=5)
      If (qmmm_ewald%nexl_atm .gt. 0) then
        Allocate(qmmm_ewald%exl_xyz(4,qmmm_ewald%nexl_atm), stat=ier)
        if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','exl_xyz')

        Allocate(qmmm_ewald%dexl_xyz(3,qmmm_ewald%nexl_atm), stat=ier)
        if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','dexl_xyz')
      End if

! for pKa FEP-QM/MM
       If (qmmm_ewald%qpka_calc_on) then
        Allocate(qmmm_ewald%mm_chg_fix_qm(qmmm_ewald%natqm_fix), &
                 stat=ier)
        if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','mm_chg_fix_qm')
       End if

      If (qmmm_ewald%erfcx_incore) then
        array_size = ishft(Nquant*Nquant,-1)
        Allocate(qmmm_ewald%qmqmerfcx_data(array_size), stat=ier)
        if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','qmqmerfcx_data')
      else
        Allocate(qmmm_ewald%qmqmerfcx_data(1), stat=ier)
        if(ier.ne.0) call Aass(1,'qmmm_ewald_allocate','qmqmerfcx_data')
      End if

      return 
      END SUBROUTINE qmmm_ewald_allocate


      SUBROUTINE qm_ewald_calc_kvec(kappa,volume,recip,kvec,totkq,   &
                                 ksqmaxq,kmaxqx,kmaxqy,kmaxqz, &
                                 Qcheck)
!
! Setup wave-vector arrays for K-space Ewald summation.
! 

  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      real(chm_real),  intent(in)    :: kappa,volume,recip(6)
      real(chm_real),  intent(out)   :: kvec(*)
      integer, intent(in)    :: totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz
      logical, intent(inout) :: Qcheck

! Local variables
      integer                :: loop_count, kx, ky, kz, ksq, ksy, ksz
      real(chm_real)                 :: beta, vfact
      real(chm_real)                 :: rkx(3), rky(3), rkz(3), rksq
      real(chm_real)                 :: mkv(3)

      beta = PI2 / (kappa*kappa)
      vfact= INVPI/volume

      loop_count = 0

      Do kx = 0, kmaxqx
        if (kx .eq. 0) then
          ksy = 0
        else
          ksy = -kmaxqy
        end if
        rkx(1) = kx * recip(1)
        rkx(2) = kx * recip(2)
        rkx(3) = kx * recip(4)
        do ky = ksy, kmaxqy
          if (kx .eq. 0 .and. ky .eq. 0) then
            ksz = 1
          else
            ksz = -kmaxqz
          end if
          rky(1) = ky * recip(2)
          rky(2) = ky * recip(3)
          rky(3) = ky * recip(5)
          do kz = ksz, kmaxqz
            rkz(1) = kz * recip(4)
            rkz(2) = kz * recip(5)
            rkz(3) = kz * recip(6)
            ksq    = kx*kx + ky*ky + kz*kz
            if (ksq .le. ksqmaxq .and. ksq .ne. 0) then
              loop_count = loop_count + 1

              mkv(1:3)         = rkx(1:3) + rky(1:3) + rkz(1:3)
              rksq             = mkv(1)**2 + mkv(2)**2 + mkv(3)**2
              kvec(loop_count) = vfact*exp(-beta*rksq)/rksq

            end if
          end do
        end do
      End do

! Check if loop_count ends up equalling totkq. If not, then 
! something wrong in the code.
      If (loop_count .ne. totkq) then
         Write(6,*)'Qm_ewald_calc_kvec> Invalid number of K vectors.'
         Write(6,*)'It should be ',totkq,' but now it is ',loop_count

         Qcheck =.FALSE.
         return
      End if
!

      return
      END SUBROUTINE qm_ewald_calc_kvec


      SUBROUTINE qm_ewald_calc_ktable(Natom,Nquant,Iqmatoms, &
                                   iastrt,iafinl,iatotl,itotal,itotkq, &
                                   totkq,   &
                                   ksqmaxq,kmaxqx,kmaxqy,kmaxqz, &
                                   X,Y,Z,Recip,Ktable,qmKtable,   &
                                   QNoPMEwald,Qcheck)
!
! Setup K-table array for K-space Ewald summation.
!

  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      integer, intent(in)    :: Natom,Nquant
      integer, intent(in)    :: Iqmatoms(Nquant),iastrt,iafinl,iatotl,itotal,itotkq
      integer, intent(in)    :: totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz
      real(chm_real),  intent(in)  :: X(*),Y(*),Z(*),recip(6)
      real(chm_real),  intent(out) :: Ktable(6,itotal,itotkq)      ! Ktable(6,iatotl,totkq); iatotl=natom
      real(chm_real),  intent(out) :: qmKtable(6,nquant,totkq)
      logical, intent(inout) :: QNoPMEwald,Qcheck
 
! Local variables
      integer                :: loop_count, kx, ky, kz, ksq, ksy, ksz
      integer                :: i, qmid, inner_loop
      real(chm_real)               :: rkx(3), rky(3), rkz(3), mkv(3)
      real(chm_real)               :: picoef
      real(chm_real)               :: xyz(3)


      loop_count = 0

      Do kx = 0, kmaxqx
        if (kx .eq. 0) then 
           ksy = 0 
        else
           ksy = -kmaxqy
        end if
        picoef = TWO_PI  * REAL(kx)
        rkx(1) = picoef * recip(1)
        rkx(2) = picoef * recip(2)
        rkx(3) = picoef * recip(4)
 
        do ky = ksy, kmaxqy
           if (kx .eq. 0 .and. ky .eq. 0) then
              ksz = 1
           else
              ksz = -kmaxqz
           end if
           picoef = TWO_PI  * REAL(ky)
           rky(1) = picoef * recip(2)
           rky(2) = picoef * recip(3)
           rky(3) = picoef * recip(5)

           do kz = ksz, kmaxqz
              picoef = TWO_PI  * REAL(kz)
              rkz(1) = picoef * recip(4)
              rkz(2) = picoef * recip(5)
              rkz(3) = picoef * recip(6)
              ksq    = kx*kx + ky*ky + kz*kz

              if (ksq .le. ksqmaxq .and. ksq .ne. 0) then

                 loop_count = loop_count + 1
                 mkv(1:3)   = rkx(1:3) + rky(1:3) + rkz(1:3)

                 if(QNoPMEwald) then
                    inner_loop = 0
                    do i = iastrt,iafinl             ! 1, Natom
                       inner_loop = inner_loop+1
                       xyz(1) = mkv(1)*X(i)
                       xyz(2) = mkv(2)*Y(i)
                       xyz(3) = mkv(3)*Z(i)

! Cache the values for doing a vectored cos ( and sin(x)=cos(x-(pi/2)) )
! X coordinates
                       Ktable(1,inner_loop,loop_count) = xyz(1)
                       Ktable(2,inner_loop,loop_count) = xyz(1) - HALFPI

! Y coordinates
                       Ktable(3,inner_loop,loop_count) = xyz(2)
                       Ktable(4,inner_loop,loop_count) = xyz(2) - HALFPI

! Z coordinates
                       Ktable(5,inner_loop,loop_count) = xyz(3)
                       Ktable(6,inner_loop,loop_count) = xyz(3) - HALFPI
                    end do

! Do a vectored cosine -> since we subtracted pi/2 from
! every other x,y,z value so we will end up with
! cos,sin,cos,sin...
                    call Vdcos(6*iatotl,Ktable(1,1,loop_count),   &
                               Ktable(1,1,loop_count))          ! iatotl=Natom
                 end if

! Cache the qm atom values for use later so we can
! access them linearly in memory
#if KEY_PARALLEL==1
                 do i = 1, Nquant
                    qmid = Iqmatoms(i)
                    xyz(1) = mkv(1)*X(qmid)
                    xyz(2) = mkv(2)*Y(qmid)
                    xyz(3) = mkv(3)*Z(qmid)
                    qmKtable(1,i,loop_count)   = xyz(1)
                    qmKtable(2,i,loop_count)   = xyz(1)-HALFPI
                    qmKtable(3,i,loop_count)   = xyz(2)
                    qmKtable(4,i,loop_count)   = xyz(2)-HALFPI
                    qmKtable(5,i,loop_count)   = xyz(3)
                    qmKtable(6,i,loop_count)   = xyz(3)-HALFPI
                 end do
                 call Vdcos(6*Nquant,qmKtable(1,1,loop_count),qmKtable(1,1,loop_count))
#else /**/
                 if(QNoPMEwald) then
                 ! don't have to compute this.
                    do i = 1, Nquant
                       qmid = Iqmatoms(i)
                       qmKtable(1:6,i,loop_count) = Ktable(1:6,Iqmatoms(i),loop_count)
                    end do
                 else
                 ! have to compute this.
                    do i = 1, Nquant
                       qmid = Iqmatoms(i)
                       xyz(1) = mkv(1)*X(qmid)
                       xyz(2) = mkv(2)*Y(qmid)
                       xyz(3) = mkv(3)*Z(qmid)
                       qmKtable(1,i,loop_count)   = xyz(1)
                       qmKtable(2,i,loop_count)   = xyz(1)-HALFPI
                       qmKtable(3,i,loop_count)   = xyz(2)
                       qmKtable(4,i,loop_count)   = xyz(2)-HALFPI
                       qmKtable(5,i,loop_count)   = xyz(3)
                       qmKtable(6,i,loop_count)   = xyz(3)-HALFPI
                    end do
                    call Vdcos(6*Nquant,qmKtable(1,1,loop_count),qmKtable(1,1,loop_count))
                 end if
#endif 
              end if
           end do                                     ! kz = ksz, kmaxqz
        end do                                        ! ky = ksy, kmaxqy
      End do                                          ! kx = 0, kmaxqx
                                   
      return
      END SUBROUTINE qm_ewald_calc_ktable


      SUBROUTINE qm_ewald_mm_pot(natom,nquant,   &
                              totkq,iatotl,iastrt,iafinl,itotal,itotkq,   &
                              ksqmaxq,kmaxqx,kmaxqy,kmaxqz,   &
                              npairs, Erfmod,   &
                              qm_xcrd,qm_coords,mmcharges,   &
                              rijdata,   &
                              kvec,ktable,qmktable,structfac_mm,empot,   &
                              kappa,QNoPMEwald,rij_incore)
!
! Compute the potential at the QM atom position due to 
! the ewald sum of the MM atoms.
! Called once per each MD step, before doing the SCF calculations.
!

  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_array_locations

#if KEY_PARALLEL==1
  use parallel  
#endif
  use erfcd_mod,only: erfcd
!
      implicit none

! Passed in
      integer, intent(in)    :: natom,nquant
      integer, intent(in)    :: totkq,iatotl,iastrt,iafinl,ksqmaxq,itotal,itotkq
      integer, intent(in)    :: kmaxqx,kmaxqy,kmaxqz,npairs,Erfmod

      real(chm_real),intent(in)    :: qm_xcrd(4*npairs),  &
                                      qm_coords(3,nquant),   &
                                      mmcharges(natom)
      real(chm_real),intent(in)    :: rijdata(QMMMNORIJ,*)
      real(chm_real),intent(in)    :: kvec(totkq),ktable(6,itotal,itotkq), & !   ktable(6,iatotl,totkq),   &
                                      qmktable(6,nquant,totkq)    ! iatotl=natom
      real(chm_real),intent(in)    :: kappa
      real(chm_real),intent(inout) :: structfac_mm(2,itotkq),empot(nquant)

      logical, intent(in)    :: QNoPMEwald,rij_incore

! Local variables
      integer                :: four_npairs, i, j, loop_count,inner_loop
      integer                :: kx, ky, kz, ksy, ksz, ksq
      real(chm_real)               :: erfcx, drfc, empottmp
      real(chm_real)               :: vec(3), r2, oneRIJ, RIJ
      real(chm_real)               :: xyz_cos(3),xyz_sin(3)
      real(chm_real)               :: ktgs(8), ksum, mmchg, sfact,rtmp_local(2),sin_sum,cos_sum
      integer  :: mstart,mstop,mstart2,mstop2
      integer  :: ISTRT_CHECK                 ! for external function

!
      four_npairs = ishft(npairs,2)                                  ! *4
! do some initialization
      mstart = 1
      mstop  = npairs
      mstart2= 1
      mstop2 = four_npairs
      structfac_mm(1:2,1:itotkq)=zero     ! initialization
#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         mstart = ISTRT_CHECK(mstop,npairs)
         mstart2= 4*(mstart-1)+1
         mstop2 = 4*mstop
      end if
#endif 

!1) Calculate Real space potential at QM atom position
      If (rij_incore) then
! out QM-MM distances have already been calculated and stored
        loop_count = 0

        do i = 1, nquant
           empottmp   = zero
#if KEY_PARALLEL==1
           loop_count = loop_count + (mstart-1) ! befor loop. 
#endif 
           do j = mstart2,mstop2,4              ! 1, four_npairs,4
              loop_count = loop_count+1
              RIJ        = rijdata(QMMMRIJ,loop_count)
              oneRIJ     = rijdata(QMMMONERIJ,loop_count)
              Call ERFCD(RIJ,kappa,erfcx,drfc,Erfmod)

              empottmp =  empottmp - qm_xcrd(j+3)*(one-erfcx)*oneRIJ  
!                                             ! qm_xcrd(j+3) = MM charge
           end do
#if KEY_PARALLEL==1
           loop_count = loop_count + (npairs-mstop) ! after loop.
#endif 
           empot(i) = empottmp
        end do
      Else
! recompute the distance between QM-MM paris
        do i = 1, nquant
           empottmp = zero
           do j = mstart2,mstop2,4              ! 1, four_npairs, 4
              vec(1:3) = qm_coords(1:3,i)-qm_xcrd(j:j+2)
              r2       = vec(1)**2 + vec(2)**2 + vec(3)**2
              oneRIJ   = one/sqrt(r2)
              RIJ      = r2*oneRIJ                         ! one/oneRIJ
              Call ERFCD(RIJ,kappa,erfcx,drfc,Erfmod)

              empottmp =  empottmp - qm_xcrd(j+3)*(one-erfcx)*oneRIJ
           end do
           empot(i) = empottmp
        end do
      End if

!2) Calculate K space potential at QM atom position
      if(QNoPMEwald) then       ! only has to do when QNoPMEwald=.true.
         loop_count = 0

         Do kx =0, kmaxqx
           if ( kx .eq. 0 ) then
              ksy = 0
           else
              ksy = -kmaxqy
           end if
           do ky = ksy, kmaxqy
              if ( kx .eq. 0 .and. ky .eq. 0 ) then
                 ksz = 1
              else
                 ksz = -kmaxqz
              end if
              do kz = ksz, kmaxqz
                 sfact= two
                 ksq  = kx*kx + ky*ky + kz*kz

                 if (ksq .le. ksqmaxq .and. ksq .ne. 0) then
                    loop_count = loop_count+1
                    ktgs(1:8) = zero

! loop over all MM atoms (skip QM)
                    inner_loop = 0
                    sin_sum=zero
                    cos_sum=zero
                    do j = iastrt,iafinl              ! 1, natom
!               if (.not. qm_atom_mask(j)) then
!               Quicker just to loop through all atoms,
!               mmchg will be zero for QM atoms
!               it is an MM atom
                       inner_loop = inner_loop + 1
                       xyz_cos(1) = ktable(1,inner_loop,loop_count)
                       xyz_sin(1) = ktable(2,inner_loop,loop_count)
                       xyz_cos(2) = ktable(3,inner_loop,loop_count)
                       xyz_sin(2) = ktable(4,inner_loop,loop_count)
                       xyz_cos(3) = ktable(5,inner_loop,loop_count)
                       xyz_sin(3) = ktable(6,inner_loop,loop_count)

                       mmchg      = mmcharges(j)
                       ktgs(1)    = ktgs(1) +   &
                                    mmchg*xyz_cos(1)*xyz_cos(2)*xyz_cos(3)
                       ktgs(2)    = ktgs(2) +   &
                                    mmchg*xyz_sin(1)*xyz_sin(2)*xyz_sin(3)
                       ktgs(3)    = ktgs(3) +   &
                                    mmchg*xyz_cos(1)*xyz_cos(2)*xyz_sin(3)
                       ktgs(4)    = ktgs(4) +   &
                                    mmchg*xyz_cos(1)*xyz_sin(2)*xyz_cos(3)
                       ktgs(5)    = ktgs(5) +   &
                                    mmchg*xyz_cos(1)*xyz_sin(2)*xyz_sin(3)
                       ktgs(6)    = ktgs(6) +   &
                                    mmchg*xyz_sin(1)*xyz_sin(2)*xyz_cos(3)
                       ktgs(7)    = ktgs(7) +   &
                                    mmchg*xyz_sin(1)*xyz_cos(2)*xyz_sin(3)
                       ktgs(8)    = ktgs(8) +   &
                                    mmchg*xyz_sin(1)*xyz_cos(2)*xyz_cos(3)
! do virial part: compute only the structure factor here.
!                 this will be used later in the routine computing gradient
!                 to compute the ewvirial.
                      rtmp_local(1)=xyz_cos(3)*xyz_cos(2)-xyz_sin(2)*xyz_sin(3)
                      rtmp_local(2)=xyz_cos(3)*xyz_sin(2)+xyz_cos(2)*xyz_sin(3)
                      sin_sum=sin_sum+mmchg*(xyz_cos(1)*rtmp_local(1) - &
                                             xyz_sin(1)*rtmp_local(2))
                      cos_sum=cos_sum+mmchg*(xyz_sin(1)*rtmp_local(1) + &
                                             xyz_cos(1)*rtmp_local(2))
!               end if
                    end do

! do virial part
! now, note that it is not complete when parallel, since it is sum over
! only iastrt,iafinl. Thus, in the end, it will be combined from all nodes.
                    structfac_mm(1,loop_count)=sin_sum
                    structfac_mm(2,loop_count)=cos_sum

! Now loop over quantum atoms
                    do j = 1, nquant
                       ksum = zero
                       xyz_cos(1) = qmktable(1,j, loop_count)
                       xyz_sin(1) = qmktable(2,j, loop_count)
                       xyz_cos(2) = qmktable(3,j, loop_count)
                       xyz_sin(2) = qmktable(4,j, loop_count)
                       xyz_cos(3) = qmktable(5,j, loop_count)
                       xyz_sin(3) = qmktable(6,j, loop_count)

                       ksum       = ksum +   &
                                  xyz_cos(1)*xyz_cos(2)*xyz_cos(3)*ktgs(1)
                       ksum       = ksum +   &
                                  xyz_sin(1)*xyz_sin(2)*xyz_sin(3)*ktgs(2)
                       ksum       = ksum +   &
                                  xyz_cos(1)*xyz_cos(2)*xyz_sin(3)*ktgs(3)
                       ksum       = ksum +   &
                                  xyz_cos(1)*xyz_sin(2)*xyz_cos(3)*ktgs(4)
                       ksum       = ksum +   &
                                  xyz_cos(1)*xyz_sin(2)*xyz_sin(3)*ktgs(5)
                       ksum       = ksum +   &
                                  xyz_sin(1)*xyz_sin(2)*xyz_cos(3)*ktgs(6)
                       ksum       = ksum +   &
                                  xyz_sin(1)*xyz_cos(2)*xyz_sin(3)*ktgs(7)
                       ksum       = ksum +   &
                                  xyz_sin(1)*xyz_cos(2)*xyz_cos(3)*ktgs(8)

                       empot(j)   = empot(j) + sfact*ksum*kvec(loop_count)
                    end do
                 end if

              end do
           end do
         End do
! now, distribute the structure_factor to collect all MM atom information.
#if KEY_PARALLEL==1
         if(numnod.gt.1) call GCOMBR(structfac_mm,2*itotkq)       
#endif
      End if       ! QNoPMEwald=.true.

! now, distribute empot to collect all MM atom information.
#if KEY_PARALLEL==1
      if (numnod.gt.1.and.QNoPMEwald) call GCOMB(empot,nquant)   
#endif

      return
      END SUBROUTINE qm_ewald_mm_pot


      SUBROUTINE qm_ewald_mm_pot_exl(nquant,   &
                                  nexl_atm, nexl_index,  &
                                  Erfmod, qm_coords, exl_xyz,   &
                                  empot, kappa)
!
! Compute the potential at the QM atom position due to the ewald 
! sum of the MM atoms in the exclusion list.
! Called once per each MD step, before doing the SCF calculations.
!

  use erfcd_mod,only: erfcd
  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      integer, intent(in)    :: nquant,nexl_atm
      integer, intent(in)    :: nexl_index(nexl_atm)
      integer, intent(in)    :: Erfmod

      real(chm_real),intent(in)    :: qm_coords(3,nquant)
      real(chm_real),intent(in)    :: exl_xyz(4,nexl_atm)
      real(chm_real),intent(in)    :: kappa
      real(chm_real),intent(out)   :: empot(nquant)

! Local variables
      integer                :: i, j
      real(chm_real)               :: erfcx, drfc, empottmp
      real(chm_real)               :: vec(3), r2, oneRIJ, RIJ

! Calculate Real space potential at QM atom position from
! MM atoms excluded from QM-MM non-bonded interactions.
      If (nexl_atm .gt. 0) then
        do i = 1, nquant
           empottmp = zero

           do j = 1, nexl_atm
              vec(1:3) = qm_coords(1:3,i)-exl_xyz(1:3,j)
              r2       = vec(1)**2 + vec(2)**2 + vec(3)**2
              oneRIJ   = one/sqrt(r2)
              RIJ      = r2*oneRIJ                         ! one/oneRIJ
              Call ERFCD(RIJ,kappa,erfcx,drfc,Erfmod)

              empottmp =  empottmp - exl_xyz(4,j)*(one-erfcx)*oneRIJ
           end do

           empot(i) = empot(i) + empottmp
        end do
      End if

      return
      END SUBROUTINE qm_ewald_mm_pot_exl


      SUBROUTINE qm_ewald_qm_pot(natom,nquant,totkq,   &
                              ksqmaxq,kmaxqx,kmaxqy,kmaxqz,   &
                              Erfmod,qm_coords,rijdata,   &
                              kvec,qmktable,eslf,   &
                              erfcx_data,kappa,   &
                              rij_incore,erfcx_incore,n1_qmqm,n2_qmqm)
!
! Compute the potential at the QM atom position due to the ewald 
! sum of the QM image atoms.
! Called once per each MD step, before doing the SCF calculations.
! The potential saved as nquant x nquant matrix, thus later it only
! needs to do matrix multiplication using Mulliken charges from QM 
! atoms
!

  use erfcd_mod,only: erfcd
  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations

      implicit none

! Passed in
      integer, intent(in)   :: natom,nquant
      integer, intent(in)   :: totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz,Erfmod
      integer, intent(in)   :: n1_qmqm,n2_qmqm

      real(chm_real),  intent(in) :: qm_coords(3,nquant),rijdata(n1_qmqm,*)  ! rijdata(QMQMNORIJ,*)
      real(chm_real),  intent(in) :: kvec(totkq),qmktable(6,nquant,totkq)
      real(chm_real),  intent(in) :: kappa
      real(chm_real),  intent(out):: eslf(nquant,nquant), erfcx_data(*)

      logical, intent(in)   :: rij_incore,erfcx_incore

! Local variables
      integer               :: i, j, loop_count,array_size
      integer               :: kx, ky, kz, ksy, ksz, ksq
      integer               :: iminus, iplus, offset
      real(chm_real)              :: esfact, qmixyz(3) 
      real(chm_real)              :: erfcx, drfc, empottmp
      real(chm_real)              :: vec(3), r2, oneRIJ, RIJ
      real(chm_real)              :: xyz_cos(3),xyz_sin(3)
      real(chm_real)              :: ktgs(3), ksum, sfact

      real(chm_real), PARAMETER   :: SQRTPI=1.77245385090551602729816748334d0
      real(chm_real)              :: INVSQRTPI=1.0d0/SQRTPI


! Initialization
      eslf = zero
      if(erfcx_incore) then
        array_size = ishft(nquant*nquant,-1)
        erfcx_data(1:array_size)=zero
      end if

!1) Self energy term
      esfact = -two*kappa*INVSQRTPI
      do i = 1, nquant
        eslf(i,i) = esfact
      end do

!2) Real space potential
      If (rij_incore) then
! Compute QM-QM (one-erfcx)/rij data and store into 
! qmqmerfcx_data in the form of the lower packed triangle.
! Hence, we need to split the QM-QM real space potential loop 
! into two.
        loop_count = 0
        do i = 2, nquant
           iminus = i-1
           do j = 1, iminus
              loop_count= loop_count+1
              oneRIJ    = rijdata(QMQMONERIJ,loop_count)
              RIJ       = rijdata(QMQMRIJ,loop_count)
 
              Call ERFCD(RIJ,kappa,erfcx,drfc,Erfmod)
 
              empottmp  = (one-erfcx)*oneRIJ
              eslf(j,i) = eslf(j,i) - empottmp
              eslf(i,j) = eslf(i,j) - empottmp
 
              if(erfcx_incore) erfcx_data(loop_count) = empottmp
           end do
        end do
      Else
        do i = 1, nquant
           qmixyz(1:3) = qm_coords(1:3,i)
           do j = 1, nquant
              if(i.ne.j) then
                 vec(1:3) = qmixyz(1:3)-qm_coords(1:3,j)
                 r2       = vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)
                 oneRIJ   = one / sqrt(r2)
                 RIJ      = r2*oneRIJ                      ! one/oneRIJ
 
                 Call ERFCD(RIJ,kappa,erfcx,drfc,Erfmod)

                 eslf(j,i) = eslf(j,i) - (one - erfcx)*oneRIJ
              end if
           end do
        end do
      End if


!3) K space potentail between QM atoms
      loop_count = 0
      Do kx = 0, kmaxqx
        if ( kx .eq. 0 ) then
           ksy = 0
        else
           ksy = -kmaxqy
        end if
        do ky = ksy, kmaxqy
           if ( kx .eq. 0 .and. ky .eq. 0 ) then
              ksz = 1
           else
              ksz = -kmaxqz
           end if
           do kz = ksz, kmaxqz
              sfact= two
              ksq  = kx*kx + ky*ky + kz*kz

              if ( ksq .le. ksqmaxq .and. ksq .ne. 0 ) then
                 loop_count = loop_count+1
                 ktgs(1:3)  = zero
                 do i = 1, nquant
                    iminus     = i-1
                    xyz_cos(1) = qmktable(1,i,loop_count)
                    xyz_sin(1) = qmktable(2,i,loop_count)
                    xyz_cos(2) = qmktable(3,i,loop_count)
                    xyz_sin(2) = qmktable(4,i,loop_count)
                    xyz_cos(3) = qmktable(5,i,loop_count)
                    xyz_sin(3) = qmktable(6,i,loop_count)

                    do j = 1, iminus
                       ktgs(1) = xyz_cos(1)*qmktable(1,j,loop_count) +   &
                                 xyz_sin(1)*qmktable(2,j,loop_count)
                       ktgs(2) = xyz_cos(2)*qmktable(3,j,loop_count) +   &
                                 xyz_sin(2)*qmktable(4,j,loop_count)
                       ktgs(3) = xyz_cos(3)*qmktable(5,j,loop_count) +   &
                                 xyz_sin(3)*qmktable(6,j,loop_count)
                       ksum    = sfact*ktgs(1)*ktgs(2)*ktgs(3)*   &
                                 kvec(loop_count)

                       eslf(j,i) = eslf(j,i) + ksum
                       eslf(i,j) = eslf(i,j) + ksum
                    end do
! for i=j
                    ktgs(1) = xyz_cos(1)*qmktable(1,i,loop_count) +   &
                              xyz_sin(1)*qmktable(2,i,loop_count)
                    ktgs(2) = xyz_cos(2)*qmktable(3,i,loop_count) +   &
                              xyz_sin(2)*qmktable(4,i,loop_count)
                    ktgs(3) = xyz_cos(3)*qmktable(5,i,loop_count) +   &
                              xyz_sin(3)*qmktable(6,i,loop_count)
                    ksum    = sfact*ktgs(1)*ktgs(2)*ktgs(3)*   &
                              kvec(loop_count)
                    eslf(i,i) = eslf(i,i) + ksum
                 end do
              end if

           end do
        end do
      End do

      return
      END SUBROUTINE qm_ewald_qm_pot


      SUBROUTINE qm_ewald_add_fock(nquant, fock_matrix, scf_mchg, empot,   &
                                eslf)
!
! 1) Compute Eslf contribution + Empot contribution at QM atom
! 2) Modify Fock matrix in diagonal elements
!

  use qmmm_module, only : qm2_params         ! , qmmm_ewald

  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions

      implicit none

! Passed in
      integer, intent(in)    :: nquant
      real(chm_real),intent(in)    :: scf_mchg(nquant)
      real(chm_real),intent(in)    :: empot(nquant), eslf(nquant,nquant)
      real(chm_real),intent(inout) :: fock_matrix(*)

! Local variables
      integer                :: i, ia, ib, i1, i2
      real(chm_real)               :: ewdpot, fct_unit


! Convert (electrons/angstrom) to (eV/Bohr)
      fct_unit = AU_TO_EV*BOHRS_TO_A  

! Add Empot+Eslf contribution to the diagonal elements of 
! the fock matrix
      Do i = 1, nquant
        ia = qm2_params%orb_loc(1,i)
        ib = qm2_params%orb_loc(2,i)

        ewdpot = DOT_PRODUCT(eslf(1:nquant,i),scf_mchg(1:nquant))
        ewdpot = (empot(i) + ewdpot) * fct_unit

        do i1 = ia, ib
           i2 = qm2_params%pascal_tri2(i1)
           fock_matrix(i2) = fock_matrix(i2) - ewdpot
        end do
      End do
 
      return
      END SUBROUTINE qm_ewald_add_fock


      SUBROUTINE qm_ewald_correct_ee(nquant,indx_for_qm,ee,empot,p)
!
! Up to this poiint, the energy for the Ewald sum only included half
! of the term from MM atoms, and half from the QM image atoms. The
! QM atoms should contribute only half, but the MM atoms should 
! contribute full. This routine compute the rest of the energy EE 
! for this.
!
! This routine should be called after a call to qm2_HELECT.
!

  use qmmm_module, only : qm2_params

  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions

#if KEY_PARALLEL==1
  use parallel  
#endif
!
      implicit none

! Passed in
      integer, intent(in)    :: nquant,indx_for_qm
      real(chm_real),intent(out)   :: ee
      real(chm_real),intent(in)    :: empot(nquant), P(*)

! Local variables
      integer                :: i, i1, ia, ib, i2
      real(chm_real)               :: etemp, fct_unit

      integer :: mstart,mstop
      integer :: mstart_keep(2),mstop_keep(2)
      integer :: ISTRT_CHECK                 ! for external function
      integer :: old_N(2) = 0
      save old_N,mstart_keep,mstop_keep

! for parallelization
      if(old_N(indx_for_qm) .ne. nquant) then
         mstart_keep(indx_for_qm)=1
         mstop_keep(indx_for_qm) =nquant
#if KEY_PARALLEL==1
         if(numnod.gt.1) mstart_keep(indx_for_qm) =  &
                         ISTRT_CHECK(mstop_keep(indx_for_qm),nquant)
      else
         if(QMPI) then
            mstart_keep(indx_for_qm)=1
            mstop_keep(indx_for_qm) =nquant
         end if
#endif /* */
      end if
      mstart = mstart_keep(indx_for_qm)
      mstop  = mstop_keep(indx_for_qm)
 
      etemp    = zero
      fct_unit = AU_TO_EV*BOHRS_TO_A

      Do i = mstart,mstop                     ! 1, nquant
        ia = qm2_params%orb_loc(1,i)
        ib = qm2_params%orb_loc(2,i)
        do i1 = ia, ib
           i2    = qm2_params%pascal_tri2(i1)
           etemp = etemp - empot(i)*p(i2)
        end do
      End do

      ee = half*etemp*fct_unit

      return
      END SUBROUTINE qm_ewald_correct_ee


      SUBROUTINE qm_ewald_core(nquant,ewald_core,core_chg,qmchg,empot, &
                               eslf)
!
! Computes the interaction of the Ewald potenital with CORE in QM 
! atoms. 
! Ewald_core in electron volts
!               Ewald_core = Sum(Core(i)*V(i))
!

  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions

      implicit none

! Passed in
      integer, intent(in)    :: nquant
      real(chm_real),intent(out)   :: ewald_core
      real(chm_real),intent(in)    :: core_chg(nquant),qmchg(nquant)
      real(chm_real),intent(in)    :: empot(nquant),eslf(nquant,nquant)

! Local variables
      integer                :: i
      real(chm_real)               :: ewdtmp, ewdpot

      ewdpot         = zero
      ewdtmp         = zero
      ewald_core     = zero

      Do i = 1, nquant
        ewdtmp     = DOT_PRODUCT(eslf(1:nquant,i),qmchg(1:nquant))
        ewdpot     = empot(i)   + half*ewdtmp 
        ewald_core = ewald_core + ewdpot*core_chg(i)
      End do

! convert energy unit: from (electrons/angstrom) to (eV/Bohr)
!                            AU_TO_EV*BOHRS_TO_A 
      ewald_core = ewald_core * AU_TO_EV * BOHRS_TO_A
     
      return
      END SUBROUTINE qm_ewald_core


      SUBROUTINE qm_ewald_real_space_gradient(natom,nquant,   &
                                           npairs, Erfmod,   &
                                           qm_xcrd,qm_coords, &
                                           scf_mchg,scf_mchg_2,   &
                                           qmmmrijdata,qmqmrijdata,   &
                                           dxyzqm,dxyzcl,   &
                                           kappa,   &
                                           qmmmrij_incore,   &
                                           qmqmrij_incore,   &
                                           n1_qmqm,n2_qmqm,n1_qmmm,n2_qmmm)
!
! Calculate Real Space Gradients
!
! Calculates the gradient at the QM atom for the real space 
! contribution from Ewald summation corrections.
!
! contribution contains,
! a) QM-MM real space contribution
! b) QM-QM real space contribution
! c) QM-exclude MM real space contribution 
!    (refer qm_ewald_real_space_gradient_exl routine)
!

  use qmmm_module, only : qm2_params
  use erfcd_mod,only: erfcd
  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations

#if KEY_PARALLEL==1
  use parallel  
#endif
!
      implicit none

! Passed in
      integer, intent(in)    :: natom,nquant
      integer, intent(in)    :: npairs,Erfmod
      integer, intent(in)    :: n1_qmqm,n2_qmqm,n1_qmmm,n2_qmmm

      real(chm_real), intent(in)   :: qm_xcrd(4*npairs),qm_coords(3,nquant),   &
                                scf_mchg(nquant),scf_mchg_2(nquant)
      real(chm_real), intent(in)   :: qmmmrijdata(n1_qmmm,*) , &
                                      qmqmrijdata(n1_qmqm,*)
      !real(chm_real), intent(in)   :: qmmmrijdata(QMMMNORIJ,*),   &
      !                                qmqmrijdata(QMQMNORIJ,*)
      real(chm_real), intent(inout):: dxyzqm(3,nquant),dxyzcl(*)
      real(chm_real), intent(in)   :: kappa

      logical, intent(in)    :: qmmmrij_incore,qmqmrij_incore

! Local variables
      integer                :: i,j,loop_count,four_npairs,   &
                                inner_loop_count
      integer                :: offset,iminus,iplus
      real(chm_real)               :: qmmulik_chg,df_qmmm,oneRIJ2,r2, &
                                onerij,rij
      real(chm_real)               :: vec(3),xyz_i(3),df_xyz(3),erfcx,drfc,  &
                                sfact
      integer  :: mstart,mstop,mstart2,mstop2
      integer  :: ISTRT_CHECK                 ! for external function

      four_npairs = ishft(npairs,2)                                 ! *4
      sfact       = AU_TO_KCAL*BOHRS_TO_A

! do some initialization
      mstart = 1
      mstop  = npairs
      mstart2= 1
      mstop2 = four_npairs
#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         mstart = ISTRT_CHECK(mstop,npairs)
         mstart2= 4*(mstart-1)+1
         mstop2 = 4*mstop
      end if
#endif 

!Step 1) do QM atoms with MM atoms in the cutoff list
      If (qmmmrij_incore) then
! distances are in qmmmrijdata
        loop_count = 0
        do i = 1, nquant
           inner_loop_count = 1
           qmmulik_chg      = scf_mchg_2(i)*sfact

#if KEY_PARALLEL==1
           loop_count = loop_count + (mstart-1) ! befor loop.
           inner_loop_count=inner_loop_count+3*(mstart-1)
#endif /* */
           do j = mstart2,mstop2,4                    ! 1,four_npairs,4
              loop_count = loop_count + 1
              vec(1:3)   = qm_coords(1:3,i)-qm_xcrd(j:j+2)

              oneRIJ     = qmmmrijdata(QMMMONERIJ,loop_count)
              oneRIJ2    = oneRIJ*oneRIJ
              rij        = qmmmrijdata(QMMMRIJ,loop_count)
              Call ERFCD(rij,kappa,erfcx,drfc,Erfmod)
! need to check
! erfcx=(2/3)exp((rij*kappa)^3)*INVSQRTPI
! drfc =two*kappa*exp(-(rij*kappa)*2)*INVSQRTPI

              df_qmmm    = qm_xcrd(j+3)*(-drfc+(one-erfcx)*oneRIJ)*   &
                                               oneRIJ2
              df_xyz(1:3)= vec(1:3)*df_qmmm*qmmulik_chg
 
              dxyzqm(1:3,i)= dxyzqm(1:3,i) + df_xyz(1:3)
              dxyzcl(inner_loop_count:inner_loop_count+2)    &
                         = dxyzcl(inner_loop_count:inner_loop_count+2)   &
                         - df_xyz(1:3)
              inner_loop_count = inner_loop_count + 3
           end do                                    ! j=1,four_npairs,4
#if KEY_PARALLEL==1
           loop_count = loop_count + (npairs-mstop) ! after loop.
#endif 
        end do                                       ! i=1,nquant
      Else
! We need to calculate the distance between QM-MM pairs on the fly.
        do i = 1, nquant
           inner_loop_count = 1
           qmmulik_chg      = scf_mchg_2(i)*sfact

#if KEY_PARALLEL==1
           inner_loop_count=inner_loop_count+3*(mstart-1)
#endif 
           do j = mstart2,mstop2,4                    ! 1,four_npairs,4
              vec(1:3)   = qm_coords(1:3,i)-qm_xcrd(j:j+2)
              r2         = vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)

              oneRIJ     = one/sqrt(r2)
              oneRIJ2    = oneRIJ*oneRIJ
              rij        = r2*oneRIJ                        ! one/oneRIJ
              Call ERFCD(rij,kappa,erfcx,drfc,Erfmod)

              df_qmmm    = qm_xcrd(j+3)*(-drfc+(one-erfcx)*oneRIJ)*   &
                                               oneRIJ2
              df_xyz(1:3)= vec(1:3)*df_qmmm*qmmulik_chg

              dxyzqm(1:3,i)= dxyzqm(1:3,i) + df_xyz(1:3)
              dxyzcl(inner_loop_count:inner_loop_count+2)    &
                         = dxyzcl(inner_loop_count:inner_loop_count+2)   &
                         - df_xyz(1:3)
              inner_loop_count = inner_loop_count + 3
           end do                                    ! j=1,four_npairs,4
        end do                                       ! i=1,nquant
      End if


!Step 2) do all real space QM atoms with QM atoms
      If (qmqmrij_incore) then
! Ignore the use of "erfcx_incore", since it is differ from 
! AMBER in computing drfc now, the distance information is
! qmqmrijdata, only lower packed half triangle
        loop_count = 0
        do i = 2, nquant
           xyz_i(1:3)  = qm_coords(1:3,i)
           qmmulik_chg = scf_mchg_2(i)*sfact
           iminus      = i-1
 
           do j = 1, iminus
              loop_count= loop_count+1
#if KEY_PARALLEL==1
              if(mynod .eq. MOD(loop_count-1,numnod)) then
#endif /*              */
              oneRIJ    = qmqmrijdata(QMQMONERIJ,loop_count)
              oneRIJ2   = oneRIJ*oneRIJ
              rij       = qmqmrijdata(QMQMRIJ,loop_count)
              Call ERFCD(rij,kappa,erfcx,drfc,Erfmod)
 
              df_qmmm    = half*qmmulik_chg*scf_mchg(j)*   &
                           (-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
!             df_qmmm    = qmmulik_chg*scf_mchg(j)*  
!    *                     (-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
 
              vec(1:3)   = xyz_i(1:3)-qm_coords(1:3,j)
              df_xyz(1:3)= vec(1:3)*df_qmmm
 
              dxyzqm(1:3,i) = dxyzqm(1:3,i) + df_xyz(1:3)
              dxyzqm(1:3,j) = dxyzqm(1:3,j) - df_xyz(1:3)
#if KEY_PARALLEL==1
              end if
#endif 
           end do                                           ! j=1,i-1
        end do                                              ! i=2,nquant

!***Note
! Since the interaction is pair-wise, we don't need this, and it should 
! not have half in df_qmmm above.
!       ! Now do the upper half triangle part (changed into this for QpKa_on.)
#if KEY_PARALLEL==1
        loop_count = 0
#endif 
        do i = 1, nquant
           xyz_i(1:3)  = qm_coords(1:3,i)
           qmmulik_chg = scf_mchg_2(i)*sfact
           iplus       = i+1
 
           do j = iplus,nquant
#if KEY_PARALLEL==1
              loop_count= loop_count+1
              if(mynod .eq. MOD(loop_count-1,numnod)) then
#endif 
              offset = qm2_params%pascal_tri1(j-1) + i
              oneRIJ = qmqmrijdata(QMQMONERIJ,offset)
              oneRIJ2= oneRIJ*oneRIJ
              rij    = qmqmrijdata(QMQMRIJ,offset)
              Call ERFCD(rij,kappa,erfcx,drfc,Erfmod)
 
              df_qmmm    = half*qmmulik_chg*scf_mchg(j)*   &
                           (-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
 
              vec(1:3)   = xyz_i(1:3)-qm_coords(1:3,j)
              df_xyz(1:3)= vec(1:3)*df_qmmm
 
              dxyzqm(1:3,i) = dxyzqm(1:3,i) + df_xyz(1:3)
              dxyzqm(1:3,j) = dxyzqm(1:3,j) - df_xyz(1:3)
#if KEY_PARALLEL==1
              end if
#endif 
           end do                                         ! j=i+1,nquant
        end do                                            ! i=1,nquant
!***End 

      Else 
! We have to calculate it all on the fly
#if KEY_PARALLEL==1
        loop_count = 0
#endif 
        do i = 1, nquant
           xyz_i(1:3)  = qm_coords(1:3,i)
           qmmulik_chg = scf_mchg_2(i)*sfact   ! refer "qm2_calc_mulliken"

           do j = 1, nquant
              if (i .ne. j) then
#if KEY_PARALLEL==1
                 loop_count = loop_count + 1
                 if(mynod .eq. MOD(loop_count-1,numnod)) then
#endif 
                 vec(1:3) = xyz_i(1:3)-qm_coords(1:3,j)
                 r2       = vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)
                 oneRIJ   = one / sqrt(r2)
                 oneRIJ2  = oneRIJ*oneRIJ
                 rij      = r2*oneRIJ                       ! one/oneRIJ
                 Call ERFCD(rij,kappa,erfcx,drfc,Erfmod)

                 df_qmmm    = half*qmmulik_chg*scf_mchg(j)*   &
                              (-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
                 df_xyz(1:3)= vec(1:3)*df_qmmm

                 dxyzqm(1:3,i)= dxyzqm(1:3,i) + df_xyz(1:3)
                 dxyzqm(1:3,j)= dxyzqm(1:3,j) - df_xyz(1:3)
#if KEY_PARALLEL==1
                 end if
#endif 
              end if
           end do
        end do
      End if

      return
      END SUBROUTINE qm_ewald_real_space_gradient


      SUBROUTINE qm_ewald_real_space_gradient_exl(natom,nquant,nexl_atm,   &
                                               nexl_index,   &
                                               Erfmod,   &
                                               qm_coords, &
                                               scf_mchg,scf_mchg_2,   &
                                               exl_xyz,   &
                                               dxyzqm,dexl_xyz,   &
                                               kappa)

!
! Calculate Real Space Gradietn
!
! Calculates the gradient at the QM atom for the real space 
! contribution from Ewald summation corrections.
!
! contribution contains,
! c) QM-exclude MM real space contribution
!

  use erfcd_mod,only: erfcd
  use qmmm_module, only : qm2_params

  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions

#if KEY_PARALLEL==1
  use parallel  
#endif
!
      implicit none

! Passed in
      integer, intent(in)    :: natom,nquant,nexl_atm
      integer, intent(in)    :: nexl_index(nexl_atm)
      integer, intent(in)    :: Erfmod

      real(chm_real),intent(in)    :: qm_coords(3,nquant), &
                                scf_mchg(nquant),scf_mchg_2(nquant)
      real(chm_real),intent(in)    :: exl_xyz(4,nexl_atm)
      real(chm_real),intent(inout) :: dxyzqm(3,nquant),dexl_xyz(3,nexl_atm)
      real(chm_real),intent(in)    :: kappa

! Local variables
      integer               :: i,j
      real(chm_real)              :: qmmulik_chg,df_qmmm,oneRIJ2,r2,onerij,rij
      real(chm_real)              :: vec(3),xyz_i(3),df_xyz(3),erfcx,drfc,  &
                               sfact

      sfact       = AU_TO_KCAL*BOHRS_TO_A

!Step 3) do QM atoms with exclude MM atoms from QM-MM nonbonded 
!        interactions
!        (IGMSEL(i)=5)
#if KEY_PARALLEL==1
      if(mynod.eq.0) then                    ! do only mynod.eq.0
#endif 
      If (nexl_atm .gt. 0) then
        do i = 1, nquant
           xyz_i(1:3)  = qm_coords(1:3,i)
           qmmulik_chg = scf_mchg_2(i)*sfact

           do j = 1, nexl_atm
              vec(1:3) = xyz_i(1:3)-exl_xyz(1:3,j)
              r2       = vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)
              oneRIJ   = one / sqrt(r2)
              oneRIJ2    = oneRIJ*oneRIJ
              rij        = r2*oneRIJ
              Call ERFCD(rij,kappa,erfcx,drfc,Erfmod)

              df_qmmm    = half*qmmulik_chg*exl_xyz(4,j)*   &
                            (-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
              df_xyz(1:3)= vec(1:3)*df_qmmm

              dxyzqm(1:3,i)   = dxyzqm(1:3,i)   + df_xyz(1:3)
              dexl_xyz(1:3,j) = dexl_xyz(1:3,j) - df_xyz(1:3)
           end do
        end do
      End if
#if KEY_PARALLEL==1
      end if
#endif 

      return
      END SUBROUTINE qm_ewald_real_space_gradient_exl   

      SUBROUTINE qm_ewald_recip_space_gradient(natom,nquant, &
                                            iastrt,iafinl,iatotl,itotal,itotkq, &
                                            iqmatoms,   &
                                            totkq,ksqmaxq,   &
                                            kmaxqx,kmaxqy,kmaxqz,   &
                                            mmcharges, &
                                            scf_mchg,scf_mchg_2,   &
                                            kvec,ktable,qmktable,structfac_mm,   &
                                            kappa, &
                                            recip,virial,   &
                                            d_ewald_mm,QNoPMEwald)
!
! Calculate Reciprocal Space Gradient
!
!    Calculate the gradient and save into "d_ewald_mm" array, which 
!    will be added into main gradient array at "ewaldf.src" file.
!

  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations

#if KEY_PARALLEL==1
  use parallel  
#endif
!
      implicit none

! Passed in
      integer, intent(in)    :: natom,nquant,iastrt,iafinl,iatotl,itotal,itotkq
      integer, intent(in)    :: totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz
      integer, intent(in)    :: iqmatoms(nquant)

      real(chm_real),intent(in)    :: mmcharges(natom), &
                                      scf_mchg(nquant),scf_mchg_2(nquant)
      real(chm_real),intent(in)    :: kvec(totkq),ktable(6,itotal,itotkq),  &  ! ktable(6,iatotl,totkq),  &
                                      qmktable(6,nquant,totkq), &              ! iatotl=natom
                                      structfac_mm(2,itotkq)
      real(chm_real),intent(in)    :: kappa,recip(6)
      real(chm_real),intent(inout)   :: d_ewald_mm(3,natom),virial(9)
      logical, intent(in)          :: QNoPMEwald

! Local variables
      integer                :: i, j, loop_count, qmid
      integer                :: kx, ky, kz, ksy, ksz, ksq
      real(chm_real)               :: qmmulik_chg,mmchg,sfact,ufact,picoef,  &
                                kvect
      real(chm_real)               :: vec(3), xyz_cos(3),xyz_sin(3),  &
                                        xyz_cos_j(3),xyz_sin_j(3)
      real(chm_real)               :: rkx(3),rky(3),rkz(3),mkv(3),ccfk(3)
      real(chm_real)               :: ktgs(8),ktg(3), ksum(nquant)
      real(chm_real)               :: fda(8), fdxyz(3)
      real(chm_real)               :: grad_qm(3,nquant)      ! local QM gradient
      real(chm_real)               :: pikapa,kxyzr(3),klen,ewpr,ewen
      integer  :: mstart,mstop,inner_loop
      integer  :: ISTRT_CHECK                 ! for external function
      real(chm_real) :: rtmp_local(2),ewen_qm_qm,ewen_qm_mm
      real(chm_real) :: sin_sum_mm,cos_sum_mm,sin_sum_qm,cos_sum_qm
#if KEY_PARALLEL==1
      integer,parameter :: i_array_length=4                  
      real(chm_real)    :: suml(4)                           
#endif

! do some initialization
      mstart = 1
      mstop  = nquant
#if KEY_PARALLEL==1
      if(numnod.gt.1) mstart = ISTRT_CHECK(mstop,nquant)
#endif 
!
      loop_count = 0
      ufact      = AU_TO_KCAL*BOHRS_TO_A
      grad_qm    = zero

! do virial part
      pikapa     = two*((PIi/kappa)**2)  ! two/(four*kappa*kappa)
      virial(1:9)= zero

      Do kx =0, kmaxqx
        if ( kx .eq. 0 ) then
           ksy = 0
        else
           ksy = -kmaxqy
        end if
        picoef = TWO_PI  * REAL(kx)
        rkx(1) = picoef * recip(1)
        rkx(2) = picoef * recip(2)
        rkx(3) = picoef * recip(4)

        do ky = ksy, kmaxqy
           if (kx .eq. 0 .and. ky .eq. 0) then
              ksz = 1
           else
              ksz = -kmaxqz
           end if
           picoef = TWO_PI  * REAL(ky)
           rky(1) = picoef * recip(2)
           rky(2) = picoef * recip(3)
           rky(3) = picoef * recip(5)

           do kz = ksz, kmaxqz
              sfact  = two
              picoef = TWO_PI  * REAL(kz)
              rkz(1) = picoef * recip(4)
              rkz(2) = picoef * recip(5)
              rkz(3) = picoef * recip(6)
              ksq    = kx*kx + ky*ky + kz*kz

              if (ksq .le. ksqmaxq .and. ksq .ne. 0) then

                 loop_count = loop_count+1
                 mkv(1:3)   = rkx(1:3) + rky(1:3) + rkz(1:3)

                 kvect      = sfact*kvec(loop_count)
                 ccfk(1:3)  = mkv(1:3)*kvect

! do virial part
                 kxyzr(1:3)     = mkv(1:3)*INVPI*half  ! mkv(1:3)/two_pi
                 klen           = kxyzr(1)*kxyzr(1)+  &
                                  kxyzr(2)*kxyzr(2)+  &
                                  kxyzr(3)*kxyzr(3)
                 ewpr           = (two/klen + pikapa)
! end

! QM-MM interactions
                 ktgs(1:8)  = zero
                 if(QNoPMEwald) then                ! only, loop when QNoPMEwald=.true.
                    inner_loop = 0
                    do j = iastrt,iafinl               ! 1, natom
! do over all MM atoms, since mmchg on qm atoms
! will be zero. Thus, no contribution
                       inner_loop = inner_loop + 1
                       xyz_cos(1) = ktable(1,inner_loop,loop_count)
                       xyz_sin(1) = ktable(2,inner_loop,loop_count)
                       xyz_cos(2) = ktable(3,inner_loop,loop_count)
                       xyz_sin(2) = ktable(4,inner_loop,loop_count)
                       xyz_cos(3) = ktable(5,inner_loop,loop_count)
                       xyz_sin(3) = ktable(6,inner_loop,loop_count)

                       mmchg      = mmcharges(j)
                       ktgs(1)    = ktgs(1) +   &
                                    mmchg*xyz_cos(1)*xyz_cos(2)*xyz_cos(3)
                       ktgs(2)    = ktgs(2) +   &
                                    mmchg*xyz_sin(1)*xyz_cos(2)*xyz_cos(3)
                       ktgs(3)    = ktgs(3) +   &
                                    mmchg*xyz_cos(1)*xyz_sin(2)*xyz_cos(3)
                       ktgs(4)    = ktgs(4) +   &
                                    mmchg*xyz_sin(1)*xyz_sin(2)*xyz_cos(3)
                       ktgs(5)    = ktgs(5) +   &
                                    mmchg*xyz_cos(1)*xyz_cos(2)*xyz_sin(3)
                       ktgs(6)    = ktgs(6) +   &
                                    mmchg*xyz_sin(1)*xyz_cos(2)*xyz_sin(3)
                       ktgs(7)    = ktgs(7) +   &
                                    mmchg*xyz_cos(1)*xyz_sin(2)*xyz_sin(3)
                       ktgs(8)    = ktgs(8) +   &
                                    mmchg*xyz_sin(1)*xyz_sin(2)*xyz_sin(3)
                    end do
                    do j = 1, nquant
                       xyz_cos(1) = qmktable(1,j,loop_count)
                       xyz_sin(1) = qmktable(2,j,loop_count)
                       xyz_cos(2) = qmktable(3,j,loop_count)
                       xyz_sin(2) = qmktable(4,j,loop_count)
                       xyz_cos(3) = qmktable(5,j,loop_count)
                       xyz_sin(3) = qmktable(6,j,loop_count)

! temporary array
                       fda(1)     = xyz_sin(1)*xyz_cos(2)*xyz_cos(3)
                       fda(2)     = xyz_cos(1)*xyz_cos(2)*xyz_cos(3)
                       fda(3)     = xyz_sin(1)*xyz_sin(2)*xyz_cos(3)
                       fda(4)     = xyz_cos(1)*xyz_sin(2)*xyz_cos(3)
                       fda(5)     = xyz_sin(1)*xyz_cos(2)*xyz_sin(3)
                       fda(6)     = xyz_cos(1)*xyz_cos(2)*xyz_sin(3)
                       fda(7)     = xyz_sin(1)*xyz_sin(2)*xyz_sin(3)
                       fda(8)     = xyz_cos(1)*xyz_sin(2)*xyz_sin(3)

! force on x-axis
                       fdxyz(1)   = -ktgs(1)*fda(1)+ktgs(2)*fda(2)   &
                                    -ktgs(3)*fda(3)+ktgs(4)*fda(4)   &
                                    -ktgs(5)*fda(5)+ktgs(6)*fda(6)   &
                                    -ktgs(7)*fda(7)+ktgs(8)*fda(8)
! force on y-axis
                       fdxyz(2)   = -ktgs(1)*fda(4)-ktgs(2)*fda(3)   &
                                    +ktgs(3)*fda(2)+ktgs(4)*fda(1)   &
                                    -ktgs(5)*fda(8)-ktgs(6)*fda(7)   &
                                    +ktgs(7)*fda(6)+ktgs(8)*fda(5)
! force on z-axis
                       fdxyz(3)   = -ktgs(1)*fda(6)-ktgs(2)*fda(5)   &
                                    -ktgs(3)*fda(8)-ktgs(4)*fda(7)   &
                                    +ktgs(5)*fda(2)+ktgs(6)*fda(1)   &
                                    +ktgs(7)*fda(4)+ktgs(8)*fda(3)

! put gradient on temporary grad_qm array
                       qmmulik_chg    = scf_mchg_2(j)*ufact
                       grad_qm(1:3,j) = grad_qm(1:3,j) +   &
                                        ccfk(1:3)*fdxyz(1:3)*qmmulik_chg
                    end do

! MM-QM interaction
                    fda(1:8) = zero
                    do j = 1, nquant
                       xyz_cos(1) = qmktable(1,j,loop_count)
                       xyz_sin(1) = qmktable(2,j,loop_count)
                       xyz_cos(2) = qmktable(3,j,loop_count)
                       xyz_sin(2) = qmktable(4,j,loop_count)
                       xyz_cos(3) = qmktable(5,j,loop_count)
                       xyz_sin(3) = qmktable(6,j,loop_count)

                       qmmulik_chg= scf_mchg_2(j)*ufact
                       fda(1)     = fda(1) +   &
                              qmmulik_chg*xyz_sin(1)*xyz_cos(2)*xyz_cos(3)
                       fda(2)     = fda(2) +   &
                              qmmulik_chg*xyz_cos(1)*xyz_cos(2)*xyz_cos(3)
                       fda(3)     = fda(3) +   &
                              qmmulik_chg*xyz_sin(1)*xyz_sin(2)*xyz_cos(3)
                       fda(4)     = fda(4) +   &
                              qmmulik_chg*xyz_cos(1)*xyz_sin(2)*xyz_cos(3)
                       fda(5)     = fda(5) +   &
                              qmmulik_chg*xyz_sin(1)*xyz_cos(2)*xyz_sin(3)
                       fda(6)     = fda(6) +   &
                              qmmulik_chg*xyz_cos(1)*xyz_cos(2)*xyz_sin(3)
                       fda(7)     = fda(7) +   &
                              qmmulik_chg*xyz_sin(1)*xyz_sin(2)*xyz_sin(3)
                       fda(8)     = fda(8) +   &
                              qmmulik_chg*xyz_cos(1)*xyz_sin(2)*xyz_sin(3)
                    end do
                    inner_loop = 0
                    do j = iastrt,iafinl               ! 1, natom
                       inner_loop = inner_loop + 1
                       xyz_cos(1) = ktable(1,inner_loop,loop_count)
                       xyz_sin(1) = ktable(2,inner_loop,loop_count)
                       xyz_cos(2) = ktable(3,inner_loop,loop_count)
                       xyz_sin(2) = ktable(4,inner_loop,loop_count)
                       xyz_cos(3) = ktable(5,inner_loop,loop_count)
                       xyz_sin(3) = ktable(6,inner_loop,loop_count)

                       ktgs(1)    = xyz_cos(1)*xyz_cos(2)*xyz_cos(3)
                       ktgs(2)    = xyz_sin(1)*xyz_cos(2)*xyz_cos(3)
                       ktgs(3)    = xyz_cos(1)*xyz_sin(2)*xyz_cos(3)
                       ktgs(4)    = xyz_sin(1)*xyz_sin(2)*xyz_cos(3)
                       ktgs(5)    = xyz_cos(1)*xyz_cos(2)*xyz_sin(3)
                       ktgs(6)    = xyz_sin(1)*xyz_cos(2)*xyz_sin(3)
                       ktgs(7)    = xyz_cos(1)*xyz_sin(2)*xyz_sin(3)
                       ktgs(8)    = xyz_sin(1)*xyz_sin(2)*xyz_sin(3)

! force on x-axis
                       fdxyz(1)   = -ktgs(1)*fda(1)+ktgs(2)*fda(2)   &
                                    -ktgs(3)*fda(3)+ktgs(4)*fda(4)   &
                                    -ktgs(5)*fda(5)+ktgs(6)*fda(6)   &
                                    -ktgs(7)*fda(7)+ktgs(8)*fda(8)
! force on y-axis
                       fdxyz(2)   = -ktgs(1)*fda(4)-ktgs(2)*fda(3)   &
                                    +ktgs(3)*fda(2)+ktgs(4)*fda(1)   &
                                    -ktgs(5)*fda(8)-ktgs(6)*fda(7)   &
                                    +ktgs(7)*fda(6)+ktgs(8)*fda(5)
! force on z-axis
                       fdxyz(3)   = -ktgs(1)*fda(6)-ktgs(2)*fda(5)   &
                                    -ktgs(3)*fda(8)-ktgs(4)*fda(7)   &
                                    +ktgs(5)*fda(2)+ktgs(6)*fda(1)   &
                                    +ktgs(7)*fda(4)+ktgs(8)*fda(3)

! qm muliken charge has been multiplied already
                       mmchg             = mmcharges(j)
                       d_ewald_mm(1:3,j) = d_ewald_mm(1:3,j)   &
                                         - ccfk(1:3)*fdxyz(1:3)*mmchg
                    end do
                 end if     ! QNoPMEwald

! QM-QM interaction
                 sin_sum_qm=zero
                 cos_sum_qm=zero
                 do i = 1, nquant
                    xyz_cos(1) = qmktable(1,i,loop_count)
                    xyz_sin(1) = qmktable(2,i,loop_count)
                    xyz_cos(2) = qmktable(3,i,loop_count)
                    xyz_sin(2) = qmktable(4,i,loop_count)
                    xyz_cos(3) = qmktable(5,i,loop_count)
                    xyz_sin(3) = qmktable(6,i,loop_count)
                    qmmulik_chg= scf_mchg_2(i)*ufact

! do virial part, for qm.
                    rtmp_local(1)=xyz_cos(3)*xyz_cos(2)-xyz_sin(2)*xyz_sin(3)
                    rtmp_local(2)=xyz_cos(3)*xyz_sin(2)+xyz_cos(2)*xyz_sin(3)
                    sin_sum_qm=sin_sum_qm+scf_mchg_2(i)*(xyz_cos(1)*rtmp_local(1) - &
                                                         xyz_sin(1)*rtmp_local(2))
                    cos_sum_qm=cos_sum_qm+scf_mchg_2(i)*(xyz_sin(1)*rtmp_local(1) + &
                                                         xyz_cos(1)*rtmp_local(2))
! end

                    do j = mstart,mstop                    ! 1,nquant
                       xyz_cos_j(1) = qmktable(1,j,loop_count)
                       xyz_sin_j(1) = qmktable(2,j,loop_count)
                       xyz_cos_j(2) = qmktable(3,j,loop_count)
                       xyz_sin_j(2) = qmktable(4,j,loop_count)
                       xyz_cos_j(3) = qmktable(5,j,loop_count)
                       xyz_sin_j(3) = qmktable(6,j,loop_count)

                       ktg(1:3)     = xyz_cos_j(1:3)*xyz_cos(1:3)+   &
                                      xyz_sin_j(1:3)*xyz_sin(1:3)
                       fdxyz(1:3)   =-xyz_cos_j(1:3)*xyz_sin(1:3)+   &
                                      xyz_sin_j(1:3)*xyz_cos(1:3)

! force on x,y,z-axis: temporary use of FDA array

! scf mulliken charge on qm atom j *atom i
! (adjusted to kcals)
                       mmchg  = scf_mchg(j)*qmmulik_chg

                       fda(1) = mmchg*ccfk(1)*fdxyz(1)*ktg(2)  *ktg(3)
                       fda(2) = mmchg*ccfk(2)*ktg(1)  *fdxyz(2)*ktg(3)
                       fda(3) = mmchg*ccfk(3)*ktg(1)  *ktg(2)  *fdxyz(3)

                       if (i .eq. j) then
                          grad_qm(1:3,i) = grad_qm(1:3,i) + fda(1:3)
                       else
                          grad_qm(1:3,i) = grad_qm(1:3,i)+half*fda(1:3)
                          grad_qm(1:3,j) = grad_qm(1:3,j)-half*fda(1:3)
                       end if
                    end do
                 end do
! do virial part
!
! total |S(m)| = sin_sum**2+ cos_sum**2      , whre sin_sum=sin_sum_mm+sin_sum_qm
!              = sin_sum_mm*sin_sum_mm+cos_sum_mm*cos_sum_mm       , mm-only component and will be handled in the mm part.
!               +two(sin_sum_mm*sin_sum_qm+cos_sum_mm*cos_sum*qm)  , mm-qm component
!               +sin_sum_qm*sin_sum_qm+cos_sun_qm*cos_sun_qm       , qm-qm component.
!
! where structfac_mm is already computed from qm_ewald_mm_pot.

                 if(QNoPMEwald) then
                    sin_sum_mm= structfac_mm(1,loop_count)
                    cos_sum_mm= structfac_mm(2,loop_count)
                    ewen_qm_mm= sin_sum_mm*sin_sum_qm + cos_sum_mm*cos_sum_qm
                 else
                    ewen_qm_mm= zero
                 end if
                 ewen_qm_qm= half*(sin_sum_qm**2+cos_sum_qm**2)
                 ewen      = ewen_qm_mm+ewen_qm_qm
                 ewen      = kvect*ewen*CCELEC
                 virial(1) = virial(1)+ewen*(one-ewpr*kxyzr(1)*kxyzr(1)) ! xx
                 virial(2) = virial(2)-ewen*ewpr*kxyzr(1)*kxyzr(2)       ! xy
                 virial(3) = virial(3)-ewen*ewpr*kxyzr(1)*kxyzr(3)       ! xz
                 virial(4) = virial(4)-ewen*ewpr*kxyzr(1)*kxyzr(2)       ! yx
                 virial(5) = virial(5)+ewen*(one-ewpr*kxyzr(2)*kxyzr(2)) ! yy
                 virial(6) = virial(6)-ewen*ewpr*kxyzr(2)*kxyzr(3)       ! yz
                 virial(7) = virial(7)-ewen*ewpr*kxyzr(1)*kxyzr(3)       ! zx
                 virial(8) = virial(8)-ewen*ewpr*kxyzr(2)*kxyzr(3)       ! zy
                 virial(9) = virial(9)+ewen*(one-ewpr*kxyzr(3)*kxyzr(3)) ! zz
! end
              end if
           end do                                     ! kz = ksz, kmaxqz
        end do                                        ! ky = ksy, kmaxqy
      End do                                          ! kx =0, kmaxqx

! Now put gradient on grad_qm into d_ewald_mm array
      Do i = 1, nquant
        qmid = iqmatoms(i)
        d_ewald_mm(1:3,qmid) = d_ewald_mm(1:3,qmid) + grad_qm(1:3,i)
      End do

! if you want to use virial, only from pme, even including qm-qm pairs.
!      if(.not.QNoPMEwald) virial(1:9)=zero

      return
      END SUBROUTINE qm_ewald_recip_space_gradient


!=================================================================================================!
!========                          PME Section Begins Here                                ========!
!=================================================================================================!
      SUBROUTINE qm_pme_mm_pot(natom,nquant,x,y,z,cg,scf_mchg_2, &
                              recip_vec,volume,empot,kappa)
!
! Compute the potential at the QM atom position due to
! the ewald sum of the MM atoms.
! Called once per each MD step, before doing the SCF calculations.
!
! Calculate K space PME potential at QM atom position
!
  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_array_locations
  use qmmm_module, only : qmmm_ewald

#if KEY_PARALLEL==1
  use parallel  
#endif
  use pme_module
  use pmeutil
!
  use dimens_fcm
  use exfunc
  use number
  use inbnd
  use image
  use consta
  use stream
  use squantm,only: lqmewd
!
      implicit none

! Passed in
      integer, intent(in)    :: natom,nquant

      real(chm_real),intent(in)    :: x(*),y(*),z(*),cg(*)
      real(chm_real),intent(in)    :: kappa,volume, recip_vec(6)
      real(chm_real),intent(inout) :: empot(nquant),scf_mchg_2(nquant)

! Local variables
      integer        :: i,j,ii,atfrst,atlast,nat
      integer        :: alloc_err
      real(chm_real) :: recip(3,3),virial(6)
      integer        :: siztheta, sizdtheta, siz_q
      integer        :: nfftdim1, nfftdim2, nfftdim3

      logical        :: ok

!!!#if KEY_PARALLEL==1 /* (paramain) */
!!!#if KEY_PARAFULL==1 /* (parfmain) */
!!!      atfrst=1+iparpt(mynod)
!!!      atlast=iparpt(mynodp)
!!!#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /* (parfmain) */
!!!      atfrst=1
!!!      atlast=natom
!!!#endif /* (parfmain) */
!!!#else /* (paramain) */
!!!      ATFRST=1
!!!      ATLAST=NATOM
!!!#endif /* (paramain) */
!
! so, it does not depend on whether use QMPI or not.
      atfrst=qmmm_ewald%iastrt
      atlast=qmmm_ewald%iafinl

      !
      NAT   =qmmm_ewald%iatotl       ! =ATLAST-ATFRST+1

      RECIP(1,1) = recip_vec(1)
      RECIP(2,2) = recip_vec(3)
      RECIP(3,3) = recip_vec(6)
      RECIP(1,2) = recip_vec(2)
      RECIP(2,1) = recip_vec(2)
      RECIP(1,3) = recip_vec(4)
      RECIP(3,1) = recip_vec(4)
      RECIP(2,3) = recip_vec(5)
      RECIP(3,2) = recip_vec(5)

      !-------------------------------------------------------------------
      ! INPUT
      !      NFFT1,NFFT2,NFFT3 are the (integer) dimensions of the charge grid array
      !      NATOM is number of atoms
      !      FORDER is the order of B-spline interpolation
      !      x,y,z:   atomic coords
      !      CG  atomic charges
      !      recip_vec: array of reciprocal unit cell vectors
      !      VOLUME: the volume of the unit cell
      !      KAPPA=ewald_coeff:   ewald convergence parameter
      ! OUTPUT
      !      siz_Q=3d charge grid array
      !      sizfftab is permanent 3d fft table storage
      !      sizffwrk is temporary 3d fft work storage
      !      siztheta is size of arrays theta1-3 dtheta1-3
      !
      !   All pointers are the integer names of the real(chm_real) variables that
      !   will be filled
      !
      !   Get memory for scratch arrays, free them after summation.
      !
      call get_fftdims(nfftdim1,nfftdim2,nfftdim3,i,j)
      siztheta  = natom*xnsymm*forder
      sizdtheta = (natom+1)*forder
      nattot=natom*xnsymm
#if KEY_PARALLEL==1
      SIZ_Q = max(2*NFFTDIM1*NFFTDIM2*mxyslabs,2*NFFTDIM1*NFFTDIM3*mxzslabs) 
#else 
      SIZ_Q = 2*NFFTDIM1*NFFTDIM2*NFFTDIM3                                   
#endif
      !
!      if(allocated(qarray) .and. siz_q .gt. size(qarray)) deallocate(qarray)
      if(.not.allocated(qarray)) then
         allocate(qarray(siz_q),stat=alloc_err)
         if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray"
      end if
!      if(allocated(qarray_mm) .and. siz_q .gt. size(qarray_mm)) deallocate(qarray_mm)
      if(.not.allocated(qarray_mm)) then
         allocate(qarray_mm(siz_q),stat=alloc_err)
         if(alloc_err /= 0 ) write(0,*)"unable to allocate qarray_mm"
      end if

      if(allocated(qm_atm_grad_comp) .and. 3*nquant.gt.size(qm_atm_grad_comp)) then
         deallocate(qm_atm_grad_comp)
      end if
      if(.not.allocated(qm_atm_grad_comp)) allocate(qm_atm_grad_comp(3,nquant))

!      if(allocated(fr1) .and. nattot.gt.size(fr1)) deallocate(fr1,fr2,fr3,cg1)
      if(.not. allocated(fr1)) then
         allocate(fr1(nattot),fr2(nattot),fr3(nattot),cg1(nattot),stat=alloc_err)
         if(alloc_err /= 0 ) write(0,*)"unable to allocate fr arrays"

      ! initialize
         fr1(1:nattot)=zero
         fr2(1:nattot)=zero
         fr3(1:nattot)=zero
         cg1(1:nattot)=zero
      end if
      call allocate_bspline(natom,nattot)

      !
      !==================QM-MM interactions==============================
#if KEY_COLFFT==1 /*colfft*/
      !==================COLUMN FFT METHOD ==============================
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QTURBO==1
      if(LQMEWD) call wrndie(-5,'<QM_PME_MM_POT>','QM/MM-PME do not support COLFFT.')
#endif 
#else /*      (colfft)*/
      !.ab.Fix: xnsymm should be set (not 0 !).
      IF(XNSYMM == 0) CALL WRNDIE(-5,'<QM_PME_MM_POT>','XNSYMM is zero: build crystal.')
      call do_pme_ksp_qm_mm_pot(natom,nquant,x,y,z,cg,recip,volume, &
                       kappa,forder,empot,virial,qm_atm_grad_comp,scf_mchg_2, &
                       sizfftab,sizffwrk,siztheta,siz_q,cg1, &
                       xnsymm,maxsym,xsymop)
#endif /*        (colfft)*/

!!!      deallocate(fr1,fr2,fr3,cg1,stat=alloc_err)
!!!      if(alloc_err /= 0 ) write(0,*)"unable to deallocate fr1,2,3"
!!!      deallocate(qarray,stat=alloc_err)
!!!      if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray"

!!!      call deallocate_bspline()

#if KEY_PARALLEL==1
      if (numnod.gt.1) call GCOMB(empot,nquant)    
      ! debug.
      !if(mynod.eq.0) then                          
      !   do i=1,nquant                             
      !      write(6,*) 'a2',i,empot(i),mynod       
      !   end do                                    
      !end if                                       
#endif

      return
      END SUBROUTINE qm_pme_mm_pot


      SUBROUTINE do_pme_ksp_qm_mm_pot(natom,nquant,x,y,z,cg,recip,volume,ewald_coeff,forder, &
                                      ewd_potential,virial,qm_atm_grad_comp_local,scf_mchg_2, &
                                      sizfftab,sizffwrk,siztheta,siz_q,cg1_local, &
                                      xnsymm,maxsyml,xsymop)

  use pme_module
  use pmeutil,only:nxyslab,mxyslabs,mxzslabs,nfft1,nfft2, &
                   nfft3,get_sc_fract, &
                   get_fftdims,fft3d0rc,fft3d_zxyrc

      ! OUTPUT
      !       ewd_potential:  ewald reciprocal or k-space  potential at qm atom site
      !
  use exfunc
  use number
  use parallel
  use stream
#if KEY_BLOCK==1
  use block_fcm              
#endif
  use memory
  use squantm, only : QMINB1_dual,QMINB2_dual,MMINB2_dual
  use gamess_fcm, only : IGMSEL

      implicit none

      integer        :: forder
      real(chm_real) :: recip(3,3),volume,ewald_coeff
      real(chm_real) :: x(*),y(*),z(*),ewd_potential(*),virial(6),qm_atm_grad_comp_local(3,*)
      real(chm_real) :: cg(*),scf_mchg_2(nquant)
      integer        :: natom,nquant
      integer   sizfftab,sizffwrk,siztheta,siz_q        ! sizes of some arrays

      ! storage: These arrays can be tossed after leaving this routine
      real(chm_real),intent(in) :: cg1_local(*)

      integer :: xnsymm,maxsyml
      integer :: xsymop(3,4,xnsymm)

      real(chm_real) :: scale
      integer nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork
      integer latm
      integer igood, kbot, ktop, i0,i
      logical    :: q_grad_and_pot

      nattot=natom*xnsymm

      !  get some integer array dimensions
      call get_fftdims(nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork)

!      if(allocated(tmpy) .and. 2*nfftdim1.ne.size(tmpy)) deallocate(tmpy)
!      if(allocated(alpha) .and. nfft1.ne.size(alpha))    deallocate(alpha,beta)
!      if(.not.allocated(tmpy)) allocate(tmpy(2*nfftdim1))
!      if(.not.allocated(alpha)) allocate(alpha(nfft1),beta(nfft1))

      scale = ONE
      call fft3d0rc(0,scale,qarray,nfftdim1,nfftdim2,tmpy,alpha,beta)
      !
      if(xnsymm > 1) call get_sc_fract(fr1,fr2,fr3,cg1_local, &
                                       natom,nattot,x,y,z,recip, &
                                       xnsymm,maxsyml,xsymop,cg)

      !       make array for keeping track of atoms important to
      !         this processor
!!#if KEY_PARALLEL==1
!!      latm=max((nattot*(3+nxyslab(0)))/nfft3*2,1)    
!!#endif
!!#if KEY_PARALLEL==0
      latm=nattot                                    
!!#endif
!      if(allocated(lmy_ks) .and. nattot.ne.size(lmy_ks)) then
!         call chmdealloc('sqnt_qm2_ewald.src','do_pme_ksp_qm_mm_pot','lmy_ks',latm,intg=lmy_ks)
!         call chmdealloc('sqnt_qm2_ewald.src','do_pme_ksp_qm_mm_pot','lmy_ks_inv',nattot,intg=lmy_ks_inv)
!      end if
!      if(.not.allocated(lmy_ks)) then
      call chmalloc('sqnt_qm2_ewald.src','do_pme_ksp_qm_mm_pot','lmy_ks',latm,intg=lmy_ks)
      call chmalloc('sqnt_qm2_ewald.src','do_pme_ksp_qm_mm_pot','lmy_ks_inv',nattot,intg=lmy_ks_inv)
!      end if
      !-------- symmetrical case ------------------------------
      !        fill frac coords and thetas in fill_ch_grid
      !              use min image charge array: cg
      if(xnsymm == 1) then
         call fill_ch_grid(igood, kbot, ktop,nattot,cg, &
#if KEY_BLOCK==1
                           CG1_local,                              & 
#endif
                           x,y,z,recip,natom,xnsymm, &
                           nfftdim1,nfftdim2,lmy_ks,latm, &
                           lmy_ks_inv,mxyslabs)
      else
         !-------- asymmetrical case ------------------------------
         !        fill frac coords  in GET_SC_FRACT but thetas in fill_ch_grid
         !           use the whole unit cell charge array: cg1_local
         call fill_ch_grid(igood, kbot, ktop, nattot,cg1_local, &
#if KEY_BLOCK==1
                           CG1_local,                             & 
#endif
                           x,y,z,recip,natom,xnsymm, &
                           nfftdim1,nfftdim2,lmy_ks,latm, &
                           lmy_ks_inv,mxyslabs)
      endif
      ! to use in gradient evaluation.
      igood_qmmm=igood

      ! Fill charge-drid array relevant for qm atoms only.
      ! This will be used later for the computation of gradient component
      ! applied to each MM atoms by each QM atoms.
      q_grad_and_pot=.true.        ! for the computation of potential.
      call fill_ch_grid_qm_mm(kbot,ktop,nquant,scf_mchg_2, &
                              x,y,z,recip,natom,xnsymm, &
                              nfftdim1,nfftdim2,mxyslabs, &
                              lmy_ks_inv,latm, &
                              qminb1_dual,q_grad_and_pot)

!      if(allocated(tmpy) .and. 2*nfftdim1.ne.size(tmpy)) deallocate(tmpy)
!      if(allocated(alpha) .and. nfft1.ne.size(alpha))    deallocate(alpha,beta)
      if(.not.allocated(tmpy)) allocate(tmpy(2*nfftdim1))
      if(.not.allocated(alpha)) allocate(alpha(nfft1),beta(nfft1))

      call fft_backrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)
      qarray_mm(1:siz_q)=Qarray(1:siz_q)                        ! save for later virial calculation.

      ! calculate, (B*C)*F(qarray)
      call convol_fr_space_qm_mm(qfinit,rewcut,ewald_coeff,volume,recip, &
                                 nfftdim1,nfftdim2,nfftdim3, &
                                 Qarray,qarray_mm,virial,q_grad_and_pot)
      ! now, forward fourier transform of qarray produces
      ! the convolution of theta(=F(B*C)) and Qarray.
      call fft_forwardrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)


      call potential_sumrc_qm_mm(igood, kbot, ktop, natom, nquant,      &
                                 Qarray,ewd_potential,qm_atm_grad_comp_local, &
                                 recip,volume,forder,nfftdim1,nfftdim2,nfftdim3, &
                                 lmy_ks_inv,latm,xnsymm, &
                                 igmsel,mminb2_dual(1:natom,1),qminb1_dual)

!!!      deallocate(tmpy,alpha,beta)
!!!      call chmdealloc('sqnt_qm2_ewald.src','do_pme_ksp_qm_mm_pot','lmy_ks',latm,intg=lmy_ks)
!!!      call chmdealloc('sqnt_qm2_ewald.src','do_pme_ksp_qm_mm_pot','lmy_ks_inv',nattot,intg=lmy_ks_inv)

      return
      END SUBROUTINE do_pme_ksp_qm_mm_pot


      SUBROUTINE qm_pme_mm_grad(natom,nquant,iqmatoms,x,y,z,d_ewald_mm,cg,  &
                                scf_mchg_2,recip_vec,volume,kappa,ewvirial)
!
! Compute the gradient at the QM and MM atom position due to
! the reciprocal space summation. Called after the scf convergence.
!
! Calculate K space PME gradient at QM and MM atom position
!
  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_array_locations

#if KEY_PARALLEL==1
  use parallel  
#endif
  use pme_module
  use pmeutil
!
  use dimens_fcm
  use exfunc
  use number
  use inbnd
  use image
  use consta
  use stream
  use squantm,only: lqmewd
!
      implicit none

! Passed in
      integer, intent(in)    :: natom,nquant
      integer, intent(in)    :: iqmatoms(nquant)

      real(chm_real),intent(in)    :: x(*),y(*),z(*),d_ewald_mm(3,*),cg(*)
      real(chm_real),intent(in)    :: kappa
      real(chm_real),intent(in)    :: scf_mchg_2(nquant),recip_vec(6),volume
      real(chm_real),intent(inout) :: ewvirial(9)

! Local variables
!!!      integer        :: ii,atfrst,atlast,nat
      integer        :: i,j
      integer        :: alloc_err
      real(chm_real) :: recip(3,3),virial(6),cfact
      integer        :: siztheta, sizdtheta, siz_q
      integer        :: nfftdim1, nfftdim2, nfftdim3

      logical        :: ok

!!!#if KEY_PARALLEL==1 /* (paramain) */
!!!#if KEY_PARAFULL==1 /* (parfmain) */
!!!      atfrst=1+iparpt(mynod)
!!!      atlast=iparpt(mynodp)
!!!#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /* (parfmain) */
!!!      atfrst=1
!!!      atlast=natom
!!!#endif /* (parfmain) */
!!!#else  /* (paramain) */
!!!      ATFRST=1
!!!      ATLAST=NATOM
!!!#endif /* (paramain) */
!!!      !
!!!      NAT=ATLAST-ATFRST+1

      RECIP(1,1) = recip_vec(1)
      RECIP(2,2) = recip_vec(3)
      RECIP(3,3) = recip_vec(6)
      RECIP(1,2) = recip_vec(2)
      RECIP(2,1) = recip_vec(2)
      RECIP(1,3) = recip_vec(4)
      RECIP(3,1) = recip_vec(4)
      RECIP(2,3) = recip_vec(5)
      RECIP(3,2) = recip_vec(5)

      !-------------------------------------------------------------------
      ! INPUT
      !      NFFT1,NFFT2,NFFT3 are the (integer) dimensions of the charge grid array
      !      NATOM is number of atoms
      !      FORDER is the order of B-spline interpolation
      !      x,y,z:   atomic coords
      !      CG  atomic charges
      !      recip_vec: array of reciprocal unit cell vectors
      !      VOLUME: the volume of the unit cell
      !      KAPPA=ewald_coeff:   ewald convergence parameter
      ! OUTPUT
      !      siz_Q=3d charge grid array
      !      sizfftab is permanent 3d fft table storage
      !      sizffwrk is temporary 3d fft work storage
      !      siztheta is size of arrays theta1-3 dtheta1-3
      !      d_ewald_mm: forces incremented by k-space sum
      !      EWVIRIAL=virial:  virial due to k-space sum (valid for atomic scaling;
      !                rigid molecule virial needs a correction term not
      !                computed here
      !
      call get_fftdims(nfftdim1,nfftdim2,nfftdim3,i,j)
      siztheta  = natom*xnsymm*forder
      sizdtheta = (natom+1)*forder
      nattot=natom*xnsymm
#if KEY_PARALLEL==1
      SIZ_Q = max(2*NFFTDIM1*NFFTDIM2*mxyslabs,2*NFFTDIM1*NFFTDIM3*mxzslabs) 
#else 
      SIZ_Q = 2*NFFTDIM1*NFFTDIM2*NFFTDIM3                                   
#endif
      !

      !==================QM-MM interactions==============================
#if KEY_COLFFT==1 /*colfft*/
      !==================COLUMN FFT METHOD ==============================
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QTURBO==1
      if(LQMEWD) call wrndie(-5,'<PME_QMMM>','QM/MM-PME do not support COLFFT.')
#endif 
#else /*      (colfft)*/

      call do_pme_ksp_qm_mm_grad(natom,nquant,iqmatoms,forder,volume,kappa, &
                                 recip,virial,x,y,z,d_ewald_mm,cg,cg1, &
                                 qm_atm_grad_comp,scf_mchg_2, &
                                 sizfftab,sizffwrk,siztheta,siz_q,xnsymm,maxsym,xsymop)
#endif /*        (colfft)*/

!     for the virial contribution.
      cfact=CCELEC/(xnsymm**2)
      ewvirial(1) = ewvirial(1) - virial(1)*cfact
      ewvirial(2) = ewvirial(2) - virial(2)*cfact
      ewvirial(3) = ewvirial(3) - virial(3)*cfact
      ewvirial(4) = ewvirial(4) - virial(2)*cfact
      ewvirial(5) = ewvirial(5) - virial(4)*cfact
      ewvirial(6) = ewvirial(6) - virial(5)*cfact
      ewvirial(7) = ewvirial(7) - virial(3)*cfact
      ewvirial(8) = ewvirial(8) - virial(5)*cfact
      ewvirial(9) = ewvirial(9) - virial(6)*cfact
!
      deallocate(fr1,fr2,fr3,cg1,stat=alloc_err)
      if(alloc_err /= 0 ) write(0,*)"unable to deallocate fr1,2,3"
      deallocate(qm_atm_grad_comp,stat=alloc_err)
      if(alloc_err /= 0 ) write(0,*)"unable to deallocate qm_atm_grad_comp"
      deallocate(qarray,stat=alloc_err)
      if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray"
      deallocate(qarray_mm,stat=alloc_err)
      if(alloc_err /= 0 ) write(0,*)"unable to deallocate qarray_mm"

      call deallocate_bspline()

      return
      END SUBROUTINE qm_pme_mm_grad


      SUBROUTINE do_pme_ksp_qm_mm_grad(natom,nquant,iqmatoms,forder,volume,ewald_coeff, &
                                       recip,virial,x,y,z,d_ewald_mm,cg,cg1_local,   &
                                       qm_atm_grad_comp_local,scf_mchg_2,  &
                                       sizfftab,sizffwrk,siztheta,siz_q,xnsymm,maxsyml,xsymop)

  use pme_module
  use pmeutil,only:nxyslab,mxyslabs,mxzslabs,nfft1,nfft2, &
                   nfft3,get_sc_fract, &
                   get_fftdims,fft3d0rc,fft3d_zxyrc

      ! OUTPUT
      !       eer:  ewald reciprocal or k-space  energy
      !       dx,dy,dz: forces incremented by k-space sum
      !       virial:  virial due to k-space sum (valid for atomic scaling;
      !                rigid molecule virial needs a correction term not
      !                computed here
      !
  use exfunc
  use number
  use parallel
  use stream
#if KEY_BLOCK==1
  use block_fcm              
#endif
  use memory
  use squantm, only : QMINB1_dual,QMINB2_dual
  use gamess_fcm, only : IGMSEL

      implicit none

      integer        :: natom,nquant
      integer        :: iqmatoms(nquant),forder
      real(chm_real) :: recip(3,3),volume,ewald_coeff
      real(chm_real) :: x(*),y(*),z(*),d_ewald_mm(3,*),scf_mchg_2(nquant),virial(6), &
                        qm_atm_grad_comp_local(3,*)
      real(chm_real) :: cg(*)

      integer :: i,iqm
      integer :: sizfftab,sizffwrk,siztheta,siz_q        ! sizes of some arrays

      ! storage: These arrays can be tossed after leaving this routine
      real(chm_real),intent(in) :: cg1_local(*)

!!!      real(chm_real) :: vir_2(6)
!!!      real(chm_real),allocatable,dimension(:) :: qarray_2

      integer :: xnsymm,maxsyml
      integer :: xsymop(3,4,xnsymm)

      integer :: nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork
      integer :: latm
      integer :: kbot, ktop, i0, igood
      logical :: q_grad_and_pot

      nattot=natom*xnsymm

      !=================================================================
      ! 1st for recip-pme gradient on qm atoms, put gradient into d_ewald_mm array.
      do i=1,nquant
         iqm=iqmatoms(i)
         d_ewald_mm(1,iqm)=d_ewald_mm(1,iqm)+scf_mchg_2(i)*qm_atm_grad_comp_local(1,i)
         d_ewald_mm(2,iqm)=d_ewald_mm(2,iqm)+scf_mchg_2(i)*qm_atm_grad_comp_local(2,i)
         d_ewald_mm(3,iqm)=d_ewald_mm(3,iqm)+scf_mchg_2(i)*qm_atm_grad_comp_local(3,i)
      end do
      !=================================================================

      !=================================================================
      ! Now, get the graient component for each mm atoms by qm atoms.

      !  get some integer array dimensions
      call get_fftdims(nfftdim1,nfftdim2,nfftdim3,nfftable,nffwork)

      !  make array for keeping track of atoms important to this processor
!!#if KEY_PARALLEL==1
!!      latm=max((nattot*(3+nxyslab(0)))/nfft3*2,1)    
!!#endif
!!#if KEY_PARALLEL==0
      latm=nattot                                    
!!#endif

      ! Fill charge-grid array relevant for qm atoms only.
      q_grad_and_pot=.false.       ! for gradient calculation
      call fill_ch_grid_qm_mm(kbot, ktop,nquant,scf_mchg_2, &
                              x,y,z,recip,natom,xnsymm, &
                              nfftdim1,nfftdim2,mxyslabs, &
                              lmy_ks_inv,latm,qminb1_dual,q_grad_and_pot)

!      if(allocated(tmpy) .and. 2*nfftdim1.ne.size(tmpy)) deallocate(tmpy)
!      if(allocated(alpha) .and. nfft1.ne.size(alpha))    deallocate(alpha,beta)
      if(.not.allocated(tmpy)) allocate(tmpy(2*nfftdim1))
      if(.not.allocated(alpha)) allocate(alpha(nfft1),beta(nfft1))

      call fft_backrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)

      ! here: Qarray contains the Fourier transformed structure factor contribution
      !              from QM atoms.
      !       Qarray_mm contains the Fourier transformed structure factor
      !                 contribution from MM atoms.
      ! Now, compute virial component & calculate, (B*C)*F(qarray).
!!!      allocate(qarray_2(siz_q))
!!!      qarray_2=Qarray
      call convol_fr_space_qm_mm(qfinit,rewcut,ewald_coeff,volume,recip, &
                                 nfftdim1,nfftdim2,nfftdim3, &
                                 Qarray,Qarray_mm,virial,q_grad_and_pot)

!!!      ! qm-qm:  if you want to use qm-qm virial only from pme calculation,
!!!      !         use the following routines.
!!!      vir_2=zero
!!!      call convol_fr_space_qm_mm(qfinit,rewcut,ewald_coeff,volume,recip, &
!!!                                 nfftdim1,nfftdim2,nfftdim3, &
!!!                                 Qarray_2,Qarray_2,vir_2,q_grad_and_pot)
!!!      virial(1:6)=virial(1:6)+half*vir_2(1:6)
!!!      deallocate(qarray_2)

      ! now, forward fourier transform of qarray produces
      ! the convolution of theta(=F(B*C)) and Qarray.
      call fft_forwardrc(Qarray,nfftdim1,nfftdim2,nfftdim3,nffwork)

      call grad_sumrc_qm_mm(igood_qmmm,kbot,ktop,natom,nquant,     &
                            Qarray,cg,scf_mchg_2,d_ewald_mm,recip, &
                            forder,nfftdim1,nfftdim2,nfftdim3,lmy_ks,latm, &
                            xnsymm,igmsel)

      !=================================================================

      deallocate(tmpy,alpha,beta)
      call chmdealloc('sqnt_qm2_ewald.src','do_pme_ksp_qm_mm_grad','lmy_ks',latm,intg=lmy_ks)
      call chmdealloc('sqnt_qm2_ewald.src','do_pme_ksp_qm_mm_grad','lmy_ks_inv',nattot,intg=lmy_ks_inv)

      return
      END SUBROUTINE do_pme_ksp_qm_mm_grad


  !***********************************************************************
  subroutine grad_sumrc_qm_mm(igood, kbot, ktop,numatoms,natqm,       &
                              qarray_local,cg,scf_mchg_2,d_ewald_mm,  &
                              recip,ordr,nfftdim1,nfftdim2,nfftdim3,  &
                              my_ks,latm,xnsymm,igmsel_local)

  !
  ! This routine compute the gradient at the QM and QM atom sites
  ! applied by all k-space terms.
  !
  ! I have to only assume xnsymm is 1. Other cases does not support yet.
  !
  ! Note: parent routine must call with
  ! natqm
  ! IGMSEL (or igmsel_dual, depending on the actual calcualtions.)
  !

  use pme_module
  use pmeutil,only: mxystart,mxyslabs,mxzslabs, &
                    nfft1,nfft2,nfft3, &
                    theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  use number
  use dimens_fcm
  use consta
  use parallel
  use stream
  use qm2_constants, only : FOURPI,TWO_PI
  use qm2_conversions, only : AU_TO_KCAL,BOHRS_TO_A

    implicit none

    integer,intent(in) :: igood, kbot, ktop
    integer,intent(in) :: numatoms,natqm,ordr
    integer,intent(in) :: nfftdim1,nfftdim2,nfftdim3,xnsymm
    real(chm_real)     :: qarray_local(*), cg(*), scf_mchg_2(natqm), d_ewald_mm(3,*)
    integer,intent(in) :: latm,my_ks(latm),igmsel_local(*)

    real(chm_real),intent(in) :: recip(9)

    integer :: igoo,ig,iqm
    integer :: I,J,K,KQ,i_keep,j_keep,k_keep,n,ITH1,ITH2,ITH3,IPT1,IPT2,IPT3
    integer :: rcskip,nfftdimrc                   ! RCFFT addition
    real(chm_real) :: CFACT,fxyz(3),vala(3),val(3),ufact

    rcskip=1
    if(nfft1/2 /= (nfft1+1)/2) CALL WRNDIE(-5,'<grad_sumrc_qm_mm>','fftx dimension not even ')
    nfftdimrc=nfft1+4
    !
    ufact      = AU_TO_KCAL*BOHRS_TO_A
    if(xnsymm == 1)then
       CFACT=ufact                    ! CCELEC, since unit conversion
    else
       CFACT=ufact/XNSYMM             ! CCELEC/XNSYMM
    end if
    !
    loopig: do ig = 1,igood
       n=my_ks(ig)
       if(xnsymm == 1)then
          igoo=ig
       else
          igoo=n
       endif

       ! only loop over mm atoms.
       !if((igmsel_local(igoo).ne.1) .and. (igmsel_local(igoo).ne.2)) then
       ! probably, this is safer.
       if((igmsel_local(n).ne.1) .and. (igmsel_local(n).ne.2)) then
          K_keep = INT(FR3(igoo)) - ORDR + 1 + NFFT3
          J_keep = INT(FR2(igoo)) - ORDR + 1 + NFFT2
          I_keep = INT(FR1(igoo)) - ORDR + 1 + NFFT1

          K        = k_keep
          fxyz(1:3)= zero
          do ITH3 = 1,ORDR
             K=K+1
             IF(K > NFFT3) K=K-NFFT3
             KQ=K
#if KEY_PARALLEL==1
             if ( K  >=  KBOT .AND. K  <=  KTOP ) then      
                KQ = K - MXYSTART(MYNOD)                    
#endif
                IPT1=(KQ-1)*NFFTDIM2 -1
                J = j_keep
                I = i_keep
                IF(I >= NFFT1) I=I-NFFT1

                vala(1)= cg(n)*NFFT1*THETA3(ITH3,ig)
                vala(2)= cg(n)*NFFT2*THETA3(ITH3,ig)
                vala(3)= cg(n)*NFFT3*DTHETA3(ITH3,igoo)

                do ITH2 = 1,ORDR
                   J=J+1
                   IF(J > NFFT2) J=J-NFFT2

                   val(1) = vala(1)*THETA2(ITH2,ig)
                   val(2) = vala(2)*DTHETA2(ITH2,igoo)
                   val(3) = vala(3)*THETA2(ITH2,ig)

                   IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                   IPT3= IPT2 + rcskip*(NFFT1-I)
                   do ITH1 = 1,ORDR
                      fxyz(1)=fxyz(1)+val(1)*qarray_local(IPT2)*DTHETA1(ITH1,igoo)
                      fxyz(2)=fxyz(2)+val(2)*qarray_local(IPT2)*THETA1(ITH1,ig)
                      fxyz(3)=fxyz(3)+val(3)*qarray_local(IPT2)*THETA1(ITH1,ig)

                      IPT2=IPT2+rcskip
                      IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                   end do
                end do
#if KEY_PARALLEL==1
             end if      
#endif
          end do
          !
          d_ewald_mm(1,n)=d_ewald_mm(1,n)+CFACT*(recip(1)*fxyz(1)+recip(4)*fxyz(2)+recip(7)*fxyz(3))
          d_ewald_mm(2,n)=d_ewald_mm(2,n)+CFACT*(recip(2)*fxyz(1)+recip(5)*fxyz(2)+recip(8)*fxyz(3))
          d_ewald_mm(3,n)=d_ewald_mm(3,n)+CFACT*(recip(3)*fxyz(1)+recip(6)*fxyz(2)+recip(9)*fxyz(3))
       end if
    end do loopig
    RETURN
  END subroutine grad_sumrc_qm_mm


      !***********************************************************************
  subroutine fill_ch_grid_qm_mm(kbot, ktop, natqm, scf_mchg_2, &
                                x,y,z,recip,natom,xnsymm, &
                                nfftdim1,nfftdim2,nfftdim3, &
                                my_ks_inv,latm,qminb_local,q_grad_and_pot)

  use pme_module
  use pmeutil,only:nfft1,nfft2,nfft3,forder, &
#if KEY_PARALLEL==1
                   mxystart,mxyslabs,  &  
#endif
                   theta1,theta2,theta3, &
                   dtheta1,dtheta2,dtheta3,fill_bspline
     !
     ! This routine fills the charge grid, Q for each qm atom. This will
     ! be called after calling fill_ch_grid.

     !
     ! I have to only assume xnsymm is 1. Other cases does not support yet.
     !

     !
     !---------------------------------------------------------------------
     ! INPUT:
     !      natqm:  number of qm atoms
     !      nfft1,nfft2,nfft3: the charge grid dimensions
     !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
     ! OUTPUT:
     !
     ! This routine fills the charge grid, Q.
     ! C.ab. scales charges (CG1) according to HybH scheme...
     !
     !---------------------------------------------------------------------
     ! INPUT:
     !      theta1,theta2,theta3: the spline coeff arrays
     !      fr1,fr2,fr3 the scaled and shifted fractional coords
     !      nfft1,nfft2,nfft3: the charge grid dimensions
     !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
     ! OUTPUT:
     !      Q the charge grid
     !
     !---------------------------------------------------------------------
     !
  use number
  use parallel
  use dimens_fcm

    implicit none

    integer :: natom,natqm
    integer :: nfftdim1,nfftdim2,nfftdim3
    real(chm_real) :: x(*),y(*),z(*), recip(9), scf_mchg_2(natqm)
    integer :: qminb_local(*)
    integer :: latm,my_ks_inv(*)
    integer :: kbot, ktop, xnsymm
    logical :: q_grad_and_pot

    real(chm_real) :: prod,proda
    real(chm_real) :: fr1n,fr2n,fr3n,w

    integer :: n,ith1,ith2,ith3,i,j,k,kq,ipt1,ipt2,ipt3,ipt,dim_q0,iqm
    integer :: enumtasks,itask,kdel,kbot0
    integer :: igood
    integer :: rcskip,nfftdimrc         ! rcfft addition

    rcskip=1
    if(nfft1/2 /= (nfft1+1)/2) call wrndie(-5,'<fill_ch_grid_qm_qm>','fftx dimension not even ')
    nfftdimrc=nfft1+4
    !
    igood=0
#if KEY_PARALLEL==1
    enumtasks = 1
    kdel = nfft3/enumtasks
    if ( kdel  ==  0 )kdel = 1
    !
    kbot0 = mxystart(mynod)
    kbot = kbot0 + 1
    ktop = kbot0 + mxyslabs
#else /**/
    kbot0 = 0
    kbot = 1
    ktop = nfft3
#endif 

    ! allocate arrays:
    if(q_grad_and_pot) then
       if(allocated(i_qarray_mm) .and. size(i_qarray_mm).ne.3*forder*natqm) deallocate(i_qarray_mm)
       if(allocated(r_qarray_mm) .and. size(r_qarray_mm).ne.3*forder*natqm) deallocate(r_qarray_mm)
       if(.not.allocated(i_qarray_mm)) allocate(i_qarray_mm(forder,forder,forder,natqm))
       if(.not.allocated(r_qarray_mm)) allocate(r_qarray_mm(forder,forder,forder,natqm))
    end if

    ! Initialization...
    dim_q0 = 2*nfftdim1*nfftdim2*nfftdim3
    if(q_grad_and_pot) then
       i_qarray_mm        = 0                   ! i have to allocate this array first somewhere
       r_qarray_mm        = zero                ! same
    else
       qarray(1:dim_q0) = zero
    end if

    !------------------------------------------
    !          MFC NOTE: THERE COULD BE A PRE-FILTER HERE TO ELIMINATE
    !                      MOST OF THE ATOMS AND KEEP A LIST OF APPROPRIATE
    !                      ATOMS EITHER FOR EACH PE, EACH X-Y SLAB,
    !                      OR EACH Q ELEMENT
    !          MFC NOte Note: Looks like I am doing that filter now....
    !------------------------------------------
    if(q_grad_and_pot) then
       ! only loop over QM atoms, since it only filters qarray terms relevant for
       ! qm atoms.
       loopiqm: do iqm = 1,natqm
          n=qminb_local(iqm)      ! pointer for the index in x/y/z arrays.
          igood=my_ks_inv(n)
          if(igood.le.0) cycle loopiqm

          w    = x(n)*recip(1)+y(n)*recip(2)+z(n)*recip(3)
          fr1n = nfft1*(w - anint(w) + half)
          w    = x(n)*recip(4)+y(n)*recip(5)+z(n)*recip(6)
          fr2n = nfft2*(w - anint(w) + half)
          w    = x(n)*recip(7)+y(n)*recip(8)+z(n)*recip(9)
          fr3n = nfft3*(w - anint(w) + half)

          k = int(fr3n) - forder + 1 + nfft3

          do ith3 = 1,forder
             k=k+1
             if(k > nfft3) k=k-nfft3
             kq=k
#if KEY_PARALLEL==1
             if ( k  >=  kbot .and. k  <=  ktop )then     
                kq = k - kbot0                            
#endif
                proda = theta3(ith3,igood)                         ! not includ echarge here. charge(n)

                j = int(fr2n) - forder + 1 + nfft2
                ipt1 = (kq-1)*nfftdim2 - 1

                i = int(fr1n) - forder + 1 + nfft1
                if(i >= nfft1) i=i-nfft1

                do ith2 = 1,forder
                   j=j+1
                   if(j > nfft2) j=j-nfft2
                   prod = theta2(ith2,igood)*proda
                   ipt2= rcskip*((ipt1+j)*nfftdimrc+i)+1
                   ipt3= ipt2 + rcskip*(nfft1-i)

                   do ith1 = 1,forder
                      ! In the end, it will be something like:
                      ! Qarray(i_qarray_mm(i,j,k,iqm))= sum_iqm [r_qarray_mm(i,j,k,iqm)*qm_chareg(iqm)]

                      i_qarray_mm(ith1,ith2,ith3,iqm)=ipt2                     ! pointer for qarray_qm
                      r_qarray_mm(ith1,ith2,ith3,iqm)=THETA1(ITH1,IGOOD)*PROD  ! value for qarray_qm, contribution
                                                                            ! from iqm atom.

                      ipt2=ipt2+rcskip
                      if(ipt2 >= ipt3) ipt2=ipt2-nfft1*rcskip
                   enddo
                enddo
#if KEY_PARALLEL==1
             endif         
#endif
          enddo
       enddo loopiqm

    else           ! for gradient calculations.
       do iqm = 1,natqm
          n=qminb_local(iqm)      ! pointer for the index in x/y/z arrays.
          do ith3 = 1,forder
             do ith2 = 1,forder
                do ith1 = 1,forder
                   ! In the end, it will be something like:
                   ! Qarray(i_qarray_mm(i,j,k,iqm))= sum_iqm [r_qarray_mm(i,j,k,iqm)*qm_chareg(iqm)]

                   ipt2 = i_qarray_mm(ith1,ith2,ith3,iqm)
                   if(ipt2.gt.0) then
                      Qarray(ipt2)=Qarray(ipt2)+r_qarray_mm(ith1,ith2,ith3,iqm)*scf_mchg_2(iqm)
                   end if
                enddo
             enddo
          enddo
       enddo
    end if

    return
  end subroutine fill_ch_grid_qm_mm


  !***********************************************************************
  subroutine convol_fr_space_qm_mm(qfinit_local,rewcut_local,ewaldcof,volume,recip, &
                        nfftdim1,nfftdim2,nfftdim3, &
                        Qarray_local,Qarray_mm_local,vir,q_grad_and_pot)

  use pme_module
  use pmeutil,only:mxzslabs,mxzstart, &
                   nfft1,nfft2,nfft3,bsp_mod1,bsp_mod2,bsp_mod3

     !
     !-------------------------------------------------------------------
     !
     !  QFINIT   - Flag indicating a long range cutoff is to be applied
     !  REWCUT   - The long range cutoff diatance (only if QFINIT)
     !  Qarray   - The main charge grid & after this routine, it is
     !             F(Q)*(C*B), in which the forward 3DFFT produces the convolution of theta*Q.
     !  Qarray_mm- The fourier transformed structure factor from mm atoms.
     !             It will be used to compute virial contribution.
     !  EWALDCOF - The kappa value
     !  VOLUME   - The volume of the perodic system
     !  RECIP    - The inverse of the box length matrix
     !  VIR      - The virial contribution from qm/mm-pme interactions, in which
     !             this will be computed only q_grad_and_pot=.false., meaning
     !             after scf-converged and when computing the gradient.
     !  BSP_MOD1 - The inverse of the B-spline coefficients (a direction)
     !  BSP_MOD2 - The inverse of the B-spline coefficients (b direction)
     !  BSP_MOD3 - The inverse of the B-spline coefficients (c direction)
     !  NFFT1    - The number of grid points (a direction)
     !  NFFT2    - The number of grid points (b direction)
     !  NFFT3    - The number of grid points (c direction)
     !  NFFTDIM1 - The dimension of grid points (a direction)
     !  NFFTDIM2 - The dimension of grid points (b direction)
     !  NFFTDIM3 - The dimension of grid points (c direction)
     !
     !-------------------------------------------------------------------
     !
     !   Finite distance cutoff code added by B. Brooks - NIH - 3/28/98
     !        ref: E.L.Pollock&J. Glosli, Computer Physics Communications,
     !        95 (1996) 93-110.
     !
  use number
  use consta
  use parallel

    implicit none

    LOGICAL,intent(in) :: QFINIT_local
    real(chm_real),intent(in) ::  REWCUT_local
    INTEGER,intent(in) :: NFFTDIM1,NFFTDIM2,NFFTDIM3
    !
    real(chm_real) :: EWALDCOF,VOLUME
    real(chm_real) :: RECIP(9),Qarray_local(*),Qarray_mm_local(*),vir(6)
    logical, intent(in) :: q_grad_and_pot             ! =.true., when computing potential.
                                                      ! =.false., when computing gradient.


    real(chm_real) :: FAC,ETERM,VTERM
    real(chm_real) :: DEN1,DEN2,DEN3,DEN4
    INTEGER        :: K,K0,K1,K2,K3,K2Q,M1,M2,M3
    INTEGER        :: K1s,K2s,K3s,M1s,M2s,M3s
    INTEGER        :: IPT1,IPT2,IPT3,NF1,NF2,NF3
    real(chm_real) :: MVAL,MCUT,MSQ,MSQR,STRUC2,VCORR,ESTR
    real(chm_real) :: MHAT(3),MHATA(3),MHATB(3)
    real(chm_real) :: MHATs(3),DENs,ETERMs,VTERMs,ESTRs,MSQs,STRUC2s,MSQRs
    LOGICAL        :: QFIN


    FAC = PI**2/EWALDCOF**2
    MCUT= TWO*PI*REWCUT_local
    QFIN=QFINIT_local

    NF1 = NFFT1/2
    IF(2*NF1 < NFFT1) NF1 = NF1+1
    NF2 = NFFT2/2
    IF(2*NF2 < NFFT2) NF2 = NF2+1
    NF3 = NFFT3/2
    IF(2*NF3 < NFFT3) NF3 = NF3+1

    DEN1 = ONE/(PI*VOLUME)

#if KEY_PMEPLSMA==1
#if KEY_PARALLEL==1
    if(mynod == 0)then     
#endif
       qarray_local(1:2)   =zero
#if KEY_PARALLEL==1
    endif                  
#endif
#endif 

    if(q_grad_and_pot) then            ! when computing potential contribution.
       IPT1=1
       do K2Q = 1, MXZSLABS
          K2=K2Q
#if KEY_PARALLEL==1
          IF(MYNOD > 0) K2 = K2Q + MXZSTART(MYNOD)     
#endif

          M2 = K2 - 1
          IF(K2 > NF2) M2 = M2 - NFFT2
          DEN2       = DEN1*BSP_MOD2(K2)
          MHATA(1:3) = RECIP(4:6)*M2

          IPT2=IPT1
          do K1 = 1, NF1+1
             M1 = K1 - 1
             IF(K1 > NF1) M1 = M1 - NFFT1
             DEN3       = DEN2*BSP_MOD1(K1)
             MHATB(1:3) = MHATA(1:3)+RECIP(1:3)*M1

             IPT3=IPT2
             IF(K1+K2 == 2) THEN
                K0=2
                IPT3=IPT3+2
             ELSE
                K0=1
             ENDIF

             do K3 = K0,NFFT3
                M3 = K3 - 1
                IF(K3 > NF3) M3 = M3 - NFFT3
                DEN4      = DEN3*BSP_MOD3(K3)
                MHAT(1:3) = MHATB(1:3)+RECIP(7:9)*M3
                MSQ       = MHAT(1)*MHAT(1)+MHAT(2)*MHAT(2)+MHAT(3)*MHAT(3)
                MSQR      = ONE/MSQ

                ETERM = EXP(-FAC*MSQ)*DEN4*MSQR

                IF(QFIN) THEN
                   MVAL=MCUT*SQRT(MSQ)
                   ETERM=ETERM*(ONE-COS(MVAL))
                ENDIF
                !
                Qarray_local(IPT3)   = ETERM * Qarray_local(IPT3)
                Qarray_local(IPT3+1) = ETERM * Qarray_local(IPT3+1)
                !
                IPT3=IPT3+2
             end do
             IPT2=IPT2+NFFT3*2
          end do
          IPT1=IPT1+NFFT3*NFFTDIM1*2
       end do
    else                 ! when computing the gradient and virial contribution.
       !
#if KEY_PMEPLSMA==1
#if KEY_PARALLEL==1
       if(mynod == 0)then     
#endif
          qarray_mm_local(1:2)=zero
#if KEY_PARALLEL==1
       endif                  
#endif
#endif 
       vir(1:6)=zero
       IPT1=1
       do K2Q = 1, MXZSLABS
          K2=K2Q
#if KEY_PARALLEL==1
          IF(MYNOD > 0) K2 = K2Q + MXZSTART(MYNOD)     
#endif

          M2 = K2 - 1
          IF(K2 > NF2) M2 = M2 - NFFT2
          DEN2       = DEN1*BSP_MOD2(K2)
          MHATA(1:3) = RECIP(4:6)*M2

          IPT2=IPT1
          K2s=mod(nfft2-K2+1,nfft2)+1
          do K1 = 1, NF1+1
             K1s=nfft1-K1+2
             M1 = K1 - 1
             IF(K1 > NF1) M1 = M1 - NFFT1
             DEN3       = DEN2*BSP_MOD1(K1)
             MHATB(1:3) = MHATA(1:3)+RECIP(1:3)*M1

             IPT3=IPT2
             IF(K1+K2 == 2) THEN
                K0=2
                IPT3=IPT3+2
             ELSE
                K0=1
             ENDIF

             do K3 = K0,NFFT3
                K3s=mod(nfft3-K3+1,nfft3)+1
                M3 = K3 - 1
                IF(K3 > NF3) M3 = M3 - NFFT3
                DEN4      = DEN3*BSP_MOD3(K3)
                MHAT(1:3) = MHATB(1:3)+RECIP(7:9)*M3
                MSQ       = MHAT(1)*MHAT(1)+MHAT(2)*MHAT(2)+MHAT(3)*MHAT(3)
                MSQR      = ONE/MSQ

                ETERM = EXP(-FAC*MSQ)*DEN4*MSQR
                VTERM = TWO*(FAC+MSQR)
                STRUC2= Qarray_mm_local(ipt3  )*Qarray_local(ipt3  ) +  &
                        Qarray_mm_local(ipt3+1)*Qarray_local(ipt3+1)

                IF(QFIN) THEN
                   MVAL=MCUT*SQRT(MSQ)
                   VCORR=STRUC2*ETERM*SIN(MVAL)*MVAL*MSQR
                   ETERM=ETERM*(ONE-COS(MVAL))
                   VIR(1) = VIR(1) -VCORR*MHAT(1)*MHAT(1)
                   VIR(2) = VIR(2) -VCORR*MHAT(1)*MHAT(2)
                   VIR(3) = VIR(3) -VCORR*MHAT(1)*MHAT(3)
                   VIR(4) = VIR(4) -VCORR*MHAT(2)*MHAT(2)
                   VIR(5) = VIR(5) -VCORR*MHAT(2)*MHAT(3)
                   VIR(6) = VIR(6) -VCORR*MHAT(3)*MHAT(3)
                ENDIF

                ESTR = ETERM * STRUC2
                VIR(1) = VIR(1) + ESTR*(VTERM*MHAT(1)*MHAT(1) - ONE)
                VIR(2) = VIR(2) + ESTR*(VTERM*MHAT(1)*MHAT(2)      )
                VIR(3) = VIR(3) + ESTR*(VTERM*MHAT(1)*MHAT(3)      )
                VIR(4) = VIR(4) + ESTR*(VTERM*MHAT(2)*MHAT(2) - ONE)
                VIR(5) = VIR(5) + ESTR*(VTERM*MHAT(2)*MHAT(3)      )
                VIR(6) = VIR(6) + ESTR*(VTERM*MHAT(3)*MHAT(3) - ONE)

                ! for k1 > 1
                if(k1 > 1)then
                   DENs = DEN1*BSP_MOD3(K3s)*BSP_MOD2(K2s)*BSP_MOD1(K1s)
                   M1s = K1s - 1
                   IF(K1s > NF1) M1s = M1s - NFFT1
                   M2s = K2s - 1
                   IF(K2s > NF2) M2s = M2s - NFFT2
                   M3s = K3s - 1
                   IF(K3s > NF3) M3s = M3s - NFFT3

                   MHATs(1) =RECIP(1)*M1s+RECIP(4)*M2s+RECIP(7)*M3s
                   MHATs(2) =RECIP(2)*M1s+RECIP(5)*M2s+RECIP(8)*M3s
                   MHATs(3) =RECIP(3)*M1s+RECIP(6)*M2s+RECIP(9)*M3s
                   MSQs = MHATs(1)*MHATs(1)+MHATs(2)*MHATs(2)+MHATs(3)*MHATs(3)
                   MSQRs=ONE/MSQs
                   !
                   ETERMs = EXP(-FAC*MSQs)*DENs*MSQRs
                   VTERMs = TWO*(FAC+MSQRs)

                   STRUC2s= Qarray_mm_local(ipt3  )*Qarray_local(ipt3  ) +  &
                            Qarray_mm_local(ipt3+1)*Qarray_local(ipt3+1)

                   ESTRs = ETERMs * STRUC2s
                   VIR(1) = VIR(1) + ESTRs*(VTERMs*MHATs(1)*MHATs(1) - ONE)
                   VIR(2) = VIR(2) + ESTRs*(VTERMs*MHATs(1)*MHATs(2)      )
                   VIR(3) = VIR(3) + ESTRs*(VTERMs*MHATs(1)*MHATs(3)      )
                   VIR(4) = VIR(4) + ESTRs*(VTERMs*MHATs(2)*MHATs(2) - ONE)
                   VIR(5) = VIR(5) + ESTRs*(VTERMs*MHATs(2)*MHATs(3)      )
                   VIR(6) = VIR(6) + ESTRs*(VTERMs*MHATs(3)*MHATs(3) - ONE)
                end if

                !
                Qarray_local(IPT3)   = ETERM * Qarray_local(IPT3)
                Qarray_local(IPT3+1) = ETERM * Qarray_local(IPT3+1)
                !
                IPT3=IPT3+2
             end do
             IPT2=IPT2+NFFT3*2
          end do
          IPT1=IPT1+NFFT3*NFFTDIM1*2
       end do

!!!!      no need to do this, since it is between qm and mm pairs.
!!!!      vir(1:6)=half*vir(1:6)

    end if

    RETURN
  END subroutine convol_fr_space_qm_mm


  !***********************************************************************
  subroutine potential_sumrc_qm_mm(igood, kbot, ktop,numatoms,natqm,       &
                              qarray_local,ewd_potential,qm_atm_grad_comp_local, &
                              recip,volume,ordr,nfftdim1,nfftdim2,nfftdim3, &
                              my_ks_inv,latm,xnsymm, &
                              igmsel_local,mminb2_local,qminb_local)

  !
  ! This routine compute the potential at the QM atom sites applied by all MM atoms.
  !
  ! I have to only assume xnsymm is 1. Other cases, should not supported yet.
  !
  ! Note: parent routine must call with
  ! natqm
  ! IGMSEL (or igmsel_dual, depending on the actual calcualtions.)
  ! MMINB2_dual(*,1 or 2)
  !

  use pme_module
  use pmeutil,only: mxystart,mxyslabs,mxzslabs, &
                    nfft1,nfft2,nfft3, &
                    theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  use number
  use dimens_fcm
  use consta
  use parallel
  use stream
  use qm2_constants, only : FOURPI,TWO_PI
  use qm2_conversions, only : AU_TO_KCAL,BOHRS_TO_A

    implicit none

    integer,intent(in) :: igood, kbot, ktop
    integer,intent(in) :: numatoms,natqm,ordr
    integer,intent(in) :: nfftdim1,nfftdim2,nfftdim3,xnsymm,latm
    real(chm_real)     :: qarray_local(*), ewd_potential(*),qm_atm_grad_comp_local(3,*),volume
    integer,intent(in) :: my_ks_inv(*), &
                          igmsel_local(*),mminb2_local(*),qminb_local(*)

    real(chm_real),intent(in) :: recip(9)

    integer :: igoo,ig,iqm,n
    integer :: I,J,K,KQ,i_keep,j_keep,k_keep,ITH1,ITH2,ITH3,IPT1,IPT2,IPT3
    integer :: rcskip,nfftdimrc                   ! RCFFT addition
    real(chm_real) :: CFACT,Pot,P_tmp1,P_tmp2,fxyz(3),vala(3),val(3),ufact,CFACT2

    rcskip=1
    if(nfft1/2 /= (nfft1+1)/2) CALL WRNDIE(-5,'<grad_sumrc_qm_mm>','fftx dimension not even ')
    nfftdimrc=nfft1+4
    !
    ufact      = AU_TO_KCAL*BOHRS_TO_A
    if(xnsymm == 1)then
       CFACT=one                    ! CCELEC, since unit conversion
       CFACT2=ufact
    else
       CFACT=one/XNSYMM             ! CCELEC/XNSYMM
       CFACT2=ufact/XNSYMM
    end if
    !
    qm_atm_grad_comp_local(1:3,1:natqm)=zero
    loopig: do iqm=1,natqm
        n=qminb_local(iqm)      ! pointer for the index in x/y/z arrays.
        igoo=my_ks_inv(n)
        ig=igoo

#if KEY_PARALLEL==1
        ! skip if igoo .eq. 0
        if(igoo.le.0) cycle loopig      
#endif

        ! skip for non-qm atoms.
        K_keep = INT(FR3(igoo)) - ORDR + 1 + NFFT3
        J_keep = INT(FR2(igoo)) - ORDR + 1 + NFFT2
        I_keep = INT(FR1(igoo)) - ORDR + 1 + NFFT1

        K        = k_keep
        Pot      = ZERO
        fxyz(1:3)= zero
        do ITH3 = 1,ORDR
           K=K+1
           IF(K > NFFT3) K=K-NFFT3
           KQ=K
#if KEY_PARALLEL==1
           if ( K  >=  KBOT .AND. K  <=  KTOP ) then      
              KQ = K - MXYSTART(MYNOD)                    
#endif
              IPT1=(KQ-1)*NFFTDIM2 -1
              J = j_keep
              I = i_keep
              IF(I >= NFFT1) I=I-NFFT1

              P_tmp1 = THETA3(ITH3,ig)   ! NFFT1*NFFT2*NFFT3/volume, it may not be correct!
              vala(1)= NFFT1*THETA3(ITH3,ig)
              vala(2)= NFFT2*THETA3(ITH3,ig)
              vala(3)= NFFT3*DTHETA3(ITH3,igoo)

              do ITH2 = 1,ORDR
                 J=J+1
                 IF(J > NFFT2) J=J-NFFT2

                 P_tmp2 = P_tmp1*THETA2(ITH2,ig)
                 val(1) = vala(1)*THETA2(ITH2,ig)
                 val(2) = vala(2)*DTHETA2(ITH2,igoo)
                 val(3) = vala(3)*THETA2(ITH2,ig)

                 IPT2= rcskip*((IPT1+J)*NFFTDIMrc+I)+1
                 IPT3= IPT2 + rcskip*(NFFT1-I)
                 do ITH1 = 1,ORDR
                    Pot    =Pot+P_tmp2*qarray_local(IPT2)*THETA1(ITH1,ig)
!     write(6,'(4I4,2F12.7,I2)') iqm,ith1,ith2,ith3,qarray_local(IPT2),P_tmp2*THETA1(ITH1,ig),mynod
                    fxyz(1)=fxyz(1)+val(1)*qarray_local(IPT2)*DTHETA1(ITH1,igoo)
                    fxyz(2)=fxyz(2)+val(2)*qarray_local(IPT2)*THETA1(ITH1,ig)
                    fxyz(3)=fxyz(3)+val(3)*qarray_local(IPT2)*THETA1(ITH1,ig)

                    IPT2=IPT2+rcskip
                    IF(IPT2 >= IPT3) IPT2=IPT2-NFFT1*rcskip
                 end do
              end do
#if KEY_PARALLEL==1
           end if      
#endif
        end do
        !
        ewd_potential(iqm)= ewd_potential(iqm) + CFACT*Pot
!     write(6,'(2I4,F12.7,I2)') ig,iqm,CFACT*Pot, mynod
        qm_atm_grad_comp_local(1,iqm)=qm_atm_grad_comp_local(1,iqm)+ &
                                      CFACT2*(recip(1)*fxyz(1)+recip(4)*fxyz(2)+recip(7)*fxyz(3))
        qm_atm_grad_comp_local(2,iqm)=qm_atm_grad_comp_local(2,iqm)+ &
                                      CFACT2*(recip(2)*fxyz(1)+recip(5)*fxyz(2)+recip(8)*fxyz(3))
        qm_atm_grad_comp_local(3,iqm)=qm_atm_grad_comp_local(3,iqm)+ &
                                      CFACT2*(recip(3)*fxyz(1)+recip(6)*fxyz(2)+recip(9)*fxyz(3))
    end do loopig
    RETURN
  END subroutine potential_sumrc_qm_mm
!=================================================================================================!
!========                           PME Section Ends Here                                 ========!
!=================================================================================================!


#endif /* (mainsquatn)*/

SUBROUTINE qm2_ewald_BLANK
  !
  ! dummy routine for compilation
  !
  RETURN
END SUBROUTINE qm2_ewald_BLANK


