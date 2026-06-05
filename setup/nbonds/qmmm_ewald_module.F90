module qmmmewald_module
  use chm_kinds
  use number
  use consta

#if KEY_MNDO97==1 /*mainsquatn*/

  implicit none
  !
  TYPE, PUBLIC :: qmmm_ewald
  ! Store information related with QM/MM-Ewald sum calculations.
  logical            :: ewald_startup        ! .True. if this is the very first MD step and
                                             !        we are doing qmewald
  !integer            :: Ewmode = 1    ! Ewald mode options
  !                                    !     1 : QM/MM-Ewald such that MM in cutoff do
  !                                    !         interact with QM density as regular
  !                                    !         QM/MM potential
  integer            :: Erfmod         ! Mode for Error function evaluation in CHARMM
                                       ! It is just copied here to use later in routines.
  integer            :: kmaxqx,         & ! Maximu K space vector to be summed in x-dir
                        kmaxqy,         & ! Maximu K space vector to be summed in y-dir
                        kmaxqz,         & ! Maximu K space vector to be summed in z-dir
                        ksqmaxq           ! Maximu K space vector square for spherical cutoff in K space
  integer            :: totkq          ! Total number of K space vectors summed
  integer            :: natom          ! Total number of Namtom, copied here by
                                       ! qm_mm convenience

  integer            :: iastrt         ! to be used for parallel calculations.
  integer            :: iafinl         ! numnod=1; iastrt=1, and iafinl=natom
  integer            :: iatotl         !           iatotl=natom
                                       ! numnod=n; iastrt: starting of natom loop.
                                       !           iafinl: ending of natom loop.
                                       !           iatotl: total term in mynod.
  integer            :: nexl_atm       ! Total number of MM excluded atoms from
                                       ! QM-MM interaction, such as MM atoms
                                       ! connected to QM atoms (IGMSEL(i)=5)
  integer, POINTER   :: nexl_index(:) => NULL()       ! Index of pointing MM position in CHARMM main
                                                      ! coordinates for excluded MM atoms from
                                                      ! QM-MM interaction
  real(chm_real)           :: kappa                   ! Kappa in CHARMM for width of gaussians
  real(chm_real)           :: volume                  ! Volume of system
  real(chm_real)           :: Recip(6)                ! Symmetric Reciprocal space Lattice vector
  real(chm_real)           :: ewald_core              ! Ewald potential with QM core charges
                                                      ! - energy in eV.
  real(chm_real), POINTER  :: exl_xyz(:,:)=>NULL()    ! Coordinates for MM atoms excluded from
                                                      ! QM-MM interactions (3,:) in x,y,z,...
  real(chm_real), POINTER  :: exl_chg(:)=>Null()      ! Charges for MM atoms excluded from QM-MM int.
  real(chm_real), POINTER  :: dexl_xyz(:,:)=>NULL()   ! Save gradient contributions from
                                                      ! excluded MM atoms
  !real(chm_real), POINTER  :: scf_mchg(:)=> NULL()    ! Mulliken charge of QM atoms
  !real(chm_real), POINTER  :: scf_mchg_2(:)=>NULL()   ! Mulliken charge of QM atoms for gradient.
  real(chm_real), POINTER  :: kvec(:)=> NULL()        ! Array storing K vectors (totkq long)
  real(chm_real), POINTER  :: ktable(:,:,:)=>NULL()   ! Table for storing complex exp(ik,r[j])
                                                      ! dimension = 6, natom, totkq
                                                      ! 1,x,y = x_cos      2,x,y = x_sin
                                                      ! 3,x,y = y_cos      4,x,y = y_sin
                                                      ! 5,x,y = z_cos      6,x,y = z_sin
  real(chm_real), POINTER  :: qmktable(:,:,:)=>NULL() ! As Ktable but stores the qmatom copies
                                                      ! a linear 1->nquant fashion
  real(chm_real), POINTER  :: structfac_mm(:,:)=>Null() ! Sin and Cos functions in the structure
                                                        ! factor from MM chages; 2,totkq
  real(chm_real), dimension(:),allocatable  :: empot  ! Nquant long, stores the potential at each
                                                ! QM atom due to the MM field.
  real(chm_real), dimension(:,:),allocatable :: eslf ! Stores the self energy of the QM atoms to
                                                ! avoid double counting.
                                                ! Nquant,Nquant matrix
  real(chm_real), dimension(:),allocatable  :: empot_pme  ! Nquant long, stores the PME component of 
                                                      ! potential at each QM atom due to the MM field.
  real(chm_real), dimension(:),allocatable  :: empot_qm_pme ! QM-QM part of PME components 
  real(chm_real), POINTER  :: d_ewald_mm(:,:)=>NULL() ! Gradients on MM atoms due to reciprocal
                                                      ! QM-MM interactions for Virial correction.
                                                      ! (3,natom)
  real(chm_real), POINTER  :: qmqmerfcx_data(:)=>NULL()
                                                ! Stores 1: (-drfc+(one-erfcx)/rij)/rij^2
  real(chm_real), POINTER  :: qmmmerfcx_data(:)=>NULL()
                                                ! Stores 1: (-drfc+(one-erfcx)/rij)/rij^2

  ! copied from qm_main and mm_main types
  real(chm_real),pointer :: rijdata_qmqm(:,:)=>Null()  ! incore data, rij, 1/rij.
  real(chm_real),pointer :: rijdata_qmmm(:,:)=>NUll()  ! incore data, rij, 1/rij.

  !
  END TYPE qmmm_ewald

  ! assign
  TYPE(qmmm_ewald),  save :: qmmm_ewald_r

  contains

  subroutine allocate_deallocate_qmmm_ewald(qm_numat,mm_natom,mm_PMEwald, &
                                            qmmm_ewald_l,qallocate)
  !
  ! allocate/deallocate mm atoms related arrays.
  ! if qallocate == .true. , allocate memory
  !                  false., deallocate memory
  use qm1_info,only : Aass

  implicit none
  integer :: qm_numat,mm_natom
  logical :: mm_PMEwald,qallocate
  TYPE(qmmm_ewald):: qmmm_ewald_l

  integer :: ier=0
  integer :: numat,natom,totkq,iarray_size,nexl_atm
  logical :: noPMEwald

  ! define
  numat      =qm_numat            ! qm_main_l%numat
  natom      =mm_natom            ! mm_main_l%natom
  noPMEwald  =.not.mm_PMEwald     ! .not.mm_main_l%PMEwald
  ! 
  iarray_size=ishft(numat*numat,-1)
  totkq      =qmmm_ewald_l%totkq
  nexl_atm   =MAX(1,qmmm_ewald_l%nexl_atm)  ! either 1 or nexl_atm

  ! deallocate if arrays are associated.
  !if(associated(qmmm_ewald_l%scf_mchg))       deallocate(qmmm_ewald_l%scf_mchg,stat=ier)
  !   if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','scf_mchg')
  !if(associated(qmmm_ewald_l%scf_mchg_2))     deallocate(qmmm_ewald_l%scf_mchg_2,stat=ier)
  !   if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','scf_mchg_2')
  if(associated(qmmm_ewald_l%kvec))           deallocate(qmmm_ewald_l%kvec,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','kvec')
  if(associated(qmmm_ewald_l%ktable))         deallocate(qmmm_ewald_l%ktable,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','ktable')
  if(associated(qmmm_ewald_l%qmktable))       deallocate(qmmm_ewald_l%qmktable,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','qmktable')
  if(associated(qmmm_ewald_l%structfac_mm))   deallocate(qmmm_ewald_l%structfac_mm,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','structfac_mm')
  if(allocated(qmmm_ewald_l%empot))          deallocate(qmmm_ewald_l%empot,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','empot')
  if(allocated(qmmm_ewald_l%eslf))           deallocate(qmmm_ewald_l%eslf,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','eslf')
  if(allocated(qmmm_ewald_l%empot_pme))      deallocate(qmmm_ewald_l%empot_pme,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','empot_pme')
  if(allocated(qmmm_ewald_l%empot_qm_pme))   deallocate(qmmm_ewald_l%empot_qm_pme,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','empot_qm_pme')
  if(associated(qmmm_ewald_l%d_ewald_mm))     deallocate(qmmm_ewald_l%d_ewald_mm,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','d_ewald_mm')
  !
  if(associated(qmmm_ewald_l%nexl_index))     deallocate(qmmm_ewald_l%nexl_index,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','nexl_index')
  if(associated(qmmm_ewald_l%exl_xyz))        deallocate(qmmm_ewald_l%exl_xyz,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','exl_xyz')
  if(associated(qmmm_ewald_l%exl_chg))        deallocate(qmmm_ewald_l%exl_chg,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','exl_chg')
  if(associated(qmmm_ewald_l%dexl_xyz))       deallocate(qmmm_ewald_l%dexl_xyz,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm_ewald','dexl_xyz')

  ! now, allocate memory, only if qallocate==.true.
  if(qallocate) then
     !allocate(qmmm_ewald_l%scf_mchg(numat),stat=ier)
     !   if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','scf_mchg')
     !allocate(qmmm_ewald_l%scf_mchg_2(numat),stat=ier)
     !   if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','scf_mchg_2')
     allocate(qmmm_ewald_l%kvec(totkq),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','kvec')
     allocate(qmmm_ewald_l%qmktable(6,numat,totkq),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','qmktable')
     !
     if(noPMEwald) then
        allocate(qmmm_ewald_l%ktable(6,qmmm_ewald_l%iatotl,totkq),stat=ier)
     else
        allocate(qmmm_ewald_l%ktable(6,1,1),stat=ier)
     end if
     if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','ktable')
     if(noPMEwald) then
        allocate(qmmm_ewald_l%structfac_mm(2,totkq),stat=ier)
     else
        allocate(qmmm_ewald_l%structfac_mm(2,1),stat=ier)
     end if
     if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','structfac_mm')
     !
     allocate(qmmm_ewald_l%empot(numat),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','empot')
     allocate(qmmm_ewald_l%eslf(numat,numat),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','eslf')
     allocate(qmmm_ewald_l%empot_pme(numat),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','empot_pme')
     allocate(qmmm_ewald_l%empot_qm_pme(numat),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','empot_qm_pme')
     allocate(qmmm_ewald_l%d_ewald_mm(3,natom),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','d_ewald_mm')
     !
     ! to exclude MM atoms from QM-MM interactions (igmsel(i)=5).
     allocate(qmmm_ewald_l%nexl_index(nexl_atm),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','nexl_index')
     allocate(qmmm_ewald_l%exl_xyz(3,nexl_atm),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','exl_xyz')
     allocate(qmmm_ewald_l%exl_chg(nexl_atm),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','exl_chg')
     allocate(qmmm_ewald_l%dexl_xyz(3,nexl_atm),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm_ewald','dexl_xyz')
  end if
  return
  end subroutine allocate_deallocate_qmmm_ewald


  SUBROUTINE qm_ewald_setup(kmaxqx,kmaxqy,kmaxqz,ksqmaxq,totkq,Qcheck)
  !
  ! Initial setup for qm_ewald, - should be called only once per run.
  ! This routine calculate the total K space vectors and other related
  ! values. These values should be constant during a QM/MM run, unless
  ! re-do the setup.
  !
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
  do kx = 0, kmaxqx
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
  end do

  !
  if (totkq .LE. 0) then
     write(6,*)'qm_ewald_setup> ','Invalie number of Total Kspace vectors.'
     write(6,*)'TOTKQ needs to be greater than 0, but now is ',totkq

     qcheck=.FALSE.
     return
  end if

  return
  END SUBROUTINE qm_ewald_setup


  SUBROUTINE qm_ewald_calc_kvec(kappa,volume,recip,totkq,      &
                                ksqmaxq,kmaxqx,kmaxqy,kmaxqz,  &
                                QNoPMEwald,Qcheck)
  !
  ! Setup wave-vector arrays for K-space Ewald summation.
  ! 
  use qm1_constant
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none

  ! Passed in
  real(chm_real),  intent(in)    :: kappa,volume,recip(6)
  !!!real(chm_real),  intent(out)   :: kvec(*)
  integer, intent(in)    :: totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz
  logical, intent(inout) :: QNoPMEwald,Qcheck

  ! Local variables
  integer                :: loop_count,kx,ky,kz,ksq,ksy,ksz
  real(chm_real)         :: rkx(3),rky(3),rkz(3),rksq
  real(chm_real)         :: mkv(3)
  real(chm_real)         :: beta,vfact
  real(chm_real),save    :: old_kappa=zero, &
                            old_volume=zero
  integer,save           :: old_kmaxqx=0,   &
                            old_kmaxqy=0,   &
                            old_kmaxqz=0,   &
                            old_ksqmaxq=0
#if KEY_PARALLEL==1
  integer :: mmynod, nnumnod

  mmynod  = mynod
  nnumnod = numnod
#endif

  if(kappa .eq.old_kappa  .and. volume.eq.old_volume .and. kmaxqx.eq.old_kmaxqx   .and. &
     kmaxqy.eq.old_kmaxqy .and. kmaxqz.eq.old_kmaxqz .and. ksqmaxq.eq.old_ksqmaxq) then

     ! check for quick return, if there are no changes since last call.
     return 
  else
     ! something has changed since last call, so recompute entire kvec.
     old_kappa  = kappa
     old_volume = volume
     old_kmaxqx = kmaxqx
     old_kmaxqy = kmaxqy
     old_kmaxqz = kmaxqz
     old_ksqmaxq= ksqmaxq

     ! we need to recompute kvec.
     beta = PI*PI/(kappa*kappa)
     vfact= INVPI/volume

     loop_count = 0
     if(QNoPMEwald) then
        ! regular Ewald case.
        do kx = 0, kmaxqx
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
                    qmmm_ewald_r%kvec(loop_count) = vfact*exp(-beta*rksq)/rksq

                 end if
              end do
           end do
        end do
     else
        ! QM/MM-PME case.
        do kx = 0, kmaxqx
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
#if KEY_PARALLEL==1
                    if(mmynod == mod(loop_count-1,nnumnod)) then
#endif
                       mkv(1:3)         = rkx(1:3) + rky(1:3) + rkz(1:3)
                       rksq             = mkv(1)**2 + mkv(2)**2 + mkv(3)**2
                       qmmm_ewald_r%kvec(loop_count) = vfact*exp(-beta*rksq)/rksq
#if KEY_PARALLEL==1
                    end if
#endif
                 end if
              end do
           end do
        end do
     end if

     ! Check if loop_count ends up equalling totkq. If not, then wrong in the code.
     if (loop_count .ne. totkq) then
        write(6,*)'qm_ewald_calc_kvec> Invalid number of K vectors.'
        write(6,*)'It should be ',totkq,' but now it is ',loop_count

        qcheck =.FALSE.
        return
     end if
  end if

  return
  END SUBROUTINE qm_ewald_calc_kvec


  SUBROUTINE qm_ewald_calc_ktable(natom,numat,qminb, &
                                  iastrt,iafinl,iatotl,itotal,itotkq,  &
                                  totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz,  &
                                  X,Y,Z,Recip,                         & 
                                  QNoPMEwald,Qcheck)
  !
  ! Setup K-table array for K-space Ewald summation.
  !
  use qm1_constant
  use parallel  !##PARALLEL

  implicit none

  ! Passed in
  integer, intent(in)       :: Natom,numat
  integer, intent(in)       :: qminb(numat),iastrt,iafinl,iatotl,itotal,itotkq
  integer, intent(in)       :: totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz
  real(chm_real),intent(in) :: X(*),Y(*),Z(*),recip(6)
  !!!real(chm_real),intent(out):: Ktable(6,itotal,itotkq)      ! Ktable(6,iatotl,totkq); iatotl=natom
  !!!real(chm_real),intent(out):: qmKtable(6,numat,totkq)
  logical, intent(in)       :: QNoPMEwald
  logical, intent(inout)    :: Qcheck
 
  ! Local variables
  integer        :: loop_count,kx,ky,kz,ksq,ksy,ksz,i,qmid,inner_loop
  real(chm_real) :: rkx(3),rky(3),rkz(3),mkv(3),xyz(3),picoef
  !
#if KEY_PARALLEL==1
  integer :: mmynod, nnumnod

  mmynod  = mynod     !##PARALLEL
  nnumnod = numnod    !##PARALLEL
#endif

  !
  ! Ktable and qmKtable were initialized in the parent subroutine.

  loop_count = 0
  loopkx: do kx = 0, kmaxqx
     if (kx .eq. 0) then 
         ksy = 0 
     else
         ksy = -kmaxqy
     end if
     picoef = TWOPI  * REAL(kx)
     rkx(1) = picoef * recip(1)
     rkx(2) = picoef * recip(2)
     rkx(3) = picoef * recip(4)
 
     loopky: do ky = ksy, kmaxqy
        if (kx .eq. 0 .and. ky .eq. 0) then
           ksz = 1
        else
           ksz = -kmaxqz
        end if
        picoef = TWOPI  * REAL(ky)
        rky(1) = picoef * recip(2)
        rky(2) = picoef * recip(3)
        rky(3) = picoef * recip(5)

        loopkz: do kz = ksz, kmaxqz
           picoef = TWOPI  * REAL(kz)
           rkz(1) = picoef * recip(4)
           rkz(2) = picoef * recip(5)
           rkz(3) = picoef * recip(6)
           ksq    = kx*kx + ky*ky + kz*kz

           if (ksq .le. ksqmaxq .and. ksq .ne. 0) then

              loop_count = loop_count + 1
              mkv(1:3)   = rkx(1:3) + rky(1:3) + rkz(1:3)

              ! for mm atoms.
              if(QNoPMEwald) then
                 inner_loop = 0
                 do i = iastrt,iafinl             ! 1, Natom
                    inner_loop = inner_loop+1
                    xyz(1) = mkv(1)*X(i)
                    xyz(2) = mkv(2)*Y(i)
                    xyz(3) = mkv(3)*Z(i)

                    ! Cache the values for doing a vectored cos ( and sin(x)=cos(x-(pi/2)) )
                    !   X coordinates
                    qmmm_ewald_r%Ktable(1,inner_loop,loop_count) = xyz(1)
                    qmmm_ewald_r%Ktable(2,inner_loop,loop_count) = xyz(1) - HALFPI

                    !   Y coordinates
                    qmmm_ewald_r%Ktable(3,inner_loop,loop_count) = xyz(2)
                    qmmm_ewald_r%Ktable(4,inner_loop,loop_count) = xyz(2) - HALFPI

                    !   Z coordinates
                    qmmm_ewald_r%Ktable(5,inner_loop,loop_count) = xyz(3)
                    qmmm_ewald_r%Ktable(6,inner_loop,loop_count) = xyz(3) - HALFPI

                    qmmm_ewald_r%Ktable(1:6,inner_loop,loop_count)=cos(qmmm_ewald_r%Ktable(1:6,inner_loop,loop_count))
                 end do
                 ! Do a vectored cosine -> since we subtracted pi/2 from
                 ! every other x,y,z value so we will end up with cos,sin,cos,sin...
                 !call vdcos(6,iatotl,qmmm_ewald_r%Ktable(1,1,loop_count), &
                 !                    qmmm_ewald_r%Ktable(1,1,loop_count))  ! iatotl=Natom
              end if

              ! Cache the qm atom values for use later so we can access them linearly in memory
#if KEY_PARALLEL==1
              ! since in parallel mode, Ktable is only constructed for a part of system,
              ! so if we compute for a part for qm atoms, it will bd broadcasted. 
              if(mmynod .ne. mod(loop_count-1,nnumnod)) cycle loopkz
#endif
              do i = 1, numat
                 qmid = qminb(i)
                 xyz(1) = mkv(1)*X(qmid)
                 xyz(2) = mkv(2)*Y(qmid)
                 xyz(3) = mkv(3)*Z(qmid)
                 qmmm_ewald_r%qmKtable(1,i,loop_count) = xyz(1)
                 qmmm_ewald_r%qmKtable(2,i,loop_count) = xyz(1)-HALFPI
                 qmmm_ewald_r%qmKtable(3,i,loop_count) = xyz(2)
                 qmmm_ewald_r%qmKtable(4,i,loop_count) = xyz(2)-HALFPI
                 qmmm_ewald_r%qmKtable(5,i,loop_count) = xyz(3)
                 qmmm_ewald_r%qmKtable(6,i,loop_count) = xyz(3)-HALFPI

                 qmmm_ewald_r%qmKtable(1:6,i,loop_count)=cos(qmmm_ewald_r%qmKtable(1:6,i,loop_count))
              end do
              !call vdcos(6,numat,qmmm_ewald_r%qmKtable(1,1,loop_count), &
              !                   qmmm_ewald_r%qmKtable(1,1,loop_count))
           end if
        end do loopkz                              ! kz = ksz, kmaxqz
     end do    loopky                              ! ky = ksy, kmaxqy
  end do       loopkx                              ! kx = 0, kmaxqx

  ! just in case. will check later for .not.QNoPMEwald case.
#if KEY_PARALLEL==1
  if(nnumnod>1.and.QNoPMEwald) &
     call gcombr(qmmm_ewald_r%qmKtable,6*numat*totkq)
#endif
                               
  return

  contains
     subroutine vdcos(ni,nj,x,y)
     !
     ! Vectorize Cosine routine
     !
     use chm_kinds

     implicit none
     ! 
     integer        :: ni,nj
     real(chm_real) :: x(ni,nj), y(ni,nj)

     y(1:ni,1:nj) = cos(x(1:ni,1:nj))
     return
     end subroutine Vdcos

  END SUBROUTINE qm_ewald_calc_ktable


  SUBROUTINE qm_ewald_mm_pot(natom,numat,numatm,                         &
                             iastrt,iafinl,iatotl,itotal,itotkq,         &
                             totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz,         &
                             mm_coord,mm_chrgs,qm_coords,                &
                             empot,QNoPMEwald) 
  !
  ! Compute the potential at the QM atom position due to 
  ! the ewald sum of the MM atoms.
  ! Called once per each MD step, before doing the SCF calculations.
  !
  use qm1_constant
  use psf, only : CG
  use erfcd_mod,only: erfcd
  use qm1_info, only : mm_main_r    ! for incore part.
  use nbndqm_mod, only: map_qmatom_to_group,map_mmatom_to_group
#if KEY_PARALLEL==1
  use parallel
#endif
  !
  implicit none

  ! Passed in
  integer, intent(in)    :: natom,numat,numatm
  integer, intent(in)    :: totkq,iatotl,iastrt,iafinl,ksqmaxq,itotal,itotkq
  integer, intent(in)    :: kmaxqx,kmaxqy,kmaxqz

  real(chm_real),intent(in)   :: mm_coord(3,natom),mm_chrgs(natom),qm_coords(3,numat)
  real(chm_real),intent(inout):: empot(numat)
  !!!real(chm_real),intent(in)   :: kvec(totkq),ktable(6,itotal,itotkq), & ! ktable(6,iatotl,totkq),
  !!!                               qmktable(6,numat,totkq)                ! iatotl=natom
  !!!real(chm_real),intent(inout):: structfac_mm(2,itotkq)

  logical, intent(in)    :: QNoPMEwald

  ! Local variables
  integer                :: i, j, loop_count,inner_loop,irs_qm,irs_mm
  integer                :: kx, ky, kz, ksy, ksz, ksq, Erfmod_local, icnt_mm
  real(chm_real)         :: erfcx, drfc, empottmp, kappa_local, rtmp
  real(chm_real)         :: vec(3), r2, oneRIJ, RIJ, oneRIJ2
  real(chm_real)         :: xyz_cos(3),xyz_sin(3)
  real(chm_real)         :: ktgs(8), ksum, mmchg, sfact,rtmp_local(2),sin_sum,cos_sum
  integer  :: mstart,mstop
  integer  :: ido_switching
#if KEY_PARALLEL==1
  integer  :: ISTRT_CHECK                 ! for external function
#endif

  !
  ! initialization
  mstart = 1
  mstop  = numatm
  qmmm_ewald_r%structfac_mm(1:2,1:itotkq)=zero     ! initialization
  !
#if KEY_PARALLEL==1
  if(numnod>1) mstart = ISTRT_CHECK(mstop,numatm)
#endif

  !1) Calculate Real space potential at QM atom position:
  !   compute the distance between QM-MM paris
  Erfmod_local= qmmm_ewald_r%Erfmod
  kappa_local = qmmm_ewald_r%kappa
  if(mm_main_r%rij_mm_incore) then
     icnt_mm = 0
     do i = 1, numat
        empottmp = zero
        if(mm_main_r%q_cut_by_group .or. mm_main_r%q_switch) irs_qm = map_qmatom_to_group(i)
        do j = mstart,mstop
           icnt_mm  = icnt_mm + 1
           Rij      = qmmm_ewald_r%rijdata_qmmm(1,icnt_mm)  ! rij value
           oneRIJ   = qmmm_ewald_r%rijdata_qmmm(2,icnt_mm)  ! one/rij value.

           ! group-by-group-based cutoff case, skip the pair if its distance is longer than cutoff.
           ! otherwise (default group-based case), include all mm atoms (default).
           if(mm_main_r%q_cut_by_group) then
              irs_mm = map_mmatom_to_group(j)
              if(.not.mm_main_r%q_mmgrp_qmgrp_cut(irs_mm,irs_qm)) cycle
           else if(mm_main_r%q_switch) then
              irs_mm = map_mmatom_to_group(j)
           end if

           Call ERFCD(Rij,kappa_local,erfcx,drfc,Erfmod_local)
           !
           ! save erfcd data for gradient: (-drfc+(one-erfcx)/rij)/rij^2 value.
           rtmp     = (one-erfcx)*oneRIJ
           oneRIJ2  = oneRIJ*oneRIJ
           qmmm_ewald_r%qmmmerfcx_data(icnt_mm) = (-drfc+rtmp)*oneRIJ2  
           !

           if(mm_main_r%q_diag_coulomb) then
              rtmp = rtmp - oneRIJ  ! full range.
           else
              ! contribution by switching function. in fact, (1-Sw(rij))*c_j/rij
              if(mm_main_r%q_switch) then
                 ido_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
                 ! apply switching function contribution
                 if(ido_switching>0) then
                    ! see below for the negative sign when adding to empottmp
                    rtmp = rtmp - (one-mm_main_r%sw_val(ido_switching))*oneRIJ
                 end if
              end if
           end if

           empottmp =  empottmp - mm_chrgs(j)*rtmp   !-(one-erfcx)*oneRIJ or
                                                     !-(one-erfcx)*oneRIJ + oneRIJ or
                                                     !-(one-erfcx)*oneRIJ + (1-Sw(rij)*oneRIJ 
        end do
        empot(i) = empottmp
     end do
  else
     do i = 1, numat
        empottmp = zero
        if(mm_main_r%q_cut_by_group .or. mm_main_r%q_switch) irs_qm = map_qmatom_to_group(i)
        do j = mstart,mstop
           vec(1:3) = qm_coords(1:3,i)-mm_coord(1:3,j)
           r2       = vec(1)**2 + vec(2)**2 + vec(3)**2

           ! group-by-group-based cutoff case, skip the pair if its distance is longer than cutoff.
           ! otherwise (default group-based case), include all mm atoms (default).
           if(mm_main_r%q_cut_by_group) then
              irs_mm = map_mmatom_to_group(j)
              if(.not.mm_main_r%q_mmgrp_qmgrp_cut(irs_mm,irs_qm)) cycle
           else if(mm_main_r%q_switch) then
              irs_mm = map_mmatom_to_group(j)
           end if

           oneRIJ   = one/sqrt(r2)
           RIJ      = r2*oneRIJ                         ! one/oneRIJ
           Call ERFCD(RIJ,kappa_local,erfcx,drfc,Erfmod_local)

           !
           rtmp     = (one-erfcx)*oneRIJ
           if(mm_main_r%q_diag_coulomb) then
              rtmp = rtmp - oneRIJ  ! full range.
           else
              ! contribution by switching function. in fact, (1-Sw(rij))*c_j/rij
              if(mm_main_r%q_switch) then
                 ido_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
                 ! apply switching function contribution
                 if(ido_switching>0) then
                    ! see below for the negative sign when adding to empottmp
                    rtmp = rtmp - (one-mm_main_r%sw_val(ido_switching))*oneRIJ
                 end if
              end if
           end if

           empottmp =  empottmp - mm_chrgs(j)*rtmp
        end do
        empot(i) = empottmp
     end do
  end if
  if(.not.QNoPMEwald)  return   ! as the following only be done with regular Ewald.

  !2) Calculate K space potential at QM atom position
  if(QNoPMEwald) then       ! only has to do when QNoPMEwald=.true.
    loop_count = 0

    do kx =0, kmaxqx
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
                ! MM atom
                do j = iastrt,iafinl              ! 1, natom
                   ! loop throught all atoms, as mmchg will be zero for QM atoms:
                   inner_loop = inner_loop + 1
                   xyz_cos(1) = qmmm_ewald_r%ktable(1,inner_loop,loop_count)
                   xyz_sin(1) = qmmm_ewald_r%ktable(2,inner_loop,loop_count)
                   xyz_cos(2) = qmmm_ewald_r%ktable(3,inner_loop,loop_count)
                   xyz_sin(2) = qmmm_ewald_r%ktable(4,inner_loop,loop_count)
                   xyz_cos(3) = qmmm_ewald_r%ktable(5,inner_loop,loop_count)
                   xyz_sin(3) = qmmm_ewald_r%ktable(6,inner_loop,loop_count)

                   !mmchg   = CG(j)
                   ktgs(1) = ktgs(1) + cg(j)*xyz_cos(1)*xyz_cos(2)*xyz_cos(3)
                   ktgs(2) = ktgs(2) + cg(j)*xyz_sin(1)*xyz_sin(2)*xyz_sin(3)
                   ktgs(3) = ktgs(3) + cg(j)*xyz_cos(1)*xyz_cos(2)*xyz_sin(3)
                   ktgs(4) = ktgs(4) + cg(j)*xyz_cos(1)*xyz_sin(2)*xyz_cos(3)
                   ktgs(5) = ktgs(5) + cg(j)*xyz_cos(1)*xyz_sin(2)*xyz_sin(3)
                   ktgs(6) = ktgs(6) + cg(j)*xyz_sin(1)*xyz_sin(2)*xyz_cos(3)
                   ktgs(7) = ktgs(7) + cg(j)*xyz_sin(1)*xyz_cos(2)*xyz_sin(3)
                   ktgs(8) = ktgs(8) + cg(j)*xyz_sin(1)*xyz_cos(2)*xyz_cos(3)
  ! do virial part: compute only the structure factor here.
  !                 this will be used later in the routine computing gradient
  !                 to compute the ewvirial.
                   rtmp_local(1)=xyz_cos(3)*xyz_cos(2)-xyz_sin(2)*xyz_sin(3)
                   rtmp_local(2)=xyz_cos(3)*xyz_sin(2)+xyz_cos(2)*xyz_sin(3)
                   sin_sum=sin_sum + cg(j)*(xyz_cos(1)*rtmp_local(1)-xyz_sin(1)*rtmp_local(2))
                   cos_sum=cos_sum + cg(j)*(xyz_sin(1)*rtmp_local(1)+xyz_cos(1)*rtmp_local(2))
                end do

  ! do virial part
  ! now, note that it is not complete when parallel, since it is sum over
  ! only iastrt,iafinl. Thus, in the end, it will be combined from all nodes.
                qmmm_ewald_r%structfac_mm(1,loop_count)=sin_sum
                qmmm_ewald_r%structfac_mm(2,loop_count)=cos_sum

  ! Now loop over quantum atoms: this has to be done for qm atoms.
  !                              so, when parallel, all qmktable is needed.
                do j = 1, numat
                   ksum = zero
                   xyz_cos(1) = qmmm_ewald_r%qmktable(1,j,loop_count)
                   xyz_sin(1) = qmmm_ewald_r%qmktable(2,j,loop_count)
                   xyz_cos(2) = qmmm_ewald_r%qmktable(3,j,loop_count)
                   xyz_sin(2) = qmmm_ewald_r%qmktable(4,j,loop_count)
                   xyz_cos(3) = qmmm_ewald_r%qmktable(5,j,loop_count)
                   xyz_sin(3) = qmmm_ewald_r%qmktable(6,j,loop_count)

                   ksum = ksum + xyz_cos(1)*xyz_cos(2)*xyz_cos(3)*ktgs(1)
                   ksum = ksum + xyz_sin(1)*xyz_sin(2)*xyz_sin(3)*ktgs(2)
                   ksum = ksum + xyz_cos(1)*xyz_cos(2)*xyz_sin(3)*ktgs(3)
                   ksum = ksum + xyz_cos(1)*xyz_sin(2)*xyz_cos(3)*ktgs(4)
                   ksum = ksum + xyz_cos(1)*xyz_sin(2)*xyz_sin(3)*ktgs(5)
                   ksum = ksum + xyz_sin(1)*xyz_sin(2)*xyz_cos(3)*ktgs(6)
                   ksum = ksum + xyz_sin(1)*xyz_cos(2)*xyz_sin(3)*ktgs(7)
                   ksum = ksum + xyz_sin(1)*xyz_cos(2)*xyz_cos(3)*ktgs(8)

                   empot(j) = empot(j) + sfact*ksum*qmmm_ewald_r%kvec(loop_count)
                end do
             end if

          end do
       end do
    end do
    ! now, distribute the structure_factor to collect all MM atom information.
#if KEY_PARALLEL==1
    if(numnod.gt.1) call GCOMBR(qmmm_ewald_r%structfac_mm,2*itotkq)
#endif
  end if       ! QNoPMEwald=.true.

  ! this is done in the subroutine qmmm_Ewald_setup_and_potential.
  ! now, distribute empot to collect all MM atom information (for qm/mm-ewald version).
  ! while for PMEwald case, empot is communicated in qm_pme_mm_pot routine.
  !!#if KEY_PARALLEL==1
  !if(numnod.gt.1.and.QNoPMEwald) call GCOMB(empot,numat) 
  !!#endif

  return
  END SUBROUTINE qm_ewald_mm_pot


  SUBROUTINE qm_ewald_mm_pot_exl(numat,qm_coords,empot)
  !
  ! Compute the potential at the QM atom position due to the ewald 
  ! sum of the MM atoms in the exclusion list.
  ! Called once per each MD step, before doing the SCF calculations.
  !
  use erfcd_mod,only: erfcd
#if KEY_PARALLEL==1
  use parallel,only: mynod
#endif

  implicit none

  ! Passed in
  integer, intent(in)    :: numat

  real(chm_real),intent(in)  :: qm_coords(3,numat)
  real(chm_real),intent(out) :: empot(numat)

  ! Local variables
  integer        :: i, j, Erfmod_local
  real(chm_real) :: erfcx,drfc,empottmp,vec(3),r2,oneRIJ,RIJ,kappa_local

  ! Calculate Real space potential at QM atom position from
  ! MM atoms excluded from QM-MM non-bonded interactions.

  ! for parallel: only the master node run this.

  if (qmmm_ewald_r%nexl_atm .gt. 0) then
     Erfmod_local = qmmm_ewald_r%Erfmod
     kappa_local  = qmmm_ewald_r%kappa
#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        do i = 1, numat
           empottmp = zero
           do j = 1, qmmm_ewald_r%nexl_atm
              vec(1:3) = qm_coords(1:3,i)-qmmm_ewald_r%exl_xyz(1:3,j)
              r2       = vec(1)**2 + vec(2)**2 + vec(3)**2
              oneRIJ   = one/sqrt(r2)
              RIJ      = r2*oneRIJ                         ! one/oneRIJ
              Call ERFCD(RIJ,kappa_local,erfcx,drfc,Erfmod_local)

              empottmp = empottmp - qmmm_ewald_r%exl_chg(j)*(one-erfcx)*oneRIJ
           end do
           empot(i) = empot(i) + empottmp
        end do
#if KEY_PARALLEL==1
     end if
#endif
  end if

  return
  END SUBROUTINE qm_ewald_mm_pot_exl


  SUBROUTINE qm_ewald_qm_pot(natom,numat,totkq,   &
                             ksqmaxq,kmaxqx,kmaxqy,kmaxqz,   &
                             qm_coords,eslf)
  !
  ! Compute the potential at the QM atom position due to the ewald 
  ! sum of the QM image atoms.
  ! Called once per each MD step, before doing the SCF calculations.
  ! The potential saved as numat x numat matrix, thus later it only
  ! needs to do matrix multiplication using Mulliken charges from QM 
  ! atoms
  !
  use erfcd_mod,only: erfcd
  use qm1_constant
  use qm1_info, only : qm_control_r,qm_main_r
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none

  ! Passed in
  integer, intent(in)   :: natom,numat
  integer, intent(in)   :: totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz

  real(chm_real), intent(in) :: qm_coords(3,numat)
  !!!real(chm_real), intent(in) :: kvec(totkq),qmktable(6,numat,totkq)
  real(chm_real), intent(out):: eslf(numat,numat)

  ! Local variables
  integer :: i, j, loop_count,array_size,Erfmod_local,icnt_qm
  integer :: kx, ky, kz, ksy, ksz, ksq
  integer :: iminus, iplus, offset
  real(chm_real) :: esfact,qmixyz(3),erfcx,drfc,empottmp,vec(3),r2,oneRIJ,RIJ,oneRIJ2, &
                    xyz_cos(3),xyz_sin(3),ktgs(3),ksum,sfact,kappa_local
  !
  real(chm_real), parameter :: SQRTPI=1.77245385090551602729816748334d0
  real(chm_real), parameter :: INVSQRTPI=1.0d0/SQRTPI
  !
  integer :: mmynod,nnumnod,icnt 

#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
#else
  mmynod  = 0
  nnumnod = 1
#endif

  ! Initialization
  eslf = zero
  Erfmod_local = qmmm_ewald_r%erfmod
  kappa_local  = qmmm_ewald_r%kappa

  !1) Self energy term
  esfact = -two*kappa_local*INVSQRTPI
#if KEY_PARALLEL==1
  if(mmynod+1 .le. numat) then 
#endif
     ! if not parallel, mmynod=0,nnumnod=1
     do i = mmynod+1, numat, nnumnod
       eslf(i,i) = esfact
     end do
#if KEY_PARALLEL==1
  end if 
#endif

  !2) Real space potential
#if KEY_PARALLEL==1
  icnt = 0 
#endif
  if(qm_main_r%rij_qm_incore) then
     icnt_qm = 0
     loopi1: do i = 2, numat
        loopj1: do j = 1, i-1
#if KEY_PARALLEL==1
           icnt     = icnt + 1
           if(mmynod == mod(icnt-1,nnumnod)) then
#endif
              icnt_qm  = icnt_qm + 1 ! the counter used.

              Rij      = qmmm_ewald_r%rijdata_qmqm(1,icnt_qm)       ! rij value
              oneRIJ   = qmmm_ewald_r%rijdata_qmqm(2,icnt_qm)       ! one/rij value.
              Call ERFCD(RIJ,kappa_local,erfcx,drfc,Erfmod_local)
              !
              empottmp  = (one-erfcx)*oneRIJ
              eslf(j,i) = eslf(j,i) - empottmp
              eslf(i,j) = eslf(i,j) - empottmp
              !
              ! save erfcd data for gradient: (-drfc+(one-erfcx)/rij)/rij^2 value.
              oneRIJ2   = oneRIJ*oneRIJ
              qmmm_ewald_r%qmqmerfcx_data(icnt_qm) = (-drfc+empottmp)*oneRIJ2
#if KEY_PARALLEL==1
           end if
#endif
        end do loopj1
     end do    loopi1
  else
     loopii: do i = 2, numat
        qmixyz(1:3) = qm_coords(1:3,i)
        loopjj: do j = 1, i-1
#if KEY_PARALLEL==1
           icnt     = icnt + 1
           if(mmynod == mod(icnt-1,nnumnod)) then
#endif
              vec(1:3) = qmixyz(1:3)-qm_coords(1:3,j)
              r2       = vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)
              oneRIJ   = one / sqrt(r2)
              RIJ      = r2*oneRIJ                      ! one/oneRIJ
 
              Call ERFCD(RIJ,kappa_local,erfcx,drfc,Erfmod_local)

              empottmp  = (one-erfcx)*oneRIJ
              eslf(j,i) = eslf(j,i) - empottmp
              eslf(i,j) = eslf(i,j) - empottmp
#if KEY_PARALLEL==1
           end if
#endif
        end do loopjj
     end do    loopii
  end if

  if(.not.( qm_control_r%q_do_cpmd_pme )) then
  !3) K space potentail between QM atoms
  loop_count = 0
  loopkx: do kx = 0, kmaxqx
    if ( kx .eq. 0 ) then
       ksy = 0
    else
       ksy = -kmaxqy
    end if
    loopky: do ky = ksy, kmaxqy
       if ( kx .eq. 0 .and. ky .eq. 0 ) then
          ksz = 1
       else
          ksz = -kmaxqz
       end if
       loopkz: do kz = ksz, kmaxqz
          sfact= two
          ksq  = kx*kx + ky*ky + kz*kz

          if ( ksq .le. ksqmaxq .and. ksq .ne. 0 ) then
             loop_count = loop_count+1
#if KEY_PARALLEL==1
             if(mmynod == mod(loop_count-1,nnumnod)) then
#endif
                ktgs(1:3)  = zero
                do i = 1, numat
                   xyz_cos(1) = qmmm_ewald_r%qmktable(1,i,loop_count)
                   xyz_sin(1) = qmmm_ewald_r%qmktable(2,i,loop_count)
                   xyz_cos(2) = qmmm_ewald_r%qmktable(3,i,loop_count)
                   xyz_sin(2) = qmmm_ewald_r%qmktable(4,i,loop_count)
                   xyz_cos(3) = qmmm_ewald_r%qmktable(5,i,loop_count)
                   xyz_sin(3) = qmmm_ewald_r%qmktable(6,i,loop_count)

                   do j = 1, i-1
                      ktgs(1) = xyz_cos(1)*qmmm_ewald_r%qmktable(1,j,loop_count) &
                               +xyz_sin(1)*qmmm_ewald_r%qmktable(2,j,loop_count)
                      ktgs(2) = xyz_cos(2)*qmmm_ewald_r%qmktable(3,j,loop_count) &
                               +xyz_sin(2)*qmmm_ewald_r%qmktable(4,j,loop_count)
                      ktgs(3) = xyz_cos(3)*qmmm_ewald_r%qmktable(5,j,loop_count) &
                               +xyz_sin(3)*qmmm_ewald_r%qmktable(6,j,loop_count)
                      ksum    = sfact*ktgs(1)*ktgs(2)*ktgs(3)*qmmm_ewald_r%kvec(loop_count)

                      eslf(j,i) = eslf(j,i) + ksum
                      eslf(i,j) = eslf(i,j) + ksum
                   end do
                   ! for i=j
                   ktgs(1) = xyz_cos(1)*qmmm_ewald_r%qmktable(1,i,loop_count) &
                            +xyz_sin(1)*qmmm_ewald_r%qmktable(2,i,loop_count)
                   ktgs(2) = xyz_cos(2)*qmmm_ewald_r%qmktable(3,i,loop_count) &
                            +xyz_sin(2)*qmmm_ewald_r%qmktable(4,i,loop_count)
                   ktgs(3) = xyz_cos(3)*qmmm_ewald_r%qmktable(5,i,loop_count) &
                            +xyz_sin(3)*qmmm_ewald_r%qmktable(6,i,loop_count)
                   ksum    = sfact*ktgs(1)*ktgs(2)*ktgs(3)*qmmm_ewald_r%kvec(loop_count)
                   eslf(i,i) = eslf(i,i) + ksum
                end do
#if KEY_PARALLEL==1
             end if
#endif
          end if

       end do loopkz
    end do    loopky
  end do      loopkx
  end if

! This is done in scf_energy routine.

  return
  END SUBROUTINE qm_ewald_qm_pot


  SUBROUTINE qm_ewald_prepare_fock(numat,empot_all,empot_local,qm_charges)
  !
  ! 1) Compute Eslf contribution + Empot contribution at QM atom
  !    to preapare correction for Fock matrix in diagonal elements
  ! 2) also copy empot to empot_local
  !
  use qm1_info, only       : qm_control_r
  use qm1_constant
#if KEY_PARALLEL==1
  use parallel,only : mynod,numnod,MAXNODE
#endif

  implicit none

  ! Passed in
  integer, intent(in)    :: numat
  real(chm_real),intent(inout):: empot_all(numat),empot_local(numat)
  real(chm_real),intent(in)   :: qm_charges(numat)
  !!!real(chm_real),intent(in)   :: empot(numat), eslf(numat,numat)

  ! Local variables
  integer        :: i
  real(chm_real) :: ewdpot
  !!!real(chm_real),parameter :: ev_a0 = EV*A0 ! convert (electrons/angstrom) to (eV/Bohr)
  integer, save :: old_N = 0
  integer, save :: mstart,mstop
#if KEY_PARALLEL==1
  integer, save :: JPARPT_local(0:MAXNODE)
#endif


  ! parallelization is synchronized with calc_mulliken routine and broad casting.
  if(old_N .ne. numat) then
     old_N = numat
     mstart= 1
     mstop = numat
     !
#if KEY_PARALLEL==1
     JPARPT_local(0)=0
     do i=1,numnod
        JPARPT_local(i)= numat*i/numnod ! for linear vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)
#endif
  end if

  ! compute Empot+Eslf contribution for the diagonal elements of the fock matrix
  ! conversion to ev_a0 is done in the qm_ewald_add_fock routine.
  if(qm_control_r%q_do_cpmd_pme) then
     do i = mstart,mstop
        ewdpot = DOT_PRODUCT(qmmm_ewald_r%eslf(1:numat,i),qm_charges(1:numat))
        empot_all(i) = (qmmm_ewald_r%empot(i) + qmmm_ewald_r%empot_qm_pme(i) + ewdpot) !!!!* ev_a0
     end do
  else
     do i = mstart,mstop
        ewdpot = DOT_PRODUCT(qmmm_ewald_r%eslf(1:numat,i),qm_charges(1:numat))
        empot_all(i) = (qmmm_ewald_r%empot(i) + ewdpot) !!!!* ev_a0
     end do
  end if
#if KEY_PARALLEL==1
  if(numnod>1) call VDGBRE(empot_all,JPARPT_local)
#endif

  ! copy empot to empot_local
  empot_local(1:numat)=qmmm_ewald_r%empot(1:numat)

  return
  END SUBROUTINE qm_ewald_prepare_fock


  real(chm_real) function qm_ewald_core(numat,nat,qm_charges) 
  !
  ! Computes the interaction of the Ewald potenital with CORE in QM atoms. 
  ! Ewald_core in electron volts
  !               Ewald_core = Sum(Core(i)*V(i))
  !
  use qm1_info, only       : qm_control_r
  use qm1_constant
  use qm1_parameters, only : CORE
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none

  ! Passed in
  integer, intent(in) :: numat
  integer, intent(in) :: nat(numat)
  real(chm_real),intent(in) :: qm_charges(numat)
  !!!real(chm_real),intent(in) :: empot(numat),eslf(numat,numat)
  real(chm_real),parameter :: ev_a0 = EV*A0 ! convert (electrons/angstrom) to (eV/Bohr)

  ! Local variables
  integer        :: i
  real(chm_real) :: ewdtmp, ewdpot, ewald_core
  integer,save :: old_N = 0
  integer,save :: mstart,mstop
#if KEY_PARALLEL==1
  integer      :: ISTRT_CHECK   ! external function
#endif

  ! for parallelization
  if(old_N .ne. numat) then
     old_N =numat
     mstart=1
     mstop =numat
#if KEY_PARALLEL==1
     if(numnod>1) mstart = ISTRT_CHECK(mstop,numat)
  !else
  !   if(QMPI) then
  !      mstart=1
  !      mstop =numat
  !   end if
#endif
  end if

  ewald_core     = zero
  if(qm_control_r%q_do_cpmd_pme) then
     do i = mstart,mstop                     ! 1, numat
        ewdtmp    = DOT_PRODUCT(qmmm_ewald_r%eslf(1:numat,i),qm_charges(1:numat))
        ewdpot    = qmmm_ewald_r%empot(i) + half*(ewdtmp + qmmm_ewald_r%empot_qm_pme(i))
        ewald_core= ewald_core + ewdpot*CORE(nat(i))
     end do
  else
     do i = mstart,mstop                     ! 1, numat
        ewdtmp    = DOT_PRODUCT(qmmm_ewald_r%eslf(1:numat,i),qm_charges(1:numat))
        ewdpot    = qmmm_ewald_r%empot(i)   + half*ewdtmp 
        ewald_core= ewald_core + ewdpot*CORE(nat(i))
     end do
  end if
  ! energy conversion will be done later.
  qm_ewald_core = ewald_core * ev_a0
     
  return
  END function qm_ewald_core


  real(chm_real) function qm_pme_energy_corr(numat,qm_charges)
  !
  ! Computes the energy component relevant to the PME part with the QM atoms.
  ! PME_corr in electron volts
  !          PME_corr = Sum(charge(i)*V(i))
  !
  use qm1_constant
#if KEY_PARALLEL==1
  use parallel
#endif
  use number,only: half

  implicit none

  ! Passed in
  integer, intent(in) :: numat
  real(chm_real),intent(in):: qm_charges(numat)
  ! real(chm_real),parameter :: ev_a0 = EV*A0 ! convert (electrons/angstrom) to (eV/Bohr)

  ! Local variables
  integer        :: i
  real(chm_real) :: ewald_corr
  integer,save   :: old_N = 0
  integer,save   :: mstart,mstop
#if KEY_PARALLEL==1
  integer        :: ISTRT_CHECK   ! external function
#endif

  ! for parallelization
  if(old_N .ne. numat) then
     old_N =numat
     mstart=1
     mstop =numat
#if KEY_PARALLEL==1
     if(numnod>1) mstart = ISTRT_CHECK(mstop,numat)
  !else
  !   if(QMPI) then
  !      mstart=1
  !      mstop =numat
  !   end if
#endif
  end if

  ewald_corr     = zero
  do i = mstart,mstop                     ! 1, numat
     !write(6,'(I4,2F12.5)') i,qmmm_ewald_r%empot_pme(i),qmmm_ewald_r%empot_qm_pme(i)
     ewald_corr= ewald_corr + (qmmm_ewald_r%empot_pme(i)+half*qmmm_ewald_r%empot_qm_pme(i))*qm_charges(i) 
  end do
  ! energy conversion will be done later. : kcal/mol = ev_a0*EVcal = ccelec (for the consistency with PME routine.)
  qm_pme_energy_corr = ewald_corr !  * ev_a0

  return
  END function qm_pme_energy_corr


  SUBROUTINE qm_ewald_real_space_gradient(natom,numat,numatm,                     &
                                          mm_coord,mm_chrgs,qm_coords,qm_charges, &
                                          dxyzqm,dxyzcl)
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
  use erfcd_mod,only: erfcd
  use qm1_info, only : qm_main_r,mm_main_r
  use qm1_constant
  use nbndqm_mod, only: map_qmatom_to_group,map_mmatom_to_group
#if KEY_PARALLEL==1
  use parallel 
#endif
  !
  implicit none

  ! Passed in
  integer, intent(in)    :: natom,numat,numatm

  real(chm_real), intent(in)   :: mm_coord(3,natom),mm_chrgs(natom), &
                                  qm_coords(3,numat),qm_charges(numat)
  real(chm_real), intent(inout):: dxyzqm(3,numat),dxyzcl(*)

  ! Local variables
  integer  :: i,j,loop_count, inner_loop_count,kk,Erfmod_local,icnt_qm,icnt_mm
  integer  :: offset,iminus,iplus
  integer  :: irs_mm,irs_qm,irs_qm2,ido_switching,ido_switching_2
  integer  :: mstart,mstop,mmynod,nnumnod
  real(chm_real)          :: qmmulik_chg,df_qmmm,oneRIJ2,r2,onerij,rij
  real(chm_real)          :: vec(3),xyz_i(3),df_xyz(3),erfcx,drfc,kappa_local
  real(chm_real)          :: sw_scale,dxyz_sw(3),rtmp
  real(chm_real),parameter:: sfact=EVCAL*EV*A0
  integer  :: ISTRT_CHECK                 ! for external function

#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
#else
  mmynod  = 0
  nnumnod = 1
#endif

  ! do some initialization
  mstart = 1
  mstop  = numatm
  Erfmod_local = qmmm_ewald_r%erfmod
  kappa_local  = qmmm_ewald_r%kappa
  !
#if KEY_PARALLEL==1
  if(nnumnod>1) mstart = ISTRT_CHECK(mstop,numatm)
#endif

  !Step 1) do QM atoms with MM atoms in the cutoff list
  !        We need to calculate the distance between QM-MM pairs on the fly.
  if(mm_main_r%rij_mm_incore) then
     icnt_mm = 0
     do i = 1, numat
        inner_loop_count = 1
        qmmulik_chg      = qm_charges(i)*sfact

#if KEY_PARALLEL==1
        inner_loop_count=inner_loop_count+3*(mstart-1)
#endif
        ! group-by-group-based cutoff case, skip the pair if its distance is longer than cutoff.
        ! otherwise (default group-based case), include all mm atoms (default).
        if(mm_main_r%q_cut_by_group) then
           irs_qm = map_qmatom_to_group(i)
           do j = mstart,mstop
              icnt_mm       = icnt_mm + 1
              irs_mm        = map_mmatom_to_group(j)
              if(mm_main_r%q_switch) then
                 ido_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
              else
                 ido_switching = -1
              end if
              !
              if(mm_main_r%q_mmgrp_qmgrp_cut(irs_mm,irs_qm)) then
                 vec(1:3)   = qm_coords(1:3,i)-mm_coord(1:3,j)

                 if(mm_main_r%q_diag_coulomb) then
                    ! d(1/r_ij)/dx_j = - (1/r_ij)^3*(x_j-x_i)
                    ! d(1/r_ij)/dx_i =   (1/r_ij)^3*(x_j-x_i)
                    onerij   = qmmm_ewald_r%rijdata_qmmm(2,icnt_mm)  ! one/rij value.
                    sw_scale =-onerij*onerij*onerij
                 else
                    ! contribution by switching function. in fact, (1-Sw(rij))*c_j/rij
                    ! d(1-Sw(rij))/drij = - dSw(rij)/drij...
                    if(ido_switching > 0) then
                       ! apply switching function contribution
                       sw_scale = one - mm_main_r%sw_val(ido_switching)
                       onerij   = qmmm_ewald_r%rijdata_qmmm(2,icnt_mm)  ! one/rij value.
                       ! d(r_ij)^-1 / dx_j = - (1/r_ij)^3*(x_j-x_i)
                       ! d(r_ij)^-1 / dx_i =   (1/r_ij)^3*(x_j-x_i) 
                       sw_scale =-sw_scale*onerij*onerij*onerij

                       rtmp     = mm_chrgs(j)*qmmulik_chg*onerij        ! q_mm*q_qm/rij
                       dxyz_sw(1:3) =-rtmp*mm_main_r%dsw_val(1:3,ido_switching)
                       ! for qm atoms.
                       mm_main_r%dxyz_sw2(1:3,ido_switching) = mm_main_r%dxyz_sw2(1:3,ido_switching)  &
                                                     +dxyz_sw(1:3)*mm_main_r%r_num_atom_qm_grp(irs_qm)
                       ! for mm atoms.
                       mm_main_r%dxyz_sw2(4:6,ido_switching) = mm_main_r%dxyz_sw2(4:6,ido_switching)  &
                                                     -dxyz_sw(1:3)*mm_main_r%r_num_atom_mm_grp(irs_mm)
                    else
                       sw_scale = zero
                    end if
                 end if

                 ! since qmmmerfcx_data = (-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
                 df_qmmm    = mm_chrgs(j)*(qmmm_ewald_r%qmmmerfcx_data(icnt_mm) + sw_scale)
                 df_xyz(1:3)= vec(1:3)*df_qmmm*qmmulik_chg

                 kk = inner_loop_count
                 dxyzqm(1:3,i)  = dxyzqm(1:3,i)  + df_xyz(1:3)
                 dxyzcl(kk:kk+2)= dxyzcl(kk:kk+2)- df_xyz(1:3)
              end if
              inner_loop_count = inner_loop_count + 3
           end do                                    ! j=1,four_numatm,4
        else
           if(mm_main_r%q_switch) irs_qm = map_qmatom_to_group(i)
           do j = mstart,mstop
              icnt_mm    = icnt_mm + 1
              vec(1:3)   = qm_coords(1:3,i)-mm_coord(1:3,j)

              if(mm_main_r%q_switch) then
                 irs_mm        = map_mmatom_to_group(j)
                 ido_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
              else
                 ido_switching = -1
              end if

              if(mm_main_r%q_diag_coulomb) then
                 ! d(1/r_ij)/dx_j = - (1/r_ij)^3*(x_j-x_i)
                 ! d(1/r_ij)/dx_i =   (1/r_ij)^3*(x_j-x_i)
                 onerij   = qmmm_ewald_r%rijdata_qmmm(2,icnt_mm)  ! one/rij value.
                 sw_scale =-onerij*onerij*onerij
              else
                 ! contribution by switching function. in fact, (1-Sw(rij))*c_j/rij
                 ! d(1-Sw(rij))/drij = - dSw(rij)/drij...
                 if(ido_switching > 0) then
                    ! apply switching function contribution
                    sw_scale = one - mm_main_r%sw_val(ido_switching)
                    onerij   = qmmm_ewald_r%rijdata_qmmm(2,icnt_mm)  ! one/rij value.
                    ! d(r_ij)^-1 / dx_j = - (1/r_ij)^3*(x_j-x_i)
                    ! d(r_ij)^-1 / dx_i =   (1/r_ij)^3*(x_j-x_i)
                    sw_scale =-sw_scale*onerij*onerij*onerij

                    rtmp     = mm_chrgs(j)*qmmulik_chg*onerij        ! q_mm*q_qm/rij
                    dxyz_sw(1:3) =-rtmp*mm_main_r%dsw_val(1:3,ido_switching)
                    ! for qm atoms: since the qm group for which the switching function is computed
                    !               could be different from irs_qm.
                    irs_qm2         = mm_main_r%q_mmgrp_point_swt(irs_mm)
                    ido_switching_2 = mm_main_r%q_backmap_dxyz_sw(ido_switching)
                    mm_main_r%dxyz_sw2(1:3,ido_switching_2) = mm_main_r%dxyz_sw2(1:3,ido_switching_2)  &
                                                  +dxyz_sw(1:3)*mm_main_r%r_num_atom_qm_grp(irs_qm2)
                    ! for mm atoms.
                    mm_main_r%dxyz_sw2(4:6,ido_switching_2) = mm_main_r%dxyz_sw2(4:6,ido_switching_2)  &
                                                  -dxyz_sw(1:3)*mm_main_r%r_num_atom_mm_grp(irs_mm)
                 else
                    sw_scale = zero
                 end if
              end if

              ! since qmmmerfcx_data = (-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
              df_qmmm    = mm_chrgs(j)*(qmmm_ewald_r%qmmmerfcx_data(icnt_mm) + sw_scale)
              df_xyz(1:3)= vec(1:3)*df_qmmm*qmmulik_chg

              kk             = inner_loop_count
              dxyzqm(1:3,i)  = dxyzqm(1:3,i)  + df_xyz(1:3)
              dxyzcl(kk:kk+2)= dxyzcl(kk:kk+2)- df_xyz(1:3)

              inner_loop_count = inner_loop_count + 3
           end do                                    ! j=1,four_numatm,4
        end if
     end do                                       ! i=1,numat
  else
     do i = 1, numat
        inner_loop_count = 1
        qmmulik_chg      = qm_charges(i)*sfact

#if KEY_PARALLEL==1
        inner_loop_count=inner_loop_count+3*(mstart-1)
#endif
        ! group-by-group-based cutoff case, skip the pair if its distance is longer than cutoff.
        ! otherwise (default group-based case), include all mm atoms (default).
        if(mm_main_r%q_cut_by_group) then
           irs_qm = map_qmatom_to_group(i)
           do j = mstart,mstop
              irs_mm = map_mmatom_to_group(j)
              if(mm_main_r%q_mmgrp_qmgrp_cut(irs_mm,irs_qm)) then
                 vec(1:3)   = qm_coords(1:3,i)-mm_coord(1:3,j)
                 r2         = vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)
                 if(mm_main_r%q_switch) then
                    ido_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
                 else
                    ido_switching = -1
                 end if

                 oneRIJ     = one/sqrt(r2)
                 oneRIJ2    = oneRIJ*oneRIJ
                 rij        = r2*oneRIJ                        ! one/oneRIJ
                 Call ERFCD(rij,kappa_local,erfcx,drfc,Erfmod_local)

                 if(mm_main_r%q_diag_coulomb) then
                    ! d(1/r_ij)/dx_j = - (1/r_ij)^3*(x_j-x_i)
                    ! d(1/r_ij)/dx_i =   (1/r_ij)^3*(x_j-x_i)
                    sw_scale =-onerij2*onerij
                 else
                    ! contribution by switching function. in fact, (1-Sw(rij))*c_j/rij
                    ! d(1-Sw(rij))/drij = - dSw(rij)/drij...
                    if(ido_switching > 0) then
                       ! apply switching function contribution
                       sw_scale = one - mm_main_r%sw_val(ido_switching)
                       ! d(r_ij)^-1 / dx_j = - (1/r_ij)^3*(x_j-x_i)
                       ! d(r_ij)^-1 / dx_i =   (1/r_ij)^3*(x_j-x_i) 
                       sw_scale =-sw_scale*onerij2*onerij

                       rtmp     = mm_chrgs(j)*qmmulik_chg*onerij        ! q_mm*q_qm/rij
                       dxyz_sw(1:3) =-rtmp*mm_main_r%dsw_val(1:3,ido_switching)
                       ! for qm atoms.
                       mm_main_r%dxyz_sw2(1:3,ido_switching) = mm_main_r%dxyz_sw2(1:3,ido_switching)  &
                                                     +dxyz_sw(1:3)*mm_main_r%r_num_atom_qm_grp(irs_qm)
                       ! for mm atoms.
                       mm_main_r%dxyz_sw2(4:6,ido_switching) = mm_main_r%dxyz_sw2(4:6,ido_switching)  &
                                                     -dxyz_sw(1:3)*mm_main_r%r_num_atom_mm_grp(irs_mm)
                    else
                       sw_scale = zero
                    end if
                 end if

                 df_qmmm    = mm_chrgs(j)*((-drfc+(one-erfcx)*oneRIJ)*oneRIJ2 + sw_scale)
                 df_xyz(1:3)= vec(1:3)*df_qmmm*qmmulik_chg

                 kk = inner_loop_count
                 dxyzqm(1:3,i)  = dxyzqm(1:3,i)  + df_xyz(1:3)
                 dxyzcl(kk:kk+2)= dxyzcl(kk:kk+2)- df_xyz(1:3)
              end if
              inner_loop_count = inner_loop_count + 3
           end do                                    ! j=1,four_numatm,4
        else
           if(mm_main_r%q_switch) irs_qm = map_qmatom_to_group(i)
           do j = mstart,mstop
              vec(1:3)   = qm_coords(1:3,i)-mm_coord(1:3,j)
              r2         = vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)

              oneRIJ     = one/sqrt(r2)
              oneRIJ2    = oneRIJ*oneRIJ
              rij        = r2*oneRIJ                        ! one/oneRIJ
              Call ERFCD(rij,kappa_local,erfcx,drfc,Erfmod_local)

              ido_switching = -1
              if(mm_main_r%q_switch) then
                 irs_mm        = map_mmatom_to_group(j)
                 ido_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
              else
                 ido_switching = -1
              end if

              if(mm_main_r%q_diag_coulomb) then
                 ! d(1/r_ij)/dx_j = - (1/r_ij)^3*(x_j-x_i)
                 ! d(1/r_ij)/dx_i =   (1/r_ij)^3*(x_j-x_i)
                 sw_scale =-onerij2*onerij
              else
                 if(ido_switching > 0) then
                    ! apply switching function contribution
                    sw_scale = one - mm_main_r%sw_val(ido_switching)
                    ! d(r_ij)^-1 / dx_j = - (1/r_ij)^3*(x_j-x_i)
                    ! d(r_ij)^-1 / dx_i =   (1/r_ij)^3*(x_j-x_i)
                    sw_scale =-sw_scale*onerij*onerij*onerij

                    rtmp     = mm_chrgs(j)*qmmulik_chg*onerij        ! q_mm*q_qm/rij
                    dxyz_sw(1:3) =-rtmp*mm_main_r%dsw_val(1:3,ido_switching)
                    ! for qm atoms.
                    irs_qm2  = mm_main_r%q_mmgrp_point_swt(irs_mm)
                    ido_switching_2 = mm_main_r%q_backmap_dxyz_sw(ido_switching)
                    mm_main_r%dxyz_sw2(1:3,ido_switching_2) = mm_main_r%dxyz_sw2(1:3,ido_switching_2)  &
                                                  +dxyz_sw(1:3)*mm_main_r%r_num_atom_qm_grp(irs_qm2)
                    ! for mm atoms.
                    mm_main_r%dxyz_sw2(4:6,ido_switching_2) = mm_main_r%dxyz_sw2(4:6,ido_switching_2)  &
                                                  -dxyz_sw(1:3)*mm_main_r%r_num_atom_mm_grp(irs_mm)
                 else
                    sw_scale = zero
                 end if
              end if

              df_qmmm    = mm_chrgs(j)*((-drfc+(one-erfcx)*oneRIJ)*oneRIJ2 + sw_scale)
              df_xyz(1:3)= vec(1:3)*df_qmmm*qmmulik_chg

              kk = inner_loop_count
              dxyzqm(1:3,i)  = dxyzqm(1:3,i)  + df_xyz(1:3)
              dxyzcl(kk:kk+2)= dxyzcl(kk:kk+2)- df_xyz(1:3)

              inner_loop_count = inner_loop_count + 3
           end do                                    ! j=1,four_numatm,4
        end if
     end do                                       ! i=1,numat
  end if


  !Step 2) do all real space QM atoms with QM atoms
  !        We have to calculate it all on the fly
#if KEY_PARALLEL==1
  loop_count = 0
#endif
  if(qm_main_r%rij_qm_incore) then
     ! this only loops over 1=2,numat, j=1,i-1, so, no half factor.
     icnt_qm   =0
     loopi1: do i = 2, numat
        xyz_i(1:3)  = qm_coords(1:3,i)
        qmmulik_chg = qm_charges(i)*sfact
        loopj1: do j = 1, i-1
#if KEY_PARALLEL==1
           loop_count = loop_count + 1
           if(mmynod .ne. mod(loop_count-1,nnumnod)) cycle loopj1
#endif
           !
           vec(1:3) = xyz_i(1:3)-qm_coords(1:3,j)

           ! since qmqmerfcx_data = (-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
           icnt_qm  = icnt_qm + 1   ! this is only needed counter.
           df_qmmm  = qmmulik_chg*qm_charges(j)*qmmm_ewald_r%qmqmerfcx_data(icnt_qm)
           df_xyz(1:3)= vec(1:3)*df_qmmm

           dxyzqm(1:3,i)= dxyzqm(1:3,i) + df_xyz(1:3)
           dxyzqm(1:3,j)= dxyzqm(1:3,j) - df_xyz(1:3)
        end do loopj1
     end do    loopi1
  else
     ! this loops over both i=1-numat and j=1-numat, so, it has a factor half.
     do i = 1, numat
        xyz_i(1:3)  = qm_coords(1:3,i)
        qmmulik_chg = qm_charges(i)*sfact 
        do j = 1, numat
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
                 Call ERFCD(rij,kappa_local,erfcx,drfc,Erfmod_local)

                 df_qmmm    = half*qmmulik_chg*qm_charges(j)*(-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
                 df_xyz(1:3)= vec(1:3)*df_qmmm

                 dxyzqm(1:3,i)= dxyzqm(1:3,i) + df_xyz(1:3)
                 dxyzqm(1:3,j)= dxyzqm(1:3,j) - df_xyz(1:3)
#if KEY_PARALLEL==1
              end if
#endif
           end if
        end do
     end do
  end if

  return
  END SUBROUTINE qm_ewald_real_space_gradient


  SUBROUTINE qm_ewald_real_space_gradient_exl(natom,numat,qm_coords,qm_charges,dxyzqm)
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
  use qm1_constant

#if KEY_PARALLEL==1
  use parallel
#endif
  !
  implicit none

  ! Passed in
  integer, intent(in)    :: natom,numat
  real(chm_real),intent(in)    :: qm_coords(3,numat),qm_charges(numat)
  real(chm_real),intent(inout) :: dxyzqm(3,numat)
!
!!!  integer, intent(in)       :: nexl_atm
!!!  integer, intent(in)       :: nexl_index(nexl_atm)
!!!  real(chm_real),intent(in) :: exl_xyz(3,nexl_atm),exl_chg(nexl_atm), &
!!!                               dexl_xyz(3,nexl_atm)

  ! Local variables
  integer        :: i,j,Erfmod_local
  real(chm_real) :: qmmulik_chg,df_qmmm,oneRIJ2,r2,onerij,rij
  real(chm_real) :: vec(3),xyz_i(3),df_xyz(3),erfcx,drfc,kappa_local
  real(chm_real),parameter:: sfact=EVCAL*EV*A0

  !Step 3) do QM atoms with exclude MM atoms from QM-MM nonbonded 
  !        interactions
  !        (IGMSEL(i)=5)
  if (qmmm_ewald_r%nexl_atm .gt. 0) then
#if KEY_PARALLEL==1
     ! do only mynod.eq.0
     if(mynod.eq.0) then 
#endif
        Erfmod_local= qmmm_ewald_r%erfmod
        kappa_local = qmmm_ewald_r%kappa
        do i = 1, numat
           xyz_i(1:3)  = qm_coords(1:3,i)
           qmmulik_chg = qm_charges(i)*sfact

           do j = 1, qmmm_ewald_r%nexl_atm
              vec(1:3) = xyz_i(1:3)-qmmm_ewald_r%exl_xyz(1:3,j)
              r2       = vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3)
              oneRIJ   = one / sqrt(r2)
              oneRIJ2    = oneRIJ*oneRIJ
              rij        = r2*oneRIJ
              Call ERFCD(rij,kappa_local,erfcx,drfc,Erfmod_local)
  ! namkh: 0419-2013.
  !        why do we need half here? should remove it.
  !            df_qmmm    = half*qmmulik_chg*qmmm_ewald_r%exl_chg(j) &
  !                                         *(-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
              df_qmmm    = qmmulik_chg*qmmm_ewald_r%exl_chg(j) &
                                            *(-drfc+(one-erfcx)*oneRIJ)*oneRIJ2
              df_xyz(1:3)= vec(1:3)*df_qmmm

              dxyzqm(1:3,i)                = dxyzqm(1:3,i)   + df_xyz(1:3)
              qmmm_ewald_r%dexl_xyz(1:3,j) = qmmm_ewald_r%dexl_xyz(1:3,j) - df_xyz(1:3)
           end do
        end do
#if KEY_PARALLEL==1
     end if
#endif
  end if

  return
  END SUBROUTINE qm_ewald_real_space_gradient_exl   

  SUBROUTINE qm_ewald_recip_space_gradient(natom,numat,qminb, &
                                           iastrt,iafinl,iatotl,itotal,itotkq, &
                                           totkq,ksqmaxq,   &
                                           kmaxqx,kmaxqy,kmaxqz,   &
                                           mmcharges,qm_charges, &
                                           recip,virial,QNoPMEwald)
  !
  ! Calculate Reciprocal Space Gradient
  !
  !    Calculate the gradient and save into "d_ewald_mm" array, which 
  !    will be added into main gradient array at "ewaldf.src" file.
  !
  use qm1_constant
#if KEY_PARALLEL==1
  use parallel
#endif
  !
  implicit none

  ! Passed in
  integer, intent(in):: natom,numat,iastrt,iafinl,iatotl,itotal, &
                        itotkq,totkq,ksqmaxq,kmaxqx,kmaxqy,kmaxqz
  integer, intent(in):: qminb(numat)

  real(chm_real),intent(in):: mmcharges(natom),qm_charges(numat)
  real(chm_real),intent(in):: recip(6)
  real(chm_real),intent(inout):: virial(9)
  logical, intent(in):: QNoPMEwald
!!!  real(chm_real),intent(in):: kvec(totkq),ktable(6,itotal,itotkq),  &  ! ktable(6,iatotl,totkq), 
!!!                              qmktable(6,numat,totkq),              &  ! iatotl=natom
!!!                              structfac_mm(2,itotkq)
!!!  real(chm_real),intent(inout):: d_ewald_mm(3,natom)

  ! Local variables
  integer :: i,j,loop_count,qmid,kx,ky,kz,ksy,ksz,ksq
  integer  :: mstart,mstop,inner_loop
  integer  :: ISTRT_CHECK                 ! for external function
  !
  real(chm_real) :: qmmulik_chg,mmchg,sfact,picoef,kvect
  real(chm_real) :: vec(3), xyz_cos(3),xyz_sin(3),xyz_cos_j(3),xyz_sin_j(3)
  real(chm_real) :: rkx(3),rky(3),rkz(3),mkv(3),ccfk(3),ktgs(8),ktg(3),ksum(numat), &
                    fda(8),fdxyz(3),pikapa,kxyzr(3),klen,ewpr,ewen
  real(chm_real) :: rtmp_local(2),ewen_qm_qm,ewen_qm_mm, &
                    sin_sum_mm,cos_sum_mm,sin_sum_qm,cos_sum_qm
  real(chm_real) :: grad_qm(3,numat)      ! local QM gradient
  !
  real(chm_real),parameter :: ufact=EVCAL*EV*A0
#if KEY_PARALLEL==1
  integer,parameter :: i_array_length=4
  real(chm_real)    :: suml(4)
  integer :: mmynod, nnumnod
#endif
  logical :: q_do_qm

  ! do some initialization
!!  mstart = 1
!!  mstop  = numat
#if KEY_PARALLEL==1
     !!if(numnod>1) mstart = ISTRT_CHECK(mstop,numat)
     mmynod  = mynod
     nnumnod = numnod
#endif
  !
  loop_count = 0
  grad_qm    = zero

  ! do virial part
  pikapa     = two*((PI/qmmm_ewald_r%kappa)**2)  ! two/(four*kappa*kappa)
  !virial(1:9)= zero  ! this is done in mndene routine.

  do kx =0, kmaxqx
    if ( kx .eq. 0 ) then
       ksy = 0
    else
       ksy = -kmaxqy
    end if
    picoef = TWOPI  * REAL(kx)
    rkx(1) = picoef * recip(1)
    rkx(2) = picoef * recip(2)
    rkx(3) = picoef * recip(4)

    do ky = ksy, kmaxqy
       if (kx .eq. 0 .and. ky .eq. 0) then
          ksz = 1
       else
          ksz = -kmaxqz
       end if
       picoef = TWOPI  * REAL(ky)
       rky(1) = picoef * recip(2)
       rky(2) = picoef * recip(3)
       rky(3) = picoef * recip(5)

       do kz = ksz, kmaxqz
          sfact  = two
          picoef = TWOPI  * REAL(kz)
          rkz(1) = picoef * recip(4)
          rkz(2) = picoef * recip(5)
          rkz(3) = picoef * recip(6)
          ksq    = kx*kx + ky*ky + kz*kz

          if (ksq .le. ksqmaxq .and. ksq .ne. 0) then

             loop_count = loop_count + 1
             mkv(1:3)   = rkx(1:3) + rky(1:3) + rkz(1:3)

             kvect      = sfact*qmmm_ewald_r%kvec(loop_count)
             ccfk(1:3)  = mkv(1:3)*kvect

  ! do virial part
             kxyzr(1:3)     = mkv(1:3)*INVPI*half  ! mkv(1:3)/two_pi
             klen           = kxyzr(1)*kxyzr(1)+kxyzr(2)*kxyzr(2)+kxyzr(3)*kxyzr(3)
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
                   xyz_cos(1) = qmmm_ewald_r%ktable(1,inner_loop,loop_count)
                   xyz_sin(1) = qmmm_ewald_r%ktable(2,inner_loop,loop_count)
                   xyz_cos(2) = qmmm_ewald_r%ktable(3,inner_loop,loop_count)
                   xyz_sin(2) = qmmm_ewald_r%ktable(4,inner_loop,loop_count)
                   xyz_cos(3) = qmmm_ewald_r%ktable(5,inner_loop,loop_count)
                   xyz_sin(3) = qmmm_ewald_r%ktable(6,inner_loop,loop_count)

                   !mmchg      = mmcharges(j)
                   ktgs(1)    = ktgs(1) + mmcharges(j)*xyz_cos(1)*xyz_cos(2)*xyz_cos(3)  ! mmchg
                   ktgs(2)    = ktgs(2) + mmcharges(j)*xyz_sin(1)*xyz_cos(2)*xyz_cos(3)  ! mmchg
                   ktgs(3)    = ktgs(3) + mmcharges(j)*xyz_cos(1)*xyz_sin(2)*xyz_cos(3)  ! mmchg
                   ktgs(4)    = ktgs(4) + mmcharges(j)*xyz_sin(1)*xyz_sin(2)*xyz_cos(3)  ! mmchg
                   ktgs(5)    = ktgs(5) + mmcharges(j)*xyz_cos(1)*xyz_cos(2)*xyz_sin(3)  ! mmchg
                   ktgs(6)    = ktgs(6) + mmcharges(j)*xyz_sin(1)*xyz_cos(2)*xyz_sin(3)  ! mmchg
                   ktgs(7)    = ktgs(7) + mmcharges(j)*xyz_cos(1)*xyz_sin(2)*xyz_sin(3)  ! mmchg
                   ktgs(8)    = ktgs(8) + mmcharges(j)*xyz_sin(1)*xyz_sin(2)*xyz_sin(3)  ! mmchg
                end do
                do j = 1, numat
                   xyz_cos(1) = qmmm_ewald_r%qmktable(1,j,loop_count)
                   xyz_sin(1) = qmmm_ewald_r%qmktable(2,j,loop_count)
                   xyz_cos(2) = qmmm_ewald_r%qmktable(3,j,loop_count)
                   xyz_sin(2) = qmmm_ewald_r%qmktable(4,j,loop_count)
                   xyz_cos(3) = qmmm_ewald_r%qmktable(5,j,loop_count)
                   xyz_sin(3) = qmmm_ewald_r%qmktable(6,j,loop_count)

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
                   qmmulik_chg    = qm_charges(j)*ufact
                   grad_qm(1:3,j) = grad_qm(1:3,j) + ccfk(1:3)*fdxyz(1:3)*qmmulik_chg
                end do

  ! MM-QM interaction
                fda(1:8) = zero
                do j = 1, numat
                   xyz_cos(1) = qmmm_ewald_r%qmktable(1,j,loop_count)
                   xyz_sin(1) = qmmm_ewald_r%qmktable(2,j,loop_count)
                   xyz_cos(2) = qmmm_ewald_r%qmktable(3,j,loop_count)
                   xyz_sin(2) = qmmm_ewald_r%qmktable(4,j,loop_count)
                   xyz_cos(3) = qmmm_ewald_r%qmktable(5,j,loop_count)
                   xyz_sin(3) = qmmm_ewald_r%qmktable(6,j,loop_count)

                   qmmulik_chg= qm_charges(j)*ufact
                   fda(1)     = fda(1) + qmmulik_chg*xyz_sin(1)*xyz_cos(2)*xyz_cos(3)
                   fda(2)     = fda(2) + qmmulik_chg*xyz_cos(1)*xyz_cos(2)*xyz_cos(3)
                   fda(3)     = fda(3) + qmmulik_chg*xyz_sin(1)*xyz_sin(2)*xyz_cos(3)
                   fda(4)     = fda(4) + qmmulik_chg*xyz_cos(1)*xyz_sin(2)*xyz_cos(3)
                   fda(5)     = fda(5) + qmmulik_chg*xyz_sin(1)*xyz_cos(2)*xyz_sin(3)
                   fda(6)     = fda(6) + qmmulik_chg*xyz_cos(1)*xyz_cos(2)*xyz_sin(3)
                   fda(7)     = fda(7) + qmmulik_chg*xyz_sin(1)*xyz_sin(2)*xyz_sin(3)
                   fda(8)     = fda(8) + qmmulik_chg*xyz_cos(1)*xyz_sin(2)*xyz_sin(3)
                end do
                inner_loop = 0
                do j = iastrt,iafinl               ! 1, natom
                   inner_loop = inner_loop + 1
                   xyz_cos(1) = qmmm_ewald_r%ktable(1,inner_loop,loop_count)
                   xyz_sin(1) = qmmm_ewald_r%ktable(2,inner_loop,loop_count)
                   xyz_cos(2) = qmmm_ewald_r%ktable(3,inner_loop,loop_count)
                   xyz_sin(2) = qmmm_ewald_r%ktable(4,inner_loop,loop_count)
                   xyz_cos(3) = qmmm_ewald_r%ktable(5,inner_loop,loop_count)
                   xyz_sin(3) = qmmm_ewald_r%ktable(6,inner_loop,loop_count)

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
                   !mmchg             = mmcharges(j)
                   qmmm_ewald_r%d_ewald_mm(1:3,j) = qmmm_ewald_r%d_ewald_mm(1:3,j) &
                                                   -ccfk(1:3)*fdxyz(1:3)*mmcharges(j) !mmchg
                end do
             end if     ! QNoPMEwald

  ! QM-QM interaction
             sin_sum_qm=zero
             cos_sum_qm=zero
             !
#if KEY_PARALLEL==1
             if(mmynod .eq. mod(loop_count-1,nnumnod)) then
#endif
                do i = 1, numat
                   xyz_cos(1) = qmmm_ewald_r%qmktable(1,i,loop_count)
                   xyz_sin(1) = qmmm_ewald_r%qmktable(2,i,loop_count)
                   xyz_cos(2) = qmmm_ewald_r%qmktable(3,i,loop_count)
                   xyz_sin(2) = qmmm_ewald_r%qmktable(4,i,loop_count)
                   xyz_cos(3) = qmmm_ewald_r%qmktable(5,i,loop_count)
                   xyz_sin(3) = qmmm_ewald_r%qmktable(6,i,loop_count)
                   qmmulik_chg= qm_charges(i)*ufact

  ! do virial part, for qm.
                   rtmp_local(1)=xyz_cos(3)*xyz_cos(2)-xyz_sin(2)*xyz_sin(3)
                   rtmp_local(2)=xyz_cos(3)*xyz_sin(2)+xyz_cos(2)*xyz_sin(3)
                   sin_sum_qm=sin_sum_qm+qm_charges(i)*( xyz_cos(1)*rtmp_local(1) &
                                                        -xyz_sin(1)*rtmp_local(2) )
                   cos_sum_qm=cos_sum_qm+qm_charges(i)*( xyz_sin(1)*rtmp_local(1) &
                                                        +xyz_cos(1)*rtmp_local(2) )
  ! end

                   do j = 1,numat  ! as parallel is done above level (loop_count).
                      xyz_cos_j(1) = qmmm_ewald_r%qmktable(1,j,loop_count)
                      xyz_sin_j(1) = qmmm_ewald_r%qmktable(2,j,loop_count)
                      xyz_cos_j(2) = qmmm_ewald_r%qmktable(3,j,loop_count)
                      xyz_sin_j(2) = qmmm_ewald_r%qmktable(4,j,loop_count)
                      xyz_cos_j(3) = qmmm_ewald_r%qmktable(5,j,loop_count)
                      xyz_sin_j(3) = qmmm_ewald_r%qmktable(6,j,loop_count)

                      ktg(1:3)     = xyz_cos_j(1:3)*xyz_cos(1:3) &
                                    +xyz_sin_j(1:3)*xyz_sin(1:3)
                      fdxyz(1:3)   =-xyz_cos_j(1:3)*xyz_sin(1:3) &
                                    +xyz_sin_j(1:3)*xyz_cos(1:3)

  ! force on x,y,z-axis: temporary use of FDA array

  ! scf mulliken charge on qm atom j *atom i
  ! (adjusted to kcals)
                      mmchg  = qm_charges(j)*qmmulik_chg

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
#if KEY_PARALLEL==1
             end if  ! (mmynod .eq. mod(loop_count-1,nnumnod))
#endif

  ! do virial part
  !
  ! total |S(m)| = sin_sum**2+ cos_sum**2      , whre sin_sum=sin_sum_mm+sin_sum_qm
  !              = sin_sum_mm*sin_sum_mm+cos_sum_mm*cos_sum_mm       , mm-only component and 
  !                                                                    will be handled in the mm part.
  !               +two(sin_sum_mm*sin_sum_qm+cos_sum_mm*cos_sum*qm)  , mm-qm component
  !               +sin_sum_qm*sin_sum_qm+cos_sun_qm*cos_sun_qm       , qm-qm component.
  !
  ! where structfac_mm is already computed from qm_ewald_mm_pot.

             if(QNoPMEwald) then
                sin_sum_mm= qmmm_ewald_r%structfac_mm(1,loop_count)
                cos_sum_mm= qmmm_ewald_r%structfac_mm(2,loop_count)
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
  end do                                          ! kx =0, kmaxqx

  ! Now put gradient on grad_qm into d_ewald_mm array
  do i = 1, numat
     qmid = qminb(i)
     qmmm_ewald_r%d_ewald_mm(1:3,qmid) = qmmm_ewald_r%d_ewald_mm(1:3,qmid) &
                                        +grad_qm(1:3,i)
  end do

  ! if you want to use virial, only from pme, even including qm-qm pairs.
  !      if(.not.QNoPMEwald) virial(1:9)=zero

  return
  END SUBROUTINE qm_ewald_recip_space_gradient


  SUBROUTINE set_initialize_for_energy_gradient(q_ewald_call)
  !
  ! initialize several arrays. 
  ! Call this routine before calling qm_ewald_calc_ktable and others.
  ! 
  use chm_kinds
  implicit none
  logical :: q_ewald_call

  !
  qmmm_ewald_r%Ktable     = zero
  qmmm_ewald_r%qmktable   = zero
  if(q_ewald_call) qmmm_ewald_r%d_ewald_mm = zero

  return
  END SUBROUTINE set_initialize_for_energy_gradient


  SUBROUTINE get_exl_crd(x,y,z,cg)
  !
  ! get exl_xyz coordinates.
  ! 
  use chm_kinds
  implicit none
  !
  real(chm_real),intent(in) :: x(*),y(*),z(*),cg(*)

  integer :: i,nexcl

  if(qmmm_ewald_r%nexl_atm .le. 0) return
  !
  do i=1, qmmm_ewald_r%nexl_atm
     nexcl = qmmm_ewald_r%nexl_index(i)

     qmmm_ewald_r%exl_xyz(1,i) = x(nexcl)
     qmmm_ewald_r%exl_xyz(2,i) = y(nexcl)
     qmmm_ewald_r%exl_xyz(3,i) = z(nexcl)
     qmmm_ewald_r%exl_chg(i)   = cg(nexcl)
  end do
  return
  END SUBROUTINE get_exl_crd


  SUBROUTINE getgrdq(natom,dx,dy,dz)
  !
  ! Add kspace gradient contribution of QM/MM into gradient array.
  !
  use qm1_info, only : qm_control_r

  use chm_kinds
  implicit none
  ! 
  integer, intent(in)    :: natom
  real(chm_real),intent(inout) :: dx(*),dy(*),dz(*)
  integer                :: i

  !
  if(qm_control_r%q_do_cpmd_pme) return

  !do i = qmmm_ewald_r%iastrt,qmmm_ewald_r%iafinl       ! 1, Natom
  do i=1,natom
     dx(i) = dx(i) + qmmm_ewald_r%d_ewald_mm(1,i)
     dy(i) = dy(i) + qmmm_ewald_r%d_ewald_mm(2,i)
     dz(i) = dz(i) + qmmm_ewald_r%d_ewald_mm(3,i)
  end do

  return
  END SUBROUTINE getgrdq

  !=====================================================================
#endif /*mainsquatn*/

end module qmmmewald_module 
