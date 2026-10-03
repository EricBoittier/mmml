! Machine Learning Potential (MLP) QM/MM method module
module mndo97_dpmm_ciface
 implicit none
#if KEY_MNDO97==1 /*mndo97*/
#if KEY_MLPTORCH==1
 interface
  subroutine mndo97_dpmm_pyinternal_setup( &
     dpmm_ptname, dpmm_ctrlname, dpmm_use_gpu, dpmm_use_omp, dpmm_types, dpmm_ntypes, dpmm_qmmax, dpmm_mmmax, dpmm_setup_err) bind(C, name="mndo97_dpmm_internal_setup")
      use, intrinsic :: iso_c_binding, only: c_char, c_int
      character(kind=c_char), dimension(*), intent(in) :: dpmm_ptname, dpmm_ctrlname
      integer(c_int), value, intent(in) :: dpmm_use_gpu
      integer(c_int), value, intent(in) :: dpmm_use_omp
      integer(c_int), dimension(*), intent(out) :: dpmm_types
      integer(c_int), intent(out) :: dpmm_ntypes, dpmm_qmmax, dpmm_mmmax
      integer(c_int), intent(out) :: dpmm_setup_err
  end subroutine mndo97_dpmm_pyinternal_setup

  subroutine mndo97_dpmm_pyinternal( &
      dpmm_nqm, dpmm_nmm, dpmm_mmmax, dpmm_types, dpmm_use_gpu, dpmm_use_omp, &
      dpmm_qmx, dpmm_qmy, dpmm_qmz, dpmm_qmatomz, &
      dpmm_mmx, dpmm_mmy, dpmm_mmz, dpmm_mmcg, &
      dpmm_e, &
      dpmm_qmdx, dpmm_qmdy, dpmm_qmdz, &
      dpmm_mmdx, dpmm_mmdy, dpmm_mmdz) bind(C, name="mndo97_dpmm_internal")
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    integer(c_int), value, intent(in) :: dpmm_nqm, dpmm_nmm
    integer(c_int), value, intent(in) :: dpmm_mmmax
    integer(c_int), dimension(*), intent(in) :: dpmm_types
    integer(c_int), value, intent(in) :: dpmm_use_gpu
    integer(c_int), value, intent(in) :: dpmm_use_omp
    real(c_double), dimension(*), intent(in) :: dpmm_qmx, dpmm_qmy, dpmm_qmz
    integer(c_int),  dimension(*), intent(in) :: dpmm_qmatomz
    real(c_double), dimension(*), intent(in) :: dpmm_mmx, dpmm_mmy, dpmm_mmz
    real(c_double), dimension(*), intent(in) :: dpmm_mmcg
    real(c_double),               intent(out) :: dpmm_e
    real(c_double), dimension(*), intent(out) :: dpmm_qmdx, dpmm_qmdy, dpmm_qmdz
    real(c_double), dimension(*), intent(out) :: dpmm_mmdx, dpmm_mmdy, dpmm_mmdz
  end subroutine mndo97_dpmm_pyinternal
 end interface
#endif
#endif
end module mndo97_dpmm_ciface 

module mndo97_mlp
  use chm_kinds
  use dimens_fcm

#if KEY_MNDO97==1 /*mndo97*/
  implicit none

  contains

  !--------------------------------------------------------------------

  subroutine mlp_memory_allocate(natom,qallocate)
     ! allocate type arrays
     use qm1_info, only: qmmm_mlp
     implicit none
     integer :: natom
     logical :: qallocate
     integer :: ier=0
     
     ! deallocate if arrays are allocated
     if(allocated(qmmm_mlp%dx_mlp)) deallocate(qmmm_mlp%dx_mlp,stat=ier)
     if(allocated(qmmm_mlp%dy_mlp)) deallocate(qmmm_mlp%dy_mlp,stat=ier)
     if(allocated(qmmm_mlp%dz_mlp)) deallocate(qmmm_mlp%dz_mlp,stat=ier)

     if(qallocate) then
        allocate(qmmm_mlp%dx_mlp(natom),stat=ier)
        allocate(qmmm_mlp%dy_mlp(natom),stat=ier)
        allocate(qmmm_mlp%dz_mlp(natom),stat=ier)
     end if
     return
  end subroutine mlp_memory_allocate

  subroutine mlp_memory_allocate_pyexternal(qallocate,lx_local,lm_local,lc_local,li_local,lo_local)
   use qm1_info, only: qmmm_mlp
   implicit none
   logical :: qallocate
   integer :: ier=0
   integer :: lx_local, lm_local, lc_local, li_local, lo_local
   
   ! deallocate if arrays are allocated
   if (allocated(qmmm_mlp%package_mlp)) deallocate(qmmm_mlp%package_mlp)
   if (allocated(qmmm_mlp%model_mlp)) deallocate(qmmm_mlp%model_mlp)
   if (allocated(qmmm_mlp%ctrl_mlp)) deallocate(qmmm_mlp%ctrl_mlp)
   if (allocated(qmmm_mlp%filein_mlp)) deallocate(qmmm_mlp%filein_mlp)
   if (allocated(qmmm_mlp%fileout_mlp)) deallocate(qmmm_mlp%fileout_mlp)

   if(qallocate) then
      allocate(character(len=lx_local) :: qmmm_mlp%package_mlp)
      allocate(character(len=lm_local) :: qmmm_mlp%model_mlp)
      allocate(character(len=lc_local) :: qmmm_mlp%ctrl_mlp)
      allocate(character(len=li_local) :: qmmm_mlp%filein_mlp)
      allocate(character(len=lo_local) :: qmmm_mlp%fileout_mlp)
   end if
   return     
  end subroutine mlp_memory_allocate_pyexternal

  subroutine mlp_memory_allocate_pyinternal_chararrays(qallocate, lm_local_c, lc_local_c)
   use qm1_info, only: qmmm_mlp
   use, intrinsic :: iso_c_binding, only: c_int
   implicit none
   logical :: qallocate
   integer(c_int) :: lm_local_c, lc_local_c
   integer :: ier=0
   
   ! deallocate if arrays are allocated
   if (allocated(qmmm_mlp%model_mlp_c)) deallocate(qmmm_mlp%model_mlp_c)
   if (allocated(qmmm_mlp%ctrl_mlp_c)) deallocate(qmmm_mlp%ctrl_mlp_c)

   if(qallocate) then
      allocate(qmmm_mlp%model_mlp_c(lm_local_c+1))
      allocate(qmmm_mlp%ctrl_mlp_c(lc_local_c+1))      
   end if
   return     
  end subroutine mlp_memory_allocate_pyinternal_chararrays

  subroutine mlp_memory_allocate_pyinternal(qallocate, ntypes_local_c, qmmax_local_c, mmmax_local_c)
   use qm1_info, only: qmmm_mlp
   use, intrinsic :: iso_c_binding, only: c_int
   implicit none
   logical :: qallocate
   integer :: ier=0
   integer(c_int) :: ntypes_local_c, qmmax_local_c, mmmax_local_c
   
   ! deallocate if arrays are allocated

   if (allocated(qmmm_mlp%types_mlp_c)) deallocate(qmmm_mlp%types_mlp_c,stat=ier)

   if (allocated(qmmm_mlp%qmx_mlp_c)) deallocate(qmmm_mlp%qmx_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%qmy_mlp_c)) deallocate(qmmm_mlp%qmy_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%qmz_mlp_c)) deallocate(qmmm_mlp%qmz_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%qm_Z_mlp_c)) deallocate(qmmm_mlp%qm_Z_mlp_c,stat=ier)

   if (allocated(qmmm_mlp%mmx_mlp_c)) deallocate(qmmm_mlp%mmx_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%mmy_mlp_c)) deallocate(qmmm_mlp%mmy_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%mmz_mlp_c)) deallocate(qmmm_mlp%mmz_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%mmcg_mlp_c)) deallocate(qmmm_mlp%mmcg_mlp_c,stat=ier)

   if (allocated(qmmm_mlp%qmdx_mlp_c)) deallocate(qmmm_mlp%qmdx_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%qmdy_mlp_c)) deallocate(qmmm_mlp%qmdy_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%qmdz_mlp_c)) deallocate(qmmm_mlp%qmdz_mlp_c,stat=ier)

   if (allocated(qmmm_mlp%mmdx_mlp_c)) deallocate(qmmm_mlp%mmdx_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%mmdy_mlp_c)) deallocate(qmmm_mlp%mmdy_mlp_c,stat=ier)
   if (allocated(qmmm_mlp%mmdz_mlp_c)) deallocate(qmmm_mlp%mmdz_mlp_c,stat=ier)

   if(qallocate) then

      allocate(qmmm_mlp%types_mlp_c(ntypes_local_c),stat=ier)

      allocate(qmmm_mlp%qmx_mlp_c(qmmax_local_c),stat=ier)
      allocate(qmmm_mlp%qmy_mlp_c(qmmax_local_c),stat=ier)
      allocate(qmmm_mlp%qmz_mlp_c(qmmax_local_c),stat=ier)
      allocate(qmmm_mlp%qm_Z_mlp_c(qmmax_local_c),stat=ier)

      allocate(qmmm_mlp%mmx_mlp_c(mmmax_local_c),stat=ier)
      allocate(qmmm_mlp%mmy_mlp_c(mmmax_local_c),stat=ier)
      allocate(qmmm_mlp%mmz_mlp_c(mmmax_local_c),stat=ier)
      allocate(qmmm_mlp%mmcg_mlp_c(mmmax_local_c),stat=ier)

      allocate(qmmm_mlp%qmdx_mlp_c(qmmax_local_c),stat=ier)
      allocate(qmmm_mlp%qmdy_mlp_c(qmmax_local_c),stat=ier)
      allocate(qmmm_mlp%qmdz_mlp_c(qmmax_local_c),stat=ier)
      
      allocate(qmmm_mlp%mmdx_mlp_c(mmmax_local_c),stat=ier)
      allocate(qmmm_mlp%mmdy_mlp_c(mmmax_local_c),stat=ier)
      allocate(qmmm_mlp%mmdz_mlp_c(mmmax_local_c),stat=ier)
   end if
   return     
  end subroutine mlp_memory_allocate_pyinternal


  subroutine setup_qmhub(qmhub_ml,qmmm_mlp_only,qmhub_python,qmhub_dpmm,qmlp_mode,qmlp_lgpu,my_replica) ! COMLYN,COMLEN)
     !
     ! initial setup for MLP QM/MM.
     !
     ! usage
     ! PYTHon|DPMM   MLPMode [int]                        ! between MLP or. delta-MLP
     !               PGPUid [int]                         ! PyTorch GPU id: 0 ~ ; -1 not use gpu
     !
     ! note
     ! for MLP case:  we will still call qm/mm-pme for the long-range contribution
     !                and substract qm/mm energy within the cutoff, plus MLP energy call for
     !                MLP contribution within the cutoff.
     ! E_qm/mm-total = E_qm/mm-pme + (E_MLP (cutoff) - E_qm/mm (cutoff))
     ! 
     !                so, first call the qm/mm-pme energy, followed by MLP & qm/mm cutoff energies
     !
     ! for delta-MLP case:
     !               first, call qm/mm-pme for the entire system's energy, plus
     !               MLP energy call for delta-MLP contribution.
     ! E_qm/mm-total = E_qm/mm-pme + E_delta-MLP (cutoff)
     !
     !                so, first call the qm/mm-pme energy, followed by delta-MLP cutoff energy
     !
     ! now, for MTP ai-qm/mm case with MLP
     ! for MLP
     ! E_qm/mm-total = [ E_se-qm/mm-pme + (E_MLP (cutoff) - E_se-qm/mm (cutoff)) ]
     !                +[ E_ai-qm/mm (cutoff) - E_MLP (cutoff) ]
     ! 
     ! for delta-MLP
     ! E_qm/mm-total = [ E_qm/mm-pme + E_delta-MLP (cutoff) ]
     !                +[ E_ai-qm/mm (cutoff) - [E_se-qm/mm (cutoff) + E_delta-MLP (cutoff)] ]
     !
     !                so, in both, in addition to the MLP/delta-MLP case energy calls,
     !                it needs to call ai-qm/mm cutoff energy once in the outer time step.
     !
     use chm_kinds
     use dimens_fcm
     use number
     use memory
     use string
     use bases_fcm
     use datstr
     use code
     use coord
     use energym
     use gamess_fcm
     use inbnd
     use mndo97
     use psf,only: natom
     use qm1_info, only: qmmm_mlp
     use iso_c_binding, only: c_char, c_null_char, c_int ! arat2025sep15
#if KEY_MLPTORCH==1
     use mndo97_dpmm_ciface, only: mndo97_dpmm_pyinternal_setup ! arat2025sep15
#endif
     use stream, only : outu,prnlev
     use parallel

     implicit none
     logical:: qmhub_ml,qmmm_mlp_only,qmhub_python,qmhub_dpmm
     integer:: qmlp_mode,qmlp_lgpu,my_replica
     !!CHARACTER(len=*):: COMLYN
     !!INTEGER ::  COMLEN

     ! local variables
     integer :: LX,LI,LO
     integer :: LM, LC  ! arat2025sep15
     integer :: LGPU, LOMP ! arat2025sep15 for gpu use
     CHARACTER(len=255):: FILIN,FILOUT,FILEXE
     character(len=255):: FILMODEL,FILCTRL ! arat2025sep15
     CHARACTER(len=16)  :: ENVGPU, ENVOMP ! arat2025sep15 for gpu use
     integer :: ier=0
     integer :: i ! arat2025sep15
     integer(c_int), allocatable :: types_mlp_local_c(:)
     if (allocated(types_mlp_local_c)) deallocate(types_mlp_local_c,stat=ier)
     allocate(types_mlp_local_c(256),stat=ier) ! max 16 types for now

     ! memory allocation
     if(qmhub_ml) then
        call mlp_memory_allocate(natom,.true.)

        !
        qmmm_mlp%qmmm_mlp = qmhub_ml           ! use mlp/delta-mlp qm/mm
        qmmm_mlp%qmhub_python  = qmhub_python            !     based on qmhub inteface or
        qmmm_mlp%qmhub_dpmm = qmhub_dpmm           !              pytorch interface

        qmmm_mlp%qmmm_mlp_only = qmmm_mlp_only ! True : only use mlp/delta-mlp qm/mm
                                               ! False: also use MTS ai-qm/mm (QMLAY_high)
        qmmm_mlp%qmlp_mode= qmlp_mode          ! mlp mode: 0: MLP; 1: delta-MLP
        qmmm_mlp%nref_replica=my_replica       ! replica id for the qm region definition
     end if

     if(qmmm_mlp%qmhub_python) then
      !QMHUB (EXTERNAL PYTHON) AS PROGRAM ! arat2025sep15
      FILEXE=''
      call get_environment_variable("PYEXE", FILEXE, LX)
      if(LX == 0) call wrndie(-5,'<setup_qmhub>','Specify: env PYEXE as "python <script.py>" or  "python -m <package_name>" , Failed to read <package>  ...')
   
      !.PT MODEL FILE FOR QMHUB PROGRAM ! arat2025sep15
      FILMODEL=''
      call get_environment_variable("PYMODEL", FILMODEL, LM)
      if(LM == 0) call wrndie(0,'<setup_qmhub>','Specify: env PYMODEL "<ptmodel_filename>", Failed to read ... -pt <ptmodel> ...')      
   
      !QMHUB CONTROL FILE ! arat2025sep15
      FILCTRL=''
      call get_environment_variable("PYCTRL", FILCTRL, LC)
      if(LC == 0) call wrndie(0,'<setup_qmhub>','Specify: env PYCTRL "<modelinfo_filename>", Failed to read ... -ptinfo <modelinfo> ...')

      !QMHUB USE-GPU ! arat2025sep15
!!      ENVGPU=''
!!      call get_environment_variable("PYGPU", ENVGPU, LGPU)
!!      if(LGPU == 0) call wrndie(0,'<setup_qmhub>','Optional, Specify: env PYGPU as GPUID to use for MODEL inference.')
!!      if (LGPU>0) then
!!         read(ENVGPU(1:LGPU), *) qmmm_mlp%use_gpu_mlp
!!       else
!!         qmmm_mlp%use_gpu_mlp = -1
!!      end if
      qmmm_mlp%use_gpu_mlp = qmlp_lgpu   ! gpu id: 0 ~ ; -1 no use gpu
      

      !QMHUB USE-OMP ! arat2025sep15
      ENVOMP=''
      call get_environment_variable("PYOMP", ENVOMP, LOMP)
      if(LOMP == 0) call wrndie(0,'<setup_qmhub>','Optional, Specify: env PYOMP as number of OMP THREADS to run PYTHON package with OMP parallelization.')
      if (LOMP>0) then
         read(ENVOMP(1:LOMP), *) qmmm_mlp%use_omp_mlp
       else
         qmmm_mlp%use_omp_mlp = -1
      end if
      
      FILIN=''
      call get_environment_variable("PYINP", FILIN, LI)
      if(prnlev >=2) write(outu,*) "INPUT FILE read successfully:", FILIN
      if(LI == 0) call wrndie(-5,'<setup_qmhub>','No input specified. Refer documentation for input file format generated by CHARMM for Python. Specify: env PYINP "<input_filename>"')
   
      call get_environment_variable("PYOUT", FILOUT, LO)
      if(LO == 0) call wrndie(-5,'<setup_qmhub>','No output specified. Refer documentation for output file format expected by CHARMM from Python. Specify: env PYOUT "<output_filename>"')


      ! Assign to the qmmm_mlp structure

      call mlp_memory_allocate_pyexternal(.true.,LX,LM,LC,LI,LO)
      qmmm_mlp%package_mlp = FILEXE(1:LX)
      qmmm_mlp%model_mlp = FILMODEL(1:LM)
      qmmm_mlp%ctrl_mlp  = FILCTRL(1:LC) 
      qmmm_mlp%filein_mlp = FILIN(1:LI)
      qmmm_mlp%fileout_mlp = FILOUT(1:LO)


   end if

   if(qmmm_mlp%qmhub_dpmm) then
      !.PT MODEL FILE FOR QMHUB PROGRAM ! arat2025sep15
      FILMODEL=''
      call get_environment_variable("PYMODEL", FILMODEL, LM)
      if(LM == 0) call wrndie(-5,'<setup_qmhub>','Specify: env PYMODEL "<ptmodel_filename>"')       
   
      !QMHUB CONTROL FILE ! arat2025sep15
      FILCTRL=''
      call get_environment_variable("PYCTRL", FILCTRL, LC)
      if(LC == 0) call wrndie(-5,'<setup_qmhub>','Specify: env PYCTRL "<modelinfo_filename>"')

      !QMHUB USE-GPU ! arat2025sep15
      ENVGPU=''
!!      call get_environment_variable("PYGPU", ENVGPU, LGPU)
!!      if(LGPU == 0) call wrndie(0,'<setup_qmhub>','Optional, Specify: env PYGPU as GPUID to use for MODEL inference.')
!!      if (LGPU>0) then
!!         read(ENVGPU(1:LGPU), *) qmmm_mlp%use_gpu_mlp_c
!!       else
!!         qmmm_mlp%use_gpu_mlp_c = -1
!!      end if
      qmmm_mlp%use_gpu_mlp_c = qmlp_lgpu   ! gpu id: 0 ~ ; -1 no use gpu

      !QMHUB USE-OMP ! arat2025sep15
      ENVOMP=''
      call get_environment_variable("PYOMP", ENVOMP, LOMP)
      if(LOMP == 0) call wrndie(0,'<setup_qmhub>','Optional, Specify: env PYOMP as number of OMP THREADS to run PYTHON package with OMP parallelization.')
      if (LOMP>0) then
         read(ENVOMP(1:LOMP), *) qmmm_mlp%use_omp_mlp_c
       else
         qmmm_mlp%use_omp_mlp_c = -1
      end if

      ! call memory allocation for char arrays
      call mlp_memory_allocate_pyinternal_chararrays(.true., LM, LC)
      qmmm_mlp%model_mlp_c = [ character(kind=c_char) :: (FILMODEL(i:i), i=1,LM), c_null_char ]
      qmmm_mlp%ctrl_mlp_c  = [ character(kind=c_char) :: (FILCTRL(i:i),  i=1,LC), c_null_char ]

#if KEY_PARALLEL==1
      if(mynod==0) then
#endif
#if KEY_MLPTORCH==1
         ! call mndo97_dpmm_pyinternal_setup to setup the model and control files
         call mndo97_dpmm_pyinternal_setup(qmmm_mlp%model_mlp_c, qmmm_mlp%ctrl_mlp_c, qmmm_mlp%use_gpu_mlp_c, &
                                          qmmm_mlp%use_omp_mlp_c, types_mlp_local_c, qmmm_mlp%ntypes_mlp_c, & 
                                          qmmm_mlp%qmmax_mlp_c, qmmm_mlp%mmmax_mlp_c, qmmm_mlp%qmmm_mlp_setup_err)
         if (qmmm_mlp%qmmm_mlp_setup_err /= 0) then
            call wrndie(-5,'<setup_qmhub>','MLP/LibTorch setup failed. Please check the model and control files.')
         end if
#endif
         ! allocate arrays based on the ntypes, qmmax, mmmax values into the qmmm_mlp structure
         call mlp_memory_allocate_pyinternal(.true., qmmm_mlp%ntypes_mlp_c, qmmm_mlp%qmmax_mlp_c, qmmm_mlp%mmmax_mlp_c)
         qmmm_mlp%types_mlp_c = types_mlp_local_c(1:qmmm_mlp%ntypes_mlp_c)
#if KEY_PARALLEL==1
      end if
#endif
   end if
#if KEY_MLPTORCH==0
      call wrndie(-5,'<setup_qmhub>','MLP/LibTorch support is not enabled in this build. Please recompile with --with-torch to enable MLP/LibTorch support FOR DPMM.')
#endif
   return
  end subroutine setup_qmhub

  subroutine qmhub_energy(E_MLP,x,y,z,dx,dy,dz,cgx,natom)
     ! 
     ! calculation of MLP/delta-MLP energy
     !
     ! note
     ! for MLP case:  we will still call qm/mm-pme for the long-range contribution
     !                and substract qm/mm energy within the cutoff, plus MLP energy call for
     !                MLP contribution within the cutoff.
     ! E_qm/mm-total = E_qm/mm-pme + (E_MLP (cutoff) - E_qm/mm (cutoff))
     ! 
     !                so, first call the qm/mm-pme energy, followed by MLP & qm/mm cutoff energies
     !
     ! for delta-MLP case:
     !               first, call qm/mm-pme for the entire system's energy, plus
     !               MLP energy call for delta-MLP contribution.
     ! E_qm/mm-total = E_qm/mm-pme + E_delta-MLP (cutoff)
     !
     !                so, first call the qm/mm-pme energy, followed by delta-MLP cutoff energy
     !
     ! now, for MTP ai-qm/mm case with MLP
     ! for MLP
     ! E_qm/mm-total = [ E_se-qm/mm-pme + (E_MLP (cutoff) - E_se-qm/mm (cutoff)) ]
     !                +[ E_ai-qm/mm (cutoff) - E_MLP (cutoff) ]
     ! 
     ! for delta-MLP
     ! E_qm/mm-total = [ E_qm/mm-pme + E_delta-MLP (cutoff) ]
     !                +[ E_ai-qm/mm (cutoff) - [E_se-qm/mm (cutoff) + E_delta-MLP (cutoff)] ]
     !
     !                so, in both, in addition to the MLP/delta-MLP case energy calls,
     !                it needs to call ai-qm/mm cutoff energy once in the outer time step.
     use chm_kinds
     use dimens_fcm
     use consta
     use number
     use gamess_fcm
     use stream
     use mndo97
     use parallel
     use scalar_module
     use qm1_info, only : qm_control_c,qm_main_c,qmmm_mlp
     use gukini_mod
     use image,only:xucell
     use stream, only : outu,prnlev

     use iso_c_binding, only: c_double, c_int
#if KEY_MLPTORCH==1
     use mndo97_dpmm_ciface, only: mndo97_dpmm_pyinternal ! arat2025sep15
#endif

     implicit none
     integer :: natom
     real(chm_real) :: E_MLP,x(natom),y(natom),z(natom),dx(natom),dy(natom),dz(natom)

     character(len=4096) :: CMDMLP !arat2026jul5
     character(len=16)  :: ENVGPU, ENVOMP ! arat2025sep15 for gpu use
     integer :: LGPU, LOMP ! arat2025sep15 for gpu use
     
     real(c_double) :: E_c ! arat2025sep15

     real(chm_real) :: cgx(natom)

     ! local variables
     integer :: I,natqm_2,mm,mmatoms,qmatoms
     integer :: OMO,IUNITFI,IUNITFO,ios  ! arat2025sep15 
     character(len=10) :: char_qmatoms, char_mmatoms
     character(len=300):: line
     integer           :: mapqm(natom),mapmm(natom),ipt,jpt            ! lw050728
     real(chm_real)    :: E,dumx,dumy,dumz,de_conv
     integer,save       :: icnt_mlp =0
     integer,parameter  :: icnt_step=100
     integer :: ier=0
     
     ! counter qm atoms
     natqm_2 = qm_main_c%numat


     ! only done by the master node
#if KEY_PARALLEL==1
     !The following call should be after the call system() line, but
     !that one is not executed on CPUs other than 0. So we call it
     !here, because this routine must be called by everyone!
#if KEY_CMPI==1 || KEY_MPI==1
     !if(mynod > 0) call break_busy_wait_mpi()
#endif
     !if(mynod > 0) then
     !   E_MLP=ZERO
     !  return
     !end if
#endif

      if(qmmm_mlp%qmhub_python) then
         OMO    = 91
         IUNITFI=254
         open(unit=IUNITFI,file=trim(qmmm_mlp%filein_mlp),status='replace')   ! PYINP file
      endif
        ! mapqm(1:natqm)   = qm-atom no. in the main (x/y/z) array.
        ! mapmm(1:mmatoms) = mm-atom no. in the main array.
      if(qmmm_mlp%qmhub_python .or. qmmm_mlp%qmhub_dpmm) then
        ! count no. of mm atoms
         qmatoms= natqm_2
         mmatoms= 0
         do i=natqm_2+1,natom
            if(qm_control_c%mminb1(i) > 0) mmatoms = mmatoms + 1
         end do
      endif

      if(qmmm_mlp%qmhub_python) then
        ! printing file for qmhub style
        ! if(prnlev >= 2) then
        !    write(outu,'(A)')  "========================="
        !    write(outu,'(A,I7) "No of QM Atoms    : ", qmatoms
        !    write(outu,'(A,I7) "No of MM Atoms    : ", mmatoms
        !    write(outu,'(A)')  "========================="
        ! end if

        ! IUNITFI
        ! write header
        ! Convert integers to character strings
         write(char_qmatoms, '(I0)') qmatoms
         write(char_mmatoms, '(I0)') mmatoms

        ! initialize iostat
         ios = 0
         write(IUNITFI, '(A,1X,A)', iostat=ios) trim(char_qmatoms), trim(char_mmatoms) ! arat2025sep15
         if (ios /= 0) then
            write(outu,'(A,I7)') 'Error writing to IUNITFI, iostat =',ios
            call wrndie(-5,'<qmhub_energy>',' File error for QMHUB. Stop')  ! dose it work okay for parallel?
         end if
      endif
      
      if(qmmm_mlp%qmhub_python .or. qmmm_mlp%qmhub_dpmm) then
        ! print qm atoms
        do i=1,natqm_2
           mm       = iabs(qm_control_c%qminb(i))
           mapqm(i) = mm

           if (qmmm_mlp%qmhub_dpmm) then
               ! allocate arrays
               qmmm_mlp%qmx_mlp_c(i)     = x(mm)
               qmmm_mlp%qmy_mlp_c(i)     = y(mm)
               qmmm_mlp%qmz_mlp_c(i)     = z(mm)
               qmmm_mlp%qm_Z_mlp_c(i)    = qm_main_c%nat(i)
           end if

           if (qmmm_mlp%qmhub_python) then
            write(IUNITFI,100) x(mm),y(mm),z(mm),qm_control_c%cgqmmm(mm),qm_main_c%nat(i)  
           end if
        end do

        ! print mm atoms
        ipt    = 0
        do i=natqm_2+1,natom
           mm = qm_control_c%mminb1(i)
           if(mm>0) then
              ipt        = ipt + 1      
              mapmm(ipt) = mm

               if (qmmm_mlp%qmhub_dpmm) then
                  ! allocate arrays
                  qmmm_mlp%mmx_mlp_c(ipt)  = x(mm)
                  qmmm_mlp%mmy_mlp_c(ipt)  = y(mm)
                  qmmm_mlp%mmz_mlp_c(ipt)  = z(mm)
                  qmmm_mlp%mmcg_mlp_c(ipt) = cgx(mm)
               end if

              if (qmmm_mlp%qmhub_python) then
               write(IUNITFI,200) x(mm),y(mm),z(mm),cgx(mm)
              endif 
           end if
        end do
        if (qmmm_mlp%qmhub_python) then
        ! close file & input is ready
         close(IUNITFI)
         endif
      endif

      write(ENVGPU,'(I0)') qmmm_mlp%use_gpu_mlp
      write(ENVOMP,'(I0)') qmmm_mlp%use_omp_mlp

      if (qmmm_mlp%qmhub_python) then

         CMDMLP = ''

         ! GPU environment
         if (qmmm_mlp%use_gpu_mlp < 0) then
            CMDMLP = 'CUDA_VISIBLE_DEVICES="" '
         else
            CMDMLP = 'CUDA_VISIBLE_DEVICES="'//trim(adjustl(ENVGPU))//'" '
         end if

         ! OMP environment
         if (qmmm_mlp%use_omp_mlp <= 1) then
            CMDMLP = trim(CMDMLP)//' OMP_NUM_THREADS=1 '
         else
            CMDMLP = trim(CMDMLP)//' OMP_NUM_THREADS="'//trim(adjustl(ENVOMP))//'" '
         end if

         ! package_mlp should be something like:
         !   python -m module.name
         ! or
         !   python script.py
         !
         CMDMLP = trim(CMDMLP)//' '//trim(qmmm_mlp%package_mlp)

         ! Optional arguments
         if (len_trim(qmmm_mlp%model_mlp) > 0) then
            CMDMLP = trim(CMDMLP)//' -pymodel "'//trim(qmmm_mlp%model_mlp)//'"'
         end if

         if (len_trim(qmmm_mlp%ctrl_mlp) > 0) then
            CMDMLP = trim(CMDMLP)//' -pyctrl "'//trim(qmmm_mlp%ctrl_mlp)//'"'
         end if

         ! Mandatory arguments
         CMDMLP = trim(CMDMLP)//' -pyinp "'//trim(qmmm_mlp%filein_mlp)//'"'
         CMDMLP = trim(CMDMLP)//' -pyout "'//trim(qmmm_mlp%fileout_mlp)//'"'

         call system(trim(CMDMLP))

        ! The following routine must be called after call system('pythod ...').
        ! This is executed only on processor 0 here, so it must have corresponding
        ! call elsewhere. Currently in the beginning of this routine!
#if KEY_PARALLEL==1
#if KEY_CMPI==1 || KEY_MPI==1
        ! write(*,*)'calling break_busy_wait: L2393' 
        !if(mynod > 0) call break_busy_wait_mpi()
#endif
#endif
        !

        !------------------------------------------------------
        !---- Get the energy and forces from QMHub output -----
        !------------------------------------------------------
        E =ZERO
        open(omo,FILE=trim(qmmm_mlp%fileout_mlp),status='old')

        ! energy value
        ! read(omo,'(A)',end=300) E
        read(omo,*,end=300) E
        if(prnlev>= 2) write(outu,500) 'QMHub: Energy read successfully. E value:',E ! *tokcal
        E_MLP = E ! *tokcal  ! in kcal/mol unit

        ! the order of the gradients needs to be checked with Raafik. 
        ! qm gradients
        de_conv = TOKCAL/BOHRR
        do i=1,natqm_2
           read(omo,'(A)',end=300) line
           read(line,*) dumx,dumy,dumz
           mm     = mapqm(i)
           dx(mm) = dx(mm) + dumx ! *de_conv  ! (dumx*TOKCAL/BOHRR)
           dy(mm) = dy(mm) + dumy ! *de_conv  ! (dumy*TOKCAL/BOHRR)
           dz(mm) = dz(mm) + dumz ! *de_conv  ! (dumz*TOKCAL/BOHRR)
        end do
        !if(prnlev>= 2) write(outu,400) 'QMHub: QM Forces read successfully'

        ! mm gradients
        do i=1,mmatoms
           read(omo,'(A)',end=300) line
           read(line,*) dumx,dumy,dumz
           mm     = mapmm(i)
           dx(mm) = dx(mm) + dumx ! *de_conv  ! dumx*TOKCAL/BOHRR ! *(-CGX(ipt))
           dy(mm) = dy(mm) + dumy ! *de_conv  ! dumy*TOKCAL/BOHRR ! *(-CGX(ipt))
           dz(mm) = dz(mm) + dumz ! *de_conv  ! dumz*TOKCAL/BOHRR ! *(-CGX(ipt))
        end do
        !if(prnlev>= 2) write(outu,400) 'QMHub: MM Forces read successfully'

300     close(omo) ! end of the output file

      endif


if (qmmm_mlp%qmhub_dpmm) then
#if KEY_MLPTORCH==1
   E_c = 0.0_c_double
   ! call mndo97_dpmm_pyinternal to get the energy and forces
   call mndo97_dpmm_pyinternal(int(natqm_2, kind=c_int), int(mmatoms, kind=c_int), qmmm_mlp%mmmax_mlp_c, &
        qmmm_mlp%types_mlp_c, qmmm_mlp%use_gpu_mlp_c, qmmm_mlp%use_omp_mlp_c, &
        qmmm_mlp%qmx_mlp_c, qmmm_mlp%qmy_mlp_c, qmmm_mlp%qmz_mlp_c, qmmm_mlp%qm_Z_mlp_c, &
        qmmm_mlp%mmx_mlp_c, qmmm_mlp%mmy_mlp_c, qmmm_mlp%mmz_mlp_c, qmmm_mlp%mmcg_mlp_c, &
        E_c, qmmm_mlp%qmdx_mlp_c, qmmm_mlp%qmdy_mlp_c, qmmm_mlp%qmdz_mlp_c, &
        qmmm_mlp%mmdx_mlp_c, qmmm_mlp%mmdy_mlp_c, qmmm_mlp%mmdz_mlp_c)

   E_MLP = real(E_c, kind=chm_real)     
   if(prnlev>= 2) then
      if(mod(icnt_mlp,icnt_step) == 0) write(outu,500) &
                                      'QMHub(DPMM with LibTorch): Energy read successfully. E value:',E_MLP ! *tokcal
   end if
   icnt_mlp = icnt_mlp + 1
   !if(prnlev>= 2) write(outu,500) 'QMHub(DPMM with LibTorch): Energy read successfully. E value:',E_MLP ! *tokcal

   ! add forces to the main array
   de_conv = TOKCAL/BOHRR
   do i=1,natqm_2
      mm     = mapqm(i)
      dx(mm) = dx(mm) + real(qmmm_mlp%qmdx_mlp_c(i), kind=chm_real) ! *de_conv  ! (dumx*TOKCAL/BOHRR)
      dy(mm) = dy(mm) + real(qmmm_mlp%qmdy_mlp_c(i), kind=chm_real) ! *de_conv  ! (dumy*TOKCAL/BOHRR)
      dz(mm) = dz(mm) + real(qmmm_mlp%qmdz_mlp_c(i), kind=chm_real) ! *de_conv  ! (dumz*TOKCAL/BOHRR)
   end do
   !if(prnlev>= 2) write(outu,400) 'QMHub(DPMM with LibTorch): QM Forces read successfully'
   do i=1,mmatoms
      mm     = mapmm(i)
      dx(mm) = dx(mm) + real(qmmm_mlp%mmdx_mlp_c(i), kind=chm_real) ! *de_conv  ! dumx*TOKCAL/BOHRR ! *(-CGX(ipt))
      dy(mm) = dy(mm) + real(qmmm_mlp%mmdy_mlp_c(i), kind=chm_real) ! *de_conv  ! dumy*TOKCAL/BOHRR ! *(-CGX(ipt))
      dz(mm) = dz(mm) + real(qmmm_mlp%mmdz_mlp_c(i), kind=chm_real) ! *de_conv  ! dumz*TOKCAL/BOHRR ! *(-CGX(ipt))
   end do
   !if(prnlev>= 2) write(outu,400) 'QMHub(DPMM with LibTorch): MM Forces read successfully'
#endif
#if KEY_MLPTORCH==0
   call wrndie(-5,'<qmhub_energy>','MLP/LibTorch support is not enabled in this build. Please recompile with --with-torch to enable MLP/LibTorch support FOR DPMM.')
#endif
end if


100  format(F20.10,F20.10,F20.10,F16.8,I4)
200  format(F16.8 ,F16.8 ,F16.8 ,F16.8)
400  format(' qmhub_energy> ',A)
500  format(' qmhub_energy> ',A,F20.10)


     return
  end subroutine qmhub_energy

#endif            /*mndo97*/
end module mndo97_mlp
