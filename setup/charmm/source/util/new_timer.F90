module new_timer
  use chm_kinds
  implicit none

  !
  integer,parameter :: MAXTIME=200, MAXTREE=10

  integer,dimension(MAXTIME),private,save ::  &
       Tpar_P,Tchild_P,Tsib_P,T_level,T_state, &
       Tnext_P,Tprev_P
  integer,dimension(-1:MAXTIME),private,save :: Tpar_level

  integer,private,save :: time_current_level

  real(kind=chm_real),dimension(MAXTIME),private,save ::  &
       T_accum,Tch_acc,T_curr

  character(len=20),dimension(MAXTIME),private,save :: T_string

  integer,private,save :: T_numadd,T_numtree
  integer,dimension(MAXTREE),private,save :: T_first,T_last

  integer,public,save ::  &
       T_total,           & ! Total time
       T_dcntrl,          & ! Dynamics total
       T_shake,           & ! Shake time
       T_shakesetup,      & ! Shake Setup
       T_energy,          & ! Energy time
       T_nonbon,          & ! Nonbond force
       T_list,            & ! List time
       T_ewald,           & ! Ewald time
       T_skin,            & ! 
       T_images,    & !10     Image Update
       T_heurcheck,       & ! Heuristic check
       T_nbonds,          & ! NBONDS
       T_map,             & ! Pbound cube setup
       T_setgrd,          & ! xdistm setup
       T_grduc,           & ! dist vs cutoff
       T_grdim,           & ! xdistm sort
       T_bldlst,          & ! xdistm Build list
       T_dir,             & ! Direct Ewald time
       T_uu,              & ! Direct U-U
       T_uv,              & ! Direct U-V
       T_vv,              & ! Direct V-V
       T_rec,             & ! Recip Ewald time
       T_adj,             & ! Adjust Ewald time
       T_self,            & ! Self Ewald time
       T_nbvir,     & !20     Finish NB virial
       T_bspl,            & ! Fill Bspline coeffs
       T_fillg,           & ! Fill charge grid
       T_scsum,           & ! Scalar sum
       T_grads,           & ! Grad sum
       T_FFT,             & ! FFT
       T_FFTcomm,         & ! FFTcomm
       T_ips    ,         & ! IPS
       T_ipsex,           & ! IPS excluded pairs
       T_ipsnb,           & ! IPS nb pairs
       T_ipsa,       & !30    IPS anisotropic contribution
       T_ipsafunc,        & ! IPS anisotropic function
       T_ipsagrid,        & ! IPS anisotropic grid
       T_ipsasum,         & ! IPS anisotropic sum
       T_ipsaforce,       & ! IPS anisotropic force
       T_ipsafft,         & ! IPS anisotropic FFT
       T_inte,            & ! INTRNL energy
       T_bond,            & ! Bond energy
       T_angle,           & ! Angle energy
       T_ephi,            & ! Dihedral energy
       T_restr,      & !40    Restraints energy
       T_ecomm,           & ! Comm energy
       T_fcomm,           & ! Comm force
       T_crdcomm,         & ! Comm coords
#if KEY_DOMDEC==1
       T_crdcomm2,        & ! Comm coords between direct and recip 
#endif
#if KEY_DOMDEC==1
       T_crdcomm3,        & ! Comm coords between direct and recip 
#endif
       T_d2r,             & ! 
       T_d2r_recv_size,   & ! Recv size
       T_d2r_recv,        & ! Recv
       T_d2r_unpack,      & ! Unpack
       T_r2r,             & ! Recip <-> Recip
#if KEY_DOMDEC==1
       T_fcomm2,          & ! Dir/Recip comm force 
#endif
       T_fcomm2wait,      & ! Dir/Recip comm wait
       T_dlb,             & ! Dynamic load balance
       T_gborn,           & ! GB time
       T_dynamc,          & ! dynamc
       T_constp,          & ! Constant pressure
       T_constt,          & ! Constant temperature
       T_commvar,         & ! Communicate variables
       T_saveshake,       & ! Save shake coordinates
       T_heur,            & ! Heuristic check
       T_tvir,            & ! Temp and virial calc
       T_GB_RADII,        & ! GB Radii
       T_GB_ENERG,        & ! GB Energy
       T_GB_FORCE,        & ! GB Force
       T_ELECVDW,         & ! Nonbond using enbfs8
       T_facts,     & !50     FACTS energy
#if KEY_DIMS==1
       T_DIMS,            & ! DIMS  
#endif
#if KEY_DIMS==1
       T_DIMNM,           & ! DIMNM, Hessian + Diag   
#endif
#if KEY_DIMS==1
       T_DIMCOMB,         & ! DIM Combinatorial tests 
#endif
#if KEY_ENSEMBLE==1
       T_EVB,             & ! Total EVB routine
       T_EVB_Comms,       & ! Communication time
       T_EVB_Energy,      & ! Lowest energy eigenvector (matrix diagonalization)
       T_EVB_Forces,      &   ! Gradients of EVB matrix
#endif 
       T_DCM,             & ! DCM energies + forces
       T_1extra,T_2extra,T_3extra,T_4extra,T_5extra, &  !60
       T_6extra,T_7extra,T_8extra,T_9extra,T_10extra, & 
       T_omm, T_omm_load, T_omm_sys, T_omm_ctx, &
       T_blade

  !                      ^^^^
  !-----------------------------------  MAX is 99 as specified above -----
  !


contains

  !------------------------------------------------------

#if KEY_DOMDEC==1
  ! *
  ! * Does gcomb for direct and recip separately
  ! *
  subroutine comm_buf(buf, q_direct_node, comm_direct, comm_recip)
    use mpi
    use memory
    use parallel
    implicit none
    ! Input / Output
    real(chm_real) buf(MAXTIME+1)
    logical, intent(in) :: q_direct_node
    integer, intent(in) :: comm_direct, comm_recip
    ! Variables
    integer comm_charmm_save

    comm_charmm_save = comm_charmm
    if (q_direct_node) then
       comm_charmm = comm_direct
    else
       comm_charmm = comm_recip
    endif

    call gcomb(buf,MAXTIME)

    comm_charmm = comm_charmm_save

    return
  end subroutine comm_buf

#endif /* KEY_DOMDEC */

# if KEY_DOMDEC==1
  subroutine write_timers(q_split, ndirect, nrecip, q_direct_node, q_recip_node, &
                          comm_direct, comm_recip, q_root)
# else /* KEY_DOMDEC */
  subroutine write_timers()
# endif /* KEY_DOMDEC */
    use stream
#if KEY_PARALLEL==1
    use parallel      
#endif
    ! VO : limit output when running string; otherwise get nearly identical output from each replica
#if KEY_MULTICOM==1 && KEY_STRINGM==1
    use multicom_aux
    use mpi, only : MPI_UNDEFINED
#endif
    use number
    implicit none
#   if KEY_DOMDEC==1
    logical, intent(in) :: q_split, q_direct_node, q_recip_node
    integer, intent(in) :: ndirect, nrecip, comm_direct, comm_recip
    logical, intent(in) :: q_root ! True if my rank (mynod_split) is 0 after direc/recip split
#   endif
    integer i,j,k
    real(chm_real) t1,t2,eps,fact
    real(chm_real) buf(MAXTIME+1)
    logical :: qoutput=.true. ! VO string

#if KEY_STRINGM==1
    qoutput = (ME_LOCAL.eq.0 .and. ( ME_STRNG.eq.0 .or. ME_STRNG.eq.MPI_UNDEFINED ))
#else
#if KEY_PARALLEL==1
    qoutput = (mynod == 0)
#endif
#endif

#if KEY_PARALLEL==1
    do i = 1,MAXTIME
       buf(i) = T_accum(i)
    enddo
#if KEY_DOMDEC==1
    if (q_split) then
       ! DOMDEC direct/reciprocal split. Accumulate on each subcommunicator separately.
       call comm_buf(buf, q_direct_node, comm_direct, comm_recip)
       ! All except root nodes of direct and recip are free to leave
       if (.not.q_root) then
          ! This psync is used to sync with direct and recip root nodes
          ! before returning
          call psync()
          return
       endif
    else
       ! In this case split_direct_recip has not been called
       call gcomb(buf,MAXTIME)
       if (.not.qoutput) return
    endif
#else /* KEY_DOMDEC */
    call gcomb(buf,MAXTIME)
    if (.not.qoutput) return
#endif 

    eps = 1.d-7
#if KEY_DOMDEC==1
    if (q_split) then
       if (.not.q_direct_node .and. q_recip_node) then
          ! Reciprocal root node waits here until direct node is done with writing
          call psync()
       endif
       if (q_direct_node) then
          write (outu,'(a)') '      $$$$$$  New timer profile Local Direct node$$$$$'
       else
          write (outu,'(a)') '      $$$$$$  New timer profile Local Reciprocal node$$$$$'
       endif
    else
       write (outu,'(a)') '      $$$$$$  New timer profile Local node$$$$$'
    endif
#else /**/
    write(outu,61)
61  format(/'      $$$$$$  New timer profile Local node$$$$$'/)
#endif 
    do k = 1,T_numtree
       i = T_first(k)
11     continue
       t1 = T_accum(i)
       if ( Tch_acc(i) > t1 )then
          t1 = Tch_acc(i)
          t2 = 0.d0
       elseif ( Tch_acc(i) > zero )then
          t2 = t1 - Tch_acc(i)
       else
          t2 =zero
       endif
       if ( t1 > eps )then
          j = T_level(i)
          if ( j  ==  1 )then
             write(OUTU,67)T_string(i),t1,t2
          elseif ( j  ==  2 )then
             write(OUTU,68)T_string(i),t1,t2
          elseif ( j  ==  3 )then
             write(OUTU,69)T_string(i),t1,t2
          elseif ( j  ==  4 )then
             write(OUTU,70)T_string(i),t1,t2
          elseif ( j  ==  5 )then
             write(OUTU,71)T_string(i),t1,t2
          elseif ( j  ==  6 )then
             write(OUTU,72)T_string(i),t1,t2
          elseif ( j  ==  7 )then
             write(OUTU,73)T_string(i),t1,t2
          elseif ( j  ==  8 )then
             write(OUTU,74)T_string(i),t1,t2
          endif
       endif
       if ( Tnext_P(i)  /=  0 )then
          i = Tnext_P(i)
          goto 11
       endif
    enddo
#if KEY_DOMDEC==1
    if (q_split) then
       if (q_direct_node) then
          fact=one/ndirect
          write(OUTU,'(a)') '         $$$$$$  Average   profile Direct $$$$$'
       else
          fact=one/nrecip
          write(OUTU,'(a)') '         $$$$$$  Average   profile Reciprocal $$$$$'
       endif
    else
       fact=one/numnod
       do i = 1,MAXTIME
          T_accum(i) = buf(i) * fact
       enddo
       write(OUTU,'(a)') '         $$$$$$  Average   profile $$$$$'
    endif
    do i = 1,MAXTIME
       T_accum(i) = buf(i) * fact
    enddo
#else /**/
    fact=one/numnod
    do i = 1,MAXTIME
       T_accum(i) = buf(i) * fact
    enddo
    write(OUTU,60)
60  format(/'         $$$$$$  Average   profile $$$$$'/)
#endif 
#else /**/
    write(OUTU,60)
60  format('$$$$$$  New timer profile $$$$$')
#endif 

    eps = 1.d-7
    do k = 1,T_numtree
       i = T_first(k)
10     continue
       t1 = T_accum(i)
       if ( Tch_acc(i)  >  t1 )then
          t1 = Tch_acc(i)
          t2 = 0.d0
       elseif ( Tch_acc(i)  >  0.d0 )then
          t2 = t1 - Tch_acc(i)
       else
          t2 = 0.d0
       endif
       if ( t1  >  eps )then
          j = T_level(i)
          if ( j  ==  1 )then
             write(OUTU,67)T_string(i),t1,t2
          elseif ( j  ==  2 )then
             write(OUTU,68)T_string(i),t1,t2
          elseif ( j  ==  3 )then
             write(OUTU,69)T_string(i),t1,t2
          elseif ( j  ==  4 )then
             write(OUTU,70)T_string(i),t1,t2
          elseif ( j  ==  5 )then
             write(OUTU,71)T_string(i),t1,t2
          elseif ( j  ==  6 )then
             write(OUTU,72)T_string(i),t1,t2
          elseif ( j  ==  7 )then
             write(OUTU,73)T_string(i),t1,t2
          elseif ( j  ==  8 )then
             write(OUTU,74)T_string(i),t1,t2
          endif
       endif
67     format(1x,a,1x,f15.2,' Other: ',f15.2)
68     format(3x,a,1x,f15.2,' Other: ',f15.2)
69     format(6x,a,1x,f15.2,' Other: ',f15.2)
70     format(9x,a,1x,f15.2,' Other: ',f15.2)
71     format(12x,a,1x,f15.2,' Other: ',f15.2)
72     format(15x,a,1x,f15.2,' Other: ',f15.2)
73     format(18x,a,1x,f15.2,' Other: ',f15.2)
74     format(19x,a,1x,f15.2,' Other: ',f15.2)
       if ( Tnext_P(i)  /=  0 )then
          i = Tnext_P(i)
          goto 10
       endif
    enddo
#if KEY_DOMDEC==1
    if (q_split .and. q_direct_node) then
#if KEY_PATHSCALE==1
       call & 
#endif
       flush(outu)
       call psync()
    endif
#endif 
    return
  end   subroutine write_timers
  !------------------------------------------------------
  subroutine mexit(i,msg)

    integer i
    character(len=*) msg
    CALL WRNDIE(-1,'New_Timers', &
         msg)
    return
  end  subroutine mexit
  !------------------------------------------------------
  subroutine init_timers()
    use number
    implicit none
    integer i,success

    T_numadd = 0
    time_current_level=0
    do i = 1,MAXTIME 
       Tpar_P(i) = 0
       Tchild_P(i) = 0
       Tsib_P(i) = 0
       T_level(i) = 0
       T_state(i) = 0
       Tnext_P(i) = 0
       Tprev_P(i) = 0
       Tpar_level(i)=0
    enddo
    do i = 1,MAXTIME 
       T_accum(i) = zero
       Tch_acc(i) = zero
       T_curr(i) = zero
    enddo
    do i = 1,MAXTIME 
       T_string(i) = ' '
    enddo
    call add_timer(T_total,-1,'Total time')


    !------ Setup ----------------------------------------
    call add_timer(T_shakesetup,T_total,'Shake Setup')
    call add_timer(T_10extra,T_total,'First List')

    !------ SHAKE ----------------------------------------
    call add_timer(T_shake,T_total,'Shake time')

    !------ dcntrl/dynamics ----------------------------------------
    call add_timer(T_dcntrl,T_total,'Dynamics total ')
    call add_timer(T_crdcomm,T_dcntrl,'Comm coords')
    call add_timer(T_dynamc,T_dcntrl,'dynamc')
    !        call add_timer(T_2extra,T_dcntrl,'shtuff')
#if KEY_DIMS==1
    !
    !  ------------ DIMS -----------------------------------------------
    call add_timer(T_DIMS,T_dynamc,'Dims Time')
    call add_timer(T_DIMNM,T_DIMS,'NM Time')
    call add_timer(T_DIMCOMB,T_DIMS,'DIMS-C Time')
#endif 
    !
    !  ------ List  -------------------------------------------------
    call add_timer(T_list,T_total,'List time')
    call add_timer(T_skin,T_list,' ')
    call add_timer(T_images,T_list,'Image Update')
    call add_timer(T_map,T_list,'Pbound cube setup')
    call add_timer(T_setgrd,T_list,'xdistm setup')
    call add_timer(T_bldlst,T_list,'xdistm Build list')
    call add_timer(T_heurcheck,T_list,'Heuristic check')
    call add_timer(T_nbonds,T_list,'NBONDS')
    call add_timer(T_grdim,T_bldlst,'xdistm sort')
    call add_timer(T_grduc,T_bldlst,'dist vs cutoff')

    !------ energy -------------------------------------------------
    ! Direct node timers
    call add_timer(T_energy,T_total,'Energy time')
    call add_timer(T_nonbon,T_energy,'Nonbond force')
    call add_timer(T_ELECVDW,T_nonbon,'Electrostatic & VDW')
    call add_timer(T_gborn,T_nonbon,'Gen Born time')
    call add_timer(T_GB_RADII,T_gborn,'GB Radii')
    call add_timer(T_GB_ENERG,T_gborn,'GB Energy')
    call add_timer(T_GB_FORCE,T_gborn,'GB Force')
    call add_timer(T_1extra,t_gborn,'GB x1')
    call add_timer(T_3extra,t_gborn,'GB x3')
    call add_timer(T_4extra,t_gborn,'GB x4')
    call add_timer(T_5extra,t_gborn,'GB x5')
    call add_timer(T_facts,T_nonbon,'FACTS time')
    call add_timer(T_ewald,T_nonbon,'Ewald time')
    call add_timer(T_dir,T_ewald,'Direct Ewald time')
    call add_timer(T_uu,T_dir,'Direct U-U')
    call add_timer(T_uv,T_dir,'Direct U-V')
    call add_timer(T_vv,T_dir,'Direct V-V')
    call add_timer(T_2extra,T_dir,'Update cellcrd')
    !               call add_timer(T_3extra,T_dir,'vdw')
    !               call add_timer(T_4extra,T_dir,'elec')
    !               call add_timer(T_5extra,T_dir,'preprocess')
    call add_timer(T_rec,T_ewald,'Recip Ewald time')
    call add_timer(T_bspl,T_rec,'Fill Bspline coeffs')
    call add_timer(T_fillg,T_rec,'Fill charge grid')
    call add_timer(T_scsum,T_rec,'Scalar sum')
    call add_timer(T_grads,T_rec,'Grad sum')
    call add_timer(T_FFT,T_rec,'FFT')
    call add_timer(T_FFTcomm,T_FFT,'FFTcomm')
    call add_timer(T_adj,T_ewald,'Adjust Ewald time')
    call add_timer(T_self,T_ewald,'Self Ewald time')
    call add_timer(T_nbvir,T_ewald,'Finish NB virial')
    call add_timer(T_ips,T_nonbon,'IPS total')
    call add_timer(T_ipsex,T_ips,'IPS excluded pair')
    call add_timer(T_ipsnb,T_ips,'IPS nb pair')
    call add_timer(T_ipsa,T_ips,'AIPS part')
    call add_timer(T_ipsafunc,T_ipsa,'AIPS function')
    call add_timer(T_ipsagrid,T_ipsa,'AIPS grid')
    call add_timer(T_ipsasum,T_ipsa,'AIPS sum')
    call add_timer(T_ipsaforce,T_ipsa,'AIPS force')
    call add_timer(T_ipsafft,T_ipsa,'AIPS FFT')
    call add_timer(T_inte,T_energy,'INTRNL energy')
    call add_timer(T_bond,T_inte,'Bond energy')
    call add_timer(T_angle,T_inte,'Angle energy')
    call add_timer(T_ephi,T_inte,'Dihedral energy')
    call add_timer(T_restr,T_inte,'Restraints energy')
    call add_timer(T_ecomm,T_energy,'Comm energy')
    call add_timer(T_fcomm,T_energy,'Comm force') 
#if KEY_ENSEMBLE==1
    call add_timer(T_EVB, T_energy, 'EVB routine')
    call add_timer(T_EVB_Comms, T_EVB, 'EVB MPI Comms')
    call add_timer(T_EVB_Energy, T_EVB, 'EVB Energy')
    call add_timer(T_EVB_Forces, T_EVB, 'EVB Forces')
#endif
#if KEY_DOMDEC==1
    call add_timer(T_dlb,T_energy,'Dynamic load balance') 
#endif
    call add_timer(T_constp,T_dynamc,'Constant pressure')
    call add_timer(T_constt,T_dynamc,'Constant temperature')
    call add_timer(T_commvar,T_dynamc,'Comm variables')
    call add_timer(T_saveshake,T_dynamc,'Save shake coords')
    call add_timer(T_heur,T_dynamc,'Heuristic check')
    call add_timer(T_tvir,T_dynamc,'Temp and virial calc')
#if KEY_DOMDEC==1
    call add_timer(T_crdcomm2,T_dynamc,'Dir/Recip comm coords')
    call add_timer(T_fcomm2,T_energy,'Dir/Recip comm force')
    call add_timer(T_fcomm2wait,T_fcomm2,'Dir/Recip wait')
    ! Reciprocal node timers
!    call add_timer(T_energy,T_total,'Energy time')
    call add_timer(T_rec,T_energy,'Recip Ewald time')
    call add_timer(T_bspl,T_rec,'Fill Bspline coeffs')
    call add_timer(T_fillg,T_rec,'Fill charge grid')
    call add_timer(T_scsum,T_rec,'Scalar sum')
    call add_timer(T_grads,T_rec,'Grad sum')
    call add_timer(T_FFT,T_rec,'FFT')
    call add_timer(T_FFTcomm,T_FFT,'FFTcomm')
!    call add_timer(T_fcomm,T_energy,'Comm force') 
    ! NOTE: T_crdcomm3 is for recip nodes only and has a different parent from T_crdcomm2
    call add_timer(T_crdcomm3,T_energy,'Dir/Recip comm coords')
    call add_timer(T_d2r,T_crdcomm3,'Dir -> Recip')
!    call add_timer(T_d2r_probe,T_d2r,'Probe')
    call add_timer(T_d2r_recv_size,T_d2r,'Recv size')
    call add_timer(T_d2r_recv,T_d2r,'Recv')
    call add_timer(T_d2r_unpack,T_d2r,'Unpack')
    call add_timer(T_r2r,T_crdcomm3,'Recip <-> Recip')
#endif 
    call add_timer(T_omm, T_total, 'OpenMM overhead')
    call add_timer(T_omm_load, T_omm, 'load plugins')
    call add_timer(T_omm_sys, T_omm, 'build system')
    call add_timer(T_omm_ctx, T_omm, 'init context')

    call add_timer(T_blade, T_total, 'BLaDE overhead')

    !--- establish what level each timer is at
    do i = 1,10000
       call get_level(success)
       if ( success  ==  1 )goto 100
    enddo
100 continue

    !--- build the lists(s) for output
    !    should imitate a depth-first tree walk
    call  build_lists()
    return
  end  subroutine init_timers
  !----------------------------------------------
  subroutine add_timer(index,par_index,string)

  use stream
    integer index,par_index
    character(len=*) string
    integer i

#if KEY_TIMER_DEBUG==1
    if ( index  >  0 )then
       write(OUTU,*)'ADD_TIMER: already added ',string
       call mexit(OUTU,'ADD_TIMER: already added ')
    endif
#endif 

    T_numadd  = T_numadd + 1
#if KEY_TIMER_DEBUG==1
    if ( T_numadd  >  MAXTIME )then
       write(OUTU,*)'ADD_TIMER: too many times.', &
            'Check MAXTIME in header'
       call mexit(OUTU,'ADD_TIMER: too many times.')
    endif
#endif 

    index = T_numadd
#if KEY_TIMER_DEBUG==1
    if ( par_index  ==  0 )then
       write(OUTU,*)'ADD_TIMER: invalid parent_index for ', &
            string
       write(OUTU,*)'Need to add parent index first'
       call mexit(OUTU,'ADD_TIMER: invalid parent_index ')
    endif
#endif 

    T_string(index) = string
    Tpar_P(index) = par_index
    Tpar_level(index) = Tpar_level(par_index)+1
    if ( par_index  >  0 )then
       ! fortran linked list of sibs
       if ( Tchild_P(par_index)  ==  0 )then
          Tchild_P(par_index) = index
       else
          i = Tchild_P(par_index)
          do while( Tsib_P(i)  >  0 )
             i = Tsib_P(i)
          enddo
          Tsib_P(i) = index
       endif
    else
       !   --- This must be the overall total timer -----
       Tpar_level(index)=1
    endif
    return
  end  subroutine add_timer
  !---------------------------------------------------------
  subroutine timer_start(istart)

  use stream
    !     use Charmm's second subprogram, not an intrinsic.
    !      external second
    real(chm_real8) tim
    integer istart
    call second(tim)
    if (istart  >  0 .and. istart  <  MAXTIME)then
#if KEY_TIMER_DEBUG==1
       if ( T_state(istart)  /=  0 )then
          write(OUTU,*)'Invalid start to time (already started): ', &
               istart,T_string(istart)
          call mexit(OUTU,'Invalid start to time ')
       elseif(Tpar_level(istart)  /=  time_current_level+1)then
          if(Tpar_level(istart)  <=  time_current_level)then
             write(OUTU,*) &
                  'Invalid start to time (same or lower level)', &
                  ' than current running timer'
             write(outu,'("starting          ",i4,a,i4)') &
                  istart,T_string(istart),Tpar_level(istart)
             write(outu,'("whose parent is   ",i4,a,i4,i4)') &
                  Tpar_P(istart),T_string(Tpar_P(istart)), &
                  Tpar_level(Tpar_P(istart) )   
             write(outu,'("current level:    ",i4)') &
                  time_current_level
             call mexit(OUTU,'Invalid start to time ')
          elseif(Tpar_level(istart)  >  time_current_level+1)then
             write(OUTU,*) &
                  'Invalid start to time (parent not running)', &
                  istart,T_string(istart)
             call mexit(OUTU,'Invalid start to time ')
          endif
       endif
#endif 
       time_current_level=time_current_level+1
       T_curr(istart) = tim
       T_state(istart) = 1
#if KEY_DBGTIMER==1
       print *,"Starting ",T_string(istart),time_current_level  
#endif
    endif
    return
  end  subroutine timer_start
  !---------------------------------------------------------
  subroutine second(tim)
  use machutil,only:jobdat
    integer iobuf, iodir, npflts
    real(chm_real) cputim, elatim
    real(chm_real8) tim
    !  for useful communication times return elapsed time, not cpu time.
    !  a portable primitive routine that returns the wall clock time does
    !  not appear to exist.
    call jobdat(elatim,cputim,iobuf,iodir,npflts)
    tim = elatim
    return
  end  subroutine second

  subroutine seconds(etim,ctim)
  use machutil,only:jobdat
    integer iobuf, iodir, npflts
    real(chm_real) cputim, elatim
    real(chm_real8) etim,ctim
    !  returns both elapsed time and cpu time
    call jobdat(elatim,cputim,iobuf,iodir,npflts)
    etim = elatim
    ctim = cputim
    return
  end  subroutine seconds

  !---------------------------------------------------------
  subroutine timer_stop(istop)
  use stream

    !     use Charmm's second subprogram, not an intrinsic.
    !      external second
    real(chm_real8) tim
    integer istop
    call second(tim)

    if (istop  >  0 .and. istop  <  MAXTIME)then
#if KEY_TIMER_DEBUG==1
       if ( T_state(istop)  ==  0 )then
          write(OUTU,*)'Invalid stop to time (not on): ', &
               istop,T_string(istop)
          call mexit(OUTU,'Invalid stop to time (not on)')
          return
       elseif(Tpar_level(istop)  /=  time_current_level)then
          if(Tpar_level(istop)  <  time_current_level)then
             write(OUTU,*) &
                  'Invalid stop to time ( lower level)', &
                  ' than current running timer', &
                  istop,T_string(istop)
             return
          elseif(Tpar_level(istop)  >  time_current_level)then
             write(OUTU,*) &
                  'Invalid stop to time (timer not running)', &
                  istop,T_string(istop)
             return
          endif
       endif
#endif 

       T_accum(istop) = T_accum(istop) + tim - T_curr(istop)

       !       ---- reset, so can't stop again without start
       T_state(istop) = 0
#if KEY_DBGTIMER==1
       print *,"Stopping ",T_string(istop),time_current_level  
#endif
       time_current_level=time_current_level-1
    endif
    return
  end subroutine timer_stop
  !---------------------------------------------------------
  subroutine timer_stpstrt(istop,istart)

  use stream
    !     use Charmm's second subprogram, not an intrinsic.
    !      external second
    real(chm_real8) tim
    integer istop,istart

    call second(tim)

#if KEY_TIMER_DEBUG==1
    if(Tpar_level(istop)  /=  Tpar_level(istart) )then
       write(OUTU,'(a,/"------> ",2i4,a15,a15)') &
            'Invalid stop/start to time (different levels)', &
            istop,istart,T_string(istop)(1:15), &
            T_string(istart)(1:15)
       call mexit(OUTU,'Invalid timer stop/start')
    endif
    if(Tpar_level(istop)  /=  time_current_level)then
       write(OUTU,*) &
            'Invalid stop/start (timer not current level)', &
            istop,T_string(istop),istart,T_string(istart)
       call mexit(OUTU,'Invalid timer stop/start')
    endif
#endif 

    if (istop  >  0 .and. istop  <  MAXTIME)then
#if KEY_TIMER_DEBUG==1
       if ( T_state(istop)  ==  0 )then
          write(OUTU,*)'Invalid stop to time: ',T_string(istop)
          call mexit(OUTU,'Invalid stop to time')
       endif
#endif 
       T_accum(istop) = T_accum(istop) + tim - T_curr(istop)
       !      --- reset, so can't stop again without start
       T_state(istop) = 0
    endif
#if KEY_DBGTIMER==1
    print *,"Stopping ",T_string(istop),time_current_level  
#endif

    if (istart  >  0 .and. istart  <  MAXTIME)then
#if KEY_TIMER_DEBUG==1
       if ( T_state(istart)  /=  0 )then
          write(OUTU,*)'Invalid start to time: ',T_string(istart)
          call mexit(OUTU,'Invalid start to time')
       endif
#endif 
       T_curr(istart) = tim
       T_state(istart) = 1
    endif
#if KEY_DBGTIMER==1
    print *,"Starting ",T_string(istart),time_current_level  
#endif
    return
  end subroutine timer_stpstrt
  !---------------------------------------------------------
  subroutine finish_timers()

  use stream
    real(chm_real8) t
    integer i,j,k,max
    ! check if anyone has T_state = 1
    do i = 1,MAXTIME
       if ( T_state(i)  ==  1 )then
          write(OUTU,*)'WARNING: timer state left on: ',T_string(i)
       endif
    enddo
    ! get the max level
    max = 0
    do i = 1,MAXTIME
       if ( T_level(i)  >  max )max = T_level(i)
    enddo

    ! accumulate child times
    ! do from maxlevel up to level 2 (level 1 has no parent)
    do j = 1, max-1
       k = max - j + 1
       do i = 1,MAXTIME
          if ( T_level(i)  ==  k )then
             if ( Tpar_P(i)  >  0 .and. Tpar_P(i)  <=  MAXTIME)then
                t = T_accum(i)
                if ( Tch_acc(i)  >  t )t = Tch_acc(i)
                Tch_acc(Tpar_P(i)) = Tch_acc(Tpar_P(i)) + t
             endif
          endif
       enddo
    enddo

    return
  end  subroutine finish_timers
  !---------------------------------------------------------
  subroutine get_level(success)

    integer success
    integer i
    do i = 1,T_numadd
       ! only look at undetermined ones
       if ( T_level(i)  ==  0 )then
          if ( Tpar_P(i)  ==  -1 )then
             T_level(i) = 1
          elseif ( T_level(Tpar_P(i))  /=  0 )then
             T_level(i) = T_level(Tpar_P(i)) + 1
          endif
       endif
    enddo
    ! check if all defined
    success = 1
    do i = 1,T_numadd
       if ( T_level(i)  ==  0 )success = 0
    enddo
    return
  end subroutine get_level
  !---------------------------------------------------------
  subroutine build_lists()

  use stream
    integer i,j,numfirst,numlast
    do i = 1,T_numadd
       if ( Tsib_P(i)  >  0 )then
          if ( Tchild_P(Tsib_P(i))  >  0 )then
             ! follow the children to get Tnext_P(i)
             j = Tchild_P(Tsib_P(i))
5            continue
             if ( Tchild_P(j)  /=  0 )then
                j = Tchild_P(j)
                goto 5
             endif
             Tnext_P(i) = j
          else
             Tnext_P(i) = Tsib_P(i)
          endif
       elseif ( Tpar_P(i)  >  0 )then
          Tnext_P(i) = Tpar_P(i)
       endif
       if ( Tnext_P(i)  /=  0 )then
          Tprev_P(Tnext_P(i)) = i
       endif
    enddo
    T_numtree = 0
    do i = 1,T_numadd
       if ( Tnext_P(i)  ==  0 )then
          T_numtree = T_numtree + 1
#if KEY_TIMER_DEBUG==1
          if ( T_numtree  >  MAXTREE )then
             write(OUTU,*)'Too many timer trees. Check MAXTREE'
             call mexit(OUTU,'Too many timer trees.' )
          endif
#endif 
          T_last(T_numtree) = i
       endif
    enddo
    ! follow ends back to get start of lists
    do j = 1,T_numtree
       i = T_last(j)
10     continue
       if ( Tprev_P(i)  /=  0 )then
          i = Tprev_P(i)
          goto 10
       endif
       T_first(j) = i
    enddo
    return
  end subroutine build_lists
end module new_timer

